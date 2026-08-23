# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingSpec: Spec for multi-tier KV cache offloading.

This spec creates a TieringOffloadingManager with a CPU primary tier
and configurable secondary tiers (e.g., Storage, Network).

Configuration via kv_connector_extra_config:
  - cpu_bytes_to_use: (required) Bytes to allocate for CPU primary tier
  - block_size: (optional) Block size for offloaded blocks (default: GPU block size)
  - eviction_policy: (optional) Primary tier eviction policy: built-in "lru"/
    "arc", or the name of a policy registered via CachePolicyFactory, or an
    out-of-tree CachePolicy class name paired with cache_policy_module_path
    (default: "lru")
  - cache_policy_module_path: (optional) Python import path to load
    eviction_policy from when it names an out-of-tree CachePolicy not
    registered via CachePolicyFactory
  - secondary_tiers: (optional) List of secondary tier configurations
    Each secondary tier config is a dict with:
      - type: (required) Type of secondary tier (e.g., "example", "storage", "network")
      - Additional tier-specific parameters are passed directly to the tier
        constructor. See each tier's documentation for supported parameters.

Example configuration:
{
    "cpu_bytes_to_use": 10737418240,  # 10 GB
    "block_size": 16,
    "eviction_policy": "lru",
    "secondary_tiers": [
        {
            "type": "example",
            "custom_param": 67
        }
    ]
}
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.base import TieringOffloadingMetrics
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

logger = init_logger(__name__)


def _env_int(name: str, default: int = 1) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


class TieringOffloadingSpec(CPUOffloadingSpec):
    """
    Spec for multi-tier KV cache offloading.

    Creates a TieringOffloadingManager with:
    - Primary tier: CPU (LRU or ARC eviction policy)
    - Secondary tiers: Configurable via extra_config

    The CPU primary tier has direct GPU access and serves as the gateway for
    all GPU↔offload operations. Secondary tiers cannot directly access GPU
    memory and must transfer data through the primary tier.
    """

    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        metrics = super().build_metric_definitions(extra_config)
        metrics[TieringOffloadingMetrics.LOOKUP_SYNC_DELAY] = (
            OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of total blocking time spent querying secondary "
                    "tiers for a request, accumulated from first lookup until "
                    "the request is allocated or finishes, in seconds."
                ),
                buckets=(
                    0.00001,
                    0.00005,
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
                ),
            )
        )
        metrics[TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY] = (
            OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of wall-clock time from a request's first deferred "
                    "secondary-tier lookup until the request is allocated or "
                    "finishes, in seconds."
                ),
                buckets=(
                    0.0001,
                    0.0005,
                    0.001,
                    0.005,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1,
                    5,
                    10,
                ),
            )
        )
        secondary_tier_configs = extra_config.get("secondary_tiers", [])
        if not isinstance(secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        for tier_config in secondary_tier_configs:
            assert isinstance(tier_config, dict)
            tier_cls = SecondaryTierFactory.get_tier_class(tier_config)
            metrics.update(tier_cls.build_metric_definitions(tier_config))
        return metrics

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        # Redeclare for mypy: parent sets this but `--follow-imports skip` hides it
        self._manager: OffloadingManager | None = None

        # Parse secondary tier configurations
        self.secondary_tier_configs = self.extra_config.get("secondary_tiers", [])
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        # Scheduler-side mmap (rank=None); kept for cleanup
        self._scheduler_mmap: SharedOffloadRegion | None = None

        # Set by create_worker when canonical_layout is enabled: True when
        # every layer's canonical bytes are parallelism-agnostic (portable),
        # False when some layers use the opaque fallback (exact-topology only)
        self.all_layers_portable: bool | None = None

        # engine_id is unique per DP replica (suffixed with _dp{rank} in both
        # the Ray and multiprocessing paths), so it names a per-replica offload
        # region.
        self._engine_id = config.engine_id

    @override
    def get_manager(self) -> OffloadingManager:
        """
        Get the TieringOffloadingManager.

        Creates a TieringOffloadingManager with:
        - Primary tier: CPU (LRU or ARC)
        - Secondary tiers: As configured in extra_config

        Returns:
            TieringOffloadingManager instance
        """
        if not self._manager:
            if int(self.extra_config.get("store_threshold", 0)) >= 2:
                raise ValueError(
                    "store_threshold is not supported for TieringOffloadingSpec"
                )

            scheduler_mmap: SharedOffloadRegion | None = None
            primary_tier: CPUPrimaryTierOffloadingManager | None = None
            secondary_tiers = []
            try:
                # Create scheduler-side SharedOffloadRegion (rank=None) so the
                # primary tier can eagerly create a memoryview over _base.
                scheduler_mmap = SharedOffloadRegion(
                    engine_id=self._engine_id,
                    num_blocks=self.num_blocks,
                    rank=None,
                    kv_bytes_per_block=self.kv_bytes_per_chunk,
                    cpu_page_size=self.cpu_page_size_per_worker,
                )
                self._scheduler_mmap = scheduler_mmap

                # Create primary tier (CPU-based)
                primary_tier = CPUPrimaryTierOffloadingManager(
                    num_blocks=self.num_blocks,
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    mmap_region=scheduler_mmap,
                )

                # Create secondary tiers
                primary_kv_view = primary_tier.get_kv_memoryview()
                for i, tier_config in enumerate(self.secondary_tier_configs):
                    tier = SecondaryTierFactory.create_secondary_tier(
                        tier_config, primary_kv_view, self
                    )
                    secondary_tiers.append(tier)
                    logger.info(
                        "Created secondary tier from config index %i, type %s",
                        i,
                        tier.tier_type,
                    )

                # Create TieringOffloadingManager. GPU-CPU transfers use the
                # inherited get_worker(). Secondary tier transfers are handled
                # by the secondary tier managers and need no additional
                # workers here.
                # Point fs tiers at the worker-written layout sidecar so they
                # can store/load group slices instead of whole staging rows.
                if self.config.canonical_layout:
                    for _tier in secondary_tiers:
                        if hasattr(_tier, "file_mapper"):
                            _tier._layout_path = (
                                f"/dev/shm/vllm_offload_{self._engine_id}"
                                ".layout.json"
                            )
                tiering_manager = TieringOffloadingManager(
                    primary_tier=primary_tier,
                    secondary_tiers=secondary_tiers,
                )
                self._manager = tiering_manager
            except Exception:
                for tier in reversed(secondary_tiers):
                    try:
                        tier.shutdown()
                    except Exception:
                        logger.exception(
                            "Failed to shut down secondary tier during "
                            "initialization cleanup"
                        )
                if primary_tier is not None:
                    try:
                        primary_tier.shutdown()
                    except Exception:
                        logger.exception(
                            "Failed to shut down primary tier during "
                            "initialization cleanup"
                        )
                elif scheduler_mmap is not None:
                    try:
                        scheduler_mmap.cleanup()
                    except Exception:
                        logger.exception(
                            "Failed to clean up scheduler mmap during "
                            "initialization cleanup"
                        )
                self._scheduler_mmap = None
                raise

            logger.info(
                "Created TieringOffloadingManager with primary tier "
                "(%s, %s blocks) and %s secondary tier(s)",
                self.eviction_policy,
                self.num_blocks,
                len(secondary_tiers),
            )

        return self._manager

    @override
    def _uses_shared_region(self) -> bool:
        # Tiering always allocates on the shared region (every platform), so the
        # replicated-layout gate must not be narrowed by the CPU spec's
        # CUDA-alike check.
        return True

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        world_size = self.config.parallel.world_size
        if self.replicated_layout:
            rank = 0
        else:
            # Fold the global physical device index into the replica-local
            # [0, world_size) slot range.
            rank = torch.accelerator.current_device_index() % world_size
        worker_mmap = SharedOffloadRegion(
            engine_id=self._engine_id,
            num_blocks=self.num_blocks,
            rank=rank,
            kv_bytes_per_block=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        try:
            if self.config.canonical_layout:
                self._validate_canonical_refs(kv_caches)
            _worker = CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_blocks=self.num_blocks,
                mmap_region=worker_mmap,
                canonical_layout=self.config.canonical_layout,
            )
            # Compact packed multi-node: promoted chunks exist only in the
            # scheduler node's mmap. When a load spec carries fs paths
            # (absolute paths on a shared mount), fill this worker's mmap
            # rows from the files first, then run the normal load.
            _row = self.kv_bytes_per_chunk
            _orig_submit_load = _worker.submit_load
            # DSPARK_FS_PREAD_SKIP: on the node that ran the promotion (the
            # scheduler node in the current single-writer scheme), this
            # worker's mmap was already filled during promotion, so the
            # direct pread below would just duplicate that write.
            _pread_skip = os.environ.get("DSPARK_FS_PREAD_SKIP") == "1"

            def _fs_submit_load(job_id, src_spec, dst_spec):
                entries = getattr(src_spec, "fs_paths", None)
                # DSPARK_RELAY_RESTORE (experimental, opt-in, default off):
                # one rank reads the canonical files, then broadcasts the loaded
                # bytes to the other TP ranks over the process group interconnect.
                # Removes the shared-storage requirement and moves the cross-node
                # copy onto the fast link. Replicated KV -> one read suffices.
                import os as _relay_os
                if entries and _relay_os.environ.get("DSPARK_RELAY_RESTORE") == "1":
                    import torch as _t
                    from vllm.distributed.parallel_state import get_tp_group
                    _tp = get_tp_group()
                    if _tp.world_size > 1:
                        _is_src = _tp.rank_in_group == 0
                        _items = [
                            (int(_bid) * _row + int(_e[1]), _e[0], int(_e[2]))
                            for _bid, _e in zip(src_spec.block_ids, entries)
                            if _e
                        ]
                        _flat = _t.frombuffer(
                            memoryview(worker_mmap.mmap_obj), dtype=_t.uint8
                        )
                        _win_bytes = 268435456
                        try:
                            _win_bytes = int(_relay_os.environ.get(
                                "DSPARK_RELAY_WINDOW_BYTES", "268435456"))
                        except (ValueError, TypeError):
                            _win_bytes = 268435456
                        _wi = 0
                        while _wi < len(_items):
                            _wb = 0
                            _wj = _wi
                            while _wj < len(_items) and (
                                    _wj == _wi
                                    or _wb + _items[_wj][2] <= _win_bytes):
                                _wb += _items[_wj][2]
                                _wj += 1
                            _window = _items[_wi:_wj]
                            if _is_src:
                                for _off, _path, _ln in _window:
                                    try:
                                        with open(_path, "rb") as _f:
                                            _data = _f.read()
                                    except OSError as _e2:
                                        logger.warning(
                                            "relay src read failed for %s: %s",
                                            _path, _e2,
                                        )
                                        continue
                                    _g = _t.frombuffer(
                                        bytearray(_data), dtype=_t.uint8
                                    ).cuda()
                                    _flat[_off : _off + len(_data)].copy_(_g)
                                _t.cuda.synchronize()
                            _buf = _t.empty(_wb, dtype=_t.uint8, device="cuda")
                            if _is_src:
                                _pos = 0
                                for _off, _p, _ln in _window:
                                    _buf[_pos : _pos + _ln].copy_(
                                        _flat[_off : _off + _ln])
                                    _pos += _ln
                            _tp.broadcast(_buf, src=0)
                            if not _is_src:
                                _pos = 0
                                for _off, _p, _ln in _window:
                                    _flat[_off : _off + _ln].copy_(
                                        _buf[_pos : _pos + _ln])
                                    _pos += _ln
                            _t.cuda.synchronize()
                            del _buf
                            _wi = _wj
                        _total = sum(_ln for _o, _p, _ln in _items)
                        logger.debug(
                            "relay restore: src=%s chunks=%d bytes=%d",
                            _is_src, len(_items), _total,
                        )
                        return _orig_submit_load(job_id, src_spec, dst_spec)
                logger.debug(
                    "wk_load job=%s spec=%s fs=%s",
                    job_id,
                    type(src_spec).__name__,
                    len(entries) if entries else None,
                )
                if entries and _pread_skip:
                    logger.debug(
                        "fs-direct preads skipped (head node): %d",
                        sum(1 for _e in entries if _e),
                    )
                    entries = None

                n_pread_threads = _env_int("DSPARK_FS_PREAD_THREADS", 1)
                if entries and n_pread_threads > 1:
                    # Parallel windowed staging: read files on a small
                    # threadpool (file I/O releases the GIL), stage a
                    # bounded window of reads on the GPU, and synchronize
                    # once per window instead of once per block. Peak extra
                    # memory is window * per-block slice size, not the full
                    # restore.
                    import torch as _t

                    window = _env_int("DSPARK_FS_PREAD_WINDOW", 16)
                    pool = getattr(_worker, "_dspark_pread_pool", None)
                    if pool is None:
                        pool = _worker._dspark_pread_pool = ThreadPoolExecutor(
                            max_workers=n_pread_threads,
                            thread_name_prefix="fs_pread",
                        )
                    items = [
                        (int(_bid) * _row + int(_ent[1]), _ent[0])
                        for _bid, _ent in zip(src_spec.block_ids, entries)
                        if _ent
                    ]

                    def _read_one(item):
                        _off, _path = item
                        try:
                            with open(_path, "rb") as _f:
                                return _f.read()
                        except OSError as _e:
                            logger.warning(
                                "fs direct read failed for %s: %s", _path, _e
                            )
                            return None

                    _flat = _t.frombuffer(
                        memoryview(worker_mmap.mmap_obj), dtype=_t.uint8
                    )
                    _n_read = 0
                    for _i in range(0, len(items), window):
                        _chunk = items[_i : _i + window]
                        _datas = list(pool.map(_read_one, _chunk))
                        _staged = []
                        for (_off, _path), _data in zip(_chunk, _datas):
                            if _data is None:
                                continue
                            # Route the write through the GPU: letting the
                            # CPU dirty pinned pages that the device then
                            # reads has produced Xid 13 channel errors on
                            # GB10, so upload the file bytes and let device
                            # DMA write them into the pinned mmap.
                            _src_gpu = _t.frombuffer(
                                bytearray(_data), dtype=_t.uint8
                            ).cuda()
                            _flat[_off : _off + len(_data)].copy_(_src_gpu)
                            _staged.append(_src_gpu)
                            _n_read += 1
                        _t.cuda.synchronize()
                        del _staged, _datas
                    logger.debug(
                        "fs-direct preads (parallel x%d, window %d): %d",
                        n_pread_threads,
                        window,
                        _n_read,
                    )
                    entries = None

                if entries:
                    _n_read = 0
                    for _bid, _ent in zip(src_spec.block_ids, entries):
                        if not _ent:
                            continue
                        _path, _soff, _ln = _ent
                        try:
                            with open(_path, "rb") as _f:
                                _data = _f.read()
                            _off = int(_bid) * _row + int(_soff)
                            # Route the write through the GPU: letting the
                            # CPU dirty pinned pages that the device then
                            # reads has produced Xid 13 channel errors on
                            # GB10, so upload the file bytes and let device
                            # DMA write them into the pinned mmap.
                            import torch as _t
                            _src_gpu = _t.frombuffer(
                                bytearray(_data), dtype=_t.uint8
                            ).cuda()
                            _flat = _t.frombuffer(
                                memoryview(worker_mmap.mmap_obj), dtype=_t.uint8
                            )
                            _flat[_off : _off + len(_data)].copy_(_src_gpu)
                            _t.cuda.synchronize()
                            _n_read += 1
                        except OSError as _e:
                            logger.warning(
                                "fs direct read failed for %s: %s", _path, _e
                            )
                    logger.debug("fs-direct preads: %d", _n_read)
                return _orig_submit_load(job_id, src_spec, dst_spec)

            _worker.submit_load = _fs_submit_load

            if self.config.canonical_layout:
                # Canonical group-slice sidecar: derive each group's
                # (row offset, length) from the canonical tensor views and
                # persist it. The fs tier stores/loads exactly these
                # true-byte slices, and the worker-direct pread path fills
                # the same slices.
                try:
                    _views = worker_mmap._views
                    _slices = []
                    for _grefs in kv_caches.group_data_refs:
                        _idxs = sorted({r.tensor_idx for r in _grefs})
                        _first = _views[_idxs[0]]
                        _last = _views[_idxs[-1]]
                        _start = int(_first.storage_offset())
                        _end = int(_last.storage_offset()) + int(_last.shape[1])
                        _slices.append([_start, _end - _start])
                    import json as _json
                    import os as _os

                    _sc = f"/dev/shm/vllm_offload_{self._engine_id}.layout.json"
                    _tmp = _sc + ".tmp"
                    with open(_tmp, "w") as _f:
                        _json.dump({"group_slices": _slices}, _f)
                    _os.replace(_tmp, _sc)
                    logger.info(
                        "KV offload layout sidecar written: %s %s", _sc, _slices
                    )
                except Exception as _e:  # noqa: BLE001
                    logger.warning(
                        "KV offload layout sidecar write failed: %s", _e
                    )
            return _worker
        except Exception:
            worker_mmap.cleanup()
            raise

    def _validate_canonical_refs(self, kv_caches: CanonicalKVCaches) -> None:
        """Require a mapping on every ref and record layer portability.

        Fails loudly rather than persist direct-layout bytes under a
        canonical format identity."""
        all_refs = [
            ref for group_refs in kv_caches.group_data_refs for ref in group_refs
        ]
        if any(ref.mapping is None for ref in all_refs):
            raise RuntimeError(
                "canonical_layout was requested but the KV cache layout "
                "could not be certified for canonical offload (offload "
                "workers must be exactly the TP group, and packed / "
                "cross-layer KV layouts are not supported). Remove "
                "canonical_layout from kv_connector_extra_config."
            )
        self.all_layers_portable = all(
            ref.mapping is not None and ref.mapping.parallelism_agnostic
            for ref in all_refs
        )
        if self.config.parallel.is_parallelism_agnostic and not (
            self.all_layers_portable
        ):
            # The scheduler-side storage namespace was already collapsed on
            # the static portability claim; opaque per-topology bytes must
            # not land in it.
            raise RuntimeError(
                "canonical_layout could not certify every layer as "
                "parallelism-agnostic, but the storage namespace is shared "
                "across topologies. Remove canonical_layout from "
                "kv_connector_extra_config or disable parallel-agnostic "
                "secondary tiers."
            )
        logger.info(
            "Canonical KV layout enabled (all_layers_portable=%s)",
            self.all_layers_portable,
        )
