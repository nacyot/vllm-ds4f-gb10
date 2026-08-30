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
      - type: (required) Type of secondary tier (e.g., "example", "fs",
        "p2p", "obj"), or the class name of an out-of-tree
        SecondaryTierManager paired with module_path.
      - module_path: (optional) Python import path to load 'type' from
        when it names an out-of-tree SecondaryTierManager not registered
        via SecondaryTierFactory.register_tier()
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

Example out-of-tree tier configuration:
{
    "cpu_bytes_to_use": 10737418240,
    "secondary_tiers": [
        {
            "type": "MyCustomTier",
            "module_path": "my_package.my_module",
            "custom_param": "value"
        }
    ]
}
"""

from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
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


def _dspark_failed_chunks_to_gpu_blocks(
    src_spec, dst_spec, blocks_per_chunk, failed_cpu_bids
):
    """[load-error-recompute] Map failed staging-chunk block ids to the GPU
    block ids they would have filled.

    Mirrors the per-group offset walk in
    SingleDirectionOffloadingHandler.transfer_async for loads
    (gpu_to_cpu=False: src_blocks_per_chunk=blocks_per_chunk,
    dst_blocks_per_chunk=1). Per group, the first chunk may be partially
    consumed (skip = block_index % blocks_per_chunk).

    Falls back to reporting the whole job's dst blocks if the walk does not
    line up; over-reporting is safe (the scheduler merely recomputes more).
    """
    dst_blocks = dst_spec.block_ids
    try:
        failed_pos = {
            i for i, b in enumerate(src_spec.block_ids) if int(b) in failed_cpu_bids
        }
        failed_gpu: set = set()
        src_off = 0
        dst_off = 0
        for group_size, block_idx in zip(dst_spec.group_sizes, dst_spec.block_indices):
            gsz = int(group_size)
            if gsz == 0:
                continue
            skip = int(block_idx) % blocks_per_chunk
            src_count = -(-(gsz + skip) // blocks_per_chunk)
            for j in range(src_count):
                if (src_off + j) in failed_pos:
                    lo = max(0, j * blocks_per_chunk - skip)
                    hi = min(gsz, (j + 1) * blocks_per_chunk - skip)
                    failed_gpu.update(
                        int(b) for b in dst_blocks[dst_off + lo : dst_off + hi]
                    )
            src_off += src_count
            dst_off += gsz
        if src_off != len(src_spec.block_ids) or dst_off != len(dst_blocks):
            raise ValueError(
                f"offset walk mismatch: src {src_off}/{len(src_spec.block_ids)} "
                f"dst {dst_off}/{len(dst_blocks)}"
            )
        return failed_gpu
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[load-error-recompute] chunk->gpu mapping failed (%s); "
            "reporting the whole job's dst blocks",
            exc,
        )
        return {int(b) for b in dst_blocks}


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
                    "Histogram of blocking time spent in a per-block tier lookup "
                    "that resolved as a hit or miss, labeled by tier, in seconds."
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
                    "Histogram of wall-clock time from a per-block tier lookup "
                    "first returning retry until that same tier lookup resolves "
                    "as a hit or miss, labeled by tier, in seconds."
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
        metrics[TieringOffloadingMetrics.READ_BYTES] = OffloadingCounterMetadata(
            documentation=(
                "Total bytes read from secondary tiers into the primary tier, "
                "labeled by tier."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.READ_TIME] = OffloadingCounterMetadata(
            documentation=(
                "Total time spent reading from secondary tiers into the primary "
                "tier, in seconds, labeled by tier."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.WRITE_BYTES] = OffloadingCounterMetadata(
            documentation=(
                "Total bytes written from the primary tier to secondary tiers, "
                "labeled by tier."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.WRITE_TIME] = OffloadingCounterMetadata(
            documentation=(
                "Total time spent writing from the primary tier to secondary "
                "tiers, in seconds, labeled by tier."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.PROMOTION_JOB_FAILURES] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Number of failed secondary-tier promotion jobs, labeled by tier."
                ),
            )
        )
        metrics[TieringOffloadingMetrics.CASCADE_JOB_FAILURES] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Number of failed secondary-tier cascade jobs, labeled by tier."
                ),
            )
        )
        metrics[TieringOffloadingMetrics.BLOCK_QUERIES] = OffloadingCounterMetadata(
            documentation=(
                "Number of block lookup queries sent to a tier, labeled by tier."
            ),
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.BLOCK_HITS] = OffloadingCounterMetadata(
            documentation="Number of block lookup hits in a tier, labeled by tier.",
            labelnames=("tier",),
        )
        metrics[TieringOffloadingMetrics.PROMOTION_ALLOCATION_FAILURES] = (
            OffloadingCounterMetadata(
                documentation=(
                    "Number of promotion attempts that failed because the "
                    "primary tier could not allocate space."
                ),
            )
        )
        metrics[TieringOffloadingMetrics.PRIMARY_WRITE_USAGE_PERC] = (
            OffloadingGaugeMetadata(
                documentation=(
                    "Current fraction of primary-tier space used by writes from "
                    "secondary tiers, labeled by tier."
                ),
            )
        )
        metrics[TieringOffloadingMetrics.PRIMARY_READ_USAGE_PERC] = (
            OffloadingGaugeMetadata(
                documentation=(
                    "Current fraction of primary-tier space used by reads to "
                    "secondary tiers, labeled by tier."
                ),
            )
        )
        metrics[TieringOffloadingMetrics.ACTIVE_PROMOTION_JOBS] = (
            OffloadingGaugeMetadata(
                documentation=(
                    "Number of active secondary-tier promotion jobs, labeled by tier."
                ),
            )
        )
        metrics[TieringOffloadingMetrics.ACTIVE_CASCADE_JOBS] = OffloadingGaugeMetadata(
            documentation=(
                "Number of active secondary-tier cascade jobs, labeled by tier."
            ),
            labelnames=("tier",),
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
                        "Created secondary tier #%d (%s)",
                        i,
                        tier.tier_type,
                    )

                # Create TieringOffloadingManager. GPU↔CPU transfers use the inherited
                # get_worker(). Secondary tier transfers are handled by the
                # secondary tier managers and need no additional workers here.
                for _tier in secondary_tiers:
                    if hasattr(_tier, "file_mapper"):
                        _tier._layout_path = (
                            f"/dev/shm/vllm_offload_{self._engine_id}.layout.json"
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
            # DSPARK compact 멀티노드: 승격 청크는 스케줄러 노드 mmap에만
            # 존재한다. 로드 스펙에 fs 경로가 오면 (공유 마운트 절대경로)
            # 이 워커의 mmap 행을 파일로 먼저 채운 뒤 정상 로드를 진행한다.
            _mv = worker_mmap.create_kv_memoryview().cast("B")
            _row = self.kv_bytes_per_chunk
            _orig_submit_load = _worker.submit_load

            # canonical 그룹 슬라이스 사이드카: 텐서 뷰 오프셋에서 그룹별
            # (행내 오프셋, 길이)를 산출해 기록. fs tier가 참바이트 파일을
            # 쓰고, 워커 pread가 같은 슬라이스를 채우는 기준이 된다.
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
                logger.info("DSPARK layout sidecar written: %s %s", _sc, _slices)
            except Exception as _e:  # noqa: BLE001
                logger.warning("DSPARK layout sidecar failed: %s", _e)

            import os as _skip_os

            _pread_skip = _skip_os.environ.get("DSPARK_FS_PREAD_SKIP") == "1"

            def _fs_submit_load(job_id, src_spec, dst_spec):
                entries = getattr(src_spec, "fs_paths", None)
                # [load-error-recompute] staging chunk block ids whose fs
                # pread failed in this job (zero-filled below); mapped to
                # GPU blocks and reported before submitting the job.
                _failed_cpu_bids = set()
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
                            _win_bytes = int(
                                _relay_os.environ.get(
                                    "DSPARK_RELAY_WINDOW_BYTES", "268435456"
                                )
                            )
                        except (ValueError, TypeError):
                            _win_bytes = 268435456
                        # DSPARK_RELAY_DIRECT (default 1): source the
                        # broadcast window straight from the staging mmap
                        # (_flat) instead of re-reading the chunk files.
                        # Contract (TieringOffloadingManager.prepare_load +
                        # _process_finished_jobs): fs_paths are attached only
                        # for keys in _fs_promoted_keys, i.e. keys whose
                        # fs->primary promotion already completed into the
                        # scheduler-side staging mmap, and prepare_load only
                        # accepts primary-tier HITs and ref-pins them for the
                        # duration of the load. The scheduler mmap and this
                        # rank-0 worker mmap are the same file
                        # (vllm_offload_{engine_id}.mmap has no rank suffix),
                        # so the bytes are already in _flat and the per-chunk
                        # pread was a second NVMe read of the same bytes.
                        # Set DSPARK_RELAY_DIRECT=0 for the legacy preads.
                        _relay_direct = (
                            _relay_os.environ.get("DSPARK_RELAY_DIRECT", "1") != "0"
                        )
                        _wi = 0
                        while _wi < len(_items):
                            _wb = 0
                            _wj = _wi
                            while _wj < len(_items) and (
                                _wj == _wi or _wb + _items[_wj][2] <= _win_bytes
                            ):
                                _wb += _items[_wj][2]
                                _wj += 1
                            _window = _items[_wi:_wj]
                            if _is_src and not _relay_direct:
                                # Legacy escape hatch only: refresh staging
                                # from the chunk files before broadcasting.
                                # On any read failure keep the
                                # staging-resident bytes (per the
                                # prepare_load contract they are the correct
                                # bytes); never partially overwrite or zero
                                # them here.
                                for _off, _path, _ln in _window:
                                    try:
                                        with open(_path, "rb") as _f:
                                            _data = _f.read()
                                    except OSError as _e2:
                                        logger.error(
                                            "[relay-read-fail] relay src "
                                            "read failed for %s: %s "
                                            "(broadcasting staging bytes)",
                                            _path,
                                            _e2,
                                        )
                                        continue
                                    if len(_data) != _ln:
                                        logger.error(
                                            "[relay-read-fail] relay src "
                                            "short read for %s: %d != %d "
                                            "(broadcasting staging bytes)",
                                            _path,
                                            len(_data),
                                            _ln,
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
                                        _flat[_off : _off + _ln]
                                    )
                                    _pos += _ln
                            _tp.broadcast(_buf, src=0)
                            if not _is_src:
                                _pos = 0
                                for _off, _p, _ln in _window:
                                    _flat[_off : _off + _ln].copy_(
                                        _buf[_pos : _pos + _ln]
                                    )
                                    _pos += _ln
                            _t.cuda.synchronize()
                            del _buf
                            _wi = _wj
                        _total = sum(_ln for _o, _p, _ln in _items)
                        logger.debug(
                            "relay restore: src=%s chunks=%d bytes=%d",
                            _is_src,
                            len(_items),
                            _total,
                        )
                        return _orig_submit_load(job_id, src_spec, dst_spec)
                logger.info(
                    "DSPARK wk_load job=%s spec=%s fs=%s",
                    job_id,
                    type(src_spec).__name__,
                    len(entries) if entries else None,
                )
                if entries and _pread_skip:
                    # head node: the scheduler already filled this mmap during
                    # promotion; the worker-direct pread would be a duplicate.
                    logger.info(
                        "DSPARK fs-direct preads skipped (head): %d",
                        sum(1 for _e in entries if _e),
                    )
                    entries = None
                _pt = 1
                try:
                    _pt = int(_skip_os.environ.get("DSPARK_FS_PREAD_THREADS", "1") or 1)
                except ValueError:
                    _pt = 1
                if entries and _pt > 1:
                    # DSPARK S2a/b: parallel reads (GIL released) in bounded
                    # windows, GPU-routed writes batched, one synchronize per
                    # window. Memory bound = window * chunk slice (~16 * 16MB).
                    from concurrent.futures import (
                        ThreadPoolExecutor as _ThreadPoolExecutor,
                    )

                    import torch as _t

                    try:
                        _win = int(
                            _skip_os.environ.get("DSPARK_FS_PREAD_WINDOW", "16") or 16
                        )
                    except ValueError:
                        _win = 16
                    _ex = getattr(_worker, "_dspark_pread_ex", None)
                    if _ex is None:
                        _ex = _worker._dspark_pread_ex = _ThreadPoolExecutor(
                            max_workers=_pt, thread_name_prefix="fs_pread"
                        )
                    _items = [
                        (int(_bid) * _row + int(_ent[1]), _ent[0], int(_ent[2]))
                        for _bid, _ent in zip(src_spec.block_ids, entries)
                        if _ent
                    ]

                    def _read_one(_it):
                        try:
                            with open(_it[1], "rb") as _f:
                                return _f.read()
                        except OSError as _e:
                            logger.warning(
                                "fs direct read failed for %s: %s", _it[1], _e
                            )
                            return None

                    _flat = _t.frombuffer(
                        memoryview(worker_mmap.mmap_obj), dtype=_t.uint8
                    )
                    _n = 0
                    for _i in range(0, len(_items), _win):
                        _chunk = _items[_i : _i + _win]
                        _datas = list(_ex.map(_read_one, _chunk))
                        _keep = []
                        for (_off, _p, _ln), _data in zip(_chunk, _datas):
                            if _data is None or len(_data) != _ln:
                                # This pread is the only source for this row
                                # slice on this node, and the offloading
                                # connector has no per-job failure channel
                                # (submit_load's result is asserted, and
                                # OffloadingConnector never reports
                                # get_block_ids_with_load_errors), so poison
                                # the range deterministically instead of
                                # silently serving stale bytes as restored
                                # KV.
                                _why = (
                                    "failed"
                                    if _data is None
                                    else f"short ({len(_data)} bytes)"
                                )
                                logger.error(
                                    "[relay-read-fail] fs pread %s for %s "
                                    "(want %d bytes); zero-filling range",
                                    _why,
                                    _p,
                                    _ln,
                                )
                                _z = _t.zeros(_ln, dtype=_t.uint8, device="cuda")
                                _flat[_off : _off + _ln].copy_(_z)
                                _keep.append(_z)
                                # [load-error-recompute] slice offsets are
                                # row-internal (< _row), so the row index
                                # recovers the staging chunk block id.
                                _failed_cpu_bids.add(_off // _row)
                                continue
                            _src_gpu = _t.frombuffer(
                                bytearray(_data), dtype=_t.uint8
                            ).cuda()
                            _flat[_off : _off + len(_data)].copy_(_src_gpu)
                            _keep.append(_src_gpu)
                            _n += 1
                        _t.cuda.synchronize()
                        del _keep, _datas
                    logger.info(
                        "DSPARK fs-direct preads (parallel x%d, win %d): %d",
                        _pt,
                        _win,
                        _n,
                    )
                    entries = None
                if entries:
                    _n = 0
                    import torch as _t

                    _flat = _t.frombuffer(
                        memoryview(worker_mmap.mmap_obj), dtype=_t.uint8
                    )
                    for _bid, _ent in zip(src_spec.block_ids, entries):
                        if not _ent:
                            continue
                        _path, _soff, _ln = _ent
                        _off = int(_bid) * _row + int(_soff)
                        _data = None
                        try:
                            with open(_path, "rb") as _f:
                                _data = _f.read()
                        except OSError as _e:
                            logger.error(
                                "[relay-read-fail] fs pread failed for "
                                "%s: %s; zero-filling range",
                                _path,
                                _e,
                            )
                        if _data is not None and len(_data) != _ln:
                            logger.error(
                                "[relay-read-fail] fs pread short read for "
                                "%s: %d != %d; zero-filling range",
                                _path,
                                len(_data),
                                _ln,
                            )
                            _data = None
                        if _data is None:
                            # No per-job failure channel exists here
                            # (submit_load's result is asserted, and
                            # OffloadingConnector never reports
                            # get_block_ids_with_load_errors), so poison the
                            # range deterministically instead of silently
                            # serving stale bytes as restored KV.
                            _z = _t.zeros(_ln, dtype=_t.uint8, device="cuda")
                            _flat[_off : _off + _ln].copy_(_z)
                            _t.cuda.synchronize()
                            # [load-error-recompute]
                            _failed_cpu_bids.add(int(_bid))
                            continue
                        # GB10 Xid 13 우회: CPU가 더럽힌 pinned 페이지를
                        # 디바이스가 읽으면 채널 오류 정황 — 파일 바이트를
                        # GPU에 올렸다가 디바이스 DMA로 mmap에 쓰게 한다.
                        _src_gpu = _t.frombuffer(
                            bytearray(_data), dtype=_t.uint8
                        ).cuda()
                        _flat[_off : _off + len(_data)].copy_(_src_gpu)
                        _t.cuda.synchronize()
                        _n += 1
                    logger.info("DSPARK fs-direct preads: %d", _n)
                if _failed_cpu_bids:
                    # [load-error-recompute] final defense line: report the
                    # zero-filled ranges as failed GPU blocks so the core
                    # scheduler (kv_load_failure_policy=recompute) truncates
                    # the affected requests at the first bad block and
                    # recomputes, instead of serving zero-filled KV as
                    # restored context. Drained once per step by
                    # OffloadingConnectorWorker.get_block_ids_with_load_errors.
                    _gpu_failed = _dspark_failed_chunks_to_gpu_blocks(
                        src_spec, dst_spec, self.blocks_per_chunk, _failed_cpu_bids
                    )
                    _errs = getattr(_worker, "_dspark_load_error_block_ids", None)
                    if _errs is None:
                        _errs = set()
                        _worker._dspark_load_error_block_ids = _errs
                    _errs.update(_gpu_failed)
                    logger.error(
                        "[load-error-recompute] job=%s failed_chunks=%d -> "
                        "gpu_blocks=%d reported for recompute",
                        job_id,
                        len(_failed_cpu_bids),
                        len(_gpu_failed),
                    )
                return _orig_submit_load(job_id, src_spec, dst_spec)

            _worker.submit_load = _fs_submit_load
            return _worker
        except Exception:
            import traceback

            print("DSPARK_COMPACT_DEBUG create_worker exception:", flush=True)
            traceback.print_exc()
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
