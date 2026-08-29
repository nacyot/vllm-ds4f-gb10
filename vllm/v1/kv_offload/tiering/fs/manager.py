# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

# DSPARK: per-event KV offload JSONL log (env-gated, no-op when unset)
import json as _ev_json
import os as _ev_os
import time as _ev_time
_EV_PATH = _ev_os.environ.get("DSPARK_KV_EVENT_LOG")

def _kv_event(rec):
    if not _EV_PATH:
        return
    try:
        rec["ts"] = round(_ev_time.time(), 3)
        with open(_EV_PATH, "a") as _f:
            _f.write(_ev_json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _env_int(name, default=1):
    try:
        v = int(_ev_os.environ.get(name, "") or default)
    except ValueError:
        v = default
    return v

_FS_LOAD_TASKS = _env_int("DSPARK_FS_LOAD_TASKS", 1)
_FS_STORE_TASKS = _env_int("DSPARK_FS_STORE_TASKS", 1)


def _keys_hex(keys):
    out = []
    for k in keys:
        try:
            out.append(bytes(k).hex())
        except Exception:  # noqa: BLE001
            out.append(str(k))
    return out


def _req_id(job_metadata):
    rc = getattr(job_metadata, "req_context", None)
    return getattr(rc, "req_id", None) if rc is not None else None


def _fanout_tasks(fn, paths, view, offs, sizes, use_o_direct, fan):
    """Split one batch into up to `fan` tasks (each a partial over a slice)."""
    import functools as _ft
    import math as _math
    n = len(paths)
    if fan <= 1 or n <= 1:
        return [_ft.partial(fn, paths, view, offs, sizes, use_o_direct)]
    per = max(1, _math.ceil(n / fan))
    tasks = []
    for i in range(0, n, per):
        sl = slice(i, i + per)
        sz = sizes[sl] if isinstance(sizes, list) else sizes
        tasks.append(_ft.partial(fn, paths[sl], view, offs[sl], sz, use_o_direct))
    return tasks


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        # DSPARK S3: split the stat batch across threads (C path releases the
        # GIL; CIFS metadata round trips are latency-bound, not CPU-bound).
        n_thr = _env_int("DSPARK_FS_LOOKUP_THREADS", 1)
        _t0 = _ev_time.time()
        if n_thr > 1 and len(paths) >= 2 * n_thr:
            import math as _math
            from concurrent.futures import ThreadPoolExecutor as _TPE
            ex = getattr(self, "_dspark_lookup_ex", None)
            if ex is None:
                ex = self._dspark_lookup_ex = _TPE(max_workers=n_thr,
                                                   thread_name_prefix="fs_lookup")
            per = _math.ceil(len(paths) / n_thr)
            fn = (lambda ps: list(batch_lookup_C(ps))) if _HAS_BATCH_LOOKUP_C \
                else (lambda ps: [os.path.exists(p) for p in ps])
            futs = [ex.submit(fn, paths[i:i + per]) for i in range(0, len(paths), per)]
            results = []
            for f in futs:
                results.extend(f.result())
            _kv_event({"op": "lookup_batch", "n": len(paths),
                       "hit": sum(1 for r in results if r), "threads": n_thr,
                       "dt": round(_ev_time.time() - _t0, 3)})
            return results
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            results = list(batch_lookup_C(paths))
        else:
            results = [os.path.exists(p) for p in paths]
        _kv_event({"op": "lookup_batch", "n": len(paths),
                   "hit": sum(1 for r in results if r), "threads": 1,
                   "dt": round(_ev_time.time() - _t0, 3)})
        return results


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]
        # DSPARK compact: 워커가 남긴 canonical 레이아웃 사이드카가 있으면
        # 그룹 슬라이스(참바이트)만 저장/로드한다. 없으면 행 통짜(레거시).
        self._group_slices: list | None = None
        self._layout_path: str | None = None

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def _maybe_load_layout(self) -> None:
        if self._group_slices is not None or not self._layout_path:
            return
        import json as _json
        import os as _os

        if _os.path.exists(self._layout_path):
            try:
                with open(self._layout_path) as _f:
                    self._group_slices = _json.load(_f)["group_slices"]
            except (OSError, ValueError, KeyError) as _e:
                self._group_slices = None
                # The sidecar file exists, so this deployment writes
                # canonical slice-layout chunk files. Whole-row fallback
                # files would share the same content-addressed namespace:
                # a slice-mode reader hitting a whole-row file loads
                # group-0 bytes into other groups, silently corrupting
                # restores across boots. Fail loudly by default;
                # DSPARK_SIDECAR_STRICT=0 restores the legacy fallback.
                if _env_int("DSPARK_SIDECAR_STRICT", 1):
                    raise RuntimeError(
                        f"KV offload layout sidecar {self._layout_path!r} "
                        f"exists but could not be read/parsed: {_e!r}. "
                        "Refusing whole-row fallback in a slice-layout "
                        "content-addressed namespace (set "
                        "DSPARK_SIDECAR_STRICT=0 to allow it)."
                    ) from _e
                if not getattr(self, "_layout_fallback_logged", False):
                    self._layout_fallback_logged = True
                    logger.error(
                        "KV offload layout sidecar %s could not be "
                        "read/parsed (%r); falling back to whole-row file "
                        "layout. Whole-row and slice-layout files share "
                        "one content-addressed namespace, so mixing them "
                        "silently corrupts restored KV data.",
                        self._layout_path,
                        _e,
                    )

    def _key_offsets_sizes(self, keys, block_ids):
        self._maybe_load_layout()
        if not self._group_slices:
            return (
                [int(bid) * self._block_size for bid in block_ids],
                self._block_size,
            )
        from vllm.v1.kv_offload.file_mapper import get_offload_group_idx

        offs = []
        sizes = []
        for key, bid in zip(keys, block_ids):
            g = get_offload_group_idx(key)
            s0, ln = self._group_slices[g]
            offs.append(int(bid) * self._block_size + s0)
            sizes.append(ln)
        return offs, sizes

    def direct_read_info(self, key, block_id):
        """(path, row_offset, nbytes) for worker-side direct fs read."""
        self._maybe_load_layout()
        path = self.file_mapper.get_file_name(key)
        if not self._group_slices:
            return (path, 0, self._block_size)
        from vllm.v1.kv_offload.file_mapper import get_offload_group_idx

        s0, ln = self._group_slices[get_offload_group_idx(key)]
        return (path, s0, ln)

    def submit_store(self, job_metadata: JobMetadata) -> None:
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        _keys = list(job_metadata.keys)
        _paths = [self.file_mapper.get_file_name(key) for key in _keys]
        _offs, _sizes = self._key_offsets_sizes(_keys, job_metadata.block_ids)
        tasks = _fanout_tasks(batch_store_block, _paths, self._primary_kv_view,
                              _offs, _sizes, self._use_o_direct, _FS_STORE_TASKS)
        _kv_event({"op": "store", "job": job_metadata.job_id,
                   "chunks": len(_keys), "tasks": len(tasks),
                   "req": _req_id(job_metadata),
                   "bytes": (sum(_sizes) if isinstance(_sizes, list) else _sizes * len(_keys)),
                   "keys": _keys_hex(_keys)})
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        _keys = list(job_metadata.keys)
        _paths = [self.file_mapper.get_file_name(key) for key in _keys]
        for _p in _paths:
            # A load counts as "use" for the external mtime-ordered pruner;
            # otherwise parked sessions age by first-write time and the
            # pruner evicts the hottest chunks first.
            try:
                os.utime(_p, None)
            except OSError:
                pass
        _offs, _sizes = self._key_offsets_sizes(_keys, job_metadata.block_ids)
        tasks = _fanout_tasks(batch_load_block, _paths, self._primary_kv_view,
                              _offs, _sizes, self._use_o_direct, _FS_LOAD_TASKS)
        _kv_event({"op": "load", "job": job_metadata.job_id,
                   "chunks": len(_keys), "tasks": len(tasks),
                   "req": _req_id(job_metadata),
                   "bytes": (sum(_sizes) if isinstance(_sizes, list) else _sizes * len(_keys)),
                   "keys": _keys_hex(_keys)})
        self._pool.enqueue_load(job_metadata.job_id, len(tasks), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            _kv_event({"op": "done", "job": job_id, "ok": bool(success)})
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def touch(self, keys, req_context) -> None:
        # DSPARK EVLOG v2: record the request's chunk chain once per request
        # (the connector calls touch with the request's keys; g0 = full-attention
        # group 0 chain identifies the session, window groups are derivable).
        if not _EV_PATH:
            return
        try:
            rid = getattr(req_context, "req_id", None)
            seen = getattr(self, "_ev_touched", None)
            if seen is None:
                seen = self._ev_touched = set()
            if rid in seen:
                return
            seen.add(rid)
            if len(seen) > 4096:
                seen.clear()
            g0 = [k for k in keys if bytes(k)[-4:] == b"\x00\x00\x00\x00"]
            _kv_event({"op": "touch", "req": rid, "n_keys": len(keys),
                       "g0": _keys_hex(g0)})
        except Exception:  # noqa: BLE001
            pass

    def dspark_invalidate_lookup(self, keys) -> None:
        """[load-error-recompute-p60] Overwrite cached async-lookup results
        for keys whose promotion load just failed.

        The short-read policy (fs_io_C _load_block) unlinks the backing
        file inside the failing I/O job, but the AsyncLookupManager cache
        entry was created while the file still passed the existence check
        and lives until every requesting request finishes. Without this
        override the stale HIT re-offers the dead chunk every scheduler
        step (silent retry livelock; E1 2026-08-28 03:15, 8487 resubmits).
        Runs on the scheduler thread, which owns _lookup_state (see the
        async_lookup.py locking design note); the single async stat result
        per cache entry has necessarily been drained before any promotion
        of that key could start, so no in-flight worker result can
        overwrite this back to True.
        """
        state_map = self._lookup_manager._lookup_state
        for key in keys:
            state = state_map.get(key)
            if state is not None:
                state.result = False

    def delete_keys(self, keys, req_context=None) -> tuple[int, int]:
        """DSPARK_TAIL_KEEP: unlink the chunk files of ``keys`` (best effort).

        Called on the scheduler thread after a store completes, with the
        chain's superseded window snapshots (a few hundred keys at most per
        call; most are already gone, and ENOENT is the cheap path).
        Lookups resolve keys by file existence, so no index needs updating
        except the per-request async lookup cache: a request that looked
        this key up earlier in its lifetime still holds a cached HIT, and
        re-offering a deleted file would end in a failed load + recompute
        (see dspark_invalidate_lookup). Files younger than a few seconds are
        never targeted by construction (they belong to the newest anchors).
        Returns (files_deleted, bytes_freed).
        """
        n = b = 0
        state_map = self._lookup_manager._lookup_state
        for key in keys:
            path = self.file_mapper.get_file_name(key)
            try:
                size = os.path.getsize(path)
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError as e:  # noqa: PERF203 — rare, keep going
                logger.debug("delete_keys: %s: %s", path, e)
                continue
            n += 1
            b += size
            state = state_map.get(key)
            if state is not None:
                state.result = False
        if n:
            _kv_event({"op": "prune", "req": getattr(req_context, "req_id", None),
                       "n": n, "bytes": b, "cand": len(keys)})
        return (n, b)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        try:
            _kv_event({"op": "req_end", "req": req_context.req_id})
            seen = getattr(self, "_ev_touched", None)
            if seen is not None:
                seen.discard(req_context.req_id)
        except Exception:  # noqa: BLE001
            pass

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
