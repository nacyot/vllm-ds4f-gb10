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
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs import io as fs_io
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

_O_DIRECT_ALIGN = 4096
# VLLM_KV_OFFLOAD_TRACE=1 logs every promotion job (blocks, bytes, seconds).
_TRACE = os.environ.get("VLLM_KV_OFFLOAD_TRACE", "0") == "1"
# A load job is split into at most n_read_threads tasks of at least this many
# blocks each, so a long restore reads on every thread while a small job
# stays a single task.
_MIN_BLOCKS_PER_TASK = 256


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
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        KV cache sharing between multiple vLLM instances using the same
        ``root_dir`` (e.g., via a shared PVC) works by default: ``NONE_HASH``
        (the chain-hash seed for block content hashes) is derived from a fixed
        default seed, so identical token content produces identical block
        filenames across instances. Setting the ``PYTHONHASHSEED`` environment
        variable to the same value on all instances overrides the default seed,
        and is required to share a cache when using a non-cryptographic
        prefix-caching hash algorithm, which seeds ``NONE_HASH`` randomly.
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
        checksums: bool = True,
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
            checksums: Record a CRC32 per block file (xattr) and verify it on
                load; a mismatch is a load failure for that block.
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
        # Keys of in-flight load (promotion) jobs, so a failed load can mark
        # its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Per load job: indices of the blocks that failed to load (the rest
        # were loaded). Each task of a split job appends its own failures
        # under the lock before it raises (so before its task_done, and the
        # last task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned.
        self._load_failures: dict[JobId, list[int]] = {}
        self._load_failures_lock = threading.Lock()
        self._n_read_threads = n_read_threads

        # Widest primary row; a group's file is at most this long.
        self._block_size: int = self._primary_row_bytes
        # Bytes persisted per key, by KV cache group. With a packed block
        # layout and a single valid worker slot (replicated layout or rank-0
        # relay) a group's KV is the prefix of slot 0, so only that prefix is
        # written; otherwise the whole CPU row is stored.
        config = offloading_spec.config
        single_slot = config.replicated_layout or bool(
            config.extra_config.get("relay_from_rank0", False)
        )
        # O_DIRECT needs page-multiple lengths; the CPU row is page aligned.
        self._group_bytes: list[int] = [
            min(round_up(group.bytes_per_block, _O_DIRECT_ALIGN), self._block_size)
            if (config.packed_layout and single_slot and group.bytes_per_block > 0)
            else self._block_size
            for group in config.groups
        ]
        self._compact = any(n != self._block_size for n in self._group_bytes)
        # A slab row must hold the whole file of its group.
        if not self._primary_layout.is_uniform:
            for group_idx, num_bytes in enumerate(self._group_bytes):
                cls = self._primary_layout.group_class[group_idx]
                assert cls < 0 or (
                    num_bytes <= self._primary_layout.classes[cls].row_bytes
                )
        self._checksums = checksums
        if checksums and fs_io._HAS_FSIO_C and not fs_io._HAS_VERIFY_C:
            logger.warning(
                "vllm.fs_io_C has no batch_verify_crc32; checksums are verified "
                "block by block in Python (slow under a busy scheduler). Rebuild "
                "the extension with csrc/build_fs_io.sh."
            )

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
            group_bytes=self._group_bytes if self._compact else None,
        )

        # Write config file; an existing one must describe the same run,
        # otherwise the directory hash would differ.
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        run_config = self.file_mapper.get_run_config()
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(run_config, f, indent=2, sort_keys=True)
        else:
            with open(config_path) as f:
                existing = json.load(f)
            if existing != run_config:
                raise ValueError(
                    f"KV offload store {config_path} was written by a different "
                    "run configuration; refusing to reuse it."
                )
        logger.info(
            "KV offload fs tier at %s: bytes per key by group %s (row %d)",
            self.file_mapper.base_path,
            self._group_bytes,
            self._block_size,
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

    def _key_sizes(self, keys: list[OffloadKey]) -> int | list[int]:
        if not self._compact:
            return self._block_size
        return [self._group_bytes[get_offload_group_idx(key)] for key in keys]

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
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        task = functools.partial(
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in keys],
            self._primary_kv_view,
            [self._slot_offset(bid) for bid in job_metadata.block_ids],
            self._key_sizes(keys),
            self._use_o_direct,
            self._checksums,
            True,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [self._slot_offset(bid) for bid in job_metadata.block_ids]
        sizes = self._key_sizes(keys)
        num_blocks = len(paths)
        n_tasks = max(
            1, min(self._n_read_threads, -(-num_blocks // _MIN_BLOCKS_PER_TASK))
        )

        def load_task(start: int, end: int) -> None:
            # Runs on a pool worker thread with the blocks [start, end) of the
            # job. Every block is loaded except the ones that failed twice
            # (see batch_load_block); record those so get_finished_jobs keeps
            # the rest. Raising marks the job as failed.
            failed = batch_load_block(
                paths[start:end],
                self._primary_kv_view,
                offsets[start:end],
                sizes if isinstance(sizes, int) else sizes[start:end],
                self._use_o_direct,
                self._checksums,
            )
            if failed:
                with self._load_failures_lock:
                    self._load_failures.setdefault(job_id, []).extend(
                        start + i for i in failed
                    )
                raise OSError(
                    f"{len(failed)} of {end - start} blocks failed to load "
                    f"(first: {paths[start + failed[0]]})"
                )

        tasks = [
            functools.partial(
                load_task, num_blocks * t // n_tasks, num_blocks * (t + 1) // n_tasks
            )
            for t in range(n_tasks)
        ]
        self._pool.enqueue_load(job_id, n_tasks, tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        for job_id, success, transfer_time in self._pool.get_finished():
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
            load_keys = self._load_job_keys.pop(job_id, None)
            with self._load_failures_lock:
                failed_indices = self._load_failures.pop(job_id, None)
            if failed_indices is not None:
                failed_indices.sort()
            if _TRACE and load_keys is not None:
                sizes = self._key_sizes(load_keys)
                logger.info(
                    "fs promotion job %d: %d blocks, %d bytes, %.3f s, %d failed",
                    job_id,
                    len(load_keys),
                    sum(sizes) if isinstance(sizes, list) else sizes * len(load_keys),
                    transfer_time,
                    len(failed_indices or ()),
                )
            if load_keys is not None and not success:
                # A batched load skips only the blocks that failed twice. The
                # loaded blocks are kept in the primary tier (reported via
                # successful_keys); only the failed keys are marked a miss and
                # recomputed. Without a failure record the job died before
                # loading (e.g. a bad offset), so every key failed.
                if failed_indices is None:
                    successful: list[OffloadKey] = []
                    failed = load_keys
                else:
                    failed_set = set(failed_indices)
                    successful = [
                        key for i, key in enumerate(load_keys) if i not in failed_set
                    ]
                    failed = [load_keys[i] for i in failed_indices]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                )
            )
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

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

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
