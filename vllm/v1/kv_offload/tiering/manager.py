# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a CPU primary tier (with direct GPU access)
and zero or more secondary tiers (Storage, Network, etc.) to provide
hierarchical KV cache offloading.

Key Design Principles:
1. Always offload to all tiers — When a block is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Secondary tiers cannot access GPU memory
   directly; all data flows through the CPU primary tier
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_read() increments ref_cnt,
   protecting blocks from eviction until complete_read() is called
"""

import os
import time
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    ParentManager,
    SecondaryTierManager,
    TieringOffloadingMetrics,
)

logger = init_logger(__name__)


@dataclass
class PendingPromotion:
    """Accumulator for blocks awaiting submit_load() for one (tier, request)."""

    req_context: ReqContext
    keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class RequestState:
    req_context: ReqContext
    pending_primary_stores: int = 0
    is_finished: bool = False
    request_level_tiers: set[SecondaryTierManager] | None = None
    pending_cascade_keys: list[OffloadKey] = field(default_factory=list)
    sync_lookup_delay: float = 0.0
    # time.monotonic() of this request's first deferred secondary-tier lookup;
    # None once consumed (observed) or while no secondary lookup is pending.
    secondary_lookup_start_time: float | None = None


class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager with a primary/secondary transfer interface.

    The inherited prepare_store/complete_store/prepare_load/complete_load are the
    GPU-facing OffloadingManager interface. These aliases expose the same operations
    from the secondary tier perspective, where read/write refers to secondary
    accessing primary. This avoids confusion when reading TieringOffloadingManager
    code (e.g. calling prepare_load inside a cascade/store path would be misleading).
    """

    def __init__(
        self,
        num_blocks: int,
        mmap_region: SharedOffloadRegion,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
    ):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,
            cache_policy_module_path=cache_policy_module_path,
            enable_events=enable_events,
        )
        self._mmap_region = mmap_region
        # read/write is for CPU<->secondary transfers,
        # load/store is for CPU<->GPU transfers.
        # These aliases avoid calling prepare_load inside a store path.
        self.prepare_read = self.prepare_load
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

        self._kv_memoryview = mmap_region.create_kv_memoryview()

    def get_kv_memoryview(self) -> memoryview:
        """Return the memoryview over the primary tier's KV cache buffer.

        The view has shape (num_blocks, row_stride_bytes) and is backed by the
        SharedOffloadRegion mmap.  Secondary tiers address block *b* as
        ``view[b]``.
        """
        return self._kv_memoryview

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self._kv_memoryview.release()
        self._mmap_region.cleanup()


class _SecondaryTierFacingParent(ParentManager):
    """Wrapper that implements ParentManager by delegating to the
    TieringOffloadingManager with exclude_tier set to the origin tier."""

    __slots__ = ("_m", "_origin")

    def __init__(
        self,
        manager: "TieringOffloadingManager",
        tier: SecondaryTierManager,
    ):
        self._m = manager
        self._origin = tier

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return self._m.on_new_request(req_context, exclude_tier=self._origin)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return self._m.lookup(key, req_context, exclude_tier=self._origin)

    def create_store_job(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> JobMetadata:
        return self._m.create_store_job(keys, req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        return self._m.on_request_finished(req_context, exclude_tier=self._origin)


_DSPARK_TRACE_ON = os.environ.get("DSPARK_LOOKUP_TRACE") == "1"
_dspark_trace_counts: dict = {}


def _dspark_trace(kind, req_id, key):
    if not _DSPARK_TRACE_ON:
        return
    try:
        from vllm.v1.kv_offload.base import get_offload_group_idx as _ggi

        g = _ggi(key)
    except Exception:
        g = -1
    k = (req_id, kind, g)
    n = _dspark_trace_counts.get(k, 0)
    if n < 60:
        _dspark_trace_counts[k] = n + 1
        logger.info("DSPARK_TRACE %s req=%s g=%d n=%d", kind, req_id, g, n)


class TieringOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a CPU primary tier (with direct GPU access)
    and zero or more secondary tiers (Storage, Network, etc.) to provide
    hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished_jobs()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
    ):
        """
        Initialize the TieringOffloadingManager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
        """
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []

        self._job_id_counter: int = 0
        # Job tracking: maps job_id to metadata for all in-flight transfers.
        # JobMetadata.is_promotion distinguishes direction:
        #   True:  secondary → primary (promotion)
        #   False: primary → secondary (cascade)
        self._transfer_jobs: dict[JobId, JobMetadata] = {}

        # Pending promotion requests accumulated during lookup() calls; flushed
        # as one batched submit_load() per (tier, request) in on_schedule_end().
        # Outer key: tier. Inner key: req_context.req_id — the same ReqContext
        # object is reused for all block lookups of a given request per engine step.
        self._pending_load_submissions: dict[
            SecondaryTierManager, dict[str, PendingPromotion]
        ] = {}

        # DSPARK compact 멀티노드: fs에서 승격된 키 추적. 승격은 스케줄러
        # 노드의 mmap에만 써지므로, 이 키들의 로드에는 파일 경로를 첨부해
        # 각 워커가 공유 fs에서 직접 읽어 자기 mmap을 채우게 한다.
        # LRU-bounded (was an unbounded, never-pruned set that also
        # permanently rerouted every load of a once-promoted key through
        # the fs-path attach). Bounding is safe: evicting an entry only
        # means a later prepare_load() will not attach an fs path for that
        # key, so workers read the staging copy instead (the head/rank0
        # mmap holds the promoted bytes). With the cap (default 500k) far
        # above the staging tier block count (~147), a key can only age out
        # of this LRU long after its staging block was itself LRU-evicted;
        # any later use then needs a fresh fs promotion, which re-inserts
        # the key here with fresh recency.
        from collections import OrderedDict as _OD

        # DSPARK primfull-retry: per-request count of lookups that found the
        # chunk on disk but could not allocate staging (primary full, e.g.
        # another request's in-flight load still pins its blocks).
        self._primfull_retries: dict = {}
        self._primfull_last_tick: dict = {}
        try:
            # Interpreted as SECONDS of sustained primary-full per request
            # (ticked at most once per second regardless of how many keys a
            # lookup pass touches).
            self._primfull_retry_cap = int(
                os.environ.get("DSPARK_PRIMFULL_RETRIES", "300") or 300
            )
        except ValueError:
            self._primfull_retry_cap = 300

        # [load-error-recompute] keys whose staging bytes were destroyed
        # by a failed worker-side load (zero-fill fallback). lookup()
        # treats them as MISS so the scheduler recomputes instead of
        # re-offering the same failing chunks forever (retry livelock).
        # An entry lifts itself once the garbage staging block is gone
        # (see lookup), after which a fresh store may legitimately
        # re-offer the key.
        self._dspark_poisoned_keys: set = set()
        self._fs_promoted_keys: dict = _OD()
        try:
            _cap = int(
                os.environ.get("DSPARK_FS_PROMOTED_KEYS_CAP", "500000") or 500000
            )
        except ValueError:
            _cap = 500000
        self._fs_promoted_keys_cap: int = max(1, _cap)

        # Gate for once-per-step execution of _maybe_process_finished_jobs().
        # Reset at the end of each step in on_schedule_end().
        self._processed_jobs_this_step: bool = False

        # Per-request state for prepared GPU->primary stores and finalization.
        # Secondary tiers are finalized only after pending primary stores reach
        # complete_store(), since complete_store() can still submit cascades.
        self._req_state: dict[str, RequestState] = {}

        # Cached ParentManager wrappers for each secondary tier.
        self._tier_parents: dict[SecondaryTierManager, _SecondaryTierFacingParent] = {
            tier: _SecondaryTierFacingParent(self, tier)
            for tier in self.secondary_tiers
        }

        # Buffers manager-level observations (e.g. lookup delay) between
        # get_stats() calls; merged in and reset each time get_stats() runs.
        self._stats = OffloadingConnectorStats()

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _maybe_process_finished_jobs(self):
        """
        Poll secondary tiers for completed jobs (at most once per step).

        Guarded by _processed_jobs_this_step: the first call in an engine step
        does the actual polling; subsequent calls are no-ops. The flag is reset
        in on_schedule_end() at the end of each step.
        """
        if self._processed_jobs_this_step:
            return
        self._processed_jobs_this_step = True
        self._process_finished_jobs()

    def _process_finished_jobs(self):
        """
        Unconditionally poll all secondary tiers for completed jobs.

        This method:
        1. Calls get_finished_jobs() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_read()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_write()
           to make blocks available
        """
        for i, tier in enumerate(self.secondary_tiers):
            for completed_job in tier.get_finished_jobs():
                job_id = completed_job.job_id
                job_metadata = self._transfer_jobs.pop(job_id, None)
                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                if job_metadata.is_promotion:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    self.primary_tier.complete_write(
                        job_metadata.keys,
                        job_metadata.req_context,
                        completed_job.success,
                    )
                    if completed_job.success and hasattr(tier, "file_mapper"):
                        self._note_fs_promoted(job_metadata.keys)
                    if not completed_job.success:
                        # [load-error-recompute-p60] engine-side promotion
                        # failed. complete_write(success=False) above already
                        # released the staging blocks; now ride the recompute
                        # report path (poison + lookup-cache invalidation).
                        self._dspark_fail_promotion(tier, job_metadata)
                else:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    self.primary_tier.complete_read(
                        job_metadata.keys, job_metadata.req_context
                    )

    def _dspark_fail_promotion(self, tier, job_metadata) -> None:
        """[load-error-recompute-p60] Handle a FAILED secondary->primary
        promotion job (fs load error: FileNotFoundError, short-read EIO,
        any OSError -- the thread pool completes them all as
        success=False).

        The staging blocks are already released by
        complete_write(success=False), so nothing stays load-pinned; what
        remains broken without this handler is the report path: the fs
        tier's AsyncLookupManager cache was populated while the (possibly
        truncated) file still existed, and the short-read unlink policy
        never invalidates it, so lookup() keeps answering HIT and the
        scheduler re-promotes the same failing chunk every step -- the
        request parks in WAITING forever (silent retry livelock, E1
        2026-08-28 03:15, 8487 resubmits in 8 min).

        Fix: (1) poison the keys so TieringOffloadingManager.lookup()
        answers MISS and the scheduler recomputes (same mechanism as the
        worker-side patch_40/41 wiring; the poison self-lifts on the next
        lookup because the staging block is already gone, see lookup());
        (2) invalidate the tier's cached lookup state so the stale HIT
        cannot re-offer the keys after the poison self-lifts.
        """
        keys = list(job_metadata.keys)
        self._dspark_poisoned_keys.update(keys)
        for _k in keys:
            self._fs_promoted_keys.pop(_k, None)
        _invalidate = getattr(tier, "dspark_invalidate_lookup", None)
        if _invalidate is not None:
            try:
                _invalidate(keys)
            except Exception:
                logger.exception(
                    "[load-error-recompute] lookup-cache invalidation "
                    "failed on tier %s",
                    tier.tier_type,
                )
        logger.error(
            "[load-error-recompute] promotion job %d failed on tier %s: "
            "poisoned %d chunk keys for recompute (req=%s)",
            job_metadata.job_id,
            tier.tier_type,
            len(keys),
            getattr(job_metadata.req_context, "req_id", None),
        )

    @override
    def primary_capacity_blocks(self) -> int:
        """Total block slots in the primary (CPU staging) tier."""
        return self.primary_tier._num_blocks

    def dspark_delete_keys(
        self, keys: Collection[OffloadKey], req_context: ReqContext | None = None
    ) -> tuple[int, int]:
        """DSPARK_TAIL_KEEP: drop superseded window snapshots from the
        secondary tiers (best effort). Returns (files_deleted, bytes_freed)
        summed over tiers that implement ``delete_keys``.

        The primary (CPU staging) copy, if any, is left alone: it is LRU
        managed and a later lookup that hits it is still correct. The
        fs-promoted LRU and the load-failure poison set are deliberately
        left alone too (review 2026-08-30): popping a promoted key while its
        staging copy is still HIT would send rank>0 workers to their own
        never-filled staging bytes instead of the fs path, and lifting a
        poison while the zero-filled staging block is still resident would
        turn the next lookup into a garbage load. A worker that follows a
        stale fs path to a deleted file simply fails the load and the
        request recomputes (load-error-recompute).
        """
        if not keys:
            return (0, 0)
        n = b = 0
        for tier in self.secondary_tiers:
            fn = getattr(tier, "delete_keys", None)
            if fn is None:
                continue
            try:
                dn, db = fn(keys, req_context)
            except Exception:  # noqa: BLE001 — retention must never break a step
                logger.exception(
                    "secondary tier %s: delete_keys failed",
                    getattr(tier, "tier_type", type(tier).__name__),
                )
                continue
            n += dn
            b += db
        return (n, b)

    def _note_fs_promoted(self, keys: Collection[OffloadKey]) -> None:
        """Record fs->primary promoted keys, LRU-bounded.

        prepare_load() attaches direct fs read paths for keys present here
        so each worker can fill its own mmap from the shared fs. Overflow
        evicts the oldest entry; see __init__ for why that is safe.
        """
        lru = self._fs_promoted_keys
        for key in keys:
            if key in lru:
                lru.move_to_end(key)
            else:
                lru[key] = None
        cap = self._fs_promoted_keys_cap
        while len(lru) > cap:
            lru.popitem(last=False)

    def lookup(
        self,
        key: OffloadKey,
        req_context: ReqContext,
        *,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> LookupResult:
        """
        Check whether a single block is offloaded and ready.

        Algorithm:
            1. Process any completed async jobs first.
            2. Query primary tier — short-circuit on hit or in-flight.
            3. On primary miss, query secondary tiers — stop on first
               hit and initiate promotion.

        Args:
            key: Block hash to look up.
            req_context: Per-request context.

        Returns:
            HIT       — block is ready in the primary tier.
            HIT_PENDING — block found but not yet readable (write
                        in-flight on the primary tier).
            RETRY     — promotion started or a secondary tier is busy.
            MISS      — block not found in any tier, or primary is full
                        and cannot accept a promotion.
        """
        # Poll first so a promotion that finished since the last call is
        # already reflected as HIT (not stale HIT_PENDING/MISS) below, and
        # so blocks freed by cascade or promotion completions are evictable
        # in time for a promotion this lookup may initiate.
        self._maybe_process_finished_jobs()

        req_state = self._req_state.get(req_context.req_id)

        # [load-error-recompute] Poisoned keys (staging bytes destroyed by
        # a failed worker load) must not be offered again: report MISS so
        # the scheduler recomputes. Self-lift: once the garbage block is
        # gone from the primary tier (evicted), drop the poison so a
        # later fresh store can re-offer the key.
        if self._dspark_poisoned_keys and key in self._dspark_poisoned_keys:
            if self.primary_tier.lookup(key, req_context) is LookupResult.MISS:
                self._dspark_poisoned_keys.discard(key)
            return LookupResult.MISS

        primary_hit = self.primary_tier.lookup(key, req_context)
        if primary_hit is LookupResult.HIT:
            return LookupResult.HIT
        if primary_hit is LookupResult.HIT_PENDING:
            return LookupResult.HIT_PENDING

        lookup_start = time.monotonic()
        any_retry = False
        for tier in self.secondary_tiers:
            if tier is exclude_tier:
                continue
            if not req_context.load_tier_filter.allows(tier.medium, tier.locality):
                continue
            result = tier.lookup(key, req_context)
            if result is LookupResult.HIT:
                if not getattr(req_context, "dspark_allow_promotion", True):
                    # DSPARK restore admission gate: the connector denied
                    # this request a promotion slot for this pass. Do not
                    # initiate the fs->staging promotion; defer the request
                    # exactly like a pending lookup so it retries next
                    # step. Keys already resident in the primary tier were
                    # answered above, so ungated staging-only lookups are
                    # unaffected.
                    self._accumulate_lookup_sync_delay(req_state, lookup_start)
                    if (
                        req_state is not None
                        and req_state.secondary_lookup_start_time is None
                    ):
                        req_state.secondary_lookup_start_time = lookup_start
                    return LookupResult.RETRY
                promoted = self._initiate_promotion(tier, key, req_context)
                if promoted:
                    # Tell the connector-side admission gate this request
                    # now owns promotion work in flight.
                    req_context.dspark_promotion_initiated = True
                self._accumulate_lookup_sync_delay(req_state, lookup_start)
                if (
                    req_state is not None
                    and promoted
                    and req_state.secondary_lookup_start_time is None
                ):
                    req_state.secondary_lookup_start_time = lookup_start
                # 업스트림 #51840 백포트: 승격 트리거 시 HIT_PENDING —
                # 요청이 진행하며 승격→로드→해제가 파이프라인으로 드레인된다.
                if not promoted:
                    _dspark_trace("primfull_miss", req_context.req_id, key)
                    # DSPARK: the chunk file EXISTS on disk; only the primary
                    # (staging) tier is full - typically another request's
                    # in-flight load still pins its blocks. That is transient,
                    # so retry next step instead of collapsing the whole
                    # hybrid restore into a definitive MISS (= full
                    # recompute). Bounded per request: after
                    # DSPARK_PRIMFULL_RETRIES consecutive full passes fall
                    # back to the old MISS behavior.
                    _now = time.monotonic()
                    _rid = req_context.req_id
                    if _now - self._primfull_last_tick.get(_rid, 0.0) >= 1.0:
                        self._primfull_last_tick[_rid] = _now
                        self._primfull_retries[_rid] = (
                            self._primfull_retries.get(_rid, 0) + 1
                        )
                    if self._primfull_retries.get(_rid, 0) <= self._primfull_retry_cap:
                        return LookupResult.RETRY
                    return LookupResult.MISS
                self._primfull_retries.pop(req_context.req_id, None)
                return LookupResult.HIT_PENDING
            if result is LookupResult.RETRY:
                any_retry = True

        self._accumulate_lookup_sync_delay(req_state, lookup_start)
        if any_retry:
            if req_state is not None and req_state.secondary_lookup_start_time is None:
                req_state.secondary_lookup_start_time = lookup_start
            return LookupResult.RETRY
        _dspark_trace("secmiss", req_context.req_id, key)
        return LookupResult.MISS

    def _accumulate_lookup_sync_delay(
        self, req_state: RequestState | None, start_time: float
    ) -> None:
        """Accumulate secondary-tier lookup time until allocation or finish."""
        if req_state is not None:
            req_state.sync_lookup_delay += time.monotonic() - start_time

    def _maybe_observe_lookup_sync_delay(self, req_state: RequestState) -> None:
        delay = req_state.sync_lookup_delay
        if delay == 0:
            return
        req_state.sync_lookup_delay = 0.0
        self._stats.observe_histogram(
            TieringOffloadingMetrics.LOOKUP_SYNC_DELAY,
            delay,
        )

    def _maybe_observe_lookup_async_delay(self, req_state: RequestState) -> None:
        """Flush a pending deferred secondary-tier lookup timer, if any."""
        start_time = req_state.secondary_lookup_start_time
        if start_time is None:
            return
        req_state.secondary_lookup_start_time = None
        self._stats.observe_histogram(
            TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _initiate_promotion(
        self,
        tier: SecondaryTierManager,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """
        Queue a block for promotion from a secondary tier to the primary tier.

        Allocates space in the primary tier immediately (sets ref_cnt=-1 so
        subsequent lookups within the same step see the slot as in-flight),
        then defers the actual submit_load() call to _flush_pending_promotions()
        so all blocks queued during one engine step are submitted as a single
        batched job.

        Args:
            tier: The secondary tier to promote from
            key: Block to promote
            req_context: Per-request context forwarded to primary.prepare_write().

        Returns:
            True if promotion was initiated, False if primary tier is full.
        """
        # Allocate space in primary tier for promoted block.
        # Must happen immediately so primary.lookup() returns None (in-flight)
        # for this key on any subsequent lookup() call within the same step,
        # preventing duplicate promotion attempts.
        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the block as unavailable
            # rather than retrying indefinitely.
            return False

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, CPULoadStoreSpec)
        # Defer submit_load to on_schedule_end(). Group by (tier, request) so
        # each request's blocks are submitted as one batched job per tier.
        tier_pending = self._pending_load_submissions.setdefault(tier, {})
        ctx_id = req_context.req_id
        if ctx_id not in tier_pending:
            tier_pending[ctx_id] = PendingPromotion(
                keys=[], block_ids=[], req_context=req_context
            )
        entry = tier_pending[ctx_id]
        entry.keys.extend(primary_write_result.keys_to_store)
        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _flush_pending_promotions(self) -> None:
        """Submit one batched submit_load() per (tier, request).

        Called from on_schedule_end() at the end of each scheduler step,
        flushing all promotion requests deferred during lookup().
        """
        if not self._pending_load_submissions:
            return

        for tier, pending_by_ctx in self._pending_load_submissions.items():
            for entry in pending_by_ctx.values():
                job_id = self._next_job_id()
                job_metadata = JobMetadata(
                    job_id=job_id,
                    keys=entry.keys,
                    block_ids=np.array(entry.block_ids, dtype=np.int64),
                    is_promotion=True,
                    req_context=entry.req_context,
                )
                self._transfer_jobs[job_id] = job_metadata
                # [load-error-recompute-p60] a submit-time failure must
                # complete the job as failed: otherwise the exception
                # escapes on_schedule_end() (engine crash) or leaks a
                # never-finishing job whose staging blocks stay
                # write-pinned forever. This engine-side path has no TP
                # collectives (unlike the worker connector submit path),
                # so swallowing here cannot desync ranks.
                try:
                    tier.submit_load(job_metadata)
                except Exception:
                    logger.exception(
                        "[load-error-recompute] submit_load failed on "
                        "tier %s job %d; synthesizing failed completion",
                        tier.tier_type,
                        job_id,
                    )
                    self._transfer_jobs.pop(job_id, None)
                    self.primary_tier.complete_write(
                        job_metadata.keys, job_metadata.req_context, False
                    )
                    self._dspark_fail_promotion(tier, job_metadata)

        self._pending_load_submissions.clear()

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        Callers only pass keys already confirmed HIT by lookup() earlier this
        step.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Blocks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        spec = self.primary_tier.prepare_load(keys, req_context)
        if self._fs_promoted_keys:
            fs_tier = next(
                (t for t in self.secondary_tiers if hasattr(t, "file_mapper")),
                None,
            )
            if fs_tier is not None:
                paths = [
                    fs_tier.direct_read_info(k, None)
                    if k in self._fs_promoted_keys
                    else None
                    for k in keys
                ]
                import logging as _lg

                _lg.getLogger(__name__).info(
                    "DSPARK prep_load keys=%d promoted_hits=%d",
                    len(list(keys)),
                    sum(1 for p in paths if p is not None),
                )
                if any(p is not None for p in paths):
                    spec.fs_paths = paths  # type: ignore[attr-defined]
        return spec

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used in all tiers.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        self.primary_tier.touch(keys, req_context)
        for tier in self.secondary_tiers:
            tier.touch(keys, req_context)

    @override
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Blocks that finished loading.
            req_context: Per-request context.
        """
        self.primary_tier.complete_load(keys, req_context)

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        For request-level tiers, blocks already present in the primary tier
        are immediately cascaded via submit_store().

        Args:
            keys: Blocks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        # Step 1: Poll for completed async jobs FIRST
        # _process_finished_jobs() handles two kinds of completions here:
        #  - Cascade completions (store to a secondary tier, either a local
        #    cascade or a store job created for a remote requester via
        #    create_store_job()): decrements ref_cnt on the primary blocks
        #    that were read, making them evictable again once ref_cnt hits 0.
        #  - Promotion completions (secondary->primary loads): sets a
        #    not-yet-ready block's ref_cnt from -1 to 0 via complete_write(),
        #    making it evictable for the first time.
        # Both must be accounted for before the eviction decision below.
        self._maybe_process_finished_jobs()

        # Step 2: Store to primary tier (new blocks only).
        # Cascading of these newly-stored blocks to ALL secondary tiers
        # happens later in complete_store(), after the GPU→Primary transfer
        # completes.
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None

        if primary_result.keys_to_store:
            state = self._req_state[req_context.req_id]
            state.pending_primary_stores += 1

        # Step 3: For request-level tiers, cascade blocks already in primary
        request_level_tiers = self._req_state[req_context.req_id].request_level_tiers
        if request_level_tiers:
            keys_to_store_set = set(primary_result.keys_to_store)
            keys_already_in_primary = tuple(
                k for k in keys if k not in keys_to_store_set
            )
            if keys_already_in_primary:
                self._cascade_existing_blocks_to_request_level_tiers(
                    keys_already_in_primary, req_context, request_level_tiers
                )

        return primary_result

    def _cascade_existing_blocks_to_request_level_tiers(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        request_level_tiers: set[SecondaryTierManager],
    ) -> None:
        """
        For tiers that requested request-level policy, submit_store() for
        blocks that are already present in the primary tier.

        A key whose primary write is still in flight (HIT_PENDING) cannot be
        dropped: prepare_store already excluded it as present, and the
        scheduler advances past its chunk, so no path offers it again. Park it
        instead. MISS keys are dropped, since nothing is there to read.

        The primary tier resolves every key it holds, so RETRY cannot reach
        here. Parking on it would have no guarantee of ever draining, which is
        what makes parking HIT_PENDING safe, so it is rejected rather than
        guessed at. (vLLM #53329)
        """
        state = self._req_state[req_context.req_id]
        ready_keys: list[OffloadKey] = []
        for key in keys:
            result = self.primary_tier.lookup(key, req_context)
            if result is LookupResult.HIT:
                ready_keys.append(key)
            elif result is LookupResult.HIT_PENDING:
                state.pending_cascade_keys.append(key)
            else:
                assert result is LookupResult.MISS, (
                    f"primary tier returned {result} for a cascade key"
                )
        if not ready_keys:
            return

        for tier in request_level_tiers:
            job_metadata = self.create_store_job(tuple(ready_keys), req_context)
            tier.submit_store(job_metadata)

    def _flush_pending_cascades(self) -> None:
        """Retry request-level cascades parked on an in-flight primary write.

        A parked key always resolves, to HIT or to MISS, so the set drains and
        a request cannot be held from finalization forever. (vLLM #53329)
        """
        for req_id, state in list(self._req_state.items()):
            if not state.pending_cascade_keys:
                continue
            assert state.request_level_tiers
            keys, state.pending_cascade_keys = state.pending_cascade_keys, []
            self._cascade_existing_blocks_to_request_level_tiers(
                keys, state.req_context, state.request_level_tiers
            )
            self._maybe_finalize_request(req_id)

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.prepare_read() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            keys: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_read().
        """
        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(keys, req_context, success)

        if success:
            # Step 2: Cascade to ALL secondary tiers
            # For each secondary tier, call primary.prepare_read() to get the
            # LoadStoreSpec AND to increment ref_cnt (protecting blocks from
            # eviction during the async transfer). One prepare_read() call per
            # secondary tier.
            for tier in self.secondary_tiers:
                job_metadata = self.create_store_job(keys, req_context)
                tier.submit_store(job_metadata)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().
        req_id = req_context.req_id
        state = self._req_state[req_id]
        assert state.pending_primary_stores > 0
        state.pending_primary_stores -= 1
        self._maybe_finalize_request(req_id)

    def create_store_job(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> JobMetadata:
        """Pin blocks in the primary tier and create a tracked store job.

        Calls prepare_read() to increment ref_cnt (protecting blocks
        from eviction during the async transfer), allocates a job ID,
        and registers the job in _transfer_jobs.

        The caller is responsible for the actual data transfer and
        reporting completion via get_finished_jobs().
        """
        primary_blocks_spec = self.primary_tier.prepare_read(keys, req_context)
        assert isinstance(primary_blocks_spec, CPULoadStoreSpec)
        job_id = self._next_job_id()
        job_metadata = JobMetadata(
            job_id=job_id,
            keys=keys,
            block_ids=primary_blocks_spec.block_ids,
            is_promotion=False,
            req_context=req_context,
        )
        self._transfer_jobs[job_id] = job_metadata
        return job_metadata

    @override
    def on_new_request(
        self,
        req_context: ReqContext,
        *,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> RequestOffloadingContext:
        """
        Query each secondary tier for its offload policy preference.

        Returns REQUEST_LEVEL if ANY secondary tier wants request-level.
        Only stores REQUEST_LEVEL tier decisions for use in prepare_store.
        """
        state = RequestState(req_context=req_context)
        for tier in self.secondary_tiers:
            if tier is exclude_tier:
                continue
            tier_ctx = tier.on_new_request(req_context)
            if tier_ctx.policy == OffloadPolicy.REQUEST_LEVEL:
                if state.request_level_tiers is None:
                    state.request_level_tiers = set()
                state.request_level_tiers.add(tier)
        self._req_state[req_context.req_id] = state

        policy = (
            OffloadPolicy.REQUEST_LEVEL
            if state.request_level_tiers
            else OffloadPolicy.BLOCK_LEVEL
        )
        return RequestOffloadingContext(policy=policy)

    @override
    def on_request_finished(
        self,
        req_context: ReqContext,
        *,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> None:
        self.primary_tier.on_request_finished(req_context)
        self._primfull_retries.pop(req_context.req_id, None)
        self._primfull_last_tick.pop(req_context.req_id, None)
        state = self._req_state[req_context.req_id]
        state.is_finished = True
        self._maybe_finalize_request(req_context.req_id, exclude_tier)

    def _maybe_finalize_request(
        self,
        req_id: str,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> None:
        """Finalize secondary tiers once no more store cascades can be submitted.

        Finalization means forwarding on_request_finished() to secondary tiers.
        It is delayed until pending GPU->primary stores finish, since their
        complete_store() callbacks may still submit primary->secondary stores.
        """
        state = self._req_state[req_id]
        if not state.is_finished:
            return
        if state.pending_primary_stores != 0:
            return
        if state.pending_cascade_keys:
            return

        for tier in self.secondary_tiers:
            if tier is exclude_tier:
                continue
            tier.on_request_finished(state.req_context)
        self._maybe_observe_lookup_sync_delay(state)
        self._maybe_observe_lookup_async_delay(state)
        del self._req_state[req_id]

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """End-of-schedule hook: process finished jobs, flush deferred
        promotions, and reset the per-step gate.

        Called once per scheduler step from
        OffloadingConnectorScheduler.build_connector_meta().
        """
        # Catch-all poll: guarantees jobs are processed even on steps where
        # lookup()/prepare_store() were never called (e.g. no requests
        # scheduled but a tier still has_pending_work()).
        self._maybe_process_finished_jobs()

        for tier in self.secondary_tiers:
            tier.serve_external_requests(self._tier_parents[tier])

        # Reset the per-step gate AFTER serve_external_requests so that
        # lookup() calls within it skip redundant _process_finished_jobs().
        self._processed_jobs_this_step = False

        self._flush_pending_promotions()
        self._flush_pending_cascades()
        for tier in self.secondary_tiers:
            tier.on_schedule_end(context)

        for req_id in context.new_req_ids:
            state = self._req_state.get(req_id)
            if state is None:
                continue
            self._maybe_observe_lookup_sync_delay(state)
            self._maybe_observe_lookup_async_delay(state)

    @override
    def has_pending_work(self) -> bool:
        # In-flight primary<->secondary transfers (pending promotions are
        # translated to transfer jobs in on_schedule_end), plus any work the
        # secondary tiers themselves still have outstanding.
        return (
            bool(self._transfer_jobs)
            or any(state.pending_cascade_keys for state in self._req_state.values())
            or any(tier.has_pending_work() for tier in self.secondary_tiers)
        )

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        """Yield events owned by the primary and secondary tiers.

        Yields:
            New OffloadingEvents collected by each tier since the last call.
        """
        yield from self.primary_tier.take_events()
        for tier in self.secondary_tiers:
            yield from tier.take_events()

    @override
    def reset_cache(self) -> None:
        """Reset transfer bookkeeping and primary-tier cache.

        Called during sleep, weight update, or resume. Each secondary tier
        drains its in-flight transfers via drain_jobs() so no tier I/O is
        touching primary memory before the primary tier is reset. A stuck
        tier will block here visibly — preferable to silent corruption
        from reusing primary slots while a transfer is mid-copy.

        Secondary tiers are intentionally not reset: persistent stores
        (FS, network) keep their data across resets. Active request state is
        retained so those requests can continue after the reset; finished
        requests are finalized and removed.
        """
        for tier in self.secondary_tiers:
            tier.drain_jobs()
        # All tier I/O has stopped; consume their completion notifications
        # so manager bookkeeping is consistent before the primary reset.
        self._process_finished_jobs()

        # Deferred promotion submissions reserve primary slots that the
        # reset below invalidates; their submit_load() has not yet been
        # called so no tier I/O is touching that memory.
        self._pending_load_submissions.clear()

        finished_req_ids = []
        for req_id, state in self._req_state.items():
            state.pending_primary_stores = 0
            state.pending_cascade_keys.clear()
            if not state.is_finished:
                continue
            for tier in self.secondary_tiers:
                tier.on_request_finished(state.req_context)
            self._maybe_observe_lookup_sync_delay(state)
            self._maybe_observe_lookup_async_delay(state)
            finished_req_ids.append(req_id)

        self.primary_tier.reset_cache()

        for req_id in finished_req_ids:
            del self._req_state[req_id]
        self._processed_jobs_this_step = False

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self.primary_tier.get_stats()

        if stats is not None and stats.is_empty():
            stats = None

        for tier in self.secondary_tiers:
            tier_stats = tier.get_stats()
            if tier_stats is None or tier_stats.is_empty():
                continue
            if stats is None:
                stats = tier_stats
            else:
                stats.aggregate(tier_stats)

        if not self._stats.is_empty():
            if stats is None:
                stats = self._stats
            else:
                stats.aggregate(self._stats)
            self._stats = OffloadingConnectorStats()

        return stats

    @override
    def shutdown(self) -> None:
        """Shutdown all tiers and release resources."""
        for tier in self.secondary_tiers:
            tier.shutdown()
        self.primary_tier.shutdown()
