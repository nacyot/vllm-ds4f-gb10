import os
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import chain, islice
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _ConnectorMetricName,
    _TransferMetricName,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    Locality,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

KV_LOAD_TIERS_KEY = "kv_load_tiers"
MATCHER_MEDIUM_KEY = "medium"
MATCHER_LOCALITY_KEY = "locality"


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_chunks: int | None
    # Number of this group's offloaded chunks per full-attention alignment
    # segment. Used to skip storing SWA chunks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_chunk_count: int | None = None
    # DSPARK tail-only: g0 정렬 세그먼트 크기(이 그룹 청크 단위) —
    # 윈도우>=세그먼트라 alignment_chunk_count가 None인 그룹에도 앵커 제공.
    anchor_chunk_count: int | None = None
    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False


def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, tokens_per_chunk)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def is_store_reachable_swa_chunk(
    absolute_chunk_index: int,
    storable_chunk_count: int,
    alignment_chunk_count: int | None,
    sliding_window_chunks: int | None,
    is_eagle_group: bool,
    tail_only: bool = False,
    anchor_chunk_count: int | None = None,
) -> bool:
    """Return whether an SWA chunk can participate in an external-cache hit."""
    if tail_only:
        # 최종 꼬리만 저장하되, 앵커는 마지막 g0 정렬 경계다 —
        # 재접근은 디코드 텍스트 재토크나이즈로 끝부분 해시가 어궸나므로
        # 분기점 이전(정렬 경계)에서 윈도우 런이 맞아야 한다.
        if sliding_window_chunks is None:
            return True
        tail = sliding_window_chunks + int(is_eagle_group)
        if anchor_chunk_count:
            anchor = (storable_chunk_count // anchor_chunk_count) * anchor_chunk_count
        else:
            anchor = storable_chunk_count
        # 앵커 앞 윈도우 + 날자로 끝 꼬리(정확 연장 대비) 둘 다 허용.
        return (
            anchor - tail <= absolute_chunk_index < anchor
            or absolute_chunk_index >= storable_chunk_count - tail
        )
    if alignment_chunk_count is None:
        return True
    assert sliding_window_chunks is not None
    position_in_segment = absolute_chunk_index % alignment_chunk_count
    segment_start = absolute_chunk_index - position_in_segment
    actual_segment_length = min(
        alignment_chunk_count, storable_chunk_count - segment_start
    )
    reachable_tail = sliding_window_chunks + int(is_eagle_group)
    return position_in_segment >= actual_segment_length - reachable_tail


# NODE3_TAILONLY_FIX F1
def is_store_anchor_now_swa_chunk(
    absolute_chunk_index: int,
    final_prompt_chunks: int,
    anchor_chunk_count: int | None,
    sliding_window_chunks: int | None,
    is_eagle_group: bool,
    checkpoint_chunks: int | None = None,
) -> bool:
    """DSPARK_TAIL_ONLY=2 (anchor-now).

    유예 없이, 프롬프트의 마지막 완전 g0 정렬 경계(prompt_anchor) 직전 w(+e)
    청크와, 선택적 체크포인트 경계(checkpoint_chunks 배수, g0 경계에 정렬)
    직전 w(+e) 청크만 통과시킨다. 조회(_sliding_window_lookup)는 g0 경계에
    정렬된 슬라이스 끝에서 역방향 w 런을 찾으므로 이 집합만으로 전장 히트가
    성립한다. 청크는 계산된 스텝에 판정되므로 GPU 블록이 살아 있고 기존
    flush 보호를 받는다(유예 방식의 구멍 문제 없음).
    """
    if sliding_window_chunks is None:
        return True
    a = anchor_chunk_count or 1
    tail = sliding_window_chunks + int(is_eagle_group)
    prompt_anchor = (final_prompt_chunks // a) * a
    ckpt = checkpoint_chunks or 0
    if ckpt:
        ckpt = max(a, (ckpt // a) * a)
    for boundary in range(
        absolute_chunk_index + 1, absolute_chunk_index + tail + 1
    ):
        if boundary % a:
            continue
        if boundary > 0 and boundary == prompt_anchor:
            return True
        if ckpt and boundary % ckpt == 0:
            return True
    return False


def resolve_mamba_align_size(
    spec: "OffloadingSpec", kv_cache_config: KVCacheConfig
) -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be rounded
    down to a multiple of the offloaded chunk size. Asserts that all such
    groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
        kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    num_workers: int
    offload_prompt_only: bool
    tail_only: bool = False
    anchor_now: bool = False  # NODE3_TAILONLY_FIX F2
    anchor_ckpt_tokens: int = 0
    anchor_tokens_override: int = 0

    @classmethod
    def from_spec(
        cls,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the tokens_per_chunk of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_tokens_per_chunk) == 1:
            alignment_tokens = full_attn_tokens_per_chunk.pop()

        def _alignment_chunk_count(
            tokens_per_chunk: int,
            sliding_window_size_in_chunks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_chunks is None:
                return None
            if alignment_tokens <= tokens_per_chunk:
                return None
            per_segment = alignment_tokens // tokens_per_chunk
            if sliding_window_size_in_chunks >= per_segment:
                return None
            return per_segment

        eagle_groups = {
            idx
            for idx, g in enumerate(kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }

        use_eagle = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle()
        )
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing chunk of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        import os as _os
        _tail_only = _os.environ.get("DSPARK_TAIL_ONLY") == "1"
        _anchor_now = _os.environ.get("DSPARK_TAIL_ONLY") == "2"  # NODE3_TAILONLY_FIX F3
        _anchor_ckpt_tokens = int(
            _os.environ.get("DSPARK_TAIL_CKPT_TOKENS", "0") or 0
        )
        _anchor_tokens_override = int(
            _os.environ.get("DSPARK_TAIL_ANCHOR_TOKENS", "0") or 0
        )
        if _anchor_now:
            logger.info(
                "KV offloading: DSPARK anchor-now store enabled (TAIL_ONLY=2) — "
                "window groups store only the w(+e) chunks before the prompt's "
                "last full-attention-aligned boundary (ckpt_tokens=%d), "
                "decided at compute time (no deferral). anchor_tokens_override=%d",
                _anchor_ckpt_tokens, _anchor_tokens_override,
            )
        if _tail_only:
            logger.info(
                "KV offloading: DSPARK tail-only store enabled — window/state "
                "groups store only the final window at request finish."
            )
        _n3_cfg = cls(  # NODE3_TAILONLY_INSTR
            tail_only=_tail_only,
            anchor_now=_anchor_now,  # NODE3_TAILONLY_FIX F3b
            anchor_ckpt_tokens=_anchor_ckpt_tokens,
            anchor_tokens_override=_anchor_tokens_override,
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    tokens_per_block=tokens_per_block,
                    tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                    hashes_per_chunk=(
                        (tokens_per_block * spec.blocks_per_chunk)
                        // spec.tokens_per_hash
                    ),
                    sliding_window_size_in_chunks=(
                        sw := get_sliding_window_size_in_chunks(
                            kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            tokens_per_block * spec.blocks_per_chunk,
                        )
                    ),
                    alignment_chunk_count=_alignment_chunk_count(
                        tokens_per_block * spec.blocks_per_chunk, sw
                    ),
                    anchor_chunk_count=(
                        alignment_tokens // (tokens_per_block * spec.blocks_per_chunk)
                        if (
                            sw is not None
                            and alignment_tokens is not None
                            and alignment_tokens
                            >= tokens_per_block * spec.blocks_per_chunk
                        )
                        else None
                    ),
                    kv_event_group_spec=get_offloading_event_group_spec(
                        kv_cache_config.kv_cache_groups[idx]
                    ),
                    is_eagle_group=idx in eagle_groups,
                )
                for idx, tokens_per_block in enumerate(spec.tokens_per_block)
            ),
            blocks_per_chunk=spec.blocks_per_chunk,
            offload_prompt_only=spec.offload_prompt_only,
        )
        # NODE3_TAILONLY_INSTR I1
        for _g in _n3_cfg.kv_group_configs:
            logger.info(
                "kvoff groupcfg g=%d tokens_per_block=%d tokens_per_chunk=%d "
                "sw_chunks=%s align_chunks=%s anchor_chunks=%s eagle=%s "
                "alignment_tokens=%s tail_only=%s blocks_per_chunk=%d",
                _g.group_idx, _g.tokens_per_block, _g.tokens_per_chunk,
                _g.sliding_window_size_in_chunks, _g.alignment_chunk_count,
                _g.anchor_chunk_count, _g.is_eagle_group, alignment_tokens,
                _tail_only, spec.blocks_per_chunk,
            )
        return _n3_cfg


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Index of the next chunk to offload.
    next_stored_chunk_idx: int = 0
    # Number of offloaded chunks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # time.monotonic() of this request's first deferred offload lookup;
    # None once consumed (observed) or while no lookup is pending.
    deferred_lookup_start_time: float | None = None
    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hashes_per_chunk * len(group_state.offload_keys)
                + group_config.hashes_per_chunk
                - 1,
                None,
                group_config.hashes_per_chunk,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def storable_chunks(
        self,
        group_config: "GroupOffloadConfig",
        group_state: RequestGroupState,
        num_offloadable_tokens: int,
    ) -> int:
        """Number of allocated leading offloaded chunks eligible for store.

        For eagle/MTP groups the volatile trailing chunk of the offloadable
        range is excluded while decoding: the draft-layer KV of the last
        accepted position may be rewritten after spec-token rejection. During
        prefill the trailing chunk is stable (the draft input for a chunk's
        last position is the next prompt token), so it is stored immediately.
        The exclusion must be applied consistently everywhere
        ``next_stored_chunk_idx`` is derived: otherwise the trailing chunk of
        each step is skipped on collection but jumped over by
        ``next_stored_chunk_idx``, so it is never re-considered and a
        permanent hole breaks prefix-reuse lookup.
        """
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
        if group_config.is_eagle_group and is_decoding:
            num_chunks = max(0, num_chunks - 1)
        # DSPARK tail-only: 윈도우/상태 그룹은 종료 전엔 새 storable 없음
        # (next_stored_chunk_idx를 동결해 종료 시 [동결점, N) 전체가 후보로
        # 남고, 도달성 필터가 최종 꼬리만 통과시킨다).
        if (
            self.config.tail_only
            and group_config.sliding_window_size_in_chunks is not None
            and not self.req.is_finished()
        ):
            return group_state.next_stored_chunk_idx
        num_allocated_chunks = (
            len(group_state.block_ids) // self.config.blocks_per_chunk
        )
        return min(num_chunks, num_allocated_chunks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )


def _parse_tier_filter(raw: Any) -> TierFilter:
    """Parse raw kv_transfer_params tier matchers into a TierFilter."""
    if not isinstance(raw, list):
        logger.warning(
            "_parse_tier_filter: expected list, got %s; ignoring",
            type(raw).__name__,
        )
        return TierFilter.ALL
    matchers: list[TierMatcher] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("_parse_tier_filter: entry is not a dict; skipping")
            continue
        medium: Medium | None = None
        locality: Locality | None = None
        raw_medium = entry.get(MATCHER_MEDIUM_KEY)
        if raw_medium is not None:
            try:
                medium = Medium(raw_medium.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown medium %r; skipping entry",
                    raw_medium,
                )
                continue
        raw_locality = entry.get(MATCHER_LOCALITY_KEY)
        if raw_locality is not None:
            try:
                locality = Locality(raw_locality.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown locality %r; skipping entry",
                    raw_locality,
                )
                continue
        matchers.append(TierMatcher(medium=medium, locality=locality))
    if not matchers:
        if not raw:  # input was [] — user explicitly wants nothing
            return TierFilter(matchers=())
        # all entries were invalid — fall back to ALL
        return TierFilter.ALL
    return TierFilter(matchers=tuple(matchers))


def _create_req_context(req: Request) -> ReqContext:
    params = req.kv_transfer_params
    load_filter = TierFilter.ALL
    if params:
        raw = params.get(KV_LOAD_TIERS_KEY)
        if raw is not None:
            load_filter = _parse_tier_filter(raw)
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=params,
        load_tier_filter=load_filter,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.config = SchedulerOffloadConfig.from_spec(
            spec, vllm_config, kv_cache_config
        )
        self.manager: OffloadingManager = spec.get_manager()
        # [load-error-recompute] load job_id -> dst GPU block ids, used to
        # translate worker-reported invalid_block_ids back to the offload
        # keys that must stop being offered (poison; see
        # update_connector_output). Entries are dropped on job completion.
        self._dspark_load_job_dst_blocks: dict[int, set[int]] = {}
        self._connector_stats = OffloadingConnectorStats()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(
            spec, kv_cache_config
        )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # Track loaded chunks to avoid redundant loads.
        self._chunks_being_loaded: set[OffloadKey] | None = (
            set() if vllm_config.cache_config.enable_prefix_caching else None
        )

        # DSPARK restore admission gate: at most this many requests may
        # have fs->staging promotions in flight at once. Concurrent
        # oversized cold restores need more staging blocks than exist and
        # evict each other's completed promotions forever (the whole capped
        # range must be simultaneously staging-resident before
        # _maximal_prefix_lookup can conclude), degrading every restore to
        # a full recompute. Requests whose lookup resolves purely from
        # staging are never gated.
        try:
            self._restore_concurrency: int = max(
                1,
                int(os.environ.get("DSPARK_RESTORE_CONCURRENCY", "1") or 1),
            )
        except ValueError:
            self._restore_concurrency = 1
        # req_ids currently allowed to initiate fs->staging promotions.
        self._promotion_owners: set[ReqId] = set()

        # DSPARK finish-store retry: req_id -> failed prepare_store attempt
        # count for finished requests whose finish-time store allocation
        # failed. Instead of permanently dropping the tail chunks (holes in
        # the parked session), the finished-request cleanup is deferred and
        # prepare_store is re-attempted on subsequent build_connector_meta
        # steps while the freed source GPU blocks remain unreallocated.
        try:
            self._finish_store_retries_cap: int = max(
                0,
                int(
                    os.environ.get("DSPARK_FINISH_STORE_RETRIES", "30") or 30
                ),
            )
        except ValueError:
            self._finish_store_retries_cap = 30
        self._pending_finish_stores: dict[ReqId, int] = {}

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)

    def _maybe_observe_lookup_async_delay(
        self, req_status: RequestOffloadState
    ) -> None:
        start_time = req_status.deferred_lookup_start_time
        if start_time is None:
            return
        req_status.deferred_lookup_start_time = None
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _calc_num_offloadable_tokens(
        self, req_status: RequestOffloadState, num_computed_tokens: int
    ) -> int:
        num = min(num_computed_tokens, req_status.req.num_tokens)
        max_offload_tokens = req_status.max_offload_tokens
        if max_offload_tokens is not None:
            num = min(num, max_offload_tokens)
        if self.config.offload_prompt_only:
            num = min(num, req_status.req.num_prompt_tokens)
        return num

    def _maximal_prefix_lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        req: Request,
        group_config: GroupOffloadConfig,
        start_chunk_idx: int,
    ) -> int | None:
        """Return the number of consecutive offloaded chunks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for local_idx, key in enumerate(keys):
            result = self.manager.lookup(key, req_context)
            match result:
                case LookupResult.HIT:
                    self._events_tracker.record_lookup(
                        req,
                        group_config,
                        start_chunk_idx + local_idx,
                        key,
                    )
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            match self.manager.lookup(keys[idx], req_context):
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # Keep only chunks needed to hit the original request, plus
                # decoded chunks.
                chunks_to_skip = max(
                    0,
                    group_state.num_hit_chunks
                    - group_config.sliding_window_size_in_chunks,
                )
                self.manager.touch(
                    group_state.offload_keys[chunks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing chunk
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                tokens_per_chunk = group_config.tokens_per_chunk
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys) >= req_status.req.num_tokens // tokens_per_chunk
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to a chunk-aligned boundary for this group.
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * tokens_per_chunk
                )
                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )

                # DSPARK reserve-for-SWA: cap full-attention restore at what fits
                # the primary staging tier, leaving room for SWA tail promotions,
                # so an oversized session restores its fitting prefix instead of
                # collapsing to a full recompute (keys loaded stay a subset of
                # what was promoted, so this is fail-safe).
                _cap_fn = getattr(self.manager, "primary_capacity_blocks", None)
                if sliding_window_size_in_chunks is None and _cap_fn is not None:
                    _reserve = int(os.environ.get("DSPARK_SWA_RESERVE", "15"))
                    _cap_blocks = _cap_fn() - _reserve
                    if _cap_blocks >= 1:
                        _sc = num_computed_tokens // tokens_per_chunk
                        max_hit_size_tokens = min(
                            max_hit_size_tokens,
                            (_sc + _cap_blocks) * tokens_per_chunk,
                        )

                # For eagle groups, query one extra chunk that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_chunks is not None:
                    query_max = min(
                        max_hit_size_tokens + tokens_per_chunk,
                        len(offload_keys) * tokens_per_chunk,
                    )

                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if num_hit_chunks is not None:  # NODE3_TAILONLY_INSTR I3
                    logger.info(
                        "kvoff lookup req=%s g=%d start=%d end=%d hits=%s "
                        "maxhit=%d computed=%d",
                        req_status.req.request_id, group_idx, start_chunk_idx,
                        num_chunks, num_hit_chunks, max_hit_size_tokens,
                        num_computed_tokens,
                    )
                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        # Pop the volatile trailing draft chunk only when it
                        # was actually part of the confirmed hit run. The
                        # extra chunk (one past max_hit_size_tokens) was
                        # queried only when query_max was raised above
                        # max_hit_size_tokens; the run reaches it iff
                        # num_hit_chunks == len(offload_keys). If the extra
                        # chunk was queried but MISSED, the run already ends
                        # exactly at the boundary, and popping again would
                        # tighten the boundary one window chunk below a
                        # g0-aligned interior cap, cascading the other
                        # groups downward on every reconciliation pass
                        # (walkdown to full recompute). At the natural
                        # end-of-request the extra chunk is never queried
                        # (query_max is clamped to the key range), so the
                        # unconditional pop semantics are preserved there.
                        _extra_chunk_queried = query_max > max_hit_size_tokens
                        if (
                            not _extra_chunk_queried
                            or num_hit_chunks >= len(offload_keys)
                        ):
                            num_hit_chunks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        tokens_per_chunk * (start_chunk_idx + num_hit_chunks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_chunks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # Possibly delay the request if any hit chunk is already being loaded.
        if self._chunks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                tokens_per_chunk = group_config.tokens_per_chunk
                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )
                offload_keys = group_state.offload_keys
                num_chunks = cdiv(
                    num_computed_tokens + num_hit_tokens, tokens_per_chunk
                )
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]
                if sliding_window_size_in_chunks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_chunks:]
                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Hit chunks are being loaded, so delay the request.
                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            lookup_start = time.monotonic()
            # DSPARK restore admission gate: this request may initiate
            # fs->staging promotions only if it already owns a slot or one
            # is free. A gated request that would need a promotion is
            # deferred by the manager exactly like the existing
            # HIT_PENDING/RETRY path (lookup returns None; the scheduler
            # retries the request next step). Lookups resolving purely
            # from staging never consult the flag's deny path.
            _owners = self._promotion_owners
            if _owners:
                # Lazily drop owners whose request state is gone.
                _owners.intersection_update(self._req_status)
            _req_context = req_status.req_context
            _req_context.dspark_allow_promotion = (
                request.request_id in _owners
                or len(_owners) < self._restore_concurrency
            )
            _req_context.dspark_promotion_initiated = False
            num_hit_tokens = self._lookup(req_status)
            if getattr(_req_context, "dspark_promotion_initiated", False):
                _owners.add(request.request_id)
            if num_hit_tokens is not None:
                # Definitive result (hits get pinned via prepare_load this
                # step, or a definitive miss): release the slot.
                _owners.discard(request.request_id)
            self._connector_stats.observe_histogram(
                _ConnectorMetricName.LOOKUP_SYNC_DELAY,
                time.monotonic() - lookup_start,
            )
            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
            else:
                self._maybe_observe_lookup_async_delay(req_status)
        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block
            tokens_per_chunk = group_config.tokens_per_chunk
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, tokens_per_block)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * tokens_per_block
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_chunks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_chunks
                    * self.config.blocks_per_chunk
                    + 1
                )

            num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)
            assert len(offload_keys) >= num_chunks
            if num_pending_gpu_blocks:
                start_chunk_idx = (
                    num_locally_computed_gpu_blocks // self.config.blocks_per_chunk
                )
                keys_to_load.extend(offload_keys[start_chunk_idx:num_chunks])

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit chunks for block-level policy; for
            # request-level, next_stored_chunk_idx stays at 0 so all
            # chunks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_chunk_idx = num_chunks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # [load-error-recompute]
        self._dspark_load_job_dst_blocks[load_job_id] = {
            int(b) for b in dst_block_ids
        }
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_chunk_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            blocks_per_chunk = self.config.blocks_per_chunk
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_chunk_idx * blocks_per_chunk
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        blocks_per_chunk = self.config.blocks_per_chunk
        store_jobs: dict[int, TransferJob] = {}
        for req_id in chain(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.finished_req_ids or (),
            # DSPARK finish-store retry: finished requests whose finish-time
            # prepare_store failed. Snapshot: entries resolve inside the loop.
            tuple(self._pending_finish_stores),
        ):
            req_status = self._req_status.get(req_id)
            if req_status is None:
                # A pending finish-store whose request state vanished
                # (reset_cache) cannot be retried.
                self._pending_finish_stores.pop(req_id, None)
                continue
            req = req_status.req

            if req.status is RequestStatus.FINISHED_ABORTED:
                num_tokens_after_batch = req.num_computed_tokens
            elif req.is_finished():
                num_tokens_after_batch = req.num_tokens
            else:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens

            num_offloadable_tokens = self._calc_num_offloadable_tokens(
                req_status, num_tokens_after_batch
            )

            if req_id in self._pending_finish_stores:
                # DSPARK finish-store retry: the request's GPU blocks were
                # freed at finish time; their content stays valid only until
                # the KV cache manager reallocates them. _update_req_states
                # already zeroed reallocated sliding-window entries (they
                # become skipped holes, same as at finish time); a
                # reallocated full-attention block invalidates the retry, so
                # give up rather than store overwritten bytes.
                _stale = False
                _cur_alloc = self._current_batch_allocated_block_ids
                if _cur_alloc:
                    for _gcfg, _gstate in zip(
                        self.config.kv_group_configs, req_status.group_states
                    ):
                        _nch = req_status.storable_chunks(
                            _gcfg, _gstate, num_offloadable_tokens
                        )
                        _start = (
                            _gstate.next_stored_chunk_idx * blocks_per_chunk
                        )
                        for _bid in _gstate.block_ids[
                            _start : _nch * blocks_per_chunk
                        ]:
                            if _bid and _bid in _cur_alloc:
                                _stale = True
                                break
                        if _stale:
                            break
                if _stale:
                    del self._pending_finish_stores[req_id]
                    logger.warning(
                        "Request %s: abandoning finish-time store retry; "
                        "source GPU blocks were reallocated",
                        req_id,
                    )
                    self._finalize_finished_req(req_id, req_status)
                    continue

            # Filter out chunks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )

                start_chunk_idx = group_state.next_stored_chunk_idx
                if num_chunks <= start_chunk_idx:
                    continue
                offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
                # For each chunk, take the last corresponding GPU block. For
                # blocks_per_chunk=3 and GPU block IDs 1 5 6 7 2 4 9 3 8,
                # this selects GPU blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_chunk_idx * blocks_per_chunk
                    + blocks_per_chunk
                    - 1 : num_chunks * blocks_per_chunk : blocks_per_chunk
                ]
                assert len(offload_keys) == len(offload_block_ids)

                _n3_hole = _n3_unreach = _n3_kept = 0  # NODE3_TAILONLY_INSTR I2
                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        _n3_hole += 1
                        continue
                    # DSPARK compact: skip chunks with interior zeroed blocks
                    # (stale SWA entries). Partial chunks violate the
                    # GPULoadStoreSpec contiguity contract and desync the
                    # scheduler chunk allocation from the worker block math.
                    _ck = start_chunk_idx + key_idx
                    _cb = group_state.block_ids[
                        _ck * blocks_per_chunk : (_ck + 1) * blocks_per_chunk
                    ]
                    if len(_cb) < blocks_per_chunk or 0 in _cb:
                        _n3_hole += 1
                        continue
                    # Skip SWA chunks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing chunks queried by _sliding_window_lookup are
                    # reachable. EAGLE/MTP requires one additional chunk that
                    # lookup later drops as its volatile draft tail.
                    abs_chunk_idx = start_chunk_idx + key_idx
                    if self.config.anchor_now:  # NODE3_TAILONLY_FIX F4
                        _reach = is_store_anchor_now_swa_chunk(
                            abs_chunk_idx,
                            req.num_prompt_tokens // group_config.tokens_per_chunk,
                            (
                                self.config.anchor_tokens_override
                                // group_config.tokens_per_chunk
                                if self.config.anchor_tokens_override
                                else group_config.anchor_chunk_count
                            ),
                            group_config.sliding_window_size_in_chunks,
                            group_config.is_eagle_group,
                            checkpoint_chunks=(
                                self.config.anchor_ckpt_tokens
                                // group_config.tokens_per_chunk
                            ),
                        )
                    else:
                        _reach = is_store_reachable_swa_chunk(
                            abs_chunk_idx,
                            num_chunks,
                            group_config.alignment_chunk_count,
                            group_config.sliding_window_size_in_chunks,
                            group_config.is_eagle_group,
                            tail_only=self.config.tail_only,
                            anchor_chunk_count=group_config.anchor_chunk_count,
                        )
                    if not _reach:
                        _n3_unreach += 1
                        continue
                    _n3_kept += 1
                    new_offload_keys.append(offload_key)
                if req.is_finished():  # NODE3_TAILONLY_INSTR I2 log
                    logger.info(
                        "kvoff swa-finish req=%s g=%d sw=%s next=%d nchunks=%d "
                        "cand=%d hole=%d unreach=%d kept=%d ntok=%d nblk=%d",
                        req_id, group_config.group_idx,
                        group_config.sliding_window_size_in_chunks,
                        start_chunk_idx, num_chunks, len(offload_keys),
                        _n3_hole, _n3_unreach, _n3_kept,
                        num_offloadable_tokens, len(group_state.block_ids),
                    )

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                if req_id in self._pending_finish_stores:
                    # Nothing storable remains; run the deferred cleanup.
                    del self._pending_finish_stores[req_id]
                    self._finalize_finished_req(req_id, req_status)
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                if req.is_finished():
                    # DSPARK finish-store retry: dropping the store here
                    # would permanently lose the request's tail chunks.
                    # Keep req_status alive, defer the on_request_finished
                    # signal (build_connector_meta skips this req_id), and
                    # re-attempt on subsequent steps: has_pending_push_work
                    # keeps the engine stepping, and the reallocation guard
                    # above aborts if the freed GPU blocks get reused.
                    _attempts = self._pending_finish_stores.get(req_id, 0) + 1
                    if _attempts <= self._finish_store_retries_cap:
                        self._pending_finish_stores[req_id] = _attempts
                        logger.warning(
                            "Request %s: cannot store finish-time chunks "
                            "(attempt %d/%d); will retry",
                            req_id,
                            _attempts,
                            self._finish_store_retries_cap,
                        )
                        continue
                    self._pending_finish_stores.pop(req_id, None)
                    logger.warning(
                        "Request %s: giving up finish-time store after %d "
                        "attempts; tail chunks are dropped",
                        req_id,
                        _attempts,
                    )
                    self._finalize_finished_req(req_id, req_status)
                    continue
                logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                if req_id in self._pending_finish_stores:
                    # Everything already stored; run the deferred cleanup.
                    del self._pending_finish_stores[req_id]
                    self._finalize_finished_req(req_id, req_status)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )
                start_chunk_idx = group_state.next_stored_chunk_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_chunk_idx:num_chunks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    chunk_idx = start_chunk_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, chunk_idx, offload_key
                    )

                    gpu_block_idx = chunk_idx * blocks_per_chunk
                    for i in range(blocks_per_chunk):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx, num_chunks
                )

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s chunks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

            if req.is_finished():
                # Register non-sliding-window blocks for flush detection.
                for bid in non_sliding_window_block_ids:
                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
                    if bid in self._current_batch_allocated_block_ids:
                        self._current_batch_jobs_to_flush.add(job_id)
                if req_id in self._pending_finish_stores:
                    # DSPARK finish-store retry succeeded: run the deferred
                    # finished-request cleanup. transfer_jobs now holds the
                    # store job, so req_status survives until
                    # update_connector_output completes it, exactly like an
                    # on-time finish store.
                    del self._pending_finish_stores[req_id]
                    self._finalize_finished_req(req_id, req_status)

        return store_jobs

    def _finalize_finished_req(
        self, req_id: ReqId, req_status: RequestOffloadState
    ) -> None:
        """Deferred finished-request cleanup (DSPARK finish-store retry).

        Mirrors the finished_req_ids cleanup in build_connector_meta():
        signal the manager exactly once, then drop req_status unless
        transfer jobs are still in flight (update_connector_output deletes
        it when the last one completes).
        """
        req_status.finished_signaled = True
        self.manager.on_request_finished(req_status.req_context)
        if not req_status.transfer_jobs:
            del self._req_status[req_id]

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )
        self.manager.on_schedule_end(schedule_end_context)

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=self._build_store_jobs(scheduler_output),
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )

        # All prepare_store calls for finished requests have been issued.
        # Signal on_request_finished and clean up state where possible.
        for req_id in scheduler_output.finished_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            if req_id in self._pending_finish_stores:
                # DSPARK finish-store retry: keep req_status alive and defer
                # the on_request_finished signal until the retry resolves
                # (_build_store_jobs runs the cleanup then). Signaling now
                # would let the tiering manager finalize and drop its
                # request state, making the retried prepare_store fail.
                continue
            req_status.finished_signaled = True
            self.manager.on_request_finished(req_status.req_context)
            if not req_status.transfer_jobs:
                del self._req_status[req_id]
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        return (
            bool(self._jobs)
            or bool(self._pending_finish_stores)
            or self.manager.has_pending_work()
        )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        # [load-error-recompute] Worker-reported KV load failures: the core
        # scheduler recomputes the affected GPU blocks, but the offload
        # metadata still claims the chunks are stored, so an unchanged
        # lookup would re-offer the same failing chunks and the request
        # would retry forever. Poison the keys of every in-flight load job
        # that overlaps the invalid GPU blocks: lookup() then MISSes them
        # (patch_41) and fs_paths are no longer attached. Job granularity
        # over-poisons; the only cost is extra recompute.
        invalid_block_ids = getattr(connector_output, "invalid_block_ids", None)
        if invalid_block_ids and self._dspark_load_job_dst_blocks:
            for job_id, dst_blocks in list(
                self._dspark_load_job_dst_blocks.items()
            ):
                overlap = invalid_block_ids.intersection(dst_blocks)
                if not overlap:
                    continue
                job_status = self._jobs.get(job_id)
                if job_status is not None and not job_status.is_store:
                    poisoned = getattr(
                        self.manager, "_dspark_poisoned_keys", None
                    )
                    if poisoned is None:
                        poisoned = set()
                        self.manager._dspark_poisoned_keys = poisoned
                    poisoned.update(job_status.keys)
                    promoted = getattr(
                        self.manager, "_fs_promoted_keys", None
                    )
                    if promoted is not None:
                        for k in job_status.keys:
                            promoted.pop(k, None)
                    logger.error(
                        "[load-error-recompute] load job %d overlaps %d "
                        "invalid GPU blocks; poisoned %d offloaded chunks "
                        "(lookup will MISS them; recompute re-fills)",
                        job_id,
                        len(overlap),
                        len(job_status.keys),
                    )
                # poison at most once per job
                del self._dspark_load_job_dst_blocks[job_id]
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.non_sliding_window_block_ids
                    )

            self._dspark_load_job_dst_blocks.pop(job_id, None)
            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats: OffloadingConnectorStats | None = None
        if not self._connector_stats.is_empty():
            stats = self._connector_stats
            self._connector_stats = OffloadingConnectorStats()

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_status = self._req_status.get(request.request_id)

        # DSPARK restore admission gate: a finished/aborted request can no
        # longer drive promotions; release its slot.
        self._promotion_owners.discard(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            return False, None

        self._maybe_observe_lookup_async_delay(req_status)

        # Update offload keys with final block hash so _build_store_jobs can
        # create store jobs for the last block(s) on the next schedule step.
        req_status.update_offload_keys()

        # Keep req_status alive: _build_store_jobs will process finished_req_ids
        # on the next step and handle cleanup after creating store jobs.
        # Register non_sliding_window_block_ids so future block reuse triggers
        # a flush via _block_id_to_pending_jobs.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored chunks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                if not status.finished_signaled:
                    self.manager.on_request_finished(status.req_context)
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from chunk 0.
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_chunk_idx = 0
            status.transfer_jobs.clear()

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._dspark_load_job_dst_blocks.clear()
        self._block_id_to_pending_jobs.clear()
        self._promotion_owners.clear()
        self._pending_finish_stores.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
