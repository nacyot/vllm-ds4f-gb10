# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict, deque
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.cpu.slot_layout import SlotLayout


class _SlotPool:
    """The blocks of one row-size class: a fresh-block cursor, a free list and
    the CachePolicy that decides which of its blocks to evict."""

    def __init__(self, first_block: int, num_blocks: int, policy: CachePolicy):
        self.first_block = first_block
        self.num_blocks = num_blocks
        self.policy = policy
        self.num_allocated_blocks = 0
        self.free_list: list[int] = []
        # Blocks in the cache that are evictable, i.e. ref_cnt 0.
        self.num_evictable_cache_blocks = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_list) + self.num_blocks - self.num_allocated_blocks

    @property
    def num_used_blocks(self) -> int:
        return (
            self.num_allocated_blocks
            - len(self.free_list)
            - self.num_evictable_cache_blocks
        )

    def allocate(self, num: int) -> list[BlockStatus]:
        num_fresh = min(num, self.num_blocks - self.num_allocated_blocks)
        num_reused = num - num_fresh
        assert len(self.free_list) >= num_reused

        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self.first_block + self.num_allocated_blocks))
            self.num_allocated_blocks += 1
        for _ in range(num_reused):
            blocks.append(BlockStatus(self.free_list.pop()))
        return blocks

    def free(self, block: BlockStatus) -> None:
        self.free_list.append(block.block_id)

    def reset(self) -> None:
        self.policy.clear()
        self.num_evictable_cache_blocks = 0
        self.free_list.clear()
        self.num_allocated_blocks = 0


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.

    Blocks are pooled per row-size class of the ``slot_layout`` (one pool
    without slabs): a key lives in the pool of its KV cache group's class and
    only evicts blocks of that pool.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        slot_layout: SlotLayout | None = None,
    ):
        self.medium: Medium = Medium.CPU
        if slot_layout is None:
            slot_layout = SlotLayout.uniform(num_blocks, 1)
        assert slot_layout.num_blocks == num_blocks
        self._layout = slot_layout
        self._num_blocks: int = num_blocks
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._pools: list[_SlotPool] = [
            _SlotPool(
                c.first_block, c.num_blocks, policy_cls(cache_capacity=c.num_blocks)
            )
            for c in slot_layout.classes
        ]
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- block pool ---

    @property
    def _policy(self) -> CachePolicy:
        """The cache policy of the (first) pool; kept for tests and tooling."""
        return self._pools[0].policy

    @property
    def _num_evictable_cache_blocks(self) -> int:
        return sum(pool.num_evictable_cache_blocks for pool in self._pools)

    def _pool_of(self, key: OffloadKey) -> _SlotPool:
        if len(self._pools) == 1:
            return self._pools[0]
        return self._pools[self._layout.class_of_group(get_offload_group_idx(key))]

    def _get_block(self, key: OffloadKey) -> BlockStatus | None:
        return self._pool_of(key).policy.get(key)

    def _get_num_free_blocks(self) -> int:
        return sum(pool.num_free_blocks for pool in self._pools)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    def _record_accesses(self, keys: Collection[OffloadKey]) -> None:
        """Record an offer without evicting its tracked candidates."""
        assert self.counts is not None
        protected: set[OffloadKey] = set()
        for key in keys:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
                protected.add(key)

        num_unprotected = len(self.counts) - len(protected)
        for key in keys:
            if key in self.counts:
                continue
            if len(self.counts) >= self.max_tracker_size:
                if num_unprotected == 0:
                    continue
                self.counts.popitem(last=False)
                num_unprotected -= 1
            self.counts[key] = 1

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        block = self._get_block(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        for key in keys:
            pool = self._pool_of(key)
            block = pool.policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                pool.policy.mark_non_evictable(key)
                pool.num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert pool.num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if len(self._pools) == 1:
            self._pools[0].policy.touch(keys, req_context)
            return
        by_pool: dict[int, list[OffloadKey]] = {}
        for key in keys:
            by_pool.setdefault(id(self._pool_of(key)), []).append(key)
        for pool in self._pools:
            pool_keys = by_pool.get(id(pool))
            if pool_keys:
                pool.policy.touch(pool_keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            pool = self._pool_of(key)
            block = pool.policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                pool.num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                pool.policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            num_keys = len(keys)
            self._record_accesses(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._get_block(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))

        # Blocks from the original input are excluded from eviction candidates:
        # a block that was already stored must remain in the cache after this call.
        protected = set(keys)
        # Per pool: how many of its blocks the new keys need, and how many it
        # must evict for that. Every pool is checked before any pool evicts,
        # so a batch never leaves half its pools evicted and then fails.
        num_needed: dict[int, int] = {}
        for key in keys_to_store:
            pool_idx = id(self._pool_of(key))
            num_needed[pool_idx] = num_needed.get(pool_idx, 0) + 1
        evictions: list[tuple[_SlotPool, int]] = []
        for pool in self._pools:
            needed = num_needed.get(id(pool), 0)
            num_blocks_to_evict = needed - pool.num_free_blocks
            if num_blocks_to_evict <= 0:
                continue
            if num_blocks_to_evict > pool.num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            evictions.append((pool, num_blocks_to_evict))

        to_evict: list[OffloadKey] = []
        for pool, num_blocks_to_evict in evictions:
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.
            evicted = pool.policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                if to_evict and self.events is not None:
                    # Earlier pools already dropped their blocks; report them.
                    self.events.append(
                        OffloadingEvent(keys=to_evict, medium=self.medium, removed=True)
                    )
                return None

            # cache-policy removes only idle blocks.
            pool.num_evictable_cache_blocks -= len(evicted)
            assert pool.num_evictable_cache_blocks >= 0

            for key, block in evicted:
                pool.free(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks: list[BlockStatus] = []
        if len(self._pools) == 1:
            blocks = self._pools[0].allocate(len(keys_to_store))
        else:
            allocated: dict[int, deque[BlockStatus]] = {
                id(pool): deque(pool.allocate(num_needed.get(id(pool), 0)))
                for pool in self._pools
            }
            for key in keys_to_store:
                blocks.append(allocated[id(self._pool_of(key))].popleft())
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._pool_of(key).policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                pool = self._pool_of(key)
                block = pool.policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    pool.num_evictable_cache_blocks += 1
                    pool.policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                pool = self._pool_of(key)
                block = pool.policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    pool.policy.remove(key)
                    pool.free(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        for pool in self._pools:
            pool.reset()
        self._num_write_pending_blocks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = sum(pool.num_used_blocks for pool in self._pools)
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)
        if len(self._pools) > 1:
            for slot_class, pool in zip(self._layout.classes, self._pools):
                # Blocks holding data (pinned or evictable), unlike the usage
                # gauge above, which counts only pinned blocks.
                held = pool.num_allocated_blocks - len(pool.free_list)
                stats.set_gauge(
                    CPUOffloadingMetrics.CPU_SLAB_USAGE_PERC,
                    held / pool.num_blocks if pool.num_blocks > 0 else 0.0,
                    (str(slot_class.row_bytes),),
                )

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
