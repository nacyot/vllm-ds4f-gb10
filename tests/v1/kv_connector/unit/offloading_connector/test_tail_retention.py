# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK window-snapshot policies (2026-08-30 plan):

* ``DSPARK_TAIL_ONLY=2`` (anchor-now): sliding-window groups store only the
  w(+e) chunks right before the prompt's last full-attention-aligned boundary.
* ``DSPARK_TAIL_KEEP=K``: after a store completes, snapshots at aligned
  boundaries older than the chain's K most recent are deleted from the
  secondary tiers via ``manager.dspark_delete_keys``.
* EAGLE extra-chunk pop (fork 09649b4be): when the one-past-boundary draft
  chunk was queried but missed, the hit run already ends at the boundary and
  must not be popped again (otherwise the boundary walks down 4096 tokens per
  reconciliation pass until full recompute).
"""

from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
    is_store_anchor_now_swa_chunk,
    superseded_anchor_chunk_ranges,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    make_offload_key,
)

# ---------------------------------------------------------------------------
# Pure predicates
# ---------------------------------------------------------------------------


def test_superseded_anchor_ranges_keep_two():
    # geometry: a=4 SWA chunks per aligned segment, window 2 chunks, no eagle
    # prompt = 20 chunks -> prompt anchor 20; keep {20, 16}; delete 12, 8, 4
    assert superseded_anchor_chunk_ranges(20, 4, 2, False, keep=2) == [
        (10, 12),
        (6, 8),
        (2, 4),
    ]
    # eagle adds one chunk to the tail
    assert superseded_anchor_chunk_ranges(20, 4, 2, True, keep=2) == [
        (9, 12),
        (5, 8),
        (1, 4),
    ]


def test_superseded_anchor_ranges_edges():
    # keep=1 keeps only the prompt anchor itself
    assert superseded_anchor_chunk_ranges(20, 4, 2, False, keep=1) == [
        (14, 16),
        (10, 12),
        (6, 8),
        (2, 4),
    ]
    # unaligned prompt: anchor rounds down (23 -> 20)
    assert superseded_anchor_chunk_ranges(23, 4, 2, False, keep=1)[0] == (14, 16)
    # boundary 0 never holds a snapshot; a tail longer than the first
    # boundary is clipped at 0, and the kept anchor 8's snapshot [2, 8)
    # clips boundary 4's range to [0, 2)
    assert superseded_anchor_chunk_ranges(8, 4, 6, False, keep=1) == [(0, 2)]
    # disabled / not applicable
    assert superseded_anchor_chunk_ranges(20, 4, 2, False, keep=0) == []
    assert superseded_anchor_chunk_ranges(20, None, 2, False, keep=2) == []
    assert superseded_anchor_chunk_ranges(20, 4, None, False, keep=2) == []
    # sweep cap
    assert len(superseded_anchor_chunk_ranges(4000, 4, 2, False, 1, 3)) == 3
    # window wider than a segment (tail 6 > a 4): the kept anchor 20's
    # snapshot is [14, 20); boundary 16's range [10, 16) must be clipped to
    # [10, 14) and boundary 12's [6, 12) is untouched
    # (older boundaries' ranges overlap each other, which is harmless)
    assert superseded_anchor_chunk_ranges(20, 4, 6, False, keep=1) == [
        (10, 14),
        (6, 12),
        (2, 8),
        (0, 4),
    ]
    # keep=2 with tail 6: kept 20 and 16 -> lowest kept snapshot [10, 16)
    assert superseded_anchor_chunk_ranges(20, 4, 6, False, keep=2) == [
        (6, 10),
        (2, 8),
        (0, 4),
    ]


def test_anchor_now_predicate_prompt_anchor_only():
    # a=4, window 2, prompt 22 chunks -> anchor 20 -> chunks 18,19 pass
    passing = [
        i for i in range(24) if is_store_anchor_now_swa_chunk(i, 22, 4, 2, False)
    ]
    assert passing == [18, 19]
    # eagle: one more trailing chunk (17,18,19)
    passing = [i for i in range(24) if is_store_anchor_now_swa_chunk(i, 22, 4, 2, True)]
    assert passing == [17, 18, 19]
    # checkpoint every 8 chunks adds boundaries 8 and 16 (aligned to a)
    passing = [
        i
        for i in range(24)
        if is_store_anchor_now_swa_chunk(i, 22, 4, 2, False, checkpoint_chunks=8)
    ]
    assert passing == [6, 7, 14, 15, 18, 19, 22, 23]


# ---------------------------------------------------------------------------
# Scheduler integration (request_runner)
# ---------------------------------------------------------------------------


def _hybrid_groups(full_bs: int = 16, swa_bs: int = 4, window: int = 8):
    return [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=full_bs, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=swa_bs,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=window,
            ),
        ),
    ]


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_anchor_now_stores_only_prompt_anchor_tail(
    request_runner, monkeypatch, async_scheduling: bool
):
    """TAIL_ONLY=2: 64-token prompt = 4 full chunks = 16 SWA chunks; the SWA
    group stores only chunks 14,15 (the window before boundary 16). Compare
    test_swa_alignment_skip (TAIL_ONLY=0) which stores 2 chunks per segment."""
    monkeypatch.setenv("DSPARK_TAIL_ONLY", "2")
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=200,
        async_scheduling=async_scheduling,
        kv_cache_groups=_hybrid_groups(),
    )
    cfg = runner.connector_scheduler.config
    assert cfg.anchor_now and not cfg.tail_only
    assert cfg.kv_group_configs[1].anchor_chunk_count == 4
    assert cfg.kv_group_configs[1].sliding_window_size_in_chunks == 2

    runner.new_request(token_ids=[0] * 64)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((0, 0), (0, 1), (0, 2), (0, 3), (1, 14), (1, 15)),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_tail_keep_deletes_superseded_snapshots(
    request_runner, monkeypatch, async_scheduling: bool
):
    """TAIL_KEEP=1 with a 64-token prompt (prompt anchor = SWA chunk 16):
    after the store completes, the manager is asked to delete the SWA keys
    of boundaries 12, 8, 4 (chunks 10-11, 6-7, 2-3). Full-attention keys and
    the kept anchor's keys are never offered for deletion."""
    monkeypatch.setenv("DSPARK_TAIL_ONLY", "2")
    monkeypatch.setenv("DSPARK_TAIL_KEEP", "1")
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=200,
        async_scheduling=async_scheduling,
        kv_cache_groups=_hybrid_groups(),
    )
    sched = runner.connector_scheduler
    assert sched.config.tail_keep == 1

    deleted: list[list] = []
    # the runner's manager is MagicMock(spec=OffloadingManager); the retention
    # hook is duck-typed (getattr), so the mock must expose it explicitly
    runner.manager.dspark_delete_keys = MagicMock(
        side_effect=lambda keys, ctx=None: (
            deleted.append(list(keys)) or (len(keys), 0)
        )
    )

    runner.new_request(token_ids=[0] * 64)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    # offload keys are derived from the request's block hashes once the
    # prefill has been scheduled; read them after the first step
    req_id = next(iter(sched._req_status))
    swa_keys = list(sched._req_status[req_id].group_states[1].offload_keys)
    full_keys = set(sched._req_status[req_id].group_states[0].offload_keys)
    assert len(swa_keys) == 16
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((0, 0), (0, 1), (0, 2), (0, 3), (1, 14), (1, 15)),
    )

    assert deleted, "store completion must trigger retention"
    offered = set()
    for batch in deleted:
        offered.update(batch)
    expected = set(swa_keys[10:12] + swa_keys[6:8] + swa_keys[2:4])
    assert offered == expected
    assert not (offered & full_keys)
    assert not (offered & set(swa_keys[14:16]))


def test_prune_gated_on_anchor_snapshot_landing(request_runner, monkeypatch):
    """A store completion that does not carry the prompt-anchor snapshot
    (e.g. a full-attention-only job) must not prune older anchors."""
    monkeypatch.setenv("DSPARK_TAIL_ONLY", "2")
    monkeypatch.setenv("DSPARK_TAIL_KEEP", "1")
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=200,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    sched = runner.connector_scheduler
    calls: list = []
    runner.manager.dspark_delete_keys = MagicMock(
        side_effect=lambda keys, ctx=None: (calls.append(list(keys)) or (len(keys), 0))
    )
    runner.new_request(token_ids=[0] * 64)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    req_id = next(iter(sched._req_status))
    req_status = sched._req_status[req_id]
    swa_keys = list(req_status.group_states[1].offload_keys)
    full_keys = set(req_status.group_states[0].offload_keys)
    # direct hook probes: a job with only g0 keys -> no prune; a job carrying
    # the anchor snapshot (chunks 14,15) -> prune
    sched._prune_superseded_anchors(req_status, set(full_keys))
    assert calls == []
    sched._prune_superseded_anchors(req_status, {swa_keys[14], swa_keys[15]})
    assert calls and set(calls[0]) == set(
        swa_keys[10:12] + swa_keys[6:8] + swa_keys[2:4]
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((0, 0), (0, 1), (0, 2), (0, 3), (1, 14), (1, 15)),
    )


def test_tail_keep_disabled_never_calls_delete(request_runner, monkeypatch):
    monkeypatch.setenv("DSPARK_TAIL_ONLY", "2")
    monkeypatch.delenv("DSPARK_TAIL_KEEP", raising=False)
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=200,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    runner.manager.dspark_delete_keys = MagicMock(return_value=(0, 0))
    runner.new_request(token_ids=[0] * 64)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((0, 0), (0, 1), (0, 2), (0, 3), (1, 14), (1, 15)),
    )
    assert not runner.manager.dspark_delete_keys.called


def test_tail_keep_skips_keys_being_loaded(request_runner, monkeypatch):
    """A superseded key that is currently being loaded by another request is
    not offered for deletion (the load would fail and force a recompute)."""
    monkeypatch.setenv("DSPARK_TAIL_ONLY", "2")
    monkeypatch.setenv("DSPARK_TAIL_KEEP", "1")
    runner = request_runner(
        block_size=4,
        num_gpu_blocks=200,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    sched = runner.connector_scheduler
    deleted: list = []
    runner.manager.dspark_delete_keys = MagicMock(
        side_effect=lambda keys, ctx=None: (deleted.extend(keys) or (len(keys), 0))
    )
    runner.new_request(token_ids=[0] * 64)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[0])
    req_id = next(iter(sched._req_status))
    req_status = sched._req_status[req_id]
    swa_keys = list(req_status.group_states[1].offload_keys)
    assert len(swa_keys) == 16
    if sched._chunks_being_loaded is None:
        sched._chunks_being_loaded = set()
    sched._chunks_being_loaded.add(swa_keys[10])
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=((0, 0), (0, 1), (0, 2), (0, 3), (1, 14), (1, 15)),
    )
    assert swa_keys[10] not in deleted
    assert swa_keys[11] in deleted


# ---------------------------------------------------------------------------
# EAGLE extra-chunk pop regression (fork 09649b4be)
# ---------------------------------------------------------------------------


def _make_req_status(
    scheduler: OffloadingConnectorScheduler,
    *,
    num_tokens: int,
    offload_keys_per_group: list[list[int]],
) -> RequestOffloadState:
    req = MagicMock()
    req.request_id = "test-req"
    req.num_tokens = num_tokens
    req.num_prompt_tokens = num_tokens
    req.kv_transfer_params = None
    num_hash_blocks = max(
        len(hashes) * scheduler.config.kv_group_configs[idx].hashes_per_chunk
        for idx, hashes in enumerate(offload_keys_per_group)
    )
    req.block_hashes = [BlockHash(str(i).encode()) for i in range(num_hash_blocks)]
    req.all_token_ids = list(range(num_tokens))
    req.lora_request = None
    state = RequestOffloadState(
        config=scheduler.config,
        req=req,
        req_context=ReqContext(req_id="test-req"),
        offloading_context=RequestOffloadingContext(policy=OffloadPolicy.BLOCK_LEVEL),
    )
    for idx, (gs, hashes) in enumerate(zip(state.group_states, offload_keys_per_group)):
        gidx = scheduler.config.kv_group_configs[idx].group_idx
        gs.offload_keys = [make_offload_key(str(h).encode(), gidx) for h in hashes]
    return state


def test_eagle_sw_extra_chunk_missed_does_not_pop(request_runner):
    """SW eagle group, W=2, keys [1,2,3,4], hits {1,2,3}, 4 misses.

    num_tokens=13 -> max_hit=12 (3 chunks); the extra draft chunk (4) is
    queried (query_max=16) but MISSES, so the confirmed run [1,2,3] already
    ends exactly at the boundary. The pop must be skipped: 3 chunks = 12
    tokens. Before 09649b4be this popped to 8 and, at an interior cap
    boundary, cascaded the other groups down every reconciliation pass."""
    block_size = 4
    groups = [
        KVCacheGroupSpec(
            ["layer0"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=2 * block_size,
            ),
            is_eagle_group=True,
        ),
    ]
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=False,
        kv_cache_groups=groups,
    )
    runner.manager.lookup.side_effect = lambda key, req_context: (
        LookupResult.HIT
        if int(get_offload_block_hash(key).decode()) in {1, 2, 3}
        else LookupResult.MISS
    )
    sched = runner.connector_scheduler
    req_status = _make_req_status(
        sched, num_tokens=13, offload_keys_per_group=[[1, 2, 3, 4]]
    )
    assert sched._lookup(req_status) == 12


def test_eagle_sw_extra_chunk_hit_pops_once(request_runner):
    """Same geometry, but the extra chunk (4) HITS: the run reaches the end
    of the queried keys, the volatile draft chunk is popped exactly once ->
    still 12 tokens (4 confirmed chunks minus 1, capped by max_hit=12)."""
    block_size = 4
    groups = [
        KVCacheGroupSpec(
            ["layer0"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=2 * block_size,
            ),
            is_eagle_group=True,
        ),
    ]
    runner = request_runner(
        block_size=block_size,
        num_gpu_blocks=100,
        async_scheduling=False,
        kv_cache_groups=groups,
    )
    runner.manager.lookup.side_effect = lambda key, req_context: (
        LookupResult.HIT
        if int(get_offload_block_hash(key).decode()) in {1, 2, 3, 4}
        else LookupResult.MISS
    )
    sched = runner.connector_scheduler
    req_status = _make_req_status(
        sched, num_tokens=13, offload_keys_per_group=[[1, 2, 3, 4]]
    )
    assert sched._lookup(req_status) == 12
