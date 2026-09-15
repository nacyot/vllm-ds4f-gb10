# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the DSV41_INDEXER_TP_SPLIT prefill row split.

With the split on, each TP rank scores a contiguous slice of a prefill chunk's
indexer rows and one all-gather rebuilds the full top-k block. These tests swap
the scoring kernels for a stand-in that tags every row with its global index,
so they cover what the split itself owns: the per-rank bounds, the padding, the
gate, and that the gathered block lands back in the original row order.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer


def _tagged_rows_local(
    q_rows,
    q_scale_rows,
    k_quant,
    k_scale,
    weights_rows,
    ks_rows,
    ke_rows,
    topk_tokens,
    per,
    candidate_rows,
    candidate_block_size,
    candidate_write,
    use_fp4_cache,
):
    """Same output shape as _dsv41_prefill_rows_local; every column of row i
    holds that row's global index (q_rows[i, 0]) and padding rows hold -1."""
    width = topk_tokens
    if candidate_rows is not None and candidate_write:
        width += candidate_rows.shape[1]
    n = q_rows.shape[0]
    out = torch.full((per, width), -1, dtype=torch.int32)
    out[:n] = q_rows[:, :1].to(torch.int32).expand(n, width)
    return out


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("num_rows", [1, 63, 64, 65, 511, 512, 4096, 8191])
def test_bounds_cover_rows_in_rank_order(num_rows: int, world: int) -> None:
    covered: list[int] = []
    pers = set()
    for rank in range(world):
        lo, hi, per = sparse_indexer._dsv41_tp_split_bounds(num_rows, world, rank)
        pers.add(per)
        assert per % sparse_indexer._DSV41_TP_SPLIT_ROW_ALIGN == 0
        assert 0 <= lo <= hi <= num_rows
        assert hi - lo <= per
        assert lo == min(rank * per, num_rows)
        covered.extend(range(lo, hi))
    assert len(pers) == 1
    assert world * pers.pop() >= num_rows
    assert covered == list(range(num_rows))


def test_split_world_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sparse_indexer, "_DSV41_TP_SPLIT_ENABLED", False)
    assert sparse_indexer._dsv41_tp_split_world(8192, 1, False) == 1

    monkeypatch.setattr(sparse_indexer, "_DSV41_TP_SPLIT_ENABLED", True)
    monkeypatch.setattr(sparse_indexer, "_DSV41_TP_SPLIT_MIN", 512)
    monkeypatch.setattr(sparse_indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        sparse_indexer, "get_tp_group", lambda: SimpleNamespace(world_size=4)
    )
    assert sparse_indexer._dsv41_tp_split_world(511, 1, False) == 1
    assert sparse_indexer._dsv41_tp_split_world(512, 2, False) == 1
    assert sparse_indexer._dsv41_tp_split_world(512, 1, True) == 1
    assert sparse_indexer._dsv41_tp_split_world(512, 1, False) == 4

    monkeypatch.setattr(
        sparse_indexer, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    assert sparse_indexer._dsv41_tp_split_world(8192, 1, False) == 1


def test_env_mismatch_refuses_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank that skips the split's all-gather would hang the others, so a
    settings mismatch across TP ranks must fail the model build instead."""
    monkeypatch.setattr(sparse_indexer, "_DSV41_TP_SPLIT_ENABLED", False)
    monkeypatch.setattr(sparse_indexer, "_DSV41_TP_SPLIT_MIN", 512)
    monkeypatch.setattr(sparse_indexer, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        sparse_indexer,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, cpu_group=None),
    )

    def ranks_report(settings):
        def all_gather_object(out, obj, group=None):
            out[:] = settings

        monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    ranks_report([(True, 512)] + [(False, 512)] * 3)
    monkeypatch.setattr(sparse_indexer, "_dsv41_tp_split_env_checked", False)
    with pytest.raises(RuntimeError, match="differ across TP ranks"):
        sparse_indexer._dsv41_check_tp_split_env_consistent()

    ranks_report([(False, 512)] * 4)
    monkeypatch.setattr(sparse_indexer, "_dsv41_tp_split_env_checked", False)
    sparse_indexer._dsv41_check_tp_split_env_consistent()


@pytest.mark.parametrize("candidate_write", [False, True])
@pytest.mark.parametrize("num_rows", [512, 1000, 4096])
@pytest.mark.parametrize("world", [2, 4])
def test_chunk_split_restores_row_order(
    monkeypatch: pytest.MonkeyPatch, world: int, num_rows: int, candidate_write: bool
) -> None:
    monkeypatch.setattr(sparse_indexer, "_dsv41_prefill_rows_local", _tagged_rows_local)
    token_start, topk_tokens, candidate_cols = 7, 4, 3
    total = token_start + num_rows + 5
    q_quant = torch.arange(total, dtype=torch.int64).unsqueeze(1).repeat(1, 2)
    weights = torch.zeros(total, 1)
    chunk = SimpleNamespace(
        token_start=token_start,
        token_end=token_start + num_rows,
        cu_seqlen_ks=torch.zeros(num_rows, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(num_rows, dtype=torch.int32),
    )
    candidate_blocks = None
    if candidate_write:
        candidate_blocks = torch.full((total, candidate_cols), -2, dtype=torch.int32)

    # Every rank's local block, in rank order, as the all-gather collects them.
    local_blocks = []
    for rank in range(world):
        lo, hi, per = sparse_indexer._dsv41_tp_split_bounds(num_rows, world, rank)
        rows = slice(token_start + lo, token_start + hi)
        local_blocks.append(
            _tagged_rows_local(
                q_quant[rows],
                None,
                None,
                None,
                weights[rows],
                chunk.cu_seqlen_ks[lo:hi],
                chunk.cu_seqlen_ke[lo:hi],
                topk_tokens,
                per,
                candidate_blocks[rows] if candidate_blocks is not None else None,
                64,
                candidate_write,
                False,
            )
        )
    gathered = torch.cat(local_blocks, dim=0)
    expected = torch.arange(token_start, token_start + num_rows, dtype=torch.int32)

    for rank in range(world):
        topk_buffer = torch.full((total, topk_tokens), -2, dtype=torch.int32)
        candidates = candidate_blocks.clone() if candidate_blocks is not None else None
        calls: list[tuple[tuple[int, ...], int]] = []

        def all_gather(local, dim, calls=calls):
            calls.append((tuple(local.shape), dim))
            return gathered

        sparse_indexer._dsv41_prefill_chunk_tp_split(
            chunk,
            q_quant,
            None,
            weights,
            None,
            None,
            topk_buffer,
            topk_tokens,
            candidates,
            64,
            candidate_write,
            False,
            world,
            rank=rank,
            all_gather_fn=all_gather,
        )
        assert calls == [(tuple(local_blocks[rank].shape), 0)]
        chunk_rows = slice(token_start, token_start + num_rows)
        assert torch.equal(
            topk_buffer[chunk_rows], expected.unsqueeze(1).expand(-1, topk_tokens)
        )
        assert (topk_buffer[:token_start] == -2).all()
        assert (topk_buffer[token_start + num_rows :] == -2).all()
        if candidates is not None:
            assert torch.equal(
                candidates[chunk_rows], expected.unsqueeze(1).expand(-1, candidate_cols)
            )
