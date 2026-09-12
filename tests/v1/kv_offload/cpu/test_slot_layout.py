# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The slab planner: which groups share a row size, how the budget splits."""

import numpy as np
import pytest

from vllm.v1.kv_offload.config import OffloadingGroupConfig
from vllm.v1.kv_offload.cpu.slot_layout import (
    SlotLayout,
    expected_chunks,
    plan_slot_layout,
    slabs_enabled,
)

GIB = 1 << 30
ROW = 139_264  # round_up(138_240, 4096): the DSv4.1-Flash packed block


def _dsv41_groups() -> list[OffloadingGroupConfig]:
    """DeepSeek-V4.1-Flash: 13 three-layer SWA groups, 2 two-layer SWA
    groups, the ratio-2 MLA group (128 tokens/block), a scratch group and
    the ratio-1 MLA group."""
    swa3 = [OffloadingGroupConfig(64, ("swa",), 114_688, 128) for _ in range(13)]
    swa2 = [OffloadingGroupConfig(64, ("swa2",), 77_824, 128) for _ in range(2)]
    return [
        *swa3,
        *swa2,
        OffloadingGroupConfig(128, ("mla2",), 138_240, None),
        OffloadingGroupConfig(8, ("scratch",), 98_304, None, prefix_cacheable=False),
        OffloadingGroupConfig(64, ("mla1",), 77_824, None),
    ]


def _plan(groups, cpu_bytes=3 * GIB, **kwargs) -> SlotLayout:
    defaults = dict(
        blocks_per_chunk=1,
        max_model_len=524_288,
        retention_interval=8192,
        row_cap=ROW,
    )
    defaults.update(kwargs)
    return plan_slot_layout(cpu_bytes, groups, **defaults)


def test_classes_group_by_aligned_row_size_and_skip_scratch_groups():
    layout = _plan(_dsv41_groups())
    rows = [c.row_bytes for c in layout.classes]
    assert rows == [139_264, 114_688, 77_824]
    assert layout.classes[0].group_idxs == (15,)
    assert layout.classes[1].group_idxs == tuple(range(13))
    assert layout.classes[2].group_idxs == (13, 14, 17)
    assert layout.group_class[16] == -1
    with pytest.raises(AssertionError):
        layout.class_of_group(16)


def test_layout_fits_the_budget_with_page_aligned_disjoint_rows():
    layout = _plan(_dsv41_groups())
    assert layout.total_bytes <= 3 * GIB
    assert 3 * GIB - layout.total_bytes < layout.max_row_bytes
    offsets = layout.offset_table()
    rows = layout.row_bytes_table()
    assert len(offsets) == len(rows) == layout.num_blocks
    assert np.all(offsets % 4096 == 0)
    # Every row ends where the next begins: no overlap, no gap.
    assert np.array_equal(offsets[1:], offsets[:-1] + rows[:-1])
    assert offsets[-1] + rows[-1] == layout.total_bytes
    for cls_idx, c in enumerate(layout.classes):
        assert layout.class_of_block(c.first_block) == cls_idx
        assert layout.class_of_block(c.first_block + c.num_blocks - 1) == cls_idx


def test_default_split_holds_the_same_number_of_max_length_requests():
    """Each slab gets the bytes one max-length request leaves in it, scaled
    by the budget: every slab then fits the same number of such requests,
    and more than the 23,130 uniform rows of 3 GiB hold of a 493K session."""
    groups = _dsv41_groups()
    layout = _plan(groups)
    tokens = 524_288
    requests_per_class = []
    for c in layout.classes:
        demand = sum(
            expected_chunks(groups[g], groups[g].tokens_per_block, tokens, 8192)
            for g in c.group_idxs
        )
        requests_per_class.append(c.num_blocks / demand)
    assert max(requests_per_class) - min(requests_per_class) < 0.01
    assert requests_per_class[0] > 2.0
    # The old layout: 23,130 rows for ~15.3K chunks per 493K session.
    assert requests_per_class[0] * tokens / 493_000 > 1.25 * 23_130 / 15_300


def test_expected_chunks_full_attention_vs_window():
    full = OffloadingGroupConfig(64, ("a",), 4096, None)
    swa = OffloadingGroupConfig(64, ("b",), 4096, 128)
    assert expected_chunks(full, 64, 524_288, 8192) == 8192
    # One window (2 chunks + the boundary chunk) per retention interval.
    assert expected_chunks(swa, 64, 524_288, 8192) == 64 * 3
    # Without retention the window is kept once per request.
    assert expected_chunks(swa, 64, 524_288, None) == 3


def test_shares_override_and_floor():
    groups = _dsv41_groups()
    default = _plan(groups)
    shares = _plan(groups, shares={"114688": 0.0})
    swa_default = default.classes[1].num_blocks
    swa_floored = shares.classes[1].num_blocks
    assert 0 < swa_floored < swa_default
    # The floor keeps about 2% of the budget for the zero-weighted slab
    # (2% before the shares are renormalized).
    assert swa_floored * 114_688 >= 0.019 * 3 * GIB
    assert shares.classes[0].num_blocks > default.classes[0].num_blocks
    with pytest.raises(ValueError, match="no KV cache group uses"):
        _plan(groups, shares={"4096": 1.0})
    with pytest.raises(ValueError):
        _plan(groups, shares={"139264": -1.0})


def test_single_row_size_is_a_uniform_layout_of_the_whole_budget():
    groups = [
        OffloadingGroupConfig(16, ("a",), 8192),
        OffloadingGroupConfig(16, ("b",), 8192),
    ]
    layout = _plan(groups, cpu_bytes=GIB, row_cap=8192)
    assert layout.is_uniform
    assert layout.num_blocks == GIB // 8192
    assert layout.group_class == (0, 0)
    uniform = SlotLayout.uniform(GIB // 8192, 8192, 2)
    assert np.array_equal(layout.offset_table(), uniform.offset_table())


def test_rows_are_capped_by_the_packed_block():
    groups = [OffloadingGroupConfig(16, ("merged",), 225_280)]
    layout = _plan(groups, row_cap=ROW)
    assert layout.max_row_bytes == ROW


@pytest.mark.parametrize(
    ("extra", "packed", "single_copy", "bpc", "expected"),
    [
        ({}, True, True, 1, True),
        ({"cpu_slabs": False}, True, True, 1, False),
        ({}, False, True, 1, False),
        ({}, True, False, 1, False),
        ({}, True, True, 2, False),
    ],
)
def test_slabs_enabled_conditions(extra, packed, single_copy, bpc, expected):
    assert (
        slabs_enabled(
            extra, packed_layout=packed, single_copy=single_copy, blocks_per_chunk=bpc
        )
        is expected
    )
