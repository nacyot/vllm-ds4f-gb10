# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-size classes ("slabs") of the CPU offload region.

With a packed block layout a KV cache group's bytes are the prefix of its
block, so groups have different byte sizes per block while every CPU row used
to be sized for the largest one. A ``SlotLayout`` carves the region into one
slab per distinct row size, and maps every group to the slab whose rows fit
it exactly. Block ids stay global: slab ``c`` owns the contiguous id range
``[first_block_c, first_block_c + num_blocks_c)``, and ``offsets[block_id]``
gives the byte offset of any block in the region.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_offload.config import OffloadingGroupConfig

# Slot rows are page aligned so O_DIRECT tiers can read and write them.
SLOT_ALIGNMENT = 4096
# Every slab keeps about this share of the budget (applied before the shares
# are renormalized), so a class whose demand estimate is off still caches
# something.
MIN_CLASS_SHARE = 0.02


@dataclass(frozen=True)
class SlotClass:
    row_bytes: int
    num_blocks: int
    first_block: int
    base_offset: int
    group_idxs: tuple[int, ...]


@dataclass(frozen=True)
class SlotLayout:
    classes: tuple[SlotClass, ...]
    # group idx -> class idx; -1 for groups that are never offloaded.
    group_class: tuple[int, ...]

    @property
    def num_blocks(self) -> int:
        return sum(c.num_blocks for c in self.classes)

    @property
    def total_bytes(self) -> int:
        return sum(c.num_blocks * c.row_bytes for c in self.classes)

    @property
    def max_row_bytes(self) -> int:
        return max(c.row_bytes for c in self.classes)

    @property
    def is_uniform(self) -> bool:
        return len(self.classes) == 1

    def class_of_group(self, group_idx: int) -> int:
        cls = self.group_class[group_idx]
        assert cls >= 0, f"KV cache group {group_idx} is not offloaded"
        return cls

    def class_of_block(self, block_id: int) -> int:
        for idx, c in enumerate(self.classes):
            if c.first_block <= block_id < c.first_block + c.num_blocks:
                return idx
        raise IndexError(f"block id {block_id} outside the slot layout")

    def offset_table(self) -> np.ndarray:
        """Byte offset of every block id in the region (int64)."""
        offsets = np.empty(self.num_blocks, dtype=np.int64)
        for c in self.classes:
            end = c.first_block + c.num_blocks
            offsets[c.first_block : end] = (
                c.base_offset + np.arange(c.num_blocks, dtype=np.int64) * c.row_bytes
            )
        return offsets

    def row_bytes_table(self) -> np.ndarray:
        """Row bytes of every block id (int64)."""
        rows = np.empty(self.num_blocks, dtype=np.int64)
        for c in self.classes:
            rows[c.first_block : c.first_block + c.num_blocks] = c.row_bytes
        return rows

    def describe(self) -> str:
        return ", ".join(
            f"{c.row_bytes} B x {c.num_blocks} slots (groups {list(c.group_idxs)})"
            for c in self.classes
        )

    @classmethod
    def uniform(
        cls, num_blocks: int, row_bytes: int, num_groups: int = 1
    ) -> "SlotLayout":
        """One slab of ``num_blocks`` rows: the layout without slabs."""
        group_idxs = tuple(range(num_groups))
        return cls(
            classes=(SlotClass(row_bytes, num_blocks, 0, 0, group_idxs),),
            group_class=tuple(0 for _ in group_idxs),
        )


def slabs_enabled(
    extra_config: Mapping[str, Any],
    *,
    packed_layout: bool,
    single_copy: bool,
    blocks_per_chunk: int,
) -> bool:
    """Whether per-group row sizes apply: a group's bytes must be the prefix
    of one host slot (packed layout, single copy), one block per chunk."""
    return (
        bool(extra_config.get("cpu_slabs", True))
        and packed_layout
        and single_copy
        and blocks_per_chunk == 1
    )


def expected_chunks(
    group: OffloadingGroupConfig,
    tokens_per_chunk: int,
    max_model_len: int,
    retention_interval: int | None,
) -> int:
    """Chunks one request of ``max_model_len`` tokens leaves in the offload
    tier for this group: every chunk for full attention, one window per
    prefix-cache retention interval (or per request) otherwise."""
    max_model_len = max(max_model_len, tokens_per_chunk)
    if group.window_tokens is None:
        return cdiv(max_model_len, tokens_per_chunk)
    interval = retention_interval or max_model_len
    window_chunks = cdiv(group.window_tokens, tokens_per_chunk) + 1
    return cdiv(max_model_len, interval) * window_chunks


def plan_slot_layout(
    cpu_bytes: int,
    groups: Sequence[OffloadingGroupConfig],
    *,
    blocks_per_chunk: int,
    max_model_len: int,
    retention_interval: int | None,
    row_cap: int,
    shares: Mapping[str, float] | None = None,
) -> SlotLayout:
    """Split ``cpu_bytes`` into one slab per distinct group row size.

    Each slab's share of the budget defaults to the bytes one request of
    ``max_model_len`` tokens leaves in it (``expected_chunks`` x row bytes),
    so every slab holds the same number of such requests. ``shares`` maps a
    row size (as a decimal string) to a relative weight and overrides the
    default for that slab. ``row_cap`` bounds a row (the packed block).
    """
    classes: dict[int, list[int]] = {}
    for idx, group in enumerate(groups):
        if not group.prefix_cacheable or group.bytes_per_block <= 0:
            continue
        row = min(round_up(group.bytes_per_block, SLOT_ALIGNMENT), row_cap)
        classes.setdefault(row, []).append(idx)
    assert classes, "no offloadable KV cache group has a known block size"
    rows = sorted(classes, reverse=True)

    weights: dict[int, float] = {}
    for row in rows:
        demand = sum(
            expected_chunks(
                groups[g],
                groups[g].tokens_per_block * blocks_per_chunk,
                max_model_len,
                retention_interval,
            )
            for g in classes[row]
        )
        weights[row] = float(demand * row)
    if shares:
        for key, weight in shares.items():
            row = int(key)
            if row not in weights:
                raise ValueError(
                    f"cpu_slab_shares names row size {row}, which no KV cache "
                    f"group uses (rows: {rows})"
                )
            if weight < 0:
                raise ValueError("cpu_slab_shares weights must be >= 0")
            weights[row] = float(weight)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("cpu_slab_shares must give at least one slab a weight")
    fractions = {row: w / total_weight for row, w in weights.items()}
    floor = MIN_CLASS_SHARE if len(rows) > 1 else 0.0
    fractions = {row: max(f, floor) for row, f in fractions.items()}
    norm = sum(fractions.values())
    fractions = {row: f / norm for row, f in fractions.items()}

    num_blocks = {row: int(cpu_bytes * fractions[row]) // row for row in rows}
    # Whole rows left over by the integer split go to the widest rows.
    leftover = cpu_bytes - sum(n * row for row, n in num_blocks.items())
    num_blocks[rows[0]] += leftover // rows[0]

    slot_classes: list[SlotClass] = []
    first_block = 0
    base_offset = 0
    group_class = [-1] * len(groups)
    for cls_idx, row in enumerate(rows):
        slot_classes.append(
            SlotClass(
                row_bytes=row,
                num_blocks=num_blocks[row],
                first_block=first_block,
                base_offset=base_offset,
                group_idxs=tuple(classes[row]),
            )
        )
        for g in classes[row]:
            group_class[g] = cls_idx
        first_block += num_blocks[row]
        base_offset += num_blocks[row] * row
    return SlotLayout(classes=tuple(slot_classes), group_class=tuple(group_class))
