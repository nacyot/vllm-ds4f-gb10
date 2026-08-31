# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK_VISION_BIDI: pure image-span visibility arithmetic (no vllm imports).

DeepSeek-V4-Flash-Vision-Exp attends bidirectionally inside every
``[IMAGE_START, IMAGE_END]`` sentinel span of the prompt (reference
``model.py``: ``get_image_visible`` + ``get_window_topk_idxs_visible``): each
in-span token sees the whole span plus its ordinary trailing causal window,
so the per-row sliding-window index list widens from ``sliding_window`` (128)
to ``sliding_window + vision_max_n_token`` (512) and gains look-ahead rows.

This module holds the pure arithmetic:

- ``image_visible_flat``: the reference ``get_image_visible`` over the
  flattened (multi-request) batch, plus the split-span ``bad`` flag.
- ``window_rows_visible``: the row start/length/content formulas in POSITION
  space. ``_compute_vision_swa_indices_and_lens_kernel``
  (vllm/v1/attention/backends/mla/sparse_swa.py) mirrors it 1:1 and only adds
  the position -> paged-slot mapping; ``slots_from_positions`` is that
  mapping in torch, for GPU self-checks (``DSPARK_VISION_BIDI_CHECK=1``).
- ``wide_segments``: splits one step's prefill rows into causal (narrow) and
  image (wide) launch segments for the SM120 packed prefill.

No ``vllm`` imports (torch + os only): the CPU tests load this file by path
without the compiled ``vllm._C`` extension.
"""

import os

import torch

# Sentinel type offsets above vocab_size; must equal the constants in
# vllm/models/deepseek_v4/image_processing.py (asserted by the CPU tests).
# IMAGE / IMAGE_NEW_LINE are span-interior types: one of them outside a
# [START, END] span means the chunk was cut inside an image.
IMAGE_START = 0
IMAGE = 2
IMAGE_NEW_LINE = 3
IMAGE_END = 4

ENV = "DSPARK_VISION_BIDI"

# Index widths the FlashInfer sparse-MLA SM120 decode kernel dispatches on
# (flashinfer ``_sparse_mla_sm120._DECODE_DSV4_DISPATCH`` topk column). Wide
# prefill rows ride the <=64-row decode-kernel path (the dual-cache prefill
# orchestrator is TOPK=128 only), so the vision index width must be in here.
DECODE_WIDTHS = (128, 512, 1024)

# The SM120 prefill orchestrator requires num_tokens > 64 (ICHECK in
# flashinfer ``sparse_mla_sm120.cu``); narrow (causal) runs shorter than this
# are absorbed into a neighbouring wide segment (their 512-wide rows are
# equally valid) instead of forming an own launch.
ORCH_MIN_ROWS = 65


def vision_bidi_enabled(hf_config) -> bool:
    """True iff this is a vision checkpoint and ``DSPARK_VISION_BIDI`` != 0.

    Default on for vision configs (``vision_n_layers > 0``); always False for
    text checkpoints, which must stay byte-identical. ``vllm/envs.py``
    documents the same variable; this reader is os.environ-direct so the
    module stays importable without vllm.
    """
    if int(getattr(hf_config, "vision_n_layers", 0) or 0) <= 0:
        return False
    return os.environ.get(ENV, "1").strip() != "0"


def vision_index_width(hf_config) -> int:
    """Per-row prefill index width for vision models: window + max span.

    128 + 384 = 512 for DeepSeek-V4-Flash-Vision-Exp (reference:
    ``width = min(seqlen, window_size + max_image_tokens)``).
    """
    return int(getattr(hf_config, "sliding_window", 0) or 0) + int(
        getattr(hf_config, "vision_max_n_token", 0) or 0
    )


def image_visible_flat(
    input_ids: torch.Tensor,
    vocab_size: int,
    max_image_tokens: int,
    valid_mask: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference ``get_image_visible`` over the flattened batch.

    Returns ``(left, right, bad)``: int32 visible counts to the left/right
    within each ``[IMAGE_START, IMAGE_END]`` span (0/0 for text rows, decode
    rows and everything outside a span) and a 0-d bool ``bad`` set when some
    request chunk holds an incomplete span: an END without a START, a START
    without an END, or span-interior sentinels (IMAGE / IMAGE_NEW_LINE)
    outside any span (a chunk cut inside an image, e.g. a prefix-cache hit
    resuming mid-span).

    With ``query_start_loc`` ([num_reqs + 1]) both the check and the
    degradation are per-request: a broken request gets ``left = right = 0``
    on all its rows (plain causal, the phase-1 behaviour) while every intact
    request keeps its exact reference visibility, and ``bad`` is the OR over
    requests. Masking the broken requests' sentinels before the cummax /
    flip-cummin pass also keeps their dangling START/END from leaking
    visibility into neighbouring requests. Without ``query_start_loc`` the
    check is stream-global and left/right are returned unmodified.

    ``valid_mask`` zeroes padded rows first so stale sentinel ids in the
    padded tail cannot fabricate spans. Everything stays on device: no host
    sync in either path.
    """
    n = input_ids.shape[0]
    if n == 0:
        z = input_ids.new_zeros(0, dtype=torch.int32)
        return z, z.clone(), input_ids.new_zeros((), dtype=torch.bool)
    ids = input_ids
    if valid_mask is not None:
        ids = torch.where(valid_mask, ids, torch.zeros_like(ids))
    is_s = ids == vocab_size + IMAGE_START
    is_e = ids == vocab_size + IMAGE_END
    cs = (is_s.to(torch.int32) - is_e.to(torch.int32)).cumsum(0, dtype=torch.int32)
    if query_start_loc is not None:
        qsl = query_start_loc.long()
        num_reqs = qsl.shape[0] - 1
        idx_l = torch.arange(n, dtype=torch.int64, device=ids.device)
        # Request of each row; padded tail rows (>= qsl[-1]) clamp to the
        # last request and carry zeroed ids, so they contribute nothing.
        req_of = (torch.searchsorted(qsl, idx_l, right=True) - 1).clamp(
            0, max(num_reqs - 1, 0)
        )
        delta = is_s.to(torch.int32) - is_e.to(torch.int32)
        sums = torch.zeros(num_reqs, dtype=torch.int32, device=ids.device)
        sums.scatter_add_(0, req_of, delta)
        # cs with the carry of earlier requests removed = the reference
        # per-request cumsum (requests are contiguous in the flat stream).
        baseline = sums.cumsum(0, dtype=torch.int32) - sums
        cs_rel = cs - baseline[req_of]
        min_rel = torch.zeros(num_reqs, dtype=torch.int32, device=ids.device)
        min_rel.scatter_reduce_(0, req_of, cs_rel, reduce="amin")
        # Span-interior sentinels outside a span: the chunk resumed mid-span
        # and holds neither START nor END. Leading IMAGE_PADs before a START
        # are legitimately outside the span (causal), so pads do not count.
        interior = (ids == vocab_size + IMAGE) | (ids == vocab_size + IMAGE_NEW_LINE)
        stray = (interior & ~((cs_rel > 0) | is_e)).to(torch.int32)
        strays = torch.zeros(num_reqs, dtype=torch.int32, device=ids.device)
        strays.scatter_add_(0, req_of, stray)
        bad_req = (sums != 0) | (min_rel < 0) | (strays > 0)
        # Recompute visibility with the broken requests' sentinels masked:
        # their rows become plain causal and cannot contaminate the other
        # requests' cummax/cummin.
        ids = torch.where(bad_req[req_of], torch.zeros_like(ids), ids)
        is_s = ids == vocab_size + IMAGE_START
        is_e = ids == vocab_size + IMAGE_END
        cs = (is_s.to(torch.int32) - is_e.to(torch.int32)).cumsum(0, dtype=torch.int32)
        bad = bad_req.any()
    else:
        bad = (cs.min() < 0) | (cs[-1] != 0)
    idx = torch.arange(n, dtype=torch.int32, device=ids.device)
    valid = (cs > 0) | is_e
    starts = torch.where(is_s, idx, torch.zeros_like(idx)).cummax(0).values
    left = (idx - starts) * valid
    ends = (
        torch.where(is_e, idx, torch.full_like(idx, n)).flip(0).cummin(0).values.flip(0)
    )
    right = (ends - idx) * valid
    return (
        left.clamp(max=max_image_tokens - 1).to(torch.int32),
        right.clamp(max=max_image_tokens).to(torch.int32),
        bad,
    )


def window_rows_visible(
    positions: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    window_size: int,
    index_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``get_window_topk_idxs_visible`` in POSITION space.

    ``_compute_vision_swa_indices_and_lens_kernel`` (sparse_swa.py) mirrors
    this 1:1 per row and only replaces each position with its paged slot id
    (and -1 beyond ``lens``); the variable names match line for line.
    ``left = right = 0`` with ``index_width == window_size`` reproduces the
    causal ``_compute_swa_indices_and_lens_kernel`` exactly.
    """
    positions = positions.long()
    left = left.long()
    right = right.long()
    # kernel: left_add = tl.maximum(left - (window_size - 1), 0)
    left_add = (left - (window_size - 1)).clamp(min=0)
    # kernel: start_pos = tl.maximum(pos - (window_size - 1) - left_add, 0)
    start = (positions - (window_size - 1) - left_add).clamp(min=0)
    # kernel: swa_len = tl.minimum(pos + right + 1 - start_pos, index_width)
    lens = (positions + right + 1 - start).clamp(max=index_width)
    cols = torch.arange(index_width, device=positions.device)
    # kernel: slot_ids = tl.where(offset < swa_len, <slot of start_pos+offset>, -1)
    rows = torch.where(
        cols[None, :] < lens[:, None], start[:, None] + cols[None, :], -1
    )
    return rows.to(torch.int32), lens.to(torch.int32)


def slots_from_positions(
    rows: torch.Tensor, block_table: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Map position-space ``rows`` (-1 padded) to paged slot ids.

    GPU self-check aid only (``DSPARK_VISION_BIDI_CHECK=1``): mirrors the
    kernel's ``block_table[pos // block_size] * block_size + pos % block_size``.
    ``block_table`` is one request's row ([num_blocks]) or per-row batched
    ([rows.shape[0], num_blocks]).
    """
    safe = rows.clamp(min=0).long()
    blocks = safe // block_size
    if block_table.dim() == 1:
        mapped = block_table.long()[blocks]
    else:
        mapped = torch.gather(block_table.long(), 1, blocks)
    slots = mapped * block_size + safe % block_size
    return torch.where(rows >= 0, slots, torch.full_like(slots, -1)).to(torch.int32)


def wide_segments(
    wide, min_narrow_rows: int = ORCH_MIN_ROWS
) -> list[tuple[int, int, bool]] | None:
    """Segment one step's prefill rows into launch runs.

    ``wide``: per-prefill-row bools (True = the row needs a wide index row,
    i.e. it is inside an image span). Returns ``[(start, end, is_wide)]``
    covering ``[0, len(wide))`` contiguously, or None when no row is wide
    (text-only step -> single causal launch, today's path).

    Narrow runs shorter than ``min_narrow_rows`` (the SM120 orchestrator's
    minimum row count) are absorbed into a neighbouring wide run: their
    512-wide causal rows are equally valid and the merge saves launches.
    Wide runs are tiled to <=64-row decode-kernel calls by the caller.
    """
    runs: list[tuple[int, int, bool]] = []
    for i, w in enumerate(wide):
        flag = bool(w)
        if runs and runs[-1][2] == flag:
            runs[-1] = (runs[-1][0], i + 1, flag)
        else:
            runs.append((i, i + 1, flag))
    if not any(flag for _, _, flag in runs):
        return None
    for k, (s, e, flag) in enumerate(runs):
        if not flag and e - s < min_narrow_rows:
            runs[k] = (s, e, True)
    merged: list[tuple[int, int, bool]] = []
    for s, e, flag in runs:
        if merged and merged[-1][2] == flag:
            merged[-1] = (merged[-1][0], e, flag)
        else:
            merged.append((s, e, flag))
    return merged
