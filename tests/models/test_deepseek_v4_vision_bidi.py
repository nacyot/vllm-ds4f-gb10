# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK_VISION_BIDI CPU tests (pure torch, no compiled vLLM extension).

Loads ``vllm/models/deepseek_v4/image_attention.py`` by file path and checks
it against DeepSeek's reference ``get_image_visible`` /
``get_window_topk_idxs_visible``, imported by path from
``/tmp/ds4v/ref_model.py`` (the two pure functions are extracted from its
source because the module itself imports GPU-only deps such as ``kernel``).
Token streams are built with the reference ``build_image_block`` from
``/tmp/ds4v/ref_image_processor.py`` at ``vocab_size`` 129280.

Reference-material tests are skipped when ``/tmp/ds4v`` is absent; the
self-contained tests (causal parity, segmentation, constants) always run.
"""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu_test

MATERIALS = Path(os.environ.get("DS4V_MATERIALS", "/tmp/ds4v"))
REF_MODEL = MATERIALS / "ref_model.py"
REF_IMAGE_PROCESSOR = MATERIALS / "ref_image_processor.py"
REPO = Path(__file__).resolve().parents[2]
IMAGE_ATTENTION = REPO / "vllm/models/deepseek_v4/image_attention.py"
IMAGE_PROCESSING = REPO / "vllm/models/deepseek_v4/image_processing.py"

VOCAB = 129280
MAX_IMG = 384
WINDOW = 128
WIDTH = 512  # WINDOW + MAX_IMG

requires_materials = pytest.mark.skipif(
    not (REF_MODEL.is_file() and REF_IMAGE_PROCESSOR.is_file()),
    reason="DeepSeek-V4 vision reference materials not present",
)


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ia = _load_by_path("dsv4_image_attention", IMAGE_ATTENTION)
ours_ip = _load_by_path("dsv4_image_processing", IMAGE_PROCESSING)


@pytest.fixture(scope="module")
def ref_ip():
    if not REF_IMAGE_PROCESSOR.is_file():
        pytest.skip("reference image processor not present")
    return _load_by_path("dsv4_ref_image_processor", REF_IMAGE_PROCESSOR)


@pytest.fixture(scope="module")
def ref_fns(ref_ip):
    """(get_image_visible, get_window_topk_idxs_visible) from ref_model.py.

    Extracted from the file's source by name: ref_model.py itself imports
    GPU-only modules, so it cannot be exec'd whole on CPU.
    """
    if not REF_MODEL.is_file():
        pytest.skip("reference model not present")
    tree = ast.parse(REF_MODEL.read_text())
    wanted = {"get_image_visible", "get_window_topk_idxs_visible"}
    ns: dict = {
        "torch": torch,
        "IMAGE_START": int(ref_ip.IMAGE_START),
        "IMAGE_END": int(ref_ip.IMAGE_END),
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            code = compile(
                ast.Module(body=[node], type_ignores=[]), str(REF_MODEL), "exec"
            )
            exec(code, ns)
    assert wanted <= set(ns), "reference functions not found in ref_model.py"
    return ns["get_image_visible"], ns["get_window_topk_idxs_visible"]


def _text(rng, n: int) -> torch.Tensor:
    return torch.from_numpy(rng.integers(0, VOCAB, size=n)).to(torch.int64)


def _block_ids(ref_ip, rng, start_pos: int, max_hw: int = 13) -> torch.Tensor:
    h = int(rng.integers(1, max_hw + 1))
    w = int(rng.integers(1, max_hw + 1))
    types, _ = ref_ip.build_image_block(h, w, start_pos)
    return (VOCAB + types).to(torch.int64)


def _stream(ref_ip, rng, num_blocks: int, tail: int | None = None) -> torch.Tensor:
    """Text/image stream: text run, block, ..., trailing text (``tail``)."""
    parts: list[torch.Tensor] = []
    n = 0
    for _ in range(num_blocks):
        t = int(rng.integers(1, 40))
        parts.append(_text(rng, t))
        n += t
        block = _block_ids(ref_ip, rng, n)
        parts.append(block)
        n += block.numel()
    t = int(rng.integers(1, 40)) if tail is None else tail
    parts.append(_text(rng, t))
    return torch.cat(parts) if parts else _text(rng, 1)


def _ref_visible(ref_fns, ids: torch.Tensor):
    left, right = ref_fns[0](ids.unsqueeze(0), VOCAB, MAX_IMG)
    return left[0].long(), right[0].long()


# --------------------------------------------------------------------- (a)
@requires_materials
def test_image_visible_flat_single_request(ref_ip, ref_fns):
    rng = np.random.default_rng(0)
    for num_blocks in (0, 1, 2, 3):
        for _ in range(5):
            ids = _stream(ref_ip, rng, num_blocks)
            left, right, bad = ia.image_visible_flat(ids, VOCAB, MAX_IMG)
            ref_l, ref_r = _ref_visible(ref_fns, ids)
            assert left.long().equal(ref_l)
            assert right.long().equal(ref_r)
            assert not bool(bad)


@requires_materials
def test_image_visible_flat_multi_request_and_decode(ref_ip, ref_fns):
    rng = np.random.default_rng(1)
    for _ in range(8):
        n_dec = int(rng.integers(0, 4))
        n_req = int(rng.integers(2, 4))
        decode_ids = _text(rng, n_dec)
        streams = [_stream(ref_ip, rng, int(rng.integers(0, 3))) for _ in range(n_req)]
        flat = torch.cat([decode_ids, *streams])
        lens = [1] * n_dec + [s.numel() for s in streams]
        qsl = torch.tensor([0, *np.cumsum(lens).tolist()], dtype=torch.int32)
        left, right, bad = ia.image_visible_flat(
            flat, VOCAB, MAX_IMG, query_start_loc=qsl
        )
        assert not bool(bad)
        exp_l = [torch.zeros(n_dec, dtype=torch.int64)]
        exp_r = [torch.zeros(n_dec, dtype=torch.int64)]
        for s in streams:
            rl, rr = _ref_visible(ref_fns, s)
            exp_l.append(rl)
            exp_r.append(rr)
        assert left.long().equal(torch.cat(exp_l))
        assert right.long().equal(torch.cat(exp_r))


@requires_materials
def test_image_visible_flat_masked_tail(ref_ip, ref_fns):
    rng = np.random.default_rng(2)
    ids = _stream(ref_ip, rng, 2)
    n = ids.numel()
    # Stale sentinel ids in the padded tail must not fabricate spans.
    pad = torch.full((7,), VOCAB + int(ref_ip.IMAGE_START), dtype=torch.int64)
    full = torch.cat([ids, pad])
    mask = torch.cat(
        [torch.ones(n, dtype=torch.bool), torch.zeros(7, dtype=torch.bool)]
    )
    qsl = torch.tensor([0, n], dtype=torch.int32)
    left, right, bad = ia.image_visible_flat(
        full, VOCAB, MAX_IMG, valid_mask=mask, query_start_loc=qsl
    )
    assert not bool(bad)
    ref_l, ref_r = _ref_visible(ref_fns, ids)
    assert left[:n].long().equal(ref_l)
    assert right[:n].long().equal(ref_r)
    assert (left[n:] == 0).all() and (right[n:] == 0).all()


# --------------------------------------------------------------------- (b)
@requires_materials
def test_bad_span_detection(ref_ip):
    st = VOCAB + int(ref_ip.IMAGE_START)
    en = VOCAB + int(ref_ip.IMAGE_END)
    pd = VOCAB + int(ref_ip.IMAGE_PAD)

    def text(n):
        return torch.arange(n, dtype=torch.int64) % 100

    # Chunk starting mid-span: END without a START anywhere.
    dangling_end = torch.cat([text(5), torch.tensor([pd, pd, en]), text(4)])
    assert bool(ia.image_visible_flat(dangling_end, VOCAB, MAX_IMG)[2])

    # Chunk ending mid-span: open START (no query_start_loc: stream-end check).
    open_start = torch.cat([text(3), torch.tensor([st, pd, pd])])
    assert bool(ia.image_visible_flat(open_start, VOCAB, MAX_IMG)[2])

    # Open START followed by a complete request: per-request ends unbalance.
    complete = torch.cat([text(2), torch.tensor([st, pd, en]), text(2)])
    both = torch.cat([open_start, complete])
    qsl = torch.tensor([0, open_start.numel(), both.numel()], dtype=torch.int32)
    assert bool(ia.image_visible_flat(both, VOCAB, MAX_IMG, query_start_loc=qsl)[2])

    # The cancelling pair: open START in req A + dangling END in req B keeps
    # the whole-stream cumsum balanced; only the per-request check catches it.
    cancel = torch.cat([open_start, dangling_end])
    qsl = torch.tensor([0, open_start.numel(), cancel.numel()], dtype=torch.int32)
    assert bool(ia.image_visible_flat(cancel, VOCAB, MAX_IMG, query_start_loc=qsl)[2])

    # Complete spans in every request -> not bad.
    two = torch.cat([complete, complete])
    qsl = torch.tensor([0, complete.numel(), two.numel()], dtype=torch.int32)
    assert not bool(ia.image_visible_flat(two, VOCAB, MAX_IMG, query_start_loc=qsl)[2])


@requires_materials
def test_per_request_causal_degradation(ref_ip, ref_fns):
    """A broken request degrades only its own rows; neighbours keep bidi."""
    st = VOCAB + int(ref_ip.IMAGE_START)
    en = VOCAB + int(ref_ip.IMAGE_END)
    pd = VOCAB + int(ref_ip.IMAGE_PAD)
    im = VOCAB + int(ref_ip.IMAGE)
    nl = VOCAB + int(ref_ip.IMAGE_NEW_LINE)

    def text(n):
        return torch.arange(n, dtype=torch.int64) % 100

    complete = torch.cat([text(2), torch.tensor([st, im, nl, en]), text(2)])
    ref_l, ref_r = _ref_visible(ref_fns, complete)
    n = complete.numel()

    def check(flat, qsl, good, bad_rows):
        left, right, bad = ia.image_visible_flat(
            flat,
            VOCAB,
            MAX_IMG,
            query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        )
        assert bool(bad)
        assert left[good].long().equal(ref_l)
        assert right[good].long().equal(ref_r)
        assert (left[bad_rows] == 0).all() and (right[bad_rows] == 0).all()

    # Open START ahead of an intact request: no forward visibility leak.
    open_start = torch.cat([text(3), torch.tensor([st, pd, pd])])
    m = open_start.numel()
    check(
        torch.cat([open_start, complete]),
        [0, m, m + n],
        slice(m, None),
        slice(0, m),
    )
    # Dangling END behind an intact request: no backward visibility leak.
    remainder = torch.cat([torch.tensor([im, nl, im, en]), text(2)])
    check(
        torch.cat([complete, remainder]),
        [0, n, n + remainder.numel()],
        slice(0, n),
        slice(n, None),
    )
    # Mid-span chunk with neither START nor END (prefix-cache resume plus a
    # budget re-cut): span-interior sentinels outside any span flag it.
    interior = torch.cat([torch.tensor([im, im, nl, im]), text(3)])
    check(
        torch.cat([complete, interior]),
        [0, n, n + interior.numel()],
        slice(0, n),
        slice(n, None),
    )


# --------------------------------------------------------------------- (c)
def _assert_rows_match(ref_rows: torch.Tensor, rows: torch.Tensor, lens: torch.Tensor):
    """ref_rows: [L, width<=512] reference matrix; rows/lens: ours at 512."""
    width = ref_rows.shape[-1]
    padded = torch.nn.functional.pad(ref_rows, (0, WIDTH - width), value=-1)
    assert rows.long().equal(padded.long())
    valid = padded != -1
    assert lens.long().equal(valid.sum(-1).long())
    # The valid prefix must be contiguous (no -1 holes before lens).
    assert (valid.long().diff(dim=-1) <= 0).all()


@requires_materials
def test_window_rows_visible_matches_reference(ref_ip, ref_fns):
    get_vis, get_rows = ref_fns
    rng = np.random.default_rng(3)
    cases: list[torch.Tensor] = []
    for target_len in (200, 700, 2048):
        max_hw = 6 if target_len == 200 else 13
        # Span at the start: leading text of 0..3 tokens puts the block's
        # compress pads at start_pos % 4 in {0, 1, 2, 3}.
        for lead in range(4):
            ids = torch.cat(
                [
                    _text(rng, lead) if lead else torch.empty(0, dtype=torch.int64),
                    _block_ids(ref_ip, rng, lead, max_hw),
                ]
            )
            cases.append(torch.cat([ids, _text(rng, target_len - ids.numel())]))
        # Span in the middle.
        mid = target_len // 2
        ids = torch.cat([_text(rng, mid), _block_ids(ref_ip, rng, mid, max_hw)])
        cases.append(torch.cat([ids, _text(rng, target_len - ids.numel())]))
        # Span at the end: IMAGE_END is the last token (length ~ target_len;
        # no truncation, which could cut inside the span). The reserve covers
        # the largest block a max_hw draw can produce.
        head = _text(rng, target_len - (60 if max_hw == 6 else 220))
        block = _block_ids(ref_ip, rng, head.numel(), max_hw)
        ids = torch.cat([head, block])
        assert ids[-1].item() == VOCAB + int(ref_ip.IMAGE_END)
        cases.append(ids)
        # Back-to-back blocks.
        b1 = _block_ids(ref_ip, rng, 7, max_hw)
        b2 = _block_ids(ref_ip, rng, 7 + b1.numel(), max_hw)
        ids = torch.cat([_text(rng, 7), b1, b2])
        cases.append(torch.cat([ids, _text(rng, target_len - ids.numel())]))
    for ids in cases:
        length = ids.numel()
        left, right = get_vis(ids.unsqueeze(0), VOCAB, MAX_IMG)
        ref_rows = get_rows(WINDOW, length, left, right, MAX_IMG)[0]
        rows, lens = ia.window_rows_visible(
            torch.arange(length), left[0], right[0], WINDOW, WIDTH
        )
        _assert_rows_match(ref_rows, rows, lens)


@requires_materials
def test_window_rows_visible_oversized_span_clamp(ref_fns):
    # Synthetic > 384-token span: exercises the left/right clamps and the
    # index_width truncation of the look-ahead.
    _, get_rows = ref_fns
    length, off, span = 700, 60, 500
    left = torch.zeros(length, dtype=torch.int64)
    right = torch.zeros(length, dtype=torch.int64)
    left[off : off + span] = torch.arange(span).clamp(max=MAX_IMG - 1)
    right[off : off + span] = torch.arange(span - 1, -1, -1).clamp(max=MAX_IMG)
    ref_rows = get_rows(WINDOW, length, left.unsqueeze(0), right.unsqueeze(0), MAX_IMG)[
        0
    ]
    rows, lens = ia.window_rows_visible(
        torch.arange(length), left, right, WINDOW, WIDTH
    )
    _assert_rows_match(ref_rows, rows, lens)


# --------------------------------------------------------------------- (d)
def test_causal_parity_zero_left_right():
    for length in (1, 5, 200, 700):
        pos = torch.arange(length)
        zero = torch.zeros(length, dtype=torch.int64)
        for index_width in (WINDOW, WIDTH):
            rows, lens = ia.window_rows_visible(pos, zero, zero, WINDOW, index_width)
            expected_lens = torch.minimum(pos + 1, torch.tensor(WINDOW))
            assert lens.long().equal(expected_lens)
            for p in list(range(min(length, 130))) + [length - 1]:
                window = torch.arange(max(p - (WINDOW - 1), 0), p + 1)
                assert rows[p, : lens[p]].long().equal(window)
                assert (rows[p, lens[p] :] == -1).all()


# --------------------------------------------------------------------- (e)
def test_wide_segments():
    ws = ia.wide_segments
    assert ws([]) is None
    assert ws([False] * 10) is None
    m = [False] * 100 + [True] * 387 + [False] * 10
    assert ws(m) == [(0, 100, False), (100, 497, True)]
    assert ws([False] * 20 + [True] * 387) == [(0, 407, True)]
    assert ws([True] * 387 + [False] * 30 + [True] * 387) == [(0, 804, True)]
    assert ws([False] * 65 + [True] * 10) == [(0, 65, False), (65, 75, True)]
    rng = np.random.default_rng(4)
    for _ in range(20):
        mask = (rng.random(int(rng.integers(1, 400))) < 0.3).tolist()
        segments = ws(mask)
        if segments is None:
            assert not any(mask)
            continue
        assert segments[0][0] == 0 and segments[-1][1] == len(mask)
        for (s0, e0, f0), (s1, e1, f1) in zip(segments, segments[1:]):
            assert e0 == s1 and f0 != f1
        for s, e, flag in segments:
            assert s < e
            if flag:
                # every wide row is inside a wide segment
                continue
            assert e - s >= ia.ORCH_MIN_ROWS
            assert not any(mask[s:e])
        wide_rows = {i for i, w in enumerate(mask) if w}
        covered = {i for s, e, flag in segments if flag for i in range(s, e)}
        assert wide_rows <= covered


# --------------------------------------------------------------------- (g)
def test_constants_and_index_width():
    assert ia.IMAGE_START == ours_ip.IMAGE_START == 0
    assert ia.IMAGE == ours_ip.IMAGE == 2
    assert ia.IMAGE_NEW_LINE == ours_ip.IMAGE_NEW_LINE == 3
    assert ia.IMAGE_END == ours_ip.IMAGE_END == 4
    cfg = SimpleNamespace(sliding_window=128, vision_max_n_token=384)
    assert ia.vision_index_width(cfg) == 512
    assert ia.vision_index_width(cfg) in ia.DECODE_WIDTHS


@requires_materials
def test_constants_match_reference(ref_ip):
    assert int(ref_ip.IMAGE_START) == ia.IMAGE_START
    assert int(ref_ip.IMAGE) == ia.IMAGE
    assert int(ref_ip.IMAGE_NEW_LINE) == ia.IMAGE_NEW_LINE
    assert int(ref_ip.IMAGE_END) == ia.IMAGE_END


def test_vision_bidi_enabled_env(monkeypatch):
    vision_cfg = SimpleNamespace(vision_n_layers=32)
    text_cfg = SimpleNamespace(vision_n_layers=0)
    monkeypatch.delenv("DSPARK_VISION_BIDI", raising=False)
    assert ia.vision_bidi_enabled(vision_cfg)
    assert not ia.vision_bidi_enabled(text_cfg)
    monkeypatch.setenv("DSPARK_VISION_BIDI", "0")
    assert not ia.vision_bidi_enabled(vision_cfg)
    monkeypatch.setenv("DSPARK_VISION_BIDI", "1")
    assert ia.vision_bidi_enabled(vision_cfg)
    assert not ia.vision_bidi_enabled(text_cfg)


def test_slots_from_positions():
    rows = torch.tensor([[0, 1, 63, 64, -1]], dtype=torch.int32)
    table = torch.tensor([7, 9], dtype=torch.int32)
    expected = [[448, 449, 511, 576, -1]]
    assert ia.slots_from_positions(rows, table, 64).tolist() == expected
    assert ia.slots_from_positions(rows, table.unsqueeze(0), 64).tolist() == expected
