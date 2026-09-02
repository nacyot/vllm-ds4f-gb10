# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSML tool-call block vs the answer-repetition detector, production shape.

Real DeepSeek-V4 token ids (live tokenizer, 2026-09-02):
  "<think>" -> [128821], "</think>" -> [128822]
  "<｜DSML｜tool_calls>" -> [30, 128825, 72461, 4941, 12548, 32]
  "<｜DSML｜invoke"      -> [30, 128825, 40148, 5406]
  " <｜DSML｜tool_calls>" -> [818, 128825, ...]   (the "<" merges with a space)
  ".<｜DSML｜tool_calls>" -> [32334, 128825, ...]
The tracker keys the markers on the ｜DSML｜ special token (leading "<" dropped
by ``_dsml_token_ids``), the reasoning_end_str transition phrase is 14 tokens
ending in 128822, DSpark steps carry 1..6 tokens.
"""

import random

import pytest

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.core.sched.dspark_loop_break import DsparkLoopBreak

THINK_START = 128821
THINK_END = 128822
END_SEQ = [
    43,
    611,
    304,
    3475,
    270,
    4630,
    2951,
    377,
    270,
    22805,
    6578,
    1928,
    16,
    THINK_END,
]
LT = 30  # plain "<"
DSML_TOOL = [128825, 72461, 4941, 12548, 32]  # after the leading "<" is stripped
DSML_INVOKE = [128825, 40148, 5406]
DSML_PARAMETER = [128825, 41523]
DSML_TOOL_CLOSE = [128825, 9543, 1458, 4941, 12548, 32]
WINDOW = 1024


def _make(ans_dsml_min=0.0):
    return DsparkLoopBreak(
        start_ids=[THINK_START],
        end_ids=END_SEQ,
        natural_end_ids=[THINK_END],
        params=RepetitionDetectionParams(
            max_pattern_size=128, min_pattern_size=8, min_count=4
        ),
        min_reasoning_tokens=512,
        check_interval=16,
        dsml_ids=[DSML_TOOL, DSML_INVOKE],
        novelty={
            "window": WINDOW,
            "min": 0.4,
            "ngram": 8,
            "consecutive": 2,
            "answer_window": WINDOW,
            "answer_min": 0.3,
            "answer_dsml_min": ans_dsml_min,
        },
    )


class _Req:
    def __init__(self, rid="r"):
        self.request_id = rid
        # the chat template opened <think> in the prompt; the model closes it
        self.prompt_token_ids = [1, 2, THINK_START]
        self.output_token_ids: list[int] = []
        self.sampling_params = type(
            "SP",
            (),
            {
                "thinking_loop_break": None,
                "thinking_token_budget": 24576,
                "extra_args": None,
                "temperature": 1.0,
            },
        )()


def _feed(lb, req, tokens, steps):
    """``steps``: per-step token counts (1 + up to 5 accepted DSpark draft
    tokens), cycled over the stream."""
    pending: dict[str, int] = {}
    pending_temp: dict[str, float] = {}
    i, k = 0, 0
    while i < len(tokens):
        n = steps[k % len(steps)]
        k += 1
        req.output_token_ids.extend(tokens[i : i + n])
        i += n
        lb.observe(req, pending, pending_temp)
    return pending


def _prose(n, seed=0):
    rnd = random.Random(seed)
    return [1000 + rnd.randrange(50000) for _ in range(n)]


def _patch_payload(n_lines=60, line_len=40, seed=1):
    """A code patch: many near-identical lines (small vocabulary, repeating
    8-grams), novelty well under 0.3 at window 1024 but not an exact cycle."""
    rnd = random.Random(seed)
    base = [2000 + rnd.randrange(30) for _ in range(line_len)]
    out = []
    for _ in range(n_lines):
        out += [t if rnd.random() > 0.05 else 2000 + rnd.randrange(30) for t in base]
    return out


def _stream(prefix_len, opener=(LT,)):
    # reasoning (natural </think>), answer prose, the tool call with a patch
    return (
        _prose(300, seed=7)
        + [THINK_END]
        + _prose(prefix_len, seed=8)
        + list(opener)
        + DSML_TOOL
        + [LT]
        + DSML_INVOKE
        + [2329, 1281, 30394, 5224, 2471, 3320]  # name="apply_patch">
        + _patch_payload()
    )


STEP_PATTERNS = [[1], [6], [3, 1, 6, 2, 5, 4, 1, 6]]


@pytest.mark.parametrize("steps", STEP_PATTERNS)
@pytest.mark.parametrize("prefix_len", list(range(200, 206)))  # all marker alignments
def test_tool_call_payload_is_not_cut(steps, prefix_len):
    lb = _make()
    req = _Req()
    pending = _feed(lb, req, _stream(prefix_len), steps)
    st = lb._state["r"]
    assert st["in_think"] is False and st["dsml_seen"] is True
    assert st["ans_nov"] is None  # the block is not even fed at the default
    assert lb.take_answer_repetition("r") is None
    assert lb.num_answer_repetition == 0 and lb.num_dsml_blocks == 1
    assert pending == {}  # no forced reasoning end either


@pytest.mark.parametrize("opener", [(818,), (32334,), (49584,), (201, LT), (4, LT)])
def test_opener_after_space_period_paren_quote_or_newline_is_recognised(opener):
    # " <", ".<", ")<" merge into one token; "\n<" and '"<' keep the bare "<".
    lb = _make()
    req = _Req()
    _feed(lb, req, _stream(200, opener=opener), [3, 1, 6, 2, 5, 4, 1, 6])
    assert lb._state["r"]["dsml_seen"] is True
    assert lb.take_answer_repetition("r") is None


def test_same_payload_as_prose_is_cut():
    # control: without the marker the identical payload fires, so the tests
    # above pass because of the marker and not because of the data
    lb = _make()
    req = _Req()
    stream = _prose(300, seed=7) + [THINK_END] + _prose(200, seed=8) + _patch_payload()
    _feed(lb, req, stream, [3, 1, 6, 2, 5, 4, 1, 6])
    reason = lb.take_answer_repetition("r")
    assert reason is not None and "section=" not in reason
    assert lb.num_answer_repetition == 1


def test_opt_in_threshold_scores_only_the_block_and_tags_the_section():
    lb = _make(ans_dsml_min=0.29)
    req = _Req()
    _feed(lb, req, _stream(200), [6])
    reason = lb.take_answer_repetition("r")
    assert reason is not None and reason.endswith(",section=dsml)")


def test_bare_invoke_marker_after_think_end_is_recognised():
    lb = _make()
    req = _Req()
    stream = (
        _prose(300, seed=7)
        + [THINK_END]
        + _prose(200, seed=8)
        + [LT]
        + DSML_INVOKE
        + _patch_payload()
    )
    _feed(lb, req, stream, [4, 1, 2, 6])
    assert lb._state["r"]["dsml_seen"] is True
    assert lb.take_answer_repetition("r") is None


@pytest.mark.parametrize("stray", [[LT], [LT] + DSML_PARAMETER, [LT] + DSML_TOOL_CLOSE])
def test_other_dsml_constructs_and_a_bare_lt_do_not_latch(stray):
    # a plain "<", a parameter tag or a closer in prose is not an opener
    lb = _make()
    req = _Req()
    stream = (
        _prose(300, seed=7)
        + [THINK_END]
        + _prose(100, seed=8)
        + stray
        + _patch_payload()
    )
    _feed(lb, req, stream, [6])
    assert lb._state["r"]["dsml_seen"] is False
    assert lb.take_answer_repetition("r") is not None


def test_in_think_tool_call_then_novel_answer_is_not_cut():
    # the model opens the tool call inside <think>, the thinking budget forces
    # the transition phrase, then a novel answer follows
    lb = _make()
    req = _Req()
    stream = (
        _prose(600, seed=3)
        + [LT]
        + DSML_TOOL
        + _patch_payload(n_lines=40)
        + END_SEQ
        + _prose(400, seed=4)
    )
    pending = _feed(lb, req, stream, [3, 1, 6, 2, 5, 4, 1, 6])
    st = lb._state["r"]
    assert st["in_think"] is False and st["dsml_seen"] is False
    assert lb.take_answer_repetition("r") is None and lb.num_answer_repetition == 0
    assert pending == {}
