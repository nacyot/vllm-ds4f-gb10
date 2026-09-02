# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK DSML leak guard: scoped to the answer section (2026-09-02)."""

from types import SimpleNamespace

import pytest

from vllm.v1.engine import output_processor as op


class _Det:
    def __init__(self):
        self.output_token_ids: list[int] = []
        self.output_text = ""

    def feed(self, ids: list[int], text: str):
        self.output_token_ids.extend(ids)
        self.output_text += text


def _proc(start=(11,), end=(12,)):
    proc = op.OutputProcessor(None, log_stats=False)
    proc._dspark_think_start_ids = list(start)
    proc._dspark_think_end_ids = list(end)
    return proc


def _state(in_reasoning: bool):
    return SimpleNamespace(
        detokenizer=_Det(), dspark_in_reasoning=in_reasoning, dspark_answer_text_start=0
    )


def test_prompt_in_think_by_ids_and_text():
    proc = _proc()
    assert proc._dspark_prompt_in_think([1, 2, 11], None) is True
    assert proc._dspark_prompt_in_think([1, 11, 5, 12], None) is False
    assert proc._dspark_prompt_in_think([1, 12, 5, 11], None) is True
    assert proc._dspark_prompt_in_think(None, "hi<｜Assistant｜><think>") is True
    assert proc._dspark_prompt_in_think(None, "<think>x</think>") is False
    assert proc._dspark_prompt_in_think(None, None) is False


def test_bare_closer_ignored_inside_reasoning_then_caught_after_end():
    proc = _proc()
    st = _state(True)
    st.detokenizer.feed([3, 4], 'the parser expects "</invoke>" here')
    assert proc._dspark_leak_guard_check(st, [3, 4]) is None
    assert st.dspark_in_reasoning is True
    # </think> arrives mid-step, answer text follows in the same step
    st.detokenizer.feed([12, 7], "</think>ok")
    assert proc._dspark_leak_guard_check(st, [12, 7]) is None
    assert st.dspark_in_reasoning is False
    st.detokenizer.feed([8], " </result>")
    assert proc._dspark_leak_guard_check(st, [8]) == "bare_closer:</result>"


def test_dialect_token_ignored_inside_reasoning():
    proc = _proc()
    st = _state(True)
    st.detokenizer.feed([128840], "<dsml:")
    assert proc._dspark_leak_guard_check(st, [128840]) is None
    st.dspark_in_reasoning = False
    st.detokenizer.feed([128841], "</dsml:")
    assert proc._dspark_leak_guard_check(st, [128841]) == "dsml_dialect_token"


def test_text_fallback_when_end_ids_unknown():
    proc = _proc(start=(), end=())
    st = _state(True)
    st.detokenizer.feed([1], "quoting </parameter> is fine")
    assert proc._dspark_leak_guard_check(st, [1]) is None
    st.detokenizer.feed([2], "</think>")
    assert proc._dspark_leak_guard_check(st, [2]) is None
    assert st.dspark_in_reasoning is False


def test_not_in_reasoning_from_start_behaves_like_before():
    proc = _proc()
    st = _state(False)
    st.detokenizer.feed([1], "</parameter>")
    assert proc._dspark_leak_guard_check(st, [1]) == "bare_closer:</parameter>"


@pytest.mark.parametrize("tail", ["</invoke>", "</parameter>", "</result>"])
def test_all_bare_closers(tail):
    proc = _proc()
    st = _state(False)
    st.detokenizer.feed([1], "x" + tail)
    assert proc._dspark_leak_guard_check(st, [1]) == f"bare_closer:{tail}"


def test_tool_call_opener_is_an_implicit_reasoning_end():
    proc = _proc()
    proc._dspark_dsml_start_ids = [30, 128825, 72461, 4941, 12548, 32]
    st = _state(True)
    st.detokenizer.feed([3], 'quoting "</invoke>" in thought')
    assert proc._dspark_leak_guard_check(st, [3]) is None
    # the model opens a tool call without </think>; the block itself is armed
    st.detokenizer.feed([30, 128825, 72461, 4941, 12548, 32], "<｜DSML｜tool_calls>")
    assert (
        proc._dspark_leak_guard_check(st, [30, 128825, 72461, 4941, 12548, 32]) is None
    )
    assert st.dspark_in_reasoning is False
    st.detokenizer.feed([7], "\n</invoke>")
    assert proc._dspark_leak_guard_check(st, [7]) == "bare_closer:</invoke>"


def test_dialect_ids_resolved_from_tokenizer():
    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return {"<dsml:": [555], "</dsml:": [556]}[text]

    assert op._dspark_dialect_token_ids(_Tok()) == frozenset({555, 556})
    assert op._dspark_dialect_token_ids(None) == op._DSPARK_DSML_DIALECT_TOKEN_IDS
