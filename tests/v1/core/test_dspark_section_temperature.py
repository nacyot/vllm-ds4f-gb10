# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side tool-turn section temperature (DSPARK_TOOL_TEMP0_SCOPE).

The tracker rides on ``DsparkLoopBreak``'s reasoning-section state. Before the
scope's trigger fires nothing is emitted (the worker already holds
``sampling_params.temperature``); after it the answer temperature is re-emitted
on every step. Scope ``dsml`` (the default) triggers on the tool-call marker
alone and latches; scope ``answer`` triggers at any end of the reasoning
section and switches back when a new one opens.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.dspark_loop_break import (
    DSPARK_ANSWER_TEMP_KEY,
    DSPARK_REASONING_TEMP_KEY,
    DSPARK_SCOPE_KEY,
    DsparkLoopBreak,
    dspark_section_temperature_enabled,
    dspark_temp0_scope,
)

THINK_START = 100
THINK_END = 200
TRANSITION = 201
DSML_A = 300
DSML_B = 301


def _make(dsml=True, natural=None, end=None, default_scope="answer"):
    end = end or [THINK_END]
    return DsparkLoopBreak(
        start_ids=[THINK_START],
        end_ids=end,
        natural_end_ids=natural or end,
        # Section tracking only: loop breaking is unconfigured.
        params=None,
        min_reasoning_tokens=32,
        check_interval=4,
        dsml_ids=[DSML_A, DSML_B] if dsml else None,
        default_scope=default_scope,
    )


class _Req:
    def __init__(
        self,
        rid="r",
        prompt=None,
        answer_temp=0.0,
        reasoning_temp=1.0,
        scope="answer",
    ):
        self.request_id = rid
        self.prompt_token_ids = prompt if prompt is not None else [1, 2, 3]
        self.output_token_ids: list[int] = []
        extra_args = None
        if answer_temp is not None:
            extra_args = {
                DSPARK_ANSWER_TEMP_KEY: answer_temp,
                DSPARK_REASONING_TEMP_KEY: reasoning_temp,
            }
            if scope is not None:
                extra_args[DSPARK_SCOPE_KEY] = scope
        self.sampling_params = SimpleNamespace(
            thinking_loop_break=None,
            thinking_token_budget=None,
            temperature=reasoning_temp,
            extra_args=extra_args,
        )


def _steps(lb, req, tokens, chunk=6):
    """Feed ``tokens`` in <= ``chunk``-token steps (the decode step size with
    speculative decoding) and return the per-step pending_temp snapshots."""
    seen = []
    for i in range(0, len(tokens), chunk):
        pending_temp: dict[str, float] = {}
        req.output_token_ids.extend(tokens[i : i + chunk])
        lb.observe(req, {}, pending_temp)
        seen.append(pending_temp)
    return seen


def _filler(n, base=10):
    return [base + i for i in range(n)]


# ----------------------------------------------------------------------
# scope "answer"


def test_prompt_outside_think_emits_immediately():
    lb = _make()
    req = _Req(prompt=[1, 2, 3])
    seen = _steps(lb, req, _filler(12))
    assert seen[0] == {"r": 0.0}
    assert lb.num_answer_temp_switches == 1
    assert lb._state["r"]["end_reason"] == "prompt"


def test_inside_think_emits_nothing_until_natural_end():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START])
    seen = _steps(lb, req, _filler(30))
    assert all(s == {} for s in seen)
    # </think> lands in the middle of a multi-token append.
    seen = _steps(lb, req, [7, 8, THINK_END, 9, 10, 11])
    assert seen == [{"r": 0.0}]
    assert lb.num_answer_temp_switches == 1
    assert lb._state["r"]["end_reason"] == "natural"


def test_reentering_think_restores_the_reasoning_temperature():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_END])
    # Prompt tail is outside a section, so the first step already switches.
    assert _steps(lb, req, _filler(3)) == [{"r": 0.0}]
    # Interleaved thinking: a new <think> hands the reasoning temperature back.
    assert _steps(lb, req, [THINK_START, 5, 6]) == [{"r": 1.0}]
    assert _steps(lb, req, _filler(6, base=50)) == [{}]
    assert _steps(lb, req, [THINK_END]) == [{"r": 0.0}]
    assert lb.num_answer_temp_switches == 1  # one log line per request


def test_forced_end_sequence_closes_the_section():
    lb = _make(natural=[THINK_END], end=[TRANSITION, THINK_END])
    req = _Req(prompt=[1, 2, THINK_START])
    assert _steps(lb, req, _filler(12)) == [{}, {}]
    # The forcing kernel emits the transition phrase + </think>; the pair may
    # be split across two steps.
    assert _steps(lb, req, [5, 5, 5, 5, 5, TRANSITION]) == [{}]
    assert _steps(lb, req, [THINK_END]) == [{"r": 0.0}]
    assert lb._state["r"]["end_reason"] == "forced"


def test_answer_scope_switches_on_the_dsml_marker_inside_think():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START])
    assert _steps(lb, req, _filler(12)) == [{}, {}]
    # The model opens a tool call without emitting </think>. The marker's
    # first token alone (a plain "<") must not latch; the full two-token
    # marker does.
    assert _steps(lb, req, [5, 6, 7, 8, 9, DSML_A]) == [{}]
    assert _steps(lb, req, [DSML_B, 11, 12]) == [{"r": 0.0}]
    assert lb.num_answer_temp_switches == 1
    assert lb._state["r"]["end_reason"] == "dsml"


def test_dsml_detection_off_without_marker_ids():
    lb = _make(dsml=False)
    req = _Req(prompt=[1, 2, THINK_START])
    assert _steps(lb, req, _filler(12) + [DSML_A, DSML_B, 13, 14]) == [{}, {}, {}]


def test_answer_temperature_is_reemitted_every_step():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START])
    _steps(lb, req, _filler(12) + [THINK_END])
    seen = _steps(lb, req, _filler(18, base=400))
    assert seen == [{"r": 0.0}, {"r": 0.0}, {"r": 0.0}]
    # ... but only one log line / one counted switch per request.
    assert lb.num_answer_temp_switches == 1


# ----------------------------------------------------------------------
# scope "dsml"


def test_dsml_scope_ignores_the_end_of_reasoning():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START], scope="dsml")
    # Reasoning, </think>, and a long answer: none of it switches. This is the
    # answer prose that ran away to max_tokens under whole-request temp 0.
    seen = _steps(lb, req, _filler(12) + [THINK_END] + _filler(24, base=500))
    assert all(s == {} for s in seen)
    assert lb.num_answer_temp_switches == 0


def test_dsml_scope_switches_on_the_marker_after_think_end():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START], scope="dsml")
    assert _steps(lb, req, _filler(12) + [THINK_END]) == [{}, {}, {}]
    assert _steps(lb, req, [5, 6, DSML_A, DSML_B, 7, 8]) == [{"r": 0.0}]
    assert _steps(lb, req, _filler(6, base=700)) == [{"r": 0.0}]
    assert lb.num_answer_temp_switches == 1
    assert lb._state["r"]["end_reason"] == "dsml"


def test_dsml_scope_switches_on_the_marker_inside_think():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START], scope="dsml")
    assert _steps(lb, req, _filler(12)) == [{}, {}]
    assert _steps(lb, req, [5, 6, 7, DSML_A, DSML_B, 8]) == [{"r": 0.0}]


def test_dsml_scope_latches_across_a_new_think_section():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START], scope="dsml")
    _steps(lb, req, _filler(6) + [THINK_END, DSML_A, DSML_B])
    # A second tool-call block or interleaved thinking does not switch back.
    assert _steps(lb, req, [THINK_START, 5, 6]) == [{"r": 0.0}]
    assert _steps(lb, req, _filler(6, base=800)) == [{"r": 0.0}]
    assert lb.num_answer_temp_switches == 1


def test_dsml_scope_single_marker_token_does_not_latch():
    # The real marker starts with a plain "<" token: a one-token prefix must
    # never latch (it would fire on any "<" in prose or code).
    lb = _make()
    req = _Req(prompt=[1, 2, 3], scope="dsml")
    assert _steps(lb, req, [5, 6, 7, 8, 9, DSML_A]) == [{}]
    assert _steps(lb, req, [DSML_B]) == [{"r": 0.0}]


def test_dsml_scope_partial_marker_latches_one_step_early():
    lb = _make()
    lb.dsml_ids = [DSML_A, DSML_B, 302]
    lb._max_marker = max(lb._max_marker, 3)
    req = _Req(prompt=[1, 2, 3], scope="dsml")
    # Two of three marker tokens have landed (the "<" and the DSML special
    # token); the switch is already in force for the next step, which is
    # where "invoke name=\"" would be sampled.
    assert _steps(lb, req, [5, 6, 7, 8, DSML_A, DSML_B]) == [{"r": 0.0}]


def test_dsml_scope_never_switches_without_a_tool_call():
    lb = _make()
    req = _Req(prompt=[1, 2, 3], scope="dsml")
    seen = _steps(lb, req, _filler(36))
    assert all(s == {} for s in seen)
    assert lb.num_answer_temp_switches == 0


def test_scope_falls_back_to_the_tracker_default():
    lb = _make(default_scope="dsml")
    req = _Req(prompt=[1, 2, THINK_START], scope=None)
    assert DSPARK_SCOPE_KEY not in req.sampling_params.extra_args
    assert _steps(lb, req, _filler(12) + [THINK_END] + _filler(6)) == [{}, {}, {}, {}]
    assert _steps(lb, req, [DSML_A, DSML_B]) == [{"r": 0.0}]


# ----------------------------------------------------------------------
# common


def test_requests_without_the_extra_args_key_are_ignored():
    lb = _make()
    req = _Req(answer_temp=None)
    assert req.sampling_params.extra_args is None
    assert _steps(lb, req, _filler(12) + [THINK_END] + _filler(6)) == [{}, {}, {}, {}]
    assert lb.num_answer_temp_switches == 0

    other = _Req(rid="s")
    other.sampling_params.extra_args = {"kv_transfer_params": {"a": 1}}
    assert _steps(lb, other, _filler(6)) == [{}]


def test_forget_clears_state():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START])
    _steps(lb, req, _filler(12) + [THINK_END])
    assert "r" in lb._state
    lb.forget("r")
    assert "r" not in lb._state
    # A fresh admission re-seeds from the prompt tail: still inside <think>.
    req.output_token_ids = []
    assert _steps(lb, req, _filler(6)) == [{}]


def test_loop_break_opt_out_does_not_disable_the_temperature_channel():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START])
    req.sampling_params.thinking_loop_break = False
    _steps(lb, req, _filler(12))
    assert _steps(lb, req, [THINK_END]) == [{"r": 0.0}]


@pytest.mark.parametrize(
    "value,scope,splits",
    [
        (None, "dsml", True),
        ("", "dsml", True),
        ("dsml", "dsml", True),
        ("DSML", "dsml", True),
        (" Answer ", "answer", True),
        ("request", "request", False),
        ("nonsense", "request", False),
    ],
)
def test_env_gate(monkeypatch, value, scope, splits):
    monkeypatch.delenv("DSPARK_TOOL_TEMP0", raising=False)
    monkeypatch.delenv("DSPARK_TOOL_TEMP0_SCOPE", raising=False)
    if value is not None:
        monkeypatch.setenv("DSPARK_TOOL_TEMP0_SCOPE", value)
    assert dspark_temp0_scope() == scope
    assert not dspark_section_temperature_enabled()  # DSPARK_TOOL_TEMP0 unset
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    assert dspark_section_temperature_enabled() is splits


def test_from_config_builds_tracker_without_loop_break_params(monkeypatch):
    cfg = SimpleNamespace(
        reasoning_config=SimpleNamespace(
            enabled=True,
            loop_break_max_pattern_size=0,
            loop_break_min_count=0,
            loop_break_min_pattern_size=0,
            reasoning_start_token_ids=[THINK_START],
            reasoning_end_token_ids=[THINK_END],
        ),
        model_config=None,
    )
    monkeypatch.delenv("DSPARK_TOOL_TEMP0", raising=False)
    monkeypatch.delenv("DSPARK_TOOL_TEMP0_SCOPE", raising=False)
    assert DsparkLoopBreak.from_config(cfg) is None
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    lb = DsparkLoopBreak.from_config(cfg)
    # model_config=None makes the tokenizer lookup fail: under scope dsml that
    # leaves the scope inert (warned about), everything else still works.
    assert lb is not None and lb.params is None and lb.dsml_ids == []
    assert lb.default_scope == "dsml"
    monkeypatch.setenv("DSPARK_TOOL_TEMP0_SCOPE", "request")
    assert DsparkLoopBreak.from_config(cfg) is None


def test_from_config_warns_when_reasoning_is_unconfigured(monkeypatch, caplog):
    monkeypatch.setenv("DSPARK_TOOL_TEMP0", "1")
    monkeypatch.delenv("DSPARK_TOOL_TEMP0_SCOPE", raising=False)
    cfg = SimpleNamespace(reasoning_config=None, model_config=None)
    with caplog.at_level("WARNING"):
        assert DsparkLoopBreak.from_config(cfg) is None
    assert "inert" in caplog.text
