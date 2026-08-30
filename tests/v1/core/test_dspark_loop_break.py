# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side reasoning loop breaker (vLLM PR #52677 port for V2)."""

from types import SimpleNamespace

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.core.sched.dspark_loop_break import DsparkLoopBreak

THINK_START = 100
THINK_END = 200
TRANSITION = 201


def _make(min_reasoning=32, interval=4, natural=None, end=None):
    end = end or [THINK_END]
    return DsparkLoopBreak(
        start_ids=[THINK_START],
        end_ids=end,
        natural_end_ids=natural or end,
        params=RepetitionDetectionParams(
            max_pattern_size=8, min_pattern_size=2, min_count=3
        ),
        min_reasoning_tokens=min_reasoning,
        check_interval=interval,
    )


class _Req:
    def __init__(self, rid="r", prompt=None, loop_break=None, budget=None):
        self.request_id = rid
        self.prompt_token_ids = prompt or [1, 2, 3]
        self.output_token_ids: list[int] = []
        self.sampling_params = SimpleNamespace(
            thinking_loop_break=loop_break, thinking_token_budget=budget
        )


def _feed(lb, req, pending, tokens, chunk=4):
    fired = False
    for i in range(0, len(tokens), chunk):
        req.output_token_ids.extend(tokens[i : i + chunk])
        fired |= lb.observe(req, pending)
    return fired


def _non_periodic(n, base=10):
    return [base + i for i in range(n)]


def test_fires_on_exact_loop_and_reports_once():
    lb = _make()
    req = _Req()
    pending: dict[str, int] = {}
    cycle = [7, 8, 9]
    assert _feed(lb, req, pending, [THINK_START] + _non_periodic(40) + cycle * 12)
    assert list(pending) == ["r"] and pending["r"] >= 32
    # keeps looping: no second report while the section is open
    _feed(lb, req, pending, cycle * 8)
    assert list(pending) == ["r"] and pending["r"] >= 32
    assert lb.num_fired == 1


def test_no_fire_on_non_periodic_output():
    lb = _make()
    req = _Req()
    pending: dict[str, int] = {}
    assert not _feed(lb, req, pending, [THINK_START] + _non_periodic(300))
    assert pending == {}


def test_no_fire_below_reasoning_floor_then_fires():
    lb = _make()
    req = _Req()
    pending: dict[str, int] = {}
    cycle = [7, 8, 9]
    assert not _feed(lb, req, pending, [THINK_START] + cycle * 8)  # 24 < 32
    assert _feed(lb, req, pending, cycle * 4)
    assert list(pending) == ["r"] and pending["r"] >= 32


def test_no_fire_outside_reasoning_section():
    lb = _make()
    req = _Req()
    pending: dict[str, int] = {}
    cycle = [7, 8, 9]
    assert not _feed(lb, req, pending, [THINK_START, 11, THINK_END] + cycle * 30)
    assert pending == {}


def test_section_end_rearms_detection():
    lb = _make()
    req = _Req()
    pending: dict[str, int] = {}
    cycle = [7, 8, 9]
    assert _feed(lb, req, pending, [THINK_START] + _non_periodic(40) + cycle * 12)
    _feed(lb, req, pending, [THINK_END], chunk=1)
    assert _feed(
        lb, req, pending, [THINK_START] + _non_periodic(40, base=1000) + cycle * 12
    )
    assert list(pending) == ["r"]


def test_natural_end_closes_section_with_transition_phrase():
    lb = _make(natural=[THINK_END], end=[TRANSITION, THINK_END])
    req = _Req()
    pending: dict[str, int] = {}
    _feed(lb, req, pending, [THINK_START] + _non_periodic(40) + [THINK_END])
    assert not _feed(lb, req, pending, [7, 8, 9] * 30)
    assert pending == {}


def test_section_started_in_prompt():
    lb = _make()
    req = _Req(prompt=[1, 2, THINK_START])
    pending: dict[str, int] = {}
    cycle = [7, 8, 9]
    assert _feed(lb, req, pending, _non_periodic(40) + cycle * 12)
    assert list(pending) == ["r"] and pending["r"] >= 32


def test_per_request_opt_out():
    lb = _make()
    req = _Req(loop_break=False)
    pending: dict[str, int] = {}
    cycle = [7, 8, 9]
    assert not _feed(lb, req, pending, [THINK_START] + _non_periodic(40) + cycle * 12)
    assert pending == {}


def test_from_config_disabled_without_params():
    cfg = SimpleNamespace(
        reasoning_config=SimpleNamespace(
            enabled=True,
            loop_break_max_pattern_size=0,
            loop_break_min_count=0,
            reasoning_start_token_ids=[THINK_START],
            reasoning_end_token_ids=[THINK_END],
        )
    )
    assert DsparkLoopBreak.from_config(cfg) is None
    cfg.reasoning_config.loop_break_max_pattern_size = 8
    cfg.reasoning_config.loop_break_min_count = 3
    cfg.reasoning_config.loop_break_min_pattern_size = 2
    assert DsparkLoopBreak.from_config(cfg) is not None


def test_budget_hit_is_counted_once_per_section():
    lb = _make()
    req = _Req(budget=40)
    pending: dict[str, int] = {}
    _feed(lb, req, pending, [THINK_START] + _non_periodic(60))
    assert lb.num_budget_hits == 1
    _feed(lb, req, pending, _non_periodic(20, base=500))
    assert lb.num_budget_hits == 1  # not re-counted while the section is open
    _feed(lb, req, pending, [THINK_END] + [THINK_START] + _non_periodic(60, base=900))
    assert lb.num_budget_hits == 2  # a new section can hit again
    assert pending == {}  # budget hits are not loop verdicts
