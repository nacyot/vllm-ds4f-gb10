# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK near-repetition detectors wired into the scheduler loop breaker."""

import random
from types import SimpleNamespace

from vllm.v1.core.sched.dspark_loop_break import DsparkLoopBreak

THINK_START, THINK_END = 900, 901


def _make(window=256, ans_window=256, consecutive=2, exact=False):
    params = None
    if exact:
        from vllm.sampling_params import RepetitionDetectionParams

        params = RepetitionDetectionParams(
            max_pattern_size=8, min_pattern_size=2, min_count=3
        )
    return DsparkLoopBreak(
        start_ids=[THINK_START],
        end_ids=[THINK_END],
        natural_end_ids=[THINK_END],
        params=params,
        min_reasoning_tokens=32,
        check_interval=4,
        novelty={
            "window": window,
            "min": 0.4,
            "ngram": 4,
            "consecutive": consecutive,
            "answer_window": ans_window,
            "answer_min": 0.3,
        },
    )


class _Req:
    def __init__(self, rid="r", in_think=True, budget=None):
        self.request_id = rid
        self.prompt_token_ids = [1, 2, THINK_START] if in_think else [1, 2]
        self.output_token_ids: list[int] = []
        self.sampling_params = SimpleNamespace(
            thinking_loop_break=None, thinking_token_budget=budget, extra_args=None
        )


def _feed(lb, req, pending, tokens, chunk=6):
    fired = False
    for i in range(0, len(tokens), chunk):
        req.output_token_ids.extend(tokens[i : i + chunk])
        fired |= lb.observe(req, pending)
    return fired


def _paraphrase_loop(n_rounds=8, length=200, seed=0):
    rnd = random.Random(seed)
    base = [1000 + rnd.randrange(5000) for _ in range(length)]
    out = list(base)
    for _ in range(n_rounds):
        out += [t if rnd.random() > 0.03 else 1000 + rnd.randrange(5000) for t in base]
    return out


def test_reasoning_paraphrase_loop_forces_end():
    lb = _make()
    req, pending = _Req(), {}
    assert _feed(lb, req, pending, _paraphrase_loop()) is True
    # fires as soon as the window collapses, well before the stream ends
    assert 256 <= pending["r"] <= len(req.output_token_ids)
    assert lb.num_fired_novelty == 1 and lb.num_fired == 1


def test_reasoning_novel_stream_never_fires():
    lb = _make()
    req, pending = _Req(), {}
    assert _feed(lb, req, pending, [10 + i for i in range(3000)]) is False
    assert pending == {} and lb.num_fired == 0


def test_reasoning_fires_once_per_section_and_rearms():
    lb = _make()
    req, pending = _Req(), {}
    _feed(lb, req, pending, _paraphrase_loop())
    assert lb.num_fired == 1
    _feed(lb, req, pending, _paraphrase_loop(seed=1))
    assert lb.num_fired == 1  # latched for this section
    _feed(lb, req, pending, [THINK_END, 5, 6, THINK_START])  # new section
    _feed(lb, req, pending, _paraphrase_loop(seed=2))
    assert lb.num_fired == 2


def test_answer_exact_cycle_finishes_request():
    lb = _make()
    req, pending = _Req(in_think=False), {}
    _feed(lb, req, pending, [10 + i for i in range(100)])
    assert lb.take_answer_repetition("r") is None
    _feed(lb, req, pending, list(range(300, 600)) * 4)  # period 300 > any cap
    reason = lb.take_answer_repetition("r")
    assert reason is not None and reason.startswith("dspark_answer_repetition(")
    assert lb.take_answer_repetition("r") is None  # popped once
    assert lb.num_answer_repetition == 1
    assert pending == {}  # answer loops do not force a reasoning end


def test_answer_after_think_section():
    lb = _make()
    req, pending = _Req(in_think=True), {}
    _feed(lb, req, pending, [10 + i for i in range(200)] + [THINK_END])
    _feed(lb, req, pending, list(range(300, 500)) * 5)
    assert lb.take_answer_repetition("r") is not None


def test_answer_detector_off_when_window_zero():
    lb = _make(ans_window=0)
    req, pending = _Req(in_think=False), {}
    _feed(lb, req, pending, list(range(300, 600)) * 4)
    assert lb.take_answer_repetition("r") is None


def test_budget_hit_logged_when_forced_end_lands_in_same_step():
    lb = _make(window=0, ans_window=0)
    req, pending = _Req(budget=64), {}
    # 62 reasoning tokens, then the crossing token and the forced </think>
    # arrive in ONE multi-token step (the DSpark case).
    _feed(lb, req, pending, [10 + i for i in range(60)], chunk=6)
    assert lb.num_budget_hits == 0
    req.output_token_ids.extend([70, 71, 72, 73, THINK_END, 5])
    lb.observe(req, pending)
    assert lb.num_budget_hits == 1
    assert req.output_token_ids[-2] == THINK_END


def test_forget_clears_pending_finish():
    lb = _make()
    req, pending = _Req(in_think=False), {}
    _feed(lb, req, pending, list(range(300, 600)) * 4)
    lb.forget("r")
    assert lb.take_answer_repetition("r") is None


def test_no_forced_end_while_dsml_block_is_written_inside_think():
    lb = _make()
    lb.dsml_markers = [[800, 801]]
    lb.dsml_ids = [800, 801]
    lb._max_marker = max(lb._max_marker, 2)
    req, pending = _Req(), {}
    _feed(lb, req, pending, [10 + i for i in range(100)] + [800, 801])
    assert lb._state["r"]["dsml_seen"] is True
    _feed(lb, req, pending, _paraphrase_loop())
    assert pending == {} and lb.num_fired == 0
