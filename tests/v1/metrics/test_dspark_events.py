# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK detector events → Prometheus counters."""

import random
from types import SimpleNamespace

import prometheus_client

from vllm.v1.core.sched.dspark_loop_break import DsparkLoopBreak
from vllm.v1.metrics.dspark import (
    EV_ANSWER_REPETITION,
    EV_BUDGET_HIT,
    EV_LOOP_BREAK_NOVELTY,
    DsparkEventsProm,
    frontend_counter,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

THINK_START, THINK_END = 900, 901


def _sample(name: str, **labels) -> float:
    v = prometheus_client.REGISTRY.get_sample_value(name, labels)
    return 0.0 if v is None else v


def test_events_prom_counts_per_engine():
    unregister_vllm_metrics()
    prom = DsparkEventsProm(["model_name", "engine"], {0: ["m", "0"]})
    prom.observe({EV_LOOP_BREAK_NOVELTY: 2, EV_ANSWER_REPETITION: 1, EV_BUDGET_HIT: 3})
    prom.observe({})
    assert (
        _sample(
            "vllm:dspark_loop_break_fired_total",
            model_name="m",
            engine="0",
            kind="novelty",
        )
        == 2
    )
    assert (
        _sample(
            "vllm:dspark_loop_break_fired_total",
            model_name="m",
            engine="0",
            kind="exact",
        )
        == 0
    )
    assert (
        _sample("vllm:dspark_answer_repetition_total", model_name="m", engine="0") == 1
    )
    assert (
        _sample("vllm:dspark_thinking_budget_hits_total", model_name="m", engine="0")
        == 3
    )
    unregister_vllm_metrics()


def test_frontend_counter_is_idempotent_and_survives_unregister():
    unregister_vllm_metrics()
    c1 = frontend_counter("vllm:dspark_dsml_leak_guard_stops", "doc", ["trigger"])
    c1.labels("bare_closer:</invoke>").inc()
    c2 = frontend_counter("vllm:dspark_dsml_leak_guard_stops", "doc", ["trigger"])
    assert c1 is c2
    assert (
        _sample(
            "vllm:dspark_dsml_leak_guard_stops_total", trigger="bare_closer:</invoke>"
        )
        == 1
    )
    unregister_vllm_metrics()
    c3 = frontend_counter("vllm:dspark_dsml_leak_guard_stops", "doc", ["trigger"])
    assert c3 is not c1  # re-created after the registry was cleared
    unregister_vllm_metrics()


class _Req:
    def __init__(self, rid="r", budget=None):
        self.request_id = rid
        self.prompt_token_ids = [1, 2, THINK_START]
        self.output_token_ids: list[int] = []
        self.sampling_params = SimpleNamespace(
            thinking_loop_break=None, thinking_token_budget=budget, extra_args=None
        )


def _paraphrase_loop(n_rounds=8, length=200, seed=0):
    rnd = random.Random(seed)
    base = [1000 + rnd.randrange(5000) for _ in range(length)]
    out = list(base)
    for _ in range(n_rounds):
        out += [t if rnd.random() > 0.03 else 1000 + rnd.randrange(5000) for t in base]
    return out


def test_tracker_drains_events():
    lb = DsparkLoopBreak(
        start_ids=[THINK_START],
        end_ids=[THINK_END],
        natural_end_ids=[THINK_END],
        params=None,
        min_reasoning_tokens=32,
        check_interval=4,
        novelty={
            "window": 256,
            "min": 0.4,
            "ngram": 4,
            "consecutive": 2,
            "answer_window": 0,
        },
    )
    req, pending = _Req(budget=64), {}
    toks = _paraphrase_loop()
    for i in range(0, len(toks), 6):
        req.output_token_ids.extend(toks[i : i + 6])
        lb.observe(req, pending)
    ev = lb.drain_events()
    assert ev.get(EV_LOOP_BREAK_NOVELTY) == 1
    assert ev.get(EV_BUDGET_HIT) == 1  # 64-token budget crossed on the way
    assert lb.drain_events() == {}  # drained
