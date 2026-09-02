# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK operational counters (Prometheus).

Engine side: the scheduler-side trackers (``dspark_loop_break.py``) count
events per step; ``Scheduler.make_stats`` drains them into
``SchedulerStats.dspark_events`` and ``DsparkEventsProm.observe`` turns them
into per-engine counters in the API-server process, exactly like the
speculative-decoding counters.

Frontend side: the DSML leak guard (output processor) and the tool-turn effort
cap (tokenizer wrapper) run in the API-server process and increment their own
counters through ``frontend_counter`` (idempotent lookup in the registry, so
``unregister_vllm_metrics`` between logger inits is harmless).
"""

from __future__ import annotations

import prometheus_client

from vllm.v1.metrics.utils import create_metric_per_engine

# event keys produced by the scheduler-side trackers
EV_LOOP_BREAK_EXACT = "loop_break_exact"
EV_LOOP_BREAK_NOVELTY = "loop_break_novelty"
EV_ANSWER_REPETITION = "answer_repetition"
EV_BUDGET_HIT = "thinking_budget_hit"
EV_SECTION_TEMP_SWITCH = "section_temp_switch"
EV_DSML_BLOCK = "dsml_block"


class DsparkEventsProm:
    """Per-engine counters for the scheduler-side detector events."""

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        base_loop = self._counter_cls(
            name="vllm:dspark_loop_break_fired",
            documentation=(
                "Reasoning loop breaker verdicts (forced reasoning end), by kind: "
                "exact token cycle or novelty collapse."
            ),
            labelnames=labelnames + ["kind"],
        )
        self.counter_loop_break: dict[int, dict[str, prometheus_client.Counter]] = {
            idx: {
                "exact": base_loop.labels(*lv, "exact"),
                "novelty": base_loop.labels(*lv, "novelty"),
            }
            for idx, lv in per_engine_labelvalues.items()
        }
        specs = [
            (
                "vllm:dspark_answer_repetition",
                "Requests finished with finish_reason=repetition by the "
                "answer-section near-repetition detector.",
            ),
            (
                "vllm:dspark_thinking_budget_hits",
                "Reasoning sections cut by the thinking token budget "
                "(forced reasoning end).",
            ),
            (
                "vllm:dspark_section_temperature_switches",
                "Tool-turn requests switched to the answer temperature at a "
                "DSML marker (DSPARK_TOOL_TEMP0_SCOPE).",
            ),
            (
                "vllm:dspark_dsml_blocks",
                "DSML tool-call openers recognised by the scheduler-side "
                "tracker (the loop breaker and the answer-repetition detector "
                "stand down for the block).",
            ),
        ]
        counters = [
            create_metric_per_engine(
                self._counter_cls(name=name, documentation=doc, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for name, doc in specs
        ]
        self.counter_answer_repetition = counters[0]
        self.counter_budget_hits = counters[1]
        self.counter_section_temp_switches = counters[2]
        self.counter_dsml_blocks = counters[3]

    def observe(self, events: dict[str, int], engine_idx: int = 0) -> None:
        if not events:
            return
        n = events.get(EV_LOOP_BREAK_EXACT, 0)
        if n:
            self.counter_loop_break[engine_idx]["exact"].inc(n)
        n = events.get(EV_LOOP_BREAK_NOVELTY, 0)
        if n:
            self.counter_loop_break[engine_idx]["novelty"].inc(n)
        n = events.get(EV_ANSWER_REPETITION, 0)
        if n:
            self.counter_answer_repetition[engine_idx].inc(n)
        n = events.get(EV_BUDGET_HIT, 0)
        if n:
            self.counter_budget_hits[engine_idx].inc(n)
        n = events.get(EV_SECTION_TEMP_SWITCH, 0)
        if n:
            self.counter_section_temp_switches[engine_idx].inc(n)
        n = events.get(EV_DSML_BLOCK, 0)
        if n:
            self.counter_dsml_blocks[engine_idx].inc(n)


def frontend_counter(
    name: str, documentation: str, labelnames: list[str]
) -> prometheus_client.Counter:
    """Get-or-create a counter on the default registry (API-server process).

    ``unregister_vllm_metrics`` drops every ``vllm:``-prefixed collector when
    a stat logger is (re)built, so callers must not cache the result across
    requests; look it up each time (cheap dict access)."""
    registry = prometheus_client.REGISTRY
    existing = registry._names_to_collectors.get(name)  # noqa: SLF001
    if isinstance(existing, prometheus_client.Counter):
        return existing
    try:
        return prometheus_client.Counter(
            name=name, documentation=documentation, labelnames=labelnames
        )
    except ValueError:
        # registered concurrently; fetch it
        existing = registry._names_to_collectors.get(name)  # noqa: SLF001
        assert isinstance(existing, prometheus_client.Counter)
        return existing
