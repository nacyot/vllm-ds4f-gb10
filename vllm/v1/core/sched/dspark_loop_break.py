# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK reasoning-loop breaker (scheduler side).

Port of vLLM PR #52677 ("Break repeating reasoning loops by forcing the
reasoning end sequence") to the Model Runner V2 path used by this fork. The
upstream PR detects the loop inside the V1 ``ThinkingBudgetStateHolder``; V2
keeps its thinking-budget state on the GPU (Triton kernels over
``req_states.all_token_ids``), so the detection is done here, on the
scheduler's authoritative CPU copy of the output token ids, and the verdict
is shipped to the worker in ``SchedulerOutput.dspark_force_reasoning_end``.
The worker then lowers that request's thinking budget to 0, which makes the
existing V2 forcing kernel emit ``reasoning_end_str`` (the same mechanism a
``thinking_token_budget`` exhaustion uses), and generation continues into the
answer. Thinking is never disabled and the request is not terminated (unlike
``SamplingParams.repetition_detection``).

Detection semantics match ``RepetitionDetectionParams``: a token pattern of
``loop_break_min_pattern_size..loop_break_max_pattern_size`` repeated
``loop_break_min_count`` consecutive times, scoped to the current reasoning
section, checked only after ``loop_break_min_reasoning_tokens`` reasoning
tokens and then every ``loop_break_check_interval`` accepted reasoning tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.sched.utils import check_sequence_repetition

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _find_last_sequence_index(target: list[int], seq: list[int]) -> int:
    if not seq:
        return -1
    n = len(seq)
    for i in range(len(target) - n, -1, -1):
        if target[i : i + n] == seq:
            return i
    return -1


class DsparkLoopBreak:
    """Per-request reasoning-section tracking + exact-cycle detection."""

    def __init__(
        self,
        start_ids: list[int],
        end_ids: list[int],
        natural_end_ids: list[int],
        params: Any,
        min_reasoning_tokens: int,
        check_interval: int,
    ) -> None:
        self.start_ids = list(start_ids)
        self.end_ids = list(end_ids)
        self.natural_end_ids = list(natural_end_ids)
        self.params = params
        self.min_reasoning_tokens = max(0, int(min_reasoning_tokens))
        self.check_interval = max(1, int(check_interval))
        self._max_marker = max(
            len(self.start_ids), len(self.end_ids), len(self.natural_end_ids), 1
        )
        self._need = params.max_pattern_size * params.min_count
        self._state: dict[str, dict[str, Any]] = {}
        self.num_fired = 0

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "DsparkLoopBreak | None":
        rc = getattr(vllm_config, "reasoning_config", None)
        if rc is None or not getattr(rc, "enabled", False):
            return None
        max_pat = int(getattr(rc, "loop_break_max_pattern_size", 0) or 0)
        min_count = int(getattr(rc, "loop_break_min_count", 0) or 0)
        if max_pat <= 0 or min_count < 2:
            return None
        start_ids = rc.reasoning_start_token_ids or []
        end_ids = rc.reasoning_end_token_ids or []
        if not start_ids or not end_ids:
            return None
        natural = getattr(rc, "natural_reasoning_end_token_ids", None) or end_ids
        from vllm.sampling_params import RepetitionDetectionParams

        params = RepetitionDetectionParams(
            max_pattern_size=max_pat,
            min_pattern_size=int(getattr(rc, "loop_break_min_pattern_size", 0) or 0),
            min_count=min_count,
        )
        lb = cls(
            start_ids,
            end_ids,
            natural,
            params,
            getattr(rc, "loop_break_min_reasoning_tokens", 256),
            getattr(rc, "loop_break_check_interval", 16),
        )
        logger.info(
            "DSPARK reasoning loop breaker enabled (scheduler-side, V2 forcing): "
            "pattern %d..%d x%d, min_reasoning=%d, interval=%d",
            params.min_pattern_size or 1,
            params.max_pattern_size,
            params.min_count,
            lb.min_reasoning_tokens,
            lb.check_interval,
        )
        return lb

    # ------------------------------------------------------------------
    def _init_state(self, request: "Request") -> dict[str, Any]:
        """The section may have begun in the prompt (DeepSeek V4 chat
        templates open ``<think>`` before generation): seed from the prompt
        tail like the V1 holder does."""
        st: dict[str, Any] = {
            "scan_pos": 0,
            "in_think": False,
            "section_begin": 0,
            "think_len": 0,
            "last_check": 0,
            "fired": False,
        }
        prompt = getattr(request, "prompt_token_ids", None)
        if prompt:
            tail = list(prompt[-4096:])
            ls = _find_last_sequence_index(tail, self.start_ids)
            le = max(
                _find_last_sequence_index(tail, self.end_ids),
                _find_last_sequence_index(tail, self.natural_end_ids),
            )
            if ls > le:
                st["in_think"] = True
                st["section_begin"] = 0
        return st

    def observe(self, request: "Request", pending: dict[str, int]) -> bool:
        """Update tracking for ``request`` after new output tokens were
        appended; on a confirmed loop, record ``pending[request_id] =
        reasoning tokens so far`` (shipped with the next SchedulerOutput).
        Returns True when it fired now."""
        sp = getattr(request, "sampling_params", None)
        if sp is not None and getattr(sp, "thinking_loop_break", None) is False:
            return False
        rid = request.request_id
        st = self._state.get(rid)
        if st is None:
            st = self._init_state(request)
            self._state[rid] = st
        out = request.output_token_ids
        n = len(out)
        scan_pos = st["scan_pos"]
        if n <= scan_pos:
            return False

        window_begin = max(0, scan_pos - (self._max_marker - 1))
        window = out[window_begin:n]
        last_start = _find_last_sequence_index(window, self.start_ids)
        last_end = max(
            _find_last_sequence_index(window, self.end_ids),
            _find_last_sequence_index(window, self.natural_end_ids),
        )
        if last_end > last_start:
            st["in_think"] = False
            st["section_begin"] = 0
            st["think_len"] = 0
            st["last_check"] = 0
            st["fired"] = False
        elif last_start > last_end:
            st["in_think"] = True
            st["section_begin"] = window_begin + last_start + len(self.start_ids)
            st["think_len"] = n - st["section_begin"]
        elif st["in_think"]:
            st["think_len"] = st["think_len"] + (n - scan_pos)
        st["scan_pos"] = n

        if st["fired"] or not st["in_think"]:
            return False
        think_len = st["think_len"]
        if think_len < self.min_reasoning_tokens:
            return False
        if think_len - st["last_check"] < self.check_interval:
            return False
        st["last_check"] = think_len
        tail_begin = max(st["section_begin"], n - self._need)
        if not check_sequence_repetition(out[tail_begin:n], self.params):
            return False
        st["fired"] = True
        self.num_fired += 1
        pending[rid] = think_len
        logger.info(
            "DSPARK loop breaker: request %s repeating after %d reasoning "
            "tokens; forcing the reasoning end sequence (fired %d total).",
            rid,
            think_len,
            self.num_fired,
        )
        return True

    def forget(self, request_id: str) -> None:
        self._state.pop(request_id, None)
