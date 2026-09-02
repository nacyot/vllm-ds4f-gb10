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

The same per-request section tracking also drives the *section temperature*
channel: a tool turn keeps the request's own temperature and switches to a
greedy one only where determinism actually buys something, namely the DSML
tool-call structure tokens. The switch rides
``SchedulerOutput.dspark_section_temperature``. Two trigger sets exist,
selected per request by ``extra_args["dspark_temp0_scope"]`` (server default:
``DSPARK_TOOL_TEMP0_SCOPE``):

``dsml`` (default)
    Switch only when the ``<｜DSML｜tool_calls>`` marker appears in the
    output, wherever it appears -- inside the think section (the model skips
    ``</think>`` at long context) or after it. Reasoning *and* answer prose
    keep the request's own temperature. This is the setting the production
    evidence asks for: greedy decoding loops in answer prose too, not only in
    reasoning (a 60k-token agent turn ended its reasoning after ~3k tokens and
    then ran the answer to max_tokens with 8-gram duplication 0.70, the same
    lines repeated 47-108 times).
``answer``
    Switch at the end of the reasoning section: a prompt that does not end
    inside ``<think>``, a natural ``</think>``, the forced end sequence, or
    the DSML marker. Re-entering ``<think>`` hands the reasoning temperature
    back. Answer prose is greedy under this scope.

Both are one scheduler step behind the token that triggers them: the marker is
only visible after the step that produced it has returned, so the tokens
sampled in that same step -- up to ``1 + num_speculative_tokens`` of them, the
start of ``<｜DSML｜invoke name="`` -- are still drawn at the request's own
temperature. The tracker shortens that as far as the channel allows by
latching on a *partial* marker at the tail of the output (the first marker
token alone is enough), so the switch is in force for the step right after the
marker starts rather than the step after it completes. Closing the gap
entirely would need the worker to detect the marker itself.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.sched.dspark_novelty import SectionNovelty
from vllm.v1.core.sched.utils import check_sequence_repetition

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# DSPARK: SamplingParams.extra_args keys written by the OpenAI chat entrypoint
# (vllm/entrypoints/openai/chat_completion/protocol.py) and read here. The
# entrypoint spells them as literals; tests assert the two agree.
DSPARK_ANSWER_TEMP_KEY = "dspark_answer_temperature"
DSPARK_REASONING_TEMP_KEY = "dspark_reasoning_temperature"
DSPARK_SCOPE_KEY = "dspark_temp0_scope"

# DSPARK: the marker the DSv4 model opens a tool call with. At long context it
# sometimes skips ``</think>`` and starts the block straight inside the think
# section (see vllm/reasoning/deepseek_v4_reasoning_parser.py), so the marker
# is looked for everywhere in the output, not only after ``</think>``.
DSPARK_DSML_TOOL_START = "<｜DSML｜tool_calls>"
# DSPARK: at long context the model sometimes omits the tool_calls opener and
# starts with an invoke block (vLLM #48931); both markers count.
DSPARK_DSML_INVOKE_START = "<｜DSML｜invoke"

# DSPARK: scopes that split the temperature. "request" (the pre-2026-09
# behaviour, temperature 0 for the whole request) is handled entirely in the
# entrypoint and never reaches this module.
DSPARK_SCOPE_DSML = "dsml"
DSPARK_SCOPE_ANSWER = "answer"
DSPARK_SPLIT_SCOPES = (DSPARK_SCOPE_DSML, DSPARK_SCOPE_ANSWER)
DSPARK_DEFAULT_SCOPE = DSPARK_SCOPE_DSML


def dspark_temp0_scope() -> str:
    """DSPARK: the server-wide ``DSPARK_TOOL_TEMP0_SCOPE``, normalised.
    Unset or empty means ``dsml``; anything unrecognised means ``request``,
    the whole-request rollback."""
    scope = (os.environ.get("DSPARK_TOOL_TEMP0_SCOPE") or "").strip().lower()
    if not scope:
        return DSPARK_DEFAULT_SCOPE
    return scope if scope in DSPARK_SPLIT_SCOPES else "request"


def dspark_section_temperature_enabled() -> bool:
    """DSPARK: whether tool turns run split-temperature. Mirrors the entrypoint
    gate: ``DSPARK_TOOL_TEMP0=1`` with a scope that splits."""
    if os.environ.get("DSPARK_TOOL_TEMP0") != "1":
        return False
    return dspark_temp0_scope() in DSPARK_SPLIT_SCOPES


def _find_last_sequence_index(target: list[int], seq: list[int]) -> int:
    if not seq:
        return -1
    n = len(seq)
    for i in range(len(target) - n, -1, -1):
        if target[i : i + n] == seq:
            return i
    return -1


def _ends_with_proper_prefix(
    target: list[int], seq: list[int], min_len: int = 2
) -> bool:
    """DSPARK: True when ``target`` ends in a *proper* prefix of ``seq`` of at
    least ``min_len`` tokens -- the marker has started but has not finished
    landing. Latching on this costs one step less than waiting for the whole
    marker (see the module docstring on the boundary lag).

    ``min_len`` defaults to 2: the markers start at the ``｜DSML｜`` special
    token (the plain ``<`` in front of it is stripped, see
    ``_dsml_token_ids``), and a one-token prefix would latch on every other
    DSML construct (``parameter``, the closers); two tokens are unambiguous.
    Before 2026-09-02 (evening) the markers kept the ``<`` and the same rule
    kept a bare ``<`` in prose or code from latching."""
    for k in range(min(len(seq) - 1, len(target)), max(1, min_len) - 1, -1):
        if target[-k:] == seq[:k]:
            return True
    return False


def _without_leading(seq: list[int], lead: list[int]) -> list[int]:
    """DSPARK: drop a leading plain-``<`` token from a DSML marker. The
    opener tokenizes as ``<`` (id 30 on DeepSeek-V4) followed by the ``｜DSML｜``
    special token, and that ``<`` merges with whatever precedes it (" <" ->
    818, ".<" -> 32334, ")<" -> 49584, verified 2026-09-02), so a marker keyed
    on ``<`` misses an opener that does not start a line."""
    if len(seq) > 2 and len(lead) == 1 and seq[0] == lead[0]:
        return seq[1:]
    return seq


def _reset_answer_tracker(st: dict[str, Any]) -> None:
    """DSPARK: start a fresh answer-section novelty window."""
    st["ans_nov"] = None
    st["ans_below"] = 0
    st["ans_last"] = 0


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
        dsml_ids: list[int] | None = None,
        default_scope: str = DSPARK_DEFAULT_SCOPE,
        novelty: dict[str, Any] | None = None,
    ) -> None:
        # DSPARK near-repetition detectors (see dspark_novelty.py). Keys:
        # window/min/ngram/consecutive (reasoning section, forces the end
        # sequence like an exact cycle) and answer_window/answer_min (answer
        # section, finishes the request with finish_reason=repetition).
        nv = novelty or {}
        self.novelty_window = max(0, int(nv.get("window", 0) or 0))
        self.novelty_min = float(nv.get("min", 0.4))
        self.novelty_ngram = max(1, int(nv.get("ngram", 8) or 8))
        self.novelty_consecutive = max(1, int(nv.get("consecutive", 2) or 2))
        self.answer_novelty_window = max(0, int(nv.get("answer_window", 0) or 0))
        self.answer_novelty_min = float(nv.get("answer_min", 0.3))
        # DSPARK: threshold inside a DSML tool-call block; 0 = detector off there.
        self.answer_dsml_min = float(nv.get("answer_dsml_min", 0.0) or 0.0)
        self.num_fired_novelty = 0
        self.num_answer_repetition = 0
        # DSPARK: DSML tool-call openers recognised (observability for the
        # skip rules, which are otherwise silent).
        self.num_dsml_blocks = 0
        self._pending_finish: dict[str, str] = {}
        # DSPARK: per-step event counts drained by Scheduler.make_stats into
        # SchedulerStats.dspark_events (Prometheus counters on the frontend).
        self._events: dict[str, int] = {}
        self.start_ids = list(start_ids)
        self.end_ids = list(end_ids)
        self.natural_end_ids = list(natural_end_ids)
        # DSPARK: ``params is None`` = section tracking only (the section
        # temperature channel needs it even when loop breaking is unconfigured).
        self.params = params
        self.min_reasoning_tokens = max(0, int(min_reasoning_tokens))
        self.check_interval = max(1, int(check_interval))
        # DSPARK: DSML tool-call marker ids; empty disables that detection.
        # DSPARK: one or more DSML start markers (tool_calls opener, and the
        # invoke opener for the omitted-opener case). ``dsml_ids`` keeps the
        # first for backwards compatibility with the tests/log lines.
        markers = dsml_ids or []
        if markers and isinstance(markers[0], int):
            markers = [list(markers)]
        self.dsml_markers: list[list[int]] = [list(m) for m in markers if m]
        self.dsml_ids = list(self.dsml_markers[0]) if self.dsml_markers else []
        # DSPARK: trigger set for requests whose extra_args do not name one.
        self.default_scope = default_scope
        # DSPARK: True when reasoning_end_str prepends a transition phrase, so
        # a forced exit is distinguishable from a natural one.
        self._has_transition = self.end_ids != self.natural_end_ids
        self._max_marker = max(
            len(self.start_ids),
            len(self.end_ids),
            len(self.natural_end_ids),
            max((len(m) for m in self.dsml_markers), default=0),
            1,
        )
        self._need = (params.max_pattern_size * params.min_count) if params else 0
        self._state: dict[str, dict[str, Any]] = {}
        self.num_fired = 0
        # thinking_token_budget hits (server default or per-request) are
        # counted here too: the forcing kernel is silent, and a forced
        # ``</think>`` is indistinguishable from a natural one in the output.
        self.num_budget_hits = 0
        # DSPARK: requests that switched to their answer temperature.
        self.num_answer_temp_switches = 0

    @classmethod
    def from_config(cls, vllm_config: VllmConfig) -> DsparkLoopBreak | None:
        rc = getattr(vllm_config, "reasoning_config", None)
        start_ids = (rc.reasoning_start_token_ids or []) if rc is not None else []
        end_ids = (rc.reasoning_end_token_ids or []) if rc is not None else []
        if (
            rc is None
            or not getattr(rc, "enabled", False)
            or not start_ids
            or not end_ids
        ):
            if dspark_section_temperature_enabled():
                # The entrypoint is handing out per-request answer temperatures
                # that nothing here can apply: a tool turn would run at its own
                # temperature end to end, which is what DSPARK_TOOL_TEMP0 exists
                # to prevent. Say so instead of failing quietly.
                logger.warning(
                    "DSPARK_TOOL_TEMP0_SCOPE=%s needs a reasoning config "
                    "(reasoning parser / reasoning_start_str + reasoning_end_str) "
                    "and this model has none, so the scope is inert. Tool turns "
                    "will run at their own temperature end to end, DSML "
                    "structure tokens included; set "
                    "DSPARK_TOOL_TEMP0_SCOPE=request for the whole-request "
                    "temperature 0 instead.",
                    dspark_temp0_scope(),
                )
            return None
        max_pat = int(getattr(rc, "loop_break_max_pattern_size", 0) or 0)
        min_count = int(getattr(rc, "loop_break_min_count", 0) or 0)
        loop_break_on = max_pat > 0 and min_count >= 2
        # DSPARK: the tracker is also the section-temperature source, so it is
        # built when either consumer is configured.
        section_temp_on = dspark_section_temperature_enabled()
        novelty = {
            "window": getattr(rc, "loop_break_novelty_window", 0) or 0,
            "min": getattr(rc, "loop_break_novelty_min", 0.4),
            "ngram": getattr(rc, "loop_break_novelty_ngram", 8) or 8,
            "consecutive": getattr(rc, "loop_break_novelty_consecutive", 2) or 2,
            "answer_window": getattr(rc, "answer_repetition_novelty_window", 0) or 0,
            "answer_min": getattr(rc, "answer_repetition_novelty_min", 0.3),
            "answer_dsml_min": getattr(rc, "answer_repetition_dsml_min", 0.0) or 0.0,
        }
        novelty_on = novelty["window"] > 0 or novelty["answer_window"] > 0
        if not loop_break_on and not section_temp_on and not novelty_on:
            return None
        natural = getattr(rc, "natural_reasoning_end_token_ids", None) or end_ids
        params = None
        if loop_break_on:
            from vllm.sampling_params import RepetitionDetectionParams

            params = RepetitionDetectionParams(
                max_pattern_size=max_pat,
                min_pattern_size=int(
                    getattr(rc, "loop_break_min_pattern_size", 0) or 0
                ),
                min_count=min_count,
            )
        scope = dspark_temp0_scope()
        # DSPARK: the DSML tool-call marker serves every consumer that gets
        # here -- the section-temperature channel, the loop breaker (never
        # force inside a tool-call block) and the answer detector (never score
        # one). Until 2026-09-02 (evening) it was tokenized only under
        # DSPARK_TOOL_TEMP0=1, so at the vendor operating point (TEMP0 unset)
        # both detector rules were inert and a code patch inside a tool call
        # was cut as an answer runaway.
        dsml_ids = cls._dsml_token_ids(vllm_config, scope, section_temp_on)
        lb = cls(
            start_ids,
            end_ids,
            natural,
            params,
            getattr(rc, "loop_break_min_reasoning_tokens", 256),
            getattr(rc, "loop_break_check_interval", 16),
            dsml_ids=dsml_ids,
            default_scope=scope
            if scope in DSPARK_SPLIT_SCOPES
            else DSPARK_SCOPE_ANSWER,
            novelty=novelty,
        )
        if novelty_on:
            logger.info(
                "DSPARK near-repetition detectors enabled: reasoning window=%d "
                "min=%.2f ngram=%d consecutive=%d (forces the end sequence); "
                "answer window=%d min=%.2f (finish_reason=repetition); "
                "tool-call blocks %s.",
                lb.novelty_window,
                lb.novelty_min,
                lb.novelty_ngram,
                lb.novelty_consecutive,
                lb.answer_novelty_window,
                lb.answer_novelty_min,
                (
                    f"min={lb.answer_dsml_min:.2f}"
                    if lb.answer_dsml_min > 0
                    else "skipped (answer_repetition_dsml_min=0)"
                )
                if lb.dsml_markers
                else "not recognised (DSML marker not tokenized)",
            )
        if params is not None:
            logger.info(
                "DSPARK reasoning loop breaker enabled (scheduler-side, V2 forcing): "
                "pattern %d..%d x%d, min_reasoning=%d, interval=%d",
                params.min_pattern_size or 1,
                params.max_pattern_size,
                params.min_count,
                lb.min_reasoning_tokens,
                lb.check_interval,
            )
        if section_temp_on:
            logger.info(
                "DSPARK tool-turn section temperature enabled "
                "(DSPARK_TOOL_TEMP0_SCOPE=%s): %s; DSML marker ids: %s.",
                scope,
                (
                    "reasoning and answer prose keep the request's own "
                    "temperature, only the DSML tool-call block is greedy"
                    if scope == DSPARK_SCOPE_DSML
                    else "reasoning keeps the request's own temperature, the "
                    "answer and the DSML tool-call block are greedy"
                ),
                lb.dsml_ids or "unavailable",
            )
        return lb

    @staticmethod
    def _dsml_token_ids(
        vllm_config: VllmConfig, scope: str, section_temp_on: bool = True
    ) -> list[int]:
        """DSPARK: tokenize the DSML tool-call start marker. Under scope
        ``answer`` failure only means a think section left by opening a tool
        call without ``</think>`` keeps the reasoning temperature; under scope
        ``dsml`` the marker is the only trigger there is, so failure makes the
        scope inert. With the section temperature off (``section_temp_on``
        False) the marker only serves the detectors, which then cannot tell a
        tool-call block from prose."""
        try:
            from vllm.tokenizers import cached_tokenizer_from_config

            tokenizer = cached_tokenizer_from_config(
                model_config=vllm_config.model_config
            )
            ids = list(
                tokenizer.encode(DSPARK_DSML_TOOL_START, add_special_tokens=False)
            )
            try:
                inv = list(
                    tokenizer.encode(DSPARK_DSML_INVOKE_START, add_special_tokens=False)
                )
            except Exception:  # noqa: BLE001
                inv = []
            # DSPARK: anchor on the ``｜DSML｜`` special token (see
            # ``_without_leading``); the two-token prefix latch stays
            # unambiguous because ``parameter`` / the closers only share the
            # first token.
            lead = list(tokenizer.encode("<", add_special_tokens=False))
            ids, inv = _without_leading(ids, lead), _without_leading(inv, lead)
            return [ids, inv] if inv else [ids]
        except Exception as exc:  # noqa: BLE001 - best effort, logged once
            if not section_temp_on:
                logger.warning(
                    "DSPARK detectors: could not tokenize %r (%s); the loop "
                    "breaker and the answer-repetition detector cannot recognise "
                    "DSML tool-call blocks and will treat them as prose.",
                    DSPARK_DSML_TOOL_START,
                    exc,
                )
            elif scope == DSPARK_SCOPE_DSML:
                logger.warning(
                    "DSPARK_TOOL_TEMP0_SCOPE=dsml could not tokenize %r (%s), and "
                    "that marker is its only trigger, so the scope is inert: tool "
                    "turns will run at their own temperature end to end, DSML "
                    "structure tokens included. Fall back to "
                    "DSPARK_TOOL_TEMP0_SCOPE=answer (switch at the end of the "
                    "reasoning section) or =request (whole-request temperature 0).",
                    DSPARK_DSML_TOOL_START,
                    exc,
                )
            else:
                logger.warning(
                    "DSPARK section temperature: could not tokenize %r (%s); a tool "
                    "call opened without </think> will keep the reasoning "
                    "temperature.",
                    DSPARK_DSML_TOOL_START,
                    exc,
                )
            return []

    # ------------------------------------------------------------------
    def _init_state(self, request: Request) -> dict[str, Any]:
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
            # DSPARK: section temperature bookkeeping. ``dsml_seen`` is
            # per-section (scope "answer"), ``dsml_latched`` is for the whole
            # request (scope "dsml": once a tool call has started, the rest of
            # the request stays greedy).
            "dsml_seen": False,
            "dsml_latched": False,
            "end_reason": "prompt",
            "end_think_len": 0,
            "temp_applied": False,
            "temp_logged": False,
            # DSPARK near-repetition trackers (reasoning section / answer).
            "nov": None,
            "nov_below": 0,
            "nov_last": 0,
            "ans_nov": None,
            "ans_below": 0,
            "ans_last": 0,
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

    def observe(
        self,
        request: Request,
        pending: dict[str, int],
        pending_temp: dict[str, float] | None = None,
    ) -> bool:
        """Update tracking for ``request`` after new output tokens were
        appended; on a confirmed loop, record ``pending[request_id] =
        reasoning tokens so far`` (shipped with the next SchedulerOutput).
        Returns True when it fired now.

        DSPARK: when ``pending_temp`` is given, requests carrying
        ``extra_args[DSPARK_ANSWER_TEMP_KEY]`` also get their section
        temperature recorded there (see ``_observe_section_temperature``)."""
        sp = getattr(request, "sampling_params", None)
        # DSPARK: the opt-out only disables loop breaking; section tracking
        # still has to run for the section temperature channel.
        opted_out = sp is not None and getattr(sp, "thinking_loop_break", None) is False
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
        last_forced_end = _find_last_sequence_index(window, self.end_ids)
        last_natural_end = _find_last_sequence_index(window, self.natural_end_ids)
        last_end = max(last_forced_end, last_natural_end)
        # DSPARK: budget-hit log race. The forcing kernel lands ``</think>`` in
        # the same multi-token step as the budget crossing, and the reset
        # below used to run before the budget check, so the hit went unlogged.
        ended_now = last_end > last_start
        prev_in_think = st["in_think"]
        prev_budget_logged = st.get("budget_logged", False)
        ended_len = (
            (window_begin + last_end) - st["section_begin"]
            if ended_now and prev_in_think
            else 0
        )
        if last_end > last_start:
            st["nov"] = None
            st["nov_below"] = 0
            st["nov_last"] = 0
            st["in_think"] = False
            st["section_begin"] = 0
            # DSPARK: keep the length for the section-temperature log line;
            # think_len is about to be reset for the next section.
            st["end_think_len"] = st["think_len"]
            st["think_len"] = 0
            st["last_check"] = 0
            st["fired"] = False
            st["budget_logged"] = False
            st["dsml_seen"] = False
            if prev_in_think:
                # DSPARK: a new answer section; the old window may hold an
                # in-think tool-call block and must not score the first
                # answer tokens (review finding, 2026-09-02).
                _reset_answer_tracker(st)
            # DSPARK: which marker closed it. Without a transition phrase the
            # two id lists are identical and every exit is a natural one;
            # with one, the natural marker is a suffix of the forced sequence,
            # so the forced sequence wins a tie on end position.
            st["end_reason"] = (
                "forced"
                if (
                    self._has_transition
                    and last_forced_end >= 0
                    and last_forced_end + len(self.end_ids)
                    >= last_natural_end + len(self.natural_end_ids)
                )
                else "natural"
            )
        elif last_start > last_end:
            if not st["in_think"] or window_begin + last_start >= st["section_begin"]:
                # a new section (or the first one): re-arm per-section flags
                st["fired"] = False
                st["last_check"] = 0
                st["budget_logged"] = False
                st["dsml_seen"] = False
                st["nov"] = None
                st["nov_below"] = 0
                st["nov_last"] = 0
                _reset_answer_tracker(st)
            st["in_think"] = True
            st["section_begin"] = window_begin + last_start + len(self.start_ids)
            st["think_len"] = n - st["section_begin"]
        elif st["in_think"]:
            st["think_len"] = st["think_len"] + (n - scan_pos)
        # DSPARK: the DSML tool-call marker. Scope "dsml" triggers on it alone,
        # so it is looked for everywhere -- inside the think section (where it
        # also ends the reasoning as far as scope "answer" is concerned) and
        # after it. A partial marker at the tail counts: latching one step
        # earlier is the only lag the scheduler-side channel can shave off.
        # The loop breaker's own section view is untouched either way.
        # ``dsml_seen`` is the per-section flag, so a second tool-call block in
        # a re-entered think section is still found under scope "answer".
        if self.dsml_markers and not st["dsml_seen"]:
            hit = False
            for marker in self.dsml_markers:
                last_dsml = _find_last_sequence_index(window, marker)
                if last_dsml >= 0 and last_dsml > last_start:
                    hit = True
                    break
                if len(marker) > 1 and _ends_with_proper_prefix(window, marker):
                    hit = True
                    break
            if hit:
                st["dsml_seen"] = True
                st["dsml_latched"] = True
                st["end_reason"] = "dsml"
                # DSPARK: the block gets its own window (scored only under
                # answer_repetition_dsml_min > 0), and the count makes the
                # otherwise silent skip rules visible on the dashboard.
                _reset_answer_tracker(st)
                self.num_dsml_blocks += 1
                self._event("dsml_block")
        st["scan_pos"] = n
        new_tokens = out[scan_pos:n]

        if pending_temp is not None:
            self._observe_section_temperature(request, st, sp, pending_temp)

        # DSPARK: feed the near-repetition trackers. A boundary step (<= 1 +
        # num_spec tokens) is attributed to the section active at its end.
        if st["in_think"] and not st["dsml_seen"]:
            if self.novelty_window:
                nov = st["nov"]
                if nov is None:
                    nov = st["nov"] = SectionNovelty(
                        ngram=self.novelty_ngram, window=self.novelty_window
                    )
                nov.push_many(new_tokens)
        elif (
            self.answer_novelty_window
            and not st["in_think"]
            and (not st["dsml_seen"] or self.answer_dsml_min > 0)
        ):
            # DSPARK: answer prose, or a tool-call block under the opt-in
            # threshold. A block opened inside <think> feeds nothing: the
            # verdict is never taken there and its tokens must not linger in
            # the window the next answer section starts with.
            ans = st["ans_nov"]
            if ans is None:
                ans = st["ans_nov"] = SectionNovelty(
                    ngram=self.novelty_ngram, window=self.answer_novelty_window
                )
            ans.push_many(new_tokens)

        if opted_out:
            return False

        budget = getattr(sp, "thinking_token_budget", None) if sp else None
        if (
            ended_now
            and prev_in_think
            and not prev_budget_logged
            and budget is not None
            and budget > 0
            and ended_len + 8 >= budget
        ):
            self.num_budget_hits += 1
            self._event("thinking_budget_hit")
            logger.info(
                "DSPARK thinking budget reached: request %s ended its reasoning "
                "section at ~%d tokens (budget %d); the end sequence was forced "
                "(budget hits %d total).",
                rid,
                ended_len,
                budget,
                self.num_budget_hits,
            )
        if st["in_think"] and not st.get("budget_logged"):
            budget = getattr(sp, "thinking_token_budget", None) if sp else None
            if budget is not None and budget > 0 and st["think_len"] >= budget:
                st["budget_logged"] = True
                self.num_budget_hits += 1
                self._event("thinking_budget_hit")
                logger.info(
                    "DSPARK thinking budget reached: request %s used %d reasoning "
                    "tokens (budget %d); the end sequence is being forced "
                    "(budget hits %d total).",
                    rid,
                    st["think_len"],
                    budget,
                    self.num_budget_hits,
                )
        if self._check_answer_repetition(rid, st):
            return True
        if self._check_reasoning_novelty(rid, st, pending):
            return True
        if (
            self.params is None
            or st["fired"]
            or not st["in_think"]
            or st["dsml_seen"]  # a DSML block is being written: never force
        ):
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
        self._event("loop_break_exact")
        pending[rid] = think_len
        logger.info(
            "DSPARK loop breaker: request %s repeating after %d reasoning "
            "tokens; forcing the reasoning end sequence (fired %d total).",
            rid,
            think_len,
            self.num_fired,
        )
        return True

    # ------------------------------------------------------------------
    def _check_reasoning_novelty(
        self, rid: str, st: dict[str, Any], pending: dict[str, int]
    ) -> bool:
        """Near-repetition inside the reasoning section: force the end
        sequence like an exact-cycle hit (same verdict channel)."""
        if (
            not self.novelty_window
            or st["fired"]
            or not st["in_think"]
            or st["dsml_seen"]
        ):
            return False
        nov = st["nov"]
        think_len = st["think_len"]
        if nov is None or think_len < self.min_reasoning_tokens:
            return False
        if think_len - st["nov_last"] < self.check_interval:
            return False
        st["nov_last"] = think_len
        if not nov.ready:
            return False
        value = nov.novelty
        if value < self.novelty_min:
            st["nov_below"] += 1
        else:
            st["nov_below"] = 0
        if st["nov_below"] < self.novelty_consecutive:
            return False
        st["fired"] = True
        self.num_fired += 1
        self.num_fired_novelty += 1
        self._event("loop_break_novelty")
        pending[rid] = think_len
        logger.info(
            "DSPARK loop breaker: request %s near-repeating after %d reasoning "
            "tokens (novelty %.2f < %.2f over %d tokens); forcing the reasoning "
            "end sequence (fired %d total, %d by novelty).",
            rid,
            think_len,
            value,
            self.novelty_min,
            self.novelty_window,
            self.num_fired,
            self.num_fired_novelty,
        )
        return True

    def _check_answer_repetition(self, rid: str, st: dict[str, Any]) -> bool:
        """Near-repetition in the answer section: the request is finished with
        finish_reason=repetition (the scheduler collects the verdict through
        ``take_answer_repetition``)."""
        if not self.answer_novelty_window or st["in_think"]:
            return False
        # DSPARK: inside a DSML tool-call block the payload (a patch, JSON, a
        # table) is legitimately repetitive -- on 2026-09-02 a 2k-token code
        # patch crossed the 0.3 prose threshold at window 1024 and the request
        # was cut mid-argument -- so the prose threshold never applies there;
        # ``answer_repetition_dsml_min`` (default 0 = off) is the opt-in
        # guard for that section, which lasts until the section ends (the
        # closer is not tracked; DeepSeek-V4 ends the turn after it).
        in_dsml = bool(st["dsml_seen"])
        threshold = self.answer_dsml_min if in_dsml else self.answer_novelty_min
        if threshold <= 0.0:
            st["ans_below"] = 0
            return False
        ans = st["ans_nov"]
        if ans is None or rid in self._pending_finish:
            return False
        if ans.count - st["ans_last"] < self.check_interval:
            return False
        st["ans_last"] = ans.count
        if not ans.ready:
            return False
        value = ans.novelty
        if value < threshold:
            st["ans_below"] += 1
        else:
            st["ans_below"] = 0
        if st["ans_below"] < self.novelty_consecutive:
            return False
        self.num_answer_repetition += 1
        self._event("answer_repetition")
        self._pending_finish[rid] = (
            f"dspark_answer_repetition(novelty={value:.2f},tokens={ans.count}"
            + (",section=dsml)" if in_dsml else ")")
        )
        logger.info(
            "DSPARK answer repetition: request %s near-repeating after %d %s "
            "tokens (novelty %.2f < %.2f over %d tokens); finishing with "
            "finish_reason=repetition (%d total).",
            rid,
            ans.count,
            "tool-call block" if in_dsml else "answer",
            value,
            threshold,
            self.answer_novelty_window,
            self.num_answer_repetition,
        )
        return True

    def _event(self, key: str) -> None:
        self._events[key] = self._events.get(key, 0) + 1

    def drain_events(self) -> dict[str, int]:
        """Return and reset the per-step event counts (metrics)."""
        if not self._events:
            return {}
        ev, self._events = self._events, {}
        return ev

    def take_answer_repetition(self, request_id: str) -> str | None:
        """Pop the answer-repetition verdict for ``request_id`` (stop_reason)."""
        return self._pending_finish.pop(request_id, None)

    # ------------------------------------------------------------------
    def _observe_section_temperature(
        self,
        request: Request,
        st: dict[str, Any],
        sp: Any,
        pending_temp: dict[str, float],
    ) -> None:
        """DSPARK: record the temperature this request should be sampled at now.

        Only requests whose ``extra_args`` carry ``dspark_answer_temperature``
        participate; ``extra_args["dspark_temp0_scope"]`` picks the trigger set
        (``dsml``: the tool-call marker alone, latched for the rest of the
        request; ``answer``: any end of the reasoning section, with a switch
        back when a new one opens). Before the trigger nothing is emitted --
        the worker already holds ``sampling_params.temperature``, which is also
        what a preemption + resume restores. After it the answer temperature is
        re-emitted on every step, which makes the channel idempotent and
        self-healing across preemption. A request whose trigger never fires
        (scope ``dsml``, no tool call) is never touched at all."""
        extra = getattr(sp, "extra_args", None) if sp is not None else None
        if not extra:
            return
        answer_temp = extra.get(DSPARK_ANSWER_TEMP_KEY)
        if answer_temp is None:
            return
        rid = request.request_id
        scope = extra.get(DSPARK_SCOPE_KEY) or self.default_scope
        if scope == DSPARK_SCOPE_DSML:
            # The tool-call marker is the only trigger, and it latches for the
            # rest of the request: a second block or interleaved thinking after
            # a tool call does not need to switch back.
            switched = st["dsml_latched"]
        else:
            switched = (not st["in_think"]) or st["dsml_seen"]
        if not switched:
            if st["temp_applied"]:
                # Scope "answer" only: back inside <think> (interleaved
                # thinking), so restore the request's own temperature.
                st["temp_applied"] = False
                reasoning_temp = extra.get(DSPARK_REASONING_TEMP_KEY)
                if reasoning_temp is None:
                    reasoning_temp = getattr(sp, "temperature", None)
                if reasoning_temp is not None:
                    pending_temp[rid] = float(reasoning_temp)
            return
        pending_temp[rid] = float(answer_temp)
        if st["temp_applied"]:
            return
        st["temp_applied"] = True
        if not st["temp_logged"]:
            st["temp_logged"] = True
            self.num_answer_temp_switches += 1
            self._event("section_temp_switch")
            # The DSML marker leaves think_len live; a closed section parked
            # its length in end_think_len before resetting.
            think_len = st["think_len"] if st["in_think"] else st["end_think_len"]
            logger.info(
                "DSPARK section temperature (scope %s): request %s switched after "
                "%d output tokens / %d reasoning tokens (trigger: %s); the rest "
                "of the turn is sampled at temperature %s (switches %d total).",
                scope,
                rid,
                len(request.output_token_ids),
                think_len,
                st["end_reason"],
                answer_temp,
                self.num_answer_temp_switches,
            )

    def forget(self, request_id: str) -> None:
        self._state.pop(request_id, None)
        self._pending_finish.pop(request_id, None)
