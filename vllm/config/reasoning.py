# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config


@config
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `reasoning_start_str` and `reasoning_end_str` to the strings that delimit
    the reasoning block (e.g. `"<think>"` and `"</think>"`).  The
    corresponding token IDs are derived automatically via
    `initialize_token_ids` and are not intended to be set directly.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String that indicates the end of reasoning content."""

    loop_break_max_pattern_size: int = 0
    """Maximum N-gram pattern size for reasoning-scoped repetition detection.
    When > 0 (together with ``loop_break_min_count``), a request whose current
    reasoning section ends in a repeating token pattern is forced out of
    reasoning via ``reasoning_end_str`` — instead of looping until
    ``max_tokens`` or a ``thinking_token_budget`` — and then answers normally.
    Set to 0 (the default) to disable. Detection semantics match
    ``SamplingParams.repetition_detection``, which by contrast finishes the
    whole request."""

    loop_break_min_pattern_size: int = 0
    """Minimum N-gram pattern size for reasoning loop breaking. If 0, it
    defaults to 1. Must be <= ``loop_break_max_pattern_size``. Raising it does
    not exclude shorter cycles -- a cycle matches at every multiple of its own
    period -- so tune benign repeats out with ``loop_break_min_count``, which
    sets how many tokens must repeat before a break fires."""

    loop_break_min_count: int = 0
    """Number of consecutive repetitions of a pattern that triggers reasoning
    loop breaking. Must be >= 2 when enabled."""

    loop_break_min_reasoning_tokens: int = 256
    """Do not check a reasoning section for loops until it has generated at
    least this many tokens."""

    loop_break_check_interval: int = 16
    """Check for loops every this many newly accepted reasoning tokens."""

    loop_break_novelty_window: int = 0
    """DSPARK near-repetition detector (0 = off). Track the fraction of NEW
    ``loop_break_novelty_ngram``-grams among the last this-many n-grams of the
    current reasoning section; when it stays below
    ``loop_break_novelty_min`` for ``loop_break_novelty_consecutive`` checks
    (spaced by ``loop_break_check_interval``) the reasoning end sequence is
    forced, exactly like an exact-cycle hit. Catches paraphrase loops and
    cycles longer than ``loop_break_max_pattern_size``. Calibration
    (DeepSeek-V4-Flash, 2026-09-02): healthy 15k-33k-token reasoning never
    dropped below 0.65 at window 1024; runaway output fell to 0.0."""

    loop_break_novelty_min: float = 0.4
    """Novelty threshold for ``loop_break_novelty_window``."""

    loop_break_novelty_ngram: int = 8
    """n-gram size for ``loop_break_novelty_window``."""

    loop_break_novelty_consecutive: int = 2
    """Consecutive sub-threshold checks required before firing."""

    answer_repetition_novelty_window: int = 0
    """DSPARK answer-section runaway guard (0 = off). Same statistic over the
    tokens generated AFTER the reasoning section (or over the whole output
    when the request never reasons); when it stays below
    ``answer_repetition_novelty_min`` for ``loop_break_novelty_consecutive``
    checks the request is finished with ``finish_reason=repetition``
    (stop_reason ``dspark_answer_repetition``). Greedy tool turns were
    observed to loop in answer prose until max_tokens."""

    answer_repetition_novelty_min: float = 0.3
    """Novelty threshold for ``answer_repetition_novelty_window``."""

    answer_repetition_dsml_min: float = 0.0
    """DSPARK: novelty threshold applied from a DSML tool-call opener
    (``<｜DSML｜tool_calls>`` / ``<｜DSML｜invoke``) to the end of the section
    (the closer is not tracked; DeepSeek-V4 ends the turn after it); 0 (the
    default) turns the answer detector off there and its tokens are not even
    fed to the window. Tool-call payloads are legitimately repetitive: on
    2026-09-02 (DeepSeek-V4-Flash, window 1024) a 2k-token code patch crossed
    the 0.3 prose threshold and the request was cut mid-argument (the client
    saw a truncated tool call and retried four times). Exact cycles measure
    ~0.0, so a small value here guards against runaway inside a tool call;
    otherwise that case is bounded by ``max_tokens`` only."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_end_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""

    _natural_reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Token IDs of the parser's own end marker (e.g. ``</think>``) without any
    transition phrase a configured ``reasoning_end_str`` may prepend. This is
    what a natural exit emits, so section tracking must recognise it. Equal
    to ``reasoning_end_token_ids`` when no transition phrase is configured."""

    _enabled: bool = field(default=False, init=False, repr=False)
    """Private field indicating whether reasoning token IDs have been initialized.
    Set to True by `initialize_token_ids` once token IDs are initialized."""

    @property
    def enabled(self) -> bool:
        """Returns True if reasoning is enabled (i.e. if token IDs have been
        initialized), False otherwise."""
        return self._enabled

    @property
    def reasoning_start_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_start_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_start_token_ids

    @property
    def natural_reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs of the parser's natural end marker. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._natural_reasoning_end_token_ids

    def __post_init__(self) -> None:
        if self.loop_break_max_pattern_size > 0:
            if self.loop_break_min_count < 2:
                raise ValueError(
                    "ReasoningConfig: loop_break_min_count must be >= 2 when "
                    "loop_break_max_pattern_size > 0"
                )
            if self.loop_break_min_pattern_size > self.loop_break_max_pattern_size:
                raise ValueError(
                    "ReasoningConfig: loop_break_min_pattern_size must be <= "
                    "loop_break_max_pattern_size"
                )
        if (
            self.loop_break_novelty_window < 0
            or self.answer_repetition_novelty_window < 0
        ):
            raise ValueError("ReasoningConfig: novelty windows must be >= 0")
        for name in (
            "loop_break_novelty_min",
            "answer_repetition_novelty_min",
            "answer_repetition_dsml_min",
        ):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"ReasoningConfig: {name} must be within [0, 1]")
        if self.loop_break_novelty_ngram < 1 or self.loop_break_novelty_consecutive < 1:
            raise ValueError(
                "ReasoningConfig: loop_break_novelty_ngram and "
                "loop_break_novelty_consecutive must be >= 1"
            )

    @property
    def reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_end_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
        ):
            self._enabled = True
            return  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        natural_end_str = ""
        if self.reasoning_parser:
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not reasoning_start_str:
                reasoning_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not reasoning_end_str:
                reasoning_end_str = end_token
            if end_token:
                natural_end_str = end_token

        if not reasoning_start_str or not reasoning_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            reasoning_end_str, add_special_tokens=False
        )

        natural_ids = (
            tokenizer.encode(natural_end_str, add_special_tokens=False)
            if natural_end_str and natural_end_str != reasoning_end_str
            else list(self._reasoning_end_token_ids)
        )
        self._natural_reasoning_end_token_ids = natural_ids or list(
            self._reasoning_end_token_ids
        )

        if not self._reasoning_start_token_ids or not self._reasoning_end_token_ids:
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
