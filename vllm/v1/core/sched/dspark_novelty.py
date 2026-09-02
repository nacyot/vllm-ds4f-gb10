# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPARK near-repetition detector: rolling distinct-n-gram ratio.

The exact-cycle test in ``check_sequence_repetition`` only fires when the
last tokens are a byte-identical period-N repetition. Greedy (or long)
deliberation collapses into *semantic* loops instead: the same paragraph,
checklist or code block re-derived with small wording changes, or cycles far
longer than the pattern cap. Those never trip the exact test but are easy to
see as a novelty collapse: the fraction of n-grams in the trailing window that
were never seen earlier in the section drops from ~0.9-1.0 (healthy) to
~0.1-0.4 (looping).

``SectionNovelty`` is a pure-Python, O(1)-per-token tracker used by the
scheduler-side loop breaker (see ``dspark_loop_break.py``). Memory is bounded
by ``history``: the set of seen n-grams is rebuilt from the trailing
``history`` tokens when it grows past that size.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable


class SectionNovelty:
    """Rolling novelty of a token stream.

    novelty = (# n-grams among the last ``window`` n-grams that were NEW when
    they appeared) / min(window, n-grams seen so far).
    """

    __slots__ = (
        "ngram",
        "window",
        "history",
        "_buf",
        "_seen",
        "_flags",
        "_new",
        "_tokens",
        "count",
    )

    def __init__(self, ngram: int = 8, window: int = 2048, history: int = 16384):
        self.ngram = max(1, int(ngram))
        self.window = max(1, int(window))
        self.history = max(self.window, int(history))
        self._buf: deque[int] = deque(maxlen=self.ngram)
        self._seen: set[tuple[int, ...]] = set()
        self._flags: deque[bool] = deque(maxlen=self.window)
        self._new = 0
        self._tokens: deque[int] = deque(maxlen=self.history)
        self.count = 0  # n-grams observed in this section

    def reset(self) -> None:
        self._buf.clear()
        self._seen.clear()
        self._flags.clear()
        self._new = 0
        self._tokens.clear()
        self.count = 0

    def push(self, tok: int) -> None:
        self._buf.append(tok)
        self._tokens.append(tok)
        if len(self._buf) < self.ngram:
            return
        g = tuple(self._buf)
        is_new = g not in self._seen
        if is_new:
            self._seen.add(g)
            if len(self._seen) > self.history:
                self._rebuild()
        if len(self._flags) == self.window:
            self._new -= self._flags[0]
        self._flags.append(is_new)
        self._new += is_new
        self.count += 1

    def push_many(self, toks: Iterable[int]) -> None:
        for t in toks:
            self.push(t)

    def _rebuild(self) -> None:
        toks = list(self._tokens)
        self._seen = {
            tuple(toks[i : i + self.ngram])
            for i in range(0, max(0, len(toks) - self.ngram + 1))
        }

    @property
    def ready(self) -> bool:
        return len(self._flags) >= self.window

    @property
    def novelty(self) -> float:
        n = len(self._flags)
        return 1.0 if n == 0 else self._new / n


def novelty_profile(
    tokens: list[int], ngram: int = 8, window: int = 2048, step: int = 256
) -> list[tuple[int, float]]:
    """Offline helper: (position, novelty) every ``step`` tokens. Used for
    threshold calibration against captured reasoning/answer token streams."""
    sn = SectionNovelty(ngram=ngram, window=window)
    out: list[tuple[int, float]] = []
    for i, t in enumerate(tokens, 1):
        sn.push(t)
        if i % step == 0 and sn.ready:
            out.append((i, round(sn.novelty, 3)))
    return out
