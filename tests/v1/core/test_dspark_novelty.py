# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

from vllm.v1.core.sched.dspark_novelty import SectionNovelty, novelty_profile


def test_fresh_stream_is_novel():
    sn = SectionNovelty(ngram=4, window=64)
    sn.push_many(range(1000))
    assert sn.ready and sn.novelty == 1.0


def test_exact_cycle_collapses_to_zero():
    sn = SectionNovelty(ngram=4, window=64)
    period = list(range(300))  # longer than any exact-cycle cap
    sn.push_many(period * 3)
    assert sn.ready and sn.novelty == 0.0


def test_paraphrase_loop_drops_below_threshold():
    rnd = random.Random(0)
    base = [rnd.randrange(50000) for _ in range(400)]
    sn = SectionNovelty(ngram=8, window=1024)
    sn.push_many(base)
    for _ in range(6):  # re-derive with ~3% token edits each time
        var = [t if rnd.random() > 0.03 else rnd.randrange(50000) for t in base]
        sn.push_many(var)
    assert sn.ready and sn.novelty < 0.4


def test_window_slides_and_recovers():
    sn = SectionNovelty(ngram=4, window=32)
    sn.push_many([1, 2, 3, 4] * 40)
    assert sn.novelty == 0.0
    sn.push_many(range(100, 200))
    assert sn.novelty == 1.0


def test_history_cap_rebuild_keeps_running():
    sn = SectionNovelty(ngram=4, window=16, history=64)
    sn.push_many(range(5000))
    assert sn.ready and 0.0 <= sn.novelty <= 1.0
    assert len(sn._seen) <= 64 + 1


def test_reset():
    sn = SectionNovelty(ngram=4, window=8)
    sn.push_many([1, 2, 3, 4] * 10)
    sn.reset()
    assert not sn.ready and sn.novelty == 1.0 and sn.count == 0


def test_profile_helper():
    toks = list(range(600)) + list(range(300)) * 4
    prof = novelty_profile(toks, ngram=4, window=256, step=128)
    assert prof[0][1] > 0.9 and prof[-1][1] < 0.1
