# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Model Runner V2 Gumbel-max sampling kernel.

Accuracy: define a target categorical distribution as a non-negative int64
count tensor summing to N, turn it into logits (= log(count)), sample many
times with `gumbel_sample`, and check the empirical distribution matches.

The count tensor is deliberately heavy-tailed (one dominant token, the rest
~18 logits below). That tail is the sensitive part: the fp32 Gumbel noise must
reach ~18 to ever sample it. A flat distribution would keep every token within
a few logits of the top and would not exercise the noise tail at all.
"""

import math

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Gumbel sampler tests", allow_module_level=True)

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

DEVICE = "cuda"
VOCAB_SIZE = 200_000
NUM_SAMPLES = 500_000
# Dominant token is exp(HEAD_LOG_GAP)x larger than the unit-count tail, so the
# tail sits ~HEAD_LOG_GAP logits below the top.
HEAD_LOG_GAP = 18.0
# 10-sigma band: a correct sampler effectively never trips it.
Z_TOLERANCE = 10.0


def _make_heavy_tailed_counts(seed: int = 1234) -> torch.Tensor:
    """Non-negative int64 counts of shape [VOCAB_SIZE]; target prob = counts/N."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    counts = torch.randint(
        1, 4, (VOCAB_SIZE,), generator=gen, dtype=torch.int64, device=DEVICE
    )
    counts[0] = round(math.exp(HEAD_LOG_GAP))  # dominant token
    return counts


def _counts_to_logits(counts: torch.Tensor) -> torch.Tensor:
    # softmax(log(count)) == count / sum(count); count 0 -> logit -inf -> prob 0.
    return counts.double().log().to(torch.float32)


def _sample(
    logits_1d: torch.Tensor,
    num_samples: int,
    *,
    use_fp64: bool = False,
    temperature: float = 1.0,
    is_drafting: bool = False,
) -> torch.Tensor:
    """Sample `num_samples` tokens from one logit vector.

    Fixed seed with a distinct `pos` per sample gives independent draws; the
    logits are broadcast with a 0-stride view to avoid materializing
    [num_samples, vocab_size].
    """
    vocab_size = logits_1d.shape[0]
    logits = logits_1d.unsqueeze(0).expand(num_samples, vocab_size)
    idx_mapping = torch.zeros(num_samples, dtype=torch.int32, device=DEVICE)
    temp = torch.tensor([temperature], dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_samples, dtype=torch.int64, device=DEVICE)
    return gumbel_sample(
        logits,
        idx_mapping,
        temp,
        seed,
        pos,
        apply_temperature=True,
        is_drafting=is_drafting,
        use_fp64=use_fp64,
    )


def _z_score(observed: int, expected: float, num_trials: int) -> float:
    p = expected / num_trials
    return (observed - expected) / math.sqrt(num_trials * p * (1 - p))


def _sample_histogram(
    logits_1d: torch.Tensor,
    num_samples: int,
    *,
    chunk: int = 1_000_000,
    is_drafting: bool = False,
) -> torch.Tensor:
    """Histogram of `num_samples` draws, accumulated in chunks.

    Chunking keeps the kernel's per-sample scratch ([chunk, num_blocks]) bounded
    so a large sample count does not blow up memory.
    """
    vocab_size = logits_1d.shape[0]
    hist = torch.zeros(vocab_size, dtype=torch.float64, device=DEVICE)
    for start in range(0, num_samples, chunk):
        size = min(chunk, num_samples - start)
        logits = logits_1d.unsqueeze(0).expand(size, vocab_size)
        idx_mapping = torch.zeros(size, dtype=torch.int32, device=DEVICE)
        temp = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
        seed = torch.tensor([0xABCD], dtype=torch.int64, device=DEVICE)
        pos = torch.arange(start, start + size, dtype=torch.int64, device=DEVICE)
        out = gumbel_sample(
            logits,
            idx_mapping,
            temp,
            seed,
            pos,
            apply_temperature=True,
            is_drafting=is_drafting,
        )
        hist += torch.bincount(out, minlength=vocab_size).double()
    return hist


# ----------------------------- Accuracy ------------------------------------


@pytest.mark.parametrize("use_fp64", [False, True])
def test_sampling_matches_target_distribution(use_fp64: bool):
    counts = _make_heavy_tailed_counts()
    total = counts.sum().item()
    logits = _counts_to_logits(counts)

    sampled = _sample(logits, NUM_SAMPLES, use_fp64=use_fp64)
    assert sampled.min() >= 0 and sampled.max() < VOCAB_SIZE

    # The dominant token (index 0) and the aggregate tail are the two
    # statistically resolvable bins (individual tail tokens are far below the
    # ~5/N detectability floor). The tail mass is small but well above noise,
    # and it lives beyond the fp32 Gumbel cap -- the regime sensitive to noise
    # precision -- so matching it is the meaningful check.
    tail_prob = (total - counts[0].item()) / total
    tail_count = (sampled != 0).sum().item()
    z = _z_score(tail_count, NUM_SAMPLES * tail_prob, NUM_SAMPLES)
    assert abs(z) < Z_TOLERANCE, (
        f"sampled tail mass {tail_count / NUM_SAMPLES:.3e} != target "
        f"{tail_prob:.3e} (z={z:.2f})"
    )


def test_full_vocab_distribution_fidelity():
    """The sampled distribution matches the target across the WHOLE vocab.

    A near-flat count tensor makes every one of the 200K bins individually
    measurable. With ~20 samples/bin, a goodness-of-fit over all bins checks
    that no part of the vocab is over- or under-represented (the heavy-tailed
    test above only resolves head vs aggregate tail). Empirically the fp32
    sampler is as faithful here as torch.multinomial; the residual error is the
    multinomial sampling-noise floor, not the kernel.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(2024)
    counts = torch.randint(
        500, 1500, (VOCAB_SIZE,), generator=gen, dtype=torch.int64, device=DEVICE
    )
    total = counts.sum().item()
    logits = _counts_to_logits(counts)

    num_samples = 4_000_000
    hist = _sample_histogram(logits, num_samples)

    # Diversity: essentially every token must be reachable (no starved region).
    coverage = (hist > 0).sum().item() / VOCAB_SIZE
    assert coverage > 0.99, f"only {coverage:.4f} of the vocab was ever sampled"

    # Goodness-of-fit across all bins (each has expected count >= ~10).
    expected = (counts.double() / total) * num_samples
    chi2 = (((hist - expected) ** 2) / expected).sum().item()
    df = VOCAB_SIZE - 1
    assert chi2 < df + 10 * math.sqrt(2 * df), f"chi2={chi2:.0f}, df={df}"


# --------------------------- Noise streams ---------------------------------


def test_drafting_uses_a_separate_noise_stream():
    """is_drafting salts the Philox offset: same inputs, different draws.

    The draft proposal and the residual resample after a rejection must be
    independent. They key noise by the same (seed, pos), so only the salt keeps
    them apart -- without it the resample inherits the very noise vector that
    picked the rejected proposal. See test_stochastic_rejection_sample in
    tests/v1/spec_decode/test_rejection_sampler_utils.py for the distributional
    consequence.

    Relocating the offset must not distort the draw either, so both streams are
    checked against the target's far-tail mass, which sits HEAD_LOG_GAP logits
    below the head -- the regime where fp32 Gumbel precision matters.
    """
    counts = _make_heavy_tailed_counts()
    total = counts.sum().item()
    logits = _counts_to_logits(counts)
    tail_prob = (total - counts[0].item()) / total

    target = _sample(logits, NUM_SAMPLES, is_drafting=False)
    draft = _sample(logits, NUM_SAMPLES, is_drafting=True)

    # The head dominates, so the streams agree on most draws by construction.
    # Compare instead which draws leave the head: shared noise makes that
    # identical, independent noise makes them differ on ~2p(1-p) of draws.
    target_tail = target != 0
    draft_tail = draft != 0
    disagree = (target_tail != draft_tail).double().mean().item()
    assert disagree > tail_prob, (
        f"streams leave the head on the same draws ({disagree:.3e} disagreement "
        f"vs tail mass {tail_prob:.3e}); the draft salt is not taking effect"
    )

    # Both streams must still reproduce the target's tail mass.
    for name, tail in (("target", target_tail), ("draft", draft_tail)):
        tail_count = tail.sum().item()
        z = _z_score(tail_count, NUM_SAMPLES * tail_prob, NUM_SAMPLES)
        assert abs(z) < Z_TOLERANCE, (
            f"{name} tail mass {tail_count / NUM_SAMPLES:.3e} != "
            f"{tail_prob:.3e} (z={z:.2f})"
        )

    # The draft stream is reproducible.
    assert torch.equal(draft, _sample(logits, NUM_SAMPLES, is_drafting=True))


# ----------------------------- Edge cases ----------------------------------


def test_greedy_temperature_zero_returns_argmax():
    """temperature == 0 skips Gumbel noise and returns the exact argmax."""
    torch.manual_seed(0)
    num_reqs = 128
    logits = torch.randn(num_reqs, VOCAB_SIZE, device=DEVICE, dtype=torch.float32)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    temp = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)

    sampled = gumbel_sample(
        logits, idx_mapping, temp, seed, pos, apply_temperature=True, is_drafting=False
    )
    assert torch.equal(sampled, logits.argmax(dim=-1))


def test_zero_count_tokens_are_never_sampled():
    """Count 0 -> -inf logit -> probability 0; must never be selected."""
    counts = _make_heavy_tailed_counts(seed=7)
    zeroed = torch.arange(1, VOCAB_SIZE, 2, device=DEVICE)  # odd indices (not head)
    counts[zeroed] = 0
    logits = _counts_to_logits(counts)

    sampled = _sample(logits, NUM_SAMPLES)
    assert sampled.min() >= 0 and sampled.max() < VOCAB_SIZE
    assert not torch.isin(sampled, zeroed).any(), "sampled a zero-probability token"


def test_single_nonzero_token_is_always_sampled():
    """A lone finite logit must win every draw, regardless of its index."""
    counts = torch.zeros(VOCAB_SIZE, dtype=torch.int64, device=DEVICE)
    counts[123_456] = 1000
    logits = _counts_to_logits(counts)

    sampled = _sample(logits, 10_000)
    assert (sampled == 123_456).all()


@pytest.mark.parametrize("vocab_size", [1, 999, 1024, 4097])
def test_vocab_size_not_multiple_of_block(vocab_size: int):
    """Per-block tail masking for non-block-aligned vocab; all bins measurable."""
    gen = torch.Generator(device=DEVICE).manual_seed(vocab_size)
    counts = torch.randint(
        20, 200, (vocab_size,), generator=gen, dtype=torch.int64, device=DEVICE
    )
    total = counts.sum().item()
    logits = _counts_to_logits(counts)
    num_samples = max(40 * vocab_size, 50_000)

    sampled = _sample(logits, num_samples)
    assert sampled.min() >= 0 and sampled.max() < vocab_size

    observed = torch.bincount(sampled, minlength=vocab_size).double()
    expected = (counts.double() / total) * num_samples
    chi2 = (((observed - expected) ** 2) / expected).sum().item()
    df = vocab_size - 1
    if df >= 1:
        assert chi2 < df + 10 * math.sqrt(2 * df), f"chi2={chi2:.1f}, df={df}"
