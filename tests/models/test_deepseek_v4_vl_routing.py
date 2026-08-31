# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 vision (``bias_vl``) MoE routing against the reference Gate.

Pure torch, no ``vllm`` import: ``vl_routing.py`` is loaded by path and the
routing primitive is a verbatim copy of ``_topk_softplus_sqrt_torch`` from
``vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`` (the
CPU/XPU fallback that the CUDA op and the Triton kernel are checked against).
The test reproduces what ``fused_topk_bias_vl`` does: a text pass (hash table
or text bias, sentinel ids masked to 0), an image pass (``bias_vl``, no hash
table) and a per-row ``torch.where`` merge, and asserts it equals the reference
``Gate.forward`` from DeepSeek's ``model.py`` on both a hash layer and a normal
layer.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VL_ROUTING_PATH = _REPO_ROOT / "vllm" / "models" / "deepseek_v4" / "vl_routing.py"


def _load_vl_routing():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_vl_routing", _VL_ROUTING_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vl_routing = _load_vl_routing()

VOCAB_SIZE = 64
N_EXPERTS = 32
TOPK = 6
ROUTE_SCALE = 1.5
RENORMALIZE = True


# --- routing primitive: copy of fused_topk_bias_router._topk_softplus_sqrt_torch
def _topk_softplus_sqrt_torch(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Pure PyTorch fallback for topk_softplus_sqrt (XPU/CPU)."""
    # scores = sqrt(softplus(gating_output))
    scores = torch.sqrt(F.softplus(gating_output.float()))

    # Bias is used for expert SELECTION only, not for weight computation.
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.float()
    else:
        scores_for_choice = scores

    topk = topk_weights.shape[-1]

    if hash_indices_table is not None and input_tokens is not None:
        # Hash MoE: expert indices predetermined by lookup table
        # hash_indices_table: [vocab_size, topk] mapping token_id -> expert_ids
        expert_ids = hash_indices_table[input_tokens.long()]  # [M, topk]
        topk_indices.copy_(expert_ids)
        # Gather weights from unbiased scores
        weights = scores.gather(1, expert_ids.long())
    else:
        # Standard topk selection using biased scores
        _, indices = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_indices.copy_(indices)
        # Gather weights from unbiased scores
        weights = scores.gather(1, indices)

    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True).clamp(min=1e-20))

    topk_weights.copy_(weights * routed_scaling_factor)
    return topk_weights, topk_indices


def _routed(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    table: torch.Tensor | None,
    indices_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """What ``fused_topk_bias`` allocates + calls on the fallback path."""
    m = logits.shape[0]
    topk_weights = torch.empty(m, TOPK, dtype=torch.float32)
    topk_ids = torch.empty(m, TOPK, dtype=indices_dtype)
    token_expert_indices = torch.empty(m, TOPK, dtype=torch.int32)
    return _topk_softplus_sqrt_torch(
        topk_weights,
        topk_ids,
        token_expert_indices,
        logits,
        RENORMALIZE,
        bias,
        input_tokens,
        table,
        ROUTE_SCALE,
    )


def ported_vl_routing(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    bias_vl: torch.Tensor,
    table: torch.Tensor | None,
    text_indices_dtype: torch.dtype = torch.int32,
    vl_indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of ``fused_topk_bias_vl`` on top of the torch fallback."""
    image_mask, safe_ids = vl_routing.sanitize_vl_input_ids(input_ids, VOCAB_SIZE)
    text_weights, text_ids = _routed(logits, bias, safe_ids, table, text_indices_dtype)
    vl_weights, vl_ids = _routed(logits, bias_vl, None, None, vl_indices_dtype)
    return vl_routing.merge_vl_routing(
        text_weights, text_ids, vl_weights, vl_ids, image_mask
    )


# --- reference: DeepSeek model.py Gate.forward (score_func = sqrtsoftplus)
def reference_gate(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    hash_layer: bool,
    bias: torch.Tensor | None,
    bias_vl: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(logits.float()).sqrt()
    original_scores = scores
    image_mask = (input_ids >= VOCAB_SIZE) if bias_vl is not None else None
    # Bias shifts scores for expert selection (topk) but does not affect
    # routing weights.
    if hash_layer:
        assert tid2eid is not None
        if image_mask is None:
            indices = tid2eid[input_ids]
        else:
            indices = tid2eid[torch.where(image_mask, 0, input_ids)]
            vl_indices = (scores + bias_vl).topk(TOPK, dim=-1)[1]
            indices = torch.where(
                image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices
            )
    else:
        assert bias is not None
        if image_mask is None:
            scores = scores + bias
        else:
            scores = scores + torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
        indices = scores.topk(TOPK, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    weights /= weights.sum(dim=-1, keepdim=True)
    weights *= ROUTE_SCALE
    return weights, indices


def _make_inputs(seed: int, m: int, image_fraction: float):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(m, N_EXPERTS, generator=g) * 3.0
    bias = torch.randn(N_EXPERTS, generator=g) + 8.0
    bias_vl = torch.randn(N_EXPERTS, generator=g) + 8.0
    tid2eid = torch.stack(
        [torch.randperm(N_EXPERTS, generator=g)[:TOPK] for _ in range(VOCAB_SIZE)]
    ).to(torch.int32)
    text_ids = torch.randint(0, VOCAB_SIZE, (m,), generator=g)
    is_image = torch.rand(m, generator=g) < image_fraction
    sentinel_type = torch.randint(0, 5, (m,), generator=g)
    input_ids = torch.where(is_image, VOCAB_SIZE + sentinel_type, text_ids)
    return logits, bias, bias_vl, tid2eid, input_ids, is_image


def test_sanitize_vl_input_ids_masks_only_sentinels():
    _, _, _, _, input_ids, is_image = _make_inputs(0, 128, 0.5)
    image_mask, safe_ids = vl_routing.sanitize_vl_input_ids(input_ids, VOCAB_SIZE)
    assert torch.equal(image_mask, is_image)
    assert safe_ids.dtype == input_ids.dtype
    assert bool((safe_ids < VOCAB_SIZE).all())
    assert torch.equal(safe_ids[~is_image], input_ids[~is_image])
    assert bool((safe_ids[is_image] == 0).all())


@pytest.mark.parametrize("hash_layer", [True, False], ids=["hash", "normal"])
@pytest.mark.parametrize("image_fraction", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_vl_routing_matches_reference_gate(hash_layer, image_fraction, seed):
    logits, bias, bias_vl, tid2eid, input_ids, _ = _make_inputs(
        seed, 96, image_fraction
    )
    ref_w, ref_ids = reference_gate(
        logits,
        input_ids,
        hash_layer=hash_layer,
        bias=None if hash_layer else bias,
        bias_vl=bias_vl,
        tid2eid=tid2eid if hash_layer else None,
    )
    got_w, got_ids = ported_vl_routing(
        logits,
        input_ids,
        bias=None if hash_layer else bias,
        bias_vl=bias_vl,
        table=tid2eid if hash_layer else None,
    )
    assert torch.equal(got_ids.long(), ref_ids.long())
    torch.testing.assert_close(got_w, ref_w, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("hash_layer", [True, False], ids=["hash", "normal"])
def test_text_rows_unchanged_by_vl_pass(hash_layer):
    """With no image rows the merge is the text pass, bit for bit."""
    logits, bias, bias_vl, tid2eid, input_ids, _ = _make_inputs(7, 64, 0.0)
    table = tid2eid if hash_layer else None
    text_bias = None if hash_layer else bias
    text_w, text_ids = _routed(logits, text_bias, input_ids, table, torch.int32)
    got_w, got_ids = ported_vl_routing(
        logits, input_ids, bias=text_bias, bias_vl=bias_vl, table=table
    )
    assert torch.equal(got_ids, text_ids)
    assert torch.equal(got_w, text_w)


def test_merge_casts_image_ids_to_text_dtype():
    logits, bias, bias_vl, tid2eid, input_ids, _ = _make_inputs(11, 48, 0.5)
    got_w, got_ids = ported_vl_routing(
        logits,
        input_ids,
        bias=None,
        bias_vl=bias_vl,
        table=tid2eid,
        text_indices_dtype=torch.int32,
        vl_indices_dtype=torch.int64,
    )
    ref_w, ref_ids = reference_gate(
        logits, input_ids, hash_layer=True, bias=None, bias_vl=bias_vl, tid2eid=tid2eid
    )
    assert got_ids.dtype == torch.int32
    assert got_w.dtype == torch.float32
    assert torch.equal(got_ids.long(), ref_ids.long())
    torch.testing.assert_close(got_w, ref_w, rtol=1e-6, atol=1e-6)
