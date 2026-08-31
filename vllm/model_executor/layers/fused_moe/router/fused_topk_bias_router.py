# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (
    can_use_dsv4_topk,
    dsv4_topk,
)


def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        return is_padding[:num_tokens] if is_padding is not None else None
    return None


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


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
    # Using biased scores as weights flattens the distribution when the bias
    # is near-uniform (e.g., DSv4-Flash where all biases ≈ 8.08).
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


def vllm_topk_softplus_sqrt(
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
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return _topk_softplus_sqrt_torch(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )

    ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_tokens,
        hash_indices_table,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


@functools.lru_cache(maxsize=8)
def _aiter_get_num_expert_group(num_experts: int) -> int:
    _AITER_MAX_EXPERTS_PER_GROUP = 32
    g = max(1, -(-num_experts // _AITER_MAX_EXPERTS_PER_GROUP))
    while num_experts % g != 0:
        g += 1
    assert num_experts % g == 0, f"{num_experts=} not divisible by {g=}"
    assert num_experts // g <= _AITER_MAX_EXPERTS_PER_GROUP, (
        f"group size {num_experts // g} exceeds limit {_AITER_MAX_EXPERTS_PER_GROUP}"
    )
    return g


def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
):
    if (
        input_tokens is not None
        and hash_indices_table is not None
        and input_tokens.dtype != hash_indices_table.dtype
    ):
        input_tokens = input_tokens.to(dtype=hash_indices_table.dtype)

    if not rocm_aiter_ops.is_fused_moe_enabled():
        assert hidden_states.size(0) == gating_output.size(0), (
            "Number of tokens mismatch"
        )

        output_indices_dtype = torch.int32 if indices_type is None else indices_type
        if scoring_func == "sqrtsoftplus" and can_use_dsv4_topk(
            gating_output,
            e_score_correction_bias,
            topk,
            renormalize,
            output_indices_dtype,
        ):
            assert e_score_correction_bias is not None
            return dsv4_topk(
                gating_output,
                e_score_correction_bias,
                output_indices_dtype,
                routed_scaling_factor,
            )

        M, _ = hidden_states.size()

        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device,
        )
        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )

        if scoring_func == "softmax":
            topk_weights, topk_ids = vllm_topk_softmax(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
            )
            if routed_scaling_factor != 1.0:
                topk_weights *= routed_scaling_factor
            return topk_weights, topk_ids
        elif scoring_func == "sigmoid":
            topk_weights, topk_ids = vllm_topk_sigmoid(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
                routed_scaling_factor,
            )
            return topk_weights, topk_ids
        elif scoring_func == "sqrtsoftplus":
            return vllm_topk_softplus_sqrt(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
                input_tokens,
                hash_indices_table,
                routed_scaling_factor,
            )
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

    elif rocm_aiter_ops.is_fused_moe_enabled() and scoring_func == "sigmoid":
        M = hidden_states.size(0)
        num_experts = gating_output.shape[-1]
        num_expert_group = _aiter_get_num_expert_group(num_experts)
        if topk >= num_expert_group:
            topk_weights = torch.empty(
                M, topk, dtype=torch.float32, device=hidden_states.device
            )
            topk_ids = torch.empty(
                M,
                topk,
                dtype=torch.int32 if indices_type is None else indices_type,
                device=hidden_states.device,
            )
            rocm_aiter_ops.biased_grouped_topk(
                gating_output,
                e_score_correction_bias,
                topk_weights,
                topk_ids,
                num_expert_group=num_expert_group,
                topk_group=num_expert_group,
                need_renorm=renormalize,
            )
            if routed_scaling_factor != 1.0:
                topk_weights *= routed_scaling_factor
            return topk_weights, topk_ids

    if scoring_func == "sqrtsoftplus":
        M = hidden_states.size(0)
        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device,
        )
        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )
        return vllm_topk_softplus_sqrt(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            e_score_correction_bias,
            input_tokens,
            hash_indices_table,
            routed_scaling_factor,
        )

    n_routed_experts = gating_output.shape[-1]
    if scoring_func == "softmax":
        scores = gating_output.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    if e_score_correction_bias is not None:
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores.view(-1, n_routed_experts)
    # For batch invariance, use sorted=True to ensure deterministic expert selection
    if hash_indices_table is not None:
        assert input_tokens is not None
        topk_indices = hash_indices_table[input_tokens]
    else:
        use_sorted = envs.VLLM_BATCH_INVARIANT
        topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[
            1
        ]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(torch.float32)
    if routed_scaling_factor != 1.0:
        topk_weights *= routed_scaling_factor
    return topk_weights, topk_indices.to(
        torch.int32 if indices_type is None else indices_type
    )


def fused_topk_bias_vl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor | None,
    bias_vl: torch.Tensor,
    vl_vocab_size: int,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 vision routing: two ``fused_topk_bias`` passes merged per row.

    Image positions are the rows whose ``input_tokens`` are out-of-vocabulary
    sentinels (``>= vl_vocab_size``). They route by ``topk(scores + bias_vl)``
    on every layer, hash layers included; text rows keep the unchanged text
    routing (``tid2eid`` lookup on hash layers, ``topk(scores + bias)``
    elsewhere). Sentinel ids are masked to 0 before the ``tid2eid`` gather.
    This mirrors the reference ``Gate.forward`` exactly, and each pass still
    runs through the existing CUDA/Triton kernels (both take one bias vector).
    """
    # Lazy import: this module is imported while the fused_moe package is
    # initializing, and ``vllm.models.deepseek_v4`` pulls that package back in.
    from vllm.models.deepseek_v4.vl_routing import (
        merge_vl_routing,
        sanitize_vl_input_ids,
    )

    if input_tokens is None:
        raise ValueError("DeepSeek V4 vision MoE routing (bias_vl) requires input_ids.")
    image_mask, safe_ids = sanitize_vl_input_ids(input_tokens, vl_vocab_size)
    # CUDA-graph padding rows keep stale input ids (the runner only rewrites
    # ids[:num_tokens]), which can be sentinels from an earlier image batch.
    # The text pass honours the padding mask inside its kernels (hash layers
    # emit expert id -1 / weight 0 for padded rows) but the image pass runs
    # dsv4_topk, which has no is_padding argument; keep the text pass on
    # padded rows so they never cost real expert work. Still branch-free on
    # tensor data (the None check is host-side and capture-invariant).
    is_padding = _get_padding_mask(image_mask.shape[0])
    if is_padding is not None:
        image_mask = image_mask & ~is_padding.to(torch.bool)
    text_weights, text_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        indices_type=indices_type,
        input_tokens=safe_ids,
        hash_indices_table=hash_indices_table,
        routed_scaling_factor=routed_scaling_factor,
    )
    vl_weights, vl_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func=scoring_func,
        e_score_correction_bias=bias_vl,
        topk=topk,
        renormalize=renormalize,
        indices_type=indices_type,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=routed_scaling_factor,
    )
    return merge_vl_routing(text_weights, text_ids, vl_weights, vl_ids, image_mask)


class FusedTopKBiasRouter(BaseRouter):
    """Router using fused top-k with e_score_correction_bias.

    ``bias_vl`` / ``vl_vocab_size`` are optional attributes set after
    construction by ``DeepseekV4MoE`` for vision checkpoints; when present and
    ``input_ids`` are given, routing goes through ``fused_topk_bias_vl``.
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        e_score_correction_bias: torch.Tensor | None = None,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        eplb_state: EplbLayerState | None = None,
        *,
        scoring_func: str = "sigmoid",
        hash_indices_table: torch.Tensor | None = None,
        num_fused_shared_experts: int = 0,
        shared_expert_weight: float = 1.0,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.e_score_correction_bias = e_score_correction_bias
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self._hash_indices_table = hash_indices_table
        # DeepSeek-V4 vision: modality-specific selection bias for image rows.
        self.bias_vl: torch.Tensor | None = None
        self.vl_vocab_size: int | None = None
        # Fused shared experts: append constant slots (ids immediately after
        # the routed experts, [global, global+n)) routed to by every token at
        # ``shared_expert_weight``, AFTER the routed top-k is renormalized.
        self.num_fused_shared_experts = num_fused_shared_experts
        self.shared_expert_weight = shared_expert_weight

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=True,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using fused top-k with bias."""
        bias_vl = getattr(self, "bias_vl", None)
        if bias_vl is not None:
            # Never fall back to text routing for a vision checkpoint: image
            # rows would silently take ``bias`` instead of ``bias_vl`` (e.g.
            # PP ranks > 0, which receive input_ids=None from the runner).
            if input_ids is None:
                raise ValueError(
                    "DeepSeek V4 vision MoE routing (bias_vl) requires input_ids."
                )
            assert self.vl_vocab_size is not None
            topk_weights, topk_ids = fused_topk_bias_vl(
                hidden_states=hidden_states,
                gating_output=router_logits,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias.data
                if self.e_score_correction_bias is not None
                else None,
                bias_vl=bias_vl.data,
                vl_vocab_size=self.vl_vocab_size,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
                input_tokens=input_ids,
                hash_indices_table=self._hash_indices_table,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias.data
                if self.e_score_correction_bias is not None
                else None,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
                input_tokens=input_ids,
                hash_indices_table=self._hash_indices_table,
                routed_scaling_factor=self.routed_scaling_factor,
            )

        if self.num_fused_shared_experts > 0:
            m = topk_ids.shape[0]
            n = self.num_fused_shared_experts
            # global_num_experts counts only the routed experts; the fused
            # shared experts occupy the slots immediately after them, i.e. ids
            # [global_num_experts, global_num_experts + n).
            base = self.global_num_experts
            shared_ids = torch.arange(
                base, base + n, dtype=topk_ids.dtype, device=topk_ids.device
            ).expand(m, n)
            shared_w = torch.full(
                (m, n),
                self.shared_expert_weight,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
            topk_ids = torch.cat([topk_ids, shared_ids], dim=-1)
            topk_weights = torch.cat([topk_weights, shared_w], dim=-1)

        return topk_weights, topk_ids
