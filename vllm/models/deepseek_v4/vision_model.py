# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal wrapper for DeepSeek-V4-Flash-Vision-Exp.

Wraps the text-only ``DeepseekV4ForCausalLM`` with the checkpoint's ViT +
aligner and the sentinel-block embedding layout.

The image block is emitted as out-of-vocabulary token ids (``vocab_size +
type``, i.e. 129280..129284). Those ids never index the embedding table: the
LM reads them for MoE routing (``bias_vl`` on image positions, sentinel
masking before the hash-layer ``tid2eid`` lookup), so every position in the
block is a "multimodal" position and ``embed_multimodal`` returns the whole
block: the learned ``image_start/pad/newline/end`` vectors in their slots and
aligner outputs at ``IMAGE`` slots, permuted into the N-layout by ``perm``.

The ViT and aligner are plain ``nn.Module``s replicated on every TP rank. They
are built under the loader's default-dtype/device context, so their parameters
land in the model dtype (bf16 for this checkpoint) on the model device without
any explicit casts here.

Launching: ``vllm serve deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`` needs no
flags. The checkpoint declares ``architectures=["DeepseekV4ForCausalLM"]``;
``ModelRegistry._upgrade_deepseek_v4_vision_arch`` selects this wrapper when
``vision_n_layers > 0`` while the resolved architecture name stays the text
one, so ``tokenizer_mode`` defaults to ``deepseek_v4`` and the
``DeepseekV4ForCausalLM`` config hook (fp8 -> deepseek_v4_fp8) still applies.
Do NOT pass ``--hf-overrides '{"architectures": [...]}'``: that bypasses both
(you would also need ``--tokenizer-mode deepseek_v4``, and the fp8 quant
method would not be rewritten).

NVIDIA only: the ROCm/XPU ``DeepseekV4MoE`` variants register no ``bias_vl``
and keep the hash-layer ``gate.bias`` mapping, so the vision checkpoint cannot
load there; the constructor refuses early instead of failing at weight load.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM
from vllm.models.deepseek_v4.image_processing import IMAGE, IMAGE_PLACEHOLDER
from vllm.models.deepseek_v4.mm_preprocess import (
    DeepseekV4VDummyInputsBuilder,
    DeepseekV4VMultiModalProcessor,
    DeepseekV4VProcessingInfo,
)
from vllm.models.deepseek_v4.vision import DeepseekV4Aligner, DeepseekV4ViT
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

_LM_PREFIX = "language_model."
_LM_ARCH = "DeepseekV4ForCausalLM"


def _flat_sizes(x: object) -> list[int]:
    """Per-item sizes from a batched field: a stacked ``[B]``/``[B, 1]``
    tensor or a list of per-item tensors (0-d or ``[1]``)."""
    if isinstance(x, (list, tuple)):
        return [int(v) for t in x for v in torch.as_tensor(t).flatten().tolist()]
    return [int(v) for v in torch.as_tensor(x).flatten().tolist()]


def _cat_flat(x: object) -> torch.Tensor:
    """Concatenate a flat field that may arrive as one concatenated tensor
    or a list of per-item tensors."""
    if isinstance(x, (list, tuple)):
        return torch.cat([torch.as_tensor(t) for t in x], dim=0)
    return torch.as_tensor(x)


def _rows_of_two(x: object) -> torch.Tensor:
    """``[B, 2]`` int rows from a batched ``(n_h, n_w)`` field that may arrive
    stacked (``[B, 2]``) or as a list of ``[2]``/``[1, 2]`` tensors."""
    if isinstance(x, (list, tuple)):
        return torch.cat([torch.as_tensor(t).reshape(-1, 2) for t in x], dim=0)
    return torch.as_tensor(x).reshape(-1, 2)


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VMultiModalProcessor,
    info=DeepseekV4VProcessingInfo,
    dummy_inputs=DeepseekV4VDummyInputsBuilder,
)
class DeepseekV4VForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3
):
    """DeepSeek V4 vision wrapper.

    ``SupportsEagle3`` is required (not optional) on this fork: the new GPU
    runner turns on aux hidden-state outputs for ``method in ("eagle3",
    "dflash", "dspark")`` and ``set_eagle3_aux_hidden_state_layers`` raises
    unless ``supports_eagle3(model)`` (``vllm/v1/worker/gpu/model_runner.py``
    and ``spec_decode/eagle/eagle3_utils.py``). Both hooks are forwarded to
    the inner LM, which owns ``model.aux_hidden_state_layers``.
    """

    # The LM's MoE gate selects `bias_vl` over `bias` per position and the
    # hash layers index tid2eid by token id, so the raw ids must reach
    # forward() alongside the merged embeddings.
    requires_raw_input_tokens = True

    # Checkpoint keys are flat (``layers.N.*``, ``embed.weight``, ``head.weight``,
    # ``hc_head_*``, ``norm.weight``, ``mtp.N.*``). Route the LM keys under
    # ``language_model.`` and leave ``vision.*``, ``aligner.*`` and ``image_*``
    # untouched; the LM's own mapper then applies its ``model.``/``lm_head``/
    # ``.scale`` rules. Prefixes are applied before suffixes, and the only key
    # ending in ``head.weight`` is the LM head (``hc_head_*`` do not match).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "layers.": f"{_LM_PREFIX}layers.",
            "embed.": f"{_LM_PREFIX}embed.",
            "norm.": f"{_LM_PREFIX}norm.",
            "hc_head": f"{_LM_PREFIX}hc_head",
            "mtp.": f"{_LM_PREFIX}mtp.",
        },
        orig_to_new_suffix={"head.weight": f"{_LM_PREFIX}head.weight"},
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        # Mirrors the platform switch in vllm/models/deepseek_v4/__init__.py:
        # only the NVIDIA DeepseekV4MoE registers gate.bias_vl and drops the
        # dead hash-layer gate.bias, so only the NVIDIA LM can load this
        # checkpoint (amd/model.py and xpu/model.py would KeyError on
        # ``layers.0.ffn.gate.e_score_correction_bias`` / ``...gate.bias_vl``).
        if current_platform.is_rocm() or current_platform.is_xpu():
            raise NotImplementedError(
                "DeepseekV4VForConditionalGeneration (DeepSeek-V4 vision) is "
                "implemented for the NVIDIA DeepSeek-V4 stack only; the ROCm "
                "and XPU DeepseekV4MoE variants have no bias_vl routing."
            )

        with self._mark_tower_model(vllm_config, "image"):
            # Plain nn.Modules: never handed a quant_config, so the BF16 tower
            # cannot be picked up by DeepseekV4FP8Config.
            self.vision = DeepseekV4ViT(config)
            self.aligner = DeepseekV4Aligner(config)
            hidden = config.hidden_size
            self.image_start = nn.Parameter(torch.empty(hidden))
            self.image_pad = nn.Parameter(torch.empty(hidden))
            self.image_newline = nn.Parameter(torch.empty(hidden))
            self.image_end = nn.Parameter(torch.empty(hidden))

        with self._mark_language_model(vllm_config):
            # Same as init_vllm_registered_model(architectures=[_LM_ARCH]) but
            # with the class given explicitly: ModelRegistry upgrades the
            # text architecture to this wrapper whenever the config has
            # vision_n_layers > 0, so resolving the inner LM by name would
            # recurse into this constructor.
            from vllm.model_executor.model_loader.utils import initialize_model

            self.language_model = initialize_model(
                vllm_config=vllm_config.with_hf_config(
                    config, architectures=[_LM_ARCH]
                ),
                prefix=maybe_prefix(prefix, "language_model"),
                model_class=DeepseekV4ForCausalLM,
            )

        # SupportsPP: the runners read this attribute off the top-level model.
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # 129280..129284 are >= vocab_size; this makes the default
        # embed_input_ids mask them to 0 before the text-embedding gather.
        vocab_size = config.vocab_size
        self.configure_mm_token_handling(vocab_size, [vocab_size + i for i in range(5)])

    # ----------------------------------------------------------------- vision

    def _encode_one(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
        types: torch.Tensor,
        perm: torch.Tensor,
    ) -> torch.Tensor:
        """One image -> its full sentinel block ``[n_tokens, hidden]``.

        Mirrors the reference merge: ``block = stack[image_start, image_pad,
        image_pad, image_newline, image_end][types]`` and
        ``block[types == IMAGE] = aligner(vit(patches))[perm]``.
        """
        feats = self.vision(patches, n_vit_h, n_vit_w)
        embeds = self.aligner(feats, n_vit_h, n_vit_w)[perm]
        params = torch.stack(
            [
                self.image_start,
                self.image_pad,
                self.image_pad,
                self.image_newline,
                self.image_end,
            ]
        ).to(embeds.dtype)
        block = params[types]
        block[types == IMAGE] = embeds.to(block.dtype)
        return block

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches = kwargs.get("patches")
        if patches is None:
            return []

        # Batched fields arrive stacked when every item has the same shape and
        # as a list otherwise; flat fields arrive concatenated or as a list.
        patch_sizes = _flat_sizes(kwargs["patch_sizes"])
        type_sizes = _flat_sizes(kwargs["type_sizes"])
        perm_sizes = _flat_sizes(kwargs["perm_sizes"])
        patches_t = _cat_flat(patches)
        types_t = _cat_flat(kwargs["types"]).flatten()
        perm_t = _cat_flat(kwargs["perm"]).flatten()
        n_vit = _rows_of_two(kwargs["n_vit"])

        # Model device/dtype anchor. The ViT mixes fp32 RMSNorm weights with
        # bf16 linears, so its first parameter is not a reliable dtype source;
        # the sentinel vectors are created under the loader's default dtype.
        device, dtype = self.image_start.device, self.image_start.dtype

        out: list[torch.Tensor] = []
        po = to = ro = 0
        for i, n_patches in enumerate(patch_sizes):
            n_types, n_perm = type_sizes[i], perm_sizes[i]
            out.append(
                self._encode_one(
                    patches_t[po : po + n_patches].to(device=device, dtype=dtype),
                    int(n_vit[i, 0]),
                    int(n_vit[i, 1]),
                    types_t[to : to + n_types].to(device=device, dtype=torch.long),
                    perm_t[ro : ro + n_perm].to(device=device, dtype=torch.long),
                )
            )
            po += n_patches
            to += n_types
            ro += n_perm
        return out

    # --------------------------------------------------------------- language

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Both input_ids (raw, incl. the OOV sentinels for routing) and the
        # merged inputs_embeds reach the LM; DeepseekV4Model uses the embeds
        # and still hands the ids to every layer's MoE gate.
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # ---------------------------------------------- target-model delegations
    # The runner and the DSpark/MTP speculators read these off the *target*
    # model object (not the inner LM), so they are forwarded explicitly.

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual buffer for the MTP/DSpark drafter; the runner
        probes it with ``hasattr`` and would otherwise silently feed the
        post-hc_head hidden state to the draft model."""
        return self.language_model.get_mtp_target_hidden_states()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_default_aux_hidden_state_layers()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    # ---------------------------------------------------------------- weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint in one pass over the stream.

        ``AutoWeightsLoader`` groups the stream by *consecutive* top-level
        prefix (``itertools.groupby``) and would call
        ``language_model.load_weights`` once per group. Safetensors keys are
        sorted, so the shard holding the tower interleaves ``embed.weight``,
        ``image_*``, ``layers.*``, ``vision.*`` and splits the LM stream. The
        LM's ``load_weights`` ends with ``finalize_mega_moe_weights()``, whose
        expert transform is one-shot (it returns early once computed), so a
        split stream would freeze the mega-MoE weights on partially loaded
        experts. Hence: stream every LM key into a single
        ``language_model.load_weights`` call and buffer the (small) tower
        tensors for one ``AutoWeightsLoader`` pass afterwards.
        """
        tower_weights: list[tuple[str, torch.Tensor]] = []

        def lm_stream() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in self.hf_to_vllm_mapper.apply(weights):
                if name.startswith(_LM_PREFIX):
                    yield name[len(_LM_PREFIX) :], weight
                else:
                    tower_weights.append((name, weight))

        loaded: set[str] = {
            _LM_PREFIX + name for name in self.language_model.load_weights(lm_stream())
        }

        tower_loader = AutoWeightsLoader(self, skip_prefixes=[_LM_PREFIX])
        loaded |= tower_loader.load_weights(tower_weights)
        return loaded

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="aligner",
            tower_model="vision",
        )
