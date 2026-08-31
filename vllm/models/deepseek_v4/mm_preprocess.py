# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multimodal processor for DeepSeek-V4-Flash-Vision-Exp.

Ported from the community ``deepseek-v4-vision`` branch onto this base
(v0.27.1). The reference expands every ``<｜deepseek_image｜>`` placeholder into
a block of out-of-vocabulary sentinel ids (``vocab_size + type``) whose leading
pad is ``3 - start_pos % 4``: the block depends on the absolute offset of the
placeholder in the prompt. That drives three deviations from the usual vLLM
processor shape:

* The renderer tokenizes text prompts before the multimodal processor runs
  (``BaseRenderer._tokenize_singleton_prompt`` -> ``_process_tokens`` ->
  ``_process_multimodal(prompt_token_ids, ...)``), so in serving the processor
  receives a token-id prompt, never text. ``_apply_hf_processor_main`` expands
  token prompts directly at the true offsets; ``_call_hf_processor`` (text)
  only runs for the profiling dummies.
* The per-item processor cache assumes an item's expansion is independent of
  the prompt it lands in. Ours is not, so ``_cached_apply_hf_processor`` always
  takes the whole-prompt, uncached path.
* The engine receiver cache and the encoder cache are keyed by the item hash
  and return the cached kwargs / encoder output for a known hash. Both depend
  on the offset here, so the block length is appended to the hash.

``_get_prompt_updates`` only lets the framework re-discover the blocks that are
already in the prompt so it can record their placeholder ranges; every position
of a block (pads, IMAGE_START/END included) receives a multimodal embedding.
"""

from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
)
from vllm.multimodal.processing.processor import MultiModalProcessingInfo

from .image_processing import (
    COMPRESS_PAD_TO,
    IMAGE_PLACEHOLDER,
    as_pil_image,
    build_image_block,
    max_tokens_image_size,
    num_image_tokens,
    preprocess_image,
    solve_image_grid,
)


def _image_list(mm_data: Mapping[str, object]) -> list[object]:
    # ``_get_hf_mm_data`` keys processor data by the HF convention
    # (``f"{modality}s"`` -> "images"), not by the vLLM modality name.
    images = mm_data.get("images")
    if images is None:
        images = mm_data.get("image")
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


class DeepseekV4VProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_token_id(self) -> int:
        tokenizer = self.get_tokenizer()
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if image_token_id is None or (
            unk_token_id is not None and image_token_id == unk_token_id
        ):
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        return image_token_id

    def get_max_image_tokens(self) -> int:
        # The solver keeps the grid (IMAGE_START/END and grid pads included)
        # within ``vision_max_n_token - 3`` and the leading pad adds at most 3,
        # so a block never exceeds ``vision_max_n_token``. Reserve a little
        # more so the budget is a strict upper bound regardless of offset.
        return self.get_hf_config().vision_max_n_token + COMPRESS_PAD_TO - 1

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        return {"image": self.get_max_image_tokens()}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        start_pos: int = 0,
    ) -> int:
        cfg = self.get_hf_config()
        _, _, n_llm_h, n_llm_w, _, _ = solve_image_grid(image_width, image_height, cfg)
        return num_image_tokens(n_llm_h, n_llm_w, start_pos)

    def get_image_size_with_most_features(self) -> ImageSize:
        width, height = max_tokens_image_size(self.get_hf_config())
        return ImageSize(width=width, height=height)


class DeepseekV4VDummyInputsBuilder(BaseDummyInputsBuilder[DeepseekV4VProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class DeepseekV4VMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VProcessingInfo]
):
    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        # Always the uncached, whole-prompt path (see module docstring).
        return self._apply_hf_processor(inputs, timing_ctx)

    def _apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        prompt_ids, mm_info, is_update_applied = super()._apply_hf_processor(
            inputs, timing_ctx
        )

        # The same image at a different offset has a different block length,
        # different kwargs and a different encoder output: key the engine-side
        # caches (receiver cache, encoder cache) by the block length too.
        image_items = mm_info.kwargs.get("image")
        image_hashes = mm_info.hashes.get("image")
        if image_items and image_hashes:
            hashes = dict(mm_info.hashes)
            hashes["image"] = [
                f"{item_hash}-n{int(item['type_sizes'].data)}"
                for item_hash, item in zip(image_hashes, image_items)
            ]
            mm_info = MultiModalProcessingInfo(
                kwargs=mm_info.kwargs,
                hashes=hashes,
                prompt_updates=mm_info.prompt_updates,
            )

        return prompt_ids, mm_info, is_update_applied

    def _expand_image_tokens(
        self,
        prompt_tokens: Sequence[int],
        images: Sequence[object],
    ) -> BatchFeature:
        """Replace each placeholder id with its sentinel block, at the true
        offset, and collect the per-image tensors."""
        cfg = self.info.get_hf_config()
        image_token_id = self.info.get_image_token_id()
        vocab_size = cfg.vocab_size

        num_placeholders = sum(1 for tok in prompt_tokens if tok == image_token_id)
        if num_placeholders != len(images):
            raise ValueError(
                f"Found {num_placeholders} image placeholder tokens "
                f"({IMAGE_PLACEHOLDER!r}) in the prompt but got {len(images)} images"
            )

        tokens: list[int] = []
        all_patches: list[torch.Tensor] = []
        all_types: list[torch.Tensor] = []
        all_perm: list[torch.Tensor] = []
        n_vit: list[list[int]] = []
        n_llm: list[list[int]] = []
        blocks: list[torch.Tensor] = []
        image_iter = iter(images)
        for tok in prompt_tokens:
            if tok != image_token_id:
                tokens.append(tok)
                continue
            raw_image = next(image_iter)
            if raw_image is None:
                # A uuid-only item (multi_modal_uuids without data) expects
                # the processor cache to fill it in; this processor always
                # takes the uncached whole-prompt path (module docstring).
                raise ValueError(
                    "DeepSeek-V4 vision does not support the multimodal "
                    "processor cache: an image block depends on its offset in "
                    "the prompt. Send the image data with every request "
                    "instead of a bare multi_modal_uuids reference."
                )
            image = as_pil_image(raw_image)
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(image, cfg)
            types, perm = build_image_block(n_llm_h, n_llm_w, len(tokens))
            block = types + vocab_size
            tokens.extend(block.tolist())
            all_patches.append(patches)
            all_types.append(types)
            all_perm.append(perm)
            n_vit.append([n_vit_h, n_vit_w])
            n_llm.append([n_llm_h, n_llm_w])
            blocks.append(block)

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor([tokens], dtype=torch.int64)
        }
        if all_patches:
            out.update(
                patches=torch.cat(all_patches, dim=0),
                patch_sizes=torch.tensor(
                    [p.shape[0] for p in all_patches], dtype=torch.int64
                ),
                types=torch.cat(all_types, dim=0),
                type_sizes=torch.tensor(
                    [t.shape[0] for t in all_types], dtype=torch.int64
                ),
                perm=torch.cat(all_perm, dim=0),
                perm_sizes=torch.tensor(
                    [p.shape[0] for p in all_perm], dtype=torch.int64
                ),
                n_vit=torch.tensor(n_vit, dtype=torch.int64),
                n_llm=torch.tensor(n_llm, dtype=torch.int64),
                token_blocks=torch.cat(blocks, dim=0),
            )
        return BatchFeature(out)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text prompts only reach here for the profiling dummies (the renderer
        # tokenizes serving prompts first). BOS, when wanted, is part of the
        # chat-template text, so never let the tokenizer add specials.
        tokenizer = self.info.get_tokenizer()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        return self._expand_image_tokens(prompt_tokens, _image_list(mm_data))

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        if isinstance(prompt, str):
            return super()._apply_hf_processor_main(
                prompt=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                enable_hf_prompt_update=enable_hf_prompt_update,
            )

        # Token prompt: the offsets are the real ones, expand in place instead
        # of the base's tokens-only + dummy-text detour (which would compute
        # the leading pads at the dummy offsets).
        processor_data, passthrough_data = self._get_hf_mm_data(mm_items)
        processed_data = self._expand_image_tokens(prompt, _image_list(processor_data))
        processed_data.update(passthrough_data)

        (prompt_ids,) = processed_data.pop("input_ids").tolist()

        return prompt_ids, processed_data, True

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        empty = torch.empty(0, dtype=torch.int64)
        patch_sizes = hf_inputs.get("patch_sizes", empty)
        type_sizes = hf_inputs.get("type_sizes", empty)
        perm_sizes = hf_inputs.get("perm_sizes", empty)
        return dict(
            patches=MultiModalFieldConfig.flat_from_sizes("image", patch_sizes),
            patch_sizes=MultiModalFieldConfig.batched("image"),
            types=MultiModalFieldConfig.flat_from_sizes("image", type_sizes),
            type_sizes=MultiModalFieldConfig.batched("image"),
            perm=MultiModalFieldConfig.flat_from_sizes("image", perm_sizes),
            perm_sizes=MultiModalFieldConfig.batched("image"),
            n_vit=MultiModalFieldConfig.batched("image"),
            n_llm=MultiModalFieldConfig.batched("image"),
            token_blocks=MultiModalFieldConfig.flat_from_sizes("image", type_sizes),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_token_id = self.info.get_image_token_id()

        def get_replacement(item_idx: int) -> list[int]:
            item = out_mm_kwargs["image"][item_idx]
            return item["token_blocks"].data.tolist()

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]
