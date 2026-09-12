# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.common.engram import next_chunk_windows
from vllm.models.deepseek_v4_1.common.mm_preprocess import image_sentinel_mask
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.

    With mmap'd engram tables the step's hash ids are also computed here and
    their pages prefaulted before the forward runs, so the forward has no
    host sync and can be captured in a CUDA graph. With
    ``mmap_prefetch_next_chunk`` the rows of the chunk each prefill continues
    with are hashed here too and populated in the background while this
    step runs.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )
        self.engram_hash = None
        self.engram_prefault_targets: list[tuple[Any, int]] = []
        inner = getattr(model, "model", None)
        if inner is None and hasattr(model, "language_model"):
            inner = getattr(model.language_model, "model", None)
        engram_hash = getattr(inner, "engram_hash", None)
        if engram_hash is not None:
            for layer in inner.layers:
                engram = getattr(layer, "engram", None)
                if engram is not None and engram.embed_tokens.mmap_table is not None:
                    engram.embed_tokens.runner_prefaults = True
                    self.engram_prefault_targets.append(
                        (engram.embed_tokens, engram.layer_hash_index)
                    )
            if self.engram_prefault_targets:
                self.engram_hash = engram_hash
        engram_config = vllm_config.engram_config
        self.engram_prefetch = bool(
            self.engram_prefault_targets
            and engram_config is not None
            and engram_config.mmap_prefetch_next_chunk
        )
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        if self.engram_hash is not None and self.engram_hash.ensure_cache():
            self._prefault_engram_rows(input_batch, window)
            if self.engram_prefetch and input_batch.has_prefill:
                self._prefetch_next_chunk(input_batch, req_states)
        return model_inputs

    def _prefault_engram_rows(
        self, input_batch: InputBatch, window: torch.Tensor
    ) -> None:
        """Hash this step's n-grams and map their table pages before the forward."""
        num_tokens = input_batch.num_tokens
        input_ids = input_batch.input_ids[:num_tokens]
        hashes = self.engram_hash(
            input_ids,
            input_batch.positions[:num_tokens],
            input_batch.query_start_loc[: input_batch.num_reqs + 1],
            image_sentinel_mask(input_ids),
            window,
            image_sentinel_mask(window),
            None,
            None,
        ).cpu()
        for embed_tokens, layer_hash_index in self.engram_prefault_targets:
            embed_tokens.prefault(hashes[:, layer_hash_index])

    def _prefetch_next_chunk(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> None:
        """Hash the next chunk of every prefilling request and populate its
        table pages in the background while this step runs."""
        num_reqs = input_batch.num_reqs
        windows = next_chunk_windows(
            input_batch.num_computed_tokens_np[:num_reqs],
            input_batch.num_scheduled_tokens[:num_reqs],
            input_batch.prefill_len_np[:num_reqs],
            self.max_num_batched_tokens,
        )
        if not windows:
            return
        all_token_ids = req_states.all_token_ids.gpu
        assert self.lookback_token_ids is not None
        depth = self.lookback_token_ids.shape[1]
        device = all_token_ids.device
        ids, positions, lookback = [], [], []
        lens = [0]
        for i, start, end in windows:
            req_idx = int(input_batch.idx_mapping_np[i])
            ids.append(all_token_ids[req_idx, start:end])
            positions.append(
                torch.arange(
                    start, end, dtype=input_batch.positions.dtype, device=device
                )
            )
            lens.append(end - start)
            # Newest token first, -1 past the sequence start, as the runner's
            # window (`_gather_lookback_kernel`).
            row = torch.full((depth,), -1, dtype=torch.int32, device=device)
            history = all_token_ids[req_idx, max(0, start - depth) : start]
            row[: history.numel()] = history.flip(0)
            lookback.append(row)
        input_ids = torch.cat(ids)
        window = torch.stack(lookback)
        query_start_loc = torch.tensor(
            lens, dtype=input_batch.query_start_loc.dtype, device=device
        ).cumsum_(0)
        assert self.engram_hash is not None
        hashes = self.engram_hash(
            input_ids,
            torch.cat(positions),
            query_start_loc,
            image_sentinel_mask(input_ids),
            window,
            image_sentinel_mask(window),
            None,
            None,
        ).cpu()
        for embed_tokens, layer_hash_index in self.engram_prefault_targets:
            embed_tokens.prefetch(hashes[:, layer_hash_index])

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs
