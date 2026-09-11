# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V4.1 vision wrapper hands the language-model weights to its child in one
lazy pass; sorting them (the previous approach) materialized every tensor,
which with a GPU-streaming loader is the whole checkpoint in device memory."""

import torch

from vllm.models.deepseek_v4_1.nvidia.vl_model import _route_language_weights


def test_route_language_weights_streams_and_buffers_the_rest():
    tensors = [torch.zeros(1) for _ in range(4)]
    stream = [
        ("vision.blocks.0.w", tensors[0]),
        ("language_model.model.layers.0.w", tensors[1]),
        ("aligner.w", tensors[2]),
        ("language_model.lm_head.weight", tensors[3]),
    ]
    rest: list[tuple[str, torch.Tensor]] = []
    routed = _route_language_weights(iter(stream), "language_model.", rest)

    first = next(routed)
    assert first == ("model.layers.0.w", tensors[1])
    # Lazy: only the tensors seen so far were buffered.
    assert rest == [("vision.blocks.0.w", tensors[0])]

    assert list(routed) == [("lm_head.weight", tensors[3])]
    assert rest == [("vision.blocks.0.w", tensors[0]), ("aligner.w", tensors[2])]
