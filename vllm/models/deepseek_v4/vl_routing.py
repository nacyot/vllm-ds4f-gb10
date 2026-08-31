# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-torch helpers for DeepSeek-V4 vision (``bias_vl``) MoE routing.

No ``vllm`` imports on purpose: the router, the language model and the DSpark
draft all call into this module, and the CPU-only tests import it by path.

Reference semantics (DeepSeek ``Gate.forward`` with ``bias_vl`` present)::

    image_mask = input_ids >= vocab_size
    if hash layer:
        indices = tid2eid[where(image_mask, 0, input_ids)]
        vl_indices = topk(scores + bias_vl)
        indices = where(image_mask[:, None], vl_indices, indices)
    else:
        indices = topk(scores + where(image_mask[:, None], bias_vl, bias))
    weights = scores.gather(indices); renormalize; scale

Image positions carry out-of-vocabulary sentinel ids (``vocab_size + t``), so
they must be masked to 0 before any ``tid2eid`` / embedding gather. Everything
here is branch-free (``torch.where`` only, no ``.any()`` / ``nonzero``) because
it runs inside the torch.compile region and the CUDA-graph capture.
"""

import torch


def sanitize_vl_input_ids(
    input_ids: torch.Tensor, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(image_mask, safe_ids)`` for a flat ``[M]`` id tensor.

    ``image_mask`` is ``input_ids >= vocab_size`` (bool ``[M]``); ``safe_ids``
    has those positions replaced by 0 and keeps the input dtype/device.
    """
    image_mask = input_ids >= vocab_size
    safe_ids = torch.where(image_mask, torch.zeros_like(input_ids), input_ids)
    return image_mask, safe_ids


def merge_vl_routing(
    text_weights: torch.Tensor,
    text_ids: torch.Tensor,
    vl_weights: torch.Tensor,
    vl_ids: torch.Tensor,
    image_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the image-pass routing on image rows, the text pass elsewhere.

    All routing tensors are ``[M, topk]``; ``image_mask`` is bool ``[M]``. The
    result keeps the text pass dtypes (the image pass may come from a kernel
    that emits a different index dtype).
    """
    mask = image_mask.unsqueeze(-1)
    weights = torch.where(mask, vl_weights.to(text_weights.dtype), text_weights)
    ids = torch.where(mask, vl_ids.to(text_ids.dtype), text_ids)
    return weights, ids
