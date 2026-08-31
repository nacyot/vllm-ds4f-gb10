# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Image preprocessing for DeepSeek-V4-Flash-Vision-Exp.

Ported 1:1 from DeepSeek's reference ``inference/image_processor.py``. Every
formula is load-bearing: the aligner emits one embedding per ``downsample x
downsample`` patch block and those embeddings are scattered into an "N-layout"
token grid, so any drift in the resize solver, the token count or the
permutation puts image features in the wrong positions.

This module imports nothing from vLLM (torch / numpy / PIL only) so it can be
unit-tested against the reference without the compiled extension.

Token block layout (``build_image_block``), in emission order::

    [IMAGE_PAD] * (3 - start_pos % 4)        # aligns the grid to the C4 block
    IMAGE_START
    grid rows in pairs, column-major inside each pair (the "N" layout):
        IMAGE * n_llm_w + IMAGE_NEW_LINE per row, odd row counts get one
        IMAGE_PAD row appended
    [IMAGE_PAD] * pad_last                    # 0 or 2, makes the grid even
    IMAGE_END

Sentinel ids in the prompt are ``vocab_size + type``.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens the aligner grid occupies (N-layout, incl. row/align
    padding and IMAGE_START / IMAGE_END, excl. the leading offset pad)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token
):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def solve_image_grid(width: int, height: int, cfg):
    """Resize solver only, no pixel work.

    Returns ``(n_vit_h, n_vit_w, n_llm_h, n_llm_w, best_height, best_width)``
    for an image of the given size. Mirrors the size logic of the reference
    ``load_image`` exactly (aspect clamp, min-pixel upscale, patch rounding,
    token-budget shrink).
    """
    p = cfg.vision_patch_size
    max_wh_ratio = cfg.vision_max_wh_ratio
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < cfg.vision_min_pixels:
        ratio = (cfg.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        cfg.vision_downsample_ratio,
        cfg.vision_max_n_token,
    )
    return best_height // p, best_width // p, n_llm_h, n_llm_w, best_height, best_width


def preprocess_image(image: Image.Image, cfg):
    """PIL image -> ``(patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w)``.

    ``patches`` is ``[n_vit_h * n_vit_w, 3, p, p]`` bf16, normalised to
    ``[-1, 1]``, in row-major patch order. ``cfg`` is any object exposing the
    ``vision_*`` attributes of the checkpoint config.
    """
    p = cfg.vision_patch_size
    image = image.convert("RGB")
    width, height = image.size
    n_vit_h, n_vit_w, n_llm_h, n_llm_w, best_height, best_width = solve_image_grid(
        width, height, cfg
    )
    if (
        cfg.vision_max_wh_ratio is not None
        and image.width >= cfg.vision_max_wh_ratio * image.height
    ):
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """Token types in emission order, plus the aligner-row order for IMAGE slots.

    ``types`` is the full block (leading pads, IMAGE_START, grid, IMAGE_END);
    ``perm[i]`` is the row-major aligner index that fills the ``i``-th IMAGE
    slot of ``types``.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2)
    ).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def num_image_tokens(n_llm_h: int, n_llm_w: int, start_pos: int) -> int:
    """``len(types)`` of ``build_image_block`` without materialising it."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    rows = n_llm_h + n_llm_h % 2
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    return compress_pad + 1 + rows * row_len + pad_last + 1


@lru_cache(maxsize=8)
def _max_tokens_image_size(
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio,
) -> tuple[int, int]:
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        vision_patch_size=patch_size,
        vision_downsample_ratio=downsample_ratio,
        vision_max_n_token=max_n_token,
        vision_min_pixels=min_pixels,
        vision_max_wh_ratio=max_wh_ratio,
    )
    unit = patch_size * downsample_ratio
    budget = max_n_token - (COMPRESS_PAD_TO - 1)
    # (tokens, patches, width, height); grids whose n_vit is exactly
    # downsample * n_llm maximise the ViT work for a given token count.
    candidates: list[tuple[int, int, int, int]] = []
    for n_llm_h in range(1, budget // 2 + 1):
        for n_llm_w in range(1, budget):
            width, height = n_llm_w * unit, n_llm_h * unit
            if max_wh_ratio is not None and width > height * max_wh_ratio:
                break
            _, _, tokens = grid_tokens(height, width, patch_size, downsample_ratio)
            if tokens > budget:
                break
            candidates.append((tokens, n_llm_h * n_llm_w, width, height))
    candidates.sort(reverse=True)
    for tokens, _, width, height in candidates:
        _, _, n_llm_h, n_llm_w, best_height, best_width = solve_image_grid(
            width, height, cfg
        )
        got = grid_tokens(best_height, best_width, patch_size, downsample_ratio)[2]
        if got == tokens and (best_height, best_width) == (height, width):
            return width, height
    raise RuntimeError("no image size reaches the vision token budget")


def max_tokens_image_size(cfg) -> tuple[int, int]:
    """``(width, height)`` of an image whose block reaches the largest token
    count the solver can emit (verified by round-tripping through the solver),
    preferring the most ViT patches among ties. Used for profiling dummies."""
    return _max_tokens_image_size(
        cfg.vision_patch_size,
        cfg.vision_downsample_ratio,
        cfg.vision_max_n_token,
        cfg.vision_min_pixels,
        cfg.vision_max_wh_ratio,
    )


def as_pil_image(image) -> Image.Image:
    """Accept the image item types vLLM's parser lets through (PIL, uint8
    HWC / CHW numpy or torch) and return a PIL image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        if (
            image.ndim == 3
            and image.shape[0] in (1, 3, 4)
            and image.shape[-1]
            not in (
                1,
                3,
                4,
            )
        ):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            raise TypeError(
                f"Expected a uint8 image array, got dtype {image.dtype}; "
                "pass a PIL image instead"
            )
        return Image.fromarray(np.ascontiguousarray(image))
    raise TypeError(f"Unsupported image type: {type(image).__name__}")
