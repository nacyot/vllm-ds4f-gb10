# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vision tower and aligner for DeepSeek-V4-Flash-Vision-Exp.

Faithful port of the reference implementation shipped with the checkpoint
(``inference/vision.py``). Deliberately built from plain ``torch.nn`` modules
and ``F.scaled_dot_product_attention`` with NO vllm imports: the tower and
aligner are BF16 in the checkpoint (259 + 4 tensors, no scale tensors) and
``DeepseekV4FP8Config`` only attaches quant methods to ``LinearBase`` /
``RoutedExperts``, so plain modules can never be quantized by accident. Keeping
the module vllm-free also lets it be unit-tested on CPU by importing the file
by path (the ``vllm`` package needs the compiled ``_C`` extension).

The tower runs one image per call with full bidirectional attention and 2D
RoPE; there is no CLS token, no learned position embedding and no windowing.
Parameter names equal the checkpoint keys under ``vision.`` / ``aligner.``:
``patch_embed.proj``, ``blocks.N.{norm1,attn.wqkv,attn.wo,norm2,mlp.w1,mlp.w2}``,
``norm``; ``w1``, ``w2``.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(8)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE tables (built on CPU, fp32): the first half of each head's rotary
    block is driven by the patch row, the second half by the column."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    # eps is 1e-6 here, NOT the LM's rms_norm_eps (1e-20): the reference Block
    # constructs RMSNorm(dim) with no eps argument, taking this default.
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # fp32 like the reference; the checkpoint tensor is BF16 and is
        # upcast on copy by both load_state_dict and vLLM's default loader.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Linear(3 * patch_size**2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-major flatten: index = c*patch^2 + py*patch + px.
        return self.proj(x.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, inter_dim: int):
        super().__init__()
        self.norm1 = VisionRMSNorm(dim)
        self.attn = VisionAttention(dim, n_heads)
        self.norm2 = VisionRMSNorm(dim)
        self.mlp = VisionMLP(dim, inter_dim)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE.

    ``config`` is any object exposing ``vision_dim``, ``vision_n_heads``,
    ``vision_patch_size``, ``vision_inter_dim``, ``vision_n_layers`` and
    ``vision_rope_theta`` (a PretrainedConfig or a SimpleNamespace).
    """

    def __init__(self, config):
        super().__init__()
        dim = config.vision_dim
        n_heads = config.vision_n_heads
        self.rope_dim = dim // n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = PatchEmbed(dim, config.vision_patch_size)
        self.blocks = nn.ModuleList(
            [
                VisionBlock(dim, n_heads, config.vision_inter_dim)
                for _ in range(config.vision_n_layers)
            ]
        )
        self.norm = VisionRMSNorm(dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """``patches``: [n_h * n_w, 3, p, p] (row-major over the patch grid)."""
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        # The tables are built on CPU and lru_cached; move to the activation
        # device (apply_rotary computes in fp32, so no dtype change needed).
        cos = cos.to(x.device)
        sin = sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """r x r space-to-depth (pixel-unshuffle) then a 2-layer GELU MLP to LM width.

    ``forward(x, n_h, n_w)`` takes the ViT output [n_h * n_w, vision_dim] with
    the ViT grid size and returns [ceil(n_h/r) * ceil(n_w/r), hidden_size]
    in row-major order over the downsampled grid.
    """

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        # Zero-pad right/bottom to a multiple of r, matching the reference.
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
