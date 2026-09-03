# SPDX-License-Identifier: Apache-2.0
"""Batched (per-chunk-of-experts) versions of b12x's W4A16 repack helpers.

b12x.moe.fused.w4a16.prepare repacks MoE expert weights into the kernel layout
one expert at a time: per expert and per matrix it issues a transposed copy, a
permute copy and 8 x (gather, shift, mask, shift, or). For DeepSeek V4 Flash
that is ~21k expert-matrices and close to a million tiny kernel launches, which
is launch-bound on GB10: 41-48 s of the boot on every rank at TP=2. The index
arithmetic is identical when the same ops run with a leading expert dimension,
so this module re-implements _repack_weight and _permute_nvfp4_scales over
chunks of experts and installs them into the b12x module. Results are
bit-identical; DSPARK_B12X_REPACK_VERIFY=1 proves it at runtime by also running
the original per-expert implementation and comparing.

Env:
  DSPARK_B12X_FAST_REPACK   1 (default) install batched versions, 0 leave b12x alone
  DSPARK_B12X_REPACK_CHUNK  experts per chunk (default 64)
  DSPARK_B12X_REPACK_MODE   fused (Triton kernel, default) | batched (torch ops only)
  DSPARK_B12X_REPACK_VERIFY 1 to compare against the original implementation
"""

from __future__ import annotations

import math
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_installed = False


def _chunk_size() -> int:
    return max(1, int(os.environ.get("DSPARK_B12X_REPACK_CHUNK", "64")))


def _repack_weight_batched(
    weight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    reuse_input_storage: bool = False,
    _prep=None,
) -> torch.Tensor:
    P = _prep
    num_experts = int(weight.shape[0])
    if tuple(weight.shape[1:]) != (size_n, size_k // 2):
        raise ValueError(
            f"expected packed weight shape {(num_experts, size_n, size_k // 2)}, "
            f"got {tuple(weight.shape)}"
        )
    if size_k % P._PACKED_TILE_SIZE != 0 or size_n % P._PACKED_TILE_N_SIZE != 0:
        raise ValueError(
            f"W4A16 repack requires K,N multiples of 16,64; got {size_k},{size_n}"
        )
    k_tiles = size_k // P._PACKED_TILE_SIZE
    n_tiles = size_n // P._PACKED_TILE_N_SIZE
    tile_n = P._PACKED_TILE_N_SIZE
    pack = P._PACK_FACTOR_4BIT
    packed_shape = (num_experts, k_tiles, n_tiles * 128)
    if reuse_input_storage:
        if not weight.is_contiguous():
            raise ValueError("reuse_input_storage requires contiguous packed weights")
        packed = weight.view(torch.int32).reshape(packed_shape)
    else:
        packed = torch.empty(packed_shape, device=weight.device, dtype=torch.int32)

    device = weight.device
    # Same index tables as _repack_4bit_no_perm.
    out_pos = torch.arange(128, device=device, dtype=torch.long)
    th_id = out_pos // 4
    warp_id = out_pos % 4
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    offsets = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long)
    pack_idx = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device, dtype=torch.long)
    elem = tc_row[:, None] + offsets[None, :]
    row = elem // pack
    pos = elem % pack
    col1 = (warp_id * 16 + tc_col)[:, None].expand(-1, 4)
    col2 = col1 + 8
    source_index = torch.cat(
        [row * tile_n + col1, row * tile_n + col2], dim=1
    )[:, pack_idx]
    source_shift = torch.cat([pos, pos], dim=1)[:, pack_idx] * 4

    chunk = _chunk_size()
    w_all = weight.view(torch.int32)  # (E, size_n, size_k // 8)
    for e0 in range(0, num_experts, chunk):
        e1 = min(num_experts, e0 + chunk)
        n_e = e1 - e0
        w = w_all[e0:e1]
        # Transposed per-expert copy (the original's qweight_scratch.copy_(expert.T)),
        # taken fully before any write so reuse_input_storage stays safe.
        qs = torch.empty((n_e, size_k // pack, size_n), device=device, dtype=torch.int32)
        if row_rotation is not None:
            rotated_rows = int(size_n) - int(row_rotation)
            qs[:, :, :rotated_rows].copy_(w[:, row_rotation:, :].transpose(1, 2))
            qs[:, :, rotated_rows:].copy_(w[:, :row_rotation, :].transpose(1, 2))
        else:
            qs.copy_(w.transpose(1, 2))
        tiles = qs.view(n_e, k_tiles, 2, n_tiles, tile_n)
        flat = tiles.permute(0, 1, 3, 2, 4).reshape(n_e, k_tiles, n_tiles, 2 * tile_n)
        result = packed[e0:e1].view(n_e, k_tiles, n_tiles, 128)
        result.zero_()
        for slot in range(8):
            gather_index = source_index[:, slot].view(1, 1, 1, 128).expand(
                n_e, k_tiles, n_tiles, 128
            )
            shift = source_shift[:, slot].view(1, 1, 1, 128)
            gathered = flat.gather(3, gather_index)
            nibble = (gathered >> shift) & 0xF
            result |= nibble << (slot * 4)
        del qs, flat
    return packed



try:
    import triton
    import triton.language as tl

    @triton.jit
    def _repack_tile_kernel(
        qs_ptr,
        out_ptr,
        idx_ptr,
        shift_ptr,
        k_tiles,
        n_tiles,
        stride_qe,
        stride_qr,
        stride_oe,
        BLOCK: tl.constexpr,
    ):
        # One program = one (expert, k_tile, n_tile) output tile of 128 int32.
        # Same index arithmetic as b12x _repack_4bit_no_perm, but the 8 slot
        # passes happen in registers: each 4-bit source nibble is read once.
        pid = tl.program_id(0)
        kt = pid % k_tiles
        tmp = pid // k_tiles
        nt = tmp % n_tiles
        e = tmp // n_tiles
        o = tl.arange(0, BLOCK)
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        for s in tl.static_range(8):
            j = tl.load(idx_ptr + o * 8 + s)
            sh = tl.load(shift_ptr + o * 8 + s)
            r = kt * 2 + j // 64
            c = nt * 64 + j % 64
            v = tl.load(qs_ptr + e * stride_qe + r * stride_qr + c)
            acc |= ((v >> sh) & 0xF) << (4 * s)
        tl.store(out_ptr + e * stride_oe + kt * (n_tiles * 128) + nt * 128 + o, acc)

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing or unusable
    _HAS_TRITON = False


def _repack_weight_fused(
    weight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    reuse_input_storage: bool = False,
    _prep=None,
) -> torch.Tensor:
    """_repack_weight with the 8-slot gather loop fused into one Triton kernel.

    Per chunk of experts: one transposed copy (as the original does per expert)
    then one kernel launch that reads every source int32 once and writes each
    output once, instead of 8 full passes with gathers and shifts.
    """
    P = _prep
    num_experts = int(weight.shape[0])
    if tuple(weight.shape[1:]) != (size_n, size_k // 2):
        raise ValueError(
            f"expected packed weight shape {(num_experts, size_n, size_k // 2)}, "
            f"got {tuple(weight.shape)}"
        )
    if size_k % P._PACKED_TILE_SIZE != 0 or size_n % P._PACKED_TILE_N_SIZE != 0:
        raise ValueError(
            f"W4A16 repack requires K,N multiples of 16,64; got {size_k},{size_n}"
        )
    k_tiles = size_k // P._PACKED_TILE_SIZE
    n_tiles = size_n // P._PACKED_TILE_N_SIZE
    tile_n = P._PACKED_TILE_N_SIZE
    pack = P._PACK_FACTOR_4BIT
    packed_shape = (num_experts, k_tiles, n_tiles * 128)
    if reuse_input_storage:
        if not weight.is_contiguous():
            raise ValueError("reuse_input_storage requires contiguous packed weights")
        packed = weight.view(torch.int32).reshape(packed_shape)
    else:
        packed = torch.empty(packed_shape, device=weight.device, dtype=torch.int32)

    device = weight.device
    out_pos = torch.arange(128, device=device, dtype=torch.long)
    th_id = out_pos // 4
    warp_id = out_pos % 4
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    offsets = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long)
    pack_idx = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device, dtype=torch.long)
    elem = tc_row[:, None] + offsets[None, :]
    row = elem // pack
    pos = elem % pack
    col1 = (warp_id * 16 + tc_col)[:, None].expand(-1, 4)
    col2 = col1 + 8
    source_index = torch.cat([row * tile_n + col1, row * tile_n + col2], dim=1)[:, pack_idx]
    source_shift = torch.cat([pos, pos], dim=1)[:, pack_idx] * 4
    idx_t = source_index.to(torch.int32).contiguous()  # (128, 8)
    shift_t = source_shift.to(torch.int32).contiguous()

    chunk = _chunk_size()
    w_all = weight.view(torch.int32)  # (E, size_n, size_k // 8)
    for e0 in range(0, num_experts, chunk):
        e1 = min(num_experts, e0 + chunk)
        n_e = e1 - e0
        w = w_all[e0:e1]
        qs = torch.empty((n_e, size_k // pack, size_n), device=device, dtype=torch.int32)
        if row_rotation is not None:
            rotated_rows = int(size_n) - int(row_rotation)
            qs[:, :, :rotated_rows].copy_(w[:, row_rotation:, :].transpose(1, 2))
            qs[:, :, rotated_rows:].copy_(w[:, :row_rotation, :].transpose(1, 2))
        else:
            qs.copy_(w.transpose(1, 2))
        out = packed[e0:e1]  # (n_e, k_tiles, n_tiles*128), contiguous
        grid = (n_e * k_tiles * n_tiles,)
        _repack_tile_kernel[grid](
            qs, out, idx_t, shift_t,
            k_tiles, n_tiles,
            qs.stride(0), qs.stride(1), out.stride(0),
            BLOCK=128,
        )
        del qs
    return packed


def _permute_nvfp4_scales_batched(
    scales: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    a_dtype: torch.dtype,
    row_rotation: int | None = None,
    _prep=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    P = _prep
    num_experts = int(scales.shape[0])
    # _nvfp4_compute_scale_factor without the per-expert loop: the max over
    # experts of the per-expert max of nonzero entries is the global max of
    # nonzero entries.
    if a_dtype == torch.float16:
        combined_scale_factor = 1.0
    else:
        max_scalar = 0.0
        if num_experts > 0:
            ws_float = scales.float() * (2**7)
            nonzero = ws_float[ws_float > 0]
            if nonzero.numel() > 0:
                max_scalar = float(nonzero.max().item())
        if max_scalar > 0.0 and max_scalar < 448 * (2**7):
            combined_scale_factor = float(
                2 ** math.floor(math.log2((448 * (2**7)) / max_scalar))
            )
        else:
            combined_scale_factor = 1.0

    scale_perm, scale_perm_single = P._scale_perms()
    group_size = 16
    perm = scale_perm if (group_size < size_k and group_size != -1) else scale_perm_single
    perm_t = torch.tensor(perm, device=scales.device, dtype=torch.long)
    reorder4 = torch.tensor([0, 2, 1, 3], device=scales.device, dtype=torch.long)

    packed_scales: torch.Tensor | None = None
    chunk = _chunk_size()
    for e0 in range(0, num_experts, chunk):
        e1 = min(num_experts, e0 + chunk)
        n_e = e1 - e0
        src = scales[e0:e1].to(a_dtype)  # (n_e, size_n, cols)
        if row_rotation is not None:
            src = torch.cat([src[:, row_rotation:], src[:, :row_rotation]], dim=1)
        src = src.transpose(1, 2).contiguous()  # per expert: expert_source.T
        # _permute_packed_scales, batched
        s = src.reshape(n_e, -1, len(perm))[:, :, perm_t].reshape(n_e, -1, size_n)
        # _process_nvfp4_packed_scales, batched (row-local ops)
        s = s.to(torch.float16)
        rows = s.shape[1]
        s = s.reshape(n_e, rows, -1, 4)[..., reorder4].reshape(n_e, rows, -1)
        if combined_scale_factor > 1.0:
            s = (s.float() * combined_scale_factor).to(torch.float16)
        s = s * (2**7)
        s[s < 2] = 0
        s = s.view(torch.int16) << 1
        s = s.view(torch.float8_e4m3fn)
        expert_packed = s[..., 1::2].contiguous()
        if packed_scales is None:
            packed_scales = torch.empty(
                (num_experts, *expert_packed.shape[1:]),
                dtype=expert_packed.dtype,
                device=expert_packed.device,
            )
        packed_scales[e0:e1].copy_(expert_packed)
    if packed_scales is None:
        packed_scales = torch.empty(
            (0, size_k // P._PACKED_TILE_SIZE, size_n // 2),
            dtype=torch.float8_e4m3fn,
            device=scales.device,
        )
    packed_global = P._process_nvfp4_packed_global_scale(
        global_scales, a_dtype=a_dtype
    ).to(torch.float32)
    packed_global = packed_global / combined_scale_factor
    return packed_scales, packed_global.contiguous()


def install() -> bool:
    """Install the batched helpers into b12x (idempotent). Returns True if active."""
    global _installed
    if _installed:
        return True
    if os.environ.get("DSPARK_B12X_FAST_REPACK", "1") != "1":
        return False
    try:
        from b12x.moe.fused.w4a16 import prepare as P
    except ImportError:
        return False
    for name in ("_repack_weight", "_permute_nvfp4_scales", "_PACKED_TILE_SIZE",
                 "_PACKED_TILE_N_SIZE", "_PACK_FACTOR_4BIT", "_scale_perms",
                 "_process_nvfp4_packed_global_scale"):
        if not hasattr(P, name):
            logger.warning("b12x fast repack: %s missing in b12x.prepare; not installed", name)
            return False
    orig_repack = P._repack_weight
    orig_scales = P._permute_nvfp4_scales
    verify = os.environ.get("DSPARK_B12X_REPACK_VERIFY") == "1"
    stats = {"repack_s": 0.0, "scales_s": 0.0, "calls": 0, "verified": 0}

    mode = os.environ.get("DSPARK_B12X_REPACK_MODE", "fused" if _HAS_TRITON else "batched")
    impl = _repack_weight_fused if (mode == "fused" and _HAS_TRITON) else _repack_weight_batched

    def repack(weight, **kw):
        ref_input = weight.clone() if verify else None
        t0 = time.perf_counter()
        out = impl(weight, _prep=P, **kw)
        stats["repack_s"] += time.perf_counter() - t0
        stats["calls"] += 1
        if verify:
            ref = orig_repack(ref_input, **{**kw, "reuse_input_storage": False})
            if not torch.equal(ref, out):
                raise RuntimeError("b12x fast repack: _repack_weight mismatch vs original")
            stats["verified"] += 1
            del ref, ref_input
        return out

    def permute_scales(scales, global_scales, **kw):
        t0 = time.perf_counter()
        out = _permute_nvfp4_scales_batched(scales, global_scales, _prep=P, **kw)
        stats["scales_s"] += time.perf_counter() - t0
        if verify:
            ref = orig_scales(scales, global_scales, **kw)
            if not (torch.equal(ref[0].view(torch.uint8), out[0].view(torch.uint8))
                    and torch.equal(ref[1], out[1])):
                raise RuntimeError("b12x fast repack: _permute_nvfp4_scales mismatch vs original")
        return out

    P._repack_weight = repack
    P._permute_nvfp4_scales = permute_scales
    install.stats = stats  # type: ignore[attr-defined]
    _installed = True
    logger.info_once(
        "b12x fast repack installed (mode=%s, chunk=%d, verify=%s)",
        impl.__name__, _chunk_size(), verify,
    )
    return True


def log_stats() -> None:
    stats = getattr(install, "stats", None)
    if stats:
        logger.info(
            "b12x fast repack: %d calls, repack %.2fs, scales %.2fs, verified %d",
            stats["calls"], stats["repack_s"], stats["scales_s"], stats["verified"],
        )
