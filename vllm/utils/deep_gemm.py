# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

import contextlib
import functools
import importlib
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, NoReturn

import torch

import vllm.envs as envs
from vllm.logger import logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv

_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {
    "qwen3_5_text",
    "qwen3_5_moe_text",
}


def should_auto_disable_deep_gemm(model_type: str | None) -> bool:
    """Check if DeepGemm should be auto-disabled for this model on Blackwell.

    Returns True if the model is known to have accuracy degradation with
    DeepGemm's E8M0 scale format on Blackwell GPUs (SM100+).
    """
    if model_type is None:
        return False
    if not (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    ):
        return False
    return model_type in _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES


class DeepGemmQuantScaleFMT(Enum):
    # Float32 scales in Float32 tensor
    FLOAT32 = 0
    # Compute float32 scales and ceil the scales to UE8M0.
    # Keep the scales in Float32 tensor.
    FLOAT32_CEIL_UE8M0 = 1
    # Compute float32 scales and ceil the scales to UE8M0.
    # Pack the scales into a int32 tensor where each int32
    # element contains 4 scale values.
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        """Initialize the oracle decision and store it in the class cache"""
        cached = getattr(cls, "_oracle_cache", None)
        if cached is not None:
            return

        use_e8m0 = (
            envs.VLLM_USE_DEEP_GEMM_E8M0
            and is_deep_gemm_supported()
            and (_fp8_gemm_nt_impl is not None)
        )
        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return

        cls._oracle_cache = (  # type: ignore
            cls.UE8M0
            if (
                current_platform.is_device_capability_family(100)
                or current_platform.is_device_capability_family(120)
            )
            else cls.FLOAT32_CEIL_UE8M0
        )

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        """Return the oracle decision, initializing it on first use.

        The cache is normally populated by ``_lazy_init()`` (e.g. during
        engine startup), but standalone consumers such as ``QuantFP8`` with an
        explicit ``use_ue8m0=True`` can reach this before any DeepGEMM kernel
        wrapper has run. Resolve the DeepGEMM symbols and initialize the
        decision here instead of asserting; without DeepGEMM this yields
        FLOAT32, matching ``is_deep_gemm_e8m0_used()``.
        """
        cached = getattr(cls, "_oracle_cache", None)
        if cached is None:
            _lazy_init()
            cls.init_oracle_cache()
            cached = cls._oracle_cache  # type: ignore[attr-defined]
        return cached


@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    is_supported_arch = current_platform.support_deep_gemm()
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return `True` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    if not is_deep_gemm_supported():
        logger.debug_once(
            "DeepGEMM E8M0 disabled: DeepGEMM not supported on this system."
        )
        return False

    _lazy_init()

    if _fp8_gemm_nt_impl is None:
        logger.info_once("DeepGEMM E8M0 disabled: _fp8_gemm_nt_impl not found")
        return False

    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on current platform.")
        return True

    logger.info_once("DeepGEMM E8M0 disabled on current configuration.")
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is unavailable in the current vLLM environment, "
        "or the available DeepGEMM package does not provide the required APIs "
        "for these kernels."
    )


_cublaslt_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_einsum_impl: Callable[..., Any] | None = None
_grouped_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None
_grouped_fp4_impl: Callable[..., Any] | None = None
_fp8_fp4_mqa_logits_impl: Callable[..., Any] | None = None
_fp8_fp4_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_tensor_impl: Callable[..., Any] | None = None
_get_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = None
_get_theoretical_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = (
    None
)
_transform_sf_into_required_layout_impl: Callable[..., Any] | None = None
_transform_weights_for_mega_moe_impl: Callable[..., Any] | None = None
_get_symm_buffer_for_mega_moe_impl: Callable[..., Any] | None = None
_fp8_fp4_mega_moe_impl: Callable[..., Any] | None = None
_pack_ue8m0_to_int_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_packed_ue8m0_tensor_impl: Callable[..., Any] | None = None
_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl: (
    Callable[..., Any] | None
) = None


@functools.cache
def _import_deep_gemm():
    """Import the deep_gemm module.

    Prefers an externally installed ``deep_gemm`` package (so users can
    pin a specific version), then falls back to the vendored copy bundled
    in the vLLM wheel.

    Returns ``None`` when neither source is usable.
    """
    # 1. Try the external (pip-installed) package first.
    try:
        module = importlib.import_module("deep_gemm")
        logger.debug_once("Imported deep_gemm module from site-packages")
        return module
    except ImportError:
        logger.info_once(
            "deep_gemm not found in site-packages, "
            "trying vendored vllm.third_party.deep_gemm"
        )

    # 2. Fall back to the vendored copy bundled in the vLLM wheel.
    try:
        module = importlib.import_module("vllm.third_party.deep_gemm")
        logger.debug_once("Imported deep_gemm module from vllm.third_party.deep_gemm")
        return module
    except ImportError:
        logger.info_once("Vendored deep_gemm not found either")
    except Exception as e:
        # The vendored module may raise RuntimeError during _C.init()
        # if JIT include files are missing (e.g. incomplete wheel).
        logger.warning_once("Failed to import vendored deep_gemm: %s", e)

    return None


def _apply_pdl(mod, enable: bool = True) -> None:
    mod_name = getattr(mod, "__name__", str(mod))
    try:
        set_pdl_fn = getattr(mod, "set_pdl", None)
        if set_pdl_fn is None:
            return
        set_pdl_fn(enable)
        logger.info_once(
            "DeepGEMM PDL %s on %s.",
            "enabled" if enable else "disabled",
            mod_name,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning_once("Failed to set DeepGEMM PDL on %s: %s", mod_name, e)


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _cublaslt_gemm_nt_impl
    global _fp8_gemm_nt_impl, _fp8_einsum_impl
    global _grouped_impl, _grouped_masked_impl, _grouped_fp4_impl
    global _fp8_fp4_mqa_logits_impl, _fp8_fp4_paged_mqa_logits_impl
    global _get_paged_mqa_logits_metadata_impl
    global _tf32_hc_prenorm_gemm_impl
    global _get_mn_major_tma_aligned_tensor_impl
    global _get_mk_alignment_for_contiguous_layout_impl
    global _get_theoretical_mk_alignment_for_contiguous_layout_impl
    global _transform_sf_into_required_layout_impl
    global _transform_weights_for_mega_moe_impl
    global _get_symm_buffer_for_mega_moe_impl, _fp8_fp4_mega_moe_impl
    global _pack_ue8m0_to_int_impl
    global _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl
    global _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl
    # fast path
    if (
        _cublaslt_gemm_nt_impl is not None
        or _fp8_gemm_nt_impl is not None
        or _fp8_einsum_impl is not None
        or _grouped_impl is not None
        or _grouped_masked_impl is not None
        or _grouped_fp4_impl is not None
        or _fp8_fp4_mqa_logits_impl is not None
        or _fp8_fp4_paged_mqa_logits_impl is not None
        or _get_paged_mqa_logits_metadata_impl is not None
        or _tf32_hc_prenorm_gemm_impl is not None
        or _get_mk_alignment_for_contiguous_layout_impl is not None
        or _transform_sf_into_required_layout_impl is not None
        or _transform_weights_for_mega_moe_impl is not None
        or _get_symm_buffer_for_mega_moe_impl is not None
        or _fp8_fp4_mega_moe_impl is not None
        or _pack_ue8m0_to_int_impl is not None
        or _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl is not None
        or _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl is not None
    ):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = _import_deep_gemm()
    if _dg is None:
        return

    # Enable PDL for DeepGEMM on architectures that support it (SM90+).
    if current_platform.is_arch_support_pdl():
        _apply_pdl(_dg, True)
    _cublaslt_gemm_nt_impl = getattr(_dg, "cublaslt_gemm_nt", None)
    _fp8_gemm_nt_impl = getattr(_dg, "fp8_gemm_nt", None)
    _fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)
    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)
    _grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)
    _grouped_fp4_impl = getattr(_dg, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    # DeepGEMM exposes fp8_fp4_*_mqa_logits as the canonical symbols that
    # handle both the FP8 and FP4 Q/K paths via a tuple-typed `q`.
    _fp8_fp4_mqa_logits_impl = getattr(_dg, "fp8_fp4_mqa_logits", None)
    _fp8_fp4_paged_mqa_logits_impl = getattr(_dg, "fp8_fp4_paged_mqa_logits", None)
    _get_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_mqa_logits_metadata", None
    )
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    _get_mn_major_tma_aligned_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_tensor", None
    )
    _get_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_mk_alignment_for_contiguous_layout", None
    )
    _get_theoretical_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_theoretical_mk_alignment_for_contiguous_layout", None
    )
    _transform_sf_into_required_layout_impl = getattr(
        _dg, "transform_sf_into_required_layout", None
    )
    _transform_weights_for_mega_moe_impl = getattr(
        _dg, "transform_weights_for_mega_moe", None
    )
    _get_symm_buffer_for_mega_moe_impl = getattr(
        _dg, "get_symm_buffer_for_mega_moe", None
    )
    _fp8_fp4_mega_moe_impl = getattr(_dg, "fp8_fp4_mega_moe", None)
    _pack_ue8m0_to_int_impl = getattr(_dg, "pack_ue8m0_to_int", None)
    _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_packed_ue8m0_tensor", None
    )
    _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl = getattr(
        _dg, "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", None
    )
    DeepGemmQuantScaleFMT.init_oracle_cache()


def get_num_sms() -> int:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    return int(dg.get_num_sms())


def set_num_sms(num_sms: int) -> None:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    dg.set_num_sms(num_sms)


def get_mk_alignment_for_contiguous_layout() -> list[int]:
    _lazy_init()
    if _get_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()
    return [mk_align_size, mk_align_size]


def get_theoretical_mk_alignment_for_contiguous_layout(
    expected_m: int | None = None,
    num_groups: int | None = None,
) -> int:
    """Per-call optimal M alignment for grouped contiguous GEMMs.

    `expected_m` is the TOTAL routed tokens (sum across experts, typically
    M × num_topk). `num_groups` is the number of experts on this rank.
    The helper divides to recover per-expert em and picks an alignment based
    on data-driven thresholds (see deep_gemm runtime.hpp comments).

    Older callers that omit `num_groups` are interpreted as passing already
    per-expert em (legacy behaviour preserved for backward compat).
    """
    _lazy_init()
    if _get_theoretical_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    if num_groups is None:
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(expected_m)
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    try:
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(
            expected_m, num_groups
        )
    except TypeError:
        per_group_m = None if expected_m is None else cdiv(expected_m, num_groups)
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(per_group_m)


def set_mk_alignment_for_contiguous_layout(value: int) -> None:
    """Set DeepGEMM's BLOCK_M cap for grouped contiguous GEMMs.

    The DG heuristic constrains BLOCK_M ≤ this value when picking a kernel
    layout. Use this in concert with `compute_aligned_M_and_alignment`'s
    per-call alignment so the workspace's per-expert padding matches the
    kernel's BLOCK_M; a mismatch leads to the scheduler reading the wrong
    expert_id from `m_indices` at `m_block_idx * BLOCK_M` stride and
    OOB-indexing the B-weights tensor (manifests as IMA under CUDA-graph
    replay).
    """
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    dg.set_mk_alignment_for_contiguous_layout(value)


@contextlib.contextmanager
def mk_alignment_scope(value: int):
    """Temporarily set DeepGEMM's BLOCK_M cap, restoring on exit.

    Use around a sequence of grouped-contiguous GEMM calls whose workspace
    is padded to `value` (typically the per_call_align returned by
    `compute_aligned_M_and_alignment`).
    """
    prev = get_mk_alignment_for_contiguous_layout()[0]
    set_mk_alignment_for_contiguous_layout(value)
    try:
        yield
    finally:
        set_mk_alignment_for_contiguous_layout(prev)


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for DeepGEMM's get_mn_major_tma_aligned_tensor"""
    _lazy_init()
    if _get_mn_major_tma_aligned_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_tensor_impl(x)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    """Pack 4 UE8M0 (uint8) scales into one int32.

    DeepGEMM's SM100/SM120 FP8/FP4 kernels accept either ``float32`` scales
    (legacy format, 4 B/scale) or ``int32`` packed UE8M0 scales (1 B/scale
    after 4:1 packing — 4× smaller than the legacy fp32 representation).
    """
    _lazy_init()
    if _pack_ue8m0_to_int_impl is None:
        return _missing()
    return _pack_ue8m0_to_int_impl(x)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 (uint8) → int32 with the MN-major TMA-aligned layout the
    DeepGEMM kernels consume directly. 16× smaller than the fp32 legacy SF
    format. Use for non-grouped 2D scale tensors.
    """
    _lazy_init()
    if _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl(x)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
    ks_tensor: torch.Tensor,
    ks: list[int],
    gran_k: int,
) -> torch.Tensor:
    """Grouped (3D, expert-batched) variant of
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor``. Use for MoE weight
    scale tensors of shape ``(num_experts, mn, k_scale)``.
    """
    _lazy_init()
    if _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl is None:
        return _missing()
    return _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl(
        sf, ks_tensor, ks, gran_k
    )


def cublaslt_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _cublaslt_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _cublaslt_gemm_nt_impl(*args, **kwargs)


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    if "is_deep_gemm_e8m0_used" in kwargs:
        use_ue8m0 = kwargs["is_deep_gemm_e8m0_used"]
        del kwargs["is_deep_gemm_e8m0_used"]
    else:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    return _fp8_gemm_nt_impl(*args, disable_ue8m0_cast=not use_ue8m0, **kwargs)


def fp8_einsum(*args, **kwargs):
    _lazy_init()
    if _fp8_einsum_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_einsum_impl(*args, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_fp4_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_fp4_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def transform_sf_into_required_layout(*args, **kwargs):
    _lazy_init()
    if _transform_sf_into_required_layout_impl is None:
        return _missing(*args, **kwargs)
    return _transform_sf_into_required_layout_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def transform_weights_for_mega_moe(*args, **kwargs):
    _lazy_init()
    if _transform_weights_for_mega_moe_impl is None:
        return _missing(*args, **kwargs)
    return _transform_weights_for_mega_moe_impl(*args, **kwargs)


def get_symm_buffer_for_mega_moe(*args, **kwargs):
    _lazy_init()
    if _get_symm_buffer_for_mega_moe_impl is None:
        return _missing(*args, **kwargs)
    return _get_symm_buffer_for_mega_moe_impl(*args, **kwargs)


def fp8_fp4_mega_moe(*args, **kwargs):
    _lazy_init()
    if _fp8_fp4_mega_moe_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_fp4_mega_moe_impl(*args, **kwargs)


def fp8_fp4_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 MQA top-k indices without materializing full logits."""
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q[1] is None
    ):
        return False
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    return sm12x_deep_gemm_fallbacks.fp8_fp4_mqa_topk_indices(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    )


def _sm12x_native_deep_gemm() -> bool:
    # [sm12x-native] prefer vendored DeepGEMM sm120 MQA kernels over the
    # Python/Triton fallbacks when the env asks for it (see patch header).
    return os.environ.get("VLLM_SM12X_NATIVE_DEEPGEMM", "0") == "1"


def _sm12x_native_tf32() -> bool:
    return os.environ.get("VLLM_SM12X_NATIVE_TF32", "0") == "1"


# [sm12x-native] Bucketed KV padding for the native DeepGEMM MQA logits path.
#
# The vendored DeepGEMM C++ API has no out-tensor argument: every
# ``fp8_fp4_mqa_logits`` call allocates its logits output internally with
# ``torch::empty({align(M, block_q), align(N + block_kv, 64)})``
# (deepgemm-src csrc/apis/attention.hpp).  Under chunked prefill the
# gathered KV length N takes a new exact value on nearly every call, so the
# CUDA caching allocator sees a stream of unique large sizes, keeps
# reserving fresh segments for new record sizes and never returns the old
# ones to the OS.  On GB10 unified memory this reserved-but-idle pool
# directly shrinks the host MemAvailable (the memory ratchet observed under
# VLLM_SM12X_NATIVE_DEEPGEMM=1).
#
# Fix: round N up to a small geometric ladder (ratio <= 9/8) before the
# native call, preferably by zero-copy view extension of the indexer's
# persistent gather workspace.  The request byte-sizes then repeat, so
# after warmup every logits allocation is served from an already-cached
# block and segment growth stops.  Correctness: the SM120 kernel walks only
# the KV blocks inside each row's [cu_seqlen_ks, cu_seqlen_ke) window
# (sm120_fp8_mqa_logits.cuh derives its KV loop from ``end - start``) and
# the logits consumers (top_k_per_row_prefill, the DCP merge) read strictly
# inside that window, so the padded tail is never read and adds no compute;
# the wrapper also slices the returned view back to the caller's exact
# shape.  Set VLLM_SM12X_DG_ALLOC_BUCKETS=0 to disable.

_SM12X_DG_BUCKET_LADDER: list[int] = []
_SM12X_DG_KV_PAD_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_SM12X_DG_KV_PAD_CACHE_MAX = 4
_SM12X_DG_DBO_ENABLED_FN: Callable[[], bool] | None = None
_SM12X_DG_ALLOC_LOG_EVERY: int | None = None
_SM12X_DG_ALLOC_LOG_CALLS: int = 0


def _sm12x_dg_alloc_buckets_enabled() -> bool:
    return os.environ.get("VLLM_SM12X_DG_ALLOC_BUCKETS", "1") == "1"


def _sm12x_dg_dbo_active() -> bool:
    """True when this call runs inside a DBO (dual-batch overlap) ubatch thread.

    The per-bucket pad cache below is process-global while DBO ubatch
    threads run concurrently on separate streams, so a cached buffer could
    see one ubatch's ``copy_`` racing the other ubatch's kernel read.  The
    import is lazy because vllm.v1.worker.ubatching pulls in
    forward_context and this wrapper is imported very early; if the probe
    is unavailable we conservatively report True (cache gets bypassed).
    """
    global _SM12X_DG_DBO_ENABLED_FN
    fn = _SM12X_DG_DBO_ENABLED_FN
    if fn is None:
        try:
            from vllm.v1.worker.ubatching import dbo_enabled as fn
        except Exception:

            def fn() -> bool:
                return True

        _SM12X_DG_DBO_ENABLED_FN = fn
    return fn()


def _sm12x_dg_alloc_log_tick() -> None:
    """Env-gated allocator telemetry for the native MQA logits path.

    ``VLLM_SM12X_DG_ALLOC_LOG=N`` (N > 0) prints one stderr line every N
    calls with the direct ratchet signals of the CUDA caching allocator:
    reserved bytes, cudaMalloc/cudaFree counters and segment count.  The
    ratchet shows reserved_bytes and num_device_alloc climbing together;
    the bucketed steady state shows both flat.  Default: off (unset, "0"
    or unparsable).  Runs regardless of VLLM_SM12X_DG_ALLOC_BUCKETS so
    both A/B legs emit the same signal.
    """
    global _SM12X_DG_ALLOC_LOG_EVERY, _SM12X_DG_ALLOC_LOG_CALLS
    every = _SM12X_DG_ALLOC_LOG_EVERY
    if every is None:
        try:
            every = int(os.environ.get("VLLM_SM12X_DG_ALLOC_LOG", "0") or "0")
        except ValueError:
            every = 0
        _SM12X_DG_ALLOC_LOG_EVERY = every
    if every <= 0:
        return
    _SM12X_DG_ALLOC_LOG_CALLS += 1
    if _SM12X_DG_ALLOC_LOG_CALLS % every:
        return
    import sys
    import time

    try:
        stats = torch.cuda.memory_stats()
        reserved = stats.get(
            "reserved_bytes.all.current", torch.cuda.memory_reserved()
        )
        print(
            "[sm12x-dg-alloc] ts=%d calls=%d reserved_bytes=%d"
            " num_device_alloc=%d num_device_free=%d segments=%d"
            % (
                int(time.time()),
                _SM12X_DG_ALLOC_LOG_CALLS,
                reserved,
                stats.get("num_device_alloc", -1),
                stats.get("num_device_free", -1),
                stats.get("segment.all.current", -1),
            ),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass


def _sm12x_dg_bucket_up(n: int) -> int:
    """Round ``n`` up to a geometric ladder value (ratio <= 9/8, 64-aligned)."""
    ladder = _SM12X_DG_BUCKET_LADDER
    if not ladder:
        base = 4096
        while base <= (1 << 21):
            ladder.extend(base * num // 8 for num in range(8, 16))
            base <<= 1
    if n > ladder[-1]:
        return n
    for val in ladder:
        if val >= n:
            return val
    return n


def _sm12x_dg_storage_rows(t: torch.Tensor) -> int:
    """Number of rows addressable in ``t``'s storage from its offset."""
    row_stride = t.stride(0)
    if row_stride <= 0:
        return 0
    elems = t.untyped_storage().nbytes() // t.element_size()
    return max(0, (elems - t.storage_offset()) // row_stride)


def _sm12x_dg_padded_kv(
    kv_fp: torch.Tensor, kv_sf: torch.Tensor, n_pad: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(kv_fp, kv_sf)`` extended to ``n_pad`` rows, or None.

    Fast path: both tensors are dense row-major views into storages that
    already have room for ``n_pad`` rows (the indexer gathers KV into a
    large persistent workspace, so this is the common case); extending the
    view copies nothing.  Slow path: copy into a cached per-bucket buffer
    (bounded count).  Rows [N, n_pad) may hold stale bytes; they are never
    read by the kernel or the topk consumers (see note above).
    """
    n, head_dim = kv_fp.shape
    if (
        kv_fp.stride(1) == 1
        and kv_fp.stride(0) == head_dim
        and kv_sf.stride(0) == 1
        and _sm12x_dg_storage_rows(kv_fp) >= n_pad
        and _sm12x_dg_storage_rows(kv_sf) >= n_pad
    ):
        return (
            kv_fp.as_strided((n_pad, head_dim), (head_dim, 1), kv_fp.storage_offset()),
            kv_sf.as_strided((n_pad,), (1,), kv_sf.storage_offset()),
        )
    if kv_fp.element_size() != 1:
        return None
    if _sm12x_dg_dbo_active():
        # Concurrent ubatch threads must not share cached buffers (the
        # cache is process-global; one ubatch's copy_ would race the other
        # ubatch's kernel read).  Bypass the cache with a private buffer:
        # it is still bucket-sized, so the caching allocator reuses the
        # same block sizes and the ratchet fix holds.
        buf_fp = torch.zeros(
            (n_pad, head_dim), dtype=torch.uint8, device=kv_fp.device
        ).view(kv_fp.dtype)
        buf_sf = torch.zeros((n_pad,), dtype=kv_sf.dtype, device=kv_sf.device)
    else:
        cache = _SM12X_DG_KV_PAD_CACHE
        key = (n_pad, head_dim, kv_fp.dtype, kv_sf.dtype, str(kv_fp.device))
        bufs = cache.pop(key, None)
        if bufs is None:
            if len(cache) >= _SM12X_DG_KV_PAD_CACHE_MAX:
                cache.pop(next(iter(cache)))
            bufs = (
                torch.zeros(
                    (n_pad, head_dim), dtype=torch.uint8, device=kv_fp.device
                ).view(kv_fp.dtype),
                torch.zeros((n_pad,), dtype=kv_sf.dtype, device=kv_sf.device),
            )
        cache[key] = bufs
        buf_fp, buf_sf = bufs
    buf_fp[:n].copy_(kv_fp)
    buf_sf[:n].copy_(kv_sf)
    return buf_fp, buf_sf


def _fp8_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    return sm12x_deep_gemm_fallbacks._fp8_mqa_logits_sm12x(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)`` where ``scales`` is None for FP8 Q
    (per-token scale is folded into ``weights``) and a packed block-scale
    tensor for MXFP4 Q.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is [M, H, D]
            float8_e4m3fn and q_scale is None (per-token scale is folded
            into ``weights``). FP4 path: q_values is packed uint8 and
            q_scale is the companion block-scale tensor.
        kv: Tuple `(k_packed, k_scales)` — FP8 layout is [N, D]
            float8_e4m3fn plus fp32 scales [N]; FP4 layout is packed uint8.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query
            position, shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query
            position, shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    if current_platform.is_device_capability_family(120) and q[1] is None:
        if _sm12x_native_deep_gemm():  # [sm12x-native]
            _lazy_init()
            if _fp8_fp4_mqa_logits_impl is not None:
                # [sm12x-native] Optional allocator telemetry (default
                # off; runs for both A/B legs), then round the KV length
                # up to a bucket ladder so the kernel's internal logits
                # allocation comes from a small repeating set of
                # byte-sizes (see _sm12x_dg_bucket_up for the rationale).
                _sm12x_dg_alloc_log_tick()
                kv_call = kv
                n_kv = 0
                if (
                    _sm12x_dg_alloc_buckets_enabled()
                    and kv[0].dim() == 2
                    and kv[1].dim() == 1
                ):
                    n_kv = kv[0].shape[0]
                    if n_kv > 0:
                        n_pad = _sm12x_dg_bucket_up(n_kv)
                        if n_pad > n_kv:
                            padded = _sm12x_dg_padded_kv(kv[0], kv[1], n_pad)
                            if padded is not None:
                                kv_call = padded
                logits = _fp8_fp4_mqa_logits_impl(
                    q,
                    kv_call,
                    weights,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    clean_logits=clean_logits,
                )
                if kv_call is not kv:
                    logits = logits[:, :n_kv]
                return logits
        return _fp8_mqa_logits_sm12x(
            q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
        )
    _lazy_init()
    if _fp8_fp4_mqa_logits_impl is None:
        return _missing()
    return _fp8_fp4_mqa_logits_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_size: int,
    num_sms: int,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Args:
        context_lens: Tensor of shape [B], dtype int32; effective context length
            per batch element.
        block_size: KV-cache block size in tokens (e.g., 64).
        num_sms: Number of SMs available. 132 for Hopper
        indices: Optional request index for each varlen row.

    Returns:
        Backend-specific tensor consumed by `fp8_fp4_paged_mqa_logits` to
        schedule work across SMs.
    """
    _lazy_init()
    if _get_paged_mqa_logits_metadata_impl is None:
        return _missing()
    kwargs = {} if indices is None else {"indices": indices}
    return _get_paged_mqa_logits_metadata_impl(
        context_lens, block_size, num_sms, **kwargs
    )


def _fp8_paged_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    return sm12x_deep_gemm_fallbacks._fp8_paged_mqa_logits_sm12x(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def fp8_fp4_paged_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 paged MQA top-k indices without full logits."""
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and q[1] is None
    ):
        return False
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    return sm12x_deep_gemm_fallbacks.fp8_fp4_paged_mqa_topk_indices(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        topk_indices,
    )


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MQA logits using a paged KV-cache.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)``; pass ``(q_tensor, None)`` for the FP8
    path and ``(q_values, q_scale)`` for MXFP4.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is
            [B, next_n, H, D] float8_e4m3fn and q_scale is None. FP4 path:
            q_values is packed uint8 and q_scale is the companion
            block-scale tensor.
        kv_cache: Paged KV-cache. FP8 layout is [num_blocks, block_size, D+4]
            or [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. Within
            each block, the D-byte FP8 values for every token are stored first,
            followed by per-token fp32 scale bytes.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.
        clean_logits: Whether to clean the unfilled logits into `-inf`.
        indices: Optional request index for each varlen row.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    if current_platform.is_device_capability_family(120) and q[1] is None:
        if _sm12x_native_deep_gemm():  # [sm12x-native]
            _lazy_init()
            if _fp8_fp4_paged_mqa_logits_impl is not None:
                _kw = {} if indices is None else {"indices": indices}
                return _fp8_fp4_paged_mqa_logits_impl(
                    q,
                    kv_cache,
                    weights,
                    context_lens,
                    block_tables,
                    schedule_metadata,
                    max_model_len,
                    clean_logits=clean_logits,
                    **_kw,
                )
        return _fp8_paged_mqa_logits_sm12x(
            q, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    _lazy_init()
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()
    kwargs = {} if indices is None else {"indices": indices}
    return _fp8_fp4_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
        **kwargs,
    )


def _tf32_hc_prenorm_gemm_sm12x(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    return sm12x_deep_gemm_fallbacks._tf32_hc_prenorm_gemm_sm12x(
        x, fn, out, sqrsum, num_split
    )


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """
    Perform the following computation:
        out = x.float() @ fn.T
        sqrsum = x.float().square().sum(-1)

    See the caller function for shape requirement
    """
    if current_platform.is_device_capability_family(120):
        if _sm12x_native_tf32():  # [sm12x-native]
            _lazy_init()
            if _tf32_hc_prenorm_gemm_impl is not None:
                return _tf32_hc_prenorm_gemm_impl(x, fn, out, sqrsum, num_split)
        return _tf32_hc_prenorm_gemm_sm12x(x, fn, out, sqrsum, num_split)
    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/csrc/utils/math.hpp#L19
def get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align(x, 16 // element_size)


DEFAULT_BLOCK_SIZE = [128, 128]


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/dd6ed14acbc7445dcef224248a77ab4d22b5f240/deep_gemm/utils/math.py#L38
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int] = DEFAULT_BLOCK_SIZE, use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    x_padded = torch.zeros(
        (_align(m, block_m), _align(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    _, fp8_max = get_fp8_min_max()
    sf = x_amax / fp8_max
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(fp8_dtype)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing `torch.testing.assert_close` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report `1 - sim`.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    supports_deep_gemm: bool | None = None,
):
    if supports_deep_gemm is None:
        supports_deep_gemm = is_deep_gemm_supported()

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    return (
        supports_deep_gemm
        and output_dtype == torch.bfloat16
        and weight_shape[0] % N_MULTIPLE == 0
        and weight_shape[1] % K_MULTIPLE == 0
    )


__all__ = [
    "calc_diff",
    "DeepGemmQuantScaleFMT",
    "fp8_gemm_nt",
    "fp8_einsum",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_mqa_topk_indices",
    "fp8_fp4_paged_mqa_logits",
    "fp8_fp4_paged_mqa_topk_indices",
    "get_paged_mqa_logits_metadata",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "get_num_sms",
    "set_num_sms",
    "should_use_deepgemm_for_fp8_linear",
    "get_col_major_tma_aligned_tensor",
    "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
    "pack_ue8m0_to_int",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
    "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
]
