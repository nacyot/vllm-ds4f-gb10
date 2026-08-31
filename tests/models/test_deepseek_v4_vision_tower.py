# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Flash-Vision-Exp vision tower + aligner vs. the reference.

Loads the real ``vision.*`` / ``aligner.*`` tensors from a checkpoint shard
into both DeepSeek's reference ``inference/vision.py`` modules and our port
(``vllm/models/deepseek_v4/vision.py``), runs a real image through both on CPU
in fp32 and checks the aligner outputs agree. Our module is imported by file
path so the test needs no compiled vLLM extension; it is skipped when the
materials (shard, reference sources, image) are not present.
"""

import importlib.util
import json
import math
import os
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_test

MATERIALS = Path(os.environ.get("DS4V_MATERIALS", "/tmp/ds4v"))
SHARD = MATERIALS / "shard1.safetensors"
REF_VISION = MATERIALS / "ref_vision.py"
REF_IMAGE_PROCESSOR = MATERIALS / "ref_image_processor.py"
IMAGE = MATERIALS / "imgs" / "carrots.jpeg"
CONFIG = Path(os.environ.get("DS4V_CONFIG", "/tmp/cfg_vis.json"))

OURS = Path(__file__).resolve().parents[2] / "vllm/models/deepseek_v4/vision.py"

VISION_PREFIX = "vision."
ALIGNER_PREFIX = "aligner."


def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vision_fields() -> tuple[dict, int]:
    with open(CONFIG) as f:
        cfg = json.load(f)
    fields = {k: v for k, v in cfg.items() if k.startswith("vision_")}
    assert fields["vision_n_layers"] > 0
    return fields, cfg["hidden_size"]


def _load_prefix(module: torch.nn.Module, f, prefix: str) -> None:
    """Strict load of every checkpoint key under ``prefix`` into ``module``."""
    # safe_open handles are not mappings: keys() is a method, no __contains__.
    ckpt_keys = {k for k in list(f.keys()) if k.startswith(prefix)}
    assert ckpt_keys, prefix
    own_keys = set(module.state_dict().keys())
    ckpt_stripped = {k[len(prefix) :] for k in ckpt_keys}
    assert own_keys == ckpt_stripped, (
        f"{prefix}: missing={sorted(ckpt_stripped - own_keys)} "
        f"unexpected={sorted(own_keys - ckpt_stripped)}"
    )
    state = {k[len(prefix) :]: f.get_tensor(k) for k in ckpt_keys}
    missing, unexpected = module.load_state_dict(state, strict=True)
    assert not missing and not unexpected


@pytest.mark.skipif(
    not (
        SHARD.exists()
        and REF_VISION.exists()
        and REF_IMAGE_PROCESSOR.exists()
        and IMAGE.exists()
        and CONFIG.exists()
    ),
    reason="DeepSeek-V4 vision materials not present",
)
def test_vision_tower_matches_reference():
    from safetensors import safe_open

    ref_vision = _import_by_path("ds4v_ref_vision", REF_VISION)
    ref_proc = _import_by_path("ds4v_ref_image_processor", REF_IMAGE_PROCESSOR)
    ours = _import_by_path("ds4v_our_vision", OURS)

    fields, hidden_size = _vision_fields()
    ref_args = types.SimpleNamespace(**fields, dim=hidden_size)
    our_cfg = types.SimpleNamespace(**fields, hidden_size=hidden_size)

    torch.manual_seed(0)
    ref_vit = ref_vision.ViT(ref_args)
    ref_aligner = ref_vision.Aligner(ref_args)
    our_vit = ours.DeepseekV4ViT(our_cfg)
    our_aligner = ours.DeepseekV4Aligner(our_cfg)

    # Parameter names must equal the checkpoint keys (strict, all consumed).
    with safe_open(str(SHARD), framework="pt", device="cpu") as f:
        for module in (ref_vit, our_vit):
            _load_prefix(module, f, VISION_PREFIX)
        for module in (ref_aligner, our_aligner):
            _load_prefix(module, f, ALIGNER_PREFIX)
    assert len(list(our_vit.state_dict())) == 259
    assert len(list(our_aligner.state_dict())) == 4
    assert our_vit.state_dict().keys() == ref_vit.state_dict().keys()
    assert our_aligner.state_dict().keys() == ref_aligner.state_dict().keys()
    # RMSNorm weights stay fp32 (reference layout) after loading BF16 tensors.
    assert our_vit.norm.weight.dtype == torch.float32
    assert our_vit.blocks[0].norm1.weight.dtype == torch.float32

    # Real image -> patches via the reference image processor.
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = ref_proc.load_image(
        {"url": str(IMAGE)}, ref_args
    )
    assert patches.dtype == torch.bfloat16
    p = fields["vision_patch_size"]
    assert patches.shape == (n_vit_h * n_vit_w, 3, p, p)
    r = fields["vision_downsample_ratio"]
    assert (n_llm_h, n_llm_w) == (math.ceil(n_vit_h / r), math.ceil(n_vit_w / r))

    patches32 = patches.float()
    with torch.inference_mode():
        for m in (ref_vit, ref_aligner, our_vit, our_aligner):
            m.float().eval()
        ref_feat = ref_vit(patches32, n_vit_h, n_vit_w)
        ref_out = ref_aligner(ref_feat, n_vit_h, n_vit_w)
        our_feat = our_vit(patches32, n_vit_h, n_vit_w)
        our_out = our_aligner(our_feat, n_vit_h, n_vit_w)

    assert ref_feat.shape == (n_vit_h * n_vit_w, fields["vision_dim"])
    assert our_feat.shape == ref_feat.shape
    assert our_out.shape == (n_llm_h * n_llm_w, hidden_size)
    assert our_out.shape == ref_out.shape
    assert our_out.dtype == torch.float32
    assert torch.isfinite(our_out).all()
    feat_diff = (our_feat - ref_feat).abs().max().item()
    out_diff = (our_out - ref_out).abs().max().item()
    assert feat_diff < 1e-3, feat_diff
    assert out_diff < 1e-3, out_diff

    # bf16 path (the serving dtype) executes end to end. Both towers are run
    # in bf16 so the check is against the reference's own bf16 numerics, not
    # against fp32: 32 bf16 blocks drift enough (per-token cosine vs fp32 can
    # dip to ~0.8) that an fp32 tolerance would say nothing about the port.
    with torch.inference_mode():
        for m in (ref_vit, ref_aligner, our_vit, our_aligner):
            m.to(torch.bfloat16)
        ref_bf = ref_aligner(ref_vit(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
        our_bf = our_aligner(our_vit(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)
    assert our_bf.dtype == torch.bfloat16
    assert our_bf.shape == our_out.shape
    assert torch.isfinite(our_bf).all()
    bf_diff = (our_bf.float() - ref_bf.float()).abs().max().item()
    assert bf_diff == 0.0, bf_diff
    cos = torch.nn.functional.cosine_similarity(our_bf.float(), our_out, dim=-1)
    assert cos.mean().item() > 0.95, cos.mean().item()
