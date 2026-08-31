# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Flash-Vision-Exp image preprocessing vs. the reference.

Imports our ``vllm/models/deepseek_v4/image_processing.py`` and DeepSeek's
reference ``inference/image_processor.py`` by file path (no compiled vLLM
extension needed) and checks that, for the two example images and a sweep of
synthetic sizes, the patches, grid sizes, token types and permutation agree
for every ``start_pos % 4``. Skipped when the materials are not present.
"""

import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.cpu_test

MATERIALS = Path(os.environ.get("DS4V_MATERIALS", "/tmp/ds4v"))
REF_IMAGE_PROCESSOR = MATERIALS / "ref_image_processor.py"
IMAGES = [MATERIALS / "imgs" / "carrots.jpeg", MATERIALS / "imgs" / "corn.jpeg"]
CONFIG = Path(os.environ.get("DS4V_CONFIG", "/tmp/cfg_vis.json"))

OURS = (
    Path(__file__).resolve().parents[2] / "vllm/models/deepseek_v4/image_processing.py"
)

requires_materials = pytest.mark.skipif(
    not (REF_IMAGE_PROCESSOR.is_file() and CONFIG.is_file() and OURS.is_file()),
    reason="DeepSeek-V4 vision materials not present",
)


def _import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ours():
    return _import_by_path("ds4v_image_processing", OURS)


@pytest.fixture(scope="module")
def ref():
    return _import_by_path("ds4v_ref_image_processor", REF_IMAGE_PROCESSOR)


@pytest.fixture(scope="module")
def cfg():
    raw = json.loads(CONFIG.read_text())
    vision = {k: v for k, v in raw.items() if k.startswith("vision_")}
    assert vision, "config has no vision_* keys"
    return SimpleNamespace(**vision, vocab_size=raw["vocab_size"])


def _ref_load(ref, cfg, image: Image.Image):
    """Run the reference ``load_image`` (bytes in) on a PIL image."""
    buf = io.BytesIO()
    image.save(buf, format="PPM")
    return ref.load_image({"data": buf.getvalue()}, cfg)


def _compare(ours, ref, cfg, image: Image.Image, label: str) -> dict[int, int]:
    r_patches, r_vit_h, r_vit_w, r_llm_h, r_llm_w = _ref_load(ref, cfg, image)
    o_patches, o_vit_h, o_vit_w, o_llm_h, o_llm_w = ours.preprocess_image(image, cfg)

    assert (o_vit_h, o_vit_w) == (r_vit_h, r_vit_w), label
    assert (o_llm_h, o_llm_w) == (r_llm_h, r_llm_w), label
    assert o_patches.dtype == torch.bfloat16, label
    assert o_patches.shape == r_patches.shape, label
    assert o_patches.shape == (
        o_vit_h * o_vit_w,
        3,
        cfg.vision_patch_size,
        cfg.vision_patch_size,
    )
    assert torch.equal(o_patches, r_patches), label

    # The solver-only entry point must agree with the pixel path.
    grid = ours.solve_image_grid(image.width, image.height, cfg)
    assert grid[:4] == (o_vit_h, o_vit_w, o_llm_h, o_llm_w), label

    counts: dict[int, int] = {}
    for start_pos in range(4):
        r_types, r_perm = ref.build_image_block(r_llm_h, r_llm_w, start_pos)
        o_types, o_perm = ours.build_image_block(o_llm_h, o_llm_w, start_pos)
        assert o_types.dtype == torch.int64 and o_perm.dtype == torch.int64
        assert torch.equal(o_types, r_types), (label, start_pos)
        assert torch.equal(o_perm, r_perm), (label, start_pos)
        n = ours.num_image_tokens(o_llm_h, o_llm_w, start_pos)
        assert n == len(o_types), (label, start_pos)
        assert n <= cfg.vision_max_n_token, (label, start_pos, n)
        # One IMAGE slot per aligner output, and the first grid token lands on
        # a compressor block boundary.
        assert int((o_types == ours.IMAGE).sum()) == o_llm_h * o_llm_w == len(o_perm)
        first_grid = start_pos + int((o_types == ours.IMAGE_START).nonzero()[0]) + 1
        assert first_grid % ours.COMPRESS_PAD_TO == 0, (label, start_pos)
        assert o_types[-1] == ours.IMAGE_END
        counts[start_pos] = n
    return counts


@requires_materials
@pytest.mark.parametrize("image_path", IMAGES, ids=[p.name for p in IMAGES])
def test_example_images_match_reference(ours, ref, cfg, image_path):
    if not image_path.is_file():
        pytest.skip(f"{image_path} not present")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    counts = _compare(ours, ref, cfg, image, image_path.name)
    _, _, _, n_llm_h, n_llm_w = ours.preprocess_image(image, cfg)
    print(
        f"{image_path.name}: size={image.size} n_llm=({n_llm_h}, {n_llm_w}) "
        f"tokens by start_pos%4={counts}"
    )


def _sweep_sizes() -> list[tuple[int, int]]:
    fixed = [
        (20, 20),
        (31, 47),
        (64, 64),
        (100, 37),
        (128, 128),
        (200, 200),
        (300, 100),
        (383, 383),
        (384, 384),
        (385, 385),
        (450, 308),
        (512, 512),
        (640, 480),
        (700, 1024),
        (756, 756),
        (768, 768),
        (1024, 701),
        (1024, 1024),
        (1932, 336),
        (1931, 335),
        (2000, 250),
        (2048, 2048),
        (3000, 300),
        (4000, 4000),
        (4000, 20),
        (20, 4000),
        (3999, 401),
        (88, 3872),
        (336, 1932),
        (1600, 200),
        (1601, 200),
        (200, 1600),
        (250, 2000),
        (1200, 150),
        (1000, 125),
        (3200, 400),
        (3201, 400),
        (4000, 500),
        (4000, 499),
        (4000, 3000),
        (3000, 4000),
        (21, 4000),
        (4000, 21),
        (999, 111),
        (111, 999),
        (1400, 1400),
        (1401, 1399),
        (2500, 2500),
        (3333, 1111),
        (1111, 3333),
        (600, 75),
        (75, 600),
        (2800, 350),
        (2801, 349),
        (500, 500),
        (501, 499),
        (1500, 1000),
        (1000, 1500),
        (2400, 1800),
        (1800, 2400),
        (3840, 2160),
        (2160, 3840),
        (1920, 1080),
        (1080, 1920),
        (4000, 333),
        (333, 4000),
    ]
    rng = np.random.default_rng(20260831)
    random = [
        (int(w), int(h))
        for w, h in rng.integers(20, 4001, size=(24, 2), dtype=np.int64)
    ]
    return fixed + random


SWEEP = _sweep_sizes()


@requires_materials
def test_sweep_matches_reference(ours, ref, cfg):
    assert len(SWEEP) >= 60
    rng = np.random.default_rng(0)
    min_pixels = cfg.vision_min_pixels
    ratio = cfg.vision_max_wh_ratio
    seen_tiny = seen_wide = False
    max_tokens = 0
    for width, height in SWEEP:
        seen_tiny |= width * height < min_pixels
        seen_wide |= width > height * ratio
        # Noise makes the patch comparison sensitive to any resize/pad drift.
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        counts = _compare(ours, ref, cfg, image, f"{width}x{height}")
        max_tokens = max(max_tokens, counts[0])
        print(f"{width}x{height}: tokens by start_pos%4={counts}")
    assert seen_tiny, "sweep must include images below vision_min_pixels"
    assert seen_wide, "sweep must include aspect ratios above vision_max_wh_ratio"

    # Profiling dummy: reaches the largest block the solver can emit.
    dummy_w, dummy_h = ours.max_tokens_image_size(cfg)
    _, _, n_llm_h, n_llm_w, _, _ = ours.solve_image_grid(dummy_w, dummy_h, cfg)
    dummy_tokens = ours.num_image_tokens(n_llm_h, n_llm_w, 0)
    assert dummy_tokens >= max_tokens, (dummy_tokens, max_tokens)
    assert dummy_tokens == cfg.vision_max_n_token - (ours.COMPRESS_PAD_TO - 1)
    # Round trip through the pixel path with the same kind of blank image the
    # dummy builder makes.
    blank = Image.new("RGB", (dummy_w, dummy_h), color=255)
    _compare(ours, ref, cfg, blank, "dummy")
    _, n_vit_h, n_vit_w, h, w = ours.preprocess_image(blank, cfg)
    assert (h, w) == (n_llm_h, n_llm_w)
    print(
        f"dummy image {dummy_w}x{dummy_h}: n_vit=({n_vit_h}, {n_vit_w}) "
        f"n_llm=({n_llm_h}, {n_llm_w}) tokens@0={dummy_tokens}"
    )


@requires_materials
def test_constants_and_helpers(ours, ref, cfg):
    assert (
        ours.IMAGE_START,
        ours.IMAGE_PAD,
        ours.IMAGE,
        ours.IMAGE_NEW_LINE,
        ours.IMAGE_END,
    ) == (
        ref.IMAGE_START,
        ref.IMAGE_PAD,
        ref.IMAGE,
        ref.IMAGE_NEW_LINE,
        ref.IMAGE_END,
    )
    assert ours.COMPRESS_PAD_TO == ref.COMPRESS_PAD_TO
    assert ours.IMAGE_PLACEHOLDER == "<｜deepseek_image｜>"

    # Pure helpers agree with the reference for a range of grids.
    p, d, budget = (
        cfg.vision_patch_size,
        cfg.vision_downsample_ratio,
        cfg.vision_max_n_token,
    )
    for bh in range(p, 60 * p, 7 * p):
        for bw in range(p, 60 * p, 5 * p):
            assert ours.grid_tokens(bh, bw, p, d) == ref.grid_tokens(bh, bw, p, d)
            assert ours.safe_resize(bh, bw, bh, bw, p, d, budget) == ref.safe_resize(
                bh, bw, bh, bw, p, d, budget
            )
    for h, w in [(10, 3), (3, 10), (1, 200), (200, 1), (37, 41)]:
        for n in (50, 100, 381):
            assert ours.solve_resize_ratio(h, w, p, d, n) == ref.solve_resize_ratio(
                h, w, p, d, n
            )

    # Sentinel ids are appended after the vocabulary, five of them.
    types, _ = ours.build_image_block(2, 3, 1)
    ids = (types + cfg.vocab_size).tolist()
    assert min(ids) >= cfg.vocab_size and max(ids) < cfg.vocab_size + 5

    # as_pil_image accepts HWC / CHW uint8 arrays and tensors.
    pixels = np.random.default_rng(1).integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    for candidate in (
        pixels,
        np.transpose(pixels, (2, 0, 1)),
        torch.from_numpy(pixels),
        Image.fromarray(pixels),
    ):
        image = ours.as_pil_image(candidate)
        assert image.size == (40, 30)
        assert np.array_equal(np.asarray(image), pixels)
    with pytest.raises(TypeError):
        ours.as_pil_image(pixels.astype(np.float32))
