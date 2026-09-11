# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader import ep_weight_filter
from vllm.model_executor.model_loader.weight_utils import (
    _split_files_by_skip_registry,
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_files_holding,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


def _write_shards(tmp_path):
    plain = tmp_path / "model-00001-of-00003.safetensors"
    mixed = tmp_path / "model-00002-of-00003.safetensors"
    table_only = tmp_path / "model-00003-of-00003.safetensors"
    save_file({"layers.0.w": torch.zeros(2), "layers.1.w": torch.zeros(2)}, plain)
    save_file(
        {
            "layers.1.engram.embed.weight": torch.zeros(2),
            "layers.1.engram.q_weight": torch.zeros(2),
        },
        mixed,
    )
    save_file({"layers.2.engram.embed.weight": torch.zeros(2)}, table_only)
    return str(plain), str(mixed), str(table_only)


def test_split_files_by_skip_registry(tmp_path, monkeypatch):
    """InstantTensor cannot skip a tensor inside a file, so a shard holding a
    registered on-disk tensor leaves its list: mixed shards go to the lazy
    safetensors tail, all-registered shards are dropped, and the expected
    names never include the registered tensors."""
    plain, mixed, table_only = _write_shards(tmp_path)
    monkeypatch.setattr(ep_weight_filter, "_SKIP_SUFFIXES", {"engram.embed.weight"})

    stream, lazy, dropped, expected = _split_files_by_skip_registry(
        [table_only, mixed, plain]
    )

    assert stream == [plain]
    assert lazy == [mixed]
    assert dropped == [table_only]
    assert expected == {"layers.0.w", "layers.1.w", "layers.1.engram.q_weight"}


def test_split_files_without_registry_streams_every_shard(tmp_path, monkeypatch):
    plain, mixed, table_only = _write_shards(tmp_path)
    monkeypatch.setattr(ep_weight_filter, "_SKIP_SUFFIXES", set())

    stream, lazy, dropped, expected = _split_files_by_skip_registry(
        [mixed, plain, table_only]
    )

    assert stream == [plain, mixed, table_only]
    assert lazy == [] and dropped == []
    assert len(expected) == 5


def test_safetensors_files_holding(tmp_path):
    """A folded draft picks its shards out of the checkpoint index; no index
    means no pruning (None), a predicate matching nothing means no shards."""
    index = {
        "weight_map": {
            "layers.0.w": "model-00001-of-00004.safetensors",
            "embed.weight": "model-00002-of-00004.safetensors",
            "mtp.0.w": "model-00003-of-00004.safetensors",
            "mtp.1.w": "model-00004-of-00004.safetensors",
            "mtp.1.scale": "model-00004-of-00004.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert safetensors_files_holding(
        str(tmp_path), lambda name: name.startswith("mtp.")
    ) == ["model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors"]
    assert safetensors_files_holding(str(tmp_path), lambda name: False) == []
    assert safetensors_files_holding(str(tmp_path / "missing"), lambda _: True) is None
    (tmp_path / "noindex").mkdir()
    assert safetensors_files_holding(str(tmp_path / "noindex"), lambda _: True) is None


if __name__ == "__main__":
    test_instanttensor_model_loader()
