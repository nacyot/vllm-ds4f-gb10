# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from types import SimpleNamespace

import pytest
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import (
    _instanttensor_draft_load_config,
    get_model_loader,
    register_model_loader,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)


def _write_indexed_shards(tmp_path):
    names = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
    for name in names:
        (tmp_path / name).write_bytes(b"\x00" * 8)
    index = {"weight_map": {f"w{i}": name for i, name in enumerate(names)}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return names


def test_default_loader_exact_overrides_are_unioned_without_index_check(tmp_path):
    # A folded draft names the exact shards holding its tensors: every one of
    # them is used, and the index completeness check must not reject the
    # intentional subset.
    names = _write_indexed_shards(tmp_path)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=[names[2], names[1]],
    )
    assert use_safetensors is True
    assert [os.path.basename(f) for f in files] == [names[1], names[2]]


def test_default_loader_glob_override_keeps_index_check(tmp_path):
    # Control: a glob override keeps the first-match ladder and the index
    # completeness check, which rejects a subset of the indexed shards.
    _write_indexed_shards(tmp_path)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(FileNotFoundError, match="missing"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=["model-0000[23]-of-00003.safetensors"],
        )


def _spec_configs(draft_model="m", draft_revision=None):
    target = SimpleNamespace(model="m", revision=None)
    draft = SimpleNamespace(model=draft_model, revision=draft_revision)
    vllm_config = SimpleNamespace(
        load_config=LoadConfig(load_format="instanttensor"),
        model_config=target,
        speculative_config=SimpleNamespace(
            target_model_config=target, draft_model_config=draft
        ),
    )
    return vllm_config, target, draft


def test_draft_load_config_auto_reads_folded_draft_with_lazy_safetensors(
    monkeypatch,
):
    monkeypatch.delenv("INSTANTTENSOR_DRAFT_LOADER", raising=False)
    vllm_config, target, draft = _spec_configs()

    resolved = _instanttensor_draft_load_config(vllm_config, draft, None)

    assert resolved.load_format == "safetensors"
    assert resolved.safetensors_load_strategy == "lazy"
    # The target keeps InstantTensor and the shared config is untouched.
    assert (
        _instanttensor_draft_load_config(vllm_config, target, None)
        is vllm_config.load_config
    )
    assert vllm_config.load_config.load_format == "instanttensor"


def test_draft_load_config_auto_keeps_instanttensor_for_a_separate_draft(
    monkeypatch,
):
    monkeypatch.delenv("INSTANTTENSOR_DRAFT_LOADER", raising=False)
    vllm_config, _, draft = _spec_configs(draft_model="other/draft")
    assert (
        _instanttensor_draft_load_config(vllm_config, draft, None)
        is vllm_config.load_config
    )


def test_draft_load_config_modes(monkeypatch):
    vllm_config, _, draft = _spec_configs(draft_model="other/draft")
    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "safetensors")
    resolved = _instanttensor_draft_load_config(vllm_config, draft, None)
    assert resolved.load_format == "safetensors"

    vllm_config, _, draft = _spec_configs()
    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "instanttensor")
    assert (
        _instanttensor_draft_load_config(vllm_config, draft, None)
        is vllm_config.load_config
    )

    monkeypatch.setenv("INSTANTTENSOR_DRAFT_LOADER", "bogus")
    with pytest.raises(ValueError, match="INSTANTTENSOR_DRAFT_LOADER"):
        _instanttensor_draft_load_config(vllm_config, draft, None)


def test_draft_load_config_is_a_noop_without_instanttensor(monkeypatch):
    monkeypatch.delenv("INSTANTTENSOR_DRAFT_LOADER", raising=False)
    vllm_config, _, draft = _spec_configs()
    vllm_config.load_config = LoadConfig(load_format="safetensors")
    assert (
        _instanttensor_draft_load_config(vllm_config, draft, None)
        is vllm_config.load_config
    )
