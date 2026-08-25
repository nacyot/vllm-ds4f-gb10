# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for exact-length reads in the tiering worker wrapper."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.tiering.spec import (
    TieringOffloadingSpec,
    _read_exact_file,
)


class _FakeRegion:
    def __init__(self, row_bytes: int = 16):
        self.mmap_obj = bytearray(row_bytes)

    def cleanup(self):
        return


class _FakeWorker:
    def __init__(self):
        self.submit_load_calls = 0
        self.prepared_load_ready: list[bool] = []

    def submit_load(self, *args):
        self.submit_load_calls += 1
        return True

    def submit_prepared_load(self, *args, local_ready: bool):
        self.prepared_load_ready.append(local_ready)
        if not local_ready:
            return False
        self.submit_load_calls += 1
        return True


def _make_spec() -> TieringOffloadingSpec:
    spec = object.__new__(TieringOffloadingSpec)
    spec.config = SimpleNamespace(
        canonical_layout=False,
        parallel=SimpleNamespace(world_size=1),
    )
    spec.replicated_layout = True
    spec.num_blocks = 1
    spec.kv_bytes_per_chunk = 16
    spec.cpu_page_size_per_worker = 16
    spec.blocks_per_chunk = 1
    spec._engine_id = "test"
    return spec


def _make_src_spec(path: str, expected_bytes: int):
    return SimpleNamespace(
        block_ids=np.array([0], dtype=np.int64),
        fs_paths=[(path, 0, expected_bytes)],
    )


def test_direct_pread_rejects_short_read_without_submitting_h2d(tmp_path, monkeypatch):
    path = tmp_path / "short.bin"
    path.write_bytes(b"short")
    region = _FakeRegion()
    worker = _FakeWorker()

    import vllm.v1.kv_offload.tiering.spec as spec_module

    monkeypatch.setattr(spec_module, "SharedOffloadRegion", lambda **_: region)
    monkeypatch.setattr(spec_module, "CPUOffloadingWorker", lambda **_: worker)

    wrapped_worker = _make_spec().create_worker(MagicMock())
    result = wrapped_worker.submit_load(
        1,
        _make_src_spec(str(path), expected_bytes=8),
        MagicMock(),
    )

    assert result is False
    assert worker.submit_load_calls == 0
    assert worker.prepared_load_ready == [False]


def test_direct_pread_rejects_eof_without_submitting_h2d(tmp_path, monkeypatch):
    path = tmp_path / "eof.bin"
    path.write_bytes(b"")
    region = _FakeRegion()
    worker = _FakeWorker()

    import vllm.v1.kv_offload.tiering.spec as spec_module

    monkeypatch.setattr(spec_module, "SharedOffloadRegion", lambda **_: region)
    monkeypatch.setattr(spec_module, "CPUOffloadingWorker", lambda **_: worker)

    wrapped_worker = _make_spec().create_worker(MagicMock())
    result = wrapped_worker.submit_load(
        1,
        _make_src_spec(str(path), expected_bytes=8),
        MagicMock(),
    )

    assert result is False
    assert worker.submit_load_calls == 0
    assert worker.prepared_load_ready == [False]


def test_relay_source_read_error_returns_structured_failure(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.bin"
    region = _FakeRegion()
    worker = _FakeWorker()

    import vllm.distributed.parallel_state as parallel_state
    import vllm.v1.kv_offload.tiering.spec as spec_module

    group = MagicMock(world_size=2, rank_in_group=0)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setenv("DSPARK_RELAY_RESTORE", "1")
    monkeypatch.setattr(spec_module, "SharedOffloadRegion", lambda **_: region)
    monkeypatch.setattr(spec_module, "CPUOffloadingWorker", lambda **_: worker)

    class _Status:
        def __init__(self, value):
            self.value = value[0]

        def item(self):
            return self.value

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda value, **_: _Status(value),
    )

    wrapped_worker = _make_spec().create_worker(MagicMock())
    src_spec = _make_src_spec(str(missing_path), expected_bytes=8)
    result = wrapped_worker.submit_load(1, src_spec, MagicMock())

    assert result is False
    group.broadcast.assert_called_once()
    assert worker.submit_load_calls == 0
    assert worker.prepared_load_ready == [False]


def test_pread_skip_preserves_existing_success_path(tmp_path, monkeypatch):
    missing_path = tmp_path / "skipped.bin"
    region = _FakeRegion()
    worker = _FakeWorker()

    import vllm.v1.kv_offload.tiering.spec as spec_module

    monkeypatch.setenv("DSPARK_FS_PREAD_SKIP", "1")
    monkeypatch.setattr(spec_module, "SharedOffloadRegion", lambda **_: region)
    monkeypatch.setattr(spec_module, "CPUOffloadingWorker", lambda **_: worker)

    wrapped_worker = _make_spec().create_worker(MagicMock())
    result = wrapped_worker.submit_load(
        1,
        _make_src_spec(str(missing_path), expected_bytes=8),
        MagicMock(),
    )

    assert result is True
    assert worker.submit_load_calls == 1
    assert worker.prepared_load_ready == [True]


def test_exact_read_preserves_success(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")

    assert _read_exact_file(str(path), expected_bytes=7, source="test") == b"payload"


def test_cpu_worker_final_guard_rejects_before_h2d():
    worker = object.__new__(CPUOffloadingWorker)
    worker._load_handler = MagicMock()
    worker._load_admission_guard = lambda local_ready: False

    accepted = worker.submit_prepared_load(
        1,
        MagicMock(),
        MagicMock(),
        local_ready=True,
    )

    assert accepted is False
    worker._load_handler.transfer_async.assert_not_called()


def test_cpu_worker_final_guard_receives_rank_local_prepare_failure():
    worker = object.__new__(CPUOffloadingWorker)
    worker._load_handler = MagicMock()
    observed: list[bool] = []

    def guard(local_ready: bool) -> bool:
        observed.append(local_ready)
        return local_ready

    worker._load_admission_guard = guard

    accepted = worker.submit_prepared_load(
        1,
        MagicMock(),
        MagicMock(),
        local_ready=False,
    )

    assert accepted is False
    assert observed == [False]
    worker._load_handler.transfer_async.assert_not_called()
