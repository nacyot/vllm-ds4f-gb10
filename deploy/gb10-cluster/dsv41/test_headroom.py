# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Headroom refusal and cancellation without vLLM, torch or cluster access."""

import importlib.util
import io
import json
import runpy
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import mock_open

import pytest

HERE = Path(__file__).parent


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(b'{"data":[{"id":"test"}]}'),
    )
    spec = importlib.util.spec_from_file_location(
        "kvoff_probe", HERE / "kvoff_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "kvoff_probe", module)
    monkeypatch.setattr(module, "POLL_SECONDS", 0.01)
    yield module
    if module._monitor.thread is not None:
        module._monitor.thread.join(timeout=2)
    assert not module._monitor.active
    assert module._monitor.thread is None


def install_answer(monkeypatch):
    calls = []

    def respond(request, **kwargs):
        calls.append(request)
        return io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"12"}}],'
            b'"usage":{"prompt_tokens":100}}\n'
            b"data: [DONE]\n"
        )

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    return calls


def test_long_prompt_refused_before_inference(probe, monkeypatch):
    calls = install_answer(monkeypatch)
    monkeypatch.setattr(probe, "mem_available_gib", lambda: 4.5)
    with pytest.raises(probe.HeadroomError):
        probe.run_prompt("ABC", 33000)
    assert calls == []


@pytest.mark.parametrize(
    "records,minimum,threshold,available",
    [
        (700, 5.2, 18000, 4.5),
        (18000, 5.2, 18000, 5.2),
        (18000, 0, 18000, 4.5),
        (18000, 5.2, 0, 4.5),
    ],
)
def test_allowed_requests_preserve_answer(
    probe, monkeypatch, records, minimum, threshold, available
):
    calls = install_answer(monkeypatch)
    monkeypatch.setattr(probe, "mem_available_gib", lambda: available)
    monkeypatch.setattr(probe, "MIN_AVAIL_GIB", minimum)
    monkeypatch.setattr(probe, "LONG_PROMPT_RECORDS", threshold)
    result = probe.run_prompt("ABC", records)
    assert result["ok"] and result["answer"] == "12"
    assert len(calls) == 1


def test_remote_base_skips_local_memory_with_warning(probe, monkeypatch, capsys):
    calls = install_answer(monkeypatch)
    monkeypatch.setattr(probe, "base", "http://remote.example:8889")
    result = probe.run_prompt("ABC", 18000)
    assert len(calls) == 1
    assert result["mem_avail_start_gib"] is None
    assert result["mem_avail_min_gib"] is None
    assert "skipping headroom" in capsys.readouterr().err


def test_meminfo_conversion_and_missing_file(probe, monkeypatch):
    monkeypatch.setattr(
        "builtins.open", mock_open(read_data="MemAvailable: 5452595 kB\n")
    )
    assert probe.mem_available_gib() == pytest.approx(5.2, abs=1e-6)
    monkeypatch.setattr("builtins.open", mock_open())
    assert probe.mem_available_gib() is None
    missing = mock_open()
    missing.side_effect = FileNotFoundError
    monkeypatch.setattr("builtins.open", missing)
    assert probe.mem_available_gib() is None


@pytest.mark.parametrize(
    "script,args",
    [
        ("kvoff_probe.py", ["negative", "ABC", "33000"]),
        ("kvoff_concurrent.py", ["negative", "2", "33000"]),
        ("prefill_probe.py", ["negative", "--tokens", "493015", "--runs", "1"]),
    ],
)
def test_cli_refusal_is_json_and_exit_three(
    probe, monkeypatch, capsys, tmp_path, script, args
):
    def respond(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        assert not url.endswith("/chat/completions")
        if url.endswith("/v1/models"):
            return io.BytesIO(b'{"data":[{"id":"test"}]}')
        if url.endswith("/tokenize"):
            return io.BytesIO(b'{"count":493015}')
        return io.BytesIO(b"vllm:prefix_cache_queries 0\n")

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    monkeypatch.setattr(probe, "mem_available_gib", lambda: 4.5)
    if script == "prefill_probe.py":
        args += ["--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", [script, *args])
    with pytest.raises(SystemExit) as exc:
        if script == "kvoff_probe.py":
            sys.exit(probe.main())
        else:
            runpy.run_path(str(HERE / script), run_name="__main__")
    assert exc.value.code == 3
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "tag": "negative",
        "skipped": "headroom",
        "mem_avail_gib": 4.5,
        "min_gib": 5.2,
    }


def test_shared_monitor_interrupts_streams_before_first_token(probe, monkeypatch):
    """Use real blocked HTTP reads: BytesIO.close cannot prove cancellation."""
    release = threading.Event()
    arrived = threading.Barrier(3)
    real_urlopen = _REAL_URLOPEN

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            arrived.wait(timeout=5)
            release.wait(timeout=5)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    available = [5.0]
    monkeypatch.setattr(probe, "base", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(probe, "MIN_AVAIL_GIB", 0)
    monkeypatch.setattr(probe, "mem_available_gib", lambda: available[0])
    monkeypatch.setattr(urllib.request, "urlopen", real_urlopen)
    try:
        with ThreadPoolExecutor(2) as executor:
            futures = [
                executor.submit(probe.run_prompt, salt, 18000)
                for salt in ("ABC", "DEF")
            ]
            arrived.wait(timeout=5)
            # Wait on the monitor lock until both responses are registered.
            deadline = threading.Event()

            end = time.monotonic() + 2
            while time.monotonic() < end:
                with probe._monitor.lock:
                    if len(probe._monitor.active) == 2:
                        monitor_thread = probe._monitor.thread
                        break
                deadline.wait(0.01)
            else:
                pytest.fail("both streams were not registered")
            assert monitor_thread.is_alive()
            available[0] = 2.7
            results = [future.result(timeout=2) for future in futures]
        assert all(r["aborted"] == "headroom" and not r["ok"] for r in results)
        assert all(r["mem_avail_start_gib"] == 5.0 for r in results)
        assert all(r["mem_avail_min_gib"] == 2.7 for r in results)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


_REAL_URLOPEN = urllib.request.urlopen


@pytest.mark.parametrize("abort_threshold,aborted", [(2.8, True), (0, False)])
def test_monitor_closes_fake_response_or_respects_disabled_abort(
    probe, monkeypatch, abort_threshold, aborted
):
    sampled = threading.Event()
    closed = threading.Event()

    class Response(io.BytesIO):
        def __iter__(self):
            assert sampled.wait(timeout=2)
            if aborted:
                assert closed.wait(timeout=2)
                return iter(())
            return super().__iter__()

        def close(self):
            closed.set()
            super().close()

    def memory():
        if threading.current_thread() is threading.main_thread():
            return 5.0
        sampled.set()
        return 2.7

    monkeypatch.setattr(probe, "MIN_AVAIL_GIB", 0)
    monkeypatch.setattr(probe, "ABORT_BELOW_GIB", abort_threshold)
    monkeypatch.setattr(probe, "mem_available_gib", memory)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: Response())
    result = probe.run_prompt("ABC", 18000)
    assert bool(result.get("aborted")) == aborted
    assert result["mem_avail_min_gib"] == 2.7
    assert closed.is_set()
