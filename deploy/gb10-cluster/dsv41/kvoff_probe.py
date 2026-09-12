#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cold-restore probe: send a long unique-prefix prompt, check the answer and
report TTFT with the prefix-cache / kv_offload counter deltas.

Usage: kvoff_probe.py <tag> [salt=A] [records=700]   (700 records ~ 8.4K tokens)
"""

import contextlib
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request

base = "http://127.0.0.1:8889"
MIN_AVAIL_GIB = float(os.environ.get("DSV41_MIN_AVAIL_GIB", "5.2"))
ABORT_BELOW_GIB = float(os.environ.get("DSV41_ABORT_BELOW_GIB", "2.8"))
LONG_PROMPT_RECORDS = int(os.environ.get("DSV41_LONG_PROMPT_RECORDS", "18000"))
POLL_SECONDS = 2


class HeadroomError(RuntimeError):
    def __init__(self, available):
        self.available = available
        super().__init__(
            f"Head MemAvailable {available:.2f} GiB < {MIN_AVAIL_GIB:.2f} GiB; "
            "restart the adopted configuration, then retry."
        )

    def report(self, tag):
        print(str(self), file=sys.stderr)
        print(
            json.dumps(
                {
                    "tag": tag,
                    "skipped": "headroom",
                    "mem_avail_gib": self.available,
                    "min_gib": MIN_AVAIL_GIB,
                }
            )
        )


def mem_available_gib():
    if urllib.parse.urlsplit(base).hostname not in ("127.0.0.1", "localhost"):
        return None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return None


class _HeadroomMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = []
        self.thread = None

    def register(self, response, start):
        state = {"response": response, "minimum": start, "aborted": False}
        with self.lock:
            self.active.append(state)
            if self.thread is None:
                self.thread = threading.Thread(target=self.watch, daemon=True)
                self.thread.start()
        return state

    def unregister(self, state):
        with self.lock:
            self.active.remove(state)

    def watch(self):
        while True:
            with self.lock:
                if not self.active:
                    self.thread = None
                    return
                available = mem_available_gib()
                for state in self.active:
                    if available is None:
                        continue
                    state["minimum"] = min(state["minimum"], available)
                    if (
                        not state["aborted"]
                        and ABORT_BELOW_GIB > 0
                        and available < ABORT_BELOW_GIB
                    ):
                        state["aborted"] = True
                        response = state["response"]
                        # close alone can wait for the reader's buffered-I/O lock.
                        try:
                            response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                        except (AttributeError, OSError):
                            response.close()
            time.sleep(POLL_SECONDS)


_monitor = _HeadroomMonitor()

model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0][
    "id"
]

KEYS = (
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:kv_offload_load_bytes",
    "vllm:kv_offload_store_bytes",
)
SUMS = (
    "vllm:kv_offload_store_time",
    "vllm:kv_offload_load_time",
    "vllm:kv_offload_lookup_sync_delay_seconds",
)


def metrics():
    out = {}
    for line in urllib.request.urlopen(base + "/metrics", timeout=10):
        s = line.decode()
        for k in KEYS:
            if s.startswith(k) and not s.startswith(k + "_created"):
                with contextlib.suppress(ValueError):
                    out[k.removeprefix("vllm:")] = out.get(
                        k.removeprefix("vllm:"), 0
                    ) + float(s.rsplit(" ", 1)[1])
        for k in SUMS:
            if s.startswith(k + "_sum"):
                with contextlib.suppress(ValueError):
                    out[k.removeprefix("vllm:") + "_sum"] = float(s.rsplit(" ", 1)[1])
    return out


def build_prompt(salt, nrec):
    facts = "\n".join(
        f"Record {salt}-{i}: sensor {i} reads {(i * 37 + 11) % 997}."
        for i in range(1, nrec)
    )
    return (
        "The following is a list of sensor records.\n"
        + facts
        + "\nQuestion: what does sensor 512 read? Answer with the number only."
    )


def run_prompt(salt, nrec, max_tokens=8):
    """Stream one request; returns prompt tokens, TTFT, answer and expected."""
    guarded = LONG_PROMPT_RECORDS > 0 and nrec >= LONG_PROMPT_RECORDS
    start = mem_available_gib() if guarded else None
    if guarded and start is None:
        print(
            "Warning: head MemAvailable unavailable; skipping headroom guards.",
            file=sys.stderr,
        )
    if start is not None and MIN_AVAIL_GIB > 0 and start < MIN_AVAIL_GIB:
        raise HeadroomError(start)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(salt, nrec)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    text = []
    usage = {}
    state = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        if start is not None:
            state = _monitor.register(resp, start)
        try:
            for raw in resp:
                if state is not None and state["aborted"]:
                    break
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[5:])
                if ch.get("usage"):
                    usage = ch["usage"]
                for c in ch.get("choices", []):
                    piece = c.get("delta", {}).get("content")
                    if piece:
                        if first is None:
                            first = time.perf_counter()
                        text.append(piece)
        except (OSError, ValueError, http.client.HTTPException):
            if state is None or not state["aborted"]:
                raise
        finally:
            if state is not None:
                _monitor.unregister(state)
    ttft = (first or time.perf_counter()) - t0
    expected = str((512 * 37 + 11) % 997)
    answer = "".join(text).strip()[:20]
    result = {
        "salt": salt,
        "prompt_tokens": usage.get("prompt_tokens"),
        "ttft_s": round(ttft, 3),
        "answer": answer,
        "expected": expected,
        "ok": answer.startswith(expected),
        "mem_avail_start_gib": start,
        "mem_avail_min_gib": state["minimum"] if state else start,
    }
    if state is not None and state["aborted"]:
        result.update(aborted="headroom", ok=False)
    return result


def main():
    tag = sys.argv[1]
    salt = sys.argv[2] if len(sys.argv) > 2 else "A"
    nrec = int(sys.argv[3]) if len(sys.argv) > 3 else 700
    m0 = metrics()
    try:
        r = run_prompt(salt, nrec)
    except HeadroomError as exc:
        exc.report(tag)
        return 3
    time.sleep(2)
    m1 = metrics()
    r["tag"] = tag
    r["delta"] = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1)}
    print(json.dumps(r, ensure_ascii=False))
    return 3 if r.get("aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
