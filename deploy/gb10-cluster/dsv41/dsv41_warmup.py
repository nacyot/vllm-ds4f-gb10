#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Post-start warm-up of the head API, run by dsv41-warmup.service.

The first request after a boot paid 21 s TTFT for JIT kernels and the
FlashInfer autotune (2026-09-13). Once /health answers, send one request of
each kind with a fresh salt, so none of them restores from the KV tiers: an
~8K greedy prompt, a short sampled request with thinking (rejection and
resample kernels) and one image (vision encoder). Prints one JSON line per
request; failures are logged, never raised: recovery belongs to the watchdog.
"""

import binascii
import http.client
import json
import os
import struct
import subprocess
import time
import urllib.request
import zlib

BASE = f"http://127.0.0.1:{os.environ.get('PORT', '8888')}"
HEALTH_WAIT_S = 1500


def log(**fields):
    print(json.dumps({"time": time.strftime("%F %T"), **fields}), flush=True)


def head_unit():
    out = subprocess.run(
        ["systemctl", "--user", "show", "dsv41-serve.service"]
        + ["-p", "ActiveState", "-p", "InvocationID"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return props.get("ActiveState", ""), props.get("InvocationID", "")


def wait_health():
    """Returns None once /health is 200, else why the warm-up is skipped."""
    _, invocation = head_unit()
    deadline = time.monotonic() + HEALTH_WAIT_S
    while time.monotonic() < deadline:
        state, current = head_unit()
        if current != invocation or state not in ("active", "activating"):
            return f"dsv41-serve {state or 'gone'}, invocation changed"
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as resp:
                if resp.status == 200:
                    return None
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(5)
    return f"no /health 200 within {HEALTH_WAIT_S} s"


def png(seed, size=64):
    """Four vertical stripes whose colours follow the seed."""
    colors = [
        bytes(((seed * 67 + i * 97) % 256, (seed * 31 + i * 53) % 256, i * 71 % 256))
        for i in range(4)
    ]
    row = b"\x00" + b"".join(colors[x * 4 // size] for x in range(size))

    def chunk(kind, data):
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def send(name, model, body):
    body = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        **body,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    text = []
    usage = {}
    error = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                data = json.loads(line[5:])
                usage = data.get("usage") or usage
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning"):
                        first = first or time.perf_counter()
                    text.append(delta.get("content") or "")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        error = f"{type(exc).__name__}: {exc}"
    log(
        request=name,
        ok=error is None and first is not None,
        error=error,
        ttft_s=round(first - t0, 3) if first else None,
        elapsed_s=round(time.perf_counter() - t0, 3),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        text="".join(text)[:80],
    )


def main():
    if os.environ.get("FRONTEND_ADDR"):
        log(skipped="FRONTEND_ADDR is set: the API is not on the head")
        return 0
    skipped = wait_health()
    if skipped:
        log(skipped=skipped)
        return 0
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as resp:
            model = json.load(resp)["data"][0]["id"]
    except (OSError, ValueError, KeyError, IndexError) as exc:
        log(skipped=f"/v1/models: {exc}")
        return 0
    seed = int(time.time())
    salt = f"W{seed}"
    log(start=salt, model=model)
    records = "\n".join(
        f"Record {salt}-{i}: sensor {i} reads {(i * 37 + 11) % 997}."
        for i in range(1, 700)
    )
    prompt = (
        "The following is a list of sensor records.\n"
        f"{records}\nQuestion: what does sensor 512 read? Answer with the number only."
    )
    no_thinking = {"chat_template_kwargs": {"thinking": False}}
    send(
        "text-8k-greedy",
        model,
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0,
            **no_thinking,
        },
    )
    question = f"[{salt}] In two sentences: why do GPU servers warm up after a boot?"
    send(
        "sampled-thinking",
        model,
        {
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 32,
            "temperature": 0.6,
            "top_p": 0.95,
            "chat_template_kwargs": {"thinking": True},
        },
    )
    image = binascii.b2a_base64(png(seed), newline=False).decode()
    content = [
        {"type": "text", "text": f"[{salt}]"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        {"type": "text", "text": "How many vertical stripes are there? Number only."},
    ]
    send(
        "image",
        model,
        {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 8,
            "temperature": 0,
            **no_thinking,
        },
    )
    log(done=salt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
