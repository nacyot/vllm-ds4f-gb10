#!/usr/bin/env python3
"""Cold-restore probe: send a long unique-prefix prompt, check the answer and
report TTFT with the prefix-cache / kv_offload counter deltas.

Usage: kvoff_probe.py <tag> [salt=A] [records=700]   (700 records ~ 8.4K tokens)
"""

import contextlib
import json
import sys
import time
import urllib.request

base = "http://127.0.0.1:8889"
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
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
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
    ttft = (first or time.perf_counter()) - t0
    expected = str((512 * 37 + 11) % 997)
    answer = "".join(text).strip()[:20]
    return {
        "salt": salt,
        "prompt_tokens": usage.get("prompt_tokens"),
        "ttft_s": round(ttft, 3),
        "answer": answer,
        "expected": expected,
        "ok": answer.startswith(expected),
    }


def main():
    tag = sys.argv[1]
    salt = sys.argv[2] if len(sys.argv) > 2 else "A"
    nrec = int(sys.argv[3]) if len(sys.argv) > 3 else 700
    m0 = metrics()
    r = run_prompt(salt, nrec)
    time.sleep(2)
    m1 = metrics()
    r["tag"] = tag
    r["delta"] = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1)}
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
