#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stream one long decode and report the gaps between streamed chunks.

A chunk gap is one engine step as seen by the client, so a step that is
stalled by another request's scheduler work shows up as an outlier gap.
The kv_offload lookup histogram is read before and after so the same run
reports how many lookups of 10 ms and more the server did meanwhile.

Usage: decode_gap_probe.py <tag> [--tokens N] [--base URL] [--out DIR]
"""

import argparse
import json
import os
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--tokens", type=int, default=1500)
ap.add_argument("--base", default="http://127.0.0.1:8889")
ap.add_argument("--out", default=os.path.expanduser("~/dsv41-prep/bench"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
base = args.base
model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0][
    "id"
]
HIST = "vllm:kv_offload_lookup_sync_delay_seconds"
PROMPT = (
    "Count from 1 upwards, one number per line, with no other text. "
    "Keep going until you are stopped."
)


def lookup_hist():
    out = {}
    for line in urllib.request.urlopen(base + "/metrics", timeout=10):
        s = line.decode()
        if not s.startswith(HIST):
            continue
        name, val = s.rsplit(" ", 1)
        if name.startswith(HIST + "_bucket"):
            key = "le_" + name.split('le="')[1].split('"')[0]
        else:
            key = name[len(HIST) + 1 :].split("{")[0]
        out[key] = float(val)
    return out


def stream():
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.tokens,
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
    stamps = []
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                if c.get("delta", {}).get("content"):
                    stamps.append(time.perf_counter() - t0)
    return stamps, usage


h0 = lookup_hist()
stamps, usage = stream()
time.sleep(2)
h1 = lookup_hist()

gaps = [(b - a, b) for a, b in zip(stamps, stamps[1:])]
gaps.sort(reverse=True)
n = usage.get("completion_tokens", 0)
decode_s = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
p99 = gaps[max(0, int(len(gaps) * 0.01) - 1)][0] if gaps else 0.0
hist_delta = {k: h1.get(k, 0) - h0.get(k, 0) for k in sorted(h1)}
median = sorted(g for g, _ in gaps)[len(gaps) // 2] if gaps else 0.0
result = {
    "tag": args.tag,
    "completion_tokens": n,
    "chunks": len(stamps),
    "ttft_s": round(stamps[0], 3) if stamps else None,
    "decode_s": round(decode_s, 3),
    "decode_tok_s": round((n - 1) / decode_s, 2) if decode_s > 0 and n > 1 else None,
    "gap_median_ms": round(median * 1000, 1),
    "gap_p99_ms": round(p99 * 1000, 1),
    "gap_max_ms": round(gaps[0][0] * 1000, 1) if gaps else None,
    "gaps_over_2x_median": sum(1 for g, _ in gaps if g > 2 * median),
    "gaps_over_150ms": sum(1 for g, _ in gaps if g > 0.15),
    "top_gaps": [
        {"ms": round(g * 1000, 1), "at_s": round(at, 2)} for g, at in gaps[:10]
    ],
    "lookups_10ms_plus": hist_delta.get("le_+Inf", 0) - hist_delta.get("le_0.01", 0),
    "lookups_50ms_plus": hist_delta.get("le_+Inf", 0) - hist_delta.get("le_0.05", 0),
    "lookup_count": hist_delta.get("count", 0),
    "lookup_sum_s": round(hist_delta.get("sum", 0), 3),
}
with open(os.path.join(args.out, f"gap-{args.tag}.json"), "w") as f:
    json.dump({"result": result, "stamps": stamps, "hist_delta": hist_delta}, f)
print(json.dumps(result))
