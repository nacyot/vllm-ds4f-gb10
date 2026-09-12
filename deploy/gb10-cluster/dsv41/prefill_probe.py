#!/usr/bin/env python3
"""Prefill throughput probe: unique long prompts (no prefix-cache hits), one at a
time, streaming; reports prompt tokens, TTFT and prompt tok/s per run and the
median. Send one warm-up request after a boot before using this (JIT).

Usage: prefill_probe.py <tag> [--tokens N] [--runs R] [--base URL] [--out DIR]
       (the record count is calibrated with /tokenize so every run has
       ~N prompt tokens; 20 s between runs so the previous run's offload
       store does not overlap the next prefill)
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvoff_probe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--tokens", type=int, default=32768)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--base", default="http://127.0.0.1:8889")
ap.add_argument("--out", default=os.path.expanduser("~/dsv41-prep/bench"))
ap.add_argument("--gap", type=float, default=20.0, help="seconds between runs")
args = ap.parse_args()
kvoff_probe.base = args.base
os.makedirs(args.out, exist_ok=True)


def count_tokens(salt, nrec):
    body = {
        "model": kvoff_probe.model,
        "messages": [{"role": "user", "content": kvoff_probe.build_prompt(salt, nrec)}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        args.base + "/tokenize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=600))["count"]


def records_for(salt, tokens):
    """Two tokenizer passes: linear guess, then rescale on the measured ratio."""
    nrec = max(10, tokens // 14)
    for _ in range(2):
        got = count_tokens(salt, nrec)
        nrec = max(10, round(nrec * tokens / got))
    return nrec


runs = []
for i in range(args.runs):
    salt = f"P{i}{int(time.time()) % 100000:05d}"
    nrec = records_for(salt, args.tokens)
    m0 = kvoff_probe.metrics()
    r = kvoff_probe.run_prompt(salt, nrec, max_tokens=4)
    m1 = kvoff_probe.metrics()
    r["prefix_hits"] = m1.get("prefix_cache_hits", 0) - m0.get("prefix_cache_hits", 0)
    r["tok_s"] = round((r["prompt_tokens"] or 0) / r["ttft_s"], 1)
    runs.append(r)
    print(json.dumps(r), flush=True)
    if i + 1 < args.runs:
        time.sleep(args.gap)

summary = {
    "tag": args.tag,
    "target_tokens": args.tokens,
    "prompt_tokens": runs[0]["prompt_tokens"],
    "ttft_median_s": statistics.median(r["ttft_s"] for r in runs),
    "tok_s_median": statistics.median(r["tok_s"] for r in runs),
    "tok_s_all": [r["tok_s"] for r in runs],
    "all_ok": all(r["ok"] for r in runs),
    "prefix_hits": sum(r["prefix_hits"] for r in runs),
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "runs": runs,
}
with open(os.path.join(args.out, f"prefill-{args.tag}.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(
    f"{args.tag}: {summary['prompt_tokens']} tok, TTFT median "
    f"{summary['ttft_median_s']:.2f} s, {summary['tok_s_median']:.0f} tok/s "
    f"(runs {summary['tok_s_all']}, ok={summary['all_ok']}, "
    f"prefix_hits={summary['prefix_hits']})"
)
