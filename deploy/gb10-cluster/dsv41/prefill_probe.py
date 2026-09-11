#!/usr/bin/env python3
"""Prefill throughput probe: unique long prompts (no prefix-cache hits), one at a
time, streaming; reports prompt tokens, TTFT and prompt tok/s per run and the
median. Send one warm-up request after a boot before using this (JIT).

Usage: prefill_probe.py <tag> [--records N] [--runs R] [--base URL] [--out DIR]
       (1450 records ~ 32K tokens with the default salt; 20 s between runs
       so the previous run's offload store does not overlap the next prefill)
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvoff_probe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--records", type=int, default=1450)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--base", default="http://127.0.0.1:8889")
ap.add_argument("--out", default=os.path.expanduser("~/dsv41-prep/bench"))
ap.add_argument("--gap", type=float, default=20.0, help="seconds between runs")
args = ap.parse_args()
kvoff_probe.base = args.base
os.makedirs(args.out, exist_ok=True)

runs = []
for i in range(args.runs):
    salt = f"PF{args.tag}-{i}-{int(time.time()) % 100000}"
    m0 = kvoff_probe.metrics()
    r = kvoff_probe.run_prompt(salt, args.records, max_tokens=4)
    m1 = kvoff_probe.metrics()
    r["prefix_hits"] = m1.get("prefix_cache_hits", 0) - m0.get("prefix_cache_hits", 0)
    r["tok_s"] = round((r["prompt_tokens"] or 0) / r["ttft_s"], 1)
    runs.append(r)
    print(json.dumps(r), flush=True)
    if i + 1 < args.runs:
        time.sleep(args.gap)

summary = {
    "tag": args.tag,
    "records": args.records,
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
