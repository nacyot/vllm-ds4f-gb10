#!/usr/bin/env python3
"""Fire N unique long prompts at once (distinct salts) and report per-request
TTFT, answer correctness and the prefix-cache / kv_offload counter deltas.
Used for the concurrent cold-restore test after a server restart.

Usage: kvoff_concurrent.py <tag> <n> <records_per_prompt> [salt_prefix]
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from kvoff_probe import metrics, run_prompt

tag, n, nrec = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
prefix = sys.argv[4] if len(sys.argv) > 4 else "S"
m0 = metrics()
t0 = time.perf_counter()
with ThreadPoolExecutor(n) as ex:
    results = list(ex.map(lambda i: run_prompt(f"{prefix}{i}", nrec), range(n)))
wall = time.perf_counter() - t0
time.sleep(2)
m1 = metrics()
delta = {k: m1.get(k, 0) - m0.get(k, 0) for k in set(m0) | set(m1)}
print(
    json.dumps(
        {
            "tag": tag,
            "n": n,
            "wall_s": round(wall, 3),
            "requests": results,
            "delta": delta,
        },
        ensure_ascii=False,
    )
)
