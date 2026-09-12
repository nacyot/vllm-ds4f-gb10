#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fire N unique long prompts at once (distinct salts) and report per-request
TTFT, answer correctness and the prefix-cache / kv_offload counter deltas.
Used for the concurrent cold-restore test after a server restart.

Usage: kvoff_concurrent.py <tag> <n> <records_per_prompt> [salt_prefix|salt,salt,...]

A comma-separated fourth argument names the salts explicitly (e.g. sessions
stored earlier), and then n is the number of those salts to use.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from kvoff_probe import HeadroomError, metrics, run_prompt

tag, n, nrec = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
prefix = sys.argv[4] if len(sys.argv) > 4 else "S"
salts = prefix.split(",")[:n] if "," in prefix else [f"{prefix}{i}" for i in range(n)]
m0 = metrics()
t0 = time.perf_counter()
try:
    with ThreadPoolExecutor(len(salts)) as ex:
        results = list(ex.map(lambda salt: run_prompt(salt, nrec), salts))
except HeadroomError as exc:
    exc.report(tag)
    sys.exit(3)
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
sys.exit(3 if any(r.get("aborted") for r in results) else 0)
