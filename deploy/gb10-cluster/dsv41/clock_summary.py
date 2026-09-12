#!/usr/bin/env python3
"""Summarize clock_ctl.sh sampler CSVs: per node, SM clock range under load
(samples with power above the idle band), max power, max temperature and the
throttle-reason bits seen. Usage: clock_summary.py <csv>... [--load-w 20]"""

import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument(
    "--load-w",
    type=float,
    default=20.0,
    help="power above which a sample counts as load",
)
args = ap.parse_args()

HW_THERMAL, HW_SLOWDOWN, SW_THERMAL, SW_POWER = 0x40, 0x8, 0x20, 0x4
for path in args.csv:
    rows = []
    with open(path) as f:
        for r in csv.reader(f):
            if len(r) < 5 or r[0].startswith("timestamp"):
                continue
            try:
                rows.append(
                    (
                        int(r[1].split()[0]),
                        float(r[2].split()[0]),
                        int(r[3]),
                        int(r[4].strip(), 16),
                    )
                )
            except ValueError:
                continue
    if not rows:
        print(f"{path}: no samples")
        continue
    load = [r for r in rows if r[1] >= args.load_w] or rows
    bits = 0
    for r in rows:
        bits |= r[3]
    flags = [
        n
        for n, b in (
            ("HW_THERMAL", HW_THERMAL),
            ("HW_SLOWDOWN", HW_SLOWDOWN),
            ("SW_THERMAL", SW_THERMAL),
            ("SW_POWER", SW_POWER),
        )
        if bits & b
    ]
    print(
        f"{path.rsplit('/', 1)[-1]}: samples={len(rows)} load={len(load)} "
        f"sm_load={min(r[0] for r in load)}-{max(r[0] for r in load)} MHz "
        f"sm_idle_max={max(r[0] for r in rows)} "
        f"power_max={max(r[1] for r in rows):.1f} W "
        f"temp_max={max(r[2] for r in rows)} C "
        f"throttle_bits=0x{bits:x} {','.join(flags) or 'none'}"
    )
