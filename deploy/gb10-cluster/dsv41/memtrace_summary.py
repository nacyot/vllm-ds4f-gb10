#!/usr/bin/env python3
"""Summarize VLLM_MEM_TRACE lines from a dsv41 rank log and a memlog.py CSV.

Usage: memtrace_summary.py <rank.log> [mem.csv] [t0 t1]

Prints the allocator's reserved/allocated peaks by scheduled-token bucket
(decode vs prefill chunks) and, with a CSV and a window, the host-side
MemAvailable minimum inside it.
"""

import csv
import re
import sys

LINE = re.compile(
    r"(\d\d-\d\d \d\d:\d\d:\d\d).*mem trace step (\d+): (\d+) tokens, (\d+) reqs; "
    r"allocated ([\d.]+) GiB \(peak ([\d.]+)\), reserved ([\d.]+) GiB "
    r"\(peak ([\d.]+)\), device free ([\d.]+) GiB"
)


def main() -> None:
    rows = []
    with open(sys.argv[1], errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                ts, step, tok, reqs, alloc, apeak, res, rpeak, free = m.groups()
                rows.append(
                    (
                        ts,
                        int(step),
                        int(tok),
                        int(reqs),
                        float(alloc),
                        float(apeak),
                        float(res),
                        float(rpeak),
                        float(free),
                    )
                )
    if not rows:
        print("no mem trace lines")
        return
    print(f"{len(rows)} trace lines, {rows[0][0]} .. {rows[-1][0]}")
    buckets = {"decode(<1024)": [], "chunk(1024-4096)": [], "chunk(>4096)": []}
    for r in rows:
        key = (
            "decode(<1024)"
            if r[2] < 1024
            else "chunk(1024-4096)"
            if r[2] <= 4096
            else "chunk(>4096)"
        )
        buckets[key].append(r)
    print(
        f"{'bucket':18s} {'n':>5s} {'alloc max':>10s} {'reserved max':>13s} "
        f"{'reserved peak':>14s} {'free min':>9s}"
    )
    for key, rs in buckets.items():
        if not rs:
            continue
        print(
            f"{key:18s} {len(rs):5d} {max(r[4] for r in rs):10.2f} "
            f"{max(r[6] for r in rs):13.2f} {max(r[7] for r in rs):14.2f} "
            f"{min(r[8] for r in rs):9.2f}"
        )
    last = rows[-1]
    print(
        f"last: step {last[1]} reserved {last[6]:.2f} GiB (peak {last[7]:.2f}), "
        f"device free {last[8]:.2f} GiB"
    )
    if len(sys.argv) >= 5:
        t0, t1 = float(sys.argv[3]), float(sys.argv[4])
        with open(sys.argv[2]) as f:
            win = [r for r in csv.DictReader(f) if t0 <= float(r["ts"]) <= t1]
        if win:
            m = min(win, key=lambda r: int(r["MemAvailable"]))
            print(
                f"host window: MemAvailable min {m['MemAvailable']} MiB at "
                f"t+{float(m['ts']) - t0:.0f}s, MemFree {m['MemFree']}, "
                f"file cache {int(m['Active_file']) + int(m['Inactive_file'])}, "
                f"Shmem {m['Shmem']}, RssShmem {m['RssShmem']}, VmSwap {m['VmSwap']}"
            )


if __name__ == "__main__":
    main()
