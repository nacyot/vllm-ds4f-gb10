#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-node host memory sampler for the DSv41 cluster (1 s interval, CSV).

Usage: memlog.py <out.csv> [interval_s=1] [engram_every=5]

Columns: /proc/meminfo (MiB), FOLL_PIN net pages, the vLLM worker's RSS split,
the resident pages of the two Engram shards (mincore, every `engram_every`
samples), then (issue #30) cumulative /proc/vmstat reclaim counters, the free
order>=9 (2 MiB) blocks of the Normal zone, and the anon/swap split of the
worker's parents (EngineCore, API server). Runs until SIGTERM. No torch.
"""

import ctypes
import ctypes.util
import glob
import mmap
import os
import signal
import subprocess
import sys
import time

MEMINFO = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Cached",
    "Shmem",
    "Mapped",
    "Active(file)",
    "Inactive(file)",
    "AnonPages",
    "Unevictable",
    "SwapFree",
)
STATUS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap")
# Issue #30 columns follow the original ones so older CSVs line up.
MEMINFO_EXTRA = ("SUnreclaim",)
VMSTAT = (
    "pswpin",
    "pswpout",
    "pgscan_kswapd",
    "pgscan_direct",
    "compact_stall",
    "compact_success",
    "allocstall_normal",
)
PARENT_STATUS = ("RssAnon", "VmSwap")
PARENTS = ("e", "a")  # EngineCore, API server: the worker's parent and grandparent
BUDDY_MIN_ORDER = 9
MODEL = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
SHARDS = sorted(glob.glob(os.path.join(MODEL, "model-0004[78]-of-00048.safetensors")))

libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_LOW_BIT = bytes(b & 1 for b in range(256))
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
MAP_FAILED = ctypes.c_void_p(-1).value


def parse_meminfo(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in MEMINFO or k in MEMINFO_EXTRA:
            out[k] = int(v.split()[0]) // 1024
    return out


def meminfo() -> dict[str, int]:
    with open("/proc/meminfo") as f:
        return parse_meminfo(f.read())


def parse_vmstat(text: str, keys=VMSTAT) -> dict[str, int]:
    out = dict.fromkeys(keys, 0)
    for line in text.splitlines():
        k, _, v = line.partition(" ")
        if k in out:
            out[k] = int(v)
    return out


def vmstat_text() -> str:
    with open("/proc/vmstat") as f:
        return f.read()


def foll_pin(text: str) -> int:
    v = parse_vmstat(text, ("nr_foll_pin_acquired", "nr_foll_pin_released"))
    return v["nr_foll_pin_acquired"] - v["nr_foll_pin_released"]


def parse_buddyinfo(text: str, min_order: int = BUDDY_MIN_ORDER) -> int:
    """Free blocks of order >= min_order in the Normal zone(s)."""
    total = 0
    for line in text.splitlines():
        _, zone, counts = line.partition("Normal")
        if zone:
            total += sum(int(c) for c in counts.split()[min_order:])
    return total


def free_order9plus() -> int:
    try:
        with open("/proc/buddyinfo") as f:
            return parse_buddyinfo(f.read())
    except OSError:
        return -1


def worker_pid() -> int | None:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "^VLLM::Worker"], capture_output=True, text=True
        ).stdout.split()
    except OSError:
        return None
    return int(out[0]) if out else None


def parent_pid(pid: int, proc: str = "/proc") -> int | None:
    try:
        with open(f"{proc}/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    # comm may contain spaces; ppid is the field after the closing paren.
    ppid = int(stat.rpartition(")")[2].split()[1])
    return ppid if ppid > 1 else None


def head_pids(worker: int | None, proc: str = "/proc") -> list[int | None]:
    """[worker, EngineCore, API server]: the worker's parent chain."""
    pids: list[int | None] = [worker]
    for _ in PARENTS:
        pids.append(parent_pid(pids[-1], proc) if pids[-1] else None)
    return pids


def parse_status(text: str, keys=STATUS) -> dict[str, int]:
    out = dict.fromkeys(keys, 0)
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in out:
            out[k] = int(v.split()[0]) // 1024
    return out


def status(pid: int | None, keys=STATUS, proc: str = "/proc") -> dict[str, int]:
    if pid is None:
        return dict.fromkeys(keys, 0)
    try:
        with open(f"{proc}/{pid}/status") as f:
            return parse_status(f.read(), keys)
    except OSError:
        return dict.fromkeys(keys, 0)


class Shard:
    def __init__(self, path: str) -> None:
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        # A read-only private mapping: mincore only needs the address, and
        # PROT_WRITE would add the shard size to Committed_AS (issue #30).
        self.base = libc.mmap(
            None, self.size, mmap.PROT_READ, mmap.MAP_PRIVATE, self.fd, 0
        )
        if self.base == MAP_FAILED:
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        self.pages = (self.size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        self.vec = ctypes.create_string_buffer(self.pages)

    def resident_mib(self) -> int:
        if libc.mincore(self.base, self.size, self.vec) != 0:
            return -1
        # Bit 0 of each byte: resident. translate+count runs in C (a Python
        # loop over 6M bytes cost a CPU core for a second per sample).
        n = self.vec.raw.translate(_LOW_BIT).count(b"\x01")
        return n * mmap.PAGESIZE // (1024 * 1024)


def columns(nshards: int) -> list[str]:
    return (
        ["ts"]
        + [k.replace("(", "_").replace(")", "") for k in MEMINFO]
        + ["foll_pin_pages", "worker_pid"]
        + list(STATUS)
        + [f"engram{i}_res_mib" for i in range(nshards)]
        + list(MEMINFO_EXTRA)
        + list(VMSTAT)
        + ["free_order9plus"]
        + [f"{p}_pid" for p in PARENTS]
        + [f"{p}_{k}" for p in PARENTS for k in PARENT_STATUS]
    )


def main() -> None:
    out_path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    engram_every = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    shards = [Shard(p) for p in SHARDS]
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    engram = [-1] * len(shards)
    i = 0
    with open(out_path, "a", buffering=1) as f:
        f.write(",".join(columns(len(shards))) + "\n")
        while not stop:
            t = time.time()
            mi = meminfo()
            vm = vmstat_text()
            pids = head_pids(worker_pid())
            st = status(pids[0])
            parents = [status(p, PARENT_STATUS) for p in pids[1:]]
            if i % engram_every == 0:
                engram = [s.resident_mib() for s in shards]
            row = (
                [f"{t:.1f}"]
                + [str(mi.get(k, 0)) for k in MEMINFO]
                + [str(foll_pin(vm)), str(pids[0] or 0)]
                + [str(st[k]) for k in STATUS]
                + [str(e) for e in engram]
                + [str(mi.get(k, 0)) for k in MEMINFO_EXTRA]
                + [str(v) for v in parse_vmstat(vm).values()]
                + [str(free_order9plus())]
                + [str(p or 0) for p in pids[1:]]
                + [str(ps[k]) for ps in parents for k in PARENT_STATUS]
            )
            f.write(",".join(row) + "\n")
            i += 1
            time.sleep(max(0.0, interval - (time.time() - t)))


if __name__ == "__main__":
    main()
