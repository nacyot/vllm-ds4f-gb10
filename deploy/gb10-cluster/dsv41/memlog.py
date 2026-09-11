#!/usr/bin/env python3
"""Per-node host memory sampler for the DSv41 cluster (1 s interval, CSV).

Usage: memlog.py <out.csv> [interval_s=1] [engram_every=5]

Columns: /proc/meminfo (MiB), FOLL_PIN net pages, the vLLM worker's RSS split,
and the resident pages of the two Engram shards (mincore, every `engram_every`
samples). Runs until SIGTERM.
"""

import ctypes
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
MODEL = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
SHARDS = sorted(glob.glob(os.path.join(MODEL, "model-0004[78]-of-00048.safetensors")))

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in MEMINFO:
                out[k] = int(v.split()[0]) // 1024
    return out


def foll_pin() -> int:
    acq = rel = 0
    with open("/proc/vmstat") as f:
        for line in f:
            if line.startswith("nr_foll_pin_acquired"):
                acq = int(line.split()[1])
            elif line.startswith("nr_foll_pin_released"):
                rel = int(line.split()[1])
    return acq - rel


def worker_pid() -> int | None:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "VLLM::Worker"], capture_output=True, text=True
        ).stdout.split()
    except OSError:
        return None
    return int(out[0]) if out else None


def status(pid: int | None) -> dict[str, int]:
    out = dict.fromkeys(STATUS, 0)
    if pid is None:
        return out
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in STATUS:
                    out[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return out


class Shard:
    def __init__(self, path: str) -> None:
        self.fd = os.open(path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        # MAP_PRIVATE + PROT_WRITE: from_buffer needs a writable buffer and a
        # private mapping never touches the file; mincore sees the page cache.
        self.map = mmap.mmap(
            self.fd,
            self.size,
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.map))
        self.pages = (self.size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        self.vec = ctypes.create_string_buffer(self.pages)

    def resident_mib(self) -> int:
        if libc.mincore(self.base, self.size, self.vec) != 0:
            return -1
        # Bit 0 of each byte: resident.
        n = sum(b & 1 for b in self.vec.raw)
        return n * mmap.PAGESIZE // (1024 * 1024)


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
    cols = (
        ["ts"]
        + [k.replace("(", "_").replace(")", "") for k in MEMINFO]
        + ["foll_pin_pages", "worker_pid"]
        + list(STATUS)
        + [f"engram{i}_res_mib" for i in range(len(shards))]
    )
    engram = [-1] * len(shards)
    i = 0
    with open(out_path, "a", buffering=1) as f:
        f.write(",".join(cols) + "\n")
        while not stop:
            t = time.time()
            mi = meminfo()
            pid = worker_pid()
            st = status(pid)
            if i % engram_every == 0:
                engram = [s.resident_mib() for s in shards]
            row = (
                [f"{t:.1f}"]
                + [str(mi.get(k, 0)) for k in MEMINFO]
                + [str(foll_pin()), str(pid or 0)]
                + [str(st[k]) for k in STATUS]
                + [str(e) for e in engram]
            )
            f.write(",".join(row) + "\n")
            i += 1
            time.sleep(max(0.0, interval - (time.time() - t)))


if __name__ == "__main__":
    main()
