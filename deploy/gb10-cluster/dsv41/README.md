# DeepSeek-V4.1-Flash on four GB10 nodes

Run `dsv41_ctl.sh` from a workstation with SSH access to gx10-6040,
gx10-f323, gx10-37cc, and gx10-27c4. It requires Bash 4 or newer; on macOS:

```bash
/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/dsv41_ctl.sh caps
/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/dsv41_ctl.sh status
```

## Files

| File | Role |
| --- | --- |
| `dsv41_ctl.sh` | Workstation start, stop, status, caps, and log commands. |
| `serve-node.sh` | Launch one TP rank as the `dsv41-serve` user service. |
| `serve-frontend.sh`, `kv_transfer_json.sh` | API server without an engine on `FRONTEND_HOST` (issue #24 spike, off by default); the shared `--kv-transfer-config` builder. |
| `dsv41.env` | Adopted server defaults, overridable through the environment. |
| `clock_ctl.sh` | Clock observations and sampling; `cap` is owner-only recovery. |
| `selftest_caps.sh` | Local cap fixtures and simulated SSH tests without node access. |
| `bench.py` | Before/after latency, throughput, and greedy-output checks. |
| `bench2.py`, `bench_prompts_v1.json` | Category and concurrency benchmarks and prompts. |
| `prefill_probe.py` | Prefill throughput measurements with unique prompts. |
| `divergence.py` | Record and compare greedy completions and log probabilities. |
| `smoke.py` | Basic API and deterministic-output smoke checks. |
| `kvoff_probe.py`, `kvoff_concurrent.py` | Cold and concurrent KV offload restore probes. |
| `kvfs_gc.sh` | KV filesystem retention and size management. |
| `memlog.py`, `memtrace_summary.py` | Memory sampling and allocator-log summaries. |
| `clock_summary.py` | Summarize sampled clock CSVs. |
| `prof_summary.py` | Summarize GPU profiler traces. |

## Startup and cap checks

`dsv41_ctl.sh start` checks all four caps before any rank starts. After the
check passes, it launches workers rank 3 (27c4), 2 (37cc), 1 (f323), waits
five seconds, then launches head rank 0 (6040). Each rank uses
`serve-node.sh` and the defaults in `dsv41.env`. The API is on head port 8889.

`dsv41_ctl.sh caps` prints `HOST SERVICE PERSISTENCE MAX_SM_MHZ`. Each node
must have an active `gpu-clock-cap.service`, persistence `Enabled`, and a
maximum SM clock of at most 2000 MHz across five samples, 200 ms apart.
Normal idle readings are 1989 MHz. This is an observation-based guard for
these nodes, not a direct query of the driver's clock-lock setting.
`RemainAfterExit=yes` can leave the cap service active after the lock is
lost, so service status alone is insufficient (issue #14).

Any failed or incomplete check returns exit code 3 and lists the affected
hosts with owner recovery instructions. `start` stops before launching a
rank; it never restores clocks automatically. `status` appends
`cap <service-state> <max-sm>MHz` to each existing node row and still prints
API health. Use `caps` for the pass/fail exit status.

`CAP_MHZ` overrides the local threshold (default 2000). The owner's
`SKIP_CAP_CHECK=1` bypasses only the automatic start gate, not `caps`.
Neither variable is forwarded to node services. Automation workers must
not use the override to bypass an observed failure.

Existing `start`/`stop` shared-memory cleanup uses glob deletion; issue #20
tracks bringing that cleanup into compliance with the worker safety rules.
Workers must not execute those paths while that conflict remains.

## KV offload host tier (issue #2)

With `KVOFF_GIB` set, the head keeps a `/dev/shm` region in front of the
filesystem tier. Its rows are sized per KV cache group (slabs): a block of
the ratio-2 MLA group (g15) takes a 139,264 B row, an SWA block 114,688 B,
and a block of g13, g14 or g17 77,824 B. At 3 GiB the boot log reports
`KV offload CPU tier slabs: 139264 B x 8657 slots ..., 114688 B x 5275
slots ..., 77824 B x 18125 slots ...` (32,057 slots; the previous uniform
layout held 23,130). The split follows the bytes one 512K session leaves in
each slab, so every slab holds about 2.1 such sessions.

- `KVOFF_SLABS=0` restores the uniform layout without a code change.
- `KVOFF_SLAB_SHARES=139264:0.4,77824:0.4,114688:0.2` overrides the split;
  read `vllm:kv_offload_cpu_slab_usage_perc{slab_bytes=...}` on `/metrics`
  after a long session to see how full each slab is before changing it.
- A slab evicts only its own rows; a store batch is refused only when one of
  its slabs has nothing evictable. The filesystem tier and its `config.json`
  are unchanged by the slab layout; a new namespace directory under
  `KVFS_DIR` after a boot means the store identity changed and is a bug.

### Promotion from the filesystem tier (issue #25)

A promotion (fs → CPU) used to reacquire the GIL twice per block while
the scheduler thread re-scanned the waiting request's 11.6K keys every
step, so a 493K restore's 11,584-block job took 19 s and two concurrent
restores 46 s to the first token. `batch_verify_crc32` in `fs_io_C` now
checks every block's recorded CRC32 under one GIL release, and a load job
is split across the 16 read threads. Measured 2026-09-12 (TP=4, adopted
configuration, KV_TRACE=1 for the job lines):

- P13 493K cold restore, three boots: promotion job 0.321 / 0.322 / 0.325 s
  (before 19.2 s), TTFT 7.31 / 6.98 / 7.22 s (before 23.5 s); hit 492,928,
  answer unchanged.
- Two concurrent 493K cold restores (P13 + S2): first token 10.5 / 11.3 s
  (before 46.9 / 46.2 s), both correct, earlyoom 0.
- bench2 C1 52.2 / 47.1, C4 34.0 / 121.0 tok/s (within 3% of i15-c1b and
  i15); greedy code, count and ko-food identical to i2f, ko-busan differs
  run to run on the same boot (execution variance, also seen in #2 and #15).
- 493K cold prefill (S4) on a fresh boot after the 82K warm-up: 392 s,
  head MemAvailable floor 4.82 GiB, earlyoom 0; boots 144–159 s.

The node checkouts are installed with `VLLM_USE_PRECOMPILED=1`, which takes
the wheel's `fs_io_C.abi3.so` and never compiles `csrc/`. After a change to
`csrc/fs_io.cpp`, rebuild on each node with the server stopped:

```bash
cd ~/vllm-dsv41 && PYTHON=~/vllm-dsv41-venv/bin/python csrc/build_fs_io.sh
~/vllm-dsv41-venv/bin/python -c "import vllm.fs_io_C as m; print(hasattr(m, 'batch_verify_crc32'))"
```

An extension without `batch_verify_crc32` falls back to the per-block
Python verifier and logs a warning at boot.

## Remote frontend (issue #24 spike, off by default)

`FRONTEND_HOST=gx10-f323 FRONTEND_ADDR=10.100.0.32 dsv41_ctl.sh start` runs
the API server on f323 as the unit `dsv41-frontend` (`serve-frontend.sh`:
`--data-parallel-size-local 0`, no weights, no GPU) and boots rank 0 with
`--headless`, which dials `tcp://FRONTEND_ADDR:DP_RPC_PORT` (29560) for the
engine handshake. Order: workers, frontend, head. `status` and `log frontend`
follow `FRONTEND_HOST`; `stop` always stops the frontend unit too; `frontend`
restarts only that unit. With the knobs empty every command line is unchanged.

Two things the launcher has to do that the flags alone do not (measured
2026-09-12): with DP=1 the head takes its master IP from `VLLM_DP_MASTER_IP`,
not `--data-parallel-address` (`vllm/config/parallel.py`, non-DP branch), and
the frontend needs the engine's exact `--kv-transfer-config` because its stats
loggers build their metric definitions from it.

Measured 2026-09-12 (same sequence on a fresh boot each): 8K TTFT 4.55 vs
4.54 s, 32K 19.46 vs 19.52 s, 8K needle correct. Head memory saving is zero:
the headless head keeps the `vllm serve --headless` parent process at the same
PSS as the API server (1,011 vs 1,015 MB); head MemAvailable 5.67 vs 5.70 GiB
after boot, 3.96 vs 3.95 GiB after the probes. f323 pays 0.83 GiB for the
frontend. A frontend-only restart never comes back (the new frontend waits for
an engine HELLO that the running engine never repeats), so a frontend restart
is an engine restart. While the frontend runs on f323, f323 is also a node
with a live server for the torch-process rule. Details:
`.notes/2026-09-12-issue-24-frontend-split/results.md`.

## Operating rules

- Clocks and persistence mode belong to the owner. Automation workers must
  not execute `nvidia-smi -pm`, `-lgc`, `-rgc`, or
  `systemctl stop|restart gpu-clock-cap.service`. Issue #14 traced lost
  locks on two nodes to a persistence-mode toggle; the owner's decision
  is to keep all four nodes capped. The service's `ExecStop` releases the
  lock, so stopping it is also prohibited.
- If recovery is needed, report the failing hosts. The owner can use
  `clock_ctl.sh cap` to reapply the service and record the action in the
  issue. That helper restarts the service and is not a worker workaround
  for the prohibition. Workers must not deliberately release caps to test
  the guard; use fixtures instead.
- Do not run pytest or processes importing torch (including vLLM probes)
  on a node with a live server. The earlyoom threshold is 2.43 GiB.
- Transfer node code with `git format-patch | git am`; do not replace files
  during boot. These workstation script changes require no node deployment.
- Keep cluster operations in small foreground steps, monitor at 30–60 second
  intervals, and stop on anomalies. Do not chain background jobs. Use
  `mktemp -d` for fresh artifacts; do not use variable or glob paths with
  `rm`. Inspect actual targets before accepting a safety prompt.
- After experiments, restore port 8889 to the adopted `dsv41.env` defaults,
  including `ENGRAM_PREFETCH=1`, `EMPTY_CACHE=1`, and
  `EMPTY_CACHE_MIN_TOKENS=65536`. End with health 200, all four cap services
  active at 1989 MHz, and head `MemAvailable` at least 4.5 GiB. Read
  `/proc/meminfo` for that threshold; `status` rounds memory to whole GiB.

## Local selftest

```bash
/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/selftest_caps.sh
```

`DSV41_CAP_FIXTURE=<file>` replaces cap SSH queries with rows of
`host service-state persistence max-sm`. This is only for local tests;
leave it unset for real checks. The selftest checks healthy and failed
caps, invalid inputs, thresholds, start blocking and order, the owner
bypass, status output, and sampled SSH responses. Its SSH stub records
commands without executing them. Temporary fixtures and logs are retained
in the printed directory; no cleanup deletion is performed.

## Proposals for the owner (not implemented)

The owner could add a `gpu-clock-cap-check.timer` (for example every five
minutes) that samples clocks and restarts the cap service if the maximum
exceeds 2000 MHz. This would require an explicit owner decision about
unattended recovery and the same observation-based detection limitations.

Alternatively, the owner could add
`ExecStartPre=/usr/bin/nvidia-smi -pm 1` to the cap unit and make cap
reapplication mandatory after any persistence-mode toggle. The pre-start
command alone cannot detect a later lost lock. Only the owner changes
these unit files; neither proposal is implemented here.
