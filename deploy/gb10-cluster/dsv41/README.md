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
| `dsv41_ctl.sh` | Workstation start, stop, shm inspection, status, caps, and log commands. |
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

`start`/`stop` first list shared-memory candidates with their sizes, then delete
only explicitly listed, unused files owned by the remote user directly under
`/dev/shm`: `sem.mp-*`, `psm_*`, and `vllm_offload_*.mmap`. Symlinks and changed
files are rejected. Active or transitioning `dsv41-serve` services protect all
candidates; files still used by a process are skipped. Inspection failures
prevent deletion and are reported. Nodes need GNU Bash 4.4+ and `fuser`.

Use `dsv41_ctl.sh shm [host]` to inspect all four nodes or one named cluster
node, including while serving. It lists candidates and whether they are in use
without stopping services or deleting files. `DSV41_SHM_DRYRUN=1` makes start/stop
cleanup read-only too: **dry-run stop still stops the services**, but leaves
cleanup candidates in place. Stop retains its six-second wait before the final
process termination and cleanup. The KV filesystem store is never a candidate.

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

### Skipping the waiting request's re-scan (issue #27)

While a promotion is in flight the request is deferred, and the core
scheduler re-queries it every step. The connector used to re-scan all of
the request's full-attention keys (11.6K for a 493K session) through
`manager.lookup` each of those steps, on the EngineCore thread, so every
other request decoding at the time paid for it. `CPUOffloadingManager`
now keeps a state epoch that advances only when a lookup result can
change (a write completing or failing, an eviction, a cache reset). The
connector records the epoch when a lookup defers on HIT_PENDING alone and
returns None without re-scanning while the epoch and the local
computed-token count hold; `touch` and the hit-chunk update still run, so
LRU order is unchanged, and a RETRY disables the shortcut. Measured
2026-09-12 (TP=4, adopted configuration):

- Per 493K cold restore, lookups in the 10-50 ms band (the per-step
  HIT_PENDING re-scans) fall from 5 to 1; two concurrent restores 3 to 2.
  The remaining longer lookups (first scan, promotion start, final HIT)
  are one-per-request and out of scope.
- TTFT (7-8 s), promotion job (0.34 s), hit (492,928 single / 985,856
  concurrent) and the answer are unchanged; bench2 C1/C4 within 3% of the
  issue #25 run, greedy code, count and ko-food identical.
- The large decode stalls during a restore (~1.7 s) are the restoring
  request's own prefill and 493K decode steps, not the lookups, so this
  removes per-step Python load on concurrent decoders rather than the
  wall-clock hiccup.

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
  during boot. Run the probe scripts from `~/vllm-dsv41/deploy/gb10-cluster/dsv41/`
  on the head, using `.venv/bin/python` from that repository; do not use
  old script copies in `~/dsv41-prep`. Python/Bash changes need no `.so` rebuild.
  Matching script entries in `~/dsv41-prep/` are symlinks to `~/vllm-dsv41/deploy/gb10-cluster/dsv41/` since 2026-09-12 (#28).
- Keep cluster operations in small foreground steps, monitor at 30–60 second
  intervals, and stop on anomalies. Do not chain background jobs. Use
  `mktemp -d` for fresh artifacts; do not use variable or glob paths with
  `rm`. The start/stop cleanup function alone may delete validated, explicitly
  listed shared-memory files; workers must not run ad hoc `rm` commands.
  Inspect actual targets before accepting a safety prompt.
- After experiments, restore port 8889 to the adopted `dsv41.env` defaults,
  including `ENGRAM_PREFETCH=1`, `EMPTY_CACHE=1`, and
  `EMPTY_CACHE_MIN_TOKENS=65536`. End with health 200, all four cap services
  active at 1989 MHz, and record head `MemAvailable`. After short probes the
  head sits at about 3.4–4.0 GiB under `EMPTY_CACHE_MIN_TOKENS=65536`; that is
  expected. Any long cold prefill must pass `dsv41_ctl.sh headroom` (≥5.2 GiB)
  first, which in practice means a fresh boot. `status` reports
  memory from `/proc/meminfo` to two decimal places.

### Long cold prefill headroom

Long cold prefill (operationally, ≥256K tokens) starts only with head
`MemAvailable` ≥5.2 GiB. Run `dsv41_ctl.sh headroom` first: it prints all four
nodes and returns exit 3 if the head is below the threshold or its reading
is missing/invalid. `headroom [min_gib]` overrides `MIN_AVAIL_GIB` (default 5.2).
This is a prefill check, not a server-start gate. Leave test fixtures unset
when checking the cluster.

The probes enforce the check in `kvoff_probe.run_prompt`, shared by
`kvoff_concurrent.py` and `prefill_probe.py`. The exact trigger is ≥18000
records (roughly 269K tokens with a three-character salt), not a tokenizer
count. The CLI refuses inference with a `skipped: "headroom"` JSON line and
exit 3. During streaming, one shared monitor samples every 2 seconds and
interrupts active long requests below 2.8 GiB. Interrupted results contain
`aborted: "headroom"`, `ok: false`, and exit 3; they are not successful
throughput samples. Results include `mem_avail_start_gib` and
`mem_avail_min_gib` (null when not monitored).

Probe settings are `DSV41_MIN_AVAIL_GIB=5.2`, `DSV41_ABORT_BELOW_GIB=2.8`,
and `DSV41_LONG_PROMPT_RECORDS=18000`. Zero disables the corresponding
threshold; zero records disables both guards. Non-Linux/missing meminfo
or a base URL outside localhost/127.0.0.1 skips the guards with a warning:
local memory would not measure that server. Model discovery, metrics and
prefill token calibration can still run before an inference refusal.

Measured head memory drop during 493K prefill is about 2.1–2.2 GiB. The old
4.5 GiB start rule was insufficient. Record end-state memory; short probes can
leave the head at 3.4–4.0 GiB under the adopted cache-release policy.
In practice, restart the adopted configuration after a restore session
before running another long cold prefill. Fresh-boot headroom varies from
4.6 to 7.1 GiB: a restart does not guarantee sufficient memory. Always run
`headroom` before 493K cold prefill and do not start below 5.2 GiB.
Keep earlyoom at 2.43 GiB.

| Run | Server state | Start GiB | Floor GiB | Drop GiB | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| #25 S4 | Fresh boot + 82K warm-up | 6.95 | 4.82 | 2.13 | Completed, 392 s |
| #15 P19 | Warm, after restore | 4.9 | 2.76 | 2.14 | Completed, 387 s |
| #15 P20 | Warm, restore + 32K | 3.05 | 2.54 at 60 s | ≥0.51 | Client stopped; server survived |
| #27 | Warm, after concurrent restore | 4.48–4.52 | 2.92 at 30 s, then <2.43 | >2.05 | earlyoom killed EngineCore |
| #19, 2026-09-12 23:01 KST | Fresh adopted boot + 82K warm-up | 4.61 | Not run | Not measured | Guard exit 3; 493K not started. Idle headroom −2.5 GiB vs #25 S4 7.11 GiB; swap usage difference 1.5 GiB, AnonPages +1.7 GiB |

Source: issue #19 manager investigation `icmt-15ef874f`, citing the #25
1-second memlog and #15/#27 results. The #19 row and memory comparison
follow the manager decision responding to `icmt-929be183`; 7.11 GiB is the
S4 idle observation from issue #25, distinct from its 6.95 GiB prefill start above.
The #27 ≥3.0 GiB floor gate remains incomplete in this work; the manager
will track it in a follow-up issue. For that outstanding floor
gate, use a fresh adopted boot, one 82K warm-up (5600 records), then verify
headroom ≥5.2 GiB before one 493K request (33000 records, **new three-character
salt**; four characters exceed MAXLEN). Record both a 1-second memlog minimum
and the probe minimum, answer, elapsed time and earlyoom events. Check the
EMPTY_CACHE release, then one 8K probe and the normal end state. Never test
the 2.8 GiB abort deliberately on the cluster; use local tests.

## Local selftest

```bash
/opt/homebrew/bin/bash deploy/gb10-cluster/dsv41/selftest_caps.sh
.venv/bin/python -m pytest deploy/gb10-cluster/dsv41/test_headroom.py -v
```

`DSV41_CAP_FIXTURE=<file>` replaces cap SSH queries with rows of
`host service-state persistence max-sm`. This is only for local tests;
leave it unset for real checks. The selftest checks healthy and failed
caps, invalid inputs, thresholds, start blocking and order, the owner
bypass, status output, and sampled SSH responses. Its SSH stub records
commands without executing them. Temporary fixtures and logs are retained
in the printed directory; no cleanup deletion is performed.

`DSV41_MEM_FIXTURE=<file>` replaces headroom SSH queries with `host gib`
rows. The same selftest covers memory thresholds, overrides, malformed or
missing head readings and transport failure. Python tests use fake responses
and a local HTTP server to check refusal without inference, CLI exit 3,
remote bypass, and shared cancellation while waiting for the first token.
Run them only on the workstation, without importing vLLM or torch.

### FlashInfer unit tests on a node

Run node unit tests only during a coordinated server downtime window, with
the servers stopped on all four nodes. These tests import torch.
`flashinfer-python` and `flashinfer-cubin` must have matching versions in
`~/vllm-dsv41-venv`; change that venv only inside the same downtime window,
because running servers can load cubin files lazily.

If matching packages are unavailable, `FLASHINFER_DISABLE_VERSION_CHECK=1`
is a temporary, per-command bypass for the import-time version check:

```bash
cd ~/vllm-dsv41
FLASHINFER_DISABLE_VERSION_CHECK=1 ~/vllm-dsv41-venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/offloading_connector/test_worker.py \
  -k 'register_kv_caches and FLASHINFER' -v
```

The bypass does not align package versions or guarantee kernel compatibility.
Omit it once the versions match. The server launcher already sets this bypass,
so a healthy server does not prove that standalone tests can import FlashInfer.
Issue #26 records the four-node 0.7.0/0.6.18 mismatch and the unavailable
`flashinfer-cubin==0.7.0` release observed on 2026-09-12 in
`.notes/2026-09-12-issue-26-flashinfer-cubin/results.md`.

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
