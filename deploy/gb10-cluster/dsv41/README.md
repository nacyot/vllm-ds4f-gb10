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
| `dsv41_ctl.sh` | Workstation start, stop, install, shm inspection, status, caps, and log commands. |
| `dsv41_lib.sh` | Node table, cap check and shm cleanup shared by `dsv41_ctl.sh` and the node scripts. |
| `dsv41_unit.sh`, `systemd/dsv41-serve.service` | The persistent `dsv41-serve` user unit on every node and its pre-start, start and post-start steps. |
| `dsv41_watchdog.sh`, `systemd/dsv41-watchdog.{service,timer}` | Head watchdog, one check per minute. |
| `dsv41_warmup.py`, `systemd/dsv41-warmup.service` | Post-start warm-up of the head API (three requests). |
| `serve-node.sh` | Launch one TP rank (the `dsv41-serve` unit's main process). |
| `serve-frontend.sh`, `kv_transfer_json.sh` | API server without an engine on `FRONTEND_HOST` (issue #24 spike, off by default); the shared `--kv-transfer-config` builder. |
| `dsv41.env` | Adopted server defaults, overridable through the environment. |
| `clock_ctl.sh` | Clock observations and sampling; `cap` is owner-only recovery. |
| `selftest_caps.sh` | Local cap, headroom, shm, unit pre-start and watchdog tests with fixtures and stubs, without node access. |
| `bench.py` | Before/after latency, throughput, and greedy-output checks. |
| `bench2.py`, `bench_prompts_v1.json` | Category and concurrency benchmarks and prompts. |
| `prefill_probe.py` | Prefill throughput measurements with unique prompts. |
| `divergence.py` | Record and compare greedy completions and log probabilities. |
| `smoke.py` | Basic API and deterministic-output smoke checks. |
| `kvoff_probe.py`, `kvoff_concurrent.py` | Cold and concurrent KV offload restore probes. |
| `kvfs_gc.sh`, `systemd/kvfs-gc.{service,timer}` | KV filesystem retention and size management; the head user timer that runs it. |
| `memlog.py`, `memtrace_summary.py` | Memory sampling (meminfo, worker/EngineCore/API anon and swap, reclaim and compaction counters, free order≥9 blocks; no torch) and allocator-log summaries. |
| `clock_summary.py` | Summarize sampled clock CSVs. |
| `prof_summary.py` | Summarize GPU profiler traces. |

## Startup and cap checks

`dsv41_ctl.sh start` checks all four caps before any rank restarts. After the
check passes, it writes the knobs set in its environment to the head's
`~/dsv41-prep/dsv41-override.env` and restarts the head's `dsv41-serve` unit.
That unit's pre-start restarts workers rank 3 (27c4), 2 (37cc), 1 (f323) and
then head rank 0 (6040) boots (see
[Persistent unit, watchdog and warm-up](#persistent-unit-watchdog-and-warm-up-issue-32)).
Each rank uses `serve-node.sh` and the defaults in `dsv41.env`. The API is on head port 8888,
the production endpoint since 2026-09-13 (it replaced the DS4F TP=4 service
that used the same port; bring-up ran on 8889). The server answers to the
model names `deepseek-v4.1-flash` and, for the old DS4F clients,
`deepseek-v4-flash-0731` and `deepseek-v4-flash-vision` (`SERVED_ALIASES` in
`dsv41.env`).

Production settings since 2026-09-13: the vision tower is loaded
(`TEXT_ONLY=0`; image inputs accepted, up to `MM_IMAGES=512` images per
prompt and 1,024 tokens per image, encoder attention on FLASH_ATTN, boot
160 s, head MemAvailable 4.9 GiB after boot) and the filesystem KV tier lives
on the external 3.6 TB USB SSD at `/mnt/kvdisk/kv/dsv41` (`KVFS_DIR`), the
disk the DS4F production used, with retention by `systemd/kvfs-gc.timer`
(every 30 minutes: files older than 7 days, then oldest-first above 3000 GiB).
The bring-up store `~/dsv41-prep/kvfs` on the root NVMe was copied there with
`rsync -aX` (xattr checksums preserved); a store moved to a new root keeps its
identity because the base path is `<root>/<model>_<hash of the run config>`.

Speculative decoding verifies drafts with block verification
(`SPEC_REJECT=block`, Sun et al. 2024) on probabilistic drafts
(`SPEC_DRAFT=probabilistic`) since 2026-09-13. Block verification leaves the
sampling distribution unchanged and applies only to sampled requests; agent
clients such as pi send no temperature, so the server samples at 1.0. On the
pi-like 4-agent benchmark (`casebench.py --mode agent`, thinking high) the
same work took 232 s instead of 247 s; greedy requests are unchanged.

Since 2026-09-15 (issue #36) the trailing prefix-cache block is kept
(`SPEC_BLOCK_DROP=0`, `disable_eagle_block_drop`) and the engram page
release is left to the kernel (`ENGRAM_RELEASE=0`). Measured together on a
fresh boot against the 2026-09-13 defaults: 4-agent total 235.6 → 208.0 s,
cold prefill 32K 1,459 → 1,918 and 128K 1,363 → 1,464 tok/s, 158K
follow-up turn TTFT 3.13 → 0.77 s, code acceptance length unchanged (5.82),
head `MemAvailable` floor 3.67 GiB during the 128K prefill. `MOE_BACKEND=b12x`
(+12% code decode alone) stays off: with it the same floor was 2.66 GiB.
Details in `.notes/2026-09-15-issue-36-knob-combo/results.md`.

Since 2026-09-15 (issue #37) a decode-only step waits for the pages of the
first engram table only and populates the second one in the background
(`ENGRAM_DECODE_ASYNC=1`, with `ENGRAM_CHUNK_RUNS=8`): the pages of a decode
step are cold NVMe reads (the two 23.6 GiB tables per rank never fit the page
cache), and waiting on both tables serially idled the GPU 5.6 ms per step.
Code c1 step 70.6 → 68.2 ms, prose 70.5 → 67.8 ms, acceptance length
unchanged. `ENGRAM_DECODE_ASYNC=2` (both tables in the background) reaches
64.6 ms on code but prose pays the GPU's in-place page faults instead.
Details in `.notes/2026-09-15-issue-37-engram-prefault/results.md`.

Since 2026-09-15 (issue #38) the indexer of a prefill chunk scores only a
quarter of the chunk's rows on each TP rank and one all-gather per indexer
layer rebuilds the top-k (`DSV41_INDEXER_TP_SPLIT=1`, from the Tech2Wild
speedrun). The logits, the candidate blocks and every row's top-k set are
identical to the unsplit loop on the GPU. Cold prefill 128K 1,666 → 1,836
and 32K 1,866 → 1,978 tok/s, 158K needle cold prefill 2,034 tok/s,
acceptance length unchanged. The four ranks compare the setting at model
build and refuse to start when it differs.
Details in `.notes/2026-09-15-issue-38-indexer-tp-split/results.md`.

The scheduler caps `LPTT`, `LPTT_MIXED`, `PPCAP` and `DECODE_STEPS` stay
unset by default. On that benchmark none of them shortened the total time.
The DS4F mixed cap of 2,048 tokens cost 33% of prefill throughput here,
because DeepSeek-V4.1 routes to 6 of 384 experts and a 2,048-token chunk
leaves each expert about 32 tokens. With a 128K cold prefill injected into
the 4-agent load, the default finished everything in 329 s. An 8,192 budget
with a 6,144 chunk cap and 4 decode-only steps per prefill took 364 s. It cut
a short request's wait behind a 128K prefill from about 87 s to 7 s by
slowing that prefill to 878 tok/s. An 8,192 budget was the upper bound: at
12,288 the boot-time FlashInfer autotune ran every worker into earlyoom.

`SHORT_RESERVE` is the exception: it is on, at 4,096. Those caps all take a
fixed amount away from every prefill chunk, which is why they cost total
time. This one takes only what is actually queued behind the prefill --
the decodes already in the running batch and the waiting requests whose new
tokens fit under the value -- capped at half the step budget, and it takes
it from the token budget and the input budget both, since every scheduled
request also costs the drafter's extra slots. With nothing waiting it
reserves nothing, so a solo prefill keeps its whole chunk and the step still
schedules `MNBT` tokens, leaving the per-expert chunk size alone. Adopted
2026-09-15 (issue #34): a 16-token request behind a 128K cold prefill waits
7.0 and 8.7 s instead of 64.5 and 68.0, the 4-agent run with that prefill
injected is unchanged at 265 s, the 4-agent run alone went 205.6 to 199.3 s,
and a solo 128K prefill 1,666 to 1,792 tok/s. A waiting request cannot
report its prefix hit before admission, so the cutoff sees the whole prompt:
a resumed long session's new turn is not treated as short even when it only
recomputes a few thousand tokens. Details in
`.notes/2026-09-15-issue-34-hol-reserve/results.md`.

`dsv41_ctl.sh caps` prints `HOST SERVICE PERSISTENCE MAX_SM_MHZ`. Each node
must have an active `gpu-clock-cap.service`, persistence `Enabled`, and a
maximum SM clock of at most 2000 MHz across five samples, 200 ms apart.
Normal idle readings are 1989 MHz. This is an observation-based guard for
these nodes, not a direct query of the driver's clock-lock setting.
`RemainAfterExit=yes` can leave the cap service active after the lock is
lost, so service status alone is insufficient (issue #14).

Any failed or incomplete check returns exit code 3 and lists the affected
hosts with owner recovery instructions. `start` stops before restarting a
rank; it never restores clocks automatically. The head unit's pre-start
repeats the same check before every start of the server, including automatic
restarts, watchdog restarts and boots (12 tries, 10 s apart, the head probed
locally); when it still fails the unit fails, logs to `user.err` and changes
no cap. `status` appends `cap <service-state> <max-sm>MHz` to each existing
node row and still prints API health. Use `caps` for the pass/fail exit status.

`CAP_MHZ` overrides the local threshold (default 2000). The owner's
`SKIP_CAP_CHECK=1` bypasses only the workstation start gate, not `caps` and
not the unit pre-start. Neither variable is written to the override file.
Automation workers must not use the override to bypass an observed failure.

`stop` and every unit pre-start first list shared-memory candidates with their
sizes, then delete only explicitly listed, unused files owned by the remote user
directly under `/dev/shm`: `sem.mp-*`, `psm_*`, and `vllm_offload_*.mmap`.
Symlinks and changed files are rejected. Active or transitioning `dsv41-serve`
services protect all candidates, except that a unit's own pre-start cleans
while that unit is activating (the same `InvocationID`); files still used by a
process are skipped. Inspection failures prevent deletion and are reported
(`stop` exits 1, a pre-start fails the unit). Nodes need GNU Bash 4.4+ and `fuser`.

Use `dsv41_ctl.sh shm [host]` to inspect all four nodes or one named cluster
node, including while serving. It lists candidates and whether they are in use
without stopping services or deleting files. `DSV41_SHM_DRYRUN=1` makes stop and
start cleanup read-only too: **dry-run stop still stops the services**, but leaves
cleanup candidates in place. For `start` it is written to the override file, so
automatic restarts stay read-only until the next `start` without it. Stop retains
its six-second wait before the final process termination and cleanup. The KV
filesystem store is never a candidate.

## Persistent unit, watchdog and warm-up (issue #32)

Since 2026-09-15 `dsv41-serve` is a persistent user unit
(`systemd/dsv41-serve.service`, the same file on the four nodes; the rank
comes from the host name) instead of a transient `systemd-run` unit, so the
server comes back after a reboot or a dead process. Linger is enabled on all
four nodes. After `git am` on the nodes, put the units in place with
`dsv41_ctl.sh install` (copies them to `~/.config/systemd/user/`, reloads and
enables `dsv41-serve` everywhere, and the watchdog timer on the head; it
starts no server and refuses while a transient `dsv41-serve` is still loaded,
so `stop` first when switching). Run `install` again after changing a unit file.

- The unit restarts always (`RestartSec=20`, at most 5 starts in 30 minutes),
  since the API server can exit 0 after `EngineDeadError`; `systemctl stop`
  does not restart it. `TimeoutStopSec=60` with `KillMode=mixed`: a graceful
  stop next to a dead NCCL peer only ends at the timeout. Logs stay in
  `~/dsv41-prep/logs/dsv41-r<rank>.log`, pre-start lines included.
- Pre-start on every rank (`dsv41_unit.sh pre`): wait for MemAvailable ≥
  100 GiB (up to 2 minutes), clean shm candidates, drop caches. The head
  first waits for `/mnt/kvdisk` (up to 3 minutes) when `KVFS_DIR` is on it,
  checks the caps, copies its override file to the three workers and
  restarts their units (27c4, 37cc, f323; worker ssh waited up to 8 minutes
  in total). Restarting the head unit therefore restarts the whole server,
  and a head boot or restart never runs next to stale workers. A worker unit
  that restarts alone (its own crash, a worker reboot) leaves the TP group
  broken until the watchdog restarts the head. Workers do not need
  `/mnt/kvdisk`: only rank 0 creates and uses the filesystem tier, and a
  worker's NFS mount of it is not restored at boot.
- Knobs: `start` rewrites `~/dsv41-prep/dsv41-override.env` on the head
  (`KEY="value"` lines; empty means the `dsv41.env` defaults) and the head
  pre-start copies it to the workers, so all ranks boot with the same knobs.
  Quotes and spaces in values now survive. The file persists across restarts
  and reboots: end every experiment with a `start` without knobs.
- Watchdog (`dsv41-watchdog.timer` on the head, every minute, log
  `~/dsv41-prep/logs/dsv41-watchdog.log`). It skips the first 600 s after the
  head unit became active. It restarts the head unit when a worker unit is not
  active or started more than 30 s after the head (no cooldown, 12 restarts in
  2 h), or when a worker is unreachable 3 checks in a row, `/health` fails 3
  checks in a row, or a 1-token probe fails twice in a row (15-minute
  cooldown, 4 restarts in 2 h). The probe runs every fifth check, and every
  check while it fails, because `/health` stays 200 while the engine waits
  on a dead rank in NCCL. It never starts an inactive unit (an operator
  stop). A failed unit, a start still activating after 30 minutes and a
  used-up budget go to `logger -p user.err` and
  `~/dsv41-prep/logs/dsv41-ATTENTION` (the latest reason; checked and removed
  by a person). Probes count in `/metrics`: pause the watchdog for such
  benchmarks with `systemctl --user stop dsv41-watchdog.timer` on the head
  and `start` it afterwards.
- Warm-up (`dsv41-warmup.service`, queued by the head's post-start on every
  start): once `/health` is 200 it sends an ~8K greedy request, a short
  sampled request with thinking and one image, each with a fresh salt, and
  logs one JSON line per request to `~/dsv41-prep/logs/dsv41-warmup.log`.
  The first request after a boot took 21 s TTFT before (JIT kernels and the
  FlashInfer autotune).

Units of the replaced DS4F production. They are disabled, not deleted, and
their files, `~/migration-027` and `~/ds4f-logs` are kept:

| Node | Unit | State | Why |
| --- | --- | --- | --- |
| 37cc, 27c4 | `tp4-worker.service` (user) | disabled and stopped (#32) | DS4F TP=4 worker; at boot it took the memory the V4.1F rank needs |
| 37cc | `kv-prune.timer` (user) | disabled and stopped (#32) | Pruned a DS4F directory on `/mnt/kvdisk` |
| 6040 | `ds4f-log-capture.service` (system) | disabled and stopped (#32) | Captured docker logs of a container that no longer exists (2.2 MB/day of errors) |
| 6040 | `tp4-head.service`, `ds4f-head.service`, `ds4f-watchdog.timer`, `serve-warmup.service` (static), `kv-prune.timer`, `kv-snapshot-ttl.timer` (user) | already disabled | DS4F head, watchdog, warm-up and KV retention |
| 27c4 | `ds4f-worker.service` (user) | already disabled | DS4F pair worker |

The owner's `uvm-stall-sentinel.timer` stays enabled on all four nodes. Every
minute, when memory PSI full avg10 is at least 50 and the Normal zone has no
free order≥9 block, it runs `drop_caches` and `compact_memory`. It ran 32
times on the head and 19 times on 37cc in the 7 days to 2026-09-15. Its
behaviour is unchanged; the decision belongs to issue #31.

## KV offload host tier (issue #2)

With `KVOFF_GIB` set, the head keeps a `/dev/shm` region in front of the
filesystem tier. Its rows are sized per KV cache group (slabs): a block of
the ratio-2 MLA group (g15) takes a 139,264 B row, an SWA block 114,688 B,
and a block of g13, g14 or g17 77,824 B. At 3 GiB the boot log reports
`KV offload CPU tier slabs: 139264 B x 8657 slots ..., 114688 B x 5275
slots ..., 77824 B x 18125 slots ...` (32,057 slots; the previous uniform
layout held 23,130). The split follows the bytes one 512K session leaves in
each slab, so every slab holds about 2.1 such sessions.

The default is 2 GiB since 2026-09-13 (about 1.4 max-length sessions). At
3 GiB the head's MemAvailable reached earlyoom's 2.43 GiB line twice: under
a sustained 4-agent load (13:06, the worker was killed) and during a fresh
boot's first 16K prefill (2,619 MiB). The drop is GPU allocator growth in
unified memory, outside every process RSS, so nothing reclaimable covers
it. 2 GiB returns about 1 GiB to the head (warm-up low 3,914 MiB, loaded
lows 3.1 to 4.9 GiB). Two concurrent max-length restores have less
promotion room than at 3 GiB.

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
- After experiments, restore port 8888 to the adopted `dsv41.env` defaults,
  including `ENGRAM_PREFETCH=1`, `ENGRAM_RELEASE=0`, `ENGRAM_DECODE_ASYNC=1`,
  `ENGRAM_CHUNK_RUNS=8`, `SPEC_BLOCK_DROP=0`, `DSV41_INDEXER_TP_SPLIT=1`,
  `EMPTY_CACHE=1`, and `EMPTY_CACHE_MIN_TOKENS=65536`: run `dsv41_ctl.sh start`
  with no knobs, so `~/dsv41-prep/dsv41-override.env` is empty on all four
  nodes and `systemctl --user show dsv41-serve -p Environment` is empty. End
  with health 200, all four cap services active at 1989 MHz, the head's
  `dsv41-watchdog.timer` active with no `dsv41-ATTENTION` file, and record head
  `MemAvailable`. After short probes the
  head sits at about 3.4–4.0 GiB under `EMPTY_CACHE_MIN_TOKENS=65536`; that is
  expected. Any long cold prefill must pass `dsv41_ctl.sh headroom` (≥5.2 GiB)
  first, which in practice means a fresh boot. `status` reports
  memory from `/proc/meminfo` to two decimal places.

### Metrics in Grafana (issue #33)

VictoriaMetrics on CT116 scrapes port 8888 directly (`job="vllm"`,
`service="ds4f"`, 15 s) for the homelab `ds4f-vllm` dashboard; the head's
`~/vmagent/scrape.yml` is unused. Read computed prefill from
`vllm:prompt_tokens_by_source_total{source="local_compute"}` and disk/CPU
restores from `source="external_kv_transfer"`. vLLM books prompt tokens when
the first token appears, so a long prefill shows as a sawtooth or one spike;
use the per-request panel (`request_prefill_kv_computed_tokens_sum` over
`request_prefill_time_seconds_sum`, 5 minutes) for speed.
`iteration_tokens_total_count` counts only steps that emit output, so the
stall signature also requires no `kv_cache_usage_perc` change for 2 minutes.

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

Why fresh-boot headroom varies (issue #30, measured 2026-09-13): the three
head processes do not grow. Their cold anonymous memory is the same on every
boot (Worker_TP0 RssAnon+VmSwap 2.84 GiB, EngineCore 0.8 GiB, API server
0.9 GiB; SUnreclaim is 2.1–2.2 GiB on all four nodes). What differs is
whether that cold anon was swapped out. A boot consumes the free order≥9
(2 MiB) blocks of the Normal zone (14,957 before the load, 44 at health 200)
and does no reclaim itself. The first cold long prefill (the 82K warm-up)
then regrows the torch allocator by 2.9 GiB in 2 MiB chunks, hits order≥9 =
0 within seconds, and forces direct compaction and kswapd reclaim (measured
warm-up: compact_stall +1,183, pgscan_kswapd +516k pages, MemFree minimum
1.5 GiB, MemAvailable minimum 3.0 GiB, TTFT 51–63 s). The kernel either
reclaims file cache and leaves the anon resident (this boot and #19: swap 0,
idle 4.6–5.2 GiB afterwards) or swaps the cold anon out and drops the file
cache (#25 b5 and #27 b2: 1.7–2.5 GiB swapped out in the last 15 s of the
warm-up, idle 7.1–7.8 GiB afterwards). Which one happens is the kernel's
anon/file cost balance at that moment (swappiness 60, prior page-cache
churn), not a setting of ours, so the idle value after a fresh boot cannot
be predicted. The head free pool stays fragmented while the server runs
(order≥9 blocks 5–55; the owner's `uvm-stall-sentinel` compacts only when PSI
full ≥ 50). Owner levers are proposed in the #30 decision issue; nothing was
changed. `memlog.py` records the counters that tell the two cases apart
(`pswpout`, `compact_stall`, `free_order9plus`, `e_/a_VmSwap`).

Head processes on the 2026-09-13 boot (MiB; `VmSwap` stayed 0 at every
point, `MemorySwapPeak=0`):

| Process | After boot RssAnon | After warm-up RssAnon | After 8K RssAnon | VmSwap |
| --- | ---: | ---: | ---: | ---: |
| VLLM::Worker_TP0 | 2,819 | 2,837 | 2,838 | 0 |
| VLLM::EngineCore | 805 | 831 | 830 | 0 |
| API server (`vllm serve`) | 865 | 918 | 918 | 0 |

For comparison, at the #25 S4 start (7.11 GiB idle) the worker held RssAnon
2,172 + VmSwap 665 = 2,837, the same total.

| Boot (KST) | Idle after boot | Warm-up minimum | Swapped out in warm-up | Idle after warm-up |
| --- | ---: | ---: | ---: | ---: |
| #25 b4, 09-12 19:22 | 6.19 | 3.34 | 0 | 5.43 → 4.7 |
| #25 b5, 09-12 19:30 | 6.37 | 3.88 then swap | 1.75 GiB | 7.11 |
| #27 b2, 09-12 21:48 | 6.92 | 4.28 then swap | 2.50 GiB | 7.3–7.8 |
| #19, 09-12 22:58 | 5.11 | 2.99 | 0 | 4.72 |
| #30, 09-13 00:37 | 4.97 | 2.99 | 0 (3 MiB) | 5.21 |

Source: head node-metrics-exporter series in VictoriaMetrics (15 s), sar,
`journalctl --user -u dsv41-serve`, `dsv41-r0.log`, and the #30 1-second
memlog `~/dsv41-prep/bench/i30-boot-memlog.csv`. Use a **new salt** for the
warm-up as well: a salt already in the kvfs store restores instead of
prefilling (W30 restored 750 MB in 1.7 s on 2026-09-13).

| Run | Server state | Start GiB | Floor GiB | Drop GiB | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| #25 S4 | Fresh boot + 82K warm-up | 6.95 | 4.82 | 2.13 | Completed, 392 s |
| #15 P19 | Warm, after restore | 4.9 | 2.76 | 2.14 | Completed, 387 s |
| #15 P20 | Warm, restore + 32K | 3.05 | 2.54 at 60 s | ≥0.51 | Client stopped; server survived |
| #27 | Warm, after concurrent restore | 4.48–4.52 | 2.92 at 30 s, then <2.43 | >2.05 | earlyoom killed EngineCore |
| #19, 2026-09-12 23:01 KST | Fresh adopted boot + 82K warm-up | 4.61 | Not run | Not measured | Guard exit 3; 493K not started. Idle headroom −2.5 GiB vs #25 S4 7.11 GiB; swap usage difference 1.5 GiB, AnonPages +1.7 GiB |
| #30, 2026-09-13 00:41 KST | Fresh adopted boot + cold 82K warm-up | 5.11 | Not run | Expected 2.9–3.0 | Guard exit 3; 493K not started. Warm-up reclaimed file cache, not anon (swap 0), so the 7.1 GiB state did not occur; expected floor = start − 2.1–2.2 |

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
missing head readings and transport failure. It also runs `dsv41_unit.sh`
and `dsv41_watchdog.sh` with recording stubs for `systemctl`, `curl`,
`logger`, `sudo` and `mountpoint`: the start flow and override quoting,
`install`, the head pre-start's worker order and its failure paths (kvdisk,
caps, unreachable worker, memory), the shm `pre` mode, and every watchdog
rule (inactive, failed, grace, worker down or late, health, probe,
unreachable, cooldown, budget). Python tests use fake responses
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
