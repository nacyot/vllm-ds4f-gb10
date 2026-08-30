# vllm-ds4f-gb10

> Experimental patch set on top of [vLLM](https://github.com/vllm-project/vllm) **v0.27.1**, for serving DeepSeek V4 Flash across two DGX Spark **GB10** nodes with unified memory. Specific hardware, reference implementation, not a production guarantee.

**Base:** vLLM v0.27.1. Build and install with vLLM's own instructions; this fork applies the patches below on top of that tree. License: Apache-2.0 (inherited from vLLM, see `LICENSE`).

**Author:** nacyot

Every change here is opt-in behind an env var and defaults to stock vLLM behavior. The blog below is the main content; a short knob index follows it.

(한국어: [README.ko.md](README.ko.md))

---

**Status (2026-08-30).** Four more changes landed and are live on the reference deployment (now the 4-node TP=4 layout, same knobs): the disk tier no longer grows without bound (anchor-only window snapshots plus per-chain retention and a TTL pruner), a scheduler-side reasoning loop breaker forces `</think>` and lets the request keep answering (Model Runner V2 port of vLLM PR #52677, with a server-default thinking budget), the DSpark verify batch is tiled into <=64-row FlashInfer decode calls so 16-way decode no longer routes through the prefill orchestrator, and the indexer short-context shortcut is guarded against CUDA-graph capture (vLLM #52492). Later the same day the draft sampler's Gumbel noise stream was decoupled from the target's (port of vLLM #54282, main 2026-08-29), which removes an output-distribution bias under `draft_sample_method=probabilistic`; the evening round then linted every fork-modified file through the repo's pre-commit hooks and cherry-picked four small upstream fixes (#52311, #52707, #53329, #53962). Details in the changelog.

**Status (2026-08-28).** The goal is now met and exceeded on the reference hardware: **six ~500k-token sessions resident** in an 18 GiB GPU pool with instant (2-4 s) switching and zero disk reads, cold-restart restore of any parked session in **12-18 s** (about 47x over full reprefill, needle-exact), and **concurrent cold swap-ins self-serializing in ~67 s** for five 450k sessions (previously 45 minutes of mutual staging eviction). Concurrency benchmarks (c8/c16/c32) hold their acceptance and scaling with offloading on.  A 2026-08-28 reliability round, validated on a second GB10 pair, hardened the failure paths: a boot-order race that could latch the pair down after a whole-rack reboot now waits for the peer, the native-DeepGemm logits allocator no longer ratchets reserved memory under mixed-length prefills (bounded bucket reuse; measured +1.89 GB per 15 min before, flat after), and a corrupted or externally pruned store chunk now degrades to one logged failure plus a recompute instead of an assert or a permanently parked request (fault-injection verified, needle-exact). See the changelog at the bottom for what landed.

# Keeping many long agent sessions resident: disk KV offloading for multi-node vLLM

**The goal.** Run many long agent sessions at once, and never worry about a session's KV cache expiring or about re-paying its multi-minute prefill. On two GB10 DGX Spark boxes with unified memory that means keeping five or more sessions of 400k+ tokens resident and instant to switch between, and letting any older session that has been evicted come back without the full reprefill cost. That is the target this work was built for. Everything else, including the restore-time work, is in service of it.

This is an experimental patch set on top of [vLLM](https://github.com/vllm-project/vllm) 0.27.1. It targets a specific setup (two DGX Spark GB10 nodes, unified memory, DeepSeek V4 Flash FP8, tensor parallel across the two nodes). It is a reference implementation, not a production guarantee. If you run something similar, the parts below should transfer; if you do not, read it as a case study.

Author: nacyot.

## The problem

When you self-host an LLM for long sessions, the real cost is not decode throughput. It is reprefill. A 500k-token agent session that gets evicted from the GPU cache has to be recomputed from scratch the moment you send the next turn, because a prefix-cache miss means the server walks the whole conversation again to rebuild its attention keys and values. On this hardware that reprefill runs into the multi-minute range. Waiting minutes for one reply effectively forces you to use a single session.

The bottleneck for a local agent, then, is not tokens per second. It is how many sessions you can keep alive. GPU memory is finite, so with several sessions the oldest gets pushed out, and reopening it is a full reprefill. This is the root of the whole thing: the KV that gets pushed out of memory is simply lost, nothing higher in the memory hierarchy holds it, and so the reprefill on the next touch cannot be avoided from inside memory at all. The only lever is to have written the KV somewhere durable before it was evicted. You cannot solve this inside the GPU alone.

The KV cache is a deterministic function of the tokens, so there is a clean answer: write it to disk when it is evicted, read it back when the session is touched again, and skip the recompute. A 469k-token session that costs 435 seconds to rebuild on the GPU is a few-GB file on disk, a few seconds away at 1 GB/s. vLLM 0.27 already ships a tiering path (GPU pool to a CPU buffer to backend storage). The problem was that it did not work in a two-node configuration.

### Setup, and the unified-memory catch

- Hardware: two DGX Spark boxes (GB10, 128 GB unified memory each), 200G direct link, tensor parallel across both nodes.
- Model: DeepSeek V4 Flash, FP8, with a sparse attention indexer.
- Engine: vLLM 0.27.1 installed into a venv so the source can be patched inline.
- Disk: an 8 TB cache volume mounted on both nodes over CIFS (`hard,retrans=6`), measured at 1 GB/s.
- Speculative decoding with a dspark-style drafter, roughly 1.5x on decode.

GB10 unified memory makes the trade subtle. The GPU and CPU share the same 128 GB, so "offload KV to CPU RAM" is zero-sum: every GiB you give to RAM is a GiB the GPU pool loses. The final design reflects this by treating RAM not as storage but as a staging buffer between the GPU and disk. That reframing is the backbone of the whole thing.

## What was built: packed canonical offloading

The place the upstream tiering breaks on two nodes is the memory model of the storage layer. vLLM's CPU cache tier is an mmap under `/dev/shm`, and across two nodes the same-named mmap is different physical memory on each box. When the head node writes a store file, the half that belongs to the worker rank is never filled, and on restore those zero bytes get promoted into the GPU and corrupt attention. A postmortem of the store files showed exactly half of every file was zeros.

One method note before the fix, because it cost real time. A restore experiment is only "passing" if it includes an unrelated fresh-session control. The restored session itself can produce plausible-looking output from a half-broken cache, so you have to open a brand new, unrelated session right after a restore and confirm the whole server is not poisoned. Early runs skipped this control and mistook a half-success for a success.

The fix removes the failure at the source rather than working around it. There are four components, all opt-in behind env gates.

**1. Canonical packed layout.** Each rank trusts only its own data, and at store time every rank writes a complete copy in a canonical layout. The half-mmap problem disappears at the storage-format level rather than being dodged. This leans on the fact that the MLA and sliding-window KV groups are replicated across tensor-parallel ranks.

**2. Group-sliced compact storage.** The initial store wrote every block as a full 1 MB slab, which put storage density at 148 to 177 KB per token; a single 500k session exceeded 80 GB. By registering, per layer and per group, how many bytes actually carry meaning and slicing only those to disk, density dropped to about 13 KB per token, a 12x reduction. A 500k session is now about 6 GB.

**3. Worker-direct pread restore.** On restore, instead of the head reading files and fanning them out over the network, each worker reads its own slice directly from shared storage. Each promoted key carries a (path, offset, length) triple, and the worker uses it to fill its own mmap. This is the core of two-node restore.

**4. Direction-split swap path.** Copies between GPU and staging use a different kernel per direction. Demotion (writing GPU to staging) goes through a batch-copy path; load (reading staging to GPU) uses a Triton kernel. Submitting large batches down one path collapsed the Blackwell driver frontend (it showed up as a clean bisection where large restores deterministically killed the GPU), and splitting by direction avoids it.

**5. TP-broadcast relay restore (opt-in).** Worker-direct pread assumes every rank can reach the store files, that is, shared storage. An alternative path has one rank (TP local rank 0) read the canonical files and broadcast the loaded bytes to the other ranks over the tensor-parallel process group. Because the KV is replicated across ranks, one read suffices, so only that one node needs the storage and the rest receive their copy over the fast interconnect. This turns the storage location into a free choice: a single node internal NVMe, an attached disk, or a network mount all behave the same, instead of requiring a mount every node can see. The broadcast is chunked into fixed-size windows (256 MiB by default) so the transient GPU staging buffer stays bounded and large-session restores stay within the configured memory budget on unified memory. Opt-in behind `DSPARK_RELAY_RESTORE`, default off.

A few smaller repairs ride along: a spec-decode asymmetry in the indexer cache, hole-skipping for partially populated blocks, and a writer-rotation half-write. All together this is around 700 lines, packaged as idempotent patch scripts so it can be reapplied after a reinstall.

The first time this pipeline ran, the GPU died deterministically right after a restore (Xid 13). After eliminating eight software hypotheses one by one, the cause turned out to be the GPU driver; upgrading it made the symptom vanish. The lesson, paid for in hours: the thing that just changed being your code does not make your code the culprit.

## What was tested

The most important property first: the client has to know nothing. Send the same conversation over the OpenAI-compatible API and the server looks up the prefix hash, finds the KV in the disk tier, and promotes it to the GPU. No session id, no restore call, no client-side state. The server metrics from the 469k qualification run prove it: zero recomputed tokens, 3,627 MB loaded from disk. A reprefill would have shown 0 MB loaded and 434 seconds of compute.

Cold build (reprefill) versus disk restore, measured by force-evicting a session and reaccessing it:

All numbers here are on shared network storage at about 1 GB/s (see the limits section for why that matters and what local NVMe would change).

| Session size | Cold build (reprefill) | Disk restore, initial | Disk restore, reworked |
|---|---|---|---|
| 94k tokens | 61 s | 8.1 s | not re-measured |
| 188k tokens | 133 s | 12.6 s | not re-measured |
| 469k tokens | 435 s | 27.8 s | ~9 s |

The "initial" column is the first working version; the "reworked" column is after the restore-path rework described in "From 28 seconds to 9" below, measured at 25.5 to 9.1 seconds on a 381k session. Small sessions were not re-measured because their stat cost was already small, so their gain is modest; the win concentrates on large sessions where serial existence-checks dominated. The speedup over reprefill grows with session size, because reprefill accelerates with context while a disk read is linear: a reworked 469k restore is roughly a 48x speedup over its 435 second reprefill.

Relay path (item 5) on a single node's local NVMe, same force-evict-and-reaccess method:

| Session size | Local-NVMe restore |
|---|---|
| 76k tokens | 2.6 s |
| 352k tokens | 7.5 s |

- Non-reading rank disk I/O, store and restore: 0 bytes.
- Four 457k-token sessions resident together: switching among them 1.7 to 2.1 s each.

But a disk restore is meant to be the rare case, not the common one. With the GPU pool set to 13 GiB (about 2.51M tokens), six sessions of about 370k tokens each stay fully resident on the GPU, and switching among them lands in 1.6 to 1.8 seconds every time. Only the seventh, evicted, session comes back from disk. In the six-resident-session qualification, store failures were zero and the memory headroom stayed comfortable.

Concurrency was checked too. Evicting two sessions and reaccessing both at once, two 188k sessions came back in 13.4 and 20.4 seconds, 2,900 MB total, zero incidents. Demotion writes going down and promotion reads coming up overlap without the staging buffer breaking. The fresh-session control passed on every run: open an unrelated session right after a restore and output stays correct with acceptance intact.

### From 28 seconds to 9: reworking the restore path

The internals of that first 28-second restore were a surprise. The GPU copy itself was 0.18 seconds, under one percent of the total. The dominant cost was the batch of existence checks that stat each chunk on disk. A single lookup round stat'd thousands of files serially, and a cold CIFS stat at 3.4 ms each made that alone 15 seconds.

Once it was clear the cost was the serial nature of the lookup, not read bandwidth, the fix followed: parallelize the existence-check stats across 16 threads, and 15 seconds became 1. Parallelizing the remote worker reads bought another second. The 469k cold restore went from 25.5 to 9.1 seconds, a 2.8x cut. Each step was confirmed with an A/B; reads were already saturating storage bandwidth so fanning them out did nothing, and only measurement revealed that serial stat was the real cost. These restore-path pieces are opt-in behind env gates as well.

## Recipe

This is an experimental fork: vLLM 0.27.1 in a venv with inline patches. `TRIAL_*` are this fork's own knobs, `VLLM_*` are upstream knobs. Several knobs exist because they are the fix for a specific bottleneck met along the way.

```
Runnable scripts are in `deploy/gb10-cluster/`. Set the machine-specific values at
the top of `serve-common.sh` (head and worker IPs, the fast-NIC name and RoCE HCA,
the KV cache directory), then run the worker first and the head second:

  ./deploy/gb10-cluster/serve-worker.sh   # worker node (rank 1)
  ./deploy/gb10-cluster/serve-head.sh     # head node (rank 0)

The recipe knobs, one per line (deploy/gb10-cluster/ds4f.env):

  TRIAL_THINK=max
  TRIAL_SPEC=dspark
  TRIAL_SPEC_N=7
  TRIAL_MOE=flashinfer_b12x
  TRIAL_LINEAR=deep_gemm
  TRIAL_MML=524288
  TRIAL_MNBT=1024
  TRIAL_GPUUTIL=0.75
  TRIAL_KVMEM=13958643712        # GPU pool, 13 GiB
  TRIAL_KVOFF=7                  # CPU staging tier, 7 GiB
  TRIAL_KVFS=1
  TRIAL_KVCANON=true
  TRIAL_KVCOMPACT=true
  TRIAL_KVBPC=16
  TRIAL_KVRT=32
  TRIAL_TAILONLY=0
  DSPARK_RELAY_RESTORE=1               # one node reads + TP broadcast (off = shared-storage pread)
  DSPARK_RELAY_WINDOW_BYTES=268435456  # 256 MiB broadcast window
  DSPARK_FS_LOOKUP_THREADS=16
  DSPARK_FS_LOAD_TASKS=16
  DSPARK_FS_PREAD_THREADS=16
  DSPARK_FS_PREAD_WINDOW=16
  DSPARK_FS_PREAD_SKIP=1
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

The full serve arg list, one per line, is in serve-common.sh. TRIAL_KVFS_DIR is set
per node from CACHE_DIR; with DSPARK_RELAY_RESTORE=1 only the head stores and reads
there, so the cache can be a single node local NVMe, an attached disk, or a mount.
```

A few knobs are not part of the offloading core but the fix for what sat on top of it. `RETENTION_INTERVAL` addresses small sliding-window and state-group blocks being billed at 1 MB each, which inflated the real cache cost of a session about fourfold; it is the key to six-session instant switching. `LOGITS_MB` caps the sparse-indexer prefill logits buffer, which had been eating unified memory (and the drafter was amplifying it); it is the key to running large sessions with the drafter on.

Speculative tokens stay at 7. Lowering it to squeeze memory distorted the output distribution, and the mistake was trusting speed and acceptance rate while skipping output-quality checks. The cause (garbled output and skipped tool calls) was later pinned with a three-way configuration comparison, and the value was returned to its normal setting. `GPUUTIL` is set low at 0.75 because a warm page cache trims the boot-time GPU memory check.

## Verification: proving the restores are real

Claims about KV restore are easy to fake by accident: a prefix-cache hit looks like a restore, a warm page cache looks like fast disk, and a recompute produces the same correct answer as a restore, only slower. The verification harness closes those holes. Method:

1. **Sessions are needle-probed.** Each session is a unique deterministic filler text (about 450k-506k tokens; unique per session so no cross-session prefix hits) with a unique secret code planted in the first sentence. Every reaccess asks for the code. A correct answer proves the restored KV is byte-faithful, not just present.
2. **Disk reads are metered, not assumed.** Every probe records the delta of `vllm:external_prefix_cache_hits_total` and `vllm:kv_offload_load_bytes_total`. "Resident" means the answer came with **zero** bytes loaded from disk; "restored" means external hits cover ~99% of the prompt and gigabytes were read.
3. **Cold means cold.** Restore tests restart both nodes' serving processes first, so the GPU pool and staging start empty and the only KV source is the disk tier.
4. **Recompute is the control.** The same session rebuilt from scratch gives the reprefill baseline the restore is judged against.

Results from the 2026-08-27 acceptance run (six ~506k sessions plus five 450k sessions, all needle-exact throughout; per-node ~120 GB unified memory, 18 GiB GPU KV pool):

```
Reaccess latency per ~500k-token session
  resident (RAM, 0 bytes from disk)   #                        2-4 s     12/12 round-robin
  parked   (restore from disk)        ####                     12-18 s   3.5-3.9 GB read, ext hits 99%+
  recompute (control)                 ############################################  ~680 s
```

```
Five 450k sessions swapped in simultaneously (cold)
  before the admission-gate fixes     ############################################  45 min  (1 restored, 4 recomputed)
  after                               ##                                            67 s    (5/5 restored, serial chain
                                                                                             19 -> 33 -> 47 -> 60 -> 67 s)
```

```
Store overhead on prefill (fresh 450k build, offloading writing every chunk)
  offloading on                       ~660-694 tok/s
  historical no-offload baseline      ~660-700 tok/s   (difference within run-to-run noise)
```

Residency capacity followed the measured per-token allocation (~5.2 KB/token with the retention-interval checkpointing), not the pessimistic "maximum concurrency" boot metric: six ~500k sessions occupy ~84% of an 18 GiB pool and all answer instantly; a seventh evicts the least-recently-used one to disk, which then costs one 12-18 s restore on its next touch. Concurrency benchmarks (2.5k-token prompts, out=512, temp 0.6) with offloading on: acceptance 1.58/1.45/1.43 at c8/c16/c32 with zero refusal lanes and monotonically rising aggregate throughput, i.e. the offloading path does not disturb speculative decoding.

The probe scripts are small (an OpenAI-client needle harness plus `/metrics` deltas) and live outside this tree; the method above is enough to reproduce them against any deployment of this fork.

### Fault injection (2026-08-28)

The load-failure wiring was verified by breaking the store on purpose, with a needle session as the oracle:

```
truncated chunk (read fails)   before: 8,487 failure lines in 483 s, request parked forever
                               after : 1 failure line -> recompute 43.8 s, needle exact
delete files mid-restore       restore completes from already-open fds: 2.5 s, needle exact
store pruned beneath session   clean full miss -> recompute 76.3 s, needle exact
```

The "before" row is itself a find from this round: the first wiring pass exposed a scheduler-side livelock (the async lookup stat cache kept re-serving HIT for a key whose promotion kept failing, one retry per scheduler step). The fix poisons failed keys and invalidates that cache, so the next lookup is a definitive MISS and the scheduler recomputes.

## Limits and open problems

An honest list.

1. Much of the 9-second restore is still control-plane and storage-bandwidth bound. The GPU copy is 0.18 seconds; the rest is spread across lookup, remote reads, tail reprefill, and tokenization. The theoretical floor is roughly 6 to 7 seconds. Going lower needs a narrower lookup and promotion prefetch, and fundamentally the real answer is a bigger pool so sessions are not evicted in the first place, which is a hardware-scale question.
2. (2026-08-30) The disk tier now has retention: with `DSPARK_TAIL_ONLY=2` a chain stores one window snapshot per prompt anchor, `DSPARK_TAIL_KEEP=K` deletes snapshots at boundaries older than the chain's K most recent once the new anchor is durable, and an external TTL pruner (window-group files only, never full-attention chunks) sweeps abandoned chains. A global size cap is still an external LRU pruner. Before this, the tier Real write rate is on the order of tens of GB per day, so it is not a near-term issue, but a GC that indexes per-session size and last access and evicts oldest-first is the next task.
3. (2026-08-30) The anchor-now tail-only store (`DSPARK_TAIL_ONLY=2`) is now validated and on in the reference deployment: a 450k-token prefill writes 62 MiB of window snapshots instead of 16 GiB, and a 101k session cold-restores in 3.1 s with a full external hit. The earlier regression was the eagle extra-chunk pop (fixed 2026-08-27) plus the store-side gate; both are covered by tests now.
4. This design optimizes for keeping many long-lived sessions persistent and instantly reaccessible, trading raw throughput and fast session turnover for reaccess latency and session persistence. That is a different target from high-churn multi-tenant serving.
5. The storage tier assumes shared storage that both nodes can see. On restore each worker reads its slice from the file the scheduler node wrote, over the shared mount, so a local per-node disk is not supported out of the box. The numbers here are on a network cache volume at about 1 GB/s, so the read-bound part of a restore (roughly half of the nine seconds: existence-check stats, promotion read, remote read) is limited by that link, not by SSD speed. Because the MLA KV is replicated across ranks, one copy suffices: the relay path (item 5 above) has a single rank read that copy and broadcast it to the others over the interconnect, so the store can sit on one node's local NVMe, an attached disk, or a network mount, and no per-node shared mount is required. It is opt-in and off by default; with it off, the default path still reads over the shared mount described here.
6. There is no always-on output-audit gate yet. Both output-quality regressions surfaced in real use rather than from a synthetic probe. Turning garbled-output sweeps, tool-call batteries, and truncated-tool-call scenarios into a standing gate is the remaining work.

Treat this fork as experimental. It is a reference for pushing disk KV offloading to measured results on a two-node unified-memory homelab, not a production drop-in.

One closing method note. Most events in this work reduce to "one discriminating experiment beats ten hypotheses." The half-mmap corruption was settled by a fresh-session control, early cache eviction by separating GPU-hit from disk-restore measurement, output distortion by a three-way comparison, and the restore bottleneck by per-stage measurement. Read the symptom, form the hypothesis, but let the next action always be the experiment that narrows it to one.

---

## Other opt-in knobs in this fork

Beyond the disk offloading recipe above, this tree carries a few more experimental, env-gated changes (all default off):

- **Restore-path parallelism** (see "From 28 seconds to 9"): `DSPARK_FS_LOOKUP_THREADS`, `DSPARK_FS_LOAD_TASKS`, `DSPARK_FS_STORE_TASKS`, `DSPARK_FS_PREAD_THREADS`, `DSPARK_FS_PREAD_WINDOW`, `DSPARK_FS_PREAD_SKIP`.
- **Per-event KV JSONL log** for offline analysis of restore/store episodes: `DSPARK_KV_EVENT_LOG=/path/file.jsonl`.
- **TP-broadcast relay restore** so only one node needs the store files (one rank reads, broadcasts to the rest over the TP group, so storage location becomes a free choice): `DSPARK_RELAY_RESTORE=1`. Broadcast window size: `DSPARK_RELAY_WINDOW_BYTES=268435456` (256 MiB).
- **DeepSeek V4 tool-call generation stabilization** (experimental, still under validation): `DSPARK_SPEC_OFF_GUIDED` (opt a tools request out of speculative-decode grammar validation), `DSPARK_DSML_LEAK_GUARD` (stop a request if tool-call markup tokens leak into free text), `DSPARK_TOOL_TEMP0` (force temperature 0 on tool turns so structure tokens stay deterministic).
- **GB10 kernel and backend knobs**: opt-in B12X MXFP4 MoE backend, DeepGemm opt-in for the mHC pre-norm path, e8m0 block-scale upcast for the Triton path, DSpark draft attention backend pinning, MLA index-width rounding.

- **Allocator ratchet fix (2026-08-28)**: `VLLM_SM12X_DG_ALLOC_BUCKETS` (bucketed logits-workspace reuse on the native SM120 mqa-logits path, default on; `0` restores exact-size allocations), `VLLM_SM12X_DG_ALLOC_LOG` (log reserved-bytes and allocator counters every N calls for slope measurement, default off).
- **Restore robustness under concurrency** (2026-08-27): `DSPARK_RESTORE_CONCURRENCY` (promotion admission gate, default 1), `DSPARK_PRIMFULL_RETRIES` (seconds of sustained staging-full to retry instead of collapsing to a MISS, default 300), `DSPARK_SWA_RESERVE` (reserve staging blocks for sliding-window tail promotion so oversized sessions restore their fitting prefix), `DSPARK_FINISH_STORE_RETRIES` (retry finish-time stores instead of dropping a parked session's tail, default 30).
- **Staging mmap placement**: `DSPARK_OFFLOAD_MMAP_DIR` (back the CPU staging mmap with a disk file instead of tmpfs; preallocated with `posix_fallocate`), `DSPARK_OFFLOAD_PIN` (host-register policy; disk-backed regions default to unpinned).
- **Relay direct broadcast**: `DSPARK_RELAY_DIRECT` (source the TP broadcast from the staging mmap instead of re-reading files, default on; fallback preads verify lengths).
- **Scheduler admission**: `DSPARK_ADMIT_RESERVED` (count in-flight prefills' reserved blocks for fresh admissions too, default on), `DSPARK_PPCAP` (cap concurrent compute-prefills).
- **fs tier hygiene**: `DSPARK_SIDECAR_STRICT` (fail loudly on layout-sidecar parse errors instead of silently switching on-disk formats), `DSPARK_FS_PROMOTED_KEYS_CAP` (LRU bound), `DSPARK_LOOKUP_TRACE` (distinguish primary-full misses from genuine misses).

- **Window-snapshot policy (2026-08-30)**: `DSPARK_TAIL_ONLY=2` (anchor-now: sliding-window groups store only the w(+e) chunks before the prompt's last full-attention-aligned boundary), `DSPARK_TAIL_CKPT_TOKENS` (optional periodic safety checkpoints during a long prefill), `DSPARK_TAIL_KEEP=K` (per-chain retention: delete snapshots at aligned boundaries older than the K most recent after the new anchor's store completes; emits `prune` events to the KV JSONL log).
- **Reasoning loop breaker (2026-08-30)**: `--reasoning-config '{"loop_break_max_pattern_size":128,"loop_break_min_pattern_size":8,"loop_break_min_count":4,"loop_break_min_reasoning_tokens":512,"loop_break_check_interval":16}'` enables scheduler-side exact-cycle detection inside the reasoning section; the verdict rides `SchedulerOutput` to the worker, which lowers that request's thinking budget so the existing V2 forcing kernel emits the reasoning end sequence and the answer continues. Per-request opt-out `thinking_loop_break: false`. `DSPARK_DEFAULT_THINKING_BUDGET=N` applies a server-side `thinking_token_budget` when the client omits it (`-1` in the request disables). Both log once per section.
- **SM120 decode tiling (2026-08-30)**: `DSPARK_SM120_DECODE_TILE=64` (default) slices the DSpark verify batch into <=64-row FlashInfer runner calls so it always takes the standalone decode kernels; `0` restores whole-batch calls.

These are documented in their commit messages. This repository will keep accumulating experiments.

---

## Changelog

### 2026-08-30 (evening) — lint pass and four upstream fixes (5 commits)

- **Every fork-modified Python file now passes the repo pre-commit hooks** (ruff-check, ruff-format, typos, pinned to the upstream revisions). 152 files were run through the hooks; 33 changed (formatting, import order, f-strings, `contextlib.suppress`, `logger.exception`, three identifiers the typos hook flagged). No behaviour change. Earlier rounds had not been linted; from now on the hooks run before every commit.
- **vLLM #52311**: off-by-one in `bad_words` draft-prefix matching (Model Runner V2), with its GPU test.
- **vLLM #52707**: `allocate_external_computed_blocks` could request a negative block count and inflate the free-queue counter; clamped at zero, test added by hand to the fork's older test file.
- **vLLM #53329**: request-level cascades no longer drop keys whose GPU->primary write is still in flight; they are parked on the request, flushed at schedule end, hold finalization, count as pending work and clear on `reset_cache`. Re-expressed on this fork's v0.27.1 tiering-manager shape. The fork's tiering tests also got the nine `RETRY -> HIT_PENDING` expectation fixes that the 2026-08-21 #51840 backport had skipped; the file is green (49/49).
- **vLLM #53962**: spec-decode padding no longer pads a decode up to exactly `max_model_len` (the next async step's cap went negative and crashed the runner). The DSPARK loop-breaker hooks in the scheduler now read their state through `getattr`, so upstream's `__init__`-less test scheduler works.
- Verification on GB10: CPU suites 261/261 (tiering, offloading scheduler, tail retention, loop breaker, single-type manager, async scheduler, fs tier, async lookup), GPU suites 59/59 (bad_words, gumbel, rejection sampler); live pair c16 x 500: 196 tok/s aggregate, spec accept 0.78-0.81, lane acceptance 1.39, 0 refusals; loop breaker fired at 5,466 reasoning tokens and the answer continued; thinking budget honoured; disk-tier restore gate on the pair: a 67.5k-token session cold-restored after a full restart with 65,536 external hits, 536 MB read, 2.2 s TTFT and the needle recalled, suffix re-ask 0.5 s.

### 2026-08-30 (afternoon) — draft Gumbel noise stream decoupled (1 commit)

- **Rejected-draft resampling no longer shares noise with the draft** (port of vLLM #54282, merged on main 2026-08-29, absent from v0.28.0/v0.28.1rc0). With `draft_sample_method=probabilistic` (this deployment's default) the residual resample drew its Gumbel noise from the same Philox offset that had produced the draft token, so the tokens the draft ranked highest were under-weighted and the output distribution drifted from the target's. The draft stream now carries an offset salt (`1 << 30`) through `gumbel_sample(is_drafting=)`; the target sampler and the rejection resample stay on the unsalted stream. Ported by hand onto this fork's pre-#52816 `gumbel.py` layout; `sample_draft` keys the draw by the position before the sampled token instead of the old "+1" trick (dflash: `sample_pos - 1`). Verification on GB10: 55/55 GPU unit tests including the new unbiasedness test (removing the salt reproduces the bias at chi2 1585 against a threshold of 70); live pair c16 x 500: 204-210 tok/s aggregate, spec accept 0.78-0.82, lane acceptance 1.28 (baseline 195-218 / 1.26), 0 refusals. `tests/v1/spec_decode/test_rejection_sampler_utils.py` was refreshed to the v0.28.0 version: the v0.27.1 copy did not apply temperature to the draft reference distribution and failed at temperature 0.6 against this fork's kernel regardless of the patch. ruff check/format and typos clean.

### 2026-08-30 — snapshot retention, reasoning loop breaker, decode tiling (4 commits)

- **Window snapshots stopped filling the disk.** The tier held 655 GiB of which 92% were sliding-window state snapshots written at every 4096-token boundary during prefill (upstream v0.28 semantics, not a bug) while a restore reads only the last one. `DSPARK_TAIL_ONLY=2` is now the reference setting (450k prefill: 16 GiB -> 62 MiB of snapshots) and `DSPARK_TAIL_KEEP=K` deletes a chain's superseded snapshots after the new anchor is durable. Review-driven guards: never touch full-attention chunks, never mutate the promoted/poison bookkeeping on delete (rank>0 would read unfilled staging), clip ranges when the window is wider than a segment. External TTL script for abandoned chains. Validated on the testbed: 101k session cold restore 3.1 s with 98,304-token external hit, prune event deleting exactly the previous anchor's 5 files.
- **Reasoning loop breaker for Model Runner V2.** vLLM PR #52677 lives in the V1 thinking-budget holder and cannot run with DSpark (V2). Ported as scheduler-side detection over the authoritative output token ids (`check_sequence_repetition`, reasoning-scoped) with the verdict shipped in `SchedulerOutput` and applied through the V2 `ThinkingBudgetState` forcing kernel. Along the way the V2 budget machinery was found unwired (no `reasoning_config` to the sampler, no budget term in the logits-processing gate, missing natural end-marker ids, request rejected at ingestion); all fixed, so `thinking_token_budget` works on this stack too. Live: an induced loop is cut after ~2k-4k reasoning tokens and the answer follows; an enumeration control never fires.
- **DSpark verify batches tiled to <=64 rows.** The FlashInfer sparse-MLA runner dispatches >64 query rows to the prefill orchestrator, which deadlocks stochastically on SM121 (mia #141, flashinfer #3700/#4732) and reallocates split-K scratch per step; with 16 sequences x (5+1) the verify shape was 96 rows on every busy step. `_forward_decode` now loops over 64-row slices reusing the warmup-reserved scratch. A/B on the testbed at 96 rows: 3-5% aggregate cost, no errors; the 4-node TP=4 layout keeps 16 sequences.
- **Indexer shortcut capture guard** (vLLM #52492): the short-context "all candidates selected" shortcut is skipped while a CUDA graph is being captured, so a replay after a long prefix-cache hit cannot select unscored candidates.
- Ops layer (outside this tree): pre-start memory compaction on the unified-memory nodes (fragmented UVM memory stalled the 18 GiB KV allocation for 20+ minutes; `compact_memory` clears it in under a minute, `vm.compaction_proactiveness=100` keeps it from recurring), watchdog boot grace 720 -> 1500 s, ten venv-only live patches preserved in-tree.

### 2026-08-28 — reliability round (2 commits)

- **Load failures recompute end to end.** A chunk that exists at lookup but fails at read (truncated, pruned, unlinked mid-flight) used to die as an assert, a zero-fill, or — as fault injection revealed — a scheduler-side livelock. Now: per-block read failures are reported, failed keys are poisoned, the async-lookup stat cache is invalidated, and the affected span is recomputed. Store failures stay fail-stop (never marked as stored). A hybrid-group zip guard stops silent block-table truncation.
- **Bounded logits workspace for native DeepGemm.** The SM120 mqa-logits path rounds KV length up a geometric ladder and serves logits from a zero-copy view of the resident workspace (small LRU of bucketed buffers as fallback, DBO ubatch threads bypass the cache). Kills the reserved-memory ratchet under mixed-length prefills.
- Ops layer (outside this tree): the pair pre-start now waits for the peer node after a whole-rack reboot instead of burning its start limit, the watchdog detects a long-stuck activating state, and the start timeout was raised to cover the new wait budgets.

### 2026-08-27 — hardening round (19 commits)

- **Concurrent cold swap-ins: 45 min -> 67 s.** Three pieces, all required: a restore admission gate so only N requests initiate fs-to-staging promotions at once; retry-instead-of-MISS when a chunk exists on disk but staging is merely full (the residual collapse path: a request taking its turn while the previous restore's load still pinned staging used to discard the whole hybrid restore); and a time-based retry budget after measuring that a per-key budget burns out within a single lookup pass.
- **Relay restore sources the broadcast from staging** (the bytes are guaranteed resident there by the promotion contract), halving restore-path disk reads; 500k restores 16-32 s -> 13.8-16.1 s. Remaining pread paths verify read lengths and poison loudly.
- **Admission counts in-flight prefill reservations** for fresh long prompts (previously only async-load admissions), so N large prompts admit in capacity-sized waves instead of over-admitting into preemption thrash.
- **Finish-time store retry** keeps a parked session's tail chunks instead of silently dropping them when staging is momentarily full at request finish.
- **Staging mmap lifecycle**: env-selectable disk backing (frees ~12 GB of tmpfs per node on unified memory), `posix_fallocate` so ENOSPC is a clean startup error rather than a SIGBUS mid-offload, unpinned by default when disk-backed.
- **Cold-restore reserve cap**: oversized sessions restore the prefix that fits staging instead of collapsing to a full recompute.
- **fs tier hardening**: mtime refresh on dedup-skip stores and on loads (external LRU pruners saw first-write age and evicted the hottest chunks first), per-block batch validation, unlink only on short-read evidence (fd exhaustion or O_DIRECT EINVAL no longer deletes good cache files; C extension rebuilt to match), strict layout-sidecar failures.
- **Eagle extra-chunk pop** no longer walks an interior restore boundary down to a full recompute (pop only when the extra chunk actually hit).
- Misc: `_fs_promoted_keys` LRU bound, env-gated lookup trace, preserved live venv-only features in-tree (`DSPARK_PPCAP` admission cap, tiering metrics extension), fs_io build recipe.

### 2026-08-25/26 — SM120 attention overlay and output quality

- Build-free Triton sparse-MLA overlay for GB10/SM12x (adopted from jasl's ds4-sm120-min-enable line) with warmup/sampling subsystems; fixes the FlashInfer sparse-MLA prefill stall under concurrency.
- Drafter ring KV keyed by persistent request index (mixed-batch acceptance collapse fix), live-path wiring included.
- Prefill top-k rebase disabled on both call sites (it dropped every non-first prefill request's compressed context: refusal storms and acceptance collapse under batching).
- B12X MXFP4 MoE route block cap; env-gated native DeepGEMM SM120 MQA logits and tf32 pre-norm routes; out-of-context fill skip in the rowwise logits kernel.

### 2026-08-21 — initial public patch set

- Two-node disk KV offloading: canonical packed layout for hybrid caches, group-slice fs IO with a layout sidecar, single-writer store dedup, TP-broadcast relay restore, restore-path parallelism, retention-interval fix for small-block cache inflation, sparse-indexer logits cap, DeepSeek V4 tool-call stabilization gates.
