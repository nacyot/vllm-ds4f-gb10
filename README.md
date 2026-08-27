# vllm-ds4f-gb10

> Experimental patch set on top of [vLLM](https://github.com/vllm-project/vllm) **v0.27.1**, for serving DeepSeek V4 Flash across two DGX Spark **GB10** nodes with unified memory. Single user, specific hardware, reference implementation, not a production guarantee.

**Base:** vLLM v0.27.1. Build and install with vLLM's own instructions; this fork applies the patches below on top of that tree. License: Apache-2.0 (inherited from vLLM, see `LICENSE`).

**Author:** nacyot

Every change here is opt-in behind an env var and defaults to stock vLLM behavior. The blog below is the main content; a short knob index and a changelog follow it.

**Status (2026-08-27).** The goal is now met and exceeded on the reference hardware: **six ~500k-token sessions resident** in an 18 GiB GPU pool with instant (2-4 s) switching and zero disk reads, cold-restart restore of any parked session in **12-18 s** (about 47x over full reprefill, needle-exact), and **concurrent cold swap-ins self-serializing in ~67 s** for five 450k sessions (previously 45 minutes of mutual staging eviction). Concurrency benchmarks (c8/c16/c32) hold their acceptance and scaling with offloading on. See the changelog at the bottom for what landed.

---

# Keeping many long agent sessions resident: disk KV offloading for multi-node vLLM

**The goal.** On two GB10 DGX Spark boxes with unified memory, keep five or more sessions of 400k+ tokens resident and instant to switch between, and let any older session that has been evicted come back without paying the full reprefill cost. That is the target this work was built for. Everything else, including the restore-time work, is in service of it.

This is an experimental patch set on top of [vLLM](https://github.com/vllm-project/vllm) 0.27.1. It targets a specific setup (two DGX Spark GB10 nodes, unified memory, DeepSeek V4 Flash FP8, tensor parallel across the two nodes) and a single-user workload. It is a reference implementation, not a production guarantee. If you run something similar, the parts below should transfer; if you do not, read it as a case study.

Author: nacyot.

## The problem

When you self-host an LLM for long sessions, the real cost is not decode throughput. It is reprefill. A 500k-token agent session that gets evicted from the GPU cache has to be recomputed from scratch the moment you send the next turn, because a prefix-cache miss means the server walks the whole conversation again to rebuild its attention keys and values. On this hardware that reprefill runs into the multi-minute range. Waiting minutes for one reply effectively forces you to use a single session.

The bottleneck for a local agent, then, is not tokens per second. It is how many sessions you can keep alive. GPU memory is finite, so with several sessions the oldest gets pushed out, and reopening it is a full reprefill. You cannot solve this inside the GPU alone.

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

| Session size | Cold build (reprefill) | Disk restore | Speedup |
|---|---|---|---|
| 94k tokens | 61 s | 8.1 s | 7.5x |
| 188k tokens | 133 s | 12.6 s | 10.5x |
| 469k tokens | 435 s | 27.8 s | 15.6x |

The speedup grows with session size, because reprefill accelerates with context while a disk read is linear.

But a disk restore is meant to be the rare case, not the common one. With the GPU pool set to 13 GiB (about 2.51M tokens), six sessions of about 370k tokens each stay fully resident on the GPU, and switching among them lands in 1.6 to 1.8 seconds every time. Only the seventh, evicted, session comes back from disk. In the six-resident-session qualification, store failures were zero and the memory headroom stayed comfortable.

Concurrency was checked too. Evicting two sessions and reaccessing both at once, two 188k sessions came back in 13.4 and 20.4 seconds, 2,900 MB total, zero incidents. Demotion writes going down and promotion reads coming up overlap without the staging buffer breaking. The fresh-session control passed on every run: open an unrelated session right after a restore and output stays correct with acceptance intact.

### From 28 seconds to 9: reworking the restore path

The internals of that first 28-second restore were a surprise. The GPU copy itself was 0.18 seconds, under one percent of the total. The dominant cost was the batch of existence checks that stat each chunk on disk. A single lookup round stat'd thousands of files serially, and a cold CIFS stat at 3.4 ms each made that alone 15 seconds.

Once it was clear the cost was the serial nature of the lookup, not read bandwidth, the fix followed: parallelize the existence-check stats across 16 threads, and 15 seconds became 1. Parallelizing the remote worker reads bought another second. The 469k cold restore went from 25.5 to 9.1 seconds, a 2.8x cut. Each step was confirmed with an A/B; reads were already saturating storage bandwidth so fanning them out did nothing, and only measurement revealed that serial stat was the real cost. These restore-path pieces are opt-in behind env gates as well.

## Recipe

This is an experimental fork: vLLM 0.27.1 in a venv with inline patches. `TRIAL_*` are this fork's own knobs, `VLLM_*` are upstream knobs. Several knobs exist because they are the fix for a specific bottleneck met along the way.

```
GPU pool:  13 GiB (TRIAL_KVMEM=13958643712, ~2.51M tokens, six 370k sessions resident)
Staging:   8 GiB  (TRIAL_KVOFF=8, buffer between GPU and disk)
Disk:      cache volume mounted CIFS hard,retrans=6 on both nodes

env:  TRIAL_KVFS=1 TRIAL_KVFS_DIR=<cache-volume>/kv/<node>-compact
      TRIAL_KVCANON=true TRIAL_KVCOMPACT=true    (canonical packed offloading)
      TRIAL_KVBPC=16 TRIAL_KVRT=32               (blocks per chunk, read threads)
      VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768 (sparse retention of window groups)
      VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128       (indexer logits buffer cap)
      TRIAL_SPEC=dspark TRIAL_SPEC_N=5            (speculative decoding)
      TRIAL_GPUUTIL=0.75 TRIAL_MNBT=1024

start: worker node unit first, then head node unit (systemd user units + linger)
guards: earlyoom + vm.min_free_kbytes 2GB + a dirty-page ceiling, persistent on both nodes
watch: promotion-failure counter, dmesg Xid, available memory, cache-volume usage
```

A few knobs are not part of the offloading core but the fix for what sat on top of it. `RETENTION_INTERVAL` addresses small sliding-window and state-group blocks being billed at 1 MB each, which inflated the real cache cost of a session about fourfold; it is the key to six-session instant switching. `LOGITS_MB` caps the sparse-indexer prefill logits buffer, which had been eating unified memory (and the drafter was amplifying it); it is the key to running large sessions with the drafter on.

Speculative tokens stay at 5. Lowering it to squeeze memory distorted the output distribution, and the mistake was trusting speed and acceptance rate while skipping output-quality checks. The cause (garbled output and skipped tool calls) was later pinned with a three-way configuration comparison, and the value was returned to its normal setting. `GPUUTIL` is set low at 0.75 because a warm page cache trims the boot-time GPU memory check.

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

## Limits and open problems

An honest list.

1. Restores now run 13.8-16.1 s for ~500k sessions (relay-direct removed a redundant per-chunk disk read), but much of that is still control-plane bound. The GPU copy is 0.18 seconds; the rest is spread across lookup, remote reads, tail reprefill, and tokenization. The theoretical floor is roughly 6 to 7 seconds. Going lower needs a narrower lookup and promotion prefetch, and fundamentally the real answer is a bigger pool so sessions are not evicted in the first place, which is a hardware-scale question.
2. The disk tier has no built-in capacity cap or GC; an external mtime-ordered pruner handles capacity. Dedup-skip stores and loads now refresh file mtimes so that pruner behaves as LRU rather than FIFO-by-first-write, but an in-tree GC that indexes per-session size and last access remains future work.
3. A more aggressive tail-only store, which cuts storage further, was implemented and shown to save space, but a regression where restore lookup misses the window groups' store timing left it off by default.
4. This design assumes a single user with unbounded context. It optimizes for reaccess latency over throughput and for session persistence over fast session turnover, which is a different target from multi-tenant serving.
5. There is no always-on output-audit gate yet. Both output-quality regressions surfaced in real use rather than from a synthetic probe. Turning garbled-output sweeps, tool-call batteries, and truncated-tool-call scenarios into a standing gate is the remaining work.

6. Worker-side load errors are not yet wired into `get_block_ids_with_load_errors`, so the configured `kv_load_failure_policy=recompute` cannot fire from a failed read; the current mitigation is a loud error plus zero-fill instead of silently passing stale bytes. Wiring the report through is the next correctness task.
7. The native DeepGEMM indexer-logits route allocates an exact-size output per call; under varied-width prefill churn this ratchets the allocator's reserved memory (A/B measured: about 70% of the worker-side creep). A bucketed reusable output buffer for the native route is the planned fix; the Triton route already has one.

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

- **Restore robustness under concurrency** (2026-08-27): `DSPARK_RESTORE_CONCURRENCY` (promotion admission gate, default 1), `DSPARK_PRIMFULL_RETRIES` (seconds of sustained staging-full to retry instead of collapsing to a MISS, default 300), `DSPARK_SWA_RESERVE` (reserve staging blocks for sliding-window tail promotion so oversized sessions restore their fitting prefix), `DSPARK_FINISH_STORE_RETRIES` (retry finish-time stores instead of dropping a parked session's tail, default 30).
- **Staging mmap placement**: `DSPARK_OFFLOAD_MMAP_DIR` (back the CPU staging mmap with a disk file instead of tmpfs; preallocated with `posix_fallocate`), `DSPARK_OFFLOAD_PIN` (host-register policy; disk-backed regions default to unpinned).
- **Relay direct broadcast**: `DSPARK_RELAY_DIRECT` (source the TP broadcast from the staging mmap instead of re-reading files, default on; fallback preads verify lengths).
- **Scheduler admission**: `DSPARK_ADMIT_RESERVED` (count in-flight prefills' reserved blocks for fresh admissions too, default on), `DSPARK_PPCAP` (cap concurrent compute-prefills).
- **fs tier hygiene**: `DSPARK_SIDECAR_STRICT` (fail loudly on layout-sidecar parse errors instead of silently switching on-disk formats), `DSPARK_FS_PROMOTED_KEYS_CAP` (LRU bound), `DSPARK_LOOKUP_TRACE` (distinguish primary-full misses from genuine misses).

These are documented in their commit messages. This repository will keep accumulating experiments.

---

## Changelog

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
