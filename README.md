# vllm-ds4f-gb10

> Experimental patch set on top of [vLLM](https://github.com/vllm-project/vllm) **v0.27.1**, for serving DeepSeek V4 Flash across two DGX Spark **GB10** nodes with unified memory. Single user, specific hardware, reference implementation, not a production guarantee.

**Base:** vLLM v0.27.1. Build and install with vLLM's own instructions; this fork applies the patches below on top of that tree. License: Apache-2.0 (inherited from vLLM, see `LICENSE`).

**Author:** nacyot

Every change here is opt-in behind an env var and defaults to stock vLLM behavior. The blog below is the main content; a short knob index follows it.

(한국어: [README.ko.md](README.ko.md))

---

# Keeping many long agent sessions resident: disk KV offloading for multi-node vLLM

**The goal.** On two GB10 DGX Spark boxes with unified memory, keep five or more sessions of 400k+ tokens resident and instant to switch between, and let any older session that has been evicted come back without paying the full reprefill cost. That is the target this work was built for. Everything else, including the restore-time work, is in service of it.

This is an experimental patch set on top of [vLLM](https://github.com/vllm-project/vllm) 0.27.1. It targets a specific setup (two DGX Spark GB10 nodes, unified memory, DeepSeek V4 Flash FP8, tensor parallel across the two nodes) and a single-user workload. It is a reference implementation, not a production guarantee. If you run something similar, the parts below should transfer; if you do not, read it as a case study.

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
# Two GB10 boxes, tensor parallel across both (mp backend) over the 200G link.
# Adjust the machine-specific values: head/worker IPs, the fast-NIC name, the KV cache dir.

# --- environment, sourced on both nodes before launch (the running recipe) ---
TRIAL_THINK=max
TRIAL_SPEC=dspark TRIAL_SPEC_N=7             # dspark drafter, ~1.5x decode
TRIAL_MOE=flashinfer_b12x TRIAL_LINEAR=deep_gemm
TRIAL_MML=524288 TRIAL_MNBT=1024 TRIAL_GPUUTIL=0.75

TRIAL_KVMEM=13958643712                      # GPU pool, 13 GiB (about 2.51M tokens)
TRIAL_KVOFF=7                                # CPU staging tier, 7 GiB (buffer between GPU and disk)
TRIAL_KVFS=1
TRIAL_KVFS_DIR=<cache-dir>/kv/<node>-compact # per node (head and worker use distinct subdirs)
TRIAL_KVCANON=true TRIAL_KVCOMPACT=true       # canonical packed offloading
TRIAL_KVBPC=16 TRIAL_KVRT=32                  # blocks per chunk, read threads
TRIAL_TAILONLY=0

DSPARK_RELAY_RESTORE=1 DSPARK_RELAY_WINDOW_BYTES=268435456   # one node reads + 256 MiB TP broadcast
DSPARK_FS_LOOKUP_THREADS=16 DSPARK_FS_LOAD_TASKS=16          # head: parallel lookup and promotion
DSPARK_FS_PREAD_THREADS=16 DSPARK_FS_PREAD_WINDOW=16 DSPARK_FS_PREAD_SKIP=1

VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768    # sparse retention of window groups
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128         # indexer logits buffer cap
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# NCCL over the fast link, pin to your NIC and RoCE HCAs
NCCL_SOCKET_IFNAME=<nic> TP_SOCKET_IFNAME=<nic> GLOO_SOCKET_IFNAME=<nic>
NCCL_IB_HCA=<hca> NCCL_CROSS_NIC=1 NCCL_IB_GID_INDEX=3

# --- shared serve args, identical on both nodes ---
ARGS=(deepseek-ai/DeepSeek-V4-Flash-0731 --served-model-name deepseek-v4-flash-0731 --trust-remote-code
  --tensor-parallel-size 2 --nnodes 2 --master-addr <head-ip> --master-port 29501 --distributed-executor-backend mp
  --kv-cache-dtype fp8_ds_mla --block-size 256
  --max-model-len $TRIAL_MML --max-num-seqs 6 --max-num-batched-tokens $TRIAL_MNBT --long-prefill-token-threshold 1024
  --gpu-memory-utilization $TRIAL_GPUUTIL --kv-cache-memory $TRIAL_KVMEM --kv-offloading-size $TRIAL_KVOFF
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","canonical_layout":true,"dspark_compact_packed":true,"blocks_per_chunk":16,"secondary_tiers":[{"type":"fs","root_dir":"'"$TRIAL_KVFS_DIR"'","n_read_threads":32,"n_write_threads":8}]}}'
  --enable-prefix-caching --enable-chunked-prefill
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4
  --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"'"$TRIAL_THINK"'"}'
  --moe-backend $TRIAL_MOE --linear-backend $TRIAL_LINEAR
  --speculative-config '{"method":"dspark","num_speculative_tokens":'"$TRIAL_SPEC_N"'}')

# --- launch: worker (rank 1) first, then head (rank 0) ---
VLLM_HOST_IP=<worker-ip> vllm serve "${ARGS[@]}" --node-rank 1 --headless
VLLM_HOST_IP=<head-ip>   vllm serve "${ARGS[@]}" --node-rank 0 --host 0.0.0.0 --port 8888

# TRIAL_KVFS_DIR is per node. With DSPARK_RELAY_RESTORE=1 only the head needs the
# store files (single canonical writer plus relay read); the worker writes and reads
# nothing there. Point it at a local NVMe path, an attached disk, or a shared mount.
```

A few knobs are not part of the offloading core but the fix for what sat on top of it. `RETENTION_INTERVAL` addresses small sliding-window and state-group blocks being billed at 1 MB each, which inflated the real cache cost of a session about fourfold; it is the key to six-session instant switching. `LOGITS_MB` caps the sparse-indexer prefill logits buffer, which had been eating unified memory (and the drafter was amplifying it); it is the key to running large sessions with the drafter on.

Speculative tokens stay at 7. Lowering it to squeeze memory distorted the output distribution, and the mistake was trusting speed and acceptance rate while skipping output-quality checks. The cause (garbled output and skipped tool calls) was later pinned with a three-way configuration comparison, and the value was returned to its normal setting. `GPUUTIL` is set low at 0.75 because a warm page cache trims the boot-time GPU memory check.

## Limits and open problems

An honest list.

1. Much of the 9-second restore is still control-plane and storage-bandwidth bound. The GPU copy is 0.18 seconds; the rest is spread across lookup, remote reads, tail reprefill, and tokenization. The theoretical floor is roughly 6 to 7 seconds. Going lower needs a narrower lookup and promotion prefetch, and fundamentally the real answer is a bigger pool so sessions are not evicted in the first place, which is a hardware-scale question.
2. The disk tier has no capacity cap or GC. Real write rate is on the order of tens of GB per day, so it is not a near-term issue, but a GC that indexes per-session size and last access and evicts oldest-first is the next task.
3. A more aggressive tail-only store, which cuts storage further, was implemented and shown to save space, but a regression where restore lookup misses the window groups' store timing left it off by default.
4. This design assumes a single user with unbounded context. It optimizes for reaccess latency over throughput and for session persistence over fast session turnover, which is a different target from multi-tenant serving.
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

These are documented in their commit messages. This repository will keep accumulating experiments.
