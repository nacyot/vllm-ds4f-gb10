# Issue #28 implementation results

## Outcome

Implemented on 2026-09-12, 23:18–23:20 KST from the approved plan and manager recipe `icmt-0458006a`.

- Head: 10 duplicate scripts archived and replaced with absolute symlinks. Each worker: 4. Total: 22 symlinks, no duplicate regular script copies remaining.
- Every original was moved intact to that node's `/home/nacyot/dsv41-prep/superseded-2026-09-12/`. Head additionally archived `bench2_summary.py` without a replacement, as approved. Total archived files: 23.
- README adds one sentence documenting the symlinks. No script implementation changed. Node checkouts were not patched in this stage; their existing tracked files remained untouched.
- SHA-256 checks passed before and after each move. All other inventoried Python/shell files retained their original checksums. Head's adjacent `bench_prompts_v1.json` still matches the repo copy.
- Commands ran sequentially in foreground. No deletion, server control, clock mutation, kvfs/venv modification, pytest, torch import, or background job was performed. No safety confirmation appeared.

## Scope limitations and validation handoff

The dedicated validate stage still owns the head symlink import/help checks, one `kvoff_probe.py i28-smoke S28 700` request, and final adopted-configuration/end-state checks. No inference has been run in implement. Use `/home/nacyot/vllm-dsv41/.venv/bin/python` on head; its existing link targets `/home/nacyot/vllm-dsv41-venv`. Workers lack a repo `.venv` link; no worker Python execution or environment change is needed for this task. The first f323 read-only inventory stopped at that absent link, then was rerun with an optional environment check; no mutation occurred in the failed read.

Worker checkout `kvoff_probe.py` matches the pre-guard copy, unlike head. These links track each node's own checkout, not head's contents. Repo version synchronization belongs to the development/deployment work; probes remain head-only. Do not claim all four probe versions now have #19's guard.

`bench2_summary.py` is now only at `~/dsv41-prep/superseded-2026-09-12/bench2_summary.py`. The old top-level tag summary command is intentionally unavailable. Its missing-C4 `StatisticsError` and path/tag argument mismatch remain unfixed; any future restoration should address those together, not point users at the archived tool.

The server configuration was not changed. Immediately after file operations, head health was 200, MemAvailable 4.59543 GiB, and all four cap services were active at 1989 MHz. These are implementation observations, not the validate stage's final sign-off.

## Inventory and movement evidence

All prep entries below were regular files before the operation. `same`/`different` are `cmp` results against the same basename in the node's repo. `absent` means no repo counterpart. Symlink target prefix is `/home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/`; archive prefix is `/home/nacyot/dsv41-prep/superseded-2026-09-12/`. Realpaths of both source and repo directories matched these literal paths; archives did not exist before creation. Every mutation named a single literal source and destination, used `mv -n`, and checked the result before continuing. Originals remain available for rollback; no archive was overwritten.

### gx10-6040

Checkout: `eea34808799cf18f0073c64623e79671e564d12f`; `git status --short` was empty before operations.

| File | cmp | Action | Original SHA-256 |
| --- | --- | --- | --- |
| `bench.py` | different | archive + symlink | `e4e11acba81a32de0fe5ea4cabe21cd4deff2553af3827e0daec3c3be29491d0` |
| `bench2.py` | same | archive + symlink | `16f8a35713d07f0ebaf72ba6794c842d1df1d911b0a6cd0aaf99d40c962a81d5` |
| `bench2_summary.py` | absent | archive only | `e255dc8ab3f66e867c24b0eb447c9c4a2cdac07f9ce9d774b39aff4a481b94da` |
| `burn.py` | absent | preserved | `22d414d750dba8e3bce5d2ba66bf5c4bf22cc8feba439923be1a759027be5792` |
| `cmp.py` | absent | preserved | `8d67e95f54fb7d1a01e7a3615b147f8bb921cf65e826c05d3e4bb58367dc5654` |
| `divergence.py` | same | archive + symlink | `bac99c085892c2fae42b240126daef1a25c037145fa0feb1ed6f4b1b0f826062` |
| `engram_layout.py` | absent | preserved | `ff05985a8b7b1073cd737a5bd7a1657f76122a1ec2b327216fb2a551860b7920` |
| `grid_probe.py` | absent | preserved | `876c9bd874d2377bf3d5ea0f7e5d7ba5f5c849482144a3ecfb98fd0c6fce3310` |
| `i3_keys.py` | absent | preserved | `de3a73af69bc8af713aef37cc833e60ef456366c46627fea55ca5b9b4b57b606` |
| `kai_disk_table.py` | absent | preserved | `837fdf148528ab93162e0f077984919ad7ee1c466467bdc6e842eb4db341d911` |
| `kvoff_concurrent.py` | different | archive + symlink | `c3e9bf28838d443d9256bd86a0f3939dad2e6b70ffeb4f8e9731023c711e0ab1` |
| `kvoff_probe.py` | different | archive + symlink | `b2e1fb7b1a80f008c2a6b16de20b434f29e478e6161130702cc39553334d1278` |
| `make_mini_v41.py` | absent | preserved | `db13a0b42596650b53a036d59619486a84aa8ecc143969b41e875df6e4a38cd8` |
| `memlog.py` | same | archive + symlink | `b9215446c5732d82021a07142b0b6f9c0ca026d37a854928f66f335091306b8c` |
| `memtrace_summary.py` | same | archive + symlink | `5cbf9fdcf66e26da3f5ec8eeec81b61b5670ee392df7de037eff7042f1df14ba` |
| `nccl_lat.py` | absent | preserved | `00d5d93db71b98b92963820988e6a3a1637aed286acfb8687b16e93c9bb57c02` |
| `pread_bench.py` | absent | preserved | `96e8555198c581516dd642e2ea50b671f3dde16e3910a11b3527a35919850a34` |
| `prefill_probe.py` | different | archive + symlink | `4ecf788e94a7114f8171ab9a3fdb7221334a028254de869f9341cb63cef2747c` |
| `prof_summary.py` | different | archive + symlink | `6e9b67316d726b8734ae89af47423e966f9ac926989e97553ddd5d59602eb170` |
| `real_engram_gpu_probe.py` | absent | preserved | `3cfa18737398637a7549a5dab5f0f7cd3c098a06eca77e536780efb37de0efa6` |
| `real_engram_probe.py` | absent | preserved | `e1ae83095103608b09542fe5ad41f6b3657eba15d136516869c4cad653e9f3b7` |
| `ref_engram.py` | absent | preserved | `f8b2d88a4962873b5e6de68ae14f5ca3a2f48ebf3488625f3ec4a1aaa7057876` |
| `rowhash.py` | absent | preserved | `120aa74bc96be6a17e7310c74f9768af1d531ec0f967c3fcfc96352b1a1d5489` |
| `smoke.py` | different | archive + symlink | `0555aafd1c29a2fa500d87cc1b758a532b3fcec6903c79553fbaac399d8abc3c` |
| `test_engram_mmap.py` | absent | preserved | `60ca076e3ac222228b2040fa57fd2ab101c37ca3eb7d46141d6a3419008f4f03` |
| `test_hash_ids.py` | absent | preserved | `567fd02230d744f833eb11fbb78452ab53fd8864adbfb52f46d2001b043114f9` |
| `test_rank_bug.py` | absent | preserved | `585a4beb965d124fbfae5a8d2fbff6906806610a78b105bd437d3c955f3d2ac0` |
| `verify_manifest.py` | absent | preserved | `ed81cd59b1d685eef64b2bb0bcee561dfcebe1f09af0e7335c39400b8eae7808` |
| `kvfs_fault.sh` | absent | preserved | `82f47ccd24541e127882b79cdc26c1c2acc44d4412e8d96c0beb0f22c6880cf5` |
| `mini_smoke.sh` | absent | preserved | `4008eeca609fdc87a98b7777b055406ada0e0f8d31d7d6b5eaa843e8a5d5f2d6` |

Move log, including individual before/after `ls -la` and checksum checks:

```text
/home/nacyot/dsv41-prep/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/bench.py: OK
/home/nacyot/dsv41-prep/bench2_summary.py: OK
/home/nacyot/dsv41-prep/bench2.py: OK
/home/nacyot/dsv41-prep/divergence.py: OK
/home/nacyot/dsv41-prep/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/memlog.py: OK
/home/nacyot/dsv41-prep/memtrace_summary.py: OK
/home/nacyot/dsv41-prep/prefill_probe.py: OK
/home/nacyot/dsv41-prep/prof_summary.py: OK
/home/nacyot/dsv41-prep/smoke.py: OK
-rw-r--r-- 1 nacyot nacyot 3831 Sep 12 08:15 /home/nacyot/dsv41-prep/kvoff_probe.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
lrwxrwxrwx 1 nacyot nacyot 64 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_probe.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_probe.py
-rw-r--r-- 1 nacyot nacyot 3831 Sep 12 08:15 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py
-rw-r--r-- 1 nacyot nacyot 4929 Sep 11 05:34 /home/nacyot/dsv41-prep/bench.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/bench.py: OK
lrwxrwxrwx 1 nacyot nacyot 58 Sep 12 23:19 /home/nacyot/dsv41-prep/bench.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/bench.py
-rw-r--r-- 1 nacyot nacyot 4929 Sep 11 05:34 /home/nacyot/dsv41-prep/superseded-2026-09-12/bench.py
-rw-r--r-- 1 nacyot nacyot 1272 Sep 12 10:56 /home/nacyot/dsv41-prep/bench2_summary.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/bench2_summary.py: OK
-rw-r--r-- 1 nacyot nacyot 1272 Sep 12 10:56 /home/nacyot/dsv41-prep/superseded-2026-09-12/bench2_summary.py
-rw-r--r-- 1 nacyot nacyot 10921 Sep 11 09:39 /home/nacyot/dsv41-prep/bench2.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/bench2.py: OK
lrwxrwxrwx 1 nacyot nacyot 59 Sep 12 23:19 /home/nacyot/dsv41-prep/bench2.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/bench2.py
-rw-r--r-- 1 nacyot nacyot 10921 Sep 11 09:39 /home/nacyot/dsv41-prep/superseded-2026-09-12/bench2.py
-rw-r--r-- 1 nacyot nacyot 3844 Sep 11 09:39 /home/nacyot/dsv41-prep/divergence.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/divergence.py: OK
lrwxrwxrwx 1 nacyot nacyot 63 Sep 12 23:19 /home/nacyot/dsv41-prep/divergence.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/divergence.py
-rw-r--r-- 1 nacyot nacyot 3844 Sep 11 09:39 /home/nacyot/dsv41-prep/superseded-2026-09-12/divergence.py
-rw-r--r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/kvoff_concurrent.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_concurrent.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_concurrent.py
-rw-r--r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/memlog.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
lrwxrwxrwx 1 nacyot nacyot 59 Sep 12 23:19 /home/nacyot/dsv41-prep/memlog.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memlog.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 19:17 /home/nacyot/dsv41-prep/memtrace_summary.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/memtrace_summary.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memtrace_summary.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 19:17 /home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py
-rwxr-xr-x 1 nacyot nacyot 3313 Sep 12 08:33 /home/nacyot/dsv41-prep/prefill_probe.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/prefill_probe.py: OK
lrwxrwxrwx 1 nacyot nacyot 66 Sep 12 23:19 /home/nacyot/dsv41-prep/prefill_probe.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/prefill_probe.py
-rwxr-xr-x 1 nacyot nacyot 3313 Sep 12 08:33 /home/nacyot/dsv41-prep/superseded-2026-09-12/prefill_probe.py
-rwxr-xr-x 1 nacyot nacyot 6445 Sep 12 08:15 /home/nacyot/dsv41-prep/prof_summary.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/prof_summary.py: OK
lrwxrwxrwx 1 nacyot nacyot 65 Sep 12 23:19 /home/nacyot/dsv41-prep/prof_summary.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/prof_summary.py
-rwxr-xr-x 1 nacyot nacyot 6445 Sep 12 08:15 /home/nacyot/dsv41-prep/superseded-2026-09-12/prof_summary.py
-rw-r--r-- 1 nacyot nacyot 2491 Sep 11 00:36 /home/nacyot/dsv41-prep/smoke.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/smoke.py: OK
lrwxrwxrwx 1 nacyot nacyot 58 Sep 12 23:19 /home/nacyot/dsv41-prep/smoke.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/smoke.py
-rw-r--r-- 1 nacyot nacyot 2491 Sep 11 00:36 /home/nacyot/dsv41-prep/superseded-2026-09-12/smoke.py
2026-09-12T23:19:06+09:00
total 72
drwxrwxr-x  2 nacyot nacyot  4096 Sep 12 23:19 .
drwxrwxr-x 15 nacyot nacyot  4096 Sep 12 23:19 ..
-rw-r--r--  1 nacyot nacyot  4929 Sep 11 05:34 bench.py
-rw-r--r--  1 nacyot nacyot 10921 Sep 11 09:39 bench2.py
-rw-r--r--  1 nacyot nacyot  1272 Sep 12 10:56 bench2_summary.py
-rw-r--r--  1 nacyot nacyot  3844 Sep 11 09:39 divergence.py
-rw-r--r--  1 nacyot nacyot  1072 Sep 11 14:28 kvoff_concurrent.py
-rw-r--r--  1 nacyot nacyot  3831 Sep 12 08:15 kvoff_probe.py
-rw-r--r--  1 nacyot nacyot  4690 Sep 11 19:36 memlog.py
-rw-r--r--  1 nacyot nacyot  2964 Sep 11 19:17 memtrace_summary.py
-rwxr-xr-x  1 nacyot nacyot  3313 Sep 12 08:33 prefill_probe.py
-rwxr-xr-x  1 nacyot nacyot  6445 Sep 12 08:15 prof_summary.py
-rw-r--r--  1 nacyot nacyot  2491 Sep 11 00:36 smoke.py
```

Post-operation integrity and status checks:

```text
2026-09-12T23:19:44+09:00
/home/nacyot/dsv41-prep/superseded-2026-09-12/bench.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/bench2.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/bench2_summary.py: OK
/home/nacyot/dsv41-prep/burn.py: OK
/home/nacyot/dsv41-prep/cmp.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/divergence.py: OK
/home/nacyot/dsv41-prep/engram_layout.py: OK
/home/nacyot/dsv41-prep/grid_probe.py: OK
/home/nacyot/dsv41-prep/i3_keys.py: OK
/home/nacyot/dsv41-prep/kai_disk_table.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/make_mini_v41.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
/home/nacyot/dsv41-prep/nccl_lat.py: OK
/home/nacyot/dsv41-prep/pread_bench.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/prefill_probe.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/prof_summary.py: OK
/home/nacyot/dsv41-prep/real_engram_gpu_probe.py: OK
/home/nacyot/dsv41-prep/real_engram_probe.py: OK
/home/nacyot/dsv41-prep/ref_engram.py: OK
/home/nacyot/dsv41-prep/rowhash.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/smoke.py: OK
/home/nacyot/dsv41-prep/test_engram_mmap.py: OK
/home/nacyot/dsv41-prep/test_hash_ids.py: OK
/home/nacyot/dsv41-prep/test_rank_bug.py: OK
/home/nacyot/dsv41-prep/verify_manifest.py: OK
/home/nacyot/dsv41-prep/kvfs_fault.sh: OK
/home/nacyot/dsv41-prep/mini_smoke.sh: OK
duplicate regular scripts: 0; original checksums and symlinks: PASS
active
1989 MHz
MemAvailable_GiB 4.59543
health 200
```

### gx10-f323

Checkout: `c375ddecc3a54ae4706044834f7030de64dd58ee`; `git status --short` was empty before operations.

| File | cmp | Action | Original SHA-256 |
| --- | --- | --- | --- |
| `burn.py` | absent | preserved | `22d414d750dba8e3bce5d2ba66bf5c4bf22cc8feba439923be1a759027be5792` |
| `kvoff_concurrent.py` | different | archive + symlink | `c3e9bf28838d443d9256bd86a0f3939dad2e6b70ffeb4f8e9731023c711e0ab1` |
| `kvoff_probe.py` | same | archive + symlink | `b2e1fb7b1a80f008c2a6b16de20b434f29e478e6161130702cc39553334d1278` |
| `memlog.py` | same | archive + symlink | `b9215446c5732d82021a07142b0b6f9c0ca026d37a854928f66f335091306b8c` |
| `memtrace_summary.py` | same | archive + symlink | `5cbf9fdcf66e26da3f5ec8eeec81b61b5670ee392df7de037eff7042f1df14ba` |
| `nccl_lat.py` | absent | preserved | `00d5d93db71b98b92963820988e6a3a1637aed286acfb8687b16e93c9bb57c02` |
| `rowhash.py` | absent | preserved | `120aa74bc96be6a17e7310c74f9768af1d531ec0f967c3fcfc96352b1a1d5489` |
| `verify_manifest.py` | absent | preserved | `ed81cd59b1d685eef64b2bb0bcee561dfcebe1f09af0e7335c39400b8eae7808` |
| `mini_smoke.sh` | absent | preserved | `4008eeca609fdc87a98b7777b055406ada0e0f8d31d7d6b5eaa843e8a5d5f2d6` |

Move log, including individual before/after `ls -la` and checksum checks:

```text
/home/nacyot/dsv41-prep/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/memlog.py: OK
/home/nacyot/dsv41-prep/memtrace_summary.py: OK
-rw-rw-r-- 1 nacyot nacyot 3831 Sep 11 14:28 /home/nacyot/dsv41-prep/kvoff_probe.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
lrwxrwxrwx 1 nacyot nacyot 64 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_probe.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_probe.py
-rw-rw-r-- 1 nacyot nacyot 3831 Sep 11 14:28 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py
-rw-rw-r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/kvoff_concurrent.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_concurrent.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_concurrent.py
-rw-rw-r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/memlog.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
lrwxrwxrwx 1 nacyot nacyot 59 Sep 12 23:19 /home/nacyot/dsv41-prep/memlog.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memlog.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 19:17 /home/nacyot/dsv41-prep/memtrace_summary.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/memtrace_summary.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memtrace_summary.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 19:17 /home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py
2026-09-12T23:19:17+09:00
total 28
drwxrwxr-x  2 nacyot nacyot 4096 Sep 12 23:19 .
drwxrwxr-x 12 nacyot nacyot 4096 Sep 12 23:19 ..
-rw-rw-r--  1 nacyot nacyot 1072 Sep 11 14:28 kvoff_concurrent.py
-rw-rw-r--  1 nacyot nacyot 3831 Sep 11 14:28 kvoff_probe.py
-rw-r--r--  1 nacyot nacyot 4690 Sep 11 19:36 memlog.py
-rw-r--r--  1 nacyot nacyot 2964 Sep 11 19:17 memtrace_summary.py
```

Post-operation integrity and status checks:

```text
2026-09-12T23:19:45+09:00
/home/nacyot/dsv41-prep/burn.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
/home/nacyot/dsv41-prep/nccl_lat.py: OK
/home/nacyot/dsv41-prep/rowhash.py: OK
/home/nacyot/dsv41-prep/verify_manifest.py: OK
/home/nacyot/dsv41-prep/mini_smoke.sh: OK
duplicate regular scripts: 0; original checksums and symlinks: PASS
active
1989 MHz
MemAvailable_GiB 9.28991
```

### gx10-37cc

Checkout: `02f43fe4e43cc71e3226fd4c6d6b61062a50823b`; `git status --short` was empty before operations.

| File | cmp | Action | Original SHA-256 |
| --- | --- | --- | --- |
| `burn.py` | absent | preserved | `22d414d750dba8e3bce5d2ba66bf5c4bf22cc8feba439923be1a759027be5792` |
| `kvoff_concurrent.py` | different | archive + symlink | `c3e9bf28838d443d9256bd86a0f3939dad2e6b70ffeb4f8e9731023c711e0ab1` |
| `kvoff_probe.py` | same | archive + symlink | `b2e1fb7b1a80f008c2a6b16de20b434f29e478e6161130702cc39553334d1278` |
| `memlog.py` | same | archive + symlink | `b9215446c5732d82021a07142b0b6f9c0ca026d37a854928f66f335091306b8c` |
| `memtrace_summary.py` | same | archive + symlink | `5cbf9fdcf66e26da3f5ec8eeec81b61b5670ee392df7de037eff7042f1df14ba` |
| `nccl_lat.py` | absent | preserved | `00d5d93db71b98b92963820988e6a3a1637aed286acfb8687b16e93c9bb57c02` |
| `rowhash.py` | absent | preserved | `120aa74bc96be6a17e7310c74f9768af1d531ec0f967c3fcfc96352b1a1d5489` |
| `verify_manifest.py` | absent | preserved | `ed81cd59b1d685eef64b2bb0bcee561dfcebe1f09af0e7335c39400b8eae7808` |
| `mini_smoke.sh` | absent | preserved | `4008eeca609fdc87a98b7777b055406ada0e0f8d31d7d6b5eaa843e8a5d5f2d6` |

Move log, including individual before/after `ls -la` and checksum checks:

```text
/home/nacyot/dsv41-prep/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/memlog.py: OK
/home/nacyot/dsv41-prep/memtrace_summary.py: OK
-rw-rw-r-- 1 nacyot nacyot 3831 Sep 11 14:28 /home/nacyot/dsv41-prep/kvoff_probe.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
lrwxrwxrwx 1 nacyot nacyot 64 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_probe.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_probe.py
-rw-rw-r-- 1 nacyot nacyot 3831 Sep 11 14:28 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py
-rw-rw-r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/kvoff_concurrent.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_concurrent.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_concurrent.py
-rw-rw-r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/memlog.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
lrwxrwxrwx 1 nacyot nacyot 59 Sep 12 23:19 /home/nacyot/dsv41-prep/memlog.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memlog.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 20:29 /home/nacyot/dsv41-prep/memtrace_summary.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/memtrace_summary.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memtrace_summary.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 20:29 /home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py
2026-09-12T23:19:17+09:00
total 28
drwxrwxr-x  2 nacyot nacyot 4096 Sep 12 23:19 .
drwxrwxr-x 11 nacyot nacyot 4096 Sep 12 23:19 ..
-rw-rw-r--  1 nacyot nacyot 1072 Sep 11 14:28 kvoff_concurrent.py
-rw-rw-r--  1 nacyot nacyot 3831 Sep 11 14:28 kvoff_probe.py
-rw-r--r--  1 nacyot nacyot 4690 Sep 11 19:36 memlog.py
-rw-r--r--  1 nacyot nacyot 2964 Sep 11 20:29 memtrace_summary.py
```

Post-operation integrity and status checks:

```text
2026-09-12T23:19:45+09:00
/home/nacyot/dsv41-prep/burn.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
/home/nacyot/dsv41-prep/nccl_lat.py: OK
/home/nacyot/dsv41-prep/rowhash.py: OK
/home/nacyot/dsv41-prep/verify_manifest.py: OK
/home/nacyot/dsv41-prep/mini_smoke.sh: OK
duplicate regular scripts: 0; original checksums and symlinks: PASS
active
1989 MHz
MemAvailable_GiB 9.0787
```

### gx10-27c4

Checkout: `3494b6a2b7c4d30ab89fe3c6f99072e76ead4a02`; `git status --short` was empty before operations.

| File | cmp | Action | Original SHA-256 |
| --- | --- | --- | --- |
| `burn.py` | absent | preserved | `22d414d750dba8e3bce5d2ba66bf5c4bf22cc8feba439923be1a759027be5792` |
| `kvoff_concurrent.py` | different | archive + symlink | `c3e9bf28838d443d9256bd86a0f3939dad2e6b70ffeb4f8e9731023c711e0ab1` |
| `kvoff_probe.py` | same | archive + symlink | `b2e1fb7b1a80f008c2a6b16de20b434f29e478e6161130702cc39553334d1278` |
| `memlog.py` | same | archive + symlink | `b9215446c5732d82021a07142b0b6f9c0ca026d37a854928f66f335091306b8c` |
| `memtrace_summary.py` | same | archive + symlink | `5cbf9fdcf66e26da3f5ec8eeec81b61b5670ee392df7de037eff7042f1df14ba` |
| `nccl_lat.py` | absent | preserved | `00d5d93db71b98b92963820988e6a3a1637aed286acfb8687b16e93c9bb57c02` |
| `rowhash.py` | absent | preserved | `120aa74bc96be6a17e7310c74f9768af1d531ec0f967c3fcfc96352b1a1d5489` |
| `verify_manifest.py` | absent | preserved | `ed81cd59b1d685eef64b2bb0bcee561dfcebe1f09af0e7335c39400b8eae7808` |
| `mini_smoke.sh` | absent | preserved | `4008eeca609fdc87a98b7777b055406ada0e0f8d31d7d6b5eaa843e8a5d5f2d6` |

Move log, including individual before/after `ls -la` and checksum checks:

```text
/home/nacyot/dsv41-prep/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/memlog.py: OK
/home/nacyot/dsv41-prep/memtrace_summary.py: OK
-rw-rw-r-- 1 nacyot nacyot 3831 Sep 11 14:28 /home/nacyot/dsv41-prep/kvoff_probe.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
lrwxrwxrwx 1 nacyot nacyot 64 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_probe.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_probe.py
-rw-rw-r-- 1 nacyot nacyot 3831 Sep 11 14:28 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py
-rw-rw-r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/kvoff_concurrent.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/kvoff_concurrent.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/kvoff_concurrent.py
-rw-rw-r-- 1 nacyot nacyot 1072 Sep 11 14:28 /home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/memlog.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
lrwxrwxrwx 1 nacyot nacyot 59 Sep 12 23:19 /home/nacyot/dsv41-prep/memlog.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memlog.py
-rw-r--r-- 1 nacyot nacyot 4690 Sep 11 19:36 /home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 19:17 /home/nacyot/dsv41-prep/memtrace_summary.py
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
lrwxrwxrwx 1 nacyot nacyot 69 Sep 12 23:19 /home/nacyot/dsv41-prep/memtrace_summary.py -> /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/memtrace_summary.py
-rw-r--r-- 1 nacyot nacyot 2964 Sep 11 19:17 /home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py
2026-09-12T23:19:17+09:00
total 28
drwxrwxr-x  2 nacyot nacyot 4096 Sep 12 23:19 .
drwxrwxr-x 11 nacyot nacyot 4096 Sep 12 23:19 ..
-rw-rw-r--  1 nacyot nacyot 1072 Sep 11 14:28 kvoff_concurrent.py
-rw-rw-r--  1 nacyot nacyot 3831 Sep 11 14:28 kvoff_probe.py
-rw-r--r--  1 nacyot nacyot 4690 Sep 11 19:36 memlog.py
-rw-r--r--  1 nacyot nacyot 2964 Sep 11 19:17 memtrace_summary.py
```

Post-operation integrity and status checks:

```text
2026-09-12T23:19:46+09:00
/home/nacyot/dsv41-prep/burn.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_concurrent.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/kvoff_probe.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memlog.py: OK
/home/nacyot/dsv41-prep/superseded-2026-09-12/memtrace_summary.py: OK
/home/nacyot/dsv41-prep/nccl_lat.py: OK
/home/nacyot/dsv41-prep/rowhash.py: OK
/home/nacyot/dsv41-prep/verify_manifest.py: OK
/home/nacyot/dsv41-prep/mini_smoke.sh: OK
duplicate regular scripts: 0; original checksums and symlinks: PASS
active
1989 MHz
MemAvailable_GiB 9.28081
```

## Repository checks

`pre-commit install` installed the configured hooks using the existing uv-managed pre-commit tool. Both `pre-commit run markdownlint-cli2 --files <three changed Markdown files>` and `pre-commit run typos --files <three changed Markdown files>` passed. `git diff --check` passed. Dynamic smoke and final operational verification remain assigned to validate.

## Validation round 1 — operational gate pending

Reviewed `6118253cd98cfff49a8b2f3e5592488654702ec2..22470fc067` against both the original request and approved plan. The 22 node symlinks, 23 preserved originals, README sentence and movement evidence meet the file-change scope. No application implementation changed; the existing bench output path and head's adjacent prompt JSON remain intact. The explicitly approved summary archival and worker repo-version limitation are recorded above.

Commands on head (2026-09-12 23:22 KST), from `/home/nacyot/dsv41-prep`:

- `/home/nacyot/vllm-dsv41/.venv/bin/python -B prefill_probe.py --help`: exit 0.
- Same interpreter imported `HeadroomError, metrics, run_prompt` from `kvoff_probe`; asserted its realpath equals the repo file and `torch` is absent from `sys.modules`: exit 0.
- `/home/nacyot/vllm-dsv41/.venv/bin/python -B kvoff_probe.py i28-smoke S28 700`: exit 0, one request, correct answer. Actual size was 9,115 tokens (the recipe's nominal 8K); no extra inference was run.

```json
{"salt": "S28", "prompt_tokens": 9115, "ttft_s": 5.043, "answer": "12", "expected": "12", "ok": true, "mem_avail_start_gib": null, "mem_avail_min_gib": null, "tag": "i28-smoke", "delta": {"kv_offload_store_bytes": 27112320.0, "external_prefix_cache_queries": 9115.0, "external_prefix_cache_hits": 0.0, "prefix_cache_hits": 0.0, "kv_offload_lookup_sync_delay_seconds_sum": 0.026078477851115167, "prefix_cache_queries": 9115.0}}
```

The short probe does not enable the long-request memory monitor, hence its null memory fields; these are not measured minima. Shell observations: before request 4.58975 GiB, afterward 3.23519 GiB at 23:22:38, then 3.42737 GiB and 3.41165 GiB at 23:23:48 after another 30-second idle observation. **Head MemAvailable remains below the required 4.5 GiB, so validation has not passed and no merge transition is authorized by this result.** Further inference was stopped. No restart or configuration change was attempted.

At 23:22:38–40, all four serving units and cap services were active, with 1989 MHz observed and head health 200. Symlink recheck passed with 10/4/4/4 links and zero duplicate regular scripts; all node checkouts remained clean. Head health 200 and cap active/1989 MHz persisted at 23:23:48. Serving units' activation timestamps remain 22:58:04 on head and 22:57:57–58 on workers, before this task's operations.

Read-only inspection of head's existing service process confirmed `--tensor-parallel-size 4 --nnodes 4 --port 8889`, `ENGRAM_PREFETCH=1`, `EMPTY_CACHE=1`, and `EMPTY_CACHE_MIN_TOKENS=65536`; its command line has the matching empty-cache and prefetch settings. The requested adopted configuration remains active. The short request is below the configured 65,536-token release threshold; retained allocator memory is a possible explanation, not a measured allocation diagnosis. Do not run a larger request or change settings to force release under this task.

`pre-commit run --from-ref 6118253cd98cfff49a8b2f3e5592488654702ec2 --to-ref HEAD` passed all applicable hooks; non-applicable code hooks skipped. `git diff --check 6118253cd98cfff49a8b2f3e5592488654702ec2..HEAD` passed. No build is needed for Markdown and filesystem symlinks. Initial read-only discovery queried nonexistent unit `dsv41-r0` and found no remote `rg`; corrected to the existing `dsv41-serve` unit and `grep`, without state changes.

Remaining gate: manager/owner coordination for head memory recovery under the existing prohibition on this worker stopping/restarting the server. Once recovery is reported, recheck memory, health and caps; do not repeat the successful inference merely to re-establish its result. Keep the workflow in validate until the end-state requirement is satisfied or the owner explicitly revises it.
