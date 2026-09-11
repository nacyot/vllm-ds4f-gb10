#!/usr/bin/env python3
"""Summarize a torch profiler trace of the V4.1 server: GPU time by category
(MoE grouped GEMM, MoE glue, dense MXFP8 GEMM, wo_a bf16 bmm, attention/indexer,
engram, NCCL, norm/mHC, other), per-step totals, GPU idle gaps with their
neighbouring kernels, NCCL kernel distribution and the unclassified kernels.

Usage: prof_summary.py <trace.json[.gz]> [--steps] [--top N] [--gap-ms 0.3]
"""

import argparse
import collections
import gzip
import json
import re

CATS = [
    ("nccl", r"nccl"),
    ("engram", r"engram|ngram"),
    (
        "moe_gemm",
        (
            r"fp8_fp4_gemm|fp8_fp4|m_grouped|grouped_gemm|mxfp4|b12x.*moe|fused_moe"
            r"|moe.*gemm"
        ),
    ),
    (
        "moe_glue",
        (
            r"moe|expert|topk_softmax|topk_sigmoid|grouped_topk|router|routing|silu"
            r"|swiglu|situ|permute|align_block|sort|cumsum|group_quant|per_token"
        ),
    ),
    (
        "attn_indexer",
        (
            r"block_scores|candidate|mask_candidates|mqa_logits|indexer|topk_log|top_k"
            r"|topk"
        ),
    ),
    (
        "attention",
        (
            r"prefill|mla|sparse|flashinfer|attn|attention|paged|rope|rotary|kv_cache"
            r"|cache_kernel|concat_and_cache|compress|swa|fmha"
        ),
    ),
    ("dense_gemm", r"mxfp8|cutlass|mm_mxfp8|nvjet|cublas|gemm|matmul|bmm|xmma|wgmma"),
    ("norm_mhc", r"rms|norm|hc_prenorm|sinkhorn|mhc|hyper"),
    ("memcpy", r"memcpy|memset"),
    (
        "elementwise",
        (
            r"elementwise|vectorized|copy_|fill|cat_|index_select|gather|scatter|arange"
            r"|reduce|softmax|where|argmax|sampl|triton_|add_|mul_|convert|cast|unrolled"
        ),
    ),
]


def cat_of(name):
    low = name.lower()
    for cat, pat in CATS:
        if re.search(pat, low):
            return cat
    return "other"


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def union_busy(kern):
    busy = 0.0
    cur_s = cur_e = None
    for e in kern:
        s, en = e["ts"], e["ts"] + e["dur"]
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, en
        else:
            cur_e = max(cur_e, en)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--gap-ms", type=float, default=0.3)
    ap.add_argument("--steps", action="store_true", help="per forward-step table")
    args = ap.parse_args()

    data = load(args.trace)
    ev = [e for e in data["traceEvents"] if e.get("ph") == "X"]
    kern = sorted(
        (e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
        key=lambda e: e["ts"],
    )
    if not kern:
        print("no GPU kernels in trace")
        return
    t0, t1 = kern[0]["ts"], max(e["ts"] + e["dur"] for e in kern)
    span = t1 - t0
    busy = union_busy(kern)
    print(
        f"events {len(ev)}, GPU kernels {len(kern)}, span {span / 1e3:.1f} ms, "
        f"GPU busy {busy / 1e3:.1f} ms ({100 * busy / span:.1f}%)"
    )

    bycat = collections.defaultdict(float)
    cnt = collections.Counter()
    byname = collections.defaultdict(float)
    cntn = collections.Counter()
    for e in kern:
        c = cat_of(e["name"])
        bycat[c] += e["dur"]
        cnt[c] += 1
        byname[e["name"]] += e["dur"]
        cntn[e["name"]] += 1
    total = sum(bycat.values())
    print("\n== GPU time by category (sum of kernel durations) ==")
    for c, v in sorted(bycat.items(), key=lambda kv: -kv[1]):
        print(f"  {c:14s} {v / 1e3:9.1f} ms {100 * v / total:5.1f}%  n={cnt[c]}")

    print(f"\n== top {args.top} kernels ==")
    for n, v in sorted(byname.items(), key=lambda kv: -kv[1])[: args.top]:
        print(
            f"  {v / 1e3:8.1f} ms n={cntn[n]:5d} avg={v / cntn[n]:8.1f}us "
            f"[{cat_of(n)}] {n[:120]}"
        )

    others = [(n, v) for n, v in byname.items() if cat_of(n) == "other"]
    if others:
        print("\n== unclassified kernels ==")
        for n, v in sorted(others, key=lambda kv: -kv[1])[:20]:
            print(f"  {v / 1e3:8.1f} ms n={cntn[n]:5d} {n[:140]}")

    gaps = []
    prev_e, prev_n = kern[0]["ts"] + kern[0]["dur"], kern[0]["name"]
    for e in kern[1:]:
        s = e["ts"]
        if s - prev_e > args.gap_ms * 1e3:
            gaps.append((s - prev_e, prev_n, e["name"]))
        if s + e["dur"] >= prev_e:
            prev_e, prev_n = s + e["dur"], e["name"]
    print(
        f"\n== GPU idle gaps > {args.gap_ms} ms: n={len(gaps)} "
        f"total={sum(g[0] for g in gaps) / 1e3:.1f} ms =="
    )
    gb = collections.defaultdict(lambda: [0, 0.0])
    for g in gaps:
        k = (g[1][:50], g[2][:50])
        gb[k][0] += 1
        gb[k][1] += g[0]
    for k, v in sorted(gb.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {v[1] / 1e3:8.1f} ms n={v[0]:4d} after [{k[0]}] before [{k[1]}]")

    nk = sorted(e["dur"] for e in kern if "nccl" in e["name"].lower())
    if nk:
        print(
            f"\n== NCCL kernels: n={len(nk)} total={sum(nk) / 1e3:.1f} ms "
            f"median={nk[len(nk) // 2]:.0f}us p90={nk[int(len(nk) * 0.9)]:.0f}us "
            f"max={nk[-1]:.0f}us"
        )

    ann = [e for e in ev if e.get("cat") in ("user_annotation", "gpu_user_annotation")]
    annc = collections.Counter(e["name"][:60] for e in ann)
    print("\n== annotations ==", annc.most_common(12))

    if args.steps:
        fwd = sorted(
            (e for e in ann if e["name"] == "gpu_model_runner: forward"),
            key=lambda e: e["ts"],
        )
        if not fwd:
            print("no forward annotations")
            return
        bounds = [e["ts"] for e in fwd] + [float("inf")]
        print(f"\n== per forward step ({len(fwd)} steps, binned by start time) ==")
        head = " ".join(f"{c[0]:>10s}" for c in CATS[:8])
        print("  step   busy_ms  span_ms  " + head)
        i = 0
        for si in range(len(fwd)):
            lo, hi = bounds[si], bounds[si + 1]
            ks = []
            while i < len(kern) and kern[i]["ts"] < hi:
                if kern[i]["ts"] >= lo:
                    ks.append(kern[i])
                i += 1
            if not ks:
                continue
            per = collections.defaultdict(float)
            for e in ks:
                per[cat_of(e["name"])] += e["dur"]
            sp = max(e["ts"] + e["dur"] for e in ks) - ks[0]["ts"]
            print(
                f"  {si:4d} {union_busy(ks) / 1e3:9.1f} {sp / 1e3:8.1f}  "
                + " ".join(f"{per[c[0]] / 1e3:10.1f}" for c in CATS[:8])
            )


if __name__ == "__main__":
    main()
