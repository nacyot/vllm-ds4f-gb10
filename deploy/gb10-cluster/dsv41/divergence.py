#!/usr/bin/env python3
"""Record greedy completions with per-token top-2 logprobs for a few prompts,
or compare two recordings and report the first differing token: whether the
target's argmax at that position differs (numerical drift) or the committed
token is not the argmax (verification/commit error).

Usage: divergence.py record <tag> [--base URL]   |   divergence.py compare <tagA> <tagB>
"""

import json
import os
import sys
import urllib.request

OUT = os.path.expanduser("~/dsv41-prep/bench")
PROMPTS = {
    "ko-food": "한국어로 두 문장: 서울의 대표적인 음식 하나를 소개해 주세요.",
    "ko-busan": "한국어로 답해 주세요. 부산에 처음 가는 친구에게 "
    "하루 일정을 추천해 주세요. 다섯 문장 이내로.",
    "code": "Write a Python function is_prime(n) with a short docstring. Code only.",
    "count": "Count from 1 to 100, separated by spaces. Output only the numbers.",
}


def record(tag, base):
    model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][
        0
    ]["id"]
    out = {}
    for name, prompt in PROMPTS.items():
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 2,
            "chat_template_kwargs": {"thinking": False},
        }
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        ch = r["choices"][0]
        toks = []
        for item in (ch.get("logprobs") or {}).get("content") or []:
            top = [(t["token"], t["logprob"]) for t in item.get("top_logprobs", [])]
            toks.append(
                {"token": item["token"], "logprob": item["logprob"], "top": top}
            )
        out[name] = {"text": ch["message"].get("content") or "", "tokens": toks}
        print(f"{name}: {len(toks)} tokens :: {out[name]['text'][:60]!r}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"div-{tag}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def compare(a, b):
    with open(os.path.join(OUT, f"div-{a}.json")) as f:
        A = json.load(f)
    with open(os.path.join(OUT, f"div-{b}.json")) as f:
        B = json.load(f)
    for name in A:
        ta, tb = A[name]["tokens"], B[name]["tokens"]
        i = next(
            (
                k
                for k in range(min(len(ta), len(tb)))
                if ta[k]["token"] != tb[k]["token"]
            ),
            None,
        )
        if i is None:
            print(
                f"{name}: identical for {min(len(ta), len(tb))} tokens "
                f"(lens {len(ta)}/{len(tb)})"
            )
            continue
        xa, xb = ta[i], tb[i]
        argmax_a = xa["top"][0][0] if xa["top"] else None
        argmax_b = xb["top"][0][0] if xb["top"] else None
        kind = (
            "commit-not-argmax"
            if (xb["token"] != argmax_b or xa["token"] != argmax_a)
            else "argmax-differs (numerical)"
        )
        print(
            f"{name}: first diff at token {i}: "
            f"{a}={xa['token']!r} (lp {xa['logprob']:.3f}, top {xa['top']}) | "
            f"{b}={xb['token']!r} (lp {xb['logprob']:.3f}, top {xb['top']}) -> {kind}"
        )
        print(f"    context: {''.join(t['token'] for t in ta[max(0, i - 8) : i])!r}")


if __name__ == "__main__":
    if sys.argv[1] == "record":
        base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8889"
        record(sys.argv[2], base)
    else:
        compare(sys.argv[2], sys.argv[3])
