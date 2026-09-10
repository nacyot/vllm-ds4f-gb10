#!/usr/bin/env python3
"""First-boot checks against the V4.1 head (default http://127.0.0.1:8889).

Greedy, thinking off. Prints each answer, token counts and tok/s, whether the
1..100 count is exact, and whether a repeated prompt is deterministic. This is
a functional smoke, not an accuracy evaluation."""

import json
import sys
import time
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8889"
models = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))
model = models["data"][0]["id"]


def chat(prompt, max_tokens=256, thinking=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": thinking},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    u = r.get("usage", {})
    txt = r["choices"][0]["message"].get("content") or ""
    n = u.get("completion_tokens", 0)
    print(f"--- {prompt[:60]!r}")
    print(
        f"    prompt_tokens={u.get('prompt_tokens')} completion_tokens={n} "
        f"time={dt:.1f}s decode~{n / max(dt, 1e-9):.1f} tok/s"
    )
    print(f"    {txt.strip()[:400]!r}", flush=True)
    return txt, u, dt


print("model:", model)
count, _, _ = chat(
    "Count from 1 to 100, separated by spaces. Output only the numbers.", 420
)
nums = [int(x) for x in count.replace(",", " ").split() if x.isdigit()]
exact = nums[:100] == list(range(1, 101))
print("count 1..100 exact:", exact, f"({len(nums)} numbers)")
a1, _, _ = chat("What is 84 * 3 / 2? Answer with the number only.", 32)
a2, _, _ = chat("What is 84 * 3 / 2? Answer with the number only.", 32)
print("repeat greedy identical:", a1 == a2, "| numeric ok:", "126" in a1)
chat("Write a Python function is_prime(n) with a short docstring. Code only.", 200)
chat("한국어로 두 문장: 서울의 대표적인 음식 하나를 소개해 주세요.", 120)
facts = "\n".join(f"Fact {i}: item {i} has value {i * 7 % 101}." for i in range(1, 700))
long_prompt = (
    "The following is a numbered list of facts.\n"
    + facts
    + "\nQuestion: what is the value of item 512? Answer with the number only."
)
txt, u, dt = chat(long_prompt, 32)
expected = str(512 * 7 % 101)
print(
    f"long-prompt needle (item 512 -> {expected}):",
    expected in txt,
    f"| prefill {u.get('prompt_tokens')} tokens in {dt:.1f}s",
)
