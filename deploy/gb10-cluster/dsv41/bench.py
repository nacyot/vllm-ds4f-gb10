#!/usr/bin/env python3
"""Before/after bench for the V4.1 server: same prompts, greedy, thinking off.

Measures TTFT and decode tok/s from streamed chat completions (single stream,
3 runs), an 8k-token prefill TTFT, 4 concurrent streams, and the greedy
outputs of the functional prompts (hashed) so a change can be rejected when
outputs drift. Writes one JSON per run.

Usage: bench.py <tag> [base_url] [out_dir]
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

tag = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8889"
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/dsv41-prep/bench")
os.makedirs(out_dir, exist_ok=True)
model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0][
    "id"
]


def stream_chat(prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    text = []
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                piece = c.get("delta", {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
    end = time.perf_counter()
    n = usage.get("completion_tokens", 0)
    ttft = (first or end) - t0
    decode_s = end - (first or end)
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": n,
        "ttft_s": round(ttft, 3),
        "total_s": round(end - t0, 3),
        "decode_tok_s": round((n - 1) / decode_s, 2)
        if decode_s > 0 and n > 1
        else None,
        "text": "".join(text),
    }


ESSAY = (
    "Write a 300-word essay on why unified memory changes local inference. No headings."
)
FUNCTIONAL = [
    ("Count from 1 to 100, separated by spaces. Output only the numbers.", 420),
    ("What is 84 * 3 / 2? Answer with the number only.", 32),
    ("Write a Python function is_prime(n) with a short docstring. Code only.", 200),
    ("한국어로 두 문장: 서울의 대표적인 음식 하나를 소개해 주세요.", 120),
]
facts = "\n".join(f"Fact {i}: item {i} has value {i * 7 % 101}." for i in range(1, 700))
NEEDLE = (
    "The following is a numbered list of facts.\n"
    + facts
    + "\nQuestion: what is the value of item 512? Answer with the number only."
)

result = {"tag": tag, "model": model, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
print("model:", model, "tag:", tag, flush=True)

runs = []
for i in range(3):
    r = stream_chat(ESSAY, 256)
    runs.append({k: v for k, v in r.items() if k != "text"})
    print(
        f"single[{i}] ttft={r['ttft_s']}s decode={r['decode_tok_s']} tok/s "
        f"n={r['completion_tokens']}",
        flush=True,
    )
result["single_stream"] = runs

r = stream_chat(NEEDLE, 8)
result["prefill_8k"] = {k: v for k, v in r.items() if k != "text"} | {
    "answer": r["text"].strip()[:20]
}
print(
    f"prefill_8k prompt={r['prompt_tokens']} ttft={r['ttft_s']}s "
    f"answer={r['text'].strip()[:10]!r}",
    flush=True,
)

t0 = time.perf_counter()
with ThreadPoolExecutor(4) as ex:
    conc = list(ex.map(lambda _: stream_chat(ESSAY, 256), range(4)))
wall = time.perf_counter() - t0
tokens = sum(c["completion_tokens"] for c in conc)
result["concurrent_4"] = {
    "wall_s": round(wall, 2),
    "aggregate_tok_s": round(tokens / wall, 2),
    "per_stream_decode_tok_s": [c["decode_tok_s"] for c in conc],
    "ttft_s": [c["ttft_s"] for c in conc],
}
print(f"concurrent_4 wall={wall:.1f}s aggregate={tokens / wall:.1f} tok/s", flush=True)

functional = []
for prompt, mt in FUNCTIONAL:
    r = stream_chat(prompt, mt)
    functional.append(
        {
            "prompt": prompt[:40],
            "sha256": hashlib.sha256(r["text"].encode()).hexdigest()[:16],
            "text": r["text"][:200],
        }
    )
r = stream_chat(NEEDLE, 8)
functional.append(
    {
        "prompt": "needle-8k",
        "sha256": hashlib.sha256(r["text"].encode()).hexdigest()[:16],
        "text": r["text"][:40],
    }
)
result["functional"] = functional
print("functional hashes:", [f["sha256"] for f in functional], flush=True)

path = os.path.join(out_dir, f"{tag}.json")
with open(path, "w") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)
print("wrote", path)
