#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Category bench for the V4.1 server: Tech2Wild's public prompt set v1
(bench_prompts_v1.json, from github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark,
same prompts so the numbers compare) plus a pi-style set (Korean chat, code, a
tool-call round trip, an 8k needle). Streaming, temperature 0, thinking off.

Per cell (category x concurrency C): decode tok/s per stream = tokens after the
first / time after the first token, aggregate = all tokens / wall, TTFT = first
token delta. Token counts come from the usage block (DSpark packs several
tokens per chunk). Speculative-decoding counters are read from /metrics before
and after the whole run. Greedy outputs of the pi set are hashed.

Usage: bench2.py <tag> [--base URL] [--levels 1,4] [--out DIR]
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--base", default="http://127.0.0.1:8888")
ap.add_argument("--levels", default="1,4")
ap.add_argument("--out", default=os.path.expanduser("~/dsv41-prep/bench"))
ap.add_argument("--skip-public", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
base = args.base
model = json.load(urllib.request.urlopen(base + "/v1/models", timeout=30))["data"][0][
    "id"
]
with open(os.path.join(HERE, "bench_prompts_v1.json")) as f:
    prompts = json.load(f)


def post(path, body, timeout=3600):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(messages, max_tokens, tools=None):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    t0 = time.perf_counter()
    first = None
    text = []
    tool_calls = {}
    usage = {}
    with post("/v1/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                delta = c.get("delta", {})
                if (delta.get("content") or delta.get("tool_calls")) and first is None:
                    first = time.perf_counter()
                if delta.get("content"):
                    text.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(
                        tc.get("index", 0), {"name": "", "arguments": ""}
                    )
                    fn = tc.get("function", {})
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
    end = time.perf_counter()
    n = usage.get("completion_tokens", 0)
    ttft = (first or end) - t0
    dec = end - (first or end)
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": n,
        "ttft_s": round(ttft, 3),
        "total_s": round(end - t0, 3),
        "decode_tok_s": round((n - 1) / dec, 2) if dec > 0 and n > 1 else None,
        "text": "".join(text),
        "tool_calls": [tool_calls[k] for k in sorted(tool_calls)],
    }


def spec_counters():
    out = {}
    try:
        for line in urllib.request.urlopen(base + "/metrics", timeout=10):
            s = line.decode()
            if s.startswith("vllm:spec_decode_num_") and "_total{" in s:
                key = s.split("{")[0].removeprefix("vllm:spec_decode_num_")
                val = float(s.rsplit(" ", 1)[1])
                if "position=" in s:
                    key += "_pos" + s.split('position="')[1].split('"')[0]
                out[key] = out.get(key, 0.0) + val
    except Exception:
        pass
    return out


def run_cell(prompt, max_tokens, c):
    with ThreadPoolExecutor(c) as ex:
        t0 = time.perf_counter()
        rs = list(
            ex.map(
                lambda _: stream_chat(
                    [{"role": "user", "content": prompt}], max_tokens
                ),
                range(c),
            )
        )
        wall = time.perf_counter() - t0
    toks = sum(r["completion_tokens"] for r in rs)
    decs = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
    return {
        "c": c,
        "agg_tok_s": round(toks / wall, 2),
        "per_stream_tok_s": round(statistics.mean(decs), 2) if decs else None,
        "ttft_mean_s": round(statistics.mean(r["ttft_s"] for r in rs), 3),
        "tokens": toks,
        "wall_s": round(wall, 2),
    }


result = {
    "tag": args.tag,
    "model": model,
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "batches": [],
}
spec0 = spec_counters()
print("model:", model, "tag:", args.tag, flush=True)

# warmup
stream_chat([{"role": "user", "content": "Say hello in one sentence."}], 32)

levels = [int(x) for x in args.levels.split(",")]
if not args.skip_public:
    cats = prompts["categories"] + [prompts["ceiling"]]
    for c in levels:
        for cat in cats:
            b = run_cell(cat["prompt"], cat["max_tokens"], c)
            b["category"] = cat["name"]
            result["batches"].append(b)
            print(
                f"  C{c} {cat['name']:14s} agg {b['agg_tok_s']:7.2f} "
                f"per-stream {b['per_stream_tok_s']} ttft {b['ttft_mean_s']}s",
                flush=True,
            )
    for c in levels:
        rows = [
            b
            for b in result["batches"]
            if b["c"] == c and b["category"] != "ceiling_count"
        ]
        agg_mean = statistics.mean(b["agg_tok_s"] for b in rows)
        per_mean = statistics.mean(
            b["per_stream_tok_s"] for b in rows if b["per_stream_tok_s"]
        )
        print(
            f"C{c} mean over 8 categories: agg {agg_mean:.2f} tok/s, "
            f"per-stream {per_mean:.2f} tok/s",
            flush=True,
        )

# pi-style set (single stream), hashed
pi = []


def add(name, r, ok=None):
    pi.append(
        {
            "name": name,
            "sha256": hashlib.sha256(
                (r["text"] + json.dumps(r["tool_calls"], sort_keys=True)).encode()
            ).hexdigest()[:16],
            "decode_tok_s": r["decode_tok_s"],
            "ttft_s": r["ttft_s"],
            "completion_tokens": r["completion_tokens"],
            "ok": ok,
            "text": r["text"][:160],
            "tool_calls": r["tool_calls"],
        }
    )
    print(
        f"  pi {name:12s} decode {r['decode_tok_s']} tok/s ttft {r['ttft_s']}s "
        f"ok={ok} :: {r['text'][:60]!r} {r['tool_calls'] if r['tool_calls'] else ''}",
        flush=True,
    )


r = stream_chat(
    [
        {
            "role": "user",
            "content": "한국어로 답해 주세요. 부산에 처음 가는 친구에게 "
            "하루 일정을 추천해 주세요. 다섯 문장 이내로.",
        }
    ],
    220,
)
add("ko-chat", r, ok=any("가" <= ch <= "힣" for ch in r["text"]))
r = stream_chat(
    [
        {
            "role": "user",
            "content": "다음 파이썬 함수의 버그를 찾아 고친 코드를 보여 주세요. "
            "설명은 한 줄만.\n\ndef mean(xs):\n"
            "    return sum(xs) / len(xs) if xs else None\n\nprint(mean([]) + 1)",
        }
    ],
    220,
)
add("ko-code", r, ok="def" in r["text"])
r = stream_chat(
    [
        {
            "role": "user",
            "content": "Write a Python function is_prime(n) with a short docstring. "
            "Code only.",
        }
    ],
    200,
)
add("code", r, ok="def is_prime" in r["text"])
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
            },
        },
    }
]
r1 = stream_chat(
    [
        {
            "role": "user",
            "content": "What is the weather in Paris in celsius? Use the tool.",
        }
    ],
    120,
    tools=tools,
)
call_ok = (
    bool(r1["tool_calls"])
    and r1["tool_calls"][0]["name"] == "get_weather"
    and "Paris" in r1["tool_calls"][0]["arguments"]
)
add("tool-call", r1, ok=call_ok)
if call_ok:
    args_json = r1["tool_calls"][0]["arguments"]
    msgs = [
        {
            "role": "user",
            "content": "What is the weather in Paris in celsius? Use the tool.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": args_json},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {"city": "Paris", "temp_c": 18, "condition": "light rain"}
            ),
        },
    ]
    r2 = stream_chat(msgs, 120, tools=tools)
    add("tool-roundtrip", r2, ok=("18" in r2["text"]))
facts = "\n".join(f"Fact {i}: item {i} has value {i * 7 % 101}." for i in range(1, 700))
r = stream_chat(
    [
        {
            "role": "user",
            "content": "The following is a numbered list of facts.\n"
            + facts
            + "\nQuestion: what is the value of item 512? Answer with the number only.",
        }
    ],
    8,
)
add("needle-8k", r, ok=str(512 * 7 % 101) in r["text"])
result["pi"] = pi

spec1 = spec_counters()
result["spec_delta"] = {
    k: spec1.get(k, 0) - spec0.get(k, 0) for k in set(spec0) | set(spec1)
}
if result["spec_delta"].get("draft_tokens", 0):
    d = result["spec_delta"]
    print(
        f"spec: drafts {d.get('drafts', 0):.0f} "
        f"draft_tokens {d.get('draft_tokens', 0):.0f} "
        f"accepted {d.get('accepted_tokens', 0):.0f} -> "
        f"{d.get('accepted_tokens', 0) / max(d.get('drafts', 1), 1):.2f} acc/draft",
        flush=True,
    )

path = os.path.join(args.out, f"{args.tag}.json")
with open(path, "w") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)
print("wrote", path)
