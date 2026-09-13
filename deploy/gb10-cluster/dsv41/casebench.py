#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed prefill/decode case benchmark for the V4.1 server (standard library only).

Ported from the DS4F prefill-quantum bench (.notes/2026-08-28-vllm-prefill-quantum/
bench2.py): exact decode token counts when the server streams usage, head
MemAvailable sampling, a multi-turn agent replay and a multi-prefill logit probe.

Modes:
  solo    one unique prefill (max_tokens 1)
  mixed   --decodes D streaming decode lanes, then one prefill
  multi   --prefills P concurrent prefills (optional --decodes D)
  hol     one big prefill; a short request fired 5 s in
  agent   --lanes L agent lanes x --turns T, staggered; each turn appends code
          and asks for --gen-tokens tokens (prefix cache carries the history)
  logits  first-token logprobs of a probe alone, then during P prefills + D lanes

Usage: casebench.py --mode MODE --tag TAG [--base URL] [--config LABEL] [--out FILE]
Appends one JSON line per run to --out and prints it.
"""

import argparse
import json
import os
import random
import threading
import time
import urllib.request

TOKENS_PER_HEX_WORD = 4.38  # V4.1 tokenizer, measured with /tokenize
CHARS_PER_TOKEN = 4.3  # used only when the stream carries no usage
BASE = "http://127.0.0.1:8888"
MODEL = ""


def post(path, payload, timeout=3600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def hex_text(rng, approx_tokens):
    n = int(approx_tokens / TOKENS_PER_HEX_WORD)
    return " ".join("".join(rng.choices("abcdef0123456789", k=6)) for _ in range(n))


def code_text(rng, approx_tokens, tag):
    parts, est, i = [], 0.0, 0
    while est < approx_tokens:
        kind = rng.choice(("process", "handle", "compute", "validate", "merge"))
        arg = rng.choice(("records", "payload", "batch", "items"))
        const = rng.randint(1000, 9999)
        fn = f"{kind}_{tag}_{i:03d}"
        block = (
            f"def {fn}({arg}, limit={const}):\n"
            f'    """{kind} step {i} for {tag}: drop entries above {const}."""\n'
            f"    out = []\n"
            f"    for entry in {arg}:\n"
            f'        key = str(entry.get("id", 0)) + "-{tag}-{i}"\n'
            f'        if len(key) % 7 == {i % 7} and entry.get("score", 0) < limit:\n'
            f'            out.append({{**entry, "tag": "{fn}"}})\n'
            f"    return out\n\n"
        )
        parts.append(block)
        est += len(block) / 3.0
        i += 1
    return "".join(parts)


def stream_chat(payload, record, stop=lambda: False):
    """Stream a chat completion; record(t, text, cumulative_completion_tokens)."""
    payload = dict(
        payload,
        stream=True,
        stream_options={"include_usage": True, "continuous_usage_stats": True},
    )
    resp = post("/v1/chat/completions", payload)
    usage = None
    for raw in resp:
        if stop():
            resp.close()
            break
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            obj = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        text = "".join(
            delta.get(k) or "" for k in ("content", "reasoning_content", "reasoning")
        )
        if text:
            record(time.time(), text, (usage or {}).get("completion_tokens"))
    return usage


class TokenEvents:
    def __init__(self):
        self.events = []  # (t, tokens)
        self.text = []
        self._cum = 0

    def __call__(self, t, text, cum):
        if cum is None:
            n = len(text) / CHARS_PER_TOKEN
        else:
            n, self._cum = cum - self._cum, cum
        self.events.append((t, n))
        self.text.append(text)


class DecodeLane(threading.Thread):
    def __init__(self, rng):
        super().__init__(daemon=True)
        self.ev = TokenEvents()
        self.err = None
        self.stop_flag = False
        self.topic = "topic-" + "".join(rng.choices("abcdefghij", k=8))

    def run(self):
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Write an extremely long, detailed, meandering essay "
                    f"about an imaginary research field called {self.topic}. "
                    "Keep going with new sections; do not stop early.",
                }
            ],
            "max_tokens": 16000,
            "temperature": 0.6,
            "chat_template_kwargs": {"thinking": False},
        }
        try:
            stream_chat(payload, self.ev, lambda: self.stop_flag)
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)

    @property
    def started(self):
        return len(self.ev.events) >= 3


def start_lanes(rng, n, rec):
    lanes = []
    for i in range(n):
        for attempt in range(3):
            lane = DecodeLane(rng)
            lane.start()
            deadline = time.time() + 120
            while time.time() < deadline and not lane.started and lane.is_alive():
                time.sleep(0.2)
            if lane.started:
                lanes.append(lane)
                break
            rec.setdefault("lane_retries", []).append(
                {"lane": i, "attempt": attempt, "err": lane.err}
            )
            lane.stop_flag = True
            time.sleep(5)
        else:
            return None
        time.sleep(0.5)
    return lanes


def window_stats(events, t0, t1):
    ev = [e for e in events if t0 <= e[0] <= t1]
    if len(ev) < 2 or t1 <= t0:
        return None
    gaps = sorted(b[0] - a[0] for a, b in zip(ev, ev[1:]))
    return {
        "chunks": len(ev),
        "tok_per_s": round(sum(n for _, n in ev) / (t1 - t0), 2),
        "mean_gap_s": round(sum(gaps) / len(gaps), 3),
        "p90_gap_s": round(gaps[int(len(gaps) * 0.9)], 3),
        "max_gap_s": round(gaps[-1], 3),
    }


def lane_summary(lanes, t0, t1, key):
    per = [window_stats(lane.ev.events, t0, t1) for lane in lanes]
    ok = [s for s in per if s]
    agg = None
    if ok:
        agg = {
            "lanes": len(ok),
            "sum_tok_per_s": round(sum(s["tok_per_s"] for s in ok), 2),
            "mean_gap_s": round(sum(s["mean_gap_s"] for s in ok) / len(ok), 3),
            "worst_max_gap_s": max(s["max_gap_s"] for s in ok),
        }
    return {key: agg, key + "_lanes": per}


class MemSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            self.samples.append(int(line.split()[1]) / 1048576)
                            break
            except OSError:
                return
            time.sleep(0.5)

    def summary(self):
        if not self.samples:
            return None
        return {
            "start_gib": round(self.samples[0], 2),
            "min_gib": round(min(self.samples), 2),
            "end_gib": round(self.samples[-1], 2),
        }


def metrics():
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        key = None
        if name.startswith("vllm:prompt_tokens_by_source_total{"):
            key = "prompt_" + name.split('source="')[1].split('"')[0]
        elif name.startswith("vllm:generation_tokens_total{"):
            key = "generation"
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_total{"):
            key = "accepted"
        elif name.startswith("vllm:spec_decode_num_draft_tokens_total{"):
            key = "drafted"
        elif name.startswith("vllm:num_preemptions_total{"):
            key = "preemptions"
        if key:
            out[key] = out.get(key, 0.0) + float(value)
    return out


def mdelta(a, b):
    return {k: round(b.get(k, 0.0) - a.get(k, 0.0), 1) for k in b}


class PrefillReq(threading.Thread):
    def __init__(self, rng, tag, approx_tokens, barrier):
        super().__init__(daemon=True)
        self.text = hex_text(rng, approx_tokens)
        self.tag = tag
        self.barrier = barrier
        self.result = {}

    def run(self):
        self.barrier.wait()
        t0 = time.time()
        try:
            body = json.loads(
                post(
                    "/v1/chat/completions",
                    {
                        "model": MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"[{self.tag}] Reply with just OK.\n\n"
                                + self.text,
                            }
                        ],
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "chat_template_kwargs": {"thinking": False},
                    },
                ).read()
            )
            t1 = time.time()
            ptoks = body.get("usage", {}).get("prompt_tokens")
            self.result = {
                "prompt_tokens": ptoks,
                "wall_s": round(t1 - t0, 2),
                "tok_per_s": round(ptoks / (t1 - t0), 1) if ptoks else None,
                "t0": t0,
                "t1": t1,
            }
        except Exception as e:  # noqa: BLE001
            self.result = {"error": repr(e)}


class AgentLane(threading.Thread):
    SYSTEM = (
        "You are a coding agent. The user pastes files as tool results; "
        "answer precisely and completely."
    )

    def __init__(self, idx, a):
        super().__init__(daemon=True)
        self.idx = idx
        self.a = a
        self.rng = random.Random(f"{a.tag}-agent-{idx}-{time.time()}")
        self.turns = []
        self.err = None

    def ask(self, tag, part, tokens):
        code = code_text(self.rng, tokens, f"{tag}{part}")
        return (
            f"Part {part} of module {tag}:\n\n```python\n{code}```\n"
            "Explain what each function in this part does, as a long numbered list."
        )

    def run(self):
        a = self.a
        time.sleep(self.idx * a.stagger)
        tag = "m" + "".join(self.rng.choices("abcdefghij", k=6))
        msgs = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": self.ask(tag, 0, a.start_tokens)},
        ]
        for turn in range(a.turns):
            if turn:
                msgs.append(
                    {"role": "user", "content": self.ask(tag, turn, a.turn_tokens)}
                )
            ev = TokenEvents()
            t0 = time.time()
            try:
                usage = stream_chat(
                    {
                        "model": MODEL,
                        "messages": msgs,
                        "max_tokens": a.gen_tokens,
                        "min_tokens": a.gen_tokens,
                        "ignore_eos": True,
                        "temperature": 0.6,
                        "chat_template_kwargs": {"thinking": False},
                    },
                    ev,
                )
            except Exception as e:  # noqa: BLE001
                self.err = repr(e)
                return
            t_end = time.time()
            stamps = [t for t, _ in ev.events]
            gaps = sorted(b - c for c, b in zip(stamps, stamps[1:]))
            ctoks = (usage or {}).get("completion_tokens") or 0
            dec_s = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
            self.turns.append(
                {
                    "turn": turn,
                    "prompt_tokens": (usage or {}).get("prompt_tokens"),
                    "completion_tokens": ctoks,
                    "ttft_s": round(stamps[0] - t0, 3) if stamps else None,
                    "decode_s": round(dec_s, 3),
                    "decode_tok_per_s": round((ctoks - 1) / dec_s, 2)
                    if dec_s
                    else None,
                    "max_gap_s": round(gaps[-1], 3) if gaps else None,
                    "gaps": [round(g, 3) for g in gaps if g >= 0.5],
                    "n_gaps": len(gaps),
                    "t0": t0,
                    "t_end": t_end,
                }
            )
            msgs.append({"role": "assistant", "content": "".join(ev.text)})


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * q))], 3) if xs else None


def run_agent(a, rec):
    lanes = [AgentLane(i, a) for i in range(a.lanes)]
    m0 = metrics()
    for lane in lanes:
        lane.start()
    for lane in lanes:
        lane.join()
    m1 = metrics()
    turns = [t for lane in lanes for t in lane.turns]
    rec["errors"] = [lane.err for lane in lanes if lane.err]
    if not turns:
        return
    wall = max(t["t_end"] for t in turns) - min(t["t0"] for t in turns)
    ctoks = sum(t["completion_tokens"] for t in turns)
    dec_s = sum(t["decode_s"] for t in turns)
    ttfts = [t["ttft_s"] for t in turns if t["ttft_s"] is not None]
    n_gaps = sum(t["n_gaps"] for t in turns)
    long_gaps = [g for t in turns for g in t["gaps"]]
    rec["agent"] = {
        "turns": len(turns),
        "wall_s": round(wall, 1),
        "completion_tokens": ctoks,
        "decode_tok_per_s_per_lane": round(ctoks / dec_s, 2) if dec_s else None,
        "ttft_mean_s": round(sum(ttfts) / len(ttfts), 2),
        "ttft_p50_s": pct(ttfts, 0.5),
        "ttft_p90_s": pct(ttfts, 0.9),
        "ttft_max_s": max(ttfts),
        "gaps_over_0_5s": len(long_gaps),
        "gap_share_over_0_5s": round(len(long_gaps) / n_gaps, 4) if n_gaps else None,
        "decode_time_in_gaps_over_0_5s": round(sum(long_gaps) / dec_s, 3)
        if dec_s
        else None,
        "metrics_delta": mdelta(m0, m1),
    }
    rec["agent_turns"] = [
        {k: v for k, v in t.items() if k not in ("gaps", "t0", "t_end")} for t in turns
    ]


def first_token(prompt):
    body = json.loads(
        post(
            "/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
                "chat_template_kwargs": {"thinking": False},
            },
        ).read()
    )
    lp = (body["choices"][0].get("logprobs") or {}).get("content") or []
    return (
        [(t["token"], round(t["logprob"], 3)) for t in lp[0]["top_logprobs"]]
        if lp
        else []
    )


def run_logits(a, rec, rng):
    body = code_text(random.Random(1234), 3000, "probe")
    instr = "\nReturn exactly 200 numbered lowercase English words, then stop."

    def probe():
        nonce = "".join(rng.choices("abcdefghij", k=10))
        t = time.time()
        top = first_token(f"[{nonce}]\n{body}{instr}")
        return {"t": round(t, 1), "lat_s": round(time.time() - t, 2), "top": top}

    rec["solo"] = [probe() for _ in range(3)]
    lanes = start_lanes(rng, a.decodes, rec) if a.decodes else []
    barrier = threading.Barrier(a.prefills)
    ps = [
        PrefillReq(rng, f"{a.tag}-p{i}", a.prefill_tokens, barrier)
        for i in range(a.prefills)
    ]
    for p in ps:
        p.start()
    under = []
    while any(p.is_alive() for p in ps):
        under.append(probe())
        time.sleep(1.0)
    for lane in lanes or []:
        lane.stop_flag = True
    ref = rec["solo"][0]["top"][0] if rec["solo"][0]["top"] else None
    worst = 0.0
    flips = 0
    for s in under:
        if not s["top"] or not ref:
            continue
        if s["top"][0][0] != ref[0]:
            flips += 1
        worst = max(worst, abs(s["top"][0][1] - ref[1]))
    rec["under_load"] = under
    rec["logits"] = {
        "samples": len(under),
        "argmax_flips": flips,
        "max_top1_logprob_delta": round(worst, 3),
        "ref": ref,
    }


def main():
    global BASE, MODEL
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=["solo", "mixed", "multi", "hol", "agent", "logits"],
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--base", default=BASE)
    ap.add_argument(
        "--out", default=os.path.expanduser("~/dsv41-prep/bench/casebench.jsonl")
    )
    ap.add_argument("--prefill-tokens", type=int, default=32000)
    ap.add_argument("--decodes", type=int, default=1)
    ap.add_argument("--prefills", type=int, default=2)
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--start-tokens", type=int, default=8000)
    ap.add_argument("--turn-tokens", type=int, default=6000)
    ap.add_argument("--gen-tokens", type=int, default=300)
    ap.add_argument("--stagger", type=float, default=15.0)
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    MODEL = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][
        0
    ]["id"]

    rng = random.Random(f"{a.tag}-{time.time()}")
    rec = {
        "tag": a.tag,
        "mode": a.mode,
        "config": a.config,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    mem = MemSampler()
    mem.start()
    lanes = []
    if a.mode in ("solo", "mixed", "multi"):
        if a.mode != "solo" and a.decodes > 0:
            lanes = start_lanes(rng, a.decodes, rec)
            if lanes is None:
                rec["error"] = "decode lanes failed to start"
            else:
                time.sleep(6)
        if "error" not in rec:
            n = a.prefills if a.mode == "multi" else 1
            barrier = threading.Barrier(n)
            ps = [
                PrefillReq(rng, f"{a.tag}-p{i}", a.prefill_tokens, barrier)
                for i in range(n)
            ]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
            rec["prefills"] = [p.result for p in ps]
            oks = [r for r in rec["prefills"] if r.get("tok_per_s")]
            if oks:
                t0 = min(r["t0"] for r in oks)
                t1 = max(r["t1"] for r in oks)
                total = sum(r["prompt_tokens"] for r in oks)
                rec["aggregate"] = {
                    "total_tokens": total,
                    "span_s": round(t1 - t0, 2),
                    "tok_per_s": round(total / (t1 - t0), 1),
                }
                if lanes:
                    time.sleep(3)
                    rec.update(lane_summary(lanes, t0 - 6, t0, "decode_before"))
                    rec.update(lane_summary(lanes, t0, t1, "decode_during"))
    elif a.mode == "hol":
        big = PrefillReq(rng, a.tag + "-big", a.prefill_tokens, threading.Barrier(1))
        big.start()
        time.sleep(5)
        t0 = time.time()
        try:
            post(
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"[{a.tag}-short] What is 17*23? "
                            "Answer with the number only.",
                        }
                    ],
                    "max_tokens": 16,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"thinking": False},
                },
            ).read()
            rec["short_wall_s"] = round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            rec["short_error"] = repr(e)
        big.join()
        rec["prefills"] = [big.result]
    elif a.mode == "agent":
        run_agent(a, rec)
    elif a.mode == "logits":
        run_logits(a, rec, rng)
    for lane in lanes or []:
        lane.stop_flag = True
    mem.stop_flag = True
    rec["head_mem"] = mem.summary()
    line = json.dumps(rec)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
