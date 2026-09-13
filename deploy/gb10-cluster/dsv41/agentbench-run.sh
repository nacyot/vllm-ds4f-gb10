#!/usr/bin/env bash
# agentbench-run.sh <tag> <config-label> [base-url] [repeats]
# Total-time benchmark of the 2026-09-13 sweep: the same seeded 4-lane x 6-turn agent
# workload (12K start, 2-8K new tokens and 150-900 output tokens per turn) repeated,
# plus the head-of-line case and the logit probe. Run on the head against a server
# with no other clients; results append to ~/dsv41-prep/bench/casebench.jsonl.
set -u
T=${1:?tag}
C=${2:?config label}
B=${3:-http://127.0.0.1:8888}
R=${4:-2}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run() { python3 "$HERE/casebench.py" --base "$B" --config "$C" "$@"; sleep 5; }

run --mode solo --tag "$T-warm" --prefill-tokens 16000 >/dev/null
run --mode mixed --tag "$T-warm2" --prefill-tokens 6000 --decodes 2 >/dev/null
for i in $(seq 1 "$R"); do
  run --mode agent --tag "$T-AG4r$i" --lanes 4 --turns 6 --stagger 8 \
    --start-tokens 12000 --turn-tokens-min 2000 --turn-tokens-max 8000 \
    --gen-min 150 --gen-max 900 --work-seed w1
done
run --mode hol --tag "$T-HOL" --prefill-tokens 128000
run --mode logits --tag "$T-LOGS" --prefills 3 --prefill-tokens 16000 --decodes 2
echo "SEQUENCE DONE $T"
