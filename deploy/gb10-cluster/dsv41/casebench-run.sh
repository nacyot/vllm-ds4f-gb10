#!/usr/bin/env bash
# casebench-run.sh <tag> <config-label> [base-url]
# The case sequence of the 2026-09-13 mixed-prefill sweep (DS4F 2026-08-28 cases plus
# agent replay and the multi-prefill logit probe). Run on the head node against a
# server with no other clients; results append to ~/dsv41-prep/bench/casebench.jsonl.
set -u
T=${1:?tag}
C=${2:?config label}
B=${3:-http://127.0.0.1:8888}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run() { python3 "$HERE/casebench.py" --base "$B" --config "$C" "$@"; sleep 5; }

run --mode solo --tag "$T-warm" --prefill-tokens 32000 >/dev/null
run --mode mixed --tag "$T-warm2" --prefill-tokens 8000 --decodes 1 >/dev/null
run --mode solo --tag "$T-S8" --prefill-tokens 8000
run --mode solo --tag "$T-S32" --prefill-tokens 32000
run --mode solo --tag "$T-S128" --prefill-tokens 128000
run --mode mixed --tag "$T-M1" --prefill-tokens 32000 --decodes 1
run --mode mixed --tag "$T-M3" --prefill-tokens 32000 --decodes 3
run --mode mixed --tag "$T-M3x8" --prefill-tokens 8000 --decodes 3
run --mode mixed --tag "$T-M1x128" --prefill-tokens 128000 --decodes 1
run --mode multi --tag "$T-P2" --prefill-tokens 32000 --prefills 2 --decodes 0
run --mode multi --tag "$T-P4" --prefill-tokens 32000 --prefills 4 --decodes 0
run --mode hol --tag "$T-HOL" --prefill-tokens 128000
run --mode agent --tag "$T-AG3" --lanes 3 --turns 5
run --mode logits --tag "$T-LOG" --prefills 3 --prefill-tokens 16000 --decodes 2
echo "SEQUENCE DONE $T"
