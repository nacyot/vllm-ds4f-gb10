#!/usr/bin/env bash
# One foreground SSH call per recipe cell. No suite chaining.
set -uo pipefail
T=${1:?config required}
CELL=${2:?cell required}
case "$T" in i39-off|i39-pf|i39-split|i39-prod) ;; *) exit 2;; esac
case "$CELL" in warm|warm2|S128r1|S128r2|S128r3|S32|decode|S128r4) ;; *) exit 2;; esac
ssh -o BatchMode=yes -o ConnectTimeout=10 gx10-6040 bash -s -- "$T" "$CELL" <<'REMOTE'
set -uo pipefail
T=$1
CELL=$2
cd ~/sglang-cmp || exit
{
  printf 'CELL_START %s %s %s\n' "$(date -Is)" "$T" "$CELL"
  mem() { awk '/^MemAvailable:/{printf "%.6f\n", $2/1048576}' /proc/meminfo; }
  available=$(mem)
  printf 'MEM_BEFORE %s GiB\n' "$available"
  case "$CELL" in
    S128*)
      for attempt in 1 2 3; do
        awk -v v="$available" 'BEGIN {exit !(v>=3)}' && break
        sleep 60
        available=$(mem)
        printf 'MEM_RECHECK %s %s GiB\n' "$attempt" "$available"
      done
      if ! awk -v v="$available" 'BEGIN {exit !(v>=3)}'; then
        printf 'CELL_END %s %s %s SKIP low_memory\n' "$(date -Is)" "$T" "$CELL"
        exit 3
      fi;;
  esac
  cb() { python3 casebench.py --base http://127.0.0.1:8888 --config "$T" --out results/casebench.jsonl "$@"; }
  before_case=$(wc -l < results/casebench.jsonl)
  before_decode=$(wc -l < results/decode.jsonl)
  case "$CELL" in
    warm) cb --mode solo --tag "$T-warm" --prefill-tokens 16000;;
    warm2) cb --mode mixed --tag "$T-warm2" --prefill-tokens 6000 --decodes 2;;
    S128*) cb --mode solo --tag "$T-$CELL" --prefill-tokens 128000;;
    S32) cb --mode solo --tag "$T-S32" --prefill-tokens 32000;;
    decode) python3 decodebench.py --base http://127.0.0.1:8888 --tag "$T" --types prose,code --levels 1,2,4 --gen 256 --out results/decode.jsonl;;
  esac
  cell_rc=$?
  printf 'MEM_AFTER %s GiB\n' "$(mem)"
  printf 'NEW_ROWS case=%s decode=%s\n' "$(( $(wc -l < results/casebench.jsonl) - before_case ))" "$(( $(wc -l < results/decode.jsonl) - before_decode ))"
  health=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/health)
  attention=absent
  test ! -e ~/dsv41-prep/logs/dsv41-ATTENTION || attention=present
  printf 'CHECK health=%s attention=%s\n' "$health" "$attention"
  printf 'CELL_END %s %s %s rc=%s\n' "$(date -Is)" "$T" "$CELL" "$cell_rc"
  test "$health" = 200 && test "$attention" = absent || exit 4
  exit "$cell_rc"
} 2>&1 | tee -a results/i39.log
exit "${PIPESTATUS[0]}"
REMOTE
