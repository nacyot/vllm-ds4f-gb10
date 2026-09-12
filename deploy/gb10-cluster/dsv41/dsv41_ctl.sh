#!/usr/bin/env bash
# dsv41_ctl.sh start|stop|status|caps|log [n] -- run from a workstation with SSH to the four nodes.
# Starts DeepSeek-V4.1-Flash TP=4 (workers 3,2,1 first, then head 0) as the user unit
# `dsv41-serve` on each node, via deploy/gb10-cluster/dsv41/serve-node.sh in ~/vllm-dsv41.
# Needs bash >= 4 (associative arrays); on macOS use /opt/homebrew/bin/bash.
set -u
CMD=${1:-status}
declare -A RANK=( [gx10-6040]=0 [gx10-f323]=1 [gx10-37cc]=2 [gx10-27c4]=3 )
HEAD=gx10-6040; WORKERS=(gx10-27c4 gx10-37cc gx10-f323); ALL=(gx10-6040 gx10-f323 gx10-37cc gx10-27c4)
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10"
# shellcheck disable=SC2016 # Expand id on the remote node.
PRE='export XDG_RUNTIME_DIR=/run/user/$(id -u); mkdir -p ~/dsv41-prep/logs'
KNOBS=""; for k in MEM_TRACE ALLOC_CONF EMPTY_CACHE EMPTY_CACHE_MIN_TOKENS LOG_PARAM_BYTES GMU KVMEM MAXLEN SEQS MNBT EAGER CAPTURE_SIZES CAPTURE_SIZES_EXPLICIT CUDAGRAPH_MODE SPEC SPEC_K SPEC_DRAFT SPEC_REJECT SPEC_EXTRA KVOFF_GIB KVOFF_SLABS KVOFF_SLAB_SHARES KVFS_DIR KV_RETENTION KV_RELAY KV_RELAY_WINDOW_MIB FI_WORKSPACE_MIB NCCL_LEAN NCCL_MAX_NCHANNELS NCCL_BUFFSIZE NCCL_LL128_BUFFSIZE NCCL_PROTO KV_TRACE TEXT_ONLY THINKING ENGRAM_MMAP ENGRAM_THREADS ENGRAM_RELEASE ENGRAM_PREFETCH ENGRAM_STATS LOAD_FORMAT INSTANTTENSOR_DRAFT_LOADER DSPARK_DRAFT_PRUNE DSPARK_STATE_DIGEST VLLM_INSTANTTENSOR_MEMAVAIL_MIN_GIB MOE_BACKEND LINEAR_BACKEND EP PROFILER_DIR EXTRA_ARGS PORT; do
  v="${!k:-}"; [ -n "$v" ] && KNOBS="$KNOBS --setenv=$k=$v"; done

cap_probe() { # host
  local h=$1 raw row
  if [ -n "${DSV41_CAP_FIXTURE:-}" ]; then
    row=$(awk -v host="$h" '
      $1 == host { count++; if (NF == 4) row = $0 }
      END { if (count == 1 && row != "") print row; else exit 1 }
    ' "$DSV41_CAP_FIXTURE") || row="$h unreachable - -"
    printf '%s\n' "$row"
    return
  fi
  if ! raw=$($SSH "nacyot@$h" '
    systemctl is-active gpu-clock-cap.service
    nvidia-smi --query-gpu=persistence_mode --format=csv,noheader || exit 1
    for i in 1 2 3 4 5; do
      nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits || exit 1
      sleep 0.2
    done
  ' 2>&1); then
    printf '%s unreachable - -\n' "$h"
    return
  fi
  printf '%s\n' "$raw" | awk -v host="$h" '
    /setlocale/ { next }
    { gsub(/^[[:space:]]+|[[:space:]]+$/, ""); n++ }
    n == 1 { state = $0; next }
    n == 2 { persistence = $0; next }
    { if ($0 !~ /^[0-9]+$/) bad = 1; if ($0 + 0 > max) max = $0 + 0 }
    END {
      if (n != 7 || bad) print host, "unknown", "-", "-"
      else print host, state, persistence, max + 0
    }
  '
}

check_caps() {
  local h host state persistence mhz extra limit=${CAP_MHZ:-2000}
  local failed=()
  if [[ ! "$limit" =~ ^[0-9]+$ ]]; then
    echo "Invalid CAP_MHZ: expected a nonnegative integer." >&2
    return 3
  fi
  printf '%-12s %-12s %-12s %s\n' HOST SERVICE PERSISTENCE MAX_SM_MHZ
  for h in "${ALL[@]}"; do
    read -r host state persistence mhz extra <<< "$(cap_probe "$h")"
    printf '%-12s %-12s %-12s %s\n' "$h" "$state" "$persistence" "$mhz"
    if [ "$host" != "$h" ] || [ -n "$extra" ] ||
       [ "$state" != active ] || [ "$persistence" != Enabled ] ||
       [[ ! "$mhz" =~ ^[0-9]+$ ]] ||
       ! awk -v mhz="$mhz" -v limit="$limit" 'BEGIN { exit !(mhz <= limit) }'; then
      failed+=("$h")
    fi
  done
  if [ "${#failed[@]}" -ne 0 ]; then
    echo "GPU clock cap released on ${failed[*]}: restore with deploy/gb10-cluster/dsv41/clock_ctl.sh cap (= sudo systemctl restart gpu-clock-cap.service on each node), then retry. Owner override: SKIP_CAP_CHECK=1." >&2
    return 3
  fi
  return 0
}

start_rank() { # host
  local h=$1 r=${RANK[$1]}
  $SSH "nacyot@$h" "$PRE; systemctl --user stop dsv41-serve.service 2>/dev/null; rm -f /dev/shm/sem.mp-* /dev/shm/psm_* 2>/dev/null;
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true;
    systemd-run --user --collect --unit=dsv41-serve -p LimitNOFILE=65536 $KNOBS \
      -p StandardOutput=append:/home/nacyot/dsv41-prep/logs/dsv41-r${r}.log -p StandardError=append:/home/nacyot/dsv41-prep/logs/dsv41-r${r}.log \
      /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/serve-node.sh ${r} 2>&1 | tail -1"
}
case $CMD in
  start)
    [ "${SKIP_CAP_CHECK:-0}" = "1" ] || check_caps || exit 3
    for w in "${WORKERS[@]}"; do echo "== $w rank ${RANK[$w]}"; start_rank "$w"; done
    sleep 5; echo "== $HEAD rank 0"; start_rank "$HEAD"
    echo "dsv41 TP=4 launched (head $HEAD :${PORT:-8889})";;
  stop)
    for h in "${ALL[@]}"; do $SSH "nacyot@$h" "$PRE; systemctl --user stop dsv41-serve.service 2>/dev/null; pkill -TERM -f 'serve-node.sh|[v]llm serve.*DeepSeek-V4.1' 2>/dev/null; true"; done
    sleep 6; for h in "${ALL[@]}"; do $SSH "nacyot@$h" "pkill -KILL -f '[v]llm serve.*DeepSeek-V4.1|[V]LLM::|[E]ngineCore' 2>/dev/null; rm -f /dev/shm/sem.mp-* /dev/shm/psm_* /dev/shm/vllm_offload_*.mmap 2>/dev/null; true"; done
    echo "dsv41 stopped";;
  caps) check_caps;;
  status)
    for h in "${ALL[@]}"; do
      printf "%s r%s: " "$h" "${RANK[$h]}"
      $SSH "nacyot@$h" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemctl --user is-active dsv41-serve.service 2>/dev/null | tr '\n' ' '; free -g | awk 'NR==2{printf \"used %s GiB avail %s GiB \",\$3,\$7}'"
      read -r host state persistence mhz <<< "$(cap_probe "$h")"
      printf 'cap %s %sMHz\n' "$state" "$mhz"
    done
    echo "health: $($SSH "nacyot@$HEAD" "curl -s -m 5 -o /dev/null -w %{http_code} http://127.0.0.1:${PORT:-8889}/health" 2>/dev/null)";;
  log) $SSH "nacyot@${2:-$HEAD}" "tail -n ${3:-40} ~/dsv41-prep/logs/dsv41-r${RANK[${2:-$HEAD}]}.log";;
  *) echo "usage: dsv41_ctl.sh start|stop|status|caps|log [host] [n]"; exit 2;;
esac
