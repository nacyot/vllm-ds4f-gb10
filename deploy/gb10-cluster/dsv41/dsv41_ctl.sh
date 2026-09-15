#!/usr/bin/env bash
# dsv41_ctl.sh start|stop|shm [host]|status|caps|headroom [min_gib]|frontend|log [host|frontend] [n] -- run from a workstation with SSH to the four nodes.
# Starts DeepSeek-V4.1-Flash TP=4 (workers 3,2,1 first, then head 0) as the user unit
# `dsv41-serve` on each node, via deploy/gb10-cluster/dsv41/serve-node.sh in ~/vllm-dsv41.
# With FRONTEND_HOST/FRONTEND_ADDR set (issue #24) the API server runs on FRONTEND_HOST
# as the unit `dsv41-frontend` (serve-frontend.sh) and rank 0 boots headless.
# Needs bash >= 4 (associative arrays); on macOS use /opt/homebrew/bin/bash.
set -u
CMD=${1:-status}
declare -A RANK=( [gx10-6040]=0 [gx10-f323]=1 [gx10-37cc]=2 [gx10-27c4]=3 )
HEAD=gx10-6040; WORKERS=(gx10-27c4 gx10-37cc gx10-f323); ALL=(gx10-6040 gx10-f323 gx10-37cc gx10-27c4)
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10"
# shellcheck disable=SC2016 # Expand id on the remote node.
PRE='export XDG_RUNTIME_DIR=/run/user/$(id -u); mkdir -p ~/dsv41-prep/logs'
KNOBS=""; for k in FRONTEND_HOST FRONTEND_ADDR DP_RPC_PORT FRONTEND_TP FRONTEND_EXTRA_ARGS MEM_TRACE ALLOC_CONF EMPTY_CACHE EMPTY_CACHE_MIN_TOKENS LOG_PARAM_BYTES GMU KVMEM MAXLEN SEQS MNBT LPTT LPTT_MIXED PPCAP PPCAP_LONG_TOKENS DECODE_STEPS EAGER CAPTURE_SIZES CAPTURE_SIZES_EXPLICIT CUDAGRAPH_MODE SPEC SPEC_K SPEC_DRAFT SPEC_REJECT SPEC_BLOCK_DROP SPEC_EXTRA KVOFF_GIB KVOFF_SLABS KVOFF_SLAB_SHARES KVFS_DIR KV_RETENTION KV_RELAY KV_RELAY_WINDOW_MIB FI_WORKSPACE_MIB NCCL_LEAN NCCL_MAX_NCHANNELS NCCL_BUFFSIZE NCCL_LL128_BUFFSIZE NCCL_PROTO KV_TRACE TEXT_ONLY MM_IMAGES THINKING ENGRAM_MMAP ENGRAM_THREADS ENGRAM_RELEASE ENGRAM_PREFETCH ENGRAM_DECODE_ASYNC ENGRAM_CHUNK_RUNS ENGRAM_STATS LOAD_FORMAT INSTANTTENSOR_DRAFT_LOADER DSPARK_DRAFT_PRUNE DSPARK_STATE_DIGEST VLLM_INSTANTTENSOR_MEMAVAIL_MIN_GIB MOE_BACKEND LINEAR_BACKEND EP DSV41_INDEXER_TP_SPLIT PROFILER_DIR EXTRA_ARGS PORT; do
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

mem_probe() { # host
  local h=$1
  if [ -n "${DSV41_MEM_FIXTURE:-}" ]; then
    awk -v host="$h" '
      $1 == host { count++; if (NF == 2) value = $2 }
      END { if (count == 1 && value != "") print value; else exit 1 }
    ' "$DSV41_MEM_FIXTURE"
  else
    $SSH "nacyot@$h" "awk '/^MemAvailable:/{printf \"%.6f\\n\", \$2/1048576; found=1} END {exit !found}' /proc/meminfo"
  fi
}

check_headroom() {
  local limit=${1:-${MIN_AVAIL_GIB:-5.2}} h value head_value="" memory_failed=0
  if [[ ! "$limit" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid MIN_AVAIL_GIB: expected a nonnegative number." >&2
    return 3
  fi
  printf '%-12s %s\n' HOST MEMAVAIL_GIB
  for h in "${ALL[@]}"; do
    if ! value=$(mem_probe "$h") || [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      printf '%-12s %s\n' "$h" unknown
      if [ "$h" = "$HEAD" ]; then memory_failed=1; fi
    else
      awk -v host="$h" -v value="$value" 'BEGIN {printf "%-12s %.2f\n", host, value}'
      if [ "$h" = "$HEAD" ]; then head_value=$value; fi
    fi
  done
  if [ "$memory_failed" = 1 ] || ! awk -v value="$head_value" -v limit="$limit" 'BEGIN {exit !(value >= limit)}'; then
    echo "493K cold prefill needs $HEAD head MemAvailable ≥ $limit GiB (available ${head_value:-unknown}; measured floor = start − 2.1 GiB, earlyoom 2.43); restart the adopted configuration first" >&2
    return 3
  fi
}

shm_cleanup() { # Runs on the node; mode is list or clean.
  local mode=$1 state path resolved metadata current users rc uid cleanup_error=0
  local files=() identities=()
  uid=$(id -u) || return 1
  state=$(systemctl --user show dsv41-serve.service -p ActiveState --value) || return 1
  printf 'dsv41-serve: %s\n' "$state"
  command -v fuser >/dev/null || { echo 'Cannot inspect shm: fuser unavailable' >&2; return 1; }
  mapfile -d '' -t files < <(find /dev/shm -mindepth 1 -maxdepth 1 -type f -uid "$uid" \
    \( -name 'sem.mp-*' -o -name 'psm_*' -o -name 'vllm_offload_*.mmap' \) -print0)
  wait "$!" || return 1
  printf 'shm files: %s\n' "${#files[@]}"
  for path in "${files[@]}"; do
    metadata=$(stat -c '%d:%i:%u:%s' -- "$path") || return 1
    identities+=("$metadata")
    printf 'candidate: %q %s bytes\n' "$path" "${metadata##*:}"
  done
  if [ "$mode" = clean ]; then
    case "$state" in
      inactive|failed) ;;
      *) printf 'skip: dsv41-serve %s\n' "$state"; return 0;;
    esac
  fi
  local i
  for i in "${!files[@]}"; do
    path=${files[$i]}
    resolved=$(realpath -e -- "$path") || { cleanup_error=1; continue; }
    case "$path" in
      /dev/shm/sem.mp-*|/dev/shm/psm_*|/dev/shm/vllm_offload_*.mmap) ;;
      *) printf 'skip: unsafe path %q\n' "$path"; cleanup_error=1; continue;;
    esac
    if [ "$resolved" != "$path" ] || [ "${path%/*}" != /dev/shm ] ||
       [ -L "$path" ] || [ ! -f "$path" ] || [ ! -O "$path" ]; then
      printf 'skip: unsafe path %q\n' "$path"; cleanup_error=1; continue
    fi
    rc=0
    users=$(fuser "$path" 2>&1) || rc=$?
    if [ "$rc" -eq 0 ]; then
      printf 'skip: in use %q (%s)\n' "$path" "$users"; continue
    elif [ "$rc" -ne 1 ] || [ -n "$users" ]; then
      printf 'skip: cannot inspect %q (%s)\n' "$path" "$users"; cleanup_error=1; continue
    fi
    if [ "$mode" = list ]; then
      printf 'unused: %q\n' "$path"; continue
    fi
    state=$(systemctl --user show dsv41-serve.service -p ActiveState --value) || return 1
    case "$state" in
      inactive|failed) ;;
      *) printf 'skip: dsv41-serve %s\n' "$state"; return 0;;
    esac
    current=$(stat -c '%d:%i:%u:%s' -- "$path") || { cleanup_error=1; continue; }
    if [ -L "$path" ] || [ ! -f "$path" ] || [ ! -O "$path" ] ||
       [ "$current" != "${identities[$i]}" ]; then
      printf 'skip: changed file %q\n' "$path"; cleanup_error=1; continue
    fi
    if rm -f -- "$path"; then
      printf 'deleted: %q\n' "$path"
    else
      cleanup_error=1
    fi
  done
  return "$cleanup_error"
}

shm_remote() { # host, optional list mode
  local h=$1 mode=${2:-clean}
  [ "${DSV41_SHM_DRYRUN:-0}" != 1 ] || mode=list
  printf '== %s shm (%s)\n' "$h" "$mode"
  $SSH "nacyot@$h" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); $(declare -f shm_cleanup); shm_cleanup $mode"
}

FE=${FRONTEND_HOST:-}
start_frontend() {
  $SSH "nacyot@$FE" "$PRE; systemctl --user stop dsv41-frontend.service 2>/dev/null;
    systemd-run --user --collect --unit=dsv41-frontend -p LimitNOFILE=65536 $KNOBS \
      -p StandardOutput=append:/home/nacyot/dsv41-prep/logs/dsv41-frontend.log -p StandardError=append:/home/nacyot/dsv41-prep/logs/dsv41-frontend.log \
      /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/serve-frontend.sh 2>&1 | tail -1"
}
start_rank() { # host
  local h=$1 r=${RANK[$1]}
  $SSH "nacyot@$h" "$PRE; systemctl --user stop dsv41-serve.service 2>/dev/null; true" || return 1
  shm_remote "$h" || return 1
  $SSH "nacyot@$h" "$PRE;
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true;
    systemd-run --user --collect --unit=dsv41-serve -p LimitNOFILE=65536 $KNOBS \
      -p StandardOutput=append:/home/nacyot/dsv41-prep/logs/dsv41-r${r}.log -p StandardError=append:/home/nacyot/dsv41-prep/logs/dsv41-r${r}.log \
      /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/serve-node.sh ${r} 2>&1 | tail -1"
}
case $CMD in
  start)
    [ "${SKIP_CAP_CHECK:-0}" = "1" ] || check_caps || exit 3
    for w in "${WORKERS[@]}"; do echo "== $w rank ${RANK[$w]}"; start_rank "$w" || exit 1; done
    if [ -n "$FE" ]; then
      [ -n "${FRONTEND_ADDR:-}" ] || { echo "FRONTEND_HOST is set but FRONTEND_ADDR is empty" >&2; exit 2; }
      sleep 5; echo "== $FE frontend"; start_frontend
    fi
    sleep 5; echo "== $HEAD rank 0"; start_rank "$HEAD" || exit 1
    if [ -n "$FE" ]; then echo "dsv41 TP=4 launched (frontend $FE :${PORT:-8888}, head $HEAD headless)"
    else echo "dsv41 TP=4 launched (head $HEAD :${PORT:-8888})"; fi;;
  frontend)
    [ -n "$FE" ] && [ -n "${FRONTEND_ADDR:-}" ] || { echo "set FRONTEND_HOST and FRONTEND_ADDR" >&2; exit 2; }
    echo "== $FE frontend (restart)"; start_frontend;;
  stop)
    for h in "${ALL[@]}"; do $SSH "nacyot@$h" "$PRE; systemctl --user stop dsv41-serve.service dsv41-frontend.service 2>/dev/null; pkill -TERM -f 'serve-node.sh|[v]llm serve.*DeepSeek-V4.1' 2>/dev/null; true"; done
    sleep 6; cleanup_failed=0
    for h in "${ALL[@]}"; do
      $SSH "nacyot@$h" "pkill -KILL -f '[v]llm serve.*DeepSeek-V4.1|[V]LLM::|[E]ngineCore' 2>/dev/null; true" || cleanup_failed=1
      shm_remote "$h" || cleanup_failed=1
    done
    [ "$cleanup_failed" = 0 ] || exit 1
    echo "dsv41 stopped";;
  shm)
    hosts=("${ALL[@]}")
    if [ "$#" -gt 2 ]; then echo 'usage: dsv41_ctl.sh shm [host]' >&2; exit 2; fi
    if [ "$#" -eq 2 ]; then
      case "$2" in
        gx10-6040|gx10-f323|gx10-37cc|gx10-27c4) hosts=("$2");;
        *) echo "Unknown shm host: $2" >&2; exit 2;;
      esac
    fi
    cleanup_failed=0
    for h in "${hosts[@]}"; do shm_remote "$h" list || cleanup_failed=1; done
    exit "$cleanup_failed";;
  caps) check_caps;;
  headroom) check_headroom "${2:-${MIN_AVAIL_GIB:-5.2}}";;
  status)
    for h in "${ALL[@]}"; do
      printf "%s r%s: " "$h" "${RANK[$h]}"
      $SSH "nacyot@$h" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemctl --user is-active dsv41-serve.service 2>/dev/null | tr '\n' ' '; awk '/^MemTotal:/{total=\$2} /^MemAvailable:/{avail=\$2} END{printf \"used %.2f GiB avail %.2f GiB \",(total-avail)/1048576,avail/1048576}' /proc/meminfo"
      read -r host state persistence mhz <<< "$(cap_probe "$h")"
      printf 'cap %s %sMHz\n' "$state" "$mhz"
    done
    API=$HEAD; LABEL=health
    if [ -n "$FE" ]; then
      API=$FE; LABEL="health (frontend $FE:${PORT:-8888})"
      printf "%s frontend: " "$FE"
      $SSH "nacyot@$FE" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemctl --user is-active dsv41-frontend.service 2>/dev/null" 2>/dev/null
    fi
    echo "$LABEL: $($SSH "nacyot@$API" "curl -s -m 5 -o /dev/null -w %{http_code} http://127.0.0.1:${PORT:-8888}/health" 2>/dev/null)";;
  log)
    if [ "${2:-}" = frontend ]; then
      [ -n "$FE" ] || { echo "set FRONTEND_HOST" >&2; exit 2; }
      $SSH "nacyot@$FE" "tail -n ${3:-40} ~/dsv41-prep/logs/dsv41-frontend.log"
    else
      $SSH "nacyot@${2:-$HEAD}" "tail -n ${3:-40} ~/dsv41-prep/logs/dsv41-r${RANK[${2:-$HEAD}]}.log"
    fi;;
  *) echo "usage: dsv41_ctl.sh start|stop|shm [host]|status|caps|headroom [min_gib]|frontend|log [host|frontend] [n]"; exit 2;;
esac
