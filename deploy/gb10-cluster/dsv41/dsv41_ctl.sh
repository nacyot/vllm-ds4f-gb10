#!/usr/bin/env bash
# dsv41_ctl.sh start|stop|install|shm [host]|status|caps|headroom [min_gib]|frontend|log [host|frontend] [n] -- run from a workstation with SSH to the four nodes.
# Serves DeepSeek-V4.1-Flash TP=4 through the persistent user unit `dsv41-serve` on each node
# (systemd/dsv41-serve.service, put in place by `install`). `start` writes the knobs to the head's
# ~/dsv41-prep/dsv41-override.env and restarts the head unit; its pre-start copies that file to the
# workers and restarts their units (ranks 3, 2, 1) before rank 0 boots (dsv41_unit.sh).
# With FRONTEND_HOST/FRONTEND_ADDR set (issue #24) the API server runs on FRONTEND_HOST
# as the transient unit `dsv41-frontend` (serve-frontend.sh) and rank 0 boots headless.
# Needs bash >= 4 (associative arrays); on macOS use /opt/homebrew/bin/bash.
set -u
CMD=${1:-status}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=dsv41_lib.sh
source "$HERE/dsv41_lib.sh"
# shellcheck disable=SC2016 # Expand id on the remote node.
PRE='export XDG_RUNTIME_DIR=/run/user/$(id -u); mkdir -p ~/dsv41-prep/logs'
env_line() { # name value: one systemd EnvironmentFile line, value double-quoted
  local v=$2
  v=${v//\\/\\\\}; v=${v//\"/\\\"}; v=${v//\$/\\\$}; v=${v//\`/\\\`}
  printf '%s="%s"' "$1" "$v"
}
KNOBS=""; OVERRIDE=""; for k in FRONTEND_HOST FRONTEND_ADDR DP_RPC_PORT FRONTEND_TP FRONTEND_EXTRA_ARGS MEM_TRACE ALLOC_CONF EMPTY_CACHE EMPTY_CACHE_MIN_TOKENS LOG_PARAM_BYTES GMU KVMEM MAXLEN SEQS MNBT LPTT LPTT_MIXED PPCAP PPCAP_LONG_TOKENS DECODE_STEPS SHORT_RESERVE EAGER CAPTURE_SIZES CAPTURE_SIZES_EXPLICIT CUDAGRAPH_MODE SPEC SPEC_K SPEC_DRAFT SPEC_REJECT SPEC_BLOCK_DROP SPEC_EXTRA KVOFF_GIB KVOFF_SLABS KVOFF_SLAB_SHARES KVFS_DIR KV_RETENTION KV_RELAY KV_RELAY_WINDOW_MIB FI_WORKSPACE_MIB NCCL_LEAN NCCL_MAX_NCHANNELS NCCL_BUFFSIZE NCCL_LL128_BUFFSIZE NCCL_PROTO KV_TRACE TEXT_ONLY MM_IMAGES THINKING ENGRAM_MMAP ENGRAM_THREADS ENGRAM_RELEASE ENGRAM_PREFETCH ENGRAM_DECODE_ASYNC ENGRAM_CHUNK_RUNS ENGRAM_STATS LOAD_FORMAT INSTANTTENSOR_DRAFT_LOADER DSPARK_DRAFT_PRUNE DSPARK_STATE_DIGEST VLLM_INSTANTTENSOR_MEMAVAIL_MIN_GIB MOE_BACKEND LINEAR_BACKEND EP DSV41_INDEXER_TP_SPLIT PROFILER_DIR EXTRA_ARGS PORT; do
  v="${!k:-}"; [ -n "$v" ] || continue
  KNOBS="$KNOBS --setenv=$k=$v"; OVERRIDE+="$(env_line "$k" "$v")"$'\n'; done
[ "${DSV41_SHM_DRYRUN:-0}" != 1 ] || OVERRIDE+='DSV41_SHM_DRYRUN="1"'$'\n'

write_override() { # host: replace ~/dsv41-prep/dsv41-override.env with $OVERRIDE
  local b64
  b64=$(printf '%s' "$OVERRIDE" | base64 | tr -d '\n')
  $SSH "nacyot@$1" "$PRE; printf %s '$b64' | base64 -d > ~/dsv41-prep/dsv41-override.env.new &&
    mv -f ~/dsv41-prep/dsv41-override.env.new ~/dsv41-prep/dsv41-override.env"
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

FE=${FRONTEND_HOST:-}
start_frontend() {
  $SSH "nacyot@$FE" "$PRE; systemctl --user stop dsv41-frontend.service 2>/dev/null;
    systemd-run --user --collect --unit=dsv41-frontend -p LimitNOFILE=65536 $KNOBS \
      -p StandardOutput=append:/home/nacyot/dsv41-prep/logs/dsv41-frontend.log -p StandardError=append:/home/nacyot/dsv41-prep/logs/dsv41-frontend.log \
      /home/nacyot/vllm-dsv41/deploy/gb10-cluster/dsv41/serve-frontend.sh 2>&1 | tail -1"
}
# shellcheck disable=SC2016 # Expand on the head.
WD_STATUS='export XDG_RUNTIME_DIR=/run/user/$(id -u)
  printf "%s, attention: %s\n" "$(systemctl --user is-active dsv41-watchdog.timer)" \
    "$(head -c 300 ~/dsv41-prep/logs/dsv41-ATTENTION 2>/dev/null || echo none)"'
case $CMD in
  start)
    [ "${SKIP_CAP_CHECK:-0}" = "1" ] || check_caps || exit 3
    if [ -n "$FE" ] && [ -z "${FRONTEND_ADDR:-}" ]; then echo "FRONTEND_HOST is set but FRONTEND_ADDR is empty" >&2; exit 2; fi
    echo "== $HEAD override: $(printf '%s' "$OVERRIDE" | awk 'END {print NR}') line(s)"
    write_override "$HEAD" || exit 1
    if [ -n "$FE" ]; then echo "== $FE frontend"; start_frontend; sleep 5; fi
    echo "== $HEAD rank 0 restart (its pre-start restarts ranks 3, 2, 1 first)"
    $SSH "nacyot@$HEAD" "$PRE; systemctl --user restart dsv41-serve.service" ||
      { echo "dsv41-serve start failed on $HEAD: see dsv41_ctl.sh log" >&2; exit 1; }
    if [ -n "$FE" ]; then echo "dsv41 TP=4 launched (frontend $FE :${PORT:-8888}, head $HEAD headless)"
    else echo "dsv41 TP=4 launched (head $HEAD :${PORT:-8888})"; fi;;
  install)
    for h in "${ALL[@]}"; do
      units=dsv41-serve.service
      [ "$h" != "$HEAD" ] || units="$units dsv41-warmup.service dsv41-watchdog.service dsv41-watchdog.timer"
      echo "== $h install $units"
      $SSH "nacyot@$h" "$PRE; case \$(systemctl --user show dsv41-serve.service -p FragmentPath --value) in
          /run/*) echo 'transient dsv41-serve still loaded: run dsv41_ctl.sh stop first' >&2; exit 3;;
        esac
        mkdir -p ~/.config/systemd/user && cd ~/vllm-dsv41/deploy/gb10-cluster/dsv41/systemd &&
        install -m 644 $units ~/.config/systemd/user/ && systemctl --user daemon-reload &&
        systemctl --user enable dsv41-serve.service" || exit 1
    done
    $SSH "nacyot@$HEAD" "$PRE; systemctl --user enable --now dsv41-watchdog.timer" || exit 1
    echo "dsv41 units installed and enabled (not started)";;
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
      read -r _ state _ mhz <<< "$(cap_probe "$h")"
      printf 'cap %s %sMHz\n' "$state" "$mhz"
    done
    API=$HEAD; LABEL=health
    if [ -n "$FE" ]; then
      API=$FE; LABEL="health (frontend $FE:${PORT:-8888})"
      printf "%s frontend: " "$FE"
      $SSH "nacyot@$FE" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemctl --user is-active dsv41-frontend.service 2>/dev/null" 2>/dev/null
    fi
    echo "$LABEL: $($SSH "nacyot@$API" "curl -s -m 5 -o /dev/null -w %{http_code} http://127.0.0.1:${PORT:-8888}/health" 2>/dev/null)"
    echo "watchdog: $($SSH "nacyot@$HEAD" "$WD_STATUS" 2>/dev/null)";;
  log)
    if [ "${2:-}" = frontend ]; then
      [ -n "$FE" ] || { echo "set FRONTEND_HOST" >&2; exit 2; }
      $SSH "nacyot@$FE" "tail -n ${3:-40} ~/dsv41-prep/logs/dsv41-frontend.log"
    else
      $SSH "nacyot@${2:-$HEAD}" "tail -n ${3:-40} ~/dsv41-prep/logs/dsv41-r${RANK[${2:-$HEAD}]}.log"
    fi;;
  *) echo "usage: dsv41_ctl.sh start|stop|install|shm [host]|status|caps|headroom [min_gib]|frontend|log [host|frontend] [n]"; exit 2;;
esac
