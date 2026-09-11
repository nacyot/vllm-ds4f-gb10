#!/usr/bin/env bash
# dsv41_ctl.sh start|stop|status|log [n]  -- run from a workstation with SSH to the four nodes.
# Starts DeepSeek-V4.1-Flash TP=4 (workers 3,2,1 first, then head 0) as the user unit
# `dsv41-serve` on each node, via deploy/gb10-cluster/dsv41/serve-node.sh in ~/vllm-dsv41.
# Needs bash >= 4 (associative arrays); on macOS use /opt/homebrew/bin/bash.
set -u
CMD=${1:-status}
declare -A RANK=( [gx10-6040]=0 [gx10-f323]=1 [gx10-37cc]=2 [gx10-27c4]=3 )
HEAD=gx10-6040; WORKERS=(gx10-27c4 gx10-37cc gx10-f323); ALL=(gx10-6040 gx10-f323 gx10-37cc gx10-27c4)
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10"
PRE='export XDG_RUNTIME_DIR=/run/user/$(id -u); mkdir -p ~/dsv41-prep/logs'
KNOBS=""; for k in MEM_TRACE ALLOC_CONF EMPTY_CACHE LOG_PARAM_BYTES GMU KVMEM MAXLEN SEQS MNBT EAGER CAPTURE_SIZES CAPTURE_SIZES_EXPLICIT CUDAGRAPH_MODE SPEC SPEC_K SPEC_DRAFT SPEC_REJECT SPEC_EXTRA KVOFF_GIB KVFS_DIR KV_RETENTION KV_RELAY KV_RELAY_WINDOW_MIB FI_WORKSPACE_MIB NCCL_LEAN KV_TRACE TEXT_ONLY THINKING ENGRAM_MMAP ENGRAM_THREADS ENGRAM_RELEASE ENGRAM_STATS LOAD_FORMAT INSTANTTENSOR_DRAFT_LOADER DSPARK_DRAFT_PRUNE DSPARK_STATE_DIGEST VLLM_INSTANTTENSOR_MEMAVAIL_MIN_GIB EXTRA_ARGS PORT; do
  v="${!k:-}"; [ -n "$v" ] && KNOBS="$KNOBS --setenv=$k=$v"; done

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
    for w in "${WORKERS[@]}"; do echo "== $w rank ${RANK[$w]}"; start_rank "$w"; done
    sleep 5; echo "== $HEAD rank 0"; start_rank "$HEAD"
    echo "dsv41 TP=4 launched (head $HEAD :${PORT:-8889})";;
  stop)
    for h in "${ALL[@]}"; do $SSH "nacyot@$h" "$PRE; systemctl --user stop dsv41-serve.service 2>/dev/null; pkill -TERM -f 'serve-node.sh|[v]llm serve.*DeepSeek-V4.1' 2>/dev/null; true"; done
    sleep 6; for h in "${ALL[@]}"; do $SSH "nacyot@$h" "pkill -KILL -f '[v]llm serve.*DeepSeek-V4.1|[V]LLM::|[E]ngineCore' 2>/dev/null; rm -f /dev/shm/sem.mp-* /dev/shm/psm_* /dev/shm/vllm_offload_*.mmap 2>/dev/null; true"; done
    echo "dsv41 stopped";;
  status)
    for h in "${ALL[@]}"; do printf "%s r%s: " "$h" "${RANK[$h]}"; $SSH "nacyot@$h" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemctl --user is-active dsv41-serve.service 2>/dev/null | tr '\n' ' '; free -g | awk 'NR==2{print \"used\",\$3,\"GiB avail\",\$7,\"GiB\"}'"; done
    echo "health: $($SSH "nacyot@$HEAD" "curl -s -m 5 -o /dev/null -w %{http_code} http://127.0.0.1:${PORT:-8889}/health" 2>/dev/null)";;
  log) $SSH "nacyot@${2:-$HEAD}" "tail -n ${3:-40} ~/dsv41-prep/logs/dsv41-r${RANK[${2:-$HEAD}]}.log";;
  *) echo "usage: dsv41_ctl.sh start|stop|status|log [host] [n]"; exit 2;;
esac
