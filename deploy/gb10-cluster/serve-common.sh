#!/usr/bin/env bash
# Shared env and serve args for the two-node GB10 launch.
# Sourced by serve-head.sh and serve-worker.sh. One option per line.

# --- machine-specific: edit these ---
HEAD_IP=${HEAD_IP:-10.0.0.1}        # head node IP on the fast link (also the master address)
WORKER_IP=${WORKER_IP:-10.0.0.2}    # worker node IP on the fast link
MASTER_PORT=${MASTER_PORT:-29501}
PORT=${PORT:-8888}                  # head OpenAI-compatible API port
NIC=${NIC:-eth0}                    # fast NIC interface name (NCCL/TP/GLOO)
HCA=${HCA:-}                        # RoCE HCA(s), e.g. rocep1s0f0 (empty = TCP over NIC)
CACHE_DIR=${CACHE_DIR:-/mnt/cache}  # KV cache dir: local NVMe, attached disk, or a mount
VENV=${VENV:-$HOME/vllm027-venv}    # venv with this fork installed
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}
CONFIG_FILE=${CONFIG_FILE:-$(dirname "${BASH_SOURCE[0]}")/ds4f.env}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-6}
# ------------------------------------

source "$VENV/bin/activate"

# load the recipe knobs (TRIAL_*, DSPARK_*, VLLM_*) into the environment
set -a
source "$CONFIG_FILE"
set +a

KV_LOAD_FAILURE_POLICY=${KV_LOAD_FAILURE_POLICY:-recompute}
TRIAL_NODE_HEADROOM_GATE=${TRIAL_NODE_HEADROOM_GATE:-false}
TRIAL_NODE_HEADROOM_RESERVE_BYTES=${TRIAL_NODE_HEADROOM_RESERVE_BYTES:-17179869184}
TRIAL_NODE_HEADROOM_J_BYTES=${TRIAL_NODE_HEADROOM_J_BYTES:-0}
TRIAL_NODE_HEADROOM_HEARTBEAT_MAX_AGE_MS=${TRIAL_NODE_HEADROOM_HEARTBEAT_MAX_AGE_MS:-30000}
TRIAL_BOUNDED_RESTORE_MAX_ROWS=${TRIAL_BOUNDED_RESTORE_MAX_ROWS:-0}
TRIAL_ENFORCE_EAGER=${TRIAL_ENFORCE_EAGER:-false}
TRIAL_MAX_CUDAGRAPH_CAPTURE_SIZE=${TRIAL_MAX_CUDAGRAPH_CAPTURE_SIZE:-0}

# per-node KV directory (head and worker use distinct subdirs; with the relay on,
# only the head actually stores and reads there)
export TRIAL_KVFS_DIR="$CACHE_DIR/kv/$(hostname)-compact"
mkdir -p "$TRIAL_KVFS_DIR"

# NCCL over the fast link
export NCCL_SOCKET_IFNAME=$NIC TP_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC
export NCCL_CROSS_NIC=1 NCCL_IB_GID_INDEX=3
[ -n "$HCA" ] && export NCCL_IB_HCA=$HCA

# serve args, identical on both nodes, one per line
ARGS=(
  "$MODEL"
  --served-model-name deepseek-v4-flash-0731
  --trust-remote-code
  --tensor-parallel-size 2
  --nnodes 2
  --master-addr "$HEAD_IP"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --kv-cache-dtype fp8_ds_mla
  --block-size 256
  --max-model-len "$TRIAL_MML"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$TRIAL_MNBT"
  --long-prefill-token-threshold 1024
  --gpu-memory-utilization "$TRIAL_GPUUTIL"
  --kv-cache-memory "$TRIAL_KVMEM"
  --kv-offloading-size "$TRIAL_KVOFF"
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"$KV_LOAD_FAILURE_POLICY\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"canonical_layout\":$TRIAL_KVCANON,\"dspark_compact_packed\":$TRIAL_KVCOMPACT,\"blocks_per_chunk\":$TRIAL_KVBPC,\"bounded_restore_max_rows\":$TRIAL_BOUNDED_RESTORE_MAX_ROWS,\"node_headroom_gate\":$TRIAL_NODE_HEADROOM_GATE,\"node_headroom_reserve_bytes\":$TRIAL_NODE_HEADROOM_RESERVE_BYTES,\"node_headroom_j_bytes\":$TRIAL_NODE_HEADROOM_J_BYTES,\"node_headroom_heartbeat_max_age_ms\":$TRIAL_NODE_HEADROOM_HEARTBEAT_MAX_AGE_MS,\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$TRIAL_KVFS_DIR\",\"n_read_threads\":$TRIAL_KVRT,\"n_write_threads\":8}]}}"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --enable-chunked-prefill
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --default-chat-template-kwargs "{\"thinking\":true,\"reasoning_effort\":\"$TRIAL_THINK\"}"
  --moe-backend "$TRIAL_MOE"
  --linear-backend "$TRIAL_LINEAR"
)

if [[ "${TRIAL_SPEC:-dspark}" != "none" ]]; then
  ARGS+=(
    --speculative-config "{\"method\":\"$TRIAL_SPEC\",\"num_speculative_tokens\":$TRIAL_SPEC_N}"
  )
fi

if [[ "$TRIAL_ENFORCE_EAGER" == "true" ]]; then
  ARGS+=(--enforce-eager)
fi

if (( TRIAL_MAX_CUDAGRAPH_CAPTURE_SIZE > 0 )); then
  ARGS+=(--max-cudagraph-capture-size "$TRIAL_MAX_CUDAGRAPH_CAPTURE_SIZE")
fi
