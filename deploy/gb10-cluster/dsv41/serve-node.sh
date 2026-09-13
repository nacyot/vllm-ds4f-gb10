#!/usr/bin/env bash
# serve-node.sh <node-rank>  -- DeepSeek-V4.1-Flash, TP=4 over the four GB10 nodes.
# Start ranks 1..3 first, then rank 0 (the head). Knobs in dsv41.env.
#
# The whole 121.6 GiB of each node is needed (about 77 GB of weights per rank plus
# activations and KV), so this refuses to start while anything else holds the memory:
# stop the production DS4F server first and restore it afterwards.
set -euo pipefail
NODE_RANK="${1:?usage: serve-node.sh <0|1|2|3>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$HERE/dsv41.env"; set +a
source "$VENV/bin/activate"

test -f "$MODEL/config.json" || { echo "model missing at $MODEL" >&2; exit 3; }
test -f "$MODEL/model-00048-of-00048.safetensors" || { echo "model incomplete at $MODEL" >&2; exit 3; }
test -f "$MODEL/model.safetensors.index.json" || { echo "index missing at $MODEL" >&2; exit 3; }
AVAIL_GIB=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL_GIB" -ge 100 ] || { echo "MemAvailable ${AVAIL_GIB} GiB < 100 GiB: production still up? refusing" >&2; exit 4; }

# fabric: same as the production DS4F launch
export VLLM_HOST_IP=$(ip -o -4 addr show "$NIC" | awk '{print $4}' | cut -d/ -f1)
export NCCL_SOCKET_IFNAME=$NIC TP_SOCKET_IFNAME=$NIC GLOO_SOCKET_IFNAME=$NIC
export NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=3 NCCL_IB_GID_AUTO=0 NCCL_CROSS_NIC=1
# NCCL_CUMEM_ENABLE=0 / UCX_MEM_MMAP_HOOK_MODE=none are community leak
# preventions for GB10 (x.com/Dragonomi/status/2092917630311313445), but
# NCCL_CUMEM_ENABLE=0 made the relay group creation fail with "Too many
# open files" on the 512K/KV 12 GiB boot; opt in explicitly if needed.
[ -n "${NCCL_CUMEM_ENABLE:-}" ] && export NCCL_CUMEM_ENABLE
[ -n "${UCX_MEM_MMAP_HOOK_MODE:-}" ] && export UCX_MEM_MMAP_HOOK_MODE
# NCCL_LEAN=1: MiaAI-Lab's DGX Spark TP=4 profile (default NCCL buffers measured
# at 4.7 GiB per node there, 0.14 GB with these); benchmark before adopting.
if [ "${NCCL_LEAN:-0}" = "1" ]; then
  export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576} NCCL_LL128_BUFFSIZE=${NCCL_LL128_BUFFSIZE:-262144}
  export NCCL_PROTO=${NCCL_PROTO:-^LL128} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}
fi
# FlashInfer JIT needs nvcc on PATH (vLLM's has_flashinfer() probes for it).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a FLASHINFER_DISABLE_VERSION_CHECK=1
# A JIT build after the 78 GB of weights are resident has ~20 GiB to work with;
# each nvcc job takes 5-6 GiB (earlyoom killed 8-way builds twice).
export MAX_JOBS=${MAX_JOBS:-2}
[ "${ENGRAM_STATS:-0}" = "1" ] && export VLLM_ENGRAM_MMAP_STATS=1
[ -n "${FI_WORKSPACE_MIB:-}" ] && export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((FI_WORKSPACE_MIB * 1024 * 1024))
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

# The served name plus the aliases of the replaced DS4F production (dsv41.env).
# shellcheck disable=SC2153 # SERVED_NAME and SERVED_ALIASES come from dsv41.env.
SERVED_NAMES=("$SERVED_NAME")
for _alias in ${SERVED_ALIASES:-}; do SERVED_NAMES+=("$_alias"); done

ARGS=(
  "$MODEL"
  --served-model-name "${SERVED_NAMES[@]}"
  --tensor-parallel-size 4
  --nnodes 4
  --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --load-format "$LOAD_FORMAT"
  --block-size 64
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --max-num-batched-tokens "$MNBT"
  --gpu-memory-utilization "$GMU"
  --kv-cache-memory "$KVMEM"
  --enable-prefix-caching
  --enable-chunked-prefill
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --default-chat-template-kwargs "{\"thinking\":$THINKING}"
)
if [ "$ENGRAM_MMAP" = "1" ]; then
  PREFETCH=false; [ "${ENGRAM_PREFETCH:-0}" = "1" ] && PREFETCH=true
  ARGS+=(--engram-config "{\"mmap\":true,\"mmap_prefault_threads\":$ENGRAM_THREADS,\"mmap_release_after_steps\":${ENGRAM_RELEASE:-3},\"mmap_prefetch_next_chunk\":$PREFETCH}")
else
  ARGS+=(--engram-config '{"cpu_offload":false}')
fi
[ -n "${LPTT:-}" ] && ARGS+=(--long-prefill-token-threshold "$LPTT")
[ -n "${LPTT_MIXED:-}" ] && export DSPARK_LPTT_MIXED="$LPTT_MIXED"
[ -n "${PPCAP:-}" ] && export DSPARK_PPCAP="$PPCAP"
[ -n "${PPCAP_LONG_TOKENS:-}" ] && export DSPARK_PPCAP_LONG_TOKENS="$PPCAP_LONG_TOKENS"
[ -n "${DECODE_STEPS:-}" ] && export DSPARK_DECODE_STEPS_PER_PREFILL="$DECODE_STEPS"
[ "$TEXT_ONLY" = "1" ] && ARGS+=(--language-model-only)
[ "$TEXT_ONLY" != "1" ] && [ -n "${MM_IMAGES:-}" ] && ARGS+=(--limit-mm-per-prompt "{\"image\":$MM_IMAGES}")
if [ "$EAGER" = "1" ]; then
  ARGS+=(--enforce-eager)
else
  SIZES="$CAPTURE_SIZES"
  if [ "$SPEC" = "dspark" ] && [ -z "$CAPTURE_SIZES_EXPLICIT" ]; then
    # The V2 runner only captures a FULL decode graph for token counts that are a
    # multiple of the verify query (k+1 per request); the draft runs k per request.
    # Cover every batch up to SEQS requests exactly (no padded rows), plus the
    # small non-spec sizes for prefill tails.
    K=$SPEC_K
    SIZES=$( { echo "$CAPTURE_SIZES" | tr , '\n'; seq "$K" "$K" $((K * SEQS)); seq $((K + 1)) $((K + 1)) $(((K + 1) * SEQS)); } | sort -n -u | paste -sd, - )
  fi
  CG="{\"cudagraph_capture_sizes\":[${SIZES}],\"max_cudagraph_capture_size\":${SIZES##*,}"
  [ -n "$CUDAGRAPH_MODE" ] && CG="$CG,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\""
  ARGS+=(--compilation-config "$CG}")
  export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}
fi
if [ "$SPEC" = "dspark" ]; then
  SC="{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"${SPEC_DRAFT:-probabilistic}\",\"rejection_sample_method\":\"${SPEC_REJECT:-standard}\""
  [ -n "$SPEC_EXTRA" ] && SC="$SC,$SPEC_EXTRA"
  ARGS+=(--speculative-config "$SC}")
fi
[ -n "${KV_RETENTION:-}" ] && export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=$KV_RETENTION
[ "${KV_TRACE:-0}" = "1" ] && export VLLM_KV_OFFLOAD_TRACE=1
AC=""
[ "${MEM_TRACE:-0}" = "1" ] && AC+='"mem_trace":true,'
[ -n "${ALLOC_CONF:-}" ] && AC+="\"cuda_alloc_conf\":\"$ALLOC_CONF\","
[ "${EMPTY_CACHE:-0}" = "1" ] && AC+='"empty_cache_after_prefill":true,'
[ "${EMPTY_CACHE:-0}" = "1" ] && [ "${EMPTY_CACHE_MIN_TOKENS:-0}" != "0" ] && AC+="\"empty_cache_min_prefill_tokens\":${EMPTY_CACHE_MIN_TOKENS},"
[ "${LOG_PARAM_BYTES:-0}" = "1" ] && AC+='"log_param_bytes":true,'
[ -n "$AC" ] && ARGS+=(--additional-config "{${AC%,}}")
# Kernel/parallel levers (issue #12). Dedicated knobs: JSON and flags do not
# survive EXTRA_ARGS through `systemd-run --setenv` and the word split below.
[ -n "${MOE_BACKEND:-}" ] && ARGS+=(--moe-backend "$MOE_BACKEND")
[ -n "${LINEAR_BACKEND:-}" ] && ARGS+=(--linear-backend "$LINEAR_BACKEND")
[ "${EP:-0}" = "1" ] && ARGS+=(--enable-expert-parallel)
if [ -n "${PROFILER_DIR:-}" ]; then
  # torch profiler armed at boot; /start_profile and /stop_profile on the head
  # drive it, every rank writes its own trace under PROFILER_DIR on its node.
  mkdir -p "$PROFILER_DIR"
  ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILER_DIR\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}")
fi
if [ "${KVOFF_GIB:-0}" != "0" ]; then
  ARGS+=(--kv-offloading-size "$KVOFF_GIB")
  if [ -n "$KVFS_DIR" ]; then
    mkdir -p "$KVFS_DIR"
    source "$HERE/kv_transfer_json.sh"
    ARGS+=(--kv-transfer-config "$KV_TRANSFER_JSON")
  fi
fi
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

if [ "$NODE_RANK" = "0" ] && [ -n "${FRONTEND_ADDR:-}" ]; then
  # Issue #24: the API server runs on FRONTEND_HOST (serve-frontend.sh); rank 0
  # boots the engine only and dials the frontend for the ZMQ handshake. With
  # DP=1 ParallelConfig takes the master IP from the env, not the flag
  # (vllm/config/parallel.py, the non-DP branch), so export it as well.
  export VLLM_DP_MASTER_IP="$FRONTEND_ADDR"
  exec vllm serve "${ARGS[@]}" --headless --data-parallel-address "$FRONTEND_ADDR" --data-parallel-rpc-port "$DP_RPC_PORT"
elif [ "$NODE_RANK" = "0" ]; then
  exec vllm serve "${ARGS[@]}" --host 0.0.0.0 --port "$PORT"
else
  exec vllm serve "${ARGS[@]}" --headless
fi
