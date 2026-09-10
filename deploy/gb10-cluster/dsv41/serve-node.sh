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
# FlashInfer JIT needs nvcc on PATH (vLLM's has_flashinfer() probes for it).
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a FLASHINFER_DISABLE_VERSION_CHECK=1
# A JIT build after the 78 GB of weights are resident has ~20 GiB to work with;
# each nvcc job takes 5-6 GiB (earlyoom killed 8-way builds twice).
export MAX_JOBS=${MAX_JOBS:-2}
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

ARGS=(
  "$MODEL"
  --served-model-name "$SERVED_NAME"
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
  ARGS+=(--engram-config "{\"mmap\":true,\"mmap_prefault_threads\":$ENGRAM_THREADS}")
else
  ARGS+=(--engram-config '{"cpu_offload":false}')
fi
[ "$TEXT_ONLY" = "1" ] && ARGS+=(--language-model-only)
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
[ "$SPEC" = "dspark" ] && ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\"}")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

if [ "$NODE_RANK" = "0" ]; then
  exec vllm serve "${ARGS[@]}" --host 0.0.0.0 --port "$PORT"
else
  exec vllm serve "${ARGS[@]}" --headless
fi
