#!/usr/bin/env bash
# Worker node (rank 1). Run this first, then serve-head.sh on the head.
set -euo pipefail
source "$(dirname "$0")/serve-common.sh"
export VLLM_HOST_IP=$WORKER_IP
exec vllm serve "${ARGS[@]}" \
  --node-rank 1 \
  --headless
