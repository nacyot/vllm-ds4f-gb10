#!/usr/bin/env bash
# Head node (rank 0). Run this after the worker is up.
set -euo pipefail
source "$(dirname "$0")/serve-common.sh"
export VLLM_HOST_IP=$HEAD_IP
exec vllm serve "${ARGS[@]}" \
  --node-rank 0 \
  --host 0.0.0.0 \
  --port "$PORT"
