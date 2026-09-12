#!/usr/bin/env bash
# serve-frontend.sh -- the DeepSeek-V4.1-Flash API server without an engine (issue #24).
# Runs on FRONTEND_HOST, binds FRONTEND_ADDR:DP_RPC_PORT and waits for the rank 0
# engine (serve-node.sh 0 with FRONTEND_ADDR set) to handshake. Knobs in dsv41.env.
# Only the tokenizer, chat template and parsers are loaded here: no weights, no GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$HERE/dsv41.env"; set +a
source "$VENV/bin/activate"

[ -n "$FRONTEND_ADDR" ] || { echo "FRONTEND_ADDR is empty: nothing for the engine to dial" >&2; exit 2; }
test -f "$MODEL/config.json" || { echo "model missing at $MODEL" >&2; exit 3; }
test -f "$MODEL/tokenizer.json" || { echo "tokenizer missing at $MODEL" >&2; exit 3; }
test -f "$MODEL/encoding/encoding.py" || { echo "encoding missing at $MODEL" >&2; exit 3; }
# The frontend takes about 1 GiB next to a live TP rank (earlyoom line 2.43 GiB).
AVAIL_GIB=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
[ "$AVAIL_GIB" -ge 4 ] || { echo "MemAvailable ${AVAIL_GIB} GiB < 4 GiB: refusing" >&2; exit 4; }

export VLLM_ENGINE_READY_TIMEOUT_S=3600

# The served name plus the aliases of the replaced DS4F production (dsv41.env).
# shellcheck disable=SC2153 # SERVED_NAME and SERVED_ALIASES come from dsv41.env.
SERVED_NAMES=("$SERVED_NAME")
for _alias in ${SERVED_ALIASES:-}; do SERVED_NAMES+=("$_alias"); done

ARGS=(
  "$MODEL"
  --served-model-name "${SERVED_NAMES[@]}"
  --tensor-parallel-size "$FRONTEND_TP"
  --distributed-executor-backend mp
  --data-parallel-size 1
  --data-parallel-size-local 0
  --data-parallel-address "$FRONTEND_ADDR"
  --data-parallel-rpc-port "$DP_RPC_PORT"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --block-size 64
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --default-chat-template-kwargs "{\"thinking\":$THINKING}"
)
[ "$TEXT_ONLY" = "1" ] && ARGS+=(--language-model-only)
# The engine's OffloadingConnector reports kv_connector_stats; the frontend's
# loggers decode them with the connector class and build their Prometheus
# metric definitions from the same extra config (spec and tiers), so pass the
# engine's --kv-transfer-config verbatim (no --kv-offloading-size: no store here).
if [ "${KVOFF_GIB:-0}" != "0" ]; then
  source "$HERE/kv_transfer_json.sh"
  [ -n "$KV_TRANSFER_JSON" ] || KV_TRANSFER_JSON='{"kv_connector":"OffloadingConnector","kv_role":"kv_both"}'
  ARGS+=(--kv-transfer-config "$KV_TRANSFER_JSON")
fi
# shellcheck disable=SC2206
[ -n "$FRONTEND_EXTRA_ARGS" ] && ARGS+=($FRONTEND_EXTRA_ARGS)

exec vllm serve "${ARGS[@]}" --host 0.0.0.0 --port "$PORT"
