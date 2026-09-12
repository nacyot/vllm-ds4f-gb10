# Sourced by serve-node.sh and serve-frontend.sh: sets KV_TRANSFER_JSON, the
# --kv-transfer-config for the tiering offload (empty when KVOFF_GIB=0 or
# KVFS_DIR is empty). The frontend needs the same connector and extra config
# as the engine because its stats loggers build their metric definitions from it.
KV_TRANSFER_JSON=""
if [ "${KVOFF_GIB:-0}" != "0" ] && [ -n "${KVFS_DIR:-}" ]; then
  # cpu_bytes_to_use is merged in by VllmConfig from --kv-offloading-size.
  # relay_from_rank0: ranks live on 4 nodes, so only rank 0's CPU tier feeds the
  # fs tier; loads are broadcast from rank 0's GPU (MLA-style KV is TP-replicated).
  # cpu_slabs / cpu_slab_shares: per-group host row sizes (issue #2), see dsv41.env.
  SLABS=true; [ "${KVOFF_SLABS:-1}" = "0" ] && SLABS=false
  SLAB_SHARES=""
  if [ -n "${KVOFF_SLAB_SHARES:-}" ]; then
    # "139264:0.4,77824:0.4" (quotes do not survive --setenv) or ready JSON.
    case $KVOFF_SLAB_SHARES in
      \{*) SLAB_SHARES="\"cpu_slab_shares\":${KVOFF_SLAB_SHARES}," ;;
      *) SLAB_SHARES="\"cpu_slab_shares\":{$(printf "%s" "$KVOFF_SLAB_SHARES" | sed -E "s/([0-9]+):/\"\\1\":/g")}," ;;
    esac
  fi
  KV_TRANSFER_JSON="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"blocks_per_chunk\":1,\"cpu_slabs\":${SLABS},${SLAB_SHARES}\"relay_from_rank0\":${KV_RELAY:-true},\"relay_window_mib\":${KV_RELAY_WINDOW_MIB:-256},\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KVFS_DIR\",\"n_read_threads\":16,\"n_write_threads\":16}]}}"
fi
