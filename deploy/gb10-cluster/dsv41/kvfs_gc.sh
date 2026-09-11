#!/usr/bin/env bash
# GC for the KV offload filesystem tier: drop block files older than TTL, then
# oldest-first until the tree is under MAX_GIB. Lost files become lookup misses
# and the tokens are recomputed (kv_load_failure_policy=recompute).
# Usage: kvfs_gc.sh <root_dir> [TTL_DAYS=7] [MAX_GIB=200] [--dry-run]
set -euo pipefail
ROOT=${1:?root_dir}; TTL=${2:-7}; MAX_GIB=${3:-200}; DRY=${4:-}
[ -d "$ROOT" ] || { echo "no such dir: $ROOT"; exit 0; }
run() { if [ -n "$DRY" ]; then echo "would: $*"; else "$@"; fi; }
n_ttl=$(find "$ROOT" -type f -name '*.bin' -mtime +"$TTL" | wc -l)
if [ "$n_ttl" -gt 0 ]; then
  find "$ROOT" -type f -name '*.bin' -mtime +"$TTL" -print0 | run xargs -0 rm -f --
fi
used=$(du -sb "$ROOT" | cut -f1); limit=$((MAX_GIB * 1024 * 1024 * 1024)); n_cap=0
if [ "$used" -gt "$limit" ]; then
  # oldest access first; %A@ = atime, %T@ = mtime fallback on noatime mounts
  while IFS= read -r -d '' f; do
    sz=$(stat -c %s "$f"); run rm -f -- "$f"; used=$((used - sz)); n_cap=$((n_cap + 1))
    [ "$used" -le "$limit" ] && break
  done < <(find "$ROOT" -type f -name '*.bin' -printf '%A@ %T@ %p\0' | sort -z -n | sed -z -E 's/^[^ ]+ [^ ]+ //')
fi
find "$ROOT" -type d -empty -delete 2>/dev/null || true
echo "kvfs_gc: root=$ROOT ttl_deleted=$n_ttl cap_deleted=$n_cap now=$(du -sh "$ROOT" | cut -f1) files=$(find "$ROOT" -type f -name '*.bin' | wc -l)"
