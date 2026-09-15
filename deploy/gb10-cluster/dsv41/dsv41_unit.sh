#!/usr/bin/env bash
# dsv41_unit.sh pre|start|post -- the steps of the dsv41-serve user unit (systemd/dsv41-serve.service),
# the same file on all four nodes; the rank comes from the host name. Output is appended to
# ~/dsv41-prep/logs/dsv41-r<rank>.log.
#   pre    every rank: wait for the previous instance's memory to come back (MemAvailable >= 100 GiB,
#          the serve-node.sh gate), clean the shm candidates of this unit, drop caches. Rank 0 first
#          waits for /mnt/kvdisk when the KV disk tier lives there, checks the four GPU clock caps
#          (never changes them), then copies its override file to the workers and restarts their
#          units (ranks 3, 2, 1).
#   start  exec serve-node.sh <rank>.
#   post   rank 0 queues dsv41-warmup.service.
set -u
STEP=${1:-}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=dsv41_lib.sh
source "$HERE/dsv41_lib.sh"
case "$STEP" in pre|start|post) ;; *) echo 'usage: dsv41_unit.sh pre|start|post' >&2; exit 2;; esac
HOST=${DSV41_HOST:-$(hostname -s)}
R=${RANK[$HOST]:-}
[ -n "$R" ] || { echo "dsv41_unit.sh: $HOST is not a cluster node" >&2; exit 2; }
# shellcheck disable=SC2034 # Read by cap_probe (dsv41_lib.sh).
DSV41_LOCAL_HOST=$HOST
LOGDIR=${DSV41_LOG_DIR:-$HOME/dsv41-prep/logs}
OVERRIDE_FILE=${DSV41_OVERRIDE_FILE:-$HOME/dsv41-prep/dsv41-override.env}
MEMINFO=${DSV41_MEMINFO:-/proc/meminfo}
KVDISK=${DSV41_KVDISK:-/mnt/kvdisk}
mkdir -p "$LOGDIR"
exec >> "$LOGDIR/dsv41-r$R.log" 2>&1
say() { printf '%s [dsv41-serve r%s %s] %s\n' "$(date '+%F %T')" "$R" "$STEP" "$*"; }

wait_kvdisk() {
  local dir i
  # shellcheck source=dsv41.env
  dir=$(set -a; source "$HERE/dsv41.env" >/dev/null 2>&1; [ "${KVOFF_GIB:-0}" = 0 ] || printf '%s' "${KVFS_DIR:-}")
  case "$dir" in "$KVDISK"/*) ;; *) return 0;; esac
  for i in $(seq 1 36); do
    mountpoint -q "$KVDISK" && return 0
    say "waiting for $KVDISK ($i/36)"; sleep 5
  done
  say "ABORT: $KVDISK is not mounted (KV disk tier $dir)"; return 1
}

wait_caps() {
  local i
  for i in $(seq 1 12); do
    check_caps && return 0
    say "GPU clock cap check failed ($i/12)"; sleep 10
  done
  logger -t dsv41-serve -p user.err "GPU clock cap check failed: dsv41-serve rank 0 not started (owner recovery: clock_ctl.sh cap)" || true
  say "ABORT: GPU clock cap check failed"; return 1
}

wait_memory() {
  local i avail
  for i in $(seq 1 24); do
    avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' "$MEMINFO")
    [ "${avail:-0}" -lt 100 ] || return 0
    say "waiting for MemAvailable >= 100 GiB (${avail:-unknown} GiB, $i/24)"; sleep 5
  done
  say "ABORT: MemAvailable ${avail:-unknown} GiB < 100 GiB"; return 1
}

restart_workers() {
  local w b64="" tries=0
  [ ! -f "$OVERRIDE_FILE" ] || b64=$(base64 < "$OVERRIDE_FILE" | tr -d '\n')
  for w in "${WORKERS[@]}"; do
    until $SSH "nacyot@$w" true >/dev/null 2>&1; do
      tries=$((tries + 1))
      [ "$tries" -lt 48 ] || { say "ABORT: $w unreachable over ssh"; return 1; }
      say "waiting for $w ssh ($tries/48)"; sleep 10
    done
    say "restarting $w rank ${RANK[$w]}"
    $SSH "nacyot@$w" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); mkdir -p ~/dsv41-prep &&
      printf %s '$b64' | base64 -d > ~/dsv41-prep/dsv41-override.env.new &&
      mv -f ~/dsv41-prep/dsv41-override.env.new ~/dsv41-prep/dsv41-override.env || exit 1
      systemctl --user reset-failed dsv41-serve.service 2>/dev/null
      systemctl --user restart dsv41-serve.service && systemctl --user is-active dsv41-serve.service" ||
      { say "ABORT: $w dsv41-serve restart failed"; return 1; }
  done
}

case "$STEP" in
  pre)
    say "pre-start (invocation ${INVOCATION_ID:-none})"
    if [ "$R" = 0 ]; then
      wait_kvdisk || exit 1
      wait_caps || exit 1
    fi
    wait_memory || exit 1
    if [ "$R" = 0 ]; then restart_workers || exit 1; fi
    mode=pre; [ "${DSV41_SHM_DRYRUN:-0}" != 1 ] || mode=list
    shm_cleanup "$mode" || { say "ABORT: shm cleanup failed"; exit 1; }
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || say "drop_caches skipped (sudo -n)"
    [ "$R" != 0 ] || sleep 5
    say "pre-start done";;
  start)
    say "exec serve-node.sh $R"
    exec "$HERE/serve-node.sh" "$R";;
  post)
    if [ "$R" = 0 ]; then
      systemctl --user start --no-block dsv41-warmup.service || say "dsv41-warmup.service not queued"
    fi;;
esac
exit 0
