# shellcheck shell=bash disable=SC2034 # The node table is used by the scripts that source this file.
# dsv41_lib.sh -- sourced by dsv41_ctl.sh (workstation), dsv41_unit.sh and dsv41_watchdog.sh
# (nodes): the node table, the GPU clock cap check and the shared-memory cleanup.
declare -A RANK=( [gx10-6040]=0 [gx10-f323]=1 [gx10-37cc]=2 [gx10-27c4]=3 )
HEAD=gx10-6040; WORKERS=(gx10-27c4 gx10-37cc gx10-f323); ALL=(gx10-6040 gx10-f323 gx10-37cc gx10-27c4)
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10"
# A node sets this to its own name: its cap probe runs locally (a node cannot ssh to itself).
DSV41_LOCAL_HOST=${DSV41_LOCAL_HOST:-}
CAP_PROBE_CMD='
    systemctl is-active gpu-clock-cap.service
    nvidia-smi --query-gpu=persistence_mode --format=csv,noheader || exit 1
    for i in 1 2 3 4 5; do
      nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits || exit 1
      sleep 0.2
    done
  '

cap_probe() { # host
  local h=$1 raw row rc=0
  if [ -n "${DSV41_CAP_FIXTURE:-}" ]; then
    row=$(awk -v host="$h" '
      $1 == host { count++; if (NF == 4) row = $0 }
      END { if (count == 1 && row != "") print row; else exit 1 }
    ' "$DSV41_CAP_FIXTURE") || row="$h unreachable - -"
    printf '%s\n' "$row"
    return
  fi
  if [ "$h" = "$DSV41_LOCAL_HOST" ]; then
    raw=$(bash -c "$CAP_PROBE_CMD" 2>&1) || rc=$?
  else
    raw=$($SSH "nacyot@$h" "$CAP_PROBE_CMD" 2>&1) || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    printf '%s unreachable - -\n' "$h"
    return
  fi
  printf '%s\n' "$raw" | awk -v host="$h" '
    /setlocale/ { next }
    { gsub(/^[[:space:]]+|[[:space:]]+$/, ""); n++ }
    n == 1 { state = $0; next }
    n == 2 { persistence = $0; next }
    { if ($0 !~ /^[0-9]+$/) bad = 1; if ($0 + 0 > max) max = $0 + 0 }
    END {
      if (n != 7 || bad) print host, "unknown", "-", "-"
      else print host, state, persistence, max + 0
    }
  '
}

check_caps() {
  local h host state persistence mhz extra limit=${CAP_MHZ:-2000}
  local failed=()
  if [[ ! "$limit" =~ ^[0-9]+$ ]]; then
    echo "Invalid CAP_MHZ: expected a nonnegative integer." >&2
    return 3
  fi
  printf '%-12s %-12s %-12s %s\n' HOST SERVICE PERSISTENCE MAX_SM_MHZ
  for h in "${ALL[@]}"; do
    read -r host state persistence mhz extra <<< "$(cap_probe "$h")"
    printf '%-12s %-12s %-12s %s\n' "$h" "$state" "$persistence" "$mhz"
    if [ "$host" != "$h" ] || [ -n "$extra" ] ||
       [ "$state" != active ] || [ "$persistence" != Enabled ] ||
       [[ ! "$mhz" =~ ^[0-9]+$ ]] ||
       ! awk -v mhz="$mhz" -v limit="$limit" 'BEGIN { exit !(mhz <= limit) }'; then
      failed+=("$h")
    fi
  done
  if [ "${#failed[@]}" -ne 0 ]; then
    echo "GPU clock cap released on ${failed[*]}: restore with deploy/gb10-cluster/dsv41/clock_ctl.sh cap (= sudo systemctl restart gpu-clock-cap.service on each node), then retry. Owner override: SKIP_CAP_CHECK=1." >&2
    return 3
  fi
  return 0
}

# Runs on the node. list: report only. clean: delete while dsv41-serve is inactive or failed.
# pre: also while activating, when called from that unit's own ExecStartPre (same InvocationID).
shm_cleanup() {
  local mode=$1 state path resolved metadata current users rc uid cleanup_error=0
  local files=() identities=()
  _shm_state_ok() {
    case "$1" in
      inactive|failed) return 0;;
      activating)
        [ "$mode" = pre ] && [ -n "${INVOCATION_ID:-}" ] &&
          [ "$(systemctl --user show dsv41-serve.service -p InvocationID --value)" = "$INVOCATION_ID" ];;
      *) return 1;;
    esac
  }
  uid=$(id -u) || return 1
  state=$(systemctl --user show dsv41-serve.service -p ActiveState --value) || return 1
  printf 'dsv41-serve: %s\n' "$state"
  command -v fuser >/dev/null || { echo 'Cannot inspect shm: fuser unavailable' >&2; return 1; }
  mapfile -d '' -t files < <(find /dev/shm -mindepth 1 -maxdepth 1 -type f -uid "$uid" \
    \( -name 'sem.mp-*' -o -name 'psm_*' -o -name 'vllm_offload_*.mmap' \) -print0)
  wait "$!" || return 1
  printf 'shm files: %s\n' "${#files[@]}"
  for path in "${files[@]}"; do
    metadata=$(stat -c '%d:%i:%u:%s' -- "$path") || return 1
    identities+=("$metadata")
    printf 'candidate: %q %s bytes\n' "$path" "${metadata##*:}"
  done
  if [ "$mode" != list ] && ! _shm_state_ok "$state"; then
    printf 'skip: dsv41-serve %s\n' "$state"; return 0
  fi
  local i
  for i in "${!files[@]}"; do
    path=${files[$i]}
    resolved=$(realpath -e -- "$path") || { cleanup_error=1; continue; }
    case "$path" in
      /dev/shm/sem.mp-*|/dev/shm/psm_*|/dev/shm/vllm_offload_*.mmap) ;;
      *) printf 'skip: unsafe path %q\n' "$path"; cleanup_error=1; continue;;
    esac
    if [ "$resolved" != "$path" ] || [ "${path%/*}" != /dev/shm ] ||
       [ -L "$path" ] || [ ! -f "$path" ] || [ ! -O "$path" ]; then
      printf 'skip: unsafe path %q\n' "$path"; cleanup_error=1; continue
    fi
    rc=0
    users=$(fuser "$path" 2>&1) || rc=$?
    if [ "$rc" -eq 0 ]; then
      printf 'skip: in use %q (%s)\n' "$path" "$users"; continue
    elif [ "$rc" -ne 1 ] || [ -n "$users" ]; then
      printf 'skip: cannot inspect %q (%s)\n' "$path" "$users"; cleanup_error=1; continue
    fi
    if [ "$mode" = list ]; then
      printf 'unused: %q\n' "$path"; continue
    fi
    state=$(systemctl --user show dsv41-serve.service -p ActiveState --value) || return 1
    if ! _shm_state_ok "$state"; then
      printf 'skip: dsv41-serve %s\n' "$state"; return 0
    fi
    current=$(stat -c '%d:%i:%u:%s' -- "$path") || { cleanup_error=1; continue; }
    if [ -L "$path" ] || [ ! -f "$path" ] || [ ! -O "$path" ] ||
       [ "$current" != "${identities[$i]}" ]; then
      printf 'skip: changed file %q\n' "$path"; cleanup_error=1; continue
    fi
    if rm -f -- "$path"; then
      printf 'deleted: %q\n' "$path"
    else
      cleanup_error=1
    fi
  done
  return "$cleanup_error"
}

shm_remote() { # host, optional list mode
  local h=$1 mode=${2:-clean}
  [ "${DSV41_SHM_DRYRUN:-0}" != 1 ] || mode=list
  printf '== %s shm (%s)\n' "$h" "$mode"
  $SSH "nacyot@$h" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); $(declare -f shm_cleanup); shm_cleanup $mode"
}
