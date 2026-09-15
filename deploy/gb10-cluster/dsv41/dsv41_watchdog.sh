#!/usr/bin/env bash
# dsv41_watchdog.sh -- one check of the V4.1F TP=4 server, run on the head every minute by
# dsv41-watchdog.timer. Restarts dsv41-serve here (its pre-start restarts the three workers) when
#   - a worker unit is not active, or started more than 30 s after the head unit: the TP group is
#     broken for certain (no cooldown, 12 such restarts in 2 h)
#   - a worker is unreachable 3 checks in a row, /health fails 3 checks in a row, or the 1-token
#     inference probe (every 5th check, then every check while it fails) fails twice in a row;
#     /health stays 200 while the engine waits on a dead rank in NCCL (15 min cooldown, 4 in 2 h)
# It never starts an inactive unit (stopped by the operator). A failed unit (start limit), a start
# that stays activating for 30 minutes and a used-up budget go to logger user.err and
# ~/dsv41-prep/logs/dsv41-ATTENTION instead. Pause it with `systemctl --user stop dsv41-watchdog.timer`.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=dsv41_lib.sh
source "$HERE/dsv41_lib.sh"
ST=${DSV41_WD_DIR:-$HOME/dsv41-prep/watchdog}
LOGDIR=${DSV41_LOG_DIR:-$HOME/dsv41-prep/logs}
GRACE=${DSV41_WD_GRACE:-600}
API="http://127.0.0.1:${PORT:-8888}"
mkdir -p "$ST" "$LOGDIR"
exec >> "$LOGDIR/dsv41-watchdog.log" 2>&1
now=${DSV41_NOW:-$(date +%s)}
say() { printf '%s [watchdog] %s\n' "$(date '+%F %T')" "$*"; }
alert() {
  logger -t dsv41-watchdog -p user.err "$*" || true
  say "ATTENTION: $*"
  printf '%s %s\n' "$(date '+%F %T')" "$*" > "$LOGDIR/dsv41-ATTENTION"
}
num() { local n; n=$(cat "$ST/$1" 2>/dev/null); [[ $n =~ ^[0-9]+$ ]] && echo "$n" || echo 0; }
bump() { echo $(( $(num "$1") + 1 )) > "$ST/$1"; num "$1"; }
zero() { local f; for f in "$@"; do echo 0 > "$ST/$f"; done; }
unit_show() { systemctl --user show dsv41-serve.service --timestamp=unix -p "$1" --value; }

state=$(unit_show ActiveState)
case "$state" in
  active) zero activating-since;;
  inactive) exit 0;;
  failed) alert "dsv41-serve is failed (start limit hit): not restarting; manual attention needed"; exit 0;;
  activating|deactivating|reloading)
    since=$(num activating-since); last=$(num activating-last)
    if [ "$since" = 0 ] || [ $((now - last)) -gt 900 ]; then since=$now; echo "$since" > "$ST/activating-since"; fi
    echo "$now" > "$ST/activating-last"
    [ $((now - since)) -lt 1800 ] ||
      alert "dsv41-serve $state for $((now - since)) s: starts are not converging (worker unreachable, caps, kvdisk?); manual attention needed"
    exit 0;;
  *) alert "dsv41-serve state '$state': manual attention needed"; exit 0;;
esac
head_enter=$(unit_show ActiveEnterTimestamp); head_enter=${head_enter#@}
[[ $head_enter =~ ^[0-9]+$ ]] || { say "cannot read the head ActiveEnterTimestamp '$head_enter'"; exit 0; }
[ $((now - head_enter)) -ge "$GRACE" ] || exit 0

reason=""; class=health; unreachable=""
for w in "${WORKERS[@]}"; do
  # shellcheck disable=SC2016 # Expand on the worker.
  out=$($SSH "nacyot@$w" 'export XDG_RUNTIME_DIR=/run/user/$(id -u)
    systemctl --user show dsv41-serve.service -p ActiveState --value
    systemctl --user show dsv41-serve.service --timestamp=unix -p ActiveEnterTimestamp --value' 2>/dev/null) || out=""
  wstate=$(sed -n 1p <<< "$out"); wenter=$(sed -n 2p <<< "$out"); wenter=${wenter#@}
  if [ -z "$wstate" ]; then
    unreachable="$unreachable $w"
  elif [ "$wstate" != active ]; then
    reason="$w dsv41-serve $wstate"; class=unit; break
  elif [[ $wenter =~ ^[0-9]+$ ]] && [ "$wenter" -gt $((head_enter + 30)) ]; then
    reason="$w dsv41-serve restarted after the head"; class=unit; break
  fi
done
if [ -z "$reason" ]; then
  if [ -n "$unreachable" ]; then
    n=$(bump unreachable.n); say "unreachable:$unreachable x$n"
    [ "$n" -lt 3 ] || reason="unreachable:$unreachable x$n"
  else
    zero unreachable.n
  fi
fi
# With FRONTEND_ADDR (issue #24) the API is not on the head.
if [ -z "$reason" ] && [ -z "${FRONTEND_ADDR:-}" ]; then
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$API/health") || true
  if [ "$code" = 200 ]; then
    zero health.n
  else
    n=$(bump health.n); say "health ${code:-000} x$n"
    [ "$n" -lt 3 ] || reason="health ${code:-000} x$n"
  fi
fi
if [ -z "$reason" ] && [ -z "${FRONTEND_ADDR:-}" ]; then
  c=$(bump probe.c)
  if [ $((c % 5)) = 0 ] || [ "$(num probe.n)" != 0 ]; then
    body="{\"model\":\"${SERVED_NAME:-deepseek-v4.1-flash}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false}}"
    code=$(curl -s -m 120 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d "$body" "$API/v1/chat/completions") || true
    if [ "$code" = 200 ]; then
      zero probe.n
    else
      n=$(bump probe.n); say "probe ${code:-000} x$n"
      [ "$n" -lt 2 ] || reason="probe ${code:-000} x$n"
    fi
  fi
fi
[ -n "$reason" ] || exit 0

budget=12
if [ "$class" = health ]; then
  budget=4; last=$(num last-restart)
  if [ $((now - last)) -lt 900 ]; then say "$reason: cooldown, last restart $((now - last)) s ago"; exit 0; fi
fi
touch "$ST/restart-history"
recent=$(awk -v t=$((now - 7200)) -v c="$class" '$1 > t && $2 == c {n++} END {print n + 0}' "$ST/restart-history")
if [ "$recent" -ge "$budget" ]; then
  alert "$reason, but the $class restart budget ($budget in 2 h) is used up: manual attention needed"
  exit 0
fi
say "$reason -> restarting dsv41-serve ($class restart $((recent + 1))/$budget in 2 h)"
echo "$now $class $reason" >> "$ST/restart-history"
echo "$now" > "$ST/last-restart"
zero health.n probe.n unreachable.n
systemctl --user restart --no-block dsv41-serve.service
