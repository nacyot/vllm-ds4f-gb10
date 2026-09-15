#!/usr/bin/env bash
# Local CLI tests: no node commands are executed. Requires Bash >= 4.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_DIR=$(mktemp -d)
TEST_DIR=$(cd "$TEST_DIR" && pwd -P)
printf 'Caps selftest artifacts: %s\n' "$TEST_DIR"
export DSV41_CAP_FIXTURE="$TEST_DIR/fixture-current"
export CAP_MHZ=2000 SKIP_CAP_CHECK=0
unset CAP_TEST_RAW CAP_TEST_SSH_EXIT
export CAP_TEST_CALLS="$TEST_DIR/ssh-calls"
export PATH="$TEST_DIR:$PATH"

# Record calls without evaluating any remote command, including start_rank cleanup.
cat > "$TEST_DIR/ssh" <<'STUB'
#!/usr/bin/env bash
call=$*
printf '%s\n' "${call//$'\n'/ }" >> "$CAP_TEST_CALLS"
case "$*" in
  *systemd-run*) printf 'started\n';;
  *curl*) printf '200';;
  *systemctl*meminfo*) printf 'active used 100.00 GiB avail 5.00 GiB ';;
  *dsv41-watchdog.timer*) printf 'active, attention: none\n';;
  *ActiveEnterTimestamp*) printf '%s\n@%s\n' "${WD_WORKER_STATE:-active}" "${WD_WORKER_ENTER:-0}"
     exit "${CAP_TEST_SSH_EXIT:-0}";;
  *) printf '%s\n' "${CAP_TEST_RAW:-active
Enabled
1989
1989
1989
1989
1989}"
     exit "${CAP_TEST_SSH_EXIT:-0}";;
esac
STUB
cat > "$TEST_DIR/sleep" <<'STUB'
#!/usr/bin/env bash
printf 'sleep %s\n' "$*" >> "$CAP_TEST_CALLS"
exit 0
STUB
chmod +x "$TEST_DIR/ssh" "$TEST_DIR/sleep"

fixture() {
  local state=${1:-active} persistence=${2:-Enabled} mhz=${3:-1989}
  printf '%s\n' 'gx10-6040 active Enabled 1989' \
    "gx10-f323 $state $persistence $mhz" \
    'gx10-37cc active Enabled 1989' 'gx10-27c4 active Enabled 1989' \
    > "$DSV41_CAP_FIXTURE"
}
run_case() {
  local name=$1 expected=$2 command=${3:-caps} rc=0
  : > "$CAP_TEST_CALLS"
  "$BASH" "$HERE/dsv41_ctl.sh" "$command" "${@:4}" > "$TEST_DIR/$name.out" \
    2> "$TEST_DIR/$name.err" || rc=$?
  if [ "$rc" -ne "$expected" ]; then
    printf 'FAIL %s: expected %s, got %s\n' "$name" "$expected" "$rc" >&2
    cat "$TEST_DIR/$name.err" >&2
    exit 1
  fi
  printf 'PASS %s (exit %s)\n' "$name" "$rc"
}
assert_no_ssh() { [ ! -s "$CAP_TEST_CALLS" ]; }
# start writes the head's override file, then restarts the head unit once; the
# workers are restarted by the head unit's pre-start, never from here.
assert_start() {
  [ "$(grep -c 'systemctl --user restart dsv41-serve' "$CAP_TEST_CALLS")" -eq 1 ]
  grep 'systemctl --user restart dsv41-serve' "$CAP_TEST_CALLS" | grep -q 'nacyot@gx10-6040'
  if grep -q systemd-run "$CAP_TEST_CALLS"; then echo 'FAIL: transient start' >&2; exit 1; fi
  awk '/dsv41-override.env.new/ { if (!w) w = NR }
    /systemctl --user restart dsv41-serve/ { r = NR }
    END { exit !(w && w < r) }' "$CAP_TEST_CALLS"
}
override_sent() {
  sed -n "s/.*printf %s '\([^']*\)' | base64 -d.*/\1/p" "$CAP_TEST_CALLS" | head -1 | base64 -d
}
fixture
run_case normal 0
assert_no_ssh
fixture active Enabled 2535
run_case released 3
grep -q 'gx10-f323' "$TEST_DIR/released.err"
grep -q 'clock_ctl.sh cap' "$TEST_DIR/released.err"
run_case blocked-start 3 start
assert_no_ssh
fixture active Disabled
run_case persistence-disabled 3
fixture inactive
run_case inactive 3
fixture unreachable - -
run_case unreachable 3
fixture active Enabled N/A
run_case nonnumeric 3
printf '%s\n' 'gx10-6040 active Enabled 1989' > "$DSV41_CAP_FIXTURE"
run_case missing-host 3
fixture
printf '%s\n' 'gx10-f323 active Enabled 1989' >> "$DSV41_CAP_FIXTURE"
run_case duplicate-host 3
fixture active Enabled 2000
run_case boundary 0
CAP_MHZ=1999 run_case lower-limit 3
CAP_MHZ=invalid run_case invalid-limit 3
fixture active Enabled 2535
CAP_MHZ=2535 run_case higher-limit 0
SKIP_CAP_CHECK=1 run_case override-start 0 start
assert_start
SKIP_CAP_CHECK=1 run_case caps-not-bypassed 3
fixture
run_case normal-start 0 start
assert_start
[ -z "$(override_sent)" ]
# shellcheck disable=SC2016 # A literal $ must reach the override file escaped.
GMU=0.7 EXTRA_ARGS='--x "a b" $y' run_case knob-start 0 start
# shellcheck disable=SC2016
printf '%s\n' 'GMU="0.7"' 'EXTRA_ARGS="--x \"a b\" \$y"' | diff -u - <(override_sent)
assert_no_glob_rm() {
  if grep -Eq 'rm -f /dev/shm/[^;]*\*' "$CAP_TEST_CALLS"; then
    echo 'FAIL: remote glob deletion' >&2; exit 1
  fi
}
assert_no_glob_rm
run_case stop 0 stop
assert_no_glob_rm
[ "$(grep -c 'shm_cleanup clean$' "$CAP_TEST_CALLS")" -eq 4 ]
awk '/sleep 6/ { waited=1 } /pkill -KILL/ { if (!waited) exit 1; kills++ }
  END { if (kills != 4) exit 1 }' "$CAP_TEST_CALLS"
DSV41_SHM_DRYRUN=1 run_case dry-stop 0 stop
[ "$(grep -c 'shm_cleanup list$' "$CAP_TEST_CALLS")" -eq 4 ]
DSV41_SHM_DRYRUN=1 run_case dry-start 0 start
override_sent | grep -qx 'DSV41_SHM_DRYRUN="1"'
run_case install 0 install
[ "$(grep -c 'install -m 644' "$CAP_TEST_CALLS")" -eq 4 ]
grep 'install -m 644' "$CAP_TEST_CALLS" | grep 'nacyot@gx10-6040' | grep -q dsv41-watchdog.timer
grep 'enable --now dsv41-watchdog.timer' "$CAP_TEST_CALLS" | grep -q 'nacyot@gx10-6040'
run_case shm-all 0 shm
[ "$(grep -c 'shm_cleanup list$' "$CAP_TEST_CALLS")" -eq 4 ]
run_case shm-host 0 shm gx10-6040
[ "$(wc -l < "$CAP_TEST_CALLS")" -eq 1 ]
grep -q 'nacyot@gx10-6040' "$CAP_TEST_CALLS"
run_case shm-invalid 2 shm invalid
assert_no_ssh
run_case status 0 status
[ "$(grep -c 'cap active 1989MHz' "$TEST_DIR/status.out")" -eq 4 ]
grep -q 'health: 200' "$TEST_DIR/status.out"
grep -q 'watchdog: active, attention: none' "$TEST_DIR/status.out"

# Exercise the real probe parser with simulated SSH output and transport errors.
unset DSV41_CAP_FIXTURE
run_case sampled-normal 0
export CAP_TEST_RAW=$'setlocale: warning\nactive\nEnabled\n1989\n1989\n1989\n1989\n1989'
run_case locale-warning 0
export CAP_TEST_RAW=$'setlocale: warning\nactive\nEnabled\n1989\n2535\n1989\n1989\n1989'
run_case sampled-maximum 3
export CAP_TEST_RAW=$'active\nEnabled\n1989\n1989\n1989\n1989'
run_case incomplete-samples 3
export CAP_TEST_RAW=$'active\nEnabled\nN/A\n1989\n1989\n1989\n1989'
run_case invalid-sample 3
CAP_TEST_SSH_EXIT=255 run_case ssh-failure 3
unset CAP_TEST_RAW
printf 'All caps selftests passed. Artifacts retained: %s\n' "$TEST_DIR"

export DSV41_MEM_FIXTURE="$TEST_DIR/memory-fixture"
mem_fixture() {
  printf '%s\n' "gx10-6040 $1" 'gx10-f323 7.00' \
    'gx10-37cc 7.00' 'gx10-27c4 7.00' > "$DSV41_MEM_FIXTURE"
}
mem_fixture 6.95
run_case memory-normal 0 headroom
assert_no_ssh
mem_fixture 4.52
run_case memory-low 3 headroom
assert_no_ssh
grep -q 'gx10-6040.*restart' "$TEST_DIR/memory-low.err"
MIN_AVAIL_GIB=4.5 run_case memory-override 0 headroom
MIN_AVAIL_GIB=6 run_case memory-argument 0 headroom 4.5
mem_fixture 5.2
run_case memory-boundary 0 headroom
mem_fixture invalid
run_case memory-invalid 3 headroom
printf '%s\n' 'gx10-f323 6.95' > "$DSV41_MEM_FIXTURE"
run_case memory-missing 3 headroom
mem_fixture 6.95
printf '%s\n' 'gx10-6040 6.95' >> "$DSV41_MEM_FIXTURE"
run_case memory-duplicate 3 headroom
mem_fixture 6.95
MIN_AVAIL_GIB=invalid run_case memory-invalid-limit 3 headroom
assert_no_ssh
unset DSV41_MEM_FIXTURE
CAP_TEST_SSH_EXIT=255 run_case memory-ssh-failure 3 headroom
printf 'All headroom selftests passed. Artifacts retained: %s\n' "$TEST_DIR"

# Execute the same cleanup body against a private fixture root. rm is a recorder;
# neither SSH commands nor actual /dev/shm files are touched.
mkdir "$TEST_DIR/shm"
sed -n '/^shm_cleanup()/,/^}/p' "$HERE/dsv41_lib.sh" |
  sed "s|/dev/shm|$TEST_DIR/shm|g" > "$TEST_DIR/shm-function.sh"
# shellcheck disable=SC2329 # These stubs are called by the sourced remote function.
(
  # shellcheck disable=SC1091
  source "$TEST_DIR/shm-function.sh"
  systemctl() {
    case "$*" in
      *InvocationID*) printf '%s\n' "${SHM_INVOCATION:-}";;
      *) printf '%s\n' "$SHM_STATE";;
    esac
  }
  fuser() {
    case "$SHM_USE" in
      busy) printf '1234\n'; return 0;;
      error) printf 'inspection failed\n' >&2; return 1;;
      *) return 1;;
    esac
  }
  # GNU stat is remote-only. The fixture preserves real path/type/owner checks.
  stat() { printf '1:2:%s:4\n' "$(id -u)"; }
  realpath() { command realpath "${@: -1}"; }
  rm() { printf '%s\n' "$*" >> "$TEST_DIR/deletions"; }
  printf 'test' > "$TEST_DIR/shm/psm_fixture"
  printf 'keep' > "$TEST_DIR/shm/unrelated"
  ln -s "$TEST_DIR/shm/unrelated" "$TEST_DIR/shm/psm_link"
  SHM_USE=unused
  for SHM_STATE in active activating deactivating unknown; do
    : > "$TEST_DIR/deletions"
    shm_cleanup clean > "$TEST_DIR/shm-$SHM_STATE.out"
    [ ! -s "$TEST_DIR/deletions" ]
    grep -q "skip: dsv41-serve $SHM_STATE" "$TEST_DIR/shm-$SHM_STATE.out"
  done
  SHM_STATE=inactive SHM_USE=busy
  shm_cleanup clean > "$TEST_DIR/shm-busy.out"
  [ ! -s "$TEST_DIR/deletions" ]
  grep -q 'skip: in use' "$TEST_DIR/shm-busy.out"
  SHM_USE=error
  if shm_cleanup clean > "$TEST_DIR/shm-error.out"; then exit 1; fi
  [ ! -s "$TEST_DIR/deletions" ]
  SHM_USE=unused
  shm_cleanup list > "$TEST_DIR/shm-list.out"
  [ ! -s "$TEST_DIR/deletions" ]
  shm_cleanup clean > "$TEST_DIR/shm-clean.out"
  printf '%s\n' "-f -- $TEST_DIR/shm/psm_fixture" > "$TEST_DIR/expected-deletions"
  diff -u "$TEST_DIR/expected-deletions" "$TEST_DIR/deletions"
  grep -q 'deleted:' "$TEST_DIR/shm-clean.out"
  # pre: an activating unit is cleaned only from its own ExecStartPre.
  export INVOCATION_ID=own
  for SHM_STATE in activating active; do
    SHM_INVOCATION=other
    [ "$SHM_STATE" = activating ] || SHM_INVOCATION=own
    : > "$TEST_DIR/deletions"
    shm_cleanup pre > "$TEST_DIR/shm-pre-$SHM_STATE.out"
    [ ! -s "$TEST_DIR/deletions" ]
    grep -q "skip: dsv41-serve $SHM_STATE" "$TEST_DIR/shm-pre-$SHM_STATE.out"
  done
  SHM_STATE=activating SHM_INVOCATION=own
  : > "$TEST_DIR/deletions"
  shm_cleanup pre > "$TEST_DIR/shm-pre-own.out"
  diff -u "$TEST_DIR/expected-deletions" "$TEST_DIR/deletions"
  SHM_INVOCATION=own
  : > "$TEST_DIR/deletions"
  shm_cleanup clean > "$TEST_DIR/shm-clean-activating.out"
  [ ! -s "$TEST_DIR/deletions" ]
  unset INVOCATION_ID
  # An unexpected enumerator result must not extend the permitted directory.
  SHM_STATE=inactive
  find() { printf '%s\0' "$TEST_DIR/shm/../shm/psm_fixture"; }
  : > "$TEST_DIR/deletions"
  if shm_cleanup clean > "$TEST_DIR/shm-unsafe.out"; then exit 1; fi
  [ ! -s "$TEST_DIR/deletions" ]
)
printf 'All shm selftests passed. Artifacts retained: %s\n' "$TEST_DIR"

# Node-side scripts (dsv41_unit.sh, dsv41_watchdog.sh) with recording stubs for
# systemctl, curl, logger, sudo and mountpoint; SSH goes to the stub above.
mkdir "$TEST_DIR/node-bin"
for tool in systemctl curl logger sudo mountpoint find fuser; do
  cat > "$TEST_DIR/node-bin/$tool" <<STUB
#!/usr/bin/env bash
printf '%s %s\n' $tool "\$*" >> "\$CAP_TEST_CALLS"
STUB
done
cat >> "$TEST_DIR/node-bin/systemctl" <<'STUB'
case "$*" in
  *InvocationID*) printf '%s\n' "${UNIT_INVOCATION:-}";;
  *ActiveEnterTimestamp*) printf '@%s\n' "${UNIT_ENTER:-0}";;
  *ActiveState*) printf '%s\n' "${UNIT_STATE:-inactive}";;
esac
STUB
cat >> "$TEST_DIR/node-bin/curl" <<'STUB'
case "$*" in
  *chat/completions*) printf '%s' "${WD_PROBE:-200}";;
  *) printf '%s' "${WD_HEALTH:-200}";;
esac
STUB
# shellcheck disable=SC2016 # Expanded by the stub.
printf '[ "${KVDISK_MOUNTED:-1}" = 1 ]\n' >>"$TEST_DIR/node-bin/mountpoint"
printf 'exit 1\n' >> "$TEST_DIR/node-bin/fuser"
chmod +x "$TEST_DIR/node-bin/"*
export DSV41_CAP_FIXTURE="$TEST_DIR/fixture-current"
mem_gib() { printf 'MemAvailable: %s kB\n' $(($1 * 1048576)) > "$TEST_DIR/meminfo"; }

unit_case() { # name expected step [VAR=value...]
  local name=$1 expected=$2 step=$3 rc=0
  : > "$CAP_TEST_CALLS"
  env PATH="$TEST_DIR/node-bin:$PATH" DSV41_HOST=gx10-6040 DSV41_LOG_DIR="$TEST_DIR/unit-$name" \
    DSV41_OVERRIDE_FILE="$TEST_DIR/override.env" DSV41_MEMINFO="$TEST_DIR/meminfo" \
    UNIT_STATE=activating UNIT_INVOCATION=own INVOCATION_ID=own "${@:4}" \
    "$BASH" "$HERE/dsv41_unit.sh" "$step" || rc=$?
  if [ "$rc" -ne "$expected" ]; then
    printf 'FAIL %s: expected %s, got %s\n' "$name" "$expected" "$rc" >&2
    cat "$TEST_DIR/unit-$name/"*.log >&2
    exit 1
  fi
  printf 'PASS %s (exit %s)\n' "$name" "$rc"
}
assert_no_worker_restart() {
  if grep -q 'restart dsv41-serve' "$CAP_TEST_CALLS"; then
    echo 'FAIL: workers restarted' >&2; exit 1
  fi
}
fixture
mem_gib 114
printf '%s\n' 'GMU="0.7"' > "$TEST_DIR/override.env"
unit_case unit-pre-head 0 pre
awk '/systemctl --user restart dsv41-serve/ {
    if ($0 ~ /nacyot@gx10-27c4/) print 3
    if ($0 ~ /nacyot@gx10-37cc/) print 2
    if ($0 ~ /nacyot@gx10-f323/) print 1
  }' "$CAP_TEST_CALLS" | diff -u <(printf '3\n2\n1\n') -
override_sent | diff -u "$TEST_DIR/override.env" -
grep -q 'mountpoint -q /mnt/kvdisk' "$CAP_TEST_CALLS"
grep -q 'sudo -n sh -c sync; echo 3 > /proc/sys/vm/drop_caches' "$CAP_TEST_CALLS"
grep -q 'pre-start done' "$TEST_DIR/unit-unit-pre-head/dsv41-r0.log"
unit_case unit-pre-worker 0 pre DSV41_HOST=gx10-27c4
if grep -Eq 'nacyot@|mountpoint' "$CAP_TEST_CALLS"; then echo 'FAIL: worker pre-start left the node' >&2; exit 1; fi
grep -q 'drop_caches' "$CAP_TEST_CALLS"
[ -s "$TEST_DIR/unit-unit-pre-worker/dsv41-r3.log" ]
unit_case unit-pre-kvdisk 1 pre KVDISK_MOUNTED=0
assert_no_worker_restart
fixture active Enabled 2535
unit_case unit-pre-caps 1 pre
assert_no_worker_restart
grep -q '^logger .*clock cap' "$CAP_TEST_CALLS"
fixture
unit_case unit-pre-unreachable 1 pre CAP_TEST_SSH_EXIT=255
assert_no_worker_restart
mem_gib 50
unit_case unit-pre-memory 1 pre
assert_no_worker_restart
mem_gib 114
unit_case unit-post-head 0 post
grep -q 'systemctl --user start --no-block dsv41-warmup.service' "$CAP_TEST_CALLS"
unit_case unit-post-worker 0 post DSV41_HOST=gx10-f323
assert_no_ssh
unit_case unit-unknown-host 2 pre DSV41_HOST=gx10-other
printf 'All unit selftests passed. Artifacts retained: %s\n' "$TEST_DIR"

wd_run() { # dir [VAR=value...]: one watchdog check; head active since 1000, now 10000
  : > "$CAP_TEST_CALLS"
  env PATH="$TEST_DIR/node-bin:$PATH" DSV41_WD_DIR="$TEST_DIR/$1" DSV41_LOG_DIR="$TEST_DIR/$1/logs" \
    DSV41_NOW=10000 UNIT_STATE=active UNIT_ENTER=1000 WD_WORKER_ENTER=990 "${@:2}" \
    "$BASH" "$HERE/dsv41_watchdog.sh"
}
restarts() { grep -c 'restart --no-block dsv41-serve' "$CAP_TEST_CALLS" || true; }
wd_run wd-inactive UNIT_STATE=inactive
[ "$(restarts)" = 0 ]
if grep -q nacyot@ "$CAP_TEST_CALLS"; then exit 1; fi
wd_run wd-failed UNIT_STATE=failed
[ "$(restarts)" = 0 ]
grep -q 'start limit' "$TEST_DIR/wd-failed/logs/dsv41-ATTENTION"
wd_run wd-grace DSV41_NOW=1300
if grep -q nacyot@ "$CAP_TEST_CALLS"; then exit 1; fi
for _ in 1 2 3 4 5; do wd_run wd-healthy; [ "$(restarts)" = 0 ]; done
grep -q 'chat/completions' "$CAP_TEST_CALLS"
wd_run wd-worker-inactive WD_WORKER_STATE=inactive
[ "$(restarts)" = 1 ]
grep -q '^10000 unit gx10-27c4 dsv41-serve inactive' "$TEST_DIR/wd-worker-inactive/restart-history"
wd_run wd-worker-late WD_WORKER_ENTER=1100
[ "$(restarts)" = 1 ]
for n in 1 2 3; do
  wd_run wd-health WD_HEALTH=503
  [ "$(restarts)" = $((n / 3)) ]
done
for n in 1 2 3 4 5 6; do
  wd_run wd-probe WD_PROBE=500
  [ "$(restarts)" = $((n / 6)) ]
done
grep -q 'probe 500 x2' "$TEST_DIR/wd-probe/restart-history"
for n in 1 2 3; do
  wd_run wd-unreachable CAP_TEST_SSH_EXIT=255
  [ "$(restarts)" = $((n / 3)) ]
done
mkdir -p "$TEST_DIR/wd-cooldown"
printf '2\n' > "$TEST_DIR/wd-cooldown/health.n"
printf '9500\n' > "$TEST_DIR/wd-cooldown/last-restart"
wd_run wd-cooldown WD_HEALTH=503
[ "$(restarts)" = 0 ]
grep -q cooldown "$TEST_DIR/wd-cooldown/logs/dsv41-watchdog.log"
mkdir -p "$TEST_DIR/wd-budget"
printf '2\n' > "$TEST_DIR/wd-budget/health.n"
printf '%s health health 503 x3\n' 3000 4000 5000 6000 > "$TEST_DIR/wd-budget/restart-history"
wd_run wd-budget WD_HEALTH=503
[ "$(restarts)" = 0 ]
grep -q 'budget' "$TEST_DIR/wd-budget/logs/dsv41-ATTENTION"
wd_run wd-budget WD_WORKER_STATE=failed
[ "$(restarts)" = 1 ]
printf 'All watchdog selftests passed. Artifacts retained: %s\n' "$TEST_DIR"
