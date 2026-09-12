#!/usr/bin/env bash
# Local CLI tests: no node commands are executed. Requires Bash >= 4.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_DIR=$(mktemp -d)
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
  *free*) printf 'active used 100 GiB avail 5 GiB ';;
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
  "$BASH" "$HERE/dsv41_ctl.sh" "$command" > "$TEST_DIR/$name.out" \
    2> "$TEST_DIR/$name.err" || rc=$?
  if [ "$rc" -ne "$expected" ]; then
    printf 'FAIL %s: expected %s, got %s\n' "$name" "$expected" "$rc" >&2
    cat "$TEST_DIR/$name.err" >&2
    exit 1
  fi
  printf 'PASS %s (exit %s)\n' "$name" "$rc"
}
assert_no_ssh() { [ ! -s "$CAP_TEST_CALLS" ]; }
assert_order() {
  awk '/systemd-run/ {
    if ($0 ~ /nacyot@gx10-27c4/) print 3
    if ($0 ~ /nacyot@gx10-37cc/) print 2
    if ($0 ~ /nacyot@gx10-f323/) print 1
    if ($0 ~ /nacyot@gx10-6040/) print 0
  }' "$CAP_TEST_CALLS" > "$TEST_DIR/order"
  printf '3\n2\n1\n0\n' > "$TEST_DIR/expected-order"
  diff -u "$TEST_DIR/expected-order" "$TEST_DIR/order"
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
assert_order
SKIP_CAP_CHECK=1 run_case caps-not-bypassed 3
fixture
run_case normal-start 0 start
assert_order
run_case status 0 status
[ "$(grep -c 'cap active 1989MHz' "$TEST_DIR/status.out")" -eq 4 ]
grep -q 'health: 200' "$TEST_DIR/status.out"

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
printf 'All caps selftests passed. Artifacts retained: %s\n' "$TEST_DIR"
