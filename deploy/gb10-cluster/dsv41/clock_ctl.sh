#!/opt/homebrew/bin/bash
# clock_ctl.sh — GPU clock helpers for the 4-node DSv41 cluster (issue #14).
#   show                     cap service state, SM clock, power, temp, throttle bits, MemAvailable per node
#   cap                      re-apply the owner's cap (systemctl restart gpu-clock-cap.service) on all nodes
# Clock changes go through the owner's gpu-clock-cap.service only; never call nvidia-smi -pm/-lgc/-rgc here.
#   sample start <tag> <s>   per-node nvidia-smi CSV sampler (500 ms) that exits after <s> seconds
#   sample stop              stop any sampler still running
#   sample fetch <tag> <dir> copy the samplers' CSVs into <dir>
#   counters <tag> <dir>     dump the throttle counters (nvidia-smi -q -d PERFORMANCE) per node into <dir>
set -u
ALL=(gx10-6040 gx10-f323 gx10-37cc gx10-27c4)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
Q="timestamp,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
cmd=${1:-}; shift || true
case $cmd in
  show)
    for h in "${ALL[@]}"; do
      printf "%s: " "$h"
      $SSH "$h" 'printf "%s " "$(systemctl is-active gpu-clock-cap.service)"; nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader | tr "\n" " "; free -g | awk "NR==2{print \"avail\",\$7,\"GiB\"}"' 2>&1 | grep -v setlocale
    done;;
  cap)
    for h in "${ALL[@]}"; do
      printf "%s: " "$h"
      $SSH "$h" 'sudo -n systemctl restart gpu-clock-cap.service && sleep 1 && nvidia-smi --query-gpu=clocks.sm --format=csv,noheader' 2>&1 | grep -v setlocale
    done;;
  sample)
    sub=${1:?start|stop|fetch}; shift
    case $sub in
      start)
        tag=${1:?tag}; secs=${2:?seconds}
        for h in "${ALL[@]}"; do
          $SSH "$h" "mkdir -p ~/dsv41-prep/logs; nohup timeout $secs nvidia-smi --query-gpu=$Q --format=csv -lms 500 > ~/dsv41-prep/logs/clk-$tag-$h.csv 2>/dev/null </dev/null & echo \$!" 2>&1 | grep -v setlocale | sed "s/^/$h sampler pid /"
        done;;
      stop)
        for h in "${ALL[@]}"; do $SSH "$h" 'pkill -f "nvidia-smi --query-gpu=timestamp" ; true' 2>&1 | grep -v setlocale; done; echo stopped;;
      fetch)
        tag=${1:?tag}; dir=${2:?dir}
        for h in "${ALL[@]}"; do scp -q "$h:~/dsv41-prep/logs/clk-$tag-$h.csv" "$dir/" && echo "$h: $(wc -l < "$dir/clk-$tag-$h.csv") lines"; done;;
      *) echo "sample start <tag> <s> | stop | fetch <tag> <dir>"; exit 2;;
    esac;;
  counters)
    tag=${1:?tag}; dir=${2:?dir}
    for h in "${ALL[@]}"; do
      $SSH "$h" 'nvidia-smi -q -d PERFORMANCE | grep -E "SW Power Capping|SW Thermal Slowdown|HW Thermal Slowdown|HW Power Braking"' 2>&1 | grep -v setlocale > "$dir/counters-$tag-$h.txt"
      printf "%s: " "$h"; tr -s ' ' < "$dir/counters-$tag-$h.txt" | tr '\n' ';'; echo
    done;;
  *) echo "usage: clock_ctl.sh show|cap|sample start <tag> <s>|sample stop|sample fetch <tag> <dir>|counters <tag> <dir>"; exit 2;;
esac
