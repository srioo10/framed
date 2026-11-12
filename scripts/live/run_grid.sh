#!/bin/bash
# PREREG LIVE-NAT-ENF grid on Pi #1 (as root, unattended): per build 3 clean runs, then 5 x (A10, B10, A50, B50);
# 20 min per run, gateway hard reset first, bursts of n flows at 300/540/780/1020 s. Logs to ~pi4r8/livenat/.
# Usage: setsid nohup run_grid.sh [builds="veto ord"] &
L=/home/pi4r8/livenat; ATT=192.168.5.2
until_s() { local d=$(( $1 - $(date +%s) )); [ $d -gt 0 ] && sleep $d; }
run() {  # build variant n rep
  $L/net_up.sh >/dev/null 2>&1 || echo "net_up failed $(date)" >> $L/runs.txt
  $L/gwreset.sh; local t0=$(date +%s)
  echo "RUN $1 $2 $3 $4 start $t0" >> $L/runs.txt
  if [ "$2" != clean ]; then
    for at in 300 540 780 1020; do
      until_s $((t0 + at))
      if [ "$2" = A ]; then a="--fail $3 --closed-port 3000"; else a="--fail 0 --dport 2000"; fi
      echo "$1 $2 $3 $4 burst@$at: $(python3 $L/sk_inject.py --src $ATT --n $3 --peers 4 --port-step 1 $a 2>&1 | tr '\n' ' ')" >> $L/inject.log
    done
  fi
  until_s $((t0 + 1200))
  echo "RUN $1 $2 $3 $4 end $(date +%s)" >> $L/runs.txt
}
for b in ${1:-veto ord}; do
  systemctl stop onepass-gwlog; $L/fl.sh gw nat_$b >> $L/runs.txt 2>&1; systemctl start onepass-gwlog
  echo "BUILD $b flashed $(date +%s)" >> $L/runs.txt
  for r in 1 2 3; do run $b clean 0 $r; done
  for r in 1 2 3 4 5; do run $b A 10 $r; run $b B 10 $r; run $b A 50 $r; run $b B 50 $r; done
done
echo "GRID DONE $(date +%s)" >> $L/runs.txt
