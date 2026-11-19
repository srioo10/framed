#!/bin/bash
# FRAME-WITNESS v2 queue (PREREG f699fba): after the fresh LIVE-FAIR runs (OFF 15) finish, flash pos (verified, as
# run_fair2.sh safe_flash) and run 5 fresh LIVE-FAULT runs (run_fault3.sh, reps 11-15). Usage (root): setsid nohup run_wq.sh
L=/home/pi4r8/livenat; R=$L/fair_runs.txt
until grep -q "^FAIR DONE off 15 " $R; do sleep 60; done
alive() { local t=$1; for i in $(seq 30); do
  awk -F, -v t=$((t * 1000)) '$2 == "F" && $1 > t {f=1} END {exit !f}' $(ls -t /opt/onepass/live/gw/gw-*.log | head -2) && return 0
  sleep 10; done; return 1; }
for try in 1 2 3; do
  systemctl stop onepass-gwlog; out=$($L/fl.sh gw pos 2>&1); systemctl start onepass-gwlog; t=$(date +%s)
  echo "$out" >> $R
  if [ $(echo "$out" | grep -c "Hash of data verified") -ge 1 ] && sleep 45 && alive $t; then
    echo "BUILD pos flashed $t verified (try $try) for FRAME-WITNESS v2 fault arm" >> $R; exec $L/run_fault3.sh; fi
  echo "FLASH CHECK FAILED pos try $try $(date +%s)" >> $R; sleep 30
done
echo "FRAME-WITNESS fault arm NOT RUN: pos flash failed 3 times $(date +%s)" >> $R
