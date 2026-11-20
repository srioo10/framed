#!/bin/bash
# FRAME-WITNESS background arm (PREREG, see the BG row): after the v2 fault arm (FAULT DONE), flash fair_veto (verified)
# and run, with a benign co-tenant behind the same NAT opening one established connection every 10-30 s to a random
# peer 192.168.150-199.1 (ports 2050-2099): 4 ABG (A10 bursts), 4 BBG (B10 bursts), 3 BGC (background only).
# BG2 rerun (PREREG BG2 row): the first BG arm was invalid (listeners.py not running, so no B or background flow was
# established). Same protocol; starts listeners.py and logs a probe; reps ABG/BBG 5-8, BGC 4-6.
# Usage (root): setsid nohup run_bg2.sh
L=/home/pi4r8/livenat; R=$L/fair_runs.txt; J=$L/fair_inject.log
pgrep -f listeners.py >/dev/null || (cd $L && setsid nohup python3 listeners.py >/dev/null 2>&1 < /dev/null &); sleep 3
until_s() { local d=$(( $1 - $(date +%s) )); [ $d -gt 0 ] && sleep $d; }
ensure_sta() { ip link show wlsta >/dev/null 2>&1 || { iw dev wlan0 interface add wlsta type managed; sleep 2; }; }
att_nat() { ensure_sta; nmcli c up bigp-nat-att >/dev/null 2>&1; $L/net_up.sh >/dev/null 2>&1; }
inj() { python3 $L/sk_inject.py --src $1 "${@:2}" 2>&1 | tr "\n" " "; }
alive() { local t=$1; for i in $(seq 30); do
  awk -F, -v t=$((t * 1000)) '$2 == "F" && $1 > t {f=1} END {exit !f}' $(ls -t /opt/onepass/live/gw/gw-*.log | head -2) && return 0
  sleep 10; done; return 1; }
flashed=0
for try in 1 2 3; do
  systemctl stop onepass-gwlog; out=$($L/fl.sh gw fair_veto 2>&1); systemctl start onepass-gwlog; t=$(date +%s)
  echo "$out" >> $R
  if [ $(echo "$out" | grep -c "Hash of data verified") -ge 1 ] && sleep 45 && alive $t; then
    echo "BUILD fair_veto flashed $t verified (try $try) for FRAME-WITNESS BG2 arm" >> $R; flashed=1; break; fi
  echo "FLASH CHECK FAILED fair_veto try $try $(date +%s)" >> $R; sleep 30
done
[ $flashed = 1 ] || { echo "FRAME-WITNESS BG2 arm NOT RUN: flash failed $(date +%s)" >> $R; exit 1; }
bg() {  # benign co-tenant until $2 (epoch s), from address $1
  while [ $(date +%s) -lt $2 ]; do
    sleep $((10 + RANDOM % 21)); k=$((150 + RANDOM % 50)); p=$((2050 + RANDOM % 50))
    echo "BG $(date +%s%3N) 192.168.$k.1:$p $(inj $1 --first $k --n 1 --fail 0 --dport $p)" >> $L/bg.log
  done
}
run() {  # variant rep
  local A try
  for try in 1 2 3; do att_nat; A=$(ip -4 -o addr show wlsta | awk "{print \$4}" | cut -d/ -f1); [ -n "$A" ] && break; sleep 20; done
  echo "PROBE (before reset) veto $1 $2 $(inj $A --first 150 --n 1 --fail 0 --dport 2050) | $(inj $A --first 120 --n 1 --fail 0 --dport 2000)" >> $J
  $L/gwreset.sh; local t0=$(date +%s)
  [ -n "$A" ] || echo "RUN veto $1 $2 INVALID no attacker address" >> $R
  echo "RUN veto $1 $2 start $t0 att $A" >> $R
  bg $A $((t0 + 1190)) & local bp=$!
  if [ $1 != BGC ]; then for at in 300 540 780 1020; do until_s $((t0 + at))
    if [ $1 = BBG ]; then a="--n 10 --peers 4 --port-step 1 --fail 0 --dport 2000"; else a="--n 10 --peers 4 --port-step 1 --fail 10 --closed-port 3000"; fi
    echo "veto $1 $2 burst@$at: $(inj $A $a)" >> $J; done; fi
  until_s $((t0 + 1200)); kill $bp 2>/dev/null; wait $bp 2>/dev/null; echo "RUN veto $1 $2 end $(date +%s)" >> $R
  [ -n "$A" ]
}
run1() { run "$@" || { echo "RUN veto $1 $2 RERUN (invalid)" >> $R; run "$@"; }; }
for r in 5 6 7 8; do run1 ABG $r; run1 BBG $r; done
for r in 4 5 6; do run1 BGC $r; done
echo "BG2 DONE $(date +%s)" >> $R
