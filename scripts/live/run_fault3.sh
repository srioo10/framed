#!/bin/bash
# PREREG LIVE-FAULT: no attacker; at t0+600 s drop the sensor's MQTT TCP at the Pi (broker unreachable), restore at t0+1000 s.
# FRAME-WITNESS v2 fresh fault arm: waits for the fresh LIVE-FAIR runs (OFF 15), flashes pos, runs F 11-15.
# Usage (root): setsid nohup run_fault3.sh
L=/home/pi4r8/livenat; R=$L/fair_runs.txt
until_s() { local d=$(( $1 - $(date +%s) )); [ $d -gt 0 ] && sleep $d; }
sensor_up() { local i t; for i in $(seq $(( ${2:-300} / 5 ))); do t=$(tail -1 $L/sub.log | cut -d. -f1); [ "${t:-0}" -gt "$1" ] && return 0; sleep 5; done; return 1; }
fault_on() { nft add table ip fault; nft add chain ip fault input '{ type filter hook input priority -10; }'; nft add rule ip fault input ip saddr 192.168.4.2 tcp dport 1883 drop; }
fault_off() { nft delete table ip fault 2>/dev/null; }
trap fault_off EXIT
pgrep -x mosquitto_sub >/dev/null || (cd $L && setsid nohup ./sub.sh >/dev/null 2>&1 < /dev/null &)
nmcli c down bigp-nat-att >/dev/null 2>&1
run() {
  fault_off; $L/gwreset.sh; local t0=$(date +%s)
  sensor_up $((t0 + 5)) || { echo "RUN fault F $1 INVALID sensor not publishing 300 s after reset ($(date +%s))" >> $R; return 1; }
  t0=$(date +%s); echo "RUN fault F $1 start $t0" >> $R
  until_s $((t0 + 600)); fault_on && echo "NOTE $(date +%s) fault F $1 fault ON" >> $R
  until_s $((t0 + 1000)); fault_off; echo "NOTE $(date +%s) fault F $1 fault OFF" >> $R
  until_s $((t0 + 1500)); echo "RUN fault F $1 end $(date +%s)" >> $R
}
for r in 11 12 13 14 15; do run $r || run $r; done
echo "FAULT DONE $(date +%s)" >> $R
