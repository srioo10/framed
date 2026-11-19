#!/bin/bash
# PREREG LIVE-COLLATERAL (4ac4852), from LIVE-POS (ab3fe55): live positive control. Attacker = Pi wlsta on the gateway AP (own key) running a
# sensor-type MQTT client. Usage (root): setsid nohup run_pos.sh cal   |   setsid nohup run_pos.sh runs
L=/home/pi4r8/livenat; R=$L/fair_runs.txt; J=$L/fair_inject.log
. $L/pos_lib.sh   # alive() + safe_flash(), extracted unchanged from run_scan.sh
until_s() { local d=$(( $1 - $(date +%s) )); [ $d -gt 0 ] && sleep $d; }
att_iot() {  # LIVE-COLLATERAL: the attacker joins the sensor NAT network (bigp-nat-att), as run_scan.sh att_nat
  ip link show wlsta >/dev/null 2>&1 || { /usr/sbin/iw dev wlan0 interface add wlsta type managed; sleep 2; }
  nmcli c up bigp-nat-att >/dev/null 2>&1; sleep 3
  $L/net_up.sh >/dev/null 2>&1  # table 100 via the sensor NAPT, table 102 return route, lab peers
  A=$(ip -4 -o addr show wlsta | awk '{print $4}' | cut -d/ -f1)
  [ -n "$A" ] || return 1
}
client_start() {  # reconnecting MQTT client, one reading every 5 s, own process group
  local U; U=$(sed -nE 's/.* -u ([^ ]+) .*/\1/p' $L/sub.sh)
  setsid bash -c "while :; do ( i=0; while :; do i=\$((i+1)); echo \"{\\\"seq\\\":\$i,\\\"t\\\":23.\$((RANDOM%10))}\"; sleep 5; done ) | mosquitto_pub -h 192.168.77.1 -A $A -t onepass/att/temperature -u '$U' -P \"\$(cat $L/.mqttpw)\" -k 30 -l; sleep 5; done" >/dev/null 2>&1 &
  CPG=$!
}
client_stop() { [ -n "$CPG" ] && kill -- -$CPG 2>/dev/null; CPG=; }
sensor_up() {  # a sensor publish after epoch $1 (s), waiting up to $2 s (default 300)
  local i t; for i in $(seq $(( ${2:-300} / 5 ))); do t=$(tail -1 $L/sub.log | cut -d. -f1); [ "${t:-0}" -gt "$1" ] && return 0; sleep 5; done; return 1; }
inj() { python3 $L/sk_inject.py --src $1 "${@:2}" 2>&1 | tr "\n" " "; }
BURST="--n 10 --peers 4 --port-step 1 --fail 10 --closed-port 3000"
HSCAN="--n 28 --peers 28 --first 120 --port-step 0 --fail 28 --closed-port 4000 --gap-ms 50"
run() {  # build variant rep dur
  # order matters (LIVE-POS amendment 1): the gateway must boot with the attacker station off its AP, else it stops
  # forwarding the sensor and the Pi radio can crash; so: station down, reset, sensor up, then the attacker joins
  local try; A=
  nmcli c down bigp-nat-att >/dev/null 2>&1; sleep 3
  $L/gwreset.sh; local t0=$(date +%s)
  sensor_up $((t0 + 5)) || echo "NOTE $(date +%s) $1 $2 $3 sensor not publishing 300 s after gateway reset" >> $R
  sleep 30  # let the sensor settle before the attacker station associates
  for try in 1 2 3; do att_iot && break; nmcli c down bigp-nat-att >/dev/null 2>&1; sleep 20; done
  [ -n "$A" ] || { echo "RUN $1 $2 $3 INVALID attacker address '$A'" >> $R; return 1; }
  client_start
  local tj=$(date +%s)
  sensor_up $((tj + 5)) 60 ||  # 60 s: well inside the 250 s age gate, so a broken sensor cannot be blocked
    { echo "NOTE $(date +%s) $1 $2 $3 ABORT: sensor silent after attacker join; retry" >> $R
                             client_stop; nmcli c down bigp-nat-att >/dev/null 2>&1; return 2; }
  local k; for k in $(seq 12); do ss -tn state established "( dport = :1883 )" src $A | grep -q 1883 && break; sleep 5; done
  ss -tn state established "( dport = :1883 )" src $A | grep -q 1883 || echo "NOTE $(date +%s) $1 $2 $3 attacker MQTT not established after 60 s" >> $R
  echo "RUN $1 $2 $3 start $t0 att $A" >> $R
  if [ "$2" = P ]; then
    for at in 300 540 780; do until_s $((t0 + at)); echo "$1 $2 $3 burst@$at: $(inj $A $BURST)" >> $J; done
    until_s $((t0 + 1020)); echo "$1 $2 $3 hscan@1020: $(inj $A $HSCAN)" >> $J
  fi
  until_s $((t0 + $4)); echo "RUN $1 $2 $3 end $(date +%s)" >> $R; client_stop; nmcli c down bigp-nat-att >/dev/null 2>&1
}
run2() { local k; for k in 1 2 3; do run "$@"; [ $? -ne 2 ] && return; done; echo "RUN $1 $2 $3 INVALID sensor silent after attacker join (3 tries)" >> $R; }
trap client_stop EXIT
case $1 in
  cal) nmcli c down bigp-nat-att >/dev/null 2>&1; safe_flash pos_mon || exit 1; run pos_mon cal 1 2400;;
  runs) nmcli c down bigp-nat-att >/dev/null 2>&1; safe_flash pos || exit 1
        run2 col PC 1 1200; for r in 1 2 3 4 5; do run2 col P $r 1200; done; run2 col PC 2 1200;;
esac
echo "COL DONE $1 $(date +%s)" >> $R
