#!/bin/bash
# PREREG LIVE-FAIR (commit 9d8e96b) on Pi #1, as root, unattended. Builds fair_veto / fair_ord staged under ~/fw
# with CONFIG_GW_TAU = tau_fair. Per build: 3 clean, 5 A10, 5 B10, 5 E (A10 bursts + 200-port scan at 1020 s);
# then 5 D runs on the Veto build (attacker on the gateway AP with its own key, scans at 300 and 1020 s).
# Usage: setsid nohup run_fair.sh [builds="veto ord"] [dev=1] &
L=/home/pi4r8/livenat; R=$L/fair_runs.txt; J=$L/fair_inject.log
until_s() { local d=$(( $1 - $(date +%s) )); [ $d -gt 0 ] && sleep $d; }
att_nat() { nmcli c up bigp-nat-att >/dev/null 2>&1; $L/net_up.sh >/dev/null 2>&1; }
att_iot() {  # attacker directly on the gateway AP (own key); route lab peers via the gateway
  nmcli c up bigp-iot-att passwd-file $L/.iotpw >/dev/null 2>&1
  local A=$(ip -4 -o addr show wlsta | awk "{print \$4}" | cut -d/ -f1); [ -n "$A" ] || return 1
  ip route replace default via 192.168.4.1 dev wlsta table 101
  ip rule show | grep -q "from $A lookup 101" || ip rule add pref 49 from $A lookup 101
  sysctl -qw net.ipv4.conf.wlsta.accept_local=1 net.ipv4.conf.wlsta.rp_filter=0
}
inj() { python3 $L/sk_inject.py --src $1 "${@:2}" 2>&1 | tr "\n" " "; }
SCAN="--n 200 --peers 1 --port-step 1 --fail 200 --closed-port 4000 --gap-ms 50"
run() {  # build variant rep
  if [ "$2" = D ]; then att_iot; else att_nat; fi
  local A=$(ip -4 -o addr show wlsta | awk "{print \$4}" | cut -d/ -f1)
  $L/gwreset.sh; local t0=$(date +%s)
  [ -n "$A" ] || echo "RUN $1 $2 $3 INVALID no attacker address" >> $R
  echo "RUN $1 $2 $3 start $t0 att $A" >> $R
  case $2 in
    A|B|E) for at in 300 540 780 1020; do until_s $((t0 + at))
             if [ "$2" = E ] && [ $at = 1020 ]; then a="$SCAN"; elif [ "$2" = B ]; then a="--n 10 --peers 4 --port-step 1 --fail 0 --dport 2000"
             else a="--n 10 --peers 4 --port-step 1 --fail 10 --closed-port 3000"; fi
             echo "$1 $2 $3 burst@$at: $(inj $A $a)" >> $J; done;;
    D) for at in 300 1020; do until_s $((t0 + at)); echo "$1 $2 $3 scan@$at: $(inj $A $SCAN)" >> $J; done;;
  esac
  until_s $((t0 + 1200)); echo "RUN $1 $2 $3 end $(date +%s)" >> $R
}
for b in ${1:-veto ord}; do
  systemctl stop onepass-gwlog; $L/fl.sh gw fair_$b >> $R 2>&1; systemctl start onepass-gwlog
  echo "BUILD fair_$b flashed $(date +%s)" >> $R
  for r in 1 2 3; do run $b clean $r; done
  for r in 1 2 3 4 5; do run $b A $r; run $b B $r; run $b E $r; done
  if [ $b = veto ] && [ "${2:-1}" = 1 ]; then for r in 1 2 3 4 5; do run $b D $r; done; att_nat; fi
done
echo "FAIR DONE $(date +%s)" >> $R
