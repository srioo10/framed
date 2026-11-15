#!/bin/bash
# PREREG LIVE-SCAN (15 Nov): run_fair2.sh plus horizontal-scan variants H (behind the NAT) and DH (own key on the
# gateway AP): 80 refused SYNs to 80 distinct lab hosts (192.168.120-199.1, one port), 50 ms apart.
# Usage: setsid nohup run_scan.sh scan
# PREREG LIVE-FAIR (commit 9d8e96b) on Pi #1, as root, unattended. Builds fair_veto / fair_ord staged under ~/fw
# with CONFIG_GW_TAU = tau_fair. Per build: 3 clean, 5 A10, 5 B10, 5 E (A10 bursts + 200-port scan at 1020 s);
# then 5 D runs on the Veto build (attacker on the gateway AP with its own key, scans at 300 and 1020 s).
# Usage: OFF=5 setsid nohup run_fair2.sh [builds="veto ord"] [dev=1] &   (OFF: repeat-number offset, PREREG LIVE-FAIR-REP)
L=/home/pi4r8/livenat; R=$L/fair_runs.txt; J=$L/fair_inject.log
until_s() { local d=$(( $1 - $(date +%s) )); [ $d -gt 0 ] && sleep $d; }
ensure_sta() { ip link show wlsta >/dev/null 2>&1 || { iw dev wlan0 interface add wlsta type managed; sleep 2; }; }  # brcmfmac resets drop the virtual STA (as run_grid.sh)
att_nat() { ensure_sta; nmcli c up bigp-nat-att >/dev/null 2>&1; $L/net_up.sh >/dev/null 2>&1; }
att_iot() {  # attacker directly on the gateway AP (own key); route lab peers via the gateway
  ensure_sta; nmcli c up bigp-iot-att passwd-file $L/.iotpw >/dev/null 2>&1
  local A=$(ip -4 -o addr show wlsta | awk "{print \$4}" | cut -d/ -f1); [ -n "$A" ] || return 1
  ip route replace default via 192.168.4.1 dev wlsta table 101
  ip rule show | grep -q "from $A lookup 101" || ip rule add pref 49 from $A lookup 101
  sysctl -qw net.ipv4.conf.wlsta.accept_local=1 net.ipv4.conf.wlsta.rp_filter=0
}
inj() { python3 $L/sk_inject.py --src $1 "${@:2}" 2>&1 | tr "\n" " "; }
SCAN="--n 200 --peers 1 --port-step 1 --fail 200 --closed-port 4000 --gap-ms 50"
HSCAN="--n 80 --peers 80 --first 120 --port-step 0 --fail 80 --closed-port 4000 --gap-ms 50"
run() {  # build variant rep
  local A try
  for try in 1 2 3; do  # a radio reset can drop the attacker STA: re-attach before giving up
    if [ "$2" = D ] || [ "$2" = DH ]; then att_iot; else att_nat; fi
    A=$(ip -4 -o addr show wlsta | awk "{print \$4}" | cut -d/ -f1); [ -n "$A" ] && break; sleep 20
  done
  $L/gwreset.sh; local t0=$(date +%s)
  [ -n "$A" ] || echo "RUN $1 $2 $3 INVALID no attacker address" >> $R
  echo "RUN $1 $2 $3 start $t0 att $A" >> $R
  case $2 in
    A|B|E|H) for at in 300 540 780 1020; do until_s $((t0 + at))
             if [ "$2" = E ] && [ $at = 1020 ]; then a="$SCAN"; elif [ "$2" = H ] && [ $at = 1020 ]; then a="$HSCAN"; elif [ "$2" = B ]; then a="--n 10 --peers 4 --port-step 1 --fail 0 --dport 2000"
             else a="--n 10 --peers 4 --port-step 1 --fail 10 --closed-port 3000"; fi
             echo "$1 $2 $3 burst@$at: $(inj $A $a)" >> $J; done;;
    D) for at in 300 1020; do until_s $((t0 + at)); echo "$1 $2 $3 scan@$at: $(inj $A $SCAN)" >> $J; done;;
    DH) for at in 300 1020; do until_s $((t0 + at)); echo "$1 $2 $3 hscan@$at: $(inj $A $HSCAN)" >> $J; done;;
  esac
  until_s $((t0 + 1200)); echo "RUN $1 $2 $3 end $(date +%s)" >> $R
  [ -n "$A" ]  # PREREG INVALID rule (no attacker address): the caller reruns once
}
run1() { run "$@" || { echo "RUN $1 $2 $3 RERUN (invalid)" >> $R; run "$@"; }; }
alive() {  # a decision (F line) logged by the gateway after time $1 (s), waiting up to 300 s
  local t=$1; for i in $(seq 30); do
    awk -F, -v t=$((t * 1000)) '$2 == "F" && $1 > t {f=1} END {exit !f}' $(ls -t /opt/onepass/live/gw/gw-*.log | head -2) && return 0
    sleep 10; done; return 1
}
safe_flash() {  # flash, require esptool's 3 hash checks and a booted gateway; 3 tries
  local try out t
  for try in 1 2 3; do
    systemctl stop onepass-gwlog; out=$($L/fl.sh gw $1 2>&1); systemctl start onepass-gwlog; t=$(date +%s)
    echo "$out" >> $R
    if [ $(echo "$out" | grep -c "Hash of data verified") -ge 1 ] && sleep 45 && alive $t; then
      echo "BUILD $1 flashed $t verified (try $try)" >> $R; return 0; fi
    echo "FLASH CHECK FAILED $1 try $try $(date +%s)" >> $R; sleep 30
  done
  return 1
}
for b in ${1:-veto ord}; do
  if ! safe_flash fair_$b; then  # leave the gateway on a known-good build and skip this one (reported as not run)
    echo "BUILD fair_$b SKIPPED: flash failed 3 times" >> $R; safe_flash fair_veto; continue; fi
  for r in 1 2 3; do run1 $b clean $((r + ${OFF:-0})); done
  for r in 1 2 3 4 5; do q=$((r + ${OFF:-0})); run1 $b A $q; run1 $b B $q; run1 $b E $q; run1 $b H $q; done
  for r in 1 2 3 4 5; do run1 $b DH $((r + ${OFF:-0})); done; att_nat
done
echo "FAIR DONE off ${OFF:-0} $(date +%s)" >> $R
