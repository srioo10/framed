#!/bin/bash
# PREREG LIVE-CROWDSEC, one run on Pi #1 (as root): 10 failed SSH logins (invalid user) from the attacker interface
# behind the sensor's NAT within 240 s, then watch 180 s for a ban on the NAT address and missed sensor publishes.
# Usage: cs_run.sh <arm R0|R1> <rep>. Appends one line to ~pi4r8/livenat/cs_runs.txt.
ARM=$1; REP=$2; NAT=192.168.4.2; ATT=192.168.5.2; L=/home/pi4r8/livenat
t0=$(date +%s)
for i in $(seq 1 10); do
  timeout 8 ssh -b $ATT -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=5 "labx$i@192.168.77.1" true >/dev/null 2>&1
  sleep 1
done
t1=$(date +%s); sleep 180
ban=$(cscli decisions list -o json 2>/dev/null | grep -c "\"$NAT\"")
# sensor publishes received at the broker in [t0, t0+240): 48 expected at one per 5 s
got=$(awk -v a=$t0 -v b=$((t0+240)) '$1>=a && $1<b' $L/sub.log | wc -l)
echo "$ARM rep=$REP t0=$t0 attack_s=$((t1-t0)) ssh_flows=10 ban_nat=$ban pubs_240s=$got/48" | tee -a $L/cs_runs.txt
cscli decisions delete --ip $NAT >/dev/null 2>&1
$L/gwreset.sh; sleep 60  # sensor rejoins the gateway and the broker before the next run
