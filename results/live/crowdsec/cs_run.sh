#!/bin/bash
# PREREG LIVE-CROWDSEC, one run on Pi #1 (as root): N failed SSH logins (invalid user) from the attacker interface
# behind the sensor's NAT within 240 s, then watch 180 s for a ban on the NAT address and missed sensor publishes.
# Invalid-user is the reliable signal over the NAT hairpin; it feeds crowdsecurity/ssh-bf and the *_user-enum / slow
# leaky scenarios, all of which arm R2 shifts by m. Usage: cs_run.sh <arm> <rep> [n=10].
ARM=$1; REP=$2; N=${3:-10}; NAT=192.168.4.2; ATT=192.168.5.2; L=/home/pi4r8/livenat
t0=$(date +%s)
for i in $(seq 1 $N); do
  timeout 8 ssh -b $ATT -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=5 "labx${i}_${RANDOM}@192.168.77.1" true >/dev/null 2>&1
  sleep 1
done
t1=$(date +%s); sleep 180
ban=$(cscli decisions list -o json 2>/dev/null | grep -c "\"$NAT\"")
got=$(awk -v a=$t0 -v b=$((t0+240)) '$1>=a && $1<b' $L/sub.log | wc -l)
echo "$ARM rep=$REP t0=$t0 attack_s=$((t1-t0)) ssh_flows=$N ban_nat=$ban pubs_240s=$got/48" | tee -a $L/cs_runs.txt
cscli decisions delete --ip $NAT >/dev/null 2>&1
$L/gwreset.sh; sleep 60
