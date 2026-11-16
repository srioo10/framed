#!/bin/bash
# PREREG LIVE-CROWDSEC batch on Pi #1 (as root): arm R1 (private-range whitelist removed) x5, restore the whitelist,
# arm R2 (+m capacity shift) in/over budget x5 each, arm R0 (defaults) x5, stop CrowdSec, then start the LIVE-NAT-ENF grid. Restoring needs the hub (lab proxy on :1080).
L=/home/pi4r8/livenat
for r in 1 2 3 4 5; do $L/cs_run.sh R1 $r; done
S=/etc/crowdsec/scenarios; mkdir -p $L/scen_bak; cp $S/ssh-bf.yaml $S/ssh-slow-bf.yaml $S/ssh-cve-2024-6387.yaml $L/scen_bak/
sed -i 's/^capacity: 5$/capacity: 15/' $S/ssh-bf.yaml; sed -i 's/^capacity: 10$/capacity: 20/' $S/ssh-slow-bf.yaml
sed -i 's/^capacity: 3$/capacity: 13/' $S/ssh-cve-2024-6387.yaml
grep -h "^capacity" $S/ssh-bf.yaml $S/ssh-slow-bf.yaml $S/ssh-cve-2024-6387.yaml | tr '\n' ' ' >> $L/cs_runs.txt; echo " (R2 capacities)" >> $L/cs_runs.txt
systemctl restart crowdsec; sleep 10
for r in 1 2 3 4 5; do $L/cs_run.sh R2-in $r 10; done
for r in 1 2 3 4 5; do $L/cs_run.sh R2-over $r 30; done
cp $L/scen_bak/*.yaml $S/; systemctl restart crowdsec; sleep 10
HTTPS_PROXY=socks5://127.0.0.1:1080 cscli parsers install crowdsecurity/whitelists >> $L/cs_runs.txt 2>&1
systemctl restart crowdsec; sleep 10
cscli parsers list | grep -q whitelists && for r in 1 2 3 4 5; do $L/cs_run.sh R0 $r; done || echo "R0 skipped: whitelist not restored" >> $L/cs_runs.txt
systemctl stop crowdsec-firewall-bouncer crowdsec
echo "CROWDSEC DONE $(date +%s)" >> $L/cs_runs.txt
exec $L/run_grid.sh
