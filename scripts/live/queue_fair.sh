#!/bin/bash
# PREREG LIVE-FAIR-REP: after the running LIVE-FAIR campaign (pid $1) exits, run replicates 2 and 3 (OFF=5, OFF=10).
L=/home/pi4r8/livenat
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "NOTE $(date +%s) LIVE-FAIR-REP queue starting replicates (OFF 5, 10)" >> $L/fair_runs.txt
OFF=5 $L/run_fair2.sh "veto ord" 1
OFF=10 $L/run_fair2.sh "veto ord" 1
