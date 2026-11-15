#!/bin/bash
# PREREG LIVE-SCAN: start run_scan.sh once the running LIVE-SOUND campaign (pid $1) exits.
L=/home/pi4r8/livenat
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "NOTE $(date +%s) queue: LIVE-SCAN starting" >> $L/fair_runs.txt
$L/run_scan.sh scan
echo "QUEUE DONE $(date +%s)" >> $L/fair_runs.txt
