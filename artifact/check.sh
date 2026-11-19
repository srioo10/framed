#!/bin/bash
# One-command check of the live claims, no hardware: re-derive every logged gateway decision from its flow records
# (engine/replay_live.c, the deployed decision path) and re-run the framing-witness search. Needs cc and python3+pandas.
# Usage: bash artifact/check.sh   (from the repo root; ~1 min on a laptop)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
echo "== 1. replay: every live score and block decision (expect 0 mismatches)"
$PY scripts/live/replay_check.py results/live/fair 0.442 | tee /tmp/fp_replay.txt
! grep -v " mismatches 0, block-decision mismatches 0" /tmp/fp_replay.txt | grep -q runs
echo "== 2. framing witnesses on the Veto build (alerts per attack variant; expect 0 in clean runs)"
$PY scripts/live/witness.py results/live/fair veto 0.442 --v2 > /tmp/fp_wit.txt
grep "^R" /tmp/fp_wit.txt | awk '{k=$3; n[k]++; if ($8 > 0) a[k]++} END {for (k in n) print " ", k, "runs", n[k], "with alert", a[k]+0}' | sort
awk '$1=="R" && $3=="clean" && $8>0 {bad=1} END {exit bad}' /tmp/fp_wit.txt
grep "^G" /tmp/fp_wit.txt | awk '$5=="real" {n+=$10; i+=$11} END {print "  connections named for the sensor'"'"'s alarmed flows:", n, "of them injected:", i}'
echo "OK"
