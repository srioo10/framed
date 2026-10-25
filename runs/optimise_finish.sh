#!/bin/bash
# Finish the optimisation (user: "don't stop, we have to finish it"): after runs/optimise.sh ends, re-tune every
# (dataset, seed) whose tuned folds are incomplete (tune_mlp skips finished folds), up to 10 rounds; then rebuild
# the claim and flow tables once all folds exist (or the rounds run out).
cd "$(dirname "$0")/.."
PY=".venv/bin/python -W ignore"
B=/home/srioo/Desktop/Projects/root/BigP/runs/bank
SETS="nf_cic:1 nf_cic:2 nf_cic:3 nf_cic:4 nf_bot:0 nf_bot:1 nf_bot:2 nf_bot:3 nf_bot:4 nf_ton:0 nf_ton:1 nf_ton:2 nf_ton:3 nf_ton:4"
missing() { for x in $SETS; do d=${x%:*}; s=${x#*:}; r=$B/${d}_host_cap40000_s$s
  [ "$(ls -d $r/config=outer_* | wc -l)" -eq "$(ls $r/config=outer_*/tuned.parquet 2>/dev/null | wc -l)" ] || echo $x; done; }
while pgrep -f runs/optimise.sh > /dev/null; do sleep 60; done
for i in $(seq 1 10); do
  todo=$(missing); [ -z "$todo" ] && break
  echo "ROUND $i $(date +%T): $todo" >> runs/logs/optimise.out
  for x in $todo; do OMP_NUM_THREADS=10 taskset -c 0-9 $PY scripts/tune_mlp.py ${x%:*} --seed ${x#*:} --variant host >> runs/logs/optimise.out 2>&1 || echo "FAILED tune $x round $i" >> runs/logs/optimise.out; done
done
echo "MISSING after rounds: $(missing)" >> runs/logs/optimise.out
rm -f results/onepass/*/*mlp_tuned* results/onepass/*/*mlp_bal* results/onepass/*/*mlp_smote*
$PY scripts/onepass_claims.py nf_cic --seeds 1 2 3 4 --out results/onepass/nf_cic >> runs/logs/optimise.out 2>&1 || echo "FAILED claims nf_cic day"
$PY scripts/onepass_claims.py nf_cic --seeds 1 2 3 4 --unit hour --out results/onepass/nf_cic_hour >> runs/logs/optimise.out 2>&1 || echo "FAILED claims nf_cic hour"
$PY scripts/onepass_claims.py nf_ton --seeds 1 2 3 4 --unit hour --out results/onepass/nf_ton_hour >> runs/logs/optimise.out 2>&1 || echo "FAILED claims nf_ton hour"
$PY scripts/onepass_claims.py nf_ton --seeds 0 1 2 3 4 --out results/onepass/nf_ton_day_exploratory >> runs/logs/optimise.out 2>&1 || echo "FAILED claims nf_ton day"
$PY scripts/host_report.py --ds nf_cic --variants none host host_iso host_perm --out results/onepass/flow_nf_cic >> runs/logs/optimise.out 2>&1 || echo "FAILED flow nf_cic"
$PY scripts/host_report.py --ds nf_ton --variants none host host_iso host_perm --out results/onepass/flow_nf_ton >> runs/logs/optimise.out 2>&1 || echo "FAILED flow nf_ton"
$PY scripts/host_report.py --ds nf_bot --variants none host --out results/onepass/flow_nf_bot >> runs/logs/optimise.out 2>&1 || echo "FAILED flow nf_bot"
echo "FINISHED $(date +%T)" >> runs/logs/optimise.out
