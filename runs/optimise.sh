#!/bin/bash
# Optimisation steps 1-3 (PREREG 2026-10-25 row), run after runs/onepass.sh has finished; one job at a time
cd "$(dirname "$0")/.."
PY=".venv/bin/python -W ignore"
until sed -n '/^RESTART2/,$p' runs/logs/onepass.out | grep -q ALLDONE; do sleep 60; done
echo "START $(date +%T)"
tune() { OMP_NUM_THREADS=10 taskset -c 0-9 $PY scripts/tune_mlp.py "$1" --seed "$2" --variant host || echo "FAILED tune $1 s$2"; echo "DONE tune $1 s$2 $(date +%T)"; }
for s in 1 2 3 4; do tune nf_cic $s; done
for s in 0 1 2 3 4; do tune nf_bot $s; done
for s in 0 1 2 3 4; do tune nf_ton $s; done
$PY scripts/onepass_claims.py nf_cic --seeds 1 2 3 4 --out results/onepass/nf_cic > /dev/null || echo "FAILED claims nf_cic day"
$PY scripts/onepass_claims.py nf_cic --seeds 1 2 3 4 --unit hour --out results/onepass/nf_cic_hour > /dev/null || echo "FAILED claims nf_cic hour"
$PY scripts/onepass_claims.py nf_ton --seeds 1 2 3 4 --unit hour --out results/onepass/nf_ton_hour > /dev/null || echo "FAILED claims nf_ton hour"
$PY scripts/onepass_claims.py nf_ton --seeds 0 1 2 3 4 --out results/onepass/nf_ton_day_exploratory > /dev/null || echo "FAILED claims nf_ton day"
$PY scripts/host_report.py --ds nf_cic --variants none host host_iso host_perm --out results/onepass/flow_nf_cic > /dev/null || echo "FAILED flow nf_cic"
$PY scripts/host_report.py --ds nf_ton --variants none host host_iso host_perm --out results/onepass/flow_nf_ton > /dev/null || echo "FAILED flow nf_ton"
$PY scripts/host_report.py --ds nf_bot --variants none host --out results/onepass/flow_nf_bot > /dev/null || echo "FAILED flow nf_bot"
echo "OPTDONE $(date +%T)"
