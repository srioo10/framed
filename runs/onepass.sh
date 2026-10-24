#!/bin/bash
# OnePass-IDS confirmatory banks (PREREG OnePass rows), 12 cores; existing configs are skipped by score_bank
cd "$(dirname "$0")/.."
run() { OMP_NUM_THREADS=12 taskset -c 0-11 .venv/bin/python -W ignore scripts/score_bank.py "$1" --seed "$2" --kind outer --variant "$3" || echo "FAILED $1 $3 s$2"; echo "DONE $1 $3 s$2 $(date +%T)"; }
run nf_bot 0 none
for s in 1 2 3 4; do for v in none host host_iso host_perm; do run nf_cic $s $v; done; done
for s in 0 1 2 3 4; do for v in none host host_perm; do run nf_ton $s $v; done; done
run nf_ton 0 host_iso
run nf_bot 0 host
for s in 1 2 3 4; do for v in none host; do run nf_bot $s $v; done; done
echo ALLDONE
