#!/bin/bash
# H0 attribution, GAMO, one-at-a-time from paper order (+ all fixed); cores ${CORES:-10-15}
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NIDS_DATA_DIR=${NIDS_DATA_DIR:-$PWD}
for seed in ${SEEDS:-0}; do for ds in unsw_official bot_iot ciciot_paper; do
  steps="none a b c d g abcdg"; [ $ds = bot_iot ] && steps="none a b c d f g abcdfg"
  for f in $steps; do
    fx=$f; [ $f = none ] && fx=""
    out=results/attrib/${ds}_gamo_fixed-${f}_s${seed}.json; [ -f $out ] && continue
    log=runs/logs/attrib_${ds}_${f}_s${seed}.log
    taskset -c ${CORES:-10-15} .venv/bin/python scripts/arlhnids_attrib.py $ds --fixed "$fx" --seed $seed > $log 2>&1
    grep -- "-> " $log || { echo "FAILED $ds $f s$seed"; tail -n 5 $log; }
  done; done; done
