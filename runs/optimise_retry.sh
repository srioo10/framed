#!/bin/bash
# Keep the optimisation going (user: "don't stop no matter what"): after runs/optimise.sh ends, rerun it while any
# tune step failed (tune_mlp skips finished folds), up to 3 retries; tuned-arm stream caches are rebuilt each time.
cd "$(dirname "$0")/.."
while pgrep -f runs/optimise.sh > /dev/null; do sleep 60; done
for i in 1 2 3; do
  sed -n '/^RESUME/,$p' runs/logs/optimise.out | grep -q "FAILED" || break
  rm -f results/onepass/*/*mlp_tuned* results/onepass/*/*mlp_bal* results/onepass/*/*mlp_smote*
  echo "RESUME retry $i $(date +%T)" >> runs/logs/optimise.out
  runs/optimise.sh >> runs/logs/optimise.out 2>&1
done
echo "RETRYDONE $(date +%T)" >> runs/logs/optimise.out
