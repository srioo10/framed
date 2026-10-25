#!/bin/bash
# every 60 s: if the CPU package is > 95 C for 10 consecutive readings, SIGSTOP the score banks and the tuner (user rule)
n=0
while true; do
  t=$(sensors | awk '/Package id 0/ {gsub(/[+°C]/,"",$4); print int($4)}')
  if [ "${t:-0}" -gt 95 ]; then n=$((n+1)); else n=0; fi
  if [ $n -ge 10 ]; then pkill -STOP -f -E '^.venv/bin/python -W ignore scripts/(score_bank|tune_mlp).py'; echo "$(date +%T) PAUSED at ${t}C"; n=0; fi
  sleep 60
done
