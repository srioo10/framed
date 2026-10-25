#!/bin/bash
# every 60 s: log CPU package temperature, free memory, swap and the job's RSS to runs/logs/monitor.csv.
# Heat rule (user): > 95 C for 10 consecutive readings -> SIGSTOP the jobs; resume them once <= 85 C.
cd "$(dirname "$0")/.."
PAT='^.venv/bin/python -W ignore scripts/(score_bank|tune_mlp|onepass_claims|host_stream|host_report).py'
n=0; paused=0
[ -f runs/logs/monitor.csv ] || echo "time,temp_c,mem_avail_gb,swap_used_gb,job_rss_gb,paused" > runs/logs/monitor.csv
while true; do
  t=$(sensors | awk '/Package id 0/ {gsub(/[+°C]/,"",$4); print int($4)}')
  avail=$(free -m | awk '/^Mem/ {printf "%.1f", $7/1024}')
  swap=$(free -m | awk '/^Swap/ {printf "%.1f", $3/1024}')
  rss=$(pgrep -f "$PAT" | xargs -r ps -o rss= -p | awk '{s+=$1} END {printf "%.1f", s/1048576}')
  if [ "${t:-0}" -gt 95 ]; then n=$((n+1)); else n=0; fi
  if [ $paused -eq 0 ] && [ $n -ge 10 ]; then pkill -STOP -f "$PAT"; paused=1; echo "$(date +%T) PAUSED at ${t}C" >> runs/logs/heatguard.out; fi
  if [ $paused -eq 1 ] && [ "${t:-99}" -le 85 ]; then pkill -CONT -f "$PAT"; paused=0; n=0; echo "$(date +%T) RESUMED at ${t}C" >> runs/logs/heatguard.out; fi
  echo "$(date +%F\ %T),$t,$avail,$swap,$rss,$paused" >> runs/logs/monitor.csv
  sleep 60
done
