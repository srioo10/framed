"""Per-decision on-chip latency from LIVE-NAT-ENF gateway logs (F lines: ..., alarm, block, us, why).
Usage: python scripts/live/latency.py <grid_dir> results/live/natenf_latency.csv"""
import sys
from pathlib import Path
import pandas as pd

d, out = Path(sys.argv[1]), sys.argv[2]
runs = sorted((int(p[6]) * 1000, p[1]) for p in (ln.split() for ln in open(d / "runs.txt"))
              if p[:1] == ["RUN"] and p[5] == "start")
rows = []
for f in sorted(d.glob("gw-*.log")):
    for ln in f.read_bytes().decode(errors="replace").splitlines():
        p = ln.split(",")
        if len(p) != 19 or p[1] != "F" or not p[0].isdigit(): continue
        ms = int(p[0]); b = [n for t0, n in runs if t0 <= ms < t0 + 1200000]
        if b: rows.append((b[-1], int(p[15]), int(p[17])))
df = pd.DataFrame(rows, columns=["build", "alarm", "us"])
assert len(df) and df.us.min() > 0
df.to_csv(out, index=False)
print(df.groupby(["build", "alarm"]).us.describe(percentiles=[.5, .99]).round(0))
