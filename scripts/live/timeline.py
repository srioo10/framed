"""Extract a two-run live timeline (ordinary vs Veto, variant A, m=10) from LIVE-NAT-ENF grid logs for the paper figure.
Usage: python scripts/live/timeline.py <grid_dir> results/live/natenf_timeline.csv"""
import sys, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import natenf_report as R

d, out = sys.argv[1], sys.argv[2]
F, B, _ = R.gw(d); F = F[F.src == R.NAT]; B = B[B.ip == R.NAT]
pt, _ = R.pubs(d)
runs = [ln.split() for ln in open(Path(d) / "runs.txt") if ln.startswith("RUN") and " start " in ln]
# scheduled offset (s); lines logged while the address was blocked carry no timestamp, only "timed out"
bursts = [(ln.split()[:4], int(re.search(r"burst@(\d+):", ln)[1])) for ln in open(Path(d) / "inject.log") if "burst@" in ln]
rows = []
def pick(build):
    best = None
    for _, b, v, n, i, _, t0 in runs:
        if (b, v, n) != (build, "A", "10"): continue
        t0 = int(t0) * 1000; x = F[(F.ms >= t0) & (F.ms < t0 + 1200000)]; bl = B[(B.ms >= t0) & (B.ms < t0 + 1200000)]
        if build == "ord":  # first block must follow the first burst (attributable run)
            if len(bl) and bl.ms.min() >= t0 + 300000 and (best is None or bl.ms.min() < best[2]): best = (i, t0, bl.ms.min())
        elif best is None or x.score.max() > best[2]: best = (i, t0, x.score.max())
    return best[:2]
for build in ("ord", "veto"):
    i, t0 = pick(build)
    x = F[(F.ms >= t0) & (F.ms < t0 + 1200000)]
    rows += [(build, "flow", (r.ms - t0) / 1e3, r.score, r.lower) for r in x.itertuples()]
    rows += [(build, "block", (m - t0) / 1e3, None, None) for m in B[(B.ms >= t0) & (B.ms < t0 + 1200000)].ms]
    rows += [(build, "burst", off, None, None) for p, off in bursts if p == [build, "A", "10", i]]
    rows += [(build, "pub", t - t0 / 1e3, None, None) for t in pt if t0 / 1e3 <= t < t0 / 1e3 + 1200]
df = pd.DataFrame(rows, columns="build kind t score lower".split())
assert set(df.kind) == {"flow", "block", "burst", "pub"}, set(df.kind)
df.to_csv(out, index=False)
print(df.groupby(["build", "kind"]).size())
