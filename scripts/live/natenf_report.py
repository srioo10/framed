"""PREREG LIVE-NAT-ENF (2026-11-12 rows): tau from the clean calibration hour, and per-run results vs the bars.
Inputs copied from Pi #1: gw-*.log (gateway console, first field = Pi ms), runs.txt, sub.log (broker side, unix s
then the sensor JSON), inject.log. Usage:
  python natenf_report.py tau DIR CAL_START_MS      -> 99th percentile of per-flow scores in [start, start + 3600 s)
  python natenf_report.py runs DIR                  -> results/live/natenf_runs.csv and the bar summary"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAT = "192.168.4.2"


def gw(d):
    F, B, S = [], [], []
    for f in sorted(Path(d).glob("gw-*.log")):
        for ln in f.read_bytes().decode(errors="replace").splitlines():
            p = ln.split(",")
            if len(p) < 3 or not p[0].isdigit():
                continue
            if p[1] == "F" and len(p) == 19:
                F.append((int(p[0]), p[3], p[5], int(p[6]), float(p[13]), float(p[14]), int(p[16])))
            elif p[1] == "B" and len(p) == 5:
                B.append((int(p[0]), p[3]))
            elif p[1] == "S" and len(p) == 15:
                S.append((int(p[0]), int(p[9]), int(p[14])))
    return (pd.DataFrame(F, columns="ms src dst dport score lower block".split()),
            pd.DataFrame(B, columns=["ms", "ip"]), pd.DataFrame(S, columns=["ms", "hevict", "abst"]))


def pubs(d):
    t, s = [], []
    for ln in (Path(d) / "sub.log").read_text().splitlines():
        a, _, b = ln.partition(" ")
        try:
            s.append(json.loads(b)["seq"]); t.append(float(a))
        except (ValueError, KeyError):
            pass
    return np.array(t), np.array(s)


def tau(d, start_ms):
    F = gw(d)[0]
    sc = F.score[(F.ms >= start_ms) & (F.ms < start_ms + 3_600_000)].to_numpy()
    print(f"calibration flows {len(sc)}, max {sc.max():.6g}, tau (p99) = {np.percentile(sc, 99):.6g}")


def runs(d):
    F, B, S = gw(d)
    pt, ps = pubs(d)
    rows, cur = [], {}
    for ln in (Path(d) / "runs.txt").read_text().splitlines():
        p = ln.split()
        if p[:1] != ["RUN"]:
            continue
        k = tuple(p[1:5])
        if p[5] == "start":
            cur[k] = int(p[6])
            continue
        a, b = cur.pop(k) * 1000, int(p[6]) * 1000
        m = (pt >= a / 1000) & (pt < b / 1000)
        t, s = pt[m], ps[m]
        gaps = np.diff(t) if len(t) > 1 else np.array([])
        nb = B[(B.ms >= a) & (B.ms < b) & (B.ip == NAT)]
        sens = F[(F.ms >= a) & (F.ms < b) & (F.src == NAT) & (F.dport == 1883)]
        sw = S[(S.ms >= a) & (S.ms < b)]
        rows.append({"build": k[0], "variant": k[1], "n": int(k[2]), "rep": int(k[3]), "start": a // 1000,
                     "nat_blocks": len(nb), "first_block_s": (nb.ms.min() - a) / 1000 if len(nb) else np.nan,
                     "pubs": len(t), "missed": int(np.clip(np.diff(s) - 1, 0, None).sum()) if len(s) > 1 else 0,
                     "downtime_s": float(np.clip(gaps - 5, 0, None)[gaps > 7.5].sum()),
                     "sensor_score_max": sens.score.max(), "sensor_lower_max": sens.lower.replace(-np.inf, np.nan).max(),
                     "host_evictions": int(sw.hevict.max() - sw.hevict.min()) if len(sw) else 0,
                     "abstains": int(sw.abst.max()) if len(sw) else 0})
    R = pd.DataFrame(rows)
    out = ROOT / "results" / "live"
    out.mkdir(exist_ok=True)
    R.to_csv(out / "natenf_runs.csv", index=False)
    print(R.to_string())
    hit = (R.nat_blocks > 0) & (R.missed > 0)
    for v in "AB":
        o = R[(R.build == "ord") & (R.variant == v) & (R.n == 10)]
        print(f"bar1 ordinary variant {v} in budget: {int(hit[o.index].sum())}/{len(o)} runs block NAT + miss (need >= 4/5)")
    vi = R[(R.build == "veto") & ((R.n == 10) | (R.variant == "clean"))]
    print(f"bar2 Veto NAT blocks in-budget + clean runs: {int(vi.nat_blocks.sum())} total over {len(vi)} runs (need 0)")
    oc = R[(R.build == "ord") & (R.variant == "clean")]
    print(f"bar3 ordinary clean runs with 0 blocks: {int((oc.nat_blocks == 0).sum())}/{len(oc)} (need >= 2/3)")


if __name__ == "__main__":
    # self-check of the miss counter: seq 1,2,5 -> 2 missed
    assert int(np.clip(np.diff(np.array([1, 2, 5])) - 1, 0, None).sum()) == 2
    tau(sys.argv[2], int(sys.argv[3])) if sys.argv[1] == "tau" else runs(sys.argv[2])
