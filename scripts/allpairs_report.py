"""PREREG ALLPAIRS (Kaggle amendment 2026-11-18): per run, clean and worst-pattern innocent host-days blocked (Veto)
/ alarmed (RF, XGBoost) with 95% Wilson CIs, Veto violations, and the range over runs.
Usage: python scripts/allpairs_report.py   (reads results/cert/gate_k4_{ds}_day_{pair*,grid_s*}.csv + allpairs_logs/)"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results/cert"


def n_days(ds, tag):
    kind = tag.replace("pair", "allp").replace("grid_s", "gseed")
    log = R / "allpairs_logs" / f"bigp-conf-{ds.replace('_', '-')}-{kind}.log"
    m = re.search(r"victims=\d+ flows / (\d+) host-days", log.read_text())
    return int(m.group(1))


def wilson(k, n, z=1.96):  # same as scripts/harm_ci.py (which runs on import)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ci(p, n):
    lo, hi = wilson(round(p * n), n)
    return f"{100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    out = []
    for f in sorted(glob.glob(str(R / "gate_k4_*_day_pair*.csv")) + glob.glob(str(R / "gate_k4_*_day_grid_s*.csv"))):
        ds, tag = re.match(r"gate_k4_(nf_\w+?)_day_(\w+)\.csv", Path(f).name).groups()
        d, n = pd.read_csv(f), n_days(ds, tag)
        att = d[(d.pattern != "clean") & (d.pattern != "adaptive")]
        r = {"ds": ds, "run": tag, "host_days": n}
        for det, col in (("gated_net", "block_day"), ("rf", "alarm_day"), ("xgb", "alarm_day")):
            c, w = d[(d.detector == det) & (d.pattern == "clean")][col].iloc[0], att[att.detector == det][col].max()
            r[f"{det}_clean"], r[f"{det}_worst"] = ci(c, n), ci(w, n)
        v = d[d.detector == "gated_net"]
        r["veto_violations"] = int(v.violations.sum())
        r["veto_clean_alarm"] = round(float(v[v.pattern == "clean"].alarm_day.iloc[0]), 4)
        out.append(r)
    t = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    print("Veto violations, all runs:", int(t.veto_violations.sum()), "->", "PASS" if t.veto_violations.sum() == 0 else "FAIL")
    t.to_csv(R / "allpairs_summary.csv", index=False)


if __name__ == "__main__":
    main()
