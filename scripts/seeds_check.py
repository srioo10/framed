"""PREREG SEEDS rule over seeds 1-4: worst innocent rate >= 2x clean and >= +5 points per seed; B 0 violations."""
import sys
from pathlib import Path
import pandas as pd

C = Path(__file__).resolve().parents[1] / "results" / "cert"
BAR = {("sk", "nf_ugr16"): ("rf", "xgb"), ("sk", "nf_ctu13"): ("xgb",),
       ("opt", "nf_ugr16"): ("rf", "xgb"), ("opt", "nf_ctu13"): ("rf", "xgb")}


def cells(study, ds, s):
    sfx = "" if s == 1 else f"_s{s}"
    if study == "sk":
        f = C / f"gate_k4_{ds}_day_sharedkey{sfx}.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
        g = d.groupby("detector")
        clean = d[d.pattern == "clean"].set_index("detector").alarm_day
        return {k: (clean[k], v.alarm_day.max(), v.violations.sum()) for k, v in g}
    f = C / f"opt_harm_{ds}_opt{sfx}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f).groupby("detector")
    return {k: (v.alarm_day_clean.iloc[0], v.block_day.max(), v.violations.sum()) for k, v in d}


ok = True
for (study, ds), dets in BAR.items():
    for s in (1, 2, 3, 4):
        c = cells(study, ds, s)
        if c is None:
            print(study, ds, f"s{s}", "missing"); continue
        for k in dets:
            cl, w, _ = c[k]
            p = w >= 2 * cl and w >= cl + 0.05
            ok &= p
            print(study, ds, f"s{s}", k, f"clean {cl:.3f} worst {w:.3f}", "meets" if p else "FAILS")
        bv = c["gated_net"][2]
        ok &= bv == 0
        print(study, ds, f"s{s}", "B", f"clean {c['gated_net'][0]:.3f} worst {c['gated_net'][1]:.3f} violations {bv}")
print("RULE", "HOLDS so far" if ok else "FAILS")
