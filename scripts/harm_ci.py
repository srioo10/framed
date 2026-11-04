"""SMALL-2: 95% intervals for host-day harm and for the zero-violation results (existing CSVs only, no training).
Harm: Wilson interval on k blocked of n victim host-days. Zero injected blocks: exact one-sided 95% upper bound
1 - 0.05**(1/n) (the 'rule of three' ~ 3/n) on the per-pattern rate of host-days blocked because of injection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results" / "cert"


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


out = {}
for ds in ("nf_ctu13", "nf_ugr16", "nf_cic", "nf_ton"):
    d = pd.read_csv(R / f"gate_k4_{ds}_day.csv")
    rates = np.concatenate([d.block_day.to_numpy(), d.alarm_day.to_numpy()])
    rates = rates[rates > 0]
    n = next(k for k in range(50, 2001) if np.allclose(rates * k, np.round(rates * k), atol=1e-6))  # host-days
    assert np.allclose(rates * n, np.round(rates * n), atol=1e-6), ds  # every rate is a whole number of host-days
    r = {"host_days": n}
    for det in ("gated_net", "rf", "xgb"):
        g = d[d.detector == det]
        if not len(g):
            continue
        clean = g[g.pattern == "clean"].block_day.iloc[0]
        worst = g[g.pattern != "clean"].block_day.max()
        r[det] = {"clean_block_day": clean, "clean_ci": wilson(round(clean * n), n),
                  "worst_block_day": worst, "worst_ci": wilson(round(worst * n), n),
                  "violations": int(g.violations.sum()), "patterns": int((g.pattern != "clean").sum())}
    v = r["gated_net"]
    assert v["violations"] == 0, ds
    r["zero_violation_upper95_per_pattern"] = 1 - 0.05 ** (1 / n)
    out[ds] = r
    print(ds, n, "Veto clean %.2f%% [%.2f, %.2f]" % (100 * v["clean_block_day"], *(100 * np.array(v["clean_ci"]))),
          "worst %.2f%% [%.2f, %.2f]" % (100 * v["worst_block_day"], *(100 * np.array(v["worst_ci"]))),
          "| 0 viol -> injected-block rate <= %.2f%% per pattern (95%%)" % (100 * r["zero_violation_upper95_per_pattern"]))
(R / "harm_ci.json").write_text(json.dumps(out, indent=1, default=float))
