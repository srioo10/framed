"""PREREG HYBRID (2026-11-10): alarm on RF, block only on Veto's certificate. Characterisation from existing files,
no new training: K19 throttle runs (RF and Veto from the same run, seed 1, pair 0) and the K4 host-day attack runs.
Writes results/cert/hybrid.csv."""
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "cert"
rows = []
for ds in ("nf_ctu13", "nf_ugr16", "nf_cic"):
    T = pd.read_csv(R / f"gate_throttle_{ds}_k19.csv", dtype={"k": str}).set_index("k")
    K = pd.read_csv(R / f"gate_k4_{ds}_day.csv")
    rf, b = K[K.detector == "rf"], K[K.detector == "gated_net"]
    r = {"dataset": ds}
    for s in ("unk", "known", "macro"):  # unthrottled attack: alarm channel = RF, block channel = Veto's certificate
        r[f"alarm_tpr_{s}"], r[f"block_tpr_{s}"] = T.loc["inf", f"rf_tpr_{s}"], T.loc["inf", f"gated_net_btpr_{s}"]
    for k in ("1", "10", "50"):  # throttled sources: share still alarmed (RF) / blocked (Veto)
        r[f"alarm_k{k}"], r[f"block_k{k}"] = T.loc[k, "rf_tpr_macro"], T.loc[k, "gated_net_btpr_macro"]
    r["harm_clean"] = b[b.pattern == "clean"].block_day.iloc[0]  # blocks come only from Veto
    r["harm_worst"] = b[b.pattern != "clean"].block_day.max()
    r["alarm_day_clean"] = rf[rf.pattern == "clean"].alarm_day.iloc[0]  # the RF alarm channel is itself poisonable
    r["alarm_day_worst"] = rf[rf.pattern != "clean"].alarm_day.max()
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(R / "hybrid.csv", index=False)
print(D.round(3).T.to_string())
