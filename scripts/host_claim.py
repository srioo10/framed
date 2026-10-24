"""Host-context claim rule (PREREG §9 host-context row) on the report rows (scripts/host_report.py --out X writes
X_rows.csv). gain = best-member TPR@1%FPR(host) - (none), per family and seed. Checks, as pre-registered:
(1) gain >= 0.03 on >= 2 families (5-seed paired CI > 0 when >= 2 seeds exist, else "n/a: 1 seed");
(2) host_iso keeps >= half of the gain; (3) host_perm loses the gain, read here as perm gain < 0.03 and
< half of the host gain (the PREREG does not give a number). Same-member gains reported alongside.
Usage: .venv/bin/python scripts/host_claim.py results/host/nf_cic_rows.csv
"""
import sys

import pandas as pd

sys.path.insert(0, "scripts")
import bank_report as br  # noqa: E402

res = pd.read_csv(sys.argv[1])
best = res.loc[res.groupby(["variant", "seed", "fold"]).tpr.idxmax()].set_index(["variant", "seed", "fold"])
seeds = sorted(res.seed.unique())
rows = []
for k in sorted(res.fold.unique()):
    b = {v: best.xs((v, k), level=("variant", "fold")) for v in ("none", "host", "host_iso", "host_perm")}
    g = {v: (b[v].tpr - b["none"].tpr) for v in ("host", "host_iso", "host_perm")}
    gh = g["host"].mean()
    ci = br.fmt(*br.ci(g["host"])) if len(seeds) > 1 else "n/a (1 seed)"
    rows.append({
        "family": k,
        **{f"TPR {v}": f"{b[v].tpr.mean():.3f} ({b[v].member.iloc[0]})" for v in b},
        **{f"FPR {v}": f"{b[v].fpr_test.mean():.4f}" for v in b},
        "host gain": f"{gh:+.3f}", "gain CI": ci,
        "iso gain": f"{g['host_iso'].mean():+.3f}", "perm gain": f"{g['host_perm'].mean():+.3f}",
        "(1) gain>=0.03": gh >= 0.03,
        "(2) iso keeps >=1/2": bool(gh >= 0.03 and g["host_iso"].mean() >= gh / 2),
        "(3) perm loses": bool(gh >= 0.03 and g["host_perm"].mean() < 0.03 and g["host_perm"].mean() < gh / 2),
    })
t = pd.DataFrame(rows).set_index("family")
cols1 = [c for c in t.columns if c.startswith("TPR")]
cols2 = [c for c in t.columns if c.startswith("FPR")]
cols3 = ["host gain", "gain CI", "iso gain", "perm gain", "(1) gain>=0.03", "(2) iso keeps >=1/2", "(3) perm loses"]
print(f"seeds {seeds}\n\nTPR@1%FPR, best member (name)\n", t[cols1].to_string(), "\n\nrealised benign-test FPR of that "
      "member (target 0.01)\n", t[cols2].to_string(), "\n\nclaim checks\n", t[cols3].to_string())
n1 = int(t["(1) gain>=0.03"].sum())
n_all = int((t["(1) gain>=0.03"] & t["(2) iso keeps >=1/2"] & t["(3) perm loses"]).sum())
print(f"\nfamilies with gain >= 0.03: {n1}; passing all three checks: {n_all}; rule needs >= 2"
      + ("" if len(seeds) > 1 else " (and a 5-seed CI > 0, not available with 1 seed)"))
sm = res.pivot_table(index=["fold", "member"], columns="variant", values="tpr")
sm = (sm.sub(sm["none"], axis=0).drop(columns="none")).round(3)
print("\nsame-member gain vs none (TPR@1%FPR)\n", sm.unstack("member")["host"].to_string())
