"""CRN gate D1, pre-declared second shot (PREREG §9 CRN row): nonlinear conditional quantile score, QCAD-style.

Per feature j, LightGBM quantile regressors (q = 0.01, 0.99) predict x_j from the other features, fitted on the
benign-train sample only (never rank / cal). Score = max_j max(0, x_j - q99_j(x_-j), q01_j(x_-j) - x_j) / w_j with
each violation divided by the conditional interval width q99 - q01, floored at 5% of x_j's marginal width. Same comparison and PASS rule as crn_gates.py.
Usage: .venv/bin/python scripts/crn_gate_qcad.py ciciot_csv none --seed 0 --out results/crn/qcad
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import models  # noqa: E402
import bank_report as br  # noqa: E402
import crn_gates as cg  # noqa: E402
import score_bank as sb  # noqa: E402

N_FIT, PARAMS = 50_000, dict(n_estimators=100, num_leaves=31, learning_rate=0.1, min_child_samples=50, verbose=-1)


class QCAD:
    def fit(self, X, seed):
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), min(N_FIT, len(X)), replace=False)]
        self.cols = [j for j in range(X.shape[1]) if X[:, j].std() > 1e-6]
        self.models, self.w = [], []
        for j in self.cols:
            other = np.delete(X, j, axis=1)
            pair = [lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=seed, **PARAMS).fit(other, X[:, j])
                    for q in (0.01, 0.99)]
            self.models.append(pair)
            lo, hi = np.quantile(X[:, j], [0.01, 0.99])
            self.w.append(max(hi - lo, 0.05 * X[:, j].std(), 1e-3))
        return self

    def score(self, X):
        s = np.zeros(len(X))
        for j, (mlo, mhi), w in zip(self.cols, self.models, self.w):
            other = np.delete(X, j, axis=1)
            lo, hi = mlo.predict(other), mhi.predict(other)
            width = np.maximum(hi - lo, 0.05 * w)  # QCAD: violation relative to the conditional interval
            s = np.maximum(s, np.maximum(0, np.maximum(X[:, j] - hi, lo - X[:, j])) / width)
        return s


def one_seed(ds, variant, seed):
    bank = sb.Bank(ds, seed, variant, 40000, 200000)
    run = sb.OUT / f"{ds}_{variant}_cap40000_s{seed}"
    alpha_min = 0.01 if ds == "unsw_official" else 0.001
    shared = None  # without causal windows the benign frame does not depend on the fold: fit once
    out = []
    for k in bank.families:
        P = (k,)
        part = pd.read_parquet(run / f"config=outer_{k}" / "part.parquet")
        rows = {r: part[part.role == r] for r in ("rank", "cal", "test_benign", "test_unknown")}
        res = {"seed": seed, "fold": k, "n": len(rows["test_unknown"])}
        for m in cg.MEMBERS:
            s = {r: (br.mcde_score(rows["rank"], d, alpha_min) if m == "mcde" else d[m].to_numpy(np.float64))
                 for r, d in rows.items()}
            res[m] = cg.tpr(s["cal"], s["test_unknown"])
        if shared is None or variant.startswith("ct_causal"):
            dben = bank.frame(bank.bsample, sb.role_removal(P, "bsample"))
            prep = models.Prep().fit(dben)
            q = QCAD().fit(prep.transform(dben), seed)
            shared = (prep, q)
        prep, q = shared
        sc = {r: q.score(prep.transform(bank.frame(d.row_id.to_numpy(), sb.role_removal(P, r)))) for r, d in rows.items()}
        res["qcad"] = cg.tpr(sc["cal"], sc["test_unknown"])
        f = cg.rank_min((sc["rank"], sc), (rows["rank"].mlp_ce.to_numpy(np.float64),
                                           {r: d.mlp_ce.to_numpy(np.float64) for r, d in rows.items()}))
        res["qcad+mlp_ce"] = cg.tpr(f("cal"), f("test_unknown"))
        out.append(res)
        print(f"s{seed} {k:16s} best member {max(res[m] for m in cg.MEMBERS):.3f} | qcad {res['qcad']:.3f} "
              f"qcad+mlp {res['qcad+mlp_ce']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("variant")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.report:
        cg.NEW = ["qcad", "qcad+mlp_ce"]
        res = pd.DataFrame([r for f in sorted(a.out.parent.glob(f"{a.out.name}_{a.ds}_s*.json"))
                            for r in json.loads(f.read_text())])
        fam = res.groupby("fold")[cg.MEMBERS + cg.NEW + ["n"]].mean()
        fam["best"] = fam[cg.MEMBERS].max(1)
        hard = cg.HARD_CIC if a.ds.startswith("ciciot") else [k for k in fam.index if fam.loc[k, "best"] < 0.5]
        hard = [k for k in hard if k in fam.index and fam.loc[k, "n"] >= br.MIN_ROWS]
        L = [f"# CRN gate D1 second shot (QCAD) — {a.ds}\n", "| family | best member | qcad | qcad+mlp_ce |", "|---|---|---|---|"]
        for k, g in res.groupby("fold"):
            L.append(f"| {k}{' (hard)' if k in hard else ''} | {fam.loc[k, 'best']:.3f} | {br.fmt(*br.ci(g.qcad))} | "
                     f"{br.fmt(*br.ci(g['qcad+mlp_ce']))} |")
        gain = {k: max(fam.loc[k, "qcad"], fam.loc[k, "qcad+mlp_ce"]) - fam.loc[k, "best"] for k in hard}
        elig = [k for k in fam.index if fam.loc[k, "n"] >= br.MIN_ROWS]
        macro = res[res.fold.isin(elig)].groupby("seed")["qcad+mlp_ce"].mean()
        L.append("\ngain over best member: " + ", ".join(f"{k} {v:+.3f}" for k, v in gain.items()))
        L.append(f"families with gain >= 0.03: {sum(v >= 0.03 for v in gain.values())}; macro qcad+mlp_ce: "
                 f"{br.fmt(*br.ci(macro))}")
        text = "\n".join(L)
        print(text)
        Path(f"{a.out}_{a.ds}.md").write_text(text + "\n")
        return
    Path(f"{a.out}_{a.ds}_s{a.seed}.json").write_text(json.dumps(one_seed(a.ds, a.variant, a.seed), indent=1,
                                                                 default=float))


if __name__ == "__main__":
    main()
