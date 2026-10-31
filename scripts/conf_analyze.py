"""CONF claims C1 / C2 / C5 (PREREG §9 CONF protocol, amendments 3-5) from the CONF result files.

Unit = pair (dataset/pair_i); a model's pair value = mean over seeds 2-4; paired differences per pair; 95% percentile CI
from 10,000 bootstrap resamples of pairs (RNG 0), pooled over datasets. A claim is "partial" (decides nothing) until
every pair has all seeds for every model it uses.
Usage: .venv/bin/python scripts/conf_analyze.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS, B_RES = (2, 3, 4), 10_000
DIRS = [ROOT / "results" / "conf", ROOT / "results" / "cert"]


def rows(pattern):
    out = []
    for d in DIRS:
        for f in sorted(glob.glob(str(d / pattern))):
            out += [json.loads(line) for line in open(f) if line.strip()]
    return pd.DataFrame(out)


def pair_means(df, key, val):
    """pair x key table of seed means, plus a completeness flag (all seeds present)."""
    m = df.groupby(["pair", key])[val].mean().unstack()
    n = df.groupby(["pair", key]).seed.nunique().unstack()
    return m, (n == len(SEEDS))


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (B_RES, len(d)))].mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(name, d, pairs, complete):
    d = pd.Series(np.asarray(d, float), index=pairs).dropna()  # partial runs: pairs missing a config
    complete, pairs, d = complete and len(d) == 24, d.index, d.to_numpy()
    m, lo, hi = boot(d)
    per = pd.Series(d, index=pairs).groupby(lambda p: p.split("/")[0]).mean().round(4).to_dict()
    return {"claim": name, "mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)], "n_pairs": len(d),
            "complete": bool(complete), "per_dataset": per}


def main():
    res = {}
    C = rows("clean_*.jsonl")
    fit, okf = pair_means(C, "model", "fitness")
    unk, _ = pair_means(C, "model", "unk")
    boss = max(("rf", "xgb"), key=lambda b: fit[b].mean())
    d1 = fit["A"] - fit[["rf", "xgb"]].max(axis=1)
    c1 = fmt("C1 A - max(RF,XGB) fitness", d1, fit.index, okf[["A", "rf", "xgb"]].all().all() and len(fit) == 24)
    c1.update(unk_A=round(unk["A"].mean(), 4), unk_better_boss=round(unk[boss].mean(), 4), better_boss=boss,
              passed=bool(c1["ci"][0] > -0.02 and unk["A"].mean() >= unk[boss].mean()))
    res["C1"] = c1

    Bl = rows("baselines_*.jsonl")
    bfit, okb = pair_means(Bl, "model", "fitness")
    best = bfit.mean().idxmax()
    c5 = fmt(f"C5 A - {best} fitness", fit["A"] - bfit[best], fit.index, okb.all().all() and len(bfit) == 24)
    c5.update(best_baseline=best, baseline_means=bfit.mean().round(4).to_dict(), passed=bool(c5["ci"][0] > 0))
    res["C5"] = c5

    K = rows("conf_k5b_*.jsonl").drop_duplicates(["pair", "seed"], keep="last")
    T = pd.concat([rows("conf_trees_*.jsonl"), rows("conf_groot_*.jsonl")], ignore_index=True)
    if len(T):
        T = T.drop_duplicates(["pair", "seed", "model", "robust_trained"], keep="last")
        for p in ("unk", "known"):
            T[f"proof_{p}"] = T[f"milp_{p}"] - T[f"undecided_{p}"] / T[f"n_{p}"]
        T["proof"] = (T.proof_unk + T.proof_known) / 2
        cfg = []
        for (mdl, rt), g in T.groupby(["model", "robust_trained"]):
            for v in ("pertree", "proof", "milp"):
                cfg.append(g.assign(cfg=f"{mdl}{'_box' if rt and mdl != 'groot' else ''}:{v}", val=g[v]))
        T = pd.concat(cfg)
        tv, okt = pair_means(T, "cfg", "val")
        kv, okk = pair_means(K.melt(["pair", "seed"], ["bab", "babk5", "ibp"], "cfg", "val"), "cfg", "val")
        common = tv.index.intersection(kv.index)
        tv, kv = tv.loc[common], kv.loc[common]
        proof_cfgs = [c for c in tv if not c.endswith(":milp")]
        best_t = tv[proof_cfgs].mean().idxmax()
        complete = len(common) == 24 and okt.loc[common].all().all() and okk.loc[common].all().all()
        c2 = fmt(f"C2 B(BaB proof) - {best_t}", kv["bab"] - tv[best_t], common, complete)
        c2.update(best_tree=best_t, tree_means=tv.mean().round(4).to_dict(), passed=bool(c2["ci"][0] > 0))
        res["C2"] = c2
        res["C2_ibp_deployed"] = fmt(f"B(IBP) - {best_t}", kv["ibp"] - tv[best_t], common, complete)
        k5 = [c for c in tv if c.endswith(":milp")]
        best_k5 = tv[k5].mean().idxmax()
        res["C2_k5_convention"] = fmt(f"B(BaB K5) - {best_k5}", kv["babk5"] - tv[best_k5], common, complete)
    out = ROOT / "results" / "conf" / "conf_summary.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        tag = "" if v["complete"] else " [PARTIAL: decides nothing]"
        print(f"{k}: {v['claim']} mean {v['mean']:+.4f} CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] n={v['n_pairs']}"
              f"{' PASS' if v.get('passed') else (' FAIL' if 'passed' in v else '')}{tag}\n   per dataset {v['per_dataset']}")
        for x in ("better_boss", "unk_A", "unk_better_boss", "best_baseline", "baseline_means", "best_tree"):
            if x in v:
                print(f"   {x}: {v[x]}")


if __name__ == "__main__":
    main()
