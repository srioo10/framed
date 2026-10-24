"""unsw_raw leak-column controls (PREREG §9, 2026-10-24 row). Same LOFO folds, capped training rows, benign_cal
thresholds and test rows as the score bank (variant ct_causal). Per held-out family, TPR@1%FPR of:
  ttl     RF (100 trees) on the dropped TTL columns sttl, dttl, ct_state_ttl
  ip      RF (100 trees) on srcip, dstip as 32-bit integers
  single  a decision tree (min leaf 50) on each single feature, original and causal ct_* included;
          the best one is picked on the test rows (oracle upper bound)
Usage: .venv/bin/python scripts/leak_controls.py --seeds 0 1 2 3 4 --out results/fusion/leak_controls_unsw_raw
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import fusion_checks as fc, load, models, windows  # noqa: E402
import bank_report as br  # noqa: E402
import score_bank  # noqa: E402

TTL = ["sttl", "dttl", "ct_state_ttl"]
IP = ["srcip", "dstip"]


def ip_int(s):
    p = s.astype(str).str.replace("﻿", "").str.replace("ï»¿", "").str.strip().str.split(".", expand=True)
    return sum(pd.to_numeric(p[i], errors="coerce").fillna(0).to_numpy(np.int64) * 256 ** (3 - i) for i in range(4))


def tpr_at(model, X, roles):
    """TPR@1% (strict >, cal threshold), realised test FPR, and two tie-robust readouts: TPR with >= at the same
    threshold together with the cal FPR that implies, and pAUC over FPR <= 1% (unknown vs benign test)."""
    p = {r: model.predict_proba(X[r])[:, 1] for r in roles}
    tau = fc.threshold(p["cal"], 0.01)
    y = np.r_[np.zeros(len(p["tb"])), np.ones(len(p["unk"]))]
    return {"tpr": float((p["unk"] > tau).mean()), "test_fpr": float((p["tb"] > tau).mean()),
            "tpr_ge": float((p["unk"] >= tau).mean()), "cal_fpr_ge": float((p["cal"] >= tau).mean()),
            "pauc": float(roc_auc_score(y, np.r_[p["tb"], p["unk"]], max_fpr=0.01))}


def one_seed(seed):
    bank = score_bank.Bank("unsw_raw", seed, "ct_causal", 40000, 200000)
    raw = load.LOADERS["unsw_raw"]()[TTL + IP]
    extra = pd.DataFrame({**{c: raw[c].to_numpy()[bank.kept] for c in TTL},
                          **{c: ip_int(raw[c])[bank.kept] for c in IP}})
    del raw
    out = []
    for k in bank.families:
        fit = np.sort(np.concatenate(bank.fit_rows((k,))))
        rows = {"fit": fit, "cal": bank.cal, "tb": bank.test_benign, "unk": bank.test[k]}
        R = {"fit": (k,), "cal": (k,), "tb": (), "unk": ()}  # PREREG §9 per-role removal (outer fold k)
        y = (bank.fam[fit] != "Benign").astype(int)
        frames = {}
        for r, idx in rows.items():
            d = bank.frame(idx, R[r])
            orig = bank.df.iloc[idx][list(windows.KEYS)].reset_index(drop=True).add_prefix("orig:")
            frames[r] = pd.concat([d, orig], axis=1)
        prep = models.Prep().fit(frames["fit"])
        cols = list(frames["fit"].columns)
        X = {r: prep.transform(f) for r, f in frames.items()}
        res = {"seed": seed, "heldout": k, "n": int(len(rows["unk"]))}
        for name, c in (("ttl", TTL), ("ip", IP)):
            Xc = {r: extra.iloc[idx][c].to_numpy(np.float64) for r, idx in rows.items()}
            rf = RandomForestClassifier(100, min_samples_leaf=2, n_jobs=4, random_state=seed).fit(Xc["fit"], y)
            for m, v in tpr_at(rf, Xc, ("cal", "tb", "unk")).items():
                res[f"{name}_{m}" if m != "tpr" else name] = v
        single = {}
        for i, c in enumerate(cols):
            t = DecisionTreeClassifier(min_samples_leaf=50, random_state=seed).fit(X["fit"][:, [i]], y)
            single[c] = tpr_at(t, {r: v[:, [i]] for r, v in X.items()}, ("cal", "tb", "unk"))["tpr"]
        best = max(single, key=single.get)
        res.update({"best_single": best, "best_single_tpr": single[best], "single": single,
                    "best_noct": max((c for c in single if c not in windows.KEYS and not c.startswith("orig:")),
                                     key=single.get)})
        res["best_noct_tpr"] = single[res["best_noct"]]
        out.append(res)
        print(f"s{seed} {k:18s} ttl {res['ttl']:.3f} ip {res['ip']:.3f} best {best} {single[best]:.3f} "
              f"best non-ct {res['best_noct']} {res['best_noct_tpr']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = pd.DataFrame([r for s in a.seeds for r in one_seed(s)])
    a.out.with_suffix(".json").write_text(json.dumps(res.to_dict("records"), indent=1, default=float))
    L = ["# unsw_raw leak-column controls, TPR@1%FPR on the held-out family (mean [95% t-CI] over seeds)\n",
         "best single feature = oracle pick on the test rows; `orig:` = the creators' ct_*, plain ct_* = causal.\n",
         "| held-out | n | TTL-only RF | IP-only RF | best single feature (most common) | its TPR | best non-ct_* feature | its TPR |",
         "|---|---|---|---|---|---|---|---|"]
    tie = ["\n## Tie-robust readouts for the TTL and IP controls (mean over seeds)\n",
           "TPR with >= at the cal threshold, the cal FPR that >= implies, and pAUC over FPR <= 1% (0.5 = chance).\n",
           "| held-out | TTL TPR(>=) | TTL cal FPR(>=) | TTL pAUC | IP TPR(>=) | IP cal FPR(>=) | IP pAUC |",
           "|---|---|---|---|---|---|---|"]
    for k, g in res.groupby("heldout"):
        tie.append(f"| {k} | " + " | ".join(f"{g[c].mean():.3f}" for c in
                   ("ttl_tpr_ge", "ttl_cal_fpr_ge", "ttl_pauc", "ip_tpr_ge", "ip_cal_fpr_ge", "ip_pauc")) + " |")
    for k, g in res.groupby("heldout"):
        L.append(f"| {k} | {int(g.n.min())} | {br.fmt(*br.ci(g.ttl))} | {br.fmt(*br.ci(g.ip))} | "
                 f"{g.best_single.mode()[0]} | {br.fmt(*br.ci(g.best_single_tpr))} | {g.best_noct.mode()[0]} | "
                 f"{br.fmt(*br.ci(g.best_noct_tpr))} |")
    L += tie
    L.append("\nrealised benign-test FPR: TTL " + ", ".join(f"{k} {v:.4f}" for k, v in
                                                          res.groupby("heldout").ttl_test_fpr.mean().items()))
    top = pd.DataFrame(list(res.single)).assign(heldout=res.heldout).groupby("heldout").mean()
    L.append("\n## Top 5 single features per held-out family (mean TPR@1% over seeds)\n")
    for k, row in top.iterrows():
        L.append(f"- {k}: " + ", ".join(f"{c} {v:.3f}" for c, v in row.sort_values(ascending=False)[:5].items()))
    text = "\n".join(L)
    print(text)
    a.out.with_suffix(".md").write_text(text + "\n")


if __name__ == "__main__":
    main()
