"""Steps 2-3 of the 2026-10-25 PREREG row: nested-LOFO tuning of the one-pass MLP and rare-class balancing.

Per outer held-out family k: an 8-point grid (trunk x width x depth) is scored on inner pseudo-unknown families
j != k (fit minus {k, j}, flow-level TPR@1%FPR against the bank's cal benign rows); the best point is refit on the
fit rows minus k and scored on the bank's outer rows. Two balancing arms (class-weighted CE, SMOTE) use the default
MLP. Output: config=outer_k/tuned.parquet next to the bank part, plus tuned.json with the inner scores.
Usage: .venv/bin/python scripts/tune_mlp.py nf_cic --seed 1 [--variant host]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import fusion_checks as fc  # noqa: E402
from nids import models  # noqa: E402
import score_bank as sb  # noqa: E402

GRID = [(arch, w, depth) for arch in ("mlp", "sn") for w in (64, 128) for depth in (2, 3)]
FIT_CAP = 40_000


def build(arch, w, depth):
    cls = models.MLP if arch == "mlp" else models.SNMLP
    return lambda d, k: cls(d, k, w, depth)


def nparams(arch, w, depth, d, k):
    return sum(p.numel() for p in build(arch, w, depth)(d, k).parameters())


def fit(arch, w, depth, Xf, yf, Xe, ye, k, seed):
    """Returns score(X) -> unknown score, proba(X) -> class posterior, and the model."""
    m = models.train_mlp(Xf, yf, Xe, ye, k, seed=seed, arch=build(arch, w, depth)).eval()
    if arch == "sn":
        gda = models.FeatureGDA().fit(m, Xf, yf, k)
        return (lambda X: gda.score(m, X)), (lambda X: models.mlp_proba(m, X)), m
    return None, (lambda X: models.mlp_proba(m, X)), m


def tpr(cal, unk):
    th = fc.threshold(cal, 0.01)
    return float((unk > th).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", default="host")
    ap.add_argument("--families", nargs="+", help="subset of outer folds (smoke tests)")
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bank = sb.Bank(a.dataset, a.seed, a.variant, 40000, 200000)
    run = sb.OUT / f"{a.dataset}_{a.variant}_cap40000_s{a.seed}"
    log = {}
    for k in a.families or bank.families:
        dest = run / f"config=outer_{k}"
        if (dest / "tuned.parquet").exists():
            print("skip", k, flush=True)
            continue
        t0 = time.time()
        P = (k,)
        fit_r, es_r = bank.fit_rows(P)
        known = sorted(set(bank.fam[fit_r]))
        code = {f: i for i, f in enumerate(known)}
        bi, nk = code["Benign"], len(known)
        # ---- inner grid: pseudo-unknown j among the known attack families
        inner = {g: [] for g in GRID}
        for j in [f for f in known if f != "Benign" and not bank.removed(bank.cap[f], P).any()]:
            Q = tuple(sorted({k, j}))
            fi, ei = bank.fit_rows(Q)
            kn = sorted(set(bank.fam[fi]))
            cj = {f: i for i, f in enumerate(kn)}
            dfi = bank.frame(fi, sb.role_removal(Q, "fit"))
            prep = models.Prep().fit(dfi)
            Xf, yf = prep.transform(dfi), np.array([cj[f] for f in bank.fam[fi]])
            Xe = prep.transform(bank.frame(ei, sb.role_removal(Q, "es")))
            ye = np.array([cj[f] for f in bank.fam[ei]])
            Xc = prep.transform(bank.frame(bank.cal, sb.role_removal(Q, "cal")))
            pj = np.sort(np.concatenate([bank.cap[j], bank.val_attack[j]]))
            Xj = prep.transform(bank.frame(pj, sb.role_removal(Q, "pseudo", P)))
            for g in GRID:
                score, proba, _ = fit(*g, Xf, yf, Xe, ye, len(kn), a.seed)
                s = score if score else (lambda X: 1 - proba(X)[:, cj["Benign"]])
                inner[g].append(tpr(s(Xc), s(Xj)))
            if bank.host is not None:  # removal-set features are cached (~0.8 GB each on nf_ton); recompute is exact
                bank.host._cache.pop(Q, None)
        d = bank.frame(fit_r[:1], sb.role_removal(P, "fit")).shape[1]
        mean = {g: float(np.mean(v)) if v else 0.0 for g, v in inner.items()}
        best = max(GRID, key=lambda g: (round(mean[g], 4), -nparams(*g, d, nk)))
        # ---- refit on the outer fit rows: tuned model + two balancing arms
        dfit = bank.frame(fit_r, sb.role_removal(P, "fit"))
        prep = models.Prep().fit(dfit)
        Xf, yf = prep.transform(dfit), np.array([code[f] for f in bank.fam[fit_r]])
        Xe = prep.transform(bank.frame(es_r, sb.role_removal(P, "es")))
        ye = np.array([code[f] for f in bank.fam[es_r]])
        score, proba, mt = fit(*best, Xf, yf, Xe, ye, nk, a.seed)
        cnt = np.bincount(yf, minlength=nk).astype(np.float64)
        wts = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / nk, dtype=torch.float32)
        wts = wts / wts.mean()
        mbal = models.train_mlp(Xf, yf, Xe, ye, nk, seed=a.seed, weight=wts).eval()
        from imblearn.over_sampling import SMOTE
        target = {c: FIT_CAP for c in range(nk) if c != bi and 6 <= cnt[c] < FIT_CAP}
        Xs, ys = SMOTE(sampling_strategy=target, k_neighbors=5, random_state=a.seed).fit_resample(Xf, yf) \
            if target else (Xf, yf)
        msm = models.train_mlp(Xs.astype(np.float32), ys, Xe, ye, nk, seed=a.seed).eval()
        part = pd.read_parquet(dest / "part.parquet", columns=["row_id", "role", "family"])
        out = []
        for role, g in part.groupby("role", sort=False):
            X = prep.transform(bank.frame(g.row_id.to_numpy(), sb.role_removal(P, role)))
            p = proba(X)
            cols = {"row_id": g.row_id.to_numpy(), "role": role, "family": g.family.to_numpy(),
                    "mlp_tuned": score(X) if score else 1 - p[:, bi],
                    "mlp_bal": 1 - models.mlp_proba(mbal, X)[:, bi],
                    "mlp_smote": 1 - models.mlp_proba(msm, X)[:, bi]}
            if best[0] == "mlp":
                cols["mlp_tuned_int8"] = 1 - models.mlp_proba(models.int8(mt), X)[:, bi]
            if role == "test_known":
                for name, q in (("mlp_tuned", p), ("mlp_bal", models.mlp_proba(mbal, X)),
                                ("mlp_smote", models.mlp_proba(msm, X))):
                    cols[f"{name}_pred"] = np.array(known)[q.argmax(1)]
            out.append(pd.DataFrame(cols))
        res = pd.concat(out, ignore_index=True)
        for c in res.columns:
            if res[c].dtype == np.float64:
                res[c] = res[c].astype(np.float32)
        res.to_parquet(dest / "tuned.parquet", index=False)
        if bank.host is not None:
            bank.host._cache.clear()
        log[k] = {"best": list(best), "params": nparams(*best, d, nk), "inner": {str(g): mean[g] for g in GRID},
                  "smote_target": {known[c]: int(cnt[c]) for c in target}, "seconds": time.time() - t0}
        print(k, json.dumps(log[k]), flush=True)
        prev = json.loads((run / "tuned.json").read_text()) if (run / "tuned.json").exists() else {}
        (run / "tuned.json").write_text(json.dumps({**prev, k: log[k]}, indent=1))


if __name__ == "__main__":
    main()
