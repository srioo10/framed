"""LOFO pilot: go/no-go for H3 before the full teacher -> student build (PLAN §5-6).

For each held-out family: train on the rest (benign always in), then score test rows of benign vs the
held-out family. Question: does a small MLP student (CE or KD from a tree teacher) give a better unknown-attack
score than simple tree baselines, above all the tree's own P(not benign)?

Pilot shortcuts (not the pre-registered protocol): one seed; family labels instead of classes; teacher is
3-fold out-of-fold XGBoost, not the 7-tree vote; fp32 only; CICIoT rows capped per family.

Usage: .venv/bin/python scripts/pilot_lofo.py DATASET [--seed 0] [--cap 40000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from torch import nn
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids import clean, load, split  # noqa: E402

FPR = 0.01  # alarm threshold set on benign validation rows only (PREREG)


class Prep:
    """Label encode -> train-mean impute -> signed log1p -> standard scale. No clamping: extreme values in an
    unseen family are signal for unknown detection."""

    def fit(self, df):
        self.codes = {c: {v: i for i, v in enumerate(pd.unique(df[c].astype(str)))}
                      for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}
        X = self._raw(df)
        self.mean = np.nanmean(X, 0)
        X = self._log(np.where(np.isnan(X), self.mean, X))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def _raw(self, df):
        X = np.column_stack([df[c].astype(str).map(self.codes[c]).to_numpy(np.float64, na_value=np.nan)
                             if c in self.codes else df[c].to_numpy(np.float64, na_value=np.nan) for c in df.columns])
        X[~np.isfinite(X)] = np.nan
        return X

    @staticmethod
    def _log(X):
        return np.sign(X) * np.log1p(np.abs(X))

    def transform(self, df):
        X = self._raw(df)
        return ((self._log(np.where(np.isnan(X), self.mean, X)) - self.mu) / self.sd).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Linear(32, k)

    def forward(self, x):
        h = self.body(x)
        return self.head(h), h


def train_mlp(Xtr, ytr, Xva, yva, k, soft=None, T=4.0, seed=0, epochs=30):
    """CE on hard labels, or KD (KL to teacher soft labels at temperature T) when `soft` is given.
    Early stop on validation CE."""
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1], k)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    St = None if soft is None else torch.from_numpy(soft.astype(np.float32))
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    best, state, g = np.inf, None, torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        for b in torch.randperm(len(yt), generator=g).split(512):
            z, _ = m(Xt[b])
            if St is None:
                loss = nn.functional.cross_entropy(z, yt[b])
            else:  # teacher probabilities sharpened/softened at T via log-space
                t = torch.softmax(torch.log(St[b] + 1e-8) / T, 1)
                loss = nn.functional.kl_div(torch.log_softmax(z / T, 1), t, reduction="batchmean") * T * T
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.cross_entropy(m(Xv)[0], yv).item()
        if v < best:
            best, state = v, {k_: t_.clone() for k_, t_ in m.state_dict().items()}
    m.load_state_dict(state)
    return m


def mlp_scores(m, Xtr, ytr, X, benign, k, rng):
    """Higher = more likely unknown/attack."""
    with torch.no_grad():
        ztr, htr = m(torch.from_numpy(Xtr))
        z, h = m(torch.from_numpy(X))
    p = torch.softmax(z, 1).numpy()
    htr, h = htr.numpy().astype(np.float64), h.numpy().astype(np.float64)
    mu = np.stack([htr[ytr == c].mean(0) for c in range(k)])
    cov = np.cov((htr - mu[ytr]).T) + 1e-6 * np.eye(h.shape[1])
    P = np.linalg.inv(cov)
    d = np.stack([np.einsum("ij,jk,ik->i", h - mu[c], P, h - mu[c]) for c in range(k)], 1)
    nrm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    ref = htr[rng.choice(len(htr), min(20000, len(htr)), replace=False)]
    knn = NearestNeighbors(n_neighbors=10).fit(nrm(ref)).kneighbors(nrm(h))[0][:, -1]
    return {"msp": 1 - p.max(1), "energy": -torch.logsumexp(z, 1).numpy(), "not_benign": 1 - p[:, benign],
            "mahalanobis": d.min(1), "knn": knn}, p.argmax(1)


def tree_scores(p, benign):
    s = np.sort(p, 1)
    return {"msp": 1 - s[:, -1], "margin": -(s[:, -1] - s[:, -2]), "not_benign": 1 - p[:, benign]}


def evaluate(scores, y_test_unknown):
    """AUROC (held-out family vs benign test rows) and detection rate at the 1% benign-val alarm threshold."""
    out = {}
    for name, (s_te, s_va) in scores.items():
        thr = np.quantile(s_va, 1 - FPR)
        out[name] = {"auroc": float(roc_auc_score(y_test_unknown, s_te)),
                     "tpr_at_fpr1": float((s_te[y_test_unknown == 1] > thr).mean()),
                     "test_benign_fpr": float((s_te[y_test_unknown == 0] > thr).mean())}
    return out


def cap_rows(idx, fam, cap, rng):
    keep = [rng.choice(idx[fam[idx] == f], min(cap, (fam[idx] == f).sum()), replace=False) for f in np.unique(fam[idx])]
    return np.sort(np.concatenate(keep))


def run(ds, seed, cap):
    rng = np.random.default_rng(seed)
    df, _ = clean.clean(load.LOADERS[ds](), ds)
    tr, va, te = split.grouped_split(df, seed)
    fam = df["family"].astype(str).to_numpy()
    tr, va, te = (cap_rows(i, fam, c, rng) for i, c in ((tr, cap), (va, cap // 4), (te, cap // 2)))
    feats = clean.feature_columns(df)
    families = [f for f in np.unique(fam) if f != "Benign"]
    res = {"dataset": ds, "seed": seed, "cap": cap, "rows": {"train": len(tr), "val": len(va), "test": len(te)},
           "held_out": {}}
    for held in families:
        t0 = time.time()
        trh, vah, _ = split.lofo(df, (tr, va, te), held)
        assert not (fam[trh] == held).any() and not (fam[vah] == held).any()
        known = sorted(set(fam[trh]))
        code = {f: i for i, f in enumerate(known)}
        b, k = code["Benign"], len(known)
        prep = Prep().fit(df.iloc[trh][feats])
        Xtr, Xva = prep.transform(df.iloc[trh][feats]), prep.transform(df.iloc[vah][feats])
        ytr, yva = np.array([code[f] for f in fam[trh]]), np.array([code[f] for f in fam[vah]])
        ev = te[np.isin(fam[te], ["Benign", held])]  # unknown-detection set
        kn = te[np.isin(fam[te], known)]              # known-class set
        Xev, Xkn = prep.transform(df.iloc[ev][feats]), prep.transform(df.iloc[kn][feats])
        yev = (fam[ev] == held).astype(int)
        ykn = np.array([code[f] for f in fam[kn]])
        vb = yva == b
        scores, f1 = {}, {}

        rf = RandomForestClassifier(300, n_jobs=-1, random_state=seed, min_samples_leaf=2).fit(Xtr, ytr)
        xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=-1,
                            random_state=seed).fit(Xtr, ytr)
        for name, m in (("rf", rf), ("xgb", xgb)):
            pe, pv = m.predict_proba(Xev), m.predict_proba(Xva[vb])
            f1[name] = float(f1_score(ykn, m.predict(Xkn), average="macro"))
            for s, (a, c) in {s: (v, tree_scores(pv, b)[s]) for s, v in tree_scores(pe, b).items()}.items():
                scores[f"{name}_{s}"] = (a, c)
        iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr[ytr == b])
        scores["iforest_benign"] = (-iso.score_samples(Xev), -iso.score_samples(Xva[vb]))

        soft = cross_val_predict(XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                                               n_jobs=-1, random_state=seed), Xtr, ytr, cv=3, method="predict_proba")
        for arm, kw in (("mlp_ce", {}), ("mlp_kd", {"soft": soft, "T": 4.0})):
            m = train_mlp(Xtr, ytr, Xva, yva, k, seed=seed, **kw)
            se, _ = mlp_scores(m, Xtr, ytr, Xev, b, k, rng)
            sv, _ = mlp_scores(m, Xtr, ytr, Xva[vb], b, k, rng)
            _, pk = mlp_scores(m, Xtr, ytr, Xkn, b, k, rng)
            f1[arm] = float(f1_score(ykn, pk, average="macro"))
            for s in se:
                scores[f"{arm}_{s}"] = (se[s], sv[s])
        r = evaluate(scores, yev)
        res["held_out"][held] = {"n_unknown_test": int(yev.sum()), "n_benign_test": int((yev == 0).sum()),
                                 "known_macro_f1": f1, "scores": r, "seconds": time.time() - t0}
        best = sorted(r.items(), key=lambda kv: -kv[1]["auroc"])[:4]
        print(f"{held:18s} n={yev.sum():6d}  " + "  ".join(f"{n} {v['auroc']:.3f}" for n, v in best), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(load.LOADERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=40000)
    a = ap.parse_args()
    res = run(a.dataset, a.seed, a.cap)
    out = ROOT / "results" / "pilot" / f"lofo_{a.dataset}_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("->", out)
