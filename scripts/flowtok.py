"""FlowTok (PREREG §9 TOK): per-flow tokenizer -> per-host variable-order sequence coder -> conformal P / A layer.

Tokenizer: light XGB (50 rounds, depth 4) on per-flow features only (no hc_*); token = (argmax known class,
confidence bucket, leaf-rarity bit). Coder: per known family, Witten-Bell interpolated Markov model of orders 0-3 over
each source host's token stream in time order (population = kept rows minus the role's removal set, as Bank.frame).
Score = -log2 P_benign(token | previous <= 3 tokens of the host). Arms: ft_tree, ft_order0, ft_mlp, ft_hc.
Writes runs/bank/{ds}_host_cap40000_s{seed}_flowtok/config=outer_{held}/part.parquet (rows identical to the base bank).
Usage: .venv/bin/python scripts/flowtok.py nf_ctu13 --seed 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import score_bank as sb  # noqa: E402
from nids import hostctx, models  # noqa: E402

ORDER, KNOWN_WIN, CONF = 3, 32, (0.6, 0.9)
CHUNK = 200_000


def contexts(tok, src, t, pop, V):
    """(n, ORDER) packed contexts (orders 1..ORDER) for rows in pop: previous tokens of the same src within pop, in time
    order; missing history = V. Rows outside pop get -1. Also returns pop rows in stream order."""
    B = V + 1
    idx = np.flatnonzero(pop)
    o = idx[np.lexsort((idx, t[idx], src[idx]))]
    s, tk = src[o], tok[o].astype(np.int64)
    C = np.full((len(tok), ORDER), -1, np.int64)
    ctx = np.zeros(len(o), np.int64)
    for d in range(1, ORDER + 1):
        same = np.r_[np.zeros(min(d, len(o)), bool), s[d:] == s[:-d]]
        prev = np.where(same, np.r_[np.full(min(d, len(o)), V), tk[:-d]], V)
        ctx = ctx + B ** (d - 1) * prev
        C[o, d - 1] = ctx
    return C, o


class Coder:
    """Witten-Bell interpolated variable-order Markov model over tokens 0..V-1 (orders 0..ORDER)."""

    def __init__(self, V, order=ORDER):
        self.V, self.B, self.order = V, V + 1, order

    def _ctx(self, C, d):
        return np.zeros(len(C), np.int64) if d == 0 else C[:, d - 1]

    def fit(self, C, tok):
        self.tab = []
        for d in range(self.order + 1):
            c = self._ctx(C, d)
            kct, nct = np.unique(c * self.B + tok, return_counts=True)
            kc, nc = np.unique(c, return_counts=True)
            uc = np.bincount(np.searchsorted(kc, kct // self.B), minlength=len(kc))  # distinct tokens per context
            self.tab.append((kct, nct, kc, nc, uc))
        return self

    @staticmethod
    def _get(keys, vals, q):
        i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
        return np.where(keys[i] == q, vals[i], 0)

    def logp(self, C, tok):
        p = np.full(len(tok), 1.0 / self.V)
        for d in range(self.order + 1):
            kct, nct, kc, nc, uc = self.tab[d]
            c = self._ctx(C, d)
            n_ct, n_c, u_c = self._get(kct, nct, c * self.B + tok), self._get(kc, nc, c), self._get(kc, uc, c)
            p = np.where(n_c > 0, (n_ct + u_c * p) / np.maximum(n_c + u_c, 1), p)
        return np.log2(p)


def bucket(p):
    return np.digitize(p, CONF)


def tree_tokens(xg, X, rarity_ref=None):
    """Tokens (class * 6 + conf * 2 + rare) and rarity for X; rarity_ref = (leaf shares, threshold) from the fit rows."""
    proba, rar = [], []
    for i in range(0, len(X), CHUNK):
        x = X[i:i + CHUNK]
        proba.append(xg.predict_proba(x))
        if rarity_ref is not None:
            L = xg.apply(x).astype(np.int64)
            rar.append(-np.log(rarity_ref[0][L + 64 * np.arange(L.shape[1])]).mean(1))
    p = np.vstack(proba)
    c = p.argmax(1)
    rare = np.zeros(len(X), np.int64) if rarity_ref is None else (np.concatenate(rar) > rarity_ref[1]).astype(np.int64)
    return c * 6 + bucket(p.max(1)) * 2 + rare


def leaf_shares(xg, X):
    L = xg.apply(X).astype(np.int64)
    assert L.max() < 64, "depth-4 trees have < 64 node ids"
    cnt = np.bincount((L + 64 * np.arange(L.shape[1])).ravel(), minlength=64 * L.shape[1])
    share = (cnt + 1) / (len(X) + 1)
    rar = -np.log(share[L + 64 * np.arange(L.shape[1])]).mean(1)
    return share, float(np.quantile(rar, 0.99))


def light_xgb(seed):
    return XGBClassifier(n_estimators=50, max_depth=4, tree_method="hist", n_jobs=4, random_state=seed)


def run_config(bank, held, meta_src, meta_t, base_part):
    P = (held,)
    fit, es = bank.fit_rows(P)
    known = sorted(set(bank.fam[fit]))
    code = {f: i for i, f in enumerate(known)}
    k = len(known)
    yf = np.array([code[f] for f in bank.fam[fit]])
    es = es[np.isin(bank.fam[es], known)]
    ye = np.array([code[f] for f in bank.fam[es]])
    n = len(bank.fam)
    pop_P, pop_all = ~bank.removed(np.arange(n), P), np.ones(n, bool)

    # per-flow features of every kept row (population independent)
    flow = bank.df[bank.feats]
    prep = models.Prep().fit(flow.iloc[fit])
    Xall = prep.transform(flow)
    xg = light_xgb(bank.seed).fit(Xall[fit], yf)
    ref = leaf_shares(xg, Xall[fit])
    tok = {"tree": tree_tokens(xg, Xall, ref)}

    mlp = models.int8(models.train_mlp(Xall[fit], yf, Xall[es], ye, k, seed=bank.seed))
    pm = models.mlp_proba(mlp, Xall)
    tok["mlp"] = pm.argmax(1) * 6 + bucket(pm.max(1)) * 2

    # ft_hc: tokenizer with host context; tokens depend on the role's population
    hcP, hcA = bank.host.removal(P), bank.host.removal(())
    prep_h = models.Prep().fit(pd.concat([flow.iloc[fit].reset_index(drop=True),
                                          pd.DataFrame(hcP[fit], columns=hostctx.COLS)], axis=1))
    XhP = prep_h.transform(pd.concat([flow.reset_index(drop=True), pd.DataFrame(hcP, columns=hostctx.COLS)], axis=1))
    xh = light_xgb(bank.seed).fit(XhP[fit], yf)
    tok_hcP = tree_tokens(xh, XhP)
    del XhP
    XhA = prep_h.transform(pd.concat([flow.reset_index(drop=True), pd.DataFrame(hcA, columns=hostctx.COLS)], axis=1))
    tok_hcA = tree_tokens(xh, XhA)
    del XhA

    V = 6 * k
    roles = {"rank": bank.rank, "cal": bank.cal, "test_benign": bank.test_benign,
             "test_unknown": np.sort(np.concatenate([r[bank.removed(r, P)] for r in bank.test.values()])),
             "test_known": np.sort(np.concatenate([r[~bank.removed(r, P)] for r in bank.test.values()]))}
    frames = {role: {"row_id": rows.astype(np.int64), "role": role, "family": bank.fam[rows]}
              for role, rows in roles.items()}
    for name, (tP, tA) in {"tree": (tok["tree"], tok["tree"]), "mlp": (tok["mlp"], tok["mlp"]),
                           "hc": (tok_hcP, tok_hcA)}.items():
        CP, _ = contexts(tP, meta_src, meta_t, pop_P, V)
        CA, oA = contexts(tA, meta_src, meta_t, pop_all, V)
        coders = {f: Coder(V).fit(CP[fit[yf == i]], tP[fit[yf == i]]) for f, i in code.items()}
        ben = coders["Benign"]
        order0 = Coder(V, order=0).fit(CP[fit[yf == code["Benign"]]], tP[fit[yf == code["Benign"]]])
        for role, rows in roles.items():
            C, tk = (CP, tP) if role in ("rank", "cal") else (CA, tA)
            frames[role][f"ft_{name}"] = -ben.logp(C[rows], tk[rows])
            if name == "tree":
                frames[role]["ft_order0"] = -order0.logp(C[rows], tk[rows])
        if name == "tree":  # known label: family with the shortest code over the host's last KNOWN_WIN tokens
            s_o, tk_o = meta_src[oA], tA[oA]
            L = np.column_stack([coders[f].logp(CA[oA], tk_o) for f in known])
            cs = np.vstack([np.zeros(k), np.cumsum(L, 0)])
            g = np.r_[0, np.flatnonzero(s_o[1:] != s_o[:-1]) + 1]
            start = np.maximum(np.arange(len(oA)) - KNOWN_WIN + 1, np.repeat(g, np.diff(np.r_[g, len(oA)])))
            win = cs[np.arange(len(oA)) + 1] - cs[start]
            pred = np.empty(n, object)
            pred[oA] = np.array(known)[win.argmax(1)]
            for role, rows in roles.items():
                frames[role]["ft_tree_pred"] = pred[rows].astype(str)
    df = pd.concat([pd.DataFrame(f) for f in frames.values()], ignore_index=True)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    b = base_part.groupby("role").row_id.apply(np.sort)
    for role, rows in roles.items():
        assert np.array_equal(np.sort(rows), b.get(role, np.empty(0, np.int64))), f"{held} {role}: rows differ from the base bank"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    base = sb.OUT / f"{a.dataset}_host_cap40000_s{a.seed}"
    out = sb.OUT / f"{a.dataset}_host_cap40000_s{a.seed}_flowtok"
    bank = sb.Bank(a.dataset, a.seed, "host", 40000, 200000)
    z = np.load(ROOT / "data" / "processed" / f"{a.dataset}_hostmeta.npz", allow_pickle=False)
    src, t = z["src"].astype(np.int64), z["t"].astype(np.int64)
    assert len(src) == len(bank.fam)
    for p in sorted(base.glob("config=outer_*/part.parquet")):
        held = p.parent.name.removeprefix("config=outer_")
        df = run_config(bank, held, src, t, sb.read_part(p, ["row_id", "role"]))
        dest = out / f"config=outer_{held}"
        dest.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest / "part.parquet", index=False)
        print(f"{held}: {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
    (out / "manifest_outer.json").write_text(json.dumps({"dataset": a.dataset, "seed": a.seed,
                                                         "seconds": time.time() - t0}))


if __name__ == "__main__":
    main()
