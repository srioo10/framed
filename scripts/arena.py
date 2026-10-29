"""The bowl's arena (PREREG §9 BOWL): training-side nested pairs only, never a test row.

Pair (R0, j): fit on bank.fit_rows(P), P = R0 + {j}; j's capped train + validation rows from single-family ("fresh")
attacker hosts are pseudo-unknown (host context with only R0 removed, as score_bank pseudo rows); es attack rows are
known-attack validation; cal rows of benign calibration hosts set the 1% threshold. Per pair, saves prepped arrays,
3-fold OOF XGB teacher probabilities on the fit rows, and the heavy RF / XGB boss scores.
Usage: .venv/bin/python scripts/arena.py nf_ctu13            (writes data/arena/{ds}/pair_*.npz)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from roc import roc_detect  # noqa: E402

OUT = ROOT / "data" / "arena"
N_PAIRS, MIN_ROWS, SEED = 6, 100, 1


def fresh_hosts(ds):
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    att = z["fam"] != "Benign"
    n = pd.Series(z["fam"][att]).groupby(z["src"][att]).nunique()
    fresh = np.zeros(len(z["ips"]), bool)
    fresh[n.index[n == 1]] = True
    return z, fresh


def pseudo_rows(bank, P, R0):
    rows = np.sort(np.concatenate([bank.cap[f] for f in bank.families] + [bank.val_attack[f] for f in bank.families]))
    return rows[bank.removed(rows, P) & ~bank.removed(rows, R0)]


def pairs(bank, src, fresh):
    """Up to N_PAIRS (R0, j), round-robin over eligible j (>= MIN_ROWS fresh pseudo rows), R0 drawn with seed 0."""
    rng = np.random.default_rng(0)
    cand = {}
    for j in bank.families:
        for r in bank.families:
            R0 = (r,)
            if r == j or bank.removed(bank.cap[j], R0).any():
                continue
            P = tuple(sorted({r, j}))
            if fresh[src[pseudo_rows(bank, P, R0)]].sum() >= MIN_ROWS:
                cand.setdefault(j, []).append(R0)
    out, js = [], sorted(cand)
    for k in range(N_PAIRS):
        if not js:
            break
        j = js[k % len(js)]
        left = [r for r in cand[j] if (r, j) not in out]
        if left:
            out.append((left[rng.integers(len(left))], j))
    return out


def build(ds):
    import host_stream as hs
    import score_bank as sb
    from nids import models
    z, fresh = fresh_hosts(ds)
    src = z["src"]
    cal_host = hs.host_split(z["ips"], SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(src[z["fam"] != "Benign"]))
    bank = sb.Bank(ds, SEED, "host", 40000, 200000)
    test = np.concatenate([bank.test_benign] + list(bank.test.values()))
    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    plist = pairs(bank, src, fresh)
    for i, (R0, j) in enumerate(plist):
        t0 = time.time()
        P = tuple(sorted(set(R0) | {j}))
        fit, es = bank.fit_rows(P)
        known = sorted(set(bank.fam[fit]))
        code = {f: c for c, f in enumerate(known)}
        es = es[np.isin(bank.fam[es], known)]
        ps = pseudo_rows(bank, P, R0)
        ps = ps[fresh[src[ps]]]
        cal = bank.cal[cal_host[src[bank.cal]] & benign_host[src[bank.cal]]]
        rows = {"fit": fit, "es": es, "pseudo": ps, "cal": cal}
        for r in rows.values():
            assert not np.isin(r, test).any(), f"{ds} pair {i}: arena row in a test role"
        dfit = bank.frame(fit, sb.role_removal(P, "fit"))
        prep = models.Prep().fit(dfit)
        X = {"fit": prep.transform(dfit), "es": prep.transform(bank.frame(es, sb.role_removal(P, "es"))),
             "pseudo": prep.transform(bank.frame(ps, sb.role_removal(P, "pseudo", R0))),
             "cal": prep.transform(bank.frame(cal, sb.role_removal(P, "cal")))}
        y = {k: np.array([code[f] for f in bank.fam[rows[k]]], np.int64) for k in ("fit", "es")}
        teacher = models.oof_proba(X["fit"], y["fit"], SEED).astype(np.float32)
        bi = code["Benign"]
        boss = {}
        for name, m in (("rf", models.rf(SEED)), ("xgb", models.xgb(SEED))):
            m.fit(X["fit"], y["fit"])
            for k in ("es", "pseudo", "cal"):
                boss[f"boss_{name}_{k}"] = -np.log(np.maximum(m.predict_proba(X[k])[:, bi], 1e-45)).astype(np.float32)
            if name == "xgb" and i == 0:
                m.save_model(out / "boss_xgb.json")  # timing reference for the cost limit
        np.savez_compressed(out / f"pair_{i}.npz", **{f"X_{k}": v for k, v in X.items()},
                            **{f"y_{k}": v for k, v in y.items()}, teacher=teacher, bi=bi, **boss,
                            **{f"rows_{k}": v for k, v in rows.items()})
        (out / f"pair_{i}.json").write_text(json.dumps({"R0": list(R0), "j": j, "known": known, "feats": bank.feats,
                                                        "n": {k: int(len(v)) for k, v in rows.items()}}))
        print(f"{ds} pair {i} R0={R0} j={j} fit={len(fit)} pseudo={len(ps)} cal={len(cal)} {time.time() - t0:.0f}s",
              flush=True)


def fitness(s_cal, s_pseudo, s_es, y_es, bi):
    """Mean of unknown TPR (pseudo) and known-attack TPR (es attacks) at exactly 1% FPR on benign cal rows."""
    s = np.r_[s_cal, s_pseudo, s_es]
    w = roc_detect(s, np.r_[np.ones(len(s_cal), bool), np.zeros(len(s) - len(s_cal), bool)])
    unk = w[len(s_cal):len(s_cal) + len(s_pseudo)].mean()
    att = y_es != bi
    known = w[len(s_cal) + len(s_pseudo):][att].mean() if att.any() else unk
    return 0.5 * (unk + known), unk, known


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    build(ap.parse_args().dataset)
