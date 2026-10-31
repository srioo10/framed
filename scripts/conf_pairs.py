"""CONF test pairs (PREREG §9 CONF): the arena's pair format built on test rows, for the single-shot confirmation.

Bank seed 1 only: grouped_split picks the test fold by seed % 5, so any other bank seed's test fold holds rows the
bowl developed on; the seed-1 test fold is the one the arena never touched (arena.py asserts it). Training seeds 2-4
vary model randomness at evaluation time. Per dataset: held-out family j (up to N_PAIRS families with >= MIN_ROWS
test rows, order drawn with seed 0). Fit on bank.fit_rows((j,)) (j removed); cal = benign rows of calibration hosts
(host_split seed 1, as the arena); pseudo = j's test rows (true unknown); es = the other families' test rows plus test benign rows
(y_es; benign es rows give the realised test FPR). Test roles see the full host-context population (role_removal
test_* -> nothing). Same arrays and teacher as arena.py; boss RF / XGB scores for each training seed (boss_rfs2_cal,
...); Prep mu / sd / cols in pair_i.json, so bowl.load_pairs(data/conf) reads them unchanged.
Usage: .venv/bin/python scripts/conf_pairs.py nf_ctu13
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data" / "conf"
N_PAIRS, MIN_ROWS, SEED, TRAIN_SEEDS = 6, 100, 1, (2, 3, 4)


def build(ds):
    import host_stream as hs
    import score_bank as sb
    from nids import models
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    src = z["src"]
    cal_host = hs.host_split(z["ips"], SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(src[z["fam"] != "Benign"]))
    bank = sb.Bank(ds, SEED, "host", 40000, 200000)
    js = [f for f in bank.families if len(bank.test[f]) >= MIN_ROWS]
    js = [js[i] for i in np.random.default_rng(0).permutation(len(js))][:N_PAIRS]
    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    cal = bank.cal[cal_host[src[bank.cal]] & benign_host[src[bank.cal]]]
    for i, j in enumerate(js):
        t0 = time.time()
        P = (j,)
        fit, _ = bank.fit_rows(P)
        known = sorted(set(bank.fam[fit]))
        code = {f: c for c, f in enumerate(known)}
        es = np.sort(np.concatenate([bank.test_benign] + [bank.test[f] for f in known if f != "Benign"]))
        rows = {"fit": fit, "es": es, "pseudo": bank.test[j], "cal": cal}
        assert not np.isin(fit, np.concatenate([es, rows["pseudo"]])).any(), f"{ds} {j}: fit row in test"
        assert not np.isin(cal, es).any(), f"{ds} {j}: cal row in test"
        dfit = bank.frame(fit, sb.role_removal(P, "fit"))
        prep = models.Prep().fit(dfit)
        X = {"fit": prep.transform(dfit), "es": prep.transform(bank.frame(es, sb.role_removal(P, "test_known"))),
             "pseudo": prep.transform(bank.frame(rows["pseudo"], sb.role_removal(P, "test_unknown"))),
             "cal": prep.transform(bank.frame(cal, sb.role_removal(P, "cal")))}
        y = {k: np.array([code[f] for f in bank.fam[rows[k]]], np.int64) for k in ("fit", "es")}
        teacher = models.oof_proba(X["fit"], y["fit"], SEED).astype(np.float32)
        bi = code["Benign"]
        boss = {}
        for name, s in ((n, s) for s in TRAIN_SEEDS for n in ("rf", "xgb")):
            m = getattr(models, name)(s).fit(X["fit"], y["fit"])
            for k in ("es", "pseudo", "cal"):
                boss[f"boss_{name}s{s}_{k}"] = -np.log(np.maximum(m.predict_proba(X[k])[:, bi], 1e-45)).astype(np.float32)
            if name == "xgb" and s == TRAIN_SEEDS[0] and i == 0:
                m.save_model(out / "boss_xgb.json")  # timing reference, as arena.py
        np.savez_compressed(out / f"pair_{i}.npz", **{f"X_{k}": v for k, v in X.items()},
                            **{f"y_{k}": v for k, v in y.items()}, teacher=teacher, bi=bi, **boss,
                            **{f"rows_{k}": v for k, v in rows.items()})
        (out / f"pair_{i}.json").write_text(json.dumps({"R0": [], "j": j, "known": known, "feats": bank.feats,
                                                        "n": {k: int(len(v)) for k, v in rows.items()},
                                                        "cols": list(dfit.columns), "mu": prep.mu.tolist(),
                                                        "sd": prep.sd.tolist()}))
        print(f"{ds} pair {i} j={j} fit={len(fit)} unk={len(rows['pseudo'])} es={len(es)} cal={len(cal)} "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    build(ap.parse_args().dataset)
