"""K15 (PREREG §9): attack-stopped rate of each uncertified detector acting as an IPS, in the K1 / K14 seed-1 setup.

Same fitted detectors as `spoof_attack.py --baselines`; tau = clean 1%-FPR quantile on calibration-host benign flows;
macro TPR over the held-out family's test rows (unknown) and the other families' (known), 5,000 rows per family (RNG 0).
Writes results/cert/ips_tpr_{ds}.csv. Usage: .venv/bin/python scripts/ips_point.py nf_ctu13
"""
import sys

import numpy as np
import pandas as pd

import host_stream as hs
import spoof_attack as sa


def main():
    ds = sys.argv[1]
    _, bank = sa.setup(ds)
    held = bank.families[0]
    dets, prep, _ = sa.detectors(bank, held, None, True)
    z = np.load(sa.ROOT / "data" / "processed" / f"{bank.ds}_hostmeta.npz", allow_pickle=False)
    cal_host = hs.host_split(z["ips"], sa.SEED)
    benign_host = ~np.isin(np.arange(len(cal_host)), np.unique(z["src"][z["fam"] != "Benign"]))
    cal = bank.cal[cal_host[z["src"][bank.cal]] & benign_host[z["src"][bank.cal]]]
    Xc = prep.transform(bank.frame(cal, (held,)))
    rng = np.random.default_rng(0)
    cap = lambda r: np.sort(rng.permutation(r)[:5000])  # noqa: E731
    T = {"unk": cap(bank.test[held]),
         "known": np.concatenate([cap(r) for f, r in bank.test.items() if f not in (held, "Benign")])}
    XT = {k: prep.transform(bank.frame(r, ())) for k, r in T.items()}
    res = []
    for n, f in dets.items():
        tau = float(np.quantile(f(Xc), 0.99))
        r = {"detector": n, "tau": tau, **{f"tpr_{k}": float((f(X) > tau).mean()) for k, X in XT.items()}}
        r["tpr_macro"] = (r["tpr_unk"] + r["tpr_known"]) / 2
        assert 0 <= r["tpr_macro"] <= 1
        res.append(r)
        print(r, flush=True)
    pd.DataFrame(res).to_csv(sa.OUT / f"ips_tpr_{ds}.csv", index=False)


if __name__ == "__main__":
    main()
