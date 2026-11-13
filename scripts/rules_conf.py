"""PREREG RULES-CONF: frame-proof shifted count rules on the CONF pairs, scored with the CONF certified readout.
Usage: .venv/bin/python scripts/rules_conf.py  ->  results/conf/rules_conf.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sensitivity  # noqa: E402

M, DS = 10, ("nf_ctu13", "nf_ugr16", "nf_cic")
RULES = {"rate": lambda g: g("hc_src_n_10s"),
         "f2b": lambda g: g("hc_src_n_120s") * g("hc_src_fail_120s"),
         "scan": lambda g: g("hc_src_n_120s") * (g("hc_src_newport_120s") + g("hc_src_newpeer_120s"))}


def raw(X, cols, mu, sd):
    return lambda c: np.expm1(np.maximum(X[:, cols.index(c)] * sd[cols.index(c)] + mu[cols.index(c)], 0))


def fitness(z, meta, rule):
    cols, mu, sd = meta["cols"], np.array(meta["mu"]), np.array(meta["sd"])
    s_cal = RULES[rule](raw(z["X_cal"], cols, mu, sd))
    tau = np.sort(s_cal)[::-1][int(np.floor(0.01 * len(s_cal)))]
    def hit(X):
        lo, _ = sensitivity.deletion_box(X, cols, mu, sd, M)
        return RULES[rule](raw(lo, cols, mu, sd)) > tau
    unk = hit(z["X_pseudo"]).mean()
    a = z["y_es"] != int(z["bi"])
    known = hit(z["X_es"][a]).mean() if a.any() else unk
    return float((unk + known) / 2)


def main():
    out, rng = {}, np.random.default_rng(0)
    for ds in DS:
        veto = {}
        for l in open(ROOT / "results" / "conf" / f"clean_{ds}.jsonl"):
            r = json.loads(l)
            if r["model"] == "B":
                veto.setdefault(r["pair"].split("/")[1], []).append(r["robust"]["10"][0])
        pairs = sorted(veto, key=lambda p: int(p.split("_")[1]))
        R = {k: [] for k in RULES}
        for p in pairs:
            z, meta = np.load(ROOT / "data" / "conf" / ds / f"{p}.npz"), json.loads((ROOT / "data" / "conf" / ds / f"{p}.json").read_text())
            for k in RULES:
                R[k].append(fitness(z, meta, k))
        v = np.array([np.mean(veto[p]) for p in pairs])
        best = max(RULES, key=lambda k: np.mean(R[k]))
        d = v - np.array(R[best])
        bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(10_000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[ds] = {"veto": float(v.mean()), "rules": {k: float(np.mean(x)) for k, x in R.items()}, "best": best,
                   "lead": float(d.mean()), "ci": [float(lo), float(hi)], "pass": bool(lo > 0), "n_pairs": len(pairs)}
        print(ds, json.dumps(out[ds]), flush=True)
    out["PASS"] = sum(out[d]["pass"] for d in DS) >= 2
    (ROOT / "results" / "conf" / "rules_conf.json").write_text(json.dumps(out, indent=1))
    print("PASS" if out["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
