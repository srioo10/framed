"""PREREG §9 AUDIT F2 / F3 on saved full-arena scores: paired bootstrap 95% intervals over the 18 pairs (10,000
resamples, seeds kept together) for fitness and ours - boss; AUROC (attack vs benign cal) and unknown / known TPR
at 1% FPR. A = round-3 champion, B = round-5 champion (plain alarm scores). Writes results/bowl/audit_f2f3.json.
Usage: .venv/bin/python scripts/audit_f2f3.py
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import arena
import bowl

SRC = {"A": "round3", "B": "round5"}
BOSSES = ("rf", "xgb", "cat", "lgbm", "xgbx", "gang")


def main():
    S = {k: np.load(bowl.OUT / v / "champion_scores.npz") for k, v in SRC.items()}
    per = {d: [] for d in (*SRC, *BOSSES)}
    auc = {d: [] for d in per}
    for z in bowl.load_pairs(bowl.ARENA, cheap=False):
        key, att = f"{z['ds']}/{z['name']}", z["y_es"] != z["bi"]
        for d in per:
            runs = [{q: S[d][f"{key}/s{sd}/{q}"] for q in ("cal", "pseudo", "es")} for sd in (1, 2)] if d in SRC else \
                   [{q: z[f"boss_{d}_{q}"] for q in ("cal", "pseudo", "es")}]
            per[d].append(np.mean([arena.fitness(r["cal"], r["pseudo"], r["es"], z["y_es"], z["bi"]) for r in runs], 0))
            auc[d].append(np.mean([roc_auc_score(np.r_[np.zeros(len(r["cal"])), np.ones(len(r["pseudo"]) + att.sum())],
                                                 np.r_[r["cal"], r["pseudo"], r["es"][att]]) for r in runs]))
    P = {d: np.array(v) for d, v in per.items()}  # (18, 3): fitness, unk, known
    idx = np.random.default_rng(0).integers(0, 18, (10_000, 18))
    ci = lambda v: [float(v.mean()), *map(float, np.quantile(v[idx].mean(1), [0.025, 0.975]))]  # noqa: E731
    out = {d: {"fitness": ci(P[d][:, 0]), "unk": ci(P[d][:, 1]), "known": ci(P[d][:, 2]), "auroc": ci(np.array(auc[d]))}
           for d in P}
    out["diff"] = {f"{o}-{b}": ci(P[o][:, 0] - P[b][:, 0]) for o in SRC for b in BOSSES}
    (bowl.OUT / "audit_f2f3.json").write_text(json.dumps(out, indent=1))
    for d in P:
        print(d, {k: [round(x, 3) for x in v] for k, v in out[d].items()})
    for k, v in out["diff"].items():
        print(k, [round(x, 3) for x in v], "significant" if v[1] > 0 or v[2] < 0 else "not significant")


if __name__ == "__main__":
    main()
