"""C7b (PREREG 2026-10-26): taxonomy back-off on CICIoT2023 with DDU class likelihoods vs softmax mass.

Per seed and outer config (incl. the DoS / DDoS / Flood+Mirai hold-outs): fit the default MLP and the DDU model on the
bank's fit rows, flag test_unknown flows above each method's own 1% cal threshold, and give each flagged flow a parent.
Usage: .venv/bin/python scripts/backoff_ciciot.py --seeds 0 1 2 3 4 --out results/onepass/backoff_ciciot
"""
import argparse
import json
import sys
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

PARENT = {"Flood": "Volumetric", "DoS": "Volumetric", "DDoS": "Volumetric", "Mirai": "Volumetric",
          "BruteForce": "Access", "Web": "Access", "Recon": "Recon", "Spoofing": "Spoofing"}


def parents(known_attacks, scores):
    """Parent per row: argmax over parents of the logsumexp (log-likelihoods) or sum (posteriors) of its children."""
    groups = sorted({PARENT[f] for f in known_attacks})
    return groups, np.array(groups)[np.column_stack([scores(groups, g) for g in groups]).argmax(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(10)
    a.out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        bank = sb.Bank("ciciot_csv", seed, "none", 40000, 200000)
        for _, P, _ in bank.configs("outer", True):
            f = a.out / f"s{seed}_{sb.SEP.join(P)}.json"
            if f.exists():
                continue
            fit, es = bank.fit_rows(P)
            known = sorted(set(bank.fam[fit]))
            code = {c: i for i, c in enumerate(known)}
            dfit = bank.frame(fit, sb.role_removal(P, "fit"))
            prep = models.Prep().fit(dfit)
            Xf, yf = prep.transform(dfit), np.array([code[c] for c in bank.fam[fit]])
            Xe = prep.transform(bank.frame(es, sb.role_removal(P, "es")))
            ye = np.array([code[c] for c in bank.fam[es]])
            mlp = models.train_mlp(Xf, yf, Xe, ye, len(known), seed=seed).eval()
            snm = models.train_mlp(Xf, yf, Xe, ye, len(known), seed=seed, arch=models.SNMLP).eval()
            gda = models.FeatureGDA().fit(snm, Xf, yf, len(known))
            Xc = prep.transform(bank.frame(bank.cal, sb.role_removal(P, "cal")))
            unk = np.sort(np.concatenate([r[bank.removed(r, P)] for r in bank.test.values()]))
            Xu = prep.transform(bank.frame(unk, sb.role_removal(P, "test_unknown")))
            att = [c for c in known if c != "Benign"]
            idx = [code[c] for c in att]
            truth = PARENT[P[0]] if all(PARENT[x] == PARENT[P[0]] for x in P) else None
            res = {"seed": seed, "held": list(P), "truth": truth, "n_unknown": int(len(unk)),
                   "leaf": bool(truth and any(PARENT[c] == truth for c in att))}
            bi = code["Benign"]
            pu = models.mlp_proba(mlp, Xu)
            s_mlp, th_mlp = 1 - pu[:, bi], fc.threshold(1 - models.mlp_proba(mlp, Xc)[:, bi], 0.01)
            ll_u = gda.class_loglik(snm, Xu)
            s_ddu, th_ddu = -torch.logsumexp(ll_u, 1).numpy(), fc.threshold(gda.score(snm, Xc), 0.01)
            for name, flag, sc in (
                    ("softmax", s_mlp > th_mlp,
                     lambda gs, g, p=pu: p[:, [code[c] for c in att if PARENT[c] == g]].sum(1)),
                    ("ddu", s_ddu > th_ddu,
                     lambda gs, g, ll=ll_u: torch.logsumexp(ll[:, [code[c] for c in att if PARENT[c] == g]], 1).numpy())):
                groups, pred = parents(att, sc)
                pred = pred[flag]
                res[name] = {"flagged": int(flag.sum()), "tpr": float(flag.mean()), "chance": 1 / len(groups),
                             "acc": float((pred == truth).mean()) if truth and len(pred) else None,
                             "confusion": pd.Series(pred).value_counts().to_dict()}
            print(json.dumps(res), flush=True)
            f.write_text(json.dumps(res, indent=1))
    rows = [json.loads(p.read_text()) for p in sorted(a.out.glob("s*.json"))]
    L = ["# C7b — taxonomy back-off on CICIoT2023 (PREREG 2026-10-26)\n",
         "Share of flagged unknown flows (own 1% cal threshold) given the correct parent; leaf cases only.\n",
         "| held-out | parent | method | flagged | correct parent | confusion (pooled) |", "|---|---|---|---|---|---|"]
    summ = {}
    for name in ("ddu", "softmax"):
        per = {}
        for r in rows:
            if not r["leaf"] or not r[name]["flagged"]:
                continue
            k = "+".join(r["held"])
            n, c = per.get(k, (0, 0, {}))[:2]
            conf = per.get(k, (0, 0, {}))[2]
            for g, v in r[name]["confusion"].items():
                conf[g] = conf.get(g, 0) + v
            per[k] = (n + r[name]["flagged"], c + r[name]["acc"] * r[name]["flagged"], conf)
        for k, (n, c, conf) in sorted(per.items()):
            L.append(f"| {k} | {PARENT[k.split('+')[0]]} | {name} | {n} | {c / n:.3f} | {conf} |")
        macro = np.mean([c / n for n, c, _ in per.values()])
        flow = sum(c for _, c, _ in per.values()) / sum(n for n, _, _ in per.values())
        chance = np.mean([r[name]["chance"] for r in rows if r["leaf"]])
        summ[name] = (macro, flow, chance)
    for name, (macro, flow, chance) in summ.items():
        tag = "C7b (claim)" if name == "ddu" else "softmax control"
        L.append(f"\n{tag}: {name} macro over leaf families {macro:.3f}, flow-weighted {flow:.3f}, chance {chance:.3f} → "
                 f"{'PASS' if macro >= 0.6 and macro >= 2 * chance else 'FAIL'}")
    text = "\n".join(L)
    print(text)
    Path(f"{a.out}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
