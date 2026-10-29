"""Prototype gate (PREREG §9 TOK / PLE). Reads prototype parts and the base bank parts of the same seed (rows joined by
(row_id, role)); no training.

Macro host-day TPR over held-out families (host_stream.fold, 1% budget); per detector the better of P / A. nf_ugr16:
families with July week 5 and August rows only; forward benign FPR with host-bootstrap CI (forward.py).
Usage: .venv/bin/python scripts/flowtok_eval.py [--seed 1] [--tag flowtok|ple]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import forward  # noqa: E402
import host_stream as hs  # noqa: E402
import score_bank as sb  # noqa: E402
from nids import load  # noqa: E402

DATASETS = ("nf_ugr16", "nf_ctu13", "nf_iot23")
ARMS = {"flowtok": ["ft_tree", "ft_order0", "ft_mlp", "ft_hc"], "ple": ["ple_ddu", "ple_ce", "clip_ddu"]}  # primary first
FT = ARMS["flowtok"]
BASE = ["rf", "xgb", "kitnet", "mlp_sn_gda"]
OUT = ROOT / "results" / "flowtok"
TAG = "flowtok"


def evaluate(ds, seed):
    z = np.load(ROOT / "data" / "processed" / f"{ds}_hostmeta.npz", allow_pickle=False)
    meta = {"src": z["src"], "t": z["t"], "fam": z["fam"]}
    cal_host = hs.host_split(z["ips"], seed)
    both = None
    if ds == "nf_ugr16":
        w = pd.read_parquet(load.CACHE / f"{ds}.parquet", columns=["family", "src"]).astype(str)
        both = set(w.family[w.src == "july_week5"]) & set(w.family[w.src.str.startswith("august")])
    run = sb.OUT / f"{ds}_host_cap40000_s{seed}"
    rows, fwd = [], {}
    for p in sorted((sb.OUT / f"{run.name}_{TAG}").glob("config=outer_*/part.parquet")):
        held = p.parent.name.removeprefix("config=outer_")
        if both is not None and held not in both:
            continue
        ft = pd.read_parquet(p, columns=["row_id", "role"] + FT)
        base = sb.read_part(run / p.parent.name / "part.parquet", ["row_id", "role", "family"] + BASE)
        part = base.merge(ft, on=["row_id", "role"], how="inner", validate="one_to_one")
        assert len(part) == len(base) == len(ft), f"{ds} {held}: FlowTok rows != base rows"
        if not (part.role == "test_unknown").any():
            continue
        for d in FT + BASE:
            s = part[d].to_numpy(np.float64)
            r = hs.fold(part, s, meta, cal_host, held)[0]
            rows += [{"ds": ds, "fold": held, "detector": d, "method": m, "tpr": r[f"tpr_{m}"], "fpr": r[f"fpr_{m}"]}
                     for m in ("P", "A")]
            if ds == "nf_ugr16":
                for (split, m), a in forward.alarms(part, s, meta, cal_host).items():
                    if split == "forward":
                        fwd.setdefault((d, m), []).append(a.astype(float))
    R = pd.DataFrame(rows).groupby(["ds", "detector", "method"])[["tpr", "fpr"]].mean().reset_index()
    if fwd:
        ci = {k: (pd.concat(v, axis=1).mean(axis=1)) for k, v in fwd.items()}
        R["fwd_fpr"] = [ci[(d, m)].mean() if (d, m) in ci else np.nan for d, m in zip(R.detector, R.method)]
        R["fwd_hi"] = [forward.ci(ci[(d, m)])[1] if (d, m) in ci else np.nan for d, m in zip(R.detector, R.method)]
    return R


def gate(R):
    best = R.loc[R.groupby(["ds", "detector"]).tpr.idxmax()].set_index(["ds", "detector"])
    verdict, passes = [], 0
    for ds in DATASETS:
        if ds not in best.index.get_level_values(0):
            continue
        f = best.loc[(ds, FT[0])]
        mb = max(best.loc[(ds, b)].tpr for b in BASE)
        tpr_ok = f.tpr >= mb - 0.05
        if ds == "nf_ugr16":
            fpr_ok = f.fwd_hi <= 0.015
        else:
            fpr_ok = f.fpr <= R[(R.ds == ds) & R.detector.isin(BASE) & (R.method == f.method)].fpr.max()
        passes += bool(tpr_ok and fpr_ok)
        verdict.append({"ds": ds, f"{FT[0]}_tpr": f.tpr, "method": f.method, "max_baseline_tpr": mb, "tpr_ok": tpr_ok,
                        "fpr_ok": fpr_ok, **({"fwd_fpr_hi": f.fwd_hi} if ds == "nf_ugr16" else {"fpr": f.fpr})})
    return pd.DataFrame(verdict), passes >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="flowtok", choices=list(ARMS))
    a = ap.parse_args()
    global FT, TAG, OUT
    FT, TAG, OUT = ARMS[a.tag], a.tag, ROOT / "results" / a.tag
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.concat([evaluate(ds, a.seed) for ds in DATASETS
                   if (sb.OUT / f"{ds}_host_cap40000_s{a.seed}_{TAG}" / "manifest_outer.json").exists()],
                  ignore_index=True)
    R.to_csv(OUT / "measures.csv", index=False)
    V, ok = gate(R)
    T = R.pivot_table(index="detector", columns=["ds", "method"], values="tpr").round(3)
    L = [f"# {TAG} prototype gate (PREREG §9 {'TOK' if TAG == 'flowtok' else 'PLE'})\n", f"Seed {a.seed}. Macro host-day TPR over held-out families.\n",
         T.to_string(), "\n## Gate\n", V.to_string(index=False),
         f"\n**Gate: {'PASS' if ok else 'FAIL'}** (needs >= 2 of 3 datasets)\n"]
    if "fwd_fpr" in R:
        U = R[R.ds == "nf_ugr16"][["detector", "method", "fwd_fpr", "fwd_hi"]].round(4)
        L += ["\n## nf_ugr16 forward benign FPR\n", U.to_string(index=False)]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
