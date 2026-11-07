"""LONGMILP claim C2-LONG (PREREG §9 LONGMILP + amendments 1-3b) from the 600 s result files.

C2-LONG: B (BaB 600 s, proof rule) - rf_box (MILP 600 s, proof rule), pair bootstrap as CONF amendment 5;
PASS iff the 95% CI lower bound > 0. Also reported (not rules): the same under undecided-as-blocked, residual
undecided shares on both sides, per-pair wins, per-dataset means, and the coverage check (rf_box's 600 s proof value
vs every other tree's 20 s undecided-as-blocked value). Partial until all 24 pairs have seeds 2-4 on both sides.
Usage: .venv/bin/python scripts/lmilp_analyze.py
"""
import json

import numpy as np
import pandas as pd

from conf_analyze import ROOT, fmt, pair_means, rows


def tree_frame(T):
    T = T.drop_duplicates(["pair", "seed", "model", "robust_trained"], keep="last").copy()
    for p in ("unk", "known"):
        T[f"proof_{p}"] = T[f"milp_{p}"] - T[f"undecided_{p}"] / T[f"n_{p}"]
    T["proof"] = (T.proof_unk + T.proof_known) / 2
    T["und"] = (T.undecided_unk + T.undecided_known) / (T.n_unk + T.n_known)
    return T


def main():
    B = rows("conf_lmilp_b_*.jsonl").drop_duplicates(["pair", "seed"], keep="last")
    B["und"] = (B.undecided_unk + B.undecided_known) / (B.n_unk + B.n_known)
    TA = tree_frame(rows("conf_lmilp_trees_*.jsonl"))
    TA["cfg"] = TA.model + np.where(TA.robust_trained.astype(bool), "_box", "")
    T = TA[TA.cfg == "rf_box"]
    bv, okb = pair_means(B.melt(["pair", "seed"], ["bab", "babk5", "und"], "cfg", "val"), "cfg", "val")
    tv, okt = pair_means(T.melt(["pair", "seed"], ["proof", "milp", "und"], "cfg", "val"), "cfg", "val")
    common = bv.index.intersection(tv.index)
    complete = len(common) == 24 and okb.loc[common].all().all() and okt.loc[common].all().all()
    bv, tv = bv.loc[common], tv.loc[common]
    res = {}
    d = bv["bab"] - tv["proof"]
    res["C2_LONG"] = fmt("C2-LONG B(BaB 600s proof) - rf_box(MILP 600s proof)", d, common, complete)
    res["C2_LONG"].update(passed=bool(res["C2_LONG"]["ci"][0] > 0), wins=int((d > 0).sum()), ties=int((d == 0).sum()))
    d2 = bv["babk5"] - tv["milp"]
    res["C2_LONG_blocked"] = fmt("B(BaB 600s undecided=blocked) - rf_box(MILP 600s undecided=blocked)", d2, common,
                                 complete)
    res["C2_LONG_blocked"].update(wins=int((d2 > 0).sum()), ties=int((d2 == 0).sum()))
    und = {"B": round(float(bv["und"].mean()), 5), "rf_box": round(float(tv["und"].mean()), 5)}
    res["C2_LONG"]["residual_undecided"] = und
    res["C2_LONG"]["convention_free"] = bool(max(und.values()) < 0.005)
    # coverage: other trees' 20 s undecided-as-blocked values upper-bound them at any limit
    O = tree_frame(pd.concat([rows("conf_trees_*.jsonl"), rows("conf_groot_*.jsonl")], ignore_index=True))
    O = O[~((O.model == "rf") & (O.robust_trained == 1))]
    O["cfg"] = O.model + np.where(O.robust_trained.astype(bool) & (O.model != "groot"), "_box", "")
    ov = O[O.pair.isin(common)].groupby(["pair", "cfg"]).milp.mean().unstack()
    rfb = float(tv["proof"].mean())
    res["coverage"] = {"rf_box_600s_proof": round(rfb, 4), "others_20s_blocked": ov.mean().round(4).to_dict(),
                       "not_covered": [c for c, v in ov.mean().items() if v > rfb]}
    # LONGMILP-COVER (+ correction): other trees at 600 s on CTU-13 / ToN; elsewhere their 20 s undecided-as-blocked
    # value (an upper bound at any limit). Upper bound per tree = 600 s blocked where rerun, else 20 s blocked.
    R = TA[TA.cfg != "rf_box"]
    if len(R):
        r_ub = R.groupby(["pair", "cfg"]).milp.mean().unstack()
        r_pf = R.groupby(["pair", "cfg"]).proof.mean().unstack()
        r_n = R.groupby(["pair", "cfg"]).seed.nunique().unstack()
        ub = ov.copy()
        ub.update(r_ub[r_n == 3])
        res["coverage_tight"] = {"others_ub": ub.mean().round(4).to_dict(),
                                 "not_covered": [c for c, v in ub.mean().items() if v > rfb]}
        best = ov.copy()  # 20 s blocked (upper bound) where not rerun ...
        best.update(r_pf[r_n == 3])  # ... 600 s proof where rerun
        best["rf_box"] = tv["proof"]
        bt = best.loc[common].max(axis=1)
        rp = [p for p in common if p.split("/")[0] in ("nf_ctu13", "nf_ton")]
        rn = r_n.reindex(index=rp, columns=["rf", "xgb", "xgb_box"])
        okc = bool(complete and len(rp) == 12 and (rn == 3).all().all())
        d3 = bv["bab"] - bt
        res["C2_LONG_BEST"] = fmt("B(BaB 600s proof) - per-pair best tree (600 s proof where rerun, 20 s blocked else)",
                                  d3, common, okc)
        res["C2_LONG_BEST"].update(wins=int((d3 > 0).sum()), ties=int((d3 == 0).sum()))
    out = ROOT / "results" / "conf" / "lmilp_summary.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if "claim" in v:
            tag = "" if v["complete"] else " [PARTIAL: decides nothing]"
            pf = " PASS" if v.get("passed") else (" FAIL" if "passed" in v else "")
            print(f"{k}: mean {v['mean']:+.4f} CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] n={v['n_pairs']} "
                  f"wins {v['wins']} ties {v['ties']}{pf}{tag}\n   per dataset {v['per_dataset']}")
    print("residual undecided", und, "convention-free" if res["C2_LONG"]["convention_free"] else "")
    print("coverage", res["coverage"])
    if "coverage_tight" in res:
        print("coverage (tightened)", res["coverage_tight"])


if __name__ == "__main__":
    main()
