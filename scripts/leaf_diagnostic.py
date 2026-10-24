"""P1 step 1: closed-leaf diagnostic (PREREG §9, 2026-10-24 P1 row; post hoc). CICIoT outer folds, RF and XGB.

RF / XGB are refit with the bank's seed, rows and preprocessing and must reproduce the stored bank scores.
Leaf statistics come from the benign-train sample (bsample) only, known-attack counts from the model's own attack
training rows. Per fold and model: where held-out and benign test rows land (4b), whether they deviate on the
features the leaf's path never split on (4c), and how many hits land in leaves big enough for r >= 1 (4d).
Usage (one process per seed): for s in 0 1 2 3 4; .venv/bin/python scripts/leaf_diagnostic.py --seeds $s --out results/fusion/leaf_diagnostic_ciciot; end
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nids import fusion_checks as fc, models  # noqa: E402
import bank_report as br  # noqa: E402
import score_bank as sb  # noqa: E402

FOLDS = ["Flood", "BruteForce", "Recon", "Spoofing", "Web"]
N_TREES, N_BTEST, DOM, MIN_B, BIG_B = 100, 50_000, 0.01, 20, 200
AGGS = ("max", "top3", "rss")
LEAFDIR = Path("/home/srioo/Desktop/Projects/root/BigP/runs/leafdiag")


def rf_trees(rf, X_by_set, pick):
    """Per picked tree: (leaf id per row for each set, lo, hi bounds per node)."""
    for t in pick:
        tr = rf.estimators_[t].tree_
        lo = np.full((tr.node_count, tr.n_features), -np.inf)
        hi = np.full((tr.node_count, tr.n_features), np.inf)
        for n in range(tr.node_count):  # parents precede children in sklearn's node order
            f, l, r = tr.feature[n], tr.children_left[n], tr.children_right[n]
            if l == -1:
                continue
            for c in (l, r):
                lo[c], hi[c] = lo[n], hi[n]
            hi[l, f] = min(hi[l, f], tr.threshold[n])  # left: x <= thr
            lo[r, f] = max(lo[r, f], tr.threshold[n])
        yield {k: rf.estimators_[t].apply(X) for k, X in X_by_set.items()}, lo, hi


def xgb_trees(model, X_by_set, pick, d):
    booster = model.get_booster()
    df = booster.trees_to_dataframe()
    leaves = {}
    for k, X in X_by_set.items():
        parts = [booster.predict(xgboost.DMatrix(X[i:i + 20000]), pred_leaf=True)[:, pick].astype(np.int64)
                 for i in range(0, len(X), 20000)]
        leaves[k] = np.vstack(parts)
    for col, t in enumerate(pick):
        g = df[df.Tree == t].set_index("Node")
        n_nodes = int(g.index.max()) + 1
        lo = np.full((n_nodes, d), -np.inf)
        hi = np.full((n_nodes, d), np.inf)
        for n in sorted(g.index):  # breadth-first numbering: parents first
            row = g.loc[n]
            if row.Feature == "Leaf":
                continue
            f, s = int(row.Feature[1:]), float(row.Split)
            yes, no = (int(str(row[c]).split("-")[1]) for c in ("Yes", "No"))
            for c in (yes, no):
                lo[c], hi[c] = lo[n], hi[n]
            hi[yes, f] = min(hi[yes, f], s)  # yes: x < split
            lo[no, f] = max(lo[no, f], s)
        yield {k: v[:, col] for k, v in leaves.items()}, lo, hi


def seg_quantiles(leaf, X, qs):
    """Per-leaf quantiles of every column. leaf: compact ids 0..L-1. Returns (len(qs), L, d)."""
    L = leaf.max() + 1
    cnt = np.bincount(leaf, minlength=L)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    out = np.empty((len(qs), L, X.shape[1]))
    for j in range(X.shape[1]):
        o = np.lexsort((X[:, j], leaf))
        v = X[o, j]
        for qi, q in enumerate(qs):
            out[qi, :, j] = v[start + np.floor(q * (cnt - 1)).astype(int)]
    return out


def aggregate(z, mask):
    """max, top-3 mean and root-sum-square of z over the masked features, per row (nan if no feature)."""
    zz = np.where(mask, z, -np.inf)
    s = -np.sort(-zz, axis=1)
    k = mask.sum(1)
    with np.errstate(invalid="ignore"):
        top3 = np.where(k > 0, np.nanmean(np.where(np.isfinite(s[:, :3]), s[:, :3], np.nan), axis=1), np.nan)
    rss = np.sqrt(np.where(mask, z ** 2, 0).sum(1))
    return {"max": np.where(k > 0, s[:, 0], np.nan), "top3": top3, "rss": np.where(k > 0, rss, np.nan)}


PARTS = ("unsplit", "halfopen", "all", "unsplit_global", "halfopen_global", "all_global")


def one_model(trees, gscale, X, gmed):
    """Accumulate 4b / 4c / 4d over the picked trees. X: 'bt' benign train, 'at' attack train, 'unk', 'ben'.
    *_global = the same deviation with global benign-train median and scale (leaf-conditioning control)."""
    hit = {k: [] for k in ("unk", "ben")}
    big = {k: [] for k in ("unk", "ben")}
    ge20 = {k: [] for k in ("unk", "ben")}
    cover = []
    dev = {(s, part, a): [] for s in ("unk", "ben") for part in PARTS for a in AGGS}
    for leaves, lo, hi in trees:
        allids = np.concatenate([leaves[k] for k in ("bt", "at", "unk", "ben")])
        uniq, inv = np.unique(allids, return_inverse=True)
        sizes = np.cumsum([0] + [len(leaves[k]) for k in ("bt", "at", "unk", "ben")])
        cid = {k: inv[sizes[i]:sizes[i + 1]] for i, k in enumerate(("bt", "at", "unk", "ben"))}
        L = len(uniq)
        nb, na = np.bincount(cid["bt"], minlength=L), np.bincount(cid["at"], minlength=L)
        dom = (nb > 0) & (na < DOM * (na + nb))
        stat = dom & (nb >= MIN_B)
        cover.append(((nb[dom] >= BIG_B).mean() if dom.any() else np.nan))
        for k in ("unk", "ben"):
            h = dom[cid[k]]
            hit[k].append(h.mean())
            ge20[k].append((nb[cid[k]][h] >= MIN_B).mean() if h.any() else np.nan)
            big[k].append((nb[cid[k]][h] >= BIG_B).mean() if h.any() else np.nan)
        # 4c: leaves with stats that hold both held-out and benign test hits
        shared = stat & (np.bincount(cid["unk"], minlength=L) > 0) & (np.bincount(cid["ben"], minlength=L) > 0)
        if not shared.any():
            continue
        rows_b = shared[cid["bt"]]
        remap = -np.ones(L, np.int64)
        remap[np.flatnonzero(shared)] = np.arange(shared.sum())
        q10, med, q90 = seg_quantiles(remap[cid["bt"][rows_b]], X["bt"][rows_b], (0.10, 0.50, 0.90))
        scale = np.maximum(q90 - q10, 0.05 * gscale)
        lo_s, hi_s = lo[uniq[shared]], hi[uniq[shared]]
        unsplit = ~np.isfinite(lo_s) & ~np.isfinite(hi_s)
        halfopen = np.isfinite(lo_s) ^ np.isfinite(hi_s)
        for k in ("unk", "ben"):
            r = shared[cid[k]]
            li = remap[cid[k][r]]
            z = np.abs(X[k][r] - med[li]) / scale[li]
            zg = np.abs(X[k][r] - gmed) / gscale
            every = np.ones_like(unsplit)
            for part, m in (("unsplit", unsplit), ("halfopen", halfopen), ("all", every)):
                for name, zz in ((part, z), (f"{part}_global", zg)):
                    for a, v in aggregate(zz, m[li]).items():
                        dev[k, name, a].append(v)
    res = {"hit_unk": np.mean(hit["unk"]), "hit_ben": np.mean(hit["ben"]),
           "ge20_unk": np.nanmean(ge20["unk"]), "ge20_ben": np.nanmean(ge20["ben"]),
           "ge200_unk": np.nanmean(big["unk"]), "ge200_ben": np.nanmean(big["ben"]),
           "dom_leaves_ge200": np.nanmean(cover)}
    for part in PARTS:
        for a in AGGS:
            u = np.concatenate(dev["unk", part, a]) if dev["unk", part, a] else np.array([])
            b = np.concatenate(dev["ben", part, a]) if dev["ben", part, a] else np.array([])
            u, b = u[np.isfinite(u)], b[np.isfinite(b)]
            ok = len(u) > 0 and len(b) > 0
            res[f"{part}_{a}_ratio"] = float(np.median(u) / max(np.median(b), 1e-9)) if ok else np.nan
            res[f"{part}_{a}_auroc"] = float(roc_auc_score(np.r_[np.zeros(len(b)), np.ones(len(u))], np.r_[b, u])) \
                if ok else np.nan
            res[f"{part}_n_unk"] = len(u)
    return res


def one_seed(seed):
    bank = sb.Bank("ciciot_csv", seed, "none", 40000, 200000)
    run = sb.OUT / f"ciciot_csv_none_cap40000_s{seed}"
    assert not np.intersect1d(bank.bsample, np.union1d(bank.rank, bank.cal)).size  # leaf stats never see rank/cal
    LEAFDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LEAFDIR / f"ciciot_s{seed}_leafstat_rows.npz", bsample=bank.bsample)
    rng = np.random.default_rng(seed)
    out = []
    for k in FOLDS:
        t0 = time.time()
        P = (k,)
        fit, es = bank.fit_rows(P)
        dfit = bank.frame(fit, P)
        prep = models.Prep().fit(dfit)
        known = sorted(set(bank.fam[fit]))
        bi = known.index("Benign")
        yf = np.array([known.index(f) for f in bank.fam[fit]])
        Xf = prep.transform(dfit)
        part = pd.read_parquet(run / f"config=outer_{k}" / "part.parquet", columns=["role", "row_id", "rf", "xgb"])
        cal = part[part.role == "cal"]
        Xc = prep.transform(bank.frame(cal.row_id.to_numpy(), P))
        diff = lambda mdl, m: float(np.abs((1 - mdl.predict_proba(Xc)[:, bi]) - cal[m].to_numpy()).max())  # noqa: E731
        rf = models.rf(seed).fit(Xf, yf)
        rep = {"rf": diff(rf, "rf")}
        xg = models.xgb(seed).fit(Xf, yf)
        rep["xgb"] = diff(xg, "xgb")
        assert rep["rf"] < 1e-4, f"RF refit does not reproduce the bank: {rep}"
        # XGB refit is deterministic but not bit-identical to the bank (PREREG §9 amendment): equivalence check
        unk = part[part.role == "test_unknown"]
        Xu = prep.transform(bank.frame(unk.row_id.to_numpy(), ()))
        s_cal = 1 - xg.predict_proba(Xc)[:, bi]
        tau_new, tau_old = fc.threshold(s_cal, 0.01), fc.threshold(cal.xgb.to_numpy(np.float64), 0.01)
        rep["xgb_tpr_diff"] = float(abs(((1 - xg.predict_proba(Xu)[:, bi]) > tau_new).mean()
                                        - (unk.xgb.to_numpy() > tau_old).mean()))
        rep["xgb_spearman"] = float(stats.spearmanr(s_cal, cal.xgb.to_numpy()).statistic)
        rep["xgb_tpr_bank"] = float((unk.xgb.to_numpy() > tau_old).mean())
        rep["xgb_tpr_refit"] = float(((1 - xg.predict_proba(Xu)[:, bi]) > tau_new).mean())
        print(f"s{seed} {k} refit: rf maxdiff {rep['rf']:.1e}; xgb maxdiff {rep['xgb']:.4f} TPR@1% bank "
              f"{rep['xgb_tpr_bank']:.4f} refit {rep['xgb_tpr_refit']:.4f} spearman {rep['xgb_spearman']:.5f}",
              flush=True)
        ben_idx = np.sort(rng.choice(bank.test_benign, min(N_BTEST, len(bank.test_benign)), replace=False))
        X = {"bt": prep.transform(bank.frame(bank.bsample, P)), "at": Xf[yf != bi],
             "unk": prep.transform(bank.frame(bank.test[k], ())), "ben": prep.transform(bank.frame(ben_idx, ()))}
        q = np.quantile(X["bt"], [0.1, 0.9], axis=0)
        gscale = np.maximum(q[1] - q[0], 1e-6)
        gmed = np.median(X["bt"], axis=0)
        d = Xf.shape[1]
        pick_rf = np.sort(rng.choice(len(rf.estimators_), N_TREES, replace=False))
        n_xgb = len(xg.get_booster().get_dump())
        pick_xgb = np.sort(rng.choice(n_xgb, N_TREES, replace=False))
        todo = [("rf", rf_trees(rf, X, pick_rf), [rf.estimators_[t].get_n_leaves() for t in pick_rf])]
        todo.append(("xgb", xgb_trees(xg, X, pick_xgb, d), None))  # refit XGB, used for every XGB number
        for name, trees, n_leaves in todo:
            r = one_model(trees, gscale, X, gmed)
            r.update({"seed": seed, "fold": k, "model": name, "refit_maxdiff": rep[name], "xgb_tpr_diff": rep["xgb_tpr_diff"],
                      "xgb_spearman": rep["xgb_spearman"],
                      "xgb_tpr_bank": rep["xgb_tpr_bank"], "xgb_tpr_refit": rep["xgb_tpr_refit"], "n_unk": len(X["unk"]),
                      "leaves_per_tree": float(np.mean(n_leaves)) if n_leaves else np.nan})
            out.append(r)
            print(f"s{seed} {k:10s} {name:3s} hit unk {r['hit_unk']:.3f} ben {r['hit_ben']:.3f} | unsplit max "
                  f"ratio {r['unsplit_max_ratio']:.2f} auroc {r['unsplit_max_auroc']:.3f} | >=200 {r['ge200_unk']:.3f}"
                  f" | {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    # one process per seed (two banks in memory at once hit the OOM killer); the report reads every seed's file
    for s in a.seeds:
        Path(f"{a.out}_s{s}.json").write_text(json.dumps(one_seed(s), indent=1, default=float))
    res = pd.DataFrame([r for f in sorted(a.out.parent.glob(f"{a.out.name}_s*.json")) for r in json.loads(f.read_text())])
    a.out.with_suffix(".json").write_text(json.dumps(res.to_dict("records"), indent=1, default=float))
    f = lambda x: br.fmt(*br.ci(x))  # noqa: E731
    L = ["# P1 step 1: closed-leaf diagnostic, CICIoT (post hoc)\n",
         f"seeds {sorted(res.seed.unique())}; {N_TREES} random trees per model; benign-dominated = < 1% known-attack "
         f"training rows; mean [95% t-CI] over seeds\n",
         "## 4b: share of tree hits in benign-dominated leaves, and of those, in leaves with >= 20 / >= 200 benign rows\n",
         "| fold | model | held-out in benign-dom. | benign test in benign-dom. | held-out >= 20 | held-out >= 200 | benign >= 200 |",
         "|---|---|---|---|---|---|---|"]
    for (k, m), g in res.groupby(["fold", "model"], sort=False):
        L.append(f"| {k} | {m} | {f(g.hit_unk)} | {f(g.hit_ben)} | {f(g.ge20_unk)} | {f(g.ge200_unk)} | {f(g.ge200_ben)} |")
    for part in PARTS:
        L.append(f"\n## 4c: {part}-feature deviation, held-out vs benign test rows in the same benign-dominated leaves\n")
        L.append("| fold | model | " + " | ".join(f"{a} ratio | {a} AUROC" for a in AGGS) + " |")
        L.append("|---|---|" + "---|---|" * len(AGGS))
        for (k, m), g in res.groupby(["fold", "model"], sort=False):
            L.append(f"| {k} | {m} | " + " | ".join(f"{f(g[f'{part}_{a}_ratio'])} | {f(g[f'{part}_{a}_auroc'])}"
                                                   for a in AGGS) + " |")
    L.append("\n## Leaf conditioning vs global statistics (same rows, same leaves): AUROC leaf minus AUROC global\n")
    L.append("| fold | model | " + " | ".join(f"{p} {a}" for p in ("unsplit", "all") for a in AGGS) + " |")
    L.append("|---|---|" + "---|" * (2 * len(AGGS)))
    for (k, m), g in res.groupby(["fold", "model"], sort=False):
        L.append(f"| {k} | {m} | " + " | ".join(f(g[f"{p}_{a}_auroc"] - g[f"{p}_global_{a}_auroc"])
                                               for p in ("unsplit", "all") for a in AGGS) + " |")
    L.append("\n## XGB bank vs refit, TPR@1%FPR on the held-out family (refit used for every XGB number)\n")
    L.append("| fold | bank | refit | cal Spearman |\n|---|---|---|---|")
    for k, g in res[res.model == "xgb"].groupby("fold", sort=False):
        L.append(f"| {k} | {f(g.xgb_tpr_bank)} | {f(g.xgb_tpr_refit)} | {g.xgb_spearman.min():.5f} |")
    L.append("\n## 4d: coverage for r >= 1 (leaves with >= 200 benign-train rows)\n")
    L.append("| model | leaves/tree | benign-dom. leaves with >= 200 | held-out hits there | benign hits there |\n|---|---|---|---|---|")
    for m, g in res.groupby("model"):
        L.append(f"| {m} | {g.leaves_per_tree.mean():.0f} | {f(g.dom_leaves_ge200)} | {f(g.ge200_unk)} | {f(g.ge200_ben)} |")
    L.append("\n## Stop rule (PREREG §9 P1 row)\n")
    ok = {}
    for k in ("Flood", "BruteForce"):
        for m in ("rf", "xgb"):
            g = res[(res.fold == k) & (res.model == m)]
            land = g.hit_unk.mean() > 0.5
            dev = any(g[f"unsplit_{x}_auroc"].mean() >= 0.70 and g[f"unsplit_{x}_ratio"].mean() >= 2 for x in AGGS)
            ok[k, m] = (land, dev)
            L.append(f"- {k} / {m}: majority in benign-dominated leaves {'yes' if land else 'NO'}; "
                     f"unsplit deviation AUROC >= 0.70 and ratio >= 2 {'yes' if dev else 'NO'}")
    passed = all(x and y for x, y in ok.values())
    L.append(f"\n**Step 1: {'PASS (build steps 2-4)' if passed else 'FAIL (mechanism not supported; stop)'}**")
    L.append(f"\nrefit vs bank: RF max |diff| {res[res.model == 'rf'].refit_maxdiff.max():.1e}; XGB max |diff| "
             f"{res.refit_maxdiff[res.model == 'xgb'].max():.4f} (unexplained; refit used throughout)")
    for k in ("Flood", "BruteForce"):
        for m in ("rf", "xgb"):
            g = res[(res.fold == k) & (res.model == m)]
            best = max(AGGS, key=lambda x: g[f"unsplit_{x}_auroc"].mean())
            L.append(f"- control {k} / {m}: best unsplit aggregate {best}: leaf AUROC {g[f'unsplit_{best}_auroc'].mean():.3f}"
                     f" vs global same features {g[f'unsplit_global_{best}_auroc'].mean():.3f}, global all features "
                     f"{g[f'all_global_{best}_auroc'].mean():.3f}")
    text = "\n".join(L)
    print(text)
    a.out.with_suffix(".md").write_text(text + "\n")


if __name__ == "__main__":
    main()
