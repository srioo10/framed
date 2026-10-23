"""Score bank for H5 Stages 1-3 (PREREG §9, 2026-10-24 rows).

For one (dataset, variant, seed): fit every member once per removal set and store raw per-row scores, so all
fusion / threshold / gate analyses run later on cached scores without refitting.

Configurations
  outer {k}       fit on benign + F minus k; scores benign_rank, benign_cal, benign test, k's test rows, known test
  pair  P         P = outer set R0 plus one inner family j; fit on benign + F minus P; scores benign_rank and
                  benign_cal, plus pseudo-unknown rows (train + val) of j for each outer fold R0 it serves
                  (a singleton pair {k, j} serves fold k with j and fold j with k)
Window rule (unsw_raw, variant ct_causal): rows used for fitting, early stopping, rank and cal are computed with
the model's removal set removed; pseudo-unknown rows of j with only R0 removed; final test rows with all flows.

Usage: .venv/bin/python scripts/score_bank.py DATASET --seed 0 [--variant ct_causal|ct_orig|ct_drop]
       [--kind outer|pairs|all] [--cap 40000] [--bcap 200000] [--extra] [--out DIR]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nids import clean, load, mcde, models, split, windows  # noqa: E402

OUT = Path(os.environ.get("NIDS_BANK_DIR", "/home/srioo/Desktop/Projects/root/BigP/runs/bank"))
EXTRA = {"ciciot_csv": [("Flood", "Mirai"), ("DoS",), ("DDoS",)]}  # joint unseen-family; two attack-type hold-outs
SEP = "__"  # family names contain "+", so configs are joined with SEP


def role_removal(P, role, R0=None):
    """Families removed from the window population for a role of the model fit with removal set P (PREREG §9):
    fit / es / bsample / rank / cal -> P; pseudo-unknown rows of outer fold R0 -> R0; final test -> nothing."""
    if role in ("fit", "es", "bsample", "rank", "cal"):
        return tuple(P)
    if role == "pseudo":
        assert R0 is not None and set(R0) < set(P)
        return tuple(R0)
    if role in ("test_benign", "test_unknown", "test_known"):
        return ()
    raise ValueError(role)


def fine_family(df, ds):
    """Family with CICIoT Flood split into its attack types (DoS / DDoS) for the attack-type hold-outs."""
    fam = df["family"].astype(str).to_numpy()
    if ds != "ciciot_csv":
        return fam.copy()
    y = df["y"].astype(str).to_numpy()
    return np.where(fam == "Flood", np.where(np.char.startswith(y.astype(str), "DDoS"), "DDoS", "DoS"), fam)


def cap_by_block(idx, groups, cap, rng):
    """Up to `cap` rows of idx, allocated across groups in proportion to group size (within-group random rank)."""
    if len(idx) <= cap:
        return np.sort(idx)
    u = rng.random(len(idx))
    if groups is None:
        return np.sort(idx[np.argsort(u)[:cap]])
    g = groups[idx]
    order = np.lexsort((u, g))
    gs = g[order]
    start = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    size = np.diff(np.r_[start, len(gs)])
    pos = np.arange(len(gs)) - np.repeat(start, size)
    frac = (pos + u[order]) / np.repeat(size, size)  # spreads every group evenly over [0, 1)
    return np.sort(idx[order[np.argsort(frac, kind="stable")[:cap]]])


def split_by_group(idx, groups, rng, share=0.5):
    """Two disjoint halves of idx; whole groups go to one side (random when there are no groups)."""
    if groups is None:
        p = rng.permutation(idx)
        cut = int(round(share * len(p)))
        return np.sort(p[:cut]), np.sort(p[cut:])
    ug, cnt = np.unique(groups[idx], return_counts=True)
    o = rng.permutation(len(ug))
    first = set(ug[o][np.cumsum(cnt[o]) <= share * len(idx)])
    a = np.isin(groups[idx], list(first))
    return np.sort(idx[a]), np.sort(idx[~a])


class Bank:
    def __init__(self, ds, seed, variant, cap, bcap):
        t0 = time.time()
        self.ds, self.seed, self.variant = ds, seed, variant
        rng = np.random.default_rng(seed)
        raw = load.LOADERS[ds]()
        self.flows = windows.Flows(raw) if ds == "unsw_raw" else None
        self.raw_fam = None
        if self.flows is not None:
            self.raw_fam = self.flows.family
        df, rep = clean.clean(raw, ds)
        del raw
        self.kept = rep["kept_rows"]
        self.df = df
        self.fam = df["family"].astype(str).to_numpy()
        self.fine = fine_family(df, ds)
        self.groups = df["group"].to_numpy() if "group" in df.columns else None
        self.feats = clean.feature_columns(df)
        if variant == "ct_drop":
            self.feats = [c for c in self.feats if c not in windows.KEYS]
        tr, va, te = split.grouped_split(df, seed)
        self.families = sorted(set(self.fam) - {"Benign"})
        b = self.fam == "Benign"
        self.cap = {f: cap_by_block(tr[self.fam[tr] == f], self.groups, cap, rng) for f in ["Benign"] + self.families}
        self.bsample = cap_by_block(tr[b[tr]], self.groups, bcap, rng)
        self.rank, self.cal = split_by_group(va[b[va]], self.groups, rng)
        self.val_attack = {f: cap_by_block(va[self.fam[va] == f], self.groups, cap // 4, rng) for f in self.families}
        self.test_benign = te[b[te]]
        self.test = {f: cap_by_block(te[self.fam[te] == f], self.groups, cap // 2, rng) for f in self.families}
        self.rng = rng
        self.t_load = time.time() - t0

    # -- helpers -------------------------------------------------------------------------------------------
    def removed(self, rows, R):
        return np.isin(self.fam[rows], list(R)) | np.isin(self.fine[rows], list(R))

    def frame(self, rows, R):
        """Feature frame for `rows`, with causal ct_* computed against all flows minus families R."""
        X = self.df.iloc[rows][self.feats].reset_index(drop=True)
        if self.variant == "ct_causal":
            pop = ~np.isin(self.raw_fam, list(R))
            X[list(windows.KEYS)] = self.flows.features(pop, self.kept[rows])
        return X

    def configs(self, kind, extra):
        outer = [(f,) for f in self.families] + (EXTRA.get(self.ds, []) if extra else [])
        pairs = {}
        for R0 in outer:
            for j in self.families:
                if j in R0 or self.removed(self.cap[j], R0).any():  # j overlaps R0 (e.g. Flood when DoS is held out)
                    continue
                P = tuple(sorted(set(R0) | {j}))
                pairs.setdefault(P, []).append((R0, j))
        out = []
        if kind in ("outer", "all"):
            out += [("outer", R0, [(R0, None)]) for R0 in outer]
        if kind in ("pairs", "all"):
            out += [("pair", P, served) for P, served in sorted(pairs.items())]
        return out

    # -- one removal set -----------------------------------------------------------------------------------
    def run_config(self, kind, P, served):
        t0 = time.time()
        seed = self.seed
        fit = np.sort(np.concatenate([r[~self.removed(r, P)] for r in self.cap.values()]))
        es = split_by_group(fit, self.groups, np.random.default_rng([seed, zlib.crc32(SEP.join(P).encode())]), share=0.9)[1]
        fit = np.setdiff1d(fit, es)
        known = sorted(set(self.fam[fit]))
        code = {f: i for i, f in enumerate(known)}
        bi, k = code["Benign"], len(known)
        dfit = self.frame(fit, role_removal(P, "fit"))
        prep = models.Prep().fit(dfit)
        Xf, yf = prep.transform(dfit), np.array([code[f] for f in self.fam[fit]])
        Xe, ye = prep.transform(self.frame(es, role_removal(P, "es"))), np.array([code[f] for f in self.fam[es]])
        rf = models.rf(seed).fit(Xf, yf)
        xg = models.xgb(seed).fit(Xf, yf)
        mlp = models.train_mlp(Xf, yf, Xe, ye, k, seed=seed)
        dben = self.frame(self.bsample, role_removal(P, "bsample"))
        bprep = mcde.RobustPrep().fit(dben)
        Xb = bprep.transform(dben)
        del dben
        modes = mcde.BenignModes().fit(Xb, seed)
        iso = models.iforest(Xb, seed)
        extra = {}
        if kind == "outer":
            soft = models.oof_proba(Xf, yf, seed)
            extra["mlp_kd"] = models.train_mlp(Xf, yf, Xe, ye, k, soft=soft, T=4.0, seed=seed)
            extra["xgb_tail"] = models.xgb_tail(Xf, yf, bi, seed)
            sprep = mcde.RobustPrep().fit(dfit)
            extra["hgb"] = (sprep, mcde.hgb(seed).fit(sprep.transform(dfit), (yf != bi).astype(int)))

        def score(rows, role, R0=None):
            d = self.frame(rows, role_removal(P, role, R0))
            served_fold = SEP.join(R0) if R0 else ""
            X = prep.transform(d)
            prf, pxg, pml = rf.predict_proba(X), xg.predict_proba(X), models.mlp_proba(mlp, X)
            xb = bprep.transform(d)
            dist, mode = modes.score(xb)
            cols = {"row_id": rows.astype(np.int64), "role": role, "served": served_fold,
                    "family": self.fam[rows], "fine": self.fine[rows],
                    "rf": 1 - prf[:, bi], "xgb": 1 - pxg[:, bi], "mlp_ce": 1 - pml[:, bi],
                    "iforest": -iso.score_samples(xb), "clust": dist, "mode": mode}
            if kind == "outer":
                cols["mlp_kd"] = 1 - models.mlp_proba(extra["mlp_kd"], X)[:, bi]
                cols["xgb_tail"] = 1 - extra["xgb_tail"].predict_proba(X)[:, bi]
                sp, h = extra["hgb"]
                cols["mcde_sup"] = h.predict_proba(sp.transform(d))[:, 1]
                if role == "test_known":
                    for name, p in (("rf", prf), ("xgb", pxg), ("mlp_ce", pml)):
                        cols[f"{name}_pred"] = np.array(known)[p.argmax(1)]
            out = pd.DataFrame(cols)
            for c in out.columns:
                if out[c].dtype == np.float64:
                    out[c] = out[c].astype(np.float32)
            return out

        parts = [score(self.rank, "rank"), score(self.cal, "cal")]
        if kind == "outer":
            R0 = P
            unk = np.sort(np.concatenate([r[self.removed(r, R0)] for r in self.test.values()]))
            kn = np.sort(np.concatenate([r[~self.removed(r, R0)] for r in self.test.values()]))
            parts += [score(self.test_benign, "test_benign"), score(unk, "test_unknown"), score(kn, "test_known")]
        else:
            for R0, j in served:
                rows = np.sort(np.concatenate([self.cap[f] for f in self.families] +
                                              [self.val_attack[f] for f in self.families]))
                rows = rows[self.removed(rows, P) & ~self.removed(rows, R0)]  # j's rows (train + val)
                parts.append(score(rows, "pseudo", R0))
        res = pd.concat(parts, ignore_index=True)
        info = {"kind": kind, "removal": list(P), "served": [[SEP.join(r), j] for r, j in served],
                "n_fit": int(len(fit)), "n_es": int(len(es)), "known": known, "seconds": time.time() - t0,
                "modes": modes.sizes.tolist()}
        return res, es, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["unsw_raw", "ciciot_csv", "unsw_official"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="ct_causal", choices=["ct_causal", "ct_orig", "ct_drop", "none"])
    ap.add_argument("--kind", default="all", choices=["outer", "pairs", "all"])
    ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--bcap", type=int, default=200000)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    variant = a.variant if a.dataset == "unsw_raw" else "none"
    bank = Bank(a.dataset, a.seed, variant, a.cap, a.bcap)
    run = a.out / f"{a.dataset}_{variant}_cap{a.cap}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    cfgs = bank.configs(a.kind, a.extra)
    n_scored = sum(len(bank.rank) + len(bank.cal) + (len(bank.test_benign) + sum(map(len, bank.test.values()))
                                                     if k == "outer" else 60_000 * len(s)) for k, _, s in cfgs)
    expected = n_scored * 64
    free = shutil.disk_usage(run).free
    if free < 3 * expected:
        sys.exit(f"not enough disk: {free / 1e9:.1f} GB free, need ~{3 * expected / 1e9:.1f} GB")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "scripts"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    ids = {"bsample": bank.bsample, "rank": bank.rank, "cal": bank.cal, "test_benign": bank.test_benign,
           **{f"cap_{f}": r for f, r in bank.cap.items()}, **{f"val_{f}": r for f, r in bank.val_attack.items()}}
    manifest = {"dataset": a.dataset, "variant": variant, "seed": a.seed, "cap": a.cap, "bcap": a.bcap,
                "commit": commit, "dirty": dirty, "load_seconds": bank.t_load, "expected_bytes": int(expected),
                "n": {k: int(len(v)) for k, v in ids.items()}, "configs": {}}
    mpath = run / f"manifest_{a.kind}{'_extra' if a.extra else ''}.json"
    for kind, P, served in cfgs:
        name = SEP.join(P)
        dest = run / f"config={kind}_{name}"
        if (dest / "part.parquet").exists():
            print("skip", dest.name, flush=True)
            continue
        res, es, info = bank.run_config(kind, P, served)
        dest.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pandas(res, preserve_index=False), dest / "part.parquet")
        ids[f"es_{kind}_{name}"] = es
        manifest["configs"][f"{kind}_{name}"] = info
        np.savez_compressed(run / f"ids_{a.kind}{'_extra' if a.extra else ''}.npz", **ids)
        manifest["actual_bytes"] = sum(f.stat().st_size for f in run.rglob("*.parquet"))
        mpath.write_text(json.dumps(manifest, indent=1))
        print(f"{kind:5s} {name:28s} fit {info['n_fit']:7d} rows {len(res):8d}  {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
