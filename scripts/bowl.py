"""The bowl (PREREG §9 BOWL): NSGA-II tournament of small neural / neural-hybrid detectors on the training-side arena.

fight:   NSGA-II on (arena fitness up, size down, time ratio vs heavy XGB down) under size <= 64 KB and ratio <= 0.5;
         cheap fitness (3 pairs per dataset, 40k fit rows, 15 epochs); Optuna sqlite storage, resumable.
death:   feasible Pareto front (top 10 by fitness) + bosses on the full arena (all pairs, full rows, 2 seeds, 30 epochs).
autopsy: where boss XGB catches pseudo-unknown flows the champion misses; a depth-3 tree names the pattern.
Usage: .venv/bin/python scripts/bowl.py fight --round 1 --trials 256
       .venv/bin/python scripts/bowl.py death --round 1
       .venv/bin/python scripts/bowl.py autopsy --round 1
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena  # noqa: E402
import fighters as fx  # noqa: E402

ROOT = HERE.parent
ARENA = ROOT / "data" / "arena"
OUT = ROOT / "results" / "bowl"
SIZE_MAX, RATIO_MAX = 65_536, 0.5
DATASETS = ("nf_ctu13", "nf_ugr16", "nf_cic")


def load_pairs(arena_dir, cheap, seed=0):
    P = []
    for ds in DATASETS:
        files = sorted((arena_dir / ds).glob("pair_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
        for f in files[:3] if cheap else files:
            z = dict(np.load(f))
            if cheap and len(z["y_fit"]) > 40_000:
                i = np.sort(np.random.default_rng(seed).choice(len(z["y_fit"]), 40_000, replace=False))
                z["X_fit"], z["y_fit"], z["teacher"] = z["X_fit"][i], z["y_fit"][i], z["teacher"][i]
            z["ds"], z["name"], z["bi"], z["dir"] = ds, f.stem, int(z["bi"]), f.parent
            z["k"] = z["teacher"].shape[1]
            P.append(z)
    return P


def suggest(trial, roster):
    """Define-by-run genome; roster["fixed"] pins power-up genes."""
    fixed = roster.get("fixed", {})
    pick = lambda name, opts: fixed[name] if name in fixed else trial.suggest_categorical(name, opts)  # noqa: E731
    g = {"family": pick("family", roster["families"])}
    if g["family"] in ("mlp", "tabm"):
        g["enc"] = pick("enc", ["raw", "quantile", "tree", "periodic"])
        g["e"] = pick("e", [2, 4])
        g["w"] = pick("w", [16, 32, 64])
        g["depth"] = pick("depth", [1, 2])
        g["sn"] = pick("sn", [0, 1])
        g["act"] = pick("act", ["relu", "selu"])
        g["members"] = pick("members", [2, 4, 8]) if g["family"] == "tabm" else 1
    elif g["family"] == "node":
        g["T"] = pick("T", [8, 16, 32, 64])
        g["D"] = pick("D", [3, 4, 5, 6])
    elif g["family"] == "leafnet":
        g["trees"] = pick("trees", [8, 16, 32])
        g["leaves"] = pick("leaves", [8, 16])
        g["le"] = pick("le", [4, 8])
        g["lw"] = pick("lw", [16, 32])
    # Optuna needs one fixed choice set per parameter name, so ensembles get their own head gene
    g["head"] = pick("head_m", ["nlog", "energy", "maha", "min"]) if g.get("members", 1) > 1 \
        else pick("head", ["nlog", "energy", "maha"])
    g["kd"] = pick("kd", [0.0, 0.5, 0.9])
    g["ls"] = pick("ls", [0.0, 0.1])
    g["focal"] = pick("focal", [0.0, 2.0])
    g["lr"] = pick("lr", [1e-3, 3e-3])
    return g


def build(g, z, seed):
    Xf, yf, k = z["X_fit"], z["y_fit"], z["k"]
    if g["family"] in ("mlp", "tabm"):
        enc, din, _ = fx.encoder(g["enc"], Xf, yf, g["e"], seed)
        return fx.Body(enc, din, k, g["w"], g["depth"], g["members"], bool(g["sn"]), g["act"])
    if g["family"] == "node":
        return fx.NODE(Xf.shape[1], k, g["T"], g["D"])
    return fx.LeafNet(Xf, yf, k, g["trees"], g["leaves"], g["le"], g["lw"], seed)


def run_pair(g, z, device, seed, epochs):
    m = build(g, z, seed)
    m = fx.train(m, z["X_fit"], z["y_fit"], z["k"], device, seed, epochs, g["lr"], soft=z["teacher"], kd=g["kd"],
                 ls=g["ls"], focal=g["focal"])
    h = fx.Head(g["head"], z["bi"]).fit(m, z["X_fit"], z["y_fit"], device)
    s = {k: h.score(m, z[f"X_{k}"], device) for k in ("cal", "pseudo", "es")}
    return m, h, s


def boss_time(z):
    import xgboost as xgb
    b = xgb.XGBClassifier()
    b.load_model(z["dir"] / "boss_xgb.json")
    b.set_params(n_jobs=1)
    return fx.time_per_row(lambda X: b.predict_proba(X), z["X_cal"][:4096])


def evaluate(g, pairs, device, seeds=(0,), epochs=15, keep_scores=False):
    fit, unk, kn, scores = [], [], [], {}
    size = ratio = None
    for z in pairs:
        for sd in seeds:
            m, h, s = run_pair(g, z, device, sd, epochs)
            f, u, k_ = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], z["bi"])
            fit.append(f), unk.append(u), kn.append(k_)
            if keep_scores:
                scores[f"{z['ds']}/{z['name']}/s{sd}"] = s
            if size is None:
                mc = copy.deepcopy(m).cpu()
                size = fx.size_bytes(mc)
                ratio = fx.time_per_row(lambda X: h.score(mc, X, "cpu"), z["X_cal"][:4096]) / boss_time(z)
    return {"fitness": float(np.mean(fit)), "unk": float(np.mean(unk)), "known": float(np.mean(kn)),
            "size": int(size), "ratio": float(ratio)}, scores


def boss_fitness(pairs):
    out = {}
    for b in ("rf", "xgb"):
        r = [arena.fitness(z[f"boss_{b}_cal"], z[f"boss_{b}_pseudo"], z[f"boss_{b}_es"], z["y_es"], z["bi"]) for z in pairs]
        out[b] = {"fitness": float(np.mean([x[0] for x in r])), "unk": float(np.mean([x[1] for x in r])),
                  "known": float(np.mean([x[2] for x in r]))}
    return out


def study(r, storage, roster):
    import optuna
    sampler = optuna.samplers.NSGAIISampler(population_size=roster.get("population", 32), seed=r)
    return optuna.create_study(study_name=f"round{r}", storage=storage, load_if_exists=True, sampler=sampler,
                               directions=["maximize", "minimize", "minimize"])


def fight(a):
    import optuna
    import torch
    torch.set_num_threads(a.threads)
    rd = OUT / f"round{a.round}"
    roster = json.loads((rd / "roster.json").read_text())
    pairs = load_pairs(a.arena, cheap=True)
    st = study(a.round, a.storage, roster)
    if len(st.trials) == 0:
        for gnm in roster.get("seeds", []):
            st.enqueue_trial(gnm, skip_if_exists=True)

    def objective(trial):
        g = suggest(trial, roster)
        try:
            res, _ = evaluate(g, pairs, a.device)
        except Exception as e:  # a genome that cannot train is a dead fighter, not a crashed bowl
            trial.set_user_attr("error", repr(e)[:300])
            trial.set_user_attr("c", [1.0, 1.0])
            trial.set_constraint("size", 1.0), trial.set_constraint("ratio", 1.0)
            return 0.0, 1e9, 1e9
        trial.set_user_attr("res", res)
        trial.set_user_attr("genome", g)
        trial.set_user_attr("c", [res["size"] - SIZE_MAX, res["ratio"] - RATIO_MAX])
        trial.set_constraint("size", res["size"] - SIZE_MAX), trial.set_constraint("ratio", res["ratio"] - RATIO_MAX)
        return res["fitness"], res["size"], res["ratio"]

    left = a.trials - len([t for t in st.trials if t.state.name == "COMPLETE"])
    if left > 0:
        st.optimize(objective, n_trials=left, n_jobs=a.jobs, gc_after_trial=True)
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE and "res" in t.user_attrs]
    feas = [t for t in done if max(t.user_attrs["c"]) <= 0]
    print(f"round {a.round}: {len(done)} fighters, {len(feas)} within limits")


def death(a):
    import optuna
    import torch
    torch.set_num_threads(a.threads)
    rd = OUT / f"round{a.round}"
    st = study(a.round, a.storage, json.loads((rd / "roster.json").read_text()))
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE and "res" in t.user_attrs
            and max(t.user_attrs["c"]) <= 0]
    front = {t.number for t in st.best_trials}
    finalists = sorted([t for t in done if t.number in front], key=lambda t: -t.values[0])[:10]
    if len(finalists) < 3:  # thin front: top feasible by fitness
        finalists = sorted(done, key=lambda t: -t.values[0])[:10]
    if not finalists:
        raise SystemExit(f"round {a.round}: no fighter within the size / speed limits")
    pairs = load_pairs(a.arena, cheap=False)
    rows, best, best_scores = [], None, None
    for t in finalists:
        g = t.user_attrs["genome"]
        res, sc = evaluate(g, pairs, a.device, seeds=(1, 2), epochs=30, keep_scores=True)
        rows.append({"trial": t.number, **g, **res})
        print(rows[-1], flush=True)
        if best is None or res["fitness"] > best["fitness"]:
            best, best_scores = rows[-1], sc
    bosses = boss_fitness(pairs)
    np.savez_compressed(rd / "champion_scores.npz", **{f"{k}/{r}": v for k, s in best_scores.items() for r, v in s.items()})
    (rd / "death.json").write_text(json.dumps({"finalists": rows, "bosses": bosses, "champion": best,
                                               "pairs": [f"{z['ds']}/{z['name']}" for z in pairs]}, indent=1))
    print("bosses", bosses)
    print("CHAMPION", best)


def autopsy(a):
    """Per pair: champion vs boss XGB at the matched 1% threshold; depth-3 tree on boss-caught-champion-missed rows."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    from roc import roc_detect
    rd = OUT / f"round{a.round}"
    D = json.loads((rd / "death.json").read_text())
    S = np.load(rd / "champion_scores.npz")
    L = [f"# Round {a.round} autopsy\n", f"Champion: {D['champion']}\n", f"Bosses: {D['bosses']}\n"]
    for key in D["pairs"]:
        ds, name = key.split("/")
        z = np.load(a.arena / ds / f"{name}.npz")
        meta = json.loads((a.arena / ds / f"{name}.json").read_text())
        c = {k: S[f"{key}/s1/{k}"] for k in ("cal", "pseudo")}
        nc = len(c["cal"])
        isb = np.r_[np.ones(nc, bool), np.zeros(len(c["pseudo"]), bool)]
        wc = roc_detect(np.r_[c["cal"], c["pseudo"]], isb)[nc:]
        wb = roc_detect(np.r_[z["boss_xgb_cal"], z["boss_xgb_pseudo"]], isb)[nc:]
        missed = (wb >= 0.5) & (wc < 0.5)
        L.append(f"\n## {key} (R0={meta['R0']}, unknown={meta['j']}): champion TPR {wc.mean():.3f}, boss XGB "
                 f"{wb.mean():.3f}, boss-only {missed.mean():.3f}, champion-only {((wc >= 0.5) & (wb < 0.5)).mean():.3f}\n")
        if missed.sum() >= 20 and (~missed).sum() >= 20:
            t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(z["X_pseudo"], missed)
            L.append("```\n" + export_text(t, feature_names=meta["feats"] + [f"f{i}" for i in range(len(meta["feats"]),
                                                                                                   z["X_pseudo"].shape[1])])
                     + "```\n")
    (rd / "autopsy.md").write_text("".join(L))
    print("".join(L))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fight", "death", "autopsy"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--trials", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arena", type=Path, default=ARENA)
    ap.add_argument("--out", type=Path, default=OUT)  # noqa: F823
    ap.add_argument("--storage", default=None)
    a = ap.parse_args()
    OUT = a.out
    a.storage = a.storage or f"sqlite:///{OUT / 'bowl.db'}"
    if a.device == "auto":
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    {"fight": fight, "death": death, "autopsy": autopsy}[a.cmd](a)


if __name__ == "__main__":
    main()
