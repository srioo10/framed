"""Tuned bosses (PREREG §9 BOWL tuned bosses): the tree bosses get a search and power-ups like the fighters.

tune:  one Optuna TPE study, 256 trials (+256 once if the best rose by > 0.005 in the last quarter), cheap arena,
       same fitness as the fight. Families rf / xgb / lgbm with their hyperparameters; tree power-ups syn (bowl's
       cross-context synthetic rows, RF argmax label, sample weight syn) and balance (inverse class-frequency weights).
death: top 3 per family on the full arena, seeds 1 and 2; the best is the tuned boss (results/bowl/tuned_boss.json).
No size / speed limit; both reported.
Usage: .venv/bin/python scripts/boss_bowl.py tune [--trials 256]
       .venv/bin/python scripts/boss_bowl.py death
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena  # noqa: E402
import bowl  # noqa: E402
import fighters as fx  # noqa: E402

DB = f"sqlite:///{bowl.OUT / 'bosses.db'}"


def suggest(t):
    fam = t.suggest_categorical("family", ["rf", "xgb", "lgbm"])
    g = {"family": fam, "syn": t.suggest_categorical("syn", [0.0, 0.5, 1.0]),
         "balance": t.suggest_categorical("balance", [0, 1])}
    if fam == "rf":
        g |= {"n": t.suggest_categorical("rf_n", [100, 300, 600]),
              "depth": t.suggest_categorical("rf_depth", [0, 12, 20, 30]),
              "leaf": t.suggest_categorical("rf_leaf", [1, 2, 5, 10]),
              "feat": t.suggest_categorical("rf_feat", ["sqrt", 0.3, 0.5]),
              "rows": t.suggest_categorical("rf_rows", [0.0, 0.5])}
    else:
        g |= {"n": t.suggest_int(f"{fam}_n", 100, 1000, step=100),
              "lr": t.suggest_float(f"{fam}_lr", 0.02, 0.3, log=True),
              "sub": t.suggest_float(f"{fam}_sub", 0.5, 1.0),
              "col": t.suggest_float(f"{fam}_col", 0.5, 1.0),
              "l2": t.suggest_float(f"{fam}_l2", 0.1, 10, log=True)}
        g |= ({"depth": t.suggest_int("xgb_depth", 3, 12), "mcw": t.suggest_float("xgb_mcw", 1, 20, log=True)}
              if fam == "xgb" else
              {"leaves": t.suggest_int("lgbm_leaves", 15, 255, log=True), "mcs": t.suggest_int("lgbm_mcs", 5, 100, log=True)})
    return g


def model(g, seed):
    if g["family"] == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(g["n"], max_depth=g["depth"] or None, min_samples_leaf=g["leaf"],
                                      max_features=g["feat"], max_samples=g["rows"] or None, n_jobs=-1,
                                      random_state=seed)
    if g["family"] == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=g["n"], max_depth=g["depth"], learning_rate=g["lr"], subsample=g["sub"],
                             colsample_bytree=g["col"], reg_lambda=g["l2"], min_child_weight=g["mcw"],
                             tree_method="hist", n_jobs=-1, random_state=seed)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=g["n"], num_leaves=g["leaves"], learning_rate=g["lr"], subsample=g["sub"],
                          subsample_freq=1, colsample_bytree=g["col"], reg_lambda=g["l2"], min_child_samples=g["mcs"],
                          n_jobs=-1, random_state=seed, verbose=-1)


def fit_pair(g, z, seed):
    """Fitted model on the pair's fit rows (+ synthetic rows when syn > 0); scores -log P(benign) on es/pseudo/cal."""
    X, y = z["X_fit"], z["y_fit"]
    w = np.ones(len(y))
    if g["balance"]:
        w = len(y) / (z["k"] * np.bincount(y, minlength=z["k"])[y])
    if g["syn"] > 0:
        _, Xs, Ps = bowl.rf_teacher(z)
        X, y, w = np.concatenate([X, Xs]), np.concatenate([y, Ps.argmax(1)]), np.concatenate([w, np.full(len(Xs), g["syn"])])
    m = model(g, seed)
    m.fit(X, y, sample_weight=w)
    col = list(m.classes_).index(z["bi"])  # a class missing from the fit rows would shift columns
    nl = lambda X: -np.log(np.maximum(m.predict_proba(X)[:, col], 1e-45))  # noqa: E731
    return m, {k: nl(z[f"X_{k}"]) for k in ("es", "pseudo", "cal")}


def evaluate(g, pairs, seeds=(0,)):
    fit, unk, kn = [], [], []
    for z in pairs:
        for sd in seeds:
            m, s = fit_pair(g, z, sd)
            f, u, k_ = arena.fitness(s["cal"], s["pseudo"], s["es"], z["y_es"], z["bi"])
            fit.append(f), unk.append(u), kn.append(k_)
    m.set_params(n_jobs=1)
    ratio = fx.time_per_row(lambda X: m.predict_proba(X), z["X_cal"][:4096]) / bowl.boss_time(z)
    return {"fitness": float(np.mean(fit)), "unk": float(np.mean(unk)), "known": float(np.mean(kn)),
            "ratio": float(ratio)}


def tune(a):
    import optuna
    pairs = bowl.load_pairs(a.arena, cheap=True)
    st = optuna.create_study(study_name="bosses", storage=DB, load_if_exists=True, direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0))

    def objective(t):
        g = suggest(t)
        res = evaluate(g, pairs)
        t.set_user_attr("genome", g), t.set_user_attr("res", res)
        print(t.number, g, {k: round(v, 4) for k, v in res.items()}, flush=True)
        return res["fitness"]

    def run(n):
        left = n - len([t for t in st.trials if t.state.name == "COMPLETE"])
        if left > 0:
            st.optimize(objective, n_trials=left, gc_after_trial=True)

    run(a.trials)
    v = [t.value for t in sorted(st.trials, key=lambda t: t.number) if t.state.name == "COMPLETE"]
    q = 3 * a.trials // 4
    if len(v) <= a.trials and max(v) - max(v[:q]) > 0.005:  # PREREG extension rule, once
        print("extension: best rose", max(v) - max(v[:q]), flush=True)
        run(2 * a.trials)
    print("best", st.best_trial.number, st.best_value, st.best_trial.user_attrs["genome"])


def death(a):
    import optuna
    st = optuna.load_study(study_name="bosses", storage=DB)
    done = [t for t in st.trials if t.state.name == "COMPLETE"]
    top = [t for f in ("rf", "xgb", "lgbm")
           for t in sorted([t for t in done if t.user_attrs["genome"]["family"] == f], key=lambda t: -t.value)[:3]]
    pairs = bowl.load_pairs(a.arena, cheap=False)
    rows = []
    for t in top:
        g = t.user_attrs["genome"]
        rows.append({"trial": t.number, "cheap": t.value, **g, **evaluate(g, pairs, seeds=(1, 2))})
        print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["fitness"])
    (bowl.OUT / "tuned_boss.json").write_text(json.dumps({"finalists": rows, "tuned_boss": best,
                                                          "trials": len(done)}, indent=1))
    print("TUNED BOSS", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "death"])
    ap.add_argument("--trials", type=int, default=256)
    ap.add_argument("--arena", type=Path, default=bowl.ARENA)
    a = ap.parse_args()
    {"tune": tune, "death": death}[a.cmd](a)
