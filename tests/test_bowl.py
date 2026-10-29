import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import arena  # noqa: E402
import bowl  # noqa: E402
import fighters as fx  # noqa: E402


def fake_arena(root, n=1500, d=6, k=3):
    import xgboost as xgb
    rng = np.random.default_rng(0)
    for ds in ("nf_ctu13", "nf_ugr16"):
        (root / ds).mkdir(parents=True)
        X = rng.normal(size=(n, d)).astype(np.float32)
        y = np.clip((X[:, 0] > 0).astype(int) + (X[:, 1] > 1), 0, k - 1)
        b = xgb.XGBClassifier(n_estimators=20, max_depth=3).fit(X, y)
        b.save_model(root / ds / "boss_xgb.json")
        cal = rng.normal(size=(400, d)).astype(np.float32) - 1
        ps = rng.normal(size=(300, d)).astype(np.float32) + 2
        bs = {f"boss_{m}_{r}": -np.log(np.maximum(b.predict_proba(v)[:, 0], 1e-45)).astype(np.float32)
              for m in ("rf", "xgb") for r, v in (("cal", cal), ("pseudo", ps), ("es", X[:200]))}
        np.savez(root / ds / "pair_0.npz", X_fit=X, y_fit=y, X_es=X[:200], y_es=y[:200], X_pseudo=ps, X_cal=cal,
                 teacher=b.predict_proba(X).astype(np.float32), bi=0, **bs)
        (root / ds / "pair_0.json").write_text(json.dumps({"R0": ["a"], "j": "b", "known": ["Benign", "x", "y"],
                                                           "feats": [f"c{i}" for i in range(d)]}))


def test_fitness_perfect_and_null():
    cal, ps = np.zeros(1000), np.ones(100)
    f, u, k = arena.fitness(cal, ps, np.ones(50), np.ones(50, int), 0)
    assert u == 1 and k == 1 and f == 1
    f0, u0, _ = arena.fitness(np.zeros(1000), np.zeros(100), np.zeros(50), np.ones(50, int), 0)
    assert abs(u0 - 0.01) < 1e-9  # all tied: exactly the 1% budget


def test_size_and_ops_formulas():
    m = fx.Body(torch.nn.Identity(), 10, 3, 8, 2)
    assert fx.size_bytes(m) == 10 * 8 + 8 + 8 * 8 + 8 + 8 * 3 + 3
    assert m.ops() == 10 * 8 + 8 * 8 + 8 * 3
    node = fx.NODE(10, 3, T=4, D=3)
    assert node.ops() == 4 * (3 + 3)
    assert node(torch.randn(5, 10))[0].shape == (5, 3)
    node.eval()
    assert node(torch.randn(5, 10))[0].shape == (5, 3)  # hard (deployed) form


def test_binmap_and_periodic_shapes():
    X = np.random.default_rng(0).normal(size=(200, 4)).astype(np.float32)
    enc, din, _ = fx.encoder("tree", X, (X[:, 0] > 0).astype(int), 2, 0)
    assert enc(torch.from_numpy(X)).shape == (200, din) == (200, 8)
    p = fx.Periodic(4, 3)
    assert p(torch.from_numpy(X)).shape == (200, 12)


def test_mini_bowl_end_to_end(tmp_path, monkeypatch):
    fake_arena(tmp_path / "arena")
    monkeypatch.setattr(bowl, "ARENA", tmp_path / "arena")
    monkeypatch.setattr(bowl, "OUT", tmp_path / "bowl")
    monkeypatch.setattr(bowl, "RATIO_MAX", 1e6)  # toy boss is tiny; the limit itself is tested via set_constraint
    rd = tmp_path / "bowl" / "round1"
    rd.mkdir(parents=True)
    rd.joinpath("roster.json").write_text(json.dumps({"families": ["mlp", "node"], "population": 4,
                                                      "seeds": [{"family": "mlp", "enc": "quantile", "e": 2, "w": 16,
                                                                 "depth": 1, "sn": 0, "act": "relu", "head": "nlog",
                                                                 "kd": 0.5, "ls": 0.0, "focal": 0.0, "lr": 3e-3}]}))
    import argparse
    a = argparse.Namespace(round=1, trials=4, jobs=1, threads=1, device="cpu", arena=tmp_path / "arena",
                           storage=f"sqlite:///{tmp_path / 'b.db'}")
    bowl.fight(a)
    bowl.death(a)
    D = json.loads((rd / "death.json").read_text())
    assert 0 <= D["champion"]["fitness"] <= 1 and set(D["bosses"]) == {"rf", "xgb"}
    bowl.autopsy(a)
    assert (rd / "autopsy.md").exists()


def test_genome_space_is_static_for_optuna(tmp_path):
    import optuna
    roster = json.loads((Path(__file__).resolve().parents[1] / "results/bowl/round1/roster.json").read_text())
    st = optuna.create_study(storage=f"sqlite:///{tmp_path / 's.db'}", sampler=optuna.samplers.RandomSampler(0))
    for g in roster["seeds"]:
        st.enqueue_trial(g)
    st.optimize(lambda t: float(len(bowl.suggest(t, roster))), n_trials=80)  # raises on a dynamic choice set
    fams = {t.params["family"] for t in st.trials}
    assert fams == set(roster["families"])
    assert all(set(g) <= set(st.trials[i].params) for i, g in enumerate(roster["seeds"]))
