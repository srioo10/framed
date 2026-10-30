import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import arena  # noqa: E402
import bowl  # noqa: E402
import fighters as fx  # noqa: E402


def fake_arena(root, n=1500, d=6, k=3, cert=False):
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
        meta = {"R0": ["a"], "j": "b", "known": ["Benign", "x", "y"], "feats": [f"c{i}" for i in range(d // 2)]}
        if cert:  # PREREG §9 CERT: the last 20 columns are host context, identity Prep scale
            from nids import hostctx
            meta.update(feats=[f"c{i}" for i in range(d - 20)], mu=[0.0] * d, sd=[1.0] * d,
                        cols=[f"c{i}" for i in range(d - 20)] + hostctx.COLS)
        (root / ds / "pair_0.json").write_text(json.dumps(meta))


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
    hy = fx.Hybrid(fx.Body(torch.nn.Identity(), 10, 3, 8, 1, members=4), fx.NODE(10, 3, T=4, D=3))
    assert hy(torch.randn(5, 10))[0].shape == (5, 4, 3) and hy.ops() == hy.body.ops() + 4 * (3 + 3)


def test_binmap_and_periodic_shapes():
    X = np.random.default_rng(0).normal(size=(200, 4)).astype(np.float32)
    enc, din, _ = fx.encoder("tree", X, (X[:, 0] > 0).astype(int), 2, 0)
    assert enc(torch.from_numpy(X)).shape == (200, din) == (200, 8)
    p = fx.Periodic(4, 3)
    assert p(torch.from_numpy(X)).shape == (200, 12)


def test_mini_bowl_end_to_end(tmp_path, monkeypatch):
    fake_arena(tmp_path / "arena")
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
                           storage=f"sqlite:///{tmp_path / 'b.db'}", part=0, parts=1, merge=False)
    bowl.fight(a)
    bowl.death(a)
    single = json.loads((rd / "death.json").read_text())
    for i in (0, 1):  # the same death match split over 2 workers, then merged
        bowl.death(argparse.Namespace(**{**vars(a), "part": i, "parts": 2}))
    bowl.death(argparse.Namespace(**{**vars(a), "parts": 2, "merge": True}))
    D = json.loads((rd / "death.json").read_text())
    assert [r["trial"] for r in D["finalists"]] == [r["trial"] for r in single["finalists"]]
    nt = lambda r: {k: v for k, v in r.items() if k != "ratio"}  # noqa: E731  ratio is wall-clock timing
    assert nt(D["champion"]) == nt(single["champion"]) and (rd / "champion_scores.npz").exists()
    assert 0 <= D["champion"]["fitness"] <= 1 and set(D["bosses"]) == {"rf", "xgb"}
    import ultra_bosses as ub
    big = ub.models
    monkeypatch.setattr(ub, "models", lambda gpu: {k: m.set_params(**{"iterations" if k == "cat" else "n_estimators": 10})
                                                   for k, m in big(gpu).items()})
    ub.run(tmp_path / "arena", tmp_path / "arena", gpu=False)
    bowl.autopsy(a)
    assert set(json.loads((rd / "death.json").read_text())["bosses"]) == {"rf", "xgb", "cat", "lgbm", "xgbx", "gang"}
    assert (rd / "autopsy.md").exists()
    r2 = tmp_path / "bowl" / "round2"
    r2.mkdir()
    r2.joinpath("roster.json").write_text(json.dumps({"families": ["tabm", "tabm_node"], "population": 4,
                                                      "genes": ["teacher", "syn"], "powerups": list(bowl.POWERUPS)}))
    a.round = 2
    bowl.ablate(a)
    A = json.loads((r2 / "ablation.json").read_text())
    assert set(A["kept"]) | set(A["dropped"]) == set(bowl.POWERUPS) and "all" in A["variants"]
    roster = bowl.apply_ablation(json.loads(r2.joinpath("roster.json").read_text()), r2)
    assert ("tabm_node" in roster["families"]) == ("node_branch" in A["kept"])
    A["dropped"], A["kept"] = list(bowl.POWERUPS), []  # worst case: every power-up pinned off, seeds still enqueue
    (r2 / "ablation.json").write_text(json.dumps(A))
    roster = json.loads(r2.joinpath("roster.json").read_text())
    roster["seeds"] = [{"family": "tabm_node", "enc": "raw", "e": 2, "w": 16, "depth": 1, "sn": 0, "act": "relu",
                        "members": 2, "head_m": "nlog", "kd": 0.0, "ls": 0.0, "focal": 0.0, "lr": 3e-3,
                        "teacher": "rf", "syn": 0.5, "T": 8, "D": 3},
                       {"family": "tabm", "enc": "raw", "e": 2, "w": 16, "depth": 1, "sn": 0, "act": "relu",
                        "members": 2, "head_m": "nlog", "kd": 0.0, "ls": 0.0, "focal": 0.0, "lr": 3e-3,
                        "teacher": "rf", "syn": 0.5}]
    r2.joinpath("roster.json").write_text(json.dumps(roster))
    bowl.fight(a)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_genome_space_is_static_for_optuna(tmp_path, r):
    import optuna
    roster = json.loads((Path(__file__).resolve().parents[1] / f"results/bowl/round{r}/roster.json").read_text())
    st = optuna.create_study(storage=f"sqlite:///{tmp_path / 's.db'}", sampler=optuna.samplers.RandomSampler(0))
    for g in roster["seeds"]:
        st.enqueue_trial(g)
    st.optimize(lambda t: float(len(bowl.suggest(t, roster))), n_trials=80)  # raises on a dynamic choice set
    fams = {t.params["family"] for t in st.trials}
    assert fams == set(roster["families"])
    assert all(set(g) <= set(st.trials[i].params) for i, g in enumerate(roster["seeds"]))


@pytest.mark.parametrize("kind,members,ens,head", [("raw", 1, 1, "nlog"), ("quantile", 4, 1, "min"),
                                                   ("tree", 2, 3, "nlog"), ("raw", 4, 3, "min")])
def test_interval_bound_is_sound(kind, members, ens, head):
    """PREREG §9 CERT Theorem 2: the IBP score bound covers the exact score at random points of every box, and equals
    it on a zero-width box."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1)
    clip = (np.r_[[-np.inf] * 3, [-2.0] * 3], np.r_[[np.inf] * 3, [2.0] * 3])

    def body(s):
        torch.manual_seed(s)
        enc, din, _ = fx.encoder(kind, X, y, 4, s)
        return fx.Body(enc, din, 3, 16, 2, members, act="selu" if s % 2 else "relu", clip=clip)
    m = (body(0) if ens == 1 else fx.Ensemble([body(i) for i in range(ens)])).eval()
    lo, hi = X - rng.uniform(0, 0.5, X.shape), X + rng.uniform(0, 0.5, X.shape)
    hi[:, 5] = np.inf  # an unbounded host-context column (mean bytes) is tamed by the clip
    up = fx.score_upper(m, lo, hi, 0, head)
    h = fx.Head(head, 0)
    for _ in range(20):
        P = lo + rng.uniform(size=X.shape) * (np.minimum(hi, 10) - lo)
        assert (h.score(m, P.astype(np.float32), "cpu") <= up + 1e-4).all()
    assert np.allclose(fx.score_upper(m, X, X, 0, head), h.score(m, X, "cpu"), atol=1e-4)


def test_cert_training_lowers_the_bound():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6)).astype(np.float32)
    y = ((X[:, 0] > 0.5) & (X[:, 3] > 0)).astype(int)
    lo, hi = X - 0.3, X + 0.3
    ben = y == 0
    ups = []
    for cert in (0.0, 1.0):
        torch.manual_seed(0)
        m = fx.Body(torch.nn.Identity(), 6, 2, 16, 2)
        m = fx.train(m, X, y, 2, "cpu", epochs=8, Lc=lo[ben], Hc=hi[ben], cert=cert, bi=0)
        ups.append(np.median(fx.score_upper(m, lo[ben], hi[ben], 0, "nlog")))
    assert ups[1] < ups[0]


def test_cert_round_constrains_certified_fpr(tmp_path, monkeypatch):
    fake_arena(tmp_path / "arena", d=24, cert=True)
    monkeypatch.setattr(bowl, "OUT", tmp_path / "bowl")
    monkeypatch.setattr(bowl, "RATIO_MAX", 1e6)
    rd = tmp_path / "bowl" / "round4"
    rd.mkdir(parents=True)
    rd.joinpath("roster.json").write_text(json.dumps({
        "families": ["mlp", "tabm"], "population": 4, "cert_mode": True, "enc_opts": ["raw", "quantile"],
        "head_opts": ["nlog"], "head_m_opts": ["nlog", "min"], "powerups": [],
        "seeds": [{"family": "mlp", "enc": "raw", "e": 4, "w": 16, "depth": 1, "sn": 0, "act": "relu", "head": "nlog",
                   "kd": 0.5, "ls": 0.0, "focal": 0.0, "lr": 3e-3, "cert": 1.0}]}))
    import argparse
    a = argparse.Namespace(round=4, trials=3, jobs=1, threads=1, device="cpu", arena=tmp_path / "arena",
                           storage=f"sqlite:///{tmp_path / 'b.db'}", part=0, parts=1, merge=False)
    bowl.ablate(a)  # no power-ups: nothing to do
    assert not (rd / "ablation.json").exists()
    monkeypatch.setattr(bowl, "CERT_MAX", 1.01)  # every toy fighter feasible, so the death match runs
    monkeypatch.setattr(bowl, "CERT_MS", (1, 10))
    bowl.fight(a)
    import optuna
    st = optuna.load_study(study_name="round4", storage=a.storage)
    t = [t for t in st.trials if "res" in t.user_attrs]
    assert t and all(len(x.user_attrs["c"]) == 3 and 0 <= x.user_attrs["res"]["cert"]["10"] <= 1 for x in t)
    bowl.death(a)
    D = json.loads((rd / "death.json").read_text())
    c = D["champion"]["cert"]
    assert set(c) == {"1", "10", "dst1", "dst10"} and c["1"] <= c["10"] and c["dst10"] <= c["10"]
