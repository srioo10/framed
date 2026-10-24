import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids import windows

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_bank", ROOT / "scripts" / "score_bank.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


def test_cap_is_proportional_and_deterministic():
    groups = np.repeat(np.arange(4), [100, 300, 600, 1000])
    idx = np.arange(len(groups))
    a = sb.cap_by_block(idx, groups, 200, np.random.default_rng(0))
    b = sb.cap_by_block(idx, groups, 200, np.random.default_rng(0))
    assert len(a) == 200 and np.array_equal(a, b)
    share = np.bincount(groups[a], minlength=4) / 200
    assert np.allclose(share, [0.05, 0.15, 0.30, 0.50], atol=0.011)
    assert not np.array_equal(a, idx[:200])  # never first-N


def test_group_split_keeps_groups_whole():
    groups = np.repeat(np.arange(50), 20)
    idx = np.arange(len(groups))
    r, c = sb.split_by_group(idx, groups, np.random.default_rng(1))
    assert not set(groups[r]) & set(groups[c]) and len(r) + len(c) == len(idx)
    fit, es = sb.split_by_group(idx, groups, np.random.default_rng(1), share=0.9)
    assert 0.05 < len(es) / len(idx) < 0.2 and not set(groups[fit]) & set(groups[es])


def test_role_removal_per_role_and_direction():
    # outer fold k: everything before test removes k; test uses all flows
    for role in ("fit", "es", "bsample", "rank", "cal"):
        assert sb.role_removal(("k",), role) == ("k",)
    assert sb.role_removal(("k",), "test_benign") == ()
    # pair {j, k}: fit/rank/cal remove both; pseudo rows for fold k keep j, for fold j keep k (mirror)
    P = ("j", "k")
    assert sb.role_removal(P, "cal") == P
    assert sb.role_removal(P, "pseudo", ("k",)) == ("k",)
    assert sb.role_removal(P, "pseudo", ("j",)) == ("j",)
    with pytest.raises(AssertionError):
        sb.role_removal(P, "pseudo", P)


def _flows(times, fam, src=None):
    n = len(times)
    raw = pd.DataFrame({"stime": np.asarray(times) + 1_421_928_000, "family": fam,
                        "srcip": src if src is not None else ["a"] * n, "dstip": ["b"] * n,
                        "sport": ["1"] * n, "dsport": ["2"] * n, "service": ["-"] * n})
    return windows.Flows(raw)


def test_window_is_whole_seconds_and_stays_in_block():
    # 150 flows at second 10, 60 at second 11, 60 at second 12 (block starts at second 0, 600 s long)
    t = np.r_[np.full(150, 10), np.full(60, 11), np.full(60, 12)]
    f = _flows(t, ["Benign"] * len(t))
    pop = np.ones(len(t), bool)
    _, t0 = f.window_start(pop)
    assert (t0[t == 12] == 11 - t.min()).all()  # Flows stores seconds relative to the first flow       # seconds 11..12 hold 120 >= 100 flows
    assert (t0[t == 11] == 10 - t.min()).all()       # second 11 alone holds 60 < 100, back to 10
    feats = f.features(pop, np.flatnonzero(t == 12)[:1])
    assert feats[0, 0] == 120              # whole current second included, not just earlier flows
    # a flow just after a block boundary never looks into the previous block
    t2 = np.r_[np.full(200, 590), np.full(5, 600)]
    f2 = _flows(t2, ["Benign"] * len(t2))
    _, t0b = f2.window_start(np.ones(len(t2), bool))
    assert (t0b[t2 == 600] == 600 - t2.min()).all()
    assert (f2.features(np.ones(len(t2), bool), np.flatnonzero(t2 == 600))[:, 0] == 5).all()


def test_removal_set_changes_counts_only_through_population():
    t = np.r_[np.full(80, 5), np.full(80, 6)]
    fam = np.r_[["Benign"] * 80, ["DoS"] * 40, ["Benign"] * 40]
    f = _flows(t, fam)
    q = np.flatnonzero((t == 6) & (fam == "Benign"))[:1]
    all_flows = f.features(np.ones(len(t), bool), q)[0, 0]
    no_dos = f.features(fam != "DoS", q)[0, 0]
    assert all_flows == 160 and no_dos == 120


BANK = os.environ.get("NIDS_BANK_CHECK")


@pytest.mark.skipif(not BANK, reason="set NIDS_BANK_CHECK to a bank run dir")
def test_manifest_no_leak():
    """rank/cal never in any fit or early-stop set; capped rows identical across configs; removal respected."""
    bank = Path(BANK)
    ids = {}
    for z in bank.glob("ids_*.npz"):
        ids.update(dict(np.load(z)))
    held = np.union1d(ids["rank"], ids["cal"])
    caps = np.concatenate([v for k, v in ids.items() if k.startswith("cap_")])
    assert not np.intersect1d(held, caps).size and not np.intersect1d(held, ids["bsample"]).size
    assert not np.intersect1d(ids["rank"], ids["cal"]).size
    for k, v in ids.items():
        if k.startswith("es_"):
            assert np.isin(v, caps).all()
    for m in bank.glob("manifest_*.json"):
        for name, info in json.loads(m.read_text())["configs"].items():
            part = pd.read_parquet(bank / f"config={name}" / "part.parquet", columns=["role", "family", "fine", "served"])
            for fold, j in info["served"]:
                if j is None:  # outer: unknown test rows are exactly the held-out families
                    unk = part[part.role == "test_unknown"]
                    assert (unk.family.isin(info["removal"]) | unk.fine.isin(info["removal"])).all()
                    continue
                ps = part[(part.role == "pseudo") & (part.get("served", fold) == fold)] if "served" in part else None
                assert ps is None or ((ps.family == j) | (ps.fine == j)).all()
            for role in ("rank", "cal"):
                assert (part[part.role == role].family == "Benign").all()


def test_mcde_onehot_prep():
    from nids import mcde
    df = pd.DataFrame({"a": [1.0, 2, 3, np.nan, 5], "p": pd.Categorical(["tcp", "udp", "tcp", None, "tcp"])})
    X = mcde.RobustPrep(onehot=True).fit(df).transform(pd.concat(
        [df, pd.DataFrame({"a": [1.0], "p": pd.Categorical(["icmp"])})], ignore_index=True))
    assert X.shape == (6, 3)
    assert (X[3, 1:] == [0.5, 0]).all()   # missing -> mode (tcp), block weight 0.5
    assert (X[5, 1:] == 0).all()          # unseen category -> zeros
    assert X[3, 0] == 0                   # median impute, centred on the median
