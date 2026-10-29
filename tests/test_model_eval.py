import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from model_eval import boot_diff, flow_threshold, macro  # noqa: E402


def test_flow_threshold_budget():
    s = np.random.default_rng(0).normal(size=10_000)
    m = np.ones_like(s, bool)
    assert (s[m] > flow_threshold(s, m)).mean() <= 0.01


def table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fold in ("a", "b"):
        for h in range(30):
            n = int(rng.integers(5, 50))
            for d in ("x", "y"):
                rows.append({"seed": 1, "fold": fold, "host": h, "detector": d, "flows": n,
                             "flows_det": int(rng.integers(0, n + 1)), "days": 1, "hit_P": 1, "hit_A": 0})
    return pd.DataFrame(rows)


def test_boot_identical_is_zero():
    H = table()
    H.loc[H.detector == "y", "flows_det"] = H.loc[H.detector == "x", "flows_det"].to_numpy()
    lo, hi = boot_diff(H, "x", "y", "flows_det", "flows", reps=200)
    assert lo == hi == 0


def test_macro_and_ci_cover_difference():
    H = table()
    d = macro(H[H.detector == "x"], "flows_det", "flows") - macro(H[H.detector == "y"], "flows_det", "flows")
    lo, hi = boot_diff(H, "x", "y", "flows_det", "flows", reps=500)
    assert lo <= d <= hi


def test_roc_detect_ties_give_exact_fpr():
    from model_eval import roc_detect
    s = np.r_[np.zeros(950), np.ones(50)]  # 5% of benign tied at the top score
    b = np.ones_like(s, bool)
    assert abs(roc_detect(s, b).mean() - 0.01) < 1e-12
