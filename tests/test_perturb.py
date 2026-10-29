import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import adversary, hostctx  # noqa: E402


def _raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    fam = np.where(np.arange(n) % 4 == 0, "Bot", "Benign")
    src = np.where(fam == "Bot", "10.0.0.66", rng.choice(["10.0.0.1", "10.0.0.2", "10.0.0.3"], n))
    return pd.DataFrame({"FLOW_START_MILLISECONDS": np.sort(rng.integers(0, 3_600_000, n)), "IPV4_SRC_ADDR": src,
                         "IPV4_DST_ADDR": rng.choice(["1.1.1.1", "2.2.2.2"], n), "L4_DST_PORT": rng.integers(1, 9, n),
                         "OUT_PKTS": rng.integers(0, 3, n), "IN_BYTES": rng.integers(40, 900, n), "family": fam})


def test_dilate_one_is_identity_and_k_moves_only_attacks():
    raw, kept = _raw(), np.arange(400)
    clean = hostctx.HostFlows(raw)
    clean.keep = kept
    same = adversary.Perturbed(raw, "dilate:1", 0, kept)
    assert np.array_equal(same.host.removal(()), clean.removal(()))
    p = adversary.Perturbed(raw, "dilate:10", 0, kept)
    t0, t1 = raw.FLOW_START_MILLISECONDS.to_numpy(), p.t_kept
    att = raw.family.to_numpy() == "Bot"
    assert np.array_equal(t0[~att], t1[~att])
    assert (np.diff(t1[att]) >= 0).all() and t1[att].max() - t1[att].min() == 10 * (t0[att].max() - t0[att].min())


def test_cover_rows_come_from_the_attacker_inside_its_span():
    raw, kept = _raw(), np.arange(400)
    p = adversary.Perturbed(raw, "cover:2", 0, kept)
    att = raw.family.to_numpy() == "Bot"
    assert len(p.cover_row) == 2 * att.sum()
    assert (p.cover_src == "10.0.0.66").all() and (p.cover_fam == "Bot").all()
    assert (raw.family.to_numpy()[p.cover_row] == "Benign").all()
    t = raw.FLOW_START_MILLISECONDS.to_numpy()[att]
    assert p.cover_t.min() >= t.min() - adversary.PAD_MS and p.cover_t.max() <= t.max() + adversary.PAD_MS
    assert p.host.removal(()).shape[0] == 400 + len(p.cover_row)
