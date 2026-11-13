"""DAMPED amendment 1: add the RF out-of-fold teacher to feature-only pairs (as conf_pairs.py does), in place.
Usage: python scripts/conf_teacher.py SRC_DIR DST_DIR  (pairs under SRC_DIR/{ds}/pair_i.npz; json copied alongside)"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import models  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for f in sorted(src.glob("*/pair_*.npz")):
    z = dict(np.load(f))
    assert "teacher" not in z
    z["teacher"] = models.oof_proba(z["X_fit"], z["y_fit"], 1).astype(np.float32)
    o = dst / f.parent.name
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / f.name, **z)
    shutil.copy(f.with_suffix(".json"), o / f.with_suffix(".json").name)
    print("teacher", f.parent.name, f.name, z["teacher"].shape, flush=True)
