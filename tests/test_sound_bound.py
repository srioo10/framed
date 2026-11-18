"""PREREG SOUND guard: the rigorous float64 bound is below the real score at random points of each box, below the
plain IBP bound, and within 1e-7 of it, i.e. far inside the old flat 1e-6 margin. Run: PYTHONPATH=src python tests/test_sound_bound.py"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fighters as fx  # noqa: E402


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d, k, bi = 27, 5, 2
    for members in (8, 1):
        m = fx.Body(nn.Identity(), d, k, 32, 2, members=members, clip=(np.full(d, -6.0), np.full(d, 6.0))).eval()
        x = rng.normal(size=(400, d))
        w = np.abs(rng.normal(size=(400, d))) * rng.choice([0, 0.05, 1.0], size=(400, 1))
        lo, hi = x - w, x + w
        sound = fx.score_lower_sound(m, lo, hi, bi, "nlog")
        plain = fx.score_lower(m, lo, hi, bi, "nlog") + 1e-6  # undo its flat margin
        assert np.all(sound <= plain + 1e-12), "sound bound above the plain IBP bound"
        assert np.max(plain - sound) < 1e-7, f"rounding terms too large: {np.max(plain - sound)}"
        with torch.no_grad():
            m64 = m.double()
            for _ in range(20):
                p = lo + (hi - lo) * rng.random(lo.shape)
                z = m64(torch.as_tensor(p))[0]
                z = z if z.dim() == 3 else z[:, None]
                lpb = torch.log_softmax(z, -1)[..., bi]
                s = -(torch.logsumexp(lpb, 1) - np.log(lpb.shape[1]))
                assert np.all(sound <= s.numpy() + 1e-12), "sound bound above a real score in the box"
            m.float()
        print(f"members={members}: ok, max(plain - sound) = {np.max(plain - sound):.2e}")
    print("SOUND BOUND OK")


if __name__ == "__main__":
    main()
