"""PREREG §9 K18: PGD points stay inside the box and do at least as well as the box centre for the attacker."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fighters as fx  # noqa: E402


class Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.f = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))

    def forward(self, x):
        return self.f(x), None


def test_pgd_in_box_and_ascends():
    m, g = Lin(), torch.Generator().manual_seed(0)
    lo = torch.randn(64, 6)
    hi = lo + torch.rand(64, 6)
    hi[:, 0] = lo[:, 0]  # zero-width dimension must stay fixed
    hi[:, 1] = float("inf")  # the insertion box's mean-bytes bound is unbounded above
    for sign in (-1.0, 1.0):
        x = fx.pgd_point(m, lo, hi, 0, sign, gen=g)
        assert torch.isfinite(x).all() and (x >= lo - 1e-6).all() and (x <= hi + 1e-6).all()
        assert torch.equal(x[:, 0], lo[:, 0])
        obj = lambda v: sign * torch.log_softmax(m(v)[0], -1)[:, 0]  # noqa: E731
        assert (obj(x) >= obj((lo + hi.clamp(max=1e4)) / 2) - 1e-4).float().mean() > 0.9
