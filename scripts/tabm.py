"""TabM-mini on the bounded encoding (PREREG §9 TABM). K members share one MLP; member i scales the input by r_i
(init random ±1, trained) and has its own head; loss = mean member CE. Arms: tabm (1 - mean P(benign)), tabm_min
(1 - min member P(benign)), tabm_mi (mutual information across members), tabm_raw (no encoding).
Writes runs/bank/{ds}_host_cap40000_s{seed}_tabm/... via ple.main.
Usage: .venv/bin/python scripts/tabm.py nf_ctu13 --seed 1
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ple  # noqa: E402

K = 16


class TabMMini(nn.Module):
    def __init__(self, d, k, w=64, members=K, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.r = nn.Parameter(torch.randint(0, 2, (members, d), generator=g).float() * 2 - 1)
        self.body = nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.head_w = nn.Parameter(torch.randn(members, w, k, generator=g) / w ** 0.5)
        self.head_b = nn.Parameter(torch.zeros(members, k))

    def forward(self, x):
        """(B, K, k) member logits."""
        h = self.body(x[:, None, :] * self.r[None])
        return torch.einsum("bmw,mwk->bmk", h, self.head_w) + self.head_b


def member_ce(z, y):
    return nn.functional.cross_entropy(z.reshape(-1, z.shape[-1]), y.repeat_interleave(z.shape[1]))


def member_kd(z, soft, T):
    """As models.train_mlp KD: KL to teacher probabilities re-tempered at T, averaged over members."""
    t = torch.softmax(torch.log(soft + 1e-8) / T, 1)
    lp = torch.log_softmax(z / T, -1)
    return nn.functional.kl_div(lp.reshape(-1, z.shape[-1]), t.repeat_interleave(z.shape[1], 0),
                                reduction="batchmean") * T * T


def train(Xtr, ytr, Xes, yes, k, seed=0, epochs=30, soft=None, T=4.0):
    """As models.train_mlp, with the mean member CE (or member KD when `soft` is given) as loss; early stop on CE."""
    torch.manual_seed(seed)
    m = TabMMini(Xtr.shape[1], k, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt, Xv, yv = map(torch.from_numpy, (Xtr, ytr, Xes, yes))
    St = None if soft is None else torch.from_numpy(soft.astype(np.float32))
    best, state, g = np.inf, None, torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        for b in torch.randperm(len(yt), generator=g).split(512):
            loss = member_ce(m(Xt[b]), yt[b]) if St is None else member_kd(m(Xt[b]), St[b], T)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = sum(member_ce(m(Xv[i:i + 50_000]), yv[i:i + 50_000]).item() * len(yv[i:i + 50_000])
                    for i in range(0, len(yv), 50_000)) / len(yv)
        if v < best:
            best, state = v, {k_: t_.clone() for k_, t_ in m.state_dict().items()}
    m.load_state_dict(state)
    return m.eval()


def member_proba(m, X):
    with torch.no_grad():
        return torch.softmax(m(torch.from_numpy(X)), -1).numpy()  # (n, K, k)


def nlog(q):
    """-log P(benign): ranks as 1 - P(benign) but does not saturate to 1.0 in float32 (PREREG §9 SEL amendment 2)."""
    return -np.log(np.maximum(np.asarray(q, np.float64), 1e-45))


def scores(p, bi):
    pm = p.mean(1)
    ent = lambda q: -(q * np.log(q + 1e-12)).sum(-1)  # noqa: E731
    return {"": nlog(pm[:, bi]), "_min": nlog(p[:, :, bi].min(1)), "_mi": ent(pm) - ent(p).mean(1)}


def fit_models(Xf, yf, Xe, ye, k, bi, seed, **_):
    out = {}
    for name, enc in (("tabm", ple.Encoder()), ("tabm_raw", None)):
        tf = (lambda X: X) if enc is None else enc.fit(Xf).transform
        m = train(tf(Xf), yf, tf(Xe), ye, k, seed=seed)
        if enc is None:
            out[name] = lambda X, m=m: scores(member_proba(m, X), bi)[""]
            continue
        for suf in ("", "_min", "_mi"):
            out[name + suf] = lambda X, m=m, tf=tf, suf=suf: scores(member_proba(m, tf(X)), bi)[suf]
    return out


if __name__ == "__main__":
    ple.main("tabm", fit_models)
