"""Two prior-art baselines (PREREG §9, 2026-10-24 rows), as close to the authors' code as our data allows.

Deep SAD (Ruff et al., ICLR 2020; github.com/lukasruff/Deep-SAD-PyTorch): bias-free MLP 128-64-32, BatchNorm
without affine, LeakyReLU; autoencoder pre-training; centre c = mean embedding of benign training rows, entries
with |c| < 0.1 pushed to +-0.1; loss ||phi(x) - c||^2 for benign, eta * (||phi(x) - c||^2 + 1e-6)^-1 for known
attacks, eta = 1; Adam lr 1e-3, weight decay 1e-6. Score = ||phi(x) - c||^2.
CADE (Yang et al., USENIX Security 2021; github.com/whyisyoung/CADE): autoencoder d-512-128-32-K (K = number of
training classes); loss = MSE + lambda * contrastive, lambda = 0.1, margin 10, unsquared distance
(same class: ||zi - zj||, different: relu(m - ||zi - zj||)), 25% same-class pairs. Score = MAD outlier score of the
nearest class centroid: |d - median_k| / (1.4826 * MAD_k).
Our deviations (stated in §9): batch 1024, at most 30 epochs per stage, early stopping on the model's own 10%
split, linear decoder output (our features are standardised, not in [0, 1]), ReLU hidden layers in CADE.
"""
import numpy as np
import torch
from torch import nn

BATCH, EPOCHS = 1024, 30


def _bn_lrelu(i, o):
    return nn.Sequential(nn.Linear(i, o, bias=False), nn.BatchNorm1d(o, eps=1e-4, affine=False), nn.LeakyReLU())


class SADNet(nn.Module):
    def __init__(self, d, h=(128, 64), rep=32):
        super().__init__()
        dims = (d, *h)
        self.enc = nn.Sequential(*[_bn_lrelu(a, b) for a, b in zip(dims, dims[1:])], nn.Linear(h[-1], rep, bias=False))
        back = (rep, *h[::-1])
        self.dec = nn.Sequential(*[_bn_lrelu(a, b) for a, b in zip(back, back[1:])], nn.Linear(h[0], d, bias=False))


class CADENet(nn.Module):
    def __init__(self, d, k, h=(512, 128, 32)):
        super().__init__()
        dims = (d, *h, k)
        layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers += [nn.Linear(a, b)] + ([nn.ReLU()] if i < len(dims) - 2 else [])
        self.enc = nn.Sequential(*layers)
        back = dims[::-1]
        layers = []
        for i, (a, b) in enumerate(zip(back, back[1:])):
            layers += [nn.Linear(a, b)] + ([nn.ReLU()] if i < len(back) - 2 else [])
        self.dec = nn.Sequential(*layers)


def _embed(f, X, chunk=100_000):
    with torch.no_grad():
        return torch.cat([f(torch.from_numpy(X[i:i + chunk])) for i in range(0, len(X), chunk)])


def _fit(net, params, step, eval_loss, n, seed, wd=0.0):
    """Mini-batch Adam on step(batch_idx, generator); keeps the weights with the lowest eval_loss()."""
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    best, state = np.inf, net.state_dict()
    for _ in range(EPOCHS):
        net.train()
        for b in torch.randperm(n, generator=g).split(BATCH):
            if len(b) < 2:  # BatchNorm needs > 1 row
                continue
            loss = step(b, g)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = eval_loss()
        if v < best:
            best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)


def deep_sad(Xtr, attack_tr, Xes, attack_es, seed, eta=1.0):
    """Returns score(X) -> squared distance to the centre."""
    torch.manual_seed(seed)
    net = SADNet(Xtr.shape[1])
    Xt, Xv = torch.from_numpy(Xtr), torch.from_numpy(Xes)
    at, av = torch.from_numpy(attack_tr), torch.from_numpy(attack_es)
    mse = lambda X: ((net.dec(net.enc(X)) - X) ** 2).mean()  # noqa: E731
    _fit(net, net.parameters(), lambda b, _: mse(Xt[b]), lambda: mse(Xv).item(), len(Xt), seed, wd=1e-6)
    net.eval()
    c = _embed(net.enc, Xtr[~attack_tr]).mean(0)
    c[(c.abs() < 0.1) & (c < 0)], c[(c.abs() < 0.1) & (c >= 0)] = -0.1, 0.1

    def sad(X, a):
        dist = ((net.enc(X) - c) ** 2).sum(1)
        return torch.where(a, eta * (dist + 1e-6) ** -1, dist).mean()
    _fit(net, net.enc.parameters(), lambda b, _: sad(Xt[b], at[b]), lambda: sad(Xv, av).item(), len(Xt), seed,
         wd=1e-6)
    net.eval()
    return lambda X: ((_embed(net.enc, X) - c) ** 2).sum(1).numpy()


def cade(Xtr, ytr, Xes, yes, k, seed, lam=0.1, margin=10.0, similar=0.25):
    """Returns score(X) -> MAD outlier score w.r.t. the nearest training-class centroid in latent space."""
    torch.manual_seed(seed)
    net = CADENet(Xtr.shape[1], k)
    Xt, Xv, yt, yv = (torch.from_numpy(a) for a in (Xtr, Xes, ytr, yes))
    by_cls = [torch.nonzero(yt == c).flatten() for c in range(k)]

    def partners(y, g):
        """Per row: a random same-class row with prob `similar`, otherwise a random row of another class."""
        same = torch.rand(len(y), generator=g) < similar
        other = (y + torch.randint(1, k, (len(y),), generator=g)) % k
        cls = torch.where(same, y, other)
        out = torch.empty(len(y), dtype=torch.long)
        for c in range(k):
            m = cls == c
            if m.any():
                pool = by_cls[c]
                out[m] = pool[torch.randint(len(pool), (int(m.sum()),), generator=g)]
        return out, same

    def loss(X, Xp, same):
        z, zp = net.enc(X), net.enc(Xp)
        d = torch.sqrt(((z - zp) ** 2).sum(1) + 1e-10)
        con = torch.where(same, d, torch.relu(margin - d)).mean()
        return ((net.dec(z) - X) ** 2).mean() + lam * con

    def step(b, g):
        p, same = partners(yt[b], g)
        return loss(Xt[b], Xt[p], same)

    gv = torch.Generator().manual_seed(seed + 1)
    pv, sv = partners(yv, gv)  # fixed partner draw for the early-stopping loss (partners from train rows)
    _fit(net, net.parameters(), step, lambda: loss(Xv, Xt[pv], sv).item(), len(Xt), seed)
    net.eval()
    z = _embed(net.enc, Xtr)
    cent = torch.stack([z[yt == c].mean(0) for c in range(k)])
    med, mad = np.empty(k), np.empty(k)
    for c in range(k):
        d = torch.linalg.norm(z[yt == c] - cent[c], dim=1).numpy()
        med[c] = np.median(d)
        mad[c] = 1.4826 * np.median(np.abs(d - med[c])) + 1e-12

    def score(X):
        dist = torch.cdist(_embed(net.enc, X), cent).numpy()
        j = dist.argmin(1)
        return np.abs(dist[np.arange(len(j)), j] - med[j]) / mad[j]
    return score
