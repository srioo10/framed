"""Bowl fighters (PREREG §9 BOWL): small neural / neural-hybrid detectors. Every module returns (logits, latent);
logits are (B, k) or (B, M, k) for member ensembles. Device-agnostic (cuda on Kaggle, cpu locally).

Encoders: "raw" (prepped features), "quantile" / "tree" bin edges (ple.Encoder / ple.TreeEncoder) followed by a
learned per-feature map 16 bins -> e. That map is a 1-D piecewise-linear function of each feature, so on device it is
a per-feature lookup (~4 comparisons + 2e MACs), not a d*16-wide input layer.
"""
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ple  # noqa: E402

TRANSC = 20  # ops charged per exp / sin / log on device


class BinMap(nn.Module):
    """x (B, d) -> per-feature piecewise-linear map of the bounded bin encoding -> (B, d * e)."""

    def __init__(self, edges, e):
        super().__init__()
        self.register_buffer("lo", torch.tensor(edges[:, :-1]))
        self.register_buffer("w", torch.tensor(edges[:, 1:] - edges[:, :-1]))
        d, b = self.lo.shape
        self.W = nn.Parameter(torch.randn(d, b, e) / math.sqrt(b))
        self.b = nn.Parameter(torch.zeros(d, e))

    def codes(self, x):
        """(B, d, bins) in [0, 1], each component non-decreasing in its feature (so a box maps to a box)."""
        x = x[:, :, None]
        return torch.where(self.w > 0, ((x - self.lo) / torch.where(self.w > 0, self.w, 1)).clamp(0, 1),
                           (x >= self.lo + self.w).float())

    def forward(self, x):
        return (torch.einsum("ndb,dbe->nde", self.codes(x), self.W) + self.b).flatten(1)

    def ops(self):
        d, b, e = self.W.shape
        return d * (math.ceil(math.log2(b)) + 2 * e)


class Periodic(nn.Module):
    """RealMLP-lite input (Holzmüller et al. NeurIPS'24): smooth clip, learned scale, periodic embedding -> (B, d * e)."""

    def __init__(self, d, e, nf=8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.f = nn.Parameter(torch.randn(d, nf) * 0.5)
        self.W = nn.Parameter(torch.randn(d, 2 * nf + 1, e) / math.sqrt(2 * nf + 1))

    def forward(self, x):
        x = self.scale * x / torch.sqrt(1 + (x / 3) ** 2)
        a = 2 * math.pi * self.f * x[:, :, None]
        z = torch.cat([torch.sin(a), torch.cos(a), x[:, :, None]], -1)
        return torch.einsum("ndf,dfe->nde", z, self.W).flatten(1)

    def ops(self):
        d, f2, e = self.W.shape
        return d * ((f2 - 1) * TRANSC + f2 * e + 4)


def encoder(kind, Xf, yf, e, seed):
    d = Xf.shape[1]
    if kind == "raw":
        return nn.Identity(), d, 0
    if kind == "periodic":
        m = Periodic(d, e)
        return m, d * e, None
    enc = (ple.Encoder().fit(Xf) if kind == "quantile" else ple.TreeEncoder().fit(Xf, yf, seed))
    return BinMap(enc.b, e), d * e, None


class Body(nn.Module):
    """enc -> [Linear (+spectral norm) -> act] * depth; TabM members scale the body input by r_m (k heads each)."""

    def __init__(self, enc, din, k, w, depth, members=1, sn=False, act="relu", clip=None):
        super().__init__()
        self.enc, self.M = enc, members
        if clip is not None:  # (lo, hi) per input column, +-inf where unclipped
            self.register_buffer("clo", torch.as_tensor(clip[0], dtype=torch.float32))
            self.register_buffer("chi", torch.as_tensor(clip[1], dtype=torch.float32))
        lin = (lambda a, b: nn.utils.parametrizations.spectral_norm(nn.Linear(a, b))) if sn else nn.Linear
        A = {"relu": nn.ReLU, "selu": nn.SELU}[act]
        dims = [din] + [w] * depth
        self.body = nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in (lin(a, b), A())])
        if members > 1:
            self.r = nn.Parameter(torch.randint(0, 2, (members, din)).float() * 2 - 1)
            self.hw = nn.Parameter(torch.randn(members, w, k) / math.sqrt(w))
            self.hb = nn.Parameter(torch.zeros(members, k))
        else:
            self.head = nn.Linear(w, k)
        self.din, self.w, self.depth, self.k = din, w, depth, k

    def forward(self, x):
        if hasattr(self, "clo"):
            x = torch.minimum(torch.maximum(x, self.clo), self.chi)
        z = self.enc(x)
        if self.M > 1:
            h = self.body(z[:, None, :] * self.r[None])
            return torch.einsum("bmw,mwk->bmk", h, self.hw) + self.hb, h.mean(1)
        h = self.body(z)
        return self.head(h), h

    def ops(self):
        per = self.din * self.w + (self.depth - 1) * self.w * self.w + self.w * self.k
        enc = self.enc.ops() if hasattr(self.enc, "ops") else 0
        return enc + self.M * (per + (self.din if self.M > 1 else 0))


class NODE(nn.Module):
    """NODE-lite (Popov et al. ICLR'20): T differentiable oblivious trees of depth D. Feature choice = softmax over
    features per level, split = sigmoid((x_f - b) / tau); tau anneals during training; eval uses hard choices and
    hard splits (the deployed form: T * D comparisons + T * k adds)."""

    def __init__(self, d, k, T, D):
        super().__init__()
        self.sel = nn.Parameter(torch.randn(T, D, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(T, D))
        self.leaf = nn.Parameter(torch.randn(T, 2 ** D, k) * 0.1)
        self.tau, self.T, self.D, self.k = 1.0, T, D, k
        self.register_buffer("bits", torch.tensor([[(i >> l) & 1 for l in range(D)] for i in range(2 ** D)]).float())

    def forward(self, x):
        if self.training:
            xs = torch.einsum("nd,tld->ntl", x, torch.softmax(self.sel / max(self.tau, 0.05), -1))
            p = torch.sigmoid((xs - self.b) / max(self.tau, 0.05))
        else:
            xs = x[:, self.sel.argmax(-1)]  # (n, T, D)
            p = (xs > self.b).float()
        # leaf weight = prod over levels of p or (1 - p) according to the leaf's bit pattern
        lw = torch.exp(torch.einsum("ntl,ql->ntq", torch.log(p + 1e-6), self.bits)
                       + torch.einsum("ntl,ql->ntq", torch.log(1 - p + 1e-6), 1 - self.bits))
        h = torch.einsum("ntq,tqk->ntk", lw, self.leaf)
        return h.sum(1), h.flatten(1)

    def ops(self):
        return self.T * (self.D + self.k)


class Ensemble(nn.Module):
    """Deep ensemble (Lakshminarayanan et al. NeurIPS'17): independently initialised bodies trained side by side;
    member logits stacked (B, sum of members, k) so every head averages / mins over all of them."""

    def __init__(self, bodies):
        super().__init__()
        self.bodies = nn.ModuleList(bodies)

    def forward(self, x):
        out = [b(x) for b in self.bodies]
        z = torch.cat([o[0] if o[0].dim() == 3 else o[0][:, None] for o in out], 1)
        return z, torch.stack([o[1] for o in out]).mean(0)

    def ops(self):
        return sum(b.ops() for b in self.bodies)


class Hybrid(nn.Module):
    """Body (TabM / MLP) + NODE-lite branch on the prepped features, logits summed: the NODE's hard oblivious splits
    supply the axis-aligned feature ANDs a smooth body misses (round-1 autopsy)."""

    def __init__(self, body, node):
        super().__init__()
        self.body, self.node = body, node

    def forward(self, x):
        z, h = self.body(x)
        zn = self.node(x)[0]
        return (z + zn[:, None] if z.dim() == 3 else z + zn), h

    def ops(self):
        return self.body.ops() + self.node.ops()


class LeafNet(nn.Module):
    """GBDT-leaf -> net (He et al. 2014): a small LightGBM's leaf index per tree -> learned embedding, summed -> MLP."""

    def __init__(self, Xf, yf, k, trees, leaves, e, w, seed):
        super().__init__()
        import lightgbm as lgb
        self.gbm = lgb.LGBMClassifier(n_estimators=trees, num_leaves=leaves, learning_rate=0.1, verbose=-1,
                                      random_state=seed, n_jobs=4).fit(Xf, yf)
        nt = self.gbm.booster_.num_trees()
        self.emb = nn.Parameter(torch.randn(nt, leaves, e) * 0.1)
        self.mlp = nn.Sequential(nn.Linear(e, w), nn.ReLU(), nn.Linear(w, k))
        self.nt, self.leaves, self.e, self.w, self.k = nt, leaves, e, w, k

    def leaf_idx(self, x):
        return torch.from_numpy(self.gbm.predict(x.cpu().numpy(), pred_leaf=True).astype(np.int64)).to(x.device)

    def forward(self, x):
        L = self.leaf_idx(x)
        h = self.emb[torch.arange(self.nt, device=x.device)[None], L].sum(1)
        return self.mlp(h), h

    def tree_nodes(self):
        return int(self.gbm.booster_.trees_to_dataframe().shape[0])

    def ops(self):
        return 4 * self.nt * math.ceil(math.log2(self.leaves)) + self.nt * self.e + self.e * self.w + self.w * self.k


def _affine(mu, r, W, b=None):
    """Interval through x @ W.T + b in centre / radius form."""
    out = mu @ W.T + (0 if b is None else b)
    return out, r @ W.abs().T


def _body_margins(body, lo, hi, bi, lower=False, raw=False):
    """Upper (or lower) bounds (B, M, k) of z_i - z_bi (raw: of z_i) over the input box [lo, hi] for one Body
    (interval bound propagation)."""
    if hasattr(body, "clo"):
        lo, hi = (torch.minimum(torch.maximum(v, body.clo), body.chi) for v in (lo, hi))
    if isinstance(body.enc, BinMap):
        El, Eh = body.enc.codes(lo), body.enc.codes(hi)
        mu = torch.einsum("ndb,dbe->nde", (El + Eh) / 2, body.enc.W).flatten(1) + body.enc.b.flatten()
        r = torch.einsum("ndb,dbe->nde", (Eh - El) / 2, body.enc.W.abs()).flatten(1)
    elif isinstance(body.enc, nn.Identity):
        lo, hi = (v.clamp(-1e30, 1e30) for v in (lo, hi))  # an unclipped +inf column gives an infinite bound below
        mu, r = (lo + hi) / 2, (hi - lo) / 2
    else:
        raise ValueError("no interval bound for this encoder")  # periodic embeddings are not monotone
    if body.M > 1:
        mu, r = mu[:, None, :] * body.r, r[:, None, :] * body.r.abs()
    for layer in body.body:
        if isinstance(layer, nn.Linear):
            mu, r = _affine(mu, r, layer.weight, layer.bias)
        else:  # ReLU / SELU are monotone
            a, b = layer(mu - r), layer(mu + r)
            mu, r = (a + b) / 2, (b - a) / 2
    if body.M > 1:
        D = body.hw if raw else body.hw - body.hw[:, :, bi:bi + 1]
        out = torch.einsum("bmw,mwk->bmk", mu, D) + (body.hb if raw else body.hb - body.hb[:, bi:bi + 1]) + \
            (-1 if lower else 1) * torch.einsum("bmw,mwk->bmk", r, D.abs())
    else:
        W, b = body.head.weight, body.head.bias
        dm, dr = _affine(mu, r, W, b) if raw else _affine(mu, r, W - W[bi], b - b[bi])
        out = (dm - dr if lower else dm + dr)[:, None]
    return torch.nan_to_num(out, nan=-float("inf") if lower else float("inf"))  # unbounded stays sound either side


def margins_upper(m, lo, hi, bi, lower=False, raw=False):
    if isinstance(m, Ensemble):
        return torch.cat([_body_margins(b, lo, hi, bi, lower, raw) for b in m.bodies], 1)
    return _body_margins(m, lo, hi, bi, lower, raw)


def pgd_point(m, lo, hi, bi, sign, steps=10, gen=None):
    """PREREG §9 K18: PGD in the box [lo, hi] from a random start, ascending sign x mean-member log P(benign)
    (sign = -1: push benign rows to look attacked; +1: push attack rows to look benign); step 2.5 x width / steps."""
    lo, hi = lo.clamp(-1e4, 1e4), hi.clamp(-1e4, 1e4)  # the mean-bytes upper bound is inf; the cert-mode clip saturates long before
    w = hi - lo
    x = lo + w * torch.rand(lo.shape, generator=gen).to(lo.device)
    was = m.training
    m.eval()
    for _ in range(steps):
        x.requires_grad_(True)
        z = m(x)[0]
        z = z if z.dim() == 3 else z[:, None]
        obj = sign * torch.log_softmax(z, -1)[..., bi].mean()
        g, = torch.autograd.grad(obj, x)
        x = torch.max(torch.min(x.detach() + 2.5 * w / steps * g.sign(), hi), lo)
    m.train(was)
    return x.detach()


def score_lower(m, lo, hi, bi, head, device="cpu", bs=50_000, f64=True):
    """Sound lower bound of the nlog / min / energy head score over each row's box (robust-alarm gating, Theorem 1'):
    member log P(benign) <= -logsumexp_i lower(z_i - z_bi). energy = nlog - logsumexp(mean z): minus the logsumexp
    of the mean upper logits; one member: the score is exactly -z_bi, so also -upper(z_bi)."""
    if f64 and next(m.parameters()).dtype != torch.float64:  # PREREG §9 AUDIT F1: float64 copy, outward margin below
        m = copy.deepcopy(m).double()
    out = []
    with torch.no_grad():
        for i in range(0, len(lo), bs):
            box = [torch.as_tensor(v[i:i + bs], dtype=torch.float64 if f64 else torch.float32, device=device)
                   for v in (lo, hi)]
            d = margins_upper(m, *box, bi, lower=True)
            lp = -torch.logsumexp(d.double(), -1)  # (B, M) upper bound of member log P(benign)
            s = -(torch.logsumexp(lp, 1) - math.log(lp.shape[1])) if head != "min" else -lp.min(1).values
            if head == "energy":
                zu = torch.nan_to_num(margins_upper(m, *box, bi, raw=True).double(), nan=float("inf")).mean(1)
                s = s - torch.logsumexp(zu, -1)
                if lp.shape[1] == 1:
                    s = torch.maximum(s, -zu[:, bi])
            out.append((s - 1e-6 if f64 else s).cpu().numpy())
    return np.concatenate(out)


def bab_block(m, h, lo, hi, bi, head, tau, max_boxes=8192, time_s=20.0, device="cpu", batch=65536):
    """PREREG §9 K5b: branch and bound by input splitting over one row's box [lo, hi]. Returns (blocked, decided):
    blocked only when the float64 IBP lower bound of every sub-box clears tau; a sub-box centre whose real score is
    <= tau is a realised attack (decided, not blocked); budget out -> (False, False). m64 = float64 copy of m."""
    t0, m64 = time.time(), copy.deepcopy(m).double()
    body = m.bodies[0] if isinstance(m, Ensemble) else m
    L, H = lo[None].astype(np.float64), hi[None].astype(np.float64)
    if hasattr(body, "clo"):  # the input clip is part of the model: points outside it score as the clipped point
        c0, c1 = body.clo.cpu().numpy().astype(np.float64), body.chi.cpu().numpy().astype(np.float64)
        L, H = np.clip(L, c0, c1), np.clip(H, c0, c1)
    L, H = np.clip(L, -1e6, 1e6), np.clip(H, -1e6, 1e6)  # shortcut: an unclipped infinite column is split at +-1e6
    n = 0
    while len(L):  # L, H: pending boxes, taken in LIFO batches of `batch` (one batch = the whole frontier while it fits)
        bL, bH, L, H = L[-batch:], H[-batch:], L[:-batch], H[:-batch]
        keep = score_lower(m64, bL, bH, bi, head, device) <= tau
        bL, bH = bL[keep], bH[keep]
        if not len(bL):
            continue
        c = (bL + bH) / 2
        if (h.score(m, c.astype(np.float32), device) <= tau).any():
            return False, True
        n += len(bL)
        if n >= max_boxes or time.time() - t0 > time_s:
            return False, False
        j, r = (bH - bL).argmax(1), np.arange(len(bL))
        L2, H1 = bL.copy(), bH.copy()
        H1[r, j] = L2[r, j] = c[r, j]
        L, H = np.concatenate([L, bL, L2]), np.concatenate([H, H1, bH])
    return True, True


def score_upper(m, lo, hi, bi, head, device="cpu", bs=50_000, f64=True):
    """Sound upper bound of the nlog / min / energy head score over each row's box (PREREG §9 CERT Theorem 2):
    P_m(benign) >= 1 / sum_i exp(upper(z_i - z_bi)); energy as in score_lower with the bounds swapped."""
    if f64:  # PREREG §9 AUDIT F1: float64 copy, outward margin below
        m = copy.deepcopy(m).double()
    out = []
    with torch.no_grad():
        for i in range(0, len(lo), bs):
            box = [torch.as_tensor(v[i:i + bs], dtype=torch.float64 if f64 else torch.float32, device=device)
                   for v in (lo, hi)]
            d = margins_upper(m, *box, bi)
            lp = -torch.logsumexp(d.double(), -1)  # (B, M) lower bound of member log P(benign)
            s = -(torch.logsumexp(lp, 1) - math.log(lp.shape[1])) if head != "min" else -lp.min(1).values
            if head == "energy":
                zl = torch.nan_to_num(margins_upper(m, *box, bi, lower=True, raw=True).double(), nan=-float("inf")).mean(1)
                s = s - torch.logsumexp(zl, -1)
                if lp.shape[1] == 1:
                    s = torch.minimum(s, -zl[:, bi])
            out.append((s + 1e-6 if f64 else s).cpu().numpy())
    return np.concatenate(out)


def size_bytes(m):
    """int8 parameters (+ tree nodes x 8 B + float32 bin edges)."""
    n = sum(p.numel() for name, p in m.named_parameters() if not name.startswith("gbm"))
    edges = sum(b.numel() * 4 for name, b in m.named_buffers() if name.endswith(("lo", "w")))
    trees = m.tree_nodes() * 8 if hasattr(m, "tree_nodes") else 0
    return n + edges + trees


def train(m, Xf, yf, k, device, seed=0, epochs=20, lr=2e-3, bs=1024, soft=None, kd=0.0, ls=0.0, focal=0.0, T=4.0,
          Xs=None, Ss=None, syn=0.0, Lc=None, Hc=None, cert=0.0, bi=0, Ld=None, Hd=None, rdet=0.0, adv=0.0, advd=0.0, tail=0.0, tailq=0.05, gnoise=0.0, gmask=None):
    """Mean-member CE (label smoothing ls, focal gamma) mixed with KD (weight kd) to teacher probabilities `soft`,
    plus syn x KD on a batch of synthetic rows Xs labelled by a teacher (Ss), plus cert x worst-case CE (interval
    bound) on boxes [Lc, Hc] of benign fit rows (PREREG §9 CERT), plus rdet x worst-case "not benign" loss on deletion
    boxes [Ld, Hd] of attack fit rows (robust-alarm gating); adv / advd: the same two terms at PGD points in the same
    boxes instead of interval bounds (PREREG §9 K18 adversarial-training baseline); early stop on CE of a 10% carve of the
    fit rows; NODE tau anneals 1 -> 0.05. tail: K19' benign-tail loss, + tail x mean of the top tailq share of benign
    scores (-log P(benign)) in the batch (CVaR), aimed at the per-host-day-max threshold."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    n = len(yf)
    perm = torch.randperm(n, generator=g).numpy()
    va, tr = perm[: n // 10], perm[n // 10:]
    m = m.to(device)
    X = torch.from_numpy(Xf).to(device)
    y = torch.from_numpy(yf).to(device)
    S = None if soft is None or kd == 0 else torch.from_numpy(soft.astype(np.float32)).to(device)
    if syn > 0:
        XS, SS = torch.from_numpy(Xs).to(device), torch.from_numpy(Ss.astype(np.float32)).to(device)
    opt = torch.optim.AdamW([p for name, p in m.named_parameters() if not name.startswith("gbm")], lr=lr,
                            weight_decay=1e-4)
    if cert > 0 or adv > 0:
        LC, HC = (torch.from_numpy(np.asarray(v, np.float32)).to(device) for v in (Lc, Hc))
    if rdet > 0 or advd > 0:
        LD, HD = (torch.from_numpy(np.asarray(v, np.float32)).to(device) for v in (Ld, Hd))
    trt = torch.from_numpy(tr).to(device)
    best, state = np.inf, None
    for ep in range(epochs):
        if isinstance(m, NODE) or hasattr(m, "node"):
            (m.node if hasattr(m, "node") else m).tau = max(0.05, 1.0 - ep / max(1, epochs - 3))
        m.train()
        for b in trt[torch.randperm(len(trt), generator=g).to(device)].split(bs):
            xb = X[b]
            if gnoise > 0:  # randomized-smoothing baseline: Gaussian augmentation on the gmask columns
                xb = xb + gnoise * torch.randn(xb.shape, generator=g).to(device) * gmask
            z, _ = m(xb)
            zz = z if z.dim() == 3 else z[:, None]
            yy = y[b].repeat_interleave(zz.shape[1])
            flat = zz.reshape(-1, k)
            ce = nn.functional.cross_entropy(flat, yy, label_smoothing=ls, reduction="none")
            if focal > 0:
                ce = ce * (1 - torch.exp(-ce)) ** focal
            loss = ce.mean()
            if S is not None:
                t = torch.softmax(torch.log(S[b] + 1e-8) / T, 1).repeat_interleave(zz.shape[1], 0)
                kl = nn.functional.kl_div(torch.log_softmax(flat / T, 1), t, reduction="batchmean") * T * T
                loss = (1 - kd) * loss + kd * kl
            if tail > 0:
                sc = -torch.log_softmax(zz, -1)[..., bi].mean(-1)[y[b] == bi]
                if len(sc):
                    loss = loss + tail * sc.topk(max(1, int(tailq * len(sc)))).values.mean()
            if syn > 0:
                j = torch.randint(len(XS), (len(b),), generator=g).to(device)
                zs = m(XS[j])[0]
                zs = zs if zs.dim() == 3 else zs[:, None]
                ts = torch.softmax(torch.log(SS[j] + 1e-8) / T, 1).repeat_interleave(zs.shape[1], 0)
                loss = loss + syn * nn.functional.kl_div(torch.log_softmax(zs.reshape(-1, k) / T, 1), ts,
                                                         reduction="batchmean") * T * T
            if cert > 0:
                j = torch.randint(len(LC), (len(b),), generator=g).to(device)
                d = margins_upper(m, LC[j], HC[j], bi).clamp(max=50)
                ramp = min(1.0, (ep + 1) / max(1, epochs // 2))  # IBP warm-up (Gowal et al. 2018)
                loss = loss + ramp * cert * torch.logsumexp(d, -1).mean()
            if rdet > 0:
                j = torch.randint(len(LD), (len(b),), generator=g).to(device)
                s = torch.logsumexp(margins_upper(m, LD[j], HD[j], bi, lower=True).clamp(min=-50), -1)  # (B, M) >= 0
                ramp = min(1.0, (ep + 1) / max(1, epochs // 2))
                loss = loss - ramp * rdet * torch.log(-torch.expm1(-s.clamp(min=1e-4))).mean()  # -log P_worst(attack)
            if adv > 0:
                j = torch.randint(len(LC), (len(b),), generator=g).to(device)
                za = m(pgd_point(m, LC[j], HC[j], bi, -1.0, gen=g))[0]
                za = za if za.dim() == 3 else za[:, None]
                ramp = min(1.0, (ep + 1) / max(1, epochs // 2))
                loss = loss - ramp * adv * torch.log_softmax(za, -1)[..., bi].mean()  # CE to Benign at the PGD point
            if advd > 0:
                j = torch.randint(len(LD), (len(b),), generator=g).to(device)
                zd = m(pgd_point(m, LD[j], HD[j], bi, 1.0, gen=g))[0]
                zd = zd if zd.dim() == 3 else zd[:, None]
                pb = torch.softmax(zd, -1)[..., bi].clamp(max=1 - 1e-6)
                ramp = min(1.0, (ep + 1) / max(1, epochs // 2))
                loss = loss - ramp * advd * torch.log1p(-pb).mean()  # -log(1 - P(benign)) at the PGD point
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vb = torch.from_numpy(va).to(device)
            zv = m(X[vb])[0]
            zv = zv if zv.dim() == 3 else zv[:, None]
            v = nn.functional.cross_entropy(zv.reshape(-1, k), y[vb].repeat_interleave(zv.shape[1])).item()
        if v < best:
            best, state = v, {a: t.detach().clone() for a, t in m.state_dict().items()}
    m.load_state_dict(state)
    return m.eval()


def nlog(p):
    return -torch.log(p.clamp_min(1e-45))


class Head:
    """Unknown-score heads over (logits, latent): nlog (-log mean member P(benign)), min (worst member), energy
    (-logsumexp of mean logits, Liu NeurIPS'20) + nlog, maha (latent Mahalanobis to the benign class, Lee NeurIPS'18)."""

    def __init__(self, kind, bi):
        self.kind, self.bi = kind, bi

    def fit(self, m, Xf, yf, device):
        if self.kind == "maha":
            H = latent(m, Xf[yf == self.bi][:50_000], device)
            self.mu = H.mean(0)
            C = np.cov(H, rowvar=False) + 1e-3 * np.eye(H.shape[1])
            self.P = np.linalg.inv(C)
        return self

    def score(self, m, X, device, bs=50_000):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                z, h = m(torch.from_numpy(X[i:i + bs]).to(device))
                zz = z if z.dim() == 3 else z[:, None]
                p = torch.softmax(zz, -1)
                if self.kind == "nlog":
                    s = nlog(p.mean(1)[:, self.bi])
                elif self.kind == "min":
                    s = nlog(p[:, :, self.bi].min(1).values)
                elif self.kind == "energy":
                    s = nlog(p.mean(1)[:, self.bi]) - torch.logsumexp(zz.mean(1), -1)
                else:
                    d = h.cpu().numpy() - self.mu
                    s = torch.from_numpy(np.einsum("ni,ij,nj->n", d, self.P, d))
                out.append(s.double().cpu().numpy())
        return np.concatenate(out)

    def ops(self, m):
        return (m.w ** 2 if self.kind == "maha" and hasattr(m, "w") else 0) + TRANSC


def latent(m, X, device, bs=50_000):
    with torch.no_grad():
        return np.vstack([m(torch.from_numpy(X[i:i + bs]).to(device))[1].float().cpu().numpy()
                          for i in range(0, len(X), bs)])


def time_per_row(fn, X, reps=3):
    """Single-thread wall-clock per row at batch len(X) (median of reps)."""
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(X)
        ts.append(time.perf_counter() - t0)
    torch.set_num_threads(nt)
    return float(np.median(ts)) / len(X)
