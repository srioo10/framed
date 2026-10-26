"""Score-bank members (PREREG §9 H5). Prep, MLP and train_mlp come from the LOFO pilot; the early-stopping
set is now an argument so it can be carved from each model's own training rows."""
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from torch import nn
from xgboost import XGBClassifier


class Prep:
    """Label encode -> train-mean impute -> signed log1p -> standard scale. No clamping: extreme values in an
    unseen family are signal for unknown detection."""

    def fit(self, df):
        self.codes = {c: {v: i for i, v in enumerate(pd.unique(df[c].astype(str)))}
                      for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}
        X = self._raw(df)
        self.mean = np.nanmean(X, 0)
        X = self._log(np.where(np.isnan(X), self.mean, X))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def _raw(self, df):
        X = np.column_stack([df[c].astype(str).map(self.codes[c]).to_numpy(np.float64, na_value=np.nan)
                             if c in self.codes else df[c].to_numpy(np.float64, na_value=np.nan) for c in df.columns])
        X[~np.isfinite(X)] = np.nan
        return X

    @staticmethod
    def _log(X):
        return np.sign(X) * np.log1p(np.abs(X))

    def transform(self, df):
        X = self._raw(df)
        return ((self._log(np.where(np.isnan(X), self.mean, X)) - self.mu) / self.sd).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d, k, w=64, depth=2):
        super().__init__()
        dims = [d] + [w] * (depth - 1) + [w // 2]  # default d-64-32: the bank's original MLP
        self.body = nn.Sequential(*[layer for a, b in zip(dims, dims[1:]) for layer in (nn.Linear(a, b), nn.ReLU())])
        self.head = nn.Linear(dims[-1], k)

    def forward(self, x):
        h = self.body(x)
        return self.head(h), h


class SNMLP(nn.Module):
    """DDU trunk (Mukhoti et al., CVPR 2023): spectral-normalised residual layers keep feature distances
    bi-Lipschitz, so a density on the features can see inputs far from every known class."""
    def __init__(self, d, k, w=64, depth=2):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        self.inp = sn(nn.Linear(d, w))  # creation order inp, res, head keeps the bank's original initialisation
        self.res = nn.ModuleList([sn(nn.Linear(w, w)) for _ in range(depth - 1)])
        self.head = nn.Linear(w, k)

    def forward(self, x):
        h = torch.relu(self.inp(x))
        for r in self.res:
            h = h + torch.relu(r(h))
        return self.head(h), h


class FeatureGDA:
    """DDU density: one Gaussian per known class on the trunk features (shared jitter), score = -log q(h)."""
    def fit(self, m, X, y, k):
        with torch.no_grad():
            H = torch.cat([m(torch.from_numpy(X[i:i + 200_000]))[1] for i in range(0, len(X), 200_000)]).double()
        yt = torch.from_numpy(y)
        self.mu, self.prec, self.logdet, self.logpi = [], [], [], []
        for c in range(k):
            h = H[yt == c]
            mu = h.mean(0)
            S = torch.cov((h - mu).T) + 1e-4 * torch.eye(H.shape[1], dtype=H.dtype)
            L = torch.linalg.cholesky(S)
            self.mu.append(mu), self.prec.append(torch.cholesky_inverse(L))
            self.logdet.append(2 * torch.log(torch.diagonal(L)).sum()), self.logpi.append(np.log(len(h) / len(H)))
        return self

    def class_loglik(self, m, X, chunk=200_000):
        """log pi_c + log N(h; mu_c, Sigma_c) up to the shared constant, one column per known class."""
        out = []
        with torch.no_grad():
            for i in range(0, len(X), chunk):
                h = m(torch.from_numpy(X[i:i + chunk]))[1].double()
                out.append(torch.stack([-0.5 * (((h - mu) @ P) * (h - mu)).sum(1) - 0.5 * ld + lpi
                                        for mu, P, ld, lpi in zip(self.mu, self.prec, self.logdet, self.logpi)], 1))
        return torch.cat(out)

    def score(self, m, X, chunk=200_000):
        return -torch.logsumexp(self.class_loglik(m, X, chunk), 1).numpy()


def train_mlp(Xtr, ytr, Xes, yes, k, soft=None, T=4.0, seed=0, epochs=30, arch=None, weight=None):
    """CE on hard labels, or KD (KL to teacher soft labels at temperature T) when `soft` is given.
    Early stop on CE of (Xes, yes), which must be carved from the model's own training rows."""
    torch.manual_seed(seed)
    m = (arch or MLP)(Xtr.shape[1], k)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    St = None if soft is None else torch.from_numpy(soft.astype(np.float32))
    Xv, yv = torch.from_numpy(Xes), torch.from_numpy(yes)
    best, state, g = np.inf, None, torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        for b in torch.randperm(len(yt), generator=g).split(512):
            z, _ = m(Xt[b])
            if St is None:
                loss = nn.functional.cross_entropy(z, yt[b], weight=weight)
            else:  # teacher probabilities sharpened/softened at T via log-space
                t = torch.softmax(torch.log(St[b] + 1e-8) / T, 1)
                loss = nn.functional.kl_div(torch.log_softmax(z / T, 1), t, reduction="batchmean") * T * T
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = nn.functional.cross_entropy(m(Xv)[0], yv).item()
        if v < best:
            best, state = v, {k_: t_.clone() for k_, t_ in m.state_dict().items()}
    m.load_state_dict(state)
    return m


def int8(m):
    """The same MLP with int8 Linear weights (dynamic quantisation): the deployable model."""
    return torch.ao.quantization.quantize_dynamic(m.eval(), {nn.Linear}, dtype=torch.qint8)


def mlp_proba(m, X, chunk=200_000):
    with torch.no_grad():
        return np.vstack([torch.softmax(m(torch.from_numpy(X[i:i + chunk]))[0], 1).numpy()
                          for i in range(0, len(X), chunk)])


def rf(seed):
    return RandomForestClassifier(300, n_jobs=-1, random_state=seed, min_samples_leaf=2)


def xgb(seed):
    return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", n_jobs=-1,
                         random_state=seed)


def oof_proba(X, y, seed):
    """3-fold out-of-fold XGB probabilities on the training rows (KD soft labels, tail weights)."""
    return cross_val_predict(xgb(seed), X, y, cv=3, method="predict_proba")


def xgb_tail(X, y, benign, seed, q=0.97, weight=8.0):
    """Idea 5 control: refit XGB with benign rows above the q-quantile of their own OOF not_benign score
    weighted `weight`x (a two-stage Neyman-Pearson surrogate). Uses training rows only."""
    oof = 1 - oof_proba(X, y, seed)[:, benign]
    b = y == benign
    w = np.ones(len(y))
    w[b & (oof >= np.quantile(oof[b], q))] = weight
    return xgb(seed).fit(X, y, sample_weight=w)


def iforest(Xb, seed):
    return IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xb)
