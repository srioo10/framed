"""KitNET (Mirsky et al., NDSS 2018) on flow features: correlation-clustered feature map, an ensemble of small tied-
weight sigmoid autoencoders (hidden = 0.75 x inputs), and an output autoencoder over the ensemble RMSEs. Fit on
benign rows only; score = output RMSE. Batch SGD replaces the original one-pass online training (offline bank).
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree


def _clusters(X, m):
    """Feature groups of size <= m from complete-linkage clustering of the correlation distance (as in corClust)."""
    Xc = X - X.mean(0)
    n = np.sqrt((Xc ** 2).sum(0)) + 1e-12
    D = np.clip(1 - (Xc.T @ Xc) / np.outer(n, n), 0, None)
    if X.shape[1] <= m:
        return [list(range(X.shape[1]))]
    root = to_tree(linkage(D[np.triu_indices(len(D), 1)], "complete"))

    def split(node):
        if node.get_count() <= m:
            return [node.pre_order()]
        return split(node.get_left()) + split(node.get_right())
    return split(root)


class _AE:
    def __init__(self, d, rng, ratio=0.75):
        h = int(np.ceil(ratio * d))
        a = 1 / d
        self.W = rng.uniform(-a, a, (d, h))
        self.bh, self.bv = np.zeros(h), np.zeros(d)
        self.lo, self.hi = None, None

    def _norm(self, X):
        return (X - self.lo) / (self.hi - self.lo + 1e-12)

    def fit(self, X, rng, epochs=5, lr=0.1, batch=256):
        self.lo, self.hi = X.min(0), X.max(0)
        Z = self._norm(X)
        sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
        for _ in range(epochs):
            for i in np.array_split(rng.permutation(len(Z)), max(1, len(Z) // batch)):  # batch=1: online
                x = Z[i]
                y = sig(x @ self.W + self.bh)
                z = sig(y @ self.W.T + self.bv)
                dz = (z - x) * z * (1 - z)
                dy = (dz @ self.W) * y * (1 - y)
                self.W -= lr * (x.T @ dy + dz.T @ y) / len(x)
                self.bh -= lr * dy.mean(0)
                self.bv -= lr * dz.mean(0)
        return self

    def rmse(self, X):
        Z = np.clip(self._norm(X), -10, 10)
        y = 1 / (1 + np.exp(-(Z @ self.W + self.bh)))
        z = 1 / (1 + np.exp(-(y @ self.W.T + self.bv)))
        return np.sqrt(((z - Z) ** 2).mean(1))


class KitNET:
    def __init__(self, seed=0, max_ae=10, n_fit=200_000, online=False):
        self.rng, self.m, self.n = np.random.default_rng(seed), max_ae, n_fit
        self.kw = {"epochs": 1, "batch": 1} if online else {}  # online = the original one-pass per-sample SGD

    def fit(self, X):
        X = np.asarray(X, np.float64)
        if len(X) > self.n:
            X = X[self.rng.choice(len(X), self.n, replace=False)]
        self.groups = _clusters(X, self.m)
        self.ens = [_AE(len(g), self.rng).fit(X[:, g], self.rng, **self.kw) for g in self.groups]
        self.out = _AE(len(self.ens), self.rng).fit(self._layer(X), self.rng, **self.kw)
        return self

    def _layer(self, X):
        return np.column_stack([ae.rmse(X[:, g]) for ae, g in zip(self.ens, self.groups)])

    def score(self, X):
        return self.out.rmse(self._layer(np.asarray(X, np.float64)))
