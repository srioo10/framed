"""Fuzzy C-means (§3.5, Eqs. 13-14) and the cluster-based undersampling of majority classes.

The paper does not say how cluster labels turn into the reduced class sizes of Table 9 ("Grouping",
Algorithm 1 line 8). We keep the reported size and draw it from every cluster in proportion to the
cluster's size, so the reduced class still covers all FCM clusters."""
import numpy as np
from scipy.special import logsumexp

FIT_ROWS = 50_000  # shortcut: centroids fit on a 50k-row sample per class, all rows then assigned; exact FCM on 1.9M rows x 183 clusters x 1,156 iterations is hours per class


def fcm(X, c, m, error, maxiter, rng):
    """Return centroids. Memberships are computed in log space: m close to 1 (1.013) makes the
    exponent 2/(m-1) = 154 and plain powers overflow."""
    X = X.astype(np.float64)
    c = min(c, len(X))
    V = X[rng.choice(len(X), c, replace=False)]
    p, u = 2.0 / (m - 1.0), None
    for _ in range(maxiter):
        d2 = np.maximum(_sqdist(X, V), 1e-12)
        a = -0.5 * p * np.log(d2)                       # u_ki ∝ d_ki^(-2/(m-1))
        u_new = np.exp(a - logsumexp(a, axis=1, keepdims=True))
        w = u_new ** m
        V = (w.T @ X) / w.sum(0)[:, None]
        if u is not None and np.abs(u_new - u).max() < error:
            break
        u = u_new
    return V


def _sqdist(X, V):
    return np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ V.T + (V ** 2).sum(1)[None], 0)


def assign(X, V, chunk=200_000):
    """Hard cluster = highest membership = nearest centroid (for any m > 1)."""
    return np.concatenate([_sqdist(X[i:i + chunk].astype(np.float64), V).argmin(1) for i in range(0, len(X), chunk)])


def undersample(X, target, params, rng):
    """Indices of `target` rows of X, allocated across FCM clusters by size (largest remainder)."""
    n = len(X)
    if target >= n:
        return np.arange(n)
    fit = X[rng.choice(n, min(n, FIT_ROWS), replace=False)]
    labels = assign(X, fcm(fit, rng=rng, **params))
    sizes = np.bincount(labels)
    exact = sizes * target / n
    quota = np.floor(exact).astype(int)
    quota[np.argsort(quota - exact)[: target - quota.sum()]] += 1
    order = rng.permutation(n)  # rank of each row within its cluster, in random order
    srt = order[np.argsort(labels[order], kind="stable")]
    lab = labels[srt]
    rank = np.empty(n, dtype=np.int64)
    rank[srt] = np.arange(n) - np.searchsorted(lab, lab)
    return np.flatnonzero(rank < quota[labels])
