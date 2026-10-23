"""Faithful MCDE baseline (Yue et al., Entropy 28(9):1026, 2026, §3.2-§4.2; PREREG §9 MCDE row).

The bank stores the raw branch outputs (HGB P(malicious), nearest-mode distance and mode id); tail
probabilities and the budgeted fusion are computed later from benign_rank / benign_cal (fusion_checks)."""
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier

K = 15
MIN_MODE = 20  # modes with fewer training vectors use the global scale


class RobustPrep:
    """Median impute -> signed log -> robust scale to the 5th-95th percentile range (fit on its own rows)."""

    def fit(self, df):
        self.codes = {c: {v: i for i, v in enumerate(pd.unique(df[c].astype(str)))}
                      for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])}
        X = self._log(self._raw(df))
        self.med = np.nanmedian(X, 0)
        X = np.where(np.isnan(X), self.med, X)
        lo, hi = np.percentile(X, [5, 95], axis=0)
        self.scale = np.where(hi - lo > 0, hi - lo, 1.0)
        self.med = np.median(X, 0)
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
        X = self._log(self._raw(df))
        X = np.where(np.isnan(X), self.med, X)
        return ((X - self.med) / self.scale).astype(np.float32)


class BenignModes:
    """Mini-batch k-means benign modes with floored diagonal scales; RMS diagonal Mahalanobis to the nearest mode."""

    def fit(self, Xb, seed):
        km = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096).fit(Xb)
        self.c = km.cluster_centers_.astype(np.float64)
        g = Xb.std(0).astype(np.float64)
        big = g > 1e-6
        floor = max(1e-3, 0.05 * np.median(g[big])) if big.any() else 1e-3
        g = np.maximum(g, floor)
        self.s = np.tile(g, (K, 1))
        lab = km.labels_
        for k in range(K):
            if (lab == k).sum() >= MIN_MODE:
                self.s[k] = np.maximum(Xb[lab == k].std(0), 0.05 * g)
        self.sizes = np.bincount(lab, minlength=K)
        return self

    def score(self, X, chunk=20_000):
        """(nearest-mode RMS distance, mode id). Finite by construction: every scale is >= 1e-3."""
        d, m = np.empty(len(X), np.float64), np.empty(len(X), np.int16)
        for i in range(0, len(X), chunk):
            x = X[i:i + chunk].astype(np.float64)
            dk = np.sqrt((((x[:, None, :] - self.c[None]) / self.s[None]) ** 2).mean(2))
            m[i:i + chunk] = dk.argmin(1)
            d[i:i + chunk] = dk.min(1)
        assert np.isfinite(d).all()
        return d, m


def hgb(seed):
    """Supervised branch: binary benign vs known attack; sklearn defaults (the paper gives no hyperparameters;
    its built-in early stopping uses a split of the training rows only)."""
    return HistGradientBoostingClassifier(random_state=seed)
