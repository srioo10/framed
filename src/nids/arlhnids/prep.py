"""ARLHNIDS §3.3 preprocessing and Eq. 2. `fit` sees whatever rows the caller passes:
the merged train+test set in paper order, the training split in leak-free order."""
import numpy as np
import pandas as pd


class Prep:
    """Label encode -> mean impute -> clamp (max > 10*median: values above q95 set to q95) -> standard scale."""

    def fit(self, df):
        self.cols = list(df.columns)
        self.codes = {c: {v: i for i, v in enumerate(pd.unique(df[c].astype(str)))}
                      for c in self.cols if not pd.api.types.is_numeric_dtype(df[c])}
        X = self._encode(df)
        self.mean = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), self.mean, X)
        q95, med, mx = np.quantile(X, 0.95, axis=0), np.median(X, axis=0), X.max(axis=0)
        self.cap = np.where(mx > 10 * med, q95, np.inf)
        X = np.minimum(X, self.cap)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def _encode(self, df):
        X = np.empty((len(df), len(self.cols)), dtype=np.float64)
        for j, c in enumerate(self.cols):
            if c in self.codes:  # unseen categories at transform time -> NaN -> mean
                X[:, j] = df[c].astype(str).map(self.codes[c]).to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                X[:, j] = df[c].to_numpy(dtype=np.float64, na_value=np.nan)
        X[~np.isfinite(X)] = np.nan
        return X

    def transform(self, df):
        X = self._encode(df)
        X = np.where(np.isnan(X), self.mean, X)
        return ((np.minimum(X, self.cap) - self.mu) / self.sd).astype(np.float32)


def majority_classes(counts):
    """Eq. 2: class x is majority iff |I_major| - |I_x| <= 0.5 * |I_major|."""
    top = max(counts.values())
    return {c for c, n in counts.items() if top - n <= 0.5 * top}
