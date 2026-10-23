"""Modified grey wolf optimiser for feature selection (§3.6, Algorithm 2, Functions 1-2, Eqs. 15-37).

Fitness (Eq. 20) = Accuracy - [alpha*(Error + FAR) + (1-alpha)*|subset|/|features|], from an XGBoost
model. The paper only says it is computed on the training data; we hold out 25% of a stratified
training sample, so fitness never sees test or val rows. Runs the fixed iteration count of Table 11.
Each iteration's candidate wolves are scored in parallel, one single-threaded XGBoost each, so the result does
not depend on the core count."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FIT_ROWS = 20_000  # shortcut: fitness on a 20k-row training sample; 50 wolves x 95 iterations of full-size XGBoost is ~days on CPU


def wolf_correlation(wolf, X):
    """Function 2 / Eq. 15: sum_i cov(F1, Fi) / (n * sum_i sd(Fi)) over the wolf's selected features."""
    sel = np.flatnonzero(wolf)
    if len(sel) < 2:
        return 0.0
    F = X[:, sel].astype(np.float64)
    cov = ((F[:, :1] - F[:, :1].mean()) * (F[:, 1:] - F[:, 1:].mean(0))).mean(0)
    return cov.sum() / (len(sel) * F.std(0).sum())


def far(y, pred, k):
    """Eq. 23, one-vs-rest FP/(FP+TN) averaged over classes."""
    out = []
    for c in range(k):
        neg = y != c
        out.append(((pred == c) & neg).sum() / max(neg.sum(), 1))
    return float(np.mean(out))


class Fitness:
    def __init__(self, X, y, alpha, seed):
        idx = np.arange(len(y))
        if len(y) > FIT_ROWS:
            idx = train_test_split(idx, train_size=FIT_ROWS, stratify=y, random_state=seed)[0]
        tr, te = train_test_split(idx, test_size=0.25, stratify=y[idx], random_state=seed)
        self.Xa, self.ya, self.Xb, self.yb = X[tr], y[tr], X[te], y[te]
        self.alpha, self.seed, self.k = alpha, seed, int(y.max()) + 1
        self.cache = {}
        self.pool = Parallel(n_jobs=-1, prefer="threads")  # XGBoost releases the GIL

    def _score(self, wolf):
        if not wolf.any():
            return -np.inf
        cols = np.flatnonzero(wolf)
        m = XGBClassifier(n_estimators=50, max_depth=6, tree_method="hist", n_jobs=1, random_state=self.seed)
        m.fit(self.Xa[:, cols], self.ya)
        pred = m.predict(self.Xb[:, cols])
        acc = float((pred == self.yb).mean())
        pen = self.alpha * ((1 - acc) + far(self.yb, pred, self.k)) + (1 - self.alpha) * len(cols) / len(wolf)
        return acc - pen

    def __call__(self, wolves):
        """Fitness of each wolf (rows of a boolean matrix); new masks are scored in parallel and cached."""
        keys = [w.tobytes() for w in wolves]
        new = {k: w for k, w in zip(keys, wolves) if k not in self.cache}
        self.cache.update(zip(new, self.pool(delayed(self._score)(w) for w in new.values())))
        return np.array([self.cache[k] for k in keys])


def select(X, y, pop, iters, seed, alpha=0.5, max_corr=0.6, log=None):
    """Return the boolean feature mask of the alpha wolf. y must be 0..k-1."""
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    wolves = rng.integers(0, 2, size=(pop, n)).astype(bool)
    wolves = wolves[[wolf_correlation(w, X) < max_corr for w in wolves]]  # Function 1
    fit = Fitness(X, y, alpha, seed)
    score = fit(wolves)
    for t in range(iters):
        a = 2 * (1 - t / iters)
        lead = wolves[np.argsort(-score, kind="stable")[:3]].astype(float)  # alpha, beta, delta
        cand = np.empty_like(wolves)
        for j in range(len(wolves)):  # moves use the leaders fixed at the start of the iteration
            A = 2 * a * rng.random((3, 1)) - a
            C = 2 * rng.random((3, 1))
            cand[j] = (lead - A * np.abs(C * lead - wolves[j])).mean(0) > 0.5  # Eqs. 25-37, §3.6.4
        f = fit(cand)
        better = f > score
        wolves[better], score[better] = cand[better], f[better]
        if log:
            log(t, float(score.max()), int(wolves[score.argmax()].sum()))
    return wolves[score.argmax()]
