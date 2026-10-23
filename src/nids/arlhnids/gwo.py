"""Modified grey wolf optimiser for feature selection (§3.6, Algorithm 2, Functions 1-2, Eqs. 15-37).

Fitness (Eq. 20) = Accuracy - [alpha*(Error + FAR) + (1-alpha)*|subset|/|features|], from an XGBoost
model. The paper only says it is computed on the training data; we hold out 25% of a stratified
training sample, so fitness never sees test or val rows. Runs the fixed iteration count of Table 11."""
import numpy as np
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
        self.tr, self.te = train_test_split(idx, test_size=0.25, stratify=y[idx], random_state=seed)
        self.X, self.y, self.alpha, self.seed, self.k = X, y, alpha, seed, int(y.max()) + 1
        self.cache = {}

    def __call__(self, wolf):
        key = wolf.tobytes()
        if key not in self.cache:
            if not wolf.any():
                self.cache[key] = -np.inf
            else:
                cols = np.flatnonzero(wolf)
                m = XGBClassifier(n_estimators=50, max_depth=6, tree_method="hist", n_jobs=-1,
                                  random_state=self.seed)
                m.fit(self.X[np.ix_(self.tr, cols)], self.y[self.tr])
                pred = m.predict(self.X[np.ix_(self.te, cols)])
                acc = float((pred == self.y[self.te]).mean())
                pen = self.alpha * ((1 - acc) + far(self.y[self.te], pred, self.k)) \
                    + (1 - self.alpha) * len(cols) / len(wolf)
                self.cache[key] = acc - pen
        return self.cache[key]


def select(X, y, pop, iters, seed, alpha=0.5, max_corr=0.6, log=None):
    """Return the boolean feature mask of the alpha wolf. y must be 0..k-1."""
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    wolves = rng.integers(0, 2, size=(pop, n)).astype(bool)
    wolves = wolves[[wolf_correlation(w, X) < max_corr for w in wolves]]  # Function 1
    fit = Fitness(X, y, alpha, seed)
    score = np.array([fit(w) for w in wolves])
    for t in range(iters):
        a = 2 * (1 - t / iters)
        lead = wolves[np.argsort(-score, kind="stable")[:3]].astype(float)  # alpha, beta, delta
        for j in range(len(wolves)):
            A = 2 * a * rng.random((3, 1)) - a
            C = 2 * rng.random((3, 1))
            x_new = (lead - A * np.abs(C * lead - wolves[j])).mean(0)  # Eqs. 25-37
            cand = x_new > 0.5                                     # §3.6.4
            f = fit(cand)
            if f > score[j]:
                wolves[j], score[j] = cand, f
        if log:
            log(t, float(score.max()), int(wolves[score.argmax()].sum()))
    return wolves[score.argmax()]
