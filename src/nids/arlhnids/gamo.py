"""Modified GAMO oversampling (§3.4, Fig. 4, Table 6, Eqs. 3-12): GAMO (Mullick et al., 2019) with attention.

Generator G = conditional feature mapping unit (noise + one-hot label -> 32-d) and one convex sample generator
unit (SGU) per oversampled class. An SGU outputs softmax weights over that class's real training rows, so every
synthetic row is a convex combination of real rows of its class. D tells real from fake given the label; M
classifies. Three-player game of Eq. 5: D and M maximise, G fools D and pushes M to misclassify its samples.

Only the classes passed in enter the module (Table 5 minority classes); 75% of each is the training/convex
basis, as Table 7 implies (e.g. UNSW Worms 130 of 174), the other 25% scores M. Rows are min-max scaled
(Eq. 3) for training; convex combinations commute with that affine map, so output is in the input's space.

Attention: Table 6 pairs each "Attention n" with a preceding "Dense n softmax"; we read the pair as one soft
feature gate, x[:n] * softmax(W x + b) (the usual tabular attention block). Read literally, a softmax layer
squashes the input and the generator collapses to one row per class (checked on UNSW)."""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import nn

Z = 32           # Table 6 noise length
STEPS = 12000    # the paper reports no epochs/batch/lr; fixed budget, same in both orders (3k left M at 0.43 on UNSW)
BATCH = 256
LR = 2e-4


class Attention(nn.Module):
    """Dense n softmax + Attention n: gate the first n inputs (the features, not the label) by softmax scores."""
    def __init__(self, d_in, n):
        super().__init__()
        self.n, self.score = n, nn.Linear(d_in, n)

    def forward(self, x):
        return x[:, :self.n] * torch.softmax(self.score(x), dim=1)


def _head(d_in, d, out):
    """Table 6 discriminator / classifier: Dense d softmax + Attention d, 64 and 32 LeakyReLU, output."""
    return nn.Sequential(Attention(d_in, d), nn.Linear(d, 64), nn.LeakyReLU(0.2),
                         nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, out))


class Generator(nn.Module):
    def __init__(self, c, bases):
        """bases: {class index: (n_i, d) tensor of that class's real basis rows} for each oversampled class."""
        super().__init__()
        self.c = c
        self.cfmu = nn.Sequential(nn.Linear(Z + c, 32), Attention(32, 32),
                                  nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU())
        self.bases = bases
        # SGU: Dense 32 softmax + Attention 32, Dense n_i softmax (text §3.4.3), convex combination of the basis
        self.sgu = nn.ModuleDict({str(i): nn.Sequential(Attention(32, 32), nn.Linear(32, len(b)), nn.Softmax(dim=1))
                                  for i, b in bases.items()})

    def forward(self, z, y):
        h = self.cfmu(torch.cat([z, nn.functional.one_hot(y, self.c).float()], 1))
        out = torch.empty(len(y), next(iter(self.bases.values())).shape[1])
        for i, b in self.bases.items():
            s = y == i
            if s.any():
                out[s] = self.sgu[str(i)](h[s]) @ b
        return out


def oversample(X, y, targets, seed, log=None):
    """X, y: rows of the classes that enter GAMO. targets: {class: final count} for classes to grow.
    Returns (X_out, y_out, info) with synthetic rows appended after the real ones, like imblearn."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    code = {c: i for i, c in enumerate(classes)}
    yi = np.array([code[v] for v in y])
    lo, span = X.min(0), np.maximum(X.max(0) - X.min(0), 1e-12)
    Xs = ((X - lo) / span).astype(np.float32)  # Eq. 3
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=yi, random_state=seed)
    Xtr, ytr = torch.from_numpy(Xs[tr]), torch.from_numpy(yi[tr])
    need = {code[c]: t - int((y == c).sum()) for c, t in targets.items() if t > (y == c).sum()}
    bases = {i: Xtr[ytr == i] for i in need}
    c, d = len(classes), X.shape[1]

    G = Generator(c, bases)
    D = _head(d + c, d, 1)
    M = _head(d, d, c)
    opt = {k: torch.optim.Adam(m.parameters(), lr=LR, betas=(0.5, 0.999)) for k, m in dict(G=G, D=D, M=M).items()}
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    fake_p = np.array([need[i] for i in need], float)
    fake_p /= fake_p.sum()  # Eqs. 9, 12: fake class weight (P_c - P_i), proportional to rows still needed
    fake_cls = np.array(list(need))

    def onehot(v):
        return nn.functional.one_hot(v, c).float()

    for step in range(STEPS):
        b = torch.from_numpy(rng.integers(0, len(ytr), BATCH))
        xr, yr = Xtr[b], ytr[b]
        yf = torch.from_numpy(rng.choice(fake_cls, BATCH, p=fake_p))
        xf = G(torch.rand(BATCH, Z), yf)  # "non-negative real numbers" noise
        # D and M (Eqs. 7-8, 11-12 / 9-10): real -> 1 / own class, fake -> 0 / own class
        lD = bce(D(torch.cat([xr, onehot(yr)], 1)), torch.ones(BATCH, 1)) \
            + bce(D(torch.cat([xf.detach(), onehot(yf)], 1)), torch.zeros(BATCH, 1))
        lM = ce(M(xr), yr) + ce(M(xf.detach()), yf)
        for k, l in (("D", lD), ("M", lM)):
            opt[k].zero_grad()
            l.backward()
            opt[k].step()
        # G: fool D, and make M's score for the true class low (-log(1 - M_i), bounded form of Eq. 9)
        pm = torch.softmax(M(xf), 1).gather(1, yf[:, None])
        lG = bce(D(torch.cat([xf, onehot(yf)], 1)), torch.ones(BATCH, 1)) - torch.log1p(-pm + 1e-6).mean()
        opt["G"].zero_grad()
        lG.backward()
        opt["G"].step()
        if log and (step % 500 == 0 or step == STEPS - 1):
            log(step, lD.item(), lM.item(), lG.item())

    G.eval()
    M.eval()
    with torch.no_grad():
        m_acc = float((M(torch.from_numpy(Xs[te])).argmax(1).numpy() == yi[te]).mean())
        Xg, yg = [], []
        for i, n in need.items():
            for s in range(0, n, 4096):
                k = min(4096, n - s)
                Xg.append(G(torch.rand(k, Z), torch.full((k,), i)).numpy())
                yg.append(np.full(k, classes[i]))
    Xg, yg = np.concatenate(Xg), np.concatenate(yg)
    info = {"gamo_m_holdout_acc": m_acc, "gamo_basis": {str(classes[i]): len(b) for i, b in bases.items()},
            "gamo_diversity": {str(classes[i]): _diversity(b.numpy(), Xg[yg == classes[i]], rng) for i, b in bases.items()}}
    Xg = Xg * span + lo
    return np.concatenate([X, Xg]).astype(np.float32), np.concatenate([y, yg]), info


def _diversity(basis, fake, rng, n=2000):
    """Collapse check in scaled space: spread = median distance to the class mean (fake / real), and median
    distance fake -> nearest basis row vs real -> nearest other basis row."""
    fake = fake[rng.choice(len(fake), min(n, len(fake)), replace=False)]
    nn = NearestNeighbors(n_neighbors=2).fit(basis)
    spread = lambda a: float(np.median(np.linalg.norm(a - a.mean(0), axis=1)))
    return {"spread_ratio": spread(fake) / max(spread(basis), 1e-12),
            "fake_to_real_nn": float(np.median(nn.kneighbors(fake, 1)[0])),
            "real_to_real_nn": float(np.median(nn.kneighbors(basis[:n])[0][:, 1])) if len(basis) > 1 else None,
            "unique_share": len(np.unique(fake.round(4), axis=0)) / len(fake)}
