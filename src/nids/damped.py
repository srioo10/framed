"""Damped (AfterImage-style) per-key statistics and their deletion box (reviewer round Nov 13, item 3).

Per key, a flow i at time t_i with feature g_i in [0, G] contributes weight e^{-lam (t - t_i)} at decision time t:
    w = sum e^{-lam a_i},  LS = sum e^{-lam a_i} g_i,  SS = sum e^{-lam a_i} g_i^2   (a_i = t - t_i >= 0)
and the features are w, mean = LS / w and std = sqrt(SS / w - mean^2).

Deletion box. If at most m flows per key are injected in any `win` seconds, injected flows add at most
    R_max = m * S,  S = sum_{k>=0} e^{-lam win k} = 1 / (1 - e^{-lam win})
to w (group the injections by age window [k win, (k+1) win); each group has <= m flows of weight <= e^{-lam k win}).
With r = removed weight in [0, min(R_max, w')] and removed LS in [0, G r], the clean mean
(LS' - a) / (w' - r) is minimised at a = G r, r = R and maximised at a = 0, r = R (both monotone in r because
LS' <= G w'), R = min(R_max, w'). The same argument with G^2 bounds SS / w; std follows by interval arithmetic.
"""
import numpy as np


def stats(ages, g, lam):
    """w, mean, std of a key's flows (ages in seconds, g values)."""
    e = np.exp(-lam * np.asarray(ages, float))
    g = np.asarray(g, float)
    w = e.sum()
    if w <= 0:
        return 0.0, 0.0, 0.0
    mu = (e * g).sum() / w
    return w, mu, float(np.sqrt(max((e * g * g).sum() / w - mu * mu, 0.0)))


def flip_weight(lam, m, win=240.0, look=120.0):
    """Max weight of real flows whose novelty flag an in-budget injection can flip from new to old. An injection
    pre-touches a port / peer, so the next real flow on it within `look` s looks old; each injection flips at most one
    real flow, and that flow is at most `look` s younger than it. Injections of age window k ([k win, (k+1) win))
    are <= m and flip flows of age >= k win - look."""
    ks = np.arange(0, 2000)
    return m * float(np.exp(-lam * np.maximum(0.0, ks * win - look)).sum())


def box(w, mu, sd, lam, m, G, win=240.0, novelty=False):
    """Intervals (lo, hi) for clean w, mean, std given the observed (w', mean', std'); G may be np.inf.
    novelty=True: g is a 0/1 new-port / new-peer flag, which injections can also flip downward on real flows, so the
    clean LS can rise by up to G * flip_weight (Lemma 1's b-side, for damped statistics)."""
    S = 1.0 / (1.0 - np.exp(-lam * win))
    R = min(m * S, w)
    F = flip_weight(lam, m, win) if novelty else 0.0
    LS, SS = mu * w, (sd * sd + mu * mu) * w
    wr = w - R
    if wr <= 1e-12:  # the clean key may have no flows left: nothing is known beyond the feature range
        mlo, mhi, e2lo, e2hi = 0.0, G, 0.0, G * G
    else:
        mlo, mhi = max((LS - G * R) / wr, 0.0) if np.isfinite(G) else 0.0, min((LS + G * F) / wr, G)
        e2lo, e2hi = max((SS - G * G * R) / wr, 0.0) if np.isfinite(G) else 0.0, min((SS + G * G * F) / wr, G * G)
    vlo = max(e2lo - mhi * mhi, 0.0)
    vhi = max(e2hi - mlo * mlo, 0.0)
    return (max(w - R, 0.0), w), (mlo, mhi), (np.sqrt(vlo), np.sqrt(vhi))
