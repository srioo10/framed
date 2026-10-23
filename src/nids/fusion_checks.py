"""Score-bank diagnostics for budgeted fusion (Stage 2 ceiling check + Stage 5 rehearsal), PREREG §9 H5.

All functions work on per-member score matrices of shape (N, M), higher = more anomalous.
`names` lists the M member columns, e.g. ["rf", "xgb", "mlp_ce", "iforest", "clust_global"].
b_rank / b_cal must be the group-stratified halves of benign_val (stime//600 blocks on unsw_raw,
file blocks on CICIoT; random on unsw_official), never a positional split.
"""
import numpy as np


def _finite(a, what):
    a = np.asarray(a)
    if not np.isfinite(a).all():
        raise ValueError(f"{what}: {int((~np.isfinite(a)).sum())} non-finite scores; fix them at the source")
    return a


def tail_pvalues(ref, x):
    """p = (1 + #{ref >= x}) / (n + 1), per member. O((n + N) log n) memory-safe."""
    ref, x = _finite(ref, "ref"), _finite(x, "x")
    n, M = ref.shape
    out = np.empty(x.shape, dtype=np.float64)
    for m in range(M):
        r = np.sort(ref[:, m])
        ge = n - np.searchsorted(r, x[:, m], side="left")
        out[:, m] = (1.0 + ge) / (n + 1.0)
    return out


def tail_pvalues_by_mode(ref, ref_mode, x, x_mode, n_min):
    """1-D tail p-values against the reference rows of the same mode (MCDE §3.3); a mode with fewer than
    `n_min` reference rows, or unseen in the reference, uses the pooled reference."""
    ref, x = _finite(ref, "ref"), _finite(x, "x")
    out = tail_pvalues(ref[:, None], x[:, None])[:, 0]
    counts = np.bincount(ref_mode, minlength=int(max(ref_mode.max(), x_mode.max())) + 1)
    for k in np.flatnonzero(counts >= n_min):
        q = x_mode == k
        if q.any():
            out[q] = tail_pvalues(ref[ref_mode == k][:, None], x[q][:, None])[:, 0]
    return out


def threshold(scores_cal, fpr):
    """Strict '>' rule with the higher quantile: at most `fpr` of cal scores exceed tau.
    Needed because RF/XGB scores are piecewise-constant and tie heavily in the tail."""
    scores_cal = _finite(scores_cal, "scores_cal")
    n = scores_cal.shape[0]
    if n * fpr < 10 - 1e-9:
        raise ValueError(f"n_cal = {n} < 10 / fpr = {10 / fpr:.0f}: threshold has no support")
    return np.quantile(scores_cal, 1.0 - fpr, axis=0, method="higher")


def budget_fusion_score(p, shares):
    """S(x) = -min_m p_m(x) / share_m. A zero share removes that member."""
    shares = np.asarray(shares, dtype=np.float64)
    with np.errstate(divide="ignore"):
        ratio = np.where(shares > 0, p / shares, np.inf)
    return -ratio.min(axis=1)


def union_ceiling_check(b_rank, b_cal, attack, names, tree_cols=("rf", "xgb"), fpr=0.01):
    """Per-family headroom diagnostic. Not a true upper bound: a recalibrated or weighted
    fusion can exceed the Bonferroni union. Read it as 'is there anything to fuse here'."""
    M = b_rank.shape[1]
    b_all = np.vstack([b_rank, b_cal])

    tau_full = threshold(b_all, fpr)
    single = (attack > tau_full).mean(axis=0)
    best_tree = single[[names.index(c) for c in tree_cols]].max()
    best_any = single.max()

    tau_split = threshold(b_all, fpr / M)
    union_tpr = (attack > tau_split).any(axis=1).mean()
    union_fpr = (b_all > tau_split).any(axis=1).mean()

    p_cal, p_atk = tail_pvalues(b_rank, b_cal), tail_pvalues(b_rank, attack)
    eq = np.full(M, 1.0 / M)
    s_cal, s_atk = budget_fusion_score(p_cal, eq), budget_fusion_score(p_atk, eq)
    tau = threshold(s_cal, fpr)
    fused_tpr = (s_atk > tau).mean()

    best_fusion = max(union_tpr, fused_tpr)
    return {
        "single_member_tpr": dict(zip(names, np.round(single, 4).tolist())),
        "best_tree_tpr": round(float(best_tree), 4),
        "best_any_member_tpr": round(float(best_any), 4),
        "bonferroni_union_tpr": round(float(union_tpr), 4),
        "bonferroni_union_fpr": round(float(union_fpr), 4),  # shows how conservative the split is
        "recalibrated_equal_fusion_tpr": round(float(fused_tpr), 4),
        "recalibrated_cal_fpr": round(float((s_cal > tau).mean()), 4),
        "headroom_vs_trees": round(float(best_fusion - best_tree), 4),    # the Stage 2 gate
        "headroom_vs_best_member": round(float(best_fusion - best_any), 4),  # does fusion beat picking one member
    }


def shares_from_alpha(alpha_sup, names,
                      sup=("rf", "xgb", "mlp_ce"), dev=("iforest", "clust_global")):
    """One free constant: alpha_sup goes to the supervised block, the rest to the
    benign-deviation block, split equally inside each block."""
    a = np.zeros(len(names))
    s = [names.index(c) for c in sup if c in names]
    d = [names.index(c) for c in dev if c in names]
    if not s:
        raise ValueError("no supervised member in names")
    a[s] = alpha_sup / len(s)
    if d:
        a[d] = (1.0 - alpha_sup) / len(d)
    return a


def pick_alpha(grid, per_family, min_tol=0.005, prior=0.9):
    """PREREG §9 tie rule. per_family: (G, F) TPR per alpha and eligible inner family.
    Tied = within max(min_tol, paired SE) of the best macro, paired SE = std_f(TPR_a - TPR_best) / sqrt(F).
    Among ties: highest harmonic mean of (TPR + 0.05), then closest to `prior`. flat = every alpha tied."""
    grid = np.asarray(grid, dtype=np.float64)
    G, F = per_family.shape
    macro = per_family.mean(axis=1)
    best = int(macro.argmax())
    diff = per_family - per_family[best]
    se = diff.std(axis=1, ddof=1) / np.sqrt(F) if F > 1 else np.zeros(G)
    tol = np.maximum(min_tol, se)
    tied = macro[best] - macro <= tol + 1e-12
    hm = F / (1.0 / (per_family + 0.05)).sum(axis=1)
    cand = np.flatnonzero(tied)
    top = cand[np.isclose(hm[cand], hm[cand].max())]
    pick = int(top[np.abs(grid[top] - prior).argmin()])
    return float(grid[pick]), {"tied": tied, "tol": tol, "harmonic": hm, "flat": bool(tied.all())}


def rehearse_alpha(inner, names, grid=None, fpr=0.01):
    """inner: list of (b_rank, b_cal, attack_j) scored by the pair model trained on F minus {k, j},
    one entry per eligible pseudo-unknown family j of outer fold k. Picks alpha_sup by macro TPR@fpr
    with the PREREG tie rule. Returns (alpha, macro curve, per-family TPR matrix, tie info)."""
    if grid is None:
        grid = np.round(np.arange(0.30, 1.0001, 0.05), 2)
    pre = [(tail_pvalues(br, bc), tail_pvalues(br, aj)) for br, bc, aj in inner]
    per_family = np.zeros((len(grid), len(pre)))
    for gi, a in enumerate(grid):
        w = shares_from_alpha(a, names)
        for fi, (p_cal, p_atk) in enumerate(pre):
            tau = threshold(budget_fusion_score(p_cal, w), fpr)
            per_family[gi, fi] = (budget_fusion_score(p_atk, w) > tau).mean()
    alpha, info = pick_alpha(grid, per_family)
    return alpha, per_family.mean(axis=1), per_family, info
