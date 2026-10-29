"""Randomised-threshold ROC point (PREREG §9 SEL amendment 1-2); dependency-free so the bowl runs on Kaggle."""
import numpy as np


def roc_detect(s, benign, fpr=0.01):
    """Per-flow detection weight at exactly `fpr` on the benign flows: 1 above the threshold, a shared fraction on
    ties at it (randomised-threshold ROC point), 0 below."""
    q = np.quantile(s[benign], 1 - fpr)
    above, tie = (s[benign] > q).mean(), (s[benign] == q).mean()
    return (s > q) + (s == q) * (max(fpr - above, 0) / tie if tie > 0 else 0)
