"""Metrics and statistical tests used in Chapters 3 and 4.

accuracy, macro-F1 (Equation 3.14), per-class recall, confusion matrix, expected calibration error (Equation 2.11),
percentile bootstrap intervals (Equation 4.2) and McNemar's test (Equation 4.1).
"""
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from .labels import K


def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0))


def per_class_recall(y, pred):
    return recall_score(y, pred, labels=list(range(K)), average=None, zero_division=0)


def confusion(y, pred, normalize=None):
    return confusion_matrix(y, pred, labels=list(range(K)), normalize=normalize)


def ece(y, probs, n_bins=15):
    probs = np.asarray(probs)
    conf = probs.max(1)
    correct = (probs.argmax(1) == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def bootstrap_ci(metric, y, pred, B=1000, seed=0, alpha=0.05):
    """Percentile bootstrap over test images. Returns (point, low, high)."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = metric(y[idx], pred[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, pred)), float(lo), float(hi)


def mcnemar(y, pred1, pred2):
    """McNemar's test on paired predictions. Exact binomial when discordant pairs < 25, else chi-square with
    continuity correction. n01: system 1 wrong and system 2 right; n10: the reverse."""
    y, a, b = map(np.asarray, (y, pred1, pred2))
    c1, c2 = a == y, b == y
    n01 = int(np.sum(~c1 & c2))
    n10 = int(np.sum(c1 & ~c2))
    n = n01 + n10
    if n == 0:
        return dict(n01=0, n10=0, statistic=0.0, p_value=1.0, method="none")
    if n < 25:
        p = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
        return dict(n01=n01, n10=n10, statistic=float(min(n01, n10)), p_value=p, method="exact")
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return dict(n01=n01, n10=n10, statistic=float(chi2), p_value=float(stats.chi2.sf(chi2, 1)), method="chi2")
