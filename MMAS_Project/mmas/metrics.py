"""Scoring (the only place labels are used) and the geometry diagnostics from the problem statement."""
import numpy as np

from .geometry import unit


def top1(pred, y):
    return 100.0 * float(np.mean(pred == y))


def mean_recall(pred, y):
    """Per-class mean recall (%), over classes present in y; the fair metric under imbalance."""
    cls = np.unique(y)
    return 100.0 * float(np.mean([np.mean(pred[y == c] == c) for c in cls]))


def mcnemar(ok_a, ok_b):
    """Exact McNemar test between two per-sample correctness vectors: (fixed, broken, p)."""
    from scipy.stats import binomtest
    fixed = int(np.sum(~ok_a & ok_b))
    broken = int(np.sum(ok_a & ~ok_b))
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    return fixed, broken, float(p)


# ------------------------------------------------------------------ diagnostics (use labels; reporting only)
def anchor_gap(Z, y, T):
    """cos(t_c, mu_c) per class: how well the text prototype points at its class centre."""
    C = len(T)
    mu = unit(np.stack([Z[y == c].mean(0) if np.any(y == c) else np.zeros(Z.shape[1]) for c in range(C)]))
    return np.sum(mu * T, 1)


def anisotropy(Z, y):
    """lambda_max / lambda_median of the pooled within-class covariance."""
    R = Z - np.stack([Z[y == c].mean(0) for c in range(y.max() + 1)])[y]
    lam = np.clip(np.linalg.eigvalsh(R.T @ R / len(R)), 0, None)
    return float(lam[-1] / max(np.median(lam), 1e-300))
