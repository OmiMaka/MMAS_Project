"""Label-free policy: choose the variant and the anchors per dataset from the test data alone.

  1. Estimate class shares pi_hat without labels (EM on the template scores, or covariance-score counts).
  2. pi_hat_max / pi_hat_min < RATIO_THR  ->  WO+P   (classes look balanced)
     otherwise                            ->  EM-WO+P
  3. Use the LLM anchors (T+D) only if their cosine predictions agree with the templates' on at least
     AGREE_THR of the inputs; AGREE_THR is set on the development sets (scripts/run_experiments.py calibrate).
"""
import numpy as np

from .anchors import agreement
from .geometry import TAU_ZS, em_prior, em_wop, ot_unbalanced, wo_fit_soft, wo_scores, wop

RATIO_THR = 2.0
AGREE_THR = 0.5          # placeholder until `calibrate` has been run on the development sets
ESTIMATORS = ("em", "em_trim", "wo")


def class_shares(Z, S0, estimator="em", smooth=1.0):
    """Label-free estimate of the class shares pi_hat.
      em       EM prior on the anchor scores (the proposal's rule). Biased by the anchor gap: it can report a
               large ratio on balanced data.
      em_trim  same pi_hat; the ratio is then taken between its 5% / 95% quantiles (see size_ratio)
      wo       counts of covariance-score argmax after an unbalanced-OT fit (the PA-WO+P estimate)"""
    if estimator in ("em", "em_trim"):
        return em_prior(S0 / TAU_ZS)[0]
    if estimator == "wo":
        C = S0.shape[1]
        M, Wi = wo_fit_soft(Z, ot_unbalanced(S0))
        return (np.bincount(wo_scores(Z, M, Wi).argmax(1), minlength=C) + smooth) / (len(Z) + smooth * C)
    raise ValueError(estimator)


def size_ratio(Z, S0, estimator="em"):
    """pi_hat_max / pi_hat_min (em_trim: 95% / 5% quantile ratio, steadier with many classes)."""
    pi = class_shares(Z, S0, estimator)
    if estimator == "em_trim":
        lo, hi = np.quantile(pi, [0.05, 0.95])
        return float(hi / max(lo, 1e-12))
    return float(pi.max() / pi.min())


def decide(Z, T, TD=None, ratio_thr=RATIO_THR, agree_thr=AGREE_THR, estimator="em"):
    ratio = size_ratio(Z, Z @ T.T, estimator)
    info = {"ratio": ratio, "estimator": estimator, "method": "WO+P" if ratio < ratio_thr else "EM-WO+P", "anchor": "T"}
    if TD is not None:
        info["agreement"] = agreement(Z, T, TD)
        if info["agreement"] >= agree_thr:
            info["anchor"] = "T+D"
    return info


def predict(Z, T, TD=None, **kw):
    info = decide(Z, T, TD, **kw)
    A = TD if info["anchor"] == "T+D" else T
    f = wop if info["method"] == "WO+P" else em_wop
    return f(Z, Z @ A.T), info
