"""Label-free covariance-weighted zero-shot classification in a shared embedding space.

Every method starts from an initial score matrix S0 = Z @ A.T (N x C), where A is any anchor matrix
(template prototypes, or templates + LLM descriptions). No function in this module reads labels.

Score (Bayes rule for classes sharing one covariance):
    s_c(z) = mu_c^T Sigma^-1 z - 1/2 mu_c^T Sigma^-1 mu_c,
    Sigma  = (1 - gamma) Sigma_W + gamma tr(Sigma_W)/D I
mu_c and Sigma_W are estimated from soft pseudo-labels Q built from S0.

  WO+P     equal class shares (Sinkhorn), then k-NN smoothing
  PA-WO+P  estimate class shares once, then enforce them
  EM-WO+P  no balance assumption: EM-estimated class prior
           pi_c <- 1/N sum_i softmax_c(s_ic / tau + log pi_c)

Frozen settings (same as the main study): eps=0.02, gamma=0.1, k=10, alpha=0.5.
"""
import numpy as np

GAMMA, EPS, K_NN, ALPHA, TAU_DECIDE, TAU_ZS = 0.1, 0.02, 10, 0.5, 0.1, 0.01


def unit(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def softmax(L):
    L = L - L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


# ------------------------------------------------------------------ mass geometry (pseudo-labels)
def sinkhorn(S, eps=EPS, it=100):
    """Balanced soft assignment: every class receives N/C inputs."""
    L = S / eps
    L = L - L.max(1, keepdims=True)
    Q = np.exp(L)
    N, C = S.shape
    for _ in range(it):
        Q /= Q.sum(1, keepdims=True)
        Q *= (N / C) / Q.sum(0)
    return Q / Q.sum(1, keepdims=True)


def sinkhorn_marg(S, eps, pi, it=100):
    """Soft assignment with prescribed class shares pi (sums to 1)."""
    L = S / eps
    L = L - L.max(1, keepdims=True)
    Q = np.exp(L)
    N = len(S)
    for _ in range(it):
        Q /= Q.sum(1, keepdims=True)
        Q *= (N * pi) / np.maximum(Q.sum(0), 1e-300)
    return Q / Q.sum(1, keepdims=True)


def ot_unbalanced(S, eps=EPS, rho=EPS, iters=200):
    """Unbalanced OT: the class marginal is only softly pulled towards N/C."""
    N, C = S.shape
    L = S / eps
    L = L - L.max(1, keepdims=True)
    K = np.exp(L)
    a, b = np.ones(N), np.ones(C)
    f, target = rho / (rho + eps), N / C
    for _ in range(iters):
        a = 1.0 / np.maximum(K @ b, 1e-300)
        b = (target / np.maximum(K.T @ a, 1e-300)) ** f
    Q = a[:, None] * K * b[None, :]
    return Q / Q.sum(1, keepdims=True)


def em_prior(L, iters=100, floor=1e-4):
    """EM estimate of the class prior pi from logits L; returns (pi, posterior)."""
    N, C = L.shape
    pi = np.full(C, 1.0 / C)
    for _ in range(iters):
        new = np.maximum(softmax(L + np.log(pi)).mean(0), floor)
        new /= new.sum()
        if np.abs(new - pi).max() < 1e-6:
            pi = new
            break
        pi = new
    return pi, softmax(L + np.log(pi))


# ------------------------------------------------------------------ noise geometry (covariance score)
def wo_fit_soft(Z, Q, gamma=GAMMA):
    """Soft class means and shrunk within-class covariance from soft assignment Q; returns (M, Sigma^-1)."""
    N, D = Z.shape
    w = Q.sum(0) + 1e-9
    M = (Q.T @ Z) / w[:, None]
    Sw = (Z.T @ Z - (M.T * w) @ M) / N
    Sw = (1 - gamma) * Sw + gamma * np.trace(Sw) / D * np.eye(D)
    return M, np.linalg.inv(Sw)


def wo_scores(Z, M, Wi):
    MW = M @ Wi
    return Z @ MW.T - 0.5 * np.sum(MW * M, 1)


def whiten_op(Wi):
    lam, U = np.linalg.eigh(np.linalg.inv(Wi))
    return (U / np.sqrt(lam)) @ U.T


# ------------------------------------------------------------------ data geometry (k-NN smoothing)
def knn_graph(X, k=K_NN, chunk=2000):
    N = len(X)
    idx, w = np.empty((N, k), int), np.empty((N, k))
    for s in range(0, N, chunk):
        G = X[s:s + chunk] @ X.T
        G[np.arange(G.shape[0]), np.arange(s, s + G.shape[0])] = -np.inf
        ii = np.argpartition(-G, k, axis=1)[:, :k]
        idx[s:s + chunk] = ii
        w[s:s + chunk] = np.take_along_axis(G, ii, 1)
    return idx, w


def _prop(w, Q, idx):
    out = np.zeros_like(Q)
    for j in range(idx.shape[1]):
        out += w[:, j:j + 1] * Q[idx[:, j]]
    return out


def propagate(Z, Q0, Wi, k=K_NN, alpha=ALPHA, iters=10, power=3):
    """Smooth class probabilities over a k-NN graph built in the whitened space."""
    X = unit((Z - Z.mean(0)) @ whiten_op(Wi))
    idx, w = knn_graph(X, min(k, len(Z) - 1))
    w = np.maximum(w, 0) ** power
    w /= w.sum(1, keepdims=True) + 1e-12
    Q = Q0.copy()
    for _ in range(iters):
        Q = (1 - alpha) * Q0 + alpha * _prop(w, Q, idx)
    return Q


# ------------------------------------------------------------------ pseudo-labels for each variant
def pseudo_labels(S0, prior="balanced"):
    """Soft assignment Q (N x C) from initial scores; prior in {'balanced', 'em'}."""
    if prior == "balanced":
        return sinkhorn(S0)
    if prior == "em":
        return em_prior(S0 / TAU_ZS)[1]
    raise ValueError(prior)


def fit(Z, S0, prior="balanced"):
    """Fit (M, Sigma^-1) on Z from pseudo-labels; used for the inductive split (F4)."""
    return wo_fit_soft(Z, pseudo_labels(S0, prior))


# ------------------------------------------------------------------ the label-free methods, from any S0
def cosine(Z, S0):
    return S0.argmax(1)


def wop(Z, S0):
    """WO+P: equal class shares."""
    M, Wi = wo_fit_soft(Z, sinkhorn(S0))
    L = wo_scores(Z, M, Wi)
    return propagate(Z, sinkhorn(L / L.std(), TAU_DECIDE), Wi).argmax(1)


def pa_wop(Z, S0, smooth=1.0):
    """PA-WO+P: estimate class shares once, then balance to them."""
    C = S0.shape[1]
    M, Wi = wo_fit_soft(Z, ot_unbalanced(S0))
    pi = (np.bincount(wo_scores(Z, M, Wi).argmax(1), minlength=C) + smooth) / (len(Z) + smooth * C)
    M, Wi = wo_fit_soft(Z, sinkhorn_marg(S0, EPS, pi))
    L = wo_scores(Z, M, Wi)
    return propagate(Z, softmax(L / L.std() / TAU_DECIDE), Wi).argmax(1)


def em_wop(Z, S0):
    """EM-WO+P: never assumes balance (EM-estimated prior on the initial scores)."""
    M, Wi = wo_fit_soft(Z, em_prior(S0 / TAU_ZS)[1])
    L = wo_scores(Z, M, Wi)
    return propagate(Z, softmax(L / L.std() / TAU_DECIDE), Wi).argmax(1)


METHODS = {"cosine": cosine, "WO+P": wop, "PA-WO+P": pa_wop, "EM-WO+P": em_wop}
