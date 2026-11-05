"""Anchor matrices (C x D) that produce the initial scores S0 = Z @ A.T.

  T                 the papers' template prototypes
  T+D               unit(T + mean description embedding), pooled over all LLM agents   (as in v11 / Table 2)
  T+D:<agent>       same, one agent only (e.g. sound agent vs. visual-scene agent)
  T+D:shuffled      F3 control: descriptions permuted across classes (same seed every time)
  T+D:random        F3 control: random-word descriptions of the same length (needs *_desc_random)
"""
import numpy as np

from .geometry import unit


def pooled_desc_mean(per_agent, T, agents=None):
    """Per class: unit mean over the description embeddings of the chosen agents (falls back to the template)."""
    agents = list(per_agent) if agents is None else agents
    rows = []
    for c in range(len(T)):
        parts = [per_agent[a][c] for a in agents if len(per_agent[a][c])]
        rows.append(unit(np.concatenate(parts)).mean(0) if parts else T[c])
    return unit(np.stack([unit(r) for r in rows]))


def templates_plus(T, per_agent, agents=None):
    return unit(T + pooled_desc_mean(per_agent, T, agents))


def shuffled(per_agent, seed=0):
    """Same descriptions, wrong classes: a permutation with no fixed points."""
    C = len(next(iter(per_agent.values())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(C)
    while C > 1 and np.any(perm == np.arange(C)):
        perm = rng.permutation(C)
    return {a: [v[perm[c]] for c in range(C)] for a, v in per_agent.items()}


def build_anchors(bundle, controls=False):
    """{name: C x D anchor matrix} for every anchor available for this bundle."""
    A = {"T": bundle.T}
    llm = bundle.desc.get("llm")
    if llm:
        A["T+D"] = templates_plus(bundle.T, llm)
        if len(llm) > 1:
            for agent in llm:
                A[f"T+D:{agent}"] = templates_plus(bundle.T, llm, [agent])
        if controls:
            A["T+D:shuffled"] = templates_plus(bundle.T, shuffled(llm))
    if controls and bundle.desc.get("random"):
        A["T+D:random"] = templates_plus(bundle.T, bundle.desc["random"])
    return A


def agreement(Z, A1, A2):
    """Fraction of inputs whose cosine argmax is the same under two anchor sets (label-free)."""
    return float(np.mean((Z @ A1.T).argmax(1) == (Z @ A2.T).argmax(1)))
