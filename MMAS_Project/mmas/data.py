"""Cached embedding bundles, LLM-description embeddings, run configs and the imbalance subsampler.

Bundle (one per dataset x encoder), as .pt (torch) or .npz:
    emb        N x D   input embeddings
    text_emb   C x D   template prototypes (mean of prompt embeddings per class)
    labels     N       int class index (used only for scoring)
    classnames C       str

Description embeddings (optional, one per dataset x encoder):
    <cache>/<dataset>_<encoder>_desc.pt   {agent: {classname: n_c x D tensor}}   (same layout as v16)
    <cache>/<dataset>_<encoder>_desc.npz  keys "<agent>::<classname>" -> n_c x D
Control variants for failure test F3 use the suffixes _desc_random (random-word descriptions).
"""
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .geometry import unit


def _to_numpy(v):
    if hasattr(v, "detach"):                       # torch tensor
        return v.detach().cpu().float().numpy()
    return np.asarray(v)


def _load_any(path):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    import torch                                   # only needed for .pt bundles
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class Bundle:
    name: str
    Z: np.ndarray                 # N x D, unit rows, float64
    y: np.ndarray                 # N int
    names: list
    T: np.ndarray                 # C x D template prototypes, unit rows
    desc: dict = field(default_factory=dict)          # {variant: {agent: [C arrays n_c x D]}}

    @property
    def C(self):
        return len(self.names)


def load_bundle(path, name=None):
    d = _load_any(path)
    Z = unit(_to_numpy(d["emb"]).astype(np.float64))
    T = unit(_to_numpy(d["text_emb"]).astype(np.float64))
    y = _to_numpy(d["labels"]).astype(int)
    names = [str(n) for n in list(d["classnames"])]
    assert Z.shape[1] == T.shape[1] and len(names) == len(T) and len(y) == len(Z), f"bad bundle {path}"
    return Bundle(name or os.path.basename(path).rsplit(".", 1)[0], Z, y, names, T)


def load_desc(path, names, T):
    """Returns {agent: [C arrays (n_c x D), unit rows]}; classes without descriptions get an empty array."""
    raw = _load_any(path)
    per_agent = {}
    if path.endswith(".npz"):
        for k, v in raw.items():
            agent, cls = k.split("::", 1)
            per_agent.setdefault(agent, {})[cls] = v
    else:
        per_agent = raw
    out = {}
    D = T.shape[1]
    for agent, by_cls in per_agent.items():
        out[agent] = [unit(_to_numpy(by_cls[n]).astype(np.float64).reshape(-1, D)) if n in by_cls
                      else np.empty((0, D)) for n in names]
    return out


def find_file(cache_dir, stems, suffix=""):
    for stem in stems:
        for ext in (".pt", ".npz"):
            p = os.path.join(cache_dir, f"{stem}{suffix}{ext}")
            if os.path.exists(p):
                return p
    return None


def load_run(run, cache_dir):
    """run: one entry of configs/runs.json. Returns a Bundle with any description embeddings attached, or None."""
    stems = [f"{a}_{run['encoder']}" for a in run.get("aliases", [run["dataset"]])]
    path = find_file(cache_dir, stems)
    if path is None:
        return None
    b = load_bundle(path, f"{run['dataset']}/{run['encoder']}")
    for variant, suffix in (("llm", "_desc"), ("random", "_desc_random")):
        p = find_file(cache_dir, stems, suffix)
        if p:
            b.desc[variant] = load_desc(p, b.names, b.T)
    return b


def load_runs_config(path, split="all"):
    runs = json.load(open(path))["runs"]
    return [r for r in runs if split == "all" or r.get("split") == split]


# ------------------------------------------------------------------ imbalance protocol
def skew_idx(y, ir, seed):
    """Geometric decay of class sizes to a ratio `ir` (largest / smallest); class order random by seed.
    Identical to the paper's subsampler."""
    rng = np.random.default_rng(seed)
    C = y.max() + 1
    n0 = np.bincount(y, minlength=C).min()
    order = rng.permutation(C)
    keep = []
    for r, c in enumerate(order):
        ix = rng.permutation(np.flatnonzero(y == c))
        keep.append(ix[:max(1, int(round(n0 * ir ** (-r / (C - 1)))))])
    return np.sort(np.concatenate(keep))


def conditions(irs=(10, 50), seeds=(0, 1, 2), balanced_only=False):
    """[(condition, seed)], seed None = full set as released."""
    out = [("balanced", None)]
    if not balanced_only:
        out += [(f"IR{ir}", s) for ir in irs for s in seeds]
    return out


def subset(y, cond, seed):
    return np.arange(len(y)) if seed is None else skew_idx(y, int(cond[2:]), seed)


def result_files(results_dir, prefix="eval"):
    return sorted(glob.glob(os.path.join(results_dir, f"{prefix}_*.json")))
