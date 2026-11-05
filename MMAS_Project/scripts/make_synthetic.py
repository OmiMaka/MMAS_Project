#!/usr/bin/env python
"""A small synthetic bundle with the two weaknesses measured on the real data (loose anchors, anisotropic
noise) plus LLM-like and random-word description embeddings, to test the whole pipeline on a laptop.

  python scripts/make_synthetic.py
  python scripts/run_experiments.py evaluate --runs configs/runs_synthetic.json
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mmas.geometry import unit   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "cache"))
    p.add_argument("--C", type=int, default=20)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--n", type=int, default=60, help="inputs per class")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    C, D = a.C, a.D

    mu = unit(rng.normal(size=(C, D))) * 0.5
    U, _ = np.linalg.qr(rng.normal(size=(D, D)))
    s = np.full(D, 0.03)
    s[:4] = [0.9, 0.7, 0.5, 0.4]                                  # a few nuisance directions dominate
    y = np.repeat(np.arange(C), a.n)
    Z = unit(mu[y] + (rng.normal(size=(len(y), D)) * s) @ U.T)
    T = unit(unit(mu) + 1.0 * unit(rng.normal(size=(C, D))))      # loose text anchors
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "synthetic_imagebind.npz"), emb=Z, text_emb=T, labels=y,
                        classnames=np.array([f"class_{c:02d}" for c in range(C)]))

    def desc(spread):
        return {f"{ag}::class_{c:02d}": unit(unit(mu[c]) + spread * rng.normal(size=(8, D)) / np.sqrt(D))
                for ag in ("audio_sensor", "scene") for c in range(C)}
    np.savez_compressed(os.path.join(a.out, "synthetic_imagebind_desc.npz"), **desc(4.0))
    np.savez_compressed(os.path.join(a.out, "synthetic_imagebind_desc_random.npz"),
                        **{f"{ag}::class_{c:02d}": unit(rng.normal(size=(8, D)))
                           for ag in ("audio_sensor", "scene") for c in range(C)})
    print(f"saved synthetic bundle to {a.out}: N={len(y)}, C={C}, D={D}")


if __name__ == "__main__":
    main()
