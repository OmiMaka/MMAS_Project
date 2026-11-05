#!/usr/bin/env python
"""Build one cached bundle (GPU, run once per dataset x encoder; everything after this is CPU).

Input: a CSV manifest with columns `path,label` (label = class name), and optionally a class-name file
(one name per line, fixing the class order; otherwise sorted label names are used).

  python scripts/extract_embeddings.py --encoder imagebind --modality audio \
      --manifest manifests/esc50.csv --out cache/esc50_imagebind.pt

Output keys: emb (N x D), text_emb (C x D, unit mean of the encoded templates), labels (N), classnames (C).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mmas.encoders import MODALITIES, get_encoder   # noqa: E402
from mmas.geometry import unit                       # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True, choices=["imagebind", "languagebind"])
    p.add_argument("--modality", required=True, choices=MODALITIES)
    p.add_argument("--manifest", required=True)
    p.add_argument("--classnames", help="text file, one class name per line (defines the class index)")
    p.add_argument("--templates", default=os.path.join(ROOT, "configs", "templates.json"))
    p.add_argument("--out", required=True, help=".pt (needs torch) or .npz")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.manifest)))
    names = [l.strip() for l in open(a.classnames) if l.strip()] if a.classnames else sorted({r["label"] for r in rows})
    idx = {n: i for i, n in enumerate(names)}
    paths = [r["path"] for r in rows]
    labels = np.array([idx[r["label"]] for r in rows])
    templates = json.load(open(a.templates))[a.modality]

    enc = get_encoder(a.encoder, a.modality, a.device)
    print(f"encoding {len(paths):,} {a.modality} inputs and {len(names)} x {len(templates)} prompts with {a.encoder}")
    emb = enc.encode_inputs(paths, a.modality)
    T = np.stack([unit(unit(enc.encode_text([t.format(n.replace('_', ' ')) for t in templates], a.modality)).mean(0))
                  for n in names])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.out.endswith(".npz"):
        np.savez_compressed(a.out, emb=emb.astype(np.float32), text_emb=T.astype(np.float32), labels=labels,
                            classnames=np.array(names))
    else:
        import torch
        torch.save({"emb": torch.tensor(emb), "text_emb": torch.tensor(T), "labels": torch.tensor(labels),
                    "classnames": names, "templates": templates, "encoder": a.encoder}, a.out)
    print(f"saved {a.out}: emb {emb.shape}, text_emb {T.shape}")


if __name__ == "__main__":
    main()
