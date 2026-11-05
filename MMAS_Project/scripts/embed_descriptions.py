#!/usr/bin/env python
"""Encode LLM class descriptions with the encoder's text tower (GPU, once per dataset x encoder).

  python scripts/embed_descriptions.py --encoder imagebind --modality audio \
      --descriptions descriptions/esc50.json --out cache/esc50_imagebind_desc.npz
  python scripts/embed_descriptions.py ... --descriptions descriptions/esc50_random.json \
      --out cache/esc50_imagebind_desc_random.npz          # F3 control

Output: .npz with keys "<agent>::<classname>" -> n_c x D (or .pt {agent: {classname: tensor}}, the v16 layout).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mmas.encoders import MODALITIES, get_encoder   # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True, choices=["imagebind", "languagebind"])
    p.add_argument("--modality", required=True, choices=MODALITIES, help="selects LanguageBind's tower")
    p.add_argument("--descriptions", required=True, help="JSON from make_descriptions.py")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    agents = json.load(open(a.descriptions))["agents"]
    enc = get_encoder(a.encoder, a.modality, a.device)
    out = {}
    for agent, by_cls in agents.items():
        for cls, texts in by_cls.items():
            if texts:
                out[(agent, cls)] = enc.encode_text(texts, a.modality).astype(np.float32)
        print(f"{agent}: {sum(len(v) for v in by_cls.values())} descriptions over {len(by_cls)} classes")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.out.endswith(".npz"):
        np.savez_compressed(a.out, **{f"{ag}::{c}": v for (ag, c), v in out.items()})
    else:
        import torch
        nested = {}
        for (ag, c), v in out.items():
            nested.setdefault(ag, {})[c] = torch.tensor(v)
        torch.save(nested, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
