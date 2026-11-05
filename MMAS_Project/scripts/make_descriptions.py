#!/usr/bin/env python
"""LLM class descriptions from two agents, written once and versioned in descriptions/ (so later runs need no API).

  sensor agent   what the class is like in the input modality (how it sounds; its heat signature for thermal)
  scene agent    the visual scene in which the class typically occurs

  export ANTHROPIC_API_KEY=...            # or `ant auth login`
  python scripts/make_descriptions.py --dataset esc50 --modality audio --classnames classnames/esc50.txt
  python scripts/make_descriptions.py --dataset esc50 --random-control     # F3: random-word twin, no API

Output: descriptions/<dataset>.json  {"model": ..., "agents": {agent: {classname: [sentence, ...]}}}
The file is rewritten after every class, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "claude-opus-5-5"

SENSOR = {
    "audio": "You describe what things SOUND like. Each description is one short sentence about the acoustic "
             "character of the sound: pitch, rhythm, timbre, loudness, duration, typical background.",
    "thermal": "You describe how things look in a THERMAL (infrared) camera. Each description is one short sentence "
               "about the heat signature: which parts are warm or cool, shape and contrast against the background.",
    "image": "You describe what things LOOK like. Each description is one short sentence about visible appearance: "
             "shape, colour, texture, parts.",
    "video": "You describe what an ACTION looks like on video. Each description is one short sentence about the "
             "motion: body parts involved, tempo, objects handled.",
    "depth": "You describe the 3-D SHAPE of things as seen by a depth camera. Each description is one short sentence "
             "about geometry: size, layout, distance, surfaces.",
}
SCENE = ("You describe the VISUAL SCENE in which something typically happens. Each description is one short "
         "sentence about the place, the surrounding objects and the people or animals present.")
COMMON = (" The sentences are text anchors for a CLIP-style text encoder, so: name the class in every sentence, "
          "use plain concrete words, at most 20 words each, no numbering, no two sentences alike.")

SCHEMA = {"type": "object", "properties": {"descriptions": {"type": "array", "items": {"type": "string"}}},
          "required": ["descriptions"], "additionalProperties": False}


def describe(client, system, classname, dataset, n, effort):
    import anthropic
    user = f"Dataset: {dataset}. Class: \"{classname.replace('_', ' ')}\". Write {n} different descriptions."
    for attempt in range(5):
        try:
            resp = client.beta.messages.create(
                model=MODEL, max_tokens=16000, system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # retry a declined request server-side
            )
        except anthropic.RateLimitError as e:
            wait = int(e.response.headers.get("retry-after", "30"))
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            print(f"  transient error ({type(e).__name__}), retrying")
            time.sleep(5 * (attempt + 1))
            continue
        if resp.stop_reason == "refusal":
            print(f"  [{classname}] declined ({getattr(resp.stop_details, 'category', None)}); left empty")
            return []
        if resp.stop_reason == "max_tokens":
            print(f"  [{classname}] hit max_tokens; retrying")
            continue
        text = next(b.text for b in resp.content if b.type == "text")
        return [s.strip() for s in json.loads(text)["descriptions"] if s.strip()][:n]
    raise RuntimeError(f"no descriptions for {classname} after 5 attempts")


def make_llm(a):
    import anthropic
    names = [l.strip() for l in open(a.classnames) if l.strip()]
    path = os.path.join(a.out_dir, f"{a.dataset}.json")
    doc = json.load(open(path)) if os.path.exists(path) else {"dataset": a.dataset, "modality": a.modality,
                                                            "model": MODEL, "agents": {}}
    agents = {f"{a.modality}_sensor": SENSOR[a.modality] + COMMON, "scene": SCENE + COMMON}
    client = anthropic.Anthropic(max_retries=3)
    for agent, system in agents.items():
        by_cls = doc["agents"].setdefault(agent, {})
        for i, n in enumerate(names):
            if n in by_cls:
                continue
            by_cls[n] = describe(client, system, n, a.dataset, a.n, a.effort)
            json.dump(doc, open(path, "w"), indent=1, ensure_ascii=False)
            print(f"{agent:14s} {i + 1:4d}/{len(names)} {n}: {len(by_cls[n])}")
    print(f"saved {path}")


def make_random(a):
    """Same number of sentences and words per sentence, words drawn from the dataset's own description
    vocabulary with class-name words removed: style and vocabulary are kept, the class semantics are not."""
    import random
    src = os.path.join(a.out_dir, f"{a.dataset}.json")
    doc = json.load(open(src))
    words = lambda s: re.findall(r"[a-zA-Z]+", s.lower())
    stems = {w[:4] for agent in doc["agents"].values() for n in agent for w in words(n.replace("_", " ")) if len(w) > 2}
    allw = {w for agent in doc["agents"].values() for v in agent.values() for s in v for w in words(s)}
    vocab = sorted(w for w in allw if w[:4] not in stems)       # drops class words and their inflections
    rng = random.Random(a.seed)
    out = {**doc, "control": "random-words", "seed": a.seed, "agents": {}}
    for agent, by_cls in doc["agents"].items():
        out["agents"][agent] = {n: [" ".join(rng.choice(vocab) for _ in range(max(3, len(words(s))))) for s in v]
                                for n, v in by_cls.items()}
    dst = os.path.join(a.out_dir, f"{a.dataset}_random.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"saved {dst} ({len(vocab)}-word vocabulary)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--modality", choices=sorted(SENSOR), default="audio")
    p.add_argument("--classnames", help="one class name per line")
    p.add_argument("--n", type=int, default=10, help="descriptions per class and agent")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--out-dir", default=os.path.join(ROOT, "descriptions"))
    p.add_argument("--random-control", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.random_control:
        make_random(a)
    elif not a.classnames:
        sys.exit("--classnames is required to generate descriptions")
    else:
        make_llm(a)


if __name__ == "__main__":
    main()
