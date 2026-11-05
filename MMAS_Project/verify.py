#!/usr/bin/env python
"""Check the proposal's claims against the result files, and decide the hypothesis.

  C1  WO+P beats reproduced cosine on every balanced run.
  C2  WO+P falls below cosine under imbalance (50x).
  C3  EM-WO+P is never more than 0.5 points below cosine under imbalance (10x and 50x).
  C4  Neither is the label-free policy (balanced, 10x and 50x).
  R   Balanced cosine / WO+P reproduce Table 1 (configs/expected.json).
  F3  LLM gain is specific: shuffled and random-word descriptions gain less than the real ones.
  F4  Inductive split: fitting mu_c, Sigma on half A still beats cosine on half B.

Hypothesis: supported if C1 holds on the held-out runs and C3-C4 hold everywhere; contradicted if
the gain disappears under F4 or random-word anchors do as well as LLM anchors (F3).

Usage: python verify.py [--results results] [--strict]      (--strict: exit 1 if a claim fails)
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mmas import data as D        # noqa: E402
from mmas import results as R     # noqa: E402

MARGIN = 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--policy", default=os.path.join(ROOT, "configs", "policy.json"))
    p.add_argument("--expected", default=os.path.join(ROOT, "configs", "expected.json"))
    p.add_argument("--metric", default="top1", choices=["top1", "mrec"])
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    cfg = R.load_policy_config(a.policy)
    expected = json.load(open(a.expected)) if os.path.exists(a.expected) else {}
    checks = {k: [] for k in ("C1", "C2", "C3", "C4", "R", "F3", "F4")}

    def add(claim, run, ok, detail):
        checks[claim].append({"run": run, "ok": bool(ok), "detail": detail})

    for path in D.result_files(a.results, "eval"):
        res = R.load(path)
        run, split = res["run"], res.get("split")
        g = {k: R.aggregate(v, a.metric) for k, v in res["parts"].items()}
        g["policy | auto"] = R.aggregate(R.policy_parts(res, cfg), a.metric)
        cos = g.get("cosine | T")
        if not cos:
            continue
        w = g.get("WO+P | T")
        if w and w["balanced"] is not None:
            add("C1", run, w["balanced"] > cos["balanced"],
                f"[{split}] WO+P {w['balanced']:.2f} vs cosine {cos['balanced']:.2f}")
        if w and w["IR50"] is not None:
            add("C2", run, w["IR50"] < cos["IR50"], f"50x: WO+P {w['IR50']:.2f} vs cosine {cos['IR50']:.2f}")
        for key, claim, cs in (("EM-WO+P | T", "C3", ("IR10", "IR50")), ("policy | auto", "C4", R.CONDS)):
            m = g.get(key)
            if not m:
                continue
            for c in cs:
                if m[c] is not None and cos[c] is not None:
                    add(claim, run, m[c] >= cos[c] - MARGIN, f"{c}: {key.split(' |')[0]} {m[c]:.2f} vs cosine {cos[c]:.2f}")
        for key, val in expected.get(run, {}).items():
            here = g.get(key, {}).get("balanced")
            if here is not None:
                add("R", run, abs(here - val) <= 0.1, f"{key}: paper {val:.1f}, here {here:.2f}")

    for path in D.result_files(a.results, "failure"):
        res = R.load(path)
        g = {k: R.aggregate(v, a.metric) for k, v in res["parts"].items()}
        base = g.get("F3 | EM-WO+P | T")
        real = g.get("F3 | EM-WO+P | T+D")
        if base and real:
            for ctrl in ("T+D:shuffled", "T+D:random"):
                c = g.get(f"F3 | EM-WO+P | {ctrl}")
                if c:
                    for cond in R.CONDS:
                        gr, gc = real[cond] - base[cond], c[cond] - base[cond]
                        add("F3", res["run"], gc < gr, f"{cond}: LLM gain {gr:+.2f} vs {ctrl} gain {gc:+.2f}")
        cosB, ind = g.get("F4 | cosine | B"), g.get("F4 | WO inductive A->B")
        tr = g.get("F4 | WO transductive B")
        if cosB and ind:
            add("F4", res["run"], ind["balanced"] > cosB["balanced"],
                f"inductive WO {ind['balanced']:.2f} vs cosine {cosB['balanced']:.2f}"
                + (f" (transductive WO {tr['balanced']:.2f})" if tr else ""))

    print(f"metric: {a.metric}   policy: estimator={cfg['estimator']} ratio_thr={cfg['ratio_thr']} "
          f"agree_thr={cfg['agree_thr']}\n")
    status = {}
    for claim, items in checks.items():
        if not items:
            status[claim] = None
            print(f"{claim:3s}  no data")
            continue
        status[claim] = all(i["ok"] for i in items)
        print(f"{claim:3s}  {'PASS' if status[claim] else 'FAIL'}  ({sum(i['ok'] for i in items)}/{len(items)})")
        for i in items:
            if not i["ok"] or len(items) <= 30:
                print(f"       {'ok ' if i['ok'] else 'NO '} {i['run']:28s} {i['detail']}")

    heldout_c1 = [i for i in checks["C1"] if i["detail"].startswith("[heldout]")]
    if not heldout_c1 or status["C3"] is None or status["C4"] is None:
        verdict = "undecided (held-out C1, C3 or C4 not yet run)"
    elif status["F4"] is False or status["F3"] is False:
        verdict = "contradicted (gain vanishes under F4, or random/shuffled descriptions match LLM ones in F3)"
    elif all(i["ok"] for i in heldout_c1) and status["C3"] and status["C4"]:
        verdict = "supported" + ("" if status["F3"] and status["F4"] else " (F3/F4 not yet run)")
    else:
        verdict = "not supported (C1 on held-out, C3 or C4 fails)"
    print(f"\nHypothesis: {verdict}")
    os.makedirs(a.results, exist_ok=True)
    json.dump({"status": status, "verdict": verdict, "checks": checks, "policy": cfg},
              open(os.path.join(a.results, "verify_report.json"), "w"), indent=1)
    print(f"saved {a.results}/verify_report.json")
    if a.strict and any(v is False for k, v in status.items() if k != "C2"):
        sys.exit(1)


if __name__ == "__main__":
    main()
