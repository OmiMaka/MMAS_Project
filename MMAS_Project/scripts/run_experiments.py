#!/usr/bin/env python
"""All CPU experiments on cached embeddings. Every subcommand is resumable (results are saved after each fit).

  evaluate   Table 1 (balanced) and Table 2 (10x / 50x imbalance, seeds 0-2): cosine, WO+P, PA-WO+P, EM-WO+P
             with template (T) and LLM (T+D) anchors, plus the inputs the label-free policy needs
  diagnose   anchor gap cos(t_c, mu_c) and anisotropy lambda_max / lambda_med per run
  failure    F3 (shuffled / random-word descriptions) and F4 (inductive half split)
  calibrate  set the policy's LLM-agreement threshold on the development runs only
  report     results/summary.csv and results/tables.md

Examples
  python scripts/run_experiments.py evaluate --split dev
  python scripts/run_experiments.py evaluate --balanced-only            # Table 1 only (fast)
  python scripts/run_experiments.py calibrate
  python scripts/run_experiments.py evaluate --split heldout            # once, after the choices are final
  python scripts/run_experiments.py report && python verify.py
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mmas import data as D                     # noqa: E402
from mmas import results as R                  # noqa: E402
from mmas.anchors import agreement, build_anchors   # noqa: E402
from mmas.geometry import METHODS, fit, propagate, softmax, sinkhorn, TAU_DECIDE, wo_scores  # noqa: E402
from mmas.metrics import anchor_gap, anisotropy, mean_recall, top1  # noqa: E402
from mmas.policy import ESTIMATORS, size_ratio  # noqa: E402

MAIN_ANCHORS = ("T", "T+D")                    # all four methods; other anchors get cosine + EM-WO+P only


def _runs(args):
    runs = D.load_runs_config(args.runs, args.split)
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r["dataset"] in keep or f"{r['dataset']}/{r['encoder']}" in keep]
    for r in runs:
        b = D.load_run(r, args.cache)
        if b is None:
            print(f"[skip] {r['dataset']}/{r['encoder']}: no bundle in {args.cache}")
            continue
        yield r, b


def _out(args, prefix, r):
    os.makedirs(args.results, exist_ok=True)
    return os.path.join(args.results, f"{prefix}_{r['dataset']}_{r['encoder']}.json")


def _score(pred, y):
    return {"top1": top1(pred, y), "mrec": mean_recall(pred, y)}


# ------------------------------------------------------------------ evaluate
def cmd_evaluate(args):
    for r, b in _runs(args):
        path = _out(args, "eval", r)
        res = R.load(path) or {"run": b.name, "parts": {}, "policy_inputs": {}, "seconds": {}}
        res.update(split=r.get("split"), N=int(len(b.y)), C=b.C, modality=r.get("modality"))
        anchors = build_anchors(b)
        conds = D.conditions(balanced_only=args.balanced_only)
        n_small = max(1, int(round(np.bincount(b.y).min() / 50)))
        print(f"== {b.name}  N={len(b.y):,} C={b.C} anchors={list(anchors)}  (smallest class at 50x: {n_small})")
        for cond, seed in conds:
            t = R.tag(cond, seed)
            ix = D.subset(b.y, cond, seed)
            Z, y = b.Z[ix], b.y[ix]
            if t not in res["policy_inputs"]:
                S0 = Z @ b.T.T
                inp = {f"ratio_{e}": size_ratio(Z, S0, e) for e in ESTIMATORS}
                inp["agreement"] = agreement(Z, b.T, anchors["T+D"]) if "T+D" in anchors else None
                res["policy_inputs"][t] = inp
            for a, A in anchors.items():
                for m, f in METHODS.items():
                    if a not in MAIN_ANCHORS and m not in ("cosine", "EM-WO+P"):
                        continue
                    key = f"{m} | {a}"
                    P = res["parts"].setdefault(key, {})
                    if t in P:
                        continue
                    t0 = time.time()
                    P[t] = _score(f(Z, Z @ A.T), y)
                    res["seconds"][key] = res["seconds"].get(key, 0.0) + time.time() - t0
                    R.save(res, path)
            R.save(res, path)
        for key in sorted(res["parts"]):
            g = R.aggregate(res["parts"][key])
            print(f"   {key:22s} " + "  ".join(f"{c} {g[c]:6.2f}" for c in R.CONDS if g[c] is not None))


# ------------------------------------------------------------------ diagnose
def cmd_diagnose(args):
    out = {}
    for r, b in _runs(args):
        gap = anchor_gap(b.Z, b.y, b.T)
        out[b.name] = {"anchor_gap_mean": float(gap.mean()), "anchor_gap_min": float(gap.min()),
                       "anchor_gap_max": float(gap.max()), "anisotropy": anisotropy(b.Z, b.y)}
        d = out[b.name]
        print(f"{b.name:28s} cos(t_c, mu_c) mean {d['anchor_gap_mean']:.3f} [{d['anchor_gap_min']:.3f}, "
              f"{d['anchor_gap_max']:.3f}]   lambda_max/lambda_med {d['anisotropy']:.3g}")
    os.makedirs(args.results, exist_ok=True)
    json.dump(out, open(os.path.join(args.results, "diagnostics.json"), "w"), indent=1)


# ------------------------------------------------------------------ failure tests F3, F4
def _inductive(Za, Sa, Zb, Sb, prior):
    """Fit mu_c, Sigma on half A (pseudo-labels only), score half B without refitting."""
    M, Wi = fit(Za, Sa, prior)
    return wo_scores(Zb, M, Wi).argmax(1)


def _transductive_wo(Z, S0, prior, smooth):
    M, Wi = fit(Z, S0, prior)
    L = wo_scores(Z, M, Wi)
    if not smooth:
        return L.argmax(1)
    Q0 = sinkhorn(L / L.std(), TAU_DECIDE) if prior == "balanced" else softmax(L / L.std() / TAU_DECIDE)
    return propagate(Z, Q0, Wi).argmax(1)


def cmd_failure(args):
    for r, b in _runs(args):
        path = _out(args, "failure", r)
        res = R.load(path) or {"run": b.name, "parts": {}}
        anchors = build_anchors(b, controls=True)
        print(f"== {b.name}  F3 anchors: {[a for a in anchors if a.startswith('T+D')] or 'none (no descriptions)'}")
        # F3: are LLM gains specific to the right descriptions?
        for cond, seed in D.conditions():
            t = R.tag(cond, seed)
            ix = D.subset(b.y, cond, seed)
            Z, y = b.Z[ix], b.y[ix]
            for a in ("T", "T+D", "T+D:shuffled", "T+D:random"):
                if a not in anchors:
                    continue
                for m in ("cosine", "EM-WO+P"):
                    P = res["parts"].setdefault(f"F3 | {m} | {a}", {})
                    if t not in P:
                        P[t] = _score(METHODS[m](Z, Z @ anchors[a].T), y)
                        R.save(res, path)
        # F4: inductive split - does the gain survive when mu_c, Sigma come from other clips?
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(b.y))
            A_ix, B_ix = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
            Za, Zb, yb = b.Z[A_ix], b.Z[B_ix], b.y[B_ix]
            Sa, Sb = Za @ b.T.T, Zb @ b.T.T
            t = R.tag("balanced", seed)
            rows = {"F4 | cosine | B": lambda: Sb.argmax(1)}
            for prior, m in (("balanced", "WO"), ("em", "EM-WO")):
                rows[f"F4 | {m} inductive A->B"] = lambda p=prior: _inductive(Za, Sa, Zb, Sb, p)
                rows[f"F4 | {m} transductive B"] = lambda p=prior: _transductive_wo(Zb, Sb, p, False)
                rows[f"F4 | {m}+P transductive B"] = lambda p=prior: _transductive_wo(Zb, Sb, p, True)
            for key, f in rows.items():
                P = res["parts"].setdefault(key, {})
                if t not in P:
                    P[t] = _score(f(), yb)
                    R.save(res, path)
        for key in sorted(res["parts"]):
            g = R.aggregate(res["parts"][key])
            print(f"   {key:34s} " + "  ".join(f"{c} {g[c]:6.2f}" for c in R.CONDS if g[c] is not None))


# ------------------------------------------------------------------ calibrate (development runs only)
def cmd_calibrate(args):
    cfg = R.load_policy_config(args.policy)
    dev = [R.load(_out(args, "eval", r)) for r in D.load_runs_config(args.runs, "dev")]
    dev = [d for d in dev if d and d.get("policy_inputs")]
    if not dev:
        sys.exit("calibrate: no development results found - run `evaluate --split dev` first")
    # ratio_thr stays at the proposal's 2; the size estimator and the LLM-agreement threshold are chosen here
    best = None
    for est in ESTIMATORS:
        for thr in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            trial = dict(cfg, estimator=est, agree_thr=float(thr))
            scores = [v[args.metric] for d in dev for v in R.policy_parts(d, trial).values()]
            s = float(np.mean(scores))
            if thr in (0.0, 0.5, 1.0):
                print(f"estimator {est:8s} agree_thr {thr:4.2f}  dev mean {args.metric} {s:6.2f}")
            if best is None or s > best[1] + 1e-9:
                best = (trial, s)
    cfg = dict(best[0], calibrated_on=[d["run"] for d in dev], calibration_metric=args.metric)
    json.dump(cfg, open(args.policy, "w"), indent=1)
    print(f"-> estimator = {cfg['estimator']}, agree_thr = {cfg['agree_thr']} "
          f"(dev mean {args.metric} {best[1]:.2f}); saved {args.policy}")


# ------------------------------------------------------------------ report
def cmd_report(args):
    cfg = R.load_policy_config(args.policy)
    rows = []
    for path in D.result_files(args.results, "eval"):
        res = R.load(path)
        ds, enc = res["run"].split("/")
        parts = dict(res["parts"])
        parts["policy | auto"] = R.policy_parts(res, cfg)
        for key, P in parts.items():
            m, a = key.split(" | ")
            row = {"dataset": ds, "encoder": enc, "split": res.get("split"), "method": m, "anchor": a}
            for metric in ("top1", "mrec"):
                for c, v in R.aggregate(P, metric).items():
                    row[f"{c}_{metric}"] = None if v is None else round(v, 2)
            rows.append(row)
    if not rows:
        sys.exit("report: no results/eval_*.json yet")
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def get(ds, enc, m, a, c="balanced", metric="top1"):
        for r in rows:
            if (r["dataset"], r["encoder"], r["method"], r["anchor"]) == (ds, enc, m, a):
                return r[f"{c}_{metric}"]
        return None

    runs = sorted({(r["encoder"], r["dataset"]) for r in rows})
    fmt = lambda v: "-" if v is None else f"{v:.1f}"
    md = ["## Table 1 - balanced top-1 (%), template anchors", "",
          "| encoder | dataset | cosine | WO+P | gain |", "|---|---|---|---|---|"]
    for enc, ds in runs:
        c, w = get(ds, enc, "cosine", "T"), get(ds, enc, "WO+P", "T")
        md.append(f"| {enc} | {ds} | {fmt(c)} | {fmt(w)} | {'-' if None in (c, w) else f'{w - c:+.1f}'} |")
    md += ["", "## Table 2 - imbalance, top-1 (%) [mean recall]", ""]
    variants = [("cosine", "T"), ("WO+P", "T"), ("PA-WO+P", "T"), ("EM-WO+P", "T"), ("EM-WO+P", "T+D"), ("policy", "auto")]
    for enc, ds in runs:
        if get(ds, enc, "cosine", "T", "IR10") is None:
            continue
        md += [f"**{ds} / {enc}**", "", "| method | balanced | 10x | 50x |", "|---|---|---|---|"]
        for m, a in variants:
            cells = [f"{fmt(get(ds, enc, m, a, c))} [{fmt(get(ds, enc, m, a, c, 'mrec'))}]" for c in R.CONDS]
            if any(get(ds, enc, m, a, c) is not None for c in R.CONDS):
                md.append(f"| {m} ({a}) | " + " | ".join(cells) + " |")
        md.append("")
    open(os.path.join(args.results, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"saved {args.results}/summary.csv and {args.results}/tables.md")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["evaluate", "diagnose", "failure", "calibrate", "report"])
    p.add_argument("--runs", default=os.path.join(ROOT, "configs", "runs.json"))
    p.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--policy", default=os.path.join(ROOT, "configs", "policy.json"))
    p.add_argument("--split", default="all", choices=["all", "dev", "heldout"])
    p.add_argument("--only", help="comma-separated datasets or dataset/encoder pairs")
    p.add_argument("--balanced-only", action="store_true", help="evaluate: Table 1 condition only")
    p.add_argument("--metric", default="top1", choices=["top1", "mrec"], help="calibrate: objective")
    args = p.parse_args()
    {"evaluate": cmd_evaluate, "diagnose": cmd_diagnose, "failure": cmd_failure,
     "calibrate": cmd_calibrate, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
