"""Result files and aggregation shared by the experiment runner, the report and verify.py.

results/eval_<dataset>_<encoder>.json
    run, split, N, C
    parts:          {"<method> | <anchor>": {"<cond>|<seed>": {"top1": .., "mrec": ..}}}
    policy_inputs:  {"<cond>|<seed>": {"ratio_em": .., "ratio_em_trim": .., "ratio_wo": .., "agreement": ..}}
    seconds:        {"<method> | <anchor>": total seconds}
"""
import json
import os

import numpy as np

CONDS = ("balanced", "IR10", "IR50")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def save(res, path):
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, path)            # atomic: an interrupted run never leaves a half-written file


def tag(cond, seed):
    return f"{cond}|{seed}"


def aggregate(parts_one, metric="top1"):
    """{"cond|seed": {...}} -> {cond: mean, cond_sd: std} over seeds."""
    out = {}
    for c in CONDS:
        v = [r[metric] for t, r in parts_one.items() if t.split("|")[0] == c]
        out[c] = float(np.mean(v)) if v else None
        if c != "balanced":
            out[c + "_sd"] = float(np.std(v)) if v else None
    return out


def policy_choice(inp, cfg, has_td=True):
    method = "WO+P" if inp[f"ratio_{cfg['estimator']}"] < cfg["ratio_thr"] else "EM-WO+P"
    anchor = "T+D" if has_td and inp.get("agreement") is not None and inp["agreement"] >= cfg["agree_thr"] else "T"
    return method, anchor


def policy_parts(res, cfg):
    """Policy results looked up from the already-computed variants (no refitting)."""
    out = {}
    has_td = any(k.endswith("| T+D") for k in res["parts"])
    for t, inp in res.get("policy_inputs", {}).items():
        m, a = policy_choice(inp, cfg, has_td)
        r = res["parts"].get(f"{m} | {a}", {}).get(t)
        if r is not None:
            out[t] = dict(r, choice=f"{m} | {a}")
    return out


def load_policy_config(path):
    cfg = {"ratio_thr": 2.0, "agree_thr": 0.5, "estimator": "em"}
    if os.path.exists(path):
        cfg.update(json.load(open(path)))
    return cfg
