#!/usr/bin/env python
"""Run the pipeline described by a YAML config (config/smoke.yaml or config/cpu.yaml).

  python scripts/run_config.py config/smoke.yaml
  python scripts/run_config.py config/cpu.yaml [--dry-run]

Each step calls the existing scripts (make_synthetic.py, run_experiments.py, verify.py) with the
paths from the config, then checks that every claim in `expect_pass` passed in verify_report.json.
Exits non-zero if a step fails or an expected claim does not pass.
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_CMDS = {"diagnose", "evaluate", "failure", "calibrate", "report"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    paths = {k: os.path.join(ROOT, v) for k, v in cfg["paths"].items()}
    os.makedirs(paths["results"], exist_ok=True)
    common = ["--runs", paths["runs"], "--cache", paths["cache"],
              "--results", paths["results"], "--policy", paths["policy"]]

    commands = []
    for step in cfg["steps"]:
        name, opts = (step, {}) if isinstance(step, str) else next(iter(step.items()))
        if name == "synthesize":
            d = cfg["data"]
            commands.append(["scripts/make_synthetic.py", "--out", paths["cache"], "--C", str(d["classes"]),
                             "--D", str(d["dim"]), "--n", str(d["per_class"]), "--seed", str(d["seed"])])
        elif name in EXPERIMENT_CMDS:
            extra = ["--split", opts["split"]] if "split" in opts else []
            if name == "calibrate" and "policy" in cfg:
                extra += ["--metric", cfg["policy"].get("calibrate_metric", "top1")]
            commands.append(["scripts/run_experiments.py", name, *common, *extra])
        elif name == "verify":
            v = ["verify.py", "--results", paths["results"], "--policy", paths["policy"]]
            if "expected" in paths:
                v += ["--expected", paths["expected"]]
            commands.append(v)
        else:
            sys.exit(f"unknown step: {name}")

    for c in commands:
        print("$ python " + " ".join(os.path.relpath(x, ROOT) if os.path.isabs(x) else x for x in c), flush=True)
        if not a.dry_run:
            subprocess.run([sys.executable, *c], cwd=ROOT, check=True)
    if a.dry_run:
        return

    rep = json.load(open(os.path.join(paths["results"], "verify_report.json")))
    failed = [c for c in cfg.get("expect_pass", []) if not rep["status"].get(c)]
    print(f"\n[{cfg['name']}] expected to pass: {', '.join(cfg.get('expect_pass', []))}")
    print(f"[{cfg['name']}] " + ("all expected checks passed" if not failed else f"FAILED: {', '.join(failed)}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
