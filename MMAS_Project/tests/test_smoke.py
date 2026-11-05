"""End-to-end smoke test on synthetic embeddings (CPU, ~1 min):  python -m pytest tests  or  python tests/test_smoke.py"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mmas import data as D                      # noqa: E402
from mmas.geometry import em_wop, unit, wop     # noqa: E402
from mmas.policy import predict                 # noqa: E402


def _run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def test_skew_idx_ratio():
    y = np.repeat(np.arange(10), 50)
    ix = D.skew_idx(y, 10, 0)
    n = np.bincount(y[ix], minlength=10)
    assert n.max() == 50 and n.min() == 5


def test_methods_label_free_shapes():
    rng = np.random.default_rng(0)
    Z, T = unit(rng.normal(size=(200, 16))), unit(rng.normal(size=(4, 16)))
    for f in (wop, em_wop):
        p = f(Z, Z @ T.T)
        assert p.shape == (200,) and p.max() < 4
    p, info = predict(Z, T, estimator="wo")
    assert info["method"] in ("WO+P", "EM-WO+P") and p.shape == (200,)


def test_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cache, res, pol = os.path.join(tmp, "cache"), os.path.join(tmp, "results"), os.path.join(tmp, "policy.json")
        _run("scripts/make_synthetic.py", "--out", cache)
        common = ["--runs", "configs/runs_synthetic.json", "--cache", cache, "--results", res, "--policy", pol]
        for cmd in ("diagnose", "evaluate", "failure", "calibrate", "report"):
            _run("scripts/run_experiments.py", cmd, *common)
        _run("verify.py", "--results", res, "--policy", pol)
        rep = json.load(open(os.path.join(res, "verify_report.json")))
        assert rep["status"]["C1"] and rep["status"]["C3"] and rep["status"]["F4"]


if __name__ == "__main__":
    test_skew_idx_ratio()
    test_methods_label_free_shapes()
    test_pipeline_end_to_end()
    print("all smoke tests passed")
