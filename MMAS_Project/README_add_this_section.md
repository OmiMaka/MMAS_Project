## Configurations (submission)

| config | what it runs | needs | runtime |
|---|---|---|---|
| `config/smoke.yaml` | synthetic end-to-end: diagnose → evaluate → failure → calibrate → report → verify | nothing (CPU, no downloads, no API) | ~1 min |
| `config/cpu.yaml` | the 13 Table 1 runs: dev → calibrate (policy frozen) → held-out once → report → verify | cached embeddings in `cache/`, `descriptions/` | ~1–2 h on 2 cores |

```bash
pip install -r requirements.txt
python scripts/run_config.py config/smoke.yaml          # expects C1–C4, F3, F4 to PASS
python scripts/run_config.py config/cpu.yaml --dry-run  # prints the full pipeline's commands
python scripts/run_config.py config/cpu.yaml            # full run; expects R, C1–C4, F3, F4
```

`run_config.py` exits non-zero if any check listed under `expect_pass` in the config fails.
The evidence bundle (result files and `verify_report.json` behind Tables 1–2) is submitted separately.
