# Learning in a shared multi-modal embedding space

**Omikumar Bhavinkumar Makadia** · ECE / WINLAB, Rutgers University · Graduate track

ImageBind and LanguageBind put audio, video, thermal and depth inputs into one embedding space with text. This lets you classify an infrared frame or a sound clip from class names alone, by cosine similarity to a text prototype. This repository replaces cosine with a **label-free, covariance-weighted score**.

The score is estimated only from the geometry of the unlabelled test data. It needs no training and keeps the frozen encoders. The repository also tests whether the gain survives when class sizes are **unknown and imbalanced**, using estimated class sizes and LLM-written class descriptions.

> **Research question.** Can a shared embedding space improve classification by relying solely on the geometry of its test data, and still achieve that gain when class sizes are unknown and imbalanced?

---

## Why cosine falls short

Write an input embedding as `z = μ_y + n`. We measured two weaknesses (`run_experiments.py diagnose`):

| | measured | consequence |
|---|---|---|
| **Anchor gap** | `cos(t_c, μ_c) ≈ 0.27–0.34` | The text prototype points only loosely at its class centre. |
| **Anisotropic noise** | `λ_max / λ_med ≈ 10³ – 1.9×10⁴` | A few nuisance directions (loudness, microphone, viewpoint) dominate an inner product that weights all directions equally. |

Array processing solves the same problem with a minimum-variance (covariance-weighted) filter. We apply that idea here.

## Method

The score is the Bayes rule for classes that share one covariance:

```
s_c(z) = μ_cᵀ Σ̂⁻¹ z − ½ μ_cᵀ Σ̂⁻¹ μ_c,        Σ̂ = (1−γ) Σ_W + γ · tr(Σ_W)/D · I
```

There are no labels, so `μ_c` and `Σ_W` are estimated from soft pseudo-labels `Q`, which come from the initial anchor scores `S0 = Z Aᵀ`. The variants differ in how they build `Q`:

| variant | class-size assumption | pseudo-labels |
|---|---|---|
| **WO+P** | equal sizes | Sinkhorn with equal marginals |
| **PA-WO+P** | sizes estimated once, then enforced | unbalanced OT → counts → Sinkhorn with those marginals |
| **EM-WO+P** | none | EM prior: `π_c ← 1/N Σ_i softmax_c(s_ic/τ + log π_c)` |

All variants then smooth the predictions over a k-NN graph built in the whitened space (the "+P").

Fixed settings, the same everywhere: `ε = 0.02`, `γ = 0.1`, `k = 10`, `α = 0.5`, `τ_decide = 0.1`, `τ_zs = 0.01` (`mmas/geometry.py`).

**Anchors.** `T` is the papers' prompt templates. `T+D` is `unit(T + mean of LLM description embeddings)`, which pools a *sensor* agent (how the class sounds, or its heat signature) and a *visual-scene* agent.

**Label-free policy** (`mmas/policy.py`):

1. Use WO+P if the estimated size ratio `π̂_max / π̂_min < 2`. Otherwise use EM-WO+P.
2. Keep the LLM anchors only if their predictions agree with the templates' on at least `agree_thr` of the inputs.
3. `agree_thr` and the size estimator are set on the **development sets only** (`calibrate`).

```
cached z, anchors  →  pseudo-labels Q, π̂  →  μ_c, Σ̂ → s_c, k-NN  →  policy (label-free)  →  verify.py (C1–C4)
```

Everything after embedding extraction runs on a CPU.

## Results so far (from the proposal)

**Table 1:** zero-shot top-1 (%), balanced test sets, template anchors. All 13 runs improve, by +4.8 on average.

| ImageBind | ESC-50 | ESC-10 | US8K | VGGSound | IN-1k | Places365 | K400† | LLVIP |
|---|---|---|---|---|---|---|---|---|
| Cosine, reproduced | 68.3 | 91.8 | 48.9 | 27.6 | 77.5 | 46.1 | 52.9 | 59.8 |
| **WO+P** | **79.1** | **96.5** | **53.7** | **32.2** | **80.4** | **47.9** | **60.2** | **69.0** |

| LanguageBind | US8K | IN-1k | Places365 | K400† | LLVIP |
|---|---|---|---|---|---|
| Cosine, reproduced | 80.2 | 78.7 | 44.8 | 63.7 | 84.4 |
| **WO+P** | **84.2** | **80.8** | **47.2** | **67.0** | **89.0** |

† Kinetics-400, 3,998-clip subset.

**Table 2:** ESC-50, ImageBind, under imbalance (class sizes decay geometrically to a 10× or 50× ratio). The LLM row is preliminary.

| | balanced | 10× | 50× |
|---|---|---|---|
| Cosine | 68.3 | 68.9 | 69.1 |
| WO+P (equal sizes) | **79.1** | 63.5 (−5.4) | 51.1 (−18.0) |
| EM-WO+P (no assumption) | 70.5 | 69.1 | 70.7 |
| EM-WO+P + LLM descriptions | 75.9 | **74.1** | **74.1** |

`configs/expected.json` holds Table 1. `verify.py` checks your re-run against it.

---

## Repository layout

```
mmas/
  geometry.py      covariance score, Sinkhorn / OT / EM priors, k-NN smoothing, WO+P, PA-WO+P, EM-WO+P
  anchors.py       T, T+D, per-agent anchors, F3 controls (shuffled / random-word), template agreement
  policy.py        label-free choice of variant and anchors
  data.py          bundle loading (.pt / .npz), runs config, imbalance subsampler (skew_idx)
  metrics.py       top-1, per-class mean recall, McNemar; anchor-gap and anisotropy diagnostics
  results.py       result-file format, aggregation over seeds, policy look-up
  encoders.py      ImageBind / LanguageBind wrappers (GPU; imported lazily)
scripts/
  extract_embeddings.py   GPU, once: inputs + templates -> cache/<dataset>_<encoder>.pt
  make_descriptions.py    LLM class descriptions (Claude API) -> descriptions/<dataset>.json; random-word control
  embed_descriptions.py   GPU, once: descriptions -> cache/<dataset>_<encoder>_desc.npz
  run_experiments.py      CPU: evaluate | diagnose | failure | calibrate | report
  make_synthetic.py       small synthetic bundle for testing the pipeline
verify.py                 checks claims C1-C4, reproduction of Table 1, F3, F4; states the hypothesis verdict
configs/
  runs.json               the 13 runs, with dev / heldout split
  expected.json           Table 1 numbers
  policy.json             policy thresholds (rewritten by `calibrate`)
  templates.json          prompt templates per modality
  runs_synthetic.json     smoke-test run
descriptions/             versioned LLM descriptions (so no API access is needed to reproduce)
tests/test_smoke.py       end-to-end test on synthetic data
```

## Installation

```bash
pip install -r requirements.txt
```

The experiments need only NumPy and SciPy. Install the following only for the steps that use them:

- `torch`: to read or write `.pt` bundles.
- `anthropic`: to generate descriptions.
- ImageBind or LanguageBind: to extract embeddings (see `requirements.txt`).

## Quick start: synthetic smoke test (about 1 minute, CPU)

```bash
python tests/test_smoke.py
```

To watch every step instead:

```bash
python scripts/make_synthetic.py --out cache
python scripts/run_experiments.py evaluate --runs configs/runs_synthetic.json
python scripts/run_experiments.py failure  --runs configs/runs_synthetic.json
python scripts/run_experiments.py calibrate --runs configs/runs_synthetic.json
python scripts/run_experiments.py report
python verify.py
```

The synthetic bundle builds in both weaknesses, a loose anchor and anisotropic noise. On it, WO+P scores ≈100% against ≈61% for cosine when classes are balanced, and falls below cosine at 50×. EM-WO+P stays well above cosine. This is a test of the code, not a result.

---

## Full pipeline

### 1. Embeddings (GPU, once)

Bundles from earlier notebooks (`cache/<dataset>_<encoder>.pt` with keys `emb`, `text_emb`, `labels`, `classnames`) **drop straight in**. Copy them to `cache/`. To build a new one from a CSV manifest (`path,label`):

```bash
python scripts/extract_embeddings.py --encoder imagebind --modality audio \
    --manifest manifests/esc50.csv --classnames classnames/esc50.txt --out cache/esc50_imagebind.pt
```

`configs/templates.json` holds the prompt templates. Replace them with the exact lists behind Table 1 if they differ. The encoder wrappers follow each project's public README and have not been run in this repository, so check one bundle against the cached version before rebuilding the rest.

### 2. LLM class descriptions (once; versioned)

```bash
export ANTHROPIC_API_KEY=...      # or: ant auth login
python scripts/make_descriptions.py --dataset esc50 --modality audio --classnames classnames/esc50.txt
python scripts/make_descriptions.py --dataset esc50 --random-control          # F3 control, no API calls
```

- **Two agents per class:** `<modality>_sensor` (acoustic character, or the thermal signature) and `scene`. The default is 10 sentences per agent.
- **API details:** it uses `claude-opus-5-5` with JSON-schema output. Declined requests are retried on a fallback model on the server side (`fallbacks: "default"`).
- **Resuming:** the file is rewritten after every class, so an interrupted run picks up where it stopped.
- **Random-word control:** it keeps the sentence count and lengths but draws words from the dataset's own description vocabulary, with class-name words removed.
- **Versioning:** commit `descriptions/` so that reproducing the results needs no API access.

### 3. Embed the descriptions (GPU, once)

```bash
python scripts/embed_descriptions.py --encoder imagebind --modality audio \
    --descriptions descriptions/esc50.json --out cache/esc50_imagebind_desc.npz
python scripts/embed_descriptions.py --encoder imagebind --modality audio \
    --descriptions descriptions/esc50_random.json --out cache/esc50_imagebind_desc_random.npz
```

Description files in the earlier v16 layout (`{agent: {classname: tensor}}`, `.pt`) also load.

### 4. Experiments (CPU)

```bash
python scripts/run_experiments.py diagnose                  # anchor gap, anisotropy -> results/diagnostics.json
python scripts/run_experiments.py evaluate --split dev      # ESC-50, ESC-10, US8K (both encoders)
python scripts/run_experiments.py failure  --split dev      # F3, F4
python scripts/run_experiments.py calibrate                 # sets the policy on dev only -> configs/policy.json
python scripts/run_experiments.py evaluate --split heldout  # VGGSound, IN-1k, Places365, K400, LLVIP: once, at the end
python scripts/run_experiments.py failure  --split heldout
python scripts/run_experiments.py report                    # results/summary.csv, results/tables.md
python verify.py                                            # C1-C4, Table 1 check, F3, F4, verdict
```

Useful flags:

- `--only esc50,us8k/languagebind`: run a subset.
- `--balanced-only`: Table 1 only.
- `--metric mrec`: calibrate and verify on per-class mean recall.

Every command **resumes**: results are saved atomically after each fit, so rerunning skips work that is already done. As a guide, ESC-50 takes about a minute and everything takes about 1–2 hours on 2 cores. ImageNet (50k × 1000) dominates.

`evaluate` runs:

- **Conditions:** balanced, plus 10× and 50× imbalance with seeds 0, 1, 2. The subsampler is `mmas.data.skew_idx`, the same as the paper's.
- **Methods:** cosine, WO+P, PA-WO+P and EM-WO+P, on anchors `T` and `T+D`. The per-agent anchors get cosine and EM-WO+P only.
- **Policy inputs:** the size ratio under each estimator and the template agreement, so that the policy (and `calibrate`) can be evaluated without refitting.

---

## Claims, failure tests and the verdict

`verify.py` reads `results/` and reports PASS/FAIL for each item. With `--strict` it exits non-zero if a claim fails.

| | claim / test | how it is checked |
|---|---|---|
| **C1** | WO+P beats reproduced cosine on every balanced run | balanced top-1, `WO+P \| T` vs `cosine \| T` |
| **C2** | WO+P falls below cosine under imbalance | 50× top-1 |
| **C3** | EM-WO+P is never more than 0.5 points below cosine under imbalance | 10× and 50× |
| **C4** | neither is the label-free policy | balanced, 10× and 50× |
| **R** | the cached bundles reproduce Table 1 | within 0.1 of `configs/expected.json` |
| **F1** | imbalance | Table 2 conditions (`evaluate`) |
| **F2** | two-class LLVIP under imbalance, where LLM anchors already hurt (−4.7 at 10×, LanguageBind) | `llvip` runs; the policy's agreement gate should switch the descriptions off |
| **F3** | LLM gain is specific | shuffled and random-word descriptions must gain less than the real ones |
| **F4** | inductive split | fit `μ_c, Σ̂` on half A of the unlabelled clips, score half B with no refit |

**The hypothesis** (the covariance score beats cosine on balanced sets, and with estimated class sizes stays within 0.5 of cosine under 10–50× imbalance):

- **Supported** if C1 holds on the held-out runs and C3–C4 hold everywhere.
- **Contradicted** if the gain vanishes under F4, or if random-word anchors do as well as LLM anchors (F3).

**Data partitions.** ESC-50, ESC-10 and US8K are development sets, used for every choice including `calibrate`. VGGSound, LLVIP, K400, ImageNet-1k and Places365 are held out: evaluate each of them **once**, after the choices are final.

### Note on the policy's size estimate

The proposal's rule estimates `π̂` with EM on the *template* scores. Because of the anchor gap, that estimate is skewed even when the classes are balanced. On the synthetic bundle, EM gives a ratio of 6.2 on balanced data, so the rule would never pick WO+P. `policy.py` therefore offers three estimators:

| estimator | how `π̂` is computed | ratio taken over |
|---|---|---|
| `em` | EM on the template scores (the proposal's rule) | max / min |
| `em_trim` | EM on the template scores | 95% / 5% quantiles |
| `wo` | counts of covariance-score predictions after an unbalanced-OT fit (the PA-WO+P estimate) | max / min |

`calibrate` picks the estimator and the LLM-agreement threshold on the development runs. The ratio threshold of 2 is unchanged. On the synthetic data, `wo` gives 1.7 on balanced data and 5–21 under imbalance.

---

## File formats

| file | contents |
|---|---|
| `cache/<ds>_<enc>.pt\|.npz` | `emb` N×D, `text_emb` C×D, `labels` N, `classnames` C |
| `cache/<ds>_<enc>_desc[_random].npz` | `"<agent>::<classname>"` → n_c×D (or `.pt` `{agent: {classname: tensor}}`) |
| `descriptions/<ds>[_random].json` | `{"model", "agents": {agent: {classname: [sentences]}}}` |
| `results/eval_<ds>_<enc>.json` | `parts["<method> \| <anchor>"]["<cond>\|<seed>"] = {top1, mrec}`, `policy_inputs`, `seconds` |
| `results/failure_<ds>_<enc>.json` | `F3 \| …` and `F4 \| …` entries in the same layout |
| `results/summary.csv` | one row per (dataset, encoder, method, anchor); balanced / IR10 / IR50 means and seed SDs, top-1 and mean recall |
| `results/tables.md` | Tables 1 and 2 regenerated from your results |
| `results/verify_report.json` | per-claim status, every check, the verdict, and the policy used |

Labels are read **only** to score predictions, and by `diagnose`, which reports the anchor gap and anisotropy. No method or policy decision sees them.

## Milestones

- **M1:** repository, configs, evidence bundle; reproduce Tables 1–2 (`evaluate`, `verify.py` R / C1–C2).
- **M2:** EM-WO+P with LLM anchors on the held-out sets; tests F3–F4 (`failure`).
- **M3:** the policy (`calibrate`, C4).
- **M4:** final report (`report`).

**Risks and fallbacks:**

- **LLM descriptions hurt** (as on LLVIP): the policy's agreement gate switches them off.
- **No API access:** the descriptions are versioned in `descriptions/`.

## Future directions (not implemented here)

**Does the shared space preserve spatial layout? (depth)** The plan is a controlled RGB-D set: 1,600 pairs (20 scenes × 10 object layouts × 8 camera views) and about 400 captions with flipped twins. It would test four things:

- **Q1:** layout-sensitive retrieval.
- **Q2:** viewpoint drift per degree, RGB vs. depth.
- **Q3:** depth vs. true or flipped captions, for pairs never seen together in training.
- **Q4:** the covariance score on layout-only negatives, with whitening by the viewpoint covariance.

**A reactive description agent.** When the classifier is uncertain about some classes, it would send feedback to the description agent, which then rewrites those descriptions and the initial mapping. This would be compared with the two agents working separately.

## Data and ethics

All benchmarks are public. Embeddings are released only where the dataset licences allow it, and LLVIP is used as released.

## References

- R. Girdhar et al. *ImageBind: One embedding space to bind them all.* CVPR 2023.
- B. Zhu et al. *LanguageBind: Extending video-language pretraining to N-modality by language-based semantic alignment.* ICLR 2024.
- Y. Lyu et al. *UniBind: LLM-augmented unified and balanced representation space to bind them all.* CVPR 2024.
- Z. Zhang et al. *OmniBind: Large-scale omni multimodal representation via binding spaces.* ICLR 2025.

Code: [github.com/OmiMaka/MMAS_Project](https://github.com/OmiMaka/MMAS_Project)
