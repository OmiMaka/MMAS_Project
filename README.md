# v17 — Class imbalance on image and video, with TransCLIP and ZLaP

`Colab_v17_imbalance_image_video.ipynb`

This notebook runs the **Table 2 class-imbalance protocol** on the image and video embedding bundles built in v8/v9. It compares our label-free zero-shot methods (cosine, WO+P, EM-WO+P) with two published transductive baselines, **TransCLIP** (NeurIPS'24) and **ZLaP** (CVPR'24). Both baselines run from their authors' code.

| | |
|---|---|
| **Datasets** | ImageNet-1k, Places365, Kinetics-400 (3,998-clip subset) |
| **Encoders** | ImageBind, LanguageBind |
| **Methods** | cosine, WO+P, EM-WO+P (ours), TransCLIP, ZLaP |
| **Conditions** | balanced; 10× and 50× imbalance (IR10, IR50), seeds 0, 1, 2 |
| **Anchors** | the papers' templates only (`T`); these class sets have no LLM descriptions |
| **Metric** | top-1 accuracy (%), averaged over seeds |
| **Runtime** | Colab T4 GPU, about 2–3 h (mostly ImageNet). An A100 is faster but not needed. |

---

## Quick start

1. Open the notebook in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**, and allow Google Drive access when asked.
4. When it finishes, send back `MyDrive/ortho-v7/results/v17_imbalance_vision.csv`. The `v17_*.json` files are optional.

**If Colab disconnects:** click Run all again. Results are saved after every subsample, so the run picks up where it stopped. At most one fit is lost.

Kinetics-400 runs first and takes about 10 minutes. Its numbers are useful even if the rest is still running.

---

## Prerequisites

### Google Drive layout

The notebook works in `/content/drive/MyDrive/ortho-v7/` and expects the bundles from v8/v9 in `cache/`:

```
MyDrive/ortho-v7/
├── cache/
│   ├── imagenet_imagebind.pt
│   ├── imagenet_languagebind.pt
│   ├── places365_imagebind.pt
│   ├── places365_languagebind.pt
│   ├── kinetics400_imagebind.pt      # or k400_imagebind.pt
│   └── kinetics400_languagebind.pt   # or k400_languagebind.pt
└── results/                          # created if missing
```

If a bundle is not in `cache/`, the notebook searches all of `MyDrive/**` for `<name>_<encoder>.pt`. A dataset/encoder pair that cannot be found is **skipped** and reported as `NOT FOUND (skipped)`, and the run carries on.

### Bundle format

Each `.pt` file is a dict loaded with `torch.load`:

| key | shape / type | meaning |
|---|---|---|
| `emb` | `N × D` tensor | sample (image/video) embeddings |
| `text_emb` | `C × D` tensor | class template prototypes |
| `labels` | `N` int tensor | ground-truth class indices (used **only for scoring**) |
| `classnames` | list of `C` str | class names |

The notebook L2-normalises the embeddings and converts them to float64.

### Dependencies

Colab already has these: `torch`, `numpy`, `scipy`, `pandas`, `psutil`. The notebook installs `cupy-cuda12x` if it is missing, because ZLaP's official code needs it.

---

## Notebook structure

| Cell | What it does |
|---|---|
| 0 | Overview and run instructions (markdown). |
| 1 | Mounts Drive, `cd`s to `ortho-v7`, finds the bundles, and builds the `RUNS` list. The order is Kinetics → Places365 → ImageNet, smallest first. |
| 2 | Clones the baselines at pinned commits, applies the TransCLIP chunk-size patch, installs CuPy, and prints the GPU and RAM. |
| 3 | Writes `llm_lib.py`: our methods (WO+P, PA-WO+P, EM-WO+P), Sinkhorn/OT/EM priors, k-NN propagation and the imbalance subsampler. |
| 4 | Writes `v16_baselines.py`: TransCLIP and ZLaP wrappers, bundle loader, and v16's evaluation loop. |
| 5 | Writes `v17_vision.py`: memory-safe TransCLIP for large N, a resumable `run_one`, `summarize`, and the Table 1 reference values. |
| 6 | **Main run.** Loops over `RUNS` × methods × subsets and writes `results/v17_<ds>_<enc>.json`. |
| 7 | **Results.** Builds the CSV, prints pivot tables, compares with Table 1, and, if v16 results exist, prints audio + vision together. |
| 8 | What to send back (markdown). |

---

## Methods

### Ours (from `llm_lib.py`)

Initial scores are `S0 = Z @ T.ᵀ` (cosine against the template prototypes).

* **cosine**: `argmax S0`.
* **WO+P**: Sinkhorn with equal class shares → soft class means and shrunk within-class covariance → whitened (LDA-style) scores → k-NN label propagation in the whitened space.
* **EM-WO+P**: the same pipeline, except the initial soft assignment uses an **EM-estimated class prior** in place of the balanced Sinkhorn, so it never assumes balanced classes.

The hyper-parameters are frozen and match the main study (`geo_final.py` / `ortho_lib.py`):

| `EPS` | `GAMMA` (shrinkage) | `K_NN` | `ALPHA` (propagation) | `TAU_DECIDE` | `TAU_ZS` |
|---|---|---|---|---|---|
| 0.02 | 0.1 | 10 | 0.5 | 0.1 | 0.01 |

`llm_lib.py` also defines `PA-WO+P` and the LLM-anchor builder (`build_anchor_scores`). Both are shared with other notebooks, but **v17 does not run them**.

### Baselines (authors' code, pinned)

| Method | Repository | Commit | Settings |
|---|---|---|---|
| TransCLIP | [MaxZanella/transduction-for-vlms](https://github.com/MaxZanella/transduction-for-vlms) | `6a9e8f3ba57a11d1c5affe248a82df56dbc7f1b8` | zero-shot mode, authors' defaults |
| ZLaP | [vladan-stojnic/ZLaP](https://github.com/vladan-stojnic/ZLaP) | `89ab14bef8904a9397f8b6a7e5f3e6af2c288b93` | transductive; `k=5`, `gamma=5`, `alpha=0.3` (README setting) |

Changes to the authors' code. None of them change the predictions.

* **TransCLIP**
  * The fixed `chunk_size = 2500` becomes a module-level `CHUNK`, set per dataset to `max(32, min(2500, 1e9 / (K·d·4)))`, which is about 1 GB per chunk. It stays 2500 wherever that fits, which includes every audio set.
  * For `N > 20,000`, the 3-NN affinity is computed in 4,096-row blocks instead of as one dense `N × N` matrix. The neighbours and weights are the same.
  * The `cls_acc` print helper is patched to use `.item()` for recent NumPy.
  * The memory changes were checked to give identical predictions on ESC-50 and US8K.
* **ZLaP**
  * FAISS k-NN search is replaced by an exact `torch` top-k with the same inner-product neighbours, because `faiss-gpu` does not pip-install on Colab.
  * A shim covers CuPy's `tol → rtol` rename in `cg`.
  * If the official CuPy path fails a smoke test, the notebook falls back to a line-by-line SciPy port (`zlap_port`). Which one was used is recorded in the `zlap_impl` column.

---

## Imbalance protocol

`llm_lib.skew_idx(y, ir, seed)` is the same subsampler as in the paper:

1. `n0` is the size of the smallest class.
2. The classes are put in a random order, using `seed`.
3. The class at rank `r` keeps `round(n0 · ir^(−r/(C−1)))` samples, with a minimum of 1. The largest class keeps `n0` and the smallest keeps `n0/ir`.

`balanced` is the full bundle as it is. `IR10` and `IR50` are each averaged over seeds 0, 1 and 2, and their standard deviation is reported too. Before each run, the notebook prints how many samples the smallest class keeps at 50×.

---

## Outputs

All outputs are written to `MyDrive/ortho-v7/results/`:

| File | Contents |
|---|---|
| `v17_imbalance_vision.csv` | **Main deliverable.** One row per (dataset, encoder, method). |
| `v17_<ds>_<enc>.json` | Every subsample's accuracy (`parts["<method> \| T"]["<cond>\|<seed>"]`), timings, `N`, `C` and `zlap_impl`. This is also the resume checkpoint. |
| `v17_<ds>_<enc>_<method>_balanced_correct.npy` | A per-sample boolean "correct" vector on the balanced set, for McNemar tests (`llm_lib.mcnemar`). |

The notebook also writes `llm_lib.py`, `v16_baselines.py` and `v17_vision.py` into `MyDrive/ortho-v7/`.

### CSV columns

| column | meaning |
|---|---|
| `dataset`, `encoder`, `method` | run identifiers |
| `anchors` | always `T` (templates) |
| `balanced` | top-1 % on the full set |
| `IR10`, `IR10_sd` | mean / std top-1 % over 3 seeds at 10× imbalance |
| `IR50`, `IR50_sd` | mean / std top-1 % over 3 seeds at 50× imbalance |
| `seconds` | total wall time for that method across all 7 subsets |
| `zlap_impl` | `official` (authors' CuPy code) or `port` (SciPy fallback) |

### Printed summaries (cell 7)

1. Per-run pivot: method × (dataset, encoder) for balanced / IR10 / IR50.
2. Mean over the image/video runs for each method.
3. **Sanity check against Table 1.** For each run, the balanced `cosine | T` and `WO+P | T` are marked `OK` if they are within 0.06 of the paper:

   | dataset | encoder | cosine | WO+P |
   |---|---|---|---|
   | ImageNet | ImageBind | 77.53 | 80.41 |
   | Places365 | ImageBind | 46.08 | 47.91 |
   | Kinetics-400 | ImageBind | 52.88 | 60.23 |
   | ImageNet | LanguageBind | 78.73 | 80.79 |
   | Places365 | LanguageBind | 44.75 | 47.15 |
   | Kinetics-400 | LanguageBind | 63.68 | 66.98 |

   `DIFFERENT` means the bundle is not the one used for Table 1.
4. If `results/v16_baselines.csv` exists, the audio runs from v16 (ESC-50, US8K, VGGSound; templates only) and the image/video runs from v17 are averaged together.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GPU: NONE - switch the runtime to T4 GPU` | Change the runtime type to T4 GPU and run all again. |
| `NOT FOUND (skipped)` for a dataset | Put the bundle at `ortho-v7/cache/<name>_<encoder>.pt` (Kinetics also accepts `k400_…`). |
| `AssertionError: TransCLIP code changed` | The clone is not at the pinned commit. Delete `/content/_TransCLIP` and rerun cell 2. |
| `run the clone cell first …` | Cell 2 did not run, or the patch did not apply. Run cell 2, then cell 6. |
| `official ZLaP could not run here … using the scipy port` | CuPy failed. The results are still valid, and the CSV column `zlap_impl = port` records it. |
| CUDA OOM on ImageNet | Restart the runtime and run all; it resumes from the JSON. An A100 avoids this. |
| You want to rerun a method from scratch | Delete its key from `parts` in the matching `v17_<ds>_<enc>.json`, or delete the file to redo the whole run. |

---

## Label use

No method reads labels when it makes predictions. Labels are used only to compute accuracy. TransCLIP's solver also takes them as input, but only to count the classes and to print its own accuracy.
