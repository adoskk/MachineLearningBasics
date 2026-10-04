# CNN Window-Sequence Accuracy

Detector-free follow-up to `CNN_vs_3DRCNN_latency`. It asks whether axial
context improves binary classification over the current per-slice CNN.

> CADS part-551 provides organ pseudo-labels, not tumors. KiTS labels 2/3
> select kidneys and LiTS label 5 selects liver. “Lesion” below therefore
> means the kidney/liver target region (an organ-localization proxy).

## Fair comparison

1. Split volumes before creating slices or windows.
2. Train the existing single-channel ResNet-18 slice classifier once.
3. Compute the axial target extent of each connected component from **training
   masks only** and set `W = ceil(q95(extent))`.
4. Freeze the CNN encoder and cache one embedding per slice.
5. Classify identical stride-1 windows with:
   - `baseline_max`: maximum baseline slice probability in the window;
   - `window_linear`: flatten `W × F`, then one linear output;
   - uni-LSTM, 1 and 2 layers;
   - bi-LSTM, 1 and 2 layers.
6. A window is positive if **any** constituent slice contains target.
   Thresholds are selected on validation windows and applied once to test.

The frozen shared encoder ensures improvements come from temporal aggregation,
not different image features.

## Metrics

The primary report includes window accuracy, equal-weight macro volume
accuracy, balanced accuracy, precision, recall, F1, AUROC, AP, confusion
matrix, per-subset metrics, trainable head parameters, and head latency.
Confidence intervals resample complete volumes because overlapping windows
are not independent. The default 300-volume cohort is split 70/15/15 and
stratified by CADS subset, giving about 45 independent test volumes.

Raw accuracy can be misleading under class imbalance; inspect balanced
accuracy, F1, and AP alongside it.

## Colab workflow

Open `notebooks/colab_window_sequence.ipynb`, or run:

```bash
python scripts/download_subset.py --subsets 0003_kits21 0004_lits \
  --local-dir /content/data/cads --max-files 600 --parts 551
python scripts/train_baseline.py --data-root /content/data/cads
python scripts/prepare_windows.py --data-root /content/data/cads
python scripts/train_temporal.py
python scripts/evaluate_models.py
```

Bulk NIfTI data should remain on `/content` local SSD, not Google Drive FUSE.
Code, checkpoints, caches, and results may stay on Drive.

## Outputs

- `results/cohort_split.json`: immutable volume split manifest
- `results/window_q95.json`: train-only extent distribution and W
- `results/window_comparison.{json,csv}`
- `results/window_predictions.json`
- `results/accuracy_comparison.png`
- `results/roc_pr_curves.png`
- `results/accuracy_vs_latency.png`

## Structure

```text
src/data/cads.py              CADS loader and organ-label filtering
src/data/window_utils.py      q95 extent and OR-labeled windows
src/data/embedding_cache.py   frozen CNN embeddings and window dataset
src/models/slice_encoder.py   current slice CNN baseline
src/models/temporal_heads.py  linear and uni/bi LSTM heads
src/training/                 baseline/head training and held-out comparison
scripts/                      reproducible CLI stages
notebooks/                    Colab end-to-end experiment
tests/test_smoke.py           synthetic data and all model variants
```
