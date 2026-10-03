# CADS lesion benchmark: 3D Faster R-CNN vs batched per-slice classifier

Compares **memory footprint + compute time** of two ways to answer the same
clinical question — *"where is the lesion?"* — on CT volumes from
[`mrmrx/CADS-dataset`](https://huggingface.co/datasets/mrmrx/CADS-dataset) (gated, 22k vols / 167 structures, NIfTI).

- **Model A — 3D Faster R-CNN** (`src/models/faster_rcnn_3d.py`): one heavy 3D
  pass per volume, predicts 3D boxes `(x1,y1,z1,x2,y2,z2)` + scores.
- **Model B — per-slice classifier** (`src/models/slice_classifier.py`):
  a 2D ResNet-18 scores each axial slice; a volume is processed as a **batch**
  of slices (`batch_size` = 1/8/32 sweep). No 3D context, but streamable.

Both labels are derived from the **same** organ mask so the comparison is
fair — specifically CADS **part_551 (abdominal organs)**: KiTS21 keeps
labels `[2,3]` (Kidney R/L, kidney-tumor proxy), LiTS keeps `[5]` (Liver,
liver-tumor proxy). See `configs/config.yaml` (`seg_parts`, `subset_labels`).

> Honesty note: CADS masks are *organ* pseudo-labels, not tumor boxes, and
> challenge test splits have no public tumor masks at all — so "lesion"
> here means the organ region (detect the kidney/liver, flag slices that
> contain it). Same question for both models, fairly comparable.

> Scope honesty: the 3D detector is a compact teaching implementation
> (single-level cubic anchors, crop-pool RoI) — representative of the RPN
> family's memory/latency profile, not a SOTA clinical detector.

## Quickstart (Windows CPU, no HF access)

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
pip install -r requirements.txt
python tests/test_smoke.py
python scripts/run_benchmark.py --synthetic --no-train --repeats 4
# -> results/benchmark.json
```

## Real data (Colab GPU, minimal training)

1. `hf auth login`, accept the gated terms on the dataset page.
2. Open `notebooks/colab_benchmark.ipynb` (train → test → benchmark) **or**:
   Either `git clone` the project in Colab, or upload the `modiface`
   folder to Google Drive once (`MyDrive/modiface`, code only — no `data/`)
   and mount it — the notebook has a cell for each, plus a path check.
   Bulk NIfTI data goes to local Colab SSD (`/content/data/cads`):
   Drive's FUSE mount drops under heavy I/O, and this also keeps Drive
   usage to code + checkpoints.
```bash
pip install -r requirements.txt
python scripts/download_subset.py --subsets 0003_kits21 0004_lits
# low on disk? cap it: --max-files 120  (~60 vols, pairs kept intact — plenty for n_volumes: 30)
python scripts/train_detector.py
python scripts/train_classifier.py
python scripts/test_detector.py
python scripts/test_classifier.py
python scripts/run_benchmark.py --no-train --repeats 20
```

## What the benchmark reports (`results/benchmark.json`)

| axis | detector | classifier |
|---|---|---|
| params / model MB | ✓ | ✓ |
| mean ± std ms/volume, vols/sec | ✓ | ✓ per batch size |
| peak CUDA MB + CPU RSS delta | ✓ | ✓ per batch size |
| quality context | AP@IoU3D 0.3 | AUROC / AP + slices/sec |

Expect: detector slower + hungrier per volume; classifier throughput grows
with batch size until GPU RAM caps it — that crossover is the point of the plot.

## Layout

```
configs/config.yaml      # all knobs (subset, sizes, epochs, batch sweep)
src/data/                # cads.py (HF -> boxes + slice labels), synthetic.py, datasets.py
src/models/              # faster_rcnn_3d.py, slice_classifier.py (architectures only)
src/detector/            # detector TRACK: train.py, evaluate.py, benchmark.py
src/classifier/          # classifier TRACK: train.py, evaluate.py, benchmark.py
src/eval/                # metrics.py, compare.py (side-by-side report, owns no model logic)
src/utils/common.py      # shared config/device/cohort/checkpoint helpers
scripts/train_detector.py / test_detector.py      # work on the detector alone
scripts/train_classifier.py / test_classifier.py  # work on the classifier alone
scripts/run_benchmark.py # thin orchestrator: trains/loads both, compares latency+memory
notebooks/colab_benchmark.ipynb
```

To improve a model, edit its track package (`src/detector/` or
`src/classifier/`) and iterate with that track's `train_*` / `test_*`
scripts — `run_benchmark.py` picks the changes up untouched.

## Cite

```bibtex
@article{xu2025cads,
  title={CADS: A Comprehensive Anatomical Dataset and Segmentation for Whole-Body Anatomy in CT},
  author={Xu, Murong and others},
  journal={arXiv preprint arXiv:2507.22953}, year={2025}
}
```
Check each subset's `README.md` license before use (CC BY / BY-NC-SA / custom vary by source).
