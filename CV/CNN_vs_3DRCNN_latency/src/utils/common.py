"""Shared runnable helpers: config, device, cohorts, checkpoints.

Model-agnostic on purpose — per-model logic lives in src/detector/ and
src/classifier/. All scripts (train_*, test_*, run_benchmark) share these
so flags behave identically everywhere.
"""
from __future__ import annotations

from pathlib import Path

import psutil
import torch
import yaml

DET_CKPT = "checkpoints/faster_rcnn_3d.pt"
CLS_CKPT = "checkpoints/slice_classifier.pt"

# Bump on every functional change. Scripts print this at startup so you can
# verify Colab/Drive is executing the version you think it is.
VERSION = "0.8.0"


def log_version(tag: str) -> None:
    print(f"[{tag}] modiface v{VERSION} | cwd={Path.cwd()}")


def get_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    if Path(path).exists():
        return yaml.safe_load(open(path)) or {}
    return {}


def mem_mb() -> float:
    """Current process RSS in MB (CPU-RAM footprint probe)."""
    return psutil.Process().memory_info().rss / 1e6


def load_cohort(cfg: dict, synthetic: bool = False, data_root: str = "data/cads"):
    """Load VolumeSamples: synthetic phantoms or a small real CADS subset."""
    from ..data.cads import load_volumes
    from ..data.synthetic import make_synthetic_cohort

    ds, pp = cfg.get("dataset", {}), cfg.get("preprocess", {})
    print(f"[data] root={Path(data_root).resolve()}")
    parts = tuple(ds.get("seg_parts", [551]))
    subset_labels = ds.get("subset_labels") or {}
    print(f"[data] seg parts={list(parts)} labels={subset_labels or 'all foreground'}")
    shape = tuple(pp.get("target_shape", [64, 128, 128]))
    if synthetic or ds.get("use_synthetic", False):
        n = int(ds.get("n_synthetic", 8))
        samples = make_synthetic_cohort(n=n, shape=shape, seed=int(ds.get("seed", 42)))
        print(f"[data] synthetic cohort: {n} volumes @ {shape}")
    else:
        samples = load_volumes(
            data_root, ds.get("subsets", ["0003_kits21"]), int(ds.get("n_volumes", 30)),
            shape, tuple(pp.get("hu_window", [-160, 240])),
            int(pp.get("slice_positive_thresh", 10)), int(pp.get("min_component_voxels", 200)),
            int(pp.get("min_box_size", 5)), int(ds.get("seed", 42)),
            parts, subset_labels,
        )
        print(f"[data] real cohort: {len(samples)} volumes")
    if not samples:
        raise RuntimeError(
            f"0 volumes loaded from {Path(data_root).resolve()}. "
            "If files are visible with ls but unreadable ('Transport endpoint is "
            "not connected'), the Drive FUSE mount dropped — remount with "
            "drive.mount('/content/drive', force_remount=True), or keep bulk "
            "data on local SSD (/content/...) instead of Drive."
        )
    return samples


def split_cohort(samples, cfg: dict):
    from ..data.datasets import split_samples

    ds = cfg.get("dataset", {})
    tr, va, te = split_samples(samples, tuple(ds.get("split", [0.8, 0.1, 0.1])), int(ds.get("seed", 42)))
    print(f"[data] split train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    return tr, va, te


def maybe_load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> bool:
    """Load weights if present. Returns True on success (False -> keep init)."""
    p = Path(path)
    if not p.exists():
        print(f"[ckpt] no checkpoint at {p}, using current init")
        return False
    try:
        model.load_state_dict(torch.load(p, map_location=device))
        print(f"[ckpt] loaded {p}")
        return True
    except Exception as e:
        print(f"[ckpt] load skipped ({p}): {e}")
        return False
