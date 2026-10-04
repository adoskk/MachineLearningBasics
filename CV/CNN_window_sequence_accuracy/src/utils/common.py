"""Shared config, device, cohort, split, and checkpoint helpers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
import yaml

VERSION = "1.1.0"
BASELINE_CKPT = "checkpoints/slice_baseline.pt"


def log_version(tag: str):
    print(f"[{tag}] window-sequence v{VERSION} | cwd={Path.cwd()}")


def load_config(path="configs/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def get_device(pref="auto") -> torch.device:
    return torch.device("cuda" if pref == "cuda" or
                        (pref == "auto" and torch.cuda.is_available()) else "cpu")


def load_cohort(cfg, synthetic=False, data_root="data/cads"):
    from ..data.cads import load_volumes
    from ..data.synthetic import make_synthetic_cohort

    ds, pp = cfg["dataset"], cfg["preprocess"]
    shape = tuple(pp["target_shape"])
    if synthetic or ds.get("use_synthetic", False):
        samples = make_synthetic_cohort(
            n=int(ds.get("n_synthetic", 12)), shape=shape, seed=int(ds["seed"])
        )
        for s in samples:
            s.subset = "synthetic"
    else:
        samples = load_volumes(
            data_root, ds["subsets"], int(ds["n_volumes"]), shape,
            tuple(pp["hu_window"]), int(pp["slice_positive_thresh"]),
            int(pp["min_component_voxels"]), int(pp["min_box_size"]),
            int(ds["seed"]), tuple(ds["seg_parts"]), ds["subset_labels"],
        )
    if not samples:
        raise RuntimeError(f"No volumes loaded from {Path(data_root).resolve()}")
    print(f"[data] loaded {len(samples)} volumes @ {shape}")
    return samples


def split_cohort(samples, cfg):
    from ..data.datasets import save_split_manifest, split_samples

    ds = cfg["dataset"]
    splits = split_samples(samples, tuple(ds["split"]), int(ds["split_seed"]))
    save_split_manifest(*splits, cfg["artifacts"]["split_manifest"], int(ds["split_seed"]))
    print("[data] train/val/test =", "/".join(str(len(x)) for x in splits))
    for name, split in zip(("train", "val", "test"), splits):
        print(f"[data] {name} subsets={dict(sorted(Counter(s.subset for s in split).items()))}")
    return splits


def load_state(model, path, device, required=True):
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing checkpoint: {p}")
        return False
    payload = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload))
    return payload
