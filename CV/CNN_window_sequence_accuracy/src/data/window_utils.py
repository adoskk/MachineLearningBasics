"""Train-only q95 window size and contiguous-window labeling."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class WindowRecord:
    volume_index: int
    volume_id: str
    subset: str
    start: int
    end: int
    label: int


def axial_component_extents(mask: np.ndarray, min_voxels: int = 200) -> list[int]:
    labels, n = ndi.label(mask > 0)
    extents = []
    for i in range(1, n + 1):
        zz = np.where(labels == i)[0]
        if len(zz) >= min_voxels:
            extents.append(int(zz.max() - zz.min() + 1))
    return extents


def compute_window_spec(train_samples, percentile=95, min_voxels=200,
                        output_path="results/window_q95.json") -> dict:
    extents = [
        extent
        for sample in train_samples
        for extent in axial_component_extents(sample.mask, min_voxels)
    ]
    if not extents:
        raise ValueError("No target components found in training masks.")
    max_depth = min(s.volume.shape[0] for s in train_samples)
    window_size = int(np.clip(math.ceil(np.percentile(extents, percentile)), 1, max_depth))
    spec = {
        "percentile": float(percentile),
        "window_size": window_size,
        "n_train_volumes": len(train_samples),
        "n_components": len(extents),
        "extent_slices": extents,
        "extent_min": int(np.min(extents)),
        "extent_median": float(np.median(extents)),
        "extent_max": int(np.max(extents)),
        "grid": "resized training masks only",
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec, indent=2))
    print(f"[window] q{percentile:g}={window_size} slices from {len(extents)} train objects")
    return spec


def window_starts(depth: int, window_size: int, stride: int = 1) -> list[int]:
    if window_size > depth:
        raise ValueError(f"window_size={window_size} exceeds depth={depth}")
    starts = list(range(0, depth - window_size + 1, stride))
    last = depth - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_window_records(samples, window_size: int, stride: int = 1) -> list[WindowRecord]:
    records = []
    for vi, sample in enumerate(samples):
        for start in window_starts(len(sample.slice_labels), window_size, stride):
            end = start + window_size
            label = int(np.asarray(sample.slice_labels[start:end]).max() > 0)
            records.append(WindowRecord(
                vi, sample.volume_id, sample.subset, start, end, label
            ))
    return records


def save_window_records(records, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(r) for r in records], indent=2))
