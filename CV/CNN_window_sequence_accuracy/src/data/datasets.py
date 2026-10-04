"""Volume-safe splits and slice datasets shared by all classifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .cads import VolumeSample


def normalize_slices(x: torch.Tensor) -> torch.Tensor:
    return (x - 0.35) / 0.2


class SliceDataset(Dataset):
    def __init__(self, samples: List[VolumeSample]):
        self.items = [
            (s.volume[z], int(s.slice_labels[z]), s.volume_id, z)
            for s in samples for z in range(s.volume.shape[0])
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image, label, volume_id, z = self.items[i]
        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()
        return normalize_slices(image), torch.tensor(label, dtype=torch.float32), volume_id, z


def split_samples(samples: List[VolumeSample], splits=(0.7, 0.15, 0.15), seed=42):
    """Create deterministic, subset-stratified, volume-disjoint splits.

    Stratifying by CADS subset keeps the KiTS/LiTS mixture approximately equal
    across train, validation, and test. Small synthetic/single-subset cohorts
    are handled by the same code.
    """
    if len(samples) < 3:
        raise ValueError("Need at least 3 volumes for disjoint train/val/test splits.")
    if len(splits) != 3 or not np.isclose(sum(splits), 1.0) or min(splits) <= 0:
        raise ValueError(f"Expected three positive split fractions summing to 1, got {splits}.")

    labels = np.asarray([s.subset or "unknown" for s in samples])
    indices = np.arange(len(samples))
    try:
        train_idx, held_idx = train_test_split(
            indices,
            train_size=float(splits[0]),
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        held_labels = labels[held_idx]
        val_fraction = float(splits[1]) / float(splits[1] + splits[2])
        val_idx, test_idx = train_test_split(
            held_idx,
            train_size=val_fraction,
            random_state=seed + 1,
            shuffle=True,
            stratify=held_labels,
        )
    except ValueError as exc:
        # Tiny development cohorts may not have two samples per subset.
        print(f"[data] subset stratification unavailable ({exc}); using seeded volume split")
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
        n_val = max(1, int(round(len(samples) * splits[1])))
        n_test = max(1, int(round(len(samples) * splits[2])))
        n_train = len(samples) - n_val - n_test
        if n_train < 1:
            raise ValueError(f"Split {splits} leaves no training volumes for n={len(samples)}.")
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    tr = [samples[int(i)] for i in train_idx]
    va = [samples[int(i)] for i in val_idx]
    te = [samples[int(i)] for i in test_idx]
    return tr, va, te


def save_split_manifest(train, val, test, path: str | Path, seed: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train": [{"volume_id": s.volume_id, "subset": s.subset} for s in train],
        "val": [{"volume_id": s.volume_id, "subset": s.subset} for s in val],
        "test": [{"volume_id": s.volume_id, "subset": s.subset} for s in test],
    }
    p.write_text(json.dumps(payload, indent=2))
