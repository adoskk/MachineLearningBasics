"""Volume-safe splits and slice datasets shared by all classifiers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

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


def split_samples(samples: List[VolumeSample], splits=(0.8, 0.1, 0.1), seed=42):
    if len(samples) < 3:
        raise ValueError("Need at least 3 volumes for disjoint train/val/test splits.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n = len(samples)
    n_val = max(1, int(round(n * splits[1])))
    n_test = max(1, int(round(n * splits[2])))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Split {splits} leaves no training volumes for n={n}.")
    tr = [samples[int(i)] for i in idx[:n_train]]
    va = [samples[int(i)] for i in idx[n_train:n_train + n_val]]
    te = [samples[int(i)] for i in idx[n_train + n_val:]]
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
