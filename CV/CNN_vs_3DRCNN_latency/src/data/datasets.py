"""PyTorch datasets shared by both models (built from VolumeSample lists)."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .cads import VolumeSample


class DetectionDataset(Dataset):
    """Yields (volume 1xDxHxW float, target dict with boxes Nx6 + labels)."""

    def __init__(self, samples: List[VolumeSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        vol = torch.from_numpy(s.volume).unsqueeze(0).float()  # 1,D,H,W
        boxes = torch.from_numpy(s.boxes).float() if len(s.boxes) else torch.zeros((0, 6), dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # single foreground class
        return vol, {"boxes": boxes, "labels": labels, "volume_id": s.volume_id}


def det_collate(batch):
    vols, targets = zip(*batch)
    return torch.stack(vols), list(targets)


class SliceDataset(Dataset):
    """Yields (slice 1xHxW float, label 0/1). One row per axial slice."""

    def __init__(self, samples: List[VolumeSample]):
        self.items = []
        for s in samples:
            D = s.volume.shape[0]
            for z in range(D):
                self.items.append((s.volume[z], int(s.slice_labels[z]), f"{s.volume_id}:{z}"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, lab, key = self.items[i]
        img = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
        # ImageNet norm approx for single channel replicated: use mean/std of windowed CT
        img = (img - 0.35) / 0.2
        return img, torch.tensor(lab, dtype=torch.float32), key


def split_samples(samples: List[VolumeSample], splits=(0.8, 0.1, 0.1), seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n = len(samples)
    n_tr = int(n * splits[0])
    n_va = int(n * splits[1])
    tr = [samples[int(i)] for i in idx[:n_tr]]
    va = [samples[int(i)] for i in idx[n_tr:n_tr + n_va]]
    te = [samples[int(i)] for i in idx[n_tr + n_va:]] or va  # tiny cohorts: reuse val
    return tr, va, te
