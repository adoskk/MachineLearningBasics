"""Frozen-CNN embedding extraction and window datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .datasets import normalize_slices
from .window_utils import build_window_records


@torch.no_grad()
def extract_embeddings(model, samples, device, batch_size=32,
                       output_path: str | Path | None = None):
    model = model.to(device).eval()
    entries = []
    for sample in samples:
        volume = torch.from_numpy(sample.volume).float()
        features, logits = [], []
        for start in range(0, len(volume), batch_size):
            batch = normalize_slices(volume[start:start + batch_size].unsqueeze(1)).to(device)
            feat = model.encode_slices(batch)
            features.append(feat.cpu())
            logits.append(model.classify_features(feat).cpu())
        entries.append({
            "volume_id": sample.volume_id,
            "subset": sample.subset,
            "embeddings": torch.cat(features),       # D,F
            "slice_logits": torch.cat(logits),       # D
            "slice_labels": torch.from_numpy(sample.slice_labels.copy()).long(),
        })
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(entries, p)
    return entries


def load_embedding_cache(path: str | Path):
    return torch.load(path, map_location="cpu", weights_only=False)


class WindowEmbeddingDataset(Dataset):
    """Fixed W x F embeddings with OR-over-slices window labels."""

    def __init__(self, entries, window_size, stride=1):
        self.entries = entries
        proxy = [
            type("WindowProxy", (), {
                "slice_labels": e["slice_labels"].numpy(),
                "volume_id": e["volume_id"],
                "subset": e.get("subset", ""),
            })()
            for e in entries
        ]
        self.records = build_window_records(proxy, window_size, stride)
        self.labels = np.asarray([r.label for r in self.records], dtype=np.int64)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        e = self.entries[r.volume_index]
        embeddings = e["embeddings"][r.start:r.end].float()
        baseline_prob = e["slice_logits"][r.start:r.end].sigmoid().max().float()
        return {
            "embeddings": embeddings,
            "label": torch.tensor(r.label, dtype=torch.float32),
            "baseline_prob": baseline_prob,
            "volume_id": r.volume_id,
            "subset": r.subset,
            "start": r.start,
            "end": r.end,
        }


def balanced_sampler(dataset: WindowEmbeddingDataset):
    labels = dataset.labels
    counts = np.bincount(labels, minlength=2)
    weights = np.asarray([1.0 / max(counts[y], 1) for y in labels], dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)
