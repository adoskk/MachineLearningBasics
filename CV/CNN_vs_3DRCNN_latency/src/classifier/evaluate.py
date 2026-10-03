"""Classifier track: quality evaluation (test script + benchmark context).

Scores every axial slice and reports slice-level AP / AUROC.
Untimed — for latency/memory see benchmark.py.
"""
from __future__ import annotations

import numpy as np
import torch

from ..eval.metrics import slice_scores


@torch.no_grad()
def evaluate_classifier(model, samples, device: torch.device, batch_size: int = 32) -> dict:
    model = model.to(device).eval()
    yt, yp = [], []
    for s in samples:
        p = model.predict_volume(torch.from_numpy(s.volume), batch_size=batch_size, device=device)
        yt.extend(s.slice_labels.tolist())
        yp.extend(p.tolist())
    q = slice_scores(np.asarray(yt), np.asarray(yp))
    return {
        **q,
        "batch_size": int(batch_size),
        "n_volumes": len(samples),
        "n_slices": int(len(yt)),
    }
