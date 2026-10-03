"""Detector track: quality evaluation (test script + benchmark context).

Reports single-class AP over 3D boxes at a fixed IoU threshold.
Untimed — for latency/memory see benchmark.py.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import DetectionDataset, det_collate
from ..eval.metrics import det_ap_3d


@torch.no_grad()
def evaluate_detector(model, samples, device: torch.device, iou_thresh: float = 0.3) -> dict:
    model = model.to(device).eval()
    loader = DataLoader(DetectionDataset(samples), batch_size=1, collate_fn=det_collate)
    preds, gt = [], []
    n_pred = 0
    for vols, tgts in loader:
        out = model(vols.to(device))
        preds.append((out[0]["boxes"].numpy(), out[0]["scores"].numpy()))
        gt.append(tgts[0]["boxes"].numpy())
        n_pred += len(out[0]["boxes"])
    ap = det_ap_3d(preds, gt, iou_thresh=iou_thresh)
    return {
        "ap": float(ap),
        "iou_thresh": float(iou_thresh),
        "n_volumes": len(samples),
        "n_gt_boxes": int(sum(len(g) for g in gt)),
        "n_pred_boxes": int(n_pred),
    }
