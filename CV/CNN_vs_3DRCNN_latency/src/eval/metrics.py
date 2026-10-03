"""Quality metrics (context for the memory/time comparison)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..models.faster_rcnn_3d import box_iou_3d


def det_ap_3d(pred_boxes_list, gt_boxes_list, iou_thresh: float = 0.3) -> float:
    """Single-class AP over a list of volumes. pred: list of (boxes, scores)."""
    hits, scores, n_gt = [], [], 0
    for (pb, ps), gb in zip(pred_boxes_list, gt_boxes_list):
        gb_t = torch.as_tensor(np.asarray(gb), dtype=torch.float32)
        n_gt += len(gb_t)
        if len(pb) == 0:
            continue
        pb_t = torch.as_tensor(np.asarray(pb), dtype=torch.float32)
        order = np.argsort(-np.asarray(ps))
        matched = set()
        ious = box_iou_3d(pb_t, gb_t) if len(gb_t) else torch.zeros((len(pb_t), 0))
        for i in order:
            best, arg = (-1.0, -1)
            if len(gb_t):
                arg = int(ious[i].argmax())
                best = float(ious[i][arg])
            if best >= iou_thresh and arg not in matched:
                matched.add(arg)
                hits.append(1)
            else:
                hits.append(0)
            scores.append(float(np.asarray(ps)[i]))
    if n_gt == 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    hits = np.asarray(hits)[order]
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    rec = np.cumsum(hits) / n_gt
    # 11-point-ish AUC via step integral
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]) + rec[0] * prec[0]) if len(rec) else 0.0
    return ap


def slice_scores(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {"ap": 0.0, "auroc": 0.5, "positivity": float(y_true.mean()) if len(y_true) else 0.0}
    if len(np.unique(y_true)) < 2:
        return out
    out["ap"] = float(average_precision_score(y_true, y_prob))
    try:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return out
