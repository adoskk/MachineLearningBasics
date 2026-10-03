"""Detector track: latency + memory benchmark (called by run_benchmark.py).

Measures per-volume wall time (warmup + CUDA-synced repeats), peak CUDA
memory, CPU RSS delta, and AP@IoU3D on the timed predictions as quality
context. Returns a plain dict — compare.py decides how to report it.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import DetectionDataset, det_collate
from ..eval.metrics import det_ap_3d
from ..utils.common import mem_mb


@torch.no_grad()
def benchmark_detector(model, samples, device: torch.device,
                       warmup: int = 5, repeats: int = 20,
                       iou_thresh: float = 0.3) -> dict:
    model = model.to(device).eval()
    loader = DataLoader(DetectionDataset(samples[: max(repeats, 1)]), batch_size=1,
                        collate_fn=det_collate)
    vols = [v.to(device) for v, _ in loader]
    gt = [s.boxes for s in samples[: len(vols)]]
    # warmup (caches, autotuner)
    for v in vols[: min(warmup, len(vols))]:
        _ = model(v)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss0 = mem_mb()
    ts, preds = [], []
    for v in vols[:repeats]:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model(v)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ts.append((time.perf_counter() - t0) * 1000)
        preds.append((out[0]["boxes"].numpy(), out[0]["scores"].numpy()))
    peak_cuda = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    ts = np.asarray(ts)
    ap = det_ap_3d(preds, gt[: len(preds)], iou_thresh=iou_thresh)
    return {
        "mean_ms_per_volume": float(ts.mean()) if len(ts) else 0.0,
        "std_ms_per_volume": float(ts.std()) if len(ts) else 0.0,
        "volumes_per_sec": float(1000 / ts.mean()) if len(ts) and ts.mean() else 0.0,
        "peak_cuda_mb": float(peak_cuda),
        "cpu_rss_delta_mb": float(mem_mb() - rss0),
        "ap_iou03": float(ap),
    }
