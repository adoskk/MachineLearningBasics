"""Classifier track: latency + memory benchmark (called by run_benchmark.py).

Sweeps the slice batch size — the knob that makes batched 2D inference
competitive with (or faster than) one heavy 3D pass. Per batch size:
per-volume wall time, slices/sec throughput, peak CUDA memory, CPU RSS
delta, and slice AP/AUROC as quality context.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..eval.metrics import slice_scores
from ..utils.common import mem_mb


@torch.no_grad()
def benchmark_classifier(model, samples, device: torch.device,
                         batch_sizes=(1, 8, 32),
                         warmup: int = 5, repeats: int = 20) -> dict:
    model = model.to(device).eval()
    out = {}
    vols = samples[: max(repeats, 1)]
    for bs in batch_sizes:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rss0 = mem_mb()
        # warmup
        for s in vols[: min(warmup, len(vols))]:
            _ = model.predict_volume(torch.from_numpy(s.volume), batch_size=bs, device=device)
        ts, yt, yp = [], [], []
        for s in vols[:repeats]:
            v = torch.from_numpy(s.volume)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            p = model.predict_volume(v, batch_size=bs, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ts.append((time.perf_counter() - t0) * 1000)
            yt.extend(s.slice_labels.tolist())
            yp.extend(p.tolist())
        ts = np.asarray(ts)
        q = slice_scores(np.asarray(yt), np.asarray(yp))
        out[f"batch_{bs}"] = {
            "mean_ms_per_volume": float(ts.mean()) if len(ts) else 0.0,
            "std_ms_per_volume": float(ts.std()) if len(ts) else 0.0,
            "volumes_per_sec": float(1000 / ts.mean()) if len(ts) and ts.mean() else 0.0,
            "slices_per_sec": float(1000 * vols[0].volume.shape[0] / ts.mean()) if len(ts) and ts.mean() else 0.0,
            "peak_cuda_mb": float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0,
            "cpu_rss_delta_mb": float(mem_mb() - rss0),
            **q,
        }
    return out
