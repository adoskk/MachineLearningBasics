"""Comparison: run both tracks' benchmarks and report them side by side.

This module owns NOTHING model-specific — it calls
src/detector/benchmark.py and src/classifier/benchmark.py, attaches
param counts, and writes the shared `benchmark.json` schema:

    {device, n_volumes_benched,
     detector: {params, model_mb, mean_ms_per_volume, ..., ap_iou03},
     slice_classifier: {params, model_mb, batch_1: {...}, batch_8: {...}, ...}}
"""
from __future__ import annotations

import json
from pathlib import Path

from ..classifier.benchmark import benchmark_classifier
from ..detector.benchmark import benchmark_detector


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def compare_and_save(det_model, cls_model, samples, device, cfg: dict,
                     save_dir: str = "results") -> dict:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    b = cfg.get("benchmark", {})
    rep = {
        "device": str(device),
        "n_volumes_benched": min(int(b.get("repeats", 20)), len(samples)),
        "detector": {
            "params": count_params(det_model),
            "model_mb": model_mb(det_model),
            **benchmark_detector(det_model, samples, device,
                                 warmup=int(b.get("warmup", 5)),
                                 repeats=int(b.get("repeats", 20))),
        },
        "slice_classifier": {
            "params": count_params(cls_model),
            "model_mb": model_mb(cls_model),
            **benchmark_classifier(cls_model, samples, device,
                                   batch_sizes=tuple(b.get("batch_sizes_cls", [1, 8, 32])),
                                   warmup=int(b.get("warmup", 5)),
                                   repeats=int(b.get("repeats", 20))),
        },
    }
    with open(f"{save_dir}/benchmark.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    d_ms = rep["detector"]["mean_ms_per_volume"]
    for k, v in rep["slice_classifier"].items():
        if k.startswith("batch_"):
            print(f"[compare] detector {d_ms:.1f} ms/vol vs classifier@{k} {v['mean_ms_per_volume']:.1f} ms/vol "
                  f"({d_ms/max(v['mean_ms_per_volume'],1e-6):.1f}x) | "
                  f"peak CUDA det={rep['detector']['peak_cuda_mb']:.0f}MB vs cls={v['peak_cuda_mb']:.0f}MB")
    return rep


# Backwards-compatible alias (old name from the monolithic benchmark module).
summarize = compare_and_save
