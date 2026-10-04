"""Component and operator profiling for the 3D detector vs slice classifier.

Two complementary views are produced:

1. Stage wall time / incremental peak memory:
   detector = backbone, RPN convs, anchors, proposal decode, RPN NMS,
              RoI pooling, box head, final NMS + device-to-host copy.
   classifier = slice preparation/H2D, 2D CNN, sigmoid/device-to-host.

2. torch.profiler operator view:
   top Conv3d/Conv2d/pooling/sort/copy/etc. operators by device or CPU time.

The normal benchmark remains the source of truth for end-to-end latency.
Stage timing synchronizes after each stage to attribute work, so its sum has
extra synchronization overhead and is intended to explain *why*, not replace
the end-to-end number.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..models.faster_rcnn_3d import decode_boxes, nms_3d
from .compare import model_mb
from ..utils.common import mem_mb


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(device: torch.device, fn: Callable):
    """Return (result, wall_ms, incremental_peak_mb, allocated_after_mb)."""
    _sync(device)
    rss0 = mem_mb()
    if device.type == "cuda":
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        before = 0
    t0 = time.perf_counter()
    result = fn()
    _sync(device)
    elapsed = (time.perf_counter() - t0) * 1000
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        inc_peak = max(0, peak - before) / 1e6
        after = torch.cuda.memory_allocated(device) / 1e6
    else:
        inc_peak = max(0.0, mem_mb() - rss0)
        after = mem_mb()
    return result, elapsed, inc_peak, after


def _append(acc: dict, name: str, ms: float, peak: float, allocated: float) -> None:
    row = acc.setdefault(name, {"times": [], "peaks": [], "allocated": []})
    row["times"].append(ms)
    row["peaks"].append(peak)
    row["allocated"].append(allocated)


def _finish(acc: dict) -> dict:
    total = sum(float(np.mean(v["times"])) for v in acc.values())
    out = {}
    for name, v in acc.items():
        mean = float(np.mean(v["times"]))
        out[name] = {
            "mean_ms": mean,
            "std_ms": float(np.std(v["times"])),
            "percent_of_stage_sum": 100.0 * mean / max(total, 1e-9),
            "incremental_peak_mb": float(np.max(v["peaks"])),
            "allocated_after_mb": float(np.max(v["allocated"])),
        }
    return out


@torch.no_grad()
def _detector_staged_once(model, volume: torch.Tensor, device: torch.device, acc: dict):
    """Run B=1 inference while timing the same operations as model.forward."""
    _, _, D, H, W = volume.shape

    feats, ms, peak, alloc = _measure(device, lambda: model.backbone(volume))
    _append(acc, "backbone_3d_convs", ms, peak, alloc)
    fd, fh, fw = feats.shape[2:]

    def rpn_heads():
        # Deliberately matches the current model: shared 3D conv is called once
        # for classification and once for regression. Profiling makes this
        # duplicated cost visible and motivates caching it as an optimization.
        logit = model.rpn_cls(F.relu(model.rpn_conv(feats)))
        delta = model.rpn_reg(F.relu(model.rpn_conv(feats)))
        logit = logit.permute(0, 2, 3, 4, 1).reshape(1, -1)
        delta = delta.permute(0, 2, 3, 4, 1).reshape(1, -1, 6)
        return logit, delta

    (logit_f, delta_f), ms, peak, alloc = _measure(device, rpn_heads)
    _append(acc, "rpn_3d_convs_and_heads", ms, peak, alloc)

    anchors, ms, peak, alloc = _measure(
        device, lambda: model._anchors_for((fd, fh, fw), (D, H, W), device)
    )
    _append(acc, "anchor_generation", ms, peak, alloc)

    def decode_topk():
        scores = logit_f[0].sigmoid()
        k = min(model.rpn_pre, scores.numel())
        top = scores.topk(k).indices
        prop = decode_boxes(anchors[top], delta_f[0][top])
        prop[:, [0, 3]] = prop[:, [0, 3]].clamp(0, W)
        prop[:, [1, 4]] = prop[:, [1, 4]].clamp(0, H)
        prop[:, [2, 5]] = prop[:, [2, 5]].clamp(0, D)
        return scores, top, prop

    (scores, top, prop), ms, peak, alloc = _measure(device, decode_topk)
    _append(acc, "proposal_topk_and_decode", ms, peak, alloc)

    keep, ms, peak, alloc = _measure(
        device, lambda: nms_3d(prop, scores[top], model.nms_thresh, model.rpn_post)
    )
    _append(acc, "rpn_nms_3d", ms, peak, alloc)
    prop = prop[keep]
    if len(prop) == 0:
        return

    pooled, ms, peak, alloc = _measure(
        device, lambda: model._roi_pool(feats[0], prop)
    )
    _append(acc, "roi_pool_3d", ms, peak, alloc)

    def box_head():
        h = model.head_fc(pooled)
        cls = model.cls_head(h).softmax(-1)[:, 1]
        reg = model.reg_head(h)
        boxes = decode_boxes(prop, reg)
        mask = cls > model.score_thresh
        return boxes[mask], cls[mask]

    (boxes, cls), ms, peak, alloc = _measure(device, box_head)
    _append(acc, "box_fc_heads", ms, peak, alloc)

    def final_nms_copy():
        keep2 = nms_3d(boxes, cls, model.nms_thresh)
        return boxes[keep2].cpu(), cls[keep2].cpu()

    _, ms, peak, alloc = _measure(device, final_nms_copy)
    _append(acc, "final_nms_and_d2h", ms, peak, alloc)


@torch.no_grad()
def profile_detector_stages(model, volume: torch.Tensor, device: torch.device,
                            warmup: int = 3, repeats: int = 10) -> dict:
    model = model.to(device).eval()
    volume = volume.to(device)
    for _ in range(warmup):
        _ = model(volume)

    e2e = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        _ = model(volume)
        _sync(device)
        e2e.append((time.perf_counter() - t0) * 1000)

    acc = {}
    for _ in range(repeats):
        _detector_staged_once(model, volume, device, acc)
    stages = _finish(acc)
    return {
        "end_to_end_mean_ms": float(np.mean(e2e)),
        "end_to_end_std_ms": float(np.std(e2e)),
        "stage_sum_ms": sum(v["mean_ms"] for v in stages.values()),
        "model_mb": model_mb(model),
        "stages": stages,
    }


@torch.no_grad()
def profile_classifier_stages(model, volume: torch.Tensor, device: torch.device,
                              batch_size: int = 32, warmup: int = 3,
                              repeats: int = 10) -> dict:
    model = model.to(device).eval()
    if volume.dim() == 4:
        volume = volume[0]
    for _ in range(warmup):
        _ = model.predict_volume(volume, batch_size=batch_size, device=device)

    e2e = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        _ = model.predict_volume(volume, batch_size=batch_size, device=device)
        _sync(device)
        e2e.append((time.perf_counter() - t0) * 1000)

    acc = {}
    for _ in range(repeats):
        for i in range(0, volume.shape[0], batch_size):
            sl, ms, peak, alloc = _measure(
                device,
                lambda i=i: ((volume[i:i + batch_size].unsqueeze(1).to(device) - 0.35) / 0.2),
            )
            _append(acc, "slice_batch_prepare_and_h2d", ms, peak, alloc)
            logits, ms, peak, alloc = _measure(device, lambda sl=sl: model(sl))
            _append(acc, "cnn_2d_forward", ms, peak, alloc)
            _, ms, peak, alloc = _measure(device, lambda logits=logits: logits.sigmoid().cpu())
            _append(acc, "sigmoid_and_d2h", ms, peak, alloc)

    # Accumulator contains per-batch calls. Convert mean/call to mean/volume.
    batches = int(np.ceil(volume.shape[0] / batch_size))
    stages = _finish(acc)
    for row in stages.values():
        row["mean_ms"] *= batches
        row["std_ms"] *= batches
    stage_sum = sum(v["mean_ms"] for v in stages.values())
    for row in stages.values():
        row["percent_of_stage_sum"] = 100.0 * row["mean_ms"] / max(stage_sum, 1e-9)
    return {
        "batch_size": batch_size,
        "n_slices": int(volume.shape[0]),
        "end_to_end_mean_ms": float(np.mean(e2e)),
        "end_to_end_std_ms": float(np.std(e2e)),
        "stage_sum_ms": stage_sum,
        "model_mb": model_mb(model),
        "stages": stages,
    }


def _event_device_time(event) -> float:
    return float(getattr(event, "self_device_time_total",
                         getattr(event, "self_cuda_time_total", 0.0)) or 0.0)


def _event_device_memory(event) -> int:
    return int(getattr(event, "self_device_memory_usage",
                       getattr(event, "self_cuda_memory_usage", 0)) or 0)


def _operator_rows(prof, device: torch.device, limit: int = 20) -> list:
    rows = []
    for e in prof.key_averages(group_by_input_shape=True):
        row = {
            "operator": e.key,
            "calls": int(e.count),
            "self_cpu_ms": float(e.self_cpu_time_total / 1000),
            "cpu_total_ms": float(e.cpu_time_total / 1000),
            "self_device_ms": _event_device_time(e) / 1000,
            "self_device_memory_mb": _event_device_memory(e) / 1e6,
            "input_shapes": str(e.input_shapes),
        }
        rows.append(row)
    key = "self_device_ms" if device.type == "cuda" else "self_cpu_ms"
    return sorted(rows, key=lambda r: r[key], reverse=True)[:limit]


@torch.no_grad()
def profile_operators(det_model, cls_model, volume: torch.Tensor,
                      device: torch.device, cls_batch_size: int = 32) -> dict:
    """One representative end-to-end run per model using torch.profiler."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    det_model, cls_model = det_model.to(device).eval(), cls_model.to(device).eval()
    det_input = volume.unsqueeze(0).unsqueeze(0).to(device)

    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof_det:
        _ = det_model(det_input)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof_cls:
        _ = cls_model.predict_volume(volume, batch_size=cls_batch_size, device=device)
    return {
        "detector": _operator_rows(prof_det, device),
        "slice_classifier": _operator_rows(prof_cls, device),
    }


def _plot_profile(report: dict, out_dir: Path) -> None:
    det = report["detector"]["stages"]
    cls = report["slice_classifier"]["stages"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, (title, rows) in zip(
        ax, [("3D Faster R-CNN stages", det), ("Slice-classifier stages", cls)]
    ):
        names = list(rows)
        vals = [rows[n]["mean_ms"] for n in names]
        axis.barh(names, vals)
        axis.invert_yaxis()
        axis.set(title=title, xlabel="wall time per volume (ms)")
        for i, v in enumerate(vals):
            axis.text(v, i, f" {v:.1f} ms", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "profile_stage_latency.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, (title, rows) in zip(
        ax, [("Detector incremental peak", det), ("Classifier incremental peak", cls)]
    ):
        names = list(rows)
        vals = [rows[n]["incremental_peak_mb"] for n in names]
        axis.barh(names, vals)
        axis.invert_yaxis()
        axis.set(title=title, xlabel="incremental peak memory (MB)")
    fig.tight_layout()
    fig.savefig(out_dir / "profile_stage_memory.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    ops = report["operators"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for axis, key, title in [
        (ax[0], "detector", "Detector: top operators"),
        (ax[1], "slice_classifier", "Classifier: top operators"),
    ]:
        rows = ops[key][:12]
        metric = "self_device_ms" if report["device"].startswith("cuda") else "self_cpu_ms"
        names = [r["operator"][:38] for r in rows][::-1]
        vals = [r[metric] for r in rows][::-1]
        axis.barh(names, vals)
        axis.set(title=title, xlabel=f"{metric} (one representative volume)")
    fig.tight_layout()
    fig.savefig(out_dir / "profile_top_operators.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def profile_models(det_model, cls_model, sample, device: torch.device,
                   cls_batch_size: int = 32, warmup: int = 3,
                   repeats: int = 10, out_dir: str = "results") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vol = torch.from_numpy(sample.volume).float()
    report = {
        "device": str(device),
        "volume_shape_dhw": list(sample.volume.shape),
        "note": ("End-to-end is canonical. Stage sums synchronize after every "
                 "stage and are diagnostic, so they include extra overhead."),
        "detector": profile_detector_stages(
            det_model, vol.unsqueeze(0).unsqueeze(0), device, warmup, repeats
        ),
        "slice_classifier": profile_classifier_stages(
            cls_model, vol, device, cls_batch_size, warmup, repeats
        ),
    }
    report["operators"] = profile_operators(
        det_model, cls_model, vol, device, cls_batch_size
    )
    with open(out / "profile.json", "w") as f:
        json.dump(report, f, indent=2)
    _plot_profile(report, out)

    print("[profile] canonical end-to-end: "
          f"detector={report['detector']['end_to_end_mean_ms']:.1f}ms, "
          f"classifier={report['slice_classifier']['end_to_end_mean_ms']:.1f}ms")
    print("[profile] detector stage attribution:")
    for name, row in report["detector"]["stages"].items():
        print(f"  {name:28s} {row['mean_ms']:8.2f}ms "
              f"({row['percent_of_stage_sum']:5.1f}%) "
              f"incremental_peak={row['incremental_peak_mb']:.1f}MB")
    print(f"[profile] outputs -> {out}/profile*.png + profile.json")
    return report
