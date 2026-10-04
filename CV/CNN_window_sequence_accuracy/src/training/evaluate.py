"""Fair held-out comparison of max-slice, linear, and LSTM window models."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from ..eval.metrics import grouped_bootstrap, metric_bundle, select_threshold
from ..models.temporal_heads import count_parameters
from .train_temporal import predict_head


def baseline_predictions(dataset):
    return {
        "probabilities": np.asarray([
            float(dataset[i]["baseline_prob"]) for i in range(len(dataset))
        ]),
        "labels": dataset.labels.copy(),
        "volume_ids": [r.volume_id for r in dataset.records],
        "starts": [r.start for r in dataset.records],
        "subsets": [r.subset for r in dataset.records],
    }


def _latency(model, dataset, device):
    if model is None:
        t0 = time.perf_counter()
        _ = baseline_predictions(dataset)
        return (time.perf_counter() - t0) * 1000 / max(len(dataset), 1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _ = predict_head(model, dataset, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000 / max(len(dataset), 1)


def _subset_metrics(pred, threshold):
    out = {}
    subsets = np.asarray(pred["subsets"])
    for subset in np.unique(subsets):
        idx = np.where(subsets == subset)[0]
        out[str(subset)] = metric_bundle(
            np.asarray(pred["labels"])[idx],
            np.asarray(pred["probabilities"])[idx],
            threshold,
        )
    return out


def evaluate_models(models, val_ds, test_ds, cfg, device, output_dir="results"):
    """models excludes baseline; baseline_max is derived from cached slice logits."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_models = {"baseline_max": None, **models}
    val_predictions, test_predictions, report = {}, {}, {}
    threshold_metric = cfg["window"]["threshold_metric"]
    repeats = int(cfg["window"]["bootstrap_repeats"])

    for name, model in all_models.items():
        val_pred = baseline_predictions(val_ds) if model is None else predict_head(model, val_ds, device)
        test_pred = baseline_predictions(test_ds) if model is None else predict_head(model, test_ds, device)
        threshold = select_threshold(
            val_pred["labels"], val_pred["probabilities"], threshold_metric
        )
        metrics = metric_bundle(test_pred["labels"], test_pred["probabilities"], threshold)
        metrics["ci95_by_volume"] = grouped_bootstrap(
            test_pred["labels"], test_pred["probabilities"], test_pred["volume_ids"],
            threshold, repeats=repeats,
        )
        metrics["per_subset"] = _subset_metrics(test_pred, threshold)
        metrics["head_parameters"] = 0 if model is None else count_parameters(model)
        metrics["head_ms_per_window"] = _latency(model, test_ds, device)
        report[name] = metrics
        val_predictions[name], test_predictions[name] = val_pred, test_pred
        print(f"[test] {name:16s} acc={metrics['accuracy']:.3f} "
              f"bal_acc={metrics['balanced_accuracy']:.3f} "
              f"f1={metrics['f1']:.3f} ap={metrics['ap']:.3f}")

    payload = {
        "window_size": int(test_ds.records[0].end - test_ds.records[0].start),
        "n_test_windows": len(test_ds),
        "threshold_selected_on": "validation",
        "metrics": report,
    }
    (out / "window_comparison.json").write_text(json.dumps(payload, indent=2))
    with open(out / "window_comparison.csv", "w", newline="") as f:
        fields = ["model", "accuracy", "balanced_accuracy", "precision", "recall",
                  "f1", "auroc", "ap", "head_parameters", "head_ms_per_window"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, row in report.items():
            writer.writerow({"model": name, **{k: row[k] for k in fields[1:]}})
    serializable_predictions = {
        name: {
            "probabilities": pred["probabilities"].tolist(),
            "labels": pred["labels"].tolist(),
            "volume_ids": pred["volume_ids"],
            "starts": pred["starts"],
            "subsets": pred["subsets"],
        } for name, pred in test_predictions.items()
    }
    (out / "window_predictions.json").write_text(
        json.dumps(serializable_predictions, indent=2)
    )
    _plots(report, test_predictions, out)
    return payload


def _plots(report, predictions, out):
    names = list(report)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.22
    for offset, key in zip((-width, 0, width),
                           ("accuracy", "balanced_accuracy", "f1")):
        ax.bar(x + offset, [report[n][key] for n in names], width, label=key)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set(title="Held-out window classification", ylabel="score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_comparison.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, pred in predictions.items():
        y, p = pred["labels"], pred["probabilities"]
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            ax[0].plot(fpr, tpr, label=name)
            ax[1].plot(recall, precision, label=name)
    ax[0].set(title="ROC curves", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-recall curves", xlabel="recall", ylabel="precision")
    if ax[0].lines:
        ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([report[n]["head_ms_per_window"] for n in names],
               [report[n]["balanced_accuracy"] for n in names])
    for n in names:
        ax.annotate(n, (report[n]["head_ms_per_window"],
                        report[n]["balanced_accuracy"]), fontsize=8)
    ax.set(xlabel="head latency (ms/window)", ylabel="balanced accuracy",
           title="Accuracy versus temporal-head latency")
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_latency.png", dpi=130)
    plt.close(fig)

    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, name in zip(axes, names):
        cm = np.asarray(report[name]["confusion_matrix"])
        image = axis.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(cm[i, j]), ha="center", va="center")
        axis.set(title=name, xlabel="predicted", ylabel="true",
                 xticks=[0, 1], yticks=[0, 1])
    for axis in axes[len(names):]:
        axis.axis("off")
    fig.suptitle("Held-out confusion matrices (validation-selected thresholds)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=130)
    plt.close(fig)
