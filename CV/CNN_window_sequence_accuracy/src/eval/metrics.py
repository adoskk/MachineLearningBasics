"""Window-level metrics with validation-only thresholding and grouped CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def select_threshold(labels, probabilities, metric="balanced_accuracy"):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    best_t, best = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = probabilities >= threshold
        score = (
            accuracy_score(labels, pred)
            if metric == "balanced_accuracy" and len(np.unique(labels)) < 2
            else balanced_accuracy_score(labels, pred)
            if metric == "balanced_accuracy"
            else f1_score(labels, pred, zero_division=0)
        )
        if score > best:
            best_t, best = float(threshold), float(score)
    return best_t


def metric_bundle(labels, probabilities, threshold):
    y, p = np.asarray(labels).astype(int), np.asarray(probabilities)
    pred = p >= threshold
    balanced = (float(accuracy_score(y, pred)) if len(np.unique(y)) < 2
                else float(balanced_accuracy_score(y, pred)))
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": balanced,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ap": float(average_precision_score(y, p)),
        "positivity": float(y.mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    result["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else 0.5
    return result


def grouped_bootstrap(labels, probabilities, volume_ids, threshold,
                      repeats=500, seed=42):
    y, p, groups = np.asarray(labels), np.asarray(probabilities), np.asarray(volume_ids)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = {k: [] for k in ("accuracy", "balanced_accuracy", "f1", "ap", "auroc")}
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        bundle = metric_bundle(y[idx], p[idx], threshold)
        for key in values:
            values[key].append(bundle[key])
    return {
        key: {"low": float(np.percentile(vals, 2.5)),
              "high": float(np.percentile(vals, 97.5))}
        for key, vals in values.items()
    }
