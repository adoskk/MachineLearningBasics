"""Profile where detector/classifier latency and memory are spent.

Example (Colab GPU):
    python scripts/profile_models.py \
      --config configs/config.yaml \
      --data-root /content/data/cads \
      --repeats 10 --out results

Outputs:
    results/profile.json
    results/profile_stage_latency.png
    results/profile_stage_memory.png
    results/profile_top_operators.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier.train import build_classifier
from src.detector.train import build_detector
from src.eval.profile import profile_models
from src.utils.common import (
    CLS_CKPT,
    DET_CKPT,
    get_device,
    load_cohort,
    load_config,
    log_version,
    maybe_load_checkpoint,
    split_cohort,
)


def main():
    ap = argparse.ArgumentParser(
        description="Stage + operator profiler for detector vs classifier."
    )
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-root", default="data/cads")
    ap.add_argument("--det-ckpt", default=DET_CKPT)
    ap.add_argument("--cls-ckpt", default=CLS_CKPT)
    ap.add_argument("--cls-batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_version("profile")
    device = get_device(cfg.get("benchmark", {}).get("device", "auto"))
    print(f"[profile] device={device}")
    _, _, test = split_cohort(
        load_cohort(cfg, args.synthetic, data_root=args.data_root), cfg
    )
    if not test:
        raise RuntimeError("test split is empty; increase dataset.n_volumes")

    detector = build_detector(cfg, device, pretrained=False)
    classifier = build_classifier(cfg, device, pretrained=False)
    maybe_load_checkpoint(detector, args.det_ckpt, device)
    maybe_load_checkpoint(classifier, args.cls_ckpt, device)

    profile_models(
        detector,
        classifier,
        test[0],
        device,
        cls_batch_size=args.cls_batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
