"""Test ONLY the detector: AP@IoU3D quality on the held-out split.

    python scripts/test_detector.py --synthetic
    python scripts/test_detector.py --ckpt checkpoints/faster_rcnn_3d.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector.evaluate import evaluate_detector
from src.detector.train import build_detector
from src.utils.common import DET_CKPT, get_device, load_cohort, load_config, log_version, maybe_load_checkpoint, split_cohort


def main():
    ap = argparse.ArgumentParser(description="Evaluate detector quality (AP@IoU3D).")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-root", default="data/cads", help="where NIfTI data lives")
    ap.add_argument("--ckpt", default=DET_CKPT, help="checkpoint to evaluate (falls back to random init)")
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_version("test-det")
    device = get_device(cfg.get("benchmark", {}).get("device", "auto"))
    _, _, te = split_cohort(load_cohort(cfg, args.synthetic, data_root=args.data_root), cfg)
    model = build_detector(cfg, device, pretrained=False)
    maybe_load_checkpoint(model, args.ckpt, device)
    rep = evaluate_detector(model, te, device, iou_thresh=args.iou)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
