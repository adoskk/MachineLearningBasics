"""Train ONLY the 3D Faster R-CNN track.

    python scripts/train_detector.py --synthetic
    python scripts/train_detector.py --config configs/config.yaml --out checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector.train import train_detector
from src.utils.common import get_device, load_cohort, load_config, log_version, split_cohort


def main():
    ap = argparse.ArgumentParser(description="Train the 3D Faster R-CNN detector track.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="train on synthetic phantoms")
    ap.add_argument("--data-root", default="data/cads", help="where NIfTI data lives")
    ap.add_argument("--out", default="checkpoints", help="checkpoint directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_version("train-det")
    device = get_device(cfg.get("benchmark", {}).get("device", "auto"))
    print(f"[train-det] device={device}")
    tr, va, _ = split_cohort(load_cohort(cfg, args.synthetic, data_root=args.data_root), cfg)
    train_detector(tr, va, cfg, device, out_dir=args.out)


if __name__ == "__main__":
    main()
