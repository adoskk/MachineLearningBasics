"""Train ONLY the per-slice classifier track.

    python scripts/train_classifier.py --synthetic
    python scripts/train_classifier.py --config configs/config.yaml --out checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier.train import train_classifier
from src.utils.common import get_device, load_cohort, load_config, log_version, split_cohort


def main():
    ap = argparse.ArgumentParser(description="Train the per-slice classifier track.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="train on synthetic phantoms")
    ap.add_argument("--data-root", default="data/cads", help="where NIfTI data lives")
    ap.add_argument("--out", default="checkpoints", help="checkpoint directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_version("train-cls")
    device = get_device(cfg.get("benchmark", {}).get("device", "auto"))
    print(f"[train-cls] device={device}")
    tr, va, _ = split_cohort(load_cohort(cfg, args.synthetic, data_root=args.data_root), cfg)
    train_classifier(tr, va, cfg, device, out_dir=args.out)


if __name__ == "__main__":
    main()
