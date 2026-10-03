"""Test ONLY the classifier: slice-level AP/AUROC on the held-out split.

    python scripts/test_classifier.py --synthetic
    python scripts/test_classifier.py --ckpt checkpoints/slice_classifier.pt --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier.evaluate import evaluate_classifier
from src.classifier.train import build_classifier
from src.utils.common import CLS_CKPT, get_device, load_cohort, load_config, log_version, maybe_load_checkpoint, split_cohort


def main():
    ap = argparse.ArgumentParser(description="Evaluate classifier quality (AP/AUROC).")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-root", default="data/cads", help="where NIfTI data lives")
    ap.add_argument("--ckpt", default=CLS_CKPT, help="checkpoint to evaluate (falls back to random init)")
    ap.add_argument("--batch-size", type=int, default=None, help="slice batch size (default: config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_version("test-cls")
    device = get_device(cfg.get("benchmark", {}).get("device", "auto"))
    _, _, te = split_cohort(load_cohort(cfg, args.synthetic, data_root=args.data_root), cfg)
    model = build_classifier(cfg, device, pretrained=False)
    maybe_load_checkpoint(model, args.ckpt, device)
    bs = args.batch_size or int(cfg.get("classification", {}).get("batch_size", 32))
    rep = evaluate_classifier(model, te, device, batch_size=bs)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
