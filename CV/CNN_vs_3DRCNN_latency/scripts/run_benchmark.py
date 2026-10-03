"""Comparison orchestrator: train (optional) -> per-track benchmarks -> compare.

Thin by design — all detector logic lives in src/detector/, all
classifier logic in src/classifier/. To improve a track, edit that
package (or its scripts/train_* / test_* entry points); this file only
wires the two tracks together for the latency/memory comparison.

    # fastest CPU smoke test (no HF access):
    python scripts/run_benchmark.py --synthetic --no-train --repeats 4

    # small real run (after download_subset.py + HF login):
    python scripts/run_benchmark.py --config configs/config.yaml --train --repeats 20

Per-track entry points for independent work:
    scripts/train_detector.py / scripts/test_detector.py
    scripts/train_classifier.py / scripts/test_classifier.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier.train import build_classifier, train_classifier
from src.detector.train import build_detector, train_detector
from src.eval.compare import compare_and_save
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
    ap = argparse.ArgumentParser(description="Compare detector vs classifier latency/memory.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic volumes")
    ap.add_argument("--data-root", default="data/cads", help="where NIfTI data lives")
    ap.add_argument("--no-train", action="store_true", help="skip training, bench checkpointed/random nets")
    ap.add_argument("--train", action="store_true", help="train both tracks a few epochs first")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_version("run")
    b = cfg.setdefault("benchmark", {})
    if args.repeats:
        b["repeats"] = args.repeats
    if args.out:
        b["output_dir"] = args.out
    device = get_device(b.get("device", "auto"))
    print(f"[run] device={device}")

    tr, va, te = split_cohort(load_cohort(cfg, args.synthetic, data_root=args.data_root), cfg)

    if args.train and not args.no_train:
        print("[run] training both tracks (see src/detector/train.py, src/classifier/train.py)...")
        det = train_detector(tr, va, cfg, device)
        cls = train_classifier(tr, va, cfg, device)
    else:
        det = build_detector(cfg, device, pretrained=False)
        cls = build_classifier(cfg, device, pretrained=False)
        maybe_load_checkpoint(det, DET_CKPT, device)
        maybe_load_checkpoint(cls, CLS_CKPT, device)

    compare_and_save(det, cls, te, device, cfg, save_dir=b.get("output_dir", "results"))


if __name__ == "__main__":
    main()
