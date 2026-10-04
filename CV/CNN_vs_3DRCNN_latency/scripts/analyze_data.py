"""Exploratory data analysis for the lesion cohort (standalone entry point).

    python scripts/analyze_data.py --synthetic --out results
    python scripts/analyze_data.py --config configs/config.yaml --data-root /content/data/cads
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.analyze import run_eda
from src.utils.common import load_config, log_version


def main():
    ap = argparse.ArgumentParser(description="EDA: splits, resolution, object sizes/locations.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true", help="analyze synthetic phantoms")
    ap.add_argument("--data-root", default="data/cads", help="where NIfTI data lives")
    ap.add_argument("--out", default="results", help="output directory for json + figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log_version("eda")
    run_eda(cfg, data_root=args.data_root, synthetic=args.synthetic, out_dir=args.out)


if __name__ == "__main__":
    main()
