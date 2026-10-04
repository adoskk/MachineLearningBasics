import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_baseline import train_baseline
from src.utils.common import get_device, load_cohort, load_config, log_version, split_cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data-root", default="data/cads")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log_version("train-baseline")
    device = get_device(cfg["runtime"]["device"])
    train, val, _ = split_cohort(load_cohort(cfg, args.synthetic, args.data_root), cfg)
    train_baseline(
        train, val, cfg, device, cfg["artifacts"]["baseline_checkpoint"]
    )


if __name__ == "__main__":
    main()
