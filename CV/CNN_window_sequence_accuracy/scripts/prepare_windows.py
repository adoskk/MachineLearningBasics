import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embedding_cache import extract_embeddings
from src.data.window_utils import compute_window_spec
from src.models.slice_encoder import SliceBaseline
from src.utils.common import (
    get_device, load_cohort, load_config, load_state, log_version, split_cohort,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data-root", default="data/cads")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log_version("prepare")
    device = get_device(cfg["runtime"]["device"])
    train, val, test = split_cohort(load_cohort(cfg, args.synthetic, args.data_root), cfg)
    compute_window_spec(
        train, cfg["window"]["percentile"],
        cfg["preprocess"]["min_component_voxels"],
        cfg["artifacts"]["window_spec"],
    )
    model = SliceBaseline(False, cfg["baseline"]["backbone"])
    load_state(model, cfg["artifacts"]["baseline_checkpoint"], device)
    cache = Path(cfg["artifacts"]["embedding_cache"])
    for name, samples in (("train", train), ("val", val), ("test", test)):
        extract_embeddings(
            model, samples, device, cfg["window"]["feature_batch_size"],
            cache / f"{name}.pt",
        )
        print(f"[cache] {name}: {cache / f'{name}.pt'}")


if __name__ == "__main__":
    main()
