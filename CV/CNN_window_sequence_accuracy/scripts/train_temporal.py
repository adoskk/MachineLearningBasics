import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embedding_cache import WindowEmbeddingDataset, load_embedding_cache
from src.training.train_temporal import train_temporal_head
from src.utils.common import get_device, load_config, log_version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log_version("train-temporal")
    device = get_device(cfg["runtime"]["device"])
    spec = json.loads(Path(cfg["artifacts"]["window_spec"]).read_text())
    window, stride = spec["window_size"], int(cfg["window"]["stride"])
    cache = Path(cfg["artifacts"]["embedding_cache"])
    train_entries = load_embedding_cache(cache / "train.pt")
    val_entries = load_embedding_cache(cache / "val.pt")
    train_ds = WindowEmbeddingDataset(train_entries, window, stride)
    val_ds = WindowEmbeddingDataset(val_entries, window, stride)
    feature_dim = int(train_entries[0]["embeddings"].shape[1])
    models = args.models or cfg["window"]["models"]
    out = Path(cfg["artifacts"]["temporal_checkpoints"])
    for name in models:
        train_temporal_head(
            name, train_ds, val_ds, feature_dim, window, cfg, device,
            out / f"{name}.pt",
        )


if __name__ == "__main__":
    main()
