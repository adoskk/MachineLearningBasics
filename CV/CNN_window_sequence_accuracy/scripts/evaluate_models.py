import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.embedding_cache import WindowEmbeddingDataset, load_embedding_cache
from src.models.temporal_heads import build_temporal_model
from src.training.evaluate import evaluate_models
from src.utils.common import get_device, load_config, log_version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log_version("evaluate")
    device = get_device(cfg["runtime"]["device"])
    spec = json.loads(Path(cfg["artifacts"]["window_spec"]).read_text())
    window, stride = spec["window_size"], int(cfg["window"]["stride"])
    cache = Path(cfg["artifacts"]["embedding_cache"])
    val_entries = load_embedding_cache(cache / "val.pt")
    test_entries = load_embedding_cache(cache / "test.pt")
    val_ds = WindowEmbeddingDataset(val_entries, window, stride)
    test_ds = WindowEmbeddingDataset(test_entries, window, stride)
    feature_dim = int(val_entries[0]["embeddings"].shape[1])
    models = {}
    ckpt_dir = Path(cfg["artifacts"]["temporal_checkpoints"])
    for name in cfg["window"]["models"]:
        payload = torch.load(ckpt_dir / f"{name}.pt", map_location=device, weights_only=False)
        model = build_temporal_model(
            name, window, feature_dim, int(cfg["window"]["hidden_size"])
        ).to(device)
        model.load_state_dict(payload["model"])
        models[name] = model
    evaluate_models(models, val_ds, test_ds, cfg, device, args.out)


if __name__ == "__main__":
    main()
