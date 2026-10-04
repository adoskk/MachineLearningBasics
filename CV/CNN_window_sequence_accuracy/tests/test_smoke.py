import tempfile
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import split_samples
from src.data.embedding_cache import WindowEmbeddingDataset
from src.data.synthetic import make_synthetic_cohort
from src.data.window_utils import build_window_records, compute_window_spec
from src.models.slice_encoder import SliceBaseline
from src.models.temporal_heads import MODEL_NAMES, build_temporal_model


def test_all_models_and_windows():
    samples = make_synthetic_cohort(12, shape=(32, 64, 64), seed=1)
    train, val, test = split_samples(samples, (0.7, 0.15, 0.15), seed=9)
    assert not ({s.volume_id for s in train} & {s.volume_id for s in test})

    with tempfile.TemporaryDirectory() as tmp:
        spec = compute_window_spec(train, 95, 10, Path(tmp) / "q95.json")
        w = spec["window_size"]
        records = build_window_records(train, w)
        assert all(r.label == int(train[r.volume_index].slice_labels[r.start:r.end].max())
                   for r in records)

        baseline = SliceBaseline(pretrained=False, backbone="tiny")
        x = torch.randn(4, 1, 64, 64)
        features = baseline.encode_slices(x)
        assert baseline(x).shape == (4,)
        assert features.shape[1] == baseline.feature_dim

        entries = []
        for sample in train:
            d = len(sample.slice_labels)
            entries.append({
                "volume_id": sample.volume_id,
                "subset": "synthetic",
                "embeddings": torch.randn(d, baseline.feature_dim),
                "slice_logits": torch.randn(d),
                "slice_labels": torch.from_numpy(sample.slice_labels),
            })
        ds = WindowEmbeddingDataset(entries, w)
        batch_x = torch.stack([ds[i]["embeddings"] for i in range(min(4, len(ds)))])
        batch_y = torch.stack([ds[i]["label"] for i in range(min(4, len(ds)))])

        for name in MODEL_NAMES:
            model = build_temporal_model(name, w, baseline.feature_dim, hidden_size=16)
            logits = model(batch_x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            assert logits.shape == batch_y.shape
            path = Path(tmp) / f"{name}.pt"
            torch.save(model.state_dict(), path)
            clone = build_temporal_model(name, w, baseline.feature_dim, hidden_size=16)
            clone.load_state_dict(torch.load(path, weights_only=True))


if __name__ == "__main__":
    test_all_models_and_windows()
    print("[smoke] window data + baseline + all temporal heads OK")
