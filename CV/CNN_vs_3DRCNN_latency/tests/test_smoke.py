import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.classifier.benchmark import benchmark_classifier
from src.classifier.evaluate import evaluate_classifier
from src.classifier.train import build_classifier
from src.data.synthetic import make_synthetic_cohort
from src.detector.benchmark import benchmark_detector
from src.detector.evaluate import evaluate_detector
from src.detector.train import build_detector
from src.eval.compare import count_params


def test_smoke_cpu():
    device = torch.device("cpu")
    cfg = {"detection": {"anchor_sizes": [8, 16]}, "classification": {}}
    samples = make_synthetic_cohort(n=4, shape=(32, 64, 64), seed=0)
    assert all(len(s.boxes) > 0 for s in samples), "synthetic vols should have lesions"

    # detector track: build -> train-mode loss -> evaluate -> benchmark
    det = build_detector(cfg, device, pretrained=False)
    det.train()
    vols = torch.stack([torch.from_numpy(s.volume).unsqueeze(0) for s in samples[:2]])
    tgts = [{"boxes": torch.from_numpy(s.boxes).float(), "labels": torch.ones(len(s.boxes), dtype=torch.long)}
            for s in samples[:2]]
    losses = det(vols, tgts)
    assert set(losses) == {"loss_rpn_cls", "loss_rpn_reg", "loss_det_cls", "loss_det_reg"}
    dq = evaluate_detector(det, samples[:2], device)
    assert "ap" in dq
    db = benchmark_detector(det, samples[:2], device, warmup=1, repeats=2)
    assert "mean_ms_per_volume" in db

    # classifier track: build -> batched predict -> evaluate -> benchmark
    cls = build_classifier(cfg, device, pretrained=False)
    p = cls.predict_volume(torch.from_numpy(samples[0].volume), batch_size=8, device=device)
    assert p.shape[0] == samples[0].volume.shape[0]
    cq = evaluate_classifier(cls, samples[:2], device, batch_size=8)
    assert "auroc" in cq
    cb = benchmark_classifier(cls, samples[:2], device, batch_sizes=(8,), warmup=1, repeats=2)
    assert "batch_8" in cb

    print(f"[smoke] det_params={count_params(det):,} cls_params={count_params(cls):,} OK")


if __name__ == "__main__":
    test_smoke_cpu()
