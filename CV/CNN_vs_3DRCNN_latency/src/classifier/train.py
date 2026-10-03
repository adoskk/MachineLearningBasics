"""Classifier track: per-slice 2D training (AdamW, batched slices)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import SliceDataset
from ..models.slice_classifier import SliceClassifier


def build_classifier(cfg: dict, device: torch.device, pretrained: bool | None = None) -> SliceClassifier:
    """Construct the slice classifier. pretrained=None -> follow config flag."""
    c = cfg.get("classification", {})
    if pretrained is None:
        pretrained = bool(c.get("pretrained", True))
    return SliceClassifier(
        pretrained=pretrained,
        backbone=c.get("backbone", "resnet18"),
    ).to(device)


def train_classifier(train_samples, val_samples, cfg: dict, device: torch.device,
                     out_dir: str = "checkpoints") -> SliceClassifier:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    c = cfg.get("classification", {})
    model = build_classifier(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.get("lr", 1e-3)))
    crit = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(SliceDataset(train_samples), batch_size=int(c.get("batch_size", 32)),
                        shuffle=True, num_workers=0)
    model.train()
    for ep in range(int(c.get("epochs", 5))):
        tot, n = 0.0, 0
        for imgs, labs, _ in tqdm(loader, desc=f"cls ep{ep+1}"):
            imgs, labs = imgs.to(device), labs.to(device)
            logits = model(imgs)
            loss = crit(logits, labs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += len(imgs)
        print(f"[cls] epoch {ep+1} loss={tot/max(n,1):.4f}")
    torch.save(model.state_dict(), f"{out_dir}/slice_classifier.pt")
    return model
