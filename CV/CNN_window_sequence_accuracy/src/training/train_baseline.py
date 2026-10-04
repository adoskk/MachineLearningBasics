"""End-to-end training for the current per-slice CNN baseline."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import SliceDataset
from ..models.slice_encoder import SliceBaseline


@torch.no_grad()
def _loss(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for images, labels, _, _ in loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        total += float(loss) * len(images)
        n += len(images)
    return total / max(n, 1)


def train_baseline(train_samples, val_samples, cfg, device,
                   output_path="checkpoints/slice_baseline.pt"):
    bc = cfg["baseline"]
    model = SliceBaseline(bc["pretrained"], bc["backbone"]).to(device)
    train_ds, val_ds = SliceDataset(train_samples), SliceDataset(val_samples)
    positives = sum(int(item[1]) for item in train_ds.items)
    negatives = len(train_ds) - positives
    pos_weight = torch.tensor([negatives / max(positives, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(bc["lr"]), weight_decay=float(bc["weight_decay"])
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(bc["batch_size"]), shuffle=True,
        num_workers=int(cfg["runtime"]["num_workers"])
    )
    val_loader = DataLoader(val_ds, batch_size=int(bc["batch_size"]), shuffle=False)

    best, best_loss, stale = None, float("inf"), 0
    history = []
    for epoch in range(int(bc["epochs"])):
        model.train()
        total, n = 0.0, 0
        for images, labels, _, _ in tqdm(train_loader, desc=f"baseline ep{epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
            n += len(images)
        train_loss = total / max(n, 1)
        val_loss = _loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[baseline] epoch={epoch+1} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best, stale = val_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(bc["patience"]):
                break
    model.load_state_dict(best)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "feature_dim": model.feature_dim,
        "history": history, "best_val_loss": best_loss,
    }, p)
    return model
