"""Training loop shared by linear and LSTM window heads."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.embedding_cache import balanced_sampler
from ..models.temporal_heads import build_temporal_model


@torch.no_grad()
def predict_head(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    logits, labels, volume_ids, starts, subsets = [], [], [], [], []
    for batch in loader:
        logits.append(model(batch["embeddings"].to(device)).cpu())
        labels.append(batch["label"])
        volume_ids.extend(batch["volume_id"])
        starts.extend(batch["start"].tolist())
        subsets.extend(batch["subset"])
    return {
        "probabilities": torch.cat(logits).sigmoid().numpy(),
        "labels": torch.cat(labels).numpy().astype(int),
        "volume_ids": volume_ids,
        "starts": starts,
        "subsets": subsets,
    }


def train_temporal_head(name, train_ds, val_ds, feature_dim, window_size,
                        cfg, device, output_path):
    wc = cfg["window"]
    model = build_temporal_model(
        name, window_size, feature_dim, int(wc["hidden_size"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(wc["lr"]), weight_decay=float(wc["weight_decay"])
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        train_ds, batch_size=int(wc["train_batch_size"]),
        sampler=balanced_sampler(train_ds),
    )
    best, best_ap, stale, history = None, -1.0, 0, []
    for epoch in range(int(wc["epochs"])):
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(loader, desc=f"{name} ep{epoch+1}"):
            x, y = batch["embeddings"].to(device), batch["label"].to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        pred = predict_head(model, val_ds, device)
        val_ap = float(average_precision_score(pred["labels"], pred["probabilities"]))
        history.append({"epoch": epoch + 1, "train_loss": total / max(n, 1),
                        "val_ap": val_ap})
        print(f"[{name}] epoch={epoch+1} loss={history[-1]['train_loss']:.4f} val_ap={val_ap:.4f}")
        if val_ap > best_ap:
            best_ap, best, stale = val_ap, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(wc["patience"]):
                break
    model.load_state_dict(best)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "name": name, "feature_dim": feature_dim,
        "window_size": window_size, "hidden_size": int(wc["hidden_size"]),
        "history": history, "best_val_ap": best_ap,
    }, p)
    return model
