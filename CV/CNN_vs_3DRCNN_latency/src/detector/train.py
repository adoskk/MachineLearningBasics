"""Detector track: 3D Faster R-CNN training (AdamW, few epochs, Colab-friendly)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import DetectionDataset, det_collate
from ..models.faster_rcnn_3d import FasterRCNN3D


def build_detector(cfg: dict, device: torch.device, pretrained: bool | None = None) -> FasterRCNN3D:
    """Construct the detector. pretrained=None -> follow config flag."""
    d = cfg.get("detection", {})
    if pretrained is None:
        pretrained = bool(d.get("pretrained_backbone", True))
    return FasterRCNN3D(
        anchor_sizes=tuple(d.get("anchor_sizes", [16, 32, 64])),
        rpn_pre_nms=int(d.get("rpn_pre_nms", 1000)),
        rpn_post_nms=int(d.get("rpn_post_nms", 256)),
        roi_size=tuple(d.get("roi_size", [7, 7, 7])),
        box_score_thresh=float(d.get("box_score_thresh", 0.3)),
        nms_thresh=float(d.get("nms_thresh", 0.4)),
        pretrained_backbone=pretrained,
    ).to(device)


def train_detector(train_samples, val_samples, cfg: dict, device: torch.device,
                   out_dir: str = "checkpoints") -> FasterRCNN3D:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    d = cfg.get("detection", {})
    model = build_detector(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(d.get("lr", 5e-4)))
    loader = DataLoader(DetectionDataset(train_samples), batch_size=1,
                        shuffle=True, collate_fn=det_collate)
    model.train()
    for ep in range(int(d.get("epochs", 5))):
        tot = 0.0
        for vols, tgts in tqdm(loader, desc=f"det ep{ep+1}"):
            vols = vols.to(device)
            tgts = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in tgts]
            losses = model(vols, tgts)
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        print(f"[det] epoch {ep+1} loss={tot/max(len(loader),1):.4f}")
    torch.save(model.state_dict(), f"{out_dir}/faster_rcnn_3d.pt")
    return model
