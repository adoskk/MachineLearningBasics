"""Per-slice classifier with BATCHED inference (the comparison's 2D side).

Model: torchvision resnet18 with 1-channel stem -> 1 logit (lesion present?).
Falls back to a tiny 4-block CNN when torchvision is missing.

The point of this model in the benchmark: a whole volume is scored as a
BATCH of D slices (batch_size=32 default). That exposes the core tradeoff:
  - 3D Faster R-CNN: 1 heavy 3D pass, big activation memory, rich 3D context
  - slice model: D cheap 2D passes, streamable / batchable, no 3D context
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.models import resnet18, ResNet18_Weights
    _HAS_TV = True
except Exception:
    _HAS_TV = False


class TinyCNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class SliceClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone
        if _HAS_TV and backbone == "resnet18":
            try:
                w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                m = resnet18(weights=w)
                m.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
                m.fc = nn.Linear(m.fc.in_features, 1)
                self.net = m
                self._kind = "resnet18"
            except Exception:
                self.net = TinyCNN2D()
                self._kind = "tiny"
        else:
            self.net = TinyCNN2D()
            self._kind = "tiny"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) -> (B,) logits."""
        out = self.net(x)
        return out.view(-1)

    @torch.no_grad()
    def predict_volume(self, volume: torch.Tensor, batch_size: int = 32, device=None) -> torch.Tensor:
        """Score every axial slice. volume: (D,H,W) or (1,D,H,W) -> (D,) probs."""
        dev = device or next(self.parameters()).device
        self.eval()
        if volume.dim() == 4:
            volume = volume[0]
        D = volume.shape[0]
        probs = []
        for i in range(0, D, batch_size):
            sl = volume[i:i + batch_size].unsqueeze(1).to(dev)  # (b,1,H,W)
            sl = (sl - 0.35) / 0.2
            probs.append(self(sl).sigmoid().cpu())
        return torch.cat(probs)
