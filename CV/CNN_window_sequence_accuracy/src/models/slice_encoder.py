"""Current per-slice CNN baseline refactored into encoder + linear head."""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.models import ResNet18_Weights, resnet18
    _HAS_TV = True
except Exception:
    _HAS_TV = False


class TinySliceEncoder(nn.Module):
    feature_dim = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18SliceEncoder(nn.Module):
    feature_dim = 512

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        old = model.conv1
        model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if weights is not None:
            model.conv1.weight.data.copy_(old.weight.data.mean(1, keepdim=True))
        self.features = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    def forward(self, x):
        return self.features(x)


class SliceBaseline(nn.Module):
    """ResNet18-compatible baseline: one binary logit per axial slice."""

    def __init__(self, pretrained=True, backbone="resnet18"):
        super().__init__()
        if _HAS_TV and backbone == "resnet18":
            try:
                self.encoder = ResNet18SliceEncoder(pretrained)
            except Exception as e:
                print(f"[model] ResNet18 unavailable ({e}); using TinySliceEncoder")
                self.encoder = TinySliceEncoder()
        else:
            self.encoder = TinySliceEncoder()
        self.feature_dim = self.encoder.feature_dim
        self.classifier = nn.Linear(self.feature_dim, 1)

    def encode_slices(self, x):
        return self.encoder(x)

    def classify_features(self, features):
        return self.classifier(features).squeeze(-1)

    def forward(self, x):
        return self.classify_features(self.encode_slices(x))
