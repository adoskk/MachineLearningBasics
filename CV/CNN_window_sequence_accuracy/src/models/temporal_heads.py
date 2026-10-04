"""Linear and LSTM aggregators over a fixed window of frozen slice embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn


class WindowLinear(nn.Module):
    """Literal linear baseline: flatten W x F embeddings into one logit."""

    def __init__(self, window_size: int, feature_dim: int):
        super().__init__()
        self.head = nn.Linear(window_size * feature_dim, 1)

    def forward(self, x):
        return self.head(x.flatten(1)).squeeze(-1)


class WindowLSTM(nn.Module):
    def __init__(self, feature_dim: int, hidden_size=128, num_layers=1,
                 bidirectional=False, dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            feature_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            representation = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            representation = hidden[-1]
        return self.head(representation).squeeze(-1)


MODEL_NAMES = [
    "window_linear",
    "lstm_uni_1",
    "lstm_uni_2",
    "lstm_bi_1",
    "lstm_bi_2",
]


def build_temporal_model(name, window_size, feature_dim, hidden_size=128):
    if name == "window_linear":
        return WindowLinear(window_size, feature_dim)
    specs = {
        "lstm_uni_1": (1, False),
        "lstm_uni_2": (2, False),
        "lstm_bi_1": (1, True),
        "lstm_bi_2": (2, True),
    }
    if name not in specs:
        raise KeyError(f"Unknown temporal model {name}; choose from {MODEL_NAMES}")
    layers, bidirectional = specs[name]
    return WindowLSTM(
        feature_dim, hidden_size, num_layers=layers,
        bidirectional=bidirectional, dropout=0.2,
    )


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
