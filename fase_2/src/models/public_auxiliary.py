"""Encoder auxiliar pequeno para pré-treino público de olhos e boca."""

from __future__ import annotations

import torch
from torch import nn


class AuxiliaryVisualEncoder(nn.Module):
    """Produz logits binários; probabilidades entram como atributos, não como target final."""

    def __init__(self, *, task: str) -> None:
        super().__init__()
        if task not in {"eye_state", "yawn"}:
            raise ValueError("task deve ser eye_state ou yawn")
        self.task = task
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Entrada deve possuir shape N×3×H×W")
        return self.classifier(self.features(images).flatten(1))

    def positive_probability(self, images: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(images), dim=1)[:, 1]
