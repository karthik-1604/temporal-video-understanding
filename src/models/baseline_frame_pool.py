"""Baseline 1: pretrained CNN frame features + temporal average pooling + MLP
classifier. The simplest baseline in the plan — no temporal modeling beyond
averaging — used as the reference point every other model (CNN+RNN,
transformer, CLIP zero-shot) is compared against.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.backbones import build_backbone


class FramePoolClassifier(nn.Module):
    """Pretrained backbone -> per-frame embeddings -> temporal average pool -> MLP."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone, pretrained)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (batch, num_frames, channels, height, width) -> (batch, num_classes)."""
        if clips.ndim != 5:
            raise ValueError(
                f"Expected 5D input (batch, num_frames, channels, height, width), "
                f"got shape {tuple(clips.shape)}"
            )
        batch, num_frames, channels, height, width = clips.shape
        frames = clips.reshape(batch * num_frames, channels, height, width)
        features = self.backbone(frames)
        features = features.reshape(batch, num_frames, -1)
        pooled = features.mean(dim=1)
        return self.classifier(pooled)
