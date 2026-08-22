"""Shared pretrained-backbone construction, used by every model that consumes
per-frame CNN features (Baseline 1's frame-pool, Baseline 2's CNN+RNN, and any
future model built the same way) so the supported-backbone list and loading
logic live in exactly one place.
"""

from __future__ import annotations

from torch import nn
from torchvision import models

BACKBONE_FACTORIES: dict[str, tuple] = {
    "resnet18": (models.resnet18, "fc", 512),
    "resnet34": (models.resnet34, "fc", 512),
    "efficientnet_b0": (models.efficientnet_b0, "classifier", 1280),
}


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Returns (backbone_with_head_stripped, feature_dim)."""
    if name not in BACKBONE_FACTORIES:
        raise ValueError(
            f"Unknown backbone {name!r}; expected one of {sorted(BACKBONE_FACTORIES)}"
        )
    factory, head_attr, feature_dim = BACKBONE_FACTORIES[name]
    backbone = factory(weights="DEFAULT" if pretrained else None)
    setattr(backbone, head_attr, nn.Identity())
    return backbone, feature_dim
