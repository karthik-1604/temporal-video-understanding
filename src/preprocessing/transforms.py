"""Clip preprocessing: resize to the backbone's expected input size and apply
ImageNet normalization (required since Baseline 1's backbones are ImageNet-
pretrained). Pure tensor ops — no real images needed to test, since
interpolation works on any input resolution including tiny random tensors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_and_normalize(frames: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (T, C, image_size, image_size) float32, ImageNet-normalized."""
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    clip = F.interpolate(
        clip, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (clip - mean) / std
