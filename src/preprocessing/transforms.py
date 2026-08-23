"""Clip preprocessing: resize to the backbone's expected input size and apply
ImageNet normalization (required since Baseline 1's backbones are ImageNet-
pretrained). Pure tensor ops — no real images needed to test, since
interpolation works on any input resolution including tiny random tensors.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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


def frames_to_pil_transformed(
    frames: np.ndarray, pil_transform: Callable[[Image.Image], torch.Tensor]
) -> torch.Tensor:
    """Apply a PIL-Image-based per-frame transform (e.g. open_clip's own
    `preprocess`, which expects a PIL Image and applies CLIP-specific
    resize/crop/normalization) to each frame, then stack into a clip tensor.
    Generic over the transform, so this is testable locally with any
    torchvision-style callable — no real CLIP model needed.
    """
    tensors = [pil_transform(Image.fromarray(frame)) for frame in frames]
    return torch.stack(tensors)
