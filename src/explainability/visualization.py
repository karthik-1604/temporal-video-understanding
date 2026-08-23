"""Heatmap-to-RGB overlay for Grad-CAM visualization. Pure numpy/PIL, no
dependency on the real model or real video frames, so it's fully testable
locally with synthetic arrays.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """heatmap: (H, W) float32 in [0, 1] -> (H, W, 3) uint8 RGB.
    Simple blue -> green -> yellow -> red gradient (jet-like), no matplotlib
    dependency needed.
    """
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * h - 3), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4 * h - 2), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4 * h - 1), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap_on_frame(
    frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """frame: (H, W, 3) uint8. heatmap: (h, w) float32 in [0, 1], any spatial
    size (resized to match frame via PIL). Returns (H, W, 3) uint8 alpha-
    blended overlay: alpha=0 returns the original frame, alpha=1 returns pure
    colormap.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    height, width = frame.shape[:2]
    heatmap_img = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8))
    heatmap_resized = np.array(
        heatmap_img.resize((width, height), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    color = _apply_colormap(heatmap_resized).astype(np.float32)
    blended = (1 - alpha) * frame.astype(np.float32) + alpha * color
    return np.clip(blended, 0, 255).astype(np.uint8)
