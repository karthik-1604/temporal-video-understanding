import numpy as np
import pytest

from src.explainability.visualization import overlay_heatmap_on_frame


def test_output_shape_matches_frame():
    frame = np.random.randint(0, 256, size=(32, 40, 3), dtype=np.uint8)
    heatmap = np.random.rand(7, 7).astype(np.float32)

    overlay = overlay_heatmap_on_frame(frame, heatmap)

    assert overlay.shape == (32, 40, 3)
    assert overlay.dtype == np.uint8


def test_alpha_zero_returns_original_frame():
    frame = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    heatmap = np.random.rand(4, 4).astype(np.float32)

    overlay = overlay_heatmap_on_frame(frame, heatmap, alpha=0.0)

    assert np.array_equal(overlay, frame)


def test_alpha_one_ignores_original_frame_content():
    frame_a = np.zeros((16, 16, 3), dtype=np.uint8)
    frame_b = np.full((16, 16, 3), 255, dtype=np.uint8)
    heatmap = np.full((4, 4), 0.5, dtype=np.float32)

    overlay_a = overlay_heatmap_on_frame(frame_a, heatmap, alpha=1.0)
    overlay_b = overlay_heatmap_on_frame(frame_b, heatmap, alpha=1.0)

    assert np.array_equal(overlay_a, overlay_b)


def test_resizes_heatmap_to_frame_size():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    heatmap = np.ones((7, 7), dtype=np.float32)  # much smaller than frame

    overlay = overlay_heatmap_on_frame(frame, heatmap, alpha=0.5)

    assert overlay.shape == (64, 64, 3)


def test_rejects_invalid_alpha():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    heatmap = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        overlay_heatmap_on_frame(frame, heatmap, alpha=1.5)
    with pytest.raises(ValueError):
        overlay_heatmap_on_frame(frame, heatmap, alpha=-0.1)


def test_high_heatmap_region_shifts_toward_red():
    frame = np.full((8, 8, 3), 128, dtype=np.uint8)
    heatmap = np.ones((8, 8), dtype=np.float32)  # max activation everywhere

    overlay = overlay_heatmap_on_frame(frame, heatmap, alpha=0.8)

    # heatmap=1.0 maps to red-dominant color in the jet-like gradient
    assert overlay[..., 0].mean() > overlay[..., 2].mean()
