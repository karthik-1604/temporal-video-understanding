import numpy as np
import torch

from src.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD, resize_and_normalize


def test_resizes_to_requested_size():
    frames = np.zeros((4, 60, 80, 3), dtype=np.uint8)

    clip = resize_and_normalize(frames, image_size=224)

    assert clip.shape == (4, 3, 224, 224)


def test_handles_non_square_and_odd_input_sizes():
    frames = np.zeros((2, 17, 33, 3), dtype=np.uint8)

    clip = resize_and_normalize(frames, image_size=112)

    assert clip.shape == (2, 3, 112, 112)


def test_black_frame_normalizes_to_negative_mean_over_std():
    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)

    clip = resize_and_normalize(frames, image_size=8)

    expected = torch.tensor([-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
    for c in range(3):
        assert torch.allclose(clip[0, c], expected[c].expand(8, 8), atol=1e-5)


def test_white_frame_normalizes_to_one_minus_mean_over_std():
    frames = np.full((1, 8, 8, 3), 255, dtype=np.uint8)

    clip = resize_and_normalize(frames, image_size=8)

    expected = torch.tensor([(1.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
    for c in range(3):
        assert torch.allclose(clip[0, c], expected[c].expand(8, 8), atol=1e-5)


def test_output_is_float32():
    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)

    clip = resize_and_normalize(frames)

    assert clip.dtype == torch.float32
