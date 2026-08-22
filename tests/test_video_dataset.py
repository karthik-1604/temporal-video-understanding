import numpy as np
import torch

from src.data.ucf101_metadata import VideoSample
from src.data.video_dataset import VideoClipDataset, default_to_tensor

SAMPLES = [
    VideoSample(
        clip_name="v_Swing_g05_c02",
        relative_path="train/Swing/v_Swing_g05_c02.avi",
        class_name="Swing",
        label=2,
    ),
    VideoSample(
        clip_name="v_Archery_g02_c03",
        relative_path="train/Archery/v_Archery_g02_c03.avi",
        class_name="Archery",
        label=1,
    ),
]


def fake_clip_reader(video_path, num_frames, strategy, seed):
    """Deterministic stand-in for real decoding: returns fixed-shape frames
    without touching any real video file, so this exercises the Dataset's
    indexing/transform logic without decord/opencv or real data.
    """
    return np.zeros((num_frames, 8, 8, 3), dtype=np.uint8)


def test_len_matches_sample_count():
    dataset = VideoClipDataset(SAMPLES, "/fake/root", num_frames=4, clip_reader=fake_clip_reader)
    assert len(dataset) == 2


def test_getitem_returns_tensor_and_correct_label():
    dataset = VideoClipDataset(SAMPLES, "/fake/root", num_frames=4, clip_reader=fake_clip_reader)

    clip, label = dataset[0]

    assert isinstance(clip, torch.Tensor)
    assert clip.shape == (4, 3, 8, 8)  # (T, C, H, W)
    assert label == 2


def test_getitem_respects_sample_order():
    dataset = VideoClipDataset(SAMPLES, "/fake/root", num_frames=4, clip_reader=fake_clip_reader)

    _, label0 = dataset[0]
    _, label1 = dataset[1]

    assert label0 == 2
    assert label1 == 1


def test_clip_reader_receives_resolved_path_and_config():
    received = {}

    def recording_reader(video_path, num_frames, strategy, seed):
        received["video_path"] = video_path
        received["num_frames"] = num_frames
        received["strategy"] = strategy
        received["seed"] = seed
        return np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)

    dataset = VideoClipDataset(
        SAMPLES,
        "/fake/root",
        num_frames=8,
        clip_reader=recording_reader,
        sampling_strategy="random",
        seed=7,
    )
    dataset[0]

    assert str(received["video_path"]).replace("\\", "/") == "/fake/root/train/Swing/v_Swing_g05_c02.avi"
    assert received["num_frames"] == 8
    assert received["strategy"] == "random"
    assert received["seed"] == 7


def test_custom_transform_is_applied():
    def sum_transform(frames):
        return torch.tensor(int(frames.sum()))

    dataset = VideoClipDataset(
        SAMPLES, "/fake/root", num_frames=4, clip_reader=fake_clip_reader, transform=sum_transform
    )

    clip, _ = dataset[0]

    assert clip.item() == 0  # fake_clip_reader returns all-zero frames


def test_default_to_tensor_normalizes_to_unit_range():
    frames = np.full((3, 4, 4, 3), 255, dtype=np.uint8)

    clip = default_to_tensor(frames)

    assert clip.shape == (3, 3, 4, 4)  # permuted to (T, C, H, W)
    assert torch.allclose(clip, torch.ones_like(clip))
