"""torch Dataset over UCF101 clips. Frame decoding is injected as a
`clip_reader` callable rather than hardcoded to a specific decoding library,
so this class's indexing/labeling/transform logic is fully unit-testable
locally with a fake reader — real decoding (src/data/video_io.py) only runs
where actual video files exist, i.e. inside a Kaggle kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.ucf101_metadata import VideoSample, resolve_video_path

ClipReader = Callable[[Path, int, str, Optional[int]], np.ndarray]


def default_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (T, C, H, W) float32 in [0, 1]."""
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return clip.float() / 255.0


class VideoClipDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[VideoSample],
        root_dir: str | Path,
        num_frames: int,
        clip_reader: ClipReader,
        sampling_strategy: str = "uniform",
        seed: Optional[int] = None,
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ) -> None:
        self.samples = list(samples)
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.clip_reader = clip_reader
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        self.transform = transform or default_to_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        video_path = resolve_video_path(self.root_dir, sample)

        frames = self.clip_reader(video_path, self.num_frames, self.sampling_strategy, self.seed)
        clip = self.transform(frames)

        return clip, sample.label
