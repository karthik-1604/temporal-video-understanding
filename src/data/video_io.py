"""Real video-clip decoding. Only exercised inside a Kaggle kernel, where
actual video files and `decord` exist — `decord` is imported lazily so this
module can be imported locally (e.g. for type references) without it
installed. Local tests use a fake clip reader instead; see
tests/test_video_dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.sampling import sample_frame_indices


def decord_clip_reader(
    video_path: str | Path,
    num_frames: int,
    strategy: str = "uniform",
    seed: int | None = None,
) -> np.ndarray:
    """Read num_frames from video_path using the given sampling strategy.
    Returns an array of shape (num_frames, height, width, channels), uint8.
    """
    import decord  # local import: only required where real videos are decoded

    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    indices = sample_frame_indices(total_frames, num_frames, strategy=strategy, seed=seed)
    return vr.get_batch(indices).asnumpy()
