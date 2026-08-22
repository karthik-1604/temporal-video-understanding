"""Frame-index sampling strategies, shared by every model and by the
temporal-resolution experiment. Operates purely on frame counts/indices —
no video decoding here, so it's fully unit-testable without real data.
"""

from __future__ import annotations

import numpy as np


def uniform_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """Evenly spaced frame indices spanning [0, total_frames - 1].

    If total_frames < num_frames, the last index is repeated to pad up to
    num_frames (short clips are more common than long ones in action-recognition
    datasets, so padding-by-repeat is preferred over erroring).
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num=num_frames)
        return [int(round(i)) for i in indices]

    indices = list(range(total_frames))
    indices += [total_frames - 1] * (num_frames - total_frames)
    return indices


def random_frame_indices(
    total_frames: int, num_frames: int, seed: int | None = None
) -> list[int]:
    """Sorted random frame indices, for use as a training-time augmentation.

    Reproducible given the same seed. Same short-clip padding behavior as
    uniform_frame_indices.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    rng = np.random.default_rng(seed)

    if total_frames >= num_frames:
        indices = rng.choice(total_frames, size=num_frames, replace=False)
        return sorted(int(i) for i in indices)

    indices = list(range(total_frames))
    indices += [total_frames - 1] * (num_frames - total_frames)
    return indices


_STRATEGIES = {
    "uniform": uniform_frame_indices,
    "random": random_frame_indices,
}


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    strategy: str = "uniform",
    seed: int | None = None,
) -> list[int]:
    """Dispatch to a named sampling strategy. Single entry point used by the
    dataset loader so the sampling strategy stays config-driven.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown sampling strategy {strategy!r}; expected one of {sorted(_STRATEGIES)}"
        )
    if strategy == "random":
        return random_frame_indices(total_frames, num_frames, seed=seed)
    return uniform_frame_indices(total_frames, num_frames)
