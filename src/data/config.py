"""Dataset configuration schema. Keeps the dataset (UCF101 now, potentially
something larger later) and the frame-sampling strategy fully config-driven
rather than hardcoded, per the project's dataset-configurability requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_STRATEGIES = {"uniform", "random"}


@dataclass(frozen=True)
class VideoDatasetConfig:
    dataset_name: str
    root_dir: str
    split: str
    num_frames: int
    sampling_strategy: str = "uniform"
    image_size: int = 224
    seed: int = 42

    def __post_init__(self) -> None:
        if self.sampling_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown sampling_strategy {self.sampling_strategy!r}; "
                f"expected one of {sorted(_VALID_STRATEGIES)}"
            )
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VideoDatasetConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
