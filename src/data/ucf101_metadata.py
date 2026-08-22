"""Metadata parsing for the UCF101 dataset layout actually used on Kaggle
(dataset ref: matthewjansen/ucf101-action-recognition) — confirmed by running
a CPU-only exploration kernel (kaggle_kernel/ucf101_explore) rather than
assumed from the classic UCF101 file format (classInd.txt / trainlist01.txt),
which this dataset does not use.

Real layout: <root>/{train,test,val}.csv with columns
`clip_name, clip_path, label`, where `label` is a class-name string (not a
numeric index) and `clip_path` is root-relative, e.g. "/train/Swing/v_Swing_g05_c02.avi".
Videos live under <root>/{train,test,val}/<ClassName>/<clip_name>.avi.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VideoSample:
    clip_name: str
    relative_path: str
    class_name: str
    label: int


def build_class_index(class_names: Iterable[str]) -> dict[str, int]:
    """Deterministic class_name -> integer label mapping, sorted alphabetically
    so it's stable regardless of row order in the source CSV.
    """
    unique_names = sorted(set(class_names))
    return {name: idx for idx, name in enumerate(unique_names)}


def load_split_metadata(
    csv_path: str | Path, class_to_index: dict[str, int]
) -> list[VideoSample]:
    """Parse one split CSV against a shared class_to_index (built once, reused
    across train/test/val) so label ids stay consistent across splits.
    """
    samples = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["label"]
            if class_name not in class_to_index:
                raise ValueError(
                    f"Class {class_name!r} in {csv_path} not present in the "
                    "shared class index — splits disagree on the class set"
                )
            samples.append(
                VideoSample(
                    clip_name=row["clip_name"],
                    relative_path=row["clip_path"].lstrip("/"),
                    class_name=class_name,
                    label=class_to_index[class_name],
                )
            )
    return samples


def load_ucf101_splits(
    root_dir: str | Path,
    splits: tuple[str, ...] = ("train", "test", "val"),
    class_index_source: str = "train",
) -> dict[str, list[VideoSample]]:
    """Load all available split CSVs under root_dir, building the class index
    from `class_index_source` (falling back to the first split found if that
    one isn't present) so every split shares the same label ids.
    """
    root_dir = Path(root_dir)
    available = {split: root_dir / f"{split}.csv" for split in splits}
    available = {split: path for split, path in available.items() if path.exists()}
    if not available:
        raise FileNotFoundError(f"No split CSVs found under {root_dir}")

    index_split = class_index_source if class_index_source in available else next(iter(available))
    with open(available[index_split], "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        class_to_index = build_class_index(row["label"] for row in reader)

    return {
        split: load_split_metadata(path, class_to_index)
        for split, path in available.items()
    }


def resolve_video_path(root_dir: str | Path, sample: VideoSample) -> Path:
    return Path(root_dir) / sample.relative_path
