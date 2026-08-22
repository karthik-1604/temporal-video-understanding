# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""CPU-only exploration script: inspect the real UCF101 dataset layout on
Kaggle before writing the metadata parser locally. Prints structure only —
does not train or download anything back to the local machine.
"""

import os
from pathlib import Path

import pandas as pd


def find_dataset_root(input_root: str = "/kaggle/input", max_depth: int = 3) -> Path:
    """Walk /kaggle/input (depth-limited) to find the dataset root, since the
    documented mount path isn't always the actual one (see KAGGLE_WORKFLOW_NOTES.md).
    """
    root = Path(input_root)
    candidates = []
    for path in root.rglob("*"):
        if path.is_file() and path.name in ("train.csv", "test.csv"):
            candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError(f"No train.csv/test.csv found under {input_root}")
    return sorted(set(candidates))[0]


# %%
dataset_root = find_dataset_root()
print(f"dataset_root = {dataset_root}")
print("top-level contents:", sorted(p.name for p in dataset_root.iterdir()))

# %%
for csv_name in ("train.csv", "test.csv"):
    csv_path = dataset_root / csv_name
    if not csv_path.exists():
        print(f"{csv_name}: NOT FOUND")
        continue
    df = pd.read_csv(csv_path)
    print(f"\n=== {csv_name} ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:\n", df.dtypes)
    print("head:\n", df.head(5).to_string())
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() < 200:
            print(f"unique values in {col}: {df[col].nunique()}")

# %%
for split in ("train", "test"):
    split_dir = dataset_root / split
    if not split_dir.exists():
        print(f"\n{split}/ directory not found")
        continue
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    print(f"\n=== {split}/ ===")
    print("num class folders:", len(class_dirs))
    print("first 5 class names:", [p.name for p in class_dirs[:5]])

    sample_class = class_dirs[0]
    sample_videos = sorted(sample_class.iterdir())[:5]
    print(f"sample videos in {sample_class.name}:", [p.name for p in sample_videos])
    print("extensions seen:", {p.suffix for p in sample_class.iterdir()})

    total_videos = sum(1 for _ in split_dir.rglob("*.avi"))
    print(f"total .avi files under {split}/: {total_videos}")

# %%
print("\ndone")
