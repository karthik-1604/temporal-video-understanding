# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -all
# ---

# %%
"""Annotation analysis: class distribution, video duration, and frame-count
statistics across the real UCF101 splits. CPU-only -- this is metadata-only
(decord opens each video's index without decoding frames), no GPU needed, per
the project's own "does this need a GPU at all?" rule. Per-class difficulty
and confusion matrices are NOT recomputed here -- they already exist in
reports/baseline{1,2,3}_metrics.json from the training kernels and are
analyzed locally instead.
"""

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

subprocess.run(["git", "clone", "--depth", "1",
                 "https://github.com/karthik-1604/temporal-video-understanding.git",
                 "repo"], check=True)
sys.path.insert(0, "repo")

subprocess.run(["pip", "install", "-q", "decord"], check=True)

import decord  # noqa: E402
import numpy as np  # noqa: E402

from src.data.ucf101_metadata import find_ucf101_root, load_ucf101_splits, resolve_video_path  # noqa: E402

ROOT = find_ucf101_root()
splits = load_ucf101_splits(ROOT)
print({k: len(v) for k, v in splits.items()})

# %%
class_distribution = {
    split: dict(sorted(Counter(s.class_name for s in samples).items()))
    for split, samples in splits.items()
}

# %%
durations, frame_counts, fps_values = [], [], []
per_class_durations = defaultdict(list)
per_class_frame_counts = defaultdict(list)
errors = []

start = time.perf_counter()
total = sum(len(v) for v in splits.values())
seen = 0
for split_name, samples in splits.items():
    for sample in samples:
        seen += 1
        path = resolve_video_path(ROOT, sample)
        try:
            vr = decord.VideoReader(str(path))
            n_frames = len(vr)
            fps = vr.get_avg_fps()
            duration = n_frames / fps if fps > 0 else None
        except Exception as e:  # a handful of corrupt/unreadable files is expected in-the-wild
            errors.append({"path": str(path), "error": str(e)})
            continue

        frame_counts.append(n_frames)
        fps_values.append(fps)
        if duration is not None:
            durations.append(duration)
            per_class_durations[sample.class_name].append(duration)
        per_class_frame_counts[sample.class_name].append(n_frames)

        if seen % 2000 == 0:
            elapsed = time.perf_counter() - start
            print(f"{seen}/{total} videos scanned in {elapsed:.1f}s ({seen / elapsed:.1f} videos/sec)")

elapsed = time.perf_counter() - start
print(f"total: {seen} videos scanned in {elapsed:.1f}s ({seen / elapsed:.1f} videos/sec), "
      f"{len(errors)} read errors")


# %%
def describe(values):
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


duration_stats = describe(durations)
frame_count_stats = describe(frame_counts)
fps_stats = describe(fps_values)

per_class_avg_duration = {
    name: float(np.mean(vals)) for name, vals in sorted(per_class_durations.items())
}
per_class_avg_frame_count = {
    name: float(np.mean(vals)) for name, vals in sorted(per_class_frame_counts.items())
}

print("duration_stats (seconds):", json.dumps(duration_stats, indent=2))
print("frame_count_stats:", json.dumps(frame_count_stats, indent=2))
print("fps_stats:", json.dumps(fps_stats, indent=2))

# %%
results = {
    "dataset_sizes": {k: len(v) for k, v in splits.items()},
    "class_distribution": class_distribution,
    "duration_stats_seconds": duration_stats,
    "frame_count_stats": frame_count_stats,
    "fps_stats": fps_stats,
    "per_class_avg_duration_seconds": per_class_avg_duration,
    "per_class_avg_frame_count": per_class_avg_frame_count,
    "read_errors": errors,
    "scan_throughput_videos_per_sec": seen / elapsed,
}

out_dir = Path("/kaggle/working/results")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "annotation_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nsaved results/annotation_analysis.json")
print("done")
