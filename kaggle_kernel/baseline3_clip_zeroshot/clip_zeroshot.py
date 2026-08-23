# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -all
# ---

# %%
"""Baseline 3: CLIP zero-shot classification, no training. Evaluated on the
same UCF101 test split as Baselines 1/2 for direct comparison, using the
real open_clip-backed CLIPZeroShotClassifier (src/models/clip_zero_shot.py).
Also specifically checks the Basketball/BasketballDunk pair that Baselines 1
and 2 disagreed sharply on (see journey.md Phase 6/8) -- CLIP's image-text
embedding space is a genuinely different representation than the ImageNet-
supervised ResNet features both prior baselines used.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

subprocess.run(["git", "clone", "--depth", "1",
                 "https://github.com/karthik-1604/temporal-video-understanding.git",
                 "repo"], check=True)
sys.path.insert(0, "repo")

_gpu_check = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True,
)
if "P100" in _gpu_check.stdout:
    print(f"P100 detected ({_gpu_check.stdout.strip()}) — reinstalling torch for sm_60 support")
    subprocess.run(
        ["pip", "install", "-q", "torch==2.5.1", "torchvision==0.20.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"],
        check=True,
    )

subprocess.run(["pip", "install", "-q", "decord", "open_clip_torch"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

assert torch.cuda.is_available(), "CUDA not available"
_probe = torch.randn(1024, 1024, device="cuda:0") @ torch.randn(1024, 1024, device="cuda:0")
assert _probe.device.type == "cuda", f"matmul landed on {_probe.device}, not cuda"
DEVICE = "cuda:0"
print(f"GPU verified: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)

# %%
from src.data.ucf101_metadata import find_ucf101_root, load_ucf101_splits, resolve_video_path  # noqa: E402
from src.data.video_io import decord_clip_reader  # noqa: E402
from src.models.clip_zero_shot import CLIPZeroShotClassifier  # noqa: E402
from src.preprocessing.transforms import frames_to_pil_transformed  # noqa: E402

ROOT = find_ucf101_root()
splits = load_ucf101_splits(ROOT)
num_classes = len({s.label for s in splits["train"]})
label_to_name = {s.label: s.class_name for s in splits["train"]}
ordered_class_names = [label_to_name[i] for i in range(num_classes)]
print(f"test={len(splits['test'])} classes={num_classes}")

NUM_FRAMES = 16

clip_classifier = CLIPZeroShotClassifier(
    class_names=ordered_class_names, model_name="ViT-B-32", pretrained="openai", device=DEVICE
)
print("CLIP model loaded")

# %%
all_logits, all_labels = [], []
eval_start = time.perf_counter()
for i, sample in enumerate(splits["test"]):
    video_path = resolve_video_path(ROOT, sample)
    frames = decord_clip_reader(video_path, NUM_FRAMES, strategy="uniform", seed=None)
    clip_tensor = frames_to_pil_transformed(frames, clip_classifier.preprocess)
    logits = clip_classifier.classify_clip(clip_tensor)
    all_logits.append(logits.cpu())
    all_labels.append(sample.label)
    if (i + 1) % 200 == 0:
        elapsed = time.perf_counter() - eval_start
        print(f"{i + 1}/{len(splits['test'])} clips in {elapsed:.1f}s ({(i + 1) / elapsed:.2f} clips/sec)")

eval_elapsed = time.perf_counter() - eval_start
eval_throughput = len(splits["test"]) / eval_elapsed
print(f"total: {len(splits['test'])} clips in {eval_elapsed:.1f}s ({eval_throughput:.2f} clips/sec)")

all_logits = torch.stack(all_logits).numpy()
all_labels = np.array(all_labels)
all_preds = all_logits.argmax(axis=1)

# %%
top1 = float((all_preds == all_labels).mean())
top5 = float(top_k_accuracy_score(all_labels, all_logits, k=5, labels=list(range(num_classes))))
macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
report = classification_report(
    all_labels, all_preds, target_names=ordered_class_names, output_dict=True, zero_division=0,
)
conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))).tolist()
print(f"zero-shot test top1={top1:.4f} top5={top5:.4f} macro_f1={macro_f1:.4f}")

i_dunk = ordered_class_names.index("BasketballDunk")
i_bball = ordered_class_names.index("Basketball")
dunk_correct = int(((all_labels == i_dunk) & (all_preds == i_dunk)).sum())
dunk_total = int((all_labels == i_dunk).sum())
print(f"BasketballDunk: {dunk_correct}/{dunk_total} correct")
print("BasketballDunk row:",
      {ordered_class_names[j]: conf_matrix[i_dunk][j] for j in range(num_classes) if conf_matrix[i_dunk][j] > 0})
print("Basketball row:",
      {ordered_class_names[j]: conf_matrix[i_bball][j] for j in range(num_classes) if conf_matrix[i_bball][j] > 0})

# %%
# Per-clip latency: GPU compute only (encode T frames + cosine classify),
# excludes decode/preprocess -- same convention as Baselines 1/2's latency
# numbers. classify_clip processes one video's frame sequence at a time.
warmup_sample = splits["test"][0]
warmup_frames = decord_clip_reader(
    resolve_video_path(ROOT, warmup_sample), NUM_FRAMES, strategy="uniform", seed=None
)
warmup_tensor = frames_to_pil_transformed(warmup_frames, clip_classifier.preprocess)
for _ in range(3):
    clip_classifier.classify_clip(warmup_tensor)
torch.cuda.synchronize()

latency_start = time.perf_counter()
n_runs = 20
for _ in range(n_runs):
    clip_classifier.classify_clip(warmup_tensor)
torch.cuda.synchronize()
latency_elapsed = time.perf_counter() - latency_start
latency_ms = latency_elapsed / n_runs * 1000
print(f"CLIP inference latency (GPU compute only): {latency_ms:.2f} ms/clip")

peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
total_params = sum(p.numel() for p in clip_classifier.model.parameters())

# %%
results = {
    "model": "baseline3_clip_zeroshot",
    "clip_model": "ViT-B-32",
    "clip_pretrained": "openai",
    "num_frames": NUM_FRAMES,
    "num_classes": num_classes,
    "test_size": len(splits["test"]),
    "test_metrics": {"top1_acc": top1, "top5_acc": top5, "macro_f1": macro_f1},
    "per_class_report": report,
    "confusion_matrix": conf_matrix,
    "class_names": ordered_class_names,
    "basketball_dunk_check": {"correct": dunk_correct, "total": dunk_total},
    "eval_throughput_clips_per_sec": eval_throughput,
    "latency_ms_per_clip_gpu_only": latency_ms,
    "peak_gpu_memory_mb": peak_mem_mb,
    "params_total": total_params,
}

out_dir = Path("/kaggle/working/results")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "baseline3_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nsaved results/baseline3_metrics.json")
print("done")
