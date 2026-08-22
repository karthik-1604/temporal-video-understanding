# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -all
# ---

# %%
"""Smoke test for the Baseline 1 pipeline on real UCF101 data, on a small
subset, before committing to a full training run. Verifies: GPU actually
works (not just torch.cuda.is_available()), the full data pipeline
(metadata -> VideoClipDataset -> decord decoding -> resize/normalize ->
model forward) runs end to end on real videos, and measures real
decode+forward throughput to size the full run.
"""

import subprocess
import sys
import time

subprocess.run(["git", "clone", "--depth", "1",
                 "https://github.com/karthik-1604/temporal-video-understanding.git",
                 "repo"], check=True)
sys.path.insert(0, "repo")

# Kaggle's preinstalled torch build has dropped support for older GPU
# architectures (P100/Pascal, sm_60) in newer releases — a known Kaggle
# platform issue (Kaggle/docker-python#1546), not fixable without reinstalling
# torch before the first `import torch` in this process. Checked, not applied
# blindly: version 1 of this kernel failed with exactly this error
# (torch.AcceleratorError: no kernel image is available) when assigned a P100.
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

import torch  # noqa: E402

# Hard GPU verification (see KAGGLE_WORKFLOW_NOTES.md: is_available() alone is
# not sufficient proof) — check the real matmul result's device, not just
# whether the call raised.
assert torch.cuda.is_available(), "CUDA not available"
_probe = torch.randn(1024, 1024, device="cuda:0") @ torch.randn(1024, 1024, device="cuda:0")
assert _probe.device.type == "cuda", f"matmul landed on {_probe.device}, not cuda"
print(f"GPU verified: {torch.cuda.get_device_name(0)}")

subprocess.run(["pip", "install", "-q", "decord"], check=True)

# %%
from torch.utils.data import DataLoader  # noqa: E402

from src.data.ucf101_metadata import find_ucf101_root, load_ucf101_splits  # noqa: E402
from src.data.video_dataset import VideoClipDataset  # noqa: E402
from src.data.video_io import decord_clip_reader  # noqa: E402
from src.models.baseline_frame_pool import FramePoolClassifier  # noqa: E402
from src.preprocessing.transforms import resize_and_normalize  # noqa: E402

ROOT = find_ucf101_root()
print(f"dataset root: {ROOT}")

splits = load_ucf101_splits(ROOT)
num_classes = len({s.label for s in splits["train"]})
print(f"train={len(splits['train'])} val={len(splits.get('val', []))} "
      f"test={len(splits['test'])} classes={num_classes}")

# %%
NUM_FRAMES = 16
SMOKE_TRAIN_N = 128
SMOKE_BATCH = 8

subset = splits["train"][:SMOKE_TRAIN_N]


def clip_reader(video_path, num_frames, strategy, seed):
    return decord_clip_reader(video_path, num_frames, strategy, seed)


def transform(frames):
    return resize_and_normalize(frames, image_size=224)


dataset = VideoClipDataset(
    subset, ROOT, num_frames=NUM_FRAMES, clip_reader=clip_reader, transform=transform
)
loader = DataLoader(dataset, batch_size=SMOKE_BATCH, shuffle=False, num_workers=2)

model = FramePoolClassifier(
    num_classes=num_classes, backbone="resnet18", pretrained=True, freeze_backbone=True
).cuda()
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"params: total={total_params:,} trainable={trainable_params:,}")

# %%
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
n_seen = 0
with torch.no_grad():
    for clips, labels in loader:
        clips = clips.cuda(non_blocking=True)
        logits = model(clips)
        n_seen += clips.shape[0]
elapsed = time.perf_counter() - start
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

videos_per_sec = n_seen / elapsed
print(f"processed {n_seen} clips in {elapsed:.2f}s -> {videos_per_sec:.2f} clips/sec")
print(f"peak GPU memory: {peak_mem_mb:.1f} MB")

full_train_n = len(splits["train"])
est_full_epoch_min = full_train_n / videos_per_sec / 60
print(f"estimated time for one full-train-set forward pass ({full_train_n} clips): "
      f"{est_full_epoch_min:.1f} min")

print("\nsmoke test passed")
