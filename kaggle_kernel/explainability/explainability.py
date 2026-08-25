# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -all
# ---

# %%
"""Explainability: Grad-CAM on Baseline 1, applied to real examples including
the Basketball/BasketballDunk pair investigated across journey.md Phases
6/8/9/11. Retrains Baseline 1's head fresh (fast, ~11 min, avoids managing a
separate checkpoint artifact -- same choice as every other Kaggle kernel in
this project) rather than reusing a saved checkpoint. Produces annotated
static frames + a showcase GIF per example; only these small artifacts (not
video data) are pulled back locally.
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

subprocess.run(["pip", "install", "-q", "decord"], check=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

assert torch.cuda.is_available(), "CUDA not available"
_probe = torch.randn(1024, 1024, device="cuda:0") @ torch.randn(1024, 1024, device="cuda:0")
assert _probe.device.type == "cuda", f"matmul landed on {_probe.device}, not cuda"
DEVICE = torch.device("cuda:0")
print(f"GPU verified: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)

# %%
from src.data.ucf101_metadata import find_ucf101_root, load_ucf101_splits, resolve_video_path  # noqa: E402
from src.data.video_dataset import VideoClipDataset  # noqa: E402
from src.data.video_io import decord_clip_reader  # noqa: E402
from src.explainability.gradcam import GradCAM  # noqa: E402
from src.explainability.visualization import overlay_heatmap_on_frame  # noqa: E402
from src.models.baseline_frame_pool import FramePoolClassifier  # noqa: E402
from src.preprocessing.transforms import resize_and_normalize  # noqa: E402

ROOT = find_ucf101_root()
splits = load_ucf101_splits(ROOT)
num_classes = len({s.label for s in splits["train"]})
label_to_name = {s.label: s.class_name for s in splits["train"]}
ordered_class_names = [label_to_name[i] for i in range(num_classes)]
NUM_FRAMES = 16
IMAGE_SIZE = 224


def clip_reader(video_path, num_frames, strategy, seed):
    return decord_clip_reader(video_path, num_frames, strategy, seed)


def transform(frames):
    return resize_and_normalize(frames, image_size=IMAGE_SIZE)


def make_loader(split_name):
    dataset = VideoClipDataset(
        splits[split_name], ROOT, num_frames=NUM_FRAMES,
        clip_reader=clip_reader, sampling_strategy="uniform", transform=transform,
    )
    return DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)


# %%
# Retrain Baseline 1's head (identical recipe to kaggle_kernel/baseline1_train)
model = FramePoolClassifier(
    num_classes=num_classes, backbone="resnet18", pretrained=True, freeze_backbone=True
).to(DEVICE)


@torch.no_grad()
def extract_embeddings(split_name):
    model.eval()
    loader = make_loader(split_name)
    feats, labels, sample_refs = [], [], []
    start = time.perf_counter()
    for clips, batch_labels in loader:
        clips = clips.to(DEVICE, non_blocking=True)
        b, t, c, h, w = clips.shape
        frames = clips.reshape(b * t, c, h, w)
        emb = model.backbone(frames).reshape(b, t, -1).mean(dim=1)
        feats.append(emb.cpu())
        labels.append(batch_labels)
    elapsed = time.perf_counter() - start
    feats, labels = torch.cat(feats), torch.cat(labels)
    print(f"{split_name}: extracted {len(labels)} embeddings in {elapsed:.1f}s "
          f"({len(labels) / elapsed:.2f} clips/sec)")
    return feats, labels


train_feats, train_labels = extract_embeddings("train")
val_feats, val_labels = extract_embeddings("val")

classifier = model.classifier
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=64, shuffle=True)
val_feats_gpu, val_labels_gpu = val_feats.to(DEVICE), val_labels.to(DEVICE)

best_val_acc, best_state = 0.0, None
for epoch in range(40):
    classifier.train()
    for feats_batch, labels_batch in train_loader:
        feats_batch, labels_batch = feats_batch.to(DEVICE), labels_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(classifier(feats_batch), labels_batch)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        val_acc = (classifier(val_feats_gpu).argmax(1) == val_labels_gpu).float().mean().item()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

classifier.load_state_dict(best_state)
print(f"retrained head: best val_acc={best_val_acc:.4f}")

# %%
# Find concrete example videos: predict on the test split's Basketball/
# BasketballDunk samples plus a perfect-F1 class (YoYo) to pick real success/
# failure cases, not synthetic ones.
model.eval()
target_classes = {"BasketballDunk", "Basketball", "YoYo"}
candidates = [s for s in splits["test"] if s.class_name in target_classes]

examples = {}  # tag -> VideoSample
with torch.no_grad():
    for sample in candidates:
        video_path = resolve_video_path(ROOT, sample)
        frames = decord_clip_reader(video_path, NUM_FRAMES, strategy="uniform", seed=None)
        clip = transform(frames).unsqueeze(0).to(DEVICE)
        pred = int(model(clip).argmax(dim=1).item())
        pred_name = ordered_class_names[pred]

        if sample.class_name == "BasketballDunk" and pred_name == "Basketball" and "dunk_misclassified" not in examples:
            examples["dunk_misclassified"] = sample
        elif sample.class_name == "Basketball" and pred_name == "Basketball" and "basketball_correct" not in examples:
            examples["basketball_correct"] = sample
        elif sample.class_name == "YoYo" and pred_name == "YoYo" and "yoyo_correct" not in examples:
            examples["yoyo_correct"] = sample

        if len(examples) == 3:
            break

print("selected examples:", {k: v.clip_name for k, v in examples.items()})

# %%
# Grad-CAM + overlay + GIF per example
gradcam = GradCAM(model, target_layer_name="layer4")
out_dir = Path("/kaggle/working/results")
out_dir.mkdir(parents=True, exist_ok=True)
GIF_SIZE = 180

example_metadata = {}
for tag, sample in examples.items():
    video_path = resolve_video_path(ROOT, sample)
    frames = decord_clip_reader(video_path, NUM_FRAMES, strategy="uniform", seed=None)  # (T,H,W,C) uint8
    clip = transform(frames).unsqueeze(0).to(DEVICE)
    clip.requires_grad_(True)

    heatmaps, pred_class = gradcam.generate(clip)  # (T, h, w), int
    pred_name = ordered_class_names[pred_class]
    heatmaps_np = heatmaps.cpu().numpy()

    gif_frames = []
    for t in range(NUM_FRAMES):
        overlay = overlay_heatmap_on_frame(frames[t], heatmaps_np[t], alpha=0.45)
        img = Image.fromarray(overlay).resize((GIF_SIZE, GIF_SIZE), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        # Two lines (true/pred separately) since one long line (e.g.
        # "true:BasketballDunk pred:Basketball") gets clipped at this width.
        draw.rectangle([0, 0, GIF_SIZE, 28], fill=(0, 0, 0))
        draw.text((2, 2), f"true: {sample.class_name}", fill=(255, 255, 255))
        pred_color = (120, 255, 120) if pred_name == sample.class_name else (255, 120, 120)
        draw.text((2, 15), f"pred: {pred_name}", fill=pred_color)
        gif_frames.append(img)

    gif_path = out_dir / f"gradcam_{tag}.gif"
    gif_frames[0].save(
        gif_path, save_all=True, append_images=gif_frames[1:], duration=200, loop=0
    )
    print(f"saved {gif_path} ({gif_path.stat().st_size} bytes)")

    example_metadata[tag] = {
        "clip_name": sample.clip_name,
        "true_class": sample.class_name,
        "predicted_class": pred_name,
        "correct": sample.class_name == pred_name,
    }

with open(out_dir / "explainability_examples.json", "w") as f:
    json.dump(example_metadata, f, indent=2)

print("\nsaved results/explainability_examples.json and per-example GIFs")
print("done")
