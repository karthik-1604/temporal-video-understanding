# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -all
# ---

# %%
"""Baseline 1 (frame-level): pretrained ResNet18 features + temporal average
pooling + MLP classifier, trained on real UCF101.

Efficiency design: the backbone is frozen (see smoketest.py's measured
15.14 clips/sec on a P100 — the backbone forward dominates cost), so instead
of re-running the backbone every epoch, embeddings are extracted once per
split and the small MLP head is trained on cached features. This matches the
smoke test's real numbers: ~11 min for one full-train-set backbone pass,
vs. near-instant for many epochs of MLP training on cached 512-dim vectors.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

assert torch.cuda.is_available(), "CUDA not available"
_probe = torch.randn(1024, 1024, device="cuda:0") @ torch.randn(1024, 1024, device="cuda:0")
assert _probe.device.type == "cuda", f"matmul landed on {_probe.device}, not cuda"
DEVICE = torch.device("cuda:0")
print(f"GPU verified: {torch.cuda.get_device_name(0)}")

subprocess.run(["pip", "install", "-q", "decord"], check=True)

torch.manual_seed(42)

# %%
from src.data.ucf101_metadata import find_ucf101_root, load_ucf101_splits  # noqa: E402
from src.data.video_dataset import VideoClipDataset  # noqa: E402
from src.data.video_io import decord_clip_reader  # noqa: E402
from src.models.baseline_frame_pool import FramePoolClassifier  # noqa: E402
from src.preprocessing.transforms import resize_and_normalize  # noqa: E402

ROOT = find_ucf101_root()
splits = load_ucf101_splits(ROOT)
class_names = sorted({s.class_name for s in splits["train"]}, key=lambda n: n)
num_classes = len({s.label for s in splits["train"]})
label_to_name = {s.label: s.class_name for s in splits["train"]}
ordered_class_names = [label_to_name[i] for i in range(num_classes)]
print(f"train={len(splits['train'])} val={len(splits.get('val', []))} "
      f"test={len(splits['test'])} classes={num_classes}")

NUM_FRAMES = 16
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4


def clip_reader(video_path, num_frames, strategy, seed):
    return decord_clip_reader(video_path, num_frames, strategy, seed)


def transform(frames):
    return resize_and_normalize(frames, image_size=224)


def make_loader(split_name, strategy):
    dataset = VideoClipDataset(
        splits[split_name], ROOT, num_frames=NUM_FRAMES,
        clip_reader=clip_reader, sampling_strategy=strategy, transform=transform,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


model = FramePoolClassifier(
    num_classes=num_classes, backbone="resnet18", pretrained=True, freeze_backbone=True
).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"params: total={total_params:,} trainable={trainable_params:,}")


# %%
@torch.no_grad()
def extract_embeddings(split_name):
    model.eval()
    loader = make_loader(split_name, strategy="uniform")
    feats, labels = [], []
    start = time.perf_counter()
    for clips, batch_labels in loader:
        clips = clips.to(DEVICE, non_blocking=True)
        b, t, c, h, w = clips.shape
        frames = clips.reshape(b * t, c, h, w)
        emb = model.backbone(frames).reshape(b, t, -1).mean(dim=1)
        feats.append(emb.cpu())
        labels.append(batch_labels)
    elapsed = time.perf_counter() - start
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    print(f"{split_name}: extracted {len(labels)} embeddings in {elapsed:.1f}s "
          f"({len(labels) / elapsed:.2f} clips/sec)")
    return feats, labels


train_feats, train_labels = extract_embeddings("train")
val_feats, val_labels = extract_embeddings("val")
test_feats, test_labels = extract_embeddings("test")

# %%
classifier = model.classifier.to(DEVICE)
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(
    TensorDataset(train_feats, train_labels), batch_size=64, shuffle=True
)
val_feats_gpu = val_feats.to(DEVICE)
val_labels_gpu = val_labels.to(DEVICE)

history = []
best_val_acc = 0.0
best_state = None

for epoch in range(EPOCHS):
    classifier.train()
    epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
    for feats_batch, labels_batch in train_loader:
        feats_batch, labels_batch = feats_batch.to(DEVICE), labels_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = classifier(feats_batch)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(labels_batch)
        epoch_correct += (logits.argmax(1) == labels_batch).sum().item()
        epoch_n += len(labels_batch)

    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_feats_gpu)
        val_loss = criterion(val_logits, val_labels_gpu).item()
        val_acc = (val_logits.argmax(1) == val_labels_gpu).float().mean().item()

    train_loss = epoch_loss / epoch_n
    train_acc = epoch_correct / epoch_n
    history.append({
        "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
        "val_loss": val_loss, "val_acc": val_acc,
    })
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:3d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

classifier.load_state_dict(best_state)
print(f"best val_acc={best_val_acc:.4f}")

# %%
classifier.eval()
test_feats_gpu = test_feats.to(DEVICE)
with torch.no_grad():
    test_logits = classifier(test_feats_gpu)
    test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
    test_preds = test_logits.argmax(1).cpu().numpy()

test_labels_np = test_labels.numpy()
top1_acc = float((test_preds == test_labels_np).mean())
top5_acc = float(top_k_accuracy_score(test_labels_np, test_probs, k=5, labels=list(range(num_classes))))
macro_f1 = float(f1_score(test_labels_np, test_preds, average="macro"))
report = classification_report(
    test_labels_np, test_preds, target_names=ordered_class_names,
    output_dict=True, zero_division=0,
)
conf_matrix = confusion_matrix(test_labels_np, test_preds, labels=list(range(num_classes))).tolist()

print(f"test top1_acc={top1_acc:.4f} top5_acc={top5_acc:.4f} macro_f1={macro_f1:.4f}")

# %%
# Inference latency/throughput on the full model (backbone + classifier),
# GPU compute only (excludes data loading/decoding), batch size 1 and 8.
model.eval()
latency_results = {}
for batch_size in (1, 8):
    dummy = torch.randn(batch_size, NUM_FRAMES, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(dummy)
        torch.cuda.synchronize()
        start = time.perf_counter()
        n_runs = 20
        for _ in range(n_runs):
            model(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    latency_results[f"batch_{batch_size}"] = {
        "mean_latency_ms": elapsed / n_runs * 1000,
        "throughput_clips_per_sec": batch_size * n_runs / elapsed,
    }
print("latency:", json.dumps(latency_results, indent=2))

peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

# %%
results = {
    "model": "baseline1_frame_pool",
    "backbone": "resnet18",
    "num_frames": NUM_FRAMES,
    "num_classes": num_classes,
    "dataset_sizes": {k: len(v) for k, v in splits.items()},
    "params": {"total": total_params, "trainable": trainable_params},
    "training": {"epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "best_val_acc": best_val_acc},
    "history": history,
    "test_metrics": {
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "macro_f1": macro_f1,
    },
    "per_class_report": report,
    "confusion_matrix": conf_matrix,
    "class_names": ordered_class_names,
    "latency": latency_results,
    "peak_gpu_memory_mb": peak_mem_mb,
}

out_dir = Path("/kaggle/working/results")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "baseline1_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

torch.save(classifier.state_dict(), out_dir / "baseline1_classifier_head.pt")

print("\nsaved results/baseline1_metrics.json and results/baseline1_classifier_head.pt")
print("done")
