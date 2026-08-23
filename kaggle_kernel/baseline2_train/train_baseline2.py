# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -all
# ---

# %%
"""Baseline 2 (CNN+RNN): pretrained ResNet18 per-frame features -> LSTM/GRU
over the frame sequence -> classifier, trained on real UCF101.

Same frozen-backbone caching idea as Baseline 1 (train_baseline1.py), but
keeps *per-frame* embeddings (not pre-averaged) so the RNN has the ordered
sequence to consume -- the whole point of this baseline is testing whether
frame order recovers accuracy Baseline 1's average pooling can't (see
journey.md Phase 6's Basketball/BasketballDunk finding). Trains both LSTM and
GRU on the identical cached sequences since the embedding extraction (the
expensive part) is shared -- comparing both costs almost nothing extra.
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
from src.models.baseline_cnn_rnn import CNNRNNClassifier  # noqa: E402
from src.preprocessing.transforms import resize_and_normalize  # noqa: E402

ROOT = find_ucf101_root()
splits = load_ucf101_splits(ROOT)
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


def make_loader(split_name):
    dataset = VideoClipDataset(
        splits[split_name], ROOT, num_frames=NUM_FRAMES,
        clip_reader=clip_reader, sampling_strategy="uniform", transform=transform,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Backbone lives on a throwaway model instance just for feature extraction --
# both LSTM and GRU classifiers below reuse the same cached embeddings.
_backbone_holder = CNNRNNClassifier(
    num_classes=num_classes, backbone="resnet18", pretrained=True, freeze_backbone=True
).to(DEVICE)
backbone = _backbone_holder.backbone
backbone.eval()


# %%
@torch.no_grad()
def extract_sequence_embeddings(split_name):
    loader = make_loader(split_name)
    feats, labels = [], []
    start = time.perf_counter()
    for clips, batch_labels in loader:
        clips = clips.to(DEVICE, non_blocking=True)
        b, t, c, h, w = clips.shape
        frames = clips.reshape(b * t, c, h, w)
        emb = backbone(frames).reshape(b, t, -1)  # (batch, T, feature_dim) -- NOT averaged
        feats.append(emb.cpu())
        labels.append(batch_labels)
    elapsed = time.perf_counter() - start
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    print(f"{split_name}: extracted {tuple(feats.shape)} embeddings in {elapsed:.1f}s "
          f"({len(labels) / elapsed:.2f} clips/sec)")
    return feats, labels


train_feats, train_labels = extract_sequence_embeddings("train")
val_feats, val_labels = extract_sequence_embeddings("val")
test_feats, test_labels = extract_sequence_embeddings("test")

feature_dim = train_feats.shape[-1]
val_feats_gpu, val_labels_gpu = val_feats.to(DEVICE), val_labels.to(DEVICE)
test_feats_gpu, test_labels_gpu = test_feats.to(DEVICE), test_labels.to(DEVICE)


# %%
def train_rnn_head(rnn_type: str):
    model = CNNRNNClassifier(
        num_classes=num_classes, backbone="resnet18", pretrained=False,
        freeze_backbone=True, rnn_type=rnn_type, hidden_dim=256,
        num_layers=2, dropout=0.3,  # num_layers>1 activates PyTorch's internal
        # inter-layer recurrent dropout -- with num_layers=1 (the first attempt)
        # that dropout is silently a no-op, leaving only the final classifier's
        # dropout as regularization despite the RNN having far more trainable
        # params than Baseline 1's head. See journey.md Phase 7/8.
    ).to(DEVICE)
    # backbone weights are unused here (forward_from_features skips it) --
    # only the rnn + classifier are trained/evaluated.
    rnn_lr = LR / 2  # first attempt reused Baseline 1's LR unchanged for a
    # differently-shaped loss landscape (RNN vs. simple MLP) -- halved here.
    optimizer = torch.optim.Adam(
        list(model.rnn.parameters()) + list(model.classifier.parameters()),
        lr=rnn_lr, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=64, shuffle=True)

    history = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
        for feats_batch, labels_batch in train_loader:
            feats_batch, labels_batch = feats_batch.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model.forward_from_features(feats_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            # RNN training is prone to occasional gradient spikes; the first
            # attempt had no clipping and showed a real instability spike in
            # GRU's later epochs (val_acc crashed 93.7% -> 83.8% one epoch).
            torch.nn.utils.clip_grad_norm_(
                list(model.rnn.parameters()) + list(model.classifier.parameters()), max_norm=2.0
            )
            optimizer.step()
            epoch_loss += loss.item() * len(labels_batch)
            epoch_correct += (logits.argmax(1) == labels_batch).sum().item()
            epoch_n += len(labels_batch)

        model.eval()
        with torch.no_grad():
            val_logits = model.forward_from_features(val_feats_gpu)
            val_loss = criterion(val_logits, val_labels_gpu).item()
            val_acc = (val_logits.argmax(1) == val_labels_gpu).float().mean().item()

        train_loss, train_acc = epoch_loss / epoch_n, epoch_correct / epoch_n
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"[{rnn_type}] epoch {epoch:3d} train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"[{rnn_type}] best val_acc={best_val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model.forward_from_features(test_feats_gpu)
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = test_logits.argmax(1).cpu().numpy()

    test_labels_np = test_labels.numpy()
    top1 = float((test_preds == test_labels_np).mean())
    top5 = float(top_k_accuracy_score(test_labels_np, test_probs, k=5, labels=list(range(num_classes))))
    macro_f1 = float(f1_score(test_labels_np, test_preds, average="macro"))
    report = classification_report(
        test_labels_np, test_preds, target_names=ordered_class_names,
        output_dict=True, zero_division=0,
    )
    conf_matrix = confusion_matrix(test_labels_np, test_preds, labels=list(range(num_classes))).tolist()
    print(f"[{rnn_type}] test top1={top1:.4f} top5={top5:.4f} macro_f1={macro_f1:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.rnn.parameters()) + \
        sum(p.numel() for p in model.classifier.parameters())

    return {
        "rnn_type": rnn_type,
        "params": {"total": total_params, "trainable": trainable_params},
        "training": {"epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "best_val_acc": best_val_acc},
        "history": history,
        "test_metrics": {"top1_acc": top1, "top5_acc": top5, "macro_f1": macro_f1},
        "per_class_report": report,
        "confusion_matrix": conf_matrix,
    }, model


lstm_results, lstm_model = train_rnn_head("lstm")
gru_results, gru_model = train_rnn_head("gru")

# %%
# Inference latency for the better of the two (full model: backbone + rnn +
# classifier), GPU compute only, batch sizes 1 and 8 -- same protocol as
# Baseline 1 for a fair comparison.
best_variant, best_model = (
    ("lstm", lstm_model) if lstm_results["test_metrics"]["top1_acc"] >= gru_results["test_metrics"]["top1_acc"]
    else ("gru", gru_model)
)
best_model.backbone.load_state_dict(backbone.state_dict())
best_model.eval()

latency_results = {}
for batch_size in (1, 8):
    dummy = torch.randn(batch_size, NUM_FRAMES, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        for _ in range(3):
            best_model(dummy)
        torch.cuda.synchronize()
        start = time.perf_counter()
        n_runs = 20
        for _ in range(n_runs):
            best_model(dummy)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    latency_results[f"batch_{batch_size}"] = {
        "mean_latency_ms": elapsed / n_runs * 1000,
        "throughput_clips_per_sec": batch_size * n_runs / elapsed,
    }
print(f"latency ({best_variant}):", json.dumps(latency_results, indent=2))
peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

# %%
results = {
    "model": "baseline2_cnn_rnn",
    "backbone": "resnet18",
    "num_frames": NUM_FRAMES,
    "num_classes": num_classes,
    "dataset_sizes": {k: len(v) for k, v in splits.items()},
    "class_names": ordered_class_names,
    "lstm": lstm_results,
    "gru": gru_results,
    "best_variant": best_variant,
    "latency": latency_results,
    "peak_gpu_memory_mb": peak_mem_mb,
}

out_dir = Path("/kaggle/working/results")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "baseline2_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

torch.save(best_model.state_dict(), out_dir / f"baseline2_{best_variant}_model.pt")

print(f"\nsaved results/baseline2_metrics.json and results/baseline2_{best_variant}_model.pt")
print("done")
