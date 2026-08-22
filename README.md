# Temporal Video Understanding with CNN-RNN and Video Transformers

How much does modeling *temporal* structure actually help video understanding,
compared to just looking at individual frames? This project answers that
question empirically on action recognition (UCF101): it compares frame-level
pooling, a CNN+RNN model, a video transformer, and zero-shot vision-language
classification on the same task, then runs a research-experiment suite around
temporal resolution, augmentation, class imbalance/bias, and robustness to
degraded input (dropped frames, low resolution, compression).

This is a model-development and experimentation project, not a video application
or product.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 — Scoping | Problem definition, dataset/compute decisions, repo setup | ✅ |
| 1 — Data pipeline | Configurable UCF101 loader, frame sampling | ✅ |
| 2 — Baselines | Frame-level pooling ✅, CLIP zero-shot (built, not yet run), CNN+LSTM/GRU | in progress |
| 3 — Annotation analysis | Class distribution, duration/frame-count stats, confusion matrix | in progress (confusion matrix done via Baseline 1 eval) |
| 4 — Explainability | Grad-CAM on selected frames, annotated showcase GIF | planned |
| 5 — Full experiment suite | Temporal-resolution/modeling/augmentation ablations, robustness, bias-mitigation, unsupervised clustering, simulated A/B evaluation | deferred (post-MVP) |
| 6 — Report & polish | Formal report, README results/charts, optional AWS exposure | deferred |

See [journey.md](journey.md) for the full running build log (decisions and why,
what exists at each step, what's next).

## Results so far

**Baseline 1** (frame-level: frozen pretrained ResNet18 features → temporal
average pooling → MLP head), trained on the real UCF101 split
(10,055 train / 1,673 val / 1,723 test, 101 classes):

| Metric | Value |
|---|---|
| Top-1 accuracy (test) | 96.11% |
| Top-5 accuracy (test) | 99.65% |
| Macro F1 (test) | 96.52% |
| Trainable / total params | 157,285 / 11,333,797 |
| Inference latency (batch 1) | 8.1 ms/clip |
| Throughput (batch 8) | 156.4 clips/sec |
| Peak GPU memory | 1.77 GB |

Full metrics (per-epoch history, per-class report, confusion matrix) in
[reports/baseline1_metrics.json](reports/baseline1_metrics.json).

The two worst-performing classes, `BasketballDunk` (F1 0.49) and `Basketball`
(F1 0.59), are confused with each other specifically — in both directions —
not just generically "hard." Both share the same visual scene (court, hoop,
players), so the only real distinguishing signal is the dunking *motion*,
which temporal average pooling destroys by construction. This is the
project's core research question in miniature, and a natural test case for
whether Baseline 2's temporal modeling actually resolves it.

## Repo structure

```text
src/
  data/            # dataset loading, frame sampling
  preprocessing/    # augmentation, feature extraction
  models/          # baselines, temporal models, CLIP zero-shot
  training/        # training loops, config-driven
  evaluation/      # metrics, ablations, robustness tests
  explainability/  # Grad-CAM / attention visualization
  analysis/        # annotation analysis, clustering
tests/             # unit tests against synthetic fixtures
configs/           # YAML/Hydra configs
kaggle_kernel/     # Kaggle-run scripts (exploration, training) + their output logs
reports/           # generated results (metrics, figures)
```

## Compute

Local development targets a CPU-only machine — all `src/` code is written to run
against tiny synthetic fixtures for testing. Real data download, preprocessing at
scale, and model training run in Kaggle GPU kernels (see `notebooks/`).

## Reproducing

Setup and run instructions will be added once the data pipeline and first
baseline are in place.
