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
| 1 — Data pipeline | Configurable UCF101 loader, frame sampling | planned |
| 2 — Baselines | Frame-level pooling, CNN+LSTM/GRU, CLIP zero-shot | planned |
| 3 — Annotation analysis | Class distribution, duration/frame-count stats, confusion matrix | planned |
| 4 — Explainability | Grad-CAM on selected frames | planned |
| 5 — Full experiment suite | Temporal-resolution/modeling/augmentation ablations, robustness, bias-mitigation, unsupervised clustering, simulated A/B evaluation | deferred (post-MVP) |
| 6 — Report & polish | Formal report, README results/charts, optional AWS exposure | deferred |

See [journey.md](journey.md) for the full running build log (decisions and why,
what exists at each step, what's next).

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
notebooks/         # Kaggle-run notebooks (paired .py + .ipynb)
reports/           # generated figures/results
```

## Compute

Local development targets a CPU-only machine — all `src/` code is written to run
against tiny synthetic fixtures for testing. Real data download, preprocessing at
scale, and model training run in Kaggle GPU kernels (see `notebooks/`).

## Reproducing

Setup and run instructions will be added once the data pipeline and first
baseline are in place.
