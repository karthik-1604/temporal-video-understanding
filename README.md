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
| 2 — Baselines | Frame-level pooling ✅, CNN+LSTM/GRU ✅, CLIP zero-shot ✅ | ✅ MVP baseline set complete |
| 3 — Annotation analysis | Class distribution, duration/frame-count stats, confusion matrix | ✅ |
| 4 — Explainability | Grad-CAM on selected frames, annotated showcase GIF | planned |
| 5 — Full experiment suite | Temporal-resolution/modeling/augmentation ablations, robustness, bias-mitigation, unsupervised clustering, simulated A/B evaluation | deferred (post-MVP) |
| 6 — Report & polish | Formal report, README results/charts, optional AWS exposure | deferred |

See [journey.md](journey.md) for the full running build log (decisions and why,
what exists at each step, what's next).

## Results so far

### Dataset

All 13,451 real UCF101 clips scanned for annotation analysis (0 read errors):

| Stat | Value |
|---|---|
| Duration (mean / median) | 7.16s / 6.41s |
| Frame count (mean / median) | 185.4 / 166.0 |
| FPS | bimodal: 25.0 or 29.97 (two source frame rates) |
| Class size (min / median / max) | 75 / 98 / 198 (`Basketball` is the largest class, 2x `BasketballDunk`) |

That last number turns out to matter: `Basketball` having exactly 2x
`BasketballDunk`'s training examples — and being the single largest class in
the dataset — is a more parsimonious explanation for the confusion bias
below than the motion-blindness hypothesis alone (a plain cross-entropy
classifier naturally leans toward the majority class on ambiguous inputs).
Full stats in [reports/annotation_analysis.json](reports/annotation_analysis.json).

### Baselines

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
which temporal average pooling destroys by construction.

**Baseline 2** (same frozen ResNet18 backbone → LSTM/GRU over the per-frame
sequence, instead of averaging) directly tests whether that matters:

| Model | Top-1 (test) | Macro F1 (test) | `BasketballDunk` correct |
|---|---|---|---|
| Baseline 1 (avg pool) | 96.11% | 96.52% | 11/17 |
| Baseline 2 (LSTM) | 92.69% | 92.51% | 0/17 |
| Baseline 2 (GRU) | 93.67% | 93.47% | 0/17 |

**It didn't — the RNN underperformed the simpler baseline, on aggregate and on
this specific pair.** First training run showed real overfitting/instability
(diagnosed via the train/val curves, not just the final number); a second run
fixed the process issues (gradient clipping, a second RNN layer to enable
real internal dropout, a lower learning rate) and training did stabilize —
but accuracy didn't improve and `BasketballDunk` still went to 0/17 correct in
both runs. That persistence across two different training regimes points past
"bad training" to a representational bottleneck: the backbone is
ImageNet-pretrained for static image classification, frozen, never fine-tuned
for motion — so per-frame embeddings for a dunk vs. generic dribbling in the
same scene may simply be near-identical in that feature space, in which case
no amount of RNN tuning can recover a distinction the input features never
encoded. Full writeup in [journey.md](journey.md) (Phase 8); full metrics in
[reports/baseline2_metrics.json](reports/baseline2_metrics.json).

**Baseline 3** (CLIP `ViT-B-32`, zero-shot — no UCF101 training data at all)
closes out the MVP baseline set:

| Model | Top-1 (test) | Macro F1 (test) | `BasketballDunk` correct |
|---|---|---|---|
| Baseline 1 (avg pool) | 96.11% | 96.52% | 11/17 |
| Baseline 2 (GRU, best RNN) | 93.67% | 93.47% | 0/17 |
| Baseline 3 (CLIP zero-shot) | 65.24% | 60.45% | 0/17 |

65% top-1 on a 101-way task with zero labeled examples is a strong absolute
result — but it also fails `BasketballDunk` completely, same as the RNNs.
**Checked whether that's confirmation of the Phase 8 motion-blindness
hypothesis before assuming it was**: it isn't a clean confirmation —
`BasketballDunk` is 1 of **11 classes** where CLIP zero-shot scores F1 = 0.0,
alongside classes with no plausible motion-blindness story at all (`YoYo`,
`PlayingDaf`, `Punch`). That looks more like a single naive prompt template
(`"a video of a person {class}"`) failing to produce a useful text embedding
for certain class names than a targeted confirmation of the earlier
hypothesis. **Updated after the annotation analysis below**: the class-count
imbalance (`Basketball` has 2x `BasketballDunk`'s training examples) is a
more parsimonious explanation for Baselines 1/2's specific bias *toward*
`Basketball` than motion-blindness alone — though it doesn't explain CLIP's
zero-shot failure, which never saw this dataset's class counts at all. Full
writeup in [journey.md](journey.md) (Phases 9 and 11); full metrics in
[reports/baseline3_metrics.json](reports/baseline3_metrics.json).

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
scale, and model training run in Kaggle GPU kernels (see `kaggle_kernel/`).

## Reproducing

Setup and run instructions will be added once the data pipeline and first
baseline are in place.
