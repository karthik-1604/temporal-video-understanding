# Journey

Running build log for this project. Updated at the end of every phase/session.

---

## Phase 0 — Scoping and repo setup (2026-08-22)

### Decisions made and why

- **Problem**: temporal video understanding / action recognition — comparing how
  much temporal modeling improves accuracy over frame-level image features, plus
  a research-experiment suite (temporal resolution, temporal modeling
  architecture, augmentation, class imbalance, robustness), annotation analysis,
  explainability, and a small unsupervised component. Chosen to match a data
  science / AI internship focused on video analysis and predictive modeling,
  not because it was the only reasonable option — an initial draft plan was reviewed
  against the actual target job description line-by-line rather than taken at
  face value.
- **Three additions folded into the original plan** after that comparison, because
  the target JD named them explicitly and the initial draft had no corresponding
  component:
  - An **LLM/VLM baseline** (CLIP zero-shot classification, no training) plus
    LLM-assisted (then hand-verified) failure-analysis writeups from the
    confusion matrix — the JD's must-have skills line pairs "video analysis
    algorithms" with "LLMs" as a single item, and nothing in the original plan
    touched LLMs at all.
  - A **simulated online A/B-testing harness** as a Phase 2 item, on top of the
    offline ablations already planned — the JD separately names "online A/B
    testing methodologies," not just offline evaluation.
  - **Reframing the class-imbalance experiment as bias mitigation** (per-class
    calibration, not just accuracy) — the JD's must-have skills line names "bias
    mitigation" specifically.
  - Rejected alternative: leave the plan as originally drafted and treat these as
    out of scope. Rejected because all three are cheap to add (CLIP zero-shot is
    inference-only; the A/B-testing harness reuses statistics already familiar
    from prior project work) relative to how directly they map onto explicit JD
    lines.
- **Build order: lean MVP first, expand to a full plan afterward** — chosen given
  a near-term application deadline, over building the full plan (all models, all
  five experiments, unsupervised clustering, optional cloud exposure) up front.
  The full plan would take considerably longer than the time available before the
  target application window.
- **Dataset: UCF101** — a well-known, manageable-size public action-recognition
  benchmark, over jumping straight to a larger dataset (e.g. Something-Something
  V2), which was kept as an explicit stretch/extension rather than the starting
  point.
- **Compute: Kaggle GPU kernels for anything data- or training-related; local
  machine (no GPU) restricted to code and unit tests on synthetic fixtures only.**
  Rejected doing any real data download or training locally — the local machine
  has no GPU and downloading a full video dataset locally would be wasted
  disk/time on a machine that can't train on it anyway.

### What exists after this step

- Git repo initialized, with repo-local (not global) git identity.
- Directory skeleton: `src/{data,preprocessing,models,training,evaluation,
  explainability,analysis}/`, `tests/`, `configs/`, `notebooks/`, `reports/`,
  `docs/images/`, `kaggle_kernel/`.
- `.gitignore` covering local-only planning docs, venv, data/checkpoints, and
  notebook checkpoints.
- Local venv with minimal dev dependencies (pytest, numpy, pyyaml) — no
  training-time libraries installed locally, since there's no local GPU.
- First real code: config-driven frame-index sampling
  (`src/data/sampling.py` — uniform and random strategies, short-clip padding)
  and a dataset config schema (`src/data/config.py`, YAML-backed,
  `configs/dataset/ucf101.yaml`) satisfying the "dataset must be configurable"
  requirement. Both are pure-logic and fully unit-tested against synthetic
  values (20 tests, `tests/test_sampling.py` + `tests/test_config.py`), no real
  video data involved — this is the shared building block every baseline and
  the temporal-resolution experiment will use.

### Next step (from Phase 0)

Build the actual UCF101 dataset loader (metadata parsing, class list, split
files) designed to run inside a Kaggle kernel against the real download, while
keeping the loader's pure logic (path resolution, label mapping) unit-testable
locally against a synthetic fake directory tree. Then stand up the first Kaggle
kernel (CPU-only, no GPU needed yet) to fetch and inspect the real UCF101
metadata before writing any baseline model code.

---

## Phase 1 — Real UCF101 layout confirmed, metadata parser built (2026-08-22)

### Decisions made and why

- **Chose the `matthewjansen/ucf101-action-recognition` Kaggle dataset** over
  several other UCF101 uploads (compared ~10 candidates by usability rating and
  download count) — highest usability rating (1.0) combined with a high
  download/vote count, and it ships pre-split `train/test/val` folders rather
  than requiring manual splitting from the raw UCF101 `.rar` archive.
- **Verified the real dataset layout with a CPU-only Kaggle exploration kernel
  before writing any parsing code**, instead of assuming the classic official
  UCF101 format (`classInd.txt` / `trainlist01.txt` / `testlist01.txt`). This
  paid off immediately: the real layout is much simpler than the classic
  format — `train.csv` / `test.csv` / `val.csv` with columns `clip_name,
  clip_path, label` (label already a class-name string, no separate class-index
  file to cross-reference) — so writing to the assumed classic format would
  have produced a parser for a format this dataset doesn't actually use.
  Confirmed 101 classes, 10,055 train clips, 1,723 test clips, all `.avi`.
  - Also confirmed the actual mount path is one level deeper than the naive
    guess: `/kaggle/input/datasets/matthewjansen/ucf101-action-recognition`,
    not `/kaggle/input/ucf101-action-recognition` — matches a known gotcha
    already documented in the cross-project Kaggle notes (mount paths aren't
    always what the dataset slug suggests).
- **No data left the local machine** — the exploration kernel ran entirely on
  Kaggle; only its small text log (~6 KB) was pulled back locally
  (`--file-pattern` restricted to `*.log`), never any `.avi` files or the CSVs
  themselves. The dataset's real schema was read from the kernel's printed
  output, not from a locally-downloaded copy.
- **Class-to-label-index mapping built from a sorted, deduplicated class list**
  (alphabetical, deterministic) rather than trusting row order in the CSV or
  any implicit ordering — makes label ids reproducible and independent of which
  split the index happens to be built from.

### What exists after this step

- `kaggle_kernel/ucf101_explore/` — a CPU-only, no-GPU, no-internet script
  kernel (`kernel_type: script`, avoiding the jupytext kernelspec bug class
  entirely) that inspects the real dataset structure. Its output log is kept
  under `kaggle_kernel/ucf101_explore/output/` as a record of the verified
  layout.
- `src/data/ucf101_metadata.py` — parses the real CSV format into a
  `VideoSample` dataclass (clip_name, relative_path, class_name, label), with
  a shared class-index builder so label ids stay consistent across
  train/test/val, plus a path-resolution helper. Pure stdlib `csv`, no pandas
  dependency needed for something this simple.
- `configs/dataset/ucf101.yaml` updated with the real, verified `root_dir`.
- 7 new unit tests (`tests/test_ucf101_metadata.py`) against synthetic CSV
  fixtures mirroring the real schema exactly — 27 tests passing total, still
  zero real data touched locally.

### Next step (from Phase 1)

Write the actual video-reading `Dataset` class (frame decoding via
decord/opencv + the existing `sample_frame_indices` sampling logic) — this one
can only be meaningfully exercised against real video files, so its correctness
will be verified inside a Kaggle kernel rather than locally; keep the frame-index
math itself (already tested) decoupled from the decoding call so the local test
suite doesn't need real video I/O. Then implement Baseline 1 (pretrained
CNN features + temporal average pooling + MLP) as the first trainable model.

---

## Phase 2 — GitHub repo live, Baseline 1 architecture (2026-08-22)

### Decisions made and why

- **Public GitHub repo created**: `karthik-1604/temporal-video-understanding`.
  Generic name (matches the README's resume-positioning title), no target-company
  name anywhere, repo-local git identity used for the push (real email, not the
  machine's global placeholder).
- **Built Baseline 1's architecture before the video-reading `Dataset` class**,
  reordering the plan from the Phase 1 next-step note — the model and the data
  pipeline are independent enough to build in either order, and getting the
  architecture's shapes/gradient-flow verified locally first means the eventual
  Kaggle training run only has to debug data plumbing, not both model and data
  at once.
- **Local torch install follows the exact pattern already validated in a prior
  project** (`causal-ott-user-modeling`): `pip install torch torchvision
  --index-url https://download.pytorch.org/whl/cpu`, confirmed via
  `torch.cuda.is_available() == False` and version string ending `+cpu`. That
  project's own journal explicitly frames this as "light packages locally... 
  heavy packages (torch+GPU) assumed present in the Kaggle notebook environment,
  not installed locally beyond a CPU build for quick smoke tests" — same
  reasoning applies here (no local GPU either).
- **All local model tests use `pretrained=False`** — real ImageNet-pretrained
  weights are only downloaded inside a Kaggle kernel (where training actually
  happens), not to the local machine. Local tests check wiring only: output
  shape, gradient flow to the classifier head, backbone freeze/unfreeze
  behavior, invalid-input handling — random weights are sufficient for all of
  that.
- **Switched `requirements-dev.txt` (a raw `pip freeze`) to a hand-authored
  `requirements.txt`**, matching the same prior project's convention — a
  `pip freeze` output of a `+cpu` torch build isn't reinstallable via a plain
  `pip install -r` (those wheels aren't on PyPI), so the file needs the
  `--index-url` documented inline instead of relying on a frozen version pin.

### What exists after this step

- `src/models/baseline_frame_pool.py` — `FramePoolClassifier`: pretrained
  backbone (resnet18/resnet34/efficientnet_b0, configurable) → per-frame
  features → temporal average pool → MLP head. Optional backbone freezing.
- 8 new unit tests (`tests/test_baseline_frame_pool.py`) — output shape, invalid
  backbone/input rejection, freeze/unfreeze behavior, gradient flow, and an
  EfficientNet variant — all against random tensors, 35 tests passing total.
- `requirements.txt` (local, light deps only + the torch CPU index note).
- Repo pushed to GitHub for the first time (2 commits at push time).

### Next step (from Phase 2)

Write the video-reading `Dataset` class and the actual Kaggle training kernel
for Baseline 1, using the real dataset ref and root_dir already confirmed in
Phase 1. Check GPU quota before committing to the first real training run.

---

## Phase 3 — Video-clip Dataset, decoupled from real decoding (2026-08-22)

### Decisions made and why

- **Frame decoding injected as a `clip_reader` callable** rather than hardcoded
  into the `Dataset` class — `src/data/video_dataset.py`'s `VideoClipDataset`
  takes any `(video_path, num_frames, strategy, seed) -> frames` function.
  The real implementation (`src/data/video_io.py`, `decord`-based, lazy-imported
  so the module stays importable without `decord` installed) only runs inside a
  Kaggle kernel against real video files; local tests inject a fake reader that
  returns fixed-shape zero arrays instead. This is the same
  "decouple pure logic from the thing that needs real data" pattern as
  Phase 0's sampling module and Phase 1's metadata parser — it's what let this
  whole data-pipeline layer get built and tested (12 more tests, 41 total) with
  zero video files or decoding libraries on the local machine.
- Caught and fixed a genuinely non-obvious test bug while writing this (not a
  `src/` bug): `numpy_array.sum()` on a `uint8` array can return a numpy scalar
  type `torch.tensor(...)` rejects with `TypeError: an integer is required` —
  fixed by wrapping in `int(...)` in the test's custom-transform fixture.

### What exists after this step

- `src/data/video_dataset.py` — `VideoClipDataset(torch.utils.data.Dataset)`,
  `default_to_tensor` (uint8 (T,H,W,C) → float32 (T,C,H,W) in [0,1]).
- `src/data/video_io.py` — `decord_clip_reader`, the real decoder (Kaggle-only).
- Fixed the "AI research role" phrasing in `journey.md`/`README.md` to
  correctly say "data science / AI internship" — this is a data science
  internship posting, not a pure research-scientist role; the earlier generic
  phrasing overcorrected.

### Next step

Write the Kaggle training kernel for Baseline 1: attach the UCF101 dataset,
build train/val `VideoClipDataset`s with the real `decord_clip_reader`, run a
short training loop (frozen backbone, MLP head only, cross-entropy). Check GPU
quota first; this is the first kernel that touches real video decoding at
scale, so also verify wall-clock/throughput on a small subset before committing
to a full-dataset run.

**Correction, same day:** the README's opening framed the project as "built for
a data science / AI internship application" — reworked to instead open with
the actual research question (how much does temporal modeling help vs.
frame-level features) and drop the application framing entirely. A README
should describe the problem being solved, not the reason the author built it;
`journey.md` is where the real motivation belongs.

GPU quota checked before starting any training work: 29.62h / 30h remaining
(resets 2026-08-29).

---

## Phase 4 — Resize/normalize transform (2026-08-22)

### What exists after this step

- `src/preprocessing/transforms.py` — `resize_and_normalize`: (T,H,W,C) uint8 →
  (T,C,image_size,image_size) float32, ImageNet-normalized. Needed because real
  UCF101 frames aren't natively 224×224. Pure tensor ops (bilinear
  interpolation), so fully unit-tested locally (5 new tests, 46 total) against
  synthetic frames of arbitrary/non-square resolutions — no real video needed.

### Next step (from Phase 4)

Write the Kaggle training kernel for Baseline 1, composing everything built so
far: `ucf101_metadata` → `VideoClipDataset` (with the real `decord_clip_reader`
and `resize_and_normalize` as its transform) → `FramePoolClassifier`. Follow the
`git clone` pattern from the prior project (clone this public repo into the
kernel's working directory, `sys.path.insert` it) so the kernel can import
`src.*` directly instead of duplicating code into the kernel script.

---

## Phase 5 — Baseline 1 training kernel, real GPU run (2026-08-22)

### Decisions made and why

- **Smoke test before the full run**: a small CPU/GPU dry run
  (`kaggle_kernel/baseline1_smoketest/`, 128 clips) validated the whole
  pipeline end to end and measured real throughput before committing to a
  full-dataset run — caught the P100 issue immediately and cheaply rather than
  discovering it partway through a full training run.
- **Hit the exact documented P100/sm_60 issue on the first smoke-test attempt**
  (`torch.AcceleratorError: no kernel image is available`) — same root cause
  already logged in `KAGGLE_WORKFLOW_NOTES.md` from the prior project. Applied
  the known fix (reinstall `torch==2.5.1`/`torchvision==0.20.1` from the
  `cu121` index before any `import torch`), but **conditionally**, only after
  detecting a P100 via `nvidia-smi` — not applied preemptively on every GPU
  kernel, per that same note's later correction (a blind pin risks its own new
  conflicts on a GPU that wouldn't have needed it). Re-run succeeded: GPU
  verified via a real matmul + `.device` check, not just `is_available()`.
- **Real smoke-test numbers drove the training design**: 15.14 clips/sec
  running the ResNet18 backbone forward pass on the assigned P100, ~913 MB peak
  GPU memory, ~11 min estimated for one full-train-set backbone pass. Since the
  backbone is frozen for Baseline 1, re-running it every epoch would waste
  quota for no benefit — instead, embeddings are extracted **once** per split
  and the small MLP head is trained on cached features (near-instant per
  epoch). This is the same "frame caching" idea the plan lists as an optional
  data-engineering demonstration, arrived at here for a concrete efficiency
  reason rather than added just to check a box.
- **Confirmed the documented "ERROR pulls unfiltered kernel output" gotcha
  again, this time via the regex file_pattern rather than a loose grep**: even
  `--file-pattern ".*\.log"` matched a `.log` file that was part of the
  kernel's own `git clone`d copy of this repo (the *committed*
  `ucf101_explore` log), not just kernel-generated output — pulled and
  discarded. Fixed for the next pull by anchoring the pattern to exclude
  `repo/`: `"^(?!repo/).*\.log"`.

### What exists after this step

- `src/models/clip_zero_shot.py` — Baseline 3 (CLIP zero-shot, the JD's
  LLM/VLM gap-closer): pure-logic prompt construction
  (`camel_case_to_words`/`class_name_to_prompt`) and cosine-similarity
  classification (`cosine_classify`), both unit-tested locally with no real
  CLIP model (9 new tests, 57 total). The real `open_clip`-backed
  `CLIPZeroShotClassifier` is lazy-imported, Kaggle-only — same pattern as
  `decord_clip_reader`.
- `kaggle_kernel/baseline1_smoketest/` — validated pipeline + real throughput
  numbers (kept as a record).
- `kaggle_kernel/baseline1_train/train_baseline1.py` — full Baseline 1
  training: embedding extraction (train/val/test), MLP head training with
  best-val-acc checkpointing, test-set evaluation (top-1/top-5 accuracy, macro
  F1, per-class report, confusion matrix), inference latency/throughput at
  batch sizes 1 and 8, param counts, peak GPU memory — all saved to
  `results/baseline1_metrics.json` + a small classifier-head checkpoint,
  nothing else pulled back locally.

### Next step (from Phase 5)

Pull and record the actual Baseline 1 results once the training kernel
completes; then decide whether Experiment A's 8-vs-16-frame comparison reuses
the same cached-embeddings design (would need a second embedding-extraction
pass at 8 frames) before moving to Baseline 2 (CNN+LSTM/GRU).

---

## Phase 6 — Baseline 1 real results (2026-08-22)

### Results

Trained on the real UCF101 split (10,055 train / 1,673 val / 1,723 test, 101
classes), frozen ResNet18 backbone + MLP head, 40 epochs on cached embeddings:

- **Test top-1 accuracy: 96.11%, top-5: 99.65%, macro F1: 96.52%**
- 11.33M total params, only 157,285 trainable (MLP head only)
- Inference latency: 8.11 ms/clip (batch 1, 123.3 clips/sec), 51.1 ms/batch at
  batch 8 (156.4 clips/sec effective throughput)
- Peak GPU memory: 1.77 GB
- Real embedding-extraction throughput: 20.1-20.7 clips/sec across
  train/val/test (vs. the smoke test's 15.14 clips/sec estimate on a 128-clip
  subset — the full run benefited from `num_workers=4` vs. the smoke test's 2,
  plus a warmed-up decode pipeline)
- Total kernel wall-clock: ~11 min for embedding extraction across all three
  splits, negligible additional time for 40 epochs of MLP training on cached
  features (exactly the payoff the cached-embeddings design was chosen for)

### Investigated finding: the two worst classes are confused with each other, in both directions

Per-class F1 ranged from 1.00 (five classes, including `YoYo`, `Typing`) down
to a clear floor at **`BasketballDunk` (F1 0.489) and `Basketball` (F1 0.586)**
— the two worst classes by a wide margin (next-worst is `Rafting` at 0.846).
Checked the actual confusion matrix rather than treating this as generic
"some classes are just harder": **6/17 `BasketballDunk` clips predicted as
`Basketball`, and 17/34 `Basketball` clips (exactly half) predicted as
`BasketballDunk`** — a genuine bidirectional confusion between exactly these
two classes, not spread across many classes.

This makes sense given the model architecture, not just "hard classes": both
actions share the same visual scene (basketball court, hoop, players), so the
only real distinguishing signal is the *temporal* dunking motion — which
**temporal average pooling destroys by construction**. This is precisely the
project's core research question in miniature (does temporal modeling help
over frame-level features?), so this exact class pair is a natural,
non-cherry-picked test case for whether Baseline 2 (CNN+LSTM) actually
resolves it.

### What exists after this step

- `reports/baseline1_metrics.json` — full results (history, per-class report,
  confusion matrix, latency, params) committed for the record.
- `kaggle_kernel/baseline1_train/output/` — the kernel's log kept alongside
  the training script, matching the exploration kernel's precedent.
- Trained classifier-head checkpoint (631 KB) kept **local only**
  (`*.pt` gitignored) — small enough to not need cloud storage, but binary
  model weights don't belong in git history; regenerable from the kernel in
  ~11 min if needed again.
- `.gitignore` also now excludes `/.claude/` (harness state files like
  `scheduled_tasks.lock`, not project content).

### Next step (from Phase 6)

Baseline 2 (CNN + LSTM/GRU) — the natural next comparison, and the one most
likely to directly test the Basketball/BasketballDunk finding above. Reuse the
same cached-embeddings idea, but keep *per-frame* embeddings (not pre-averaged)
so the LSTM/GRU has the sequence to consume, rather than reusing Baseline 1's
already-pooled vectors.

---

## Phase 7 — Baseline 2 (CNN+RNN) architecture (2026-08-22)

### Decisions made and why

- **Extracted `build_backbone` out of `baseline_frame_pool.py` into a shared
  `src/models/backbones.py`** before writing Baseline 2 — it was a private
  (`_build_backbone`) helper used by exactly one model; Baseline 2 needs the
  identical backbone-loading logic (same supported names, same "strip the
  classification head" behavior) so duplicating it would risk the two models
  silently drifting (e.g. one gaining a new backbone option the other doesn't
  support). Refactored first, confirmed Baseline 1's 8 tests still pass
  unchanged, then built Baseline 2 on top of the shared version.
- **`CNNRNNClassifier` exposes `forward_from_features`** (skips the backbone,
  takes already-extracted `(batch, num_frames, feature_dim)` tensors) as a
  first-class method rather than duplicating the RNN+classifier logic in the
  training kernel — same efficiency idea as Baseline 1 (frozen backbone run
  once, not every epoch), but this time keeping the *per-frame* sequence
  instead of pre-averaging, since the whole point of Baseline 2 is giving the
  RNN the ordered sequence Baseline 1's average pooling threw away.
- **Added a frame-order-sensitivity test as the key architectural check**
  (`test_sensitive_to_frame_order`: reversing frame order must change the
  prediction) — this is the one property that actually distinguishes Baseline
  2 from Baseline 1 architecturally, so it's worth testing explicitly rather
  than just checking output shapes.

### What exists after this step

- `src/models/backbones.py` — shared `build_backbone`, 3 new tests.
- `src/models/baseline_cnn_rnn.py` — `CNNRNNClassifier`: backbone (shared,
  optionally frozen) → LSTM or GRU (configurable layers/hidden size/
  bidirectional) → dropout + linear classifier on the final timestep. 12 new
  tests (order-sensitivity, LSTM/GRU, bidirectional, multi-layer, freeze
  behavior, gradient flow, `forward_from_features` parity) — 71 tests passing
  total, all still against random tensors/synthetic data.

### Next step (from Phase 7)

Write the Kaggle training kernel for Baseline 2: extract *per-frame* (not
averaged) embeddings once per split (~330 MB for the train split at 16 frames
× 512-dim float32 — checked it fits comfortably in Kaggle's RAM/GPU memory
before committing to this design), then train the LSTM/GRU + classifier on
the cached sequences. Compare directly against Baseline 1's 96.11% top-1 and
specifically check whether the Basketball/BasketballDunk confusion improves.

---

## Phase 8 — Baseline 2 real results: a genuine negative result, investigated (2026-08-22)

### v1 result: both RNN variants underperformed Baseline 1

Trained both LSTM and GRU heads on identical cached per-frame embeddings
(same 10,055/1,673/1,723 split, 40 epochs):

| Model | Test top-1 | Test macro F1 |
|---|---|---|
| Baseline 1 (avg pool) | 96.11% | 96.52% |
| Baseline 2 LSTM | 92.92% | 92.69% |
| Baseline 2 GRU (best variant) | 95.01% | 95.24% |

Neither beat Baseline 1 — the opposite of the naive expectation that temporal
modeling should help. Checked the *specific* case this baseline was meant to
test (Phase 6's Basketball/BasketballDunk confusion) rather than just noting
the aggregate number moved the wrong way: it got **worse**, not better.
LSTM predicted **zero** of the 17 `BasketballDunk` test clips correctly (all
17 went to `Basketball`); GRU got 1/17. Baseline 1 had gotten 11/17 correct.

**Investigated via the actual training curves, not accepted at face value**:
both variants showed real overfitting (train acc 97-99% vs. val acc
92-94% — a much wider gap than Baseline 1's own train/val gap). GRU's last few
epochs additionally showed a real instability spike: val_acc dropped from
93.7% (epoch 35) to 83.8% (epoch 39) alongside a loss spike from 0.23 to 0.55.
Identified three concrete, non-hand-wavy contributing factors:

1. **No gradient clipping**, despite RNN training's well-known susceptibility
   to occasional gradient spikes — directly consistent with the observed
   instability spike.
2. **`num_layers=1` silently disabled PyTorch's internal recurrent dropout**
   (`nn.LSTM`/`nn.GRU`'s `dropout` argument only applies *between* stacked
   layers, a real PyTorch gotcha, not obvious from the constructor signature)
   — so the only regularization was the final classifier's dropout, despite
   the RNN having 4-5x more trainable parameters (614K-814K) than Baseline
   1's head (157K).
3. **Reused Baseline 1's learning rate (1e-3) unchanged** for a differently
   shaped loss landscape (recurrent vs. a simple 2-layer MLP) rather than
   retuning it.

### Fix and re-run

Kept the identical cached embeddings and evaluation protocol, changed only:
`num_layers=2` (activates real internal dropout), gradient clipping
(`max_norm=2.0`), and halved learning rate (5e-4). Re-run in progress —
results recorded below once complete. v1's full results kept for the record
at `reports/baseline2_metrics_v1_overfit.json` regardless of outcome, since
the before/after comparison is itself the more valuable artifact than either
run alone.

### v2 result: training is more stable, but the core finding didn't change — and that's the actual answer

| Model | v1 test top-1 | v2 test top-1 | v1 BasketballDunk correct | v2 BasketballDunk correct |
|---|---|---|---|---|
| LSTM | 92.92% | 92.69% | 0/17 | 0/17 |
| GRU | 95.01% | 93.67% | 1/17 | 0/17 |

The fix worked exactly as targeted — no more instability spikes (GRU's worst
mid-training dip is now ~0.89-0.93, not the v1 crash to 0.838), and the
train/val gap is somewhat tighter. But **test accuracy did not improve** (GRU
is actually 1.3 points lower), and **`BasketballDunk` still goes to 0/17
correct in both variants**, identically to before. Ruling out "the training
process was broken" as a sufficient explanation (it demonstrably wasn't,
across two different regularization/LR regimes) points at a different, more
fundamental bottleneck: **the frozen ResNet18 backbone is ImageNet-pretrained
for static image classification, not motion.** Consecutive frame embeddings
from a scene with only fine-grained motion difference (a dunk vs. dribbling,
same court, same players, same camera framing) may simply be nearly identical
in that feature space — in which case no amount of RNN capacity or tuning can
recover a distinction that was never encoded in the per-frame features the
RNN receives as input. Average pooling did comparatively better on this pair
not because it's a better *temporal* model (it has no temporal awareness at
all) but because BasketballDunk apparently has *some* distinguishing static
visual signal (different camera angle/framing near the hoop) that pooling
picks up and the RNN's harder optimization problem loses.

This is left as a well-reasoned hypothesis, not confirmed further this
session — the natural test would be unfreezing/fine-tuning the backbone (so
gradients could shape features toward motion-relevant information) or moving
to a genuinely video-native architecture (Model 4, Phase 2), rather than
continuing to tune the RNN on top of a fixed image backbone.

### What exists after this step

- `kaggle_kernel/baseline2_train/train_baseline2.py` — reflects v2 (the fixed
  version actually used for the final reported numbers); v1's code is
  preserved in git history for anyone reconstructing the before/after.
- `reports/baseline2_metrics.json` — v2 (final) full results.
- `reports/baseline2_metrics_v1_overfit.json` — v1 results, kept for the
  before/after record.

### Next step (from Phase 8)

Baseline 3 (CLIP zero-shot) — architecture already built (Phase 5); write and
run its Kaggle kernel. Also worth specifically checking whether CLIP's
image-text embedding space handles the Basketball/BasketballDunk pair any
differently, given it's a genuinely different representation than
ImageNet-supervised ResNet features.

---

## Phase 9 — Baseline 3 (CLIP zero-shot) real results (2026-08-22)

### Results

`ViT-B-32` (OpenAI weights), zero-shot on the same 1,723-clip test split, no
UCF101 training data used at all:

| Metric | Value |
|---|---|
| Top-1 accuracy | 65.24% |
| Top-5 accuracy | 89.20% |
| Macro F1 | 60.45% |
| Params | 151.28M (all frozen, zero-shot) |
| Latency (GPU compute only) | 27.65 ms/clip |
| Eval throughput (incl. decode) | 8.55 clips/sec |
| Peak GPU memory | 778 MB |

65% top-1 on a 101-way task with **zero labeled training examples** is a
strong result in absolute terms (in line with published CLIP zero-shot
numbers on UCF101), just well below the trained baselines (96.1% / 92.7% /
93.7%) — expected, since it has no access to this dataset's label
distribution or visual idiosyncrasies at all.

### Investigated: does CLIP also fail on Basketball/BasketballDunk? Yes — but the reason is more nuanced than it first looks

**CLIP zero-shot also got 0/17 `BasketballDunk` correct** (all 17 → `Basketball`)
— the same complete failure as both LSTM and GRU. First read: three
architecturally distinct models (supervised-pooled, recurrent, contrastive
zero-shot) all failing on the identical pair looks like strong triangulating
evidence for the Phase 8 hypothesis (frame-level features don't encode the
motion that distinguishes them).

**Checked further before accepting that reading**: `BasketballDunk` is not an
isolated CLIP failure — **11 of 101 classes get F1 = 0.0** for CLIP zero-shot
(`BalanceBeam`, `BasketballDunk`, `BreastStroke`, `JumpingJack`, `Lunges`,
`ParallelBars`, `PlayingDaf`, `Punch`, `Shotput`, `StillRings`, `YoYo`).
Notably `YoYo` — one of Baseline 1's *perfect* (F1 = 1.0) classes — is in this
list, and several others (`PlayingDaf`, an uncommon percussion instrument;
generic single-word actions like `Punch`/`Lunges`) look more like the single
naive prompt template (`"a video of a person {class}"`) failing to produce a
useful CLIP text embedding for that class at all, than motion-blindness
specifically. **This is a broad zero-shot prompt-coverage gap, not a targeted
confirmation of the Phase 8 hypothesis** — CLIP's `BasketballDunk` failure is
*consistent with* that hypothesis but doesn't independently confirm it, since
the same failure mode (F1 = 0) hits classes with no plausible motion-blindness
story at all.

Honest combined picture: Baseline 1's partial success on `BasketballDunk`
(11/17, via a classifier *trained* on this dataset's labels) is most likely
explained by that trained classifier exploiting some correlated static cue
(e.g. camera framing near the hoop) specific to this dataset — not by
average-pooled features somehow capturing motion, which they structurally
cannot. The RNNs, given the same underlying per-frame features but a harder
optimization problem and fewer effective examples reaching the recurrent
path, didn't find/exploit that same shortcut. CLIP, with no training signal
on this dataset at all, has no opportunity to find it either. None of this
constitutes evidence that any of these models genuinely understand the
dunking motion — that remains an open, well-motivated question for Model 4
(a real video-native architecture) in Phase 2, not resolved by the MVP
baselines.

### What exists after this step

- `kaggle_kernel/baseline3_clip_zeroshot/` — kernel + log.
- `reports/baseline3_metrics.json` — full results.
- MVP baseline set complete: 3 baselines, all with real results on the
  identical test split, differences investigated rather than just reported.

### Next step (from Phase 9)

Annotation analysis (class distribution, duration/frame-count stats — the
confusion matrices already exist from all 3 baselines) and the Explainability
phase (Grad-CAM + the showcase GIF). Both are now data-analysis/visualization
work on already-collected results, not new training runs.

---

## Phase 10 — Annotation analysis kernel + Grad-CAM module (2026-08-22)

### Decisions made and why

- **Annotation analysis kernel is CPU-only**: it only needs `decord.VideoReader`
  to read each video's frame count/fps (header/index only, no frame decode),
  so there's no GPU work at all — matches the project's own "does this need a
  GPU at all?" rule from the Kaggle workflow notes.
- **Class-distribution/duration stats need a Kaggle run** (the real CSVs and
  videos aren't local), but **per-class difficulty and confusion matrices
  don't** — those already exist in the three `reports/baselineN_metrics.json`
  files pulled back earlier. Split the annotation-analysis work accordingly:
  new Kaggle kernel only for what isn't already available locally.
- **Grad-CAM built and unit-tested locally first**, same pattern as every
  other model component: the hook mechanics (activation/gradient capture,
  channel-weighting, normalization, shape) don't depend on real images at all
  — a `pretrained=False` model with random input and `requires_grad_(True)`
  exercises the exact same autograd graph structure as a real trained model.
  Real semantics (does the heatmap highlight something meaningful) can only be
  checked against real frames, which happens in the Kaggle explainability
  kernel next.

### What exists after this step

- `kaggle_kernel/annotation_analysis/` — CPU-only kernel scanning all
  ~13,451 videos for class distribution + duration/frame-count/fps stats.
- `src/explainability/gradcam.py` — `GradCAM`: hooks a backbone's target layer
  (default `layer4`, works for resnet18/34), backprops the target (or
  predicted) class's logit, produces one normalized heatmap per frame. 9 new
  unit tests (82 total) — shape, value range, target-class selection, frozen/
  unfrozen backbone, invalid-input handling — all against random tensors.

### Next step (from Phase 10)

Pull the annotation-analysis kernel's results once complete; then build the
Kaggle explainability kernel that retrains Baseline 1's head (fast, ~11 min,
avoids managing a separate checkpoint artifact), runs `GradCAM` against a
handful of real examples (including the Basketball/BasketballDunk pair and a
perfect-F1 class like `YoYo`), overlays heatmaps on frames, and assembles a
small annotated GIF for the README alongside static example images.

**Operational note**: the annotation-analysis kernel appeared stuck (still
`RUNNING` after ~50 min of polling, far longer than the ~11-15 min full-decode
training kernels took for similarly-sized scans) and was about to be killed
and rewritten with sampling + parallelism. Checked the actual Kaggle web
dashboard directly first (at the user's prompt) rather than trusting the CLI
polling loop's read alone — the real per-kernel execution log showed a
perfectly healthy ~30 videos/sec scan, on track to finish in ~7-8 minutes; it
completed normally shortly after. The CLI's `kernels status`/`kernels logs`
(without `-f`) gave no visibility into actual progress during the run, only a
terminal-state check — worth remembering before assuming "no status change in
N minutes" means stuck, especially for a kernel design (no periodic status
transitions) that only prints to its own log, not to the CLI-visible status.

---

## Phase 11 — Full annotation analysis (2026-08-22)

### Results

Scanned all 13,451 real videos (zero read errors) via `decord.VideoReader`
header/index reads (no frame decoding, CPU-only, ~35 videos/sec):

| Stat | Value |
|---|---|
| Duration (mean / median) | 7.16s / 6.41s |
| Duration (min / max) | 1.07s / 71.04s |
| Duration (p10 / p90) | 3.20s / 12.00s |
| Frame count (mean / median) | 185.4 / 166.0 |
| Frame count (min / max) | 29 / 1,776 |
| FPS | bimodal: 25.0 or 29.97 (two source frame rates, expected for web-sourced UCF101) |

### Investigated finding: class imbalance is the more parsimonious explanation for the Basketball/BasketballDunk result

**`Basketball` has 198 training examples — exactly 2x `BasketballDunk`'s 98 —
and is the single largest class in the entire dataset** (next-largest is
`CricketShot` at 125; median class size is 98, mean 99.6; smallest classes
sit at 75). This wasn't visible when the confusion was first found (Phase 6),
since that only used the *test*-split confusion matrix, not train-split class
counts.

This is a more parsimonious explanation than either earlier hypothesis
(Phase 8's "frozen backbone lacks motion," Phase 9's partial CLIP evidence)
for the *direction* of the bias specifically — a classifier trained with
plain unweighted cross-entropy on 2x more `Basketball` examples than
`BasketballDunk` examples will naturally lean toward predicting the majority
class on any genuinely ambiguous input, independent of whether the underlying
features encode motion at all. It doesn't replace those hypotheses (CLIP's
zero-shot failure, with *no* class-count exposure during training at all,
still needs a separate explanation — most likely the prompt-template gap
already identified in Phase 9), but it's likely the dominant factor for
Baselines 1 and 2's specific bias *direction* toward `Basketball`.

This directly motivates **Experiment D (class-weighted loss / focal loss,
already planned as Phase 2 bias-mitigation work)** as a concrete, targeted
test — re-run Baseline 1 or 2 with class-weighted cross-entropy and check
specifically whether the `BasketballDunk` recall improves, rather than only
reporting an aggregate macro-F1 change.

### What exists after this step

- `reports/annotation_analysis.json` — full class distribution (all
  splits/classes), duration/frame-count/fps stats, per-class averages.
- `kaggle_kernel/annotation_analysis/output/` — kernel log kept for the
  record.

### Next step

Build the Kaggle explainability kernel (Grad-CAM + showcase GIF, per the plan
above) — the last MVP item. The class-imbalance finding above is Phase-2
follow-up (Experiment D), not blocking the MVP.
