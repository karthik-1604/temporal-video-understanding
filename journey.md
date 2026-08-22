# Journey

Running build log for this project. Updated at the end of every phase/session.

---

## Phase 0 — Scoping and repo setup (2026-08-22)

### Decisions made and why

- **Problem**: temporal video understanding / action recognition — comparing how
  much temporal modeling improves accuracy over frame-level image features, plus
  a research-experiment suite (temporal resolution, temporal modeling
  architecture, augmentation, class imbalance, robustness), annotation analysis,
  explainability, and a small unsupervised component. Chosen to match an AI
  research role focused on video understanding and predictive modeling, not
  because it was the only reasonable option — an initial draft plan was reviewed
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

### Next step

Write the actual video-reading `Dataset` class (frame decoding via
decord/opencv + the existing `sample_frame_indices` sampling logic) — this one
can only be meaningfully exercised against real video files, so its correctness
will be verified inside a Kaggle kernel rather than locally; keep the frame-index
math itself (already tested) decoupled from the decoding call so the local test
suite doesn't need real video I/O. Then implement Baseline 1 (pretrained
CNN features + temporal average pooling + MLP) as the first trainable model.
