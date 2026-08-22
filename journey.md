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

### Next step

Build the actual UCF101 dataset loader (metadata parsing, class list, split
files) designed to run inside a Kaggle kernel against the real download, while
keeping the loader's pure logic (path resolution, label mapping) unit-testable
locally against a synthetic fake directory tree. Then stand up the first Kaggle
kernel (CPU-only, no GPU needed yet) to fetch and inspect the real UCF101
metadata before writing any baseline model code.
