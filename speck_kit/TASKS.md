# TASKS — 3D Log Bark Segmentation

> Spec-driven backlog synced to the current repository state.
> This file is also intended to be a reusable example of how to track architecture work.

Status legend:
- `[x]` complete
- `[ ]` pending
- `[~]` in progress

---

## Phase 1 — Foundation [x]

- [x] Project structure, setup, and editable install
- [x] YAML config loader with dot access
- [x] Pure-NumPy PLY loader
- [x] Dataset with sampling, normalization, caching, and augmentation
- [x] PointNet++ segmentation model
- [x] Training loop with logging and checkpoints
- [x] Inference pipeline from checkpoint
- [x] Metrics, visualizer, and logger utilities
- [x] CLI entry point
- [x] Synthetic unit tests

---

## Phase 2 — Repository Hardening [x]

- [x] Fix failing test fixtures and restore green baseline
- [x] Verify `pytest tests/ -v` passes with the current suite
- [x] Prevent augmentation leakage into validation
- [x] Add and validate `config/smoke.yaml`
- [x] Validate `main.py info`, training smoke test, and inference path
- [x] Update Open3D pin to `0.19.0`
- [x] Fix Dev Container dependency installation path
- [x] Refresh README to reflect actual repository behavior

---

## Phase 3 — Analysis Tooling [~]

- [~] Validate `compare_*.py` scripts inside the intended runtime environment
- [ ] Consolidate overlapping comparison scripts if their outputs can be unified
- [ ] Decide whether compare tooling belongs in root or a dedicated `analysis/` folder

---

## Phase 4 — Model and Data Improvements [~]

- [x] **TASK-20**: Implement `preprocessing.preprocess_dataset()` for offline PLY caching (.npy)
  - ✅ Purpose: Load all PLY, normalize, optional downsample, cache for faster training
  - ✅ Integration: Eliminates repeated I/O and normalization per epoch
  - ✅ Implementation: `preprocess_dataset(cfg, overwrite=False)` fully working
  - ✅ Usage: `python main.py preprocess --overwrite` creates cache, then set `use_cache: true` in config
  - ✅ Tests: 7 new tests pass, 17 original tests still pass (no regressions)
  - 📊 Result: 31 PLY files cached in data/processed/ (62 .npy + metadata files)
- [ ] Add exact mesh-area computation instead of approximation-only inference
- [ ] Add batch inference with CSV summary output
- [ ] Add k-fold cross-validation once dataset size justifies it
- [ ] Evaluate UV features as an optional input branch
- [ ] Add experiment presets for geometry-only vs geometry+RGB training
- [ ] Benchmark alternative architectures against the current PointNet++ baseline

---

## Phase 5 — Template Quality for Future Architectures [ ]

- [ ] Extract reusable patterns from this repo into a lighter starter scaffold
- [ ] Define a standard checkpoint schema shared across future projects
- [ ] Define a standard smoke-test policy for new architectures
- [ ] Decide which documents are source-of-truth versus historical notes

---

## Review Prompts

Useful prompts to reuse with an assistant:

```text
Implement a new architecture following SPEC.md and CONSTITUTION.md.
Keep checkpoint compatibility with the current predictor contract.
Do not break smoke validation or unit tests.
```

```text
Review a training change against the current validation policy.
Check that validation uses a non-augmented dataset instance.
```

```text
Audit a new feature branch and identify where the docs drift from the code.
Treat speck_kit as reusable reference material, not aspirational fiction.
```