# TASKS — 3D Log Bark Segmentation

> Spec-Driven Development task list.
> Each task references SPEC sections and CONSTITUTION principles.
> Status: [ ] = pending, [x] = complete, [~] = in progress

---

## Phase 1 — Foundation ✅ COMPLETE

- [x] **TASK-01** · Project structure and DevContainer
  - `.devcontainer/devcontainer.json` + `Dockerfile` (Python 3.10-slim)
  - `requirements.txt` with pinned versions
  - `setup.py` for editable install
  - *Ref: CONSTITUTION §2.2 Reproducibility*

- [x] **TASK-02** · Configuration system
  - `config/default.yaml` — all hyperparameters in one place
  - `config/config_loader.py` — YAML → SimpleNamespace with dot access
  - *Ref: SPEC-07 Configuration Reference*

- [x] **TASK-03** · PLY loader
  - `data/loader.py` — `load_ply_labeled()` without external dependencies
  - Parses binary PLY, extracts label_1 float, computes vertex normals from faces
  - Handles unlabeled files gracefully (n_bark = 0)
  - *Ref: SPEC-01*

- [x] **TASK-04** · PyTorch Dataset
  - `data/dataset.py` — `BarkDataset` loads PLY directly, caches in RAM
  - Fixed-size sampling, normalization, geometric augmentation
  - `get_class_weights()` for imbalanced data
  - `summary()` for dataset inspection
  - *Ref: SPEC-02*

- [x] **TASK-05** · PointNet++ architecture
  - `model/pointnet2.py` — full encoder-decoder for segmentation
  - SetAbstraction × 3 + FeaturePropagation × 3
  - FocalLoss with per-class alpha weights
  - CPU-compatible (no CUDA-only ops)
  - *Ref: SPEC-03*

- [x] **TASK-06** · Training loop
  - `training/trainer.py` — full loop with validation, checkpointing, resume
  - Checkpoint format as specified in SPEC-04
  - StepLR scheduler, gradient clipping
  - *Ref: SPEC-04*

- [x] **TASK-07** · Inference pipeline
  - `inference/predictor.py` — `BarkPredictor.from_checkpoint()`
  - Reconstructs model from checkpoint config (no yaml needed)
  - Outputs colored .ply + bark area summary
  - *Ref: SPEC-05*

- [x] **TASK-08** · Metrics and utilities
  - `utils/metrics.py` — IoU, mIoU, F1, bark area estimation
  - `utils/logger.py` — CSV logger per epoch
  - `utils/visualizer.py` — Open3D colored point cloud
  - *Ref: SPEC-06*

- [x] **TASK-09** · Entry point
  - `main.py` — `info / train / infer / visualize / preprocess`
  - *Ref: SPEC-07*

- [x] **TASK-10** · Unit tests
  - `tests/test_preprocessing.py` — synthetic PLY fixtures
  - Tests for loader, normalization, dataset, model, metrics
  - *Ref: CONSTITUTION §5 Quality Standards*

---

## Phase 2 — Validation & Robustness [ ] PENDING

- [ ] **TASK-11** · Run `pytest tests/ -v` and fix any failures
  - Expected: 15+ tests passing
  - *Ref: CONSTITUTION §5 Testing*

- [ ] **TASK-12** · Run `python main.py info` with real data
  - Verify all 19 PLY files are detected
  - Verify labeled vs unlabeled count is correct
  - Fix any PLY parsing issues for new file formats
  - *Ref: SPEC-07 info command*

- [ ] **TASK-13** · First training run (smoke test)
  - Run `python main.py train` with `epochs: 3` in config
  - Verify loss decreases, checkpoint is saved
  - Verify `train_log.csv` is created correctly
  - Revert epochs to 100 after smoke test

- [ ] **TASK-14** · Verify inference on labeled log
  - Run `python main.py infer --input data/raw/18B_decimated.ply`
  - Compare predicted bark fraction vs ground truth
  - Expected: rough agreement (not perfect with few training samples)

---

## Phase 3 — Dataset Expansion [ ] PENDING

- [ ] **TASK-15** · Label remaining unlabeled logs in Blender
  - Files needing labels: `19A_800N`, `45B_800N` + any others with n_bark=0
  - Workflow: open in Blender → assign corteza material → export PLY
  - Target: at least 8 labeled logs before final training

- [ ] **TASK-16** · Dataset quality audit
  - Run `python main.py info` after all logs are labeled
  - Check bark fraction distribution across logs (should vary: 5%–20%)
  - Flag any log with < 1% bark as potentially mislabeled

- [ ] **TASK-17** · Full training run (100 epochs)
  - `python main.py train`
  - Monitor `train_log.csv`: loss should decrease, bark IoU should exceed 0.4
  - Expected training time: 3–8 hours on CPU

---

## Phase 4 — Model Improvement [ ] PENDING

- [ ] **TASK-18** · Hyperparameter tuning
  - Try `batch_size: 2` if memory is limited
  - Try `learning_rate: 0.0005` if training is unstable
  - Try `num_points: 2048` for faster iteration
  - Log each experiment in a separate run folder

- [ ] **TASK-19** · Experiment with class weights
  - Set `class_weights: [1.0, 5.0]` in config for aggressive bark weighting
  - Compare bark IoU vs auto-computed weights
  - *Ref: SPEC-04 §class weights*

- [ ] **TASK-20** · Add UV texture features (optional enhancement)
  - The PLY files contain `s, t` UV coordinates
  - These encode texture information that may help distinguish bark visually
  - Modify `data/loader.py` to optionally include UV as extra features
  - Modify `data/dataset.py` and `model/pointnet2.py` accordingly
  - *Ref: CONSTITUTION §7 Extension Points*

- [ ] **TASK-21** · Cross-validation (when dataset ≥ 10 logs)
  - Implement k-fold split in `training/trainer.py`
  - Report mean ± std mIoU across folds
  - This is required for any academic publication

---

## Phase 5 — Production Readiness [ ] PENDING

- [ ] **TASK-22** · Batch inference script
  - Add `predict_folder()` to `inference/predictor.py`
  - Process all PLY in a folder and write a CSV summary with bark fractions
  - `python main.py infer --folder data/new_scans/ --output results.csv`

- [ ] **TASK-23** · Area calibration
  - Current area estimation uses ConvexHull approximation
  - Improve with actual mesh surface area from PLY face data
  - Validate against known physical measurements
  - *Ref: SPEC-05 §Area estimation*

- [ ] **TASK-24** · Export model for deployment
  - Add `torch.onnx.export()` option in `inference/predictor.py`
  - Enables integration with industrial systems without Python
  - Test ONNX output matches PyTorch output to 1e-4 tolerance

- [ ] **TASK-25** · Documentation and publication
  - Complete `README.md` with example outputs (screenshots)
  - Write `docs/methodology.md` for academic paper supplement
  - Add example notebook: `notebooks/exploratory_analysis.ipynb`
  - Update BibTeX citation in README

---

## How to Use This File for Spec-Driven Development

When asking an AI assistant to implement a task:

```
"Implement TASK-20 from TASKS.md.
 Constraints from CONSTITUTION §2.3 (CPU-first) and §2.1 (modularity).
 Follow the contract in SPEC-01 for loader changes and SPEC-02 for dataset changes.
 Do not break any existing tests in tests/test_preprocessing.py."
```

When reviewing an implementation:

```
"Review this implementation of TASK-18 against SPEC-04.
 Check that the checkpoint format matches the spec exactly.
 Verify CONSTITUTION §2.5 (fail loudly) is satisfied."
```

When adding a new feature not in this file:

1. Add a new TASK entry with a SPEC reference
2. Update the relevant SPEC section with the new contract
3. Check CONSTITUTION to confirm no principles are violated
4. Then implement

---

## Metrics Targets

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 2 | Tests passing | 15/15 |
| Phase 3 | Training convergence | val loss decreasing by epoch 20 |
| Phase 3 | bark IoU (2 logs) | > 0.30 |
| Phase 4 | bark IoU (8+ logs) | > 0.55 |
| Phase 4 | mIoU (8+ logs) | > 0.70 |
| Phase 5 | bark IoU (15+ logs) | > 0.65 |
