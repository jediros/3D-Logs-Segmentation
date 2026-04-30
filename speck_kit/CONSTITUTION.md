# CONSTITUTION — 3D Log Bark Segmentation

## 1. Purpose

This repository exists to segment residual bark on debarked logs from 3D scans.
It is both a working project and a reference architecture for future experiments.

---

## 2. Core Principles

### 2.1 Modularity first

Each module must have one clear responsibility.
Prefer a shallow, explicit dependency structure over convenience imports.

### 2.2 Current-state docs over aspirational docs

Documentation must describe what the repository does now.
Future ideas belong in `TASKS.md`, not inside behavioral contracts.

### 2.3 CPU-first baseline

The repository must remain usable on CPU.
GPU support is optional and must not break CPU execution.

### 2.4 Reproducibility is part of the design

Pinned dependencies, smoke validation, and self-describing checkpoints are not optional extras.
They are part of the architecture.

### 2.5 Data policy must be explicit

If bark-free samples are kept, say so.
If they are filtered, say so.
Silent policy drift is not acceptable.

### 2.6 Fail clearly

Missing files and broken inputs should fail with descriptive errors.
Bark-free but valid samples may warn without being treated as corrupted.

---

## 3. Domain Definitions

| Term | Definition |
|------|------------|
| Log | Debarked trunk segment scanned as a 3D surface |
| Bark | Residual bark on the surface, class `1` |
| Wood | Clean wood surface, class `0` |
| PLY file | Blender-exported point cloud or mesh with per-vertex attributes |
| Boundary vertex | Vertex with ambiguous label weight, optionally mapped to `-1` |
| Point features | `(N, C)` array with xyz and optional normals and/or RGB |
| mIoU | Mean IoU across classes |
| bark IoU | IoU for the bark class |

---

## 4. Quality Standards

### Code

- Public interfaces should stay small and explicit.
- File paths should use `pathlib.Path`.
- Feature toggles must not silently change checkpoint compatibility.

### Testing

- Synthetic fixtures should cover loader and dataset behavior.
- The repository should maintain a cheap smoke validation path.
- The baseline test suite must stay green before structural refactors.

### Documentation

- SPEC defines behavior.
- PLAN explains design choices.
- IMPLEMENT defines operational checks.
- TASKS tracks future work.

---

## 5. Architecture Constraints

### Loader constraint

`load_ply_labeled()` is the single data-entry contract for PLY parsing.
If the input format changes, update the loader instead of spreading parsing logic elsewhere.

### Dataset constraint

`BarkDataset` owns sampling, normalization, and augmentation.
Training code should consume tensors, not raw parsing logic.

### Checkpoint constraint

Saved checkpoints must carry enough config to reconstruct inference.
At minimum this includes:
- `num_classes`
- `use_normals`
- `use_rgb`
- `num_points`

---

## 6. Extension Points

| Extension | Where | Rule |
|-----------|-------|------|
| New backbone | `model/` | keep `(B, N, C) -> (B, N, K)` contract |
| New input features | `data/loader.py`, `data/dataset.py`, `model/pointnet2.py` | update checkpoint schema too |
| GPU support | `training/trainer.py`, `inference/predictor.py` | CPU path must remain valid |
| Batch inference | `inference/predictor.py` | preserve current single-file path |
| New CLI commands | `main.py` | keep global `--config` behavior consistent |

---

## 7. Template Rule for Future Reuse

If this folder is copied into a new architecture, the first obligation is to remove stale specifics.
Reusable structure is valuable; stale claims are not.