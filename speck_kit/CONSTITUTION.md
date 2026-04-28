# CONSTITUTION — 3D Log Bark Segmentation

## 1. Purpose

This project exists to solve a real industrial and academic problem:
**quantifying residual bark area on debarked log surfaces** using 3D point cloud segmentation.

Manual quantification on production lines is impractical. This system provides
an automated, reproducible pipeline that takes 3D surface scans and produces
per-point bark/wood classification, enabling precise bark area measurements.

---

## 2. Core Principles

### 2.1 Modularity above all
Every component of this system is a standalone module with a clear interface.
No module should import from another module it does not directly depend on.
The dependency graph flows in one direction only:

```
config → data → preprocessing → model → training → inference → utils
```

### 2.2 Reproducibility is non-negotiable
Any researcher must be able to clone this repository, run one command, and
reproduce results identically. The DevContainer with pinned dependencies
is the contract for reproducibility. Never rely on system-level packages.

### 2.3 CPU-first design
The system must run completely on CPU without any code changes.
GPU support may be added as an optional enhancement but must never
break CPU-only execution. This ensures accessibility for academic users
without specialized hardware.

### 2.4 Data stays out of the repository
Raw scan files (.ply), trained models (.pth), and inference outputs
are never committed to git. The repository contains only code, configuration,
and documentation. Data lives in `data/raw/` which is git-ignored.

### 2.5 Fail loudly and clearly
Every module must validate its inputs and raise descriptive errors.
A missing file, a corrupted PLY, or an unlabeled scan must produce
a clear human-readable message — never a cryptic Python traceback.

### 2.6 Labels come from Blender, not from code
The labeling workflow is: scan → Blender (material assignment) → PLY export.
The codebase does not include manual labeling tools. Ground truth is defined
by the domain expert in Blender, and the pipeline trusts that workflow.

---

## 3. Domain Definitions

| Term | Definition |
|------|-----------|
| **Log** | A debarked tree trunk segment, scanned as a 3D surface |
| **Bark** | Residual bark remaining on the surface after debarking (class 1) |
| **Wood** | Clean wood surface after debarking (class 0) |
| **PLY file** | 3D point cloud file exported from Blender with per-vertex material labels |
| **label_1** | Float field in PLY: Blender's blend weight for the bark material (0.0–1.0) |
| **label_0** | Float field in PLY: Blender's blend weight for the wood material (0.0–1.0) |
| **Binary label** | label_1 > 0.5 → bark (1), else wood (0) |
| **Boundary vertex** | Vertex where 0.1 < label_1 < 0.9 — interpolated by Blender, ambiguous |
| **Point cloud** | Set of 3D points (N, 6): [x, y, z, nx, ny, nz] — coords + surface normals |
| **mIoU** | Mean Intersection over Union — primary evaluation metric |
| **bark IoU** | IoU for bark class specifically — secondary key metric |

---

## 4. Non-Goals

The following are explicitly out of scope for this project:

- **Real-time inference** on production lines (latency is not a constraint)
- **GPU training** as a requirement (nice-to-have, not required)
- **Multi-class segmentation** beyond bark/wood binary classification
- **Automatic scan acquisition** — scans are assumed to be pre-processed
- **Web interface or API** — command-line usage is sufficient
- **Windows support outside WSL2** — Linux/macOS/WSL2 only

---

## 5. Quality Standards

### Code
- All public functions must have docstrings with Args/Returns
- No function longer than 80 lines
- No module longer than 400 lines
- All file paths handled via `pathlib.Path`, never raw strings

### Testing
- Every data transformation must have at least one unit test
- Tests must not require real PLY files — use synthetic fixtures
- `pytest tests/` must pass with zero failures before any commit

### Data
- Minimum 5 labeled logs recommended for meaningful model training
- Each labeled log must have at least 1% bark points (n_bark / n_total > 0.01)
- Logs failing this threshold are flagged in `python main.py info`

---

## 6. Architecture Constraints

### The loader contract
`load_ply_labeled()` is the single source of truth for reading data.
No other module reads PLY files directly. If the PLY format changes,
only `data/loader.py` needs to change.

### The dataset contract
`BarkDataset` is the single interface between raw data and the model.
It handles sampling, normalization, and augmentation internally.
The training loop never touches raw numpy arrays directly.

### The checkpoint contract
Every saved checkpoint must include:
- `model_state` — model weights
- `config` — dict with num_classes, use_normals, num_points
- `epoch`, `best_miou`, `val_metrics`

This ensures `BarkPredictor.from_checkpoint()` can reconstruct the model
without access to any configuration file.

---

## 7. Extension Points

The system is designed to be extended in these specific ways:

| Extension | Where | Notes |
|-----------|-------|-------|
| New architecture | `model/` | Add new file, implement `forward(xyz_feat) -> (B,N,C)` |
| New augmentation | `data/dataset.py` → `_augment()` | Keep geometric validity |
| Multi-class labels | `data/loader.py` + `config/default.yaml` | Change num_classes |
| GPU support | `training/trainer.py` → `device` | `torch.device("cuda")` |
| New input format | `data/loader.py` | Add new `load_*` function |
| Batch inference | `inference/predictor.py` | Add `predict_folder()` method |
