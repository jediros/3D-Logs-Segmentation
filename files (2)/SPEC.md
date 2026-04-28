# SPEC — 3D Log Bark Segmentation

> This document specifies **what** the system does and **why**.
> It does not specify **how** — that is the responsibility of the implementation.
> Any AI assistant or developer implementing features must consult this document first.

---

## Module Specifications

---

### SPEC-01 · `data/loader.py`

**Purpose**: Read `.ply` files exported from Blender and return structured numpy arrays.

**Contract**:

```python
load_ply_labeled(
    filepath,               # Path to .ply file
    label_threshold=0.5,    # float → int conversion threshold
    ignore_boundary=False,  # mark ambiguous vertices as -1
    boundary_margin=0.1,    # defines ambiguous zone
    compute_normals=True,   # compute vertex normals from faces
) -> (cloud, labels, metadata)
```

**Outputs**:
- `cloud`: `np.ndarray (N, 6)` float32 — `[x, y, z, nx, ny, nz]`
- `labels`: `np.ndarray (N,)` int32 — `0=wood, 1=bark, -1=ignore`
- `metadata`: dict with `n_bark`, `n_wood`, `bark_fraction`, `dimensions_mm`, etc.

**Invariants**:
- Must work with no external dependencies (no trimesh, no open3d)
- Must handle PLY files where label fields have any name containing "label_1" or "corteza"
- Must raise `FileNotFoundError` if file does not exist
- Normals must have unit length (L2 norm = 1.0 ± 1e-5)
- `cloud.shape[0]` must equal `labels.shape[0]`

**Known edge cases**:
- `test45B`: label_1 = 0.0 everywhere → detected as unlabeled, n_bark = 0
- Boundary vertices: label_1 ∈ (0.1, 0.9) → class assigned by threshold, or -1 if ignore_boundary=True
- Files with `label_1 + label_0 ≠ 1.0` → still valid, only label_1 is used

---

### SPEC-02 · `data/dataset.py`

**Purpose**: PyTorch Dataset that loads PLY files and prepares tensors for training.

**Contract**:

```python
BarkDataset(
    ply_dir,              # folder containing .ply files
    num_points=4096,      # exact point count per sample
    augment=False,        # geometric augmentation (train only)
    use_normals=True,     # include normals as features
    ignore_boundary=True, # discard boundary points from sampling
    cache=True,           # preload all PLY into RAM
)
```

**`__getitem__` output**:
- `points`: `FloatTensor (num_points, 6)` — normalized xyz + normals
- `labels`: `LongTensor (num_points,)` — values in {0, 1}

**Sampling rule**:
- If N > num_points: random sample without replacement
- If N < num_points: use all N points + random sample with replacement to fill

**Normalization rule** (applied inside `__getitem__`):
- Center XYZ at origin (subtract centroid)
- Scale XYZ to unit sphere (divide by max radius)
- Normals are NOT modified by normalization

**Augmentation rules** (training only):
- Random rotation around Y axis (log longitudinal axis): full 360°
- Small random tilt around X and Z: ±8 degrees
- Gaussian jitter on XYZ: σ = 0.002
- Random uniform scale: [0.95, 1.05]
- Normals rotated consistently with geometry

**Filtering rule**:
- Logs with `n_bark = 0` are automatically excluded from the dataset
- Warning is printed for each excluded file

---

### SPEC-03 · `model/pointnet2.py`

**Purpose**: PointNet++ encoder-decoder for per-point binary segmentation.

**Architecture**:

```
Input (B, N, 6)
    ↓
SetAbstraction(npoint=1024, r=0.1, k=32)  → (B, 1024, 64)
SetAbstraction(npoint=256,  r=0.2, k=64)  → (B, 256,  128)
SetAbstraction(npoint=64,   r=0.4, k=128) → (B, 64,   256)
    ↓
FeaturePropagation(256+128 → 256)         → (B, 256,  256)
FeaturePropagation(256+64  → 128)         → (B, 1024, 128)
FeaturePropagation(128+3   → 128)         → (B, N,    128)
    ↓
Conv1d(128→128) + BN + ReLU + Dropout(0.5)
Conv1d(128→num_classes)
    ↓
Output (B, N, 2)  ← logits, no softmax
```

**Contracts**:
- Input shape: `(B, N, 6)` if use_normals=True, `(B, N, 3)` otherwise
- Output shape: `(B, N, num_classes)` — raw logits
- Must run on CPU with batch_size=4, num_points=4096 in < 60s per batch
- `count_parameters()` must return an integer between 100,000 and 10,000,000

**FocalLoss**:
- `forward(logits, targets)` where logits=(B,N,C), targets=(B,N)
- Applies per-class alpha weights + focal term (1-p)^γ
- Default γ = 2.0 per original paper

---

### SPEC-04 · `training/trainer.py`

**Purpose**: Full training loop with validation, checkpointing, and logging.

**`train(cfg, resume=False)` behavior**:
1. Load dataset from `cfg.paths.raw_data`
2. Split into train/val using `cfg.training.val_split` (random, seeded)
3. Enable augmentation on train split only
4. Compute class weights automatically if `cfg.training.class_weights` is null
5. Train for `cfg.training.epochs` epochs
6. After each epoch: validate, log metrics, save checkpoint
7. Save `best_model.pth` whenever val mIoU improves
8. Always save `last_checkpoint.pth` for resume support

**Checkpoint format** (required fields):
```python
{
    "epoch":           int,
    "model_state":     OrderedDict,       # model.state_dict()
    "optimizer_state": OrderedDict,       # optimizer.state_dict()
    "best_miou":       float,
    "val_metrics":     dict,              # from SegmentationMetrics.compute()
    "config": {
        "num_classes": int,
        "use_normals": bool,
        "num_points":  int,
    }
}
```

**Resume behavior**:
- `--resume` loads `last_checkpoint.pth` and continues from next epoch
- If checkpoint not found, starts from epoch 1 with a warning

---

### SPEC-05 · `inference/predictor.py`

**Purpose**: Load a trained model and segment new logs.

**`BarkPredictor.from_checkpoint(path)` contract**:
- Must reconstruct model architecture from checkpoint `config` field
- Must NOT require access to `config/default.yaml`
- Must print model path and best mIoU on load

**`predict_ply(ply_path, ...)` contract**:
- Accepts any `.ply` file, labeled or unlabeled
- Returns dict with: `bark_fraction`, `n_bark_points`, `n_wood_points`, `labels`, `pts`
- If `save_ply=True`: writes colored point cloud to `output_dir/<stem>_segmented.ply`
- If `visualize=True`: opens Open3D window (requires display)

**Area estimation**:
- Uses `compute_bark_area()` from `utils/metrics.py`
- If `surface_area_m2` provided: exact area calculation
- If not provided: estimated via ConvexHull of point cloud

---

### SPEC-06 · `utils/metrics.py`

**Purpose**: Evaluation metrics and bark area computation.

**`SegmentationMetrics` contract**:
- Accumulates predictions via `update(preds, targets)`
- Ignores points where `targets == -1`
- `compute()` returns dict with: `iou`, `miou`, `precision`, `recall`, `f1`, `accuracy`
- All values are Python floats (not numpy scalars)
- `reset()` clears all accumulated state

**`compute_bark_area(pts, labels, surface_area_m2)` contract**:
- `pts`: (N, 3) float array
- `labels`: (N,) int array with values 0, 1 (or -1 ignored)
- Returns dict with `bark_fraction` in [0, 1] and areas in m² or scan units

---

### SPEC-07 · `main.py`

**Purpose**: Single entry point for all user-facing operations.

**Commands**:

| Command | Description | Key args |
|---------|-------------|----------|
| `info` | Scan `data/raw/`, print label status per file | — |
| `train` | Train PointNet++ | `--resume` |
| `infer` | Segment a new log | `--input`, `--visualize`, `--no-ply` |
| `visualize` | Open 3D viewer for any PLY | `--file` |
| `preprocess` | Optional: convert PLY to .npy cache | `--overwrite` |

**`info` command output format** (must include):
- Per-file: filename, n_vertices, n_bark, n_wood, bark%, labeled YES/NO
- Total summary row
- Warning list for files without labels
- Final recommendation: "Ready to train: X/Y logs"

---

## Data Flow Diagram

```
Blender (.obj + texture)
    │
    │  assign materials → export PLY
    ▼
data/raw/*.ply
    │
    │  load_ply_labeled()
    ▼
cloud (N,6) + labels (N,)          ← loader.py
    │
    │  BarkDataset.__getitem__()
    │  normalize + sample + augment
    ▼
points (4096,6) + labels (4096,)   ← dataset.py
    │
    │  DataLoader → batches
    ▼
(B, 4096, 6) tensor                ← trainer.py
    │
    │  PointNet2Segmentation.forward()
    ▼
logits (B, 4096, 2)                ← pointnet2.py
    │
    │  FocalLoss.forward()
    ▼
loss scalar → backprop → checkpoint

── inference path ──────────────────
checkpoint.pth
    │
    │  BarkPredictor.from_checkpoint()
    ▼
new_log.ply → predict_ply() → _segmented.ply + bark_fraction
```

---

## Configuration Reference

All parameters in `config/default.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `paths.raw_data` | `data/raw` | Folder with input PLY files |
| `paths.checkpoints` | `training/checkpoints` | Where .pth files are saved |
| `paths.logs` | `training/logs` | Where train_log.csv is saved |
| `preprocessing.num_points` | `4096` | Points sampled per log |
| `preprocessing.ignore_boundary` | `true` | Discard ambiguous border vertices |
| `preprocessing.label_threshold` | `0.5` | Float-to-int conversion threshold |
| `model.num_classes` | `2` | Output classes (do not change) |
| `model.use_normals` | `true` | Use surface normals as features |
| `training.epochs` | `100` | Total training epochs |
| `training.batch_size` | `4` | Batch size (keep ≤ 4 for CPU) |
| `training.learning_rate` | `0.001` | Initial learning rate |
| `training.lr_decay` | `0.7` | LR multiplier per decay step |
| `training.lr_decay_step` | `20` | Epochs between LR decay |
| `training.val_split` | `0.2` | Fraction of logs for validation |
| `training.class_weights` | `null` | null = auto-compute from data |
| `inference.model_path` | `training/checkpoints/best_model.pth` | Default model for inference |
