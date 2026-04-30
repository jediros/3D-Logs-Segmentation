# SPEC — 3D Log Bark Segmentation

> Reference specification for the current repository state.
> Keep this file concrete enough to guide implementation, but reusable enough
> to serve as a template for future architectures.

---

## Module Specifications

---

### SPEC-01 · `data/loader.py`

**Purpose**: Read Blender-exported `.ply` files and return point features, labels, and metadata.

**Contract**:

```python
load_ply_labeled(
    filepath,
    label_threshold=0.5,
    ignore_boundary=False,
    boundary_margin=0.1,
    compute_normals=True,
    use_rgb=False,
) -> (cloud, labels, metadata)
```

**Outputs**:
- `cloud`: `np.ndarray (N, C)` float32
  - `C = 3` -> xyz
  - `C = 6` -> xyz + normals or xyz + rgb
  - `C = 9` -> xyz + normals + rgb
- `labels`: `np.ndarray (N,)` int32 with values `0=wood`, `1=bark`, `-1=ignore`
- `metadata`: dict with counts, fractions, bounds, feature count, and format flags

**Invariants**:
- Must work without trimesh or Open3D
- Must raise `FileNotFoundError` when the file is missing
- Must accept label fields named exactly `label_1` or `corteza`
- If normals are computed, they must be unit-length within numerical tolerance
- `cloud.shape[0] == labels.shape[0]`

**Known edge cases**:
- Files with `label_1 = 0.0` everywhere are valid and treated as bark-free logs
- Boundary vertices can be mapped to `-1` when `ignore_boundary=True`
- Files without RGB fields remain valid even if `use_rgb=True`; the loader warns and falls back to available features

---

### SPEC-02 · `data/dataset.py`

**Purpose**: Prepare fixed-size training samples from PLY files.

**Contract**:

```python
BarkDataset(
    ply_dir,
    num_points=4096,
    augment=False,
    use_normals=True,
    use_rgb=False,
    ignore_boundary=True,
    cache=True,
)
```

**`__getitem__` output**:
- `points`: `FloatTensor (num_points, C)`
- `labels`: `LongTensor (num_points,)` with values in `{0, 1}`

**Feature rules**:
- XYZ is always present
- Normals are optional
- RGB is optional
- Normalization modifies XYZ only

**Sampling rule**:
- If `N > num_points`: sample without replacement
- If `N < num_points`: keep all points and fill by sampling with replacement

**Augmentation rule**:
- Train-time only
- Y-axis rotation, small X/Z tilts, XYZ jitter, and uniform scaling
- RGB values are never augmented

**Filtering rule**:
- Readable bark-free logs are kept in the dataset
- Broken or unreadable PLY files are skipped with a warning
- Validation must run with augmentation disabled

---

### SPEC-03 · `model/pointnet2.py`

**Purpose**: PointNet++ encoder-decoder for binary per-point segmentation.

**Input contract**:
- `(B, N, 3)` for xyz only
- `(B, N, 6)` for xyz + normals or xyz + rgb
- `(B, N, 9)` for xyz + normals + rgb

**Output contract**:
- `(B, N, num_classes)` raw logits

**Architecture**:
- SetAbstraction x 3
- FeaturePropagation x 3
- Conv/BN/ReLU/Dropout head
- Default target classes: wood and bark

**Model contract**:
- Must run on CPU
- `count_parameters()` must return a value between `100,000` and `10,000,000`

**Loss contract**:
- `FocalLoss.forward(logits, targets)` expects logits `(B, N, C)` and targets `(B, N)`
- Supports class weighting through `alpha`
- Default `gamma = 2.0`

---

### SPEC-04 · `training/trainer.py`

**Purpose**: Train, validate, checkpoint, and log experiments.

**`train(cfg, resume=False)` behavior**:
1. Load all PLY files from `cfg.paths.raw_data`
2. Build a seeded train/validation split at log level
3. Use separate dataset instances so validation never receives augmentation
4. Use configured class weights, or compute them automatically when null
5. Train for `cfg.training.epochs`
6. Validate and log metrics after each epoch
7. Save `best_model.pth` when validation mIoU improves
8. Always save `last_checkpoint.pth`

**Checkpoint format**:

```python
{
    "epoch": int,
    "model_state": OrderedDict,
    "optimizer_state": OrderedDict,
    "best_miou": float,
    "val_metrics": dict,
    "config": {
        "num_classes": int,
        "use_normals": bool,
        "use_rgb": bool,
        "num_points": int,
    },
}
```

**Resume behavior**:
- `--resume` continues from `last_checkpoint.pth` when present
- Missing checkpoint falls back to a fresh start

---

### SPEC-05 · `inference/predictor.py`

**Purpose**: Load a trained model and segment new logs.

**`BarkPredictor.from_checkpoint(path)` contract**:
- Reconstructs model shape from checkpoint config
- Does not require `config/default.yaml`
- Restores `use_normals`, `use_rgb`, and `num_points`
- Prints checkpoint summary on load

**`predict_ply(ply_path, ...)` contract**:
- Accepts labeled or unlabeled `.ply`
- Returns a dict with bark ratio, point counts, predicted labels, coordinates, and source path
- Saves `<stem>_segmented.ply` when `save_ply=True`
- Can visualize predictions through Open3D when requested

**Area estimation**:
- Uses `compute_bark_area()` from `utils/metrics.py`
- Current public pipeline estimates area from point geometry
- Exact mesh-area injection is a future extension, not a current CLI feature

---

### SPEC-06 · `utils/metrics.py`

**Purpose**: Compute segmentation metrics and bark-area estimates.

**`SegmentationMetrics` contract**:
- `update(preds, targets)` accumulates confusion counts
- Ignores targets outside valid class range
- `compute()` returns Python floats for IoU, mIoU, precision, recall, F1, and accuracy
- `reset()` clears internal state

**`compute_bark_area(pts, labels, surface_area_m2=None)` contract**:
- Works with `(N, 3)` coordinates and binary labels
- Returns bark fraction and area summary fields
- Uses ConvexHull when available and falls back to a cylindrical approximation otherwise

---

### SPEC-07 · `main.py`

**Purpose**: Single CLI entry point for the project.

**Commands**:

| Command | Description | Key args |
|---------|-------------|----------|
| `info` | Scan `data/raw/` and print per-file status | - |
| `train` | Train PointNet++ | `--resume` |
| `infer` | Segment a log | `--input`, `--visualize`, `--no-ply` |
| `visualize` | Open a PLY viewer | `--file` |
| `preprocess` | Optional preprocessing helper | `--overwrite` |

**CLI rule**:
- Global options such as `--config` must appear before the subcommand

**`info` output must include**:
- Per-file vertex, bark, wood, percentage, and label status columns
- A total summary row
- A warning list for bark-free files
- A final readiness line such as `Listos para entrenar: X/Y troncos`

---

## Data Flow Diagram

```text
Blender export (.ply)
    |
    |  load_ply_labeled()
    v
cloud (N,C) + labels (N,)
    |
    |  BarkDataset.__getitem__()
    |  filter ignored labels + sample + normalize + augment
    v
points (num_points,C) + labels (num_points,)
    |
    |  DataLoader
    v
(B, N, C)
    |
    |  PointNet2Segmentation.forward()
    v
logits (B, N, 2)
    |
    |  FocalLoss / metrics
    v
checkpoint + CSV logs

-- inference path --

checkpoint.pth
    |
    |  BarkPredictor.from_checkpoint()
    v
new_log.ply -> predict_ply() -> segmented.ply + bark summary
```