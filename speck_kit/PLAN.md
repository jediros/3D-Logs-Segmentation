# PLAN — 3D Log Bark Segmentation

> Implementation strategy for the current repository.
> This plan is intentionally reusable as a reference when bootstrapping future architectures.

---

## 1. Technology Stack

| Layer | Technology | Current choice |
|-------|------------|----------------|
| Language | Python | 3.10 in Dev Container, 3.12 validated locally |
| Deep learning | PyTorch | 2.2.2 |
| Numerical stack | NumPy / SciPy | 1.26.4 / 1.13.0 |
| 3D visualization | Open3D | 0.19.0 |
| Config | YAML + SimpleNamespace | lightweight and readable |
| Testing | pytest | synthetic PLY fixtures |
| Logging | CSV | easy offline analysis |
| Environment | Dev Container + local venv | reproducible and practical |

---

## 2. Design Decisions

### 2.1 Why PointNet++

The task is surface segmentation, not volumetric occupancy.
PointNet++ fits well because it captures local neighborhoods without voxelizing empty space.

### 2.2 Why keep RGB optional but first-class

The current repository already supports scanner RGB in addition to geometry.
That keeps the model useful for two scenarios:
- geometry-only scans
- geometry + appearance scans

Future architectures should preserve this pattern: optional feature channels without breaking the xyz-only path.

### 2.3 Why keep bark-free logs

The current project intentionally keeps logs with zero bark in the dataset.
This is useful because the model must also learn what clean wood looks like.

Future templates should decide this policy explicitly instead of silently filtering those samples out.

### 2.4 Why split train and validation with separate dataset instances

The repository previously had augmentation leakage into validation.
The current design fixes that by constructing separate base datasets:
- train dataset with `augment=True`
- validation dataset with `augment=False`

This is a pattern worth copying into future projects whenever dataset objects hold behavioral flags.

### 2.5 Why keep the loader dependency-light

The PLY parser stays in pure NumPy.
That reduces dependency friction and localizes format-specific logic to one module.

### 2.6 Why normalize inside the dataset

Normalization inside `__getitem__` keeps raw files untouched and allows feature toggles or point-count changes without regenerating cached artifacts.

---

## 3. Data Pipeline

```text
binary PLY
    ↓
header parse
    ↓
vertex dtype build
    ↓
vertex array decode
    ↓
optional face parse
    ↓
optional normal computation
    ↓
label extraction (`label_1` or `corteza`)
    ↓
feature assembly: xyz | xyz+normals | xyz+rgb | xyz+normals+rgb
    ↓
dataset sampling
    ↓
XYZ normalization
    ↓
train-time augmentation
    ↓
PointNet++
```

---

## 4. Training Strategy

### 4.1 Split policy

The split is done at log level, not point level.
That gives a more honest validation signal for small datasets.

### 4.2 Loss policy

The implementation supports either:
- explicit class weights from config
- automatic class weights when config sets them to `null`

The current default config uses an explicit bark-heavy weighting.

### 4.3 Optimization policy

- Adam optimizer
- StepLR schedule
- gradient clipping at `1.0`
- CPU-first execution

### 4.4 Logging policy

Each run writes:
- `training/logs/run_info.txt`
- `training/logs/train_log.csv`
- `training/checkpoints/best_model.pth`
- `training/checkpoints/last_checkpoint.pth`

---

## 5. Runtime Shape Strategy

The model parameter count is mostly independent of `num_points`, but runtime is not.
The current repo uses `16384` points by default and a smaller smoke config for quick checks.

That is the preferred template pattern:
- one realistic default config
- one cheap smoke config for validation

---

## 6. Current Constraints Worth Preserving

- CPU must remain a supported baseline
- checkpoints must be self-describing
- inference must work without re-reading YAML config
- docs must distinguish current behavior from future aspirations

---

## 7. Inference Area Strategy

The current public implementation estimates bark area from predicted point labels and geometric approximations.

Current order of preference:
1. ConvexHull area when available
2. Cylindrical fallback when ConvexHull fails

Exact mesh-area integration is still a future enhancement.

---

## 8. Reuse Notes for Future Architectures

This repository is a good template because it already demonstrates a few reusable patterns:
- feature-flagged inputs (`use_normals`, `use_rgb`)
- self-contained checkpoints
- smoke-test config beside full config
- strict separation between training and inference concerns
- thin CLI over modular internals