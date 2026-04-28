# PLAN — 3D Log Bark Segmentation

> This document describes **how** the system is built — architecture decisions,
> technology choices, and implementation strategy.
> It bridges SPEC (what) and IMPLEMENT (step-by-step execution).

---

## 1. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.10 | Stable LTS, broad ML ecosystem, type hints support |
| Deep Learning | PyTorch 2.2 (CPU) | Dynamic graphs, easy debugging, no CUDA required |
| 3D I/O | Custom binary parser | Zero external deps for PLY — avoids trimesh/plyfile version conflicts |
| Normal computation | NumPy + face cross-product | Faster than Open3D for offline preprocessing |
| Visualization | Open3D 0.18 | Best interactive 3D viewer for point clouds in Python |
| Config | YAML + SimpleNamespace | Human-readable, dot-access without dataclass overhead |
| Containerization | Docker + VS Code DevContainer | Reproducible across machines, academic-friendly |
| Testing | pytest | Standard, compatible with CI/CD |
| Logging | CSV (pandas-compatible) | Simple, portable, easy to plot in any tool |

---

## 2. Architecture Decisions

### 2.1 Why PointNet++ and not a voxel-based method

Bark segmentation on log surfaces is fundamentally a **surface geometry problem**.
The key discriminating features are:

- **Surface normals**: bark has high normal variation (rough), wood is smooth
- **Local curvature**: bark patches have irregular boundaries
- **Spatial continuity**: bark regions tend to cluster

PointNet++ captures all three via its hierarchical local grouping (SetAbstraction).
Voxel methods (VoxNet, SparseConvNet) waste computation on empty interior space.
PointNet (v1) lacks local geometric awareness — critical for bark texture discrimination.

### 2.2 Why Focal Loss instead of CrossEntropy

Bark is 5–15% of surface points. With standard CrossEntropy:
- The model learns to predict "all wood" and achieves 88%+ accuracy
- bark IoU stays near zero — the model never learns bark

Focal Loss suppresses easy examples (abundant wood) and focuses learning on hard ones
(rare bark, ambiguous boundaries). The γ=2 factor from the original paper is used.

Additional per-class alpha weights (inverse frequency) compound this effect.

### 2.3 Why load PLY without external libraries

The PLY format from Blender is well-defined binary little-endian.
Implementing the parser in pure NumPy gives:
- Zero dependency conflicts (plyfile, trimesh have frequent API changes)
- Full control over field extraction
- Faster loading for small files (< 1MB)

The tradeoff: only supports the exact Blender export format.
If a new scanner produces a different PLY variant, `data/loader.py` is the only file to change.

### 2.4 Why cache PLY in RAM

With 19 logs × ~200KB each = ~4MB total.
Loading from disk each epoch adds ~2s per epoch × 100 epochs = ~3 min wasted.
RAM cache eliminates this entirely. For datasets > 500MB, disable with `cache=False`.

### 2.5 Why normalize inside `__getitem__` not in preprocessing

Normalization inside the Dataset means:
- Raw PLY files are never modified
- Augmentation happens after normalization (correct order)
- No intermediate processed files to manage
- Different num_points settings work without re-preprocessing

The cost: normalization runs every epoch. For 4096 points, this is < 1ms per sample.

---

## 3. Data Pipeline Design

```
PLY file (Blender export)
    ↓
_parse_ply_header()         → n_vertices, n_faces, property names
    ↓
_build_vertex_dtype()       → numpy dtype matching PLY fields
    ↓
np.frombuffer()             → raw vertex array
    ↓
_parse_faces()              → face index array (N_faces, 3)
    ↓
compute_vertex_normals()    → per-vertex normals from face cross-products
    ↓
label extraction            → label_1 float → threshold → int32
    ↓
cloud = [xyz | normals]     → (N, 6) float32
labels = [0/1/-1]           → (N,) int32
    ↓
BarkDataset.__getitem__()
    ↓
filter boundary points (-1)
    ↓
sample to num_points        → random choice without replacement
    ↓
normalize XYZ               → center + unit sphere
    ↓
augment (train only)        → rotate Y, tilt XZ, jitter, scale
    ↓
FloatTensor (4096, 6) + LongTensor (4096,)
```

---

## 4. Training Strategy

### 4.1 Data split

With few logs (2–5), a random 80/20 split at the **log level** is used.
This means the validation log(s) are completely unseen during training —
more realistic than point-level splits.

With ≥ 10 logs: consider k-fold cross-validation (TASK-21).

### 4.2 Learning rate schedule

```
epoch 1-20:   lr = 0.001
epoch 21-40:  lr = 0.0007
epoch 41-60:  lr = 0.00049
...
StepLR: γ=0.7 every 20 epochs
```

This aggressive decay prevents oscillation in late training
and helps the model refine bark boundary decisions.

### 4.3 Gradient clipping

`clip_grad_norm_(model.parameters(), max_norm=1.0)` is applied every step.
With small batches (4) and imbalanced data, gradients can spike early in training.

### 4.4 Class weight computation

```python
counts[0] = total wood points across all training logs
counts[1] = total bark points across all training logs
weight[c] = total / (2 * counts[c])
```

Example with 88% wood / 12% bark:
- weight[0] (wood)  = 1.0 / (2 × 0.88) ≈ 0.57
- weight[1] (bark)  = 1.0 / (2 × 0.12) ≈ 4.17

Combined with Focal Loss, this gives bark ~7× more gradient signal than wood.

---

## 5. Model Sizing (CPU constraints)

With `num_points=4096`, batch_size=4 on CPU:

| Layer | Output shape | Approx params |
|-------|-------------|---------------|
| SA1 (npoint=1024) | (4, 1024, 64) | ~8K |
| SA2 (npoint=256) | (4, 256, 128) | ~50K |
| SA3 (npoint=64) | (4, 64, 256) | ~200K |
| FP3 | (4, 256, 256) | ~130K |
| FP2 | (4, 1024, 128) | ~100K |
| FP1 | (4, 4096, 128) | ~50K |
| Head | (4, 4096, 2) | ~33K |
| **Total** | | **~571K** |

Forward pass time on modern laptop CPU: ~8–15 seconds per batch.
Full epoch with 2 logs, batch_size=4: ~30–60 seconds.
100 epochs: ~1–2 hours.

To reduce training time, set `num_points: 2048` in config (halves computation).

---

## 6. File Organization Rationale

```
3D_logs_seg/
├── data/          ← I/O only. No model code here.
├── preprocessing/ ← Transformations that don't require labels.
├── model/         ← Architecture only. No training logic here.
├── training/      ← Training loop only. No architecture here.
├── inference/     ← Uses trained model. No training here.
├── utils/         ← Shared utilities with no upward imports.
└── config/        ← Configuration only. No business logic.
```

Each directory has exactly one responsibility.
`utils/` is the only module imported by multiple others.
`config/` is imported by everyone but imports nothing from the project.

---

## 7. PLY Label Format Reference

Blender exports material blend weights as float fields per vertex:

```
label_0 = blend weight for material slot 0 (wood)
label_1 = blend weight for material slot 1 (bark)
```

For pure vertices (assigned to one material): values are exactly 0.0 or 1.0.
For boundary vertices (on the edge between two materials): values sum to 1.0
but neither is 0 or 1 — typically in range (0.1, 0.9).

The loader uses `label_1 > 0.5` as the decision boundary.
With `ignore_boundary=True`, vertices where `0.1 < label_1 < 0.9`
are marked as `-1` and excluded from training loss computation.

---

## 8. Inference Area Calculation

Given predicted labels on a uniformly sampled surface:

```
bark_fraction = n_bark_points / (n_bark_points + n_wood_points)
bark_area     = bark_fraction × total_surface_area
```

`total_surface_area` comes from:
1. **Preferred**: actual mesh surface area from PLY face data (exact)
2. **Fallback**: ConvexHull of point cloud (approximate, ±10–20%)
3. **Manual override**: `--area` flag in `python main.py infer`

For production use, the actual mesh area should always be used.
