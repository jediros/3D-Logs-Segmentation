# 3D Log Bark Segmentation

**PointNet++ semantic segmentation of residual bark on debarked logs**

Modular Python application for quantifying residual bark area on debarked log surfaces using 3D point cloud segmentation. Input: labeled `.ply` files exported from Blender. Output: per-point bark/wood classification + bark area estimation.

---

## Overview

After industrial debarking, quantifying residual bark is difficult to automate on production lines. This project uses 3D surface scans (exported as `.ply` from Blender with material-based labels) to train a PointNet++ segmentation model that classifies each surface point as:

- `0` → wood (clean surface)
- `1` → residual bark

---

## Requirements

- Docker Engine 24+
- VS Code with [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- No GPU required — all training runs on CPU

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/3D_logs_seg
cd 3D_logs_seg

# 2. Place your labeled .ply files
cp /path/to/your/scans/*.ply data/raw/

# 3. Open in VS Code
code .
# → VS Code: "Reopen in Container"  (Ctrl+Shift+P → Dev Containers: Reopen in Container)
# → Wait ~3 min for first build

# 4. Inside the container terminal:
python main.py info      # check dataset status
python main.py train     # train the model
python main.py infer --input data/raw/new_log.ply --visualize
```

---

## Data Preparation (Blender)

Scans are processed in Blender from `.obj` + texture files:

1. Import the `.obj` scan in Blender
2. Create two material slots: `madera` (wood) and `corteza` (bark)
3. In Edit Mode, assign faces to the appropriate material
4. Export: `File → Export → Stanford PLY` with **Include Attributes** enabled

The exported `.ply` will contain `label_0` and `label_1` float fields per vertex (Blender material blend weights). The loader converts these automatically:

```
label_1 > 0.5  →  class 1 (bark)
label_1 ≤ 0.5  →  class 0 (wood)
```

---

## Project Structure

```
3D_logs_seg/
│
├── main.py                      # entry point: info / train / infer / visualize
│
├── config/
│   ├── default.yaml             # all hyperparameters and paths
│   └── config_loader.py         # YAML → SimpleNamespace
│
├── data/
│   ├── loader.py                # PLY parser + normal computation (no external deps)
│   ├── dataset.py               # PyTorch Dataset — loads PLY directly
│   └── raw/                     # ← place your .ply files here (not committed to git)
│
├── preprocessing/
│   └── sampler.py               # normalization, voxel downsampling utilities
│
├── model/
│   └── pointnet2.py             # PointNet++ encoder-decoder + FocalLoss
│
├── training/
│   ├── trainer.py               # full training loop with checkpointing
│   ├── checkpoints/             # saved .pth models
│   └── logs/                    # train_log.csv per-epoch metrics
│
├── inference/
│   └── predictor.py             # load checkpoint → segment new log → compute area
│
├── utils/
│   ├── metrics.py               # IoU, F1, mIoU, bark area estimation
│   ├── logger.py                # CSV logger + console output
│   └── visualizer.py            # Open3D colored point cloud rendering
│
├── tests/
│   └── test_preprocessing.py    # pytest unit tests
│
├── .devcontainer/
│   ├── devcontainer.json        # VS Code Dev Container config (Python 3.10)
│   └── Dockerfile
│
└── requirements.txt
```

---

## Commands

```bash
# Inspect dataset — shows label status for all PLY files in data/raw/
python main.py info

# Train PointNet++ on CPU
python main.py train

# Resume interrupted training
python main.py train --resume

# Segment a new log (outputs colored .ply to outputs/)
python main.py infer --input data/raw/new_log.ply

# Segment and open 3D viewer
python main.py infer --input data/raw/new_log.ply --visualize

# Visualize any PLY with its labels
python main.py visualize --file data/raw/18B_decimated.ply

# Run tests
pytest tests/ -v
```

---

## Architecture

PointNet++ MSG (Multi-Scale Grouping) adapted for binary surface segmentation:

```
Input: (B, N, 6)   xyz + surface normals
         │
    SetAbstraction × 3    ← hierarchical local feature extraction
    (FPS + ball query + mini-PointNet per group)
         │
    FeaturePropagation × 3  ← interpolate features back to all points
    (inverse-distance weighted k-NN)
         │
    MLP head + Dropout
         │
Output: (B, N, 2)  per-point logits → argmax → {0: wood, 1: bark}
```

**Loss function**: Focal Loss (γ=2) with inverse-frequency class weights.
Bark typically represents 5–15% of surface points, making standard cross-entropy unreliable.

**Training**: CPU-only, batch size 4, Adam optimizer, StepLR scheduler.
Typical training time: 3–8 hours for 100 epochs on a modern laptop CPU.

---

## Configuration

All parameters are in `config/default.yaml`. Key settings:

```yaml
preprocessing:
  num_points: 4096          # points sampled per log
  ignore_boundary: true     # discard ambiguous border vertices

model:
  use_normals: true         # use surface normals as additional features

training:
  epochs: 100
  batch_size: 4
  learning_rate: 0.001
  val_split: 0.2            # fraction of logs used for validation
```

---

## Output

After inference, results are saved to `outputs/`:

- `<log_name>_segmented.ply` — colored point cloud (brown=wood, dark=bark)
- Console summary:

```
Segmentando: new_log.ply
  Corteza: 612 pts (13.2%)
  Madera:  4036 pts
  Guardado: outputs/new_log_segmented.ply
```

The colored `.ply` can be opened in [CloudCompare](https://cloudcompare.org/) or [MeshLab](https://www.meshlab.net/) for visual inspection.

---

## Metrics

Evaluation uses per-class IoU and mean IoU (mIoU):

| Metric | Description |
|--------|-------------|
| mIoU | Mean Intersection over Union across both classes |
| bark IoU | IoU for the bark class specifically (primary metric) |
| Accuracy | Fraction of correctly classified points |
| F1 (bark) | Harmonic mean of precision and recall for bark |

Training logs are saved to `training/logs/train_log.csv` and can be plotted with any tool (pandas, Excel, etc.).

---

## Dataset Notes

- Minimum recommended: **5 labeled logs** for meaningful generalization
- Logs with `label_1 = 0` throughout (unlabeled) are automatically skipped
- Class imbalance (bark 5–15%) is handled via Focal Loss + class weights
- Data augmentation: random Y-axis rotation, small tilt (±8°), Gaussian jitter, scale ±5%

---

## Citing

If you use this work in academic publications:

```bibtex
@software{3d_logs_seg,
  author  = {Your Name},
  title   = {3D Log Bark Segmentation using PointNet++},
  year    = {2025},
  url     = {https://github.com/your-username/3D_logs_seg}
}
```

---

## License

MIT License — see `LICENSE` for details.
