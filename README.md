# 3D Log Bark Segmentation

PointNet++ semantic segmentation of residual bark on debarked logs.

Classifies each point of a log surface as **wood** or **bark** and estimates the bark area from 3D point clouds exported from Blender.

---

## Results

Best model after 100 epochs (May 2026, 41 labeled logs):

| Metric | Value |
|---|---|
| Best mIoU | **0.758** (epoch 67) |
| Bark IoU | **0.543** |
| Accuracy | **0.973** |
| Training time | ~77 min (CPU) |

---

## How It Works

1. Export log scans from Blender as `.ply` with material labels (`madera` / `corteza`).
2. Place files in `data/raw/`.
3. Train PointNet++ to classify each point as wood (0) or bark (1).
4. Run inference on new logs to get a color-coded segmented cloud and bark area estimate.

---

## Requirements

- Python 3.10+
- CPU only (no GPU required)
- Docker + VS Code Dev Containers (recommended)

Key dependencies: `torch==2.2.2`, `open3d==0.18.0`, `numpy==1.26.4`

---

## Installation

### Option A: Dev Container (recommended)

1. Open the repository in VS Code.
2. Run **Dev Containers: Reopen in Container**.
3. Use the integrated terminal inside the container.

### Option B: Local virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r .devcontainer/requirements.txt
```

If `python -m venv` fails in WSL/Debian due to missing `ensurepip`:

```bash
python3 -m pip install --user --break-system-packages virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
```

---

## Quick Start

```bash
git clone https://github.com/jediros/3D-Logs-Segmentation.git
cd 3D-Logs-Segmentation
```

```bash
# 1. Check your dataset
python main.py info

# 2. Train
python main.py train

# 3. Segment a new log
python main.py infer --input data/raw/new_log.ply --visualize
```

---

## Command Reference

| Command | Description |
|---|---|
| `python main.py info` | Analyze PLY files and show label statistics |
| `python main.py train` | Train the model |
| `python main.py train --resume` | Resume training from last checkpoint |
| `python main.py infer --input <file.ply>` | Segment a log |
| `python main.py infer --input <file.ply> --visualize` | Segment and open 3D viewer |
| `python main.py visualize --file <file.ply>` | Visualize a point cloud with labels |
| `python main.py preprocess` | Pre-process PLY → .npy cache (optional) |
| `python main.py preprocess --overwrite` | Rebuild all cached files |

Global options must come before the subcommand:

```bash
python main.py --config config/smoke.yaml train
```

---

## Blender Export

1. Import the scan into Blender.
2. Assign two materials to faces: `madera` (wood) and `corteza` (bark).
3. Export as Stanford PLY with attributes included.

The loader expects label fields such as `label_1` or `corteza`. Label conversion uses threshold 0.5.

---

## Preprocessing Cache

For large datasets, pre-process once to `.npy` for faster training:

```bash
python main.py preprocess
```

Then enable caching in `config/default.yaml`:

```yaml
preprocessing:
  use_cache: true
```

Run `preprocess` again if you add new `.ply` files. Use `--overwrite` to rebuild all cached files.

---

## Architecture

PointNet++ encoder-decoder for binary segmentation.

- **Input:** `(B, N, C)` — C = 9 (xyz + normals + rgb)
- **Encoder:** 3× SetAbstraction blocks
- **Decoder:** 3× FeaturePropagation blocks + MLP head + dropout
- **Output:** `(B, N, 2)` logits

Training: focal loss (γ=2), class weighting, Adam optimizer, StepLR decay, gradient clipping.

---

## Configuration

Main config: `config/default.yaml`

| Parameter | Value |
|---|---|
| `preprocessing.num_points` | 16384 |
| `model.use_normals` | true |
| `model.use_rgb` | true |
| `training.epochs` | 100 |
| `training.batch_size` | 4 |
| `training.learning_rate` | 0.001 |
| `training.val_split` | 0.2 |
| `training.class_weights` | [1.0, 10.0] |

---

## Outputs

After inference:

- `outputs/<log_name>_segmented.ply` — color-coded point cloud (green=wood, red=bark)
- Console summary with bark points, wood points, and bark ratio

Analysis scripts for visual comparison are in `extras/`.

---

## Metrics Tracked

- Per-class IoU (wood, bark)
- Mean IoU
- Precision, Recall, F1
- Accuracy

Training logs: `training/logs/train_log.csv`

---

## Project Layout

```
main.py
config/         # YAML configs
data/           # Raw and processed point clouds (not committed)
preprocessing/  # Sampling and normalization
model/          # PointNet++ architecture
training/       # Trainer, checkpoints, logs
inference/      # Predictor for new logs
utils/          # Metrics, logger, visualizer
tests/          # Unit tests
extras/         # Analysis and comparison scripts
.devcontainer/  # Docker environment
```

---

## Testing

```bash
pytest tests/ -v
```

---

## License

MIT License.
