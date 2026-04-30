# 3D Log Bark Segmentation

PointNet++ semantic segmentation of residual bark on debarked logs.

This repository provides a modular Python pipeline to classify each point of a log surface as wood or bark, and estimate bark area from 3D point clouds.

## Current Status (April 2026)

- Core pipeline is working end-to-end: info -> train -> infer.
- Unit tests are passing: 17/17 in `tests/test_preprocessing.py`.
- Train/validation split behavior was hardened so validation does not use augmentation.
- Dependency pin was updated to `open3d==0.19.0` for Python 3.12 compatibility.
- Dev Container build now installs dependencies from `.devcontainer/requirements.txt`.
- Dataset policy is intentionally permissive: logs with no bark are kept (important for learning bark-free surfaces).

## What This Project Does

- Input: binary PLY files exported from Blender (with material label fields).
- Output: per-point prediction for two classes.
  - 0 = wood
  - 1 = bark
- Extra output: bark fraction and area estimation summary.

## Requirements

- Python 3.10 or newer (3.10 recommended in Dev Container)
- CPU only is supported by default
- Optional: Docker + VS Code Dev Containers for reproducible setup

## Installation

### Option A: Dev Container (recommended)

1. Open the repository in VS Code.
2. Run Dev Containers: Reopen in Container.
3. Use the integrated terminal inside the container.

### Option B: Local virtual environment

    python -m venv .venv
    source .venv/bin/activate
    pip install -r .devcontainer/requirements.txt

If `python -m venv` fails in WSL/Debian because `ensurepip` is missing, use:

    python3 -m pip install --user --break-system-packages virtualenv
    python3 -m virtualenv .venv
    source .venv/bin/activate

## Quick Start

1. Clone repository and enter folder:

    git clone <your-repository-url>
    cd 3D-Logs-Segmentation

2. Put your PLY files in data/raw:

    mkdir -p data/raw
    cp /path/to/your/scans/*.ply data/raw/

3. Inspect dataset status:

    python main.py info

4. Train the model:

    python main.py train

5. Run inference on a new log:

    python main.py infer --input data/raw/new_log.ply --visualize

Note: global options must be declared before the subcommand. Example:

    python main.py --config config/default.yaml train

## Blender Export Notes

Recommended workflow:

1. Import scan into Blender.
2. Create two materials for faces:
   - madera for wood
   - corteza for bark
3. Export as Stanford PLY with attributes included.

The loader expects label fields such as label_1 (or corteza). Label conversion uses threshold 0.5 by default.

## Command Reference

    python main.py info
    python main.py train
    python main.py train --resume
    python main.py infer --input data/raw/new_log.ply
    python main.py infer --input data/raw/new_log.ply --visualize
    python main.py visualize --file data/raw/example.ply
    python main.py preprocess --overwrite

**Offline Preprocessing (TASK-20)**:

To accelerate training with large datasets, preprocess all PLY files to cached .npy format:

    python main.py preprocess --overwrite

Then enable caching in `config/default.yaml`:

    preprocessing:
      use_cache: true

On subsequent runs with `use_cache: true`, the trainer loads from `data/processed/*.npy` instead of loading and normalizing PLY files each epoch. This saves I/O and computation.

If you add new `.ply` files later, run preprocessing again before training with cache enabled:

    python main.py preprocess

This runs incrementally (processes only missing cache files).
Use `--overwrite` only when you need to rebuild all cached files.

## Validation and Testing

- Run unit tests:

    pytest tests/ -v

- Quick data health check:

    python main.py info

- Smoke training with custom config:

    python main.py --config config/smoke.yaml train

## Current Architecture

PointNet++ encoder-decoder for binary segmentation.

- Input tensor shape: (B, N, C)
  - C = 3 for xyz
  - C = 6 for xyz+normals or xyz+rgb
  - C = 9 for xyz+normals+rgb
- Core blocks:
  - SetAbstraction x3
  - FeaturePropagation x3
  - MLP head + dropout
- Output shape: (B, N, 2) logits

Training uses focal loss (gamma 2), class weighting, Adam, StepLR, and gradient clipping.

Train/validation note:

- Training uses augmentation.
- Validation uses a separate dataset instance with augmentation disabled.

## Configuration

Main configuration file: config/default.yaml

Current defaults include:

- preprocessing.num_points: 16384
- preprocessing.ignore_boundary: true
- model.use_normals: true
- model.use_rgb: true
- training.epochs: 250
- training.batch_size: 4
- training.learning_rate: 0.001
- training.val_split: 0.2
- training.class_weights: [1.0, 15.0]

The project supports custom experiment configs by passing `--config` before the command.

## Outputs

Inference writes results to outputs:

- <log_name>_segmented.ply with color-coded predictions
- Console summary with bark points, wood points, and bark ratio
- Optional HTML comparison reports from `compare_*.py` scripts

Area estimation is computed in utils/metrics.py.

## Metrics

The project tracks:

- per-class IoU
- mean IoU
- precision
- recall
- F1
- accuracy

Training logs are saved to training/logs/train_log.csv.

## Project Layout

    main.py
    config/
    data/
    preprocessing/
    model/
    training/
    inference/
    utils/
    tests/
    .devcontainer/

Additional analysis scripts in repository root:

- `compare_inference.py`
- `compare_inference_3.py`
- `compare_inference_3_1.py`
- `compare_lite.py`
- `compare_mesh_lite.py`
- `compare_solid_mesh.py`

## Practical Notes

- Use python main.py info before training to verify label quality and file integrity.
- If a file has no bark labels, review it in Blender before including it in experiments.
- Logs with no bark are not auto-removed by dataset filtering (intentional behavior).
- Runtime depends on dataset size, number of points, and epoch count.

Environment caveat for local (non-container) runs:

- `compare_*.py` scripts require Open3D runtime system libraries.
- If you see `libgomp.so.1` errors, run inside the Dev Container or install the missing system library in your host environment.

## License

MIT License.
