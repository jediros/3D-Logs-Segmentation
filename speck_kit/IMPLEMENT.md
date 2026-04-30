# IMPLEMENT — 3D Log Bark Segmentation

> Practical execution guide for the current repository.
> Use it as a reusable checklist when standing up similar projects.

---

## 1. Environment Validation

All commands assume repository root.
The preferred environment is the VS Code Dev Container, but a local virtual environment also works.

### IMPL-01 · Verify the Python environment

```bash
python --version
python -c "import torch, numpy, open3d, yaml, tqdm; print('OK')"
python -c "from data.loader import load_ply_labeled; print('imports OK')"
```

Expected:
- Python 3.10.x inside container, or a supported local interpreter
- imports succeed

### IMPL-02 · Verify data visibility

```bash
python main.py info
```

Expected:
- per-file summary
- total summary row
- warning list for bark-free files
- final readiness line

### IMPL-03 · Verify tests

```bash
pytest tests/ -v
```

Expected:
- current repository baseline is 17 passing tests

---

## 2. Smoke Validation

### IMPL-04 · Run a cheap training check

```bash
python main.py --config config/smoke.yaml train
```

Expected:
- training starts on CPU
- log files are written
- checkpoints are created

Why this matters:
- it validates the full training path without mutating the main config

### IMPL-05 · Validate inference from checkpoint

```bash
python main.py infer --input data/raw/YOUR_LOG.ply
```

Expected:
- bark and wood counts are printed
- segmented PLY is written to `outputs/`

---

## 3. Full Training Workflow

### IMPL-06 · Review the main config

Check the following before long training:
- `model.use_normals`
- `model.use_rgb`
- `model.num_points`
- `training.epochs`
- `training.class_weights`

### IMPL-07 · Launch training

```bash
python main.py train
```

Expected:
- CPU training loop runs
- validation metrics are logged every epoch
- `best_model.pth` updates when mIoU improves

### IMPL-08 · Resume if needed

```bash
python main.py train --resume
```

---

## 4. Visualization Workflow

### IMPL-09 · Inspect a labeled file

```bash
python main.py visualize --file data/raw/YOUR_LOG.ply
```

### IMPL-10 · Inspect inference results

```bash
python main.py infer --input data/raw/YOUR_LOG.ply --visualize
```

Notes:
- visualization requires a display-capable environment
- compare scripts may depend on additional runtime libraries and are safest inside the Dev Container

---

## 5. Reusable Checklist for Future Architectures

When cloning this pattern for another project, keep these checkpoints:

1. Add a smoke config before the first long training run.
2. Make checkpoints self-describing.
3. Separate train and validation datasets if behavior flags differ.
4. Validate CLI, training, and inference independently.
5. Keep one authoritative current-state document and one future-work backlog.