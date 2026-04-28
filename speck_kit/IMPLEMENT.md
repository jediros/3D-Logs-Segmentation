# IMPLEMENT — 3D Log Bark Segmentation

> Step-by-step execution guide.
> This document tells **who does what, when, and exactly how**.
> Follow phases in order. Do not skip validation steps.

---

## How to Use This Document

Each step has:
- **Command(s)** to run exactly as written
- **Expected output** — what success looks like
- **If it fails** — what to check

All commands assume you are inside the DevContainer terminal in VS Code,
working directory `/workspace` (= your project root).

---

## Phase 1 — Environment Setup

### IMPL-01 · Verify DevContainer is running

```bash
python --version
```
**Expected**: `Python 3.10.x`

```bash
python -c "import torch, numpy, open3d, yaml, tqdm; print('OK')"
```
**Expected**: `OK`

```bash
python -c "from data.loader import load_ply_labeled; print('imports OK')"
```
**Expected**: `imports OK`

**If imports fail**: run `pip install -e .` and retry.

---

### IMPL-02 · Verify PLY files are in place

```bash
ls -lh data/raw/*.ply | wc -l
```
**Expected**: number of your PLY files (e.g. `19`)

```bash
python main.py info
```
**Expected output format**:
```
Archivos PLY encontrados en data/raw: 19

  Archivo                          Verts  Corteza   Madera  %Corteza  Labels
  ─────────────────────────────────────────────────────────────────────────
  18B_decimated.ply                 4148      474     3674     11.4%      SI
  7B_800N_decimated.ply             4615      216     4399      4.7%      SI
  ...
  19A_800N_decimated.ply            4341        0     4341      0.0%      NO

  Listos para entrenar: X/19 troncos
```

**If a file shows 0 corteza**: it needs re-labeling in Blender (see IMPL-06).

---

### IMPL-03 · Run test suite

```bash
pytest tests/ -v
```
**Expected**: all tests pass (green). Zero failures.

**If tests fail**: check error message, fix the specific module, re-run.
Do not proceed to training with failing tests.

---

## Phase 2 — First Training Run (Smoke Test)

### IMPL-04 · Reduce epochs for smoke test

Edit `config/default.yaml` — change one line:
```yaml
training:
  epochs: 3      # ← change from 100 to 3
```

### IMPL-05 · Run smoke test

```bash
python main.py train
```

**Expected output**:
```
Dispositivo: cpu
Dataset: X troncos (Y train, Z val)
Parametros: 571,xxx

Logs en: training/logs
----------------------------------------------------------------------
  Epoch   1/3 | loss train=X.XXXX val=X.XXXX | mIoU=X.XXXX bark_IoU=X.XXXX ...
  Epoch   2/3 | ...
  Epoch   3/3 | ...
  -> Mejor modelo guardado (mIoU=X.XXXX)

Listo en X.X min  |  Mejor mIoU: X.XXXX
```

**Verify checkpoint was saved**:
```bash
ls -lh training/checkpoints/
```
**Expected**: `best_model.pth` and `last_checkpoint.pth`

**Verify log was created**:
```bash
cat training/logs/train_log.csv
```
**Expected**: CSV with 3 rows (one per epoch)

**If training crashes with memory error**: reduce `batch_size` to `2` in config.
**If training crashes with shape error**: paste the traceback and check SPEC-03.

---

### IMPL-06 · Restore full epochs

Edit `config/default.yaml`:
```yaml
training:
  epochs: 100    # ← restore
```

---

## Phase 3 — Label Remaining Logs in Blender

### IMPL-07 · Identify unlabeled logs

```bash
python main.py info
```
Note every file with `Labels: NO`. These need Blender treatment.

### IMPL-08 · Re-export unlabeled logs from Blender

For each unlabeled log:

1. Open Blender with the original `.obj` file of that log
2. Select the mesh object
3. Open the **Properties panel** → **Material Properties** (sphere icon)
4. Verify two material slots exist: one for wood, one for bark
   - If missing: click `+` to add, name them `madera` and `corteza`
5. Go to **Edit Mode** (`Tab`)
6. Select all faces that are bark: use `Select → Select All by Trait → Material` on the bark material
   - If not yet assigned: manually select bark faces, click `Assign` on the bark material slot
7. Exit Edit Mode (`Tab`)
8. `File → Export → Stanford PLY (.ply)`
   - Enable: **Include Attributes** ✓
   - Enable: **Triangulate Faces** ✓
   - Save with the same filename to `data/raw/`
9. Re-run `python main.py info` to verify `Labels: SI`

**Verify**:
```bash
python main.py info
```
The file should now show `Labels: SI` with n_bark > 0.

---

## Phase 4 — Full Training

### IMPL-09 · Check dataset is ready

```bash
python main.py info
```
**Minimum before full training**:
- At least 5 logs with `Labels: SI`
- At least one log with bark% > 5%

### IMPL-10 · Launch full training

```bash
python main.py train
```

Training runs for 100 epochs. Expected time: 1–8 hours on CPU.

**Monitor progress** in a second terminal:
```bash
# Watch the CSV log update in real time
watch -n 30 tail -5 training/logs/train_log.csv
```

**Signs of healthy training**:
- `train_loss` decreasing over first 20 epochs
- `bark_IoU` above 0.0 by epoch 5
- `bark_IoU` above 0.3 by epoch 40 (with 5+ logs)

**Signs of problems**:
- `bark_IoU` stays at 0.0 after epoch 10 → increase class weight for bark
- `train_loss` oscillates wildly → reduce learning rate to 0.0005
- `val_loss` much higher than `train_loss` → overfitting, add more logs

### IMPL-11 · Resume interrupted training

If training is interrupted for any reason:
```bash
python main.py train --resume
```
Continues from the last saved checkpoint automatically.

---

## Phase 5 — Inference

### IMPL-12 · Segment a labeled log (validation)

```bash
python main.py infer --input data/raw/18B_decimated.ply
```

**Expected**:
```
Segmentando: 18B_decimated.ply
  Corteza: XXX pts (XX.X%)
  Madera:  XXXX pts
  Guardado: outputs/18B_decimated_segmented.ply
```

Compare predicted bark% with ground truth from `python main.py info`.
They will not match perfectly — this is expected.

### IMPL-13 · Visual inspection

```bash
python main.py infer --input data/raw/18B_decimated.ply --visualize
```

An Open3D window opens showing the segmented log:
- **Brown** = wood
- **Dark brown** = bark

Rotate with left mouse button, pan with middle button, zoom with scroll.

### IMPL-14 · Segment a new (unlabeled) log

```bash
python main.py infer --input data/raw/NEW_LOG.ply --visualize
```

If you know the real surface area of the log:
```bash
python main.py infer --input data/raw/NEW_LOG.ply --area 0.45
```
Where `0.45` is the surface area in m².

### IMPL-15 · Open segmented result in CloudCompare

```bash
# The output .ply can be opened externally
cloudcompare outputs/18B_decimated_segmented.ply
```

Or open in MeshLab:
```bash
meshlab outputs/18B_decimated_segmented.ply
```

---

## Phase 6 — Analysis and Reporting

### IMPL-16 · Plot training curves

```bash
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training/logs/train_log.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(df['epoch'], df['train_loss'], label='train')
axes[0].plot(df['epoch'], df['val_loss'],   label='val')
axes[0].set_title('Loss'); axes[0].legend()

axes[1].plot(df['epoch'], df['miou'])
axes[1].set_title('mIoU')

axes[2].plot(df['epoch'], df['bark_iou'])
axes[2].set_title('Bark IoU')

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=150)
print('Saved: outputs/training_curves.png')
"
```

### IMPL-17 · Print final metrics

```bash
python -c "
import pandas as pd
df = pd.read_csv('training/logs/train_log.csv')
best = df.loc[df['miou'].idxmax()]
print(f'Best epoch:   {int(best.epoch)}')
print(f'Best mIoU:    {best.miou:.4f}')
print(f'Best bark IoU:{best.bark_iou:.4f}')
print(f'Best accuracy:{best.accuracy:.4f}')
"
```

---

## Troubleshooting Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | Project not installed | `pip install -e .` |
| `FileNotFoundError: PLY` | Wrong working directory | `cd /workspace` |
| `RuntimeError: Dataset muy pequeño` | < 2 labeled logs | Label more logs in Blender |
| `bark_IoU = 0.000` after 20 epochs | Class imbalance too severe | Set `class_weights: [1.0, 8.0]` in config |
| Training very slow (> 5 min/epoch) | num_points too high | Set `num_points: 2048` in config |
| Open3D visualization fails | No display in container | Use `--no-ply` flag, open .ply externally |
| `JSON decode error` in devcontainer | devcontainer.json corrupted | Rewrite with Python (see setup guide) |
| `cat >` inside Python file | heredoc wrote command literally | Rewrite file with `Path.write_text()` |

---

## Checklist Before Sharing / Publishing

```
[ ] pytest tests/ -v  → all passing
[ ] python main.py info → all training logs have Labels: SI
[ ] README.md updated with your actual results (bark IoU, dataset size)
[ ] data/raw/ is in .gitignore and NOT pushed to GitHub
[ ] training/checkpoints/ is in .gitignore
[ ] BibTeX citation updated with your name and year
[ ] config/default.yaml reflects your final hyperparameters
```
