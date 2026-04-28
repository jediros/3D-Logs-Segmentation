import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config.config_loader import load_config
from data.dataset import BarkDataset
from model.pointnet2 import PointNet2Segmentation, build_loss
from utils.metrics import SegmentationMetrics
from utils.logger import TrainingLogger


def save_checkpoint(state, ckpt_dir, filename):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(ckpt_dir) / filename)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Checkpoint: epoch={ckpt.get('epoch','?')} "
          f"best_miou={ckpt.get('best_miou',0):.4f}")
    return ckpt


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for pts, lbl in tqdm(loader, desc="  Train", leave=False, ncols=80):
        pts, lbl = pts.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = model(pts)
        B, N, C = logits.shape
        loss = criterion(logits.contiguous(), lbl.contiguous())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def evaluate(model, loader, criterion, device, num_classes=2):
    model.eval()
    metrics = SegmentationMetrics(num_classes, class_names=["madera", "corteza"])
    total, n = 0.0, 0
    with torch.no_grad():
        for pts, lbl in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
            pts, lbl = pts.to(device), lbl.to(device)
            logits   = model(pts)
            B, N, C  = logits.shape
            total += criterion(logits.contiguous(), lbl.contiguous()).item()
            n       += 1
            metrics.update(logits.argmax(-1), lbl)
    return total / max(n, 1), metrics.compute()


def train(cfg, resume=False):
    torch.manual_seed(cfg.training.seed)
    device  = torch.device("cpu")
    use_rgb = getattr(cfg.model, "use_rgb", False)

    print(f"\nDispositivo: {device}")
    print(f"Features: xyz"
          + (" + normales" if cfg.model.use_normals else "")
          + (" + RGB" if use_rgb else ""))

    full_ds = BarkDataset(
        ply_dir=cfg.paths.raw_data,
        num_points=cfg.model.num_points,
        use_normals=cfg.model.use_normals,
        use_rgb=use_rgb,
        ignore_boundary=cfg.preprocessing.ignore_boundary,
        cache=True,
    )
    full_ds.summary()

    n_total = len(full_ds)
    n_val   = max(1, int(n_total * cfg.training.val_split))
    n_train = n_total - n_val
    if n_train < 1:
        raise RuntimeError(
            f"Dataset muy pequeño ({n_total}). Necesitas al menos 2 troncos.")

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.training.seed))
    train_ds.dataset.augment = True

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True,  num_workers=cfg.training.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.training.num_workers)

    model = PointNet2Segmentation(
        num_classes=cfg.model.num_classes,
        use_normals=cfg.model.use_normals,
        use_rgb=use_rgb,
    ).to(device)
    print(f"Parametros: {model.count_parameters():,}")

    weights = (full_ds.get_class_weights()
               if cfg.training.class_weights is None
               else torch.tensor(cfg.training.class_weights, dtype=torch.float32))
    criterion = build_loss(class_weights=weights, use_focal=True)

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.training.learning_rate,
                           weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.lr_decay_step,
        gamma=cfg.training.lr_decay)

    start_epoch, ckpt_dir = 1, Path(cfg.paths.checkpoints)
    if resume and (ckpt_dir / "last_checkpoint.pth").exists():
        ckpt = load_checkpoint(ckpt_dir / "last_checkpoint.pth", model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1

    logger = TrainingLogger(cfg.paths.logs, config_summary={
        "epochs":      cfg.training.epochs,
        "batch_size":  cfg.training.batch_size,
        "lr":          cfg.training.learning_rate,
        "num_points":  cfg.model.num_points,
        "use_rgb":     use_rgb,
        "n_train":     n_train,
        "n_val":       n_val,
    })

    print(f"\nEntrenando {cfg.training.epochs} epochs en CPU...")
    print("-" * 70)

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, vm = evaluate(model, val_loader, criterion, device,
                          cfg.model.num_classes)
        lr = optimizer.param_groups[0]["lr"]
        is_best = logger.log_epoch(epoch, cfg.training.epochs, tl, vl, vm, lr)

        state = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_miou":       logger.best_miou,
            "val_metrics":     vm,
            "config": {
                "num_classes": cfg.model.num_classes,
                "use_normals": cfg.model.use_normals,
                "use_rgb":     use_rgb,          # NUEVO
                "num_points":  cfg.model.num_points,
            },
        }
        save_checkpoint(state, ckpt_dir, "last_checkpoint.pth")
        if is_best:
            save_checkpoint(state, ckpt_dir, "best_model.pth")
            print(f"    -> Mejor modelo (mIoU={logger.best_miou:.4f})")
        scheduler.step()

    logger.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)