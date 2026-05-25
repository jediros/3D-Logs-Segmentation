import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from config.config_loader import load_config
from data.dataset import BarkDataset
from model.pointnet2 import PointNet2Segmentation, build_loss
from utils.metrics import SegmentationMetrics
from utils.logger import TrainingLogger


def save_checkpoint(state, ckpt_dir, filename):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(ckpt_dir) / filename)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
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
    metrics = SegmentationMetrics(num_classes, class_names=["wood", "bark"])
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
    _dev = getattr(cfg.training, "device", "auto")
    if _dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(_dev)
    use_rgb = getattr(cfg.model, "use_rgb", False)
    use_cache = getattr(cfg.preprocessing, "use_cache", False)

    print(f"\nDevice: {device}")
    print(f"Features: xyz"
          + (" + normals" if cfg.model.use_normals else "")
          + (" + RGB" if use_rgb else ""))
    if use_cache:
        print(f"Using cached .npy from {cfg.paths.processed_data}")

    full_ds = BarkDataset(
        ply_dir=cfg.paths.raw_data,
        num_points=cfg.model.num_points,
        use_normals=cfg.model.use_normals,
        use_rgb=use_rgb,
        ignore_boundary=cfg.preprocessing.ignore_boundary,
        cache=True,
        use_cache_dir=use_cache,
        cache_dir=cfg.paths.processed_data if use_cache else None,
    )
    full_ds.summary()

    n_total = len(full_ds)
    n_val   = max(1, int(n_total * cfg.training.val_split))
    n_train = n_total - n_val
    if n_train < 1:
        raise RuntimeError(
            f"Dataset too small ({n_total}). Need at least 2 logs.")

    train_split, val_split = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.training.seed))

    # Use independent dataset instances so validation never receives augmentation.
    train_base = BarkDataset(
        ply_dir=cfg.paths.raw_data,
        num_points=cfg.model.num_points,
        augment=True,
        use_normals=cfg.model.use_normals,
        use_rgb=use_rgb,
        ignore_boundary=cfg.preprocessing.ignore_boundary,
        cache=True,
        use_cache_dir=use_cache,
        cache_dir=cfg.paths.processed_data if use_cache else None,
    )
    val_base = BarkDataset(
        ply_dir=cfg.paths.raw_data,
        num_points=cfg.model.num_points,
        augment=False,
        use_normals=cfg.model.use_normals,
        use_rgb=use_rgb,
        ignore_boundary=cfg.preprocessing.ignore_boundary,
        cache=True,
        use_cache_dir=use_cache,
        cache_dir=cfg.paths.processed_data if use_cache else None,
    )

    train_ds = Subset(train_base, train_split.indices)
    val_ds = Subset(val_base, val_split.indices)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True,  num_workers=cfg.training.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=cfg.training.num_workers)

    model = PointNet2Segmentation(
        num_classes=cfg.model.num_classes,
        use_normals=cfg.model.use_normals,
        use_rgb=use_rgb,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    weights = (full_ds.get_class_weights()
               if cfg.training.class_weights is None
               else torch.tensor(cfg.training.class_weights, dtype=torch.float32))
    criterion = build_loss(class_weights=weights.to(device), use_focal=True)

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.training.learning_rate,
                           weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.lr_decay_step,
        gamma=cfg.training.lr_decay)

    start_epoch, ckpt_dir = 1, Path(cfg.paths.checkpoints)
    if resume and (ckpt_dir / "last_checkpoint.pth").exists():
        ckpt = load_checkpoint(ckpt_dir / "last_checkpoint.pth", model, optimizer, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1

    logger = TrainingLogger(cfg.paths.logs, config_summary={
        "epochs":        cfg.training.epochs,
        "batch_size":    cfg.training.batch_size,
        "lr":            cfg.training.learning_rate,
        "num_points":    cfg.model.num_points,
        "use_rgb":       use_rgb,
        "class_weights": str(cfg.training.class_weights),
        "device":        str(device),
        "n_train":       n_train,
        "n_val":         n_val,
    }, mlflow_cfg=getattr(cfg, "mlflow", None))

    print(f"\nTraining {cfg.training.epochs} epochs on {device}...")
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
                "use_rgb":     use_rgb,
                "num_points":  cfg.model.num_points,
            },
        }
        save_checkpoint(state, ckpt_dir, "last_checkpoint.pth")
        if is_best:
            save_checkpoint(state, ckpt_dir, "best_model.pth")
            print(f"    -> Best model (mIoU={logger.best_miou:.4f})")
        scheduler.step()

    logger.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)