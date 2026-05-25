import csv
import time
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir, config_summary=None, mlflow_cfg=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time       = time.time()
        self.best_miou        = 0.0
        self.csv_path         = self.log_dir / "train_log.csv"
        self._csv_initialized = False
        self._mlflow          = None

        if config_summary:
            with open(self.log_dir / "run_info.txt", "w") as f:
                f.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for k, v in config_summary.items():
                    f.write(f"{k}: {v}\n")

        if mlflow_cfg and getattr(mlflow_cfg, "enabled", False):
            self._start_mlflow(mlflow_cfg, config_summary or {})

        print(f"\nLogs in: {self.log_dir}")
        if self._mlflow:
            print(f"MLflow:  {self._mlflow['tracking_uri']}  "
                  f"(experiment: {self._mlflow['experiment_name']})")
        print("-" * 70)

    def _start_mlflow(self, cfg, params):
        try:
            import mlflow
            uri = getattr(cfg, "tracking_uri", "") or "mlruns"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(getattr(cfg, "experiment_name", "bark-segmentation"))
            mlflow.start_run()
            if params:
                mlflow.log_params(params)
            self._mlflow = {
                "tracking_uri":    uri,
                "experiment_name": getattr(cfg, "experiment_name", "bark-segmentation"),
            }
        except Exception as e:
            print(f"  [MLflow] Could not start run: {e}")
            self._mlflow = None

    def log_epoch(self, epoch, total_epochs, train_loss, val_loss, val_metrics, lr):
        miou     = val_metrics.get("miou", 0.0)
        accuracy = val_metrics.get("accuracy", 0.0)
        iou_map  = val_metrics.get("iou", {})
        bark_iou = iou_map.get("bark", iou_map.get("corteza", 0.0))
        is_best  = miou > self.best_miou
        if is_best:
            self.best_miou = miou
        elapsed = (time.time() - self.start_time) / 60
        marker  = " *" if is_best else ""
        print(f"  Epoch {epoch:3d}/{total_epochs} | "
              f"loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"mIoU={miou:.4f} bark_IoU={bark_iou:.4f} acc={accuracy:.4f} | "
              f"lr={lr:.6f} | {elapsed:.1f}min{marker}")

        row = {"epoch": epoch, "train_loss": round(train_loss,6),
               "val_loss": round(val_loss,6), "miou": round(miou,6),
               "bark_iou": round(bark_iou,6), "accuracy": round(accuracy,6),
               "lr": round(lr,8), "best_miou": round(self.best_miou,6),
               "elapsed_min": round(elapsed,2)}
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=row.keys()).writeheader()
            self._csv_initialized = True
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

        if self._mlflow:
            try:
                import mlflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    "miou":       miou,
                    "bark_iou":   bark_iou,
                    "accuracy":   accuracy,
                    "lr":         lr,
                    "best_miou":  self.best_miou,
                }, step=epoch)
            except Exception:
                pass

        return is_best

    def finalize(self):
        mins = (time.time() - self.start_time) / 60
        print(f"\n{chr(9472)*70}")
        print(f"  Done in {mins:.1f} min  |  Best mIoU: {self.best_miou:.4f}")
        print(f"  Log: {self.csv_path}")
        if self._mlflow:
            try:
                import mlflow
                mlflow.log_metric("final_best_miou", self.best_miou)
                mlflow.end_run()
                print(f"  MLflow run ended.")
            except Exception:
                pass
