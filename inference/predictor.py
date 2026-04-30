"""
inference/predictor.py
----------------------
Loads a trained model and segments new logs from .ply files.
Supports models trained with or without RGB — reads config from checkpoint.
"""

from pathlib import Path
import numpy as np
import torch

from data.loader import load_ply_labeled
from preprocessing.sampler import normalize_pointcloud
from model.pointnet2 import PointNet2Segmentation
from utils.metrics import compute_bark_area
from utils.visualizer import visualize_segmentation, save_colored_cloud


class BarkPredictor:
    """
    Bark/wood segmentation predictor from .ply files.

    Automatically reconstructs the correct architecture from the checkpoint
    (with or without RGB, with or without normals).
    """

    def __init__(self, model, num_points=4096, use_normals=True, use_rgb=False):
        self.model       = model
        self.num_points  = num_points
        self.use_normals = use_normals
        self.use_rgb     = use_rgb
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path) -> "BarkPredictor":
        """
        Loads predictor from .pth checkpoint.
        Reads use_rgb, use_normals and num_points directly from the checkpoint.
        Does not require access to config/default.yaml.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt     = torch.load(path, map_location="cpu")
        cfg      = ckpt.get("config", {})
        use_rgb  = cfg.get("use_rgb", False)     # backward-compatible with checkpoints without RGB

        model = PointNet2Segmentation(
            num_classes=cfg.get("num_classes", 2),
            use_normals=cfg.get("use_normals", True),
            use_rgb=use_rgb,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        rgb_info = " +RGB" if use_rgb else ""
        print(f"Model loaded: {path.name}")
        print(f"  Features: xyz"
              + (" + normals" if cfg.get("use_normals", True) else "")
              + rgb_info)
        print(f"  Best val mIoU: {ckpt.get('best_miou', 0):.4f}")

        return cls(
            model=model,
            num_points=cfg.get("num_points", 4096),
            use_normals=cfg.get("use_normals", True),
            use_rgb=use_rgb,
        )

    def predict_cloud(self, cloud: np.ndarray) -> np.ndarray:
        """
        Predicts labels for a normalized point cloud.

        Args:
            cloud: (N, 3), (N, 6) or (N, 9) depending on active features

        Returns:
            labels: (N,) int32   0=wood  1=bark
        """
        N = len(cloud)
        idx = (np.random.choice(N, self.num_points, replace=False)
               if N >= self.num_points
               else np.concatenate([np.arange(N),
                    np.random.choice(N, self.num_points - N, replace=True)]))

        tensor = torch.from_numpy(cloud[idx].astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            preds = self.model(tensor).argmax(-1).squeeze(0).numpy()

        if N > self.num_points:
            from scipy.spatial import cKDTree
            _, nb = cKDTree(cloud[idx, :3]).query(cloud[:, :3], k=1)
            return preds[nb].astype(np.int32)

        return preds[:N].astype(np.int32)

    def predict_ply(
        self,
        ply_path,
        save_ply:   bool = True,
        output_dir       = "outputs",
        visualize:  bool = False,
    ) -> dict:
        """
        Full pipeline: .ply -> segmentation -> bark area.

        Args:
            ply_path:   path to the .ply file (with or without labels)
            save_ply:   save colored .ply cloud in output_dir
            output_dir: results folder
            visualize:  open Open3D 3D window

        Returns:
            dict with bark_fraction, n_bark_points, n_wood_points, labels, pts
        """
        ply_path   = Path(ply_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSegmenting: {ply_path.name}")

        # Load with the same features the model expects
        cloud, _, meta = load_ply_labeled(
            ply_path,
            compute_normals=self.use_normals,
            use_rgb=self.use_rgb,
        )

        if meta.get("has_rgb") is False and self.use_rgb:
            print(f"  [WARNING] Model was trained with RGB but this PLY "
                  f"has no RGB fields. Inference may be less accurate.")

        cloud_norm = normalize_pointcloud(cloud)
        labels     = self.predict_cloud(cloud_norm)
        results    = compute_bark_area(cloud[:, :3], labels)
        results.update({
            "labels":   labels,
            "pts":      cloud[:, :3],
            "ply_path": str(ply_path),
        })

        print(f"  Bark:  {results['n_bark_points']:,} pts "
              f"({results['bark_fraction']*100:.1f}%)")
        print(f"  Wood:  {results['n_wood_points']:,} pts")

        if save_ply:
            out = output_dir / (ply_path.stem + "_segmented.ply")
            save_colored_cloud(cloud[:, :3], labels, out)

        if visualize:
            visualize_segmentation(cloud[:, :3], labels, title=ply_path.stem)

        return results