"""
inference/predictor.py
----------------------
Carga modelo entrenado y segmenta troncos nuevos desde .ply.
Soporta modelos entrenados con o sin RGB — lee la config del checkpoint.
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
    Predictor de segmentacion corteza/madera desde .ply.

    Reconstruye automaticamente la arquitectura correcta desde el checkpoint
    (con o sin RGB, con o sin normales).
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
        Carga predictor desde .pth.
        Lee use_rgb, use_normals y num_points directamente del checkpoint.
        No necesita acceso al config/default.yaml.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {path}")

        ckpt     = torch.load(path, map_location="cpu")
        cfg      = ckpt.get("config", {})
        use_rgb  = cfg.get("use_rgb", False)     # retrocompatible con checkpoints sin RGB

        model = PointNet2Segmentation(
            num_classes=cfg.get("num_classes", 2),
            use_normals=cfg.get("use_normals", True),
            use_rgb=use_rgb,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        rgb_info = " +RGB" if use_rgb else ""
        print(f"Modelo cargado: {path.name}")
        print(f"  Features: xyz"
              + (" + normales" if cfg.get("use_normals", True) else "")
              + rgb_info)
        print(f"  Mejor mIoU val: {ckpt.get('best_miou', 0):.4f}")

        return cls(
            model=model,
            num_points=cfg.get("num_points", 4096),
            use_normals=cfg.get("use_normals", True),
            use_rgb=use_rgb,
        )

    def predict_cloud(self, cloud: np.ndarray) -> np.ndarray:
        """
        Predice etiquetas para una nube de puntos normalizada.

        Args:
            cloud: (N, 3), (N, 6) o (N, 9) segun features activas

        Returns:
            labels: (N,) int32   0=madera  1=corteza
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
        Pipeline completo: .ply -> segmentacion -> area de corteza.

        Args:
            ply_path:   ruta al .ply (con o sin labels)
            save_ply:   guardar nube coloreada .ply en output_dir
            output_dir: carpeta de resultados
            visualize:  abrir ventana 3D Open3D

        Returns:
            dict con bark_fraction, n_bark_points, n_wood_points, labels, pts
        """
        ply_path   = Path(ply_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSegmentando: {ply_path.name}")

        # Cargar con las mismas features que el modelo espera
        cloud, _, meta = load_ply_labeled(
            ply_path,
            compute_normals=self.use_normals,
            use_rgb=self.use_rgb,
        )

        if meta.get("has_rgb") is False and self.use_rgb:
            print(f"  [AVISO] El modelo fue entrenado con RGB pero este PLY "
                  f"no tiene campos RGB. La inferencia puede ser menos precisa.")

        cloud_norm = normalize_pointcloud(cloud)
        labels     = self.predict_cloud(cloud_norm)
        results    = compute_bark_area(cloud[:, :3], labels)
        results.update({
            "labels":   labels,
            "pts":      cloud[:, :3],
            "ply_path": str(ply_path),
        })

        print(f"  Corteza: {results['n_bark_points']:,} pts "
              f"({results['bark_fraction']*100:.1f}%)")
        print(f"  Madera:  {results['n_wood_points']:,} pts")

        if save_ply:
            out = output_dir / (ply_path.stem + "_segmented.ply")
            save_colored_cloud(cloud[:, :3], labels, out)

        if visualize:
            visualize_segmentation(cloud[:, :3], labels, title=ply_path.stem)

        return results