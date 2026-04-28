"""
data/dataset.py
---------------
PyTorch Dataset que carga directamente los .ply etiquetados de Blender.

Soporta features:
    - XYZ siempre
    - Normales (use_normals=True)
    - RGB del scanner (use_rgb=True)

Shape del tensor de salida:
    use_normals=False, use_rgb=False  ->  (num_points, 3)
    use_normals=True,  use_rgb=False  ->  (num_points, 6)   <- anterior
    use_normals=True,  use_rgb=True   ->  (num_points, 9)   <- nuevo con RGB
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from data.loader import load_ply_labeled, scan_ply_folder


class BarkDataset(Dataset):
    """
    Dataset de segmentacion corteza/madera desde .ply de Blender.

    Args:
        ply_dir:          carpeta con archivos .ply etiquetados
        num_points:       puntos exactos por muestra
        augment:          augmentacion geometrica (solo en train)
        use_normals:      incluir normales como features
        use_rgb:          incluir RGB del scanner como features
        ignore_boundary:  descartar puntos de frontera entre materiales
        cache:            precargar todos los PLY en RAM

    Dimension de features:
        use_normals=True,  use_rgb=False  ->  6  (xyz + normales)
        use_normals=True,  use_rgb=True   ->  9  (xyz + normales + rgb)
        use_normals=False, use_rgb=True   ->  6  (xyz + rgb)
        use_normals=False, use_rgb=False  ->  3  (solo xyz)
    """

    def __init__(
        self,
        ply_dir,
        num_points:      int  = 4096,
        augment:         bool = False,
        use_normals:     bool = True,
        use_rgb:         bool = False,
        ignore_boundary: bool = True,
        cache:           bool = True,
    ):
        self.ply_dir         = Path(ply_dir)
        self.num_points      = num_points
        self.augment         = augment
        self.use_normals     = use_normals
        self.use_rgb         = use_rgb
        self.ignore_boundary = ignore_boundary

        self.ply_files = scan_ply_folder(self.ply_dir)
        if not self.ply_files:
            raise RuntimeError(
                f"No se encontraron archivos .ply en: {self.ply_dir}"
            )

        self.ply_files = self._filter_labeled(self.ply_files)
        if not self.ply_files:
            raise RuntimeError(
                f"Ningun .ply tiene labels en: {self.ply_dir}\n"
                f"Asegurate de exportar con materiales asignados en Blender."
            )

        print(f"Dataset: {len(self.ply_files)} troncos en {self.ply_dir}")

        self._cache = {}
        if cache:
            self._load_cache()

    def _n_features(self) -> int:
        """Calcula el numero de features segun la configuracion."""
        n = 3                              # XYZ siempre
        if self.use_normals: n += 3        # normales
        if self.use_rgb:     n += 3        # RGB
        return n

    def _filter_labeled(self, files: list) -> list:
        valid = []
        for f in files:
            try:
                _, labels, meta = load_ply_labeled(
                    f,
                    compute_normals=False,
                    use_rgb=False,
                )
                if meta["n_bark"] > 0 or meta["n_wood"] > 0:
                    valid.append(f)
                else:
                    print(f"  [skip] {f.name} — sin labels")
            except Exception as e:
                print(f"  [skip] {f.name} — error: {e}")
        return valid

    def _load_cache(self):
        print("  Cargando PLY en memoria...")
        for ply_path in self.ply_files:
            cloud, labels, meta = load_ply_labeled(
                ply_path,
                ignore_boundary=self.ignore_boundary,
                compute_normals=self.use_normals,
                use_rgb=self.use_rgb,
            )
            self._cache[str(ply_path)] = (cloud, labels, meta)
            pct = meta["bark_fraction"] * 100
            rgb_info = " +RGB" if meta.get("has_rgb") else ""
            print(f"    {ply_path.name}: {meta['n_vertices']:,} pts  "
                  f"corteza={meta['n_bark']:,} ({pct:.1f}%)  "
                  f"features={meta['n_features']}{rgb_info}")
        print(f"  Cache listo: {len(self._cache)} troncos")

    def __len__(self) -> int:
        return len(self.ply_files)

    def __getitem__(self, idx: int):
        """
        Returns:
            points: FloatTensor (num_points, C)
                    C=6 con normales, C=9 con normales+RGB
            labels: LongTensor  (num_points,)  valores 0 o 1
        """
        ply_path = self.ply_files[idx]
        key      = str(ply_path)

        if key in self._cache:
            cloud, labels, _ = self._cache[key]
        else:
            cloud, labels, _ = load_ply_labeled(
                ply_path,
                ignore_boundary=self.ignore_boundary,
                compute_normals=self.use_normals,
                use_rgb=self.use_rgb,
            )

        # Excluir puntos ignorados (-1)
        valid_mask   = labels != -1
        pts_valid    = cloud[valid_mask]
        labels_valid = labels[valid_mask]

        # Sampleo a num_points exacto
        pts_out, labels_out = self._sample(pts_valid, labels_valid)

        # Normalizar XYZ (columnas 0-2 siempre)
        pts_out = self._normalize(pts_out)

        # Augmentacion geometrica (solo en train)
        if self.augment:
            pts_out = self._augment(pts_out)

        return (
            torch.from_numpy(pts_out.astype(np.float32)),
            torch.from_numpy(labels_out.astype(np.int64)),
        )

    def _sample(self, pts: np.ndarray, labels: np.ndarray):
        n = len(pts)
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.concatenate([
                np.arange(n),
                np.random.choice(n, self.num_points - n, replace=True),
            ])
        return pts[idx], labels[idx]

    def _normalize(self, pts: np.ndarray) -> np.ndarray:
        """
        Normaliza SOLO las coordenadas XYZ (columnas 0-2).
        Normales y RGB no se modifican.
        """
        pts = pts.copy()
        centroid = pts[:, :3].mean(axis=0)
        pts[:, :3] -= centroid
        max_r = np.sqrt((pts[:, :3] ** 2).sum(axis=1)).max()
        if max_r > 0:
            pts[:, :3] /= max_r
        return pts

    def _augment(self, pts: np.ndarray) -> np.ndarray:
        """
        Augmentacion geometrica:
        - Rotacion eje Y (longitudinal del tronco): 360 grados
        - Inclinacion leve X y Z: +/- 8 grados
        - Jitter gaussiano en XYZ: sigma=0.002
        - Escala aleatoria: +/- 5%

        RGB no se modifica — el color no cambia al rotar el tronco.
        Las normales se rotan de forma consistente con la geometria.
        """
        pts = pts.copy()
        xyz = pts[:, :3]

        # Rotacion eje Y
        angle = np.random.uniform(0, 2 * np.pi)
        c, s  = np.cos(angle), np.sin(angle)
        rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        xyz   = xyz @ rot_y.T

        # Inclinacion leve
        for axis in [0, 2]:
            a = np.random.uniform(-0.14, 0.14)
            c2, s2 = np.cos(a), np.sin(a)
            if axis == 0:
                rot = np.array([[1,0,0],[0,c2,-s2],[0,s2,c2]], dtype=np.float32)
            else:
                rot = np.array([[c2,-s2,0],[s2,c2,0],[0,0,1]], dtype=np.float32)
            xyz = xyz @ rot.T

        # Jitter en XYZ
        xyz += np.random.normal(0, 0.002, xyz.shape).astype(np.float32)

        # Escala
        xyz *= np.random.uniform(0.95, 1.05)
        pts[:, :3] = xyz

        # Rotar normales si las hay (columnas 3-5)
        if self.use_normals and pts.shape[1] >= 6:
            pts[:, 3:6] = pts[:, 3:6] @ rot_y.T

        # RGB (columnas 6-8 si use_rgb=True) — NO se modifican

        return pts

    def get_class_weights(self) -> torch.Tensor:
        """
        Calcula pesos de clase para compensar desbalance corteza/madera.

        Returns:
            tensor([peso_madera, peso_corteza])
        """
        counts = np.zeros(2, dtype=np.int64)
        for ply_path in self.ply_files:
            key = str(ply_path)
            if key in self._cache:
                _, labels, _ = self._cache[key]
            else:
                _, labels, _ = load_ply_labeled(
                    ply_path,
                    compute_normals=False,
                    use_rgb=False,
                )
            counts[0] += int((labels == 0).sum())
            counts[1] += int((labels == 1).sum())

        total   = counts.sum()
        weights = total / (2.0 * counts + 1e-6)
        print(f"  Pesos de clase: madera={weights[0]:.3f}  corteza={weights[1]:.3f}")
        return torch.tensor(weights, dtype=torch.float32)

    def summary(self):
        """Imprime resumen estadistico del dataset."""
        print(f"\n{'─'*65}")
        print(f"  {'Archivo':<30} {'Pts':>8} {'Corteza':>8} {'%':>6} {'Features':>9}")
        print(f"  {'─'*63}")
        total_pts  = 0
        total_bark = 0
        for ply_path in self.ply_files:
            key = str(ply_path)
            if key in self._cache:
                _, _, meta = self._cache[key]
            else:
                _, _, meta = load_ply_labeled(
                    ply_path,
                    compute_normals=False,
                    use_rgb=False,
                )
            pct = meta["bark_fraction"] * 100
            print(f"  {ply_path.name:<30} {meta['n_vertices']:>8,} "
                  f"{meta['n_bark']:>8,} {pct:>5.1f}%"
                  f"  {meta.get('n_features', '?'):>6}")
            total_pts  += meta["n_vertices"]
            total_bark += meta["n_bark"]
        pct_t = total_bark / max(total_pts, 1) * 100
        print(f"  {'─'*63}")
        print(f"  {'TOTAL':<30} {total_pts:>8,} {total_bark:>8,} {pct_t:>5.1f}%")
        print(f"{'─'*65}\n")