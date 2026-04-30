"""
data/dataset.py
---------------
PyTorch Dataset that loads labeled .ply files exported from Blender.

Supported features:
    - XYZ always
    - Normals (use_normals=True)
    - RGB from scanner (use_rgb=True)

Output tensor shape:
    use_normals=False, use_rgb=False  ->  (num_points, 3)
    use_normals=True,  use_rgb=False  ->  (num_points, 6)   <- previous
    use_normals=True,  use_rgb=True   ->  (num_points, 9)   <- new with RGB
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from data.loader import load_ply_labeled, scan_ply_folder


class BarkDataset(Dataset):
    """
    Bark/wood segmentation dataset from Blender-exported .ply files.

    Args:
        ply_dir:          folder containing labeled .ply files
        num_points:       exact number of points per sample
        augment:          geometric augmentation (training only)
        use_normals:      include normals as features
        use_rgb:          include scanner RGB as features
        ignore_boundary:  discard boundary points between materials
        cache:            preload all PLY files into RAM

    Feature dimensions:
        use_normals=True,  use_rgb=False  ->  6  (xyz + normals)
        use_normals=True,  use_rgb=True   ->  9  (xyz + normals + rgb)
        use_normals=False, use_rgb=True   ->  6  (xyz + rgb)
        use_normals=False, use_rgb=False  ->  3  (xyz only)
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
        use_cache_dir:   bool = False,
        cache_dir:       str  = None,
    ):
        self.ply_dir         = Path(ply_dir)
        self.num_points      = num_points
        self.augment         = augment
        self.use_normals     = use_normals
        self.use_rgb         = use_rgb
        self.ignore_boundary = ignore_boundary
        self.use_cache_dir   = use_cache_dir
        self.cache_dir       = Path(cache_dir) if cache_dir else None

        self.ply_files = scan_ply_folder(self.ply_dir)
        if not self.ply_files:
            raise RuntimeError(
                f"No .ply files found in: {self.ply_dir}"
            )

        self.ply_files = self._filter_labeled(self.ply_files)
        if not self.ply_files:
            raise RuntimeError(
                f"No .ply files have labels in: {self.ply_dir}\n"
                f"Make sure to export with materials assigned in Blender."
            )

        print(f"Dataset: {len(self.ply_files)} logs in {self.ply_dir}")
        if use_cache_dir and cache_dir:
            print(f"  Using cached .npy from {cache_dir}")

        self._cache = {}
        if cache and not use_cache_dir:
            self._load_cache()

    def _n_features(self) -> int:
        """Returns the number of features based on current configuration."""
        n = 3                              # XYZ always
        if self.use_normals: n += 3        # normals
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
                    print(f"  [skip] {f.name} — no labels")
            except Exception as e:
                print(f"  [skip] {f.name} — error: {e}")
        return valid

    def _load_cache(self):
        print("  Loading PLY files into memory...")
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
                  f"bark={meta['n_bark']:,} ({pct:.1f}%)  "
                  f"features={meta['n_features']}{rgb_info}")
        print(f"  Cache ready: {len(self._cache)} logs")

    def __len__(self) -> int:
        return len(self.ply_files)

    def __getitem__(self, idx: int):
        """
        Returns:
            points: FloatTensor (num_points, C)
                    C=6 with normals, C=9 with normals+RGB
            labels: LongTensor  (num_points,)  values 0 or 1
        """
        ply_path = self.ply_files[idx]
        key      = str(ply_path)

        # Load from cached .npy if enabled
        if self.use_cache_dir and self.cache_dir:
            import json
            npy_path = self.cache_dir / (ply_path.stem + ".npy")
            labels_path = self.cache_dir / (ply_path.stem + "_labels.npy")
            meta_path = self.cache_dir / (ply_path.stem + "_meta.json")
            
            if npy_path.exists() and labels_path.exists():
                cloud = np.load(npy_path)
                labels = np.load(labels_path)
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Cached files not found for {ply_path.name}. "
                    f"Run: python main.py preprocess"
                )
        elif key in self._cache:
            cloud, labels, _ = self._cache[key]
        else:
            cloud, labels, _ = load_ply_labeled(
                ply_path,
                ignore_boundary=self.ignore_boundary,
                compute_normals=self.use_normals,
                use_rgb=self.use_rgb,
            )

        # Exclude ignored points (-1)
        valid_mask   = labels != -1
        pts_valid    = cloud[valid_mask]
        labels_valid = labels[valid_mask]

        # Sample to exactly num_points
        pts_out, labels_out = self._sample(pts_valid, labels_valid)

        # Normalize XYZ (columns 0-2 always)
        pts_out = self._normalize(pts_out)

        # Geometric augmentation (training only)
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
        Normalizes ONLY XYZ coordinates (columns 0-2).
        Normals and RGB are not modified.
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
        Geometric augmentation:
        - Y-axis rotation (log longitudinal axis): 360 degrees
        - Slight X and Z tilt: +/- 8 degrees
        - Gaussian jitter on XYZ: sigma=0.002
        - Random scale: +/- 5%

        RGB is not modified — color does not change when rotating the log.
        Normals are rotated consistently with the geometry.
        """
        pts = pts.copy()
        xyz = pts[:, :3]

        # Y-axis rotation
        angle = np.random.uniform(0, 2 * np.pi)
        c, s  = np.cos(angle), np.sin(angle)
        rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        xyz   = xyz @ rot_y.T

        # Slight tilt
        for axis in [0, 2]:
            a = np.random.uniform(-0.14, 0.14)
            c2, s2 = np.cos(a), np.sin(a)
            if axis == 0:
                rot = np.array([[1,0,0],[0,c2,-s2],[0,s2,c2]], dtype=np.float32)
            else:
                rot = np.array([[c2,-s2,0],[s2,c2,0],[0,0,1]], dtype=np.float32)
            xyz = xyz @ rot.T

        # XYZ jitter
        xyz += np.random.normal(0, 0.002, xyz.shape).astype(np.float32)

        # Scale
        xyz *= np.random.uniform(0.95, 1.05)
        pts[:, :3] = xyz

        # Rotate normals if present (columns 3-5)
        if self.use_normals and pts.shape[1] >= 6:
            pts[:, 3:6] = pts[:, 3:6] @ rot_y.T

        # RGB (columns 6-8 if use_rgb=True) — NOT modified

        return pts

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes class weights to compensate for bark/wood imbalance.

        Returns:
            tensor([wood_weight, bark_weight])
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
        print(f"  Class weights: wood={weights[0]:.3f}  bark={weights[1]:.3f}")
        return torch.tensor(weights, dtype=torch.float32)

    def summary(self):
        """Prints a statistical summary of the dataset."""
        print(f"\n{'─'*65}")
        print(f"  {'File':<30} {'Pts':>8} {'Bark':>8} {'%':>6} {'Features':>9}")
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
                    ignore_boundary=self.ignore_boundary,
                    compute_normals=self.use_normals,
                    use_rgb=self.use_rgb,
                )
            pct = meta["bark_fraction"] * 100
            print(f"  {ply_path.name:<30} {meta['n_vertices']:>8,} "
                  f"{meta['n_bark']:>8,} {pct:>5.1f}%"
                  f"  {meta.get('n_features', '?'):>6}")
            total_pts  += meta["n_vertices"]
            total_bark += meta["n_bark"]
        pct_t = total_bark / max(total_pts, 1) * 100
        print(f"  {'─'*63}")
        print(f"  {'TOTAL':<30} {total_pts:>8,} {total_bark:>8,} {pct_t:>5.1f}%")  # TOTAL row stays in English
        print(f"{'─'*65}\n")