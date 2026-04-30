from pathlib import Path
import numpy as np
import json
from datetime import datetime

from data.loader import load_ply_labeled, scan_ply_folder


def normalize_pointcloud(cloud):
    cloud = cloud.copy()
    centroid = cloud[:, :3].mean(axis=0)
    cloud[:, :3] -= centroid
    max_r = np.sqrt((cloud[:, :3] ** 2).sum(axis=1)).max()
    if max_r > 0:
        cloud[:, :3] /= max_r
    return cloud


def voxel_downsample(cloud, labels, voxel_size):
    voxel_idx = np.floor(cloud[:, :3] / voxel_size).astype(np.int32)
    seen, keep = {}, []
    for i, vi in enumerate(map(tuple, voxel_idx)):
        if vi not in seen:
            seen[vi] = i
            keep.append(i)
    keep = np.array(keep)
    return cloud[keep], labels[keep]


def preprocess_dataset(cfg, overwrite=False):
    """
    Offline preprocessing: loads all PLY files, normalizes, optionally downsamples,
    and caches as .npy for faster training.
    
    Args:
        cfg: Configuration object with paths and preprocessing settings
        overwrite: If True, re-process all files even if cache exists
    """
    raw_dir = Path(cfg.paths.raw_data)
    cache_dir = Path(cfg.paths.processed_data)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    ply_files = scan_ply_folder(raw_dir)
    if not ply_files:
        print(f"[ERROR] No PLY files found in {raw_dir}")
        return
    
    stats = {"total": 0, "cached": 0, "skipped": 0, "errors": 0}
    
    print(f"\n[Preprocessing] Found {len(ply_files)} PLY files")
    print(f"  Output: {cache_dir}")
    if hasattr(cfg.preprocessing, 'voxel_size') and cfg.preprocessing.voxel_size:
        print(f"  Voxel downsampling: {cfg.preprocessing.voxel_size}")
    print()
    
    for ply_path in sorted(ply_files):
        npy_path = cache_dir / (ply_path.stem + ".npy")
        labels_path = cache_dir / (ply_path.stem + "_labels.npy")
        meta_path = cache_dir / (ply_path.stem + "_meta.json")
        
        # Skip if already cached and not overwrite
        if npy_path.exists() and labels_path.exists() and not overwrite:
            print(f"  ✓ {ply_path.name} (already cached)")
            stats["skipped"] += 1
            continue
        
        # Load PLY
        try:
            cloud, labels, meta = load_ply_labeled(
                ply_path,
                compute_normals=cfg.model.use_normals,
                use_rgb=cfg.model.use_rgb,
                ignore_boundary=cfg.preprocessing.ignore_boundary,
            )
        except Exception as e:
            print(f"  ✗ {ply_path.name} — Error: {e}")
            stats["errors"] += 1
            continue
        
        # Normalize point cloud
        cloud = normalize_pointcloud(cloud)
        
        # Optional downsampling
        if hasattr(cfg.preprocessing, 'voxel_size') and cfg.preprocessing.voxel_size:
            orig_n = len(cloud)
            cloud, labels = voxel_downsample(
                cloud, labels,
                voxel_size=cfg.preprocessing.voxel_size
            )
            meta["n_points_original"] = orig_n
            meta["n_points_downsampled"] = len(cloud)
            meta["voxel_size"] = cfg.preprocessing.voxel_size
            print(f"  ✓ {ply_path.name} — "
                  f"{orig_n:,} → {len(cloud):,} pts "
                  f"(corteza: {meta['n_bark']}, madera: {meta['n_wood']})")
        else:
            print(f"  ✓ {ply_path.name} — "
                  f"{len(cloud):,} pts "
                  f"(corteza: {meta['n_bark']}, madera: {meta['n_wood']})")
        
        # Cache point cloud
        np.save(npy_path, cloud)
        
        # Cache labels (separate file)
        np.save(labels_path, labels)
        
        # Cache metadata
        meta["cached_at"] = str(datetime.now())
        meta["preprocessed"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        
        stats["cached"] += 1
        stats["total"] += 1
    
    print(f"\n[Summary] Cached: {stats['cached']}, Skipped: {stats['skipped']}, "
          f"Errors: {stats['errors']}")
    if stats["cached"] > 0:
        print(f"[Success] Preprocessing complete. Ready to train with "
              f"`use_cache_dir=True`")
