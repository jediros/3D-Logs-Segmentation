"""
tests/test_preprocess_cache.py
------------------------------
Test the TASK-20 offline preprocessing pipeline and cache loading.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from config.config_loader import load_config
from preprocessing.sampler import preprocess_dataset, normalize_pointcloud
from data.dataset import BarkDataset


class TestPreprocessDataset:
    """Test offline preprocessing function"""
    
    def test_preprocess_creates_cache(self, tmp_path):
        """Verify preprocess_dataset creates .npy, _labels.npy, and _meta.json files"""
        
        # Setup: create minimal test PLY
        from tests.test_preprocessing import make_test_ply
        
        ply_dir = tmp_path / "raw"
        ply_dir.mkdir()
        cache_dir = tmp_path / "processed"
        cache_dir.mkdir()
        
        # Create one test PLY
        ply_path = make_test_ply(ply_dir, n_verts=300, bark_fraction=0.33, filename="test_log.ply")
        
        # Load config and override paths
        cfg = load_config()
        cfg.paths.raw_data = str(ply_dir)
        cfg.paths.processed_data = str(cache_dir)
        cfg.preprocessing.voxel_size = None  # No downsampling for test
        
        # Run preprocessing
        preprocess_dataset(cfg, overwrite=False)
        
        # Verify outputs
        npy_file = cache_dir / (ply_path.stem + ".npy")
        labels_file = cache_dir / (ply_path.stem + "_labels.npy")
        meta_file = cache_dir / (ply_path.stem + "_meta.json")
        
        assert npy_file.exists(), f"Cache file not created: {npy_file}"
        assert labels_file.exists(), f"Labels file not created: {labels_file}"
        assert meta_file.exists(), f"Meta file not created: {meta_file}"
        
        # Verify file contents
        cloud = np.load(npy_file)
        labels = np.load(labels_file)
        with open(meta_file) as f:
            meta = json.load(f)
        
        assert len(cloud) > 0, "Cloud is empty"
        assert len(labels) == len(cloud), "Labels size mismatch"
        assert meta["n_bark"] > 0, "No bark points"
        assert meta["n_wood"] > 0, "No wood points"
        print(f"✓ Preprocessing created cache: {len(cloud)} points")
    
    def test_preprocess_with_downsampling(self, tmp_path):
        """Verify voxel downsampling works during preprocessing"""
        
        from tests.test_preprocessing import make_test_ply
        
        ply_dir = tmp_path / "raw"
        ply_dir.mkdir()
        cache_dir = tmp_path / "processed"
        cache_dir.mkdir()
        
        ply_path = make_test_ply(ply_dir, n_verts=500, bark_fraction=0.4, filename="test_downsample.ply")
        
        cfg = load_config()
        cfg.paths.raw_data = str(ply_dir)
        cfg.paths.processed_data = str(cache_dir)
        cfg.preprocessing.voxel_size = 0.1  # Enable downsampling
        
        preprocess_dataset(cfg, overwrite=False)
        
        meta_file = cache_dir / (ply_path.stem + "_meta.json")
        with open(meta_file) as f:
            meta = json.load(f)
        
        # Should have downsampling metadata
        assert "n_points_original" in meta, "Downsampling metadata missing"
        assert meta["n_points_downsampled"] < meta["n_points_original"], \
            "Downsampling should reduce points"
        print(f"✓ Downsampling: {meta['n_points_original']} → {meta['n_points_downsampled']}")
    
    def test_cache_skip_without_overwrite(self, tmp_path):
        """Verify preprocessing skips existing cache without overwrite flag"""
        
        from tests.test_preprocessing import make_test_ply
        
        ply_dir = tmp_path / "raw"
        ply_dir.mkdir()
        cache_dir = tmp_path / "processed"
        cache_dir.mkdir()
        
        ply_path = make_test_ply(ply_dir, n_verts=200, bark_fraction=0.25, filename="test_skip.ply")
        
        cfg = load_config()
        cfg.paths.raw_data = str(ply_dir)
        cfg.paths.processed_data = str(cache_dir)
        cfg.preprocessing.voxel_size = None
        
        # First run
        preprocess_dataset(cfg, overwrite=False)
        npy_file = cache_dir / (ply_path.stem + ".npy")
        mtime1 = npy_file.stat().st_mtime
        
        # Second run (should skip)
        import time
        time.sleep(0.1)  # Small delay
        preprocess_dataset(cfg, overwrite=False)
        mtime2 = npy_file.stat().st_mtime
        
        assert mtime1 == mtime2, "File was modified despite skip"
        print("✓ Cache correctly skipped without overwrite")


class TestCacheLoading:
    """Test BarkDataset loading from cache"""
    
    def test_dataset_loads_from_cache(self, tmp_path):
        """Verify BarkDataset can load from cached .npy files"""
        
        from tests.test_preprocessing import make_test_ply
        
        ply_dir = tmp_path / "raw"
        ply_dir.mkdir()
        cache_dir = tmp_path / "processed"
        cache_dir.mkdir()
        
        ply_path = make_test_ply(ply_dir, n_verts=300, bark_fraction=0.33, filename="test_load.ply")
        
        cfg = load_config()
        cfg.paths.raw_data = str(ply_dir)
        cfg.paths.processed_data = str(cache_dir)
        cfg.preprocessing.voxel_size = None
        
        # Preprocess
        preprocess_dataset(cfg, overwrite=False)
        
        # Load from cache
        ds = BarkDataset(
            ply_dir=str(ply_dir),
            num_points=4096,
            use_cache_dir=True,
            cache_dir=str(cache_dir),
            cache=False,
        )
        
        assert len(ds) == 1, "Dataset should have 1 file"
        
        # Get item
        pts, labels = ds[0]
        assert pts.shape[0] == 4096, "Should sample to num_points"
        assert labels.shape[0] == 4096, "Labels should match points"
        assert pts.dtype == torch.float32, "Points should be float32"
        assert labels.dtype == torch.int64, "Labels should be int64"
        
        print(f"✓ Loaded from cache: {pts.shape}, {labels.shape}")
    
    def test_dataset_raises_on_missing_cache(self, tmp_path):
        """Verify BarkDataset raises error if cache files are missing"""
        
        from tests.test_preprocessing import make_test_ply
        
        ply_dir = tmp_path / "raw"
        ply_dir.mkdir()
        cache_dir = tmp_path / "processed"
        cache_dir.mkdir()
        
        ply_path = make_test_ply(ply_dir, n_verts=300, bark_fraction=0.33, filename="test_missing.ply")
        
        # Try to load from cache without preprocessing
        ds = BarkDataset(
            ply_dir=str(ply_dir),
            num_points=4096,
            use_cache_dir=True,
            cache_dir=str(cache_dir),
            cache=False,
        )
        
        with pytest.raises(FileNotFoundError):
            _ = ds[0]
        
        print("✓ Correctly raised error for missing cache")


class TestTrainingWithCache:
    """Integration test: training with cached data"""
    
    def test_train_config_with_cache_flag(self):
        """Verify config exposes the use_cache flag"""
        cfg = load_config()
        # Just verify the key exists and is a bool (value can be True or False)
        use_cache = getattr(cfg.preprocessing, "use_cache", None)
        assert use_cache is not None, "Config missing use_cache key"
        assert isinstance(use_cache, bool), "use_cache should be a bool"
        print(f"✓ Config has use_cache flag (currently: {use_cache})")
    
    def test_dataset_instantiation_with_cache_disabled(self, tmp_path):
        """Verify normal training works (cache disabled)"""
        
        from tests.test_preprocessing import make_test_ply
        
        ply_dir = tmp_path / "raw"
        ply_dir.mkdir()
        make_test_ply(ply_dir, n_verts=200, bark_fraction=0.25, filename="test1.ply")
        
        # Default: cache disabled, load from PLY
        ds = BarkDataset(
            ply_dir=str(ply_dir),
            num_points=4096,
            use_cache_dir=False,
            cache=False,
        )
        
        assert len(ds) == 1
        pts, labels = ds[0]
        assert pts.shape == (4096, 6), "Should be xyz + normals"
        print("✓ Normal training (no cache) works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
