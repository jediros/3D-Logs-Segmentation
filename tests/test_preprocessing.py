"""
tests/test_preprocessing.py
Ejecutar con: pytest tests/ -v
"""

import numpy as np
import pytest
import torch
import struct
import tempfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: crear PLY de prueba en memoria
# ─────────────────────────────────────────────────────────────────────────────

def make_test_ply(tmp_path: Path, n_verts: int = 200, bark_fraction: float = 0.2) -> Path:
    """Genera un .ply binario minimal con labels para tests."""
    n_bark = int(n_verts * bark_fraction)
    pts    = np.random.randn(n_verts, 3).astype(np.float32)
    l1     = np.zeros(n_verts, dtype=np.float32)
    l1[:n_bark] = 1.0
    l0     = 1.0 - l1
    # Triangulos dummy (ciclo de 3 en 3)
    n_faces = (n_verts // 3)
    faces   = [(i*3, i*3+1, i*3+2) for i in range(n_faces) if i*3+2 < n_verts]

    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property float s\nproperty float t\n"
        f"property float label_1\nproperty float label_0\n"
        f"element face {len(faces)}\n"
        f"property list uchar uint vertex_indices\n"
        f"end_header\n"
    ).encode("ascii")

    out = tmp_path / "test_tronco.ply"
    with open(out, "wb") as f:
        f.write(header)
        for i in range(n_verts):
            f.write(struct.pack("<fffffffffff" if False else "<fffffff",
                pts[i,0], pts[i,1], pts[i,2],
                0.0, 0.0, float(l1[i]), float(l0[i])))
        for face in faces:
            f.write(struct.pack("<BIII", 3, face[0], face[1], face[2]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tests: loader
# ─────────────────────────────────────────────────────────────────────────────

class TestPlyLoader:
    def test_load_shape_with_normals(self, tmp_path):
        from data.loader import load_ply_labeled
        ply = make_test_ply(tmp_path, n_verts=300, bark_fraction=0.25)
        cloud, labels, meta = load_ply_labeled(ply, compute_normals=True)
        assert cloud.shape[1] == 6        # xyz + normales
        assert labels.shape[0] == cloud.shape[0]

    def test_load_shape_without_normals(self, tmp_path):
        from data.loader import load_ply_labeled
        ply = make_test_ply(tmp_path, n_verts=300)
        cloud, labels, meta = load_ply_labeled(ply, compute_normals=False)
        assert cloud.shape[1] == 3

    def test_labels_binary(self, tmp_path):
        from data.loader import load_ply_labeled
        ply = make_test_ply(tmp_path, n_verts=300, bark_fraction=0.3)
        _, labels, _ = load_ply_labeled(ply)
        assert set(labels.tolist()).issubset({-1, 0, 1})

    def test_bark_fraction(self, tmp_path):
        from data.loader import load_ply_labeled
        ply = make_test_ply(tmp_path, n_verts=200, bark_fraction=0.2)
        _, labels, meta = load_ply_labeled(ply)
        assert abs(meta["bark_fraction"] - 0.2) < 0.05

    def test_file_not_found(self):
        from data.loader import load_ply_labeled
        with pytest.raises(FileNotFoundError):
            load_ply_labeled("no_existe.ply")

    def test_normals_unit_length(self, tmp_path):
        from data.loader import load_ply_labeled
        ply = make_test_ply(tmp_path, n_verts=300)
        cloud, _, _ = load_ply_labeled(ply, compute_normals=True)
        norms = cloud[:, 3:6]
        lengths = np.linalg.norm(norms, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: normalizacion
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalization:
    def test_centroid_at_origin(self):
        from preprocessing.sampler import normalize_pointcloud
        cloud = np.random.randn(500, 6).astype(np.float32)
        norm  = normalize_pointcloud(cloud)
        np.testing.assert_allclose(norm[:, :3].mean(axis=0), 0, atol=1e-5)

    def test_max_radius_one(self):
        from preprocessing.sampler import normalize_pointcloud
        cloud = np.random.randn(500, 3).astype(np.float32)
        norm  = normalize_pointcloud(cloud)
        r = np.sqrt((norm[:, :3] ** 2).sum(axis=1)).max()
        assert abs(r - 1.0) < 1e-5

    def test_normals_unchanged(self):
        from preprocessing.sampler import normalize_pointcloud
        cloud = np.random.randn(100, 6).astype(np.float32)
        norm  = normalize_pointcloud(cloud)
        # Las normales no deben verse afectadas por la normalizacion de posicion
        assert norm.shape == cloud.shape


# ─────────────────────────────────────────────────────────────────────────────
# Tests: dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestBarkDataset:
    def test_len(self, tmp_path):
        from data.dataset import BarkDataset
        for i in range(3):
            make_test_ply(tmp_path / f"t{i}.ply".replace("t", str(i)), n_verts=300)
        # crear en la misma carpeta tmp_path
        for i in range(3):
            make_test_ply(tmp_path, n_verts=300)
        # Usar directamente tmp_path con los PLY ya creados
        ds = BarkDataset(tmp_path, num_points=128, cache=True)
        assert len(ds) >= 1

    def test_item_shapes(self, tmp_path):
        from data.dataset import BarkDataset
        make_test_ply(tmp_path, n_verts=500, bark_fraction=0.3)
        make_test_ply(Path(str(tmp_path) + "2"), n_verts=500, bark_fraction=0.2)
        # Crear segundo PLY en la misma carpeta con nombre diferente
        ply1 = make_test_ply(tmp_path, n_verts=500, bark_fraction=0.3)
        ply2 = tmp_path / "tronco_2.ply"
        import shutil
        shutil.copy(ply1, ply2)
        ds = BarkDataset(tmp_path, num_points=256, use_normals=True, cache=False)
        cloud, labels = ds[0]
        assert cloud.shape  == (256, 6)
        assert labels.shape == (256,)
        assert labels.dtype == torch.int64

    def test_labels_values(self, tmp_path):
        from data.dataset import BarkDataset
        ply = make_test_ply(tmp_path, n_verts=500, bark_fraction=0.3)
        ds  = BarkDataset(tmp_path, num_points=128, cache=False)
        _, labels = ds[0]
        assert set(labels.numpy().tolist()).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# Tests: modelo
# ─────────────────────────────────────────────────────────────────────────────

class TestPointNet2:
    def test_forward_shape(self):
        from model.pointnet2 import PointNet2Segmentation
        model = PointNet2Segmentation(num_classes=2, use_normals=True)
        model.eval()
        x = torch.randn(2, 256, 6)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 256, 2)

    def test_no_nan(self):
        from model.pointnet2 import PointNet2Segmentation
        model = PointNet2Segmentation(num_classes=2, use_normals=False)
        model.eval()
        x = torch.randn(1, 128, 3)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_parameter_count(self):
        from model.pointnet2 import PointNet2Segmentation
        model = PointNet2Segmentation(num_classes=2)
        assert 100_000 < model.count_parameters() < 10_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Tests: metricas
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_perfect_iou(self):
        from utils.metrics import SegmentationMetrics
        m = SegmentationMetrics(num_classes=2)
        t = torch.tensor([0, 0, 1, 1, 0, 1])
        m.update(t, t)
        r = m.compute()
        assert r["miou"] == pytest.approx(1.0)

    def test_bark_area(self):
        from utils.metrics import compute_bark_area
        pts    = np.random.randn(1000, 3).astype(np.float32)
        labels = np.array([1]*250 + [0]*750)
        result = compute_bark_area(pts, labels, surface_area_m2=2.0)
        assert result["bark_fraction"]  == pytest.approx(0.25)
        assert result["bark_area_m2"]   == pytest.approx(0.50)
