from pathlib import Path
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

COLOR_WOOD  = np.array([0.76, 0.60, 0.42])
COLOR_BARK  = np.array([0.30, 0.15, 0.05])
COLOR_UNLAB = np.array([0.75, 0.75, 0.75])

def _check():
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D no instalado.")

def _colors_from_labels(labels):
    colors = np.tile(COLOR_UNLAB, (len(labels), 1))
    colors[labels == 0] = COLOR_WOOD
    colors[labels == 1] = COLOR_BARK
    return colors

def visualize_cloud(pts, title="Nube de puntos", colors=None):
    _check()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(
        colors if colors is not None else np.tile([0.7,0.7,0.7], (len(pts),1)))
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1024, height=768)

def visualize_segmentation(pts, labels, title="Segmentacion corteza/madera"):
    _check()
    n_bark = (labels==1).sum()
    print(f"  Corteza: {n_bark} ({n_bark/len(labels)*100:.1f}%)  Madera: {(labels==0).sum()}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(_colors_from_labels(labels))
    o3d.visualization.draw_geometries([pcd], window_name=title, width=1024, height=768)

def save_colored_cloud(pts, labels, output_path):
    _check()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(_colors_from_labels(labels))
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"  Guardado: {output_path}")
