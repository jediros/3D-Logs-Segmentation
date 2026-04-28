import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import open3d as o3d

# Importamos tu cargador oficial
from data.loader import load_ply_labeled

def generate_refined_mesh(pts, labels, target_label, alpha=12.0):
    """Genera una malla para una clase específica."""
    target_pts = pts[labels == target_label]
    if len(target_pts) < 50:
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_pts)
    pcd.estimate_normals()
    
    # Alpha shape para crear la superficie
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh

def visualize_solid_comparison(filename, max_points=8000):
    raw_path = Path("data/raw") / filename
    segmented_filename = Path(filename).stem + "_segmented.ply"
    pred_path = Path("outputs") / segmented_filename

    if not pred_path.exists():
        print(f"❌ No se encuentra: {pred_path}")
        return

    print(f"Transformando nubes de puntos en mallas sólidas para {filename}...")

    # 1. CARGAR DATOS
    _, labels_gt, _ = load_ply_labeled(raw_path, compute_normals=False)
    pcd_pred = o3d.io.read_point_cloud(str(pred_path))
    pts = np.asarray(pcd_pred.points)
    colors_ia = np.asarray(pcd_pred.colors)
    labels_pred = (colors_ia[:, 0] < 0.5).astype(int)

    # 2. REDUCCIÓN DE PUNTOS (Para velocidad del mesh)
    n_total = len(pts)
    indices = np.random.choice(n_total, min(n_total, max_points), replace=False)
    pts_v = pts[indices]
    l_gt_v = labels_gt[indices]
    l_pred_v = labels_pred[indices]

    # 3. CREAR FIGURA
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("ORIGINAL (Malla Sólida)", "PREDICCIÓN IA (Malla Sólida)")
    )

    def add_meshes_to_panel(panel_pts, panel_labels, column, suffix):
        # Malla de Madera (Gris) - Alpha más alto para que sea más cerrada
        mesh_wood = generate_refined_mesh(panel_pts, panel_labels, 0, alpha=15.0)
        # Malla de Corteza (Roja)
        mesh_bark = generate_refined_mesh(panel_pts, panel_labels, 1, alpha=10.0)

        if mesh_wood:
            fig.add_trace(go.Mesh3d(
                x=np.asarray(mesh_wood.vertices)[:,0], y=np.asarray(mesh_wood.vertices)[:,1], z=np.asarray(mesh_wood.vertices)[:,2],
                i=np.asarray(mesh_wood.triangles)[:,0], j=np.asarray(mesh_wood.triangles)[:,1], k=np.asarray(mesh_wood.triangles)[:,2],
                color='lightgray', opacity=0.3, name=f"Madera {suffix}", showscale=False
            ), row=1, col=column)

        if mesh_bark:
            fig.add_trace(go.Mesh3d(
                x=np.asarray(mesh_bark.vertices)[:,0], y=np.asarray(mesh_bark.vertices)[:,1], z=np.asarray(mesh_bark.vertices)[:,2],
                i=np.asarray(mesh_bark.triangles)[:,0], j=np.asarray(mesh_bark.triangles)[:,1], k=np.asarray(mesh_bark.triangles)[:,2],
                color='red', opacity=1.0, name=f"Corteza {suffix}", showscale=False
            ), row=1, col=column)

    # Procesar ambos paneles
    add_meshes_to_panel(pts_v, l_gt_v, 1, "GT")
    add_meshes_to_panel(pts_v, l_pred_v, 2, "IA")

    # Estética
    fig.update_layout(
        template="plotly_dark",
        title=f"Comparativa de Superficies Sólidas: {filename}",
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        scene2=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    )

    output_html = f"outputs/SOLID_{Path(filename).stem}.html"
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"✅ ¡Éxito! Visualización sólida creada: {output_html}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_solid_comparison(sys.argv[1])