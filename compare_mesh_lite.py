import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import open3d as o3d

# Importamos tu cargador oficial
from data.loader import load_ply_labeled

def generate_fast_mesh(pts, labels):
    """Genera una malla rápida solo para los puntos de corteza."""
    bark_pts = pts[labels == 1]
    if len(bark_pts) < 30:
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bark_pts)
    pcd.estimate_normals()
    
    # Alpha shape: 10.0 es un buen equilibrio. 
    # Si la malla se ve muy "flaca", sube a 15. Si se ve muy "boluda", baja a 5.
    alpha = 10.0 
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh

def visualize_comparison_mesh_lite(filename, max_points=6000):
    raw_path = Path("data/raw") / filename
    segmented_filename = Path(filename).stem + "_segmented.ply"
    pred_path = Path("outputs") / segmented_filename

    if not pred_path.exists():
        print(f"❌ No se encuentra: {pred_path}")
        return

    print(f"Procesando malla ligera para {filename}...")

    # 1. CARGAR DATOS
    _, labels_gt, _ = load_ply_labeled(raw_path, compute_normals=False)
    pcd_pred = o3d.io.read_point_cloud(str(pred_path))
    pts = np.asarray(pcd_pred.points)
    colors_ia = np.asarray(pcd_pred.colors)
    labels_pred = (colors_ia[:, 0] < 0.5).astype(int)

    # 2. SUBSAMPLING (Para que el navegador no se trabe)
    n_total = len(pts)
    indices = np.random.choice(n_total, min(n_total, max_points), replace=False)
    
    pts_v = pts[indices]
    l_gt_v = labels_gt[indices]
    l_pred_v = labels_pred[indices]

    # 3. GENERAR MALLAS LIGERAS (Basadas solo en los puntos reducidos)
    mesh_gt = generate_fast_mesh(pts_v, l_gt_v)
    mesh_pred = generate_fast_mesh(pts_v, l_pred_v)

    # 4. CREAR FIGURA
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("ANTES: Original (Con Malla)", "DESPUÉS: Predicción IA (Con Malla)")
    )

    # --- PANEL 1: ORIGINAL ---
    # Puntos madera
    fig.add_trace(go.Scatter3d(
        x=pts_v[l_gt_v==0,0], y=pts_v[l_gt_v==0,1], z=pts_v[l_gt_v==0,2],
        mode='markers', marker=dict(size=1.5, color='lightgray', opacity=0.5), name="Madera"
    ), row=1, col=1)
    # Malla corteza
    if mesh_gt:
        fig.add_trace(go.Mesh3d(
            x=np.asarray(mesh_gt.vertices)[:,0], y=np.asarray(mesh_gt.vertices)[:,1], z=np.asarray(mesh_gt.vertices)[:,2],
            i=np.asarray(mesh_gt.triangles)[:,0], j=np.asarray(mesh_gt.triangles)[:,1], k=np.asarray(mesh_gt.triangles)[:,2],
            color='red', opacity=0.8, name="Superficie Corteza"
        ), row=1, col=1)

    # --- PANEL 2: PREDICCIÓN ---
    # Puntos madera
    fig.add_trace(go.Scatter3d(
        x=pts_v[l_pred_v==0,0], y=pts_v[l_pred_v==0,1], z=pts_v[l_pred_v==0,2],
        mode='markers', marker=dict(size=1.5, color='lightgray', opacity=0.5), name="Madera"
    ), row=1, col=2)
    # Malla corteza
    if mesh_pred:
        fig.add_trace(go.Mesh3d(
            x=np.asarray(mesh_pred.vertices)[:,0], y=np.asarray(mesh_pred.vertices)[:,1], z=np.asarray(mesh_pred.vertices)[:,2],
            i=np.asarray(mesh_pred.triangles)[:,0], j=np.asarray(mesh_pred.triangles)[:,1], k=np.asarray(mesh_pred.triangles)[:,2],
            color='red', opacity=0.8, name="Superficie IA"
        ), row=1, col=2)

    # Estética
    fig.update_layout(
        template="plotly_dark", title=f"Visualización de Áreas: {filename}",
        margin=dict(l=0, r=0, b=0, t=50), showlegend=False
    )

    output_html = f"outputs/MESH_LITE_{Path(filename).stem}.html"
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"✅ Archivo creado: {output_html}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_comparison_mesh_lite(sys.argv[1])