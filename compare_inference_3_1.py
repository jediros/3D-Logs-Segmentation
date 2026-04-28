import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import open3d as o3d

# Importamos tus cargadores y definiciones
from data.loader import load_ply_labeled
from utils.visualizer import COLOR_WOOD, COLOR_BARK

def generate_mesh_from_points(points, labels):
    """Crea una malla 3D solo de los puntos que la IA marcó como corteza."""
    bark_pts = points[labels == 1]
    if len(bark_pts) < 20: # Necesitamos un mínimo de puntos para triangular
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bark_pts)
    pcd.estimate_normals()
    
    # Reconstrucción Alpha Shapes
    # Si la malla sale con muchos huecos, sube el 8.0 a 12.0
    alpha = 8.0 
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh

def visualize_comparison(filename):
    raw_path = Path("data/raw") / filename
    segmented_filename = filename.replace(".ply", "_segmented.ply")
    pred_path = Path("outputs") / segmented_filename

    if not pred_path.exists():
        print(f"❌ No se encuentra: {pred_path}. Haz la inferencia primero.")
        return

    print("Cargando datos y reconstruyendo malla sólida...")

    # 1. CARGAR ORIGINAL (Blender)
    _, labels_gt, _ = load_ply_labeled(raw_path, compute_normals=False)
    
    # 2. CARGAR PREDICCIÓN (IA)
    pcd_pred = o3d.io.read_point_cloud(str(pred_path))
    pts = np.asarray(pcd_pred.points)
    colors_ia = np.asarray(pcd_pred.colors)

    # CORRECCIÓN DE COLORES:
    # Según tu visualizer.py: Bark tiene R=0.30 y Wood tiene R=0.76.
    # Por lo tanto, si Rojo < 0.5, es Label 1 (Corteza).
    labels_pred = (colors_ia[:, 0] < 0.5).astype(int)

    # 3. GENERAR MALLA DE LA CORTEZA PREDICHA
    mesh = generate_mesh_from_points(pts, labels_pred)

    # 4. CONFIGURAR VISUALIZACIÓN (3 Paneles)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Original (Blender)", "Puntos IA (Corregidos)", "Malla Corteza IA")
    )

    # Función para colorear puntos en el gráfico
    def get_viz_colors(lbls):
        return ['red' if l == 1 else 'lightgray' for l in lbls]

    # Panel 1: Original
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers', marker=dict(size=2, color=get_viz_colors(labels_gt)),
        name="Manual"
    ), row=1, col=1)

    # Panel 2: Puntos IA
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers', marker=dict(size=2, color=get_viz_colors(labels_pred)),
        name="IA Puntos"
    ), row=1, col=2)

    # Panel 3: Malla IA
    if mesh and len(mesh.triangles) > 0:
        m_verts = np.asarray(mesh.vertices)
        m_faces = np.asarray(mesh.triangles)
        fig.add_trace(go.Mesh3d(
            x=m_verts[:, 0], y=m_verts[:, 1], z=m_verts[:, 2],
            i=m_faces[:, 0], j=m_faces[:, 1], k=m_faces[:, 2],
            color='brown', opacity=0.9, name="Volumen IA"
        ), row=1, col=3)
    else:
        print("⚠️ No se pudo generar la malla (pocos puntos detectados).")

    # Ajustes finales
    fig.update_layout(
        template="plotly_dark",
        title=f"Análisis Técnico de Segmentación: {filename}",
        showlegend=False
    )
    
    output_html = f"outputs/FINAL_CHECK_{Path(filename).stem}.html"
    fig.write_html(output_html)
    print(f"\n✅ ¡Todo listo! Revisa el archivo: {output_html}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_comparison(sys.argv[1])
    else:
        print("Uso: python compare_inference.py 24B_800N_decimated.ply") 