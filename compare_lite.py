import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import open3d as o3d

# Importamos tu cargador oficial
from data.loader import load_ply_labeled

def visualize_comparison_lite(filename, max_points=5000):
    """
    Crea una comparativa ligera Antes/Después.
    max_points: limita los puntos mostrados para que el HTML sea fluido.
    """
    raw_path = Path("data/raw") / filename
    segmented_filename = Path(filename).stem + "_segmented.ply"
    pred_path = Path("outputs") / segmented_filename

    if not pred_path.exists():
        print(f"❌ No se encuentra el archivo: {pred_path}")
        return

    print(f"Generando vista ligera para {filename}...")

    # 1. CARGAR DATOS
    # Original
    _, labels_gt, _ = load_ply_labeled(raw_path, compute_normals=False)
    # Predicción
    pcd_pred = o3d.io.read_point_cloud(str(pred_path))
    pts = np.asarray(pcd_pred.points)
    colors_ia = np.asarray(pcd_pred.colors)
    labels_pred = (colors_ia[:, 0] < 0.5).astype(int)

    # 2. SUBSAMPLING (El truco para la velocidad)
    # Seleccionamos índices aleatorios para no saturar el navegador
    n_total = len(pts)
    if n_total > max_points:
        indices = np.random.choice(n_total, max_points, replace=False)
        pts_viz = pts[indices]
        labels_gt_viz = labels_gt[indices]
        labels_pred_viz = labels_pred[indices]
    else:
        pts_viz = pts
        labels_gt_viz = labels_gt
        labels_pred_viz = labels_pred

    # 3. CONFIGURAR COLORES (Rojo para Corteza, Gris para Madera)
    def get_colors(lbls):
        return ['red' if l == 1 else 'lightgray' for l in lbls]

    # 4. CREAR LA FIGURA (2 PANELES)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("ANTES: Original (Blender)", "DESPUÉS: Predicción (IA)")
    )

    # Panel 1: Original
    fig.add_trace(go.Scatter3d(
        x=pts_viz[:, 0], y=pts_viz[:, 1], z=pts_viz[:, 2],
        mode='markers', 
        marker=dict(size=2, color=get_colors(labels_gt_viz), opacity=0.7),
        name="Manual"
    ), row=1, col=1)

    # Panel 2: Predicción IA
    fig.add_trace(go.Scatter3d(
        x=pts_viz[:, 0], y=pts_viz[:, 1], z=pts_viz[:, 2],
        mode='markers', 
        marker=dict(size=2, color=get_colors(labels_pred_viz), opacity=0.7),
        name="IA"
    ), row=1, col=2)

    # Ajustes estéticos para que sea rápido de mover
    fig.update_layout(
        template="plotly_dark",
        title=f"Comparativa Rápida: {filename} ({max_points} puntos)",
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False,
        # Desactivamos efectos pesados
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        scene2=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    )

    output_html = f"outputs/LITE_{Path(filename).stem}.html"
    # El flag include_plotlyjs='cdn' hace que el archivo sea mucho más pequeño
    fig.write_html(output_html, include_plotlyjs='cdn')
    
    print(f"✅ Archivo LITE creado: {output_html}")
    print(f"Puntos reducidos de {n_total:,} a {max_points:,} para fluidez.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_comparison_lite(sys.argv[1])
    else:
        print("Uso: python compare_lite.py nombre.ply")