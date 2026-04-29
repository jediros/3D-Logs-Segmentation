import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import open3d as o3d

# Importamos tus definiciones oficiales
from data.loader import load_ply_labeled
from utils.visualizer import COLOR_WOOD, COLOR_BARK

def get_plotly_color(color_array):
    # Convierte [0.76, 0.60, 0.42] a 'rgb(193, 153, 107)'
    return f'rgb({int(color_array[0]*255)}, {int(color_array[1]*255)}, {int(color_array[2]*255)})'

def visualize_comparison(filename):
    raw_path = Path("data/raw") / filename
    segmented_filename = filename.replace(".ply", "_segmented.ply")
    pred_path = Path("outputs") / segmented_filename

    if not pred_path.exists():
        print("❌ Archivo segmentado no encontrado.")
        return

    print("Cargando y comparando etiquetas reales...")

    # 1. Cargar ORIGINAL (Ground Truth)
    _, labels_gt, _ = load_ply_labeled(raw_path, compute_normals=False)
    
    # 2. Cargar PREDICCIÓN (Leemos los colores que guardó la IA)
    pcd_pred = o3d.io.read_point_cloud(str(pred_path))
    pts = np.asarray(pcd_pred.points)
    colors_ia = np.asarray(pcd_pred.colors)

    # Lógica de colores correcta basada en tu visualizer.py
    # Si el color tiene poco rojo (< 0.5), es CORTEZA (porque Bark=0.30 y Wood=0.76)
    labels_pred = (colors_ia[:, 0] < 0.5).astype(int)

    # 3. Crear Visualización
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Original (Blender)", "Predicción IA (Corregida)")
    )

    # Colores para Plotly
    c_wood = get_plotly_color(COLOR_WOOD)
    c_bark = 'rgb(255, 0, 0)' # Usamos Rojo brillante para que resalte en la comparativa

    def map_colors(lbls):
        return [c_bark if l == 1 else 'lightgray' for l in lbls]

    # Panel Izquierdo
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers', marker=dict(size=2, color=map_colors(labels_gt)),
        name="Manual"
    ), row=1, col=1)

    # Panel Derecho
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers', marker=dict(size=2, color=map_colors(labels_pred)),
        name="IA"
    ), row=1, col=2)

    fig.update_layout(template="plotly_dark", title=f"Validación de Etiquetas: {filename}")
    
    output_html = f"outputs/CORRECTED_{Path(filename).stem}.html"
    fig.write_html(output_html)
    print(f"✅ Comparación corregida en: {output_html}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_comparison(sys.argv[1])