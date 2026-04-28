import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import open3d as o3d  # Usaremos Open3D que es más flexible para leer PLYs

# Importamos tu cargador solo para el original (Ground Truth)
from data.loader import load_ply_labeled

def get_colors_from_labels(labels):
    # 0: Madera (Gris), 1: Corteza (Rojo)
    return ['rgb(200, 200, 200)' if l == 0 else 'rgb(255, 0, 0)' for l in labels]

def visualize_comparison(filename):
    raw_path = Path("data/raw") / filename
    segmented_filename = filename.replace(".ply", "_segmented.ply")
    pred_path = Path("outputs") / segmented_filename

    if not raw_path.exists() or not pred_path.exists():
        print(f"❌ No se encuentran los archivos. Asegúrate de que existan:\n- {raw_path}\n- {pred_path}")
        return

    print(f"Cargando archivos...")

    # 1. Cargar el ORIGINAL usando tu loader (que tiene los labels de Blender)
    try:
        cloud_gt, labels_gt, meta_gt = load_ply_labeled(raw_path, compute_normals=False)
        pts_gt = cloud_gt[:, :3]
        colors_gt = get_colors_from_labels(labels_gt)
    except Exception as e:
        print(f"❌ Error cargando original: {e}")
        return

    # 2. Cargar la PREDICCIÓN usando Open3D (evita el error de buffer)
    try:
        pcd = o3d.io.read_point_cloud(str(pred_path))
        pts_pred = np.asarray(pcd.points)
        # Extraer colores: si la IA lo pintó, Open3D los lee como float 0-1
        colors_pred = np.asarray(pcd.colors)
        
        if len(colors_pred) == 0:
            # Si no tiene colores, todo gris
            colors_pred = ['rgb(200, 200, 200)'] * len(pts_pred)
        else:
            # Convertir de float (0.0-1.0) a formato rgb string para Plotly
            colors_pred = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in colors_pred]
    except Exception as e:
        print(f"❌ Error cargando predicción: {e}")
        return

    # 3. Crear Visualización
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(f"Original (Blender)", f"Predicción IA")
    )

    # Panel Izquierdo: Original
    fig.add_trace(
        go.Scatter3d(x=pts_gt[:, 0], y=pts_gt[:, 1], z=pts_gt[:, 2],
                     mode='markers', marker=dict(size=2, color=colors_gt), name="Manual"),
        row=1, col=1
    )

    # Panel Derecho: Predicción
    fig.add_trace(
        go.Scatter3d(x=pts_pred[:, 0], y=pts_pred[:, 1], z=pts_pred[:, 2],
                     mode='markers', marker=dict(size=2, color=colors_pred), name="IA"),
        row=1, col=2
    )

    fig.update_layout(template="plotly_dark", title=f"Comparación: {filename}")
    
    output_html = f"outputs/CHECK_{Path(filename).stem}.html"
    fig.write_html(output_html)
    print(f"\n✅ ¡Hecho! Archivo creado: {output_html}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python compare_inference.py 24B_800N_decimated.ply")
    else:
        visualize_comparison(sys.argv[1])