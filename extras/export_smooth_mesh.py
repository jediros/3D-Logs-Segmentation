import sys
import numpy as np
import open3d as o3d
from pathlib import Path

def generate_smooth_mesh(pts, labels, target_label):
    """
    Crea una malla suavizada y limpia para una clase específica.
    """
    # 1. Filtrar puntos por clase
    target_pts = pts[labels == target_label]
    if len(target_pts) < 100:
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_pts)
    
    # 2. LIMPIEZA DE RUIDO (Más estricta para evitar 'picos')
    # nb_neighbors: puntos a mirar, std_ratio: qué tan lejos pueden estar (menor = más estricto)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    
    # 3. ESTIMAR NORMALES
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=30))
    
    # 4. TRIANGULACIÓN (Alpha Shapes)
    # Bajamos el alpha a 3.5 para que los parches sean más ceñidos a los puntos
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=3.5)

    # 5. SUAVIZADO Y OPTIMIZACIÓN
    if len(mesh.triangles) > 0:
        # Suavizado Laplaciano (quita lo rugoso/puntiagudo)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=15)
        mesh.compute_vertex_normals()
        
        # SIMPLIFICACIÓN (Nombre corregido: simplify_quadric_decimation)
        # Esto reduce el peso del archivo y suaviza visualmente las uniones
        target_triangles = min(len(mesh.triangles), 15000)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    
    return mesh

def main(filename):
    segmented_path = Path("outputs") / (Path(filename).stem + "_segmented.ply")
    if not segmented_path.exists():
        print(f"❌ No existe: {segmented_path}")
        return

    print(f"Procesando {filename} para obtener superficies suaves...")
    pcd_ia = o3d.io.read_point_cloud(str(segmented_path))
    pts = np.asarray(pcd_ia.points)
    colors = np.asarray(pcd_ia.colors)

    # Separación: Rojo < 0.5 es Corteza en tus materiales
    labels = (colors[:, 0] < 0.5).astype(int)

    # Exportar CORTEZA SUAVE (Rojo)
    print("Generando malla de corteza...")
    mesh_bark = generate_smooth_mesh(pts, labels, 1)
    if mesh_bark:
        mesh_bark.paint_uniform_color([0.8, 0.2, 0.2]) 
        out_bark = Path("outputs") / (Path(filename).stem + "_SMOOTH_BARK.ply")
        o3d.io.write_triangle_mesh(str(out_bark), mesh_bark)
        print(f"✅ Guardado: {out_bark}")

    # Exportar MADERA SUAVE (Gris)
    print("Generando malla de madera...")
    mesh_wood = generate_smooth_mesh(pts, labels, 0)
    if mesh_wood:
        mesh_wood.paint_uniform_color([0.7, 0.7, 0.7]) 
        out_wood = Path("outputs") / (Path(filename).stem + "_SMOOTH_WOOD.ply")
        o3d.io.write_triangle_mesh(str(out_wood), mesh_wood)
        print(f"✅ Guardado: {out_wood}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Uso: python export_smooth_mesh.py 15A_color.ply")