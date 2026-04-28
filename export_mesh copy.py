import sys
import numpy as np
import open3d as o3d
from pathlib import Path

def generate_ball_pivoting_mesh(pts, labels, target_label):
    """
    Convierte puntos en una malla usando Ball Pivoting.
    Es mucho más limpio para superficies de troncos que Alpha Shapes.
    """
    # 1. Filtrar puntos por clase
    target_pts = pts[labels == target_label]
    if len(target_pts) < 100:
        return None
    
    # 2. Crear objeto de nube de puntos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_pts)
    
    # 3. Limpieza de ruido (Elimina puntos aislados que crean 'picos')
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 4. Estimar normales (VITAL para una malla limpia)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)

    # 5. Calcular radios automáticos para el algoritmo Ball Pivoting
    # Se basa en la distancia promedio entre puntos para que no queden huecos
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    # Probamos con 4 tamaños de 'pelota' para cerrar la malla perfectamente
    radii = [avg_dist * factor for factor in [1.5, 3, 6, 12]]
    
    # 6. Generar la malla
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    return mesh

def main(filename):
    # 1. Rutas de archivos
    segmented_path = Path("outputs") / (Path(filename).stem + "_segmented.ply")
    
    if not segmented_path.exists():
        print(f"❌ No se encuentra el archivo segmentado: {segmented_path}")
        return

    print(f"Cargando puntos de: {segmented_path.name}...")
    
    # 2. Leer la nube de puntos guardada por la IA
    pcd_ia = o3d.io.read_point_cloud(str(segmented_path))
    pts = np.asarray(pcd_ia.points)
    colors = np.asarray(pcd_ia.colors)

    # 3. Lógica de separación (Rojo < 0.5 es Corteza)
    labels = (colors[:, 0] < 0.5).astype(int)

    print("Reconstruyendo superficies sólidas (Ball Pivoting)...")

    # 4. Malla de CORTEZA
    mesh_bark = generate_ball_pivoting_mesh(pts, labels, 1)
    if mesh_bark:
        mesh_bark.paint_uniform_color([1, 0, 0]) # Rojo Puro
        out_bark = Path("outputs") / (Path(filename).stem + "_SOLID_BARK.ply")
        o3d.io.write_triangle_mesh(str(out_bark), mesh_bark)
        print(f"✅ Corteza sólida guardada: {out_bark}")

    # 5. Malla de MADERA
    mesh_wood = generate_ball_pivoting_mesh(pts, labels, 0)
    if mesh_wood:
        mesh_wood.paint_uniform_color([0.7, 0.7, 0.7]) # Gris Madera
        out_wood = Path("outputs") / (Path(filename).stem + "_SOLID_WOOD.ply")
        o3d.io.write_triangle_mesh(str(out_wood), mesh_wood)
        print(f"✅ Madera sólida guardada: {out_wood}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Uso: python export_refined_mesh.py 15A_color.ply")