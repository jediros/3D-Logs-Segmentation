"""
data/loader.py
--------------
Loads .ply files exported from Blender with material labels.

Supported format (Blender with two materials + vertex color):
    element vertex N
        property float x / y / z
        property uchar red / green / blue   (optional, scanner RGB)
        property float s / t                (UV coords, optional)
        property float label_1              (bark weight 0.0-1.0)
        property float label_0              (wood weight 0.0-1.0)
    element face M
        property list uchar uint vertex_indices

Output shape based on configuration:
    use_normals=False, use_rgb=False  ->  (N, 3)   xyz
    use_normals=True,  use_rgb=False  ->  (N, 6)   xyz + normals
    use_normals=False, use_rgb=True   ->  (N, 6)   xyz + rgb
    use_normals=True,  use_rgb=True   ->  (N, 9)   xyz + normals + rgb

Label conversion:
    label_1 > threshold  ->  1  (bark)
    label_1 <= threshold ->  0  (wood)
    ignore_boundary=True: points in mixed zone -> -1 (ignored in loss)
"""

from pathlib import Path
import numpy as np


# -----------------------------------------------------------------------------
# Low-level parsing
# -----------------------------------------------------------------------------

def _parse_ply_header(f) -> dict:
    properties   = []
    n_vertices   = 0
    n_faces      = 0
    current_elem = None

    while True:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line == "end_header":
            data_offset = f.tell()
            break
        if line.startswith("element vertex"):
            n_vertices   = int(line.split()[-1])
            current_elem = "vertex"
        elif line.startswith("element face"):
            n_faces      = int(line.split()[-1])
            current_elem = "face"
        elif line.startswith("property") and current_elem == "vertex":
            parts = line.split()
            properties.append({"type": parts[1], "name": parts[2]})

    return {
        "data_offset": data_offset,
        "n_vertices":  n_vertices,
        "n_faces":     n_faces,
        "properties":  properties,
    }


def _build_vertex_dtype(properties: list) -> np.dtype:
    type_map = {
        "float":  "<f4", "float32": "<f4",
        "double": "<f8", "float64": "<f8",
        "int":    "<i4", "int32":   "<i4",
        "uint":   "<u4", "uint32":  "<u4",
        "uchar":  "u1",  "uint8":   "u1",   # RGB viene como uchar
        "char":   "i1",
    }
    return np.dtype([(p["name"], type_map.get(p["type"], "<f4")) for p in properties])


def _parse_faces(f, n_faces: int) -> np.ndarray:
    """Parses triangular faces: 1 byte (count=3) + 3 x uint32 = 13 bytes."""
    face_bytes = f.read(n_faces * 13)
    faces = np.zeros((n_faces, 3), dtype=np.int32)
    for i in range(n_faces):
        off = i * 13 + 1
        faces[i] = np.frombuffer(face_bytes[off:off + 12], dtype="<u4")
    return faces


# -----------------------------------------------------------------------------
# Vertex normals
# -----------------------------------------------------------------------------

def compute_vertex_normals(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Computes per-vertex normals by averaging adjacent face normals.

    Args:
        pts:   (N, 3) float32
        faces: (F, 3) int32

    Returns:
        normals: (N, 3) float32 normalized to unit length
    """
    edge1 = pts[faces[:, 1]] - pts[faces[:, 0]]
    edge2 = pts[faces[:, 2]] - pts[faces[:, 0]]
    face_normals = np.cross(edge1, edge2).astype(np.float32)
    mag = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= (mag + 1e-8)

    vertex_normals = np.zeros_like(pts, dtype=np.float32)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)

    mag = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    return vertex_normals / (mag + 1e-8)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_ply_labeled(
    filepath,
    label_threshold: float = 0.5,
    ignore_boundary: bool = False,
    boundary_margin: float = 0.1,
    compute_normals: bool = True,
    use_rgb: bool = False,
):
    """
    Loads a Blender .ply file with material labels and optional features.

    Args:
        filepath:         path to the .ply file
        label_threshold:  float -> int threshold (default 0.5)
        ignore_boundary:  mark boundary zone as -1
        boundary_margin:  boundary margin (default 0.1)
        compute_normals:  compute normals from mesh faces
        use_rgb:          include normalized RGB channels as features

    Returns:
        cloud:    (N, C) float32
                  C=3  xyz only
                  C=6  xyz+normals  or  xyz+rgb
                  C=9  xyz+normals+rgb
        labels:   (N,) int32    0=wood  1=bark  -1=ignore
        metadata: dict with log statistics

    Example:
        # With RGB enabled
        cloud, labels, meta = load_ply_labeled("log.ply", use_rgb=True)
        print(cloud.shape)   # (422482, 9)  xyz + normals + rgb

        # Without RGB (previous behavior)
        cloud, labels, meta = load_ply_labeled("log.ply", use_rgb=False)
        print(cloud.shape)   # (422482, 6)  xyz + normals
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PLY file not found: {filepath}")

    with open(filepath, "rb") as f:
        hdr   = _parse_ply_header(f)
        n_v   = hdr["n_vertices"]
        n_f   = hdr["n_faces"]
        props = hdr["properties"]
        dtype = _build_vertex_dtype(props)

        raw   = np.frombuffer(f.read(n_v * dtype.itemsize), dtype=dtype)
        faces = _parse_faces(f, n_f) if n_f > 0 else None

    prop_names = [p["name"] for p in props]

    # ── XYZ coordinates ────────────────────────────────────────────────────────
    pts = np.stack([
        raw["x"].astype(np.float32),
        raw["y"].astype(np.float32),
        raw["z"].astype(np.float32),
    ], axis=1)

    # ── Labels ────────────────────────────────────────────────────
    if "label_1" in prop_names:
        l1 = raw["label_1"].astype(np.float32)
    elif "corteza" in prop_names:
        l1 = raw["corteza"].astype(np.float32)
    else:
        l1 = np.zeros(n_v, dtype=np.float32)

    labels = np.where(l1 > label_threshold, 1, 0).astype(np.int32)
    if ignore_boundary:
        ambiguous = (l1 > boundary_margin) & (l1 < (1.0 - boundary_margin))
        labels[ambiguous] = -1

    # ── Build feature array ─────────────────────────────────────────────
    feature_parts = [pts]   # always includes XYZ

    if compute_normals and faces is not None:
        normals = compute_vertex_normals(pts, faces)
        feature_parts.append(normals)

    if use_rgb:
        # Check that the PLY has RGB fields
        has_rgb = all(c in prop_names for c in ["red", "green", "blue"])
        if has_rgb:
            # uchar [0,255] -> float32 [0.0, 1.0]
            r = raw["red"].astype(np.float32)   / 255.0
            g = raw["green"].astype(np.float32) / 255.0
            b = raw["blue"].astype(np.float32)  / 255.0
            rgb = np.stack([r, g, b], axis=1)
            feature_parts.append(rgb)
        else:
            print(f"  [WARNING] use_rgb=True but {filepath.name} has no RGB fields. "
                  f"Available fields: {prop_names}")

    cloud = np.concatenate(feature_parts, axis=1).astype(np.float32)

    # ── Metadata ──────────────────────────────────────────────────
    n_bark = int((labels == 1).sum())
    n_wood = int((labels == 0).sum())
    dims   = pts.max(axis=0) - pts.min(axis=0)

    metadata = {
        "filepath":       str(filepath),
        "n_vertices":     n_v,
        "n_faces":        n_f,
        "n_bark":         n_bark,
        "n_wood":         n_wood,
        "n_ignored":      int((labels == -1).sum()),
        "bark_fraction":  n_bark / max(n_bark + n_wood, 1),
        "has_normals":    compute_normals and faces is not None,
        "has_rgb":        use_rgb and all(c in prop_names for c in ["red","green","blue"]),
        "n_features":     cloud.shape[1],
        "properties":     prop_names,
        "bounds_min":     pts.min(axis=0).tolist(),
        "bounds_max":     pts.max(axis=0).tolist(),
        "dimensions_mm":  dims.tolist(),
    }

    return cloud, labels, metadata


def load_ply_for_inference(filepath, compute_normals: bool = True, use_rgb: bool = False):
    """Loads .ply without requiring labels — for inference on new logs."""
    cloud, _, meta = load_ply_labeled(
        filepath,
        compute_normals=compute_normals,
        use_rgb=use_rgb,
    )
    return cloud, meta


def scan_ply_folder(folder) -> list:
    """Lists all .ply files in a folder, sorted."""
    folder = Path(folder)
    return sorted(list(folder.glob("*.ply")) + list(folder.glob("*.PLY")))