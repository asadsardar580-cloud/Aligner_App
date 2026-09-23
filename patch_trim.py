import re

with open('core_geometry.py', 'r') as f:
    orig = f.read()

# Modify trim_to_arch signature
orig = orig.replace(
    'def trim_to_arch(verts: np.ndarray, faces: np.ndarray, arch_frame: dict,\n                 margin_mm: float = ARCH_TRIM_MARGIN_MM,\n                 curve: np.ndarray | None = None):',
    'def trim_to_arch(verts: np.ndarray, faces: np.ndarray, arch_frame: dict,\n                 margin_mm=ARCH_TRIM_MARGIN_MM,\n                 curve: np.ndarray | None = None):'
)

# Modify margin_mm type check and logic
old_logic = """    margin_mm = float(margin_mm)
    if margin_mm <= 0:
        raise ValueError(f"margin_mm must be positive, got {margin_mm}.")"""

new_logic = """    if isinstance(margin_mm, (int, float)):
        margin_mm = float(margin_mm)
        if margin_mm <= 0:
            raise ValueError(f"margin_mm must be positive, got {margin_mm}.")
        is_scalar_margin = True
    else:
        margin_mm = np.asarray(margin_mm, float)
        is_scalar_margin = False"""
orig = orig.replace(old_logic, new_logic)

old_dist_logic = """    from scipy.spatial import cKDTree
    dist, _ = cKDTree(samples).query(centroids2d)
    keep = dist < margin_mm
    if not keep.any():
        raise ValueError(
            f"No face centroid lies within {margin_mm:.1f}mm of the fitted arch curve. "
            f"The occlusal plane landmarks are almost certainly wrong.")"""

new_dist_logic = """    from scipy.spatial import cKDTree
    dist, idx = cKDTree(samples).query(centroids2d)
    if is_scalar_margin:
        keep = dist < margin_mm
    else:
        if len(margin_mm) != len(samples):
            raise ValueError("margin_mm array must match number of curve samples")
        keep = dist < margin_mm[idx]
        
    if not keep.any():
        raise ValueError(
            f"No face centroid lies within the margin of the fitted arch curve. "
            f"The occlusal plane landmarks are almost certainly wrong.")"""
orig = orig.replace(old_dist_logic, new_dist_logic)

with open('core_geometry.py', 'w') as f:
    f.write(orig)
