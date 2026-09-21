import pymeshlab as ml
import os
import vedo

def visualize_curvature(source_path: str, target_path: str):
    ms = ml.MeshSet()

    ms.load_new_mesh(source_path)
    ms.load_filter_script('Compute_curvature_principal_directions.mlx')
    ms.apply_filter_script()

    ms.save_current_mesh(target_path, binary=False)
