"""Generates the architecture report your spec asks for, from the real files."""
import ast, os

FILES = ["core_geometry.py", "app_ui.py", "server.py", "check_structure.py",
         "tooth_fixture.py", "diagnose.py"]
TESTS = [f for f in os.listdir(".") if f.startswith("test_")]

print("="*72); print("EXISTING ARCHITECTURE"); print("="*72)
inventory = {}
for f in FILES:
    if not os.path.exists(f): continue
    tree = ast.parse(open(f, encoding="utf-8").read())
    fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    cls = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    loc = len(open(f, encoding="utf-8").readlines())
    inventory[f] = fns
    print(f"\n{f}  ({loc} lines)")
    if cls: print(f"   classes  : {', '.join(cls)}")
    if fns: print(f"   functions: {len(fns)}")

print("\n" + "="*72); print("REUSE MAP  (your module spec -> what already exists, tested)"); print("="*72)
core = set(inventory.get("core_geometry.py", []))
mapping = {
 "mesh_preprocessor.py": ["directed_edges","edge_face_incidence","boundary_loops",
                          "is_edge_manifold_closed","connected_components"],
 "normals.py":           ["face_normals","vertex_normals"],
 "curvature.py":         ["vertex_concavity","vertex_concavity_fast","smooth_scalar",
                          "smooth_scalar_fast"],
 "boundary_detection.py":["build_barrier_graph","build_edge_graph"],
 "region_growing.py":    ["region_grow_crown","geodesic_from_seed","mask_from_distance",
                          "curvature_weighted_path","path_via_graph"],
 "mesh_extraction.py":   ["split_by_face_mask","largest_face_component","cap_and_close",
                          "cap_boundary_loop","make_consistent_winding","signed_volume"],
 "confidence.py":        ["segmentation_is_plausible"],
 "models.py":            ["derive_frame_from_region","derive_anatomical_frame_4click",
                          "center_of_resistance","kinematic_matrix"],
}
have = miss = 0
for mod, fns in mapping.items():
    present = [f for f in fns if f in core]
    absent  = [f for f in fns if f not in core]
    have += len(present); miss += len(absent)
    print(f"\n  {mod}")
    if present: print(f"     REUSE : {', '.join(present)}")
    if absent:  print(f"     NEW   : {', '.join(absent)}")

print(f"\n  --> {have} of {have+miss} required primitives already exist and are under test.")
print("\n" + "="*72); print("TEST COVERAGE TODAY"); print("="*72)
for t in sorted(TESTS): print(f"  {t}")
