"""Does precomputing the whole arch pick the right separation on its own,
and give a usable per-tooth record without any slider?"""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch
from tooth_segmentation import arch_geometry

for n_teeth, spacing in [(5, 7.0), (8, 6.4), (6, 5.8)]:
    verts, faces, centres = synthetic_arch(n_teeth=n_teeth, spacing=spacing,
                                           crown_r=3.2, grid=160, blend=3.0)
    edges = cg.directed_edges(faces)
    conc = cg.boundary_field(verts, faces, edges=edges)
    frame = arch_geometry.estimate_arch_frame(verts)

    pre = cg.precompute_arch(verts, faces, frame.occlusal_normal, conc,
                             edges=edges)
    found = len(pre["teeth"])
    print(f"\nArch of {n_teeth} teeth at {spacing}mm spacing")
    print(f"  auto-chose separation = {pre['chosen_separation_mm']}mm  "
          f"-> found {found} teeth  (score {pre['score']:.3f})")
    print(f"  sweep: {[(s, n) for s, n, _ in pre['trials']]}")

    sizes = [t["n_faces"] for t in pre["teeth"]]
    print(f"  tooth sizes: min {min(sizes):,} max {max(sizes):,} "
          f"(ratio {max(sizes)/max(min(sizes),1):.1f}x)")

    withframe = sum(1 for t in pre["teeth"] if "c_res" in t)
    print(f"  teeth with derived C_res: {withframe}/{found}")
    assert found == n_teeth, f"expected {n_teeth}, found {found}"
    assert withframe == found, "every tooth needs a frame precomputed"

    # a click anywhere on a crown must resolve instantly to one tooth
    hit_face = None
    fc = verts[faces].mean(axis=1)
    d = np.linalg.norm(fc[:, :2] - centres[1], axis=1)
    probe = int(np.argmin(d))
    owner = [t["id"] for t in pre["teeth"] if probe in set(t["face_indices"].tolist())]
    print(f"  probe face {probe} -> {owner[0] if owner else 'UNASSIGNED'}")
    assert owner, "a click on a crown must land on a precomputed tooth"

print("\nPASS  precompute picks its own separation and yields per-tooth")
print("      face sets, frames and C_res - no tolerance slider involved.")
