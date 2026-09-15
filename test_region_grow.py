"""Tests the magic-wand selection: does one click select the whole crown,
wrap to hidden surfaces, stop at the sulcus, and stay stable across a range
of tolerance values?"""
import numpy as np, core_geometry as cg
from tooth_fixture import tooth_on_base

verts, faces, crown_r = tooth_on_base()
edges = cg.directed_edges(faces)
conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges), faces, 2, edges=edges)
print(f"Fixture: {len(verts):,} verts / {len(faces):,} faces, crown radius {crown_r}mm\n")

apex = np.array([0.0, 0.0, 7.0])   # one click on top of the crown

print("TOLERANCE SWEEP  (does the slider have a forgiving plateau?)")
results = {}
for tol in [2, 4, 6, 8, 10, 14, 20, 30, 50, 80]:
    mask, dist = cg.region_grow_crown(verts, faces, conc, apex, tol, edges=edges)
    frac = mask.sum()/len(faces)
    sel = verts[np.unique(faces[mask])] if mask.sum() else np.empty((0,3))
    radius = np.sqrt(sel[:,0]**2 + sel[:,1]**2).max() if len(sel) else 0
    results[tol] = (frac, radius)
    print(f"  tol={tol:>3}: {mask.sum():>6,} faces ({frac:>5.1%})  max radius {radius:5.2f}mm")

# The crown sits inside r=4mm. A good selection reaches the sulcus (~4mm) and stops.
plateau = [t for t,(f,r) in results.items() if 3.5 <= r <= 5.0]
assert len(plateau) >= 3, f"expected a forgiving plateau of tolerances, got {plateau}"
print(f"\nPASS  tolerances {plateau} all select the crown and stop at the sulcus")
print(f"      -> the slider has a wide safe range, not one magic value")

# --- full pipeline at a mid-plateau tolerance ---
tol = plateau[len(plateau)//2]
mask, dist = cg.region_grow_crown(verts, faces, conc, apex, tol, edges=edges)
mask = cg.largest_face_component(faces, mask)
(cv, cf), (bv, bf) = cg.split_by_face_mask(verts, faces, mask)
ok, reason = cg.segmentation_is_plausible(len(faces), len(cf), len(bf), list(range(10)))
print(f"\nFull pipeline at tol={tol}:")
print(f"  crown {len(cf):,} tris ({len(cf)/len(faces):.1%}), base {len(bf):,} tris, plausible={ok}")
assert ok, reason

rim_loops = cg.boundary_loops(cf)
print(f"  crown boundary loops: {len(rim_loops)} (lengths {[len(l) for l in rim_loops]})")
rim = cv[max(rim_loops, key=len)].copy()

cv2, cf2 = cg.cap_and_close(cv, cf)
bv2, bf2 = cg.cap_and_close(bv, bf)
cf2 = cg.make_consistent_winding(cv2, cf2)
assert cg.is_edge_manifold_closed(cf2), "crown not watertight"
assert cg.is_edge_manifold_closed(bf2), "base not watertight"
print(f"  crown watertight, volume {cg.signed_volume(cv2, cf2):.1f} mm3")

# --- 2-click frame: both clicks on the SAME (visible) side ---
mesial = np.array([-crown_r, 0.0, 0.0])
distal = np.array([ crown_r, 0.0, 0.0])
fr = cg.derive_frame_from_region(mesial, distal, cv2, rim)
for x,y in (("u_md","u_bl"),("u_md","u_oa"),("u_bl","u_oa")):
    assert abs(np.dot(fr[x], fr[y])) < 1e-9, f"{x}/{y} not orthogonal"
tilt = np.degrees(np.arccos(np.clip(abs(np.dot(fr["u_oa"], [0,0,1.0])), 0, 1)))
c_res = cg.center_of_resistance(fr, 10.0)
print(f"  frame orthonormal, u_OA tilt {tilt:.2f}deg, C_res {c_res.round(2)}")
assert tilt < 2.0, f"long axis should be near-vertical, got {tilt:.1f}deg"
assert c_res[2] < fr["centroid"][2], "C_res must be apical"

# --- the key claim: selection wraps to surfaces a camera cannot see ---
sel_pts = verts[np.unique(faces[mask])]
angles = np.degrees(np.arctan2(sel_pts[:,1], sel_pts[:,0]))
coverage = np.histogram(angles, bins=12, range=(-180,180))[0]
assert (coverage > 0).all(), f"selection must wrap 360 deg, empty sectors: {coverage}"
print(f"\nPASS  selection covers all 12 angular sectors — it wrapped fully around the")
print(f"      crown, including the side facing away from the camera.")
print("\nALL REGION-GROW TESTS PASSED")
