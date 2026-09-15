import numpy as np, core_geometry as cg
from tooth_fixture import tooth_on_base

verts, faces, crown_r = tooth_on_base()
edges = cg.directed_edges(faces)
conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges), faces, 2, edges=edges)
print(f"Fixture: {len(verts):,} verts / {len(faces):,} faces, crown radius {crown_r}mm\n")

def ring(degs, r):
    return np.array([[r*np.cos(np.radians(d)), r*np.sin(np.radians(d)), 0.0] for d in degs])

def run(degs, label, expect_ok):
    a = ring(degs, crown_r)
    loop = cg.closed_loop_through_anchors_fast(verts, faces, conc, a, 8.0, edges=edges)
    if not loop:
        print(f"  {label}: no loop"); return False
    (cv,cf),(bv,bf),info = cg.split_mesh_by_loop_auto(verts, faces, loop)
    ok, reason = cg.segmentation_is_plausible(len(faces), len(cf), len(bf), loop)
    print(f"  {label}: loop={len(loop)} | crown={len(cf):,} tris ({len(cf)/len(faces):.1%}) | "
          f"components={info['n_components']} | ok={ok}")
    if not ok: print(f"       -> {reason[:100]}")
    assert ok == expect_ok, f"expected ok={expect_ok}, got {ok}"
    return ok

print("BROKEN — 3 anchors clustered on one side (your reported failure):")
run([0, 25, 50], "0/25/50 deg", expect_ok=False)

print("\nBROKEN — 3 anchors, legs share a route:")
run([0, 10, 350], "0/10/350 deg", expect_ok=False)

print("\nGOOD — 4 anchors in circuit (mesial/buccal/distal/lingual):")
run([0, 90, 180, 270], "0/90/180/270 deg", expect_ok=True)

# full pipeline on the good case
a = ring([0, 90, 180, 270], crown_r)
loop = cg.closed_loop_through_anchors_fast(verts, faces, conc, a, 8.0, edges=edges)
(cv,cf),(bv,bf),info = cg.split_mesh_by_loop_auto(verts, faces, loop)
rim = cv[max(cg.boundary_loops(cf), key=len)].copy()
cv2, cf2 = cg.cap_and_close(cv, cf)
bv2, bf2 = cg.cap_and_close(bv, bf)
cf2 = cg.make_consistent_winding(cv2, cf2)
assert cg.is_edge_manifold_closed(cf2), "crown not watertight"
frame = cg.derive_anatomical_frame(a[0], a[2], a[1], cv2, rim)   # mesial, distal, buccal
c_res = cg.center_of_resistance(frame, 10.0)
print(f"\nFull pipeline: crown watertight, vol={cg.signed_volume(cv2,cf2):.1f}mm3, "
      f"C_res {c_res.round(2)} (apical of crown centroid {frame['centroid'].round(2)})")
assert c_res[2] < frame['centroid'][2], "C_res must sit apical (below) the crown centroid"
print("\nALL LOOP-FAILURE TESTS PASSED")

# --- 4-click frame: axes measured between opposing landmarks ---
print("\n4-CLICK FRAME vs 3-CLICK FRAME:")
f3 = cg.derive_anatomical_frame(a[0], a[2], a[1], cv2, rim)
f4 = cg.derive_anatomical_frame_4click(a[0], a[1], a[2], a[3], cv2, rim)
for nm, fr in (("3-click", f3), ("4-click", f4)):
    for x,y in (("u_md","u_bl"),("u_md","u_oa"),("u_bl","u_oa")):
        assert abs(np.dot(fr[x], fr[y])) < 1e-9, f"{nm} {x}/{y} not orthogonal"
    tilt = np.degrees(np.arccos(np.clip(abs(np.dot(fr["u_oa"], [0,0,1.0])), 0, 1)))
    c = cg.center_of_resistance(fr, 10.0)
    print(f"  {nm}: u_OA tilt from vertical = {tilt:5.1f}deg | C_res = {c.round(2)}")

tilt4 = np.degrees(np.arccos(np.clip(abs(np.dot(f4["u_oa"], [0,0,1.0])), 0, 1)))
assert tilt4 < 1.0, f"4-click u_OA should be near-vertical on a symmetric fixture, got {tilt4:.1f}deg"
print(f"\nPASS  4-click frame gives a near-vertical long axis ({tilt4:.2f}deg) on a symmetric crown;")
print("      the 3-click version tilts because its FA anchor now sits at the margin.")
