"""Exercises the exact core_geometry call sequence SegmentWorker.run() uses,
without Qt. If this passes, any failure in app_ui is in the Qt/VTK layer."""
import numpy as np, core_geometry as cg
from test_core_geometry import grooved_tube

verts, faces, thetas, groove_z, height = grooved_tube(300, 120)
edges = cg.directed_edges(faces)
conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges), faces, 2, edges=edges)

anchors = np.array([[5*np.cos(t), 5*np.sin(t), height*0.5+1.8*np.sin(t)]
                    for t in (0.0, 2*np.pi/3, 4*np.pi/3)])

loop = cg.closed_loop_through_anchors_fast(verts, faces, conc, anchors, 8.0, edges=edges)
assert loop, "no loop"
(cv, cf), (bv, bf) = cg.split_mesh_by_loop(verts, faces, loop, anchors[2])
rim_loops = cg.boundary_loops(cf)
rim = cv[max(rim_loops, key=len)].copy()
cv2, cf2 = cg.cap_and_close(cv, cf)
bv2, bf2 = cg.cap_and_close(bv, bf)
cf2 = cg.make_consistent_winding(cv2, cf2)
assert cg.is_edge_manifold_closed(cf2), "crown not watertight"
assert cg.is_edge_manifold_closed(bf2), "base not watertight"

frame = cg.derive_anatomical_frame(anchors[0], anchors[1], anchors[2], cv2, rim)
c_res = cg.center_of_resistance(frame, 10.0)
M = cg.kinematic_matrix(frame, c_res, tip_deg=5.0, d_md=0.5)
moved = cg.apply_matrix(cv2, M)
assert moved.shape == cv2.shape
assert np.allclose(cg.apply_matrix(np.array([c_res]), cg.kinematic_matrix(frame, c_res, tip_deg=5.0))[0], c_res, atol=1e-9)

# reset-to-T0 must be exact (this is why _apply_kinematics transforms from rest)
back = cg.apply_matrix(cv2, cg.kinematic_matrix(frame, c_res))
assert np.allclose(back, cv2, atol=1e-12), "identity transform not exact"

print(f"PASS  UI dataflow: {len(loop)}-vert margin, crown {len(cf2):,} tris watertight, "
      f"base {len(bf2):,} tris watertight, C_res fixed, identity exact")
