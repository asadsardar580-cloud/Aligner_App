import numpy as np, core_geometry as cg
from tooth_fixture import tooth_on_base
verts, faces, cr = tooth_on_base()
edges = cg.directed_edges(faces)
conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges), faces, 2, edges=edges)
graph = cg.build_barrier_graph(verts, faces, conc, edges=edges)

# exactly what the UI does: seed click -> distance field -> threshold -> cut
dist = cg.geodesic_from_seed(graph, verts, np.array([0.,0.,7.]))
mask = cg.mask_from_distance(faces, dist, 10.0)
mask = cg.largest_face_component(faces, mask)
(cv,cf),(bv,bf) = cg.split_by_face_mask(verts, faces, mask)
ok, why = cg.segmentation_is_plausible(len(faces), len(cf), len(bf), loop=[])
assert ok, why
rim = cv[max(cg.boundary_loops(cf), key=len)].copy()
cv2,cf2 = cg.cap_and_close(cv,cf); bv2,bf2 = cg.cap_and_close(bv,bf)
cf2 = cg.make_consistent_winding(cv2,cf2)
assert cg.is_edge_manifold_closed(cf2) and cg.is_edge_manifold_closed(bf2)
fr = cg.derive_frame_from_region(np.array([-cr,0,0.]), np.array([cr,0,0.]), cv2, rim)
c = cg.center_of_resistance(fr, 10.0)
M = cg.kinematic_matrix(fr, c, tip_deg=5.0)
moved = cg.apply_matrix(cv2, M)
assert np.allclose(cg.apply_matrix(np.array([c]), M)[0], c, atol=1e-9)
assert np.allclose(cg.apply_matrix(cv2, cg.kinematic_matrix(fr,c)), cv2, atol=1e-12)
print(f"PASS  worker path: crown {len(cf2):,} tris watertight, C_res fixed under tip, identity exact")
