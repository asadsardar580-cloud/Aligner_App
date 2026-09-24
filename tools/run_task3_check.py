import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_geometry as cg
import stl_io
import arch_frame
import self_intersection as si

def main():
    with open("case_lower.stl", "rb") as f: raw = f.read()
    verts, faces = stl_io.parse_stl_bytes(raw)
    verts, faces, _ = cg.sanitize_scan(verts, faces)
    verts, faces, _ = cg.condition_mesh(verts, faces)
    
    with open("case_lower.stl_output.json", 'r') as lf: 
        case = json.load(lf)
    labels = np.array(case.get('labels', []))
    
    centroid = verts.mean(axis=0)
    d = verts - centroid
    _, _, vt = np.linalg.svd(d[::37], full_matrices=False)
    occ_axis = vt[2]
    slab = verts[(d @ occ_axis) > np.percentile(d @ occ_axis, 90)]
    s = slab - slab.mean(axis=0)
    _, _, sv_ = np.linalg.svd(s[::7], full_matrices=False)
    along = s @ sv_[0]
    across = s @ sv_[1]
    pts = [slab[np.argmin(along)], slab[np.argmax(along)], slab[np.argmax(np.abs(across)) if abs(across).max() > 0 else 0]]
    frame = arch_frame.fit_occlusal_frame(list(map(float, pts[0])), list(map(float, pts[1])), list(map(float, pts[2])), centroid.tolist())
    
    tv, tf, tinfo = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    
    import scipy.sparse.csgraph
    graph_tv = cg.build_edge_graph(tv, tf, np.zeros(len(tv)))
    v_to_fdi = np.zeros(len(verts), dtype=int)
    if len(labels) > 0: v_to_fdi = labels[:len(verts)]
    tooth_idx = np.where(v_to_fdi[:len(tv)] > 0)[0]
    if len(tooth_idx) > 0:
        dist_to_tooth = scipy.sparse.csgraph.dijkstra(graph_tv, directed=False, indices=tooth_idx, min_only=True)
    else:
        dist_to_tooth = np.full(len(tv), float('inf'))
        
    protected_faces = np.min(dist_to_tooth[tf], axis=1) < 3.0
    
    tv2, tf2, rim, uinfo = cg.clear_undercut_periphery(tv, tf, frame, tinfo["rim_loop"], protected_mask=protected_faces)
    cv, cf, base_info = cg.build_cast_base(tv2, tf2, frame, rim=rim)
    
    blob = cg.write_binary_stl_bytes(cv, cf)
    rv, rf = stl_io.parse_stl_bytes(blob)
    
    rep = si.self_intersection_report(rv, rf)
    print("New total pairs:", rep["intersecting_pairs"])
    
    n_scan = len(tf2)
    n_wall = len(rim) * 2
    
    pairs = si.candidate_pairs(rv, rf)
    hit, _ = si._pairs_intersect(rv, rf, pairs, si.DEFAULT_TOUCH_TOL_MM)
    intersecting = pairs[hit]
    
    counts = {"scan-wall": 0, "scan-floor": 0, "wall-wall": 0, "wall-floor": 0, "floor-floor": 0, "scan-scan": 0}
    for a, b in intersecting:
        ta = "scan" if a < n_scan else ("wall" if a < n_scan + n_wall else "floor")
        tb = "scan" if b < n_scan else ("wall" if b < n_scan + n_wall else "floor")
        key = f"{ta}-{tb}"
        if tb + "-" + ta in ["scan-wall", "scan-floor", "wall-floor"]: key = f"{tb}-{ta}"
        counts[key] += 1
        
    print("New per-class counts:", counts)
    print(f"Area deleted: {cg.mesh_surface_area(tv, tf) - cg.mesh_surface_area(tv2, tf2):.2f} mm2")
    
if __name__ == "__main__":
    main()
