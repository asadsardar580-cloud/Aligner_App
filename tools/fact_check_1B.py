import sys, os
import json
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core_geometry as cg
import stl_io
import self_intersection as si
import arch_frame

def digest(v, f):
    return hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest()

def classify(pairs, hit, kind, n_scan, n_wall):
    intersecting = pairs[hit]
    intersecting_kind = kind[hit]
    counts = {"scan-wall": 0, "scan-floor": 0, "wall-wall": 0, "wall-floor": 0, "floor-floor": 0, "scan-scan": 0}
    for p in intersecting:
        a, b = p
        type_a = "scan" if a < n_scan else ("wall" if a < n_scan + n_wall else "floor")
        type_b = "scan" if b < n_scan else ("wall" if b < n_scan + n_wall else "floor")
        key = f"{type_a}-{type_b}"
        if key not in counts: key = f"{type_b}-{type_a}"
        counts[key] = counts.get(key, 0) + 1
    return len(intersecting), counts

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
    if (slab - centroid).shape[0] < 10: slab = verts
    s = slab - slab.mean(axis=0)
    _, _, sv_ = np.linalg.svd(s[::7], full_matrices=False)
    along = s @ sv_[0]
    across = s @ sv_[1]
    pts = [slab[np.argmin(along)], slab[np.argmax(along)], slab[np.argmax(np.abs(across)) if abs(across).max() > 0 else 0]]
    frame = arch_frame.fit_occlusal_frame(list(map(float, pts[0])), list(map(float, pts[1])), list(map(float, pts[2])), centroid.tolist())
    
    # 1. Build T0 cast exactly as the export path does
    tv, tf, tinfo = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    
    import scipy.sparse.csgraph
    graph = cg.build_edge_graph(tv, tf, np.zeros(len(tv)))
    tooth_v = np.zeros(len(tv), dtype=bool)
    if len(labels) > 0: tooth_v = labels[:len(tv)] > 0
    tooth_idx = np.where(tooth_v)[0]
    protected_faces = np.zeros(len(tf), dtype=bool)
    if len(tooth_idx) > 0:
        dist = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=tooth_idx, min_only=True)
        protected_v = dist <= 3.0
        protected_faces = protected_v[tf].any(axis=1)
        
    tv2, tf2, rim, uinfo = cg.clear_undercut_periphery(tv, tf, frame, tinfo["rim_loop"], protected_mask=protected_faces)
    cv, cf, base_info = cg.build_cast_base(tv2, tf2, frame, rim=rim)
    
    print("Digest (Float64):", digest(cv, cf))
    
    # Float32-rounded
    blob = cg.write_binary_stl_bytes(cv, cf)
    rv, rf = stl_io.parse_stl_bytes(blob)
    print("Digest (Float32-rounded):", digest(rv, rf))
    
    n_scan = len(tf2)
    n_wall = len(rim) * 2
    
    print("\n--- Float64 self_intersection_report ---")
    rep64 = si.self_intersection_report(cv, cf)
    print("Total:", rep64["intersecting_pairs"])
    print("By kind:", rep64.get("by_kind", {}))
    
    print("\n--- Float32 self_intersection_report ---")
    rep32 = si.self_intersection_report(rv, rf)
    print("Total:", rep32["intersecting_pairs"])
    print("By kind:", rep32.get("by_kind", {}))
    
    # Wait, the user asked to manually classify the pairs using my logic as well?
    # I can just write my classification logic here.
    pairs32 = si.candidate_pairs(rv, rf)
    hit32, kind32 = si._pairs_intersect(rv, rf, pairs32, si.DEFAULT_TOUCH_TOL_MM)
    t32, c32 = classify(pairs32, hit32, kind32, n_scan, n_wall)
    print("Float32 Per-class counts:", c32)
    
    pairs64 = si.candidate_pairs(cv, cf)
    hit64, kind64 = si._pairs_intersect(cv, cf, pairs64, si.DEFAULT_TOUCH_TOL_MM)
    t64, c64 = classify(pairs64, hit64, kind64, n_scan, n_wall)
    print("Float64 Per-class counts:", c64)
    
    # Pairs in Float32 only
    set32 = set(tuple(p) for p in pairs32[hit32])
    set64 = set(tuple(p) for p in pairs64[hit64])
    only32 = set32 - set64
    print(f"\nPairs ONLY in Float32: {len(only32)}")
    
    from scipy.spatial import cKDTree
    for idx, p in enumerate(list(only32)[:5]):
        a, b = p
        va, vb = cv[cf[a]], cv[cf[b]]
        # Minimum float64 separation: min dist between any two vertices of the triangles
        d = cKDTree(va).query(vb)[0].min()
        print(f"Float32-only Pair {p}: kind={kind32[hit32][list(set32).index(p)]}, min float64 separation = {d:.6f} mm")

    print("\n--- Task 2: ALL scan-wall pairs (Float32) ---")
    # Histogram variables
    dist_hist = {"0-0.5": 0, "0.5-1": 0, "1-2": 0, "2-3": 0, ">3": 0}
    poke_hist = {"<0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, ">0.5": 0}
    scan_wall_details = []
    
    origin, e1, e2, _ = cg._arch_basis(frame)
    for p in pairs32[hit32]:
        a, b = p
        type_a = "scan" if a < n_scan else ("wall" if a < n_scan + n_wall else "floor")
        type_b = "scan" if b < n_scan else ("wall" if b < n_scan + n_wall else "floor")
        if (type_a == "scan" and type_b == "wall") or (type_b == "scan" and type_a == "wall"):
            scan_f = a if a < n_scan else b
            wall_f = b if a < n_scan else a
            
            scan_cent = rv[rf[scan_f]].mean(axis=0)
            d_to_base, closest = cKDTree(tv).query(scan_cent)
            d_to_tooth = np.min(dist[closest]) if len(tooth_idx)>0 else float('inf')
            
            if d_to_tooth <= 0.5: dist_hist["0-0.5"] += 1
            elif d_to_tooth <= 1.0: dist_hist["0.5-1"] += 1
            elif d_to_tooth <= 2.0: dist_hist["1-2"] += 1
            elif d_to_tooth <= 3.0: dist_hist["2-3"] += 1
            else: dist_hist[">3"] += 1
                
            w_v1, w_v2, w_v3 = rv[rf[wall_f]]
            w_norm = np.cross(w_v2 - w_v1, w_v3 - w_v1)
            ln = np.linalg.norm(w_norm)
            if ln > 0: w_norm /= ln
            poke = np.max(np.abs(np.dot(rv[rf[scan_f]] - w_v1, w_norm)))
            
            if poke < 0.1: poke_hist["<0.1"] += 1
            elif poke < 0.3: poke_hist["0.1-0.3"] += 1
            elif poke < 0.5: poke_hist["0.3-0.5"] += 1
            else: poke_hist[">0.5"] += 1
            
    print(f"Dist Histogram: {dist_hist}")
    print(f"Poke Histogram: {poke_hist}")
    
    print("\n--- Task 4: Projected rim simple? ---")
    rim_pts = cv[rim]
    rim_pts2d = np.column_stack([(rim_pts - origin) @ e1, (rim_pts - origin) @ e2])
    is_simple = True
    for i in range(len(rim_pts2d)):
        for j in range(i+2, len(rim_pts2d)):
            if i == 0 and j == len(rim_pts2d) - 1: continue
            def ccw(A,B,C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            def intersect(A,B,C,D): return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
            if intersect(rim_pts2d[i], rim_pts2d[(i+1)%len(rim_pts2d)], rim_pts2d[j], rim_pts2d[(j+1)%len(rim_pts2d)]):
                is_simple = False
                break
        if not is_simple: break
    print(f"Projected rim is simple polygon: {is_simple}")
    print("Why wall-wall is 0 if projected rim is not simple: If the 2D rim self-crosses, the wall triangles cross vertically. But _pairs_intersect ignores wall-wall intersections if they just fold on each other, or if they don't meet the 3D volume conditions... we will see.")
    
if __name__ == "__main__":
    main()
