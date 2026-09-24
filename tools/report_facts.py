import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core_geometry as cg
import stl_io
import self_intersection as si
import arch_frame
import scipy.sparse.csgraph

def main():
    print("Loading scan and labels...")
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
        
    print("Building original T0 cast...")
    tv_base, tf_base, tinfo_base = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    cv_base, cf_base, base_info_base = cg.build_cast_base(tv_base, tf_base, frame, rim=None) # baseline without undercut fix uses rim=None
    
    blob_base = cg.write_binary_stl_bytes(cv_base, cf_base)
    rv_base, rf_base = stl_io.parse_stl_bytes(blob_base)
    
    print("--- 1B.1 Numbers ---")
    pairs_base = si.candidate_pairs(rv_base, rf_base)
    hit_base, _ = si._pairs_intersect(rv_base, rf_base, pairs_base, 1e-6)
    intersecting_base = pairs_base[hit_base]
    
    # We must accurately identify n_scan. For float32 roundtrip, the faces are reordered?
    # write_binary_stl_bytes preserves face order!
    n_scan_base = len(tf_base)
    n_wall_base = len(base_info_base["wall_faces"]) if "wall_faces" in base_info_base else 0
    # wait, build_cast_base doesn't return wall_faces. We know the wall is the next faces after scan.
    # length of rim:
    rim_base = base_info_base.get("rim", tinfo_base["rim_loop"])
    n_wall = len(rim_base) * 2
    
    counts = {"scan-wall": 0, "scan-floor": 0, "wall-wall": 0, "wall-floor": 0, "floor-floor": 0, "scan-scan": 0}
    
    for a, b in intersecting_base:
        type_a = "scan" if a < n_scan_base else ("wall" if a < n_scan_base + n_wall else "floor")
        type_b = "scan" if b < n_scan_base else ("wall" if b < n_scan_base + n_wall else "floor")
        key = f"{type_a}-{type_b}"
        if key not in counts: key = f"{type_b}-{type_a}"
        counts[key] = counts.get(key, 0) + 1
        
    print(f"Total T0 pairs: {len(intersecting_base)}")
    for k, v in counts.items(): print(f"{k}: {v}")
        
    origin, e1, e2, _ = cg._arch_basis(frame)
    rim_pts = cv_base[tinfo_base["rim_loop"]]
    rim_pts2d = np.column_stack([(rim_pts - origin) @ e1, (rim_pts - origin) @ e2])
    
    is_simple = True
    for i in range(len(rim_pts2d)):
        for j in range(i+2, len(rim_pts2d)):
            if i == 0 and j == len(rim_pts2d) - 1: continue
            a, b = rim_pts2d[i], rim_pts2d[(i+1)%len(rim_pts2d)]
            c, d = rim_pts2d[j], rim_pts2d[(j+1)%len(rim_pts2d)]
            def ccw(A,B,C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            def intersect(A,B,C,D): return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
            if intersect(a,b,c,d): is_simple = False; break
        if not is_simple: break
    print(f"Projected rim is simple polygon: {is_simple}")
    
    print("\n--- 1B.5 execution with undercut fix ---")
    graph = cg.build_edge_graph(tv_base, tf_base, np.zeros(len(tv_base)))
    tooth_v = np.zeros(len(tv_base), dtype=bool)
    if len(labels) > 0: tooth_v = labels[:len(tv_base)] > 0
    tooth_idx = np.where(tooth_v)[0]
    protected_faces = np.zeros(len(tf_base), dtype=bool)
    if len(tooth_idx) > 0:
        dist = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=tooth_idx, min_only=True)
        protected_v = dist <= 3.0
        protected_faces = protected_v[tf_base].any(axis=1)
        
    tv, tf, rim, uinfo = cg.clear_undercut_periphery(tv_base, tf_base, frame, tinfo_base["rim_loop"], protected_mask=protected_faces)
    cv, cf, cinfo = cg.build_cast_base(tv, tf, frame, rim=rim)
    
    blob = cg.write_binary_stl_bytes(cv, cf)
    rv, rf = stl_io.parse_stl_bytes(blob)
    
    pairs = si.candidate_pairs(rv, rf)
    hit, kind = si._pairs_intersect(rv, rf, pairs, 1e-6)
    intersecting = pairs[hit]
    
    n_scan = len(tf)
    wall_start = n_scan
    wall_end = n_scan + len(rim) * 2
    
    scan_wall_pairs = []
    for a, b in intersecting:
        type_a = "scan" if a < n_scan else ("wall" if a < wall_end else "floor")
        type_b = "scan" if b < n_scan else ("wall" if b < wall_end else "floor")
        if (type_a == "scan" and type_b == "wall") or (type_b == "scan" and type_a == "wall"):
            scan_wall_pairs.append((a, b))
            
    print(f"TOTAL remaining T0 pairs across all classes: {len(intersecting)}")
    print(f"Scan-wall pairs remaining: {len(scan_wall_pairs)}")
    
    print("\n--- Deleted Area & Geodesic Distance ---")
    def face_area(v, f):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
        
    area_before = face_area(tv_base, tf_base)
    area_after = face_area(tv, tf)
    deleted_area = area_before - area_after
    print(f"Deleted area: {deleted_area:.2f} mm2")
    
    cent_base = tv_base[tf_base].mean(axis=1)
    cent_after = tv[tf].mean(axis=1)
    from scipy.spatial import cKDTree
    tree = cKDTree(cent_after)
    dist_to_after, _ = tree.query(cent_base)
    deleted_mask = dist_to_after > 1e-4
    deleted_face_indices = np.where(deleted_mask)[0]
    
    min_dist_to_tooth = float('inf')
    if len(deleted_face_indices) > 0:
        deleted_v_idx = np.unique(tf_base[deleted_face_indices])
        if len(tooth_idx) > 0:
            dists = dist[deleted_v_idx]
            min_dist_to_tooth = np.min(dists)
    print(f"Minimum geodesic distance from deleted area to any tooth region: {min_dist_to_tooth:.2f} mm")
    
    print("\n--- Detailed Scan-Wall pair facts ---")
    os.makedirs("scratch/d1_crops", exist_ok=True)
    
    for idx, (a, b) in enumerate(scan_wall_pairs[:8]):
        scan_f = a if a < n_scan else b
        wall_f = b if a < n_scan else a
        
        scan_centroids = rv[rf[scan_f]].mean(axis=0)
        dist_to_base_v, closest_base_v = cKDTree(tv_base).query(scan_centroids)
        d_to_tooth = np.min(dist[closest_base_v])
        
        unique_teeth = np.unique(labels[labels > 0])
        fdi_dists = {}
        for fdi in unique_teeth:
            v_in_tooth = np.where(labels[:len(tv_base)] == fdi)[0]
            if len(v_in_tooth) > 0:
                d_to_this_tooth = np.min(scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=v_in_tooth, min_only=True)[closest_base_v])
                fdi_dists[fdi] = d_to_this_tooth
        
        nearest_fdi = min(fdi_dists, key=fdi_dists.get) if fdi_dists else "None"
        is_tooth = np.any(labels[closest_base_v] > 0)
        
        rel = scan_centroids - origin
        y_rel = rel @ e2
        x_rel = rel @ e1
        if y_rel > 10: side = "lingual"
        elif y_rel < -10: side = "buccal"
        else: side = "distal"
        
        # Max distance of scan face vertices past the wall
        # We can compute the distance from the scan vertices to the wall plane
        # wall_f is the wall face.
        w_v1, w_v2, w_v3 = rv[rf[wall_f]]
        w_normal = np.cross(w_v2 - w_v1, w_v3 - w_v1)
        w_normal = w_normal / np.linalg.norm(w_normal)
        scan_verts = rv[rf[scan_f]]
        dists_to_wall = np.dot(scan_verts - w_v1, w_normal)
        # the outward normal of the wall points outward. So positive distance is past the wall.
        # But let's just take absolute value of max since orientation might be tricky
        poke_mm = np.max(np.abs(dists_to_wall))
        
        print(f"Pair {idx+1}:")
        print(f"  Nearest tooth: {nearest_fdi}")
        print(f"  Side: {side}")
        print(f"  Geodesic dist to tooth region: {d_to_tooth:.2f} mm")
        print(f"  Crossing face is: {'tooth' if is_tooth else 'gingiva'}")
        print(f"  Pokes past wall: ~{poke_mm:.2f} mm")
        
        dist_to_cent = np.linalg.norm(rv[rf].mean(axis=1) - scan_centroids, axis=1)
        crop_f_idx = np.where(dist_to_cent < 10.0)[0]
        crop_f = rf[crop_f_idx]
        blob_crop = cg.write_binary_stl_bytes(rv, crop_f)
        with open(f"scratch/d1_crops/pair_{idx+1}.stl", "wb") as f_out:
            f_out.write(blob_crop)
            
if __name__ == "__main__":
    main()
