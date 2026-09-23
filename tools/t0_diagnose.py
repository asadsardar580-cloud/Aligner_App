import sys
import os
import json
import time
import numpy as np
import scipy.sparse.csgraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core_geometry as cg
import stl_io
import manufacturing as mfg
import self_intersection
import arch_frame

SCAN = "case_lower.stl"
LABELS = "case_lower.stl_output.json"
REPORT = "reports/t0_intersections.json"

def main():
    os.makedirs("reports", exist_ok=True)
    
    with open(SCAN, "rb") as f:
        raw = f.read()
        
    verts, faces = stl_io.parse_stl_bytes(raw)
    verts, faces, sanity = cg.sanitize_scan(verts, faces)
    verts, faces, report = cg.condition_mesh(verts, faces)
    n_scan = len(faces)
    
    # Load labels
    with open(LABELS, "r") as f:
        lbl_data = json.load(f)
    v_labels = np.array(lbl_data["labels"])
    if len(v_labels) != len(verts):
        raise ValueError("Labels length mismatch")
    
    # Identify tooth vertices (label > 0 and label != 0)
    # Note: usually tooth labels are > 0.
    is_tooth_v = (v_labels > 0)
    
    # Frame
    centroid = verts.mean(axis=0)
    d = verts - centroid
    _, _, vt = np.linalg.svd(d[::37], full_matrices=False)
    occ_axis = vt[2]
    slab = verts[(d @ occ_axis) > np.percentile(d @ occ_axis, 90)]
    if (slab - centroid).shape[0] < 10:
        slab = verts
    s = slab - slab.mean(axis=0)
    _, _, sv_ = np.linalg.svd(s[::7], full_matrices=False)
    along = s @ sv_[0]
    across = s @ sv_[1]
    pts = [slab[np.argmin(along)], slab[np.argmax(along)],
           slab[np.argmax(np.abs(across)) if abs(across).max() > 0 else 0]]
    frame = arch_frame.fit_occlusal_frame(
        list(map(float, pts[0])), list(map(float, pts[1])),
        list(map(float, pts[2])), centroid.tolist())
        
    tv, tf, tinfo = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    
    # Check if the rim loop projected onto the floor plane is simple
    loops = cg.boundary_loops(tf)
    if len(loops) == 1:
        rim = loops[0]
        origin, e1, e2, u_occ = cg._arch_basis(frame)
        rel = tv - origin
        Q = np.column_stack([rel[rim] @ e1, rel[rim] @ e2])
        crossings = cg._crossing_pairs(Q)
        print(f"Rim loop 2D crossings: {len(crossings)}")
    else:
        print(f"Rim loop count != 1: {len(loops)}")
        
    cv, cf, cinfo = cg.build_cast_base(tv, tf, frame, base_thickness_mm=cg.CAST_BASE_THICKNESS_MM)
    
    # Face boundaries
    # The returned cf has: scan faces, wall faces, floor faces.
    # scan faces: 0 to len(tf)-1
    # wall faces: len(tf) to len(tf) + cinfo["wall_tris"] - 1
    # floor faces: len(tf) + cinfo["wall_tris"] to end
    n_scan_cf = len(tf)
    n_wall_cf = cinfo["wall_tris"]
    n_floor_cf = cinfo["floor_tris"]
    
    # geodesic dist on original verts
    graph = cg.build_edge_graph(verts, faces, np.zeros(len(verts)))
    tooth_idx = np.where(is_tooth_v)[0]
    if len(tooth_idx) > 0:
        dist_to_tooth = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=tooth_idx, min_only=True)
    else:
        dist_to_tooth = np.full(len(verts), 9999.0)
        
    # Classify each scan face in cf (which corresponds to tf)
    # We need to map tf back to original faces or just classify tf directly.
    # Wait, tv has the same prefix as verts. So tf's vertices refer to tv, where tv[:len(verts)] == verts.
    # A face is 'tooth' if any vertex is in is_tooth_v
    # A face is 'tooth-tooth contact' if vertices belong to MULTIPLE different tooth labels > 0.
    
    blob = cg.write_binary_stl_bytes(cv, cf)
    rv, rf = stl_io.parse_stl_bytes(blob)
    
    pairs = self_intersection.candidate_pairs(rv, rf)
    if len(pairs) > 0:
        chunk = 400000
        ov_list = []
        for i in range(0, len(pairs), chunk):
            p = pairs[i:i+chunk]
            hit, _ = self_intersection._pairs_intersect(rv, rf, p, 1e-6)
            ov_list.append(hit)
        ov = np.concatenate(ov_list)
        intersecting = pairs[ov]
    else:
        intersecting = np.zeros((0, 2), dtype=int)
        
    print(f"Total intersecting pairs: {len(intersecting)}")
    
    counts = {
        "scan-scan": 0, "scan-wall": 0, "scan-floor": 0,
        "wall-wall": 0, "wall-floor": 0, "floor-floor": 0
    }
    
    for a, b in intersecting:
        def get_type(idx):
            if idx < n_scan_cf: return "scan"
            if idx < n_scan_cf + n_wall_cf: return "wall"
            return "floor"
        
        ta, tb = get_type(a), get_type(b)
        
        def refine_scan_type(idx):
            # idx is a face index in cf
            face_verts = cf[idx]
            # face_verts can point to added hole-fill vertices > len(v_labels)-1
            face_labels = [v_labels[v] if v < len(v_labels) else 0 for v in face_verts]
            tooth_labels = set(l for l in face_labels if l > 0)
            if len(tooth_labels) > 1:
                return "scan(tooth-tooth contact)"
            if len(tooth_labels) == 1:
                return "scan(tooth)"
            
            # check distance
            # face_verts distance to tooth. We take min distance of its 3 vertices
            min_dist = min([dist_to_tooth[v] if v < len(dist_to_tooth) else 9999.0 for v in face_verts])
            if min_dist <= 3.0:
                return "scan(gingiva inside band)"
            return "scan(periphery)"
            
        if ta == "scan": ta = refine_scan_type(a)
        if tb == "scan": tb = refine_scan_type(b)
        
        key = f"{ta} / {tb}" if ta <= tb else f"{tb} / {ta}"
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
        
    print(counts)
    with open(REPORT, "w") as f:
        json.dump(counts, f, indent=2)

if __name__ == "__main__":
    main()
