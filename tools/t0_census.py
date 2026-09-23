import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core_geometry as cg
import stl_io
import manufacturing as mfg
import self_intersection
import segmentation_providers
import arch_frame

SCAN = "case_lower.stl"
LABELS = "case_lower.stl_output.json"
REPORT = "reports/t0_census.json"

def main():
    os.makedirs("reports", exist_ok=True)
    
    # 1.1 Load case_lower.stl
    print("Loading scan...")
    with open(SCAN, "rb") as f:
        raw = f.read()
        
    verts, faces = stl_io.parse_stl_bytes(raw)
    verts, faces, sanity = cg.sanitize_scan(verts, faces)
    verts, faces, report = cg.condition_mesh(verts, faces)
    
    # Fit occlusal frame exactly as real_scan_regression does
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
        
    print("Building T0 cast...")
    t0_build = time.perf_counter()
    tv, tf, tinfo = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)

    graph = cg.build_edge_graph(tv, tf, np.zeros(len(tv)))
    import scipy.sparse.csgraph
    tooth_v = np.zeros(len(tv), dtype=bool)
    with open(LABELS, 'r') as lf: case = json.load(lf)
    if 'labels' in case:
        labels = case['labels']
        tooth_v = np.array(labels[:len(tv)]) > 0
    tooth_idx = np.where(tooth_v)[0]
    protected_faces = np.zeros(len(tf), dtype=bool)
    if len(tooth_idx) > 0:
        dist = scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=tooth_idx, min_only=True)
        protected_v = dist <= 3.0
        protected_faces = protected_v[tf].any(axis=1)

    tv, tf, rim, uinfo = cg.clear_undercut_periphery(tv, tf, frame, tinfo["rim_loop"], protected_mask=protected_faces); cv, cf, cinfo = cg.build_cast_base(tv, tf, frame, base_thickness_mm=cg.CAST_BASE_THICKNESS_MM, rim=rim)

    t_build = time.perf_counter() - t0_build
    
    # record metrics
    m_rep = cg.manifold_report(cf)
    
    # float32 round trip
    blob = cg.write_binary_stl_bytes(cv, cf)
    val = mfg.validate_printable_stl(blob)
    
    # 1.2 Self intersection
    print("Self intersection check...")
    rv, rf = stl_io.parse_stl_bytes(blob)
    
    t0_si = time.perf_counter()
    si_rep = self_intersection.self_intersection_report(rv, rf)
    t_si = time.perf_counter() - t0_si
    
    print("Self intersection count:", si_rep["intersecting_pairs"])
    
    out = {
        "t0_cast": {
            "vertices": len(cv),
            "faces": len(cf),
            "open_edges": m_rep["open_edges"],
            "nonmanifold_edges": m_rep["nonmanifold_edges"],
            "components": val["connected_components"],
            "winding": "consistent" if m_rep.get("consistent_winding", True) else "inconsistent",
            "volume": val.get("volume_mm3", 0.0),
            "float32_val": val
        },
        "self_intersection": {
            "count": si_rep["intersecting_pairs"],
            "by_kind": si_rep["by_kind"],
            "examples": si_rep["examples"][:50]
        },
        "timings": {
            "build_cast": t_build,
            "self_intersection": t_si
        }
    }
    with open(REPORT, "w") as f:
        json.dump(out, f, indent=2)
        
    if si_rep["intersecting_pairs"] > 0:
        print("STOPPING as requested.")
        sys.exit(1)
        
    print("Continuing with 1.3 and 1.4...")
    # TODO: 1.3 and 1.4

if __name__ == "__main__":
    main()
