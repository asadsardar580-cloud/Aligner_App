import sys
import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np
import core_geometry as cg
import stl_io
import self_intersection as si
import arch_frame

def main():
    with open("case_lower.stl", "rb") as f: raw = f.read()
    verts, faces = stl_io.parse_stl_bytes(raw)
    verts, faces, _ = cg.sanitize_scan(verts, faces)
    verts, faces, _ = cg.condition_mesh(verts, faces)
    
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
        
    tv_base, tf_base, tinfo_base = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    cv_base, cf_base, base_info_base = cg.build_cast_base(tv_base, tf_base, frame, rim=tinfo_base["rim_loop"]) 
    
    blob_base = cg.write_binary_stl_bytes(cv_base, cf_base)
    rv_base, rf_base = stl_io.parse_stl_bytes(blob_base)
    
    print("Running candidate_pairs...")
    pairs_base = si.candidate_pairs(rv_base, rf_base)
    print(f"Candidates: {len(pairs_base)}")
    hit_base, _ = si._pairs_intersect(rv_base, rf_base, pairs_base, 0.001)
    intersecting_base = pairs_base[hit_base]
    print(f"Total T0 pairs with 0.001: {len(intersecting_base)}")

if __name__ == "__main__":
    main()
