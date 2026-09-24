import sys, os
import json
import numpy as np
import scipy.sparse.csgraph
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_geometry as cg
import stl_io
import self_intersection as si
import arch_frame

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
    
    tv, tf, tinfo = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    cv, cf, base_info = cg.build_cast_base(tv, tf, frame, rim=tinfo["rim_loop"])
    blob = cg.write_binary_stl_bytes(cv, cf)
    rv, rf = stl_io.parse_stl_bytes(blob)
    n_scan = len(tf)
    
    pairs32 = si.candidate_pairs(rv, rf)
    hit32, kind32 = si._pairs_intersect(rv, rf, pairs32, si.DEFAULT_TOUCH_TOL_MM)
    
    # FDI label map
    v_to_fdi = np.zeros(len(verts), dtype=int)
    if len(labels) > 0: v_to_fdi = labels[:len(verts)]
    
    origin, e1, e2, e3 = cg._arch_basis(frame)
    
    print("\n--- Task 3: 35 scan-scan pairs ---")
    scan_scan = [p for p in pairs32[hit32] if p[0] < n_scan and p[1] < n_scan]
    for idx, p in enumerate(scan_scan[:5]):  # print first 5 to check
        a, b = p
        cent = rv[rf[a]].mean(axis=0)
        # find nearest tooth
        dist, closest = cKDTree(verts).query(cent)
        fdi = v_to_fdi[closest]
        y_coord = (cent - origin) @ e2
        side = "lingual" if y_coord < 0 else "buccal"
        k = kind32[hit32][np.where((pairs32[hit32] == p).all(axis=1))[0][0]]
        print(f"Face {a} and {b}: nearest tooth {fdi}, side: {side}, kind: {k}, tooth/gingiva: {'tooth' if fdi > 0 else 'gingiva'}")

    print("\n--- Task 5: Labels at lingual of 46 and 47 ---")
    # Distance from lingual tooth-label boundary to scan boundary
    fdi_46 = 46
    fdi_47 = 47
    for fdi in [46, 47]:
        t_v_idx = np.where(v_to_fdi == fdi)[0]
        if len(t_v_idx) == 0: continue
        t_pts = verts[t_v_idx]
        # find lingual boundary
        y_coords = (t_pts - origin) @ e2
        lingual_t_pts = t_pts[y_coords < 0]
        if len(lingual_t_pts) > 0:
            ling_bound = np.min((lingual_t_pts - origin) @ e2)
            # Find scan boundary on the lingual side of this tooth
            x_coords = (lingual_t_pts - origin) @ e1
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            # Scan boundary vertices in this x range
            scan_bound_idx = tinfo["rim_loop"]
            sb_pts = tv[scan_bound_idx]
            sb_x = (sb_pts - origin) @ e1
            sb_y = (sb_pts - origin) @ e2
            mask = (sb_x >= min_x) & (sb_x <= max_x) & (sb_y < 0)
            if np.any(mask):
                sb_ling = np.min(sb_y[mask])
                dist_mm = np.abs(sb_ling - ling_bound)
                print(f"Tooth {fdi} lingual boundary to scan boundary: {dist_mm:.2f} mm")
    
    # Export crop
    crop_mask = np.zeros(len(verts), dtype=bool)
    for fdi in [46, 47]:
        t_v_idx = np.where(v_to_fdi == fdi)[0]
        if len(t_v_idx) > 0:
            crop_mask[t_v_idx] = True
    # Expand slightly
    for _ in range(5):
        crop_mask[faces[crop_mask[faces].any(axis=1)].ravel()] = True
        
    crop_faces = faces[crop_mask[faces].all(axis=1)]
    # Split by tooth vs gingiva
    is_tooth_f = (v_to_fdi[crop_faces] > 0).any(axis=1)
    tooth_f = crop_faces[is_tooth_f]
    gingiva_f = crop_faces[~is_tooth_f]
    
    os.makedirs("scratch/d1_crops", exist_ok=True)
    cg.write_binary_stl(verts, tooth_f, "scratch/d1_crops/46_47_tooth.stl")
    cg.write_binary_stl(verts, gingiva_f, "scratch/d1_crops/46_47_gingiva.stl")
    print("Exported scratch/d1_crops/46_47_tooth.stl and scratch/d1_crops/46_47_gingiva.stl")

if __name__ == "__main__":
    main()
