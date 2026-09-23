import re
with open('core_geometry.py', 'r') as f:
    text = f.read()

new_func = """def clear_undercut_periphery(verts: np.ndarray, faces: np.ndarray, arch_frame: dict,
                             rim: np.ndarray, protected_mask: np.ndarray | None = None,
                             max_iters: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    import self_intersection as si
    from scipy.spatial import cKDTree
    
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    rim = np.asarray(rim, int)
    
    curve, _ = fit_arch_curve(verts, arch_frame)
    samples = _polyline_samples(curve)
    
    origin, e1, e2, u_occ = _arch_basis(arch_frame)
    rel = verts[faces].mean(axis=1) - origin
    centroids2d = np.column_stack([rel @ e1, rel @ e2])
    dist_to_curve, idx_to_curve = cKDTree(samples).query(centroids2d)
    
    margin_arr = np.full(len(samples), np.max(dist_to_curve) + 0.1)
    
    min_margin = np.zeros(len(samples))
    if protected_mask is not None:
        protected_dists = dist_to_curve[protected_mask]
        protected_idx = idx_to_curve[protected_mask]
        for d, idx in zip(protected_dists, protected_idx):
            min_margin[idx] = max(min_margin[idx], d + 0.1)
            
        smoothed_min = np.copy(min_margin)
        window = 5
        for i in range(len(min_margin)):
            start = max(0, i - window)
            end = min(len(min_margin), i + window + 1)
            smoothed_min[i] = np.max(min_margin[start:end])
        min_margin = smoothed_min
    
    out_v, out_f, out_rim = verts, faces, rim
    total_undercuts_fixed = 0
    
    for iteration in range(max_iters):
        heights = (out_v - origin) @ u_occ
        z_floor_height = np.min(heights) - CAST_BASE_THICKNESS_MM
        
        n_rim = len(out_rim)
        floor_rim_v = np.copy(out_v[out_rim])
        floor_rim_v = floor_rim_v - np.outer(((floor_rim_v - origin) @ u_occ) - z_floor_height, u_occ)
        
        temp_v = np.vstack([out_v, floor_rim_v])
        
        wall_faces = []
        for i in range(n_rim):
            j = (i + 1) % n_rim
            v_a = out_rim[i]
            v_b = out_rim[j]
            v_a_floor = len(out_v) + i
            v_b_floor = len(out_v) + j
            wall_faces.append([v_a, v_b, v_b_floor])
            wall_faces.append([v_a, v_b_floor, v_a_floor])
            
        wall_faces = np.array(wall_faces, dtype=int)
        temp_f = np.vstack([out_f, wall_faces])
        
        pairs = si.candidate_pairs(temp_v, temp_f)
        if not len(pairs):
            break
            
        hit, _ = si._pairs_intersect(temp_v, temp_f, pairs, 1e-6)
        intersecting = pairs[hit]
        
        n_scan = len(out_f)
        bad_faces = set()
        
        for a, b in intersecting:
            ta = "scan" if a < n_scan else "wall"
            tb = "scan" if b < n_scan else "wall"
            if ta == "scan" and tb != "scan": bad_faces.add(a)
            if tb == "scan" and ta != "scan": bad_faces.add(b)
            
        if not bad_faces:
            break
            
        bad_faces_arr = np.array(list(bad_faces))
        bad_centroids = out_v[out_f[bad_faces_arr]].mean(axis=1)
        rel_bad = bad_centroids - origin
        bad_2d = np.column_stack([rel_bad @ e1, rel_bad @ e2])
        
        bad_dist, bad_idx = cKDTree(samples).query(bad_2d)
        
        prev_margin = np.copy(margin_arr)
        
        for d, idx in zip(bad_dist, bad_idx):
            margin_arr[idx] = min(margin_arr[idx], d - 0.05)
            
        if protected_mask is not None:
            margin_arr = np.maximum(margin_arr, min_margin)
            
        smoothed = np.copy(margin_arr)
        window = 5
        for i in range(len(margin_arr)):
            start = max(0, i - window)
            end = min(len(margin_arr), i + window + 1)
            smoothed[i] = np.mean(margin_arr[start:end])
            
        if protected_mask is not None:
            smoothed = np.maximum(smoothed, min_margin)
            
        margin_arr = smoothed
        
        if np.allclose(margin_arr, prev_margin):
            # No changes can be made (likely hit protected band limit)
            break
        
        try:
            out_v, out_f, tinfo = trim_to_arch(verts, faces, arch_frame, margin_mm=margin_arr, curve=curve)
            out_rim = tinfo["rim_loop"]
        except ValueError as e:
            raise ValueError(f"clear_undercut_periphery: trim_to_arch failed after lowering margin: {e}")
            
        total_undercuts_fixed += len(bad_faces)
        
    else:
        # Just break, don't raise
        pass
        
    info = {
        "iterations": iteration,
        "undercuts_fixed": total_undercuts_fixed
    }
    return out_v, out_f, out_rim, info
"""

import re
if 'def clear_undercut_periphery' in text:
    text = re.sub(r'def clear_undercut_periphery.*', new_func, text, flags=re.DOTALL)
else:
    text += '\n\n' + new_func

with open('core_geometry.py', 'w') as f:
    f.write(text)
