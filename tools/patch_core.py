import sys
import numpy as np

with open('core_geometry.py', 'r') as f:
    code = f.read()

old_trim = """    if is_scalar_margin:
        keep = dist < margin_mm
    else:
        if len(margin_mm) != len(samples):
            raise ValueError("margin_mm array must match number of curve samples")
        keep = dist < margin_mm[idx]"""

new_trim = """    if is_scalar_margin:
        keep = dist < margin_mm
    elif isinstance(margin_mm, dict):
        is_lingual = np.linalg.norm(centroids2d - cinfo["centre"], axis=1) < np.linalg.norm(samples[idx] - cinfo["centre"], axis=1)
        mb = margin_mm["buccal"][idx]
        ml = margin_mm["lingual"][idx]
        keep = dist < np.where(is_lingual, ml, mb)
    else:
        if len(margin_mm) != len(samples):
            raise ValueError("margin_mm array must match number of curve samples")
        keep = dist < margin_mm[idx]"""

code = code.replace(old_trim, new_trim)

old_clear = """    margin_arr = np.full(len(samples), np.max(dist_to_curve) + 0.1)
    
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
    total_undercuts_fixed = 0"""

new_clear = """    margin_buccal = np.full(len(samples), np.max(dist_to_curve) + 0.1)
    margin_lingual = np.full(len(samples), np.max(dist_to_curve) + 0.1)
    
    min_buccal = np.zeros(len(samples))
    min_lingual = np.zeros(len(samples))
    
    is_lingual_face = np.linalg.norm(centroids2d - curve_info["centre"], axis=1) < np.linalg.norm(samples[idx_to_curve] - curve_info["centre"], axis=1)
    
    if protected_mask is not None:
        prot_buccal = protected_mask & ~is_lingual_face
        prot_lingual = protected_mask & is_lingual_face
        
        for d, idx in zip(dist_to_curve[prot_buccal], idx_to_curve[prot_buccal]):
            min_buccal[idx] = max(min_buccal[idx], d + 0.1)
        for d, idx in zip(dist_to_curve[prot_lingual], idx_to_curve[prot_lingual]):
            min_lingual[idx] = max(min_lingual[idx], d + 0.1)
            
        def smooth_min(arr, win=5):
            res = np.copy(arr)
            for i in range(len(arr)):
                start = max(0, i - win)
                end = min(len(arr), i + win + 1)
                res[i] = np.max(arr[start:end])
            return res
            
        min_buccal = smooth_min(min_buccal)
        min_lingual = smooth_min(min_lingual)
    
    out_v, out_f, out_rim = verts, faces, rim
    total_undercuts_fixed = 0"""

code = code.replace(old_clear, new_clear)
code = code.replace("curve, _ = fit_arch_curve(verts, arch_frame)", "curve, curve_info = fit_arch_curve(verts, arch_frame)")

old_loop_tail = """        bad_dist, bad_idx = cKDTree(samples).query(bad_2d)
        
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
        except ValueError as e:"""

new_loop_tail = """        bad_dist, bad_idx = cKDTree(samples).query(bad_2d)
        bad_is_lingual = np.linalg.norm(bad_2d - curve_info["centre"], axis=1) < np.linalg.norm(samples[bad_idx] - curve_info["centre"], axis=1)
        
        prev_buccal = np.copy(margin_buccal)
        prev_lingual = np.copy(margin_lingual)
        
        for d, idx, is_ling in zip(bad_dist, bad_idx, bad_is_lingual):
            if is_ling:
                margin_lingual[idx] = min(margin_lingual[idx], d - 0.05)
            else:
                margin_buccal[idx] = min(margin_buccal[idx], d - 0.05)
                
        def smooth_margin(arr, min_arr, win=5):
            if protected_mask is not None:
                arr = np.maximum(arr, min_arr)
            res = np.copy(arr)
            for i in range(len(arr)):
                start = max(0, i - win)
                end = min(len(arr), i + win + 1)
                res[i] = np.mean(arr[start:end])
            if protected_mask is not None:
                res = np.maximum(res, min_arr)
            return res
            
        margin_buccal = smooth_margin(margin_buccal, min_buccal)
        margin_lingual = smooth_margin(margin_lingual, min_lingual)
        
        if np.allclose(margin_buccal, prev_buccal) and np.allclose(margin_lingual, prev_lingual):
            break
            
        margin_dict = {"buccal": margin_buccal, "lingual": margin_lingual}
        
        try:
            out_v, out_f, tinfo = trim_to_arch(verts, faces, arch_frame, margin_mm=margin_dict, curve=curve)
            out_rim = tinfo["rim_loop"]
        except ValueError as e:"""

code = code.replace(old_loop_tail, new_loop_tail)
with open('core_geometry.py', 'w') as f:
    f.write(code)
print("Patched core_geometry.py")
