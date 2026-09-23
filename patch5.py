import re

with open('core_geometry.py', 'r') as f:
    orig = f.read()

pattern = r'fmask = np\.ones\(len\(out_f\), dtype=bool\)\s*fmask\[list\(bad_faces\)\] = False\s*fmask = largest_face_component\(out_f, fmask\)\s*total_pruned_scan_faces \+= \(\~fmask\)\.sum\(\)\s*out_f = out_f\[fmask\]\s*verts, out_f, _, out_rim = _fill_interior_holes\(verts, out_f\)'

repl = """fmask = np.ones(len(out_f), dtype=bool)
        fmask[list(bad_faces)] = False
        fmask = largest_face_component(out_f, fmask)
        
        for _ in range(4):
            loops = boundary_loops(out_f[fmask])
            if not loops: break
            rim_loop = max(loops, key=len)
            sub_keep, opened = _open_pinch_vertices(out_f[fmask], rim_loop)
            if not opened: break
            drop = np.where(fmask)[0][~sub_keep]
            fmask[drop] = False
            fmask = largest_face_component(out_f, fmask)
            
        total_pruned_scan_faces += (~fmask).sum()
        out_f = out_f[fmask]
        verts, out_f, _, out_rim = _fill_interior_holes(verts, out_f)"""

new_orig = re.sub(pattern, repl, orig)

with open('core_geometry.py', 'w') as f:
    f.write(new_orig)
