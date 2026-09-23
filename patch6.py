import re

with open('core_geometry.py', 'r') as f:
    orig = f.read()

with open('temp_bcb_new.py', 'r') as f:
    new_func = f.read()

match = re.search(r'^def build_cast_base\(', orig, re.MULTILINE)
start_idx = match.start()
match2 = re.search(r'^def ', orig[start_idx+10:], re.MULTILINE)
end_idx = start_idx + 10 + match2.start() if match2 else len(orig)

new_orig = orig[:start_idx] + new_func + '\n' + orig[end_idx:]

# Now apply patch5 logic
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

new_orig = re.sub(pattern, repl, new_orig)

# Wait, the 2D check was also still in temp_bcb_new.py?
# Let's remove the 2D check again.
pattern2 = r'# check 2D out of bounds.*?if not bad_faces:'
repl2 = 'if not bad_faces:'
new_orig = re.sub(pattern2, repl2, new_orig, flags=re.DOTALL)

with open('core_geometry.py', 'w') as f:
    f.write(new_orig)
