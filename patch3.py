import re
with open('core_geometry.py', 'r') as f: orig = f.read()

pattern = r'fmask = largest_face_component\(out_f, fmask\)\s*fmask = _fill_interior_holes\(out_f, fmask\)\s*total_pruned_scan_faces \+= \(\~fmask\)\.sum\(\)\s*out_f = out_f\[fmask\]\s*out_rim = None'
repl = 'fmask = largest_face_component(out_f, fmask)\n        total_pruned_scan_faces += (~fmask).sum()\n        out_f = out_f[fmask]\n        verts, out_f, _, out_rim = _fill_interior_holes(verts, out_f)'
orig = re.sub(pattern, repl, orig)

with open('core_geometry.py', 'w') as f: f.write(orig)
