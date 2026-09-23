import re
with open('core_geometry.py', 'r') as f: orig = f.read()

pattern = r'orig_idx = face_hash\[tuple\(sorted\(out_f\[idx\]\)\)\]\s*if protected_faces\[orig_idx\]:\s*raise ValueError\(\s*f"Undercut fix cannot converge without entering the protected band at iteration \{iteration\}"\)'
repl = 'orig_idx = face_hash[tuple(sorted(out_f[idx]))]\n                  if protected_faces[orig_idx]:\n                      print(f"Protected face hit: {orig_idx}")\n                      raise ValueError("hit band")'
orig = re.sub(pattern, repl, orig)

with open('core_geometry.py', 'w') as f: f.write(orig)
