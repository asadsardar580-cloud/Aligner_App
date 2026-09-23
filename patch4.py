import re
with open('core_geometry.py', 'r') as f: orig = f.read()

# Remove the 2D check block
pattern = r'# check 2D out of bounds.*?if not bad_faces:'
repl = 'if not bad_faces:'
orig = re.sub(pattern, repl, orig, flags=re.DOTALL)

with open('core_geometry.py', 'w') as f: f.write(orig)
