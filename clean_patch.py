import re
with open('patch_undercut.py', 'r') as f:
    text = f.read()

text = re.sub(r'# First, trim it with the current margin so out_f aligns with margin_arr logic\s+out_v, out_f, tinfo = trim_to_arch\(verts, faces, arch_frame, margin_mm=margin_arr, curve=curve\)\s+out_rim = tinfo\["rim_loop"\]', '', text)

with open('patch_undercut.py', 'w') as f:
    f.write(text)
