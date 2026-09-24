import sys
with open("core_geometry.py", "r") as f: code = f.read()
old = '''    elif isinstance(margin_mm, dict):
        is_lingual = np.linalg.norm(centroids2d - cinfo["centre"], axis=1) < np.linalg.norm(samples[idx] - cinfo["centre"], axis=1)'''
new = '''    elif isinstance(margin_mm, dict):
        center_pt = cinfo.get("centre", np.mean(samples, axis=0))
        is_lingual = np.linalg.norm(centroids2d - center_pt, axis=1) < np.linalg.norm(samples[idx] - center_pt, axis=1)'''
code = code.replace(old, new)
with open("core_geometry.py", "w") as f: f.write(code)
print("Patched.")
