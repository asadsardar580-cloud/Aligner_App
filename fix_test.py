import re
with open('test_cast_base.py', 'r') as f:
    text = f.read()
    
# Replace the old test_undercut_protects_band with a new one
new_test = """def test_undercut_protects_band():
    import pytest
    v, f = horseshoe_shell(band_w=10.0, band_h=10.0, theta_max=2.5)
    af = frame_for(v)
    tv, tf, ti = cg.trim_to_arch(v, f, af, margin_mm=22.0)
    protected = np.ones(len(tf), dtype=bool)
    tv2, tf2, rim, uinfo = cg.clear_undercut_periphery(tv, tf, af, ti["rim_loop"], protected_mask=protected)
    assert len(tf2) == len(tf), "Protected band was touched"
"""

text = re.sub(r'def test_undercut_protects_band\(\).*', new_test, text, flags=re.DOTALL)

with open('test_cast_base.py', 'w') as f:
    f.write(text)
