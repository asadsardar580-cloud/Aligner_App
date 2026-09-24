"""The cheek / tongue panels of scratch/labels_check.png, the right way round.

TASK 2 step 7. `tools/render_labels_check.py` flipped its sagittal axis when
the -u_sag end of the arch was the WIDER one - which is exactly when +u_sag
already pointed anteriorly - so +u_sag came out POSTERIOR, the "cheek side"
camera looked from inside the arch, and the two titles were swapped.

The titles rest on two facts, and each is pinned here on geometry whose
answer is known independently:
  1. `arch_axes` makes +u_sag point ANTERIORLY - on the synthetic horseshoe
     the incisors are at +y, and on its mirror image at -y;
  2. `render` puts the camera on the +eye side - whatever is furthest along
     +eye is what is drawn on top.
With both, `label_views`' eye along +u_sag looks at the arch from outside:
the cheek and lip side.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from test_cast_base import horseshoe_shell

_HERE = os.path.dirname(os.path.abspath(__file__))


def _tool():
    spec = importlib.util.spec_from_file_location(
        "render_labels_check",
        os.path.join(_HERE, "tools", "render_labels_check.py"))
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)      # the tool chdirs to the repo root
    finally:
        os.chdir(cwd)
    return mod


U_OCC = np.array([0.0, 0.0, 1.0])


def test_anterior_is_the_narrow_end_on_the_horseshoe_and_its_mirror():
    rl = _tool()
    v, _ = horseshoe_shell(n_s=240, n_t=60, teeth=(-0.25, 0.0, 0.25))
    _, u_sag, w_post, w_ant = rl.arch_axes(v, U_OCC)
    assert u_sag[1] > 0.99, u_sag                  # incisors at +y
    assert w_post > w_ant
    m = v * np.array([1.0, -1.0, 1.0])             # the arch mirrored in y
    _, u_sag_m, _, _ = rl.arch_axes(m, U_OCC)
    assert u_sag_m[1] < -0.99, u_sag_m
    print(f"PASS  +u_sag = {np.round(u_sag, 3)} (anterior +y); mirrored "
          f"{np.round(u_sag_m, 3)}; widths {w_post:.1f} vs {w_ant:.1f} mm")


def test_the_old_rule_pointed_posterior():
    """The control: the inverted comparison, on the same arch, points the
    other way. If this ever stops holding, the fixture no longer
    discriminates and the test above proves nothing."""
    v, _ = horseshoe_shell(n_s=240, n_t=60, teeth=(-0.25, 0.0, 0.25))
    d = v - v.mean(axis=0)
    inplane = d - np.outer(d @ U_OCC, U_OCC)
    _, _, vt = np.linalg.svd(inplane[::37], full_matrices=False)
    u_tra, u_sag = vt[0], vt[1]
    s = d @ u_sag
    w_lo = np.ptp(d[s < np.percentile(s, 25)] @ u_tra)
    w_hi = np.ptp(d[s > np.percentile(s, 75)] @ u_tra)
    old = -u_sag if w_lo > w_hi else u_sag
    assert old[1] < -0.99, old
    print(f"PASS  the old rule gives {np.round(old, 3)}: posterior")


def test_render_puts_the_camera_on_the_plus_eye_side():
    """Two overlapping squares, red at z=+1 and blue at z=-1, seen along
    eye=+z: the red one must cover the centre pixel."""
    rl = _tool()
    sq = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    v = np.vstack([sq + [0, 0, 1.0], sq + [0, 0, -1.0]])
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    red, blue = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    col = np.array([red, red, blue, blue])
    img = rl.render(v, f, col, np.array([0, 0, 1.0]), np.array([0, 1.0, 0]),
                    w=41, h=41)
    px = img[20, 20]
    assert px[0] > 0 and px[2] == 0, px
    img = rl.render(v, f, col, np.array([0, 0, -1.0]), np.array([0, 1.0, 0]),
                    w=41, h=41)
    assert img[20, 20][2] > 0 and img[20, 20][0] == 0
    print("PASS  eye +z shows the +z square, eye -z the -z square")


def test_the_cheek_panel_looks_from_outside_the_arch():
    rl = _tool()
    v, _ = horseshoe_shell(n_s=240, n_t=60, teeth=(-0.25, 0.0, 0.25))
    _, u_sag, _, _ = rl.arch_axes(v, U_OCC)
    (t0, eye0), (t1, eye1) = rl.label_views(u_sag, U_OCC)
    assert t0.startswith("cheek") and eye0[1] > 0      # from the front (+y)
    assert t1.startswith("tongue") and eye1[1] < 0     # from inside the U
    print(f"PASS  '{t0}' eye {np.round(eye0, 2)}; '{t1}' eye {np.round(eye1, 2)}")


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
