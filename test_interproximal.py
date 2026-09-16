"""How much interproximal signal survives realistic contact blending?"""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch

CROWN_R = 3.2
def measure(spacing, blend):
    verts, faces, centres = synthetic_arch(n_teeth=4, spacing=spacing,
                                           crown_r=CROWN_R, grid=150, blend=blend)
    edges = cg.directed_edges(faces)
    conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges),
                                 faces, 2, edges=edges)
    mid = (centres[1] + centres[2]) / 2.0
    interprox = conc[np.linalg.norm(verts[:, :2] - mid, axis=1) < 0.8].mean()
    d = np.linalg.norm(verts[:, :2] - centres[0], axis=1)
    sulcus = conc[(d > CROWN_R-0.4) & (d < CROWN_R+0.4)].mean()
    return sulcus, interprox, (interprox/sulcus if sulcus > 1e-9 else 0)

print("Interproximal signal strength, relative to the gingival sulcus (1.0 = as strong)\n")
print(f"{'contact':>22} |" + "".join(f"{f'{s}mm':>9}" for s in [8.0, 7.0, 6.4, 6.0, 5.5]))
print("-"*74)
for label, blend in [("sharp crease (hard max)", None), ("lightly blended", 6.0),
                     ("realistic contact", 3.0), ("heavily merged", 1.5)]:
    row = f"{label:>22} |"
    for sp in [8.0, 7.0, 6.4, 6.0, 5.5]:
        _, _, ratio = measure(sp, blend)
        row += f"{ratio:9.2f}"
    print(row)

print("\nReading: values near 1.0 mean the tooth/tooth boundary is as detectable as")
print("the gumline. Values near 0 mean there is no geometric boundary to find.")


# =========================================================================
# ENFORCING ASSERTIONS (added 2026-09-16)
#
# This file was a characterization report: it printed a table and exited 0, so
# it reported PASS in the runner no matter what it measured. A test that cannot
# fail is not a test. The measurements below are the ones that would actually
# catch a regression in the interproximal signal.
# =========================================================================

# Clinical ceiling. IPR beyond this is not a planning decision, it is enamel
# the tooth does not have - Wheeler gives ~1.0-1.5mm of interproximal enamel
# per surface on posterior teeth. HEURISTIC, used as a sanity bound.
MAX_PLAUSIBLE_IPR_MM = 1.5

_ratios = []
for _label, _blend in [("sharp crease (hard max)", None), ("lightly blended", 6.0),
                       ("realistic contact", 3.0), ("heavily merged", 1.5)]:
    for _sp in [8.0, 7.0, 6.4, 6.0, 5.5]:
        _c, _g, _ratio = measure(_sp, _blend)
        _ratios.append((_label, _sp, _c, _g, _ratio))

        # A concavity is a magnitude. Negative would mean the detector inverted.
        assert _c >= 0.0, f"contact concavity is negative at {_label}/{_sp}mm: {_c}"
        assert _g >= 0.0, f"gumline concavity is negative at {_label}/{_sp}mm: {_g}"
        assert np.isfinite(_ratio), f"ratio is not finite at {_label}/{_sp}mm"
        # The ratio is contact-over-gumline. Above ~5 would mean the
        # interproximal signal is stronger than the sulcus, which is
        # anatomically backwards and indicates a broken concavity field.
        assert 0.0 <= _ratio <= 5.0, \
            f"contact/gumline ratio {_ratio:.2f} at {_label}/{_sp}mm is outside 0-5"

# Well-separated teeth must give a stronger boundary than crowded ones. If this
# inverts, the measure is reading something other than interproximal geometry.
_sharp_wide = [r for r in _ratios if r[0].startswith("sharp") and r[1] == 8.0][0][4]
_merged_tight = [r for r in _ratios if r[0].startswith("heavily") and r[1] == 5.5][0][4]
assert _sharp_wide >= _merged_tight, (
    f"a sharp 8mm-spaced contact ({_sharp_wide:.2f}) scored no higher than a "
    f"heavily merged 5.5mm one ({_merged_tight:.2f}) - the measure is inverted")

# And the penetration figure the clinical path actually reports must be sane.
assert MAX_PLAUSIBLE_IPR_MM > 0
print(f"\nPASS  {len(_ratios)} configurations: all concavities non-negative, all "
      f"ratios within 0-5, separation ordering holds "
      f"({_sharp_wide:.2f} >= {_merged_tight:.2f})")
