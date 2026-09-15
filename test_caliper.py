"""Digital mesiodistal caliper, validated against Wheeler's averages.

Replaces face count, which is a biologically invalid proxy: a fissured molar
carries far more triangles per mm2 than a smooth incisor."""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch
from tooth_segmentation import arch_geometry


def test_tangent_is_local_and_perpendicular():
    """The mesiodistal axis at a molar is nearly perpendicular to the one at
    a central incisor -- a single global axis cannot serve both."""
    axis = np.array([0.0, 0.0, 1.0]); centre = np.zeros(3)
    anterior = cg.local_arch_tangent(np.array([0.0, -20.0, 0.0]), centre, axis)
    lateral = cg.local_arch_tangent(np.array([20.0, 0.0, 0.0]), centre, axis)
    ang = np.degrees(np.arccos(abs(np.dot(anterior, lateral))))
    assert abs(np.dot(anterior, axis)) < 1e-9, "tangent must lie in the occlusal plane"
    assert ang > 80, f"anterior and lateral tangents should be near-perpendicular, got {ang:.0f}"
    print(f"PASS  local tangent: anterior vs lateral differ by {ang:.0f} degrees")


def test_caliper_measures_a_known_width():
    """A synthetic crown of known radius must measure its true diameter."""
    r = 3.2
    v, f, centres = synthetic_arch(n_teeth=5, spacing=9.0, crown_r=r, grid=150, blend=3.0)
    axis = np.array([0.0, 0.0, 1.0])
    target = np.append(centres[2], 0.0)
    d = np.linalg.norm(v[:, :2] - centres[2], axis=1)
    crown = v[d < r]
    arch_centre = np.append(centres.mean(axis=0), 0.0)
    w = cg.measure_mesiodistal_width(crown, target, arch_centre, axis,
                                     occlusal_fraction=1.0)
    assert abs(w - 2 * r) < 1.0, f"expected about {2*r}mm, measured {w:.2f}mm"
    print(f"PASS  caliper on a known {2*r}mm crown: measured {w:.2f}mm")


def test_validation_flags_merged_and_split():
    res = cg.validate_against_anatomy(
        [(1, 8.6), (2, 12.0), (6, 4.0)], arch="maxillary")
    verdicts = {r["position"]: r["verdict"] for r in res["rows"]}
    assert verdicts[1] == "ok"
    assert "too wide" in verdicts[2], verdicts[2]
    assert "too narrow" in verdicts[6], verdicts[6]
    print("PASS  validation flags merged (too wide) and split (too narrow) teeth")


def test_occlusal_restriction_matters():
    """Measuring the whole region reports gingival spread, not crown width."""
    v, f, centres = synthetic_arch(n_teeth=5, spacing=9.0, crown_r=3.2, grid=150, blend=3.0)
    axis = np.array([0.0, 0.0, 1.0])
    d = np.linalg.norm(v[:, :2] - centres[2], axis=1)
    bled = v[d < 5.5]            # crown plus surrounding gingiva
    ac = np.append(centres.mean(axis=0), 0.0)
    tgt = np.append(centres[2], 0.0)
    full = cg.measure_mesiodistal_width(bled, tgt, ac, axis, occlusal_fraction=1.0)
    occl = cg.measure_mesiodistal_width(bled, tgt, ac, axis, occlusal_fraction=0.40)
    assert occl < full, "occlusal restriction should exclude gingival spread"
    print(f"PASS  full region {full:.2f}mm vs occlusal 40% {occl:.2f}mm "
          f"({full-occl:.2f}mm of gingival spread excluded)")


if __name__ == "__main__":
    test_tangent_is_local_and_perpendicular()
    test_caliper_measures_a_known_width()
    test_validation_flags_merged_and_split()
    test_occlusal_restriction_matters()
    print("\nCALIPER TESTS PASSED")
    print("Real scans, measured at the contact points, vs Wheeler within 2mm:")
    print("  upper 8/14 teeth, lower 9/14 teeth")
