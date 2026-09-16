"""Fail-safe and fault-tolerance contracts.

These tests exist because the behaviours below are only visible at the moment
something has ALREADY gone wrong — a corrupt scan, a pivot outside the alveolus,
a boolean that will not close. Nothing in the happy path exercises them, so
without a test they rot silently and are discovered by a clinician.
"""
import math
import warnings

import numpy as np
import pytest

import core_geometry as cg


# --------------------------------------------------------------------------
# The interproximal payload: WHICH constant drives WHICH field.
#
# `over_threshold` was derived from contact_mm (0.30) while sitting beside a
# field literally named `threshold_mm` (0.05) that was compared against nothing.
# Any reader assumes the flag means 0.05. Two consumers gate on this flag —
# export_planned_setup's IPR refusal and the client's status line — so its VALUE
# must not change; the names around it are what had to be fixed.
# --------------------------------------------------------------------------

# gap_before must exceed SOCKET_EXCLUSION (1.5mm): the measure drops base
# vertices within that distance of the T0 crown, because those ARE the socket
# the crown was cut from, not a neighbour. A 1.2mm wall is discarded entirely
# and the numbers then come from whatever is left at the corners.
GAP_BEFORE = 2.0


def _two_crowns(gap_after, gap_before=GAP_BEFORE):
    """A crown between two walls, with an exactly known before/after clearance.

    The 'base' plays the role of the adjacent teeth: the measure asks how far
    the crown's mesial and distal thirds are from the nearest base vertex.

    The wall lattice is sampled on the SAME y/z values as the crown, so for
    every crown vertex there is a wall vertex at identical y and z and the
    nearest-neighbour distance is exactly the x gap. Offset lattices give
    sqrt(dx^2 + dy^2 + dz^2) instead and the fixture stops asserting what it
    claims to.
    """
    ys = np.linspace(0.0, 2.0, 5)
    g = np.linspace(0.0, 4.0, 9)
    xx, yy, zz = np.meshgrid(g, ys, ys, indexing="ij")
    t0 = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = np.array([[0, 1, 2]], dtype=np.int64)     # unused by the measure

    wall_yz = np.linspace(-1.0, 3.0, 9)               # superset of ys, same spacing
    wall = [[x, y, z]
            for x in (-gap_before, 4.0 + gap_before)
            for y in wall_yz for z in wall_yz]
    base = np.array(wall, dtype=float)

    # Slide toward the distal wall. Distal clearance becomes gap_after; mesial
    # clearance grows, so `min_clearance_mm` is gap_after and `max_closure_mm`
    # is gap_before - gap_after.
    t1 = t0 + np.array([gap_before - gap_after, 0.0, 0.0])
    return t0, t1, faces, base


def test_over_threshold_is_the_CONTACT_flag_not_the_noise_floor():
    t0, t1, faces, base = _two_crowns(gap_after=0.20)
    r = cg.measure_interproximal_penetration(
        t0, t1, faces, base, None, np.array([1.0, 0.0, 0.0]),
        threshold_mm=0.05, contact_mm=0.30)

    # 0.20mm clearance is inside contact (0.30) and far outside the noise floor.
    assert r["min_clearance_mm"] == pytest.approx(0.20, abs=1e-6)
    assert r["over_threshold"] is True, "0.20mm must read as in contact at 0.30mm"
    assert r["contact_threshold_mm"] == 0.30, "the flag's own constant must be named"
    assert r["noise_floor_mm"] == 0.05, "the noise floor must be named separately"
    # Legacy keys stay for the two existing consumers.
    assert r["contact_mm"] == 0.30 and r["threshold_mm"] == 0.05
    print(f"PASS  over_threshold gates on contact_threshold_mm=0.30, "
          f"clearance {r['min_clearance_mm']}mm")


def test_a_clearance_between_the_two_constants_proves_which_one_fires():
    """0.15mm: above the 0.05 noise floor, below the 0.30 contact threshold.

    If `over_threshold` had ever been computed from `threshold_mm`, this case
    would read False. It reads True, which is the behaviour both consumers have
    always depended on.
    """
    t0, t1, faces, base = _two_crowns(gap_after=0.15)
    r = cg.measure_interproximal_penetration(
        t0, t1, faces, base, None, np.array([1.0, 0.0, 0.0]),
        threshold_mm=0.05, contact_mm=0.30)
    assert 0.05 < r["min_clearance_mm"] < 0.30
    assert r["over_threshold"] is True
    print(f"PASS  {r['min_clearance_mm']}mm sits between the constants and fires "
          f"on contact, not on the noise floor")


def test_the_noise_floor_actually_suppresses_sub_scanner_closure():
    """A 0.01mm 'closure' is the mesh, not the movement.

    Intraoral scanners resolve to 20-50 microns. `threshold_mm` was carried in
    the payload and compared against nothing, so a closure two orders below the
    scanner's own resolution was reported as a real number someone could act on.
    """
    t0, t1, faces, base = _two_crowns(gap_after=GAP_BEFORE - 0.01)
    r = cg.measure_interproximal_penetration(
        t0, t1, faces, base, None, np.array([1.0, 0.0, 0.0]),
        threshold_mm=0.05, contact_mm=0.30)
    assert r["max_closure_mm"] == 0.0, \
        f"a 0.01mm closure survived the {r['noise_floor_mm']}mm noise floor"

    # ...and a closure ABOVE the floor must survive untouched, or the floor is
    # not a floor, it is a mute button.
    t0, t1, faces, base = _two_crowns(gap_after=GAP_BEFORE - 0.30)
    r2 = cg.measure_interproximal_penetration(
        t0, t1, faces, base, None, np.array([1.0, 0.0, 0.0]),
        threshold_mm=0.05, contact_mm=0.30)
    assert r2["max_closure_mm"] == pytest.approx(0.30, abs=1e-6)
    print(f"PASS  0.01mm closure -> 0.0 (below the floor); "
          f"0.30mm closure -> {r2['max_closure_mm']} (kept)")


# --------------------------------------------------------------------------
# cap_boundary_loop: the only silent `except: pass` that was on the live
# geometry path. It sits directly under hole capping, which condition_mesh and
# cap_and_close both rely on, so a numerical failure there was indistinguishable
# from "Delaunay legitimately produced too few triangles".
# --------------------------------------------------------------------------

def test_a_failed_cap_warns_instead_of_failing_silently():
    # A degenerate loop: three coincident points. Delaunay cannot triangulate it.
    verts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    loop = [0, 1, 2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tris = cg.cap_boundary_loop(verts, loop)
    # The fan fallback is the caller's job; this function returns empty and says why.
    assert len(tris) == 0
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("cap_boundary_loop" in m for m in msgs), \
        f"a Delaunay failure produced no RuntimeWarning: {msgs}"
    assert any("centroid fan" in m for m in msgs), \
        "the warning must say what happens next, not just that something failed"
    print(f"PASS  degenerate loop -> RuntimeWarning: {msgs[0][:78]}...")


# --------------------------------------------------------------------------
# NaN/Inf gate on upload.
# --------------------------------------------------------------------------

def _damaged_scan():
    """A valid quad plus two vertices with NaN/Inf and the faces using them."""
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                  [np.nan, 0.5, 0.5], [0.5, np.inf, 0]])
    f = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 4], [2, 3, 5]], dtype=np.int64)
    return v, f


def test_the_degenerate_filter_provably_does_NOT_catch_nan():
    """The reason the gate has to exist, measured rather than asserted about.

    `condition_mesh` drops a face when `area <= 1e-12`. Every comparison against
    NaN is False, so a NaN triangle has `area = nan`, is not `<= 1e-12`, and
    survives the one filter whose entire job is to remove it.
    """
    v, f = _damaged_scan()
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    with np.errstate(invalid="ignore"):
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert np.isnan(area[2]) and np.isnan(area[3]), "fixture does not produce NaN areas"
    assert not (area[2] <= 1e-12) and not (area[3] <= 1e-12), \
        "NaN compared True against a threshold - the language changed under us"
    print("PASS  NaN areas are not <= 1e-12, so they survive condition_mesh's filter")


def test_sanitize_removes_nonfinite_and_moves_nothing():
    v, f = _damaged_scan()
    before = v.copy()
    v2, f2, rep = cg.sanitize_scan(v, f)

    assert np.isfinite(v2).all(), "a non-finite vertex survived the gate"
    assert rep["repaired"] is True
    assert rep["nonfinite_vertices"] == 2
    assert rep["faces_dropped_nonfinite"] == 2
    assert len(f2) == 2

    # RULE 3.1. Every surviving coordinate must be BIT-IDENTICAL to what arrived.
    # A repair that quietly re-centred a damaged arch would fix the crash and
    # break inter-arch registration, which is the worse outcome by far.
    assert v2.tobytes() == before[:4].tobytes(), "surviving vertices were altered"
    print(f"PASS  {rep['nonfinite_vertices']} non-finite vertices removed, "
          f"{len(v2)} survivors bit-identical to the upload")


def test_a_clean_scan_is_returned_UNCHANGED():
    """Not merely equal - the same objects.

    The gate must be free on the overwhelmingly common path, and must not
    launder a clean mesh through a copy that could quietly change its dtype or
    its ordering.
    """
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    f = np.array([[0, 1, 2]], dtype=np.int64)
    v2, f2, rep = cg.sanitize_scan(v, f)
    assert v2 is v and f2 is f
    assert rep["repaired"] is False and rep["method"] == "none needed"
    print("PASS  a clean scan passes through untouched (same objects)")


def test_the_numpy_gate_agrees_with_trimesh():
    """The brief asked for `trimesh.repair.sanitize()`, which does not exist.

    This is the measurement behind choosing NumPy instead: on the same input,
    the delete-only gate and `Trimesh.remove_infinite_values` - the nearest real
    equivalent - must produce the same surviving geometry.
    """
    trimesh = pytest.importorskip("trimesh")
    v, f = _damaged_scan()
    v2, f2, _ = cg.sanitize_scan(v.copy(), f.copy())

    m = trimesh.Trimesh(vertices=v.copy(), faces=f.copy(),
                        process=False, validate=False)
    m.remove_infinite_values()
    tv = np.asarray(m.vertices, float)
    tf = np.asarray(m.faces, np.int64)

    assert np.array_equal(np.sort(v2, axis=0), np.sort(tv, axis=0)), \
        "the NumPy gate and trimesh disagree on which vertices survive"
    assert len(f2) == len(tf)
    print(f"PASS  NumPy gate agrees with trimesh {trimesh.__version__}: "
          f"{len(v2)} verts, {len(f2)} faces both ways")


# --------------------------------------------------------------------------
# C_res projection clamp.
# --------------------------------------------------------------------------

_FRAME = {"u_oa": np.array([0.0, 0.0, 1.0]),
          "rim_centroid": np.array([0.0, 0.0, 0.0]),
          "centroid": np.array([0.0, 0.0, 4.0])}


def test_an_in_band_root_length_is_untouched_and_silent():
    rep = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = cg.center_of_resistance(_FRAME, 10.0, report=rep)
    assert "cres_clamp" not in rep
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert c[2] == pytest.approx(4.0 - 10.0)
    print("PASS  10.0mm is inside [7,15]: pivot unmoved, nothing warned")


@pytest.mark.parametrize("requested,expected,bound", [(20.0, 15.0, "max"),
                                                      (3.0, 7.0, "min")])
def test_out_of_band_clamps_AND_says_so_loudly(requested, expected, bound):
    """Clamp, and report it loudly - never a silent pivot shift.

    A 2mm pivot shift nobody was told about is only visible months later, as a
    tooth that tipped where it should have translated.
    """
    rep = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = cg.center_of_resistance(_FRAME, requested, report=rep, tooth_id="t7")
    note = rep["cres_clamp"]
    assert note["requested_mm"] == requested
    assert note["clamped_mm"] == expected
    assert note["bound"] == bound
    assert note["tooth_id"] == "t7"
    assert note["shift_mm"] == pytest.approx(abs(requested - expected))
    assert c[2] == pytest.approx(4.0 - expected)
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)], \
        "the clamp was applied without a warning - that is the silent case"
    print(f"PASS  {requested}mm -> {expected}mm ({bound} bound), "
          f"pivot shifted {note['shift_mm']}mm, warned")


def test_a_nonfinite_root_length_is_refused_not_clamped():
    """Every comparison against NaN is False, so min/max would pass it straight
    through and C_res would come out NaN - which then passes the alveolus gate
    for exactly the same reason (CLAUDE.md section 14)."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cg.clamp_cres_projection(bad)
    print("PASS  NaN and Inf root lengths are refused before the clamp")


def test_the_clamp_band_is_not_the_validation_band():
    """Two different bands for two different jobs; conflating them would either
    refuse a plausible slider position or clamp an obvious typo."""
    import validation
    assert (cg.CRES_PROJECTION_MIN_MM, cg.CRES_PROJECTION_MAX_MM) == (7.0, 15.0)
    assert validation.ROOT_LENGTH_MIN["value"] < cg.CRES_PROJECTION_MIN_MM
    assert validation.ROOT_LENGTH_MAX["value"] > cg.CRES_PROJECTION_MAX_MM
    print(f"PASS  clamp band [7,15] sits inside the refusal band "
          f"[{validation.ROOT_LENGTH_MIN['value']:.0f},"
          f"{validation.ROOT_LENGTH_MAX['value']:.0f}]")


# --------------------------------------------------------------------------
# CSG repair cascade.
# --------------------------------------------------------------------------

def _box(corner, s=1.0):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * s + corner
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
                 dtype=np.int64)
    return v, f


def test_a_clean_boolean_takes_no_repair_rung():
    m3 = pytest.importorskip("manifold3d")
    import api_core
    solid, info = api_core._batch_union_with_cascade(
        m3, [_box(np.zeros(3)), _box(np.full(3, 0.5))], "clean")
    assert info["csg_repair_stage"] == "none"
    assert "fallback_reason" not in info, \
        "a boolean that never failed must not claim a fallback was applied"
    assert solid.volume() == pytest.approx(1.875, abs=1e-6)
    print(f"PASS  clean union: no repair, volume {solid.volume():.4f}")


def test_an_open_shell_is_repaired_and_the_repair_is_RECORDED():
    """The lab prints what comes out, so a silent repair is the thing to avoid."""
    m3 = pytest.importorskip("manifold3d")
    pytest.importorskip("trimesh")
    import api_core
    v, f = _box(np.zeros(3))
    holed = (v, f[2:].copy())                 # lid removed: an open shell
    solid, info = api_core._batch_union_with_cascade(
        m3, [holed, _box(np.full(3, 0.5))], "holed")

    assert info["csg_repair_stage"] == "fix_normals+fill_holes"
    assert info["fallback_reason"] == "CSG_REPAIR_CASCADE_APPLIED"
    assert "not bit-identical" in info["fallback_detail"]
    assert info["csg_attempts"][0]["stage"] == "none"
    assert info["csg_attempts"][0]["result"] != "ok"
    assert solid.volume() == pytest.approx(1.875, abs=1e-6), \
        "the repair closed the shell but changed the solid"
    print(f"PASS  open shell repaired at '{info['csg_repair_stage']}', "
          f"volume {solid.volume():.4f}, recorded as {info['fallback_reason']}")


def test_an_unrepairable_solid_is_REFUSED_by_name_not_forced():
    """No displacement-carving rung: a tray thermoformed from a solid the
    software had to carve into shape is worse than a tray never made."""
    m3 = pytest.importorskip("manifold3d")
    import api_core
    from fastapi import HTTPException
    rng = np.random.default_rng(0)
    soup = (rng.normal(size=(30, 3)), rng.integers(0, 30, size=(40, 3)).astype(np.int64))
    with pytest.raises(HTTPException) as ei:
        api_core._batch_union_with_cascade(m3, [soup, _box(np.zeros(3))], "Stage 9")
    detail = ei.value.detail
    assert ei.value.status_code == 422
    assert "Stage 9" in detail, "the refusal must name what failed"
    for stage in ("none",) + api_core.CSG_REPAIR_CASCADE:
        assert stage in detail, f"the refusal does not say it tried '{stage}'"
    print("PASS  unrepairable input refused 422 naming every rung tried")


def test_the_weld_rung_does_not_snap_vertices_to_its_own_grid():
    """1e-5 is how duplicates are FOUND, not what replaces them.

    Snapping to the lattice would move every vertex, which is a different
    operation with a different justification.
    """
    pytest.importorskip("trimesh")
    import api_core
    v, f = _box(np.zeros(3))
    v = v + 1e-7                              # deliberately off-lattice
    v2, f2 = api_core._repair_for_csg(v, f, "weld_1e-5")
    originals = {tuple(r) for r in v}
    for row in v2:
        assert tuple(row) in originals, \
            f"vertex {row} is not an original coordinate - it was snapped"
    print(f"PASS  weld keeps original coordinates ({len(v2)} verts, off-lattice)")


def test_missing_manifold3d_is_a_sentence_not_a_traceback():
    import builtins

    import api_core
    from fastapi import HTTPException
    real = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "manifold3d":
            raise ImportError("no module named manifold3d")
        return real(name, *a, **kw)

    builtins.__import__ = blocked
    try:
        with pytest.raises(HTTPException) as ei:
            api_core._require_manifold3d()
    finally:
        builtins.__import__ = real
    assert ei.value.status_code == 503
    assert "pip install manifold3d" in ei.value.detail
    assert "works without it" in ei.value.detail, \
        "the message must say what still works, not only what broke"
    print("PASS  a missing manifold3d wheel is a 503 with an instruction")


if __name__ == "__main__":
    test_over_threshold_is_the_CONTACT_flag_not_the_noise_floor()
    test_a_clearance_between_the_two_constants_proves_which_one_fires()
    test_the_noise_floor_actually_suppresses_sub_scanner_closure()
    test_a_failed_cap_warns_instead_of_failing_silently()
    test_the_degenerate_filter_provably_does_NOT_catch_nan()
    test_sanitize_removes_nonfinite_and_moves_nothing()
    test_a_clean_scan_is_returned_UNCHANGED()
    test_the_numpy_gate_agrees_with_trimesh()
    test_an_in_band_root_length_is_untouched_and_silent()
    test_out_of_band_clamps_AND_says_so_loudly(20.0, 15.0, "max")
    test_out_of_band_clamps_AND_says_so_loudly(3.0, 7.0, "min")
    test_a_nonfinite_root_length_is_refused_not_clamped()
    test_the_clamp_band_is_not_the_validation_band()
    test_a_clean_boolean_takes_no_repair_rung()
    test_an_open_shell_is_repaired_and_the_repair_is_RECORDED()
    test_an_unrepairable_solid_is_REFUSED_by_name_not_forced()
    test_the_weld_rung_does_not_snap_vertices_to_its_own_grid()
    test_missing_manifold3d_is_a_sentence_not_a_traceback()
    print("\nALL FAIL-SAFE TESTS PASSED")
