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


if __name__ == "__main__":
    test_over_threshold_is_the_CONTACT_flag_not_the_noise_floor()
    test_a_clearance_between_the_two_constants_proves_which_one_fires()
    test_the_noise_floor_actually_suppresses_sub_scanner_closure()
    test_a_failed_cap_warns_instead_of_failing_silently()
    print("\nALL FAIL-SAFE TESTS PASSED")
