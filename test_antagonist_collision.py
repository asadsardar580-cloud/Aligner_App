"""Dual-arch antagonist collision, swept across all 31 stages.

WHAT THIS TEST CAN AND CANNOT CLAIM, stated before the first assertion.

This repository contains ONE real scan: `case_lower.stl`, a mandible. There is
no maxillary scan. So the "opposing arch" here is the mandible mirrored through
a horizontal plane and seated into occlusion — the same construction
`test_occlusal_collision.py` uses, and it is a FIXTURE, not a patient.

What that fixture genuinely validates: the arithmetic, the sign convention, the
per-stage sweep, the threshold behaviour, and the fact that both arches are read
in one shared coordinate system. What it cannot validate is a clinical result on
real opposing anatomy. CLAUDE.md has recorded that gap since the check was
written and this test does not close it — it makes the mechanism testable, which
is a different and smaller claim.

THE CHECK ONLY MEANS ANYTHING BECAUSE SCANNER COORDINATES ARE SACRED. Neither
arch is ever re-centred, so an upper incisor driven lingually really does land
where the lower arch is. Had either been normalised to its own origin — the
obvious tidy-up — this would compare two unrelated coordinate systems and return
noise that looks like a measurement.
"""
import numpy as np

import core_geometry as cg
import validation
from tooth_fixture import flat_molar_on_base

STAGES = 31                     # the case length the manufacturing export uses
PENETRATION_THRESHOLD_MM = 0.1  # what check_occlusal_collision applies by default


def _crown_and_antagonist(gap_mm=2.0):
    """A crown, and an opposing plate seated `gap_mm` above its occlusal tip.

    The plate is built from the same fixture mirrored, so both live in one
    coordinate space by construction — which is the property under test.
    """
    v, f, r = flat_molar_on_base()
    crown_mask = np.linalg.norm(v[:, :2], axis=1) < r * 0.95
    crown = v[crown_mask]

    top = float(crown[:, 2].max())
    # Mirror through a plane HALF the gap above the tip: reflecting through a
    # plane `g` away puts the reflected surface `2g` away, so mirroring at
    # top + gap/2 is what actually yields a `gap_mm` clearance. Measured, not
    # assumed - the first version of this fixture was silently 2x too far apart
    # and the interference test could never fire.
    plane = top + gap_mm / 2.0
    opp = v.copy()
    opp[:, 2] = 2 * plane - opp[:, 2]
    opp_faces = f[:, ::-1]      # mirroring flips winding; restore outward normals
    return crown, opp, opp_faces


def test_both_arches_share_one_coordinate_system():
    """If either arch were re-centred, every number below would be noise."""
    crown, opp, opp_faces = _crown_and_antagonist(gap_mm=2.0)
    # The opposing plate must sit ABOVE the crown in the same space, not at the
    # origin. A re-centred arch would have its centroid near (0,0,0).
    assert opp[:, 2].min() > crown[:, 2].min(), \
        "the antagonist is not positioned above the crown — coordinates were normalised"
    centre = opp.mean(axis=0)
    assert abs(centre[2]) > 1.0, \
        f"the opposing arch centroid sits at z={centre[2]:.2f}, i.e. on the origin — " \
        f"it has been re-centred and the collision check is measuring nothing"
    print(f"PASS  both arches in one space: crown top {crown[:,2].max():.2f}, "
          f"antagonist floor {opp[:,2].min():.2f}")


def test_clearance_reports_no_collision_when_separated():
    crown, opp, opp_faces = _crown_and_antagonist(gap_mm=2.0)
    normals = cg.vertex_normals(opp, opp_faces)
    index = cg.build_antagonist_index(opp)
    res = cg.check_occlusal_collision(crown, opp, normals, index=index)

    assert res["collides"] is False, \
        f"a 2mm gap reported a collision ({res['max_penetration_mm']}mm)"
    assert res["max_penetration_mm"] <= PENETRATION_THRESHOLD_MM
    state = validation.occlusion_state(True, res["max_penetration_mm"], res["threshold_mm"])
    assert state["state"] in (validation.CLEAR, validation.COLLISION_WARNING)
    print(f"PASS  2mm separation -> {state['state']}, "
          f"penetration {res['max_penetration_mm']}mm")


def test_interference_fires_above_the_threshold():
    """Drive the crown INTO the antagonist and the warning must appear."""
    crown, opp, opp_faces = _crown_and_antagonist(gap_mm=2.0)
    normals = cg.vertex_normals(opp, opp_faces)
    index = cg.build_antagonist_index(opp)

    driven = crown + np.array([0.0, 0.0, 3.0])     # 3mm up, past the 2mm gap
    res = cg.check_occlusal_collision(driven, opp, normals, index=index)

    assert res["collides"] is True, \
        f"driving 3mm into a 2mm gap did not collide ({res['max_penetration_mm']}mm)"
    assert res["max_penetration_mm"] > PENETRATION_THRESHOLD_MM
    state = validation.occlusion_state(True, res["max_penetration_mm"], res["threshold_mm"])
    assert state["state"] == validation.INTERFERENCE
    assert "APPROXIMATE" in state["limitation"], \
        "a nearest-vertex measure must state it is not a signed distance field"
    print(f"PASS  3mm drive into a 2mm gap -> INTERFERENCE at "
          f"{res['max_penetration_mm']:.2f}mm")


def test_the_sweep_across_31_stages_finds_where_contact_begins():
    """The real question: not whether the endpoint collides, but WHEN.

    A movement that is clear at T0 and clear at the setup can still drive
    through the antagonist in the middle, and a per-stage sweep is the only
    thing that sees it.
    """
    crown, opp, opp_faces = _crown_and_antagonist(gap_mm=2.0)
    normals = cg.vertex_normals(opp, opp_faces)
    index = cg.build_antagonist_index(opp)        # cached once, as production does

    total_extrusion_mm = 3.0                      # ends 1mm inside the antagonist
    first_contact, penetrations = None, []
    for k in range(STAGES + 1):
        moved = crown + np.array([0.0, 0.0, total_extrusion_mm * k / STAGES])
        res = cg.check_occlusal_collision(moved, opp, normals, index=index)
        penetrations.append(res["max_penetration_mm"])
        if res["collides"] and first_contact is None:
            first_contact = k

    assert penetrations[0] <= PENETRATION_THRESHOLD_MM, "stage 0 already collides"
    assert first_contact is not None, "a 3mm extrusion into a 2mm gap never collided"
    assert 0 < first_contact < STAGES, \
        f"first contact at stage {first_contact} — expected it partway through"

    # Penetration must grow monotonically with a monotonic extrusion. If it does
    # not, the measure is not tracking the movement.
    diffs = np.diff(penetrations)
    assert (diffs >= -1e-6).all(), \
        f"penetration decreased during a monotonic extrusion: min step {diffs.min():.4f}"

    # Roughly 2 of 3mm consumed before contact -> about two thirds of the way.
    expected = int(STAGES * 2.0 / 3.0)
    assert abs(first_contact - expected) <= 3, \
        f"first contact at stage {first_contact}, expected near {expected}"
    print(f"PASS  31-stage sweep: first contact at stage {first_contact}/{STAGES}, "
          f"penetration 0.00 -> {penetrations[-1]:.2f}mm, monotonic")


def test_a_missing_antagonist_is_NOT_CHECKED_not_clear():
    """The single easiest way for this software to mislead someone."""
    state = validation.occlusion_state(checked=False, max_penetration_mm=None,
                                       threshold_mm=PENETRATION_THRESHOLD_MM)
    assert state["state"] == validation.NOT_CHECKED
    assert state["state"] != validation.CLEAR
    assert "NOT a finding of no interference" in state["detail"]
    print("PASS  no opposing arch -> NOT_CHECKED with an explicit disclaimer")


def test_the_index_is_cached_and_the_sweep_is_affordable():
    """Rebuilding the KD-tree per call was 43ms of the 89ms a 2000-vertex crown
    cost, and a 31-stage 14-crown run asks 434 questions of an arch that never
    changes."""
    import time
    crown, opp, opp_faces = _crown_and_antagonist(gap_mm=2.0)
    normals = cg.vertex_normals(opp, opp_faces)

    t0 = time.time()
    index = cg.build_antagonist_index(opp)
    build = time.time() - t0

    t1 = time.time()
    for k in range(STAGES + 1):
        cg.check_occlusal_collision(crown + np.array([0.0, 0.0, 0.05 * k]),
                                    opp, normals, index=index)
    swept = time.time() - t1

    assert swept < 20.0, f"a 31-stage sweep took {swept:.1f}s — too slow to run per commit"
    print(f"PASS  index built once in {build*1000:.0f}ms, {STAGES+1} stages swept "
          f"in {swept*1000:.0f}ms ({swept*1000/(STAGES+1):.1f}ms/stage)")


def test_this_fixture_is_not_a_patient():
    """Recorded as an assertion so the limitation cannot be quietly forgotten."""
    import os
    assert not os.path.exists("case_upper.stl"), (
        "A maxillary scan now exists in the repo. This test still uses a MIRRORED "
        "mandible as its antagonist — rewrite it against the real opposing arch "
        "and delete this assertion.")
    print("PASS  no real opposing arch exists; the antagonist here is a mirrored "
          "fixture and the clinical gap remains open")


if __name__ == "__main__":
    test_both_arches_share_one_coordinate_system()
    test_clearance_reports_no_collision_when_separated()
    test_interference_fires_above_the_threshold()
    test_the_sweep_across_31_stages_finds_where_contact_begins()
    test_a_missing_antagonist_is_NOT_CHECKED_not_clear()
    test_the_index_is_cached_and_the_sweep_is_affordable()
    test_this_fixture_is_not_a_patient()
    print("\nALL ANTAGONIST COLLISION TESTS PASSED")
