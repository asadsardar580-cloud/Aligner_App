"""The antagonist collision check, and the manufacturing export's crown gate.

WHY BOTH LIVE HERE: they are the two things standing between a plan and a
printed tray, and both are about refusing to hand a lab something wrong.

The collision check is only possible at all because scanner coordinates are
sacred (CLAUDE.md rule 3.1). Neither arch is ever re-centred, so the upper and
lower scans sit in the same raw space and an upper incisor retracted lingually
really does land where the lower arch is. Had either been normalised to its own
origin — the obvious thing to do — this check would be measuring nothing.

There is NO upper-arch scan in this repository, so the antagonist here is
synthetic. What is pinned is the arithmetic and the wiring, not a clinical
result on a real opposing pair.
"""
import numpy as np
from fastapi import HTTPException

import core_geometry as cg
import api_core
from session_store import STORE
from test_cut_endpoint import arch_session


def flat_antagonist(z=0.0, extent=20.0, n=41, facing_up=True):
    """A flat plate as a stand-in opposing arch, with known normals."""
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    verts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    normals = np.tile([0.0, 0.0, 1.0 if facing_up else -1.0], (len(verts), 1))
    return verts, normals


def test_penetration_is_measured_and_signed():
    opp, nrm = flat_antagonist(z=0.0)

    clear = cg.check_occlusal_collision(np.array([[0, 0, 2.0], [3, 3, 0.5]]), opp, nrm)
    assert not clear["collides"], clear
    assert clear["max_penetration_mm"] == 0.0

    # 0.05mm in is under the 0.1mm threshold — real contact, not interference.
    graze = cg.check_occlusal_collision(np.array([[0, 0, -0.05]]), opp, nrm)
    assert not graze["collides"], graze

    deep = cg.check_occlusal_collision(np.array([[0, 0, -0.9], [1, 1, -0.3]]), opp, nrm)
    assert deep["collides"], deep
    assert abs(deep["max_penetration_mm"] - 0.9) < 1e-6, deep
    assert deep["points_penetrating"] == 2

    # THE SIGN IS THE WHOLE POINT. Flip the antagonist's normals and the same
    # geometry reads as clearance — an unsigned nearest-distance check cannot
    # tell 0.9mm of penetration from 0.9mm of gap, and would warn on every
    # tooth in normal occlusion.
    flipped = cg.check_occlusal_collision(np.array([[0, 0, -0.9]]), opp, -nrm)
    assert not flipped["collides"], \
        "the check is not using the normal's sign, so it cannot tell in from out"
    print(f"PASS  penetration signed correctly: 0.9mm in -> collides "
          f"({deep['points_penetrating']} pts), 0.05mm in -> clear, "
          f"2.0mm out -> clear, normals flipped -> clear")


def test_empty_and_missing_inputs_never_raise():
    """A missing antagonist means the opposing arch is not loaded. That is the
    ordinary single-arch case and must never block anything."""
    opp, nrm = flat_antagonist()
    assert cg.check_occlusal_collision(np.empty((0, 3)), opp, nrm)["collides"] is False
    assert cg.check_occlusal_collision(np.array([[0, 0, -5.0]]),
                                       np.empty((0, 3)), np.empty((0, 3)))["collides"] is False
    assert api_core._antagonist(None) is None
    assert api_core._antagonist("no-such-session") is None
    assert api_core._occlusal_check(None, np.array([[0, 0, 0.0]])) is None
    print("PASS  absent, unknown and empty antagonists all skip cleanly, never raise")


def test_kinematics_reports_interference_without_blocking():
    sid, reqs, _ = arch_session()
    opp_sid = STORE.create("upper")
    try:
        out = api_core.cut(sid, reqs[0])
        tid = out["tooth_id"]
        rec = STORE.require(sid, f"tooth:{tid}")

        # Put the antagonist plate just above the crown, facing down at it, so
        # an extrusion drives the tooth into it.
        cv = rec["cv"]
        top = cv[:, 2].max()
        opp, nrm = flat_antagonist(z=top + 0.5, extent=60.0, n=61, facing_up=False)
        STORE.put(opp_sid, "verts", opp)
        STORE.put(opp_sid, "faces", np.zeros((0, 3), int))
        STORE.put(opp_sid, "vertex_normals", nrm)     # skip normal derivation

        quiet = api_core.kinematics(sid, tid, api_core.KinematicsRequest(
            d_oa=0.1, opposing_session_id=opp_sid))
        assert quiet["occlusal_warning"] is None, quiet["occlusion"]

        loud = api_core.kinematics(sid, tid, api_core.KinematicsRequest(
            d_oa=2.0, opposing_session_id=opp_sid))
        assert loud["occlusion"]["collides"] is True, loud["occlusion"]
        assert loud["occlusal_warning"] == api_core.ANTAGONIST_WARNING
        # NON-BLOCKING: the movement is still committed and the matrix returned.
        assert loud["matrix"] is not None
        assert STORE.require(sid, f"tooth:{tid}")["clinical"]["d_oa"] == 2.0

        none = api_core.kinematics(sid, tid, api_core.KinematicsRequest(d_oa=2.0))
        assert none["occlusion"] is None and none["occlusal_warning"] is None

        print(f"PASS  0.1mm -> no warning; 2.0mm -> "
              f"{loud['occlusion']['max_penetration_mm']:.2f}mm penetration warned and "
              f"still committed; no opposing session -> silent")
    finally:
        STORE.drop(sid)
        STORE.drop(opp_sid)


def test_stage_export_records_interference_and_still_ships():
    sid, reqs, _ = arch_session()
    opp_sid = STORE.create("upper")
    try:
        out = api_core.cut(sid, reqs[0])
        rec = STORE.require(sid, f"tooth:{out['tooth_id']}")
        top = rec["cv"][:, 2].max()
        opp, nrm = flat_antagonist(z=top + 0.3, extent=60.0, n=61, facing_up=False)
        STORE.put(opp_sid, "verts", opp)
        STORE.put(opp_sid, "faces", np.zeros((0, 3), int))
        STORE.put(opp_sid, "vertex_normals", nrm)
        api_core.kinematics(sid, out["tooth_id"],
                            api_core.KinematicsRequest(d_oa=1.5))

        res = api_core.build_stage_bundle(sid, api_core.StageExportRequest(
            opposing_session_id=opp_sid))
        occ = res["manifest"]["occlusion"]
        assert occ["checked"] is True
        assert occ["stages_with_interference"], occ
        assert occ["warning"] == api_core.ANTAGONIST_WARNING
        assert occ["worst_penetration_mm"] > 0.1
        # The files still exist. A warning is not a refusal.
        assert len(res["manifest"]["stage_files"]) == res["manifest"]["stages"]

        clean = api_core.build_stage_bundle(sid, api_core.StageExportRequest())
        assert clean["manifest"]["occlusion"]["checked"] is False, \
            "no opposing session must report 'not checked', never 'no interference'"
        print(f"PASS  export records interference on stages "
              f"{occ['stages_with_interference']} (worst "
              f"{occ['worst_penetration_mm']:.2f}mm) in {occ['seconds']:.3f}s and still "
              f"ships {res['manifest']['stages']} stages; unchecked is reported as such")
    finally:
        STORE.drop(sid)
        STORE.drop(opp_sid)


# =========================================================================
# The crown gate
# =========================================================================

def test_topology_screen_names_the_specific_defect():
    # A 3mm cube: watertight, chi=2, 27mm3 — a solid.
    v = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0],
                  [0, 0, 3], [3, 0, 3], [3, 3, 3], [0, 3, 3]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    ok = cg.crown_is_printable(v, f)
    assert ok["ok"] and ok["euler_characteristic"] == 2 and ok["genus"] == 0

    tiny = cg.crown_is_printable(v * 0.5, f)      # 3.4mm3
    assert not tiny["ok"] and "mm3" in tiny["reason"]

    holed = cg.crown_is_printable(v, f[:-2])
    assert not holed["ok"] and "not closed" in holed["reason"]

    print(f"PASS  screen: 27mm3 cube passes; 3.4mm3 rejected on volume; "
          f"open box rejected as not closed")


def test_the_gate_refuses_a_shell_with_the_clinical_message():
    """The stated gate — watertight, single-component, chi=2 — is NOT enough.

    Measured on four auto-cut crowns from the real scan: every one was already
    watertight and single-bodied, because /cut asserts is_edge_manifold_closed
    and refuses otherwise. chi caught one, a volume floor caught a 0.2mm speck,
    and one crown passed every topological test and still fractured into three
    pieces when fused. So the decisive check is the boolean itself.
    """
    sid, reqs, _ = arch_session()
    try:
        out = api_core.cut(sid, reqs[0])
        api_core.kinematics(sid, out["tooth_id"],
                            api_core.KinematicsRequest(d_md=0.4))

        rec = STORE.require(sid, f"tooth:{out['tooth_id']}")
        # Shrink the crown to a speck: still watertight, still chi=2, still one
        # body — and no longer a tooth.
        rec["cv"] = (rec["cv"] - rec["cv"].mean(axis=0)) * 0.02 + rec["cv"].mean(axis=0)
        STORE.put(sid, f"tooth:{out['tooth_id']}", rec)
        screen = cg.crown_is_printable(rec["cv"], rec["cf"])
        assert screen["watertight"] and screen["euler_characteristic"] == 2, \
            "the shrunken crown must still pass every topological test, or this is vacuous"

        try:
            api_core.build_stage_bundle(sid, api_core.StageExportRequest())
            raise AssertionError("a shell was exported as a printable stage model")
        except HTTPException as e:
            assert e.status_code == 422, e.status_code
            d = str(e.detail)
            assert d.startswith(api_core.SHELL_REFUSAL), d[:120]
            assert "mm3" in d, d
        print("PASS  a watertight Euler-2 shell is refused with the clinical message, "
              "naming the tooth and its measurement")
    finally:
        STORE.drop(sid)


def test_a_mixed_case_refuses_wholesale():
    """A stage model missing a tooth is a wrong model: the lab would thermoform
    a tray with a gap where a tooth should be and nothing would catch it."""
    sid, reqs, _ = arch_session()
    try:
        good = api_core.cut(sid, reqs[0])
        bad = api_core.cut(sid, reqs[1])
        api_core.kinematics(sid, good["tooth_id"], api_core.KinematicsRequest(d_md=0.4))

        rec = STORE.require(sid, f"tooth:{bad['tooth_id']}")
        rec["cv"] = (rec["cv"] - rec["cv"].mean(axis=0)) * 0.02 + rec["cv"].mean(axis=0)
        STORE.put(sid, f"tooth:{bad['tooth_id']}", rec)

        try:
            api_core.build_stage_bundle(sid, api_core.StageExportRequest())
            raise AssertionError("a partial model was exported")
        except HTTPException as e:
            assert e.status_code == 422
            assert "1 of 2 crown(s)" in str(e.detail), str(e.detail)[:160]
            assert "rather than shipping a model with a tooth missing" in str(e.detail)
        print("PASS  one bad crown of two refuses the whole export, not a partial model")
    finally:
        STORE.drop(sid)


if __name__ == "__main__":
    test_penetration_is_measured_and_signed()
    test_empty_and_missing_inputs_never_raise()
    test_kinematics_reports_interference_without_blocking()
    test_stage_export_records_interference_and_still_ships()
    test_topology_screen_names_the_specific_defect()
    test_the_gate_refuses_a_shell_with_the_clinical_message()
    test_a_mixed_case_refuses_wholesale()
    print("\nALL OCCLUSAL COLLISION AND CROWN GATE TESTS PASSED")
