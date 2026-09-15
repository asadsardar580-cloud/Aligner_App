"""Space analysis: crown width against Wheeler, and interproximal clearance.

What is pinned here is that the software MEASURES and FLAGS, and never
prescribes. A merged segmentation reads as an implausible width; two crowns
0.1mm apart read as a contact. Neither blocks anything, and every number carries
the method that produced it — because "IPR Contact 11-21: 0.22 mm" looks far
more precise than a vertex-to-vertex minimum on a triangulated mesh actually is.
"""
import numpy as np
from fastapi import HTTPException

import api_core
import space_analysis as sa
from session_store import STORE
from test_hydration import build_cut_session


def _block(centre, half_md, half_bl, half_oa, n=8):
    """A rectangular cloud of the given half-extents — a stand-in crown whose
    mesiodistal width is known exactly, so the caliper can be checked."""
    g = np.linspace(-1, 1, n)
    pts = np.array([[x, y, z] for x in g for y in g for z in g], float)
    return pts * np.array([half_md, half_bl, half_oa], float) + np.asarray(centre, float)


def test_the_caliper_recovers_a_known_width():
    """A 7mm-wide block must measure ~7mm, not a face count."""
    axis = np.array([0.0, 0.0, 1.0])
    centre = np.array([0.0, -20.0, 0.0])      # sits on +? of the arch centre
    crown = _block(centre, 3.5, 2.5, 4.0)     # 7.0mm mesiodistal
    teeth = [{"tooth_id": "t1", "fdi": 13, "verts": crown}]
    out = sa.crown_widths(teeth, "upper", axis, np.array([0.0, 0.0, 0.0]))
    w = out["teeth"][0]["width_mm"]
    # The occlusal 40% restriction means the measured extent is of the upper
    # slab, which for a straight block is still the full width.
    assert 6.0 < w < 8.0, f"a 7mm block measured {w}mm"
    print(f"PASS  caliper recovers a 7.0mm block as {w}mm")


def test_a_merged_crown_is_flagged_REVIEW_not_refused():
    """Twice the Wheeler width is two teeth merged by the segmenter."""
    axis = np.array([0.0, 0.0, 1.0])
    arch_centre = np.array([0.0, 0.0, 0.0])
    # FDI 13 is a maxillary canine: Wheeler 7.5mm.
    normal = {"tooth_id": "ok", "fdi": 13, "verts": _block([0, -20, 0], 3.75, 2.5, 4.0)}
    merged = {"tooth_id": "bad", "fdi": 12, "verts": _block([0, -20, 0], 7.0, 2.5, 4.0)}
    out = sa.crown_widths([normal, merged], "upper", axis, arch_centre)
    rows = {r["tooth_id"]: r for r in out["teeth"]}

    assert rows["bad"]["status"] == "REVIEW", "a 14mm crown was not flagged"
    assert "merged" in rows["bad"]["detail"], \
        "the message must name the likely cause, not just the number"
    assert rows["bad"]["delta_mm"] > 0
    # It is a flag, not a refusal: the row still carries its measurement.
    assert rows["bad"]["width_mm"] > 12
    assert out["tolerance_basis"] == "literature/reference"
    print(f"PASS  merged crown flagged REVIEW at {rows['bad']['width_mm']}mm "
          f"(Wheeler {rows['bad']['expected_mm']}mm), not refused")


def test_an_unidentified_tooth_reads_UNKNOWN_not_PASS():
    """No FDI means no Wheeler comparison is possible. Silence must not pass."""
    axis = np.array([0.0, 0.0, 1.0])
    teeth = [{"tooth_id": "t1", "fdi": None, "verts": _block([0, -20, 0], 3.5, 2.5, 4.0)}]
    out = sa.crown_widths(teeth, "upper", axis, np.array([0.0, 0.0, 0.0]))
    r = out["teeth"][0]
    assert r["status"] == "UNKNOWN", f"unidentified tooth reported {r['status']}"
    assert "No FDI" in r["detail"]
    assert out["reliable"] is False, "a pass rate over one tooth is not a grade"
    print("PASS  no FDI -> UNKNOWN with a reason, and reliable stays False")


def test_touching_and_separated_crowns_are_distinguished():
    a = {"tooth_id": "a", "fdi": 11, "verts": _block([0.0, 0, 0], 2.0, 2.0, 3.0)}
    touching = {"tooth_id": "b", "fdi": 21, "verts": _block([4.1, 0, 0], 2.0, 2.0, 3.0)}
    apart = {"tooth_id": "c", "fdi": 22, "verts": _block([9.0, 0, 0], 2.0, 2.0, 3.0)}

    out = sa.interproximal_contacts([a, touching, apart])
    by_pair = {c["pair"]: c for c in out["contacts"]}

    assert by_pair["11-21"]["state"] == "CONTACT", \
        f"crowns 0.1mm apart read as {by_pair['11-21']['state']}"
    assert by_pair["11-22"]["state"] == "CLEAR"
    assert by_pair["11-21"]["readout"].startswith("IPR Contact 11-21:")
    assert "OVERESTIMATES" in out["limitation"], \
        "a vertex-to-vertex minimum must say it is not a surface distance"
    print(f"PASS  {by_pair['11-21']['readout']} (CONTACT) vs "
          f"{by_pair['11-22']['gap_mm']}mm (CLEAR)")


def test_noise_floor_absorbs_sub_50_micron_gaps():
    """Scanners resolve to 20-50 microns; enamel in real contact still measures
    a small positive separation in a triangulated mesh."""
    a = {"tooth_id": "a", "fdi": 11, "verts": _block([0.0, 0, 0], 2.0, 2.0, 3.0)}
    b = {"tooth_id": "b", "fdi": 21, "verts": _block([4.02, 0, 0], 2.0, 2.0, 3.0)}
    out = sa.interproximal_contacts([a, b])
    c = out["contacts"][0]
    assert c["state"] == "CONTACT" and c["gap_mm"] <= sa.NOISE_FLOOR_MM
    assert "noise" in c["detail"]
    assert out["thresholds_basis"] == "software heuristic"
    print(f"PASS  a {c['gap_mm']}mm gap is absorbed as scanner noise, labelled as such")


def test_endpoint_refuses_without_an_occlusal_reference():
    """The mesiodistal direction is derived per tooth from the arch curve,
    which only exists in that frame."""
    sid, cut, v, f = build_cut_session()
    try:
        STORE.put(sid, "arch_frame", None)
        try:
            api_core.space_analysis_report(sid)
            raise AssertionError("space analysis ran with no occlusal reference")
        except HTTPException as e:
            assert e.status_code == 409, f"got {e.status_code}, want 409"
            assert "occlusal" in str(e.detail).lower()
        print("PASS  /space-analysis refuses with 409 and says why")
    finally:
        api_core.close_session(sid)


def test_endpoint_runs_on_a_real_cut_session():
    sid, cut, v, f = build_cut_session()
    try:
        out = api_core.space_analysis_report(sid)
        assert out["crowns_measured"] == 1
        assert out["summary"], "no renderable summary line"
        row = out["widths"]["teeth"][0]
        # This fixture is a single tooth centred ON the arch centre, where the
        # radial direction - and therefore the mesiodistal axis - is undefined.
        # The report must say UNMEASURABLE rather than return 0.0, which would
        # read as a measured zero-width crown.
        assert row["status"] in ("UNKNOWN", "UNMEASURABLE"), row["status"]
        if row["status"] == "UNMEASURABLE":
            assert row["width_mm"] is None and "undefined" in row["detail"]
        else:
            assert row["width_mm"] > 0
        import json
        json.dumps(out)
        print(f"PASS  endpoint on a real cut: status {row['status']}, "
              f"width={row['width_mm']}, summary present")
    finally:
        api_core.close_session(sid)


if __name__ == "__main__":
    test_the_caliper_recovers_a_known_width()
    test_a_merged_crown_is_flagged_REVIEW_not_refused()
    test_an_unidentified_tooth_reads_UNKNOWN_not_PASS()
    test_touching_and_separated_crowns_are_distinguished()
    test_noise_floor_absorbs_sub_50_micron_gaps()
    test_endpoint_refuses_without_an_occlusal_reference()
    test_endpoint_runs_on_a_real_cut_session()
    print("\nALL SPACE ANALYSIS TESTS PASSED")
