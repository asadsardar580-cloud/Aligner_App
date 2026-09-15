"""Structured validation and explicit collision states.

Two things are pinned here, and both are about what the software is allowed to
CLAIM rather than about geometry.

1. A refusal must carry its numbers. The measured value, the threshold, and
   where that threshold came from. A bare string tells a clinician nothing they
   can act on, and tells a UI nothing it can render.

2. NOT_CHECKED is not CLEAR. The antagonist check is skipped whenever the
   opposing arch was never loaded. Reporting that as "no interference" would be
   a clinical claim the software never made, and it is the single easiest way
   for this system to mislead someone.
"""
import math

import numpy as np
from fastapi import HTTPException

import api_core
import validation as V
from session_store import STORE
from test_hydration import build_cut_session
from tooth_fixture import flat_molar_on_base


def test_every_threshold_declares_its_provenance():
    """A heuristic and a literature value are different kinds of claim."""
    thresholds = [V.MIN_SELECTED_FACES, V.MAX_GINGIVA_FRACTION,
                  V.MIN_LARGEST_COMPONENT_FRACTION, V.MAX_SECOND_LABEL_FRACTION,
                  V.ROOT_LENGTH_MIN, V.ROOT_LENGTH_MAX, V.CROWN_HEIGHT_BAND]
    for t in thresholds:
        assert t["basis"] in (V.HEURISTIC, V.LITERATURE), \
            f"threshold {t} does not say where it came from"
        assert t["note"], "a threshold with no rationale cannot be argued with"
        assert t["units"], "a threshold without units is not a measurement"
    n_heur = sum(1 for t in thresholds if t["basis"] == V.HEURISTIC)
    print(f"PASS  {len(thresholds)} thresholds, all with provenance "
          f"({n_heur} heuristic, {len(thresholds)-n_heur} literature-derived)")


def test_a_refusal_carries_its_numbers():
    r = V.validate_root_length(0.0)
    assert not r.ok, "a 0mm root must be refused"
    d = r.as_dict()
    err = d["errors"][0]
    assert err["metrics"], "the refusal carried no measurement"
    assert err["threshold"] is not None, "the refusal did not say what it compared against"
    assert err["threshold"]["basis"], "the threshold did not say where it came from"
    assert "0.0" in err["message"] or "0" in err["message"], \
        "the message does not contain the offending value"
    print(f"PASS  refusal carries metrics + threshold + provenance: "
          f"{err['threshold']['value']}{err['threshold']['units']} ({err['threshold']['basis']})")


def test_root_length_band_rejects_nonsense_and_accepts_wheeler():
    for bad in (0.0, -9.0, 1.5, 250.0, float("nan"), float("inf")):
        assert not V.validate_root_length(bad).ok, f"root length {bad} was accepted"
    for good in (9.0, 10.0, 13.0):     # premolar/molar, incisor, canine
        assert V.validate_root_length(good).ok, f"Wheeler value {good} was refused"
    print("PASS  root length: 6 nonsense values refused, Wheeler 9/10/13mm accepted")


def test_nan_cres_is_caught_because_comparisons_against_nan_are_false():
    """The bug this exists to prevent.

    `if depth <= 0 or lateral > max_lateral` looks exhaustive. It is not: every
    comparison against NaN evaluates False, so a NaN pivot passed the gate and
    went on into a transform matrix. Finiteness must be its own check, first.
    """
    nan = float("nan")
    assert not (nan <= 0), "sanity: NaN <= 0 is False"
    assert not (nan > 5.0), "sanity: NaN > 5 is False"

    r = V.validate_pivot(nan, 0.5, 4.0)
    assert not r.ok, "a NaN pivot passed validation"
    assert any(c["id"] == "cres_finite" for c in r.errors), \
        "NaN was caught by the wrong check — finiteness must be explicit"

    assert V.validate_pivot(9.0, 0.4, 4.0).ok, "a healthy pivot was refused"
    assert not V.validate_pivot(-2.0, 0.4, 4.0).ok, "a pivot above the margin was accepted"
    assert not V.validate_pivot(9.0, 40.0, 4.0).ok, "a wildly lateral pivot was accepted"
    print("PASS  NaN C_res refused by an explicit finiteness check, not by a comparison")


def test_landmarks_must_be_distinct_and_finite():
    ok = V.validate_landmarks([-5, 0, 2], [5, 0, 2])
    assert ok.ok, "valid landmarks were refused"
    assert not V.validate_landmarks([0, 0, 0], [0, 0, 0]).ok, "coincident points accepted"
    assert not V.validate_landmarks([0, 0], [5, 0, 2]).ok, "a 2D point was accepted"
    assert not V.validate_landmarks([float("nan"), 0, 0], [5, 0, 2]).ok, "NaN accepted"
    print("PASS  landmarks: coincident, 2D and NaN all refused with reasons")


def test_selection_warnings_report_what_was_discarded():
    # A selection in pieces: 100 faces, largest island only 40.
    r = V.validate_selection(n_selected_vertices=300, n_vertices=10000,
                             n_selected_faces=100, largest_component_faces=40)
    assert r.ok, "a fragmented selection should WARN, not refuse"
    w = [c for c in r.warnings if c["id"] == "selection_connected"]
    assert w, "the silent prune was not reported"
    assert w[0]["metrics"]["discarded_faces"] == 60, "did not say how much was discarded"

    # Mostly gingiva.
    r2 = V.validate_selection(300, 10000, 100, 100, label_histogram={0: 80, 36: 20})
    assert any(c["id"] == "selection_is_crown_not_gingiva" for c in r2.warnings)

    # Two teeth in one flood.
    r3 = V.validate_selection(300, 10000, 100, 100, label_histogram={36: 60, 37: 40})
    assert any(c["id"] == "selection_is_one_tooth" for c in r3.warnings)

    # No segmentation: the checks must be declared skipped, not silently passed.
    r4 = V.validate_selection(300, 10000, 100, 100, label_histogram=None)
    assert any(c["id"] == "selection_label_checks_skipped" for c in r4.checks), \
        "skipped checks must be visible, not absent"
    print("PASS  selection warnings: fragmentation (60 faces named), gingiva, two teeth, "
          "and 'skipped' stated explicitly")


def test_not_checked_is_not_clear():
    """The most important assertion in this file."""
    nc = V.occlusion_state(checked=False, max_penetration_mm=None, threshold_mm=0.1)
    assert nc["state"] == V.NOT_CHECKED
    assert nc["state"] != V.CLEAR
    assert "NOT a finding of no interference" in nc["detail"], \
        "the payload must say in words that this is not a clearance result"

    clear = V.occlusion_state(True, 0.0, 0.1)
    assert clear["state"] == V.CLEAR
    warn = V.occlusion_state(True, 0.05, 0.1)
    assert warn["state"] == V.COLLISION_WARNING
    bad = V.occlusion_state(True, 0.9, 0.1)
    assert bad["state"] == V.INTERFERENCE
    err = V.occlusion_state(True, None, 0.1, error="KD-tree failed")
    assert err["state"] == V.COMPUTATION_ERROR

    for s in (clear, warn, bad):
        assert "APPROXIMATE" in s["limitation"], \
            "a nearest-vertex measurement must state that it is not an SDF"
    print("PASS  5 distinct collision states; NOT_CHECKED carries an explicit disclaimer")


def test_cut_refuses_out_of_range_vertex_ids_instead_of_wrapping():
    """A negative id used to index from the END of the array — a crown built
    from the far side of the arch, reported as success."""
    sid, cut, v, f = build_cut_session()
    try:
        for bad_ids in ([-1, 2, 3], [len(v) + 5, 1, 2]):
            try:
                api_core.cut(sid, api_core.CutRequest(
                    vertex_ids=bad_ids, mesial_pt=[-1, 0, 0], distal_pt=[1, 0, 0],
                    root_length_mm=9.0))
                raise AssertionError(f"{bad_ids[:1]} was accepted")
            except HTTPException as e:
                assert e.status_code == 422, f"got {e.status_code}, want 422"
                assert isinstance(e.detail, dict), "refusal was a bare string"
                assert e.detail["errors"][0]["metrics"]["first_offender"] is not None
        print("PASS  out-of-range and negative vertex ids refused, offender named")
    finally:
        api_core.close_session(sid)


def test_cut_refuses_an_impossible_root_length_with_structure():
    sid, cut, v, f = build_cut_session()
    try:
        r = np.linalg.norm(v[:, :2], axis=1)
        sel = np.where((r < 3.0) & (v[:, 2] > v[:, 2].min() + 0.35))[0]
        try:
            api_core.cut(sid, api_core.CutRequest(
                vertex_ids=sel.tolist(), mesial_pt=[-4, 0, 1], distal_pt=[4, 0, 1],
                root_length_mm=0.0))
            raise AssertionError("a 0mm root length was accepted")
        except HTTPException as e:
            assert e.status_code == 422
            assert isinstance(e.detail, dict), "refusal was a bare string"
            assert e.detail["errors"], "structured refusal carried no errors"
            assert e.detail["summary"], "structured refusal carried no summary"
        print("PASS  /cut refuses a 0mm root with a structured, renderable payload")
    finally:
        api_core.close_session(sid)


def test_cut_reports_validation_on_success_too():
    """Success is not silence. The cut that worked still says what it measured."""
    sid, cut, v, f = build_cut_session()
    try:
        assert "validation" in cut, "/cut success carries no validation block"
        assert cut["validation"]["ok"] is True
        assert "metrics" in cut["validation"] and cut["validation"]["metrics"]
        assert "crown_advisory" in cut, "cut_guard's metrics are not surfaced"
        adv = cut["crown_advisory"]
        assert "compactness" in adv and "rim_concavity" in adv
        assert adv["thresholds_are"].startswith("unvalidated"), \
            "the advisory must say its thresholds are not validated"
        import json
        json.dumps(cut["validation"])
        print(f"PASS  successful cut reports metrics: compactness {adv['compactness']}, "
              f"volume {adv['volume_mm3']}mm3, advisory_ok={adv['advisory_ok']}")
    finally:
        api_core.close_session(sid)


def test_cut_guard_is_no_longer_bypassed():
    """It used to return ok=True with 'Bypass active' regardless of input."""
    import cut_guard
    v, f, _ = flat_molar_on_base()
    conc = np.zeros(len(v))
    verdict = cut_guard.check_crown(v, f, np.arange(min(50, len(v))), conc)
    assert "Bypass active" not in verdict["diagnosis"], "the hard bypass is still in place"
    assert "advisory_ok" in verdict, "the un-bypassed verdict exposes no advisory"
    # A zero concavity field means the rim check genuinely fails; it must SAY so
    # while still not gating.
    assert verdict["reached_margin"] is False, "a zero concavity rim should fail the rim check"
    assert verdict["ok"] is True, "the advisory must not gate until it is calibrated"
    print("PASS  cut_guard un-bypassed: reports a real verdict, still does not gate")


if __name__ == "__main__":
    test_every_threshold_declares_its_provenance()
    test_a_refusal_carries_its_numbers()
    test_root_length_band_rejects_nonsense_and_accepts_wheeler()
    test_nan_cres_is_caught_because_comparisons_against_nan_are_false()
    test_landmarks_must_be_distinct_and_finite()
    test_selection_warnings_report_what_was_discarded()
    test_not_checked_is_not_clear()
    test_cut_refuses_out_of_range_vertex_ids_instead_of_wrapping()
    test_cut_refuses_an_impossible_root_length_with_structure()
    test_cut_reports_validation_on_success_too()
    test_cut_guard_is_no_longer_bypassed()
    print("\nALL VALIDATION TESTS PASSED")
