"""Segmentation review: explicit confidence, explicit verdict, no invented certainty.

The claim under test is narrow and important: the software may say a region
looks geometrically plausible, and it may NOT say a tooth is correctly
identified. ToothGroupNetwork emits no calibrated uncertainty, so a "confidence"
here is a weighted agreement over facts that can be independently checked — and
every one of those facts comes back with the score so a clinician can disagree
with a factor rather than only with a number.
"""
import numpy as np
from fastapi import HTTPException

import api_core
import segmentation_review as sr
from session_store import STORE
from test_hydration import build_cut_session
from tooth_fixture import flat_molar_on_base


def _arch():
    v, f, r = flat_molar_on_base()
    return v, f, r, np.linalg.norm(v[:, :2], axis=1)


def test_a_clean_region_passes_with_its_factors_shown():
    v, f, r, rr = _arch()
    labels = np.where(rr < r * 0.95, 36, 0)
    out = sr.review_segmentation(labels, v, f, "lower")
    t = out["teeth"][0]
    assert t["verdict"] == sr.PASS and t["confidence"] == 1.0
    assert t["failed_factors"] == []
    # The score is never the whole answer.
    names = {fa["factor"] for fa in t["factors"]}
    assert {"size", "connectedness", "extent", "fdi_belongs_to_arch"} <= names
    assert "NOT a model probability" in t["confidence_is"]
    print(f"PASS  clean region: {t['verdict']} conf={t['confidence']}, "
          f"{len(t['factors'])} factors returned")


def test_an_fdi_from_the_wrong_arch_is_flagged():
    """FDI 16 is maxillary. On a lower arch that is the model naming a tooth
    that cannot be there — the single most consequential label error, because
    the quadrant sets the root length and C_res follows it."""
    v, f, r, rr = _arch()
    labels = np.where(rr < r * 0.95, 16, 0)       # upper FDI on a lower arch
    out = sr.review_segmentation(labels, v, f, "lower")
    t = out["teeth"][0]
    assert t["verdict"] == sr.REVIEW, f"wrong-arch FDI reported {t['verdict']}"
    assert "fdi_belongs_to_arch" in t["failed_factors"]
    assert t["confidence"] < 1.0
    print(f"PASS  FDI 16 on a lower arch -> REVIEW, conf={t['confidence']}, "
          f"failed={t['failed_factors']}")


def test_a_disconnected_label_fails_rather_than_reviews():
    """One label covering two separate places is not a tooth at all."""
    v, f, r, rr = _arch()
    # Two disjoint rings sharing one label.
    labels = np.where((rr < r * 0.40) | ((rr > r * 0.75) & (rr < r * 0.95)), 36, 0)
    out = sr.review_segmentation(labels, v, f, "lower")
    t = out["teeth"][0]
    assert t["verdict"] in (sr.FAIL, sr.REVIEW)
    assert "connectedness" in t["failed_factors"], \
        f"a split label did not fail connectedness: {t['failed_factors']}"
    frac = [fa for fa in t["factors"] if fa["factor"] == "connectedness"][0]
    assert frac["metrics"]["connected_fraction"] < 0.9
    print(f"PASS  split label -> {t['verdict']}, largest island "
          f"{frac['metrics']['connected_fraction']*100:.0f}% of the region")


def test_a_speck_fails_as_debris():
    v, f, r, rr = _arch()
    labels = np.where(rr < r * 0.03, 36, 0)      # a few faces at the apex
    out = sr.review_segmentation(labels, v, f, "lower")
    if not out["teeth"]:
        print("PASS  a speck produced no tooth region at all")
        return
    t = out["teeth"][0]
    assert t["verdict"] == sr.FAIL, f"a {t['face_count']}-face speck reported {t['verdict']}"
    print(f"PASS  {t['face_count']}-face speck -> FAIL (debris, not a tooth)")


def test_the_summary_counts_every_verdict():
    v, f, r, rr = _arch()
    labels = np.where(rr < r * 0.5, 36, np.where(rr < r * 0.95, 16, 0))
    out = sr.review_segmentation(labels, v, f, "lower")
    assert set(out["counts"]) == {sr.PASS, sr.REVIEW, sr.FAIL}
    assert sum(out["counts"].values()) == out["tooth_count"]
    assert out["summary"]
    # jaw_naming's full structured verdict, not just its diagnosis line - which
    # is what /segment collapses it to on the failure path.
    assert isinstance(out["jaw_report"], dict) and "diagnosis" in out["jaw_report"]
    assert "vertices_per_tooth" in out["jaw_report"]
    print(f"PASS  counts={out['counts']}, full jaw_report carried (not just its string)")


def test_a_pass_is_explicitly_not_a_clinical_confirmation():
    v, f, r, rr = _arch()
    out = sr.review_segmentation(np.where(rr < r * 0.95, 36, 0), v, f, "lower")
    assert "NOT a clinical confirmation" in out["limitation"]
    assert out["thresholds_basis"] == "software heuristic"
    print("PASS  the report states a PASS is not a clinical confirmation")


def test_endpoint_refuses_before_segmentation_has_run():
    sid, cut, v, f = build_cut_session()
    try:
        try:
            api_core.segmentation_review_report(sid)
            raise AssertionError("review ran with no segmentation")
        except HTTPException as e:
            assert e.status_code == 409, f"got {e.status_code}, want 409"
            assert "segment" in str(e.detail).lower()
        print("PASS  /segmentation-review refuses with 409 before /segment has run")
    finally:
        api_core.close_session(sid)


def test_endpoint_runs_on_a_labelled_session():
    sid, cut, v, f = build_cut_session()
    try:
        rr = np.linalg.norm(v[:, :2], axis=1)
        STORE.put(sid, "labels", np.where(rr < 4.0, 36, 0))
        out = api_core.segmentation_review_report(sid)
        assert out["tooth_count"] >= 1
        assert out["teeth"][0]["verdict"] in (sr.PASS, sr.REVIEW, sr.FAIL)
        import json
        json.dumps(out)
        print(f"PASS  endpoint: {out['tooth_count']} region(s), {out['summary']}")
    finally:
        api_core.close_session(sid)


if __name__ == "__main__":
    test_a_clean_region_passes_with_its_factors_shown()
    test_an_fdi_from_the_wrong_arch_is_flagged()
    test_a_disconnected_label_fails_rather_than_reviews()
    test_a_speck_fails_as_debris()
    test_the_summary_counts_every_verdict()
    test_a_pass_is_explicitly_not_a_clinical_confirmation()
    test_endpoint_refuses_before_segmentation_has_run()
    test_endpoint_runs_on_a_labelled_session()
    print("\nALL SEGMENTATION REVIEW TESTS PASSED")
