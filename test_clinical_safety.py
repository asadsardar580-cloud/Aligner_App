"""Biomechanical guardrails and the clinical report.

The assertion that matters most here is not that a threshold fires. It is that
GREEN never claims biological safety. This software sees millimetres and
degrees; root resorption risk is driven by force, duration, root morphology,
patient biology and appliance fit, none of which it can observe. A green chip a
clinician reads as a safety guarantee is worse than no chip at all.
"""
import json
import os
import shutil
import tempfile

import clinical_safety as cs
import domain
import export_clinical_report as ecr


def _rx(**kw):
    base = dict(tip_deg=0.0, torque_deg=0.0, rotation_deg=0.0,
                d_md=0.0, d_bl=0.0, d_oa=0.0)
    base.update(kw)
    return base


def test_green_explicitly_does_not_claim_biological_safety():
    out = cs.assess_case([{"tooth_id": "t1", "fdi": 33,
                           "prescription": _rx(tip_deg=2.0)}], stages=4)
    assert out["overall"] == cs.GREEN
    assert "NOT a statement that the movement is biologically safe" in out["meaning"][cs.GREEN]
    assert "none of which this software can see" in out["limitation"]
    print("PASS  GREEN states in words that it is a planning envelope, not safety")


def test_rate_limits_fire_per_stage_not_per_case():
    """A 10-degree tip is fine over 10 stages and impossible over 2."""
    fine = cs.assess_tooth(_rx(tip_deg=10.0), stages=10, fdi=33)
    assert fine["status"] in (cs.GREEN, cs.YELLOW), f"10deg/10 stages -> {fine['status']}"
    assert fine["rates"]["max_rotation_deg_per_stage"] == 1.0

    too_fast = cs.assess_tooth(_rx(tip_deg=10.0), stages=2, fdi=33)
    assert too_fast["status"] == cs.RED
    codes = [f["code"] for f in too_fast["findings"]]
    assert "rotation_rate" in codes, codes
    assert too_fast["rates"]["max_rotation_deg_per_stage"] == 5.0
    print("PASS  10deg over 10 stages ok (1.0/stage); over 2 stages RED (5.0/stage)")


def test_translation_rate_ceiling_is_a_quarter_millimetre():
    ok = cs.assess_tooth(_rx(d_md=2.0), stages=8, fdi=34)      # 0.25/stage exactly
    assert ok["rates"]["max_translation_mm_per_stage"] == 0.25
    assert "translation_rate" not in [f["code"] for f in ok["findings"]]

    bad = cs.assess_tooth(_rx(d_md=2.0), stages=4, fdi=34)     # 0.5/stage
    assert bad["status"] == cs.RED
    assert "translation_rate" in [f["code"] for f in bad["findings"]]
    print("PASS  0.25mm/stage accepted at the boundary, 0.5mm/stage RED")


def test_extrusion_and_intrusion_have_different_limits_and_correct_signs():
    """d_oa positive is occlusal (extrusion), negative apical (intrusion). The
    two limits differ by a millimetre, so a sign error swaps them."""
    # 2.5mm extrusion is over the 2.0 limit.
    ex = cs.assess_tooth(_rx(d_oa=2.5), stages=40, fdi=11)
    assert ex["totals"]["extrusion_mm"] == 2.5 and ex["totals"]["intrusion_mm"] == 0.0
    assert ex["status"] == cs.RED
    assert "total_extrusion" in [f["code"] for f in ex["findings"]]

    # 2.5mm intrusion is UNDER the 3.0 limit — the same magnitude, opposite sign,
    # a different verdict. This is what a sign error would destroy.
    intr = cs.assess_tooth(_rx(d_oa=-2.5), stages=40, fdi=11)
    assert intr["totals"]["intrusion_mm"] == 2.5 and intr["totals"]["extrusion_mm"] == 0.0
    assert "total_intrusion" not in [f["code"] for f in intr["findings"]]

    deep = cs.assess_tooth(_rx(d_oa=-3.5), stages=40, fdi=11)
    assert "total_intrusion" in [f["code"] for f in deep["findings"]]
    print("PASS  +2.5mm extrusion RED, -2.5mm intrusion allowed, -3.5mm flagged "
          "— signs and the two limits are distinct")


def test_root_resorption_thresholds_are_literature_labelled():
    ex = cs.assess_tooth(_rx(d_oa=2.5), stages=40, fdi=11)
    f = [x for x in ex["findings"] if x["code"] == "total_extrusion"][0]
    assert f["threshold"]["basis"] == "literature/reference"
    assert "resorption" in f["threshold"]["note"]
    # And a rate limit is a heuristic, not literature — they must not be conflated.
    assert cs.RATE_LIMITS["translation_mm"]["basis"] == "software heuristic"
    print("PASS  resorption limits labelled literature, rate limits labelled heuristic")


def test_yellow_means_needs_auxiliaries_not_danger():
    y = cs.assess_tooth(_rx(rotation_deg=20.0), stages=20, fdi=13)
    assert y["status"] == cs.YELLOW
    codes = [f["code"] for f in y["findings"]]
    assert any(c.startswith("auxiliary") for c in codes), codes
    out = cs.assess_case([{"tooth_id": "t", "fdi": 13,
                           "prescription": _rx(rotation_deg=20.0)}], 20)
    assert "attachments or IPR" in out["meaning"][cs.YELLOW]
    print("PASS  20deg rotation -> YELLOW, described as needing auxiliaries")


def test_case_rollup_takes_the_worst_tooth():
    teeth = [{"tooth_id": "a", "fdi": 31, "prescription": _rx(tip_deg=1.0)},
             {"tooth_id": "b", "fdi": 36, "prescription": _rx(d_oa=2.5)}]
    out = cs.assess_case(teeth, stages=40)
    assert out["overall"] == cs.RED, "one RED tooth did not make the case RED"
    assert out["counts"][cs.RED] == 1 and out["counts"][cs.GREEN] == 1
    assert "re-plan or accept knowingly" in out["summary"]
    print(f"PASS  case rollup: {out['counts']} -> {out['overall']}")


# ------------------------------------------------------------- report ---

def _case():
    c = domain.Case(label="mid-course correction")
    c.arches["lower"] = domain.Arch(arch="lower", session_id="LO")
    c.arches["lower"].teeth["t1"] = domain.Tooth(
        tooth_id="t1", arch="lower", fdi=33, root_length_mm=13.0, c_res=[1, 2, 3],
        prescription=domain.Prescription(tip_deg=6.5, d_md=0.4), review="confirmed")
    c.arches["lower"].teeth["t2"] = domain.Tooth(
        tooth_id="t2", arch="lower", fdi=36, root_length_mm=10.0,
        prescription=domain.Prescription(rotation_deg=20.0))
    return c


def test_report_carries_the_disclaimer_verbatim():
    r = ecr.build(_case())
    assert r["disclaimer"] == (
        "Generated by Project Aligner Workstation. For professional orthodontic "
        "clinician review only. Requires clinical verification prior to thermoforming.")
    print("PASS  the medical disclaimer is present verbatim")


def test_report_leads_with_what_is_outstanding():
    r = ecr.build(_case())
    o = r["outstanding"]
    # t2 is unreviewed and its 10mm root contradicts FDI 36 (a molar: Wheeler 9).
    assert 36 in o["teeth_unreviewed"], o["teeth_unreviewed"]
    assert 36 in o["root_length_mismatches"], o["root_length_mismatches"]
    # One arch only, so the antagonist check could not have run.
    assert o["antagonist_checked"] is False
    print(f"PASS  outstanding: unreviewed {o['teeth_unreviewed']}, "
          f"root mismatch {o['root_length_mismatches']}, antagonist not checked")


def test_report_contains_no_patient_identifier():
    r = ecr.build(_case())
    blob = json.dumps(r).lower()
    for banned in ("patient_name", "date_of_birth", "\"dob\"", "mrn", "surname"):
        assert banned not in blob, f"{banned} reached the clinical report"
    assert r["case_reference"] == _case().case_id or len(r["case_reference"]) >= 8
    assert "No patient identifier" in r["phi"]
    print("PASS  no identifier fields in the report; case referenced by opaque id")


def test_ipr_rows_say_they_are_measurements_not_prescriptions():
    space = {"interproximal": {"contacts": [
        {"pair": "31-41", "fdi": [31, 41], "gap_mm": 0.08,
         "state": "CONTACT", "detail": "in clinical contact"}]}}
    r = ecr.build(_case(), space_analysis=space)
    assert len(r["ipr_table"]) == 1
    row = r["ipr_table"][0]
    assert row["reduction_mm"] is None, "the software must not prescribe a reduction"
    assert "NOT A PRESCRIPTION" in row["caution"]
    print(f"PASS  IPR row {row['contact']} at {row['measured_gap_mm']}mm carries no "
          f"prescribed reduction")


def test_both_output_formats_write_and_agree():
    tmp = tempfile.mkdtemp(prefix="report_")
    try:
        r = ecr.build(_case())
        jp = ecr.write_json(r, os.path.join(tmp, "p.json"))
        hp = ecr.write_html(r, os.path.join(tmp, "p.html"))
        assert os.path.getsize(jp) > 500 and os.path.getsize(hp) > 800

        back = json.load(open(jp, encoding="utf-8"))
        assert back["stage_count"] == r["stage_count"]
        assert back["tooth_count"] == r["tooth_count"] == 2

        page = open(hp, encoding="utf-8").read()
        assert ecr.DISCLAIMER[:40] in page, "the HTML lost the disclaimer"
        assert "FDI 33" in page and "FDI 36" in page
        assert "&lt;" not in r["case_reference"], "case id needs no escaping"
        # The safety chip must be visible in the printable output.
        assert any(s in page for s in ("GREEN", "YELLOW", "RED"))
        print(f"PASS  json {os.path.getsize(jp)}B and html {os.path.getsize(hp)}B agree, "
              f"disclaimer and per-tooth rows present in both")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_green_explicitly_does_not_claim_biological_safety()
    test_rate_limits_fire_per_stage_not_per_case()
    test_translation_rate_ceiling_is_a_quarter_millimetre()
    test_extrusion_and_intrusion_have_different_limits_and_correct_signs()
    test_root_resorption_thresholds_are_literature_labelled()
    test_yellow_means_needs_auxiliaries_not_danger()
    test_case_rollup_takes_the_worst_tooth()
    test_report_carries_the_disclaimer_verbatim()
    test_report_leads_with_what_is_outstanding()
    test_report_contains_no_patient_identifier()
    test_ipr_rows_say_they_are_measurements_not_prescriptions()
    test_both_output_formats_write_and_agree()
    print("\nALL CLINICAL SAFETY TESTS PASSED")
