"""`manufacturing_v2.aggregate_gate_v3` - the one verdict a solidified stage
ships with. PURE and FAIL-CLOSED, and every family of gates in it has a
control here that makes it FAIL (CLAUDE.md rule 10).

TASK 2 step 4. The gate composes three things that are each measured
elsewhere:

  * the kit's DEFORMATION gates on the planned surface (dc.aggregate_gate_v2
    decides them; this module only carries its verdict) - minus the two that
    measured the raw cast as though it were the file, SUPERSEDED_BY_SOLID;
  * the attachment check - every attachment placed is inside the solid;
  * the SOLID's gates on the re-read bytes (print_solid.solid_gate).

The superseded pair is the load-bearing change, so it has its own test: a
record whose raw-cast measurements are MISSING must still be PRINT READY when
everything the file is judged by passes, and a mutation that made either gate
required again would turn that test red.
"""
from __future__ import annotations

import numpy as np
import pytest

import core_geometry as cg
import deform_construction as dc
import manufacturing_v2 as mfg2
import print_solid as ps
from test_solid_gate import cube, measure


@pytest.fixture(scope="module")
def good_solid():
    return measure(*cube())


def good_kit_record():
    """A planned surface whose deformation passes every kit check - and that
    carries NO raw-cast self-intersection or file measurement at all."""
    return {
        "plan_diagnostics": {"weights_bounded": True},
        "stage_report": {"moving_teeth_exact_rigid": {"45": True},
                         "untouched_vertices_bit_identical": True,
                         "pinned_vertices_bit_identical": True,
                         "inverted_triangles": 0, "degenerate_triangles": 0,
                         "implicit_ipr_mm": 0.0},
        "index_buffer_unchanged": True,
        "prescribed_ipr_mm": 0.0,
    }


def moved(solid, kit=None, att=None):
    return {mfg2.R_STAGE_KIND: mfg2.STAGE_KIND_MOVED, mfg2.R_MOVING_TEETH: 1,
            mfg2.R_DEFORMATION: kit if kit is not None else good_kit_record(),
            mfg2.R_ATTACHMENTS: att if att is not None else {"placed": 0,
                                                             "inside": []},
            mfg2.R_SOLID: solid}


def test_the_vocabulary():
    assert set(mfg2.SUPERSEDED_BY_SOLID) == {"no_self_intersection",
                                             "written_file_topology"}
    assert set(mfg2.DEFORMATION_GATES) == set(dc.REQUIRED_GATES) - \
        set(mfg2.SUPERSEDED_BY_SOLID)
    assert mfg2.STAGE_GATES[-len(ps.SOLID_GATES):] == ps.SOLID_GATES
    assert mfg2.T0_GATES == (mfg2.T0_GATE,) + ps.SOLID_GATES
    print(f"PASS  moved stage: {len(mfg2.STAGE_GATES)} gates; T0: "
          f"{len(mfg2.T0_GATES)}; superseded: {mfg2.SUPERSEDED_BY_SOLID}")


def test_a_good_moved_stage_is_ready_without_raw_cast_measurements(good_solid):
    """The positive control, and the superseded pair's pin."""
    g = mfg2.aggregate_gate_v3(moved(good_solid))
    assert g["print_ready"] is True, g["failed_gates"]
    # The kit itself fails the superseded pair on this record ...
    kit = dc.aggregate_gate_v2(good_kit_record())
    assert set(kit["failed_gates"]) == set(mfg2.SUPERSEDED_BY_SOLID)
    # ... and they are neither required nor reported as gates here.
    for name in mfg2.SUPERSEDED_BY_SOLID:
        assert name not in g["required_gates"] and name not in g["gates"]
    print("PASS  PRINT READY with the raw-cast pair unmeasured; the kit alone "
          f"would have failed {kit['failed_gates']}")


@pytest.mark.parametrize("rec", [{}, None, {mfg2.R_STAGE_KIND: "stage 3"}])
def test_an_empty_or_unknown_record_fails_everything(rec):
    g = mfg2.aggregate_gate_v3(rec)
    assert g["print_ready"] is False
    assert g["failed_gates"][0] == "stage_kind"
    assert set(mfg2.STAGE_GATES) <= set(g["failed_gates"])
    print(f"PASS  {rec!r}: {len(g['failed_gates'])} failures, stage_kind first")


@pytest.mark.parametrize("mutate, gate", [
    (lambda k: k["stage_report"].update(inverted_triangles=1), "no_inverted_triangles"),
    (lambda k: k["stage_report"].update(degenerate_triangles=2), "no_degenerate_triangles"),
    (lambda k: k["stage_report"].update(moving_teeth_exact_rigid={"45": False}),
     "moving_teeth_exact_rigid"),
    (lambda k: k["stage_report"].update(untouched_vertices_bit_identical=False),
     "untouched_vertices_bit_identical"),
    (lambda k: k["stage_report"].update(pinned_vertices_bit_identical=False),
     "pinned_vertices_bit_identical"),
    (lambda k: k["stage_report"].update(implicit_ipr_mm=0.2), "implicit_ipr_within_prescription"),
    (lambda k: k["plan_diagnostics"].update(weights_bounded=False), "weights_bounded"),
    (lambda k: k.update(index_buffer_unchanged=False), "index_buffer_unchanged"),
])
def test_each_deformation_gate_still_refuses(good_solid, mutate, gate):
    kit = good_kit_record()
    mutate(kit)
    g = mfg2.aggregate_gate_v3(moved(good_solid, kit=kit))
    assert g["failed_gates"] == [gate], g["failed_gates"]
    print(f"PASS  {gate} refuses")


def test_a_bad_solid_refuses_a_good_deformation():
    a, fa = cube()
    b, fb = cube(at=(5.0, 5.0, 5.0))
    bad = measure(np.vstack([a, b]), np.vstack([fa, fb + 8]))
    g = mfg2.aggregate_gate_v3(moved(bad))
    assert "solid_no_self_intersection" in g["failed_gates"], g["failed_gates"]
    assert all(n in ps.SOLID_GATES for n in g["failed_gates"])
    print(f"PASS  crossing solid refuses: {g['failed_gates']}")


@pytest.mark.parametrize("att, ok", [
    ({"placed": 0, "inside": []}, True),
    ({"placed": 1, "inside": [True]}, True),
    ({"placed": 1, "inside": [False]}, False),     # dropped as a crumb
    ({"placed": 2, "inside": [True]}, False),      # one unaccounted for
    ({"placed": 1, "inside": None}, False),        # not measured
    ({}, False),
])
def test_the_attachment_gate(good_solid, att, ok):
    rec = moved(good_solid, att=att)
    if att == {}:
        rec.pop(mfg2.R_ATTACHMENTS)
    g = mfg2.aggregate_gate_v3(rec)
    assert (mfg2.ATTACHMENT_GATE not in g["failed_gates"]) is ok, g["gates"][mfg2.ATTACHMENT_GATE]
    print(f"PASS  attachments {att} -> {'ok' if ok else 'FAIL'}")


def test_t0_must_carry_no_movement(good_solid):
    rec = {mfg2.R_STAGE_KIND: mfg2.STAGE_KIND_T0, mfg2.R_MOVING_TEETH: 0,
           mfg2.R_SOLID: good_solid}
    assert mfg2.aggregate_gate_v3(rec)["print_ready"] is True
    for n in (1, None, False):
        g = mfg2.aggregate_gate_v3({**rec, mfg2.R_MOVING_TEETH: n})
        assert g["failed_gates"] == [mfg2.T0_GATE], (n, g["failed_gates"])
    print("PASS  T0 ready at 0 moving teeth; 1 / None / False refuse by name")


def test_describe_failures_names_the_gate_and_the_number(good_solid):
    kit = good_kit_record()
    kit["stage_report"]["implicit_ipr_mm"] = 0.14
    g = mfg2.aggregate_gate_v3(moved(good_solid, kit=kit))
    lines = mfg2.describe_failures(g)
    assert len(lines) == 1 and lines[0].startswith("implicit_ipr_within_prescription")
    assert "0.14" in lines[0], lines[0]
    print(f"PASS  {lines[0][:110]}")


# ---------------------------------------------------------------------------
# Producer / consumer: the verdict shipped IS the gate's answer on the record
# ---------------------------------------------------------------------------

def test_the_builders_write_every_key_the_gate_reads(monkeypatch):
    """Built for real (at the plumbing voxel, 0.1 mm), then re-gated."""
    monkeypatch.setattr(ps, "VOXEL_SIZE_MM", 0.1)
    from test_cast_base import frame_for, horseshoe_shell
    v, f, apices = horseshoe_shell(n_s=240, n_t=60, teeth=(-0.25, 0.0, 0.25),
                                   return_apices=True)
    af = frame_for(v)
    near = [np.flatnonzero(np.linalg.norm(v - v[a], axis=1) < 3.0) for a in apices]
    teeth = {f"t{i}": ids for i, ids in enumerate(near)}

    plan0 = mfg2.build_case_plan(v, f, af, {}, np.concatenate(near),
                                 static_by_tooth=teeth)
    out0 = mfg2.build_t0_stage(plan0)

    u_md = np.asarray(af["u_tra"], float)
    frame = {"u_md": u_md, "u_bl": np.cross(af["u_occ"], u_md),
             "u_oa": np.asarray(af["u_occ"], float),
             "centroid": v[near[1]].mean(axis=0)}
    c_res = v[near[1]].mean(axis=0) - 10.0 * np.asarray(af["u_occ"], float)
    moving = {"t1": {"vertices": near[1], "frame": frame, "c_res": c_res,
                     "clinical": {"d_oa": 0.15}}}
    plan1 = mfg2.build_case_plan(v, f, af, moving,
                                 np.concatenate([near[0], near[2]]),
                                 static_by_tooth={"t0": near[0], "t2": near[2]})
    mats = mfg2.stage_matrices_for(plan1, 1, 1)
    out1 = mfg2.build_stage_v2(plan1, mats)

    for out in (out0, out1):
        rec = out["record"]
        for key in (mfg2.R_STAGE_KIND, mfg2.R_MOVING_TEETH, mfg2.R_SOLID):
            assert key in rec, key
        assert mfg2.aggregate_gate_v3(rec) == out["gate"]
    for key in (mfg2.R_DEFORMATION, mfg2.R_ATTACHMENTS):
        assert key in out1["record"], key
    assert out0["gate"]["stage_kind"] == mfg2.STAGE_KIND_T0
    assert out1["gate"]["stage_kind"] == mfg2.STAGE_KIND_MOVED
    print(f"PASS  T0 {out0['gate']['verdict']} / moved {out1['gate']['verdict']}"
          f" {out1['gate']['failed_gates']}; re-gating each record reproduces "
          f"its shipped verdict exactly")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
