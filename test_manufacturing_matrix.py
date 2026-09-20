"""End-to-end manufacturing matrix: every stage of every movement must print.

WHY THIS IS SEPARATE FROM test_manufacturing_interface.py. That file tests the
INTERFACE - one tooth, one matrix, the tools it produces. This one runs the
whole export: `build_stage_bundle` builds the cast, fuses every crown, writes
real STL bytes, rereads them, welds them as a downstream reader would and
applies the hard gates. An interface that measures perfectly and a stage that
fuses into three bodies are entirely compatible, and this project has shipped
exactly that combination.

THE BAR, in the brief's own words: not "at least one stage is printable" but
"every stage that is claimed to be print-ready passes its complete
manufacturing gates". So these tests assert per STAGE, never per case.
"""
from __future__ import annotations

import numpy as np
import pytest

import api_core
import core_geometry as cg
import manufacturing as mfg
from api_core import build_stage_bundle, StageExportRequest
from test_staging_export import moved_session


def _stages(prescriptions, n):
    """Run the real exporter and return (manifest stage records, session id)."""
    sid, tids, _ = moved_session(prescriptions=prescriptions)
    try:
        bundle = build_stage_bundle(sid, StageExportRequest(stages=n))
        return bundle["manifest"]["stage_files"], bundle
    finally:
        api_core.close_session(sid)


def _assert_every_stage_prints(label, stages):
    bad = []
    for s in stages:
        v = s["stl_validation"]
        m = s.get("manifold_status", {})
        ok = (v["print_ready"] and m.get("single_positive_body")
              and s["manifold_bodies"] == 1 and v["connected_components"] == 1
              and v["open_edges"] == 0 and v["nonmanifold_edges"] == 0)
        if not ok:
            bad.append(f"stage {s['stage']}: failed={v['failed_gates']} "
                       f"bodies={s['manifold_bodies']} "
                       f"comp={v['connected_components']} "
                       f"open={v['open_edges']} NM={v['nonmanifold_edges']} "
                       f"manifold={m.get('status')} "
                       f"single_positive={m.get('single_positive_body')}")
    assert not bad, f"{label}:\n  " + "\n  ".join(bad)
    print(f"PASS  {label}: {len(stages)}/{len(stages)} stages print-ready, "
          f"volumes "
          + ", ".join(f"{s['volume_mm3']:.1f}" for s in stages))


# ===========================================================================
# The extrusion sweep. Extrusion is the movement that broke the old exporter:
# the root plug travelled with the tooth and emerged from the gingiva, and the
# replacement bridged to tissue that is progressively further away.
# ===========================================================================

EXTRUSIONS = [0.0, 0.1, 0.25, 0.5, 1.0, 1.2, 2.0]


@pytest.mark.parametrize("mm", EXTRUSIONS)
def test_extrusion_sweep_every_stage_is_one_printable_body(mm):
    """0 through 2.0 mm. ZERO IS IN THE SWEEP DELIBERATELY - a tooth that has
    not moved must still fuse, and an interface that only works once something
    has moved is one that cannot be trusted at stage 1 of every plan."""
    stages, _ = _stages([dict(d_oa=mm), dict(d_md=0.6)], 3)
    # NOT asserted to be exactly 3. `stages` is a request, and the planner
    # adds stages when a movement would exceed the per-stage rate limits - so
    # a bigger prescription legitimately comes back with more. Asserting the
    # count here made the sweep fail on arithmetic rather than on geometry.
    assert len(stages) >= 1
    _assert_every_stage_prints(f"extrusion {mm}mm", stages)


def test_the_primary_regression_1_2mm_over_five_stages():
    """The brief's named regression: 1.2mm, 5 stages. It used to read
    `1,1,2,2,3` bodies - only the first stage fused - and the failure grew
    with the movement, which is the signature of the interface losing contact
    as the tooth lifts."""
    stages, _ = _stages([dict(d_oa=1.2), dict(d_md=0.6)], 5)
    assert len(stages) == 5
    _assert_every_stage_prints("extrusion 1.2mm x 5 stages", stages)
    bodies = [s["manifold_bodies"] for s in stages]
    assert bodies == [1, 1, 1, 1, 1], f"body counts per stage: {bodies}"
    # The old exporter read 1,1,2,2,3 here: the failure GREW with the
    # movement, which is what a reconstruction losing contact looks like.
    assert len(set(bodies)) == 1


# ===========================================================================
# MIXED TIPPING WITH PARTIAL RIM SEPARATION - the brief calls this the
# important missing case, and it is: it is the only one where the interface
# has to be a bridge and a seat AT THE SAME TIME, around one rim, with no
# seam between them.
# ===========================================================================

PARTIAL = [
    ("tip 8 deg + 0.5mm extrusion", dict(tip_deg=8.0, d_oa=0.5)),
    ("torque 8 deg + 0.4mm extrusion", dict(torque_deg=8.0, d_oa=0.4)),
    ("tip 6 deg", dict(tip_deg=6.0)),
    ("torque 10 deg", dict(torque_deg=10.0)),
]


@pytest.mark.parametrize("label,clinical", PARTIAL)
def test_mixed_tipping_with_partial_rim_separation(label, clinical):
    """Part of the rim lifts clear while the rest stays in the tissue.

    Asserted in two places, because either alone is satisfiable by a wrong
    implementation: the INTERFACE must report both lifted and seated points
    (otherwise the fixture is not producing the case at all, and a test that
    silently stops exercising its own case is worse than no test), and every
    fused STAGE must still be one printable body.
    """
    sid, tids, _ = moved_session(prescriptions=[clinical, dict(d_md=0.4)])
    try:
        v = api_core.STORE.require(sid, "verts")
        f = api_core.STORE.require(sid, "faces")
        af = api_core.STORE.get(sid, "arch_frame")
        ex = api_core.STORE.get(sid, "extracted_faces")
        sv, sf, _ = api_core._seal_sockets(v, f, sid, ex, flush=True)
        curve, _ = cg.fit_arch_curve(v, af)
        tv, tf, ti = cg.trim_to_arch(sv, sf, af, margin_mm=7.0, curve=curve)
        bv, bf, _ = cg.build_cast_base(tv, tf, af, base_thickness_mm=3.0,
                                       rim=ti["rim_loop"])
        rec = api_core.STORE.get(sid, f"tooth:{tids[0]}")
        probe = mfg.CastProbe(bv, bf)
        M = cg.kinematic_matrix(rec["frame"], rec["c_res"], **clinical)
        rim0 = v[np.asarray(rec["socket_rim"], np.int64)]
        r = mfg.build_stage_tooth_interface(
            bv, bf, rec["cv"], rec["cf"], rim0, rec["frame"]["u_oa"], M, probe=probe)
        assert r.ok, f"{label} was refused: {r.refusal_reason} / {r.diagnostics}"
        d = r.diagnostics
        assert d["interface_mode"] == "mixed", d["interface_mode"]
        assert d["rim_points_lifted"] > 0, (
            f"{label} produced no lifted rim points - the fixture is no longer "
            f"exercising partial separation: {d}")
        assert d["rim_points_seated"] > 0, (
            f"{label} lifted the WHOLE rim - that is plain separation, not the "
            f"mixed case this test exists for: {d}")
        assert d["connector_volume_mm3"] > 0, d
        print(f"      {label}: lifted/seated "
              f"{d['rim_points_lifted']}/{d['rim_points_seated']}, "
              f"penetration {d['crown_penetration_mm']:.3f}mm, "
              f"separation {d['rim_separation_max_mm']:.3f}mm, "
              f"connector {d['connector_volume_mm3']:.1f}mm3")
        bundle = build_stage_bundle(sid, StageExportRequest(stages=3))
        _assert_every_stage_prints(label, bundle["manifest"]["stage_files"])
    finally:
        api_core.close_session(sid)


# ===========================================================================
# Order independence
# ===========================================================================

def test_the_fused_result_does_not_depend_on_boolean_order():
    """The SAME plan, fused in three different orders, must agree AS GEOMETRY.

    WHAT THIS USED TO TEST, AND WHY IT WAS THE WRONG THING. The old version
    reversed the PRESCRIPTION LIST - `[d_oa=1.0, d_md=0.8]` against
    `[d_md=0.8, d_oa=1.0]` - which does not reverse an order, it gives the two
    teeth each other's movement. That is a different model, and a different
    model is entitled to a different surface: measured, the two "orders"
    differ by 0.580mm, while their VOLUMES agree to 5e-4 because the total
    material is much the same. A volume comparison could never have caught it,
    which is the point of comparing surfaces.

    So this builds ONE session and hands the identical solids to the boolean
    in three sequences - batch forward, batch reversed, and strictly
    sequential - and compares the written STLs by two-sided
    point-to-TRIANGLE distance, topology, body count and volume.
    """
    import zipfile
    import stl_io
    presc = [dict(d_oa=1.0), dict(d_md=0.8)]
    sid, tids, _ = moved_session(prescriptions=presc)
    try:
        runs = {}
        for order in ("forward", "reverse", "sequential"):
            bundle = build_stage_bundle(sid, StageExportRequest(stages=2),
                                        part_order=order)
            runs[order] = (bundle["manifest"]["stage_files"],
                           zipfile.ZipFile(bundle["buf"]))
        base_stages, base_zip = runs["forward"]
        assert len(base_stages) >= 1
        worst = 0.0
        for order in ("reverse", "sequential"):
            other, ozip = runs[order]
            assert len(other) == len(base_stages), (
                f"{order} produced {len(other)} stages against "
                f"{len(base_stages)} forward")
            for sa, sb in zip(base_stages, other):
                va, vb = sa["volume_mm3"], sb["volume_mm3"]
                assert abs(va - vb) / max(va, 1.0) < 5e-4, (
                    f"{order} stage {sa['stage']}: {va:.3f} vs {vb:.3f}")
                assert sa["manifold_bodies"] == sb["manifold_bodies"] == 1
                assert (sa["stl_validation"]["connected_components"]
                        == sb["stl_validation"]["connected_components"])
                assert (sa["stl_validation"]["open_edges"]
                        == sb["stl_validation"]["open_edges"] == 0)
                av, af = stl_io.parse_stl_bytes(base_zip.read(sa["file"]))
                bv, bf = stl_io.parse_stl_bytes(ozip.read(sb["file"]))
                d1 = float(mfg._point_to_surface(av, bv, bf).max())
                d2 = float(mfg._point_to_surface(bv, av, af).max())
                worst = max(worst, d1, d2)
                # 0.01mm sits well inside the scanner's own 20-50 micron
                # resolution, so anything under it is not a difference anyone
                # could have measured on the cast this was built from.
                assert max(d1, d2) <= 0.01, (
                    f"{order} stage {sa['stage']}: surfaces differ by "
                    f"{max(d1, d2):.5f}mm ({d1:.5f} / {d2:.5f})")
                assert (sa["roi_compliance"]["modified_points_outside_envelope"]
                        == sb["roi_compliance"]["modified_points_outside_envelope"])
        print(f"PASS  boolean-order independence over {len(base_stages)} "
              f"stages x 2 orders: worst two-sided surface difference "
              f"{worst:.6f}mm")
    finally:
        api_core.close_session(sid)


# ===========================================================================
# The other movement axes, end to end rather than interface-only
# ===========================================================================

AXES = [
    ("intrusion", dict(d_oa=-1.0)),
    ("buccolingual", dict(d_bl=1.0)),
    ("mesiodistal", dict(d_md=1.0)),
    ("rotation", dict(rotation_deg=8.0)),
    ("combined", dict(tip_deg=4.0, torque_deg=-3.0, rotation_deg=5.0,
                      d_md=0.5, d_bl=0.3, d_oa=0.6)),
]


@pytest.mark.parametrize("label,clinical", AXES)
def test_every_movement_axis_fuses_end_to_end(label, clinical):
    stages, _ = _stages([clinical, dict(d_md=0.4)], 3)
    _assert_every_stage_prints(label, stages)



# ===========================================================================
# ADJACENT, CROWDED AND CONVERGING TEETH
#
# The gingival bridge between two reconstructions is the thing that can
# silently disappear: two collars that meet consume the tissue between them,
# and the fused solid stays perfectly watertight while the cast grows a trench.
# The old guard compared the bridge with `2 * fusion_overlap_mm` - a nominal
# number describing no geometry that was ever built. These cases put the teeth
# close enough for the question to be real and check the MEASURED answer.
# ===========================================================================

ADJACENT = [
    # label,            arch positions,   prescriptions
    ("adjacent 6.5mm apart", (-0.14, 0.14),
     [dict(d_oa=0.4), dict(d_oa=0.4)]),
    ("crowded 2.2mm apart", (-0.10, 0.10),
     [dict(d_oa=0.3), dict(d_oa=0.3)]),
    ("moving toward one another", (-0.14, 0.14),
     [dict(d_md=0.8), dict(d_md=-0.8)]),
    ("overlapping cervical rims", (-0.10, 0.10),
     [dict(d_md=0.6, d_oa=0.3), dict(d_md=-0.6, d_oa=0.3)]),
]


@pytest.mark.parametrize("label,positions,clinical", ADJACENT)
def test_adjacent_reconstructions_keep_a_measured_gingival_bridge(
        label, positions, clinical):
    """Two close teeth, moved, must not consume the tissue between them.

    Asserted in three places, because each alone is satisfiable by a wrong
    implementation:

      * the FIXTURE must actually be adjacent - a test that quietly stops
        exercising its own case is worse than no test, so the measured bridge
        is asserted to be small;
      * every stage must still be one printable body;
      * the bridge measurement must come from the RECONSTRUCTION GEOMETRY.
        `2 * fusion_overlap_mm` is 0.5mm whatever was built, so a check
        against it passes for a trench of any width.
    """
    sid, tids, _ = moved_session(teeth=positions, prescriptions=clinical)
    try:
        v = api_core.STORE.require(sid, "verts")
        rims = [v[np.asarray(api_core.STORE.get(sid, f"tooth:{t}")["socket_rim"],
                             np.int64)] for t in tids]
        bridge0 = mfg.bridge_between(rims[0], rims[1])
        assert bridge0 < 12.0, (
            f"{label}: the fixture's teeth are {bridge0:.2f}mm apart, which is "
            f"not adjacent - this case has stopped exercising itself")
        bundle = build_stage_bundle(sid, StageExportRequest(stages=3))
        stages = bundle["manifest"]["stage_files"]
        _assert_every_stage_prints(label, stages)
        for s in stages:
            pairs = s["adjacent_bridges"]
            assert pairs, f"{label} stage {s['stage']}: no bridge was measured"
            for b in pairs:
                # THE BRIDGE MUST NOT DISAPPEAR. A merged pair is refused
                # upstream; what is asserted here is that real tissue is left.
                assert b["clear_gap_between_reconstructions_mm"] > 0.0, (
                    f"{label} stage {s['stage']}: the gingival bridge is gone: {b}")
                # The measurement must be geometry, not the nominal constant.
                assert b["measure"].startswith("minimum distance between"), b
                # AND IT CANNOT PASS SILENTLY. Over the policy fraction the
                # stage must be NOT PRINT READY, with this gate named - the
                # threshold has no measurement behind it, so it gates rather
                # than refuses, and the one thing that must never happen is
                # for it to be consumed quietly.
                if not b["ok"]:
                    assert s["print_ready"] is False, b
                    assert "gingival_bridge_preserved" in                         s["manufacturing_gate"]["failed_gates"], b
        worst = min(b["clear_gap_between_reconstructions_mm"]
                    for s in stages for b in s["adjacent_bridges"]
                    if "clear_gap_between_reconstructions_mm" in b)
        print(f"      {label}: T0 bridge {bridge0:.3f}mm, narrowest clear gap "
              f"between reconstructions {worst:.3f}mm over {len(stages)} stages")
    finally:
        api_core.close_session(sid)


# ===========================================================================
# THE AGGREGATE MANUFACTURING GATE
# ===========================================================================

def test_the_aggregate_gate_is_what_decides_print_ready():
    """`print_ready` on a stage is the aggregate gate's answer and nothing else.

    The narrower boolean/topology verdict is kept beside it under its own
    name, so the two can never be confused again: a stage that passes the
    written-STL checks is NOT thereby ready to manufacture.
    """
    stages, _ = _stages([dict(d_oa=0.6), dict(d_md=0.4)], 3)
    for s in stages:
        gate = s["manufacturing_gate"]
        assert set(gate) >= {"print_ready", "failed_gates", "gates", "verdict"}
        assert s["print_ready"] is gate["print_ready"]
        assert s["verdict"] == gate["verdict"]
        assert "passes_boolean_topology_regression" in s
        names = {g["gate"] for g in gate["gates"]}
        # Every item the brief requires, present by name. A gate that is not
        # in this set cannot have been evaluated.
        for required in ("written_stl_topology", "single_positive_manifold_body",
                         "no_self_touching_boundary",
                         "synthetic_exposure_within_bound",
                         "no_exposed_clearance_wall",
                         "unaffected_cast_fidelity_two_sided",
                         "reconstruction_inside_envelope",
                         "interface_continuous_around_every_rim",
                         "no_transition_ledge", "old_site_restored",
                         "crown_is_an_exact_rigid_transform",
                         "gingival_bridge_preserved", "root_length_independent",
                         "clinical_consistency"):
            assert required in names, f"{required} is not in the gate"
        if gate["print_ready"]:
            assert gate["verdict"] == "PRINT READY"
            assert not gate["failed_gates"]
    print("PASS  aggregate gate: "
          + ", ".join(f"stage {s['stage']} {s['verdict']}" for s in stages))


def test_a_missing_measurement_fails_the_gate_exactly_like_a_bad_one():
    """NOT_CHECKED is not CLEAR, and the gate is written so it cannot be.

    This is the trap the collision check fell into (CLAUDE.md section 11):
    `checked: false` was read as "no interference". Here an empty record must
    fail EVERY gate rather than pass by default.
    """
    empty = mfg.aggregate_print_gate({})
    assert empty["print_ready"] is False
    assert empty["verdict"] == "NOT PRINT READY"
    assert len(empty["failed_gates"]) == len(empty["gates"]), (
        "a record with no measurements in it must fail every gate")
    # And one good measurement must not carry the rest.
    one = mfg.aggregate_print_gate({"stl_validation": {"print_ready": True}})
    assert one["print_ready"] is False
    assert "written_stl_topology" not in one["failed_gates"]
    print(f"PASS  an empty stage record fails all "
          f"{len(empty['gates'])} gates")


def test_every_stage_records_where_its_geometry_came_from():
    """Provenance, exposure and volume separation are recorded per stage.

    Without provenance the exposure measurement is impossible - "is this
    triangle invented geometry standing on the outside of the model" has no
    heuristic answer - so this pins that the labels reached the manifest.
    """
    stages, _ = _stages([dict(d_oa=0.6), dict(d_md=0.4)], 2)
    for s in stages:
        exp = s["synthetic_exposure"]
        assert exp["unattributed_area_mm2"] == 0.0, (
            f"stage {s['stage']}: {exp['unattributed_area_mm2']}mm2 of the "
            f"finished surface could not be traced to any source solid")
        by = exp["area_by_source_mm2"]
        assert by["ORIGINAL_CAST"] > 0 and by["CROWN"] > 0
        assert exp["within_bound"], exp
        vol = s["volumes_mm3"]
        for key in ("original_cast_mm3", "cast_after_local_clearance_mm3",
                    "local_clearance_removed_mm3", "connector_total_mm3",
                    "crown_total_mm3", "crown_cast_overlap_mm3",
                    "connector_cast_overlap_mm3", "fused_mm3"):
            assert key in vol, key
        # The clearance report must state whether it cut and, if not, why.
        cl = s["clearance"]
        assert "emitted" in cl and "cast_bodies_after_subtract" in cl
        if not cl["emitted"] and cl["tools_built"]:
            assert cl["reason"], "a withheld clearance tool must say why"
    print("PASS  provenance: "
          + ", ".join(f"stage {s['stage']} synthetic "
                      f"{s['synthetic_exposure']['exposed_synthetic_fraction']*100:.2f}%"
                      for s in stages))


if __name__ == "__main__":
    import sys
    failures = []
    for name, obj in sorted(list(globals().items())):
        if not name.startswith("test_"):
            continue
        marks = getattr(obj, "pytestmark", [])
        params = []
        for mk in marks:
            if mk.name == "parametrize":
                params = mk.args[1]
        try:
            if params:
                for p in params:
                    obj(*(p if isinstance(p, (tuple, list)) else (p,)))
            else:
                obj()
        except Exception as e:                            # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} FAILED")
        sys.exit(1)
    print("\nALL MANUFACTURING MATRIX TESTS PASSED")
