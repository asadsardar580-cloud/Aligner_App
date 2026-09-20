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

def test_the_fused_result_does_not_depend_on_tooth_order():
    """Two teeth, fused in both orders, must agree.

    A boolean pipeline whose answer depends on the order the parts arrive in
    is one whose answer is partly luck, and with per-tooth reconstruction
    there are now real opportunities for that: each tooth's interface is built
    against the SAME immutable cast, so neither may see the other's edit.

    The tolerance is on VOLUME rather than on vertices: CSG retessellates, so
    identical geometry legitimately arrives with different triangle counts.
    """
    forward = [dict(d_oa=1.0), dict(d_md=0.8)]
    reverse = [dict(d_md=0.8), dict(d_oa=1.0)]
    a, _ = _stages(forward, 2)
    b, _ = _stages(reverse, 2)
    assert len(a) == len(b) == 2
    for sa, sb in zip(a, b):
        va, vb = sa["volume_mm3"], sb["volume_mm3"]
        # 0.05% of ~27,600mm3 is ~14mm3 - far tighter than any interface
        # volume, so a reconstruction applied in the wrong order cannot hide.
        assert abs(va - vb) / max(va, 1.0) < 5e-4, (
            f"stage {sa['stage']}: {va:.3f} forward vs {vb:.3f} reversed")
        assert sa["manifold_bodies"] == sb["manifold_bodies"] == 1
    print(f"PASS  order independence: "
          + ", ".join(f"stage {sa['stage']} {sa['volume_mm3']:.3f} vs "
                      f"{sb['volume_mm3']:.3f}" for sa, sb in zip(a, b)))


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
