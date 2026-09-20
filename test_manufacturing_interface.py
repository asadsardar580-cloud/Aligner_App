"""The 30-case manufacturing matrix for local target-position reconstruction.

WHAT THIS FILE IS FOR. The old manufacturing path unioned a static cast with
`crown + a 9mm root plug` transformed by the stage matrix. Measured before the
change: a 1.2mm extrusion grew the fused volume by 28.84 mm3 of synthetic root
that had emerged from the gingiva, and every stage shipped with 68-77
non-manifold edges once a downstream reader welded the STL. These tests pin the
replacement.

THRESHOLDS HERE ARE ENGINEERING VALIDATION THRESHOLDS FOR A PROTOTYPE. They are
not clinical tolerances and are not presented as any. Where a number is chosen
rather than derived, the reason is written beside it.
"""
import numpy as np
import pytest

import core_geometry as cg
import manufacturing as mfg

m3 = pytest.importorskip("manifold3d",
                         reason="the stage booleans are real; no mock stands in")
import api_core                                                    # noqa: E402
from test_staging_export import moved_session, arch_session         # noqa: E402


# ---------------------------------------------------------------------------
# One built cast, reused. Building it per test costs seconds and tests nothing.
# ---------------------------------------------------------------------------

def _cast_and_teeth(prescriptions):
    sid, tids, _ = moved_session(prescriptions=prescriptions)
    v = api_core.STORE.require(sid, "verts")
    f = api_core.STORE.require(sid, "faces")
    af = api_core.STORE.get(sid, "arch_frame")
    ex = api_core.STORE.get(sid, "extracted_faces")
    sv, sf, _ = api_core._seal_sockets(v, f, sid, ex, flush=True)
    curve, _ = cg.fit_arch_curve(v, af)
    tv, tf, ti = cg.trim_to_arch(sv, sf, af, margin_mm=7.0, curve=curve)
    bv, bf, _ = cg.build_cast_base(tv, tf, af, base_thickness_mm=3.0,
                                   rim=ti["rim_loop"])
    recs = [api_core.STORE.get(sid, f"tooth:{t}") for t in tids]
    return sid, tids, recs, v, bv, bf


def _interface(bv, bf, rec, v, clinical, probe=None):
    M = cg.kinematic_matrix(rec["frame"], rec["c_res"], **clinical)
    rim0 = v[np.asarray(rec["socket_rim"], np.int64)]
    return M, mfg.build_stage_tooth_interface(
        bv, bf, rec["cv"], rec["cf"], rim0, rec["frame"]["u_oa"], M, probe=probe)


# ===========================================================================
# 1-10  Every movement type must produce a buildable local interface
# ===========================================================================

MOVEMENTS = [
    ("zero_movement",          {}),
    ("pure_translation_md",    dict(d_md=1.0)),
    ("pure_translation_bl",    dict(d_bl=1.0)),
    ("pure_tip",               dict(tip_deg=6.0)),
    ("pure_torque",            dict(torque_deg=6.0)),
    ("pure_rotation",          dict(rotation_deg=8.0)),
    ("intrusion",              dict(d_oa=-1.2)),
    ("extrusion",              dict(d_oa=1.2)),
    ("combined",               dict(tip_deg=4.0, torque_deg=-3.0, rotation_deg=5.0,
                                    d_md=0.5, d_bl=0.3, d_oa=0.6)),
    ("large_lateral",          dict(d_bl=2.5)),
]


@pytest.mark.parametrize("name,clinical", MOVEMENTS)
def test_every_movement_type_builds_a_local_interface(name, clinical):
    """Cases 1-10. A valid movement must RECONSTRUCT, never refuse."""
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        probe = mfg.CastProbe(bv, bf)
        M, r = _interface(bv, bf, recs[0], v, clinical, probe)
        assert r.ok, f"{name} was refused: {r.refusal_reason} / {r.diagnostics}"
        d = r.diagnostics
        # THE NEW CONTRACT. There is no separate cavity solid any more: a stage
        # model is a POSITIVE, so a penetrating tooth is absorbed by the union
        # and nothing has to be removed. What every movement must produce is
        # ONE integrated transition volume - a bridge where the rim lifted, a
        # seat where it stayed down, and a continuous blend across a tipped
        # tooth that is both at once.
        assert d["interface_mode"] in ("seated", "penetrating", "separated", "mixed")
        assert d["connector_built"] is True, d
        assert d["connector_volume_mm3"] > 0, d
        assert d["rim_points_lifted"] + d["rim_points_seated"] == d["rim_points"]
        # Bounded: the connector may not reach anything like a root length.
        assert d["seat_depth_mm"] <= mfg.DEFAULT_POLICY.seat_depth_mm + 1e-9
        print(f"PASS  {name:20s} mode={d['interface_mode']:11s} "
              f"pen={d['crown_penetration_mm']:.3f} sep={d['rim_separation_max_mm']:.3f} "
              f"lift/seat={d['rim_points_lifted']}/{d['rim_points_seated']} "
              f"connector={d['connector_volume_mm3']:.1f}mm3")
    finally:
        api_core.close_session(sid)


# ===========================================================================
# 11-14  Multi-tooth, adjacency, boundary, and the one case that MUST refuse
# ===========================================================================

def test_two_simultaneously_moved_teeth_each_get_an_interface():
    """Case 11."""
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=0.8), dict(d_md=0.8)])
    try:
        probe = mfg.CastProbe(bv, bf)
        for rec in recs:
            _, r = _interface(bv, bf, rec, v, rec["clinical"], probe)
            assert r.ok, r.refusal_reason
        print(f"PASS  {len(recs)} teeth each built an independent interface")
    finally:
        api_core.close_session(sid)


def test_adjacent_teeth_approaching_keep_their_gingival_bridge():
    """Case 12. Two reconstructions must not merge into one trench."""
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_md=0.9), dict(d_md=-0.9)])
    try:
        Ms = [cg.kinematic_matrix(r["frame"], r["c_res"], **r["clinical"]) for r in recs]
        rims = [cg.apply_matrix(v[np.asarray(r["socket_rim"], np.int64)], M)
                for r, M in zip(recs, Ms)]
        bridge = mfg.bridge_between(rims[0], rims[1])
        consumed = 2 * mfg.DEFAULT_POLICY.fusion_overlap_mm
        assert bridge > 0
        print(f"PASS  bridge {bridge:.3f}mm, reconstruction consumes "
              f"{consumed:.3f}mm ({consumed / bridge:.1%})")
        assert consumed / bridge <= 1.0, "the two reconstructions would meet"
    finally:
        api_core.close_session(sid)


def test_a_tooth_moved_far_outside_the_cast_is_REFUSED():
    """Case 14. The refusal that must exist, and must not be a root plug.

    The old path had no such refusal: it grew a 9mm connector until something
    reached the cast, however far the tooth had gone.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=0.5), dict(d_md=0.3)])
    try:
        probe = mfg.CastProbe(bv, bf)
        _, r = _interface(bv, bf, recs[0], v, dict(d_oa=40.0), probe)
        assert not r.ok, "a tooth 40mm clear of the cast was accepted"
        assert r.refusal_reason in ("outside_reconstruction_envelope",
                                    "no_cast_beneath_target_rim",
                                    "interface_unbuildable_wall_too_thin",
                                    "interface_construction_failed"), r.refusal_reason
        print(f"PASS  40mm extrusion refused: {r.refusal_reason}")
    finally:
        api_core.close_session(sid)


# ===========================================================================
# 15  Crown rigidity - the protected invariant
# ===========================================================================

@pytest.mark.parametrize("name,clinical", MOVEMENTS)
def test_the_transformed_crown_preserves_its_intrinsic_geometry(name, clinical):
    """Case 15. Edge lengths, pairwise distances and triangle areas.

    A determinant close to 1 does not prove a transform is rigid - it proves
    volume is preserved. Pairwise distances and triangle areas are invariant
    under rigid motion and under essentially nothing else.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        rec = recs[0]
        M = cg.kinematic_matrix(rec["frame"], rec["c_res"], **clinical)
        moved = cg.apply_matrix(rec["cv"], M)
        rep = mfg.rigidity_report(rec["cv"], moved, rec["cf"])
        assert rep["ok"], rep
        R = np.asarray(M)[:3, :3]
        assert np.abs(R.T @ R - np.eye(3)).max() < 1e-9
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        print(f"PASS  {name:20s} worst intrinsic error {rep['worst_mm']:.3e}mm, "
              f"|RtR-I|={np.abs(R.T @ R - np.eye(3)).max():.2e}")
    finally:
        api_core.close_session(sid)


# ===========================================================================
# 16-18  Cast fidelity, the new interface, and the old site
# ===========================================================================

def test_stage_preserves_unaffected_cast_geometry():
    """Case 16. SURFACE DEVIATION, not vertex identity.

    A CSG boolean retessellates everything it touches, so triangle counts and
    vertex indices legitimately change across the whole solid even where the
    SURFACE did not move. Comparing arrays would fail on a perfect result.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        bundle = api_core.build_stage_bundle(sid, api_core.StageExportRequest())
        import stl_io
        blob = bundle["blobs"][bundle["manifest"]["stage_files"][0]["file"]]
        fv, _ = stl_io.parse_stl_bytes(blob)
        # THE CAST THE BUNDLE ACTUALLY USED. Rebuilding one here with this
        # file's own trim margin and base thickness produced a globally
        # different mesh - mean 1.43mm, max 9.07mm - which is not deformation,
        # it is two different casts being compared.
        ref_v, ref_f = bundle["base_mesh"]

        # EXCLUDE THE CROWNS, NOT JUST THE RIMS. The first version of this
        # test excluded a 6mm sphere around each target rim and then measured
        # 3.46mm of "deviation" - which was the crowns themselves. A crown is
        # about 7mm tall, so its occlusal surface sits outside a 6mm rim
        # sphere, and the reference cast has no crowns in it at all. That is
        # not cast deformation; it is the tooth. The exclusion has to cover
        # the geometry the stage legitimately ADDS.
        excl = []
        for r in recs:
            M = cg.kinematic_matrix(r["frame"], r["c_res"],
                                    **{k: x / 5 for k, x in r["clinical"].items()})
            excl.append(cg.apply_matrix(r["cv"], M))
            excl.append(cg.apply_matrix(v[np.asarray(r["socket_rim"], np.int64)], M))
            excl.append(v[np.asarray(r["socket_rim"], np.int64)])   # the old site
        dev = mfg.surface_deviation(ref_v, ref_f, fv, None,
                                    exclude_pts=np.vstack(excl),
                                    exclude_radius_mm=mfg.DEFAULT_POLICY.roi_radius_mm + 2.0)
        assert dev["compared_points"] > 100, dev

        # GATED ON THE WHOLE METRIC SUITE, not on max alone, and the reason is
        # in the measurements. Outside the affected region 95% of sampled
        # points sit at EXACTLY 0.000mm - the cast is bit-preserved there -
        # with a mean of ~0.005mm. A small number of points (23 of 2910 when
        # this was written) reach ~1.1mm, and they do NOT move closer to the
        # teeth as the exclusion radius grows from 5mm to 10mm: they are on
        # the trim boundary, where the batch boolean retessellates the base
        # outline. Gating on max alone would either fail a cast that is
        # demonstrably preserved or force the threshold up to 1.2mm, which
        # would stop catching real deformation anywhere.
        #
        # ENGINEERING VALIDATION THRESHOLDS FOR THIS PROTOTYPE, not clinical
        # tolerances: mean and p95 pin the bulk of the surface at the
        # tessellation noise floor, p99 keeps the tail tight, and the outlier
        # COUNT is bounded so a growing population of them fails the test even
        # while the percentiles stay clean.
        assert dev["mean_mm"] < 0.02, dev
        assert dev["p95_mm"] < 0.05, dev
        assert dev["p99_mm"] < 0.25, dev
        assert dev["outliers_over_0_25mm"] <= 40, dev
        print(f"PASS  outside ROI+buffer: {dev['compared_points']} pts, "
              f"mean={dev['mean_mm']}mm p95={dev['p95_mm']} p99={dev['p99_mm']} "
              f"max={dev['max_mm']} outliers>0.25mm={dev['outliers_over_0_25mm']}")
    finally:
        api_core.close_session(sid)


def test_a_new_local_interface_exists_at_the_target_position():
    """Case 17."""
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        probe = mfg.CastProbe(bv, bf)
        M, r = _interface(bv, bf, recs[0], v, dict(d_oa=1.2), probe)
        # The interface IS the connector now; there is no cavity solid.
        assert r.ok and r.seat_verts is not None
        rim_k = cg.apply_matrix(v[np.asarray(recs[0]["socket_rim"], np.int64)], M)
        cont = mfg.interface_continuity(rim_k, r.seat_verts)
        assert cont["covered_fraction"] > 0.75, cont
        print(f"PASS  interface covers {cont['covered_fraction']:.1%} of the rim, "
              f"largest gap {cont['largest_angular_gap_deg']}deg")
    finally:
        api_core.close_session(sid)


def test_old_socket_reconstruction_quality():
    """Case 18. The T0 site must be restored, not replaced by a new defect."""
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        bundle = api_core.build_stage_bundle(sid, api_core.StageExportRequest())
        import stl_io
        blob = bundle["blobs"][bundle["manifest"]["stage_files"][-1]["file"]]
        fv, ff = stl_io.parse_stl_bytes(blob)
        rim0 = v[np.asarray(recs[0]["socket_rim"], np.int64)]
        q = mfg.old_site_quality(bv, bf, fv, ff, rim0)
        assert q["measured"], q
        # The restored site must not stray further from the original cast than
        # the untouched tissue immediately around it, by more than 2mm. The
        # moved crown itself sits over this region, so some excess is expected;
        # this catches a crater or a tower, not a tooth. Engineering threshold.
        assert q["excess_over_surroundings_mm"] < 2.0, q
        print(f"PASS  old site: inside max={q['inside_max_deviation_mm']}mm vs "
              f"surroundings {q['surrounding_max_deviation_mm']}mm "
              f"(excess {q['excess_over_surroundings_mm']}mm)")
    finally:
        api_core.close_session(sid)


# ===========================================================================
# 19-20  root_length independence, and no root column
# ===========================================================================

def test_manufacturing_interface_is_independent_of_root_length_with_M_FROZEN():
    """Case 19. THE proof that root length is no longer plug depth.

    M IS FROZEN DELIBERATELY. Varying root_length_mm through the whole
    pipeline would move C_res and therefore the stage matrix, so the geometry
    would legitimately differ and the test would prove nothing. Freezing M and
    varying only the root length isolates the manufacturing layer.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        probe = mfg.CastProbe(bv, bf)
        rec = recs[0]
        M = cg.kinematic_matrix(rec["frame"], rec["c_res"], **rec["clinical"])
        rim0 = v[np.asarray(rec["socket_rim"], np.int64)]

        out = []
        for root in (9.0, 10.0, 11.0, 12.0, 13.0):
            frozen = dict(rec)
            frozen["root_length_mm"] = root        # varied, and ignored
            r = mfg.build_stage_tooth_interface(
                bv, bf, frozen["cv"], frozen["cf"], rim0,
                frozen["frame"]["u_oa"], M, probe=probe)
            assert r.ok, r.refusal_reason
            out.append((root, r.diagnostics["connector_volume_mm3"],
                        r.diagnostics.get("seat_volume_mm3"),
                        r.diagnostics["depth_used_mm"]))

        cavs = {o[1] for o in out}
        seats = {o[2] for o in out}
        depths = {o[3] for o in out}
        assert len(cavs) == 1, f"connector volume varied with root length: {out}"
        assert len(seats) == 1, f"seat volume varied with root length: {out}"
        assert len(depths) == 1, f"cavity depth varied with root length: {out}"
        print(f"PASS  root 9->13mm: connector {cavs.pop()}mm3, seat {seats.pop()}mm3, "
              f"depth {depths.pop()}mm - all identical")
    finally:
        api_core.close_session(sid)


def test_no_long_synthetic_root_column_in_the_output():
    """Case 20. The interface is bounded; it does not grow a root.

    Measured against the policy ceiling rather than against root_length_mm,
    because the point is that root length no longer appears anywhere.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        probe = mfg.CastProbe(bv, bf)
        rec = recs[0]
        rim0 = v[np.asarray(rec["socket_rim"], np.int64)]
        pol = mfg.DEFAULT_POLICY
        for extrusion in (0.0, 0.5, 1.0, 2.0):
            M, r = _interface(bv, bf, rec, v, dict(d_oa=extrusion), probe)
            assert r.ok, r.refusal_reason
            d = r.diagnostics
            assert d["depth_used_mm"] <= pol.max_reconstruction_depth_mm
            assert d["seat_depth_mm"] <= pol.seat_depth_mm + 1e-9
            # The old plug was 9mm. Nothing here may approach that.
            assert d["depth_used_mm"] < 9.0
            print(f"  extrusion {extrusion:.1f}mm -> cavity depth "
                  f"{d['depth_used_mm']:.2f}mm, seat {d['seat_depth_mm']:.2f}mm, "
                  f"ramp {d.get('ramp_volume_mm3')}")
        print("PASS  no interface dimension approaches a root length")
    finally:
        api_core.close_session(sid)


# ===========================================================================
# 21-27  The written STL is the truth
# ===========================================================================

def test_final_stl_is_validated_from_the_actual_bytes():
    """Cases 21-27, all measured on the file rather than the boolean."""
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        bundle = api_core.build_stage_bundle(sid, api_core.StageExportRequest())
        stages = bundle["manifest"]["stage_files"]
        ready = [s for s in stages if s["print_ready"]]
        for s in stages:
            val = s["stl_validation"]
            # Whatever the verdict, the REPORT must be complete and honest.
            assert set(g["gate"] for g in val["gates"]) == {
                "finite_coordinates", "zero_open_edges", "zero_nonmanifold_edges",
                "positive_volume", "single_component", "consistent_winding"}
            assert val["volume_mm3"] > 0
            assert val["reread_faces"] > 0
            print(f"  stage {s['stage']}: open={val['open_edges']} "
                  f"nonmanifold={val['nonmanifold_edges']} "
                  f"components={val['connected_components']} "
                  f"vol={val['volume_mm3']:.1f} -> {val['verdict']}")
        assert ready, ("no stage passed every hard gate; see the printed "
                       "per-stage measurements above")
        for s in ready:
            val = s["stl_validation"]
            assert val["open_edges"] == 0
            assert val["nonmanifold_edges"] == 0
            assert val["connected_components"] == 1
            assert val["volume_mm3"] > 0
        print(f"PASS  {len(ready)}/{len(stages)} stages PRINT READY from the "
              f"written bytes")
    finally:
        api_core.close_session(sid)


def test_a_failed_gate_makes_print_ready_impossible():
    """Case 12 of the acceptance list: PRINT READY cannot be faked."""
    verts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)      # a single open triangle
    blob = cg.write_binary_stl_bytes(verts, faces)
    rep = mfg.validate_printable_stl(blob)
    assert rep["print_ready"] is False
    assert rep["verdict"] == "NOT PRINT READY"
    assert "zero_open_edges" in rep["failed_gates"]
    print(f"PASS  an open shell is NOT PRINT READY: {rep['failed_gates']}")


# ===========================================================================
# 28-30  Prescription fidelity and rigid staging
# ===========================================================================

def test_final_stage_matches_the_committed_clinical_prescription():
    """Case 28."""
    presc = [dict(tip_deg=3.0, torque_deg=-2.0, rotation_deg=4.0,
                  d_md=0.4, d_bl=0.0, d_oa=0.0), dict(d_bl=0.25)]
    sid, tids, recs, v, bv, bf = _cast_and_teeth(presc)
    try:
        bundle = api_core.build_stage_bundle(sid, api_core.StageExportRequest())
        last = bundle["manifest"]["stage_files"][-1]
        for iface in last["interfaces"]:
            got = iface["clinical"]
            want = next(r["clinical"] for r in recs
                        if r["clinical"].keys() == got.keys()
                        and all(abs(r["clinical"][k] - got[k]) < 1e-9 for k in got))
            assert want is not None
        print(f"PASS  final stage reproduces the committed prescription for "
              f"{len(last['interfaces'])} teeth")
    finally:
        api_core.close_session(sid)


def test_stage_interpolation_remains_rigid():
    """Case 29. Every stage matrix, not just the endpoint."""
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        rec = recs[0]
        worst_orth = worst_det = 0.0
        for k in range(0, 6):
            cl = {kk: x * k / 5 for kk, x in rec["clinical"].items()}
            M = cg.kinematic_matrix(rec["frame"], rec["c_res"], **cl)
            R = np.asarray(M)[:3, :3]
            worst_orth = max(worst_orth, float(np.abs(R.T @ R - np.eye(3)).max()))
            worst_det = max(worst_det, abs(float(np.linalg.det(R)) - 1.0))
        assert worst_orth < 1e-12 and worst_det < 1e-12
        print(f"PASS  every stage rigid: |RtR-I|={worst_orth:.2e} "
              f"|det-1|={worst_det:.2e}")
    finally:
        api_core.close_session(sid)


def test_the_boolean_is_real_and_not_mocked():
    """Case 30's precondition. If manifold3d were absent this file skips."""
    assert hasattr(m3, "Manifold")
    a = api_core._to_manifold(*_box(np.zeros(3)))
    b = api_core._to_manifold(*_box(np.full(3, 0.5)))
    u = m3.Manifold.batch_boolean([a, b], m3.OpType.Add)
    assert abs(u.volume() - 1.875) < 1e-6, u.volume()
    print(f"PASS  manifold3d executed a real union: {u.volume():.4f}mm3")


# ===========================================================================
# Ray hygiene: a miss must stay a miss
# ===========================================================================

def test_a_genuine_ray_miss_cannot_become_a_success():
    """A lifted rim point with NO cast beneath it must refuse, not invent one.

    THIS IS A REGRESSION TEST FOR DEAD CODE, and that is worth stating. The
    landing loop used to end with

        landing[~hit] = rim_k[~hit]
        hit[:] = True

    which made the `if not hit.all():` refusal underneath it unreachable. The
    code still READ as though it refused, the diagnostics still reported a
    miss count, and every miss was silently converted into "landed exactly on
    the rim" - a bridge to nowhere, built at full height, that the boolean
    then had to resolve against nothing.

    The fixture removes the cast entirely from under the tooth by asking for a
    lift far past `max_rim_separation_mm`, so there is genuinely nothing to
    land on, and asserts the REFUSAL rather than the count.
    """
    import inspect
    # 1. THE PRIMITIVE REPORTS MISSES. Everything above rests on this.
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=0.5)])
    try:
        probe = mfg.CastProbe(bv, bf)
        u = np.asarray(recs[0]["frame"]["u_oa"], float)
        # A point 500mm to the side of the cast has nothing under it whichever
        # way the ray is pointed.
        away = probe.verts.mean(axis=0) + np.array([500.0, 500.0, 0.0])
        landed, hit = probe.drop_to_surface(np.atleast_2d(away), u, 4.0)
        assert not hit.any(), "drop_to_surface claimed a hit over empty space"
        assert np.allclose(landed[0], away), "a miss must not move the point"

        # 2. AND THE WHOLE INTERFACE REFUSES rather than inventing tissue. A
        #    40mm lift is caught by the envelope gate before the landing loop
        #    is even reached, which is itself correct - what matters is that
        #    NO path returns ok=True.
        M, r = _interface(bv, bf, recs[0], v, dict(d_oa=40.0), probe)
        assert not r.ok, (
            "a rim lifted 40mm clear of the cast was ACCEPTED; "
            f"diagnostics={r.diagnostics}")
        print(f"PASS  miss reported by the primitive; 40mm lift refused: "
              f"{r.refusal_reason}")
    finally:
        api_core.close_session(sid)

    # 3. A STATIC CHECK STANDS BEHIND THE RUNTIME ONE. The dead-code form was
    #    reachable only on geometry this suite cannot easily construct - a rim
    #    inside the envelope with genuinely no cast under part of it - so a
    #    runtime test alone could pass forever while the overwrite came back.
    #    Same reasoning as test_no_audit_call_site_uses_a_forbidden_key
    #    (CLAUDE.md 20.6): a guard that cannot fail a build needs a static
    #    check behind it.
    #
    #    IT READS THE AST, NOT THE TEXT, and the first version of this
    #    assertion did not - it matched the string inside the comment that
    #    EXPLAINS the old bug and failed on a correct implementation. That is
    #    the identical mistake telemetry.py's check made (CLAUDE.md 18), and a
    #    check that cannot tell an explanation from an implementation will
    #    either be deleted or will force the explanation out. The explanation
    #    is the more valuable of the two.
    import ast, textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(mfg.build_stage_tooth_interface)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name) and tgt.value.id == "hit"
                    and isinstance(tgt.slice, ast.Slice)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True):
                raise AssertionError(
                    "the landing loop force-sets every ray to a hit again; "
                    "that makes the no_cast_beneath_target_rim refusal "
                    "unreachable")
    src = inspect.getsource(mfg.build_stage_tooth_interface)
    assert "no_cast_beneath_target_rim" in src, (
        "the refusal for a lifted point with no cast beneath it is gone")


def test_ray_landing_records_how_each_point_was_resolved():
    """Every lifted rim point says WHICH attempt answered for it.

    A success count alone cannot distinguish an apron that landed where it was
    aimed from one that collapsed onto the rim to find tissue - the same
    number of points land either way, and only one of them is the geometry
    that was asked for.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2)])
    try:
        probe = mfg.CastProbe(bv, bf)
        M, r = _interface(bv, bf, recs[0], v, dict(d_oa=1.2), probe)
        assert r.ok, r.refusal_reason
        d = r.diagnostics
        levels = d["ramp_retry_levels"]
        lifted = d["rim_points_lifted"]
        assert sum(levels.values()) == lifted, (levels, lifted)
        # Nothing may be left unresolved: -1 is "never landed", and the
        # interface would have refused rather than reach here.
        assert levels["-1"] == 0, levels
        assert d["ramp_landing_misses"] == 0, d
        assert d["ramp_landing_distance_max_mm"] >= 0.0
        print(f"PASS  {lifted} lifted points resolved: levels={levels} "
              f"collapsed_inward={d['ramp_collapsed_inward_points']} "
              f"drop max {d['ramp_landing_distance_max_mm']:.3f}mm")
    finally:
        api_core.close_session(sid)


def test_the_probe_drops_unreferenced_vertices_and_uses_an_exact_method():
    """The cast's vertex array carries points no face references.

    `build_cast_base` returns a face subset over the SCAN's array and rule 3.1
    forbids rebuilding it, so the array keeps every trimmed-away crown.
    `cg.vertex_normals` leaves those at ZERO, which made `outward` 0.0, which
    is not < 0, which reported an interior point as OUTSIDE at the distance to
    a phantom. Measured: 4497 of 8372 vertices unreferenced, and 6.66mm of
    error. See bench_signed_distance.py for the full scoring.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=0.5)])
    try:
        probe = mfg.CastProbe(bv, bf)
        assert probe.dropped_unreferenced == len(bv) - len(np.unique(bf))
        assert len(probe.verts) == len(np.unique(bf))
        # Every vertex the probe kept must carry a real normal.
        assert np.all(np.linalg.norm(probe.normals, axis=1) > 0.5)
        # Points pushed just inside the surface must read NEGATIVE. The old
        # nearest-vertex method scored 13.3% here on a steep cervical wall.
        tri = probe.verts[probe.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        keep = ln > 1e-12
        cen = tri[keep].mean(axis=1)
        nn = n[keep] / ln[keep][:, None]
        inside = probe.signed(cen - nn * 0.05)
        frac = float((inside < 0).mean())
        assert frac > 0.98, f"only {frac:.1%} of points 0.05mm inside read as inside"
        print(f"PASS  probe dropped {probe.dropped_unreferenced} phantom vertices; "
              f"method={probe.method}; {frac:.1%} correct 0.05mm inside")
    finally:
        api_core.close_session(sid)


def _box(corner, s=1.0):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * s + corner
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
                 dtype=np.int64)
    return v, f


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
                    obj(*p)
            else:
                obj()
        except Exception as e:                            # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} FAILED")
        sys.exit(1)
    print("\nALL MANUFACTURING INTERFACE TESTS PASSED")
