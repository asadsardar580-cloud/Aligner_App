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

        # CONTINUITY IS MEASURED AGAINST THE SOLID, not against its vertices.
        # `interface_continuity` compares each rim point with the nearest
        # connector VERTEX at a 1mm tolerance, which was right for a loft whose
        # top ring hugged the rim and is wrong for a collar that ENCLOSES it:
        # the rim is now strictly inside the solid and the nearest ring vertex
        # can legitimately be 2mm away, so the vertex measure reads 11% covered
        # for a band that covers the rim completely. Inside-ness is the
        # property that matters, and the exact signed distance answers it.
        cont = r.diagnostics["continuity"]
        assert cont["covered_fraction"] > 0.75, cont
        assert cont["continuous"], cont
        assert cont["max_rim_signed_distance_to_connector_mm"] < 0.0, cont
        # The crown's own cervical crease must be INSIDE the connector - that
        # is what stops the connector's surface crossing a crease, which is
        # where every self-touch in the fused solid was measured to sit.
        assert r.diagnostics["crease_points_outside_collar"] == 0, r.diagnostics
        # And the vertex-based measure still exists and still works on the
        # geometry it was written for.
        assert mfg.interface_continuity(rim_k, rim_k)["covered_fraction"] == 1.0
        print(f"PASS  interface encloses {cont['covered_fraction']:.1%} of the "
              f"rim, largest gap {cont['largest_angular_gap_deg']}deg, "
              f"deepest rim point {cont['max_rim_signed_distance_to_connector_mm']}mm inside")
    finally:
        api_core.close_session(sid)


def test_old_socket_reconstruction_quality():
    """Case 18. The T0 site must be restored, not replaced by a new defect.

    MEASURED ON THE CAST'S OWN SURFACE, which is why this reads the production
    manifest rather than calling the metric on a reread STL. The finished model
    over the old site is often the CROWN - a barely-moved tooth is still
    standing there - and scoring the crown's occlusal surface as if it were
    restored gingiva reported a 4.42mm "crater" at a site nothing had touched.
    The export separates the two with per-triangle provenance; a test parsing
    an STL cannot, so it asserts the integrated result.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        bundle = api_core.build_stage_bundle(sid, api_core.StageExportRequest())
        pol = mfg.DEFAULT_POLICY
        seen = 0
        for meta in bundle["manifest"]["stage_files"]:
            for row in meta["interfaces"]:
                q = row["old_site"]
                assert q["measured"], q
                assert "point-to-TRIANGLE" in q.get("measure", "") or                     not q.get("assessable"), q
                if not q.get("assessable"):
                    # A DETERMINATE ANSWER, not an unchecked one.
                    assert 0.0 <= q["obscured_fraction"] <= 1.0, q
                    assert q["reason"], q
                    continue
                seen += 1
                assert q["crater_depth_mm"] <= pol.max_old_site_defect_mm, q
                assert q["plateau_height_mm"] <= pol.max_old_site_defect_mm, q
                assert q["largest_step_change_mm"] <= pol.max_old_site_step_mm, q
                assert q["patches"] <= 1, q
        print(f"PASS  old site integrated into the manifest: {seen} assessable "
              f"record(s) across {len(bundle['manifest']['stage_files'])} stages")
    finally:
        api_core.close_session(sid)


def test_old_site_quality_actually_detects_a_crater():
    """And the metric can FAIL, which is the half a passing case cannot show.

    CLAUDE.md section 5: verify a fixture reproduces the bug before trusting a
    regression test. A crater is dug into the restored site deliberately and
    the measurement has to name it.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2)])
    try:
        rim0 = v[np.asarray(recs[0]["socket_rim"], np.int64)]
        u = np.asarray(recs[0]["frame"]["u_oa"], float)
        centre = rim0.mean(axis=0)
        clean = mfg.old_site_quality(bv, bf, bv, bf, rim0, u_oa=u)
        assert clean["measured"] and clean.get("assessable"), clean
        assert clean["crater_depth_mm"] < 1e-6, clean

        # Push every REFERENCED cast vertex over the old site 1.5mm along
        # -u_oa. Referenced matters: 47 of the cast's vertices sit inside this
        # site and NONE of them are used by a face - they are the crown the
        # trim removed, kept because rule 3.1 forbids rebuilding the scan's
        # array - so digging by 3D proximity moved 47 phantoms and changed no
        # surface at all. The first version of this control did exactly that
        # and "passed" by reporting a 0.0mm crater.
        dug = np.array(bv, copy=True)
        fb = np.asarray(bf, np.int64)
        nrm = u / (np.linalg.norm(u) or 1.0)
        cen = dug[fb].mean(axis=1) - centre
        in_plane = np.linalg.norm(cen - np.outer(cen @ nrm, nrm), axis=1)
        over = in_plane < float(
            np.linalg.norm(rim0 - centre, axis=1).max()) * 0.5
        inside = np.zeros(len(dug), bool)
        inside[np.unique(fb[over])] = True
        assert inside.sum() > 10, f"the control moved {inside.sum()} vertices"
        dug[inside] -= u * 1.5
        bad = mfg.old_site_quality(bv, bf, dug, bf, rim0, u_oa=u)
        assert bad["measured"] and bad.get("assessable"), bad
        assert bad["crater_depth_mm"] > 1.0, bad
        assert bad["crater_depth_mm"] > mfg.DEFAULT_POLICY.max_old_site_defect_mm
        assert bad["largest_step_change_mm"] > mfg.DEFAULT_POLICY.max_old_site_step_mm
        assert clean["largest_step_change_mm"] < 1e-6, clean
        print(f"PASS  a deliberate 1.5mm crater is reported as "
              f"{bad['crater_depth_mm']}mm deep, step change "
              f"{bad['largest_step_change_mm']}mm, against "
              f"{clean['crater_depth_mm']}mm / {clean['largest_step_change_mm']}mm clean")
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
    # THE VERDICT IS DELIBERATELY NARROWER THAN "PRINT READY". This function
    # measures boolean/topology properties of the written bytes and nothing
    # else, so it must not claim the aggregate manufacturing verdict - which
    # also needs transition quality, old-site quality, two-sided cast
    # fidelity, ROI compliance and seat/ramp exposure.
    assert rep["verdict"] == "FAILS BOOLEAN/TOPOLOGY REGRESSION"
    assert "PRINT READY" not in rep["verdict"], (
        "this gate must not claim the aggregate manufacturing verdict")
    scope = rep["gate_scope"]
    assert "zero_nonmanifold_edges" in scope["covers"]
    for unmeasured in ("transition_quality", "old_site_quality",
                       "two_sided_cast_fidelity", "roi_compliance"):
        assert unmeasured in scope["does_not_cover"], unmeasured
    assert "zero_open_edges" in rep["failed_gates"]
    print(f"PASS  an open shell FAILS the topology gate: {rep['failed_gates']}")


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
        # EVERY rim point is resolved now, not only the lifted ones: the
        # connector is one collar whose lower ring lands for the whole rim,
        # so a seated point has a landing too and has to say how it found it.
        assert sum(levels.values()) == d["rim_points"], (levels, d["rim_points"])
        assert lifted <= d["rim_points"]
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


def test_unused_vertices_cannot_change_the_manufacturing_signed_distance():
    """Padding the array with arbitrary unreferenced vertices must change nothing.

    THE INVARIANT THIS PROTECTS is rule 3.1: the scan's vertex array is never
    rebuilt, so the cast legitimately carries vertices no face references -
    measured, 4497 of 8372, every crown that was trimmed away. The defect that
    caused was not that they exist, it is that a geometric query READ them:
    `cg.vertex_normals` leaves an unreferenced vertex's normal at ZERO, so the
    nearest-vertex sign test computed `outward = 0.0`, `0 < 0` is False, and an
    interior point was reported OUTSIDE at the distance to a phantom.

    So the contract is not "there are no unused vertices". It is "every
    geometric query uses the active triangles only", and this test states it
    the way it can actually fail: by inventing new unused vertices in wild
    positions and requiring the answers to be bit-identical.
    """
    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=0.5)])
    try:
        rng = np.random.default_rng(11)
        probe_a = mfg.CastProbe(bv, bf)

        # 5000 unreferenced vertices scattered far outside the cast, plus some
        # placed INSIDE it, which is the case that would actually flip a sign.
        far = rng.uniform(-500, 500, size=(4000, 3))
        inside = bv[np.unique(bf)].mean(axis=0) + rng.normal(scale=1.0, size=(1000, 3))
        padded = np.vstack([bv, far, inside])
        probe_b = mfg.CastProbe(padded, bf)     # same faces, same indices

        assert probe_b.dropped_unreferenced == probe_a.dropped_unreferenced + 5000
        assert np.array_equal(probe_a.verts, probe_b.verts)
        assert np.array_equal(probe_a.faces, probe_b.faces)

        pts = bv[np.unique(bf)][rng.choice(len(np.unique(bf)), 300, replace=False)]
        pts = pts + rng.normal(scale=0.3, size=pts.shape)
        a, b = probe_a.signed(pts), probe_b.signed(pts)
        assert np.array_equal(a, b), (
            f"unused vertices changed the signed distance: "
            f"max delta {np.abs(a - b).max()}")

        # And the interface built on top of it must be identical too - BUILT
        # THROUGH THE CONSTRUCTOR, not handed a probe. Passing `probe_a` and
        # `probe_b` explicitly made this half of the test vacuous: the only
        # uses of `base_verts` inside build_stage_tooth_interface are
        # `np.asarray(base_verts, float)` and `probe or CastProbe(...)`, so
        # with a probe supplied the padded array was converted and discarded,
        # and the comparison would have held even if the defect were fully
        # reintroduced. CLAUDE.md section 5: verify a fixture reproduces the
        # bug before trusting a regression test.
        M1, r1 = _interface(bv, bf, recs[0], v, dict(d_oa=0.5), None)
        M2, r2 = _interface(padded, bf, recs[0], v, dict(d_oa=0.5), None)
        assert r1.ok and r2.ok
        assert r1.diagnostics["interface_mode"] == r2.diagnostics["interface_mode"]
        assert (r1.diagnostics["connector_volume_mm3"]
                == r2.diagnostics["connector_volume_mm3"])
        print(f"PASS  5000 arbitrary unused vertices changed nothing: "
              f"{len(pts)} probes bit-identical, connector "
              f"{r1.diagnostics['connector_volume_mm3']:.4f}mm3 both ways")
    finally:
        api_core.close_session(sid)


def _box(corner, s=1.0):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * s + corner
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
                 dtype=np.int64)
    return v, f


# ===========================================================================
# 39-42  THE TRANSITION PROFILE IS MEASURED ON THE SURFACE, NOT ON VERTICES
#
# `transition_quality` decides `no_transition_ledge`, which is the only gate
# that has ever refused an otherwise-printable stage. It used to read the
# profile off the NEAREST VERTEX to each radial probe - the fourth appearance
# in this codebase of the mistake `CastProbe.signed`, `surface_deviation` and
# `old_site_quality` were each corrected for - and it was wrong in BOTH
# directions, which is why both controls are here:
#
#   * it MISSED a textbook cylindrical collar, reporting share 0.0000,
#     because the nearest vertex to every probe sat on the flat plate;
#   * it REJECTED a provably linear ramp at share 1.0000 when the band was
#     spanned by single large triangles, because the same vertex answers
#     several consecutive rings and then the query jumps.
#
# Each fixture is a solid of revolution whose profile is known exactly before
# the measurement runs, so the expected answer is not an opinion. Tests 41 and
# 42 assert the OLD method's answer as well, so the fixture is proved to
# reproduce the defect rather than merely to pass afterwards - CLAUDE.md
# section 5: verify a fixture reproduces the bug before trusting it.
# ===========================================================================

_NTHETA = 96


def _revolution_ring(r, z, n=_NTHETA):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n, float(z))])


def _revolution_solid(profile_radii, profile_heights, outer_r=9.0, floor=-3.0):
    """A closed solid of revolution: a top disc, the given radial profile, a
    flat outer plate, a skirt and a bottom disc."""
    verts, faces = [], []

    def add_ring(r, z):
        i = len(verts)
        verts.extend(_revolution_ring(r, z))
        return i

    def bridge(i0, i1, flip=False):
        out = []
        for k in range(_NTHETA):
            k2 = (k + 1) % _NTHETA
            a, b, c, d = i0 + k, i0 + k2, i1 + k2, i1 + k
            out += ([[a, b, c], [a, c, d]] if not flip
                    else [[a, c, b], [a, d, c]])
        return out

    def cap(i0, c, flip=False):
        return [([c, i0 + (k + 1) % _NTHETA, i0 + k] if not flip
                 else [c, i0 + k, i0 + (k + 1) % _NTHETA])
                for k in range(_NTHETA)]

    rings = [add_ring(r, z) for r, z in zip(profile_radii, profile_heights)]
    plate = add_ring(outer_r, profile_heights[-1])
    bot_out = add_ring(outer_r, floor)
    bot_in = add_ring(0.001, floor)
    top_c = len(verts)
    verts.append([0.0, 0.0, float(profile_heights[0])])
    bot_c = len(verts)
    verts.append([0.0, 0.0, floor])

    faces += cap(rings[0], top_c)
    for a, b in zip(rings, rings[1:]):
        faces += bridge(a, b)
    faces += bridge(rings[-1], plate)
    faces += bridge(plate, bot_out)
    faces += bridge(bot_out, bot_in)
    faces += cap(bot_in, bot_c, flip=True)
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def _nearest_vertex_share(verts, rim, u, band_mm=1.5, step_mm=0.25):
    """The profile EXACTLY as it was shipped, for the reproduction proof."""
    from scipy.spatial import cKDTree
    centre = rim.mean(axis=0)
    radial = rim - centre
    rn = np.linalg.norm(radial, axis=1, keepdims=True)
    rdir = np.divide(radial, np.where(rn < 1e-12, 1.0, rn))
    tree = cKDTree(verts)
    h = []
    for off in np.arange(0.0, band_mm + 1e-9, step_mm):
        _, idx = tree.query(rim + rdir * off, workers=-1)
        h.append(float(np.median((verts[idx] - centre) @ u)))
    h = np.asarray(h)
    total = float(abs(h[0] - h[-1]))
    worst = float(np.abs(np.diff(h)).max())
    return h, (worst / total if total > 1e-6 else 0.0)


_UP = np.array([0.0, 0.0, 1.0])
_COLLAR_FALL = 0.946    # the historical collar's own fall, CLAUDE.md s.23


def test_a_cylindrical_collar_is_still_called_a_ledge():
    """THE CONTROL THAT MATTERS. A vertical wall at the cervical radius with
    a flat plate beyond it is the exact artefact the check exists to catch."""
    v, f = _revolution_solid([4.0, 4.0], [_COLLAR_FALL, 0.0])
    rim = _revolution_ring(4.0, _COLLAR_FALL)
    r = mfg.transition_quality(v, f, rim, _UP)
    assert r["measured"] is True
    assert r["looks_like_a_ledge"] is True, r
    assert r["largest_step_share"] > 0.99, r
    assert abs(r["total_fall_mm"] - _COLLAR_FALL) < 1e-3, r
    print(f"PASS  a cylindrical collar is a ledge: share "
          f"{r['largest_step_share']}, fall {r['total_fall_mm']}mm")


def test_the_old_nearest_vertex_profile_MISSED_that_collar():
    """The defect, in the direction nobody looked for. On the same solid the
    shipped measurement reported NO fall at all, because the nearest vertex
    to every probe outside the wall sits on the flat plate."""
    v, f = _revolution_solid([4.0, 4.0], [_COLLAR_FALL, 0.0])
    rim = _revolution_ring(4.0, _COLLAR_FALL)
    h, share = _nearest_vertex_share(v, rim, _UP)
    assert share < 0.75, (
        "the fixture no longer reproduces the false negative", h, share)
    assert float(np.abs(np.diff(h)).max()) < 1e-6, h
    print(f"PASS  the old profile MISSED the collar: share {share:.4f}, "
          f"a completely flat {np.round(h, 4).tolist()}")


@pytest.mark.parametrize("rings,fall", [(13, 0.946), (13, 3.5), (2, 0.946)])
def test_a_smooth_cone_is_called_a_blend_at_any_mesh_density(rings, fall):
    """A linear ramp is a blend whether it is meshed with twelve strips or
    one. `rings=2` is the same surface spanned by single large triangles -
    the case the nearest-vertex profile called a ledge."""
    rr = list(np.linspace(4.0, 5.5, rings))
    hh = list(np.linspace(fall, 0.0, rings))
    v, f = _revolution_solid(rr, hh)
    rim = _revolution_ring(4.0, fall)
    r = mfg.transition_quality(v, f, rim, _UP)
    assert r["measured"] is True
    assert r["looks_like_a_ledge"] is False, r
    # six equal steps down a linear ramp
    assert abs(r["largest_step_share"] - 1.0 / 6.0) < 0.02, r
    assert r["monotonic"] is True, r
    assert abs(r["total_fall_mm"] - fall) < 1e-2, r
    print(f"PASS  a {fall}mm cone in {rings} rings is a blend: share "
          f"{r['largest_step_share']}")


def test_the_old_nearest_vertex_profile_REJECTED_that_same_cone():
    """The other direction, on geometry that is provably a straight line. The
    surface is identical to the 13-ring cone above; only the mesh is coarser.
    The old method reports a 1.0 share - a total false positive - and this is
    what refused 0.6mm extrusion + 3 degrees of tip at stage 3."""
    v, f = _revolution_solid([4.0, 5.5], [_COLLAR_FALL, 0.0])
    rim = _revolution_ring(4.0, _COLLAR_FALL)
    h, share = _nearest_vertex_share(v, rim, _UP)
    assert share > 0.75, (
        "the fixture no longer reproduces the false positive", h, share)
    r = mfg.transition_quality(v, f, rim, _UP)
    assert r["looks_like_a_ledge"] is False, r
    print(f"PASS  the old profile REJECTED a linear ramp at share "
          f"{share:.4f}; the corrected one reads "
          f"{r['largest_step_share']} and accepts it")


def test_an_unmeasurable_transition_fails_the_gate_rather_than_passing_it():
    """A missing measurement is not a pass. `looks_like_a_ledge` must be None,
    never False, so `aggregate_print_gate` refuses exactly as for a real
    ledge - the same doctrine as NOT_CHECKED is not CLEAR."""
    v, f = _revolution_solid([4.0, 5.5], [_COLLAR_FALL, 0.0])
    rim = _revolution_ring(4.0, _COLLAR_FALL)
    real = mfg._raycast_scene

    def refuse(*a, **k):
        return None

    mfg._raycast_scene = refuse
    try:
        r = mfg.transition_quality(v, f, rim, _UP)
    finally:
        mfg._raycast_scene = real
    assert r["measured"] is False
    assert r["looks_like_a_ledge"] is None, r
    assert r["looks_like_a_ledge"] is not False
    stage = {"interfaces": [{"ok": True, "transition": r}]}
    gate = next(g for g in mfg.aggregate_print_gate(stage)["gates"]
                if g["gate"] == "no_transition_ledge")
    assert gate["passed"] is False, gate
    print("PASS  an unmeasurable transition fails no_transition_ledge")


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
