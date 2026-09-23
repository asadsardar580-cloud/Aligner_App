"""The /cut and /kinematics endpoints, exercised directly.

These are the surfaces the tetherball glitch was actually reported on, and
nothing covered them: test_api_core.py targets an API generation that no
longer exists (api_core has no parse_stl_bytes, preview_selection, unpack or
get_session, and create_session now takes arch as a Form field), so it has
been red independently of this work.

What is pinned here is the endpoint CONTRACT rather than the geometry --
the geometry is test_kinematics_frame.py's job:

  * a cut without an occlusal plane is refused with 409, not guessed at;
  * a cut with one anchors C_res on the socket axis, in the bone;
  * the response survives json.dumps, because api_core's frame serialisation
    only unwraps numpy ONE level deep and a nested array would surface as an
    opaque "Geometry error" 500;
  * /kinematics still holds C_res fixed under a pure rotation.
"""
import io
import json
import os
import tempfile
import zipfile
import numpy as np
from fastapi import HTTPException

import core_geometry as cg
import api_core
import arch_frame
import stl_io
from session_store import STORE
from tooth_fixture import flat_molar_on_base


def build_session():
    # 4mm of clinical crown on a 10mm-wide molar — realistic proportions that
    # clear cut_guard's compactness floor (0.597 against 0.50). The default
    # 2.2mm crown used by test_kinematics_frame.py measures 0.498 and is
    # rejected as a shell, which is the guard working correctly on a fixture
    # deliberately squatter than any real tooth.
    verts, faces, crown_r = flat_molar_on_base(crown_h=4.0, crown_r=5.0)
    edges = cg.directed_edges(faces)
    conc = cg.boundary_field(verts, faces, edges=edges)
    graph = cg.build_barrier_graph(verts, faces, conc, edges=edges)

    sid = STORE.create("lower")
    for key, val in (("verts", verts), ("faces", faces), ("edges", edges),
                     ("concavity", conc), ("graph", graph)):
        STORE.put(sid, key, val)

    dist = cg.geodesic_from_seed(graph, verts, np.array([0.0, 0.0, 2.2]))
    mask = cg.largest_face_component(faces, cg.mask_from_distance(faces, dist, 12.0))
    # A full, valid selection on purpose. A truncated one is the more
    # interesting geometry, but cut_guard.check_crown rejects it first as a
    # shell (compactness 0.471 against a 0.50 floor) and the request never
    # reaches the frame math -- useful defence in depth, and it means the
    # lopsided-crown case belongs in test_kinematics_frame.py, which exercises
    # derive_frame_from_region directly. What is under test here is the
    # endpoint contract.

    req = api_core.CutRequest(
        vertex_ids=np.unique(faces[mask]).tolist(),
        mesial_pt=[-crown_r, 0.0, 0.0], distal_pt=[crown_r, 0.0, 0.0],
        root_length_mm=9.0)
    return sid, req, verts


def occlusal_frame_for(verts):
    return arch_frame.fit_occlusal_frame([-15, 0, 0.2], [15, 0, 0.2], [0, 15, 0.2],
                                         verts.mean(axis=0))


def test_cut_without_occlusal_plane_is_refused():
    sid, req, _ = build_session()
    try:
        api_core.cut(sid, req)
    except HTTPException as e:
        assert e.status_code == 409, f"expected 409, got {e.status_code}"
        assert "occlusal" in str(e.detail).lower()
        print(f"PASS  /cut without an occlusal plane -> 409, not a guessed apical axis")
        return
    finally:
        STORE.drop(sid)
    raise AssertionError("/cut proceeded with no apical reference")


def test_cut_anchors_the_pivot_in_bone():
    sid, req, verts = build_session()
    try:
        af = occlusal_frame_for(verts)
        STORE.put(sid, "arch_frame", af)
        out = api_core.cut(sid, req)

        frame = STORE.require(sid, f"tooth:{out['tooth_id']}")["frame"]
        d = np.asarray(out["c_res"]) - frame["rim_centroid"]
        lateral = float(np.linalg.norm(d - (d @ frame["u_oa"]) * frame["u_oa"]))
        depth = float(-(d @ frame["u_oa"]))

        assert lateral < 1e-9, f"C_res {lateral:.3f}mm off the socket axis"
        assert depth > 0, "C_res must be apical of the cervical margin"
        assert out["axis_deviation_deg"] <= cg.MAX_AXIS_DEVIATION_DEG + 1e-6
        assert out["watertight"] is True
        print(f"PASS  /cut -> source={out['axis_source']}, "
              f"deviation={out['axis_deviation_deg']} deg, C_res {depth:.2f}mm apical, "
              f"lateral {lateral:.1e}mm, crown height {out['dimensions']['height_oa']:.2f}mm")
        return out
    finally:
        STORE.drop(sid)


def test_response_is_json_serialisable():
    """api_core unwraps numpy one level deep only. A nested array in the frame
    or in a diagnostics sub-dict passes the hasattr check, then fails encoding
    and lands in the bare `except Exception` as an opaque geometry error."""
    sid, req, verts = build_session()
    try:
        STORE.put(sid, "arch_frame", occlusal_frame_for(verts))
        out = api_core.cut(sid, req)
        blob = json.dumps(out)
        k = api_core.kinematics(sid, out["tooth_id"],
                                api_core.KinematicsRequest(tip_deg=5.0, d_bl=0.2))
        json.dumps(k)
        print(f"PASS  /cut response serialises ({len(blob):,} bytes) and so does /kinematics")
    finally:
        STORE.drop(sid)


def test_kinematics_holds_cres_fixed():
    sid, req, verts = build_session()
    try:
        STORE.put(sid, "arch_frame", occlusal_frame_for(verts))
        out = api_core.cut(sid, req)
        tid, c_res = out["tooth_id"], np.asarray(out["c_res"])

        M = np.array(api_core.kinematics(
            sid, tid, api_core.KinematicsRequest(tip_deg=5.0))["matrix"]).reshape(4, 4)
        assert np.allclose(cg.apply_matrix(np.array([c_res]), M)[0], c_res, atol=1e-9), \
            "C_res moved under a pure rotation"

        ident = np.array(api_core.kinematics(
            sid, tid, api_core.KinematicsRequest())["matrix"]).reshape(4, 4)
        assert np.allclose(ident, np.eye(4), atol=1e-12), "zero movement is not identity"

        k = api_core.kinematics(sid, tid, api_core.KinematicsRequest(tip_deg=5.0, d_bl=0.2))
        assert k["axis_source"] == out["axis_source"]
        print(f"PASS  /kinematics: C_res fixed under pure rotation, zero movement is "
              f"exact identity, {k['staging']['stages_required']} stages "
              f"({k['staging']['driver']}-driven)")
    finally:
        STORE.drop(sid)


def arch_session(teeth=(-0.45, 0.45), n_s=170, n_t=46):
    """A HORSESHOE session, for everything that builds a cast base.

    This replaced a flat square plate carrying two domes, which was adequate for
    /cut — that only needs a crown and a sulcus — and actively misleading for
    /export, which trims to the dental arch. A flat plate has no occlusal ridge,
    so fit_arch_curve reads a RING around the whole plate and the trim keeps a
    band along the plate EDGE: measured, that threw away 351 of 423 socket-cup
    vertices while still reporting a watertight "base". A fixture that passes
    for the wrong reason is worse than one that fails.
    """
    from test_cast_base import horseshoe_shell, frame_for
    verts, faces, apices = horseshoe_shell(n_s=n_s, n_t=n_t, teeth=teeth,
                                           return_apices=True)
    edges = cg.directed_edges(faces)
    conc = cg.boundary_field(verts, faces, edges=edges)
    graph = cg.build_barrier_graph(verts, faces, conc, edges=edges)

    sid = STORE.create("lower")
    for key, val in (("verts", verts), ("faces", faces), ("edges", edges),
                     ("concavity", conc), ("graph", graph)):
        STORE.put(sid, key, val)
    STORE.put(sid, "arch_frame", frame_for(verts))

    reqs = []
    for apex in apices:
        dist = cg.geodesic_from_seed(graph, verts, verts[apex])
        mask = cg.largest_face_component(
            faces, cg.mask_from_distance(faces, dist, 6.0))
        used = np.unique(faces[mask])
        pts = verts[used]
        # Mesial/distal along the arch: the two extremes of the selection in the
        # direction the crown is longest.
        d = pts - pts.mean(axis=0)
        axis = np.linalg.svd(d, full_matrices=False)[2][0]
        proj = d @ axis
        reqs.append(api_core.CutRequest(
            vertex_ids=used.tolist(),
            mesial_pt=pts[proj.argmin()].tolist(),
            distal_pt=pts[proj.argmax()].tolist(),
            root_length_mm=9.0))
    return sid, reqs, len(faces)


def test_cut_removes_faces_and_caps_the_socket():
    sid, req, verts = build_session()
    try:
        STORE.put(sid, "arch_frame", occlusal_frame_for(verts))
        n_faces = len(STORE.require(sid, "faces"))
        out = api_core.cut(sid, req)

        removed = out["removed_faces"]
        assert removed, "no faces removed — the tooth would still be on the cast"
        assert len(set(removed)) == len(removed), "duplicate face indices"
        assert all(0 <= i < n_faces for i in removed), "face index out of range"
        assert out["extracted_face_count"] == len(removed)

        cap = out["socket_cap"]
        assert len(cap["faces"]) >= 3, "socket left open"
        assert all(len(t) == 3 for t in cap["faces"])
        assert cap["vertices"] and all(
            len(p) == 3 and all(np.isfinite(p)) for p in cap["vertices"])

        # the session mesh itself must be untouched — every client-held vertex
        # index and every precomputed field depends on it
        assert len(STORE.require(sid, "faces")) == n_faces, "session faces mutated"
        assert len(STORE.require(sid, "verts")) == len(verts), "session verts mutated"
        print(f"PASS  /cut removes {len(removed):,} faces, caps the socket with "
              f"{len(cap['faces'])} fan tris, leaves the session mesh intact")
    finally:
        STORE.drop(sid)


def test_socket_is_a_carved_cup_with_depth():
    """The socket must descend into the bone, not lid the hole.

    The fan this replaced put its apex at the rim centroid, so the cap sat on
    the margin plane at depth 0.00mm — a flat lid, which at a grazing angle is
    the dark irregular opening that was reported.
    """
    sid, req, verts = build_session()
    try:
        STORE.put(sid, "arch_frame", occlusal_frame_for(verts))
        out = api_core.cut(sid, req)
        cap = out["socket_cap"]
        v = STORE.require(sid, "verts")
        rec = STORE.require(sid, f"tooth:{out['tooth_id']}")
        info = rec["socket_info"]
        rim = v[rec["socket_rim"]]
        n = np.asarray(info["normal"], float)

        assert cap["vertices"], "the cup created no interior vertices"
        pts = np.asarray(cap["vertices"], float)
        depth = float(np.max((rim.mean(axis=0) - pts) @ n))
        assert depth > 3.0, f"socket only reaches {depth:.2f}mm below the margin"
        assert abs(depth - cap["depth_mm"]) < 0.01, (depth, cap["depth_mm"])

        # negative index -(k+1) addresses appended vertex k; everything else is
        # a rim vertex in the arch's own buffer
        for tri in cap["faces"]:
            for i in tri:
                assert (0 <= i < len(v)) or (1 <= -i <= len(pts)), f"bad index {i}"

        print(f"PASS  socket is a {cap['profile']} reaching {depth:.2f}mm below the "
              f"margin ({len(cap['faces'])} tris, {len(pts)} new verts, "
              f"clamped={cap['depth_clamped']})")
    finally:
        STORE.drop(sid)


def test_the_displayed_socket_cup_is_the_one_that_gets_printed():
    """What the clinician approves on screen must be what the lab receives.

    This assertion is the REVERSE of the one it replaces. While /export rebuilt
    the base with cap_and_close, the printed socket was different geometry from
    the displayed cup, and the old test pinned that gap shut so a refactor could
    not silently close it. The base is now built by trim -> extrude and /export
    replays the cup /cut already stored, so the gap is closed on purpose and the
    test has to prove it rather than forbid it.

    Rebuilding the cup at export time would be cheap and still wrong: the depth
    clamp raycasts against the cast, and the cast has changed since the cut, so
    a rebuilt cup could sit at a different depth than the one on screen.
    """
    sid, reqs, _ = arch_session()
    try:
        out = api_core.cut(sid, reqs[0])
        cup = np.asarray(out["socket_cap"]["vertices"], float)
        assert len(cup) > 0

        # `out_dir` is confined to <repo>/exports since Phase 0.6 (B10), so a
        # writable directory has to be requested INSIDE it. That is the
        # capability under test here - keep_local writing the files - not the
        # removed one of writing anywhere on the filesystem.
        import shutil
        rel = f"_test_{sid[:8]}"
        tmp = os.path.join(api_core.EXPORTS_ROOT, rel)
        try:
            res = api_core.build_export_bundle(
                sid, api_core.ExportRequest(out_dir=rel, keep_local=True))
            assert os.path.exists(os.path.join(tmp, "manifest.json")), "keep_local wrote nothing"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        base = [x for x in res["files"] if x["role"] == "base"][0]
        blob = zipfile.ZipFile(res["buf"]).read(base["name"])
        bv, _ = stl_io.parse_stl_bytes(blob)

        # Every interior point of the displayed cup is present in the printed
        # base. STL is float32 and parse_stl_bytes welds, so match by proximity.
        from scipy.spatial import cKDTree
        d, _ = cKDTree(bv).query(cup)
        assert d.max() < 1e-3, \
            f"{int((d >= 1e-3).sum())} of {len(cup)} cup vertices are missing from the " \
            f"exported base (worst {d.max():.4f}mm) — the displayed socket is not printed"

        sock = res["manifest"]["sockets"]
        assert len(sock) == 1 and sock[0]["tooth_id"] == out["tooth_id"]
        assert sock[0]["triangles"] == len(out["socket_cap"]["faces"])
        assert res["manifest"]["base_construction"].startswith("trim_to_arch")
        print(f"PASS  the displayed cup is the printed one: {len(cup)} cup verts and "
              f"{sock[0]['triangles']} tris found in the exported base "
              f"({base['triangles']:,} tris, {sock[0]['profile']} at "
              f"{sock[0]['depth_mm']}mm)")
    finally:
        STORE.drop(sid)


def test_cap_and_close_never_builds_the_exported_base():
    """cap_and_close seals an open shell's perimeter by fanning across it, and
    an arch perimeter is a horseshoe — so the cap webs over the tongue space.
    Measured on a real mandibular scan: 2104 triangles for a 2101-vertex loop, a
    full hull triangulation, in 100.2 seconds.

    It is still right for the CROWN, where the cervical rim genuinely is a hole
    to fill. It must never touch the base again.
    """
    sid, reqs, _ = arch_session()
    try:
        api_core.cut(sid, reqs[0])
        v = STORE.require(sid, "verts")
        f = STORE.require(sid, "faces")
        extracted = STORE.require(sid, "extracted_faces")
        _, capped = cg.cap_and_close(v.copy(), f[~extracted])

        res = api_core.build_export_bundle(sid, api_core.ExportRequest())
        base = [x for x in res["files"] if x["role"] == "base"][0]
        assert base["triangles"] != len(capped), \
            "the exported base matches cap_and_close exactly — the old path is back"
        assert res["manifest"]["base_health"]["open_edges"] == 0
        assert res["manifest"]["base_health"]["nonmanifold_edges"] == 0
        assert res["manifest"]["cast_base"]["winding_consistent"] is True
        print(f"PASS  base is {base['triangles']:,} tris from trim->extrude, not "
              f"{len(capped):,} from cap_and_close; 0 open / 0 non-manifold")
    finally:
        STORE.drop(sid)


def test_the_exported_file_survives_a_slicer_reading_it():
    """Validate the FILE, not the index buffer.

    Binary STL stores positions, not indices, so every reader welds on load.
    A base can therefore measure watertight in memory and arrive at the lab
    non-manifold — and this is not theoretical. On a real mandibular scan the
    round-trip caught three separate defects that every count-based check in
    the pipeline reported as clean:

      * build_socket_cup emitted one floor vertex per rim vertex, and a folded
        margin projects two rim points onto one, so two interior vertices
        shared a position and welded into a 4-face edge;
      * _open_pinch_vertices originally DUPLICATED a pinch vertex rather than
        deleting the smaller fan — two copies at one position, which a weld
        rejoins straight back into the pinch;
      * the socket rim itself passed through one vertex twice, so the cup built
        the same spoke from both visits.

    The fixture is synthetic, so it will not reproduce those on its own. What
    this pins is that the check exists and that the shipped file passes it.
    """
    sid, reqs, _ = arch_session()
    try:
        api_core.cut(sid, reqs[0])
        api_core.cut(sid, reqs[1])
        res = api_core.build_export_bundle(sid, api_core.ExportRequest())

        written = res["manifest"]["base_health_as_written"]
        assert written["open_edges"] == 0 and written["nonmanifold_edges"] == 0, written

        base = [x for x in res["files"] if x["role"] == "base"][0]
        rv, rf = stl_io.parse_stl_bytes(zipfile.ZipFile(res["buf"]).read(base["name"]))
        health = cg.manifold_report(rf)
        assert health["watertight"], f"the delivered STL is not watertight: {health}"
        assert cg._winding_is_consistent(rf), \
            "the delivered STL closes but its normals are not consistently oriented"
        assert cg.signed_volume(rv, rf) > 0

        # And the weld actually did something to measure: float32 plus welding
        # must not silently drop triangles either.
        assert len(rf) == base["triangles"], \
            f"the STL lost {base['triangles'] - len(rf)} triangles on the way out"
        print(f"PASS  the exported STL re-reads watertight after a weld: {len(rv):,} verts, "
              f"{len(rf):,} tris, {health['total_edges']:,} edges, volume "
              f"{cg.signed_volume(rv, rf):,.0f}mm3")
    finally:
        STORE.drop(sid)


def test_two_cuts_are_disjoint_and_cumulative():
    sid, reqs, n_faces = arch_session()
    try:
        a = api_core.cut(sid, reqs[0])
        b = api_core.cut(sid, reqs[1])
        sa, sb = set(a["removed_faces"]), set(b["removed_faces"])
        assert not (sa & sb), f"second cut re-extracted {len(sa & sb)} faces"
        assert b["extracted_face_count"] == len(sa) + len(sb), "mask not cumulative"
        assert a["tooth_id"] != b["tooth_id"]
        assert len(STORE.require(sid, "faces")) == n_faces, "session faces mutated"
        print(f"PASS  two cuts removed {len(sa):,} + {len(sb):,} disjoint faces, "
              f"cumulative mask {b['extracted_face_count']:,}")
    finally:
        STORE.drop(sid)


def test_recutting_an_extracted_tooth_is_refused():
    sid, reqs, _ = arch_session()
    try:
        api_core.cut(sid, reqs[0])
        try:
            api_core.cut(sid, reqs[0])
        except HTTPException as e:
            assert e.status_code == 400, f"expected 400, got {e.status_code}"
            assert "already been extracted" in str(e.detail)
            print("PASS  re-cutting an extracted tooth -> 400, not a phantom crown")
            return
        raise AssertionError("the same tooth was extracted twice")
    finally:
        STORE.drop(sid)


def test_export_streams_a_zip_the_browser_can_download():
    """The defect was never a hang: export wrote STLs to a server directory and
    returned JSON paths. No blob, no Content-Disposition — nothing could ever
    reach the clinician's Downloads folder."""
    sid, reqs, _ = arch_session()
    try:
        a = api_core.cut(sid, reqs[0])
        api_core.cut(sid, reqs[1])
        api_core.kinematics(sid, a["tooth_id"],
                            api_core.KinematicsRequest(tip_deg=4.0, d_bl=0.3))

        res = api_core.export_setup(sid, api_core.ExportRequest())
        assert res.media_type == "application/zip", res.media_type
        cd = res.headers.get("content-disposition", "")
        assert "attachment" in cd and cd.endswith('.zip"'), cd

        blob = api_core.build_export_bundle(sid, api_core.ExportRequest())["buf"].read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        assert zf.testzip() is None, "corrupt ZIP"
        names = zf.namelist()
        assert "manifest.json" in names
        stls = [n for n in names if n.endswith(".stl")]
        assert sum("Base_Socketed" in n for n in names) == 1, names
        assert len(stls) == 3, f"expected base + 2 crowns, got {stls}"

        man = json.loads(zf.read("manifest.json"))
        assert man["base_watertight"] is True
        assert len(man["teeth"]) == 2
        moved = [t for t in man["teeth"] if t["moved"]]
        assert len(moved) == 1 and moved[0]["prescription"]["tip_deg"] == 4.0
        assert moved[0]["staging"]["stages_required"] > 0
        assert moved[0]["clearance"] is not None
        print(f"PASS  /export streams {len(blob):,}-byte ZIP: {', '.join(sorted(names))}")
    finally:
        STORE.drop(sid)


def test_exported_stls_parse_back_with_the_right_geometry():
    """Round-trip every STL, and check the baked movement is really baked.

    parse_stl_bytes welds and lexicographically sorts vertices, so the vertex
    ORDER does not survive — compare sorted vertex sets with a float32
    tolerance, and the TRIANGLE count exactly.
    """
    sid, reqs, _ = arch_session()
    try:
        a = api_core.cut(sid, reqs[0])
        b = api_core.cut(sid, reqs[1])
        api_core.kinematics(sid, a["tooth_id"],
                            api_core.KinematicsRequest(tip_deg=6.0, d_md=0.4))

        blob = api_core.build_export_bundle(sid, api_core.ExportRequest())["buf"].read()
        zf = zipfile.ZipFile(io.BytesIO(blob))

        def sorted_verts(x):
            return np.array(sorted(map(tuple, np.round(np.asarray(x, float), 3))))

        for tid, moved_expected in ((a["tooth_id"], True), (b["tooth_id"], False)):
            t = STORE.require(sid, f"tooth:{tid}")
            name = [n for n in zf.namelist() if tid in n][0]
            pv, pf = stl_io.parse_stl_bytes(zf.read(name))
            assert len(pf) == len(t["cf"]), f"{name}: {len(pf)} tris vs {len(t['cf'])}"

            M = t.get("matrix")
            assert (M is not None) == moved_expected
            expected = t["cv"] if M is None else cg.apply_matrix(t["cv"], np.asarray(M, float))
            got, want = sorted_verts(pv), sorted_verts(expected)
            assert got.shape == want.shape, (got.shape, want.shape)
            assert np.abs(got - want).max() < 1e-3, np.abs(got - want).max()

            if moved_expected:
                drift = np.abs(sorted_verts(pv) - sorted_verts(t["cv"])).max()
                assert drift > 0.1, "a moved tooth was exported at T0"
        print("PASS  every STL round-trips; the moved crown is baked, the other is at T0")
    finally:
        STORE.drop(sid)


def test_a_base_that_cannot_be_built_is_refused_with_no_override():
    """allow_unsealed is gone, and that is the point.

    It existed to get past a base cap_and_close could not seal — which was a
    misdiagnosis. The scan was a healthy open shell (2101 open edges, 0
    non-manifold, one perimeter loop); the capping was the defect. A
    trim-and-extrude base closes by construction, so an unsealed one is a bug to
    fix rather than a condition to wave through, and an escape hatch here would
    hide exactly the failures the assertions exist to surface.
    """
    assert "allow_unsealed" not in api_core.ExportRequest.model_fields, \
        "allow_unsealed is back; a base that fails its own assertions must not ship"

    sid, reqs, _ = arch_session()
    try:
        api_core.cut(sid, reqs[0])
        # A trim margin far too tight to leave a band at all.
        try:
            api_core.build_export_bundle(
                sid, api_core.ExportRequest(trim_margin_mm=0.05))
            raise AssertionError("a base was exported from an impossible trim")
        except HTTPException as e:
            assert e.status_code == 422, e.status_code
            assert "cast base could not be built" in str(e.detail), e.detail
            assert "unsealed" not in str(e.detail).lower(), "an override was offered"
        print("PASS  an unbuildable base -> 422 naming the failure, with no override")
    finally:
        STORE.drop(sid)


def test_export_without_an_occlusal_plane_is_refused():
    """The trim measures from the occlusal ridge and the base is extruded along
    the occlusal normal, so without the plane both are guesses — and a guessed
    trim throws away real dentition. Same refusal as /cut, for the same reason."""
    sid, reqs, _ = arch_session()
    try:
        api_core.cut(sid, reqs[0])
        STORE.put(sid, "arch_frame", None)
        try:
            api_core.build_export_bundle(sid, api_core.ExportRequest())
            raise AssertionError("a cast base was built without an occlusal plane")
        except HTTPException as e:
            assert e.status_code == 409, e.status_code
            assert "occlusal" in str(e.detail).lower()
        print("PASS  /export without an occlusal plane -> 409, not a guessed trim")
    finally:
        STORE.drop(sid)


def test_export_without_a_cut_is_refused():
    sid, _, verts = build_session()
    try:
        STORE.put(sid, "arch_frame", occlusal_frame_for(verts))
        try:
            api_core.build_export_bundle(sid, api_core.ExportRequest())
        except HTTPException as e:
            assert e.status_code == 400, e.status_code
            print("PASS  /export with nothing extracted -> 400, not a corrupt ZIP")
            return
        raise AssertionError("exported a setup with no teeth")
    finally:
        STORE.drop(sid)


def test_teeth_listing_reports_committed_poses():
    sid, reqs, _ = arch_session()
    try:
        a = api_core.cut(sid, reqs[0])
        api_core.cut(sid, reqs[1])
        api_core.kinematics(sid, a["tooth_id"], api_core.KinematicsRequest(tip_deg=3.0))
        listing = api_core.list_teeth(sid)
        assert len(listing["teeth"]) == 2, listing
        by_id = {t["tooth_id"]: t for t in listing["teeth"]}
        assert by_id[a["tooth_id"]]["clinical"]["tip_deg"] == 3.0
        assert by_id[a["tooth_id"]]["matrix"] is not None
        json.dumps(listing)
        print(f"PASS  /teeth lists {len(listing['teeth'])} teeth with their committed poses")
    finally:
        STORE.drop(sid)


def test_cut_is_fast_on_a_full_size_arch():
    """The regression this pins is expensive and was already shipped.

    /cut used to cap_and_close and manifold-check the FULL base on every cut.
    Measured on synthetic arches: 14.2s at 96k faces, 29.7s at 204k, 43.9s at
    351k -- almost all of it inside cap_and_close. Returning removed-face
    indices instead of a rebuilt base removes it entirely. If someone
    reintroduces base capping here, this test is what catches it.
    """
    import time
    n = 320                                        # ~203k faces
    xs = np.linspace(-22.0, 22.0, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = 4.0 * np.exp(-(R ** 2) / (2 * (5.0 / 1.6) ** 2))
    Z -= 0.9 * np.exp(-((R - 5.0) ** 2) / (2 * 0.7 ** 2))
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            faces.append([a, c, b]); faces.append([b, c, d])
    faces = np.array(faces)

    edges = cg.directed_edges(faces)
    conc = cg.boundary_field(verts, faces, edges=edges)
    graph = cg.build_barrier_graph(verts, faces, conc, edges=edges)
    sid = STORE.create("lower")
    try:
        for key, val in (("verts", verts), ("faces", faces), ("edges", edges),
                         ("concavity", conc), ("graph", graph)):
            STORE.put(sid, key, val)
        STORE.put(sid, "arch_frame", occlusal_frame_for(verts))

        dist = cg.geodesic_from_seed(graph, verts, np.array([0.0, 0.0, 4.0]))
        mask = cg.largest_face_component(faces, cg.mask_from_distance(faces, dist, 12.0))
        req = api_core.CutRequest(
            vertex_ids=np.unique(faces[mask]).tolist(),
            mesial_pt=[-5.0, 0.0, 0.0], distal_pt=[5.0, 0.0, 0.0], root_length_mm=9.0)

        t0 = time.perf_counter()
        out = api_core.cut(sid, req)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, (
            f"/cut took {elapsed:.1f}s on {len(faces):,} faces. The full-base "
            f"cap_and_close is back — it measured 29.7s at this size.")
        print(f"PASS  /cut on {len(faces):,} faces took {elapsed:.2f}s "
              f"(was ~29.7s with full-base capping), removed "
              f"{len(out['removed_faces']):,} faces")
    finally:
        STORE.drop(sid)


if __name__ == "__main__":
    test_cut_without_occlusal_plane_is_refused()
    test_cut_anchors_the_pivot_in_bone()
    test_response_is_json_serialisable()
    test_kinematics_holds_cres_fixed()
    test_cut_removes_faces_and_caps_the_socket()
    test_socket_is_a_carved_cup_with_depth()
    test_the_displayed_socket_cup_is_the_one_that_gets_printed()
    test_cap_and_close_never_builds_the_exported_base()
    test_the_exported_file_survives_a_slicer_reading_it()
    test_two_cuts_are_disjoint_and_cumulative()
    test_recutting_an_extracted_tooth_is_refused()
    test_export_streams_a_zip_the_browser_can_download()
    test_exported_stls_parse_back_with_the_right_geometry()
    test_a_base_that_cannot_be_built_is_refused_with_no_override()
    test_export_without_an_occlusal_plane_is_refused()
    test_export_without_a_cut_is_refused()
    test_teeth_listing_reports_committed_poses()
    test_cut_is_fast_on_a_full_size_arch()
    print("\nALL CUT ENDPOINT TESTS PASSED")
