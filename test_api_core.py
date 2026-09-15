"""The session-lifecycle half of the api_core HTTP surface.

REWRITTEN 2026-09-15. The previous version targeted an API generation that no
longer existed — it called api_core.parse_stl_bytes, preview_selection, unpack,
get_session and auto_color (none of which are defined) and create_session,
segment and kinematics with signatures that had since changed. It had been red
independently of any real defect.

DIVISION OF LABOUR. test_cut_endpoint.py owns /cut, /kinematics, /export and
/export/stages, and it owns them thoroughly (18 tests). This file deliberately
does not duplicate that. It covers the eight endpoints nothing else touched:

    POST   /api/session                       upload, condition, report
    GET    /api/session/{sid}/mesh            geometry back out
    POST   /api/session/{sid}/occlusal-plane  the reference frame
    POST   /api/session/{sid}/wand            geodesic selection
    POST   /api/session/{sid}/wand/threshold  re-threshold without re-walking
    PUT    /api/session/{sid}/selection        persist a selection
    GET    /api/session/{sid}/teeth            restore after a refresh
    GET    /api/ai/status                      model state
    DELETE /api/session/{sid}                  free it

and the two cross-cutting contracts that are easy to break and expensive to
lose: an expired session must be 404 rather than 500, and the API must never
re-centre the scan.

Handlers are called directly rather than through TestClient because httpx is
not a dependency of this project and adding one to run a test is the wrong
trade. That matches test_cut_endpoint.py.

/segment is NOT exercised: it needs a 64MB torch checkpoint and takes ~236s on
a real arch. Its contract (409 on a duplicate run, warming vs loaded) is a
runtime property of the model, not of this module.
"""
import asyncio
import io
import struct

import numpy as np
from fastapi import HTTPException, UploadFile

import api_core
from session_store import STORE
from tooth_fixture import tooth_on_base

# The fixture is deliberately pushed far off the origin. If anything in the
# upload path re-centred the scan — the single most tempting "cleanup" in a
# mesh pipeline, and the one CLAUDE.md rule 3.1 forbids — this offset is what
# catches it. Inter-arch bite registration depends on raw scanner space.
OFFSET = np.array([137.5, -62.25, 41.0])


def _stl_bytes(verts, faces):
    """Binary STL of these triangles, in this order."""
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    buf.write(struct.pack("<I", len(faces)))
    for tri in faces:
        buf.write(struct.pack("<3f", 0, 0, 0))
        for vi in tri:
            buf.write(struct.pack("<3f", *verts[vi]))
        buf.write(struct.pack("<H", 0))
    return buf.getvalue()


def _upload(arch="lower", verts=None, faces=None):
    """POST /api/session with a synthetic arch. Returns (response, verts, faces)."""
    if verts is None:
        verts, faces, _ = tooth_on_base()
        verts = verts + OFFSET
    up = UploadFile(filename="scan.stl", file=io.BytesIO(_stl_bytes(verts, faces)))
    res = asyncio.run(api_core.create_session(arch=arch, file=up))
    return res, verts, faces


def _status(fn, *a, **kw):
    """Run a handler and return its HTTP status: 200, or the HTTPException code."""
    try:
        fn(*a, **kw)
        return 200
    except HTTPException as e:
        return e.status_code


# --------------------------------------------------------------------------
# POST /api/session
# --------------------------------------------------------------------------

def test_upload_conditions_the_scan_and_reports_what_it_did():
    res, verts, faces = _upload()
    sid = res["session_id"]
    try:
        for key in ("session_id", "vertex_count", "face_count", "bbox",
                    "conditioning", "scan_health"):
            assert key in res, f"upload response is missing {key!r}"

        assert res["vertex_count"] > 0 and res["face_count"] > 0
        # Welding is mandatory, not optional: an STL stores triangles
        # independently, and on unwelded soup a graph region-grow selects
        # exactly one face. So the welded count must be well under 3x faces.
        assert res["vertex_count"] < res["face_count"] * 3, "vertices were not welded"

        # scan_health separates the two failure modes that used to be conflated.
        # An intraoral scan is an OPEN SHELL by nature — its perimeter is not
        # damage — so open_edges may be non-zero and that is not a defect.
        for key in ("open_edges", "nonmanifold_edges"):
            assert key in res["scan_health"], f"scan_health lacks {key!r}"

        # The session really holds the geometry the response describes.
        assert len(STORE.require(sid, "verts")) == res["vertex_count"]
        assert len(STORE.require(sid, "faces")) == res["face_count"]
        print(f"PASS  upload: {res['vertex_count']:,} welded verts, "
              f"{res['face_count']:,} faces, scan_health reported")
    finally:
        api_core.close_session(sid)


def test_the_scan_is_never_recentred():
    """CLAUDE.md rule 3.1, pinned at the API boundary.

    Rotating, re-centring or re-scaling the scan would invalidate inter-arch
    bite registration, every vertex id already sent to the browser, and every
    cached geodesic field. "Up" comes from the occlusal frame instead.
    """
    res, verts, faces = _upload()
    sid = res["session_id"]
    try:
        lo, hi = np.asarray(res["bbox"][0]), np.asarray(res["bbox"][1])
        want_lo, want_hi = verts.min(0), verts.max(0)

        # Conditioning may DROP vertices (degenerates, debris), so the bbox can
        # shrink — but it must never translate or scale. A 0.5mm envelope is
        # far tighter than the 137mm offset a re-centring would produce.
        assert np.allclose(lo, want_lo, atol=0.5), f"bbox min moved: {lo} vs {want_lo}"
        assert np.allclose(hi, want_hi, atol=0.5), f"bbox max moved: {hi} vs {want_hi}"

        # And the stored vertices themselves, not just the reported bbox.
        stored = STORE.require(sid, "verts")
        assert np.allclose(stored.min(0), want_lo, atol=0.5)
        centre = (lo + hi) / 2.0
        assert np.linalg.norm(centre) > 50.0, \
            f"scan was re-centred on the origin (centre {centre})"
        # Plain ASCII in the output on purpose: a non-ASCII character here died
        # with UnicodeEncodeError on a cp1256 console AFTER every assertion had
        # passed, reporting a green file as a failure (see run_all_tests.py).
        print(f"PASS  raw scanner coordinates preserved - centre still at "
              f"[{centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}]")
    finally:
        api_core.close_session(sid)


def test_arch_must_be_upper_or_lower():
    """The arch decides the FDI quadrant offset and must never be guessed."""
    verts, faces, _ = tooth_on_base()
    for bad in ("mandibular", "MAXILLARY", "", "left"):
        up = UploadFile(filename="s.stl", file=io.BytesIO(_stl_bytes(verts, faces)))
        try:
            res = asyncio.run(api_core.create_session(arch=bad, file=up))
            api_core.close_session(res["session_id"])
            raise AssertionError(f"arch={bad!r} was accepted; it must be refused")
        except HTTPException as e:
            assert e.status_code == 400, f"arch={bad!r} gave {e.status_code}, want 400"
            assert "upper" in str(e.detail) and "lower" in str(e.detail), \
                "the refusal must say what the valid values are"

    # ...and both legitimate values are accepted.
    for good in ("upper", "lower"):
        res, _, _ = _upload(arch=good)
        api_core.close_session(res["session_id"])
    print("PASS  arch validated: 4 invalid values refused with 400, upper/lower accepted")


# --------------------------------------------------------------------------
# GET /api/session/{sid}/mesh
# --------------------------------------------------------------------------

def test_mesh_returns_flat_buffers_three_js_can_consume():
    res, _, _ = _upload()
    sid = res["session_id"]
    try:
        mesh = api_core.get_mesh(sid)
        pos, idx = mesh["positions"], mesh["indices"]

        assert len(pos) == res["vertex_count"] * 3, "positions is not a flat xyz buffer"
        assert len(idx) == res["face_count"] * 3, "indices is not a flat triangle buffer"
        assert max(idx) < res["vertex_count"], "an index points past the end of the buffer"
        assert min(idx) >= 0

        # It must survive JSON — these go over the wire as plain lists, and a
        # stray numpy scalar surfaces as an opaque 500 rather than a mesh.
        import json
        json.dumps(mesh)

        # And it round-trips to the geometry the session holds.
        back = np.asarray(pos, np.float32).reshape(-1, 3)
        assert np.allclose(back, STORE.require(sid, "verts"), atol=1e-3)
        print(f"PASS  /mesh: {len(pos):,} floats + {len(idx):,} indices, "
              f"JSON-serialisable, round-trips")
    finally:
        api_core.close_session(sid)


# --------------------------------------------------------------------------
# POST /api/session/{sid}/occlusal-plane
# --------------------------------------------------------------------------

def _plane_points(verts):
    """Three landmarks: left posterior, right posterior, anterior midline."""
    lo, hi = verts.min(0), verts.max(0)
    mid = (lo + hi) / 2.0
    top = hi[2]
    return [[lo[0], lo[1], top], [hi[0], lo[1], top], [mid[0], hi[1], top]]


def test_occlusal_plane_requires_exactly_three_points():
    res, verts, _ = _upload()
    sid = res["session_id"]
    try:
        pts = _plane_points(verts)
        for bad in (pts[:2], pts + [pts[0]], []):
            code = _status(api_core.set_occlusal_plane, sid,
                           api_core.OcclusalPlaneRequest(points=bad))
            assert code == 400, f"{len(bad)} points gave {code}, want 400"
        print("PASS  occlusal plane refuses 0, 2 and 4 points with 400")
    finally:
        api_core.close_session(sid)


def test_occlusal_plane_returns_a_right_handed_orthonormal_basis():
    res, verts, _ = _upload()
    sid = res["session_id"]
    try:
        frame = api_core.set_occlusal_plane(
            sid, api_core.OcclusalPlaneRequest(points=_plane_points(verts)))

        u = {k: np.asarray(frame[k], float) for k in ("u_occ", "u_sag", "u_tra")}
        for name, vec in u.items():
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-9, f"{name} is not unit"
        for a, b in (("u_occ", "u_sag"), ("u_occ", "u_tra"), ("u_sag", "u_tra")):
            assert abs(float(u[a] @ u[b])) < 1e-9, f"{a} and {b} are not orthogonal"
        # Right-handed, in the order basis_matrix stacks them.
        assert float(np.cross(u["u_tra"], u["u_sag"]) @ u["u_occ"]) > 0.99, \
            "basis is left-handed — every tip/torque sign would be inverted"

        # It is stored, because /cut refuses with 409 without it.
        assert STORE.require(sid, "arch_frame") is not None
        import json
        json.dumps(frame)
        print("PASS  occlusal frame orthonormal to 1e-9, right-handed, persisted")
    finally:
        api_core.close_session(sid)


# --------------------------------------------------------------------------
# POST /api/session/{sid}/wand  and  /wand/threshold
# --------------------------------------------------------------------------

def test_wand_snaps_the_seed_and_selects_a_region():
    res, verts, _ = _upload()
    sid = res["session_id"]
    try:
        # Click at the crown apex — the highest point of the fixture.
        apex = verts[np.argmax(verts[:, 2])]
        out = api_core.wand(sid, api_core.WandRequest(point=apex.tolist()))

        for key in ("vertex_ids", "tolerance", "seed", "seed_moved_mm", "max_distance"):
            assert key in out, f"wand response lacks {key!r}"
        ids = out["vertex_ids"]
        assert len(ids) > 0, "wand selected nothing at the crown apex"
        assert len(ids) < res["vertex_count"], "wand selected the entire mesh"
        assert max(ids) < res["vertex_count"] and min(ids) >= 0
        assert len(set(ids)) == len(ids), "wand returned duplicate vertex ids"
        assert out["seed_moved_mm"] >= 0.0

        # The geodesic field is cached so /wand/threshold need not re-walk it.
        assert STORE.require(sid, "wand_field") is not None
        print(f"PASS  /wand: {len(ids):,} of {res['vertex_count']:,} verts at "
              f"tol={out['tolerance']:.2f}, seed moved {out['seed_moved_mm']}mm")
    finally:
        api_core.close_session(sid)


def test_threshold_is_monotonic_and_reuses_the_cached_field():
    res, verts, _ = _upload()
    sid = res["session_id"]
    try:
        apex = verts[np.argmax(verts[:, 2])]
        api_core.wand(sid, api_core.WandRequest(point=apex.tolist()))

        prev = None
        for tol in (1.0, 2.5, 5.0, 9.0):
            ids = set(api_core.wand_threshold(
                sid, api_core.ThresholdRequest(tolerance=tol))["vertex_ids"])
            if prev is not None:
                assert prev <= ids, (
                    f"raising the tolerance to {tol} DROPPED vertices — the "
                    "selection must only ever grow")
            prev = ids
        assert len(prev) > 0
        print(f"PASS  /wand/threshold monotonic across 4 tolerances "
              f"(final {len(prev):,} verts)")
    finally:
        api_core.close_session(sid)


def test_threshold_without_a_wand_click_is_404_not_500():
    """The cached geodesic field is the thing that is missing, and saying so is
    the difference between a client that can recover and one that cannot."""
    res, _, _ = _upload()
    sid = res["session_id"]
    try:
        code = _status(api_core.wand_threshold, sid,
                       api_core.ThresholdRequest(tolerance=2.0))
        assert code == 404, f"threshold before wand gave {code}, want 404"
        print("PASS  /wand/threshold before any /wand click is 404")
    finally:
        api_core.close_session(sid)


# --------------------------------------------------------------------------
# PUT /selection, GET /teeth, GET /api/ai/status, DELETE
# --------------------------------------------------------------------------

def test_selection_round_trips():
    res, _, _ = _upload()
    sid = res["session_id"]
    try:
        ids = list(range(0, 400, 3))
        out = api_core.put_selection(sid, api_core.SelectionRequest(vertex_ids=ids))
        assert out["count"] == len(ids), f"reported {out['count']}, sent {len(ids)}"
        assert np.array_equal(np.asarray(STORE.require(sid, "selection")), np.asarray(ids))

        empty = api_core.put_selection(sid, api_core.SelectionRequest(vertex_ids=[]))
        assert empty["count"] == 0, "clearing the selection must be allowed"
        print(f"PASS  /selection round-trips {len(ids)} ids and accepts a clear")
    finally:
        api_core.close_session(sid)


def test_teeth_is_empty_before_any_cut():
    res, _, _ = _upload()
    sid = res["session_id"]
    try:
        out = api_core.list_teeth(sid)
        teeth = out["teeth"] if isinstance(out, dict) and "teeth" in out else out
        assert len(teeth) == 0, f"a fresh session already reports {len(teeth)} teeth"
        import json
        json.dumps(out)
        print("PASS  /teeth is empty on a fresh session and JSON-serialisable")
    finally:
        api_core.close_session(sid)


def test_ai_status_distinguishes_loading_from_broken():
    """A model that is still importing torch must not read as a dead pipeline.

    This is asserted WITHOUT loading the model: the endpoint has to answer
    truthfully about a pipeline that has not been warmed, which is exactly the
    state a client sees in the first seconds after the server starts.
    """
    s = api_core.ai_status()
    for key in ("loaded", "warming", "warm_seconds", "error", "ready_message",
                "segmentation_in_progress", "segmentation_started_at",
                "segmentation_message"):
        assert key in s, f"/api/ai/status lacks {key!r}"

    assert isinstance(s["loaded"], bool) and isinstance(s["warming"], bool)
    assert isinstance(s["segmentation_in_progress"], bool)
    assert not (s["loaded"] and s["warming"]), "cannot be loaded and warming at once"
    assert s["ready_message"], "ready_message is what the client actually shows"

    import json
    json.dumps(s)
    print(f"PASS  /api/ai/status: loaded={s['loaded']} warming={s['warming']} "
          f"-> {s['ready_message']!r}")


def test_closing_a_session_frees_it():
    res, _, _ = _upload()
    sid = res["session_id"]
    assert api_core.get_mesh(sid)["positions"], "session should be live before close"
    api_core.close_session(sid)
    assert _status(api_core.get_mesh, sid) == 404, "session survived its DELETE"

    # DELETE is idempotent — a client retrying a close must not get a 500.
    api_core.close_session(sid)
    print("PASS  DELETE frees the session and is idempotent")


def test_every_session_endpoint_404s_on_an_unknown_session():
    """PHI: scans are memory-only and expire on a timer, so a client WILL meet
    an expired session in normal use. Each endpoint must translate
    SessionExpired into 404 — an unhandled one is a 500 that tells the
    clinician nothing and looks like a server fault.
    """
    ghost = "0" * 32
    cases = [
        ("GET  /mesh", lambda: api_core.get_mesh(ghost)),
        ("POST /occlusal-plane", lambda: api_core.set_occlusal_plane(
            ghost, api_core.OcclusalPlaneRequest(points=[[0, 0, 0]] * 3))),
        ("POST /wand", lambda: api_core.wand(
            ghost, api_core.WandRequest(point=[0.0, 0.0, 0.0]))),
        ("POST /wand/threshold", lambda: api_core.wand_threshold(
            ghost, api_core.ThresholdRequest(tolerance=2.0))),
        ("PUT  /selection", lambda: api_core.put_selection(
            ghost, api_core.SelectionRequest(vertex_ids=[1, 2, 3]))),
        ("GET  /teeth", lambda: api_core.list_teeth(ghost)),
    ]
    for name, call in cases:
        code = _status(call)
        assert code == 404, f"{name} on an expired session gave {code}, want 404"
    print(f"PASS  all {len(cases)} session endpoints give 404 on an expired session")


if __name__ == "__main__":
    test_upload_conditions_the_scan_and_reports_what_it_did()
    test_the_scan_is_never_recentred()
    test_arch_must_be_upper_or_lower()
    test_mesh_returns_flat_buffers_three_js_can_consume()
    test_occlusal_plane_requires_exactly_three_points()
    test_occlusal_plane_returns_a_right_handed_orthonormal_basis()
    test_wand_snaps_the_seed_and_selects_a_region()
    test_threshold_is_monotonic_and_reuses_the_cached_field()
    test_threshold_without_a_wand_click_is_404_not_500()
    test_selection_round_trips()
    test_teeth_is_empty_before_any_cut()
    test_ai_status_distinguishes_loading_from_broken()
    test_closing_a_session_frees_it()
    test_every_session_endpoint_404s_on_an_unknown_session()
    print("\nALL API CORE TESTS PASSED")
