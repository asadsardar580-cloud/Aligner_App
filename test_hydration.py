"""Case reload: can a client that kept only the session id rebuild the case?

A browser refresh destroys every piece of React state. The server, however,
already holds almost everything — the audit that motivated this found that the
blocker was EXPOSURE, not storage: arch_frame, labels, per-tooth crown geometry,
fdi and root_length_mm were all in the session and reachable by no endpoint.

What is pinned here is that the restore path is COMPLETE, not that it is fast:
given a session id and nothing else, every piece of the case comes back, and
comes back IDENTICAL to what the cut produced. A hydration that silently
returns a slightly different crown is worse than one that fails.

Scope note: this is browser-refresh hydration. Surviving a SERVER restart is a
different problem — SessionStore is in-memory by PHI design — and is not
claimed here.
"""
import numpy as np
from fastapi import HTTPException

import api_core
import arch_frame
import core_geometry as cg
from session_store import STORE
from tooth_fixture import flat_molar_on_base


def build_cut_session():
    """A session with an occlusal plane and one tooth already extracted."""
    v, f, crown_r = flat_molar_on_base()
    sid = STORE.create("lower")
    STORE.put(sid, "verts", v)
    STORE.put(sid, "faces", f)
    STORE.put(sid, "scan_health", cg.manifold_report(f))
    edges = cg.directed_edges(f)
    conc = cg.boundary_field(v, f, edges=edges)
    STORE.put(sid, "edges", edges)
    STORE.put(sid, "concavity", conc)
    STORE.put(sid, "graph", cg.build_barrier_graph(v, f, conc, edges=edges))

    lo, hi = v.min(0), v.max(0)
    mid = (lo + hi) / 2.0
    frame = arch_frame.fit_occlusal_frame(
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [mid[0], hi[1], hi[2]], v.mean(axis=0))
    STORE.put(sid, "arch_frame", frame)

    # Select the crown: every vertex above the base plane, within the dome.
    centre = np.array([0.0, 0.0])
    r = np.linalg.norm(v[:, :2] - centre, axis=1)
    sel = np.where((r < crown_r * 0.95) & (v[:, 2] > v[:, 2].min() + 0.35))[0]

    res = api_core.cut(sid, api_core.CutRequest(
        vertex_ids=sel.tolist(),
        mesial_pt=[-crown_r, 0.0, float(v[:, 2].max())],
        distal_pt=[crown_r, 0.0, float(v[:, 2].max())],
        root_length_mm=9.0))
    return sid, res, v, f


def test_session_summary_reports_progress_without_payloads():
    sid, cut, v, f = build_cut_session()
    try:
        s = api_core.get_session(sid)
        assert s["session_id"] == sid
        assert s["arch"] == "lower"
        assert s["vertex_count"] == len(v)
        assert s["face_count"] == len(f)
        assert s["has_occlusal_frame"] is True, "the plane was established; summary must say so"
        assert s["has_segmentation"] is False, "segmentation never ran here"
        assert s["tooth_count"] == 1
        assert s["extracted_face_count"] > 0
        import json
        json.dumps(s)
        print(f"PASS  summary: {s['tooth_count']} tooth, frame={s['has_occlusal_frame']}, "
              f"{s['extracted_face_count']} faces extracted")
    finally:
        api_core.close_session(sid)


def test_occlusal_frame_survives_a_refresh_bit_for_bit():
    """Re-clicking three landmarks would produce a DIFFERENT frame, silently
    re-basing every long-axis reconciliation in the case. It must be fetchable."""
    sid, cut, v, f = build_cut_session()
    try:
        stored = arch_frame.to_json(STORE.get(sid, "arch_frame"))
        got = api_core.get_occlusal_frame(sid)
        for k in ("u_occ", "u_sag", "u_tra"):
            assert np.allclose(np.asarray(got[k], float), np.asarray(stored[k], float), atol=0), \
                f"{k} came back changed"
        print("PASS  occlusal frame round-trips exactly (u_occ/u_sag/u_tra)")
    finally:
        api_core.close_session(sid)


def test_frame_and_labels_404_when_never_established():
    """404 with a reason, not an empty success. A client that cannot tell
    "no plane yet" from "plane is all zeros" will restore a broken case."""
    v, f, _ = flat_molar_on_base()
    sid = STORE.create("lower")
    STORE.put(sid, "verts", v)
    STORE.put(sid, "faces", f)
    try:
        for name, call in (("frame", lambda: api_core.get_occlusal_frame(sid)),
                           ("labels", lambda: api_core.get_labels(sid))):
            try:
                call()
                raise AssertionError(f"/{name} returned success with nothing established")
            except HTTPException as e:
                assert e.status_code == 404, f"/{name} gave {e.status_code}, want 404"
                assert e.detail, f"/{name} 404 carried no reason"
        print("PASS  /frame and /labels 404 with a reason before they exist")
    finally:
        api_core.close_session(sid)


def test_labels_round_trip():
    sid, cut, v, f = build_cut_session()
    try:
        fake = np.zeros(len(v), dtype=int)
        fake[: len(v) // 3] = 36
        STORE.put(sid, "labels", fake)
        got = api_core.get_labels(sid)
        assert len(got["labels"]) == len(v), "label array length must match the mesh"
        assert np.array_equal(np.asarray(got["labels"]), fake), "labels came back altered"
        assert got["jaw"] == "lower"
        print(f"PASS  /labels round-trips {len(v):,} per-vertex FDI labels")
    finally:
        api_core.close_session(sid)


def test_teeth_light_payload_is_unchanged():
    """Backward compatibility: the default shape is what it always was."""
    sid, cut, v, f = build_cut_session()
    try:
        light = api_core.list_teeth(sid)
        t = light["teeth"][0]
        assert set(t.keys()) == {"tooth_id", "c_res", "clinical", "matrix", "frame"}, \
            f"default payload grew new keys: {sorted(t.keys())}"
        assert "crown" not in t and "removed_faces" not in t
        print("PASS  /teeth default payload unchanged (5 keys, no geometry)")
    finally:
        api_core.close_session(sid)


def test_teeth_with_geometry_restores_the_crown_exactly():
    """The whole point. A restored crown must be the crown that was cut."""
    sid, cut, v, f = build_cut_session()
    try:
        full = api_core.list_teeth(sid, geometry=True)
        t = full["teeth"][0]
        assert t["tooth_id"] == cut["tooth_id"]

        # Crown geometry identical to what /cut handed the client.
        cut_pos = np.asarray(cut["crown"]["positions"], np.float32)
        hyd_pos = np.asarray(t["crown"]["positions"], np.float32)
        assert cut_pos.shape == hyd_pos.shape, "restored crown has a different vertex count"
        assert np.array_equal(cut_pos, hyd_pos), "restored crown geometry differs from the cut"
        assert np.array_equal(np.asarray(cut["crown"]["indices"]),
                              np.asarray(t["crown"]["indices"])), "crown topology differs"

        # The identity the lab manifest depends on.
        assert t["fdi"] == cut["fdi"], "FDI changed across a reload"
        assert t["root_length_mm"] == cut["root_length_mm"], "root length changed across a reload"
        assert np.allclose(t["c_res"], cut["c_res"], atol=0), "C_res moved across a reload"

        # Enough to rebuild the cast: which faces went, and the socket cup.
        assert sorted(t["removed_faces"]) == sorted(cut["removed_faces"]), \
            "restored extraction mask differs from the cut"
        assert len(t["socket_cap"]["faces"]) == len(cut["socket_cap"]["faces"]), \
            "socket cup triangle count differs"
        assert t["root_cone"]["vertices"], "no virtual root returned"
        assert t["root_cone"]["length_mm"] == cut["root_length_mm"], \
            "restored root cone has a different length from the one that was cut"

        import json
        json.dumps(full)
        print(f"PASS  crown restored exactly: {len(hyd_pos)//3:,} verts, FDI {t['fdi']}, "
              f"{len(t['removed_faces'])} faces, {len(t['socket_cap']['faces'])} cup tris")
    finally:
        api_core.close_session(sid)


def test_committed_movement_survives_a_refresh():
    sid, cut, v, f = build_cut_session()
    tid = cut["tooth_id"]
    try:
        api_core.kinematics(sid, tid, api_core.KinematicsRequest(
            tip_deg=6.5, torque_deg=-3.0, rotation_deg=2.0, d_md=0.4, d_bl=-0.2, d_oa=0.1))

        t = api_core.list_teeth(sid, geometry=True)["teeth"][0]
        c = t["clinical"]
        assert c is not None, "a committed prescription vanished on reload"
        assert abs(c["tip_deg"] - 6.5) < 1e-9 and abs(c["d_md"] - 0.4) < 1e-9
        assert t["matrix"] is not None and len(t["matrix"]) == 16

        # The matrix must be REBUILDABLE from the prescription, not just stored:
        # that is what lets a reloaded client re-derive every intermediate stage.
        rebuilt = cg.kinematic_matrix(
            STORE.get(sid, f"tooth:{tid}")["frame"], np.asarray(t["c_res"], float),
            c["tip_deg"], c["torque_deg"], c["rotation_deg"], c["d_md"], c["d_bl"], c["d_oa"])
        assert np.allclose(np.asarray(t["matrix"], float).reshape(4, 4), rebuilt, atol=1e-12), \
            "stored matrix does not match the one its own prescription rebuilds"
        print("PASS  prescription and matrix survive reload, and agree to 1e-12")
    finally:
        api_core.close_session(sid)


def test_hydration_endpoints_404_on_an_expired_session():
    ghost = "0" * 32
    for name, call in (("GET /session", lambda: api_core.get_session(ghost)),
                       ("GET /frame", lambda: api_core.get_occlusal_frame(ghost)),
                       ("GET /labels", lambda: api_core.get_labels(ghost)),
                       ("GET /teeth", lambda: api_core.list_teeth(ghost, geometry=True))):
        try:
            call()
            raise AssertionError(f"{name} succeeded on an expired session")
        except HTTPException as e:
            assert e.status_code == 404, f"{name} gave {e.status_code}, want 404"
    print("PASS  all 4 hydration endpoints 404 on an expired session")


if __name__ == "__main__":
    test_session_summary_reports_progress_without_payloads()
    test_occlusal_frame_survives_a_refresh_bit_for_bit()
    test_frame_and_labels_404_when_never_established()
    test_labels_round_trip()
    test_teeth_light_payload_is_unchanged()
    test_teeth_with_geometry_restores_the_crown_exactly()
    test_committed_movement_survives_a_refresh()
    test_hydration_endpoints_404_on_an_expired_session()
    print("\nALL HYDRATION TESTS PASSED")
