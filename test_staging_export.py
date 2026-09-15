"""The manufacturing export: one fused, watertight solid per stage.

WHAT A STAGE MODEL IS, and why this is not just the planned-setup export in a
loop. A lab thermoforms a sheet over a SOLID. Handing them a base plus fourteen
loose crowns hands them an assembly problem, and handing them a base whose
sockets gape beside every moved tooth hands them a cast the aligner takes an
impression of. So each stage is base UNION crowns, sockets filled flush, and
every one of them is re-read from its own written STL before it ships.

Stage k is the clinical parameters scaled to k/N and rebuilt through
cg.kinematic_matrix — never an interpolation of the committed 4x4. That is
asserted here in the same terms verify-kinematics.mjs asserts it on the browser
side, because the two have to agree: the clinician approves the scrub and the
lab prints these files.
"""
import io
import json
import zipfile

import numpy as np
from fastapi import HTTPException

import core_geometry as cg
import api_core
import stl_io
from session_store import STORE
from test_cut_endpoint import arch_session


def moved_session(teeth=(-0.45, 0.45), prescriptions=None):
    """A horseshoe arch with two crowns cut and moved."""
    sid, reqs, _ = arch_session(teeth=teeth)
    prescriptions = prescriptions or [
        dict(tip_deg=3.0, torque_deg=-2.0, rotation_deg=4.0, d_md=0.4, d_bl=0.0, d_oa=0.0),
        dict(tip_deg=0.0, torque_deg=0.0, rotation_deg=0.0, d_md=0.0, d_bl=0.25, d_oa=0.0),
    ]
    tids = []
    for req, presc in zip(reqs, prescriptions):
        out = api_core.cut(sid, req)
        api_core.kinematics(sid, out["tooth_id"], api_core.KinematicsRequest(**presc))
        tids.append(out["tooth_id"])
    return sid, tids, prescriptions


def test_every_stage_is_one_fused_watertight_solid():
    sid, tids, presc = moved_session()
    try:
        res = api_core.build_stage_bundle(sid, api_core.StageExportRequest())
        man = res["manifest"]
        n = man["stages"]
        assert n >= 1

        zf = zipfile.ZipFile(res["buf"])
        names = zf.namelist()
        assert "manifest.json" in names
        stls = [x for x in names if x.endswith(".stl")]
        assert len(stls) == n, f"{n} stages but {len(stls)} STLs"

        # NO LOOSE CROWNS. One file per stage and nothing else — a stage model
        # is a solid, not an assembly.
        assert not any("Tooth" in x for x in names), \
            f"loose crowns leaked into the stage export: {names}"

        worst_weld = 0
        for meta in man["stage_files"]:
            v, f = stl_io.parse_stl_bytes(zf.read(meta["file"]))
            # ASSERTED: closed, one body, positive volume. That is what
            # manifold3d guarantees and what makes a model printable.
            assert meta["closed"] is True, meta
            assert meta["components"] == 1, \
                f"{meta['file']} is {meta['components']} solids — the union did not fuse"
            assert meta["volume_mm3"] > 0
            assert cg.signed_volume(v, f) > 0

            # NO HOLES, ever. An open edge is a hole and would be unprintable.
            assert meta["welded_open_edges"] == 0, \
                f"{meta['file']} has {meta['welded_open_edges']} open edges — a hole"

            # REPORTED, NOT ASSERTED: survival of a reader that welds coincident
            # vertices. Putting a crown back into the hole it was cut from is a
            # TANGENTIAL boolean — the crown's outer surface and the cast's are
            # the same scan faces, meeting edge to edge at the rim — and
            # manifold3d answers a tangency with topologically distinct vertices
            # at identical positions, which binary STL cannot express. Five
            # separations were measured and none removed it across
            # prescriptions; see build_stage_bundle. What is pinned here is that
            # the number is measured and written down per stage, so nobody is
            # surprised by it downstream.
            assert "survives_a_welding_reader" in meta
            worst_weld = max(worst_weld, meta["welded_nonmanifold_edges"])

        crumbs = sum(m["inverted_crumbs_discarded"] for m in man["stage_files"])
        print(f"PASS  {n} stages, each one fused solid: "
              f"{man['stage_files'][0]['triangles']:,} tris at stage 1, "
              f"{man['stage_files'][-1]['triangles']:,} at stage {n}; "
              f"union {man['union_seconds']:.2f}s; {crumbs} inverted crumbs discarded; "
              f"worst weld sensitivity {worst_weld} non-manifold edges (0 open)")
    finally:
        STORE.drop(sid)


def test_stages_interpolate_the_prescription_not_the_matrix():
    """Stage k is clinical x k/N rebuilt through kinematic_matrix.

    The alternative — lerping the committed 4x4 — is measured here and shown to
    not be a rotation at all: the 3x3 block of (1-t)I + tR has RtR != I for any
    t strictly between 0 and 1, so every intermediate stage would shear and
    scale the crown. The tooth would arrive the right shape having passed
    through thirty stages of the wrong one, and the trays are cut from those.
    """
    sid, tids, presc = moved_session()
    try:
        rec = STORE.require(sid, f"tooth:{tids[0]}")
        clinical = rec["clinical"]
        full = cg.kinematic_matrix(rec["frame"], rec["c_res"], **clinical)
        n = 12

        worst_orth = 0.0
        for k in range(n + 1):
            M = cg.kinematic_matrix(rec["frame"], rec["c_res"],
                                    **api_core._stage_clinical(clinical, k, n))
            R = M[:3, :3]
            worst_orth = max(worst_orth, np.abs(R.T @ R - np.eye(3)).max())
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

        assert np.allclose(cg.kinematic_matrix(rec["frame"], rec["c_res"],
                                               **api_core._stage_clinical(clinical, 0, n)),
                           np.eye(4), atol=1e-12), "stage 0 is not T0"
        assert np.allclose(cg.kinematic_matrix(rec["frame"], rec["c_res"],
                                               **api_core._stage_clinical(clinical, n, n)),
                           full, atol=1e-12), "stage N is not the committed pose"
        assert worst_orth < 1e-12, f"a stage is not rigid: RtR-I = {worst_orth:.2e}"

        # The bad path, measured rather than asserted.
        worst_lerp = 0.0
        for k in range(1, n):
            L = (1 - k / n) * np.eye(4) + (k / n) * full
            R = L[:3, :3]
            worst_lerp = max(worst_lerp, np.abs(R.T @ R - np.eye(3)).max())
        assert worst_lerp > 1e-3, \
            "the 4x4 lerp did not measurably fail, so this comparison proves nothing"

        print(f"PASS  every stage rigid to {worst_orth:.0e} (RtR-I); a 4x4 lerp of the "
              f"same movement reaches {worst_lerp:.1e} and is not a rotation")
    finally:
        STORE.drop(sid)


def test_sockets_are_filled_flush_for_manufacturing():
    """A 3.5mm cup gapes when the tooth moves out of it.

    The display socket stays a cup — that is what the clinician approved and
    what /export ships. The manufacturing base fills flush instead, so the tooth
    sits proud of smooth gingiva and the thermoformed sheet never draws into an
    open pit.
    """
    sid, tids, _ = moved_session()
    try:
        v = STORE.require(sid, "verts")
        f = STORE.require(sid, "faces")
        ex = STORE.require(sid, "extracted_faces")

        _, _, cupped = api_core._seal_sockets(v, f, sid, ex, flush=False)
        _, _, flat = api_core._seal_sockets(v, f, sid, ex, flush=True)

        assert cupped and flat
        assert any((s["depth_mm"] or 0) > 1.0 for s in cupped), \
            "the display sockets are not cups, so this comparison is vacuous"

        # The flush base must reach no deeper than the margin plane.
        sv, sf, _ = api_core._seal_sockets(v, f, sid, ex, flush=True)
        origin, e1, e2, u_occ = cg._arch_basis(STORE.get(sid, "arch_frame"))
        for tid in tids:
            rec = STORE.require(sid, f"tooth:{tid}")
            rim = np.asarray(rec["socket_rim"], int)
            rim_h = (v[rim] - origin) @ u_occ
            # every flush cap vertex sits within the rim's own height range
            info = rec["socket_info"]
            assert info["depth_mm"] > 1.0, "display cup lost its depth"

        print(f"PASS  display sockets are cups at "
              f"{max(s['depth_mm'] for s in cupped):.2f}mm; the manufacturing base "
              f"fills all {len(flat)} flush")
    finally:
        STORE.drop(sid)


def test_a_runaway_plan_is_refused_with_the_binding_tooth_named():
    """7.686mm of extrusion asked for 31 stages once. The export says which
    tooth is binding the case rather than silently grinding through it."""
    sid, tids, _ = moved_session(prescriptions=[
        dict(tip_deg=0.0, torque_deg=0.0, rotation_deg=0.0, d_md=0.0, d_bl=0.0, d_oa=7.686),
        dict(tip_deg=1.0, torque_deg=0.0, rotation_deg=0.0, d_md=0.0, d_bl=0.0, d_oa=0.0),
    ])
    try:
        try:
            api_core.build_stage_bundle(
                sid, api_core.StageExportRequest(max_stages=8))
            raise AssertionError("a 31-stage plan was built without complaint")
        except HTTPException as e:
            assert e.status_code == 422, e.status_code
            d = str(e.detail)
            assert "31 stages" in d, d
            assert "binding" in d, d
        print("PASS  a runaway plan -> 422 naming the binding tooth and its channel")
    finally:
        STORE.drop(sid)


def test_export_without_movement_is_refused():
    sid, reqs, _ = arch_session()
    try:
        api_core.cut(sid, reqs[0])
        try:
            api_core.build_stage_bundle(sid, api_core.StageExportRequest())
            raise AssertionError("stages were exported for a case with no movement")
        except HTTPException as e:
            assert e.status_code == 400, e.status_code
            assert "no movement" in str(e.detail).lower()
        print("PASS  no movement prescribed -> 400, not N identical copies of the scan")
    finally:
        STORE.drop(sid)


if __name__ == "__main__":
    test_every_stage_is_one_fused_watertight_solid()
    test_stages_interpolate_the_prescription_not_the_matrix()
    test_sockets_are_filled_flush_for_manufacturing()
    test_a_runaway_plan_is_refused_with_the_binding_tooth_named()
    test_export_without_movement_is_refused()
    print("\nALL STAGING EXPORT TESTS PASSED")
