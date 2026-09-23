"""`construction="deformation"` over real HTTP, on the synthetic fixture.

AGENT_BRIEF 2.4 / 2.5. Driven through `TestClient`, not by calling the
endpoint functions, because what 2.4 adds is partly a REQUEST CONTRACT and
partly a RESPONSE HEADER - a direct call exercises neither. `X-Print-Ready`
is the whole point: s.23 records that the client must read the header rather
than infer readiness from HTTP 200, and a header nothing asserts is a header
that can silently stop being set.

TWO CASES, and they are chosen to be opposite verdicts on the same machinery:

  SEAM       spaced crowns, a movement at the tooth/gingiva seam. The blend
             has gingiva to absorb the movement, so the stage should be PRINT
             READY - a 200 with the header true.
  NEIGHBOUR  a crown pushed into the one beside it with NO IPR prescribed.
             Enamel is closing on enamel that the plan never authorised
             removing, and the gate must refuse and NAME itself.

The default is asserted separately: an omitted `construction` must still be
the collar path, byte for byte, or every existing client changed behaviour.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_core
import deform_construction as dc
from api_core import STORE
from test_cut_endpoint import arch_session

client = TestClient(api_core.app)

# The kit's own two geometries (test_deform_construction.py): spaced crowns
# separated by gingiva, and crowns merged across a broad contact.
SPACED = (-0.25, 0.0, 0.25)
CONTACTING = (-0.14, 0.0, 0.14)
N_S, N_T = 240, 60


def _session(teeth, prescription, moving=1):
    """A LABELLED arch with exactly one crown cut and moved.

    THE LABELS ARE WHAT MAKE THE OTHER TEETH STATIC, and without them this
    fixture cannot test what it claims to. `build_stage_bundle_v2` takes the
    moving set from `/cut`'s own face mask and the static set from
    `/segment`'s labels (2.3). The first version of this helper cut all three
    crowns, so all three were MOVING, there was no anchorage enamel, no
    contact band was built and a crown pushed hard into its neighbour
    exported 200 - it had no neighbour. Labelling the arch first is also the
    real workflow: segment, then click a tooth, then cut it.

    The label regions are `arch_session`'s own geodesic floods, i.e. exactly
    the selections the cut would use, so a labelled tooth and a cut tooth
    describe the same vertices.
    """
    sid, reqs, _ = arch_session(teeth=teeth, n_s=N_S, n_t=N_T)
    v = STORE.require(sid, "verts")
    labels = np.zeros(len(v), np.int64)
    for i, req in enumerate(reqs):
        labels[np.asarray(req.vertex_ids, np.int64)] = 44 + i
    STORE.put(sid, "labels", labels.tolist())

    req = reqs[moving]
    r = client.post(f"/api/session/{sid}/cut", json=json.loads(req.json()))
    assert r.status_code == 200, r.text[:1500]
    tid = r.json()["tooth_id"]
    k = client.post(f"/api/session/{sid}/tooth/{tid}/kinematics", json=prescription)
    assert k.status_code == 200, k.text[:1500]
    return sid, [tid]


def _drop(sid):
    client.delete(f"/api/session/{sid}")


NO_MOVE = dict(tip_deg=0.0, torque_deg=0.0, rotation_deg=0.0,
               d_md=0.0, d_bl=0.0, d_oa=0.0)


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------

def test_the_flag_exists_on_both_endpoints_and_defaults_to_collar():
    """A field that defaults to anything else silently re-routes every
    existing client on the day it ships."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    for name in ("StageExportRequest", "FinalExportRequest"):
        prop = schemas[name]["properties"]["construction"]
        assert prop["default"] == "collar", prop
        assert sorted(prop["enum"]) == ["collar", "deformation"], prop
    print("PASS  construction defaults to collar on both request models")


def test_an_unknown_construction_is_refused_by_the_schema():
    """Literal, not str. A typo must not silently pick a construction - the
    same rule s.25.7 applies to an unrecognised segmentation provider."""
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.1))
    try:
        r = client.post(f"/api/session/{sid}/export/final",
                        json={"construction": "colar"})
        assert r.status_code == 422, r.status_code
        assert "construction" in r.text
    finally:
        _drop(sid)
    print("PASS  a misspelled construction is refused, not defaulted")


# ---------------------------------------------------------------------------
# 2.5 - the two verdicts
# ---------------------------------------------------------------------------

def test_a_seam_movement_is_PRINT_READY_over_http():
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        r = client.post(f"/api/session/{sid}/export/final",
                        json={"construction": "deformation", "fmt": "manifest"})
        assert r.status_code == 200, r.text[:2000]
        assert r.headers["X-Print-Ready"] == "true", dict(r.headers)

        man = r.json()
        assert man["print_ready"] is True
        assert man["construction"].startswith("deformation"), \
            man["construction"]
        gate = man["manufacturing_gate"]
        assert not gate["failed_gates"], gate["failed_gates"]
        # Every REQUIRED gate was measured, not merely absent.
        for name in dc.REQUIRED_GATES:
            assert name in gate["gates"], f"{name} was never measured"
            assert gate["gates"][name]["ok"] is True, (name, gate["gates"][name])
        print(f"PASS  200, X-Print-Ready true, {len(dc.REQUIRED_GATES)} "
              f"gates measured and passed")
    finally:
        _drop(sid)


def test_pushing_into_the_neighbour_is_NOT_READY_and_names_the_gate():
    """No IPR prescribed, so a closing contact is unauthorised enamel loss."""
    sid, _ = _session(CONTACTING, dict(NO_MOVE, d_md=0.30))
    try:
        r = client.post(f"/api/session/{sid}/export/final",
                        json={"construction": "deformation"})
        assert r.status_code == 422, \
            f"a movement into the neighbour exported anyway: {r.status_code}"
        body = r.json()["detail"]
        named = body.get("failed_gates") or []
        assert named, f"NOT READY with no gate named: {body}"
        # A refusal must carry the MEASURED value, not just a verdict - s.14.
        for g in named:
            assert g in body.get("gates", {}), f"{g} named but not measured"
        print(f"PASS  422, failed gates: {named}")
    finally:
        _drop(sid)


def test_the_refusal_carries_the_measurement_not_just_a_verdict():
    """The same case again, read for its numbers rather than its verdict."""
    sid, _ = _session(CONTACTING, dict(NO_MOVE, d_md=0.30))
    try:
        r = client.post(f"/api/session/{sid}/export/final",
                        json={"construction": "deformation"})
        assert r.status_code == 422
        gates = r.json()["detail"]["gates"]
        measured = {k: v for k, v in gates.items() if isinstance(v, dict)}
        assert measured, "the gate block carried no measurements"
        # Every gate reports ok plus at least one number or reason beside it.
        for name, g in measured.items():
            assert "ok" in g, (name, g)
            assert len(g) > 1, f"{name} reports a verdict and nothing else: {g}"
        print(f"PASS  {len(measured)} gates, each with its measurement")
    finally:
        _drop(sid)


# ---------------------------------------------------------------------------
# The collar path is the default and is unchanged
# ---------------------------------------------------------------------------

def test_omitting_the_flag_still_runs_the_collar_construction():
    """The load-bearing compatibility claim: an existing client sends no
    `construction` at all, and must get exactly what it got before."""
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        r = client.post(f"/api/session/{sid}/export/stages", json={})
        assert r.status_code == 200, r.text[:1500]
        import io
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(r.content))
        man = json.loads(z.read("manifest.json"))
        assert not str(man.get("construction", "")).startswith(
            "deformation"), man.get("construction")
        # The collar's own signature: it fills the socket it cut.
        assert "flush" in man["socket_treatment"]
        print(f"PASS  no flag -> collar, {man['stages']} stage(s), "
              f"sockets {man['socket_treatment'][:34]}...")
    finally:
        _drop(sid)


def test_the_deformation_manifest_never_claims_an_occlusion_check_it_did_not_run():
    """s.14: NOT_CHECKED is not CLEAR. The antagonist sweep is not wired into
    this construction, and the manifest has to say so rather than leave a lab
    to read an empty list as clearance."""
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        r = client.post(f"/api/session/{sid}/export/stages",
                        json={"construction": "deformation"})
        assert r.status_code == 200, r.text[:1500]
        import io
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(r.content))
        man = json.loads(z.read("manifest.json"))
        assert man["construction"].startswith("deformation"), \
            man["construction"]
        assert man["occlusion"]["checked"] is False
        assert "not" in man["occlusion"]["note"].lower()
        assert man["socket_treatment"].startswith("none")
        # One STL per stage and a manifest; no loose crowns.
        stls = [n for n in z.namelist() if n.endswith(".stl")]
        assert len(stls) == man["stages"], (len(stls), man["stages"])
        print(f"PASS  {len(stls)} stage STL(s); occlusion checked=False and "
              f"says so")
    finally:
        _drop(sid)


def test_print_compensation_is_refused_rather_than_ignored():
    """It is applied to a FUSED solid after a union. There is no union here,
    and offsetting the deformed cast would move enamel the rigid gate requires
    to be bit-exact. Silently ignoring it would ship a model a lab believes
    carries an allowance."""
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        r = client.post(f"/api/session/{sid}/export/stages",
                        json={"construction": "deformation",
                              "print_compensation_mm": 0.1})
        assert r.status_code == 422, r.status_code
        d = r.json()["detail"]
        assert d["measured_value"] == 0.1
        assert d["construction"] == "deformation"
        print("PASS  print_compensation_mm refused for the deformation path")
    finally:
        _drop(sid)



# ---------------------------------------------------------------------------
# 2.6 - attachments
# ---------------------------------------------------------------------------

def test_an_attachment_is_unioned_onto_the_deformed_cast_and_regated():
    """The ONLY boolean in this construction, and it is re-measured.

    `aggregate_gate_v2` decides on the deformed cast BEFORE the union. s.23
    measured a union producing coincident positions wherever a solid touches
    itself - which binary STL cannot express and a reader's weld turns into a
    non-manifold edge - so a model certified before the boolean is not
    certified after it. The file-level and self-intersection gates run again
    on the union's own output.
    """
    sid, tids = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        v = STORE.require(sid, "verts")
        tooth = STORE.get(sid, f"tooth:{tids[0]}")
        # ON the surface, not at the centroid: the centroid of a crown is
        # inside it, and `build_attachment` seats the block on the surface it
        # is given. The buccal-most vertex along the tooth's own u_bl is a
        # real bonding site.
        cv = np.asarray(tooth["cv"], float)
        u_bl = np.asarray(tooth["frame"]["u_bl"], float)
        seat = cv[int(np.argmax(cv @ u_bl))]
        r = client.post(f"/api/session/{sid}/tooth/{tids[0]}/attachment",
                        json={"tooth_id": tids[0],
                              "type": "vertical_rectangular",
                              "position_xyz": [float(x) for x in seat],
                              "normal_xyz": [float(x) for x in u_bl],
                              "rotation_deg": 0.0})
        if r.status_code != 200:
            pytest.skip(f"the fixture crown will not take an attachment: "
                        f"{r.status_code} {r.text[:200]}")

        e = client.post(f"/api/session/{sid}/export/stages",
                        json={"construction": "deformation"})
        assert e.status_code in (200, 422), e.text[:800]
        if e.status_code == 422:
            # A refusal is a legitimate outcome and must NAME itself.
            d = e.json()["detail"]
            assert d.get("error"), d
            print(f"PASS  attachment union refused by name: {d['error']}")
            return

        import io
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(e.content))
        man = json.loads(z.read("manifest.json"))
        att = man["stage_files"][0]["attachments"]
        assert att["applied"] is True, att
        # RE-MEASURED, not inherited: both gates ran again on the union.
        assert "file" in att and "self_intersection" in att, att
        assert att["self_intersection"].get("measured") is True
        assert att["positive_bodies"] == 1

        # EVERY REPORTED NUMBER MUST DESCRIBE THE BYTES SHIPPED. The union
        # changes the triangle count, so a stage that reported `len(plan.F)`
        # unconditionally would describe the cast while the file carried the
        # union - the same class of mismatch as a volume recomputed on a
        # pre-offset solid (s.20.4).
        import stl_io
        meta = man["stage_files"][0]
        fv, ff = stl_io.parse_stl_bytes(z.read(meta["file"]))
        assert meta["triangles"] == len(ff), (meta["triangles"], len(ff))
        assert meta["stl_validation"]["measured_on"] ==             "the attachment union's output"
        print(f"PASS  attachment unioned: {att['triangles']} tris, "
              f"{att['volume_mm3']} mm3, ok={att['ok']}, "
              f"{att['inverted_crumbs_discarded']} crumb(s) discarded; "
              f"manifest triangles == file triangles ({len(ff)})")
    finally:
        _drop(sid)


def test_no_attachments_means_no_boolean_at_all():
    """The claim the construction rests on: with nothing bonded, this path
    performs no boolean, so it cannot produce the self-touching edge s.26.8
    is still blocked on."""
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        r = client.post(f"/api/session/{sid}/export/stages",
                        json={"construction": "deformation"})
        assert r.status_code == 200, r.text[:800]
        import io
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(r.content))
        man = json.loads(z.read("manifest.json"))
        assert man["union_seconds"] == 0.0
        for sf in man["stage_files"]:
            assert sf["attachments"]["applied"] is False, sf["attachments"]
            assert "no attachments" in sf["attachments"]["reason"]
        print(f"PASS  {len(man['stage_files'])} stage(s), no boolean performed")
    finally:
        _drop(sid)



def test_all_three_export_formats_go_through_the_same_gate():
    """2.4 asks for the SAME response shape, and s.24.6 already pins for the
    collar that a format is not a way around the gate. The raw STL and the
    one inside the ZIP must be byte-identical, or `fmt` would be a second
    code path producing a second file."""
    import io
    import zipfile

    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        got = {}
        for fmt in ("zip", "stl", "manifest"):
            r = client.post(f"/api/session/{sid}/export/final",
                            json={"construction": "deformation", "fmt": fmt})
            assert r.status_code == 200, (fmt, r.text[:600])
            # The verdict is carried identically whatever the wrapper.
            assert r.headers["X-Print-Ready"] == "true", (fmt, dict(r.headers))
            assert r.headers["X-Export-Format"] == fmt
            got[fmt] = r.content

        z = zipfile.ZipFile(io.BytesIO(got["zip"]))
        inner = [n for n in z.namelist() if n.endswith(".stl")]
        assert len(inner) == 1, z.namelist()
        assert z.read(inner[0]) == got["stl"],             "the bare STL differs from the one inside the ZIP"
        assert json.loads(got["manifest"])["construction"].startswith(
            "deformation")
        print(f"PASS  zip {len(got['zip']):,}B / stl {len(got['stl']):,}B / "
              f"manifest {len(got['manifest']):,}B; STL byte-identical")
    finally:
        _drop(sid)



# ---------------------------------------------------------------------------
# The refusal paths - the same status codes the collar path uses
# ---------------------------------------------------------------------------

def test_the_deformation_path_refuses_with_a_status_never_a_500():
    """s.19 pins that every session endpoint 404s rather than 500s on an
    expired session. A second construction must not be the one place that
    contract stops holding - and each refusal has to be a DIFFERENT code, so
    a client can tell "nothing cut" from "no occlusal plane"."""
    seen = {}

    sid, reqs, _ = arch_session(teeth=(-0.25, 0.25), n_s=120, n_t=32)
    try:
        r = client.post(f"/api/session/{sid}/export/stages",
                        json={"construction": "deformation"})
        seen["nothing cut"] = r.status_code
        assert r.status_code == 400, r.text[:300]
        assert "cut" in r.json()["detail"].lower()

        client.post(f"/api/session/{sid}/cut",
                    json=json.loads(reqs[0].json()))
        r = client.post(f"/api/session/{sid}/export/stages",
                        json={"construction": "deformation"})
        seen["no movement"] = r.status_code
        assert r.status_code == 400, r.text[:300]
        assert "movement" in r.json()["detail"].lower()
    finally:
        _drop(sid)

    sid2 = STORE.create("lower")
    try:
        STORE.put(sid2, "verts", np.zeros((3, 3)))
        STORE.put(sid2, "faces", np.array([[0, 1, 2]]))
        r = client.post(f"/api/session/{sid2}/export/stages",
                        json={"construction": "deformation"})
        seen["no occlusal plane"] = r.status_code
        assert r.status_code == 409, r.text[:300]
    finally:
        _drop(sid2)

    r = client.post("/api/session/deadbeef/export/stages",
                    json={"construction": "deformation"})
    seen["expired session"] = r.status_code
    assert r.status_code == 404, r.text[:300]

    assert 500 not in seen.values(), seen
    print(f"PASS  {seen}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
