"""Interproximal contacts, measured before solidifying (Task 2 step 5).

`core_geometry.measure_interproximal_penetration` is the function the step
names. Its original keys measure CLOSURE - how much the vertex-to-vertex gap
shrank - which is not penetration in either direction:

  * a crown moved 1.0 mm into a 1.5 mm space closes the gap by 1.0 mm and
    touches nothing (case c on the real scan is exactly this);
  * a crown driven 0.2 mm INTO its neighbour reports a closure of 1.7 mm -
    the translation, read off the far side - and nothing in the closure keys
    says that 0.2 mm of it is overlap, because an unsigned distance has no
    sign for "inside".

So it now takes `measure_penetration=True`, which adds a SIGNED, exact
point-to-triangle depth against the neighbour's own surface. The first test
below is the control that shows why: the two measurements disagree on both
cases, and only the new one is right. The closure keys are unchanged for
every existing caller.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import core_geometry as cg
import manufacturing_v2 as mfg2
import print_solid as ps

MD = np.array([1.0, 0.0, 0.0])


def box(lo, hi, n=6):
    """A closed axis-aligned box, outward winding, `n` x `n` per face."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V, F = [], []
    for axis in range(3):
        for side in (0, 1):
            u, w = [a for a in range(3) if a != axis]
            g = np.linspace(0, 1, n + 1)
            a, b = np.meshgrid(g, g, indexing="ij")
            P = np.zeros((n + 1, n + 1, 3))
            P[..., axis] = hi[axis] if side else lo[axis]
            P[..., u] = lo[u] + a * (hi[u] - lo[u])
            P[..., w] = lo[w] + b * (hi[w] - lo[w])
            base = sum(len(x) for x in V)
            V.append(P.reshape(-1, 3))
            i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            q = (i * (n + 1) + j).ravel() + base
            t = np.vstack([np.column_stack([q, q + n + 1, q + 1]),
                           np.column_stack([q + 1, q + n + 1, q + n + 2])])
            # outward: flip where the natural (u, w) order points inward
            normal_sign = (1 if side else -1) * (1 if (u, w) in ((1, 2), (2, 0), (0, 1)) else -1)
            F.append(t if normal_sign > 0 else t[:, ::-1])
    V, F = np.vstack(V), np.vstack(F)
    V, F, _ = cg.weld_vertices(V, F)
    assert cg.signed_volume(V, F) > 0
    return V, F


def pen(crown_t0, crown_t1, nb):
    return cg.measure_interproximal_penetration(
        crown_t0, crown_t1, None, nb[0], nb[1], MD, socket_exclusion_mm=0.0,
        measure_penetration=True)


def test_closure_is_not_penetration_and_the_new_key_is():
    crown, _ = box((0, 0, 0), (5, 5, 5))
    nb = box((6.5, 0, 0), (11.5, 5, 5))            # 1.5 mm space distally

    # into the space by 1.0 mm: closure 1.0, penetration 0
    r = pen(crown, crown + MD * 1.0, nb)
    assert r["max_closure_mm"] == pytest.approx(1.0, abs=1e-6)
    assert r["penetration_mm"] == 0.0
    assert r["gap_t0_mm"] == pytest.approx(1.5, abs=1e-6)
    assert r["gap_mm"] == pytest.approx(0.5, abs=1e-6)

    # into the neighbour by 0.2 mm: closure reads the 1.7 mm translation,
    # penetration reads the 0.2 mm overlap
    r2 = pen(crown, crown + MD * 1.7, nb)
    assert r2["max_closure_mm"] == pytest.approx(1.7, abs=1e-6)
    assert r2["penetration_mm"] == pytest.approx(0.2, abs=1e-6)
    assert r2["penetration_vertices"] > 0
    assert r2["penetration_t0_mm"] == 0.0
    print(f"PASS  into the space: closure {r['max_closure_mm']} / penetration "
          f"{r['penetration_mm']}; into the neighbour: closure "
          f"{r2['max_closure_mm']} / penetration {r2['penetration_mm']:.4f}")


def test_the_closure_keys_are_unchanged_for_existing_callers():
    crown, _ = box((0, 0, 0), (5, 5, 5))
    nb = box((6.5, 0, 0), (11.5, 5, 5))
    old = cg.measure_interproximal_penetration(crown, crown + MD, None,
                                               nb[0], nb[1], MD)
    new = cg.measure_interproximal_penetration(crown, crown + MD, None,
                                               nb[0], nb[1], MD,
                                               measure_penetration=True)
    assert "penetration_mm" not in old
    assert {k: new[k] for k in old} == old
    print(f"PASS  {len(old)} original keys identical; "
          f"{len(new) - len(old)} added only on request")


def test_a_point_beyond_an_open_patch_is_not_called_inside():
    """An open patch has no inside past its edge. A flat square facing +z:
    a point under its middle is behind it; a point under the air beside it
    is not, whatever the sign of (p - q).n says."""
    g = np.linspace(0, 4, 9)
    a, b = np.meshgrid(g, g, indexing="ij")
    V = np.column_stack([a.ravel(), b.ravel(), np.zeros(a.size)])
    i, j = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    q = (i * 9 + j).ravel()
    F = np.vstack([np.column_stack([q, q + 9, q + 1]),
                   np.column_stack([q + 1, q + 9, q + 10])])
    n = np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]])
    assert n[2] > 0
    depth, dist = cg._depth_behind_patch(
        np.array([[2.0, 2.0, -0.3], [6.0, 2.0, -0.3], [2.0, 2.0, 0.3]]), V, F)
    assert depth[0] == pytest.approx(0.3, abs=1e-9)
    assert depth[1] == 0.0, "a point beyond the patch edge was called inside"
    assert depth[2] == 0.0
    print(f"PASS  behind the middle {depth[0]:.3f}; beyond the edge "
          f"{depth[1]}; in front {depth[2]}")


def test_an_unmeasurable_penetration_is_None_never_zero():
    crown, _ = box((0, 0, 0), (5, 5, 5))
    r = cg.measure_interproximal_penetration(crown, crown, None,
                                             np.zeros((3, 3)),
                                             np.zeros((0, 3), np.int64), MD,
                                             socket_exclusion_mm=0.0,
                                             measure_penetration=True)
    assert r["penetration_mm"] is None and r["penetration_failure"]
    print(f"PASS  no neighbour faces -> None: {r['penetration_failure']}")


@pytest.mark.parametrize("fdi, present, expect", [
    (43, {42, 43, 44}, [44, 42]),
    (43, {42, 43, 45}, [45, 42]),            # across an extraction space
    (41, {31, 41, 42}, [42, 31]),            # across the midline
    (47, {46, 47}, [46]),                    # distal-most
    (11, {11, 12, 21}, [12, 21]),
    (43, {43}, []),
    (99, {42, 44}, []),
])
def test_arch_neighbours(fdi, present, expect):
    assert sorted(mfg2.arch_neighbours(fdi, present)) == sorted(expect)


def test_ipr_keys_are_normalised_and_typos_refused():
    assert mfg2.normalise_ipr_by_contact(
        {"44-43": 0.2, "31/41": 0.1, "45_46": 0}) == \
        {"43-44": 0.2, "31-41": 0.1, "45-46": 0.0}
    for bad in ({"43": 0.2}, {"43-44-45": 0.1}, {"ab-44": 0.1},
                {"43-44": -0.1}, {"43-44": float("nan")}, {"43-44": "x"}):
        with pytest.raises(ValueError):
            mfg2.normalise_ipr_by_contact(bad)
    print("PASS  '44-43' / '31/41' / '45_46' normalised; 6 typos refused")


def _rows(**kw):
    row = {"contact": "43-44", "penetration_mm": 0.0, "penetration_t0_mm": 0.0,
           "gap_t0_mm": 0.3, "gap_mm": 0.1, "prescribed_ipr_mm": 0.0}
    row.update(kw)
    return {"measured": True, "tolerance_mm": 0.05, "contacts": [row]}


@pytest.mark.parametrize("contacts, ok", [
    (_rows(penetration_mm=0.04), True),
    (_rows(penetration_mm=0.05), True),                            # at tolerance
    (_rows(penetration_mm=0.06), False),                           # no IPR
    (_rows(penetration_mm=0.06, prescribed_ipr_mm=0.1), True),
    (_rows(penetration_mm=0.12, prescribed_ipr_mm=0.1), False),
    (_rows(penetration_mm=float("nan")), False),
    (_rows(penetration_mm=None), False),
    ({"measured": True, "tolerance_mm": 0.05, "contacts": []}, True),
    ({"measured": False, "reason": "no labels", "contacts": []}, False),
    (None, False),
])
def test_the_contact_gate(contacts, ok):
    g = mfg2._contact_gate(contacts)
    assert g["ok"] is ok, g


def test_the_refusal_names_the_contact_and_the_millimetres():
    g = mfg2._contact_gate(_rows(penetration_mm=0.132))
    assert g["ok"] is False
    assert g["measured"] == ["43-44: 0.132 mm penetration with no IPR "
                             "prescribed (allowed 0.050 mm)"], g["measured"]
    print(f"PASS  {g['measured'][0]}")


# ---------------------------------------------------------------------------
# Over HTTP, on the fixture - the contact gate is in the verdict
# ---------------------------------------------------------------------------

@pytest.fixture
def _plumbing_voxel(monkeypatch):
    monkeypatch.setattr(ps, "VOXEL_SIZE_MM", 0.1)


def _two_box_arch():
    """A crown (FDI 45) and its neighbour (FDI 46) as one vertex array, the
    way the cast holds them, with each one's own faces."""
    cv, cf = box((0, 0, 0), (5, 5, 5))
    nv, nf = box((5.5, 0, 0), (10.5, 5, 5))          # 0.5 mm apart
    V0 = np.vstack([cv, nv])
    spec = {"contacts": [{
        "moving_fdi": 45, "neighbour_fdi": 46, "neighbour_moving": False,
        "moving_ids": np.arange(len(cv)), "moving_faces": cf,
        "neighbour_ids": np.arange(len(nv)) + len(cv),
        "neighbour_faces": nf + len(cv), "u_md": MD}], "unmeasured": []}
    return V0, len(cv), spec


def test_measure_contacts_names_a_real_overlap_and_ipr_clears_it():
    V0, nc, spec = _two_box_arch()
    Vk = V0.copy()
    Vk[:nc] += MD * 0.7                                # 0.2 mm into 46
    c = mfg2.measure_contacts(V0, Vk, spec)
    assert c["measured"] is True
    row = c["contacts"][0]
    assert row["contact"] == "45-46"
    assert row["penetration_mm"] == pytest.approx(0.2, abs=1e-6)
    g = mfg2._contact_gate(c)
    assert g["ok"] is False
    assert g["measured"] == ["45-46: 0.200 mm penetration with no IPR "
                             "prescribed (allowed 0.050 mm)"], g["measured"]
    # IPR for THAT contact clears it; IPR for another contact does not.
    assert mfg2._contact_gate(
        mfg2.measure_contacts(V0, Vk, spec, {"46-45": 0.25}))["ok"] is True
    assert mfg2._contact_gate(
        mfg2.measure_contacts(V0, Vk, spec, {"44-45": 0.25}))["ok"] is False
    print(f"PASS  {g['measured'][0]}; 0.25 mm IPR at 46-45 clears it, at "
          f"44-45 does not")


def test_a_static_neighbour_is_measured_on_its_real_enamel_not_its_blend():
    """THE CONTROL for the V0 choice. The contact band lets a static
    neighbour's enamel blend out of the moving crown's way; measured against
    that blended surface the overlap vanishes. Here the neighbour's stage
    positions are pushed 0.3 mm away, as a blend would - and the 0.2 mm
    overlap with its REAL (T0) enamel must still be found."""
    V0, nc, spec = _two_box_arch()
    Vk = V0.copy()
    Vk[:nc] += MD * 0.7
    Vk[nc:] += MD * 0.3                                # the blend's carve
    row = mfg2.measure_contacts(V0, Vk, spec)["contacts"][0]
    assert row["penetration_mm"] == pytest.approx(0.2, abs=1e-6), row
    # A MOVING neighbour is taken where it moved to.
    spec["contacts"][0]["neighbour_moving"] = True
    row = mfg2.measure_contacts(V0, Vk, spec)["contacts"][0]
    assert row["penetration_mm"] == 0.0, row
    print("PASS  static neighbour: 0.2 mm into its T0 enamel despite a 0.3 mm "
          "blend; moving neighbour: measured where it moved to")


def test_no_labels_means_not_measured_and_refused():
    V0, _, _ = _two_box_arch()
    c = mfg2.measure_contacts(V0, V0, None)
    assert c["measured"] is False and "labels" in c["reason"]
    assert mfg2._contact_gate(c)["ok"] is False
    c = mfg2.measure_contacts(V0, V0, {"contacts": [], "unmeasured":
                                       ["tooth 3 has no FDI label"]})
    assert c["measured"] is False and mfg2._contact_gate(c)["ok"] is False
    print("PASS  no labels / an unlabelled mover -> not measured -> refused")


def test_the_contacting_fixture_is_refused_by_the_band_not_by_contact(_plumbing_voxel):
    """MEASURED, and recorded so nobody reads the contact gate as blind: on
    this fixture the labelled crowns never touch - 0.52-0.58 mm of unlabelled
    bridge separates the patches - so a 0.30 mm push leaves 0.50 mm of real
    clearance and the contact gate is RIGHT to pass it. What refuses it is the
    kit's implicit-IPR band, which sees the bridge being carved."""
    from test_export_deformation_api import (CONTACTING, NO_MOVE, _drop,
                                             _session, client)
    sid, _ = _session(CONTACTING, dict(NO_MOVE, d_md=0.30))
    try:
        r = client.post(f"/api/session/{sid}/export/final",
                        json={"construction": "deformation", "fmt": "manifest"})
        assert r.status_code == 422, r.text[:500]
        d = r.json()["detail"]
        assert "implicit_ipr_within_prescription" in d["failed_gates"]
        g = d["gates"][mfg2.CONTACT_GATE]
        assert g["ok"] is True, g
        gaps = {c["contact"]: round(c["gap_mm"], 3) for c in g["contacts"]}
        assert all(v > 0.05 for v in gaps.values()), gaps
        print(f"PASS  refused by {d['failed_gates']}; contacts clear, "
              f"real gaps {gaps}")
    finally:
        _drop(sid)


def test_a_typo_in_the_ipr_contact_is_refused(_plumbing_voxel):
    from test_export_deformation_api import NO_MOVE, SPACED, _drop, _session, client
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        r = client.post(f"/api/session/{sid}/export/final",
                        json={"construction": "deformation",
                              "ipr_by_contact_mm": {"4344": 0.2}})
        assert r.status_code == 422 and "ipr_by_contact_mm" in r.text
        print("PASS  '4344' refused with 422, not read as no IPR")
    finally:
        _drop(sid)


def test_the_seam_movement_reports_its_contacts(_plumbing_voxel):
    from test_export_deformation_api import NO_MOVE, SPACED, _drop, _session, client
    sid, _ = _session(SPACED, dict(NO_MOVE, d_oa=0.15))
    try:
        r = client.post(f"/api/session/{sid}/export/final",
                        json={"construction": "deformation", "fmt": "manifest"})
        assert r.status_code == 200, r.text[:500]
        c = r.json()["print_model"]["contacts"]
        assert c["measured"] is True
        assert sorted(x["contact"] for x in c["contacts"]) == ["44-45", "45-46"]
        for x in c["contacts"]:
            assert x["penetration_mm"] <= 0.05, x
        print("PASS  " + json.dumps([{k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in x.items()
                                      if k in ("contact", "penetration_mm",
                                               "gap_t0_mm", "gap_mm")}
                                     for x in c["contacts"]]))
    finally:
        _drop(sid)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
