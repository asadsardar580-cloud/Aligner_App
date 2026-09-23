"""deform_construction.py, driven through the project's REAL functions.

cg.trim_to_arch -> cg.build_cast_base builds the closed T0 cast WITH THE TEETH
STILL IN THE SURFACE; cg.kinematic_matrix makes every stage matrix;
cg.write_binary_stl_bytes -> stl_io.parse_stl_bytes -> cg.weld_vertices is the
round trip a lab's reader performs. Fixture: test_cast_base.horseshoe_shell -
curled undercut walls, domed crowns ringed by a cervical sulcus.

TWO FIXTURES, because they test two different claims:
  SPACED      crowns separated by gingiva - the tooth/gingiva SEAM, which is
              where every collar failure (B2, B4, B5, B6) lived.
  CONTACTING  neighbouring crowns merged across 302 faces - a broad contact.
              Two surfaces glued along a curve cannot slide past each other
              without the glue folding, so a sliding or pushing movement must
              be REFUSED unless IPR is prescribed. Measured, not assumed.
"""
import numpy as np
import pytest

import core_geometry as cg
import stl_io
import deform_construction as dc
import self_intersection as si
from test_cast_base import horseshoe_shell, frame_for

N_S, N_T, THETA_MAX = 240, 60, 1.95
SPACED = (-0.25, 0.0, 0.25)
CONTACTING = (-0.14, 0.0, 0.14)


def _case(teeth):
    v, f, apices = horseshoe_shell(n_s=N_S, n_t=N_T, teeth=teeth, tooth_h=4.0,
                                   return_apices=True)
    af = frame_for(v)
    tv, tf, tinfo = cg.trim_to_arch(v, f, af)
    bv, bf, _ = cg.build_cast_base(tv, tf, af, rim=tinfo["rim_loop"])
    i_s, j = np.divmod(np.arange(N_S * N_T), N_T)
    u = np.linspace(-1, 1, N_S)[i_s]
    w = (np.linspace(-THETA_MAX, THETA_MAX, N_T) / THETA_MAX)[j]
    used_top = np.zeros(len(bv), bool)
    used_top[np.unique(bf[: len(tf)])] = True
    crowns = {}
    for k, pos in enumerate(teeth):
        inside = np.zeros(len(bv), bool)
        inside[: N_S * N_T] = np.sqrt(((u - pos) / 0.13) ** 2 + (w / 0.30) ** 2) < 0.8
        crowns[k] = np.flatnonzero(inside & used_top)
    pinned = np.zeros(len(bv), bool)
    pinned[np.asarray(tinfo["rim_loop"])] = True
    pinned[len(tv):] = True                                   # floor vertices
    return {"V": bv, "F": bf, "af": af, "crowns": crowns, "pinned": pinned, "apices": apices}


@pytest.fixture(scope="module")
def spaced():
    return _case(SPACED)


@pytest.fixture(scope="module")
def contacting():
    return _case(CONTACTING)


def _frame_at(V, apex, af):
    u_oa = np.asarray(af["u_occ"], float)
    radial = V[apex] - np.asarray(af["origin"], float)
    radial -= u_oa * (radial @ u_oa)
    u_bl = radial / np.linalg.norm(radial)
    return {"u_md": np.cross(u_oa, u_bl), "u_bl": u_bl, "u_oa": u_oa}


def _file_round_trip(V, F):
    pv, pf = stl_io.parse_stl_bytes(cg.write_binary_stl_bytes(V, F))
    wv, wf, _ = cg.weld_vertices(pv, pf)
    mr = cg.manifold_report(wf)
    ncomp, _ = cg._face_components(wf, len(wv))
    return {"open": int(mr["open_edges"]), "nonmanifold": int(mr["nonmanifold_edges"]),
            "components": int(ncomp), "volume": float(cg.signed_volume(wv, wf)),
            "winding_ok": bool(cg._winding_is_consistent(wf))}


def build_stage(case, band_mm=None, envelope_mm=5.0, prescribed_ipr_mm=0.0, **clinical):
    V, F, crowns = case["V"], case["F"], case["crowns"]
    static = np.concatenate([crowns[0], crowns[2]])
    band = dc.contact_band(V, F, crowns[1], static, band_mm) if band_mm else None
    plan = dc.plan_deformation(V, F, {1: crowns[1]}, static, case["pinned"],
                               envelope_mm=envelope_mm, contact_band_vertices=band)
    fr = _frame_at(V, case["apices"][1], case["af"])
    M = cg.kinematic_matrix(fr, V[case["apices"][1]] - 10.0 * fr["u_oa"], **clinical)
    Vk = dc.stage_positions(plan, V, {1: M}, apply=cg.apply_matrix)
    rep = dc.stage_report(plan, V, Vk, F, {1: M}, apply=cg.apply_matrix)
    V32 = Vk.astype(np.float32).astype(np.float64)              # what the file will hold
    six = si.self_intersection_report(V32, F, active_faces=rep["moved_faces_mask"])
    record = {"plan_diagnostics": plan.diagnostics, "stage_report": rep,
              "self_intersection": six, "file": _file_round_trip(Vk, F),
              "index_buffer_unchanged": True, "prescribed_ipr_mm": prescribed_ipr_mm}
    return Vk, dc.aggregate_gate_v2(record), rep


CLINICAL = [
    dict(tip_deg=2.0, d_md=0.25),                                   # one ordinary stage
    dict(rotation_deg=15.0),                                        # relapse derotation
    dict(d_bl=1.0), dict(d_oa=1.0), dict(d_oa=-1.0),
    dict(torque_deg=10.0, tip_deg=8.0),
    dict(tip_deg=10.0, torque_deg=10.0, rotation_deg=10.0, d_bl=1.0, d_oa=0.5),
]
_ids = [",".join(f"{k}={v}" for k, v in c.items()) for c in CLINICAL]


def test_t0_casts_are_clean_before_anything_moves(spaced, contacting):
    for case in (spaced, contacting):
        six = si.self_intersection_report(case["V"].astype(np.float32).astype(float), case["F"])
        assert six["measured"] and six["intersecting_pairs"] == 0, six
        rt = _file_round_trip(case["V"], case["F"])
        assert (rt["open"], rt["nonmanifold"], rt["components"], rt["winding_ok"]) == (0, 0, 1, True)


@pytest.mark.parametrize("clinical", CLINICAL, ids=_ids)
def test_seam_movements_are_PRINT_READY(spaced, clinical):
    _, gate, rep = build_stage(spaced, **clinical)
    assert gate["print_ready"], (gate["failed_gates"], rep)


def test_crown_driven_into_its_neighbour_is_REFUSED(spaced):
    _, gate, _ = build_stage(spaced, d_md=6.0)
    assert not gate["print_ready"]
    assert {"no_self_intersection", "no_inverted_triangles"} & set(gate["failed_gates"])


def test_violent_movement_in_a_tiny_envelope_is_REFUSED(spaced):
    _, gate, _ = build_stage(spaced, envelope_mm=0.6, d_oa=-6.0, rotation_deg=40.0)
    assert not gate["print_ready"]


def test_sliding_a_merged_contact_is_REFUSED_by_geometry(contacting):
    _, gate, _ = build_stage(contacting, rotation_deg=15.0)
    assert "no_inverted_triangles" in gate["failed_gates"]


def test_pushing_into_a_tight_contact_needs_PRESCRIBED_IPR(contacting):
    _, gate, rep = build_stage(contacting, band_mm=1.5, d_md=0.3)
    assert rep["implicit_ipr_mm"] > dc.IPR_TOLERANCE_MM
    assert gate["failed_gates"] == ["implicit_ipr_within_prescription"], gate
    _, gate2, _ = build_stage(contacting, band_mm=1.5, d_md=0.3,
                              prescribed_ipr_mm=round(rep["implicit_ipr_mm"] + 0.05, 2))
    assert gate2["print_ready"], gate2


def test_separating_contact_builds_and_REPORTS_its_bridge(contacting):
    _, gate, rep = build_stage(contacting, band_mm=1.5, d_oa=1.0)
    assert gate["print_ready"], gate
    assert rep["implicit_ipr_mm"] == 0.0 and rep["interproximal_bridge_mm"] > 0.1


def test_empty_record_fails_every_gate():
    g = dc.aggregate_gate_v2({})
    assert not g["print_ready"] and g["failed_gates"] == list(dc.REQUIRED_GATES)


def test_gate_reads_only_keys_the_producer_writes(spaced):
    _, _, rep = build_stage(spaced, rotation_deg=3.0)
    for k in ("moving_teeth_exact_rigid", "untouched_vertices_bit_identical",
              "pinned_vertices_bit_identical", "inverted_triangles",
              "degenerate_triangles", "implicit_ipr_mm", *dc.ADVISORY):
        assert k in rep, f"gate/advisory key {k!r} is not produced by stage_report"


def test_same_index_buffer_every_stage_and_deterministic_bytes(spaced):
    V1, _, _ = build_stage(spaced, rotation_deg=6.0)
    V2, _, _ = build_stage(spaced, rotation_deg=6.0)
    assert dc.stage_digest(V1, spaced["F"]) == dc.stage_digest(V2, spaced["F"])
    assert cg.write_binary_stl_bytes(V1, spaced["F"]) == cg.write_binary_stl_bytes(V2, spaced["F"])


if __name__ == "__main__":
    # run_all_tests.py executes entries as `python <file>` and trusts the exit
    # code. Without this block this file would exit 0 having run NOTHING.
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
