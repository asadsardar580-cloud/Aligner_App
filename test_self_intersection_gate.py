"""`no_self_intersection` - the 17th gate, and a control that makes it fail.

AGENT_BRIEF defect G1. No geometric self-intersection test existed anywhere in
this repository. Every topology gate measures CONNECTIVITY, and a mesh can be
closed, manifold, one connected component and correctly wound while two of its
triangles pass through each other. A slicer will not print that.

The step that can introduce it is already in the pipeline:
`cg.offset_along_normals`, used for print compensation, is a vertex-normal
offset - and this project has measured that operation folding at concavities
three separate times (CLAUDE.md s.8's offset comparison table, s.23's
LOCAL_CLEARANCE self-touch). Print compensation defaults to 0.0, so the fold
is latent rather than active, which is exactly the kind of defect that ships.

The control below is a notched solid whose two groove walls face each other
1.2 mm apart. Measured on it (26,972 faces after refinement):

    compensation 0.0 mm ->    0 intersecting pairs   gate PASSES
    compensation 0.3 mm ->  216 intersecting pairs   gate FAILS
    compensation 0.7 mm -> 1378 intersecting pairs   gate FAILS

AGENT_BRIEF A2 rule 12: a gate that cannot fail proves nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

import core_geometry as cg
import manufacturing as mfg
import self_intersection as si

m3 = pytest.importorskip("manifold3d")

GATE = "no_self_intersection"


def _notched_solid(edge_mm=0.4):
    """A cube with a deep narrow groove: two concave walls 1.2 mm apart.

    Refined, because a vertex-normal offset cannot fold a groove whose walls
    are single large quads - the shared corner vertices move in an averaged
    direction. The coarse 28-face version showed 0 intersections at every
    offset, which would have made this a control that proves nothing.
    """
    solid = (m3.Manifold.cube([20.0, 20.0, 10.0], True)
             - m3.Manifold.cube([1.2, 30.0, 6.0], True).translate([0, 0, 3.5]))
    mesh = solid.refine_to_length(edge_mm).to_mesh()
    return (np.asarray(mesh.vert_properties, float)[:, :3],
            np.asarray(mesh.tri_verts, np.int64))


def _as_written(v, f, compensation_mm):
    """Exactly what `build_stage_bundle` writes: compensate, then round."""
    if compensation_mm > 0.0:
        v = cg.offset_along_normals(v, f, compensation_mm)
    return v.astype(np.float32).astype(np.float64)


def _gate_row(stage):
    return next(g for g in mfg.aggregate_print_gate(stage)["gates"]
                if g["gate"] == GATE)


# ---------------------------------------------------------------------------
# The gate exists and fails closed
# ---------------------------------------------------------------------------

def test_the_aggregate_gate_now_has_seventeen_gates():
    gate = mfg.aggregate_print_gate({})
    names = [g["gate"] for g in gate["gates"]]
    assert len(names) == 17, names
    assert GATE in names
    assert gate["print_ready"] is False
    assert len(gate["failed_gates"]) == 17, gate["failed_gates"]
    print(f"PASS  17 gates, all 17 fail on an empty record")


def test_a_missing_report_FAILS_rather_than_passing():
    """`NOT_CHECKED` is not `CLEAR`."""
    assert _gate_row({})["passed"] is False
    print("PASS  a stage with no self-intersection report fails the gate")


def test_a_report_that_could_not_RUN_fails():
    """`measured: False` is a failure, never a clean result."""
    row = _gate_row({mfg.KEY_SELF_INTERSECTION: {
        "measured": False, "reason": "non-finite coordinates"}})
    assert row["passed"] is False, row
    assert row["measured"]["reason"] == "non-finite coordinates"
    print("PASS  measured=False fails, with the reason carried through")


def test_the_evidence_report_cites_the_new_gate():
    rep = mfg.manufacturing_evidence_report({})
    claims = [c for c in rep["claims"] if GATE in c["evidence_gates"]]
    assert len(claims) == 1, [c["claim"] for c in rep["claims"]]
    assert claims[0]["verified"] is False
    print(f"PASS  evidence claim: {claims[0]['claim']!r}")


# ---------------------------------------------------------------------------
# The control: print compensation folds a concavity
# ---------------------------------------------------------------------------

def test_the_unfolded_solid_passes_the_gate():
    """The control for the control. If this fails, the fixture is wrong."""
    v, f = _notched_solid()
    rep = si.self_intersection_report(*(_as_written(v, f, 0.0), f))
    assert rep["measured"] is True
    assert rep["intersecting_pairs"] == 0, rep["examples"][:3]
    assert _gate_row({mfg.KEY_SELF_INTERSECTION: rep})["passed"] is True
    print(f"PASS  compensation 0.0 mm: 0 intersecting pairs, gate passes")


@pytest.mark.parametrize("comp_mm", [0.3, 0.7])
def test_print_compensation_folds_the_concavity_and_the_gate_FAILS(comp_mm):
    v, f = _notched_solid()
    folded = _as_written(v, f, comp_mm)
    rep = si.self_intersection_report(folded, f)

    assert rep["measured"] is True
    assert rep["intersecting_pairs"] > 0, (
        f"compensation {comp_mm} mm did not fold the groove - this control "
        f"proves nothing")

    row = _gate_row({mfg.KEY_SELF_INTERSECTION: rep})
    assert row["passed"] is False, row
    print(f"PASS  compensation {comp_mm} mm: "
          f"{rep['intersecting_pairs']} intersecting pairs, "
          f"{rep['faces_involved']} faces, kinds={rep['by_kind']} -> gate FAILS")


def test_every_topology_gate_would_have_MISSED_the_fold():
    """Why this gate had to be added, demonstrated rather than asserted.

    The folded mesh is still closed, still one component and still correctly
    wound. Nothing that measures connectivity can see the fold.
    """
    v, f = _notched_solid()
    folded = _as_written(v, f, 0.7)

    blob = cg.write_binary_stl_bytes(folded, f)
    validation = mfg.validate_printable_stl(blob)

    assert validation["open_edges"] == 0, validation
    assert validation["nonmanifold_edges"] == 0, validation
    assert validation["connected_components"] == 1, validation

    rep = si.self_intersection_report(folded, f)
    assert rep["intersecting_pairs"] > 0

    print(f"PASS  the folded model reads closed / 0 non-manifold / "
          f"1 component, and has {rep['intersecting_pairs']} "
          f"self-intersecting triangle pairs")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
