"""The solid's gate - every check measured on the RE-READ bytes, each with a
control that makes it FAIL (CLAUDE.md rule 10).

TASK 2 step 3. `print_solid.measure_solid` parses the written STL back, welds
it the way a slicer does, and measures: manifold3d status / bodies / volume,
MeshLib self-collision, open and non-manifold edges, components, winding,
crown fidelity (exact point-to-triangle, Open3D) and model height.
`print_solid.solid_gate` is the pure, fail-closed verdict over those numbers.

The controls are small closed meshes built by hand, so each one breaks
exactly ONE property and the test can say which gate caught it. The one
exception is deliberate: the self-intersection gate is also shown catching a
REAL MeshLib output - the creased fixture of test_print_solid.py, whose
solidified mesh carries micro-folds - because that is the failure this gate
exists for, and a hand-built overlap alone would not prove it is reachable.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import core_geometry as cg
import manufacturing as mfg
import print_solid as ps

CUBE_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                   [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                   [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)


def cube(size=10.0, at=(0.0, 0.0, 0.0)):
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                 float) * size + np.asarray(at, float)
    return v, CUBE_F.copy()


def blob_of(v, f):
    return cg.write_binary_stl_bytes(np.asarray(v, float), np.asarray(f, np.int64))


def face_points(v, f, n=60, seed=0):
    """Points ON the surface: random barycentric samples of its faces."""
    rng = np.random.default_rng(seed)
    fi = rng.integers(0, len(f), n)
    b = rng.dirichlet((1, 1, 1), n)
    t = np.asarray(v, float)[np.asarray(f)[fi]]
    return np.einsum("ij,ijk->ik", b, t)


U_OCC = np.array([0.0, 0.0, 1.0])


def measure(v, f, pts=None):
    if pts is None:
        pts = face_points(v, f)
    return ps.measure_solid(blob_of(v, f), pts, np.arange(len(pts)),
                            ["t"] * len(pts), U_OCC)


# ---------------------------------------------------------------------------
# Keys: producer / consumer (CLAUDE.md rule 9)
# ---------------------------------------------------------------------------

def _gate_reads():
    """Every K_* constant `solid_gate` reads, found in its own source."""
    tree = ast.parse(inspect.getsource(ps.solid_gate))
    return sorted({n.id for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and n.id.startswith("K_")})


def test_every_key_the_gate_reads_is_written_by_the_producer():
    v, f = cube()
    m = measure(v, f)
    reads = _gate_reads()
    assert len(reads) >= 12, reads
    missing = [k for k in reads if getattr(ps, k) not in m]
    assert not missing, f"the gate reads keys measure_solid never writes: {missing}"
    print(f"PASS  {len(reads)} keys read by solid_gate, all written by measure_solid")


def test_a_clean_cube_passes_every_gate():
    """The positive control - without it every FAIL below could be the gate
    failing everything."""
    v, f = cube()
    g = ps.solid_gate(measure(v, f))
    assert g["failed_gates"] == [], g["failed_gates"]
    assert set(g["gates"]) == set(ps.SOLID_GATES)
    print(f"PASS  a 10 mm cube passes all {len(ps.SOLID_GATES)} gates")


def test_an_empty_record_fails_every_gate():
    """A missing measurement fails exactly like a bad one."""
    for rec in ({}, None):
        g = ps.solid_gate(rec)
        assert g["failed_gates"] == list(ps.SOLID_GATES), g["failed_gates"]
    print(f"PASS  empty record: {len(ps.SOLID_GATES)}/{len(ps.SOLID_GATES)} fail")


# ---------------------------------------------------------------------------
# One control per gate
# ---------------------------------------------------------------------------

def test_solidified_fails_on_a_refusal_and_so_does_everything_else():
    rec = ps.refused_measurement(ps.SolidifyRefused("cast_has_open_edges",
                                                    {"open_edges": 3}))
    g = ps.solid_gate(rec)
    assert g["failed_gates"] == list(ps.SOLID_GATES)
    assert g["gates"]["solidified"]["measured"]["reason"] == "cast_has_open_edges"
    print("PASS  a solidify refusal fails `solidified` by name, with its reason")


def test_two_bodies_fail_single_body_and_single_component():
    a, fa = cube()
    b, fb = cube(at=(20.0, 0.0, 0.0))
    v, f = np.vstack([a, b]), np.vstack([fa, fb + 8])
    m = measure(v, f)
    g = ps.solid_gate(m)
    assert m[ps.K_BODIES] == 2 and m[ps.K_COMPONENTS] == 2, m
    assert set(g["failed_gates"]) == {"solid_single_body", "file_single_component"}, \
        g["failed_gates"]
    print(f"PASS  two cubes: bodies {m[ps.K_BODIES]}, components "
          f"{m[ps.K_COMPONENTS]} -> {g['failed_gates']}")


def test_an_inside_out_solid_fails_positive_volume():
    v, f = cube()
    m = measure(v, f[:, ::-1])
    g = ps.solid_gate(m)
    assert "solid_positive_volume" in g["failed_gates"], (m[ps.K_VOLUME], g)
    print(f"PASS  inside-out cube: volume {m[ps.K_VOLUME]} -> "
          f"{g['failed_gates']}")


def test_an_open_mesh_fails_status_and_open_edges():
    v, f = cube()
    m = measure(v, f[:-1])
    g = ps.solid_gate(m)
    assert m[ps.K_MANIFOLD_STATUS] != "NoError", m[ps.K_MANIFOLD_STATUS]
    for name in ("solid_manifold_status_ok", "file_zero_open_edges",
                 "solid_single_body", "solid_positive_volume"):
        assert name in g["failed_gates"], (name, g["failed_gates"])
    print(f"PASS  open cube: status {m[ps.K_MANIFOLD_STATUS]}, "
          f"{m[ps.K_OPEN_EDGES]} open edges -> {g['failed_gates']}")


def test_an_edge_shared_by_four_faces_fails_nonmanifold():
    """Two cubes touching along one edge. After the reader's weld that edge
    carries four faces."""
    a, fa = cube()
    b, fb = cube(at=(10.0, 10.0, 0.0))
    v, f = np.vstack([a, b]), np.vstack([fa, fb + 8])
    m = measure(v, f)
    g = ps.solid_gate(m)
    assert m[ps.K_NONMANIFOLD_EDGES] > 0, m
    assert "file_zero_nonmanifold_edges" in g["failed_gates"], g["failed_gates"]
    print(f"PASS  edge-touching cubes: {m[ps.K_NONMANIFOLD_EDGES]} non-manifold "
          f"edge(s) -> {g['failed_gates']}")


def test_one_flipped_face_fails_winding():
    v, f = cube()
    f = f.copy()
    f[0] = f[0][::-1]
    m = measure(v, f)
    g = ps.solid_gate(m)
    assert m[ps.K_WINDING] is False
    assert "file_consistent_winding" in g["failed_gates"], g["failed_gates"]
    print(f"PASS  one flipped face -> {g['failed_gates']}")


def test_interpenetrating_cubes_fail_self_intersection():
    """Two cubes overlapping in space but not in index: the file is two
    closed shells that cross."""
    a, fa = cube()
    b, fb = cube(at=(5.0, 5.0, 5.0))
    v, f = np.vstack([a, b]), np.vstack([fa, fb + 8])
    m = measure(v, f)
    g = ps.solid_gate(m)
    assert m[ps.K_SELF_COLLIDING] is True and m[ps.K_SELF_COLLIDING_PAIRS] > 0, m
    assert "solid_no_self_intersection" in g["failed_gates"], g["failed_gates"]
    print(f"PASS  crossing cubes: {m[ps.K_SELF_COLLIDING_PAIRS]} colliding pairs "
          f"-> {g['failed_gates']}")


def test_a_real_solidified_micro_fold_fails_self_intersection():
    """THE REACHABLE CASE. The creased fixture from test_print_solid.py,
    solidified with the production calls, carries MeshLib self-collisions -
    measured 2026-09-24. This is the gate catching what solidification
    itself cannot promise, on a mesh MeshLib actually produced."""
    from test_print_solid import _pushed_through_the_wall
    V, F, *_ = _pushed_through_the_wall(creased=True)
    Vs, Fs, rep = ps.solidify(V, F)
    m = ps.measure_solid(cg.write_binary_stl_bytes(Vs, Fs))
    g = ps.solid_gate(m)
    assert m[ps.K_SELF_COLLIDING] is True, m[ps.K_SELF_COLLIDING]
    assert "solid_no_self_intersection" in g["failed_gates"], g["failed_gates"]
    print(f"PASS  creased fixture, solidified ({rep['triangles']:,} tris): "
          f"{m[ps.K_SELF_COLLIDING_PAIRS]} colliding pairs, e.g. "
          f"{m['meshlib_self_colliding_examples'][0]['at']} -> gate FAILS")


def test_crown_points_off_the_surface_fail_p95_then_max():
    v, f = cube()
    pts = face_points(v, f, n=200)
    # every point 0.05 mm outside: p95 over 0.03 -> FAIL p95 (and not max)
    out = pts.copy()
    out[:, 2] = np.where(np.isclose(out[:, 2], 10.0), out[:, 2] + 0.05, out[:, 2])
    on_top = np.isclose(pts[:, 2], 10.0)
    assert on_top.sum() > 10
    g1 = ps.solid_gate(measure(v, f, np.vstack([out[on_top]] * 1)))
    assert "crown_fidelity_p95" in g1["failed_gates"], g1["gates"]["crown_fidelity_p95"]
    assert "crown_fidelity_max" not in g1["failed_gates"]
    # one point 0.2 mm away among 199 exact ones: max fails, p95 passes
    one = pts.copy()
    k = int(np.flatnonzero(on_top)[0])
    one[k, 2] += 0.2
    m2 = measure(v, f, one)
    g2 = ps.solid_gate(m2)
    assert "crown_fidelity_max" in g2["failed_gates"], g2["gates"]["crown_fidelity_max"]
    assert "crown_fidelity_p95" not in g2["failed_gates"]
    assert m2[ps.K_CROWN_WORST][0]["deviation_mm"] == pytest.approx(0.2, abs=1e-4)
    assert len(m2[ps.K_CROWN_WORST]) == ps.WORST_LOCATIONS
    print(f"PASS  0.05 mm off -> {g1['failed_gates']}; one point 0.2 mm off -> "
          f"{g2['failed_gates']}, worst location ranked first")


def test_crown_fidelity_with_no_points_or_nan_distances_fails():
    v, f = cube()
    g = ps.solid_gate(ps.measure_solid(blob_of(v, f), None, None, None, U_OCC))
    assert {"crown_fidelity_p95", "crown_fidelity_max"} <= set(g["failed_gates"])
    assert "no tooth vertices" in str(g["gates"]["crown_fidelity_p95"]["measured"])

    real = mfg._point_to_surface
    try:
        mfg._point_to_surface = lambda pts, *a, **k: np.full(len(pts), np.nan)
        m = measure(v, f)
    finally:
        mfg._point_to_surface = real
    g = ps.solid_gate(m)
    assert m[ps.K_CROWN_P95] is None and m[ps.K_CROWN_MAX] is None
    assert {"crown_fidelity_p95", "crown_fidelity_max"} <= set(g["failed_gates"])
    print("PASS  no points -> FAIL; NaN distances -> None and FAIL, never 0.0")


def test_nan_and_bool_never_pass_a_numeric_gate():
    """Every comparison against NaN is False - a `<=` alone cannot be the
    whole check (CLAUDE.md lessons)."""
    base = measure(*cube())
    for bad in (float("nan"), float("inf"), True, None, "0.0"):
        m = dict(base)
        m[ps.K_CROWN_P95] = bad
        m[ps.K_CROWN_MAX] = bad
        m[ps.K_VOLUME] = bad
        g = ps.solid_gate(m)
        for name in ("crown_fidelity_p95", "crown_fidelity_max",
                     "solid_positive_volume"):
            assert name in g["failed_gates"], (bad, name)
    print("PASS  NaN / inf / bool / None / str never pass a numeric gate")


def test_model_height_is_a_warning_only():
    v, f = cube(size=25.0)
    m = measure(v, f)
    g = ps.solid_gate(m)
    assert m[ps.K_HEIGHT] == pytest.approx(25.0, abs=1e-5)
    assert g["advisory"]["model_height_over_limit"] is True
    assert g["failed_gates"] == [], "a height warning must never refuse"
    v, f = cube(size=15.0)
    assert ps.solid_gate(measure(v, f))["advisory"]["model_height_over_limit"] is False
    m = dict(m)
    m[ps.K_HEIGHT] = None
    assert ps.solid_gate(m)["advisory"]["model_height_over_limit"] is None
    print("PASS  25 mm -> warning, not a refusal; 15 mm -> no warning; "
          "unmeasured -> None, not False")


def test_face_components_agrees_with_the_project_s_own_count():
    """The vectorised count must give manufacturing.components' answer."""
    a, fa = cube()
    b, fb = cube(at=(10.0, 10.0, 0.0))                 # edge-touching
    c, fc = cube(at=(10.0, 10.0, 10.0))                # vertex-touching b
    cases = {"one": fa,
             "two apart": np.vstack([fa, cube(at=(30, 0, 0))[1] + 8]),
             "edge-touching (after weld)": None,
             "three": np.vstack([fa, fb + 8, fc + 16])}
    for name, f in cases.items():
        if f is None:
            wv, wf, _ = cg.weld_vertices(np.vstack([a, b]), np.vstack([fa, fb + 8]))
            f = wf
        assert ps.face_components(f) == mfg.components(f), name
    print("PASS  face_components == manufacturing.components on "
          f"{len(cases)} cases")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
