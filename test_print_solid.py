"""print_solid - the printable model, by voxel solidification.

TASK 2 step 2. The decision of record is that the raw cast MAY cross itself
and solidification makes the output one watertight, intersection-free body.
So the fixture here is built to cross itself on purpose, the way the real
mandible does: scan surface just inside the trim rim is pushed down and out
through the base wall. The test first PROVES the input crosses itself - with
the same MeshLib call the export path gates on - because a solidifier handed
a clean cast would pass this test for the wrong reason.

THE PUSH IS SMOOTH, AND THAT IS A MEASURED CHOICE, NOT A CONVENIENT ONE. A
first version moved a RING of vertices rigidly, which leaves two sharp creases
at the ring's edges. Solidified with the exact calls below, that fixture came
out with MeshLib self-collisions: micro-folds of ~0.05 mm triangles sharing
one vertex at near-perpendicular normals - 8 pairs straight out of
voxelisation on the harshest push, and 0 -> 2-3 pairs INTRODUCED BY
DECIMATION on two milder ones. Deterministic, so not noise. A smooth bump (the
shape of real anatomy curling through a wall) gave 0 on both magnitudes
tried. So solidification does NOT guarantee an intersection-free output on
every input; `test_solid_gate.py` keeps the creased fixture as the control
that shows the gate catching exactly that.

Run at the PRODUCTION voxel size (0.05 mm). Slow (~1 min per solidify on this
fixture), and that is the price of testing the parameters that ship.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

import core_geometry as cg
import deform_construction as dc
import print_solid as ps
from test_cast_base import frame_for, horseshoe_shell


def _closed_cast():
    """The synthetic arch as a closed T0 cast, the construction's own way."""
    v, f = horseshoe_shell(n_s=240, n_t=60, teeth=(-0.25, 0.0, 0.25))
    af = frame_for(v)
    tv, tf, ti = cg.trim_to_arch(v, f, af, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    V0, F, _ = cg.build_cast_base(tv, tf, af,
                                  base_thickness_mm=cg.CAST_BASE_THICKNESS_MM,
                                  rim=ti["rim_loop"])
    return V0, F, len(tv), np.asarray(ti["rim_loop"]), af


def _self_colliding_pairs(V, F):
    from meshlib import mrmeshnumpy as mn
    from meshlib import mrmeshpy as mr
    mesh = mn.meshFromFacesVerts(np.asarray(F, np.int32), np.asarray(V, np.float32))
    pairs = mr.findSelfCollidingTriangles(mr.MeshPart(mesh))
    return np.array([[int(p.aFace), int(p.bFace)] for p in pairs], np.int64) \
        .reshape(-1, 2)


def _pushed_through_the_wall(creased: bool = False):
    """Scan surface near the trim rim moved down and out, through the base
    wall below it. The rim itself is not moved, so the cast stays CLOSED and
    only its GEOMETRY is wrong - which is exactly the real scan's defect.

    smooth (default)  a Gaussian bump in geodesic distance from the rim,
                      peak 1.0 mm in, moving 1.5 mm down and 0.5 mm out.
    creased           a ring 0.3-2 mm in moved rigidly 2 mm down and 1 mm
                      out - two sharp creases; see the module docstring.
    """
    V0, F, n_scan, rim, af = _closed_cast()
    scan_faces = (F < n_scan).all(axis=1)
    G = dc.edge_graph(V0[:n_scan], F[scan_faces])
    d = dijkstra(G, directed=False, indices=rim, min_only=True, limit=4.0)
    d = np.where(np.isfinite(d), d, np.inf)
    vn = dc.vertex_normals(V0, F)
    u_occ = np.asarray(af["u_occ"], float)
    V = V0.copy()
    if creased:
        ring = np.flatnonzero((d > 0.3) & (d < 2.0))
        V[ring] += vn[ring] * 1.0 - u_occ * 2.0
    else:
        w = np.where(np.isfinite(d) & (d > 0), np.exp(-((d - 1.0) / 0.5) ** 2), 0.0)
        V[:n_scan] += w[:, None] * (vn[:n_scan] * 0.5 - u_occ * 1.5)
    V = V.astype(np.float32).astype(np.float64)
    return V, F, n_scan, int(scan_faces.sum())


@pytest.fixture(scope="module")
def crossed():
    V, F, n_scan, n_scan_faces = _pushed_through_the_wall()
    pairs = _self_colliding_pairs(V, F)
    Vs, Fs, rep = ps.solidify(V, F)
    return {"V": V, "F": F, "n_scan": n_scan, "n_scan_faces": n_scan_faces,
            "pairs": pairs,
            "Vs": Vs, "Fs": Fs, "rep": rep}


def test_the_fixture_really_crosses_the_base_wall(crossed):
    """The control. Without it the next test could pass on a clean cast."""
    P, nsf, F = crossed["pairs"], crossed["n_scan_faces"], crossed["F"]
    # build_cast_base keeps the scan's faces as the PREFIX of the face array
    # (every appended wall/floor face touches an appended vertex), so a pair
    # with exactly one face below n_scan_faces is scan-vs-wall.
    assert (F[:nsf] < crossed["n_scan"]).all()
    assert (F[nsf:] >= crossed["n_scan"]).any(axis=1).all()
    scan_wall = int(((P[:, 0] < nsf) ^ (P[:, 1] < nsf)).sum())
    assert len(P) > 0 and scan_wall > 0, (len(P), scan_wall)
    assert cg.manifold_report(crossed["F"])["open_edges"] == 0
    print(f"PASS  input is closed and crosses itself: {len(P)} colliding "
          f"pairs, {scan_wall} of them scan-vs-wall")


def test_solidify_makes_one_body_with_no_self_intersection(crossed):
    Vs, Fs, rep = crossed["Vs"], crossed["Fs"], crossed["rep"]
    import manifold3d as m3
    solid = m3.Manifold(m3.Mesh(vert_properties=Vs.astype(np.float32),
                                tri_verts=Fs.astype(np.uint32)))
    assert solid.status() == m3.Error.NoError, solid.status()
    bodies = solid.decompose()
    assert len(bodies) == 1, [b.volume() for b in bodies]
    assert solid.volume() > 0
    assert len(_self_colliding_pairs(Vs, Fs)) == 0

    # And the same through the bytes a lab receives.
    blob = cg.write_binary_stl_bytes(Vs, Fs)
    import stl_io
    pv, pf = stl_io.parse_stl_bytes(blob)
    wv, wf, _ = cg.weld_vertices(pv, pf)
    mr_ = cg.manifold_report(wf)
    assert mr_["open_edges"] == 0 and mr_["nonmanifold_edges"] == 0, mr_
    assert cg._winding_is_consistent(wf)
    print(f"PASS  1 body, {solid.volume():.1f} mm3, 0 self-intersections, "
          f"{rep['triangles']:,} triangles, {rep['crumbs_discarded']} crumb(s) "
          f"({rep['crumbs_discarded_mm3']} mm3) discarded, "
          f"{rep['seconds']['total']} s")


def test_the_report_states_the_parameters_that_made_the_file(crossed):
    rep = crossed["rep"]
    assert rep["voxel_size_mm"] == ps.VOXEL_SIZE_MM == 0.05
    assert rep["decimate_max_error_mm"] == ps.DECIMATE_MAX_ERROR_MM == 0.01
    assert rep["sign_detection_mode"] == "OpenVDB"
    assert rep["offset_mode"] == "Standard"
    assert rep["meshlib_version"] == "3.1.4.297"
    assert rep["triangles"] == len(crossed["Fs"])
    assert rep["bodies_before_keep"] == 1 + rep["crumbs_discarded"] + \
        rep["internal_voids_filled"]
    for k in ("voxelise", "decimate", "keep_largest_body", "total"):
        assert rep["seconds"][k] >= 0.0, k
    print(f"PASS  voxel {rep['voxel_size_mm']} mm, {rep['sign_detection_mode']}"
          f"/{rep['offset_mode']}, meshlib {rep['meshlib_version']}")


def test_an_open_cast_is_refused_not_voxelised():
    """build_cast_base guarantees a closed cast. An open one is a
    construction defect, and sign detection would only guess an inside."""
    V0, F, *_ = _closed_cast()
    with pytest.raises(ps.SolidifyRefused) as e:
        ps.solidify(V0, F[:-1])
    assert e.value.reason == "cast_has_open_edges"
    assert e.value.detail["open_edges"] > 0
    print(f"PASS  refused: {e.value.reason}, "
          f"{e.value.detail['open_edges']} open edges")


def test_the_largest_body_is_kept_and_the_rest_counted():
    """A crumb far from the cast must not survive into the file."""
    V0, F, *_ = _closed_cast()
    # A 1 mm cube well below the floor: its own closed body.
    cube_v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                      float) + V0.min(axis=0) - 5.0
    cube_f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                       [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                       [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    V = np.vstack([V0, cube_v])
    F2 = np.vstack([F, cube_f + len(V0)])
    Vs, Fs, rep = ps.solidify(V, F2)
    assert rep["crumbs_discarded"] >= 1
    assert rep["crumbs_discarded_mm3"] >= 0.9, rep["crumbs_discarded_mm3"]
    assert Vs.min(axis=0)[2] > cube_v.max(axis=0)[2], "the cube survived"
    print(f"PASS  kept {rep['kept_volume_mm3']} mm3, discarded "
          f"{rep['crumbs_discarded']} body/bodies ({rep['crumbs_discarded_mm3']} mm3)")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
