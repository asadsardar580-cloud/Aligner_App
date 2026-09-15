"""The socket cup: a formed alveolar depression, not a lid over a hole.

WHAT THE DEFECT ACTUALLY WAS — this matters, because the obvious diagnosis is
wrong and a test written against it would assert something false.

The socket was closed with a triangle fan from the rim centroid, and the
symptom was a dark irregular opening under a lifted crown. The natural reading
is that a scalloped, non-convex margin makes the fan self-intersect. Measured,
it does not: a radial scallop r = R + A*cos(k*th) is star-shaped about its own
centroid for ANY amplitude, so the fan cannot cross itself. At A = 7.5 on
R = 8 (radius swinging 0.5..15.5) there are still zero crossings.

The defect is DEPTH. The fan's apex IS the rim centroid, so the cap lands on
the margin plane — depth 0.00mm — and a flat lid over a hole is exactly what
reads as a dark disc at a grazing angle.

So test_the_flat_fan_has_no_depth below is the fixture check CLAUDE.md section
5 demands: it proves the old construction fails the property the new one must
have, before anything else is asserted.
"""
import itertools
import numpy as np
import core_geometry as cg

DEPTH = 3.5


# ---------------------------------------------------------------- fixtures

def scalloped_rim(n=160, R=8.0, amp=1.5, lobes=4, z_amp=1.2, z_lobes=2):
    """A cervical margin: radial scallop plus the interproximal rise and fall."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = R + amp * np.cos(lobes * th)
    return np.column_stack([r * np.cos(th), r * np.sin(th), z_amp * np.cos(z_lobes * th)])


def non_star_rim(n=140):
    """A rim whose centroid cannot see all of it — forces the flat fallback.

    It has to be a C. Any polar form r(θ) — a scallop, even a limaçon like
    8+6.6cos(θ) whose radius swings 1.4 to 14.6 — is single-valued in θ and so
    remains star-shaped however extreme it gets. Only a shape that wraps back
    on itself puts part of the boundary out of the centroid's sight, and here
    the centroid lands in the C's opening, outside the polygon entirely.
    """
    outer = np.linspace(np.radians(20), np.radians(340), n // 2)
    inner = outer[::-1]
    pts = np.vstack([
        np.column_stack([9.0 * np.cos(outer), 9.0 * np.sin(outer)]),
        np.column_stack([5.0 * np.cos(inner), 5.0 * np.sin(inner)]),
    ])
    return np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts))])


UP = np.array([0.0, 0.0, 1.0])


def cup_of(rim, **kw):
    return cg.build_socket_cup(rim, UP, depth_mm=kw.pop("depth_mm", DEPTH), **kw)


def all_points(rim, new_pts):
    return np.vstack([np.asarray(rim, float), np.asarray(new_pts, float).reshape(-1, 3)])


# ------------------------------------------------- 1. the fixture bites

def test_the_flat_fan_has_no_depth():
    """The OLD construction, run directly. If this stops failing the fixture
    has drifted and everything below guards nothing."""
    rim = scalloped_rim()
    apex = rim.mean(axis=0)
    normal = _rim_normal(rim)
    fan_depth = float((rim.mean(axis=0) - apex) @ normal)

    new_pts, faces, info = cup_of(rim)
    cup_depth = info["deepest_below_rim_mm"]

    print(f"PASS  flat fan reaches {fan_depth:.3f}mm below the margin; "
          f"cup reaches {cup_depth:.3f}mm")
    assert abs(fan_depth) < 1e-9, (
        f"the centroid fan should sit ON the margin plane, got {fan_depth:.3f}mm — "
        f"fixture no longer reproduces the defect")
    assert cup_depth > 3.0, f"the cup did not descend: {cup_depth:.3f}mm"


def _rim_normal(rim):
    c = np.asarray(rim, float).mean(axis=0)
    _, _, vt = np.linalg.svd(np.asarray(rim, float) - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    return n if n @ UP >= 0 else -n


# ------------------------------------------------- 2. depth and shape

def test_deepest_ring_reaches_exactly_depth_mm():
    rim = scalloped_rim()
    new_pts, faces, info = cup_of(rim)
    assert info["profile"] == "cup", info["profile"]
    assert abs(info["deepest_below_rim_mm"] - DEPTH) < 0.01, info["deepest_below_rim_mm"]
    print(f"PASS  deepest ring {info['deepest_below_rim_mm']:.4f}mm below the rim plane "
          f"(target {DEPTH})")


def test_deepest_ring_is_not_degenerate():
    """A scale floor of zero collapses the deepest loop to a point: the
    bridging quads become zero-area slivers and the floor cannot be
    triangulated. The floor must stay a real loop."""
    rim = scalloped_rim()
    new_pts, faces, info = cup_of(rim)
    assert info["floor_area_fraction"] > 0.05, (
        f"deepest ring holds only {info['floor_area_fraction']:.1%} of the rim's area — "
        f"it has collapsed toward a point")

    pts = all_points(rim, new_pts)
    tri = pts[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    assert (areas > 1e-9).all(), f"{int((areas <= 1e-9).sum())} zero-area triangles"
    print(f"PASS  floor holds {info['floor_area_fraction']:.1%} of the rim area, "
          f"smallest triangle {areas.min():.2e} mm2")


# ------------------------------------------------- 3. no self-intersection

def _tri_pairs_cross_2d(tris2d, eps=1e-9):
    """Count triangle pairs whose edges PROPERLY cross in the projection.

    The sign test must run on perpendicular DISTANCE, not on the raw cross
    product, and it must have an epsilon. Radial inset places the vertices at
    angular index i and index i+n/2 on the same line through the rim centre, so
    those segments are exactly collinear and every determinant is ±1e-16 of
    floating-point noise. A strict `> 0` sign comparison reads that noise as
    "the endpoints lie on opposite sides" and reports 25 crossings between
    triangles whose bounding boxes do not even overlap. Collinear touching is
    not a proper crossing.
    """
    def seg(p1, p2, p3, p4):
        def side(a, b, c):
            n = np.hypot(b[0] - a[0], b[1] - a[1])
            if n < eps:
                return 0.0
            return ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / n
        d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
        d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
        if min(abs(d1), abs(d2), abs(d3), abs(d4)) < eps:
            return False                               # touching / collinear
        return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)

    bad = 0
    for a, b in itertools.combinations(range(len(tris2d)), 2):
        A, B = tris2d[a], tris2d[b]
        # cheap reject: disjoint bounding boxes cannot cross
        if (A[:, 0].max() < B[:, 0].min() or B[:, 0].max() < A[:, 0].min() or
                A[:, 1].max() < B[:, 1].min() or B[:, 1].max() < A[:, 1].min()):
            continue
        if any(np.allclose(p, q) for p in A for q in B):
            continue                                   # share a vertex
        if any(seg(A[i], A[(i+1) % 3], B[j], B[(j+1) % 3]) for i in range(3) for j in range(3)):
            bad += 1
    return bad


def test_cup_does_not_self_intersect_in_projection():
    """Guards the NEW ring code, which genuinely can fold — not the old fan,
    which provably cannot."""
    rim = scalloped_rim()
    new_pts, faces, info = cup_of(rim)
    pts = all_points(rim, new_pts)
    n = info["normal"]
    e1, e2 = cg._perp_basis(n)
    flat = np.column_stack([pts @ e1, pts @ e2])
    bad = _tri_pairs_cross_2d(flat[faces])
    assert bad == 0, f"{bad} triangle pairs cross in the margin plane"
    print(f"PASS  {len(faces)} cup triangles, zero crossings in projection")


# ------------------------------------------------- 4. orientation

def test_winding_is_consistent_and_floor_faces_occlusally():
    """A cup's side walls are near-parallel to u_oa, so a per-triangle dot
    against it is noise. Consistency plus the floor's unambiguous direction is
    what actually pins the orientation."""
    rim = scalloped_rim()
    new_pts, faces, info = cup_of(rim)
    pts = all_points(rim, new_pts)

    seen = {}
    for f in faces:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(u, v), max(u, v))
            seen.setdefault(key, []).append((u, v))
    flipped = [k for k, dirs in seen.items() if len(dirs) == 2 and dirs[0] == dirs[1]]
    assert not flipped, f"{len(flipped)} edges traversed the same way by both faces"

    tri = pts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.where(ln < 1e-12, 1.0, ln)
    depth_along = (rim.mean(axis=0) - tri.mean(axis=1)) @ info["normal"]
    floor = depth_along > DEPTH * 0.9
    assert floor.any(), "no floor triangles found"
    assert (fn[floor] @ info["normal"] > 0).all(), (
        "floor triangles face into the bone — the socket would render dark, the "
        "exact defect fixed in Phase 3")
    print(f"PASS  winding consistent across {len(seen)} edges; all {int(floor.sum())} "
          f"floor triangles face occlusally")


# ------------------------------------------------- 5. topology

def test_cup_is_a_disc_bounded_by_the_rim():
    rim = scalloped_rim()
    new_pts, faces, info = cup_of(rim)

    inc = cg.edge_face_incidence(faces)
    over = [e for e, fl in inc.items() if len(fl) > 2]
    assert not over, f"{len(over)} non-manifold edges (more than two faces)"

    loops = cg.boundary_loops(faces)
    assert len(loops) == 1, f"expected exactly one boundary loop, got {len(loops)}"
    assert sorted(loops[0]) == list(range(len(rim))), \
        "the boundary is not the rim itself"
    print(f"PASS  cup is edge-manifold, one boundary loop of {len(loops[0])} verts = the rim")


# ------------------------------------------------- 6. frame independence

def test_congruent_under_random_rigid_transforms():
    """The cup must not assume any world axis."""
    rng = np.random.default_rng(11)
    rim = scalloped_rim()
    base_pts, base_faces, base_info = cup_of(rim)
    base_all = all_points(rim, base_pts)
    ref = np.sort(np.linalg.norm(base_all - base_all.mean(axis=0), axis=1))

    worst = 0.0
    for _ in range(500):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
        t = rng.normal(scale=25.0, size=3)
        pts, fcs, inf = cg.build_socket_cup(rim @ R.T + t, R @ UP, depth_mm=DEPTH)
        got = all_points(rim @ R.T + t, pts)
        d = np.sort(np.linalg.norm(got - got.mean(axis=0), axis=1))
        worst = max(worst, float(np.abs(d - ref).max()),
                    abs(inf["deepest_below_rim_mm"] - base_info["deepest_below_rim_mm"]))
        assert len(fcs) == len(base_faces)
    assert worst < 1e-6, f"cups differ by {worst:.2e} under rigid motion"
    print(f"PASS  500 random rigid transforms give congruent cups (max drift {worst:.1e} mm)")


# ------------------------------------------------- 7. fallbacks and guards

def test_c_shaped_rim_still_produces_valid_geometry():
    """A rim the centroid cannot see all of. Radial inset leaves every angle
    untouched, so it survives this and still cups — which is why the guard
    validates the loops it would build rather than testing star-shapedness.
    Whichever profile it picks, the geometry must be sound."""
    rim = non_star_rim()
    new_pts, faces, info = cup_of(rim)
    assert not info["angularly_monotonic"], "fixture is no longer non-monotonic"
    assert abs(info["deepest_below_rim_mm"] - DEPTH) < 0.01
    loops = cg.boundary_loops(faces)
    assert len(loops) == 1, f"{len(loops)} boundary loops"
    inc = cg.edge_face_incidence(faces)
    assert not [e for e, fl in inc.items() if len(fl) > 2], "non-manifold"
    print(f"PASS  C-shaped rim (angularly_monotonic=False) builds a valid '{info['profile']}' "
          f"at {info['deepest_below_rim_mm']:.2f}mm, one loop, manifold")


def test_the_inset_guard_detects_a_folded_loop():
    """The guard that decides cup-vs-fallback, tested directly."""
    ring = np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2*np.pi, 40, endpoint=False)])
    assert cg.polygon_is_simple(ring)

    bowtie = np.array([[-5.0, -3.0], [5.0, 3.0], [5.0, -3.0], [-5.0, 3.0]])
    assert not cg.polygon_is_simple(bowtie), "a bowtie is not a simple polygon"

    # pentagram: a regular pentagon visited in the order 0,2,4,1,3
    pent = np.array([[np.cos(t), np.sin(t)]
                     for t in np.linspace(0, 2*np.pi, 5, endpoint=False)])
    assert not cg.polygon_is_simple(pent[[0, 2, 4, 1, 3]]), "a pentagram self-crosses"
    print("PASS  inset guard accepts a ring, rejects a bowtie and a pentagram")


def test_the_guard_catches_a_REAL_inset_pinch():
    """The failure the bowtie and pentagram both miss.

    Those two cross at index separation 2 — neighbours in the loop. A rim that
    pinches under its own inset crosses at index separations of 39 and 79:
    far apart in the vertex list, adjacent in space. This fixture is the case
    that caught a genuine hole in the guard, so it is measured here explicitly.

    The prefilter in polygon_is_simple is SPATIAL — it compares each edge's x/y
    extents — and the index arithmetic in it only EXCLUDES adjacent edges,
    which legitimately share an endpoint. It therefore does not care how far
    apart two crossing edges are in the list, which the separations below
    demonstrate.
    """
    t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    r = 8 + 6.0 * np.cos(4 * t)                      # r_min = 2.0
    P = np.column_stack([r * np.cos(t), r * np.sin(t)])
    rad = np.linalg.norm(P, axis=1)

    def inset(d):
        return P - (P / rad[:, None]) * d

    assert cg.polygon_is_simple(inset(1.9)), "1.9mm is below the clearance and is fine"
    # 2.0mm drives the troughs exactly onto the centre: vertices coincide, the
    # edges have zero length, and the signed-distance predicate used to return
    # 0.0 for every test and call the pinched polygon simple.
    assert not cg.polygon_is_simple(inset(2.0)), \
        "the collapse at exactly r_min was reported as simple"
    for d in (2.5, 3.0):
        assert not cg.polygon_is_simple(inset(d)), f"missed the fold at {d}mm"

    # and the production path never gets there: the inset is capped below it
    rim = np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))])
    _, _, info = cup_of(rim)
    assert info["inset_mm"] < info["min_clearance_mm"], info
    assert info["inset_mm"] <= cg.SOCKET_INSET_FRACTION * info["min_clearance_mm"] + 1e-9
    print(f"PASS  guard catches the pinch at r_min (index seps 39/79, spatial "
          f"prefilter); production caps inset at {info['inset_mm']:.2f}mm of "
          f"{info['min_clearance_mm']:.2f}mm clearance")


def test_flat_fallback_runs_end_to_end_on_a_pathological_rim():
    """The safety net, exercised through build_socket_cup rather than asserted.

    It has to be reached by a rim, not by depth_mm=0, or it stays dead code
    that a real scan would exercise first. A margin that pinches back to its
    own centroid — a cut that necked to a point — leaves no inward clearance
    at all, so there is nothing to inset and the rings are skipped. The floor
    must still satisfy everything a cup does.
    """
    outer = np.linspace(np.radians(4), np.radians(356), 90)
    ring = np.column_stack([9.0 * np.cos(outer), 9.0 * np.sin(outer)])
    # A slit running in to the centroid and back out. Placing the neck at the
    # RING's own centroid is a fixed point: with c = mean(ring), adding two
    # points at c leaves the overall mean at c, so the neck sits exactly on the
    # centroid and the inward clearance collapses to nothing.
    c = ring.mean(axis=0)
    slit = np.array([c + [0.0, -0.002], c + [0.0, 0.002]])
    P = np.vstack([ring, slit])
    rim = np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))])

    new_pts, faces, info = cup_of(rim)
    assert info["profile"] == "flat_fallback", info["profile"]
    assert info["min_clearance_mm"] < 0.05, info["min_clearance_mm"]
    assert info["inset_validated"] is False

    # ... and it is still a proper socket floor
    assert abs(info["deepest_below_rim_mm"] - DEPTH) < 0.01
    loops = cg.boundary_loops(faces)
    assert len(loops) == 1, f"{len(loops)} boundary loops"
    assert sorted(loops[0]) == list(range(len(rim))), "boundary is not the rim"
    inc = cg.edge_face_incidence(faces)
    assert not [e for e, fl in inc.items() if len(fl) > 2], "non-manifold fallback"

    pts = all_points(rim, new_pts)
    tri = pts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.where(ln < 1e-12, 1.0, ln)
    depth_along = (rim.mean(axis=0) - tri.mean(axis=1)) @ info["normal"]
    floor = depth_along > DEPTH * 0.9
    assert floor.any() and (fn[floor] @ info["normal"] > 0).all(), \
        "fallback floor faces into the bone"
    print(f"PASS  pinched rim (clearance {info['min_clearance_mm']:.4f}mm) reaches "
          f"flat_fallback: {len(faces)} tris at {info['deepest_below_rim_mm']:.2f}mm, "
          f"one loop, manifold, floor faces occlusally")


def test_zero_depth_degenerates_to_a_flat_cap():
    """depth_mm = 0 is the old flat lid, and must still be valid geometry —
    it is also the branch the fallback shares."""
    rim = scalloped_rim()
    new_pts, faces, info = cg.build_socket_cup(rim, UP, depth_mm=0.0)
    assert info["profile"] == "flat_fallback", info["profile"]
    assert info["deepest_below_rim_mm"] < 1e-9
    assert len(cg.boundary_loops(faces)) == 1
    inc = cg.edge_face_incidence(faces)
    assert not [e for e, fl in inc.items() if len(fl) > 2], "non-manifold"
    print(f"PASS  depth 0 gives a flat cap at the margin, one loop and manifold "
          f"({len(faces)} tris)")


def test_depth_clamps_against_a_thin_cast():
    """3.5mm under a molar is fine; through a thin anterior base it erupts out
    of the underside. A socket that breaks through is not printable."""
    rim = scalloped_rim(z_amp=0.0)
    # a floor 2.0mm below the rim plane
    g = np.linspace(-15, 15, 40)
    X, Y = np.meshgrid(g, g, indexing="ij")
    fv = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, -2.0)])
    ff = []
    for i in range(39):
        for j in range(39):
            a, b = i*40+j, i*40+j+1
            c, d = (i+1)*40+j, (i+1)*40+j+1
            ff.append([a, c, b]); ff.append([b, c, d])
    ff = np.array(ff)

    _, _, free = cg.build_socket_cup(rim, UP, depth_mm=DEPTH)
    _, _, clamped = cg.build_socket_cup(rim, UP, depth_mm=DEPTH,
                                        base_verts=fv, base_faces=ff)
    assert not free["depth_clamped"]
    assert clamped["depth_clamped"], "depth was not clamped against the cast floor"
    assert abs(clamped["depth_mm"] - 1.5) < 1e-6, clamped["depth_mm"]
    assert clamped["deepest_below_rim_mm"] < 2.0 - cg.SOCKET_FLOOR_CLEARANCE_MM + 1e-6
    print(f"PASS  cast floor 2.0mm down clamps depth {DEPTH} -> "
          f"{clamped['depth_mm']:.2f}mm (0.5mm clearance kept)")


def test_line_rim_is_refused():
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    sliver = np.column_stack([5 * np.cos(th), 0.02 * np.sin(th), np.zeros(120)])
    try:
        cg.build_socket_cup(sliver, UP)
    except ValueError as e:
        assert "line" in str(e).lower()
        print(f"PASS  a rim that is a line is refused: {str(e)[:60]}...")
        return
    raise AssertionError("a sliver rim produced a socket")


def test_scalloped_rim_is_not_rejected_as_non_planar():
    """CLAUDE.md section 5: s2/s1 is anti-correlated with error, so there must
    be no planarity gate. A deeply scalloped margin must still build a cup."""
    rim = scalloped_rim(amp=2.5, z_amp=3.0)
    _, _, info = cup_of(rim)
    assert info["profile"] == "cup", info["profile"]
    print(f"PASS  heavily scalloped rim (z +/-3.0mm) still builds a cup "
          f"(s1/s0={info['rim_ring_ratio']:.3f})")


if __name__ == "__main__":
    test_the_flat_fan_has_no_depth()
    test_deepest_ring_reaches_exactly_depth_mm()
    test_deepest_ring_is_not_degenerate()
    test_cup_does_not_self_intersect_in_projection()
    test_winding_is_consistent_and_floor_faces_occlusally()
    test_cup_is_a_disc_bounded_by_the_rim()
    test_congruent_under_random_rigid_transforms()
    test_c_shaped_rim_still_produces_valid_geometry()
    test_the_inset_guard_detects_a_folded_loop()
    test_the_guard_catches_a_REAL_inset_pinch()
    test_flat_fallback_runs_end_to_end_on_a_pathological_rim()
    test_zero_depth_degenerates_to_a_flat_cap()
    test_depth_clamps_against_a_thin_cast()
    test_line_rim_is_refused()
    test_scalloped_rim_is_not_rejected_as_non_planar()
    print("\nALL SOCKET CUP TESTS PASSED")
