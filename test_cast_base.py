"""The virtual cast base: trim the scan to a horseshoe, extrude it to a solid.

WHAT THIS IS DEFENDING AGAINST, measured on a real mandibular scan before any
of it was written:

An intraoral scan is an OPEN SHELL. case_lower.stl reports open_edges 2101,
nonmanifold_edges 0, and condition_mesh finds exactly ONE hole -- those 2101
edges are the perimeter of the scanned region, not a defect. cap_and_close
sealed that perimeter by fanning across it, and the perimeter of an arch is a
HORSESHOE, so the cap spanned the U-shaped tongue opening: a web of long thin
triangles over empty space. Topologically closed, anatomically a membrane.

The membrane test below is therefore the centre of this file, and per CLAUDE.md
section 5 it first proves the fixture REPRODUCES the bug: cap_and_close on the
same band must put triangles in the tongue space, or the test is asserting
something vacuous.

The fixture is a parametric horseshoe band whose edges curl UNDER (theta_max
past 90 degrees), because a height field cannot reproduce the defect that makes
this hard -- see test_repair_is_local_and_keeps_the_rim.
"""
import numpy as np

import core_geometry as cg
import arch_frame


# =========================================================================
# Fixture
# =========================================================================

ARCH_W, ARCH_D, SPAN, BAND_W, BAND_H = 26.0, 20.0, 2.35, 9.6, 16.0


def horseshoe_shell(n_s=150, n_t=44, arch_w=ARCH_W, arch_d=ARCH_D, band_w=BAND_W,
                    band_h=BAND_H, theta_max=1.95, span=SPAN, cusps=7,
                    jitter=0.0, seed=0, teeth=(), tooth_h=2.6, tooth_d=1.1,
                    return_apices=False, tooth_w=0.13):
    """A scan-shaped open shell: a horseshoe band with steep, undercut walls.

    Cross-section is (band_w*sin(theta), band_h*cos(theta)) for theta running
    past +/-pi/2, so the buccal and lingual edges CURL BACK UNDER the widest
    point. Both parts matter, and the proportions are not decoration:

      * band_w 9.6 against a 9.0mm trim margin puts the trim boundary at
        theta = 1.22 rad, where the wall's slope is d(height)/d(lateral) = 4.5
        — near vertical. That is what makes the boundary's occlusal PROJECTION
        fold back on itself, which is the defect prune_to_simple exists for.
        A shallower band cuts the boundary across a gentle slope and the
        projection comes out clean, so the test would pass vacuously.
      * a height field over a grid has no undercut at all and its boundary
        projects to a simple rectangle.

    `jitter` displaces vertices along the arch tangent, standing in for the
    irregular triangulation of a real scan.

    Returns (verts, faces). Topologically a rectangle: one boundary loop.
    """
    a = np.linspace(-span, span, n_s)
    centre = np.column_stack([arch_w * np.sin(a), arch_d * np.cos(a)])
    tang = np.gradient(centre, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    norm = np.column_stack([-tang[:, 1], tang[:, 0]])

    th = np.linspace(-theta_max, theta_max, n_t)
    lateral = band_w * np.sin(th)                 # + is buccal
    height = band_h * np.cos(th)

    # cusps along the ridge, so the occlusal quartile is a real ridge and not a
    # flat rail that the angular binning could sample anywhere.
    ridge = 1.0 + 0.10 * np.cos(cusps * np.pi * np.linspace(-1, 1, n_s))

    V = np.empty((n_s, n_t, 3))
    V[:, :, :2] = centre[:, None, :] + norm[:, None, :] * lateral[None, :, None]
    V[:, :, 2] = height[None, :] * ridge[:, None]

    # Crowns, if asked for: a dome on the ridge ringed by a cervical sulcus,
    # displaced along the cross-section's own radial direction so the band grows
    # outward rather than the height field being bent. `teeth` are positions
    # along the arch in [-1, 1].
    apices = []
    if len(teeth):
        u = np.linspace(-1.0, 1.0, n_s)[:, None]
        w = (th / theta_max)[None, :]
        rad = np.stack([np.sin(th), np.cos(th)], axis=1)      # (lateral, height)
        for pos in teeth:
            r = np.sqrt(((u - pos) / tooth_w) ** 2 + (w / 0.30) ** 2)
            disp = tooth_h * np.exp(-(r ** 2) / (2 * 0.6 ** 2))
            disp = disp - tooth_d * np.exp(-((r - 1.0) ** 2) / (2 * 0.22 ** 2))
            V[:, :, :2] += norm[:, None, :] * (disp * rad[None, :, 0])[:, :, None]
            V[:, :, 2] += disp * rad[None, :, 1]
            i_s = int(np.argmin(np.abs(np.linspace(-1.0, 1.0, n_s) - pos)))
            apices.append(int(i_s * n_t + n_t // 2))

    if jitter:
        rng = np.random.default_rng(seed)
        V[:, :, :2] += tang[:, None, :] * rng.normal(0.0, jitter, (n_s, n_t))[:, :, None]

    verts = V.reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_s - 1), np.arange(n_t - 1), indexing="ij")
    a0 = (i * n_t + j).ravel()
    b0, c0, d0 = a0 + 1, a0 + n_t, a0 + n_t + 1
    faces = np.vstack([np.column_stack([a0, c0, b0]),
                       np.column_stack([b0, c0, d0])])
    if return_apices:
        return verts, faces.astype(int), apices
    return verts, faces.astype(int)


def frame_for(verts, span=SPAN, arch_w=ARCH_W, arch_d=ARCH_D, band_h=BAND_H):
    """Occlusal frame from landmarks on the fixture: the two posterior ends of
    the ridge and the anterior midline."""
    L = np.array([arch_w * np.sin(-span), arch_d * np.cos(-span), band_h])
    R = np.array([arch_w * np.sin(span), arch_d * np.cos(span), band_h])
    A = np.array([0.0, arch_d, band_h])
    return arch_frame.fit_occlusal_frame(L, R, A, verts.mean(axis=0))


def _occlusal_frame_from_geometry(verts, faces):
    """An arch frame for a real scan, standing in for the three clicked landmarks.

    The app gets these from the clinician. A test cannot click, so the two
    posterior cusp tips and the anterior midline are picked geometrically: the
    scan's least-variance axis is the occlusal normal, signed to point AWAY from
    the perimeter loop (which runs around the vestibule, apical of everything),
    and the landmarks are the highest points of the two posterior quadrants and
    the midline.
    """
    verts = np.asarray(verts, float)
    c = verts.mean(axis=0)
    _, _, vt = np.linalg.svd(verts - c, full_matrices=False)
    ax = vt[2] / np.linalg.norm(vt[2])
    perimeter = np.asarray(max(cg.boundary_loops(faces), key=len))
    if ax @ (c - verts[perimeter].mean(axis=0)) < 0:
        ax = -ax
    e1 = vt[0] / np.linalg.norm(vt[0])
    e2 = np.cross(ax, e1)
    P = np.column_stack([(verts - c) @ e1, (verts - c) @ e2])
    H = (verts - c) @ ax

    top = H > np.percentile(H, 90)
    Pt, Ht, idx = P[top], H[top], np.where(top)[0]
    y = Pt[:, 1]
    back = y < np.percentile(y, 15)
    left = back & (Pt[:, 0] < np.median(Pt[:, 0]))
    right = back & (Pt[:, 0] >= np.median(Pt[:, 0]))
    front = y > np.percentile(y, 92)
    return arch_frame.fit_occlusal_frame(
        verts[idx[left][np.argmax(Ht[left])]],
        verts[idx[right][np.argmax(Ht[right])]],
        verts[idx[front][np.argmax(Ht[front])]], c)


def add_debris(verts, faces, where, size=4.0, n=9):
    """A disconnected patch of stray capture — tongue, cheek, the operator's
    finger. The trim must drop it; largest_face_component is what does so."""
    xs = np.linspace(-size, size, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    blob = np.column_stack([(X + where[0]).ravel(), (Y + where[1]).ravel(),
                            np.full(X.size, where[2])])
    off = len(verts)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a0 = (i * n + j).ravel() + off
    b0, c0, d0 = a0 + 1, a0 + n, a0 + n + 1
    extra = np.vstack([np.column_stack([a0, c0, b0]), np.column_stack([b0, c0, d0])])
    return np.vstack([verts, blob]), np.vstack([faces, extra]).astype(int)


def curve_distance(pts2d, curve):
    from scipy.spatial import cKDTree
    return cKDTree(cg._polyline_samples(curve)).query(np.atleast_2d(pts2d))[0]


def in_tongue_space(pts2d, curve, reach_mm):
    """Which of `pts2d` land in the horseshoe's concave interior, past the band.

    A point is over the arch opening when it lies inside the region bounded by
    the arch curve and the chord closing its two posterior ends, AND further
    from the curve than the band physically reaches. cg._point_in_polygon wraps
    the index, so handing it the OPEN curve closes it with exactly that chord.

    `reach_mm` is measured from the rim itself rather than set to the trim
    margin, and the difference is not pedantry: trim_to_arch keeps faces whose
    CENTROID is within the margin, so a kept face's vertices reach further --
    measured 9.38mm against a 9.0mm margin. Testing against the margin flags
    two floor triangles 23 microns past it and calls a correct cast a membrane.
    """
    pts2d = np.atleast_2d(pts2d)
    inside = np.array([cg._point_in_polygon(p, curve) for p in pts2d])
    return inside & (curve_distance(pts2d, curve) > reach_mm)


# =========================================================================
# Task A — the trim
# =========================================================================

def test_trim_leaves_one_loop_and_drops_the_noise():
    v, f = horseshoe_shell()
    v, f = add_debris(v, f, (0.0, 6.0, -2.0))       # tongue floor, inside the U
    v, f = add_debris(v, f, (40.0, 0.0, -3.0))      # cheek, outside the arch
    # A float ON the band but not attached to it — saliva bridge, a shard of the
    # opposing arch. The two above are killed by the distance predicate alone; it
    # takes this one to exercise the largest-island step, which is the thing A1
    # is actually about.
    v, f = add_debris(v, f, (0.0, ARCH_D, 9.0), size=2.0, n=5)
    af = frame_for(v)

    n_in = len(f)
    tv, tf, info = cg.trim_to_arch(v, f, af)

    assert info["boundary_loops"] == 1, info
    # `components` reports the OUTPUT, which is one island by construction. It
    # used to report islands FOUND, and that was read as "the exported base has
    # two components" — the reporting was the defect, not the geometry.
    assert info["components"] == 1, info["components"]
    # Two islands survive the distance predicate: the band, and the float sitting
    # on it. The far blobs never get that far — the predicate alone removes them,
    # which is why a fixture needs BOTH kinds to test both steps.
    assert info["islands_found"] >= 2, \
        f"the float did not arrive as a separate island ({info['islands_found']})"
    assert len(info["islands_removed"]) == info["islands_found"] - 1
    assert info["island_faces_removed"] == sum(d["faces"] for d in info["islands_removed"])
    assert info["island_faces_removed"] == 32, \
        f"the 5x5 float is 32 faces; {info['island_faces_removed']} were removed"
    assert info["faces_removed"] > 0

    # The debris is gone: nothing survives anywhere near where it was.
    origin, e1, e2, _ = cg._arch_basis(af)
    to2 = lambda X: np.column_stack([(X - origin) @ e1, (X - origin) @ e2])
    kept2 = to2(tv[tf].mean(axis=1))
    for centre in ((0.0, 6.0), (40.0, 0.0)):
        near = np.linalg.norm(kept2 - to2(np.array([[centre[0], centre[1], 0.0]]))[0],
                              axis=1) < 4.0
        assert not near.any(), f"{int(near.sum())} debris faces survived near {centre}"

    # And the band is bounded by the margin. Checked on the faces the PREDICATE
    # kept — the first faces_kept of them — not on the whole list:
    # _fill_interior_holes appends fan triangles whose centroids sit wherever
    # the hole was, which for a hole touching the trim boundary is just outside.
    d = curve_distance(kept2[:info["faces_kept"]], info["curve"])
    assert d.max() <= info["margin_mm"] + 1e-9, \
        f"a trimmed face is {d.max():.2f}mm out, past the {info['margin_mm']}mm margin"

    # The dentition survives: every ridge-top face is still there.
    origin2, _, _, u_occ = cg._arch_basis(af)
    h_in = (v[f].mean(axis=1) - origin2) @ u_occ
    crown = h_in > np.percentile(h_in, 96)
    h_out = (tv[tf].mean(axis=1) - origin2) @ u_occ
    assert h_out.max() >= h_in[crown].max() - 1e-9, "the trim cut into the occlusal ridge"

    print(f"PASS  trim: {n_in:,} -> {info['faces_kept']:,} faces, {info['components']} islands "
          f"-> 1, debris dropped, ridge intact (max {d.max():.2f}mm of {info['margin_mm']}mm)")


def test_trim_fills_an_interior_hole():
    """A horseshoe strip has one boundary loop; what the face predicate produces
    does not. Measured on a real scan at margin 7.0: two loops, 2225 and 81 --
    an interior HOLE, not a disconnection, and no choice of margin reliably
    avoids one. So the trim fills them and the guarantee becomes structural."""
    v, f = horseshoe_shell()
    af = frame_for(v)

    # Punch a hole in the middle of the band, well inside the trim margin.
    origin, e1, e2, u_occ = cg._arch_basis(af)
    rel = v[f].mean(axis=1) - origin
    h = rel @ u_occ
    hole = np.argsort(-h)[200:260]              # a patch just off the ridge crest
    keep = np.ones(len(f), bool)
    keep[hole] = False
    holed = f[keep]
    assert len(cg.boundary_loops(holed)) >= 2, "the fixture did not actually gain a hole"

    tv, tf, info = cg.trim_to_arch(v, holed, af)
    assert info["holes_filled"] >= 1, info
    assert info["boundary_loops"] == 1
    assert cg._open_edge_count(tf) == info["rim_points"]
    print(f"PASS  trim fills {info['holes_filled']} interior hole(s) -> exactly one "
          f"boundary loop of {info['rim_points']} vertices")


# =========================================================================
# Task B — the extrusion
# =========================================================================

def test_extruded_base_is_a_closed_solid():
    v, f = horseshoe_shell()
    af = frame_for(v)
    tv, tf, tinfo = cg.trim_to_arch(v, f, af)
    bv, bf, info = cg.build_cast_base(tv, tf, af, rim=tinfo["rim_loop"])

    health = cg.manifold_report(bf)
    assert health["open_edges"] == 0, f"{health['open_edges']} open edges"
    assert health["nonmanifold_edges"] == 0, f"{health['nonmanifold_edges']} non-manifold edges"
    assert cg._winding_is_consistent(bf), "normals are not consistently oriented"
    assert info["volume_mm3"] > 0, info["volume_mm3"]

    # Flat bottom, planar to 1e-9.
    origin, _, _, u_occ = cg._arch_basis(af)
    floor_v = bv[len(tv):]
    dev = np.abs((floor_v - origin) @ u_occ - info["base_plane_offset_mm"]).max()
    assert dev < 1e-9, f"floor is {dev:.2e}mm out of plane"

    print(f"PASS  cast base: {info['faces']:,} faces, 0 open / 0 non-manifold of "
          f"{health['total_edges']:,}, volume {info['volume_mm3']:,.0f}mm3, "
          f"floor planar to {dev:.1e}mm")


def test_no_face_spans_the_arch_opening():
    """THE MEMBRANE BUG. It must be impossible, not merely absent.

    The fixture is proved to reproduce it first: cap_and_close on the very same
    band must put triangles in the tongue space. Asserting only that the new
    base is clean would pass just as happily on a fixture where nothing could
    ever have gone wrong.
    """
    v, f = horseshoe_shell()
    af = frame_for(v)
    tv, tf, tinfo = cg.trim_to_arch(v, f, af)
    origin, e1, e2, _ = cg._arch_basis(af)
    to2 = lambda X: np.column_stack([(X - origin) @ e1, (X - origin) @ e2])
    curve = tinfo["curve"]
    # How far the band physically reaches, measured on the rim it was cut to.
    reach = float(curve_distance(to2(tv[tinfo["rim_loop"]]), curve).max())

    # --- the fixture reproduces the bug ---------------------------------
    ov, of = cg.cap_and_close(tv.copy(), tf.copy())
    cap = of[len(tf):]
    cap_d = curve_distance(to2(ov[cap].mean(axis=1)), curve)
    spanning = in_tongue_space(to2(ov[cap].mean(axis=1)), curve, reach)
    assert spanning.sum() > 0, \
        "cap_and_close did not span the opening on this fixture, so the test is vacuous"

    # --- the new base cannot ---------------------------------------------
    bv, bf, info = cg.build_cast_base(tv, tf, af, rim=tinfo["rim_loop"])
    floor = bf[-info["floor_tris"]:]
    assert not in_tongue_space(to2(bv[floor].mean(axis=1)), curve, reach).any(), \
        "a floor triangle sits in the tongue space"
    # Not just the floor — no face of the finished cast may span the opening.
    assert not in_tongue_space(to2(bv[bf].mean(axis=1)), curve, reach).any(), \
        "some face of the cast base spans the arch opening"

    # WHY it cannot, stated as the structural fact rather than a threshold:
    # every floor vertex IS a projected rim vertex (prune_to_simple only ever
    # deletes), so the floor lies in the convex hull of the rim's projection and
    # can never reach further from the arch than the rim itself does.
    d = curve_distance(to2(bv[floor].mean(axis=1)), curve)
    assert d.max() <= reach + 1e-9, \
        f"a floor triangle centroid is {d.max():.3f}mm out, past the rim's own {reach:.3f}mm"

    print(f"PASS  membrane impossible: cap_and_close put {int(spanning.sum())} of "
          f"{len(cap)} cap triangles over the opening, out to {cap_d.max():.1f}mm; "
          f"trim->extrude puts 0 of {len(bf):,}, every floor tri within "
          f"{d.max():.2f}mm of the ridge against the rim's own {reach:.2f}mm")


def test_thickness_is_measured_from_the_lowest_point():
    """On a rim with 4mm of vertical scallop, the base plane must sit
    base_thickness_mm below the LOWEST point — not below the centroid, which
    would leave part of the wall inverted and the surface poking through."""
    v, f = horseshoe_shell()
    af = frame_for(v)
    origin, e1, e2, u_occ = cg._arch_basis(af)

    # Scallop the two long edges vertically by +/-2mm (4mm peak to peak).
    tv, tf, tinfo = cg.trim_to_arch(v, f, af)
    rim = tinfo["rim_loop"]
    phase = np.linspace(0, 12 * np.pi, len(rim))
    tv = tv.copy()
    tv[rim] += np.outer(2.0 * np.sin(phase), u_occ)

    heights = (tv[np.unique(tf)] - origin) @ u_occ
    scallop = float(((tv[rim] - origin) @ u_occ).max() - ((tv[rim] - origin) @ u_occ).min())
    assert scallop >= 4.0, f"fixture scallop is only {scallop:.1f}mm"

    bv, bf, info = cg.build_cast_base(tv, tf, af, base_thickness_mm=12.0, rim=rim)
    expected = heights.min() - 12.0
    assert abs(info["base_plane_offset_mm"] - expected) < 1e-9, \
        f"plane at {info['base_plane_offset_mm']:.4f}, expected {expected:.4f}"

    floor_h = (bv[len(tv):] - origin) @ u_occ
    assert floor_h.max() <= heights.min() - 12.0 + 1e-9, "the floor is not below every surface point"
    assert cg.manifold_report(bf)["watertight"]
    print(f"PASS  base plane {info['base_plane_offset_mm']:.3f} = lowest surface "
          f"{heights.min():.3f} - 12.0, on a rim with {scallop:.1f}mm of scallop")


def test_repair_is_local_and_keeps_the_rim():
    """The floor is repaired IN PLACE, not decimated, and the gap stays small.

    Measured on the real scan, the projected rim carries 31-47 crossings
    depending on the margin, every one at index separation 2-39 out of ~2140 —
    the boundary stepping over its own recent path on a steep wall. Left alone,
    ear_clip_polygon bails and the floor has holes.

    An earlier build fixed that by decimating the outline to 0.5mm spacing,
    which discarded 78% of the rim. That is not a triangulation aid: the wall
    bridges each rim vertex to its nearest surviving floor vertex, so four or
    five rim vertices sharing one floor vertex fan into slivers and the cast
    wall comes out STRIATED. Local repair alone measures a 4.6% gap on the real
    scan, and this pins the same property on the fixture.

    margin_mm is forced to 9.0: the fixture's band is 9.6mm wide, so only a trim
    that cuts near its steep wall produces the projection folding this is about.
    At the 7.0 default the fixture's rim is clean and the test would be vacuous.
    """
    for jitter in (0.0, 0.6):
        v, f = horseshoe_shell(jitter=jitter, seed=3)
        af = frame_for(v)
        tv, tf, tinfo = cg.trim_to_arch(v, f, af, margin_mm=9.0)
        origin, e1, e2, _ = cg._arch_basis(af)
        rim = tinfo["rim_loop"]
        Q = np.column_stack([(tv[rim] - origin) @ e1, (tv[rim] - origin) @ e2])

        crossings = len(cg._crossing_pairs(Q))
        assert crossings > 0, "the fixture's rim does not fold, so this is vacuous"
        short = cg.ear_clip_polygon(Q)
        assert len(short) < len(Q) - 2, \
            (f"ear clipping the unpruned rim returned {len(short)} of {len(Q) - 2} — it "
             f"did not bail, so the floor would not have had holes and this is vacuous")
        # Every crossing must be LOCAL; a long-range one would mean the buccal
        # and lingual rails genuinely cross and repairing it would web the arch.
        xp = cg._crossing_pairs(Q)
        sep = np.minimum(np.abs(xp[:, 0] - xp[:, 1]), len(Q) - np.abs(xp[:, 0] - xp[:, 1]))
        assert sep.max() <= cg.MAX_PRUNE_ARC_FRACTION * len(Q), \
            f"a crossing spans {sep.max()} of {len(Q)} — that is not local"

        surv, pinfo = cg.prune_to_simple(Q)
        full, forced = cg.ear_clip_polygon_robust(Q[surv])
        assert len(full) == len(surv) - 2, \
            f"repaired rim still triangulates short: {len(full)} of {len(surv) - 2}"
        assert pinfo["worst_arc_fraction"] <= cg.MAX_PRUNE_ARC_FRACTION

        # THE INVARIANT IS AREA, NOT COUNT. ear_clip_polygon_robust always
        # returns n-2, so counting is vacuous against it: before the needle drop
        # existed this fixture returned every triangle it owed and was missing
        # 23% of the floor's area. Nothing else in the pipeline would have seen
        # that, and it is a hole in the bottom of a printed cast.
        P = Q[surv]
        a, b, c_ = P[full[:, 0]], P[full[:, 1]], P[full[:, 2]]
        tri_area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c_[:, 1] - a[:, 1])
                                - (b[:, 1] - a[:, 1]) * (c_[:, 0] - a[:, 0])).sum()
        outline = abs(cg._signed_area_2d(P))
        assert abs(tri_area - outline) < 1e-6 * outline, \
            f"floor area {tri_area:.4f} against outline {outline:.4f}"

        # Decimation is gone. This fixture is a REGULAR GRID trimmed exactly at
        # its steep wall, which folds far harder than scan data — it measures
        # 33-42% where the real scan measures 5.6%, and the <10% requirement is
        # asserted there, in test_real_scan_repair_keeps_the_rim. What is pinned
        # here is that we are nowhere near the 78% decimation used to throw away.
        assert pinfo["gap_fraction"] < 0.50, \
            f"local repair dropped {100 * pinfo['gap_fraction']:.1f}% — decimation is back"

        bv, bf, info = cg.build_cast_base(tv, tf, af, rim=rim)
        assert cg.manifold_report(bf)["watertight"]
        assert info["components"] == 1
        print(f"PASS  repair is local (jitter {jitter}): {crossings} crossings, max "
              f"separation {sep.max()}/{len(Q)}; unpruned ear-clip {len(short)}/{len(Q) - 2}; "
              f"{pinfo['repairs']} repairs + {pinfo['needles_removed']} needles -> "
              f"{len(surv)}/{len(Q)} kept (gap {100 * pinfo['gap_fraction']:.1f}%), "
              f"{forced} forced clips, area exact to {pinfo['area_error']:.0e}, watertight")


def test_real_scan_repair_keeps_the_rim():
    """The <10% gap requirement, asserted on real scan data.

    The synthetic fixture is a regular grid trimmed at a near-vertical wall and
    folds far harder than anything a scanner produces (33-42%). This is the case
    the requirement is actually about, so it runs against case_lower.stl and
    skips if that is not present.
    """
    import os
    if not os.path.exists("case_lower.stl"):
        print("SKIP  real-scan repair (case_lower.stl not present)")
        return

    import stl_io
    v, f = stl_io.parse_stl_bytes(open("case_lower.stl", "rb").read())
    v, f, _ = cg.condition_mesh(v, f)
    af = _occlusal_frame_from_geometry(v, f)

    tv, tf, ti = cg.trim_to_arch(v, f, af)
    bv, bf, bi = cg.build_cast_base(tv, tf, af, rim=ti["rim_loop"])

    assert bi["gap_fraction"] < 0.10, \
        (f"rim {bi['rim_points']} -> floor {bi['floor_points']} is a "
         f"{100 * bi['gap_fraction']:.1f}% gap; over 10% striates the cast wall")
    assert bi["prune"]["area_error"] < 1e-9
    assert bi["components"] == 1, bi["components"]
    assert bi["forced_clips"] == 0, \
        f"{bi['forced_clips']} corners had to be force-clipped on real scan data"
    assert cg.manifold_report(bf)["watertight"]
    print(f"PASS  real scan at margin {ti['margin_mm']}mm: kept {100 * ti['kept_fraction']:.1f}%, "
          f"rim {bi['rim_points']} -> floor {bi['floor_points']} (gap "
          f"{100 * bi['gap_fraction']:.1f}%), {bi['repairs']} local repairs, "
          f"{bi['forced_clips']} forced, area exact to {bi['prune']['area_error']:.0e}, "
          f"{bi['components']} component, watertight")


def test_rigid_transforms_give_congruent_bases():
    """500 random rigid transforms must give the identical cast.

    The arch frame rotates with the geometry, so nothing may depend on a world
    axis — that is the whole reason _arch_basis reads u_tra/u_occ from the frame
    instead of running a PCA of the mesh.

    The mesh here is deliberately fine enough that the trim is unambiguous, and
    the assertion is then EXACT: same face count, same volume to the bit. Two
    coarser fixtures were measured and both wobble, for reasons worth recording
    rather than hiding behind a tolerance:

      * at n_t=28 the band is ~0.6% different in volume — a handful of face
        centroids sit within rounding of the margin and flip in or out;
      * at n_t=24 it swings by 80%, because the trim nearly severs the band and
        `np.bincount(labels).argmax()` picks a different largest component.

    Neither is reachable on scan data (187k faces across the same band), but
    both are real, and the second is the one to remember: on a mesh coarse
    enough for the trim to nearly disconnect the arch, which component survives
    is not stable.
    """
    v0, f0 = horseshoe_shell(n_s=110, n_t=32)
    af0 = frame_for(v0)
    tv, tf, ti = cg.trim_to_arch(v0, f0, af0, margin_mm=7.0)
    bv, bf, ref = cg.build_cast_base(tv, tf, af0, rim=ti["rim_loop"])

    rng = np.random.default_rng(11)
    worst_vol = 0.0
    for _ in range(500):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        t = rng.normal(scale=40.0, size=3)
        v = v0 @ Q.T + t

        af = dict(af0)
        for key in ("u_occ", "u_sag", "u_tra"):
            af[key] = np.asarray(af0[key], float) @ Q.T
        af["origin"] = np.asarray(af0["origin"], float) @ Q.T + t

        tv2, tf2, ti2 = cg.trim_to_arch(v, f0, af, margin_mm=7.0)
        _, bf2, info = cg.build_cast_base(tv2, tf2, af, rim=ti2["rim_loop"])
        assert len(bf2) == len(bf), f"face count changed: {len(bf2)} vs {len(bf)}"
        assert info["floor_tris"] == ref["floor_tris"], "floor triangulation changed"
        worst_vol = max(worst_vol, abs(info["volume_mm3"] - ref["volume_mm3"]))

    rel = worst_vol / abs(ref["volume_mm3"])
    assert rel < 1e-9, f"volume drifted {worst_vol:.6f}mm3 ({rel:.2e} relative)"
    print(f"PASS  500 rigid transforms: {len(bf):,} faces and {ref['floor_tris']} floor "
          f"tris every time, volume {ref['volume_mm3']:,.0f}mm3 drifting {worst_vol:.2e} "
          f"({rel:.1e} relative)")


def test_refuses_rather_than_shipping_a_broken_base():
    """No allow_unsealed anywhere. A base that fails its own assertions is a bug
    to fix, and an override would hide exactly the failures these exist for."""
    v, f = horseshoe_shell(n_s=64, n_t=18)
    af = frame_for(v)
    tv, tf, ti = cg.trim_to_arch(v, f, af)

    # Two boundary loops: punch a hole straight through the band.
    holed = np.delete(tf, np.arange(40, 70), axis=0)
    try:
        cg.build_cast_base(tv, holed, af)
        raise AssertionError("a two-loop surface was accepted")
    except ValueError as e:
        assert "boundary loop" in str(e), str(e)

    # A rim whose rails genuinely cross in projection must hit the MEMBRANE
    # guard, not scrape through. A lemniscate is the honest fixture: densely
    # sampled, so it survives decimation, and self-crossing half a loop apart,
    # so repairing it would delete an arc long enough to chord across the gap —
    # which is the web over the arch opening in miniature.
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    denom = 1.0 + np.sin(t) ** 2
    fig8 = np.column_stack([20.0 * np.cos(t) / denom,
                            20.0 * np.sin(t) * np.cos(t) / denom])
    try:
        cg.prune_to_simple(fig8)
        raise AssertionError("a figure-eight was accepted as prunable")
    except ValueError as e:
        assert "chord across the gap" in str(e), str(e)

    # And an outline too small to be an arch is refused rather than triangulated.
    try:
        cg.prune_to_simple(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]]))
        raise AssertionError("a 4-point crossing outline was accepted")
    except ValueError:
        pass

    print("PASS  refuses a multi-loop surface, a self-crossing rim (membrane guard) "
          "and a degenerate outline, with no override")


if __name__ == "__main__":
    test_trim_leaves_one_loop_and_drops_the_noise()
    test_trim_fills_an_interior_hole()
    test_extruded_base_is_a_closed_solid()
    test_no_face_spans_the_arch_opening()
    test_thickness_is_measured_from_the_lowest_point()
    test_repair_is_local_and_keeps_the_rim()
    test_real_scan_repair_keeps_the_rim()
    test_rigid_transforms_give_congruent_bases()
    test_refuses_rather_than_shipping_a_broken_base()
    print("\nALL CAST BASE TESTS PASSED")

def test_undercut_wall_crosses_scan():
    import self_intersection as si
    v, f = horseshoe_shell(band_w=10.0, band_h=10.0, theta_max=2.5)
    af = frame_for(v)
    tv, tf, ti = cg.trim_to_arch(v, f, af, margin_mm=22.0)
    tv, tf, rim, uinfo = cg.clear_undercut_periphery(tv, tf, af, ti["rim_loop"])
    cv, cf, ci = cg.build_cast_base(tv, tf, af, rim=rim)
    pairs = si.candidate_pairs(cv, cf)
    if len(pairs):
        hit, _ = si._pairs_intersect(cv, cf, pairs, 1e-6)
        assert hit.sum() == 0, "Expected undercut fix to remove intersections"

def test_undercut_protects_band():
    import pytest
    v, f = horseshoe_shell(band_w=10.0, band_h=10.0, theta_max=2.5)
    af = frame_for(v)
    tv, tf, ti = cg.trim_to_arch(v, f, af, margin_mm=22.0)
    protected = np.ones(len(tf), dtype=bool)
    tv2, tf2, rim, uinfo = cg.clear_undercut_periphery(tv, tf, af, ti["rim_loop"], protected_mask=protected)
    assert len(tf2) == len(tf), "Protected band was touched"
