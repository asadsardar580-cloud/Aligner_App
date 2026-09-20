"""Which signed-distance method may the manufacturing gate rely on?

WHY THIS EXISTS. `CastProbe.signed` is nearest-VERTEX distance signed against
that vertex's normal. It is cheap, it is what `check_occlusal_collision` has
always used, and it has already produced one wrong answer that cost a full
debugging round: marching inward with it reported 0.1mm of material under a
cast at least 3mm thick, because on a steep cervical wall the nearest vertex's
normal points sideways and the dot product flips. That failure was worked
around in `local_thickness` by switching to ray-triangle intersection. The
question this file answers is the general one: WHERE is nearest-vertex signing
reliable, and where must the gate use something else?

NO NEW DEPENDENCY IS EVALUATED HERE. Open3D and trimesh are both already
installed and already load-bearing (CLAUDE.md section 19: the vendored
inference pipeline imports both). The point is to choose between tools that
are present, not to add one.

THE GROUND TRUTH IS INDEPENDENT OF ALL THREE CANDIDATES, and that is the whole
value of the file. A benchmark that scores Open3D against Open3D returns 100%
and means nothing - the same mistake `benchmark_segmentation.py` refuses to
make with ToothGroupNetwork's own predictions.

  * MAGNITUDE: exact closest-point-on-triangle against EVERY triangle, no
    spatial index, no candidate pruning. O(points x faces) and slow on
    purpose - there is nothing in it to be approximately right about.
  * SIGN: ray parity. A closed surface is crossed an odd number of times from
    any interior point. Three independent random directions vote, and a point
    where they disagree is reported as AMBIGUOUS and excluded rather than
    guessed - a grazed edge is exactly the case where a parity test is not
    entitled to an opinion.

Run: python bench_signed_distance.py
"""

from __future__ import annotations

import time

import numpy as np


# ---------------------------------------------------------------------------
# Reference implementation. Deliberately slow and obviously correct.
# ---------------------------------------------------------------------------

def closest_point_on_triangles(p, v0, v1, v2):
    """Closest point on each triangle to the single point `p`.

    Ericson, Real-Time Collision Detection 5.1.5 - the barycentric region
    test, written out in full rather than clamped-and-rescaled. A
    clamp-then-rescale version of exactly this was wrong earlier in this
    project and returned a distance LARGER than the distance to the nearest
    vertex, which is impossible.
    """
    ab, ac, ap = v1 - v0, v2 - v0, p - v0
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    bp = p - v1
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    cp = p - v2
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < 1e-20, 1.0, denom)

    out = np.empty_like(v0)

    # Interior of the face.
    v = vb / safe
    w = vc / safe
    out[:] = v0 + ab * v[:, None] + ac * w[:, None]

    # Edge AB.
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    if m.any():
        den = d1[m] - d3[m]
        t = np.clip(d1[m] / np.where(den == 0, 1.0, den), 0, 1)
        out[m] = v0[m] + ab[m] * t[:, None]
    # Edge AC.
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    if m.any():
        den = d2[m] - d6[m]
        t = np.clip(d2[m] / np.where(den == 0, 1.0, den), 0, 1)
        out[m] = v0[m] + ac[m] * t[:, None]
    # Edge BC.
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    if m.any():
        den = (d4[m] - d3[m]) + (d5[m] - d6[m])
        t = np.clip((d4[m] - d3[m]) / np.where(den == 0, 1.0, den), 0, 1)
        out[m] = v1[m] + (v2[m] - v1[m]) * t[:, None]
    # Vertex regions.
    m = (d1 <= 0) & (d2 <= 0)
    out[m] = v0[m]
    m = (d3 >= 0) & (d4 <= d3)
    out[m] = v1[m]
    m = (d6 >= 0) & (d5 <= d6)
    out[m] = v2[m]
    return out


def exact_unsigned(pts, verts, faces, chunk=8):
    """Exact distance to the triangle SET. No index, no candidate pruning."""
    tri = verts[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        for j, p in enumerate(pts[i:i + chunk]):
            q = closest_point_on_triangles(p, v0, v1, v2)
            out[i + j] = float(np.sqrt(((q - p) ** 2).sum(1)).min())
    return out


def _crossings(pts, direction, verts, faces):
    """How many triangles each ray crosses. Moller-Trumbore, t > 0 only."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    tri = verts[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    parallel = np.abs(det) < 1e-12
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=~parallel)

    n = np.empty(len(pts), np.int64)
    grazed = np.zeros(len(pts), bool)
    for i, p in enumerate(pts):
        tvec = p - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        w = np.einsum("j,ij->i", d, qvec) * inv
        t = np.einsum("ij,ij->i", e2, qvec) * inv
        hit = (~parallel) & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-9)
        n[i] = int(hit.sum())
        # A hit within 1e-6 of a barycentric boundary is a grazed edge: the
        # neighbouring triangle may or may not also report it, so parity along
        # THIS ray is not trustworthy for this point.
        if hit.any():
            uu, ww = u[hit], w[hit]
            grazed[i] = bool((np.minimum.reduce([uu, ww, 1 - uu - ww]) < 1e-6).any())
    return n, grazed


def reference_signed(pts, verts, faces, seed=7):
    """(signed, ambiguous). Negative = inside. Ambiguous points are excluded."""
    pts = np.atleast_2d(np.asarray(pts, float))
    mag = exact_unsigned(pts, verts, faces)

    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(3, 3))
    votes, bad = [], np.zeros(len(pts), bool)
    for d in dirs:
        n, grazed = _crossings(pts, d, verts, faces)
        votes.append((n % 2) == 1)
        bad |= grazed
    votes = np.array(votes)
    tally = votes.sum(0)
    inside = tally >= 2
    ambiguous = bad | ((tally != 0) & (tally != 3))   # not unanimous
    return np.where(inside, -mag, mag), ambiguous


# ---------------------------------------------------------------------------
# The candidates
# ---------------------------------------------------------------------------

def castprobe_signed(probe, pts):
    """What the manufacturing gate actually calls today."""
    return probe.signed(pts)


def nearest_vertex_signed(probe, pts):
    """What it called before, and what this benchmark exists to retire."""
    return probe.signed_nearest_vertex(pts)[0]


def trimesh_signed(tm, pts):
    """trimesh.proximity, WITHOUT rtree.

    `trimesh.proximity.signed_distance` imports `rtree`, which is not
    installed here and is not going to be added for a benchmark.
    `closest_point_naive` needs no spatial index and is the same arithmetic
    without the pruning, so the ACCURACY measured is trimesh's real answer;
    only its speed is pessimistic, and the report says so.
    """
    import trimesh
    cp, d, fid = trimesh.proximity.closest_point_naive(tm, pts)
    n = tm.face_normals[fid]
    return np.where(np.einsum("ij,ij->i", pts - cp, n) < 0, -d, d)


def open3d_signed(scene, pts, nsamples=1):
    """Open3D's raycasting SDF.

    `nsamples` is the number of ray directions it votes over when deciding
    the sign. The docs recommend an ODD value > 1 for meshes that are not
    perfectly watertight, so that a majority always exists. The cast IS
    watertight by construction (`build_cast_base` asserts zero open and zero
    non-manifold edges), so n=1 and n=11 are both measured here to show
    whether the extra samples buy anything on this geometry.
    """
    import open3d as o3d
    t = o3d.core.Tensor(np.ascontiguousarray(pts, np.float32))
    if nsamples > 1:
        return scene.compute_signed_distance(t, nsamples=nsamples).numpy().astype(float)
    return scene.compute_signed_distance(t).numpy().astype(float)


# ---------------------------------------------------------------------------
# Difficult point classes
# ---------------------------------------------------------------------------

def _face_frames(verts, faces):
    tri = verts[faces]
    cen = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-12
    n[keep] /= ln[keep][:, None]
    side = np.maximum.reduce([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)])
    return cen, n, side, keep


def build_cases(verts, faces, u_occ, n_per=60, seed=3):
    """Every class the brief names, as (name, points, note)."""
    rng = np.random.default_rng(seed)
    cen, fn, side, ok = _face_frames(verts, faces)
    big = ok & (side > 0.6)          # room to place a point away from an edge
    idx_all = np.flatnonzero(big)

    def interior(idx, margin=0.30):
        """Barycentric point at least `margin` of the way in from every edge."""
        k = len(idx)
        a = rng.uniform(margin, 1 - 2 * margin, size=k)
        b = rng.uniform(margin, 1 - margin - a)
        tri = verts[faces[idx]]
        return (tri[:, 0]
                + (tri[:, 1] - tri[:, 0]) * a[:, None]
                + (tri[:, 2] - tri[:, 0]) * b[:, None])

    def pick(pool, k):
        return rng.choice(pool, size=min(k, len(pool)), replace=False)

    cases = []
    base = pick(idx_all, n_per)
    P = interior(base)
    N = fn[base]
    cases.append(("on_surface", P.copy(), "exactly on a triangle"))
    for d in (0.01, 0.05, 0.10, 0.50):
        cases.append((f"inside_{d:.2f}mm", P - N * d, "along the face normal"))
    for d in (0.10, 1.00):
        cases.append((f"outside_{d:.2f}mm", P + N * d, "along the face normal"))

    steep = np.flatnonzero(big & (np.abs(fn @ u_occ) < 0.25))
    if len(steep):
        s = pick(steep, n_per)
        cases.append(("steep_wall_0.05in", interior(s) - fn[s] * 0.05, "|n.u_occ|<0.25"))
        s = pick(steep, n_per)
        cases.append(("steep_wall_0.30in", interior(s) - fn[s] * 0.30, "|n.u_occ|<0.25"))

    floor = np.flatnonzero(big & (fn @ u_occ < -0.9))
    if len(floor):
        s = pick(floor, n_per)
        cases.append(("base_wall_0.05in", interior(s) - fn[s] * 0.05, "cast underside"))

    # Concavity: faces whose neighbours lean back over them. Measured as the
    # mean of (neighbour_centroid - centroid) . n; positive means the surface
    # curls toward the normal side, i.e. a pocket.
    from scipy.spatial import cKDTree
    tre = cKDTree(cen)
    nb = tre.query(cen[idx_all], k=9, workers=-1)[1]
    rel = cen[nb] - cen[idx_all][:, None, :]
    curl = np.einsum("ij,ij->i", rel.mean(1), fn[idx_all])
    conc = idx_all[curl > np.percentile(curl, 90)]
    if len(conc):
        s = pick(conc, n_per)
        cases.append(("concavity_0.05in", interior(s) - fn[s] * 0.05, "top decile curl"))

    # Near an EDGE and near a VERTEX - where "nearest vertex" and "nearest
    # surface point" diverge most.
    s = pick(idx_all, n_per)
    tri = verts[faces[s]]
    a = rng.uniform(0.002, 0.02, size=len(s))
    near_edge = (tri[:, 0]
                 + (tri[:, 1] - tri[:, 0]) * rng.uniform(0.3, 0.7, len(s))[:, None]
                 + (tri[:, 2] - tri[:, 0]) * a[:, None])
    cases.append(("near_edge_0.05in", near_edge - fn[s] * 0.05, "<2% in from an edge"))
    near_vert = (tri[:, 0]
                 + (tri[:, 1] - tri[:, 0]) * a[:, None]
                 + (tri[:, 2] - tri[:, 0]) * rng.uniform(0.002, 0.02, len(s))[:, None])
    cases.append(("near_vertex_0.05in", near_vert - fn[s] * 0.05, "<2% from a corner"))
    return cases


def add_free_space_cases(cases, verts, u_occ, crowns, n_per=60, seed=5):
    """Points in free space that a nearest-vertex normal will mis-sign.

    Two of them, both named in the brief: over the arch OPENING (the tongue
    space, where the nearest vertex sits on a lingual wall whose normal points
    away from the point) and in the INTERPROXIMAL corridor between two teeth.
    """
    rng = np.random.default_rng(seed)
    u = np.asarray(u_occ, float)
    c = verts.mean(0)
    hi = float((verts @ u).max())
    p = c + u * (hi - float(c @ u)) * 0.6
    cases.append(("over_arch_opening", p + rng.normal(scale=2.0, size=(n_per, 3)),
                  "inside the horseshoe"))
    if crowns is not None and len(crowns) >= 2:
        mid = 0.5 * (crowns[0].mean(0) + crowns[1].mean(0))
        cases.append(("interproximal", mid + rng.normal(scale=1.0, size=(n_per, 3)),
                      "midway between two crowns"))
    return cases


# ---------------------------------------------------------------------------

def run(n_per=60, verbose=True):
    import manufacturing as mfg
    import api_core
    from test_manufacturing_interface import _cast_and_teeth

    sid, tids, recs, v, bv, bf = _cast_and_teeth([dict(d_oa=1.2), dict(d_md=0.6)])
    try:
        af = api_core.STORE.get(sid, "arch_frame")
        u_occ = np.asarray(af["u_occ"], float)

        # THE PROBE IS BUILT ON THE RAW ARRAYS, deliberately. That is what the
        # production path hands it, and handing it a pre-compacted copy here
        # would hide the very defect this benchmark found - `CastProbe` now
        # compacts internally and reports how many it dropped.
        probe = mfg.CastProbe(bv, bf)

        # The GROUND TRUTH and the sample points need the compacted geometry:
        # the cast carries vertices no face references (it is trimmed out of
        # the scan's array, and rule 3.1 forbids rebuilding that array), and
        # sampling the uncompacted array is what made the first run of this
        # benchmark report 2.4mm of error for points that were supposed to be
        # exactly on the surface.
        used = np.unique(bf)
        remap = -np.ones(len(bv), np.int64)
        remap[used] = np.arange(len(used))
        cv, cf = bv[used], remap[bf]
        if verbose:
            print(f"cast: {len(bv)} vertices in the array, {len(used)} referenced "
                  f"by {len(cf)} faces ({len(bv) - len(used)} unreferenced)")
            print(f"CastProbe dropped {probe.dropped_unreferenced} unreferenced "
                  f"vertices; signed() method = {probe.method}")

        cases = build_cases(cv, cf, u_occ, n_per=n_per)
        add_free_space_cases(cases, cv, u_occ,
                             [np.asarray(r["cv"], float) for r in recs], n_per=n_per)

        import trimesh
        tm = trimesh.Trimesh(vertices=cv, faces=cf, process=False, validate=False)
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cv),
                                       o3d.utility.Vector3iVector(cf))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(om))

        methods = [
            ("near-vert(old)", lambda p: nearest_vertex_signed(probe, p)),
            ("CastProbe(new)", lambda p: castprobe_signed(probe, p)),
            ("trimesh", lambda p: trimesh_signed(tm, p)),
            ("o3d n=1", lambda p: open3d_signed(scene, p, 1)),
            ("o3d n=11", lambda p: open3d_signed(scene, p, 11)),
        ]

        head = f"{'case':20s} {'n':>4s} {'amb':>4s} |"
        for m, _ in methods:
            head += f" {m + ' sign':>14s} {'max err':>9s} |"
        if verbose:
            print()
            print(head)
            print("-" * len(head))

        totals = {m: [0, 0, 0.0, 0.0] for m, _ in methods}
        timing = {m: 0.0 for m, _ in methods}
        per_case = {}
        npts = 0

        for name, pts, note in cases:
            ref, amb = reference_signed(pts, cv, cf)
            keep = ~amb
            if keep.sum() == 0:
                if verbose:
                    print(f"{name:20s} {len(pts):4d} {int(amb.sum()):4d} | "
                          f"all ambiguous, excluded")
                continue
            r, p = ref[keep], pts[keep]
            npts += len(p)
            row = f"{name:20s} {len(pts):4d} {int(amb.sum()):4d} |"
            per_case[name] = {}
            for m, fn_ in methods:
                t0 = time.perf_counter()
                got = np.asarray(fn_(p), float)
                timing[m] += time.perf_counter() - t0
                sign_ok = np.where(np.abs(r) < 1e-9, True, np.sign(got) == np.sign(r))
                err = np.abs(np.abs(got) - np.abs(r))
                row += f" {100.0 * sign_ok.mean():13.1f}% {err.max():9.4f} |"
                per_case[name][m] = (float(sign_ok.mean()), float(err.max()))
                t = totals[m]
                t[0] += len(p)
                t[1] += int(sign_ok.sum())
                t[2] += float(err.sum())
                t[3] = max(t[3], float(err.max()))
            if verbose:
                print(row)

        if verbose:
            print("-" * len(head))
            row = f"{'ALL':20s} {npts:4d} {'':4s} |"
            for m, _ in methods:
                n, okc, se, mx = totals[m]
                row += f" {100.0 * okc / max(n, 1):13.1f}% {mx:9.4f} |"
            print(row)
            print()
            for m, _ in methods:
                n, okc, se, mx = totals[m]
                print(f"  {m:10s} sign {100.0 * okc / max(n, 1):6.2f}%  "
                      f"mean |err| {se / max(n, 1):8.5f} mm  max {mx:8.4f} mm  "
                      f"{timing[m] * 1e6 / max(npts, 1):9.2f} us/point")
            print("\n  trimesh has no rtree here, so its timing is the naive")
            print("  O(points x faces) path and is NOT representative of its")
            print("  indexed speed. Its accuracy column is trimesh's real answer.")
        return {"totals": totals, "timing": timing, "per_case": per_case,
                "points": npts}
    finally:
        api_core.close_session(sid)


if __name__ == "__main__":
    run()
