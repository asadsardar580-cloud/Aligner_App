"""Geometric self-intersection detection for triangle meshes.

WHY THIS MODULE EXISTS
Every validator in Aligner_App measures INDEX topology (open / non-manifold
edges, components, winding) or COINCIDENT POSITIONS. None asks whether two
non-adjacent triangles actually cross or touch in space. A vertex-normal
offset, a deformation fold-over, or a crown pushed into its neighbour all leave
the index buffer untouched - so every topology gate passes on a surface that
crosses itself. This is the measurement that closes that blind spot.

METHOD
  broad phase   uniform grid over padded triangle AABBs. Conservative: two
                triangles whose padded boxes overlap always share a cell, so a
                pair is never missed (verified against brute force in tests).
  narrow phase  float64. Two triangles intersect iff an edge of one meets the
                other triangle (holds for crossing, touching, coplanar overlap
                and containment). Shared-vertex pairs test only the edges that
                do not contain the shared vertex; edge-adjacent pairs are tested
                for a folded flap (anti-parallel normals).

A pair within `touch_tol_mm` of contact counts as intersecting. In a valid
closed mesh two NON-adjacent triangles never touch, so a touch is itself the
defect (it is exactly what a self-touching boundary is). Run it on the
float32-ROUNDED positions - the geometry the file will actually contain.

Pure NumPy/SciPy. No new dependency.
"""
from __future__ import annotations

import numpy as np

DEFAULT_TOUCH_TOL_MM = 1e-6      # below float32 resolution at arch scale (~4e-6 mm)
FOLD_COS = -0.99999              # edge-adjacent normals this anti-parallel = folded flap


# ---------------------------------------------------------------------------
# Broad phase
# ---------------------------------------------------------------------------

def candidate_pairs(verts, faces, active=None, cell=None, pad=DEFAULT_TOUCH_TOL_MM):
    """Unique face pairs (i < j) whose padded AABBs share a grid cell.

    `active` (bool per face, optional): keep only pairs with >= 1 active face.
    Per stage, pass the faces that have a moved vertex - unchanged faces were
    already proven clean at T0 and cannot start intersecting each other.
    """
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)
    tri = V[F]
    lo = tri.min(axis=1) - pad
    hi = tri.max(axis=1) + pad
    if cell is None:
        ext = (hi - lo).max(axis=1)
        cell = max(float(np.median(ext)) * 1.5, 1e-6)

    origin = lo.min(axis=0)
    ilo = np.floor((lo - origin) / cell).astype(np.int64)
    ihi = np.floor((hi - origin) / cell).astype(np.int64)
    span = ihi - ilo + 1                                  # cells per axis, per face
    count = span.prod(axis=1)

    if active is not None:
        active = np.asarray(active, bool)

    is_giant = count > 1000
    normal_idx = np.flatnonzero(~is_giant)
    giant_idx = np.flatnonzero(is_giant)

    out = []

    if len(normal_idx) > 0:
        n_count = count[normal_idx]
        fid = np.repeat(normal_idx, n_count)
        start = np.repeat(np.cumsum(n_count) - n_count, n_count)
        local = np.arange(n_count.sum()) - start
        
        n_fid = np.repeat(np.arange(len(normal_idx)), n_count)
        sy, sz = span[fid, 1], span[fid, 2]
        ix = ilo[fid, 0] + local // (sy * sz)
        rem = local % (sy * sz)
        iy = ilo[fid, 1] + rem // sz
        iz = ilo[fid, 2] + rem % sz
        dims = ihi.max(axis=0) + 1
        key = (ix * dims[1] + iy) * dims[2] + iz

        order = np.lexsort((fid, key))
        key, fid = key[order], fid[order]
        bounds = np.flatnonzero(np.diff(key)) + 1
        starts = np.r_[0, bounds]
        sizes = np.diff(np.r_[starts, len(key)])

        for g in np.unique(sizes[sizes > 1]):
            grp = starts[sizes == g]
            members = fid[grp[:, None] + np.arange(g)[None, :]]
            if active is not None:
                keep = active[members].any(axis=1)
                members = members[keep]
                if not len(members):
                    continue
            iu, ju = np.triu_indices(g, k=1)
            a = members[:, iu].ravel()
            b = members[:, ju].ravel()
            out.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]))

    if len(giant_idx) > 0:
        for i in giant_idx:
            ov = np.all((lo[i] <= hi) & (lo <= hi[i]), axis=1)
            ov[i] = False
            if active is not None and not active[i]:
                ov &= active
            ov_idx = np.flatnonzero(ov)
            if len(ov_idx) > 0:
                a = np.full(len(ov_idx), i, dtype=np.int64)
                out.append(np.column_stack([np.minimum(a, ov_idx), np.maximum(a, ov_idx)]))
    if not out:
        return np.zeros((0, 2), np.int64)
    pairs = np.vstack(out)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    n = np.int64(len(F))
    uniq = np.unique(pairs[:, 0] * n + pairs[:, 1])
    pairs = np.column_stack([uniq // n, uniq % n])
    if active is not None:
        pairs = pairs[active[pairs[:, 0]] | active[pairs[:, 1]]]
    # Exact AABB overlap filter: a shared cell is necessary, not sufficient.
    ov = np.all((lo[pairs[:, 0]] <= hi[pairs[:, 1]]) & (lo[pairs[:, 1]] <= hi[pairs[:, 0]]), axis=1)
    return pairs[ov]


# ---------------------------------------------------------------------------
# Narrow phase (vectorised, float64)
# ---------------------------------------------------------------------------

def _dot(x, y):
    return np.einsum("ij,ij->i", x, y)


def _unit_normals(a, b, c):
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    return n / np.where(ln > 0, ln, 1.0)[:, None], ln


def _point_seg_dist(x, s0, s1):
    d = s1 - s0
    L = _dot(d, d)
    t = np.where(L > 0, _dot(x - s0, d) / np.where(L > 0, L, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(x - (s0 + d * t[:, None]), axis=1)


def _inside_tri(x, a, b, c, nu, tol):
    """x assumed on the triangle's plane; inside or within `tol` of it."""
    out = np.zeros(len(x), bool)
    ok = np.ones(len(x), bool)
    for p0, p1 in ((a, b), (b, c), (c, a)):
        e = p1 - p0
        le = np.linalg.norm(e, axis=1)
        sd = _dot(np.cross(e, x - p0), nu) / np.where(le > 0, le, 1.0)
        ok &= sd >= -tol
    out |= ok
    # Within tol of any edge (covers slivers where the half-plane test is fragile)
    near = np.minimum(np.minimum(_point_seg_dist(x, a, b), _point_seg_dist(x, b, c)),
                      _point_seg_dist(x, c, a)) <= tol
    return out | near


def _seg_seg_dist_2d(p, q, r, s):
    """Min distance between 2D segments pq and rs (0 if they cross)."""
    def orient(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    o1, o2 = orient(p, q, r), orient(p, q, s)
    o3, o4 = orient(r, s, p), orient(r, s, q)
    cross = (o1 * o2 < 0) & (o3 * o4 < 0)

    def psd(x, s0, s1):
        d = s1 - s0
        L = (d * d).sum(1)
        t = np.where(L > 0, ((x - s0) * d).sum(1) / np.where(L > 0, L, 1.0), 0.0)
        t = np.clip(t, 0, 1)
        return np.linalg.norm(x - (s0 + d * t[:, None]), axis=1)
    d = np.minimum(np.minimum(psd(p, r, s), psd(q, r, s)), np.minimum(psd(r, p, q), psd(s, p, q)))
    return np.where(cross, 0.0, d)


def _seg_tri(p, q, a, b, c, tol):
    """Segment pq meets triangle abc (or comes within tol). Vectorised."""
    nu, ln = _unit_normals(a, b, c)
    degenerate = ln <= 1e-300
    dp = _dot(p - a, nu)
    dq = _dot(q - a, nu)
    sp = np.where(dp > tol, 1, np.where(dp < -tol, -1, 0))
    sq = np.where(dq > tol, 1, np.where(dq < -tol, -1, 0))
    same_side = sp * sq > 0
    coplanar = (sp == 0) & (sq == 0)
    crossing = ~same_side & ~coplanar

    hit = np.zeros(len(p), bool)
    if crossing.any():
        i = np.flatnonzero(crossing)
        den = dp[i] - dq[i]
        t = np.where(np.abs(den) > 0, dp[i] / np.where(np.abs(den) > 0, den, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        x = p[i] + (q[i] - p[i]) * t[:, None]
        x = x - nu[i] * _dot(x - a[i], nu[i])[:, None]          # snap onto plane
        hit[i] = _inside_tri(x, a[i], b[i], c[i], nu[i], tol)
    if coplanar.any():
        i = np.flatnonzero(coplanar & ~degenerate)
        if len(i):
            u = b[i] - a[i]
            u = u / np.linalg.norm(u, axis=1)[:, None]
            v = np.cross(nu[i], u)

            def to2(x):
                d = x - a[i]
                return np.column_stack([_dot(d, u), _dot(d, v)])
            P, Q, A, B, C = to2(p[i]), to2(q[i]), to2(a[i]), to2(b[i]), to2(c[i])
            z = np.zeros((len(i), 1))
            inP = _inside_tri(np.c_[P, z], np.c_[A, z], np.c_[B, z], np.c_[C, z],
                              np.tile([0.0, 0.0, 1.0], (len(i), 1)), tol)
            inQ = _inside_tri(np.c_[Q, z], np.c_[A, z], np.c_[B, z], np.c_[C, z],
                              np.tile([0.0, 0.0, 1.0], (len(i), 1)), tol)
            dmin = np.minimum(np.minimum(_seg_seg_dist_2d(P, Q, A, B), _seg_seg_dist_2d(P, Q, B, C)),
                              _seg_seg_dist_2d(P, Q, C, A))
            hit[i] = inP | inQ | (dmin <= tol)
    return hit


def _pairs_intersect(V, F, pairs, tol):
    """Classify candidate pairs. Returns (hit mask, kind array)."""
    A = F[pairs[:, 0]]
    B = F[pairs[:, 1]]
    shared = (A[:, :, None] == B[:, None, :])                  # (P,3,3)
    ns = shared.any(axis=2).sum(axis=1)                         # shared vertices
    hit = np.zeros(len(pairs), bool)
    kind = np.full(len(pairs), "", dtype=object)

    # 0 shared vertices: any edge of A vs tri B, any edge of B vs tri A.
    i0 = np.flatnonzero(ns == 0)
    if len(i0):
        a, b = A[i0], B[i0]
        h = np.zeros(len(i0), bool)
        for (s, t) in ((0, 1), (1, 2), (2, 0)):
            h |= _seg_tri(V[a[:, s]], V[a[:, t]], V[b[:, 0]], V[b[:, 1]], V[b[:, 2]], tol)
            h |= _seg_tri(V[b[:, s]], V[b[:, t]], V[a[:, 0]], V[a[:, 1]], V[a[:, 2]], tol)
        hit[i0] = h
        kind[i0[h]] = "cross_or_touch"

    # 1 shared vertex: only the edges opposite the shared vertex.
    i1 = np.flatnonzero(ns == 1)
    if len(i1):
        a, b = A[i1], B[i1]
        sh = shared[i1]
        ka = np.argmax(sh.any(axis=2), axis=1)                  # shared slot in A
        kb = np.argmax(sh.any(axis=1), axis=1)                  # shared slot in B
        r = np.arange(len(i1))
        a1, a2 = a[r, (ka + 1) % 3], a[r, (ka + 2) % 3]
        b1, b2 = b[r, (kb + 1) % 3], b[r, (kb + 2) % 3]
        h = _seg_tri(V[a1], V[a2], V[b[:, 0]], V[b[:, 1]], V[b[:, 2]], tol)
        h |= _seg_tri(V[b1], V[b2], V[a[:, 0]], V[a[:, 1]], V[a[:, 2]], tol)
        hit[i1] = h
        kind[i1[h]] = "shared_vertex_overlap"

    # 2 shared vertices: folded flap (normals anti-parallel).
    i2 = np.flatnonzero(ns == 2)
    if len(i2):
        na, _ = _unit_normals(V[A[i2, 0]], V[A[i2, 1]], V[A[i2, 2]])
        nb, _ = _unit_normals(V[B[i2, 0]], V[B[i2, 1]], V[B[i2, 2]])
        h = _dot(na, nb) <= FOLD_COS
        hit[i2] = h
        kind[i2[h]] = "folded_edge"

    # 3 shared vertices: duplicate triangle.
    i3 = np.flatnonzero(ns == 3)
    hit[i3] = True
    kind[i3] = "duplicate_face"
    return hit, kind


def self_intersection_report(verts, faces, active_faces=None,
                             touch_tol_mm=DEFAULT_TOUCH_TOL_MM,
                             chunk=400_000, examples=12):
    """Measure geometric self-intersection. Never raises on bad geometry.

    Returns a dict. `measured` is False only if the test could not run; the
    caller's gate must treat that as a FAILURE (fail closed), never as clean.
    """
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)
    rep = {"measured": False, "method": "grid-AABB broad phase + float64 segment/triangle",
           "touch_tol_mm": float(touch_tol_mm)}
    if not np.isfinite(V).all():
        rep["reason"] = "non-finite coordinates"
        return rep
    pairs = candidate_pairs(V, F, active=active_faces, pad=touch_tol_mm)
    rep["candidate_pairs"] = int(len(pairs))
    hits, kinds = [], []
    for s in range(0, len(pairs), chunk):
        h, k = _pairs_intersect(V, F, pairs[s:s + chunk], touch_tol_mm)
        hits.append(pairs[s:s + chunk][h])
        kinds.append(k[h])
    bad = np.vstack(hits) if hits else np.zeros((0, 2), np.int64)
    kinds = np.concatenate(kinds) if kinds else np.zeros(0, object)
    rep.update({
        "measured": True,
        "intersecting_pairs": int(len(bad)),
        "faces_involved": int(len(np.unique(bad))) if len(bad) else 0,
        "by_kind": {str(k): int((kinds == k).sum()) for k in np.unique(kinds)} if len(kinds) else {},
        "examples": [{"faces": [int(x) for x in bad[j]], "kind": str(kinds[j]),
                      "where": [round(float(t), 4) for t in V[F[bad[j, 0]]].mean(axis=0)]}
                     for j in range(min(examples, len(bad)))],
    })
    return rep
