#!/usr/bin/env python3
"""
core_geometry.py — Clinical Micro-Planner: Core Engine (Modules 2, 3, 4)
=========================================================================
Pure NumPy/SciPy. No Qt, no VTK, no Trimesh. That isolation is the whole
point: everything in this file is testable without a display or a GPU, and
test_core_geometry.py exercises all of it. If a bug appears in the running
app, this module is the part you can bisect deterministically.

Mesh convention throughout: `verts` is (N,3) float, `faces` is (M,3) int.
Triangles only.
"""

from __future__ import annotations

import heapq
import math
import warnings
import numpy as np
from scipy.spatial import Delaunay


# =========================================================================
# Mesh topology primitives
# =========================================================================

def build_vertex_adjacency(faces: np.ndarray, n_verts: int) -> list[set[int]]:
    """Undirected vertex-neighbour sets from a triangle list."""
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    for a, b, c in faces:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def edge_face_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Map each undirected edge -> list of face indices touching it."""
    inc: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            inc.setdefault(key, []).append(fi)
    return inc


def manifold_report(faces: np.ndarray) -> dict:
    """Why a mesh is not watertight, not merely that it is not.

    is_edge_manifold_closed answers yes/no, which is the wrong granularity for
    a clinician staring at a refused export. The two failures have completely
    different causes and remedies:

      * OPEN edges (one face) are holes — a cut that left a second loop, or a
        boundary capping did not reach.
      * NON-MANIFOLD edges (three or more faces) are almost never caused by
        cutting. They arrive in the uploaded scan: intraoral scanners produce
        them where the mesh folds back on itself, and condition_mesh does not
        repair them (it welds, drops degenerates and debris, and fills only
        small holes). One such edge anywhere on the arch blocks export forever,
        and blaming "a cut left a second open loop" sends the clinician hunting
        in the wrong place.
    """
    counts = _edge_face_counts(faces)
    if len(counts) == 0:
        return {"watertight": True, "open_edges": 0, "nonmanifold_edges": 0,
                "total_edges": 0, "worst_edge_faces": 0}
    open_e = int((counts == 1).sum())
    nonman = int((counts > 2).sum())
    return {
        "watertight": not open_e and not nonman,
        "open_edges": open_e,
        "nonmanifold_edges": nonman,
        "total_edges": int(len(counts)),
        "worst_edge_faces": int(counts.max()),
    }


def _edge_key(faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Undirected edge keys, one per half-edge, in face order.

    Half-edges are emitted per face as (a,b), (b,c), (c,a) — the same order
    edge_face_incidence walks them — so first-occurrence order is preserved and
    callers that relied on the dict's insertion order still see it.
    """
    f = np.asarray(faces, np.int64)
    if len(f) == 0:
        return np.empty(0, np.int64), 1
    he = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1).reshape(-1, 2)
    n = int(f.max()) + 1
    lo = np.minimum(he[:, 0], he[:, 1])
    hi = np.maximum(he[:, 0], he[:, 1])
    return lo * n + hi, n


def _edge_face_counts(faces: np.ndarray) -> np.ndarray:
    """How many faces touch each undirected edge. Vectorised.

    edge_face_incidence builds a Python dict with three insertions per face,
    which measures 1.1-1.6s on an arch and is called from manifold_report,
    boundary_loops and _boundary_half_edges — so it was most of the cost of
    building a cast base. The dict form is kept for callers that need the face
    LISTS; this is for the callers that only need the counts.
    """
    key, _ = _edge_key(faces)
    if len(key) == 0:
        return np.empty(0, np.int64)
    return np.unique(key, return_counts=True)[1]


def is_edge_manifold_closed(faces: np.ndarray) -> bool:
    """True iff every edge is shared by exactly two faces — the topological
    precondition for a watertight solid. This is the check that decides
    whether an STL is safe to hand to a boolean engine or a printer."""
    inc = edge_face_incidence(faces)
    return all(len(f) == 2 for f in inc.values())


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Ordered vertex loops along the open boundary (edges with one face).
    Returns each loop as a cyclic list of vertex indices."""
    key, n = _edge_key(faces)
    if len(key) == 0:
        return []
    uniq, first, counts = np.unique(key, return_index=True, return_counts=True)
    sel = counts == 1
    if not sel.any():
        return []
    # First-occurrence order, matching what the edge_face_incidence dict used to
    # give. The walk below starts from `nbr`'s insertion order, so preserving it
    # keeps loop starts and directions byte-identical to the previous build.
    ks = uniq[sel][np.argsort(first[sel])]
    border = [(int(k // n), int(k % n)) for k in ks]

    nbr: dict[int, list[int]] = {}
    for u, v in border:
        nbr.setdefault(u, []).append(v)
        nbr.setdefault(v, []).append(u)

    loops, seen_edges = [], set()
    for start in nbr:
        for first_step in nbr[start]:
            e0 = (start, first_step) if start < first_step else (first_step, start)
            if e0 in seen_edges:
                continue
            loop, prev, cur = [start], start, first_step
            seen_edges.add(e0)
            while cur != start:
                loop.append(cur)
                nxt = None
                for cand in nbr.get(cur, []):
                    ek = (cur, cand) if cur < cand else (cand, cur)
                    if cand != prev and ek not in seen_edges:
                        nxt = cand
                        break
                if nxt is None:
                    break
                seen_edges.add((cur, nxt) if cur < nxt else (nxt, cur))
                prev, cur = cur, nxt
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln < 1e-12, 1.0, ln)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = face_normals(verts, faces)
    vn = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vn, faces[:, i], fn)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(ln < 1e-12, 1.0, ln)


# =========================================================================
# Module 2a: curvature — finding the cervical sulcus
# =========================================================================

def vertex_concavity(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Per-vertex concavity estimate, normalized to roughly [-1, 1].
    POSITIVE = concave (a valley — the cervical sulcus reads strongly here).
    NEGATIVE = convex (a ridge — cusps and the crown bulge).

    Method: mean projection of edge vectors onto the outward vertex normal,
    scaled by local edge length. In a valley the neighbours sit outward
    along the normal (positive projection); on a ridge they fall away
    (negative). This is a discrete mean-curvature proxy, not the exact
    cotangent-Laplacian mean curvature — chosen because it is O(E), stable
    on the irregular triangulations intraoral scanners produce, and needs
    no area weighting to behave. Exact H would be sharper on clean meshes
    and noisier on real scan data.
    """
    n_verts = len(verts)
    vn = vertex_normals(verts, faces)
    adj = build_vertex_adjacency(faces, n_verts)

    conc = np.zeros(n_verts)
    for i in range(n_verts):
        nb = adj[i]
        if not nb:
            continue
        d = verts[list(nb)] - verts[i]
        ln = np.linalg.norm(d, axis=1)
        ok = ln > 1e-12
        if not np.any(ok):
            continue
        conc[i] = np.mean((d[ok] / ln[ok, None]) @ vn[i])

    m = np.max(np.abs(conc))
    return conc / m if m > 1e-12 else conc


def smooth_scalar(values: np.ndarray, faces: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Laplacian smoothing of a per-vertex scalar. Scan noise produces
    speckled curvature; without this the cut path chases individual noisy
    vertices instead of the actual anatomical groove."""
    adj = build_vertex_adjacency(faces, len(values))
    out = values.copy()
    for _ in range(iterations):
        prev = out.copy()
        for i in range(len(out)):
            nb = adj[i]
            if nb:
                out[i] = 0.5 * prev[i] + 0.5 * np.mean(prev[list(nb)])
    return out


# =========================================================================
# Module 2b: magnetic scissors — curvature-weighted surface path
# =========================================================================

def curvature_weighted_path(
    verts: np.ndarray,
    faces: np.ndarray,
    concavity: np.ndarray,
    start: int,
    goal: int,
    concavity_weight: float = 8.0,
) -> list[int]:
    """
    Dijkstra along mesh EDGES from `start` to `goal`, where edge cost is
    geometric length inflated on convex terrain and discounted in valleys:

        cost = length * (1 + w * (1 - concavity_normalized))

    So the path prefers to travel through the groove even when that is
    geometrically longer — the 3D analogue of Photoshop's magnetic lasso
    snapping to an edge. Raising `concavity_weight` makes it hug the sulcus
    harder; lowering it makes it cut straighter. 8.0 is a starting value,
    not a tuned constant — it should be exposed in the UI.

    Never leaves the mesh surface, so there is no 2D->3D projection step and
    therefore none of the inside/out ambiguity that sank the lasso approach.
    """
    adj = build_vertex_adjacency(faces, len(verts))
    # map concavity from [-1,1] to a cost multiplier in [1, 1+2w]
    penalty = 1.0 + concavity_weight * (1.0 - concavity)

    dist = np.full(len(verts), np.inf)
    prev = np.full(len(verts), -1, dtype=int)
    dist[start] = 0.0
    pq = [(0.0, start)]
    visited = np.zeros(len(verts), dtype=bool)

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        if u == goal:
            break
        for v in adj[u]:
            if visited[v]:
                continue
            length = float(np.linalg.norm(verts[v] - verts[u]))
            step = length * 0.5 * (penalty[u] + penalty[v])
            nd = d + step
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[goal]):
        return []
    path, cur = [], goal
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def nearest_vertex(verts: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(verts - point, axis=1)))


def closed_loop_through_anchors(
    verts: np.ndarray,
    faces: np.ndarray,
    concavity: np.ndarray,
    anchor_points: np.ndarray,
    concavity_weight: float = 8.0,
) -> list[int]:
    """
    Chain curvature-weighted paths through the clinician's anchor clicks and
    back to the first, yielding a closed vertex loop around the crown.

    The mesial and distal contact clicks from Module 3 sit exactly on the
    interproximal boundary, so the same three landmarks serve double duty:
    they anchor the cut AND define the anatomical axes. Adding 2-4 more
    clicks lingually tightens a difficult margin.

    Consecutive duplicates are dropped so the loop stays a simple cycle.
    """
    anchors = [nearest_vertex(verts, p) for p in anchor_points]
    loop: list[int] = []
    for i in range(len(anchors)):
        seg = curvature_weighted_path(
            verts, faces, concavity, anchors[i], anchors[(i + 1) % len(anchors)], concavity_weight
        )
        if not seg:
            return []
        loop.extend(seg[:-1])  # drop endpoint; next segment supplies it

    dedup: list[int] = []
    for v in loop:
        if not dedup or dedup[-1] != v:
            dedup.append(v)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup


# =========================================================================
# Module 2c: topological dissection
# =========================================================================

def split_mesh_by_loop(
    verts: np.ndarray,
    faces: np.ndarray,
    loop: list[int],
    seed_point: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Cut the mesh along a closed vertex loop into (crown, base).

    Flood-fills across face adjacency, refusing to cross any edge that lies
    on the loop. The component containing the face nearest `seed_point`
    (the FA click) is the crown — so which side is which is DETERMINED by
    where the clinician clicked, never inferred from winding order or
    region size. That removes the inside/out swap failure mode by
    construction rather than by heuristic.

    Both halves are re-indexed independently, so loop vertices are
    duplicated and each half owns its own open boundary — exactly what the
    capping step needs.
    """
    loop_edges = set()
    L = len(loop)
    for i in range(L):
        u, v = loop[i], loop[(i + 1) % L]
        loop_edges.add((u, v) if u < v else (v, u))

    inc = edge_face_incidence(faces)
    face_adj: list[list[int]] = [[] for _ in range(len(faces))]
    for edge, fl in inc.items():
        if len(fl) == 2 and edge not in loop_edges:
            face_adj[fl[0]].append(fl[1])
            face_adj[fl[1]].append(fl[0])

    centroids = verts[faces].mean(axis=1)
    seed_face = int(np.argmin(np.linalg.norm(centroids - seed_point, axis=1)))

    crown_mask = np.zeros(len(faces), dtype=bool)
    stack = [seed_face]
    crown_mask[seed_face] = True
    while stack:
        f = stack.pop()
        for g in face_adj[f]:
            if not crown_mask[g]:
                crown_mask[g] = True
                stack.append(g)

    def extract(mask):
        sub = faces[mask]
        used = np.unique(sub)
        remap = -np.ones(len(verts), dtype=int)
        remap[used] = np.arange(len(used))
        return verts[used].copy(), remap[sub]

    return extract(crown_mask), extract(~crown_mask)


# =========================================================================
# Module 2d: watertight capping
# =========================================================================

def cap_boundary_loop(
    verts: np.ndarray,
    loop_idx: list[int],
) -> np.ndarray:
    """
    Triangulate an open boundary loop into a flat cap.

    Best-fit plane by PCA -> project to 2D -> Delaunay -> discard triangles
    whose centroid falls outside the loop polygon (Delaunay triangulates the
    convex hull, and a cervical margin is usually mildly concave lingually,
    so unfiltered output would bridge across the opening).

    Falls back to a centroid fan when Delaunay degenerates or the filter
    leaves too few triangles. The fan is uglier and can self-intersect on a
    strongly non-convex loop, but it always produces a closed surface — and
    it lives inside the socket where no one sees it. Returns faces indexed
    into `verts`, possibly with one appended centroid vertex (see
    cap_and_close).
    """
    pts = verts[loop_idx]
    centroid = pts.mean(axis=0)
    # full_matrices=False: only vt is read, and the default builds an (N,N) U
    # for an N-vertex loop just to throw it away.
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    pts2d = np.column_stack([(pts - centroid) @ e1, (pts - centroid) @ e2])

    try:
        tri = Delaunay(pts2d)
        keep = []
        for simplex in tri.simplices:
            c = pts2d[simplex].mean(axis=0)
            if _point_in_polygon(c, pts2d):
                keep.append([loop_idx[simplex[0]], loop_idx[simplex[1]], loop_idx[simplex[2]]])
        if len(keep) >= max(3, len(loop_idx) // 4):
            return np.array(keep, dtype=int)
    except Exception as e:
        # SAY SO. This was a silent `pass`, and it is the only silent swallow on
        # the live geometry path - directly under hole capping, which both
        # condition_mesh and cap_and_close depend on. Silent, a genuine
        # numerical failure in Delaunay is indistinguishable from the benign
        # "produced too few interior triangles" case three lines above, and both
        # end up in the same fan fallback with no way to tell which happened.
        # The fallback is correct and stays; only the silence was wrong.
        warnings.warn(
            f"cap_boundary_loop: Delaunay failed on a {len(loop_idx)}-vertex loop "
            f"({type(e).__name__}: {e}). Falling back to a centroid fan, which is "
            f"topologically safe but coarser.",
            RuntimeWarning, stacklevel=2)
    return np.empty((0, 3), dtype=int)  # caller applies the fan fallback


def _point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting parity test."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-30)
            if x < xint:
                inside = not inside
    return inside


def _split_self_touching_loop(loop) -> list[list[int]]:
    """Break a boundary walk that revisits a vertex into simple sub-cycles.

    A boundary loop is not guaranteed to be a simple cycle: where the surface
    pinches to a single vertex, the walk passes through it twice. Fanning such a
    loop from one centroid creates the spoke to that vertex TWICE, giving the
    edge four faces.

    That is worth stating plainly because cap_and_close's docstring claims the
    fan "cannot hit this" — every edge it introduces terminates at a brand new
    centroid vertex, so no EXISTING edge can gain faces. True, and it misses the
    case above, where the fan collides with its own other spoke. Measured on a
    trimmed cast band: 3 edges with 4 faces each, every one of them a spoke to a
    hole-fill centroid.

    Each sub-cycle keeps its consecutive pairs from the original walk, so they
    are all still boundary edges. The pinch vertex ends up shared between two
    fans, which is a non-manifold VERTEX — allowed, and what the surface already
    was — rather than a non-manifold edge, which is not.
    """
    loops: list[list[int]] = []
    stack: list[int] = []
    seen: dict[int, int] = {}
    for v in loop:
        v = int(v)
        if v in seen:
            k = seen[v]
            cycle = stack[k:]
            if len(cycle) >= 3:
                loops.append(cycle)
            for u in cycle:
                seen.pop(u, None)
            del stack[k:]
        seen[v] = len(stack)
        stack.append(v)
    if len(stack) >= 3:
        loops.append(stack)
    return loops


def _creates_nonmanifold(faces: np.ndarray, cap: np.ndarray) -> bool:
    """True if appending `cap` would give any edge more than two faces."""
    combined = np.vstack([faces, cap])
    return any(len(fl) > 2 for fl in edge_face_incidence(combined).values())


def _boundary_half_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    """Directed half-edges lying on the open boundary, in the direction the
    single adjacent face traverses them."""
    f = np.asarray(faces, np.int64)
    if len(f) == 0:
        return set()
    he = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1).reshape(-1, 2)
    key, _ = _edge_key(f)
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sel = counts[inv] == 1
    return {(int(a), int(b)) for a, b in he[sel]}


def _orient_cap_against_boundary(cap: np.ndarray, half_edges: set[tuple[int, int]]) -> np.ndarray:
    """
    Flip cap triangles so each shared boundary edge is traversed in the
    OPPOSITE direction to the surface face already using it — the condition
    for consistent orientation across the seam.

    Doing this locally matters for performance, not just tidiness:
    make_consistent_winding() is a Python BFS over every face in the mesh,
    which is acceptable for an isolated crown but would run for tens of
    seconds on a 350k-face arch base. The cap is the only newly created
    geometry, and scanner STLs already arrive consistently wound, so
    orienting just the seam is both sufficient and O(cap).
    """
    cap = cap.copy()
    for k, (a, b, c) in enumerate(cap):
        for u, v in ((a, b), (b, c), (c, a)):
            if (int(u), int(v)) in half_edges:   # same direction as the surface -> wrong
                cap[k] = cap[k][::-1]
                break
    return cap


def cap_and_close(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cap every open boundary loop, returning a closed mesh.

    Delaunay first (better triangle quality, flat cap), centroid fan as
    fallback — but the Delaunay result is VALIDATED before acceptance,
    because of a failure mode found in testing and worth stating plainly:

    a cervical margin loop is not planar. Projecting it to a best-fit plane
    and running Delaunay can produce a chord between two loop vertices that
    ALREADY share an edge elsewhere on the crown surface. Appending that cap
    gives the shared edge four incident faces. The mesh then has no open
    boundary — it passes a naive "any holes left?" check — while being
    non-manifold, which is exactly the input that makes a CSG boolean fail
    or silently return a wrong solid.

    The fan cannot hit this: every edge it introduces terminates at a newly
    created centroid vertex, so no EXISTING edge can gain extra faces. It is
    the geometrically worse but topologically guaranteed option, and it sits
    inside the socket where nobody sees it.

    One qualification to that guarantee, found while building the cast base and
    corrected here: a boundary walk that pinches through the same vertex twice
    makes the fan collide with its OWN other spoke, giving that edge four faces.
    Splitting the walk into simple sub-cycles first is what keeps the claim
    true. See _split_self_touching_loop.
    """
    verts = verts.copy()
    faces = faces.copy()

    for loop in boundary_loops(faces):
        half_edges = _boundary_half_edges(faces)

        cap = cap_boundary_loop(verts, loop)
        if len(cap) > 0 and not _creates_nonmanifold(faces, cap):
            faces = np.vstack([faces, _orient_cap_against_boundary(cap, half_edges)])
            continue

        for cycle in _split_self_touching_loop(loop):
            verts = np.vstack([verts, verts[cycle].mean(axis=0)])
            ci = len(verts) - 1
            fan = np.array(
                [[cycle[i], cycle[(i + 1) % len(cycle)], ci] for i in range(len(cycle))],
                dtype=int,
            )
            faces = np.vstack([faces, _orient_cap_against_boundary(fan, half_edges)])

    return verts, faces


def make_consistent_winding(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Propagate a consistent orientation across the mesh, then flip everything
    if the signed volume comes out negative (i.e. normals pointing inward).

    Necessary because capped triangles arrive with arbitrary winding, and a
    boolean engine handed inconsistently-wound input produces either a crash
    or — worse — a plausible-looking but wrong solid.
    """
    faces = faces.copy()
    inc = edge_face_incidence(faces)
    adj: list[list[int]] = [[] for _ in range(len(faces))]
    for fl in inc.values():
        if len(fl) == 2:
            adj[fl[0]].append(fl[1])
            adj[fl[1]].append(fl[0])

    oriented = np.zeros(len(faces), dtype=bool)
    for seed in range(len(faces)):
        if oriented[seed]:
            continue
        oriented[seed] = True
        stack = [seed]
        while stack:
            f = stack.pop()
            fa = faces[f]
            for g in adj[f]:
                if oriented[g]:
                    continue
                shared = set(fa) & set(faces[g])
                if len(shared) == 2:
                    a, b = tuple(shared)
                    if _same_direction(fa, a, b) == _same_direction(faces[g], a, b):
                        faces[g] = faces[g][::-1]
                oriented[g] = True
                stack.append(g)

    if signed_volume(verts, faces) < 0:
        faces = faces[:, ::-1]
    return faces


def _same_direction(face: np.ndarray, a: int, b: int) -> bool:
    """True if edge (a,b) traverses `face` in its winding direction."""
    f = list(face)
    return f[(f.index(a) + 1) % 3] == b


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2))) / 6.0)


# =========================================================================
# Module 3: anatomical frame + C_res
# =========================================================================

def derive_anatomical_frame(
    mesial_pt: np.ndarray,
    distal_pt: np.ndarray,
    fa_pt: np.ndarray,
    crown_verts: np.ndarray,
    rim_verts: np.ndarray,
) -> dict:
    """
    Build an ORTHONORMAL local frame from the three landmark clicks.

    Why this differs from the master-plan formula: the plan specifies
    u_MD = mesial->distal, u_BL = FA surface normal, u_OA = u_MD x u_BL.
    The cross product does give a u_OA perpendicular to both — but u_MD and
    u_BL are not perpendicular TO EACH OTHER (the FA normal has no reason to
    be square to the contact-to-contact line on a real tooth). That is not a
    frame, it is a skewed basis, and rotating about skewed axes shears the
    tooth slightly on every slider move. The error is small enough to look
    fine on screen and large enough to be wrong on the printed model.

    Fix: Gram-Schmidt. u_MD is kept exactly as clicked (it is the most
    clinically meaningful direction). The FA normal is orthogonalized
    against it to give u_BL. u_OA closes the frame. Same landmarks, same
    clinical meaning, guaranteed perpendicular.

    Signs are then pinned to anatomy, not to arbitrary cross-product
    handedness: u_BL points away from the crown body (buccal), u_OA points
    away from the cut rim (occlusal).
    """
    u_md = distal_pt - mesial_pt
    n = np.linalg.norm(u_md)
    if n < 1e-9:
        raise ValueError("Mesial and distal clicks coincide — cannot define the mesiodistal axis.")
    u_md = u_md / n

    crown_centroid = crown_verts.mean(axis=0)
    fa_out = fa_pt - crown_centroid          # outward-ish at the facial surface
    fa_out = fa_out - np.dot(fa_out, u_md) * u_md   # <-- Gram-Schmidt
    n = np.linalg.norm(fa_out)
    if n < 1e-9:
        raise ValueError("FA click is collinear with the mesiodistal axis — re-click the facial surface.")
    u_bl = fa_out / n

    u_oa = np.cross(u_md, u_bl)
    u_oa = u_oa / np.linalg.norm(u_oa)

    rim_centroid = rim_verts.mean(axis=0)
    if np.dot(u_oa, crown_centroid - rim_centroid) < 0:
        u_oa = -u_oa                      # occlusal = away from the cervical rim
        u_md = -u_md                      # keep the basis right-handed

    return dict(
        centroid=crown_centroid,
        u_md=u_md,
        u_bl=u_bl,
        u_oa=u_oa,
        rim_centroid=rim_centroid,
    )


# -------------------------------------------------------------------------
# Long-axis resolution against the occlusal plane
# -------------------------------------------------------------------------
# THE TETHERBALL GLITCH, and why the estimator had to change.
#
# The old long axis was normalize(crown_centroid - rim_centroid). On an
# incisor those centroids are ~5mm apart and the axis is crisp. On a molar
# the crown is short and wide, so the vertical separation collapses to ~1mm
# while lateral asymmetry -- an uneven wand selection, plus the vertices
# cap_and_close adds over the rim -- displaces the crown centroid sideways by
# a comparable amount. The axis then points SIDEWAYS, C_res is extrapolated
# ~10mm along it into empty space outside the tooth, and the gizmo dutifully
# rotates the crown about a point that is not in the patient. Measured on a
# simulated flat molar (5.5mm rim radius, 2.2mm crown height, rim ring offset
# 1.2mm laterally):
#
#     centroid difference                        28.5 deg off the true axis
#     centroid difference, lopsided selection    37.0 deg mean, 38.8 worst
#     rim plane normal                            0.0 deg
#     rim plane normal, 15% of rim dropped        0.07 deg mean, 0.20 worst
#
# The cervical rim is an anatomical ring around the tooth neck, near
# perpendicular to the long axis, and a BROAD flat molar rim defines that
# plane BETTER than a narrow one -- the estimator improves exactly where the
# old one collapsed. Same technique as derive_frame_from_margin.
#
# Fitting a plane is still not enough on its own, because nothing in the
# crown geometry knows which way is apical. That is what the occlusal plane
# (arch_frame.fit_occlusal_frame) was established for, so the axis is signed
# and reconciled against it here rather than trusted blindly.

MIN_RIM_POINTS = 20           # fewer than this is not a ring worth fitting.
MIN_RIM_RING_RATIO = 0.15     # s1/s0: below this the rim is a LINE, not a ring,
                              # and its in-plane axes are degenerate. A true
                              # sliver measures 0.010; a mesiodistally elongated
                              # molar ring 0.80; a circular ring 1.00. Loose on
                              # purpose -- see below for why it is the only
                              # singular-value gate here.
MAX_AXIS_DEVIATION_DEG = 20.0 # how far a tooth may lean from the arch apical


# HOW THE TOOTH'S LONG AXIS IS DERIVED. Two formulations exist and they are NOT
# equivalent; this selects between them.
#
#   "rim_plane"     (DEFAULT, and what every measurement in CLAUDE.md was taken
#                   against) u_OA is the normal of the best-fit plane through
#                   the cervical rim, signed occlusally and clamped to
#                   MAX_AXIS_DEVIATION_DEG of the arch apical direction. u_BL is
#                   then u_OA x u_MD.
#
#   "cross_product" u_MD and u_BL are primary and u_OA = u_MD x u_BL, which is
#                   the textbook definition of an anatomical triad.
#
# WHY THE DEFAULT IS NOT THE TEXTBOOK ONE. u_BL has no independent landmark in a
# two-click cut - it is derived from the arch frame's buccal direction, which is
# an ARCH-level quantity, not a per-tooth one. Deriving the per-tooth long axis
# from it therefore imports the arch's average inclination into every tooth and
# discards the tooth's own cervical rim, which is the one piece of per-tooth
# anatomy actually present in the scan. The rim-normal formulation was adopted
# after the tetherball glitch (CLAUDE.md §5), where an axis taken from crown
# geometry collapsed on a short broad molar and put C_res outside the tooth.
#
# Both are available so the difference can be MEASURED on a real case rather
# than argued. Change the default only with numbers.
LONG_AXIS_MODE = "rim_plane"                              # direction before the software pulls it back.
                              # A genuinely tipped molar keeps its inclination;
                              # a noise-dominated axis cannot survive.

# WHY THERE IS NO "PLANARITY" GATE, having tried one.
#
# The obvious conditioning test is s2/s1 -- "reject the rim if its plane fit is
# not flat enough". It is worse than useless here, because it is ANTI-correlated
# with the error it is supposed to detect. Measured on synthetic rims:
#
#   cos(2th) scallop, full ring     tilt 0.000 deg   s2/s1 0.00 -> 0.75
#   partial arc 360 deg             tilt   0.16 deg  s2/s1 0.301
#   partial arc 120 deg             tilt  43.26 deg  s2/s1 0.041
#   gingival leak lobe 6mm/120 deg  tilt  34.08 deg  s2/s1 0.063
#
# A cervical margin is scalloped interproximally, and a pure cos(2th) scallop is
# Fourier-orthogonal to the cos(th)/sin(th) modes a plane fit actually uses: it
# inflates s2 to 0.75 while tilting the normal by exactly zero. Meanwhile the
# perturbations that DO tilt the normal -- a partial rim, a lobe of gingiva --
# are 1th-heavy and REDUCE s2/s1. So any threshold tight enough to mean
# something rejects perfect anterior fits and waves through 43-degree errors.
#
# The 1th mode cannot be gated away in principle, because a genuine tooth
# inclination is also a 1th mode -- they are the same signal. That is precisely
# why the arch's occlusal plane is the authority: the clamp below catches every
# case in the table (9, 15, 27, 34, 43 deg all pulled back to 20) without having
# to tell noise from anatomy in the rim alone.
#
# s2/s1 is still REPORTED as rim_planarity, because it is informative to a human
# reading a cut report. It just must not gate anything.


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors, via arctan2 rather than arccos.

    arccos(a.b) loses precision exactly where it matters here: near 0 deg its
    derivative is infinite, so a dot product of 0.9999999 (rounding noise on
    two nominally identical axes) reads as a spurious 0.026 deg. arctan2 of
    the cross-product norm against the dot is well conditioned across the
    whole range.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.degrees(np.arctan2(float(np.linalg.norm(np.cross(a, b))),
                                       float(a @ b))))


def resolve_long_axis(
    rim_verts: np.ndarray,
    crown_centroid: np.ndarray,
    arch_frame: dict | None = None,
    max_axis_deviation_deg: float = MAX_AXIS_DEVIATION_DEG,
) -> tuple[np.ndarray, dict]:
    """
    The tooth's occlusoapical axis, pointing OCCLUSALLY (apical is -u_oa).

    Four outcomes, all reported in the returned info dict so a clinician can
    see when the software overrode the geometry rather than having it happen
    invisibly:

        "rim_plane"           the rim fit well and agrees with the arch
        "rim_plane_clamped"   the rim fit well but leaned too far; pulled back
                              to exactly max_axis_deviation_deg
        "arch_apical"         the rim could not define a plane; the arch decides
        "centroid_difference" no arch frame supplied (legacy callers) and the
                              rim was degenerate -- the old, weak estimator

    `arch_frame` is the dict from arch_frame.fit_occlusal_frame; only its
    "u_occ" is required. Passing None keeps the pre-existing behaviour for
    callers that have no occlusal plane (app_ui.py and the older tests).
    """
    rim = np.asarray(rim_verts, float)
    crown_centroid = np.asarray(crown_centroid, float)
    rim_centroid = rim.mean(axis=0) if len(rim) else crown_centroid

    # --- candidate: normal of the best-fit plane through the cervical rim ---
    axis, planarity, ring_ratio = None, None, None
    # The count check must come BEFORE the SVD, not after: with
    # full_matrices=False an (N,3) input with N < 3 returns s and vt of length
    # N, so s[2] and vt[2] raise IndexError rather than degrading gracefully.
    if len(rim) >= MIN_RIM_POINTS:
        # full_matrices=False matters, and not as a micro-optimisation: the
        # default builds an (N,N) U to read three numbers out of vt. On a
        # 4000-point rim that is a 128MB allocation and 343ms against 0.2ms.
        _, s, vt = np.linalg.svd(rim - rim_centroid, full_matrices=False)
        if s[0] > 1e-9:
            planarity = float(s[2] / s[1]) if s[1] > 1e-12 else float("inf")
            ring_ratio = float(s[1] / s[0])
            if ring_ratio >= MIN_RIM_RING_RATIO:
                # vt rows are already unit length; normalising again would only
                # hide a degenerate fit rather than catch it.
                axis = vt[2]

    source = "rim_plane"
    if axis is None:
        source = "centroid_difference"
        d = crown_centroid - rim_centroid
        if np.linalg.norm(d) > 1e-9:
            axis = d / np.linalg.norm(d)

    u_occ = None
    if arch_frame is not None:
        u_occ = np.asarray(arch_frame["u_occ"], float)
        n = np.linalg.norm(u_occ)
        if n < 1e-9:
            raise ValueError("arch_frame['u_occ'] is a zero vector.")
        u_occ = u_occ / n

    info = dict(axis_source=source, axis_deviation_deg=None, axis_corrected=False,
                rim_planarity=(None if planarity is None else round(planarity, 4)),
                rim_ring_ratio=(None if ring_ratio is None else round(ring_ratio, 4)),
                max_axis_deviation_deg=float(max_axis_deviation_deg))

    if axis is None:
        if u_occ is None:
            raise ValueError(
                "Cannot derive a long axis: the cervical rim is degenerate and the "
                "crown centroid coincides with the rim centroid. Re-cut the tooth.")
        info.update(axis_source="arch_apical", axis_corrected=True)
        return u_occ.copy(), info

    # --- sign, BEFORE any cross product -----------------------------------
    # Ordering is load-bearing. Signing the axis occlusally first guarantees
    # axis.u_occ >= 0, which is what removes the 180deg degeneracy from the
    # clamp below: an anti-parallel pair has a vanishing cross product and no
    # defined rotation axis.
    if u_occ is not None:
        if axis @ u_occ < 0:
            axis = -axis
    else:
        # No occlusal plane to appeal to. The crown-minus-rim vector is far too
        # noisy to set the AXIS on a flat molar, but it is still reliable for
        # the SIGN -- even at 47 degrees of lateral error it names the correct
        # hemisphere. It stops being reliable when the crown barely rises above
        # the rim plane at all, and an inverted axis puts C_res above the crown:
        # smooth, confident, entirely wrong movement.
        h = float((crown_centroid - rim_centroid) @ axis)
        if abs(h) < 0.5:
            raise ValueError(
                f"Cannot tell which way is occlusal: the crown rises only "
                f"{abs(h):.2f}mm above the rim plane along its own axis, so the "
                f"sign of the long axis is a coin flip. Establish the occlusal "
                f"plane (arch_frame) and pass it in — that is what it is for.")
        if h < 0:
            axis = -axis                    # occlusal = away from the rim

    if u_occ is None:
        return axis, info

    deviation = angle_between_deg(axis, u_occ)
    info["axis_deviation_deg"] = round(deviation, 3)

    if source == "centroid_difference":
        # The rim could not define a plane AND we have an occlusal plane. The
        # arch is the better authority, full stop -- this is the molar case
        # that put C_res outside the tooth.
        info.update(axis_source="arch_apical", axis_corrected=True)
        return u_occ.copy(), info

    if deviation > max_axis_deviation_deg:
        k = np.cross(axis, u_occ)
        nk = np.linalg.norm(k)
        if nk < 1e-9:
            info.update(axis_source="arch_apical", axis_corrected=True)
            return u_occ.copy(), info
        axis = rotation_matrix_axis_angle(k / nk, deviation - max_axis_deviation_deg) @ axis
        axis = axis / np.linalg.norm(axis)
        info.update(axis_source="rim_plane_clamped", axis_corrected=True)

    return axis, info


# The PHYSIOLOGICAL band for the distance C_res is projected apical of the
# crown. Wheeler's own root lengths span 9-13mm across the dentition (incisor
# 10, canine 13, premolar and molar 9), so 7-15 is that range with roughly 2mm
# of headroom either side for a short-rooted or a long-rooted patient.
#
# This is NOT validation.ROOT_LENGTH_MIN/MAX (4-30mm), and the two must not be
# conflated. That band REFUSES a typo — 0mm or 100mm is not a prescription. This
# one CLAMPS a plausible-but-out-of-envelope value, because C_res is extrapolated
# along the long axis by exactly this distance and a 20mm projection puts the
# pivot through the inferior alveolar canal. Refusing there would block a cut
# over a slider position; clamping silently would move the axis a tooth rotates
# about without saying so. So it clamps AND says so, every time.
CRES_PROJECTION_MIN_MM = 7.0
CRES_PROJECTION_MAX_MM = 15.0


def clamp_cres_projection(root_length_mm: float, tooth_id=None, fdi=None):
    """Clamp into the physiological band and describe what happened.

    Returns (clamped_mm, note_or_None). The note is a structured dict carrying
    the requested value, the clamped value and the bound that fired — never a
    bare string, because a refusal or an adjustment that cannot be inspected
    cannot be argued with.
    """
    requested = float(root_length_mm)
    if not math.isfinite(requested):
        # Every comparison against NaN is False, so the clamp below would pass a
        # NaN straight through and C_res would come out NaN — which then passes
        # the alveolus gate for exactly the same reason (CLAUDE.md §14).
        raise ValueError(f"root_length_mm must be finite, got {requested!r}")

    lo, hi = CRES_PROJECTION_MIN_MM, CRES_PROJECTION_MAX_MM
    clamped = min(max(requested, lo), hi)
    if clamped == requested:
        return clamped, None

    note = {
        "event": "cres_projection_clamped",
        "tooth_id": tooth_id, "fdi": fdi,
        "requested_mm": round(requested, 4),
        "clamped_mm": round(clamped, 4),
        "shift_mm": round(abs(requested - clamped), 4),
        "bound": "min" if clamped == lo else "max",
        "band_mm": [lo, hi],
        "provenance": "software heuristic, envelope around Wheeler root lengths",
        "detail": (
            f"C_res projection clamped from {requested:.2f}mm to {clamped:.2f}mm "
            f"for tooth {tooth_id or fdi or '?'}. The pivot therefore sits "
            f"{abs(requested - clamped):.2f}mm from where the requested root "
            f"length would have put it, which changes the arc of every rotation "
            f"on this tooth. Set a root length inside {lo:.0f}-{hi:.0f}mm to "
            f"control the pivot directly."),
    }
    warnings.warn(note["detail"], RuntimeWarning, stacklevel=3)
    return clamped, note


def center_of_resistance(frame: dict, root_length_mm: float = 10.0,
                         cres_fraction: float | None = None,
                         report: dict | None = None,
                         tooth_id=None, fdi=None) -> np.ndarray:
    """C_res on the SOCKET AXIS, root_length_mm apical of the crown centroid.

    Scanner STLs contain no root, so this is a parametric stand-in for
    anatomy that was never captured, not a measurement. Root lengths worth
    exposing per tooth type: incisor 10mm, canine 13mm, premolar/molar 9mm.
    For molars the true C_res sits near the root trifurcation and a single
    long-axis offset is a coarser approximation than it is anteriorly.

    THE LATERAL COMPONENT IS THE BUG, NOT THE DEPTH. The old form was
    `centroid - u_oa * L`, and the crown centroid drifts laterally whenever
    the wand under-selects one side or cap_and_close piles vertices over the
    rim. That drift was inherited by the pivot: on a flat molar with a
    truncated selection, C_res sat 4.50mm outside a tooth of 5mm semi-axis.

    So project the crown centroid onto the axis through the rim centroid and
    keep only its AXIAL height:

        h     = (centroid - rim_centroid) . u_oa      (a scalar)
        C_res = rim_centroid + u_oa * (h - root_length_mm)

    Algebraically this is the old formula minus exactly and only the lateral
    component -- the noise term, and nothing else. The pivot is therefore on
    the socket axis by construction (lateral offset 0.00mm no matter how
    lopsided the selection), while the depth stays exactly what every existing
    caller and test already expects.

    `cres_fraction` switches to the textbook anchor instead, measuring from
    the cervical margin rather than the crown centroid:

        C_res = rim_centroid - u_oa * (cres_fraction * root_length_mm)

    C_res genuinely sits about 1/3 to 1/2 of root length apical to the
    alveolar crest, which the margin approximates, so ~0.4 is the defensible
    value. It is off by default because it shifts the pivot depth by a
    tooth-dependent amount (8mm on an incisor, 1.5mm on a shallow molar cut)
    -- a real clinical change that deserves its own release and its own
    verification, not a free ride on an axis fix.

    Falls back to the crown centroid for frames carrying no rim_centroid.

    THE PROJECTION IS CLAMPED to [7, 15] mm and the clamp is reported, never
    silent — pass a `report` dict and it gains a "cres_clamp" key. A 2mm pivot
    shift nobody was told about is the kind of error that is only visible
    months later as a tooth that tipped when it should have translated.
    """
    root_length_mm, _clamp_note = clamp_cres_projection(root_length_mm, tooth_id, fdi)
    if report is not None and _clamp_note is not None:
        report["cres_clamp"] = _clamp_note

    u_oa = np.asarray(frame["u_oa"], float)
    rim_centroid = frame.get("rim_centroid") if hasattr(frame, "get") else None
    if rim_centroid is None:
        return np.asarray(frame["centroid"], float) - u_oa * root_length_mm

    rim_centroid = np.asarray(rim_centroid, float)
    if cres_fraction is not None:
        return rim_centroid - u_oa * (cres_fraction * root_length_mm)

    h = float((np.asarray(frame["centroid"], float) - rim_centroid) @ u_oa)
    return rim_centroid + u_oa * (h - root_length_mm)


# =========================================================================
# Module 4: kinematics
# =========================================================================

def rotation_matrix_axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis."""
    theta = np.radians(degrees)
    if abs(theta) < 1e-12:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def kinematic_matrix(
    frame: dict,
    c_res: np.ndarray,
    tip_deg: float = 0.0,
    torque_deg: float = 0.0,
    rotation_deg: float = 0.0,
    d_md: float = 0.0,
    d_bl: float = 0.0,
    d_oa: float = 0.0,
) -> np.ndarray:
    """
    M = T(C_res + t_local) . R_torque . R_tip . R_rot . T(-C_res)

    Clinical axis assignment — note this corrects the swap present in the
    earlier build, where Tip was rotating about the mesiodistal axis:
        Tip      (mesiodistal angulation) rotates about u_BL
        Torque   (buccolingual inclination) rotates about u_MD
        Rotation (axial)                    rotates about u_OA

    Translations are expressed in the local anatomical frame, so "+1mm
    mesial" means the same thing regardless of how the scan was oriented
    when exported.
    """
    R = (rotation_matrix_axis_angle(frame["u_md"], torque_deg)
         @ rotation_matrix_axis_angle(frame["u_bl"], tip_deg)
         @ rotation_matrix_axis_angle(frame["u_oa"], rotation_deg))

    t_local = d_md * frame["u_md"] + d_bl * frame["u_bl"] + d_oa * frame["u_oa"]

    T_neg = np.eye(4); T_neg[:3, 3] = -c_res
    R4 = np.eye(4);    R4[:3, :3] = R
    T_pos = np.eye(4); T_pos[:3, 3] = c_res + t_local
    return T_pos @ R4 @ T_neg


def apply_matrix(verts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Bake a 4x4 into vertex coordinates. Baking (rather than holding a
    separate object transform) is what prevents the geometry/transform
    desync that corrupted the earlier WebGL build."""
    h = np.column_stack([verts, np.ones(len(verts))])
    return (h @ M.T)[:, :3]


# =========================================================================
# Module 7: staging (arithmetic only — no clinical verdicts)
# =========================================================================

def staging_estimate(
    tip_deg: float, torque_deg: float, rotation_deg: float,
    d_md: float, d_bl: float, d_oa: float,
    max_translation_per_stage: float = 0.25,
    max_rotation_per_stage: float = 2.0,
) -> dict:
    """Divide requested movement by per-stage limits. Returns the raw
    numbers and which component drives the count. Deliberately returns no
    warning strings, colours, or recommendations — the clinician reads the
    numbers and decides. Both limits are caller-supplied defaults, not
    constants baked into the engine."""
    translation = float(np.linalg.norm([d_md, d_bl, d_oa]))
    max_rot = max(abs(tip_deg), abs(torque_deg), abs(rotation_deg))
    n_trans = int(np.ceil(translation / max_translation_per_stage)) if translation > 0 else 0
    n_rot = int(np.ceil(max_rot / max_rotation_per_stage)) if max_rot > 0 else 0
    return dict(
        total_translation_mm=translation,
        max_rotation_deg=max_rot,
        stages_from_translation=n_trans,
        stages_from_rotation=n_rot,
        stages_required=max(n_trans, n_rot),
        driver="translation" if n_trans >= n_rot else "rotation",
    )


# =========================================================================
# Vectorized hot paths
# -------------------------------------------------------------------------
# The reference implementations above are readable and are what the test
# suite pins correctness against. They are also per-vertex Python loops,
# which measured ~8.5s for one segmentation on a 100k-vertex mesh and scale
# linearly — 30-45s on a 300-500k-vertex intraoral scan, with the GUI frozen
# throughout. The functions below are numerically equivalent (verified in
# test_core_geometry.py) and run on NumPy/SciPy primitives instead.
# =========================================================================

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra


def directed_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    UNIQUE directed neighbour pairs (i -> j), both directions.

    The deduplication is load-bearing, not tidiness. A raw half-edge list
    repeats every interior edge (each is shared by two faces) while listing
    boundary edges once, so any per-vertex average built on it silently
    double-weights the mesh interior relative to the boundary. The reference
    implementation iterates a set of neighbours and does not have this bias;
    testing the two against each other is what exposed it.

    Uniqueness is done by packing each pair into a single int64 key rather
    than np.unique(..., axis=0). The 2D form lexsorts a (2.4M, 2) array and
    measured 1.43s on a 100k-vertex mesh; since three separate call sites
    needed the edge list, that one line accounted for most of the runtime of
    the whole segmentation. Packing to 1D makes it a plain sort.
    """
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    both = np.vstack([e, e[:, ::-1]]).astype(np.int64)
    n = int(faces.max()) + 1
    key = np.unique(both[:, 0] * n + both[:, 1])
    return (key // n).astype(np.int64), (key % n).astype(np.int64)


def _concavity_raw(verts: np.ndarray, faces: np.ndarray, edges=None) -> np.ndarray:
    """Unnormalized concavity. Split out so the scaling choice is explicit."""
    vn = vertex_normals(verts, faces)
    i, j = edges if edges is not None else directed_edges(faces)

    d = verts[j] - verts[i]
    ln = np.linalg.norm(d, axis=1)
    ok = ln > 1e-12
    i, j, d, ln = i[ok], j[ok], d[ok], ln[ok]
    unit = d / ln[:, None]

    contrib = np.einsum('ij,ij->i', unit, vn[i])
    total = np.zeros(len(verts))
    count = np.zeros(len(verts))
    np.add.at(total, i, contrib)
    np.add.at(count, i, 1.0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def vertex_concavity_fast(verts: np.ndarray, faces: np.ndarray, edges=None) -> np.ndarray:
    """Vectorized equivalent of vertex_concavity(), scaled by global maximum.

    Kept for reference-equivalence testing. For anything that has to work on
    real scan data, prefer vertex_concavity_robust -- see its docstring for
    why the global maximum is the wrong scale.
    """
    conc = _concavity_raw(verts, faces, edges)
    m = np.max(np.abs(conc))
    return conc / m if m > 1e-12 else conc


def vertex_concavity_robust(verts: np.ndarray, faces: np.ndarray, edges=None,
                            percentile: float = 99.0, clip: float = 3.0) -> np.ndarray:
    """
    Concavity scaled by a high PERCENTILE instead of the maximum.

    Measured failure this fixes (test_bleed.py): dividing by the global
    maximum makes one outlier vertex set the scale for the whole mesh. On a
    dense noisy scan the maximum is a noise spike, so genuine anatomy gets
    compressed toward zero -- the cervical sulcus signal fell from +0.322 to
    +0.064 across clean-to-noisy at the same mesh density, a 5x weaker
    barrier, while |conc|max stayed pinned at exactly 1.000 in every case.
    A weak barrier is precisely what lets a region grow leak through the
    gumline and across interproximal contacts.

    The 99th percentile is set by real anatomy rather than by the single
    worst vertex, and clipping afterwards stops the remaining 1% of extreme
    values from dominating the exponential in build_barrier_graph.
    """
    conc = _concavity_raw(verts, faces, edges)
    scale = np.percentile(np.abs(conc), percentile)
    if scale < 1e-12:
        return conc
    return np.clip(conc / scale, -clip, clip)


def adaptive_smoothing_iterations(verts: np.ndarray, faces: np.ndarray, edges=None,
                                  target_radius_mm: float = 0.35) -> int:
    """
    Smoothing iterations needed to average over a fixed PHYSICAL radius.

    A fixed count is wrong across mesh densities: two iterations on a
    0.52mm-edge mesh smooths over roughly 0.7mm of tissue, but on a
    0.20mm-edge scan it covers only 0.28mm, leaving the curvature field
    dominated by per-vertex scanner noise exactly where the barrier needs to
    be reliable. Laplacian diffusion spreads about sqrt(n) * edge_length, so
    n scales with the square of the ratio.
    """
    i, j = edges if edges is not None else directed_edges(faces)
    mean_edge = float(np.linalg.norm(verts[j] - verts[i], axis=1).mean())
    if mean_edge < 1e-9:
        return 2
    return int(np.clip(round((target_radius_mm / mean_edge) ** 2), 1, 12))


def smooth_scalar_fast(values: np.ndarray, faces: np.ndarray, iterations: int = 2, edges=None) -> np.ndarray:
    """Vectorized equivalent of smooth_scalar()."""
    i, j = edges if edges is not None else directed_edges(faces)
    count = np.zeros(len(values))
    np.add.at(count, i, 1.0)
    safe = count > 0

    out = values.copy()
    for _ in range(iterations):
        acc = np.zeros(len(values))
        np.add.at(acc, i, out[j])
        mean = np.divide(acc, count, out=out.copy(), where=safe)
        out = 0.5 * out + 0.5 * mean
    return out


def build_edge_graph(
    verts: np.ndarray,
    faces: np.ndarray,
    concavity: np.ndarray,
    concavity_weight: float = 8.0,
    edges=None,
) -> csr_matrix:
    """
    Sparse weighted adjacency for the magnetic scissors, built ONCE and
    reused across every leg of the loop. The reference implementation
    rebuilt a Python adjacency list per Dijkstra call, which was pure waste:
    three anchors means three legs means three rebuilds of the same graph.
    """
    penalty = 1.0 + concavity_weight * (1.0 - concavity)
    i, j = edges if edges is not None else directed_edges(faces)
    length = np.linalg.norm(verts[j] - verts[i], axis=1)
    w = length * 0.5 * (penalty[i] + penalty[j])
    w = np.maximum(w, 1e-12)  # csgraph treats 0 as "no edge"
    return csr_matrix((w, (i, j)), shape=(len(verts), len(verts)))


def path_via_graph(graph: csr_matrix, start: int, goal: int) -> list[int]:
    """Single-source Dijkstra in compiled SciPy, then walk predecessors."""
    _, pred = _sp_dijkstra(graph, indices=start, return_predecessors=True)
    if pred[goal] == -9999 and goal != start:
        return []
    path, cur = [], goal
    while cur != start and cur != -9999:
        path.append(cur)
        cur = pred[cur]
    if cur != start:
        return []
    path.append(start)
    return path[::-1]


def closed_loop_through_anchors_fast(
    verts: np.ndarray,
    faces: np.ndarray,
    concavity: np.ndarray,
    anchor_points: np.ndarray,
    concavity_weight: float = 8.0,
    edges=None,
) -> list[int]:
    """Vectorized equivalent of closed_loop_through_anchors()."""
    graph = build_edge_graph(verts, faces, concavity, concavity_weight, edges=edges)
    anchors = [nearest_vertex(verts, p) for p in anchor_points]

    loop: list[int] = []
    for k in range(len(anchors)):
        seg = path_via_graph(graph, anchors[k], anchors[(k + 1) % len(anchors)])
        if not seg:
            return []
        loop.extend(seg[:-1])

    dedup: list[int] = []
    for v in loop:
        if not dedup or dedup[-1] != v:
            dedup.append(v)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup


def segmentation_is_plausible(
    n_total_faces: int,
    n_crown_faces: int,
    n_base_faces: int,
    loop: list[int],
    max_crown_fraction: float = 0.35,
    min_crown_faces: int = 50,
) -> tuple[bool, str]:
    """
    Guard against a cut loop that failed to encircle the tooth.

    Failure mode this exists to catch: if the anchor path doubles back on
    itself instead of forming a circuit, the loop separates nothing. The
    flood fill in split_mesh_by_loop then spreads across the whole mesh and
    returns the ENTIRE ARCH as the "crown" -- which renders as the whole cast
    turning blue and every kinematics slider moving the full model.

    A single crown is a small fraction of a full arch (typically 3-10% of
    faces). Anything above a third means the loop did not close, and the
    caller should refuse the result rather than hand back a bad segmentation
    that looks like a working one.

    Returns (ok, human-readable reason).
    """
    if n_crown_faces == 0 or n_base_faces == 0:
        return False, "The cut produced an empty region — the loop did not divide the mesh."

    frac = n_crown_faces / max(n_total_faces, 1)
    if frac > max_crown_fraction:
        return False, (
            f"The margin loop did not close around the tooth: the selected region is "
            f"{frac:.0%} of the whole arch. This happens when the clicks trace a path "
            f"that doubles back instead of encircling the crown — most often when the "
            f"lingual side is missing, so both routes between the contacts run buccally. "
            f"Re-click going all the way around the tooth."
        )
    if n_crown_faces < min_crown_faces:
        return False, (
            f"The selected region is only {n_crown_faces} triangles — too small to be a "
            f"crown. The clicks are probably too close together."
        )
    if len(loop) != len(set(loop)):
        repeats = len(loop) - len(set(loop))
        return False, (
            f"The margin path crosses itself ({repeats} repeated vertices), so it does not "
            f"form a clean boundary. Spread the clicks more evenly around the tooth."
        )
    return True, "ok"


def connected_components(faces: np.ndarray, loop: list[int]) -> list[np.ndarray]:
    """Face-connected components after cutting along `loop`, largest first."""
    loop_edges = set()
    L = len(loop)
    for k in range(L):
        u, v = loop[k], loop[(k + 1) % L]
        loop_edges.add((u, v) if u < v else (v, u))

    inc = edge_face_incidence(faces)
    adj: list[list[int]] = [[] for _ in range(len(faces))]
    for edge, fl in inc.items():
        if len(fl) == 2 and edge not in loop_edges:
            adj[fl[0]].append(fl[1])
            adj[fl[1]].append(fl[0])

    seen = np.zeros(len(faces), dtype=bool)
    comps = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        stack, members = [start], []
        seen[start] = True
        while stack:
            f = stack.pop()
            members.append(f)
            for g in adj[f]:
                if not seen[g]:
                    seen[g] = True
                    stack.append(g)
        comps.append(np.array(members))
    comps.sort(key=len, reverse=True)
    return comps


def split_mesh_by_loop_auto(
    verts: np.ndarray, faces: np.ndarray, loop: list[int],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict]:
    """
    Split along `loop` and return (crown, base, info), where the crown is the
    SMALLEST connected component.

    Why not seed the flood fill from the FA click, as split_mesh_by_loop
    does: once the FA point became one of the margin anchors, it sits ON the
    cut boundary. The nearest face to a boundary point can lie on either
    side, so seeding from it is a coin flip. On any real arch the crown is
    unambiguously the smaller piece -- a single tooth is a few percent of the
    model -- so size is a far more reliable discriminator than a seed that
    may be sitting exactly on the fence.
    """
    comps = connected_components(faces, loop)

    def extract(idx):
        sub = faces[idx]
        used = np.unique(sub)
        remap = -np.ones(len(verts), dtype=int)
        remap[used] = np.arange(len(used))
        return verts[used].copy(), remap[sub]

    info = dict(n_components=len(comps), sizes=[int(len(c)) for c in comps[:5]])
    if len(comps) < 2:
        return (np.empty((0, 3)), np.empty((0, 3), dtype=int)), extract(comps[0]), info

    crown_idx = comps[-1]
    base_idx = np.concatenate(comps[:-1])
    return extract(crown_idx), extract(base_idx), info


def derive_anatomical_frame_4click(
    mesial_pt: np.ndarray,
    buccal_pt: np.ndarray,
    distal_pt: np.ndarray,
    lingual_pt: np.ndarray,
    crown_verts: np.ndarray,
    rim_verts: np.ndarray,
) -> dict:
    """
    Orthonormal frame from four margin landmarks clicked in circuit order.

    Improvement over the 3-click version: there, the buccolingual axis came
    from (FA point - crown centroid). Once the FA click moved down to the
    margin to serve as a loop anchor, that vector points outward AND
    downward, tilting u_BL and therefore u_OA -- so C_res landed off-axis
    rather than straight apical.

    Here both axes are measured directly between opposing landmarks at the
    same anatomical level:
        u_MD  from mesial contact -> distal contact
        u_BL  from lingual click  -> buccal click   (outward = buccal)
    Gram-Schmidt still applies, because two clicked directions on a real
    tooth will not be exactly perpendicular.
    """
    u_md = distal_pt - mesial_pt
    n = np.linalg.norm(u_md)
    if n < 1e-9:
        raise ValueError("Mesial and distal clicks coincide.")
    u_md = u_md / n

    u_bl = buccal_pt - lingual_pt
    if np.linalg.norm(u_bl) < 1e-9:
        raise ValueError("Buccal and lingual clicks coincide.")
    u_bl = u_bl - np.dot(u_bl, u_md) * u_md          # Gram-Schmidt
    n = np.linalg.norm(u_bl)
    if n < 1e-9:
        raise ValueError("Buccal-lingual direction is parallel to mesiodistal — re-click.")
    u_bl = u_bl / n

    u_oa = np.cross(u_md, u_bl)
    u_oa = u_oa / np.linalg.norm(u_oa)

    crown_centroid = crown_verts.mean(axis=0)
    rim_centroid = rim_verts.mean(axis=0)
    if np.dot(u_oa, crown_centroid - rim_centroid) < 0:
        u_oa = -u_oa
        u_md = -u_md                                  # keep the basis right-handed

    return dict(centroid=crown_centroid, u_md=u_md, u_bl=u_bl, u_oa=u_oa,
                rim_centroid=rim_centroid)


def build_edge_graph_blocked(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray,
    concavity_weight: float = 8.0, edges=None, blocked: np.ndarray | None = None,
) -> csr_matrix:
    """Edge graph with `blocked` vertices removed (their edges dropped)."""
    penalty = 1.0 + concavity_weight * (1.0 - concavity)
    i, j = edges if edges is not None else directed_edges(faces)
    if blocked is not None:
        keep = ~(blocked[i] | blocked[j])
        i, j = i[keep], j[keep]
    length = np.linalg.norm(verts[j] - verts[i], axis=1)
    w = np.maximum(length * 0.5 * (penalty[i] + penalty[j]), 1e-12)
    return csr_matrix((w, (i, j)), shape=(len(verts), len(verts)))


def two_path_loop(
    verts: np.ndarray,
    faces: np.ndarray,
    concavity: np.ndarray,
    anchor_points: np.ndarray,
    concavity_weight: float = 8.0,
    edges=None,
    block_radius: float = 0.8,
) -> tuple[list[int], list[int], list[int]]:
    """
    Build a closed margin loop from THREE anchors that are all on the SAME
    visible face of the tooth (mesial contact, buccal midpoint, distal
    contact), by finding two vertex-disjoint paths between the contacts.

    Why this exists: a loop that encircles a crown must cross both the buccal
    and the lingual side, but those are never visible at the same time. A
    surface picker returns the frontmost face, so a "lingual" click taken
    from a buccal viewpoint silently lands on the buccal surface -- and four
    coplanar clicks enclose a patch of the labial face instead of the tooth.
    Asking the clinician to rotate mid-selection is possible but fragile.

    Instead:
      1. Trace path A: mesial -> buccal -> distal   (all clicks visible)
      2. Block every vertex within `block_radius` mm of path A
      3. Trace path B: distal -> mesial, which is now FORCED around the
         lingual side because the buccal route is unavailable
      4. The two paths together form the closed circuit

    The blocked *radius* (not just the exact path vertices) stops path B from
    running parallel to path A one row of triangles away, which would enclose
    a sliver instead of the tooth.

    Returns (loop, path_a, path_b) so the caller can use path B's midpoint as
    a genuine lingual landmark for the anatomical frame.
    """
    if edges is None:
        edges = directed_edges(faces)

    idx = [nearest_vertex(verts, p) for p in anchor_points]
    if len(idx) != 3:
        raise ValueError("two_path_loop expects exactly 3 anchors (mesial, buccal, distal).")
    mesial_i, buccal_i, distal_i = idx

    graph = build_edge_graph(verts, faces, concavity, concavity_weight, edges=edges)
    leg1 = path_via_graph(graph, mesial_i, buccal_i)
    leg2 = path_via_graph(graph, buccal_i, distal_i)
    if not leg1 or not leg2:
        return [], [], []
    path_a = leg1[:-1] + leg2

    from scipy.spatial import cKDTree
    tree = cKDTree(verts[path_a])
    tree_free = cKDTree(verts)
    blocked = np.zeros(len(verts), dtype=bool)
    near = tree.query_ball_point(verts, r=block_radius)
    for vi, hits in enumerate(near):
        if hits:
            blocked[vi] = True
    # Unblock a NEIGHBOURHOOD around each endpoint, not just the endpoint
    # vertex itself: path A passes through both contacts, so the radius block
    # also covers everything adjacent to them. Clearing only the two vertices
    # leaves them with no usable edges and path B can never start.
    for endpoint in (mesial_i, distal_i):
        for vi in tree_free.query_ball_point(verts[endpoint], r=block_radius * 1.6):
            blocked[vi] = False

    graph_b = build_edge_graph_blocked(
        verts, faces, concavity, concavity_weight, edges=edges, blocked=blocked)
    path_b = path_via_graph(graph_b, distal_i, mesial_i)
    if not path_b:
        return [], path_a, []

    loop = path_a[:-1] + path_b[:-1]
    dedup: list[int] = []
    for v in loop:
        if not dedup or dedup[-1] != v:
            dedup.append(v)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    return dedup, path_a, path_b


def crown_dimensions(crown_verts: np.ndarray, frame: dict) -> dict:
    """
    Crown extents along its own anatomical axes, in mm.

    Reported so a wrong selection is obvious at a glance. A real anterior
    crown runs roughly 8-11mm occlusoapically and 6-9mm mesiodistally. A
    patch of the labial surface reads only 1-3mm tall -- the number tells the
    clinician immediately that the cut grabbed a surface fragment, which mesh
    size alone cannot reveal (a small patch is a perfectly plausible
    percentage of the arch).
    """
    c = crown_verts - crown_verts.mean(axis=0)
    return dict(
        height_oa=float(np.ptp(c @ frame["u_oa"])),
        width_md=float(np.ptp(c @ frame["u_md"])),
        depth_bl=float(np.ptp(c @ frame["u_bl"])),
    )


def derive_frame_from_margin(
    margin_verts: np.ndarray,
    mesial_pt: np.ndarray,
    buccal_pt: np.ndarray,
    crown_verts: np.ndarray,
    distal_pt: np.ndarray,
) -> dict:
    """
    Anatomical frame derived from the FULL margin loop rather than point
    samples.

    Progression of this function across the build, because the reasoning
    matters more than the final formula:
      v1  u_BL from (FA point - crown centroid). Skewed: not perpendicular
          to u_MD, so rotations sheared. Fixed with Gram-Schmidt.
      v2  u_BL from buccal click - lingual click. Better, but once the
          clicks sat at the margin the pair still tilted the long axis 38.7deg.
      v3  (here) u_OA = normal of the best-fit plane through the whole
          cervical margin loop.

    The cervical margin IS an anatomical ring around the tooth neck, close to
    perpendicular to the long axis. Fitting a plane to all several-hundred
    loop vertices averages out click imprecision and local scan noise, where
    any two- or three-point sample inherits it directly. On the symmetric
    fixture this drops long-axis tilt from 5.5deg to under 1deg.

    u_MD is then the clicked contact-to-contact direction projected into that
    plane, and u_BL closes the right-handed frame, signed toward the buccal
    click.
    """
    rim_centroid = margin_verts.mean(axis=0)
    _, _, vt = np.linalg.svd(margin_verts - rim_centroid, full_matrices=False)
    u_oa = vt[2] / np.linalg.norm(vt[2])          # least-variance dir = plane normal

    crown_centroid = crown_verts.mean(axis=0)
    if np.dot(u_oa, crown_centroid - rim_centroid) < 0:
        u_oa = -u_oa                              # occlusal = toward the crown body

    u_md = distal_pt - mesial_pt
    u_md = u_md - np.dot(u_md, u_oa) * u_oa       # project into the margin plane
    n = np.linalg.norm(u_md)
    if n < 1e-9:
        raise ValueError("Mesial/distal clicks project to nothing in the margin plane.")
    u_md = u_md / n

    u_bl = np.cross(u_oa, u_md)
    u_bl = u_bl / np.linalg.norm(u_bl)
    if np.dot(u_bl, buccal_pt - rim_centroid) < 0:
        u_bl = -u_bl
        u_md = -u_md                              # keep the basis right-handed

    return dict(centroid=crown_centroid, u_md=u_md, u_bl=u_bl, u_oa=u_oa,
                rim_centroid=rim_centroid)


# =========================================================================
# Magic-wand selection (replaces the 4-click margin circuit)
# -------------------------------------------------------------------------
# Why the click-the-margin approach had to go: a crown's cervical margin
# wraps 360 degrees around the tooth, but a camera only ever sees one side.
# vtkCellPicker returns the frontmost visible cell, so a "lingual" click made
# while looking at the labial surface silently lands on the labial surface --
# and the resulting loop encloses a patch of the front face instead of the
# crown. No amount of clicking discipline fixes that; the points are simply
# not simultaneously visible.
#
# Region growing has no such constraint. It spreads across mesh connectivity,
# so it wraps around to surfaces the camera cannot see, and it stops where
# the anatomy says to stop.
# =========================================================================

def region_grow_crown(
    verts: np.ndarray,
    faces: np.ndarray,
    concavity: np.ndarray,
    seed_point: np.ndarray,
    tolerance: float,
    edges=None,
    barrier_weight: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Photoshop-style flood selection bounded by the cervical sulcus.

    Dijkstra from the seed where crossing CONCAVE terrain is expensive:

        cost = length * (1 + barrier_weight * max(0, concavity))

    Note this inverts the magnetic-scissors weighting. The scissors wanted to
    travel ALONG the sulcus, so concavity was cheap there. Here we want the
    flood to stop AT the sulcus, so concavity is expensive: the growth runs
    freely over the smooth convex crown and hits a wall of cost at the neck.

    `tolerance` is the distance budget in mesh units (mm), and behaves like
    Photoshop's tolerance slider: raise it to take more, lower it to take
    less. Because crossing the sulcus costs roughly 25x its geometric length,
    there is a wide plateau of tolerance values that all select exactly the
    crown -- which is what makes the slider forgiving rather than fiddly.

    Returns (face_mask, vertex_distances).
    """
    i, j = edges if edges is not None else directed_edges(faces)
    penalty = np.exp(barrier_weight * np.maximum(concavity, 0.0))
    length = np.linalg.norm(verts[j] - verts[i], axis=1)
    w = np.maximum(length * 0.5 * (penalty[i] + penalty[j]), 1e-12)
    graph = csr_matrix((w, (i, j)), shape=(len(verts), len(verts)))

    seed = nearest_vertex(verts, seed_point)
    dist = _sp_dijkstra(graph, indices=seed)

    in_region = dist <= tolerance
    face_mask = in_region[faces].all(axis=1)
    return face_mask, dist


def split_by_face_mask(
    verts: np.ndarray, faces: np.ndarray, face_mask: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Split into (crown, base) using a face mask rather than a boundary loop.

    Structurally safer than loop-based splitting: a mask always partitions the
    faces, so there is no way to produce the "loop failed to close, flood fill
    swallowed the whole arch" outcome. Both halves are re-indexed
    independently, so each owns its own open boundary for capping.
    """
    def extract(mask):
        sub = faces[mask]
        if len(sub) == 0:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)
        used = np.unique(sub)
        remap = -np.ones(len(verts), dtype=int)
        remap[used] = np.arange(len(used))
        return verts[used].copy(), remap[sub]

    return extract(face_mask), extract(~face_mask)


def largest_face_component(faces: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected island of a selection.

    Region growing can occasionally jump an interproximal contact where two
    crowns touch and no concavity separates them, leaving a stray island on
    the neighbouring tooth. Keeping the largest island discards that without
    disturbing the main selection.
    """
    idx = np.where(face_mask)[0]
    if len(idx) == 0:
        return face_mask
    sub = faces[idx]
    inc: dict[tuple[int, int], list[int]] = {}
    for local, (a, b, c) in enumerate(sub):
        for u, v in ((a, b), (b, c), (c, a)):
            inc.setdefault((u, v) if u < v else (v, u), []).append(local)

    adj: list[list[int]] = [[] for _ in range(len(sub))]
    for fl in inc.values():
        if len(fl) == 2:
            adj[fl[0]].append(fl[1])
            adj[fl[1]].append(fl[0])

    seen = np.zeros(len(sub), dtype=bool)
    best: list[int] = []
    for s in range(len(sub)):
        if seen[s]:
            continue
        stack, members = [s], []
        seen[s] = True
        while stack:
            f = stack.pop()
            members.append(f)
            for g in adj[f]:
                if not seen[g]:
                    seen[g] = True
                    stack.append(g)
        if len(members) > len(best):
            best = members

    out = np.zeros(len(faces), dtype=bool)
    out[idx[np.array(best)]] = True
    return out


def buccal_direction(rim_centroid: np.ndarray, arch_frame: dict | None):
    """Which way is buccal at this tooth, or None if the arch cannot say.

    Measured from the arch frame's origin -- the posterior midpoint, which
    sits INSIDE the horseshoe -- out to the tooth, projected into the occlusal
    plane. For a molar that vector runs transversely toward its own side; for
    an incisor it runs anteriorly, which is labial. Only the hemisphere
    matters, so the approximation is safe.
    """
    if arch_frame is None or "origin" not in arch_frame:
        return None
    u_occ = np.asarray(arch_frame["u_occ"], float)
    u_occ = u_occ / np.linalg.norm(u_occ)
    out = np.asarray(rim_centroid, float) - np.asarray(arch_frame["origin"], float)
    out = out - (out @ u_occ) * u_occ
    n = np.linalg.norm(out)
    return None if n < 1e-6 else out / n


def derive_frame_from_region(
    mesial_pt: np.ndarray,
    distal_pt: np.ndarray,
    crown_verts: np.ndarray,
    rim_verts: np.ndarray,
    arch_frame: dict | None = None,
    max_axis_deviation_deg: float = MAX_AXIS_DEVIATION_DEG,
    long_axis_mode: str | None = None,
) -> dict:
    """
    Anatomical frame needing only TWO clicks, both on the visible labial side.

    The occlusoapical axis comes from the geometry -- see resolve_long_axis
    for why it is the cervical rim's plane normal and not, as it was, the
    crown centroid minus the rim centroid. Only the mesiodistal direction
    needs the clinician, and both interproximal contacts are visible from a
    single labial view, so nothing has to be clicked on a hidden surface.

    Pass `arch_frame` (from arch_frame.fit_occlusal_frame) and the axis is
    signed and reconciled against the occlusal plane, so a noise-dominated
    molar axis can never send C_res outside the tooth. Omit it and the
    function behaves as it always did -- kept for app_ui.py and the older
    tests, which have no occlusal plane to offer.

    Gram-Schmidt keeps the frame orthonormal: the clicked mesiodistal line
    and the long axis will not be exactly perpendicular on a real tooth.
    """
    crown_verts = np.asarray(crown_verts, float)
    rim_verts = np.asarray(rim_verts, float)
    crown_centroid = crown_verts.mean(axis=0)
    # An empty rim means "the cut left no open margin". mean() of nothing is
    # NaN with a RuntimeWarning, and NaN would propagate silently through the
    # whole frame; fall back to the crown centroid, which resolve_long_axis
    # then reports as an arch_apical fallback.
    rim_centroid = rim_verts.mean(axis=0) if len(rim_verts) else crown_centroid.copy()

    u_oa, axis_info = resolve_long_axis(
        rim_verts, crown_centroid, arch_frame, max_axis_deviation_deg)

    u_md = np.asarray(distal_pt, float) - np.asarray(mesial_pt, float)
    if np.linalg.norm(u_md) < 1e-9:
        raise ValueError("Mesial and distal clicks coincide.")
    u_md = u_md - np.dot(u_md, u_oa) * u_oa          # Gram-Schmidt against the long axis
    n = np.linalg.norm(u_md)
    if n < 1e-9:
        raise ValueError("Mesiodistal line is parallel to the long axis — re-click the contacts.")
    u_md = u_md / n

    u_bl = np.cross(u_oa, u_md)
    u_bl = u_bl / np.linalg.norm(u_bl)

    if (long_axis_mode or LONG_AXIS_MODE) == "cross_product":
        # The textbook triad: u_OA = u_MD x u_BL. u_BL is taken from the arch
        # frame's buccal direction rather than the rim, so the tooth's own
        # cervical anatomy does not enter the long axis at all. Re-orthogonalise
        # in the same order so the basis stays right-handed and unit.
        # buccal_direction() gives the PER-TOOTH outward radial direction at this
        # rim. The arch-level u_tra is NOT a substitute and using it was measured
        # to put the long axis 90 degrees out: buccal for a molar is roughly
        # +/-u_tra, for an incisor roughly u_sag, and no single arch vector is
        # buccal for every tooth. Buccal is radial, and radial is per-tooth.
        u_bl_ref = buccal_direction(rim_centroid, arch_frame)
        if u_bl_ref is not None:
            u_bl_ref = np.asarray(u_bl_ref, float)
            nb = np.linalg.norm(u_bl_ref)
            if nb > 1e-9:
                u_bl_ref = u_bl_ref / nb
                u_bl_ref = u_bl_ref - np.dot(u_bl_ref, u_md) * u_md
                nb = np.linalg.norm(u_bl_ref)
                if nb > 1e-9:
                    u_bl = u_bl_ref / nb
                    alt = np.cross(u_md, u_bl)
                    na = np.linalg.norm(alt)
                    if na > 1e-9:
                        alt = alt / na
                        # Keep the occlusal sign the rim established; flipping it
                        # would put C_res above the crown.
                        if np.dot(alt, u_oa) < 0:
                            alt = -alt
                            u_bl = -u_bl
                        axis_info = dict(axis_info)
                        axis_info["axis_mode"] = "cross_product"
                        axis_info["axis_shift_deg"] = round(float(np.degrees(
                            np.arccos(np.clip(np.dot(alt, u_oa), -1.0, 1.0)))), 3)
                        u_oa = alt
    else:
        axis_info = dict(axis_info)
        axis_info["axis_mode"] = "rim_plane"
        axis_info["axis_shift_deg"] = 0.0

    # PIN u_BL TO ANATOMY, NOT TO CLICK ORDER.
    #
    # With only two clicks there is no buccal landmark, so u_BL's sign came
    # entirely from which contact was clicked first: click distal-then-mesial
    # and u_MD flips, u_BL flips with it, and positive torque silently becomes
    # LINGUAL crown torque instead of buccal. A sign-inverted prescription that
    # animates perfectly smoothly is the worst kind of bug.
    #
    # The arch frame can settle it. Flip u_MD alongside u_BL to keep the basis
    # right-handed -- the same two-axis flip derive_frame_from_margin does at
    # its buccal check, and derive_anatomical_frame_4click at its occlusal one.
    #
    # This does NOT make every axis unambiguous, and no frame could: the arch
    # is mirror-symmetric, so (occlusal, buccal, mesial->distal) cannot all be
    # right-handed on both sides. Pinning u_BL moves the residual ambiguity to
    # u_MD, where it depends on the QUADRANT (knowable, and what FDI sign
    # conventions already encode) rather than on the click order (arbitrary).
    # u_md_points_distal reports which way it landed.
    buccal = buccal_direction(rim_centroid, arch_frame)
    if buccal is not None and u_bl @ buccal < 0:
        u_bl, u_md = -u_bl, -u_md

    # u_MD x u_BL = u_MD x (u_OA x u_MD) = u_OA, so this basis is right-handed
    # by construction. Verified anyway: it costs a 3x3 determinant, it survives
    # future edits to the two-axis flip above, and unlike the orthogonality
    # checks it also catches NaN (every comparison against NaN is False, so
    # `not (det > 0.999)` fires). THREE.Matrix4.makeBasis turns a reflection
    # into a garbage quaternion silently -- a failure visually identical to the
    # bug this function exists to fix.
    det = float(np.linalg.det(np.column_stack([u_md, u_bl, u_oa])))
    if not (det > 0.999):
        raise ValueError(
            f"Derived frame is not a right-handed orthonormal basis (det={det:.6f}). "
            f"Refusing to return it: the gizmo would render a sheared tooth.")

    frame = dict(centroid=crown_centroid, u_md=u_md, u_bl=u_bl, u_oa=u_oa,
                 rim_centroid=rim_centroid, basis_det=round(det, 9))
    frame.update(axis_info)
    frame["u_bl_points_buccal"] = None if buccal is None else True
    frame["u_md_points_distal"] = (
        None if buccal is None
        else bool(u_md @ (np.asarray(distal_pt, float) - np.asarray(mesial_pt, float)) > 0))
    return frame


def build_barrier_graph(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray,
    barrier_weight: float = 25.0, edges=None, mode: str = "exponential",
    steepness: float = 4.0,
) -> csr_matrix:
    """
    Barrier-weighted graph for region growing, built ONCE per arch.

    mode="exponential" (default):  cost = length * exp(steepness * max(0, c))
    mode="linear" (previous):      cost = length * (1 + barrier_weight * max(0, c))

    Why exponential. The linear form makes the sulcus merely expensive: with
    weight 25 and a robust concavity of 1.0 the barrier costs 26x its
    geometric length, so a flood only has to find a slightly weak spot --
    a shallow patch of margin, a scanner artifact -- to leak through and
    swallow the gingiva. The exponential form makes it a wall: at steepness
    4 the same crossing costs e^4 = 55x, and a strong margin at c = 2 costs
    e^8 = 2981x. Weak spots stop being viable routes rather than merely
    costly ones, which is the behaviour a cervical margin actually has.

    Pair this with vertex_concavity_robust. Feeding max-normalized concavity
    into an exponential just exponentiates a signal that noise has already
    flattened.
    """
    i, j = edges if edges is not None else directed_edges(faces)
    valley = np.maximum(concavity, 0.0)
    if mode == "exponential":
        penalty = np.exp(steepness * valley)
    else:
        penalty = 1.0 + barrier_weight * valley
    length = np.linalg.norm(verts[j] - verts[i], axis=1)
    w = np.maximum(length * 0.5 * (penalty[i] + penalty[j]), 1e-12)
    return csr_matrix((w, (i, j)), shape=(len(verts), len(verts)))


def geodesic_from_seed(graph: csr_matrix, verts: np.ndarray, seed_point: np.ndarray) -> np.ndarray:
    """Barrier-weighted distance from the clicked point to every vertex.
    Measured at ~0.02s even on a 360k-vertex mesh."""
    return _sp_dijkstra(graph, indices=nearest_vertex(verts, seed_point))


def mask_from_distance(faces: np.ndarray, dist: np.ndarray, tolerance: float) -> np.ndarray:
    """Threshold a distance field into a face selection. Pure array op."""
    inside = dist <= tolerance
    return inside[faces].all(axis=1)


# =========================================================================
# Whole-arch auto-segmentation (visualization aid)
# -------------------------------------------------------------------------
# IMPORTANT SCOPE NOTE. This is heuristic geometry, not learned segmentation.
# It is intended to COLOUR the arch so a clinician can see tooth-from-gingiva
# at a glance, and to propose seeds. It is NOT a substitute for confirming an
# individual cut: on crowding, worn cusps, restorations or partially erupted
# teeth it will over- or under-segment. Treat its output as a visual guide
# the clinician overrides, never as a boundary to cut on unreviewed.
# =========================================================================

def find_tooth_seeds(
    verts: np.ndarray,
    occlusal_axis: np.ndarray,
    min_separation_mm: float = 5.5,
    max_seeds: int = 16,
    relative_height: float = 0.45,
) -> np.ndarray:
    """
    Cusp-tip seeds: high points along the occlusal axis, greedily
    non-maximum-suppressed so two seeds cannot land on the same crown.

    `relative_height` is a fraction of the arch's total occlusal height
    range, NOT a percentile of vertices. That distinction was a real bug:
    an arch is mostly flat gingival base, so a 60th-percentile cutoff sits
    at roughly base level and admits thousands of base vertices as
    candidates. Once the genuine cusps were taken, the seed list filled up
    with gingiva -- 16 seeds on a 5-tooth arch. Measuring against the height
    RANGE instead keeps candidates on the crowns where they belong.

    `min_separation_mm` defaults just under a typical mesiodistal crown
    width. Too small and a multi-cusped molar yields several seeds and gets
    split; too large and adjacent teeth merge. Worth exposing in the UI.
    """
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    height = verts @ axis

    lo, hi = float(height.min()), float(height.max())
    if hi - lo < 1e-9:
        return np.array([], dtype=int)
    cutoff = lo + relative_height * (hi - lo)

    candidates = np.where(height >= cutoff)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)
    candidates = candidates[np.argsort(height[candidates])[::-1]]

    seeds: list[int] = []
    taken = np.empty((0, 3))
    for idx in candidates:
        if len(seeds) >= max_seeds:
            break
        p = verts[idx]
        if len(taken) and np.min(np.linalg.norm(taken - p, axis=1)) < min_separation_mm:
            continue
        seeds.append(int(idx))
        taken = np.vstack([taken, p])
    return np.array(seeds, dtype=int)


def auto_segment_arch(
    verts: np.ndarray,
    faces: np.ndarray,
    concavity: np.ndarray,
    occlusal_axis: np.ndarray,
    edges=None,
    tolerance: float = 12.0,
    min_separation_mm: float = 5.5,
    max_seeds: int = 16,
    barrier_weight: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Watershed over the barrier-weighted mesh graph.

    One multi-source Dijkstra from every cusp seed at once gives an
    (n_seeds, n_verts) distance field. Each vertex belongs to whichever seed
    reaches it most cheaply; anything no seed reaches within `tolerance`
    is gingiva. Because the metric already charges heavily for crossing
    concave terrain, the basins naturally stop at the cervical sulcus, and
    the ridge between two adjacent crowns falls where their two floods meet.

    Returns (face_labels, seed_vertex_indices). Label 0 is gingiva/base;
    teeth are 1..n. Faces spanning two basins are assigned to the label
    holding the majority of their corners, which keeps the boundary crisp
    rather than leaving a seam of unlabelled triangles.
    """
    if edges is None:
        edges = directed_edges(faces)
    graph = build_barrier_graph(verts, faces, concavity, barrier_weight, edges=edges)

    seeds = find_tooth_seeds(verts, occlusal_axis, min_separation_mm, max_seeds)
    if len(seeds) == 0:
        return np.zeros(len(faces), dtype=np.int16), seeds

    dist = _sp_dijkstra(graph, indices=seeds)          # (n_seeds, n_verts)
    if dist.ndim == 1:
        dist = dist[None, :]

    nearest = np.argmin(dist, axis=0)
    best = np.min(dist, axis=0)
    vert_labels = np.where(best <= tolerance, nearest + 1, 0).astype(np.int16)

    corner = vert_labels[faces]                        # (n_faces, 3)
    face_labels = np.zeros(len(faces), dtype=np.int16)
    all_same = (corner[:, 0] == corner[:, 1]) & (corner[:, 1] == corner[:, 2])
    face_labels[all_same] = corner[all_same, 0]

    mixed = np.where(~all_same)[0]
    for f in mixed:
        vals, counts = np.unique(corner[f], return_counts=True)
        face_labels[f] = vals[np.argmax(counts)]

    return face_labels, seeds


def boundary_field(
    verts: np.ndarray, faces: np.ndarray, edges=None,
    target_radius_mm: float = 0.6, percentile: float = 99.0, clip: float = 3.0,
) -> np.ndarray:
    """
    The concavity field to use for anything operating on real scan data.

    Three choices here, each measured rather than assumed
    (test_ordering.py, on a 203k-face arch):

    1. SMOOTH FIRST, NORMALIZE SECOND. This is the big one and it was the
       opposite of my original pipeline. Normalizing before smoothing sets
       the scale from raw per-vertex noise and bakes that in permanently;
       no amount of later smoothing recovers the anatomy. Reversing the two
       steps took barrier contrast from 1.4x to 19.6x on a noisy dense mesh
       -- a 14x improvement from reordering two existing operations.

    2. Percentile, not maximum. A single outlier vertex should not set the
       scale for a whole arch.

    3. Smoothing radius in MILLIMETRES, not iterations. A fixed iteration
       count averages over less and less tissue as scans get denser, which
       is exactly backwards: denser scans carry more per-vertex noise, not
       less. 0.6mm is wide enough to survive scanner noise and narrow
       enough to keep the cervical margin sharp.

    Returns roughly [-clip, clip]; positive is concave (valley).
    """
    raw = _concavity_raw(verts, faces, edges)
    iters = adaptive_smoothing_iterations(verts, faces, edges, target_radius_mm)
    smooth = smooth_scalar_fast(raw, faces, iters, edges=edges)
    scale = np.percentile(np.abs(smooth), percentile)
    if scale < 1e-12:
        return smooth
    return np.clip(smooth / scale, -clip, clip)


def snap_seed_to_ridge(
    verts: np.ndarray,
    concavity: np.ndarray,
    seed_point: np.ndarray,
    search_radius_mm: float = 3.0,
) -> np.ndarray:
    """
    Move a clicked seed to the most convex point nearby -- the cusp tip or
    incisal edge.

    Why this matters more than it sounds. Region growing spreads by geodesic
    distance, so where the seed sits determines what is reachable within a
    given tolerance. A click on the middle of the facial surface is roughly
    twice as far from the lingual surface as it is from the cervical sulcus,
    so the flood fills the facial aspect and stops at the gumline before it
    ever wraps over the incisal edge. The clinician sees the selection
    "trapped" on the facial surface and reasonably concludes the edge is
    blocking it -- but measured on a purpose-built fixture the edge carries a
    barrier of exactly 1.00x while the sulcus carries 144x. Nothing is
    blocking; the far side is simply out of range.

    Seeding from the ridge instead puts both surfaces at comparable geodesic
    distance, so one tolerance covers the whole crown.
    """
    d = np.linalg.norm(verts - seed_point, axis=1)
    near = np.where(d <= search_radius_mm)[0]
    if len(near) == 0:
        return seed_point
    # most negative concavity = most convex = ridge or cusp tip
    return verts[near[np.argmin(concavity[near])]]


# =========================================================================
# Mesh conditioning on upload
# =========================================================================

def remove_small_components(faces: np.ndarray, min_fraction: float = 0.02):
    """Drop disconnected islands below `min_fraction` of the largest one.

    Intraoral scans routinely carry loose debris: fragments of tongue,
    cheek, saliva bridges, stray shells floating near the arch. They are
    small, disconnected, and they corrupt everything downstream -- they skew
    the PCA occlusal axis, they attract watershed seeds, and they turn up as
    speckle in a selection.
    """
    comps = connected_components(faces, loop=[])
    if not comps:
        return np.ones(len(faces), dtype=bool), 0
    biggest = len(comps[0])
    keep = np.zeros(len(faces), dtype=bool)
    removed = 0
    for c in comps:
        if len(c) >= min_fraction * biggest:
            keep[c] = True
        else:
            removed += 1
    return keep, removed


def sanitize_scan(verts: np.ndarray, faces: np.ndarray):
    """Make a scan SAFE to compute on, without moving a single surviving vertex.

    WHY THIS HAS TO EXIST, MEASURED. `condition_mesh`'s degenerate filter is
    `area <= 1e-12`, and every comparison against NaN is False — so a triangle
    with a NaN corner has `area = nan`, is not `<= 1e-12`, and SURVIVES the one
    filter whose job is to remove it. The mesh then reaches `boundary_loops` and
    `cap_boundary_loop`, where it dies as `LinAlgError: SVD did not converge`.
    That reaches the clinician as an opaque 500 about a file that opened fine in
    every other program they own.

    Intraoral scanners and the converters around them do emit these: a dropped
    frame, a division by a zero-length normal, a truncated float in an ASCII
    STL. It is not exotic and it is not the clinician's fault.

    RULE 3.1 IS THE CONSTRAINT ON THE REPAIR, not an aside. This function may
    only DELETE — non-finite vertices and the faces that reference them, and
    faces whose indices are out of range or repeat a corner. It must never
    translate, rescale, re-centre or rotate anything, because inter-arch bite
    registration depends on both scans sitting in one raw scanner space. A
    repair that quietly re-centred a damaged arch would fix the crash and break
    the occlusion, which is the worse outcome by far.

    THE BRIEF ASKED FOR `trimesh.repair.sanitize()`. There is no such function —
    not in trimesh 5.1.0, which is what is installed, and not in any release.
    The nearest real equivalent is `Trimesh.remove_infinite_values()`, and this
    is implemented in NumPy instead for two reasons: it must be provably
    delete-only to satisfy rule 3.1, and a scan must still open if an optional
    wheel is missing. `test_failsafes.py` cross-checks it against trimesh's own
    result on the same input and asserts they agree, so the choice is measured
    rather than asserted — and skips if trimesh is absent.

    Returns (verts, faces, report). On a clean scan the arrays are returned
    UNCHANGED — the same objects — and `report["repaired"]` is False.
    """
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    report = {"input_vertices": int(len(v)), "input_faces": int(len(f)),
              "nonfinite_vertices": 0, "faces_dropped_nonfinite": 0,
              "faces_dropped_out_of_range": 0, "faces_dropped_repeated_corner": 0,
              "repaired": False, "method": None}

    finite = np.isfinite(v).all(axis=1)
    n_bad = int((~finite).sum())
    in_range = (f >= 0) & (f < len(v))
    bad_range = ~in_range.all(axis=1)
    repeated = ((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))         if len(f) else np.zeros(0, dtype=bool)

    report["nonfinite_vertices"] = n_bad
    report["faces_dropped_out_of_range"] = int(bad_range.sum())
    report["faces_dropped_repeated_corner"] = int((repeated & ~bad_range).sum())

    if n_bad == 0 and not bad_range.any() and not repeated.any():
        report["method"] = "none needed"
        return verts, faces, report

    # Drop faces first, using only in-range rows to index `finite`.
    keep = ~bad_range & ~repeated
    if n_bad:
        touches_bad = np.zeros(len(f), dtype=bool)
        safe = np.where(keep)[0]
        touches_bad[safe] = ~finite[f[safe]].all(axis=1)
        report["faces_dropped_nonfinite"] = int(touches_bad.sum())
        keep &= ~touches_bad
    f2 = f[keep]

    # Then drop orphaned vertices and remap. Positions are COPIED, never
    # recomputed: `v[used]` is a gather, so every surviving coordinate is the
    # bit pattern that arrived. A test asserts that against the raw upload.
    used = np.unique(f2) if len(f2) else np.zeros(0, dtype=np.int64)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    v2 = v[used]
    f2 = remap[f2]

    report["repaired"] = True
    report["method"] = "numpy delete-only"
    report["output_vertices"] = int(len(v2))
    report["output_faces"] = int(len(f2))
    report["vertices_dropped"] = int(len(v) - len(v2))
    report["warning"] = (
        f"The uploaded scan contained {n_bad} vertices with NaN or infinite "
        f"coordinates and {int(len(f) - len(f2))} unusable faces. They have been "
        f"REMOVED, not corrected — no surviving vertex was moved, so the scan "
        f"stays in its original scanner coordinates and inter-arch registration "
        f"is unaffected. Inspect the area around the damage before cutting.")

    if not np.isfinite(v2).all():
        raise ValueError("sanitize_scan left non-finite vertices; refusing the scan")
    return v2, f2, report


def condition_mesh(verts: np.ndarray, faces: np.ndarray,
                   min_component_fraction: float = 0.02,
                   fill_holes: bool = True,
                   max_hole_verts: int = 60):
    """
    Prepare a raw scan for analysis WITHOUT altering the geometry that
    survives.

    This is the part of "redesign on upload" that is safe to automate, and
    the boundary of what is safe matters clinically. Welding, dropping
    degenerate triangles, deleting disconnected debris and capping small
    holes all either remove non-anatomy or add surface where there was none
    -- no retained vertex moves. The test asserts exactly that: every vertex
    kept is bit-identical to the upload.

    What this deliberately does NOT do is smooth the surface. Laplacian or
    Taubin smoothing would make the mesh prettier and the curvature field
    cleaner, and it would also move the enamel surface the aligner is
    supposed to grip and the printer is supposed to reproduce. Sub-0.1mm
    accuracy is the whole point of the appliance. The noise problem is
    real, but it belongs in the ANALYSIS, not the geometry: boundary_field
    already smooths the curvature field over a fixed physical radius while
    leaving vertex positions untouched, which buys the same robustness at
    no clinical cost.

    Returns (verts, faces, report).
    """
    report = {"input_vertices": int(len(verts)), "input_faces": int(len(faces))}
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64)

    # weld by exact bit pattern - recovers connectivity, moves nothing
    uniq, inverse = np.unique(v, axis=0, return_inverse=True)
    report["welded_vertices"] = int(len(v) - len(uniq))
    v2 = uniq
    # `inverse` is indexed by VERTEX here, so faces are remapped through it.
    # (In parse_stl_bytes the unique runs over triangle corners, where a
    # reshape(-1, 3) is correct instead -- easy to conflate, and it produced
    # a reshape error the first time this ran.)
    inverse = np.asarray(inverse).ravel()
    f2 = inverse[f]

    # degenerate triangles
    a, b, c = v2[f2[:, 0]], v2[f2[:, 1]], v2[f2[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    repeated = ((f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2]))
    bad = (area <= 1e-12) | repeated
    report["degenerate_faces_removed"] = int(bad.sum())
    f2 = f2[~bad]

    # loose debris
    keep, n_islands = remove_small_components(f2, min_component_fraction)
    report["debris_islands_removed"] = int(n_islands)
    report["debris_faces_removed"] = int((~keep).sum())
    f2 = f2[keep]

    used = np.unique(f2)
    remap = -np.ones(len(v2), dtype=np.int64)
    remap[used] = np.arange(len(used))
    v2, f2 = v2[used], remap[f2]

    # small holes only. A large opening is usually the underside of the
    # scan, and capping it with a flat fan would invent anatomy.
    if fill_holes:
        loops = boundary_loops(f2)
        small = [l for l in loops if len(l) <= max_hole_verts]
        report["holes_found"] = int(len(loops))
        report["holes_filled"] = int(len(small))
        report["holes_left_open"] = int(len(loops) - len(small))
        for loop in small:
            half = _boundary_half_edges(f2)
            cap = cap_boundary_loop(v2, loop)
            if len(cap) == 0 or _creates_nonmanifold(f2, cap):
                centre = v2[loop].mean(axis=0)
                v2 = np.vstack([v2, centre])
                ci = len(v2) - 1
                cap = np.array([[loop[i], loop[(i + 1) % len(loop)], ci]
                                for i in range(len(loop))], dtype=np.int64)
            f2 = np.vstack([f2, _orient_cap_against_boundary(cap, half)])

    report["output_vertices"] = int(len(v2))
    report["output_faces"] = int(len(f2))
    return v2, f2, report


def _score_segmentation(labels: np.ndarray, faces: np.ndarray, n_expected=(8, 16)) -> float:
    """Score a candidate whole-arch segmentation without ground truth.

    Rewards a plausible tooth COUNT and consistent tooth SIZES. Real teeth in
    one arch differ in size, but not by an order of magnitude, so a split
    molar or two merged incisors both show up as size outliers. This is what
    lets the separation parameter be chosen automatically instead of being a
    slider the clinician has to guess at.
    """
    ids = np.unique(labels[labels > 0])
    n = len(ids)
    if n == 0:
        return -1e9
    sizes = np.array([(labels == i).sum() for i in ids], dtype=float)
    lo, hi = n_expected
    count_penalty = 0.0 if lo <= n <= hi else min(abs(n - lo), abs(n - hi)) * 0.15
    cv = sizes.std() / max(sizes.mean(), 1e-9)          # size consistency
    tiny = (sizes < 0.15 * np.median(sizes)).sum() / n   # fragments
    return -(count_penalty + cv * 0.6 + tiny * 1.5)


def precompute_arch(
    verts: np.ndarray,
    faces: np.ndarray,
    occlusal_axis: np.ndarray,
    concavity: np.ndarray,
    edges=None,
    separations=(4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5),
    tolerance: float = 12.0,
    root_length_mm: float = 10.0,
) -> dict:
    """
    Analyse the whole arch ONCE, at upload, so interaction is instant.

    This is the architectural answer to a workflow that made the clinician
    tune a tolerance slider and patch gaps with a brush for every tooth: all
    of that was the software computing a selection while the user waited on
    it. Doing the analysis up front means a click selects an
    already-segmented tooth, with its anatomical frame and centre of
    resistance already derived.

    The separation parameter is swept and scored rather than exposed, since
    guessing it was the main reason auto-colouring over-segmented -- the
    right value depends on the patient's crown widths, which the software
    can measure and the clinician should not have to.

    Returns per-tooth records; nothing here invents geometry. Every face
    belongs to the uploaded mesh.
    """
    if edges is None:
        edges = directed_edges(faces)

    best, best_score, best_seeds = None, -np.inf, None
    trials = []
    for sep in separations:
        labels, seeds = auto_segment_arch(verts, faces, concavity, occlusal_axis,
                                          edges=edges, tolerance=tolerance,
                                          min_separation_mm=sep, max_seeds=20)
        sc = _score_segmentation(labels, faces)
        trials.append((sep, int(len(np.unique(labels[labels > 0]))), round(float(sc), 3)))
        if sc > best_score:
            best, best_score, best_seeds, best_sep = labels, sc, seeds, sep

    teeth = []
    for lab in np.unique(best[best > 0]):
        mask = best == lab
        mask = largest_face_component(faces, mask)
        if mask.sum() < 50:
            continue
        sub = faces[mask]
        pts = verts[np.unique(sub)]

        rec = dict(
            id=f"tooth_{len(teeth)+1:02d}",
            label=int(lab),
            n_faces=int(mask.sum()),
            face_indices=np.where(mask)[0].astype(np.int32),
            centroid=pts.mean(axis=0),
            bbox=(pts.min(axis=0), pts.max(axis=0)),
        )

        # rim + frame, derived now rather than at click time.
        # Routed through resolve_long_axis rather than repeating the centroid
        # difference inline: that estimator is what put C_res outside flat
        # molars in the /cut path, and a second copy here would regress the
        # same way. `occlusal_axis` is already occlusally signed by
        # arch_geometry.estimate_arch_frame, so it serves as u_occ directly.
        (cv, cf), _ = split_by_face_mask(verts, faces, mask)
        loops = boundary_loops(cf)
        if loops:
            rim = cv[max(loops, key=len)]
            try:
                u_oa, axis_info = resolve_long_axis(
                    rim, rec["centroid"], {"u_occ": np.asarray(occlusal_axis, float)})
            except ValueError:
                pass                     # degenerate region; leave it frameless
            else:
                rec["u_oa"] = u_oa
                rec["rim_centroid"] = rim.mean(axis=0)
                rec["axis_source"] = axis_info["axis_source"]
                rec["axis_deviation_deg"] = axis_info["axis_deviation_deg"]
                rec["c_res"] = center_of_resistance(
                    dict(centroid=rec["centroid"], rim_centroid=rec["rim_centroid"],
                         u_oa=u_oa), root_length_mm)
        teeth.append(rec)

    # neighbours by centroid proximity, for later interproximal work
    if teeth:
        cents = np.array([t["centroid"] for t in teeth])
        for i, t in enumerate(teeth):
            d = np.linalg.norm(cents - cents[i], axis=1)
            order = np.argsort(d)[1:3]
            t["neighbours"] = [teeth[j]["id"] for j in order]

    return dict(labels=best, seeds=best_seeds, teeth=teeth,
                chosen_separation_mm=float(best_sep), score=float(best_score),
                trials=trials)


def region_boundary_strength(
    verts: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray,
    concavity: np.ndarray, edges=None,
) -> dict:
    """
    Mean concavity along the boundary between each pair of adjacent regions.

    A genuine interproximal contact sits in a concave valley. A watershed
    line running over the middle of a molar's occlusal table does not -- it
    falls between two cusps of the SAME tooth, where the surface is convex or
    flat. Measuring the boundary tells the two apart, which a size or count
    heuristic cannot.
    """
    if edges is None:
        edges = directed_edges(faces)
    i, j = edges
    li, lj = vertex_labels[i], vertex_labels[j]
    cross = (li != lj) & (li > 0) & (lj > 0)
    if not np.any(cross):
        return {}

    a = np.minimum(li[cross], lj[cross])
    b = np.maximum(li[cross], lj[cross])
    strength = 0.5 * (concavity[i[cross]] + concavity[j[cross]])

    out: dict[tuple[int, int], list] = {}
    for pa, pb, s in zip(a, b, strength):
        out.setdefault((int(pa), int(pb)), []).append(float(s))
    return {k: (float(np.mean(v)), len(v)) for k, v in out.items()}


def merge_weak_regions(
    verts: np.ndarray, faces: np.ndarray, vertex_labels: np.ndarray,
    concavity: np.ndarray, target_count: int, edges=None,
    min_boundary_edges: int = 20, max_fraction: float = 0.08,
    forbidden: set | None = None,
) -> tuple[np.ndarray, list]:
    """
    Merge regions across the WEAKEST boundaries until `target_count` remain.

    This is the correction for molar over-segmentation. A watershed seeded on
    cusp tips necessarily splits a multi-cusped molar, because each cusp is a
    separate local maximum -- no separation distance fixes that, since the
    cusps of one molar are closer together than two adjacent premolars.
    Merging afterwards, in order of how weak the dividing boundary is, undoes
    the split without disturbing real interproximal contacts: the fissure
    between two cusps of one tooth is a much weaker concave line than the
    contact between two teeth.

    `max_fraction` caps the size of any merged region as a fraction of the
    arch. It is not optional: without it the merges CHAIN. Measured on a real
    maxillary scan, unconstrained merging to 14 teeth produced one region
    covering 15.3% of the arch and a size ratio of 10.9x, because each merge
    creates a bigger region that then has more weak boundaries to merge
    across. Capping at 8% gave 14 teeth with a largest region of 7.8% and a
    ratio of 5.6x. For reference the RAW watershed scored 3.1x with no
    oversized region at all -- it simply over-segments -- so merging is only
    worth doing with the cap in place.

    Merges are recorded and returned so the decision is auditable rather than
    silent.
    """
    labels = vertex_labels.copy()
    n_faces = len(faces)
    log = []
    # Pairs that splitting deliberately separated. Without this the two
    # operations fight: a freshly cleaved boundary is by construction the
    # weakest one available, so the very next merge re-fuses it. Measured,
    # that oscillated for 12 rounds -- split 1, merge 1, split 1, merge 1 --
    # and never converged.
    forbidden = set() if forbidden is None else set(forbidden)

    def face_sizes(lbl):
        fl = lbl[faces]
        same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
        fla = np.where(same, fl[:, 0], 0)
        ids = np.unique(fla[fla > 0])
        return {int(k): int((fla == k).sum()) for k in ids}

    while True:
        sizes = face_sizes(labels)
        if len(sizes) <= target_count:
            break
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)
        candidates = {
            k: v for k, v in strengths.items()
            if v[1] >= min_boundary_edges
            and (sizes.get(k[0], 0) + sizes.get(k[1], 0)) <= max_fraction * n_faces
            and (k[0], k[1]) not in forbidden and (k[1], k[0]) not in forbidden
        }
        if not candidates:
            break
        (pa, pb), (s, n) = min(candidates.items(), key=lambda kv: kv[1][0])
        labels[labels == pb] = pa
        log.append(dict(merged=(pa, pb), boundary_concavity=round(s, 4), edges=n,
                        remaining=len(sizes) - 1))
    return labels, log


def order_teeth_along_arch(tooth_centroids, occlusal_axis, arch_centroid=None):
    """
    Order tooth regions around the dental arch and number them outward from
    the midline, giving quadrant-relative positions 1..7.

    The arch is a horseshoe, so ordering centroids by their angle about the
    arch centre walks continuously from one posterior end to the other. The
    two middle entries of that walk are the central incisors, which fixes the
    midline without needing any anatomical landmark to be clicked.

    Positions follow the dental convention the clinician uses: 1 = central
    incisor, 2 = lateral, 3 = canine, 4 and 5 = premolars, 6 = first molar,
    7 = second molar. Third molars are not expected -- they are routinely
    excluded from orthodontic planning and are often incompletely captured by
    an intraoral scanner anyway.

    Returns a list of dicts with index, side, position, and angle. Side is
    "A"/"B" relative to the scan, not left/right: distinguishing patient left
    from right needs the scan's own orientation metadata, and guessing it
    would be worse than leaving it to the clinician.
    """
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    cents = np.asarray(tooth_centroids, dtype=float)
    if arch_centroid is None:
        arch_centroid = cents.mean(axis=0)

    rel = cents - arch_centroid
    inplane = rel - np.outer(rel @ axis, axis)

    # a stable 2D basis in the occlusal plane
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    ang = np.arctan2(inplane @ e2, inplane @ e1)

    # The horseshoe is open at the back, so rotate the cut to the widest
    # angular gap. Ordering then starts at one posterior end.
    order = np.argsort(ang)
    gaps = np.diff(np.concatenate([ang[order], [ang[order][0] + 2 * np.pi]]))
    start = (np.argmax(gaps) + 1) % len(order)
    walk = np.concatenate([order[start:], order[:start]])

    n = len(walk)
    half = n // 2
    out = []
    for rank, idx in enumerate(walk):
        if rank < half:                       # first half: posterior -> midline
            side, position = "A", half - rank
        else:                                 # second half: midline -> posterior
            side, position = "B", rank - half + 1
        out.append(dict(index=int(idx), side=side, position=int(position),
                        angle=float(ang[idx])))
    return out


# Mesiodistal crown diameters, permanent dentition, after Wheeler's Dental
# Anatomy averages (mm). Position 1 = central incisor .. 7 = second molar.
# Used to validate a segmentation against real anatomy. Never used to alter
# geometry: these are population averages and an individual patient will
# differ, so they bound plausibility rather than define truth.
MAXILLARY_MD_WIDTH = {1: 8.5, 2: 6.5, 3: 7.5, 4: 7.0, 5: 6.7, 6: 10.0, 7: 9.0}
MANDIBULAR_MD_WIDTH = {1: 5.0, 2: 5.5, 3: 7.0, 4: 7.0, 5: 7.0, 6: 11.0, 7: 10.5}

# Individual variation is real; Wheeler reports ranges of roughly +/-1mm on
# incisors and +/-1.5mm on molars. A measurement outside this tolerance
# suggests a segmentation error rather than an unusual patient.
MD_WIDTH_TOLERANCE_MM = 2.0


def local_arch_tangent(centroid, arch_centre, occlusal_axis):
    """Mesiodistal direction at one point on the arch.

    The arch is a horseshoe, so the mesiodistal axis at any tooth runs
    perpendicular to the radius from the arch centre, within the occlusal
    plane. Deriving it per-tooth rather than using one global axis is what
    makes the measurement valid for molars, whose mesiodistal direction is
    nearly perpendicular to that of the central incisors.
    """
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    rel = np.asarray(centroid, dtype=float) - np.asarray(arch_centre, dtype=float)
    radial = rel - np.dot(rel, axis) * axis
    n = np.linalg.norm(radial)
    if n < 1e-9:
        return None
    radial = radial / n
    tangent = np.cross(axis, radial)
    return tangent / np.linalg.norm(tangent)


def measure_mesiodistal_width(region_verts, centroid, arch_centre, occlusal_axis,
                              occlusal_fraction: float = 0.40):
    """
    Digital caliper: mesiodistal crown width in millimetres.

    Projects the region onto the local arch tangent and takes the extent.
    Replaces face count, which is a biologically invalid proxy for size: a
    heavily fissured first molar carries far more triangles per square
    millimetre than a smooth central incisor, and a partially captured tooth
    loses faces without losing width.

    `occlusal_fraction` restricts the measurement to the occlusal portion of
    the region, and it is not a fudge factor -- clinical mesiodistal width is
    measured at the interproximal CONTACT POINTS, which sit near the junction
    of the occlusal and middle thirds, not at the widest point anywhere on
    the tooth.

    It also corrects a real error. Measured on a mandibular scan, regions
    came out 9-14mm tall where a clinical crown is 7-11mm: the flood had run
    past the cervical margin onto gingiva. Measuring the whole region then
    reported gingival spread rather than crown width and inflated every tooth
    by 2-7mm. Restricting to the occlusal 40% brought a canine from 9.21mm to
    7.16mm against a Wheeler average of 7.0mm.

    Pass occlusal_fraction=1.0 to measure the full region, which is useful
    for diagnosing how far a selection has bled.

    Returns width in mm, or None if the tangent is undefined.
    """
    tangent = local_arch_tangent(centroid, arch_centre, occlusal_axis)
    if tangent is None:
        return None
    pts = np.asarray(region_verts, dtype=float)
    if occlusal_fraction < 1.0:
        axis = occlusal_axis / np.linalg.norm(occlusal_axis)
        h = pts @ axis
        cut = h.min() + (1.0 - occlusal_fraction) * (h.max() - h.min())
        sel = pts[h >= cut]
        if len(sel) >= 3:
            pts = sel
    proj = pts @ tangent
    return float(proj.max() - proj.min())


def validate_against_anatomy(measurements, arch="maxillary",
                             tolerance_mm: float = MD_WIDTH_TOLERANCE_MM):
    """
    Compare measured mesiodistal widths against Wheeler's averages.

    `measurements` is a list of (position, width_mm). Returns per-tooth
    verdicts and a summary. A width far above expectation usually means two
    teeth were merged; far below usually means one tooth was split or only
    partially captured.
    """
    table = MAXILLARY_MD_WIDTH if arch == "maxillary" else MANDIBULAR_MD_WIDTH
    rows, n_ok = [], 0
    # Caller should check `reliable` before quoting pass_rate: when the
    # region count differs from the dentition, positions are assigned
    # sequentially and real teeth get numbered as third molars, so the score
    # is computed over a partial and shifted set. A full arch is BOTH sides,
    # so the expected count is 2 x len(table) = 14, not 7 -- getting that
    # wrong inverted the flag and marked a correct maxilla unscoreable while
    # blessing a 19-region mandible.
    for position, width in measurements:
        expected = table.get(position)
        if expected is None or width is None:
            rows.append(dict(position=position, width=width, expected=None,
                             delta=None, verdict="no reference"))
            continue
        delta = width - expected
        if abs(delta) <= tolerance_mm:
            verdict, ok = "ok", True
        elif delta > 0:
            verdict, ok = "too wide (merged teeth?)", False
        else:
            verdict, ok = "too narrow (split or partial?)", False
        n_ok += ok
        rows.append(dict(position=position, width=round(width, 2),
                         expected=expected, delta=round(delta, 2), verdict=verdict))
    scored = [r for r in rows if r["expected"] is not None]
    return dict(rows=rows, n_ok=n_ok, n_scored=len(scored),
                pass_rate=(n_ok / len(scored)) if scored else 0.0,
                reliable=(len(scored) == 2 * len(table)))


def segment_arch_clinical(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray,
    occlusal_axis: np.ndarray, edges=None,
    seed_separation_mm: float = 3.0,
    cusp_merge_mm: float = 6.5,
    anterior_merge_mm: float = 4.5,
    posterior_merge_mm: float = 8.0,
    position_scaled_merge: bool = True,
    target_teeth: int = 14,
    max_tooth_fraction: float = 0.085,
    tolerance: float = 12.0,
    max_seeds: int = 60,
):
    """
    Whole-arch segmentation calibrated against a real maxillary scan with
    clinician-supplied ground truth (7-7, third molars excluded).

    Two-stage, and the order matters:

    1. DELIBERATELY over-seed at ~3mm. A molar has four or five cusps and each
       is a local maximum, so any seeding fine enough to catch small teeth
       will split molars. Fighting that with a larger separation only merges
       genuine neighbours instead -- the cusps of one molar sit closer
       together (2-5mm) than two adjacent premolars do (7-11mm).

    2. MERGE by cusp distance. Because stage 1 placed seeds finely, the
       distance between two regions' seeds now discriminates: under ~6.5mm
       means two cusps of one tooth, beyond it means two teeth. Merging is
       ordered by weakest boundary concavity, and capped by size so merges
       cannot chain into a blob.

    An earlier version seeded at 6mm and merged on boundary strength alone.
    It produced exactly 14 regions -- and they were not teeth: the second
    premolar came out largest in the arch and a first molar came out
    smallest, because the weakest boundary globally often joins a molar cusp
    to the neighbouring premolar. Count matching ground truth proved nothing;
    only checking sizes against expected crown anatomy caught it.

    The merge window SCALES WITH ARCH POSITION when
    `position_scaled_merge` is on, and it has to. A fixed 6.5mm window was
    calibrated on a maxilla and then swallowed adjacent mandibular incisors,
    which are the narrowest teeth in the mouth at 5.3-5.7mm mesiodistally --
    two adjacent centrals sit closer together than the cusps of a molar, so
    one constant cannot separate "two cusps of one tooth" from "two teeth" in
    both regions of the same arch. Anteriorly the window must be under about
    5mm; at a five-cusped mandibular first molar spanning 11mm it needs to be
    far wider. Interpolating between `anterior_merge_mm` and
    `posterior_merge_mm` by normalised distance from the midline handles both
    without asking the clinician to declare upper or lower.

    Returns (vertex_labels, info).
    """
    if edges is None:
        edges = directed_edges(faces)
    graph = build_barrier_graph(verts, faces, concavity, edges=edges)
    seeds = find_tooth_seeds(verts, occlusal_axis,
                             min_separation_mm=seed_separation_mm, max_seeds=max_seeds)
    if len(seeds) == 0:
        return np.zeros(len(verts), dtype=np.int32), dict(n_teeth=0, seeds=0)

    dist = _sp_dijkstra(graph, indices=seeds)
    if dist.ndim == 1:
        dist = dist[None, :]
    labels = np.where(dist.min(axis=0) <= tolerance,
                      dist.argmin(axis=0) + 1, 0).astype(np.int32)
    seed_pts = verts[seeds]
    n_faces = len(faces)

    # normalised arch position of each seed: 0 at the midline, 1 posteriorly
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    arch_centre = seed_pts.mean(axis=0)
    rel = seed_pts - arch_centre
    inplane = rel - np.outer(rel @ axis, axis)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(basis, axis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    b1 = basis - np.dot(basis, axis) * axis
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    seed_ang = np.arctan2(inplane @ b2, inplane @ b1)
    ang_span = max(np.ptp(seed_ang), 1e-9)
    ang_mid = seed_ang.mean()
    arch_pos = np.clip(np.abs(seed_ang - ang_mid) / (0.5 * ang_span), 0.0, 1.0)

    def window_for(a, b):
        if not position_scaled_merge:
            return cusp_merge_mm
        t = 0.5 * (arch_pos[a - 1] + arch_pos[b - 1])
        return anterior_merge_mm + t * (posterior_merge_mm - anterior_merge_mm)

    def face_sizes(lbl):
        fl = lbl[faces]
        same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
        fla = np.where(same, fl[:, 0], 0)
        ids = np.unique(fla[fla > 0])
        return {int(k): int((fla == k).sum()) for k in ids}

    merges = []
    while True:
        sizes = face_sizes(labels)
        if len(sizes) <= target_teeth:
            break
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)
        cand = {}
        for (a, b), (s, n) in strengths.items():
            if n < 15:
                continue
            if np.linalg.norm(seed_pts[a - 1] - seed_pts[b - 1]) > window_for(a, b):
                continue
            if sizes.get(a, 0) + sizes.get(b, 0) > max_tooth_fraction * n_faces:
                continue
            cand[(a, b)] = (s, n)
        if not cand:
            break
        (pa, pb), (s, n) = min(cand.items(), key=lambda kv: kv[1][0])
        labels[labels == pb] = pa
        merges.append(dict(merged=(int(pa), int(pb)), concavity=round(s, 4)))

    sizes = face_sizes(labels)
    return labels, dict(n_teeth=len(sizes), n_seeds=int(len(seeds)),
                        n_merges=len(merges),
                        size_ratio=round(max(sizes.values()) / max(min(sizes.values()), 1), 2)
                        if sizes else 0.0)


def identify_dentition(
    tooth_centroids, occlusal_axis, arch_depth_axis,
    teeth_per_side: int = 7,
):
    """
    Number teeth outward from the true midline and flag anything beyond
    position `teeth_per_side` as a third molar to be excluded.

    Two clinical facts drive this. Orthodontic planning runs 7-to-7: third
    molars are excluded by convention and are usually captured only partially
    by an intraoral scanner, so they appear as undersized posterior
    fragments. And the two central incisors are the most ANTERIOR teeth in
    the arch, which locates the midline from geometry alone.

    Finding the midline anteriorly rather than by splitting the ordered walk
    in half matters when third molars are present asymmetrically -- one side
    scanned, the other not. A symmetric split then puts 8 regions on one side
    and 9 on the other and shifts every number by one, which on a real
    mandible made a central incisor come out as the largest anterior tooth
    when it should be the narrowest in the mouth.

    Returns a list of dicts: index, side, position, is_third_molar.
    """
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    depth = arch_depth_axis / np.linalg.norm(arch_depth_axis)
    cents = np.asarray(tooth_centroids, dtype=float)
    centre = cents.mean(axis=0)

    rel = cents - centre
    inplane = rel - np.outer(rel @ axis, axis)

    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    ang = np.arctan2(inplane @ e2, inplane @ e1)

    order = np.argsort(ang)
    gaps = np.diff(np.concatenate([ang[order], [ang[order][0] + 2 * np.pi]]))
    start = (np.argmax(gaps) + 1) % len(order)
    walk = list(np.concatenate([order[start:], order[:start]]))

    # anterior extreme locates the midline; sign is resolved by taking
    # whichever end of the depth axis holds the tighter cluster of centroids
    d = inplane @ depth
    if np.ptp(d) < 1e-9:
        anterior_rank = len(walk) // 2
    else:
        for sign in (1.0, -1.0):
            cand = int(np.argmax(d * sign))
            if cand in walk:
                break
        # the midline sits between the two most anterior teeth
        ranks = {idx: r for r, idx in enumerate(walk)}
        most_ant = sorted(walk, key=lambda i: -(d[i] * sign))[:2]
        anterior_rank = int(round(np.mean([ranks[i] for i in most_ant])))

    out = []
    for rank, idx in enumerate(walk):
        if rank <= anterior_rank:
            side = "A"
            position = anterior_rank - rank + 1
        else:
            side = "B"
            position = rank - anterior_rank
        out.append(dict(index=int(idx), side=side, position=int(position),
                        is_third_molar=bool(position > teeth_per_side)))
    return out


# =========================================================================
# Digital mesiodistal caliper
# -------------------------------------------------------------------------
# Wheeler's Dental Anatomy average mesiodistal crown diameters, in mm. These
# replace face-count heuristics for validating a segmentation. Face count is
# a biologically invalid proxy for tooth size: a heavily fissured first molar
# carries far more triangles per square millimetre than a smooth central
# incisor, and a partially captured tooth loses faces without losing width.
# =========================================================================

WHEELER_MAXILLARY_MD = {1: 8.5, 2: 6.5, 3: 7.5, 4: 7.0, 5: 6.7, 6: 10.3, 7: 9.2, 8: 8.5}
WHEELER_MANDIBULAR_MD = {1: 5.0, 2: 5.5, 3: 7.0, 4: 7.0, 5: 7.0, 6: 11.0, 7: 10.5, 8: 10.0}

# Population spread is wide -- Wheeler's own tables carry roughly +/-0.5mm
# standard deviation, and real dentitions vary more than that. A measured
# width within this tolerance of the average is unremarkable; well outside it
# suggests the region is not the tooth it was numbered as.
MD_TOLERANCE_MM = 1.8


def arch_walk(centroids, occlusal_axis):
    """Order tooth centroids continuously around the arch horseshoe.

    Shared by the caliper and by identify_dentition so both agree on which
    tooth neighbours which -- the tangent at a tooth is meaningless if the
    neighbour list disagrees with the numbering.
    """
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    cents = np.asarray(centroids, dtype=float)
    rel = cents - cents.mean(axis=0)
    inplane = rel - np.outer(rel @ axis, axis)

    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(basis, axis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    e1 = basis - np.dot(basis, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    ang = np.arctan2(inplane @ e2, inplane @ e1)
    order = np.argsort(ang)
    gaps = np.diff(np.concatenate([ang[order], [ang[order][0] + 2 * np.pi]]))
    start = (np.argmax(gaps) + 1) % len(order)
    return list(np.concatenate([order[start:], order[:start]])), ang


def arch_tangents(centroids, occlusal_axis):
    """
    Unit mesiodistal direction at each tooth: the local tangent to the arch.

    Estimated by central difference between the neighbouring teeth's
    centroids, projected into the occlusal plane. A tooth's mesiodistal axis
    runs along the arch curve, so the chord between its neighbours
    approximates it well and needs no curve fitting; terminal teeth fall back
    to a one-sided difference.

    Returns an array of unit vectors indexed like `centroids`.
    """
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    cents = np.asarray(centroids, dtype=float)
    walk, _ = arch_walk(cents, axis)
    n = len(walk)

    tangents = np.zeros_like(cents)
    for rank, idx in enumerate(walk):
        if n == 1:
            tangents[idx] = np.array([1.0, 0.0, 0.0])
            continue
        prev_i = walk[rank - 1] if rank > 0 else walk[rank]
        next_i = walk[rank + 1] if rank < n - 1 else walk[rank]
        t = cents[next_i] - cents[prev_i]
        t = t - np.dot(t, axis) * axis          # keep it in the occlusal plane
        nrm = np.linalg.norm(t)
        tangents[idx] = t / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
    return tangents


def mesiodistal_width(points: np.ndarray, tangent: np.ndarray) -> float:
    """True mesiodistal width in mm: extent of the region along the arch
    tangent.

    This is what a caliper measures between the contact points, and unlike
    face count it is in the same units as every published anatomy table.

    Caveat worth carrying: the measured region runs to wherever the flood
    stopped, roughly the cervical margin. If a selection bled onto gingiva
    the width inflates, so an implausibly WIDE result is as much a
    segmentation warning as a narrow one.
    """
    t = tangent / np.linalg.norm(tangent)
    proj = np.asarray(points, dtype=float) @ t
    return float(proj.max() - proj.min())


def measure_dentition(verts, faces, face_labels, occlusal_axis, arch_depth_axis,
                      is_maxillary: bool, teeth_per_side: int = 7):
    """
    Measure every segmented tooth with the caliper and check it against
    Wheeler's averages.

    Returns a list of per-tooth records with side, position, measured width,
    expected width, deviation, and a verdict. Nothing here alters geometry --
    it is a check on whether the segmentation is anatomically credible.
    """
    ids = np.unique(face_labels[face_labels > 0])
    if len(ids) == 0:
        return []

    cents, pts_by_id = [], {}
    for k in ids:
        pts = verts[np.unique(faces[face_labels == k])]
        pts_by_id[int(k)] = pts
        cents.append(pts.mean(axis=0))
    cents = np.array(cents)

    tangents = arch_tangents(cents, occlusal_axis)
    dent = identify_dentition(cents, occlusal_axis, arch_depth_axis, teeth_per_side)
    table = WHEELER_MAXILLARY_MD if is_maxillary else WHEELER_MANDIBULAR_MD

    out = []
    for d in dent:
        i = d["index"]
        k = int(ids[i])
        width = mesiodistal_width(pts_by_id[k], tangents[i])
        expected = table.get(d["position"])
        rec = dict(label=k, side=d["side"], position=d["position"],
                   is_third_molar=d["is_third_molar"],
                   width_mm=round(width, 2),
                   expected_mm=expected,
                   n_faces=int((face_labels == k).sum()))
        if expected is not None:
            dev = width - expected
            rec["deviation_mm"] = round(dev, 2)
            rec["verdict"] = ("ok" if abs(dev) <= MD_TOLERANCE_MM
                              else ("too wide" if dev > 0 else "too narrow"))
        else:
            rec["verdict"] = "unscored (beyond position 7)"
        out.append(rec)
    return out


def _width_envelope(position: float, tolerance_mm: float = 1.5):
    """
    Permissive (ceiling, floor) for mesiodistal width at an arch position.

    Takes the wider of the maxillary and mandibular Wheeler values for the
    ceiling and the narrower for the floor, so one envelope serves both
    arches without asking the clinician to declare upper or lower. It is
    deliberately loose: at position 1 the ceiling is 8.5 + 1.5 = 10.0mm,
    which still admits a wide maxillary central while rejecting two
    mandibular centrals merged into 10.6mm.

    `position` may be fractional; values are interpolated between table
    entries because a region's arch position is estimated continuously during
    merging, before final numbering exists.
    """
    p = float(np.clip(position, 1.0, 7.0))
    lo_i, hi_i = int(np.floor(p)), int(np.ceil(p))
    t = p - lo_i
    def blend(table):
        return table[lo_i] * (1 - t) + table[hi_i] * t
    upper, lower = blend(MAXILLARY_MD_WIDTH), blend(MANDIBULAR_MD_WIDTH)
    return max(upper, lower) + tolerance_mm, min(upper, lower) - tolerance_mm


def constrained_merge(
    verts: np.ndarray, faces: np.ndarray, labels: np.ndarray,
    concavity: np.ndarray, seed_pts: np.ndarray, occlusal_axis: np.ndarray,
    edges=None, target_teeth: int = 14,
    anterior_merge_mm: float = 4.5, posterior_merge_mm: float = 8.0,
    width_tolerance_mm: float = 1.5, occlusal_fraction: float = 0.40,
    min_boundary_edges: int = 15,
):
    """
    Merge regions under a hard anatomical width constraint.

    Wheeler's table stops being a report card and becomes a constraint. Two
    rules:

    CEILING (blocks merges). Before merging, the hypothetical combined
    mesiodistal width is measured. If it exceeds the envelope for that arch
    position, the merge is refused outright. This is what prevents two
    mandibular centrals -- 5.0mm each, and closer together than the cusps of
    a molar -- from fusing into a 10.6mm blob, which no distance-based rule
    could stop because the distance genuinely is small.

    FLOOR (forces merges). After the main pass, any region measuring far
    below the anatomical minimum is a fragment: a molar cusp left stranded,
    or a tooth split in two. Each is merged into its most viable neighbour,
    chosen by weakest dividing boundary among those that keep the result
    under the ceiling.

    All widths are measured on the occlusal fraction only. Measuring whole
    regions reports gingival spread rather than crown width and inflated
    every tooth by 2-7mm on a real mandible.
    """
    if edges is None:
        edges = directed_edges(faces)
    labels = labels.copy()
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    n_faces = len(faces)

    def region_map():
        fl = labels[faces]
        same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
        fla = np.where(same, fl[:, 0], 0)
        return fla, np.unique(fla[fla > 0])

    def region_points(fla, k):
        return verts[np.unique(faces[fla == k])]

    # continuous arch position of every seed: 0 at midline, 1 posteriorly
    centre = seed_pts.mean(axis=0)
    rel = seed_pts - centre
    inplane = rel - np.outer(rel @ axis, axis)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(basis, axis)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    b1 = basis - np.dot(basis, axis) * axis
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    seed_ang = np.arctan2(inplane @ b2, inplane @ b1)
    span = max(np.ptp(seed_ang), 1e-9)
    arch_pos = np.clip(np.abs(seed_ang - seed_ang.mean()) / (0.5 * span), 0.0, 1.0)

    def pos_of(k):
        return 1.0 + arch_pos[k - 1] * 6.0          # arch position -> tooth number 1..7

    def window_of(a, b):
        t = 0.5 * (arch_pos[a - 1] + arch_pos[b - 1])
        return anterior_merge_mm + t * (posterior_merge_mm - anterior_merge_mm)

    def width_of_union(fla, a, b, arch_centre):
        pts = np.vstack([region_points(fla, a), region_points(fla, b)])
        cen = pts.mean(axis=0)
        return measure_mesiodistal_width(pts, cen, arch_centre, axis,
                                         occlusal_fraction=occlusal_fraction)

    blocked_by_width = 0
    merges = []

    # ---- pass 1: ceiling-constrained merging -------------------------
    while True:
        fla, ids = region_map()
        if len(ids) <= target_teeth:
            break
        cents = {int(k): region_points(fla, k).mean(axis=0) for k in ids}
        arch_centre = np.mean(list(cents.values()), axis=0)
        sizes = {int(k): int((fla == k).sum()) for k in ids}
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)

        cand = {}
        for (a, b), (s, n) in strengths.items():
            if n < min_boundary_edges:
                continue
            if np.linalg.norm(seed_pts[a - 1] - seed_pts[b - 1]) > window_of(a, b):
                continue
            ceiling, _ = _width_envelope(0.5 * (pos_of(a) + pos_of(b)), width_tolerance_mm)
            w = width_of_union(fla, a, b, arch_centre)
            if w is None or w > ceiling:
                blocked_by_width += 1
                continue
            cand[(a, b)] = (s, n, w)
        if not cand:
            break
        (pa, pb), (s, n, w) = min(cand.items(), key=lambda kv: kv[1][0])
        labels[labels == pb] = pa
        merges.append(dict(kind="ceiling-ok", merged=(int(pa), int(pb)),
                           width=round(w, 2), concavity=round(s, 4)))

    # ---- pass 2: rescue fragments below the anatomical floor ---------
    forced = 0
    for _ in range(40):
        fla, ids = region_map()
        cents = {int(k): region_points(fla, k).mean(axis=0) for k in ids}
        arch_centre = np.mean(list(cents.values()), axis=0)
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)

        runt, runt_w = None, None
        for k in ids:
            k = int(k)
            w = measure_mesiodistal_width(region_points(fla, k), cents[k], arch_centre,
                                          axis, occlusal_fraction=occlusal_fraction)
            _, floor = _width_envelope(pos_of(k), width_tolerance_mm)
            if w is not None and w < floor and (runt_w is None or w < runt_w):
                runt, runt_w = k, w
        if runt is None:
            break

        options = {}
        for (a, b), (s, n) in strengths.items():
            if runt not in (a, b) or n < min_boundary_edges:
                continue
            ceiling, _ = _width_envelope(0.5 * (pos_of(a) + pos_of(b)), width_tolerance_mm)
            w = width_of_union(fla, a, b, arch_centre)
            if w is None or w > ceiling:
                continue
            options[(a, b)] = (s, w)
        if not options:
            break
        (pa, pb), (s, w) = min(options.items(), key=lambda kv: kv[1][0])
        labels[labels == pb] = pa
        forced += 1
        merges.append(dict(kind="floor-rescue", merged=(int(pa), int(pb)),
                           width=round(w, 2), was=round(runt_w, 2)))

    fla, ids = region_map()
    return labels, dict(n_teeth=int(len(ids)), n_merges=len(merges),
                        blocked_by_width=blocked_by_width, forced_merges=forced,
                        log=merges)


def detect_arch_type(verts, occlusal_axis, fill_threshold: float = 0.02) -> str:
    """
    Maxillary or mandibular, from geometry alone.

    A maxillary cast carries a palatal vault that fills the inside of the
    horseshoe; a mandibular cast has open tongue space there. Measured on
    real scans, mesh fill within 30% of the arch radius was 8.5% for the
    maxilla and 0.0% for the mandible -- unambiguous.

    This is what lets the merge constraints pick the right Wheeler table
    without the clinician declaring which arch they loaded. Note it is
    calibrated on two casts; a heavily trimmed maxillary model with the
    palate cut away would defeat it, and the caller should treat the answer
    as a default the clinician can override rather than a fact.
    """
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    centre = verts.mean(axis=0)
    rel = verts - centre
    proj = rel @ axis
    radial = np.linalg.norm(rel - np.outer(proj, axis), axis=1)
    rmax = np.percentile(radial, 98)
    fill = float((radial < 0.30 * rmax).mean())
    return "maxillary" if fill > fill_threshold else "mandibular"


def expected_width_at(arch_pos: float, table: dict) -> float:
    """Wheeler width interpolated by normalised arch position (0 = midline,
    1 = posterior). Used during merging, before tooth positions are known."""
    p = 1.0 + float(np.clip(arch_pos, 0.0, 1.0)) * (len(table) - 1)
    lo, hi = int(np.floor(p)), int(np.ceil(p))
    lo = max(1, min(lo, len(table))); hi = max(1, min(hi, len(table)))
    if lo == hi:
        return table[lo]
    t = p - lo
    return table[lo] * (1 - t) + table[hi] * t


def segment_arch_constrained(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray,
    occlusal_axis: np.ndarray, edges=None,
    seed_separation_mm: float = 3.0,
    anterior_merge_mm: float = 4.5,
    posterior_merge_mm: float = 8.0,
    target_teeth: int = 14,
    tolerance: float = 12.0,
    max_seeds: int = 60,
    width_ceiling_mm: float = 3.0,
    width_floor_mm: float = 2.5,
    occlusal_fraction: float = 0.40,
    arch_type: str | None = None,
):
    """
    Watershed plus merging under HARD anatomical width constraints.

    Wheeler's mesiodistal averages stop being a report card and become a
    constraint in the loop:

      CEILING  a merge is rejected if the combined crown would exceed the
               expected width for that arch position by more than
               `width_ceiling_mm`. This is what structurally prevents two
               5.0mm mandibular centrals from fusing into a 10.6mm blob --
               previously they merged because their seeds sat closer together
               than the cusps of a molar, and no distance rule could tell the
               two cases apart.

      FLOOR    after the main pass, any region measuring more than
               `width_floor_mm` below expectation is actively merged with its
               most viable neighbour -- a second molar coming out at 5.4mm
               against an expected 9.0mm is a split tooth, not a small one.

    Widths are measured at the occlusal `occlusal_fraction` of each region,
    which both matches where interproximal contacts actually sit and stops
    gingival flood-bleed from inflating the bounding box.

    `arch_type` is auto-detected when None.

    CALIBRATION AND ITS LIMIT. `width_ceiling_mm` defaults to 3.0mm, not the
    1.5mm that strict Wheeler variance would suggest. Swept on two real
    arches: at 1.5mm the ceiling blocked 387 merges on the maxilla and 682 on
    the mandible, so neither reached the target count and mandibular accuracy
    fell to 3/8. At 3.0mm both arches score 10/14 within 2mm of Wheeler --
    the best result achieved, and the first where the two arches agree.

    The reason a tight ceiling backfires is a circular dependency worth
    naming: the ceiling needs a region's TOOTH POSITION to look up its
    expected width, but position is only knowable once segmentation is
    finished. During merging the position is approximated from arch angle,
    and while regions still outnumber teeth that approximation is poor -- so
    a tight ceiling rejects merges using a wrong expectation. A looser
    ceiling still blocks the gross errors (two centrals fusing) while
    tolerating the estimate's noise.

    This also means the floor pass rarely fires: its rescue merges are
    themselves subject to the ceiling. Breaking the circularity properly
    needs a second pass that re-runs merging once positions ARE known, which
    is the obvious next step and is not implemented here.
    """
    if edges is None:
        edges = directed_edges(faces)
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    if arch_type is None:
        arch_type = detect_arch_type(verts, axis)
    table = MAXILLARY_MD_WIDTH if arch_type == "maxillary" else MANDIBULAR_MD_WIDTH

    graph = build_barrier_graph(verts, faces, concavity, edges=edges)
    seeds = find_tooth_seeds(verts, axis, min_separation_mm=seed_separation_mm,
                             max_seeds=max_seeds)
    if len(seeds) == 0:
        return np.zeros(len(verts), dtype=np.int32), dict(n_teeth=0, arch_type=arch_type)

    dist = _sp_dijkstra(graph, indices=seeds)
    if dist.ndim == 1:
        dist = dist[None, :]
    labels = np.where(dist.min(axis=0) <= tolerance,
                      dist.argmin(axis=0) + 1, 0).astype(np.int32)
    seed_pts = verts[seeds]

    # normalised arch position per seed
    centre = seed_pts.mean(axis=0)
    rel = seed_pts - centre
    inplane = rel - np.outer(rel @ axis, axis)
    b = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(b, axis)) > 0.9:
        b = np.array([0.0, 1.0, 0.0])
    b1 = b - np.dot(b, axis) * axis; b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    ang = np.arctan2(inplane @ b2, inplane @ b1)
    span = max(np.ptp(ang), 1e-9)
    arch_pos = np.clip(np.abs(ang - ang.mean()) / (0.5 * span), 0.0, 1.0)
    arch_centre = verts.mean(axis=0)

    def region_verts(lbl, k):
        return verts[np.unique(faces[_face_labels(lbl)[0] == k])]

    def _face_labels(lbl):
        fl = lbl[faces]
        same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
        return np.where(same, fl[:, 0], 0), None

    def width_of(pts, cent):
        if len(pts) < 4:
            return 0.0
        w = measure_mesiodistal_width(pts, cent, arch_centre, axis,
                                      occlusal_fraction=occlusal_fraction)
        return 0.0 if w is None else w

    def snapshot(lbl):
        fla, _ = _face_labels(lbl)
        ids = np.unique(fla[fla > 0])
        info = {}
        for k in ids:
            pts = verts[np.unique(faces[fla == k])]
            cent = pts.mean(axis=0)
            info[int(k)] = dict(pts=pts, centroid=cent, width=width_of(pts, cent))
        return fla, info

    fla, info = snapshot(labels)
    rejected = 0
    merges = []

    # ---- main pass: distance window AND width ceiling ----
    while len(info) > target_teeth:
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)
        cand = {}
        for (a, bb), (s, n) in strengths.items():
            if n < 15 or a not in info or bb not in info:
                continue
            t = 0.5 * (arch_pos[a - 1] + arch_pos[bb - 1])
            window = anterior_merge_mm + t * (posterior_merge_mm - anterior_merge_mm)
            if np.linalg.norm(seed_pts[a - 1] - seed_pts[bb - 1]) > window:
                continue
            combined = np.vstack([info[a]["pts"], info[bb]["pts"]])
            cent = combined.mean(axis=0)
            w = width_of(combined, cent)
            if w > expected_width_at(t, table) + width_ceiling_mm:
                rejected += 1
                continue
            cand[(a, bb)] = (s, n, w)
        if not cand:
            break
        (pa, pb), (s, n, w) = min(cand.items(), key=lambda kv: kv[1][0])
        labels[labels == pb] = pa
        merges.append(dict(merged=(int(pa), int(pb)), width=round(w, 2)))
        fla, info = snapshot(labels)

    # ---- floor pass: rescue regions far too narrow to be a whole tooth ----
    forced = 0
    for _ in range(target_teeth):
        fla, info = snapshot(labels)
        undersized = []
        for k, d in info.items():
            t = arch_pos[k - 1] if k - 1 < len(arch_pos) else 0.5
            if d["width"] < expected_width_at(t, table) - width_floor_mm:
                undersized.append((d["width"], k))
        if not undersized:
            break
        undersized.sort()
        _, k = undersized[0]
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)
        best = None
        for (a, bb), (s, n) in strengths.items():
            if k not in (a, bb) or n < 10:
                continue
            other = bb if a == k else a
            if other not in info:
                continue
            combined = np.vstack([info[k]["pts"], info[other]["pts"]])
            cent = combined.mean(axis=0)
            w = width_of(combined, cent)
            t = 0.5 * (arch_pos[k - 1] + arch_pos[other - 1])
            if w > expected_width_at(t, table) + width_ceiling_mm:
                continue
            if best is None or s < best[0]:
                best = (s, other, w)
        if best is None:
            break
        _, other, w = best
        lo, hi = min(k, other), max(k, other)
        labels[labels == hi] = lo
        forced += 1
        merges.append(dict(merged=(int(lo), int(hi)), width=round(w, 2), forced=True))

    fla, info = snapshot(labels)
    return labels, dict(n_teeth=len(info), arch_type=arch_type,
                        n_seeds=int(len(seeds)), n_merges=len(merges),
                        merges_rejected_by_ceiling=rejected, forced_merges=forced)


def segment_arch_two_pass(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray,
    occlusal_axis: np.ndarray, arch_depth_axis: np.ndarray, edges=None,
    seed_separation_mm: float = 3.0,
    anterior_merge_mm: float = 4.5,
    posterior_merge_mm: float = 8.0,
    target_teeth: int = 14,
    teeth_per_side: int = 7,
    pass1_slack: int = 14,
    tolerance: float = 12.0,
    max_seeds: int = 60,
    width_ceiling_mm: float = 2.5,
    width_floor_mm: float = 2.5,
    occlusal_fraction: float = 0.40,
    arch_type: str | None = None,
):
    """
    Two-pass segmentation that breaks the position/width circular dependency.

    The problem this solves: a width ceiling needs a region's TOOTH POSITION
    to look up its Wheeler value, but position is only knowable once
    segmentation is close to finished. Estimating position from arch angle on
    a 26-region fracture applies the wrong anatomical constraint to the wrong
    region, so a strict ceiling rejects the very merges needed to reach the
    right count -- measured, it blocked 387 merges on a maxilla and 682 on a
    mandible and made accuracy worse than no constraint at all.

    PASS 1 merges on topology alone -- cusp distance and boundary weakness --
    and stops EARLY, at `target_teeth + pass1_slack`. Stopping early is the
    whole trick: merging cannot be undone, so pass 1 must not fuse two real
    teeth. Leaving slack means pass 2 still has room to work.

    PASS 2 re-derives tooth positions with identify_dentition after EVERY
    merge, so each region is bound to its actual Wheeler width rather than an
    angular guess, and the strict ceiling becomes meaningful. Re-identifying
    each iteration matters because the numbering is only trustworthy as the
    count approaches the true one.

    PASS 3 is the floor: with positions now accurate, regions measuring far
    below their expected width are genuine splits, and are force-merged with
    the most viable neighbour.

    RESULTS AND AN UNSOLVED CASE. On a real maxilla this reaches the target
    14 regions with 10 of 14 crowns within 2mm of Wheeler. On a real mandible
    it does NOT reach 14: it stalls at 18-20 regions across every parameter
    combination swept. The mandible carries both extremes in one arch --
    incisors at 5.0-5.5mm, the narrowest teeth in the mouth, and five-cusped
    first molars at 11mm -- and pass 1 fuses adjacent incisors before pass 2
    can protect them. Since merging is irreversible, pass 2 cannot recover.
    Splitting, not merging, is what that case needs.

    Beware the accuracy figure when the count is wrong: identify_dentition
    numbers regions sequentially, so with 19 regions it labels real teeth as
    third molars and the score is computed over a partial set. Use
    `n_teeth == target_teeth` as a precondition before trusting any
    per-tooth verdict.

    Returns (vertex_labels, info).
    """
    if edges is None:
        edges = directed_edges(faces)
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    if arch_type is None:
        arch_type = detect_arch_type(verts, axis)
    table = MAXILLARY_MD_WIDTH if arch_type == "maxillary" else MANDIBULAR_MD_WIDTH
    arch_centre = verts.mean(axis=0)

    graph = build_barrier_graph(verts, faces, concavity, edges=edges)
    seeds = find_tooth_seeds(verts, axis, min_separation_mm=seed_separation_mm,
                             max_seeds=max_seeds)
    if len(seeds) == 0:
        return np.zeros(len(verts), dtype=np.int32), dict(n_teeth=0, arch_type=arch_type)

    dist = _sp_dijkstra(graph, indices=seeds)
    if dist.ndim == 1:
        dist = dist[None, :]
    labels = np.where(dist.min(axis=0) <= tolerance,
                      dist.argmin(axis=0) + 1, 0).astype(np.int32)
    seed_pts = verts[seeds]

    rel = seed_pts - seed_pts.mean(axis=0)
    inplane = rel - np.outer(rel @ axis, axis)
    b = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(b, axis)) > 0.9:
        b = np.array([0.0, 1.0, 0.0])
    b1 = b - np.dot(b, axis) * axis; b1 /= np.linalg.norm(b1)
    ang = np.arctan2(inplane @ np.cross(axis, b1), inplane @ b1)
    span = max(np.ptp(ang), 1e-9)
    arch_pos = np.clip(np.abs(ang - ang.mean()) / (0.5 * span), 0.0, 1.0)

    def regions_of(lbl):
        fl = lbl[faces]
        same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
        fla = np.where(same, fl[:, 0], 0)
        out = {}
        for k in np.unique(fla[fla > 0]):
            pts = verts[np.unique(faces[fla == k])]
            cent = pts.mean(axis=0)
            w = measure_mesiodistal_width(pts, cent, arch_centre, axis,
                                          occlusal_fraction=occlusal_fraction)
            out[int(k)] = dict(pts=pts, centroid=cent, width=0.0 if w is None else w)
        return out

    def width_of(pts):
        cent = pts.mean(axis=0)
        w = measure_mesiodistal_width(pts, cent, arch_centre, axis,
                                      occlusal_fraction=occlusal_fraction)
        return 0.0 if w is None else w

    # ---------------- PASS 1: topology only, stop early ----------------
    pass1_target = target_teeth + pass1_slack
    info = regions_of(labels)
    p1 = 0
    while len(info) > pass1_target:
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)
        cand = {}
        for (a, bb), (s, n) in strengths.items():
            if n < 15 or a not in info or bb not in info:
                continue
            t = 0.5 * (arch_pos[a - 1] + arch_pos[bb - 1])
            window = anterior_merge_mm + t * (posterior_merge_mm - anterior_merge_mm)
            if np.linalg.norm(seed_pts[a - 1] - seed_pts[bb - 1]) > window:
                continue
            cand[(a, bb)] = (s, n)
        if not cand:
            break
        (pa, pb), _ = min(cand.items(), key=lambda kv: kv[1][0])
        labels[labels == pb] = pa
        p1 += 1
        info = regions_of(labels)

    # ---------------- PASS 2: positions known, strict ceiling ----------------
    def positions_now(inf):
        keys = sorted(inf)
        cents = np.array([inf[k]["centroid"] for k in keys])
        dent = identify_dentition(cents, axis, arch_depth_axis,
                                  teeth_per_side=teeth_per_side)
        return {keys[d["index"]]: d for d in dent}

    p2 = rejected = 0
    while len(info) > target_teeth:
        pos = positions_now(info)
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)
        cand = {}
        for (a, bb), (s, n) in strengths.items():
            if n < 12 or a not in info or bb not in info:
                continue
            combined = np.vstack([info[a]["pts"], info[bb]["pts"]])
            w = width_of(combined)
            # the merged region takes the identity of the larger fragment
            bigger = a if len(info[a]["pts"]) >= len(info[bb]["pts"]) else bb
            p = pos.get(bigger, {}).get("position", teeth_per_side)
            expected = table.get(min(max(p, 1), teeth_per_side), table[teeth_per_side])
            if w > expected + width_ceiling_mm:
                rejected += 1
                continue
            cand[(a, bb)] = (s, n)
        if not cand:
            break
        (pa, pb), _ = min(cand.items(), key=lambda kv: kv[1][0])
        labels[labels == pb] = pa
        p2 += 1
        info = regions_of(labels)

    # ---------------- PASS 3: floor, rescue split teeth ----------------
    forced = 0
    for _ in range(target_teeth):
        pos = positions_now(info)
        undersized = []
        for k, d in info.items():
            p = pos.get(k, {}).get("position")
            if p is None or p > teeth_per_side:
                continue
            if d["width"] < table[p] - width_floor_mm:
                undersized.append((d["width"] - table[p], k, p))
        if not undersized:
            break
        undersized.sort()
        _, k, p = undersized[0]
        strengths = region_boundary_strength(verts, faces, labels, concavity, edges)
        best = None
        for (a, bb), (s, n) in strengths.items():
            if k not in (a, bb) or n < 8:
                continue
            other = bb if a == k else a
            if other not in info:
                continue
            w = width_of(np.vstack([info[k]["pts"], info[other]["pts"]]))
            if w > table[p] + width_ceiling_mm:
                continue
            if best is None or s < best[0]:
                best = (s, other)
        if best is None:
            break
        other = best[1]
        lo, hi = min(k, other), max(k, other)
        labels[labels == hi] = lo
        forced += 1
        info = regions_of(labels)

    return labels, dict(n_teeth=len(info), arch_type=arch_type,
                        n_seeds=int(len(seeds)),
                        pass1_merges=p1, pass2_merges=p2,
                        pass2_rejected=rejected, floor_merges=forced)


def split_fused_region(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray,
    labels: np.ndarray, region_label: int, occlusal_axis: np.ndarray,
    arch_centre: np.ndarray, new_label: int, edges=None,
    barrier_weight: float = 6.0, min_cleave_concavity: float = 0.15,
):
    """
    Cleave a fused region into two along its weakest internal boundary.

    The missing operation in a merge-only architecture. Merging is
    irreversible, so a pass that fuses two mandibular incisors -- which sit
    5.0mm apart, closer than the cusps of a single 11mm molar -- can never be
    recovered by more merging. Splitting is what recovers it.

    Method: place one seed at the occlusal-most point of each end of the
    region along its local mesiodistal tangent, then run a two-source
    watershed on a subgraph restricted to that region. The cleave line falls
    where the two floods meet, which under barrier weighting is the line of
    maximum concavity between them -- the interproximal contact. The cut is
    therefore found by anatomy rather than imposed as a geometric midpoint,
    which matters because a fused pair is rarely two equal halves.

    Returns (labels, ok). `ok` is False when the region cannot be split --
    too few vertices, a degenerate tangent, or one side collapsing to almost
    nothing, which usually means the region was a single tooth after all.
    """
    if edges is None:
        edges = directed_edges(faces)
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)

    idx = np.where(labels == region_label)[0]
    if len(idx) < 50:
        return labels, False

    pts = verts[idx]
    tangent = local_arch_tangent(pts.mean(axis=0), arch_centre, axis)
    if tangent is None:
        return labels, False

    proj = pts @ tangent
    height = pts @ axis
    lo_cut, hi_cut = np.percentile(proj, 25), np.percentile(proj, 75)
    mesial = idx[proj <= lo_cut]
    distal = idx[proj >= hi_cut]
    if len(mesial) == 0 or len(distal) == 0:
        return labels, False

    # occlusal-most point at each end: the cusp tip or incisal edge
    seed_a = int(mesial[np.argmax(verts[mesial] @ axis)])
    seed_b = int(distal[np.argmax(verts[distal] @ axis)])
    if seed_a == seed_b:
        return labels, False

    # subgraph over this region only, so the flood cannot escape through
    # neighbouring teeth and return
    i, j = edges
    inside = np.zeros(len(verts), dtype=bool)
    inside[idx] = True
    keep = inside[i] & inside[j]
    if not np.any(keep):
        return labels, False

    remap = -np.ones(len(verts), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    li, lj = remap[i[keep]], remap[j[keep]]
    c = np.maximum(concavity, 0.0)
    penalty = np.exp(barrier_weight * c)
    length = np.linalg.norm(verts[j[keep]] - verts[i[keep]], axis=1)
    w = np.maximum(length * 0.5 * (penalty[i[keep]] + penalty[j[keep]]), 1e-12)
    sub = csr_matrix((w, (li, lj)), shape=(len(idx), len(idx)))

    d = _sp_dijkstra(sub, indices=[int(remap[seed_a]), int(remap[seed_b])])
    if d.ndim == 1:
        d = d[None, :]
    reachable = np.isfinite(d).any(axis=0)
    side_b = (d[1] < d[0]) & reachable

    # a split that leaves a sliver was not a real interproximal contact
    frac = side_b.sum() / max(reachable.sum(), 1)
    if frac < 0.20 or frac > 0.80:
        return labels, False

    # WHAT does the cut pass through? A real interproximal contact sits in a
    # concave valley; bisecting a single healthy crown cuts straight across
    # convex enamel. Without this check the splitter cheerfully halves an
    # intact tooth -- the sliver guard alone accepts it, since half a crown
    # is neither a sliver nor the whole. That was the engine of the
    # split-then-merge oscillation: cut a good tooth, merge it back, repeat.
    side_full = np.zeros(len(verts), dtype=bool)
    side_full[idx[side_b]] = True
    in_region = np.zeros(len(verts), dtype=bool)
    in_region[idx] = True
    cross = in_region[i] & in_region[j] & (side_full[i] != side_full[j])
    if not np.any(cross):
        return labels, False
    cleave_concavity = float(np.mean(0.5 * (concavity[i[cross]] + concavity[j[cross]])))
    if cleave_concavity < min_cleave_concavity:
        return labels, False

    out = labels.copy()
    out[idx[side_b]] = new_label
    return out, True


def sculpt_arch_widths(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray, labels: np.ndarray,
    occlusal_axis: np.ndarray, arch_depth_axis: np.ndarray, arch_type: str,
    edges=None, teeth_per_side: int = 7, split_trigger: float = 1.5,
    occlusal_fraction: float = 0.40, max_splits: int = 8,
    skip_regions: set | None = None,
):
    """
    Split regions that are too wide to be one tooth, then re-evaluate.

    Triggered when a region's measured mesiodistal width exceeds
    `split_trigger` times the Wheeler expectation for its identified
    position. Each successful cleave is fed back through position
    identification, because splitting changes the numbering of everything
    distal to it.
    """
    if edges is None:
        edges = directed_edges(faces)
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    table = MAXILLARY_MD_WIDTH if arch_type == "maxillary" else MANDIBULAR_MD_WIDTH
    arch_centre = verts.mean(axis=0)
    labels = labels.copy()
    log = []

    for _ in range(max_splits):
        fl = labels[faces]
        same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
        fla = np.where(same, fl[:, 0], 0)
        ids = np.unique(fla[fla > 0])
        if len(ids) == 0:
            break

        info = {}
        for k in ids:
            pts = verts[np.unique(faces[fla == k])]
            cent = pts.mean(axis=0)
            w = measure_mesiodistal_width(pts, cent, arch_centre, axis,
                                          occlusal_fraction=occlusal_fraction)
            info[int(k)] = dict(centroid=cent, width=0.0 if w is None else w)

        keys = sorted(info)
        cents = np.array([info[k]["centroid"] for k in keys])
        dent = identify_dentition(cents, axis, arch_depth_axis,
                                  teeth_per_side=teeth_per_side)
        pos = {keys[d["index"]]: d["position"] for d in dent}

        skip = set() if skip_regions is None else skip_regions
        worst = None
        for k, d in info.items():
            p = pos.get(k)
            if p is None or k in skip:
                continue
            expected = table.get(min(max(p, 1), teeth_per_side), table[teeth_per_side])
            ratio = d["width"] / max(expected, 1e-6)
            if ratio >= split_trigger and (worst is None or ratio > worst[0]):
                worst = (ratio, k, d["width"], expected)
        if worst is None:
            break

        ratio, k, width, expected = worst
        new_label = int(labels.max()) + 1
        labels, ok = split_fused_region(verts, faces, concavity, labels, k, axis,
                                        arch_centre, new_label, edges=edges)
        if not ok:
            # cannot cleave it; stop rather than loop on the same region
            log.append(dict(region=int(k), width=round(width, 2),
                            expected=expected, result="split failed"))
            break
        log.append(dict(region=int(k), width=round(width, 2), expected=expected,
                        ratio=round(ratio, 2), result="split",
                        pair=(int(k), int(new_label))))

    return labels, log


def segment_arch_sculpted(
    verts: np.ndarray, faces: np.ndarray, concavity: np.ndarray,
    occlusal_axis: np.ndarray, arch_depth_axis: np.ndarray, edges=None,
    target_teeth: int = 14, teeth_per_side: int = 7,
    split_trigger: float = 1.5, width_ceiling_mm: float = 2.5,
    max_rounds: int = 12, arch_type: str | None = None, **seg_kwargs,
):
    """
    Alternate splitting and merging until the arch settles.

    Neither operation alone converges. Merging cannot undo a fused pair, so a
    merge-only pipeline stalls on the mandible where 5.0mm incisors sit
    closer together than the cusps of an 11mm molar. Splitting alone moves
    the region count the WRONG way -- cleaving a fused pair took a maxilla
    from 14 regions to 15 with no gain, because the fragments then needed
    merging that never came.

    So: split what is too wide, merge what is too many, repeat. A round that
    changes nothing ends the loop. Regions that fail to cleave are recorded
    and not retried, which is what stops the loop oscillating on a region
    that merely looks wide.

    This does not reach 14/14 on either real arch -- see the summary in
    HANDOVER.md. It is the architecture that makes further progress
    possible, not a finished result.
    """
    if edges is None:
        edges = directed_edges(faces)
    axis = occlusal_axis / np.linalg.norm(occlusal_axis)
    if arch_type is None:
        arch_type = detect_arch_type(verts, axis)

    labels, info = segment_arch_two_pass(
        verts, faces, concavity, axis, arch_depth_axis, edges=edges,
        target_teeth=target_teeth, teeth_per_side=teeth_per_side,
        width_ceiling_mm=width_ceiling_mm, arch_type=arch_type, **seg_kwargs)

    history = [dict(stage="two_pass", regions=info["n_teeth"])]
    protected: set = set()      # boundaries created by splitting
    failed_splits: set = set()  # regions that would not cleave
    for rnd in range(max_rounds):
        before = int(len(np.unique(labels[labels > 0])))

        labels, split_log = sculpt_arch_widths(
            verts, faces, concavity, labels, axis, arch_depth_axis, arch_type,
            edges=edges, teeth_per_side=teeth_per_side,
            split_trigger=split_trigger, max_splits=3,
            skip_regions=failed_splits)
        n_split = 0
        for L in split_log:
            if L["result"] == "split":
                protected.add(L["pair"])
                n_split += 1
            else:
                failed_splits.add(L["region"])

        n_merged = 0
        if len(np.unique(labels[labels > 0])) > target_teeth:
            labels, merge_log = merge_weak_regions(
                verts, faces, labels, concavity, target_count=target_teeth,
                edges=edges, max_fraction=0.09, forbidden=protected)
            n_merged = len(merge_log)

        after = int(len(np.unique(labels[labels > 0])))
        history.append(dict(stage=f"round{rnd+1}", splits=n_split,
                            merges=n_merged, regions=after))
        if n_split == 0 and n_merged == 0:
            break

    final = int(len(np.unique(labels[labels > 0])))
    return labels, dict(n_teeth=final, arch_type=arch_type,
                        two_pass=info, history=history)
# ======================================================================
# RESTORED SECTION — export, socket carving and clearance.
# ======================================================================

def _stl_blob(verts: np.ndarray, faces: np.ndarray) -> tuple[bytes, int]:
    """The binary-STL byte stream and its triangle count.

    Single implementation shared by write_binary_stl (to a path) and
    write_binary_stl_bytes (to memory), so the file a lab receives on disk and
    the one it receives inside a ZIP can never drift apart.
    """
    import struct
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln < 1e-12, 1.0, ln)

    rec = np.zeros(len(f), dtype=np.dtype([
        ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    rec["normal"] = n.astype("<f4")
    rec["v"] = tri.astype("<f4")

    blob = (b"Clinical Micro-Planner".ljust(80, b"\0")
            + struct.pack("<I", len(f))
            + rec.tobytes())
    return blob, len(f)


def write_binary_stl(verts: np.ndarray, faces: np.ndarray, path: str) -> int:
    """Write a binary STL. Returns the triangle count.

    Signature and return value are load-bearing: export_planned_setup,
    export_nested_pair and the existing tests all depend on them. Add new
    delivery mechanisms alongside (see write_binary_stl_bytes), never by
    changing this.
    """
    blob, n_tri = _stl_blob(verts, faces)
    with open(path, "wb") as fh:
        fh.write(blob)
    return n_tri


def write_binary_stl_bytes(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """The same STL, in memory — for streaming a ZIP straight to the browser."""
    return _stl_blob(verts, faces)[0]

# =========================================================================
# Socket cup — a formed alveolar depression, not a lid over a hole
# =========================================================================
# WHAT WAS ACTUALLY WRONG, measured rather than assumed.
#
# The socket was closed with a triangle fan from the rim centroid, and the
# reported symptom was a "dark irregular opening". The natural diagnosis is
# that a scalloped, non-convex margin makes the fan self-intersect. It does
# not. A radial scallop r = R + A*cos(k*th) is star-shaped about its own
# centroid for ANY amplitude, and a fan over a star-shaped polygon provably
# cannot self-intersect:
#
#     r = 8 + 1.5*cos(4th)   star-shaped True   fan crossings 0
#     r = 8 + 7.5*cos(4th)   star-shaped True   fan crossings 0   (r 0.5..15.5)
#
# Nor does 3D non-planarity break it: on a realistic margin the fan stays
# angularly monotonic about its apex with adjacent-normal dot >= 0.996 even at
# +/-4mm of vertical scallop.
#
# The defect is DEPTH. The fan's apex is the rim centroid, so the cap sits AT
# the margin plane — measured apex depth 0.00mm on a rim spanning -1.50..+1.50.
# It is a flat lid over a hole, and at a grazing angle that is exactly the dark
# disc that was reported. An alveolus descends into the bone.

SOCKET_DEPTH_MM = 3.5         # a molar socket floor below the cervical margin
SOCKET_INSET_FRACTION = 0.70  # total lateral inset, as a fraction of the rim's
                              # minimum inward clearance. Radial inset is safe
                              # strictly below that clearance and folds above
                              # it (measured: r = 8 + 6*cos(4t), clearance 2.0,
                              # gives 0 crossings at 1.9mm and 8 at 2.0mm), so
                              # the margin is what keeps the deepest ring a real
                              # loop rather than a collapsed point.
SOCKET_INNER_SCALE = 0.25     # retained name for the floor's share of the rim
                              # radius; see SOCKET_INSET_FRACTION, which is what
                              # actually bounds the inset.
SOCKET_FLOOR_CLEARANCE_MM = 0.5   # never cut closer than this to the underside
MIN_SOCKET_INSET_MM = 0.25    # below this the rings are not worth building: a
                              # margin necked to 0.1mm of clearance would yield
                              # a 0.07mm inset, which is a flat lid wearing the
                              # word "cup". The honest answer there is the flat
                              # floor, and saying so is what makes the fallback
                              # reachable by real pathology instead of dead code.


def _signed_area_2d(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _points_in_triangle_2d(P, a, b, c) -> np.ndarray:
    """Barycentric containment, vectorised over P."""
    v0, v1 = c - a, b - a
    v2 = P - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return np.zeros(len(P), bool)
    d20, d21 = v2 @ v0, v2 @ v1
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    return (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)


def ear_clip_polygon(poly2d: np.ndarray) -> np.ndarray:
    """Triangulate a simple (possibly non-convex) polygon by ear clipping.

    Not scipy.spatial.Delaunay: Delaunay triangulates the CONVEX HULL, so on a
    scalloped cervical margin it spans exactly the concavities that matter —
    cap_boundary_loop already has to filter its output for this reason. Not
    trimesh or shapely either; this is ~40 lines of NumPy and adds no
    dependency to a module whose isolation is the point.

    Returns triangles as indices into poly2d, wound counter-clockwise.
    """
    poly2d = np.asarray(poly2d, float)
    n = len(poly2d)
    if n < 3:
        return np.empty((0, 3), int)

    order = np.arange(n)
    if _signed_area_2d(poly2d) < 0:      # normalise to CCW so "ear" means convex
        order = order[::-1].copy()

    tris, guard = [], 0
    while len(order) > 3 and guard < 4 * n:
        guard += 1
        m = len(order)
        # One gather per pass, then scalar work per candidate and an early
        # break. Vectorising convexity across all remaining vertices was tried
        # and is SLOWER (34ms against 21ms at n=234): the first or second
        # candidate is almost always an ear, so computing the other 232 is
        # wasted. The containment test still uses a boolean mask rather than a
        # Python list comprehension, which is where the real cost was.
        pts = poly2d[order]
        clipped = False
        for i in range(m):
            A_, B_, C_ = pts[i - 1], pts[i], pts[(i + 1) % m]
            if (B_[0] - A_[0]) * (C_[1] - A_[1]) - (B_[1] - A_[1]) * (C_[0] - A_[0]) <= 1e-12:
                continue                  # reflex or collinear — not an ear
            keep = np.ones(m, bool)
            keep[[(i - 1) % m, i, (i + 1) % m]] = False
            if _points_in_triangle_2d(pts[keep], A_, B_, C_).any():
                continue                  # another vertex sits inside it
            tris.append([order[(i - 1) % m], order[i], order[(i + 1) % m]])
            order = np.delete(order, i)
            clipped = True
            break
        if not clipped:
            break                         # polygon is not simple; caller decides
    if len(order) == 3:
        tris.append(list(order))
    return np.asarray(tris, int) if tris else np.empty((0, 3), int)


def ear_clip_polygon_robust(poly2d: np.ndarray) -> tuple[np.ndarray, int]:
    """Ear clipping that always returns n-2 triangles. Returns (tris, forced).

    Deliberately NOT a change to ear_clip_polygon. That function bailing is
    meaningful — it is how a caller learns the polygon is not simple, and a test
    pins exactly that. This one is for the floor of the cast base, where the
    crossings have already been repaired and a short return would be a hole in
    the bottom of a printed model rather than a useful signal.

    When no strictly-convex empty ear exists the polygon is, in exact
    arithmetic, still guaranteed to have one (Meisters' two-ears theorem). What
    defeats the search is a NEEDLE — the boundary spiking out and back so a
    vertex's two neighbours nearly coincide — where the ear triangle's area
    falls under the 1e-12 convexity epsilon. Clipping the largest-area candidate
    anyway costs a sliver in a flat, invisible, coplanar face and keeps every
    vertex, which is the whole point: deleting vertices instead is what striated
    the wall. Measured on the real scan the fallback fires ZERO times; it earns
    its keep on a regular synthetic grid, which is a far worse case than scan
    data ever is.
    """
    poly2d = np.asarray(poly2d, float)
    n = len(poly2d)
    if n < 3:
        return np.empty((0, 3), int), 0

    order = np.arange(n)
    if _signed_area_2d(poly2d) < 0:
        order = order[::-1].copy()

    tris, forced, guard = [], 0, 0
    while len(order) > 3 and guard < 6 * n:
        guard += 1
        m = len(order)
        pts = poly2d[order]
        prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        cross = ((pts[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
                 - (pts[:, 1] - prev[:, 1]) * (nxt[:, 0] - prev[:, 0]))

        clipped = False
        for i in np.where(cross > 1e-12)[0]:
            keep = np.ones(m, bool)
            keep[[(i - 1) % m, i, (i + 1) % m]] = False
            if _points_in_triangle_2d(pts[keep], prev[i], pts[i], nxt[i]).any():
                continue
            tris.append([order[(i - 1) % m], order[i], order[(i + 1) % m]])
            order = np.delete(order, i)
            clipped = True
            break

        if not clipped:
            i = int(np.argmax(cross))     # least-bad corner; a sliver, not a hole
            tris.append([order[(i - 1) % m], order[i], order[(i + 1) % m]])
            order = np.delete(order, i)
            forced += 1

    if len(order) == 3:
        tris.append(list(order))
    return (np.asarray(tris, int) if tris else np.empty((0, 3), int)), forced


def _is_star_shaped_2d(poly2d: np.ndarray) -> bool:
    """True if every boundary point is visible from the origin.

    Equivalently: the boundary winds monotonically in angle about it, and winds
    exactly once. That is the condition under which insetting every vertex
    toward the centre yields nested loops that cannot cross.

    Direction-agnostic on purpose. An earlier version closed the loop with
    ang[0] + 2*pi, which silently assumed counter-clockwise winding;
    boundary_loops returns whichever direction the triangulation gives, and on
    a real clockwise rim that closure produced a spurious +717 degree step, so
    a perfectly good ring (radius 4.01..5.11) was declared non-star-shaped and
    every socket fell back to a flat floor. Wrapping each step into (-pi, pi]
    handles both directions.
    """
    ang = np.arctan2(poly2d[:, 1], poly2d[:, 0])
    step = np.diff(np.append(ang, ang[0]))
    step = (step + np.pi) % (2 * np.pi) - np.pi        # per-edge turn
    if not ((step > 0).all() or (step < 0).all()):
        return False
    return abs(abs(float(step.sum())) - 2 * np.pi) < 1e-6   # winds exactly once


def _polygon_degenerate(poly2d: np.ndarray, eps: float = 1e-9) -> bool:
    """True if the closed polygon has a zero-length edge or a self-touch.

    DEGENERACY IS CHECKED FIRST, and this is not a formality — it was a hole.

    An inset that reaches a vertex's own radius collapses it onto the centre.
    Several vertices then coincide, their edges have zero length, and the
    signed-distance predicate in _crossing_pairs divides by nothing and returns
    0.0 for every test, so it reported a PINCHED polygon as simple. Measured on
    r = 8 + 6*cos(4t) inset by exactly r_min = 2.0mm: 8 genuine crossings at
    index separation 39 and 79, and the guard said True. Deeper insets (2.5,
    3.0) it caught correctly — the miss is specific to the collapse itself,
    which is the very case the guard exists for.
    """
    P = np.asarray(poly2d, float)
    n = len(P)
    if n < 3:
        return True
    if (np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1) < eps).any():
        return True                        # zero-length edge: vertices coincide
    from scipy.spatial import cKDTree
    pairs = cKDTree(P).query_pairs(eps, output_type="ndarray")
    if len(pairs):
        gap = np.abs(pairs[:, 0] - pairs[:, 1])
        if ((gap > 1) & (gap != n - 1)).any():
            return True                    # non-adjacent vertices touch: a pinch
    return False


def _crossing_pairs(poly2d: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Index pairs (i, j), i < j, whose closed-polygon edges properly cross.

    ONE implementation, shared by polygon_is_simple (which asks whether any
    exist) and prune_to_simple (which has to remove them). Two copies of a
    predicate this fiddly would drift, and the guard passing while the pruner
    looped forever is exactly the failure that costs a day.

    The prefilter is SPATIAL — it compares each edge's x/y extents. The index
    arithmetic that follows only EXCLUDES adjacent edges; it never restricts how
    far apart in the loop a crossing may be found. Computing four signed
    distances for all n^2 pairs is exact but allocates four float (n,n) arrays:
    25.6ms at n=400. Edge boxes on a ring overlap only with their neighbours, so
    the prefilter leaves O(n) pairs to test properly.
    """
    P = np.asarray(poly2d, float)
    n = len(P)
    if n < 3:
        return np.empty((0, 2), int)
    A, B = P, np.roll(P, -1, axis=0)

    lox, hix = np.minimum(A[:, 0], B[:, 0]), np.maximum(A[:, 0], B[:, 0])
    loy, hiy = np.minimum(A[:, 1], B[:, 1]), np.maximum(A[:, 1], B[:, 1])
    overlap = ((lox[:, None] <= hix[None, :] + eps) & (lox[None, :] <= hix[:, None] + eps) &
               (loy[:, None] <= hiy[None, :] + eps) & (loy[None, :] <= hiy[:, None] + eps))
    i = np.arange(n)
    diff = np.abs(i[:, None] - i[None, :])
    overlap &= (diff > 1) & (diff != n - 1)
    pairs = np.argwhere(np.triu(overlap, 1))
    if len(pairs) == 0:
        return np.empty((0, 2), int)

    p, q = pairs[:, 0], pairs[:, 1]

    def side(a, b, c):
        d = b - a
        ln = np.linalg.norm(d, axis=-1)
        cr = d[:, 0] * (c[:, 1] - a[:, 1]) - d[:, 1] * (c[:, 0] - a[:, 0])
        return np.where(ln < eps, 0.0, cr / np.where(ln < eps, 1.0, ln))

    d1, d2 = side(A[q], B[q], A[p]), side(A[q], B[q], B[p])
    d3, d4 = side(A[p], B[p], A[q]), side(A[p], B[p], B[q])
    cross = (((d1 > eps) & (d2 < -eps)) | ((d1 < -eps) & (d2 > eps))) & \
            (((d3 > eps) & (d4 < -eps)) | ((d3 < -eps) & (d4 > eps)))
    return pairs[cross]


def polygon_is_simple(poly2d: np.ndarray, eps: float = 1e-9) -> bool:
    """True if no two non-adjacent edges of the closed polygon properly cross.

    This is what the socket rings actually need, and it is tested directly
    rather than inferred from star-shapedness. A real cervical margin is a
    clean ring but its boundary loop carries small angular zigzags from the
    triangulation, so strict angular monotonicity rejects rims that inset
    perfectly well — measured on a real rim of radius 4.01..5.11, which is
    about as well behaved as a margin gets.

    O(n^2), fully vectorised: ~137 rim vertices is 19k edge pairs and runs in
    single-digit milliseconds; a 2175-vertex cast rim measures 0.15s.
    """
    if _polygon_degenerate(poly2d, eps):
        return False
    return len(_crossing_pairs(poly2d, eps)) == 0


def first_hit_distance(origin, direction, verts, faces, max_dist=50.0,
                       radius_mm=3.0):
    """Distance to the first triangle a ray meets, or None.

    Möller–Trumbore, vectorised, behind a cheap radius prefilter so it does not
    become the expensive part of a cut. `radius_mm` must exceed the mesh's
    longest triangle edge — see the prefilter comment.
    """
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    if len(f) == 0:
        return None

    # Prefilter on ONE vertex per face, by squared perpendicular distance to
    # the ray. Testing all three vertices with min/max straddle bounds is
    # exact but measured 117ms at 203k faces — most of the socket's whole
    # budget. One vertex and two dot products is ~10x cheaper and just as safe
    # in practice: `radius_mm` only has to exceed the longest triangle edge,
    # and intraoral scan triangles are sub-millimetre.
    e1_, e2_ = _perp_basis(d)
    p0 = v[f[:, 0]] - o
    x0, y0 = p0 @ e1_, p0 @ e2_
    near = (x0 * x0 + y0 * y0) < (radius_mm * radius_mm)
    if not near.any():
        return None
    f = f[near]

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    edge1, edge2 = v1 - v0, v2 - v0
    h = np.cross(d, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    ok = np.abs(det) > 1e-12
    if not ok.any():
        return None
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = o - v0
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, edge1)
    vv = np.einsum("j,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", edge2, q) * inv
    hit = ok & (u >= -1e-9) & (vv >= -1e-9) & (u + vv <= 1 + 1e-9) & (t > 1e-6) & (t < max_dist)
    return float(t[hit].min()) if hit.any() else None


def _perp_basis(d):
    """Two unit vectors spanning the plane perpendicular to d."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def _floor_triangulation(poly2d: np.ndarray) -> tuple[np.ndarray, int]:
    """Ear-clip a floor so it traverses boundary edge k as (k -> k+1).

    ear_clip_polygon normalises its OUTPUT to counter-clockwise, which means a
    clockwise input comes back with every boundary edge traversed backwards.
    Both callers bridge the floor with a wall that traverses those edges the
    other way, so on a clockwise rim the floor and the wall end up agreeing —
    and two faces that agree on a shared edge are inconsistently oriented.

    This is invisible to every count-based check: edge multiplicities are
    orientation-blind, so the mesh reports zero open and zero non-manifold edges
    while half its normals point into the solid. Measured on a real mandibular
    scan, one sealed socket put 229 directed edges in agreement. A boolean
    engine handed that returns a plausible wrong solid rather than an error,
    which is why _winding_is_consistent exists and is asserted separately.
    """
    tris, forced = ear_clip_polygon_robust(poly2d)
    if len(tris) == 0:
        raise ValueError(
            "The rim does not triangulate: its projection into the margin plane is "
            "not a simple polygon. This usually means the cut left a self-crossing "
            "boundary rather than one clean loop.")
    if _signed_area_2d(poly2d) < 0:
        tris = tris[:, ::-1]
    return tris, forced


def build_socket_cup(rim_xyz, u_oa, depth_mm: float = SOCKET_DEPTH_MM, rings: int = 3,
                     base_verts=None, base_faces=None):
    """A carved alveolar cup closing the cervical rim, descending into the bone.

    Returns (new_vertices, faces, info):
      * faces index the RIM vertices as 0..n_rim-1 and newly created interior
        vertices as n_rim.., so the caller offsets them into its own buffer;
      * new_vertices are only the interior ones, in that order.

    Construction, and why each step is the way it is:

    1. Plane fit by SVD of the mean-centred rim, normal from the smallest
       singular vector. The ONLY singular-value gate is s1/s0, which catches a
       rim that is a line rather than a ring. There is deliberately no s2/s1
       "planarity" gate: CLAUDE.md section 5 records that it is anti-correlated
       with the error it appears to measure — a cos(2*th) margin scallop drives
       it to 0.75 with a perfectly exact normal.
    2. Depth runs along the fitted plane normal, not u_oa. The two agree in
       hemisphere (the normal is signed against u_oa) and usually within a few
       degrees, but measuring "depth below the rim plane" along the plane's own
       normal is what makes that depth exactly the number asked for.
    3. Rings inset by a fixed DISTANCE in polar coordinates about the rim
       centre, r_k(th) = r(th) - d_k — not by a uniform scale factor, and not
       by a normal/medial-axis offset. All three differ, measured on
       r = 8 + A*cos(4*th), counting self-crossings:

           A     inset 1.0 / 2.0 / 3.0 mm      operation
           1.5     0 /  0 /  0                 radial (this)
           1.5     0 /  0 /  4                 normal offset
           4.5     0 /  0 /  0                 radial (this)
           4.5     0 /  4 / 12                 normal offset
           6.0     0 /  8 / 24                 radial (this)
           6.0     0 / 12 / 16                 normal offset

       Scaling interleaves the loops: it shrinks the scallop's AMPLITUDE along
       with the radius, so on A=1.5 a 0.75 scale puts ring 1's peaks at 7.13
       past ring 0's troughs at 6.50 — 26 crossing pairs before this changed.
       Normal offset pinches wherever it exceeds the local medial-axis
       distance, which is the straight-skeleton behaviour and why it fails at
       A=4.5 where radial does not.

       Radial inset does NOT nest unconditionally — an earlier version of this
       docstring claimed it did, and the A=6.0 row above is the counterexample.
       It nests when d stays below the minimum radius, because r(th) - d is
       then still positive and, for an angularly monotonic rim, still
       single-valued in th. So the total inset is capped at
       SOCKET_INSET_FRACTION of the measured minimum inward clearance
       (info["min_clearance_mm"], info["inset_mm"]), which also keeps the
       deepest ring a real loop — an inset all the way to the centre collapses
       it to a point, the bridging quads degenerate and there is nothing left
       to ear-clip. Rims that are not angularly monotonic have no such proof,
       so every loop that would be built is validated with polygon_is_simple;
       failing that, or a rim with no room to inset at all, falls to a flat
       ear-clipped floor at full depth reported as info["profile"] ==
       "flat_fallback" rather than silently folding.
    4. Winding is consistent by CONSTRUCTION, then the whole cup is flipped
       once if its floor faces the wrong way. Per-triangle orientation — right
       for the old flat fan — is wrong here: a cup's side walls are nearly
       parallel to u_oa, so their dot against it is near zero and the sign
       becomes noise. The floor is horizontal, so the test is unambiguous there
       and consistent winding carries it to every other triangle.
    5. If a base mesh is supplied, depth is clamped so the cup cannot erupt
       through the underside of the cast.
    """
    rim = np.asarray(rim_xyz, float)
    n_rim = len(rim)
    if n_rim < 3:
        raise ValueError(f"A socket rim needs at least 3 vertices, got {n_rim}.")
    rings = max(1, int(rings))

    u_oa = np.asarray(u_oa, float)
    nu = np.linalg.norm(u_oa)
    if nu < 1e-9:
        raise ValueError("u_oa is a zero vector; the apical direction is undefined.")
    u_oa = u_oa / nu

    centroid = rim.mean(axis=0)
    _, s, vt = np.linalg.svd(rim - centroid, full_matrices=False)
    if s[0] < 1e-9 or s[1] / s[0] < MIN_RIM_RING_RATIO:
        raise ValueError(
            f"The rim is a line, not a ring (s1/s0 = {0.0 if s[0] < 1e-9 else s[1]/s[0]:.4f}), "
            f"so it does not bound a socket.")

    normal = vt[2] / np.linalg.norm(vt[2])
    if normal @ u_oa < 0:
        normal = -normal                       # occlusal, matching u_oa
    e1 = vt[0] / np.linalg.norm(vt[0])
    e2 = np.cross(normal, e1)

    P = np.column_stack([(rim - centroid) @ e1, (rim - centroid) @ e2])

    # --- depth clamp against the cast ------------------------------------
    info_depth_clamped = False
    if base_verts is not None and base_faces is not None and len(base_faces):
        hit = first_hit_distance(centroid, -normal, base_verts, base_faces,
                                 max_dist=depth_mm + 5.0)
        if hit is not None and hit < depth_mm + SOCKET_FLOOR_CLEARANCE_MM:
            depth_mm = max(0.0, hit - SOCKET_FLOOR_CLEARANCE_MM)
            info_depth_clamped = True

    star = _is_star_shaped_2d(P)
    new_pts, faces = [], []

    # Validate the rings we would actually build, rather than inferring from
    # star-shapedness. Radial inset leaves every vertex's ANGLE untouched, so
    # the question is simply whether each inset loop stays simple — and that is
    # cheap to answer outright. Angular monotonicity is too strict a proxy: a
    # real rim's triangulation puts small backward steps in the loop and would
    # send a perfectly good margin down the flat fallback.
    # --- de-duplicate the rim's PROJECTION --------------------------------
    # Where a cervical margin folds, two rim vertices distinct in 3D project to
    # the same point in the margin plane. The rim keeps both — nothing moves —
    # but EVERY interior vertex is generated from the projection, so a duplicate
    # there puts two interior vertices at one position. Binary STL stores
    # positions and every reader welds on load, so those two merge on the way
    # into the lab's slicer and take their faces with them: measured on a real
    # lower molar, a socket that reported itself watertight came back from a
    # round-trip with a non-manifold edge carrying four faces.
    #
    # So the interior is built on the UNIQUE projections and the rim is bridged
    # to it through `image`, exactly as build_cast_base bridges its wall to a
    # decimated floor. Where consecutive rim vertices share an image the quad
    # degenerates to a single triangle, which is correct and closes the same way.
    _, _first = np.unique(P, axis=0, return_index=True)
    surv = np.sort(_first)
    m = len(surv)
    if m < 3:
        raise ValueError(
            f"The rim projects onto only {m} distinct points in the margin plane, so "
            f"it does not bound a socket floor.")
    _lookup = {(float(P[i, 0]), float(P[i, 1])): pos for pos, i in enumerate(surv)}
    image = np.array([_lookup[(float(x), float(y))] for x, y in P], int)
    Pd = P[surv]

    def bridge_from_rim(target):
        """Rim (0..n_rim-1) to a ring of `m` vertices, through `image`."""
        out = []
        for i in range(n_rim):
            j = (i + 1) % n_rim
            ta, tb = int(target[image[i]]), int(target[image[j]])
            if ta == tb:
                out.append([i, j, ta])
            else:
                out.append([i, j, tb])
                out.append([i, tb, ta])
        return out

    # WHICH of the three conditions sends a socket to the flat floor, in words
    # and with the number. The fallback is usually a verdict on the CUT, not on
    # this function: measured on one real molar, a crown picked straight from
    # segmentation labels has a ragged self-touching margin whose loop passes
    # 0.134mm from its own centre (rim 7.3 x 2.3mm, area 6.2mm2) and falls back,
    # while the same tooth flooded geodesically the way the app does gives a
    # clean 10.6 x 5.6mm ring with 1.342mm of clearance and cups. Saying so is
    # what points the clinician at the selection instead of at the engine.
    radius = np.linalg.norm(Pd, axis=1)
    min_clearance = float(radius.min())
    total_inset = SOCKET_INSET_FRACTION * min_clearance
    fallback_reason = None

    if depth_mm <= 1e-6:
        insettable = False
        fallback_reason = (
            f"Socket depth is {depth_mm:.3f}mm, so there is nothing to cup — the cap "
            f"sits flat on the margin plane by definition.")
    elif total_inset < MIN_SOCKET_INSET_MM:
        insettable = False
        fallback_reason = (
            f"The rim passes within {min_clearance:.3f}mm of its own centre, so the "
            f"widest safe inset is {SOCKET_INSET_FRACTION:.2f} x that = "
            f"{total_inset:.3f}mm, under the {MIN_SOCKET_INSET_MM}mm minimum. Anything "
            f"below that is a flat lid wearing the word 'cup'. A margin this narrow "
            f"usually means the selection is ragged rather than the tooth: check the "
            f"cut before the geometry.")
    else:
        _unit = Pd / radius[:, None]
        insettable = all(
            polygon_is_simple(Pd - _unit * (total_inset * k / rings))
            for k in range(1, rings + 1))
        if not insettable:
            fallback_reason = (
                f"The inset rings self-intersect at {total_inset:.3f}mm, so a cup would "
                f"fold through itself. The margin is re-entrant enough that radial inset "
                f"is not safe on it.")

    if insettable:
        # Inset by DISTANCE, not by scale and not along the normal. See the
        # docstring's measured table for why all three differ, and why this one
        # is capped at a fraction of the minimum inward clearance rather than
        # trusted unconditionally.
        unit = Pd / radius[:, None]

        rings_idx = []
        for k in range(1, rings + 1):
            frac = k / rings
            pts2d = Pd - unit * (total_inset * frac)
            depth = depth_mm * frac
            start = n_rim + len(new_pts)
            for xy in pts2d:
                new_pts.append(centroid + xy[0] * e1 + xy[1] * e2 - normal * depth)
            rings_idx.append(np.arange(start, start + m))
        floor_2d = Pd - unit * total_inset
        # Ear-clip the FLOOR loop, not the rim. Radial inset preserves angular
        # order but not shape, so the rim's triangulation is not automatically
        # valid on the smaller polygon.
        floor_tris, floor_forced = _floor_triangulation(floor_2d)

        faces.extend(bridge_from_rim(rings_idx[0]))
        for a_loop, b_loop in zip(rings_idx[:-1], rings_idx[1:]):
            for i in range(m):
                j = (i + 1) % m
                faces.append([a_loop[i], a_loop[j], b_loop[j]])
                faces.append([a_loop[i], b_loop[j], b_loop[i]])
        inner = rings_idx[-1]
        for t in floor_tris:
            faces.append([inner[t[0]], inner[t[1]], inner[t[2]]])
        profile = "cup"
    else:
        # No room to inset (or zero depth): a flat ear-clipped floor at depth,
        # bridged straight to the rim. Correct, just not cupped.
        floor_2d = Pd
        floor_tris, floor_forced = _floor_triangulation(Pd)
        start = n_rim
        for xy in Pd:
            new_pts.append(centroid + xy[0] * e1 + xy[1] * e2 - normal * depth_mm)
        floor = np.arange(start, start + m)
        faces.extend(bridge_from_rim(floor))
        for t in floor_tris:
            faces.append([floor[t[0]], floor[t[1]], floor[t[2]]])
        profile = "flat_fallback"

    new_pts = np.asarray(new_pts, float).reshape(-1, 3)
    faces = np.asarray(faces, int).reshape(-1, 3)

    # --- one global orientation decision, taken on the floor --------------
    all_pts = np.vstack([rim, new_pts])
    floor_face_idx = np.arange(len(faces) - len(floor_tris), len(faces))
    tri = all_pts[faces[floor_face_idx]]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if float(fn.sum(axis=0) @ normal) < 0:
        faces = faces[:, ::-1]

    deepest = float(np.max((rim.mean(axis=0) - all_pts[n_rim:]) @ normal)) if len(new_pts) else 0.0
    inner_area = abs(_signed_area_2d(floor_2d))
    info = dict(
        profile=profile, rings=rings, depth_mm=float(depth_mm),
        depth_clamped=info_depth_clamped,
        # What actually gates the cup: every loop that would be built was
        # checked for self-intersection. `angularly_monotonic` is descriptive
        # only — it used to be called star_shaped and used as the gate, and it
        # reads False on rims that cup perfectly well (a C-shaped margin does),
        # so the name was actively misleading about which test decides.
        inset_validated=bool(insettable),
        # None when it cupped; otherwise the condition that fired, with numbers.
        fallback_reason=fallback_reason,
        angularly_monotonic=bool(star),
        min_clearance_mm=round(min_clearance, 4),
        inset_mm=round(total_inset if insettable else 0.0, 4),
        inset_fraction=SOCKET_INSET_FRACTION,
        rim_points=int(n_rim), unique_rim_points=int(m),
        new_points=int(len(new_pts)),
        deepest_below_rim_mm=deepest,
        floor_area_fraction=float(inner_area / max(abs(_signed_area_2d(P)), 1e-12)),
        rim_ring_ratio=float(s[1] / s[0]),
        normal=normal,
    )
    return new_pts, faces, info


# =========================================================================
# Module 8: the virtual cast base — trim to a horseshoe, extrude to a solid
# =========================================================================
#
# WHAT THE REAL EXPORT PROVED, measured on case_lower.stl (94,848 v / 187,625 f).
#
# An intraoral scan is an OPEN SHELL. This one reports open_edges 2101,
# nonmanifold_edges 0, and condition_mesh finds exactly ONE hole: those 2101
# edges are the perimeter of the scanned region, not a defect to repair.
#
# cap_and_close sealed that perimeter by fanning across it — and the perimeter
# of an arch is a HORSESHOE, so the cap spans the U-shaped tongue opening.
# cap_boundary_loop on that loop returns 2104 triangles for a 2101-vertex loop:
# a full hull triangulation, i.e. it bridges the opening. Topologically closed,
# anatomically a membrane over the tongue space. No lab can print it. It also
# took 100.2 SECONDS — Delaunay plus one Python _point_in_polygon per triangle.
#
# trim -> extrude replaces it and measures 4.1s on the same scan.

ARCH_TRIM_MARGIN_MM = 7.0     # half-width of the retained band, measured from
                              # the occlusal ridge curve. Measured on a real
                              # mandibular scan:
                              #
                              #   margin  kept    material still inside the arch
                              #    (mm)  faces    opening beyond the margin
                              #     3    58.4%    29.78%
                              #     5    82.0%    15.30%
                              #     7    96.7%     3.35%
                              #     9    99.6%     0.41%
                              #    11   100.0%     0.00%
                              #
                              # 9.0 was the previous default and removed 0.44% —
                              # not a trim. 7.0 removes the deep lingual sulcus
                              # and vestibule floor, which sits 6-9mm apical of
                              # the cervical margin (height medians -9.3mm at
                              # 7-9mm out and -10.9mm at 9-11mm, against crowns
                              # that live within ~5mm of the ridge), and still
                              # keeps every crown plus ~1.5mm of gingival collar.
                              # 5.0 removes 18% and cuts into the molars' buccal
                              # and lingual walls.
CAST_BASE_THICKNESS_MM = 12.0 # flat bottom, this far apical of the LOWEST point
                              # of the kept surface.
ARCH_CURVE_BINS = 72          # angular bins used to sample the occlusal ridge
ARCH_CURVE_MIN_BIN = 20       # a bin with fewer vertices than this is noise
ARCH_CURVE_SMOOTH = 3         # [0.25, 0.5, 0.25] passes over the ridge polyline
RIDGE_BAND_MM = 0.5           # how far below a bin's summit still counts as
                              # ridge, for the weighted mean in fit_arch_curve
ARCH_CURVE_DENSIFY = 12       # samples per polyline segment, so a KD-tree
                              # distance-to-POINTS approximates distance-to-CURVE
MIN_FLOOR_OUTLINE_POINTS = 4  # only catches an outline that has COLLAPSED. Two
                              # earlier guards here were both wrong, and in
                              # opposite directions:
                              #   * "no more than X% of the rim may be dropped"
                              #     dates from when pruning crossings was the
                              #     only thing removing vertices, and became
                              #     wrong once the outline was decimated on
                              #     purpose — a real scan rim goes 2145 -> 469
                              #     points at 0.5mm spacing, which it read as
                              #     catastrophic;
                              #   * a floor of 24 points rejected a flat
                              #     rectangular cast, whose outline is FOUR
                              #     corners once its collinear edge points are
                              #     dropped, and which is perfectly printable.
                              # The membrane is guarded by MAX_PRUNE_ARC_FRACTION
                              # and correctness by the ear clip; this only stops
                              # a polygon from vanishing.
MAX_PRUNE_ARC_FRACTION = 0.15 # THE MEMBRANE GUARD. Pruning deletes the arc
                              # between two crossing edges, which leaves a chord
                              # from one end to the other — and a long chord is
                              # exactly the web across the tongue space this
                              # whole phase exists to remove. Every crossing
                              # measured is local: separation <= 38 of ~2200 on
                              # the real scan (1.7%), max arc 0.10 on the
                              # synthetic worst case. Anything longer is not a
                              # boundary stepping over itself and is refused.
MIN_EAR_THINNESS = 1e-6       # 2*area / longest_side^2 — dimensionless, so one
                              # threshold serves a coarse mesh and a dense one.
                              # A corner below it is a NEEDLE (the boundary
                              # spiking out and back so its neighbours nearly
                              # coincide) or a collinear triple; ear clipping can
                              # never resolve one, and it contributes no shape.
                              #
                              # Fixed, with no escalation. Escalating this was
                              # tried and is actively harmful: it amplified a
                              # 1e-15 rounding difference into a 0.6% volume
                              # change under a pure rigid transform, because
                              # flipping ear_clip_polygon's own 1e-12 comparison
                              # jumped the tolerance a whole decade. Measured,
                              # one fixed decade does the job — on a regular
                              # synthetic grid it takes the floor from 62 forced
                              # clips and 23% of area missing to 1 forced clip
                              # and area exact to 3.7e-16, and the value between
                              # 1e-6 and 1e-3 makes no difference to either.
COINCIDENT_PROJECTION_MM = 1e-9   # two rim vertices distinct in 3D can project
                                  # onto ONE point in the base plane where the
                                  # margin folds. Deduplicating those is what
                                  # lets the floor triangulate at full rim
                                  # resolution, and missing it is what made an
                                  # earlier build decimate the outline to 0.5mm
                                  # spacing and throw away 78% of the rim. It
                                  # also has to happen for the export to survive
                                  # a reader's weld — see weld_vertices.
                                  #
                                  # DECIMATION IS GONE, and the numbers are why.
                                  # Local crossing repair plus this dedup, with
                                  # nothing else, measured on the real scan:
                                  #   margin 7: rim 2133 -> 2014 (gap 5.6%)
                                  #   margin 9: rim 2144 -> 2046 (gap 4.6%)
                                  # and the ear clip completed exactly (2012/2012
                                  # and 2044/2044, zero forced, triangulated area
                                  # equal to the outline's to the last digit).
                                  # The 78% loss was never necessary; it striated
                                  # the cast wall, because many rim vertices
                                  # sharing one floor vertex fan into slivers.


def _arch_basis(arch_frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(origin, e1, e2, u_occ) — an orthonormal frame with e1,e2 spanning the
    occlusal plane.

    Taken from the arch frame rather than from a PCA of the mesh. The frame's
    axes were derived from clicked landmarks, so they rotate WITH the geometry:
    congruence of the resulting base under a rigid transform is then exact by
    construction instead of depending on SVD picking the same eigenvector twice.
    e2 comes out equal to u_sag, so the 2D plane is (transverse, sagittal).
    """
    u_occ = np.asarray(arch_frame["u_occ"], float)
    u_occ = u_occ / np.linalg.norm(u_occ)
    e1 = np.asarray(arch_frame["u_tra"], float)
    e1 = e1 - u_occ * (e1 @ u_occ)
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        raise ValueError("arch_frame's transverse axis is parallel to its occlusal normal.")
    e1 = e1 / n1
    return np.asarray(arch_frame["origin"], float), e1, np.cross(u_occ, e1), u_occ


def _polyline_samples(curve: np.ndarray, factor: int = ARCH_CURVE_DENSIFY) -> np.ndarray:
    """Densify an OPEN polyline so nearest-point-in-cloud approximates
    nearest-point-on-curve. Open, not closed: an arch is a horseshoe, and
    joining the last control point back to the first would lay a chord straight
    across the tongue space — the very region the trim exists to discard."""
    curve = np.asarray(curve, float)
    if len(curve) < 2:
        return curve
    t = np.linspace(0.0, 1.0, max(1, int(factor)), endpoint=False)[:, None]
    segs = curve[:-1][None, :, :] * (1 - t[:, :, None]) + curve[1:][None, :, :] * t[:, :, None]
    return np.vstack([segs.reshape(-1, 2), curve[-1]])


def fit_arch_curve(verts: np.ndarray, arch_frame: dict,
                   bins: int = ARCH_CURVE_BINS,
                   min_bin: int = ARCH_CURVE_MIN_BIN,
                   smooth: int = ARCH_CURVE_SMOOTH,
                   height_percentile: float = 75.0) -> tuple[np.ndarray, dict]:
    """The occlusal ridge, as a smoothed polyline in the arch plane.

    THE SIGN MATTERS AND IS EASY TO GET BACKWARDS. arch_frame's u_occ points
    AWAY from the tissue, out of the mouth (arch_frame.py). The occlusal ridge —
    cusp tips and incisal edges — is therefore at MAXIMUM +u_occ. The -u_occ
    extreme is the floor of the mouth, and a curve fitted through it would trace
    the tongue and trim away the dentition.

    Method: take the dentition-bearing vertices (top quartile by height), bin
    them by angle about their own 2D centroid, and keep the highest vertex in
    each bin. Then CUT THE SEQUENCE AT ITS LARGEST ANGULAR GAP, because an arch
    is an open horseshoe: ordering bins from -pi to +pi without that cut leaves
    one long chord leaping across the posterior opening, and every face in the
    tongue space then measures "close to the curve" and survives the trim.
    arch_walk uses the same largest-gap rule for the same reason.

    A parametric fit was tried first and rejected on measurement: polynomials of
    degree 3-6 in the angle parameter fit a horseshoe with 2.9-7.1mm of residual,
    which is worse than the anatomy being measured.

    Returns (curve2d, info); curve2d is (K,2) in the (e1,e2) basis.
    """
    verts = np.asarray(verts, float)
    origin, e1, e2, u_occ = _arch_basis(arch_frame)
    rel = verts - origin
    P = np.column_stack([rel @ e1, rel @ e2])
    H = rel @ u_occ                                  # + is occlusal. See above.

    crown = H >= np.percentile(H, height_percentile)
    if crown.sum() < max(3, min_bin):
        raise ValueError(
            f"Only {int(crown.sum())} vertices sit in the occlusal quartile; there is "
            f"no ridge to fit. Check that the occlusal plane landmarks are correct.")
    centre = P[crown].mean(axis=0)

    ang = np.arctan2(P[:, 1] - centre[1], P[:, 0] - centre[0])
    bidx = np.clip(((ang + np.pi) / (2 * np.pi) * bins).astype(int), 0, bins - 1)
    # Each bin's ridge point is a HEIGHT-WEIGHTED MEAN of everything within
    # RIDGE_BAND_MM of that bin's summit, not the single highest vertex.
    # argmax is discontinuous in the input: two near-equally-high vertices swap
    # under a floating-point perturbation and the control point jumps across the
    # bin, which moved the fitted curve enough to change the trim by dozens of
    # faces under a pure rigid transform. A vertex enters this weighting with
    # weight zero and grows smoothly, so the curve is continuous.
    pts, angs = [], []
    for k in range(bins):
        sel = np.where(bidx == k)[0]
        if len(sel) < min_bin:
            continue
        h = H[sel]
        w = np.clip(h - (h.max() - RIDGE_BAND_MM), 0.0, None)
        tot = w.sum()
        pts.append((P[sel] * w[:, None]).sum(axis=0) / tot if tot > 1e-12
                   else P[sel[np.argmax(h)]])
        angs.append(-np.pi + 2 * np.pi * (k + 0.5) / bins)
    if len(pts) < 3:
        raise ValueError(
            f"The occlusal ridge sampled to only {len(pts)} points across {bins} bins. "
            f"Either the scan is tiny or the occlusal plane is wrong.")
    pts = np.asarray(pts, float)
    angs = np.asarray(angs, float)

    # Cut at the largest angular gap: that gap IS the arch opening.
    gaps = np.diff(np.append(angs, angs[0] + 2 * np.pi))
    start = (int(np.argmax(gaps)) + 1) % len(angs)
    pts = np.roll(pts, -start, axis=0)
    gap_deg = float(np.degrees(gaps.max()))

    curve = pts.copy()
    for _ in range(max(0, int(smooth))):
        curve = 0.25 * np.roll(curve, 1, axis=0) + 0.5 * curve + 0.25 * np.roll(curve, -1, axis=0)
        curve[0], curve[-1] = pts[0], pts[-1]         # smoothing must not retract the ends

    return curve, {
        "control_points": int(len(pts)),
        "bins": int(bins),
        "opening_gap_deg": round(gap_deg, 2),
        "centre": centre.tolist(),
        "height_percentile": float(height_percentile),
    }


def _face_components(faces: np.ndarray, n_verts: int) -> tuple[int, np.ndarray]:
    """(n_components, per-face label) over edge adjacency, vectorised.

    Deliberately not largest_face_component: that is a Python-dict BFS over
    every face, fine for a crown-sized selection and far too slow for a whole
    arch. Sorted edge keys plus scipy's connected_components measures ~1s at
    187k faces.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
    m = len(faces)
    if m == 0:
        return 0, np.empty(0, int)
    E = np.vstack([np.sort(faces[:, [0, 1]], axis=1),
                   np.sort(faces[:, [1, 2]], axis=1),
                   np.sort(faces[:, [0, 2]], axis=1)])
    key = E[:, 0].astype(np.int64) * int(n_verts) + E[:, 1]
    order = np.argsort(key, kind="stable")
    key = key[order]
    fid = np.tile(np.arange(m), 3)[order]
    same = np.where(key[1:] == key[:-1])[0]
    g = coo_matrix((np.ones(len(same)), (fid[same], fid[same + 1])), shape=(m, m))
    return _cc(g, directed=False)


def weld_vertices(verts: np.ndarray, faces: np.ndarray):
    """Merge vertices sharing an exact position; drop faces that collapse.

    Returns (verts, faces, welded). Moves nothing — it only recovers the
    connectivity two coincident vertices already implied, which is the same
    thing condition_mesh does on upload.

    Needed on the export path because binary STL stores POSITIONS, not indices:
    a reader welds on load whether or not we do, so any mesh validated before
    welding is not the mesh the lab receives. build_socket_cup's flat floor
    emits one vertex per rim vertex, and on a real cervical margin two rim
    points project to the same spot — which turned a base that reported itself
    watertight into one carrying non-manifold edges the moment it was re-read.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    uniq, inverse = np.unique(v, axis=0, return_inverse=True)
    if len(uniq) == len(v):
        return v, f, 0
    g = inverse.ravel()[f]
    keep = (g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 0] != g[:, 2])
    return uniq, g[keep], int(len(v) - len(uniq))


def _open_pinch_vertices(faces: np.ndarray, loop) -> tuple[np.ndarray, int]:
    """Delete the smaller fan at every boundary vertex the surface touches
    itself at. Returns (face_mask_to_keep, pinches_opened).

    A trimmed band can neck to a single vertex: two stretches of surface meet
    there at a point, every edge still manifold, and the boundary walk passes
    through it twice. That is fatal downstream — build_cast_base pairs each rim
    vertex with one floor vertex, so a rim visiting a vertex twice builds two
    spokes onto the same pair — and it is not hypothetical: a real mandibular
    scan trimmed at 9mm produces exactly one such vertex.

    DELETING the smaller fan, not duplicating the vertex. Duplication is the
    textbook repair and it is wrong HERE, for a reason only visible at the far
    end of the pipeline: the two copies sit at the same position, binary STL
    stores positions rather than indices, and every slicer welds on load — so
    the pinch comes straight back, and worse, the two floor vertices the wall
    builds from them coincide too. Measured on the real scan, the exported base
    re-parsed with 3 non-manifold edges (one at 4 faces) after reporting itself
    watertight. Deleting a fan of a few faces opens the neck instead, moves no
    vertex, and leaves nothing for a weld to rejoin.

    Only vertices repeated in `loop` are examined, so this costs a few passes
    over the face array rather than a walk over every vertex on the arch.
    """
    counts: dict[int, int] = {}
    for v in loop:
        counts[int(v)] = counts.get(int(v), 0) + 1
    repeated = [v for v, c in counts.items() if c > 1]
    keep = np.ones(len(faces), bool)
    if not repeated:
        return keep, 0

    faces = np.asarray(faces, int)
    opened = 0
    for v in repeated:
        inc = np.where(keep & (faces == v).any(axis=1))[0]
        if len(inc) < 2:
            continue
        # Two incident faces belong to the same fan when they share an edge
        # through v — i.e. they have another vertex in common.
        by_other: dict[int, list[int]] = {}
        for fi in inc:
            for o in faces[fi]:
                if int(o) != v:
                    by_other.setdefault(int(o), []).append(int(fi))
        parent = {int(fi): int(fi) for fi in inc}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for shared in by_other.values():
            for other in shared[1:]:
                ra, rb = find(shared[0]), find(other)
                if ra != rb:
                    parent[ra] = rb

        fans: dict[int, list[int]] = {}
        for fi in inc:
            fans.setdefault(find(int(fi)), []).append(int(fi))
        if len(fans) < 2:
            continue
        ordered = sorted(fans.values(), key=len, reverse=True)
        for fan in ordered[1:]:
            keep[fan] = False
        opened += 1
    return keep, opened


def _open_edge_count(faces: np.ndarray) -> int:
    """How many edges have exactly one face. Vectorised — used where only the
    count matters and walking the loops would be wasteful."""
    counts = _edge_face_counts(faces)
    return 0 if len(counts) == 0 else int((counts == 1).sum())


def _fill_interior_holes(verts: np.ndarray, faces: np.ndarray):
    """Centroid-fan every boundary loop except the longest.

    The longest loop is the cast's outer margin and must stay open — it is what
    the wall is built from. Everything else is a hole: a scan void the trim
    exposed, or a socket left by an extracted tooth.

    A fan is used rather than cap_boundary_loop for the reason already in
    cap_and_close's docstring — every edge a fan creates terminates at a brand
    new centroid vertex, so it cannot give an existing edge a third face and
    needs no O(mesh) validation.

    The half-edge set is computed ONCE, not per loop: filling one hole adds
    faces only along that hole, so the other loops' boundary directions are
    unchanged, and _boundary_half_edges is a Python pass over every face.
    """
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    loops = boundary_loops(faces)
    if not loops:
        raise ValueError("The trimmed band has no open boundary at all, so it is "
                         "already a closed surface and there is nothing to extrude.")
    loops.sort(key=len, reverse=True)

    outer = loops[0]
    if len(loops) == 1:
        return verts, faces, 0, outer

    half = _boundary_half_edges(faces)
    verts = np.asarray(verts, float).copy()
    faces = np.asarray(faces, int).copy()
    filled = 0
    for loop in loops[1:]:
        # Split first: a walk that pinches through a vertex twice would give the
        # spoke to that vertex four faces. See _split_self_touching_loop.
        for cycle in _split_self_touching_loop(loop):
            verts = np.vstack([verts, verts[cycle].mean(axis=0)])
            ci = len(verts) - 1
            fan = np.array([[cycle[i], cycle[(i + 1) % len(cycle)], ci]
                            for i in range(len(cycle))], dtype=int)
            faces = np.vstack([faces, _orient_cap_against_boundary(fan, half)])
            filled += 1
    return verts, faces, filled, outer


def distance_to_arch_curve(points: np.ndarray, arch_frame: dict,
                           curve: np.ndarray | None = None,
                           verts: np.ndarray | None = None) -> np.ndarray:
    """Distance from each point to the fitted occlusal ridge, in the arch plane.

    This is EXACTLY the quantity `trim_to_arch` compares against `margin_mm`,
    exposed so a caller can ask "would the trim keep this?" rather than
    reimplement the basis and the densification and drift from it.

    It exists because the trim can cut the cast out from under a socket rim
    and nothing downstream could see that it had. Measured on this project's
    real scan, FDI 46 (a first molar) at the shipped 7.0mm margin: 53 of its
    280 cervical rim points ended up as much as 1.2961mm OUTSIDE the finished
    cast, and `build_stage_tooth_interface` then refused
    `interface_unbuildable_wall_too_thin` with 40 unresolvable points - a
    correct refusal about geometry that had been deleted, reported as though
    the cast were too thin. At 9.0mm: 0 points outside, and the interface
    builds. The local cast thickness is 2.8573mm at both margins, so
    thickness was never the variable.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if curve is None:
        if verts is None:
            raise ValueError("distance_to_arch_curve needs `curve`, or `verts` "
                             "to fit one from.")
        curve, _ = fit_arch_curve(np.asarray(verts, float), arch_frame)
    samples = _polyline_samples(np.asarray(curve, float))
    origin, e1, e2, _ = _arch_basis(arch_frame)
    rel = pts - origin
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(samples).query(np.column_stack([rel @ e1, rel @ e2]))
    return np.asarray(dist, float)


def trim_to_arch(verts: np.ndarray, faces: np.ndarray, arch_frame: dict,
                 margin_mm=ARCH_TRIM_MARGIN_MM,
                 curve: np.ndarray | None = None):
    """Cut the scan down to a horseshoe band about the dental arch.

    DELETES FACES. MOVES NO VERTEX — CLAUDE.md rule 3.1. The returned vertex
    array is the original one with a few hole-fill centroids APPENDED, so every
    index the caller already holds keeps its meaning.

    Faces are kept when their centroid lies within margin_mm of the fitted
    occlusal ridge curve. Everything else — vestibule, floor of mouth, tongue,
    stray capture — goes.

    Two steps here are not defensive, they are required, and both were measured:

      * LARGEST COMPONENT. The face-centroid predicate leaves 1 to 4 connected
        islands depending on the margin (measured 4 at 6mm, 2 at 7mm, 3 at 9mm
        on a real scan). Stray islands would each contribute their own boundary
        loop.
      * FILL INTERIOR HOLES. "A horseshoe strip has exactly one boundary loop"
        is true of the strip and NOT true of what the predicate produces: at
        margin 7.0 the largest component came out with two loops, 2225 and 81 —
        an interior hole, not a disconnection, and no choice of margin reliably
        avoids one. Filling first makes the single-loop guarantee structural.

    Returns (verts, faces, info).
    """
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    if isinstance(margin_mm, (int, float)):
        margin_mm = float(margin_mm)
        if margin_mm <= 0:
            raise ValueError(f"margin_mm must be positive, got {margin_mm}.")
        is_scalar_margin = True
    else:
        margin_mm = margin_mm if isinstance(margin_mm, dict) else np.asarray(margin_mm, float)
        is_scalar_margin = False

    # The arch curve is a property of the CASE, not of the current extraction
    # state, so the caller may supply one fitted on the original scan. It
    # matters: fit_arch_curve reads the occlusal ridge, and by export time the
    # extracted teeth are gone — so refitting here leaves the ridge missing
    # exactly where the teeth were, and the trim then cuts away the very sockets
    # it was meant to keep. Measured on a two-tooth fixture: 351 of 423 socket
    # cup vertices discarded.
    if curve is None:
        curve, cinfo = fit_arch_curve(verts, arch_frame)
    else:
        curve = np.asarray(curve, float)
        cinfo = {"supplied": True, "control_points": int(len(curve))}
    samples = _polyline_samples(curve)
    origin, e1, e2, _ = _arch_basis(arch_frame)

    rel = verts[faces].mean(axis=1) - origin
    centroids2d = np.column_stack([rel @ e1, rel @ e2])
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(samples).query(centroids2d)
    if is_scalar_margin:
        keep = dist < margin_mm
    elif isinstance(margin_mm, dict):
        center_pt = cinfo.get("centre", np.mean(samples, axis=0))
        is_lingual = np.linalg.norm(centroids2d - center_pt, axis=1) < np.linalg.norm(samples[idx] - center_pt, axis=1)
        mb = margin_mm["buccal"][idx]
        ml = margin_mm["lingual"][idx]
        keep = dist < np.where(is_lingual, ml, mb)
    else:
        if len(margin_mm) != len(samples):
            raise ValueError("margin_mm array must match number of curve samples")
        keep = dist < margin_mm[idx]
        
    if not keep.any():
        raise ValueError(
            f"No face centroid lies within the margin of the fitted arch curve. "
            f"The occlusal plane landmarks are almost certainly wrong.")

    # Largest island only, and REPORT WHAT WENT. The previous info dict returned
    # the number of islands FOUND under the name `components`, which reads as
    # "the exported base has this many components" and was taken that way — the
    # base is and always was one solid (measured: 1 component, 190,082 faces).
    # What a reader actually needs is what the trim discarded, so that is what
    # is returned now, with face counts.
    islands_removed: list[dict] = []

    def largest_island(m):
        n_c, lab = _face_components(faces[m], len(verts))
        idx = np.where(m)[0]
        sizes = np.bincount(lab)
        biggest = int(sizes.argmax())
        out = np.zeros(len(faces), bool)
        out[idx[lab == biggest]] = True
        for k in np.argsort(sizes)[::-1]:
            if int(k) != biggest:
                islands_removed.append({"faces": int(sizes[k])})
        return out, n_c

    mask, islands_found = largest_island(keep)

    # Open any neck in the outer boundary before anything downstream sees it.
    # Deleting the smaller fan can strand a spur, so the island is re-selected
    # and the walk repeated; a couple of rounds is all a real scan needs.
    pinches = 0
    for _ in range(4):
        rim_loop = max(boundary_loops(faces[mask]), key=len)
        sub_keep, opened = _open_pinch_vertices(faces[mask], rim_loop)
        if not opened:
            break
        drop = np.where(mask)[0][~sub_keep]
        mask[drop] = False
        mask, _ = largest_island(mask)
        pinches += opened

    out_v, out_f, filled, outer = _fill_interior_holes(verts, faces[mask])

    # A closed loop of L vertices owns exactly L edges, so "the only open edges
    # left are the outer loop's" is an exact test and costs one vectorised pass
    # instead of a second boundary walk over 187k faces.
    open_e = _open_edge_count(out_f)
    if open_e != len(outer):
        raise ValueError(
            f"After filling {filled} interior hole(s) the trimmed band still has "
            f"{open_e} open edges against an outer loop of {len(outer)}. More than one "
            f"boundary loop survives, so the trim disconnected something; widen "
            f"margin_mm (currently {margin_mm:.1f}mm).")

    return out_v, out_f, {
        "margin_mm": margin_mm,
        "faces_in": int(len(faces)),
        "faces_removed": int(len(faces) - mask.sum()),
        "faces_kept": int(mask.sum()),
        "kept_fraction": round(float(mask.mean()), 4),
        # The OUTPUT is one island by construction; these say what was found and
        # what was thrown away, which is what the old `components` field was
        # mistaken for.
        "components": 1,
        "islands_found": int(islands_found),
        "islands_removed": islands_removed,
        "island_faces_removed": int(sum(d["faces"] for d in islands_removed)),
        "pinches_opened": int(pinches),
        "holes_filled": int(filled),
        "boundary_loops": 1,
        "rim_points": int(len(outer)),
        "rim_loop": np.asarray(outer, int),
        "curve": curve,
        "curve_info": cinfo,
    }


def prune_to_simple(poly2d: np.ndarray,
                    min_points: int = MIN_FLOOR_OUTLINE_POINTS):
    """Indices of a subsequence of `poly2d` that the floor can be built on.

    Extruding the rim needs its projection onto the base plane triangulated, and
    that projection is not a simple polygon: the rim runs along steep buccal and
    lingual walls, and projected along u_occ it steps over itself. Measured
    crossings on a real mandibular scan — 31 at margin 9, 47 at margin 7 — and
    every one of them is LOCAL, at index separation 2-39 out of ~2140. They are
    the boundary crossing its own recent path on a wall, not the buccal and
    lingual rails genuinely crossing.

    So the repair is local too, and that is all it is:

      1. deduplicate coincident projections;
      2. delete the shorter arc between each pair of crossing edges;
      3. hand the result to the robust ear clip.

    DECIMATION IS GONE AND THE NUMBERS ARE WHY. An earlier build reduced the
    outline to 0.5mm spacing first, which discarded 1675 of 2144 rim vertices —
    a 78% gap. That is not a triangulation aid, it is a visible defect: the wall
    bridges each rim vertex to its nearest surviving floor vertex, so four or
    five rim vertices sharing one floor vertex fan into slivers and the cast
    wall comes out striated. With decimation removed, measured on the same scan:

        margin 7.0: rim 2133 -> 2014 (gap 5.6%, 10 repairs)
        margin 9.0: rim 2144 -> 2046 (gap 4.6%,  7 repairs)

    and the ear clip completes exactly — 2012/2012 and 2044/2044, zero forced
    clips, triangulated area equal to the outline's signed area to the last
    digit. The wall is then 1:1 for 95% of the rim.

    WHAT WAS ACTUALLY BLOCKING THE EAR CLIP was step 1, not resolution. Two rim
    vertices distinct in 3D project onto one point where the margin folds; the
    duplicate stalls the clip, and an earlier version had dropped the coincidence
    check out of this loop, so the spacing ladder was papering over it. It has to
    happen anyway for the export to survive a reader's weld (see weld_vertices).

    Everything here DELETES vertices and never moves or inserts one, which is
    what keeps the floor loop a SUBSEQUENCE of the rim loop — each rim vertex
    maps to the nearest survivor at or before it, and the wall needs no
    arc-length lofting.

    Returns (survivor_indices, info).
    """
    from scipy.spatial import cKDTree

    P = np.asarray(poly2d, float)
    n0 = len(P)
    if n0 < 3:
        raise ValueError(f"A polygon needs at least 3 vertices, got {n0}.")

    keep = np.arange(n0)
    min_keep = max(3, int(min_points))

    # 1. Coincident projections. Iterated, because dropping one member of a pair
    #    can leave another pair behind it.
    coincident = 0
    while len(keep) > min_keep:
        pairs = cKDTree(P[keep]).query_pairs(COINCIDENT_PROJECTION_MM,
                                             output_type="ndarray")
        if not len(pairs):
            break
        drop = np.unique(pairs[:, 1])
        coincident += len(drop)
        keep = np.delete(keep, drop)

    def thinness(Q):
        """2*area / longest_side^2 for each corner's ear triangle."""
        A, B, C = np.roll(Q, 1, axis=0), Q, np.roll(Q, -1, axis=0)
        area2 = np.abs((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1])
                       - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
        longest = np.maximum.reduce([np.linalg.norm(B - A, axis=1),
                                     np.linalg.norm(C - B, axis=1),
                                     np.linalg.norm(C - A, axis=1)])
        return np.where(longest < 1e-12, 0.0,
                        area2 / np.where(longest < 1e-12, 1.0, longest) ** 2)

    # 2. Local crossing repair, interleaved with dropping degenerate corners.
    #    The needle drop is not cosmetic: without it a regular-grid rim leaves
    #    the clipper forcing 62 corners and losing 23% of the floor's area.
    repairs = needles = 0
    worst_arc = 0.0
    for _ in range(6 * n0):
        if len(keep) < 4 or len(keep) < min_keep:
            break

        # Non-adjacent corners only, so a whole run is not removed in one pass —
        # dropping both neighbours of a kept vertex moves the outline far more
        # than dropping either.
        bad = np.where(thinness(P[keep]) < MIN_EAR_THINNESS)[0]
        if len(bad):
            sel, last = [], -9
            for b in bad:
                if b - last > 1:
                    sel.append(int(b))
                    last = b
            if len(sel) > len(keep) - 3:
                sel = sel[:max(0, len(keep) - 3)]
            if sel:
                keep = np.delete(keep, sel)
                needles += len(sel)
                continue

        pairs = _crossing_pairs(P[keep])
        if not len(pairs):
            break
        i, j = int(pairs[0][0]), int(pairs[0][1])
        m = len(keep)
        inner, outer = j - i, m - (j - i)
        frac = min(inner, outer) / m
        worst_arc = max(worst_arc, frac)
        if frac > MAX_PRUNE_ARC_FRACTION:
            raise ValueError(
                f"Repairing the projected rim would delete {100 * frac:.1f}% of it "
                f"in one arc, past the {100 * MAX_PRUNE_ARC_FRACTION:.0f}% limit, "
                f"leaving a chord across the gap. Every crossing on a real cast "
                f"margin is local — the boundary stepping over itself on a steep "
                f"wall — so a crossing this long means the buccal and lingual rails "
                f"genuinely cross in the occlusal view, and bridging it would web "
                f"over the arch opening.")
        if inner <= outer:
            keep = np.delete(keep, np.arange(i + 1, j + 1))
        else:
            keep = np.delete(keep, np.r_[np.arange(j + 1, m), np.arange(0, i + 1)])
        repairs += 1

    removed = n0 - len(keep)
    if len(keep) < min_keep:
        raise ValueError(
            f"Repairing the projected rim collapsed it to {len(keep)} points from {n0}. "
            f"The boundary does not enclose a cast outline.")

    # 3. THE INVARIANT IS AREA, NOT COUNT. ear_clip_polygon_robust always returns
    #    n-2 triangles, so a count check against it is vacuous — it would pass on
    #    a triangulation that folds over itself. Measured on a regular-grid rim
    #    before the needle drop above existed: 544 of 544 triangles returned and
    #    23% of the floor's area missing. Comparing the triangulated area with
    #    the outline's own signed area is what actually catches that, and on
    #    every case that passes it matches to ~1e-16 relative.
    tris, forced = ear_clip_polygon_robust(P[keep])
    if len(tris) != len(keep) - 2:
        raise ValueError(
            f"The repaired outline triangulates to {len(tris)} triangles where "
            f"{len(keep) - 2} are owed, so the floor would have holes.")

    Pk = P[keep]
    a_, b_, c_ = Pk[tris[:, 0]], Pk[tris[:, 1]], Pk[tris[:, 2]]
    tri_area = 0.5 * np.abs((b_[:, 0] - a_[:, 0]) * (c_[:, 1] - a_[:, 1])
                            - (b_[:, 1] - a_[:, 1]) * (c_[:, 0] - a_[:, 0])).sum()
    outline_area = abs(_signed_area_2d(Pk))
    if outline_area <= 0 or abs(tri_area - outline_area) > 1e-6 * outline_area:
        raise ValueError(
            f"The floor triangulates to {tri_area:.4f} mm2 against an outline enclosing "
            f"{outline_area:.4f} mm2 ({100 * abs(tri_area - outline_area) / max(outline_area, 1e-12):.1f}% "
            f"out). The triangles overlap or fall outside the outline, which is a hole in "
            f"the bottom of a printed cast however many of them there are.")

    return keep, {"input_points": int(n0), "kept_points": int(len(keep)),
                  "pruned_vertices": int(removed),
                  "pruned_fraction": round(removed / n0, 4),
                  "gap_fraction": round(removed / n0, 4),
                  "coincident_removed": int(coincident),
                  "needles_removed": int(needles),
                  "repairs": int(repairs),
                  "forced_clips": int(forced),
                  "area_error": float(abs(tri_area - outline_area) / outline_area),
                  "worst_arc_fraction": round(worst_arc, 4)}


def _winding_is_consistent(faces: np.ndarray) -> bool:
    """True iff every directed edge appears exactly once across all triangles.

    For a closed mesh that is precisely consistent orientation: the two faces on
    an edge must traverse it in opposite directions. Worth asserting separately
    from manifold_report, because edge COUNTS are orientation-blind — a base can
    report zero open and zero non-manifold edges while half its normals point
    inward, and a boolean engine handed that returns a plausible wrong solid.
    """
    f = np.asarray(faces, int)
    if len(f) == 0:
        return True
    E = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = int(f.max()) + 1
    fwd = np.sort(E[:, 0].astype(np.int64) * n + E[:, 1])
    rev = np.sort(E[:, 1].astype(np.int64) * n + E[:, 0])
    return bool((np.diff(fwd) > 0).all() and np.array_equal(fwd, rev))


def build_cast_base(verts: np.ndarray, faces: np.ndarray, arch_frame: dict,
                    base_thickness_mm: float = CAST_BASE_THICKNESS_MM,
                    rim: np.ndarray | None = None):
    """Extrude a trimmed horseshoe band into a closed, printable cast base.

    Input must be the output of trim_to_arch: one boundary loop, no interior
    holes. Returns (verts, faces, info) for a mesh with ZERO open edges and ZERO
    non-manifold edges — asserted directly here, never via cap_and_close.

    Construction:

    1. Project the rim into the occlusal plane and prune it to a simple outline
       (see prune_to_simple — this is load-bearing, not tidying).
    2. Put the floor plane base_thickness_mm apical of the LOWEST point of the
       kept surface. The brief said the lowest RIM point; on a scan the apical
       extreme usually is on the rim, but nothing guarantees it, and a low patch
       of surface would then poke through the bottom of the cast.
    3. Wall: for rim edge (a,b) with floor images (a',b'), emit (a,b,b') and
       (a,b',a'); when b was pruned away a' == b' and the quad degenerates to
       the single triangle (a,b,a'). Because the floor loop is a SUBSEQUENCE of
       the rim, this is exact — no resampling, no arc-length correspondence.
       Zero open edges then follows from combinatorics alone: every rim edge
       gets one top face and one wall face, every floor edge one floor face and
       one wall face, every diagonal and every vertical edge exactly two wall
       faces.
    4. Orientation: the wall is wound against the trimmed surface's own boundary
       half-edges, the floor is wound to agree with the wall, and the whole mesh
       is flipped ONCE if the signed volume comes out negative. Never per
       triangle — the walls are near parallel to u_occ, so the sign of their dot
       against it is noise. Exactly the lesson build_socket_cup records.

    THE MEMBRANE CANNOT COME BACK. The floor is the ear-clipping of the rim's
    own horseshoe outline, so it covers the band and nothing else; measured on a
    real scan, all 2051 floor triangles have centroids within 9.02mm of the
    ridge against a 9.0mm trim margin. cap_and_close, by contrast, fanned the
    same horseshoe into 2104 triangles straight across the tongue space.
    """
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)

    # `rim` is an optimisation, never a shortcut past the check: trim_to_arch
    # already walked the boundary and can hand its loop over, but the
    # single-loop precondition is still verified either way — an L-vertex closed
    # loop owns exactly L edges, so the open-edge count settles it outright.
    if rim is None:
        loops = boundary_loops(faces)
        if len(loops) != 1:
            raise ValueError(
                f"build_cast_base needs exactly one boundary loop and got {len(loops)} "
                f"(sizes {sorted((len(L) for L in loops), reverse=True)[:5]}). Run "
                f"trim_to_arch first — it fills interior holes and guarantees this.")
        rim = np.asarray(loops[0], int)
    else:
        rim = np.asarray(rim, int)
        open_e = _open_edge_count(faces)
        if open_e != len(rim):
            raise ValueError(
                f"build_cast_base was handed a rim of {len(rim)} vertices but the "
                f"surface has {open_e} open edges, so it has more than one boundary "
                f"loop. Run trim_to_arch first — it fills interior holes.")

    # The wall pairs each rim vertex with one floor vertex, so a rim that
    # revisits a vertex would build two spokes onto the same pair and give that
    # edge four faces — the same trap _split_self_touching_loop describes for
    # the hole fans. There is no correct wall for a pinched rim, so refuse.
    if len(np.unique(rim)) != len(rim):
        dup = len(rim) - len(np.unique(rim))
        raise ValueError(
            f"The trimmed rim passes through {dup} vertex/vertices twice, so the cast "
            f"surface pinches to a point on its own margin and there is no single wall "
            f"to extrude. Adjust the trim margin so the band does not neck.")

    origin, e1, e2, u_occ = _arch_basis(arch_frame)
    rel = verts - origin
    height = rel @ u_occ
    Q = np.column_stack([rel[rim] @ e1, rel[rim] @ e2])

    surv, pinfo = prune_to_simple(Q)
    floor2d = Q[surv]
    # Shared with build_socket_cup so the two cannot drift: it normalises the
    # floor to traverse boundary edge k as (k -> k+1) whichever way the rim
    # wound, which is what lets the wall below meet it head-on.
    tris, forced_clips = _floor_triangulation(floor2d)
    if len(tris) != len(floor2d) - 2:
        raise ValueError(
            f"Ear clipping the floor outline returned {len(tris)} triangles where "
            f"{len(floor2d) - 2} are owed, so it bailed on a polygon it could not "
            f"triangulate and the floor would have holes.")

    used = np.unique(faces)
    plane_h = float(height[used].min()) - float(base_thickness_mm)
    floor_xyz = (origin + np.outer(floor2d[:, 0], e1) + np.outer(floor2d[:, 1], e2)
                 + u_occ * plane_h)

    n_v = len(verts)
    # Each rim vertex takes the nearest survivor at or before it; anything before
    # the first survivor wraps to the last, which is the cyclic predecessor.
    image = np.searchsorted(surv, np.arange(len(rim)), side="right") - 1
    image[image < 0] = len(surv) - 1

    m = len(rim)
    wall = []
    for i in range(m):
        a, b = int(rim[i]), int(rim[(i + 1) % m])
        ai, bi = n_v + int(image[i]), n_v + int(image[(i + 1) % m])
        if ai == bi:
            wall.append([a, b, ai])
        else:
            wall.append([a, b, bi])
            wall.append([a, bi, ai])
    wall = np.asarray(wall, int)
    floor = np.asarray(tris, int) + n_v

    # Seam check: if the rim loop runs the same way the surface traverses its own
    # boundary, the wall as built duplicates that direction instead of opposing
    # it. Flip the whole new block — wall and floor together — so the block stays
    # internally consistent while the seam becomes correct.
    half = _boundary_half_edges(faces)
    if (int(rim[0]), int(rim[1])) in half:
        wall = wall[:, ::-1]
        floor = floor[:, ::-1]

    out_v = np.vstack([verts, floor_xyz])
    out_f = np.vstack([faces, wall, floor])

    if signed_volume(out_v, out_f) < 0:
        out_f = out_f[:, ::-1]

    health = manifold_report(out_f)
    if health["open_edges"] or health["nonmanifold_edges"]:
        raise ValueError(
            f"The extruded cast base did not close: {health['open_edges']} open edge(s) "
            f"and {health['nonmanifold_edges']} non-manifold edge(s) of "
            f"{health['total_edges']}. The wall is closed by construction, so this means "
            f"the input band was not a clean single-loop surface.")
    if not _winding_is_consistent(out_f):
        raise ValueError(
            "The extruded cast base closed but its winding is inconsistent, so some "
            "normals point into the solid. A boolean engine handed this returns a "
            "plausible wrong result rather than an error.")

    # ONE SOLID, asserted rather than assumed. A cast with a second component is
    # a fragment floating inside the arch, and it would be fused into every
    # staged model downstream. trim_to_arch keeps the largest island, so this can
    # only fire on a bug — which is exactly when an assertion earns its keep.
    n_comp, _lab = _face_components(out_f, len(out_v))
    if n_comp != 1:
        raise ValueError(
            f"The cast base is {n_comp} disconnected solids, not one. A fragment is "
            f"floating free of the arch.")

    volume = signed_volume(out_v, out_f)
    if volume <= 0:
        raise ValueError(f"The cast base encloses {volume:.1f} mm3, which is not a solid.")

    return out_v, out_f, {
        "base_thickness_mm": float(base_thickness_mm),
        # NOT rounded: a test asserts the floor is planar to 1e-9 against this
        # number, and rounding it to 4 decimals would make the floor look 4.5e-5
        # out of plane when it is exact to 1e-15.
        "base_plane_offset_mm": float(plane_h),
        "lowest_surface_mm": float(height[used].min()),
        "wall_tris": int(len(wall)),
        "floor_tris": int(len(floor)),
        "rim_points": int(len(rim)),
        "floor_points": int(len(floor2d)),
        # rim -> floor loss. Was 78% while the outline was decimated, which
        # striated the wall; local repair alone measures 4.6% on a real scan.
        "gap_fraction": round(1.0 - len(floor2d) / max(len(rim), 1), 4),
        "pruned_vertices": pinfo["pruned_vertices"],
        "repairs": pinfo["repairs"],
        "forced_clips": int(forced_clips),
        "components": int(n_comp),
        "prune": pinfo,
        "open_edges": int(health["open_edges"]),
        "nonmanifold_edges": int(health["nonmanifold_edges"]),
        "total_edges": int(health["total_edges"]),
        "watertight": True,
        "winding_consistent": True,
        "volume_mm3": round(volume, 3),
        "faces": int(len(out_f)),
        "vertices": int(len(out_v)),
    }


def offset_along_normals(verts: np.ndarray, faces: np.ndarray,
                         distance_mm: float) -> np.ndarray:
    return np.asarray(verts, dtype=float) + vertex_normals(verts, faces) * distance_mm

def export_nested_pair(crown_verts, crown_faces, base_verts, base_faces,
                       out_dir: str, arch_name: str = "arch",
                       clearance_mm: float = 0.15, stage: int | None = None):
    import os
    os.makedirs(out_dir, exist_ok=True)

    crown_ok = is_edge_manifold_closed(crown_faces)
    base_ok = is_edge_manifold_closed(base_faces)
    if not (crown_ok and base_ok):
        raise ValueError("Refusing to export: meshes not watertight.")

    crown_out = np.asarray(crown_verts, dtype=float)
    if clearance_mm > 0:
        crown_out = offset_along_normals(crown_out, crown_faces, -clearance_mm)

    tag = f"_Stage{stage:02d}" if stage is not None else ""
    tooth_path = os.path.join(out_dir, f"{arch_name}{tag}_Tooth.stl")
    base_path = os.path.join(out_dir, f"{arch_name}{tag}_Base_Socket.stl")

    n_tooth = write_binary_stl(crown_out, crown_faces, tooth_path)
    n_base = write_binary_stl(base_verts, base_faces, base_path)

    return dict(tooth_stl=tooth_path, base_stl=base_path, tooth_faces=n_tooth, base_faces=n_base, clearance_mm=clearance_mm)

MIN_CROWN_VOLUME_MM3 = 20.0   # below this a "crown" is not a tooth. Measured on
                              # auto-segmented crowns from a real mandibular
                              # scan: a genuine one is 63-136mm3, while the
                              # rejects came in at 8.8mm3 and 0.0mm3 — the
                              # latter a 0.2 x 0.2 x 0.7mm speck that is
                              # watertight, single-bodied and Euler-2, and would
                              # sail through every topological test there is.


def crown_is_printable(verts: np.ndarray, faces: np.ndarray) -> dict:
    """Is this crown a solid a boolean engine can work with, or a shell?

    WHY TOPOLOGY ALONE IS NOT THE ANSWER, measured on four auto-cut crowns:

        FDI  watertight   chi  bodies   volume     fuses into the cast?
         46      yes        2     1     136.3        yes
         34      yes        2     1      63.3        NO — fractures into 3
         36      yes       -2     1       8.8        no
         32      yes        2     1       0.0        no  (a 0.2mm speck)

    Every crown /cut produces is ALREADY watertight — the endpoint asserts
    is_edge_manifold_closed and refuses 422 otherwise — so a gate on
    watertightness is a no-op. Euler characteristic catches FDI 36 and nothing
    else. Volume catches FDI 32. Neither catches FDI 34, which passes every
    topological test and still fractures.

    So this function is a SCREEN, not the gate. It exists to produce a specific
    diagnosis cheaply; the decisive test is whether the tooth's manufacturing
    solid fuses into one positive-volume body, which only a boolean can answer.
    See _screen_crowns_for_manufacturing in api_core.

    chi = V - E + F, and chi == 2 is exactly genus 0 for a closed orientable
    surface (chi = 2 - 2g).
    """
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    health = manifold_report(faces)
    n_v = int(len(np.unique(faces))) if len(faces) else 0
    n_e = int(len(_edge_face_counts(faces)))
    n_f = int(len(faces))
    chi = n_v - n_e + n_f
    bodies = int(_face_components(faces, len(verts))[0]) if len(faces) else 0
    vol = abs(signed_volume(verts, faces)) if len(faces) else 0.0

    reason = None
    if not health["watertight"]:
        reason = (f"the crown is not closed ({health['open_edges']} open, "
                  f"{health['nonmanifold_edges']} non-manifold edges)")
    elif bodies != 1:
        reason = f"the crown is {bodies} disconnected pieces, not one solid"
    elif chi != 2:
        reason = (f"the crown has Euler characteristic {chi} (genus {(2 - chi) // 2}), "
                  f"so it is a shell with tunnels through it rather than a solid")
    elif vol < MIN_CROWN_VOLUME_MM3:
        reason = (f"the crown encloses only {vol:.1f}mm3, under the {MIN_CROWN_VOLUME_MM3}mm3 "
                  f"floor — that is a sliver of surface, not a tooth")

    return {"watertight": bool(health["watertight"]),
            "euler_characteristic": chi,
            "genus": (2 - chi) // 2,
            "bodies": bodies,
            "volume_mm3": round(float(vol), 3),
            "ok": reason is None,
            "reason": reason}


def build_antagonist_index(opp_verts):
    """The KD-tree for an opposing arch, built once and reused.

    Measured at 94,848 vertices: the build is 43ms and a 2000-vertex query is
    35ms. A 31-stage, 14-tooth export asks 434 questions of the same arch, so
    rebuilding per call would spend 19 SECONDS on a tree that never changes.
    """
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(opp_verts, float))


def check_occlusal_collision(crown_verts, opp_verts, opp_normals,
                             threshold_mm: float = 0.1, index=None) -> dict:
    """Does this crown drive into the opposing arch?

    Signed nearest-vertex distance: for each crown vertex find the closest
    vertex on the antagonist and sign the offset against that vertex's outward
    normal. Negative means the crown point sits BEHIND the opposing surface,
    i.e. inside it.

    APPROXIMATE BY CONSTRUCTION, and that is deliberate — this is a non-blocking
    warning, never a gate. Two things it does not do:

      * it measures to the nearest VERTEX, not the nearest point on the surface,
        so on a coarse mesh it under-reports penetration in the middle of a
        large triangle. Intraoral triangles are sub-millimetre, so the error is
        well under the 0.1mm threshold it is asked about;
      * it cannot distinguish a genuine intersection from two surfaces that
        merely touch.

    A true SDF would need the opposing arch closed, and an intraoral scan is an
    open shell (see CLAUDE.md section 9) — so there is no inside to test against
    without building a cast base for the antagonist too. That is a far larger
    job than a proximity warning warrants.

    Pass `index` from build_antagonist_index to reuse the KD-tree across calls;
    without it one is built per call, which is 43ms of the ~89ms a 2000-vertex
    crown costs against a 95k-vertex arch.
    """
    crown = np.asarray(crown_verts, float)
    opp = np.asarray(opp_verts, float)
    if len(crown) == 0 or len(opp) == 0:
        return {"max_penetration_mm": 0.0, "points_penetrating": 0, "collides": False,
                "threshold_mm": float(threshold_mm)}

    tree = index if index is not None else build_antagonist_index(opp)
    # workers=-1: the query is embarrassingly parallel and a staging export asks
    # it hundreds of times. Measured on 8 cores against a 94,848-vertex arch —
    # 2000 crown vertices go from 45ms to 15ms, 8000 from 161ms to 38ms, which
    # takes a 14-crown 31-stage run from ~15s of collision checking to ~5s.
    dist, idx = tree.query(crown, workers=-1)
    normals = np.asarray(opp_normals, float)[idx]
    # Positive when the crown point is on the outside of the opposing surface.
    signed = np.einsum("ij,ij->i", crown - opp[idx], normals)
    penetration = np.where(signed < 0, dist, 0.0)
    worst = float(penetration.max()) if len(penetration) else 0.0
    n_in = int((penetration > threshold_mm).sum())
    return {"max_penetration_mm": round(worst, 4),
            "points_penetrating": n_in,
            "collides": bool(worst > threshold_mm),
            "threshold_mm": float(threshold_mm)}


def points_inside_closed_mesh(points, mesh_verts, mesh_faces, shrink=0.0):
    from scipy.spatial import cKDTree
    tri = mesh_verts[mesh_faces]
    centroids = tri.mean(axis=1)
    normals = face_normals(mesh_verts, mesh_faces)

    tree = cKDTree(centroids)
    _, idx = tree.query(np.asarray(points, dtype=float))
    delta = np.asarray(points, dtype=float) - centroids[idx]
    signed = np.einsum('ij,ij->i', delta, normals[idx])
    return signed < -shrink

def carve_socket(base_verts, base_faces, tool_verts, tool_faces,
                 clearance_mm: float = 0.15, margin_mm: float = 2.0):
    base_verts = np.asarray(base_verts, dtype=float)
    tool = np.asarray(tool_verts, dtype=float)
    if clearance_mm > 0:
        tool = offset_along_normals(tool, tool_faces, clearance_mm)

    lo = tool.min(axis=0) - margin_mm
    hi = tool.max(axis=0) + margin_mm
    near = np.all((base_verts >= lo) & (base_verts <= hi), axis=1)
    if not np.any(near):
        return base_verts, base_faces, dict(faces_removed=0, carved=False)

    inside = np.zeros(len(base_verts), dtype=bool)
    idx = np.where(near)[0]
    inside[idx] = points_inside_closed_mesh(base_verts[idx], tool, tool_faces)

    drop = inside[base_faces].any(axis=1)
    if not np.any(drop):
        return base_verts, base_faces, dict(faces_removed=0, carved=False)

    kept = base_faces[~drop]
    used = np.unique(kept)
    remap = -np.ones(len(base_verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    out_v, out_f = base_verts[used].copy(), remap[kept]

    out_v, out_f = cap_and_close(out_v, out_f)
    return out_v, out_f, dict(faces_removed=int(drop.sum()), carved=True,
                              watertight=bool(is_edge_manifold_closed(out_f)))

def _depth_behind_patch(points, surf_v, surf_f):
    """Per point: (depth BEHIND an oriented surface patch, unsigned distance).

    EXACT POINT-TO-TRIANGLE (Open3D BVH), never nearest-vertex (CLAUDE.md
    lessons). A point is behind the patch when (p - q) . n < 0, with q its
    closest point and n that triangle's outward normal - the scan's own
    winding, which is outward on the closed cast.

    AN OPEN PATCH HAS NO INSIDE PAST ITS EDGE. A tooth's surface patch ends at
    its cervical line and at the contact the scanner never saw; a point whose
    closest point lies ON that boundary is beyond what was observed, and the
    sign there says nothing. Such points get depth 0, never a guess.

    Returns (depth >= 0, distance >= 0), both float64; raises on Open3D
    failure so the caller can record it and return None - never 0.0.
    """
    import open3d as o3d
    P = np.atleast_2d(np.asarray(points, float))
    V = np.asarray(surf_v, float)
    Fc = np.asarray(surf_f, np.int64)
    if not len(P):
        return np.zeros(0), np.zeros(0)
    if not len(Fc):
        raise ValueError("the surface patch has no faces")
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)),
                     o3d.core.Tensor(Fc.astype(np.uint32)))
    ans = sc.compute_closest_points(o3d.core.Tensor(P.astype(np.float32)))
    tri = ans["primitive_ids"].numpy().astype(np.int64)
    uv = ans["primitive_uvs"].numpy().astype(float)
    # q from the float64 triangle, not Open3D's float32 point, so a point ON
    # the surface measures ~1e-15, not ~1e-7.
    w = np.column_stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]])
    T = V[Fc[tri]]
    q = np.einsum("ij,ijk->ik", w, T)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    d = P - q
    signed = np.einsum("ij,ij->i", d, n)
    dist = np.linalg.norm(d, axis=1)

    # Boundary of the patch: edges used once, and their vertices.
    E = np.sort(np.vstack([Fc[:, [1, 2]], Fc[:, [2, 0]], Fc[:, [0, 1]]]), axis=1)
    key = E[:, 0] * (int(Fc.max()) + 1) + E[:, 1]
    uniq, cnt = np.unique(key, return_counts=True)
    open_key = set(uniq[cnt == 1].tolist())
    bverts = set(E[np.isin(key, uniq[cnt == 1])].ravel().tolist())
    n_f = len(Fc)
    # Edge i of a triangle is the one OPPOSITE vertex i: (b,c), (c,a), (a,b).
    edge_open = np.isin(key, list(open_key)).reshape(3, n_f).T
    eps = 1e-5
    on_open_edge = ((w < eps) & edge_open[tri]).any(axis=1)
    at_vertex = (w > 1.0 - eps)
    vid = Fc[tri]
    on_open_vertex = (at_vertex & np.isin(vid, list(bverts))).any(axis=1)
    beyond = on_open_edge | on_open_vertex
    depth = np.where((signed < 0.0) & ~beyond, -signed, 0.0)
    return depth, dist


def measure_interproximal_penetration(
    crown_verts_t0, crown_verts_t1, crown_faces,
    base_verts, base_faces, mesiodistal_axis,
    threshold_mm: float = 0.05, contact_mm: float = 0.30,
    socket_exclusion_mm: float = 1.5, measure_penetration: bool = False,
):
    """Interproximal CLOSURE and, when asked, true PENETRATION.

    WHAT THE ORIGINAL KEYS MEASURE - AND WHAT THEY DO NOT. `max_closure_mm` is
    how much the vertex-to-vertex gap on the mesial / distal third SHRANK
    (g0 - g1). It is a closure: a crown moved 1.0 mm into a 1.5 mm space
    reports 1.0 mm while touching nothing, and a crown driven INTO its
    neighbour cannot report more than the gap it started with, because an
    unsigned distance cannot go below zero. Every existing caller uses it as
    a closure, and it is unchanged.

    `measure_penetration=True` (Task 2 step 5) adds what the name promises:
    how far the MOVED crown lies behind the neighbour's enamel surface -
    signed, exact point-to-triangle (`_depth_behind_patch`), measured on every
    crown vertex against `base_verts`/`base_faces` as given (pass the
    neighbour's own patch). No socket exclusion applies to it. Keys:

      penetration_mm      deepest moved-crown vertex behind the surface
      penetration_t0_mm   the same at T0 - what the scan already showed
      penetration_vertices  moved-crown vertices deeper than `threshold_mm`
      penetration_at      where the deepest one is
      gap_t0_mm / gap_mm  exact minimum distance, crown to surface
      penetration_failure None, or why it could not be measured - in which
                          case every value above is None, never 0.0.
    """
    from scipy.spatial import cKDTree
    extra = {}
    if measure_penetration:
        try:
            d0, g0v = _depth_behind_patch(crown_verts_t0, base_verts, base_faces)
            d1, g1v = _depth_behind_patch(crown_verts_t1, base_verts, base_faces)
            if not (np.isfinite(d0).all() and np.isfinite(d1).all()
                    and len(d1)):
                raise ValueError("non-finite or empty penetration measurement")
            k = int(np.argmax(d1))
            extra = dict(
                penetration_mm=float(d1.max()),
                penetration_t0_mm=float(d0.max()),
                penetration_vertices=int((d1 > threshold_mm).sum()),
                penetration_at=[float(x) for x in
                                np.atleast_2d(np.asarray(crown_verts_t1, float))[k]],
                gap_t0_mm=float(g0v.min()), gap_mm=float(g1v.min()),
                penetration_method=("signed exact point-to-triangle (Open3D), "
                                    "outward = the scan's winding; points "
                                    "beyond the patch boundary excluded"),
                penetration_failure=None)
        except Exception as e:                            # noqa: BLE001
            extra = dict(penetration_mm=None, penetration_t0_mm=None,
                         penetration_vertices=None, penetration_at=None,
                         gap_t0_mm=None, gap_mm=None,
                         penetration_method=None,
                         penetration_failure=f"{type(e).__name__}: {e}")
    axis = np.asarray(mesiodistal_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t0 = np.asarray(crown_verts_t0, dtype=float)
    t1 = np.asarray(crown_verts_t1, dtype=float)

    proj = t0 @ axis
    lo, hi = np.percentile(proj, 33), np.percentile(proj, 67)
    mesial = proj <= lo
    distal = proj >= hi

    base = np.asarray(base_verts, dtype=float)
    socket_tree = cKDTree(t0)
    far_from_socket = socket_tree.query(base)[0] > socket_exclusion_mm
    if not np.any(far_from_socket):
        return dict(max_closure_mm=0.0, min_clearance_mm=None, over_threshold=False,
                    noise_floor_mm=threshold_mm, contact_threshold_mm=contact_mm,
                    threshold_mm=threshold_mm, contact_mm=contact_mm, per_side={},
                    **extra)
    tree = cKDTree(base[far_from_socket])
    out = {}
    worst = 0.0
    for name, sel in (("mesial", mesial), ("distal", distal)):
        if not np.any(sel):
            continue
        g0 = float(tree.query(t0[sel])[0].min())
        g1 = float(tree.query(t1[sel])[0].min())
        closure = max(0.0, g0 - g1)
        # `threshold_mm` is the NOISE FLOOR, and until now it was carried in the
        # payload and compared against nothing at all. Scanners resolve to
        # 20-50 microns, so a 0.001mm "closure" is the mesh, not the movement,
        # and reporting it as a real closure invites someone to act on it.
        if closure < threshold_mm:
            closure = 0.0
        worst = max(worst, closure)
        out[name] = dict(clearance_before_mm=round(g0, 4), clearance_after_mm=round(g1, 4),
                         closed_by_mm=round(closure, 4), in_contact=bool(g1 <= contact_mm))

    tightest = min((d["clearance_after_mm"] for d in out.values() if "clearance_after_mm" in d), default=None)
    return dict(max_closure_mm=round(worst, 4), min_clearance_mm=tightest,
                # NAMED HONESTLY. This flag is computed from contact_mm (0.30),
                # NOT from threshold_mm (0.05) - a reader seeing `threshold_mm`
                # beside `over_threshold` reasonably assumes the flag means
                # 0.05mm, and it never has. Both constants are now in the
                # payload under names that say which is which. The VALUE of
                # over_threshold is unchanged: export_planned_setup and the
                # client status line both gate on it at 0.30mm.
                over_threshold=bool(tightest is not None and tightest <= contact_mm),
                contact_threshold_mm=contact_mm,
                noise_floor_mm=threshold_mm,
                threshold_mm=threshold_mm, contact_mm=contact_mm, per_side=out,
                **extra)

def sweep_interproximal_stages(
    crown_verts_t0, crown_faces, base_verts, base_faces, mesiodistal_axis,
    matrices, threshold_mm: float = 0.05, contact_mm: float = 0.30,
    socket_exclusion_mm: float = 1.5, warn_mm: float = 0.5,
):
    """Interproximal closure at EVERY stage, not just the endpoint.

    WHY THE ENDPOINT IS THE WRONG PLACE TO MEASURE. A tooth that is clear at T0
    and clear at the planned setup can still close an embrasure to nothing
    partway through — a rotation that swings a contact point past its neighbour
    before bringing it back is the ordinary case, not a contrived one. The
    clinician approves the scrub and the lab prints every stage, so the middle
    is exactly where this has to be checked.

    THE KD-TREES ARE BUILT ONCE. This is the same lesson the antagonist check
    already learned (CLAUDE.md §11): rebuilding the tree per call was 43ms of
    the 89ms a 2000-vertex crown cost, and a 31-stage sweep asks the same
    question of a base that never changes. Here the base tree and the socket
    exclusion both depend only on T0, so both hoist out of the loop entirely.

    `matrices` is one 4x4 per stage, in order. Returns one row per stage plus a
    summary. `state` is YELLOW above `warn_mm` and GREEN otherwise — never RED:
    closing an embrasure is frequently the INTENT of the plan, and this software
    measures a gap, it does not prescribe enamel reduction.

    The measurement is a vertex-to-vertex minimum, which OVERESTIMATES the true
    clearance: on a triangulated surface the closest points generally lie inside
    faces rather than at vertices. Every row says so.
    """
    from scipy.spatial import cKDTree

    axis = np.asarray(mesiodistal_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t0 = np.asarray(crown_verts_t0, dtype=float)
    base = np.asarray(base_verts, dtype=float)

    proj = t0 @ axis
    lo, hi = np.percentile(proj, 33), np.percentile(proj, 67)
    sides = (("mesial", proj <= lo), ("distal", proj >= hi))

    # Both of these depend only on T0, so they are computed once for the whole
    # sweep instead of once per stage.
    socket_tree = cKDTree(t0)
    far_from_socket = socket_tree.query(base)[0] > socket_exclusion_mm
    if not np.any(far_from_socket):
        return {"stages": [], "worst_closure_mm": 0.0, "stages_yellow": [],
                "threshold_mm": warn_mm,
                "detail": "No base geometry outside the socket to measure against."}
    tree = cKDTree(base[far_from_socket])

    baseline = {name: float(tree.query(t0[sel])[0].min())
                for name, sel in sides if np.any(sel)}

    rows, worst, yellow = [], 0.0, []
    for k, M in enumerate(matrices, start=1):
        moved = apply_matrix(t0, np.asarray(M, float))
        closure, per_side = 0.0, {}
        for name, sel in sides:
            if not np.any(sel):
                continue
            g1 = float(tree.query(moved[sel], workers=-1)[0].min())
            c = max(0.0, baseline[name] - g1)
            # The noise floor, applied here for the same reason it is applied in
            # measure_interproximal_penetration: scanners resolve to 20-50
            # microns, so a 0.001mm "closure" is the mesh, not the movement.
            if c < threshold_mm:
                c = 0.0
            closure = max(closure, c)
            per_side[name] = {"clearance_before_mm": round(baseline[name], 4),
                              "clearance_after_mm": round(g1, 4),
                              "closed_by_mm": round(c, 4),
                              "in_contact": bool(g1 <= contact_mm)}
        tightest = min((d["clearance_after_mm"] for d in per_side.values()),
                       default=None)
        state = "YELLOW" if closure > warn_mm else "GREEN"
        if state == "YELLOW":
            yellow.append(k)
        worst = max(worst, closure)
        rows.append({"stage": k, "max_closure_mm": round(closure, 4),
                     "min_clearance_mm": tightest, "state": state,
                     "over_threshold": bool(tightest is not None
                                            and tightest <= contact_mm),
                     "per_side": per_side})

    return {"stages": rows, "worst_closure_mm": round(worst, 4),
            "stages_yellow": yellow, "threshold_mm": warn_mm,
            "contact_threshold_mm": contact_mm, "noise_floor_mm": threshold_mm,
            "limitation": ("A vertex-to-vertex minimum OVERESTIMATES the true "
                           "clearance. This measures a gap closure and does not "
                           "prescribe enamel reduction.")}


def export_planned_setup(
    crown_verts, crown_faces, base_verts, base_faces,
    mesiodistal_axis, out_dir: str, arch_name: str = "arch",
    crown_verts_t0=None, clearance_mm: float = 0.15, penetration_threshold_mm: float = 0.05,
    stage: int | None = None, authorise_ipr: bool = False,
):
    import os
    os.makedirs(out_dir, exist_ok=True)
    t0 = crown_verts if crown_verts_t0 is None else crown_verts_t0
    pen = measure_interproximal_penetration(
        t0, crown_verts, crown_faces, base_verts, base_faces,
        mesiodistal_axis, threshold_mm=penetration_threshold_mm)

    if pen["over_threshold"] and not authorise_ipr:
        return dict(exported=False, penetration=pen)

    carved_v, carved_f, carve_info = carve_socket(
        base_verts, base_faces, crown_verts, crown_faces, clearance_mm=clearance_mm)

    tag = f"_Stage{stage:02d}" if stage is not None else ""
    tooth_path = os.path.join(out_dir, f"{arch_name}{tag}_Tooth.stl")
    base_path = os.path.join(out_dir, f"{arch_name}{tag}_Base_Socket.stl")
    n_t = write_binary_stl(crown_verts, crown_faces, tooth_path)
    n_b = write_binary_stl(carved_v, carved_f, base_path)

    return dict(exported=True, tooth_stl=tooth_path, base_stl=base_path, socket=carve_info, penetration=pen)

def clear_undercut_periphery(verts: np.ndarray, faces: np.ndarray, arch_frame: dict,
                             rim: np.ndarray, protected_mask: np.ndarray | None = None,
                             max_iters: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    import self_intersection as si
    from scipy.spatial import cKDTree
    
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    rim = np.asarray(rim, int)
    
    curve, curve_info = fit_arch_curve(verts, arch_frame)
    samples = _polyline_samples(curve)
    
    origin, e1, e2, u_occ = _arch_basis(arch_frame)
    rel = verts[faces].mean(axis=1) - origin
    centroids2d = np.column_stack([rel @ e1, rel @ e2])
    dist_to_curve, idx_to_curve = cKDTree(samples).query(centroids2d)
    
    margin_buccal = np.full(len(samples), np.max(dist_to_curve) + 0.1)
    margin_lingual = np.full(len(samples), np.max(dist_to_curve) + 0.1)
    
    min_buccal = np.zeros(len(samples))
    min_lingual = np.zeros(len(samples))
    
    is_lingual_face = np.linalg.norm(centroids2d - curve_info["centre"], axis=1) < np.linalg.norm(samples[idx_to_curve] - curve_info["centre"], axis=1)
    
    if protected_mask is not None:
        prot_buccal = protected_mask & ~is_lingual_face
        prot_lingual = protected_mask & is_lingual_face
        
        for d, idx in zip(dist_to_curve[prot_buccal], idx_to_curve[prot_buccal]):
            min_buccal[idx] = max(min_buccal[idx], d + 0.1)
        for d, idx in zip(dist_to_curve[prot_lingual], idx_to_curve[prot_lingual]):
            min_lingual[idx] = max(min_lingual[idx], d + 0.1)
            
        def smooth_min(arr, win=5):
            res = np.copy(arr)
            for i in range(len(arr)):
                start = max(0, i - win)
                end = min(len(arr), i + win + 1)
                res[i] = np.max(arr[start:end])
            return res
            
        min_buccal = smooth_min(min_buccal)
        min_lingual = smooth_min(min_lingual)
    
    out_v, out_f, out_rim = verts, faces, rim
    total_undercuts_fixed = 0
    
    for iteration in range(max_iters):
        heights = (out_v - origin) @ u_occ
        z_floor_height = np.min(heights) - CAST_BASE_THICKNESS_MM
        
        n_rim = len(out_rim)
        floor_rim_v = np.copy(out_v[out_rim])
        floor_rim_v = floor_rim_v - np.outer(((floor_rim_v - origin) @ u_occ) - z_floor_height, u_occ)
        
        temp_v = np.vstack([out_v, floor_rim_v])
        
        wall_faces = []
        for i in range(n_rim):
            j = (i + 1) % n_rim
            v_a = out_rim[i]
            v_b = out_rim[j]
            v_a_floor = len(out_v) + i
            v_b_floor = len(out_v) + j
            wall_faces.append([v_a, v_b, v_b_floor])
            wall_faces.append([v_a, v_b_floor, v_a_floor])
            
        wall_faces = np.array(wall_faces, dtype=int)
        temp_f = np.vstack([out_f, wall_faces])
        
        pairs = si.candidate_pairs(temp_v, temp_f)
        if not len(pairs):
            break
            
        hit, _ = si._pairs_intersect(temp_v, temp_f, pairs, 1e-6)
        intersecting = pairs[hit]
        
        n_scan = len(out_f)
        bad_faces = set()
        
        for a, b in intersecting:
            ta = "scan" if a < n_scan else "wall"
            tb = "scan" if b < n_scan else "wall"
            if ta == "scan" and tb != "scan": bad_faces.add(a)
            if tb == "scan" and ta != "scan": bad_faces.add(b)
            
        if not bad_faces:
            break
            
        bad_faces_arr = np.array(list(bad_faces))
        bad_centroids = out_v[out_f[bad_faces_arr]].mean(axis=1)
        rel_bad = bad_centroids - origin
        bad_2d = np.column_stack([rel_bad @ e1, rel_bad @ e2])
        
        bad_dist, bad_idx = cKDTree(samples).query(bad_2d)
        bad_is_lingual = np.linalg.norm(bad_2d - curve_info["centre"], axis=1) < np.linalg.norm(samples[bad_idx] - curve_info["centre"], axis=1)
        
        prev_buccal = np.copy(margin_buccal)
        prev_lingual = np.copy(margin_lingual)
        
        for d, idx, is_ling in zip(bad_dist, bad_idx, bad_is_lingual):
            if is_ling:
                margin_lingual[idx] = min(margin_lingual[idx], d - 0.05)
            else:
                margin_buccal[idx] = min(margin_buccal[idx], d - 0.05)
                
        def smooth_margin(arr, min_arr, win=5):
            if protected_mask is not None:
                arr = np.maximum(arr, min_arr)
            res = np.copy(arr)
            for i in range(len(arr)):
                start = max(0, i - win)
                end = min(len(arr), i + win + 1)
                res[i] = np.mean(arr[start:end])
            if protected_mask is not None:
                res = np.maximum(res, min_arr)
            return res
            
        margin_buccal = smooth_margin(margin_buccal, min_buccal)
        margin_lingual = smooth_margin(margin_lingual, min_lingual)
        
        if np.allclose(margin_buccal, prev_buccal) and np.allclose(margin_lingual, prev_lingual):
            break
            
        margin_dict = {"buccal": margin_buccal, "lingual": margin_lingual}
        
        try:
            out_v, out_f, tinfo = trim_to_arch(verts, faces, arch_frame, margin_mm=margin_dict, curve=curve)
            out_rim = tinfo["rim_loop"]
        except ValueError as e:
            raise ValueError(f"clear_undercut_periphery: trim_to_arch failed after lowering margin: {e}")
            
        total_undercuts_fixed += len(bad_faces)
        
    else:
        # Just break, don't raise
        pass
        
    info = {
        "iterations": iteration,
        "undercuts_fixed": total_undercuts_fixed
    }
    return out_v, out_f, out_rim, info
