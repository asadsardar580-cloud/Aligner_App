"""Topology-preserving stage construction: DEFORM, DON'T CUT.

THE CONSTRUCTION CONTRACT
Build the closed T0 cast ONCE from the conditioned scan WITH THE TEETH STILL IN
THE SURFACE (cg.trim_to_arch -> cg.build_cast_base). build_cast_base keeps the
scan's vertex array as a prefix and only appends floor vertices, so every scan
vertex id - every tooth's vertex set - is valid in the closed solid.

Per stage k, with the SAME face array F for every stage:
    moving-tooth vertices   V_k[t] = apply_matrix(V0[t], M_i^k)        exact rigid
    envelope gingiva        V_k[g] = V0[g] + sum_i w_i(g) (M_i^k V0[g] - V0[g])
    everything else         V_k[o] = V0[o]                             never written

F never changes, so closedness, manifoldness, winding and component count of
the T0 solid are INHERITED, not re-derived. No socket, no cap, no collar, no
boolean. What can still go wrong is GEOMETRY only - an inverted triangle or a
self-intersection (a fold, or a crown pushed into its neighbour) - and both are
measured by gates that fail closed.

w_i is the discrete harmonic function on the envelope with Dirichlet data
1 on tooth i, 0 on every other tooth, on the envelope edge, on the trim rim and
on the base walls/floor. The Laplacian uses cotangent weights CLAMPED
non-negative, which makes it an M-matrix: the discrete maximum principle then
guarantees 0 <= w_i <= 1 and sum_i w_i <= 1 (no overshoot, no "wrong-way" motion).

Pure NumPy/SciPy. No new dependency.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components, dijkstra

CONSTRUCTION_ID = "deformation_v1"
DEFAULT_ENVELOPE_MM = 5.0      # geodesic reach from the moving tooth. Aligner edge sits 0-2 mm onto
                               # gingiva; 5 mm keeps the transition gentle and past the trimline.
WEIGHT_TOL = 1e-9


def apply_matrix(verts, M):
    """Identical arithmetic to core_geometry.apply_matrix. In the app, PASS cg.apply_matrix
    so the rigidity gate compares against the very function the crown uses."""
    h = np.column_stack([verts, np.ones(len(verts))])
    return (h @ np.asarray(M, float).T)[:, :3]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def edge_graph(V, F):
    F = np.asarray(F, np.int64)
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    w = np.maximum(np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1), 1e-12)  # csgraph: 0 = no edge
    n = len(V)
    G = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    return G


def cotan_weights(V, F, cap=1e3):
    """Symmetric edge weights 0.5*(cot a + cot b), clamped to (0, cap].

    Clamping to a small POSITIVE floor (not zero) keeps the graph connected
    while making L = D - W an M-matrix: that is what buys the maximum principle.
    """
    F = np.asarray(F, np.int64)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def cot(p, q, r):
        u, v = q - p, r - p
        return np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-30)
    ca, cb, cc = cot(a, b, c), cot(b, c, a), cot(c, a, b)
    I = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    J = np.r_[F[:, 2], F[:, 0], F[:, 1]]
    n = len(V)
    W = sp.coo_matrix((0.5 * np.r_[ca, cb, cc], (I, J)), shape=(n, n)).tocsr()
    W = (W + W.T).tocsr()
    pos = W.data[W.data > 0]
    floor = 1e-6 * (float(np.median(pos)) if len(pos) else 1.0)
    W.data = np.clip(W.data, floor, cap)
    return W


# ---------------------------------------------------------------------------
# Planning (once per case)
# ---------------------------------------------------------------------------

@dataclass
class DeformationPlan:
    free_idx: np.ndarray                    # vertices that blend (envelope gingiva + contact bands)
    tooth_sets: dict                        # tooth id -> vertex ids (disjoint)
    weights: dict                           # tooth id -> weights over free_idx
    pinned_mask: np.ndarray                 # vertices that may NEVER move
    band_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    diagnostics: dict = field(default_factory=dict)


DEFAULT_CONTACT_BAND_MM = 1.5   # neighbour enamel released to blend at a contact
IPR_TOLERANCE_MM = 0.05         # clinical rule: <= 0.05 mm is scanner noise / PDL tolerance


def contact_band(V, F, moving_vertices, neighbour_vertices, width_mm=DEFAULT_CONTACT_BAND_MM):
    """Neighbour-crown vertices within `width_mm` (geodesic, through the mesh) of a
    moving crown.

    WHY: where two crowns touch, the scanner never sees the contact point; the
    surface joining them there is the scanner's interpolation, not captured
    enamel. Faces that span two rigid bodies in relative motion must shear and,
    past their own width, fold (measured: every inverted face in the contacting
    fixture was such a bridge face, none was gingiva). Releasing a narrow band of
    the NEIGHBOUR's contact zone keeps the MOVING crown bit-exact and confines
    the deviation to the anchorage tooth's interproximal zone - where it is
    MEASURED per stage as implicit IPR and gated, never hidden.
    """
    moving_vertices = np.asarray(moving_vertices, np.int64)
    if not len(moving_vertices) or not len(neighbour_vertices):
        return np.zeros(0, np.int64)
    d = dijkstra(edge_graph(np.asarray(V, float), F), directed=False,
                 indices=moving_vertices, min_only=True, limit=width_mm)
    return np.intersect1d(np.asarray(neighbour_vertices, np.int64), np.flatnonzero(np.isfinite(d)))


def plan_deformation(V0, F, moving_teeth, static_tooth_vertices, pinned_mask,
                     envelope_mm=DEFAULT_ENVELOPE_MM, contact_band_vertices=None):
    """Solve the blend weights once per case.

    moving_teeth           {tooth_id: vertex ids of that tooth's crown region}
    static_tooth_vertices  vertex ids of every NON-moving tooth (enamel that must
                           stay bit-identical); overlap with a moving tooth is
                           resolved in favour of the moving tooth and REPORTED
    pinned_mask            bool per vertex: trim rim + base walls/floor (+ anything
                           else that must never move)
    contact_band_vertices  neighbour enamel released to blend (see contact_band);
                           reported per stage as implicit IPR, never silently
    """
    V0 = np.asarray(V0, float)
    F = np.asarray(F, np.int64)
    n = len(V0)
    ids = sorted(moving_teeth)
    sets = {t: np.unique(np.asarray(moving_teeth[t], np.int64)) for t in ids}

    all_moving = np.concatenate([sets[t] for t in ids]) if ids else np.zeros(0, np.int64)
    if len(all_moving) != len(np.unique(all_moving)):
        raise ValueError("moving tooth vertex sets overlap; assign each shared contact vertex "
                         "to exactly one tooth before planning")
    moving_mask = np.zeros(n, bool)
    moving_mask[all_moving] = True
    if (moving_mask & pinned_mask).any():
        raise ValueError("a moving tooth touches the trim rim or base; raise the trim margin")

    static_mask = np.zeros(n, bool)
    static_mask[np.asarray(static_tooth_vertices, np.int64)] = True
    shared_contact = int((static_mask & moving_mask).sum())
    static_mask &= ~moving_mask
    band_idx = np.zeros(0, np.int64)
    if contact_band_vertices is not None and len(contact_band_vertices):
        band_idx = np.setdiff1d(np.asarray(contact_band_vertices, np.int64), np.flatnonzero(moving_mask))
        band_idx = band_idx[~np.asarray(pinned_mask, bool)[band_idx]]
        static_mask[band_idx] = False

    G = edge_graph(V0, F)
    dist = dijkstra(G, directed=False, indices=all_moving, min_only=True, limit=envelope_mm) \
        if len(all_moving) else np.full(n, np.inf)
    envelope = np.isfinite(dist) & ~moving_mask
    free = envelope & ~static_mask & ~pinned_mask
    free_idx = np.flatnonzero(free)

    W = cotan_weights(V0, F)
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = (sp.diags(deg) - W).tocsr()
    known_idx = np.flatnonzero(~free)
    LUU = L[free_idx][:, free_idx].tocsc()
    LUK = L[free_idx][:, known_idx].tocsr()

    # A free island with no edge to known data would make LUU singular.
    touch_known = np.asarray(W[free_idx][:, known_idx].sum(axis=1)).ravel() > 0
    ncomp, comp = connected_components(W[free_idx][:, free_idx], directed=False)
    anchored = np.zeros(ncomp, bool)
    np.logical_or.at(anchored, comp, touch_known)
    isolated = ~anchored[comp]

    weights = {}
    if len(free_idx):
        keep = np.flatnonzero(~isolated)
        solver = spla.splu(LUU[keep][:, keep].tocsc()) if len(keep) else None
        for t in ids:
            vals = np.zeros(n)
            vals[sets[t]] = 1.0
            w = np.zeros(len(free_idx))
            if solver is not None:
                rhs = -(LUK[keep] @ vals[known_idx])
                w[keep] = solver.solve(rhs)
            weights[t] = w
    total = sum(weights.values()) if weights else np.zeros(len(free_idx))
    wmin = min((float(w.min()) for w in weights.values() if len(w)), default=0.0)
    wmax = max((float(w.max()) for w in weights.values() if len(w)), default=0.0)
    for t in ids:                                   # clip round-off only; report the raw extrema
        weights[t] = np.clip(weights[t], 0.0, 1.0)

    diag = {
        "construction": CONSTRUCTION_ID,
        "envelope_mm": float(envelope_mm),
        "laplacian": "cotangent, clamped non-negative (M-matrix)",
        "free_vertices": int(len(free_idx)),
        "moving_vertices": int(len(all_moving)),
        "static_tooth_vertices": int(static_mask.sum()),
        "shared_contact_vertices_assigned_to_moving": shared_contact,
        "isolated_free_vertices": int(isolated.sum()),
        "contact_band_vertices": int(len(band_idx)),
        "weight_min_raw": wmin,
        "weight_max_raw": wmax,
        "weight_sum_max_raw": float(total.max()) if len(total) else 0.0,
        "weights_bounded": bool(wmin >= -WEIGHT_TOL and wmax <= 1 + WEIGHT_TOL
                                and (float(total.max()) if len(total) else 0.0) <= 1 + WEIGHT_TOL),
    }
    return DeformationPlan(free_idx, sets, weights, np.asarray(pinned_mask, bool), band_idx, diag)


# ---------------------------------------------------------------------------
# Per stage
# ---------------------------------------------------------------------------

def stage_positions(plan, V0, matrices, apply=apply_matrix):
    """Vertex positions for one stage. `matrices`: {tooth_id: 4x4 stage matrix}."""
    V0 = np.asarray(V0, float)
    Vk = V0.copy()
    fi = plan.free_idx
    disp = np.zeros((len(fi), 3))
    for t in sorted(plan.tooth_sets):                    # fixed order -> deterministic bytes
        w = plan.weights.get(t)
        if w is None:
            continue
        nz = np.flatnonzero(w > 0)
        if len(nz):
            P = V0[fi[nz]]
            disp[nz] += w[nz, None] * (apply(P, matrices[t]) - P)
    moved = np.flatnonzero(np.any(disp != 0.0, axis=1))
    Vk[fi[moved]] = V0[fi[moved]] + disp[moved]          # never write a vertex that did not move
    for t in sorted(plan.tooth_sets):
        idx = plan.tooth_sets[t]
        Vk[idx] = apply(V0[idx], matrices[t])            # exact rigid, same function as the gate
    return Vk


def vertex_normals(V, F):
    """Area-weighted vertex normals; outward on a positively oriented closed solid."""
    F = np.asarray(F, np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(np.asarray(V, float))
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    ln = np.linalg.norm(vn, axis=1)
    return vn / np.where(ln > 0, ln, 1.0)[:, None]


def _face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    return n / np.where(ln > 0, ln, 1.0)[:, None], ln


def stage_report(plan, V0, Vk, F, matrices, apply=apply_matrix):
    """Construction-level measurements. Pure; never raises on bad geometry."""
    V0 = np.asarray(V0, float)
    F = np.asarray(F, np.int64)
    n = len(V0)
    touched = np.zeros(n, bool)
    touched[plan.free_idx] = True
    for idx in plan.tooth_sets.values():
        touched[idx] = True
    moved_faces = touched[F].any(axis=1)

    rigid = {str(t): bool(np.array_equal(Vk[idx], apply(V0[idx], matrices[t])))
             for t, idx in plan.tooth_sets.items()}
    untouched = ~touched
    bit_identical = bool(np.array_equal(Vk[untouched], V0[untouched]))
    pinned_ok = bool(np.array_equal(Vk[plan.pinned_mask], V0[plan.pinned_mask]))

    n0, a0 = _face_normals(V0, F[moved_faces])
    nk, ak = _face_normals(Vk, F[moved_faces])
    cosang = np.einsum("ij,ij->i", n0, nk)
    inverted = int((cosang <= 0.0).sum())
    degenerate = int((ak <= 1e-12 * max(float(np.median(a0)) if len(a0) else 1.0, 1e-30)).sum())

    E = np.sort(np.vstack([F[moved_faces][:, [0, 1]], F[moved_faces][:, [1, 2]],
                           F[moved_faces][:, [2, 0]]]), axis=1)
    E = np.unique(E, axis=0)
    l0 = np.linalg.norm(V0[E[:, 0]] - V0[E[:, 1]], axis=1)
    lk = np.linalg.norm(Vk[E[:, 0]] - Vk[E[:, 1]], axis=1)
    ratio = lk / np.maximum(l0, 1e-12)

    gdisp = np.linalg.norm(Vk[plan.free_idx] - V0[plan.free_idx], axis=1)

    # IMPLICIT IPR. A band vertex pushed INTO its own tooth (against the T0 outward
    # normal) is enamel the model has removed - IPR by another name. It is measured
    # and gated at the clinical tolerance; pulled OUTWARD it is interproximal
    # bridging (block-out), which is reported.
    band_in = band_out = band_max = 0.0
    if len(plan.band_idx):
        vn = vertex_normals(V0, F)[plan.band_idx]
        d = Vk[plan.band_idx] - V0[plan.band_idx]
        along = np.einsum("ij,ij->i", d, vn)
        band_in = float(max(0.0, -along.min()))
        band_out = float(max(0.0, along.max()))
        band_max = float(np.linalg.norm(d, axis=1).max())
    return {
        "construction": CONSTRUCTION_ID,
        "faces_changed": int(moved_faces.sum()),
        "moving_teeth_exact_rigid": rigid,
        "untouched_vertices_bit_identical": bit_identical,
        "pinned_vertices_bit_identical": pinned_ok,
        "inverted_triangles": inverted,
        "min_normal_cosine": float(cosang.min()) if len(cosang) else 1.0,
        "degenerate_triangles": degenerate,
        "edge_stretch_min": float(ratio.min()) if len(ratio) else 1.0,
        "edge_stretch_max": float(ratio.max()) if len(ratio) else 1.0,
        "gingiva_max_displacement_mm": float(gdisp.max()) if len(gdisp) else 0.0,
        "contact_band_vertices": int(len(plan.band_idx)),
        "implicit_ipr_mm": band_in,
        "interproximal_bridge_mm": band_out,
        "neighbour_contact_zone_max_displacement_mm": band_max,
        "moved_faces_mask": moved_faces,                  # consumed by self_intersection
    }


def stage_digest(Vk, F):
    """Content hash of a stage: same scan + same plan + same code -> same digest."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(Vk, np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(F, np.int64)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Aggregate gate for this construction. PURE and FAIL-CLOSED: a measurement that
# is missing fails exactly like a bad one. Gate names are module constants so a
# harness can never read a key the producer does not write (Loop C).
# ---------------------------------------------------------------------------

REQUIRED_GATES = (
    "index_buffer_unchanged",          # F is the T0 cast's own face array -> topology inherited
    "weights_bounded",                 # maximum principle held: 0 <= w <= 1, sum <= 1
    "moving_teeth_exact_rigid",        # crown == apply_matrix(M, crown_T0), bitwise
    "untouched_vertices_bit_identical",
    "pinned_vertices_bit_identical",   # trim rim + walls + floor never move
    "no_inverted_triangles",
    "no_degenerate_triangles",
    "no_self_intersection",            # measured on float32-rounded positions
    "implicit_ipr_within_prescription",
    "written_file_topology",           # re-read bytes: closed, manifold, 1 component, V>0, winding
)
ADVISORY = ("edge_stretch_min", "edge_stretch_max", "gingiva_max_displacement_mm",
            "interproximal_bridge_mm", "neighbour_contact_zone_max_displacement_mm")


def aggregate_gate_v2(record):
    """record keys: plan_diagnostics, stage_report, self_intersection, file,
    index_buffer_unchanged (bool), prescribed_ipr_mm (float, 0 if none)."""
    r = record or {}
    pd, sr, sx, fi = (r.get(k) or {} for k in ("plan_diagnostics", "stage_report",
                                               "self_intersection", "file"))
    rigid = sr.get("moving_teeth_exact_rigid")
    presc = r.get("prescribed_ipr_mm")
    ipr = sr.get("implicit_ipr_mm")
    checks = {
        "index_buffer_unchanged": r.get("index_buffer_unchanged") is True,
        "weights_bounded": pd.get("weights_bounded") is True,
        "moving_teeth_exact_rigid": isinstance(rigid, dict) and len(rigid) > 0 and all(rigid.values()),
        "untouched_vertices_bit_identical": sr.get("untouched_vertices_bit_identical") is True,
        "pinned_vertices_bit_identical": sr.get("pinned_vertices_bit_identical") is True,
        "no_inverted_triangles": sr.get("inverted_triangles") == 0,
        "no_degenerate_triangles": sr.get("degenerate_triangles") == 0,
        "no_self_intersection": sx.get("measured") is True and sx.get("intersecting_pairs") == 0,
        "implicit_ipr_within_prescription": (
            isinstance(ipr, float) and isinstance(presc, (int, float))
            and ipr <= max(IPR_TOLERANCE_MM, float(presc)) + 1e-9),
        "written_file_topology": (fi.get("open") == 0 and fi.get("nonmanifold") == 0
                                  and fi.get("components") == 1 and fi.get("winding_ok") is True
                                  and isinstance(fi.get("volume"), float) and fi["volume"] > 0),
    }
    failed = [g for g in REQUIRED_GATES if not checks[g]]
    return {"print_ready": not failed, "verdict": "PRINT READY" if not failed else "NOT PRINT READY",
            "failed_gates": failed, "construction": CONSTRUCTION_ID,
            "advisory": {k: sr.get(k) for k in ADVISORY}}
