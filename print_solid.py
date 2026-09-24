"""The printable model: a VOXEL SOLIDIFICATION of the closed cast.

DECISION OF RECORD (Task 2, 24 Sep 2026 - not to be re-evaluated). The model a
lab prints is made by voxel-solidifying the closed cast, not by exact CSG. The
raw cast (`trim_to_arch` -> `build_cast_base`, then the per-stage deformation)
may cross itself - on the real mandible the scan surface near the trim rim
crosses the base wall 87 times - and solidification is what turns that into
one watertight, intersection-free solid. The calls below are the ones tested
by the clinical owner on MeshLib 3.1.4.297 against that closed cast: 1 body,
0 self-intersections, crown deviation p95 0.002 mm, STL re-read clean.

WHAT THIS MODULE DOES NOT DO: decide PRINT READY. `solidify` builds the solid,
`measure_solid` measures the WRITTEN BYTES, and `solid_gate` is the pure,
fail-closed verdict over those measurements. `manufacturing_v2` composes it
with the deformation and contact gates into the stage's one aggregate verdict.

EVERY MEASUREMENT IS TAKEN ON THE RE-READ FILE (CLAUDE.md rule 5): the bytes are
parsed back with `stl_io.parse_stl_bytes` and welded with `cg.weld_vertices`
exactly as a slicer would, and manifold3d, MeshLib's self-collision test, the
topology counts and the crown fidelity all run on THAT mesh. The project's own
`self_intersection.py` is deliberately not used on this path.

MeshLib is free for non-commercial / educational use only (README.md).
"""
from __future__ import annotations

import time

import numpy as np

import core_geometry as cg
import stl_io

# ---------------------------------------------------------------------------
# Parameters of the solidification - the tested values, not tuning knobs
# ---------------------------------------------------------------------------

#: Voxel edge. 0.05 mm is what was tested on the real cast.
VOXEL_SIZE_MM = 0.05
#: Decimation may move the surface by at most this much.
DECIMATE_MAX_ERROR_MM = 0.01
#: OpenVDB sign detection. NOT HoleWindingRule: in testing it did not finish
#: in 280 s on the real cast. The offset MODE stays at MeshLib's default
#: (Standard).
SIGN_DETECTION_MODE = "OpenVDB"

# ---------------------------------------------------------------------------
# Limits. Crown fidelity p95/max and the 19 mm height are the clinical
# owner's (Task 2 step 3); the height is a WARNING only.
# ---------------------------------------------------------------------------

CROWN_DEVIATION_P95_LIMIT_MM = 0.03
CROWN_DEVIATION_MAX_LIMIT_MM = 0.10
MODEL_HEIGHT_WARN_MM = 19.0
#: How many worst crown-fidelity locations every manifest carries.
WORST_LOCATIONS = 20

# ---------------------------------------------------------------------------
# Measurement keys. Written by `measure_solid`, read by `solid_gate`, and
# pinned by a producer/consumer test (CLAUDE.md rule 9).
# ---------------------------------------------------------------------------

K_SOLIDIFIED = "solidified"
K_SOLIDIFY_REFUSAL = "solidify_refusal"
K_MANIFOLD_STATUS = "manifold3d_status"
K_BODIES = "manifold3d_bodies"
K_VOLUME = "manifold3d_volume_mm3"
K_SELF_COLLIDING = "meshlib_self_colliding"
K_SELF_COLLIDING_PAIRS = "meshlib_self_colliding_pairs"
K_OPEN_EDGES = "file_open_edges"
K_NONMANIFOLD_EDGES = "file_nonmanifold_edges"
K_COMPONENTS = "file_components"
K_WINDING = "file_winding_consistent"
K_CROWN_POINTS = "crown_points"
K_CROWN_P95 = "crown_deviation_p95_mm"
K_CROWN_MAX = "crown_deviation_max_mm"
K_CROWN_WORST = "crown_worst_locations"
K_CROWN_FAILURE = "crown_deviation_failure"
K_HEIGHT = "model_height_mm"
K_TRIANGLES = "triangles"
K_REREAD_VERTICES = "reread_vertices"

#: The solid's gates, in report order. Every one is REQUIRED.
SOLID_GATES = (
    "solidified",                    # solidify produced a solid (no refusal)
    "solid_manifold_status_ok",      # manifold3d status() is NoError
    "solid_single_body",             # decompose() gives exactly 1 body
    "solid_positive_volume",         # volume > 0
    "solid_no_self_intersection",    # MeshLib findSelfCollidingTriangles is False
    "file_zero_open_edges",          # re-read + weld: 0 open edges
    "file_zero_nonmanifold_edges",   # re-read + weld: 0 non-manifold edges
    "file_single_component",         # re-read + weld: 1 component
    "file_consistent_winding",       # re-read + weld: every directed edge once
    "crown_fidelity_p95",            # planned tooth surface -> solid, p95
    "crown_fidelity_max",            # planned tooth surface -> solid, max
)

#: Reported, never refusing.
SOLID_ADVISORY = ("model_height_over_limit",)


class SolidifyRefused(Exception):
    """The cast cannot be solidified. Carries the measurements that refused it."""

    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _meshlib():
    from meshlib import mrmeshnumpy as mn
    from meshlib import mrmeshpy as mr
    return mr, mn


def meshlib_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("meshlib")
    except Exception:                                     # noqa: BLE001
        return "NOT INSTALLED"


def _manifold(verts, faces):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(
        vert_properties=np.ascontiguousarray(np.asarray(verts, np.float32)),
        tri_verts=np.ascontiguousarray(np.asarray(faces, np.uint32))))


def _status_name(status) -> str:
    return str(getattr(status, "name", status)).split(".")[-1]


# ---------------------------------------------------------------------------
# Step 2 - solidify
# ---------------------------------------------------------------------------

def solidify(verts, faces, voxel_size_mm: float | None = None,
             decimate_max_error_mm: float | None = None):
    """Voxel-solidify a CLOSED cast. Returns (verts, faces, report).

    a. the input must have 0 open edges - `build_cast_base` guarantees it, so
       an open edge here is a construction defect, and it is REFUSED rather
       than handed to a sign-detection pass that would guess an inside;
    b-e. the exact MeshLib calls the decision of record names;
    f. keep ONLY the largest body by manifold3d volume. Testing produced 63
       floating crumbs totalling 0.2 mm3; each is counted and reported, and a
       negative-volume part - an ENCLOSED VOID, not a crumb (CLAUDE.md
       lessons) - is counted separately as filled;
    g. timings and triangle counts.

    The output positions are float32 values (what the STL will hold), carried
    in float64 arrays.

    The parameters default to the module constants, read AT CALL TIME, and the
    report states the values actually used - so a manifest can never describe
    a voxel size other than the one that made the file.
    """
    if voxel_size_mm is None:
        voxel_size_mm = VOXEL_SIZE_MM
    if decimate_max_error_mm is None:
        decimate_max_error_mm = DECIMATE_MAX_ERROR_MM
    mr, mn = _meshlib()
    t_all = time.perf_counter()
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)

    # a. closed, or refused
    mrep = cg.manifold_report(F)
    if mrep["open_edges"] != 0 or not len(F):
        raise SolidifyRefused(
            "cast_has_open_edges",
            {"reason": ("the cast handed to solidification is not closed. "
                        "build_cast_base guarantees a closed cast, so this is "
                        "a construction defect, not something to voxelise "
                        "around."),
             "open_edges": int(mrep["open_edges"]),
             "nonmanifold_edges": int(mrep["nonmanifold_edges"]),
             "faces": int(len(F))})

    # b. the mesh
    t0 = time.perf_counter()
    mesh = mn.meshFromFacesVerts(F.astype(np.int32), V.astype(np.float32))

    # c. parameters - OpenVDB sign detection, default (Standard) mode
    p = mr.GeneralOffsetParameters()
    p.voxelSize = float(voxel_size_mm)
    p.signDetectionMode = getattr(mr.SignDetectionMode, SIGN_DETECTION_MODE)

    # d. zero offset = the solid bounded by the cast
    out = mr.generalOffsetMesh(mr.MeshPart(mesh), 0.0, p)
    t_voxel = time.perf_counter() - t0
    tri_voxel = int(out.topology.numValidFaces())

    # e. decimate in place
    t0 = time.perf_counter()
    s = mr.DecimateSettings()
    s.maxError = float(decimate_max_error_mm)
    s.packMesh = True
    dres = mr.decimateMesh(out, s)
    t_decimate = time.perf_counter() - t0
    tri_decimated = int(out.topology.numValidFaces())

    Vo = np.asarray(mn.getNumpyVerts(out), np.float64)
    Fo = np.asarray(mn.getNumpyFaces(out.topology), np.int64)

    # f. the largest body only
    t0 = time.perf_counter()
    solid = _manifold(Vo, Fo)
    status = _status_name(solid.status())
    if status != "NoError":
        raise SolidifyRefused(
            "solidified_mesh_not_manifold",
            {"reason": ("manifold3d will not accept the solidified mesh, so "
                        "its bodies cannot be separated."),
             "manifold3d_status": status, "triangles": int(len(Fo))})
    parts = solid.decompose()
    vols = np.array([float(q.volume()) for q in parts])
    if not len(vols) or vols.max() <= 0.0:
        raise SolidifyRefused(
            "solidified_mesh_has_no_body",
            {"reason": "solidification produced no positive-volume body.",
             "part_volumes_mm3": [round(float(x), 4) for x in vols]})
    keep = int(np.argmax(vols))
    others = np.delete(vols, keep)
    crumbs = others[others > 0.0]
    voids = others[others < 0.0]
    kept = parts[keep].to_mesh()
    Vs = np.asarray(kept.vert_properties, np.float32)[:, :3].astype(np.float64)
    Fs = np.asarray(kept.tri_verts, np.int64)
    t_keep = time.perf_counter() - t0

    report = {
        "method": "voxel solidification (MeshLib generalOffsetMesh, offset 0)",
        "meshlib_version": meshlib_version(),
        "voxel_size_mm": float(voxel_size_mm),
        "sign_detection_mode": SIGN_DETECTION_MODE,
        "offset_mode": _status_name(p.mode),
        "decimate_max_error_mm": float(decimate_max_error_mm),
        "decimate_error_introduced_mm": float(getattr(dres, "errorIntroduced", float("nan"))),
        "input_vertices": int(len(V)),
        "input_faces": int(len(F)),
        "input_open_edges": int(mrep["open_edges"]),
        "input_nonmanifold_edges": int(mrep["nonmanifold_edges"]),
        "triangles_after_voxelisation": tri_voxel,
        "triangles_after_decimation": tri_decimated,
        "triangles": int(len(Fs)),
        "vertices": int(len(Vs)),
        "bodies_before_keep": int(len(vols)),
        "kept_volume_mm3": round(float(vols[keep]), 4),
        "crumbs_discarded": int(len(crumbs)),
        "crumbs_discarded_mm3": round(float(crumbs.sum()), 4),
        "internal_voids_filled": int(len(voids)),
        "internal_voids_filled_mm3": round(float(-voids.sum()), 4),
        "seconds": {"voxelise": round(t_voxel, 3),
                    "decimate": round(t_decimate, 3),
                    "keep_largest_body": round(t_keep, 3),
                    "total": round(time.perf_counter() - t_all, 3)},
    }
    return Vs, Fs, report


# ---------------------------------------------------------------------------
# Step 3 - measure the WRITTEN bytes
# ---------------------------------------------------------------------------

def face_components(faces) -> int:
    """Connected components over shared EDGES (not shared vertices).

    Two bodies touching at one vertex are two components to a printer and one
    to a vertex graph, so adjacency is by edge. Same answer as
    `manufacturing.components`, vectorised: that one walks a Python dict and a
    million-triangle solid takes it far too long to be on an export path.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    f = np.asarray(faces, np.int64)
    if not len(f):
        return 0
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    fid = np.tile(np.arange(len(f)), 3)
    n = int(f.max()) + 1
    key = e[:, 0] * n + e[:, 1]
    order = np.argsort(key, kind="stable")
    key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]
    a, b = fid[:-1][same], fid[1:][same]
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(len(f), len(f)))
    return int(connected_components(g, directed=False)[0])


def _crown_deviation(points, ids, labels, wv, wf):
    """Exact point-to-TRIANGLE distance from each planned tooth point to the
    solid, via the project's Open3D BVH (`manufacturing._point_to_surface`).

    NaN, not 0.0, when it cannot be measured (CLAUDE.md rule 6), and an EMPTY
    point set cannot be measured either - "no tooth deviated" over zero teeth
    is a vacuous pass.
    """
    import manufacturing as mfg
    pts = np.asarray(points, float).reshape(-1, 3) if points is not None \
        else np.zeros((0, 3))
    if not len(pts):
        return {K_CROWN_POINTS: 0, K_CROWN_P95: None, K_CROWN_MAX: None,
                K_CROWN_WORST: [],
                K_CROWN_FAILURE: "no tooth vertices were supplied"}
    d = mfg._point_to_surface(pts, wv, wf)
    if d is None or len(d) != len(pts) or not np.isfinite(d).all():
        return {K_CROWN_POINTS: int(len(pts)), K_CROWN_P95: None,
                K_CROWN_MAX: None, K_CROWN_WORST: [],
                K_CROWN_FAILURE: (getattr(mfg, "LAST_DISTANCE_FAILURE", None)
                                  or "non-finite distances")}
    worst = np.argsort(-d, kind="stable")[:WORST_LOCATIONS]
    ids = np.asarray(ids) if ids is not None else None
    labels = np.asarray(labels) if labels is not None else None
    return {
        K_CROWN_POINTS: int(len(pts)),
        K_CROWN_P95: float(np.percentile(d, 95)),
        K_CROWN_MAX: float(d.max()),
        K_CROWN_WORST: [
            {"rank": r + 1,
             "vertex_id": int(ids[i]) if ids is not None else int(i),
             "tooth": (str(labels[i]) if labels is not None else None),
             "position": [round(float(x), 4) for x in pts[i]],
             "deviation_mm": round(float(d[i]), 5)}
            for r, i in enumerate(worst)],
        K_CROWN_FAILURE: None,
    }


def measure_solid(blob: bytes, tooth_points=None, tooth_ids=None,
                  tooth_labels=None, u_occ=None) -> dict:
    """Every step-3 measurement, on the RE-READ bytes.

    `tooth_points`  the planned (moved) position of every tooth vertex, float64.
    `tooth_ids` / `tooth_labels`  per point, for naming the worst locations.
    `u_occ`  the occlusal axis, for the model height (cusp tips to floor).
    """
    mr, mn = _meshlib()
    t_all = time.perf_counter()
    out: dict = {K_SOLIDIFIED: True, K_SOLIDIFY_REFUSAL: None}

    # the file, as a reader receives it
    pv, pf = stl_io.parse_stl_bytes(blob)
    wv, wf, merged = cg.weld_vertices(pv, pf)
    out[K_REREAD_VERTICES] = int(len(wv))
    out[K_TRIANGLES] = int(len(wf))
    out["reader_weld_merged_vertices"] = int(merged)

    # topology of the file
    t0 = time.perf_counter()
    mrep = cg.manifold_report(wf)
    out[K_OPEN_EDGES] = int(mrep["open_edges"])
    out[K_NONMANIFOLD_EDGES] = int(mrep["nonmanifold_edges"])
    out[K_COMPONENTS] = face_components(wf)
    out[K_WINDING] = bool(cg._winding_is_consistent(wf))
    t_topo = time.perf_counter() - t0

    # manifold3d
    t0 = time.perf_counter()
    solid = _manifold(wv, wf)
    status = _status_name(solid.status())
    out[K_MANIFOLD_STATUS] = status
    if status == "NoError":
        out[K_BODIES] = int(len(solid.decompose()))
        out[K_VOLUME] = float(solid.volume())
    else:
        out[K_BODIES] = None
        out[K_VOLUME] = None
    t_m3 = time.perf_counter() - t0

    # MeshLib self-collision - the exact call, then the pairs only if it hit
    t0 = time.perf_counter()
    mesh = mn.meshFromFacesVerts(wf.astype(np.int32), wv.astype(np.float32))
    hit = mr.findSelfCollidingTriangles(mr.MeshPart(mesh), None)
    out[K_SELF_COLLIDING] = bool(hit)
    out[K_SELF_COLLIDING_PAIRS] = None
    if hit:
        pairs = mr.findSelfCollidingTriangles(mr.MeshPart(mesh))
        P = np.array([[int(q.aFace), int(q.bFace)] for q in pairs], np.int64)
        out[K_SELF_COLLIDING_PAIRS] = int(len(P))
        out["meshlib_self_colliding_examples"] = [
            {"faces": [int(a), int(b)],
             "at": [round(float(x), 3) for x in
                    wv[wf[[a, b]].ravel()].mean(axis=0)]}
            for a, b in P[:WORST_LOCATIONS]]
    t_si = time.perf_counter() - t0

    # crown fidelity
    t0 = time.perf_counter()
    out.update(_crown_deviation(tooth_points, tooth_ids, tooth_labels, wv, wf))
    t_crown = time.perf_counter() - t0

    # model height, cusp tips to floor - a warning only
    if u_occ is not None and len(wv):
        u = np.asarray(u_occ, float)
        u = u / np.linalg.norm(u)
        h = wv @ u
        out[K_HEIGHT] = float(h.max() - h.min())
    else:
        out[K_HEIGHT] = None

    out["seconds"] = {"topology": round(t_topo, 3), "manifold3d": round(t_m3, 3),
                      "self_collision": round(t_si, 3),
                      "crown_fidelity": round(t_crown, 3),
                      "total": round(time.perf_counter() - t_all, 3)}
    return out


def refused_measurement(refusal: SolidifyRefused) -> dict:
    """The measurement record of a stage whose cast could not be solidified.

    Every other key is ABSENT, so every other gate fails on a missing
    measurement exactly as it would on a bad one - and `solidified` fails by
    name, carrying the refusal.
    """
    return {K_SOLIDIFIED: False,
            K_SOLIDIFY_REFUSAL: {"reason": refusal.reason, **refusal.detail}}


# ---------------------------------------------------------------------------
# The solid's gate - PURE, FAIL-CLOSED
# ---------------------------------------------------------------------------

def _finite(x) -> bool:
    # isinstance before comparing: every comparison against NaN is False, so
    # `x <= limit` alone would PASS a NaN on the negated form (CLAUDE.md).
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and bool(np.isfinite(x))


def solid_gate(m: dict) -> dict:
    """{gate: {"ok", "measured", "limit"}} for every SOLID_GATES entry, plus
    the failed names and the advisory warnings. A missing key fails."""
    m = m or {}
    p95, mx = m.get(K_CROWN_P95), m.get(K_CROWN_MAX)
    vol = m.get(K_VOLUME)
    checks = {
        "solidified": (m.get(K_SOLIDIFIED) is True,
                       m.get(K_SOLIDIFY_REFUSAL) or m.get(K_SOLIDIFIED),
                       "a solid, no refusal"),
        "solid_manifold_status_ok": (m.get(K_MANIFOLD_STATUS) == "NoError",
                                     m.get(K_MANIFOLD_STATUS), "NoError"),
        "solid_single_body": (m.get(K_BODIES) == 1 and not isinstance(m.get(K_BODIES), bool),
                              m.get(K_BODIES), 1),
        "solid_positive_volume": (_finite(vol) and vol > 0.0, vol, "> 0"),
        "solid_no_self_intersection": (m.get(K_SELF_COLLIDING) is False,
                                       m.get(K_SELF_COLLIDING_PAIRS)
                                       if m.get(K_SELF_COLLIDING) else m.get(K_SELF_COLLIDING),
                                       "findSelfCollidingTriangles is False"),
        "file_zero_open_edges": (m.get(K_OPEN_EDGES) == 0, m.get(K_OPEN_EDGES), 0),
        "file_zero_nonmanifold_edges": (m.get(K_NONMANIFOLD_EDGES) == 0,
                                        m.get(K_NONMANIFOLD_EDGES), 0),
        "file_single_component": (m.get(K_COMPONENTS) == 1, m.get(K_COMPONENTS), 1),
        "file_consistent_winding": (m.get(K_WINDING) is True, m.get(K_WINDING), True),
        "crown_fidelity_p95": (_finite(p95) and p95 <= CROWN_DEVIATION_P95_LIMIT_MM,
                               p95 if p95 is not None else m.get(K_CROWN_FAILURE),
                               CROWN_DEVIATION_P95_LIMIT_MM),
        "crown_fidelity_max": (_finite(mx) and mx <= CROWN_DEVIATION_MAX_LIMIT_MM,
                               mx if mx is not None else m.get(K_CROWN_FAILURE),
                               CROWN_DEVIATION_MAX_LIMIT_MM),
    }
    gates = {name: {"ok": bool(checks[name][0]), "measured": checks[name][1],
                    "limit": checks[name][2]} for name in SOLID_GATES}
    h = m.get(K_HEIGHT)
    advisory = {
        "model_height_mm": h,
        "model_height_warn_mm": MODEL_HEIGHT_WARN_MM,
        # None when unmeasured - a warning that could not be evaluated is not
        # a clear bill.
        "model_height_over_limit": (bool(h > MODEL_HEIGHT_WARN_MM)
                                    if _finite(h) else None),
    }
    return {"gates": gates,
            "failed_gates": [g for g in SOLID_GATES if not gates[g]["ok"]],
            "advisory": advisory}
