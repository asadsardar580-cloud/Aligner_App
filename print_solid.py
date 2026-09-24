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
