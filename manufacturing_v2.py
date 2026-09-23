"""The "deform, don't cut" manufacturing construction.

WHY THIS EXISTS, in one paragraph. The collar path cuts the crown out along
`socket_rim`, caps the old socket flat on the SAME loop, lofts a connector from
the SAME loop, and re-fuses everything by CSG. Near-tangent, self-touching
contact is therefore structural, and worst for small movements - which is
exactly this product's scope. Five collar iterations changed how hard it tries
and never the geometry it must fit (AGENT_BRIEF B1).

This construction removes the failure mode instead of repairing it:

  * build the T0 cast ONCE, from the conditioned scan with the teeth still in
    the surface (`trim_to_arch` -> `build_cast_base`);
  * per stage, keep the SAME face array: the moving crown is an exact rigid
    transform, gingiva inside the envelope blends harmonically, everything
    else is never written;
  * topology is therefore INHERITED from T0 - closed, manifold, one component,
    consistent winding. There is no socket, no cap, no collar and no boolean,
    so only GEOMETRY can fail, and every way it can is measured.

This module is PURE. It imports no FastAPI and holds no session state, so the
whole construction can be driven from a test, a script or a request handler
without a server. `api_core` adapts it to HTTP; nothing here knows about HTTP.

The reference implementation of the deformation maths is `deform_construction`
(the kit, 16 tests) and of the geometric validator is `self_intersection` (6
tests). Neither is reimplemented here - this module composes them with the
project's own cast construction, stage matrices and file validation.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

import core_geometry as cg
import deform_construction as dc
import manufacturing as mfg
import self_intersection as si
import stage_matrix
import stl_io

#: Identifies the construction in every manifest it writes.
CONSTRUCTION_ID = dc.CONSTRUCTION_ID

#: Packages whose version changes what this code DOES.
_MANIFEST_PACKAGES = ("numpy", "scipy", "manifold3d", "open3d", "trimesh")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class DeformationPolicy:
    """Bounds and clinical defaults. Values are Asaad's (AGENT_BRIEF Part E)."""

    #: E1. Geodesic reach of the blend from a moving tooth. The aligner edge
    #: sits 0-2 mm onto the gingiva; 5 mm keeps the transition gentle and
    #: beyond the trimline.
    envelope_mm: float = dc.DEFAULT_ENVELOPE_MM

    #: E2. Neighbour enamel released to blend at a contact.
    band_mm: float = dc.DEFAULT_CONTACT_BAND_MM

    #: E3. Penetration at or below this is scanner noise / PDL tolerance.
    #: Anything more needs a prescription for that contact.
    ipr_tolerance_mm: float = dc.IPR_TOLERANCE_MM

    #: Cast construction, shared with the collar path.
    trim_margin_mm: float = cg.ARCH_TRIM_MARGIN_MM
    base_thickness_mm: float = cg.CAST_BASE_THICKNESS_MM

    def as_dict(self) -> dict:
        return {"envelope_mm": self.envelope_mm, "band_mm": self.band_mm,
                "ipr_tolerance_mm": self.ipr_tolerance_mm,
                "trim_margin_mm": self.trim_margin_mm,
                "base_thickness_mm": self.base_thickness_mm}


DEFAULT_POLICY = DeformationPolicy()


class CasePlanRefused(Exception):
    """The case cannot be planned. Carries the measurements that refused it."""

    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class CasePlan:
    """Everything a stage needs, solved once per case."""

    V0: np.ndarray                     # T0 cast vertices (scan array + floor)
    F: np.ndarray                      # T0 cast faces - the SAME array every stage
    plan: dc.DeformationPlan
    tooth_records: dict                # tooth id -> {"frame", "c_res", "clinical"}
    policy: DeformationPolicy
    t0_self_intersection: dict = field(default_factory=dict)
    t0_file: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, encoding="utf-8",
                              errors="replace").stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def _package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in _MANIFEST_PACKAGES:
        try:
            out[name] = md.version(name)
        except Exception:                                 # noqa: BLE001
            out[name] = "NOT INSTALLED"
    import sys
    out["python"] = sys.version.split()[0]
    return out


def _file_dict(validation: dict) -> dict:
    """`validate_printable_stl`'s output in the shape `aggregate_gate_v2` reads.

    Two vocabularies for one measurement is how a gate comes to read a key
    nobody writes, so the translation happens HERE, once, and the gate keeps
    its own names.
    """
    edges = validation.get("edges_post_read_before_weld") or {}
    winding = None
    for g in validation.get("gates") or []:
        if g.get("gate") == "consistent_winding":
            winding = bool(g.get("passed"))
    vol = validation.get("volume_mm3")
    return {
        "open": int(validation.get("open_edges", edges.get("open", -1))),
        "nonmanifold": int(validation.get("nonmanifold_edges",
                                          edges.get("nonmanifold", -1))),
        "components": int(validation.get("connected_components", -1)),
        "winding_ok": winding,
        "volume": float(vol) if isinstance(vol, (int, float)) else None,
    }


# ---------------------------------------------------------------------------
# 2.2  build_case_plan
# ---------------------------------------------------------------------------

def build_case_plan(scan_v, scan_f, arch_frame, moving, static_teeth_vertices,
                    policy: DeformationPolicy = DEFAULT_POLICY,
                    arch_curve=None) -> CasePlan:
    """Solve the blend once, for a whole case.

    `moving`   {tooth_id: {"vertices": ids on the SCAN array,
                           "frame": ..., "c_res": ..., "clinical": {...}}}
    `static_teeth_vertices`  vertex ids of every non-moving tooth.

    THE T0 CAST IS BUILT WITH THE TEETH STILL IN THE SURFACE. Nothing is cut,
    so there is no socket to fill and no rim to loft from - which is the whole
    point of the construction.

    PINNED = the trim rim loop, plus every appended floor vertex.
    `build_cast_base` keeps the input vertex array as a PREFIX and only
    appends, so `i >= len(trimmed verts)` identifies the floor exactly
    (AGENT_BRIEF A1.2). Those vertices carry the cast's own boundary; if the
    blend reached them the base would deform.

    REFUSES when the T0 cast self-intersects, and carries the census. A stage
    inherits T0's topology and its geometry, so a T0 defect is present in
    every stage that could ever be built from it - gating it here is the
    difference between one honest refusal and N unexplained ones.
    """
    t_start = time.perf_counter()
    scan_v = np.asarray(scan_v, float)
    scan_f = np.asarray(scan_f, np.int64)

    # --- the T0 cast, teeth in place ------------------------------------
    t0 = time.perf_counter()
    tv, tf, trim_info = cg.trim_to_arch(scan_v, scan_f, arch_frame,
                                        margin_mm=policy.trim_margin_mm,
                                        curve=arch_curve)
    n_trimmed = len(tv)
    V0, F, base_info = cg.build_cast_base(
        tv, tf, arch_frame, base_thickness_mm=policy.base_thickness_mm,
        rim=trim_info.get("rim_loop"))
    t_cast = time.perf_counter() - t0

    # --- pinned: the trim rim and everything appended below it ----------
    pinned = np.zeros(len(V0), bool)
    pinned[n_trimmed:] = True                      # floor + wall vertices
    rim = trim_info.get("rim_loop")
    if rim is not None and len(np.asarray(rim)):
        pinned[np.asarray(rim, np.int64)] = True

    # --- T0 must not already self-intersect ------------------------------
    t0 = time.perf_counter()
    V0_32 = V0.astype(np.float32).astype(np.float64)
    t0_si = si.self_intersection_report(V0_32, F)
    t_si = time.perf_counter() - t0

    blob0 = cg.write_binary_stl_bytes(V0_32, F)
    t0_file = _file_dict(mfg.validate_printable_stl(blob0))

    if not t0_si.get("measured") or t0_si.get("intersecting_pairs", -1) != 0:
        raise CasePlanRefused(
            "t0_cast_self_intersects",
            {"reason": ("the T0 cast crosses itself before any tooth is "
                        "moved. Every stage inherits this geometry, so no "
                        "stage built from it can be print ready."),
             "t0_self_intersection": t0_si,
             "t0_file": t0_file,
             "cast": {"vertices": int(len(V0)), "faces": int(len(F)),
                      "trimmed_scan_vertices": int(n_trimmed),
                      "appended_floor_vertices": int(len(V0) - n_trimmed),
                      "trim_margin_mm": float(policy.trim_margin_mm),
                      "build_seconds": round(t_cast, 3)},
             "decision": "AGENT_BRIEF E7 - the T0 cleanup rule is Asaad's."})

    # --- contact bands on the STATIC side --------------------------------
    static_ids = np.unique(np.asarray(static_teeth_vertices, np.int64)) \
        if static_teeth_vertices is not None else np.zeros(0, np.int64)
    moving_sets = {t: np.unique(np.asarray(m["vertices"], np.int64))
                   for t, m in moving.items()}

    band = np.zeros(0, np.int64)
    if len(static_ids) and moving_sets:
        bands = [dc.contact_band(V0, F, ids, static_ids, width_mm=policy.band_mm)
                 for ids in moving_sets.values()]
        band = np.unique(np.concatenate(bands)) if bands else band

    # --- weights ----------------------------------------------------------
    t0 = time.perf_counter()
    plan = dc.plan_deformation(V0, F, moving_sets, static_ids, pinned,
                               envelope_mm=policy.envelope_mm,
                               contact_band_vertices=band)
    t_weights = time.perf_counter() - t0

    return CasePlan(
        V0=V0, F=F, plan=plan,
        tooth_records={t: {"frame": m["frame"], "c_res": m["c_res"],
                           "clinical": m.get("clinical") or {}}
                       for t, m in moving.items()},
        policy=policy,
        t0_self_intersection=t0_si,
        t0_file=t0_file,
        diagnostics={
            "construction": CONSTRUCTION_ID,
            "cast": {"vertices": int(len(V0)), "faces": int(len(F)),
                     "trimmed_scan_vertices": int(n_trimmed),
                     "appended_floor_vertices": int(len(V0) - n_trimmed)},
            "trim": {k: v for k, v in trim_info.items()
                     if k not in ("curve", "rim_loop")},
            "cast_base": {k: v for k, v in base_info.items()
                          if not isinstance(v, np.ndarray)},
            "pinned_vertices": int(pinned.sum()),
            "contact_band_vertices": int(len(band)),
            "seconds": {"cast": round(t_cast, 3),
                        "t0_self_intersection": round(t_si, 3),
                        "weights": round(t_weights, 3),
                        "total": round(time.perf_counter() - t_start, 3)},
        })


# ---------------------------------------------------------------------------
# 2.2  build_stage_v2
# ---------------------------------------------------------------------------

def build_stage_v2(case_plan: CasePlan, stage_matrices: dict,
                   prescribed_ipr: float = 0.0, stage: int = 1,
                   total_stages: int = 1) -> dict:
    """One stage: positions, measurements, bytes, verdict. Pure.

    The order is the brief's, and each step measures what the NEXT one will
    consume - positions, then the construction report, then the float32
    rounding, then self-intersection ON THOSE ROUNDED POSITIONS, then the
    bytes, then the file's own re-read, then the gate.
    """
    t_start = time.perf_counter()
    V0, F = case_plan.V0, case_plan.F
    plan = case_plan.plan

    # 1. positions
    t0 = time.perf_counter()
    Vk = dc.stage_positions(plan, V0, stage_matrices, apply=cg.apply_matrix)
    t_pos = time.perf_counter() - t0

    # 2. construction report
    t0 = time.perf_counter()
    report = dc.stage_report(plan, V0, Vk, F, stage_matrices,
                             apply=cg.apply_matrix)
    t_report = time.perf_counter() - t0

    # 3. round to float32 - the precision the file will actually hold
    Vk32 = Vk.astype(np.float32).astype(np.float64)

    # 4. self-intersection, restricted to faces that actually moved. Faces
    #    whose vertices are all untouched were proven clean at T0 and cannot
    #    have started intersecting each other.
    touched = np.zeros(len(V0), bool)
    touched[plan.free_idx] = True
    for idx in plan.tooth_sets.values():
        touched[idx] = True
    active = touched[F].any(axis=1)

    t0 = time.perf_counter()
    sx = si.self_intersection_report(Vk32, F, active_faces=active)
    t_si = time.perf_counter() - t0

    # 5. bytes
    blob = cg.write_binary_stl_bytes(Vk32, F)

    # 6. the file's own re-read
    t0 = time.perf_counter()
    validation = mfg.validate_printable_stl(blob)
    t_file = time.perf_counter() - t0
    file_dict = _file_dict(validation)

    # 7. the gate
    record = {
        "plan_diagnostics": plan.diagnostics,
        "stage_report": report,
        "self_intersection": sx,
        "file": file_dict,
        # The face array is the T0 cast's own, never rebuilt - that is what
        # makes the topology inherited rather than re-derived.
        "index_buffer_unchanged": bool(F is case_plan.F),
        "prescribed_ipr_mm": float(prescribed_ipr or 0.0),
    }
    gate = dc.aggregate_gate_v2(record)

    # 8. manifest
    manifest = {
        "construction": CONSTRUCTION_ID,
        "stage": int(stage),
        "stages": int(total_stages),
        "policy": case_plan.policy.as_dict(),
        "weight_diagnostics": plan.diagnostics,
        "tooth_matrices": {str(t): np.asarray(M, float).tolist()
                           for t, M in stage_matrices.items()},
        "prescribed_ipr_mm": float(prescribed_ipr or 0.0),
        "gate": {"verdict": gate["verdict"],
                 "print_ready": gate["print_ready"],
                 "failed_gates": gate["failed_gates"]},
        "measured": {
            "stage_report": report,
            "self_intersection": sx,
            "file": file_dict,
            "file_validation": {k: validation.get(k) for k in
                                ("open_edges", "nonmanifold_edges",
                                 "connected_components", "volume_mm3",
                                 "reread_vertices", "reread_faces",
                                 "reader_weld_merged_vertices")},
        },
        "advisory": gate["advisory"],
        "stl_sha256": hashlib.sha256(blob).hexdigest(),
        "stl_bytes": int(len(blob)),
        "stage_digest": dc.stage_digest(Vk32, F),
        "git_commit": _git_commit(),
        "packages": _package_versions(),
        "seconds": {
            "positions": round(t_pos, 3),
            "stage_report": round(t_report, 3),
            "self_intersection": round(t_si, 3),
            "file_validation": round(t_file, 3),
            "total": round(time.perf_counter() - t_start, 3),
        },
    }

    return {"blob": blob, "verts": Vk32, "faces": F, "gate": gate,
            "manifest": manifest, "report": report,
            "self_intersection": sx, "file": file_dict}


# ---------------------------------------------------------------------------
# 2.2  stage matrices - the SAME derivation the collar path uses
# ---------------------------------------------------------------------------

def stage_matrices_for(case_plan: CasePlan, k: int, n: int) -> dict:
    """{tooth_id: 4x4} at stage k of n, via `stage_matrix.stage_matrix`.

    Both constructions call the same function, and
    `test_stage_matrix_shared.py` pins them at 0 ULP.
    """
    return {t: stage_matrix.stage_matrix(rec["frame"], rec["c_res"],
                                         rec["clinical"], k, n)
            for t, rec in case_plan.tooth_records.items()}
