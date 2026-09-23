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


def gate_evidence(gate: dict, record: dict) -> dict:
    """Each REQUIRED gate, with the measurement it was decided on.

    `dc.aggregate_gate_v2` returns a verdict and the names that failed - not
    the numbers behind them. That is the right shape for a gate (it has one
    job and it is fail-closed) and the wrong shape for a manifest: a lab or a
    reviewer reading "NOT PRINT READY: no_self_intersection" needs the count.

    THE VERDICT IS NOT RECOMPUTED HERE. `ok` is read straight out of
    `failed_gates`, so this cannot disagree with the gate, drift from it, or
    quietly become a second implementation of it. All it adds is provenance:
    which measured value each name was decided on.
    """
    failed = set(gate.get("failed_gates") or ())
    sr = record.get("stage_report") or {}
    pd = record.get("plan_diagnostics") or {}
    sx = record.get("self_intersection") or {}
    fi = record.get("file") or {}
    ev = {
        "index_buffer_unchanged": {
            "face_array_is_the_T0_cast_s_own": record.get("index_buffer_unchanged")},
        "weights_bounded": {k: pd.get(k) for k in
                            ("weights_bounded", "weight_min", "weight_max",
                             "weight_sum_max", "free_vertices")},
        "moving_teeth_exact_rigid": {"per_tooth": sr.get("moving_teeth_exact_rigid")},
        "untouched_vertices_bit_identical": {
            "bit_identical": sr.get("untouched_vertices_bit_identical")},
        "pinned_vertices_bit_identical": {
            "bit_identical": sr.get("pinned_vertices_bit_identical")},
        "no_inverted_triangles": {"inverted_triangles": sr.get("inverted_triangles")},
        "no_degenerate_triangles": {"degenerate_triangles": sr.get("degenerate_triangles")},
        "no_self_intersection": {k: sx.get(k) for k in
                                 ("measured", "intersecting_pairs",
                                  "faces_involved", "by_kind")},
        "implicit_ipr_within_prescription": {
            "implicit_ipr_mm": sr.get("implicit_ipr_mm"),
            "prescribed_ipr_mm": record.get("prescribed_ipr_mm"),
            "tolerance_mm": dc.IPR_TOLERANCE_MM,
            "tolerance_provenance": ("clinical rule - at or below this a closure "
                                     "is scanner noise and PDL tolerance, not "
                                     "enamel"),
        },
        "written_file_topology": dict(fi),
    }
    return {name: {"ok": name not in failed, **(ev.get(name) or {})}
            for name in dc.REQUIRED_GATES}


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
    gate = dict(dc.aggregate_gate_v2(record))
    # Provenance beside the verdict. `gate_evidence` reads `ok` out of
    # `failed_gates` rather than re-deciding, so it can never disagree.
    gate["gates"] = gate_evidence(gate, record)

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
                 "failed_gates": gate["failed_gates"],
                 "gates": gate["gates"]},
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
# ---------------------------------------------------------------------------
# 2.3  the tooth vertex sets, on the ORIGINAL scan ids
# ---------------------------------------------------------------------------

def tooth_vertex_sets(scan_faces, moving_faces: dict, static_faces: dict,
                      n_vertices: int | None = None) -> dict:
    """Which scan vertices are a moving tooth, and which are anchorage enamel.

    THE IDS ARE THE SCAN'S OWN, AND THEY STAY THE SCAN'S OWN, all the way to
    the cast. `trim_to_arch` deletes faces and appends hole-fill centroids
    without moving or renumbering a vertex, and `build_cast_base` keeps the
    array it is given as a PREFIX and only appends the floor (AGENT_BRIEF
    A1.2). So scan id i IS cast id i, and no remap exists to get wrong. That
    is the whole reason the sets are derived here, from `/cut`'s own face
    masks and `/segment`'s own labels, rather than by re-finding the teeth on
    the cast - re-finding them is where a correspondence bug would live, and
    CLAUDE.md s.24.3 records what one costs when four loaders each invent
    their own vertex order and every count check still passes.

    A SET IS THE VERTICES OF FACES, NOT THE VERTICES OF A LABEL, and the
    difference is what keeps the rigid gate satisfiable. A triangle whose
    three vertices carry three different owners spans two bodies in relative
    motion: it MUST shear, so it cannot be inside a set that
    `moving_teeth_exact_rigid` then requires to be rigid. Taking the vertices
    of the faces a tooth wholly owns drops those seam triangles and leaves
    their vertices free to blend - which is exactly what the envelope is for.
    A vertex is only dropped when EVERY incident face straddles, i.e. when it
    sits on the seam itself.

    `moving_faces` / `static_faces`   {tooth_id: bool mask or face ids over
                                       the ORIGINAL scan faces}

    Returns the two sets plus the census the brief asks for. Overlaps resolve
    to MOVING in both directions:

      * moving vs static - two labels meeting at a contact point the scanner
        never saw. Reported here, and `dc.plan_deformation` independently
        reports its own count, so the two can be compared.
      * moving vs moving - the kit REFUSES an overlap ("assign each shared
        contact vertex to exactly one tooth before planning") rather than
        picking for us, because a vertex owned by two rigid bodies has no
        correct position. Resolved by lowest tooth id so the answer does not
        depend on dict ordering, and reported per pair.
    """
    F = np.asarray(scan_faces, np.int64)
    n = int(n_vertices) if n_vertices is not None else int(F.max()) + 1

    def _mask(spec):
        m = np.zeros(len(F), bool)
        a = np.asarray(spec)
        if a.dtype == bool:
            if len(a) != len(F):
                raise ValueError(
                    f"face mask is {len(a)} long for {len(F)} faces")
            m |= a
        else:
            m[a.astype(np.int64)] = True
        return m

    def _verts(spec):
        return np.unique(F[_mask(spec)]) if len(F) else np.zeros(0, np.int64)

    moving = {t: _verts(spec) for t, spec in (moving_faces or {}).items()}
    static_by_tooth = {t: _verts(spec) for t, spec in (static_faces or {}).items()}

    # --- moving vs moving: the kit will not choose, so choose here -------
    order = sorted(moving, key=lambda t: str(t))
    claimed = np.zeros(n, bool)
    moving_overlaps, disjoint = {}, {}
    for t in order:
        ids = moving[t]
        clash = ids[claimed[ids]] if len(ids) else ids
        if len(clash):
            moving_overlaps[str(t)] = int(len(clash))
        keep = ids[~claimed[ids]] if len(ids) else ids
        claimed[keep] = True
        disjoint[t] = keep

    # --- moving vs static: static yields ---------------------------------
    static_all = (np.unique(np.concatenate(list(static_by_tooth.values())))
                  if static_by_tooth else np.zeros(0, np.int64))
    shared = int(claimed[static_all].sum()) if len(static_all) else 0
    static_ids = static_all[~claimed[static_all]] if len(static_all) else static_all

    return {
        "moving": disjoint,
        "static": static_ids,
        "diagnostics": {
            "source": "original scan face ids",
            "n_scan_vertices": n,
            "n_scan_faces": int(len(F)),
            "moving_vertices": {str(t): int(len(v)) for t, v in disjoint.items()},
            "moving_faces": {str(t): int(_mask(s).sum())
                             for t, s in (moving_faces or {}).items()},
            "static_vertices": int(len(static_ids)),
            "static_teeth": sorted(str(t) for t in static_by_tooth),
            "shared_contact_vertices_assigned_to_moving": shared,
            "moving_moving_overlaps_resolved": moving_overlaps,
            "note": ("a vertex claimed by both a moving tooth and a static "
                     "one is MOVING; the static set is what remains."),
        },
    }


def tooth_faces_from_labels(scan_faces, labels, fdi: int) -> np.ndarray:
    """The faces one FDI wholly owns: all three vertices carry that label.

    ALL THREE, not any. A face with one vertex on the neighbour spans two
    teeth, and if those teeth move differently it must shear - putting it in
    either tooth's rigid set makes `moving_teeth_exact_rigid` unsatisfiable by
    construction. Used when a tooth is labelled but was never cut; a cut tooth
    has `/cut`'s own `face_mask`, which is the better source because it is the
    selection the clinician actually approved.
    """
    F = np.asarray(scan_faces, np.int64)
    lab = np.asarray(labels).astype(np.int64).reshape(-1)
    if not len(F):
        return np.zeros(0, bool)
    if lab.max(initial=-1) >= 0 and len(lab) <= int(F.max()):
        raise ValueError(f"{len(lab)} labels for a mesh of at least "
                         f"{int(F.max()) + 1} vertices")
    return (lab[F] == int(fdi)).all(axis=1)
# ---------------------------------------------------------------------------
# 2.6  attachments - the ONLY boolean in this construction
# ---------------------------------------------------------------------------

def union_attachments(verts, faces, solids, stage: int = 1) -> dict:
    """Union `solids` onto the deformed cast, then re-measure the result.

    THIS IS THE ONE BOOLEAN IN THE DEFORMATION PATH, and it is here because
    an attachment is genuinely added material - a composite button bonded to
    enamel, which no amount of blending the existing surface can produce. The
    cast itself is never booleaned: its topology is inherited from T0 and that
    is what keeps every stage's index buffer identical.

    RE-MEASURED, NOT ASSUMED. `aggregate_gate_v2` decided on the deformed cast
    BEFORE this union, and a boolean can break what it certified - s.23
    measured a union producing coincident positions wherever a solid touches
    itself, which binary STL cannot express and a reader's weld turns into a
    non-manifold edge. So the file-level and self-intersection gates are run
    again on the union's own output, and both answers are reported: what the
    cast measured, and what the file a lab receives measures.

    Refuses rather than repairs. One positive-volume body or nothing: more
    than one means an attachment is floating clear of the tooth it is meant to
    be bonded to, which is `fuse_to_crown`'s rule and the same rule s.16 sets
    for a bonded attachment.
    """
    import manifold3d as m3

    t0 = time.perf_counter()
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)
    if not solids:
        return {"applied": False, "verts": V, "faces": F,
                "reason": "no attachments on any moving tooth"}

    def _solid(v, f):
        return m3.Manifold(m3.Mesh(
            vert_properties=np.asarray(v, np.float32),
            tri_verts=np.asarray(f, np.uint32)))

    fused = _solid(V, F)
    for s in solids:
        fused = fused + _solid(s["verts"], s["faces"])

    # A tangential boolean leaves the odd inside-out shell, whose volume is
    # NEGATIVE - a void, not geometry. s.10 measured [27572.4, -1.9] mm3.
    bodies = [b for b in fused.decompose() if b.volume() > 0]
    crumbs = len(fused.decompose()) - len(bodies)
    if len(bodies) != 1:
        raise AttachmentUnionRefused(
            "attachment_not_bonded",
            {"reason": (f"the union produced {len(bodies)} positive-volume "
                        f"bodies. More than one means an attachment is not "
                        f"touching the tooth it is bonded to."),
             "positive_bodies": len(bodies), "inverted_crumbs_discarded": crumbs,
             "stage": int(stage),
             "attachments": [{"tooth_id": s.get("tooth_id"),
                              "attachment_id": s.get("attachment_id"),
                              "shape": s.get("shape")} for s in solids]})

    mesh = bodies[0].to_mesh()
    uv = np.asarray(mesh.vert_properties[:, :3], float)
    uf = np.asarray(mesh.tri_verts, np.int64)

    # The gates again, on what the boolean actually produced.
    uv32 = uv.astype(np.float32).astype(np.float64)
    blob = cg.write_binary_stl_bytes(uv32, uf)
    validation = mfg.validate_printable_stl(blob)
    file_d = _file_dict(validation)
    sx = si.self_intersection_report(uv32, uf)

    ok = (file_d.get("open") == 0 and file_d.get("nonmanifold") == 0
          and file_d.get("components") == 1 and file_d.get("winding_ok") is True
          and isinstance(file_d.get("volume"), float) and file_d["volume"] > 0
          and sx.get("measured") is True and sx.get("intersecting_pairs") == 0)

    return {
        "applied": True, "ok": bool(ok),
        "verts": uv32, "faces": uf, "blob": blob,
        "file": file_d, "self_intersection": sx,
        "attachments": [{"tooth_id": s.get("tooth_id"),
                         "attachment_id": s.get("attachment_id"),
                         "shape": s.get("shape"),
                         "volume_mm3": round(float(s.get("volume_mm3") or 0.0), 3)}
                        for s in solids],
        "positive_bodies": 1, "inverted_crumbs_discarded": int(crumbs),
        "volume_mm3": round(float(bodies[0].volume()), 3),
        "triangles": int(len(uf)),
        "seconds": round(time.perf_counter() - t0, 3),
        "note": ("the cast's own gate was decided BEFORE this union; these "
                 "are the file-level and self-intersection gates re-run on "
                 "the union's output."),
    }


class AttachmentUnionRefused(Exception):
    """The attachment union did not produce one bonded solid."""

    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason, self.detail = reason, detail
