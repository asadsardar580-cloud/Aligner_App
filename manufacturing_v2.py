"""The "deform, don't cut" manufacturing construction, printed as a SOLID.

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
  * the PLANNED SURFACE that results is then voxel-solidified
    (`print_solid.solidify`) and the solid - not the planned surface - is the
    file a lab receives.

WHY THE LAST STEP (Task 2, decision of record, 24 Sep 2026). On the real
mandible the raw T0 cast crosses itself where the scan surface near the trim
rim passes through the base wall, before any tooth moves. The previous version
of this module REFUSED such a cast (`t0_cast_self_intersects`) and removed the
periphery with `clear_undercut_periphery`; both are gone. The raw cast may now
cross itself, and solidification turns it into one watertight body - whose
own self-intersection, topology and crown fidelity are gated on the RE-READ
BYTES of the file (`print_solid.measure_solid`). So:

  * the kit's gates that describe the planned surface's own DEFORMATION -
    index buffer, weights, crown rigidity, untouched/pinned vertices,
    inversion, degeneracy, implicit IPR - still apply, unchanged;
  * the two kit gates that measured the RAW cast as if it were the file
    (`no_self_intersection`, `written_file_topology`) are superseded by the
    solid's gates on the file that actually ships (SUPERSEDED_BY_SOLID).

This module is PURE. It imports no FastAPI and holds no session state, so the
whole construction can be driven from a test, a script or a request handler
without a server. `api_core` adapts it to HTTP; nothing here knows about HTTP.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

import core_geometry as cg
import deform_construction as dc
import print_solid as ps
import stage_matrix

#: Identifies the deformation construction in every manifest it writes.
CONSTRUCTION_ID = dc.CONSTRUCTION_ID
#: Identifies how the shipped FILE was made from the planned surface.
PRINT_MODEL_ID = "voxel_solidified_v1"

#: Packages whose version changes what this code DOES.
_MANIFEST_PACKAGES = ("numpy", "scipy", "manifold3d", "open3d", "trimesh",
                      "meshlib")

# ---------------------------------------------------------------------------
# The aggregate gate's vocabulary. Module constants, so a harness can never
# read a name the producer does not write (CLAUDE.md rule 9).
# ---------------------------------------------------------------------------

#: Kit gates that measured the RAW deformed cast as though it were the file.
#: The file is now the solid, gated on its own re-read bytes.
SUPERSEDED_BY_SOLID = ("no_self_intersection", "written_file_topology")
#: The kit's gates on the planned surface's deformation - all still required.
DEFORMATION_GATES = tuple(g for g in dc.REQUIRED_GATES
                          if g not in SUPERSEDED_BY_SOLID)
#: Every moved tooth's contacts: penetration <= max(0.05 mm, prescribed IPR).
CONTACT_GATE = "interproximal_contacts_within_prescription"
#: Every attachment placed on a moving tooth is inside the shipped solid.
ATTACHMENT_GATE = "attachments_in_solid"
#: A T0 export carries no movement at all.
T0_GATE = "t0_is_unmoved"

STAGE_KIND_T0 = "t0"
STAGE_KIND_MOVED = "moved"

#: The keys of the record `aggregate_gate_v3` reads. `build_t0_stage` and
#: `build_stage_v2` write them, and return the record so a test can pin that
#: the verdict shipped IS the gate's answer on it.
R_STAGE_KIND = "stage_kind"
R_MOVING_TEETH = "moving_teeth"
R_DEFORMATION = "deformation"
R_ATTACHMENTS = "attachments"
R_CONTACTS = "contacts"
R_SOLID = "solid"

#: Required, in report order, for a stage with movement ...
STAGE_GATES = DEFORMATION_GATES + (CONTACT_GATE, ATTACHMENT_GATE) + \
    ps.SOLID_GATES
#: ... and for the T0 export.
T0_GATES = (T0_GATE,) + ps.SOLID_GATES


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


@dataclass
class CasePlan:
    """Everything a stage needs, solved once per case."""

    V0: np.ndarray                     # T0 cast vertices (scan array + floor)
    F: np.ndarray                      # T0 cast faces - the SAME array every stage
    plan: dc.DeformationPlan | None    # None when nothing moves (the T0 export)
    tooth_records: dict                # tooth id -> {"frame", "c_res", "clinical"}
    policy: DeformationPolicy
    u_occ: np.ndarray                  # occlusal axis, for the model height
    #: every tooth's vertex ids on the cast, moving AND static, keyed by a
    #: name a clinician can read - the points crown fidelity is measured on.
    tooth_vertices: dict = field(default_factory=dict)
    t0_topology: dict = field(default_factory=dict)
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


# ---------------------------------------------------------------------------
# The T0 cast and the case plan
# ---------------------------------------------------------------------------

def build_t0_cast(scan_v, scan_f, arch_frame,
                  policy: DeformationPolicy = DEFAULT_POLICY, arch_curve=None):
    """`trim_to_arch` -> `build_cast_base`, the teeth still in the surface.

    NO `clear_undercut_periphery`. It re-trimmed the scan until the cast
    stopped crossing its own wall; solidification now absorbs that crossing,
    and a periphery clearance only removes anatomy the solid would have kept.
    """
    t0 = time.perf_counter()
    scan_v = np.asarray(scan_v, float)
    scan_f = np.asarray(scan_f, np.int64)
    tv, tf, trim_info = cg.trim_to_arch(scan_v, scan_f, arch_frame,
                                        margin_mm=policy.trim_margin_mm,
                                        curve=arch_curve)
    V0, F, base_info = cg.build_cast_base(
        tv, tf, arch_frame, base_thickness_mm=policy.base_thickness_mm,
        rim=trim_info.get("rim_loop"))
    return {"V0": V0, "F": np.asarray(F, np.int64), "n_trimmed": int(len(tv)),
            "rim": trim_info.get("rim_loop"), "trim_info": trim_info,
            "base_info": base_info, "seconds": time.perf_counter() - t0}


def build_case_plan(scan_v, scan_f, arch_frame, moving, static_teeth_vertices,
                    policy: DeformationPolicy = DEFAULT_POLICY,
                    arch_curve=None, static_by_tooth: dict | None = None
                    ) -> CasePlan:
    """Solve the blend once, for a whole case.

    `moving`   {tooth_id: {"vertices": ids on the SCAN array,
                           "frame": ..., "c_res": ..., "clinical": {...}}}
               - EMPTY for the T0 export, and then no blend is solved at all.
    `static_teeth_vertices`  vertex ids of every non-moving tooth.
    `static_by_tooth`        the same ids per tooth, so a crown-fidelity
                             failure can be NAMED; optional.

    PINNED = the trim rim loop, plus every appended floor vertex.
    `build_cast_base` keeps the input vertex array as a PREFIX and only
    appends, so `i >= len(trimmed verts)` identifies the floor exactly
    (AGENT_BRIEF A1.2). Those vertices carry the cast's own boundary; if the
    blend reached them the base would deform.

    THE T0 CAST IS NO LONGER REFUSED FOR CROSSING ITSELF (Task 2). A stage
    inherits T0's geometry, and the shipped file is the SOLID made from it,
    which is gated on its own bytes.
    """
    t_start = time.perf_counter()
    cast = build_t0_cast(scan_v, scan_f, arch_frame, policy, arch_curve)
    V0, F, n_trimmed = cast["V0"], cast["F"], cast["n_trimmed"]

    # --- pinned: the trim rim and everything appended below it ----------
    pinned = np.zeros(len(V0), bool)
    pinned[n_trimmed:] = True                      # floor + wall vertices
    rim = cast["rim"]
    if rim is not None and len(np.asarray(rim)):
        pinned[np.asarray(rim, np.int64)] = True

    static_ids = np.unique(np.asarray(static_teeth_vertices, np.int64)) \
        if static_teeth_vertices is not None else np.zeros(0, np.int64)
    moving_sets = {t: np.unique(np.asarray(m["vertices"], np.int64))
                   for t, m in (moving or {}).items()}

    # --- contact bands on the STATIC side, then the weights --------------
    band = np.zeros(0, np.int64)
    plan, t_weights = None, 0.0
    if moving_sets:
        if len(static_ids):
            bands = [dc.contact_band(V0, F, ids, static_ids,
                                     width_mm=policy.band_mm)
                     for ids in moving_sets.values()]
            band = np.unique(np.concatenate(bands)) if bands else band
        t0 = time.perf_counter()
        plan = dc.plan_deformation(V0, F, moving_sets, static_ids, pinned,
                                   envelope_mm=policy.envelope_mm,
                                   contact_band_vertices=band)
        t_weights = time.perf_counter() - t0

    # --- every tooth, by a readable name, for crown fidelity -------------
    teeth = {str(t): ids for t, ids in moving_sets.items()}
    if static_by_tooth:
        taken = np.zeros(len(V0), bool)
        for ids in moving_sets.values():
            taken[ids] = True
        for name, ids in static_by_tooth.items():
            ids = np.unique(np.asarray(ids, np.int64))
            ids = ids[~taken[ids]]
            if len(ids):
                teeth[str(name)] = ids
    elif len(static_ids):
        teeth["static teeth"] = static_ids

    topo = cg.manifold_report(F)
    return CasePlan(
        V0=V0, F=F, plan=plan,
        tooth_records={t: {"frame": m["frame"], "c_res": m["c_res"],
                           "clinical": m.get("clinical") or {}}
                       for t, m in (moving or {}).items()},
        policy=policy,
        u_occ=np.asarray(arch_frame["u_occ"], float),
        tooth_vertices=teeth,
        t0_topology={k: topo[k] for k in ("open_edges", "nonmanifold_edges",
                                          "total_edges")},
        diagnostics={
            "construction": CONSTRUCTION_ID,
            "print_model": PRINT_MODEL_ID,
            "cast": {"vertices": int(len(V0)), "faces": int(len(F)),
                     "trimmed_scan_vertices": int(n_trimmed),
                     "appended_floor_vertices": int(len(V0) - n_trimmed),
                     "construction": ("trim_to_arch -> build_cast_base; no "
                                      "clear_undercut_periphery")},
            "trim": {k: v for k, v in cast["trim_info"].items()
                     if k not in ("curve", "rim_loop")},
            "cast_base": {k: v for k, v in cast["base_info"].items()
                          if not isinstance(v, np.ndarray)},
            "pinned_vertices": int(pinned.sum()),
            "contact_band_vertices": int(len(band)),
            "tooth_vertex_sets": {k: int(len(v)) for k, v in teeth.items()},
            "seconds": {"cast": round(cast["seconds"], 3),
                        "weights": round(t_weights, 3),
                        "total": round(time.perf_counter() - t_start, 3)},
        })


# ---------------------------------------------------------------------------
# Step 5 - interproximal contacts, measured BEFORE solidifying
# ---------------------------------------------------------------------------

#: FDI order along each arch, distal-right to distal-left. A tooth's
#: neighbours are the nearest PRESENT teeth either side in this order - so
#: across an extraction space the neighbour is the next tooth that exists.
_ARCH_ORDER = {
    "upper": [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28],
    "lower": [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38],
    "upper_primary": [55, 54, 53, 52, 51, 61, 62, 63, 64, 65],
    "lower_primary": [85, 84, 83, 82, 81, 71, 72, 73, 74, 75],
}


def arch_neighbours(fdi: int, present) -> list:
    """The present teeth either side of `fdi` along its arch (0, 1 or 2)."""
    fdi = int(fdi)
    present = {int(x) for x in present}
    for order in _ARCH_ORDER.values():
        if fdi in order:
            i = order.index(fdi)
            out = []
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(order) and order[j] not in present:
                    j += step
                if 0 <= j < len(order):
                    out.append(order[j])
            return out
    return []


def contact_key(a, b) -> str:
    """'43-44' - the lower FDI first, so either spelling names one contact."""
    x, y = sorted((int(a), int(b)))
    return f"{x}-{y}"


def normalise_ipr_by_contact(spec) -> dict:
    """{'43-44': mm} from any of '43-44', '44-43', '43/44', '43_44'.
    Raises ValueError on a key that is not two FDI numbers or a value that
    is not a finite, non-negative number - a typo must not become 'no IPR'."""
    out = {}
    for k, val in (spec or {}).items():
        parts = str(k).replace("/", "-").replace("_", "-").split("-")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            raise ValueError(f"IPR contact {k!r} is not two FDI numbers, "
                             f"e.g. '43-44'")
        try:
            mm = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"IPR for {k!r} is not a number: {val!r}")
        if not np.isfinite(mm) or mm < 0.0:
            raise ValueError(f"IPR for {k!r} must be finite and >= 0: {mm}")
        out[contact_key(*parts)] = mm
    return out


def measure_contacts(V0, Vk, contact_spec, ipr_by_contact=None,
                     tolerance_mm: float = dc.IPR_TOLERANCE_MM) -> dict:
    """`cg.measure_interproximal_penetration` for each moved tooth against
    each neighbour, with `measure_penetration=True`.

    `contact_spec`  None when the neighbours cannot be identified (no
                    segmentation labels) - then NOTHING is measured and the
                    gate fails, because "no contact found" is not "no
                    contact". Otherwise a dict:
        {"contacts": [{"moving_fdi", "neighbour_fdi", "neighbour_moving",
                       "moving_ids", "neighbour_ids", "neighbour_faces",
                       "moving_faces", "u_md"}, ...],
         "unmeasured": [reason, ...]}   - moving teeth with no FDI, etc.

    THE NEIGHBOUR IS ITS REAL ENAMEL. A static neighbour is taken at T0
    (V0), NOT at its stage position: its contact band is released to blend,
    and the blend carves that enamel out of the moving crown's way - so
    measured against the blended surface a crown driven into its neighbour
    reads as clear (measured: a 0.30 mm push into a merged contact read
    <= 0.05 mm that way). The overlap with the enamel the patient actually
    has is the IPR the plan needs. A MOVING neighbour is taken at its stage
    position, which for its own vertices is its exact rigid transform.
    """
    ipr = normalise_ipr_by_contact(ipr_by_contact)
    if contact_spec is None:
        return {"measured": False, "tolerance_mm": tolerance_mm, "contacts": [],
                "reason": ("no segmentation labels, so a moving tooth's "
                           "neighbours cannot be identified and no contact "
                           "was measured")}
    rows, failures = [], list(contact_spec.get("unmeasured") or [])
    for c in contact_spec.get("contacts") or []:
        mids = np.asarray(c["moving_ids"], np.int64)
        nids = np.asarray(c["neighbour_ids"], np.int64)
        nf = np.asarray(c["neighbour_faces"], np.int64).reshape(-1, 3)
        name = contact_key(c["moving_fdi"], c["neighbour_fdi"])
        # the neighbour's own patch, re-indexed compactly
        remap = -np.ones(int(max(nids.max(initial=-1), nf.max(initial=-1))) + 1,
                         np.int64)
        remap[nids] = np.arange(len(nids))
        local = remap[nf] if len(nf) else nf
        local = local[(local >= 0).all(axis=1)] if len(local) else local
        nb = Vk if c.get("neighbour_moving") else V0
        r = cg.measure_interproximal_penetration(
            V0[mids], Vk[mids], c.get("moving_faces"), nb[nids], local,
            c["u_md"], threshold_mm=tolerance_mm, socket_exclusion_mm=0.0,
            measure_penetration=True)
        presc = float(ipr.get(name, 0.0))
        pen = r.get("penetration_mm")
        if pen is None:
            failures.append(f"{name}: {r.get('penetration_failure')}")
        rows.append({
            "contact": name, "moving_fdi": int(c["moving_fdi"]),
            "neighbour_fdi": int(c["neighbour_fdi"]),
            "neighbour_moving": bool(c.get("neighbour_moving")),
            "penetration_mm": pen,
            "penetration_t0_mm": r.get("penetration_t0_mm"),
            "penetration_vertices": r.get("penetration_vertices"),
            "penetration_at": r.get("penetration_at"),
            "gap_t0_mm": r.get("gap_t0_mm"), "gap_mm": r.get("gap_mm"),
            "closure_mm": r.get("max_closure_mm"),
            "prescribed_ipr_mm": presc,
            "allowed_mm": max(tolerance_mm, presc),
            "method": r.get("penetration_method"),
        })
    return {"measured": not failures, "tolerance_mm": tolerance_mm,
            "contacts": rows, "unmeasured": failures,
            "reason": "; ".join(failures) if failures else None}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and bool(np.isfinite(x))


def _contact_gate(c) -> dict:
    """Every measured contact: penetration <= max(0.05 mm, prescribed IPR
    for THAT contact). PURE - the allowance is recomputed here from the
    numbers, never read from a row's own verdict.

    The failure NAMES the contact and the millimetres (step 5).
    """
    limit = "penetration <= max(0.05 mm, IPR prescribed for that contact)"
    if not isinstance(c, dict) or c.get("measured") is not True:
        why = c.get("reason") if isinstance(c, dict) else None
        return {"ok": False, "measured": why or "contacts not measured",
                "limit": limit}
    rows = c.get("contacts")
    tol = c.get("tolerance_mm")
    if not isinstance(rows, list) or not _finite(tol):
        return {"ok": False, "measured": "contacts not measured", "limit": limit}
    bad = []
    for r in rows:
        pen, presc = r.get("penetration_mm"), r.get("prescribed_ipr_mm")
        name = r.get("contact")
        if not _finite(pen) or not _finite(presc):
            bad.append(f"{name}: not measured")
            continue
        allowed = max(float(tol), float(presc))
        if pen > allowed + 1e-9:
            bad.append(
                f"{name}: {pen:.3f} mm penetration with "
                + (f"{presc:.3f} mm IPR prescribed" if presc > 0
                   else "no IPR prescribed")
                + f" (allowed {allowed:.3f} mm)")
    summary = [{k: r.get(k) for k in ("contact", "penetration_mm",
                                      "penetration_t0_mm", "gap_t0_mm",
                                      "gap_mm", "prescribed_ipr_mm")}
               for r in rows]
    return {"ok": not bad,
            "measured": bad if bad else (summary or "no neighbouring teeth"),
            "limit": limit, "contacts": summary}


# ---------------------------------------------------------------------------
# The aggregate gate - PURE and FAIL-CLOSED
# ---------------------------------------------------------------------------

def gate_evidence(kit_gate: dict, record: dict) -> dict:
    """Each DEFORMATION gate, with the measurement it was decided on.

    `dc.aggregate_gate_v2` returns a verdict and the names that failed - not
    the numbers behind them. THE VERDICT IS NOT RECOMPUTED HERE: `ok` is read
    straight out of the kit's `failed_gates`, so this cannot disagree with the
    kit or quietly become a second implementation of it.
    """
    failed = set(kit_gate.get("failed_gates") or ())
    sr = record.get("stage_report") or {}
    pd = record.get("plan_diagnostics") or {}
    ev = {
        "index_buffer_unchanged": {
            "face_array_is_the_T0_cast_s_own": record.get("index_buffer_unchanged")},
        "weights_bounded": {k: pd.get(k) for k in
                            ("weights_bounded", "weight_min_raw", "weight_max_raw",
                             "weight_sum_max_raw", "free_vertices")},
        "moving_teeth_exact_rigid": {"per_tooth": sr.get("moving_teeth_exact_rigid")},
        "untouched_vertices_bit_identical": {
            "bit_identical": sr.get("untouched_vertices_bit_identical")},
        "pinned_vertices_bit_identical": {
            "bit_identical": sr.get("pinned_vertices_bit_identical")},
        "no_inverted_triangles": {"inverted_triangles": sr.get("inverted_triangles")},
        "no_degenerate_triangles": {"degenerate_triangles": sr.get("degenerate_triangles")},
        "implicit_ipr_within_prescription": {
            "implicit_ipr_mm": sr.get("implicit_ipr_mm"),
            "prescribed_ipr_mm": record.get("prescribed_ipr_mm"),
            "tolerance_mm": dc.IPR_TOLERANCE_MM,
            "tolerance_provenance": ("clinical rule - at or below this a closure "
                                     "is scanner noise and PDL tolerance, not "
                                     "enamel"),
        },
    }
    return {name: {"ok": name not in failed, **(ev.get(name) or {})}
            for name in DEFORMATION_GATES}


def _attachment_gate(att) -> dict:
    """Every attachment placed is INSIDE the shipped solid.

    Voxel solidification keeps the largest body only, so an attachment that
    does not touch its tooth would be discarded as a crumb and the file would
    ship WITHOUT it. This is the check that it did not happen. No attachments
    placed is a measured fact ("none"), not a missing measurement.
    """
    if not isinstance(att, dict) or "placed" not in att:
        return {"ok": False, "measured": None, "limit": "every attachment inside"}
    placed = att.get("placed")
    inside = att.get("inside")
    if placed == 0:
        return {"ok": True, "measured": "no attachments placed",
                "limit": "every attachment inside"}
    ok = (isinstance(placed, int) and isinstance(inside, list)
          and len(inside) == placed and all(x is True for x in inside))
    return {"ok": bool(ok), "measured": att, "limit": "every attachment inside"}


def aggregate_gate_v3(record: dict) -> dict:
    """The stage's ONE verdict. PURE, FAIL-CLOSED.

    record = {
      "stage_kind":  STAGE_KIND_T0 | STAGE_KIND_MOVED,
      "moving_teeth": int,
      "deformation": the kit's record for dc.aggregate_gate_v2 (moved only),
      "contacts":    measure_contacts(...)                  (moved only),
      "attachments": {"placed": n, "inside": [bool, ...]}   (moved only),
      "solid":       print_solid.measure_solid(...) or refused_measurement,
    }

    A T0 record must carry no movement; anything that is neither kind fails
    every gate it could have been judged by.
    """
    r = record or {}
    kind = r.get(R_STAGE_KIND)
    solid = ps.solid_gate(r.get(R_SOLID))
    gates = dict(solid["gates"])
    advisory = dict(solid["advisory"])

    if kind == STAGE_KIND_T0:
        required = T0_GATES
        n = r.get(R_MOVING_TEETH)
        gates[T0_GATE] = {"ok": n == 0 and not isinstance(n, bool),
                          "measured": n, "limit": 0}
    elif kind == STAGE_KIND_MOVED:
        required = STAGE_GATES
        kit_record = dict(r.get(R_DEFORMATION) or {})
        kit = dc.aggregate_gate_v2(kit_record)
        gates.update(gate_evidence(kit, kit_record))
        gates[CONTACT_GATE] = _contact_gate(r.get(R_CONTACTS))
        gates[ATTACHMENT_GATE] = _attachment_gate(r.get(R_ATTACHMENTS))
        advisory.update(kit.get("advisory") or {})
    else:
        required = STAGE_GATES
        for g in required:
            gates.setdefault(g, {"ok": False, "measured": None})
            gates[g] = {**gates[g], "ok": False}
        gates["stage_kind"] = {"ok": False, "measured": kind,
                               "limit": [STAGE_KIND_T0, STAGE_KIND_MOVED]}

    failed = [g for g in required if not (gates.get(g) or {}).get("ok")]
    if kind not in (STAGE_KIND_T0, STAGE_KIND_MOVED):
        failed = ["stage_kind"] + failed
    return {"print_ready": not failed,
            "verdict": "PRINT READY" if not failed else "NOT PRINT READY",
            "failed_gates": failed,
            "required_gates": list(required),
            "gates": {g: gates[g] for g in list(required) +
                      (["stage_kind"] if "stage_kind" in gates else [])},
            "construction": CONSTRUCTION_ID,
            "print_model": PRINT_MODEL_ID,
            "stage_kind": kind,
            "advisory": advisory}


def describe_failures(gate: dict) -> list:
    """One readable line per failed gate - name, measured value, limit."""
    out = []
    for name in gate.get("failed_gates") or []:
        g = (gate.get("gates") or {}).get(name) or {}
        meas = g.get("measured")
        if meas is None:
            meas = {k: v for k, v in g.items() if k != "ok"} or "not measured"
        text = f"{name}: measured {meas}"
        if "limit" in g:
            text += f", limit {g['limit']}"
        out.append(text[:400])
    return out


# ---------------------------------------------------------------------------
# The solid, shared by the T0 export and every moved stage
# ---------------------------------------------------------------------------

def _occupancy(points, verts, faces):
    """Which points are INSIDE a closed mesh (Open3D ray parity)."""
    import open3d as o3d
    pts = np.atleast_2d(np.asarray(points, np.float32))
    if not len(pts) or not len(faces):
        return np.zeros(len(pts), bool)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(verts, np.float32)),
                     o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return sc.compute_occupancy(o3d.core.Tensor(pts)).numpy() > 0.5


def _solid_for(case_plan: CasePlan, Vk, shells=()):
    """Planned surface (+ attachment shells) -> solid -> bytes -> measurements.

    `shells`: closed attachment solids [(verts, faces), ...], already carried
    by their tooth's stage matrix. They join the SAME voxelisation as the
    cast, so an attachment becomes part of the solid by the same operation
    that makes the solid - no boolean. A tooth vertex buried under one is
    legitimately covered, so it is left out of crown fidelity and COUNTED.
    """
    t0 = time.perf_counter()
    Vk32 = np.asarray(Vk, float).astype(np.float32).astype(np.float64)
    Vin, Fin = [Vk32], [case_plan.F]
    off = len(Vk32)
    for sv, sf in shells:
        sv = np.asarray(sv, float).astype(np.float32).astype(np.float64)
        Vin.append(sv)
        Fin.append(np.asarray(sf, np.int64) + off)
        off += len(sv)
    Vin, Fin = np.vstack(Vin), np.vstack(Fin)

    names = sorted(case_plan.tooth_vertices)
    ids = (np.concatenate([case_plan.tooth_vertices[n] for n in names])
           if names else np.zeros(0, np.int64))
    labels = (np.concatenate([[n] * len(case_plan.tooth_vertices[n])
                              for n in names]) if names else np.zeros(0, str))
    pts = np.asarray(Vk, float)[ids] if len(ids) else np.zeros((0, 3))
    covered = np.zeros(len(pts), bool)
    for sv, sf in shells:
        covered |= _occupancy(pts, sv, sf)

    blob, srep = None, None
    try:
        Vs, Fs, srep = ps.solidify(Vin, Fin)
    except ps.SolidifyRefused as e:
        measured = ps.refused_measurement(e)
    else:
        blob = cg.write_binary_stl_bytes(Vs, Fs)
        probes = [np.asarray(sv, float).mean(axis=0) for sv, _ in shells]
        measured = ps.measure_solid(blob, pts[~covered], ids[~covered],
                                    labels[~covered], case_plan.u_occ)
        measured["crown_points_covered_by_attachments"] = int(covered.sum())
        # Attachment presence, on the RE-READ bytes like every other number.
        inside = []
        if probes:
            import stl_io
            pv, pf = stl_io.parse_stl_bytes(blob)
            wv, wf, _ = cg.weld_vertices(pv, pf)
            inside = [bool(x) for x in _occupancy(np.array(probes), wv, wf)]
        measured["attachment_centroids_inside"] = inside
    return {"blob": blob, "solid": measured, "solidify": srep,
            "shells": len(shells),
            "seconds": round(time.perf_counter() - t0, 3)}


def _stage_manifest(case_plan, stage, total, kind, gate, solid, extra):
    m = solid["solid"]
    srep = solid["solidify"] or {}
    blob = solid["blob"]
    return {
        "construction": CONSTRUCTION_ID,
        "print_model": PRINT_MODEL_ID,
        "stage": int(stage), "stages": int(total), "stage_kind": kind,
        "policy": case_plan.policy.as_dict(),
        "gate": {"verdict": gate["verdict"], "print_ready": gate["print_ready"],
                 "failed_gates": gate["failed_gates"],
                 "failures": describe_failures(gate),
                 "gates": gate["gates"]},
        "advisory": gate["advisory"],
        "crown_deviation": {
            "p95_mm": m.get(ps.K_CROWN_P95), "max_mm": m.get(ps.K_CROWN_MAX),
            "limit_p95_mm": ps.CROWN_DEVIATION_P95_LIMIT_MM,
            "limit_max_mm": ps.CROWN_DEVIATION_MAX_LIMIT_MM,
            "points": m.get(ps.K_CROWN_POINTS),
            "points_covered_by_attachments":
                m.get("crown_points_covered_by_attachments"),
            "worst_locations": m.get(ps.K_CROWN_WORST),
            "failure": m.get(ps.K_CROWN_FAILURE)},
        "triangles": m.get(ps.K_TRIANGLES),
        "solidify": srep,
        "solid_measurements": {k: v for k, v in m.items()
                               if k != ps.K_CROWN_WORST},
        "meshlib_version": ps.meshlib_version(),
        "stl_sha256": hashlib.sha256(blob).hexdigest() if blob else None,
        "stl_bytes": int(len(blob)) if blob else 0,
        "git_commit": _git_commit(),
        "packages": _package_versions(),
        **extra,
    }


# ---------------------------------------------------------------------------
# The T0 export - no movement
# ---------------------------------------------------------------------------

def build_t0_stage(case_plan: CasePlan) -> dict:
    """Stage 0: the T0 cast itself, solidified and gated. Nothing moves, so
    there is no deformation to gate and no contact to measure - only the
    solid, and `t0_is_unmoved` pins that nothing did."""
    t_start = time.perf_counter()
    solid = _solid_for(case_plan, case_plan.V0)
    record = {R_STAGE_KIND: STAGE_KIND_T0,
              R_MOVING_TEETH: int(len(case_plan.tooth_records)),
              R_SOLID: solid["solid"]}
    gate = aggregate_gate_v3(record)
    manifest = _stage_manifest(case_plan, 0, 0, STAGE_KIND_T0, gate, solid, {
        "seconds": {"solid": solid["seconds"],
                    "total": round(time.perf_counter() - t_start, 3)}})
    return {"blob": solid["blob"], "gate": gate, "manifest": manifest,
            "solid": solid["solid"], "solidify": solid["solidify"],
            "report": None, "record": record}


# ---------------------------------------------------------------------------
# A moved stage
# ---------------------------------------------------------------------------

def build_stage_v2(case_plan: CasePlan, stage_matrices: dict,
                   prescribed_ipr: float = 0.0, stage: int = 1,
                   total_stages: int = 1, attachment_shells=(),
                   contact_spec=None, ipr_by_contact=None) -> dict:
    """One stage: planned surface, measurements, solid, bytes, verdict. Pure.

    The order: positions -> the kit's construction report -> the
    interproximal contacts (BEFORE solidifying: the solid unions whatever
    overlaps, so a crown driven into its neighbour is only visible on the
    planned surface) -> solidify the planned surface (with any attachment
    shells) -> write -> measure the RE-READ bytes -> the aggregate gate.

    `ipr_by_contact` {'43-44': mm}. It also authorises the kit's implicit-IPR
    band on the teeth it names: the band gate is judged against the larger of
    `prescribed_ipr` and any IPR prescribed at a contact of a moving tooth.
    """
    if case_plan.plan is None:
        raise ValueError("this case plan has no movement; use build_t0_stage")
    t_start = time.perf_counter()
    V0, F = case_plan.V0, case_plan.F
    plan = case_plan.plan

    # 1. positions
    t0 = time.perf_counter()
    Vk = dc.stage_positions(plan, V0, stage_matrices, apply=cg.apply_matrix)
    t_pos = time.perf_counter() - t0

    # 2. the kit's construction report
    t0 = time.perf_counter()
    report = dc.stage_report(plan, V0, Vk, F, stage_matrices,
                             apply=cg.apply_matrix)
    t_report = time.perf_counter() - t0

    # 3. contacts, on the planned surface
    t0 = time.perf_counter()
    contacts = measure_contacts(V0, Vk, contact_spec, ipr_by_contact,
                                tolerance_mm=case_plan.policy.ipr_tolerance_mm)
    t_contacts = time.perf_counter() - t0
    ipr = normalise_ipr_by_contact(ipr_by_contact)
    moving_fdis = {str(c["moving_fdi"]) for c in
                   ((contact_spec or {}).get("contacts") or [])}
    band_presc = max([float(prescribed_ipr or 0.0)] +
                     [mm for k, mm in ipr.items()
                      if set(k.split("-")) & moving_fdis])

    # 4-6. the solid, its bytes, and their measurements
    solid = _solid_for(case_plan, Vk, attachment_shells)

    # 6. the gate
    kit_record = {
        "plan_diagnostics": plan.diagnostics,
        "stage_report": report,
        # The face array is the T0 cast's own, never rebuilt - that is what
        # makes the planned surface's topology inherited, not re-derived.
        "index_buffer_unchanged": bool(F is case_plan.F),
        "prescribed_ipr_mm": float(band_presc),
    }
    record = {R_STAGE_KIND: STAGE_KIND_MOVED,
              R_MOVING_TEETH: int(len(plan.tooth_sets)),
              R_DEFORMATION: kit_record,
              R_CONTACTS: contacts,
              R_ATTACHMENTS: {
                  "placed": int(len(attachment_shells)),
                  "inside": solid["solid"].get("attachment_centroids_inside")},
              R_SOLID: solid["solid"]}
    gate = aggregate_gate_v3(record)

    manifest = _stage_manifest(case_plan, stage, total_stages,
                               STAGE_KIND_MOVED, gate, solid, {
        "weight_diagnostics": plan.diagnostics,
        "tooth_matrices": {str(t): np.asarray(M, float).tolist()
                           for t, M in stage_matrices.items()},
        "prescribed_ipr_mm": float(prescribed_ipr or 0.0),
        "ipr_by_contact_mm": ipr,
        "band_ipr_allowance_mm": float(band_presc),
        "contacts": contacts,
        "planned_surface": {k: v for k, v in report.items()
                            if k != "moved_faces_mask"},
        "planned_surface_digest": dc.stage_digest(
            Vk.astype(np.float32).astype(np.float64), F),
        "seconds": {"positions": round(t_pos, 3),
                    "stage_report": round(t_report, 3),
                    "contacts": round(t_contacts, 3),
                    "solid": solid["seconds"],
                    "total": round(time.perf_counter() - t_start, 3)},
    })
    return {"blob": solid["blob"], "planned_verts": Vk, "faces": F,
            "gate": gate, "manifest": manifest, "report": report,
            "solid": solid["solid"], "solidify": solid["solidify"],
            "contacts": contacts, "record": record}


# ---------------------------------------------------------------------------
# Stage matrices - the SAME derivation the collar path uses
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

    # Per tooth too, with the moving-claimed vertices removed, so a crown
    # fidelity failure can name the tooth it is on.
    static_named = {t: ids[~claimed[ids]] if len(ids) else ids
                    for t, ids in static_by_tooth.items()}

    return {
        "moving": disjoint,
        "static": static_ids,
        "static_by_tooth": static_named,
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
    # A label array shorter than the mesh must be REFUSED, not indexed. The
    # first version of this guard was `lab.max(initial=-1) >= 0 and ...`,
    # which an EMPTY array skips - and `lab[F]` then raises IndexError, an
    # opaque 500 about a correspondence problem the caller can act on.
    if len(lab) <= int(F.max()):
        raise ValueError(f"{len(lab)} labels for a mesh of at least "
                         f"{int(F.max()) + 1} vertices")
    return (lab[F] == int(fdi)).all(axis=1)
