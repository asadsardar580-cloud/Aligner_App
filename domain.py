"""The formal treatment domain: Case -> Arch -> Tooth -> Prescription -> Stage.

WHAT THIS IS AND IS NOT.

It is a typed, serialisable description of a TREATMENT. It is not a new storage
layer and it does not own geometry. The session store keeps the scan in memory
and the geometry engine keeps producing the same arrays it always has; this sits
above both and records the clinical decisions made about them.

THE THREE LAYERS, KEPT APART DELIBERATELY:

  raw scan geometry   verts/faces/graph/concavity, in SessionStore, memory-only,
                      never written to disk, never mutated after upload
  treatment state     THIS MODULE - prescriptions, FDI, C_res, staging. Small,
                      pure data, safe to persist, meaningless without the scan
  derived visuals     crown meshes, socket cups, root cones - all REDERIVABLE
                      from the two above, so none of it is persisted

The split is what makes a case file safe. Saving a prescription is saving six
numbers and a tooth id; saving the scan would be saving the patient. The first
is a treatment plan, the second is biometric data, and only one of them belongs
in a file that outlives the session.

WHY THE ARCHES ARE SEPARATE SESSIONS AND A CASE SITS ABOVE THEM. STORE.create()
makes one session per arch and nothing links them, so the server could not find
a tooth's antagonist by itself - `opposing_session_id` had to be passed in by
the client, the only party that knew both. A Case makes that relationship
explicit and server-side, which is what lets the antagonist be resolved
deterministically instead of depending on the caller remembering to send it.
"""
from __future__ import annotations

import math
import time
import uuid

import audit
from dataclasses import dataclass, field, asdict

SCHEMA_VERSION = 1

# Wheeler averages, the same four buckets the client uses (toothGizmo.js
# ROOT_DEFAULTS_MM). Duplicated here ON PURPOSE rather than imported: a case
# file must stay readable without a running frontend, and these are the numbers
# C_res was actually extrapolated along when the case was planned.
ROOT_LENGTH_MM = {"incisor": 10.0, "canine": 13.0, "premolar": 9.0, "molar": 9.0}

# Per-stage movement ceilings. HEURISTIC, not literature - the limits the
# staging estimate already used, carried here so the domain can compute a tray
# count without importing the geometry engine.
PER_STAGE_LIMITS = {
    "tip_deg": 2.0, "torque_deg": 2.0, "rotation_deg": 2.0,
    "d_md": 0.25, "d_bl": 0.25, "d_oa": 0.25,
}


def tooth_class(fdi) -> str | None:
    """FDI -> tooth type. None when segmentation never ran; never a guess."""
    if fdi is None:
        return None
    pos = int(str(int(fdi))[-1])
    if pos <= 2:
        return "incisor"
    if pos == 3:
        return "canine"
    if pos <= 5:
        return "premolar"
    return "molar"


@dataclass
class Prescription:
    """The six clinical values, ABSOLUTE FROM T0 - not increments.

    This is the whole movement. The 4x4 is always rebuildable from these plus
    the frame and C_res, which is why staging can interpolate the prescription
    and never the matrix, and why a restored tooth is re-derived rather than
    trusted as sixteen stored floats.
    """
    tip_deg: float = 0.0        # about u_BL - angulation
    torque_deg: float = 0.0     # about u_MD - inclination
    rotation_deg: float = 0.0   # about u_OA - axial
    d_md: float = 0.0           # mm, mesiodistal
    d_bl: float = 0.0           # mm, buccolingual
    d_oa: float = 0.0           # mm, occlusoapical

    def is_zero(self) -> bool:
        return not any(abs(v) > 0 for v in asdict(self).values())

    def scaled(self, f: float) -> "Prescription":
        """Stage k of N is this, scaled by k/N, REBUILT through the matrix
        solver. Never a component-wise lerp of the committed 4x4: the 3x3 block
        of (1-t)I + tR is not orthonormal for any intermediate t, so every stage
        would shear. Measured 1.68e-2 orthogonality error the wrong way against
        4.44e-16 this way.
        """
        return Prescription(**{k: v * f for k, v in asdict(self).items()})

    def stages(self, limits: dict | None = None, with_channel: bool = False):
        """Stages needed = the most demanding channel, rounded up."""
        lim = limits or PER_STAGE_LIMITS
        worst, channel = 0, None
        for k, v in asdict(self).items():
            ceiling = lim.get(k)
            if not ceiling:
                continue
            n = math.ceil(abs(v) / ceiling - 1e-9)
            if n > worst:
                worst, channel = n, k
        return (worst, channel) if with_channel else worst


@dataclass
class Tooth:
    tooth_id: str
    arch: str                              # 'upper' | 'lower'
    fdi: int | None = None                 # None when segmentation never ran
    root_length_mm: float | None = None    # what C_res was extrapolated along
    c_res: list[float] | None = None
    prescription: Prescription = field(default_factory=Prescription)
    # Review state, never inferred. 'unreviewed' is NOT 'ok'.
    review: str = "unreviewed"             # unreviewed | confirmed | flagged
    notes: str = ""

    @property
    def tooth_class(self) -> str | None:
        return tooth_class(self.fdi)

    @property
    def expected_root_mm(self) -> float | None:
        c = self.tooth_class
        return ROOT_LENGTH_MM[c] if c else None

    def root_length_matches_fdi(self) -> bool | None:
        """None when it cannot be judged, which is a third answer and not a
        failure. A shipped manifest once carried FDI 33 - a mandibular canine -
        on a 10mm root where Wheeler gives 13, because one session-wide slider
        fed every tooth.
        """
        exp = self.expected_root_mm
        if exp is None or self.root_length_mm is None:
            return None
        return abs(self.root_length_mm - exp) < 0.01


@dataclass
class Arch:
    arch: str                              # 'upper' | 'lower'
    session_id: str | None = None          # the in-memory scan, if still live
    has_occlusal_frame: bool = False
    has_segmentation: bool = False
    scan_vertex_count: int = 0
    scan_face_count: int = 0
    teeth: dict[str, Tooth] = field(default_factory=dict)


@dataclass
class Case:
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    label: str = ""                        # clinician's own note. NEVER a name.
    arches: dict[str, Arch] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    # The decision record, persisted with the plan. Never leaves the machine.
    audit_entries: list = field(default_factory=list)

    # ---- audit -------------------------------------------------------------
    def trail(self) -> "audit.AuditTrail":
        return audit.AuditTrail.from_list(self.audit_entries)

    def record(self, action: str, **fields):
        """Append to the decision record and keep it on the case."""
        t = self.trail()
        e = t.record(action, **fields)
        self.audit_entries = t.to_list()
        self.updated = time.time()
        return e

    # ---- relationships the server could not previously resolve -------------
    def opposing(self, arch: str) -> Arch | None:
        """The antagonist arch, resolved SERVER-SIDE.

        This only means anything because scanner coordinates are sacred: both
        arches sit in the same raw R^3 space, so an upper incisor driven
        lingually really does land where the lower arch is. Had either been
        re-centred on its own origin - the obvious tidy-up - this would compare
        two unrelated coordinate systems and the collision result would be
        noise that looks like a measurement.
        """
        other = "lower" if arch == "upper" else "upper"
        return self.arches.get(other)

    def opposing_session_id(self, arch: str) -> str | None:
        opp = self.opposing(arch)
        return opp.session_id if opp else None

    def tooth(self, tooth_id: str):
        for a in self.arches.values():
            if tooth_id in a.teeth:
                return a, a.teeth[tooth_id]
        return None

    def all_teeth(self) -> list[Tooth]:
        return [t for a in self.arches.values() for t in a.teeth.values()]

    # ---- staging ----------------------------------------------------------
    def stage_count(self, limits: dict | None = None) -> int:
        """Case tray count = MAX over committed teeth. The case is as long as
        its slowest tooth; averaging would under-prescribe every other one."""
        return max([t.prescription.stages(limits) for t in self.all_teeth()] or [0])

    def binding_teeth(self, limits: dict | None = None) -> list[dict]:
        """Which teeth set the tray count, and which channel binds each one. A
        clinician asking 'why is this 31 stages' needs the tooth and the
        channel, not the number."""
        rows = []
        for t in self.all_teeth():
            n, channel = t.prescription.stages(limits, with_channel=True)
            if n > 0:
                rows.append({"tooth_id": t.tooth_id, "fdi": t.fdi,
                             "stages": n, "driver": channel})
        total = max([r["stages"] for r in rows] or [0])
        for r in rows:
            r["binds_the_case"] = r["stages"] == total
        rows.sort(key=lambda r: -r["stages"])
        return rows

    # ---- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage_count"] = self.stage_count()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Case":
        v = d.get("schema_version", 0)
        if v > SCHEMA_VERSION:
            raise ValueError(
                f"Case file schema v{v} is newer than this build understands "
                f"(v{SCHEMA_VERSION}). Refusing rather than silently dropping "
                f"fields a later version added.")
        arches = {}
        for name, a in (d.get("arches") or {}).items():
            teeth = {}
            for tid, t in (a.get("teeth") or {}).items():
                p = t.get("prescription") or {}
                teeth[tid] = Tooth(
                    tooth_id=t["tooth_id"], arch=t.get("arch", name),
                    fdi=t.get("fdi"), root_length_mm=t.get("root_length_mm"),
                    c_res=t.get("c_res"), prescription=Prescription(**p),
                    review=t.get("review", "unreviewed"), notes=t.get("notes", ""))
            arches[name] = Arch(
                arch=a.get("arch", name), session_id=a.get("session_id"),
                has_occlusal_frame=a.get("has_occlusal_frame", False),
                has_segmentation=a.get("has_segmentation", False),
                scan_vertex_count=a.get("scan_vertex_count", 0),
                scan_face_count=a.get("scan_face_count", 0), teeth=teeth)
        return Case(case_id=d.get("case_id") or uuid.uuid4().hex[:12],
                    created=d.get("created", time.time()),
                    updated=d.get("updated", time.time()),
                    label=d.get("label", ""), arches=arches,
                    audit_entries=d.get("audit_entries") or [],
                    schema_version=min(v, SCHEMA_VERSION))
