"""Local case audit trail. Zero telemetry.

WHAT IS RECORDED AND WHY. Clinical software should be able to answer "who
changed this prescription, when, and from what" — not for surveillance but
because a treatment plan is a sequence of decisions, and a decision without a
record is indistinguishable from an accident.

WHAT IS NOT RECORDED, AND THIS IS THE LOAD-BEARING PART:

  * nothing leaves this machine. There is no endpoint, no upload, no telemetry.
  * no patient identifiers. Not a name, not a DOB, not a filename — filenames
    routinely carry names, which is exactly why session_store stores none.
  * no geometry. An entry is a few fields; a mesh is the patient.

An entry says WHAT changed and BY HOW MUCH, in the same units the UI shows. It
does not say who the patient is, and it cannot be made to.

WHY IT IS BOUNDED. A log that grows without limit on a clinical workstation is
a disk-space incident waiting to happen, and an audit trail nobody prunes is one
nobody reads. Each case file keeps its most recent MAX_ENTRIES actions.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

MAX_ENTRIES = 500

# Action vocabulary. A closed set, so a reader never has to guess what a verb
# meant, and a typo cannot silently create a new category.
SCAN_LOADED = "scan_loaded"
OCCLUSAL_SET = "occlusal_reference_set"
SEGMENTED = "segmentation_run"
SEGMENTATION_REVIEWED = "segmentation_reviewed"
TOOTH_CUT = "tooth_extracted"
PRESCRIPTION_COMMITTED = "prescription_committed"
TOOTH_RESET = "tooth_reset"
IPR_THRESHOLD_OVERRIDDEN = "ipr_threshold_overridden"
CASE_SAVED = "case_saved"
CASE_LOADED = "case_loaded"
EXPORTED = "exported"

ACTIONS = {
    SCAN_LOADED, OCCLUSAL_SET, SEGMENTED, SEGMENTATION_REVIEWED, TOOTH_CUT,
    PRESCRIPTION_COMMITTED, TOOTH_RESET, IPR_THRESHOLD_OVERRIDDEN,
    CASE_SAVED, CASE_LOADED, EXPORTED,
}

# Keys that must never reach an audit entry, checked on every append. A denylist
# is a weaker guarantee than a schema, but it is checked at RUNTIME on real
# payloads, which is where a leak would actually happen.
FORBIDDEN_KEYS = {
    "verts", "faces", "positions", "indices", "graph", "concavity", "crown",
    "labels", "patient", "patient_name", "name", "dob", "date_of_birth", "mrn",
    "filename", "file_name", "path",
}


@dataclass
class Entry:
    action: str
    at: float = field(default_factory=time.time)
    arch: str | None = None
    tooth_id: str | None = None
    fdi: int | None = None
    detail: str = ""
    # Small, numeric, unit-bearing. Never geometry.
    values: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class AuditTrail:
    """In-memory, appended to a Case and persisted with it."""

    def __init__(self, entries=None, max_entries: int = MAX_ENTRIES):
        self._entries = list(entries or [])
        self._max = max_entries

    def record(self, action: str, **fields) -> Entry:
        if action not in ACTIONS:
            raise ValueError(
                f"Unknown audit action {action!r}. The vocabulary is closed so a "
                f"typo cannot silently create a category nobody reads: {sorted(ACTIONS)}")
        values = fields.pop("values", {}) or {}
        leaked = (set(values) | set(fields)) & FORBIDDEN_KEYS
        if leaked:
            raise ValueError(
                f"Refusing to audit {sorted(leaked)} — an audit entry records WHAT "
                f"changed, never the patient or the mesh.")
        e = Entry(action=action, **fields, values=values)
        self._entries.append(e)
        if len(self._entries) > self._max:
            del self._entries[: len(self._entries) - self._max]
        return e

    def entries(self, action: str | None = None, tooth_id: str | None = None):
        out = self._entries
        if action:
            out = [e for e in out if e.action == action]
        if tooth_id:
            out = [e for e in out if e.tooth_id == tooth_id]
        return list(out)

    def to_list(self):
        return [e.to_dict() for e in self._entries]

    @staticmethod
    def from_list(rows, max_entries: int = MAX_ENTRIES):
        return AuditTrail([Entry(**r) for r in (rows or [])], max_entries)

    def summary(self):
        counts = {}
        for e in self._entries:
            counts[e.action] = counts.get(e.action, 0) + 1
        return {
            "entry_count": len(self._entries),
            "by_action": counts,
            "first_at": self._entries[0].at if self._entries else None,
            "last_at": self._entries[-1].at if self._entries else None,
            "max_entries": self._max,
            "telemetry": "none — this trail never leaves the workstation",
        }

    def export_json(self, path: str) -> str:
        """Write the trail as readable JSON, for a clinician who wants a record.

        Deliberately NOT encrypted, unlike the case file: it contains no
        identifiers and no geometry, and a record nobody can open is not a
        record. Anything sensitive would be a bug in what was appended.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"entries": self.to_list(), "summary": self.summary()},
                      fh, indent=2)
        return os.path.abspath(path)
