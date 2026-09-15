"""Structured clinical validation: results that carry their numbers.

WHY THIS EXISTS. The API's success responses were richly structured — axis
source, deviation, reconciliation, socket diagnostics, crown dimensions — while
every FAILURE collapsed to a bare string in an HTTPException. That is exactly
backwards: the moment a clinician most needs the measured numbers is the moment
the software refuses to proceed. A UI cannot render "widen the selection" into
anything actionable; it can render "the selection is 61% gingiva, threshold 50%".

TWO RULES THIS MODULE ENFORCES.

1. NEVER REDUCE A VERDICT TO A BOOLEAN. A check yields a message, the metrics
   it measured, and the threshold it compared against — so the reader can
   disagree with the threshold rather than only with the outcome.

2. EVERY THRESHOLD DECLARES ITS PROVENANCE. `HEURISTIC` means someone picked a
   number that worked on the cases to hand; `LITERATURE` means it traces to
   published anatomy. They are not the same kind of claim and must never be
   rendered as though they were. Nothing here is clinically validated: a check
   passing means the geometry is self-consistent, not that a movement is safe.
"""
from __future__ import annotations

import math

# --- provenance of a threshold -------------------------------------------
HEURISTIC = "software heuristic"      # chosen by measurement on available cases
LITERATURE = "literature/reference"   # traceable to published anatomical data

# --- severity -------------------------------------------------------------
ERROR = "error"      # refuse the operation
WARNING = "warning"  # proceed, but say so
INFO = "info"        # measured, no judgement

# --- collision / proximity states ----------------------------------------
# `NOT_CHECKED` is a first-class state and is NOT a synonym for CLEAR. The
# antagonist check is skipped whenever the opposing arch was never loaded, and
# reporting that as "no interference" would be a clinical claim the software
# never made.
NOT_CHECKED = "NOT_CHECKED"
CLEAR = "CLEAR"
COLLISION_WARNING = "WARNING"
INTERFERENCE = "INTERFERENCE"
COMPUTATION_ERROR = "COMPUTATION_ERROR"


def threshold(value, units, basis, note=""):
    """A comparison value that knows where it came from."""
    return {"value": value, "units": units, "basis": basis, "note": note}


class Result:
    """Accumulates checks. Serialises to {ok, errors, warnings, info, metrics}."""

    def __init__(self, subject: str):
        self.subject = subject
        self.checks: list[dict] = []
        self.metrics: dict = {}

    def add(self, check_id, ok, severity, message, metrics=None, limit=None):
        self.checks.append({
            "id": check_id,
            "ok": bool(ok),
            "severity": severity,
            "message": message,
            "metrics": metrics or {},
            "threshold": limit,
        })
        if metrics:
            self.metrics.update(metrics)
        return self

    def error(self, check_id, ok, message, metrics=None, limit=None):
        return self.add(check_id, ok, ERROR, message, metrics, limit)

    def warn(self, check_id, ok, message, metrics=None, limit=None):
        return self.add(check_id, ok, WARNING, message, metrics, limit)

    def note(self, check_id, message, metrics=None):
        return self.add(check_id, True, INFO, message, metrics, None)

    @property
    def errors(self):
        return [c for c in self.checks if c["severity"] == ERROR and not c["ok"]]

    @property
    def warnings(self):
        return [c for c in self.checks if c["severity"] == WARNING and not c["ok"]]

    @property
    def ok(self):
        return not self.errors

    def as_dict(self):
        return {
            "subject": self.subject,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": [c for c in self.checks if c["severity"] == INFO],
            "metrics": self.metrics,
            # So a caller never has to re-derive the headline.
            "summary": self.summary(),
        }

    def summary(self):
        if self.errors:
            return self.errors[0]["message"]
        if self.warnings:
            return f"{len(self.warnings)} warning(s): {self.warnings[0]['message']}"
        return "All checks passed."


# =========================================================================
# Pre-cut validation
# =========================================================================

# Selection/geometry limits. All HEURISTIC unless marked otherwise — they were
# chosen from the cases in this repo, not from published data.
MIN_SELECTED_FACES = threshold(
    1, "faces", HEURISTIC, "a selection covering no complete face cannot be cut")
MAX_GINGIVA_FRACTION = threshold(
    0.50, "fraction", HEURISTIC,
    "over half the selected vertices labelled gingiva usually means the flood escaped the crown")
MIN_LARGEST_COMPONENT_FRACTION = threshold(
    0.80, "fraction", HEURISTIC,
    "the non-largest islands are discarded silently; below this, enough is discarded to change the crown")
MAX_SECOND_LABEL_FRACTION = threshold(
    0.25, "fraction", HEURISTIC,
    "a selection spanning two labelled teeth produces a crown that is neither")

# Root length. Wheeler averages run 9-13mm by tooth type; the band here is
# deliberately wider than any single tooth to allow clinical judgement, and
# exists to catch 0, negatives and typos rather than to prescribe anatomy.
ROOT_LENGTH_MIN = threshold(4.0, "mm", LITERATURE,
                            "shorter than any erupted permanent root (Wheeler: 9-13mm typical)")
ROOT_LENGTH_MAX = threshold(30.0, "mm", LITERATURE,
                            "longer than any human root; a typo, not a prescription")

# Crown dimensions, for reporting. Ranges are broad on purpose.
CROWN_HEIGHT_BAND = threshold((4.0, 15.0), "mm", LITERATURE,
                              "clinical crown height, incisor through molar")


def validate_selection(n_selected_vertices, n_vertices, n_selected_faces,
                       largest_component_faces=None, label_histogram=None,
                       gingiva_label=0):
    """Everything checkable about a selection BEFORE any geometry is built.

    `label_histogram` maps FDI label -> count over the selected vertices, when
    segmentation has run. Absent, the label checks are skipped and said to be
    skipped rather than silently passed.
    """
    r = Result("selection")

    ids_ok = n_selected_vertices > 0
    r.error("selection_non_empty", ids_ok,
            "No vertices selected." if not ids_ok else "Selection is non-empty.",
            {"selected_vertices": int(n_selected_vertices),
             "mesh_vertices": int(n_vertices)})

    faces_ok = n_selected_faces >= MIN_SELECTED_FACES["value"]
    r.error("selection_covers_faces", faces_ok,
            ("Selection covers no complete face; widen it."
             if not faces_ok else f"{int(n_selected_faces)} complete faces selected."),
            {"selected_faces": int(n_selected_faces)}, MIN_SELECTED_FACES)

    if largest_component_faces is not None and n_selected_faces > 0:
        frac = largest_component_faces / float(n_selected_faces)
        ok = frac >= MIN_LARGEST_COMPONENT_FRACTION["value"]
        r.warn("selection_connected", ok,
               (f"Selection is in pieces: the largest connected region is "
                f"{frac*100:.0f}% of it, and the rest is discarded."
                if not ok else f"Selection is {frac*100:.0f}% one connected region."),
               {"largest_component_fraction": round(float(frac), 4),
                "discarded_faces": int(n_selected_faces - largest_component_faces)},
               MIN_LARGEST_COMPONENT_FRACTION)

    if label_histogram:
        total = sum(label_histogram.values()) or 1
        ging = label_histogram.get(gingiva_label, 0) / total
        ok = ging <= MAX_GINGIVA_FRACTION["value"]
        r.warn("selection_is_crown_not_gingiva", ok,
               (f"{ging*100:.0f}% of the selection is labelled gingiva."
                if not ok else f"{(1-ging)*100:.0f}% of the selection is labelled tooth."),
               {"gingiva_fraction": round(float(ging), 4)}, MAX_GINGIVA_FRACTION)

        teeth = {k: v for k, v in label_histogram.items() if k != gingiva_label}
        if len(teeth) >= 2:
            ranked = sorted(teeth.values(), reverse=True)
            second = ranked[1] / float(sum(teeth.values()))
            ok2 = second <= MAX_SECOND_LABEL_FRACTION["value"]
            r.warn("selection_is_one_tooth", ok2,
                   (f"Selection spans {len(teeth)} labelled teeth; the second covers "
                    f"{second*100:.0f}% of it. The crown would be neither tooth."
                    if not ok2 else f"Selection is dominated by one labelled tooth."),
                   {"labels_present": sorted(int(k) for k in teeth),
                    "second_label_fraction": round(float(second), 4)},
                   MAX_SECOND_LABEL_FRACTION)
    else:
        r.note("selection_label_checks_skipped",
               "Segmentation has not run, so tooth-identity checks were not performed.")
    return r


def validate_root_length(mm):
    r = Result("root_length")
    finite = isinstance(mm, (int, float)) and math.isfinite(mm)
    r.error("root_length_finite", finite,
            "Root length must be a finite number." if not finite else "Root length is finite.",
            {"root_length_mm": (None if not finite else float(mm))})
    if finite:
        lo, hi = ROOT_LENGTH_MIN["value"], ROOT_LENGTH_MAX["value"]
        ok = lo <= mm <= hi
        r.error("root_length_in_band", ok,
                (f"Root length {mm:.1f}mm is outside {lo:.0f}-{hi:.0f}mm. C_res is extrapolated "
                 f"along the long axis by exactly this distance, so a wrong value is a wrong pivot."
                 if not ok else f"Root length {mm:.1f}mm is plausible."),
                {"root_length_mm": float(mm)},
                ROOT_LENGTH_MIN if mm < lo else ROOT_LENGTH_MAX)
    return r


def validate_landmarks(mesial_pt, distal_pt):
    """Mesial/distal must exist, be 3D, be finite, and not coincide."""
    r = Result("landmarks")
    for name, pt in (("mesial", mesial_pt), ("distal", distal_pt)):
        shaped = hasattr(pt, "__len__") and len(pt) == 3
        r.error(f"{name}_is_3d", shaped,
                f"{name}_pt must be [x, y, z]." if not shaped else f"{name}_pt is 3D.")
        if shaped:
            fin = all(isinstance(c, (int, float)) and math.isfinite(c) for c in pt)
            r.error(f"{name}_is_finite", fin,
                    f"{name}_pt contains a non-finite coordinate." if not fin
                    else f"{name}_pt is finite.")
    if (hasattr(mesial_pt, "__len__") and hasattr(distal_pt, "__len__")
            and len(mesial_pt) == 3 and len(distal_pt) == 3):
        try:
            d = math.dist(list(map(float, mesial_pt)), list(map(float, distal_pt)))
        except (TypeError, ValueError):
            return r
        # The mesiodistal axis is built from this difference. Coincident points
        # make it undefined; core_geometry refuses, but as a 500 rather than a
        # message anyone can act on.
        sep = threshold(0.5, "mm", HEURISTIC,
                        "below this the mesiodistal axis is numerically undefined")
        ok = d >= sep["value"]
        r.error("landmarks_separated", ok,
                (f"Mesial and distal points are {d:.2f}mm apart; the mesiodistal axis "
                 f"cannot be derived from them." if not ok
                 else f"Landmarks are {d:.1f}mm apart."),
                {"mesiodistal_span_mm": round(float(d), 3)}, sep)
    return r


def validate_pivot(depth_mm, lateral_mm, max_lateral_mm):
    """C_res must be finite AND inside the alveolus.

    Finiteness is checked FIRST and separately, because every comparison against
    NaN is False — `depth <= 0 or lateral > max_lateral` silently passes a NaN
    pivot straight through, which is how a NaN reaches a transform matrix.
    """
    r = Result("pivot")
    finite = all(isinstance(x, (int, float)) and math.isfinite(x)
                 for x in (depth_mm, lateral_mm, max_lateral_mm))
    r.error("cres_finite", finite,
            ("C_res is not finite — the long axis or the root length produced NaN."
             if not finite else "C_res is finite."),
            {"depth_mm": None if not finite else round(float(depth_mm), 3),
             "lateral_mm": None if not finite else round(float(lateral_mm), 3)})
    if not finite:
        return r

    lim = threshold(round(float(max_lateral_mm), 3), "mm", HEURISTIC,
                    "derived from the 20 degree arch-axis clamp plus 1mm")
    ok = depth_mm > 0 and lateral_mm <= max_lateral_mm
    r.error("cres_in_alveolus", ok,
            (f"Derived pivot is not inside the alveolus: C_res sits {depth_mm:.1f}mm apical of "
             f"the cervical margin and {lateral_mm:.1f}mm off to the side (limit "
             f"{max_lateral_mm:.1f}mm). Rotating about it would swing the tooth through the arch "
             f"rather than seating it in the socket."
             if not ok else f"C_res is {depth_mm:.1f}mm apical, {lateral_mm:.1f}mm lateral."),
            {"depth_mm": round(float(depth_mm), 3),
             "lateral_mm": round(float(lateral_mm), 3)}, lim)
    return r


def merge(*results, subject="cut"):
    """Combine several Results into one payload."""
    out = Result(subject)
    for r in results:
        out.checks.extend(r.checks)
        out.metrics.update(r.metrics)
    return out


# =========================================================================
# Collision / proximity state
# =========================================================================

def occlusion_state(checked, max_penetration_mm, threshold_mm, error=None):
    """Map a raw antagonist measurement onto an explicit state.

    The distinction that matters: `checked=False` yields NOT_CHECKED, never
    CLEAR. The opposing arch simply was not loaded, and the software has no
    opinion about interference it never looked for.
    """
    if error:
        return {"state": COMPUTATION_ERROR, "detail": str(error),
                "method": "nearest-vertex signed distance", "checked": False}
    if not checked:
        return {"state": NOT_CHECKED,
                "detail": "No opposing arch is loaded, so no antagonist check was performed. "
                          "This is NOT a finding of no interference.",
                "method": None, "checked": False}

    pen = float(max_penetration_mm or 0.0)
    if pen > threshold_mm:
        state = INTERFERENCE
        detail = (f"Trajectory penetrates the antagonist by {pen:.2f}mm "
                  f"(threshold {threshold_mm:.2f}mm).")
    elif pen > 0:
        state = COLLISION_WARNING
        detail = f"Contact with the antagonist measured at {pen:.2f}mm, below the threshold."
    else:
        state = CLEAR
        detail = "No penetration of the antagonist detected at the sampled vertices."

    return {
        "state": state,
        "detail": detail,
        "checked": True,
        "max_penetration_mm": round(pen, 3),
        "threshold_mm": float(threshold_mm),
        "method": "nearest-vertex signed distance",
        # Stated every time, because the number looks more exact than it is.
        "limitation": "APPROXIMATE. Distance to the nearest opposing VERTEX, signed by that "
                      "vertex's normal — not a signed distance field, and it cannot see a true "
                      "surface intersection between vertices. A real SDF needs a closed mesh and "
                      "an intraoral scan is an open shell. This is a warning, never a gate.",
    }
