"""Space analysis: crown widths against anatomy, and interproximal contact.

WHAT THIS IS FOR. Two questions a clinician asks before moving anything:

  "Is this actually one tooth?"   — a crown measuring 14mm mesiodistally where
                                    Wheeler gives 7 is two teeth merged by the
                                    segmenter, not an unusual patient.
  "Is there room to move it?"     — how much clearance exists between adjacent
                                    crowns right now, and how much a planned
                                    movement would consume.

NOTHING HERE PRESCRIBES. It reports measurements and flags implausibility. IPR
is a clinical decision about enamel reduction on a real patient; this software
measures a gap in a mesh. Those are different things and the second must never
be printed as though it were the first — so every number is reported with its
method and its limitation, and no threshold here blocks anything.

WHY THE MEASURING MACHINERY WAS ALREADY HERE AND UNUSED. core_geometry has
carried `local_arch_tangent`, `measure_mesiodistal_width` and
`validate_against_anatomy` — with Wheeler's tables — since the caliper work, and
until now they were called only by test_caliper.py. This module is the wiring,
not new geometry: the arithmetic is the arithmetic that was already tested.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import core_geometry as cg
import validation

# The directive's ceiling. core_geometry.MD_WIDTH_TOLERANCE_MM stays 2.0 for the
# callers that already depend on it; this is the value THIS analysis applies, and
# it is editable per request because it is a judgement about how much individual
# variation to tolerate, not a constant of nature. Wheeler reports roughly +/-1mm
# on incisors and +/-1.5mm on molars.
DEFAULT_WIDTH_TOLERANCE_MM = 1.5

# Below this, a "gap" is scanner noise rather than anatomy. Intraoral scanners
# resolve to roughly 20-50 microns and adjacent enamel in real contact still
# measures a small positive separation in a triangulated mesh.
NOISE_FLOOR_MM = 0.05

# At or under this, two crowns are touching in the clinical sense.
CONTACT_MM = 0.30


def _fdi_position(fdi) -> int | None:
    """FDI -> position 1..8 within its quadrant. 8s have no Wheeler entry."""
    if fdi is None:
        return None
    p = int(str(int(fdi))[-1])
    return p if 1 <= p <= 7 else None


def crown_widths(teeth: list[dict], arch: str, occlusal_axis, arch_centre,
                 tolerance_mm: float = DEFAULT_WIDTH_TOLERANCE_MM) -> dict:
    """Mesiodistal width per crown, measured and compared against Wheeler.

    `teeth` is a list of {tooth_id, fdi, verts}. The width is taken on the
    occlusal 40% of the crown, which is not a fudge factor: clinical
    mesiodistal width is measured at the CONTACT POINTS, and including the
    cervical flare would inflate every tooth.
    """
    rows, measurements = [], []
    for t in teeth:
        v = np.asarray(t["verts"], float)
        if len(v) < 4:
            continue
        centroid = v.mean(axis=0)
        pos = _fdi_position(t.get("fdi"))
        raw = cg.measure_mesiodistal_width(v, centroid, arch_centre, occlusal_axis)
        if raw is None:
            # local_arch_tangent is undefined when the crown centroid sits ON the
            # arch centre: the radius has no direction, so there is no per-tooth
            # mesiodistal axis to project onto. Real arches never do this, but a
            # single-tooth fixture does, and returning 0.0 would read as a
            # measured zero-width crown rather than an absent measurement.
            rows.append({"tooth_id": t["tooth_id"], "fdi": t.get("fdi"),
                         "position": pos, "width_mm": None, "status": "UNMEASURABLE",
                         "detail": "The crown centroid coincides with the arch centre, "
                                   "so the mesiodistal direction is undefined here."})
            continue
        width = float(raw)
        rows.append({"tooth_id": t["tooth_id"], "fdi": t.get("fdi"),
                     "position": pos, "width_mm": round(width, 2)})
        if pos is not None:
            measurements.append((pos, width))

    jaw = "maxillary" if arch in ("upper", "maxillary") else "mandibular"
    verdict = (cg.validate_against_anatomy(measurements, arch=jaw,
                                           tolerance_mm=tolerance_mm)
               if measurements else
               {"rows": [], "n_ok": 0, "n_scored": 0, "pass_rate": None,
                "reliable": False})

    # Attach the per-tooth verdict back onto the rows the client renders.
    by_pos = {r["position"]: r for r in verdict["rows"]}
    for r in rows:
        if r.get("status") == "UNMEASURABLE":
            continue
        w = by_pos.get(r["position"])
        if r["position"] is None:
            r["status"] = "UNKNOWN"
            r["detail"] = "No FDI — segmentation has not identified this tooth."
            continue
        if not w or w.get("expected") is None:
            r["status"] = "UNKNOWN"
            r["detail"] = "No Wheeler average for this position (third molars)."
            continue
        r["expected_mm"] = w["expected"]
        r["delta_mm"] = round(r["width_mm"] - w["expected"], 2)
        ok = abs(r["delta_mm"]) <= tolerance_mm
        r["status"] = "PASS" if ok else "REVIEW"
        r["detail"] = ("within Wheeler +/-%.1fmm" % tolerance_mm if ok else
                       ("%.1fmm WIDER than Wheeler — often two teeth merged"
                        % r["delta_mm"] if r["delta_mm"] > 0 else
                        "%.1fmm narrower than Wheeler — often a split or partial capture"
                        % abs(r["delta_mm"])))

    return {
        "teeth": rows,
        "tolerance_mm": tolerance_mm,
        "tolerance_basis": validation.LITERATURE,
        "n_scored": verdict["n_scored"],
        "n_within_tolerance": verdict["n_ok"],
        # reliable is False unless a full dentition was scored. Quoting a pass
        # rate over 4 of 14 teeth would read as a grade for the whole arch.
        "reliable": bool(verdict.get("reliable")),
        "measurement": "occlusal 40% extent along the per-tooth arch tangent",
        "limitation": "Wheeler averages bound PLAUSIBILITY, not truth. An individual "
                      "patient legitimately differs; a REVIEW flag means look at the "
                      "segmentation, not that the tooth is abnormal.",
    }


def interproximal_contacts(teeth: list[dict], noise_floor_mm: float = NOISE_FLOOR_MM,
                           contact_mm: float = CONTACT_MM,
                           max_pairs_distance_mm: float = 12.0) -> dict:
    """Closest approach between every plausibly-adjacent pair of crowns.

    A cKDTree per crown, queried with the neighbour's vertices. This is a
    VERTEX-TO-VERTEX minimum, not a surface-to-surface distance: on a
    triangulated mesh the true closest points usually lie inside faces, so the
    figure is a slight overestimate of the real gap. Stated on every row rather
    than buried, because 0.22mm reads far more precise than it is.
    """
    prepared = []
    for t in teeth:
        v = np.asarray(t["verts"], float)
        if len(v) < 4:
            continue
        prepared.append({"tooth_id": t["tooth_id"], "fdi": t.get("fdi"),
                         "verts": v, "centroid": v.mean(axis=0),
                         "tree": cKDTree(v)})

    rows = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            a, b = prepared[i], prepared[j]
            # Cheap reject: crowns further apart than a molar's width cannot be
            # in interproximal contact, and pairing all-with-all is O(n^2) trees.
            if np.linalg.norm(a["centroid"] - b["centroid"]) > max_pairs_distance_mm:
                continue
            d, _ = a["tree"].query(b["verts"], workers=-1)
            gap = float(d.min())

            if gap <= noise_floor_mm:
                state, detail = "CONTACT", ("touching within scanner noise "
                                            f"(<= {noise_floor_mm}mm)")
            elif gap <= contact_mm:
                state, detail = "CONTACT", "in clinical contact"
            else:
                state, detail = "CLEAR", "separated"

            label = (f"{a['fdi']}-{b['fdi']}" if a.get("fdi") and b.get("fdi")
                     else f"{a['tooth_id'][:4]}-{b['tooth_id'][:4]}")
            rows.append({
                "pair": label,
                "tooth_ids": [a["tooth_id"], b["tooth_id"]],
                "fdi": [a.get("fdi"), b.get("fdi")],
                "gap_mm": round(gap, 3),
                "state": state,
                "detail": detail,
                "readout": f"IPR Contact {label}: {gap:.2f} mm",
            })

    rows.sort(key=lambda r: r["gap_mm"])
    return {
        "contacts": rows,
        "noise_floor_mm": noise_floor_mm,
        "contact_mm": contact_mm,
        "thresholds_basis": validation.HEURISTIC,
        "method": "vertex-to-vertex minimum via cKDTree, workers=-1",
        "limitation": "A vertex-to-vertex minimum OVERESTIMATES the true gap: on a "
                      "triangulated surface the closest points generally lie inside "
                      "faces, not at vertices. This measures a gap in a mesh and is "
                      "NOT a prescription for enamel reduction.",
    }


def analyse(teeth: list[dict], arch: str, occlusal_axis, arch_centre,
            tolerance_mm: float = DEFAULT_WIDTH_TOLERANCE_MM,
            noise_floor_mm: float = NOISE_FLOOR_MM,
            contact_mm: float = CONTACT_MM) -> dict:
    """Both halves, plus a summary a status bar can render in one line."""
    widths = crown_widths(teeth, arch, occlusal_axis, arch_centre, tolerance_mm)
    contacts = interproximal_contacts(teeth, noise_floor_mm, contact_mm)

    needs_review = [r for r in widths["teeth"] if r.get("status") == "REVIEW"]
    touching = [c for c in contacts["contacts"] if c["state"] == "CONTACT"]
    if needs_review:
        summary = (f"{len(needs_review)} crown(s) outside Wheeler "
                   f"+/-{tolerance_mm}mm — check the segmentation.")
    elif touching:
        summary = f"{len(touching)} interproximal contact(s) at or under {contact_mm}mm."
    elif widths["teeth"]:
        summary = "All measured crowns are within anatomical tolerance and separated."
    else:
        summary = "No extracted crowns to measure yet."

    return {"widths": widths, "interproximal": contacts, "summary": summary,
            "crowns_measured": len(widths["teeth"])}
