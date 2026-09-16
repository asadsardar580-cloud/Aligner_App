"""Biomechanical guardrails: per-stage rates and whole-case totals.

WHAT A THRESHOLD IN THIS FILE IS, AND IS NOT.

It is a number that says "this movement is outside the range clear aligners are
normally planned within, look again". It is NOT a statement that the tooth will
be safe below it or damaged above it. Root resorption risk is driven by force
magnitude, duration, root morphology, patient biology and appliance fit —
none of which this software can see. It sees millimetres and degrees.

So every threshold carries its provenance, and the report says in words that
GREEN means "within the planning envelope", never "biologically safe". A green
chip that a clinician reads as a safety guarantee is worse than no chip.

THE TWO KINDS OF LIMIT, AND WHY THEY ARE DIFFERENT.

  RATE   how far a tooth moves per tray. This is about whether the aligner can
         actually deliver the movement: exceed it and the tray simply does not
         track, so the plan is fiction regardless of biology. The staging engine
         already enforces these by deriving stage count FROM them.

  TOTAL  how far the tooth moves across the whole case. This is where the
         biological concern actually lives — sustained intrusion and extrusion
         are the movements associated in the literature with apical root
         resorption. Exceeding a total is not blocked; it is flagged loudly.

VELOCITY IS RATE PER TRAY, NOT PER DAY. The software does not know the wear
schedule. A 0.25mm/stage plan is a different velocity at 7-day changes than at
14-day, and only the clinician knows which.
"""
from __future__ import annotations

import validation

# --- status levels --------------------------------------------------------
GREEN = "GREEN"     # within the normal planning envelope
YELLOW = "YELLOW"   # achievable, but typically needs attachments or IPR
RED = "RED"         # outside the envelope: re-plan, or accept it knowingly

# --- per-stage rate limits ------------------------------------------------
# These are the ceilings the staging engine already divides by, carried here so
# the safety layer and the tray count cannot drift apart.
RATE_LIMITS = {
    "translation_mm": validation.threshold(
        0.25, "mm/stage", validation.HEURISTIC,
        "beyond this the tray does not track the tooth, so the plan is fiction "
        "regardless of biology"),
    "rotation_deg": validation.threshold(
        2.0, "deg/stage", validation.HEURISTIC,
        "applies to tip, torque and axial rotation alike"),
}

# --- whole-case totals ----------------------------------------------------
# Extrusion and intrusion are singled out because these are the movements most
# associated in the orthodontic literature with apical root resorption. The
# numbers are the directive's; they bound PLANNING, not biology.
TOTAL_LIMITS = {
    "extrusion_mm": validation.threshold(
        2.0, "mm total", validation.LITERATURE,
        "sustained extrusion beyond this is associated with increased apical root "
        "resorption risk; it is a flag for clinical judgement, not a computed verdict"),
    "intrusion_mm": validation.threshold(
        3.0, "mm total", validation.LITERATURE,
        "sustained intrusion beyond this is associated with increased apical root "
        "resorption risk; same caveat"),
    "translation_mm": validation.threshold(
        6.0, "mm total", validation.HEURISTIC,
        "beyond this, aligner-only bodily movement typically needs auxiliaries"),
    "rotation_deg": validation.threshold(
        45.0, "deg total", validation.HEURISTIC,
        "large axial rotation on a round-rooted tooth usually needs attachments"),
}

# Movement that is deliverable but generally needs help. Between GREEN and RED.
ATTACHMENT_ADVISED = {
    "rotation_deg": 15.0,
    "extrusion_mm": 0.5,      # aligners grip poorly in extrusion without one
    "translation_mm": 2.0,
}


def _worst(a: str, b: str) -> str:
    order = {GREEN: 0, YELLOW: 1, RED: 2}
    return a if order[a] >= order[b] else b


def assess_tooth(prescription: dict, stages: int, fdi=None, tooth_id=None) -> dict:
    """Rate and total assessment for one tooth's committed prescription."""
    p = {k: float(prescription.get(k, 0.0) or 0.0) for k in
         ("tip_deg", "torque_deg", "rotation_deg", "d_md", "d_bl", "d_oa")}
    n = max(int(stages), 1)

    findings, status = [], GREEN

    # --- rates ----------------------------------------------------------
    max_rot = max(abs(p["tip_deg"]), abs(p["torque_deg"]), abs(p["rotation_deg"])) / n
    max_tr = max(abs(p["d_md"]), abs(p["d_bl"]), abs(p["d_oa"])) / n

    if max_rot > RATE_LIMITS["rotation_deg"]["value"] + 1e-9:
        status = RED
        findings.append({
            "level": RED, "code": "rotation_rate",
            "message": f"{max_rot:.2f} deg per stage exceeds "
                       f"{RATE_LIMITS['rotation_deg']['value']} — the tray will not "
                       f"track this. Add stages.",
            "metrics": {"deg_per_stage": round(max_rot, 3)},
            "threshold": RATE_LIMITS["rotation_deg"]})
    if max_tr > RATE_LIMITS["translation_mm"]["value"] + 1e-9:
        status = RED
        findings.append({
            "level": RED, "code": "translation_rate",
            "message": f"{max_tr:.3f} mm per stage exceeds "
                       f"{RATE_LIMITS['translation_mm']['value']} — add stages.",
            "metrics": {"mm_per_stage": round(max_tr, 4)},
            "threshold": RATE_LIMITS["translation_mm"]})

    # --- totals ---------------------------------------------------------
    # d_oa is occlusoapical: positive is occlusal (extrusion), negative apical
    # (intrusion). Getting this sign backwards would swap two limits that differ
    # by a millimetre, so it is named explicitly rather than inferred.
    extrusion = max(0.0, p["d_oa"])
    intrusion = max(0.0, -p["d_oa"])
    total_tr = max(abs(p["d_md"]), abs(p["d_bl"]))
    total_rot = max(abs(p["tip_deg"]), abs(p["torque_deg"]), abs(p["rotation_deg"]))

    for name, value, key in (("extrusion", extrusion, "extrusion_mm"),
                             ("intrusion", intrusion, "intrusion_mm"),
                             ("translation", total_tr, "translation_mm"),
                             ("rotation", total_rot, "rotation_deg")):
        lim = TOTAL_LIMITS[key]
        if value > lim["value"] + 1e-9:
            status = _worst(status, RED if key in ("extrusion_mm", "intrusion_mm") else YELLOW)
            findings.append({
                "level": RED if key in ("extrusion_mm", "intrusion_mm") else YELLOW,
                "code": f"total_{name}",
                "message": f"total {name} {value:.2f}{lim['units'].split()[0]} exceeds "
                           f"{lim['value']}{lim['units'].split()[0]}. {lim['note']}",
                "metrics": {f"total_{name}": round(value, 3)},
                "threshold": lim})

    # --- auxiliaries advised --------------------------------------------
    if status == GREEN:
        for name, value, key in (("rotation", total_rot, "rotation_deg"),
                                 ("extrusion", extrusion, "extrusion_mm"),
                                 ("translation", total_tr, "translation_mm")):
            if value > ATTACHMENT_ADVISED[key]:
                status = YELLOW
                findings.append({
                    "level": YELLOW, "code": f"auxiliary_{name}",
                    "message": f"{value:.2f} of {name} typically needs an attachment "
                               f"or IPR to deliver reliably.",
                    "metrics": {name: round(value, 3)},
                    "threshold": validation.threshold(
                        ATTACHMENT_ADVISED[key], "mm or deg", validation.HEURISTIC,
                        "point at which auxiliaries are usually planned")})

    return {
        "tooth_id": tooth_id, "fdi": fdi, "status": status,
        "stages": n,
        "rates": {"max_rotation_deg_per_stage": round(max_rot, 3),
                  "max_translation_mm_per_stage": round(max_tr, 4)},
        "totals": {"extrusion_mm": round(extrusion, 3),
                   "intrusion_mm": round(intrusion, 3),
                   "translation_mm": round(total_tr, 3),
                   "rotation_deg": round(total_rot, 3)},
        "findings": findings,
        "prescription": p,
    }


def assess_case(teeth: list[dict], stages: int) -> dict:
    """Whole-case roll-up. `teeth` is [{tooth_id, fdi, prescription}, ...]."""
    rows = [assess_tooth(t.get("prescription", {}), stages,
                         fdi=t.get("fdi"), tooth_id=t.get("tooth_id")) for t in teeth]
    counts = {GREEN: 0, YELLOW: 0, RED: 0}
    for r in rows:
        counts[r["status"]] += 1
    overall = RED if counts[RED] else (YELLOW if counts[YELLOW] else GREEN)

    if counts[RED]:
        summary = (f"{counts[RED]} tooth/teeth outside the planning envelope — "
                   f"re-plan or accept knowingly.")
    elif counts[YELLOW]:
        summary = f"{counts[YELLOW]} tooth/teeth likely need attachments or IPR."
    elif rows:
        summary = f"All {len(rows)} teeth are within the normal planning envelope."
    else:
        summary = "No committed movements to assess."

    return {
        "overall": overall, "counts": counts, "teeth": rows,
        "stages": stages, "summary": summary,
        "meaning": {
            GREEN: "Within the range clear aligner cases are normally planned in. "
                   "This is NOT a statement that the movement is biologically safe.",
            YELLOW: "Deliverable, but typically requires attachments or IPR.",
            RED: "Outside the planning envelope: the tray will not track it, or the "
                 "total movement is in the range associated with elevated root "
                 "resorption risk.",
        },
        "limitation": "Millimetres and degrees only. Root resorption risk is driven by "
                      "force, duration, root morphology, patient biology and appliance "
                      "fit — none of which this software can see. Every threshold here "
                      "bounds PLANNING, not biology, and requires clinical judgement.",
    }
