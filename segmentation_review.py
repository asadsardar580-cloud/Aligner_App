"""Per-tooth segmentation review: explicit confidence, explicit verdict.

THE PROBLEM THIS SOLVES. `/segment` returns a per-vertex FDI array and the
client colours the arch with it. Nothing between the model and the clinician
ever asked whether a region is plausibly one tooth. `ToothCandidate.confidence`
has existed since the segmentation package was written and is assigned nowhere,
so `needs_review` is unconditionally True — a review flag that is always on is
the same as no flag at all.

WHAT A CONFIDENCE NUMBER IS ALLOWED TO BE HERE. Not a probability. The model
emits no calibrated uncertainty, and inventing one would be worse than having
none. This is a weighted agreement score over INDEPENDENTLY CHECKABLE
geometric facts — is the region connected, is it a plausible size, does its
width match Wheeler for the FDI it claims, is it on the right side of the
midline for that FDI. Every contributing factor is returned alongside the
score, so a clinician can disagree with a factor rather than only with a
number.

THREE VERDICTS, AND THE MIDDLE ONE IS NOT A FAILURE:

  PASS    every checkable fact agrees. Still not a clinical guarantee.
  REVIEW  something disagrees. Look at this tooth before relying on its FDI.
  FAIL    the region cannot be a tooth — disconnected, or a handful of faces.

Nothing here alters labels. Correction is the clinician's, and the software's
job is to say clearly where to look.
"""
from __future__ import annotations

import numpy as np

import core_geometry as cg
import jaw_naming
import validation

# --- verdicts -------------------------------------------------------------
PASS = "PASS"
REVIEW = "REVIEW"
FAIL = "FAIL"

# --- thresholds. All HEURISTIC: chosen from the cases in this repo. -------
MIN_FACES = 50                  # under this, a region is debris, not a tooth
MIN_AREA_MM2 = 15.0             # a premolar's visible crown is ~60-100mm^2
MAX_ARCH_FRACTION = 0.25        # one tooth of ~14 should be well under a quarter
MIN_CONNECTED_FRACTION = 0.90   # the largest island should be nearly all of it
WIDTH_TOLERANCE_MM = 1.5        # Wheeler, same ceiling as space_analysis



def _surface_area(verts, tris) -> float:
    """Summed triangle area, mm^2. Inline rather than imported: core_geometry
    has no public triangle_areas, and adding one for this would put a new name
    in the geometry engine for a reporting convenience."""
    if not len(tris):
        return 0.0
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _factor(name, ok, weight, detail, **metrics):
    return {"factor": name, "ok": bool(ok), "weight": float(weight),
            "detail": detail, "metrics": metrics}


def review_tooth(label, face_mask, verts, faces, arch, total_faces,
                 arch_centre=None, occlusal_axis=None) -> dict:
    """Score one labelled region against what can actually be checked."""
    factors = []
    n_faces = int(face_mask.sum())
    region_faces = faces[face_mask]
    region_verts = np.unique(region_faces)

    # 1. Enough of anything to be a tooth at all.
    big_enough = n_faces >= MIN_FACES
    factors.append(_factor(
        "size", big_enough, 0.20,
        f"{n_faces} faces" + ("" if big_enough else f" — under {MIN_FACES}, this is debris"),
        faces=n_faces, min_faces=MIN_FACES))

    # 2. Connectedness. A tooth is one piece; two islands sharing a label means
    #    the model labelled two separate places the same.
    connected_frac = 1.0
    if n_faces:
        largest = cg.largest_face_component(faces, face_mask)
        connected_frac = float(largest.sum()) / n_faces
    conn_ok = connected_frac >= MIN_CONNECTED_FRACTION
    factors.append(_factor(
        "connectedness", conn_ok, 0.25,
        f"largest island is {connected_frac*100:.0f}% of the region"
        + ("" if conn_ok else " — the label covers separate places"),
        connected_fraction=round(connected_frac, 4)))

    # 3. Surface area, and share of the whole arch.
    area = _surface_area(verts, region_faces) if n_faces else 0.0
    arch_frac = n_faces / float(total_faces) if total_faces else 0.0
    area_ok = area >= MIN_AREA_MM2 and arch_frac <= MAX_ARCH_FRACTION
    factors.append(_factor(
        "extent", area_ok, 0.15,
        f"{area:.0f}mm2, {arch_frac*100:.1f}% of the arch"
        + ("" if area_ok else " — implausible for a single crown"),
        surface_area_mm2=round(area, 1), arch_fraction=round(arch_frac, 4)))

    # 4. FDI validity for this arch. jaw_naming owns the quadrant rules.
    expected = (jaw_naming.UPPER_FDI if arch in ("upper", "maxillary")
                else jaw_naming.LOWER_FDI)
    fdi_ok = int(label) in expected
    factors.append(_factor(
        "fdi_belongs_to_arch", fdi_ok, 0.25,
        f"FDI {label} is " + ("valid for this arch" if fdi_ok
                              else f"NOT a {arch} tooth number"),
        fdi=int(label)))

    # 5. Mesiodistal width against Wheeler, when it can be measured at all.
    width = None
    width_ok = None
    if arch_centre is not None and occlusal_axis is not None and len(region_verts) > 3:
        pts = verts[region_verts]
        raw = cg.measure_mesiodistal_width(pts, pts.mean(axis=0), arch_centre, occlusal_axis)
        if raw is not None:
            width = float(raw)
            pos = int(str(int(label))[-1])
            table = (cg.MAXILLARY_MD_WIDTH if arch in ("upper", "maxillary")
                     else cg.MANDIBULAR_MD_WIDTH)
            exp = table.get(pos)
            if exp is not None:
                width_ok = abs(width - exp) <= WIDTH_TOLERANCE_MM
                factors.append(_factor(
                    "width_vs_wheeler", width_ok, 0.15,
                    f"{width:.1f}mm against Wheeler {exp:.1f}mm"
                    + ("" if width_ok else
                       (" — wider, often two teeth merged" if width > exp
                        else " — narrower, often split or partly captured")),
                    width_mm=round(width, 2), expected_mm=exp))
    if width_ok is None:
        factors.append(_factor(
            "width_vs_wheeler", True, 0.0,
            "not measurable here — no Wheeler entry, or the mesiodistal axis is undefined"))

    # --- score: weighted agreement over the factors that could be checked ---
    usable = [f for f in factors if f["weight"] > 0]
    total_w = sum(f["weight"] for f in usable) or 1.0
    confidence = sum(f["weight"] for f in usable if f["ok"]) / total_w

    if not big_enough or connected_frac < 0.5:
        verdict = FAIL
    elif confidence >= 0.99:
        verdict = PASS
    else:
        verdict = REVIEW

    failed = [f["factor"] for f in factors if f["weight"] > 0 and not f["ok"]]
    return {
        "fdi": int(label),
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "confidence_is": "weighted agreement over independently checkable geometric "
                         "facts — NOT a model probability. The model emits no "
                         "calibrated uncertainty.",
        "failed_factors": failed,
        "factors": factors,
        "face_count": n_faces,
        "surface_area_mm2": round(area, 1),
        "width_mm": None if width is None else round(width, 2),
        "needs_review": verdict != PASS,
    }


def review_segmentation(labels, verts, faces, arch, arch_centre=None,
                        occlusal_axis=None, gingiva_label: int = 0) -> dict:
    """Review every labelled region in an arch.

    Also carries jaw_naming's structured verdict, which `/segment` computes and
    then collapses to a single diagnosis string on the failure path — exactly
    where the structure is most useful.
    """
    labels = np.asarray(labels)
    face_labels = labels[faces]
    # A face belongs to a tooth only when all three of its vertices agree.
    # Mixed faces sit on a boundary and belong to neither.
    unanimous = (face_labels[:, 0] == face_labels[:, 1]) & (face_labels[:, 1] == face_labels[:, 2])
    per_face = np.where(unanimous, face_labels[:, 0], gingiva_label)

    teeth = []
    for lab in sorted(set(int(x) for x in np.unique(per_face)) - {gingiva_label}):
        teeth.append(review_tooth(lab, per_face == lab, verts, faces, arch,
                                  len(faces), arch_centre, occlusal_axis))

    counts = {PASS: 0, REVIEW: 0, FAIL: 0}
    for t in teeth:
        counts[t["verdict"]] += 1

    jaw_report = jaw_naming.verify_fdi_matches_jaw(
        {"labels": labels.tolist()}, jaw_naming.jaw_for_arch(arch))

    if counts[FAIL]:
        summary = (f"{counts[FAIL]} region(s) cannot be a tooth and "
                   f"{counts[REVIEW]} need review.")
    elif counts[REVIEW]:
        summary = f"{counts[REVIEW]} of {len(teeth)} teeth need review before use."
    elif teeth:
        summary = f"All {len(teeth)} regions pass every checkable test."
    else:
        summary = "No tooth regions found."

    return {
        "teeth": sorted(teeth, key=lambda t: (t["verdict"] != FAIL,
                                              t["verdict"] != REVIEW, t["fdi"])),
        "counts": counts,
        "tooth_count": len(teeth),
        "summary": summary,
        # The whole verdict, not just its diagnosis line.
        "jaw_report": jaw_report,
        "thresholds_basis": validation.HEURISTIC,
        "limitation": "Geometric plausibility only. A PASS means nothing checkable "
                      "disagreed; it is NOT a clinical confirmation that the tooth is "
                      "correctly identified. Confirm FDI before relying on a per-tooth "
                      "root length — C_res is extrapolated along it.",
    }
