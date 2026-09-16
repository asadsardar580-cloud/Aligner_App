"""Hybrid segmentation: TGN -> classical geodesic flood -> manual landmarks.

WHY A FALLBACK CHAIN AT ALL. ToothGroupNetwork is one model trained on one
distribution. On a crowded mandible it routinely returns a label split across
two places, or a region twice the width of any real tooth. Those are not
borderline calls — they are geometrically impossible, detectable without any
ground truth, and recoverable by a method that does not use the model at all.

THE CHAIN, AND WHAT EACH TIER ACTUALLY KNOWS:

  1 TGN            Knows what a tooth looks like. Knows nothing about whether
                   its own output is connected or plausibly sized.
  2 Geodesic flood Knows nothing about teeth. Follows the curvature barrier at
                   the cervical margin, so whatever it returns is ONE connected
                   region bounded by real concavity. Cannot name a tooth.
  3 Manual         The clinician. Always available, always correct by
                   definition, and the only tier that can resolve identity when
                   the first two disagree.

Tier 2 recovers the SHAPE and keeps tier 1's NAME, which is the right division:
the flood cannot invent an FDI number and the model's numbering is usually right
even when its boundary is not. Any tooth that falls through to tier 2 is marked
REVIEW_REQUIRED, because a recovered region is a repair, not a confirmation.

THE TRIGGER IS NOT AN mIoU. The spec asks for "mIoU < 0.75", and no IoU can be
computed at runtime — that needs ground truth the clinician does not have while
planning. What CAN be computed is the geometric plausibility score that
segmentation_review already produces, plus connectedness, which is the specific
failure the fallback repairs. Using a threshold named mIoU for a number that is
not an mIoU is how a plausibility proxy becomes a quoted accuracy figure.
"""
from __future__ import annotations

import numpy as np

import core_geometry as cg
import segmentation_review

# Below this plausibility score, or with a region this fragmented, tier 1's
# result for that tooth is not used as-is. HEURISTIC.
MIN_PLAUSIBILITY = 0.75          # same number the spec names, honestly labelled
MIN_CONNECTED_FRACTION = 0.90

STATUS_OK = "OK"
STATUS_REVIEW = "REVIEW_REQUIRED"
STATUS_FAILED = "FAILED"

TIER_MODEL = "tgn"
TIER_GEODESIC = "classical_geodesic_flood"
TIER_MANUAL = "manual_landmarks"


def _region_face_mask(labels, faces, label, gingiva=0):
    fl = np.asarray(labels)[faces]
    unanimous = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
    return unanimous & (fl[:, 0] == label)


def geodesic_recover(verts, faces, graph, concavity, seed_vertex,
                     tolerance=None) -> np.ndarray:
    """Tier 2: re-grow one crown from a seed, following the curvature barrier.

    This is the SAME machinery the magic wand uses interactively — a geodesic
    walk over a graph whose edge weights rise at concavity, so the front slows
    and stops in the sulcus. Whatever comes back is one connected region
    bounded by real anatomy, which is precisely the property tier 1 lost.
    """
    import cut_guard
    dist = cg.geodesic_from_seed(graph, verts, seed_vertex)
    tol = tolerance if tolerance else cut_guard.auto_tolerance(dist)["tolerance"]
    return np.nonzero(dist <= tol)[0]


def run(labels, verts, faces, arch, graph=None, concavity=None,
        arch_centre=None, occlusal_axis=None, gingiva=0) -> dict:
    """Review tier 1's output and repair what fails, tooth by tooth.

    Returns the possibly-repaired labels plus a per-tooth account of which tier
    produced each region and why.
    """
    labels = np.asarray(labels).copy()
    review = segmentation_review.review_segmentation(
        labels, verts, faces, arch, arch_centre, occlusal_axis, gingiva)

    rows, repaired = [], 0
    for t in review["teeth"]:
        fdi = t["fdi"]
        conn = next((f["metrics"].get("connected_fraction", 1.0)
                     for f in t["factors"] if f["factor"] == "connectedness"), 1.0)
        needs = (t["confidence"] < MIN_PLAUSIBILITY
                 or conn < MIN_CONNECTED_FRACTION
                 or t["verdict"] == segmentation_review.FAIL)

        row = {"fdi": fdi, "tier": TIER_MODEL, "status": STATUS_OK,
               "plausibility": t["confidence"], "connected_fraction": round(conn, 4),
               "failed_factors": t["failed_factors"], "reason": ""}

        if not needs:
            rows.append(row)
            continue

        row["reason"] = (f"plausibility {t['confidence']:.2f} < {MIN_PLAUSIBILITY} "
                         if t["confidence"] < MIN_PLAUSIBILITY else "")
        if conn < MIN_CONNECTED_FRACTION:
            row["reason"] += (f"largest island {conn*100:.0f}% of the region")

        if graph is None or concavity is None:
            # Tier 2 needs the precomputed barrier graph. Without it the honest
            # move is to hand the tooth to tier 3, not to guess.
            row.update(tier=TIER_MANUAL, status=STATUS_REVIEW,
                       reason=row["reason"] + " — no barrier graph available for "
                                              "geodesic recovery; needs manual landmarks")
            rows.append(row)
            continue

        mask = _region_face_mask(labels, faces, fdi, gingiva)
        if not mask.any():
            row.update(tier=TIER_MANUAL, status=STATUS_FAILED,
                       reason=row["reason"] + " — no unanimous faces to seed from")
            rows.append(row)
            continue

        # Seed from the LARGEST island's centroid, not the whole region's: a
        # split label's overall centroid can sit in the gap between its two
        # pieces, which is gingiva, and the flood would grow from the wrong place.
        largest = cg.largest_face_component(faces, mask)
        ids = np.unique(faces[largest])
        centroid = verts[ids].mean(axis=0)
        seed = int(ids[np.argmin(np.linalg.norm(verts[ids] - centroid, axis=1))])

        try:
            grown = geodesic_recover(verts, faces, graph, concavity, seed)
        except Exception as e:                       # noqa: BLE001 - reported, not swallowed
            row.update(tier=TIER_MANUAL, status=STATUS_FAILED,
                       reason=row["reason"] + f" — geodesic recovery failed: {e}")
            rows.append(row)
            continue

        if len(grown) < 10:
            row.update(tier=TIER_MANUAL, status=STATUS_REVIEW,
                       reason=row["reason"] + " — geodesic flood returned almost nothing")
            rows.append(row)
            continue

        # Repaint: clear the old region, apply the recovered one. The FDI number
        # is kept from tier 1 — the flood cannot name a tooth.
        labels[labels == fdi] = gingiva
        labels[grown] = fdi
        repaired += 1
        row.update(tier=TIER_GEODESIC, status=STATUS_REVIEW,
                   recovered_vertices=int(len(grown)),
                   reason=row["reason"] + " — shape recovered by geodesic flood, "
                                          "FDI kept from the model")
        rows.append(row)

    return {
        "labels": labels,
        "teeth": rows,
        "repaired_count": repaired,
        "needs_review": [r["fdi"] for r in rows if r["status"] != STATUS_OK],
        "tiers": {TIER_MODEL: sum(1 for r in rows if r["tier"] == TIER_MODEL),
                  TIER_GEODESIC: sum(1 for r in rows if r["tier"] == TIER_GEODESIC),
                  TIER_MANUAL: sum(1 for r in rows if r["tier"] == TIER_MANUAL)},
        "threshold": {"min_plausibility": MIN_PLAUSIBILITY,
                      "min_connected_fraction": MIN_CONNECTED_FRACTION,
                      "basis": "software heuristic",
                      "note": "This is NOT an mIoU threshold. No IoU is computable at "
                              "runtime — that needs annotated ground truth. It is the "
                              "geometric plausibility score, and connectedness, which "
                              "is the specific failure the fallback repairs."},
        "limitation": "A repaired region is a REPAIR, not a confirmation. Every tooth "
                      "that left tier 1 is marked REVIEW_REQUIRED and should be "
                      "confirmed before its FDI is relied on — C_res is extrapolated "
                      "along a root length chosen from that number.",
    }
