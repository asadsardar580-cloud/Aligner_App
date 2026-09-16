"""Segmentation benchmarking against ground truth — and a refusal when there is none.

READ THIS FIRST. This module computes real metrics: per-class IoU, mean IoU, FDI
classification accuracy, Chamfer distance and Hausdorff distance on crown
boundaries. The arithmetic is complete and correct.

WHAT IT WILL NOT DO IS INVENT A GROUND TRUTH. This repository contains exactly
one real scan and one label file, and that label file is ToothGroupNetwork's own
PREDICTION. Scoring a model against its own output returns mIoU = 1.0, FDI
accuracy = 100%, and Chamfer = 0.0 — numbers that look like validation and mean
nothing whatsoever. `benchmark()` therefore refuses to run without an
independently annotated file, and says what is missing.

That refusal is the feature. The project already carries four mutually
inconsistent accuracy claims inherited from prose (11/14 and 5/14 in an old
handover; 8/14 and 9/14 in one test's hard-coded print; 10/14 and 10/14 in two
more places), none of which any code reproduces. Adding a fifth from a
self-comparison would be worse than having none.

TO ACTUALLY PRODUCE NUMBERS you need a scan whose per-vertex FDI labels were
assigned by a clinician or an independent annotation pipeline, in the format
described by `GROUND_TRUTH_FORMAT` below. Then:

    python benchmark_segmentation.py case_lower.stl ground_truth.json

WHAT CAN BE MEASURED WITHOUT GROUND TRUTH is intrinsic quality — whether a
region is connected, plausibly sized, correctly quadranted. That is
`segmentation_review.py`, and `confidence_breakdown()` here exposes it in the
shape `/segment` reports. It is a geometric plausibility proxy and is labelled
as one; it is NOT an IoU estimate and must never be read as one.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import core_geometry as cg
import segmentation_review
import stl_io

GROUND_TRUTH_FORMAT = {
    "description": "Per-vertex FDI labels assigned independently of the model under test.",
    "schema": {
        "jaw": "'upper' | 'lower'",
        "labels": "list[int], one per mesh vertex, 0 = gingiva, else FDI 11-48",
        "annotator": "who or what produced these labels — REQUIRED, and it must "
                     "not be the model being scored",
        "mesh_sha256": "optional but strongly advised: the scan these labels "
                       "belong to, so a mismatch is caught rather than measured",
    },
}


class NoGroundTruth(RuntimeError):
    """Raised rather than returning a number nobody can trust."""


# =========================================================================
# Metrics
# =========================================================================

def per_class_iou(pred, truth, ignore=(0,)) -> dict:
    """Intersection over union per FDI class.

    A class present in neither prediction nor truth is absent from the result
    rather than scored 1.0 — counting agreement about a tooth that does not
    exist would inflate mIoU by the number of teeth the patient is missing.
    """
    pred = np.asarray(pred).ravel()
    truth = np.asarray(truth).ravel()
    if pred.shape != truth.shape:
        raise ValueError(
            f"Prediction has {pred.size} labels and ground truth has {truth.size}. "
            f"These are not the same mesh.")
    out = {}
    classes = (set(int(c) for c in np.unique(pred)) |
               set(int(c) for c in np.unique(truth))) - set(ignore)
    for c in sorted(classes):
        p, t = pred == c, truth == c
        union = int((p | t).sum())
        if union == 0:
            continue
        out[c] = float((p & t).sum()) / union
    return out


def mean_iou(pred, truth, ignore=(0,)) -> float:
    ious = per_class_iou(pred, truth, ignore)
    return float(np.mean(list(ious.values()))) if ious else 0.0


def fdi_accuracy(pred, truth, ignore=(0,)) -> dict:
    """Vertex-level label accuracy, plus the confusion that matters clinically.

    A tooth labelled as its NEIGHBOUR is a different kind of error from one
    labelled as gingiva: the first silently gives the wrong Wheeler root length
    and therefore the wrong C_res, the second just loses the tooth.
    """
    pred = np.asarray(pred).ravel()
    truth = np.asarray(truth).ravel()
    tooth = ~np.isin(truth, list(ignore))
    if not tooth.any():
        return {"accuracy": 0.0, "n_tooth_vertices": 0,
                "mislabelled_as_gingiva": 0, "mislabelled_as_other_tooth": 0}
    correct = int((pred[tooth] == truth[tooth]).sum())
    as_ging = int(np.isin(pred[tooth], list(ignore)).sum())
    return {
        "accuracy": correct / float(tooth.sum()),
        "n_tooth_vertices": int(tooth.sum()),
        "mislabelled_as_gingiva": as_ging,
        "mislabelled_as_other_tooth": int(tooth.sum()) - correct - as_ging,
    }


def _boundary_vertices(verts, faces, labels, label) -> np.ndarray:
    """Vertices on the cervical margin of one labelled region.

    The margin is where a face has vertices of more than one label — i.e. the
    boundary of the region as the segmentation drew it.
    """
    lab = np.asarray(labels)
    fl = lab[faces]
    mixed = ~((fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2]))
    touching = mixed & (fl == label).any(axis=1)
    if not touching.any():
        return np.empty((0, 3), float)
    ids = np.unique(faces[touching])
    return verts[ids[lab[ids] == label]]


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance, mm.

    Chamfer is a MEAN and is therefore forgiving of a few bad points — which is
    why Hausdorff is reported alongside it. A boundary that is right everywhere
    except one 4mm excursion scores well here and badly there, and the pair of
    numbers together says which situation you are in.
    """
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    da, _ = cKDTree(b).query(a, workers=-1)
    db, _ = cKDTree(a).query(b, workers=-1)
    return float((da.mean() + db.mean()) / 2.0)


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric worst-case nearest-neighbour distance, mm."""
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    da, _ = cKDTree(b).query(a, workers=-1)
    db, _ = cKDTree(a).query(b, workers=-1)
    return float(max(da.max(), db.max()))


def boundary_metrics(verts, faces, pred, truth, ignore=(0,)) -> dict:
    """Chamfer and Hausdorff per tooth, between predicted and true margins."""
    rows = []
    classes = sorted((set(int(c) for c in np.unique(truth))) - set(ignore))
    for c in classes:
        pb = _boundary_vertices(verts, faces, pred, c)
        tb = _boundary_vertices(verts, faces, truth, c)
        rows.append({
            "fdi": c,
            "chamfer_mm": round(chamfer_distance(pb, tb), 4),
            "hausdorff_mm": round(hausdorff_distance(pb, tb), 4),
            "predicted_boundary_points": int(len(pb)),
            "true_boundary_points": int(len(tb)),
        })
    finite = [r["chamfer_mm"] for r in rows if np.isfinite(r["chamfer_mm"])]
    worst = [r["hausdorff_mm"] for r in rows if np.isfinite(r["hausdorff_mm"])]
    return {
        "per_tooth": rows,
        "mean_chamfer_mm": round(float(np.mean(finite)), 4) if finite else None,
        "max_hausdorff_mm": round(float(np.max(worst)), 4) if worst else None,
        "note": "Chamfer is a MEAN and forgives isolated excursions; Hausdorff is the "
                "worst case. Read them together — a good Chamfer with a bad Hausdorff "
                "is a boundary that is right except in one place.",
    }


# =========================================================================
# Confidence breakdown — intrinsic, no ground truth required
# =========================================================================

def confidence_breakdown(labels, verts, faces, arch, arch_centre=None,
                         occlusal_axis=None) -> dict:
    """The shape `/segment` reports. Intrinsic geometry only.

    `miou_estimate` is present because the API contract asks for it, and it is
    NOT an IoU. No IoU can be computed without ground truth. It is a geometric
    plausibility proxy, and the payload says so in the key next to it — a number
    called miou that is not an mIoU is exactly the kind of thing that becomes a
    quoted accuracy figure three documents later.
    """
    review = segmentation_review.review_segmentation(
        labels, verts, faces, arch, arch_centre, occlusal_axis)
    teeth = review["teeth"]
    if not teeth:
        return {"miou_estimate": 0.0, "boundary_clarity": 0.0,
                "component_isolation": 0.0, "overall_score": 0.0,
                "is_measured_accuracy": False,
                "meaning": "No tooth regions found.", "review": review}

    def _mean(fn):
        vals = [fn(t) for t in teeth]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _factor(t, name):
        for f in t["factors"]:
            if f["factor"] == name and f["weight"] > 0:
                return 1.0 if f["ok"] else 0.0
        return None

    component_isolation = _mean(lambda t: _factor(t, "connectedness"))
    # Boundary clarity: how cleanly regions separate. A region whose margin is a
    # large fraction of its own area has a ragged, interdigitated boundary.
    clarity = []
    lab = np.asarray(labels)
    fl = lab[faces]
    mixed = ~((fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2]))
    for t in teeth:
        c = t["fdi"]
        own = ((fl == c).any(axis=1)).sum()
        edge = (mixed & (fl == c).any(axis=1)).sum()
        clarity.append(1.0 - min(1.0, edge / float(own)) if own else 0.0)
    boundary_clarity = float(np.mean(clarity)) if clarity else 0.0

    proxy = _mean(lambda t: t["confidence"])
    overall = float(np.mean([proxy, boundary_clarity, component_isolation]))

    return {
        "miou_estimate": round(proxy, 4),
        "boundary_clarity": round(boundary_clarity, 4),
        "component_isolation": round(component_isolation, 4),
        "overall_score": round(overall, 4),
        # The two keys that stop this becoming a quoted accuracy figure.
        "is_measured_accuracy": False,
        "meaning": "GEOMETRIC PLAUSIBILITY, not segmentation accuracy. No IoU can be "
                   "computed without independently annotated ground truth, which this "
                   "installation does not have. 'miou_estimate' keeps the API's key "
                   "name; it is a plausibility proxy and must not be quoted as mIoU. "
                   "Run benchmark_segmentation.py with annotated data for real metrics.",
        "review": review,
    }


# =========================================================================
# The benchmark itself
# =========================================================================

def load_ground_truth(path: str, expect_vertices: int | None = None) -> dict:
    if not os.path.exists(path):
        raise NoGroundTruth(
            f"No ground-truth file at {path!r}. Required format:\n"
            + json.dumps(GROUND_TRUTH_FORMAT, indent=2))
    with open(path, encoding="utf-8") as fh:
        gt = json.load(fh)
    if "labels" not in gt:
        raise NoGroundTruth(f"{path!r} has no 'labels' array.")
    annotator = (gt.get("annotator") or "").strip()
    if not annotator:
        raise NoGroundTruth(
            f"{path!r} does not say who annotated it. This field is required and it "
            f"must not be the model under test — scoring a model against its own "
            f"output returns mIoU 1.0 and means nothing.")
    if annotator.lower() in ("tgn", "toothgroupnetwork", "model", "prediction"):
        raise NoGroundTruth(
            f"{path!r} is annotated by {annotator!r}, which is the model being scored. "
            f"That is not ground truth.")
    if expect_vertices is not None and len(gt["labels"]) != expect_vertices:
        raise NoGroundTruth(
            f"{path!r} has {len(gt['labels'])} labels for a {expect_vertices}-vertex "
            f"mesh. These are not the same scan.")
    return gt


def benchmark(scan_path: str, ground_truth_path: str, predicted_labels=None,
              out_path: str = "segmentation_benchmark_report.json") -> dict:
    """Score a prediction against annotated ground truth. Writes a JSON report."""
    with open(scan_path, "rb") as fh:
        verts, faces = stl_io.parse_stl_bytes(fh.read())
    verts, faces, _ = cg.condition_mesh(verts, faces)

    gt = load_ground_truth(ground_truth_path, expect_vertices=len(verts))
    truth = np.asarray(gt["labels"], int)

    if predicted_labels is None:
        raise NoGroundTruth(
            "No prediction supplied. Run /segment (or tgn_bridge) first and pass its "
            "labels in — this function scores a prediction, it does not produce one.")
    pred = np.asarray(predicted_labels, int)

    t0 = time.time()
    ious = per_class_iou(pred, truth)
    report = {
        "generated": time.time(),
        "scan": os.path.basename(scan_path),
        "annotator": gt.get("annotator"),
        "jaw": gt.get("jaw"),
        "vertex_count": int(len(verts)),
        "mean_iou": round(mean_iou(pred, truth), 4),
        "per_class_iou": {str(k): round(v, 4) for k, v in ious.items()},
        "fdi_accuracy": fdi_accuracy(pred, truth),
        "boundary": boundary_metrics(verts, faces, pred, truth),
        "seconds": round(time.time() - t0, 2),
        # No patient identifier reaches this file. The scan is referenced by
        # basename only, and the labels are integers.
        "phi": "none — basenames and integers only",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nusage: python benchmark_segmentation.py <scan.stl> <ground_truth.json>")
        print("\nRequired ground-truth format:")
        print(json.dumps(GROUND_TRUTH_FORMAT, indent=2))
        sys.exit(2)
    try:
        r = benchmark(sys.argv[1], sys.argv[2])
    except NoGroundTruth as e:
        print(f"REFUSED: {e}")
        sys.exit(1)
    print(json.dumps({k: v for k, v in r.items() if k != "boundary"}, indent=2))
    print(f"mean chamfer {r['boundary']['mean_chamfer_mm']}mm, "
          f"max hausdorff {r['boundary']['max_hausdorff_mm']}mm")
