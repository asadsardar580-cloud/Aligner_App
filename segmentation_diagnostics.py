"""Is a segmentation SPATIALLY COHERENT, and is it indexed to the right mesh?

WHY THIS MODULE EXISTS. The real scan's stored labels were reported as a model
failure: "all twelve labelled teeth flood a quarter of the arch, best IoU
0.01-0.05, FDI 31 spans 33.8 x 34.4 x 15.3mm". Every one of those numbers was
real and the conclusion drawn from them was wrong. The labels are coherent;
they were being read against the wrong vertex array.

    per-tooth bounding-box diagonal, median over the 12 labelled teeth
      labels indexed to the app's array (np.unique lexicographic)   50.71 mm
      labels indexed to the array TGN actually loaded               13.97 mm

Both arrays hold exactly 94,848 vertices, so every count agrees, every length
check passes, and the correspondence is silently wrong. A tooth is 6-12mm
across; 50mm is most of a mandible. That is what this module measures, and it
is the check that tells a model failure apart from a mapping failure - the two
look identical in any metric computed on the labels alone, including IoU.

NOTHING HERE IS A CLINICAL GRADE. A plausible bounding box means the label is
localised, not that it is the right tooth. `segmentation_review.py` scores
whether a region deserves a clinician's eye; this answers the prior question
of whether the labels are attached to the geometry at all.
"""
from __future__ import annotations

import numpy as np

# A mandibular tooth's crown, corner to corner. Wheeler's mesiodistal widths
# run 5.0mm (lower central incisor) to 11.2mm (lower first molar), and a crown
# is taller than it is wide, so the box diagonal of ONE tooth sits comfortably
# under 20mm. ENGINEERING PLAUSIBILITY BOUND for detecting a mis-indexed label
# array, not a clinical tolerance and not a size check on the tooth: a label
# over this is not a big tooth, it is not one tooth.
MAX_PLAUSIBLE_TOOTH_DIAGONAL_MM = 20.0

# Below this fraction of the label in one connected piece, the label is
# scattered. `segmentation_fallback` already re-grows such a tooth with the
# geodesic flood; this only reports it.
MIN_LARGEST_COMPONENT_FRACTION = 0.60

THRESHOLD_NOTE = ("engineering plausibility bounds for detecting a "
                  "mis-indexed or scattered label array, not clinical "
                  "tolerances")


def _label_components(faces_of_label, n_verts):
    """How many disconnected pieces does this label's surface fall into?

    Over FACE adjacency through shared vertices, which is what makes a piece
    one surface. Counting vertex adjacency instead would join two teeth that
    merely touch at a contact point.
    """
    if len(faces_of_label) == 0:
        return 0, 0.0, []
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception:
        return -1, float("nan"), []

    f = np.asarray(faces_of_label, np.int64)
    rows = np.repeat(np.arange(len(f)), 3)
    cols = f.reshape(-1)
    inc = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                     shape=(len(f), n_verts)).tocsr()
    adj = inc @ inc.T                      # faces sharing >= 1 vertex
    n, lab = connected_components(adj, directed=False)
    sizes = np.bincount(lab, minlength=n)
    order = np.argsort(sizes)[::-1]
    return int(n), float(sizes.max() / sizes.sum()), sizes[order].tolist()


def label_report(verts, faces, labels, background=0):
    """Per-label geometry. The diagnostic the brief asks to render.

    Returns one row per label with vertex count, face count, bounding box,
    box diagonal, centroid and disconnected-component count, plus an overall
    verdict on whether the label array is indexed to this mesh at all.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    lab = np.asarray(labels).astype(np.int64).reshape(-1)

    if len(lab) != len(v):
        return {
            "indexed_to_this_mesh": False,
            "reason": (f"label array is {len(lab)} long and the mesh has "
                       f"{len(v)} vertices - they cannot correspond"),
            "n_vertices": int(len(v)), "n_labels": int(len(lab)),
            "teeth": [], "threshold_note": THRESHOLD_NOTE,
        }

    rows = []
    for value in sorted({int(x) for x in np.unique(lab)}):
        if value == background:
            continue
        m = lab == value
        pts = v[m]
        # STRICT face membership: all three corners carry the label. A
        # majority rule would drag the cervical ring of gingiva in with every
        # tooth and inflate every box.
        fm = m[f].all(axis=1)
        sub = f[fm]
        ext = (pts.max(axis=0) - pts.min(axis=0)) if len(pts) else np.zeros(3)
        diag = float(np.linalg.norm(ext))
        ncomp, largest, sizes = _label_components(sub, len(v))
        rows.append({
            "label": value,
            "vertices": int(m.sum()),
            "faces": int(fm.sum()),
            "bbox_mm": [round(float(x), 3) for x in ext],
            "bbox_diagonal_mm": round(diag, 3),
            "centroid": [round(float(x), 3) for x in pts.mean(axis=0)]
            if len(pts) else None,
            "components": ncomp,
            "largest_component_fraction": (None if largest != largest
                                           else round(largest, 4)),
            "component_face_counts": sizes[:6],
            "plausible_size": bool(diag <= MAX_PLAUSIBLE_TOOTH_DIAGONAL_MM),
            "connected": bool(ncomp == 1),
        })

    diags = [r["bbox_diagonal_mm"] for r in rows]
    plausible = sum(r["plausible_size"] for r in rows)
    median = float(np.median(diags)) if diags else float("nan")
    # THE INDEXING VERDICT. A label array read against the wrong vertex
    # ordering scatters EVERY tooth across the whole arch, so the median is
    # what separates the two failures: one bad tooth is a model problem, all
    # of them is a mapping problem.
    indexed = bool(rows) and median <= MAX_PLAUSIBLE_TOOTH_DIAGONAL_MM
    return {
        "indexed_to_this_mesh": indexed,
        "reason": (None if indexed else
                   "every labelled tooth spans far more than a tooth - the "
                   "label array is almost certainly indexed to a different "
                   "vertex ordering of the same mesh, not to this one"),
        "n_vertices": int(len(v)),
        "n_labels": int(len(lab)),
        "n_teeth": len(rows),
        "median_bbox_diagonal_mm": round(median, 3) if diags else None,
        "teeth_with_plausible_size": plausible,
        "teeth_connected": sum(r["connected"] for r in rows),
        "teeth_scattered": [r["label"] for r in rows
                            if r["largest_component_fraction"] is not None
                            and r["largest_component_fraction"]
                            < MIN_LARGEST_COMPONENT_FRACTION],
        "teeth": rows,
        "max_plausible_tooth_diagonal_mm": MAX_PLAUSIBLE_TOOTH_DIAGONAL_MM,
        "threshold_note": THRESHOLD_NOTE,
    }


def format_report(rep):
    """One line per tooth, for a terminal. The brief asks for this rendered."""
    out = []
    if not rep.get("indexed_to_this_mesh"):
        out.append(f"LABELS ARE NOT INDEXED TO THIS MESH: {rep.get('reason')}")
    out.append(f"{rep.get('n_teeth', 0)} labelled teeth over "
               f"{rep.get('n_vertices')} vertices; median tooth box diagonal "
               f"{rep.get('median_bbox_diagonal_mm')} mm")
    out.append(f"{'label':>6} {'verts':>7} {'faces':>7}  "
               f"{'bbox (mm)':>22} {'diag':>7} {'parts':>6} {'largest':>8} "
               f"{'size':>6} {'review':>7}")
    for r in rep.get("teeth", []):
        bb = r["bbox_mm"]
        # SAY WHICH TEST FAILED, not just THAT one did. The old column was a
        # bare "<<" for either condition, so a tooth of implausible SIZE - two
        # teeth merged into one label - read identically to one that is merely
        # in two PIECES, which is the far more common and far less serious
        # case. They lead to different actions.
        size = "ok" if r["plausible_size"] else "BIG"
        review = "ok" if (r["plausible_size"] and r["connected"]) else "REVIEW"
        out.append(f"{r['label']:>6} {r['vertices']:>7} {r['faces']:>7}  "
                   f"{bb[0]:>6.2f} x{bb[1]:>6.2f} x{bb[2]:>6.2f} "
                   f"{r['bbox_diagonal_mm']:>7.2f} {r['components']:>6} "
                   f"{str(r['largest_component_fraction']):>8} "
                   f"{size:>6} {review:>7}")
    return "\n".join(out)
