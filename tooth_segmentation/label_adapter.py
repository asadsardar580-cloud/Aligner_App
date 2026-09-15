"""
Adapter: per-face labels from ANY segmentation source -> independent,
watertight tooth meshes.

Deliberately model-agnostic. MeshSegNet, ToothGroupNetwork,
DilatedToothSegNet and the manual magic-wand selection all ultimately
produce the same thing: an integer label per face. Everything downstream is
identical, and it is already tested. So the model is a swappable front end,
not an architecture decision.

    labels (from anywhere)
        -> propagate to full resolution   (models run on ~10k cells)
        -> clean islands
        -> split, cap, orient             (core_geometry, tested)
        -> independent watertight meshes
"""
import numpy as np
from scipy.spatial import cKDTree

import core_geometry as cg
from .models import ToothCandidate

GINGIVA_LABEL = 0


def face_centroids(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return verts[faces].mean(axis=1)


def propagate_labels(
    coarse_points: np.ndarray,
    coarse_labels: np.ndarray,
    fine_points: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Lift labels from a downsampled mesh back to full resolution.

    Necessary because MeshSegNet-class networks infer on meshes decimated to
    roughly 10,000 cells, while an intraoral scan is 200k-500k triangles.
    Without this step the labels simply do not apply to the mesh you intend
    to cut.

    Uses k-nearest-neighbour voting, with a caveat worth recording: I
    expected voting to beat nearest-single at interproximal seams, where one
    misplaced nearest cell can drag a stripe of wrong labels along a
    boundary. Measured on the synthetic arch, it did NOT -- k=1 scored
    99.25% and k=3 scored 99.23%, a wash. The default stays at k=3 because
    voting is more robust to the isolated mislabelled cells that real
    network output contains and the synthetic fixture does not, but on this
    evidence k=1 would serve equally well and is cheaper. Re-measure on real
    model output before treating either as settled.

    Ties are broken by the nearest neighbour, keeping results deterministic
    (spec section 10).
    """
    if len(coarse_points) == 0:
        raise ValueError("No coarse points to propagate from.")
    k = min(k, len(coarse_points))
    tree = cKDTree(coarse_points)
    _, idx = tree.query(fine_points, k=k)
    if k == 1:
        return coarse_labels[idx]

    neighbour_labels = coarse_labels[idx]                       # (n_fine, k)
    out = np.empty(len(fine_points), dtype=coarse_labels.dtype)
    for i, row in enumerate(neighbour_labels):
        vals, counts = np.unique(row, return_counts=True)
        best = counts.max()
        tied = vals[counts == best]
        # ties broken by the nearest neighbour, keeping the result
        # deterministic (spec section 10)
        out[i] = row[0] if row[0] in tied else tied[0]
    return out


def clean_labels(faces: np.ndarray, face_labels: np.ndarray,
                 min_faces: int = 50) -> tuple[np.ndarray, dict]:
    """
    Keep only the largest connected island per label; reassign the rest to
    gingiva.

    Learned models produce speckle -- isolated faces labelled as a tooth on
    the far side of the arch. Left alone, those wreck the bounding box,
    centroid and principal axes of the candidate, and they make capping fail
    because the "tooth" has multiple disconnected boundaries.
    """
    cleaned = face_labels.copy()
    stats = {}
    for lab in np.unique(face_labels):
        if lab == GINGIVA_LABEL:
            continue
        mask = face_labels == lab
        largest = cg.largest_face_component(faces, mask)
        removed = int(mask.sum() - largest.sum())
        if largest.sum() < min_faces:
            cleaned[mask] = GINGIVA_LABEL
            stats[int(lab)] = dict(kept=0, removed=int(mask.sum()), rejected_too_small=True)
            continue
        cleaned[mask & ~largest] = GINGIVA_LABEL
        stats[int(lab)] = dict(kept=int(largest.sum()), removed=removed,
                               rejected_too_small=False)
    return cleaned, stats


def labels_to_candidates(verts, faces, face_labels, config=None) -> list[ToothCandidate]:
    """Build a ToothCandidate per label, with the geometry stats spec 9 asks
    for. IDs stay candidate_NNN -- no FDI numbering."""
    from .config import SegmentationConfig
    cfg = config or SegmentationConfig()

    out = []
    for n, lab in enumerate(sorted(l for l in np.unique(face_labels) if l != GINGIVA_LABEL), 1):
        mask = face_labels == lab
        sub = faces[mask]
        pts = verts[np.unique(sub)]

        v0, v1, v2 = verts[sub[:, 0]], verts[sub[:, 1]], verts[sub[:, 2]]
        area = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

        centred = pts - pts.mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(np.cov(centred.T))
        order = np.argsort(eigvals)[::-1]
        extent = pts.max(axis=0) - pts.min(axis=0)

        frac = mask.sum() / max(len(faces), 1)
        warnings = []
        if frac > cfg.max_candidate_fraction:
            warnings.append(f"occupies {frac:.1%} of the arch — likely merged with neighbours")
        if mask.sum() < cfg.min_candidate_faces:
            warnings.append(f"only {int(mask.sum())} triangles — likely fragmentary")

        out.append(ToothCandidate(
            id=f"candidate_{n:03d}", face_mask=mask,
            vertex_count=len(pts), triangle_count=int(mask.sum()),
            surface_area=area, centroid=pts.mean(axis=0),
            bounding_box=(tuple(pts.min(axis=0)), tuple(pts.max(axis=0))),
            principal_axes=eigvecs[:, order],
            height=float(extent[2]), width=float(extent[0]), depth=float(extent[1]),
            warnings=warnings,
        ))
    return out


def extract_tooth_mesh(verts, faces, face_mask):
    """One independent, watertight tooth mesh.

    Independence is by construction: split_by_face_mask re-indexes into
    fresh arrays, so moving one tooth later cannot touch another or the
    gingiva (spec section 12).
    """
    (cv, cf), _ = cg.split_by_face_mask(verts, faces, face_mask)
    if len(cf) == 0:
        return None
    cv2, cf2 = cg.cap_and_close(cv, cf)
    cf2 = cg.make_consistent_winding(cv2, cf2)
    return cv2, cf2, cg.is_edge_manifold_closed(cf2)
