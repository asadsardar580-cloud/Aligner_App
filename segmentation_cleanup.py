"""Tidy a label array: one region per tooth, no specks.

WHAT THIS FIXES, and it is two distinct complaints that share a cause.

  * A tooth label that falls into several pieces. Only one of them is the
    tooth; the others are stray triangles the model put on a neighbour or out
    on the gingiva. Measured on `case_lower.stl` after the model runs, the
    largest piece of FDI 37 holds 57% of its label and FDI 44's holds 53%.
  * Gingiva specks scattered over a crown, and tooth specks out on the
    gingiva. Individually tiny, collectively the thing a clinician sees
    first, and they make a crown's rim ragged where it matters most.

TWO RULES, APPLIED IN THIS ORDER, and the order is load-bearing:

  1. For each tooth, keep its LARGEST connected region; every other piece of
     that tooth stops being that tooth and becomes background.
  2. Any region smaller than `MIN_ISLAND_FACES` - of any label, background
     included - is reassigned to the majority label of the anatomy
     immediately around it.

Rule 2 after rule 1, because rule 1 produces small orphans and rule 2 is what
gives them to whatever actually surrounds them. Rule 2 BEFORE rule 1 would
instead hand a crown's stray speck straight back to the crown.

CONNECTED MEANS CONNECTED THROUGH THE MESH, never through proximity. Two
crowns touching at an interproximal contact are two regions to the surface
and one region to any distance test, and the scanner never saw that contact
point anyway (s.16).

SIZE IS COUNTED IN FACES, not vertices, because a face is a piece of surface
and a vertex is not: a long thin ribbon of 60 vertices can carry fewer than
50 faces, and it is the area that decides whether a speck is visible.

THIS CHANGES LABELS, NEVER GEOMETRY. No vertex moves, no face is added or
removed, and the array comes back the same length - so every vertex id the
client holds keeps its meaning (CLAUDE.md non-negotiable 2).
"""
from __future__ import annotations

import numpy as np

#: A region smaller than this is a speck, not anatomy. ENGINEERING BOUND, not
#: a clinical tolerance: on a scan of this density 50 faces is roughly a
#: quarter of a square millimetre of surface, far below any real feature of a
#: crown or a papilla, and far above the one- and two-triangle noise that the
#: model scatters at a boundary.
MIN_ISLAND_FACES = 50

#: Rule 2 can cascade - reassigning one island can make its neighbour the new
#: smallest - so it runs to a fixed point. Bounded, because a pathological
#: alternation must terminate rather than spin.
MAX_PASSES = 8


def _vertex_adjacency(faces, n_verts):
    """Mesh edges as a symmetric sparse matrix."""
    from scipy.sparse import coo_matrix

    f = np.asarray(faces, np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    data = np.ones(len(e) * 2, np.int8)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    return coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts)).tocsr()


def _regions(labels, faces, adj, value):
    """Connected regions of `value`, as (vertex mask, face count) pairs.

    A region is connected through mesh edges whose BOTH ends carry `value`,
    which is what makes it one patch of surface rather than a scatter that
    happens to share a label.
    """
    from scipy.sparse.csgraph import connected_components

    sel = np.flatnonzero(labels == value)
    if not len(sel):
        return []
    sub = adj[sel][:, sel]
    n, comp = connected_components(sub, directed=False)

    f = np.asarray(faces, np.int64)
    member = -np.ones(len(labels), np.int64)
    member[sel] = comp
    # A face belongs to a region when all three of its corners do.
    fc = member[f]
    whole = (fc[:, 0] == fc[:, 1]) & (fc[:, 1] == fc[:, 2]) & (fc[:, 0] >= 0)
    face_counts = np.bincount(fc[whole, 0], minlength=n) if whole.any() \
        else np.zeros(n, np.int64)

    out = []
    for k in range(n):
        out.append((sel[comp == k], int(face_counts[k])))
    return out


def _surrounding_majority(labels, adj, ids, exclude):
    """The commonest label on the anatomy immediately around `ids`.

    Neighbours of the region that are not in it. Returns None when the region
    has no neighbours at all - an isolated shell, which must be left alone
    rather than guessed at.
    """
    nb = np.unique(adj[ids].indices)
    nb = nb[~np.isin(nb, ids)]
    if not len(nb):
        return None
    vals = labels[nb]
    vals = vals[vals != exclude] if (vals != exclude).any() else vals
    if not len(vals):
        return None
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(counts)])


def clean_labels(verts, faces, labels, background=0,
                 min_island_faces=MIN_ISLAND_FACES, protected=None):
    """Apply both rules. Returns (labels, report); the input is not mutated.

    `protected` is a boolean mask of vertices a clinician has set by hand,
    which are never rewritten. NOTHING POPULATES IT TODAY - the brush in this
    app is a SELECTION tool and the backend stores no per-vertex manual label
    - so the parameter exists to make the guarantee structural rather than to
    describe a feature that ships. When brush label editing lands it passes
    its mask here and the guarantee is already enforced and already tested.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    lab = np.asarray(labels).astype(np.int64).reshape(-1).copy()
    if len(lab) != len(v):
        raise ValueError(f"{len(lab)} labels for {len(v)} vertices")

    keep = (np.zeros(len(lab), bool) if protected is None
            else np.asarray(protected, bool).reshape(-1).copy())
    if len(keep) != len(lab):
        raise ValueError(f"protected mask is {len(keep)} long for "
                         f"{len(lab)} labels")

    adj = _vertex_adjacency(f, len(lab))
    before = lab.copy()
    report = {"min_island_faces": int(min_island_faces),
              "protected_vertices": int(keep.sum()),
              "dropped_secondary_regions": [], "reassigned_islands": [],
              "passes": 0}

    # --- rule 1: one region per tooth ------------------------------------
    for value in sorted(set(int(x) for x in np.unique(lab)) - {background}):
        regions = _regions(lab, f, adj, value)
        if len(regions) <= 1:
            continue
        regions.sort(key=lambda r: (r[1], len(r[0])), reverse=True)
        for ids, nfaces in regions[1:]:
            movable = ids[~keep[ids]]
            if not len(movable):
                continue
            lab[movable] = background
            report["dropped_secondary_regions"].append(
                {"fdi": value, "vertices": int(len(movable)),
                 "faces": int(nfaces)})

    # --- rule 2: specks go to whatever surrounds them ---------------------
    for _ in range(MAX_PASSES):
        report["passes"] += 1
        changed = 0
        for value in sorted(set(int(x) for x in np.unique(lab))):
            for ids, nfaces in _regions(lab, f, adj, value):
                if nfaces >= min_island_faces:
                    continue
                movable = ids[~keep[ids]]
                if not len(movable):
                    continue
                target = _surrounding_majority(lab, adj, ids, value)
                if target is None or target == value:
                    continue
                lab[movable] = target
                changed += 1
                report["reassigned_islands"].append(
                    {"from": value, "to": int(target),
                     "vertices": int(len(movable)), "faces": int(nfaces)})
        if not changed:
            break

    report["vertices_changed"] = int((lab != before).sum())
    report["teeth_before"] = sorted(set(int(x) for x in np.unique(before))
                                    - {background})
    report["teeth_after"] = sorted(set(int(x) for x in np.unique(lab))
                                   - {background})
    report["teeth_lost"] = sorted(set(report["teeth_before"])
                                  - set(report["teeth_after"]))
    report["protected_respected"] = bool(
        np.array_equal(lab[keep], before[keep])) if keep.any() else True
    return lab, report
