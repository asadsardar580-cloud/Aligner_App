"""Which end of this scan is the occlusal end? Decided without any labels.

The census says the tooth labels sit DEEPER than the gingiva labels. That is
either a segmentation that is upside down or a depth convention that is, and
every downstream conclusion depends on which. s.20.1 settled the same question
for the viewport and its lesson applies here: decide it from the SHAPE, with
more than one independent signal, and log a disagreement rather than picking.

THREE SIGNALS, none of which uses a label:

  skewness   cusp tips are a sparse scatter reaching past the body of the
             cast, so the third moment of the projection leans occlusally.
  clusters   a thin slab through the crowns cuts ~16 separate teeth; a thin
             slab through the gingiva cuts one continuous band. This is the
             strongest of the three and it is close to a definition.
  area       the occlusal end tapers to cusps; the tissue end is the full
             cross-section of the cast.

    .venv\\Scripts\\python.exe scratch\\which_end_is_occlusal.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import core_geometry as cg                                   # noqa: E402
import stl_io                                                # noqa: E402
from segmentation_diagnostics import occlusal_axis                    # noqa: E402


def slab_clusters(v, f, keep, link_mm):
    """Connected components of the vertices inside a slab, linked through the
    mesh's own edges - so 'separate' means separate ANATOMY, not separate
    points."""
    idx = np.flatnonzero(keep)
    if len(idx) < 10:
        return 0, 0
    remap = -np.ones(len(v), np.int64)
    remap[idx] = np.arange(len(idx))
    e = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        pair = f[:, [a, b]]
        m = keep[pair[:, 0]] & keep[pair[:, 1]]
        e.append(remap[pair[m]])
    if not len(e):
        return 0, len(idx)
    E = np.concatenate(e, axis=0)
    if not len(E):
        return len(idx), len(idx)
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])),
                   shape=(len(idx), len(idx)))
    n, lab = connected_components(A, directed=False)
    # Ignore specks: a cluster under 30 vertices is triangulation noise.
    sizes = np.bincount(lab)
    return int((sizes >= 30).sum()), len(idx)


def main():
    raw = open("case_lower.stl", "rb").read()
    v0, f0 = stl_io.parse_stl_bytes(raw)
    v0, f0, _ = cg.sanitize_scan(v0, f0)
    v, f, _ = cg.condition_mesh(v0, f0)

    u, _ev = occlusal_axis(v, f)
    h = v @ u
    lo, hi = float(h.min()), float(h.max())
    span = hi - lo
    print("=" * 74)
    print("WHICH END IS OCCLUSAL - three signals, no labels")
    print("=" * 74)
    print(f"u_occ  {np.round(u, 4)}   projection span {span:.2f} mm")

    x = h - h.mean()
    skew = float((x ** 3).mean() / (x.std() ** 3))
    print(f"\n1. SKEWNESS {skew:+.3f}  ->  "
          f"{'+u_occ' if skew > 0 else '-u_occ'} is the occlusal end")

    print("\n2. CLUSTERS in a 1.5 mm slab, stepping along the axis")
    print(f"   {'from +u_occ end':>16} {'clusters':>9} {'verts':>8}")
    step, thick = 1.5, 1.5
    rows = []
    d = 0.0
    while d < span - thick:
        top = hi - d
        keep = (h <= top) & (h > top - thick)
        n, cnt = slab_clusters(v, f, keep, thick)
        rows.append((d, n, cnt))
        print(f"   {d:>13.1f} mm {n:>9} {cnt:>8,}")
        d += step

    near = [n for dd, n, _ in rows if dd <= 4.0]
    far = [n for dd, n, _ in rows if dd >= span - 6.0]
    print(f"\n   near the +u_occ end: {near}")
    print(f"   near the -u_occ end: {far}")
    verdict = ("+u_occ" if (max(near) if near else 0) > (max(far) if far else 0)
               else "-u_occ")
    print(f"   ->  the end that cuts into SEPARATE lumps is {verdict}; "
          f"that is the occlusal end")

    print("\n3. CROSS-SECTION VERTEX COUNT (the occlusal end tapers)")
    first = rows[0][2] if rows else 0
    last = rows[-1][2] if rows else 0
    print(f"   slab at the +u_occ end {first:,} verts; "
          f"at the -u_occ end {last:,} verts")

    print("\n" + "=" * 74)
    agree = (skew > 0) and verdict == "+u_occ"
    print("VERDICT: the three signals AGREE - +u_occ is occlusal"
          if agree else
          "VERDICT: THE SIGNALS DISAGREE - do not trust the depth convention")
    print("=" * 74)


if __name__ == "__main__":
    main()
