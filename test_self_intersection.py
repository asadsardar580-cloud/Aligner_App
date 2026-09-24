"""self_intersection.py scored against an INDEPENDENT ground truth.

The reference below uses exact rational arithmetic (fractions.Fraction built
from the float coordinates, so no rounding anywhere) and a different algorithm
(orientation predicates) from the float64 detector. Agreement between the two
is the evidence; neither is trusted on its own. Controls with known answers
come first, per CLAUDE.md's rule that a fixture must reproduce the defect.
"""
import itertools
from fractions import Fraction as Fr

import numpy as np

import self_intersection as si


# ---------------------------------------------------------------- exact reference
def _f(p):
    return tuple(Fr(float(x)) for x in p)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _orient3d(a, b, c, d):
    u, v, w = _sub(b, a), _sub(c, a), _sub(d, a)
    return (u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def _orient2d(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _sgn(x):
    return (x > 0) - (x < 0)


def _on_seg2(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _seg_seg2(p, q, r, s):
    o1, o2, o3, o4 = _orient2d(p, q, r), _orient2d(p, q, s), _orient2d(r, s, p), _orient2d(r, s, q)
    if _sgn(o1) * _sgn(o2) < 0 and _sgn(o3) * _sgn(o4) < 0:
        return True
    return ((o1 == 0 and _on_seg2(p, q, r)) or (o2 == 0 and _on_seg2(p, q, s))
            or (o3 == 0 and _on_seg2(r, s, p)) or (o4 == 0 and _on_seg2(r, s, q)))


def _pt_in_tri2(p, a, b, c):
    d1, d2, d3 = _orient2d(a, b, p), _orient2d(b, c, p), _orient2d(c, a, p)
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)


def exact_seg_tri(p, q, a, b, c):
    op, oq = _orient3d(a, b, c, p), _orient3d(a, b, c, q)
    if _sgn(op) * _sgn(oq) > 0:
        return False
    if op == 0 and oq == 0:                                     # coplanar
        n = (_orient3d((0, 0, 0), _sub(b, a), _sub(c, a), (1, 0, 0)),
             _orient3d((0, 0, 0), _sub(b, a), _sub(c, a), (0, 1, 0)),
             _orient3d((0, 0, 0), _sub(b, a), _sub(c, a), (0, 0, 1)))
        k = max(range(3), key=lambda i: abs(n[i]))
        keep = [i for i in range(3) if i != k]
        P, Q, A, B, C = [(x[keep[0]], x[keep[1]]) for x in (p, q, a, b, c)]
        return (_pt_in_tri2(P, A, B, C) or _pt_in_tri2(Q, A, B, C)
                or _seg_seg2(P, Q, A, B) or _seg_seg2(P, Q, B, C) or _seg_seg2(P, Q, C, A))
    s = [_sgn(_orient3d(p, q, a, b)), _sgn(_orient3d(p, q, b, c)), _sgn(_orient3d(p, q, c, a))]
    return all(x >= 0 for x in s) or all(x <= 0 for x in s)


def exact_tri_tri(A, B):
    for s, t in ((0, 1), (1, 2), (2, 0)):
        if exact_seg_tri(A[s], A[t], *B) or exact_seg_tri(B[s], B[t], *A):
            return True
    return False


# ---------------------------------------------------------------- controls
def soup(tris):
    V = np.array([p for t in tris for p in t], float)
    F = np.arange(len(V)).reshape(-1, 3)
    return V, F


def count(V, F):
    rep = si.self_intersection_report(V, F)
    assert rep["measured"]
    return rep["intersecting_pairs"]


def test_controls_with_known_answers():
    T = [(0, 0, 0), (4, 0, 0), (0, 4, 0)]
    assert count(*soup([T, [(1, 1, -1), (1, 1, 1), (3, 3, 0)]])) == 1      # pierce
    assert count(*soup([T, [(1, 1, 0), (2, 1, 3), (1, 2, 3)]])) == 1      # vertex ON face (touch)
    assert count(*soup([T, [(1, 1, 1e-3), (2, 1, 3), (1, 2, 3)]])) == 0   # 1 micron clear
    assert count(*soup([T, [(1, 1, 0), (3, 1, 0), (1, 2, 0)]])) == 1      # coplanar overlap
    assert count(*soup([T, [(5, 5, 0), (6, 5, 0), (5, 6, 0)]])) == 0      # coplanar disjoint
    assert count(*soup([T, [(0, 0, 5), (4, 0, 5), (0, 4, 5)]])) == 0      # parallel planes
    assert count(*soup([T, [(2, -1, 0), (2, 1, 0), (2, 0, 3)]])) == 1     # edge-edge crossing in plane


def test_self_touch_of_two_cubes_sharing_an_edge_is_caught():
    """The exact B2 signature: two solids meeting along an EDGE with distinct
    vertex ids. Index topology is clean; geometry touches itself."""
    def cube(o):
        o = np.asarray(o, float)
        v = np.array(list(itertools.product([0, 1], repeat=3)), float) + o
        f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return v, f
    v1, f1 = cube((0, 0, 0))
    v2, f2 = cube((1, 1, 0))                     # shares the edge x=1,y=1 only
    V = np.vstack([v1, v2])
    F = np.vstack([f1, f2 + 8])
    assert count(V, F) > 0
    v3, f3 = cube((3, 0, 0))                     # clear by 2 mm
    assert count(np.vstack([v1, v3]), np.vstack([f1, f3 + 8])) == 0


def test_folded_flap_is_caught():
    V = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.8, 0]], float)
    F = np.array([[0, 1, 2], [1, 0, 3]])          # consistent winding, folded onto itself
    assert count(V, F) == 1
    V2 = V.copy()
    V2[3] = [0.5, -1, 0]                          # flat, unfolded neighbour
    assert count(V2, F) == 0


def test_clean_closed_sphere_reports_zero():
    import deform_construction  # noqa: F401  (import check only)
    V, F = icosphere(3)
    assert count(V, F) == 0
    V2 = V.copy()
    V2[0] = -V2[0] * 1.5                          # drag one vertex through the far side
    assert count(V2, F) > 0


def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(level):
        mid = {}
        V = list(map(tuple, V))
        newF = []
        for a, b, c in F:
            ids = []
            for u, w in ((a, b), (b, c), (c, a)):
                k = (min(u, w), max(u, w))
                if k not in mid:
                    m = (np.array(V[u]) + np.array(V[w])) / 2
                    V.append(tuple(m))
                    mid[k] = len(V) - 1
                ids.append(mid[k])
            ab, bc, ca = ids
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.array(V)
        F = np.array(newF)
    V = V / np.linalg.norm(V, axis=1)[:, None] * 10.0
    return V, F


# ---------------------------------------------------------------- agreement with exact
def test_float_detector_agrees_with_exact_reference_on_random_soups():
    rng = np.random.default_rng(7)
    fn = fp = total_true = 0
    for trial in range(12):
        n = 40
        V = rng.uniform(0, 10, (n * 3, 3))
        V = np.round(V, 3)                                   # exact decimal-ish inputs
        F = np.arange(n * 3).reshape(-1, 3)
        rep_pairs = set()
        pairs = si.candidate_pairs(V, F)
        hit, _ = si._pairs_intersect(V, F, pairs, si.DEFAULT_TOUCH_TOL_MM)
        rep_pairs = {tuple(p) for p in pairs[hit]}
        for i, j in itertools.combinations(range(n), 2):
            truth = exact_tri_tri([_f(V[k]) for k in F[i]], [_f(V[k]) for k in F[j]])
            got = (i, j) in rep_pairs
            total_true += truth
            fn += truth and not got
            fp += got and not truth
    assert total_true > 20, "fixture too sparse to mean anything"
    assert fn == 0, f"float detector MISSED {fn} true intersections"
    assert fp == 0, f"float detector flagged {fp} pairs the exact reference calls clear"


def test_broad_phase_never_misses_an_overlapping_box():
    rng = np.random.default_rng(3)
    V = rng.uniform(0, 20, (900, 3))
    V[:300] *= [1, 1, 0.05]                                  # mix huge flat and small triangles
    F = np.arange(900).reshape(-1, 3)
    got = {tuple(p) for p in si.candidate_pairs(V, F, pad=0.0)}
    lo, hi = V[F].min(1), V[F].max(1)
    for i, j in itertools.combinations(range(len(F)), 2):
        if np.all(lo[i] <= hi[j]) and np.all(lo[j] <= hi[i]):
            assert (i, j) in got, f"broad phase dropped overlapping pair {(i, j)}"


if __name__ == "__main__":
    # run_all_tests.py executes entries as `python <file>` and trusts the exit
    # code. Without this block this file would exit 0 having run NOTHING.
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))

def test_giant_floor():
    import numpy as np
    import self_intersection as si
    rng = np.random.default_rng(42)
    V = rng.uniform(0, 10, (100, 3))
    V[0] = [-1000, -1000, -1]
    V[1] = [1000, -1000, -1]
    V[2] = [0, 1000, -1]
    V[3] = [0, 0, -2]
    V[4] = [1, 0, 1]
    V[5] = [0, 1, 1]
    F = np.arange(99).reshape(-1, 3)
    c = count(V, F)
    assert c > 0

def test_subset_property():
    import numpy as np
    import self_intersection as si
    rng = np.random.default_rng(8)
    n = 300
    V = rng.uniform(0, 50, (n * 3, 3))
    F = np.arange(n * 3).reshape(-1, 3)
    V[0:3] = V[3:6] + [0.1, 0, 0]
    V[9:12] = V[12:15] + [0, 0.1, 0]
    rep = si.self_intersection_report(V, F)
    total_pairs = rep['intersecting_pairs']
    for _ in range(5):
        mask = rng.uniform(0, 1, n) > 0.5
        subset_F = F[mask]
        subset_rep = si.self_intersection_report(V, subset_F)
        assert subset_rep['intersecting_pairs'] <= total_pairs
        
        # Test exact subset agreement
        # Every pair in subset MUST exist in full mesh
        pairs_subset = si.candidate_pairs(V, subset_F)
        if len(pairs_subset) > 0:
            hit, _ = si._pairs_intersect(V, subset_F, pairs_subset, si.DEFAULT_TOUCH_TOL_MM)
            sub_pairs = pairs_subset[hit]
            
            pairs_full = si.candidate_pairs(V, F)
            hit_full, _ = si._pairs_intersect(V, F, pairs_full, si.DEFAULT_TOUCH_TOL_MM)
            full_pairs = pairs_full[hit_full]
            
            # Map subset pairs to full mesh indices
            # subset_F[p] maps to original F indices
            orig_indices = np.where(mask)[0]
            
            full_set = set((min(fp[0], fp[1]), max(fp[0], fp[1])) for fp in full_pairs)
            for p in sub_pairs:
                a, b = orig_indices[p[0]], orig_indices[p[1]]
                a, b = min(a, b), max(a, b)
                assert (a, b) in full_set, f'Pair {a},{b} found in subset but not in full mesh'
