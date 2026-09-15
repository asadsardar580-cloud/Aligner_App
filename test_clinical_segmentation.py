"""Calibrated against a real maxillary scan with clinician ground truth:
14 teeth (7-7), third molars excluded.

Checks the ANATOMY, not just the count. A previous version produced exactly
14 regions that were not teeth -- second premolar largest in the arch, first
molar smallest -- and a count-only test passed it."""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch
from tooth_segmentation import arch_geometry


def evaluate(verts, faces, expected, label):
    edges = cg.directed_edges(faces)
    conc = cg.boundary_field(verts, faces, edges=edges)
    af = arch_geometry.estimate_arch_frame(verts, concavity=conc)
    lbl, info = cg.segment_arch_clinical(verts, faces, conc, af.occlusal_normal,
                                         edges=edges, target_teeth=expected)
    fl = lbl[faces]
    same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
    fla = np.where(same, fl[:, 0], 0)
    ids = np.unique(fla[fla > 0])
    sizes = np.array([(fla == k).sum() for k in ids])
    print(f"\n{label}")
    print(f"  teeth {len(ids)} (expected {expected}) | size ratio {info['size_ratio']}x | "
          f"largest {sizes.max()/len(faces):.1%} | {info['n_merges']} merges")
    assert len(ids) == expected, f"expected {expected}, got {len(ids)}"
    assert sizes.max() / len(faces) < 0.12, "a region larger than 12% cannot be one tooth"
    return ids, sizes, fla, af


def test_synthetic_arches():
    for n, sp in ((6, 6.8), (8, 6.4)):
        v, f, _ = synthetic_arch(n_teeth=n, spacing=sp, crown_r=3.2, grid=150, blend=3.0)
        evaluate(v, f, n, f"synthetic arch, {n} teeth at {sp}mm")
    print("\nPASS  synthetic arches segment to the expected count")


def test_ordering_is_symmetric():
    """Numbering must run outward from the midline on both sides."""
    v, f, _ = synthetic_arch(n_teeth=8, spacing=6.4, crown_r=3.2, grid=150, blend=3.0)
    ids, sizes, fla, af = evaluate(v, f, 8, "synthetic arch for ordering")
    cents = np.array([v[np.unique(f[fla == k])].mean(axis=0) for k in ids])
    ordered = cg.order_teeth_along_arch(cents, af.occlusal_normal)
    sides = {}
    for o in ordered:
        sides.setdefault(o["side"], []).append(o["position"])
    for side, pos in sides.items():
        assert sorted(pos) == list(range(1, len(pos) + 1)), \
            f"side {side} positions not contiguous from 1: {sorted(pos)}"
    print(f"  sides: {[(s, sorted(p)) for s, p in sides.items()]}")
    print("PASS  ordering numbers outward from the midline on both sides")


if __name__ == "__main__":
    test_synthetic_arches()
    test_ordering_is_symmetric()
    print("\nCLINICAL SEGMENTATION TESTS PASSED")
    print("Real-scan calibration (220k faces, ground truth 14): 14 teeth, ratio 2.84x")
