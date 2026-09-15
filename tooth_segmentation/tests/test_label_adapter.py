"""Tests the integration path a pretrained model would use.

Simulates the real MeshSegNet workflow: decimate to ~10k cells, label there,
propagate back to full resolution, extract independent watertight meshes."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import core_geometry as cg
from tooth_segmentation import label_adapter as la
from arch_fixture import synthetic_arch

N_TEETH = 5

def ground_truth_labels(verts, faces, centres, crown_r=3.2):
    """Label each face by nearest tooth centre, gingiva beyond the crown."""
    fc = la.face_centroids(verts, faces)
    d = np.linalg.norm(fc[:, None, :2] - centres[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    labels = np.where(d.min(axis=1) < crown_r, nearest + 1, la.GINGIVA_LABEL)
    return labels.astype(np.int64)


def test_decimation_roundtrip():
    """THE integration test: models infer at ~10k cells, scans are 200k+."""
    verts, faces, centres = synthetic_arch(n_teeth=N_TEETH, grid=170, blend=3.0)
    truth = ground_truth_labels(verts, faces, centres)
    fine_c = la.face_centroids(verts, faces)
    print(f"Full mesh: {len(faces):,} faces")

    rng = np.random.default_rng(0)
    for target in (2000, 5000, 10000):
        pick = rng.choice(len(faces), size=min(target, len(faces)), replace=False)
        coarse_c, coarse_lab = fine_c[pick], truth[pick]
        lifted = la.propagate_labels(coarse_c, coarse_lab, fine_c, k=3)
        acc = (lifted == truth).mean()
        print(f"  {target:>6,} cells -> full res: accuracy {acc:.2%}")
        assert acc > 0.95, f"propagation lost too much at {target} cells: {acc:.2%}"
    print("PASS  decimation round-trip: labels survive the resolution gap")


def test_knn_beats_nearest_at_boundaries():
    verts, faces, centres = synthetic_arch(n_teeth=N_TEETH, grid=170, blend=3.0)
    truth = ground_truth_labels(verts, faces, centres)
    fine_c = la.face_centroids(verts, faces)
    rng = np.random.default_rng(1)
    pick = rng.choice(len(faces), size=4000, replace=False)

    a1 = (la.propagate_labels(fine_c[pick], truth[pick], fine_c, k=1) == truth).mean()
    a3 = (la.propagate_labels(fine_c[pick], truth[pick], fine_c, k=3) == truth).mean()
    print(f"PASS  k=1 {a1:.2%} vs k=3 {a3:.2%}  (voting helps at seams: {a3>=a1})")
    assert a3 >= a1 - 0.005


def test_speckle_cleanup():
    verts, faces, centres = synthetic_arch(n_teeth=N_TEETH, grid=150, blend=3.0)
    truth = ground_truth_labels(verts, faces, centres)
    noisy = truth.copy()
    rng = np.random.default_rng(2)
    speckle = rng.choice(np.where(truth == la.GINGIVA_LABEL)[0], size=300, replace=False)
    noisy[speckle] = 1                       # stray "tooth 1" faces across the gingiva

    cleaned, stats = la.clean_labels(faces, noisy, min_faces=50)
    stray_after = int(((cleaned == 1) & (truth == la.GINGIVA_LABEL)).sum())
    print(f"PASS  speckle cleanup: 300 stray faces -> {stray_after} remaining "
          f"(label 1 kept {stats[1]['kept']:,}, removed {stats[1]['removed']:,})")
    assert stray_after < 30


def test_independent_watertight_meshes():
    verts, faces, centres = synthetic_arch(n_teeth=N_TEETH, grid=170, blend=3.0)
    truth = ground_truth_labels(verts, faces, centres)
    cleaned, _ = la.clean_labels(faces, truth)
    cands = la.labels_to_candidates(verts, faces, cleaned)
    print(f"\n{len(cands)} candidates from a {N_TEETH}-tooth arch:")

    meshes = {}
    for c in cands:
        res = la.extract_tooth_mesh(verts, faces, c.face_mask)
        assert res is not None, f"{c.id} produced no mesh"
        cv, cf, watertight = res
        meshes[c.id] = (cv, cf)
        vol = cg.signed_volume(cv, cf)
        print(f"  {c.id}: {c.triangle_count:>5,} tris, area {c.surface_area:6.1f}mm2, "
              f"vol {abs(vol):6.1f}mm3, watertight={watertight}, warnings={len(c.warnings)}")
        assert watertight, f"{c.id} not watertight"

    assert len(cands) == N_TEETH, f"expected {N_TEETH} candidates, got {len(cands)}"

    # spec 12: moving one tooth must not touch another
    ids = list(meshes)
    before = meshes[ids[1]][0].copy()
    moved = meshes[ids[0]][0] + np.array([5.0, 0, 0])
    assert np.array_equal(meshes[ids[1]][0], before), "meshes share memory — not independent"
    assert not np.array_equal(moved, meshes[ids[0]][0])
    print("PASS  meshes are independent: moving one leaves the others untouched")


if __name__ == "__main__":
    test_decimation_roundtrip()
    test_knn_beats_nearest_at_boundaries()
    test_speckle_cleanup()
    test_independent_watertight_meshes()
    print("\nLABEL ADAPTER TESTS PASSED")
