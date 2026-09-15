"""Tests for stages 2-5, on synthetic meshes (spec 17: no reliance on
patient STLs for automated tests)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import core_geometry as cg
from tooth_segmentation import mesh_preprocessor, curvature, arch_geometry, SegmentationConfig
from arch_fixture import synthetic_arch


def test_preprocess_welds_and_reports():
    verts, faces, _ = synthetic_arch(n_teeth=4, grid=90)
    # duplicate every vertex to simulate an unwelded STL triangle soup
    soup_verts = verts[faces].reshape(-1, 3)
    soup_faces = np.arange(len(soup_verts)).reshape(-1, 3)
    assert len(soup_verts) == 3 * len(faces)

    v2, f2, rep = mesh_preprocessor.preprocess(soup_verts, soup_faces)
    assert rep.n_duplicate_vertices_merged > 0
    assert len(v2) == len(verts), f"weld should recover {len(verts)} verts, got {len(v2)}"
    assert rep.n_faces == len(faces)
    print(f"PASS  preprocess: welded {rep.n_duplicate_vertices_merged:,} duplicates -> "
          f"{rep.summary()}")


def test_preprocess_is_non_destructive():
    verts, faces, _ = synthetic_arch(n_teeth=3, grid=70)
    vcopy, fcopy = verts.copy(), faces.copy()
    mesh_preprocessor.preprocess(verts, faces)
    assert np.array_equal(verts, vcopy) and np.array_equal(faces, fcopy)
    print("PASS  preprocess: original arrays untouched (spec 4, non-destructive)")


def test_preprocess_detects_degenerate():
    verts, faces, _ = synthetic_arch(n_teeth=3, grid=70)
    bad = np.vstack([faces, [[0, 0, 1], [5, 5, 5]]])   # two degenerate faces
    _, f2, rep = mesh_preprocessor.preprocess(verts, bad)
    assert rep.n_degenerate_faces == 2, rep.n_degenerate_faces
    assert len(f2) == len(faces)
    print(f"PASS  preprocess: detected and removed {rep.n_degenerate_faces} degenerate faces")


def test_curvature_density_independent():
    """Spec 6: must not assume fixed mesh density."""
    ratios = []
    for grid in (90, 130, 180):
        v, f, centres = synthetic_arch(n_teeth=4, grid=grid, blend=3.0)
        feats = curvature.curvature_features(v, f)
        c = feats["concavity"]
        d = np.linalg.norm(v[:, :2] - centres[0], axis=1)
        sulcus = c[(d > 2.8) & (d < 3.6)].mean()
        flat = c[d > 9.0].mean()
        ratios.append(sulcus - flat)
    spread = max(ratios) - min(ratios)
    assert spread < 0.25, f"curvature contrast varies too much with density: {ratios}"
    print(f"PASS  curvature: sulcus contrast stable across 3 mesh densities "
          f"({[round(r,3) for r in ratios]}, spread {spread:.3f})")


def test_arch_frame_survives_rotation():
    """Spec 8: must not assume the STL is axis-aligned."""
    errors = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        R = cg.rotation_matrix_axis_angle(axis, rng.uniform(20, 160))
        v, f, _ = synthetic_arch(n_teeth=5, grid=90, rotate=R)
        frame = arch_geometry.estimate_arch_frame(v)
        expected = R @ np.array([0.0, 0.0, 1.0])     # occlusal was +Z before rotation
        err = np.degrees(np.arccos(np.clip(abs(np.dot(frame.occlusal_normal, expected)), 0, 1)))
        errors.append(err)
    assert max(errors) < 12.0, f"occlusal axis recovery too poor: {errors}"
    print(f"PASS  arch frame: occlusal normal recovered from arbitrary rotations, "
          f"max error {max(errors):.1f}deg  (errors {[round(e,1) for e in errors]})")


def test_arch_frame_orthonormal():
    v, f, _ = synthetic_arch(n_teeth=5, grid=90)
    fr = arch_geometry.estimate_arch_frame(v)
    ax = [fr.occlusal_normal, fr.arch_width_axis, fr.arch_depth_axis]
    for i in range(3):
        assert abs(np.linalg.norm(ax[i]) - 1) < 1e-9
        for j in range(i+1, 3):
            assert abs(np.dot(ax[i], ax[j])) < 1e-9
    print(f"PASS  arch frame: orthonormal, explained variance "
          f"{fr.explained_variance.round(3)}")


def test_config_not_hardcoded():
    """Spec 7: weights configurable, not baked into call sites."""
    cfg = SegmentationConfig()
    cfg.boundary.concavity_weight = 99.0
    assert cfg.boundary.concavity_weight == 99.0
    assert SegmentationConfig().boundary.concavity_weight == 1.4, "defaults leaked between instances"
    print("PASS  config: weights configurable and per-instance isolated")


if __name__ == "__main__":
    test_preprocess_welds_and_reports()
    test_preprocess_is_non_destructive()
    test_preprocess_detects_degenerate()
    test_curvature_density_independent()
    test_arch_frame_survives_rotation()
    test_arch_frame_orthonormal()
    test_config_not_hardcoded()
    print("\nSTAGES 2-5 ALL PASSED")
