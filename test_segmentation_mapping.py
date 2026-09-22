"""Labels must belong to the mesh they are read against.

THE DEFECT THESE PIN. The real scan's stored labels were reported as a
segmentation-model failure: every one of the twelve labelled teeth flooded a
quarter of the arch, the best IoU against any label was 0.01-0.05, and FDI 31
spanned 33.8 x 34.4 x 15.3mm. Every one of those numbers was correct. The
conclusion was not. The labels were being indexed against a DIFFERENT VERTEX
ORDERING of the same mesh:

    labels read by index against the app's array   median tooth box 50.71 mm
    the same labels transferred by position        median tooth box 13.97 mm

Four loaders produce 94,848 vertices from this scan in four different orders,
so every count check and every length assertion passed while three of the four
correspondences were wrong. No metric computed on the labels alone can see
this - including IoU, which scores a mis-indexed array against itself
perfectly. It takes a GEOMETRIC check, which is what `segmentation_diagnostics`
is.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import segmentation_diagnostics as sd
import segmentation_providers as sp


# ===========================================================================
# A synthetic mesh whose right answer is known: four separated cubes.
# ===========================================================================

def _cube(cx, cy, cz, half=3.0):
    o = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    v = o * half + np.array([cx, cy, cz], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]], np.int64)
    return v, f


def _four_teeth():
    """Four 6mm cubes 20mm apart, labelled 11, 12, 13, 14."""
    vs, fs, labs = [], [], []
    for i, fdi in enumerate((11, 12, 13, 14)):
        v, f = _cube(i * 20.0, 0.0, 0.0)
        fs.append(f + len(np.concatenate(vs)) if vs else f)
        vs.append(v)
        labs.append(np.full(len(v), fdi, np.int64))
    return (np.concatenate(vs), np.concatenate(fs), np.concatenate(labs))


def test_a_correctly_indexed_label_array_is_reported_coherent():
    v, f, lab = _four_teeth()
    rep = sd.label_report(v, f, lab)
    assert rep["indexed_to_this_mesh"] is True, rep
    assert rep["n_teeth"] == 4
    # a 6mm cube's box diagonal is 6*sqrt(3) = 10.39mm
    assert abs(rep["median_bbox_diagonal_mm"] - 6 * np.sqrt(3)) < 0.01, rep
    assert rep["teeth_with_plausible_size"] == 4
    assert rep["teeth_connected"] == 4
    for row in rep["teeth"]:
        assert row["components"] == 1, row
        assert row["largest_component_fraction"] == 1.0, row
    print(f"PASS  four separated cubes: median box "
          f"{rep['median_bbox_diagonal_mm']}mm, all connected")


def test_a_SHUFFLED_label_array_is_caught_although_every_count_agrees():
    """THE WHOLE POINT. Same mesh, same labels, same counts, same histogram -
    only the correspondence is destroyed. Every length check still passes."""
    v, f, lab = _four_teeth()
    rng = np.random.default_rng(20260921)
    shuffled = lab[rng.permutation(len(lab))]

    assert len(shuffled) == len(lab) == len(v)
    assert np.array_equal(np.bincount(shuffled), np.bincount(lab)), \
        "the fixture must not change the label histogram"

    rep = sd.label_report(v, f, shuffled)
    assert rep["indexed_to_this_mesh"] is False, rep
    assert rep["teeth_with_plausible_size"] == 0, rep
    assert rep["median_bbox_diagonal_mm"] > 20.0, rep
    print(f"PASS  a shuffled array with identical counts is caught: median "
          f"box {rep['median_bbox_diagonal_mm']}mm, "
          f"{rep['teeth_with_plausible_size']}/4 plausible")


def test_the_transfer_restores_a_reordered_array_exactly():
    v, f, lab = _four_teeth()
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(v))
    moved, info = sp.transfer_labels_by_position(v[perm], lab[perm], v)
    assert np.array_equal(moved, lab), "labels did not come back"
    assert info["identity"] is False, "the fixture did not actually reorder"
    assert info["reordered_vertices"] > 0
    assert info["max_position_gap_mm"] == 0.0
    print(f"PASS  transfer restored {len(lab)} labels exactly, "
          f"{info['reordered_vertices']} vertices reordered")


def test_the_transfer_is_the_identity_when_nothing_moved():
    v, f, lab = _four_teeth()
    moved, info = sp.transfer_labels_by_position(v, lab, v)
    assert np.array_equal(moved, lab)
    assert info["identity"] is True, info
    assert info["reordered_vertices"] == 0
    print("PASS  an unreordered provider is reported as the identity")


def test_labels_from_a_DIFFERENT_mesh_are_REFUSED_not_approximated():
    """A nearest-neighbour transfer with a generous tolerance would paper over
    exactly the failure this exists to catch: every query finds SOME nearest
    vertex, so the result looks like a segmentation that merely needs review."""
    v, f, lab = _four_teeth()
    other = v + np.array([0.0, 0.0, 5.0])      # the same mesh, 5mm away
    with pytest.raises(sp.LabelTransferError) as e:
        sp.transfer_labels_by_position(other, lab, v)
    assert "no matching vertex" in str(e.value)
    print(f"PASS  labels from another mesh are refused: {str(e.value)[:80]}")


def test_a_length_mismatch_is_refused_before_anything_geometric():
    v, f, lab = _four_teeth()
    with pytest.raises(sp.LabelTransferError):
        sp.transfer_labels_by_position(v, lab[:-1], v)
    rep = sd.label_report(v, f, lab[:-1])
    assert rep["indexed_to_this_mesh"] is False
    assert "cannot correspond" in rep["reason"]
    print("PASS  a length mismatch is refused, and reported as such")


def test_a_scattered_label_is_reported_as_scattered():
    """One tooth split across two places. This is a MODEL failure, and it must
    read differently from a mapping failure - the median stays plausible."""
    v, f, lab = _four_teeth()
    bad = lab.copy()
    bad[lab == 14] = 11                 # FDI 11 now covers two distant cubes
    rep = sd.label_report(v, f, bad)
    row = next(r for r in rep["teeth"] if r["label"] == 11)
    assert row["components"] == 2, row
    assert row["largest_component_fraction"] < 0.6, row
    assert 11 in rep["teeth_scattered"]
    assert row["plausible_size"] is False
    # and the OTHER teeth are still fine, which is what separates this from a
    # mis-indexed array
    assert rep["teeth_connected"] == 2, rep
    print(f"PASS  a split label reads as 2 components, largest "
          f"{row['largest_component_fraction']}, while its neighbours stay clean")


# ===========================================================================
# The provider registry
# ===========================================================================

def test_the_candidate_provider_raises_rather_than_returning_labels():
    """Same rule as the CBCT interface (CLAUDE.md section 16): a stub that
    returned plausible labels would drive a cut, a C_res and a printed
    aligner, and output from a function of that name is reasonably believed."""
    p = sp.get("meshsegnet")
    st = p.available()
    assert st["available"] is False
    assert "not been installed" in st["reason"]
    with pytest.raises(NotImplementedError) as e:
        p.segment(np.zeros((3, 3)), np.zeros((1, 3), np.int64), "lower")
    assert "candidate" in str(e.value).lower()
    print("PASS  MeshSegNetProvider refuses rather than inventing labels")


def test_the_audit_says_what_is_NOT_verified_rather_than_guessing():
    """Half an audit stated as fact is worse than an audit that names its own
    gaps. Anything needing the upstream repository is NOT_VERIFIED_HERE."""
    audit = sp.get("meshsegnet").audit
    for key in ("license", "windows_cpu_support", "inference_time",
                "axis_convention", "label_convention"):
        assert "NOT_VERIFIED_HERE" in str(audit[key]), (key, audit[key])
    assert audit["package_present_in_this_environment"] is False
    print("PASS  the MeshSegNet audit names its own gaps")


def test_the_registry_has_exactly_one_default_and_it_is_the_shipped_model():
    reg = sp.registry()
    defaults = [k for k, v in reg.items() if v["default"]]
    assert defaults == ["toothgroupnetwork"], defaults
    assert sp.get() is sp.get("toothgroupnetwork")
    with pytest.raises(KeyError):
        sp.get("tcatseg")
    print(f"PASS  registry: {sorted(reg)}, default {defaults[0]}")


# ===========================================================================
# The real scan. Skipped, never faked, when the files are absent.
# ===========================================================================

_STL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "case_lower.stl")
_LAB = _STL + "_output.json"


@pytest.mark.skipif(not (os.path.exists(_STL) and os.path.exists(_LAB)),
                    reason="the real scan or its label file is absent")
def test_the_real_scan_labels_are_coherent_ONCE_TRANSFERRED_BY_POSITION():
    """The measurement that overturned "the segmentation is not spatially
    coherent". Both readings are asserted, so the test proves the defect
    reproduces before it proves the fix."""
    import json
    import stl_io

    v, f = stl_io.parse_stl_bytes(open(_STL, "rb").read())
    labels = np.asarray(json.load(open(_LAB))["labels"], np.int64)
    assert len(labels) == len(v), "the counts agree - that is the trap"

    before = sd.label_report(v, f, labels)
    assert before["indexed_to_this_mesh"] is False, \
        "the defect no longer reproduces; this test is no longer proving anything"
    assert before["median_bbox_diagonal_mm"] > 40.0, before
    assert before["teeth_with_plausible_size"] == 0, before

    src = sp.first_occurrence_vertex_order(v, f)
    moved, info = sp.transfer_labels_by_position(src, labels, v)
    assert info["max_position_gap_mm"] == 0.0, info
    assert info["reordered_vertices"] == len(v), \
        "the two orderings should differ everywhere"

    after = sd.label_report(v, f, moved)
    assert after["indexed_to_this_mesh"] is True, after
    assert after["median_bbox_diagonal_mm"] < 20.0, after
    assert after["teeth_with_plausible_size"] >= 9, after
    print(f"PASS  real scan: median tooth box "
          f"{before['median_bbox_diagonal_mm']}mm -> "
          f"{after['median_bbox_diagonal_mm']}mm, plausible "
          f"{before['teeth_with_plausible_size']}/{before['n_teeth']} -> "
          f"{after['teeth_with_plausible_size']}/{after['n_teeth']}")


@pytest.mark.skipif(not os.path.exists(_STL),
                    reason="the real scan is absent")
def test_first_occurrence_order_is_what_a_dedup_loader_produces():
    """The helper is only correct if it reproduces what the pipeline's loader
    actually did. Checked against Open3D's own dedup, not against a comment."""
    o3d = pytest.importorskip("open3d")
    import stl_io

    v, f = stl_io.parse_stl_bytes(open(_STL, "rb").read())
    mine = sp.first_occurrence_vertex_order(v, f)
    m = o3d.io.read_triangle_mesh(_STL)
    m.remove_duplicated_vertices()
    theirs = np.asarray(m.vertices)
    assert mine.shape == theirs.shape
    assert np.array_equal(mine, theirs), "the helper no longer matches Open3D"
    # and it is genuinely a different order from the app's own array
    assert not np.array_equal(mine, v), \
        "the two orderings are identical here, so nothing is being tested"
    print(f"PASS  first-occurrence order matches Open3D exactly "
          f"({len(mine)} vertices) and differs from the app's array")


# ===========================================================================
# TWO PROVIDERS, TWO INTERNAL ORDERINGS, ONE CANONICAL MESH
#
# The application's mesh is whatever `condition_mesh` produced, and NOTHING a
# provider does to its own copy may become that. The two installed providers
# get there by genuinely different routes:
#
#   ToothGroupNetwork   writes an OBJ, a loader reads it back in ITS order,
#                       and the labels come home by POSITION
#   CrossTooth          classifies FACES and maps them home by COORDINATE
#                       (a 3-neighbour KNN on cell centroids)
#
# so neither one's array is canonical, and a regression has to prove that
# holds even when the orderings differ at every index.
# ===========================================================================

def _two_tooth_mesh():
    """Two separated blocks: unambiguous labels, real face connectivity."""
    import trimesh
    a = trimesh.creation.box(extents=(6, 6, 6))
    b = trimesh.creation.box(extents=(6, 6, 6))
    b.apply_translation([20.0, 0.0, 0.0])
    m = trimesh.util.concatenate([a, b])
    v = np.asarray(m.vertices, float)
    f = np.asarray(m.faces, np.int64)
    lab = np.where(v[:, 0] > 10.0, 41, 31).astype(np.int64)
    return v, f, lab


def test_two_providers_with_DIFFERENT_internal_orderings_agree_on_the_mesh():
    """The regression the whole module exists for, stated as two providers.

    Provider A is given the canonical array and hands its labels back in a
    SHUFFLED order - the ToothGroupNetwork hazard, where a loader reorders
    and every count check still passes. Provider B never indexes an array at
    all; it labels geometry and the labels come home by coordinate - the
    CrossTooth route. Both must land on the same canonical labels.

    If this ever fails, the symptom in the product is not an exception. It is
    teeth scattered across the arch, which reads as a bad model: measured on
    the real scan, the median per-tooth bounding box was 50.71 mm against
    13.97 mm, and that was reported as a segmentation failure for a model
    that had segmented correctly.
    """
    v, f, truth = _two_tooth_mesh()

    # --- provider A: a different ORDER, same points -------------------
    perm = np.random.default_rng(4).permutation(len(v))
    a_verts, a_labels = v[perm], truth[perm]
    assert not np.array_equal(a_verts, v), "the fixture did not reorder"
    moved_a, info_a = sp.transfer_labels_by_position(a_verts, a_labels, v)
    assert info_a["identity"] is False
    assert info_a["reordered_vertices"] == int((perm != np.arange(len(v))).sum())
    assert info_a["max_position_gap_mm"] == 0.0
    np.testing.assert_array_equal(moved_a, truth)

    # --- provider B: no ordering at all, labels carried by coordinate --
    from sklearn.neighbors import KNeighborsClassifier
    cent = v[f].mean(axis=1)
    cell_lab = truth[f[:, 0]]
    keep = np.random.default_rng(9).permutation(len(cent))[: len(cent) // 2]
    knn = KNeighborsClassifier(n_neighbors=3).fit(cent[keep], cell_lab[keep])
    moved_b = knn.predict(v).astype(np.int64)
    np.testing.assert_array_equal(moved_b, truth)

    # --- the point: two routes, one answer ----------------------------
    np.testing.assert_array_equal(moved_a, moved_b)
    for moved in (moved_a, moved_b):
        rep = sd.label_report(v, f, moved)
        assert rep["indexed_to_this_mesh"] is True
        assert {r["label"] for r in rep["teeth"]} == {31, 41}
    print(f"PASS  a shuffled-order provider and a coordinate-keyed provider "
          f"agree on all {len(v)} canonical vertices")


def test_the_fixture_REPRODUCES_the_defect_when_the_transfer_is_skipped():
    """Read by INDEX, the same labels are wrong. Otherwise nothing is tested.

    CLAUDE.md section 5 records the rule: verify a fixture reproduces the bug
    before trusting the regression that it passes.
    """
    v, f, truth = _two_tooth_mesh()
    perm = np.random.default_rng(4).permutation(len(v))
    by_index = truth[perm]                       # what a length check accepts

    assert len(by_index) == len(v), "the count agrees - that is the trap"
    assert sorted(np.bincount(by_index)) == sorted(np.bincount(truth)),         "the per-label COUNTS agree too"
    wrong = int((by_index != truth).sum())
    assert wrong > 0, "the fixture did not reproduce the defect"

    rep = sd.label_report(v, f, by_index)
    diag = max(r["bbox_diagonal_mm"] for r in rep["teeth"])
    good = sd.label_report(v, f, truth)
    diag_ok = max(r["bbox_diagonal_mm"] for r in good["teeth"])
    assert diag > diag_ok * 1.5, (diag, diag_ok)
    print(f"PASS  by index: {wrong} of {len(v)} labels wrong, largest tooth "
          f"box {diag:.2f} mm against {diag_ok:.2f} mm correct - and every "
          f"count check passes")


def test_a_providers_array_is_NEVER_written_back_as_the_session_mesh():
    """Static: no provider result may be stored as `verts`.

    `transfer_labels_by_position` makes the ORDER irrelevant, and this makes
    sure nobody sidesteps it by adopting the provider's array instead - which
    would be the tidy-looking fix and would silently invalidate every vertex
    id the client holds, the extraction mask, and the brush index (s.6).
    """
    import ast

    src = open("api_core.py", encoding="utf-8").read()
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = ast.dump(node.func)
        if "STORE" not in fn or "'put'" not in fn:
            continue
        if len(node.args) < 3:
            continue
        key = node.args[1]
        if not (isinstance(key, ast.Constant) and key.value in
                ("verts", "faces")):
            continue
        val = ast.dump(node.args[2])
        if "used_verts" in val or "provider" in val or "result.labels" in val:
            bad.append((key.value, node.lineno))
    assert not bad, f"a provider array is being stored as the mesh: {bad}"
    print("PASS  no provider array is ever stored as `verts` or `faces`")


if __name__ == "__main__":
    import sys
    failures = []
    for name, obj in sorted(list(globals().items())):
        if not name.startswith("test_"):
            continue
        try:
            obj()
        except Exception as e:                            # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} FAILED")
        sys.exit(1)
    print("\nALL SEGMENTATION MAPPING TESTS PASSED")
