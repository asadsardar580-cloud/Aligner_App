"""Segmentation benchmarking and the hybrid fallback chain.

The most important assertion in this file is that `benchmark()` REFUSES to score
a model against its own output. This repository already carries four mutually
inconsistent accuracy claims inherited from prose, none of which any code
reproduces; a fifth produced by self-comparison would read as validation and
mean nothing. The refusal is the feature.
"""
import json
import os
import shutil
import tempfile

import numpy as np

import benchmark_segmentation as bs
import core_geometry as cg
import segmentation_fallback as sf
from tooth_fixture import flat_molar_on_base


def _arch():
    v, f, r = flat_molar_on_base()
    return v, f, r, np.linalg.norm(v[:, :2], axis=1)


# ------------------------------------------------------------- metrics ---

def test_iou_is_exact_on_known_overlaps():
    truth = np.array([0, 0, 36, 36, 36, 36])
    perfect = truth.copy()
    assert bs.mean_iou(perfect, truth) == 1.0

    # 2 of 4 predicted correctly, union 4 -> IoU 0.5
    half = np.array([0, 0, 36, 36, 0, 0])
    assert abs(bs.per_class_iou(half, truth)[36] - 0.5) < 1e-12

    # A class in neither must not be scored: counting agreement about a tooth
    # the patient does not have would inflate mIoU by the missing teeth.
    ious = bs.per_class_iou(np.array([0, 0, 36, 36]), np.array([0, 0, 36, 36]))
    assert set(ious) == {36}, f"scored classes that are absent: {sorted(ious)}"
    print("PASS  IoU exact: perfect 1.0, half-overlap 0.5, absent classes unscored")


def test_iou_refuses_a_shape_mismatch():
    try:
        bs.per_class_iou(np.zeros(5), np.zeros(7))
        raise AssertionError("scored two different meshes against each other")
    except ValueError as e:
        assert "not the same mesh" in str(e)
    print("PASS  a label-count mismatch is refused, not silently broadcast")


def test_fdi_accuracy_separates_the_two_error_kinds():
    """A tooth labelled as its NEIGHBOUR gives the wrong Wheeler root length and
    therefore the wrong C_res. Labelled as gingiva, it is merely lost."""
    truth = np.array([36, 36, 36, 36])
    as_neighbour = np.array([36, 36, 37, 37])
    as_gingiva = np.array([36, 36, 0, 0])

    a = bs.fdi_accuracy(as_neighbour, truth)
    b = bs.fdi_accuracy(as_gingiva, truth)
    assert a["mislabelled_as_other_tooth"] == 2 and a["mislabelled_as_gingiva"] == 0
    assert b["mislabelled_as_gingiva"] == 2 and b["mislabelled_as_other_tooth"] == 0
    assert abs(a["accuracy"] - 0.5) < 1e-12
    print("PASS  neighbour-confusion and gingiva-loss counted separately")


def test_chamfer_and_hausdorff_disagree_where_they_should():
    """Chamfer is a mean and forgives one bad point; Hausdorff does not. That is
    why both are reported."""
    a = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    b = a.copy()
    assert bs.chamfer_distance(a, b) == 0.0 and bs.hausdorff_distance(a, b) == 0.0

    outlier = np.vstack([a, [[0.0, 4.0, 0.0]]])
    ch = bs.chamfer_distance(outlier, a)
    ha = bs.hausdorff_distance(outlier, a)
    assert ha >= 4.0 - 1e-9, f"hausdorff missed a 4mm excursion: {ha}"
    assert ch < ha / 2, f"chamfer {ch} should be far more forgiving than hausdorff {ha}"
    print(f"PASS  one 4mm excursion: chamfer {ch:.2f}mm vs hausdorff {ha:.2f}mm")


# --------------------------------------------------- the refusal itself ---

def test_benchmark_refuses_ground_truth_annotated_by_the_model():
    tmp = tempfile.mkdtemp(prefix="gt_")
    try:
        p = os.path.join(tmp, "gt.json")
        for annotator in ("TGN", "ToothGroupNetwork", "prediction", "model"):
            json.dump({"jaw": "lower", "labels": [0, 0, 36], "annotator": annotator},
                      open(p, "w", encoding="utf-8"))
            try:
                bs.load_ground_truth(p)
                raise AssertionError(f"accepted {annotator!r} as ground truth")
            except bs.NoGroundTruth as e:
                assert "not ground truth" in str(e)
        print("PASS  4 model-authored annotators refused as ground truth")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_benchmark_refuses_unattributed_or_mismatched_ground_truth():
    tmp = tempfile.mkdtemp(prefix="gt_")
    try:
        p = os.path.join(tmp, "gt.json")
        json.dump({"jaw": "lower", "labels": [0, 0, 36]}, open(p, "w", encoding="utf-8"))
        try:
            bs.load_ground_truth(p)
            raise AssertionError("accepted ground truth with no annotator")
        except bs.NoGroundTruth as e:
            assert "does not say who annotated" in str(e)

        json.dump({"jaw": "lower", "labels": [0, 0, 36], "annotator": "Dr Smith"},
                  open(p, "w", encoding="utf-8"))
        try:
            bs.load_ground_truth(p, expect_vertices=5000)
            raise AssertionError("accepted labels for a different mesh")
        except bs.NoGroundTruth as e:
            assert "not the same scan" in str(e)

        # A properly attributed file for the right mesh loads.
        gt = bs.load_ground_truth(p, expect_vertices=3)
        assert gt["annotator"] == "Dr Smith"
        print("PASS  unattributed refused, mesh mismatch refused, valid GT loads")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_this_repo_has_no_ground_truth_and_the_code_says_so():
    """The file that looks like ground truth is the model's own prediction."""
    if not os.path.exists("case_lower.stl_output.json"):
        print("PASS  (no prediction file present to mistake for ground truth)")
        return
    d = json.load(open("case_lower.stl_output.json", encoding="utf-8"))
    assert "annotator" not in d, \
        "the prediction file now claims an annotator — check it is not being " \
        "passed off as ground truth"
    try:
        bs.load_ground_truth("case_lower.stl_output.json")
        raise AssertionError("the model's own output was accepted as ground truth")
    except bs.NoGroundTruth:
        pass
    print("PASS  case_lower.stl_output.json is refused as ground truth (it is a prediction)")


# ------------------------------------------- confidence, without truth ---

def test_confidence_breakdown_never_claims_to_be_accuracy():
    v, f, r, rr = _arch()
    labels = np.where(rr < r * 0.95, 36, 0)
    out = bs.confidence_breakdown(labels, v, f, "lower")
    for k in ("miou_estimate", "boundary_clarity", "component_isolation", "overall_score"):
        assert k in out, f"the API contract key {k} is missing"
        assert 0.0 <= out[k] <= 1.0
    assert out["is_measured_accuracy"] is False, \
        "a plausibility proxy must not present as measured accuracy"
    assert "must not be quoted as mIoU" in out["meaning"]
    print(f"PASS  breakdown present (overall {out['overall_score']}), "
          f"explicitly flagged is_measured_accuracy=False")


# ------------------------------------------------------ fallback chain ---

def test_a_clean_segmentation_stays_on_tier_one():
    v, f, r, rr = _arch()
    labels = np.where(rr < r * 0.95, 36, 0)
    out = sf.run(labels, v, f, "lower")
    assert out["repaired_count"] == 0
    assert out["tiers"][sf.TIER_MODEL] == 1
    assert out["needs_review"] == []
    assert np.array_equal(out["labels"], labels), "a clean result was altered"
    print("PASS  a plausible region stays on tier 1 and is not touched")


def test_a_split_label_is_flagged_and_routed_off_tier_one():
    """The specific failure the fallback exists for."""
    v, f, r, rr = _arch()
    labels = np.where((rr < r * 0.40) | ((rr > r * 0.75) & (rr < r * 0.95)), 36, 0)
    out = sf.run(labels, v, f, "lower")          # no graph -> must route to manual
    row = out["teeth"][0]
    assert row["status"] != sf.STATUS_OK, "a split label stayed OK"
    assert row["connected_fraction"] < sf.MIN_CONNECTED_FRACTION
    assert row["tier"] == sf.TIER_MANUAL, \
        "without a barrier graph the honest move is manual, not a guess"
    assert "no barrier graph" in row["reason"]
    assert 36 in out["needs_review"]
    print(f"PASS  split label ({row['connected_fraction']*100:.0f}% largest island) "
          f"-> {row['tier']}, {row['status']}")


def test_geodesic_recovery_repairs_a_split_label_when_a_graph_exists():
    v, f, r, rr = _arch()
    edges = cg.directed_edges(f)
    conc = cg.boundary_field(v, f, edges=edges)
    graph = cg.build_barrier_graph(v, f, conc, edges=edges)

    labels = np.where((rr < r * 0.40) | ((rr > r * 0.75) & (rr < r * 0.95)), 36, 0)
    out = sf.run(labels, v, f, "lower", graph=graph, concavity=conc)
    row = out["teeth"][0]

    assert row["tier"] == sf.TIER_GEODESIC, f"expected geodesic recovery, got {row['tier']}"
    assert row["status"] == sf.STATUS_REVIEW, \
        "a repaired region must be REVIEW_REQUIRED — a repair is not a confirmation"
    assert out["repaired_count"] == 1
    assert row["recovered_vertices"] > 10
    # The FDI is kept from the model: the flood cannot name a tooth.
    assert set(np.unique(out["labels"])) <= {0, 36}
    print(f"PASS  geodesic recovery: {row['recovered_vertices']} verts re-grown, "
          f"FDI 36 kept, marked {row['status']}")


def test_the_threshold_says_it_is_not_an_miou():
    v, f, r, rr = _arch()
    out = sf.run(np.where(rr < r * 0.95, 36, 0), v, f, "lower")
    assert "NOT an mIoU" in out["threshold"]["note"]
    assert out["threshold"]["basis"] == "software heuristic"
    print("PASS  the fallback threshold states it is not an mIoU threshold")


if __name__ == "__main__":
    test_iou_is_exact_on_known_overlaps()
    test_iou_refuses_a_shape_mismatch()
    test_fdi_accuracy_separates_the_two_error_kinds()
    test_chamfer_and_hausdorff_disagree_where_they_should()
    test_benchmark_refuses_ground_truth_annotated_by_the_model()
    test_benchmark_refuses_unattributed_or_mismatched_ground_truth()
    test_this_repo_has_no_ground_truth_and_the_code_says_so()
    test_confidence_breakdown_never_claims_to_be_accuracy()
    test_a_clean_segmentation_stays_on_tier_one()
    test_a_split_label_is_flagged_and_routed_off_tier_one()
    test_geodesic_recovery_repairs_a_split_label_when_a_graph_exists()
    test_the_threshold_says_it_is_not_an_miou()
    print("\nALL BENCHMARK + FALLBACK TESTS PASSED")
