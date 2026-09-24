"""CrossTooth as a second segmentation provider.

WHAT THESE TESTS ARE FOR. Not "does the model segment well" - no test in this
repository can answer that, because there is no independently annotated ground
truth here (CLAUDE.md s.17). They pin the things that CAN be wrong silently:

  * the Windows/CPU bypass leaks state into a process shared with
    ToothGroupNetwork;
  * the feature preparation drifts from the vendored `dataset/data.py`;
  * the class -> FDI table acquires a transposed digit;
  * the shipped loader's 16,000-face truncation comes back;
  * a provider that cannot run returns something plausible instead of raising;
  * the toggle silently falls back to the other model.

The ones that need the checkpoint skip when it is absent. They are never
faked: a segmentation test that passes without a model is worse than no test.
"""
import os
import sys

import numpy as np
import pytest

import crosstooth_bridge as ctb
import segmentation_providers as sp

HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT = os.path.join(HERE, "CrossTooth", "models", "PTv1",
                     "point_best_model.pth")
_SRC = os.path.join(HERE, "CrossTooth", "models", "PTv1",
                    "point_transformer_seg.py")
needs_model = pytest.mark.skipif(
    not (os.path.exists(_CKPT) and os.path.exists(_SRC)),
    reason="the CrossTooth checkout or its checkpoint is absent")


# ===========================================================================
# 1. The class -> FDI table
# ===========================================================================

def test_the_class_to_fdi_table_matches_the_vendored_utils_file():
    """Transcribed against `CrossTooth/utils.py`, not against memory.

    The table is built arithmetically in the bridge precisely so a
    two-digit typo cannot hide in 32 entries; this reads the real file and
    checks the arithmetic against it.
    """
    utils = os.path.join(HERE, "CrossTooth", "utils.py")
    if not os.path.exists(utils):
        pytest.skip("the CrossTooth checkout is absent")
    src = open(utils, encoding="utf-8").read()

    # utils.FDI2color maps FDI -> (hex, name, rgb); invert it by NAME.
    import re
    fdi_by_name = {}
    block = src.split("FDI2color = {", 1)[1]
    for fdi, name in re.findall(r"^\s*(\d\d):\s*\([^,]+,\s*\"(\w+)\"", block,
                                re.M):
        fdi_by_name[name] = int(fdi)
    assert len(fdi_by_name) == 32, fdi_by_name

    for jaw, prefix in (("lower", ("LL", "LR")), ("upper", ("UL", "UR"))):
        table = ctb.CLASS_TO_FDI[jaw]
        assert sorted(table) == list(range(1, 17))
        for k in range(1, 9):
            assert table[k] == fdi_by_name[f"{prefix[0]}{k}"], (jaw, k)
        for k in range(9, 17):
            assert table[k] == fdi_by_name[f"{prefix[1]}{k - 8}"], (jaw, k)

    # And the two jaws never collide - a jaw mix-up must be visible as an
    # out-of-quadrant FDI, which is what jaw_naming.verify_fdi_matches_jaw
    # gates on downstream.
    assert not (set(ctb.CLASS_TO_FDI["lower"].values()) &
                set(ctb.CLASS_TO_FDI["upper"].values()))
    print("PASS  32 class->FDI entries agree with CrossTooth/utils.py")


def test_the_two_discarded_classes_become_background_not_a_tooth():
    """predict.py: `pred_mask[pred_mask == 17] = 0; ... == 18] = 0`.

    The head has 19 outputs and only 17 of them mean anything. A 17 leaking
    through as an FDI would be a tooth that does not exist.
    """
    assert ctb.NUM_CLASSES == 19
    assert set(ctb.BACKGROUND_CLASSES) == {0, 17, 18}
    for jaw in ("lower", "upper"):
        for k in ctb.BACKGROUND_CLASSES:
            assert k not in ctb.CLASS_TO_FDI[jaw]
    print("PASS  classes 0, 17 and 18 are background")


# ===========================================================================
# 2. The feature preparation, against the vendored dataset/data.py
# ===========================================================================

def _ring_mesh(n=64, r=10.0, h=2.0):
    """A closed ring prism. Faces are known, so centroids and normals are."""
    import trimesh
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inner = np.c_[r * np.cos(ang), r * np.sin(ang), np.zeros(n)]
    outer = np.c_[(r + 3) * np.cos(ang), (r + 3) * np.sin(ang),
                  np.full(n, h)]
    v = np.vstack([inner, outer])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + i], [j, n + j, n + i]]
    return trimesh.Trimesh(v, np.array(f), process=False)


def test_the_six_channels_are_face_centroids_and_UNIT_face_normals():
    """vedo's `normals(cells=True)` returns UNIT cell normals.

    A raw cross product is proportional to twice the triangle AREA, so on a
    mesh with mixed triangle sizes three of the six input channels would vary
    over orders of magnitude while the checkpoint expects them in [-1, 1].
    """
    m = _ring_mesh()
    v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    c, nrm, degenerate = ctb.face_centroids_and_normals(v, f)

    assert c.shape == (len(f), 3) and nrm.shape == (len(f), 3)
    assert degenerate == 0
    np.testing.assert_allclose(c, v[f].mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-12)
    # Same orientation as trimesh's own, which is an independent computation.
    dots = (nrm * np.asarray(m.face_normals)).sum(axis=1)
    assert float(dots.min()) > 0.999, float(dots.min())
    print(f"PASS  {len(f)} faces, unit normals, agree with trimesh to "
          f"{1 - float(dots.min()):.2e}")


def test_a_degenerate_face_gets_a_ZERO_normal_and_is_counted():
    """A zero-area triangle has no normal. NaN would poison the whole batch.

    `condition_mesh` drops degenerates upstream, so this is the contract for
    anything that reaches here another way - and the count is reported rather
    than swallowed.
    """
    v = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])
    f = np.array([[0, 1, 2], [0, 1, 3]], np.int64)   # first is collinear
    _, nrm, degenerate = ctb.face_centroids_and_normals(v, f)
    assert degenerate == 1
    assert np.all(np.isfinite(nrm))
    np.testing.assert_array_equal(nrm[0], [0., 0., 0.])
    assert abs(np.linalg.norm(nrm[1]) - 1.0) < 1e-12
    print("PASS  a degenerate face is a zero normal, counted, never NaN")


@needs_model
def test_the_prepared_tensor_has_the_shape_and_range_the_checkpoint_expects():
    """`(1, 6, N)`, XYZ inside the unit sphere, normals untouched.

    `PointcloudNormalize(radius=1)` normalises COLUMNS 0:3 ONLY. Normalising
    all six would rescale the normals by the arch's radius in millimetres and
    hand the model a feature it has never seen.
    """
    a = ctb.CrossToothAdapter(num_points=256)
    m = _ring_mesh()
    prep = a.prepare(np.asarray(m.vertices, float),
                     np.asarray(m.faces, np.int64))
    t = prep["tensor"]
    assert tuple(t.shape) == (1, 6, 256), tuple(t.shape)

    rows = t[0].numpy().T
    radii = np.linalg.norm(rows[:, 0:3], axis=1)
    assert radii.max() <= 1.0 + 1e-6, float(radii.max())
    assert abs(radii.max() - 1.0) < 1e-5, "the largest radius must BE 1"
    real = prep["source_face"] >= 0
    lens = np.linalg.norm(rows[real][:, 3:6], axis=1)
    np.testing.assert_allclose(lens, 1.0, atol=1e-5)
    print(f"PASS  (1, 6, 256); max radius {radii.max():.6f}; normals unit")


@needs_model
def test_a_SMALL_mesh_is_zero_padded_and_the_padding_is_dropped_again():
    """`ToothData` pads to num_points with zeros. Two things must hold.

    The padding rows must exist (they are part of the input distribution the
    checkpoint was trained on) and they must NOT come back as predictions
    attached to a real face. `source_face` carries -1 for them, because index
    -1 in numpy selects the LAST element - a padding row's class would land
    silently on the final triangle of the mesh.
    """
    a = ctb.CrossToothAdapter(num_points=512)
    m = _ring_mesh(n=32)                       # 64 faces, far under 512
    f = np.asarray(m.faces, np.int64)
    prep = a.prepare(np.asarray(m.vertices, float), f)

    assert prep["padded_rows"] == 512 - len(f)
    assert prep["sampled"] == len(f)
    assert int((prep["source_face"] < 0).sum()) == prep["padded_rows"]
    assert sorted(prep["source_face"][prep["source_face"] >= 0]) == \
        list(range(len(f)))
    print(f"PASS  {len(f)} faces padded to 512; "
          f"{prep['padded_rows']} rows marked -1")


def test_the_subset_does_NOT_take_the_first_16000_faces_in_file_order():
    """THE SHIPPED LOADER'S TRUNCATION, AND WHY IT IS REPLACED.

        permute = np.random.permutation(self.args.num_points)   # 0..15999
        pointcloud = pointcloud[permute]

    On a mesh with more faces than `num_points` that index array cannot reach
    past 15,999, so the model is shown a contiguous prefix of the file. On a
    scan written arch-end to arch-end that is a fraction of one side.

    The fixture is built so the defect is VISIBLE: faces are ordered along x,
    so a prefix covers a slice and a spatially uniform subset covers the lot.
    """
    n = 5000
    x = np.linspace(0.0, 100.0, n)
    pts = np.c_[x, np.zeros(n), np.zeros(n)]

    keep = ctb.spatially_uniform_subset(pts, 500, seed=1)
    assert len(keep) == 500
    span = float(pts[keep][:, 0].max() - pts[keep][:, 0].min())
    assert span > 95.0, f"the subset only spans {span:.1f} of 100"

    prefix_span = float(pts[:500][:, 0].max() - pts[:500][:, 0].min())
    assert prefix_span < 15.0, prefix_span
    print(f"PASS  uniform subset spans {span:.1f}; the old prefix spans "
          f"{prefix_span:.1f}")


def test_the_subset_is_deterministic_and_independent_of_input_order():
    """One scan, one answer - and shuffling the face array must not move it.

    Determinism matters because a clinician re-running segmentation on the
    same arch and getting different teeth has no way to tell a model
    disagreement from a sampling one.
    """
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(4000, 3)) * 12.0

    a = ctb.spatially_uniform_subset(pts, 400, seed=1)
    b = ctb.spatially_uniform_subset(pts, 400, seed=1)
    np.testing.assert_array_equal(a, b)

    perm = rng.permutation(len(pts))
    c = ctb.spatially_uniform_subset(pts[perm], 400, seed=1)
    # Same POINTS, whatever order they arrived in.
    assert set(map(tuple, np.round(pts[a], 9))) == \
        set(map(tuple, np.round(pts[perm][c], 9)))
    print(f"PASS  {len(a)} points, identical under a full shuffle")


def test_asking_for_more_points_than_there_are_returns_every_one():
    pts = np.random.default_rng(0).normal(size=(120, 3))
    keep = ctb.spatially_uniform_subset(pts, 500, seed=1)
    np.testing.assert_array_equal(keep, np.arange(120))
    print("PASS  k >= n returns all n")


# ===========================================================================
# 3. The Windows / CPU bypass
# ===========================================================================

def test_the_pointops_shim_is_removed_from_sys_modules_again():
    """`models` IS ToothGroupNetwork'S NAME TOO, and that is the hazard.

    tgn_bridge design rule 3 exists because the vendored network claims
    top-level names like `models`. This backend warms TGN on a background
    thread, so a synthetic `models` package left installed for the life of
    the process is how one model captures the other's imports.
    """
    before = {n: sys.modules.get(n) for n in ctb._POINTOPS_CHAIN}
    with ctb._pointops_installed() as shim:
        assert sys.modules[ctb._POINTOPS_CHAIN[-1]].pointops is shim
    after = {n: sys.modules.get(n) for n in ctb._POINTOPS_CHAIN}
    assert before == after, {k: (before[k], after[k]) for k in before
                             if before[k] is not after[k]}
    print("PASS  sys.modules restored exactly")


def test_a_pre_existing_module_on_the_chain_is_left_alone():
    """If somebody else owns `models`, it is not ours to replace or delete."""
    import types
    sentinel = types.ModuleType("models")
    sentinel.MINE = True
    sys.modules["models"] = sentinel
    try:
        with ctb._pointops_installed():
            assert sys.modules["models"] is sentinel
        assert sys.modules["models"] is sentinel
        assert sys.modules["models"].MINE is True
    finally:
        sys.modules.pop("models", None)
    print("PASS  a foreign `models` survives untouched")


def test_queryandgroup_returns_a_PAIR_and_the_tgn_shim_still_returns_ONE():
    """The two networks disagree about this function's return value.

    ToothGroupNetwork:  x = pointops.queryandgroup(...)
    CrossTooth:         x, idx = pointops.queryandgroup(...)

    CrossTooth reuses that `idx` so the key and value projections group over
    the SAME neighbourhood; recomputing it would be a different model. The
    adaptation must not reach `pointops_cpu`, which TGN depends on.
    """
    import torch

    import pointops_cpu

    n, c, k = 40, 5, 4
    xyz = torch.rand(n, 3)
    feat = torch.rand(n, c)
    off = torch.tensor([n], dtype=torch.int32)

    plain = pointops_cpu.queryandgroup(k, xyz, xyz, feat, None, off, off, True)
    assert isinstance(plain, torch.Tensor), type(plain)

    shim = ctb._CrossToothPointops(pointops_cpu)
    grouped, idx = shim.queryandgroup(nsample=k, xyz=xyz, new_xyz=xyz,
                                      feat=feat, idx=None, offset=off,
                                      new_offset=off, use_xyz=True)
    assert grouped.shape == plain.shape
    torch.testing.assert_close(grouped, plain)

    # Passing the index back must reuse it rather than re-query.
    again, idx2 = shim.queryandgroup(nsample=k, xyz=xyz, new_xyz=xyz,
                                     feat=feat, idx=idx, offset=off,
                                     new_offset=off, use_xyz=False)
    torch.testing.assert_close(idx.float(), idx2.float())
    assert again.shape == (n, k, c)
    print("PASS  pair for CrossTooth, tensor for TGN, index reused")


def test_the_cuda_int_tensor_patch_is_reverted_even_when_the_body_raises():
    """It is process-global while it is in force, so it must always come off.

    `TransitionDown.forward` calls `torch.cuda.IntTensor` on a list of ints,
    which on a CPU build raises `TypeError: type torch.cuda.IntTensor not
    available`. Stride is 4 on four of the five encoder stages, so it fires
    on every forward pass.
    """
    import torch

    original = torch.cuda.IntTensor
    with ctb._cuda_int_tensor_on("cpu"):
        t = torch.cuda.IntTensor([3, 6, 9])
        assert t.dtype == torch.int32 and t.device.type == "cpu"
        assert t.tolist() == [3, 6, 9]
    assert torch.cuda.IntTensor is original

    with pytest.raises(ValueError):
        with ctb._cuda_int_tensor_on("cpu"):
            raise ValueError("boom")
    assert torch.cuda.IntTensor is original
    print("PASS  torch.cuda.IntTensor restored, including on a raise")


# ===========================================================================
# 4. End to end
# ===========================================================================

@needs_model
def test_the_checkpoint_loads_STRICT():
    """A partial load is a model that runs and predicts noise.

    `strict=True` is what turns an architecture that has drifted from the
    weights into a failure at load rather than a plausible-looking answer.
    """
    a = ctb.CrossToothAdapter().load()
    assert a.model is not None
    assert a.model.training is False, "must be in eval mode: BatchNorm"
    head = a.model.cls[-1]
    assert tuple(head.weight.shape) == (ctb.NUM_CLASSES, 32)
    first = a.model.enc1[0].linear
    assert first.weight.shape[1] == ctb.IN_CHANNELS
    print(f"PASS  loaded in {a.load_seconds}s, cls head "
          f"{tuple(head.weight.shape)}")


@needs_model
def test_a_sphere_is_all_gingiva_and_never_a_mouthful_of_teeth():
    """The cheapest honest end-to-end check available without ground truth.

    A ball is not an arch. A model that answers "16 teeth" here is not
    segmenting, it is pattern-matching the output space - and this is the one
    negative control that needs no annotation at all.
    """
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
    a = ctb.CrossToothAdapter(num_points=1024).load()
    out = a.segment(np.asarray(m.vertices, float),
                    np.asarray(m.faces, np.int64), "lower")
    teeth = sorted({int(x) for x in out["labels"]} - {0})
    assert len(out["labels"]) == len(m.vertices)
    assert len(teeth) <= 1, f"a sphere produced {teeth}"
    print(f"PASS  sphere -> {len(teeth)} tooth label(s)")


@needs_model
def test_every_label_is_a_valid_FDI_for_the_jaw_that_was_asked_for():
    import trimesh

    import jaw_naming

    m = trimesh.creation.icosphere(subdivisions=3, radius=18.0)
    v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    a = ctb.CrossToothAdapter(num_points=512).load()
    for jaw, valid in (("lower", jaw_naming.LOWER_FDI),
                       ("upper", jaw_naming.UPPER_FDI)):
        out = a.segment(v, f, jaw)
        got = {int(x) for x in out["labels"]} - {0}
        assert got <= valid, (jaw, sorted(got - valid))
    print("PASS  labels stay inside the requested jaw's FDI set")


@needs_model
def test_an_unknown_jaw_is_refused_rather_than_guessed():
    a = ctb.CrossToothAdapter(num_points=256)
    with pytest.raises(ValueError, match="jaw"):
        a.segment(np.zeros((3, 3)), np.array([[0, 1, 2]]), "left")
    print("PASS  jaw='left' is refused")


# ===========================================================================
# 5. Quadrant coherence - what it can and cannot say
# ===========================================================================

def test_coherence_reports_a_clean_two_sided_arch_as_separated_and_ordered():
    """A horseshoe: class 1 and 9 at the midline, 8 and 16 at the back.

    THE MIRROR IN X IS THE POINT OF THE FIXTURE. The first draft put both
    quadrants on the same side by multiplying the wrong term, which made the
    two groups the same points - and `separated` still read True. A fixture
    that does not reproduce the arrangement being tested proves nothing, so
    the two ends are asserted to be on opposite sides of the midline here
    before anything else is checked.
    """
    cent, cls = [], []
    for side, base in ((-1, 0), (+1, 8)):
        for k in range(8):
            t = (k + 1) * 0.18
            c = np.array([side * 25 * np.sin(t), 25 * np.cos(t), 0.0])
            cent += list(c + np.random.default_rng(k).normal(size=(6, 3)) * .2)
            cls += [base + k + 1] * 6
    cent, cls = np.array(cent), np.array(cls)
    assert cent[cls <= 8][:, 0].max() < 0 < cent[cls >= 9][:, 0].min()

    out = ctb.quadrant_coherence(cent, cls)
    assert out["measurable"] and out["separated"] and out["ordered"]
    assert out["group_overlap_mm"] == 0.0
    assert out["quadrant_names_verified"] is False
    print(f"PASS  separation {out['group_separation_mm']:.2f} mm, rank "
          f"{out['incisor_to_molar_rank_correlation']:.3f}")


def test_coherence_NEVER_claims_the_quadrant_NAMES_are_verified():
    """A coherent split does not say which group is the patient's left.

    That is a property of the training data's orientation convention. If this
    flag could ever read True, a downstream reader would take a geometric
    measurement as a clinical one - which is the CLAUDE.md s.16 hazard, and
    the reason it is asserted rather than trusted.
    """
    cent = np.array([[-10., 0, 0]] * 8 + [[10., 0, 0]] * 8)
    cls = np.array(list(range(1, 17)))
    out = ctb.quadrant_coherence(cent, cls)
    assert out["quadrant_names_verified"] is False
    assert "left" in out["quadrant_names_note"]

    src = open(os.path.join(HERE, "crosstooth_bridge.py"),
               encoding="utf-8").read()
    assert '"quadrant_names_verified"] = False' in src or \
           'out["quadrant_names_verified"] = False' in src
    print("PASS  the names are never reported as verified")


def test_coherence_says_UNMEASURABLE_rather_than_passing_on_no_teeth():
    """NOT_CHECKED is not CLEAR - CLAUDE.md s.14."""
    out = ctb.quadrant_coherence(np.zeros((5, 3)), np.zeros(5, int))
    assert out["measurable"] is False
    assert "cells" in out["reason"]
    assert "separated" not in out
    print(f"PASS  unmeasurable: {out['reason']}")


# ===========================================================================
# 6. The provider registry and the toggle
# ===========================================================================

def test_crosstooth_is_registered_and_is_NOW_the_default():
    """CHANGED, and on a measurement rather than a preference.

    This test previously asserted the opposite, and its reason was sound at
    the time: CrossTooth won every geometric measure and was 19x faster, and
    that is not grounds to pick the model that decides where a clinician cuts
    (s.25.9). What changed is that a NEW measurement exists which is about
    the thing the app actually needs from a segmentation - the tooth/gum
    boundary - and which needs no annotation to take.

    Scored on `case_lower.stl` through the whole path, model then hybrid then
    cleanup, on the occlusal band (>= 85% tooth) and the gingival band
    (>= 90% gum):

        ToothGroupNetwork   84.58 / 100.0   FAILS the occlusal band
        CrossTooth          93.66 / 100.0   passes both

    Both models get the boundary itself right; TGN loses because it MERGES
    TEETH on this scan - its FDI 37 is two connected regions of 6,822 and
    5,134 faces, its FDI 44 is 12,642 and 11,247 - so keeping one region per
    tooth must discard 16,381 faces of real enamel.

    STILL NOT AN ACCURACY CLAIM. No model here has been scored against an
    independent annotation and none can be (s.17, s.25.6). The FDI NUMBERING
    remains unverified for both; this is the boundary and the region
    structure, both checkable without ground truth.
    """
    reg = sp.registry()
    assert "crosstooth" in reg
    assert sp.DEFAULT_PROVIDER == "crosstooth"
    assert [k for k, v in reg.items() if v["default"]] == ["crosstooth"]
    assert sp.get("crosstooth") is sp.get("CrossTooth")
    # The shipped model must stay REACHABLE, not merely present: it is the
    # fallback if the licence question (s.26.13) forces CrossTooth out.
    assert sp.get("toothgroupnetwork").name == "toothgroupnetwork"
    print(f"PASS  registry {sorted(reg)}, default {sp.DEFAULT_PROVIDER}")


def test_the_default_switch_does_not_resolve_the_licence():
    """s.26.13: `CrossTooth/` carries no LICENSE, no COPYING and no `.git`.

    Making it the default is a TECHNICAL decision taken on measurements. It
    does not grant any rights, and this test exists so that nobody reads the
    switch as having settled the question.
    """
    import os

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CrossTooth")
    if not os.path.isdir(root):
        import pytest
        pytest.skip("CrossTooth/ is not in this checkout")
    names = {n.lower() for n in os.listdir(root)}
    assert not {n for n in names if n.startswith(("license", "licence", "copying"))}, (
        "a licence file has appeared - re-read s.26.13, the commercial "
        "blocker may be resolved and this test should be updated deliberately")
    src = sp.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    assert "COMMERCIAL BLOCKER" in text, \
        "the licence caveat was removed from the provider registry"
    print("PASS  default switched; licence still unresolved and still stated")


def test_the_crosstooth_audit_declares_the_quadrant_naming_UNVERIFIED():
    audit = sp.CrossToothProvider.audit
    assert audit["quadrant_naming_verified"] is False
    assert "agreement" in audit["quadrant_naming_evidence"]
    assert "not accuracy" in audit["quadrant_naming_evidence"]
    # The two departures from upstream must be stated, not buried.
    assert "selective_downsample" in audit["resampling"]
    assert "pointops" in audit["windows_bypass"]
    print("PASS  the audit names its own departures and its own gaps")


def test_an_unknown_provider_is_refused_by_name_and_never_falls_back():
    """A typo must not silently segment with the other model."""
    with pytest.raises(KeyError, match="crosstoth"):
        sp.get("crosstoth")
    print("PASS  'crosstoth' raises rather than resolving")


def test_the_environment_default_ignores_a_value_it_does_not_recognise():
    """An env-var typo must not change which model segments a patient's arch.

    Read back through the module's own resolver rather than by re-importing,
    so the test cannot pass by exercising a different code path from the one
    that runs at startup.
    """
    old = os.environ.get("ALIGNER_SEGMENTATION_PROVIDER")
    try:
        os.environ["ALIGNER_SEGMENTATION_PROVIDER"] = "crosstooth"
        assert sp._default_from_environment() == "crosstooth"
        os.environ["ALIGNER_SEGMENTATION_PROVIDER"] = "toothgroupnetwork"
        assert sp._default_from_environment() == "toothgroupnetwork"
        # A typo falls back to the module's own default, whatever that is -
        # the property under test is that it is IGNORED, not which model wins.
        os.environ["ALIGNER_SEGMENTATION_PROVIDER"] = "not-a-model"
        assert sp._default_from_environment() == sp.DEFAULT_PROVIDER_NAME
        os.environ["ALIGNER_SEGMENTATION_PROVIDER"] = ""
        assert sp._default_from_environment() == sp.DEFAULT_PROVIDER_NAME
    finally:
        os.environ.pop("ALIGNER_SEGMENTATION_PROVIDER", None)
        if old is not None:
            os.environ["ALIGNER_SEGMENTATION_PROVIDER"] = old
    print("PASS  a bad env value is ignored, not obeyed")


def test_the_meshsegnet_candidate_still_raises_rather_than_stubbing():
    """Adding a real second provider must not soften the declared one."""
    with pytest.raises(NotImplementedError):
        sp.get("meshsegnet").segment(np.zeros((3, 3)),
                                     np.array([[0, 1, 2]]), "lower")
    print("PASS  the candidate still raises")


@needs_model
def test_load_REBUILDS_when_the_configuration_changes_instead_of_ignoring_it():
    """A cached singleton must not answer a question it was not asked.

    `load()` used to return the existing adapter whenever one existed, so
    `load(num_points=512)` then `load()` handed a 512-point adapter to a
    caller asking for the trained 16,000 - reporting `loaded: True` while
    doing it. The full suite found it: the real-scan test passed alone and
    failed after an earlier test had loaded a small adapter, and it failed
    by claiming the labels belonged to a different mesh. The failure mode is
    the one this repository keeps meeting - it ANSWERS rather than raising.
    """
    import crosstooth_bridge as b

    b.unload()
    b.load(num_points=512)
    assert b.adapter().num_points == 512
    b.load()
    assert b.adapter().num_points == b.NUM_POINTS, \
        "load() served a stale configuration"
    b.load(num_points=512)
    assert b.adapter().num_points == 512
    # Same configuration twice must NOT rebuild: the lock and the cache still
    # have to work, or every /segment pays a model load.
    first = b.adapter()
    b.load(num_points=512)
    assert b.adapter() is first
    b.unload()
    print(f"PASS  512 -> {b.NUM_POINTS} -> 512 rebuilds; a repeat does not")


@needs_model
def test_the_provider_returns_labels_on_the_callers_array_by_COORDINATE():
    """No vertex ordering is involved anywhere in this provider.

    That is the claim `transfer` makes, and it is the reason the position
    transfer is not run here. Shuffling the vertex array must therefore move
    the labels with it exactly - an index-keyed provider would not.
    """
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=3, radius=18.0)
    v, f = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)

    import crosstooth_bridge
    crosstooth_bridge.load(num_points=512)
    a = crosstooth_bridge.adapter()
    base = a.segment(v, f, "lower")["labels"]

    perm = np.random.default_rng(3).permutation(len(v))
    inv = np.argsort(perm)
    moved = a.segment(v[perm], inv[f], "lower")["labels"]
    np.testing.assert_array_equal(moved, base[perm])

    res = sp.get("crosstooth").segment(v, f, "lower")
    assert res.transfer["not_applicable"] is True
    assert "COORDINATE" in res.transfer["reason"]
    assert len(res.labels) == len(v)
    print("PASS  labels follow the vertices under a full shuffle")


# ===========================================================================
# 7. The endpoint routing, without running a model
# ===========================================================================

def _lower_arch_session():
    """A real session over a synthetic arch, by the route uploads take."""
    import asyncio
    import io
    import struct

    from fastapi import UploadFile

    import api_core
    from tooth_fixture import tooth_on_base

    verts, faces, _ = tooth_on_base()
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    buf.write(struct.pack("<I", len(faces)))
    for tri in faces:
        buf.write(struct.pack("<3f", 0, 0, 0))
        for vi in tri:
            buf.write(struct.pack("<3f", *verts[vi]))
        buf.write(struct.pack("<H", 0))
    up = UploadFile(filename="scan.stl", file=io.BytesIO(buf.getvalue()))
    res = asyncio.run(api_core.create_session(arch="lower", file=up))
    return res["session_id"]


def test_the_endpoint_refuses_an_unknown_provider_with_400_not_a_fallback():
    """A typo in `?provider=` must not quietly run the other model.

    400 rather than 404 deliberately: the SESSION exists, the request does
    not name a thing that does. A 404 here would send someone looking at the
    session store.
    """
    import asyncio

    from fastapi import HTTPException

    import api_core

    sid = _lower_arch_session()
    with pytest.raises(HTTPException) as e:
        asyncio.run(api_core.segment(sid, provider="crosstoth"))
    assert e.value.status_code == 400
    assert "crosstoth" in str(e.value.detail)
    print(f"PASS  400: {e.value.detail}")


def test_the_endpoint_refuses_an_UNAVAILABLE_provider_with_409_and_a_reason():
    """`meshsegnet` is declared, not installed. Naming it must say so.

    409 because the request is well formed and the server cannot honour it
    right now - which is the same code a duplicate run already returns, and
    the reason travels with it rather than the client having to guess.
    """
    import asyncio

    from fastapi import HTTPException

    import api_core

    sid = _lower_arch_session()
    with pytest.raises(HTTPException) as e:
        asyncio.run(api_core.segment(sid, provider="meshsegnet"))
    assert e.value.status_code == 409
    assert "meshsegnet" in str(e.value.detail)
    assert "not been installed" in str(e.value.detail)
    print(f"PASS  409: {str(e.value.detail)[:70]}...")


def test_a_MISSING_session_still_404s_before_the_provider_is_consulted():
    """Session first. A dead session must not report a model problem."""
    import asyncio

    from fastapi import HTTPException

    import api_core

    with pytest.raises(HTTPException) as e:
        asyncio.run(api_core.segment("no-such-session", provider="crosstooth"))
    assert e.value.status_code == 404
    print("PASS  404 for an expired session, whatever the provider")


def test_the_ai_status_payload_carries_the_registry_for_the_picker():
    """The client's picker reads this. `loaded` still means TGN, as always."""
    import api_core

    st = api_core.ai_status()
    assert st["default_provider"] == sp.DEFAULT_PROVIDER
    assert set(st["providers"]) == set(sp.registry())
    assert st["providers"]["crosstooth"]["available"] is True
    assert st["providers"]["meshsegnet"]["available"] is False
    assert "segmentation_provider" in st
    print(f"PASS  /api/ai/status lists {sorted(st['providers'])}")


# ===========================================================================
# 8. The real scan. Skipped, never faked.
# ===========================================================================

_STL = os.path.join(HERE, "case_lower.stl")


@needs_model
@pytest.mark.skipif(not os.path.exists(_STL), reason="the real scan is absent")
@pytest.mark.slow
def test_the_real_scan_produces_teeth_that_are_geometrically_plausible():
    """GEOMETRY, which needs no annotation. Not accuracy, which does.

    The bar is the one s.24.3 established: a label array indexed to the wrong
    mesh scatters every tooth across the whole arch, and the MEDIAN per-tooth
    box diagonal is what separates that from a model that segmented badly.
    """
    import core_geometry as cg
    import segmentation_diagnostics as sd
    import stl_io

    v0, f0 = stl_io.parse_stl_bytes(open(_STL, "rb").read())
    v, f, _ = cg.condition_mesh(v0, f0)

    out = sp.get("crosstooth").segment(v, f, "lower")
    rep = sd.label_report(v, f, out.labels)
    assert rep["indexed_to_this_mesh"] is True, rep.get("reason")

    rows = rep["teeth"]
    assert len(rows) >= 12, f"only {len(rows)} teeth"
    diags = [r["bbox_diagonal_mm"] for r in rows]
    assert float(np.median(diags)) < sd.MAX_PLAUSIBLE_TOOTH_DIAGONAL_MM

    whole = sum(1 for r in rows if r["largest_component_fraction"] >= 0.95)
    assert whole >= len(rows) - 2, \
        [(r["label"], r["largest_component_fraction"]) for r in rows]

    co = out.meta["quadrant_coherence"]
    assert co["measurable"] and co["separated"], co
    assert out.meta["labels_are_fdi_verified"] is False
    print(f"PASS  {len(rows)} teeth, median diagonal "
          f"{np.median(diags):.3f} mm, {whole} at >=95% one piece, "
          f"quadrants separated by {co['group_separation_mm']:.2f} mm")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
