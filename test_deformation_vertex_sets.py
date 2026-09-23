"""The tooth vertex sets come from the ORIGINAL scan ids, and stay there.

AGENT_BRIEF 2.3. Three claims, each with a control that reproduces the
failure first (CLAUDE.md s.5 - a regression test whose fixture cannot produce
the bug proves nothing):

  A1.2   a scan vertex id IS a cast vertex id. `trim_to_arch` appends and
         `build_cast_base` keeps its input as a prefix, so the sets need no
         remap - and CLAUDE.md s.24.3 is what a remap costs when it is wrong
         while every count check still passes.
  ALL 3  a face belongs to a tooth when all three of its vertices do. The
         control shows the `any` rule overlapping two crowns that the `all`
         rule separates.
  SHARED a vertex claimed by a moving tooth and a static one is MOVING, and
         the count is reported - by this module and, independently, by
         `dc.plan_deformation`.
"""
from __future__ import annotations

import numpy as np
import pytest

import core_geometry as cg
import deform_construction as dc
import manufacturing_v2 as v2
from test_cast_base import frame_for, horseshoe_shell

N_S, N_T = 240, 60
TEETH = (-0.25, 0.0, 0.25)


@pytest.fixture(scope="module")
def case():
    """The kit's own fixture, plus per-vertex FDI labels over the crowns."""
    v, f = horseshoe_shell(n_s=N_S, n_t=N_T, teeth=TEETH, tooth_h=4.0)
    af = frame_for(v)
    tv, tf, tinfo = cg.trim_to_arch(v, f, af)
    bv, bf, _ = cg.build_cast_base(tv, tf, af, rim=tinfo["rim_loop"])

    i_s, j = np.divmod(np.arange(N_S * N_T), N_T)
    u = np.linspace(-1, 1, N_S)[i_s]
    w = (np.linspace(-1.95, 1.95, N_T) / 1.95)[j]

    labels = np.zeros(len(v), np.int64)          # 0 = gingiva
    for fdi, pos in zip((44, 45, 46), TEETH):
        inside = np.sqrt(((u - pos) / 0.13) ** 2 + (w / 0.30) ** 2) < 0.8
        labels[: N_S * N_T][inside] = fdi
    return {"v": v, "f": f, "af": af, "tv": tv, "tf": tf, "tinfo": tinfo,
            "bv": bv, "bf": bf, "labels": labels}


@pytest.fixture(scope="module")
def contacting():
    """The kit's CONTACTING geometry: crowns merged across a broad contact.

    The SPACED fixture cannot show the `any`-rule defect at all - its crowns
    are separated by gingiva, so no face has vertices on two teeth. That is
    the whole reason this second fixture exists, and asserting the straddle
    count is non-zero is what keeps it honest.
    """
    pos = (-0.14, 0.0, 0.14)
    v, f = horseshoe_shell(n_s=N_S, n_t=N_T, teeth=pos, tooth_h=4.0)
    i_s, j = np.divmod(np.arange(N_S * N_T), N_T)
    u = np.linspace(-1, 1, N_S)[i_s]
    w = (np.linspace(-1.95, 1.95, N_T) / 1.95)[j]
    labels = np.zeros(len(v), np.int64)
    for fdi, p in zip((44, 45, 46), pos):
        inside = np.sqrt(((u - p) / 0.13) ** 2 + (w / 0.30) ** 2) < 0.8
        labels[: N_S * N_T][inside] = fdi
    return {"f": f, "labels": labels}


# ---------------------------------------------------------------------------
# A1.2 - the id a set is written in is the id the cast reads it in
# ---------------------------------------------------------------------------

def test_a_scan_vertex_id_is_a_cast_vertex_id(case):
    """Measured on the real functions, not asserted from the docstring."""
    v, bv = case["v"], case["bv"]
    assert len(bv) >= len(v)
    assert np.array_equal(bv[: len(v)], v), \
        "build_cast_base no longer keeps the scan array as a prefix"
    # bit-identical, not allclose: a set is an integer index, and a coordinate
    # that shifted by an ULP would mean the prefix was rebuilt rather than kept.
    assert (bv[: len(v)] == v).all()
    print(f"PASS  scan {len(v):,} verts is the prefix of cast {len(bv):,}; "
          f"{len(bv) - len(v):,} appended")


def test_the_appended_vertices_are_the_ones_that_must_be_pinned(case):
    """`i >= len(trimmed)` identifies the floor exactly - the rule
    `build_case_plan` pins on. If the prefix property broke, this pins the
    wrong vertices and the cast would deform at its own boundary."""
    tv, bv, bf = case["tv"], case["bv"], case["bf"]
    appended = np.arange(len(tv), len(bv))
    assert len(appended) > 0
    # Every appended vertex is referenced by the cast, i.e. none is a phantom.
    assert np.isin(appended, np.unique(bf)).all()
    print(f"PASS  {len(appended):,} appended vertices, all referenced")


# ---------------------------------------------------------------------------
# ALL THREE, and the control that shows why
# ---------------------------------------------------------------------------

def test_the_any_vertex_rule_overlaps_two_crowns_and_the_all_rule_does_not(contacting):
    """THE CONTROL. `any` is the tempting reading and it is wrong.

    A face with one vertex on each of two teeth lands in BOTH tooth's face
    sets under `any`, so both sets claim its vertices and the kit refuses to
    plan. Under `all` the face belongs to neither and its vertices are free to
    blend, which is what the envelope is for.
    """
    f, lab = contacting["f"], contacting["labels"]

    def verts_under(rule, fdi):
        owned = (lab[f] == fdi)
        m = owned.any(axis=1) if rule == "any" else owned.all(axis=1)
        return np.unique(f[m])

    any_44, any_45 = verts_under("any", 44), verts_under("any", 45)
    all_44, all_45 = verts_under("all", 44), verts_under("all", 45)

    n_any = len(np.intersect1d(any_44, any_45))
    n_all = len(np.intersect1d(all_44, all_45))
    # The fixture must be able to produce the defect, or this proves nothing.
    assert n_any > 0, "fixture has no straddling faces - it cannot show the bug"
    assert n_all == 0, f"the all-three rule still overlaps at {n_all} vertices"
    print(f"PASS  `any` overlaps at {n_any} vertices, `all` at {n_all}")


def test_a_seam_vertex_is_in_no_rigid_set(case):
    """A vertex whose every incident face straddles must be free to blend."""
    f, lab = case["f"], case["labels"]
    m44 = v2.tooth_faces_from_labels(f, lab, 44)
    sets = v2.tooth_vertex_sets(f, {44: m44}, {}, n_vertices=len(lab))
    moving = sets["moving"][44]

    # A gingival vertex adjacent to the crown: labelled 0, touching label-44
    # faces. It must not be in the rigid set.
    touching = np.unique(f[(lab[f] == 44).any(axis=1)])
    seam = touching[lab[touching] == 0]
    assert len(seam) > 0, "fixture has no seam vertices"
    assert not np.isin(seam, moving).any(), \
        "a gingival seam vertex was put in the rigid set"
    print(f"PASS  {len(seam)} seam vertices, none in the rigid set")


def test_labels_shorter_than_the_mesh_are_refused(case):
    with pytest.raises(ValueError, match="labels"):
        v2.tooth_faces_from_labels(case["f"], case["labels"][:-5], 44)
    print("PASS  a short label array is refused, not silently indexed")


# ---------------------------------------------------------------------------
# The sets themselves
# ---------------------------------------------------------------------------

def test_moving_is_the_vertices_of_the_moving_tooth_and_static_is_the_rest(case):
    f, lab = case["f"], case["labels"]
    n = len(lab)
    out = v2.tooth_vertex_sets(
        f, {45: v2.tooth_faces_from_labels(f, lab, 45)},
        {44: v2.tooth_faces_from_labels(f, lab, 44),
         46: v2.tooth_faces_from_labels(f, lab, 46)}, n_vertices=n)

    moving, static, d = out["moving"][45], out["static"], out["diagnostics"]
    assert len(moving) > 0 and len(static) > 0
    assert not np.intersect1d(moving, static).size, "the sets overlap"
    assert (lab[moving] == 45).all(), "a non-45 vertex is in the moving set"
    assert (np.isin(lab[static], (44, 46))).all(), "gingiva leaked into static"
    assert d["source"] == "original scan face ids"
    assert d["static_teeth"] == ["44", "46"]
    print(f"PASS  moving {len(moving)}, static {len(static)}, "
          f"shared {d['shared_contact_vertices_assigned_to_moving']}")


def test_a_shared_contact_vertex_goes_to_moving_and_is_counted(case):
    """Forced: hand the SAME faces to both sides. Every vertex is shared."""
    f, lab = case["f"], case["labels"]
    m = v2.tooth_faces_from_labels(f, lab, 45)
    out = v2.tooth_vertex_sets(f, {45: m}, {45: m, 44: v2.tooth_faces_from_labels(
        f, lab, 44)}, n_vertices=len(lab))

    n_45 = len(np.unique(f[m]))
    assert len(out["moving"][45]) == n_45, "the moving set lost vertices"
    assert out["diagnostics"]["shared_contact_vertices_assigned_to_moving"] == n_45
    assert not np.isin(out["moving"][45], out["static"]).any()
    print(f"PASS  all {n_45} shared vertices went to moving, and were counted")


def test_two_moving_teeth_are_made_disjoint_deterministically(case):
    """The kit REFUSES overlapping moving sets rather than choosing. So the
    choice is made here, by lowest id, and must not depend on dict order."""
    f, lab = case["f"], case["labels"]
    m44 = v2.tooth_faces_from_labels(f, lab, 44)
    m45 = v2.tooth_faces_from_labels(f, lab, 45)
    overlap = np.unique(f[m44])[:20]
    m44b = m44 | np.isin(np.arange(len(f)),
                         np.flatnonzero((np.isin(f, overlap)).all(axis=1)))

    a = v2.tooth_vertex_sets(f, {44: m44b, 45: m45 | m44b}, {}, n_vertices=len(lab))
    b = v2.tooth_vertex_sets(f, {45: m45 | m44b, 44: m44b}, {}, n_vertices=len(lab))

    for out in (a, b):
        assert not np.intersect1d(out["moving"][44], out["moving"][45]).size
    assert np.array_equal(a["moving"][44], b["moving"][44]), \
        "the answer depends on dict insertion order"
    assert np.array_equal(a["moving"][45], b["moving"][45])
    assert a["diagnostics"]["moving_moving_overlaps_resolved"] == \
        b["diagnostics"]["moving_moving_overlaps_resolved"]
    assert a["diagnostics"]["moving_moving_overlaps_resolved"], \
        "the fixture produced no overlap - it cannot show the resolution"
    print(f"PASS  order-independent; resolved "
          f"{a['diagnostics']['moving_moving_overlaps_resolved']}")


def test_the_sets_plan_without_the_kit_refusing_them(case):
    """End of the chain: what 2.3 produces is what 2.2 consumes.

    `plan_deformation` raises on overlapping moving sets and on a moving tooth
    that reaches the pinned rim. Passing is the assertion.
    """
    f, lab, bv, bf = case["f"], case["labels"], case["bv"], case["bf"]
    out = v2.tooth_vertex_sets(
        f, {45: v2.tooth_faces_from_labels(f, lab, 45)},
        {44: v2.tooth_faces_from_labels(f, lab, 44),
         46: v2.tooth_faces_from_labels(f, lab, 46)}, n_vertices=len(bv))

    pinned = np.zeros(len(bv), bool)
    pinned[np.asarray(case["tinfo"]["rim_loop"])] = True
    pinned[len(case["tv"]):] = True

    plan = dc.plan_deformation(bv, bf, out["moving"], out["static"], pinned,
                               envelope_mm=5.0)
    assert len(plan.free_idx) > 0, "nothing blends - the envelope found nothing"
    # The kit counts the moving/static overlap independently. Ours is measured
    # before it yields, so with a disjoint input both must read zero.
    assert plan.diagnostics["shared_contact_vertices_assigned_to_moving"] == 0
    assert out["diagnostics"]["shared_contact_vertices_assigned_to_moving"] == 0
    print(f"PASS  planned: {len(plan.free_idx)} free vertices, "
          f"{len(plan.tooth_sets[45])} rigid")


def test_a_face_mask_of_the_wrong_length_is_refused(case):
    with pytest.raises(ValueError, match="face mask"):
        v2.tooth_vertex_sets(case["f"], {45: np.zeros(7, bool)}, {})
    print("PASS  a wrong-length face mask is refused")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
