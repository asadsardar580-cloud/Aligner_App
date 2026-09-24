"""The gum is labelled gum, and the enamel is labelled tooth.

TASK 1 step 4, on the real scan. Two bands, stated apical of the cusp tips:

    12-18 mm   well below any cervical margin on a mandible   >= 90% GUM
     0- 4 mm   the occlusal third of the crowns               >= 85% TOOTH

They are deliberately NOT adjacent: the cervical margin lives in the gap
between them and is not scored, so a margin drawn a fraction high or low does
not move either number.

WHY THIS TEST EXISTS AT ALL. Every metric the repo had computed on the labels
alone - IoU included - is blind to this failure, because a label array that
is confidently wrong about which lumps are teeth scores perfectly against
itself. s.24.3 records the same blindness for a mis-indexed array. What
catches it is a GEOMETRIC question with an answer that needs no annotation:
is the thing labelled gum actually down in the gingiva?

THE BAND'S SIGN IS THE WHOLE DIFFICULTY and the first version of this
measurement had it backwards - see `segmentation_diagnostics.occlusal_axis`.
`test_the_band_is_not_measured_upside_down` is the control that would catch
it happening again.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import core_geometry as cg
import segmentation_cleanup as sc
import segmentation_diagnostics as sd
import stl_io

SCAN = "case_lower.stl"
pytestmark = pytest.mark.skipif(not os.path.exists(SCAN),
                                reason=f"{SCAN} is not in this checkout")


@pytest.fixture(scope="module")
def mesh():
    raw = open(SCAN, "rb").read()
    v, f = stl_io.parse_stl_bytes(raw)
    v, f, _ = cg.sanitize_scan(v, f)
    v, f, _ = cg.condition_mesh(v, f)
    return v, f


@pytest.fixture(scope="module")
def labels(mesh):
    """The default provider's labels, cleaned - i.e. what the session holds.

    The hybrid step is not run here: it needs the barrier graph, it is
    covered by `test_benchmark_segmentation.py`, and after the step-2 fix it
    is a no-op on this scan (it refuses both candidates by name and returns
    tier 1's array bit-identical). Skipping it keeps this test about the
    BOUNDARY rather than about the repair.
    """
    import segmentation_providers as sp

    v, f = mesh
    prov = sp.get()
    avail = prov.available()
    if not avail.get("available"):
        pytest.skip(f"{prov.name} unavailable: {avail.get('reason')}")
    res = prov.segment(v, f, "lower")
    lab = np.asarray(res.labels).astype(np.int64).reshape(-1)
    cleaned, report = sc.clean_labels(v, f, lab)
    return lab, cleaned, report, prov.name


# ---------------------------------------------------------------------------
# The two bands
# ---------------------------------------------------------------------------

def test_the_gingival_band_is_gum(mesh, labels):
    v, f = mesh
    _, cleaned, _, name = labels
    r = sd.band_report(v, f, cleaned)
    assert r["measured"], r
    gin = r["gingival"]
    assert gin["n"] > 0, "the gingival band is empty - nothing was measured"
    assert gin["gum_share"] is not None
    assert gin["gum_share"] >= sd.MIN_GUM_SHARE_GINGIVAL, (
        f"{name}: only {gin['gum_share'] * 100:.2f}% of the "
        f"{sd.GINGIVAL_BAND_MM[0]:.0f}-{sd.GINGIVAL_BAND_MM[1]:.0f}mm band is "
        f"gum, under the {sd.MIN_GUM_SHARE_GINGIVAL * 100:.0f}% bar "
        f"({gin['n']:,} vertices)")
    print(f"PASS  {name}: gingival band {gin['gum_share'] * 100:.2f}% gum "
          f"({gin['n']:,} verts)")


def test_the_occlusal_band_is_tooth(mesh, labels):
    v, f = mesh
    _, cleaned, _, name = labels
    r = sd.band_report(v, f, cleaned)
    occ = r["occlusal"]
    assert occ["n"] > 0, "the occlusal band is empty - nothing was measured"
    assert occ["tooth_share"] is not None
    assert occ["tooth_share"] >= sd.MIN_TOOTH_SHARE_OCCLUSAL, (
        f"{name}: only {occ['tooth_share'] * 100:.2f}% of the "
        f"{sd.OCCLUSAL_BAND_MM[0]:.0f}-{sd.OCCLUSAL_BAND_MM[1]:.0f}mm band is "
        f"tooth, under the {sd.MIN_TOOTH_SHARE_OCCLUSAL * 100:.0f}% bar "
        f"({occ['n']:,} vertices)")
    print(f"PASS  {name}: occlusal band {occ['tooth_share'] * 100:.2f}% tooth "
          f"({occ['n']:,} verts)")


# ---------------------------------------------------------------------------
# The controls - without these the two above can pass for the wrong reason
# ---------------------------------------------------------------------------

def test_the_band_is_not_measured_upside_down(mesh):
    """THE CONTROL FOR THE RULER, and it is the one that was needed.

    The first version of this measurement took the smallest-variance PCA axis
    and slabbed its "top 10%". `np.linalg.svd` promises no sign, so that
    slab was the floor of the mouth and every depth came out inverted - the
    census then reported 100% gingiva in the OCCLUSAL band, which reads as a
    broken model rather than a broken ruler.

    Asserted here as a fact about the SHAPE, with no labels involved: a thin
    slab through the crowns cuts into many separate lumps, a thin slab
    through the gingiva cuts one continuous band, and the axis must point at
    the many-lumps end.
    """
    v, f = mesh
    axis, ev = sd.occlusal_axis(v, f)
    assert ev["decisive"], (
        f"the two ends are too alike to call: "
        f"{ev['max_clusters_plus_end']} vs {ev['max_clusters_minus_end']} "
        f"clusters. Refusing is correct - do not guess a sign.")

    # Independent of the cluster rule: the labelled crowns must come out
    # SHALLOW. Uses the mesh's own curvature, not a segmentation.
    depth = sd.depth_below_cusp_tips(v, axis)
    assert depth.min() == 0.0
    assert depth.max() > 10.0, depth.max()

    # And the flipped axis must fail the very test the true one passes, or
    # the test cannot tell the two apart.
    conc = cg.boundary_field(v, f)
    hi = np.percentile(conc, 98)
    cuspy = conc >= hi
    d_true = depth[cuspy].mean()
    d_flip = sd.depth_below_cusp_tips(v, -axis)[cuspy].mean()
    assert d_true < d_flip, (
        f"high-curvature vertices sit {d_true:.2f}mm deep on the chosen axis "
        f"and {d_flip:.2f}mm on its opposite - the sign is not distinguishable")
    print(f"PASS  axis {np.round(axis, 4)}, clusters "
          f"{ev['max_clusters_plus_end']} vs {ev['max_clusters_minus_end']}, "
          f"flipped={ev['flipped']}; cusps at {d_true:.2f}mm not {d_flip:.2f}mm")


def test_an_inverted_label_array_FAILS_both_bands(mesh, labels):
    """The gates must be able to fail. Swap gum and tooth and both must go."""
    v, f = mesh
    _, cleaned, _, _ = labels
    flipped = np.where(cleaned == 0, 99, 0).astype(np.int64)
    r = sd.band_report(v, f, flipped)
    assert not r["ok"]
    assert set(r["failed"]) == {"occlusal_band_is_tooth", "gingival_band_is_gum"}
    print(f"PASS  an inverted array fails both: {r['failed']}")


def test_a_band_with_no_vertices_fails_rather_than_passing_vacuously(mesh):
    """s.14: a measurement that cannot be taken is not a pass."""
    v, f = mesh
    # SCALED, not translated. Moving the mesh leaves its depth SPAN alone, so
    # both bands stay populated - the first version of this fixture did that
    # and asserted nothing. Shrinking it to a tenth puts the whole scan inside
    # the occlusal band and leaves the gingival one genuinely empty.
    small = v * 0.1
    r = sd.band_report(small, f, np.zeros(len(v), np.int64))
    assert r["depth_span_mm"] < sd.GINGIVAL_BAND_MM[0], r["depth_span_mm"]
    assert r["gingival"]["n"] == 0
    assert r["gingival"]["gum_share"] is None
    assert not r["ok"] and "gingival_band_is_gum" in r["failed"]
    print("PASS  an empty band returns None and fails, never passes on zero")


def test_the_cleanup_does_not_move_geometry_or_lose_a_tooth(mesh, labels):
    """The cleanup rewrites labels ONLY. Vertex ids are the client/server
    contract (CLAUDE.md non-negotiable 2) and a tooth that vanishes silently
    is worse than one that is flagged."""
    v, f = mesh
    raw, cleaned, report, name = labels
    assert len(cleaned) == len(raw) == len(v)
    assert not report.get("teeth_lost"), report["teeth_lost"]
    assert report["protected_respected"] is True
    before = set(int(x) for x in np.unique(raw)) - {0}
    after = set(int(x) for x in np.unique(cleaned)) - {0}
    assert after == before, (before - after, after - before)
    print(f"PASS  {name}: {len(after)} teeth in and out, "
          f"{report['vertices_changed']:,} vertices relabelled, "
          f"{len(report['reassigned_islands'])} specks reassigned")


def test_a_protected_vertex_is_never_rewritten(mesh):
    """Clinician brush edits override the cleanup.

    NOTHING POPULATES THIS TODAY - the brush in this app is a selection tool
    and the backend stores no per-vertex manual label - so the guarantee is
    enforced and tested here before there is a feature to rely on it.
    """
    v, f = mesh
    lab = np.zeros(len(v), np.int64)
    lab[:200] = 46                       # a speck, far under the island floor
    protected = np.zeros(len(v), bool)
    protected[:200] = True
    out, rep = sc.clean_labels(v, f, lab, protected=protected)
    assert np.array_equal(out[:200], lab[:200]), \
        "the cleanup rewrote vertices a clinician had set by hand"
    assert rep["protected_respected"] is True
    print("PASS  a protected speck survives a cleanup that would remove it")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
