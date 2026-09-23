"""A distance that could not be measured must fail every gate that reads it.

AGENT_BRIEF defect B13 and invariant A1.6. `_point_to_surface` caught EVERY
exception and dropped through to the candidate-search approximation - the very
method its own docstring spends two paragraphs explaining is wrong (measured
13.3% correct on a steep cervical wall; 793 of 4710 vertices reporting exactly
9.0000 mm on a cast underside). A transient Open3D failure therefore had every
manufacturing gate evaluated on a known-wrong method, with nothing recorded.

A wrong answer is worse than no answer, because it answers.

These tests make Open3D raise and then check the whole chain: the distance is
NaN, the reason is recorded, and each consuming gate FAILS rather than reading
the NaN as a zero.
"""
from __future__ import annotations

import numpy as np
import pytest

import manufacturing as mfg


@pytest.fixture
def broken_open3d(monkeypatch):
    """Make `import open3d` raise inside `_point_to_surface`."""
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "open3d" or name.startswith("open3d."):
            raise RuntimeError("simulated Open3D failure")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    mfg.LAST_DISTANCE_FAILURE = None
    yield
    mfg.LAST_DISTANCE_FAILURE = None


def _plate():
    v = np.array([[0., 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    f = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
    return v, f


def _tessellated_plate(n=24, half=12.0):
    """A grid-tessellated plane and a rim on it.

    `old_site_quality` needs at least 8 occlusally-facing triangles near the
    site before it will assess anything, so a two-triangle plate refuses at
    its ray cast and never reaches the distance call this guard protects.
    """
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    v = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b, c, d = a + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            f += [[a, c, b], [b, c, d]]
    rim = np.array([[3 * np.cos(t), 3 * np.sin(t), 0.0]
                    for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
    return v, np.asarray(f, np.int64), rim


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------

def test_open3d_failure_returns_NaN_not_an_approximation(broken_open3d):
    v, f = _plate()
    pts = np.array([[1.0, 1.0, 3.0], [5.0, 5.0, 1.0]])

    d = mfg._point_to_surface(pts, v, f)

    assert d.shape == (2,), d.shape
    assert np.all(np.isnan(d)), d
    print(f"PASS  Open3D raised -> {d} (all NaN), not a fallback answer")


def test_the_reason_is_recorded(broken_open3d):
    v, f = _plate()
    mfg._point_to_surface(np.array([[1.0, 1.0, 3.0]]), v, f)

    assert mfg.LAST_DISTANCE_FAILURE is not None
    assert "simulated Open3D failure" in mfg.LAST_DISTANCE_FAILURE
    print(f"PASS  reason recorded: {mfg.LAST_DISTANCE_FAILURE[:70]}")


def test_a_successful_call_clears_the_reason():
    v, f = _plate()
    mfg.LAST_DISTANCE_FAILURE = "stale"
    d = mfg._point_to_surface(np.array([[1.0, 1.0, 3.0]]), v, f)
    assert np.all(np.isfinite(d))
    assert mfg.LAST_DISTANCE_FAILURE is None
    print("PASS  a successful measurement clears the recorded reason")


def test_the_approximation_still_exists_but_is_never_reached_automatically():
    """It is kept, named as an approximation, and called by nothing."""
    import inspect
    assert hasattr(mfg, "_point_to_surface_approximate")
    src = inspect.getsource(mfg._point_to_surface)
    assert "_point_to_surface_approximate" not in src.split('"""')[-1], \
        "the exact path can still fall through to the approximation"
    print("PASS  the approximation exists and is not reachable automatically")


# ---------------------------------------------------------------------------
# surface_deviation
# ---------------------------------------------------------------------------

def test_surface_deviation_reports_NOT_measured_rather_than_zero(broken_open3d):
    """0.0 mm would read as a perfectly reproduced cast."""
    v, f = _plate()
    out = mfg.surface_deviation(v, f, v + np.array([0.0, 0.0, 0.3]), f)

    assert out["measured"] is False, out
    assert np.isnan(out["max_mm"]), out
    assert mfg.KEY_DISTANCE_FAILURE in out, sorted(out)
    print(f"PASS  surface_deviation: measured=False, max_mm=nan, "
          f"reason recorded")


# ---------------------------------------------------------------------------
# The gates that read it
# ---------------------------------------------------------------------------

def test_a_NaN_in_EITHER_direction_fails_the_fidelity_gate():
    """The trap: `max(0.001, nan)` is 0.001, so the old check passed.

    `nan > 0.001` is False, so Python's `max` keeps its first argument. A NaN
    in the SECOND direction therefore sailed through `max(...) <= threshold`.
    Measured before the fix, this exact record returned passed=True.
    """
    for fwd, rev in (({"max_mm": 0.001}, {"max_mm": float("nan")}),
                     ({"max_mm": float("nan")}, {"max_mm": 0.001}),
                     ({"max_mm": float("nan")}, {"max_mm": float("nan")})):
        gate = mfg.aggregate_print_gate({
            "cast_fidelity": {"original_to_final": fwd,
                              "final_to_original": rev}})
        row = next(g for g in gate["gates"]
                   if g["gate"] == "unaffected_cast_fidelity_two_sided")
        assert row["passed"] is False, (fwd, rev, row)
    print("PASS  a NaN in either direction (or both) fails the fidelity gate")


def test_a_real_pair_of_small_distances_still_passes():
    """The control for the control: the gate must still be able to PASS."""
    gate = mfg.aggregate_print_gate({
        "cast_fidelity": {"original_to_final": {"max_mm": 0.0,
                                                "measured": True},
                          "final_to_original": {"max_mm": 0.0,
                                                "measured": True}}})
    row = next(g for g in gate["gates"]
               if g["gate"] == "unaffected_cast_fidelity_two_sided")
    assert row["passed"] is True, row
    print("PASS  two finite 0.0 mm measurements still pass the gate")


def test_an_unmeasured_direction_fails_even_with_a_finite_number():
    """`measured: False` beside a stale number must not pass."""
    gate = mfg.aggregate_print_gate({
        "cast_fidelity": {
            "original_to_final": {"max_mm": 0.0, "measured": True},
            "final_to_original": {"max_mm": 0.0, "measured": False,
                                  mfg.KEY_DISTANCE_FAILURE: "boom"}}})
    row = next(g for g in gate["gates"]
               if g["gate"] == "unaffected_cast_fidelity_two_sided")
    assert row["passed"] is False, row
    assert row["measured"][mfg.KEY_DISTANCE_FAILURE] == "boom"
    print("PASS  measured=False fails even beside a finite 0.0")


def test_old_site_quality_refuses_to_assess_when_open3d_is_broken(
        broken_open3d):
    """Whatever fails first, the answer is "not assessed", never "pristine"."""
    v, f, rim = _tessellated_plate()
    out = mfg.old_site_quality(v, f, v, f, rim,
                               u_oa=np.array([0.0, 0.0, 1.0]))

    assert out["measured"] is False, out
    gate = mfg.aggregate_print_gate({"interfaces": [{"old_site": out}]})
    row = next(g for g in gate["gates"] if g["gate"] == "old_site_restored")
    assert row["passed"] is False, row
    print(f"PASS  old_site_quality: measured=False "
          f"({out.get('reason', '')[:50]}), gate fails")


def test_old_site_quality_guards_its_own_distance_call(monkeypatch):
    """The specific guard added in 0.4, reached directly.

    With Open3D broken the function refuses earlier, at its ray cast, so that
    path never exercises the distance guard. Here the ray cast works and only
    `_point_to_surface` fails - which is the case the guard exists for.

    Without it, `nan > tol_mm` is False for every point, so `moved` is all
    False and the site reports as one nothing had touched: a pristine old
    socket, from a measurement that never happened.
    """
    v, f, rim = _tessellated_plate()

    ok = mfg.old_site_quality(v, f, v, f, rim,
                              u_oa=np.array([0.0, 0.0, 1.0]))
    assert ok.get("measured") is True, (
        f"the fixture must REACH the distance call for this control to mean "
        f"anything: {ok.get('reason')}")

    monkeypatch.setattr(mfg, "_point_to_surface",
                        lambda pts, *a, **k: np.full(len(np.atleast_2d(pts)),
                                                     np.nan))
    monkeypatch.setattr(mfg, "LAST_DISTANCE_FAILURE", "simulated")

    out = mfg.old_site_quality(v, f, v, f, rim,
                               u_oa=np.array([0.0, 0.0, 1.0]))

    assert out["measured"] is False, out
    assert mfg.KEY_DISTANCE_FAILURE in out, sorted(out)
    gate = mfg.aggregate_print_gate({"interfaces": [{"old_site": out}]})
    row = next(g for g in gate["gates"] if g["gate"] == "old_site_restored")
    assert row["passed"] is False, row
    print("PASS  a NaN site distance refuses to assess, and the gate fails")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
