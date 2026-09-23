"""Both constructions derive the SAME stage matrix, to 0 ULP.

AGENT_BRIEF 2.2. The collar path and the deformation path now pose the same
tooth. If each derived its own stage matrix they could drift by a rounding
difference and nothing would catch it: the crowns would be posed almost
identically, every topology gate would pass, and the two paths would be
quietly incomparable - so a difference in their gate verdicts could not be
attributed to the construction rather than to the pose.

0 ULP, not "close". `np.array_equal` on the raw float64, because the point is
that there is ONE derivation, not two that agree.
"""
from __future__ import annotations

import numpy as np
import pytest

import api_core
import core_geometry as cg
import stage_matrix

PRESCRIPTIONS = [
    {"d_oa": 0.25},
    {"tip_deg": 3.0, "torque_deg": -2.0, "rotation_deg": 4.0,
     "d_md": 0.4, "d_bl": 0.0, "d_oa": 0.0},
    {"rotation_deg": 15.0},
    {"d_bl": 1.0, "torque_deg": 10.0, "tip_deg": 8.0},
    {"d_oa": -0.5, "d_md": -0.2},
    {},                                    # all zeros: must give the identity
]


def _frame():
    """A tooth frame shaped as `/cut` stores it - numpy axes, not lists.

    `cg.kinematic_matrix` scales the axis vectors, so a list would raise
    `TypeError: can't multiply sequence by non-int of type float`.
    """
    return {"u_oa": np.array([0.0, 0.0, 1.0]),
            "u_md": np.array([1.0, 0.0, 0.0]),
            "u_bl": np.array([0.0, 1.0, 0.0])}


# ---------------------------------------------------------------------------
# The clinical channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clinical", PRESCRIPTIONS)
@pytest.mark.parametrize("k,n", [(0, 5), (1, 5), (3, 5), (5, 5), (1, 1), (7, 31)])
def test_the_collar_path_and_the_shared_function_agree_exactly(clinical, k, n):
    a = api_core._stage_clinical(clinical, k, n)
    b = stage_matrix.stage_clinical(clinical, k, n)
    assert a == b, (a, b)
    for key in stage_matrix.CLINICAL_KEYS:
        assert a[key] == b[key], (key, a[key], b[key])


def test_api_core_delegates_rather_than_reimplementing():
    """Static: the collar path must not grow its own copy of the arithmetic."""
    import inspect
    src = inspect.getsource(api_core._stage_clinical)
    assert "stage_matrix.stage_clinical" in src, \
        "api_core._stage_clinical no longer delegates to the shared function"
    assert "k / n" not in src, \
        "the scaling arithmetic has been reimplemented in api_core"
    print("PASS  api_core._stage_clinical delegates to stage_matrix")


# ---------------------------------------------------------------------------
# The 4x4 itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("clinical", PRESCRIPTIONS)
def test_the_two_paths_produce_bit_identical_matrices(clinical):
    """0 ULP. Not allclose - identical."""
    frame, c_res, n = _frame(), np.array([0.0, 0.0, -10.0]), 5
    for k in range(n + 1):
        collar = cg.kinematic_matrix(
            frame, c_res, **api_core._stage_clinical(clinical, k, n))
        deform = stage_matrix.stage_matrix(frame, c_res, clinical, k, n)

        assert np.array_equal(collar, deform), (
            f"stage {k}/{n} differs; max |delta| = "
            f"{np.abs(collar - deform).max():.3e}")
        assert collar.dtype == deform.dtype

        ulp = np.abs(collar - deform) / np.maximum(np.abs(collar), 1e-300)
        assert ulp.max() == 0.0, ulp.max()
    print(f"PASS  {clinical} - all {n + 1} stages identical at 0 ULP")


def test_manufacturing_v2_uses_the_same_function():
    """The deformation path's own entry point, not just the module."""
    import inspect

    import manufacturing_v2 as v2
    src = inspect.getsource(v2.stage_matrices_for)
    assert "stage_matrix.stage_matrix" in src, src
    assert "kinematic_matrix" not in src, \
        "manufacturing_v2 re-derives the matrix instead of sharing it"
    print("PASS  manufacturing_v2.stage_matrices_for calls the shared function")


def test_stage_zero_is_the_identity_and_stage_n_is_the_whole_prescription():
    """The two endpoints anchor the scaling; everything between is linear."""
    frame, c_res = _frame(), np.array([0.0, 0.0, -10.0])
    clinical = {"tip_deg": 6.0, "d_oa": 0.5}

    M0 = stage_matrix.stage_matrix(frame, c_res, clinical, 0, 4)
    assert np.array_equal(M0, np.eye(4)), M0

    M4 = stage_matrix.stage_matrix(frame, c_res, clinical, 4, 4)
    full = cg.kinematic_matrix(frame, c_res, **clinical)
    assert np.array_equal(M4, full)
    print("PASS  stage 0 is the identity; stage N is the full prescription")


def test_a_stage_is_rebuilt_not_interpolated():
    """The invariant A1.3 exists to protect, demonstrated.

    A 4x4 lerp is not a rotation at any intermediate t: the 3x3 block of
    (1-t)I + tR is not orthonormal. The rebuilt stage is rigid; the lerp is
    not, and by orders of magnitude.
    """
    frame, c_res = _frame(), np.array([0.0, 0.0, -10.0])
    clinical = {"rotation_deg": 20.0, "tip_deg": 10.0}
    full = cg.kinematic_matrix(frame, c_res, **clinical)

    rebuilt = stage_matrix.stage_matrix(frame, c_res, clinical, 1, 2)
    lerp = 0.5 * np.eye(4) + 0.5 * full

    def rigidity(M):
        R = M[:3, :3]
        return float(np.abs(R.T @ R - np.eye(3)).max())

    r_rebuilt, r_lerp = rigidity(rebuilt), rigidity(lerp)
    assert r_rebuilt < 1e-12, r_rebuilt
    assert r_lerp > 1e-3, r_lerp
    print(f"PASS  rebuilt R'R-I = {r_rebuilt:.2e}; a lerp reaches "
          f"{r_lerp:.2e} and is not a rotation")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
