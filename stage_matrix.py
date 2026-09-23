"""The stage matrix, in ONE place, for every construction.

WHY THIS MODULE EXISTS. Two manufacturing paths now pose the same tooth: the
collar path in `api_core.build_stage_bundle` and the deformation path in
`manufacturing_v2`. If each derived its own stage matrix they could drift by a
rounding difference and nothing would notice - the crowns would be posed
almost identically, every topology gate would pass, and the two paths would be
quietly incomparable. AGENT_BRIEF 2.2 therefore requires the deformation path
to call the SAME function, and a test to prove the two agree at 0 ULP.

It lives here rather than in `api_core` because `manufacturing_v2` is pure and
must not import FastAPI, and rather than in `core_geometry` because it is a
staging policy (how a prescription is divided over N aligners), not geometry.

THE RULE IT ENCODES, which is not negotiable (AGENT_BRIEF A1.3):
stage k is the clinical prescription scaled to k/N and rebuilt through
`cg.kinematic_matrix`. It is NEVER an interpolation of the committed 4x4 -
the 3x3 block of (1-t)I + tR is not orthonormal for any t strictly between 0
and 1, so a matrix lerp shears the crown at every intermediate stage.
Measured: every rebuilt stage is rigid to 4.4e-16 (R^T R - I) while a 4x4 lerp
of the same movement reaches 1.68e-2 and is not a rotation at all.
"""
from __future__ import annotations

import numpy as np

import core_geometry as cg

#: The six clinical channels, in the order the browser and the backend agree on.
CLINICAL_KEYS = ("tip_deg", "torque_deg", "rotation_deg", "d_md", "d_bl", "d_oa")


def stage_clinical(clinical: dict, k: int, n: int) -> dict:
    """The six channels at stage k of n - the Python mirror of the client's
    `clinicalAtStage`. Absolute from T0, never an interpolation of the 4x4."""
    f = (k / n) if n > 0 else 0.0
    return {key: float((clinical or {}).get(key, 0.0) or 0.0) * f
            for key in CLINICAL_KEYS}


def stage_matrix(frame, c_res, clinical: dict, k: int, n: int) -> np.ndarray:
    """The 4x4 for one tooth at stage k of n.

    `frame` and `c_res` come from the tooth record exactly as `/cut` stored
    them. This is the whole derivation: scale the prescription, then rebuild
    through `cg.kinematic_matrix`. Both constructions call this.
    """
    return cg.kinematic_matrix(frame, c_res, **stage_clinical(clinical, k, n))
