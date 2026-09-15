"""
arch_frame.py — establish a biological coordinate system for the arch.

THE PROBLEM THIS SOLVES (3Shape, 09:00+)
-----------------------------------------
An intraoral scanner is held at whatever angle the operator managed. Nothing
in the resulting STL knows which way is apical. Apply "torque" in that space
and the tooth twists diagonally, because torque is defined about the
mesiodistal axis of a tooth in an arch, not about scanner-X.

HOW THIS DIFFERS FROM ROTATING THE SCAN
----------------------------------------
The obvious implementation is to rotate every vertex so the occlusal plane
becomes Z=0. Do not do that. Your own hard constraint says the arch is never
re-centred or re-scaled, because inter-arch bite registration lives in raw
scanner coordinates -- and rotating mid-session would also invalidate every
vertex index already handed to the browser, every cached geodesic field, and
every seed point, while forcing an inverse transform on export that has to be
right every single time or the printed cast is wrong.

3Shape does not move the scan either. Defining the occlusal plane there
establishes a REFERENCE, and the software then knows which way is up. That is
what this module produces: a rotation basis stored as session metadata. The
geometry never moves. Exactly the same reasoning as framing the camera
instead of centring the mesh.

WHAT IT IS FOR
--------------
  * the gizmo's rotation rings become tip / torque / rotation instead of
    world XYZ;
  * "apical" is defined for virtual-root extrapolation, so C_res goes into
    the bone rather than off at an angle;
  * a per-tooth long axis derived from a flat molar rim can be checked
    against the arch and flagged when it disagrees.
"""

import numpy as np


def fit_occlusal_frame(left_pt, right_pt, anterior_pt, arch_centroid):
    """
    Build an arch basis from three clicked landmarks.

    left_pt / right_pt   posterior landmarks, one per side (molar cusp tips)
    anterior_pt          midline landmark (incisor contact or canine tip)
    arch_centroid        mean of the arch vertices, used only to resolve the
                         sign of the occlusal normal

    Returns a dict with an orthonormal basis:
        u_occ   occlusal normal, pointing AWAY from the tissue (out of the mouth)
        u_sag   sagittal / midline, pointing anteriorly
        u_tra   transverse, completing a right-handed set (toward the left side)

    The sign of `u_occ` is resolved geometrically rather than assumed. Three
    points define a plane but not a side, and getting it backwards puts C_res
    above the crown instead of in the bone -- which produces smooth,
    confident, entirely wrong movement.
    """
    L = np.asarray(left_pt, float)
    R = np.asarray(right_pt, float)
    A = np.asarray(anterior_pt, float)
    C = np.asarray(arch_centroid, float)

    n = np.cross(R - L, A - L)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        raise ValueError(
            "The three landmarks are collinear, so they do not define a plane. "
            "Pick one point on each posterior segment and one at the midline.")
    u_occ = n / nn

    posterior_mid = (L + R) / 2.0
    # The occlusal normal must point out of the mouth. The arch centroid sits
    # on the tissue side of the occlusal plane, so the normal pointing away
    # from it is the one we want.
    if u_occ @ (posterior_mid - C) < 0:
        u_occ = -u_occ

    # Transverse axis comes from the INTERMOLAR line, not from the anterior
    # landmark. Deriving the midline from the anterior point instead would be
    # circular -- that point would lie on the sagittal axis by construction
    # and its lateral offset would always read zero, so a misclicked midline
    # could never be detected.
    tra = R - L
    tra = tra - u_occ * (tra @ u_occ)
    nt = np.linalg.norm(tra)
    if nt < 1e-6:
        raise ValueError("The two posterior landmarks coincide in the occlusal plane.")
    u_tra = tra / nt
    u_sag = np.cross(u_occ, u_tra)

    # point the sagittal axis anteriorly; flip the transverse with it so the
    # basis stays right-handed
    if u_sag @ (A - posterior_mid) < 0:
        u_sag = -u_sag
        u_tra = -u_tra

    half = np.linalg.norm(R - L) / 2.0
    lateral_offset = float(abs((A - posterior_mid) @ u_tra))
    depth = float((A - posterior_mid) @ u_sag)

    return {
        "u_occ": u_occ, "u_sag": u_sag, "u_tra": u_tra,
        "origin": posterior_mid,
        "intermolar_width_mm": round(float(np.linalg.norm(R - L)), 3),
        "arch_depth_mm": round(depth, 3),
        "anterior_lateral_offset_mm": round(lateral_offset, 3),
        "midline_warning": (
            None if half < 1e-9 or lateral_offset / half < 0.15 else
            f"Anterior landmark sits {lateral_offset:.1f} mm off the posterior "
            f"midpoint. If that is a genuine midline shift, fine; if it is a "
            f"misclick, every torque value will be biased."),
    }


def to_json(frame):
    """Serialisable form for the client."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in frame.items()}


def basis_matrix(frame):
    """
    3x3 whose COLUMNS are (u_tra, u_sag, u_occ).

    Columns, not rows: this maps arch-local coordinates into scanner
    coordinates, which is the direction the gizmo needs to orient itself.
    Its transpose goes the other way.
    """
    return np.column_stack([frame["u_tra"], frame["u_sag"], frame["u_occ"]])


def reconcile_tooth_frame(tooth_frame, arch_frame, warn_deg=25.0):
    """
    Compare a tooth's long axis against the arch's apical direction.

    derive_frame_from_region gets u_oa from crown centroid minus rim centroid.
    On an incisor that is crisp. On a molar with a broad flat rim the two
    centroids are close together and the axis is dominated by noise -- which
    is precisely the tooth where a wrong pivot does the most damage, because
    C_res is being extrapolated 9 mm along it.

    Returns the angle and a flag. It does not overwrite the tooth axis: the
    tooth knows its own inclination and a genuinely tipped molar SHOULD
    disagree with the arch. This reports, the clinician decides.

    SIGN CONVENTION, corrected. This compared u_oa against -u_occ and then took
    abs(cos) to make the numbers come out. u_oa points OCCLUSALLY, so against
    the apical direction a perfect tooth reads 180 deg and only the abs() was
    dragging it back to 0. That abs() also destroyed the one thing this check
    is uniquely able to catch: an INVERTED long axis scored a flawless 0 deg,
    exactly like a perfect one, while C_res sat above the crown instead of in
    the bone. Compare against +u_occ and keep the sign.
    """
    u_oa = np.asarray(tooth_frame["u_oa"], float)
    u_oa = u_oa / np.linalg.norm(u_oa)
    u_occ = np.asarray(arch_frame["u_occ"], float)
    u_occ = u_occ / np.linalg.norm(u_occ)

    # atan2 of the cross norm against the dot: well conditioned near 0 and 180,
    # where arccos loses half its digits.
    deg = float(np.degrees(np.arctan2(
        float(np.linalg.norm(np.cross(u_oa, u_occ))), float(u_oa @ u_occ))))
    inverted = deg > 90.0

    if inverted:
        note = (f"Tooth long axis points {deg:.1f} deg from the arch OCCLUSAL "
                f"direction, i.e. it is inverted — it points into the bone. "
                f"C_res is being extrapolated the wrong way, above the crown "
                f"instead of into the socket. Do not apply kinematics.")
    elif deg > warn_deg:
        note = (f"Tooth long axis is {deg:.1f} deg from the arch occlusal "
                f"direction. Either this tooth is genuinely tipped, or the "
                f"rim was too flat to define an axis. Check before relying "
                f"on C_res -- the pivot is extrapolated along this vector.")
    else:
        note = None

    return {
        "angle_to_arch_occlusal_deg": round(deg, 2),
        "inverted": inverted,
        "agrees": (not inverted) and deg <= warn_deg,
        "note": note,
    }
