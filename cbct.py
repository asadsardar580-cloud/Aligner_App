"""CBCT integration points — INTERFACE ONLY. Nothing here segments a volume.

READ THIS BEFORE USING ANY OF IT. This module defines where CBCT would attach
and what it would have to supply. It does NOT read DICOM, does NOT segment
roots, and every entry point raises NotImplementedError with what is missing.

That is deliberate. A stub that returns plausible-looking roots would be the
single most dangerous thing in this repo: the virtual root cones the app draws
today are explicitly labelled "virtual root (estimated)" and rendered as
translucent wireframe precisely so a clinician can never mistake a mathematical
projection for imaging. Something that silently returned a cone through a
function called `segment_roots` would destroy that distinction.

WHAT REAL CBCT INTEGRATION ACTUALLY REQUIRES, and why it is not a small task:

1. REGISTRATION. The intraoral scan and the CBCT volume are captured in
   different coordinate systems, on different days, with the patient's jaw in a
   different position. Aligning them is a rigid registration against shared
   crown surfaces — and it MUST NOT move the scan, because scanner coordinates
   are sacred (rule 3.1). The transform is stored as metadata and applied to
   the CBCT, never the other way round.

2. SEGMENTATION. Root surfaces in a CBCT are low-contrast against trabecular
   bone, and metal restorations scatter badly. This is its own model.

3. VALIDATION. A registration that is 2mm out moves C_res by 2mm, and C_res is
   the pivot for every movement in the case. Any real implementation must
   report its registration residual, and the UI must show it.

Until those exist, `root_length_mm` stays a Wheeler average and the cone stays
labelled an estimate.
"""
from __future__ import annotations

# What a caller must provide for registration to be attempted at all.
REQUIRED_FOR_REGISTRATION = (
    "a DICOM series or NIfTI volume with consistent slice spacing",
    "the intraoral scan already loaded in the session (the fixed reference)",
    "at least three corresponding crown surfaces visible in both",
)

# The residual above which a registration must not be used for C_res. HEURISTIC
# and deliberately tight: C_res sits 9-13mm from the crown, so an angular error
# at the crown is amplified at the apex.
MAX_REGISTRATION_RESIDUAL_MM = 0.5


def registration_requirements() -> dict:
    """What CBCT integration would need. Safe to call; informational only."""
    return {
        "implemented": False,
        "requires": list(REQUIRED_FOR_REGISTRATION),
        "max_residual_mm": MAX_REGISTRATION_RESIDUAL_MM,
        "residual_basis": "software heuristic",
        "invariant": "Registration transforms the CBCT INTO scanner space. The "
                     "intraoral scan is never moved — inter-arch bite registration "
                     "and every vertex id already issued depend on it.",
        "current_root_source": "Wheeler population averages, drawn as a translucent "
                               "wireframe cone labelled 'virtual root (estimated)'.",
    }


def register_volume(volume_path: str, session_id: str):
    """Rigid-register a CBCT volume into the session's scanner space."""
    raise NotImplementedError(
        "CBCT registration is not implemented. It requires: "
        + "; ".join(REQUIRED_FOR_REGISTRATION)
        + ". Returning an unregistered or approximate transform here would move "
          "C_res by the registration error, and C_res is the pivot for every "
          "movement in the case.")


def segment_roots(volume_path: str, session_id: str):
    """Extract true anatomical root surfaces from a registered CBCT."""
    raise NotImplementedError(
        "CBCT root segmentation is not implemented. Until it is, roots are "
        "Wheeler averages rendered as translucent wireframe cones labelled "
        "'virtual root (estimated)'. A function of this name must never return "
        "a cone — a clinician seeing output from segment_roots would reasonably "
        "believe it came from imaging.")


def replace_virtual_roots(session_id: str):
    """Swap the estimated cones for segmented CBCT roots."""
    raise NotImplementedError(
        "Nothing can replace the virtual roots until register_volume and "
        "segment_roots exist, and until a registration residual below "
        f"{MAX_REGISTRATION_RESIDUAL_MM}mm has been demonstrated and surfaced in "
        "the UI.")
