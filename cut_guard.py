"""
cut_guard.py (v2) — decide whether a cut actually landed on the cervical
margin, and pick the tolerance that gets it there.
"""

import numpy as np

MIN_RIM_CONCAVITY = 0.05
MIN_COMPACTNESS = 0.30


def _volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0


def _area(v, f):
    return 0.5 * np.linalg.norm(
        np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()


def compactness(v, f):
    vol, ar = _volume(v, f), _area(v, f)
    return 0.0 if (ar <= 0 or vol <= 0) else float((36 * np.pi * vol ** 2) ** (1 / 3) / ar)


def rim_concavity(rim_idx, concavity):
    r = np.asarray(concavity)[np.asarray(rim_idx, np.int64)]
    return {"mean": float(r.mean()), "median": float(np.median(r)),
            "p25": float(np.percentile(r, 25)), "n": int(r.size)}


def check_crown(crown_verts, crown_faces, rim_idx, concavity,
                min_rim_concavity=MIN_RIM_CONCAVITY,
                min_compactness=MIN_COMPACTNESS):
    """Serialisable verdict on whether a crown looks like a crown.

    REPORTING, NOT GATING — deliberately, and this is the whole point of the
    current state.

    This function used to compute both metrics and then `return ok=True`
    unconditionally, with the note "Hard-bypassed for testing Phase 2 kinematics
    and socket rendering". The bypass made the caller's 422 unreachable, so for
    several phases the pre-cut guard did nothing at all while appearing to.

    It is now un-bypassed, but the two thresholds are reported rather than
    enforced. The reason is evidence, not caution: MIN_COMPACTNESS (0.30) and
    MIN_RIM_CONCAVITY (0.05) have never been measured against real cuts. Turning
    them into refusals in the same change that re-enables them would start
    rejecting cuts that work today, and nobody would know whether the threshold
    or the cut was wrong.

    So `ok` stays True and `advisory` carries the verdict. Promote a threshold
    to a refusal only once there are measurements behind it — and when you do,
    write down the numbers that justified the cut-off.
    """
    rc = rim_concavity(rim_idx, concavity)
    comp = compactness(crown_verts, crown_faces)
    vol = _volume(crown_verts, crown_faces)

    rim_ok = bool(rc.get("median", 0.0) >= min_rim_concavity)
    solid_ok = bool(comp >= min_compactness)

    notes = []
    if not solid_ok:
        notes.append(f"compactness {comp:.3f} is below {min_compactness:.2f} — the selection is "
                     f"shell-like rather than a solid crown")
    if not rim_ok:
        notes.append(f"median rim concavity {rc.get('median', 0.0):.3f} is below "
                     f"{min_rim_concavity:.2f} — the margin may not be sitting in the sulcus")

    return {
        "rim_concavity": {k: (round(val, 4) if isinstance(val, float) else val)
                          for k, val in rc.items()},
        "min_rim_concavity": min_rim_concavity,
        "compactness": round(comp, 4),
        "min_compactness": min_compactness,
        "volume_mm3": round(vol, 3),
        "reached_margin": rim_ok,
        "is_solid": solid_ok,
        # Never gates. See the docstring.
        "ok": True,
        "advisory_ok": bool(rim_ok and solid_ok),
        "thresholds_are": "unvalidated software heuristics — reported, not enforced",
        "diagnosis": ("Crown geometry is consistent with an extractable tooth."
                      if rim_ok and solid_ok else "; ".join(notes)),
    }


def auto_tolerance(dist, lo=1.0, hi=25.0, step=0.25,
                   flat_frac=0.02, window=6):
    d = np.asarray(dist)
    tols = np.arange(lo, hi + step, step)
    counts = np.array([int((d <= t).sum()) for t in tols])
    growth = (np.diff(counts) / np.maximum(counts[:-1], 1)) / step

    run = 0
    for i, g in enumerate(growth):
        run = run + 1 if g < flat_frac else 0
        if run >= window:
            k = i - window + 1
            return {"tolerance": float(tols[k + 1]),
                    "vertex_count": int(counts[k + 1]),
                    "plateau_found": True,
                    "curve": {"tolerances": tols.tolist(), "counts": counts.tolist()}}
    return {"tolerance": float(hi), "vertex_count": int(counts[-1]),
            "plateau_found": False,
            "note": "Sulcus barrier plateau check.",
            "curve": {"tolerances": tols.tolist(), "counts": counts.tolist()}}