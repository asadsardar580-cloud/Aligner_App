"""
inspect_segmentation_output.py — read what the network actually returned.

Run this on the JSON your /api/segment endpoint produced, BEFORE trusting any
label for planning.

    python inspect_segmentation_output.py result.json --jaw lower

WHAT IT CHECKS, AND WHY EACH ONE EARNED ITS PLACE
-------------------------------------------------
* Which numbering space the labels are in. ToothGroupNetwork's internal
  classes are 1..16 (two quadrants of eight, central incisor outward). FDI is
  11-18/21-28 for the maxilla and 31-38/41-48 for the mandible. The +20 offset
  in predict_utils.py converts maxillary FDI to mandibular FDI — it does NOT
  convert internal indices to FDI. If your array holds 1..16 you are reading
  a pre-conversion value and the offset never applied to it.

* Vertices per label. A tooth built from twelve vertices is a stray patch. A
  label set can look complete while its contents are nonsense, which is the
  same trap as counting fourteen regions and finding the second premolar is
  the largest tooth in the arch.

* Missing positions and out-of-range labels. A gap at one position with an
  unexplained high-numbered label present is the signature of the grouping
  module failing to assign a cluster to a tooth class.
"""

import argparse
import json
import sys
from collections import Counter

# ToothGroupNetwork internal class -> FDI, by quadrant.
# 1..8 and 9..16 run central incisor -> third molar.
POSITION_NAMES = [
    "central incisor", "lateral incisor", "canine", "first premolar",
    "second premolar", "first molar", "second molar", "third molar",
]

UPPER_FDI = list(range(11, 19)) + list(range(21, 29))
LOWER_FDI = list(range(31, 39)) + list(range(41, 49))


def internal_to_fdi(v, jaw):
    """Map internal class 1..16 onto FDI for the given jaw."""
    if not 1 <= v <= 16:
        return None
    table = LOWER_FDI if jaw == "lower" else UPPER_FDI
    return table[v - 1]


def classify_space(values, jaw):
    """Decide which numbering space these labels are in."""
    nz = [v for v in values if v != 0]
    if not nz:
        return "empty"
    valid_fdi = set(LOWER_FDI if jaw == "lower" else UPPER_FDI)
    other_fdi = set(UPPER_FDI if jaw == "lower" else LOWER_FDI)
    in_fdi = sum(1 for v in nz if v in valid_fdi)
    in_other = sum(1 for v in nz if v in other_fdi)
    in_internal = sum(1 for v in nz if 1 <= v <= 16)
    if in_fdi == len(nz):
        return "fdi_correct_jaw"
    if in_other == len(nz):
        return "fdi_wrong_jaw"
    if in_internal == len(nz):
        return "internal_1_to_16"
    return "mixed_or_unknown"


def load_labels(path, key=None):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}, "(bare array)"
    meta = {k: v for k, v in data.items()
            if not isinstance(v, (list, dict))}
    if key:
        return data[key], meta, key
    for candidate in ("labels", "sem", "fdi", "predictions"):
        if candidate in data and isinstance(data[candidate], list):
            return data[candidate], meta, candidate
    arrays = [k for k, v in data.items() if isinstance(v, list)]
    raise SystemExit(f"No obvious label array. Arrays present: {arrays}. "
                     f"Re-run with --key <name>.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--jaw", required=True, choices=["upper", "lower"])
    ap.add_argument("--key", default=None)
    ap.add_argument("--min-vertices", type=int, default=100,
                    help="below this a region is a patch, not a tooth")
    args = ap.parse_args()

    labels, meta, key = load_labels(args.path, args.key)
    counts = Counter(labels)
    total = len(labels)
    background = counts.get(0, 0)

    print("=" * 70)
    print(f"FILE          {args.path}")
    print(f"ARRAY         {key}   ({total:,} vertices)")
    print(f"DECLARED JAW  {args.jaw}")
    for k, v in meta.items():
        print(f"META          {k} = {v!r}")
        if k.lower() in ("jaw", "jaw_type") and str(v).lower() != args.jaw:
            print(f"   ** file records jaw={v!r}, you declared {args.jaw!r} **")
    print(f"BACKGROUND    {background:,} ({background/max(total,1):.1%})")

    space = classify_space(labels, args.jaw)
    print()
    print("=" * 70)
    print(f"NUMBERING SPACE: {space}")
    print("=" * 70)
    explain = {
        "fdi_correct_jaw": "Labels are FDI for the declared jaw. Good.",
        "fdi_wrong_jaw": (
            "Labels are FDI for the OPPOSITE jaw. The +20 offset was applied "
            "to the wrong arch, or get_jaw disagreed with your UI. Every "
            "tooth in this scan is misnamed. Do not plan from it."),
        "internal_1_to_16": (
            "Labels are ToothGroupNetwork internal class indices, not FDI. "
            "The conversion to FDI has not run, or you are reading a "
            "pre-conversion array. Mapping shown below is what they WOULD be."),
        "mixed_or_unknown": (
            "Labels do not fit any single space. Some values are outside "
            "1..16 and outside FDI entirely — see the table."),
        "empty": "No tooth labels at all; only background.",
    }
    print(explain[space])

    print()
    print("=" * 70)
    print("LABEL HISTOGRAM")
    print("=" * 70)
    print(f"{'label':>7} {'vertices':>10} {'share':>7}  interpretation")
    problems = []
    for v in sorted(counts):
        if v == 0:
            continue
        n = counts[v]
        share = n / max(total, 1)
        fdi = internal_to_fdi(v, args.jaw)
        if 1 <= v <= 16:
            pos = POSITION_NAMES[(v - 1) % 8]
            quad = 1 if v <= 8 else 2
            note = f"internal {v} -> FDI {fdi}  (quadrant {quad}, {pos})"
        elif v in UPPER_FDI or v in LOWER_FDI:
            note = f"FDI {v}"
        else:
            note = "** OUT OF RANGE — not internal 1..16, not FDI **"
            problems.append(v)
        flag = "  <-- SLIVER" if n < args.min_vertices else ""
        print(f"{v:>7} {n:>10,} {share:>6.2%}  {note}{flag}")
        if n < args.min_vertices:
            problems.append(v)

    if space == "internal_1_to_16" or any(1 <= v <= 16 for v in counts):
        print()
        print("=" * 70)
        print("ANATOMICAL COMPLETENESS (7-to-7, third molars excluded)")
        print("=" * 70)
        for quad, base in ((1, 0), (2, 8)):
            missing, present = [], []
            for i in range(7):                  # positions 1..7, skip 8
                v = base + i + 1
                (present if counts.get(v) else missing).append(
                    (v, POSITION_NAMES[i]))
            print(f"  quadrant {quad}: {len(present)}/7 present")
            for v, name in missing:
                print(f"      MISSING internal {v:>2}  ({name}) "
                      f"-> would be FDI {internal_to_fdi(v, args.jaw)}")

    if problems:
        print()
        print("=" * 70)
        print("READ THIS")
        print("=" * 70)
        oor = sorted({p for p in problems if not (1 <= p <= 16)})
        if oor:
            print(f"  Labels {oor} are outside every known numbering space.")
            print("  In this codebase a high sentinel value usually marks a")
            print("  cluster the grouping module could not assign to a tooth")
            print("  class. Find where it is produced:")
            print("      grep -rn '100' ToothGroupNetwork/inference_pipelines/")
            print("      grep -rn '+ 100\\|100 +' ToothGroupNetwork/")
            print("  A missing tooth position alongside an unassigned cluster")
            print("  of similar size means that tooth WAS segmented and then")
            print("  failed classification. The geometry may be usable; the")
            print("  NAME is not.")
        print()
        print("  Segmentation is not verified. Do not carve a socket or export")
        print("  an appliance from these labels.")
        sys.exit(1)

    print("\n  Labels are internally consistent. Still verify against the mesh.")


if __name__ == "__main__":
    main()
