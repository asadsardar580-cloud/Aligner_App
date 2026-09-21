#!/usr/bin/env python
"""Run every installed segmentation provider over ONE scan and compare them.

    python benchmark_providers.py                       # case_lower.stl, lower
    python benchmark_providers.py --scan x.stl --jaw upper
    python benchmark_providers.py --providers crosstooth

WHAT THIS IS NOT. It is not an accuracy benchmark and it must never be quoted
as one. `benchmark_segmentation.py` implements per-class IoU, FDI accuracy,
Chamfer and Hausdorff correctly and then REFUSES TO RUN without an
independently attributed annotation, for the reason CLAUDE.md s.17 records:
this repository's only label file is ToothGroupNetwork's own prediction, and
scoring a model against its own output returns mIoU 1.0 and means nothing.
Scoring model A against model B is the same error wearing a second hat - it
says which two models agree, not which one is right.

SO WHAT IS MEASURED HERE IS GEOMETRY AND AGREEMENT, and the two answer
different questions:

  * GEOMETRY is checkable without any ground truth at all. A tooth is one
    connected lump of a plausible size. `segmentation_diagnostics` reports
    per-label vertex and face counts, bounding box, box diagonal and
    disconnected component count, and the median box diagonal is what
    separates "this model segmented badly" from "these labels are not
    indexed to this mesh" (s.24.3). A model can win on geometry and still be
    wrong about which tooth is which.

  * AGREEMENT is symmetric and attributes nothing. It is reported because it
    is the only thing that can settle a CONVENTION question - if two
    independently trained models disagree on which side of the arch is the
    3x quadrant, one of them is mirrored, and that shows up as an agreement
    score near zero against a near-perfect score under the mirror.

Runtime is measured wall-clock, once, on this machine. It is a property of
this CPU and this scan, not of the models.
"""

import argparse
import json
import sys
import time

import numpy as np

import core_geometry as cg
import segmentation_diagnostics as sd
import segmentation_providers as sp
import stl_io

DEFAULT_SCAN = "case_lower.stl"


def load_scan(path):
    v0, f0 = stl_io.parse_stl_bytes(open(path, "rb").read())
    v, f, info = cg.condition_mesh(v0, f0)
    return np.asarray(v, float), np.asarray(f, np.int64), info


def mirror_quadrants(labels):
    """Swap the two quadrants of whichever jaw these labels belong to.

    A mirrored convention is the failure mode a class-index model has that an
    FDI-native one does not, and it is invisible in every per-label metric:
    each tooth is still one plausible lump, just named as its opposite number.
    """
    out = np.asarray(labels, np.int64).copy()
    for lo, hi, delta in ((11, 18, 10), (21, 28, -10),
                          (31, 38, 10), (41, 48, -10)):
        m = (labels >= lo) & (labels <= hi)
        out[m] = labels[m] + delta
    return out


def agreement(a, b):
    """Exact-FDI agreement on the vertices BOTH call a tooth, plus the mirror.

    Restricted to the overlap on purpose. Counting a vertex one model calls
    gingiva and the other calls a molar as a disagreement mixes up two
    different questions - where the margin is, and which tooth it is - and
    the gingival margin is where every segmenter differs by a few rows of
    triangles anyway. Coverage is reported separately so the restriction
    cannot hide a model that labelled almost nothing.
    """
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    both = (a != 0) & (b != 0)
    n = int(both.sum())
    if n == 0:
        return {"overlap_vertices": 0, "exact_fdi_agreement": None,
                "agreement_if_mirrored": None}
    return {
        "overlap_vertices": n,
        "a_only_vertices": int(((a != 0) & (b == 0)).sum()),
        "b_only_vertices": int(((b != 0) & (a == 0)).sum()),
        "exact_fdi_agreement": round(float((a[both] == b[both]).mean()), 4),
        "agreement_if_mirrored": round(
            float((a[both] == mirror_quadrants(b)[both]).mean()), 4),
    }


def geometry_summary(report):
    """The three numbers that say whether labels describe teeth at all."""
    rows = report.get("teeth", [])
    diags = [r["bbox_diagonal_mm"] for r in rows
             if r.get("bbox_diagonal_mm") is not None]
    frac = [r["largest_component_fraction"] for r in rows
            if r.get("largest_component_fraction") is not None]
    return {
        "teeth": len(rows),
        "median_box_diagonal_mm": round(float(np.median(diags)), 3)
        if diags else None,
        "max_box_diagonal_mm": round(float(np.max(diags)), 3)
        if diags else None,
        "plausible_size_teeth": int(sum(1 for r in rows
                                        if r.get("plausible_size"))),
        "single_component_teeth": int(sum(1 for r in rows
                                          if r.get("components") == 1)),
        "worst_largest_component_fraction": round(float(np.min(frac)), 4)
        if frac else None,
        "indexed_to_this_mesh": report.get("indexed_to_this_mesh"),
    }


def run(scan, jaw, wanted):
    v, f, cond = load_scan(scan)
    print(f"scan       {scan}")
    print(f"mesh       {len(v):,} vertices / {len(f):,} faces  jaw={jaw}")
    print(f"welded     {cond['welded_vertices']}   "
          f"degenerate removed {cond['degenerate_faces_removed']}\n")

    results, failures = {}, {}
    for name in wanted:
        provider = sp.get(name)
        ready = provider.available()
        if not ready.get("available"):
            failures[name] = ready.get("reason")
            print(f"[{name}] SKIPPED - {ready.get('reason')}")
            continue
        print(f"[{name}] running...", flush=True)
        t0 = time.time()
        try:
            out = provider.segment(v, f, jaw)
        except Exception as e:                            # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"
            print(f"[{name}] FAILED - {failures[name]}")
            continue
        seconds = round(time.time() - t0, 2)
        labels = np.asarray(out.labels, np.int64)
        report = sd.label_report(v, f, labels)
        results[name] = {
            "labels": labels,
            "seconds": seconds,
            "geometry": geometry_summary(report),
            "fdi_present": sorted({int(x) for x in labels} - {0}),
            "meta": out.meta,
        }
        print(f"[{name}] {seconds}s  "
              f"{results[name]['geometry']['teeth']} teeth\n")

    print("=" * 74)
    print("GEOMETRY - checkable without ground truth. NOT accuracy.")
    print("=" * 74)
    head = (f"{'provider':<20}{'sec':>8}{'teeth':>7}{'med diag':>10}"
            f"{'max diag':>10}{'1-piece':>9}{'worst frac':>12}")
    print(head)
    for name, r in results.items():
        g = r["geometry"]
        print(f"{name:<20}{r['seconds']:>8}{g['teeth']:>7}"
              f"{g['median_box_diagonal_mm'] or -1:>10.3f}"
              f"{g['max_box_diagonal_mm'] or -1:>10.3f}"
              f"{g['single_component_teeth']:>9}"
              f"{g['worst_largest_component_fraction'] or -1:>12.4f}")
        print(f"{'':<20}FDI: {r['fdi_present']}")

    pairs = {}
    names = list(results)
    if len(names) >= 2:
        print("\n" + "=" * 74)
        print("AGREEMENT - symmetric, attributes nothing, settles conventions.")
        print("=" * 74)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ag = agreement(results[a]["labels"], results[b]["labels"])
                pairs[f"{a} vs {b}"] = ag
                print(f"{a} vs {b}")
                print(f"  overlap {ag['overlap_vertices']:,} vertices "
                      f"({a}-only {ag['a_only_vertices']:,}, "
                      f"{b}-only {ag['b_only_vertices']:,})")
                print(f"  exact FDI          {ag['exact_fdi_agreement']:.4f}")
                print(f"  if {b} mirrored    "
                      f"{ag['agreement_if_mirrored']:.4f}")
                verdict = ("SAME quadrant convention"
                           if ag["exact_fdi_agreement"] >
                           ag["agreement_if_mirrored"]
                           else "OPPOSITE quadrant convention - one is "
                                "mirrored")
                print(f"  -> {verdict}")

    print("\nNEITHER MODEL IS VALIDATED. There is no independently annotated "
          "\nground truth in this repository, so nothing above is an accuracy "
          "\nfigure and none of it may be quoted as one.")

    return {
        "scan": scan, "jaw": jaw,
        "vertices": int(len(v)), "faces": int(len(f)),
        "providers": {n: {k: r[k] for k in
                          ("seconds", "geometry", "fdi_present", "meta")}
                      for n, r in results.items()},
        "agreement": pairs,
        "skipped": failures,
        "is_measured_accuracy": False,
        "meaning": "geometric plausibility and inter-model agreement; NOT "
                   "accuracy, which needs an independent annotation",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default=DEFAULT_SCAN)
    ap.add_argument("--jaw", default="lower", choices=("lower", "upper"))
    ap.add_argument("--providers", default=None,
                    help="comma-separated; default is every one that is "
                         "available")
    ap.add_argument("--json", default=None, help="write the record here")
    args = ap.parse_args(argv)

    wanted = ([p.strip() for p in args.providers.split(",")]
              if args.providers else list(sp.registry()))
    record = run(args.scan, args.jaw, wanted)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=2, default=str)
        print(f"\nwritten to {args.json}")
    return 0 if record["providers"] else 1


if __name__ == "__main__":
    sys.exit(main())
