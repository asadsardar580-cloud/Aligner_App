"""The real mandible, FDI 45, 0.25mm extrusion, through construction=deformation.

AGENT_BRIEF Phase 2 acceptance. The run must end in PRINT READY **or a NAMED
refusal with measured values**, with timing - and on this scan the expected
outcome is the second: `docs/phase_reports/phase_1.md` measured 3591
self-intersecting pairs in the T0 cast before any tooth is moved, and every
stage inherits T0's geometry. `build_case_plan` refuses on exactly that, with
the census attached, rather than building N stages that could never be print
ready and refusing each one separately for a reason that does not name the
cause.

THE PATH IS THE PRODUCT'S OWN, not a shortcut: upload -> occlusal plane ->
CrossTooth segmentation -> click-to-select -> cut -> commit the prescription
-> export, driven over HTTP through TestClient. s.26.8 established this chain
for the collar construction; this is the same chain with one request field
changed.

    .venv\\Scripts\\python.exe real_scan_deformation.py [--fdi 45] [--stages 1]

Exit 0 = PRINT READY. Exit 1 = a NAMED refusal (an expected outcome here, and
the run still prints everything it measured). Exit 2 = the run could not reach
the gate at all, which is the only real failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

SCAN = "case_lower.stl"


def _t(label, fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    dt = time.perf_counter() - t0
    print(f"  {label:<34} {dt:>8.2f}s")
    return out, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdi", type=int, default=45)
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--extrusion", type=float, default=0.25)
    ap.add_argument("--provider", default="crosstooth")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    import api_core
    from api_core import STORE

    client = TestClient(api_core.app)
    t_all = time.perf_counter()
    timings = {}

    print("=" * 74)
    print(f"REAL SCAN, construction=deformation - FDI {args.fdi}, "
          f"{args.extrusion}mm extrusion, {args.stages} stage(s)")
    print("=" * 74)

    # --- upload -----------------------------------------------------------
    try:
        raw = open(SCAN, "rb").read()
    except FileNotFoundError:
        print(f"REFUSED TO GUESS: {SCAN} is not here. Nothing synthetic is "
              f"substituted for a real scan.")
        return 2

    print(f"\n[1] upload  ({len(raw) / 1e6:.1f} MB)")
    r, timings["upload"] = _t("POST /api/session", client.post,
                              "/api/session", files={"file": (SCAN, raw,
                                                              "model/stl")},
                              data={"arch": "lower"})
    if r.status_code != 200:
        print(f"    FAILED {r.status_code}: {r.text[:400]}")
        return 2
    sid = r.json()["session_id"]
    v = STORE.require(sid, "verts")
    f = STORE.require(sid, "faces")
    print(f"    {len(v):,} verts / {len(f):,} faces   session {sid}")

    # --- occlusal plane ---------------------------------------------------
    print("\n[2] occlusal plane")
    centroid = v.mean(axis=0)
    d = v - centroid
    _, _, vt = np.linalg.svd(d[::37], full_matrices=False)
    occ = vt[2]
    slab = v[(d @ occ) > np.percentile(d @ occ, 90)]
    s = slab - slab.mean(axis=0)
    _, _, sv = np.linalg.svd(s[::7], full_matrices=False)
    along, across = s @ sv[0], s @ sv[1]
    pts = [slab[np.argmin(along)], slab[np.argmax(along)],
           slab[np.argmax(np.abs(across))]]
    r, timings["occlusal_plane"] = _t(
        "POST /occlusal-plane", client.post,
        f"/api/session/{sid}/occlusal-plane",
        json={"points": [[float(x) for x in p] for p in pts]})
    if r.status_code != 200:
        print(f"    FAILED {r.status_code}: {r.text[:400]}")
        return 2

    # --- segmentation -----------------------------------------------------
    print(f"\n[3] segmentation  (provider={args.provider})")
    r, timings["segment"] = _t(
        f"POST /segment?provider={args.provider}", client.post,
        f"/api/session/{sid}/segment?provider={args.provider}")
    if r.status_code != 200:
        print(f"    FAILED {r.status_code}: {r.text[:400]}")
        return 2
    seg = r.json()
    labels = np.asarray(STORE.get(sid, "labels"), np.int64)
    found = sorted(set(int(x) for x in np.unique(labels)) - {0})
    print(f"    {len(found)} teeth: {found}")
    if args.fdi not in found:
        print(f"    FDI {args.fdi} was not segmented. Nothing is substituted.")
        return 2

    # --- click-to-select --------------------------------------------------
    print(f"\n[4] click-to-select FDI {args.fdi}")
    own = np.flatnonzero(labels == args.fdi)
    seed = int(own[np.argmin(np.linalg.norm(v[own] - v[own].mean(axis=0),
                                            axis=1))])
    r, timings["select"] = _t("POST /select-tooth", client.post,
                              f"/api/session/{sid}/select-tooth",
                              json={"vertex_id": seed})
    if r.status_code != 200:
        print(f"    FAILED {r.status_code}: {r.text[:400]}")
        return 2
    sel = r.json()
    ids = np.asarray(sel["vertex_ids"], np.int64)
    purity = float((labels[ids] == args.fdi).mean()) * 100.0
    print(f"    {len(ids):,} vertices, purity {purity:.1f}%, "
          f"fdi {sel.get('fdi')}")

    # --- cut --------------------------------------------------------------
    print("\n[5] cut")
    pts_sel = v[ids]
    dd = pts_sel - pts_sel.mean(axis=0)
    axis = np.linalg.svd(dd, full_matrices=False)[2][0]
    proj = dd @ axis
    root = float(sel.get("root_length_mm") or 0.0) or 14.0
    r, timings["cut"] = _t("POST /cut", client.post,
                           f"/api/session/{sid}/cut",
                           json={"vertex_ids": ids.tolist(),
                                 "mesial_pt": pts_sel[proj.argmin()].tolist(),
                                 "distal_pt": pts_sel[proj.argmax()].tolist(),
                                 "root_length_mm": root})
    if r.status_code != 200:
        print(f"    FAILED {r.status_code}: {r.text[:600]}")
        return 2
    tid = r.json()["tooth_id"]
    print(f"    tooth {tid}, root {root}mm")

    # --- the prescription -------------------------------------------------
    print(f"\n[6] prescription  d_oa = +{args.extrusion}mm")
    r, timings["kinematics"] = _t(
        "POST /kinematics", client.post,
        f"/api/session/{sid}/tooth/{tid}/kinematics",
        json={"tip_deg": 0.0, "torque_deg": 0.0, "rotation_deg": 0.0,
              "d_md": 0.0, "d_bl": 0.0, "d_oa": float(args.extrusion)})
    if r.status_code != 200:
        print(f"    FAILED {r.status_code}: {r.text[:400]}")
        return 2

    # --- export, deformation ---------------------------------------------
    print("\n[7] export  construction=deformation")
    r, timings["export"] = _t(
        "POST /export/final", client.post,
        f"/api/session/{sid}/export/final",
        json={"construction": "deformation", "fmt": "manifest",
              "max_stages": max(1, int(args.stages))})
    timings["total"] = time.perf_counter() - t_all

    print("\n" + "=" * 74)
    print(f"HTTP {r.status_code}   X-Print-Ready: "
          f"{r.headers.get('X-Print-Ready', '(absent)')}")
    print("=" * 74)

    body = r.json()
    if r.status_code == 200:
        gate = body["manufacturing_gate"]
        print(f"\nVERDICT  {gate['verdict']}")
        for name, g in (gate.get("gates") or {}).items():
            print(f"  {'PASS' if g.get('ok') else 'FAIL'}  {name}")
        rc = 0 if gate.get("print_ready") else 1
    else:
        detail = body.get("detail", body)
        if isinstance(detail, dict):
            print(f"\nNAMED REFUSAL: {detail.get('error')}")
            print(f"  {detail.get('reason', '')}")
            for key in ("t0_self_intersection", "cast", "t0_file",
                        "failed_gates", "vertex_sets", "decision"):
                if key in detail:
                    print(f"\n  {key}:")
                    print("    " + json.dumps(detail[key], indent=2,
                                              default=str)[:1800]
                          .replace("\n", "\n    "))
        else:
            print(f"\n{json.dumps(detail, indent=2, default=str)[:1500]}")
        rc = 1

    print("\nTIMING")
    for k, val in timings.items():
        print(f"  {k:<20} {val:>9.2f}s")

    client.delete(f"/api/session/{sid}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
