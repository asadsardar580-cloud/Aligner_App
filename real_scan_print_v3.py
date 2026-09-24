"""TASK 2 step 6 - the real mandible, through the real API path, four cases.

    a. T0 export, no movement                     -> must be PRINT READY
    b. FDI 45 extrusion 0.25 mm                    (1 stage asked)
    c. FDI 43 distal 1.0 mm, into the space by 44  (5 stages asked)
    d. FDI 41 labial 0.5 mm                        (1 stage asked)

upload -> occlusal plane -> segmentation -> click-to-select -> cut ->
prescription -> POST /export/stages construction=deformation, driven over
HTTP through TestClient - the product's own chain, one fresh session per
case. Every stage STL and every manifest is written to exports/real_v3/<case>/
(gitignored: patient-derived), plus summary.json.

THE STAGE COUNT IS THE APP'S, NOT THIS SCRIPT'S. `cg.staging_estimate`
divides by 0.25 mm per stage, so 1.0 mm is 4 stages and 0.5 mm is 2. The
counts the task asked for are recorded beside the counts built; the staging
rule is a clinical default and is not overridden here.

DIRECTION IS CHECKED FROM ANATOMY, NOT FROM THE FRAME THAT PRODUCED IT. The
final stage's own matrix (from its manifest) is applied to the tooth's
labelled vertices, and the result is measured against the NEIGHBOURS' and the
ARCH's labelled geometry:
  b  the crown's displacement along the arch's occlusal axis;
  c  exact point-to-triangle gap to 44 must SHRINK by about 1.0 mm and the
     gap to 42 must GROW;
  d  the labial surface must move AWAY from the tongue side - the centroid of
     every labelled tooth, which lies inside the horseshoe.
"Distal" is prescribed the way the product defines it: +d_md runs along the
tooth frame's u_md, and the frame reports `u_md_points_distal` (u_BL is
pinned to anatomy, so u_MD's sign depends on the quadrant). The script PRINTS
which sign that gave. If any movement goes the wrong way the run STOPS with
exit 4 and says so - no sign is flipped to make it pass.

    .venv\\Scripts\\python.exe real_scan_print_v3.py [--cases abcd]

Exit 0 = every case PRINT READY and every direction right; 1 = at least one
NOT PRINT READY (named); 2 = a case could not reach the gate; 4 = a movement
went the wrong way; 77 = the scan is not on this machine (not verified).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile

import numpy as np

SCAN = "case_lower.stl"
OUT = os.path.join("exports", "real_v3")
SKIP_EXIT_CODE = 77

CASES = {
    "a": {"title": "T0 export, no movement", "fdi": None, "asked_stages": 0},
    "b": {"title": "FDI 45 extrusion 0.25 mm", "fdi": 45,
          "move": ("d_oa", 0.25), "asked_stages": 1},
    "c": {"title": "FDI 43 distal 1.0 mm into the space by 44", "fdi": 43,
          "move": ("distal", 1.0), "asked_stages": 5},
    "d": {"title": "FDI 41 labial 0.5 mm", "fdi": 41,
          "move": ("d_bl", 0.5), "asked_stages": 1},
}


class Stop(Exception):
    """A case could not reach the gate. Carries the exit code."""

    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------------------
# Anatomy - pure, and independent of the tooth frame
# ---------------------------------------------------------------------------

def mesial_distal_neighbours(fdi: int, present) -> tuple:
    """(mesial FDI, distal FDI) as present, or None for a side with none.
    Mesial is toward the midline: for FDI qn the mesial neighbour is q(n-1),
    and for a central incisor it is the other central across the midline."""
    q, n = divmod(int(fdi), 10)
    present = {int(x) for x in present}
    other = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}[q]
    mesial = [q * 10 + k for k in range(n - 1, 0, -1)] + \
        [other * 10 + k for k in range(1, 9)]
    distal = [q * 10 + k for k in range(n + 1, 9)]
    m = next((x for x in mesial if x in present), None)
    d = next((x for x in distal if x in present), None)
    return m, d


def contact_clicks(crown, mesial_ref, distal_ref):
    """(mesial_pt, distal_pt) on the crown, as a clinician clicks them: the
    crown point nearest each neighbour. With a neighbour missing (the last
    tooth in the arch, an extraction space), that click goes to the crown's
    far end AWAY from the neighbour that is there. (None, None) with no
    neighbour at all - the side would be a guess."""
    P = np.asarray(crown, float)
    c = P.mean(axis=0)

    def nearest(ref):
        return P[np.argmin(np.linalg.norm(P - ref, axis=1))]

    def farthest_from(ref):
        u = c - np.asarray(ref, float)
        return P[np.argmax(P @ (u / np.linalg.norm(u)))]

    if mesial_ref is None and distal_ref is None:
        return None, None
    m = nearest(mesial_ref) if mesial_ref is not None else farthest_from(distal_ref)
    d = nearest(distal_ref) if distal_ref is not None else farthest_from(mesial_ref)
    return m, d


def apply4(P, M):
    P = np.asarray(P, float)
    return (np.column_stack([P, np.ones(len(P))]) @ np.asarray(M, float).T)[:, :3]


def surface_distance(points, verts, faces):
    """Exact point-to-triangle distance (Open3D), per point."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(verts, np.float32)),
                     o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return sc.compute_distance(o3d.core.Tensor(
        np.asarray(points, np.float32))).numpy().astype(float)


def check_extrusion(crown0, crown1, u_occ, expect_mm):
    u = np.asarray(u_occ, float) / np.linalg.norm(u_occ)
    d = float((crown1.mean(axis=0) - crown0.mean(axis=0)) @ u)
    ok = d > 0.5 * expect_mm
    return ok, {"crown_displacement_along_occlusal_mm": round(d, 4),
                "expected_mm": expect_mm,
                "reads": "extruded (toward the occlusal plane)" if d > 0
                         else "INTRUDED"}


def check_distal(crown0, crown1, distal_patch, mesial_patch, expect_mm):
    g_d0 = float(surface_distance(crown0, *distal_patch).min())
    g_d1 = float(surface_distance(crown1, *distal_patch).min())
    out = {"gap_to_distal_neighbour_before_mm": round(g_d0, 4),
           "gap_to_distal_neighbour_after_mm": round(g_d1, 4),
           "distal_gap_shrank_by_mm": round(g_d0 - g_d1, 4),
           "expected_shrink_mm": expect_mm}
    ok = (g_d0 - g_d1) > 0.5 * expect_mm
    if mesial_patch is not None:
        g_m0 = float(surface_distance(crown0, *mesial_patch).min())
        g_m1 = float(surface_distance(crown1, *mesial_patch).min())
        out.update({"gap_to_mesial_neighbour_before_mm": round(g_m0, 4),
                    "gap_to_mesial_neighbour_after_mm": round(g_m1, 4)})
        ok = ok and g_m1 > g_m0
    out["reads"] = "moved distally" if ok else "DID NOT MOVE DISTALLY"
    return ok, out


def check_labial(crown0, crown1, tongue_ref, u_occ, expect_mm):
    u = np.asarray(u_occ, float) / np.linalg.norm(u_occ)

    def flat(x):
        x = np.asarray(x, float)
        return x - np.outer(x @ u, u) if x.ndim == 2 else x - (x @ u) * u

    c0 = crown0.mean(axis=0)
    lab = flat(c0 - tongue_ref)
    lab /= np.linalg.norm(lab)
    disp = float((crown1.mean(axis=0) - c0) @ lab)
    face = ((crown0 - c0) @ lab) > 0                 # the labial half
    r0 = float(np.linalg.norm(flat(crown0[face] - tongue_ref), axis=1).mean())
    r1 = float(np.linalg.norm(flat(crown1[face] - tongue_ref), axis=1).mean())
    ok = disp > 0.5 * expect_mm and r1 > r0
    return ok, {"crown_displacement_labial_mm": round(disp, 4),
                "labial_surface_distance_from_tongue_side_before_mm": round(r0, 4),
                "labial_surface_distance_from_tongue_side_after_mm": round(r1, 4),
                "expected_mm": expect_mm,
                "reads": ("moved labially (away from the tongue)" if ok
                          else "DID NOT MOVE LABIALLY")}


def _worst(values):
    """The largest value, or NaN when ANY is missing - an unmeasured stage
    must not be hidden behind a measured one."""
    vals = list(values)
    if not vals or any(x is None for x in vals):
        return float("nan")
    return float(max(vals))


def own_patch(F, labels, fdi):
    """The faces one FDI wholly owns, compacted to their own vertices."""
    m = (labels[F] == int(fdi)).all(axis=1)
    faces = F[m]
    ids = np.unique(faces)
    remap = -np.ones(int(F.max()) + 1, np.int64)
    remap[ids] = np.arange(len(ids))
    return ids, remap[faces]


# ---------------------------------------------------------------------------
# One case, through the API
# ---------------------------------------------------------------------------

def _post(client, url, label, **kw):
    t0 = time.perf_counter()
    r = client.post(url, **kw)
    dt = time.perf_counter() - t0
    print(f"    {label:<40} {r.status_code}  {dt:8.2f}s")
    return r, dt


def run_case(client, key, raw, args, out_root):
    from api_core import STORE

    spec = CASES[key]
    print("\n" + "=" * 78)
    print(f"CASE {key}: {spec['title']}")
    print("=" * 78)
    t_case = time.perf_counter()
    row = {"case": key, "title": spec["title"],
           "asked_stages": spec["asked_stages"]}
    sid = None
    try:
        r, _ = _post(client, "/api/session", "upload",
                     files={"file": (os.path.basename(args.scan), raw,
                                     "model/stl")},
                     data={"arch": "lower"})
        if r.status_code != 200:
            raise Stop(f"upload {r.status_code}: {r.text[:300]}")
        sid = r.json()["session_id"]
        v = STORE.require(sid, "verts")
        F = np.asarray(STORE.require(sid, "faces"), np.int64)

        # occlusal plane - the same landmark picking real_scan_deformation uses
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
        r, _ = _post(client, f"/api/session/{sid}/occlusal-plane",
                     "occlusal plane",
                     json={"points": [[float(x) for x in p] for p in pts]})
        if r.status_code != 200:
            raise Stop(f"occlusal plane {r.status_code}: {r.text[:300]}")
        af = STORE.get(sid, "arch_frame")

        r, _ = _post(client, f"/api/session/{sid}/segment?provider={args.provider}",
                     f"segment ({args.provider})")
        if r.status_code != 200:
            raise Stop(f"segment {r.status_code}: {r.text[:300]}")
        labels = np.asarray(STORE.get(sid, "labels"), np.int64)
        present = sorted(set(int(x) for x in np.unique(labels)) - {0})
        print(f"    teeth: {present}")

        tid, frame = None, None
        if spec["fdi"]:
            fdi = spec["fdi"]
            if fdi not in present:
                raise Stop(f"FDI {fdi} was not segmented; nothing is "
                           f"substituted")
            own = np.flatnonzero(labels == fdi)
            seed = int(own[np.argmin(np.linalg.norm(
                v[own] - v[own].mean(axis=0), axis=1))])
            r, _ = _post(client, f"/api/session/{sid}/select-tooth",
                         f"select FDI {fdi}", json={"vertex_id": seed})
            if r.status_code != 200:
                raise Stop(f"select-tooth {r.status_code}: {r.text[:300]}")
            ids = np.asarray(r.json()["vertex_ids"], np.int64)

            # The two contact clicks, placed as a clinician would: the crown
            # point nearest the mesial neighbour, and the one nearest the
            # distal neighbour.
            m_fdi, d_fdi = mesial_distal_neighbours(fdi, present)
            mesial_pt, distal_pt = contact_clicks(
                v[ids],
                None if m_fdi is None else v[labels == m_fdi].mean(axis=0),
                None if d_fdi is None else v[labels == d_fdi].mean(axis=0))
            if mesial_pt is None:
                raise Stop(f"FDI {fdi} has no neighbour on either side to "
                           f"place its contact clicks against")
            root = 14.0
            r, _ = _post(client, f"/api/session/{sid}/cut", "cut",
                         json={"vertex_ids": ids.tolist(),
                               "mesial_pt": mesial_pt.tolist(),
                               "distal_pt": distal_pt.tolist(),
                               "root_length_mm": root})
            if r.status_code != 200:
                raise Stop(f"cut {r.status_code}: {r.text[:400]}")
            tid = r.json()["tooth_id"]
            frame = STORE.get(sid, f"tooth:{tid}")["frame"]

            axis, mm = spec["move"]
            presc = {"tip_deg": 0.0, "torque_deg": 0.0, "rotation_deg": 0.0,
                     "d_md": 0.0, "d_bl": 0.0, "d_oa": 0.0}
            if axis == "distal":
                pdist = frame.get("u_md_points_distal")
                if pdist is None:
                    raise Stop("the frame cannot say which way u_md points, "
                               "so 'distal' has no sign")
                presc["d_md"] = mm if pdist else -mm
                print(f"    frame u_md_points_distal={pdist} -> distal "
                      f"{mm} mm is d_md = {presc['d_md']:+.2f}")
            else:
                presc[axis] = mm
            row["prescription"] = presc
            r, _ = _post(client, f"/api/session/{sid}/tooth/{tid}/kinematics",
                         "kinematics", json=presc)
            if r.status_code != 200:
                raise Stop(f"kinematics {r.status_code}: {r.text[:300]}")

        # export - every stage, each solidified and gated
        body = {"construction": "deformation", "max_stages": args.max_stages}
        r, t_export = _post(client, f"/api/session/{sid}/export/stages",
                            "export/stages (deformation)", json=body)
        if r.status_code != 200:
            raise Stop(f"export {r.status_code}: {r.text[:600]}")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        out_dir = os.path.join(out_root, key)
        os.makedirs(out_dir, exist_ok=True)
        for name in z.namelist():
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(z.read(name))
        man = json.loads(z.read("manifest.json"))
        for sf in man["stage_files"]:
            with open(os.path.join(out_dir, f"stage_{sf['stage']:02d}_manifest"
                                            f".json"), "w") as fh:
                json.dump(sf, fh, indent=2, default=str)

        stages = man["stage_files"]
        last = stages[-1]
        row.update({
            "built_stages": [s["stage"] for s in stages],
            "verdicts": {s["stage"]: s["manufacturing_gate"]["verdict"]
                         for s in stages},
            "verdict": ("PRINT READY" if all(
                s["manufacturing_gate"]["print_ready"] for s in stages)
                else "NOT PRINT READY"),
            "failures": {s["stage"]: s.get("failures") for s in stages
                         if not s["manufacturing_gate"]["print_ready"]},
            # the WORST stage, so the table never hides one behind another
            "crown_p95_mm": _worst(s["crown_deviation"]["p95_mm"] for s in stages),
            "crown_max_mm": _worst(s["crown_deviation"]["max_mm"] for s in stages),
            "triangles": last["triangles"],
            "export_seconds": round(t_export, 1),
            "contacts": last.get("contacts"),
            "height_warning": last["manufacturing_gate"]["advisory"].get(
                "model_height_over_limit"),
            "model_height_mm": last["manufacturing_gate"]["advisory"].get(
                "model_height_mm"),
            "files": sorted(os.listdir(out_dir)),
        })
        for s in stages:
            print(f"    stage {s['stage']:>2}: {s['manufacturing_gate']['verdict']:<16}"
                  f" tris {s['triangles']}, crown p95 "
                  f"{s['crown_deviation']['p95_mm']} / max "
                  f"{s['crown_deviation']['max_mm']} mm")
            for line in s.get("failures") or []:
                print(f"        FAIL {line[:160]}")

        # direction - from anatomy, on the final stage's own matrix
        row["direction"] = "n/a (no movement)"
        if spec["fdi"]:
            M = last["manifest"]["tooth_matrices"][tid]
            crown_ids = np.flatnonzero(labels == spec["fdi"])
            c0 = v[crown_ids]
            c1 = apply4(c0, M)
            axis, mm = spec["move"]
            # The INTENT is anatomical (occlusal / distal / labial) and so is
            # the check: it compares against the magnitude asked for, never
            # against the sign that was sent - a wrong sign must read wrong.
            mm = abs(mm)
            if axis == "d_oa":
                ok, det = check_extrusion(c0, c1, af["u_occ"], mm)
            elif axis == "distal":
                m_fdi, d_fdi = mesial_distal_neighbours(spec["fdi"], present)
                did, dfaces = own_patch(F, labels, d_fdi)
                mid, mfaces = own_patch(F, labels, m_fdi)
                ok, det = check_distal(c0, c1, (v[did], dfaces),
                                       (v[mid], mfaces), mm)
            else:
                teeth = labels > 0
                ok, det = check_labial(c0, c1, v[teeth].mean(axis=0),
                                       af["u_occ"], mm)
            row["direction"] = "OK" if ok else "WRONG"
            row["direction_detail"] = det
            print(f"    direction: {det['reads']}  {json.dumps(det)}")
            if not ok:
                raise Stop(f"CASE {key}: THE MOVEMENT WENT THE WRONG WAY - "
                           f"{det['reads']}. Stopping; no sign was flipped.",
                           code=4)
        row["status"] = "gated"
    except Stop as e:
        row["status"] = f"stopped: {e}"
        row["exit"] = e.code
        print(f"\n    STOPPED: {e}")
    finally:
        row["case_seconds"] = round(time.perf_counter() - t_case, 1)
        if sid:
            client.delete(f"/api/session/{sid}")
    return row


def table(rows) -> str:
    head = ("| case | verdict | crown dev p95 / max (mm) | triangles | "
            "seconds | direction |")
    out = [head, "|---|---|---|---|---|---|"]
    for r in rows:
        if "verdict" not in r:
            out.append(f"| {r['case']} {r['title']} | {r.get('status')} | - | - "
                       f"| {r.get('case_seconds')} | - |")
            continue
        det = r.get("direction_detail") or {}
        dtext = r["direction"] + (f" ({det.get('reads')})" if det else "")
        out.append(
            f"| {r['case']} {r['title']} (stages {r['built_stages']}, asked "
            f"{r['asked_stages']}) | {r['verdict']} | {r['crown_p95_mm']:.4f} / "
            f"{r['crown_max_mm']:.4f} | {r['triangles']:,} | "
            f"{r['case_seconds']} | {dtext} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=SCAN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cases", default="abcd")
    ap.add_argument("--provider", default="crosstooth")
    ap.add_argument("--max-stages", type=int, default=30)
    args = ap.parse_args(argv)

    try:
        raw = open(args.scan, "rb").read()
    except FileNotFoundError:
        print(f"NOT VERIFIED: {args.scan} is not here. Nothing synthetic is "
              f"substituted for a real scan.")
        return SKIP_EXIT_CODE

    from fastapi.testclient import TestClient
    import api_core
    import print_solid as ps
    client = TestClient(api_core.app)
    os.makedirs(args.out, exist_ok=True)
    print(f"scan {args.scan} ({len(raw) / 1e6:.1f} MB), out {args.out}, "
          f"voxel {ps.VOXEL_SIZE_MM} mm, meshlib {ps.meshlib_version()}")

    rows, code = [], 0
    for key in args.cases:
        row = run_case(client, key, raw, args, args.out)
        rows.append(row)
        if row.get("exit") == 4:
            code = 4
            break
        if row.get("exit"):
            code = max(code, row["exit"])
        elif row.get("verdict") != "PRINT READY":
            code = max(code, 1)

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print("\n" + table(rows))
    print(f"\nsummary: {os.path.join(args.out, 'summary.json')}   exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
