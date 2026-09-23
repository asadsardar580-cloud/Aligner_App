"""REAL-SCAN MANUFACTURING REGRESSION, end to end, on a real de-identified scan.

WHY THIS IS A SCRIPT AND NOT A TEST. It runs against `case_lower.stl`, a real
de-identified mandibular scan that is deliberately NOT in version control
(`.gitignore` excludes every scan, because an arch mesh identifies a person the
way a fingerprint does). A pytest case that silently skips when the file is
absent reports PASS for a regression nobody ran, which is the failure mode this
project has corrected twice. So it is a script that must be invoked, prints
every number it measured, and exits non-zero on a failure.

WHAT IT EXERCISES, in the brief's order:

    upload -> occlusal plane -> tooth selection -> cut -> movement -> staging
    -> manufacturing reconstruction -> final STL -> reread -> all gates

THE TOOTH SELECTION IS SEEDED FROM THE SEGMENTATION LABELS, not from an
arbitrary coordinate. CLAUDE.md section 10 measured why: a crown taken straight
from the model's labels has a ragged, self-touching margin (232 rim points,
0.134mm minimum radius about its own centre) and falls back to a flat socket
floor, while the same tooth flooded geodesically as the app's wand does gives a
clean ring (213 points, 1.342mm) and cups properly. So the label's centroid
picks WHERE to click and the wand's own flood decides WHAT is selected - which
is exactly what a clinician does.

    python real_scan_regression.py [--stages 3] [--fdi 46]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np

import api_core
import arch_frame
import core_geometry as cg
import manufacturing as mfg
import stl_io
import segmentation_providers
import segmentation_diagnostics
from session_store import STORE

SCAN = "case_lower.stl"
LABELS = "case_lower.stl_output.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fail(msg):
    print(f"FAIL  {msg}")
    return False


#: The verdict strings `mfg.aggregate_print_gate` produces. Compared as data,
#: never re-derived here - the harness must not be able to disagree with the
#: gate about what the gate said.
VERDICT_READY = "PRINT READY"
VERDICT_NOT_READY = "NOT PRINT READY"


#: The suite's SKIP signal (`run_all_tests.SKIP_EXIT_CODE`), duplicated
#: here so this script has no import dependency on the runner.
SKIP_EXIT_CODE = 77

#: Set by `main()` from `--expect-exit`, read by the `__main__` wrapper.
#: A module global rather than a return value because it has to survive every
#: exit path in `main()`, including the early refusals that never build a
#: stage and therefore never produce a gate to compare against.
_EXPECT_EXIT = None


def exit_code_for_process(natural_code, expect_exit=None) -> int:
    """Map the run's natural exit code against a pinned expectation. PURE.

    WHY THIS EXISTS ALONGSIDE `exit_code_for`. That function pins a VERDICT,
    and a verdict only exists once a stage has been built and gated. On the
    real scan today, the collar path refuses BEFORE that: interface
    construction fails, `main()` returns 3, and no stage gate is ever
    produced - so `--expect "NOT PRINT READY"` cannot express the current
    state, because there is no verdict to compare.

    Measured on 2026-09-23, commit 85bad95, every configuration tried:

        --profile smoke (FDI 31, 32)   exit 3
            interface_construction_failed
            collar_top_unresolved_points 14, collar_top_rise_max_mm 4.338
        --fdi 45 --stages 1            exit 3
            interface_unbuildable_wall_too_thin
            crown_penetration_mm 3.719, local_cast_thickness_mm 5.2326

    So the expectation is pinned at the PROCESS level instead. The property
    the brief wants is preserved exactly: the suite stays green on today's
    known state, and ANY change in either direction - a refusal that moves to
    a different stage, or a run that starts succeeding - turns it red.

    Update it deliberately when the construction changes (Phase 3), never to
    make the suite pass.
    """
    if expect_exit is None:
        return natural_code
    # SKIP SURVIVES THE PIN. 77 means the scan is not on this machine, which
    # is normal - scans are excluded from version control. Mapping it to a
    # mismatch would turn "not verified here" into a failure and hide the
    # distinction the suite's SKIP state exists to preserve.
    if natural_code == SKIP_EXIT_CODE:
        return natural_code
    return 0 if natural_code == expect_exit else 1


def exit_code_for(stage_gates, expect=None) -> int:
    """The process exit code for a finished run. PURE - no I/O, no globals.

    WHY THIS IS EXTRACTED AND TESTED. Until Phase 0 this script printed the
    failed gates as a NOTE and returned 0, because `ok` was only ever cleared
    by open or non-manifold edges. A stage that was topologically clean and
    failed eight of sixteen manufacturing gates exited 0 and printed the
    success banner - so the one harness that drives real anatomy could not
    fail, and any CI job adopting it would have gone green on a model the
    system itself refuses to print.

    Two modes:

    * `expect=None` - the honest default. Every stage must be PRINT READY.
      An EMPTY list is a failure, not a pass: a run that produced no stage
      measured nothing, and "nothing to check" must never read as "clean".
    * `expect="<verdict>"` - a pinned expectation, used by the suite so that
      today's known state stays green while a change in EITHER direction turns
      it red. A stage that unexpectedly becomes PRINT READY is just as much a
      mismatch as one that regresses, because the expectation is the claim
      being tested.
    """
    verdicts = [
        (g or {}).get("verdict",
                      VERDICT_READY if (g or {}).get("print_ready") else VERDICT_NOT_READY)
        for g in (stage_gates or [])
    ]
    if not verdicts:
        return 1
    if expect is None:
        return 0 if all(v == VERDICT_READY for v in verdicts) else 1
    return 0 if all(v == expect for v in verdicts) else 1


def pick_seed_points(verts, labels, want_fdi=None):
    """One click point per labelled tooth, at that label's own centroid.

    Returned as (fdi, seed_vertex_id, n_labelled). The centroid of a label is
    not generally ON the surface, so the nearest labelled VERTEX to it is used -
    a click lands on the mesh.
    """
    labels = np.asarray(labels, np.int64)
    out = []
    for fdi in sorted(set(int(x) for x in np.unique(labels)) - {0}):
        if want_fdi is not None and fdi not in want_fdi:
            continue
        idx = np.where(labels == fdi)[0]
        if len(idx) < 200:
            continue
        pts = verts[idx]
        c = pts.mean(axis=0)
        seed = idx[np.argmin(np.linalg.norm(pts - c, axis=1))]
        out.append((fdi, int(seed), int(len(idx))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--fdi", type=int, nargs="*", default=None,
                    help="Run only these specific teeth (e.g. --fdi 45 46).")
    ap.add_argument("--max-teeth", type=int, default=None,
                    help="Limit the number of teeth extracted.")
    ap.add_argument("--profile", type=str, choices=["smoke", "real-risk", "full-arch"], default=None,
                    help="Run specific profile sets of teeth.")
    ap.add_argument("--expect-exit", type=int, default=None,
                    dest="expect_exit",
                    help="Pin the PROCESS exit code. Exit 0 only if the run "
                         "returns exactly this; anything else exits 1. Use "
                         "when the refusal happens before any stage gate "
                         "exists, so --expect has no verdict to compare.")
    ap.add_argument("--expect", type=str, default=None,
                    choices=[VERDICT_READY, VERDICT_NOT_READY],
                    help="Pin the expected verdict for EVERY stage. Exit 0 "
                         "only if every stage matches; a mismatch in either "
                         "direction exits 1. Without it, every stage must be "
                         "PRINT READY.")
    args = ap.parse_args()

    global _EXPECT_EXIT
    _EXPECT_EXIT = args.expect_exit

    if args.profile == "smoke":
        args.max_teeth = 2
        args.fdi = None
    elif args.profile == "real-risk":
        args.max_teeth = 4
        args.fdi = [31, 32, 45, 46]
    elif args.profile == "full-arch":
        args.max_teeth = 32
        args.fdi = None
    elif args.max_teeth is None:
        args.max_teeth = 2
        
    print("=" * 74)
    print("REAL-SCAN MANUFACTURING REGRESSION")
    print("=" * 74)

    if not os.path.exists(SCAN):
        print(f"REAL-SCAN MANUFACTURING REGRESSION = NOT EXECUTED")
        print(f"  reason: {SCAN} is not present in this working copy.")
        print(f"  Scans are excluded from version control by design; nothing "
              f"here is fabricated in its absence.")
        # 77 is the suite's SKIP signal (run_all_tests.py). It is deliberately
        # NOT 0: a regression nobody ran must never report PASS.
        return 77

    raw = open(SCAN, "rb").read()
    print(f"\n[1] INPUT")
    print(f"    file                {SCAN}")
    print(f"    bytes               {len(raw):,}")
    print(f"    sha256              {sha256(raw)}")

    t0 = time.perf_counter()
    verts, faces = stl_io.parse_stl_bytes(raw)
    print(f"    parsed              {len(verts):,} vertices / {len(faces):,} "
          f"triangles in {time.perf_counter() - t0:.2f}s")
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    print(f"    bounding box mm     "
          f"[{lo[0]:.2f}, {lo[1]:.2f}, {lo[2]:.2f}] .. "
          f"[{hi[0]:.2f}, {hi[1]:.2f}, {hi[2]:.2f}]")
    print(f"    dimensions mm       "
          f"{hi[0]-lo[0]:.2f} x {hi[1]-lo[1]:.2f} x {hi[2]-lo[2]:.2f}")

    # --- upload, exactly as the endpoint does ---------------------------
    t0 = time.perf_counter()
    v, f, cond = cg.condition_mesh(verts, faces)
    edges = cg.directed_edges(f)
    conc = cg.boundary_field(v, f, edges=edges)
    graph = cg.build_barrier_graph(v, f, conc, edges=edges)
    sid = STORE.create("lower")
    for key, val in (("verts", v), ("faces", f), ("edges", edges),
                     ("concavity", conc), ("graph", graph)):
        STORE.put(sid, key, val)
    print(f"\n[2] UPLOAD / CONDITIONING          {time.perf_counter() - t0:.2f}s")
    print(f"    conditioned         {len(v):,} vertices / {len(f):,} triangles")
    rep = cg.manifold_report(f)
    print(f"    scan health         open {rep['open_edges']}, "
          f"non-manifold {rep['nonmanifold_edges']}")

    ok = True
    try:
        # --- occlusal plane ---------------------------------------------
        # Three landmarks from the scan's own anatomy: the two posterior
        # extremes of the arch and the most anterior point, which is what the
        # three-click picker asks a clinician for.
        t0 = time.perf_counter()
        centroid = v.mean(axis=0)
        # 31 is the primary failure (undercut/overhang wall).
        # 32 is a tipped neighbor.
        # 45, 46 are contacting teeth (Blocker B).
        fdi_list = [31, 32, 45, 46]
        d = v - centroid
        _, _, vt = np.linalg.svd(d[::37], full_matrices=False)
        occ_axis = vt[2]
        slab = v[(d @ occ_axis) > np.percentile(d @ occ_axis, 90)]
        if (slab - centroid).shape[0] < 10:
            slab = v
        s = slab - slab.mean(axis=0)
        _, _, sv_ = np.linalg.svd(s[::7], full_matrices=False)
        along = s @ sv_[0]
        across = s @ sv_[1]
        pts = [slab[np.argmin(along)], slab[np.argmax(along)],
               slab[np.argmax(np.abs(across)) if abs(across).max() > 0 else 0]]
        frame = arch_frame.fit_occlusal_frame(
            list(map(float, pts[0])), list(map(float, pts[1])),
            list(map(float, pts[2])), centroid.tolist())
        STORE.put(sid, "arch_frame", frame)
        print(f"\n[3] OCCLUSAL PLANE                 "
              f"{time.perf_counter() - t0:.2f}s")
        print(f"    u_occ               "
              f"{np.round(frame['u_occ'], 4).tolist()}")

        # --- tooth selection, seeded from the labels --------------------
        if not os.path.exists(LABELS):
            print(f"\n[4] TOOTH SELECTION                NOT AVAILABLE")
            print(f"    {LABELS} is absent; a seed point cannot be derived "
                  f"from the segmentation without it.")
            return 2
        lab = json.load(open(LABELS))
        # THE LABELS INDEX NEITHER ARRAY THIS SCRIPT HOLDS. They are keyed
        # to the order the segmentation pipeline's own loader produced when
        # it welded the raw STL, and the count agrees with both of ours -
        # 94,848 - so nothing here could ever have noticed.
        raw_lab = np.asarray(lab["labels"], np.int64)
        # AND THEY DO NOT INDEX `verts` EITHER, WHICH IS WHERE THIS USED TO GO
        # WRONG. An STL has no vertex list - it is triangle soup - so every
        # reader invents an order when it welds. `stl_io.parse_stl_bytes`
        # welds with np.unique, which is LEXICOGRAPHIC; the segmentation
        # pipeline's loader keeps FIRST OCCURRENCE. Both give 94,848 vertices
        # for this scan, so remapping `verts -> v` by position moved the
        # labels between two arrays that were already in the same order and
        # preserved the error exactly. Measured, per-tooth box diagonal
        # median: 50.71mm remapped the old way, 13.97mm from the right source.
        src = segmentation_providers.first_occurrence_vertex_order(verts, faces)
        try:
            lab_arr, xfer = (segmentation_providers
                             .transfer_labels_by_position(
                                 src, raw_lab, v))
        except segmentation_providers.LabelTransferError as e:
            print(f"    LABEL TRANSFER FAILED: {e}")
            return 2
        print(f"    label transfer      {xfer['destination_vertices']:,} "
              f"vertices matched exactly (worst gap "
              f"{xfer['max_position_gap_mm']:.3g} mm), "
              f"{xfer['reordered_vertices']:,} reordered")
        rep = segmentation_diagnostics.label_report(v, f, lab_arr)
        print("\n[3b] SEGMENTATION SPATIAL DIAGNOSTIC")
        print("    " + segmentation_diagnostics.format_report(rep)
              .replace("\n", "\n    "))
        seeds = pick_seed_points(v, lab_arr, args.fdi)
        print(f"\n[4] TOOTH SELECTION")
        print(f"    labelled teeth      {len(seeds)} "
              f"(source: {LABELS}, the model's own PREDICTION - not ground truth)")

        # THE WAND'S AUTO TOLERANCE DOES NOT ISOLATE A TOOTH ON THIS SCAN, and
        # that is a measurement, not an excuse. Seeded at each label's own
        # centroid, `cut_guard.auto_tolerance` returned 25.00mm on every one of
        # the twelve labelled teeth and the flood took 26,000+ of 94,848
        # vertices - a quarter of the arch. Narrowing the tolerance by hand
        # does not fix it either: at 1.5mm the flood is 111 vertices and only
        # 37% of them belong to the tooth that was clicked, because
        # `snap_seed_to_ridge` moves the seed up to 3mm to the nearest
        # high-concavity point, which is the interdental sulcus - between two
        # teeth rather than on one. The first run of this script cut a "crown"
        # 35 x 28 x 14mm with 853 rim points, and the manufacturing layer
        # refused it by name with local_cast_thickness 0.0mm, which is the
        # correct answer to a selection that is not a tooth.
        #
        # So the selection falls back to the SEGMENTATION LABEL's own largest
        # connected component - tier 2 of the shipped segmentation fallback
        # (CLAUDE.md section 17), which is what the app itself uses when the
        # model's label is split. It is a real single tooth, and the report
        # says which route produced it.
        cut_ok = []
        for fdi, seed, n_lab in seeds:
            if len(cut_ok) >= args.max_teeth:
                break
            t0 = time.perf_counter()
            route = "wand"
            try:
                w = api_core.wand(sid, api_core.WandRequest(
                    point=v[seed].tolist()))
            except Exception as e:                        # noqa: BLE001
                print(f"    FDI {fdi:>2}  wand refused: {type(e).__name__}: {e}")
                continue
            ids = np.asarray(w["vertex_ids"], np.int64)
            label_ids = np.asarray(lab_arr == fdi).nonzero()[0]
            purity = (float(np.isin(ids, label_ids).mean())
                      if len(ids) else 0.0)
            if purity < 0.75 or len(ids) > 8000:
                # A MAJORITY OF CORNERS, not all three. Requiring every
                # corner to carry the label took the selection from a 4337
                # vertex tooth down to 19 vertices - the model's labels are
                # per-vertex and ragged at the margin, so almost no triangle
                # has three matching corners, and the "crown" that came out
                # enclosed 0.0mm3 and was correctly refused as a sliver.
                sel = np.zeros(len(v), bool)
                sel[label_ids] = True
                fmask = sel[f].sum(axis=1) >= 2
                fmask = cg.largest_face_component(f, fmask)
                ids = np.unique(f[fmask])
                route = "label component"
            print(f"    FDI {fdi:>2}  wand flood {len(w['vertex_ids']):>6} "
                  f"(tol {w['tolerance']:.2f}, {purity*100:.0f}% on this tooth) "
                  f"-> using {route}: {len(ids)} vertices")
            pts = v[ids]
            dd = pts - pts.mean(axis=0)
            axis = np.linalg.svd(dd, full_matrices=False)[2][0]
            proj = dd @ axis
            try:
                out = api_core.cut(sid, api_core.CutRequest(
                    vertex_ids=ids.tolist(),
                    mesial_pt=pts[proj.argmin()].tolist(),
                    distal_pt=pts[proj.argmax()].tolist(),
                    root_length_mm=float(
                        api_core.validation.ROOT_LENGTH_MIN
                        if False else 11.0)))
            except Exception as e:                        # noqa: BLE001
                msg = str(e)
                print(f"             -> CUT REFUSED: {msg[:160]}")
                continue
            rec = STORE.get(sid, f"tooth:{out['tooth_id']}")
            print(f"             -> crown {len(rec['cv']):>5} verts, "
                  f"rim {len(rec['socket_rim']):>4}, "
                  f"socket {(rec.get('socket_info') or {}).get('profile')}, "
                  f"{time.perf_counter() - t0:.2f}s")
            cut_ok.append((fdi, out["tooth_id"]))

        if not cut_ok:
            print("\n    NO CROWN COULD BE CUT FROM THE REAL SCAN.")
            print("    REAL-SCAN MANUFACTURING REGRESSION = NOT EXECUTED "
                  "beyond tooth selection.")
            return 3

        # --- movement ----------------------------------------------------
        print(f"\n[5] MOVEMENT")
        plan = {
            31: {"d_oa": 0.6, "tip_deg": 2.0},
            32: {"d_md": 0.4},
            45: {"d_md": 0.2},
            46: {"d_md": -0.2},
        }
        tooth_map = dict(cut_ok)
        for fdi, move in plan.items():
            if fdi in tooth_map:
                tid = tooth_map[fdi]
                api_core.kinematics(sid, tid, api_core.KinematicsRequest(**move))
                print(f"    FDI {fdi:>2}  {move}")

        # --- staging + manufacturing -------------------------------------
        print(f"\n[6] STAGING + MANUFACTURING RECONSTRUCTION")
        t0 = time.perf_counter()
        try:
            bundle = api_core.build_stage_bundle(
                sid, api_core.StageExportRequest(stages=args.stages))
        except Exception as e:                            # noqa: BLE001
            # REPORT WHICH STAGE WAS REACHED, and refuse to call the rest
            # executed. If the real-scan path cannot be completed, say exactly
            # where it stopped rather than presenting synthetic results as
            # real-scan evidence.
            detail = getattr(e, "detail", None) or str(e)
            print(f"    REFUSED after {time.perf_counter() - t0:.1f}s")
            if isinstance(detail, dict):
                print(f"    gate            {detail.get('gate')}")
                print(f"    error           {detail.get('error')}")
                d = detail.get("diagnostics") or {}
                for kk in sorted(d.keys()):
                    print(f"      {kk:<30} {d[kk]}")
            else:
                print(f"    detail          {str(detail)[:500]}")
            print()
            print("=" * 74)
            print("REAL-SCAN MANUFACTURING REGRESSION = EXECUTED THROUGH "
                  "TOOTH SELECTION ONLY")
            print("  reached      upload, conditioning, occlusal plane, tooth "
                  "selection, cut, movement, staging entry")
            print("  NOT reached  manufacturing reconstruction, final STL, "
                  "reread, gates")
            print("  cause        no single-tooth crown could be produced from "
                  "this scan headlessly; see [4] for the measured numbers.")
            print("  Nothing below this line is fabricated.")
            print("=" * 74)
            return 3
        man = bundle["manifest"]
        print(f"    stages              {man['stages']}  "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"    phase seconds       "
              f"{json.dumps(man.get('phase_seconds'))}")

        # --- final STL + reread + all gates ------------------------------
        print(f"\n[7] FINAL STL, REREAD, ALL GATES")
        zf = zipfile.ZipFile(bundle["buf"])
        stage_gates = []
        for meta in man["stage_files"]:
            blob = bundle["blobs"][meta["file"]]
            rv, rf = stl_io.parse_stl_bytes(blob)
            wv, wf, merged = cg.weld_vertices(rv, rf)
            rr = cg.manifold_report(wf)
            gate = meta["manufacturing_gate"]
            stage_gates.append(gate)
            print(f"    stage {meta['stage']:>2}  {meta['file']}")
            print(f"        sha256          {sha256(blob)}")
            print(f"        bytes/tris      {len(blob):,} / {meta['triangles']:,}")
            print(f"        volume mm3      {meta['volume_mm3']:.3f}")
            print(f"        reread+weld     open {rr['open_edges']}, "
                  f"non-manifold {rr['nonmanifold_edges']}, merged {merged}")
            print(f"        self-touch      "
                  f"{meta['self_touch']['coincident_position_groups']} groups, "
                  f"{meta['self_touch']['zero_area_triangles']} zero-area tris")
            print(f"        bodies/comp     {meta['manifold_bodies']} / "
                  f"{meta['stl_validation']['connected_components']}")
            print(f"        boolean gate    {meta['stl_validation']['verdict']}")
            print(f"        AGGREGATE GATE  {gate['verdict']}"
                  + (f"  failed: {gate['failed_gates']}"
                     if gate["failed_gates"] else ""))
            fid = meta["cast_fidelity"]
            if fid.get("measured"):
                print(f"        cast fidelity   "
                      f"final->orig max {fid['final_to_original']['max_mm']}mm, "
                      f"orig->final max {fid['original_to_final']['max_mm']}mm")
            print(f"        ROI             "
                  f"compliant={meta['roi_compliance']['compliant']}, "
                  f"outside {meta['roi_compliance'].get('modified_points_outside_envelope')}")
            print(f"        synthetic area  "
                  f"{meta['synthetic_exposure']['exposed_synthetic_fraction']*100:.2f}%"
                  f" of the surface")
            for row in meta["interfaces"]:
                o = row.get("old_site") or {}
                tq = row.get("transition") or {}
                print(f"        FDI {row.get('fdi')}  mode "
                      f"{row.get('interface_mode')}, "
                      f"connector {row.get('connector_volume_mm3')}mm3, "
                      f"rim_separation {row.get('rim_separation_max_mm')}mm, "
                      f"continuity {(row.get('continuity') or {}).get('covered_fraction')}, "
                      f"ledge {tq.get(mfg.KEY_LOOKS_LIKE_A_LEDGE)}, "
                      f"old-site assessable {o.get('assessable')}")
            if rr["open_edges"] or rr["nonmanifold_edges"]:
                ok = _fail(f"stage {meta['stage']} rereads with "
                           f"{rr['open_edges']} open / "
                           f"{rr['nonmanifold_edges']} non-manifold edges")
            if not gate["print_ready"]:
                print(f"        NOTE: stage {meta['stage']} is NOT PRINT READY "
                      f"- {gate['failed_gates']}")
                # AND IT FAILS THE RUN. The NOTE used to be the whole response.
                ok = _fail(f"stage {meta['stage']} is NOT PRINT READY "
                           f"- {gate['failed_gates']}")

        print(f"\n    manifest entries    {len(zf.namelist())}")

        code = exit_code_for(stage_gates, expect=args.expect)
        if args.expect:
            print(f"\n    expectation         every stage == {args.expect!r}")
            print(f"    observed            "
                  f"{[g.get('verdict') for g in stage_gates]}")
        # A topology failure recorded by `_fail` above must still be able to
        # fail the run even when the verdicts match the expectation.
        if not ok and code == 0 and args.expect is None:
            code = 1
        print("=" * 74)
        print("REAL-SCAN MANUFACTURING REGRESSION = EXECUTED"
              if code == 0 else
              "REAL-SCAN MANUFACTURING REGRESSION = EXECUTED, FAILURES ABOVE")
        print("=" * 74)
        return code
    finally:
        api_core.close_session(sid)


if __name__ == "__main__":
    _code = main()
    if _EXPECT_EXIT is not None:
        _mapped = exit_code_for_process(_code, _EXPECT_EXIT)
        print(f"\n    expected exit code  {_EXPECT_EXIT}")
        print(f"    observed exit code  {_code}"
              + ("" if _mapped == 0 else "   <-- MISMATCH"))
        _code = _mapped
    sys.exit(_code)
