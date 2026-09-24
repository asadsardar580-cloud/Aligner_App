#!/usr/bin/env python3
"""The canonical suite. Needs only numpy + scipy - no display, no Qt.

WHAT CHANGED IN PHASE 0.2, AND WHY. This runner decided PASS from an exit code
alone, and a missing entry file printed SKIP while the suite still reported
success. Three separate ways to report a pass for something that never ran:

  * a missing file -> SKIP, suite green. A deleted or renamed test disappeared
    silently.
  * any exit code of 0 -> PASS, including a file that executed no assertion at
    all. A pytest-style module without a `__main__` block runs as a script,
    does nothing, and exits 0.
  * a script that printed its own failures and returned 0 anyway - which is
    exactly what `real_scan_regression.py` did until 0.1.

So: a missing entry is a FAIL. SKIP is a deliberate signal, exit code 77, and
prints as "SKIP (NOT VERIFIED)" because a skip is an absence of evidence, not
evidence of absence. And every entry is statically checked for something that
can actually fail before it is run at all.
"""
import ast
import os
import subprocess
import sys

#: An entry that could not verify anything - the real scan without the scan
#: file, for instance. Deliberately not 0, and deliberately not a failure.
SKIP_EXIT_CODE = 77

#: (name, path) or (name, path, [args])
TESTS = [
    ("core geometry",        "test_core_geometry.py"),
    ("kinematics frame",     "test_kinematics_frame.py"),
    ("socket cup",           "test_socket_cup.py"),
    ("cast base",            "test_cast_base.py"),
    ("staging export",       "test_staging_export.py"),
    ("occlusal collision",   "test_occlusal_collision.py"),
    ("cut endpoint",         "test_cut_endpoint.py"),
    ("loop failure guard",   "test_loop_failure.py"),
    ("region growing",       "test_region_grow.py"),
    ("UI dataflow",          "test_ui_dataflow.py"),
    ("worker path",          "test_worker_path.py"),
    ("interproximal study",  "test_interproximal.py"),
    ("segmentation 2-5",     os.path.join("tooth_segmentation","tests","test_stages_2_to_5.py")),
    ("label adapter",        os.path.join("tooth_segmentation","tests","test_label_adapter.py")),
    ("api core",             "test_api_core.py"),
    ("case hydration",       "test_hydration.py"),
    ("clinical validation",  "test_validation.py"),
    ("domain + case file",   "test_domain.py"),
    ("space analysis/IPR",   "test_space_analysis.py"),
    ("segmentation review",  "test_segmentation_review.py"),
    ("attachments + CBCT",   "test_attachments.py"),
    ("benchmark + fallback", "test_benchmark_segmentation.py"),
    ("scan cache + restore", "test_scan_cache.py"),
    ("antagonist collision", "test_antagonist_collision.py"),
    ("clinical safety",      "test_clinical_safety.py"),
    ("fail-safes",           "test_failsafes.py"),
    ("telemetry + PHI",      "test_telemetry.py"),
    ("manufacturing iface",  "test_manufacturing_interface.py"),
    ("manufacturing matrix", "test_manufacturing_matrix.py"),
    ("segmentation mapping", "test_segmentation_mapping.py"),
    ("crosstooth provider",  "test_crosstooth_adapter.py"),
    ("click-to-select",      "test_click_to_select.py"),
    ("pointops shim",        "verify_pointops.py"),
    ("face order",           "test_face_order.py"),
    ("auto colour",          "test_auto_color.py"),
    ("bleed fix",            "test_bleed_fix.py"),
    ("incisal edge",         "test_incisal_edge.py"),
    ("conditioning",         "test_conditioning.py"),
    ("precompute arch",      "test_precompute.py"),
    ("clinical segmentation","test_clinical_segmentation.py"),
    ("md caliper",           "test_caliper.py"),
    ("constrained merge",    "test_constrained_merge.py"),
    ("real-scan harness",    "test_real_scan_harness.py"),
    ("suite runner",         "test_suite_runner.py"),
    ("solid bodies/voids",   "test_solid_bodies.py"),
    ("distance failure",     "test_distance_failure.py"),
    ("self-intersect gate",  "test_self_intersection_gate.py"),
    ("export confinement",   "test_export_confinement.py"),
    ("export archive build",  "test_build_export.py"),
    ("self-intersection",    "test_self_intersection.py"),
    ("deformation construction", "test_deform_construction.py"),
    ("stage matrix shared",  "test_stage_matrix_shared.py"),
    ("deformation vertex sets", "test_deformation_vertex_sets.py"),
    ("deformation export API", "test_export_deformation_api.py"),
    ("tooth/gum label bands", "test_label_bands.py"),
    ("print solid (voxel)",  "test_print_solid.py"),
    ("print solid gate",     "test_solid_gate.py"),
    ("print gate v3",        "test_print_gate_v3.py"),
    ("contacts / IPR gate",  "test_contact_gate.py"),
    # THE REAL SCAN, AND THE EXPECTATION IS THE POINT. This records today's
    # KNOWN state of the collar path, so the suite stays green on it while ANY
    # change in either direction turns it red. Update it deliberately when the
    # construction changes (Phase 3), never to make the suite pass.
    #
    # PINNED AT THE PROCESS LEVEL, NOT THE VERDICT LEVEL, and that is a
    # measurement rather than a preference. AGENT_BRIEF 0.1 expects
    # `--expect "NOT PRINT READY"`, which compares a per-stage verdict - but
    # on this scan the collar path refuses BEFORE a stage is ever built, so
    # there is no verdict to compare. Measured 2026-09-23:
    #
    #   --profile smoke (FDI 31, 32)  exit 3  interface_construction_failed
    #                                         collar_top_unresolved_points 14
    #   --fdi 45 --stages 1           exit 3  interface_unbuildable_wall_too_thin
    #                                         crown_penetration_mm 3.719
    #
    # Exits 77 when the scan is absent, which --expect-exit deliberately
    # passes through as SKIP rather than treating as a mismatch.
    ("real scan (local)",    "real_scan_regression.py",
     ["--profile", "smoke", "--expect-exit", "3"]),
]


def _entry(row):
    """(name, path, args) from a 2- or 3-tuple."""
    if len(row) == 3:
        return row[0], row[1], list(row[2])
    return row[0], row[1], []


def can_fail(path):
    """Can running this file as a script actually assert anything?

    Returns (ok, reason). A pytest-style module with no `__main__` block runs
    as a script, collects nothing, executes nothing and exits 0 - which this
    runner would have reported as PASS. Checked statically so that a file
    added in the future cannot quietly join the suite without teeth.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError) as e:
        return False, f"cannot parse: {type(e).__name__}: {e}"

    for node in tree.body:
        if isinstance(node, ast.If):
            # if __name__ == "__main__":
            src = ast.dump(node.test)
            if "__name__" in src and "__main__" in src:
                return True, "__main__ block"

    # A module-level assert counts whether it sits at the top level or inside a
    # top-level `for` / `if` / `with` / `try` - all of those execute when the
    # file is run as a script. `test_precompute.py` asserts inside a top-level
    # loop and is a real test. What does NOT count is an assert inside a `def`
    # or a `class`: pytest would call it, running the file as a script would
    # not, and that is precisely the toothless case this guard exists for.
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assert):
            return True, "module-level assert"
        stack.extend(ast.iter_child_nodes(node))

    return False, ("no __main__ block and no assert that runs at module level "
                   "- executing this file as a script would run no test and "
                   "exit 0")


def main():
    print("=" * 66)
    print("CLINICAL MICRO-PLANNER - TEST SUITE")
    print("=" * 66)

    # Force UTF-8 on the children. A Windows console here reports cp1256, and a
    # test that printed "z-bar" or "mm^3" died with UnicodeEncodeError AFTER
    # every assertion in it had passed - a green test file reported as a
    # failure, which is worse than either outcome on its own. The suite must not
    # depend on the operator's codepage.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    passed, skipped, failed = [], [], []
    for row in TESTS:
        name, path, args = _entry(row)

        if not os.path.exists(path):
            print(f"  FAIL  {name:<26} ({path} missing)")
            failed.append(name)
            continue

        ok, reason = can_fail(path)
        if not ok:
            print(f"  FAIL  {name:<26} (toothless entry)")
            print(f"        {reason}")
            failed.append(name)
            continue

        r = subprocess.run([sys.executable, path, *args],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()]

        if r.returncode == 0:
            print(f"  PASS  {name:<26} {tail[-1][:56] if tail else ''}")
            passed.append(name)
        elif r.returncode == SKIP_EXIT_CODE:
            print(f"  SKIP (NOT VERIFIED)  {name:<26}")
            for line in tail[-2:]:
                print(f"        {line[:90]}")
            skipped.append(name)
        else:
            print(f"  FAIL  {name:<26} (exit {r.returncode})")
            err = (r.stderr.strip().splitlines() or tail or ["?"])[-1]
            print(f"        {err[:100]}")
            failed.append(name)

    print("=" * 66)
    print(f"{len(passed)} PASS / {len(skipped)} SKIP / {len(failed)} FAIL")
    if skipped:
        print(f"  NOT VERIFIED: {', '.join(skipped)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("ALL EXECUTED TESTS PASSED")
    print("\nGeometry verified. The production UI is the React frontend.")
    print("Run: start_frontend.bat")


if __name__ == "__main__":
    main()
