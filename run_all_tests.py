#!/usr/bin/env python3
"""Runs every headless test. Needs only numpy + scipy - no display, no Qt."""
import subprocess, sys, os

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
    ("face order",           "test_face_order.py"),
    ("auto colour",          "test_auto_color.py"),
    ("bleed fix",            "test_bleed_fix.py"),
    ("incisal edge",         "test_incisal_edge.py"),
    ("conditioning",         "test_conditioning.py"),
    ("precompute arch",      "test_precompute.py"),
    ("clinical segmentation","test_clinical_segmentation.py"),
    ("md caliper",           "test_caliper.py"),
    ("constrained merge",    "test_constrained_merge.py"),
]

def main():
    print("=" * 66)
    print("CLINICAL MICRO-PLANNER - TEST SUITE")
    print("=" * 66)

    # Force UTF-8 on the children. A Windows console here reports cp1256, and a
    # test that printed "z-bar" or "mm^3" died with UnicodeEncodeError AFTER
    # every assertion in it had passed — a green test file reported as a
    # failure, which is worse than either outcome on its own. The suite must not
    # depend on the operator's codepage.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    failed = []
    for name, path in TESTS:
        if not os.path.exists(path):
            print(f"  SKIP  {name:<22} ({path} missing)"); continue
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        if r.returncode == 0:
            last = [l for l in r.stdout.strip().splitlines() if l.strip()]
            print(f"  PASS  {name:<22} {last[-1][:60] if last else ''}")
        else:
            print(f"  FAIL  {name:<22}")
            print("        " + (r.stderr.strip().splitlines() or ["?"])[-1][:100])
            failed.append(name)

    print("=" * 66)
    if failed:
        print(f"{len(failed)} FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    print("\nGeometry verified. app_ui.py needs a real scan to exercise -")
    print("run:  python app_ui.py")

if __name__ == "__main__":
    main()
