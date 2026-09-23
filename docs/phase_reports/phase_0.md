# Phase 0 — Honest harness and hygiene

**Date:** 2026-09-23 · **Baseline:** `87851bb` · **Head:** `3ac5b6e` · **Engineer:** Claude Opus 5

Every claim below is labelled **VERIFIED** (command run, output pasted) or **WRITTEN** (not executed).

---

## 1. Summary

Every measurement in this repository can now fail loudly. The real-scan harness could not report a failure at all (it printed failed gates as a NOTE and returned 0), the canonical runner had three separate ways to report a pass for something that never ran, and four measurements answered with a known-wrong method or a silent zero rather than admitting they could not be taken.

Nine tasks, twelve commits, **six new test files (69 tests)**, one new gate (the 17th) with a control that makes it fail, and two security holes closed. `run_all_tests.py` is **52 PASS / 0 SKIP / 0 FAIL**; `pytest` is **476 passed**.

**No geometry was added or changed.** `build_stage_tooth_interface`, every `InterfacePolicy` value, every gate threshold, `kinematic_matrix` and `center_of_resistance` are untouched — verified by diff below.

---

## 2. Tasks

| Task | Files and functions changed | Commit |
|---|---|---|
| 0.1 | `manufacturing.py` `KEY_LOOKS_LIKE_A_LEDGE` (4 sites); `real_scan_regression.py` `exit_code_for`, `--expect`, NOT-READY→`_fail`, absent scan 2→77; **new** `test_real_scan_harness.py` | `49b2429` |
| 0.2 | `run_all_tests.py` rewritten: missing=FAIL, SKIP on 77, per-entry args, `can_fail` static guard, 3 new entries; **new** `test_suite_runner.py` | `7879c39` |
| 0.2a | tracked `deform_construction.py`, `test_deform_construction.py`; `.gitignore` `.venv_check/` | `5ca69e3` |
| 0.3 | `api_core.py` `_solid_bodies` counts after re-union; `KEY_INTERNAL_VOIDS`, `KEY_INTERNAL_VOIDS_FILLED`, `KEY_CRUMBS_ALIAS`; call sites + manifest; **new** `test_solid_bodies.py` | `9ef20d8` |
| 0.4 | `manufacturing.py` `_point_to_surface` (NaN + reason), `_point_to_surface_approximate`, `surface_deviation`, `old_site_quality`, fidelity-gate finiteness; **new** `test_distance_failure.py` | `f1895e1` |
| 0.5 | `self_intersection.py` + kit test added; `manufacturing.py` `KEY_SELF_INTERSECTION`, gate 17, evidence claim 10; `api_core.build_stage_bundle` measurement site; **new** `test_self_intersection_gate.py` | `8abb02d` |
| 0.6 | `api_core.py` `EXPORTS_ROOT`, `MAX_UPLOAD_BYTES`, `_confined_out_dir`, chunked upload; `test_cut_endpoint.py` adapted; **new** `test_export_confinement.py` | `af238c5` |
| 0.7 | `build_ai_export.py` `git_state`, `build_info`, `verify`, STORED empties, `require_clean`; **new** `test_build_export.py` | `e581283` |
| 0.7a | `build_ai_export.py` skip `.venv_check` | `3323ba6` |
| 0.8 | `requirements.txt` pinned, desktop split out; **new** `requirements-desktop.txt` | `85bad95` |
| 0.8a | `requirements-dev.txt` pinned, count 36→52, finding recorded | `7b198ec` |
| 0.9 | `CLAUDE.md` → `docs/HISTORY_CLAUDE_2026-09.md`; slim `CLAUDE.md`; README counts; `check_structure.py` skips `.venv_check` | `e01bb78` |
| 0.1b | `real_scan_regression.py` `exit_code_for_process`, `--expect-exit`; suite entry pinned | `3ac5b6e` |

---

## 3. VERIFIED

### Canonical suite — **52 PASS / 0 SKIP / 0 FAIL**

```
.venv\Scripts\python.exe run_all_tests.py

  PASS  real scan (local)              observed exit code  3
==================================================================
52 PASS / 0 SKIP / 0 FAIL
ALL EXECUTED TESTS PASSED
RUNALL_EXIT=0
```

### pytest — **476 passed**

```
.venv\Scripts\python.exe -m pytest -q -p no:randomly
476 passed, 5 warnings in 773.23s (0:12:53)
PYTEST_EXIT=0
```

### check_structure — 117/117 (run after every edit)

```
117/117 files structurally sound
All files structurally sound.
```

### 0.1 — the harness can fail

```
natural 3, no expectation -> 3      natural 3, expect 3  -> 0
natural 0, expect 3       -> 1      natural 1, expect 3  -> 1
natural 77, expect 3      -> 77 (SKIP preserved)
```

Producer/consumer key, executed not just grepped:
```
PASS  producer and consumer agree on 'looks_like_a_ledge'
PASS  an unmeasurable transition reports looks_like_a_ledge=None
```

### 0.2 — the runner's four display paths, by control

```
  PASS  self-intersection          6 passed, 1 warning in 7.82s
  PASS  suite runner               11 passed
  SKIP (NOT VERIFIED)  skip control
        no scan here
  FAIL  fail control               (exit 1)
        AssertionError: deliberate control failure
  FAIL  toothless control          (toothless entry)
        no __main__ block and no assert that runs at module level
2 PASS / 1 SKIP / 2 FAIL
runner exit code = 1
```

Kit tests: `test_self_intersection.py` **6 passed**, `test_deform_construction.py` **16 passed** — the brief's 22.

### 0.3 — bodies after the re-union

```
bodies                     1
internal_voids             1
internal_voids_filled_mm3  216.0
raw                        [1000.0, -216.0, 8.0]      (old rule: 2 bodies)
after re-union             [1000.0]
```
`body_count_agrees_with_stl` passes; written STL has 1 connected component. Regression: `test_staging_export.py` + `test_manufacturing_matrix.py` **38 passed**.

### 0.4 — NaN fails every gate

```
PASS  Open3D raised -> [nan nan] (all NaN), not a fallback answer
PASS  reason recorded: exact point-to-triangle distance unavailable: RuntimeError...
PASS  surface_deviation: measured=False, max_mm=nan, reason recorded
PASS  a NaN in either direction (or both) fails the fidelity gate
PASS  two finite 0.0 mm measurements still pass the gate
PASS  a NaN site distance refuses to assess, and the gate fails
```
Regression: interface + matrix + staging **87 passed**.

### 0.5 — the 17th gate, with a control that fails it

```
PASS  17 gates, all 17 fail on an empty record
PASS  compensation 0.0 mm: 0 intersecting pairs, gate passes
PASS  compensation 0.3 mm: 216 intersecting pairs, 300 faces -> gate FAILS
PASS  compensation 0.7 mm: 1378 intersecting pairs -> gate FAILS
PASS  the folded model reads closed / 0 non-manifold / 1 component,
      and has 1378 self-intersecting triangle pairs
```
Regression: matrix + interface + staging **87 passed**.

### 0.6 — confinement and the upload cap

```
EXPORTS_ROOT     ...\Aligner_App\exports
MAX_UPLOAD_BYTES 209715200
refused 'C:/Windows/Temp' -> 422      refused '../../evil' -> 422
refused 'C:/'             -> 422      refused '~'          -> 422
PASS  an 8 MB body against a 1 MB cap -> 413
PASS  a 64-triangle upload loaded
```
Regression: api_core + cut endpoint + hydration **40 passed**.

### 0.7 — the archive

```
REFUSING to build the export archive: the working tree is dirty.
    M build_ai_export.py
    ?? AGENT_BRIEF.md
    ...
exit=2

PASS  built 341 entries, 8 empty written STORED, 341 verified;
      commit deadbeef, python 3.12.7, manifold3d 3.5.2
PASS  forged STORED size caught: testzip() reports a bad entry: empty.txt
PASS  corrupted payload caught: testzip() reports a bad entry: a.txt
PASS  unopenable archive reported: the archive will not open: BadZipFile
PASS  a leaked .stl still refuses the build
```

### 0.8 — fresh virtualenv

```
py -3.12 -m venv .venv_check
.venv_check\Scripts\python -m pip install -r requirements.txt   INSTALL_EXIT=0
  numpy 2.5.2   scipy 1.18.1   fastapi 0.141.1   uvicorn 0.52.4
  manifold3d 3.5.2   torch 2.13.0   scikit-learn 1.9.0   trimesh 5.1.0
  open3d 0.19.0   einops 0.8.2   cryptography 50.0.1   python-multipart 0.0.32
  PyQt6 absent (correct)   pyvista absent (correct)   vtk absent (correct)

run_all_tests.py, requirements.txt only:
  35 PASS / 0 SKIP / 17 FAIL      all: ModuleNotFoundError: No module named 'pytest'

+ requirements-dev.txt:
  51 PASS / 0 SKIP / 1 FAIL       only: real scan (local) (exit 3)
```
The last line is what the 0.1 addendum then pinned.

### 0.9 — documentation

```
CLAUDE.md                        100 lines
docs/HISTORY_CLAUDE_2026-09.md   3069 lines
@-import of the history: none (correct)
README: 37 routes / 117 files / 52 entries — each measured, not copied
```

### Protected code untouched

```
git diff 87851bb..HEAD --stat -- core_geometry.py     (no output)
```
`build_stage_tooth_interface` and `InterfacePolicy`: the only `manufacturing.py` diffs are the three diagnostic-key constants, the `_point_to_surface` NaN path, `surface_deviation`, `old_site_quality`, and two gate additions. No policy value and no threshold changed.

---

## 4. WRITTEN, not executed

- The `--expect "NOT PRINT READY"` verdict path of `exit_code_for` is unit-tested but has **never run against a real stage**, because no real-scan configuration reaches a stage gate. Its logic is covered; its integration is not.
- `build_ai_export.py` has not produced a *shipped* archive this phase — the tree was dirty throughout by design. The build was exercised into a temp path with the clean-tree check satisfied by a stub.
- `semgrep`, Playwright and the Node verifiers were **not** re-run this phase. They were green in the audit two days ago and nothing in Phase 0 touches their inputs, but that is inference, not verification.
- The self-intersection gate has never been evaluated on a **real** fused stage, because the real scan refuses before one is built.

---

## 5. Real-scan measurements

`case_lower.stl` — 94,848 verts / 187,625 faces. Labels from `case_lower.stl_output.json` (ToothGroupNetwork's own prediction, not ground truth).

| Run | Exit | Refusal | Measured |
|---|---|---|---|
| `--profile smoke` (FDI 31, 32) | 3 | `interface_construction_failed` | `collar_top_unresolved_points` 14 · `collar_top_rise_max_mm` 4.338 · `collar_top_window_lift_points` 361 · `collar_top_clear_of_cast_min_mm` −0.1565 |
| `--fdi 45 --stages 1` | 3 | `interface_unbuildable_wall_too_thin` | `crown_penetration_mm` 3.719 · `local_cast_thickness_mm` 5.2326 · `crown_points_inside_cast` 3274 |

Selection itself is healthy: FDI 31 → 4,616 vertices / 388 rim; FDI 32 → 3,265 / 182. Both sockets take `flat_fallback`. The wand's auto-tolerance returns 25.00 mm with 17% and 13% on the clicked tooth, so the label component is used — unchanged from the documented behaviour.

**No stage was built, so no aggregate gate ran on real anatomy this phase.**

---

## 6. Failures not fixed

| # | Finding | Why not fixed here |
|---|---|---|
| F1 | The real scan reaches no stage gate. FDI 31 is the undercut incisor (B4); FDI 45 refuses on penetration under a mesiodistal push. | Phase 0 adds no geometry. This is what Phases 2–5 exist for. Pinned at exit 3 so any movement shows up. |
| F2 | A NaN in the *second* fidelity direction used to **pass** the gate — `max(0.001, nan)` is `0.001`. | Fixed in 0.4, but it is a *finding*: the same `x <= t` shape may exist in other gates. Not audited exhaustively. |
| F3 | `scratch/manufacturing.py` is a stray shadow copy of the engine, plus four debug scripts, all inside `check_structure`'s walk. | Deleting files is not Phase 0's job. Flagged for a deliberate cleanup. |
| F4 | `requirements.txt` alone cannot run the suite (17 of 52 entries need pytest). | Correct behaviour; documented rather than "fixed" by polluting runtime deps. Confirmed by Asaad. |
| F5 | The self-intersection gate is unproven on real fused geometry. | Blocked by F1. |
| F6 | `test_manufacturing_matrix.py` carried a hard-coded claim count (9) in two places. | Updated to 10 as the brief directs. Worth noting the pattern: a count assertion breaks whenever the set grows. |

---

## 7. Decisions needed from Asaad

**D1 — Real-scan pin. RESOLVED this session.** Asked and answered: pin the process exit code at 3. Implemented in `3ac5b6e`. Recorded here because the expectation must be revisited deliberately in Phase 3, never adjusted to keep the suite green.

**D2 — Fresh-install semantics. RESOLVED this session.** `requirements.txt` stays runtime-only; the suite needs `requirements-dev.txt` too. Recorded in that file.

**D3 — `scratch/` cleanup.** `scratch/manufacturing.py` is a shadow copy of the engine sitting in the tree. *Options:* (a) delete the five scratch `.py` files; (b) move them to an evidence archive; (c) leave them. *Recommendation:* (a) — a second `manufacturing.py` on disk is exactly the trap that produced B8, and nothing imports these.

**D4 — Upload ceiling.** `MAX_UPLOAD_BYTES` is 200 MB, chosen as ~20× the real scan. *Options:* (a) keep; (b) lower to 50 MB. *Recommendation:* (a) until a scanner is seen emitting something larger — it is a resource bound, not a clinical threshold.

---

## 8. Ready for the next phase?

**Yes.** Phase 0's acceptance is met in full:

- `check_structure.py` — all files sound (117/117) ✅
- `run_all_tests.py` — 0 FAIL ✅ · `pytest` green (476) ✅
- the real-scan smoke run is honest — it exits **3**, not 0, and the suite entry pins that ✅
- the ZIP builds from a clean tree and passes verification ✅ (exercised; a shipped build needs a clean tree, which Phase 1 should produce first)

**Caveat on one acceptance line.** The brief expected the smoke run to exit **1**; it exits **3** because the collar path refuses before a stage exists. The property the brief wanted — the suite cannot be green about an unverified real scan, and any change turns it red — is preserved, one level lower. Phase 3 must replace this pin with the case-table expectations, not adjust it.

**Blockers for Phase 1:** none. Phase 1 is read-only for the export path and needs only `case_lower.stl`, which is present.

**One caution.** Phase 1.2 stops work if the T0 cast has self-intersections. The detector is now in the tree and verified against synthetic controls (216 and 1378 pairs on a folded groove; 0 on the same geometry unfolded), so that check is ready to run — but it has never been pointed at the real cast, and that is the first thing Phase 1 should do.
