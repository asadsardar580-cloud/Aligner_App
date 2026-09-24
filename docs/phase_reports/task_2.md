# Task 2 — a print-ready STL from the deformation path, by voxel solidification

## 1. Summary

`construction="deformation"` on `/export/final` and `/export/stages` now ships a
**voxel-solidified solid**: T0 cast with teeth in place → the existing
deformation (crown rigid, gingiva harmonic) → contacts measured →
`print_solid.solidify` (MeshLib, exact calls of record) → STL → every gate
measured on the **re-read bytes** → `manufacturing_v2.aggregate_gate_v3` →
STL + manifest. The `t0_cast_self_intersects` refusal and
`clear_undercut_periphery` are gone from this path.

**The real scan was NOT run.** `case_lower.stl` is not in this container
(scans are PHI and gitignored), nor are the segmentation checkpoints. Step 6
is delivered as `real_scan_print_v3.py`, proven end to end on a synthetic
arch; the a–d table on the real scan does not exist yet.

Environment: Linux, Python 3.12.3, numpy 2.5.2, scipy 1.18.1, manifold3d
3.5.2, open3d 0.19.0, meshlib 3.1.4.297 — a fresh venv, **not** the Windows
venv.

---

## 2. Steps

| Step | What | Commit |
|---|---|---|
| 1 | `meshlib==3.1.4.297` pinned; README licence note (non-commercial / educational only) | `04e89ab` |
| 2 | **new** `print_solid.py` `solidify()`; **new** `test_print_solid.py` | `2efc33c` |
| 3 | `print_solid.measure_solid`, `solid_gate`; **new** `test_solid_gate.py` | `499e99c` |
| 4 | `manufacturing_v2` rewritten around the solid (`aggregate_gate_v3`, `build_t0_stage`, `build_stage_v2`); `api_core.build_stage_bundle_v2`; T0 export; **new** `test_print_gate_v3.py` | `490d960` |
| 5 | `cg.measure_interproximal_penetration(..., measure_penetration=True)`; `manufacturing_v2.measure_contacts` + contact gate; `ipr_by_contact_mm`; **new** `test_contact_gate.py` | `4809bbc` |
| 6 | **new** `real_scan_print_v3.py`; **new** `test_real_scan_print_v3.py` | `ccf35f6` |
| 7 | `tools/render_labels_check.py` anterior sign; **new** `test_labels_check_orientation.py` | `21d0e27` |

---

## 3. Findings

### F1 — solidification does NOT guarantee an intersection-free output (VERIFIED)

The decision of record says solidification makes the output intersection-free.
On synthetic casts with scan surface pushed through the base wall, with the
exact calls:

| push (out, down) mm | input colliding pairs | after voxelise | after decimate / final |
|---|---|---|---|
| rigid ring 1.0, 2.0 | 1521 | 8 | 12 |
| rigid ring −1.0, 2.0 | 8 | 0 | 3 |
| rigid ring 0.5, 1.0 | 51 | 0 | 2 |
| smooth bump 0.5, 1.5 | 246 | – | **0** |
| smooth bump 1.0, 2.0 | 798 | – | **0** |

The colliding pairs are micro-folds: ~0.05 mm triangles sharing one vertex at
near-perpendicular normals (cos −0.06, −0.14). Deterministic (identical hashes
over three runs). Sharp creases produce them — once straight out of
voxelisation, otherwise introduced by decimation; smooth crossings (the shape
real anatomy has) gave 0. The gate catches them:
`test_solid_gate.py::test_a_real_solidified_micro_fold_fails_self_intersection`
refuses a MeshLib-produced mesh by name. **Whether the real cast is clean at
0.05 mm is the clinical owner's measurement (reported clean); it is not
re-verified here.**

### F2 — `measure_interproximal_penetration` measured closure, not penetration (VERIFIED)

Its keys are how much the vertex-to-vertex gap shrank. Two boxes 1.5 mm apart:
1.0 mm into the space reads closure 1.0 (touching nothing); 0.2 mm into the
neighbour reads closure 1.7 (the translation) and nothing says 0.2 mm of it is
overlap. Gating on it would refuse case c as "needs 1.0 mm IPR" and could not
see a real overlap. Fixed by adding, opt-in, a signed exact point-to-triangle
depth; the original keys are byte-identical for every existing caller.

### F3 — measure the neighbour's REAL enamel (VERIFIED)

A static neighbour's contact band blends, carving its enamel out of the
crown's way; against that blended surface a push reads clear. Contacts are
measured against the neighbour at T0 (a moving neighbour where it moved to).
On the CONTACTING fixture the labelled crowns never touch (0.44–0.60 mm of
unlabelled bridge), so a 0.30 mm push is refused by the kit's implicit-IPR
band, not by the contact gate — which is right to pass it.

### F4 — stage counts (VERIFIED from `cg.staging_estimate`)

0.25 mm per stage: case c (1.0 mm) builds **4** stages, not 5; case d (0.5 mm)
builds **2**, not 1. The staging rule is a clinical default and was not
overridden; the runner records both counts.

### F5 — "distal" has a quadrant-dependent sign (VERIFIED in code)

`derive_frame_from_region` pins u_BL to anatomy and flips u_MD to stay
right-handed; `frame["u_md_points_distal"]` reports which way it landed. The
runner sends distal as `+mm if u_md_points_distal else -mm` and prints it (on
the synthetic arch, 43 gave `False` → `d_md = -1.00`).

---

## 4. VERIFIED

(pasted in the step commits; the final-tree runs are in §5)

- `test_print_solid.py` 5 passed — 244 scan-vs-wall crossings → 1 body, 0
  self-intersections, 948,708 triangles, 44 s at 0.05 mm.
- `test_solid_gate.py` 16 passed — a failing control for each of the 11 solid gates.
- `test_print_gate_v3.py` 24 passed — failing control per deformation gate,
  contact, attachment and T0 gate; superseded-pair pin; re-gating a real
  build's record reproduces its shipped verdict.
- `test_export_deformation_api.py` 14 passed — seam stage at 0.05 mm: PRINT
  READY, 21 gates, 891,502 triangles, crown p95 0.0041 / max 0.0054 mm, solid
  55–58 s; T0 export PRINT READY at 0.05 mm.
- `test_contact_gate.py` 29 passed.
- `test_real_scan_print_v3.py` — synthetic arch, 0.1 mm voxel (harness proof,
  not the product on a scan):

| case | verdict | crown dev p95 / max (mm) | triangles | s | direction |
|---|---|---|---|---|---|
| a T0 (stage 0) | PRINT READY | 0.0141 / 0.0364 | 527,700 | 28.6 | n/a |
| b 45 extrusion 0.25 (stage 1) | PRINT READY | 0.0141 / 0.0365 | 527,818 | 26.3 | OK +0.2413 mm occlusal |
| c 43 distal 1.0 (stages 1–4) | PRINT READY | 0.0143 / 0.0645 | 527,796 | 98.1 | OK gap to 44 1.668 → 0.776, to 42 3.311 → 4.221 |
| d 41 labial 0.5 (stages 1–2) | PRINT READY | 0.0145 / 0.0364 | 527,748 | 50.7 | OK +0.454 mm labial |

  Control: "labial" sent as −0.5 mm → exit 4, `THE MOVEMENT WENT THE WRONG WAY`.
- `test_labels_check_orientation.py` 4 passed — the old rule pointed +u_sag
  posterior on the horseshoe (anterior +y); the fix points it anterior.
- `real_scan_print_v3.py` without the scan → `NOT VERIFIED: case_lower.stl is
  not here.` exit 77.

## 5. The final tree

### `run_all_tests.py` — VERIFIED

```
$ .venv/bin/python -u run_all_tests.py
  PASS  deformation export API     14 passed in 307.79s
  PASS  print solid (voxel)        5 passed in 107.31s
  PASS  print solid gate           16 passed in 167.61s
  PASS  print gate v3              24 passed in 27.10s
  PASS  contacts / IPR gate        29 passed in 30.56s
  PASS  real-scan v3 runner        5 passed in 257.18s
  PASS  labels picture sides       4 passed in 0.07s
  SKIP (NOT VERIFIED)  real scan (local)   observed exit code 77 (no scan)
57 PASS / 1 SKIP / 4 FAIL
  FAILED: manufacturing iface, segmentation mapping, crosstooth provider, export confinement
```

### `pytest -q` — VERIFIED

```
5 failed, 625 passed, 19 skipped, 21 warnings in 1403.66s (0:23:23)
FAILED test_cast_base.py::test_undercut_wall_crosses_scan
FAILED test_crosstooth_adapter.py::test_the_ai_status_payload_carries_the_registry_for_the_picker
FAILED test_export_confinement.py::test_an_escape_attempt_is_refused_422_not_silently_redirected[C:/Windows/Temp]
FAILED test_export_confinement.py::test_an_escape_attempt_is_refused_422_not_silently_redirected[C:/]
FAILED test_manufacturing_interface.py::test_an_export_that_needs_no_raise_trims_exactly_as_before
```

**Every failure above reproduces identically on the base commit `f36c70b`
in this venv** (checked one by one in a clean worktree), and none touches
this task's code:

| failure | cause |
|---|---|
| export confinement ×2 | `C:/...` is a relative path on Linux — Windows-only test |
| crosstooth provider | the CrossTooth checkpoint is not in the repository |
| segmentation mapping | reads `case_lower.stl` (absent) |
| manufacturing iface, test_undercut_wall_crosses_scan | `build_cast_base` floor triangulation 49.6% / 215.6% out — the `/export` and Phase 1B undercut path, both untouched here |

### Found while running the suite, NOT fixed (outside this task)

- **`fix_test.py` rewrites `test_cast_base.py` whenever `pytest -q` runs.**
  It is a Phase 1B patch script (committed in `c34e186`) whose name matches
  pytest's `*_test.py` pattern, so collection executes it: it re-applies a
  regex replacement and converts the file's CRLF endings to LF, leaving the
  tree dirty. Restored with `git checkout` after the run. Deleting it, or
  adding it to `pytest.ini`'s ignores, is a one-line fix for another task.
- `test_failsafes.py` needs `networkx` (via trimesh's repair), which reaches
  the real venv only as a dependency of `torch`.

### check_structure — VERIFIED

```
$ .venv/bin/python check_structure.py
  156/156 files structurally sound
```

## 6. WRITTEN, not executed

- `real_scan_print_v3.py` on `case_lower.stl` — cases a–d, the real table.
- `tools/render_labels_check.py` re-rendered on the real scan.
- Anything on Windows or the Windows venv.
- Timing and triangle count on the real scan (expected ~1M triangles, ~47 MB).

## 7. Decisions for Asaad

1. **F1.** Solidification can leave micro-folds on creased input; the gate
   refuses them. If the real scan hits this, the options are yours (e.g. a
   post-decimation self-intersection repair) — nothing was added beyond the
   calls of record.
2. **F4.** Stage counts come from 0.25 mm/stage. Cases c and d as asked (5
   and 1 stages) would need a different per-stage limit.
3. **MeshLib licence** — non-commercial only; outside the A2.14 licence rule.
4. **Attachments** now join the same voxelisation as the cast (no boolean);
   crown points buried under an attachment are excluded from fidelity and
   counted, and every attachment must be inside the solid. Not asked for in
   Task 2 — it replaced the union that no longer applies.
