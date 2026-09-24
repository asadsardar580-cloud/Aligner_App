# Aligner_App — Clinical Micro-Planner

Slim operating file: current rules only, kept under 200 lines.

- **Plan of record:** `AGENT_BRIEF.md`.
- **Full history:** `docs/HISTORY_CLAUDE_2026-09.md`. Read it only when a task needs it, and **never @-import it**.

## Mission
Single-tooth clear-aligner planning for **anterior relapse and mid-course correction**. Explicitly not a full treatment-planning system.

Each stage outputs **one** closed, manifold, self-intersection-free solid (binary STL) with the tooth in its planned position, ready to print for aligner thermoforming.

Benchmarks: Deltaface, 3Shape Ortho System. This is a clinical tool, not an animation sandbox.

## Live entry points
| Layer | File |
|---|---|
| API | `api_core.py` — `uvicorn api_core:app` (launched by `start_backend.bat`) |
| Client | `frontend/src/App.jsx` — `npm run dev` |
| Engine | `core_geometry.py` — pure NumPy/SciPy, headless |
| Manufacturing, current | `deform_construction.py` + `manufacturing_v2.py` ("deform, don't cut") → `print_solid.py` (voxel solid, MeshLib: non-commercial licence) |
| Manufacturing, legacy | `manufacturing.py` (collar) — **frozen**, retired in Phase 5 |
| Geometric validator | `self_intersection.py` |
| Segmentation | `segmentation_providers.py` (ToothGroupNetwork default, CrossTooth selectable) |

- `app_ui.py` (PyQt6) is legacy.
- `_archive/` is dead code.
- Never run the app from `Aligner_App_AI_Export/`.

## Commands
Windows: **always** use `.venv\Scripts\python.exe`.

```
.venv\Scripts\python.exe check_structure.py        # after EVERY edit
.venv\Scripts\python.exe run_all_tests.py          # canonical; PASS / SKIP(77) / FAIL
.venv\Scripts\python.exe -m pytest -q              # must also be green
.venv\Scripts\python.exe real_scan_regression.py --profile smoke   # real scan; exit 1 = not print-ready
node frontend/verify-kinematics.mjs                # browser/backend pin at 1e-12
cd frontend && npm run lint && npm run build && npm run smoke
```

## Non-negotiables
1. **Scanner coordinates are sacred.** Never rotate, re-centre or re-scale. Session `verts`/`faces` are written only at the two load paths.
2. **Vertex ids are the client/server contract.** Never weld, reorder or rebuild the arch array. `build_cast_base` keeps the scan vertex array as a prefix.
3. **Biological kinematics.**
   - Rotation is about C_res, ~10–13 mm apical on the long axis.
   - `M = T(c+t)·R_torque·R_tip·R_rot·T(−c)`.
   - Stage k is rebuilt from clinical × k/N. Never interpolate a 4×4.
4. **Enamel.**
   - The moving crown is `cg.apply_matrix(V0, M)`, bit-for-bit.
   - Static enamel is bit-identical, except inside a declared contact band, which is measured and reported.
   - Never smooth enamel.
5. **Print readiness.**
   - Only an aggregate gate may say PRINT READY. It is a pure function, and a missing measurement fails.
   - Every exported file is validated on its **re-read bytes**.
6. **Failed measurements.** A measurement that cannot be taken returns NaN/None and fails. Never 0.0. Never fall back to a method known to be wrong.
7. **Thresholds.** Fix the measurement, never the threshold. A new threshold needs a measurement and Asaad's approval; until then it is advisory.
8. **Interproximal contacts.** ≤ 0.05 mm is tolerance. > 0.05 mm needs prescribed IPR, otherwise refuse and name the contact.
9. **Diagnostic keys** read by gates or harnesses are module constants, each with a producer/consumer test.
10. **Every new gate ships with a control that makes it fail.**

## Manufacturing construction (current direction)
1. Build the T0 cast **once**, with the teeth in the surface: `trim_to_arch` → `build_cast_base`.
2. For each stage, keep the face array identical to T0:
   - moving crown → exact rigid transform;
   - gingiva within the envelope → harmonic blend (clamped-cotan M-matrix, so 0 ≤ w ≤ 1);
   - everything else → untouched.
3. Topology is inherited from T0: no socket, cap, collar or boolean.
4. The planned surface may cross itself (the real T0 cast does, at the rim). It is **voxel-solidified** (`print_solid.solidify`, MeshLib, 0.05 mm, OpenVDB sign) and the SOLID is what ships. Decision of record (Task 2) — do not re-evaluate.
5. `manufacturing_v2.aggregate_gate_v3`: the kit's deformation gates on the planned surface (inversion, degeneracy, rigidity, implicit IPR, ...); contact penetration per neighbour (≤ 0.05 mm or IPR prescribed for THAT contact); and the solid's gates on the **re-read bytes** — manifold3d status/1 body/volume, MeshLib self-collision, open/non-manifold edges, 1 component, winding, crown fidelity p95 ≤ 0.03 / max ≤ 0.10 mm. Height > 19 mm is a warning only.

## Lessons that must not be relearned
- Nearest-vertex distance is not nearest-surface distance. Use point-to-triangle distance (Open3D BVH).
- Loaders invent vertex orders, so labels must be transferred **by position**. Equal counts prove nothing.
- STL stores positions only. Coincident-but-distinct vertices become non-manifold edges on any reader's weld.
- In manifold3d `decompose()`, a negative-volume part is an **enclosed void**, not a crumb. Count bodies after any re-union.
- Vertex-normal offsets fold at concavities. Gate them with the self-intersection test.
- An `if depth <= 0 or lateral > max` check is not exhaustive: every comparison against NaN is False. Check finiteness first.
- A green suite plus a broken product means the tests cannot see the failure. Add a check that cannot be satisfied by trying harder.
- React: a dependency array is evaluated during render, so declare anything it names above it. Run `npm run smoke`.

## Workflow
- One phase per session (see `AGENT_BRIEF.md`). Stop and write `docs/phase_reports/phase_N.md`.
- Label every claim **VERIFIED** (command + output) or **WRITTEN** (not executed).
- One commit per task. Never build the export ZIP from a dirty tree.
- Scope discipline: no feature creep while the single-tooth manufacturing loop is unproven.
- Asaad (clinician) has final authority on clinical defaults: envelope, contact band, IPR, base.

## Current state — 23 Sep 2026
**Collar path**
- The real scan runs end to end but is **never PRINT READY** (B1–B6).
- The code is frozen; only Phase 0 bookkeeping fixes are allowed.

**Deformation path**
- Verified on synthetic fixtures through the real API: seam movement and T0 export PRINT READY at 0.05 mm (crown p95 0.004 mm).
- Contacts: signed point-to-triangle penetration per neighbour, plus the kit's implicit-IPR band.
- Solidification is NOT guaranteed intersection-free: a creased synthetic crossing left MeshLib micro-folds (gated, refused).
- It has not yet run on the real scan: `real_scan_print_v3.py` (cases a–d) is written and smoke-tested on a synthetic arch only.

**Data and validation**
- One real scan (mandible, `case_lower.stl`).
- No maxilla, no second patient, no lab or thermoforming validation yet.
