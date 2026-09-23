# AGENT BRIEF — Aligner_App → a print-ready stage STL

**For:** Claude Code or Antigravity · **Clinical owner / final authority:** Asaad · **Written:** 23 Sep 2026 · **Baseline commit:** `87851bb`

> This file is the plan of record for manufacturing. It supersedes every manufacturing plan in the old CLAUDE.md.
> Work **one phase per session**. Stop at the end of the phase and write the Phase Report (Part D). Do not start the next phase in the same session.

---

## Part A — Operating rules (non-negotiable)

### A1. Invariants — never break
1. **Scanner coordinates are sacred.** Never rotate, re-centre or re-scale session geometry. Session `verts`/`faces` are written only at the two load paths.
2. **Vertex ids are the client/server contract.** Never weld, reorder or rebuild the arch array. `cg.build_cast_base` keeps the input vertex array as a prefix and only appends floor vertices — keep it that way.
3. **Kinematics are frozen.** `cg.kinematic_matrix` and `cg.center_of_resistance` are not edited. Stage k is rebuilt from clinical × k/N — never interpolate 4×4s. Reuse the existing stage-matrix code path; do not re-derive it.
4. **Enamel.** In every output the moving crown equals `cg.apply_matrix(V0_crown, M)` bit-for-bit. Static enamel is bit-identical except vertices in a declared contact band (measured and reported). No smoothing of enamel, ever.
5. **Only an aggregate gate may say PRINT READY.** It is a pure function of the stage record; a missing measurement fails exactly like a bad one.
6. **A measurement that cannot be taken returns NaN/None and fails** — never 0.0, never a silent fallback to a method known to be wrong.
7. **Interproximal rule (clinical).** Penetration ≤ 0.05 mm is tolerance. > 0.05 mm requires prescribed IPR at that contact; otherwise refuse and name the contact and the amount.

### A2. Working rules
8. After **every edit**: `.venv\Scripts\python.exe check_structure.py`. After every task: that task's tests. End of phase: `run_all_tests.py` **and** `python -m pytest -q`.
9. In every report, label each claim **VERIFIED** (command + pasted output) or **WRITTEN** (not executed). "Should work" is not a status.
10. **Fix measurements, never thresholds.** A new threshold needs a measurement behind it and Asaad's approval; until then it is ADVISORY (reported, never refusing).
11. **The collar is frozen.** Do not add passes, parameters, retries or policy values to `manufacturing.build_stage_tooth_interface` / `InterfacePolicy`. Only the Phase 0 measurement/bookkeeping fixes touch `manufacturing.py`.
12. **Every new gate ships with a control that makes it FAIL.** A gate that cannot fail proves nothing.
13. **Diagnostic keys** read by any gate or harness are constants defined in the producer module, covered by a producer/consumer key test. (This is what stops the `collar_has_ledge` class of bug.)
14. **No new dependency** without Asaad's approval; licence must be MIT / BSD / Apache-2.0 / MPL-2.0; pin exact versions.
15. **Scope:** single-tooth micro-planner (anterior relapse, mid-course correction). No auth/SaaS, whole-arch segmentation or UI redesign unless a phase lists it.
16. **Git:** one commit per task (`Phase N.x: …`). Never build the export ZIP from a dirty tree. Never run the app from `Aligner_App_AI_Export/`.
17. **Windows:** always `.venv\Scripts\python.exe`, never bare `python`.

---

## Part B — Already established (do not re-derive)

### B1. Root cause of the loop — architectural
The collar path cuts the crown out along `socket_rim`, caps the old socket flat on the **same loop**, lofts a connector from the **same loop**, then re-fuses everything by CSG. Near-tangent, self-touching contact is therefore structural — worst for small movements, which is exactly this product's scope. Five collar iterations changed how hard it tries, never the geometry it must fit. **Replacement: "deform, don't cut" (B3).**

### B2. Defect register (verified in code on 23 Sep 2026 unless marked INFERRED)

| ID | Defect | Where | Phase |
|---|---|---|---|
| B7 | Real-scan regression exits 0 on NOT PRINT READY | `real_scan_regression.py` ~395–404 | 0.1 |
| B12 | Reads `collar_has_ledge`; producer writes `looks_like_a_ledge` | `real_scan_regression.py` ~389 | 0.1 |
| R1 | Runner: a missing entry prints SKIP yet the suite "passes"; PASS = exit code only, so a pytest file without `__main__` would pass having run nothing (no current entry is affected — checked) | `run_all_tests.py` | 0.2 |
| B5 | `_solid_bodies` counts positive shells **before** re-union and drops negative shells as "crumbs" — they are enclosed voids (silently filled). Reproduced with manifold3d: raw decompose `[1000, 8, −216]` → reports 2 bodies; exported solid is 1 body, 1 STL component. Matches both real runs (`single_positive_manifold_body` passed, `body_count_agrees_with_stl` failed) | `api_core._solid_bodies` | 0.3 |
| B13 | `_point_to_surface` swallows any Open3D failure and falls back to the approximate 24-candidate method | `manufacturing._point_to_surface` | 0.4 |
| G1 | No geometric self-intersection test exists anywhere. Print compensation (`cg.offset_along_normals`, v + n·d) can fold at concavities while every topology gate passes (INFERRED from the project's own §8/§23 measurements of normal-offset folding) | `api_core`, `manufacturing` | 0.5 |
| B10 / B15 | Client-controlled `out_dir` write; unbounded upload read | `api_core` ~2281, ~3880, ~358 | 0.6 |
| B8 | Archive built from a dirty tree; ships the loosened seat gate `bot_sd <= -margin` | `build_ai_export.py` | 0.7 |
| Z1 | 8 empty ZIP entries recorded STORED with compress_size 2 → Info-ZIP reports bad CRC; Python `testzip()` misses it. Not reproduced with `ZIP_DEFLATED` on Python 3.12.3, so environment-specific: write empty files as STORED explicitly and verify the invariant | `build_ai_export.py` | 0.7 |
| B18 | No version pins; desktop GUI deps (PyQt6, pyvista, pyvistaqt, vtk) in server requirements | `requirements.txt` | 0.8 |
| D1 | CLAUDE.md is 3,069 lines of partly superseded history. Claude Code loads CLAUDE.md into context every session; official guidance is < 200 lines | `CLAUDE.md` | 0.9 |
| B2, B4, B6 | Collar self-touch; undercut incisor rim; old-site flat cap on a saddle-shaped rim (B6 hypotheses, INFERRED: planar-interior cap on a non-planar rim; star-shaped `r(θ)` mask in `old_site_quality`; partial carving by the clearance tool) | collar path | eliminated by construction (Phases 2–5) |
| B3 | Contacting pair 45/46 | collar path | Phase 4 |

**Out of scope for this plan (tracked, not worked):** B9 auth · B11 model licences (Asaad to contact the ToothGroupNetwork and CrossTooth authors) · B14/B19 data (Asaad supplies scans) · B16 shadow rig · B17 cancellation.

### B3. The replacement construction — "deform, don't cut"
Reference code in this kit: `deform_construction.py`, `self_intersection.py` (+ tests).

- Build the closed **T0 cast once**, from the conditioned scan **with the teeth still in the surface**: `cg.trim_to_arch` → `cg.build_cast_base`. Vertex array = trimmed scan array + appended floor vertices. Face array `F`.
- **Per stage, with the same `F` for every stage:**
  - moving-crown vertices: `cg.apply_matrix(V0, M)` — exact.
  - gingiva inside the envelope (default 5 mm geodesic): `V0 + Σ_i w_i (M_i·V0 − V0)`. `w_i` is harmonic on a clamped-cotangent Laplacian (an M-matrix ⇒ discrete maximum principle ⇒ 0 ≤ w ≤ 1 and Σw ≤ 1). Dirichlet data: 1 on the moving crown; 0 on other teeth, the trim rim, walls, floor and the envelope edge.
  - everything else: never written (bit-identical).
- **Topology is inherited from T0** (closed, manifold, one component, consistent winding). There is no socket, cap, collar or boolean. Only geometry can fail, and it is measured: triangle inversion vs T0, degenerate triangles, geometric self-intersection on float32-rounded positions, implicit IPR at contacts.
- `aggregate_gate_v2`: 10 required gates, pure, fail-closed, plus advisory metrics.

**VERIFIED on 23 Sep 2026** (Linux, Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, manifold3d 3.5.3 — **not** the Windows venv; re-verify in Phase 2):
- 22/22 reference tests pass through the project's **real** `trim_to_arch`, `build_cast_base`, `kinematic_matrix`, `apply_matrix`, `write_binary_stl_bytes`, `parse_stl_bytes`, `weld_vertices`, on `test_cast_base.horseshoe_shell` (curled undercut walls, domed crowns with a cervical sulcus). `check_structure.py`: 105/105.
- **Seam movements — 7/7 PRINT READY:** 2° + 0.25 mm; 15° derotation; 1 mm labial; ±1 mm extrusion/intrusion; 10° torque + 8° tip; 10°/10°/10° + 1 mm + 0.5 mm. Each: 0 inverted, 0 self-intersections, crown bit-exact, everything else bit-identical, file round trip closed / manifold / 1 component.
- **Refusals fire:** crown driven into its neighbour; violent movement in a tiny envelope; a mutant writing the crown as `V0 + (M·V0 − V0)` (numerically ~1e-15 off) is caught by the bit-exact rigidity gate.
- **Self-intersection detector vs exact rational-arithmetic ground truth:** 2,420 true intersecting pairs, 0 missed, 0 false alarms; a one-sided mutant misses 605. Touch tolerance 1 µm (flags ≤ 1 µm, clears ≥ 2 µm).
- **Speed:** full check 6.5 s at 327,680 faces. At ~95k vertices: weights 0.30 s once per case; **0.38 s per stage** including self-intersection and STL bytes. (Current collar path: 106–144 s for 1–2 teeth.)

### B4. Contacts — the one hard sub-problem (measured)
Where two crowns are merged in the scan (the contact point is hidden from the scanner and the mesh bridges it), faces spanning two rigid bodies in relative motion must shear, and beyond their own width they fold. On a merged-contact fixture, **every inverted face was a bridge face; none was gingiva.**

Implemented policy:
- `dc.contact_band`: release the **neighbour's** enamel within 1.5 mm (geodesic) of the moving crown so it can blend. The moving crown stays bit-exact.
- Per stage, band displacement is measured along the T0 outward normal:
  - **inward = implicit IPR**, gated at max(0.05 mm, prescribed IPR for that contact);
  - **outward = interproximal bridge** (block-out), advisory.
- Measured on the fixture:
  - A 0.3 mm mesial push into a tight contact gives 0.14 mm implicit IPR and is refused; with ≥ 0.19 mm prescribed IPR it is PRINT READY.
  - A 1 mm extrusion (separating) is PRINT READY, with a 0.50 mm bridge reported.
  - A 15° derotation of a broadly merged contact is refused (sliding).
- **Limit:** large sliding at a broadly merged contact cannot be absorbed by any fixed-topology construction. If Phase 3 shows clinically necessary movements refused for this reason, escalate to a **research spike**, not an implementation: separate crowns at the contact, complete the proximal surfaces, then exact union. This needs a licence decision (CGAL is GPL or commercial).

### B5. NOT verified — do not claim
The real scan (T0 self-intersections, tooth sets, contact census, timing) · Windows and the pinned versions · maxilla · a second patient · lab and thermoforming acceptance · clinical acceptability of the envelope and band defaults.

---

## Part C — Phases

### Phase 0 — Honest harness and hygiene (no new geometry)
**Goal:** every measurement can fail loudly, and the tree is reproducible.

**0.1 `real_scan_regression.py`**
- Read `looks_like_a_ledge`. Define the key name as a constant in `manufacturing.py` and import it.
- `if not gate["print_ready"]:` → also set `ok = _fail(...)` (keep the NOTE text).
- Extract `exit_code_for(stage_gates, expect=None) -> int` as a pure function. Unit-test it:
  - no `expect`: any NOT PRINT READY → 1; all ready → 0; empty list → 1;
  - `expect="NOT PRINT READY"`: exit 0 only if **every** stage's verdict equals the expectation; any mismatch in either direction → 1.
- Add the `--expect <verdict>` flag to the script.
- **Acceptance:** `real_scan_regression.py --profile smoke` on `case_lower.stl` now **exits 1**. That is correct: the collar path is not print-ready. With `--expect "NOT PRINT READY"` it exits 0.

**0.2 `run_all_tests.py`**
- A missing entry file → **FAIL**, not SKIP.
- SKIP only when an entry exits **77**. Print it as `SKIP (NOT VERIFIED)`. Summary line: `N PASS / M SKIP / K FAIL`; exit 1 if K > 0.
- Support per-entry arguments.
- Add these entries:
  - `("self-intersection", "test_self_intersection.py")`
  - `("deformation construction", "test_deform_construction.py")` — both files carry `__main__` → `pytest.main`.
  - `("real scan (local)", "real_scan_regression.py", ["--profile", "smoke", "--expect", "NOT PRINT READY"])` — exits 77 when the scan is absent. The expectation records today's **known** state of the collar path, so the suite stays green while any change in either direction turns it red. Update the expectation deliberately when the construction changes (Phase 3).
- Static guard: FAIL any entry whose source has neither a `__main__` block nor a module-level `assert`.

**0.3 `api_core._solid_bodies`**
- Count bodies **after** re-union: positive-volume parts of `result.decompose()`.
- Return and record `internal_voids` (count) and `internal_voids_filled_mm3` (Σ |negative volumes|). Rename the "crumbs" manifest keys, keeping the old key as an alias for one release.
- Test: `cube(10) − cube(6) + cube(2)`, all centred → bodies = 1, voids = 1, filled = 216.0 ± 1e-6, STL components = 1, `body_count_agrees_with_stl` passes.

**0.4 `manufacturing._point_to_surface`**
- Delete `except Exception: pass`. On Open3D failure return an all-NaN array and record the exception.
- Verify every caller's gate treats NaN as a failure.
- Test: monkeypatch `open3d` to raise → NaN returned, the fidelity gate fails, the manifest records the reason.

**0.5 Self-intersection gate on the existing path**
- Add `self_intersection.py` and its test from the kit.
- In `build_stage_bundle`, after the float32 rounding (and after print compensation), run `self_intersection_report(sv, sf)`. Record it under a constant key.
- Add gate `no_self_intersection` to `mfg.aggregate_print_gate` (now 17 gates). Update `manufacturing_evidence_report`, and the empty-record test (17/17 fail).
- Control: a synthetic fused stage with a print compensation large enough to fold at a concavity → the gate fails.
- If adding this gate makes existing matrix tests fail, that is a **finding**: report which cases fail and where. Do not weaken the gate.

**0.6 Security (small, contained)**
- Remove `out_dir` from `ExportRequest` and `StageExportRequest`, or confine it to `<repo>/exports` via `Path.resolve()` + `is_relative_to` (422 otherwise).
- Upload: read in chunks; 413 above `MAX_UPLOAD_BYTES` (default 200 MB).
- Tests for both.

**0.7 `build_ai_export.py`**
- Refuse to build if `git status --porcelain` is non-empty (exit 2 and list the files).
- Add `BUILD_INFO.json`: commit, UTC time, Python and key package versions read from the venv.
- Write zero-byte files via `ZipInfo` + `ZIP_STORED`.
- After writing, re-open the archive and verify every entry: read it fully (CRC check), and assert STORED ⇒ `compress_size == file_size`. Fail the build otherwise.
- Test with a tree containing an empty file.

**0.8 Dependencies**
- Run `.venv\Scripts\python.exe -m pip freeze` and pin the **exact installed** versions of direct dependencies in `requirements.txt`. Do not trust version lists in any document.
- Move PyQt6, pyvista, pyvistaqt and vtk to `requirements-desktop.txt`.
- Fresh-install check: `py -3.12 -m venv .venv_check` → `.venv_check\Scripts\pip install -r requirements.txt` → `.venv_check\Scripts\python run_all_tests.py`.

**0.9 Documentation**
- Move the current `CLAUDE.md` to `docs/HISTORY_CLAUDE_2026-09.md`. Do **not** @-import it.
- Install the kit's `CLAUDE.slim.md` as `CLAUDE.md`. Fix the README counts. Create `docs/phase_reports/`.

**Phase 0 acceptance:**
- `check_structure.py` reports all files sound.
- `run_all_tests.py` reports 0 FAIL, and `pytest` is green.
- The real-scan smoke run **without** `--expect` exits 1 (honest); the suite entry with `--expect "NOT PRINT READY"` passes.
- The ZIP builds from a clean tree and passes verification.

### Phase 1 — Measure the real scan at T0 (read-only for the export path)
New script `tools/t0_census.py`. No changes to `api_core` or `manufacturing`.

**1.1** Load `case_lower.stl` through the upload-path functions: `stl_io.parse_stl_bytes`, then the same sanitize/condition steps as `create_session`. Fit the occlusal frame the same way `real_scan_regression.py` does.
- Build the T0 cast **with teeth in place**: `trim_to_arch` → `build_cast_base`, reusing the existing per-export trim-margin floor.
- Record: vertices, faces, open and non-manifold edges, components, winding, volume, and the float32 round trip via `mfg.validate_printable_stl`.

**1.2** Run a full `self_intersection_report` on the float32-rounded T0 cast. Record the count, `by_kind`, and the first 50 examples with positions.
- **If the count is > 0: stop writing code** and report the locations. The cleanup rule is decision E7 for Asaad.

**1.3** For every labelled tooth, using labels transferred by position (the `/select-tooth` route), record:
- vertex count and number of boundary loops;
- whether it touches the trim rim (must be False);
- the minimum geodesic distance to the trim rim (advisory if below the envelope radius).

**1.4** Contact census for every adjacent pair:
- bridge faces (faces with vertices in both teeth) and bridge length;
- for pairs with no bridge, the minimum gap.

**1.5** Timings: T0 cast build and the full self-intersection check.

**Output:** `reports/t0_census.json`, plus a table in `docs/phase_reports/phase_1.md`.

### Phase 2 — Integrate "deform, don't cut" behind a flag
**2.1** Add `deform_construction.py` and `test_deform_construction.py` from the kit. Run `.venv\Scripts\python.exe test_deform_construction.py` → expect 16 passed. Record the package versions.

**2.2** New module `manufacturing_v2.py` — pure, with no FastAPI imports.
- `build_case_plan(scan_v, scan_f, arch_frame, moving, static_teeth_vertices, policy) -> CasePlan`:
  - T0 cast with teeth, built as in 1.1;
  - pinned = `rim_loop` ∪ {i ≥ len(trimmed verts)};
  - contact bands via `dc.contact_band` on the static side (`policy.band_mm`);
  - weights via `dc.plan_deformation`;
  - T0 self-intersection must be 0, otherwise refuse the case and include the census data.
- `build_stage_v2(case_plan, stage_matrices, prescribed_ipr) -> StageResult`:
  1. positions: `dc.stage_positions(..., apply=cg.apply_matrix)`;
  2. report: `dc.stage_report`;
  3. round to float32;
  4. `self_intersection_report(..., active_faces=moved)`;
  5. `cg.write_binary_stl_bytes`;
  6. `mfg.validate_printable_stl(blob)` → the file dict;
  7. `dc.aggregate_gate_v2`;
  8. manifest: construction id, policy, weight diagnostics, per-tooth 4×4, gate verdict with failed gates and measured values, advisory metrics, SHA-256 of the STL bytes, git commit, package versions.
- **Stage matrices:** call the **same** function the collar path uses. Extract it into a shared pure function if needed, and test that both paths produce identical matrices (0 ULP).

**2.3** Tooth vertex sets:
- Moving set = vertices of that tooth's labelled/extracted faces, on the **original** scan ids — not the crown copy.
- Static set = vertices of all other tooth-labelled faces.
- Shared contact vertices go to the moving tooth, and the count is reported.

**2.4** API: `/export/final` and `/export/stages` accept `construction: Literal["collar", "deformation"] = "collar"`.
- The deformation path returns the same response shape.
- `X-Print-Ready` comes from `aggregate_gate_v2`.
- No UI change is required; verify with the Playwright E2E suite.

**2.5** API tests with the synthetic fixture through `construction=deformation`:
- a seam movement → 200 and `X-Print-Ready: true`;
- a push into the neighbour → NOT READY with the gate named.

**2.6** Attachments, if present: stage solid = manifold3d union(deformed cast, M·attachment). This is the **only** boolean in this path. Re-run the file-level and self-intersection gates on the fused result. `index_buffer_unchanged` applies to the pre-attachment cast.

**Acceptance:**
- All suites are green.
- Real scan, FDI 45, one stage, 0.25 mm extrusion, via `construction=deformation` → either PRINT READY or a **named** refusal with measured values. A crash or a silent result fails the phase.
- Timing is recorded.

### Phase 3 — Real-scan regression v2 (the check that cannot be satisfied by trying harder)
**3.1** Create `tests/real_cases.json`. Each row: `{fdi, six clinical values, stages, prescribed_ipr, expected: "PRINT READY" | "<gate name>"}`. Asaad signs the expectations **before** anything runs.

**3.2** Initial table (anterior relapse scope):
- FDI 31, 32, 41, 42 × {derotation 5°, 10°, 15°; labial 0.5 mm, 1.0 mm; lingual 0.5 mm; extrusion 0.5 mm; intrusion 0.5 mm; tip 5°; torque 5°} × 5 stages;
- plus FDI 45 and 46 (B3) at 0.25 mm extrusion.

**3.3** `real_scan_regression.py --construction deformation --cases tests/real_cases.json`:
- exits 1 on any mismatch;
- saves each stage STL and manifest under `scratch/real_v2/` (gitignored);
- prints a verdict table with the measured values.

**3.4** Replace the Phase 0 "real scan (local)" entry with this case-table run. It exits 77 when the scan is absent; per-case expectations replace the single `--expect`.

**Acceptance:** a verdict table in the phase report; every refusal names its gate and value; no unexplained mismatch.

### Phase 4 — Contacts and IPR (needs Asaad's decisions first — Part E)
**4.1** Add `IPRPrescription {fdi_a, fdi_b, amount_mm, from_stage}` to `domain.py`. Persist it with the plan, expose it through the API, and add a minimal input in `ToothInspector.jsx`.

**4.2** Extend `dc.stage_report` to report implicit IPR **per neighbour tooth**. Gate each contact against its own prescription, not the global maximum.

**4.3** Manifest: per-contact bridge height (advisory) and implicit IPR.

**4.4** If Phase 3 shows necessary movements still refused for sliding at merged contacts even with prescribed IPR, write a research-spike document (proximal completion + exact union, with licence options). **Do not implement it.**

**Acceptance:** FDI 45/46 either builds, or refuses with "IPR x.xx mm required at 45/46 from stage k".

### Phase 5 — Retire the collar
**5.1 Preconditions:** the Phase 3 table passes on the mandible **and** on at least one maxillary scan **and** on at least one second-patient scan (Asaad supplies the scans).

**5.2** Switch the default construction to `"deformation"`.
- Keep `"collar"` behind the flag for one release.
- Then delete `build_stage_tooth_interface` and the collar-only gates and tests.
- Move `manufacturing.py.bak`, `log_*.txt` and `test_sweep.py` out of the repo into an evidence archive.

**Acceptance:** suites green; docs updated.

### Phase 6 — Physical validation (Asaad, with agent support)
**6.1** Print the T0 model and one final-stage model. Intraoral-scan the printed models and compare them with the source STLs (two-sided `mfg.surface_deviation`). Report p95 and maximum deviation.

**6.2** Thermoform one aligner on the stage model. Check seating, and note any issues at contacts and the gingival margin.

**6.3** Log the results in `docs/VALIDATION_LOG.md`.

### Phase 7 (optional)
3MF export as an **additional** format (trimesh exporter or lib3mf, BSD-2). It is not a fix for anything.

---

## Part D — Phase Report template
Save as `docs/phase_reports/phase_N.md`.

1. **Summary** (3 lines).
2. **Tasks** — for each: files and functions changed, commit hash.
3. **VERIFIED** — commands run, with the pasted tail of the output (pass counts, exit codes).
4. **WRITTEN, not executed** — list.
5. **Real-scan measurements** — table, if any.
6. **Failures not fixed**, and why.
7. **Decisions needed from Asaad** — numbered; each with options and a recommendation.
8. **Ready for the next phase?** yes/no, plus blockers.

---

## Part E — Decisions reserved for Asaad (clinical owner)

| # | Decision | Default / recommendation |
|---|---|---|
| E1 | Envelope radius | **5 mm.** The aligner edge sits 0–2 mm onto the gingiva; 5 mm keeps the transition gentle and beyond the trimline. |
| E2 | Contact band width, and whether deviation of the neighbour's contact zone is acceptable (it is reported as bridge / implicit IPR) | **1.5 mm** |
| E3 | IPR workflow | Tolerance **0.05 mm** (your rule); prescriptions per contact and per stage. |
| E4 | Gingival follow at the seam | **100% (w = 1)** for manufacturing: crown exact, no exposed band. Literature ratios (≈80% for extrusion/intrusion) are for visual simulation only. |
| E5 | Base height and thickness | Set per your printer and lab. The research cites SprintRay's guide: ≈3 mm below the lowest gingival margin, ≤ 19 mm total height. Confirm for your printer. |
| E6 | Print compensation | Keep at **0** until a lab asks. Once G1 is in place, a folding offset is refused. |
| E7 | T0 cleanup rule | Needed only if the real scan has self-intersections (Phase 1). |
| E8 | Phase 3 expectation table | Sign it before Phase 3 runs. |
| E9 | Moving–moving contacts: which tooth carries the band | The tooth with the smaller movement at that contact. |

---

## Part F — Kit contents
- `AGENT_BRIEF.md` — this file
- `CLAUDE.slim.md` — becomes `CLAUDE.md` in Phase 0.9
- `self_intersection.py`, `test_self_intersection.py`
- `deform_construction.py`, `test_deform_construction.py`
