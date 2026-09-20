# Manufacturing reconstruction — changelog

**Status: PARTIAL.** The architecture is replaced, implemented and tested. One
acceptance criterion is **not** met and is documented precisely in §N and §S:
later stages of an extrusion still fuse into more than one body, so they are
refused by the hard gate rather than shipped.

---

## A. Executive summary

The exported cast distorted when a tooth moved because the manufacturing path
unioned a **static** cast with `crown + a 9 mm root plug` transformed by the
stage matrix. Nothing reconstructed the cast at the tooth's **new** cervical
position, and the plug travelled with the tooth — so an extruding tooth carried
synthetic root material up out of the gingiva as visible positive geometry.

That path is gone. A tooth now gets a **bounded local interface** built at its
target position from the actual transformed rim, the actual transformed crown
and the actual local cast, and the boolean order is
**add tissue → (subtract cavity) → union crown + connector**.

> **CORRECTION, 2026-09-20.** This line used to read "subtract -> add ->
> union", which contradicted section I and the code. The order is ADD FIRST:
> subtracting before the ramp is built removes the cast material the ramp has
> to land on and orphans it, and the fused model went from 1 body to 4 across
> an extrusion. The cavity stage is also currently EMPTY - a stage model is a
> POSITIVE, so a penetrating tooth is absorbed by the union and nothing has to
> be removed. The wiring is kept so the manifest can state that it removed
> nothing, rather than being silent about it.
The final STL is validated from **the bytes that were written**, not from the
in-memory boolean.

| | Before | After |
|---|---|---|
| Fusion connector | 9 mm root plug, depth = `root_length_mm` | bounded seat, 1.2 mm, independent of root length |
| Cast at target position | never reconstructed | cavity + emergence ramp, adaptive |
| Boolean | union only | add ramp → (subtract cavity, currently empty) → union crown + connector |
| Volume growth over a 1.2 mm extrusion | **+28.84 mm³** of emerged root | +62.8 mm³, all attributable to the measured ramp/interface |
| Final gate | in-memory index buffer | **the written STL**, after a reader's weld |
| Welded non-manifold edges (extrusion, 5 stages) | 75, 77, 73, 70, 68 | **0, 1, 0, 1, 0** |
| Unaffected cast deviation | never measured | mean **0.0046 mm**, p95 **0.0000 mm**, p99 **0.041 mm** |

---

## B. Confirmed original root causes

All seven claims were proven by inspection before any edit.

1. **Upload conditioning does not deform retained vertices.** `condition_mesh`
   welds by exact bit pattern, drops degenerate faces and debris, fills holes by
   *appending*. Every vertex operation is a gather. `sanitize_scan` is
   delete-only. **Not a cause.**
2. **Crown movement is rigid.** `kinematic_matrix` = `T(c+t)·R·T(−c)`, R a
   product of three Rodrigues rotations. `verify-kinematics.mjs` pins
   `RᵀR−I = 4.44e-16`, `det−1 = 6.66e-16`. **Not a cause.**
3. **One static base for all stages.** `base_solid` built once above the loop;
   every iteration started `parts = [base_solid]`. **Cause.**
4. **`_seal_sockets(flush=True)` fills the ORIGINAL socket.**
   `build_socket_cup(v[rim], depth_mm=0.0)` — original array, original ids,
   called once before the loop. **Cause.**
5. **`_rim_plug` used `root_length_mm` as fusion geometry.** Its own docstring:
   *"DEPTH IS THE ROOT LENGTH, not some small seating value."* **Cause.**
6. **Union with no target-position cavity.** The loop's only boolean was
   `batch_boolean(parts, OpType.Add)`; no `Subtract` existed anywhere in the
   stage path. **Cause.**
7. **Welded non-manifold edges measured but not gated.** The code *did* reread
   the STL into `rf` and measure it into `welded` — then gated on
   `manifold_report(sf)`, the in-memory buffer. The reread vertices `rv` were
   never examined at all. **Cause of the validation gap.**

---

## C. Files modified

| File | Nature of change |
|---|---|
| `api_core.py` | stage loop rearchitected; `_manufacturing_tooth` stripped of plugging; export weld before write; hard gate on the written STL; per-stage/per-tooth diagnostics; `/export/final` added; `build_stage_bundle` returns `blobs` and `base_mesh` |
| `frontend/src/App.jsx` | button renamed; `exportFinal` + PRINT READY / NOT PRINT READY verdict; drag path now sends `opposing_session_id` |
| `test_staging_export.py` | unchanged — left failing deliberately, see §N |
| `run_all_tests.py` | registered `manufacturing iface` |

## D. Files added

| File | Purpose |
|---|---|
| `manufacturing.py` | the whole interface layer — `InterfacePolicy`, `CastProbe`, `build_stage_tooth_interface`, `affected_region`, `surface_deviation`, `_point_to_surface`, `rigidity_report`, `interface_continuity`, `transition_quality`, `old_site_quality`, `validate_printable_stl`, `components`, `bridge_between`, `overlap_volume` |
| `test_manufacturing_interface.py` | the 30-case matrix (33 tests) |
| `MANUFACTURING_RECONSTRUCTION_CHANGELOG.md` | this file |

**`manufacturing.py` is a separate module on purpose.** The brief's central
requirement is that `root_length_mm` stop being fusion geometry. A separate
module makes that structural: nothing in it imports a root length and there is
no parameter one could arrive through.

## E. Deleted / deprecated

* **`_rim_plug` no longer participates in the manufacturing export.** The
  function still exists (unreferenced by the stage path) so the deletion is
  reviewable rather than archaeological; `PLUG_INSET_FRACTION` and
  `PLUG_LIFT_MM` likewise.
* `root_length_mm` remains **only** for C_res estimation and the wireframe
  virtual root. Proven by `test_manufacturing_interface_is_independent_of_root_length_with_M_FROZEN`.

## F. Functions materially changed

* **`_manufacturing_tooth`** — was `crown ∪ _rim_plug(root_length_mm)`; is now
  the crown alone.
* **`build_stage_bundle`** — see §I.
* **`surface_deviation`** — nearest-*vertex* → nearest-*triangle*. This was not
  cosmetic: vertex distance reported **3.46 mm** of phantom deformation
  18–32 mm from any tooth, because the base's large triangles put new vertices
  far from old corners while sitting exactly on the original surface.

## G. Old algorithm

```
seal original sockets flush  →  trim  →  build ONE cast base
for each stage k:
    parts = [that same static base]
    for each tooth: parts += crown ∪ plug(depth = root_length_mm), transformed by M_k
    union(parts) → write STL → measure the reread → gate on the IN-MEMORY buffer
```

## H. New algorithm

```
seal original sockets flush  →  trim  →  build ONE immutable cast  (old site restored)
build CastProbe once
for each stage k:
    for each tooth:
        rim_k, crown_k, u_oa_k = transform by the SAME M_k the crown gets
        classify each rim point against the cast (signed distance)
        depth  = max(min_depth, penetration + overlap + clearance)
                 + emergence floor IF the rim is still seated
                 clamped by local cast thickness × safe_wall_fraction
                 clamped by max_reconstruction_depth
        cavity = socket cup at rim_k, OUTSET, raised above the surface
        ramp   = apron from rim_k to the cast, radius ∝ local lift   (where lifted)
        seat   = truncated cone, lifted into the crown, flared into the cast
        refuse if outside the envelope / wall too thin / unbuildable
    detect adjacent-ROI overlap → refuse if the gingival bridge would go
    base_k = base + ramps            (build tissue first)
    base_k = base_k − cavities       (then cut the socket through it)
    stage  = base_k ∪ seats ∪ rigid crowns
    weld → write STL → reread → reader-weld → HARD GATES
```

## I. Exact boolean order

1. `batch_boolean([base] + ramp_tools, Add)`
2. `batch_boolean([·] + cavity_tools, Subtract)`
3. `batch_boolean([·] + seat_tools + crown_solids, Add)`

**Batch at every step, never sequential per tooth**, so tessellation error
cannot accumulate across teeth.

**Ramp before cavity was measured, not assumed.** Subtracting first removes the
cast material the ramp has to land on and orphans it — the model went from 1
body to 4 across an extrusion.

## J. Interface construction logic

Everything is driven by measurement; every constant is a **ceiling**, never a
shape.

| Quantity | Source |
|---|---|
| cavity depth | actual crown penetration + overlap + clearance |
| depth clamp | measured local cast thickness × 0.50, and a 4.0 mm ceiling |
| ramp radius | **per rim point**, proportional to that point's lift |
| ramp landing | ray-cast onto the real cast, bounded to 4.0 mm |
| seat | fixed 1.2 mm, independent of root length |
| refusal | separation > 4.0 mm, wall too thin, unbuildable, bridge consumed |

`emergence_depth_mm = 1.5` is a **tool-generation default**, applied only where
the rim is still seated, never as an anatomical claim.

## K. Validation gates

`manufacturing.validate_printable_stl(blob)` — on the written bytes, after
simulating a downstream reader's weld:

`finite_coordinates` · `zero_open_edges` · `zero_nonmanifold_edges` ·
`positive_volume` · `single_component` · `consistent_winding`

Any failure ⇒ **NOT PRINT READY**. `/export/final` returns **no file**.

## L. Test matrix and exact results

`test_manufacturing_interface.py` — **33 passed, 0 failed** (25.9 s).

Covers: zero movement · pure MD · pure BL · pure tip · pure torque · pure
rotation · intrusion · extrusion · combined · large lateral (2.5 mm) · two
simultaneous teeth · adjacent teeth approaching · 40 mm outside cast → refused ·
crown rigidity across all 10 movements · unaffected-cast fidelity · interface
continuity · old-site quality · root-length independence with **M frozen** ·
no root column · written-STL validation · open edges · non-manifold · components
· volume · winding · finite coordinates · prescription fidelity · rigid staging ·
real (non-mocked) manifold3d boolean.

## M. Real-scan status

**NOT RUN. Explicitly pending.** `case_lower.stl` is present and is a real
de-identified mandibular scan, but the real-scan regression was not executed
within this pass. It is not claimed, and nothing in this document is based on
it. CLAUDE.md §19 already records why it is hard: the real scan's auto-cut
crowns are thin shells the manufacturing screen correctly refuses, so the path
needs a wand cut seeded from segmentation-label centroids.

## N. Remaining limitations

1. **Later extrusion stages fuse into more than one body.** Measured, 1.2 mm
   extrusion over 5 stages: bodies `1, 1, 2, 2, 3`. Only stage 1 is PRINT READY
   (`1/5`). On the default mixed prescription, `0/2`. The hard gate refuses the
   rest rather than shipping them, which is correct behaviour for geometry that
   is not one solid — but it means **the print path is usable only for stages
   that pass**, and most do not.
2. **`test_staging_export.py::test_every_stage_is_one_fused_watertight_solid`
   and `test_failsafes.py::test_print_compensation...` now FAIL** on
   `components == 1`. They are left failing deliberately. They are recording
   limitation 1 accurately and weakening them would erase the only automated
   signal for it.
3. **23 of 2910 sampled cast points deviate up to 1.15 mm** on the trim
   boundary, where the batch boolean retessellates the base outline. They do
   not move closer to the teeth as the exclusion radius grows 5 → 10 mm, so
   they are not interface artefacts. p95 is exactly 0.0000 mm.
4. **The cervical tangency is not fully eliminated.** 68–77 → 0–1 welded
   non-manifold edges, but not reliably 0 on every stage. CLAUDE.md §10 records
   five earlier attempts at this same tangency; this is the sixth and it got
   closest, not all the way.
5. **No browser verification** of the two new UI controls beyond SSR smoke.

## O. Commands used

```
python -m compileall .
python check_structure.py
python run_all_tests.py
python -m pytest -q
python -m pytest test_manufacturing_interface.py -q
node frontend/verify-kinematics.mjs
node frontend/verify-framing.mjs
node frontend/verify-shadowrig.mjs
semgrep --config .semgrep.yml --error --metrics=off .
cd frontend && npm run lint && npm run build && npm run smoke
python build_ai_export.py
```

## P. Export behaviour

| Endpoint | Ships | Gate |
|---|---|---|
| `POST /export` | base + loose crowns, for **inspection** | watertight base |
| `POST /export/stages` | every stage, each with its own verdict | records, does not refuse |
| `POST /export/final` | **ONE** stage, ZIP of STL + `manifest.json` | **hard** — no file if any gate fails |

UI: *Export Printable Cast* → **Export Setup / Inspection**; **Export All
Stages**; **Export Final Print STL** with a PRINT READY / NOT PRINT READY
verdict naming the failed gate and its measured value.

## Q. Final archive path

`C:\Users\Lenovo\OneDrive\Desktop\Aligner_App\Aligner_App_AI_Export.zip`
Backup taken first:
`Aligner_App_AI_Export_pre_manufacturing_fix_backup.zip` (816,171 bytes).

## R. Final project status

**PARTIAL.** Architecture replaced and verified on synthetic geometry; the
print gate is real and refuses correctly; body fusion for extruded stages and
the real-scan regression are outstanding.

## S. Known unresolved issue

**The single unresolved defect is body fusion on extruded stages** (§N.1). The
seat reaches the cast at small movements and loses it as the tooth lifts,
because the cavity floor and the ramp compete for the same millimetre of
tissue. The parameters were swept (cavity outset 0.00–0.30 mm, seat outset
0.30–1.20 mm, seat depth 0.30–1.20 mm, raise conditional and unconditional);
the configuration shipped is the best measured on non-manifold edges, which was
judged the harder gate. A configuration that keeps more stages in one piece
exists (bodies `1,2,1,1,2`) but costs topology (`0,16,8,2,1` non-manifold), and
that trade is recorded in the code beside the constant.

---

## FILE-BY-FILE EDIT LOG

### `manufacturing.py` — NEW, 1,000+ lines

* `InterfacePolicy` — every bound with a written justification.
* `CastProbe.signed` — nearest-vertex signed distance, for classification only.
* `CastProbe.local_thickness` — **Möller–Trumbore**, replacing a signed-distance
  march that reported **0.1 mm of material under a 9 mm cast** because the
  nearest vertex's normal flips on a steep cervical wall. Every interface was
  being refused `wall_too_thin`.
* `CastProbe.drop_to_surface` — bounded ray cast. Unbounded, it landed on the
  **base underside 10–12 mm down** and refused 8 of 10 ordinary movements.
* `build_stage_tooth_interface` — the interface. Adaptive depth, bidirectional
  (cavity where penetrating, ramp where separated), bounded seat.
* `affected_region` — geodesic/surface ROI; bboxes are diagnostics only.
* `surface_deviation` + `_point_to_surface` — point-to-**triangle**.
* `rigidity_report`, `interface_continuity`, `transition_quality`,
  `old_site_quality`, `overlap_volume`, `bridge_between`, `components`.
* `validate_printable_stl` — the hard gate on written bytes.

### `api_core.py`

* **added** `import manufacturing as mfg`.
* **changed** `_manufacturing_tooth` — plug removed; returns the crown.
* **changed** `build_stage_bundle` — per-stage interfaces, adjacency check,
  add→(subtract)→union, export weld before write, hard gate on the reread,
  `require_print_ready` flag, full diagnostics, returns `blobs` + `base_mesh`.
* **added** `cast_probe` / `original_cast` above the stage loop.
* **added** `FinalExportRequest`, `POST /api/session/{sid}/export/final`.
* **unchanged**: `kinematic_matrix`, `center_of_resistance`, `_stage_clinical`,
  every rotation and matrix convention, the scanner frame.

### `frontend/src/App.jsx`

* **changed** "Export Printable Cast" → "Export Setup / Inspection".
* **changed** "Export Stages" → "Export All Stages".
* **added** `exportFinal`, `printVerdict` state, PRINT READY / NOT PRINT READY
  block with the failed gate and its measured value, `S.printReady` /
  `S.printNotReady`.
* **added** `opposingSessionIdRef`, published in an effect below its
  declaration — the drag path now sends the **same payload** as the typed path.
  Dragging previously omitted `opposing_session_id`, so the antagonist check
  silently did not run for a dragged movement and did for a typed one.

### `run_all_tests.py`

* **added** `("manufacturing iface", "test_manufacturing_interface.py")`.

### `test_manufacturing_interface.py` — NEW

33 tests, the 30-case matrix.

### Not modified, deliberately

`core_geometry.py` (kinematics, C_res, staging, scanner frame),
`stl_io.py`, `arch_frame.py`, `toothGizmo.js`.
