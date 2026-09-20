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

---

## U. 2026-09-20 — SIGNED-DISTANCE CORRECTION, SERIALISATION FORENSICS, CLEARANCE EXPERIMENT

### Files changed

| file | status | what changed | why | behavioural impact | tests |
|---|---|---|---|---|---|
| `bench_signed_distance.py` | **added** | independent signed-distance benchmark | the manufacturing gate relied on a method nobody had scored | chose the method on measurement, not habit | run directly |
| `manufacturing.py` | modified | `CastProbe` compacts + signs exactly; `manifold_status`; `collapse_short_nonmanifold_edges`; clearance measurement; `edges_post_read_before_weld`; narrowed verdict | see below | stage fragmentation fixed | `test_manufacturing_interface.py` |
| `api_core.py` | modified | clearance wiring, float32 weld, unwelded serialisation probe, single-body Manifold gate, recorded repair rung, narrowed verdict, clearance-before-bridge order | see below | per-stage gating and forensics | `test_manufacturing_matrix.py` |
| `test_manufacturing_matrix.py` | **added** | 18-case end-to-end matrix | interface-level tests cannot see whether a stage fuses | 13 pass / 5 fail | itself |
| `test_manufacturing_interface.py` | modified | +4 tests: ray-miss hygiene (AST-guarded), retry levels, probe compaction, unused-vertex invariance | brief | 37 pass | itself |
| `frontend/src/shadowRig.js` | modified | ortho box derived from perpendicular reach; `CATCHER_SPAN`, `FRUSTUM_SLACK` | catcher corners sat outside the shadow map | corners outside 3 → 0 | `verify-shadowrig.mjs` |
| `frontend/src/App.jsx` | modified | shadows gate + labelled toggle + `__viewportDiagnostics()` | viewport blanking on the third click | cast stays lit | `e2e/occlusal-darkening.spec.js` |
| `frontend/src/shadowHarness.js` | **added** | real-WebGL reproduction of the rig with a per-pixel cast mask | arithmetic checks cannot tell you the viewport is lit | A–F isolation | `e2e-shadow` |
| `frontend/shadow-harness.html` | **added** | dev-only page for the harness | never reaches `dist/` | — | — |
| `frontend/playwright.shadow.config.js` | **added** | dev-server config for the harness | keeps the diagnostic out of the shipped bundle | — | — |
| `frontend/e2e-shadow/shadow-darkening.spec.js` | **added** | component isolation, measured pixels | — | 1 pass | itself |
| `frontend/e2e/occlusal-darkening.spec.js` | **added** | real-workflow luminance regression on the real scan | the actual reported bug | 2 pass | itself |
| `frontend/verify-shadowrig.mjs` | modified | corner/off-axis containment replaces the side-length test | the old test passed while 3 of 4 corners were outside | catches it now | itself |
| `frontend/eslint.config.js` | modified | node globals for the new configs and specs | lint | clean | `npm run lint` |

**No dependency was added, removed or upgraded.** `manifold3d`, Open3D,
`trimesh` and SciPy remain the geometry stack. `trimesh.proximity.signed_distance`
imports `rtree`, which is not installed here; `closest_point_naive` needs no
spatial index, gives trimesh's real accuracy, and was benchmarked instead.

### The signed-distance benchmark, in full

* **Reference construction** — independent of all three candidates.
  *Magnitude:* exact closest-point-on-triangle (Ericson 5.1.5, written out in
  full rather than clamp-and-rescale, because a clamp-and-rescale version of
  exactly this was wrong earlier in this project) against EVERY triangle — no
  index, no candidate pruning. *Sign:* ray parity — a closed surface is crossed
  an odd number of times from any interior point — with three independent
  random directions voting, and any point where a ray grazes a barycentric
  boundary within 1e-6 reported AMBIGUOUS and excluded rather than guessed.
* **Points** — 870 surviving of 900, across **15 classes**: exactly on a
  triangle; 0.01 / 0.05 / 0.10 / 0.50 mm inside; 0.10 / 1.00 mm outside; steep
  cervical wall (|n·u_occ| < 0.25) at 0.05 and 0.30 mm in; cast underside;
  concavity (top decile of local curl); within 2% of an edge; within 2% of a
  vertex; over the arch opening; interproximal.
* **Old accuracy** — 77.7% sign-correct overall as shipped; **13.3%** on a
  steep cervical wall; 25.0% in a concavity; max magnitude error **6.66 mm**.
  With compaction alone: 91.4% / 90.0% / 76.7%, max error unchanged.
* **New accuracy** — **100%** sign-correct on every class; max magnitude error
  **0.0014 mm**. trimesh agrees at 100% / 0.0000 mm.
* **Open3D `nsamples`** — the docs ask for an odd value > 1 so a sign vote
  always has a majority. The cast is watertight by construction, so n=1 is
  already 100% on every class here; n=11 is used anyway because it costs
  0.17 ms on a rim and removes the dependence on that assertion holding for a
  cast some future change builds differently.
* **Performance**, warm-up excluded, 7,824-face cast: 44 points (a rim) —
  CastProbe-old 4.857 ms, Open3D n=1 0.362 ms, n=11 **0.529 ms**; 500 points —
  4.958 / 2.514 / 4.062 ms; 5,000 points — 5.916 / 5.895 / 13.501 ms. Scene
  build 0.52 ms, once.

**This is evidence, not proof for all possible geometry.** One synthetic cast,
one point sampler, one arch frame. Randomised and surface-near property
coverage is still to be added.

### Serialisation forensics

Measured at four points instead of two. On extrusion 0.25 mm stage 2:

| point | open | non-manifold |
|---|---|---|
| after the boolean, in memory | 0 | **0** |
| after our export weld | 0 | **3** (merged 20) |
| after STL write + reread | 0 | 3 |
| after the reader's weld | 0 | 3 (**merged 0**) |

**STL serialisation does not change the topology, and the downstream weld does
not either.** manifold3d's output is clean; welding the coincident-but-distinct
vertices it deliberately keeps is what creates the edges. `parse_stl_bytes`
already dedups by position, so an UNWELDED write rereads with 40 non-manifold
edges where our weld leaves 3 — the difference is the degenerate faces
`weld_vertices` drops and a plain reader keeps.

### Clearance experiment — implemented, measured, deliberately not emitted

Two forms were built: a uniform +0.05 mm dilation of the transformed crown, and
a depth-modulated version (+`clearance_mm` at the surface, −`fusion_overlap_mm`
past a 0.15 mm band). **Both fix extrusion 0.25 mm; both sever the tipping,
rotation and buccolingual cases into two bodies** — 8 of 18 matrix cases
failing against 5 without.

The deciding measurement is the graze fraction: `crown_points_in_graze_band /
deeply_buried` reads **113 / 14** and **121 / 11**. The crown sits in its own
filled socket, so nearly every crown vertex is near-coincident with the cast
and there is no buried region for the modulation to hold on to; clearing the
graze band clears essentially the whole interface and leaves only the seat to
fuse. `overlap_seat_cast_mm3` still read 51–67 mm³ throughout, because it is
measured against the PRE-clearance cast and cannot see the void — which is what
made the severance confusing until the graze count was added.

Recorded per tooth per stage as `clearance_tool_emitted: false`, with the band,
offset range, graze/deep counts and tool volume. What is missing before it can
be emitted is a connector guaranteed to bite the POST-clearance cast; enlarging
the seat to cover it would be the parameter sweep this work explicitly avoids.

### Verdict wording

`validate_printable_stl` no longer returns "PRINT READY". It returns
**PASSES / FAILS BOOLEAN/TOPOLOGY REGRESSION** plus a `gate_scope` naming what
it covers and, explicitly, what it does not: transition quality, interface
continuity, old-site quality, two-sided cast fidelity, ROI compliance,
seat/ramp exposure, crown rigidity, prescription consistency. The final-export
endpoint carries the same narrowed verdict and a `verdict_scope`.

---

## V. 2026-09-21 — THE SOLID WAS TOUCHING ITSELF; THE TRANSITION COLLAR, AND THE AGGREGATE GATE

### What the defect actually was, and how the last explanation was wrong

The five failing matrix cases were not five problems. They were one, and the
account in section U above was **refuted by its own control**.

**A coincident position in manifold3d's output means the boundary TOUCHES
ITSELF.** That is now established rather than assumed, by running the engine on
cases whose answer is known:

| union | coincident positions | non-manifold after weld |
|---|---|---|
| two cubes overlapping, axis aligned | 0 | 0 |
| two cubes overlapping, one rotated 13/27/41° | 0 | 0 |
| cube ∪ sphere (transversal) | 0 | 0 |
| two cubes meeting FACE TO FACE (tangent) | 0 | 0 |
| **two cubes meeting along an EDGE** | **2** | **1** |

So a clean transversal union produces none, a face tangency produces none, and
only a genuinely self-touching solid produces them. Binary STL stores
POSITIONS, so a reader welds whether or not we do — and welding a self-touch is
exactly what turns it into a non-manifold edge.

**"A small movement is worse than a large one" is FALSE.** Measured across the
whole extrusion sweep with seats and crowns fused, self-touch counts per stage:

| extrusion | stage 1 / 2 / 3 |
|---|---|
| 0.0 mm | 9 / 16 / 27 |
| 0.25 mm | 7 / 18 / 34 |
| 1.0 mm | 26 / 19 / 33 |
| 1.2 mm | 23 / 27 / 43 |
| 2.0 mm | 12 / 33 / 47 |

The 1.2 mm case — the one section U called clean across five stages — carries
the defect at every stage. It passed because every edge it produced happened to
be short enough for `collapse_short_nonmanifold_edges` to take. **The defect was
universal; the repair rung's success was not.**

### Where the solid touched itself, proven with provenance

Every input solid is now stamped with a manifold3d original id
(`Manifold.as_original()`), which survives the boolean and comes back on
`run_original_id` / `run_index`. So every triangle in a fused stage can be
traced to the geometry it came from, and every offending edge reported with the
sources meeting on it.

On extrusion 0.25 mm stage 1, all six touches read **distance-to-rim 0.08889 mm
and distance-to-nearest-cast-vertex 0.17222 mm — the same two numbers to five
decimals.** They lie on the boundary edge of the cast's flat flush socket cap,
at the midpoint of a rim edge. The rest sat on the crown's own cervical rim.

**All three geometries were built from ONE loop.** The crown is cut at
`socket_rim`, the old site is capped at `socket_rim`, and the connector was
lofted from `socket_rim` transformed — so the connector's wall crossed the cast
and the crown exactly where each has a sharp edge. A surface crossing another
surface AT ITS CREASE is a tangential contact.

### The fix: a transition collar that encloses the crease

The connector is no longer a plug lofted from the rim into the crown. It is a
bounded collar whose solid CONTAINS the crown's cervical crease and whose own
surface meets the crown and the cast only where both are smooth. Both rings are
placed by MEASURED signed distance against the actual transformed crown and the
actual cast, never by an offset chosen in advance:

* **upper ring** — starts `max(emergence_height_mm, local lift)` above the rim,
  then marches up while it is inside the cast and outward while it is inside
  the crown, until it is at least `clearance_mm` clear of both.
* **lower ring** — seeded `seat_bottom_outset_mm` outward, dropped onto the
  cast with a shrink ladder, then marched INWARD ALONG −grad(signed distance)
  until it is `fusion_overlap_mm + clearance_mm` inside real cast material.
* the radial direction is taken about the rim's own centroid and every offset
  is OUTWARD, which cannot fold a star-shaped loop.

CASE A (penetrating), CASE B (separated) and CASE C (mixed) all use this one
construction and need no partition, because both rings are solved PER RIM POINT
against the real surfaces.

**Six things had to be measured rather than reasoned, and each one was a
separate wrong answer first:**

1. **The upper ring must clear the CAST, not just sit above the rim.** On a
   penetrating tooth the cast surface is above the cervical margin, so
   `rim + 0.30 mm` is still buried. Every residual touch read
   distance-to-rim 0.30000 — exactly the starting height — until the ring was
   solved against the cast's signed distance instead.
2. **The lower ring must SEAT, not merely get inside.** Stopping at
   `clearance_mm` settled it 0.08 mm under the surface, so its bottom disc ran
   nearly PARALLEL to the flat old-site cap a fraction of a millimetre away —
   and the forensics named that touch `EMERGENCE_RECONSTRUCTION +
   OLD_SOCKET_REPAIR`. Measured on the 18-case matrix, everything else fixed:
   0.05 mm → 15 of 18, **0.30 mm → 16 of 18**, 0.60 mm → 13, 1.20 mm → 3. The
   value that works is `fusion_overlap_mm + clearance_mm`, which the policy
   already names.
3. **The downward march must follow the field, not the axis.** Descending the
   tooth's long axis is right where the cast beneath is a floor and wrong where
   it is a wall: on a steep interproximal face the axis runs ALONG the surface
   and 1.2 mm of marching never gets inside — a 1.2 mm extrusion was refused
   `wall_too_thin` on a cast measured 18.77 mm thick.
4. **The shrink ladder is load-bearing.** A lingual rim point seeded 1.2 mm
   outward lands OVER THE ARCH OPENING, where there is no cast: one point in 44
   fell through and the nearest surface was 3.93 mm away.
5. **The loft's end caps must be CONES.** A fan to the loop's own centroid is a
   dished surface whenever the loop is non-planar, and the upper ring is
   deliberately non-planar — 0.30 mm of rise at some points and 1.96 mm at
   others on one tipped tooth. The dish sagged BELOW the crease it was meant to
   enclose.
6. **The upper ring must clear the whole rim NEAR it, not only its own point.**
   `top[i] = rim[i] + rise[i]` follows the cervical scallop, so where the
   margin falls away the NEIGHBOURING ring point sits below the rim point
   beside it — measured, `top[k]` was 0.078 mm under `rim[i]`, and the two
   crease points left unenclosed were inside by only 0.025 mm and 0.014 mm. A
   running maximum over ±3 indices fills those local dips and leaves the ring
   low wherever the rim is genuinely low, so the collar does not become a tower
   to fix a notch.

**The collar rises as far as the tooth has lifted.** With a fixed 0.30 mm start
the upper ring stayed just above the cervical margin while the rim had risen
0.6 mm, so the whole emergence was compressed into the wall below it: measured
by `transition_quality`, **0.771 mm of a 0.946 mm fall landed in ONE 0.25 mm
ring — 82 % of it**, which is the "abrupt ring / vertical wall" the brief asks
to reject, and the gate rejected it. Rising with the lift spreads the same fall
across the crown's own flare instead of inventing a shelf to stand on: the same
stage now reads 0.312 mm of a 0.446 mm fall, share 0.699, not a ledge.

### CASE A: the crown-derived local clearance, and why it is withheld

The tool is an EXACT dilation of the transformed crown — a Minkowski sum with a
`clearance_mm` sphere, computed once per tooth on its T0 crown and carried by
the same stage matrix the crown gets, because dilation commutes with a rigid
transform. Measured on a 250-triangle crown: 8 segments cost 325 ms and 12 cost
520 ms for volumes 49.836 and 50.107 mm³, a 0.5 % difference on a 0.05 mm
clearance. The array alternative — a vertex-normal offset — gives 46.86 mm³ and
**can fold**: it showed up in a fused stage as a `LOCAL_CLEARANCE` self-touch,
which is the normal-offset failure section 8 tabulated, in a third place.

**It is emitted only when the measured guard passes.** The subtraction is
performed, `decompose()` is counted, and the result kept only if the cast is
still ONE body — because subtracting the dilated crown fragmented the cast into
**4 bodies** on extrusion 0.25 mm stage 1, where the crown passes through the
thin lip between the old socket cap and the gingival wall. A withheld tool is
recorded with its reason, never silently dropped, and
`interface_total_cavity_volume_mm3` now reports what was ACTUALLY REMOVED
(measured as a volume difference) rather than the volume of a tool that was
never used — which told a lab 85.16 mm³ of cast had been excavated on a stage
where nothing was cut at all.

### Four measurement defects that were failing gates on correct geometry

* **`_point_to_surface` was an approximation calling itself exact.** It took
  the `candidates=24` triangles whose CENTROIDS are nearest, and on a cast
  whose underside triangles are 20 mm across the triangle a point actually sits
  on is not among them: 793 of 4710 fused-stage vertices reported distances of
  exactly 9.0000 mm — the distance to whatever unrelated triangle made the
  shortlist — while the cast reproduced through manifold3d to 2e-6 mm. Now
  Open3D's BVH, exact, with the candidate search as the fallback.
* **The fidelity reference carried 4497 phantom vertices.** `bv` legitimately
  keeps every vertex the trim removed (rule 3.1 forbids rebuilding the scan's
  array). Sampling them as surface points reported **3.39 mm of "cast
  deformation"** from points that are not on the cast at all. The comparison
  copy is compacted; the scan's array is untouched.
* **The old-site height field was reading the cast's UNDERSIDE.** A
  nearest-centroid query in the rim plane happily returns a triangle on the
  bottom of the base: the old site read a **crater 22.58 mm deep** — the cast's
  own thickness — and 13 disconnected patches. Faces are now restricted to
  those facing occlusally before anything is measured.
* **An old site the tooth still covers is not a defect.** A barely-moved tooth
  stands over the site it will leave, so the finished model's surface there is
  the CROWN and the nearest cast triangle is on the socket wall. Reading its
  height reported a 2.4 mm crater at a site nothing had touched. Samples with
  no cast surface within 2.5 grid cells are reported as obscured; below a
  quarter of the site assessable, the answer is `assessable: false` with the
  obscured fraction — a determinate answer, not an unchecked one.

### The aggregate manufacturing gate

`validate_printable_stl` answers a narrow question and says so in its own
`gate_scope`. **`mfg.aggregate_print_gate` is now the only thing in this
codebase entitled to say PRINT READY**, and it is a pure function of the stage
record, so a measurement that was never taken fails it exactly as a bad one
does — `NOT_CHECKED` is not `CLEAR`. Sixteen gates:

written-STL topology · single positive Manifold body · body count agrees with
the STL · **no self-touching boundary** · synthetic exposure within bound · no
exposed clearance wall · two-sided unaffected-cast fidelity · reconstruction
inside the envelope · every interface built · interface continuous around every
rim · no transition ledge · old site restored · crown is an exact rigid
transform · gingival bridge preserved · root-length independent · clinical
consistency.

An empty stage record fails **all sixteen**, which is asserted.

`/export/final` enforces this gate, not the narrow one, and its manifest carries
the gate's own answer rather than a hard-coded `print_ready: True`. The client
reads `X-Print-Ready` from the response instead of inferring readiness from
HTTP 200.

**Exposure is measurable only because of provenance.** A triangle that survives
to the fused boundary and came from the connector IS exposed synthetic anatomy;
one buried by the crown or the cast is simply absent from the output. No
heuristic. Measured on extrusion 0.25 mm stage 1: 8379.20 mm² cast, 164.70 mm²
connector, 6.87 mm² old-site closure, 84.59 mm² crown — **1.99 %** synthetic,
0.00 mm² unattributed, and 0.00 mm² of clearance wall.

### The allowed reconstruction envelope, enforced

`affected_region` is no longer diagnostic-only. The envelope is the union, over
every moved tooth, of the cast surface within `roi_radius_mm + seat_bottom_outset_mm`
of the TARGET rim and of the T0 rim — target site and old site, the only two
places this pipeline may change the cast. Every cast point that moved further
than `max_unaffected_deviation_mm` must lie inside it, and outside it the cast
is compared BOTH ways by point-to-TRIANGLE distance, never by vertex index.
Measured on extrusion 0.25 mm: final→original max **0.0 mm**, original→final max
**0.0 mm**, 0 modified points outside the envelope.

### Order independence was testing the wrong thing

The old test reversed the PRESCRIPTION LIST — `[d_oa=1.0, d_md=0.8]` against
`[d_md=0.8, d_oa=1.0]` — which does not reverse an order, it gives the two teeth
each other's movement. That is a different model, entitled to a different
surface: measured, the two "orders" differ by **0.580 mm**, while their VOLUMES
agree to 5e-4 because the total material is much the same. **A volume
comparison could never have caught it.**

`build_stage_bundle` now takes `part_order` — forward, reverse or sequential —
which changes ONLY the sequence the identical solids are handed to the boolean
in. The test compares the written STLs by two-sided point-to-triangle distance,
topology, body count, volume and ROI compliance across all three.

### Adjacent, crowded and converging teeth

The bridge check compared the gingival bridge with `2 * fusion_overlap_mm` — a
nominal 0.5 mm describing no geometry that was ever built, so it passed for a
trench of any width. It is now the **minimum distance between the two connector
solids**, and the connector clamps its own reach: each may take
`0.5 × max_bridge_removal_fraction × d_neighbour − 2 × clearance_mm`, derived
from the rule rather than chosen, so the pair can never take more than half.

**The fraction GATES; only a merged pair is REFUSED.**
`max_bridge_removal_fraction` is 0.50 with no measurement behind it, and
CLAUDE.md section 14's rule is explicit — promote a threshold to a refusal only
with numbers behind it. Refusing on it would block ordinary crowding: measured
on two teeth 2.36 mm apart the pair takes 50.9 % and leaves 1.16 mm of
interdental tissue, which is a papilla, not a trench. So it fails the aggregate
gate, where it is visible and named, and two reconstructions that have actually
merged are refused outright.

### What is measured per stage now

Per stage: self-touch groups and their source pairs · edge forensics (edge id,
endpoint coordinates, length, incident face ids, areas, normals, provenance,
at each of the five points of the serialisation chain) · synthetic exposure by
source · two-sided cast fidelity · ROI compliance · volumes separated into
original cast / after clearance / removed / connectors / crowns / crown-cast
overlap / connector-cast overlap / fused · adjacent bridges · clinical
consistency · per-phase seconds. Per tooth: interface mode, continuity,
transition quality, old-site quality, rigidity (edge lengths, sampled pairwise
distances, triangle areas, RᵀR−I, det−1), clearance decision and reason.

The unwelded serialisation probe is now GATED. Benchmarked on a 203,522-face
mesh — the real cast base is 190,036 — the write, parse, weld and two reports
cost **0.90 s PER STAGE**, so a 31-stage plan paid ~28 s of forensics on top of
a 4.2 s export, and `/export/final` paid it again. It answers a question that
only arises when something is wrong, so it runs when something is wrong, and it
is wrapped: a diagnostic must never be the thing that fails an export.
