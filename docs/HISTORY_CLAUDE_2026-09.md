# PROJECT ALIGNER: MASTER ARCHITECTURE & MEMORY FILE

## 1. THE CORE MISSION
We are building a commercial-grade, professional clear aligner CAD application. 
**Our benchmarks are Deltaface and 3Shape Ortho System.** 
This is NOT an artistic 3D animation sandbox; it is a clinical orthodontic tool. 
Teeth must move biologically. Meshes must be mathematically watertight for 3D printing and thermoforming. Code must be structured for enterprise deployment.

## 2. THE TECH STACK
*   **Frontend:** React, Vite, Three.js. 
    *   *Key UI:* Deltaface-style 3D Transform Gizmo (`TransformControls`), spatial grid-based Brush tools, dynamic UI sliders.
*   **Backend:** Python, FastAPI, NumPy. 
    *   *Key Logic:* AI segmentation (ToothGroupNetwork), geodesic flood-fill boundary detection, 4x4 homogeneous transformation matrices, boolean mesh sealing.

### LIVE ENTRY POINTS — everything else is legacy or dead

| | Live |
|---|---|
| API | **`api_core.py`** — `uvicorn api_core:app`, 33 paths. Launched by `start_backend.bat`. |
| Client | **`frontend/src/App.jsx`** — `npm run dev`, launched by `start_frontend.bat`. |
| Geometry | **`core_geometry.py`** — pure NumPy/SciPy, headless, the engine. |
| Segmentation | **`segmentation_providers.py`** — the registry. `tgn_bridge.py` and `crosstooth_bridge.py` are the two model bridges; `/segment?provider=` picks one (§25). |
| Docs | **`CLAUDE.md`** (this file) and **`README.md`**. No other document is authoritative. |

`app_ui.py` is a legacy PyQt6 desktop shell over the same engine — it runs, but it is not the
product. **`_archive/` is dead code** (see `_archive/README.md`); as of 2026-09-16 it holds
`server.py`, `HANDOVER.md`, two stale `App.jsx` copies, and the empty root `config.py`/`models.py`
that used to shadow the real `tooth_segmentation/` modules.

### TEST COMMANDS

```
python -m compileall .                  # syntax, whole tree
python check_structure.py               # undefined names without importing (99 files)
python run_all_tests.py                 # CANONICAL runner — 40 entries
python -m pytest -q                     # runs alongside; both must pass
python bench_signed_distance.py         # scores the signed-distance method
python real_scan_regression.py          # the real scan, end to end — see §23, §24.7
python benchmark_providers.py           # every segmentation model, one scan — §25
python verify_pointops.py               # the CPU pointops shim, against upstream - s26.1
python -m pytest test_click_to_select.py        # click a tooth, get that tooth - s26.4
python -m pytest test_segmentation_mapping.py   # labels belong to the mesh — §24.3
node frontend/verify-kinematics.mjs     # cross-language kinematics pin
node frontend/verify-palette.mjs        # 32 distinct FDI colours — §24.5
cd frontend && npm run lint && npm run build && npm run smoke
```

`npm run smoke` is not optional: a `vite build` succeeds on code that throws during render, and a
dependency array referencing a later `const` has already white-screened this app twice.
**Every suite entry now carries assertions** — the three characterization scripts were
hardened on 2026-09-16.
`npx playwright test` (in `frontend/`) drives the production build; `node frontend/verify-kinematics.mjs`
pins the browser against the backend at 1e-12 (observed 1.78e-15).

## 3. ARCHITECTURAL NON-NEGOTIABLES
Read these rules carefully before writing any code:
1.  **Scanner Coordinates are Sacred:** NEVER rotate, re-center, or re-scale the raw STL mesh vertices. Inter-arch bite registration depends on the raw scanner space. To establish "Up" and "Forward," we use an **Occlusal Plane Reference Frame** (`arch_frame.py`) stored as session metadata, not by shifting the geometry.
2.  **Biological Kinematics Only:** Teeth do not move around their geometric center. They move around their biological Center of Resistance ($C_{res}$), located ~10-13mm down the root into the bone.
3.  **Orthodontic Axes:** The 3D Gizmo must operate in the tooth's local anatomical space:
    *   *Tip (Angulation)* = rotation about the buccolingual axis.
    *   *Torque (Inclination)* = rotation about the mesiodistal axis.
    *   *Rotation* = rotation about the apical (long) axis.
4.  **Matrix Conventions:** Three.js uses column-major matrices; NumPy uses row-major. Matrices passed from the backend must be transposed or correctly mapped to avoid perspective-divide explosions (vertex shearing).
5.  **Watertight Geometry:** All extracted crowns and remaining bases must be capped and closed cleanly (`cap_and_close`) to guarantee printable `.stl` files.

## 4. DEVELOPER COMMUNICATION PROTOCOL
*   **No Robotic Fluff:** Do not say "Here is your code" or "I understand." Start your response directly with the architectural solution or the code.
*   **Think Like a Tech Lead:** If I ask for a fix that breaks a load-bearing architectural constraint (like rotating the global mesh), push back and provide the mathematically sound alternative.
*   **Precise Code:** Deliver complete, functional blocks of code. Do not use placeholders like `// ...existing code...` unless the file is massive and the insertion point is explicitly clear.

## 5. RESOLVED: THE TETHERBALL GLITCH
**The Bug (fixed):** manipulating a cut molar with the 3D Gizmo swung the tooth in a massive arc through empty space instead of pivoting inside its socket.

**The Cause:** `derive_frame_from_region` took the long axis as crown centroid minus rim centroid. On a short broad molar that difference collapses to ~1.5mm while an uneven wand selection drifts the crown centroid sideways by as much, so the axis pointed sideways and $C_{res}$ was extrapolated ~9mm along it, landing outside the tooth.

**How it was fixed:**
1. `resolve_long_axis` (`core_geometry.py`) takes the axis from the **normal of the plane fitted through the cervical rim**, signs it against the arch's `u_occ`, and clamps it to `MAX_AXIS_DEVIATION_DEG` (20°) of the arch apical direction. Genuine tooth inclination survives; noise cannot.
2. `/cut` retrieves `arch_frame` from the session and **refuses with 409** if the occlusal plane was never established — the apical direction is never guessed.
3. `center_of_resistance` projects the crown centroid onto the axis through the rim centroid, so the pivot sits on the socket axis by construction (lateral offset 0.00mm) at unchanged depth.
4. `precompute_arch` routes through the same helper, so the auto-segmentation path cannot regress separately.

**Two rules learned here, worth not relearning:**
* **Do not gate the rim plane fit on singular-value "planarity" (`s2/s1`).** It is ANTI-correlated with the error. A cos(2θ) margin scallop drives it to 0.75 with a perfectly exact normal, while a 120° partial rim scores 0.041 with 43° of error. The only legitimate singular-value gate is `s1/s0` for a rim that is a *line* rather than a ring. The arch clamp is what catches real error.
* **Verify a fixture reproduces the bug before trusting a regression test.** `tooth_on_base` builds Z on a uniform XY grid, so a selected region's lateral centroid is fixed by its SHAPE regardless of the height field — reshaping the dome cannot reproduce a centroid drift. The drift must come from a lopsided *selection*.

Covered by `test_kinematics_frame.py` and `test_cut_endpoint.py`.

## 6. RESOLVED: PHASE 2 — EXTRACTION, MULTI-TOOTH STATE, HEALTH

**Ghosting (fixed).** `/cut` returned only the crown, and the client faked extraction by painting the selected vertices dark, so the tooth's geometry stayed on the cast. `/cut` now returns `removed_faces` plus a `socket_cap` fan, and the client rewrites the arch's **index buffer only**.

**The architectural rule this established — do not rebuild the arch.** The session's `verts`/`faces`/`edges`/`graph`/`concavity` are never mutated. Extraction is tracked as a cumulative `extracted_faces` mask over the original faces. Everything precomputed at upload stays valid, and every vertex id the client holds keeps its meaning.

**Why the base is not re-capped per cut.** Measured on synthetic arches, `cap_and_close` on a full base: **4.2s at 96k faces, 9.2s at 204k, 16.8s at 351k** (19.4s with the boundary walk and manifold check). `/cut` was paying that on every cut. *(These figures supersede an earlier 12.5/25.9/38.6s reading that is not reproducible — see §8.)* Since the base is the original mesh *minus faces*, the client already holds every vertex — sending face indices costs ~15k ints and 0.002s. Verified end to end: **/cut on 203k faces went from ~29.7s to 2.85s.** The socket is closed with a centroid fan (every new edge ends at a new vertex, so it cannot make an edge non-manifold and needs no O(mesh) check). Full watertight capping now lives in `/export`, once, off the interactive path — CLAUDE.md rule 5 still holds for anything that reaches a printer.

**Multi-tooth persistence (fixed).** `ToothGizmo.detach()` did `crown.matrix.identity()`, and `attach()` calls `detach()` — so cutting tooth 2 snapped tooth 1 back to T0. `detach()` now bakes `deltaMatrix()` into the crown, and `attach(mesh, frame, cRes, delta)` resumes: with `P0 = T(C_res)·R(frame)` the pivot starts at `delta·P0` while the crown keeps `P0⁻¹`, since `(delta·P0)⁻¹·delta = P0⁻¹`. So `deltaMatrix()` still returns the total delta from T0 and drags compose. Verified to machine precision: resuming displaces the tooth 8.9e-16 mm. Clicking a cut crown re-attaches the gizmo at its current pose.

**Health.** `make_consistent_winding` now runs on the **crown only** (0.34s) — never the base, which is a full-mesh Python BFS. `disposeMesh()` frees geometry *and* material everywhere (materials leak separately, and only geometry was being disposed). `BrushIndex` is built from vertices referenced by `geometry.index`, so an extracted tooth cannot be painted. `frameArch`'s inverse power iteration bails to a safe axis instead of letting NaN reach `camera.up`.

New endpoints: `POST /export` (watertight STLs), `GET /teeth` (committed poses, so a reloaded client can restore the case). Covered by `test_cut_endpoint.py` (11 tests).

## 7. RESOLVED: PHASE 3 — PRECISION CONTROLS & CLINICAL RENDERING

**Two-way numeric inputs.** Section 4 is now six `ClinicalInput` fields (tip/torque 0.5°, rotation 1.0°, translations 0.1 mm) with −/+ steppers and arrow-key nudging. `ToothGizmo.setClinical()` is the inverse of `read()`: values are **absolute from T0**, so a typed prescription is rebuilt whole rather than nudged. Typed and dragged movements share one persistence path (`rec.delta` ← `gizmo.deltaMatrix()`, then POST `/kinematics`).

**Three convention bugs this exposed, all measured, none of which a single-axis move would reveal:**
1. **`read()` used Euler order `ZYX`.** The backend composes `R_torque(u_MD)·R_tip(u_BL)·R_rot(u_OA)`, which in the pivot's local frame is `Rx·Ry·Rz` — three.js calls that `XYZ`. A (−12.5°, 7.5°, −4.0°) prescription round-tripped as (−11.8°, 8.5°, −5.7°): **1.71° of pure decomposition error.**
2. **`read()` reported the delta's raw translation column.** That column is `c + t − R·c`, not `t`, so a **pure 8° tip showed 0.94 mm** of mesiodistal translation that did not exist. Under two-way binding that phantom value feeds back and really moves the tooth.
3. **The socket fan was unoriented.** `boundary_loops` walks the rim either way, so triangles faced occlusally at random and `DoubleSide` drew them lit from behind — the dark smudge. A *global* winding flip is not enough: a scalloped margin is non-convex, and the test caught **2 of 137 triangles** still reversed. Orientation is now per triangle against `u_oa`.

**Cross-language pinning.** `frontend/verify-kinematics.mjs` (run `node verify-kinematics.mjs`) checks `setClinical`/`read` against golden matrices from `cg.kinematic_matrix` — agreement 1.8e-15, round-trip exact. `test_kinematics_frame.py::test_matches_the_javascript_golden_matrices` asserts Python still produces those same numbers. **Update both files together, never one alone**; otherwise the browser and the exported STL drift apart silently.

**Rendering.** `MeshPhysicalMaterial` (roughness 0.35, metalness 0.05, clearcoat 0.6) — clearcoat is a *Physical* property, `MeshStandardMaterial` does not have it. Socket caps get their own `flatShading` material so the cervical margin stays a crease. Lights are parented to the **camera**, not the scene: scanner axes are arbitrary, so a world-fixed key light lands somewhere different on every case. ACES tone mapping plus a `RoomEnvironment` PMREM give the clearcoat something to reflect.

## 8. RESOLVED: PHASE 4 — SOCKET TOPOLOGY & EXPORT DELIVERY

**The socket defect was DEPTH, not self-intersection.** The obvious diagnosis — a scalloped, non-convex margin makes the centroid fan fold — is wrong, and a test written against it would assert something false. A radial scallop `r = R + A·cos(kθ)` is star-shaped about its own centroid for *any* amplitude, so the fan cannot cross itself: measured **0 crossings** even at `r ∈ [0.5, 15.5]`, and on a realistic 3D margin the fan stays angularly monotonic about its apex (adjacent-normal dot ≥ 0.996 at ±4 mm of vertical scallop). The fan's apex *is* the rim centroid, so the cap landed on the margin plane at **depth 0.00 mm** — a flat lid over a hole.

**Z-fighting was a real second contributor, measured not assumed.** Of the flat cap's 700 triangles, **190 sat within 0.25 mm laterally and 0.05 mm in depth of a surviving arch face**. The carved cup has **0 of 1636**. So the depth fix resolves both the geometry and the shimmer.

**`build_socket_cup`** fits the rim plane (no `s2/s1` gate — see §5), ear-clips the floor in NumPy (not `Delaunay`, which spans concavities), and bridges concentric loops down to `depth_mm`.

**Three things that had to be measured rather than reasoned:**
* **Inset by DISTANCE, not by scale.** `ring_k = c + (1−k/rings)·(p−c)` looks equivalent and is not: it shrinks the scallop's amplitude along with the radius, so on `r = 8+1.5cos(4θ)` ring 1's peaks (7.13) push past ring 0's troughs (6.50) — **26 crossing triangle pairs**. Subtracting a constant radius keeps the amplitude, and nesting becomes unconditional for any inset below `r_min`.
* **Do not gate on angular monotonicity.** A real rim is a clean ring (radius 4.01–5.11) but its boundary loop carries small backward angular steps from the triangulation, *and* `boundary_loops` may return it clockwise. An earlier star-shape test closed the loop with `+2π`, so every clockwise rim was declared non-star and fell back to a flat floor. The guard now validates the loops it would actually build (`polygon_is_simple`).
* **Orient the cup GLOBALLY, not per triangle.** Per-triangle orientation was right for the flat fan and is wrong for a cup: side walls are near-parallel to `u_oa`, so their dot against it is noise. Winding is consistent by construction, then flipped once using the floor, where the test is unambiguous.

**Export never had a download path.** It wrote STLs to a server directory and returned JSON paths — no blob, no anchor, no `Content-Disposition`, so nothing could reach the clinician however long they waited. The "hang" was that plus ~40 s of silence. `/export` now streams an in-memory ZIP (base + one STL per tooth + a self-describing `manifest.json` carrying FDI, prescription, staging and clearance) and the client turns it into a download; the status line states the ~40 s up front. It stays a sync `def` deliberately — **FastAPI already dispatches sync endpoints to a worker thread, so `run_in_threadpool` would fix nothing** and would break the tests that call it directly. `write_binary_stl(verts, faces, path) -> int` is unchanged; `write_binary_stl_bytes` shares one `_stl_blob` implementation.

**The display cup never reached a printable file.** `/export` built the base with `cap_and_close`, which closed the socket its own way, so what the clinician saw was not what the lab printed. *(Resolved in §9: `/export` now replays the exact cup `/cut` stored.)*

Measured: socket cup **43.6 ms at 150k faces** (24.4 ms cup + 19.2 ms depth clamp), 71.8 ms at 204k, 104.2 ms at 351k. `/export` **10.8 s at 150k**, 14.1 s at 204k, 25.2 s at 351k.

**TIMING CORRECTION — the 38.6s figure was wrong.** §6 recorded `cap_and_close` at 12.5/25.9/38.6s for 96k/204k/351k faces. Re-measured on the same fixture, isolated *and* in the original sequential shape, it is **4.2/9.2/16.8s** — a uniform ~2.3x, not a fixture difference. The only change to that path since was `full_matrices=False` on the cap's SVD, and toggling it back accounts for none of it (17.2s vs 18.4s, i.e. noise). Most likely machine load during the original run. The 25.2s total `/export` at 351k is consistent with 16.8s of capping plus the boundary walk, STL writing and zipping. **The conclusion is unchanged and the ratio is still overwhelming — ~17s against ~0.002s for the face-index path — but do not cite 38.6s.**

**The inset does NOT nest unconditionally.** An earlier docstring claimed it did. Radial inset is safe strictly below the rim's minimum inward clearance and folds above it: on `r = 8 + 6cos(4θ)` (clearance 2.0mm) it gives **0 crossings at 1.9mm and 8 at 2.0mm**. Total inset is now capped at `SOCKET_INSET_FRACTION` (0.70) of measured clearance, reported as `info["min_clearance_mm"]` / `info["inset_mm"]`. The three offset operations differ and must not be swapped — measured self-crossings at 1/2/3mm on `r = 8 + A·cos(4θ)`:

| A | radial (used) | normal offset | scale |
|---|---|---|---|
| 1.5 | 0 / 0 / 0 | 0 / 0 / 4 | folds at 26 pairs |
| 4.5 | 0 / 0 / 0 | 0 / 4 / 12 | — |
| 6.0 | 0 / 8 / 24 | 0 / 12 / 16 | — |

**The guard had a hole at exactly the pinch.** `polygon_is_simple`'s prefilter is **spatial** (it compares each edge's x/y extents; the index arithmetic only *excludes adjacent* edges), so index separation is irrelevant — the real crossings it must catch sit at separations of 39 and 79. What it missed was the collapse itself: at inset = clearance the trough vertices land on the centroid, edges reach zero length, and the signed-distance predicate returned 0.0 for every test and called a pinched polygon simple. It now rejects zero-length edges and coincident non-adjacent vertices first.

**`flat_fallback` is no longer dead code.** Nothing reached it — scallops to amplitude 7.6, C-shapes with 0.4mm walls and spirals all cup — so `MIN_SOCKET_INSET_MM` (0.25) now sends rims with too little clearance to the flat floor rather than building a 0.07mm "cup" that is a flat lid by another name. `test_flat_fallback_runs_end_to_end_on_a_pathological_rim` reaches it through a margin necked to its own centroid.

**`info["star_shaped"]` was renamed.** It read False on rims that cup perfectly well (a C-shaped margin does) and was no longer the gate. `inset_validated` is what decides; `angularly_monotonic` is the descriptive measurement.

**A scan can arrive unprintable, and `/export` used to blame the cut.** `condition_mesh` welds, drops degenerates and debris and fills small holes — it never repairs an edge shared by three or more faces, which intraoral scanners routinely produce. One such edge blocked export forever with a message about "a cut left a second open loop". `manifold_report` now separates open edges from non-manifold ones and `scan_health` is recorded at upload. *(The `allow_unsealed` escape hatch added here has since been deleted — §9 explains why the diagnosis behind it was wrong.)*

## 9. RESOLVED: PHASE 5 — THE VIRTUAL CAST BASE

**`/export` was handing the lab a membrane, and the non-manifold hypothesis in §8 was refuted by the shipped manifest.** Measured on the real mandibular scan: `open_edges 2101`, `nonmanifold_edges 0`, `holes_found 1`. Zero, before and after. The scan is an **open shell**, as every intraoral scan is; those 2101 edges are the perimeter of the scanned region, not damage. `cap_and_close` sealed that perimeter by fanning across it — and an arch perimeter is a **horseshoe**, so the cap spans the U-shaped tongue opening. Measured: `cap_boundary_loop` returned **2104 triangles for a 2101-vertex loop** (a full hull triangulation) in **100.2 seconds**. Topologically closed, anatomically a web over the tongue space.

**`cg.trim_to_arch` → `cg.build_cast_base` replaces it.** Trim the scan to a horseshoe band about the fitted occlusal ridge, extrude that band to a flat-bottomed solid, assert zero open and zero non-manifold edges directly. `cap_and_close` keeps the crown, where the cervical rim genuinely is a hole to fill, and never touches the base again. **Measured end to end on the real scan: trim 0.6s + extrude 0.5s, `/export` 7.5s including the round-trip check, base 190,036 faces, volume 33,953 mm³, 0 open / 0 non-manifold / winding consistent.**

**Three of the brief's own prescriptions were wrong, all refuted by measurement:**
* **`margin_mm = 3.0` destroys the dentition.** Vertex distance-to-ridge percentiles [50/75/90/95/99/max] = **[2.8, 4.7, 6.5, 7.3, 9.3, 11.2] mm**; a 3mm margin keeps **52.8%** of vertices and slices every crown lengthwise. `ARCH_TRIM_MARGIN_MM` is **9.0** — a molar half-width (~5.5mm) plus a ~3mm gingival collar.
* **"Highest points along −u_occ" has the sign backwards.** `u_occ` points *out of the mouth*; the ridge is at max **+u_occ**. The −u_occ extreme is the floor of the mouth.
* **"Assert exactly one boundary loop" is not true of what a face predicate produces.** At margin 7.0 the largest component came out with two loops (2225 + 81) — an interior **hole**, not a disconnection, and no margin reliably avoids one. `trim_to_arch` fills every loop but the longest, which makes the guarantee structural.

**Task B could not work as written: the projected rim is not a simple polygon.** Crossings in the occlusal projection: raw perimeter 11, trimmed at margin 8 → 42, at margin 6 → 71. All **local** (index separation ≤ 38 of ~2200, height gaps 0.4–1.3mm): the boundary stepping over itself on a steep wall. Ear-clipping it returns 2052 of 2157 triangles, so the floor has holes and the base has open edges. Three repairs were measured before the one that worked:

| repair | result |
|---|---|
| 2D Laplacian smoothing | **fails** — 40 passes, vertices moved 0.81mm, still not simple |
| offset outline (rails at ±margin about a fitted curve) | **fails** — ridge curvature radius 10–12mm against an 8–11mm margin, so the lingual rail folds; polynomial fits of degree 3–6 all non-simple |
| flattened copy of the top's connectivity | **rejected** — manifold by construction, but **11.6% of the scan area is undercut** so the bottom self-intersects and `manifold3d` would reject it |
| **decimate the outline, then prune crossings** | **works** — every case ear-clips first time |

**The decisive step is decimation, not a cleverer repair.** The bottom of a cast is an outline, not a record of the scan. At `FLOOR_OUTLINE_SPACING_MM` (0.5) every case tried triangulates on the first attempt — a 2145-point real rim, a 546-point synthetic one, an 84-point coarse one — and at half that spacing all but one fail.

**Five things that had to be measured rather than reasoned, and that no count-based check would have caught:**

* **`polygon_is_simple` passing is NOT the same as being triangulable.** It tests proper crossings with a normalised epsilon and happily accepts polygons `ear_clip_polygon` bails on — measured, a rim that passed still clipped to 279 of 438. The success condition is now the ear clip itself.
* **The blocker is a NEEDLE, not a collinear run.** That was the first guess and it was wrong: the boundary spikes out and back so a vertex's two neighbours nearly coincide, the tip is far from the chord joining them, and a distance-to-chord test scores it perfectly fine while the ear triangle is degenerate. `MIN_EAR_THINNESS` scores `2·area / longest_side²` — dimensionless, so one threshold serves a coarse mesh and a dense one.
* **Escalating a degeneracy epsilon is actively harmful.** It amplified a **1e-15 rounding difference into a 0.6% volume change under a pure rigid transform**, because flipping `ear_clip_polygon`'s own `1e-12` comparison jumped the tolerance a full decade. Spacing escalates instead: geometric, and each attempt re-decimates from the original so nothing compounds. With that removed, 500 random rigid transforms give **bit-identical** bases.
* **`argmax` per angular bin makes the ridge discontinuous.** Two near-equally-high vertices swap under a floating-point perturbation and the control point jumps across the bin. `fit_arch_curve` takes a height-**weighted mean** within `RIDGE_BAND_MM` of each bin's summit, so a vertex enters with weight zero and grows smoothly.
* **Cut the ridge at its largest angular gap.** An arch is an open horseshoe; ordering bins from −π to +π without that cut leaves one long chord leaping across the posterior opening, and every face in the tongue space then measures "close to the curve" and survives the trim. `arch_walk` uses the same rule for the same reason.

**The membrane is now structurally impossible, and the test proves the fixture reproduces the bug first.** The floor is the ear-clipping of the rim's own horseshoe outline, and every floor vertex *is* a projected rim vertex (pruning only deletes), so the floor lies inside the convex hull of the rim's projection and can never reach further from the arch than the rim does. `MAX_PRUNE_ARC_FRACTION` (0.15) is what keeps that true: deleting an arc leaves a chord, and a long chord *is* the web. Every crossing measured is local — max arc 0.10 synthetic, 0.03 real. Measured: `cap_and_close` puts 268 of 546 cap triangles over the opening out to 15.3mm; trim → extrude puts **0 of 10,852**.

**Four bugs that reported themselves as clean.** Edge multiplicities are orientation-blind and STL stores positions, so a base can measure watertight and still be wrong. `_winding_is_consistent` (every directed edge exactly once) and a **re-read of the written STL** are what caught these:

1. **`_floor_triangulation` returned CCW triangles regardless of the rim's winding**, so on a clockwise rim the floor and the wall agreed on their shared edge — 229 directed edges in agreement on one sealed socket, zero open and zero non-manifold the whole time. Latent in `build_socket_cup` since Phase 4.
2. **`build_socket_cup` emitted one floor vertex per rim vertex.** A folded margin projects two rim points onto one, so two interior vertices shared a position; a reader welds them and takes their faces with them. The interior is now built on the **unique** projections and bridged through an `image` map, exactly as the cast wall bridges to its decimated floor.
3. **A pinch vertex must be DELETED, not duplicated.** Duplication is the textbook repair and is wrong here for a reason only visible at the far end: the two copies share a position and an STL reader welds the pinch straight back. `_open_pinch_vertices` deletes the smaller fan instead.
4. **The socket rim itself can pass through one vertex twice** (239-vertex rim, one vertex visited twice on a real molar), so the cup built the same spoke from both visits. `/cut` now takes the largest **simple** sub-cycle; the spur stays open and the export's hole fill caps it.

Also fixed while in here: **the arch curve must be fitted on the ORIGINAL scan.** Refitting at export time reads a ridge with the extracted teeth missing — measured on a two-tooth fixture, the trim then discarded 351 of 423 socket-cup vertices while still reporting a watertight base. And **`cap_and_close`'s fan could make its own spokes non-manifold** when a boundary walk pinched; its docstring claimed it could not.

**`allow_unsealed` is deleted.** It existed to get past a base `cap_and_close` could not seal, which was a misdiagnosis. A trim-and-extrude base closes by construction, so an unsealed one is a bug to fix rather than a condition to wave through.

**Per-tooth root length.** The shipped manifest had FDI 33 — a mandibular canine — on `root_length_mm 10.0` where Wheeler gives 13, because `App.jsx` held one slider for the session and `CutRequest.root_length_mm` defaulted to 10.0. `/segment`'s labels are now kept client-side, the modal FDI over the selection derives the root length per tooth via `rootDefaultForFDI`, the sidebar shows what it was derived from, and **`CutRequest.root_length_mm` is required with no default** — C_res is extrapolated along the long axis by exactly this distance, so a silent default is a silently wrong pivot.

Covered by `test_cast_base.py` (9 tests) and `test_cut_endpoint.py` (18).

## 10. RESOLVED: PHASE 5 — TRIM CORRECTNESS, STAGING, MANUFACTURING EXPORT

**The floor was throwing away 78% of the rim, and it never needed to.** `prune_to_simple`
decimated the projected outline to 0.5mm spacing before triangulating it. That is not a
triangulation aid, it is a visible defect: the wall bridges each rim vertex to its nearest
surviving floor vertex, so four or five rim vertices sharing one floor vertex fan into slivers
and the cast wall comes out **striated**. Removing the decimation and doing only what was
always needed — local crossing repair plus **coincident-projection dedup** — measures:

| | before | after |
|---|---|---|
| rim → floor (real scan, margin 7) | 2133 → 469 (78.0%) | 2133 → 2014 (**5.6%**) |
| forced ear clips | — | **0** |
| triangulated vs outline area | — | exact to 3e-16 |

**The dedup was the missing piece, not a cleverer triangulator.** Phase 4 had dropped
`_polygon_degenerate` out of the prune loop, so two rim points projecting onto one stalled the
ear clip, and the spacing ladder papered over it. Two other things had to go back:

* **The needle drop is load-bearing.** Without it a regular-grid rim gives **62 forced clips
  and 23% of the floor's area missing**; with it, 1 forced clip and area exact. Fixed tolerance,
  no escalation — escalating an epsilon amplified a 1e-15 rounding difference into a 0.6%
  volume change under a pure rigid transform.
* **THE INVARIANT IS AREA, NOT COUNT.** `ear_clip_polygon_robust` always returns n−2, so
  counting against it is vacuous — it passed while a quarter of the floor was missing.
  Comparing triangulated area with the outline's signed area is what catches that.

**`margin_mm` is 7.0, chosen from measurement.** kept_fraction 58.4 / 82.0 / 96.7 / 99.6 % at
3/5/7/9 mm; material still inside the arch opening beyond the margin 29.8 / 15.3 / 3.35 /
0.41 %. The 7–11mm band sits 6–9mm apical of the cervical margin — lingual sulcus and
vestibule floor — while crowns live within ~5mm of the ridge. 9.0 removed 0.44%, which is not
a trim; 5.0 removes 18% and cuts into the molars' walls.

**There was never a floating island in the exported base.** Measured: `components = 1`, sizes
`[190082]`. The manifest's `components` field reported islands **found**, not islands
**surviving**, and that is what read as "the base has 2 components". It now reports the output
(always 1), plus `islands_found` and `islands_removed` with face counts, and `build_cast_base`
asserts one component outright.

**Why every socket fell back, and it is a verdict on the CUT.** `fallback_reason` now names
the condition with its number. Measured on the same real tooth two ways:

| selection | rim | min radius about its own centre | result |
|---|---|---|---|
| raw FDI labels | 232 pts, 7.3 × 2.3 mm | 0.134 mm | flat_fallback |
| geodesic wand flood, as the app does | 213 pts, 10.6 × 5.6 mm | 1.342 mm | **cup** |

A label-picked crown has a ragged self-touching margin; a real flood does not. On a proper
flood the real scan now produces both outcomes and says which fired — one tooth cups, another
reports "the inset rings self-intersect at 0.798mm".

**Staging.** Stage k is `clinical × k/N` rebuilt through `kinematic_matrix` — **never** an
interpolation of the committed 4×4. Measured: every stage is rigid to 4.4e-16 (RᵀR−I), while a
4×4 lerp of the same movement reaches **1.68e-2** and is not a rotation at all. Both numbers
are pinned, in `verify-kinematics.mjs` and in `test_staging_export.py`, because the clinician
approves the scrub and the lab prints the files. Scrub cost, 14 crowns: **12.5 µs/frame**,
0.075% of a 60fps budget — React is kept out of the per-frame path entirely (refs + rAF).

**The npm benchmark, and what it cost.** Baseline 207.1 KB gzip. The four libraries together
came to **+51.5 KB (+24.8%)**, over the agreed +20% gate. The breakdown decided it:

| library | gzip delta | kept |
|---|---|---|
| framer-motion | **+39.4 KB** | **no** |
| @radix-ui accordion + slider | +11.0 KB | yes |
| lucide-react | +1.7 KB | yes |

framer-motion's only job here was scrub transitions, which is precisely the thing that must
not go through React. Final bundle **222.2 KB (+7.3%)**.

**The manufacturing export fuses base + crowns per stage**, sockets filled flush
(`build_socket_cup(depth_mm=0)`) so a moved tooth does not leave a gaping cup. Two things had
to be learned the hard way:

* **`_face_components` lies about CSG output.** It walks edge adjacency on the index buffer,
  and manifold3d's duplicate vertices split a geometrically joined solid into two
  index-disconnected groups — it called a tooth "floating" while a boolean intersection showed
  141 mm³ of overlap. Use `Manifold.decompose()`.
* **Inverted crumbs.** A tangential boolean leaves the odd tiny inside-out shell: measured,
  `decompose()` returned `[27572.4, -1.9]` mm³. Dropping non-positive-volume bodies is the
  cleanup; more than one surviving body is a real floating tooth and is refused.

**WHAT IS ASSERTED AND WHAT IS ONLY REPORTED, and the distinction is deliberate.** A fused
stage is asserted closed, one body, positive volume, zero open edges. Whether it survives a
reader that WELDS coincident vertices is **reported, not asserted** — unlike `/export`, which
holds its base to that bar. Putting a crown back into the hole it was cut from is a tangential
boolean: the crown's outer surface and the cast's are the same scan faces meeting edge to edge
at the rim, and manifold3d answers a tangency with topologically distinct vertices at one
position, which binary STL cannot express. Five separations were measured and none removed it
across prescriptions — a root-length plug, lifting it into the crown, insetting it to 75%,
seating the tooth 0.1–2.5mm (52 → 22 but never 0), and eroding the socket by 2–4 face rings
(74–104, and at 4 rings the tooth stops touching the base). The count moves with the
prescription, which is the signature of a tangency rather than a bug.

> **SUPERSEDED 2026-09-20 — the root plug is no longer manufacturing anatomy.**
> The paragraph below describes `_rim_plug` as the intended manufacturing
> connector. It is not, and has been removed from the stage export. A plug
> whose depth is `root_length_mm` travels with the tooth and EMERGES when the
> tooth extrudes: measured, a 1.2mm extrusion added 28.84mm3 of visible
> synthetic root. It is replaced by a bounded local target-position interface
> (`manufacturing.py`) — see §21 and MANUFACTURING_RECONSTRUCTION_CHANGELOG.md.
> The virtual root remains exactly as described for the VIEWPORT and for C_res.

**Virtual roots are geometry now, not just a number.** `/cut` returns a cone from the cervical
rim to an apex `root_length_mm` apical, placed from the rim centroid exactly as
`center_of_resistance` places the pivot. Drawn wireframe at 0.28 opacity in a violet used
nowhere else, so it can never read as scan data. The same cone, as a solid, is what the
manufacturing export uses as a plug — a tooth being extruded carries its plug with it, so any
plug shorter than the movement would clear the base entirely at the last stages.

**Not profiled: OutlinePass.** It needs a real WebGL context and there is no browser here, and
a number that cannot be measured will not be reported. The hover highlight ships as an
emissive boost instead — a uniform change on one material, against four full-screen passes per
frame whether anything is hovered or not. Profile it in a browser before switching.

## 11. RESOLVED: PHASE 6 — EXPORT GATING, CRUMB PURGE, ANTAGONIST COLLISION

**The stated gate would not have worked, and the measurement says why.** Task 1 asked to require
crowns be "watertight, single-component solids (Euler χ=2)". Measured on four auto-cut crowns:

| FDI | watertight | χ | bodies | volume | passes the stated gate? | actually fuses? |
|---|---|---|---|---|---|---|
| 46 | yes | 2 | 1 | 136.3 mm³ | yes | **yes** |
| 32 | yes | 2 | 1 | 63.3 mm³ | **yes** | **NO — fractures into 3** |
| 36 | yes | −2 | 1 | 8.8 mm³ | no | no |
| 43 | yes | 2 | 1 | **0.0 mm³** | **yes** | no (a 0.2 × 0.2 × 0.7 mm speck) |

**Every crown `/cut` produces is already watertight and single-bodied** — the endpoint asserts
`is_edge_manifold_closed` and refuses 422 otherwise — so that half of the gate is a **no-op**. χ
catches one. A volume floor catches the speck. **Neither catches FDI 32**, which passes every
topological test there is and still comes apart.

**So the gate tests the property that matters, not a proxy for it:** build the tooth's
manufacturing solid and require ONE positive-volume body. Solidity was considered and rejected —
crowns that fuse measure 0.104, 0.120, 0.181 and the one that fractures measures 0.0785, so
separating them needs a cut at ~0.09 from four samples. `crown_is_printable` stays as a cheap
screen because it gives a *specific* diagnosis; the boolean is what decides.

The refusal names every failing tooth and **refuses the whole export**, not a partial model: a
stage model missing a tooth is a wrong model, and a lab would thermoform a tray with a gap in it.

**`_solid_bodies` purges inverted crumbs at EVERY boolean, not just the last.** The counts
compound: crown+plug alone produced 2 raw bodies (1 crumb) on one tooth and 10 raw (7 crumbs) on
another, and those went into the stage union to be counted again — 16 raw bodies where 5 were
real. A crumb is a fragment with **negative volume** (measured `[27572.4, -1.9]` mm³), which is a
void, not geometry.

**The antagonist check only works because scanner coordinates are sacred.** Rule 3.1 means
neither arch is ever re-centred, so both sit in the same raw space and an upper incisor retracted
lingually really does land where the lower arch is. Had either been normalised to its own origin
— the obvious thing to do — this check would be measuring nothing.

**The two arches are separate SESSIONS.** `STORE.create(arch)` makes one per arch and nothing
links them, so the server cannot find the antagonist by itself. `opposing_session_id` is passed
by the client, which is the only party that knows both. Absent → the check is skipped and the
manifest says `checked: false`, which is **not** the same as "no interference".

**Signed, not unsigned.** Nearest-vertex distance signed against that vertex's normal. An
unsigned check cannot tell 0.9 mm of penetration from 0.9 mm of clearance and would warn on every
tooth in normal occlusion; the test flips the normals and asserts the verdict flips with them. It
is approximate by construction — nearest *vertex*, not nearest surface point, and it cannot see a
true intersection — because a real SDF needs the antagonist closed, and an intraoral scan is an
open shell. It is a warning, never a gate: the movement still commits, the export still ships.

**Two performance facts worth keeping.** The KD-tree must be cached — rebuilding it per call is
43 ms of the 89 ms a 2000-vertex crown costs, and a 31-stage 14-crown run asks 434 questions of
an arch that never changes. And `workers=-1` on the query is a 3–4× win on 8 cores (2000 verts
45 → 15 ms; 8000, 161 → 38 ms), taking that run from ~15 s to ~5 s.

Measured end to end, 31 stages: **CSG 1.00 s (0.032 s/stage)**, collision 0.44 s, 4.21 s total
including STL writing and zipping; 16 crumbs purged, every stage closed and one body.

## 12. RESOLVED: STARTUP RELIABILITY & SEGMENTATION PROGRESS

**The brief's two bugs were already fixed, and saying so was the useful answer.** CORS already
allowed both dev origins (OPTIONS preflight 200 from `localhost:5173` *and* `127.0.0.1:5173`, and
400 from an unlisted origin, so it is not a blanket `*`); `/api/ai/status` already existed;
`/segment` was already `async def` + `asyncio.to_thread` with a 409 duplicate guard; the client
already polled. `api_core:app` was already what served :8000 — `/openapi.json` reports *Virtual
Diagnostic Setup API v3.0, 14 routes*, not `server.py`. Two things were genuinely wrong.

**ROOT CAUSE 1 — the startup handler blocked the socket from ever being created.** Read straight
out of the installed uvicorn 0.52.4: `Server.startup()` does `await self.lifespan.startup()` and
only *then* `loop.create_server(...)`. So a slow `@app.on_event("startup")` does not make the
server slow, it makes the port **refuse connections** — which in a browser is exactly
`TypeError: Failed to fetch`, indistinguishable from a backend that was never launched. `_warm_tgn`
was calling `tgn_bridge.load()` inline: torch import, CPU shims, a 64MB checkpoint. Measured
`warm_seconds` **9.7s**, and `--reload` repeats it on every file save.

A/B on a minimal app with the same 9.7s load, so the mechanism is measured and not argued:

| startup shape | first 200 | connection failures before it |
|---|---|---|
| inline (old) | 11.93s | **5** |
| daemon thread (new) | **1.59s** | **0** |

End to end on the real app: port answered at **5.98s** with `loaded:false, warming:true`, model
ready at 15.7s. `tgn_bridge.load` is idempotent under `_LOCK`, and `/segment` calls it again
anyway, so a request landing mid-warm-up waits on the same lock and gets the same pipeline — and
one landing after a *failed* load gets the real error instead of a connection refusal.

**`warming` is a distinct state from `loaded:false`, deliberately.** Conflating them made a healthy
server report "AI unavailable" about a model that was still importing torch.

**ROOT CAUSE 2 — `python -m uvicorn api_core:app` does not work on this machine.** The brief's
literal command resolves to `Python312\python.exe`, which has **none** of the backend deps
(`No module named uvicorn`; no fastapi, no python-multipart, no torch, no open3d). Everything lives
in `.venv`. So `start_backend.bat` prefers `.venv\Scripts\python.exe` and preflights the imports,
the checkpoint and the port before launching. **Killing a `--reload` server needs both PIDs** — the
parent creates the listening socket and hands it to a `multiprocessing.spawn` child, so killing the
parent alone leaves the child holding :8000 and the next launch dies with `[WinError 10048]`.

**The "stuck feel" was never a bug — segmentation on the real scan takes 236 seconds.** Measured
against the live server on `case_lower.stl` (94,848 verts / 187,625 faces): upload 3.7s, `/segment`
**235.8s**, 11 teeth found. During it, `/api/ai/status` was sampled **232 times and the slowest
reply was 139ms** — the event loop was never blocked, which is why a progress ticker works at all.
A duplicate run fired 3s in returned **409**. The only honest fix for four minutes of waiting is to
show the clock, so the client renders live elapsed seconds.

**Elapsed time is counted from the LOCAL clock, not from `segmentation_started_at`.** That field is
server epoch time; a browser with a skewed clock renders a negative or absurd number, and local
elapsed is also the number the user is actually waiting on.

**"Still not working" was a backend that had simply been stopped, and the UI had no way to say
so.** The next session found :8000 dead with Vite still up on 5173, and every action failing. The
stack itself was intact — once restarted, upload returned **200 in 4.2s** (94,848 verts) with
`access-control-allow-origin: http://localhost:5173`, `/mesh` 8.6 MB, all with the browser's own
Origin. Nothing was broken; nothing was running. **A backend that was never started must not be
indistinguishable from a broken app**, so the sidebar now carries a permanent health badge polling
`/api/ai/status` every 3s: connected / AI loading / AI unavailable / **"Backend not running — run
start_backend.bat"**. Both branches verified against real ports — `fetch` to a refused port
**rejects with a TypeError** rather than resolving with `ok:false`, which is what the offline branch
keys on. A dead *model* is deliberately a separate amber state from a dead *backend*, because
upload, wand and manual cutting all still work without the model.

**Two Windows diagnostics that mislead, both hit here.** `Get-NetTCPConnection -State Listen` can
report nothing while a bind still fails with `[WinError 10048]` — the server was mid-startup between
the two calls, so trust the bind, not the probe. And a venv's `--reload` child reports the **base**
interpreter path (`Python312\python.exe`) in its command line while the parent correctly shows
`.venv\Scripts\python.exe`; that is normal multiprocessing spawn behaviour, not the wrong
interpreter.

**The poll is gated on `segmenting`, NOT on `busy`.** `loadArch`, `executeCut`, `exportSetup` and
`exportStages` all set `busy`, and the poll's error branch rewrites the status line every second —
so on a machine where the model failed to load, a 40s export read "AI unavailable: ..." instead of
"Trimming the arch...". `segmentStartedAt` is a ref nulled *synchronously* when the run ends, so a
poll still in flight cannot overwrite the completion message.

## 13. RESOLVED: PHASE B — HARD FAILURES & REPOSITORY HYGIENE

**The repository had no version control.** That was the real blocker, not any individual defect:
every cleanup was irreversible and CI was impossible. `git init` + a baseline commit came first, and
every change since is a separate revertible commit. **`.gitignore` excludes all patient-derived
data** — scans, meshes, `*_output.json` label files — because nothing derived from a real patient
belongs in version history.

**One syntax error existed in the whole repo.** `tgn_find_width_config.py:161` ended
`main()python tgn_find_width_config.py ToothGroupNetwork` — a shell command pasted onto the call.
The usage line already existed in the module docstring, so it was a stray duplicate, not displaced
documentation. Measured: **67 application files, 1 error; 97 vendored files, 0.** The two vendored
`SyntaxWarning`s (invalid escape `'\d'`) stay — vendored code is not edited to silence warnings.

**Half the lint problems were not real, and that is the interesting part.** eslint was linting
`ssr_out/` — a build artifact — because `globalIgnores` covered `dist` but not it. Seven of fourteen
reported problems were unactionable errors about generated bundle code, drowning the seven that
mattered. **Linting generated output does not raise standards, it lowers signal.**

**Adding a dependency the linter asked for would have white-screened the app.** `exhaustive-deps`
wanted `refreshStaging` in the array at `App.jsx:997`; `refreshStaging` was declared at `:1015`.
Dependency arrays are evaluated **during render**, so a `const` declared below is still in its
temporal dead zone — the identical crash `opposingSessionId` caused. The declaration was moved above
its user first. **This is now the second occurrence: treat any lint fix that adds a same-component
value to a deps array as suspect until you have checked the declaration order.**

**`useStagePlayback` wrote a ref during render** (`cur.current = stage`). React may render and then
discard that render, publishing a value for a render that never committed. Moved into an effect; the
tick only advances every 280 ms (~17 frames), so the post-commit write always lands first. The hook
also moved to its own module — a file exporting both a component and a hook breaks Fast Refresh for
both.

**Two empty files were shadowing real modules.** Root `config.py` and `models.py` were 0 bytes, and
`import config` from the repo root resolved to them, yielding zero symbols instead of
`tooth_segmentation/config.py`. Verified before moving that neither the app nor the vendored tree
does a bare `import config` / `import models` — every such import in ToothGroupNetwork is relative.
Latent, never fired, now impossible.

**`check_structure.py` was checking two hard-coded files** while being invoked as a repo-wide gate.
It now walks the tree: **61/61 sound, no false positives** — a 30× increase in what the gate covers.

**pytest could not collect at all before configuration** — 30 errors, from the generated
`Aligner_App_AI_Export/` holding a second copy of every test file ("import file mismatch"), and from
`_archive/test_stl_parser.py` opening `server.py` by relative path. Both excluded.
**`run_all_tests.py` stays canonical**; pytest runs alongside it.

**Measured, before → after:**

| | before | after |
|---|---|---|
| version control | none | 6 commits |
| syntax errors (app) | 1 | **0** |
| `npm run lint` | 14 problems | **0** |
| `check_structure.py` coverage | 2 files | **61 files** |
| `pytest` | not installed, 30 collection errors | **112 passed** |
| `run_all_tests.py` | 24/24 | **24/24** (unchanged) |
| `verify-kinematics.mjs` | 1.78e-15 / 4.44e-16 | **identical** |
| smoke markup | 7621 B | **identical** |

The last two lines are the point: §25's protected paths were touched (`StagingTimeline`,
`App.jsx` deps) and produced **byte-identical** numbers. Any drift there is a regression.

## 14. RESOLVED: PHASES C-F — RELOAD, VALIDATION, UI, CI

**Case reload: the blocker was EXPOSURE, not storage.** The server already held the occlusal frame,
the segmentation labels, every crown's geometry, its FDI and its root length — and no endpoint
returned any of it. `GET /teeth` existed and the client never called it, because poses alone cannot
rebuild a scene with no crowns in it. Four endpoints now expose what was always there
(`GET /session/{sid}`, `/frame`, `/labels`, `/teeth?geometry=true`), `/teeth`'s default payload stays
byte-compatible, and the session id — the one thing the server genuinely cannot recover — is kept in
`localStorage`. Ids only; nothing patient-derived leaves memory.
**A restored pose is REBUILT from the six clinical values via `deltaFromClinical`, not trusted as 16
stored floats**, so it lands exactly where a staging frame would put it.

**`_jsonable()` exists because the one-level numpy unwrap was not enough.** `socket_info` carries a
`(3,)` normal two levels down, which FastAPI surfaces as an opaque 500. Caught by a `json.dumps`
assertion in a test, not in production.

**Three defects that produced wrong answers silently:**
* **Negative vertex ids WRAPPED.** `sel[ids] = True` with a negative id selects from the END of the
  array — a crown built from the far side of the arch, reported as success.
* **`root_length_mm` was unbounded.** C_res is extrapolated along the long axis by exactly this
  distance, so 0 or 100 is a wrong pivot, not a wrong number.
* **A NaN C_res passed the alveolus gate.** `if depth <= 0 or lateral > max_lateral` looks
  exhaustive and is not: **every comparison against NaN is False.** Finiteness must be its own
  check, first. The test asserts the language behaviour so the reason cannot be forgotten.

**`cut_guard.check_crown` was hard-bypassed and is now un-bypassed but still does not gate.** It
computed both metrics then returned `ok=True` unconditionally, making the caller's 422 unreachable —
the pre-cut guard did nothing for several phases while appearing to. `MIN_COMPACTNESS` (0.30) and
`MIN_RIM_CONCAVITY` (0.05) have never been measured against real cuts, so they are reported as
`crown_advisory`. **Promote a threshold to a refusal only with numbers behind it**, otherwise a
rejected cut cannot be blamed on either the threshold or the cut.

**Every threshold declares its provenance** — `software heuristic` or `literature/reference`. They
are different kinds of claim and must never render alike. **A refusal now carries the value it
measured, the threshold, and where that threshold came from**; success responses were always richly
structured while failures collapsed to a bare string, which is backwards.

**`NOT_CHECKED` is not `CLEAR`.** Five explicit collision states
(`NOT_CHECKED / CLEAR / WARNING / INTERFERENCE / COMPUTATION_ERROR`). `checked: false` was being read
as "no interference", a clinical claim the software never made. Every measured state also states in
plain language that nearest-vertex signed distance is **approximate** and cannot see an intersection
between sampled vertices. The validation panel renders three visual states, and **"not determined" is
grey with a dash, never a green tick.**

**Two things E2E found that nothing else would have:**
1. **A stale `--reload` spawn child held port 8000 while its parent was gone.** `tasklist` showed no
   such PID, `netstat` showed it LISTENING, a bind failed with `WinError 10048`. Its command line
   contains **`spawn_main`, not `uvicorn`** — so a filter on "uvicorn" misses it, which is how a
   server running OLD CODE survived two restarts and served 14 routes while the module defined 17.
   **Kill on `spawn_main` too.**
2. **CORS did not allow the preview origin (4173).** A CORS block and a refused connection both
   reject `fetch()` with the same `TypeError`, so the connection badge reported "Backend not running"
   for a server running perfectly. **The badge cannot distinguish the two** — remember that before
   trusting it.

E2E runs against the **production build**, not the dev server: dev serves ~1,900 unbundled modules
and measured **15.8 s to first paint**, which turns every timeout into a coin flip.

| | before | after |
|---|---|---|
| `run_all_tests.py` | 24/24 | **26/26** (+ hydration, validation) |
| `pytest` | 112 | **131** |
| browser E2E | none | **7/7** |
| CI | none | 2 jobs, AI path excluded by design |
| API routes | 14 | **17** |

## 15. RESOLVED: DOMAIN MODEL, CASE FILES, BVH, SELECTABLE LONG AXIS

**three-mesh-bvh was declared and imported nowhere; it is now real, and the number
says why it had to be.** On 204,800 faces at real arch density, 300 raycasts,
identical hit results both ways:

| | ms per raycast |
|---|---|
| three.js default | **12.974** |
| three-mesh-bvh | **0.023** (555x) |
| bounds tree build | 133.4 ms, once per mesh load |

12.97 ms is **78% of a 16.7 ms frame budget** and the hover handler raycasts on
every pointer move, so the default path could not hold 60 FPS on a real arch
however fast the renderer was. **Rebuild the tree wherever `setIndex` is called** —
extraction rewrites the arch index in place, and a stale tree keeps reporting hits
on triangles that are no longer drawn. The tree is also a separate allocation that
`geometry.dispose()` does not free. The brush keeps its uniform grid: that answers
"which vertices lie within r of this point", which a raycast BVH does not accelerate.

**The "< 1e-15" acceptance bar was unachievable and is now 1e-12.** Observed
agreement is 1.78e-15 — about **8 ULP of 1.0 in float64**, the noise floor for a
chain of 4x4 products, not slack better code could remove. A bar of 1e-15 fails a
CORRECT implementation. 1e-12 mm is three orders below any clinical tolerance and
still catches what this check is for: a convention drift (row vs column major,
Euler order, pivot handling) moves the error by **orders of magnitude, never by a
few ULP**. Every run now prints the observed value and its ULP count.

**The domain model is treatment state, and it is kept apart from geometry on
purpose.** Three layers: the scan (memory-only, never on disk), the plan
(`domain.py` — prescriptions, FDI, C_res, review flags), and derived visuals
(crowns, cups, cones — all rederivable, none persisted). **A full case is 602
bytes with zero geometry keys**, asserted against the real payload rather than
promised in a comment. A prescription is six numbers and a tooth id; a mesh of
somebody's dentition is biometric data whether or not a name is attached.

`Case.opposing()` resolves the antagonist **server-side**. It only means anything
because scanner coordinates are sacred: both arches sit in the same raw R^3 space.
Had either been re-centred on its own origin — the obvious tidy-up — the collision
result would be noise that looks like a measurement.

**Case files are Fernet-encrypted, and the threat model is stated rather than
implied.** It protects a case file copied somewhere it should not be — email,
synced folder, USB, backup. It does **not** protect against someone with this
workstation: the key lives beside the data. It is **not** a regulatory control.
A tampered or truncated file is **refused by the HMAC**, not half-loaded — a
treatment plan missing teeth is more dangerous than one that will not open.

**Session eviction is no longer silent.** Loading a fifth arch destroyed a case
and the only evidence was a 404 on the next request, which reads as "expired" and
sends people to the TTL. `STORE.evictions()` records the last 16.

**Both long-axis derivations now exist, and the naive reading of the textbook one
is 90 degrees wrong.** `LONG_AXIS_MODE` selects `"rim_plane"` (DEFAULT, unchanged,
what every measurement here was taken against) or `"cross_product"` (u_OA = u_MD x
u_BL). Measured on the flat-molar fixture:

* using the arch-level `u_tra` as the buccal reference: **90.000 deg out**
* using the per-tooth `buccal_direction()`: **0.644 deg** from rim_plane

**Buccal is radial, and radial is per-tooth.** It is roughly ±u_tra for a molar and
roughly u_sag for an incisor; no single arch-level vector is buccal for every
tooth. That is why the textbook triad needs a per-tooth reference the arch frame
alone does not provide — and why the rim-normal form, which uses the tooth's own
cervical anatomy, was adopted after the tetherball glitch (§5). **Change the
default only with numbers.**

## 16. RESOLVED: SPACE ANALYSIS, SEGMENTATION REVIEW, ATTACHMENTS, AUDIT, CBCT HOOKS

**Phase 3 was wiring, not new geometry.** `local_arch_tangent`,
`measure_mesiodistal_width`, `validate_against_anatomy` and Wheeler's tables had
existed since the caliper work and were called by nothing but `test_caliper.py`.
`GET /space-analysis` now reports per-crown mesiodistal width on the occlusal 40%
along the tooth's OWN arch tangent. Measured: a 7.0mm block reads 7.0mm; a doubled
crown reads 14.0mm against Wheeler's 6.5 and is flagged **REVIEW with the likely
cause** ("often two teeth merged"), not just the number.

**FOUR states, because two would lie.** `UNKNOWN` = no FDI, so no comparison is
possible. `UNMEASURABLE` = the crown centroid coincides with the arch centre,
where the radial direction and therefore the mesiodistal axis are undefined —
**returning 0.0 there would read as a measured zero-width crown**. `reliable`
stays False unless a full dentition was scored: a pass rate over four teeth is not
a grade for an arch.

**IPR measures a gap in a mesh; it does not prescribe enamel reduction.** Every
contact row states that a vertex-to-vertex minimum **OVERESTIMATES** the true gap —
on a triangulated surface the closest points generally lie inside faces, not at
vertices. "IPR Contact 11-21: 0.22 mm" reads far more precise than it is.

**Segmentation confidence is weighted agreement, NOT a model probability.**
ToothGroupNetwork emits no calibrated uncertainty and inventing one would be worse
than having none. `GET /segmentation-review` scores independently checkable facts —
connectedness, size, extent, FDI-belongs-to-arch, width vs Wheeler — and **returns
every contributing factor with the score**, so a clinician can disagree with a
factor rather than only with a number. Measured: FDI 16 on a lower arch →
REVIEW at 0.706 naming `fdi_belongs_to_arch`; a split label → 65% largest island.
`ToothCandidate.confidence` had been assigned nowhere since the package was
written, so its `needs_review` was unconditionally True — **a review flag that is
always on is the same as no flag.**

**An attachment's ORIENTATION is its function, so it is built in the tooth's
frame.** A vertical rectangle resists rotation because it stands along u_OA.
Measured: upright tooth extent `[2, 1, 3]`, same attachment on a 45° tooth
`[2, 2.83, 2.83]` — it follows the tooth, not world Z. A world-axis block would be
"vertical" only for teeth that happen to stand upright in scanner space, which is
none of them. Fused through the same `manifold3d` path the export uses, with the
same crumb filter; **two bodies means it is not bonded** and is refused.

**The audit trail cannot be made to record a patient.** A closed action
vocabulary (a typo cannot create a silent category) and a runtime denylist checked
on every append: geometry keys, names, DOB, MRN and filenames are all refused.
Bounded to the newest 500 entries — an unbounded log on a clinical workstation is
a disk incident, and one nobody prunes is one nobody reads. Zero telemetry.

**CBCT is an INTERFACE, and every entry point raises.** A stub returning a cone
from `segment_roots` would be the most dangerous thing in this repo: the app draws
Wheeler cones as translucent wireframe labelled *"virtual root (estimated)"*
precisely so they cannot be mistaken for imaging, and output from a function of
that name would reasonably be believed to come from a scanner. Registration must
transform **the CBCT into scanner space, never the scan** — rule 3.1 — and a 2mm
residual moves C_res by 2mm.

## 17. RESOLVED: THE FIVE OPEN ITEMS

**There is no ground truth in this repository, and the benchmark refuses to
pretend otherwise.** `case_lower.stl_output.json` is ToothGroupNetwork's own
PREDICTION. Scoring a model against its own output returns mIoU 1.0, FDI accuracy
100% and Chamfer 0.0 — numbers that look like validation and mean nothing.
`benchmark_segmentation.py` implements per-class IoU, FDI accuracy, Chamfer and
Hausdorff correctly and completely, then **refuses to run** without an
independently attributed annotation. It rejects an annotator field naming the
model, a missing attribution, and a label count that does not match the mesh.

**The `/segment` confidence breakdown is a plausibility proxy and says so in two
fields.** `is_measured_accuracy: false` and a `meaning` string stating the key
`miou_estimate` keeps the API's name but **must not be quoted as an mIoU**. This
project already carried four mutually inconsistent accuracy claims inherited from
prose; a fifth from a self-comparison would have been worse than none.

**The hybrid fallback repairs SHAPE and keeps the model's NAME.** Tier 2 is the
classical geodesic flood, which follows the curvature barrier and therefore
returns ONE connected region by construction — the exact property tier 1 loses.
It cannot name a tooth, so the FDI is kept from the model. Measured: a split
label (65% largest island) is repaired by re-growing 5,671 vertices, and marked
**REVIEW_REQUIRED, because a repair is not a confirmation**. The trigger is NOT an
mIoU: no IoU is computable at runtime.

**Scans are now cached on disk, which reverses a standing rule, and the cost is
stated rather than absorbed.** A dental arch mesh is BIOMETRIC data — it
identifies a person the way a fingerprint does. So scans are encrypted with the
same Fernet key as case files, keyed by **SHA-256 of the raw upload**, and no
filename is stored. Measured: zstd to **20% of raw size**; restart recovery
restored 28,900 vertices with no re-upload. **A content hash cannot be
re-pointed** — a random id could be aimed at a different scan and the plan would
silently apply to wrong anatomy.

**Three characterization scripts became tests.** They printed tables and exited
0, so they reported PASS regardless of what they measured. Now: 20 interproximal
configurations asserted non-negative with the separation ordering holding
(1.47 >= 1.15); the colour buffer asserted `(22500, 3)` float32 in `[0,1]` —
a 0-255 palette slipping in clips silently to white in WebGL; 91% of identified
incisal points asserted inside the occlusal 15% band.

**The antagonist sweep found first contact at stage 22/31** on a 3mm extrusion
into a 2mm gap — exactly the predicted two-thirds — and penetration is monotonic.
The fixture construction had a real bug first: **reflecting through a plane `g`
above the tip puts the surface `2g` away**, so the gap was silently doubled and
the interference test could never fire. **There is still no real maxillary scan**;
the antagonist is a mirrored mandible, and a test asserts that so the gap cannot
be forgotten.

**GREEN never claims biological safety.** Rate limits are HEURISTIC (the tray
will not track it); total extrusion/intrusion limits are LITERATURE-labelled and
differ by a millimetre, so **a sign error would swap them** — `+2.5mm` extrusion
is RED while `-2.5mm` intrusion is allowed, and a test pins exactly that. The
clinical report leads with what is outstanding and carries the disclaimer
verbatim; IPR rows carry `reduction_mm: null`, because the software measures a
gap and does not prescribe enamel reduction.

**`npm run smoke` caught a TDZ crash that `npm run build` reported as success.**
Adding `placeAttachment` to a deps array above its own declaration threw
`Cannot access 'placeAttachment' before initialization` while the bundle built
cleanly at 946 KB. **That is the FIFTH occurrence of this trap in App.jsx.** The
rule, now stated for the last time: **a deps array is evaluated during render, so
anything it names must be declared above it** — route through a ref and publish
it in an effect placed below the declaration.

**An attachment's normal must be converted to world space.** `face.normal` is in
OBJECT space, and a cut crown carries its prescription as a committed 4x4 — using
it raw places attachments correctly on an unmoved tooth and progressively wrongly
on a moved one.

## 18. RESOLVED: FAIL-SAFES, FAULT TOLERANCE, AND A DEVSECOPS STAGE

**The happy paths in this brief were already built and green. What was almost
entirely absent was the CONTINGENCY half of each** — and two audits found
**four defects that reported themselves as fine**, which are worth more than
any fail-safe added beside them.

**1. `three.current` was REPLACED after `sun` and `catcher` were attached to it.**
Two lines assigned them onto the object; a few lines later
`three.current = { scene, camera, renderer, controls, raycaster }` discarded the
whole thing. `aimShadows` opens with `if (!sun || !catcher) return;`, so **the
entire shadow rig has never armed** — no throw, no warning, just no shadow. This
is the same class as the record-vs-`Object3D` confusion in §18's old entry, and
the same lesson: **the silent half of a bug is the half that survives a fix.**

**2. `shadowsDirty` was written by `aimShadows` and nothing else**, while the
comment beside the flag claimed a cut or a committed transform set it too. Every
extraction and every tooth movement left the shadow of the **pre-cut arch** on
the catcher. It is now set in `executeCut`, both kinematics commit handlers, and
both attachment handlers. `autoUpdate=false` was correct; the re-arm was broken.

**3. There was no React error boundary at all.** This app has white-screened
**five** times from temporal-dead-zone errors, every one of which built cleanly.
`ErrorBoundary.jsx` leads with the recovery — *the scan, the occlusal reference,
every cut and every committed movement are held in the session on the backend* —
because after five white screens that matters more than a stack trace.

**4. `over_threshold` was computed from `contact_mm` (0.30) while sitting beside
a field named `threshold_mm` (0.05) that was compared against nothing.** Any
reader assumes the flag means 0.05. The flag's **value is unchanged** — two
consumers gate on it — but the payload now carries `contact_threshold_mm` and
`noise_floor_mm` under names that say which is which, and `threshold_mm` finally
functions as the noise floor it claimed to be: **a 0.01mm closure is the mesh,
not the movement**, and scanners resolve to 20–50 microns.

Also: `core_geometry.py:438` was **the only silent `except: pass` on the live
geometry path**, sitting directly under hole capping. It now warns and names the
fan fallback.

### Three measurements that decided a design, and would have been wrong to guess

**The NaN gate exists because of how NaN compares.** `condition_mesh` drops a
face when `area <= 1e-12`. Every comparison against NaN is False, so a NaN
triangle has `area = nan`, is not `<= 1e-12`, and **survives the one filter
whose job is to remove it** — then dies in `cap_boundary_loop` as `LinAlgError:
SVD did not converge`, which reaches a clinician as an opaque 500 about a file
that opens everywhere else. `sanitize_scan` is **delete-only**, because rule 3.1
is the constraint on the repair: a repair that quietly re-centred a damaged arch
would fix the crash and break inter-arch registration, which is worse. A test
asserts surviving vertices are **bit-identical** to the upload, and a clean scan
returns *the same objects*.

**`trimesh.repair.sanitize()` does not exist** — not in trimesh 5.1.0 and not in
any release. The nearest real API is `Trimesh.remove_infinite_values()`. The gate
is NumPy so it is provably delete-only and works with the optional wheel absent;
a test cross-checks the two produce identical surviving geometry.

**The root-cone clamp uses the scan's GLOBAL apical extent, not a local one.**
The obvious choice is the depth in a cylinder around the tooth, and it is wrong.
Sampling 12 ridge points on `case_lower.stl`:

| bound | median depth | clamps a 13mm canine |
|---|---|---|
| local, within 6mm | 12.24 mm | **7 of 12** |
| global | 14.58 mm | **0 of 12** |

The local bound measures the vestibular depth, which is normal anatomy. The
global bound fires only on a cropped or shallow scan. **A warning that fires on
healthy anatomy is the same as no warning** — the same lesson as
`ToothCandidate.confidence` being unconditionally `needs_review`.

**Hoisting the KD-trees is what makes a per-stage interproximal sweep possible.**
94,848-vertex base, 2,000-vertex crown, 31 stages: **8.856 s → 0.514 s, 17.2×**
(285.7 → 16.6 ms/stage). Same lesson as the antagonist index in §11. It runs once
per commit on `/kinematics`, never during a scrub — a KD-tree query inside a
60 FPS loop is exactly what the refs-and-rAF path exists to avoid.

### Four things that were right to refuse, and one that was wrong to assume

**Welding the arch to fix a BVH build would be a cure worse than the disease.**
`mergeVertices` renumbers vertices, and an arch vertex id is the contract with
the backend session — `BrushIndex`, the wand selection and the cumulative
extraction mask are all keyed on it, and the server's `verts` array is never
rebuilt (§6). Repairing a raycast by silently re-pointing every selection at
different anatomy is far worse than a 13 ms cast. `allowWeld` defaults to
**false**; crowns, whose ids are local and replaced wholesale from each payload,
opt in. The arch's realistic failure — NaN coordinates — is caught at upload
instead, where it can be repaired without renumbering.

**`|det − 1| < 1e-6` does NOT catch a transposition, and the brief names it for
that job.** `det(Mᵀ) = det(M)` for every matrix there is, so transposing a rigid
4×4 leaves the determinant at exactly 1.0 and a bar of any tightness passes it.
What a transposition actually does is move the translation out of the last column
into the **bottom row**, which is the perspective-divide explosion rule 4 warns
about. Both checks now run: observed `|det−1|` **3.33e-16**, bottom row
**0.00e+0**, agreement unchanged at **1.78e-15**.

**An inverted-winding input does not make `manifold3d` raise.** It unions an
inverted solid and returns a different volume, silently — measured 0.125 mm³
where 1.875 was correct. The repair cascade cannot catch that and does not claim
to. What it does catch: an open shell repairs at rung 1 to the correct volume and
is recorded as `CSG_REPAIR_CASCADE_APPLIED`; triangle soup is **refused 422
naming every rung tried**. There is deliberately no displacement-carving rung —
a tray thermoformed from a solid the software had to carve into shape is worse
than a tray that was never made.

**The weld rung uses 1e-5 to FIND duplicates, not to replace them.** Snapping
survivors to the lattice is a different operation; a test asserts every surviving
coordinate is an original, from a deliberately off-lattice input.

### The telemetry denylist needed two sets, and the first version proved it

Reusing `audit.py`'s single denylist **dropped 6 of 8 attributes including every
count**, because `faces` is an array of triangles in an audit entry and a scalar
187,625 in a span — the single most useful thing to record about `/cut`.
Identifiers are dropped whatever their type (*a name is an ordinary scalar
string, and no shape check will ever flag it*); geometry keys matter only for
non-scalars, and **every non-scalar is reduced to its type and length whatever
the key is called**, which is what catches `attrs={"debug": verts.tolist()}`.
Measured: 94,848 vertices under three innocent key names produce a **315-byte
span**.

**`record_span()` exists because the first wiring lied.** `/cut` logged
`duration_ms: 0.0` beside `seconds: 0.352`, because a context manager opened
where the counts are known wraps nothing. **A field named duration that reads
zero for a multi-second operation is worse than no field.** Now 452.1 ms on a
real cut.

**`SimpleSpanProcessor`, not `Batch`.** A span buffered in memory is a span that
is NOT on disk when the process dies, and the ones worth having are written just
before something went wrong. Batching also made every telemetry test read an
empty file, which is how it was found. Both backends emit an identical schema so
a support engineer never has to work out which produced a line.

**FILE EXPORTER ONLY, and that is the design rather than a default.** There is no
OTLP exporter, no collector endpoint, and no environment variable that can turn
one on. An observability SDK is a data-egress path wearing a helpful hat, and the
ordinary way it gets configured is `OTEL_EXPORTER_OTLP_ENDPOINT`, set by somebody
not thinking about PHI. A semgrep rule fails the build on any network exporter,
and a test parses `telemetry.py`'s **AST — not its prose** — to prove none is
referenced. (The first version grepped the file and failed on the docstring that
*explains* why they must not appear. A test that cannot tell an explanation from
an implementation will either be deleted or will force the explanation out, and
the explanation is the more valuable of the two.)

### Two semgrep rules could not fire, and a clean scan hid it

**Every rule was verified against a deliberately bad probe file before being
trusted**, which caught both:

* `pattern-not-inside: def test_$F(...)` is not valid semgrep — there is no
  partial-identifier metavariable. It failed to parse, which **disabled the whole
  rule** while the scan still reported success.
* `pattern: dangerouslySetInnerHTML={...}` never matches; semgrep parses a bare
  JSX attribute as an expression. It has to be anchored to an element.

**A clean scan from a rule that cannot match is indistinguishable from a clean
scan from a rule that can.** Repo scan: **9 rules, 209 targets, 0 findings, 0
parse errors.** It is a CI job and deliberately **not** a pre-commit hook: a hook
that costs seconds per commit is bypassed with `--no-verify` within a week, and a
check that is routinely bypassed is worse than one that is absent, because it is
believed in.

### Other behaviour that changed

**`space_analysis` measured T0 crowns** (`t["cv"]`, the crown exactly as cut)
while the clinician was looking at the planned setup — so every contact it
reported described the malocclusion they started with. It now poses each crown,
and the payload **states which pose it measured**, because a contact table is
read as a statement about the plan.

**C_res projection is clamped to `[7, 15]` mm and reported every time.** This is
NOT `validation.ROOT_LENGTH_MIN/MAX` (4–30mm) and the two must not be conflated:
that band refuses a typo, this one clamps a plausible-but-out-of-envelope value.
A silent 2mm pivot shift is only visible months later, as a tooth that tipped
where it should have translated. A non-finite root length is **refused** rather
than clamped, because `min`/`max` pass NaN straight through.

**Low segmentation confidence opens the landmark picker instead of driving the
cut.** `AUTO_CUT_MIN_CONFIDENCE` (0.70) is separate from the PASS bar (0.99) on
purpose: PASS is strict because any disagreeing factor deserves an eye, and would
be useless as a blocking gate. The cut is **not refused** — the landmarks are
cleared, the two-click picker is armed with the tooth selected, and the clinician
places them. What is refused is the machine's guess doing it for them.

**The re-upload banner is persistent, and that is the entire point.** The backend
has returned a per-arch `missing[]` with reasons since the scan cache landed and
nothing consumed it; the session-restore path put its message in the status bar,
which the next `setStatus` overwrites — often within the second. An E2E spec
waits through several health-poll ticks and asserts the banner is still there.

**`.gitignore` now excludes `storage/` and the Fernet keys.** An arch mesh
identifies a person the way a fingerprint does, and a key committed beside its
own ciphertext is not encryption. Verified nothing of the sort ever reached
history.

**§2.5 needed no work.** The `<details>` fallback triggers above **+50 KB gzip**
and Radix measured **+11.0 KB**. Recorded, not built.

Measured, before → after:

| | before | after |
|---|---|---|
| `run_all_tests.py` | 34/34 | **36/36** |
| `pytest` | 225 | **237** |
| `check_structure.py` | 84 files | **88 files** |
| browser E2E | 11 passed / 1 skipped | **19 specs**, 12 pass / 7 need the backend |
| semgrep | none | **9 rules, 209 targets, 0 findings** |
| `npm run lint` | 0 | **0** |
| bundle gzip | 250.7 KB | **264.19 KB** (+13.5 KB, `BufferGeometryUtils` + fail-safe UI) |
| `verify-kinematics` | 1.78e-15 | **1.78e-15**, plus `|det−1|` 3.33e-16 |
| API paths | 31 | **33** (`/audit`, `/telemetry`) |

Covered by `test_failsafes.py` (18), `test_telemetry.py` (12) and
`frontend/e2e/failsafes.spec.js` (7).

## 19. KNOWN OPEN ISSUES
* **RESOLVED 2026-09-15 — the suite is 24/24.** `test_api_core.py` was rewritten against the
  current endpoints and `test_face_order.py` took the one-line `stl_io.parse_stl_bytes` fix. Both
  had been red on a **stale API surface**, never on geometry, which is why nothing downstream ever
  broke because of them. `test_api_core.py` now covers the nine endpoints `test_cut_endpoint.py`
  does not (session, mesh, occlusal-plane, wand, threshold, selection, teeth, ai/status, delete) in
  14 tests / 44 assertions, and pins two contracts that had no test at all: **the scan is never
  re-centred** (fixture offset to `[137.5, -62.25, 41.0]`, asserted to survive upload — rule 3.1 at
  the API boundary) and **every session endpoint 404s rather than 500s on an expired session**.
  `/segment` is deliberately excluded: a 64MB checkpoint and ~236s is a runtime property of the
  model, not of the module.
  **RESOLVED 2026-09-16: the suite is 36/36 and every entry carries enforcing assertions.**
* `/teeth` **is** called on load now — see §14. A refresh restores crowns, poses, the occlusal
  frame and the labels; only the session id lives in `localStorage`, and when it no longer
  resolves the persistent re-upload banner says so (§18).
* **OPEN — the CSG repair cascade has never fired on real geometry.** It is exercised on an open
  shell and on triangle soup, and both behave as designed, but no real scan has yet produced a
  boolean that `manifold3d` refuses. The rungs are the right ones in principle; which rung a real
  failure lands on is unmeasured.
* **OPEN — the C_res clamp has never fired outside a test.** The slider is bounded 7–16mm in the
  UI and `rootDefaultForFDI` returns 9–13, so reaching it needs an unsegmented case with the
  slider at 16. The arithmetic and the warning are pinned; a clinical trigger is not.
* **OPEN — `sanitize_scan` has never seen a genuinely damaged scan.** `case_lower.stl` is clean,
  so the gate is proved on a synthetic fixture and cross-checked against trimesh. That the repair
  is delete-only is asserted; that it is *sufficient* for whatever a real scanner emits is not.
* **OPEN — the OTel bridge runs only because semgrep pulled the SDK in.** `opentelemetry` is
  deliberately absent from `requirements.txt`, so on a clean install the builtin writer is what
  runs. Both paths are tested and emit an identical schema, but a production install exercises
  the builtin one.
* `u_bl` is pinned buccal from the arch frame, so `u_md`'s sign varies by quadrant (unavoidable — the arch is mirror-symmetric). The frame reports `u_md_points_distal`; the sliders do not yet use it.
* **A pinched cast rim loses a small spur.** `_open_pinch_vertices` deletes the smaller fan at a neck, and `/cut` drops the smaller lobe of a self-touching socket rim. Both are a handful of triangles and both are recorded, but neither is reconstructed.
* **`trim_to_arch` is not stable on a very coarse mesh.** Measured: at ~18 samples across the band the trim nearly severs the arch and `np.bincount(labels).argmax()` picks a different largest component under rotation — an 80% volume swing. Unreachable at scan density (187k faces across the same band) and worth remembering before anyone decimates a scan upstream.
* **SUPERSEDED 2026-09-22 by s.26.8 - the reconstruction now BUILDS on a real crown and
  reaches a written, re-read STL.** FDI 45 goes STL -> occlusal plane -> segmentation ->
  click-to-select (100% purity) -> cut -> C_res movement -> local cast reconstruction ->
  staged fused solid -> written STL -> reread, closed, one component, ZERO open edges.
  It is still NOT PRINT READY: four aggregate gates fail, and the one the forensics can
  localise is a single 0.09515mm self-touching edge whose four incident faces are all
  ORIGINAL_CAST. The wand is still not the route and no longer needs to be (s.26.4).
  The text below is kept because it is what the numbers used to say.
* **UPDATED 2026-09-21 — real crowns now cut; the MANUFACTURING RECONSTRUCTION on them does
  not yet build.** The old text here said no real crown could be produced headlessly, and that
  was a consequence of the label-mapping defect (§24.3), not of the scan. With the labels
  transferred by position, **seven real teeth cut successfully** — FDI 32, 34, 41, 42, 44, 45
  and 46 — three of them producing a proper socket cup rather than the flat fallback, and one
  of the seven built a complete local interface. What has still not been shown is a real crown
  reaching a validated final STL: staging refuses with `interface_unbuildable_wall_too_thin`,
  `outside_reconstruction_envelope` and `interface_construction_failed`, by name and with
  numbers (§24.7). **The wand itself is still not the route** — its auto tolerance returns
  25.00 mm on this scan and floods a quarter of the arch — so the selection comes from the
  segmentation label's own largest connected component, which is what the shipped fallback
  does anyway (§17).
* **The antagonist check has never run against a real opposing arch.** There is no upper-arch
  scan in this repo; every test uses a synthetic plate or the lower arch mirrored into occlusion.
  The arithmetic and the wiring are pinned; a clinical result is not.
* **OutlinePass is unprofiled** (see §10). The emissive-boost hover ships in its place.
* `PLUG_INSET_FRACTION`, `PLUG_LIFT_MM` and the flush socket fill are manufacturing-only
  geometry. They never touch the display path or `/export`.
* **RESOLVED 2026-09-15 — the record-vs-`Object3D` mistake in `aimShadows`.** `box.expandByObject(rec)`
  (`App.jsx:1104`) was handed a plain record (`App.jsx:621`), not an `Object3D`, and
  `Box3.expandByObject` calls `updateWorldMatrix` on its first line — so every "Define Occlusal
  Plane" threw. The plane was still established (`setArchFrame` and the POST both complete before
  `aimShadows` runs, so `/cut` never 409'd), but a step that had worked reported *"Occlusal Plane
  cancelled: object.updateWorldMatrix is not a function"* and the shadow rig never armed.
  **Both call sites had to be fixed, not just the throwing one:** `:1123` set `castShadow` on the
  same record, which is a *silent* no-op, so repairing `:1104` alone would have armed the sun while
  the arch still cast nothing. The lesson worth keeping: **the same type confusion shows up once as
  a crash and once as silence, and the silent one is the one that survives a fix.**
* **RESOLVED 2026-09-15 — `requirements.txt` did not describe a working install.** `manifold3d` was
  commented out and `torch`/`scikit-learn` absent. The first diagnosis was also incomplete:
  **`trimesh` and `open3d` are equally load-bearing** — `inference_pipelines/inference_pipeline_tgn.py:1`
  imports `gen_utils`, which imports `trimesh`, and both import `open3d`. All six are now listed.
  Verify a dependency by following the import chain, not by reading the app's own top-level imports.
* ~~3 of the suite entries assert nothing~~ **RESOLVED 2026-09-16** — `test_interproximal`, `test_auto_color` and `test_incisal_edge` now carry enforcing
  assertions. Every suite entry can now fail. `test_interproximal.py`, `test_auto_color.py` and
  `test_incisal_edge.py` are characterization scripts that print measurements and exit 0, so they
  report PASS unconditionally. The suite is 21 real tests, not 24.
* **Segmentation cannot be cancelled.** It is 236s on a real scan and the only exits are waiting
  or restarting the backend. `asyncio.to_thread` gives no cancellation point inside the model, so a
  real cancel needs the work in a subprocess, not a thread.
* **`_SEGMENTATION_STATE` is per PROCESS, not per session.** There is one model and one run at a
  time, so this is right — but the progress message does not name the arch, so a second tab
  segmenting the other arch sees the first tab's message and a 409 without being told whose.
* **Attachments and CBCT are the next phase.** Staging exists now; the base is a real solid and
  `manifold3d` union means something, so bonded attachments are a boolean on a surface that can
  take one.

---

## 20. RESOLVED: VIEWPORT FRAMING, THE SHADOW RIG, AND A MANUFACTURING OFFSET

Three things: which way up an arch is framed, a shadow rig that had never once
executed, and a print allowance that belongs after the boolean rather than
before it.

### 20.1 The framing sign was decided by where the scan sat in scanner space

`frameArch` derives the occlusal axis as the smallest-eigenvalue eigenvector of
the vertex covariance — an arch is a flat-ish horseshoe, so the direction it is
flattest in is the occlusal normal. That part was right. Resolving the
eigenvector's arbitrary SIGN was not:

```js
if (up.dot(camera.position.clone().sub(bs.center)) < 0) up.negate();
```

On a fresh page `camera.position` is still `(0,0,0)`, so this reduces to
`u · (−boundingCentre)`: it asks where the scan happens to sit relative to the
scanner's origin, which is not a fact about the patient. Measured on
`case_lower.stl` the bounding centre projects **5.88 mm** onto that axis, so the
old rule got the right answer — by 5.88 mm of luck. **Translating the scan 6 mm
flips the view upside-down**, and rule 3.1 says a translation must change
nothing. That is the defect, independently of whether it manifests on a given
file.

**The occlusal end is recoverable from the shape, and two independent signals
agree.** Projected onto the flattest PCA axis (spreads `[3.15, 16.03, 21.23]`
mm, so the axis itself is unambiguous), on 62,544 sampled vertices:

| | occlusal end | tissue end |
|---|---|---|
| skewness of the projection | **+0.697** | −0.697 |
| radial spread, outer decile by count (σ) | **5.86 mm** | 3.14 mm |

Skewness is primary: cusp tips are a sparse scatter reaching past the body of
the cast, so the third moment leans occlusally. The radial spread corroborates,
and **the direction of that second signal is the opposite of what it sounds
like.** The occlusal slab is the handful of highest points — incisal edges far
forward of the arch centroid, molar cusps much nearer it — so its radii VARY.
The tissue slab is a near-continuous band running right around the periphery at
an almost constant radius. The planning pass had this backwards on the
intuition that "cusp tips form a thin ring"; they do not form a ring at all,
the gingival margin does. Shipped as first written it would have logged a
disagreement warning on every single load.

The slabs are taken by equal COUNT, not by axial extent: the outer tenth of the
extent puts 119 vertices in the occlusal slab against 2695 in the tissue slab,
because one stray cusp tip stretches the extent that defines it.

`occlusalOrientation(positions)` is exported as a pure function over a flat
`[x,y,z,…]` array, so it is testable with no WebGL context, no
`BufferGeometry` and no DOM.

**Ground truth always wins.** `frameArch` now takes `{ jaw, frame }`. Given an
established occlusal basis it uses `u_occ` directly and skips the heuristic
entirely. `u_occ` points out of the mouth for BOTH jaws (`arch_frame.py:50`
resolves its sign against the arch centroid), so which side the camera sits on
is **not** a jaw question — the brief asks for an upper/lower branch on the
camera position and it genuinely does not need one. The jaw decides only the
roll: `camera.up` is `−u_sag` for a mandible and `+u_sag` for a maxilla, so the
incisors sit at the bottom of the screen for a lower arch and at the top for an
upper, which is how an opposing pair reads in the mouth.

**THE SIDE WAS NEVER THE PROBLEM. THE ROLL WAS.** Confirmed from a browser on
a real patient scan: the mandible still rendered as a maxilla after everything
above shipped. The occlusal SIGN was correct all along — checked three
independent ways (cluster count per slice, face-normal dispersion, skewness, all
agreeing), and the backend's own `u_occ` agrees with them to 3.9°.

What was wrong was `camera.up`. It was set to the occlusal axis while the camera
sits **34.7° off that same axis**, so the screen's up direction was whatever
survived projecting `u_occ` onto the view plane — about the scanner's −X on this
scan. That is an arbitrary axis, and an arch rolled 180° in-plane reads exactly
like the opposing jaw. The console line the app now prints is what pinned it:

```
[Arch Framing] mandibular: occlusal axis -0.053,-0.113,-0.992 ... camera 55.0,-14.5,-84.3 up -0.053,-0.113,-0.992
```

`camera.up` identical to the occlusal axis, camera 34.7° from it.

**`archAnterior` fixes it, and the signal is anatomy rather than a heuristic:**
a dental arch is widest between the molars and narrowest at the incisors —
intermolar ~55 mm against intercanine ~35 mm — for every human arch, upper or
lower, and it survives a partial scan. So of the two in-plane axes the SAGITTAL
one is whichever shows the greater width contrast between its ends, and ANTERIOR
is the narrow end. Measured on the real scan, outer quartiles, width taken along
the other axis:

| | one end | other end | ratio |
|---|---|---|---|
| axis A | **38.6 mm** | 71.5 mm | **1.85** ← sagittal |
| axis B | 44.3 mm | 43.3 mm | 1.02 |

Two independent checks confirm the sign: the point centroid sits 3.09 mm
anterior of the mid-extent (a U opening posteriorly), and slicing across the
arch gives two arms posteriorly against one blob anteriorly
(`[2,2,2,2,2,2,2,2,2,2,1,1]`). Below a 1.15 ratio the rule REFUSES and the roll
is left where it was, because returning a coin flip here reproduces the exact
symptom being fixed.

The resulting `camera.up` on the real scan is `-0.170,-0.979,0.117` — **90.0°
from what it was**. The load-time view now agrees with the post-occlusal-plane
view instead of contradicting it, and both put the anterior teeth at the bottom
of the screen for a mandible.

*(A note on how this was found, because it is the second time in this section.
The planning pass called §2 "the mandibular framing fix" and reasoned entirely
about the occlusal sign. The sign was already right. Neither the plan nor the
first implementation asked the simpler question — what is `camera.up` actually
set to? — and no headless test could have, because every one of them checks an
axis and none of them checks the roll. It took one line of console output from a
real browser.)*

**Setting the occlusal plane re-aims the camera only when the guess was
wrong.** `handleDefineOcclusalPlane` compares the stored load-time axis against
`data.u_occ` and re-frames on `dot < 0` alone. Re-framing on a correct guess
would yank the view back to default from wherever the clinician had just
orbited to, three clicks into their workflow, for no visible reason.

### 20.2 The shadow rig had never executed once

Until `2d372b4`, `three.current` was reassigned after `sun` and `catcher` were
attached to it, so `aimShadows` hit `if (!sun || !catcher) return;` on every
call and returned silently. Fixing that did not fix a regression — **it turned
a path on for the first time**, which is why "Set Occlusal Plane turns the
scene black" appeared immediately afterwards. The response is therefore to
harden a path nobody has watched run, not to hunt a regression in old code.

`frontend/src/shadowRig.js` is new and holds all of the arithmetic and none of
the three.js wiring: `computeShadowRig({box, u_occ, u_sag, u_tra})` takes plain
arrays, imports nothing, and returns plain arrays. What changed inside it:

* **`near`/`far` are computed.** They were hardcoded `1` / `400` at
  construction and `aimShadows` only ever set `left/right/top/bottom`. Being
  precise about when 400 is actually wrong, because the loose version of this
  claim does not survive measurement: one arch puts the catcher at **156.4 mm**
  of light depth and two arches in occlusion at **180.9 mm** — both
  comfortably inside 400. The crossover is a combined bounding radius of
  **101.6 mm**, which no single mouth reaches. What reaches it is rule 3.1:
  this app never re-centres a scan, so two arches captured against different
  scanner origins keep that separation in the combined box, and 180 mm apart
  puts the catcher at **473.8 mm**. The planes were not wrong for a mouth; they
  were unconnected to the scene.
* **The sun is off-axis**, at `+u_occ + 0.35·u_sag + 0.25·u_tra` — 23.3° off
  the occlusal axis. A light exactly along the view axis lights every visible
  surface head-on and the cusps lose their modelling.
* **Intensity 1.2**, up from 1.1.
* The catcher stays on the `−u_occ` side, which is already correct for both
  jaws once you know `u_occ` points out of the mouth.

**THE BLACK VIEWPORT WAS THE CATCHER, AND THE GUARD BELOW DOES NOT CATCH IT.**
Confirmed in a browser after the first pass shipped: the cast still went dark on
"Set Occlusal Plane". The rig was computing a perfectly finite answer, so the
degrade-to-no-shadow guard never fired — it was guarding against the wrong
failure.

The catcher was `PlaneGeometry(400, 400)`, fixed, while the shadow camera's
orthographic box is `2 x 1.6 x radius` — **151.7 mm** on a real arch. So
**85.6% of the catcher lay outside the shadow map**, where sampling the depth
texture with clamped UVs returns SHADOWED. And 400 mm of plane at the catcher's
distance covers 143 mm of visible frame, so the plane fills the viewport on its
own: the misread is a full-screen 0.22-alpha black wash over everything behind
the cast. Sized to the frustum (`rig.catcherSize`), every texel it samples is
one the shadow map actually rendered.

Three things this cost, worth recording because the mistake was in the
reasoning and not in the arithmetic:

* The planning pass asserted the black screen was "a path nobody has ever seen
  run" and hardened the arithmetic. The arithmetic was fine. The defect was a
  plain sizing mismatch that had been sitting in the constructor since the rig
  was written, and would have been found by asking "what is 400 measured
  against?" rather than by auditing the maths.
* `receiveShadow` is set on exactly one object in the app, the catcher — which
  was recorded below as evidence that self-shadow acne could not be the cause.
  That was true and it pointed at the catcher, and the note stopped one step
  short of saying so.
* A guard is only as good as its model of the failure. This one turns a
  non-finite rig into a missing shadow, which is right, and does nothing at all
  about a finite rig that is wrong.

**A failure degrades to no shadow, never to no image.** `aimShadows` wraps the
apply in `try/catch`; if the rig is non-finite or throws, `sun.intensity` stays
`0` and `catcher.visible` stays `false`, and the console says so. The scene
keeps the camera-parented three-point rig and stays lit. A viewport with no
shadow is a cosmetic loss; a black one is a dead tool.

**§3.2 and §3.3 needed no work and were not touched — with one caveat that is
recorded rather than silently satisfied.** The 3-point rig IS camera-parented,
at `App.jsx:522-528` (`camera.add(key, fill, rim)`), which is the substantive
requirement: the cast stays lit from every viewing angle however the raw
scanner axes happen to be oriented. **The intensities are not the brief's
numbers.** It asks for key 1.0 / fill 0.5 / rim 0.4; the rig ships 2.1 / 0.55 /
0.9 against `scene.environmentIntensity` 0.45 and ambient 0.18. Those were
tuned against the PMREM environment and the clearcoat, and changing them to
match a number in a brief would darken a viewport this section exists to stop
going dark. Left alone deliberately.

`tissueMaterial` (`App.jsx:84-91`) already carries `roughness 0.35, metalness
0.05, clearcoat 0.6, side: DoubleSide` exactly as §3.3 specifies. Nothing sets
`receiveShadow` on the arch or the crowns, so self-shadow acne cannot be the
cause of a black screen — worth recording, because it is the obvious suspect
and it is ruled out.

### 20.3 Two new Node checks, and what they can and cannot prove

`frontend/verify-framing.mjs` (25 checks) and `frontend/verify-shadowrig.mjs`
(43 checks) run under plain `node`, alongside `verify-kinematics.mjs`.

The framing fixtures are synthetic arches built from anatomy — a broad gingival
band under discrete crowns of unequal height at unequal distance from the arch
centroid — and the rule is asked to recover an occlusal direction it was never
told. **That part is only as good as the model.** The translation and rotation
cases are not: they hold for any input at all, and they are the ones that would
have caught the real defect. The old rule is reproduced in the file and shown
answering differently for the same cast placed 60 mm either side of where it
started.

A deliberate adversary pins the documented tie-break: a thin spike over a
ragged slab, where skewness says one end and radial spread says the other.
Skewness wins and the disagreement is logged. The spike is kept short on
purpose — the first draft reached +28 and made σ_z 7.4 against σ_xy 7.8, at
which point Z stopped being the flattest axis, PCA picked something in-plane,
and the fixture stopped testing the tie-break at all.

**Neither file can tell you the viewport is lit.** There is no browser here.
They assert that every number handed to a light is finite, correctly signed and
inside its own frustum, so that if the screen is still black the arithmetic is
excluded and the search moves to the three.js wiring.

### 20.4 Print compensation goes AFTER the union, not before it

The brief asks to dilate moved crowns by 0.15 mm before the CSG boolean. That
is the right instinct in the wrong place. `build_stage_bundle` does
`OpType.Add` — a UNION producing the POSITIVE a lab draws a sheet over, with
the sockets filled flush. Dilating each crown before that union pushes every
tooth surface outward, so the finished tray is oversized on **every wall** by
the full offset. At 0.15 mm that is **60% of this app's own
`max_translation_per_stage` of 0.25 mm**: the aligner would give away most of a
stage of prescribed movement as slop. Measured on a fissured crown, +0.15 mm
takes it from **131.9 to 153.5 mm³, +16.4%**.

The 0.15 mm in `carve_socket` / `export_nested_pair` is untouched. That path is
a nested insert, where a clearance between two parts genuinely belongs.

So `StageExportRequest` gains **`print_compensation_mm: float = 0.0`**, applied
after the union to the whole fused solid via `cg.offset_along_normals`:

* **Default 0.0 — off unless a lab asks for it.** A test asserts 0.0 leaves
  every stage STL byte-identical, which also rules out a NaN normal multiplied
  by zero.
* The index buffer is untouched, so `closed`, `components == 1` and the welded
  edge counts still measure the model actually written. `volume_mm3` is
  recomputed with `cg.signed_volume`, because `solid.volume()` is the
  pre-offset Manifold and no longer describes the STL.
* **Out of range is refused, never clamped.** Non-finite, negative, or above
  `MAX_PRINT_COMPENSATION_MM` (0.5 mm — twice a full stage of movement) raises
  a 422 naming both the ceiling and the `MAX_TRANSLATION_PER_STAGE_MM` it is
  measured against. A silently clamped clearance is invisible in the exported
  STL, which is the one place it would matter.
* The manifest carries `print_compensation_mm` and `print_compensation_note` on
  **every** export, 0.0 included, beside `socket_treatment`. A lab reading
  "0.0" knows the model is true to anatomy; a lab reading nothing has to guess.

**What is not shown.** The growth was measured at +3.5% worst case on the
two-crown synthetic arch from `moved_session()`, not on a real segmented scan —
the +16.4% above is the isolated fissured-crown probe and was not re-derived on
a fused model. And no lab has consumed the new manifest fields; that they are
the fields a lab actually wants is a design claim, not something a test settles.

### 20.5 The export mirror is not a place to run the app from

`Aligner_App_AI_Export/` is a snapshot for handing to a reader. It is a full
tree copy, so it contains `api_core.py`, `start_backend.bat` and
`start_frontend.bat` — and `build_ai_export.py` skips every `.h5/.pth/.ckpt/.pt`
by design (`SKIP_EXT`: a model checkpoint is not source). The project root has
12 checkpoints; the mirror has none and never will.

A backend started in there therefore comes up with **all of the code and none of
the model**, and reports:

```
FileNotFoundError: checkpoint not found:
  ...\Aligner_App\Aligner_App_AI_Export\ToothGroupNetwork\ckpts\0707_cosannealing_val.h5
```

which reads as a corrupted install. It is not. It is the wrong working copy.

**What makes this worth a guard rather than a note.** The mirror is refreshed
from the tree, so its frontend is CURRENT — the app starts, the viewport works,
every recent fix is present, and the only thing missing is the AI. There is
nothing to notice. It cost a full diagnostic round here: the checkpoint was
confirmed present and loadable (17.5 s) against a path that the running backend
was never using, and the first explanation offered — a OneDrive placeholder —
was wrong.

`api_core.py` now detects `Aligner_App_AI_Export` on its own path at import,
prints a banner to the backend console, and replaces the `error` field in
`/api/ai/status` so the health chip says *"Wrong folder: this backend is running
from Aligner_App_AI_Export..."* instead of a truncated file path. The chip
truncates at 60 characters, so the actionable words come first.

Refreshing the mirror is what makes it look runnable, and that is a real
trade-off: a stale mirror misrepresents the code, a fresh one invites this
mistake. It is kept fresh and guarded rather than left stale.

### 20.6 The audit trail refused every upload, and the denylist was right

Found by running the stack and reading the server log rather than by any test:

```
[audit] entry refused (ValueError: Refusing to audit ['faces'] - an audit entry
records WHAT changed, never the patient or the mesh.)
```

The upload call site passed `values={"faces": int(len(faces))}` - a scalar
COUNT - and `faces` is on `audit.FORBIDDEN_KEYS` because there it names the
triangle ARRAY. So `audit.record` refused the whole entry, `_record` caught the
refusal and printed it, the request carried on, and **every scan upload since
the audit trail was wired in went unrecorded**. `scan_loaded` never once fired.

**The denylist was correct and the call site was wrong**, which is the reverse
of the instinct when a guard rejects your data. Renaming to `vertex_count` /
`face_count` is the whole fix.

**This is the SECOND time this exact confusion has been got wrong.**
`telemetry.py` splits `IDENTIFIER_KEYS` from `GEOMETRY_KEYS` precisely because
`faces` is an array in an audit entry and a scalar in a span (§18) — and then
the audit call site fell into it anyway, three sections later, in the same
commit series that wrote the explanation.

So it now has a test that reads the SOURCE instead of waiting for a clinician to
perform the action: `test_no_audit_call_site_uses_a_forbidden_key` walks
`api_core.py`'s AST, finds all **6** `_record` call sites, and asserts no literal
key in any of them appears in `audit.FORBIDDEN_KEYS`. A companion test asserts
the denylist still refuses a real mesh, so the first cannot pass by the guard
going soft.

**The lesson worth keeping: a runtime denylist is only discovered when the
guarded action is taken.** Everything else in this codebase that guards at
runtime — the PHI filter, the alveolus gate, the CSG cascade — is exercised by a
test that takes the action. This one was not, because `_record` deliberately
never raises, so the failure had no way to reach a test. **A guard that cannot
fail a build needs a static check standing behind it.**

Verified against the live server after the fix:

```
AUDIT entries: 1
  scan_loaded   arch=lower   94,848 vertices, 187,625 faces
      values={'vertex_count': 94848, 'face_count': 187625, 'welded': 0,
              'open_edges': 2101, 'sanitize_repaired': False}
```

`open_edges 2101` is the same number §9 measured on this scan, which is a useful
cross-check that the entry describes the mesh that was actually loaded.

### 20.7 The mirror no longer ships the buttons that start it

§20.5 guards the export snapshot at import and prints a banner. That leaves the
launchers sitting in the mirror looking exactly like the real ones, and
`start_backend.bat`'s checkpoint step only WARNS:

```
[2/3] AI checkpoint...
      NOT FOUND - the API still starts and manual cutting still works,
      but "Segment Teeth" will report the model as unavailable.
```

which is easy to scroll past on the way to a working-looking app.

Two changes, defence in depth:

* **`build_ai_export.py` withholds both launchers** (`LAUNCHERS`, folded into
  `SKIP_NAMES`) and writes `DO_NOT_RUN_FROM_HERE.txt` in their place, explaining
  what the snapshot is for and where to run the app from. **The cleanest guard
  is the one where the button does not exist.**
* **Both launchers refuse outright** when their own `%CD%` contains
  `Aligner_App_AI_Export`, naming the folder they are in and pointing at the
  parent. This covers snapshots already extracted somewhere before this change.

Measured: the rebuilt archive is **267 entries, 0.81 MB**, with
`start_backend.bat` and `start_frontend.bat` absent and `DO_NOT_RUN_FROM_HERE.txt`
present — asserted after the build, not assumed.

### 20.8 The stack, run end to end

Neither service was running when this session started, which is itself the
commonest form of "the AI is unavailable" (§12 added the health badge for
exactly this). Started and measured against the real scan:

| | |
|---|---|
| backend first answer | immediate, `warming: true` |
| AI model ready | **12.3 s** (19.1 s on a second cold start) |
| `/api/ai/status` | `loaded: true`, `error: null`, `device: cpu` |
| Vite dev server | ready in **1.19 s**, `http://localhost:5173` |
| upload, 9.4 MB scan | **2.9 s**, 94,848 verts / 187,625 faces |
| CORS on the dev origin | `access-control-allow-origin: http://localhost:5173` |
| occlusal plane | 0.03 s |
| wand | 0.09 s |

**The 12 checkpoints are present in the project root and the model loads from
it.** There was never anything wrong with AI availability in this working copy;
what was wrong is that nothing had been started, and the one way to get it
persistently wrong — starting from the snapshot — is now refused in three
places.

One thing that is NOT a defect, recorded so it is not chased: a wand click at an
arbitrary point flooded **13,697 vertices** of a 94,848-vertex arch and the
subsequent cut was refused 422 *"Crown did not close watertight."* That is the
correct answer to a selection that is not one tooth. It is the guard working,
not a bug in the wand.


### 20.9 Still unverifiable here

**There is no browser in this environment.** The black viewport cannot be
reproduced or confirmed fixed headlessly, and neither can the upside-down
mandible. The framing rule is proved translation- and rotation-invariant in
Node and correct on the one real scan in the repo; that it fixes *your*
inverted mandible needs one load in a browser. Both are flagged rather than
reported as verified.

**No frame-rate figures.** §5 asks for 60 FPS and there is no profiler here.
The claim rests on the existing architecture — refs plus rAF, React kept out of
the per-frame path — and the prior 12.5 µs/frame scrub measurement, not on
anything measured this pass.


## 21. RESOLVED (PARTIAL): LOCAL TARGET-POSITION CAST RECONSTRUCTION

**The exported cast distorted when a tooth moved, and the cause was
architectural.** `build_stage_bundle` filled the ORIGINAL socket flush, built
ONE static cast, then unioned that same cast with `crown + a 9mm root plug`
transformed by the stage matrix. Nothing reconstructed the cast at the tooth's
NEW cervical position, and the plug travelled with the tooth — so an extruding
tooth carried synthetic root material up out of the gingiva as visible positive
geometry. `_rim_plug`'s own docstring said so: *"DEPTH IS THE ROOT LENGTH, not
some small seating value."*

Measured before the change, two teeth with one extruded 1.2mm over 5 stages:
fused volume grew **27579.63 → 27608.44 mm3 (+28.84)** monotonically, and every
stage read **68–77 welded non-manifold edges** with
`survives_a_welding_reader: False`. The gate missed all of it because it
validated the in-memory index buffer rather than the bytes written.

**The interface is BIDIRECTIONAL, and that is the correction.** A moved tooth
is penetrating (intrusion, tipping in) — tissue must be REMOVED — or separated
(extrusion, tipping away) — tissue must be ADDED — and usually both around
different parts of one rim. **Separation is not a refusal.** An ordinary
extrusion lifts the whole rim clear of the gingiva; refusing that would reject
the movement local reconstruction exists to handle.

**Five defects were found in the new code by the measurements themselves:**

* **Cast thickness read 0.1mm on a 9mm cast.** A signed-distance march using
  nearest-VERTEX distance flips sign on a steep cervical wall, so the march
  "left the solid" immediately and every interface was refused `wall_too_thin`.
  Möller–Trumbore against the real triangles has no such failure mode.
* **The ramp was a cylindrical collar** — precisely the artificial annular
  ledge the brief forbids. Its radius is now per rim point and proportional to
  that point's lift, so it vanishes where the tooth never left the tissue.
* **Ramp volume 201mm3 for a 1.2mm lift**, because `_loft` caps both ring
  interiors and builds a solid frustum rather than a blend.
* **Unbounded drop rays landed on the base underside 10–12mm down**, which made
  the envelope check refuse 8 of 10 ordinary movements.
* **`surface_deviation` reported 3.46mm of "deformation" 18–32mm from any
  tooth.** Nearest-VERTEX distance on a base whose underside carries very large
  triangles calls a point sitting exactly ON the original surface a 3.5mm
  deviation. Point-to-TRIANGLE distance reports mean **0.0046mm**, p95
  **0.0000mm**.

**A sixth was a positional-argument bug of my own making:** adding `seat_verts`
to the result dataclass before `diagnostics` meant `diag` was passed into the
seat slot and then overwritten. Nothing raised; the seat was built correctly and
every diagnostic silently vanished. The constructor takes keywords now.

**Boolean order, and it was measured rather than assumed:**

```
base + ramps        build the tissue first
     − cavities     then cut the socket through it
     ∪ seats ∪ rigid crowns
```

Subtracting first removes the cast material the ramp has to land on and orphans
it — the model went from 1 body to 4 across an extrusion.

**`root_length_mm` no longer reaches the manufacturing path at all**, and the
proof freezes the stage matrix: varying root length 9→13mm with M held fixed
leaves the cavity volume, seat volume and depth bit-identical. Varying it
through the whole pipeline would legitimately move C_res and prove nothing.

**Measured after the change**, same case: welded non-manifold edges per stage
**0, 1, 0, 1, 0** (was 75, 77, 73, 70, 68); unaffected cast deviation outside
the ROI mean **0.0046mm**, p95 **0.0000mm**, p99 **0.041mm**, with 23 of 2910
points reaching 1.15mm on the trim boundary where the boolean retessellates the
base outline.

**RESOLVED 2026-09-20 — the stage fragmentation was caused by an UPSTREAM
GEOMETRIC SIGNED-DISTANCE CLASSIFICATION DEFECT.** It is a geometry-processing
bug, in the part of the pipeline that decides where the cast surface is.
Extrusion measured `1, 1, 2, 2, 3` bodies with only 1 of 5 stages passing; it
now measures `1, 1, 1, 1, 1` and **5 of 5 stages pass the current
boolean/topology regression** — NOT "print ready", which is a wider claim the
aggregate gate does not yet make. `test_staging_export.py` and
`test_failsafes.py` pass on merit, untouched.

`CastProbe.signed` was nearest-VERTEX distance signed against that vertex's
normal. `bench_signed_distance.py` scores it against an INDEPENDENT ground
truth — exact closest-point-on-triangle for the magnitude, a three-ray parity
vote for the sign, neither of which is any of the candidates — over 870 points
in 15 difficult classes:

| class | old, as shipped | old + compaction | exact |
|---|---|---|---|
| steep cervical wall, 0.05mm in | **13.3%** | 90.0% | 100% |
| concavity, 0.05mm in | 25.0% | 76.7% | 100% |
| ALL | 77.7% | 91.4% | **100%** |
| max magnitude error | 6.66 mm | 6.66 mm | **0.0014 mm** |

13.3% is worse than a coin toss, and **a cervical rim IS a steep cervical
wall** — `rim_signed` decides lifted-versus-seated per rim point and sizes the
whole transition volume, so a wrong sign built the bridge to the wrong height
at scattered points around the margin, and the self-intersecting result
survived as a non-manifold edge.

**TWO INDEPENDENT DEFECTS, and the middle column is why both had to be fixed.**
`build_cast_base` returns a face subset over the SCAN's vertex array and rule
3.1 forbids rebuilding it, so the array keeps every trimmed-away crown —
measured, **4497 of 8372 vertices unreferenced**. `cg.vertex_normals` leaves
those at ZERO, so `outward` is 0.0, `0 < 0` is False, and an interior point is
reported OUTSIDE at the distance to a phantom the surface does not contain.
Compaction alone recovers most of the accuracy, which means the phantom
vertices were the dominant term and the steep-wall diagnosis was the smaller
one. Neither fix alone reaches a number a gate may rely on.

**THE EXACT METHOD IS ALSO THE FAST ONE**, which removes the usual reason to
keep an approximation. On the 7,824-face cast, warm-up excluded: at 44 points
(a rim) Open3D n=11 is **0.529 ms against 4.857 ms**; scene build is 0.52 ms,
once. No dependency was added — Open3D and trimesh were both already installed
and load-bearing (§19). `trimesh.proximity.signed_distance` needs `rtree`,
which is absent here; `closest_point_naive` gives the same accuracy without an
index and was benchmarked instead.

**WHERE THE TOPOLOGY ACTUALLY BREAKS — PROVEN, not inferred.** The chain is
now measured at four points instead of two, and the earlier welding hypothesis
was wrong in an instructive way. On extrusion 0.25mm stage 2:

| point | open | non-manifold |
|---|---|---|
| after the boolean, in memory | 0 | **0** |
| after OUR export weld | 0 | **3** (merged 20) |
| after STL write + reread | 0 | 3 |
| after the reader's weld | 0 | 3 (**merged 0**) |

So: **STL serialisation does not change the topology, and the downstream weld
does not either** — it finds nothing left to merge. manifold3d's output is
clean, and welding the coincident-but-distinct vertices it deliberately keeps
is what creates the non-manifold edges. `parse_stl_bytes` already dedups by
position, so the raw reread of an UNWELDED write reads 40 non-manifold edges
where our weld leaves 3; the difference is the degenerate faces `weld_vertices`
drops and a plain reader keeps. Either way the mesh carries coincident vertices
at the crown/cast interface, and that is a property of the GEOMETRY.

> **SUPERSEDED 2026-09-21 by §23.** The three paragraphs below are wrong and
> the measurements that refute them are in §23: manifold3d's output is NOT
> clean (it carries coincident positions wherever the solid touches itself),
> a small movement is NOT worse than a large one (the 1.2mm case has the same
> defect at every stage and passed only because its edges were short enough
> for the repair rung), and what was missing was not a connector that bites
> the post-clearance cast - the connector was crossing the crown's and the
> cast's CREASES. They are left in place because the reasoning is what §23
> corrects.

**THE GEOMETRIC CAUSE, and why the prescribed fix does not yet work.** A
barely-moved crown sits almost exactly in its own filled socket, so its outer
surface and the cast's are the same scan triangles microns apart over a long
band, and the union produces a solid that TOUCHES ITSELF along it — valid as an
indexed mesh, inexpressible in any position-based format. A small movement is
therefore worse than a large one: 1.2mm is clean across all five stages, 0.25mm
is not.

Penetration-aware crown-derived local clearance was implemented and measured in
two forms — a uniform +0.05mm dilation of the transformed crown, and a
depth-modulated version pulling the tool INSIDE the crown where it is buried
(+clearance at the surface, −fusion_overlap past a 0.15mm band). **Both fix
extrusion 0.25mm and both break the tipping, rotation and buccolingual cases
into two bodies**, 8 of 18 matrix cases against 5 without. The measurement that
explains it: `crown_points_in_graze_band / deeply_buried` reads **113 / 14** and
**121 / 11**. The crown is in its own socket, so nearly every vertex grazes and
there is no buried region for the modulation to hold on to — clearing the graze
band clears essentially the whole interface, and only the seat is left to fuse.
`overlap_seat_cast_mm3` still read 51–67 mm³ throughout, because it is measured
against the PRE-clearance cast and cannot see the void.

The tool is therefore **measured and deliberately not emitted**
(`clearance_tool_emitted: false`), with the graze fraction, band, offsets and
tool volume recorded per tooth per stage. What is missing before it can be
emitted is a connector guaranteed to bite the POST-clearance cast; enlarging
the seat to cover it would be the parameter sweep this work explicitly avoids.

> **SUPERSEDED 2026-09-21 by §23.** All five of these cases now pass, and the
> order-independence one was testing the wrong thing entirely.

**WHAT IS STILL NOT FIXED, measured and reproducible.** `pytest` is
**294 passed, 5 failed**, and all five are in the new
`test_manufacturing_matrix.py`, left failing deliberately as the only
automated signal:

* `extrusion 0.25mm` stage 2 — 3 non-manifold edges
* `tip 8 deg + 0.5mm extrusion`, `rotation 8 deg`, `combined` — 1–2 edges
* `test_the_fused_result_does_not_depend_on_tooth_order`

The cause is a **grazing CSG contact at the cervical margin**, not fragmentation:
bodies 1, components 1, zero open edges, and one to three non-manifold edges
after the reader's weld. Measured on an intrusion: ONE edge 0.0204 mm long
carrying FOUR faces, 0.204 mm from the moved tooth's original rim, one incident
face of area 2.6e-5 mm² whose normal is exactly anti-parallel to its neighbour,
and both directed edges appearing twice — which is why winding fails with it.

`mfg.collapse_short_nonmanifold_edges` repairs the micro-slivers, capped at
0.05 mm (scanner resolution is 20–50 µm), all-or-nothing, and RECORDED in the
manifest as `nonmanifold_edge_repair`. It took the matrix from 14 failures to
5. It deliberately refuses the remaining edges, which measure **0.066, 0.49,
1.84 and 2.05 mm** — a 2 mm non-manifold edge is two surfaces genuinely
meeting along a line, and collapsing that would be hiding a defect rather than
repairing one.

> **SUPERSEDED 2026-09-21 by §23 — this claim is FALSE.** Self-touch counts
> across the whole extrusion sweep are of the same order at 0.0mm, 0.25mm,
> 1.0mm, 1.2mm and 2.0mm.

**A SMALL EXTRUSION IS WORSE THAN A LARGE ONE, which points at the fix.** 1.2 mm
is clean across all five stages; 0.25 mm fails. At small movements the crown
sits almost exactly in its original socket, so its outer surface and the cast's
gingival surface are the same scan triangles a few microns apart over a long
band — maximal grazing. The indicated repair is the one the brief prescribes
and this pass did not implement: cut the cast back from the crown
(`cavity_outset_mm`) for penetrating teeth so the surfaces meet transversally
instead of grazing.

**The real-scan regression was NOT run for manufacturing.** Nothing in §21 is
based on it.

Covered by `test_manufacturing_interface.py`, `test_manufacturing_matrix.py`,
`bench_signed_distance.py` and `MANUFACTURING_RECONSTRUCTION_CHANGELOG.md`.
**The counts in this section are historical — see §23 for the current ones.**

## 22. RESOLVED: OCCLUSAL-PLANE ESTABLISHMENT DARKENED THE CAST

Reported from manual testing: load an arch, click the three occlusal-plane
landmarks, and on the third click the whole cast goes dark. Tracked and fixed
as a **viewport presentation bug**, entirely separately from the manufacturing
reconstruction — no STL, CSG, C_res or kinematics path was touched.

**IT WAS THE SHADOW RIG, and the A/B that proves it had to gate `aimShadows`
BEFORE it armed rather than turn it down afterwards.** Measured in a real
browser against the production build on `case_lower.stl`, luminance taken from
the actual framebuffer:

| | lit pixels | mean luminance |
|---|---|---|
| before the plane | 92,589 | 193.74 |
| after, rig armed | **0** | — |
| after, rig gated out | **93,503** | 200.50 (**1.035x**) |
| rig then armed via the toggle | 93,530 | 206.60 |

The first attempt measured "shadows off" AFTER arming and found the cast still
gone, which pointed away from the rig and cost a long detour. Dropping the
intensity does not undo what arming already did — `aimShadows` also sets
`castShadow` on the arch and resizes the shadow camera — so the control now
gates the rig at the top of `aimShadows`, and the comparison is clean.

**What was excluded, by measurement rather than by argument.** The cast is not
darkened: with the rig armed the renderer still submits all **187,625
triangles**, the material stays visible with `colorWrite` on, `side` is
DoubleSide, normals are unit length (0 degenerate of 94,848), the colour buffer
reads mean 0.478, the cast projects to NDC (0, 0) inside the frustum at the
same distance as before, and every light still points at it. Orbiting 180°
does not bring it back. The catcher is not between the camera and the cast.
The backend's occlusal frame is correct: `fit_occlusal_frame` agrees with an
independent PCA/skewness axis to **0.9°** with a sign margin of +8.6 mm.

**A SECOND, REAL DEFECT IN THE RIG was found on the way and fixed.** §20.2
sized the catcher to `2 * s`, edge for edge with the shadow camera's box, and
`verify-shadowrig.mjs` asserted exactly that — so the check passed while
**three of four catcher corners sat outside the shadow map**, at NDC 1.13 and
1.23. Two things a side-length test cannot see: a square's corners reach √2
further than its edges, and the catcher is dropped along `−u_occ` while the sun
is 23.3° off `u_occ`, so its *centre* is already ~0.4·radius off the light
axis. A fragment outside the map samples the depth texture with clamped UVs and
returns SHADOWED whatever is really there. The ortho box is now DERIVED from
the largest perpendicular distance from the light axis of anything that must be
in it — rotation-invariant, because three.js orients the shadow camera with the
light's own `up` and `shadowRig.js` does not model that. Measured: half-box
62.3 mm → 106.4 mm, corners outside 3 → **0**.

**What shipped.** The rig is NOT deleted. `shadowsOn` defaults **off**, a
clearly-labelled *"Shadows (display only — never affects export)"* control
re-arms it, and turning it on after the plane is set calls `aimShadows`
properly rather than only raising an intensity. A viewport with no shadow is a
cosmetic loss; one with no model is a dead tool — the same rule §20.2 settled.

**WHAT IS NOT EXPLAINED, and is why this is a mitigation rather than a repair.**
Why arming the rig inside `handleDefineOcclusalPlane` stops the cast
rasterising, when it neither receives shadows nor changes any material flag, is
still open. Deferring the call by two animation frames was tried and does NOT
fix it, so it is not a stale-matrix race; that deferral was removed rather than
left in as cargo. Armed later through the toggle, the identical call works.

**Three things the browser test got wrong first, each of which reported
something false**, and all three are now written into the spec:

1. **Headless Chromium composites the WebGL canvas as EMPTY** without a GL
   backend. The first run read exactly 14.79 — the clear colour `0x0d0f12` —
   both before and after, and a capture that never contains the model cannot
   tell you the model got darker. `--use-gl=angle --enable-unsafe-swiftshader`.
2. **"Define Occlusal Plane" is always in the DOM**, merely disabled, so
   waiting for it photographed an empty viewport while the status line still
   read *"Uploading case_lower.stl..."*.
3. **The Wand is armed at the same time as the picker** and listens on the same
   element, so every landmark click also selects ~21,800 vertices and
   overwrites the status line — which makes the picker's own `(n/3)` useless as
   a progress signal *and* repaints the cast between the before and after
   frames. Progress is read from the button label, which is durable state, and
   the rig is A/B tested through the toggle with the selection held fixed.

New: `frontend/src/shadowHarness.js` + `shadow-harness.html` +
`playwright.shadow.config.js` (a real WebGL rig harness with a per-pixel cast
mask, run against the dev server so it never reaches `dist/`),
`frontend/e2e/occlusal-darkening.spec.js` (2 tests, the real workflow on the
real scan, skipping rather than substituting a stand-in), and
`window.__viewportDiagnostics()` — read-only, the only thing published on
`window`, because the live objects live in a closure no test can reach.

`e2e/occlusal-darkening.spec.js` releases its backend session in an
`afterEach`: `STORE` keeps four and evicts the oldest, and a 9.4 MB upload per
test was pushing other specs' sessions out and making *their* assertions fail
downstream, which read as flakiness in files that had done nothing wrong.

## 23. RESOLVED: THE SOLID WAS TOUCHING ITSELF — THE TRANSITION COLLAR AND THE AGGREGATE GATE

**§21's diagnosis above is SUPERSEDED, and its own control is what refuted it.**
Three claims in it are wrong and are corrected here: that manifold3d's output
is clean, that a small movement is worse than a large one, and that what was
missing was a connector guaranteed to bite the post-clearance cast. The five
failing matrix cases were not five problems. They were one.

### A coincident position means the boundary TOUCHES ITSELF

Established by running the engine on cases whose answer is known, rather than
inferred from the failing ones:

| union | coincident positions | non-manifold after weld |
|---|---|---|
| two cubes overlapping, axis aligned | 0 | 0 |
| two cubes overlapping, one rotated 13/27/41° | 0 | 0 |
| cube ∪ sphere (transversal) | 0 | 0 |
| two cubes meeting FACE TO FACE (tangent) | 0 | 0 |
| **two cubes meeting along an EDGE** | **2** | **1** |

A clean transversal union produces none; even a face tangency produces none;
only a genuinely self-touching solid produces them. Binary STL stores
POSITIONS, so a reader welds whether or not we do, and welding a self-touch is
what turns it into a non-manifold edge. So the count of coincident positions —
`self_touch.coincident_position_groups`, now recorded on every stage — is the
number that actually predicts the gate.

### "A small movement is worse than a large one" is FALSE

Self-touch groups per stage, seats and crowns fused, across the extrusion sweep:

| extrusion | stage 1 / 2 / 3 |
|---|---|
| 0.0 mm | 9 / 16 / 27 |
| 0.25 mm | 7 / 18 / 34 |
| 1.0 mm | 26 / 19 / 33 |
| 1.2 mm | 23 / 27 / 43 |
| 2.0 mm | 12 / 33 / 47 |

The 1.2 mm case — §21's "clean across all five stages" — carries the defect at
every stage. It passed because every edge it produced happened to be short
enough for `collapse_short_nonmanifold_edges` to take. **The defect was
universal; the repair rung's success was not.**

### Where it touched itself, proven with provenance

Every input solid now carries a manifold3d original id (`as_original()`), which
survives the boolean and comes back on `run_original_id` / `run_index`, so
every triangle in a fused stage traces to the geometry it came from and every
offending edge is reported with the sources meeting on it. `OLD_SOCKET_REPAIR`
cannot come from a run — the flush closure is welded into the cast's own vertex
array long before any boolean — so it is separated geometrically, by the
triangle centroid lying on the cap, and that is stated in the code rather than
implied.

On extrusion 0.25 mm stage 1 all six touches read **distance-to-rim 0.08889 mm
and distance-to-nearest-cast-vertex 0.17222 mm — the same two numbers to five
decimals**: they lie on the boundary edge of the cast's flat flush socket cap,
at the midpoint of a rim edge. The rest sat on the crown's own cervical rim.

**All three geometries were built from ONE loop.** The crown is cut at
`socket_rim`, the old site is capped at `socket_rim`, and the connector was
lofted from `socket_rim` transformed — so the connector's wall crossed the cast
and the crown exactly where each has a sharp edge. A surface crossing another
surface AT ITS CREASE is a tangential contact.

### The fix: a collar that ENCLOSES the crease

The connector is no longer a plug lofted into the crown. It is a bounded collar
whose solid CONTAINS the crown's cervical crease and whose surface meets the
crown and the cast only where both are smooth. Both rings are placed by MEASURED
signed distance against the actual transformed crown and the actual cast:

* **upper ring** — starts `max(emergence_height_mm, local lift)` above the rim,
  marches up while inside the cast and outward while inside the crown until it
  is `clearance_mm` clear of both;
* **lower ring** — seeded `seat_bottom_outset_mm` outward, dropped onto the cast
  through a shrink ladder, then marched **along −grad(signed distance)** until
  it is `fusion_overlap_mm + clearance_mm` inside real cast material.

CASE A (penetrating), CASE B (separated) and CASE C (mixed) all use this one
construction and need no partition, because both rings are solved PER RIM POINT
against the real surfaces.

**ENCLOSURE IS ENFORCED WITHIN A BUDGET, NOT GUARANTEED, and the manifest says
which.** After the loft is built, `rim_k` is measured against the collar's own
signed distance and any point not inside by `clearance_mm` has its two rings
pushed radially outward, up to eight times, bounded by the same outward
allowance the neighbour clamp protects. `crease_points_outside_collar` and
`crease_inside_collar_max_mm` report what was achieved: measured on a 0.6mm
extrusion with 3 degrees of tip, one tooth left 3 of 46 crease points outside
by 0.0431mm and the fused stage still came out with ZERO self-touching
contacts. So enclosure is the mechanism, and the self-touch count is the
measurement that actually decides.

**Six things had to be measured, and each was a wrong answer first:**

1. **The upper ring must clear the CAST, not just sit above the rim.** On a
   penetrating tooth the cast surface is above the cervical margin. Every
   residual touch read distance-to-rim 0.30000 — exactly the starting height.
2. **The lower ring must SEAT.** Stopping at `clearance_mm` settled it 0.08 mm
   under the surface, so its bottom disc ran nearly PARALLEL to the flat
   old-site cap and the forensics named that touch
   `EMERGENCE_RECONSTRUCTION + OLD_SOCKET_REPAIR`. Matrix score by target:
   0.05 mm → 15/18, **0.30 mm → 16/18**, 0.60 mm → 13/18, 1.20 mm → 3/18. The
   value that works is `fusion_overlap_mm + clearance_mm`, which the policy
   already names.
3. **March along the FIELD, not the axis.** On a steep interproximal face the
   long axis runs ALONG the surface and 1.2 mm of marching never gets inside —
   a 1.2 mm extrusion was refused `wall_too_thin` on a cast 18.77 mm thick.
4. **The shrink ladder is load-bearing.** A lingual point seeded 1.2 mm outward
   lands OVER THE ARCH OPENING; one point in 44 fell through and the nearest
   surface was 3.93 mm away.
5. **The loft's end caps must be CONES.** A fan to the loop's centroid dishes
   whenever the loop is non-planar, and the upper ring is deliberately
   non-planar (0.30 mm of rise at some points, 1.96 mm at others). The dish
   sagged BELOW the crease it was meant to enclose.
6. **The upper ring must clear the whole rim NEAR it.** Following the cervical
   scallop put `top[k]` 0.078 mm *under* `rim[i]`; the two crease points left
   unenclosed were inside by only 0.025 mm and 0.014 mm. A running maximum over
   ±3 indices fills local dips and leaves the ring low where the rim is
   genuinely low, so the collar does not become a tower to fix a notch.

**The collar rises as far as the tooth has lifted.** With a fixed 0.30 mm start,
`transition_quality` measured **0.771 mm of a 0.946 mm fall in ONE 0.25 mm
ring — 82%** — the "abrupt ring / vertical wall" the brief asks to reject, and
the gate rejected it. Rising with the lift spreads the same fall across the
crown's own flare: 0.312 mm of 0.446 mm, share 0.699, not a ledge.

### CASE A: crown-derived clearance, emitted under a measured guard

The tool is an EXACT dilation — a Minkowski sum with a `clearance_mm` sphere,
taken once per tooth on its T0 crown and carried by the same stage matrix,
because dilation commutes with a rigid transform. Measured on a 250-triangle
crown: 8 segments 325 ms / 49.836 mm³, 12 segments 520 ms / 50.107 mm³. The
array alternative, a vertex-normal offset, gives 46.86 mm³ and **can fold** — it
appeared in a fused stage as a `LOCAL_CLEARANCE` self-touch, which is §8's
normal-offset failure in a third place.

It is emitted only if the subtraction leaves the cast as ONE body, because it
fragmented the cast into **4 bodies** on extrusion 0.25 mm stage 1. A withheld
tool is recorded with its reason, and `interface_total_cavity_volume_mm3` now
reports what was ACTUALLY REMOVED — it used to report the volume of a tool that
was never used, telling a lab 85.16 mm³ had been excavated where nothing was cut.

### Four measurement defects that were failing gates on correct geometry

* **`_point_to_surface` was an approximation calling itself exact.** Its
  24-nearest-centroid shortlist misses the triangle a point sits on when the
  cast's underside triangles are 20 mm across: 793 of 4710 fused-stage vertices
  reported exactly 9.0000 mm while the cast reproduced through manifold3d to
  2e-6 mm. Now Open3D's BVH, with the candidate search as fallback.
* **The fidelity reference carried 4497 phantom vertices** (rule 3.1 keeps every
  trimmed-away vertex), reporting **3.39 mm of "cast deformation"** from points
  not on the cast. The comparison copy is compacted; the scan's array is not.
* **The old-site height field read the cast's UNDERSIDE** — a crater 22.58 mm
  deep, the cast's own thickness, and 13 patches. Faces are now restricted to
  those facing occlusally first.
* **An old site the tooth still covers is not a defect.** Below a quarter of the
  site assessable, the answer is `assessable: false` with the obscured
  fraction — determinate, not unchecked. And `largest_local_step_mm` (the
  surface's own relief, 1.71 mm on a pristine cast) is REPORTED while
  `largest_step_change_mm` is what the gate uses.

### The aggregate manufacturing gate

**`mfg.aggregate_print_gate` is the only thing in this codebase entitled to say
PRINT READY.** It is a pure function of the stage record, so a measurement that
was never taken fails it exactly as a bad one does — `NOT_CHECKED` is not
`CLEAR`, and an empty record fails all sixteen gates, which is asserted.

written-STL topology · single positive Manifold body · body count agrees with
the STL · no self-touching boundary · synthetic exposure within bound · no
exposed clearance wall · two-sided unaffected-cast fidelity · reconstruction
inside the envelope · every interface built · interface continuous around every
rim · no transition ledge · old site restored · crown is an exact rigid
transform · gingival bridge preserved · root-length independent · clinical
consistency.

`/export/final` enforces THIS gate, its manifest carries the gate's own answer
instead of a hard-coded `print_ready: True`, and the client reads
`X-Print-Ready` rather than inferring readiness from HTTP 200.

**Exposure is measurable only because of provenance**: a triangle that survives
to the fused boundary and came from the connector IS exposed synthetic anatomy;
one buried by the crown or the cast is simply absent. Measured on extrusion
0.25 mm stage 1 — 8379.20 mm² cast, 164.70 mm² connector, 6.87 mm² old-site
closure, 84.59 mm² crown: **1.99% synthetic, 0.00 mm² unattributed, 0.00 mm² of
clearance wall.**

### The envelope, enforced; and two tests that were testing the wrong thing

`affected_region` is no longer diagnostic-only. The ALLOWED RECONSTRUCTION
ENVELOPE is the union, over every moved tooth, of the cast surface within
`roi_radius_mm + seat_bottom_outset_mm` of the TARGET rim and of the T0 rim.
Every cast point that moved further than `max_unaffected_deviation_mm` must lie
inside it; outside it the cast is compared BOTH ways by point-to-TRIANGLE
distance. Measured on extrusion 0.25 mm: **0.0 mm both directions, 0 modified
points outside the envelope.**

**Order independence.** The old test reversed the PRESCRIPTION LIST, which does
not reverse an order — it gives the two teeth each other's movement. That is a
different model: measured, the two "orders" differ by **0.580 mm** while their
volumes agree to 5e-4. `build_stage_bundle` now takes `part_order`
(forward / reverse / sequential), changing only the sequence the identical
solids reach the boolean in, and the test compares written STLs by two-sided
point-to-triangle distance.

**The gingival bridge.** `2 * fusion_overlap_mm` is 0.5 mm whatever was built,
so it passed for a trench of any width. It is now the minimum distance between
the two connector solids, and the connector clamps its own reach to
`0.5 × max_bridge_removal_fraction × d_neighbour − 2 × clearance_mm`, derived
from the rule. **The fraction GATES; only a merged pair is REFUSED** — §14's
rule is that a threshold with no measurement behind it may not be a refusal, and
refusing on it would block ordinary crowding: two teeth 2.36 mm apart take
50.9% and leave 1.16 mm of interdental tissue, which is a papilla, not a trench.

### Where it stands

`test_manufacturing_matrix.py` is **25 cases, all passing**, including adjacent,
crowded, converging and overlapping-cervical-rim pairs.
`test_manufacturing_interface.py` is **38**.

**THE MATRIX ASSERTS THE BOOLEAN/TOPOLOGY GATE, NOT THE AGGREGATE ONE, and the
difference is real.** Every one of the 25 cases produces a closed, single-bodied,
correctly wound STL with zero self-touching contacts; the aggregate gate then
asks the wider question, and it does not always answer yes. Measured on a
movement the matrix does not cover - 0.6mm extrusion with 3 degrees of tip over
3 stages - stages 1 and 2 are PRINT READY and stage 3 is refused
`no_transition_ledge` with `largest_step_share` 0.9274. That is the gate doing
its job on an emergence profile that is still too abrupt for that combination,
and it is recorded as a limitation rather than tuned away: the threshold was
written for the collar bug it caught, and moving it to make a case pass is the
parameter sweep this work does not do.

**REAL-SCAN MANUFACTURING REGRESSION = EXECUTED THROUGH TOOTH SELECTION ONLY.**
> **SUPERSEDED 2026-09-21 by §24.3 — the second bullet below is the right
> observation with the wrong array, and the conclusion drawn from it is
> false.** The labels index neither `verts` nor `v`: they are keyed to the
> order the segmentation pipeline's own loader produced, which is FIRST
> OCCURRENCE, while both of this project's arrays are LEXICOGRAPHIC. So the
> remap described below moved labels between two arrays that were already in
> the same order and preserved the error exactly. Transferred from the right
> source the labels ARE spatially coherent — median per-tooth box diagonal
> 13.97 mm against 50.71 mm — and **seven real teeth now cut successfully**.
> The claim that no single-tooth crown can be produced from this scan
> headlessly is withdrawn. What remains unverified is the manufacturing
> reconstruction on those crowns; see §24.7.

`real_scan_regression.py` drives `case_lower.stl` through upload, conditioning,
occlusal plane, tooth selection, cut, movement and staging entry, and stops
there because no single-tooth crown can be produced from this scan headlessly.
Both routes were measured, and both fail for their own reason:

* **the wand's auto tolerance returns 25.00 mm on every one of the twelve
  labelled teeth** and floods 26,408 of 94,848 vertices — a quarter of the arch.
  Narrowing by hand does not help: at 1.5 mm the flood is 111 vertices and only
  37% belong to the tooth clicked, because `snap_seed_to_ridge` moves the seed
  up to 3 mm to the nearest high-concavity point, which is the interdental
  sulcus — between two teeth rather than on one. Best IoU against any label
  over tolerances 1–8 mm: **0.01 to 0.05**.
* **the segmentation labels index the RAW file, not the conditioned mesh.**
  `condition_mesh` welds with `np.unique(v, axis=0)`, which reorders the array
  lexicographically even when it removes nothing — both are 94,848 vertices, so
  the count agrees and the correspondence does not. The script remaps them by
  position, which is the only thing the two arrays share.

Nothing in §23 is based on the real scan. **This is the same open issue §19 has
carried since Phase 5** — the staged export still needs one interactive wand cut
to be judged end to end — and it is now measured rather than asserted.

Covered by `test_manufacturing_interface.py`, `test_manufacturing_matrix.py`,
`real_scan_regression.py`, `bench_signed_distance.py` and
`MANUFACTURING_RECONSTRUCTION_CHANGELOG.md` §V. **The test counts in this
section are historical — §24 adds to both files.**


## 24. RESOLVED: THE LEDGE WAS A MEASUREMENT, THE SEGMENTATION WAS A MAPPING

Two blockers, both reported as defects in the thing being measured, and both
turning out to be defects in the measuring. Neither threshold moved.

### 24.1 The stage-3 ledge refusal was an artefact of nearest-vertex sampling

`no_transition_ledge` refused 0.6mm extrusion + 3 degrees of tip at stage 3
with `largest_step_share` 0.9274. `transition_quality` read its radial profile
off the **nearest VERTEX** to each probe - the fourth appearance in this
codebase of the mistake `CastProbe.signed`, `surface_deviation` and
`old_site_quality` were each corrected for. Measured on that case, the nearest
vertex sits **0.52 to 1.31mm** from the probe while the rings are **0.25mm**
apart, so the sampled profile was four to five times coarser than its own
sampling interval. What that produces is LATCH AND JUMP: one vertex answers
several consecutive rings, the query then switches, and the whole difference
appears as a single step. Stage 3, same written STL, both ways:

| | ring heights, mm |
|---|---|
| nearest vertex, as shipped | −0.2266, −0.2307, **−0.4292, −0.4292**, −0.3741, **−0.4292**, −0.4406 |
| ray cast off the outer surface | 1.8251, 0.7563, −0.3238, −0.8028, −1.5329, −1.8032, −2.0234 |

One value repeated three times and **not monotonic**, against a strictly
decreasing profile spreading 3.85mm of fall across six steps. Share 0.9274
against **0.2807**. Every tooth in every stage reads 0.2467 to 0.2816 on the
real surface, which is a stable emergence profile, not a cliff.

**THE OLD MEASUREMENT WAS WRONG IN BOTH DIRECTIONS, and the second direction
is the one nobody looked for.** Four controls, each a solid of revolution
whose profile is known before the measurement runs:

| control | old method | corrected |
|---|---|---|
| **a cylindrical collar** - the exact artefact this check exists to catch | share **0.0000**, ledge **False** | share 1.0000, ledge **True** |
| a smooth cone, 13 radial rings | 0.2222, blend | 0.1667, blend |
| **the SAME cone, one triangle strip** | share **1.0000**, ledge **True** | 0.1667, **blend** |
| a 3.5mm cone | 0.5000, blend | 0.1667, blend |

So it **missed a textbook collar entirely** - every probe's nearest vertex sat
on the flat plate, giving a perfectly flat profile with no fall at all - and
it **rejected a provably linear ramp** when the band was spanned by single
large triangles. The gate that had never refused anything but this one case
had, in fact, never been able to catch the thing it was written for.

**The denominator was wrong too.** `total` was `|h[0] − h[-1]|`, which fails
in both directions: a profile that dips and recovers reports a share above 1.0
while being smooth, and a pure spike returning to where it started has an
endpoint difference of ZERO and is waved through. It is now the path length,
`sum |diff|`. On the failing case the denominator **alone** flips the verdict -
0.9276 against 0.6123 on identical numbers.

**THE THRESHOLD IS UNCHANGED AT 0.75.** The fix is the measurement. A
measurement that cannot be taken returns `looks_like_a_ledge: None`, so the
gate refuses exactly as it does for a real ledge, and there is deliberately no
fall back to the nearest-vertex profile: a method known to be wrong is worse
than no method, because it answers.

### 24.2 `local_thickness` encoded a ray miss as a measurement of zero

Found while driving the real scan. A ray that hits nothing returned **0.0** -
"the cast is 0mm thick here" - when what happened is "there is no cast under
this point at all". The caller takes the MINIMUM over the rim, so **one** rim
point over an interproximal embrasure, over a neighbour's open socket or past
the arch's inner edge dragged the whole tooth's thickness to zero. Measured on
the real scan: every tooth reported `local_cast_thickness_mm 0.0` on a cast at
least 3mm thick.

The caller already filtered with `np.isfinite`, so it was written against the
contract the callee did not honour - and 0.0 is finite, so the filter never
fired. A miss is now NaN, with `rim_points_with_no_cast_below` reported
separately, because a thin wall and a missing one are different facts. Same
rule as `test_a_genuine_ray_miss_cannot_become_a_success` already pins for
`drop_to_surface`.

### 24.3 The real scan's segmentation was never incoherent

Reported as a production blocker: all twelve labelled teeth flood a quarter of
the arch, best IoU 0.01-0.05, FDI 31 spans 33.8 x 34.4 x 15.3mm. **Every one
of those numbers is correct and the conclusion drawn from them was wrong.**

An STL has no vertex list - it is triangle soup - so every reader invents an
order when it welds, and this project has four:

| loader | order | vertices |
|---|---|---|
| `trimesh.load_mesh(obj, process=False)` | preserved, **0 rows differ** | 94,848 |
| `open3d.io.read_triangle_mesh(obj)` | NOT preserved | 94,848 |
| open3d + `remove_duplicated_vertices` | first occurrence | 94,848 |
| `stl_io.parse_stl_bytes` / `condition_mesh` | `np.unique` - LEXICOGRAPHIC | 94,848 |

All four agree on the count, so every length check and every assertion passed
while three of the four correspondences were wrong. Measured on the cached
labels, per-tooth bounding-box diagonal, median over the twelve teeth:

| | median box | plausible | connected |
|---|---|---|---|
| read by index against the app's array | **50.71 mm** | 0/12 | 0/12 |
| transferred by POSITION from the right source | **13.97 mm** | 9/12 | 7/12 |

The transfer matched **all 94,848 vertices at gap 0.0** with the permutation
differing at **every single index**. Before it, each tooth fell into 83 to 435
disconnected pieces with its largest component between 2.8% and 20.6% of the
label. After it: **seven of twelve are a single connected region**, and
**eleven of twelve have their largest piece at 98.7% or better** - the
remainder being a handful of stray triangles. Only FDI 37 is genuinely split,
at 0.518.

**`real_scan_regression.py` had a position remap already, and it fixed the
wrong hop** - it moved labels from `stl_io.parse_stl_bytes` to `condition_mesh`,
two arrays already in the same order, and preserved the error exactly. The
labels never belonged to either.

**NO METRIC COMPUTED ON THE LABELS ALONE CAN SEE THIS, IoU INCLUDED**, because
a mis-indexed array scores against itself perfectly. It takes a GEOMETRIC
check, which is what `segmentation_diagnostics.label_report` is: per label,
vertex and face count, bounding box, box diagonal, centroid and disconnected
component count, plus an `indexed_to_this_mesh` verdict from the median. One
implausible tooth is a model problem; all of them is a mapping problem, and
the median is what separates the two.

**What the corrected labels leave is a genuine and much smaller model
question.** Three of twelve are still wrong: FDI 37 is two teeth merged into
one label (two components, largest 0.518), and 36 and 47 are oversized at
26.9mm and 27.5mm. That is `segmentation_fallback`'s job, and it can only do
it now that the other nine do not also look broken.

**The live path was correct already, by the good fortune of one keyword
argument.** `run_segmentation` writes an OBJ in our order, and the inference
pipeline reads it with `read_txt_obj_ls(use_tri_mesh=True)` -
`trimesh.load_mesh(process=False)` - which preserves it. The vendored source
carries the comment *"In some cases, trimesh can change vertex order"*
directly above that function. Nothing checked it. `/segment` now transfers by
position regardless and records the transfer, so a loader change shows up as a
number rather than as scattered teeth months later.

### 24.4 The provider abstraction, and an audit that names its own gaps

`segmentation_providers.py`: `ToothGroupNetworkProvider` (primary, wired to the
weights) and `MeshSegNetProvider` (**candidate, not installed, not
benchmarked**). The candidate's `segment()` **raises**, on the CLAUDE.md s.16
precedent - a stub returning plausible labels would drive a cut, a C_res and a
printed aligner, and output from a function of that name is reasonably
believed. Its audit records that MeshSegNet classifies **faces**, not vertices,
from a 15-channel per-cell feature vector, and that its published pipeline
mean-centres the mesh, which collides with rule 3.1 and would have to be
inverted exactly rather than approximately. Licence, dependencies, Windows CPU
support, inference time, axis and label conventions are recorded as
**NOT_VERIFIED_HERE**: they need the upstream repository, which this
environment does not have. **No benchmark of TGN against MeshSegNet has been
run, and none is quoted.** TCATSeg was not installed, as instructed.

### 24.5 One colour table, and seventeen teeth used to share

`PALETTE[i % PALETTE.length]` over fifteen colours and 32 teeth meant FDI 11
and 31 were both `#e6194b`, 12 and 32 both `#3cb44b`, and so on for every pair
sixteen apart - an upper right central incisor and a lower left central
incisor rendered identically. The colour also depended on the tooth's **index
in an array literal**, not on its FDI number.

`frontend/src/fdiPalette.js` makes the colour a **pure function of the FDI
number**: same tooth, same colour, across a refresh, a re-segmentation, a
change of provider, and any ordering anything returns teeth in. The 32 teeth
take the 32 evenly spaced hues; consecutive teeth take slots 11 apart (coprime
with 32, so every slot is hit once) which puts 123.75 degrees between
neighbours; and because 11 x 3 = 1 (mod 32), the two teeth landing on ADJACENT
hues always differ by 3 in the index and therefore always land in different
lightness buckets - the one place hue separation is weakest is exactly where
lightness separation is guaranteed.

**THE FIRST VERSION FAILED ITS OWN CHECK**, which is the argument for having
one. Lightness around 0.46-0.77 at saturation near 0.6 produced a band of pale
colours: worst pair dE 11.83, and **FDI 14 landed dE 4.09 from the GINGIVA**.
The constants shipped are the maximum of the worst-case separation over a
search of 15 hue strides x 15 lightness cycles x 9 saturation cycles.
`verify-palette.mjs` measures, and fails below a floor:

| | dE |
|---|---|
| minimum over all 496 pairs | **15.98** |
| minimum between neighbours and mirrored pairs | 50.76 |
| minimum from any tooth to the gingiva | 52.55 |
| minimum from any tooth to the viewport background | 62.72 |

The background row is there because two colours can be far apart from each
other and both invisible. Selection is a **treatment laid over** the tooth's
own colour, never a replacement, so a selected tooth is still identifiable as
that tooth. An unknown id returns grey, never a confident tooth colour.

### 24.6 The evidence report, and three export formats behind one gate

`mfg.manufacturing_evidence_report` restates the aggregate gate as the nine
claims a person actually asks about an exported file - the tooth is at the
target matrix as a rigid body; the cast outside the interface is unchanged;
the old socket is restored; a new local interface exists; there is no
artificial root column; no synthetic seat or ramp is exposed; there is no
collar or ledge at the margin; the model is one physical body; the written STL
survives a round trip. **Each row names the gates that are its evidence**, so
a claim can never be made by a row that measured nothing, and a claim with
missing evidence reads false rather than being omitted. An empty stage fails
all nine, asserted.

`/export/final` gains `fmt`: `zip` (default, unchanged), `stl` (bare, for a
slicer) and `manifest` (the record, no geometry). **A format is not a way
around the gate** - all three are the same already-refused-or-validated bytes,
carry the same `X-Print-Ready`, and a test asserts the raw STL and the STL
inside the ZIP are byte-identical.

### 24.7 Where the real scan now stands

**REAL-SCAN MANUFACTURING REGRESSION = STILL NOT VERIFIED END TO END**, and it
is much closer and far better measured than it was.

What now works that did not: the labels are coherent, and **seven real teeth
cut successfully** from their own label components - FDI 32, 34, 41, 42, 44,
45 and 46, three of them producing a proper socket cup rather than the flat
fallback. One of the seven built a complete local interface (`ok=True`).

What still refuses, by name and with numbers, on a cast where all seven
sockets were filled: `interface_unbuildable_wall_too_thin` (the lower ring
cannot seat), `outside_reconstruction_envelope` (rim separation 4.2-4.7mm),
and `interface_construction_failed` (31 of 166 collar points unresolved).

Driven alone, FDI 45 reaches staging and refuses `wall_too_thin` with
`rim_separation_max_mm 2.9085` and `local_cast_thickness_mm` **0.0155**. That
last number is the one the 24.2 fix made honest - it read a flat 0.0 before,
which was a ray miss wearing a measurement's clothes. 0.0155mm is a real
knife-edge somewhere under that rim, almost certainly where the trim boundary
or an embrasure passes beneath it, and `wall_limit` is `safe_wall_fraction`
times it, so the refusal is arithmetically correct.

**WHAT THAT EXPOSES IS A STATISTIC, NOT A THRESHOLD, and it is left alone
deliberately.** `local_thickness` is reduced with `np.nanmin` over the whole
rim, so ONE knife-edge point governs the wall limit for the entire tooth.
A low percentile, or a per-point limit, is very likely the right reduction -
but changing it would be tuning a gate until a case passes, which is the one
thing this work does not do. It needs its own evidence: what the thickness
profile around a real rim actually looks like, and what a wall limit should
mean when part of the rim overhangs nothing at all. That is the next piece of
work, and it is a geometry question rather than a parameter.

These are real-anatomy difficulties on a genuine intraoral scan, not the
mapping bug. **Nothing in sections 21, 23 or 24 that concerns manufacturing
gates is based on the real scan.**

### 24.8 The frontend, and the dependency audit

**WHAT SHIPPED.** `theme.js` holds the design tokens - an 8px spacing scale,
one accent, the surface and state colours, and motion durations that all pass
through `duration()` so one media query stills the interface.
`installGlobalStyles` writes the focus, hover, pressed, disabled and
`prefers-reduced-motion` rules **once**, as a stylesheet, rather than at the
sixty-odd inline style sites that would each have to remember them.
`:focus-visible`, not `:focus`, so a mouse click leaves no ring while keyboard
navigation still shows one.

`ToothLegend.jsx` is the first component extracted from `App.jsx`. It turns a
colour back into an FDI number, which is what makes 32 colours useful, and it
makes the palette **inspectable** - a duplicate colour is obvious in a legend
and invisible in a viewport, which is how seventeen teeth came to share one. It
carries the per-tooth review badges from `segmentation_diagnostics` with their
REASONS, and it says in words when the labels are not indexed to this mesh,
because no accuracy metric can show that.

**IT TAKES A SUMMARY, NOT THE LABEL ARRAY.** The per-vertex labels are 94,848
integers on a real scan and stay in a ref; the legend receives at most 32
counts. Passing the array would drag it into a dependency list and into every
reconciliation, which is the rule that keeps three.js transforms and stage
scrubbing off the render path.

**The deps-array trap was checked, not assumed, for the sixth time.**
`loadSegDiagnostics` had to enter two dependency arrays. It is declared at
`App.jsx:378` and both consumers are at 1255 and 1461, so there is no temporal
dead zone and the fix is safe - §13's rule is to verify the declaration order
before taking that lint fix, and this is what verifying it looks like.

**DEPENDENCY AUDIT.** `three-mesh-bvh` is load-bearing (`bvh.js`, 555x on
raycasts, §15). `@radix-ui/react-accordion` is used by `Panel.jsx`,
`@radix-ui/react-slider` and `lucide-react` by `StagingTimeline.jsx`.
**`three-bvh-csg` was imported nowhere** - it appeared only in `package.json`
and the lockfile - and has been removed. Every boolean in this app is backend
`manifold3d`, so its presence also implied browser-side CSG that does not
exist. The bundle is unchanged by the removal because an unimported package is
already tree-shaken; what it removes is install weight and a false signal. No
UI framework was added and framer-motion was not brought back.

### 24.9 What was NOT done

* **The workspace was not redesigned.** The top bar, left workflow rail,
  right context panel and bottom staging dock are not built, `App.jsx` is
  still one ~2,700-line file, and only `ToothLegend` has been extracted from
  it. The existing `S` style object has NOT been migrated onto `theme.js`;
  the tokens are the base for that work, not the work.
* **No browser performance measurements were taken this pass** - initial
  load, segmentation responsiveness, 3D interaction, stage scrubbing, shadow
  toggle and attachment placement are all unmeasured here. There is no
  browser in this environment.
* **MeshSegNet was not benchmarked** against ToothGroupNetwork. See 24.4.
* **Real-scan manufacturing is still NOT VERIFIED end to end.** See 24.7.
* **The viewport darkening is still mitigated, not root-caused** (section 22).

Covered by `test_segmentation_mapping.py` (12), the four transition controls
and the unmeasurable-transition test in `test_manufacturing_interface.py`, the
evidence and export-format tests in `test_manufacturing_matrix.py`,
`frontend/verify-palette.mjs` and `real_scan_regression.py`.

## 25. RESOLVED: CROSSTOOTH AS A SECOND SEGMENTATION PROVIDER

CrossTooth (CVPR 2025, *3D Dental Model Segmentation with Geometrical
Boundary Preserving*) is installed, wired to its shipped weights, and
selectable per request. ToothGroupNetwork is untouched and remains the
default. **No accuracy claim is made for either model, and §25.6 is about
why that is not a hedge.**

### 25.1 The repository could not import its own model, on any platform

`models/PTv1/point_transformer_seg.py` opens with

```python
from models.PointTransformer.libs.pointops.functions import pointops
```

and **`CrossTooth/models/` contains only `PTv1/`.** That path does not exist
in the shipped checkout, so the file cannot import on Linux with CUDA either;
the missing piece is the Point Transformer CUDA extension, which the README
tells you to install from a different repository. Three more blockers sat
behind it, each read out of the vendored source rather than guessed:

| blocker | what was done |
|---|---|
| the dotted path above does not exist | registered in `sys.modules` against this project's `pointops_cpu.py`, **for the duration of the import only** |
| CrossTooth's `queryandgroup` returns `(grouped, idx)`; TGN's returns `grouped` | `_CrossToothPointops` adapts the return value. `pointops_cpu.py` is **not** edited — TGN depends on its single-value contract |
| `TransitionDown.forward` calls `torch.cuda.IntTensor(n_o)` on a list of ints | patched to a CPU int32 constructor across one inference call, then restored |
| `dataset/data.py` loads meshes with `vedo`, which is not installed | not needed — this app holds the mesh already; the feature preparation is replicated from that file |

**THE SHIM IS SCOPED TO THE IMPORT, AND THAT IS THE LOAD-BEARING PART.**
`models` is a top-level name **ToothGroupNetwork also claims** — tgn_bridge's
design rule 3 exists for it — and this backend warms TGN on a background
thread. A synthetic `models` package left installed for the life of the
process is exactly how one model silently captures the other's imports. A
chain name that is **already** in `sys.modules` is left alone and never
restored, because it is not ours; measured, `sys.modules` comes back
identical, and a test asserts it with a foreign `models` planted in the way.

`torch.cuda.IntTensor` is process-global while patched, so it is held only
across one inference call, and a test asserts it is restored **even when the
body raises**. Nothing else in this repository constructs a CUDA tensor.

### 25.2 The shipped loader shows the model a quarter of one arch

`dataset/data.py` does:

```python
permute = np.random.permutation(self.args.num_points)   # 0..15999
pointcloud = pointcloud[permute]
```

On a mesh with **more** faces than `num_points` that index array cannot reach
past 15,999, so the model is handed the **first 16,000 triangles in file
order**, shuffled. This project's real scan has 187,625. The researchers
avoid it by decimating offline with a curvature-aware
`selective_downsample.exe` — a Windows binary that is **not in the
repository**.

`spatially_uniform_subset` replaces it with a voxel grid bisected onto the
target count, one representative per voxel, deterministic and independent of
input order. **THIS IS NOT THE RESEARCHERS' DECIMATION AND THE DIFFERENCE IS
REAL**: theirs puts more cells near tooth boundaries, which is where a
boundary-preserving model most wants them. Uniform is the honest substitute,
not an equivalent one, and the audit says so.

FPS was rejected on a measurement, not on taste: `pointops_cpu`'s
`furthestsampling` is a Python loop, so 16,000 passes over 187,625 points is
minutes of overhead for a *selection*. Random was rejected because at 8.5%
coverage it leaves clumps and bald patches.

**The voxel encoding was 10.5 s of a 24 s run.** `np.unique(keys, axis=0)`
lexsorts three columns; encoding each voxel as one int64 (the strides are the
per-axis voxel counts, so it cannot collide) took **prepare from 10.5 s to
1.79 s and the whole segmentation from 24.2 s to 12.9 s**, with
**bit-identical labels**.

### 25.3 The order of pad → permute → normalise is preserved deliberately

`PointcloudNormalize(radius=1)` touches **columns 0:3 only** — normalising all
six would rescale the unit face normals by the arch's radius in millimetres.
And the zero-padding rows go in **before** normalisation, so on a sparse mesh
they pull the centroid toward the origin. That is part of the input
distribution the checkpoint was trained on; "tidying" it by normalising first
would feed the model something it has never seen.

**Rule 3.1 is not at risk.** The normalisation is an input transform on a
copy, no scan coordinate is written back, and the centroid and radius are
returned so the transform is inspectable rather than implicit.

**A padding row carries `source_face = -1`, and that is not cosmetic.** Index
`-1` in NumPy selects the LAST element, so a padding row's prediction would
land silently on the final triangle of the mesh.

### 25.4 It classifies FACES, and the mapping home is the researchers' own

The 6 channels are the face centroid and the unit face normal, so a
prediction is one class per triangle. `prepare_data/upsample_points.py` maps
them back with `KNeighborsClassifier(n_neighbors=3)` fitted on the predicted
cell centroids in original millimetre coordinates, and that is what is used
here rather than a majority-of-incident-faces vote — it is theirs, and it is
defined for a vertex whose every incident face was dropped by the subset.

**SO THIS PROVIDER HAS NO VERTEX-ORDER DEPENDENCY AT ALL**, which is worth
stating because it is the defect §24.3 exists around. Nothing is indexed by
position in an array; the mapping is a coordinate query. `transfer_labels_by_
position` is therefore **not run** for it: it would compare the caller's array
with itself and report the identity it was handed, which is a check that
cannot fail and so is worth nothing. The transfer record says exactly that,
and a test shuffles the vertex array and asserts the labels move with it.

### 25.5 The quadrant naming, which the source cannot settle and a measurement can

`CrossTooth/utils.py` gives classes 1-8 the "L" quadrant and 9-16 the "R"
quadrant. **Which physical side of the patient that is depends on the
orientation convention of the training data**, and this repository has no
annotated ground truth (§17). Two things are checkable without one, and
`quadrant_coherence` measures both: are the two groups separated by a plane,
and does the class index run anterior to posterior.

**THE REFERENCE POINT FOR THE SECOND ONE WAS WRONG FIRST, and it would have
condemned a correct arch.** An arch is a horseshoe — incisors and molars both
sit on its PERIPHERY — so distance from the arch centre is not monotone in
the tooth index. Distance from the quadrant's own central incisor runs along
the curve and is. Measured on the real scan, the same labels:

| reference | 31-38 | 41-48 |
|---|---|---|
| arch centre | 1.000 | **0.929** |
| the quadrant's own central incisor | 1.000 | **1.000** |

**The naming itself was settled by measurement against the other model.**
Both were run over `case_lower.stl` and compared on the 56,707 vertices both
call a tooth:

| | agreement |
|---|---|
| exact FDI, as mapped | **0.4884** |
| exact FDI if CrossTooth's quadrants were mirrored | **0.0559** |

So the two independently trained models **agree on which side is 3x and which
is 4x**. That is the one thing inter-model agreement can settle, and it is
not accuracy: `labels_are_fdi_verified` is False, always, and a test asserts
it can never read True.

### 25.6 The benchmark reports geometry and agreement, and refuses to report accuracy

`benchmark_providers.py` runs every installed provider over one scan.
**It is not an accuracy benchmark and must never be quoted as one** —
`benchmark_segmentation.py` already implements IoU, FDI accuracy, Chamfer and
Hausdorff correctly and then **refuses to run** without an independently
attributed annotation, for the reason §17 records. Scoring model A against
model B is that same error wearing a second hat: it says which two models
agree, not which one is right.

What it does report is **geometry**, which needs no annotation — a tooth is
one connected lump of a plausible size — and **agreement**, which is symmetric
and attributes nothing but can settle a convention. Measured,
`case_lower.stl`, 94,848 vertices / 187,625 faces, CPU:

| | ToothGroupNetwork | CrossTooth |
|---|---|---|
| seconds | 255.04 | **13.30** |
| teeth labelled | 11 | 16 |
| median per-tooth box diagonal | 14.738 mm | 13.239 mm |
| **max** per-tooth box diagonal | **51.007 mm** | 17.805 mm |
| teeth in ONE connected piece | 4 of 11 | **12 of 16** |
| worst largest-component fraction | **0.5230** | 0.9879 |

CrossTooth's own breakdown: prepare 1.8 s, inference 9.0 s, upsample 2.1 s;
mean per-cell softmax confidence 0.975; quadrant groups separated by 34.53 mm
with **zero overlap**.

**THE 51.007 mm ROW IS THE ONE TO READ, and §24.3 is why it is not a mapping
bug this time.** One TGN label spans half a mandible while the MEDIAN stays at
a plausible 14.738 mm — a mis-indexed array scatters EVERY tooth, so the
median is what separates the two failures, and here it says the labels do
belong to this mesh and one region is genuinely wrong.

**What this does NOT establish.** That CrossTooth's FDI numbers are correct.
The two models' exact-FDI agreement is 0.4884, and the confusion is a
one-tooth shift along the 3x quadrant: TGN's own label set is missing 33 and
35 and carries a two-component 37, which is *consistent with* TGN
under-segmenting there and CrossTooth being right — and consistency is not
proof. Either model could be the shifted one. **Nothing here licenses
switching the default**, and it has not been switched.

### 25.7 The toggle

`POST /api/session/{sid}/segment?provider=crosstooth`. Per request, not a
server mode, because the point of two models is running both over ONE scan —
a server-wide switch makes that a restart apiece. `ALIGNER_SEGMENTATION_
PROVIDER` sets the process default, and **an unrecognised value is ignored
rather than obeyed**: a typo in an environment variable must not silently
change which model segments a patient's arch.

**EVERY PROVIDER LANDS IN THE SAME DOWNSTREAM PATH** — the FDI/jaw check, the
hybrid geodesic fallback, the spatial diagnostic, the audit entry, the
telemetry span and the response shape are shared. A second model does not get
a softer gate than the first.

Three smaller things this exposed:

* **The gate skips ToothGroupNetwork deliberately.** Its `available()`
  reported `loaded`, which is False for the first seconds of every process
  while the background warm-up imports torch (§12) — and the shipped
  behaviour for a request landing in that window is to WAIT on the load lock,
  not to be refused. Gating on it would have turned a 10-second wait into a
  409 in the one path with a clinician in front of it. `available()` now
  reports whether the model CAN run rather than whether it HAS run.
* **`ToothGroupNetworkProvider.segment` now loads its own pipeline.** It
  relied on the endpoint having warmed it, and a provider that only works
  when someone else warmed it is not a provider — `benchmark_providers.py`
  drives it directly.
* **`diag["provider"]` recorded `DEFAULT_PROVIDER`.** Those were the same
  value for as long as there was one provider; it would now mislabel every
  A/B run. It records the backend that actually ran.

The client's picker is a plain `<select>` — no framework, no new dependency,
**+0.31 KB gzip** (272.44 → 272.75 KB). It reads the registry once and hides
itself if that fetch fails, because a model chooser is a convenience and must
never be able to stop the shipped model running. It reports the provider from
the RESPONSE, not the one that was requested: they are the same today, and
reading the request back would make any future fallback invisible in exactly
the place someone would look to find out what produced these teeth.

**The deps-array trap was checked, not assumed, for the SEVENTH time.**
`segProvider` enters `runSegmentation`'s dependency array at `App.jsx:1496`
and is declared at `:381`.

### 25.8 einops is the only package added, and the three that were not

`einops` is genuinely load-bearing — `PointTransformerLayer.forward` uses
`einops.reduce` and `einops.rearrange`, and that file is vendored code this
project does not edit. Verified by following the import chain rather than by
reading `CrossTooth/requirements.txt`, which §19 records getting wrong the
other way round for trimesh and open3d: `point_transformer_seg.py` imports
exactly `os`, `torch`, `torch.nn`, `einops` and `pointops`.

Not installed, each for a reason: **pointops** (shimmed, §25.1); **vedo**
(only `dataset/data.py` uses it, to load a mesh we already hold — a VTK stack
to re-read a mesh in memory); **spconv, flash-attn, torch-scatter,
torch-sparse, torch-cluster, torch-points3d, pykeops** (none on the PTv1
inference path; they belong to the training code and to the 12 competing
methods in `compete/`).

`*.pth`, `*.pt` and `*.ckpt` join `*.h5` in `.gitignore`. `build_ai_export.py`
already refused all four on the same ground: a checkpoint is not source, and a
27MB binary in history is paid for on every clone forever. `CrossTooth/` joins
`ToothGroupNetwork/` in `check_structure.py`'s skip list — its `compete/`
folder carries 12 more research checkouts, three of which have undefined names
that are not ours to fix. **`crosstooth_bridge.py` is ours and stays in the
walk** (99/99).

### 25.9 What was NOT done

* **No accuracy benchmark exists**, for either model, and none can be run
  here. See §25.6.
* **The default was not changed.** CrossTooth wins every geometric measure
  taken and is 19× faster, and that is still not grounds to make it the model
  that decides where a clinician cuts.
* **Only the lower jaw was exercised on real anatomy.** There is no maxillary
  scan in this repository (§19), so the upper class→FDI branch is covered by
  a synthetic fixture and by the table test, not by a real arch.
* **The edge-segmentation head is discarded.** `point_seg_result,
  edge_seg_result = model(...)` — the second output is the boundary head the
  paper's title is about, and `predict.py` ignores it too. It is where a
  cervical-margin refinement would come from, and it is unexplored.
* **`num_points` is fixed at the trained 16,000.** `nsample` is a fixed k, so
  changing the point count changes the neighbourhood scale it represents. It
  is a parameter with a warning attached rather than a knob.
* **No multi-subset voting.** One voxel subset is drawn per scan; aggregating
  votes over several would cover the faces the subset misses, at a multiple
  of the runtime. Unmeasured.

Covered by `test_crosstooth_adapter.py` (32), `benchmark_providers.py` and
`crosstooth_bridge.py`.

## 26. RESOLVED: THE NEIGHBOURHOOD SEARCH, THE TRIM, AND THE CLICK

Five defects, and four of them reported themselves as something else: a model
that segments badly, a cast that is too thin, a knife-edge rim, and a tooth
boundary that was never found. The fifth never reported at all.

### 26.1 `pointops_cpu` returned the wrong neighbours, in two independent ways

Both networks in this repository are Point Transformers, and every layer of
both is a k-nearest-neighbour gather. The CPU shim they run on here was wrong
twice, and neither fault raises: **a neighbourhood search that returns the
wrong neighbours does not crash, it produces a model that segments badly,
which is indistinguishable from a model that is not very good.**

**1. `torch.cdist` in float32 cancels catastrophically at scanner
coordinates.** It computes the matrix-multiply expansion
`||a||² + ||b||² − 2a·b`, and rule 3.1 means this repository NEVER re-centres
a scan — the coordinates are wherever the scanner put them. Measured on
`case_lower.stl`, 2,000 queries into 16,000 points, k=16, against an exact
float64 reference:

| offset from origin | f32, mm | f32, `donot_use_mm` | f64, mm |
|---|---|---|---|
| 0.0 mm | **20 wrong rows** | 0 | 0 |
| 37.1 mm (this scan) | **94** | 0 | — |
| 137.5 mm (what `test_api_core` pins) | **551** | 1 | 1 |
| 500.0 mm | **1917** | 2 | 2 |
| ms per 1000×16000 block | 53.8 | 137.9 | **104.2** |

**FLOAT64 IS BOTH THE MOST ACCURATE AND THE FASTER OF THE TWO CORRECT
OPTIONS**, because it keeps the BLAS matmul path while `donot_use_mm_for_
euclid_dist` materialises an (m, n, 3) difference tensor. The first choice
made here was `donot_use_mm`, which is correct and made a TGN segmentation
886.81 s; float64 brought it to **369.87 s**. `_KNN_BLOCK_ELEMS` halved to
8,000,000 so a float64 distance matrix costs the same peak bytes as the
float32 one did, not twice as many.

**2. `interpolation` weighted by inverse SQUARED distance.** The upstream
wrapper returns `sqrt(dist2)` and weights by `1/(d + 1e-8)`; this module kept
the squared value, declared that it had with `KNN_RETURNS_SQUARED_DISTANCE`,
and then **ignored its own flag**. That is a different interpolation, and
`TransitionUp` calls it on four of five decoder stages in both networks. k=1
was unaffected either way — a single weight normalises to 1.0 — which is why
`heads.py:50` never showed it.

**It was confined to this one function, and that was PROVEN rather than
assumed.** `verify_pointops.py` AST-walks `pointops_cpu.py`,
`crosstooth_bridge.py`, `ToothGroupNetwork/models`, `CrossTooth/models` and
`CrossTooth/compete` for every `a, b = knnquery(...)` and classifies each as
discards / converts / **READS IT RAW**: 7 call sites, all safe. The rest of
the file is checked against independent NumPy references written from the
vendored upstream wrapper.

**The correction is visible in the output.** ToothGroupNetwork went from 11
teeth to 12 on this scan, and from 4 of 11 in one connected piece to 7 of 12.

### 26.2 The arch trim deleted the cast out from under a molar's rim

`interface_unbuildable_wall_too_thin` on FDI 46, over a cast measuring
**2.8573 mm** thick. The refusal was arithmetically correct and named the
wrong thing: the cast was not thin, it was **absent**.

`build_cast_base` keeps the horseshoe band within `ARCH_TRIM_MARGIN_MM` of the
fitted ridge. §10 chose 7.0 from whole-arch measurements that are still right.
What they do not describe is a wide molar, whose cervical rim reaches further
from the ridge than the band does — FDI 46's rim reaches **9.385 mm**.
Measured, same scan, same tooth, same everything else:

| margin | rim points outside the cast | max outside | interface |
|---|---|---|---|
| **7.0** | **53 of 280** | **1.2961 mm** | REFUSED, 40 points |
| 9.0 | 0 | 0.0000 mm | **ok** |
| 12.0 | 0 | 0.0000 mm | ok |
| 15.0 | 0 | 0.0000 mm | ok |

and `local_cast_thickness_mm` reads 2.8573 at **every** one of those margins,
so thickness was never the variable.

**THE MARGIN IS NOT RAISED GLOBALLY.** That would undo a measured decision to
make one case pass and loosen the trim for every case that never needed it.
`build_stage_bundle` computes a FLOOR from the rims that actually exist —
`max(distance_to_arch_curve(rim)) + seat_bottom_outset_mm +
TRIM_RIM_HEADROOM_MM` — raises the margin only if the request is below it, and
records `margin_requested_mm`, `margin_used_mm`,
`margin_raised_for_socket_rims` and `socket_rim_max_distance_to_ridge_mm` in
the manifest. Measured: FDI 46 present → 7.0 raised to **11.585 mm**; FDI 45
alone (rims reach 4.106 mm) → **7.0 left exactly as it was**.

`cg.distance_to_arch_curve` exists so the floor is built on the trim's OWN
quantity rather than a reimplementation of it, and a test asserts no face the
helper puts outside the band survives the trim at 4, 7 and 11 mm.

### 26.3 §24.7's open question is answered, and the answer is that the probe was aimed at the wrong object

§24.7 recorded `local_cast_thickness_mm` **0.0155** on FDI 45 and asked
whether reducing the rim's thickness profile with `np.nanmin` is defensible,
calling it "the next piece of work". The follow-up measurement on the raw scan
looked worse still — 196 of 221 rim points with no cast beneath them, minimum
0.1178 mm — and that reading is an **artefact**.

**An intraoral scan is an OPEN SHELL.** A ray dropped from a cervical rim
point runs down the OUTSIDE of the gingival wall and hits nothing, so the
misses measure the scan's topology, not the cast's thickness. Against the
solid the manufacturing path actually builds:

| | rim points | no cast below | min | p10 | median |
|---|---|---|---|---|---|
| FDI 45, probed on the raw scan | 221 | **196** | 0.1178 | 0.1717 | 0.2990 |
| FDI 45, probed on the cast base | 221 | **0** | **25.6760** | 26.0392 | 27.5319 |
| FDI 46, probed on the raw scan | 280 | **226** | 0.2446 | 1.4435 | 4.0355 |
| FDI 46, probed on the cast base | 280 | **0** | **2.8573** | 5.9982 | 26.3189 |

**SO `np.nanmin` IS NOT THE PROBLEM AND THE REDUCTION IS LEFT ALONE.** There
is no knife-edge under those rims. §24.7's question was well posed against the
number it had; the number was measuring a different object. The `local_
thickness` fix in §24.2 — a ray miss returns NaN rather than 0.0 — remains
correct and is what made the artefact legible instead of silent.

### 26.4 Clicking a tooth selects that tooth

The workflow the product is for is *click a tooth → the whole tooth is
selected*. Until now `onPointerDown` went straight to `/wand`, so the geodesic
flood was the only route, and §19 has recorded since Phase 5 that it does not
isolate a tooth on this scan. Measured over all sixteen teeth, clicking each
tooth's own surface:

| route | of the selected vertices, how many are the clicked tooth |
|---|---|
| segmentation label (`POST /select-tooth`) | **100.0%** |
| geodesic wand | 15.7%, taking 25.3% of the arch per click |

`/select-tooth` reads the label under the click and returns that label's own
**largest connected component** — the same rule tier 2 of `segmentation_
fallback` already uses, because a label carries a handful of triangles on its
neighbour and a crown built from two disconnected pieces cannot close
watertight. It refuses 409 on an unsegmented arch and points at the wand, 400
on an out-of-range vertex id (**a negative id must not select from the end of
the array** — §14 records that exact defect in `/cut`), 500 on a label array
of the wrong length, and returns `route: "gingiva"` with `fdi: null` on
tissue. **The brush and the wand are untouched**; they are the CORRECTION
tools now, not the selection tool.

### 26.5 The wand's auto tolerance was a search ceiling wearing a measurement's name

`cut_guard.auto_tolerance` scans 1.0 to 25.0 mm for a plateau in the flood's
growth — the sulcus barrier — and when there is none it returns its own
CEILING with `plateau_found: False`. **`/wand` computed that flag and dropped
it**, returning `tolerance: 25.0` in the same field a found plateau uses. On
this scan the ceiling comes back for all sixteen teeth.

The flood is NOT changed — the wand needs no segmentation at all and refusing
here would break the one route that works without it. What changed is that the
response carries `plateau_found`, `tolerance_is_a_ceiling_not_a_measurement`
and `selected_fraction_of_arch`, and the client says so in words instead of
reporting an auto-tolerance.

### 26.6 A contacting pair refuses, and it now refuses under its own name

FDI 45 builds `ok=True` driven alone. In an export that also moves FDI 46 it
refused `interface_unbuildable_wall_too_thin` — over a cast **25.926 mm**
thick with a wall limit of 12.963 mm, which sends the reader to look for a thin
cast that is not there.

What bound it is the INTERDENTAL CLAMP. The two teeth are in contact: their
transformed cervical rims come within **0.1126 mm**, so
`max_bridge_removal_fraction` leaves the collar's foot **0.25 mm** of outward
reach instead of 1.20 mm, it is seeded on the steep socket wall rather than out
on the ridge, and **5 of 221 points finish 0.0139 mm short** of the 0.05 mm
they must be buried by. Same for 46 against 45: 5 of 280, short by 0.0427 mm.

**THE GATE IS UNCHANGED — the same cases refuse, to the micron.** Only the
name and the evidence changed: `interface_unbuildable_interdental_bridge_too_
narrow`, with `collar_bottom_unresolved_points_outset_clamped` and
`refusal_bound_by`. The real fix for a contacting pair is ONE shared
reconstruction across the pair, which the original plan contemplated and which
does not exist. **It is recorded as missing rather than approximated**, and no
threshold was moved toward it.

> **A gate was found disabled and has been restored.** `manufacturing.py`'s
> bottom-ring check had been edited to `if False:` to get the real scan
> through. A bottom ring not inside real cast material is a collar whose foot
> fuses with nothing — the exposed synthetic wall `aggregate_print_gate`
> exists to refuse. The other edit in the same pass, the window-lift clearance
> re-establishment, is KEPT: it re-runs the identical march the construction
> loop already performs, with the same reach limit, and it is what unblocked
> FDI 45 (204 of 221 lifted points were left inside the crown by the window
> lift, and `interface_construction_failed` was the result).

### 26.7 The forensics built for this had never once run

`edge_forensics` measures the whole serialisation chain and attributes every
offending edge to the geometry that made it. On the real scan it reported
`ran: false, reason: "IndexError: index 168560 is out of bounds for axis 0
with size 168560"`.

`stage_labels` is re-indexed onto the WELDED faces twenty lines after
`_pre_weld_v, sf_pre_weld = sv, sf`, so the call handed it **168,562 pre-weld
faces and a 168,560-entry post-weld label array**. The exception was swallowed
by the try/except around the call, and the manifest said the forensics did not
run — so the one tool that can name the source of an offending edge was
silently absent on exactly the stages that have one. The comment four lines
above the capture already warns about this class of mistake; it happened
anyway, at the call site. `_pre_weld_labels` is captured with the arrays it
belongs to.

**A guard that reports its own failure as "did not run" is a guard nobody will
chase.**

### 26.8 The real scan, end to end

`STL → occlusal plane → CrossTooth segmentation → click-to-select → cut →
C_res movement → local cast reconstruction → staged fused solid → written STL
→ reread` **now completes on `case_lower.stl`.** This is the first time a real
crown has reached a written, re-read manufacturing STL.

FDI 45, 0.25 mm extrusion, 1 stage, on the real mandible:

| | |
|---|---|
| segmentation | 16 teeth, `indexed_to_this_mesh: true`, median box 13.248 mm |
| click-to-select | 3,178 vertices, 6,133 faces, **purity 100.0%** |
| cut | crown 3,179 verts, rim 221, socket `flat_fallback` |
| trim margin | 7.0 requested, 7.0 used (rims reach 4.106 mm) |
| cast base | 162,596 faces, 28,841.727 mm³ |
| interface | **built** — connector 241.0982 mm³, connector∩cast 163.3153 mm³ |
| fused stage | 168,560 triangles, 28,954.8856 mm³, closed, 1 component |
| open edges | **0**, before and after the weld and after the reread |
| **NOT PRINT READY** | 4 gates: `written_stl_topology`, `body_count_agrees_with_stl`, `no_self_touching_boundary`, `old_site_restored` |

Gates that PASSED, and they are not trivial: synthetic exposure 0.0066,
exposed clearance wall 0.0 mm², unaffected-cast fidelity **0.0 mm in both
directions**, reconstruction inside the envelope 0.0, every interface built,
interface continuous around the rim.

**THE REMAINING BLOCKER IS ONE EDGE, AND THE FORENSICS NAMES IT.** Edge
163183, **0.09515 mm** long, four incident faces of 0.0111 / 0.0017 / 0.0047 /
0.0088 mm², whose normals come in two pairs identical to four decimals —
`[-0.8727, 0.0595, 0.4846]` twice and `[-0.9155, 0.0618, 0.3976]` twice. That
is two sheets of surface lying on each other. Provenance:
**`ORIGINAL_CAST` on all four faces.** Zero non-manifold edges before our
export weld, one after, and the reader's weld then merges nothing further —
so the weld is what expresses it, and the geometry is what carries it.

**IT IS NOT INHERITED FROM THE CAST.** Asked directly, with no tooth cut at
all: the trimmed, extruded cast base has **0 coincident position groups, 0
zero-area triangles and 0 non-manifold edges after the export weld, at margins
7.0, 9.0 and 11.585**, and the conditioned scan rounded to float32 has 0 as
well. The self-touch is created by the union, where the collar's wall leaves
the cast 0.38 mm outside the target rim's bounding box.

`collapse_short_nonmanifold_edges` caps at 0.05 mm — scanner resolution is
20–50 µm — and correctly refuses a 0.09515 mm edge. **The cap was not raised.**
This is the §23 grazing-contact class on real anatomy, at one point, and
closing it needs the connector's exit from the cast to cross transversally
there, which is geometry work and not a threshold.

### 26.9 The per-tooth verification table, CrossTooth, `case_lower.stl`

16 teeth over 94,848 vertices, median box diagonal **13.248 mm**,
`indexed_to_this_mesh: true`.

```
 label   verts   faces               bbox (mm)    diag  parts  largest   size  review
    31    1938    3606    5.66 x  6.19 x  6.70   10.73      2   0.9964     ok  REVIEW
    32    2339    4395    6.77 x  6.57 x  7.99   12.36      1      1.0     ok      ok
    33    2949    5689    6.75 x  7.86 x  8.61   13.47      1      1.0     ok      ok
    34    3231    6284    7.13 x  7.31 x  7.53   12.69      1      1.0     ok      ok
    35    3306    6385    8.29 x  8.06 x  6.02   13.03      1      1.0     ok      ok
    36    6264   12197   12.07 x 11.43 x  6.37   17.80      2   0.9999     ok  REVIEW
    37    5253   10158   11.51 x 11.49 x  6.26   17.43      2   0.9999     ok  REVIEW
    38    4001    7698   10.58 x  9.79 x  4.51   15.11      1      1.0     ok      ok
    41    1964    3682    5.82 x  6.29 x  7.39   11.31      1      1.0     ok      ok
    42    2201    4114    6.04 x  6.91 x  7.24   11.69      2   0.9886     ok  REVIEW
    43    2759    5200    7.07 x  8.06 x  8.40   13.62      1      1.0     ok      ok
    44    2773    5333    7.20 x  7.31 x  7.68   12.82      2   0.9998     ok  REVIEW
    45    3186    6135    7.75 x  7.50 x  6.60   12.65      1      1.0     ok      ok
    46    6204   12093   11.04 x 11.39 x  6.64   17.20      1      1.0     ok      ok
    47    5046    9768   11.24 x 10.98 x  6.25   16.91      1      1.0     ok      ok
    48    5653   10962   10.98 x 11.03 x  5.21   16.41      1      1.0     ok      ok
```

**Every tooth is of plausible size** — no label spans two teeth, which is the
failure §24.3's corrected mapping left behind on ToothGroupNetwork. Eleven of
sixteen are a single connected region and the five flagged REVIEW carry
largest-component fractions of **0.9886 to 0.9999**, i.e. a handful of stray
triangles on a neighbour, which is exactly what `/select-tooth`'s
largest-component rule discards. `format_report` now separates the two
verdicts: `size` (BIG = two teeth merged, a model problem) from `review` (in
more than one piece, usually a few triangles), because the old single `<<`
column made them look the same and they lead to different actions.

**NONE OF THIS IS ACCURACY.** No segmentation model in this repository has
been scored against an independent annotation and none can be here (§17).

### 26.10 TGN vs CrossTooth on the corrected shim, same scan

| | ToothGroupNetwork | CrossTooth |
|---|---|---|
| seconds | 369.87 | **19.59** |
| teeth labelled | 12 | **16** |
| median per-tooth box diagonal | 15.072 mm | **13.248 mm** |
| **max** per-tooth box diagonal | **49.355 mm** | 17.805 mm |
| teeth in ONE connected piece | 7 of 12 | **11 of 16** |
| worst largest-component fraction | **0.5218** | 0.9886 |

Vertex agreement on the 56,920 vertices both call a tooth: exact FDI
**0.4853**; if CrossTooth's quadrants were mirrored, **0.0616**. So the two
independently trained models agree on which side is 3x and which is 4x, which
is the one thing inter-model agreement can settle. **It is not accuracy, and
the default was not changed.**

### 26.11 Provider-specific ordering can never become the canonical mesh

The regression §24.3 asked for exists and is three tests. Provider A is handed
the canonical array and returns labels in a SHUFFLED order — the
ToothGroupNetwork hazard, where a loader reorders and every count check still
passes. Provider B never indexes an array at all; it labels geometry and the
labels come home by coordinate — the CrossTooth route. Both must land on the
same canonical labels, and do.

**The fixture is shown to REPRODUCE the defect first** (§5's rule): read by
index, the same labels are wrong at 8 of 16 vertices with a largest-tooth box
of 27.35 mm against 10.39 mm — **while the total count and every per-label
count still agree.** A third test walks `api_core.py`'s AST and asserts no
provider array is ever stored as `verts` or `faces`, which would be the
tidy-looking fix and would silently invalidate every vertex id the client
holds, the extraction mask and the brush index.

### 26.12 The workspace

`WorkflowRail.jsx` and `ToothInspector.jsx`, both pure, both taking summaries,
neither holding state or a ref. The 94,848-integer label array stays where it
was.

**FOUR STATES IN THE RAIL, and the fourth is the point.** `done`, `current`,
`blocked` and `pending` — a blocked step carries its REASON ("cut a crown
first") and a pending one does not pretend to have one, the same distinction
`ValidationPanel` draws between a failing check and one never computed and
§14 draws between `NOT_CHECKED` and `CLEAR`. Seven accordion headings
described the workflow's ORDER and never its STATE: a restored case showed
seven closed panels and the clinician opened each to find out where they were.

The inspector renders a measurement that was not taken as an **em dash, never
a zero and never a green tick**, and `0` deliberately does not take that path
because a genuine zero is a result. Its colour swatch comes from
`colorForFDI`, the same pure function the viewport uses — a second palette is
how seventeen teeth came to share one colour (§24.5).

**`workflowSteps` is a MODULE-SCOPE PURE FUNCTION, not a hook.** A deps array
is evaluated during render and this app has white-screened five times from one
naming a `const` declared below it. A pure function over a plain object cannot
participate in that at all. **This is the eighth time that trap has been
checked rather than assumed.**

Measured: bundle 272.75 → **276.41 KB gzip** (+3.66 KB, no new dependency),
`npm run lint` 0 problems, `npm run smoke` RENDER OK, `npx playwright test`
**17 passed / 9 skipped** (the nine need a running backend).

### 26.13 CrossTooth carries no licence, and that is a commercial blocker

`CrossTooth/` contains **no LICENSE, no COPYING and no `.git`**, so there is
no licence text and no recoverable upstream commit to look one up against.
The code is vendored and unmodified and the checkpoint is excluded from
version control (§25.8), but *"a CVPR paper exists"* is not a grant of
rights. **This is independent of every technical finding above** and it
applies to the stated commercial mission, not to the research use it is
being put to here. It needs a licence from the authors before CrossTooth
output can ship in a product.

### 26.14 What was NOT done

* **The real scan still does not produce a PRINT READY stage.** One
  0.09515 mm self-touching edge, fully characterised in §26.8. No cap, gate
  or threshold was moved to get past it.
* **A contacting pair of moving teeth cannot both be reconstructed.** The
  shared-reconstruction branch does not exist; the refusal now names the
  measured bridge width instead (§26.6).
* **`old_site_restored` fails on the real scan and was not investigated.**
  It is one of the four failing gates and the attention went to the
  self-touch, which is the one the forensics could localise.
* **No accuracy benchmark exists for either model**, and none can be run
  here. §17 and §25.6 stand.
* **Only the mandible was exercised.** There is still no maxillary scan.
* **`App.jsx` is still one file** (~3,100 lines). Three components have been
  extracted from it; the top bar, the bottom staging dock and the migration
  of the `S` style object onto `theme.js` have not been done.
* **No browser performance figures.** There is a browser for Playwright but
  no profiler, and §24.9's list is unchanged.
* **The viewport darkening is still MITIGATED, not root-caused** (§22).

Covered by `test_click_to_select.py` (10), `verify_pointops.py`, the four
trim regressions in `test_manufacturing_interface.py`, the three ordering
regressions in `test_segmentation_mapping.py`, and `benchmark_providers.py`.
