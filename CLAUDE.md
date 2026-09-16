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
| API | **`api_core.py`** — `uvicorn api_core:app`, 31 routes. Launched by `start_backend.bat`. |
| Client | **`frontend/src/App.jsx`** — `npm run dev`, launched by `start_frontend.bat`. |
| Geometry | **`core_geometry.py`** — pure NumPy/SciPy, headless, the engine. |
| Docs | **`CLAUDE.md`** (this file) and **`README.md`**. No other document is authoritative. |

`app_ui.py` is a legacy PyQt6 desktop shell over the same engine — it runs, but it is not the
product. **`_archive/` is dead code** (see `_archive/README.md`); as of 2026-09-16 it holds
`server.py`, `HANDOVER.md`, two stale `App.jsx` copies, and the empty root `config.py`/`models.py`
that used to shadow the real `tooth_segmentation/` modules.

### TEST COMMANDS

```
python -m compileall .                  # syntax, whole tree
python check_structure.py               # undefined names without importing (84 files)
python run_all_tests.py                 # CANONICAL runner — 34 entries
python -m pytest -q                     # runs alongside; both must pass
node frontend/verify-kinematics.mjs     # cross-language kinematics pin
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

## 18. KNOWN OPEN ISSUES
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
  **Read 24/24 with the caveat below** — 3 of those entries assert nothing.
* `u_bl` is pinned buccal from the arch frame, so `u_md`'s sign varies by quadrant (unavoidable — the arch is mirror-symmetric). The frame reports `u_md_points_distal`; the sliders do not yet use it.
* `/teeth` exists but the client does not yet call it on load, so a browser refresh still loses the crowns from the scene (the server keeps them).
* **A pinched cast rim loses a small spur.** `_open_pinch_vertices` deletes the smaller fan at a neck, and `/cut` drops the smaller lobe of a self-touching socket rim. Both are a handful of triangles and both are recorded, but neither is reconstructed.
* **`trim_to_arch` is not stable on a very coarse mesh.** Measured: at ~18 samples across the band the trim nearly severs the arch and `np.bincount(labels).argmax()` picks a different largest component under rotation — an 80% volume swing. Unreachable at scan density (187k faces across the same band) and worth remembering before anyone decimates a scan upstream.
* **The staged export still needs one interactive wand cut to be judged end to end.** It passes
  on the synthetic arch and now REFUSES the real scan's auto-cut crowns cleanly and by name
  instead of fracturing — which is the correct behaviour, since those crowns are thin shells
  (8.8 mm³ at genus 2, a 0.2 mm speck, and one that fractures into 3). What has not been shown
  is a real crown cut with the wand passing the gate and fusing, because that cannot be produced
  headlessly. FDI 46 (136.3 mm³, χ=2) passes the screen and fuses to one body, which is the
  closest evidence available here.
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