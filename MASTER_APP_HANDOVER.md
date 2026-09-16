# MASTER APP HANDOVER — Clinical Micro-Planner (Project Aligner)

Clear aligner CAD workstation. Generated 2026-09-15 from a file-by-file scan of
`C:\Users\Lenovo\OneDrive\Desktop\Aligner_App`.

---

## 0a. WHAT CHANGED SINCE THIS DOCUMENT WAS WRITTEN (read this first)

This document was written against the repo as found. Successive hardening passes have since
changed the following, and **CLAUDE.md §13-§16 are the authoritative record**:

| | Then | Now |
|---|---|---|
| Version control | **none at all** | git, 9 commits, baseline first |
| Syntax errors | 1 (`tgn_find_width_config.py:161`) | **0** |
| `npm run lint` | 14 problems | **0** |
| `check_structure.py` | 2 hard-coded files | **61 files, tree-wide** |
| `pytest` | not installed | **131 passed** |
| Browser E2E | none | **7/7 (Playwright)** |
| CI | none | 2 jobs, AI path excluded by design |
| Case survives a refresh | no | **yes** |
| `cut_guard` | hard-bypassed, gate unreachable | un-bypassed, reports, does not yet gate |
| Collision reporting | `checked: false` read as "clear" | **5 explicit states** |
| Dead code | 9 files loose in the tree | quarantined to `_archive/` |
| Domain model | none | `Case -> Arch -> Tooth -> Prescription -> Stage` |
| Case persistence | none | encrypted local file, **treatment state only, 602 bytes** |
| Antagonist resolution | client passed `opposing_session_id` | **resolved server-side** |
| Viewport raycast | three.js default, 12.974 ms | **three-mesh-bvh, 0.023 ms (555x)** |
| Space analysis / IPR | machinery existed, called by nothing | `GET /space-analysis` |
| Segmentation review | none | `GET /segmentation-review`, PASS/REVIEW/FAIL + factors |
| Attachments | none | 4 parametric shapes, fused via manifold3d |
| Audit trail | none | closed vocabulary, PHI denylist, bounded |
| CBCT | none | interface only — every entry point raises |
| `run_all_tests.py` | 24/24 | **36/36** |
| API routes | 14 | **33 paths** |

### Fail-safe sprint (2026-09-16)

The happy paths above were already green. This pass added the CONTINGENCY half of
each, plus a DevSecOps stage that did not exist. **Four defects were found that
reported themselves as fine** — the dangerous kind:

| | Then | Now |
|---|---|---|
| Shadow rig | `three.current` was **replaced** after `sun`/`catcher` were attached, so `aimShadows` returned early every time and the rig never armed | both live in the literal |
| Shadow staleness | `shadowsDirty` set only by `aimShadows`; every cut left the **pre-cut** silhouette | set at every commit point |
| White screens | no React error boundary at all | `ErrorBoundary.jsx`, recovery message first |
| `over_threshold` | computed from `contact_mm` (0.30) beside a field named `threshold_mm` (0.05) that was compared against **nothing** | both constants named; the floor now suppresses sub-scanner closure |
| Silent `except: pass` | `core_geometry.py:438`, the only one on the live geometry path | warns and names the fallback |
| `space_analysis` | measured **T0** crowns while the clinician looked at the setup | poses each crown; the payload states which pose |
| NaN/Inf scan | survived the degenerate filter (every comparison against NaN is False), died later as "SVD did not converge" | `sanitize_scan`, **delete-only** — no surviving vertex moves |
| C_res projection | unbounded | clamped `[7,15]`mm and reported **loudly** |
| CSG failure | unguarded boolean | `fix_normals`+`fill_holes` → weld 1e-5 → **refuse by name** |
| Missing `manifold3d` | ImportError traceback, 500 | 503 naming the pip command |
| BVH build failure | silent early return, 13 ms/cast forever | `mergeVertices` retry — **crowns only**, since welding renumbers and arch ids are the session contract |
| Raycaster | collected and **sorted** every hit | `firstHitOnly`; all five sites take `[0]` |
| Hover without emissive | silent no-op, and the restore invented `emissiveIntensity` | `color.addScalar(0.12)`, restoring the **saved** hex |
| WebGL context loss | unhandled; the canvas stays black forever | `preventDefault` + shadow re-arm |
| Root cones | could hang below the cast in space never imaged | clamped to the scan's apical extent |
| Interproximal | measured once, on the final pose | swept every stage; trees hoisted, **8.86 s → 0.51 s (17.2×)** |
| Lost session | one status-bar line, overwritten by the next `setStatus` | persistent re-upload banner |
| Low AI confidence | the label drove the cut regardless | `< 0.70` arms the landmark picker |
| SAST | none | `.semgrep.yml`, 9 rules, CI job, **every rule verified to fire** |
| Telemetry | none | file-only spans; no exporter can be configured |
| Audit trail | written, called by nothing | recorded at upload, cut, prescription, both exports |
| `pytest` | 131 | **237** |
| `check_structure.py` | 61 files | **88 files** |
| Browser E2E | 7/7 | **19 specs** (12 pass; 7 need the backend and skip) |

**Superseded below:** §4.9's "LIVE BUG" is fixed; §4.10's `requirements.txt` row is fixed; §7's
test counts are now **36 suite entries / 19 browser E2E specs**; §0's dead files have moved to
`_archive/`; §1's "declared, never imported" row no longer applies to `three-mesh-bvh`.
**Every route and line count in the body predates this sprint.**

## 0. HOW TO READ THIS — AUTHORITATIVE FILES

This repo contains several stale duplicates. **Read the LIVE column; ignore the DEAD column.**
Every claim below is cited to a file and line that was verified to resolve at generation time.

| Concern | LIVE — read this | DEAD — ignore |
|---|---|---|
| HTTP API | `api_core.py` (1,507 lines, 14 routes, the FastAPI app) | `server.py` (224 lines, 10 stale routes) |
| React client | `frontend/src/App.jsx` (1,526 lines) | `frontend_src/App.jsx` (192-line stub), `frontend_App.jsx` (root, Aug 28) |
| Project record | `CLAUDE.md` (§1-§13, measured, current) | `HANDOVER.md` (15.9 KB, **architecture section is false** — see §4.8) |
| TGN checkpoint tool | `tgn_diagnose_model.py` | `tgn_diagnose_model.py.py` (double extension) |
| Segmentation models/config | `tooth_segmentation/models.py`, `tooth_segmentation/config.py` | root `models.py`, root `config.py` — **both 0 bytes, and they shadow the real ones** |

**Entry point:** `start_backend.bat` → `.venv\Scripts\python.exe -m uvicorn api_core:app --host
127.0.0.1 --port 8000 --reload`, and `start_frontend.bat` → `npm run dev` on port 5173.
A bare `python -m uvicorn` fails on this machine: the system Python 3.12 has none of the deps.

---

## 1. EXECUTIVE SUMMARY & TECH STACK

A client-server CAD workstation for single-step clear aligner planning: load an intraoral arch
scan, isolate a tooth, move it with biomechanically correct kinematics, export print-ready STLs.
Benchmarks are Deltaface and 3Shape Ortho System. It is a clinical tool, not a 3D sandbox.

### Architecture

```
frontend/src/App.jsx  (React 19 + three.js 0.185, Vite 8)
        |  HTTP, 12 endpoints, CORS-locked to :5173
        v
api_core.py           (FastAPI — 14 routes, sessions, orchestration, ZIP export)
        |
        +--> core_geometry.py     5,638 lines. Pure NumPy/SciPy. ALL geometry + kinematics.
        +--> tgn_bridge.py  -->   ToothGroupNetwork/  (vendored, PyTorch, never edited)
        +--> session_store.py     In-memory, TTL, PHI-conscious
        +--> arch_frame.py        Occlusal reference basis
```

`app_ui.py` (920 lines, PyQt6 + PyVista) is a **parallel legacy desktop shell** over the same
`core_geometry`. It is not dead — it still runs — but it is not the product path.

### Stack, as actually installed

| Layer | Real |
|---|---|
| Python | **De facto 3.12.7** (`.venv/pyvenv.cfg`). Source floor 3.10 (bare PEP-604 unions, `api_core.py:132`). **No stated minimum exists** — no `pyproject.toml`, `setup.py`, or `python_requires` anywhere. |
| Backend | FastAPI 0.141.1, uvicorn 0.52.4, python-multipart 0.0.32, NumPy 2.5.2, SciPy 1.18.1 |
| Geometry booleans | `manifold3d` 3.5.2 — load-bearing; listed in `requirements.txt` since 2026-09-15 |
| AI segmentation | PyTorch 2.13.0+cpu, open3d 0.19.0, ToothGroupNetwork (vendored) |
| Collision / proximity | **`scipy.spatial.cKDTree`** — see §6, the brief's "python-fcl" is not present |
| Frontend | React 19.2.8, three 0.185.1, Vite 8.2.2, @radix-ui accordion + slider, lucide-react |
| Declared, never imported | `three-mesh-bvh` (`package.json:21`), `three-bvh-csg` (`:20`) |

### THE CORE INVARIANT — Sacred Scanner Coordinates

**Raw STL vertices are never rotated, re-centred, or re-scaled.** Inter-arch bite registration
depends on both arches sitting in the same raw scanner space. "Up" and "forward" come from an
**Occlusal Plane Reference Frame** stored as session metadata (`arch_frame.py:49-52`), never by
moving geometry.

This is not style. Three things depend on it directly:
1. The antagonist collision check (§2, Module 5) only measures anything real because neither arch
   was normalised to its own origin.
2. Every vertex index already sent to the browser stays valid — extraction is an index-buffer
   rewrite, never a mesh rebuild (`App.jsx:143-201`).
3. Cached geodesic fields and precomputed adjacency survive every cut.

---

## 2. COMPLETED MODULES

### Module 1 — Ingestion & Spatial Calibration ✅
Dual-arch STL/OBJ upload (`api_core.py:183`), welded in pure NumPy (`stl_io.py` — welding is
mandatory, not optional: on unwelded triangle soup a graph region-grow selects exactly one face).
Occlusal frame from 3 clicked landmarks → `u_occ` / `u_sag` / `u_tra` (`arch_frame.py:102`).
Note `basis_matrix()` stacks them as columns in the order **(u_tra, u_sag, u_occ)**.

Rendering is **`MeshPhysicalMaterial`**, roughness 0.35 / metalness 0.05 / clearcoat 0.6
(`App.jsx:69`), lit by a genuine 3-point rig **parented to the camera** (`App.jsx:424-431`) —
scanner axes are arbitrary, so a world-fixed key light lands somewhere different on every case.
ACES tone mapping (`:410`) + `RoomEnvironment` PMREM (`:472`) give the clearcoat something to
reflect. There is **no MatCap material** anywhere (§6).

### Module 2 — Segmentation & Crown Isolation ✅
Two paths. **AI:** ToothGroupNetwork via `tgn_bridge.py` — 236 s on a real 187k-face arch,
11 teeth. **Manual:** 3-click landmarks (mesial, distal, FA point) + curvature-weighted geodesic
flood-fill (`/wand`, `api_core.py:285`).

Crown capping is `cap_and_close` (`core_geometry.py:539`): **Delaunay first, centroid fan as
fallback**, and the fan wins whenever Delaunay's output would break manifoldness — the gate is
`_creates_nonmanifold` at `:573`. This matters more than it sounds (§4.7).

### Module 3 — Biological Kinematics & C_res ✅
Teeth pivot about the Centre of Resistance, extrapolated apically into alveolar bone along the
long axis. The axis comes from the **normal of the plane fitted through the cervical rim**
(`resolve_long_axis`), signed against `u_occ` and clamped to 20° of arch apical — not from
`crown_centroid − rim_centroid`, which collapses on a short broad molar and caused the
"tetherball glitch" (CLAUDE.md §5).

Root lengths, `toothGizmo.js:467`: **incisor 10 mm, canine 13 mm, premolar 9 mm, molar 9 mm**
(Wheeler averages, four buckets not 32 values). `CutRequest.root_length_mm` is **required with no
default** (`api_core.py:140`) — a silent default once shipped a mandibular canine on 10 mm.

### Module 4 — 6-DoF Kinematic Engine & 3D Gizmo ✅
Affine transforms about C_res. Composition is `R_torque(u_MD)·R_tip(u_BL)·R_rot(u_OA)` = Euler
**`XYZ`** in the pivot's local frame (an earlier `ZYX` cost 1.71° of pure decomposition error).
Two-way numeric steppers are absolute from T0, not nudges. Multi-tooth persistence verified to
8.9e-16 mm.

### Module 5 — Proximity & Collision ✅ *(technique differs from the brief — see §6)*
- **Antagonist occlusion** (`core_geometry.py:5478`): nearest-vertex distance **signed** against
  that vertex's normal. Unsigned cannot tell 0.9 mm of penetration from 0.9 mm of clearance.
  Approximate by construction — nearest *vertex*, not nearest surface point — because a real SDF
  needs a closed mesh and an intraoral scan is an open shell. **A warning, never a gate.**
- **Interproximal** (`:5573`): reports **clearance closure** (how much gap the movement consumed),
  not inter-tooth penetration depth.
- Both on **`scipy.spatial.cKDTree`**. The tree is cached (`:5467`) — rebuilding per call was
  43 ms of the 89 ms a 2,000-vertex crown cost, and a 31-stage × 14-crown run asks 434 questions
  of an arch that never changes. `workers=-1` (`:5517`) is a 3-4× win on 8 cores.

### Module 6 / 8 — Virtual Horseshoe Cast Base ✅
`trim_to_arch` → `build_cast_base`. Trim to a horseshoe band about the fitted occlusal ridge,
extrude to a flat-bottomed solid, assert zero open and zero non-manifold edges.

This **replaced** `cap_and_close` on the base, which was producing a membrane: an arch perimeter is
a horseshoe, so the cap spanned the tongue opening — measured **2,104 triangles for a 2,101-vertex
loop in 100.2 seconds**. Trim+extrude puts **0 of 10,852** faces over the opening.
Measured: trim 0.6 s + extrude 0.5 s, base 190,036 faces, 33,953 mm³, watertight.
`ARCH_TRIM_MARGIN_MM` is **7.0**, chosen from measurement (§4.2).

### Module 7 — Staging Timeline ✅ **COMPLETE** *(the brief lists this as in progress)*
Stage k is `clinical × k/N` **rebuilt through `kinematic_matrix`** — never an interpolation of the
committed 4×4. Measured: every stage rigid to **4.4e-16** (RᵀR−I); a 4×4 lerp of the same movement
reaches **1.68e-2** and is not a rotation at all. `verify-kinematics.mjs:184` *requires the bad
path to measurably fail*, which proves the claim rather than asserting it.
Scrub cost, 14 crowns: **12.5 µs/frame** — 0.075 % of a 60 fps budget. React is kept out of the
per-frame path entirely (refs + rAF, `StagingTimeline.jsx:112-143`). Zero network calls per frame.

### Module 9 — Manufacturing CSG Export ✅ **COMPLETE** *(the brief lists this as in progress)*
`manifold3d` boolean union fusing moved crowns + root plugs into the socketed base, one watertight
solid per stage, sockets filled flush. Measured, 31 stages: **CSG 1.00 s (0.032 s/stage)**,
collision 0.44 s, **4.21 s total** including STL writing and zipping; 16 inverted crumbs purged.

---

## 3. MODULE STATUS & GENUINELY OPEN WORK

| Module | Status |
|---|---|
| 1 Ingestion / calibration | ✅ Complete |
| 2 Segmentation / isolation | ✅ Complete |
| 3 C_res kinematics | ✅ Complete |
| 4 6-DoF gizmo | ✅ Complete |
| 5 Proximity / collision | ✅ Complete (cKDTree, not FCL) |
| 6/8 Virtual cast base | ✅ Complete |
| 7 Staging timeline | ✅ **Complete and measured** |
| 9 Manufacturing CSG export | ✅ **Complete and measured** |

### What is actually open

1. **Segmentation cannot be cancelled.** 236 s on a real scan; the only exits are waiting or
   restarting. `asyncio.to_thread` offers no cancellation point inside the model — a real cancel
   needs a subprocess.
2. **`/teeth` exists but the client never calls it on load** — a browser refresh loses the crowns
   from the scene even though the server still has them.
3. **The antagonist check has never run against a real opposing arch.** No upper-arch scan exists
   in this repo; every test uses a synthetic plate or the lower arch mirrored. Arithmetic pinned,
   clinical result not.
4. **The staged export has never been judged end-to-end on an interactive wand cut.** It correctly
   *refuses* the real scan's auto-cut crowns (thin shells: 8.8 mm³ at genus 2, a 0.2 mm speck, one
   that fractures into 3) — right behaviour, but not the same as a passing run.
5. **OutlinePass is unprofiled** (§6). Emissive boost ships in its place.
6. **A pinched cast rim loses a small spur** — recorded, not reconstructed.
7. **Attachments and CBCT** are the next phase.

---

## 4. FAILURE ANALYSIS, BOTTLENECKS & LESSONS LEARNED

### 4.1 The `components: 2` floating island was a logging artifact
There was never a floating island in the exported base. Measured: `components = 1`, sizes
`[190082]`. The manifest's `components` field reported islands **found**, not islands
**surviving**. It now reports the output (always 1) plus `islands_found` / `islands_removed`, and
`build_cast_base` asserts one component outright.
**Lesson: a diagnostic that reports an intermediate is a bug report about the wrong thing.**

### 4.2 Wall striations came from decimation, not from a bad triangulator
`prune_to_simple` decimated the projected rim outline to 0.5 mm before triangulating. The wall
bridges each rim vertex to its nearest surviving floor vertex, so four or five rim vertices sharing
one floor vertex fan into slivers — visible striations.

| | before | after |
|---|---|---|
| rim → floor (real scan, margin 7) | 2,133 → 469 (**78.0 % discarded**) | 2,133 → 2,014 (**5.6 %**) |
| forced ear clips | — | **0** |
| triangulated vs outline area | — | exact to 3e-16 |

The fix was **coincident-projection dedup**, not a cleverer triangulator. Two rim points projecting
onto one stalled the ear clip, and the spacing ladder papered over it.
**Lesson — THE INVARIANT IS AREA, NOT COUNT.** `ear_clip_polygon_robust` always returns n−2, so
counting against it is vacuous: it passed while a quarter of the floor was missing.
*(The brief cites 1,688 of 2,157 — the recorded measurement is 1,664 of 2,133. See §6.)*

### 4.3 Why sockets fell back to `flat_fallback` — a verdict on the CUT, not the cup
`fallback_reason` now names the condition with its number. Same real tooth, two ways:

| selection | rim | min radius about its centre | result |
|---|---|---|---|
| raw FDI labels | 232 pts, 7.3 × 2.3 mm | 0.134 mm | `flat_fallback` |
| geodesic wand flood (as the app does) | 213 pts, 10.6 × 5.6 mm | 1.342 mm | **cup** |

A label-picked crown has a ragged self-touching margin; a real flood does not. `MIN_SOCKET_INSET_MM`
(0.25) sends rims with too little clearance to a flat floor rather than building a 0.07 mm "cup"
that is a flat lid by another name.

### 4.4 OutlinePass was NOT profiled — and that is the honest answer
`EffectComposer` / `OutlinePass` appear nowhere in the source. The decision to ship an **emissive
boost** instead (`App.jsx:1229`) rests on an *a priori* argument — four full-screen passes per
frame whether anything is hovered or not, versus a uniform change on one material. **It was never
measured, because there is no browser in this environment and a number that cannot be measured will
not be reported** (`App.jsx:1210-1228`). Profile it in a browser before switching.
**Lesson: refusing to invent a number is part of the engineering.**

### 4.5 Matrix interpolation vs parameter interpolation
LERPing a 4×4 directly is not a rotation: the 3×3 block of `(1−t)I + tR` is not orthonormal for any
intermediate `t`, so the crown shears at every stage. Measured **1.68e-2** orthogonality error
against **4.4e-16** for parameter interpolation. Both numbers are pinned in **two languages**
(`verify-kinematics.mjs`, `test_staging_export.py`) because the clinician approves the scrub and the
lab prints the files.

### 4.6 Bundle weight — framer-motion removed, but its replacement was never written
Baseline 207.1 KB gzip. Four libraries together came to +51.5 KB (+24.8 %), over the agreed +20 %
gate. framer-motion alone was **+39.4 KB** and its only job was scrub transitions — precisely the
thing that must not go through React. Removed; final bundle 222.2 KB (+7.3 %).

**Open defect found by this scan:** `Panel.jsx:12-13` claims the replacement is "a CSS height
transition [that] costs nothing and looks the same". **There is no such CSS.** `src/index.css` is
**0 bytes**, `className="panel-trigger"` (`Panel.jsx:29`) matches no rule, and no Radix accordion
keyframes exist. The panels snap open with no transition. The removal is real; the mitigation is not.

### 4.7 Four bugs that reported themselves as clean
Edge multiplicities are orientation-blind and STL stores positions, so a base can measure watertight
and still be wrong. `_winding_is_consistent` and **a re-read of the written STL** caught these:
a floor returning CCW regardless of rim winding (229 directed edges agreeing on one sealed socket,
zero open the whole time); one floor vertex per rim vertex so a folded margin welded two into one;
a pinch vertex **deleted, not duplicated** (duplication is the textbook repair and is wrong — an STL
reader welds the copies straight back); and a socket rim passing through one vertex twice.

Related, and the reason Delaunay is gated in `cap_and_close`: a non-planar cervical loop lets
Delaunay emit a chord between two vertices that already share an edge elsewhere, giving that edge
four faces. The result has **no open boundary and is still non-manifold** — it passes a naive
"any holes left?" check and then breaks the CSG union.
**Lesson: watertight ≠ correct. Test the property you actually need.**

### 4.8 `HANDOVER.md` is stale and actively misleading
It describes `server.py` as "FastAPI routes over api_core" and `api_core.py` as "backend logic (no
FastAPI import, so it is testable)". **Both are false.** `api_core.py` *is* the FastAPI app with 14
routes; `server.py` is a superseded prototype nothing launches. Any AI reading that file will
model the architecture wrongly. This document supersedes it.

### 4.9 ~~LIVE BUG~~ **FIXED 2026-09-15** — `App.jsx:1104` threw on every occlusal-plane definition
```js
for (const rec of Object.values(arches.current)) if (rec) box.expandByObject(rec);   // was
for (const rec of Object.values(arches.current)) if (rec?.mesh) box.expandByObject(rec.mesh);  // now
```
`arches.current[...]` is a plain record `{mesh, geometry, view, brushIndex, ...}` (`App.jsx:621`),
not an `Object3D`. `Box3.expandByObject` calls `object.updateWorldMatrix(...)` on its first line
(`three/src/math/Box3.js:308`) → **TypeError**.

**Precise impact** — the obvious reading overstates this. `setArchFrame(data)` (`:780`) and the POST
have already succeeded when `aimShadows(data)` (`:784`) throws, so **the occlusal plane IS
established on both client and server, and `/cut` will not 409.** The damage is:
1. A step that worked reports `"Occlusal Plane cancelled: object.updateWorldMatrix is not a
   function"` — a clinician would retry a step that already succeeded.
2. The shadow subsystem never arms: `sun.intensity` stays 0, the catcher stays invisible.

**Fixed:** both call sites now use `rec.mesh`, which is a genuine `THREE.Mesh` (`App.jsx:617`).
`App.jsx:1123` (`rec.castShadow = true` on the same plain record) was the same mistake — a *silent*
no-op rather than a throw, so fixing only `:1104` would have armed the rig while leaving the arch
still casting no shadow. Both were corrected together. Verified: `npm run smoke` renders clean and
`npm run build` succeeds.

### 4.10 Other defects found by this scan
| Where | Defect |
|---|---|
| `App.jsx:1319` ← `:1375` | `/wand/threshold` fires on **every** range-input `onChange`. No debounce — one round trip per pixel of slider drag. |
| `App.jsx:519`, `:1308` | `PUT /selection` is fire-and-forget — no `await`, no `.catch`. A failed persist is invisible and client/server state silently diverges. |
| `App.jsx:1053` | `needsRender` is written and never read. Implies an on-demand render loop; `:528-540` renders unconditionally every frame. |
| `brush.js:233` | `applyKinematics` exported, never imported. |
| `requirements.txt` | ~~`manifold3d`, `torch`, `sklearn` absent or commented out~~ **FIXED 2026-09-15.** `manifold3d` uncommented; `torch`, `scikit-learn`, **`trimesh` and `open3d`** added. The last two were missed by the original diagnosis and are equally load-bearing: `inference_pipeline_tgn.py:1` imports `gen_utils`, which imports `trimesh`, and both import `open3d`. A clean install now works. |
| root `config.py`, `models.py` | 0 bytes, and they **shadow** real modules of the same name in `tooth_segmentation/`. |
| `.github/` | Contains only an agent definition. **No CI — the test suite never runs automatically**, which is what let two tests stay red against a stale API surface for as long as they did. |

---

## 5. EXACT REPOSITORY FILE MAP

Health: **GREEN** = live and working · **RED** = known failing · **DEAD** = stale/unused ·
**3P** = vendored third party.

### 5.1 Backend — application core

| File | Lines | Purpose | Key API | Health |
|---|---|---|---|---|
| `core_geometry.py` | **5,638** | The clinical engine. Pure NumPy/SciPy, no Qt/VTK/trimesh, deliberately headless-testable. Topology, curvature, geodesic region grow, segmentation, kinematics, capping, cast base, collision, STL writers. ~125 public functions. | `manifold_report`, `cap_and_close`, `derive_frame_from_region`, `center_of_resistance`, `kinematic_matrix`, `staging_estimate`, `build_socket_cup`, `trim_to_arch`, `build_cast_base`, `check_occlusal_collision`, `build_antagonist_index`, `ear_clip_polygon_robust`, `crown_is_printable`, `write_binary_stl` | GREEN |
| `api_core.py` | **1,507** | The live FastAPI app. Sessions, segmentation, wand, cut, kinematics, streamed ZIP exports. Mostly orchestration + HTTP error translation. | 14 routes (§5.4); `run_segmentation`, `build_export_bundle`, `build_stage_bundle`, `_to_manifold`, `_solid_bodies`, `_screen_crowns_for_manufacturing` | GREEN |
| `tgn_bridge.py` | 306 | Sole contact point with the vendored TGN checkout. **Nothing imports at module load**, so a broken checkout cannot stop the server — segmentation is an enhancement, never allowed to take down the clinical loop. `load()` is idempotent under `_LOCK` and never raises. | `load`, `status`, `predictor`, `diagnose`, `_RepoOnPath` | GREEN |
| `cpu_compat.py` | 272 | Process-wide monkeypatch making CUDA-assuming TGN code run on CPU-only PyTorch. | `install`, `uninstall`, `selftest` | GREEN |
| `pointops_cpu.py` | 174 | Pure-PyTorch CPU drop-in for the `pointops` CUDA extension. `install()` inserts a `sys.meta_path` finder so the vendored repo imports it **without a single edit to third-party source**. | `furthestsampling`, `knnquery`, `queryandgroup`, `interpolation`, `install` | GREEN |
| `session_store.py` | 164 | In-memory, thread-safe (`RLock`), TTL-expiring session store. PHI-first by design: scans never touch disk, 1-hour TTL, max 4 resident (deterministic LRU), **original filename stored nowhere**. | `SessionStore`, `SessionExpired`, singleton `STORE` | GREEN |
| `arch_frame.py` | 184 | Occlusal basis from 3 landmarks. Geometry never moves. `reconcile_tooth_frame` **keeps the sign** — an earlier `abs(cos)` let an inverted axis score a perfect 0° while C_res was extrapolated above the crown. | `fit_occlusal_frame`, `to_json`, `basis_matrix`, `reconcile_tooth_frame` | GREEN |
| `jaw_naming.py` | 123 | Upper/lower verification — decides the FDI quadrant offset, must never be guessed. Parses 3 different label shapes; returns a structured verdict + human diagnosis. | `verify_fdi_matches_jaw`, `extract_labels`, `jaw_for_arch`, `drop_third_molars`, `scan_filename` | GREEN |
| `cut_guard.py` | 76 | Heuristic sanity gate on a wand cut + auto tolerance search. | `check_crown`, `auto_tolerance`, `compactness`, `rim_concavity` | GREEN |
| `stl_io.py` | 41 | Binary STL reader, pure NumPy. Validates header/count against file length; welds with `np.unique` on **exact bit patterns** so no coordinate is ever altered. | `parse_stl_bytes` | GREEN |
| `curvature.py` / `normals.py` | 19 / 4 | Thin wrappers over `core_geometry`. Duplicated in `tooth_segmentation/`. | — | GREEN |
| `app_ui.py` | 920 | Legacy PyQt6 + PyVista desktop shell over the same engine. Runs, but not the product path. | `PlannerWindow`, `SegmentWorker`, `PrepWorker` | GREEN (legacy) |
| `server.py` | 224 | Older FastAPI router layer. Describes itself as "thin routing over api_core"; **no longer matches it**. Nothing launches it. | — | **DEAD** |
| `config.py`, `models.py` | **0** | Empty. Shadow real modules in `tooth_segmentation/`. | — | **DEAD** |

### 5.2 Backend — dev, diagnostic, fixtures

| File | Lines | Purpose | Health |
|---|---|---|---|
| `run_all_tests.py` | 67 | Headless runner, 24 entries. Forces UTF-8 on children (a cp1256 console once killed a fully-passing test on a `mm³`). | GREEN |
| `scan_report.py` | 164 | STL scan health report. | GREEN |
| `inspect_segmentation_output.py` | 202 | TGN JSON inspector — FDI mapping, spacing classification. | GREEN |
| `tgn_diagnose_model.py` | 149 | Checkpoint + valid model-name inspector. | GREEN |
| `tgn_match_architecture.py` | 195 | Matches checkpoint prefixes to declared model attributes. | GREEN |
| `tgn_find_width_config.py` | 160 | Scans TGN repo for a width/config literal. | GREEN |
| `check_structure.py` | 89 | AST undefined-name checker, no import needed. | GREEN |
| `inspect_project.py` | 49 | AST-based architecture report. | GREEN |
| `diagnose.py` | 75 | Prints what pyvistaqt actually exposes. | GREEN |
| `transport_bench.py` | 40 | Wire-size benchmark: STL vs JSON vs gzip. | GREEN |
| `arch_fixture.py` / `tooth_fixture.py` / `incisor_fixture.py` | 58/71/41 | Synthetic mesh fixtures for the suite. | GREEN |
| `tgn_diagnose_model.py.py` | 150 | Duplicate, double extension. | **DEAD** |

### 5.3 `tooth_segmentation/` — modular segmentation package (521 lines)

| File | Lines | Purpose |
|---|---|---|
| `label_adapter.py` | 157 | Coarse model output → full mesh resolution via `cKDTree`; kNN beats nearest at boundaries. Speckle cleanup, per-tooth extraction. |
| `mesh_preprocessor.py` | 70 | Degenerate-face removal, optional welding. |
| `arch_geometry.py` | 65 | `ArchFrame` estimation from vertices + concavity. |
| `models.py` | 60 | Dataclasses: `PreprocessReport`, `ToothCandidate`, `ArchFrame`. |
| `config.py` | 27 | `SegmentationConfig`, `BoundaryWeights`. |
| `curvature.py` / `normals.py` | 21 / 5 | Wrappers. |
| `tests/` | 211 | The only two test files outside the repo root. |

### 5.4 The 14 HTTP routes (`api_core.py`)

| # | Method | Path | Line | Notes |
|---|---|---|---|---|
| 1 | GET | `/api/ai/status` | 83 | `loaded`, `warming`, `warm_seconds`, `error`, `segmentation_in_progress` |
| 2 | POST | `/api/session` | 183 | multipart: `arch` Form (`upper`/`lower`) + `file` |
| 3 | GET | `/api/session/{sid}/mesh` | 213 | → positions + indices (8.6 MB on a real arch) |
| 4 | POST | `/api/session/{sid}/occlusal-plane` | 221 | 3 landmark points → frame |
| 5 | POST | `/api/session/{sid}/segment` | 239 | async; `asyncio.to_thread`; **409 on duplicate**; ~236 s |
| 6 | POST | `/api/session/{sid}/wand` | 285 | geodesic flood from a click |
| 7 | POST | `/api/session/{sid}/wand/threshold` | 308 | re-threshold existing flood |
| 8 | PUT | `/api/session/{sid}/selection` | 316 | persist selection |
| 9 | POST | `/api/session/{sid}/cut` | 326 | → crown, root cone, socket cap, removed faces, C_res. **409 without an occlusal plane** |
| 10 | POST | `/api/session/{sid}/tooth/{tid}/kinematics` | 572 | clinical 6-tuple; `opposing_session_id` optional |
| 11 | GET | `/api/session/{sid}/teeth` | 611 | committed poses — **client never calls this** |
| 12 | POST | `/api/session/{sid}/export` | 1012 | streamed ZIP: base + per-tooth STL + manifest |
| 13 | POST | `/api/session/{sid}/export/stages` | 1481 | fused manufacturing solids; `X-Occlusal-Interference` header, CORS-exposed at `:1497` |
| 14 | DELETE | `/api/session/{sid}` | 1505 | close + free |

CORS: `allow_origins=["http://localhost:5173","http://127.0.0.1:5173"]` (`api_core.py:30-35`).
Startup hook `_warm_tgn` (`:50`) loads the model on a **daemon thread** so uvicorn binds the port
immediately — an inline load made the port refuse connections for its whole duration.

### 5.5 Frontend (~2,694 lines of hand-written source)

| File | Lines | Purpose | Health |
|---|---|---|---|
| `src/App.jsx` | **1,526** | The whole client: three.js bootstrap, lighting/shadow rig, material factories, index-only extraction, all 12 API calls, sidebar, inline style object `S`. | GREEN (1 live bug, §4.9) |
| `src/toothGizmo.js` | 479 | C_res-pivoted gizmo + the kinematics math mirroring Python. Row-major transpose done **once, on purpose** (`:337-343`). `quatFromFrame` checks finiteness *before* orthogonality — NaN passes every comparison. | GREEN |
| `src/brush.js` | 249 | Uniform-grid vertex hash + pointer drag loop, stroke interpolation, backface rejection. This is the hand-rolled substitute for `three-mesh-bvh`. | GREEN |
| `src/frameArch.js` | 187 | Camera framing **without mutating coordinates** — PCA occlusal axis via inverse power iteration, bails to a safe axis rather than letting NaN reach `camera.up`. | GREEN |
| `src/StagingTimeline.jsx` | 181 | Radix Slider + lucide transport; rAF playback clock, 280 ms/stage. | GREEN |
| `src/Panel.jsx` | 61 | Radix Accordion sidebar sections. | GREEN (claims absent CSS, §4.6) |
| `src/main.jsx` | 10 | React root, StrictMode. | GREEN |
| `src/index.css` | **0** | Empty. Imported by `main.jsx:3`. | **DEAD** |
| `verify-kinematics.mjs` | 190 | Cross-language golden-matrix check. **In no npm script and no CI** — must be run by hand. | GREEN |
| `ssr_smoke.jsx` | 15 | SSR render of `<App/>` — catches temporal-dead-zone errors a `vite build` cannot. | GREEN |
| `vite.config.js` | 7 | Bare. No proxy, so `API` is hardcoded to `http://127.0.0.1:8000` (`App.jsx:21`). | GREEN |
| `index.html` | 13 | Untouched template — still `<title>frontend</title>`. | GREEN |
| `README.md` | 16 | Untouched `create-vite` boilerplate. | **DEAD** |
| `frontend_src/App.jsx` | 192 | Stale stub of the live file. | **DEAD** |

npm scripts: `dev`, `build`, `lint`, `preview`, `smoke`. **No `test` script.**

### 5.6 `ToothGroupNetwork/` — 3P, VENDORED, NEVER EDITED
~3,255 lines across the top level, `inference_pipelines/`, `models/`, `train_configs/`, plus
`external_libs/` (pointnet2_utils, pointops, scheduler) and 12 × 64 MB `.h5` checkpoints
(**excluded from the export archive**). Reached **only** through `tgn_bridge.py`, which guarantees
it is never modified: `diagnose()` reports missing `__init__.py` files and `materialise_packages()`
writes them only on explicit request. `_RepoOnPath` gives the repo first claim on imports during
its own load then demotes it, so names like `models` cannot capture the app's own modules.

### 5.7 Manifests — **generated in memory, zero on disk**

The brief asks for `manifest.json`; there are **none in the repo**. Both are built and streamed
inside the export ZIP, landing on disk only under `keep_local=True` (`api_core.py:992`, `:1464`).

**Schema 1 — planned setup** (`build_export_bundle`, `:966-986`): `generator`, `arch`, `base_file`,
`base_watertight`, `base_health`, **`base_health_as_written`** (re-measured from the written STL
after the float32 + weld round-trip), `scan_health_at_upload` (so export can blame the scan, not the
cut), `faces_removed`, `units`, `coordinate_space`, `base_construction`, `trim`, `cast_base`,
`sockets`, `teeth[]` (`tooth_id`, `fdi`, `file`, `moved`, `prescription`, `staging`, `clearance`,
`root_length_mm`, `c_res`).

**Schema 2 — staged manufacturing** (`build_stage_bundle`, `:1423-1458`): adds `kind`, `stages`,
`socket_treatment`, `stage_files[]` (per stage: `occlusal_interference`, `closed`, `genus`,
`welded_open_edges`, `survives_a_welding_reader`), `union_seconds`, and `occlusion` (`checked`,
`threshold_mm`, `stages_with_interference`, `worst_penetration_mm`).
**`checked: false` means the antagonist was never loaded — NOT "no interference".**

**PHI:** neither schema carries a patient-identifying field. No `patient_name`, `dob`, `mrn`, or
address key exists. Identifiers are opaque hex session/tooth IDs. `scan_filename()`
(`jaw_naming.py:21-26`) strips every non-alphanumeric char and `api_core.py:107` passes a hard-coded
`patient_id="conditioned"`, so no caller-supplied name ever reaches a filename.

> **One note for whoever receives the export archive:** `case_lower.stl_output.json` (657 KB) is
> included at your instruction. It holds 94,848 per-vertex FDI labels derived from a **real
> mandibular scan** and has an `id_patient` key (currently empty). No name, DOB or MRN — but arch
> geometry is biometric data. The `.stl` scans themselves are excluded by the extension allowlist.

---

## 6. CLAIMS-vs-MEASUREMENT RECONCILIATION

The originating brief prescribed content that the code contradicts. Each row was verified against
the cited line. **Where they differ, this report follows the code.**

| # | Brief claimed | Measured reality |
|---|---|---|
| 1 | "`python-fcl` distance trees calculating interproximal penetration depth in mm" | **REFUTED.** No `fcl` import, string, or requirement anywhere. It is `scipy.spatial.cKDTree` (`core_geometry.py:5467`, `:5517`, `:5573`). The interproximal metric is **clearance closure against the base**, not inter-tooth penetration depth. |
| 2 | "Delaunay/centroid-fan **root** capping" | **MISFILED.** The pairing is real and well-built — Delaunay-first, gated by `_creates_nonmanifold` (`core_geometry.py:573`), fan as guaranteed fallback (`:555-558`) — but it caps the **crown's cervical rim**. The socket and cast-base floors **reject Delaunay outright** (`:3883`). There is no root capping: the root is a display **cone** (`api_core.py:643`). |
| 3 | "ceramic enamel **MatCap** shaders" | **REFUTED.** `MeshMatcapMaterial` appears nowhere. It is `MeshPhysicalMaterial` + clearcoat (`App.jsx:69`), ACES (`:410`), PMREM (`:472`). |
| 4 | "`three-mesh-bvh`" as core stack | **DECLARED, NEVER IMPORTED** (`package.json:21`). Same for `three-bvh-csg` (`:20`). Spatial lookup is a hand-rolled uniform grid (`brush.js:26-103`). Tree-shaken out of the build, so the cost is confusion, not bytes. |
| 5 | "OutlinePass rejected due to full-screen depth/blur bottlenecks on 200,000-face meshes" | **NEVER PROFILED.** No browser available; stated explicitly at `App.jsx:1210-1228`. The emissive boost shipped on an *a priori* argument. Reporting this as measured would invent a number. |
| 6 | Modules 7 + 9 are "being attempted" | **BOTH COMPLETE AND MEASURED.** Staging 12.5 µs/frame, rigid to 4.4e-16. CSG 31 stages in 1.00 s. |
| 7 | "discarding **1,688 of 2,157** rim vertices" | Recorded measurement is **2,133 → 469 kept** (1,664 discarded), fixed to **2,133 → 2,014**. |
| 8 | "Python 3.10+" | **No stated minimum exists.** De facto 3.12.7; source floor 3.10 (`api_core.py:132`). |
| 9 | Analyse `manifest.json` | **Zero on disk** — generated in memory, streamed in the ZIP (§5.7). |
| 10 | "Socket fallback … clearance < 0.25 mm" | Near — the constant is `MIN_SOCKET_INSET_MM = 0.25`, an **inset** floor, not a clearance floor. |
| 11 | "3-point studio lighting rig" | **CONFIRMED** — and better than claimed: key/fill/rim are **parented to the camera** (`App.jsx:424-431`), with a 4th world-fixed sun for shadows only. |
| 12 | Root lengths 10 / 13 / 9 mm | **CONFIRMED** (`toothGizmo.js:467`). The brief omits premolar, also 9 mm. |
| 13 | `(u_occ, u_sag, u_tra)` | **CONFIRMED** as names (`arch_frame.py:102`). Note `basis_matrix()` stacks columns **(u_tra, u_sag, u_occ)**. |
| 14 | `components: 2` logging artifact | **CONFIRMED** (§4.1). |
| 15 | Matrix-lerp distortion | **CONFIRMED and pinned in two languages** (§4.5). |
| 16 | framer-motion +39.4 KB removed, 12.5 µs/frame | **CONFIRMED** — but the stated CSS replacement does not exist (§4.6). |
| 17 | Virtual horseshoe base purges the tongue web | **CONFIRMED** (§2, Module 6/8). |
| 18 | Radix UI, lucide-react | **CONFIRMED**, both imported and used. |

---

## 7. TEST SUITE

`python run_all_tests.py` → **24 PASS, 0 FAIL** (runner exit code 0, verified without a pipeline in
the way). 24 entries, **4,166 lines, 112 `def test_*`, 401 assertions**. All 24 files exist; nothing
skips. The runner has **no allow-list or xfail** — it judges purely by exit code, so a green run
means every file genuinely exited 0.

### The two long-standing reds were cleared on 2026-09-15

Both were **stale API surface, never broken geometry.**

**`test_api_core.py` — rewritten.** The old file called five functions that do not exist
(`preview_selection`, `unpack`, `get_session`, `auto_color`, `parse_stl_bytes`) plus three with
changed signatures, and died at line 26. It now covers the nine endpoints nothing else touched —
`POST /api/session`, `GET /mesh`, `POST /occlusal-plane`, `POST /wand`, `POST /wand/threshold`,
`PUT /selection`, `GET /teeth`, `GET /api/ai/status`, `DELETE` — in **14 tests / 44 assertions**,
deliberately not duplicating `test_cut_endpoint.py`, which owns `/cut`, `/kinematics` and both
exports. Two contracts in it are worth knowing about:

* **The scan is never re-centred.** The fixture is pushed to `[137.5, -62.25, 41.0]` and the test
  asserts the returned bbox still carries that offset. Re-centring is the most tempting cleanup in
  a mesh pipeline and CLAUDE.md rule 3.1 forbids it, because inter-arch bite registration, every
  vertex id already sent to the browser, and every cached geodesic field all depend on raw scanner
  space. This pins it at the API boundary.
* **Every session endpoint gives 404, never 500, on an expired session.** Scans are memory-only and
  expire on a timer, so a client *will* meet one in normal use; an unhandled `SessionExpired` is a
  500 that looks like a server fault and tells the clinician nothing.

`/segment` is deliberately not exercised — it needs a 64 MB torch checkpoint and ~236 s on a real
arch, and its contract is a runtime property of the model, not of this module.

**`test_face_order.py` — one-line fix**, as diagnosed: `api.parse_stl_bytes` →
`stl_io.parse_stl_bytes` (the parser moved out of `api_core` into `stl_io`, which `api_core` imports
at `:20`). It now imports `stl_io` directly rather than dragging FastAPI, torch and the whole
geometry engine in to check a struct. Its substance — server face index *i* must equal STL triangle
*i*, so `THREE.STLLoader`'s non-indexed geometry lines up — was always valid.

### ⚠ 3 entries still contain ZERO assertions and can never fail
`test_interproximal.py`, `test_auto_color.py`, `test_incisal_edge.py` are characterization scripts:
they print measurements and exit 0, so the runner reports PASS regardless of what they measured.
**RESOLVED 2026-09-16 — every suite entry now carries enforcing assertions.** The paragraph below
is kept because the lesson is worth more than the defect: a script that prints and exits 0 reports
PASS whatever it measured. It no longer describes this repo.

~~This matters more now that the suite reads 24/24 — treat it as 21 enforcing tests plus 3
reports, and do not read their PASS as evidence of anything.~~

### ~~3 entries contain ZERO assertions and can never fail~~ — FIXED
~~`test_interproximal.py`, `test_auto_color.py`, `test_incisal_edge.py` are characterization
scripts. They print measurements and exit 0, so they report PASS regardless.~~

All three were hardened on 2026-09-16 and now assert: 20 interproximal configurations
non-negative with the separation ordering holding, the colour buffer `(22500, 3)` float32 in
`[0,1]` (a 0-255 palette slipping in clips silently to white in WebGL), and 91% of identified
incisal points inside the occlusal 15% band. **Every entry in the suite can now fail.**

### Highest-value tests
| File | Lines | Guards |
|---|---|---|
| `test_cut_endpoint.py` | 656 | The `/cut` + `/kinematics` contract: 409 without a plane, pivot in bone, displayed cup == printed cup, exported STL survives a re-read |
| `test_cast_base.py` | 605 | **No face spans the horseshoe tongue opening**; extruded solid closed; rigid-transform congruence |
| `test_socket_cup.py` | 456 | The socket is a carved cup with depth, not a flat lid; inset-fold guard fires |
| `test_kinematics_frame.py` | 382 | The tetherball fix + the JS golden-matrix pin |
| `test_occlusal_collision.py` | 245 | Penetration signed; interference **reported, not blocked**; `checked: false` ≠ "no interference" |

### Cross-language pinning
`verify-kinematics.mjs` and `test_kinematics_frame.py:317-362` hard-code **the same three golden
4×4 matrices**, the same frame, and the same `C_res = [1.5, -2.0, -7.0]`. Python asserts
`cg.kinematic_matrix` still reproduces them; JS asserts `deltaFromClinical` does, after converting
three.js's column-major `elements` to row-major. Agreement required at **< 1e-12** on both sides.
This locks composition order (`Rx·Ry·Rz`, not `ZYX`), pivot handling, and row/column-major
convention. **Update both files together, never one alone** — otherwise the browser and the exported
STL drift apart silently.
⚠ `verify-kinematics.mjs` is in **no npm script and no CI**, so half the pin runs only when a human
remembers `node verify-kinematics.mjs`.

---

## 8. RECOMMENDED NEXT ACTIONS

1. ~~Fix `App.jsx:1104` and `:1123`~~ — **DONE 2026-09-15** (§4.9).
2. ~~Fix `requirements.txt`~~ — **DONE 2026-09-15**, and widened to include `trimesh` and `open3d`,
   which the original diagnosis missed (§4.10).
3. ~~Fix `test_face_order.py` and rewrite `test_api_core.py`~~ — **DONE 2026-09-15**. Suite is
   24/24 (§7). The remaining test debt is the **3 zero-assertion entries**, which should either
   grow assertions or be moved out of the suite and run as the reports they are.
4. **Delete or quarantine the dead layer** (§0) — especially `HANDOVER.md`, whose architecture
   section is false, and `frontend_src/App.jsx`.
5. **Wire `verify-kinematics.mjs` into an npm `test` script**, and add any CI at all.
6. **Debounce `/wand/threshold`** (`App.jsx:1375`).
7. Then the real clinical gaps: a cancellable segmentation, `/teeth` on load, and an upper-arch
   scan so the antagonist check can be judged against real occlusion.

---

*Generated by a file-by-file scan. Every `file:line` citation in this document was verified to
resolve at generation time. Where this document and the originating brief disagree, §6 states which
line settles it.*
