# Clinical Micro-Planner — Project Aligner

A clear aligner CAD workstation. Load an intraoral arch scan, establish an occlusal reference,
isolate a tooth, move it with C_res-pivoted biomechanics, stage the movement, and export
print-ready STLs.

> **Status: clinical-planning prototype.** It is not a validated medical device and has not
> undergone clinical verification, regulatory review, or evaluation against a real patient dataset.
> Passing unit tests means the geometry is self-consistent — it does not mean a movement is
> clinically correct. Treat every threshold in the UI as a software heuristic unless it is
> explicitly labelled otherwise.

---

## Architecture

```
frontend/src/App.jsx          React 19 + three.js 0.185, built by Vite
        |   HTTP to 127.0.0.1:8000, CORS-locked to :5173
        v
api_core.py                   THE live FastAPI app — 37 routes
        |
        +-- core_geometry.py      5,638 lines. Pure NumPy/SciPy. All geometry
        |                         and kinematics. No UI imports, fully headless.
        +-- arch_frame.py         Occlusal reference basis (u_occ/u_sag/u_tra)
        +-- session_store.py      In-memory, TTL, PHI-conscious
        +-- stl_io.py             Binary STL reader with exact vertex welding
        +-- jaw_naming.py         FDI/jaw verification
        +-- cut_guard.py          Pre-cut heuristics
        +-- segmentation_providers.py   The model registry; /segment?provider=
        +-- tgn_bridge.py --------> ToothGroupNetwork/  (vendored, PyTorch, CPU)
        +-- crosstooth_bridge.py -> CrossTooth/         (vendored, PyTorch, CPU)
        +-- tooth_segmentation/   Classical segmentation package
```

`app_ui.py` is a **legacy PyQt6 + PyVista desktop shell** over the same geometry engine. It still
runs, but it is not the product path.

**`_archive/` contains dead code and is not part of the application.** See `_archive/README.md`.

## The invariants that are not negotiable

1. **Scanner coordinates are sacred.** Raw STL vertices are never rotated, re-centred or rescaled.
   Inter-arch bite registration depends on both arches sharing raw scanner space, and every vertex
   id already sent to the browser must keep its meaning. "Up" comes from the occlusal reference
   frame stored as session metadata — never by moving geometry.
2. **Teeth pivot about the Centre of Resistance**, ~10-13 mm down the root, not about their
   geometric centre.
3. **Anatomical axes:** tip = about the buccolingual axis, torque = about the mesiodistal axis,
   rotation = about the apical axis.
4. **Matrix conventions:** three.js is column-major, NumPy is row-major. `verify-kinematics.mjs`
   pins both sides to the same golden matrices — update it and `test_kinematics_frame.py` together.
5. **Staging interpolates the prescription, never the 4×4.** A component-wise lerp of the matrix is
   not a rotation; measured, it reaches 1.68e-2 orthogonality error against 4.44e-16 for the
   parameter path.

Full detail, including the measured regressions behind each one, is in **`docs/HISTORY_CLAUDE_2026-09.md`** — the
authoritative architecture document.

---

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # pytest, for development

cd frontend
npm install
```

Python 3.12 is what this is developed against; 3.10 is the floor (bare PEP-604 unions are used
without `from __future__ import annotations`).

> **MeshLib licence — read before any commercial use.** `meshlib` (pinned in `requirements.txt`)
> makes the printable model: `print_solid.py` voxel-solidifies the closed cast so the exported STL
> is watertight and free of self-intersections. MeshLib is **free for non-commercial and
> educational use only**; commercial use needs a paid licence from its vendor. It is the one
> dependency outside the project's MIT / BSD / Apache-2.0 / MPL-2.0 rule, added as an explicit
> exception by the clinical owner. Settle the licence before this app is used commercially.

## Run

Two terminals, or double-click each script:

```
start_backend.bat     ->  uvicorn api_core:app on http://127.0.0.1:8000
start_frontend.bat    ->  vite dev server on   http://localhost:5173
```

`start_backend.bat` prefers `.venv\Scripts\python.exe`, preflights the required imports, the AI
checkpoint and the port, and tells you which process holds :8000 if it is taken. A bare
`python -m uvicorn api_core:app` will fail unless the venv is active — the system Python has none
of the dependencies.

The API answers immediately; the segmentation model loads in a background thread. Watch the
sidebar's connection badge, or poll `/api/ai/status`.

## Workflow

1. **Load** an arch STL (upper and/or lower).
2. **Occlusal reference** — click left molar cusp, right molar cusp, anterior midline. Required:
   `/cut` refuses with 409 without it, because C_res would otherwise be extrapolated along a
   guessed axis.
3. **Segment** — AI or manual wand selection. Two models are installed and the
   picker beside the button chooses per run: **ToothGroupNetwork** (the default,
   255 s on the real arch) and **CrossTooth** (13 s). Neither is validated — see
   *Current limitations*.
4. **Select** a tooth with the wand or brush, set mesial/distal points.
5. **Cut** — extraction is an index-buffer rewrite; the arch mesh itself is never rebuilt.
6. **Move** — 3D gizmo or the numeric sidebar. Values are absolute from T0, not nudges.
7. **Stage** — scrub the timeline from T0 to the planned setup.
8. **Export** — `/export` for the planned setup, `/export/stages` for fused manufacturing solids.

## Testing

```powershell
python -m compileall .                  # syntax, whole tree
python check_structure.py               # undefined names, without importing (117 files)
python run_all_tests.py                 # canonical runner — 52 entries; PASS / SKIP(77) / FAIL
python verify_pointops.py               # the CPU pointops shim, against upstream
python -m pytest -q                     # runs alongside; both must pass
python benchmark_providers.py           # every segmentation model over one scan
python bench_signed_distance.py         # scores the signed-distance method
python real_scan_regression.py          # the real scan, end to end
node frontend/verify-kinematics.mjs     # cross-language kinematics pin
node frontend/verify-framing.mjs        # occlusal framing, rigid-motion invariant
node frontend/verify-shadowrig.mjs      # shadow rig arithmetic
node frontend/verify-palette.mjs        # 32 distinct FDI colours, measured in Lab

cd frontend
npm run lint
npm run build
npm run smoke                           # SSR render of <App/>; catches TDZ crashes a build cannot
npx playwright test                     # browser E2E against the production build
```

**`npm run smoke` earns its place.** A `vite build` succeeds on code that throws during render — a
dependency array referencing a `const` declared further down the component crashed the app to a
white screen while building cleanly. The smoke test renders `<App/>` in node and catches exactly
that.

**The three assertion-free entries were fixed on 2026-09-16.** `test_interproximal.py`,
`test_auto_color.py` and `test_incisal_edge.py` used to print measurements and exit 0, reporting
PASS regardless of what they measured. Every suite entry now carries enforcing assertions, so
every entry can fail.

## Current limitations

- **Segmentation is not solved.** No reproducible mm-accuracy figure exists for the AI path; the
  numbers quoted in older documents describe the *classical* segmenter, which `/segment` does not
  use. Do not treat any segmentation output as verified. What IS now checked is whether the labels
  belong to the mesh at all — `segmentation_diagnostics` measures per-tooth bounding box, counts
  and disconnected components, because a label array read against the wrong vertex ordering scores
  perfectly on every accuracy metric including IoU. On the real scan that check moved the median
  per-tooth box from 50.71 mm to 13.97 mm (docs/HISTORY_CLAUDE_2026-09.md §24.3).
- **Two models are installed and NEITHER is validated.** `benchmark_providers.py` measures
  geometry (a tooth is one connected lump of a plausible size) and inter-model agreement, and it
  deliberately reports no accuracy figure, because that needs an independent annotation this
  repository does not have. On the real lower arch CrossTooth is 19x faster and produces more
  plausible per-tooth geometry — 16 teeth against 11, worst largest-component fraction 0.988
  against 0.523, largest tooth box 17.8 mm against 51.0 mm — and **the default was not changed on
  that basis**, because "better geometry" is not "correct teeth". The two models agree on the
  quadrant convention (0.4884 as mapped against 0.0559 mirrored) and disagree by one tooth along
  the 3x quadrant; which of them is the shifted one is not established (docs/HISTORY_CLAUDE_2026-09.md §25).
- **Real-scan manufacturing is NOT VERIFIED end to end.** Seven real teeth now cut successfully,
  one of seven builds a complete local interface, and staging refuses by name with measured
  numbers. No manufacturing gate result anywhere is based on the real scan (docs/HISTORY_CLAUDE_2026-09.md §24.7).
- **The workspace redesign is not done.** The FDI colour system, the tooth legend, the design
  tokens and the focus and reduced-motion rules shipped; the top bar, workflow rail and context
  panel did not, and `App.jsx` is still one large file (docs/HISTORY_CLAUDE_2026-09.md §24.9).
- **Segmentation cannot be cancelled** — ~236 s on a real scan, and `asyncio.to_thread` gives no
  cancellation point inside the model.
- **A browser refresh now restores the case**, since the hydration endpoints landed: crowns, poses,
  the occlusal frame and the labels come back, and only the session id lives in `localStorage`.
  Sessions are still in-memory with a sliding 1-hour TTL and a 4-session cap, so a **server restart**
  is unrecoverable by design and the persistent re-upload banner says so.
- **The antagonist collision check has never run against a real opposing arch.** `checked: false`
  means "not checked", never "no interference".
- **`cut_guard.check_crown` measures but does not gate.** The unconditional `ok=True` was removed,
  but `MIN_COMPACTNESS` and `MIN_RIM_CONCAVITY` have never been measured against real cuts, so the
  result is reported as `crown_advisory` rather than used as a refusal. A threshold becomes a
  refusal only with numbers behind it.
- No authentication, TLS, authorization or audit logging. **This is a localhost development
  prototype**; do not expose it to a network.

## PHI

Scans are held in memory only and never written to disk. Sessions expire on a timer. Original
filenames are stored nowhere, because filenames routinely carry patient names. Exports run on
127.0.0.1 and write only to the local `exports/` directory, with no patient name in any filename.
`.gitignore` excludes every scan, mesh and derived label file from version control.

## Manufacturing architecture (authoritative, 2026-09-20)

* Raw scanner coordinates remain **immutable** — never translated, rotated,
  rescaled or re-centred.
* Tooth movement is **rigid about C_res**; the crown in every stage is a rigid
  transform of the extracted T0 crown.
* The **live viewport cast is static**. Nothing in the viewport reconstructs.
* The **old socket is restored** flush at its original position.
* Each stage reconstructs a **bounded local target-position interface** around
  the transformed cervical rim: a cavity where the crown penetrates, an
  emergence ramp where the rim has lifted clear.
* **`root_length_mm` is NOT manufacturing plug depth.** It is used only for
  C_res estimation and the wireframe virtual root. Proven with the stage matrix
  frozen.
* Boolean order is **cast − crown-derived local clearance → union rigid crown
  + transition collar**. The clearance is an exact Minkowski dilation of the
  transformed crown and is emitted only when the subtraction leaves the cast as
  one body; a withheld tool is recorded with its reason.
* The connector is a **bounded transition collar that ENCLOSES the crown's
  cervical crease**, with both rings placed by measured signed distance against
  the actual crown and the actual cast. That is what stops the fused solid
  touching itself — see docs/HISTORY_CLAUDE_2026-09.md §23.
* The **final STL is validated after write and readback**, simulating a
  downstream reader's weld.
* **TWO VERDICTS, and they are not the same claim.**
  `validate_printable_stl` returns **PASSES / FAILS BOOLEAN/TOPOLOGY
  REGRESSION** on the written bytes — finite coordinates, zero open edges, zero
  non-manifold edges, positive volume, one connected component, consistent
  winding — and carries a `gate_scope` naming what it does not cover.
  **`manufacturing.aggregate_print_gate` is the only thing entitled to say
  PRINT READY**, and it adds: no self-touching boundary, synthetic exposure
  within bound, no exposed clearance wall, two-sided unaffected-cast fidelity,
  reconstruction inside the allowed envelope, interface continuity, transition
  quality, old-site quality, crown rigidity, gingival bridge, root-length
  independence and clinical consistency. It is a pure function of the stage
  record, so a MISSING measurement fails it exactly as a bad one does.
  `POST /export/final` enforces it and returns no file on failure.

Wording note: these are **engineering validation gates for a prototype**. They
are not a claim of clinical validation, and **no real de-identified scan has
been run through them** — `real_scan_regression.py` reaches tooth selection on
`case_lower.stl` and stops there, for reasons it measures and prints.

See `MANUFACTURING_RECONSTRUCTION_CHANGELOG.md` for the full record, including
what is **not** yet fixed.
