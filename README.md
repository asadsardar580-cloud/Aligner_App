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
api_core.py                   THE live FastAPI app — 14 routes
        |
        +-- core_geometry.py      5,638 lines. Pure NumPy/SciPy. All geometry
        |                         and kinematics. No UI imports, fully headless.
        +-- arch_frame.py         Occlusal reference basis (u_occ/u_sag/u_tra)
        +-- session_store.py      In-memory, TTL, PHI-conscious
        +-- stl_io.py             Binary STL reader with exact vertex welding
        +-- jaw_naming.py         FDI/jaw verification
        +-- cut_guard.py          Pre-cut heuristics
        +-- tgn_bridge.py ------> ToothGroupNetwork/   (vendored, PyTorch, CPU)
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

Full detail, including the measured regressions behind each one, is in **`CLAUDE.md`** — the
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
3. **Segment** — AI (ToothGroupNetwork, ~4 minutes on a full arch) or manual wand selection.
4. **Select** a tooth with the wand or brush, set mesial/distal points.
5. **Cut** — extraction is an index-buffer rewrite; the arch mesh itself is never rebuilt.
6. **Move** — 3D gizmo or the numeric sidebar. Values are absolute from T0, not nudges.
7. **Stage** — scrub the timeline from T0 to the planned setup.
8. **Export** — `/export` for the planned setup, `/export/stages` for fused manufacturing solids.

## Testing

```powershell
python -m compileall .                  # syntax, whole tree
python check_structure.py               # undefined names, without importing (61 files)
python run_all_tests.py                 # canonical runner — 24 entries
python -m pytest -q                     # runs alongside; both must pass
node frontend/verify-kinematics.mjs     # cross-language kinematics pin

cd frontend
npm run lint
npm run build
npm run smoke                           # SSR render of <App/>; catches TDZ crashes a build cannot
```

**`npm run smoke` earns its place.** A `vite build` succeeds on code that throws during render — a
dependency array referencing a `const` declared further down the component crashed the app to a
white screen while building cleanly. The smoke test renders `<App/>` in node and catches exactly
that.

**Read `24/24` with a caveat:** three entries (`test_interproximal.py`, `test_auto_color.py`,
`test_incisal_edge.py`) contain no assertions. They are characterization scripts that print
measurements and exit 0, so they report PASS regardless. The suite is 21 enforcing tests plus 3
reports.

## Current limitations

- **Segmentation is not solved.** No reproducible mm-accuracy figure exists for the AI path; the
  numbers quoted in older documents describe the *classical* segmenter, which `/segment` does not
  use. Do not treat any segmentation output as verified.
- **Segmentation cannot be cancelled** — ~236 s on a real scan, and `asyncio.to_thread` gives no
  cancellation point inside the model.
- **A browser refresh loses the case.** The server holds nearly everything, but several keys are not
  exposed by any endpoint and the session id lives only in React state. Sessions are in-memory with
  a sliding 1-hour TTL and a 4-session cap, so a server restart is unrecoverable by design.
- **The antagonist collision check has never run against a real opposing arch.** `checked: false`
  means "not checked", never "no interference".
- **`cut_guard.check_crown` is bypassed** — it computes its metrics and returns `ok=True`
  unconditionally, so the `/cut` refusal that depends on it is unreachable.
- No authentication, TLS, authorization or audit logging. **This is a localhost development
  prototype**; do not expose it to a network.

## PHI

Scans are held in memory only and never written to disk. Sessions expire on a timer. Original
filenames are stored nowhere, because filenames routinely carry patient names. Exports run on
127.0.0.1 and write only to the local `exports/` directory, with no patient name in any filename.
`.gitignore` excludes every scan, mesh and derived label file from version control.
