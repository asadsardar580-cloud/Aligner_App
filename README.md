# Clinical Micro-Planner

Single-step clear aligner planning: load an arch scan, isolate one tooth,
move it with C_res-pivoted biomechanics, export print-ready STLs.

---

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python run_all_tests.py     # confirm the install is sound
python app_ui.py            # the actual application
```

---

## What is tested and what is not

This distinction matters more than usual here, because the two halves were
built under different conditions.

**Verified** — `core_geometry.py` and everything in `tooth_segmentation/`.
Pure NumPy/SciPy, no display required, covered by the suite in
`run_all_tests.py`. If a result is geometrically wrong, the fault is almost
certainly here, and you can reproduce it headlessly.

**Not verified by automated test** — `app_ui.py` and `server.py`. PyQt6, VTK
and Trimesh could not be installed in the environment where these were
written, so no line of the GUI has ever been executed by their author. They
are written carefully and instrumented heavily, but expect first-run
friction there rather than in the geometry.

`check_structure.py` closes part of that gap: it parses files with `ast` to
catch undefined names without importing them, which `py_compile` cannot do.
Run it after any edit to `app_ui.py`.

---

## Using the application

1. **Load Maxillary STL** — coordinates are never re-centred or rescaled, so
   inter-arch bite registration is preserved exactly as exported.
2. Wait for `ready to segment` in the status bar. Curvature analysis runs on
   a background thread (~2.6s on a 360k-vertex scan).
3. **Segment Tooth**, then **click once on the tooth**. A blue preview
   appears immediately.
4. **Drag "Selection spread"** and watch the preview grow or shrink. The
   cervical groove acts as a barrier, so a wide range of values all stop at
   the gumline. Status bar shows the selected fraction — **a single crown is
   2-10% of the arch.**
5. Click the **mesial** and **distal** contacts (both visible from one
   labial view), then **Confirm & Cut**.
6. Move the sliders. The tooth pivots about the white C_res sphere, not
   about its own centre.

Log file: `~/micro_planner.log` — records every press, release, cell ID and
traceback.

---

## Layout

```
app_ui.py              PyQt6 + PyVista application
core_geometry.py       All geometry and kinematics. No UI imports.
check_structure.py     AST-based undefined-name checker
diagnose.py            Probes which VTK interactor accepts observers
server.py              Optional FastAPI backend (see transport note below)
run_all_tests.py       Runs everything

tooth_segmentation/    Automatic multi-tooth segmentation (in progress)
    config.py          All tunable weights, nothing hard-coded at call sites
    models.py          PreprocessReport, ToothCandidate, ArchFrame
    mesh_preprocessor.py
    normals.py
    curvature.py
    arch_geometry.py   PCA orientation, no axis-alignment assumption
    label_adapter.py   Bridges any pretrained model to the tested pipeline
```

---

## Findings worth keeping

**Tip and torque were swapped.** Tip rotated about the mesiodistal axis when
it should rotate about buccolingual. Invisible on screen — the tooth moved
smoothly and looked correct while pivoting about the wrong anatomical axis.

**Delaunay capping can produce non-manifold meshes.** On a non-planar margin
loop it creates chords duplicating existing mesh edges, giving them four
faces. The mesh then has no open boundary — it passes a naive hole check —
while being unusable for boolean export. Caps are now validated, with a
centroid fan as the structurally safe fallback.

**Clicking a margin circuit is impossible.** A crown's margin wraps 360
degrees but a camera sees one side; the picker returns the frontmost visible
cell, so a "lingual" click silently lands on the labial surface. Replaced
with region growing, which spreads through mesh connectivity and reaches
surfaces the camera cannot see.

**Deriving the buccolingual axis from the FA point tilted C_res by 38.7
degrees** once that point moved to the margin. Measuring between opposing
landmarks instead brings it to 0.0.

**VTK's LeftButtonReleaseEvent never fires** on PyQt6 6.11 / VTK 9.6.2 /
pyvista 0.48.4. The interactor style claims the left button on press for
camera rotation and consumes the release. Press comes from VTK, release from
a Qt event filter.

**Vectorisation took segmentation from ~8.5s to 0.46s** on a 360k-vertex
mesh. `np.unique(axis=0)` alone accounted for 1.43s across three call sites;
actual Dijkstra was 0.02s and was never the bottleneck.

---

## Two decisions to make deliberately

**Client-server.** Measured: a full-arch scan is 34.2MB, and a naive
per-click round trip costs 78s on clinic DSL against 0.46s of local compute
— 170x slower. `transport_bench.py` reproduces this. If you go remote
anyway, upload once per session and send only the click coordinates
thereafter. That is what `server.py` implements.

**Patient data.** Intraoral scans are PHI. A desktop tool keeps them on the
clinician's machine; the moment they cross a network this becomes a system
requiring encryption, retention limits, audit logging and a BAA with the
host. `server.py` is a prototype and is production-ready on none of those
axes. Also worth keeping the project and any scans out of cloud-synced
folders such as OneDrive.

---

## Not implemented, deliberately

**AI auto-segmentation.** Needs trained weights and a GPU. `label_adapter.py`
is the integration point: MeshSegNet, ToothGroupNetwork and
DilatedToothSegNet all emit per-face labels, and everything downstream of
that is already tested. Check each repository's LICENSE and the underlying
dataset terms before integrating — academic dental models frequently carry
non-commercial restrictions, and weights inherit dataset terms.

**Clinical verdicts.** Staging returns numbers, never stoplights or
recommendations. Every result is computationally generated and requires
clinician verification. Nothing here claims clinical accuracy.
