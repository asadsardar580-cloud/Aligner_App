# Clinical Micro-Planner — Handover

Single-step clear aligner planning. Load an intraoral arch scan, isolate one
tooth, move it with biomechanically correct kinematics, export print-ready
STLs. Built for anterior relapse and mid-course correction, not full
treatment planning.

---

## 1. WHAT THIS IS

A three-layer system:

```
core_geometry.py          Pure NumPy/SciPy. All geometry and kinematics.
                          No UI imports. Fully tested headlessly.
        |
        +-- app_ui.py     PyQt6 + PyVista desktop app
        |
        +-- api_core.py   Backend logic (no FastAPI import, so it is testable)
            server.py     FastAPI routes over api_core
            App.jsx       React + Three.js frontend
        |
        +-- tooth_segmentation/   Whole-arch automatic segmentation package
```

The decoupling is load-bearing. `core_geometry.py` runs with only numpy and
scipy, so it can be tested without a display, a GPU, or a browser. Every
parameter that matters lives there.

---

## 2. WHAT WORKS (measured, not asserted)

Run `python run_all_tests.py` — 12 suites, all passing.

**Geometry core**
- Concavity/curvature field, robust to scan noise (see §4)
- Barrier-weighted geodesic region growing ("magic wand")
- Watertight capping with Delaunay + centroid-fan fallback
- Orthonormal anatomical frames, C_res, 4x4 kinematics
- Whole-arch watershed segmentation

**Performance** (360k-vertex mesh)
- Segmentation: 0.46s, down from ~8.5s before vectorisation
- Whole-arch precompute at upload: ~8s on a real 220k-face scan
- Click-to-tooth lookup after precompute: <1ms

**Backend**
- Session model: upload the scan once, then send coordinates only
- `/kinematics` returns a 674-byte 4x4 matrix, never geometry

**Validated on real patient data** (220,196-face maxillary scan)
- Scan quality: 0 degenerate faces, 0 debris, 0 non-manifold edges,
  1 component, already welded. Conditioning found nothing to clean.
- Curvature signal stronger than any synthetic fixture: 27.8x barrier at p95
- Tooth detection after fixes: 14 teeth (matches clinical ground truth of
  7-7, third molars excluded), largest region 7.8% of arch, size ratio 5.6x

---

## 3. WHAT IS NOT DONE

- **Boolean STL export.** `/segment` produces watertight crown and base
  meshes, but the socket-carving CSG subtraction and nested export are not
  wired up. Trimesh + manifold3d were never installed or tested.
- **IPR and staging in the UI.** The arithmetic exists (`staging_estimate`)
  and deliberately returns numbers, never clinical verdicts. Not surfaced.
- **Gingival morphing (ARAP).** Deliberately benched. The socket stays static
  when a tooth moves.
- **Attachments.** Architectural hooks only.
- **FDI numbering.** Teeth are `tooth_01..N` in arch order. No 11/12/13.
- **AI segmentation.** `tooth_segmentation/label_adapter.py` is the
  integration point and is tested with synthetic labels: it lifts labels from
  a decimated mesh (what MeshSegNet-class models infer on) back to full
  resolution at 98.7-99.5% accuracy, cleans speckle, and emits independent
  watertight meshes. No model, no weights. `/ai/propose-selection` returns
  501 by design rather than faking it.
- **Auth, TLS, audit logging.** `server.py` is a localhost prototype.

---

## 4. BUGS FOUND AND WHY THEY MATTER

These are the hard-won parts. Each cost real time and several were invisible
on screen.

**Tip and torque were swapped.** Tip rotated about the mesiodistal axis when
it should rotate about buccolingual. The tooth moved smoothly and looked
correct while pivoting about the wrong anatomical axis.

**Delaunay capping produced non-manifold meshes.** On a non-planar margin
loop it creates chords duplicating existing mesh edges, giving them four
faces. The mesh then has no open boundary — it passes a naive hole check —
while being unusable for boolean export. Caps are now validated with a
centroid fan as the structurally safe fallback.

**Clicking a margin circuit is physically impossible.** A crown's cervical
margin wraps 360 degrees but a camera sees one side, and the picker returns
the frontmost visible cell — so a "lingual" click silently lands on the
labial surface. Replaced with region growing, which spreads through mesh
connectivity and reaches surfaces the camera cannot see.

**Deriving the buccolingual axis from the FA point tilted C_res by 38.7
degrees** once that landmark moved to the margin. Measuring between opposing
landmarks instead gives 0.0.

**VTK's LeftButtonReleaseEvent never fires** on PyQt6 6.11 / VTK 9.6.2 /
pyvista 0.48.4. The interactor style claims the left button on press for
camera rotation and consumes the release. Press comes from VTK, release from
a Qt event filter.

**Max-normalized curvature collapses on noisy scans.** Dividing by the global
maximum lets one scan artifact set the scale for all real anatomy. Twelve
outlier vertices — 0.006% of a mesh — cut the sulcus signal 4x and doubled
selection bleed. `boundary_field` normalizes by percentile and smooths over a
fixed physical radius; its signal was identical on clean and noisy meshes.

**Shift+drag is the browser's own selection gesture.** Without preventDefault
and pointer capture the browser starts a native drag and pointermove never
reaches the handler — which is why brush ADD appeared dead while ERASE (Alt)
worked.

**Raycasting only the active mesh causes cross-arch bleed.** It sounds like
the safe, strict thing to do and is exactly wrong: with both arches in
occlusion a ray aimed at the mandible passes through and strikes whichever
arch is active behind it. Test all arches, take the nearest hit.

**THE BIG ONE — the occlusal axis pointed at the palate.** Seeding assumed
cusp tips are the highest points. On a maxillary cast that is false: the
palatal vault sits ~8mm ABOVE the occlusal plane (measured: interior median
height 14.5mm, perimeter 6.3mm). The watershed seeded the vault, shattered it
into 94-face fragments, and merged whole teeth into 10,000-face blobs — size
ratios of 80x. Flipping the axis alone brought that to 2.1x.

The fix took two attempts, and the second is the lesson. A
perimeter-vs-interior rule fixed the real scan and immediately broke a
synthetic fixture whose flat base extends outward past the crowns. The rule
that works for both depends on neither height nor position: **cusps and
incisal edges are sharply convex; palate, gingiva and cast base are smooth.**
On the real scan the occlusal half measures a convex fraction of 0.389
against 0.035.

---

## 4b. SEGMENTATION: WHERE IT ACTUALLY STANDS

Validated against a real maxilla and mandible with clinician ground truth
(14 teeth, 7-7, third molars excluded).

    MAXILLA   14 regions, 11 of 14 crowns within 2mm of Wheeler  (79%)
    MANDIBLE  16 regions, 5 of 14                                 (36%)

The maxilla is usable with review. The mandible is not solved.

Pipeline, in order: barrier-weighted watershed from cusp seeds -> merge by
cusp distance with a position-scaled window -> two-pass merge with Wheeler
width ceilings -> alternating split/merge sculpting.

WHY THE MANDIBLE RESISTS. It carries both anatomical extremes in one arch:
incisors at 5.0-5.5mm, the narrowest teeth in the mouth, and five-cusped
first molars at 11mm. Adjacent incisors sit closer together than the cusps
of a single molar, so no distance threshold separates "two teeth" from "two
cusps" across the whole arch. The position-scaled merge window (4.5mm
anterior to 8.0mm posterior) narrows the gap but does not close it.

THINGS THAT LOOK LIKE PROGRESS AND ARE NOT:
  - Matching the tooth COUNT proves nothing. An early version produced
    exactly 14 regions in which the second premolar was the largest tooth in
    the arch and a first molar the smallest.
  - Face count is a biologically invalid size proxy. A fissured molar
    carries far more triangles per mm2 than a smooth incisor. Use
    measure_mesiodistal_width, which measures at the occlusal 40% where
    interproximal contacts actually sit.
  - validate_against_anatomy returns `reliable`. When the region count
    differs from the dentition, identify_dentition numbers regions
    sequentially and labels real teeth as third molars, so pass_rate is
    computed over a partial, shifted set. Check the flag before quoting it.

OPEN PROBLEM: THE SCULPTING LOOP DOES NOT CONVERGE. Splitting and merging
fight each other -- a freshly cleaved boundary is by construction the weakest
available, so the next merge re-fuses it. Two guards exist and are not
enough:
  - `forbidden` pairs in merge_weak_regions protect split boundaries, but go
    stale when a protected region later merges into a third and changes
    label. Union-find label tracking would fix this.
  - `min_cleave_concavity` in split_fused_region refuses cuts that do not
    pass through a concave valley. This fixed a real bug -- the splitter was
    bisecting healthy crowns, since half a tooth passes the sliver guard --
    and took the mandible from 12 oscillating rounds to converging in 2. It
    now stops at 16 regions instead.

The maxilla still runs all 12 rounds without settling.

## 5. WHAT WE FAILED AT

**Synthetic fixtures hid the real bug for the entire project.** Every
parameter was tuned against invented geometry. The palate problem could not
appear in a fixture that had no palate. The first real scan exposed it in
minutes. **Get real data early.**

**Feature sprawl outran reliability.** Three selection tools — auto-colour,
magic wand, brush — were built before any one of them worked on real data.
Each brought its own failure modes and the clinician ended up fighting the
software instead of using it. Auto-colour still does not earn its place: it
colours but produces no cut, and over-segments molars.

**Region merging is still not solved.** Raw watershed gives 24 regions at
3.1x size ratio with zero oversized blobs — clean, but over-segmented.
Merging across weak boundaries CHAINS: each merge creates a bigger region
with more weak boundaries, snowballing into one 15.3% blob. A size cap at 8%
contains it (14 teeth, largest 7.8%, ratio 5.6x) but 5.6x is still high, and
the cap is a patch rather than a principle.

**The desktop app was never verified end-to-end by its author.** PyQt6, VTK
and Trimesh could not be installed in the environment it was written in.
Every GUI bug was diagnosed remotely from logs and screenshots. `app_ui.py`
and `server.py` carry inline markers where behaviour is unverified.

**Two self-inflicted wounds worth avoiding.** A patch that replaced a line
range silently deleted the `ReleaseFilter` class while three references
remained — `py_compile` passed because syntax was fine. `check_structure.py`
now catches undefined names via AST without importing. Separately, files
downloaded through a browser arrived named `Label adapter · PY` and a copy
chain appended a filename into a source file, producing a SyntaxError that
broke the whole package.

---

## 6. THE PLAN

**Immediate**
1. Make the sculpting loop converge. Track labels through merges with
   union-find so `forbidden` pairs survive relabelling, and stop the loop on
   a repeated state rather than a fixed round count.
2. Decide whether geometry alone is the right tool for the mandible. Three
   independent approaches have now plateaued around 36-79%. A pretrained
   mesh-segmentation model behind label_adapter.py may be the honest answer
   for instance separation, with this geometry pipeline doing what it does
   well -- watertight cutting, frames, C_res, kinematics -- on top of its
   labels.
3. Wire the boolean export: dilate the moved tooth 0.15mm, CSG-subtract to
   carve the socket, emit two STLs on one shared origin. This is independent
   of segmentation quality and unblocks the actual clinical workflow, since
   the manual magic-wand and brush already produce usable single-tooth
   selections.

**Then**
4. Surface IPR (penetration projected onto the mesiodistal axis, so "into its
   own socket" is not confused with "into the neighbour") and staging.
5. FDI numbering, once instance segmentation is reliable.
6. Consider a pretrained model (MeshSegNet, ToothGroupNetwork,
   DilatedToothSegNet) behind `label_adapter.py`. **Check each repository's
   LICENSE and the underlying dataset terms first** — academic dental models
   frequently carry non-commercial restrictions and weights inherit dataset
   terms.

---

## 7. HARD CONSTRAINTS — DO NOT VIOLATE

**Never re-centre or re-scale a loaded arch.** Inter-arch bite registration
depends on raw scanner coordinates. Three.js tutorials centre models by
default; doing so silently offsets every click sent to the backend.

**Never smooth the geometry used for cutting or export.** Laplacian or Taubin
smoothing makes the cast prettier and the curvature field cleaner, and it
moves the enamel surface the aligner grips and the printer reproduces.
Sub-0.1mm accuracy is the point of the appliance. Conditioning may weld,
delete debris, and cap holes — `test_conditioning.py` asserts every retained
vertex is bit-identical to the upload. Smooth the ANALYSIS field instead;
`boundary_field` already does.

**Reconstruct where data is missing; never overwrite where data exists.**
Hole filling and closing scanner gaps are legitimate. Replacing a worn
incisal edge with an idealised one fabricates anatomy that is not in the
patient's mouth.

**No clinical verdicts.** Staging returns numbers, never stoplights or
recommendations. IPR readouts are raw measurements. Every result is
computationally generated and requires clinician verification. Nothing here
claims clinical accuracy.

**Patient data is PHI.** Intraoral scans are health information. A desktop
tool keeps them local; the moment they cross a network this becomes a system
requiring encryption in transit and at rest, retention limits, audit logging,
and a business-associate agreement with the host. `server.py` satisfies none
of those. Also: strip patient names from filenames — filenames travel with
files in ways contents do not. Keep scans out of cloud-synced folders.

**Measured transport numbers, if client-server is revisited.** A full-arch
scan is 34.2MB. A naive per-click round trip costs 78s on clinic DSL against
0.46s of local compute — 170x slower. `transport_bench.py` reproduces this.
The session model (upload once, send coordinates after) is what makes the
remote architecture viable at all.

---

## 8. FILE MAP

```
core_geometry.py          ~1900 lines. Everything geometric. Start here.
api_core.py               Backend logic, no FastAPI import
server.py                 FastAPI routes
app_ui.py                 PyQt6 desktop app (unverified by author)
frontend_src/App.jsx      React + Three.js client
check_structure.py        AST undefined-name checker (py_compile misses these)
scan_report.py            Local scan calibration report, statistics only
diagnose.py               Probes which VTK interactor accepts observers
run_all_tests.py          Runs everything

tooth_segmentation/
    config.py             All tunable weights
    models.py             PreprocessReport, ToothCandidate, ArchFrame
    mesh_preprocessor.py  Non-destructive preprocessing
    curvature.py          Curvature features
    arch_geometry.py      PCA orientation + the convexity sign rule
    label_adapter.py      Bridge for any pretrained segmentation model
```

Test fixtures: `arch_fixture.py` (parametric arch, configurable crowding),
`tooth_fixture.py` (single crown), `incisor_fixture.py` (sharp incisal edge).
Synthetic fixtures are useful but were repeatedly too generous — validate
against real scans.

---

## 9. IF YOU CHANGE ONE THING

Run `python check_structure.py app_ui.py core_geometry.py` after every edit.
`py_compile` validates syntax only; a file can compile perfectly and die at
startup on a NameError. That happened here and cost a full round trip.
