# Phase 2 — Integrate "deform, don't cut" behind a flag

## 1. Summary

The deformation construction is wired end to end behind
`construction: Literal["collar","deformation"] = "collar"` on `/export/stages`
and `/export/final`, with the collar path untouched and still the default.
Both constructions now derive their stage matrices from ONE function, proven
identical at 0 ULP, so a difference in their verdicts can be attributed to the
construction rather than to the pose.

On the real mandible the phase ends in a **named refusal with measured
values**, which is the acceptance condition's second branch: the T0 cast
self-intersects before any tooth is moved, so no stage built from it could
ever be print ready, and `build_case_plan` refuses once with the census
attached rather than building N stages and refusing each for a reason that
does not name the cause.

---

## 2. Tasks

| Task | Files and functions | Commit |
|---|---|---|
| 2.1 | verification only — `test_deform_construction.py`, `test_self_intersection.py` in this venv | `44ed90c` |
| 2.2 | **new** `manufacturing_v2.py` (`DeformationPolicy`, `CasePlan`, `CasePlanRefused`, `build_case_plan`, `build_stage_v2`, `stage_matrices_for`, `gate_evidence`, `_file_dict`); **new** `stage_matrix.py` (`CLINICAL_KEYS`, `stage_clinical`, `stage_matrix`); `api_core._stage_clinical` delegates; **new** `test_stage_matrix_shared.py` | `44ed90c` |
| 2.3 | `manufacturing_v2.tooth_vertex_sets`, `manufacturing_v2.tooth_faces_from_labels`; **new** `test_deformation_vertex_sets.py` | `acae16b` |
| 2.4 | `api_core.StageExportRequest.construction` / `.prescribed_ipr_mm`, `api_core.build_stage_bundle_v2`, `api_core._v2_evidence_report`, `api_core._v2_attachment_solids`, dispatch in `export_stages` / `export_final` | `c4ef6ea` |
| 2.5 | **new** `test_export_deformation_api.py` | `c4ef6ea` |
| 2.6 | `manufacturing_v2.union_attachments`, `manufacturing_v2.AttachmentUnionRefused`, wiring in `build_stage_bundle_v2` | `c4ef6ea` |
| acceptance | **new** `real_scan_deformation.py`; `run_all_tests.py` (+3 entries) | `c4ef6ea` |

---

## 3. VERIFIED

### 2.1 — the kit, in THIS venv

```
$ .venv/Scripts/python.exe test_deform_construction.py
16 passed, 1 warning in 18.76s

$ .venv/Scripts/python.exe test_self_intersection.py
6 passed, 1 warning in 21.86s
```

Package versions on this machine (Windows, **not** the Linux figures the
brief's Part B records — re-verified as Phase 2 asked):

```
python 3.12.7 Windows-11-10.0.26200-SP0
numpy          2.5.2      scipy          1.18.1
manifold3d     3.5.2      open3d         0.19.0
trimesh        5.1.0      torch          2.13.0
scikit-learn   1.9.0      fastapi        0.141.1
pydantic       2.13.4     pytest         9.1.1
```

**This also settles an open question from Phase 1.** Phase 1 modified
`self_intersection.py` — a kit reference file — to bound broad-phase memory
(`cell = max(mean(ext) * 2.0, 1.5)` in place of `median(ext) * 1.5`, plus
chunked pair generation), and recorded the suite as *expected* to pass, which
is WRITTEN, not VERIFIED. The kit's own brute-force control passes here, so
the refactor did not change the maths.

### 2.2 — one stage matrix, 0 ULP

```
$ .venv/Scripts/python.exe -m pytest test_stage_matrix_shared.py -q
46 passed, 3 warnings in 1.20s
```

36 clinical-channel combinations, 6 prescriptions × 6 stages of bit-identical
4×4s (`np.array_equal` on raw float64, not `allclose`), a static check that
`api_core` delegates and that `manufacturing_v2` does not re-derive, stage 0 ==
identity and stage N == the full prescription, and the A1.3 control:

```
rebuilt R'R-I = 4.44e-16; a lerp reaches 2.58e-02 and is not a rotation
```

### 2.3 — the vertex sets

```
$ .venv/Scripts/python.exe -m pytest test_deformation_vertex_sets.py -s
PASS  scan 14,400 verts is the prefix of cast 15,008; 608 appended
PASS  608 appended vertices, all referenced
PASS  `any` overlaps at 30 vertices, `all` at 0
PASS  71 seam vertices, none in the rigid set
PASS  a short label array is refused, not silently indexed
PASS  moving 280, static 560, shared 0
PASS  all 280 shared vertices went to moving, and were counted
PASS  order-independent; resolved {'45': 280}
PASS  planned: 419 free vertices, 280 rigid
PASS  a wrong-length face mask is refused
11 passed, 1 warning in 1.53s
```

### 2.4 / 2.5 / 2.6 — the flag, over HTTP

```
$ .venv/Scripts/python.exe -m pytest test_export_deformation_api.py -s
PASS  construction defaults to collar on both request models
PASS  a misspelled construction is refused, not defaulted
PASS  200, X-Print-Ready true, 10 gates measured and passed
PASS  422, failed gates: ['implicit_ipr_within_prescription']
PASS  10 gates, each with its measurement
PASS  no flag -> collar, 1 stage(s), sockets filled flush at the gingival margi...
PASS  1 stage STL(s); occlusion checked=False and says so
PASS  print_compensation_mm refused for the deformation path
PASS  attachment unioned: 14112 tris, 28339.961 mm3, ok=True, 1 crumb(s) discarded
PASS  1 stage(s), no boolean performed
12 passed, 13 warnings in 24.83s
```

### All three export formats, on the deformation path

2.4 asks for the same response shape. §24.6 already pins for the collar that a
format is not a way around the gate; the same holds here.

```
zip       200  X-Print-Ready=true X-Export-Format=zip       346,338 bytes  application/zip
stl       200  X-Print-Ready=true X-Export-Format=stl       705,684 bytes  model/stl
manifest  200  X-Print-Ready=true X-Export-Format=manifest    6,789 bytes  application/json
raw STL is byte-identical to the one inside the ZIP: True
reread 7,058 verts / 14,112 faces
```

Pinned by `test_all_three_export_formats_go_through_the_same_gate`.

### The collar path, re-run on merit

```
$ .venv/Scripts/python.exe -m pytest test_staging_export.py \
      test_manufacturing_matrix.py test_manufacturing_interface.py -q
87 passed, 3 warnings in 239.67s (0:03:59)
```

### The canonical runner, on the final tree

```
$ .venv/Scripts/python.exe -u run_all_tests.py
  PASS  self-intersection          6 passed, 1 warning in 7.74s
  PASS  deformation construction   16 passed, 1 warning in 5.85s
  PASS  stage matrix shared        46 passed, 2 warnings in 0.10s
  PASS  deformation vertex sets    11 passed, 1 warning in 0.42s
  PASS  deformation export API     12 passed, 13 warnings in 24.83s
  PASS  real scan (local)          observed exit code  3
==================================================================
55 PASS / 0 SKIP / 0 FAIL
ALL EXECUTED TESTS PASSED
```

52 entries before this phase, 55 after. `real scan (local)` still observes
exit code 3 — Phase 0 pinned that as today's collar state and Phase 2 did not
change it.

### pytest, on the final tree

```
$ .venv/Scripts/python.exe -u -m pytest -q
550 passed, 17 warnings in 631.81s (0:10:31)
```

545 before the hardening commit, 550 after (+5: format parity, the refusal
status codes, and the split short/empty label guard).

### Structure, frontend and the cross-language pins

```
$ .venv/Scripts/python.exe check_structure.py
  117/117 files structurally sound

$ cd frontend && npm run lint          # 0 problems
$ npm run build                        # ok (chunk-size advisory only)
$ npm run smoke
RENDER OK — 21713 bytes of markup, no exception
BOUNDARY PASS-THROUGH OK — 21713 bytes
BOUNDARY FALLBACK OK — 1703 bytes, recovery text and reload control present

$ node verify-kinematics.mjs
every stage stays rigid (R'R-I)    : 4.44e-16, det-1 6.66e-16
  bar 1e-12, observed 1.78e-15 (~8 ULP of 1.0 - the float64 noise floor)
$ node verify-palette.mjs             99/99 checks passed
$ node verify-framing.mjs             PASS
$ node verify-shadowrig.mjs           PASS

$ npx playwright test
  9 skipped
  17 passed (1.6m)
```

17 passed / 9 skipped is **identical to §26.12's baseline** — the nine need a
running backend. 2.4 required no UI change and caused none.

---

## 4. WRITTEN, not executed

* **Nothing in the deformation path has produced a PRINT READY stage on real
  anatomy.** Every print-ready result in section 3 is synthetic.
* **The `occlusion` block of a deformation manifest is `checked: false`.** The
  antagonist sweep is not wired into this construction. It is reported as
  NOT_CHECKED with a note saying so, never as clearance (§14).
* **`union_attachments`' refusal branch (`AttachmentUnionRefused`) has not
  fired.** The one attachment case measured produced a single bonded body.
* **No maxillary scan exists**, so only the mandible was exercised — unchanged
  from §25.9.

---

## 5. Real-scan measurements

`case_lower.stl`, FDI 45, one stage, 0.25 mm extrusion, through
`construction=deformation`, driven over HTTP by `real_scan_deformation.py`.
**The whole product chain runs** — upload, occlusal plane, CrossTooth
segmentation, click-to-select, cut, prescription, export — and the refusal
comes from the geometry, not from a step that could not be reached.

```
$ .venv/Scripts/python.exe real_scan_deformation.py --fdi 45 --stages 1 --extrusion 0.25

[1] upload  (9.4 MB)
  POST /api/session                      7.37s
    94,848 verts / 187,625 faces
[2] occlusal plane
  POST /occlusal-plane                   0.01s
[3] segmentation  (provider=crosstooth)
  POST /segment?provider=crosstooth     43.97s
    16 teeth: [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
[4] click-to-select FDI 45
  POST /select-tooth                     0.27s
    3,178 vertices, purity 100.0%, fdi 45
[5] cut
  POST /cut                              1.42s      tooth 8787fc5e, root 14.0mm
[6] prescription  d_oa = +0.25mm
  POST /kinematics                       1.42s
[7] export  construction=deformation
  POST /export/final                    48.52s

HTTP 422   X-Print-Ready: (absent)

NAMED REFUSAL: t0_cast_self_intersects
  the T0 cast crosses itself before any tooth is moved. Every stage inherits
  this geometry, so no stage built from it can be print ready.

TIMING
  upload  7.37s  occlusal_plane  0.01s  segment  43.97s  select  0.27s
  cut     1.42s  kinematics      1.42s  export   48.52s  total  103.04s
```

**The measured values the refusal carries:**

| | |
|---|---|
| method | grid-AABB broad phase + float64 segment/triangle |
| touch tolerance | 1e-06 mm |
| candidate pairs | 2,501,958 |
| **intersecting pairs** | **3,591** |
| faces involved | 2,641 of 130,186 (2.03%) |
| by kind | `cross_or_touch` 3,516 · `shared_vertex_overlap` 70 · `folded_edge` 5 |
| first example | faces (992, 124324) at (18.7815, −24.8442, 4.3256) |

**The T0 cast itself, and it is watertight — which is the point:**

| | |
|---|---|
| vertices / faces | 96,835 / 130,186 |
| trimmed scan vertices | 94,849 |
| appended floor vertices | 1,986 |
| trim margin | 7.0 mm |
| build | 4.765 s |
| written STL | open 0 · non-manifold 0 · components 1 · winding consistent · volume 15,661.2533 mm³ |

**This independently reproduces Phase 1 to the vertex** — 96,835 / 130,186,
volume 15,661.2533 mm³, 3,591 pairs — by a different route: Phase 1 ran
`tools/t0_census.py` on `stl_io.parse_stl_bytes(blob)` (the mesh a reader
reconstructs, welded by position) and this runs `build_case_plan` on the
float32-rounded cast faces, which is the object AGENT_BRIEF 1.2 actually
names. The two disagreeing would have been a methodology defect; they agree
exactly, so the count is a property of the geometry rather than of how it was
measured.

**One observation about the header.** The brief asks that `X-Print-Ready` come
from `aggregate_gate_v2`, and on this 422 the header is **absent** rather than
`false` — the refusal happens in `build_case_plan`, before any stage exists
and therefore before any aggregate gate has been evaluated. Emitting `false`
would claim a gate ran. This matches the collar path, whose own 422s carry no
header either, so a client reading the header sees the same contract from both
constructions: a header means a gate was evaluated; its absence with a 422
means the case was refused before one could be.


---

## 6. Failures not fixed, and why

1. **The T0 cast self-intersects on `case_lower.stl`.** This is Phase 1's
   finding, re-measured here independently, and it is the reason the
   acceptance run refuses. Phase 1 left it as **Decision D1 (E7)** and no
   cleanup rule has been given, so nothing was invented: the refusal carries
   the census and names the decision it is waiting on. The trim margin is not
   a fix — a sweep measured 5 mm → 3,007 pairs, 7 mm → 3,591, 9 mm → 3,495,
   11 mm → 3,758.

2. **`print_compensation_mm` is refused, not supported, on the deformation
   path.** It is an offset applied to a fused solid after a union; there is no
   union here, and offsetting the deformed cast would move enamel that
   `moving_teeth_exact_rigid` requires to be bit-exact. Refusing is the honest
   behaviour; supporting it needs a design decision about where an allowance
   belongs in a construction with no boolean.

3. **The antagonist check is not wired in** (see §4).

---

## 7. Decisions needed from Asaad

**D1 — the T0 cleanup rule (carried over from Phase 1, still open).**
The real scan's T0 cast carries self-intersections before any tooth moves, and
`no_self_intersection` is a REQUIRED gate, so the deformation path refuses the
case outright. Phase 1 characterised them and this phase reproduced the cast
to the vertex.

| option | consequence |
|---|---|
| **(a) Fix `build_cast_base`'s extrusion on undercut geometry** *(recommended)* | 93.7% of the pairs are scan↔skirt — the extruded wall crossing the scan surface — and §9 records 11.6% of this scan's area as undercut. This is a construction defect in one function, and fixing it benefits the collar path too. |
| (b) Accept a bounded T0 count and gate on the DELTA | Makes the gate relative rather than absolute; a stage could then ship with real self-intersections inherited from T0. |
| (c) Pre-clean the scan before the cast is built | Reverses rule 3.1's spirit — it would move scanner coordinates to make a downstream step succeed. |

**D2 — should a deformation export run the antagonist sweep?**
Recommended: yes, in Phase 3, sharing the collar path's cached KD-tree rather
than duplicating it. Until then the manifest says `checked: false`.

---

## 8. Ready for the next phase?

**Yes for the code; no for the real-scan claim.** Every suite is green, the
flag is live on both endpoints with the collar untouched, and the acceptance
run ends in a named refusal with measured values and timing — which is what
Phase 2's acceptance allows. Phase 3 (real-scan regression v2) cannot show a
print-ready deformation stage on `case_lower.stl` until **D1** is decided.
