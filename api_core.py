import os
import io
import time
import json
import uuid
import zipfile
import traceback
import asyncio
import threading
import numpy as np
from scipy.spatial import cKDTree
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import core_geometry as cg
import jaw_naming
import tgn_bridge
import stl_io
from session_store import STORE, SessionExpired
import cut_guard
import arch_frame
import validation
import domain
import case_store
import space_analysis
import segmentation_review
import segmentation_fallback
import segmentation_providers
import segmentation_diagnostics
import segmentation_cleanup
import benchmark_segmentation
import scan_cache_manager
import clinical_safety
import export_clinical_report
import attachments as attachments_mod
import audit
import telemetry
import manufacturing as mfg
import self_intersection as si_mod
from typing import Literal
import stage_matrix
import manufacturing_v2 as mfg2
import print_solid as ps
import deform_construction as dc


# One trail per SESSION, because a session is what a clinician is working in and
# a Case may not exist yet — the first cut happens long before anything is saved.
# The trail travels into the Case when one is built.
_TRAILS: dict = {}


def _trail(sid: str) -> audit.AuditTrail:
    t = _TRAILS.get(sid)
    if t is None:
        t = _TRAILS[sid] = audit.AuditTrail()
    return t


def _record(sid: str, action: str, **fields):
    """Append to the session's audit trail. NEVER raises.

    audit.record refuses a forbidden key by raising, which is right for the
    module and wrong here: a request must not fail because someone added a
    field to a log line. The refusal is recorded as a dropped entry instead, so
    the denylist still holds and the failure is still visible.
    """
    try:
        _trail(sid).record(action, **fields)
    except Exception as e:                           # noqa: BLE001 — by design
        print(f"[audit] entry refused ({type(e).__name__}: {e})")


def _structured(result):
    """A refusal that carries its numbers.

    FastAPI serialises a dict detail as {"detail": {...}}, so a client gets the
    failing check, the metric it measured and the threshold it compared against
    — instead of a sentence it can only print. Success responses here have
    always been richly structured; failures collapsing to a bare string was
    exactly backwards.
    """
    return result.as_dict()


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

#: Everything a client may ask the server to WRITE lands under here. Nothing
#: outside it is reachable through the API (see `_confined_out_dir`).
EXPORTS_ROOT = os.path.join(CURRENT_DIR, "exports")

#: Upload ceiling. A real mandibular scan is 9.4 MB; 200 MB is generous for a
#: dense full-arch capture and still bounds a single request. The body used to
#: be read with one unbounded `await file.read()`, so a 9 GB POST became 9 GB
#: of process memory on an API with no authentication in front of it.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024


def _confined_out_dir(requested):
    """Resolve a client-supplied export directory INSIDE `EXPORTS_ROOT`.

    B10. `out_dir` went straight from the request body into
    `os.makedirs(...)` and `open(..., "wb")` with no validation at all, gated
    only by a `keep_local` boolean the same client sets. That is an arbitrary
    directory-creation and file-write primitive on an API with no
    authentication - latent on a localhost desktop binding, fatal anywhere
    else.

    Confined rather than deleted, because `keep_local` is a real workflow: a
    clinician asking for the files to stay on this machine. `Path.resolve()`
    first, so `..`, a symlink and an absolute path are all normalised before
    the check, and `is_relative_to` decides. Anything outside is a 422 that
    names the root - not a silent redirect, which would write somewhere the
    caller did not ask for.
    """
    from pathlib import Path
    root = Path(EXPORTS_ROOT).resolve()
    if requested is None:
        return root
    try:
        target = Path(requested).expanduser()
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
    except (OSError, ValueError) as e:
        raise HTTPException(422, f"out_dir is not a usable path: {e}")
    if not target.is_relative_to(root):
        raise HTTPException(
            422,
            f"out_dir must be inside {root}. Exports are confined to that "
            f"directory so a request cannot write anywhere on the server's "
            f"filesystem.")
    return target
CKPT_FPS = os.path.join(CURRENT_DIR, "ToothGroupNetwork", "ckpts", "0707_cosannealing_val.h5")
CKPT_BDL = CKPT_FPS

# STARTED FROM THE EXPORT MIRROR? SAY SO, RATHER THAN LOOK BROKEN.
#
# Aligner_App_AI_Export/ is a snapshot for handing to a reader, and
# build_ai_export.py skips every .h5/.pth/.ckpt/.pt by design (SKIP_EXT: model
# checkpoints are not source). But the snapshot is a full tree copy, so it also
# contains api_core.py and start_backend.bat — and a backend started in there
# comes up with all of the code and none of the model.
#
# The symptom is a FileNotFoundError naming a checkpoint path inside the
# mirror, which reads as a corrupted install. It is not: it is the wrong
# working copy, and the fix is to start from the project root. Worth naming
# because the mirror is refreshed from the tree, so its frontend is CURRENT —
# the app runs, looks right, and only the AI is missing.
EXPORT_DIR_NAME = "Aligner_App_AI_Export"
RUNNING_FROM_EXPORT = (
    os.path.basename(CURRENT_DIR) == EXPORT_DIR_NAME
    or (os.sep + EXPORT_DIR_NAME + os.sep) in (CURRENT_DIR + os.sep))
WRONG_COPY_HINT = (
    f"Wrong folder: this backend is running from {EXPORT_DIR_NAME}, which is an "
    f"export snapshot and carries no model checkpoints by design. Stop it and run "
    f"start_backend.bat from the project root instead."
)
if RUNNING_FROM_EXPORT:
    print("=" * 78)
    print("[Clinical AI] " + WRONG_COPY_HINT)
    print(f"              running from: {CURRENT_DIR}")
    print("=" * 78)

app = FastAPI(title="Virtual Diagnostic Setup API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    # 5173 is `npm run dev`; 4173 is `npm run preview`, which serves the
    # production build and is what the E2E suite drives. Without 4173 the
    # preview server cannot reach the API at all, and the failure is
    # indistinguishable from a dead backend: a CORS block and a refused
    # connection both reject fetch() with the same TypeError, so the
    # connection badge reports "Backend not running" for a server that is
    # running perfectly. Local development origins only — no wildcard.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SEGMENTATION_STATE = {
    "in_progress": False,
    "started_at": None,
    "message": "idle",
    # True from process start until the model has finished loading (or failed).
    # Without this, `loaded: false` during the first seconds is indistinguishable
    # from a pipeline that is genuinely broken, and the client says "AI
    # unavailable" about a model that is simply still importing torch.
    "warming": False,
    "warm_seconds": None,
    # Which backend the run in progress belongs to. `warming` only ever
    # described ToothGroupNetwork's startup load, and conflating the two
    # would make a CrossTooth run report that it was waiting for a model it
    # does not use.
    "provider": None,
}


@app.on_event("startup")
def _warm_tgn():
    """Start loading the AI pipeline WITHOUT blocking the server.

    This used to call tgn_bridge.load() inline. FastAPI runs startup handlers
    BEFORE uvicorn accepts connections, and that load imports torch, installs
    the CPU shims and builds a pipeline from a 64MB checkpoint — so every fetch
    from the browser failed for the whole of that window, and `--reload` repeated
    it on every single file save. That is the intermittent "Failed to fetch" the
    UI reports, and it looks exactly like a backend that will not start.

    A daemon thread lets uvicorn bind the port immediately. Nothing is lost:
    tgn_bridge.load is idempotent and thread-safe (`if _STATE["loaded"] and not
    force: return status()`), and /segment calls it again anyway — so a request
    arriving mid-warm-up simply waits on the same lock and gets the same
    pipeline, and a request arriving after a FAILED load gets the real error
    rather than a connection refusal.
    """
    def _load():
        t0 = time.time()
        status = tgn_bridge.load(CKPT_FPS, CKPT_BDL, model_name="tgnet")
        _SEGMENTATION_STATE["warming"] = False
        _SEGMENTATION_STATE["warm_seconds"] = round(time.time() - t0, 1)
        if status["loaded"]:
            print(f"[Clinical AI] Pipeline ready in {_SEGMENTATION_STATE['warm_seconds']}s.")
        else:
            print(f"[Clinical AI] Pipeline not ready: {status.get('error')}")

    _SEGMENTATION_STATE["warming"] = True
    print("[Clinical AI] Initializing AI Bridge in the background; API is live now.")
    threading.Thread(target=_load, name="tgn-warmup", daemon=True).start()


@app.get("/api/session/{sid}/audit")
def get_audit(sid: str):
    """The session's audit trail. Local, bounded, and free of identifiers."""
    t = _TRAILS.get(sid)
    if t is None:
        return {"entries": [], "summary": audit.AuditTrail().summary()}
    return {"entries": t.to_list(), "summary": t.summary()}


@app.get("/api/telemetry")
def get_telemetry(limit: int = 50):
    """Span status and the most recent spans. For support, not for a clinician.

    There is no endpoint to CONFIGURE the exporter, deliberately — see
    telemetry.py. This one only reads.
    """
    return {"status": telemetry.status(), "recent": telemetry.read_spans(limit)}


@app.get("/api/ai/status")
def ai_status():
    model = tgn_bridge.status()
    loaded = model.get("loaded", False)
    warming = _SEGMENTATION_STATE["warming"]
    return {
        "loaded": loaded,
        # Distinguish "still loading" from "broken". The client shows the first
        # as progress and the second as a failure, and conflating them made a
        # healthy server look dead for its first few seconds.
        "warming": warming,
        "warm_seconds": _SEGMENTATION_STATE["warm_seconds"],
        # RUNNING_FROM_EXPORT wins over the raw error: "checkpoint not found:
        # <a long path>" is true and useless, while the real fault is that the
        # backend was started in the export snapshot. The chip truncates to 60
        # characters, so the actionable words go first.
        "error": (WRONG_COPY_HINT if RUNNING_FROM_EXPORT and not loaded
                  else model.get("error")),
        "ready_message": ("Loading the AI segmentation model..." if warming
                          else "AI segmentation ready." if loaded
                          else model.get("error") or "AI segmentation unavailable."),
        "segmentation_in_progress": _SEGMENTATION_STATE["in_progress"],
        "segmentation_started_at": _SEGMENTATION_STATE["started_at"],
        "segmentation_message": _SEGMENTATION_STATE["message"],
        "segmentation_provider": _SEGMENTATION_STATE["provider"],
        # `loaded` above is ToothGroupNetwork's, which is what the health chip
        # has always meant. The registry is what a provider picker reads.
        "default_provider": segmentation_providers.DEFAULT_PROVIDER,
        "providers": {n: {"available": bool(p.get("available")),
                          "reason": p.get("reason")}
                      for n, p in segmentation_providers.registry().items()},
    }


def _vertices_as_the_pipeline_loaded_them(obj_path, fallback):
    """The vertex array ToothGroupNetwork ACTUALLY read, in ITS order.

    WHY THIS IS NOT JUST `v`. The pipeline reads the OBJ through
    `gen_utils.read_txt_obj_ls(use_tri_mesh=True)`, i.e.
    `trimesh.load_mesh(path, process=False)`, and the comment directly above
    that function in the vendored source reads "In some cases, trimesh can
    change vertex order". Measured on this project's real scan the order IS
    preserved - 0 of 94,848 rows differ - but the same OBJ read by Open3D
    comes back in a DIFFERENT order with the same count, so the property is
    a fact about one loader and one version, not about the file. Loading it
    back the way the pipeline does turns the assumption into a measurement,
    and `transfer_labels_by_position` then makes the ordering irrelevant.
    """
    try:
        import trimesh
        m = trimesh.load_mesh(obj_path, process=False)
        used = np.asarray(m.vertices, float)
        if used.ndim == 2 and used.shape[1] == 3 and len(used):
            return used
    except Exception:                                     # noqa: BLE001
        pass
    return np.asarray(fallback, float)


def run_segmentation(v, f, jaw, return_input_vertices=False):
    safe_dir = CURRENT_DIR.replace("\\", "/")
    safe_filename = jaw_naming.scan_filename(jaw, patient_id="conditioned", suffix=".obj")
    temp_path = f"{safe_dir}/{safe_filename}"
    temp_json = temp_path + "_output.json"

    with open(temp_path, "w") as out:
        for vert in v:
            out.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        for face in f:
            out.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    try:
        segmenter = tgn_bridge.predictor()
        segmenter.process(temp_path, temp_json)
        with open(temp_json, 'r') as jf:
            result_json = json.load(jf)
        if return_input_vertices:
            # BEFORE the finally clause deletes the file.
            return result_json, _vertices_as_the_pipeline_loaded_them(
                temp_path, v)
        return result_json
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        if os.path.exists(temp_json): os.remove(temp_json)

class OcclusalPlaneRequest(BaseModel):
    points: list[list[float]]

class SelectToothRequest(BaseModel):
    """Either a vertex id or a point in scanner coordinates.

    A vertex id when the client raycast the arch and read the nearest corner
    of the hit triangle; a point when it only has the intersection. Both
    resolve against THIS session's mesh, never a provider's array.
    """
    vertex_id: int = None
    point: list[float] = None


class WandRequest(BaseModel):
    point: list[float]
    tolerance: float | None = None
    
class ThresholdRequest(BaseModel):
    tolerance: float
    
class SelectionRequest(BaseModel):
    vertex_ids: list[int]

class CutRequest(BaseModel):
    vertex_ids: list[int]
    mesial_pt: list[float]
    distal_pt: list[float]
    # REQUIRED, with no default. It used to default to 10.0, which is an
    # incisor's root: a shipped manifest carried FDI 33, a mandibular canine,
    # on 10.0mm where Wheeler gives 13. C_res is extrapolated along the long
    # axis by exactly this distance, so a silent default is a silently wrong
    # pivot — and the client can always derive it from the FDI number
    # (rootDefaultForFDI) once segmentation has run.
    root_length_mm: float

class ExportRequest(BaseModel):
    out_dir: str | None = None
    # PHI: the ZIP is streamed to the clinician and nothing is left on the
    # server unless this is asked for. When it is, files land in the existing
    # exports dir (not cloud-synced) and no filename carries a patient name.
    keep_local: bool = False
    # Cast base geometry. There is deliberately NO allow_unsealed: it existed to
    # work around a base that cap_and_close could not seal, which was a
    # misdiagnosis — the scan was a healthy open shell and the capping was the
    # defect. A trim-and-extrude base closes by construction, so an unsealed one
    # is a bug to fix rather than a condition to override.
    trim_margin_mm: float = cg.ARCH_TRIM_MARGIN_MM
    base_thickness_mm: float = cg.CAST_BASE_THICKNESS_MM

class KinematicsRequest(BaseModel):
    tip_deg: float = 0.0
    torque_deg: float = 0.0
    rotation_deg: float = 0.0
    d_md: float = 0.0
    d_bl: float = 0.0
    d_oa: float = 0.0
    # The ANTAGONIST's session, if the opposing arch is loaded. The two arches
    # are separate sessions — STORE.create(arch) makes one per arch and nothing
    # links them — so the server cannot find the opposing scan on its own. The
    # client knows both and passes this; absent, the check is skipped silently.
    #
    # This works at all only because scanner coordinates are sacred (CLAUDE.md
    # rule 3.1): both arches sit in the same raw space, so an upper crown driven
    # lingually really does land where the lower arch is.
    opposing_session_id: str | None = None

@app.post("/api/session")
async def create_session(arch: str = Form(...), file: UploadFile = File(...)):
    try:
        sid = STORE.create(arch)          
    except ValueError as e:
        raise HTTPException(400, str(e))
    # CHUNKED AND CAPPED (B15). This was one unbounded `await file.read()`,
    # so the whole body became process memory before anything looked at it.
    # Reading in chunks lets the ceiling be enforced on the way in rather than
    # after the damage, and the session is released so a refused upload does
    # not also evict a real case from the four-session store.
    chunks, total = [], 0
    while True:
        chunk = await file.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            STORE.drop(sid)
            raise HTTPException(
                413,
                f"The uploaded scan exceeds {MAX_UPLOAD_BYTES // (1024*1024)} "
                f"MB. A full-arch intraoral scan is typically under 20 MB; if "
                f"this file is genuinely larger, decimate it before upload.")
        chunks.append(chunk)
    raw = b"".join(chunks)
    _t_up = time.perf_counter()
    verts, faces = stl_io.parse_stl_bytes(raw)
    # Content-addressable cache, keyed by the hash of the RAW upload so it
    # identifies what the clinician sent rather than what conditioning made of
    # it. Encrypted at rest; no filename stored. This is what lets a restart
    # restore a case without a manual re-upload.
    _scan_hash = scan_cache_manager.scan_hash(raw)
    # FINITENESS GATE, BEFORE ANYTHING ELSE TOUCHES THE BUFFER. condition_mesh's
    # degenerate filter is `area <= 1e-12`, which is False for NaN, so a NaN
    # triangle survives the only filter meant to remove it and the scan dies
    # deeper in as "LinAlgError: SVD did not converge" — an opaque 500 about a
    # file that opens fine everywhere else. Delete-only: no vertex is moved, so
    # rule 3.1 and inter-arch registration are untouched.
    verts, faces, sanity = cg.sanitize_scan(verts, faces)
    if sanity["repaired"] and not len(faces):
        raise HTTPException(422, "The uploaded scan has no usable triangles: every "
                                 "face referenced a vertex with NaN or infinite "
                                 "coordinates. Re-export the scan from the scanner.")
    verts, faces, report = cg.condition_mesh(verts, faces)
    report["sanitize"] = sanity
    STORE.put(sid, "verts", verts)
    STORE.put(sid, "faces", faces)
    STORE.put(sid, "scan_hash", _scan_hash)
    try:
        scan_cache_manager.put(raw, arch, len(verts), len(faces))
    except OSError as e:
        # A cache failure must not block a clinician mid-case. The session works
        # exactly as before; only restart-recovery is lost, and saying so beats
        # failing an upload over a disk problem.
        print(f"[scan cache] could not cache {_scan_hash[:12]}: {e}")

    # Record the scan's own topology BEFORE any cut. condition_mesh welds,
    # drops degenerates and debris and fills small holes, but it does not
    # repair non-manifold edges — so an arch can arrive already unprintable.
    # Knowing that here is what lets /export blame the scan instead of the cut.
    scan_health = cg.manifold_report(faces)
    STORE.put(sid, "scan_health", scan_health)

    # NO FILENAME. The upload's own name routinely carries a patient's, which is
    # exactly why session_store stores none and why audit.py's denylist refuses
    # the key outright.
    _record(sid, audit.SCAN_LOADED, arch=arch,
            detail=f"{len(verts):,} vertices, {len(faces):,} faces",
            # COUNTS, AND THE NAMES HAVE TO SAY SO. `faces` is on
            # audit.FORBIDDEN_KEYS because there it means the triangle ARRAY,
            # so passing a scalar under that name got the whole entry refused
            # and EVERY upload went unaudited. The denylist was right; the call
            # site was wrong. This is the same distinction telemetry.py splits
            # IDENTIFIER_KEYS from GEOMETRY_KEYS for, and it has now been got
            # wrong twice - test_failsafes pins every call site against the
            # denylist so it cannot happen a third time.
            values={"vertex_count": int(len(verts)), "face_count": int(len(faces)),
                    "welded": int(report.get("welded_vertices", 0)),
                    "open_edges": int(scan_health.get("open_edges", 0)),
                    "sanitize_repaired": bool(sanity["repaired"])})
    telemetry.record_span(
        "session.create", (time.perf_counter() - _t_up) * 1000,
        arch=arch, vertices=int(len(verts)), faces=int(len(faces)),
        bytes_in=len(raw), sanitize_repaired=bool(sanity["repaired"]),
        nonfinite_vertices=int(sanity["nonfinite_vertices"]))
    
    edges = cg.directed_edges(faces)
    conc = cg.boundary_field(verts, faces, edges=edges)
    graph = cg.build_barrier_graph(verts, faces, conc, edges=edges)
    STORE.put(sid, "edges", edges)
    STORE.put(sid, "concavity", conc)
    STORE.put(sid, "graph", graph)
    
    return {"session_id": sid, "vertex_count": int(len(verts)), "face_count": int(len(faces)),
            "bbox": [verts.min(0).tolist(), verts.max(0).tolist()],
            "conditioning": report, "scan_health": scan_health,
            "scan_hash": _scan_hash,
            "scan_cached": scan_cache_manager.has(_scan_hash)}

@app.get("/api/session/{sid}/mesh")
def get_mesh(sid: str):
    try:
        v, f = STORE.require(sid, "verts"), STORE.require(sid, "faces")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    return {"positions": np.asarray(v, np.float32).ravel().tolist(), "indices": np.asarray(f, np.uint32).ravel().tolist()}

@app.post("/api/session/{sid}/occlusal-plane")
def set_occlusal_plane(sid: str, req: OcclusalPlaneRequest):
    try:
        v = STORE.require(sid, "verts")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    if len(req.points) != 3:
        raise HTTPException(400, "Must provide exactly 3 points: [Left, Right, Anterior]")
    
    centroid = v.mean(axis=0)
    try:
        frame = arch_frame.fit_occlusal_frame(req.points[0], req.points[1], req.points[2], centroid)
    except ValueError as e:
        raise HTTPException(400, str(e))
        
    STORE.put(sid, "arch_frame", frame)
    return arch_frame.to_json(frame)

@app.post("/api/session/{sid}/segment")
async def segment(sid: str, provider: str = None):
    """Segment this arch. `?provider=` picks the backend; see
    GET /api/segmentation/providers for what is installed.

    THE TOGGLE IS PER REQUEST, not a server mode, because the point of having
    two models is running both over ONE scan and comparing. A server-wide
    switch would make that a restart apiece. The process-wide default comes
    from ALIGNER_SEGMENTATION_PROVIDER and is `toothgroupnetwork` unless it
    is set.

    EVERY PROVIDER LANDS IN THE SAME DOWNSTREAM PATH - the FDI/jaw check, the
    hybrid geodesic fallback, the spatial diagnostic, the audit entry and the
    response shape are shared. A second model must not come with a second set
    of guards, or the guards are not guards.
    """
    try:
        v, f, arch = (STORE.require(sid, "verts"), STORE.require(sid, "faces"), STORE.arch(sid))
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    try:
        backend = segmentation_providers.get(provider)
    except KeyError as e:
        raise HTTPException(400, str(e))
    # THE GATE DELIBERATELY SKIPS ToothGroupNetwork, and that is not an
    # oversight. Its `available()` reports `loaded`, which is False for the
    # first several seconds of every process while the background warm-up
    # runs (s.12) - and the shipped behaviour for a request landing in that
    # window is to WAIT on the same lock and get the same pipeline, not to be
    # refused. Gating on it here would turn a 10-second wait into a 409 and
    # would be a regression in the one path that has a user in front of it.
    if backend.name != "toothgroupnetwork":
        ready = backend.available()
        if not ready.get("available"):
            raise HTTPException(
                409, f"Segmentation provider {backend.name!r} is not "
                     f"available: {ready.get('reason')}")

    if _SEGMENTATION_STATE["in_progress"]:
        raise HTTPException(409, "AI segmentation is already running. Please wait for it to finish.")

    _SEGMENTATION_STATE.update({
        "in_progress": True,
        "started_at": time.time(),
        "provider": backend.name,
        "message": ("Waiting for the AI model to finish loading..."
                    if _SEGMENTATION_STATE["warming"]
                    and backend.name == "toothgroupnetwork"
                    else f"Running {backend.name} on this arch..."),
    })

    try:
        provider_meta = {}
        if backend.name == "toothgroupnetwork":
            # Idempotent. If the background warm-up is still running this blocks on
            # the same lock and returns the same pipeline; if the warm-up failed it
            # returns that error rather than a second attempt's.
            model_status = await asyncio.to_thread(
                tgn_bridge.load, CKPT_FPS, CKPT_BDL, "tgnet")
            if not model_status["loaded"]:
                raise RuntimeError(model_status.get("error") or "AI segmentation model did not initialize.")
            _SEGMENTATION_STATE["message"] = "Running ToothGroupNetwork on this arch..."

            result_json, used_verts = await asyncio.to_thread(
                run_segmentation, v, f, arch, True)
            check = jaw_naming.verify_fdi_matches_jaw(result_json, arch)
            if not check["ok"]:
                raise HTTPException(500, check["diagnosis"])

            labels, _ = jaw_naming.extract_labels(result_json, expect_jaw=arch)
            if len(labels) != len(v):
                raise HTTPException(500, "Label array does not match mesh.")
        else:
            # ANY OTHER PROVIDER GOES THROUGH THE REGISTRY, and it is
            # responsible for returning labels already on THIS array - which
            # for CrossTooth means a coordinate KNN rather than an index, so
            # there is no ordering to get wrong. The identical FDI/jaw check
            # runs on the result either way: a second model does not get a
            # softer gate than the first.
            try:
                result = await asyncio.to_thread(backend.segment, v, f, arch)
            except NotImplementedError as e:
                raise HTTPException(501, str(e))
            labels = [int(x) for x in np.asarray(result.labels).reshape(-1)]
            if len(labels) != len(v):
                raise HTTPException(
                    500, f"{backend.name} returned {len(labels)} labels for "
                         f"{len(v)} vertices.")
            check = jaw_naming.verify_fdi_matches_jaw(labels, arch)
            if not check["ok"]:
                raise HTTPException(500, check["diagnosis"])
            provider_meta = _jsonable(result.meta)
            transfer = result.transfer

        # THE LABELS ARE MOVED ONTO OUR ARRAY BY POSITION, NEVER BY INDEX.
        # The length check above is the one that used to stand alone, and it
        # cannot see the failure it appears to guard: four different loaders
        # produce 94,848 vertices from this project's real scan in four
        # different ORDERS, so the count agrees while the correspondence is
        # wrong. Measured on the cached real-scan labels read by index, the
        # median per-tooth bounding box is 50.71mm - most of a mandible - and
        # 13.97mm once transferred by position. That was reported as a model
        # failure for a model that had segmented the arch correctly.
        if backend.name == "toothgroupnetwork":
            try:
                labels, transfer = segmentation_providers.\
                    transfer_labels_by_position(used_verts, labels, v)
            except segmentation_providers.LabelTransferError as e:
                raise HTTPException(500, f"Segmentation labels do not belong "
                                         f"to this mesh: {e}")

        # HYBRID FALLBACK. Tier 1 is the model; any tooth whose region is
        # geometrically impossible - split across two places, or implausibly
        # sized - is re-grown by the classical geodesic flood, which follows the
        # curvature barrier at the cervical margin and therefore returns ONE
        # connected region by construction. The FDI is kept from the model,
        # which the flood cannot supply, and the tooth is marked
        # REVIEW_REQUIRED because a repair is not a confirmation.
        af = STORE.get(sid, "arch_frame")
        occ = np.asarray(af["u_occ"], float) if af else None
        centre = v.mean(axis=0) if af is not None else None
        hybrid = segmentation_fallback.run(
            labels, v, f, arch,
            graph=STORE.get(sid, "graph"), concavity=STORE.get(sid, "concavity"),
            arch_centre=centre, occlusal_axis=occ)
        labels = hybrid["labels"]

        # --- CLEANUP: one region per tooth, and no specks -----------------
        # Two complaints with one cause. A tooth label that falls into
        # several pieces has exactly one real tooth in it; the rest are stray
        # triangles on a neighbour or out on the gingiva. And gingiva specks
        # scattered over a crown are individually tiny and collectively the
        # first thing a clinician sees - they make the cervical rim ragged
        # where it matters most.
        #
        # AFTER the hybrid step, not before: the hybrid may hand a tooth back
        # to manual landmarks, and cleaning a region that is about to be
        # replaced would be work thrown away. Labels only - no vertex moves,
        # no face is added or removed, and the array comes back the same
        # length, so every vertex id the client holds keeps its meaning.
        try:
            labels, cleanup = segmentation_cleanup.clean_labels(v, f, labels)
        except Exception as e:                            # noqa: BLE001
            # Reported, never swallowed: a cleanup that could not run must
            # not look like a cleanup that found nothing to do.
            cleanup = {"error": f"{type(e).__name__}: {e}"}

        STORE.put(sid, "labels", labels)
        STORE.put(sid, "segmentation_cleanup", cleanup)
        STORE.put(sid, "segmentation_tiers", hybrid["teeth"])
        # The spatial diagnostic, computed on what the session will actually
        # hold. A label array that is not indexed to this mesh is a different
        # failure from a model that segmented badly, and no metric computed
        # on the labels alone - IoU included - can tell them apart.
        try:
            diag = segmentation_diagnostics.label_report(v, f, labels)
        except Exception as e:                            # noqa: BLE001
            diag = {"error": f"{type(e).__name__}: {e}"}
        diag["label_transfer"] = transfer
        # IS THE GUM LABELLED GUM? A different question from "is this label
        # spatially coherent", answerable with no annotation, and the one a
        # clinician notices first. Measured on what the session will actually
        # hold, against the session's own occlusal frame when there is one.
        try:
            diag["bands"] = segmentation_diagnostics.band_report(
                v, f, labels, u_occ=occ)
        except Exception as e:                            # noqa: BLE001
            diag["bands"] = {"measured": False,
                             "reason": f"{type(e).__name__}: {e}"}
        diag["cleanup"] = cleanup
        # THE PROVIDER THAT ACTUALLY RAN, not the configured default. Those
        # were the same value for as long as there was one provider, and
        # recording the default would now mislabel every A/B run.
        diag["provider"] = backend.name
        diag["provider_meta"] = provider_meta
        STORE.put(sid, "segmentation_diagnostics", diag)
        _n_teeth = int(len(set(int(x) for x in labels)) - (1 if 0 in set(
            int(x) for x in labels) else 0))
        _record(sid, audit.SEGMENTED, arch=arch,
                detail=f"{_n_teeth} region(s) labelled by {backend.name}",
                values={"teeth": _n_teeth,
                        "repaired": int(sum(1 for t in hybrid["teeth"]
                                            if t.get("tier") != "model"))})
        telemetry.record_span(
            "segment",
            (time.time() - _SEGMENTATION_STATE["started_at"]) * 1000,
            arch=arch, vertices=int(len(v)), arch_faces=int(len(f)),
            teeth=_n_teeth, provider=backend.name,
            repaired=int(sum(1 for t in hybrid["teeth"]
                             if t.get("tier") != "model")))
        return {
            "labels": [int(x) for x in labels], "jaw": arch, "report": check,
            "provider": backend.name, "provider_meta": provider_meta,
            # Intrinsic geometric plausibility. NOT measured accuracy - no IoU is
            # computable without annotated ground truth, and the payload says so
            # in `is_measured_accuracy` and `meaning`.
            "segmentation_confidence_breakdown": _jsonable(
                benchmark_segmentation.confidence_breakdown(
                    labels, v, f, arch, centre, occ)),
            "cleanup": _jsonable(cleanup),
            "bands": _jsonable(diag.get("bands")),
            "fallback": _jsonable({k: hybrid[k] for k in
                                   ("teeth", "repaired_count", "needs_review",
                                    "tiers", "threshold", "limitation")}),
        }
    finally:
        _SEGMENTATION_STATE.update({
            "in_progress": False,
            "started_at": None,
            "provider": None,
            "message": "idle",
        })

@app.post("/api/session/{sid}/wand")
def wand(sid: str, req: WandRequest):
    try:
        v = STORE.require(sid, "verts"); g = STORE.require(sid, "graph"); conc = STORE.require(sid, "concavity")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    if len(req.point) != 3:
        raise HTTPException(400, "point must be [x, y, z] in scanner coordinates")
    
    clicked = np.asarray(req.point, float)
    seed = cg.snap_seed_to_ridge(v, conc, clicked, search_radius_mm=3.0)
    
    dist = cg.geodesic_from_seed(g, v, seed)
    STORE.put(sid, "wand_field", dist)
    STORE.put(sid, "wand_seed", seed)
    
    auto = cut_guard.auto_tolerance(dist)
    tol = req.tolerance if req.tolerance else auto["tolerance"]

    ids = np.nonzero(dist <= tol)[0]
    finite = dist[np.isfinite(dist)]
    # A TOLERANCE THAT WAS NEVER FOUND MUST NOT READ LIKE ONE THAT WAS.
    # `auto_tolerance` scans 1.0 to 25.0mm looking for a plateau in the flood's
    # growth - the sulcus barrier - and when there is none it returns its own
    # CEILING, 25.0, with `plateau_found: False`. That flag was computed and
    # then dropped here, so the response said `tolerance: 25.0` in exactly the
    # field a found plateau uses. Measured on this project's real scan, the
    # ceiling comes back for all sixteen teeth and floods 26,408 of 94,848
    # vertices - a quarter of the arch, 15.7% of it belonging to the tooth
    # clicked - and the client had no way to tell that from a measurement.
    #
    # THE FLOOD IS NOT CHANGED. The wand is the CORRECTION tool now (see
    # /select-tooth), and refusing here would break the one route that needs
    # no segmentation at all. What changes is that the response says so.
    auto_used = not req.tolerance
    return {"vertex_ids": ids.tolist(), "tolerance": tol,
            "tolerance_source": "caller" if req.tolerance else "auto",
            "plateau_found": bool(auto["plateau_found"]),
            "tolerance_is_a_ceiling_not_a_measurement":
                bool(auto_used and not auto["plateau_found"]),
            "selected_fraction_of_arch": round(float(len(ids) / max(len(v), 1)), 4),
            "note": (
                "No plateau in the flood's growth was found between 1.0 and "
                f"{auto['tolerance']:.1f} mm, so this is the scan's ceiling "
                "rather than a measured tooth boundary. Click the tooth "
                "instead (it reads the segmentation label), or set a "
                "tolerance by hand."
                if auto_used and not auto["plateau_found"] else None),
            "seed": seed.tolist(),
            "seed_moved_mm": round(float(np.linalg.norm(seed - clicked)), 3),
            "max_distance": float(finite.max()) if finite.size else 0.0}

@app.post("/api/session/{sid}/select-tooth")
def select_tooth(sid: str, req: SelectToothRequest):
    """A click on a tooth selects THAT TOOTH, from the canonical labels.

    THE WAND IS NOT THE ROUTE FOR AN ORDINARY SELECTION, and measurement is
    why. `/wand` floods geodesically from a snapped seed, and on this
    project's real scan its auto tolerance returns 25.00 mm and selects
    26,408 of 94,848 vertices - a quarter of the arch (s.19, s.24.7).
    Narrowing it by hand does not help: `snap_seed_to_ridge` moves the seed
    up to 3 mm to the nearest high-concavity point, which is the interdental
    sulcus BETWEEN two teeth, and the best IoU against any label over
    tolerances of 1 to 8 mm measured 0.01 to 0.05.

    Segmentation has already answered this question. This endpoint reads the
    label under the click and returns that label's own region, so the
    clinical path is click -> whole tooth. The wand and the brush stay
    exactly as they are, for CORRECTING a region the model got wrong.

    THE REGION IS THE LABEL'S LARGEST CONNECTED COMPONENT, not every vertex
    carrying the label, and that is the same rule `real_scan_regression.py`
    and tier 2 of `segmentation_fallback` already use. A label can carry a
    handful of stray triangles on a neighbour - measured on the real scan,
    eleven of twelve teeth have their largest piece at 98.7% or better - and
    a crown built from two disconnected pieces cannot close watertight, so
    `/cut` would refuse it 422 with a message about the cut rather than
    about the label.
    """
    try:
        v = STORE.require(sid, "verts")
        f = STORE.require(sid, "faces")
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    labels = STORE.get(sid, "labels")
    if labels is None:
        raise HTTPException(
            409, "This arch has not been segmented. Run Segment Teeth first, "
                 "or use the wand to select a region by hand.")
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if len(labels) != len(v):
        raise HTTPException(500, "The stored labels do not match this mesh.")

    # THE CLICK IS RESOLVED ON THIS MESH, never on a provider's array. The
    # client raycasts the geometry it is drawing, which is this `v`/`f` in
    # this order, so a vertex id or a point both land here unambiguously.
    if req.vertex_id is not None:
        if not (0 <= req.vertex_id < len(v)):
            raise HTTPException(400, f"vertex_id {req.vertex_id} is outside "
                                     f"this mesh's {len(v)} vertices")
        vid = int(req.vertex_id)
    elif req.point is not None and len(req.point) == 3:
        from scipy.spatial import cKDTree
        tree = STORE.get(sid, "vertex_tree")
        if tree is None:
            tree = cKDTree(v)
            STORE.put(sid, "vertex_tree", tree)
        _, vid = tree.query(np.asarray(req.point, float))
        vid = int(vid)
    else:
        raise HTTPException(400, "provide either vertex_id or point [x,y,z]")

    fdi = int(labels[vid])
    if fdi == 0:
        return {"fdi": None, "vertex_ids": [], "clicked_vertex": vid,
                "route": "gingiva",
                "message": "That point is gingiva, not a tooth. Click a "
                           "tooth, or use the wand to select by hand."}

    mask = labels == fdi
    # Faces whose three corners all carry the label - the same strict rule
    # `segmentation_diagnostics` uses, so the two agree about what a tooth is.
    fmask = mask[f].all(axis=1)
    n_faces_all = int(fmask.sum())
    components = 1
    if n_faces_all:
        fmask = cg.largest_face_component(f, fmask)
    kept = int(fmask.sum())
    ids = np.unique(f[fmask].reshape(-1)) if kept else np.where(mask)[0]

    diag = STORE.get(sid, "segmentation_diagnostics") or {}
    row = next((r for r in diag.get("teeth", []) if r.get("label") == fdi), {})

    STORE.put(sid, "selection", ids)
    return {
        "fdi": fdi,
        "vertex_ids": [int(x) for x in ids],
        "clicked_vertex": vid,
        "route": "segmentation label",
        "label_vertices": int(mask.sum()),
        "label_faces": n_faces_all,
        "kept_faces": kept,
        "discarded_faces": n_faces_all - kept,
        # Carried through so the client never has to re-derive a verdict the
        # server already measured, and so a tooth the diagnostic flagged is
        # flagged at the moment it is selected rather than after the cut.
        "components": row.get("components"),
        "largest_component_fraction": row.get("largest_component_fraction"),
        "bbox_diagonal_mm": row.get("bbox_diagonal_mm"),
        "plausible_size": row.get("plausible_size"),
    }


@app.post("/api/session/{sid}/wand/threshold")
def wand_threshold(sid: str, req: ThresholdRequest):
    try:
        dist = STORE.require(sid, "wand_field")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    return {"vertex_ids": np.nonzero(dist <= req.tolerance)[0].tolist(), "tolerance": req.tolerance}

@app.put("/api/session/{sid}/selection")
def put_selection(sid: str, req: SelectionRequest):
    try:
        n = len(STORE.require(sid, "verts"))
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    ids = np.asarray(req.vertex_ids, dtype=np.int64)
    STORE.put(sid, "selection", ids)
    return {"count": int(ids.size)}

@app.post("/api/session/{sid}/cut")
def cut(sid: str, req: CutRequest):
    _t_cut = time.perf_counter()
    try:
        try:
            v, f = STORE.require(sid, "verts"), STORE.require(sid, "faces")
            conc = STORE.require(sid, "concavity")
            # STORE.get, not STORE.require: require() raises SessionExpired,
            # which the handler above turns into a 404 and would report a
            # missing occlusal plane as an expired session.
            af = STORE.get(sid, "arch_frame")
        except SessionExpired as e:
            raise HTTPException(404, str(e))

        if af is None:
            # Refusing rather than guessing. Without the occlusal plane there
            # is no reliable apical direction, and C_res is extrapolated along
            # it -- a silently guessed axis is exactly what put the pivot
            # outside the tooth and sent the crown swinging off the cast.
            raise HTTPException(409,
                "Occlusal plane not defined for this session. Establish it first "
                "(POST /api/session/{sid}/occlusal-plane with the left posterior, "
                "right posterior and anterior midline landmarks). The tooth's long "
                "axis is reconciled against it, and C_res is extrapolated along "
                "that axis into the bone; without it the pivot is a guess.")

        # Cumulative record of what has already left the arch. The session's
        # verts/faces/edges/graph/concavity are NEVER mutated: every one of
        # them is precomputed at upload (the barrier graph is the expensive
        # step) and every vertex index the client holds — selections, wand
        # fields, seeds — indexes the original array. Tracking extraction as a
        # face mask over the original mesh keeps all of that valid while still
        # letting the client render a genuinely socketed base.
        extracted = STORE.get(sid, "extracted_faces")
        if extracted is None:
            extracted = np.zeros(len(f), bool)

        # BOUNDS-CHECK BEFORE INDEXING. `sel[ids] = True` with a negative id
        # silently wraps and selects from the END of the array — a crown built
        # from the wrong side of the arch, reported as success. An out-of-range
        # id raised an opaque 500. Both are now a message naming the offender.
        ids = np.asarray(req.vertex_ids, np.int64)
        if ids.size:
            bad = ids[(ids < 0) | (ids >= len(v))]
            if bad.size:
                raise HTTPException(422, _structured(
                    validation.Result("selection").error(
                        "vertex_ids_in_range", False,
                        f"{bad.size} vertex id(s) are outside 0..{len(v)-1} "
                        f"(e.g. {int(bad[0])}). A negative id would wrap and select "
                        f"from the far end of the arch.",
                        {"out_of_range_count": int(bad.size),
                         "first_offender": int(bad[0]),
                         "mesh_vertices": int(len(v))})))

        root_check = validation.validate_root_length(req.root_length_mm)
        land_check = validation.validate_landmarks(req.mesial_pt, req.distal_pt)
        if not (root_check.ok and land_check.ok):
            raise HTTPException(422, _structured(
                validation.merge(root_check, land_check, subject="cut request")))

        sel = np.zeros(len(v), bool)
        sel[ids] = True
        face_mask = sel[f].all(axis=1)
        if not face_mask.any():
            raise HTTPException(400, "Selection covers no complete face; widen it.")

        # Never re-extract. Without this a brush stroke that strays onto an
        # already-removed tooth yields a crown built from faces that are no
        # longer part of the cast.
        overlap = int((face_mask & extracted).sum())
        face_mask &= ~extracted
        if not face_mask.any():
            raise HTTPException(400,
                f"This selection lies entirely inside a tooth that has already been "
                f"extracted ({overlap} faces). Select a tooth still on the cast.")

        # Measure the selection BEFORE largest_face_component discards the
        # islands, so the client can be told what was thrown away. This used to
        # be a silent prune — a selection in five pieces became a crown built
        # from one of them, with nothing in the response saying so.
        faces_before_prune = int(face_mask.sum())
        face_mask = cg.largest_face_component(f, face_mask)
        faces_after_prune = int(face_mask.sum())

        sel_verts = np.unique(f[face_mask])
        labels_arr = STORE.get(sid, "labels")
        hist = None
        if labels_arr is not None and len(labels_arr) == len(v):
            vals, counts = np.unique(np.asarray(labels_arr)[sel_verts], return_counts=True)
            hist = {int(k): int(c) for k, c in zip(vals, counts)}
        selection_check = validation.validate_selection(
            n_selected_vertices=int(ids.size), n_vertices=int(len(v)),
            n_selected_faces=faces_before_prune,
            largest_component_faces=faces_after_prune, label_histogram=hist)

        used_verts = np.unique(f[face_mask])
        (cv, cf), (bv, bf) = cg.split_by_face_mask(v, f, face_mask)

        loops = cg.boundary_loops(cf)
        if not loops:
            raise HTTPException(400, "Selection has no open margin. Re-select the boundary.")
        
        # Two different rims, deliberately.
        #
        # The GEOMETRIC rim is the largest loop only, matching app_ui.py and
        # every test. The long axis is now the normal of the plane fitted
        # through these points, and a scan hole or a leak lobe adds a second
        # loop somewhere else on the crown; the least-variance direction of a
        # two-ring point cloud is arbitrary, so merging them would make the new
        # estimator worse than the one it replaces.
        #
        # The GUARD rim stays the union of every loop, because cut_guard reads
        # the concavity field at those vertices to decide whether the cut
        # actually landed in the sulcus, and a second open loop is exactly the
        # kind of thing it should be allowed to see.
        rim_local_idx = np.unique(np.concatenate([np.asarray(L) for L in loops]))
        rim_global_idx = used_verts[rim_local_idx]

        # The socket rim must be a SIMPLE cycle. A cervical margin can pinch to
        # a single vertex, and boundary_loops then walks through it twice —
        # which makes the cup build two spokes onto the same vertex and gives
        # that edge four faces. Measured on a real lower molar: a 239-vertex rim
        # with one vertex visited twice, and the exported base refused.
        #
        # The spur left outside the largest simple cycle stays open, and
        # trim_to_arch's interior-hole fill caps it at export. It is a handful
        # of triangles inside a socket; the alternative is a cup that cannot be
        # printed.
        socket_loop = np.asarray(
            max(cg._split_self_touching_loop(max(loops, key=len)), key=len))
        rim_pts = cv[socket_loop].copy()
        socket_rim_global = used_verts[socket_loop]      # indexes the ORIGINAL mesh

        cv, cf = cg.cap_and_close(cv, cf)
        # Consistent winding on the CROWN only. app_ui.py:249 does this and
        # /cut never did, so capped crowns reached the browser with mixed
        # winding and computeVertexNormals produced flipped patches that
        # THREE.DoubleSide quietly hid. Measured 0.34s on a 15k-tri crown.
        # Deliberately NOT applied to the base: it is a full-mesh Python BFS
        # and its own docstring warns it runs for tens of seconds on an arch.
        cf = cg.make_consistent_winding(cv, cf)
        if not cg.is_edge_manifold_closed(cf):
            raise HTTPException(422, "Crown did not close watertight.")

        # Reporting, not gating — see cut_guard.check_crown's docstring. Its two
        # thresholds have never been measured against real cuts, so they are
        # surfaced as `crown_advisory` in the response and promoted to refusals
        # only once there are numbers behind them.
        verdict = cut_guard.check_crown(cv, cf, rim_global_idx, conc)

        # The arch frame is what makes this biological rather than geometric:
        # the long axis is signed and reconciled against the occlusal plane, so
        # C_res lands in the bone along the tooth's own root instead of 10mm
        # sideways into empty space.
        frame = cg.derive_frame_from_region(
            np.asarray(req.mesial_pt, float), np.asarray(req.distal_pt, float),
            cv, rim_pts, arch_frame=af)
        # The clamp report rides back to the client. A pivot shifted 2mm without
        # anyone being told is only visible months later, as a tooth that tipped
        # where it should have translated.
        cres_report: dict = {}
        # The tooth id does not exist yet (it is minted below), so it is stamped
        # onto the note afterwards rather than guessed here.
        c_res = cg.center_of_resistance(frame, root_length_mm=req.root_length_mm,
                                        report=cres_report)

        # Backstop, deliberately sharing NO code with the frame derivation.
        # Everything above is one chain of reasoning; if it is wrong anywhere,
        # it is wrong confidently. This asks the only question that matters —
        # is the pivot in the bone, under the tooth? — using the arch's own
        # occlusal normal and nothing from the tooth axis. tan(clamp) is the
        # widest lateral excursion the clamp can produce at a given depth.
        d = c_res - frame["rim_centroid"]
        u_occ = np.asarray(af["u_occ"], float)
        depth = float(-(d @ u_occ))
        lateral = float(np.linalg.norm(d - (d @ u_occ) * u_occ))
        max_lateral = depth * np.tan(np.radians(cg.MAX_AXIS_DEVIATION_DEG)) + 1.0
        # Finiteness is checked SEPARATELY and FIRST inside validate_pivot.
        # `depth <= 0 or lateral > max_lateral` looks exhaustive and is not:
        # every comparison against NaN is False, so a NaN pivot passed straight
        # through this gate and into a transform matrix.
        pivot_check = validation.validate_pivot(depth, lateral, max_lateral)
        if not pivot_check.ok:
            raise HTTPException(422, _structured(pivot_check))

        # Independent second opinion — reconcile_tooth_frame measures the same
        # disagreement without sharing any code with the correction above.
        reconcile = arch_frame.reconcile_tooth_frame(frame, af)
        dims = cg.crown_dimensions(cv, frame)

        # --- the socketed base, without rebuilding the arch -------------------
        # The base is the original mesh MINUS faces, so the client already holds
        # every vertex position it needs; it only needs to know which faces to
        # stop drawing. Sending face indices costs ~15k ints and 0.002s against
        # rebuilding and re-sealing a ~15MB base mesh. Measured on the real
        # mandibular scan, cap_boundary_loop on the arch's own horseshoe
        # perimeter alone took 100.2s — and produced a membrane across the
        # tongue space for its trouble. /export now builds the base by
        # trim_to_arch -> build_cast_base instead, at 3.1s for the pair.
        #
        # The socket cup built here is KEPT, not just rendered: /export reuses
        # these exact triangles, so what the clinician approves on screen is
        # what the lab receives. Watertightness for the whole base is
        # established once, at /export.
        removed_faces = np.where(face_mask)[0]
        extracted = extracted | face_mask
        STORE.put(sid, "extracted_faces", extracted)

        # --- the socket: a carved cup, not a lid over the hole ----------------
        # The fan this replaces put its apex AT the rim centroid, so the cap
        # landed on the margin plane at depth 0.00mm. That flat lid is what read
        # as a dark irregular opening under a lifted crown. build_socket_cup
        # descends into the bone and clamps itself against the underside of the
        # cast so it cannot erupt through a thin anterior base.
        rim_xyz = v[socket_rim_global]
        n_rim = len(socket_rim_global)
        cup_pts, cup_faces, cup_info = cg.build_socket_cup(
            rim_xyz, np.asarray(frame["u_oa"], float),
            base_verts=v, base_faces=f[~extracted])

        # Rim vertices already exist in the client's buffer; interior vertices
        # are new. Negative index -(k+1) means "appended vertex k", extending
        # the single-centroid -1 convention this replaces.
        socket_faces = []
        for tri in cup_faces:
            socket_faces.append([
                int(socket_rim_global[i]) if i < n_rim else -(int(i) - n_rim + 1)
                for i in tri])

        tid = uuid.uuid4().hex[:8]
        if "cres_clamp" in cres_report:
            cres_report["cres_clamp"]["tooth_id"] = tid
        _record(sid, audit.TOOTH_CUT, arch=STORE.arch(sid), tooth_id=tid,
                detail=f"crown extracted on a {req.root_length_mm:.1f}mm root",
                values={"selected_vertices": len(req.vertex_ids),
                        "root_length_mm": float(req.root_length_mm),
                        "cres_clamped": bool("cres_clamp" in cres_report)})
        # record_span, not span(): the work is already done by the time the
        # counts are known, so a context manager here would wrap nothing and
        # report duration_ms 0.0 for a multi-second cut.
        telemetry.record_span(
            "cut", (time.perf_counter() - _t_cut) * 1000,
            arch=STORE.arch(sid), selected_vertices=len(req.vertex_ids),
            arch_faces=int(len(f)), root_length_mm=float(req.root_length_mm),
            cres_clamped=bool("cres_clamp" in cres_report))
        STORE.put(sid, f"tooth:{tid}", {"cv": cv, "cf": cf, "bv": bv, "bf": bf,
                                        "frame": frame, "c_res": c_res,
                                        "root_length_mm": float(req.root_length_mm),
                                        "reconcile": reconcile, "dimensions": dims,
                                        "face_mask": face_mask,
                                        # Cached, not recomputed on read. _tooth_fdi
                                        # needs the ORIGINAL faces and the labels
                                        # array; storing the answer at cut time means
                                        # a reloaded client gets the same FDI the lab
                                        # manifest carries, even if /segment is rerun
                                        # afterwards with a different result.
                                        "fdi": None,   # filled immediately below
                                        "socket_rim": socket_rim_global,
                                        # The cup's own geometry, so /export can
                                        # replay it verbatim rather than rebuild
                                        # it and risk a different socket than
                                        # the one that was approved.
                                        "socket_cup_pts": cup_pts,
                                        "socket_cup_faces": cup_faces,
                                        "socket_info": cup_info})
        # The server's own FDI, read through the crown's face mask rather than
        # the raw selection, so the client's per-tooth root length and the lab
        # manifest cannot disagree. None when /segment never ran — never a guess.
        _rec = STORE.get(sid, f"tooth:{tid}")
        _rec["fdi"] = _tooth_fdi(sid, _rec, f)
        STORE.put(sid, f"tooth:{tid}", _rec)

        return {"tooth_id": tid, "watertight": True,
                # Present ONLY when the projection was clamped. An absent key
                # means the requested root length was used as given.
                "cres_clamp": cres_report.get("cres_clamp"),
                # What the cut measured but did not refuse over. Warnings here
                # are real findings — a selection in pieces, a crown that is
                # mostly gingiva, two labelled teeth in one flood — and the UI
                # is expected to show them rather than treat success as silence.
                "validation": _structured(validation.merge(
                    selection_check, pivot_check, subject="cut")),
                "crown_advisory": verdict,
                "fdi": _rec["fdi"],
                "root_length_mm": float(req.root_length_mm),
                # THE ROOT IS FABRICATED AND THE CLIENT MUST SHOW IT THAT WAY.
                # There is no root in an intraoral scan — it stops at the
                # gingival margin. This cone is the cervical rim swept to a
                # point root_length_mm apical along the tooth's own long axis:
                # an estimate drawn from two numbers, the rim and a Wheeler
                # average. It is sent as geometry so it follows the real margin
                # rather than a circle, and the client renders it wireframe at
                # low opacity in a non-tissue colour so it can never be mistaken
                # for captured data.
                "root_cone": _root_cone(v[socket_rim_global], frame,
                                        float(req.root_length_mm), arch_verts=v),
                "removed_faces": removed_faces.astype(np.int64).tolist(),
                "extracted_face_count": int(extracted.sum()),
                # A negative index -(k+1) in socket_cap.faces means appended
                # vertex k; the client resolves it against its own buffer.
                "socket_cap": {"vertices": cup_pts.tolist(),
                               "faces": socket_faces,
                               "profile": cup_info["profile"],
                               "depth_mm": cup_info["depth_mm"],
                               "depth_clamped": cup_info["depth_clamped"],
                               # WHY it fell back, with the numbers. A flat
                               # socket is usually a verdict on the selection:
                               # a ragged margin whose loop passes a tenth of a
                               # millimetre from its own centre has no room to
                               # inset, while the same tooth flooded cleanly
                               # cups. Silence here sent people looking at the
                               # geometry engine instead of at the cut.
                               "min_clearance_mm": cup_info["min_clearance_mm"],
                               "inset_mm": cup_info["inset_mm"],
                               "inset_validated": cup_info["inset_validated"],
                               "fallback_reason": cup_info["fallback_reason"]},
                "c_res": c_res.tolist(),
                "axis_source": frame["axis_source"],
                "axis_deviation_deg": frame["axis_deviation_deg"],
                "axis_corrected": frame["axis_corrected"],
                "reconcile": reconcile,
                "dimensions": dims,
                "frame": {k: (val.tolist() if hasattr(val, "tolist") else val)
                          for k, val in frame.items()},
                "crown": {"positions": np.asarray(cv, np.float32).ravel().tolist(),
                          "indices":   np.asarray(cf, np.uint32).ravel().tolist()}}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Geometry error: {str(e)}")

@app.post("/api/session/{sid}/tooth/{tid}/kinematics")
def kinematics(sid: str, tid: str, req: KinematicsRequest):
    _t_kin = time.perf_counter()
    try:
        t = STORE.require(sid, f"tooth:{tid}")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
        
    M = cg.kinematic_matrix(t["frame"], t["c_res"],
                            req.tip_deg, req.torque_deg, req.rotation_deg,
                            req.d_md, req.d_bl, req.d_oa)
                            
    moved = cg.apply_matrix(t["cv"], M)
    pen = cg.measure_interproximal_penetration(
        t["cv"], moved, t["cf"], t["bv"], t["bf"], t["frame"]["u_md"])
    
    staging = cg.staging_estimate(req.tip_deg, req.torque_deg, req.rotation_deg,
                                  req.d_md, req.d_bl, req.d_oa)

    # PER-STAGE interproximal, computed here rather than per scrub frame. The
    # timeline needs a yellow chip on the stages where an embrasure closes, and
    # measuring that during a scrub would put a KD-tree query inside a 60 FPS
    # loop. This runs once per commit, the client indexes the array, and the
    # per-frame path stays refs-and-rAF as it was.
    #
    # Stage k is the prescription x k/N rebuilt through kinematic_matrix — never
    # an interpolation of M, which is not a rotation at any intermediate t.
    n_stages = int(staging.get("stages_required") or 0)
    stage_clearance = None
    if n_stages > 0:
        mats = []
        for k in range(1, n_stages + 1):
            frac = k / n_stages
            mats.append(cg.kinematic_matrix(
                t["frame"], t["c_res"],
                req.tip_deg * frac, req.torque_deg * frac, req.rotation_deg * frac,
                req.d_md * frac, req.d_bl * frac, req.d_oa * frac))
        try:
            stage_clearance = cg.sweep_interproximal_stages(
                t["cv"], t["cf"], t["bv"], t["bf"], t["frame"]["u_md"], mats)
        except Exception as e:                       # noqa: BLE001
            # NEVER BLOCKING. A warning chip failing to compute must not stop a
            # movement from committing.
            stage_clearance = {"error": f"{type(e).__name__}: {e}", "stages": []}

    # Persist the committed pose on the tooth record. The browser owns the live
    # gizmo state, but the server must be able to rebuild the planned setup at
    # export without replaying the session, and a reloaded client must be able
    # to recover where each tooth was left.
    t["matrix"] = M
    t["clinical"] = {"tip_deg": req.tip_deg, "torque_deg": req.torque_deg,
                     "rotation_deg": req.rotation_deg, "d_md": req.d_md,
                     "d_bl": req.d_bl, "d_oa": req.d_oa}
    STORE.put(sid, f"tooth:{tid}", t)

    occlusal = _occlusal_check(req.opposing_session_id, cg.apply_matrix(t["cv"], M))

    _record(sid, audit.PRESCRIPTION_COMMITTED, arch=STORE.arch(sid), tooth_id=tid,
            fdi=t.get("fdi"),
            detail=f"{staging.get('stages_required')} stages, "
                   f"{staging.get('driver')}-driven",
            # Small, numeric, unit-bearing — the six clinical values in the
            # units the UI shows. Never the matrix.
            values={"tip_deg": req.tip_deg, "torque_deg": req.torque_deg,
                    "rotation_deg": req.rotation_deg, "d_md": req.d_md,
                    "d_bl": req.d_bl, "d_oa": req.d_oa,
                    "stages": int(staging.get("stages_required") or 0)})
    telemetry.record_span(
        "kinematics", (time.perf_counter() - _t_kin) * 1000,
        arch=STORE.arch(sid), stages=int(staging.get("stages_required") or 0),
        crown_vertices=int(len(t["cv"])),
        ipr_stages_yellow=len((stage_clearance or {}).get("stages_yellow", [])),
        occlusion_checked=occlusal is not None)

    return {"matrix": M.ravel().tolist(), "staging": staging, "clearance": pen,
            # One row per stage. `stages_yellow` is what the timeline chips.
            "stage_clearance": stage_clearance,
            "c_res": np.asarray(t["c_res"], float).tolist(),
            "axis_source": t["frame"].get("axis_source"),
            "axis_deviation_deg": t["frame"].get("axis_deviation_deg"),
            "u_bl_points_buccal": t["frame"].get("u_bl_points_buccal"),
            "occlusion": occlusal,
            # An EXPLICIT state, because `checked: false` was being read as
            # "no interference" — which is a clinical claim the software never
            # made. NOT_CHECKED, CLEAR, WARNING, INTERFERENCE and
            # COMPUTATION_ERROR are five different things and the UI must be
            # able to tell them apart.
            "occlusion_state": validation.occlusion_state(
                checked=occlusal is not None,
                max_penetration_mm=(occlusal or {}).get("max_penetration_mm"),
                # The check reports the threshold it actually applied, rather
                # than the caller assuming a constant that could drift from it.
                threshold_mm=(occlusal or {}).get("threshold_mm", 0.1)),
            "occlusal_warning": (ANTAGONIST_WARNING if occlusal and occlusal["collides"]
                                 else None)}

def _jsonable(o):
    """Recursively unwrap numpy so FastAPI can serialise a nested dict.

    The existing per-field `val.tolist() if hasattr(val, "tolist")` pattern
    unwraps exactly ONE level, which is fine for a flat frame dict and wrong for
    anything holding a dict of diagnostics — socket_info carries a (3,) normal
    two levels down, and an un-unwrapped array surfaces as an opaque 500 rather
    than a payload. Cheap, and it removes a whole class of that bug.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):       # np.float64, np.int64, np.bool_
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


class CaseSaveRequest(BaseModel):
    """Which live sessions belong to this case, plus an optional label.

    The label is the clinician's own note and MUST NOT be a patient name — the
    whole persistence design turns on the file carrying a plan, not a person.
    """
    case_id: str | None = None
    label: str = ""
    sessions: dict[str, str] = {}      # {'upper': sid, 'lower': sid}


def _case_from_sessions(sessions: dict, case_id=None, label="") -> domain.Case:
    """Build a Case by reading what the live sessions already know.

    Nothing new is computed and no geometry is copied — this reads the session
    keys that already exist and lifts the clinical decisions out of them.
    """
    c = domain.Case(label=label)
    if case_id:
        c.case_id = case_id
    for arch_name, sid in (sessions or {}).items():
        try:
            v = STORE.require(sid, "verts")
            f = STORE.require(sid, "faces")
        except SessionExpired:
            continue
        a = domain.Arch(
            arch=STORE.arch(sid), session_id=sid,
            has_occlusal_frame=STORE.get(sid, "arch_frame") is not None,
            has_segmentation=STORE.get(sid, "labels") is not None,
            scan_vertex_count=int(len(v)), scan_face_count=int(len(f)),
            scan_hash=STORE.get(sid, "scan_hash"))
        for key in STORE.keys(sid):
            if not key.startswith("tooth:"):
                continue
            t = STORE.get(sid, key)
            clin = t.get("clinical") or {}
            a.teeth[key.split(":", 1)[1]] = domain.Tooth(
                tooth_id=key.split(":", 1)[1], arch=a.arch, fdi=t.get("fdi"),
                root_length_mm=float(t["root_length_mm"]),
                c_res=np.asarray(t["c_res"], float).tolist(),
                prescription=domain.Prescription(**{
                    k: float(clin.get(k, 0.0)) for k in
                    ("tip_deg", "torque_deg", "rotation_deg", "d_md", "d_bl", "d_oa")}))
        c.arches[a.arch] = a
    return c


@app.post("/api/case")
def save_case(req: CaseSaveRequest):
    """Persist TREATMENT STATE to an encrypted local file. Opt-in, never automatic.

    The scan is not written and never will be: restoring re-attaches it by
    reloading the STL. See case_store.py for what the encryption does and does
    not protect against — it is stated there rather than implied.
    """
    c = _case_from_sessions(req.sessions, req.case_id, req.label)
    if not c.arches:
        raise HTTPException(404, "None of those sessions are still live; nothing to save.")
    try:
        path = case_store.save(c)
    except OSError as e:
        raise HTTPException(500, f"Could not write the case file: {e}")
    return {"case_id": c.case_id, "saved": True, "bytes": os.path.getsize(path),
            "tooth_count": len(c.all_teeth()), "stage_count": c.stage_count(),
            "scan_persisted": False,
            "note": "Treatment state only. Reload the arch scans to continue."}


class SessionRestoreRequest(BaseModel):
    """Rebuild live sessions for a saved case, from the scan cache."""
    case_id: str


@app.post("/api/session/restore")
def restore_sessions(req: SessionRestoreRequest):
    """Re-create in-memory sessions for a saved case WITHOUT a re-upload.

    The treatment plan records each arch's scan_hash. That hash finds the exact
    bytes in the local cache, so the mesh a plan is re-applied to is provably
    the mesh it was planned on - a random id could be re-pointed at a different
    scan and the plan would silently apply to the wrong anatomy.

    What is NOT restored here is the derived layer: crowns, sockets and root
    cones are rebuilt by replaying the cuts, exactly as they were the first
    time. Nothing is recomputed differently.
    """
    try:
        case = case_store.load(req.case_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No saved case {req.case_id!r}.")
    except ValueError as e:
        raise HTTPException(422, str(e))

    restored, missing = {}, []
    for name, arch in case.arches.items():
        h = arch.scan_hash
        if not h:
            missing.append({"arch": name, "reason": "the plan records no scan hash"})
            continue
        try:
            raw = scan_cache_manager.get(h)
        except scan_cache_manager.ScanNotCached as e:
            missing.append({"arch": name, "scan_hash": h, "reason": str(e)})
            continue

        sid = STORE.create(arch.arch)
        verts, faces = stl_io.parse_stl_bytes(raw)
        # Same gate on the restore path. The cache stores the RAW upload bytes,
        # so a scan that was damaged on arrival is still damaged on rehydration.
        verts, faces, sanity = cg.sanitize_scan(verts, faces)
        verts, faces, report = cg.condition_mesh(verts, faces)
        report["sanitize"] = sanity
        STORE.put(sid, "verts", verts)
        STORE.put(sid, "faces", faces)
        STORE.put(sid, "scan_hash", h)
        STORE.put(sid, "scan_health", cg.manifold_report(faces))
        edges = cg.directed_edges(faces)
        conc = cg.boundary_field(verts, faces, edges=edges)
        STORE.put(sid, "edges", edges)
        STORE.put(sid, "concavity", conc)
        STORE.put(sid, "graph", cg.build_barrier_graph(verts, faces, conc, edges=edges))

        arch.session_id = sid
        restored[name] = {"session_id": sid, "scan_hash": h,
                          "vertex_count": int(len(verts)),
                          "face_count": int(len(faces)),
                          "tooth_count": len(arch.teeth)}

    if not restored:
        raise HTTPException(409, {
            "error": "No arch could be restored — none of this case's scans are in "
                     "the local cache. Re-upload the original files; their hashes "
                     "must match what the plan records.",
            "missing": missing})

    case_store.save(case)   # session ids refreshed
    return {"case_id": case.case_id, "restored": restored, "missing": missing,
            "stage_count": case.stage_count(),
            "note": "Sessions are live again. Occlusal frame, segmentation labels and "
                    "committed poses are in the plan; replay the cuts to rebuild crowns."}


@app.get("/api/scans")
def list_cached_scans():
    return {"scans": scan_cache_manager.list_scans(), "stats": scan_cache_manager.stats()}


@app.delete("/api/scans/{scan_hash}")
def forget_scan(scan_hash: str):
    """Delete one cached scan. The clinician's control over their own data."""
    return {"deleted": scan_cache_manager.forget(scan_hash)}


@app.get("/api/case/{case_id}/safety")
def case_safety(case_id: str):
    """Biomechanical guardrails for a saved case: GREEN / YELLOW / RED per tooth.

    GREEN means "within the range clear aligner cases are normally planned in".
    It is NOT a statement that the movement is biologically safe - root
    resorption risk is driven by force, duration, root morphology and patient
    biology, none of which this software can see.
    """
    try:
        case = case_store.load(case_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No saved case {case_id!r}.")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _jsonable(clinical_safety.assess_case(
        [{"tooth_id": t.tooth_id, "fdi": t.fdi,
          "prescription": {k: getattr(t.prescription, k) for k in
                           ("tip_deg", "torque_deg", "rotation_deg",
                            "d_md", "d_bl", "d_oa")}}
         for t in case.all_teeth()], case.stage_count()))


@app.post("/api/case/{case_id}/report")
def clinical_report(case_id: str):
    """The clinician-facing treatment plan: JSON plus a printable HTML summary.

    Leads with what is uncertain - teeth not yet reviewed, movements outside the
    envelope, checks that did not run - rather than burying it. Carries no
    patient identifier.
    """
    try:
        case = case_store.load(case_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No saved case {case_id!r}.")
    except ValueError as e:
        raise HTTPException(422, str(e))

    space = None
    for arch in case.arches.values():
        if arch.session_id and STORE.get(arch.session_id, "verts") is not None:
            try:
                space = space_analysis_report(arch.session_id)
                break
            except HTTPException:
                pass   # no occlusal frame yet; the report simply omits IPR

    report = export_clinical_report.build(case, space_analysis=space)
    out_dir = os.path.join(CURRENT_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"plan_{case.case_id}")
    return {
        "report": _jsonable(report),
        "json_path": export_clinical_report.write_json(report, base + ".json"),
        "html_path": export_clinical_report.write_html(report, base + ".html"),
        "disclaimer": export_clinical_report.DISCLAIMER,
    }


@app.get("/api/case")
def list_cases():
    return {"cases": case_store.list_cases()}


@app.get("/api/case/{case_id}")
def get_case(case_id: str):
    try:
        c = case_store.load(case_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No saved case {case_id!r}.")
    except ValueError as e:
        # Failed its integrity check, or a newer schema. Refuse loudly.
        raise HTTPException(422, str(e))
    d = c.to_dict()
    d["binding_teeth"] = c.binding_teeth()
    d["needs_scan_reload"] = True
    return d


@app.delete("/api/case/{case_id}")
def delete_case(case_id: str):
    return {"deleted": bool(case_store.delete(case_id))}


@app.get("/api/session/{sid}")
def get_session(sid: str):
    """Is this session still alive, and what does it already know?

    The first call a reloading client makes. A browser refresh loses every
    piece of React state but the session id can be persisted, and from that id
    everything below is recoverable — so this answers "can I restore, and how
    far along was I?" in one round trip.

    Deliberately cheap: no geometry, no labels. Just the shape of the case.
    """
    try:
        v = STORE.require(sid, "verts")
        f = STORE.require(sid, "faces")
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    n_teeth = sum(1 for k in STORE.keys(sid) if k.startswith("tooth:"))
    extracted = STORE.get(sid, "extracted_faces")
    labels = STORE.get(sid, "labels")
    return {
        "session_id": sid,
        "arch": STORE.arch(sid),
        "vertex_count": int(len(v)),
        "face_count": int(len(f)),
        "bbox": [v.min(0).tolist(), v.max(0).tolist()],
        "scan_health": STORE.get(sid, "scan_health"),
        # What has been done to it so far. The client uses these to decide
        # which workflow steps to mark complete without fetching the payloads.
        "has_occlusal_frame": STORE.get(sid, "arch_frame") is not None,
        "has_segmentation": labels is not None,
        "tooth_count": n_teeth,
        "extracted_face_count": 0 if extracted is None else int(extracted.sum()),
    }


@app.get("/api/session/{sid}/segmentation-review")
def segmentation_review_report(sid: str):
    """Per-tooth verdict on the last /segment run: PASS, REVIEW or FAIL.

    /segment returns a per-vertex FDI array and the client colours the arch with
    it. Until now nothing between the model and the clinician ever asked whether
    a region is plausibly one tooth, and ToothCandidate.confidence - which has
    existed since the segmentation package was written - is assigned nowhere, so
    its needs_review flag was unconditionally True.

    The confidence returned here is NOT a model probability. ToothGroupNetwork
    emits no calibrated uncertainty and inventing one would be worse than having
    none; this is weighted agreement over independently checkable geometric
    facts, and every contributing factor comes back with the score.
    """
    try:
        v = STORE.require(sid, "verts")
        f = STORE.require(sid, "faces")
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    labels = STORE.get(sid, "labels")
    if labels is None:
        raise HTTPException(409,
            "Segmentation has not been run for this session, so there is nothing "
            "to review. POST /api/session/{sid}/segment first.")

    af = STORE.get(sid, "arch_frame")
    occ = np.asarray(af["u_occ"], float) if af else None
    out = segmentation_review.review_segmentation(
        labels, v, f, STORE.arch(sid),
        arch_centre=(v.mean(axis=0) if af is not None else None),
        occlusal_axis=occ)
    return _jsonable(out)


class AttachmentRequest(BaseModel):
    """What the viewport sends when a clinician clicks a crown surface."""
    tooth_id: str
    type: str
    position_xyz: list[float]
    normal_xyz: list[float]
    dimensions_hwd: dict | None = None      # {height, width, depth} in mm
    rotation_deg: float = 0.0


@app.get("/api/attachments/shapes")
def attachment_shapes():
    """The catalogue, with each shape's mechanical purpose and the size band."""
    return {"shapes": attachments_mod.SHAPES,
            "limits_mm": {"min": attachments_mod.MIN_DIM_MM,
                          "max": attachments_mod.MAX_DIM_MM},
            "note": "Orientation is the mechanical point. An attachment is built in "
                    "the TOOTH'S anatomical frame, so 'vertical' means along that "
                    "tooth's long axis, not along world Z."}


@app.post("/api/session/{sid}/tooth/{tid}/attachment")
def place_attachment(sid: str, tid: str, req: AttachmentRequest):
    """Fuse a composite attachment onto a crown, before staging.

    The attachment moves WITH the crown, so it must be part of the solid the
    stage models are built from. Bonding it to an already-moved crown would put
    it where the tooth ends up rather than where the clinician placed it.
    """
    try:
        STORE.require(sid, "verts")
        tooth = STORE.get(sid, f"tooth:{tid}")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    if tooth is None:
        raise HTTPException(404, f"No extracted tooth {tid!r} in this session.")

    hwd = req.dimensions_hwd or {}
    size = {}
    if hwd:
        # The viewport speaks height/width/depth; the geometry speaks the tooth's
        # own axes. Mapping them here rather than in the client keeps one
        # vocabulary at the boundary.
        size = {"oa": float(hwd.get("height", 3.0)),
                "md": float(hwd.get("width", 2.0)),
                "bl": float(hwd.get("depth", 1.0))}

    try:
        att = attachments_mod.build_attachment(
            req.type, tooth["frame"], req.position_xyz, size_mm=size or None)
        fused = attachments_mod.fuse_to_crown(tooth["cv"], tooth["cf"], att)
    except ValueError as e:
        # Unknown shape, implausible dimensions, or not touching the crown. Each
        # names its own cause; none is a 500.
        raise HTTPException(422, str(e))

    existing = list(tooth.get("attachments") or [])
    record = {
        "attachment_id": uuid.uuid4().hex[:8],
        "shape": req.type,
        "dimensions_mm": att["dimensions_mm"],
        "purpose": att["purpose"],
        "position_xyz": [float(x) for x in req.position_xyz],
        "normal_xyz": [float(x) for x in req.normal_xyz],
        "rotation_deg": float(req.rotation_deg),
        "fused_volume_mm3": fused["volume_mm3"],
    }
    existing.append(record)
    tooth["attachments"] = existing
    # The fused solid replaces the crown for manufacturing. The ORIGINAL crown
    # is kept so an attachment can be removed without re-cutting the tooth.
    tooth.setdefault("crown_without_attachments", {"cv": tooth["cv"], "cf": tooth["cf"]})
    tooth["cv"], tooth["cf"] = fused["verts"], fused["faces"]
    STORE.put(sid, f"tooth:{tid}", tooth)

    return {"attachment": record, "attachments": existing,
            "crown": {"positions": np.asarray(fused["verts"], np.float32).ravel().tolist(),
                      "indices": np.asarray(fused["faces"], np.uint32).ravel().tolist()},
            "bodies": fused["bodies"], "volume_mm3": fused["volume_mm3"]}


@app.delete("/api/session/{sid}/tooth/{tid}/attachment")
def clear_attachments(sid: str, tid: str):
    """Restore the crown as it was cut. Removing one attachment from a fused
    solid is not a boolean subtraction of the same block - the union has already
    merged the surfaces - so the honest operation is to go back to the original
    and re-place whichever attachments are still wanted."""
    try:
        STORE.require(sid, "verts")
        tooth = STORE.get(sid, f"tooth:{tid}")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    if tooth is None:
        raise HTTPException(404, f"No extracted tooth {tid!r}.")

    orig = tooth.get("crown_without_attachments")
    if not orig:
        return {"cleared": 0, "note": "This crown has no attachments."}
    tooth["cv"], tooth["cf"] = orig["cv"], orig["cf"]
    n = len(tooth.get("attachments") or [])
    tooth["attachments"] = []
    tooth.pop("crown_without_attachments", None)
    STORE.put(sid, f"tooth:{tid}", tooth)
    return {"cleared": n,
            "crown": {"positions": np.asarray(orig["cv"], np.float32).ravel().tolist(),
                      "indices": np.asarray(orig["cf"], np.uint32).ravel().tolist()}}


@app.get("/api/session/{sid}/space-analysis")
def space_analysis_report(sid: str, tolerance_mm: float = space_analysis.DEFAULT_WIDTH_TOLERANCE_MM,
                          noise_floor_mm: float = space_analysis.NOISE_FLOOR_MM,
                          contact_mm: float = space_analysis.CONTACT_MM):
    """Crown widths against Wheeler, and interproximal clearance between crowns.

    Wires up measuring machinery that has existed since the caliper work and was
    called by nothing but a test: local_arch_tangent, measure_mesiodistal_width
    and validate_against_anatomy. The arithmetic is unchanged; this is the wiring.

    Thresholds are query parameters because they are judgements about how much
    individual variation to tolerate, not constants. Nothing here blocks an
    export — a REVIEW flag means look at the segmentation.
    """
    try:
        STORE.require(sid, "verts")
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    af = STORE.get(sid, "arch_frame")
    if af is None:
        raise HTTPException(409,
            "Space analysis needs the occlusal reference: the mesiodistal "
            "direction is derived per tooth from the arch curve, which is "
            "defined in that frame. Establish the occlusal plane first.")

    teeth = []
    for key in STORE.keys(sid):
        if not key.startswith("tooth:"):
            continue
        t = STORE.get(sid, key)
        # THE POSED CROWN, NOT T0. This read `t["cv"]` — the crown exactly as it
        # was cut — so every contact reported here described the malocclusion the
        # clinician started with, not the setup they were looking at. Space
        # analysis exists to answer "does this plan fit", and measuring T0
        # answers a question nobody asked while looking like it answered theirs.
        M = t.get("matrix")
        cv = t["cv"] if M is None else cg.apply_matrix(t["cv"], np.asarray(M, float))
        teeth.append({"tooth_id": key.split(":", 1)[1], "fdi": t.get("fdi"),
                      "verts": cv, "posed": M is not None})
    if not teeth:
        return {"crowns_measured": 0, "widths": {"teeth": []},
                "interproximal": {"contacts": []},
                "summary": "No crowns have been extracted yet."}

    v = STORE.require(sid, "verts")
    out = space_analysis.analyse(
        teeth, STORE.arch(sid), np.asarray(af["u_occ"], float), v.mean(axis=0),
        tolerance_mm=tolerance_mm, noise_floor_mm=noise_floor_mm,
        contact_mm=contact_mm)
    return _jsonable(out)


@app.get("/api/session/{sid}/frame")
def get_occlusal_frame(sid: str):
    """The occlusal reference frame, if one has been established.

    Stored since the plane was set, but until now only POST /occlusal-plane
    ever returned it — so a refresh meant re-clicking three landmarks and
    getting a DIFFERENT frame, which silently re-bases every long-axis
    reconciliation in the case.
    """
    try:
        STORE.require(sid, "verts")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    frame = STORE.get(sid, "arch_frame")
    if frame is None:
        raise HTTPException(404, "No occlusal plane has been established for this session.")
    return arch_frame.to_json(frame)


@app.get("/api/session/{sid}/labels")
def get_labels(sid: str):
    """Per-vertex FDI labels from the last /segment run.

    Stored since segmentation, but only POST /segment ever returned them — so
    a refresh cost a ~4 minute AI pass to recover data the server already had.
    """
    try:
        STORE.require(sid, "verts")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    labels = STORE.get(sid, "labels")
    if labels is None:
        raise HTTPException(404, "Segmentation has not been run for this session.")
    return {"labels": np.asarray(labels).astype(int).tolist(),
            "jaw": jaw_naming.jaw_for_arch(STORE.arch(sid))}


@app.get("/api/segmentation/providers")
def segmentation_provider_registry():
    """Which segmentation backends exist, and the state of each.

    A CANDIDATE IS NOT A CHOICE. `meshsegnet` is listed with
    `available: false` and its audit attached, so the UI can show what has
    been considered without offering a backend that would raise.
    """
    return {"default": segmentation_providers.DEFAULT_PROVIDER,
            "providers": segmentation_providers.registry()}


@app.get("/api/session/{sid}/segmentation-diagnostics")
def segmentation_spatial_diagnostics(sid: str, recompute: bool = False):
    """Per-tooth bounding box, counts and disconnected components.

    THE QUESTION THIS ANSWERS IS NOT "is the model good". It is whether the
    labels are attached to this mesh at all - a failure no metric computed on
    the labels alone can see, IoU included, because a mis-indexed array still
    scores against itself perfectly.
    """
    try:
        v = STORE.require(sid, "verts")
        f = STORE.require(sid, "faces")
    except SessionExpired as e:
        raise HTTPException(404, str(e))
    labels = STORE.get(sid, "labels")
    if labels is None:
        raise HTTPException(404, "Segmentation has not been run for this session.")
    cached = STORE.get(sid, "segmentation_diagnostics")
    if cached is not None and not recompute:
        return _jsonable(cached)
    rep = segmentation_diagnostics.label_report(v, f, labels)
    STORE.put(sid, "segmentation_diagnostics", rep)
    return _jsonable(rep)


@app.get("/api/session/{sid}/teeth")
def list_teeth(sid: str, geometry: bool = False):
    """Every extracted tooth and its committed pose.

    `geometry=false` (default) is the light pose-only payload this endpoint has
    always returned — kept byte-compatible so nothing that already calls it
    breaks.

    `geometry=true` adds everything a reloading client needs to REBUILD the
    scene: crown meshes, the virtual root cone, the socket cup in the client's
    own dual-index format, and the extracted face indices. Without it a refresh
    could restore poses but had no crowns to apply them to, which is why the
    client never called this endpoint at all.

    It is heavy on purpose — a 14-crown case is several MB of JSON — so it is
    opt-in and fetched once on restore, not polled.
    """
    try:
        STORE.require(sid, "verts")
        faces = STORE.require(sid, "faces")
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    extracted = STORE.get(sid, "extracted_faces")
    out = []
    for key in STORE.keys(sid):
        if not key.startswith("tooth:"):
            continue
        t = STORE.get(sid, key)
        M = t.get("matrix")
        rec = {
            "tooth_id": key.split(":", 1)[1],
            "c_res": np.asarray(t["c_res"], float).tolist(),
            "clinical": t.get("clinical"),
            "matrix": None if M is None else np.asarray(M, float).ravel().tolist(),
            "frame": {k: (val.tolist() if hasattr(val, "tolist") else val)
                      for k, val in t["frame"].items()},
        }
        if geometry:
            cv, cf = t["cv"], t["cf"]
            rim_idx = np.asarray(t["socket_rim"], dtype=np.int64)
            n_rim = len(rim_idx)
            # socket_rim holds GLOBAL vertex ids into the original scan, which
            # is never mutated — so this resolves the same rim the cut saw.
            rim_xyz = STORE.require(sid, "verts")[rim_idx]
            rec.update({
                "fdi": t.get("fdi"),
                "root_length_mm": float(t["root_length_mm"]),
                "dimensions": _jsonable(t.get("dimensions")),
                "reconcile": _jsonable(t.get("reconcile")),
                "crown": {"positions": np.asarray(cv, np.float32).ravel().tolist(),
                          "indices":   np.asarray(cf, np.uint32).ravel().tolist()},
                "root_cone": _root_cone(rim_xyz, t["frame"], float(t["root_length_mm"]),
                                        arch_verts=STORE.require(sid, "verts")),
                # Same dual-index convention /cut uses: a non-negative entry is
                # an arch vertex id, -(k+1) is appended cup vertex k.
                "socket_cap": {
                    "vertices": np.asarray(t["socket_cup_pts"]).tolist(),
                    "faces": [[int(rim_idx[i]) if i < n_rim else -(int(i) - n_rim + 1)
                               for i in tri] for tri in t["socket_cup_faces"]],
                    "info": _jsonable(t.get("socket_info")),
                },
                # WHICH faces this tooth removed, not just how many. The client
                # rebuilds its liveFaces mask from these.
                "removed_faces": np.where(np.asarray(t["face_mask"]))[0].astype(np.int64).tolist(),
            })
        out.append(rec)

    return {"teeth": out,
            "extracted_face_count": 0 if extracted is None else int(extracted.sum()),
            "face_count": int(len(faces))}


def _root_cone(rim_xyz: np.ndarray, frame: dict, root_length_mm: float,
               arch_verts: np.ndarray | None = None) -> dict:
    """The cervical rim swept to an apex root_length_mm apical along u_oa.

    World coordinates, so the client can parent it beside the crown and drive
    both with the same matrix. Returned as a ring plus an apex with a triangle
    fan, which is a surface rather than a solid — it is a visual aid, never
    something that reaches a printable file, and nothing in /export looks at it.

    The apex is placed from the RIM CENTROID, matching center_of_resistance's
    own construction (rim_centroid + u_oa*(h - root_length)), so the drawn root
    and the pivot the tooth actually rotates about are derived the same way
    rather than two independent guesses about where the apex is.

    THE DRAWN APEX IS CLAMPED TO THE SCAN'S OWN APICAL EXTENT, and the clamp is
    reported. A Wheeler root length is an average for a tooth TYPE; the patient's
    alveolar housing is not consulted because this software has no CBCT. On a
    short-rooted case or a shallow scan the 13mm canine cone therefore projects
    out through the back of the model into empty space, and a violet cone hanging
    below the cast looks like a finding rather than an artefact.

    The bound used is the deepest point the SCAN has data for ANYWHERE, measured
    along the tooth's own long axis from the rim centroid. It is not an alveolar
    crest — nothing here can measure one — and `clamped` plus `requested_mm` say
    exactly that, so the drawing never implies anatomy that was not imaged.

    A GLOBAL EXTENT, NOT A LOCAL ONE, AND THAT WAS MEASURED. The obvious choice
    is the scan's depth in a cylinder around this tooth, and it is wrong: on the
    real mandibular scan, sampling 12 ridge points, the local depth within 6mm
    is a median 12.24mm and would clamp SEVEN OF TWELVE canines at Wheeler's
    13mm — it measures the vestibular depth, which is normal anatomy, not a
    defect. The global extent is a median 14.58mm and clamps none of 12 at 9,
    10 or 13mm. So it fires only when the scan genuinely holds no data that deep
    anywhere, which is a cropped or shallow scan — the artefact actually worth
    flagging. A warning that fires on healthy anatomy is the same as no warning.

    C_res IS NOT AFFECTED. The pivot is the biomechanical quantity and it is
    derived in core_geometry from the requested root length; clamping a drawing
    must never quietly move the axis a tooth rotates about.
    """
    rim = np.asarray(rim_xyz, float)
    u_oa = np.asarray(frame["u_oa"], float)
    u_oa = u_oa / np.linalg.norm(u_oa)
    centroid = np.asarray(frame.get("rim_centroid", rim.mean(axis=0)), float)

    requested = float(root_length_mm)
    drawn, clamped, available = requested, False, None
    if arch_verts is not None and len(arch_verts):
        # Signed projection onto u_oa relative to the rim centroid: occlusal is
        # positive, apical negative. The scan's most negative value is as deep
        # as the model goes.
        t = (np.asarray(arch_verts, float) - centroid) @ u_oa
        available = float(-t.min())
        if np.isfinite(available) and 0.0 < available < requested:
            drawn, clamped = available, True

    apex = centroid - u_oa * drawn

    n = len(rim)
    verts = np.vstack([rim, apex])
    faces = [[i, (i + 1) % n, n] for i in range(n)]
    out = {"vertices": verts.tolist(), "faces": faces,
           "apex": apex.tolist(), "length_mm": float(drawn),
           "requested_mm": requested, "clamped": bool(clamped),
           "label": "virtual root (estimated)"}
    if available is not None:
        out["scan_apical_extent_mm"] = round(available, 3)
    if clamped:
        out["clamp_note"] = (
            f"[Virtual Root] Projection constrained to alveolar boundary — "
            f"{requested:.1f}mm requested, {drawn:.1f}mm drawn (the scan has no "
            f"data deeper than that). C_res is unchanged and still uses "
            f"{requested:.1f}mm.")
    return out


ANTAGONIST_WARNING = "Warning: Trajectory creates occlusal interference with antagonist."


def _antagonist(opposing_sid: str | None):
    """(verts, vertex_normals) of the opposing arch, or None.

    Normals are computed once and cached on the opposing session: a 95k-vertex
    arch costs a full pass to normal, and the staging export asks for the same
    antagonist once per tooth per stage.

    Returns None — never raises — when the id is absent, unknown or expired. A
    missing antagonist means the clinician has not loaded the opposing arch, not
    that anything is wrong, and it must never block a movement or an export.
    """
    if not opposing_sid:
        return None
    try:
        ov = STORE.require(opposing_sid, "verts")
        of = STORE.require(opposing_sid, "faces")
    except (SessionExpired, KeyError):
        return None
    normals = STORE.get(opposing_sid, "vertex_normals")
    if normals is None or len(normals) != len(ov):
        normals = cg.vertex_normals(ov, of)
        STORE.put(opposing_sid, "vertex_normals", normals)
    index = STORE.get(opposing_sid, "antagonist_index")
    if index is None:
        index = cg.build_antagonist_index(ov)
        STORE.put(opposing_sid, "antagonist_index", index)
    return ov, normals, index


def _occlusal_check(opposing_sid: str | None, crown_xyz: np.ndarray):
    """Collision of one posed crown against the opposing arch, or None."""
    ant = _antagonist(opposing_sid)
    if ant is None:
        return None
    return cg.check_occlusal_collision(crown_xyz, ant[0], ant[1], index=ant[2])


def _tooth_fdi(sid: str, tooth: dict, faces: np.ndarray):
    """The tooth's FDI number, or None.

    Derived from the per-vertex labels /segment stores, via the crown's own
    face mask. Returns None when segmentation was never run — a wrong tooth
    number on a lab manifest is worse than an absent one, so this never guesses.
    """
    labels = STORE.get(sid, "labels")
    mask = tooth.get("face_mask")
    if labels is None or mask is None:
        return None
    ids = np.unique(faces[mask])
    lab = np.asarray(labels)[ids]
    lab = lab[lab > 0]
    if lab.size == 0:
        return None
    vals, counts = np.unique(lab, return_counts=True)
    return int(vals[counts.argmax()])


def _seal_sockets(v: np.ndarray, f: np.ndarray, sid: str, extracted: np.ndarray,
                  flush: bool = False):
    """Close every extracted tooth's socket with the cup the clinician approved.

    `flush=True` rebuilds each socket at ZERO depth instead — a flat cap on the
    margin plane rather than a 3.5mm alveolus. That is for the MANUFACTURING
    export and only for it. A staged model is the gingiva with the teeth in
    their planned positions, and a tooth that has translated out of its socket
    leaves the cup gaping: the STL is still watertight, but the cast has a pit
    beside every moved tooth and a thermoformed aligner would take an impression
    of it. Filled flush, the tooth simply sits proud of smooth gingiva, which is
    what a setup model looks like.

    /cut stores the cup it sent to the browser; this replays those exact
    triangles into the export mesh. Rebuilding them here would be cheap (43.6ms
    at 150k faces) and subtly wrong — the depth clamp raycasts against the cast,
    and the cast has changed since, so a rebuilt cup could sit at a different
    depth than the one on screen. CLAUDE.md section 9 recorded "the displayed
    socket cup and the printed socket are different geometry" as an open issue;
    replaying is what closes it.

    Manifoldness is preserved by construction: each socket rim edge already has
    exactly one face on the arch, and the cup contributes the second.

    ORIENTATION HAS TO BE DECIDED HERE, not taken from the cup. build_socket_cup
    winds itself so its floor faces occlusally, which is right for a cup drawn
    on its own and says nothing about the arch it is being sewn into: the two
    faces on a shared rim edge must traverse it in OPPOSITE directions, and
    whether they do depends on the scan's winding. Measured on a real
    mandibular scan, the base closed with zero open and zero non-manifold edges
    and still had normals pointing into the solid — which a boolean engine
    answers with a plausible wrong result rather than an error.

    The decision is global per cup, taken from the first triangle that shares a
    rim edge. Per-triangle orientation would be wrong here for the reason
    CLAUDE.md section 8 records: a cup's side walls are near-parallel to the
    long axis, so any geometric test on them is noise. This test is
    combinatorial, and the cup is internally consistent, so one edge settles it.
    """
    verts = v.copy()
    faces = [f[~extracted]]
    sealed = []
    # Directed boundary edges of the socketed arch — every socket's rim at once,
    # in the direction the surviving arch face traverses it.
    half = cg._boundary_half_edges(f[~extracted])
    for key in STORE.keys(sid):
        if not key.startswith("tooth:"):
            continue
        t = STORE.get(sid, key)
        rim = t.get("socket_rim")
        pts, cup = t.get("socket_cup_pts"), t.get("socket_cup_faces")
        if rim is None or pts is None or cup is None:
            continue                      # cut before cups were retained
        rim = np.asarray(rim, np.int64)
        if flush:
            # Rebuild flat on the margin plane. depth 0 is already a tested path
            # in build_socket_cup ("depth 0 gives a flat cap at the margin, one
            # loop and manifold"), so this is the same code, not a second one.
            pts, cup, _flat = cg.build_socket_cup(
                v[rim], np.asarray(t["frame"]["u_oa"], float), depth_mm=0.0)
        n_rim = len(rim)
        offset = len(verts)
        verts = np.vstack([verts, np.asarray(pts, float)])
        cup = np.asarray(cup, np.int64)
        # 0..n_rim-1 index the rim in the ORIGINAL mesh; n_rim.. are the cup's
        # own interior points, which have just landed at `offset`.
        remapped = np.where(cup < n_rim, rim[np.clip(cup, 0, n_rim - 1)],
                            cup - n_rim + offset)

        flip = None
        for tri in remapped:
            a, b, c_ = int(tri[0]), int(tri[1]), int(tri[2])
            for u, w in ((a, b), (b, c_), (c_, a)):
                if (u, w) in half:
                    flip = True            # same direction as the arch -> wrong
                    break
                if (w, u) in half:
                    flip = False           # already opposed
                    break
            if flip is not None:
                break
        if flip:
            remapped = remapped[:, ::-1]
        faces.append(remapped)
        sealed.append({"tooth_id": key.split(":", 1)[1],
                       "profile": (t.get("socket_info") or {}).get("profile"),
                       "depth_mm": (t.get("socket_info") or {}).get("depth_mm"),
                       "min_clearance_mm": (t.get("socket_info") or {}).get("min_clearance_mm"),
                       "inset_mm": (t.get("socket_info") or {}).get("inset_mm"),
                       "inset_validated": (t.get("socket_info") or {}).get("inset_validated"),
                       "fallback_reason": (t.get("socket_info") or {}).get("fallback_reason"),
                       "triangles": int(len(cup))})

    # Weld before anything measures this mesh. An STL reader welds on load, so a
    # base validated unwelded is not the base the lab receives — see
    # cg.weld_vertices for the case that forced this.
    verts, faces, welded = cg.weld_vertices(verts, np.vstack(faces))
    if welded:
        for s in sealed:
            s["welded_vertices"] = int(welded)
    return verts, faces.astype(np.int64), sealed


def build_export_bundle(sid: str, req: ExportRequest) -> dict:
    """The planned setup as a self-describing ZIP, in memory.

    THIS is where watertightness is established, and deliberately not in /cut.

    THE BASE IS BUILT BY TRIM -> EXTRUDE, NEVER BY cap_and_close, and the reason
    is not performance. An intraoral scan is an OPEN SHELL whose perimeter is a
    horseshoe; measured on a real mandibular scan, condition_mesh reports
    open_edges 2101, nonmanifold_edges 0 and exactly one hole — those 2101 edges
    are the edge of the scanned region, not a defect. cap_and_close sealed that
    perimeter by fanning across it, which in an STL viewer is a web of long thin
    triangles stretched over the tongue space. Topologically closed,
    anatomically a membrane, and unprintable.

    cg.trim_to_arch cuts the scan to a horseshoe band and cg.build_cast_base
    extrudes it to a flat-bottomed solid, asserting zero open and zero
    non-manifold edges directly. Measured 3.1s for the pair at 187,625 faces,
    against 100.2s for cap_boundary_loop on that one perimeter loop alone.

    cap_and_close is still right for the CROWN, where the cervical rim genuinely
    is a hole to fill, and /cut still uses it there.

    Split out from the endpoint so it can be tested by calling it, rather than
    by draining a StreamingResponse's async iterator.
    """
    try:
        v, f = STORE.require(sid, "verts"), STORE.require(sid, "faces")
        arch = STORE.arch(sid)
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    extracted = STORE.get(sid, "extracted_faces")
    if extracted is None or not extracted.any():
        raise HTTPException(400, "Nothing has been extracted yet; there is no setup to export.")

    af = STORE.get(sid, "arch_frame")
    if af is None:
        # Same refusal as /cut, for the same reason: the trim measures distance
        # to the occlusal ridge and the base is extruded along the occlusal
        # normal. Without the plane both are guesses, and a guessed trim throws
        # away real dentition.
        raise HTTPException(409,
            "Occlusal plane not defined for this session, so the cast base cannot be "
            "built — the trim measures from the occlusal ridge and the base is extruded "
            "along the occlusal normal. Establish it first (POST /api/session/{sid}/"
            "occlusal-plane with the left posterior, right posterior and anterior "
            "midline landmarks).")

    sealed_v, sealed_f, sockets = _seal_sockets(v, f, sid, extracted)
    try:
        # Fit the arch curve on the ORIGINAL scan, before anything was cut. The
        # curve is read off the occlusal ridge, and by now the extracted teeth
        # are gone — refitting on the socketed mesh leaves the ridge missing
        # exactly where the teeth were, and the trim then discards the sockets
        # themselves. Measured on a two-tooth fixture: 351 of 423 cup vertices.
        curve, _ = cg.fit_arch_curve(v, af)
        tv, tf, trim_info = cg.trim_to_arch(sealed_v, sealed_f, af,
                                            margin_mm=req.trim_margin_mm,
                                            curve=curve)
        tv, tf, rim, uinfo = cg.clear_undercut_periphery(tv, tf, af, trim_info["rim_loop"]); bv, bf, base_info = cg.build_cast_base(tv, tf, af,
                                               base_thickness_mm=req.base_thickness_mm,
                                               rim=rim)
    except ValueError as e:
        # build_cast_base's own assertions name the failing check and the edge
        # count. There is deliberately no override: a correctly built cast base
        # has no reason to be unsealed, and an escape hatch here would hide the
        # very failures these assertions exist to surface.
        raise HTTPException(422, f"The cast base could not be built. {e}")

    health = cg.manifold_report(bf)
    if not health["watertight"]:
        raise HTTPException(422,
            f"The cast base closed its own assertions but re-measures as not watertight: "
            f"{health['open_edges']} open and {health['nonmanifold_edges']} non-manifold "
            f"of {health['total_edges']} edges. This is a bug in build_cast_base, not "
            f"something the operator can work around.")

    base_name = f"{arch}_Base_Socketed.stl"
    base_blob = cg.write_binary_stl_bytes(bv, bf)

    # Validate the FILE, not the mesh in memory. Binary STL stores positions at
    # float32, and every reader welds on load, so a base that measures watertight
    # as an index buffer can still arrive at the lab non-manifold — measured, 3
    # such edges from two socket-floor vertices that shared a position. This is
    # the only check that speaks about what actually leaves the building.
    _, rf = stl_io.parse_stl_bytes(base_blob)
    written = cg.manifold_report(rf)
    if not written["watertight"]:
        raise HTTPException(422,
            f"The cast base is watertight in memory but not as written: re-reading the "
            f"STL gives {written['open_edges']} open and {written['nonmanifold_edges']} "
            f"non-manifold edges of {written['total_edges']}. STL stores positions at "
            f"float32 and readers weld on load, so two vertices that share a position "
            f"merge and take their faces with them. This is a bug in the base builder, "
            f"not something the operator can work around.")

    blobs = {base_name: base_blob}
    files = [{"role": "base", "name": base_name, "triangles": int(len(bf))}]
    manifest_teeth = []

    for key in STORE.keys(sid):
        if not key.startswith("tooth:"):
            continue
        tid = key.split(":", 1)[1]
        t = STORE.get(sid, key)
        M = t.get("matrix")
        # Bake the planned movement into the exported crown. Vertices, not a
        # separate transform: the printer gets geometry, and a detached
        # transform is the desync that corrupted the earlier build.
        cv = t["cv"] if M is None else cg.apply_matrix(t["cv"], np.asarray(M, float))
        name = f"{arch}_Tooth_{tid}.stl"
        blobs[name] = cg.write_binary_stl_bytes(cv, t["cf"])
        files.append({"role": "crown", "tooth_id": tid, "name": name,
                      "moved": M is not None, "triangles": int(len(t["cf"]))})

        clinical = t.get("clinical")
        staging = cg.staging_estimate(**clinical) if clinical else None
        clearance = None
        if M is not None:
            clearance = cg.measure_interproximal_penetration(
                t["cv"], cv, t["cf"], t["bv"], t["bf"], t["frame"]["u_md"])
        manifest_teeth.append({
            "tooth_id": tid,
            "fdi": _tooth_fdi(sid, t, f),
            "file": name,
            "moved": M is not None,
            "prescription": clinical,
            "staging": staging,
            "clearance": clearance,
            "root_length_mm": t.get("root_length_mm"),
            "c_res": np.asarray(t["c_res"], float).tolist(),
        })

    manifest = {
        "generator": "Clinical Micro-Planner",
        "arch": arch,
        "base_file": base_name,
        "base_watertight": bool(health["watertight"]),
        "base_health": health,
        # The same measurement taken on the STL after a round-trip through
        # float32 and a weld, i.e. on what a slicer will actually load.
        "base_health_as_written": written,
        "scan_health_at_upload": STORE.get(sid, "scan_health"),
        "faces_removed": int(extracted.sum()),
        "units": "mm",
        "coordinate_space": "raw scanner coordinates — never re-centred or rescaled",
        # How the base was made, so a lab arguing with the geometry can see the
        # parameters rather than guess them.
        "base_construction": "trim_to_arch -> build_cast_base (never cap_and_close)",
        "trim": {k: val for k, val in trim_info.items() if k not in ("curve", "rim_loop")},
        "cast_base": {k: val for k, val in base_info.items() if k != "prune"},
        "sockets": sockets,
        "teeth": manifest_teeth,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, blob in blobs.items():
            z.writestr(name, blob)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    buf.seek(0)

    out_dir = None
    if req.keep_local:
        out_dir = str(_confined_out_dir(req.out_dir or os.path.join(
            EXPORTS_ROOT, sid[:8])))
        os.makedirs(out_dir, exist_ok=True)
        for name, blob in blobs.items():
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(blob)
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

    return {"buf": buf, "filename": f"{arch}_setup.zip", "files": files,
            "manifest": manifest, "out_dir": out_dir,
            "base_watertight": bool(health["watertight"]),
            "base_health": health, "faces_removed": int(extracted.sum()),
            "trim": trim_info, "cast_base": base_info, "sockets": sockets}


@app.post("/api/session/{sid}/export")
def export_setup(sid: str, req: ExportRequest):
    """Stream the planned setup to the browser as a ZIP.

    Stays a sync `def` on purpose. FastAPI already dispatches sync endpoints to
    a worker thread, so this is off the event loop as written; wrapping it in
    run_in_threadpool would add ceremony and fix nothing. The wait is real but
    much shorter than it was — trim + extrude measures 3.1s at 187k faces where
    cap_and_close's Delaunay took over 100s on the same perimeter — and the cure
    for the rest is the honest progress message the client shows, not a second
    thread.
    """
    _t_exp = time.perf_counter()
    try:
        # The span wraps the BUILD, not the stream. StreamingResponse returns
        # immediately and the body is consumed afterwards, so timing the return
        # would record a few microseconds for a 7.5s operation.
        with telemetry.span("export", arch=STORE.arch(sid)) as _sp:
            bundle = build_export_bundle(sid, req)
            _sp.set(files=len(bundle["files"]),
                    seconds=round(time.perf_counter() - _t_exp, 3))
        _record(sid, audit.EXPORTED, arch=STORE.arch(sid),
                detail=f"printable cast, {len(bundle['files'])} file(s)",
                values={"files": len(bundle["files"]),
                        "seconds": round(time.perf_counter() - _t_exp, 2)})
        return StreamingResponse(
            bundle["buf"], media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{bundle["filename"]}"',
                     "X-Export-Files": str(len(bundle["files"]))})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Export failed: {str(e)}")


class StageExportRequest(BaseModel):
    out_dir: str | None = None
    keep_local: bool = False
    trim_margin_mm: float = cg.ARCH_TRIM_MARGIN_MM
    base_thickness_mm: float = cg.CAST_BASE_THICKNESS_MM
    # Cap the run. 31 stages x 14 crowns is ~70s of boolean work; a mis-typed
    # 7.7mm extrusion can ask for far more, and the clinician should be told
    # rather than left watching a spinner.
    max_stages: int = 64
    # The opposing arch's session, if loaded. See KinematicsRequest — the two
    # arches are separate sessions and only the client knows both.
    opposing_session_id: str | None = None
    # Print compensation, in mm, applied to the FUSED solid AFTER the union —
    # never to the crowns before it.
    #
    # WHY NOT BEFORE THE BOOLEAN, which is the obvious reading. This export is
    # an OpType.Add union that produces the POSITIVE a lab draws a sheet over,
    # with the sockets filled flush. Dilating each crown before that union
    # pushes every tooth surface outward, so the finished tray is oversized on
    # every wall by the full offset. At the 0.15 mm usually quoted for a nested
    # insert that is 60% of this app's own max_translation_per_stage (0.25 mm)
    # — the aligner would give away most of a stage of prescribed movement as
    # slop. Measured on a fissured crown, +0.15 mm takes it from 131.9 to
    # 153.5 mm3, +16.4%.
    #
    # Applied to the fused model instead, it is what it claims to be: a uniform
    # allowance for a printer that comes out undersized, on a model that is
    # otherwise true to anatomy. DEFAULT 0.0 — off unless a lab asks for it,
    # and written into the manifest whenever it is not, so nobody downstream
    # has to guess whether the model they hold is true-to-anatomy.
    #
    # The 0.15 mm in carve_socket / export_nested_pair is untouched: that path
    # is a nested insert, where a clearance between two parts genuinely belongs.
    print_compensation_mm: float = 0.0

    # WHICH CONSTRUCTION BUILDS THE STAGE. "collar" is everything shipped to
    # date and is the default; nothing about it changes when this field is
    # absent, and every existing client keeps the behaviour it had.
    #
    #   collar       cut the crown, fill the old socket, build a transition
    #                collar at the new cervical position, and union the parts
    #                with manifold3d. s.21/s.23/s.26.
    #   deformation  build the T0 cast ONCE with the teeth still in its
    #                surface and, per stage, move the moving crown's vertices
    #                by the rigid stage matrix while a harmonic blend carries
    #                the gingiva inside a geodesic envelope. The index buffer
    #                is inherited from T0 and never rebuilt, so there is no
    #                socket, no cap, no collar and NO BOOLEAN - which is the
    #                whole point: s.26.8's remaining real-scan blocker is a
    #                0.09515mm self-touching edge created by a union, and a
    #                construction with no union cannot create one.
    #
    # It is a request field rather than a server mode for the same reason
    # `provider` is (s.25.7): the point of two constructions is running both
    # over ONE case, and a server-wide switch makes that a restart apiece.
    construction: Literal["collar", "deformation"] = "collar"

    # Enamel the clinician has PRESCRIBED to be removed interproximally, in mm.
    # Only the deformation construction reads it, and only to judge the
    # implicit IPR it measures: a contact that closes by more than this plus
    # the 0.05mm scanner-noise tolerance is enamel the plan never authorised,
    # and `implicit_ipr_within_prescription` fails. DEFAULT 0.0 - no IPR
    # unless it was prescribed, so the gate starts at its strictest.
    prescribed_ipr_mm: float = 0.0

    # IPR PRESCRIBED PER CONTACT, e.g. {"43-44": 0.2}, in mm. Only the
    # deformation construction reads it. Before solidifying, every moved
    # tooth's penetration into each neighbour is measured (Task 2 step 5):
    # <= 0.05 mm is tolerance; more needs IPR prescribed for THAT contact, or
    # the stage is NOT PRINT READY naming the contact and the millimetres. A
    # key that is not two FDI numbers is refused, never read as "no IPR".
    ipr_by_contact_mm: dict[str, float] = {}


# The per-stage translation limit staging_estimate divides by. Named here so
# the compensation guard can say what it is comparing against.
MAX_TRANSLATION_PER_STAGE_MM = 0.25
# Ceiling on print compensation: twice a full stage of prescribed movement.
# Anything past this is not a printer allowance, it is a mistake, and it would
# be invisible in the exported STL — which is exactly why it is refused loudly.
MAX_PRINT_COMPENSATION_MM = 0.5

# Headroom kept beyond a socket rim when the manufacturing trim widens itself.
# The trim's predicate is on FACE CENTROIDS, so the band's real edge sits up to
# about half a face inside the nominal margin, and the collar's foot is seeded
# `seat_bottom_outset_mm` further out than the rim again. 1.0mm covers both on
# a scan whose mean edge is ~0.4mm. It is headroom on a structural guarantee,
# not a tuned threshold: the guarantee is that the trim never deletes the cast
# the reconstruction has to seat into.
TRIM_RIM_HEADROOM_MM = 1.0

# How far the plug is raised into the crown so the two share volume rather than
# a surface. Well under the shallowest clinical crown, and buried either way.
PLUG_LIFT_MM = 1.0
# And how far its cross-section is drawn in from the rim it is built on. The
# plug is made FROM the crown's own margin, so at full width its wall is tangent
# to the crown's wall the whole way round — and a boolean between tangent
# surfaces produces isolated point contacts, not a clean intersection curve.
# Measured at full width: manifold3d returned genus 0 and a watertight index
# buffer carrying 30 pairs of vertices at distance 0.0, which a reader's weld
# turns into 30 non-manifold edges. Drawn in, the plug sits strictly inside both
# solids and the boolean has ordinary surfaces to cut.
PLUG_INSET_FRACTION = 0.75


def _to_manifold(verts: np.ndarray, faces: np.ndarray):
    # Routed through the guard because this is the FIRST thing any CSG path
    # touches — the two `import manifold3d` lines further down only ever run on
    # objects this function already produced, so they cannot be reached with the
    # wheel missing.
    m3 = _require_manifold3d()
    return m3.Manifold(m3.Mesh(vert_properties=np.asarray(verts, np.float32),
                               tri_verts=np.asarray(faces, np.uint32)))


def _rim_plug(rim_xyz: np.ndarray, u_oa: np.ndarray, depth_mm: float):
    """A closed solid hanging from the cervical rim, `depth_mm` apical.

    WHY A STAGE MODEL NEEDS THIS. The crown was cut from the arch along this
    exact rim, so its capped bottom and the flush-filled socket occupy the SAME
    vertices. Two closed solids that share a surface do not overlap — they kiss
    — and a union of them keeps both copies of every shared vertex. Measured:
    manifold3d returned a mesh with 24 pairs of vertices at distance 0.0, which
    reports watertight as an index buffer and comes back with 24 non-manifold
    edges the moment a reader welds it.

    A plug gives the two solids real volume in common, so the boolean has
    something to cut. It is buried inside the fused model and never reaches a
    surface, which is why it can be this crude.

    DEPTH IS THE ROOT LENGTH, not some small seating value. The plug travels
    WITH the crown: a tooth being extruded carries its plug up out of the
    gingiva, and any plug shorter than the movement would clear the base
    entirely at the last stages and fuse nothing. A 7.686mm extrusion against a
    9mm root still leaves 1.3mm buried. This is the same virtual root the
    viewport draws in wireframe — there it explains the pivot, here it is what
    makes the model buildable.
    """
    rim = np.asarray(rim_xyz, float)
    centre = rim.mean(axis=0)
    rim = centre + (rim - centre) * PLUG_INSET_FRACTION
    pts, faces, _info = cg.build_socket_cup(rim, u_oa, depth_mm=float(depth_mm))
    verts = np.vstack([rim, np.asarray(pts, float)])
    v2, f2 = cg.cap_and_close(verts, np.asarray(faces, int))
    f2 = cg.make_consistent_winding(v2, f2)

    # LIFT IT INTO THE CROWN. Hanging the plug from the rim reproduces the very
    # problem it exists to solve one level down: the crown's cap and the plug's
    # cap are both ON the rim, so they kiss instead of overlapping and the two
    # stay separate solids. Measured before this line existed — crown + plug
    # came back with genus -3, which for a closed mesh means two components, not
    # one fused body. Raising the plug so its top sits inside the crown gives
    # the boolean real volume to work with at both ends.
    u = np.asarray(u_oa, float)
    return v2 + (u / np.linalg.norm(u)) * PLUG_LIFT_MM, f2


MANIFOLD3D_MISSING = (
    "manifold3d is not installed, so fused stage models cannot be built. "
    "Install it with `pip install manifold3d` (it is listed in requirements.txt). "
    "Everything else — cutting, kinematics, staging and the printable cast "
    "export — works without it.")


def _require_manifold3d():
    """Import manifold3d or raise a 503 that says what to do about it.

    An ImportError escaping a request handler is a 500 with a traceback about a
    module name, which tells a clinician nothing and a support engineer only
    slightly more. This is the one dependency in the stack that is optional in
    practice — the wheel does not build everywhere — so its absence gets a
    sentence rather than a stack.
    """
    try:
        import manifold3d as m3
        return m3
    except ImportError as e:
        raise HTTPException(503, f"{MANIFOLD3D_MISSING} ({e})")


# The order matters and each rung is here because it fixes a different thing.
CSG_REPAIR_CASCADE = ("fix_normals+fill_holes", "weld_1e-5")


def _repair_for_csg(v: np.ndarray, f: np.ndarray, stage: str):
    """One rung of the CSG repair cascade. Returns (verts, faces).

    RUNG 1 — `fix_normals` then `fill_holes`. manifold3d requires a closed,
    consistently-wound input; a crown whose winding was scrambled by a boolean
    upstream, or that carries a one-triangle hole, is refused by the library
    with a message about manifoldness that names neither cause.

    RUNG 2 — weld at 1e-5 mm. A boolean between tangent surfaces emits
    topologically distinct vertices at identical positions (see the long note in
    the stage loop); when those positions differ by a few ULP instead of exactly
    zero, nothing downstream can weld them and the solid never closes. 1e-5 mm
    is 10 nanometres — four orders below the 0.1mm the appliance is built to —
    so what it moves is numerically invisible and clinically nothing.

    THIS IS THE END OF THE CASCADE. There is deliberately no displacement-
    carving rung: a stage model that still will not close after this is refused
    by name. A tray thermoformed from a solid the software had to force closed
    is worse than a tray that was never made.
    """
    v = np.asarray(v, float)
    f = np.asarray(f, np.int64)

    if stage == "fix_normals+fill_holes":
        import trimesh
        import trimesh.repair
        m = trimesh.Trimesh(vertices=v, faces=f, process=False, validate=False)
        trimesh.repair.fix_normals(m)
        trimesh.repair.fill_holes(m)
        return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)

    if stage == "weld_1e-5":
        v2, f2 = _repair_for_csg(v, f, "fix_normals+fill_holes")
        keys = np.round(v2 / 1e-5).astype(np.int64)
        _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                      return_inverse=True)
        # Keep the ORIGINAL coordinate of the first occurrence rather than the
        # rounded key: the grid is how duplicates are found, not what replaces
        # them, so no surviving vertex is snapped to a lattice.
        welded = v2[first]
        f3 = np.asarray(inverse).ravel()[f2]
        degenerate = ((f3[:, 0] == f3[:, 1]) | (f3[:, 1] == f3[:, 2])
                      | (f3[:, 0] == f3[:, 2]))
        return welded, f3[~degenerate]

    raise ValueError(f"unknown CSG repair stage {stage!r}")


def _batch_union_with_cascade(m3, part_arrays, label: str):
    """Union a list of (verts, faces) with a repair cascade. Returns (solid, info).

    `info` carries `fallback_reason` when any repair was needed, so the manifest
    records that a stage was not built from the meshes as they arrived. A silent
    repair is the thing to avoid here: the lab prints what comes out.
    """
    attempts = []
    for stage in (None,) + CSG_REPAIR_CASCADE:
        try:
            if stage is None:
                parts = [_to_manifold(v, f) for v, f in part_arrays]
            else:
                parts = [_to_manifold(*_repair_for_csg(v, f, stage))
                         for v, f in part_arrays]
            solid = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
            mesh = solid.to_mesh()
            if len(np.asarray(mesh.tri_verts)) == 0:
                raise ValueError("the boolean produced an empty solid")
            info = {"csg_repair_stage": stage or "none",
                    "csg_attempts": attempts + [{"stage": stage or "none",
                                                 "result": "ok"}]}
            if stage is not None:
                info["fallback_reason"] = "CSG_REPAIR_CASCADE_APPLIED"
                info["fallback_detail"] = (
                    f"{label}: the boolean failed on the meshes as built and "
                    f"succeeded after '{stage}'. The geometry that was printed is "
                    f"not bit-identical to the geometry that was cut.")
            return solid, info
        except HTTPException:
            raise
        except Exception as e:                       # noqa: BLE001 — cascade
            attempts.append({"stage": stage or "none",
                             "result": f"{type(e).__name__}: {e}"})

    tried = "; ".join(f"{a['stage']} -> {a['result']}" for a in attempts)
    raise HTTPException(422,
        f"{label}: the boolean union could not be completed. Tried {tried}. "
        f"The stage is REFUSED rather than forced closed — a tray thermoformed "
        f"from a solid the software had to carve into shape is worse than a "
        f"tray that was never made.")


def _provenance_labels(mesh, sources: dict, old_site_pts=None,
                       old_site_tol_mm: float = 0.25):
    """Per-triangle geometry source for a fused stage.

    The manifold3d runs give CAST / CROWN / SEAT / EMERGENCE / CLEARANCE
    directly. OLD_SOCKET_REPAIR cannot come from a run, because the flush
    closure of the site the tooth left is welded into the cast's own vertex
    array long before any boolean - so it is separated GEOMETRICALLY, by the
    triangle centroid lying on the cap the closure built. That is stated
    rather than implied: a cast triangle within `old_site_tol_mm` of a cap
    triangle's centroid is reported as OLD_SOCKET_REPAIR.
    """
    tri = np.asarray(mesh.tri_verts, np.int64)
    lab = np.array(["UNATTRIBUTED"] * len(tri), dtype=object)
    rid = np.asarray(mesh.run_original_id)
    ridx = np.asarray(mesh.run_index)
    for i, oid in enumerate(rid):
        lab[ridx[i] // 3:ridx[i + 1] // 3] = sources.get(int(oid),
                                                         f"ORIGINAL_{int(oid)}")
    if old_site_pts is not None and len(old_site_pts):
        vp = np.asarray(mesh.vert_properties, float)[:, :3]
        cast = np.where(lab == "ORIGINAL_CAST")[0]
        if len(cast):
            cent = vp[tri[cast]].mean(axis=1)
            d, _ = cKDTree(np.asarray(old_site_pts, float)).query(
                cent, workers=-1)
            lab[cast[d <= old_site_tol_mm]] = "OLD_SOCKET_REPAIR"
    return lab


#: Manifest keys for what `_solid_bodies` measured. Constants because a gate
#: and a harness both read them (AGENT_BRIEF A2 rule 13).
KEY_INTERNAL_VOIDS = "internal_voids"
KEY_INTERNAL_VOIDS_FILLED = "internal_voids_filled_mm3"
#: Retained for one release so an older manifest reader keeps working. It
#: always equals `internal_voids`, under a name that described the wrong thing.
KEY_CRUMBS_ALIAS = "inverted_crumbs_discarded"


def _solid_bodies(solid):
    """Bodies AFTER the re-union, and enclosed voids named as what they are.

    Returns `(solid, bodies, info)`.

    TWO DEFECTS, ONE LINE APART (AGENT_BRIEF B5).

    1. THE COUNT WAS TAKEN BEFORE THE RE-UNION. This counted the positive
       parts of `decompose()` and returned that number, then unioned them and
       returned the union - so `bodies` described the INPUT to the last
       boolean, not the solid that is actually exported. Measured on the
       control below: `cube(10) - cube(6) + cube(2)` decomposes to
       `[1000.0, -216.0, 8.0]`, which the old rule reported as 2 bodies, while
       the re-unioned solid decomposes to `[1000.0]` - ONE body, and the
       written STL has one connected component. That is exactly the shape of
       the real-scan failure: `single_positive_manifold_body` passed while
       `body_count_agrees_with_stl` failed, on both recorded runs.

    2. A NEGATIVE-VOLUME PART IS AN ENCLOSED VOID, NOT A CRUMB. The old name
       came from a real observation - `[27572.4, -1.9]` mm3, where the -1.9
       genuinely was a sliver left by a tangential boolean - and then
       generalised it to every negative part. A cavity fully inside the solid
       decomposes to a negative part too, and dropping it FILLS the cavity.
       That may be the right thing for a printable cast, but it is a change to
       the geometry and it was happening silently. It is now measured and
       reported: how many, and how much material was added by filling them.

    `bodies` > 1 is a real fracture and the caller decides what to do.
    """
    import manifold3d as m3
    parts = solid.decompose()
    vols = [float(c.volume()) for c in parts]
    keep = [c for c, v in zip(parts, vols) if v > 0]
    voids = [v for v in vols if v <= 0]

    info = {
        KEY_INTERNAL_VOIDS: len(voids),
        KEY_INTERNAL_VOIDS_FILLED: round(float(sum(abs(v) for v in voids)), 6),
        # The alias keeps one release of manifest compatibility.
        KEY_CRUMBS_ALIAS: len(voids),
        "raw_part_volumes_mm3": [round(v, 6) for v in vols],
    }

    if not keep:
        info["bodies_before_reunion"] = 0
        return solid, 0, info

    fused = keep[0] if len(keep) == 1 else m3.Manifold.batch_boolean(
        keep, m3.OpType.Add)

    # THE COUNT THAT MATTERS: decompose the solid that is actually returned.
    after = [float(c.volume()) for c in fused.decompose()]
    info["bodies_before_reunion"] = len(keep)
    info["reunion_part_volumes_mm3"] = [round(v, 6) for v in after]
    return fused, sum(1 for v in after if v > 0), info


def _manufacturing_tooth(rec: dict, verts: np.ndarray):
    """The crown, once, at T0 — then transformed per stage. NO PLUG.

    THE ROOT PLUG IS GONE FROM THIS PATH, and that deletion is the point of
    the whole manufacturing correction. This function used to build
    `_rim_plug(verts[rim], u_oa, depth=rec["root_length_mm"])` and union it
    into the crown, so every exported tooth carried a 9-13mm synthetic root
    column that travelled with it. `_rim_plug`'s own docstring said so:
    *"DEPTH IS THE ROOT LENGTH, not some small seating value."*

    Two things were wrong with that, and only the second is obvious:

      1. `root_length_mm` is a C_res ESTIMATION parameter. Using it as
         manufacturing fusion depth welded a clinical estimate to a
         geometric construction, so changing one silently changed the other.
      2. A plug that moves with the tooth EMERGES when the tooth extrudes.
         Measured: a 1.2mm extrusion grew the fused volume by 28.84mm3 of
         visible synthetic material. That was the cast distortion.

    What replaced it is `manufacturing.build_stage_tooth_interface`, which
    reconstructs the cast locally at the tooth's target position instead of
    growing a root to reach the old one. The union needs common volume, and
    it now gets it from a cavity cut at the NEW rim rather than from a column
    hanging off the old one.

    Returns (solid, bodies, void_info). `bodies` > 1 still means the crown
    fractures — a shell rather than a tooth — which is what the export screen
    acts on. The plug used to serve double duty as that probe; the crown's own
    decomposition answers it directly and without inventing anatomy.
    """
    crown = _to_manifold(rec["cv"], rec["cf"])
    return _solid_bodies(crown)


SHELL_REFUSAL = ("Cannot export raw shell geometry. Crown must be fully extracted and "
                 "sealed via the Wand/Brush tool to generate 3D printable stages.")


def _screen_crowns_for_manufacturing(teeth: list, verts: np.ndarray):
    """Refuse the export unless every crown is a solid a boolean can fuse.

    Mutates each entry with `solid`, `bodies`, `voids` and `screen`, so the
    manufacturing solids are built exactly once.

    TWO CHECKS, AND THE SECOND IS THE ONE THAT MATTERS.

    cg.crown_is_printable is a cheap topological screen that produces a specific
    diagnosis. It is NOT sufficient on its own, measured on four auto-cut crowns
    from a real scan: every one was already watertight and single-bodied (/cut
    asserts that and refuses otherwise), Euler characteristic caught one, a
    volume floor caught a 0.2mm speck, and one crown passed every topological
    test there is and still fractured into three pieces when fused.

    So the decisive test is the boolean itself: build the tooth's manufacturing
    solid and require ONE positive-volume body. A crown too thin for its own
    plug to overlap comes apart there and nowhere else, and that is precisely
    the failure that produced "fused into 7 separate solids".

    Refuses the WHOLE export listing every failing tooth, not just the first,
    and not a partial model: a stage model missing a tooth is a wrong model, and
    a lab would thermoform a tray with a gap where a tooth should be.
    """
    failures = []
    for t in teeth:
        rec = t["rec"]
        who = f"FDI {t['fdi']}" if t["fdi"] is not None else f"tooth {t['tid'][:8]}"
        screen = cg.crown_is_printable(rec["cv"], rec["cf"])
        t["screen"] = screen
        if not screen["ok"]:
            failures.append(f"{who}: {screen['reason']}")
            t["solid"], t["bodies"], t["voids"] = None, 0, {}
            continue

        solid, bodies, void_info = _manufacturing_tooth(rec, verts)
        t["solid"], t["bodies"], t["voids"] = solid, bodies, void_info
        if bodies != 1:
            failures.append(
                f"{who}: watertight and Euler-2 at {screen['volume_mm3']:.1f}mm3, but it "
                f"fuses into {bodies} separate pieces — the crown is too thin a shell for "
                f"its own root plug to overlap, so a boolean cannot make one solid of it")

    if failures:
        raise HTTPException(422,
            SHELL_REFUSAL + " " + f"{len(failures)} of {len(teeth)} crown(s) are not solids: "
            + "; ".join(failures) + ". The whole export is refused rather than shipping a "
            "model with a tooth missing.")


def _stage_clinical(clinical: dict, k: int, n: int) -> dict:
    """The six channels at stage k of n.

    DELEGATES to `stage_matrix.stage_clinical`, which is now the single source
    for both constructions (AGENT_BRIEF 2.2). The collar path and the
    deformation path must not be able to derive different matrices for the
    same prescription - they would be posed almost identically, every
    topology gate would pass, and the two paths would be quietly
    incomparable. `test_stage_matrix_shared.py` pins them at 0 ULP.
    """
    return stage_matrix.stage_clinical(clinical, k, n)


def build_stage_bundle(sid: str, req: StageExportRequest,
                       require_print_ready: bool = False,
                       part_order: str = "forward") -> dict:
    """One FUSED, watertight solid per stage — the models a lab thermoforms over.

    Not loose crowns. A vacuum-forming model is a single solid: the gingiva with
    the teeth standing in their stage-k positions, fused, so the sheet draws down
    over one continuous surface. Handing a lab a base plus fourteen separate
    crowns hands them an assembly problem.

    Three things this does that the planned-setup export does not:

    1. THE SOCKETS ARE FILLED FLUSH, not cupped. A tooth that has translated out
       of its 3.5mm alveolus leaves the cup gaping beside it — watertight, and a
       pit in the cast that the aligner would take an impression of.
    2. The crowns are posed by CLINICAL PARAMETERS scaled to k/N and rebuilt
       through cg.kinematic_matrix, matching the browser's timeline exactly.
    3. Every stage is unioned with manifold3d and then re-read from its own
       written STL, because a mesh that measures watertight as an index buffer
       can still arrive non-manifold after a reader welds it.

    Measured on the real scan: base (178k tris) union crown (14k tris) = 0.162s,
    so a 31-stage 14-crown case is roughly 70s of boolean work.
    """
    t_bundle0 = time.perf_counter()
    _t_stage = time.perf_counter()
    try:
        v, f = STORE.require(sid, "verts"), STORE.require(sid, "faces")
        arch = STORE.arch(sid)
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    extracted = STORE.get(sid, "extracted_faces")
    if extracted is None or not extracted.any():
        raise HTTPException(400, "Nothing has been extracted yet; there are no stages to export.")

    af = STORE.get(sid, "arch_frame")
    if af is None:
        raise HTTPException(409,
            "Occlusal plane not defined for this session, so the cast base cannot be built.")

    # --- the teeth, and how long the case is ------------------------------
    teeth = []
    for key in STORE.keys(sid):
        if not key.startswith("tooth:"):
            continue
        t = STORE.get(sid, key)
        clinical = t.get("clinical") or {}
        st = cg.staging_estimate(**{k: float(clinical.get(k, 0.0) or 0.0) for k in
                                    ("tip_deg", "torque_deg", "rotation_deg",
                                     "d_md", "d_bl", "d_oa")})
        teeth.append({"tid": key.split(":", 1)[1], "rec": t, "clinical": clinical,
                      "staging": st, "fdi": _tooth_fdi(sid, t, f)})

    total = max((t["staging"]["stages_required"] for t in teeth), default=0)
    if total <= 0:
        raise HTTPException(400,
            "No movement has been prescribed, so every stage would be identical to the "
            "scan. Move at least one tooth before exporting stages.")
    if total > req.max_stages:
        binding = max(teeth, key=lambda t: t["staging"]["stages_required"])
        raise HTTPException(422,
            f"This plan needs {total} stages, past the {req.max_stages} this export will "
            f"build. Tooth {binding['fdi'] or binding['tid']} is binding it at "
            f"{binding['staging']['stages_required']} "
            f"({binding['staging']['driver']}-driven). Reduce the movement or raise "
            f"max_stages deliberately — {total} aligners is a two-year course.")

    comp = float(req.print_compensation_mm or 0.0)
    if not np.isfinite(comp) or comp < 0.0 or comp > MAX_PRINT_COMPENSATION_MM:
        raise HTTPException(422,
            f"print_compensation_mm={req.print_compensation_mm!r} is outside "
            f"[0.0, {MAX_PRINT_COMPENSATION_MM}]. This is a uniform allowance for a "
            f"printer that runs undersized, not a place to encode tooth movement — "
            f"{MAX_PRINT_COMPENSATION_MM} mm is already twice the "
            f"{MAX_TRANSLATION_PER_STAGE_MM} mm this plan moves a tooth in a whole stage.")

    # --- the base, sockets filled FLUSH -----------------------------------
    sealed_v, sealed_f, sockets = _seal_sockets(v, f, sid, extracted, flush=True)
    try:
        curve, _ = cg.fit_arch_curve(v, af)
        # THE TRIM MAY NOT CUT THE CAST OUT FROM UNDER A SOCKET RIM, and until
        # now it could. `ARCH_TRIM_MARGIN_MM` (7.0) was chosen in s.10 from
        # measurements of the WHOLE arch - kept fraction 96.7%, material still
        # inside the arch opening 3.35% - and those numbers are still right.
        # What they do not describe is a wide molar, whose cervical rim reaches
        # further from the fitted ridge than the band does. Measured on this
        # project's real scan, FDI 46 at margin 7.0: 53 of 280 rim points ended
        # up to 1.2961mm OUTSIDE the finished cast, and the transition collar
        # then refused `interface_unbuildable_wall_too_thin` with 40 points its
        # foot could not seat into. It was refusing correctly, about geometry
        # the trim had deleted, while reporting a cast 2.8573mm thick as too
        # thin.
        #
        #   margin   FDI 45            FDI 46
        #     7.0    ok                REFUSED (53 rim points outside, 1.2961mm)
        #     9.0    ok                ok      (0 outside)
        #    12.0    ok                ok
        #    15.0    ok                ok
        #
        # SO THE MARGIN IS NOT RAISED GLOBALLY - that would undo a measured
        # decision to make one case pass, and would loosen the trim for every
        # case that never needed it. It is raised only as far as the rims that
        # actually exist require, per export, and the manifest records by how
        # much and why. A case with no cuts trims exactly as it did before.
        trim_margin = float(req.trim_margin_mm)
        rim_reach = 0.0
        for _t in teeth:
            # `socket_rim` is an ndarray, and `x or []` evaluates bool(x) -
            # which raises on any array of length > 1.
            _raw = _t["rec"].get("socket_rim")
            if _raw is None:
                continue
            _rim = np.asarray(_raw, np.int64).reshape(-1)
            if not len(_rim):
                continue
            rim_reach = max(rim_reach, float(
                cg.distance_to_arch_curve(v[_rim], af, curve).max()))
        margin_floor = (rim_reach + mfg.DEFAULT_POLICY.seat_bottom_outset_mm
                        + TRIM_RIM_HEADROOM_MM) if rim_reach > 0.0 else 0.0
        trim_raised = margin_floor > trim_margin
        if trim_raised:
            trim_margin = float(margin_floor)
        tv, tf, trim_info = cg.trim_to_arch(sealed_v, sealed_f, af,
                                            margin_mm=trim_margin, curve=curve)
        trim_info["margin_requested_mm"] = round(float(req.trim_margin_mm), 4)
        trim_info["margin_used_mm"] = round(float(trim_margin), 4)
        trim_info["margin_raised_for_socket_rims"] = bool(trim_raised)
        trim_info["socket_rim_max_distance_to_ridge_mm"] = round(rim_reach, 4)
        tv, tf, rim, uinfo = cg.clear_undercut_periphery(tv, tf, af, trim_info["rim_loop"]); bv, bf, base_info = cg.build_cast_base(tv, tf, af,
                                               base_thickness_mm=req.base_thickness_mm,
                                               rim=rim)
    except ValueError as e:
        raise HTTPException(422, f"The cast base could not be built. {e}")

    m3 = _require_manifold3d()
    # PROVENANCE. Every input solid is given a manifold3d "original" id, which
    # survives the boolean and comes back on `run_original_id` / `run_index` -
    # so every triangle in a fused stage can be traced to the geometry it came
    # from, and an offending edge can be reported with the sources meeting on
    # it rather than with a guess. `as_original()` is free; it only stamps an
    # id on the solid.
    geometry_sources: dict[int, str] = {}

    def _tag(solid, label):
        s_ = solid.as_original()
        geometry_sources[int(s_.original_id())] = label
        return s_

    base_solid = _tag(_to_manifold(bv, bf), "ORIGINAL_CAST")
    # Built ONCE for the whole bundle: the cast does not change between teeth
    # or between stages, and rebuilding the KD-tree per query was 43ms of the
    # 89ms a crown cost when the antagonist check made that mistake (section 11).
    cast_probe = mfg.CastProbe(bv, bf)
    # Kept for the unaffected-cast fidelity comparison at the end of each
    # stage, and COMPACTED for it. `bv` legitimately carries every vertex the
    # trim removed - rule 3.1 forbids rebuilding the scan's array - and 4497
    # of this cast's 8372 vertices are unreferenced. Sampling them as if they
    # were surface points reported 3.39mm of "cast deformation" from phantoms
    # that are not on the cast at all. The same defect, in a third place.
    _ref_used = np.unique(np.asarray(bf, np.int64))
    _ref_map = np.zeros(len(bv), np.int64)
    _ref_map[_ref_used] = np.arange(len(_ref_used))
    original_cast = (np.array(bv[_ref_used], copy=True),
                     _ref_map[np.asarray(bf, np.int64)])
    original_cast_unreferenced = int(len(bv) - len(_ref_used))
    # THE OLD SITE, kept as a point set so a cast triangle can be attributed to
    # it. `_seal_sockets(flush=True)` welds the closure into the cast's own
    # vertex array, so there is no run id to read it off afterwards - the cap's
    # own triangle centroids are what separates OLD_SOCKET_REPAIR from
    # ORIGINAL_CAST, and rebuilding them here is the same call `_seal_sockets`
    # made with the same inputs.
    old_site_pts = []
    for _t in teeth:
        _rec = _t["rec"]
        _raw_rim = _rec.get("socket_rim")
        if _raw_rim is None:
            continue
        _rim = np.asarray(_raw_rim, np.int64)
        if len(_rim) < 3:
            continue
        try:
            _pts, _cup, _ = cg.build_socket_cup(
                v[_rim], np.asarray(_rec["frame"]["u_oa"], float), depth_mm=0.0)
        except Exception:                                # noqa: BLE001
            continue
        _cv = np.vstack([v[_rim], np.asarray(_pts, float)])
        old_site_pts.append(_cv[np.asarray(_cup, np.int64)].mean(axis=1))
    old_site_pts = (np.vstack(old_site_pts) if old_site_pts
                    else np.zeros((0, 3)))
    # Screens every crown AND builds its manufacturing solid, once. manifold3d
    # transforms lazily, so a stage then costs one batch boolean rather than
    # rebuilding every mesh.
    _screen_crowns_for_manufacturing(teeth, v)

    # The antagonist, resolved once. None when the opposing arch is not loaded,
    # which is the ordinary single-arch case and skips the check entirely.
    ant = _antagonist(req.opposing_session_id)
    t_collide = 0.0
    t_ipr = 0.0
    t_metrics = 0.0
    t_iface = 0.0
    t_cast = round(time.perf_counter() - t_bundle0, 3)

    # PER-STAGE INTERPROXIMAL, not once on the final pose. A tooth that is clear
    # at T0 and clear at the setup can still close an embrasure to nothing in
    # the middle, and the middle is where the trays are — measuring only the
    # endpoint answers a question the clinician did not ask. This is the same
    # measurement /export takes once; the cost of taking it every stage is
    # reported in the manifest rather than assumed to be free.
    IPR_WARN_MM = 0.5

    # --- one fused solid per stage ----------------------------------------
    blobs, stage_meta = {}, []
    t_union = 0.0
    for k in range(1, total + 1):
        interference = []
        ipr_rows, ipr_worst = [], 0.0

        # === LOCAL TARGET-POSITION RECONSTRUCTION ==========================
        # The architecture this replaces: `parts = [base_solid]` followed by
        # `parts.append(crown_and_a_9mm_root_plug.transform(M))` and a single
        # union. That never reconstructed the cast at the tooth's NEW cervical
        # position, and the plug travelled with the tooth — so an extruding
        # tooth carried synthetic root material up out of the gingiva as
        # visible positive geometry. Measured before this change: fused volume
        # grew 27579.63 -> 27608.44 mm3 monotonically with a 1.2mm extrusion.
        #
        # The order is SUBTRACT, then ADD, then UNION:
        #   cast - cavities            make room where the tooth penetrates
        #        + emergence ramps     add tissue where the rim lifted clear
        #        u rigid crowns        the crown, and nothing but the crown
        interfaces, cavity_tools, ramp_tools, seat_tools = [], [], [], []
        connector_pts = {}
        crown_solids = {}
        stage_matrices = {}
        for t in teeth:
            rec, clinical = t["rec"], _stage_clinical(t["clinical"], k, total)
            M = cg.kinematic_matrix(rec["frame"], rec["c_res"], **clinical)
            stage_matrices[t["tid"]] = M

            rim_t0 = v[np.asarray(rec["socket_rim"], np.int64)]
            # THE NEIGHBOURS' TARGET RIMS, so the connector can clamp its own
            # reach and the gingival bridge cannot be consumed by construction
            # rather than only detected afterwards.
            nb = []
            for other in teeth:
                if other["tid"] == t["tid"]:
                    continue
                o_rec = other["rec"]
                o_M = cg.kinematic_matrix(
                    o_rec["frame"], o_rec["c_res"],
                    **_stage_clinical(other["clinical"], k, total))
                nb.append(cg.apply_matrix(
                    v[np.asarray(o_rec["socket_rim"], np.int64)], o_M))
            _t_if = time.perf_counter()
            iface = mfg.build_stage_tooth_interface(
                bv, bf, rec["cv"], rec["cf"], rim_t0,
                rec["frame"]["u_oa"], M, probe=cast_probe,
                neighbour_rims=nb)
            t_iface += time.perf_counter() - _t_if
            row = iface.manifest_row()
            row.update({"stage": k, "tooth_id": t["tid"], "fdi": t["fdi"],
                        "clinical": clinical})

            if not iface.ok:
                raise HTTPException(422, _jsonable({
                    "error": f"Stage {k}: the local target-position interface "
                             f"could not be built for "
                             f"{('FDI ' + str(t['fdi'])) if t['fdi'] is not None else 'tooth ' + t['tid'][:8]}.",
                    "gate": iface.refusal_reason,
                    "detail": "The manufacturing layer refuses rather than "
                              "falling back to a synthetic root connector. A "
                              "model the software had to invent a root for is "
                              "not the model that was planned.",
                    "diagnostics": row}))

            # Geodesic ROI — NOT a bounding box. A rectangular XYZ box around a
            # rim on a curved arch sweeps in the neighbouring teeth and the
            # gingiva on the far side of the ridge.
            roi_mask, roi_info = mfg.affected_region(
                bv, bf, cg.apply_matrix(rim_t0, M), mfg.DEFAULT_POLICY.roi_radius_mm)
            row.update(roi_info)
            # MEASURED AGAINST THE CONNECTOR, which is the geometry that IS
            # the interface. It used to be measured against `cavity_verts`,
            # which is None whenever no material is removed, so every tooth in
            # every shipped manifest read `continuous: false, reason: "no
            # interface geometry"` while the connector was in fact a
            # continuous band covering the whole rim.
            row["continuity"] = iface.diagnostics.get("continuity") or (
                mfg.interface_continuity(cg.apply_matrix(rim_t0, M),
                                         iface.seat_verts))

            crown_solid = _tag(
                t["solid"].transform(np.asarray(M[:3, :4], float)), "CROWN")
            crown_solids[t["tid"]] = crown_solid

            # NO SUBTRACTION. This is the last structural correction, and it
            # is counter-intuitive enough to be worth stating plainly.
            #
            # `(cast - crown) u crown` is an identity on GEOMETRY and not on
            # TOPOLOGY: the subtract carves a cavity whose walls ARE the
            # crown's surface, and the union then lays that same surface back
            # on top of itself. manifold3d answers those coincident surfaces
            # with topologically distinct vertices at identical positions, and
            # a downstream reader's weld turns every one of them into a
            # non-manifold edge. Measured across a 1.2mm extrusion:
            #
            #     with the crown-cut : 0, 11, 5, 9, 5   non-manifold edges
            #     without it         : 0,  0, 0, 1, 0
            #
            # A stage model is a POSITIVE the lab draws a sheet over, so a
            # penetrating tooth does not need material removed at all - the
            # union absorbs it. Cavities belong to nested inserts, which is
            # exactly where the 0.15mm clearance in carve_socket still lives.
            # The classification above is still what shapes the connector and
            # what the manifest reports; it simply no longer drives a cut.
            # CASE A / CASE C: the crown-derived local clearance is MEASURED
            # in build_stage_tooth_interface and deliberately not emitted -
            # see the long note there for the graze counts that decided it.
            # The wiring stays so the manifest can state that nothing was
            # removed, rather than being silent about it.
            # CASE A: the crown-derived local clearance tool, built HERE
            # because an exact dilation is a Minkowski sum and that is the CSG
            # engine's job. Two facts make it affordable: dilation COMMUTES
            # with a rigid transform, so the sum is taken once per tooth on
            # its T0 crown and then carried by the same stage matrix the crown
            # gets; and the sphere is coarse on purpose - measured on a
            # 250-triangle crown, 8 segments cost 325ms and 12 cost 520ms for
            # volumes 49.836 and 50.107mm3, a 0.5% difference on a 0.05mm
            # clearance. The array alternative, a vertex-normal offset, gives
            # 46.86mm3 and can fold.
            if iface.diagnostics.get("clearance_required"):
                dil = t.get("dilated")
                if dil is None:
                    dil = m3.Manifold.minkowski_sum(
                        t["solid"],
                        m3.Manifold.sphere(mfg.DEFAULT_POLICY.clearance_mm, 8))
                    t["dilated"] = dil
                tool = _tag(dil.transform(np.asarray(M[:3, :4], float)),
                            "LOCAL_CLEARANCE")
                cavity_tools.append(tool)
                row["clearance_tool_volume_mm3"] = round(float(tool.volume()), 4)
            if iface.seat_verts is not None:
                # Unioned WITH the crown, not into it: the crown solid stays a
                # rigid transform of T0 and the rigidity tests measure it alone.
                seat_solid = _tag(
                    _to_manifold(iface.seat_verts, iface.seat_faces),
                    "EMERGENCE_RECONSTRUCTION"
                    if iface.diagnostics.get("rim_points_lifted") else "SEAT")
                seat_tools.append(seat_solid)
                connector_pts[t["tid"]] = np.asarray(iface.seat_verts, float)
                # REAL INTERSECTION VOLUMES. A nominal 0.25mm overlap is not
                # evidence that anything fuses; the measured common volume is.
                # A connector with nominal dimensions and zero actual
                # intersection is a failed interface, and only this catches it.
                row["overlap_seat_crown_mm3"] = mfg.overlap_volume(
                    m3, seat_solid, crown_solid)
                row["overlap_seat_cast_mm3"] = mfg.overlap_volume(
                    m3, seat_solid, base_solid)
            row["overlap_crown_cast_mm3"] = mfg.overlap_volume(
                m3, crown_solid, base_solid)
            interfaces.append(row)

        # Adjacent reconstructions must not merge into one trench.
        for i in range(len(teeth)):
            for j in range(i + 1, len(teeth)):
                Mi, Mj = stage_matrices[teeth[i]["tid"]], stage_matrices[teeth[j]["tid"]]
                ri = cg.apply_matrix(v[np.asarray(teeth[i]["rec"]["socket_rim"], np.int64)], Mi)
                rj = cg.apply_matrix(v[np.asarray(teeth[j]["rec"]["socket_rim"], np.int64)], Mj)
                bridge = mfg.bridge_between(ri, rj)
                # MEASURED FROM THE RECONSTRUCTIONS, not from a nominal
                # constant. `2 * fusion_overlap_mm` is 0.5mm whatever was
                # built, so it passed for a trench of any width.
                gi = connector_pts.get(teeth[i]["tid"])
                gj = connector_pts.get(teeth[j]["tid"])
                if gi is None or gj is None:
                    continue
                gap = float(cKDTree(gj).query(gi, workers=-1)[0].min())
                consumed = max(0.0, bridge - gap)
                # REFUSE ONLY WHEN THE BRIDGE IS ACTUALLY GONE, and gate on
                # the rest. `max_bridge_removal_fraction` is 0.50 with no
                # measurement behind it, and CLAUDE.md section 14's rule is
                # explicit: promote a threshold to a refusal only with numbers
                # behind it. Refusing on it would block ordinary crowding -
                # measured on two teeth 2.36mm apart, the pair takes 50.9% and
                # leaves 1.16mm of interdental tissue, which is a papilla, not
                # a trench. So the fraction FAILS THE AGGREGATE GATE, where it
                # is visible and named, and only two reconstructions that have
                # actually merged are refused outright.
                if bridge > 1e-9 and gap <= 0.0:
                    raise HTTPException(422, _jsonable({
                        "error": f"Stage {k}: two reconstructions have merged "
                                 f"and the gingival bridge between them is gone.",
                        "gate": "adjacent_reconstruction_overlap",
                        "bridge_mm": round(bridge, 4),
                        "would_consume_mm": round(consumed, 4),
                        "clear_gap_between_reconstructions_mm": round(gap, 4),
                        "measure": "minimum distance between the two connector "
                                   "solids, not 2 * fusion_overlap_mm",
                        "max_fraction": mfg.DEFAULT_POLICY.max_bridge_removal_fraction,
                        "teeth": [teeth[i]["tid"], teeth[j]["tid"]]}))

        # BATCH, not sequential: one boolean per operation rather than one per
        # tooth, so tessellation error cannot accumulate across teeth.
        # ADD THE TISSUE FIRST, THEN CUT THE SOCKET INTO IT.
        # Subtracting first removes the cast material the emergence ramp has
        # to land on, so the ramp comes out as a floating body - measured, the
        # fused model went from 1 body to 4 as the extrusion progressed. This
        # is also the physically sensible order: gingiva follows the tooth up,
        # and the socket is then formed through the built-up tissue.
        # CLEARANCE FIRST, THEN THE BRIDGE. The order matters twice over and
        # both were measured, not reasoned:
        #
        #   * Subtracting a full-depth generic SOCKET before building the ramp
        #     removes the cast material the ramp has to land on and orphans
        #     it - the fused model went from 1 body to 4 across an extrusion.
        #     That is why the old order added first.
        #   * But the crown-derived CLEARANCE is a different tool. Applied
        #     after the ramp it cuts straight THROUGH the emergence bridge,
        #     because the dilated crown passes exactly where the bridge rises
        #     to meet the rim: measured on tipping and rotation, bodies went
        #     to 2 with components 2 - the tooth and its bridge severed from
        #     the cast.
        #
        # So the clearance acts on the CAST ALONE, before any reconstruction
        # exists to be damaged, and it removes only 0.05mm around the crown -
        # far too little to take away the surface the ramp lands on.
        prepared = base_solid
        clearance_report = {"tools_built": len(cavity_tools), "emitted": False,
                            "reason": None if cavity_tools else "no penetrating tooth",
                            "cast_bodies_after_subtract": None,
                            "removed_volume_mm3": None}
        if cavity_tools:
            # EMIT ONLY IF IT DOES NOT BREAK THE CAST. Subtracting the dilated
            # crown fragmented the cast into 4 bodies on extrusion 0.25mm
            # stage 1 - the crown passes through the thin lip left between the
            # old socket cap and the gingival wall, and isolates it. That is a
            # measured guard, not a preference: the subtraction is performed,
            # `decompose()` is counted, and the result is kept only if the cast
            # is still ONE body. A withheld tool is recorded with its reason,
            # never silently dropped.
            trial = m3.Manifold.batch_boolean([prepared] + cavity_tools,
                                              m3.OpType.Subtract)
            bodies_after = len(trial.decompose())
            clearance_report["cast_bodies_after_subtract"] = int(bodies_after)
            if bodies_after == 1 and not trial.is_empty():
                removed = float(base_solid.volume()) - float(trial.volume())
                clearance_report.update(
                    emitted=True, reason=None,
                    removed_volume_mm3=round(removed, 4))
                prepared = trial
            else:
                clearance_report["reason"] = (
                    f"withheld: the subtraction leaves {bodies_after} cast "
                    f"bodies, which would sever the reconstruction")
        for r_ in interfaces:
            if r_.get("clearance_measured"):
                r_["clearance_tool_emitted"] = bool(clearance_report["emitted"])
                r_["clearance_withheld_reason"] = clearance_report["reason"]
        if ramp_tools:
            prepared = m3.Manifold.batch_boolean([prepared] + ramp_tools,
                                                 m3.OpType.Add)
        diag_cavity_tools = len(cavity_tools)

        # THE CROWN, AND NOTHING BUT THE CROWN. No plug, no skirt. The same
        # Manifold objects that were used as cutters, so the cut and the fill
        # are the identical solid and cannot disagree numerically.
        # ORDER INDEPENDENCE IS A PROPERTY OF THE GEOMETRY, so it has to be
        # exercisable. `part_order` changes ONLY the sequence the solids are
        # handed to the boolean in - never which solids, never their geometry -
        # so a difference in the result is a difference the engine introduced.
        # "batch" is the shipped path.
        crown_parts = [crown_solids[t["tid"]] for t in teeth]
        if part_order == "reverse":
            parts = [prepared] + crown_parts[::-1] + seat_tools[::-1]
        elif part_order == "sequential":
            acc = prepared
            for q in seat_tools + crown_parts:
                acc = m3.Manifold.batch_boolean([acc, q], m3.OpType.Add)
            parts = [acc]
        else:
            parts = [prepared] + seat_tools + crown_parts

        # Per-tooth measurements that do not affect the geometry: interproximal
        # closure and the antagonist check. Both are reported, never blocking.
        for t in teeth:
            rec = t["rec"]
            M = stage_matrices[t["tid"]]

            # How far this tooth has closed its embrasures BY THIS STAGE.
            # Measured against the tooth's own T0 crown, so it is a closure, not
            # an absolute gap — the same convention /export uses.
            ti = time.perf_counter()
            try:
                ipr = cg.measure_interproximal_penetration(
                    rec["cv"], cg.apply_matrix(rec["cv"], M), rec["cf"],
                    rec["bv"], rec["bf"], rec["frame"]["u_md"])
            except Exception as e:                   # noqa: BLE001
                # NEVER BLOCKING. This is a warning chip on a timeline; a
                # measurement failing must not stop a stage from being built.
                ipr = {"error": f"{type(e).__name__}: {e}"}
            t_ipr += time.perf_counter() - ti
            closure = float(ipr.get("max_closure_mm") or 0.0)
            ipr_worst = max(ipr_worst, closure)
            if closure > IPR_WARN_MM or ipr.get("over_threshold") or "error" in ipr:
                ipr_rows.append({"tooth_id": t["tid"], "fdi": t["fdi"], **ipr})

            # NON-BLOCKING. A tooth may legitimately pass through contact on its
            # way somewhere — what the clinician needs is to be told, not
            # stopped. So this is recorded per stage and never raises.
            if ant is not None:
                tc = time.perf_counter()
                hit = cg.check_occlusal_collision(
                    cg.apply_matrix(rec["cv"], M), ant[0], ant[1], index=ant[2])
                t_collide += time.perf_counter() - tc
                if hit["collides"]:
                    interference.append({"tooth_id": t["tid"], "fdi": t["fdi"],
                                         **hit})
        t0 = time.perf_counter()
        try:
            solid = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
            mesh = solid.to_mesh()
            if len(np.asarray(mesh.tri_verts)) == 0:
                raise ValueError("the boolean produced an empty solid")
            csg_info = {"csg_repair_stage": "none"}
        except HTTPException:
            raise
        except Exception as direct_err:              # noqa: BLE001 — cascade
            # REPAIR AND RETRY, THEN REFUSE. The parts are re-read as arrays so
            # each rung can be applied to them; manifold3d transforms lazily, so
            # this costs nothing on the overwhelmingly common path where the
            # direct boolean succeeds and this block never runs.
            part_arrays = []
            for pm in parts:
                pmesh = pm.to_mesh()
                part_arrays.append((np.asarray(pmesh.vert_properties, float)[:, :3],
                                    np.asarray(pmesh.tri_verts, np.int64)))
            solid, csg_info = _batch_union_with_cascade(
                m3, part_arrays, f"Stage {k}")
            csg_info.setdefault("csg_attempts", []).insert(
                0, {"stage": "none",
                    "result": f"{type(direct_err).__name__}: {direct_err}"})
            mesh = solid.to_mesh()
        # Same crumb purge as every other boolean here — see _solid_bodies.
        solid, n_bodies, void_info = _solid_bodies(solid)
        mesh = solid.to_mesh()
        t_union += time.perf_counter() - t0
        sv = np.asarray(mesh.vert_properties, float)[:, :3]
        sf = np.asarray(mesh.tri_verts, np.int64)
        stage_labels = _provenance_labels(mesh, geometry_sources, old_site_pts)
        if len(sf) == 0:
            raise HTTPException(422,
                f"Stage {k} fused to an empty solid. The crowns and the base did not "
                f"intersect, which usually means the prescription has moved a tooth clear "
                f"of the cast.")

        # Print compensation on the FUSED solid. The index buffer is untouched,
        # so every topological assertion below still measures the model that is
        # actually written — only the vertex positions move, each along its own
        # vertex normal. The volume has to be recomputed from the mesh, because
        # `solid` is the pre-offset Manifold and its .volume() no longer
        # describes the STL.
        stage_volume = float(solid.volume())
        if comp > 0.0:
            sv = cg.offset_along_normals(sv, sf, comp)
            stage_volume = float(cg.signed_volume(sv, sf))

        name = f"{arch}_Stage_{k:02d}.stl"

        # EXPORT WELD, before the bytes are written. Measured across the five
        # points the brief asks for, on the OLD pipeline: the raw in-memory
        # fused mesh read 0 non-manifold edges, and the reread of the written
        # file read 68 — because the STL round trip merges coincident
        # positions (5227 -> 5192 vertices) while keeping all 10,466 faces,
        # and 70 of those become degenerate once their corners coincide.
        # `weld_vertices` drops exactly those, so they never reach the file.
        #
        # This is a SERIALISATION-POLICY fix, not the geometry fix. The
        # geometry fix is the local reconstruction above; welding stops a
        # correct solid being corrupted on the way out. Both were needed, and
        # the five-point measurement is what separated them.
        pre_weld = cg.manifold_report(sf)
        # WELD AT THE PRECISION THE FILE WILL HAVE, not at float64.
        #
        # `weld_vertices` merges on EXACT position equality (np.unique over
        # float64 rows) while binary STL stores float32. So two vertices that
        # differ in the ninth decimal survive our weld, are written to the same
        # float32 triple, and the READER's weld then merges them - turning an
        # edge shared by two faces into one shared by three. Measured: bodies
        # 1, components 1, zero open edges, and exactly 1-2 non-manifold edges
        # on intrusion, buccolingual, mesiodistal, rotation and combined
        # movements, with winding reported inconsistent as a consequence.
        #
        # Quantising first makes our weld see precisely what the reader will
        # see, so the round trip can no longer discover a coincidence we did
        # not already resolve. It is NOT a tolerance being loosened: float32 is
        # the format's own precision, and rounding to it is what writing the
        # file does anyway - this only moves that rounding to BEFORE the weld
        # instead of after it.
        sv = sv.astype(np.float32).astype(np.float64)

        # GEOMETRIC SELF-INTERSECTION, measured HERE and nowhere else: after
        # print compensation and after the float32 rounding, so it sees the
        # positions the STL will actually contain. Every topology gate in this
        # file measures connectivity, and a mesh whose triangles pass through
        # each other is closed, manifold, one component and correctly wound.
        # `cg.offset_along_normals` is a vertex-normal offset and this project
        # has measured that operation folding at concavities three times
        # (s.8's offset table; s.23's LOCAL_CLEARANCE self-touch), so the one
        # step that can introduce it is immediately above.
        _t_si = time.perf_counter()
        self_intersection = si_mod.self_intersection_report(sv, sf)
        self_intersection["seconds"] = round(time.perf_counter() - _t_si, 3)

        _pre_weld_v, sf_pre_weld = sv, sf
        # AND ITS OWN LABELS. `stage_labels` is re-indexed onto the
        # WELDED faces twenty lines below, so keeping only the
        # pre-weld faces here left `serialisation_forensics` indexing
        # 168,562 faces into a 168,560-entry label array. Measured on
        # the real scan it raised `IndexError: index 168560 is out of
        # bounds`, was swallowed by the try/except around the call,
        # and the manifest reported `edge_forensics.ran: false` - so
        # the one tool built to attribute an offending edge to the
        # geometry that made it was silently absent on exactly the
        # stages that have one. The comment below already warns about
        # this class of mistake; it happened anyway, at the call site.
        _pre_weld_labels = stage_labels

        # SELF-TOUCH, MEASURED BEFORE THE WELD. manifold3d records a boundary
        # that touches itself as two vertices at an IDENTICAL position under
        # different indices; the control in mfg.self_touch_report shows a
        # clean transversal union produces none. Binary STL stores positions,
        # so welding is not optional - which means every self-touch the
        # boolean leaves becomes a non-manifold edge in the file. This is the
        # number that actually predicts the gate, so it is recorded on EVERY
        # stage, passing or not.
        touch = mfg.self_touch_report(sv, sf, stage_labels, limit=12)

        # THE WELD DROPS DEGENERATE FACES, so the provenance array has to come
        # with it: measured, 9448 labelled triangles against 9444 surviving
        # ones, and indexing one with the other is an error that only shows up
        # on the stages where something actually collapsed.
        sv, sf, export_welded, _kept = mfg._weld_with_face_map(sv, sf)
        stage_labels = stage_labels[_kept]
        post_weld = cg.manifold_report(sf)

        # WHAT WOULD A READER DO IF WE DID NOT WELD? The control for the whole
        # serialisation argument: without it the chain cannot tell "our weld
        # created this" from "any reader would have".
        #
        # GATED, AND NOT BECAUSE IT IS UNIMPORTANT. Benchmarked on a
        # 203,522-face mesh - the real cast base is 190,036 - the write, parse,
        # weld and two reports cost 0.90s PER STAGE, so a 31-stage plan paid
        # ~28s of forensics on top of a 4.2s export, and /export/final paid it
        # again. It answers a question that only arises when something is
        # wrong, so it runs when something is wrong. Wrapped, because a
        # diagnostic must never be the thing that fails an export: compare the
        # interproximal block above.
        unwelded_probe = {"ran": False,
                          "reason": "no non-manifold edge after the export weld"}
        if post_weld["nonmanifold_edges"] or pre_weld["nonmanifold_edges"]:
            try:
                _pb = cg.write_binary_stl_bytes(_pre_weld_v, sf_pre_weld)
                _ppv, _ppf = stl_io.parse_stl_bytes(_pb)
                _praw = cg.manifold_report(_ppf)
                _pwv, _pwf, _pmerged = cg.weld_vertices(_ppv, _ppf)
                _pw = cg.manifold_report(_pwf)
                unwelded_probe = {
                    "ran": True,
                    "written_unwelded_faces": int(len(sf_pre_weld)),
                    "reread_open": int(_praw["open_edges"]),
                    "reread_nonmanifold": int(_praw["nonmanifold_edges"]),
                    "reader_weld_merged": int(_pmerged),
                    "after_reader_weld_open": int(_pw["open_edges"]),
                    "after_reader_weld_nonmanifold": int(_pw["nonmanifold_edges"])}
            except Exception as e:                        # noqa: BLE001
                unwelded_probe = {"ran": False,
                                  "reason": f"{type(e).__name__}: {e}"}

        # A RECORDED REPAIR RUNG, not a silent one. A grazing CSG contact at
        # the cervical margin leaves one or two very short edges carrying four
        # faces; see collapse_short_nonmanifold_edges for the measurement. It
        # is attempted only when the weld left something non-manifold, it is
        # capped at scanner resolution, and it is KEPT ONLY IF IT WORKED -
        # otherwise the original mesh goes to the gates and is refused there.
        nm_repair = {"attempted": False}
        if post_weld["nonmanifold_edges"] > 0:
            cv2, cf2, rinfo = mfg.collapse_short_nonmanifold_edges(sv, sf)
            after = cg.manifold_report(cf2)
            # ALL OR NOTHING. A partial improvement is not worth keeping: the
            # gates refuse the stage either way, and shipping a mesh that has
            # been altered but is still invalid makes the refusal harder to
            # diagnose, not easier. Only a complete fix - zero non-manifold,
            # zero open, consistent winding - is kept.
            better = (after["nonmanifold_edges"] == 0
                      and after["open_edges"] == 0
                      and cg._winding_is_consistent(cf2))
            rinfo_json = {kk: vv for kk, vv in rinfo.items()
                          if kk != "kept_faces"}
            nm_repair = {"attempted": True, "kept": bool(better), **rinfo_json,
                         "nonmanifold_before": int(post_weld["nonmanifold_edges"]),
                         "nonmanifold_after": int(after["nonmanifold_edges"]),
                         "open_after": int(after["open_edges"])}
            if better:
                sv, sf = cv2, cf2
                stage_labels = stage_labels[rinfo["kept_faces"]]
                post_weld = after

        # === EVERY REQUIRED MEASUREMENT, ON THE MODEL THAT WAS WRITTEN =====
        #
        # These used to be functions with tests and no caller. A metric that is
        # tested and not integrated proves the arithmetic and gates nothing, so
        # each one is computed here, recorded per stage, and read by
        # mfg.aggregate_print_gate - which fails on a MISSING measurement
        # exactly as it fails on a bad one.
        t_metrics0 = time.perf_counter()

        # THE ALLOWED RECONSTRUCTION ENVELOPE. Geodesic, not a box: a
        # rectangular XYZ region around a rim on a curved arch sweeps in the
        # neighbouring teeth and the gingiva on the far side of the ridge. It
        # is the union, over every moved tooth, of the cast surface within
        # roi_radius_mm of the TARGET rim and of the T0 rim - target site and
        # old site, which are the only two places this pipeline is allowed to
        # change the cast.
        env_pts = []
        for t in teeth:
            rim0_ = v[np.asarray(t["rec"]["socket_rim"], np.int64)]
            env_pts.append(rim0_)
            env_pts.append(cg.apply_matrix(rim0_, stage_matrices[t["tid"]]))
        env_pts = np.vstack(env_pts) if env_pts else np.zeros((0, 3))
        env_radius = (mfg.DEFAULT_POLICY.roi_radius_mm
                      + mfg.DEFAULT_POLICY.seat_bottom_outset_mm)

        # The cast, as it survives in the finished model. Provenance is what
        # makes this exact: only triangles that came from the cast are
        # compared with the cast, so a crown 2mm above the gingiva cannot be
        # mistaken for the gingiva having moved.
        cast_mask = np.isin(stage_labels, ["ORIGINAL_CAST", "OLD_SOCKET_REPAIR"])
        cast_f = sf[cast_mask]
        ov_, of_ = original_cast
        fidelity = {"measured": False, "reason": "no cast surface survived"}
        roi_compliance = {"compliant": False, "reason": "not measured"}
        if len(cast_f):
            keep_v = np.unique(cast_f)
            cast_pts = sv[keep_v]
            d_env = (cKDTree(env_pts).query(cast_pts, workers=-1)[0]
                     if len(env_pts) else np.full(len(cast_pts), 1e9))
            outside = d_env > env_radius
            # TWO-SIDED, because one direction cannot see the other's holes:
            # original -> final misses material that was ADDED, final ->
            # original misses material that was REMOVED.
            f2o = mfg.surface_deviation(ov_, of_, sv, cast_f,
                                        exclude_pts=env_pts,
                                        exclude_radius_mm=env_radius)
            o2f = mfg.surface_deviation(sv, cast_f, ov_, of_,
                                        exclude_pts=env_pts,
                                        exclude_radius_mm=env_radius)
            fidelity = {"measured": True,
                        "envelope_radius_mm": round(float(env_radius), 4),
                        "final_to_original": f2o,
                        "original_to_final": o2f}

            # ROI COMPLIANCE. Every cast point that MOVED must be inside the
            # envelope. Measured by triangle-surface distance, never by vertex
            # index: a boolean retessellates, so index equality would fail on
            # a perfect result and pass on a resampled bad one.
            moved_d = mfg._point_to_surface(cast_pts, ov_, of_)
            moved = moved_d > mfg.DEFAULT_POLICY.max_unaffected_deviation_mm
            viol = moved & outside
            tri_ok = np.isin(cast_f, keep_v[viol]).any(axis=1)
            viol_area = float(mfg._face_areas(sv, cast_f[tri_ok]).sum())
            roi_compliance = {
                "compliant": bool(not viol.any()),
                "cast_points_compared": int(len(cast_pts)),
                "points_outside_envelope": int(outside.sum()),
                "modified_points": int(moved.sum()),
                "modified_points_outside_envelope": int(viol.sum()),
                "outside_envelope_area_mm2": round(viol_area, 4),
                "worst_outside_deviation_mm": round(
                    float(moved_d[outside].max()) if outside.any() else 0.0, 6),
                "envelope_definition": "surface neighbourhood of the T0 rim and "
                                       "the target rim, roi_radius + "
                                       "seat_bottom_outset",
            }

        exposure = mfg.synthetic_exposure(sv, sf, stage_labels)

        # VOLUMES, SEPARATED. One total tells a lab nothing about where the
        # material came from; these say exactly how much cast there was, how
        # much the clearance took out, how much the connectors and the crowns
        # put back, and how much of that was already shared.
        volumes = {
            "original_cast_mm3": round(float(base_solid.volume()), 4),
            "cast_after_local_clearance_mm3": round(float(prepared.volume()), 4),
            "local_clearance_removed_mm3":
                clearance_report["removed_volume_mm3"] or 0.0,
            "connector_total_mm3": round(float(sum(
                r.get("connector_volume_mm3") or 0.0 for r in interfaces)), 4),
            "crown_total_mm3": round(float(sum(
                float(crown_solids[t["tid"]].volume()) for t in teeth)), 4),
            "crown_cast_overlap_mm3": round(float(sum(
                r.get("overlap_crown_cast_mm3") or 0.0 for r in interfaces)), 4),
            "connector_cast_overlap_mm3": round(float(sum(
                r.get("overlap_seat_cast_mm3") or 0.0 for r in interfaces)), 4),
            "fused_mm3": round(float(solid.volume()), 4),
        }

        for row in interfaces:
            tid = row["tooth_id"]
            rec = next(t["rec"] for t in teeth if t["tid"] == tid)
            M_ = stage_matrices[tid]
            rim0_ = v[np.asarray(rec["socket_rim"], np.int64)]
            rim_k_ = cg.apply_matrix(rim0_, M_)
            u_k = M_[:3, :3] @ np.asarray(rec["frame"]["u_oa"], float)
            u_k = u_k / (np.linalg.norm(u_k) or 1.0)
            try:
                row["transition"] = mfg.transition_quality(sv, sf, rim_k_, u_k)
            except Exception as e:                       # noqa: BLE001
                row["transition"] = {"error": f"{type(e).__name__}: {e}"}
            try:
                row["old_site"] = mfg.old_site_quality(
                    ov_, of_, sv, cast_f if len(cast_f) else sf, rim0_,
                    u_oa=np.asarray(rec["frame"]["u_oa"], float))
            except Exception as e:                       # noqa: BLE001
                row["old_site"] = {"measured": False,
                                   "reason": f"{type(e).__name__}: {e}"}
            row["rigidity"] = mfg.rigidity_report(
                rec["cv"], cg.apply_matrix(rec["cv"], M_), rec["cf"],
                matrix=M_)

        # ADJACENT RECONSTRUCTIONS, MEASURED. The old check compared the
        # gingival bridge with `2 * fusion_overlap_mm` - a nominal number that
        # describes no geometry that was built. What actually threatens the
        # bridge is how much of it the two CONNECTORS occupy, so that is what
        # is measured: the clear gap left between the two reconstruction
        # solids, against the bridge that was there before them.
        adjacent = []
        for i_ in range(len(teeth)):
            for j_ in range(i_ + 1, len(teeth)):
                ti_, tj_ = teeth[i_]["tid"], teeth[j_]["tid"]
                ri_ = cg.apply_matrix(
                    v[np.asarray(teeth[i_]["rec"]["socket_rim"], np.int64)],
                    stage_matrices[ti_])
                rj_ = cg.apply_matrix(
                    v[np.asarray(teeth[j_]["rec"]["socket_rim"], np.int64)],
                    stage_matrices[tj_])
                bridge = mfg.bridge_between(ri_, rj_)
                gi, gj = connector_pts.get(ti_), connector_pts.get(tj_)
                if gi is None or gj is None or bridge <= 1e-9:
                    adjacent.append({"teeth": [ti_, tj_],
                                     "bridge_mm": round(float(bridge), 4),
                                     "ok": True,
                                     "note": "no reconstruction geometry to compare"})
                    continue
                gap = float(cKDTree(gj).query(gi, workers=-1)[0].min())
                consumed = max(0.0, bridge - gap)
                frac = consumed / bridge
                adjacent.append({
                    "teeth": [ti_, tj_],
                    "bridge_mm": round(float(bridge), 4),
                    "clear_gap_between_reconstructions_mm": round(gap, 4),
                    "consumed_mm": round(consumed, 4),
                    "consumed_fraction": round(frac, 4),
                    "max_fraction": mfg.DEFAULT_POLICY.max_bridge_removal_fraction,
                    "ok": bool(frac <= mfg.DEFAULT_POLICY.max_bridge_removal_fraction),
                    "measure": "minimum distance between the two connector solids, "
                               "not 2 * fusion_overlap_mm"})

        # THE STAGE'S PRESCRIPTION IS ITS OWN SHARE OF THE COMMITTED ONE, and
        # the final stage IS the committed one. Checked rather than assumed,
        # because a staging bug would otherwise reach the lab as geometry.
        clin_bad = []
        for t in teeth:
            want = {kk: float(vv) * k / total
                    for kk, vv in t["clinical"].items()}
            got = _stage_clinical(t["clinical"], k, total)
            for kk in want:
                if abs(float(got.get(kk, 0.0)) - want[kk]) > 1e-9:
                    clin_bad.append({"tooth_id": t["tid"], "field": kk,
                                     "expected": want[kk],
                                     "got": float(got.get(kk, 0.0))})

        t_metrics += time.perf_counter() - t_metrics0

        blob = cg.write_binary_stl_bytes(sv, sf)

        edge_forensics = {"ran": False, "reason": "no offending edge"}
        if post_weld["nonmanifold_edges"] or post_weld["open_edges"]:
            try:
                edge_forensics = mfg.serialisation_forensics(
                    _pre_weld_v, sf_pre_weld, _pre_weld_labels, limit=8,
                    write=cg.write_binary_stl_bytes,
                    read=stl_io.parse_stl_bytes)
                edge_forensics["ran"] = True
            except Exception as e:                        # noqa: BLE001
                edge_forensics = {"ran": False,
                                  "reason": f"{type(e).__name__}: {e}"}

        # WHAT IS ASSERTED, AND WHAT IS ONLY REPORTED — the distinction matters
        # and is not the one /export uses.
        #
        # ASSERTED: the fused solid is closed and single-bodied as an index
        # buffer. manifold3d guarantees that and it is the property that makes
        # the model printable.
        #
        # REPORTED, NOT ASSERTED: whether it survives a reader that WELDS
        # coincident vertices. /export holds its base to that bar because there
        # the duplicates were a bug in geometry this codebase builds. Here they
        # are not. Putting a crown back into the hole it was cut from is a
        # tangential boolean — the crown's outer surface and the cast's outer
        # surface are the same scan faces, meeting edge to edge at the rim — and
        # manifold3d resolves that by emitting topologically distinct vertices
        # at identical positions. Binary STL cannot express that distinction, so
        # a welding reader merges them and sees non-manifold edges.
        #
        # Five ways of separating the surfaces were measured and none removed it
        # across prescriptions: a root-length plug (0-42 edges), lifting the
        # plug into the crown, insetting it to 75%, seating the tooth 0.1-2.5mm
        # into the gingiva (52 -> 22 but never 0), and eroding the socket by 2-4
        # face rings (74-104, and at 4 rings the tooth stops touching the base
        # at all). The count moves with the prescription, which is the signature
        # of a tangency rather than a construction bug.
        #
        # So it is measured and written into the manifest per stage. A lab whose
        # slicer welds will see these; one that does not, will not. Refusing on
        # it would block every export while the geometry is in fact closed.
        # === THE HARD GATE, ON THE ACTUAL BYTES =============================
        # This is the correction the whole validation half of the brief turns
        # on. The old code DID reread the STL and DID measure it into `welded`
        # — and then gated on `manifold_report(sf)`, the IN-MEMORY buffer, so
        # 68-77 non-manifold edges per stage were recorded in the manifest and
        # shipped regardless. A validator that reads a different object from
        # the one the lab receives is not a validator.
        validation = mfg.validate_printable_stl(blob, expect_components=1)
        welded = {"open_edges": validation["open_edges"],
                  "nonmanifold_edges": validation["nonmanifold_edges"],
                  "total_edges": validation["total_edges"],
                  "watertight": validation["open_edges"] == 0
                                and validation["nonmanifold_edges"] == 0}
        n_comp = validation["connected_components"]

        mstat = mfg.manifold_status(solid, m3)
        if require_print_ready and not mstat.get("single_positive_body"):
            raise HTTPException(422, _jsonable({
                "error": f"Stage {k} is not ONE positive-volume manifold body.",
                "manifold_status": mstat,
                "detail": "manifold3d's decompose() is the physical body count; "
                          "the STL component count is a separate check and both "
                          "must pass.",
                "stage": k}))

        if require_print_ready and not validation["print_ready"]:
            raise HTTPException(422, _jsonable({
                "error": f"Stage {k} failed manufacturing validation of the "
                         f"WRITTEN STL.",
                "failed_gates": validation["failed_gates"],
                "gates": validation["gates"],
                "detail": "Measured on the bytes that would have been shipped, "
                          "after simulating a downstream reader's weld — not on "
                          "the in-memory boolean result.",
                "stage": k}))

        # manifold3d's own body count, cross-checked against the written file's
        # connected components. They answer different questions and both have
        # to agree before a stage ships.
        if require_print_ready and n_bodies != 1:
            raise HTTPException(422,
                f"Stage {k} fused into {n_bodies} separate solids. A tooth has moved clear "
                f"of the cast and is floating — the model cannot be thermoformed.")

        blobs[name] = blob
        stage_meta.append({
            "stage": k, "file": name, "triangles": int(len(sf)),
            "volume_mm3": round(stage_volume, 3),
            "components": int(n_comp),
            # `inverted_crumbs_discarded` is kept as an alias for one release;
            # it always equals `internal_voids`. See `_solid_bodies`.
            mfg.KEY_SELF_INTERSECTION: self_intersection,
            KEY_INTERNAL_VOIDS: int(void_info[KEY_INTERNAL_VOIDS]),
            KEY_INTERNAL_VOIDS_FILLED: float(
                void_info[KEY_INTERNAL_VOIDS_FILLED]),
            KEY_CRUMBS_ALIAS: int(void_info[KEY_CRUMBS_ALIAS]),
            "solid_decomposition": void_info,
            # "none" on the ordinary path. Anything else means the boolean
            # failed on the meshes as built and this stage was printed from
            # repaired geometry — which the lab is entitled to know.
            "csg_repair_stage": csg_info.get("csg_repair_stage", "none"),
            "fallback_reason": csg_info.get("fallback_reason"),
            "fallback_detail": csg_info.get("fallback_detail"),

            # --- local target-position reconstruction, per tooth -----------
            # Everything the brief's amendment 16 asks to record. The ROI is a
            # geodesic surface region; the bboxes beside it are DIAGNOSTICS.
            "interfaces": interfaces,
            # WHAT WAS ACTUALLY REMOVED FROM THE CAST, measured as the volume
            # difference the subtraction made. Reporting the tool's own volume
            # here told a lab 85.16mm3 of cast had been excavated on a stage
            # where nothing was cut at all.
            "interface_total_cavity_volume_mm3":
                clearance_report["removed_volume_mm3"] or 0.0,
            "clearance": clearance_report,
            "interface_total_ramp_volume_mm3": round(sum(
                i.get("ramp_volume_mm3") or 0.0 for i in interfaces), 4),
            "interface_total_affected_faces": int(sum(
                i.get("affected_faces") or 0 for i in interfaces)),
            "interface_total_affected_area_mm2": round(sum(
                i.get("affected_area_mm2") or 0.0 for i in interfaces), 4),

            # --- the five-point serialisation chain (amendments 11, 12) -----
            "edges_pre_export_weld": {
                "open": int(pre_weld["open_edges"]),
                "nonmanifold": int(pre_weld["nonmanifold_edges"])},
            "edges_post_export_weld": {
                "open": int(post_weld["open_edges"]),
                "nonmanifold": int(post_weld["nonmanifold_edges"])},
            "export_weld_merged_vertices": int(export_welded),
            "nonmanifold_edge_repair": nm_repair,
            "unwelded_serialisation_probe": unwelded_probe,
            # THE OFFENDING EDGES THEMSELVES, not just how many. Edge id,
            # endpoint coordinates, length, every incident face with its area,
            # its normal and the geometry it came from, at each point of the
            # chain. Computed only when there is something to explain - the
            # dissection is cheap, but writing and rereading the STL a second
            # time is not.
            "edge_forensics": edge_forensics,
            "self_touch": touch,
            "edges_post_read_and_reader_weld": {
                "open": int(validation["open_edges"]),
                "nonmanifold": int(validation["nonmanifold_edges"])},
            "reader_weld_merged_vertices": int(
                validation["reader_weld_merged_vertices"]),

            # --- the verdict, from the written bytes -----------------------
            # --- every integrated manufacturing measurement ---------------
            "synthetic_exposure": exposure,
            "cast_fidelity": fidelity,
            "roi_compliance": roi_compliance,
            "adjacent_bridges": adjacent,
            "clinical_consistent": not clin_bad,
            "clinical_consistency_detail": clin_bad or None,
            "volumes_mm3": volumes,

            "stl_validation": validation,
            # manifold3d's own account of the solid, recorded after the last
            # boolean. It answers a DIFFERENT question from the STL gates -
            # physical bodies rather than index-buffer components - and the
            # verdict below requires both.
            "manifold_status": mstat,
            "manifold_bodies": int(n_bodies),
            # NARROW verdict from the file alone. The aggregate gate below is
            # the only thing entitled to say a stage may be manufactured.
            "passes_boolean_topology_regression": bool(
                validation["print_ready"] and mstat.get("single_positive_body")),
            "occlusal_interference": interference,
            "occlusal_warning": ANTAGONIST_WARNING if interference else None,
            # YELLOW, not red, and deliberately: closing an embrasure is often
            # the intent of the plan. What the clinician needs is to be told
            # WHEN it happens, on which tooth, and by how much.
            "interproximal_max_closure_mm": round(ipr_worst, 4),
            "interproximal_state": ("YELLOW" if ipr_worst > IPR_WARN_MM else "GREEN"),
            "interproximal_threshold_mm": IPR_WARN_MM,
            "interproximal": ipr_rows,
            "interproximal_warning": (
                f"Interproximal closure reaches {ipr_worst:.2f}mm at this stage "
                f"(threshold {IPR_WARN_MM}mm). This is a MEASURED GAP CLOSURE, "
                f"not a prescription for enamel reduction, and a vertex-to-vertex "
                f"minimum OVERESTIMATES the true clearance."
                if ipr_worst > IPR_WARN_MM else None),
            "closed": True,
            "genus": int(solid.genus()),
            # Honest, per stage. See the note above for why this is reported
            # rather than refused on.
            "welded_open_edges": int(welded["open_edges"]),
            "welded_nonmanifold_edges": int(welded["nonmanifold_edges"]),
            "survives_a_welding_reader": bool(welded["watertight"]),
            "teeth": [{"tooth_id": t["tid"], "fdi": t["fdi"],
                       "clinical": _stage_clinical(t["clinical"], k, total)}
                      for t in teeth],
        })

        # === THE AGGREGATE MANUFACTURING GATE =============================
        # The ONLY thing in this codebase entitled to say PRINT READY. It is a
        # pure function of the record just built, so a measurement that was
        # never taken fails it exactly as a bad one does.
        gate = mfg.aggregate_print_gate(stage_meta[-1])
        stage_meta[-1]["manufacturing_gate"] = gate
        stage_meta[-1]["print_ready"] = bool(gate["print_ready"])
        stage_meta[-1]["verdict"] = gate["verdict"]

    manifest = {
        "generator": "Clinical Micro-Planner",
        "arch": arch,
        "kind": "staged manufacturing models — fused solids, ready to thermoform",
        "stages": total,
        "units": "mm",
        "coordinate_space": "raw scanner coordinates — never re-centred or rescaled",
        "socket_treatment": "filled flush at the gingival margin; the tooth sits proud",
        # Stated on every export, 0.0 included. A lab reading "0.0" knows the
        # model is true to anatomy; a lab reading nothing has to assume.
        "print_compensation_mm": round(comp, 4),
        "print_compensation_note": (
            "true to anatomy; no allowance applied" if comp <= 0.0 else
            f"every surface of the FUSED model is offset {comp} mm outward along its "
            f"vertex normal, applied AFTER the union. The model is deliberately not "
            f"true to anatomy: subtract {comp} mm before measuring it against the plan."),
        "base_construction": "trim_to_arch -> build_cast_base (never cap_and_close)",
        "trim": {k: val for k, val in trim_info.items() if k not in ("curve", "rim_loop")},
        "cast_base": {k: val for k, val in base_info.items() if k != "prune"},
        "sockets": sockets,
        "teeth": [{"tooth_id": t["tid"], "fdi": t["fdi"],
                   "prescription": t["clinical"],
                   "stages_required": t["staging"]["stages_required"],
                   "driver": t["staging"]["driver"],
                   "binds_the_case": t["staging"]["stages_required"] == total,
                   "root_length_mm": t["rec"].get("root_length_mm")} for t in teeth],
        "stage_files": stage_meta,
        # WHERE THE TIME GOES, per phase, measured rather than assumed. No
        # premature optimisation: these are recorded so a future change has a
        # baseline to beat, and so a lab waiting on an export can be told
        # honestly what it is waiting for.
        "union_seconds": round(t_union, 2),
        "phase_seconds": {
            "cast_build": round(t_cast, 3),
            "local_interfaces": round(t_iface, 3),
            "boolean_union": round(t_union, 3),
            "quality_metrics": round(t_metrics, 3),
            "interproximal": round(t_ipr, 3),
            "antagonist_collision": round(t_collide, 3),
            "total_bundle": round(time.perf_counter() - t_bundle0, 3),
            "stages": total,
            "teeth": len(teeth),
        },
        # Case-level occlusion summary. `checked` false means the opposing arch
        # was never loaded — which is NOT the same as "no interference", and the
        # manifest says so rather than leaving a lab to assume clearance.
        "occlusion": {
            "checked": ant is not None,
            "threshold_mm": 0.1,
            "stages_with_interference": [s["stage"] for s in stage_meta
                                         if s["occlusal_interference"]],
            "worst_penetration_mm": round(max(
                (i["max_penetration_mm"] for s in stage_meta
                 for i in s["occlusal_interference"]), default=0.0), 4),
            "seconds": round(t_collide, 3),
            "warning": ANTAGONIST_WARNING if any(
                s["occlusal_interference"] for s in stage_meta) else None,
        },
        # Case-level interproximal summary, swept across every stage rather than
        # sampled at the endpoint. The seconds are here so the cost of the sweep
        # is a number in the manifest, not an assumption.
        "interproximal": {
            "threshold_mm": IPR_WARN_MM,
            "stages_yellow": [s["stage"] for s in stage_meta
                              if s["interproximal_state"] == "YELLOW"],
            "worst_closure_mm": round(max(
                (s["interproximal_max_closure_mm"] for s in stage_meta),
                default=0.0), 4),
            "seconds": round(t_ipr, 3),
            "limitation": ("A vertex-to-vertex minimum OVERESTIMATES the true "
                           "clearance: on a triangulated surface the closest "
                           "points generally lie inside faces, not at vertices. "
                           "This measures a gap closure and does not prescribe "
                           "enamel reduction."),
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, blob in blobs.items():
            z.writestr(name, blob)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    buf.seek(0)

    out_dir = None
    if req.keep_local:
        out_dir = str(_confined_out_dir(req.out_dir or os.path.join(
            EXPORTS_ROOT, sid[:8], "stages")))
        os.makedirs(out_dir, exist_ok=True)
        for name, blob in blobs.items():
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(blob)
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

    _record(sid, audit.EXPORTED, arch=arch,
            detail=f"{total} staged manufacturing models",
            values={"stages": int(total), "teeth": len(teeth),
                    # WHERE THE TIME GOES, per phase, measured rather than assumed. No
        # premature optimisation: these are recorded so a future change has a
        # baseline to beat, and so a lab waiting on an export can be told
        # honestly what it is waiting for.
        "union_seconds": round(t_union, 2),
        "phase_seconds": {
            "cast_build": round(t_cast, 3),
            "local_interfaces": round(t_iface, 3),
            "boolean_union": round(t_union, 3),
            "quality_metrics": round(t_metrics, 3),
            "interproximal": round(t_ipr, 3),
            "antagonist_collision": round(t_collide, 3),
            "total_bundle": round(time.perf_counter() - t_bundle0, 3),
            "stages": total,
            "teeth": len(teeth),
        },
                    "interproximal_seconds": round(t_ipr, 2),
                    "collision_seconds": round(t_collide, 2),
                    "stages_yellow": len([s for s in stage_meta
                                          if s["interproximal_state"] == "YELLOW"])})
    telemetry.record_span(
        "staging", (time.perf_counter() - _t_stage) * 1000,
        arch=arch, stages=int(total), teeth=len(teeth),
        union_seconds=round(t_union, 3),
        interproximal_seconds=round(t_ipr, 3),
        collision_seconds=round(t_collide, 3),
        repaired_stages=len([s for s in stage_meta if s.get("fallback_reason")]))

    return {"buf": buf, "filename": f"{arch}_stages.zip", "manifest": manifest,
            "stages": total, "out_dir": out_dir, "union_seconds": t_union,
            # /export/final picks ONE of these rather than rebuilding the
            # bundle, so the file it ships is byte-identical to the one that
            # was validated.
            "blobs": blobs,
            # The IMMUTABLE cast this bundle was actually built from. The
            # fidelity test has to compare the stage against THIS, not against
            # a cast rebuilt with its own parameters - a different trim margin
            # or base thickness makes every vertex differ and reads as metres
            # of phantom deformation.
            "base_mesh": (bv, bf)}


# ---------------------------------------------------------------------------
# The deformation construction, behind `construction="deformation"`
# ---------------------------------------------------------------------------

def _v2_evidence_report(gate: dict) -> dict:
    """`aggregate_gate_v3`'s own REQUIRED gates, restated as claims.

    NOT `mfg.manufacturing_evidence_report`, and the difference is not
    cosmetic. That function reads the COLLAR's keys - interface continuity,
    transition ledge, old-site restoration, synthetic exposure - and the
    deformation construction builds none of those things, so every one of its
    rows would read `false` for the honest reason that the measurement does
    not exist here. A claim that reads false because it was never applicable
    is indistinguishable, in a manifest, from one that failed.

    So each row here names a gate that WAS required and measured - the list
    is the gate's own `required_gates`, which differs between the T0 export
    and a moved stage - and `all_claims_verified` is the aggregate gate's own
    answer, never a second opinion about it.
    """
    gates = gate.get("gates") or {}
    return {
        "construction": mfg2.CONSTRUCTION_ID,
        "print_model": mfg2.PRINT_MODEL_ID,
        "all_claims_verified": bool(gate.get("print_ready")),
        "claims": [{"claim": name, "verified": bool(gates.get(name, {}).get("ok")),
                    "evidence": gates.get(name)}
                   for name in (gate.get("required_gates") or [])],
        "advisory": gate.get("advisory") or {},
        "note": ("These are aggregate_gate_v3's required gates restated one "
                 "for one. The collar path's evidence report does not apply "
                 "to this construction and is deliberately not produced."),
    }


def _v2_attachment_solids(teeth, matrices):
    """Rebuild each attachment from its RECORD and carry it by the stage matrix.

    REBUILT, NOT STORED. `place_attachment` keeps the parameters - shape,
    dimensions, seat point, normal, rotation - and fuses the geometry into the
    crown; the attachment's own mesh is not persisted anywhere. That is the
    project's standing rule (s.15: the plan is prescriptions, derived visuals
    are rederivable and none is persisted), and `build_attachment` is
    deterministic in those parameters, so rebuilding reproduces exactly what
    the clinician placed.

    It is carried by the SAME stage matrix the crown's vertices are, so the
    attachment cannot drift off the tooth it is bonded to.
    """
    solids = []
    for t in teeth:
        recs = t["rec"].get("attachments") or []
        M = matrices.get(t["tid"])
        if not recs or M is None:
            continue
        for rec in recs:
            dims = rec.get("dimensions_mm") or {}
            att = attachments_mod.build_attachment(
                rec["shape"], t["rec"]["frame"], rec["position_xyz"],
                size_mm={k: float(dims[k]) for k in ("md", "oa", "bl")
                         if k in dims} or None)
            solids.append({
                "tooth_id": t["tid"],
                "attachment_id": rec.get("attachment_id"),
                "shape": rec["shape"],
                "volume_mm3": rec.get("fused_volume_mm3"),
                "verts": cg.apply_matrix(np.asarray(att["verts"], float), M),
                "faces": np.asarray(att["faces"], np.int64)})
    return solids


def _v2_contact_spec(f, labels, movers, moving_faces, static_faces, sets):
    """Each moved tooth paired with its neighbours along the arch (step 5).

    None when there are no segmentation labels: a neighbour is identified by
    its FDI, and without labels no neighbour can be named - so the contact
    gate fails as NOT MEASURED rather than passing on "none found". A moving
    tooth with no FDI is listed as unmeasured, for the same reason.

    Each side is the faces that tooth wholly owns, restricted to the vertices
    `tooth_vertex_sets` gave it - the same sets the plan was solved on.
    """
    if labels is None:
        return None
    F = np.asarray(f, np.int64)
    present = (set(int(x) for x in np.unique(labels)) - {0}) | \
        {int(t["fdi"]) for t in movers if t["fdi"]}
    by_fdi = {int(t["fdi"]): t for t in movers if t["fdi"]}

    def patch(mask, ids):
        own = np.zeros(int(F.max()) + 1, bool)
        own[np.asarray(ids, np.int64)] = True
        faces = F[np.asarray(mask, bool)]
        return faces[own[faces].all(axis=1)]

    contacts, unmeasured = [], []
    for t in movers:
        if not t["fdi"]:
            unmeasured.append(f"tooth {t['tid']} has no FDI label, so its "
                              f"neighbours cannot be identified")
            continue
        fdi = int(t["fdi"])
        mids = sets["moving"][t["tid"]]
        for n in mfg2.arch_neighbours(fdi, present):
            if n in by_fdi:
                other = by_fdi[n]
                nids = sets["moving"][other["tid"]]
                nfaces = patch(moving_faces[other["tid"]], nids)
            elif f"fdi{n}" in static_faces:
                nids = sets["static_by_tooth"][f"fdi{n}"]
                nfaces = patch(static_faces[f"fdi{n}"], nids)
            else:
                continue
            if not len(nfaces):
                unmeasured.append(f"{mfg2.contact_key(fdi, n)}: the neighbour "
                                  f"has no surface of its own to measure")
                continue
            contacts.append({"moving_fdi": fdi, "neighbour_fdi": int(n),
                             "neighbour_moving": n in by_fdi,
                             "moving_ids": mids,
                             "moving_faces": patch(moving_faces[t["tid"]], mids),
                             "neighbour_ids": nids, "neighbour_faces": nfaces,
                             "u_md": np.asarray(t["rec"]["frame"]["u_md"], float)})
    return {"contacts": contacts, "unmeasured": unmeasured}


def build_stage_bundle_v2(sid: str, req: StageExportRequest,
                          only_stage: str | None = None) -> dict:
    """The deformation construction, printed as a voxel-solidified SOLID.

    SAME RETURN SHAPE as `build_stage_bundle`, deliberately: `/export/stages`
    and `/export/final` then need a dispatch and nothing else. What is NOT
    shared is the gate - `mfg2.aggregate_gate_v3` decides here.

    Per stage (Task 2): the T0 cast with the teeth in place -> the existing
    deformation (crown rigid, gingiva harmonic) -> solidify -> write the STL
    -> measure the RE-READ bytes -> one aggregate verdict. No socket is ever
    opened, no `clear_undercut_periphery`, and a T0 cast that crosses itself
    is no longer refused: the solid is what ships and it is gated on its own
    bytes.

    NO MOVEMENT IS THE T0 EXPORT, not an error. With nothing cut, or every
    prescription zero, the bundle is ONE file - stage 0, the T0 cast itself,
    solidified and gated like any stage.

    `only_stage`  None builds every stage (`/export/stages`); "final" builds
    only the stage `req.stage` names (0 = the last). Each stage is rebuilt
    from clinical x k/N, never from its neighbour, so one stage needs no
    other - and a solid costs about a minute, so `/export/final` should not
    pay for stages it will not ship.
    """
    t0_bundle = time.perf_counter()
    try:
        v, f = STORE.require(sid, "verts"), STORE.require(sid, "faces")
        arch = STORE.arch(sid)
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    af = STORE.get(sid, "arch_frame")
    if af is None:
        raise HTTPException(409,
            "Occlusal plane not defined for this session, so the cast base "
            "cannot be built.")

    # --- the teeth, and how long the case is ------------------------------
    teeth = []
    for key in STORE.keys(sid):
        if not key.startswith("tooth:"):
            continue
        t = STORE.get(sid, key)
        clinical = t.get("clinical") or {}
        st = cg.staging_estimate(**{k: float(clinical.get(k, 0.0) or 0.0) for k in
                                    stage_matrix.CLINICAL_KEYS})
        teeth.append({"tid": key.split(":", 1)[1], "rec": t, "clinical": clinical,
                      "staging": st, "fdi": _tooth_fdi(sid, t, f)})

    total = max((t["staging"]["stages_required"] for t in teeth), default=0)
    if total > req.max_stages:
        binding = max(teeth, key=lambda t: t["staging"]["stages_required"])
        raise HTTPException(422,
            f"This plan needs {total} stages, past the {req.max_stages} this "
            f"export will build. Tooth {binding['fdi'] or binding['tid']} is "
            f"binding it at {binding['staging']['stages_required']} "
            f"({binding['staging']['driver']}-driven).")
    # A tooth that was cut but never moved is not a MOVING tooth: it has no
    # stage matrix but the identity, and planning a blend for it would only
    # release its neighbours' enamel for nothing.
    movers = [t for t in teeth if t["staging"]["stages_required"] > 0]
    t0_export = total <= 0

    if float(req.print_compensation_mm or 0.0) != 0.0:
        # Refused rather than ignored. The collar path applies it to a FUSED
        # solid after a union; here it would move enamel the crown-fidelity
        # and rigidity gates require to be where the plan put it.
        raise HTTPException(422, _jsonable({
            "error": "print_compensation_mm is not available for "
                     "construction='deformation'.",
            "measured_value": float(req.print_compensation_mm),
            "detail": ("The allowance is applied to the fused solid AFTER the "
                       "union. This construction performs no union, and "
                       "offsetting the model would move the enamel that "
                       "`moving_teeth_exact_rigid` and crown fidelity require "
                       "to be where the plan put it."),
            "construction": "deformation"}))

    if only_stage == "final":
        wanted = int(req.stage or total)
        if wanted < 0 or wanted > total or (wanted == 0 and not t0_export):
            raise HTTPException(422, _jsonable({
                "error": f"Stage {wanted} does not exist in this plan.",
                "available_stages": ([0] if t0_export
                                     else list(range(1, total + 1)))}))
        wanted_stages = [wanted]
    else:
        wanted_stages = [0] if t0_export else list(range(1, total + 1))

    # --- 2.3: the vertex sets, on the ORIGINAL scan ids -------------------
    labels = STORE.get(sid, "labels")
    labels = (np.asarray(labels).astype(np.int64).reshape(-1)
              if labels is not None and len(labels) == len(v) else None)

    moving_faces, static_faces = {}, {}
    moving_source = {}
    for t in movers:
        fm = t["rec"].get("face_mask")
        if fm is not None and len(np.asarray(fm)) == len(f):
            moving_faces[t["tid"]] = np.asarray(fm, bool)
            moving_source[t["tid"]] = "cut face_mask"
        elif labels is not None and t["fdi"]:
            moving_faces[t["tid"]] = mfg2.tooth_faces_from_labels(
                f, labels, int(t["fdi"]))
            moving_source[t["tid"]] = f"segmentation label {t['fdi']}"
        else:
            raise HTTPException(422, _jsonable({
                "error": f"Tooth {t['fdi'] or t['tid']} has neither a cut "
                         f"face mask nor a usable segmentation label, so its "
                         f"vertices cannot be identified on the scan.",
                "construction": "deformation"}))

    moving_fdis = {int(t["fdi"]) for t in movers if t["fdi"]}
    if labels is not None:
        for fdi in sorted(set(int(x) for x in np.unique(labels)) - {0} - moving_fdis):
            m = mfg2.tooth_faces_from_labels(f, labels, fdi)
            if m.any():
                static_faces[f"fdi{fdi}"] = m
    # Cut-but-unmoved teeth are anchorage like any other; without a label
    # they would otherwise be missing from crown fidelity.
    for t in teeth:
        if t in movers:
            continue
        fm = t["rec"].get("face_mask")
        name = f"fdi{t['fdi']}" if t["fdi"] else f"tooth {t['tid']}"
        if name not in static_faces and fm is not None \
                and len(np.asarray(fm)) == len(f):
            static_faces[name] = np.asarray(fm, bool)

    sets = mfg2.tooth_vertex_sets(f, moving_faces, static_faces,
                                  n_vertices=int(len(v)))

    # --- 2.2: the case plan, solved once ----------------------------------
    policy = mfg2.DeformationPolicy(
        trim_margin_mm=float(req.trim_margin_mm),
        base_thickness_mm=float(req.base_thickness_mm))
    moving_spec = {t["tid"]: {"vertices": sets["moving"][t["tid"]],
                              "frame": t["rec"]["frame"],
                              "c_res": t["rec"]["c_res"],
                              "clinical": t["clinical"]}
                   for t in movers}
    try:
        plan = mfg2.build_case_plan(v, f, af, moving_spec, sets["static"],
                                    policy=policy,
                                    static_by_tooth=sets["static_by_tooth"])
    except ValueError as e:
        raise HTTPException(422, _jsonable({
            "error": "deformation_plan_refused", "detail": str(e),
            "construction": "deformation",
            "vertex_sets": sets["diagnostics"]}))

    # --- step 5: who touches whom ------------------------------------------
    try:
        ipr_by_contact = mfg2.normalise_ipr_by_contact(
            getattr(req, "ipr_by_contact_mm", None) or {})
    except ValueError as e:
        raise HTTPException(422, _jsonable({
            "error": f"ipr_by_contact_mm: {e}", "construction": "deformation"}))
    contact_spec = (_v2_contact_spec(f, labels, movers, moving_faces,
                                     static_faces, sets)
                    if movers else None)

    # --- the stages -------------------------------------------------------
    prescribed_ipr = float(getattr(req, "prescribed_ipr_mm", 0.0) or 0.0)
    blobs, stage_meta = {}, []
    t_stages = time.perf_counter()
    for k in wanted_stages:
        if k == 0:
            out = mfg2.build_t0_stage(plan)
            matrices = {}
        else:
            matrices = mfg2.stage_matrices_for(plan, k, total)
            # An attachment on a cut-but-unmoved tooth stays where it was
            # bonded: the identity carries it.
            carry = {**{t["tid"]: np.eye(4) for t in teeth}, **matrices}
            shells = [(s["verts"], s["faces"]) for s in
                      _v2_attachment_solids(teeth, carry)]
            out = mfg2.build_stage_v2(plan, matrices,
                                      prescribed_ipr=prescribed_ipr,
                                      stage=k, total_stages=total,
                                      attachment_shells=shells,
                                      contact_spec=contact_spec,
                                      ipr_by_contact=ipr_by_contact)
        name = f"{arch}_Stage_{k:02d}.stl"
        gate = out["gate"]
        m = out["solid"]
        if out["blob"] is not None:
            blobs[name] = out["blob"]
        # EVERY NUMBER HERE DESCRIBES THE BYTES IN THE FILE - the solid, read
        # back and welded as a slicer would, never the planned surface it was
        # made from.
        file_gates = [g for g in ps.SOLID_GATES if g.startswith("file_")]
        file_failed = [g for g in file_gates if g in gate["failed_gates"]]
        stage_meta.append({
            "stage": k, "file": name if out["blob"] is not None else None,
            "stage_kind": gate["stage_kind"],
            "triangles": m.get(ps.K_TRIANGLES),
            "volume_mm3": m.get(ps.K_VOLUME),
            "components": m.get(ps.K_COMPONENTS),
            "manufacturing_gate": gate,
            "failures": mfg2.describe_failures(gate),
            "stl_validation": {
                "open_edges": m.get(ps.K_OPEN_EDGES),
                "nonmanifold_edges": m.get(ps.K_NONMANIFOLD_EDGES),
                "connected_components": m.get(ps.K_COMPONENTS),
                "winding_consistent": m.get(ps.K_WINDING),
                "volume_mm3": m.get(ps.K_VOLUME),
                "verdict": "PASS" if (m.get(ps.K_SOLIDIFIED)
                                      and not file_failed) else "FAIL",
                "failed_gates": file_failed,
                "measured_on": "the solidified model's re-read bytes",
            },
            "self_intersection": {
                "measured_by": "MeshLib findSelfCollidingTriangles",
                "self_colliding": m.get(ps.K_SELF_COLLIDING),
                "pairs": m.get(ps.K_SELF_COLLIDING_PAIRS)},
            "crown_deviation": out["manifest"]["crown_deviation"],
            # The IPR report: every moved tooth against each neighbour.
            "contacts": out.get("contacts"),
            "solidify": out["solidify"],
            "attachments": {"placed": sum(len(t["rec"].get("attachments") or [])
                                          for t in teeth) if k else 0,
                            "joined_by": "the same voxelisation as the cast",
                            "centroids_inside_solid":
                                m.get("attachment_centroids_inside")},
            "report": ({kk: vv for kk, vv in out["report"].items()
                        if kk != "moved_faces_mask"}
                       if out["report"] is not None else None),
            "manifest": out["manifest"],
            # The collar path carries these; a lab reading one manifest
            # against the other should not have to wonder whether a missing
            # key means "clear" or "not measured". s.14: NOT_CHECKED is not
            # CLEAR.
            "occlusal_interference": [],
            "occlusal_interference_checked": False,
        })

    t_stages = time.perf_counter() - t_stages

    manifest = {
        "generator": "Clinical Micro-Planner",
        "arch": arch,
        "construction": mfg2.CONSTRUCTION_ID,
        "print_model": mfg2.PRINT_MODEL_ID,
        "kind": ("staged manufacturing models - ONE inherited index buffer, "
                 "deformed per stage, then voxel-solidified; no socket, no "
                 "collar, no boolean"),
        "stages": total,
        "built_stages": wanted_stages,
        "t0_export": bool(t0_export),
        "units": "mm",
        "coordinate_space": "raw scanner coordinates - never re-centred or rescaled",
        "socket_treatment": ("none - the T0 cast is built with the teeth still "
                             "in its surface, so no socket is ever opened"),
        "print_compensation_mm": 0.0,
        "print_compensation_note": "true to anatomy; no allowance applied",
        "base_construction": ("trim_to_arch -> build_cast_base (never "
                              "cap_and_close, no clear_undercut_periphery)"),
        "solidification": {"voxel_size_mm": ps.VOXEL_SIZE_MM,
                           "sign_detection_mode": ps.SIGN_DETECTION_MODE,
                           "decimate_max_error_mm": ps.DECIMATE_MAX_ERROR_MM,
                           "meshlib_version": ps.meshlib_version(),
                           "licence": ("MeshLib - free for non-commercial / "
                                       "educational use only")},
        "policy": policy.as_dict(),
        "vertex_sets": sets["diagnostics"],
        "vertex_set_sources": moving_source,
        "case_plan": plan.diagnostics,
        "t0_topology": plan.t0_topology,
        "teeth": [{"tooth_id": t["tid"], "fdi": t["fdi"],
                   "prescription": t["clinical"],
                   "stages_required": t["staging"]["stages_required"],
                   "driver": t["staging"]["driver"],
                   "moving": t in movers,
                   "binds_the_case": bool(total) and
                                     t["staging"]["stages_required"] == total,
                   "root_length_mm": t["rec"].get("root_length_mm"),
                   "moving_vertices": int(len(sets["moving"].get(t["tid"], [])))}
                  for t in teeth],
        "stage_files": stage_meta,
        "union_seconds": 0.0,
        "phase_seconds": {
            "case_plan": plan.diagnostics["seconds"]["total"],
            "stages": round(t_stages, 3),
            "total": round(time.perf_counter() - t0_bundle, 3)},
        # The collar path's occlusion block, in its NOT_CHECKED state. The
        # antagonist sweep is a property of the plan, not of the construction,
        # and running it here would duplicate rather than share it.
        "occlusion": {
            "checked": False,
            "threshold_mm": 0.1,
            "stages_with_interference": [],
            "worst_penetration_mm": 0.0,
            "note": ("the antagonist check is not wired into this "
                     "construction; `checked: false` is NOT a finding of "
                     "clearance."),
        },
        "disclaimer": export_clinical_report.DISCLAIMER,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for nm, blob in blobs.items():
            z.writestr(nm, blob)
        z.writestr("manifest.json", json.dumps(_jsonable(manifest), indent=2))
    buf.seek(0)

    return {"buf": buf, "filename": f"{arch}_stages_deformation.zip",
            "manifest": manifest, "stages": total,
            "out_dir": None, "union_seconds": 0.0,
            "blobs": blobs, "base_mesh": (plan.V0, plan.F)}


@app.post("/api/session/{sid}/export/stages")
def export_stages(sid: str, req: StageExportRequest):
    """Stream the staged manufacturing models as a ZIP."""
    try:
        bundle = (build_stage_bundle_v2(sid, req)
                  if req.construction == "deformation"
                  else build_stage_bundle(sid, req))
        return StreamingResponse(
            bundle["buf"], media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{bundle["filename"]}"',
                     "X-Stage-Count": str(bundle["stages"]),
                     # So the client can say it in the status line without
                     # unzipping the manifest it just handed to the user.
                     "X-Occlusal-Interference": (
                         ",".join(str(s) for s in
                                  bundle["manifest"]["occlusion"]["stages_with_interference"])
                         or "none"),
                     "Access-Control-Expose-Headers":
                         "X-Stage-Count, X-Occlusal-Interference"})
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Stage export failed: {str(e)}")


@app.delete("/api/session/{sid}")
def close_session(sid: str):
    STORE.drop(sid)
    return {}


class FinalExportRequest(StageExportRequest):
    """One validated stage, for printing. Inherits the staging settings."""
    stage: int = 0          # 0 means "the last stage", the planned setup
    # WHAT COMES BACK. "zip" is the default and is unchanged: a binary STL
    # cannot carry a verdict, and a lab handed a bare STL has no record of
    # what was checked. The other two exist because a slicer wants a file it
    # can open and a reviewer wants the numbers without a 3D model attached -
    # and both still go through the SAME gate, so neither is a way around it.
    #   zip       the STL and manifest.json together  (default)
    #   stl       the raw binary STL, for a slicer or CAD package
    #   manifest  the manifest as JSON, no geometry
    fmt: str = "zip"


@app.post("/api/session/{sid}/export/final")
def export_final(sid: str, req: FinalExportRequest):
    """ONE fused manufacturing STL for one stage, or a refusal. Never both.

    WHY THIS IS A SEPARATE ENDPOINT FROM /export AND /export/stages, and it is
    not tidiness:

      * `/export` ships the setup for INSPECTION - a base plus one STL per
        crown, loose parts. It was labelled "Export Printable Cast" in the UI,
        which is the one thing it is not. A lab handed those files has to
        assemble them and has no way to know whether the assembly is sound.
      * `/export/stages` ships every stage so the clinician can review the
        whole plan. Each stage carries its own verdict; a stage that fails
        validation still ships, marked NOT PRINT READY, because refusing the
        review of a plan is not the same as refusing to print it.
      * THIS endpoint ships ONE file and applies the hard gate. If the written
        bytes do not pass every gate, there is no file - a 422 naming the gate
        and its measured value.

    So it is structurally impossible for the UI to download something from
    here and call it print ready when it is not: the only 200 response is a
    validated one.

    Returns a ZIP of the STL plus manifest.json. A binary STL cannot carry the
    verdict or the diagnostics, and a lab that receives a bare STL has no
    record of what was checked.
    """
    try:
        # Build every stage WITHOUT the global gate, then hard-gate only the
        # one being printed. Gating the whole bundle here would refuse a
        # perfectly good stage 1 because stage 4 is not print ready, which
        # tells the clinician nothing useful about the file they asked for.
        # The deformation path builds ONLY the stage asked for: each stage is
        # rebuilt from clinical x k/N and needs no other, and its solid costs
        # about a minute.
        bundle = (build_stage_bundle_v2(sid, req, only_stage="final")
                  if req.construction == "deformation"
                  else build_stage_bundle(sid, req, require_print_ready=False))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Final export failed: {str(e)}")

    manifest = bundle["manifest"]
    stages = manifest["stage_files"]
    if not stages:
        raise HTTPException(422, "No stages were produced; there is nothing to print.")

    wanted = req.stage or stages[-1]["stage"]
    chosen = next((s for s in stages if s["stage"] == wanted), None)
    if chosen is None:
        raise HTTPException(422, _jsonable({
            "error": f"Stage {wanted} does not exist in this plan.",
            "available_stages": [s["stage"] for s in stages]}))

    # THE AGGREGATE GATE DECIDES, and nothing else may. `build_stage_bundle`
    # enforces the boolean/topology gate; this is where the rest of the list -
    # transition quality, interface continuity, old-site quality, two-sided
    # cast fidelity, ROI compliance, synthetic exposure, crown rigidity,
    # gingival bridge, clinical consistency - has to hold before a file is
    # handed over as ready to manufacture.
    gate = chosen.get("manufacturing_gate") or {}
    if not gate.get("print_ready"):
        # NAMED, with the measured value, in one readable line per failure.
        failures = chosen.get("failures") or []
        raise HTTPException(422, _jsonable({
            "error": (f"Stage {wanted} is NOT PRINT READY"
                      + (": " + "; ".join(failures) if failures else ".")),
            "failures": failures,
            "failed_gates": gate.get("failed_gates"),
            "gates": gate.get("gates"),
            "boolean_topology_gate": chosen["stl_validation"]["failed_gates"],
            "detail": "The written file may still pass the narrower "
                      "boolean/topology regression; the aggregate "
                      "manufacturing gate is what this endpoint enforces."}))

    blob = bundle["blobs"][chosen["file"]]
    name = f"{manifest['arch']}_Stage_{wanted:02d}_FINAL.stl"

    final_manifest = {
        "generator": "Clinical Micro-Planner",
        "kind": "ONE fused manufacturing model for a single stage",
        "arch": manifest["arch"],
        "stage": wanted,
        "of_stages": manifest.get("stages"),
        "file": name,
        # EARNED, not asserted. This is the aggregate gate's own answer, and
        # the endpoint above refuses the export when it is False.
        "print_ready": bool(gate.get("print_ready")),
        "verdict": gate.get("verdict"),
        "verdict_scope": gate.get("gate_scope"),
        "boolean_topology_verdict": chosen["stl_validation"].get("verdict"),
        # The exact wording matters. This is an engineering statement about
        # geometry, not a clinical one about a patient.
        "verdict_meaning": (
            "Every gate in `manufacturing_gate` passed, measured on the "
            "actual written STL after a downstream reader's weld and on the "
            "reconstruction that produced it. This is NOT a claim of clinical "
            "validation, and NO REAL DE-IDENTIFIED SCAN has been run through "
            "this gate - every number behind it is synthetic."),
        "manufacturing_gate": gate,
        # THE SAME GATES, RESTATED AS THE CLAIMS A PERSON ACTUALLY ASKS
        # ABOUT. Each row names its evidence, so a claim cannot be made by a
        # row that measured nothing; a claim with missing evidence reads
        # false, never true by default.
        "evidence_report": (_v2_evidence_report(gate)
                            if req.construction == "deformation"
                            else mfg.manufacturing_evidence_report(chosen)),
        "validation": chosen["stl_validation"],
        "interfaces": chosen.get("interfaces"),
        "volume_mm3": chosen.get("volume_mm3"),
        "triangles": chosen.get("triangles"),
        "serialisation_chain": {
            "pre_export_weld": chosen.get("edges_pre_export_weld"),
            "post_export_weld": chosen.get("edges_post_export_weld"),
            "export_weld_merged_vertices": chosen.get("export_weld_merged_vertices"),
            "post_read_and_reader_weld": chosen.get("edges_post_read_and_reader_weld"),
            "reader_weld_merged_vertices": chosen.get("reader_weld_merged_vertices"),
        },
        "coordinate_space": "raw scanner coordinates — never re-centred or rescaled",
        "construction": ("deformation" if req.construction == "deformation"
                         else "collar"),
        # What made the FILE, and what it measured. The deformation path's
        # file is the voxel-solidified model, so its triangle count, timings,
        # crown deviation and MeshLib version belong beside the verdict.
        "print_model": ({
            "method": mfg2.PRINT_MODEL_ID,
            "stage_kind": chosen.get("stage_kind"),
            "solidify": chosen.get("solidify"),
            "crown_deviation": chosen.get("crown_deviation"),
            "contacts": chosen.get("contacts"),
            "self_intersection": chosen.get("self_intersection"),
            "attachments": chosen.get("attachments"),
            "meshlib_version": ps.meshlib_version(),
            "seconds": (chosen.get("manifest") or {}).get("seconds"),
        } if req.construction == "deformation" else None),
        "manufacturing_architecture": (
            ("ONE T0 cast built with the teeth in its surface; per stage the "
             "SAME index buffer, the moving crown carried by the rigid stage "
             "matrix and the gingiva blended harmonically inside a geodesic "
             "envelope; that planned surface is then VOXEL-SOLIDIFIED "
             "(MeshLib, 0.05 mm) into the one body this file carries. No "
             "socket, no cap, no collar, no boolean.")
            if req.construction == "deformation" else
            "immutable cast -> local target-position interface -> subtract "
            "cavity -> add emergence ramp -> union rigid crown + bounded seat. "
            "No root-length plug participates."),
        "print_compensation_mm": manifest.get("print_compensation_mm", 0.0),
        "occlusion": manifest.get("occlusion"),
        "disclaimer": export_clinical_report.DISCLAIMER,
    }

    fmt = (req.fmt or "zip").lower()
    if fmt not in ("zip", "stl", "manifest"):
        raise HTTPException(422, _jsonable({
            "error": f"Unknown export format {req.fmt!r}.",
            "available": ["zip", "stl", "manifest"]}))

    # EVERY FORMAT IS THE SAME GATED BYTES. The gate above has already run and
    # already refused; choosing a format cannot route around it, and the
    # X-Print-Ready header is set identically on all three.
    if fmt == "stl":
        buf = io.BytesIO(blob)
        media = "model/stl"
        download = name
    elif fmt == "manifest":
        buf = io.BytesIO(
            json.dumps(_jsonable(final_manifest), indent=2).encode("utf-8"))
        media = "application/json"
        download = f"{manifest['arch']}_Stage_{wanted:02d}_FINAL_manifest.json"
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(name, blob)
            z.writestr("manifest.json",
                       json.dumps(_jsonable(final_manifest), indent=2))
        media = "application/zip"
        download = f"{manifest['arch']}_Stage_{wanted:02d}_FINAL.zip"
    buf.seek(0)

    _record(sid, audit.EXPORTED, arch=manifest["arch"],
            detail=f"final print STL, stage {wanted}, format {fmt}",
            values={"stage": int(wanted), "triangles": int(chosen["triangles"]),
                    "print_ready": bool(gate.get("print_ready"))})

    return StreamingResponse(
        buf, media_type=media,
        headers={"Content-Disposition":
                     f'attachment; filename="{download}"',
                 "X-Print-Ready": ("true" if gate.get("print_ready")
                                   else "false"),
                 "X-Stage": str(wanted),
                 "X-Open-Edges": str(chosen["stl_validation"]["open_edges"]),
                 "X-Nonmanifold-Edges": str(chosen["stl_validation"]["nonmanifold_edges"]),
                 "X-Components": str(chosen["stl_validation"]["connected_components"]),
                 "X-Claims-Verified": (
                     "true" if final_manifest["evidence_report"]
                     ["all_claims_verified"] else "false"),
                 "X-Export-Format": fmt,
                 "Access-Control-Expose-Headers":
                     "X-Print-Ready, X-Stage, X-Open-Edges, "
                     "X-Nonmanifold-Edges, X-Components, X-Claims-Verified, "
                     "X-Export-Format"})
