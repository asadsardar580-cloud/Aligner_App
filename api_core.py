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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_FPS = os.path.join(CURRENT_DIR, "ToothGroupNetwork", "ckpts", "0707_cosannealing_val.h5")
CKPT_BDL = CKPT_FPS

app = FastAPI(title="Virtual Diagnostic Setup API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
        "error": model.get("error"),
        "ready_message": ("Loading the AI segmentation model..." if warming
                          else "AI segmentation ready." if loaded
                          else model.get("error") or "AI segmentation unavailable."),
        "segmentation_in_progress": _SEGMENTATION_STATE["in_progress"],
        "segmentation_started_at": _SEGMENTATION_STATE["started_at"],
        "segmentation_message": _SEGMENTATION_STATE["message"],
    }


def run_segmentation(v, f, jaw):
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
        return result_json
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        if os.path.exists(temp_json): os.remove(temp_json)

class OcclusalPlaneRequest(BaseModel):
    points: list[list[float]]

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
    raw = await file.read()               
    verts, faces = stl_io.parse_stl_bytes(raw)
    verts, faces, report = cg.condition_mesh(verts, faces)
    STORE.put(sid, "verts", verts)
    STORE.put(sid, "faces", faces)

    # Record the scan's own topology BEFORE any cut. condition_mesh welds,
    # drops degenerates and debris and fills small holes, but it does not
    # repair non-manifold edges — so an arch can arrive already unprintable.
    # Knowing that here is what lets /export blame the scan instead of the cut.
    scan_health = cg.manifold_report(faces)
    STORE.put(sid, "scan_health", scan_health)
    
    edges = cg.directed_edges(faces)
    conc = cg.boundary_field(verts, faces, edges=edges)
    graph = cg.build_barrier_graph(verts, faces, conc, edges=edges)
    STORE.put(sid, "edges", edges)
    STORE.put(sid, "concavity", conc)
    STORE.put(sid, "graph", graph)
    
    return {"session_id": sid, "vertex_count": int(len(verts)), "face_count": int(len(faces)),
            "bbox": [verts.min(0).tolist(), verts.max(0).tolist()],
            "conditioning": report, "scan_health": scan_health}

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
async def segment(sid: str):
    try:
        v, f, arch = (STORE.require(sid, "verts"), STORE.require(sid, "faces"), STORE.arch(sid))
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    if _SEGMENTATION_STATE["in_progress"]:
        raise HTTPException(409, "AI segmentation is already running. Please wait for it to finish.")

    _SEGMENTATION_STATE.update({
        "in_progress": True,
        "started_at": time.time(),
        "message": ("Waiting for the AI model to finish loading..."
                    if _SEGMENTATION_STATE["warming"]
                    else "Running ToothGroupNetwork on this arch..."),
    })

    try:
        # Idempotent. If the background warm-up is still running this blocks on
        # the same lock and returns the same pipeline; if the warm-up failed it
        # returns that error rather than a second attempt's.
        model_status = await asyncio.to_thread(
            tgn_bridge.load, CKPT_FPS, CKPT_BDL, "tgnet")
        if not model_status["loaded"]:
            raise RuntimeError(model_status.get("error") or "AI segmentation model did not initialize.")
        _SEGMENTATION_STATE["message"] = "Running ToothGroupNetwork on this arch..."

        result_json = await asyncio.to_thread(run_segmentation, v, f, arch)
        check = jaw_naming.verify_fdi_matches_jaw(result_json, arch)
        if not check["ok"]:
            raise HTTPException(500, check["diagnosis"])

        labels, _ = jaw_naming.extract_labels(result_json, expect_jaw=arch)
        if len(labels) != len(v):
            raise HTTPException(500, "Label array does not match mesh.")

        STORE.put(sid, "labels", labels)
        return {"labels": [int(x) for x in labels], "jaw": arch, "report": check}
    finally:
        _SEGMENTATION_STATE.update({
            "in_progress": False,
            "started_at": None,
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
    return {"vertex_ids": ids.tolist(), "tolerance": tol, "seed": seed.tolist(), "seed_moved_mm": round(float(np.linalg.norm(seed - clicked)), 3), "max_distance": float(finite.max()) if finite.size else 0.0}

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

        sel = np.zeros(len(v), bool)
        sel[np.asarray(req.vertex_ids, np.int64)] = True
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

        face_mask = cg.largest_face_component(f, face_mask)
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

        verdict = cut_guard.check_crown(cv, cf, rim_global_idx, conc)
        if not verdict["ok"]:
            raise HTTPException(422, verdict["diagnosis"])

        # The arch frame is what makes this biological rather than geometric:
        # the long axis is signed and reconciled against the occlusal plane, so
        # C_res lands in the bone along the tooth's own root instead of 10mm
        # sideways into empty space.
        frame = cg.derive_frame_from_region(
            np.asarray(req.mesial_pt, float), np.asarray(req.distal_pt, float),
            cv, rim_pts, arch_frame=af)
        c_res = cg.center_of_resistance(frame, root_length_mm=req.root_length_mm)

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
        if depth <= 0 or lateral > max_lateral:
            raise HTTPException(422,
                f"Derived pivot is not inside the alveolus: C_res sits {depth:.1f}mm "
                f"apical of the cervical margin and {lateral:.1f}mm off to the side "
                f"(limit {max_lateral:.1f}mm). Rotating about it would swing the tooth "
                f"through the arch rather than seating it in the socket. Re-check the "
                f"selection and the occlusal plane landmarks.")

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
        STORE.put(sid, f"tooth:{tid}", {"cv": cv, "cf": cf, "bv": bv, "bf": bf,
                                        "frame": frame, "c_res": c_res,
                                        "root_length_mm": float(req.root_length_mm),
                                        "reconcile": reconcile, "dimensions": dims,
                                        "face_mask": face_mask,
                                        "socket_rim": socket_rim_global,
                                        # The cup's own geometry, so /export can
                                        # replay it verbatim rather than rebuild
                                        # it and risk a different socket than
                                        # the one that was approved.
                                        "socket_cup_pts": cup_pts,
                                        "socket_cup_faces": cup_faces,
                                        "socket_info": cup_info})
        return {"tooth_id": tid, "watertight": True,
                # The server's own FDI, read through the crown's face mask
                # rather than the raw selection, so the client's per-tooth root
                # length and the lab manifest cannot disagree. None when
                # /segment never ran — never a guess.
                "fdi": _tooth_fdi(sid, STORE.get(sid, f"tooth:{tid}"), f),
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
                                        float(req.root_length_mm)),
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

    return {"matrix": M.ravel().tolist(), "staging": staging, "clearance": pen,
            "c_res": np.asarray(t["c_res"], float).tolist(),
            "axis_source": t["frame"].get("axis_source"),
            "axis_deviation_deg": t["frame"].get("axis_deviation_deg"),
            "u_bl_points_buccal": t["frame"].get("u_bl_points_buccal"),
            "occlusion": occlusal,
            "occlusal_warning": (ANTAGONIST_WARNING if occlusal and occlusal["collides"]
                                 else None)}

@app.get("/api/session/{sid}/teeth")
def list_teeth(sid: str):
    """Every extracted tooth and its committed pose.

    Lets a reloaded client restore the whole setup — which crowns exist, where
    each was left, and which arch faces are gone — instead of losing the case
    to a browser refresh.
    """
    try:
        STORE.require(sid, "verts")
    except SessionExpired as e:
        raise HTTPException(404, str(e))

    extracted = STORE.get(sid, "extracted_faces")
    out = []
    for key in STORE.keys(sid):
        if not key.startswith("tooth:"):
            continue
        t = STORE.get(sid, key)
        M = t.get("matrix")
        out.append({
            "tooth_id": key.split(":", 1)[1],
            "c_res": np.asarray(t["c_res"], float).tolist(),
            "clinical": t.get("clinical"),
            "matrix": None if M is None else np.asarray(M, float).ravel().tolist(),
            "frame": {k: (val.tolist() if hasattr(val, "tolist") else val)
                      for k, val in t["frame"].items()},
        })
    return {"teeth": out,
            "extracted_face_count": 0 if extracted is None else int(extracted.sum())}


def _root_cone(rim_xyz: np.ndarray, frame: dict, root_length_mm: float) -> dict:
    """The cervical rim swept to an apex root_length_mm apical along u_oa.

    World coordinates, so the client can parent it beside the crown and drive
    both with the same matrix. Returned as a ring plus an apex with a triangle
    fan, which is a surface rather than a solid — it is a visual aid, never
    something that reaches a printable file, and nothing in /export looks at it.

    The apex is placed from the RIM CENTROID, matching center_of_resistance's
    own construction (rim_centroid + u_oa*(h - root_length)), so the drawn root
    and the pivot the tooth actually rotates about are derived the same way
    rather than two independent guesses about where the apex is.
    """
    rim = np.asarray(rim_xyz, float)
    u_oa = np.asarray(frame["u_oa"], float)
    u_oa = u_oa / np.linalg.norm(u_oa)
    centroid = np.asarray(frame.get("rim_centroid", rim.mean(axis=0)), float)
    apex = centroid - u_oa * float(root_length_mm)

    n = len(rim)
    verts = np.vstack([rim, apex])
    faces = [[i, (i + 1) % n, n] for i in range(n)]
    return {"vertices": verts.tolist(), "faces": faces,
            "apex": apex.tolist(), "length_mm": float(root_length_mm),
            "label": "virtual root (estimated)"}


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
        bv, bf, base_info = cg.build_cast_base(tv, tf, af,
                                               base_thickness_mm=req.base_thickness_mm,
                                               rim=trim_info["rim_loop"])
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
        out_dir = req.out_dir or os.path.join(CURRENT_DIR, "exports", sid[:8])
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
    try:
        bundle = build_export_bundle(sid, req)
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
    import manifold3d as m3
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


def _solid_bodies(solid):
    """Drop inverted crumbs. Returns (solid, bodies, crumbs_discarded).

    A tangential boolean — which every one of these is, the crown going back
    into the hole it was cut from — leaves manifold3d emitting the odd tiny
    inside-out shell. Measured on one stage, decompose() returned
    [27572.4, -1.9] mm3: a 1.9mm3 fragment with NEGATIVE volume. That is a void,
    not geometry, and it made a perfectly fused model read as two solids.

    Applied at EVERY boolean, not just the last one, because the count
    compounds: crown+plug alone produced 2 raw bodies (1 crumb) on one tooth and
    10 raw (7 crumbs) on another, and those crumbs then went into the stage
    union and were counted again — 16 raw bodies at the end where 5 were real.

    `bodies` is the number of genuine positive-volume solids. More than one is
    a real fracture and the caller decides what to do about it.
    """
    import manifold3d as m3
    parts = solid.decompose()
    keep = [c for c in parts if c.volume() > 0]
    crumbs = len(parts) - len(keep)
    if not keep:
        return solid, 0, crumbs
    if len(keep) == 1:
        return keep[0], 1, crumbs
    return m3.Manifold.batch_boolean(keep, m3.OpType.Add), len(keep), crumbs


def _manufacturing_tooth(rec: dict, verts: np.ndarray):
    """The crown fused with its plug, once, at T0 — then transformed per stage.

    Built once and reused because manifold3d can transform a Manifold lazily,
    which turns N stages x M teeth of mesh rebuilding into N batch booleans.

    Returns (solid, bodies, crumbs). `bodies` > 1 means the crown and its plug
    did NOT fuse — which is the decisive signal that the crown is a shell rather
    than a tooth, and is what the export gate acts on. Measured: a 136mm3 crown
    gives 1 body, a 63mm3 one gives 3.
    """
    crown = _to_manifold(rec["cv"], rec["cf"])
    rim = np.asarray(rec.get("socket_rim"), np.int64) if rec.get("socket_rim") is not None else None
    if rim is None or len(rim) < 3:
        return _solid_bodies(crown)
    depth = float(rec.get("root_length_mm") or cg.SOCKET_DEPTH_MM)
    try:
        pv, pf = _rim_plug(verts[rim], np.asarray(rec["frame"]["u_oa"], float), depth)
    except ValueError:
        return _solid_bodies(crown)   # a rim too degenerate to plug
    plug, _pb, _pc = _solid_bodies(_to_manifold(pv, pf))
    return _solid_bodies(crown + plug)


SHELL_REFUSAL = ("Cannot export raw shell geometry. Crown must be fully extracted and "
                 "sealed via the Wand/Brush tool to generate 3D printable stages.")


def _screen_crowns_for_manufacturing(teeth: list, verts: np.ndarray):
    """Refuse the export unless every crown is a solid a boolean can fuse.

    Mutates each entry with `solid`, `bodies`, `crumbs` and `screen`, so the
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
            t["solid"], t["bodies"], t["crumbs"] = None, 0, 0
            continue

        solid, bodies, crumbs = _manufacturing_tooth(rec, verts)
        t["solid"], t["bodies"], t["crumbs"] = solid, bodies, crumbs
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
    """The six channels at stage k of n — the Python mirror of the client's
    clinicalAtStage. Absolute from T0, never an interpolation of the 4x4: the
    3x3 block of (1-t)I + tR is not orthonormal for any t in between, so a
    matrix lerp shears the crown at every intermediate stage."""
    f = (k / n) if n > 0 else 0.0
    return {key: float(clinical.get(key, 0.0) or 0.0) * f
            for key in ("tip_deg", "torque_deg", "rotation_deg", "d_md", "d_bl", "d_oa")}


def build_stage_bundle(sid: str, req: StageExportRequest) -> dict:
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

    # --- the base, sockets filled FLUSH -----------------------------------
    sealed_v, sealed_f, sockets = _seal_sockets(v, f, sid, extracted, flush=True)
    try:
        curve, _ = cg.fit_arch_curve(v, af)
        tv, tf, trim_info = cg.trim_to_arch(sealed_v, sealed_f, af,
                                            margin_mm=req.trim_margin_mm, curve=curve)
        bv, bf, base_info = cg.build_cast_base(tv, tf, af,
                                               base_thickness_mm=req.base_thickness_mm,
                                               rim=trim_info["rim_loop"])
    except ValueError as e:
        raise HTTPException(422, f"The cast base could not be built. {e}")

    import manifold3d as m3
    base_solid = _to_manifold(bv, bf)
    # Screens every crown AND builds its manufacturing solid, once. manifold3d
    # transforms lazily, so a stage then costs one batch boolean rather than
    # rebuilding every mesh.
    _screen_crowns_for_manufacturing(teeth, v)

    # The antagonist, resolved once. None when the opposing arch is not loaded,
    # which is the ordinary single-arch case and skips the check entirely.
    ant = _antagonist(req.opposing_session_id)
    t_collide = 0.0

    # --- one fused solid per stage ----------------------------------------
    blobs, stage_meta = {}, []
    t_union = 0.0
    for k in range(1, total + 1):
        parts = [base_solid]
        interference = []
        for t in teeth:
            rec, clinical = t["rec"], _stage_clinical(t["clinical"], k, total)
            M = cg.kinematic_matrix(rec["frame"], rec["c_res"], **clinical)
            parts.append(t["solid"].transform(np.asarray(M[:3, :4], float)))

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
        solid = m3.Manifold.batch_boolean(parts, m3.OpType.Add)
        # Same crumb purge as every other boolean here — see _solid_bodies.
        solid, n_bodies, crumbs = _solid_bodies(solid)
        mesh = solid.to_mesh()
        t_union += time.perf_counter() - t0
        sv = np.asarray(mesh.vert_properties, float)[:, :3]
        sf = np.asarray(mesh.tri_verts, np.int64)
        if len(sf) == 0:
            raise HTTPException(422,
                f"Stage {k} fused to an empty solid. The crowns and the base did not "
                f"intersect, which usually means the prescription has moved a tooth clear "
                f"of the cast.")

        name = f"{arch}_Stage_{k:02d}.stl"
        blob = cg.write_binary_stl_bytes(sv, sf)

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
        rv, rf = stl_io.parse_stl_bytes(blob)
        welded = cg.manifold_report(rf)
        # manifold3d's OWN component count, not _face_components. The latter
        # walks edge adjacency on the index buffer, and the same duplicate
        # vertices described above split a geometrically joined solid into two
        # index-disconnected groups — measured, it called a tooth "floating"
        # while a boolean intersection with the base showed 141 mm3 of overlap.
        n_comp = n_bodies
        if n_comp != 1:
            raise HTTPException(422,
                f"Stage {k} fused into {n_comp} separate solids. A tooth has moved clear "
                f"of the cast and is floating — the model cannot be thermoformed.")
        closed = cg.manifold_report(sf)
        if not closed["watertight"]:
            raise HTTPException(422,
                f"Stage {k} is not closed: {closed['open_edges']} open and "
                f"{closed['nonmanifold_edges']} non-manifold edges of "
                f"{closed['total_edges']}. It is not safe to print.")

        blobs[name] = blob
        stage_meta.append({
            "stage": k, "file": name, "triangles": int(len(sf)),
            "volume_mm3": round(float(solid.volume()), 3),
            "components": int(n_comp),
            "inverted_crumbs_discarded": int(crumbs),
            "occlusal_interference": interference,
            "occlusal_warning": ANTAGONIST_WARNING if interference else None,
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

    manifest = {
        "generator": "Clinical Micro-Planner",
        "arch": arch,
        "kind": "staged manufacturing models — fused solids, ready to thermoform",
        "stages": total,
        "units": "mm",
        "coordinate_space": "raw scanner coordinates — never re-centred or rescaled",
        "socket_treatment": "filled flush at the gingival margin; the tooth sits proud",
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
        "union_seconds": round(t_union, 2),
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
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, blob in blobs.items():
            z.writestr(name, blob)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    buf.seek(0)

    out_dir = None
    if req.keep_local:
        out_dir = req.out_dir or os.path.join(CURRENT_DIR, "exports", sid[:8], "stages")
        os.makedirs(out_dir, exist_ok=True)
        for name, blob in blobs.items():
            with open(os.path.join(out_dir, name), "wb") as fh:
                fh.write(blob)
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

    return {"buf": buf, "filename": f"{arch}_stages.zip", "manifest": manifest,
            "stages": total, "out_dir": out_dir, "union_seconds": t_union}


@app.post("/api/session/{sid}/export/stages")
def export_stages(sid: str, req: StageExportRequest):
    """Stream the staged manufacturing models as a ZIP."""
    try:
        bundle = build_stage_bundle(sid, req)
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