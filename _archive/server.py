#!/usr/bin/env python3
"""
server.py — FastAPI routing over api_core.

Thin by design: every route is a few lines of validation plus a call into
api_core, which is fully covered by test_api_core.py. If a request produces a
wrong result, the logic is testable without a server running.

    pip install fastapi uvicorn python-multipart
    uvicorn server:app --reload
    open http://127.0.0.1:8000/docs

NOT PRODUCTION READY. No authentication, no TLS, no audit log, and sessions
live in process memory and vanish on restart. Intraoral scans are patient
health information: once they cross a network this becomes a system subject
to HIPAA/GDPR, requiring encryption in transit and at rest, retention limits,
audit logging, and a business-associate agreement with whoever hosts it.
Settle that before pointing this at real patient data.
"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import api_core as api

app = FastAPI(title="Clinical Micro-Planner API", version="1.0.0")

# Wide open for local React development. Restrict to your actual origin
# before this is reachable from anywhere but localhost.
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


class PreviewRequest(BaseModel):
    seed: list[float] = Field(..., description="[x,y,z] click on the crown, scan coordinates")
    tolerance: float = Field(10.0, gt=0, le=100, description="magic-wand spread, mm")


class SegmentRequest(BaseModel):
    mesial: list[float] = Field(..., description="[x,y,z] mesial contact")
    distal: list[float] = Field(..., description="[x,y,z] distal contact")
    root_length_mm: float = Field(10.0, ge=0, le=25)
    # auto path
    seed: list[float] | None = None
    tolerance: float | None = Field(None, gt=0, le=100)
    # explicit path: exactly what the clinician painted
    face_indices: list[int] | None = None


class AutoColorRequest(BaseModel):
    tolerance: float = Field(12.0, gt=0, le=100)
    min_separation_mm: float = Field(5.5, gt=0, le=20,
        description="Minimum distance between cusp seeds. Below a crown width teeth split; above it they merge.")
    max_seeds: int = Field(16, ge=1, le=32)


class Frame(BaseModel):
    centroid: list[float]
    u_md: list[float]
    u_bl: list[float]
    u_oa: list[float]


class KinematicsRequest(BaseModel):
    frame: Frame
    root_length_mm: float = Field(10.0, ge=0, le=25)
    tip_deg: float = Field(0.0, ge=-20, le=20)
    torque_deg: float = Field(0.0, ge=-20, le=20)
    rotation_deg: float = Field(0.0, ge=-20, le=20)
    d_md: float = Field(0.0, ge=-3, le=3)
    d_bl: float = Field(0.0, ge=-3, le=3)
    d_oa: float = Field(0.0, ge=-3, le=3)


@app.post("/session", summary="Upload the arch scan ONCE")
async def create_session(arch: str = "maxillary", file: UploadFile = File(...)):
    """Returns JSON with session_id. Every later call sends coordinates only,
    which is what keeps this architecture viable over a clinic connection."""
    try:
        return api.create_session(await file.read(), arch=arch)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/session/{sid}/preview", summary="Magic-wand selection preview")
def preview(sid: str, req: PreviewRequest):
    """Returns selected FACE INDICES as binary npz, with stats in headers.

    Indices rather than geometry: the client already holds the arch it
    uploaded, so it highlights faces locally. ~32KB for a 200k-face arch.
    """
    try:
        stats, blob = api.preview_selection(sid, req.seed, req.tolerance)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content=blob, media_type="application/octet-stream",
                    headers={"X-Selected-Faces": str(stats["n_selected"]),
                             "X-Selected-Fraction": str(stats["fraction"]),
                             "X-Plausible": str(stats["plausible"]),
                             "Access-Control-Expose-Headers":
                                 "X-Selected-Faces,X-Selected-Fraction,X-Plausible"})


@app.post("/session/{sid}/segment", summary="Commit the cut")
def segment(sid: str, req: SegmentRequest):
    """Returns JSON metadata including the anatomical frame and C_res.
    Fetch the meshes separately via /session/{sid}/meshes."""
    try:
        meta, blob = api.segment(sid, req.seed, req.tolerance, req.mesial,
                                 req.distal, req.root_length_mm, req.face_indices)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    api.SESSIONS[sid]["last_meshes"] = blob
    return meta


@app.get("/session/{sid}/meshes", summary="Download crown + base as binary npz")
def meshes(sid: str):
    try:
        s = api.get_session(sid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    blob = s.get("last_meshes")
    if blob is None:
        raise HTTPException(404, "No segmentation on this session yet.")
    return Response(content=blob, media_type="application/octet-stream")


@app.post("/session/{sid}/auto-color", summary="Colour whole arch: tooth vs gingiva")
def auto_color(sid: str, req: AutoColorRequest):
    """Returns int16 face labels as a raw buffer; 0 = gingiva, 1..n = teeth.
    Metadata travels in headers so the body stays a plain array."""
    try:
        meta, blob = api.auto_color(sid, req.tolerance, req.min_separation_mm, req.max_seeds)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    import json as _json
    return Response(content=blob, media_type="application/octet-stream",
                    headers={"X-Meta": _json.dumps(meta),
                             "Access-Control-Expose-Headers": "X-Meta"})


class ToothAtRequest(BaseModel):
    face_index: int = Field(..., ge=0)


@app.post("/session/{sid}/tooth-at", summary="Which precomputed tooth owns this face?")
def tooth_at(sid: str, req: ToothAtRequest):
    """A click resolves to a whole tooth. No tolerance, no second request."""
    try:
        meta, blob = api.tooth_at_face(sid, req.face_index)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    import json as _json
    return Response(content=blob, media_type="application/octet-stream",
                    headers={"X-Meta": _json.dumps(meta),
                             "Access-Control-Expose-Headers": "X-Meta"})


@app.get("/session/{sid}/tooth-labels", summary="Per-face tooth labels")
def tooth_labels(sid: str):
    try:
        meta, blob = api.tooth_labels(sid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    import json as _json
    return Response(content=blob, media_type="application/octet-stream",
                    headers={"X-Meta": _json.dumps(meta),
                             "Access-Control-Expose-Headers": "X-Meta"})


@app.post("/kinematics", summary="4x4 transform for a movement")
def kinematics(req: KinematicsRequest):
    """
    Stateless and tiny -- ~670 bytes, no session needed.

    Returns the MATRIX, never a transformed mesh. A slider drag fires dozens
    of times a second; shipping a moved crown each time would push megabytes
    per drag. The client applies the matrix to geometry it already holds,
    which in Three.js is one assignment evaluated on the GPU.
    """
    try:
        return api.kinematics(req.frame.model_dump(), req.root_length_mm,
                              req.tip_deg, req.torque_deg, req.rotation_deg,
                              req.d_md, req.d_bl, req.d_oa)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/health")
def health():
    api.expire_sessions()
    return dict(status="ok", active_sessions=len(api.SESSIONS))


@app.post("/ai/propose-selection", summary="NOT IMPLEMENTED")
def propose_selection():
    """
    Returns 501 deliberately rather than a mock.

    A working version needs a trained mesh-segmentation network (MeshSegNet,
    ToothGroupNetwork, DilatedToothSegNet), its weights, and GPU inference.
    An endpoint that quietly returned a hardcoded answer would demo
    convincingly and become dangerous the moment anyone downstream forgot it
    was fake. When you have weights, this is where they go: it should return
    a proposed seed and tolerance for the clinician to confirm, and
    tooth_segmentation/label_adapter.py already handles everything after.
    """
    raise HTTPException(501, "AI selection not implemented. Use /session/{sid}/preview.")
