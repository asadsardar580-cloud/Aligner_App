"""Clicking a tooth selects that tooth.

THE WORKFLOW THIS PROTECTS. STL -> occlusal plane -> automatic segmentation ->
click a tooth -> the whole tooth is selected -> cut. The brush and the wand
stay available for CORRECTING a region the model got wrong; they are not the
route for an ordinary selection, and until `/select-tooth` existed they were,
because `onPointerDown` went straight to `/wand`.

WHY THAT MATTERED, measured on this project's real scan over all sixteen
teeth, clicking each tooth's own surface:

    route                 vertices that belong to the clicked tooth
    segmentation label                     100.0%
    geodesic wand                           15.7%   (25.3% of the arch/click)

The wand's auto tolerance returned 25.00 mm for every single tooth, because
`snap_seed_to_ridge` moves the seed up to 3 mm to the nearest high-concavity
point - which is the interdental sulcus BETWEEN two teeth, not a tooth.
"""
from __future__ import annotations

import asyncio
import io
import os
import struct

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

import api_core
import core_geometry as cg
import segmentation_diagnostics as sd
from api_core import STORE

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# A two-tooth arch with a known answer, built through the real upload path.
# ---------------------------------------------------------------------------

def _stl_bytes(verts, faces):
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    buf.write(struct.pack("<I", len(faces)))
    for tri in faces:
        buf.write(struct.pack("<3f", 0, 0, 0))
        for vi in tri:
            buf.write(struct.pack("<3f", *verts[vi]))
        buf.write(struct.pack("<H", 0))
    return buf.getvalue()


def _two_block_arch():
    """Two separated blocks plus a stray triangle on the far one.

    THE STRAY IS THE POINT OF THE FIXTURE. A real label carries a handful of
    triangles on its neighbour - measured on the real scan, eleven of twelve
    teeth have their largest piece at 98.7% or better, which means the other
    1.3% is somewhere else. A selection that returns both pieces cannot close
    watertight, and `/cut` would refuse it 422 with a message about the cut
    rather than about the label.
    """
    import trimesh
    a = trimesh.creation.box(extents=(6, 6, 6))
    b = trimesh.creation.box(extents=(6, 6, 6))
    b.apply_translation([20.0, 0.0, 0.0])
    m = trimesh.util.concatenate([a, b])
    v = np.asarray(m.vertices, float)
    f = np.asarray(m.faces, np.int64)
    return v, f


def _session_with_labels():
    """Upload through the real endpoint, then attach labels and diagnostics."""
    v, f = _two_block_arch()
    up = UploadFile(filename="scan.stl", file=io.BytesIO(_stl_bytes(v, f)))
    res = asyncio.run(api_core.create_session(arch="lower", file=up))
    sid = res["session_id"]
    verts = STORE.require(sid, "verts")
    faces = STORE.require(sid, "faces")

    labels = np.where(verts[:, 0] > 10.0, 41, 31).astype(np.int64)
    # One stray vertex of tooth 41 planted on tooth 31's block, so the
    # largest-component rule has something to discard.
    near = int(np.argmin(np.linalg.norm(verts - verts[labels == 31][0], axis=1)))
    labels[near] = 41

    STORE.put(sid, "labels", labels)
    STORE.put(sid, "segmentation_diagnostics",
              sd.label_report(verts, faces, labels))
    return sid, verts, faces, labels


# ---------------------------------------------------------------------------
# The selection itself
# ---------------------------------------------------------------------------

def test_clicking_a_tooth_selects_that_whole_tooth_and_nothing_else():
    sid, v, f, labels = _session_with_labels()
    vid = int(np.where(labels == 31)[0][0])

    out = api_core.select_tooth(sid, api_core.SelectToothRequest(vertex_id=vid))
    ids = np.asarray(out["vertex_ids"], np.int64)

    assert out["fdi"] == 31
    assert out["route"] == "segmentation label"
    assert len(ids) > 0
    assert (labels[ids] == 31).all(), "the selection reached another tooth"
    # and it is the WHOLE tooth, not a patch of it
    assert set(ids.tolist()) == set(np.where(labels == 31)[0].tolist())
    print(f"PASS  clicked v{vid} -> FDI 31, {len(ids)} vertices, "
          f"{out['kept_faces']} faces, all of them tooth 31")


def test_a_stray_island_on_a_neighbour_is_discarded_not_selected():
    """The largest connected component, not every vertex carrying the label."""
    sid, v, f, labels = _session_with_labels()
    vid = int(np.where(v[:, 0] > 10.0)[0][0])          # on the far block

    out = api_core.select_tooth(sid, api_core.SelectToothRequest(vertex_id=vid))
    ids = np.asarray(out["vertex_ids"], np.int64)

    assert out["fdi"] == 41
    assert out["discarded_faces"] >= 0
    # Nothing selected may sit on the other block.
    assert (v[ids][:, 0] > 10.0).all(), \
        "the selection includes the stray island on the neighbouring tooth"
    assert len(ids) < int((labels == 41).sum()), \
        "the stray vertex was not discarded - the fixture did not reproduce it"
    print(f"PASS  FDI 41: {int((labels == 41).sum())} labelled vertices -> "
          f"{len(ids)} selected, {out['discarded_faces']} stray face(s) dropped")


def test_a_point_and_a_vertex_id_select_the_same_tooth():
    """The client may have either; they must not disagree."""
    sid, v, f, labels = _session_with_labels()
    vid = int(np.where(labels == 31)[0][3])

    a = api_core.select_tooth(sid, api_core.SelectToothRequest(vertex_id=vid))
    b = api_core.select_tooth(sid, api_core.SelectToothRequest(
        point=[float(x) for x in v[vid]]))
    assert a["fdi"] == b["fdi"] == 31
    assert a["vertex_ids"] == b["vertex_ids"]
    print(f"PASS  point and vertex_id agree on FDI {a['fdi']}")


def test_clicking_gingiva_says_so_and_selects_nothing():
    """NOT_CHECKED is not CLEAR, and a click on tissue is not a tooth.

    Returning the nearest tooth instead would be a guess that reads exactly
    like a selection, and the clinician would cut it.
    """
    sid, v, f, labels = _session_with_labels()
    gum = np.zeros(len(v), np.int64)
    STORE.put(sid, "labels", gum)

    out = api_core.select_tooth(sid, api_core.SelectToothRequest(vertex_id=0))
    assert out["fdi"] is None
    assert out["vertex_ids"] == []
    assert out["route"] == "gingiva"
    assert "gingiva" in out["message"]
    print(f"PASS  gingiva click: {out['message'][:60]}...")


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def test_an_unsegmented_arch_is_refused_409_and_told_what_to_do():
    """And it points at the wand, which needs no segmentation at all."""
    v, f = _two_block_arch()
    up = UploadFile(filename="scan.stl", file=io.BytesIO(_stl_bytes(v, f)))
    sid = asyncio.run(api_core.create_session(arch="lower",
                                              file=up))["session_id"]
    with pytest.raises(HTTPException) as e:
        api_core.select_tooth(sid, api_core.SelectToothRequest(vertex_id=0))
    assert e.value.status_code == 409
    assert "wand" in str(e.value.detail)
    print(f"PASS  409: {e.value.detail[:70]}...")


def test_a_vertex_id_outside_the_mesh_is_refused_not_wrapped():
    """A NEGATIVE id must not select from the END of the array.

    CLAUDE.md s.14 records this exact defect in `/cut`: `sel[ids] = True` with
    a negative id built a crown from the far side of the arch and reported
    success.
    """
    sid, v, f, labels = _session_with_labels()
    for bad in (-1, len(v), len(v) + 1000):
        with pytest.raises(HTTPException) as e:
            api_core.select_tooth(sid,
                                  api_core.SelectToothRequest(vertex_id=bad))
        assert e.value.status_code == 400, bad
    print(f"PASS  -1, {len(v)} and {len(v)+1000} all refused 400")


def test_neither_a_point_nor_a_vertex_id_is_refused():
    sid, v, f, labels = _session_with_labels()
    with pytest.raises(HTTPException) as e:
        api_core.select_tooth(sid, api_core.SelectToothRequest())
    assert e.value.status_code == 400
    print("PASS  an empty request is refused 400")


def test_an_expired_session_404s_before_anything_else():
    with pytest.raises(HTTPException) as e:
        api_core.select_tooth("no-such-session",
                              api_core.SelectToothRequest(vertex_id=0))
    assert e.value.status_code == 404
    print("PASS  404 for an expired session")


def test_labels_that_do_not_match_the_mesh_are_refused_not_indexed():
    """A stale label array is the s.24.3 defect. It must not be used."""
    sid, v, f, labels = _session_with_labels()
    STORE.put(sid, "labels", np.zeros(len(v) + 7, np.int64))
    with pytest.raises(HTTPException) as e:
        api_core.select_tooth(sid, api_core.SelectToothRequest(vertex_id=0))
    assert e.value.status_code == 500
    assert "do not match" in str(e.value.detail)
    print("PASS  a label array of the wrong length is refused")


# ---------------------------------------------------------------------------
# The real scan
# ---------------------------------------------------------------------------

_STL = os.path.join(HERE, "case_lower.stl")
_CKPT = os.path.join(HERE, "CrossTooth", "models", "PTv1",
                     "point_best_model.pth")


@pytest.mark.slow
@pytest.mark.skipif(not (os.path.exists(_STL) and os.path.exists(_CKPT)),
                    reason="the real scan or the CrossTooth checkpoint is absent")
def test_on_the_real_scan_every_tooth_selects_itself_and_only_itself():
    """Sixteen clicks, sixteen whole teeth, nothing borrowed from a neighbour.

    This is the claim the wand could not make: measured on the same click
    points, the wand's mean purity over these teeth is 15.7%.
    """
    import segmentation_providers as sp
    import stl_io

    v0, f0 = stl_io.parse_stl_bytes(open(_STL, "rb").read())
    v, f, _ = cg.condition_mesh(v0, f0)

    labels = np.asarray(sp.get("crosstooth").segment(v, f, "lower").labels,
                        np.int64)
    sid = STORE.create("lower")
    STORE.put(sid, "verts", v)
    STORE.put(sid, "faces", f)
    STORE.put(sid, "labels", labels)
    diag = sd.label_report(v, f, labels)
    STORE.put(sid, "segmentation_diagnostics", diag)

    assert len(diag["teeth"]) >= 12, f"only {len(diag['teeth'])} teeth"
    purities = []
    for row in diag["teeth"]:
        fdi = row["label"]
        m = labels == fdi
        centre = v[m].mean(axis=0)
        vid = int(np.where(m)[0][int(np.argmin(
            np.linalg.norm(v[m] - centre, axis=1)))])

        out = api_core.select_tooth(sid,
                                    api_core.SelectToothRequest(vertex_id=vid))
        ids = np.asarray(out["vertex_ids"], np.int64)
        assert out["fdi"] == fdi, (out["fdi"], fdi)
        assert len(ids) > 0
        purity = float((labels[ids] == fdi).mean())
        purities.append(purity)
        assert purity == 1.0, (fdi, purity)

    assert float(np.mean(purities)) == 1.0
    print(f"PASS  {len(purities)} teeth, every selection 100% its own tooth")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
