"""A request cannot write outside `exports/`, and cannot exhaust memory.

AGENT_BRIEF defects B10 and B15.

B10 - `out_dir` went straight from the request body into `os.makedirs(...)`
and `open(..., "wb")` with no validation, gated only by a `keep_local`
boolean the same client sets. On an API with no authentication in front of
it that is an arbitrary directory-creation and file-write primitive. It is
confined rather than deleted, because `keep_local` is a real workflow: a
clinician asking for the files to stay on this machine.

B15 - the upload was one unbounded `await file.read()`, so the whole body
became process memory before anything looked at it.
"""
from __future__ import annotations

import asyncio
import io
import os
import struct

import pytest
from fastapi import HTTPException, UploadFile

import api_core
from api_core import STORE


# ---------------------------------------------------------------------------
# out_dir confinement
# ---------------------------------------------------------------------------

def test_the_default_is_the_exports_root():
    assert str(api_core._confined_out_dir(None)) == \
        str(os.path.realpath(api_core.EXPORTS_ROOT))
    print(f"PASS  default -> {api_core.EXPORTS_ROOT}")


def test_a_relative_path_lands_under_the_root():
    got = str(api_core._confined_out_dir("case-a/stages"))
    root = str(os.path.realpath(api_core.EXPORTS_ROOT))
    assert got.startswith(root), got
    print(f"PASS  'case-a/stages' -> {got}")


@pytest.mark.parametrize("bad", [
    "C:/Windows/Temp",
    "/etc",
    "../../evil",
    "../sibling",
    "C:/",
    "~",
    "exports/../../outside",
])
def test_an_escape_attempt_is_refused_422_not_silently_redirected(bad):
    """422 naming the root. A silent redirect would write somewhere the
    caller did not ask for, which is its own kind of wrong."""
    with pytest.raises(HTTPException) as e:
        api_core._confined_out_dir(bad)
    assert e.value.status_code == 422, (bad, e.value.status_code)
    assert "exports" in str(e.value.detail).lower()
    print(f"PASS  {bad!r} refused 422")


def test_the_request_models_still_accept_out_dir():
    """Confined, not removed - the workflow survives."""
    r = api_core.StageExportRequest(keep_local=True, out_dir="case-b")
    assert r.out_dir == "case-b"
    print("PASS  out_dir is still a usable field")


def test_both_export_call_sites_go_through_the_helper():
    """Static: neither write site may reconstruct a path of its own."""
    import inspect
    for fn in (api_core.build_export_bundle, api_core.build_stage_bundle):
        src = inspect.getsource(fn)
        if "keep_local" not in src:
            continue
        assert "_confined_out_dir" in src, \
            f"{fn.__name__} writes without confining the path"
    print("PASS  both export write sites call _confined_out_dir")


# ---------------------------------------------------------------------------
# Upload cap
# ---------------------------------------------------------------------------

def _stl_bytes(n_tris):
    buf = io.BytesIO()
    buf.write(b"\0" * 80)
    buf.write(struct.pack("<I", n_tris))
    for i in range(n_tris):
        buf.write(struct.pack("<3f", 0, 0, 1))
        for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
            buf.write(struct.pack("<3f", *c))
        buf.write(struct.pack("<H", 0))
    return buf.getvalue()


class _EndlessUpload:
    """Streams more than the cap, so the read loop must stop on its own."""

    def __init__(self, limit):
        self.sent = 0
        self.limit = limit

    async def read(self, size=-1):
        if self.sent >= self.limit:
            return b""
        n = size if size and size > 0 else 65536
        self.sent += n
        return b"\0" * n


def test_an_oversized_upload_is_refused_413(monkeypatch):
    monkeypatch.setattr(api_core, "MAX_UPLOAD_BYTES", 1 * 1024 * 1024)
    monkeypatch.setattr(api_core, "UPLOAD_CHUNK_BYTES", 256 * 1024)

    before = len(STORE.keys_all()) if hasattr(STORE, "keys_all") else None
    with pytest.raises(HTTPException) as e:
        asyncio.run(api_core.create_session(
            arch="lower", file=_EndlessUpload(8 * 1024 * 1024)))

    assert e.value.status_code == 413, e.value.status_code
    assert "exceeds" in str(e.value.detail)
    print(f"PASS  an 8 MB body against a 1 MB cap -> 413")
    if before is not None:
        assert len(STORE.keys_all()) == before, \
            "a refused upload left its session behind"


def test_a_normal_upload_still_works():
    """The control for the cap: a real scan must still load."""
    up = UploadFile(filename="scan.stl", file=io.BytesIO(_stl_bytes(64)))
    res = asyncio.run(api_core.create_session(arch="lower", file=up))
    sid = res["session_id"]
    try:
        assert STORE.require(sid, "verts") is not None
        print(f"PASS  a 64-triangle upload loaded: "
              f"{res.get('vertex_count', '?')} vertices")
    finally:
        STORE.drop(sid)


def test_the_cap_is_a_named_constant_not_a_literal():
    assert api_core.MAX_UPLOAD_BYTES == 200 * 1024 * 1024
    assert api_core.UPLOAD_CHUNK_BYTES > 0
    print(f"PASS  MAX_UPLOAD_BYTES = "
          f"{api_core.MAX_UPLOAD_BYTES // (1024*1024)} MB")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
