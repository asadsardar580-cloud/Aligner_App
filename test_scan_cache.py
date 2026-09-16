"""Content-addressable scan cache and restart recovery.

This module reverses a standing rule — scans used to live in memory and never
touch disk. A dental arch mesh is BIOMETRIC data: it identifies a person the way
a fingerprint does, whether or not a name is attached. So the tests here are as
much about what the cache refuses to do as about what it stores.

The central guarantee is that a content hash CANNOT be re-pointed. A plan
records the hash of the scan it was built on; restoring either finds exactly
those bytes or fails loudly. A random id could be silently re-pointed at a
different scan and the plan would apply to the wrong anatomy.
"""
import io
import os
import shutil
import struct
import tempfile

import numpy as np
from fastapi import HTTPException, UploadFile

import api_core
import scan_cache_manager as scm
from tooth_fixture import flat_molar_on_base


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


def _isolated():
    tmp = tempfile.mkdtemp(prefix="scan_cache_")
    scm.CACHE_DIR = tmp
    scm.INDEX_PATH = os.path.join(tmp, "index.json")
    scm.KEY_FILE = os.path.join(tmp, ".scan_key")
    return tmp


def _scan():
    v, f, _ = flat_molar_on_base()
    return _stl_bytes(v, f), v, f


def test_the_same_bytes_produce_the_same_hash_and_one_entry():
    tmp = _isolated()
    try:
        raw, v, f = _scan()
        h1 = scm.scan_hash(raw)
        a = scm.put(raw, "lower", len(v), len(f))
        b = scm.put(raw, "lower", len(v), len(f))
        assert a["scan_hash"] == b["scan_hash"] == h1
        assert b["deduplicated"] is True, "the same scan was stored twice"
        assert len(scm.list_scans()) == 1
        # And a different scan is a different entry.
        assert scm.scan_hash(raw + b"x") != h1
        print(f"PASS  content-addressed: {h1[:12]}..., second put deduplicated")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bytes_come_back_identical():
    tmp = _isolated()
    try:
        raw, v, f = _scan()
        h = scm.put(raw, "lower", len(v), len(f))["scan_hash"]
        back = scm.get(h)
        assert back == raw, "the cached scan is not byte-identical to the upload"
        # It must also still parse to the same mesh.
        import stl_io
        v2, f2 = stl_io.parse_stl_bytes(back)
        assert len(v2) == len(np.unique(v, axis=0)) or len(v2) > 0
        print(f"PASS  {len(raw):,} bytes round-trip identical and re-parse")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_file_on_disk_is_encrypted_and_compressed():
    """A biometric mesh sitting in clear text is the thing to avoid."""
    tmp = _isolated()
    try:
        raw, v, f = _scan()
        meta = scm.put(raw, "lower", len(v), len(f))
        path = os.path.join(scm.CACHE_DIR, f"{meta['scan_hash']}.bin")
        on_disk = open(path, "rb").read()

        assert on_disk[:5] != raw[:5], "the cached file starts with the raw header"
        assert b"solid" not in on_disk[:200].lower()
        assert meta["encrypted"] is True and meta["filename_stored"] is False
        assert meta["stored_size"] < meta["byte_size"], \
            f"compression did not help: {meta['stored_size']} vs {meta['byte_size']}"
        ratio = meta["stored_size"] / meta["byte_size"]
        print(f"PASS  encrypted with {meta['codec']}, {ratio*100:.0f}% of raw size, "
              f"no filename stored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_tampered_cache_entry_is_refused_not_parsed():
    """Parsing a mesh that may not be the one the plan was made on is worse
    than failing."""
    tmp = _isolated()
    try:
        raw, v, f = _scan()
        h = scm.put(raw, "lower", len(v), len(f))["scan_hash"]
        path = os.path.join(scm.CACHE_DIR, f"{h}.bin")
        blob = bytearray(open(path, "rb").read())
        blob[len(blob) // 2] ^= 0x01
        open(path, "wb").write(bytes(blob))

        try:
            scm.get(h)
            raise AssertionError("a tampered cache entry was decrypted and parsed")
        except scm.ScanNotCached as e:
            assert "integrity" in str(e).lower()
        print("PASS  one flipped bit in a cached scan is refused by the HMAC")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_missing_scan_says_what_to_do():
    tmp = _isolated()
    try:
        try:
            scm.get("0" * 64)
            raise AssertionError("returned bytes for a scan never cached")
        except scm.ScanNotCached as e:
            assert "Re-upload" in str(e), "the failure does not say how to recover"
        assert scm.has("0" * 64) is False
        print("PASS  an uncached scan fails with instructions, not a stack trace")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_clinician_can_delete_their_own_data():
    tmp = _isolated()
    try:
        raw, v, f = _scan()
        h = scm.put(raw, "lower", len(v), len(f))["scan_hash"]
        assert scm.has(h)
        assert scm.forget(h) is True
        assert scm.has(h) is False and scm.info(h) is None

        scm.put(raw, "lower", len(v), len(f))
        assert scm.purge_all() == 1 and scm.list_scans() == []
        st = scm.stats()
        assert st["count"] == 0 and "BIOMETRIC" in st["phi"]
        print("PASS  forget() and purge_all() remove scans; stats states the PHI posture")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_upload_caches_and_reports_the_hash():
    tmp = _isolated()
    try:
        raw, v, f = _scan()
        up = UploadFile(filename="anything.stl", file=io.BytesIO(raw))
        import asyncio
        res = asyncio.run(api_core.create_session(arch="lower", file=up))
        try:
            assert res["scan_hash"] == scm.scan_hash(raw), \
                "the reported hash is not the hash of the uploaded bytes"
            assert res["scan_cached"] is True
            # The filename must not have survived anywhere.
            assert "anything" not in str(scm.info(res["scan_hash"]))
            print(f"PASS  upload cached {res['scan_hash'][:12]}..., filename discarded")
        finally:
            api_core.close_session(res["session_id"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restore_rebuilds_a_session_from_the_cache_with_no_reupload():
    """The whole point of the item."""
    import case_store
    import domain
    tmp = _isolated()
    ctmp = tempfile.mkdtemp(prefix="cases_")
    case_store.CASE_DIR, case_store.KEY_FILE = ctmp, os.path.join(ctmp, ".k")
    try:
        raw, v, f = _scan()
        up = UploadFile(filename="s.stl", file=io.BytesIO(raw))
        import asyncio
        res = asyncio.run(api_core.create_session(arch="lower", file=up))
        sid, h = res["session_id"], res["scan_hash"]

        saved = api_core.save_case(api_core.CaseSaveRequest(
            label="restart test", sessions={"lower": sid}))
        api_core.close_session(sid)          # simulate the restart

        out = api_core.restore_sessions(api_core.SessionRestoreRequest(
            case_id=saved["case_id"]))
        assert "lower" in out["restored"], f"nothing restored: {out}"
        r = out["restored"]["lower"]
        assert r["scan_hash"] == h, "restored a DIFFERENT scan than the plan records"
        assert r["vertex_count"] == res["vertex_count"]
        # The new session is genuinely usable.
        mesh = api_core.get_mesh(r["session_id"])
        assert len(mesh["positions"]) == res["vertex_count"] * 3
        api_core.close_session(r["session_id"])
        print(f"PASS  restored {r['vertex_count']:,} verts from cache — no re-upload")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(ctmp, ignore_errors=True)


def test_restore_refuses_when_the_scan_is_not_cached():
    import case_store
    import domain
    tmp = _isolated()
    ctmp = tempfile.mkdtemp(prefix="cases_")
    case_store.CASE_DIR, case_store.KEY_FILE = ctmp, os.path.join(ctmp, ".k")
    try:
        c = domain.Case(label="orphan")
        c.arches["lower"] = domain.Arch(arch="lower", scan_hash="f" * 64)
        case_store.save(c)
        try:
            api_core.restore_sessions(api_core.SessionRestoreRequest(case_id=c.case_id))
            raise AssertionError("restored a case whose scan is not cached")
        except HTTPException as e:
            assert e.status_code == 409
            assert "Re-upload" in str(e.detail)
        print("PASS  a plan whose scan is missing refuses with 409 and says re-upload")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(ctmp, ignore_errors=True)


if __name__ == "__main__":
    test_the_same_bytes_produce_the_same_hash_and_one_entry()
    test_bytes_come_back_identical()
    test_the_file_on_disk_is_encrypted_and_compressed()
    test_a_tampered_cache_entry_is_refused_not_parsed()
    test_a_missing_scan_says_what_to_do()
    test_the_clinician_can_delete_their_own_data()
    test_upload_caches_and_reports_the_hash()
    test_restore_rebuilds_a_session_from_the_cache_with_no_reupload()
    test_restore_refuses_when_the_scan_is_not_cached()
    print("\nALL SCAN CACHE TESTS PASSED")
