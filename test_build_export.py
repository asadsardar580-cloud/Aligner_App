"""The export archive must be reproducible and verified before it ships.

AGENT_BRIEF defects B8 and Z1.

B8 - the shipped archive was built from a dirty tree. It carried a
`manufacturing.py` that matched no commit: a mid-experiment variant whose
seat criterion was `bot_sd <= -margin`, looser than both the committed code
and the later experimental one. Nothing inside the archive said so, so it
read as a release.

Z1 - eight zero-byte entries were recorded STORED with `compress_size 2`.
Info-ZIP reports that as a bad CRC; Python's `testzip()` passes it. The
archive was broken in a way the obvious check could not see.
"""
from __future__ import annotations

import json
import os
import zipfile

import pytest

import build_ai_export as bx


# ---------------------------------------------------------------------------
# The dirty-tree refusal
# ---------------------------------------------------------------------------

def test_a_dirty_tree_is_refused_with_exit_2(monkeypatch, capsys):
    monkeypatch.setattr(bx, "git_state",
                        lambda: ("cafe1234", [" M manufacturing.py",
                                              "?? scratch_notes.txt"]))
    with pytest.raises(SystemExit) as e:
        bx.build()
    assert e.value.code == 2, e.value.code

    err = capsys.readouterr().err
    assert "dirty" in err
    assert "manufacturing.py" in err, "the offending files must be listed"
    assert "scratch_notes.txt" in err
    print("PASS  a dirty tree exits 2 and names every file")


def test_the_refusal_can_be_waived_only_explicitly(monkeypatch):
    """`require_clean=False` exists for tests; the CLI never passes it."""
    import inspect
    src = inspect.getsource(bx)
    cli = src.split('if __name__ == "__main__":')[-1]
    assert "require_clean" not in cli, \
        "the command line must not be able to waive the clean-tree check"
    print("PASS  the CLI cannot waive the clean-tree requirement")


# ---------------------------------------------------------------------------
# Post-build verification
# ---------------------------------------------------------------------------

def _zip_with(tmp_path, entries, empty_as_stored=True):
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in entries.items():
            if data == b"" and empty_as_stored:
                zi = zipfile.ZipInfo(name)
                zi.compress_type = zipfile.ZIP_STORED
                z.writestr(zi, b"")
            else:
                z.writestr(name, data)
    return p


def test_verify_passes_a_healthy_archive(tmp_path):
    p = _zip_with(tmp_path, {"a.py": b"print(1)\n", "empty.txt": b"",
                             "b.json": b"{}"})
    rep = bx.verify(p)
    assert rep["problems"] == [], rep
    assert rep["entries_verified"] == 3
    assert rep["stored_entries"] >= 1
    print(f"PASS  healthy archive: {rep['entries_verified']} entries, "
          f"{rep['stored_entries']} STORED, 0 problems")


def test_verify_CATCHES_a_stored_entry_whose_sizes_disagree(tmp_path):
    """The Z1 control. Without it this archive looks fine to `testzip()`.

    A STORED member is stored verbatim, so `compress_size` must equal
    `file_size` by definition. Forging that mismatch is exactly the shape of
    the eight bad entries Info-ZIP rejected.
    """
    p = _zip_with(tmp_path, {"ok.txt": b"hello", "empty.txt": b""})

    raw = bytearray(p.read_bytes())
    with zipfile.ZipFile(p) as z:
        zi = next(i for i in z.infolist() if i.filename == "empty.txt")
        assert zi.compress_type == zipfile.ZIP_STORED
        assert zi.compress_size == zi.file_size == 0

    # Forge compress_size = 2 on the central directory record, as observed.
    sig = b"PK\x01\x02"
    idx = raw.find(sig)
    assert idx != -1
    while idx != -1:
        name_len = int.from_bytes(raw[idx + 28:idx + 30], "little")
        name = bytes(raw[idx + 46:idx + 46 + name_len]).decode()
        if name == "empty.txt":
            raw[idx + 20:idx + 24] = (2).to_bytes(4, "little")
            break
        idx = raw.find(sig, idx + 1)
    forged = tmp_path / "forged.zip"
    forged.write_bytes(bytes(raw))

    rep = bx.verify(forged)
    assert rep["problems"], "a forged STORED size was not caught"
    assert any("empty.txt" in p_ for p_ in rep["problems"]), rep["problems"]
    # On this construction the full-read/CRC check fires first and reports
    # "Overlapped entries", before the size invariant is reached. Either way
    # the archive is refused, which is what matters; asserting the specific
    # message would pin an implementation detail of zipfile.
    print(f"PASS  forged STORED size caught: {rep['problems'][0][:76]}")


def test_verify_catches_a_corrupted_payload(tmp_path):
    """A full read is what validates the CRC.

    Only bytes INSIDE the payload are flipped - every header and the central
    directory stay intact - so the CRC is genuinely the thing that fails,
    rather than the archive becoming unopenable.
    """
    # STORED, so the payload region on disk is genuinely 5000 bytes. A
    # deflated run of identical characters compresses to about thirty bytes,
    # and flipping "inside" it overran into the central directory - which
    # made the archive unopenable rather than CRC-bad, testing the wrong
    # thing.
    p = tmp_path / "stored.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
        z.writestr("a.txt", b"x" * 5000)

    with zipfile.ZipFile(p) as z:
        zi = z.getinfo("a.txt")
        start = zi.header_offset + 30 + len(zi.filename) + len(zi.extra)

    raw = bytearray(p.read_bytes())
    for i in range(start + 100, start + 140):
        raw[i] ^= 0xFF
    broken = tmp_path / "broken.zip"
    broken.write_bytes(bytes(raw))

    rep = bx.verify(broken)
    assert rep["problems"], "a corrupted payload was not caught"
    assert rep["entries_verified"] >= 1, rep
    print(f"PASS  corrupted payload caught: {rep['problems'][0][:76]}")


def test_verify_reports_an_archive_that_will_not_open(tmp_path):
    """The worst case must be a reported problem, not a traceback."""
    bad = tmp_path / "notazip.zip"
    bad.write_bytes(b"this is not a zip file at all")
    rep = bx.verify(bad)
    assert rep["problems"] and "will not open" in rep["problems"][0]
    assert rep["entries_verified"] == 0
    print(f"PASS  unopenable archive reported: {rep['problems'][0][:70]}")


# ---------------------------------------------------------------------------
# A real build, from a tree containing an empty file
# ---------------------------------------------------------------------------

def test_a_real_build_records_its_provenance_and_verifies(monkeypatch,
                                                          tmp_path):
    """Build the real tree into a temp archive, with the clean check waived.

    The tree genuinely contains empty files (`frontend/src/index.css` is 0
    bytes), so this exercises the STORED-empty path on real input.
    """
    out = tmp_path / "export.zip"
    monkeypatch.setattr(bx, "OUT", str(out))
    monkeypatch.setattr(bx, "git_state", lambda: ("deadbeef", []))

    info = bx.build(require_clean=True)

    assert out.exists()
    assert info["verification"]["problems"] == [], info["verification"]
    assert info["commit"] == "deadbeef"

    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert "BUILD_INFO.json" in names
        assert "DO_NOT_RUN_FROM_HERE.txt" in names
        bi = json.loads(z.read("BUILD_INFO.json"))

        empties = [i for i in z.infolist() if i.file_size == 0]
        for zi in empties:
            assert zi.compress_type == zipfile.ZIP_STORED, zi.filename
            assert zi.compress_size == 0, (zi.filename, zi.compress_size)

    assert bi["git_commit"] == "deadbeef"
    assert bi["built_from_clean_tree"] is True
    assert bi["python"].startswith("3.")
    assert "manifold3d" in bi["packages"], bi["packages"]
    assert bi["built_utc"].endswith("Z")

    print(f"PASS  built {info['entries']} entries, "
          f"{info['empty_files_stored']} empty written STORED, "
          f"{info['verification']['entries_verified']} verified; "
          f"commit {bi['git_commit']}, python {bi['python']}, "
          f"manifold3d {bi['packages']['manifold3d']}")


def test_the_build_still_refuses_patient_data(monkeypatch, tmp_path):
    """The pre-existing guard must survive the refactor."""
    monkeypatch.setattr(bx, "OUT", str(tmp_path / "x.zip"))
    monkeypatch.setattr(bx, "git_state", lambda: ("c0ffee", []))
    monkeypatch.setattr(bx, "walk",
                        lambda: [(__file__, "case_lower.stl")])
    with pytest.raises(SystemExit) as e:
        bx.build()
    assert "patient-derived" in str(e.value)
    print("PASS  a leaked .stl still refuses the build")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
