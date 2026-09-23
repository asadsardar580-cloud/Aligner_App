#!/usr/bin/env python3
"""Rebuild `Aligner_App_AI_Export.zip` from the current tree.

WHY THIS IS A SCRIPT AND NOT A MANUAL STEP. The archive is the handover
artifact — it is what somebody reads when they pick this project up without the
repository — and it was last built by hand. By this sprint it was missing
`ErrorBoundary.jsx`, `AttachmentPanel.jsx`, `AttachmentPlacementTool.js`,
`telemetry.py`, `.semgrep.yml`, two e2e specs and six test modules, while still
looking complete. **An artifact that silently drifts from the tree is worse than
no artifact**, because it is believed.

WHAT IS EXCLUDED, AND THE FIRST RULE IS THE ONE THAT MATTERS:

  * **Nothing patient-derived.** No `.stl`, `.obj`, `.ply`, no `storage/` (the
    encrypted scan cache), and no key file. A dental arch mesh identifies a
    person the way a fingerprint does, and a Fernet key shipped beside its own
    ciphertext is not encryption. This mirrors `.gitignore` deliberately — the
    two must not disagree about what is safe to hand over.
  * Build output and dependencies: `node_modules/`, `dist/`, `ssr_out/`,
    `__pycache__/`, `.venv/`, `.git/`. They are large, regenerable, and
    `pytest` once failed to collect at all because a previous archive held a
    second copy of every test file ("import file mismatch").
  * Model checkpoints (`*.h5`, `*.pth`) — 64MB each and gitignored.

`case_lower.stl_output.json` IS included, deliberately: it is ToothGroupNetwork's
own prediction on the sample scan, it is the only label file anyone can inspect
without the model, and `benchmark_segmentation.py` refuses to score against it
precisely because it is not ground truth. Keeping it makes that refusal legible.

Run:  python build_ai_export.py
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Aligner_App_AI_Export.zip")

#: Packages whose version changes what the archive's code actually DOES.
#: manifold3d decides every boolean; numpy 1 vs 2 changes array semantics.
BUILD_INFO_PACKAGES = ("numpy", "scipy", "manifold3d", "open3d", "trimesh",
                       "torch", "scikit-learn", "fastapi", "pydantic")

# Directories never descended into.
SKIP_DIRS = {
    ".git", ".venv", ".venv_check", "venv", "node_modules", "__pycache__",
    ".pytest_cache",
    ".ruff_cache", "dist", "ssr_out", "ssr_out2", "test-results",
    "playwright-report", "blob-report", "exports", "segmented", "scratch",
    "Aligner_App_AI_Export",          # never nest the archive inside itself
    "storage",                        # ENCRYPTED PATIENT SCANS AND THE KEY
}

# Extensions never included. `.stl` is first for a reason.
SKIP_EXT = {
    ".stl", ".obj", ".ply",           # patient geometry
    ".h5", ".pth", ".ckpt", ".pt",    # model checkpoints
    ".pyc", ".pyo", ".zip", ".log",
}

# THE LAUNCHERS ARE DELIBERATELY NOT SHIPPED.
#
# The mirror is a full tree copy with every checkpoint stripped, so a backend
# started inside it comes up with all of the code and none of the model: the app
# runs, the viewport works, every recent fix is present, and the only thing
# missing is the AI. There is nothing to notice. Both launchers now refuse to
# run from here and api_core.py prints a banner at import, but the cleanest
# guard is the one where the button does not exist - you cannot double-click a
# file that is not there. DO_NOT_RUN_FROM_HERE.txt takes their place.
LAUNCHERS = {"start_backend.bat", "start_frontend.bat"}

SKIP_NAMES = {".scan_key", ".case_key", "Aligner_App_AI_Export.zip"} | LAUNCHERS

DO_NOT_RUN = """THIS IS A READING COPY. DO NOT RUN THE APP FROM HERE.

Aligner_App_AI_Export/ is a snapshot of the project for handing to someone who
does not have the repository. It is a full copy of the source, refreshed from
the tree, so everything in it is CURRENT.

What it does NOT contain is the AI model checkpoints. build_ai_export.py strips
every .h5/.pth/.ckpt/.pt on purpose - a 64 MB model file per checkpoint, twelve
of them, is not source code.

That matters more than it sounds. A backend started in this folder would come up
with all of the code and none of the model, so the app would start, the 3D
viewport would work, every recent fix would be present, and "Segment Teeth"
would report the AI as unavailable with nothing on screen to explain why. It
reads as a corrupted install. It is not - it is the wrong working copy.

start_backend.bat and start_frontend.bat are therefore omitted from this
snapshot. To RUN the application, use the project root:

    <project root>/start_backend.bat
    <project root>/start_frontend.bat

To READ the project, start with CLAUDE.md - it is the authoritative record.
"""

# Kept although the extension rule would otherwise be ambiguous.
KEEP_ANYWAY = {"case_lower.stl_output.json"}

# Hidden directories are skipped wholesale, with these exceptions. `.github`
# holds the CI definition, which is part of the handover: somebody picking this
# up needs to see what the gates ARE, not infer them from a passing badge they
# cannot run.
KEEP_HIDDEN_DIRS = {".github"}


def included(path: str) -> bool:
    name = os.path.basename(path)
    if name in KEEP_ANYWAY:
        return True
    if name in SKIP_NAMES:
        return False
    ext = os.path.splitext(name)[1].lower()
    if ext in SKIP_EXT:
        return False
    return True


def walk():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS
                             and (not d.startswith(".") or d in KEEP_HIDDEN_DIRS))
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
            if any(part in SKIP_DIRS for part in rel.split("/")):
                continue
            if not included(full):
                continue
            yield full, rel


def git_state():
    """(commit, dirty_paths). A build from a dirty tree is not reproducible."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8",
                              errors="replace").stdout.strip()
    return git("rev-parse", "HEAD"), [
        ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]


def build_info(commit):
    """What produced this archive, recorded INSIDE it.

    B8. The shipped archive was built from a dirty tree and carried a
    `manufacturing.py` matching no commit - a mid-experiment variant whose
    seat criterion was looser than both the committed and the latest code.
    Nothing in the archive said so, so it read as a release. Versions come
    from THIS interpreter rather than a requirements file, because the file
    records an intent and the interpreter records a fact.
    """
    import importlib.metadata as md
    versions = {}
    for name in BUILD_INFO_PACKAGES:
        try:
            versions[name] = md.version(name)
        except Exception:                                 # noqa: BLE001
            versions[name] = "NOT INSTALLED"
    return {
        "git_commit": commit,
        "built_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "built_from_clean_tree": True,
        "python": sys.version.split()[0],
        "python_executable": os.path.basename(sys.executable),
        "packages": versions,
        "note": ("Built by build_ai_export.py from a clean working tree. "
                 "Every entry was re-read and CRC-checked after writing."),
    }


def verify(path) -> dict:
    """Re-open the finished archive and check every entry.

    Z1. Eight zero-byte entries were recorded STORED with `compress_size 2`,
    which Info-ZIP reports as a bad CRC while Python's `testzip()` passes -
    so the archive was broken in a way the obvious check could not see. Two
    invariants are asserted here rather than hoped for:

      * every entry reads back in full, which is what validates its CRC;
      * a STORED entry has `compress_size == file_size`, by definition.
    """
    problems, stored, total = [], 0, 0
    try:
        zf = zipfile.ZipFile(path)
    except Exception as e:                                # noqa: BLE001
        # An archive that will not even open is the worst outcome, and it
        # must be a reported problem rather than a traceback out of build().
        return {"entries_verified": 0, "stored_entries": 0,
                "problems": [f"the archive will not open: "
                             f"{type(e).__name__}: {e}"]}
    with zf as z:
        bad = z.testzip()
        if bad is not None:
            problems.append(f"testzip() reports a bad entry: {bad}")
        for zi in z.infolist():
            total += 1
            try:
                data = z.read(zi.filename)           # full read = CRC check
            except Exception as e:                        # noqa: BLE001
                problems.append(f"{zi.filename}: unreadable ({type(e).__name__}: {e})")
                continue
            if len(data) != zi.file_size:
                problems.append(
                    f"{zi.filename}: read {len(data)} bytes, header says "
                    f"{zi.file_size}")
            if zi.compress_type == zipfile.ZIP_STORED:
                stored += 1
                if zi.compress_size != zi.file_size:
                    problems.append(
                        f"{zi.filename}: STORED but compress_size "
                        f"{zi.compress_size} != file_size {zi.file_size}")
    return {"entries_verified": total, "stored_entries": stored,
            "problems": problems}


def build(require_clean=True) -> dict:
    commit, dirty = git_state()
    if require_clean and dirty:
        print("REFUSING to build the export archive: the working tree is "
              "dirty.", file=sys.stderr)
        print("An archive built from a dirty tree matches no commit and "
              "cannot be reproduced. This has happened: the shipped ZIP "
              "carried a manufacturing.py from no commit, with a looser "
              "seat criterion than either the committed or the latest code.",
              file=sys.stderr)
        for line in dirty:
            print(f"    {line}", file=sys.stderr)
        raise SystemExit(2)

    files = list(walk())

    # A final assertion rather than a comment. The exclusion rules above are
    # easy to edit and hard to re-verify by eye, so the one thing that must
    # never happen is checked outright before anything is written.
    leaked = [r for _, r in files
              if os.path.splitext(r)[1].lower() in {".stl", ".obj", ".ply"}
              or r.startswith("storage/") or r.endswith((".scan_key", ".case_key"))]
    if leaked:
        raise SystemExit(f"REFUSING to build: patient-derived files matched: {leaked}")

    tmp = OUT + ".tmp"
    empties = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for full, rel in files:
            # ZERO-BYTE FILES ARE WRITTEN STORED, EXPLICITLY (Z1). Deflating
            # an empty member produced a 2-byte payload recorded as STORED in
            # eight entries of a previous archive - Info-ZIP called that a bad
            # CRC, Python's testzip() did not notice. Not reproduced on this
            # interpreter, so it is environment-specific; writing empties
            # STORED and then asserting `compress_size == file_size` in
            # `verify()` makes the invariant hold regardless.
            if os.path.getsize(full) == 0:
                zi = zipfile.ZipInfo(rel, date_time=time.localtime(
                    os.path.getmtime(full))[:6])
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = 0o644 << 16
                z.writestr(zi, b"")
                empties += 1
            else:
                z.write(full, rel)
        # Written into the ARCHIVE rather than kept in the tree: it is only
        # true of the snapshot, and a file in the project root saying "do not
        # run the app from here" would be false exactly where it sits.
        z.writestr("DO_NOT_RUN_FROM_HERE.txt", DO_NOT_RUN)
        z.writestr("BUILD_INFO.json",
                   json.dumps(build_info(commit), indent=2) + "\n")

    report = verify(tmp)
    if report["problems"]:
        os.remove(tmp)
        print("REFUSING to ship the archive: post-build verification failed.",
              file=sys.stderr)
        for p in report["problems"]:
            print(f"    {p}", file=sys.stderr)
        raise SystemExit(3)

    os.replace(tmp, OUT)

    by_top: dict[str, int] = {}
    for _, rel in files:
        top = rel.split("/")[0] if "/" in rel else "(root)"
        by_top[top] = by_top.get(top, 0) + 1

    by_top["(root)"] = by_top.get("(root)", 0) + 2   # DO_NOT_RUN + BUILD_INFO
    return {"entries": len(files) + 2, "bytes": os.path.getsize(OUT),
            "by_top": by_top, "launchers_excluded": sorted(LAUNCHERS),
            "commit": commit, "empty_files_stored": empties,
            "verification": report}


if __name__ == "__main__":
    t0 = time.perf_counter()
    info = build()
    print(f"Aligner_App_AI_Export.zip  {info['entries']} entries, "
          f"{info['bytes'] / 1e6:.2f} MB, {time.perf_counter() - t0:.1f}s")
    for top, n in sorted(info["by_top"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {top}")
    print(f"\n  commit           {info['commit']}")
    print(f"  empty files      {info['empty_files_stored']} written STORED")
    print(f"  verified         {info['verification']['entries_verified']} "
          f"entries re-read and CRC-checked, "
          f"{info['verification']['stored_entries']} STORED")
    print("\nNo .stl / .obj / .ply, no storage/, no key file — asserted, not assumed.")
    print(f"Launchers withheld so the snapshot cannot be started by "
          f"accident: {', '.join(info['launchers_excluded'])}")
    sys.exit(0)
