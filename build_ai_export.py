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

import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Aligner_App_AI_Export.zip")

# Directories never descended into.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
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

SKIP_NAMES = {".scan_key", ".case_key", "Aligner_App_AI_Export.zip"}

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


def build() -> dict:
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
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for full, rel in files:
            z.write(full, rel)
    os.replace(tmp, OUT)

    by_top: dict[str, int] = {}
    for _, rel in files:
        top = rel.split("/")[0] if "/" in rel else "(root)"
        by_top[top] = by_top.get(top, 0) + 1

    return {"entries": len(files), "bytes": os.path.getsize(OUT), "by_top": by_top}


if __name__ == "__main__":
    t0 = time.perf_counter()
    info = build()
    print(f"Aligner_App_AI_Export.zip  {info['entries']} entries, "
          f"{info['bytes'] / 1e6:.2f} MB, {time.perf_counter() - t0:.1f}s")
    for top, n in sorted(info["by_top"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {top}")
    print("\nNo .stl / .obj / .ply, no storage/, no key file — asserted, not assumed.")
    sys.exit(0)
