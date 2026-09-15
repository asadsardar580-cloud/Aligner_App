"""Opt-in, encrypted, local persistence for TREATMENT STATE ONLY.

WHAT IS SAVED, AND WHAT IS DELIBERATELY NOT.

Saved: prescriptions, FDI numbers, root lengths, C_res, review flags, staging —
the clinical decisions. Small, pure data (a full case is a few KB).

NOT saved, ever: the scan. No vertices, no faces, no graph, no concavity field,
no crown meshes, no socket cups. Those stay in memory exactly as before and die
with the process.

That line is the whole design. A prescription is six numbers and a tooth id — a
treatment plan. A mesh of somebody's dentition is biometric data that identifies
them whether or not a name is attached. Only one of those belongs in a file that
outlives the session, so restoring a case after a server restart re-attaches the
scan by reloading the STL, and the plan is waiting for it.

WHAT THE ENCRYPTION ACTUALLY PROTECTS AGAINST — stated plainly, because
overstating it would be worse than not having it.

It protects a case file that gets copied somewhere it should not be: an email
attachment, a synced folder, a USB stick, a backup. That is a real and common
exposure, and it is the one this addresses.

It does NOT protect against someone with access to this workstation. The key
lives beside the data, so anyone who can read the case file can read the key.
Making the clinician type a passphrase per case would change that, and would
also be the kind of friction people work around by writing it down.

It is NOT a regulatory control. Full-disk encryption, an OS account, physical
security and the PHI posture in session_store.py are what actually protect the
patient. This narrows one specific hole.

Fernet is used rather than anything hand-rolled: AES-128-CBC with an
HMAC-SHA256 authentication tag, so a truncated or tampered file is REJECTED
rather than silently decrypted into a half-case. A corrupted treatment plan that
loads anyway is more dangerous than one that refuses to.
"""
from __future__ import annotations

import json
import os
import stat
import time

from cryptography.fernet import Fernet, InvalidToken

import domain

# Next to exports/, which is already the non-cloud-synced local output area.
CASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")
KEY_FILE = os.path.join(CASE_DIR, ".case_key")


def _ensure_dir():
    os.makedirs(CASE_DIR, exist_ok=True)


def _load_or_create_key() -> bytes:
    """One key per installation, generated on first use."""
    _ensure_dir()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as fh:
            return fh.read().strip()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as fh:
        fh.write(key)
    try:
        # Owner read/write only. Advisory on Windows — NTFS ACLs are what
        # actually govern there — but correct on POSIX and harmless here.
        os.chmod(KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return key


def _path(case_id: str) -> str:
    # The id is a uuid4 hex slice, never anything derived from a patient. A
    # filename is the easiest place for a name to leak, so none can reach here.
    safe = "".join(ch for ch in str(case_id) if ch.isalnum() or ch in "-_")[:64]
    if not safe:
        raise ValueError("case_id is empty after sanitising")
    return os.path.join(CASE_DIR, f"{safe}.case")


def save(case: domain.Case) -> str:
    """Write the case. Returns the path. Opt-in: nothing calls this implicitly."""
    _ensure_dir()
    case.updated = time.time()
    payload = json.dumps(case.to_dict(), separators=(",", ":")).encode("utf-8")
    token = Fernet(_load_or_create_key()).encrypt(payload)
    path = _path(case.case_id)
    # Write to a temp file and replace, so an interrupted save cannot leave a
    # half-written plan that decrypts to nothing.
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(token)
    os.replace(tmp, path)
    return path


def load(case_id: str) -> domain.Case:
    path = _path(case_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved case {case_id!r} in {CASE_DIR}")
    with open(path, "rb") as fh:
        token = fh.read()
    try:
        raw = Fernet(_load_or_create_key()).decrypt(token)
    except InvalidToken as e:
        # Authentication failed: wrong key, or the file was altered. Refuse.
        raise ValueError(
            f"Case {case_id!r} could not be decrypted — it was written with a "
            f"different key, or the file has been modified. Refusing to load a "
            f"treatment plan that failed its integrity check.") from e
    return domain.Case.from_dict(json.loads(raw.decode("utf-8")))


def list_cases() -> list[dict]:
    """Every saved case, newest first. Reads each one to report its shape."""
    if not os.path.isdir(CASE_DIR):
        return []
    out = []
    for fn in os.listdir(CASE_DIR):
        if not fn.endswith(".case"):
            continue
        cid = fn[:-len(".case")]
        try:
            c = load(cid)
        except (ValueError, FileNotFoundError, json.JSONDecodeError):
            out.append({"case_id": cid, "readable": False})
            continue
        out.append({
            "case_id": c.case_id, "readable": True, "label": c.label,
            "created": c.created, "updated": c.updated,
            "arches": sorted(c.arches), "tooth_count": len(c.all_teeth()),
            "stage_count": c.stage_count(),
            # The scan is NOT in the file. Restoring needs the STL reloaded.
            "needs_scan_reload": True,
        })
    out.sort(key=lambda r: r.get("updated", 0), reverse=True)
    return out


def delete(case_id: str) -> bool:
    path = _path(case_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
