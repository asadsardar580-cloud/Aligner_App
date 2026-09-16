"""Content-addressable scan cache, so a restart does not cost a re-upload.

THIS REVERSES AN EARLIER DECISION, DELIBERATELY AND ON RECORD. Until now the
rule was absolute: scans live in memory and never touch disk
(session_store.py:16-17), and the encrypted case file carries the treatment plan
only. That made a server restart cost a manual re-upload of a 9MB scan.

Persisting scans buys back that workflow and changes the PHI posture, so the
change is made with its costs stated rather than quietly:

  * A dental arch mesh is BIOMETRIC DATA. It identifies a person whether or not
    a name is attached, in the same way a fingerprint does. Writing one to disk
    is a materially different act from writing six prescription numbers.
  * Scans are therefore stored ENCRYPTED, with the same Fernet key the case
    files use — AES-128-CBC plus an HMAC-SHA256 tag, so a tampered or truncated
    scan is refused rather than silently parsed into a wrong mesh.
  * The cache is content-addressable by SHA-256 of the RAW UPLOAD BYTES. The
    same scan uploaded twice occupies one entry, and the hash is computed before
    conditioning so it identifies what the clinician actually sent.
  * NO FILENAME IS EVER STORED. Filenames routinely carry patient names, which
    is precisely why session_store stores none; the cache key is a hash and the
    metadata is counts and timestamps.
  * Nothing leaves the machine. No endpoint uploads a cached scan anywhere.

WHY SHA-256 AND NOT A UUID. A content hash means the cache cannot go stale
against the plan: a case records the hash of the scan it was planned on, so
restoring either finds exactly that mesh or fails loudly. A random id could be
re-pointed at a different scan and the plan would silently apply to the wrong
anatomy.

COMPRESSION. zstandard when available, zlib otherwise — an STL is highly
compressible ASCII-free float data and this typically halves it. The codec used
is recorded per entry so a cache written by one installation still reads on
another.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import zlib

from cryptography.fernet import Fernet, InvalidToken

_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_ROOT, "storage", "scans")
INDEX_PATH = os.path.join(CACHE_DIR, "index.json")
KEY_FILE = os.path.join(CACHE_DIR, ".scan_key")

try:
    import zstandard as _zstd
    _HAVE_ZSTD = True
except ImportError:
    _zstd = None
    _HAVE_ZSTD = False


class ScanNotCached(FileNotFoundError):
    """The plan references a scan this installation does not hold."""


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _key() -> bytes:
    _ensure_dir()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as fh:
            return fh.read().strip()
    k = Fernet.generate_key()
    with open(KEY_FILE, "wb") as fh:
        fh.write(k)
    try:
        os.chmod(KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return k


def scan_hash(raw: bytes) -> str:
    """SHA-256 of the raw upload. Computed BEFORE conditioning, so it identifies
    what the clinician sent rather than what the pipeline made of it."""
    return hashlib.sha256(raw).hexdigest()


def _compress(raw: bytes) -> tuple[bytes, str]:
    if _HAVE_ZSTD:
        return _zstd.ZstdCompressor(level=10).compress(raw), "zstd"
    return zlib.compress(raw, 6), "zlib"


def _decompress(blob: bytes, codec: str) -> bytes:
    if codec == "zstd":
        if not _HAVE_ZSTD:
            raise ScanNotCached(
                "This scan was cached with zstandard, which is not installed here. "
                "pip install zstandard, or re-upload the scan.")
        return _zstd.ZstdDecompressor().decompress(blob)
    return zlib.decompress(blob)


def _index() -> dict:
    if not os.path.exists(INDEX_PATH):
        return {}
    try:
        with open(INDEX_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt index must not take the cache with it: the blobs are named
        # by hash, so the index is a convenience and can be rebuilt.
        return {}


def _write_index(ix: dict):
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ix, fh, indent=2)
    os.replace(tmp, INDEX_PATH)


def put(raw: bytes, arch: str, vertex_count: int = 0, face_count: int = 0) -> dict:
    """Cache a scan. Idempotent — the same bytes produce the same entry."""
    _ensure_dir()
    h = scan_hash(raw)
    path = os.path.join(CACHE_DIR, f"{h}.bin")

    ix = _index()
    if h in ix and os.path.exists(path):
        ix[h]["last_seen"] = time.time()
        _write_index(ix)
        return dict(ix[h], cached=True, deduplicated=True)

    blob, codec = _compress(raw)
    token = Fernet(_key()).encrypt(blob)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(token)
    os.replace(tmp, path)

    meta = {
        "scan_hash": h,
        "arch_type": arch,
        "creation_timestamp": time.time(),
        "last_seen": time.time(),
        "byte_size": len(raw),
        "stored_size": os.path.getsize(path),
        "vertex_count": int(vertex_count),
        "face_count": int(face_count),
        "codec": codec,
        # Stated on every entry so an auditor reading the index does not have to
        # infer it from the absence of a field.
        "filename_stored": False,
        "encrypted": True,
    }
    ix[h] = meta
    _write_index(ix)
    return dict(meta, cached=True, deduplicated=False)


def get(h: str) -> bytes:
    """The original raw upload bytes, byte-identical, or a loud failure."""
    path = os.path.join(CACHE_DIR, f"{h}.bin")
    if not os.path.exists(path):
        raise ScanNotCached(
            f"Scan {h[:12]}... is not in this installation's cache. The treatment "
            f"plan references a scan that was never cached here, or the cache was "
            f"cleared. Re-upload the original file — its hash must match.")
    with open(path, "rb") as fh:
        token = fh.read()
    try:
        blob = Fernet(_key()).decrypt(token)
    except InvalidToken as e:
        raise ScanNotCached(
            f"Cached scan {h[:12]}... failed its integrity check — written with a "
            f"different key, or modified on disk. Refusing to parse a mesh that "
            f"may not be the one the plan was made on.") from e

    raw = _decompress(blob, _index().get(h, {}).get("codec", "zlib"))

    # The hash is the contract. If the bytes that come back do not hash to the
    # name they were filed under, something is wrong that silence would hide.
    if scan_hash(raw) != h:
        raise ScanNotCached(
            f"Cached scan {h[:12]}... does not hash to its own name after "
            f"decompression. The cache entry is corrupt.")
    return raw


def has(h: str) -> bool:
    return os.path.exists(os.path.join(CACHE_DIR, f"{h}.bin"))


def info(h: str) -> dict | None:
    return _index().get(h)


def list_scans() -> list[dict]:
    rows = [dict(m, present=has(k)) for k, m in _index().items()]
    rows.sort(key=lambda r: r.get("last_seen", 0), reverse=True)
    return rows


def forget(h: str) -> bool:
    """Delete one cached scan. The clinician's control over their own data."""
    path = os.path.join(CACHE_DIR, f"{h}.bin")
    existed = os.path.exists(path)
    if existed:
        os.remove(path)
    ix = _index()
    if h in ix:
        del ix[h]
        _write_index(ix)
    return existed


def purge_all() -> int:
    """Remove every cached scan. Returns how many went."""
    n = 0
    for h in list(_index()):
        if forget(h):
            n += 1
    return n


def stats() -> dict:
    rows = list_scans()
    return {
        "count": len(rows),
        "present": sum(1 for r in rows if r["present"]),
        "raw_bytes": sum(r.get("byte_size", 0) for r in rows),
        "stored_bytes": sum(r.get("stored_size", 0) for r in rows),
        "codec": "zstd" if _HAVE_ZSTD else "zlib",
        "encrypted": True,
        "location": CACHE_DIR,
        "phi": "Scans are BIOMETRIC data. They are encrypted at rest, keyed by "
               "content hash, and no filename is stored. Nothing leaves this "
               "machine. Use forget()/purge_all() to remove them.",
    }
