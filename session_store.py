"""
session_store.py — holds a conditioned arch scan in memory so the client can
send coordinates instead of geometry.

WHY SESSIONS AND NOT A STATELESS ENDPOINT
------------------------------------------
Measured on this project's own scans: a full arch is 34.2MB, a per-click round
trip costs 78s on clinic DSL against 0.46s of local compute. The magic wand is
a click-and-see-instantly interaction. Re-uploading the arch per click makes it
unusable. So geometry crosses the wire exactly twice per case — once on upload,
once on export — and everything between is coordinates, indices and matrices.

PHI CONSTRAINTS BUILT IN, NOT BOLTED ON
----------------------------------------
An intraoral scan is health information. This store therefore:
  * holds scans in memory only, never writing them to disk;
  * expires them on a timer, so a forgotten browser tab does not leave a
    patient's arch resident indefinitely;
  * caps how many can be resident at once, so a loop of failed uploads cannot
    exhaust RAM;
  * stores the ORIGINAL FILENAME NOWHERE. Filenames travel with files in ways
    contents do not, and they routinely carry patient names.
The caller still owns the wider obligations: this runs on 127.0.0.1 and is not
a networked health-information system.
"""

import threading
import time
import uuid

DEFAULT_TTL_SECONDS = 60 * 60          # one clinical session
DEFAULT_MAX_SESSIONS = 4               # two patients, two arches each


class SessionExpired(KeyError):
    """Raised when a session id is unknown or has aged out."""


class _Session:
    __slots__ = ("id", "arch", "created", "touched", "data")

    def __init__(self, sid, arch, now):
        # timestamps come from the store's clock, never time.time() directly.
        # Mixing the two silently disables expiry: the comparison
        # `clock() - touched > ttl` is meaningless across two time bases, and
        # it fails OPEN — the session lives forever with a patient's scan in it.
        self.id = sid
        self.arch = arch
        self.created = now
        self.touched = now
        self.data = {}


class SessionStore:
    def __init__(self, ttl_seconds=DEFAULT_TTL_SECONDS,
                 max_sessions=DEFAULT_MAX_SESSIONS, clock=time.time):
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._clock = clock
        self._lock = threading.RLock()
        self._sessions = {}

    # -- lifecycle ---------------------------------------------------------

    def create(self, arch):
        """
        Start a session. `arch` is 'upper' or 'lower' and comes from the
        clinician's selection in the UI — never inferred from a filename.
        """
        arch = str(arch).strip().lower()
        if arch not in ("upper", "lower"):
            raise ValueError(
                f"arch must be 'upper' or 'lower', got {arch!r}. This value "
                f"decides the FDI quadrant offset; do not guess it.")
        with self._lock:
            self._reap()
            if len(self._sessions) >= self._max:
                # drop the least recently touched rather than refusing —
                # a clinician mid-case should never be blocked by a stale tab.
                # `created` breaks ties so eviction is deterministic when two
                # sessions were touched in the same clock tick.
                oldest = min(self._sessions.values(),
                             key=lambda s: (s.touched, s.created))
                self._sessions.pop(oldest.id, None)
            sid = uuid.uuid4().hex
            self._sessions[sid] = _Session(sid, arch, self._clock())
            return sid

    def _get(self, sid):
        s = self._sessions.get(sid)
        if s is None:
            raise SessionExpired(
                "Unknown session. It may have expired, or the server may have "
                "restarted — uvicorn --reload clears memory on every code edit.")
        if self._clock() - s.touched > self._ttl:
            self._sessions.pop(sid, None)
            raise SessionExpired("Session expired; re-upload the scan.")
        s.touched = self._clock()
        return s

    def arch(self, sid):
        with self._lock:
            return self._get(sid).arch

    def put(self, sid, key, value):
        with self._lock:
            self._get(sid).data[key] = value

    def get(self, sid, key, default=None):
        with self._lock:
            return self._get(sid).data.get(key, default)

    def keys(self, sid):
        """Snapshot of this session's data keys.

        A list, not a view: callers iterate it while calling get() (and the
        lock is not reentrant across those calls), so handing out a live
        dict_keys would raise the moment anything writes mid-iteration.
        """
        with self._lock:
            return sorted(self._get(sid).data.keys())

    def require(self, sid, key):
        """Fetch a value that must already exist, with a useful error."""
        with self._lock:
            s = self._get(sid)
            if key not in s.data:
                raise SessionExpired(
                    f"Session has no {key!r} yet. Call the endpoints in order: "
                    f"upload, then segment, then select.")
            return s.data[key]

    def drop(self, sid):
        with self._lock:
            self._sessions.pop(sid, None)

    def _reap(self):
        now = self._clock()
        for sid in [s.id for s in self._sessions.values()
                    if now - s.touched > self._ttl]:
            self._sessions.pop(sid, None)

    # -- introspection -----------------------------------------------------

    def stats(self):
        """Serialisable, and deliberately free of anything patient-identifying."""
        with self._lock:
            self._reap()
            now = self._clock()
            return {
                "count": len(self._sessions),
                "max": self._max,
                "ttl_seconds": self._ttl,
                "sessions": [
                    {"id": s.id[:8] + "...",
                     "arch": s.arch,
                     "age_seconds": round(now - s.created, 1),
                     "keys": sorted(s.data.keys())}
                    for s in self._sessions.values()
                ],
            }


STORE = SessionStore()
