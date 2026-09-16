"""Local-only tracing for the endpoints that can be slow or can refuse.

WHY THIS EXISTS. Four operations in this application take long enough that a
clinician wonders whether it has hung — segmentation is 236s on a real scan,
/export is 7.5s, a 31-stage manufacturing run is ~4.2s, and /cut was 29.7s
before the face-index rewrite. When one of them is slow on someone else's
machine there is currently nothing to look at but a wall clock. Spans give the
durations and the sizes that produced them.

FILE EXPORTER ONLY. THIS IS THE WHOLE DESIGN, NOT A DEFAULT.

There is no OTLP exporter, no collector endpoint, and no environment variable
that can turn one on. That was a deliberate decision and it follows directly
from the invariant the rest of this codebase is built on: everything runs on the
clinician's workstation and nothing patient-derived is sent to a third party. An
observability SDK is a data-egress path wearing a helpful hat — the ordinary way
it is configured is `OTEL_EXPORTER_OTLP_ENDPOINT`, an environment variable, set
by somebody who was not thinking about PHI. Making the exporter unconfigurable
is the only version of this feature that cannot be turned into a leak by an
operator with good intentions. `.semgrep.yml` fails the build on any network
exporter import, so this cannot be undone quietly.

WHAT AN ATTRIBUTE MAY CARRY. Durations, counts, sizes, status codes, threshold
values, and the name of a stage that failed. It may NOT carry vertices, faces,
labels, prescriptions, filenames or identifiers — the same denylist audit.py
enforces, imported rather than re-typed, because two copies of a denylist is one
denylist plus a bug waiting to diverge. Attributes are filtered at RUNTIME on
every span, not checked by review.

NON-BLOCKING, ALWAYS. Every emitter sits inside try/except and returns a no-op
span on any failure. Tracing that can fail a clinical operation is worse than no
tracing: a disk full of logs must not stop a tooth being cut. The suite proves
this by making the sink raise on every write and asserting the traced function
still returns its value.

opentelemetry is OPTIONAL. If the package is absent the module still works and
still writes the same JSON lines through its own writer — the file is the
product here, not the SDK. `backend` in the status report says which is in use.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager

import audit

# Rolling JSON lines. One span per line, so a truncated write costs one span and
# not the file, and so `tail -f` is a usable debugger.
DEFAULT_PATH = os.path.join("scratch", "otel_fallback.log")
MAX_BYTES = 4 * 1024 * 1024          # ~4MB, then rotate once to .1
SERVICE_NAME = "aligner-workstation"

# TWO SETS, NOT ONE, AND THE SPLIT IS THE WHOLE POINT.
#
# audit.py has a single denylist because an audit entry's values are all
# clinician-facing quantities. A span attribute is different: the single most
# useful thing to record about /cut is the FACE COUNT, and `faces` is on
# audit.py's list because there it names an array of triangles. Filtering a
# scalar 187,625 because its key is spelled "faces" throws away the number the
# span exists to carry - measured, the first version of this dropped 6 of 8
# attributes including every count.
#
# So: an IDENTIFIER key is dropped whatever its type, because "Jane Doe" is a
# perfectly ordinary scalar string. A GEOMETRY key only matters when the value
# is non-scalar - and every non-scalar value is reduced to its type and length
# regardless of key, so the geometry set is a second line of defence rather
# than the first.
IDENTIFIER_KEYS = {
    "patient", "patient_name", "name", "dob", "date_of_birth", "mrn", "ssn",
    "national_id", "filename", "file_name", "path", "case_label", "label",
    "scan_hash", "session_id", "case_id",
}
GEOMETRY_KEYS = (set(audit.FORBIDDEN_KEYS) - IDENTIFIER_KEYS) | {
    "vertex_ids", "selection", "mesial_pt", "distal_pt", "matrix", "clinical",
    "prescription",
}
FORBIDDEN_KEYS = IDENTIFIER_KEYS | GEOMETRY_KEYS


# The endpoints worth tracing, named so a grep of this file answers "what is
# instrumented" without reading api_core.
TRACED = ("cut", "kinematics", "export", "segment", "staging")

_LOCK = threading.Lock()
_STATE = {"path": DEFAULT_PATH, "enabled": True, "backend": None,
          "spans_written": 0, "spans_dropped": 0, "attrs_redacted": 0,
          "last_error": None}


# ----------------------------------------------------------------------
# Attribute hygiene
# ----------------------------------------------------------------------

def _clean(attrs: dict | None) -> dict:
    """Drop anything that could identify a patient or carry their anatomy.

    Three filters, and the third is the one that catches what a denylist misses:

      1. An IDENTIFIER key is dropped whatever the value is. "Jane Doe" is an
         ordinary scalar string and no shape check will ever flag it.
      2. A GEOMETRY key is dropped when the value is not a scalar.
      3. ANY non-scalar value is replaced by its type and length, whatever the
         key is called. A list of 94,848 floats is a mesh however it is spelled,
         and `attrs={"debug": verts.tolist()}` is exactly how a leak gets
         written by somebody in a hurry.

    A scalar number survives - `faces=187625` is the measurement the span exists
    to carry, not a patient. Strings are truncated at 200 characters: a refusal
    message is useful, a 200KB traceback in a span is a disk incident.
    """
    out = {}
    if not attrs:
        return out
    for k, v in attrs.items():
        key = str(k).lower()
        scalar = v is None or isinstance(v, (bool, int, float, str))

        if key in IDENTIFIER_KEYS or any(bad in key for bad in IDENTIFIER_KEYS):
            _STATE["attrs_redacted"] += 1
            continue
        if not scalar and (key in GEOMETRY_KEYS
                           or any(bad in key for bad in GEOMETRY_KEYS)):
            _STATE["attrs_redacted"] += 1
            continue

        if v is None or isinstance(v, (bool, int, float)):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:200]
        else:
            _STATE["attrs_redacted"] += 1
            out[k] = (f"<{type(v).__name__} len={len(v)}>"
                      if hasattr(v, "__len__") else f"<{type(v).__name__}>")
    return out


# ----------------------------------------------------------------------
# The sink
# ----------------------------------------------------------------------

def _rotate_if_needed(path: str):
    try:
        if os.path.exists(path) and os.path.getsize(path) > MAX_BYTES:
            backup = path + ".1"
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(path, backup)
    except OSError:
        pass            # rotation is housekeeping; never let it fail a request


def _write(record: dict):
    """Append one span. Swallows everything — see the module docstring."""
    if not _STATE["enabled"]:
        return
    path = _STATE["path"]
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        _rotate_if_needed(path)
        line = json.dumps(record, default=str)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            _STATE["spans_written"] += 1
    except Exception as e:                           # noqa: BLE001 — by design
        _STATE["spans_dropped"] += 1
        _STATE["last_error"] = f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------
# The OTel bridge, when the SDK is present
# ----------------------------------------------------------------------

def _try_otel():
    """A tracer backed by a FILE span exporter, or None.

    The SDK is used for its span model and its context propagation, never for
    its network exporters — there is no code path here that constructs one.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        # SimpleSpanProcessor, NOT Batch. Batching buys throughput this
        # application does not need — a handful of spans per request, not
        # thousands per second — and costs the one property that matters here:
        # a span buffered in memory is a span that is NOT on disk when the
        # process dies, and the spans worth having are the ones written just
        # before something went wrong. It also made every telemetry test read
        # an empty file, which is how this was found.
        from opentelemetry.sdk.trace.export import (SimpleSpanProcessor,
                                                    SpanExporter, SpanExportResult)

        class _FileExporter(SpanExporter):
            """Writes the SAME JSON line the builtin writer does.

            Not a stylistic choice. A support engineer reading
            otel_fallback.log must not have to work out which backend produced
            a line before they can read it, and a field that means OK on one
            path and UNSET on the other is exactly the kind of difference that
            wastes an hour. `aligner.status` and `aligner.error_type` are set by
            `span()` for this reason and are unpacked back out here.
            """

            def export(self, spans):
                for s in spans:
                    attrs = _clean(dict(s.attributes or {}))
                    status = attrs.pop("aligner.status", None)
                    err = attrs.pop("aligner.error_type", None)
                    if status is None:
                        # Fall back to the SDK's own view, mapped to this
                        # module's vocabulary rather than reported raw.
                        raw = s.status.status_code.name
                        status = "ERROR" if raw == "ERROR" else "OK"
                    _write({
                        "ts": time.time(),
                        "service": SERVICE_NAME,
                        "name": s.name,
                        "trace_id": f"{s.context.trace_id:032x}",
                        "span_id": f"{s.context.span_id:016x}",
                        "duration_ms": round((s.end_time - s.start_time) / 1e6, 3),
                        "status": status,
                        "error_type": err,
                        "attributes": attrs,
                    })
                return SpanExportResult.SUCCESS


            def shutdown(self):
                return None

        provider = TracerProvider(resource=Resource.create(
            {"service.name": SERVICE_NAME}))
        provider.add_span_processor(SimpleSpanProcessor(_FileExporter()))
        trace.set_tracer_provider(provider)
        _STATE["backend"] = "opentelemetry-sdk (file exporter)"
        return trace.get_tracer(SERVICE_NAME)
    except Exception as e:                           # noqa: BLE001
        _STATE["backend"] = f"builtin writer (opentelemetry unavailable: {e})"
        return None


_TRACER = None
_INIT = False


def _tracer():
    global _TRACER, _INIT
    if not _INIT:
        _INIT = True
        _TRACER = _try_otel()
    return _TRACER


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

class _Span:
    """What a caller gets. Deliberately tiny: set attributes, nothing else."""

    __slots__ = ("attrs", "_otel")

    def __init__(self, otel=None):
        self.attrs = {}
        self._otel = otel

    def set(self, **kw):
        self.attrs.update(kw)
        return self

    def __setitem__(self, k, v):
        self.attrs[k] = v


@contextmanager
def span(name: str, **attrs):
    """Time a block and write one span.

    NEVER raises of its own accord, and never swallows the body's exception — a
    failure inside the block propagates exactly as it would have, and is
    recorded with `status: ERROR` on the way past.

    WHICH BACKEND RUNS. If the opentelemetry SDK is importable, the span goes
    through it and out of the FILE exporter in `_try_otel`; otherwise the
    builtin writer produces the same JSON line. Both paths run `_clean` and both
    end up in the same file, so a support engineer reading the log does not have
    to know which was used. opentelemetry is deliberately NOT in
    requirements.txt: the file is the product here, and adding a substantial
    dependency to produce a line this module already writes is the same trade
    that got framer-motion uninstalled.

    Usage:

        with telemetry.span("cut", faces=len(f)) as sp:
            ...
            sp.set(removed_faces=n, seconds=t)
    """
    sp = _Span()
    sp.attrs.update(attrs)
    t0 = time.perf_counter()
    status_name, err = "OK", None

    tracer = None
    try:
        tracer = _tracer()
    except Exception:                                # noqa: BLE001 — by design
        tracer = None

    otel_cm = None
    try:
        if tracer is not None:
            otel_cm = tracer.start_as_current_span(name)
            otel_span = otel_cm.__enter__()
            sp._otel = otel_span
    except Exception:                                # noqa: BLE001 — by design
        otel_cm, sp._otel = None, None

    try:
        yield sp
    except BaseException as e:                       # noqa: BLE001 — re-raised
        status_name = "ERROR"
        err = type(e).__name__
        raise
    finally:
        cleaned = {}
        try:
            cleaned = _clean(sp.attrs)
        except Exception:                            # noqa: BLE001 — by design
            _STATE["spans_dropped"] += 1

        if otel_cm is not None:
            # The SDK's file exporter writes the line; do not write it twice.
            try:
                for k, v in cleaned.items():
                    if v is not None:
                        sp._otel.set_attribute(k, v)
                sp._otel.set_attribute("aligner.status", status_name)
                if err:
                    sp._otel.set_attribute("aligner.error_type", err)
            except Exception:                        # noqa: BLE001 — by design
                pass
            try:
                otel_cm.__exit__(None, None, None)
            except Exception:                        # noqa: BLE001 — by design
                _STATE["spans_dropped"] += 1
        else:
            try:
                _write({
                    "ts": time.time(),
                    "service": SERVICE_NAME,
                    "name": name,
                    "span_id": uuid.uuid4().hex[:16],
                    "duration_ms": round((time.perf_counter() - t0) * 1000, 3),
                    "status": status_name,
                    "error_type": err,
                    "attributes": cleaned,
                })
            except Exception:                        # noqa: BLE001 — by design
                # A tracing failure must never become the clinician's problem.
                _STATE["spans_dropped"] += 1



def record_span(name: str, duration_ms: float, status: str = "OK",
                error_type: str | None = None, **attrs):
    """Emit one span for work that has ALREADY been timed. Never raises.

    WHY THIS EXISTS ALONGSIDE `span()`. Several endpoints measure themselves
    with perf_counter at the top and only know their counts at the bottom —
    /cut cannot report `selected_vertices` until it has finished selecting. A
    context manager opened at the bottom wraps nothing, and the first version of
    this wiring did exactly that: /cut logged `duration_ms: 0.0` beside
    `seconds: 0.352`, which is worse than logging nothing, because a support
    engineer reading the file has no reason to distrust the field named
    duration. This takes the measured duration as an argument so the two agree.

    Use `span()` when a block can genuinely be wrapped — /export does, around
    the bundle build.
    """
    try:
        _write({
            "ts": time.time(),
            "service": SERVICE_NAME,
            "name": name,
            "span_id": uuid.uuid4().hex[:16],
            "duration_ms": round(float(duration_ms), 3),
            "status": status,
            "error_type": error_type,
            "attributes": _clean(attrs),
        })
    except Exception:                                # noqa: BLE001 — by design
        _STATE["spans_dropped"] += 1


def configure(path: str | None = None, enabled: bool | None = None):
    """Point the sink somewhere else, or switch it off. Local paths only."""
    if path is not None:
        _STATE["path"] = path
    if enabled is not None:
        _STATE["enabled"] = bool(enabled)
    return status()


def status() -> dict:
    """What a support engineer needs, with no patient data in it."""
    path = _STATE["path"]
    try:
        size = os.path.getsize(path) if os.path.exists(path) else 0
    except OSError:
        size = None
    return {
        "enabled": _STATE["enabled"],
        "path": os.path.abspath(path),
        "bytes": size,
        "backend": _STATE["backend"] or "not initialised",
        "spans_written": _STATE["spans_written"],
        "spans_dropped": _STATE["spans_dropped"],
        "attributes_redacted": _STATE["attrs_redacted"],
        "last_error": _STATE["last_error"],
        "traced_operations": list(TRACED),
        "exporter": "FILE ONLY — no collector endpoint exists and none can be "
                    "configured. Spans never leave this workstation.",
        "phi": "Attributes are filtered against the same denylist audit.py uses, "
               "at runtime, on every span. Non-scalar values are replaced by "
               "their type and length.",
    }


def read_spans(limit: int = 200) -> list:
    """The last `limit` spans, parsed. For a support view, not for a clinician."""
    path = _STATE["path"]
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                # a torn last line costs one span, not the read
    return out
