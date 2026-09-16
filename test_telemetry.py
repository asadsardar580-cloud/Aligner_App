"""Telemetry: spans are emitted, PHI never is, and nothing it does can fail a cut.

THE THREE CLAIMS THIS MODULE MAKES, each asserted rather than documented:

  1. The traced operations actually emit spans with the durations and counts
     they claim to carry.
  2. No attribute can carry a patient identifier or a mesh, enforced at RUNTIME
     on every span rather than by review.
  3. A failing sink NEVER fails the operation. Tracing that can stop a tooth
     being cut is worse than no tracing at all.

The third is the one worth a test you cannot talk yourself out of: it is proved
by making every write raise and asserting the traced function still returns its
value and still raises its own exceptions unchanged.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

import audit
import telemetry


@pytest.fixture(autouse=True)
def _isolated_sink():
    """Every test writes to its own file. The default path is scratch/, and a
    test that appended there would make the next run's assertions depend on the
    previous one's."""
    d = tempfile.mkdtemp()
    telemetry.configure(path=os.path.join(d, "otel.log"), enabled=True)
    telemetry._STATE.update(spans_written=0, spans_dropped=0,
                            attrs_redacted=0, last_error=None)
    try:
        yield d
    finally:
        telemetry.configure(path=telemetry.DEFAULT_PATH, enabled=True)
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# 1. Spans are emitted, and carry what they claim to.
# --------------------------------------------------------------------------

def test_a_span_records_duration_status_and_attributes():
    with telemetry.span("cut", arch="lower") as sp:
        sp.set(selected_vertices=1528, arch_faces=187625, seconds=2.85)

    rows = telemetry.read_spans()
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "cut"
    assert r["status"] == "OK"
    assert r["duration_ms"] >= 0
    assert r["attributes"]["arch_faces"] == 187625, \
        "a FACE COUNT is the measurement the span exists for and must survive"
    assert r["attributes"]["selected_vertices"] == 1528
    print(f"PASS  span 'cut' recorded {r['duration_ms']}ms and "
          f"{len(r['attributes'])} attributes")


def test_a_failing_block_is_recorded_as_ERROR_and_still_raises():
    """The span must not swallow the body's exception — that would turn a
    refused export into a silent success."""
    with pytest.raises(ValueError, match="stage 9 refused"):
        with telemetry.span("staging", stages=31):
            raise ValueError("stage 9 refused")

    r = telemetry.read_spans()[0]
    assert r["status"] == "ERROR"
    assert r["error_type"] == "ValueError"
    assert r["attributes"]["stages"] == 31
    print(f"PASS  a raising block is ERROR/{r['error_type']} and the exception "
          f"reached the caller")


def test_every_named_operation_is_traceable():
    for name in telemetry.TRACED:
        with telemetry.span(name):
            pass
    got = [r["name"] for r in telemetry.read_spans()]
    assert sorted(got) == sorted(telemetry.TRACED)
    print(f"PASS  all {len(telemetry.TRACED)} traced operations emit: "
          f"{', '.join(telemetry.TRACED)}")


# --------------------------------------------------------------------------
# 2. The PHI denylist holds, at runtime, on real payloads.
# --------------------------------------------------------------------------

def test_identifiers_are_dropped_whatever_their_type():
    """A name is an ordinary scalar string. No shape check will ever flag it,
    which is why identifiers need a key denylist of their own."""
    with telemetry.span("segment", patient_name="Jane Doe", dob="1990-01-01",
                        mrn=884213, filename="mrs_smith_lower.stl",
                        scan_hash="d13175e7b9baaf79", teeth=11) as sp:
        sp.set(case_label="Smith, J — crowding")

    attrs = telemetry.read_spans()[0]["attributes"]
    for leaked in ("patient_name", "dob", "mrn", "filename", "scan_hash",
                   "case_label"):
        assert leaked not in attrs, f"{leaked} reached a span"
    for value in ("Jane", "Smith", "1990", "884213", "d13175e7"):
        assert value not in json.dumps(attrs), f"{value!r} reached a span"
    assert attrs["teeth"] == 11, "a harmless count was dropped with the rest"
    print(f"PASS  6 identifiers dropped, the count survived: {attrs}")


def test_geometry_cannot_reach_a_span_under_ANY_key_name():
    """The filter that catches what a denylist misses.

    `attrs={"debug": verts.tolist()}` is exactly how a leak gets written by
    somebody in a hurry, and no denylist of key names will ever contain "debug".
    """
    verts = np.random.default_rng(0).normal(size=(94848, 3)).tolist()
    with telemetry.span("cut", debug=verts, notes=verts, whatever=verts):
        pass

    raw = json.dumps(telemetry.read_spans()[0])
    assert len(raw) < 2000, f"a span carrying 94,848 vertices was {len(raw)} bytes"
    attrs = telemetry.read_spans()[0]["attributes"]
    for k in ("debug", "notes", "whatever"):
        assert attrs[k] == "<list len=94848>", \
            f"{k} kept its contents instead of its shape: {attrs[k][:60]}"
    print(f"PASS  94,848 vertices under three innocent key names -> "
          f"{attrs['debug']}, whole span {len(raw)} bytes")


def test_the_denylist_is_derived_from_audits_and_cannot_drift():
    """Two copies of a denylist is one denylist plus a bug waiting to happen."""
    assert audit.FORBIDDEN_KEYS <= telemetry.FORBIDDEN_KEYS, \
        "telemetry accepts a key audit.py refuses"
    # ...but the SPLIT is real and intentional: `faces` is geometry in an audit
    # entry and a scalar count in a span.
    assert "faces" in audit.FORBIDDEN_KEYS
    assert "faces" in telemetry.GEOMETRY_KEYS
    assert "faces" not in telemetry.IDENTIFIER_KEYS
    with telemetry.span("cut", faces=187625):
        pass
    assert telemetry.read_spans()[0]["attributes"]["faces"] == 187625
    print("PASS  telemetry's denylist is a superset of audit's, split by kind")


def test_a_long_string_is_truncated_not_stored_whole():
    with telemetry.span("export", detail="x" * 50_000):
        pass
    assert len(telemetry.read_spans()[0]["attributes"]["detail"]) == 200
    print("PASS  a 50,000-character attribute is truncated to 200")


# --------------------------------------------------------------------------
# 3. Nothing telemetry does can fail a clinical operation.
# --------------------------------------------------------------------------

def _break_the_sink(monkeypatch, exc=OSError("disk full")):
    real = open

    def boom(path, *a, **kw):
        if str(path).endswith(".log"):
            raise exc
        return real(path, *a, **kw)
    monkeypatch.setattr("builtins.open", boom)


def test_an_unwritable_sink_never_fails_the_operation(monkeypatch):
    _break_the_sink(monkeypatch)

    def cut_a_tooth():
        with telemetry.span("cut", faces=187625) as sp:
            sp.set(removed_faces=2906)
            return "crown extracted"

    assert cut_a_tooth() == "crown extracted"
    monkeypatch.undo()
    st = telemetry.status()
    assert st["spans_dropped"] >= 1, "a failed write was not counted"
    assert "disk full" in (st["last_error"] or ""), \
        "the failure must be visible in status, not merely swallowed"
    print(f"PASS  sink raising OSError: operation returned normally, "
          f"{st['spans_dropped']} span(s) dropped, last_error recorded")


def test_a_broken_sink_does_not_mask_the_bodys_own_exception(monkeypatch):
    """The nastier failure: tracing that eats a refusal. An export that should
    have refused must still refuse when the log is unwritable."""
    _break_the_sink(monkeypatch)
    with pytest.raises(RuntimeError, match="stage 9 refused"):
        with telemetry.span("staging"):
            raise RuntimeError("stage 9 refused")
    print("PASS  a refusal still propagates through a broken sink")


def test_disabling_the_sink_is_silent_and_total():
    telemetry.configure(enabled=False)
    with telemetry.span("cut", faces=1):
        pass
    assert telemetry.read_spans() == []
    telemetry.configure(enabled=True)
    print("PASS  disabled telemetry writes nothing at all")


# --------------------------------------------------------------------------
# The exporter is local, and that is a property of the code, not a promise.
# --------------------------------------------------------------------------

def test_there_is_no_network_exporter_anywhere_in_the_module():
    """A collector endpoint is a patient-data egress path that an environment
    variable can enable. The absence has to be checkable, not stated.

    THIS READS THE CODE, NOT THE PROSE. The first version of this test grepped
    the file for "OTLPSpanExporter" and "http://" — and failed, because
    telemetry.py's own docstring EXPLAINS at length why those must not appear.
    A test that cannot tell an explanation from an implementation will either be
    deleted or will force the explanation out, and the explanation is the more
    valuable of the two. So the module is parsed and only executable nodes are
    examined: imports, names, calls, and string literals that survive into code.
    """
    import ast
    tree = ast.parse(open(telemetry.__file__, encoding="utf-8").read())

    banned = {"OTLPSpanExporter", "JaegerExporter", "ZipkinExporter",
              "OTEL_EXPORTER_OTLP_ENDPOINT"}
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docstrings.add(d)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in banned, f"telemetry.py references {node.id}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in banned, f"telemetry.py references .{node.attr}"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            assert "otlp" not in mod.lower(), f"telemetry.py imports {mod}"
            for a in node.names:
                assert a.name not in banned, f"telemetry.py imports {a.name}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue                       # prose, not an endpoint
            low = node.value.lower()
            assert not low.startswith(("http://", "https://")),                 f"telemetry.py contains a URL literal: {node.value[:60]!r}"
            assert node.value not in banned,                 f"telemetry.py contains the literal {node.value!r}"

    st = telemetry.status()
    assert "FILE ONLY" in st["exporter"]
    assert st["path"] == os.path.abspath(st["path"]), "the sink must be a local path"
    print(f"PASS  no network exporter in telemetry.py's CODE (prose excluded); "
          f"sink is {os.path.basename(st['path'])}")



def test_rotation_keeps_the_file_bounded():
    """An unbounded log on a clinical workstation is a disk incident, and one
    nobody prunes is one nobody reads — the same reason audit.py is bounded."""
    telemetry.MAX_BYTES, real_max = 2000, telemetry.MAX_BYTES
    try:
        for i in range(400):
            with telemetry.span("cut", i=i, pad="y" * 150):
                pass
        path = telemetry._STATE["path"]
        assert os.path.getsize(path) <= 4000, "the live file grew unbounded"
        assert os.path.exists(path + ".1"), "nothing was rotated out"
        print(f"PASS  rotated at {telemetry.MAX_BYTES}B: live file "
              f"{os.path.getsize(path)}B plus one .1 backup")
    finally:
        telemetry.MAX_BYTES = real_max


if __name__ == "__main__":
    import contextlib

    class _MP:
        """A monkeypatch stand-in, so this file runs under run_all_tests.py too."""

        def __init__(self):
            self._undo = []

        def setattr(self, target, value):
            mod, _, name = target.rpartition(".")
            import builtins
            obj = builtins if mod == "builtins" else __import__(mod)
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)
            self._undo.clear()

    @contextlib.contextmanager
    def sink():
        d = tempfile.mkdtemp()
        telemetry.configure(path=os.path.join(d, "otel.log"), enabled=True)
        telemetry._STATE.update(spans_written=0, spans_dropped=0,
                                attrs_redacted=0, last_error=None)
        try:
            yield d
        finally:
            shutil.rmtree(d, ignore_errors=True)

    simple = [test_a_span_records_duration_status_and_attributes,
              test_a_failing_block_is_recorded_as_ERROR_and_still_raises,
              test_every_named_operation_is_traceable,
              test_identifiers_are_dropped_whatever_their_type,
              test_geometry_cannot_reach_a_span_under_ANY_key_name,
              test_the_denylist_is_derived_from_audits_and_cannot_drift,
              test_a_long_string_is_truncated_not_stored_whole,
              test_disabling_the_sink_is_silent_and_total,
              test_there_is_no_network_exporter_anywhere_in_the_module,
              test_rotation_keeps_the_file_bounded]
    for fn in simple:
        with sink():
            fn()

    for fn in (test_an_unwritable_sink_never_fails_the_operation,
               test_a_broken_sink_does_not_mask_the_bodys_own_exception):
        with sink():
            mp = _MP()
            try:
                fn(mp)
            finally:
                mp.undo()

    print("\nALL TELEMETRY TESTS PASSED")
