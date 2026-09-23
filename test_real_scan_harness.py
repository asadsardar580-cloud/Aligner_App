"""The real-scan harness must be able to FAIL.

WHY THIS FILE EXISTS. `real_scan_regression.py` is the only thing in this
project that drives real anatomy end to end, and until Phase 0 it could not
report a failure: it printed the failed manufacturing gates as a NOTE and
returned 0, because `ok` was only ever cleared by open or non-manifold edges.
A stage that was topologically clean and failed eight of sixteen aggregate
gates exited 0 and printed the success banner.

That is the exact shape of defect this project has corrected three times
already (CLAUDE.md s.20.6 audit denylist, s.26.7 edge forensics, s.26 the
`collar_has_ledge` key): a measurement that cannot fail is indistinguishable
from a measurement that passed.

`exit_code_for` is extracted as a PURE function precisely so it can be tested
without the scan, without a model, and without 145 seconds of CSG.
"""
from __future__ import annotations

import manufacturing as mfg
import real_scan_regression as rsr

READY = rsr.VERDICT_READY
NOT_READY = rsr.VERDICT_NOT_READY


def _gate(verdict, failed=()):
    """A stage-gate record shaped like `mfg.aggregate_print_gate`'s output."""
    return {"verdict": verdict,
            "print_ready": verdict == READY,
            "failed_gates": list(failed)}


# ---------------------------------------------------------------------------
# No expectation: every stage must be PRINT READY
# ---------------------------------------------------------------------------

def test_all_ready_exits_zero():
    assert rsr.exit_code_for([_gate(READY), _gate(READY)]) == 0
    print("PASS  two ready stages -> 0")


def test_any_not_ready_exits_one():
    gates = [_gate(READY), _gate(NOT_READY, ["old_site_restored"])]
    assert rsr.exit_code_for(gates) == 1
    print("PASS  one NOT PRINT READY stage among ready ones -> 1")


def test_every_stage_not_ready_exits_one():
    assert rsr.exit_code_for([_gate(NOT_READY), _gate(NOT_READY)]) == 1
    print("PASS  all stages NOT PRINT READY -> 1")


def test_an_empty_run_is_a_FAILURE_not_a_pass():
    """A run that produced no stage measured nothing.

    "Nothing to check" must never read as "clean" - that is how a harness
    reports success for a regression nobody ran.
    """
    assert rsr.exit_code_for([]) == 1
    assert rsr.exit_code_for(None) == 1
    print("PASS  an empty stage list -> 1, in both spellings")


# ---------------------------------------------------------------------------
# With an expectation: a mismatch in EITHER direction fails
# ---------------------------------------------------------------------------

def test_expectation_met_exits_zero():
    gates = [_gate(NOT_READY, ["no_self_touching_boundary"]), _gate(NOT_READY)]
    assert rsr.exit_code_for(gates, expect=NOT_READY) == 0
    print("PASS  every stage matches the pinned expectation -> 0")


def test_unexpectedly_ready_is_ALSO_a_mismatch():
    """An improvement is a mismatch too, and that is deliberate.

    The suite pins today's KNOWN state of the collar path. If a change makes a
    stage print-ready, the pinned expectation is now wrong and must be updated
    deliberately (Phase 3), not absorbed silently.
    """
    gates = [_gate(NOT_READY), _gate(READY)]
    assert rsr.exit_code_for(gates, expect=NOT_READY) == 1
    print("PASS  a stage that unexpectedly became PRINT READY -> 1")


def test_expecting_ready_and_getting_not_ready_exits_one():
    gates = [_gate(READY), _gate(NOT_READY, ["body_count_agrees_with_stl"])]
    assert rsr.exit_code_for(gates, expect=READY) == 1
    print("PASS  expecting PRINT READY and getting a refusal -> 1")


def test_an_empty_run_fails_even_with_an_expectation():
    assert rsr.exit_code_for([], expect=NOT_READY) == 1
    print("PASS  an empty stage list -> 1 even when an expectation is pinned")


def test_a_gate_without_a_verdict_falls_back_to_print_ready():
    """Defensive: an older manifest carried `print_ready` and no `verdict`."""
    assert rsr.exit_code_for([{"print_ready": True}]) == 0
    assert rsr.exit_code_for([{"print_ready": False}]) == 1
    print("PASS  a verdict-less gate is read from `print_ready`")


# ---------------------------------------------------------------------------
# Producer / consumer key pinning  (AGENT_BRIEF A2 rule 13)
# ---------------------------------------------------------------------------

def test_the_ledge_key_the_harness_reads_is_the_one_the_producer_writes():
    """The `collar_has_ledge` class of bug, made impossible.

    The harness read `collar_has_ledge` while `transition_quality` wrote
    `looks_like_a_ledge`, so the real-scan report printed `ledge None` on
    every run regardless of the geometry - and `None` is also the honest value
    for "the profile could not be sampled", so the blind column was
    indistinguishable from a legitimate unmeasurable.
    """
    import inspect

    assert mfg.KEY_LOOKS_LIKE_A_LEDGE == "looks_like_a_ledge"

    # The producer emits exactly this key, on both of its paths.
    src = inspect.getsource(mfg.transition_quality)
    assert "KEY_LOOKS_LIKE_A_LEDGE" in src, \
        "transition_quality no longer emits the key through the constant"
    assert '"looks_like_a_ledge"' not in src, \
        "a literal spelling has crept back into the producer"

    # The consumer reads it through the same constant, not a literal.
    consumer = inspect.getsource(rsr)
    assert "mfg.KEY_LOOKS_LIKE_A_LEDGE" in consumer, \
        "the harness is not reading the producer's constant"
    assert "collar_has_ledge" not in consumer, \
        "the dead key is back in the harness"
    print(f"PASS  producer and consumer agree on {mfg.KEY_LOOKS_LIKE_A_LEDGE!r}")


def test_transition_quality_really_emits_that_key():
    """Executed, not just grepped - the constant must survive a real call.

    A rim placed far from the only triangle in the mesh means no ray along the
    long axis can land, which is the genuine "cannot be measured" branch.
    """
    import numpy as np
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], np.int64)
    rim = np.array([[500.0, 0.0, 0.0], [501.0, 0.0, 0.0], [500.5, 1.0, 0.0]])

    out = mfg.transition_quality(verts, faces, rim,
                                 np.array([0.0, 0.0, 1.0]))

    assert mfg.KEY_LOOKS_LIKE_A_LEDGE in out, sorted(out)
    assert out[mfg.KEY_LOOKS_LIKE_A_LEDGE] is None, \
        "an unmeasurable transition must report None, never False"
    assert out["measured"] is False
    print(f"PASS  an unmeasurable transition reports "
          f"{mfg.KEY_LOOKS_LIKE_A_LEDGE}=None, reason: "
          f"{out.get('reason', '')[:60]}")


# ---------------------------------------------------------------------------
# Pinning the PROCESS exit code, for a refusal that never reaches a gate
# ---------------------------------------------------------------------------

def test_no_expectation_passes_the_natural_code_through():
    for code in (0, 1, 2, 3, 77):
        assert rsr.exit_code_for_process(code) == code
    print("PASS  without --expect-exit the natural code is untouched")


def test_a_matching_exit_code_passes():
    assert rsr.exit_code_for_process(3, 3) == 0
    print("PASS  natural 3 against expect-exit 3 -> 0")


def test_a_mismatch_in_EITHER_direction_fails():
    """Including an improvement. 0 means the run started succeeding, which
    invalidates the pinned expectation and must be looked at, not absorbed."""
    for natural in (0, 1, 2, 4):
        assert rsr.exit_code_for_process(natural, 3) == 1, natural
    print("PASS  0, 1, 2 and 4 against expect-exit 3 all -> 1")


def test_SKIP_survives_the_pin():
    """77 means the scan is not on this machine, which is normal.

    Mapping it to a mismatch would turn "not verified here" into a failure
    and destroy the distinction the suite's SKIP state exists to preserve.
    """
    import run_all_tests
    assert rsr.SKIP_EXIT_CODE == run_all_tests.SKIP_EXIT_CODE == 77
    assert rsr.exit_code_for_process(77, 3) == 77
    assert rsr.exit_code_for_process(77, 0) == 77
    print("PASS  an absent scan still SKIPs, whatever exit code is pinned")


def test_the_suite_entry_pins_the_measured_exit_code():
    import run_all_tests
    row = next(r for r in run_all_tests.TESTS
               if run_all_tests._entry(r)[1] == "real_scan_regression.py")
    _n, _p, args = run_all_tests._entry(row)
    assert "--expect-exit" in args, args
    assert args[args.index("--expect-exit") + 1] == "3", args
    print(f"PASS  real-scan entry pins: {args}")


# ---------------------------------------------------------------------------
# An absent scan is a SKIP signal, not a pass and not a failure
# ---------------------------------------------------------------------------

def test_an_absent_scan_exits_77_not_0(monkeypatch, capsys):
    """Scans are excluded from version control, so absence is normal.

    It must NOT be 0: a regression nobody ran is the exact thing this suite
    exists to stop reporting as a pass. 77 is `run_all_tests.SKIP_EXIT_CODE`,
    printed as "SKIP (NOT VERIFIED)".
    """
    import run_all_tests

    monkeypatch.setattr(rsr.os.path, "exists", lambda p: False)
    monkeypatch.setattr(rsr.sys, "argv", ["real_scan_regression.py"])

    code = rsr.main()
    out = capsys.readouterr().out

    assert code == run_all_tests.SKIP_EXIT_CODE == 77, code
    assert "NOT EXECUTED" in out
    assert "nothing here is fabricated" in out.replace("\n", " ")
    print(f"PASS  an absent scan exits {code} (SKIP), not 0")


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
