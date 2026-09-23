"""The canonical runner must not be able to report a pass for nothing.

WHAT THIS PROTECTS. `run_all_tests.py` decided PASS from an exit code alone,
and a missing entry file printed SKIP while the suite still reported success.
That gave three separate ways to report a pass for something that never ran:

  * a missing file      -> SKIP, suite green
  * exit code 0         -> PASS, including a file that asserted nothing
  * a script that printed its own failures and returned 0 anyway

The third is not hypothetical: `real_scan_regression.py` did exactly that
until Phase 0.1. So the runner's own gates get a control that makes each of
them fail, per AGENT_BRIEF A2 rule 12.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import run_all_tests as R


def _tmp(name, src):
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / name
    p.write_text(src, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# The static "can this entry fail?" guard
# ---------------------------------------------------------------------------

def test_a_pytest_file_with_no_main_block_is_TOOTHLESS():
    """The control. Run as a script this collects nothing and exits 0."""
    p = _tmp("toothless.py", "import os\n\n\ndef test_x():\n    assert False\n")
    ok, reason = R.can_fail(p)
    assert ok is False, "a file that cannot assert anything was accepted"
    assert "no test" in reason
    print(f"PASS  toothless file rejected: {reason[:60]}...")


def test_a_module_level_assert_counts():
    ok, reason = R.can_fail(_tmp("toothy.py", "assert 1 == 1\n"))
    assert ok is True and reason == "module-level assert"
    print("PASS  a top-level assert counts")


def test_an_assert_inside_a_TOP_LEVEL_LOOP_counts():
    """`test_precompute.py` is shaped exactly like this and is a real test.

    A first version of the guard only looked at `tree.body` and rejected it -
    a false positive that would have removed a working entry from the suite.
    """
    ok, reason = R.can_fail(
        _tmp("looped.py", "for i in [1, 2]:\n    assert i > 0\n"))
    assert ok is True and reason == "module-level assert"
    print("PASS  an assert inside a top-level for-loop counts")


def test_an_assert_only_inside_a_def_does_NOT_count():
    """pytest would call it; running the file as a script would not."""
    ok, _ = R.can_fail(
        _tmp("deffed.py", "def f():\n    assert False\n"))
    assert ok is False
    print("PASS  an assert reachable only through a function does not count")


def test_a_main_block_counts_even_without_a_module_level_assert():
    ok, reason = R.can_fail(_tmp(
        "mainblock.py",
        "def test_x():\n    assert False\n\n\n"
        "if __name__ == '__main__':\n    pass\n"))
    assert ok is True and reason == "__main__ block"
    print("PASS  a __main__ block counts")


def test_an_unparseable_file_is_rejected_not_skipped():
    ok, reason = R.can_fail(_tmp("broken.py", "def (\n"))
    assert ok is False and "cannot parse" in reason
    print(f"PASS  unparseable file rejected: {reason[:50]}...")


# ---------------------------------------------------------------------------
# The entry table itself
# ---------------------------------------------------------------------------

def test_every_entry_exists_and_can_fail():
    """The guard, applied to the real table. No entry may be toothless."""
    problems = []
    for row in R.TESTS:
        name, path, _args = R._entry(row)
        if not os.path.exists(path):
            problems.append(f"{name}: {path} is missing")
            continue
        ok, reason = R.can_fail(path)
        if not ok:
            problems.append(f"{name}: {reason}")
    assert not problems, "\n".join(problems)
    print(f"PASS  all {len(R.TESTS)} entries exist and can fail")


def test_the_three_phase_0_entries_are_registered():
    paths = {R._entry(row)[1] for row in R.TESTS}
    for required in ("test_self_intersection.py",
                     "test_deform_construction.py",
                     "real_scan_regression.py"):
        assert required in paths, f"{required} is not a suite entry"
    print("PASS  self-intersection, deformation construction and the real "
          "scan are all registered")


def test_the_real_scan_entry_pins_todays_known_state():
    """The expectation is the test. It must be present and explicit.

    PINNED AT THE PROCESS LEVEL, not the verdict level, because the collar
    path refuses before a stage is ever built, so there is no per-stage
    verdict for `--expect` to compare. Measured 2026-09-23: `--profile smoke`
    exits 3 with `interface_construction_failed`. With the pin the suite
    stays green on that, and any change in either direction turns it red.
    """
    row = next(r for r in R.TESTS
               if R._entry(r)[1] == "real_scan_regression.py")
    _name, _path, args = R._entry(row)
    assert "--expect-exit" in args, args
    assert args[args.index("--expect-exit") + 1] == "3", args
    assert "--profile" in args and "smoke" in args, args
    print(f"PASS  real-scan entry pins: {args}")


def test_per_entry_arguments_are_supported():
    assert R._entry(("n", "p.py")) == ("n", "p.py", [])
    assert R._entry(("n", "p.py", ["--x", "1"])) == ("n", "p.py", ["--x", "1"])
    print("PASS  2- and 3-tuple entries both parse")


def test_the_skip_code_is_77_and_is_not_zero():
    """SKIP must be a deliberate signal, never the default success code."""
    assert R.SKIP_EXIT_CODE == 77
    assert R.SKIP_EXIT_CODE != 0
    print("PASS  SKIP_EXIT_CODE is 77")


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
