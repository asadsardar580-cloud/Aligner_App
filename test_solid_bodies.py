"""A negative-volume part is an enclosed void, and the body count comes last.

AGENT_BRIEF defect B5. `_solid_bodies` counted the positive parts of
`decompose()` and returned that number, then unioned them and returned the
union - so `bodies` described the INPUT to the last boolean rather than the
solid that is exported. On both recorded real-scan runs this produced the
signature pair `single_positive_manifold_body` PASSED while
`body_count_agrees_with_stl` FAILED, which is not a contradiction once you
know the two numbers were measured on different solids.

The control is the brief's: `cube(10) - cube(6) + cube(2)`, all centred.
  raw decompose      [1000.0, -216.0, 8.0]   -> the old rule said 2 bodies
  after the re-union [1000.0]                -> 1 body, and the STL agrees
"""
from __future__ import annotations

import numpy as np
import pytest

import api_core
import core_geometry as cg
import manufacturing as mfg
import stl_io

m3 = pytest.importorskip("manifold3d")


def _nested_control():
    """A solid, an enclosed cavity, and a free-floating cube inside it."""
    outer = m3.Manifold.cube([10.0, 10.0, 10.0], True)
    hole = m3.Manifold.cube([6.0, 6.0, 6.0], True)
    inner = m3.Manifold.cube([2.0, 2.0, 2.0], True)
    return (outer - hole) + inner


# ---------------------------------------------------------------------------
# The control from the brief
# ---------------------------------------------------------------------------

def test_the_control_decomposes_the_way_the_brief_measured_it():
    """Pin the input, so a manifold3d change is visible as itself."""
    vols = sorted(round(float(p.volume()), 4)
                  for p in _nested_control().decompose())
    assert vols == [-216.0, 8.0, 1000.0], vols
    print(f"PASS  raw decompose volumes {vols}")


def test_bodies_are_counted_after_the_reunion():
    _solid, bodies, info = api_core._solid_bodies(_nested_control())
    assert bodies == 1, f"expected 1 exported body, got {bodies}"
    assert info["bodies_before_reunion"] == 2, info
    assert info["reunion_part_volumes_mm3"] == [1000.0], info
    print(f"PASS  {info['bodies_before_reunion']} positive parts before the "
          f"re-union, {bodies} body after it")


def test_the_void_is_counted_and_its_filled_volume_measured():
    _solid, _bodies, info = api_core._solid_bodies(_nested_control())
    assert info[api_core.KEY_INTERNAL_VOIDS] == 1, info
    assert abs(info[api_core.KEY_INTERNAL_VOIDS_FILLED] - 216.0) < 1e-6, info
    print(f"PASS  internal_voids=1, "
          f"internal_voids_filled_mm3="
          f"{info[api_core.KEY_INTERNAL_VOIDS_FILLED]}")


def test_the_old_manifest_key_survives_as_an_alias():
    """One release of compatibility, and it must not drift from the new key."""
    _solid, _bodies, info = api_core._solid_bodies(_nested_control())
    assert info[api_core.KEY_CRUMBS_ALIAS] == info[api_core.KEY_INTERNAL_VOIDS]
    assert api_core.KEY_CRUMBS_ALIAS == "inverted_crumbs_discarded"
    print("PASS  inverted_crumbs_discarded == internal_voids")


def test_the_written_stl_has_one_component_and_the_gate_agrees():
    """The whole point: the engine's count and the file's count must match.

    This is `body_count_agrees_with_stl`, the gate that failed on every real
    run while `single_positive_manifold_body` passed.
    """
    solid, bodies, _info = api_core._solid_bodies(_nested_control())
    mesh = solid.to_mesh()
    sv = np.asarray(mesh.vert_properties, float)[:, :3]
    sf = np.asarray(mesh.tri_verts, np.int64)

    blob = cg.write_binary_stl_bytes(sv, sf)
    validation = mfg.validate_printable_stl(blob)

    assert validation["connected_components"] == 1, validation
    assert bodies == validation["connected_components"], (
        f"engine says {bodies} bodies, the file says "
        f"{validation['connected_components']} components")

    gate = mfg.aggregate_print_gate({
        "manifold_bodies": bodies,
        "stl_validation": validation,
    })
    row = next(g for g in gate["gates"]
               if g["gate"] == "body_count_agrees_with_stl")
    assert row["passed"] is True, row
    print(f"PASS  bodies={bodies}, STL components="
          f"{validation['connected_components']}, "
          f"body_count_agrees_with_stl passed")


# ---------------------------------------------------------------------------
# The genuine sliver case the old name came from must still be handled
# ---------------------------------------------------------------------------

def test_a_solid_with_no_voids_reports_zero_and_still_counts_one_body():
    _solid, bodies, info = api_core._solid_bodies(
        m3.Manifold.cube([4.0, 4.0, 4.0], True))
    assert bodies == 1
    assert info[api_core.KEY_INTERNAL_VOIDS] == 0
    assert info[api_core.KEY_INTERNAL_VOIDS_FILLED] == 0.0
    print("PASS  a plain cube: 1 body, 0 voids, 0.0 mm3 filled")


def test_two_genuinely_separate_solids_still_report_two_bodies():
    """The re-union must not hide a real fracture."""
    a = m3.Manifold.cube([2.0, 2.0, 2.0], True)
    b = m3.Manifold.cube([2.0, 2.0, 2.0], True).translate([50.0, 0.0, 0.0])
    _solid, bodies, info = api_core._solid_bodies(a + b)
    assert bodies == 2, f"a real fracture was hidden: {info}"
    print(f"PASS  two separated cubes still report {bodies} bodies")


def test_an_empty_solid_reports_zero_bodies():
    _solid, bodies, info = api_core._solid_bodies(m3.Manifold())
    assert bodies == 0
    assert info["bodies_before_reunion"] == 0
    print("PASS  an empty solid reports 0 bodies")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
