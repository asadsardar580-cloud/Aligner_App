"""`real_scan_print_v3.py` - the step-6 runner - proven on a SYNTHETIC arch.

WHY THIS EXISTS. The runner's job is the real mandible, and `case_lower.stl`
is deliberately not in version control, so on any machine without it the
runner exits 77 and has verified nothing. A runner that has never run is not
evidence; this drives the SAME `main()` end to end - upload, occlusal plane,
segmentation, select, cut, prescription, /export/stages, the direction
checks, the table - on a synthetic lower arch, so the harness itself is known
to work before its first real-scan run.

WHAT IS SYNTHETIC, stated so nobody mistakes this for the real result:
  * the arch: `horseshoe_shell` with nine narrow teeth labelled 45..34 and a
    ~2 mm space distal of 43, so case c has room to move into;
  * the segmentation: a provider registered FOR THIS TEST ONLY that labels
    by position (distance to each crown's apex). The trained models need
    checkpoints that are not in the repository;
  * the voxel: 0.1 mm, for speed - stated in every manifest written.

The direction checks are the part worth pinning: each case must come out
"OK", and a deliberately WRONG-signed prescription must stop the run with
exit 4 rather than be silently corrected.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import core_geometry as cg
import print_solid as ps
import real_scan_print_v3 as rs
import segmentation_providers as sp
from test_cast_base import horseshoe_shell

TEETH = {45: -0.47, 44: -0.36, 43: -0.235, 42: -0.135, 41: -0.045,
         31: 0.045, 32: 0.135, 33: 0.235, 34: 0.33}
LABEL_RADIUS_MM = 3.2


def _arch():
    v, f, ap = horseshoe_shell(n_s=480, n_t=60, teeth=tuple(TEETH.values()),
                               return_apices=True, tooth_w=0.035)
    return v, f, {fdi: v[a] for fdi, a in zip(TEETH, ap)}


class _PositionProvider(sp.SegmentationProvider):
    """Labels a vertex with the FDI of the apex it is within
    LABEL_RADIUS_MM of. Test-only; never registered outside this module."""

    name = "synthetic_positions"

    def __init__(self, apices):
        self.apices = apices

    def available(self):
        return {"available": True, "reason": None}

    def segment(self, verts, faces, jaw):
        v = np.asarray(verts, float)
        lab = np.zeros(len(v), np.int64)
        best = np.full(len(v), np.inf)
        for fdi, a in self.apices.items():
            d = np.linalg.norm(v - a, axis=1)
            take = (d < LABEL_RADIUS_MM) & (d < best)
            lab[take], best[take] = fdi, d[take]
        return sp.ProviderResult(labels=lab, provider=self.name, jaw=jaw,
                                 transfer={"identity": True}, meta={})


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    v, f, apices = _arch()
    path = tmp_path_factory.mktemp("scan") / "synthetic_lower.stl"
    path.write_bytes(cg.write_binary_stl_bytes(v, f))
    prov = _PositionProvider(apices)
    sp._REGISTRY[prov.name] = prov
    mp = pytest.MonkeyPatch()
    mp.setattr(ps, "VOXEL_SIZE_MM", 0.1)
    yield str(path)
    mp.undo()
    sp._REGISTRY.pop(prov.name, None)


def test_neighbours_mesial_is_toward_the_midline():
    present = set(TEETH)
    assert rs.mesial_distal_neighbours(43, present) == (42, 44)
    assert rs.mesial_distal_neighbours(41, present) == (31, 42)
    assert rs.mesial_distal_neighbours(45, present) == (44, None)
    assert rs.mesial_distal_neighbours(33, present) == (32, 34)
    assert rs.mesial_distal_neighbours(43, present - {44}) == (42, 45)


def test_contact_clicks_fall_back_to_the_far_end_not_a_guess():
    crown = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    m, d = rs.contact_clicks(crown, np.array([-5.0, 0, 0]), np.array([9.0, 0, 0]))
    assert m[0] == 0 and d[0] == 3
    m, d = rs.contact_clicks(crown, np.array([-5.0, 0, 0]), None)   # last tooth
    assert m[0] == 0 and d[0] == 3
    m, d = rs.contact_clicks(crown, None, np.array([9.0, 0, 0]))
    assert m[0] == 0 and d[0] == 3
    assert rs.contact_clicks(crown, None, None) == (None, None)


def test_the_scan_absent_is_a_skip_not_a_pass(tmp_path):
    code = rs.main(["--scan", str(tmp_path / "nope.stl"),
                    "--out", str(tmp_path)])
    assert code == rs.SKIP_EXIT_CODE


def test_all_four_cases_run_end_to_end_on_the_synthetic_arch(synthetic, tmp_path):
    out = str(tmp_path / "real_v3")
    code = rs.main(["--scan", synthetic, "--out", out,
                    "--provider", "synthetic_positions"])
    rows = json.load(open(os.path.join(out, "summary.json")))
    by = {r["case"]: r for r in rows}
    assert set(by) == set("abcd"), [r.get("status") for r in rows]
    for k, r in by.items():
        assert r["status"] == "gated", (k, r["status"])
        assert os.path.exists(os.path.join(out, k, "manifest.json"))
        assert any(n.endswith(".stl") for n in r["files"]), r["files"]
    assert by["a"]["built_stages"] == [0]
    assert by["b"]["built_stages"] == [1]
    assert by["c"]["built_stages"] == [1, 2, 3, 4]      # 1.0 mm / 0.25 mm
    assert by["d"]["built_stages"] == [1, 2]            # 0.5 mm / 0.25 mm
    for k in "bcd":
        assert by[k]["direction"] == "OK", (k, by[k]["direction_detail"])
    c = by["c"]["direction_detail"]
    assert c["distal_gap_shrank_by_mm"] == pytest.approx(1.0, abs=0.35), c
    assert c["gap_to_mesial_neighbour_after_mm"] > \
        c["gap_to_mesial_neighbour_before_mm"]
    print("\n" + rs.table(rows))
    print(f"exit {code}; verdicts: " +
          json.dumps({k: r["verdict"] for k, r in by.items()}))


def test_a_wrong_way_movement_stops_the_run(synthetic, tmp_path, monkeypatch):
    """The control for the direction checks. 'Labial' is prescribed as
    -0.5 mm - the wrong sign - and the run must STOP with exit 4, naming it,
    not flip the sign and carry on."""
    monkeypatch.setitem(rs.CASES, "d", dict(rs.CASES["d"], move=("d_bl", -0.5)))
    out = str(tmp_path / "wrong")
    code = rs.main(["--scan", synthetic, "--out", out, "--cases", "dc",
                    "--provider", "synthetic_positions"])
    assert code == 4
    rows = json.load(open(os.path.join(out, "summary.json")))
    assert len(rows) == 1, "the run carried on after a wrong-way movement"
    assert "WRONG WAY" in rows[0]["status"], rows[0]["status"]
    print(f"PASS  exit 4: {rows[0]['status'][:120]}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
