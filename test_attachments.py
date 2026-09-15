"""Parametric attachments: oriented in the tooth's frame, fused before staging.

The claim under test is the one that matters mechanically: an attachment's
ORIENTATION is its function. A vertical rectangle resists rotation because it
stands along the tooth's long axis — if it were built along world Z it would do
that only for teeth that happen to stand upright in scanner space, which is
none of them.
"""
import numpy as np

import attachments as at


def _frame(u_md, u_oa):
    u_md = np.asarray(u_md, float) / np.linalg.norm(u_md)
    u_oa = np.asarray(u_oa, float) / np.linalg.norm(u_oa)
    u_oa = u_oa - np.dot(u_oa, u_md) * u_md
    u_oa /= np.linalg.norm(u_oa)
    u_bl = np.cross(u_md, u_oa)
    return {"u_md": u_md.tolist(), "u_oa": u_oa.tolist(), "u_bl": u_bl.tolist()}


def _crown_box(half=3.0, n=6):
    """A closed box standing in for a crown, watertight by construction."""
    g = np.linspace(-half, half, n)
    v, f = at._box(2 * half, 2 * half, 2 * half)
    return v, f


def test_every_shape_builds_and_states_its_purpose():
    fr = _frame([1, 0, 0], [0, 0, 1])
    for shape in at.SHAPES:
        a = at.build_attachment(shape, fr, [0, 0, 0])
        assert len(a["verts"]) >= 8 and len(a["faces"]) >= 12
        assert a["purpose"], f"{shape} does not say what it is for"
        assert a["oriented_in"].startswith("tooth anatomical frame")
    print(f"PASS  {len(at.SHAPES)} shapes build, each stating its mechanical purpose")


def test_orientation_follows_the_TOOTH_not_the_world():
    """The load-bearing claim."""
    upright = _frame([1, 0, 0], [0, 0, 1])
    a = at.build_attachment("vertical_rectangular", upright, [0, 0, 0])
    ext_a = a["verts"].max(0) - a["verts"].min(0)
    assert abs(ext_a[2] - 3.0) < 1e-6, "on an upright tooth the 3mm axis should be world Z"

    # Same attachment, tooth tilted 45 degrees. If the block were built in world
    # axes its extent would be unchanged; it must instead follow the tooth.
    tilted = _frame([1, 0, 0], [0, 1, 1])
    b = at.build_attachment("vertical_rectangular", tilted, [0, 0, 0])
    ext_b = b["verts"].max(0) - b["verts"].min(0)
    assert abs(ext_b[2] - 3.0) > 0.1, \
        "the attachment did not rotate with the tooth — it is in world axes"
    assert abs(ext_b[0] - 2.0) < 1e-6, "the mesiodistal axis should be unchanged here"
    print(f"PASS  upright extent {np.round(ext_a,2)} vs tilted {np.round(ext_b,2)} "
          f"- the block follows u_OA")


def test_implausible_dimensions_are_refused_with_a_reason():
    fr = _frame([1, 0, 0], [0, 0, 1])
    for bad, why in (({"oa": 0.1}, "too small to survive thermoforming"),
                     ({"md": 25.0}, "a fixed appliance, not an attachment")):
        try:
            at.build_attachment("vertical_rectangular", fr, [0, 0, 0], size_mm=bad)
            raise AssertionError(f"accepted {bad} ({why})")
        except ValueError as e:
            assert "mm" in str(e) and ("thermoforming" in str(e) or "appliance" in str(e))
    try:
        at.build_attachment("banana", fr, [0, 0, 0])
        raise AssertionError("an unknown shape was accepted")
    except ValueError as e:
        assert "Available" in str(e), "the refusal should list what IS available"
    print("PASS  0.1mm, 25mm and an unknown shape all refused with actionable reasons")


def test_fusing_produces_exactly_one_positive_volume_body():
    fr = _frame([1, 0, 0], [0, 0, 1])
    cv, cf = _crown_box(3.0)
    # Seat it on the +u_bl face of the box. u_bl for this frame is (0,-1,0),
    # so the facial surface is at y = -3.
    a = at.build_attachment("vertical_rectangular", fr, [0.0, -3.0, 0.0])
    fused = at.fuse_to_crown(cv, cf, a)
    assert fused["bodies"] == 1
    crown_only = 6.0 ** 3
    assert fused["volume_mm3"] > crown_only, \
        f"fused volume {fused['volume_mm3']} is not larger than the bare crown {crown_only}"
    print(f"PASS  fused to one body, {fused['volume_mm3']}mm3 vs {crown_only:.0f}mm3 bare crown")


def test_an_attachment_floating_off_the_crown_is_refused():
    """Two bodies means it is not bonded. A stage model with a detached block
    in it would thermoform a tray with a void."""
    fr = _frame([1, 0, 0], [0, 0, 1])
    cv, cf = _crown_box(3.0)
    far = at.build_attachment("vertical_rectangular", fr, [0.0, -40.0, 0.0])
    try:
        at.fuse_to_crown(cv, cf, far)
        raise AssertionError("a floating attachment fused successfully")
    except ValueError as e:
        assert "not touching" in str(e), str(e)
    print("PASS  a detached attachment is refused, naming the cause")


def test_cbct_hooks_refuse_rather_than_return_plausible_roots():
    """A stub returning a cone from segment_roots would be the most dangerous
    thing in this repo: the app draws Wheeler cones labelled 'estimated'
    precisely so they cannot be mistaken for imaging."""
    import cbct
    req = cbct.registration_requirements()
    assert req["implemented"] is False
    assert req["requires"] and req["max_residual_mm"] > 0
    assert "never moved" in req["invariant"]

    for fn, args in ((cbct.register_volume, ("v.dcm", "sid")),
                     (cbct.segment_roots, ("v.dcm", "sid")),
                     (cbct.replace_virtual_roots, ("sid",))):
        try:
            fn(*args)
            raise AssertionError(f"{fn.__name__} returned something instead of refusing")
        except NotImplementedError as e:
            assert len(str(e)) > 40, f"{fn.__name__} refused without explaining what is missing"
    print("PASS  3 CBCT entry points refuse with what is missing; none fabricate a root")


if __name__ == "__main__":
    test_every_shape_builds_and_states_its_purpose()
    test_orientation_follows_the_TOOTH_not_the_world()
    test_implausible_dimensions_are_refused_with_a_reason()
    test_fusing_produces_exactly_one_positive_volume_body()
    test_an_attachment_floating_off_the_crown_is_refused()
    test_cbct_hooks_refuse_rather_than_return_plausible_roots()
    print("\nALL ATTACHMENT TESTS PASSED")
