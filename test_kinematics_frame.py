"""The tetherball glitch: does a flat molar pivot inside its socket?

THE BUG. derive_frame_from_region took the long axis as
normalize(crown_centroid - rim_centroid). That difference is ~8mm of clean
signal on an incisor, but on a short broad molar it collapses to ~1.5mm --
and the crown centroid drifts laterally by a comparable amount whenever the
wand under-selects one side of the tooth. The "long axis" then points
sideways, C_res is extrapolated 9-10mm along it into empty space OUTSIDE the
tooth, and the gizmo swings the crown through a wide arc off the cast.

Measured on the fixture below at the primary case: the old estimator lands
40.8 degrees off apical and puts C_res 4.50mm outside a tooth whose semi-axis
is 5mm. That is the tetherball, reproduced deterministically.

WHY TWO FIXTURES. The direct one pins exact numbers on the function's actual
contract (it takes crown_verts and rim_verts as independent arrays, so a unit
test should supply them independently -- the same idiom as
test_core_geometry.py's frame tests). The mesh one runs the real pipeline,
region grow through cap_and_close, to catch anything the synthetic arrays
would miss.
"""
import numpy as np
import core_geometry as cg
from tooth_fixture import flat_molar_on_base

CLAMP = cg.MAX_AXIS_DEVIATION_DEG
UP = np.array([0.0, 0.0, 1.0])          # the fixtures' true occlusal direction
ARCH = {"u_occ": UP, "origin": np.array([-30.0, 0.0, 0.0])}   # tooth lies +x of origin


def tilt_from_up(v):
    return cg.angle_between_deg(np.asarray(v, float), UP)


def old_axis(crown, rim):
    """The estimator this change replaces, run directly."""
    d = np.asarray(crown).mean(axis=0) - np.asarray(rim).mean(axis=0)
    return d / np.linalg.norm(d)


# =====================================================================
# Fixture A — direct arrays. Wide flat molar, full scalloped cervical
# ring, and a wand that missed the mesial strip x < cut_x.
# =====================================================================

def flat_molar(cut_x=-1.5, n_rim=240, grid=61):
    th = np.linspace(-np.pi, np.pi, n_rim, endpoint=False)
    rim = np.column_stack([5.0 * np.cos(th), 4.0 * np.sin(th), 1.2 * np.cos(2 * th)])
    g = np.linspace(-5, 5, grid)
    X, Y = np.meshgrid(g, g, indexing="ij")
    m = ((X / 5.0) ** 2 + (Y / 4.0) ** 2 <= 1.0) & (X > cut_x)
    crown = np.column_stack([X[m], Y[m], np.full(int(m.sum()), 1.6)])
    return crown, rim


MESIAL = np.array([-5.0, 0.0, 1.6])
DISTAL = np.array([5.0, 0.0, 1.6])


def test_the_bug_reproduces():
    """If this stops failing, the fixture has drifted and every assertion
    below is guarding nothing."""
    print("\n  selection      OLD tilt   OLD C_res lateral    NEW tilt")
    worst = 0.0
    for cut_x in (-5.1, -3.0, -1.5, 0.0):
        crown, rim = flat_molar(cut_x)
        old = old_axis(crown, rim)
        c_old = crown.mean(axis=0) - old * 9.0          # the old C_res formula
        lat = np.linalg.norm(c_old[:2])
        fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim, arch_frame=ARCH)
        print(f"  x > {cut_x:5.1f}   {tilt_from_up(old):6.2f} deg   "
              f"{lat:5.2f} mm             {tilt_from_up(fr['u_oa']):.3f} deg")
        if cut_x == -1.5:
            worst = tilt_from_up(old)
            assert lat > 4.0, f"C_res only {lat:.2f}mm out; fixture no longer bites"
    assert worst > 25.0, f"old estimator only {worst:.1f} deg off — bug not reproduced"
    print(f"PASS  old estimator reproduces the tetherball ({worst:.1f} deg, C_res outside the tooth)")


def test_axis_is_anatomical():
    crown, rim = flat_molar()
    fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim, arch_frame=ARCH)
    deg = tilt_from_up(fr["u_oa"])
    assert deg < 1.0, f"corrected axis is {deg:.2f} deg off apical"
    assert fr["u_oa"] @ UP > 0, "u_oa must point occlusally"
    assert fr["axis_source"] == "rim_plane", fr["axis_source"]
    print(f"PASS  corrected axis {deg:.3f} deg off apical "
          f"(source={fr['axis_source']}, planarity={fr['rim_planarity']}, "
          f"ring={fr['rim_ring_ratio']})")


def test_frame_is_right_handed_orthonormal():
    crown, rim = flat_molar()
    fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim, arch_frame=ARCH)
    for a, b in (("u_md", "u_bl"), ("u_md", "u_oa"), ("u_bl", "u_oa")):
        assert abs(np.dot(fr[a], fr[b])) < 1e-9, f"{a}/{b} not orthogonal"
    for k in ("u_md", "u_bl", "u_oa"):
        assert abs(np.linalg.norm(fr[k]) - 1) < 1e-9, f"{k} not unit length"
    det = np.linalg.det(np.column_stack([fr["u_md"], fr["u_bl"], fr["u_oa"]]))
    assert abs(det - 1) < 1e-12, f"basis not right-handed (det={det:.12f})"
    print(f"PASS  frame orthonormal to 1e-9, right-handed (det={det:.12f})")


def test_cres_sits_on_the_socket_axis():
    crown, rim = flat_molar()
    fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim, arch_frame=ARCH)
    c_res = cg.center_of_resistance(fr, 9.0)            # molar root, Wheeler
    d = c_res - fr["rim_centroid"]
    lateral = np.linalg.norm(d - (d @ fr["u_oa"]) * fr["u_oa"])
    depth = -(d @ fr["u_oa"])

    assert lateral < 1e-9, f"C_res is {lateral:.3f}mm off the socket axis"
    assert depth > 0, "C_res must be apical of the cervical margin"
    rim_radius = np.linalg.norm(rim[:, :2] - fr["rim_centroid"][:2], axis=1).max()
    assert lateral < rim_radius, "C_res escaped the socket footprint"
    print(f"PASS  C_res {depth:.2f}mm apical of the margin, lateral offset "
          f"{lateral:.1e}mm (was 4.50mm)")


def test_cres_depth_is_unchanged_for_a_clean_frame():
    """The fix removes the LATERAL error and nothing else. On a frame with no
    lateral drift the new formula must agree with the old one exactly, or the
    pivot depth has silently moved."""
    fr = dict(centroid=np.array([0., 0., 10.]), u_md=np.array([1., 0., 0.]),
              u_bl=np.array([0., 1., 0.]), u_oa=np.array([0., 0., 1.]),
              rim_centroid=np.array([0., 0., 5.]))
    new = cg.center_of_resistance(fr, 10.0)
    old = fr["centroid"] - fr["u_oa"] * 10.0
    assert np.allclose(new, old, atol=1e-12), f"depth moved: {new} vs {old}"
    print(f"PASS  clean frame: new C_res {new.round(3)} == old formula exactly")


def test_rotation_stays_within_the_arc_bound():
    """Displacement must stay inside r*theta for the farthest crown vertex --
    the signature of rotating about a pivot the tooth is actually attached to.

    Note what this does NOT claim: at 5 deg the old pivot produces a similar
    MAGNITUDE (1.02mm against 0.90mm). The tetherball was never mainly about
    magnitude at small angles. It was that the pivot sat 4.5mm outside the
    tooth and the rotation axis was 40.8 deg off, so "tip" was a diagonal
    twist that grew without bound as the clinician kept dragging. Those two
    facts are pinned by test_the_bug_reproduces and test_axis_is_anatomical;
    this one guards against a future pivot wandering off again."""
    crown, rim = flat_molar()
    fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim, arch_frame=ARCH)
    c_res = cg.center_of_resistance(fr, 9.0)

    reach = np.linalg.norm(crown - c_res, axis=1).max()
    for deg in (1.0, 5.0, 15.0, 30.0):
        M = cg.kinematic_matrix(fr, c_res, tip_deg=deg)
        swing = np.linalg.norm(cg.apply_matrix(crown, M) - crown, axis=1).max()
        bound = 2 * reach * np.sin(np.radians(deg) / 2) * 1.001   # exact chord
        assert swing <= bound, f"{deg} deg tip swung {swing:.3f}mm, past {bound:.3f}mm"
        assert np.allclose(cg.apply_matrix(np.array([c_res]), M)[0], c_res, atol=1e-9), \
            "C_res moved under pure rotation"
    print(f"PASS  tip of 1/5/15/30 deg all stay within the exact chord bound "
          f"(farthest vertex {reach:.1f}mm from C_res); C_res fixed to 1e-9")


# =====================================================================
# Reconciliation against the arch
# =====================================================================

def test_genuine_inclination_survives():
    """A tooth really tipped 12 deg keeps its 12 deg. The clamp rejects noise;
    it does not flatten every tooth onto the arch normal."""
    crown, rim = flat_molar()
    R = cg.rotation_matrix_axis_angle(np.array([0.0, 1.0, 0.0]), 12.0)
    fr = cg.derive_frame_from_region(MESIAL @ R.T, DISTAL @ R.T,
                                     crown @ R.T, rim @ R.T, arch_frame=ARCH)
    deg = tilt_from_up(fr["u_oa"])
    assert 11.0 < deg < 13.0, f"real inclination distorted to {deg:.2f} deg"
    assert fr["axis_source"] == "rim_plane", "should not have been corrected"
    assert fr["axis_corrected"] is False
    print(f"PASS  genuinely tipped tooth keeps {deg:.2f} deg of inclination, uncorrected")


def test_wild_axis_is_clamped():
    """An arch whose apical direction disagrees wildly: the axis is pulled back
    to exactly the clamp, never left where it was."""
    crown, rim = flat_molar()
    v = np.array([0.6, 0.0, 0.8]); v /= np.linalg.norm(v)
    fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim,
                                     arch_frame={"u_occ": v, "origin": np.array([-30., 0., 0.])})
    dev = cg.angle_between_deg(fr["u_oa"], v)
    assert fr["axis_source"] == "rim_plane_clamped", fr["axis_source"]
    assert fr["axis_corrected"] is True
    assert abs(dev - CLAMP) < 1e-6, f"clamp landed at {dev:.6f}, expected {CLAMP}"
    print(f"PASS  {fr['axis_deviation_deg']:.1f} deg disagreement clamped to {dev:.4f} deg")


def test_scalloped_anterior_rim_is_not_rejected():
    """The regression that killed the first design. A cos(2th) margin scallop
    drives s2/s1 to 0.5 while the plane normal stays EXACT, so any 'planarity'
    gate would reject a perfect anterior fit. There must be no such gate."""
    th = np.linspace(-np.pi, np.pi, 240, endpoint=False)
    rim = np.column_stack([4 * np.cos(th), 3 * np.sin(th), 2.0 * np.cos(2 * th)])
    crown = np.column_stack([2.5 * np.cos(th), 1.8 * np.sin(th), np.full(240, 6.0)])
    fr = cg.derive_frame_from_region(np.array([-4., 0., 6.]), np.array([4., 0., 6.]),
                                     crown, rim, arch_frame=ARCH)
    assert fr["axis_source"] == "rim_plane", \
        f"a perfect scalloped fit was rejected as {fr['axis_source']}"
    assert fr["rim_planarity"] > 0.4, "fixture should look 'non-planar' by s2/s1"
    assert tilt_from_up(fr["u_oa"]) < 0.5
    print(f"PASS  scalloped rim (s2/s1={fr['rim_planarity']:.3f}) accepted, "
          f"{tilt_from_up(fr['u_oa']):.3f} deg off — no planarity gate")


def test_sliver_rim_falls_back_to_the_arch():
    """A rim that is a LINE, not a ring. Its normal is arbitrary within a whole
    plane, so it must not be trusted -- note it happens to read 0 deg here,
    which is exactly why the gate cannot be 'is the answer plausible'."""
    th = np.linspace(-np.pi, np.pi, 240, endpoint=False)
    rim = np.column_stack([5 * np.cos(th), 0.05 * np.sin(th), 0.02 * np.cos(2 * th)])
    crown = np.column_stack([3 * np.cos(th), 0.03 * np.sin(th), np.full(240, 2.0)])
    fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim, arch_frame=ARCH)
    assert fr["axis_source"] == "arch_apical", fr["axis_source"]
    assert np.allclose(fr["u_oa"], UP), "sliver rim must adopt the arch direction"
    print(f"PASS  sliver rim (s1/s0={fr['rim_ring_ratio']:.4f}) falls back to arch apical")


def test_tiny_rim_does_not_crash():
    """fewer than 3 points: full_matrices=False makes vt (N,3), so an
    unguarded vt[2] would IndexError rather than degrade."""
    crown, rim = flat_molar()
    for n in (0, 1, 2, 5):
        fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim[:n], arch_frame=ARCH)
        assert fr["axis_source"] == "arch_apical", f"n={n}: {fr['axis_source']}"
    print("PASS  rims of 0/1/2/5 points fall back cleanly, no IndexError")


def test_inverted_axis_is_detected():
    """reconcile_tooth_frame used abs(cos), which scored an INVERTED axis as a
    flawless 0 deg while C_res sat above the crown."""
    import arch_frame as af
    good = af.reconcile_tooth_frame({"u_oa": UP}, {"u_occ": UP})
    bad = af.reconcile_tooth_frame({"u_oa": -UP}, {"u_occ": UP})
    assert good["agrees"] and not good["inverted"], good
    assert bad["inverted"] and not bad["agrees"], bad
    assert bad["angle_to_arch_occlusal_deg"] > 179.0, bad
    print(f"PASS  inverted axis reported as {bad['angle_to_arch_occlusal_deg']:.1f} deg "
          f"(was a perfect 0.0), upright as {good['angle_to_arch_occlusal_deg']:.1f} deg")


def test_u_bl_is_pinned_to_anatomy_not_click_order():
    """Clicking distal-then-mesial must not invert the buccolingual axis, or
    positive torque silently becomes lingual crown torque."""
    crown, rim = flat_molar()
    a = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim, arch_frame=ARCH)
    b = cg.derive_frame_from_region(DISTAL, MESIAL, crown, rim, arch_frame=ARCH)
    assert np.allclose(a["u_bl"], b["u_bl"], atol=1e-12), \
        f"u_bl flipped with click order: {a['u_bl']} vs {b['u_bl']}"
    assert np.allclose(a["u_oa"], b["u_oa"], atol=1e-12)
    buccal = cg.buccal_direction(a["rim_centroid"], ARCH)
    assert a["u_bl"] @ buccal > 0, "u_bl must point buccally"
    print(f"PASS  u_bl pinned buccal ({a['u_bl'].round(3)}) regardless of click order")


def test_legacy_callers_still_work():
    """app_ui.py and the older tests pass four positional args and no occlusal
    plane. That path must keep working."""
    crown, rim = flat_molar(-5.1)
    fr = cg.derive_frame_from_region(MESIAL, DISTAL, crown, rim)
    assert fr["u_oa"] @ (fr["centroid"] - fr["rim_centroid"]) > 0, "u_oa must be occlusal"
    for a, b in (("u_md", "u_bl"), ("u_md", "u_oa"), ("u_bl", "u_oa")):
        assert abs(np.dot(fr[a], fr[b])) < 1e-9
    assert fr["u_bl_points_buccal"] is None, "no arch frame means no buccal claim"
    assert np.isfinite(cg.center_of_resistance(fr, 9.0)).all()
    print(f"PASS  no-arch-frame call still valid (source={fr['axis_source']}, "
          f"{tilt_from_up(fr['u_oa']):.2f} deg off apical)")


# =====================================================================
# Fixture B — the real pipeline, region grow through cap_and_close
# =====================================================================

def test_end_to_end_on_a_real_mesh():
    verts, faces, crown_r = flat_molar_on_base()
    edges = cg.directed_edges(faces)
    conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges),
                                 faces, 2, edges=edges)
    mask, _ = cg.region_grow_crown(verts, faces, conc, np.array([0., 0., 2.2]),
                                   12.0, edges=edges)
    # truncate the selection: the wand missed the mesial strip. This is the
    # lateral drift that breaks the old estimator, and it is what the bug
    # report describes -- a molar whose selection did not wrap evenly.
    fc = verts[faces].mean(axis=1)
    mask = cg.largest_face_component(faces, mask & (fc[:, 0] > -1.5))

    (cv, cf), _ = cg.split_by_face_mask(verts, faces, mask)
    loops = cg.boundary_loops(cf)
    assert loops, "no cervical rim"
    rim = cv[np.asarray(max(loops, key=len))].copy()
    cv2, cf2 = cg.cap_and_close(cv, cf)
    assert cg.is_edge_manifold_closed(cf2), "crown not watertight"

    mesial = np.array([-crown_r, 0.0, 0.0])
    distal = np.array([crown_r, 0.0, 0.0])
    fr = cg.derive_frame_from_region(mesial, distal, cv2, rim, arch_frame=ARCH)
    c_res = cg.center_of_resistance(fr, 9.0)

    old_deg = tilt_from_up(old_axis(cv2, rim))
    new_deg = tilt_from_up(fr["u_oa"])
    d = c_res - fr["rim_centroid"]
    lateral = np.linalg.norm(d - (d @ fr["u_oa"]) * fr["u_oa"])

    assert new_deg <= CLAMP + 1e-6, f"axis {new_deg:.2f} deg past the clamp"
    assert lateral < 1e-9, f"C_res {lateral:.3f}mm off the socket axis"
    swing = np.linalg.norm(
        cg.apply_matrix(cv2, cg.kinematic_matrix(fr, c_res, tip_deg=5.0)) - cv2, axis=1).max()
    assert swing < 2.0, f"5 deg tip displaced the crown {swing:.2f}mm — tetherball"
    print(f"PASS  real mesh ({len(cf2):,} tris, {len(rim)} rim verts): "
          f"old {old_deg:.1f} deg -> new {new_deg:.2f} deg, C_res lateral "
          f"{lateral:.1e}mm, 5 deg tip swings {swing:.3f}mm")


def test_matches_the_javascript_golden_matrices():
    """The browser and the backend must produce the SAME transform.

    frontend/verify-kinematics.mjs hard-codes these matrices and checks that
    ToothGizmo.setClinical() reproduces them, so the sidebar's numeric inputs
    drive the viewport with exactly the transform that will be exported. This
    test pins the other end: if cg.kinematic_matrix ever changes convention —
    composition order, pivot handling, row/column major — one of the two suites
    goes red instead of the printed aligner quietly not matching the screen.

    Update both files together, never one alone.
    """
    frame = dict(
        u_md=np.array([0.935563726474988, 0.25837028799393946, 0.2407598554289369]),
        u_bl=np.array([-0.28340921275056274, 0.9560359997469121, 0.07532851595530501]),
        u_oa=np.array([-0.2107124387223975, -0.13870818818603006, 0.9676571224859604]),
        centroid=np.array([1.0, 2.0, 3.0]), rim_centroid=np.array([1.0, 2.0, -1.0]))
    c_res = np.array([1.5, -2.0, -7.0])

    golden = [
        (dict(tip_deg=8.0, torque_deg=0, rotation_deg=0, d_md=0, d_bl=0, d_oa=0),
         [0.9910497450693709, -0.013120564189333068, 0.1328467297049066, 0.9171113619516237,
          0.007846842123106063, 0.99916310094437, 0.04014380126402818, 0.26756254755227804,
          -0.13326225972356157, -0.038742076694265284, 0.9903232914693998, 0.054672276482610194,
          0, 0, 0, 1]),
        (dict(tip_deg=8.0, torque_deg=-5.0, rotation_deg=3.0, d_md=0.4, d_bl=-0.2, d_oa=0.3),
         [0.993606916899914, -0.04194230274913117, 0.10481477915230929, 1.0271020747414203,
          0.028206835731153637, 0.9912109636078271, 0.129248597832543, 0.7153803172391093,
          -0.10931454206409535, -0.12546580754828365, 0.9860571291915974, 0.18697547806706005,
          0, 0, 0, 1]),
        (dict(tip_deg=-12.5, torque_deg=7.5, rotation_deg=-4.0, d_md=-0.6, d_bl=0.35, d_oa=-0.25),
         [0.9857833847971419, 0.04697988817187355, -0.16131958456817322, -1.6218057434962687,
          -0.07922406915713918, 0.9766499253545972, -0.19969594430294357, -1.111468181514139,
          0.14817106709501204, 0.20963733782944718, 0.9664872070874542, -0.3975766886659393,
          0, 0, 0, 1]),
    ]

    worst = 0.0
    for kin, expected in golden:
        M = cg.kinematic_matrix(frame, c_res, **kin).ravel()
        worst = max(worst, float(np.abs(M - np.array(expected)).max()))
    assert worst < 1e-12, (
        f"cg.kinematic_matrix no longer matches the matrices "
        f"frontend/verify-kinematics.mjs pins (max diff {worst:.2e}). The browser "
        f"and the exported STL would disagree — update both files together.")
    print(f"PASS  backend matches the browser's golden matrices (max diff {worst:.2e})")


if __name__ == "__main__":
    test_the_bug_reproduces()
    test_axis_is_anatomical()
    test_frame_is_right_handed_orthonormal()
    test_cres_sits_on_the_socket_axis()
    test_cres_depth_is_unchanged_for_a_clean_frame()
    test_rotation_stays_within_the_arc_bound()
    test_genuine_inclination_survives()
    test_wild_axis_is_clamped()
    test_scalloped_anterior_rim_is_not_rejected()
    test_sliver_rim_falls_back_to_the_arch()
    test_tiny_rim_does_not_crash()
    test_inverted_axis_is_detected()
    test_u_bl_is_pinned_to_anatomy_not_click_order()
    test_legacy_callers_still_work()
    test_matches_the_javascript_golden_matrices()
    test_end_to_end_on_a_real_mesh()
    print("\nALL KINEMATICS FRAME TESTS PASSED")
