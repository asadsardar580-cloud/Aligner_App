#!/usr/bin/env python3
"""
Test suite for core_geometry.py.

Synthetic fixture: a tube with a SINUSOIDAL groove running around it. The
groove stands in for the cervical sulcus, and its height varies with angle
so that "follow the groove" and "take the short way round" are genuinely
different paths. That distinction is what makes the magnetic-scissors test
discriminating rather than decorative.
"""

import numpy as np
import core_geometry as cg


# ---------------------------------------------------------------- fixture
def grooved_tube(n_theta=120, n_z=48, R=5.0, height=12.0,
                 groove_depth=1.2, groove_amp=1.8, sigma=0.7):
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(0, height, n_z)
    groove_z = height * 0.5 + groove_amp * np.sin(thetas)

    verts = np.zeros((n_theta * n_z, 3))
    for i, th in enumerate(thetas):
        for j, z in enumerate(zs):
            r = R - groove_depth * np.exp(-((z - groove_z[i]) ** 2) / (2 * sigma ** 2))
            verts[i * n_z + j] = (r * np.cos(th), r * np.sin(th), z)

    faces = []
    for i in range(n_theta):
        ii = (i + 1) % n_theta
        for j in range(n_z - 1):
            a, b = i * n_z + j, i * n_z + j + 1
            c, d = ii * n_z + j, ii * n_z + j + 1
            faces.append([a, c, b])
            faces.append([b, c, d])
    return verts, np.array(faces), thetas, groove_z, height


def groove_z_at(point, height=12.0, amp=1.8):
    return height * 0.5 + amp * np.sin(np.arctan2(point[1], point[0]))


# ------------------------------------------------------------------ tests
def test_concavity_finds_groove():
    verts, faces, thetas, groove_z, height = grooved_tube()
    conc = cg.smooth_scalar(cg.vertex_concavity(verts, faces), faces, iterations=2)

    dist_to_groove = np.abs(verts[:, 2] - np.array([groove_z_at(v) for v in verts]))
    in_groove = dist_to_groove < 0.4
    off_groove = dist_to_groove > 2.5

    mean_in, mean_off = conc[in_groove].mean(), conc[off_groove].mean()
    assert mean_in > mean_off, f"groove {mean_in:.4f} not more concave than flank {mean_off:.4f}"
    print(f"PASS  concavity: groove={mean_in:+.4f} vs flank={mean_off:+.4f} "
          f"(separation {mean_in - mean_off:+.4f})")


def test_magnetic_scissors_follows_groove():
    """The real test: with curvature weighting ON the path should track the
    sinusoidal groove; with it OFF it should cut across. If both behave the
    same, the weighting isn't doing anything."""
    verts, faces, thetas, groove_z, height = grooved_tube()
    conc = cg.smooth_scalar(cg.vertex_concavity(verts, faces), faces, iterations=2)

    a = cg.nearest_vertex(verts, np.array([5.0, 0.0, groove_z_at([5, 0, 0])]))
    b = cg.nearest_vertex(verts, np.array([-5.0, 0.0, groove_z_at([-5, 0, 0])]))

    weighted = cg.curvature_weighted_path(verts, faces, conc, a, b, concavity_weight=8.0)
    plain = cg.curvature_weighted_path(verts, faces, conc, a, b, concavity_weight=0.0)
    assert weighted and plain

    def mean_deviation(path):
        pts = verts[path]
        return float(np.mean([abs(p[2] - groove_z_at(p)) for p in pts]))

    dev_w, dev_p = mean_deviation(weighted), mean_deviation(plain)
    assert dev_w < dev_p, f"weighted {dev_w:.3f} should hug groove better than plain {dev_p:.3f}"
    assert dev_w < 0.5, f"weighted path strayed {dev_w:.3f}mm from the groove"
    print(f"PASS  magnetic scissors: curvature-weighted stays {dev_w:.3f}mm from groove, "
          f"unweighted strays {dev_p:.3f}mm ({dev_p / max(dev_w, 1e-6):.1f}x worse)")


def test_closed_loop_and_split():
    verts, faces, thetas, groove_z, height = grooved_tube()
    conc = cg.smooth_scalar(cg.vertex_concavity(verts, faces), faces, iterations=2)

    anchors = []
    for th in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
        z = height * 0.5 + 1.8 * np.sin(th)
        anchors.append([5.0 * np.cos(th), 5.0 * np.sin(th), z])
    anchors = np.array(anchors)

    loop = cg.closed_loop_through_anchors(verts, faces, conc, anchors, concavity_weight=8.0)
    assert len(loop) > 10, f"loop too short: {len(loop)}"
    assert len(loop) == len(set(loop)), "loop revisits a vertex — not a simple cycle"

    seed_above = np.array([0.0, 0.0, height * 0.95])
    (cv, cf), (bv, bf) = cg.split_mesh_by_loop(verts, faces, loop, seed_above)
    assert len(cf) > 0 and len(bf) > 0, "split produced an empty side"
    assert len(cf) + len(bf) == len(faces), "faces lost or duplicated in the split"

    crown_mean_z = cv[:, 2].mean()
    base_mean_z = bv[:, 2].mean()
    assert crown_mean_z > base_mean_z, "seed-side component is not the upper (crown) piece"
    # "mean z", not "z-bar": U+0304 COMBINING MACRON is not in cp1252/cp1256, so
    # on a Windows console this print raised UnicodeEncodeError and the whole
    # file failed AFTER every assertion in it had already passed.
    print(f"PASS  loop+split: {len(loop)}-vertex cycle, crown {len(cf)} tris (mean z={crown_mean_z:.2f}) / "
          f"base {len(bf)} tris (mean z={base_mean_z:.2f}), no faces lost")
    return (cv, cf), (bv, bf)


def test_capping_produces_watertight_solids():
    (cv, cf), (bv, bf) = test_closed_loop_and_split()
    for name, (v, f) in (("crown", (cv, cf)), ("base", (bv, bf))):
        assert not cg.is_edge_manifold_closed(f), f"{name} should start open"
        v2, f2 = cg.cap_and_close(v, f)
        assert cg.is_edge_manifold_closed(f2), f"{name} still open after capping"

        f2 = cg.make_consistent_winding(v2, f2)
        vol = cg.signed_volume(v2, f2)
        assert vol > 0, f"{name} volume {vol:.3f} not positive after orientation fix"

        V, F = len(v2), len(f2)
        E = len(cg.edge_face_incidence(f2))
        chi = V - E + F
        assert chi == 2, f"{name} Euler characteristic {chi}, expected 2 (sphere topology)"
        print(f"PASS  capping[{name}]: watertight, volume={vol:.2f}mm3, Euler chi={chi}")


def test_frame_is_orthonormal_where_plan_formula_is_not():
    """Demonstrates the actual defect in the master-plan axis formula and
    that Gram-Schmidt fixes it."""
    mesial = np.array([-4.0, 0.3, 8.0])
    distal = np.array([4.2, -0.5, 8.3])
    fa = np.array([0.4, 5.1, 9.0])
    crown = np.random.default_rng(1).normal(0, 1.5, (400, 3)) + np.array([0, 0, 9.0])
    rim = np.random.default_rng(2).normal(0, 0.8, (60, 3)) + np.array([0, 0, 5.5])

    # --- master-plan formula, literally applied ---
    u_md_naive = (distal - mesial) / np.linalg.norm(distal - mesial)
    u_bl_naive = (fa - crown.mean(axis=0)) / np.linalg.norm(fa - crown.mean(axis=0))
    skew = abs(np.dot(u_md_naive, u_bl_naive))
    assert skew > 1e-3, "fixture failed to produce a skewed case"

    # --- corrected frame ---
    fr = cg.derive_anatomical_frame(mesial, distal, fa, crown, rim)
    for a, b in (("u_md", "u_bl"), ("u_md", "u_oa"), ("u_bl", "u_oa")):
        assert abs(np.dot(fr[a], fr[b])) < 1e-9, f"{a}·{b} not orthogonal"
    for k in ("u_md", "u_bl", "u_oa"):
        assert abs(np.linalg.norm(fr[k]) - 1) < 1e-9
    assert np.dot(fr["u_oa"], fr["centroid"] - fr["rim_centroid"]) > 0, "u_oa not occlusal"
    print(f"PASS  frame: plan formula gives u_MD·u_BL={skew:.4f} (skewed, would shear every "
          f"rotation); Gram-Schmidt version is orthonormal to 1e-9")


def test_cres_and_pivot():
    fr = dict(centroid=np.array([0., 0., 10.]), u_md=np.array([1., 0., 0.]),
              u_bl=np.array([0., 1., 0.]), u_oa=np.array([0., 0., 1.]),
              rim_centroid=np.array([0., 0., 5.]))
    c_res = cg.center_of_resistance(fr, 10.0)
    assert np.allclose(c_res, [0, 0, 0]), c_res

    M = cg.kinematic_matrix(fr, c_res, tip_deg=10.0)
    moved = cg.apply_matrix(np.array([c_res]), M)[0]
    assert np.allclose(moved, c_res, atol=1e-9), "C_res must not move under pure rotation"

    crown_pt = cg.apply_matrix(np.array([fr["centroid"]]), M)[0]
    assert not np.allclose(crown_pt, fr["centroid"]), "crown should swing about C_res"
    print(f"PASS  C_res pivot: C_res fixed under 10° tip; crown centroid swept "
          f"{np.linalg.norm(crown_pt - fr['centroid']):.3f}mm")


def test_tip_torque_axes_are_clinically_correct():
    fr = dict(centroid=np.zeros(3), u_md=np.array([1., 0., 0.]),
              u_bl=np.array([0., 1., 0.]), u_oa=np.array([0., 0., 1.]),
              rim_centroid=np.array([0., 0., -1.]))
    c = np.zeros(3)
    tip = cg.kinematic_matrix(fr, c, tip_deg=10.0)[:3, :3]
    assert np.allclose(tip, cg.rotation_matrix_axis_angle(fr["u_bl"], 10.0)), "Tip must rotate about u_BL"
    torque = cg.kinematic_matrix(fr, c, torque_deg=10.0)[:3, :3]
    assert np.allclose(torque, cg.rotation_matrix_axis_angle(fr["u_md"], 10.0)), "Torque must rotate about u_MD"
    print("PASS  axes: Tip→u_BL, Torque→u_MD (the swap from the earlier build is fixed)")


def test_staging_arithmetic():
    s = cg.staging_estimate(tip_deg=6.0, torque_deg=0, rotation_deg=0,
                            d_md=0.9, d_bl=0, d_oa=0)
    assert s["stages_from_translation"] == 4   # ceil(0.9/0.25)
    assert s["stages_from_rotation"] == 3      # ceil(6/2)
    assert s["stages_required"] == 4 and s["driver"] == "translation"
    assert not any(isinstance(v, str) and ("expand" in v.lower() or "warn" in v.lower())
                   for v in s.values()), "staging must not emit clinical advice"
    print(f"PASS  staging: 0.90mm + 6.0° → {s['stages_required']} stages "
          f"(driver: {s['driver']}), numbers only, no verdicts")


if __name__ == "__main__":
    test_concavity_finds_groove()
    test_magnetic_scissors_follows_groove()
    test_capping_produces_watertight_solids()
    test_frame_is_orthonormal_where_plan_formula_is_not()
    test_cres_and_pivot()
    test_tip_torque_axes_are_clinically_correct()
    test_staging_arithmetic()
    print("\nALL CORE ENGINE TESTS PASSED")
