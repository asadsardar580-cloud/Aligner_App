"""Synthetic dental arch: teeth along a parabola on a gingival base, with
CONFIGURABLE interproximal spacing so we can measure where curvature-based
separation stops working. Height field is the max of per-tooth domes, which
reproduces the real physics of contact: well-spaced crowns leave a valley
between them, crowded ones merge into a single smooth surface."""
import numpy as np


def synthetic_arch(n_teeth=6, spacing=7.0, crown_r=3.2, crown_h=7.0,
                   sulcus_depth=0.9, sulcus_w=0.6, grid=190, extent=26.0,
                   rotate=None, noise=0.0, seed=0, blend=None):
    """`blend` controls how sharply adjacent crowns meet.

    None  -> hard max: crowns meet in a sharp crease. Mathematically a strong
             concavity, and NOT what a real scan looks like -- it makes
             interproximal detection appear far easier than it is.
    float -> smooth-max with stiffness `blend` (lower = smoother). Models the
             real situation: enamel contact plus scanner smoothing merge
             adjacent crowns into a continuous surface with little or no
             valley. This is the honest test case.
    """
    xs = np.linspace(-extent, extent, grid)
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    # tooth centres along a parabolic arch
    t = np.linspace(-1, 1, n_teeth)
    cx = t * spacing * (n_teeth - 1) / 2.0
    cy = 0.35 * (cx ** 2) / max(spacing, 1e-6) - 6.0
    centres = np.column_stack([cx, cy])

    domes = []
    for (a, b) in centres:
        R = np.sqrt((X - a) ** 2 + (Y - b) ** 2)
        dome = crown_h * np.exp(-(R ** 2) / (2 * (crown_r / 1.6) ** 2))
        dome -= sulcus_depth * np.exp(-((R - crown_r) ** 2) / (2 * sulcus_w ** 2))
        domes.append(dome)
    stack = np.stack(domes + [np.zeros_like(X)])
    if blend is None:
        Z = stack.max(axis=0)
    else:
        m = stack.max(axis=0)
        Z = m + np.log(np.exp(blend * (stack - m)).sum(axis=0)) / blend

    if noise > 0:
        Z = Z + np.random.default_rng(seed).normal(0, noise, Z.shape)

    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    if rotate is not None:
        verts = verts @ rotate.T

    faces = []
    g = grid
    for i in range(g - 1):
        for j in range(g - 1):
            a, b = i * g + j, i * g + j + 1
            c, d = (i + 1) * g + j, (i + 1) * g + j + 1
            faces.append([a, c, b]); faces.append([b, c, d])
    return verts, np.array(faces), centres
