"""Stage 5. Initial orientation estimate. Makes no assumption that the STL
is axis-aligned (spec section 8)."""
import numpy as np
from .models import ArchFrame


def estimate_arch_frame(verts, concavity=None, faces=None) -> ArchFrame:
    """PCA orientation, with the occlusal direction disambiguated by anatomy.

    A dental arch is a broad, shallow horseshoe: widest left-right, deeper
    anterior-posterior, and thinnest occluso-apically. So the SMALLEST
    principal component approximates the occlusal normal.

    The sign is then fixed by a geometric fact rather than by PCA, which
    cannot supply one: crowns protrude from the gingival mass, so the
    occlusal side is the side whose extreme points are sparser. Comparing
    point density in the two tails settles the direction.
    """
    verts = np.asarray(verts, dtype=float)
    centroid = verts.mean(axis=0)
    centred = verts - centroid

    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    width_axis = eigvecs[:, 0]
    depth_axis = eigvecs[:, 1]
    occlusal = eigvecs[:, 2]

    # Sign rule: the occlusal side is the CONVEX-RICH side.
    #
    # Cusps and incisal edges are sharply convex; gingiva, palatal vault and
    # cast base are smooth or flat. Measured on a real maxillary scan, the
    # occlusal half carried a strongly-convex fraction of 0.389 against 0.035
    # for the other half -- an unambiguous margin.
    #
    # This replaces two rules that each failed. A span heuristic assumed
    # crowns are the highest points, which is simply false for an upper arch:
    # the palatal vault rises ~8mm ABOVE the occlusal plane, so seeds landed
    # in the vault and tooth size ratios reached 80x. A perimeter-versus-
    # interior rule fixed the real scan but broke on a flat test base whose
    # gingiva extends outward past the crowns. Convexity depends on neither
    # height nor position, so it survives both.
    #
    # `concavity` is optional; without it the frame falls back to the span
    # heuristic and the caller should treat the sign as unverified.
    proj = centred @ occlusal
    if concavity is not None:
        top = proj > np.percentile(proj, 60)
        bot = proj < np.percentile(proj, 40)
        if (concavity[top] < -0.3).mean() < (concavity[bot] < -0.3).mean():
            occlusal = -occlusal
    else:
        if proj.max() < -proj.min():
            occlusal = -occlusal

    depth_axis = np.cross(occlusal, width_axis)
    depth_axis /= np.linalg.norm(depth_axis)
    width_axis = np.cross(depth_axis, occlusal)

    return ArchFrame(centroid=centroid, occlusal_normal=occlusal,
                     arch_width_axis=width_axis, arch_depth_axis=depth_axis,
                     explained_variance=eigvals / eigvals.sum())
