"""Stage 4. Curvature features, normalized and mesh-density independent."""
import numpy as np
import core_geometry as cg


def curvature_features(verts, faces, edges=None, smoothing_iterations=2):
    """Returns dict of per-vertex features in roughly [-1, 1].

    Uses the concavity estimator from core_geometry (verified against a
    reference implementation to 1e-16). It is normalized by its own maximum
    and averaged over unique neighbours, so it does not scale with triangle
    count -- which is what spec section 6 requires for robustness across STL
    resolutions.
    """
    if edges is None:
        edges = cg.directed_edges(faces)
    raw = cg.vertex_concavity_fast(verts, faces, edges=edges)
    smooth = cg.smooth_scalar_fast(raw, faces, iterations=smoothing_iterations, edges=edges)
    return dict(concavity_raw=raw, concavity=smooth,
                valley=np.maximum(smooth, 0.0), ridge=np.maximum(-smooth, 0.0))