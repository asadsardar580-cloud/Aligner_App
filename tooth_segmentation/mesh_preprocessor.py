"""Stage 2. Non-destructive: returns a derived working mesh, never modifies
the caller's arrays (spec section 4)."""
import time
import numpy as np
import core_geometry as cg
from .models import PreprocessReport


def _degenerate_mask(verts, faces, area_eps=1e-12):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    repeated = ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
                | (faces[:, 0] == faces[:, 2]))
    return (area <= area_eps) | repeated


def preprocess(verts, faces, weld_tolerance=0.0):
    """Weld duplicates, drop degenerates, report topology.

    Welding is by exact bit pattern when weld_tolerance == 0. That default is
    deliberate: a tolerance-based weld silently moves the clinician's scan
    geometry, and STL exporters already emit bit-identical coordinates at
    shared triangle corners, so exact matching recovers full connectivity
    without altering a single vertex position.
    """
    t = {}
    t0 = time.time()
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    if weld_tolerance > 0:
        keys = np.round(verts / weld_tolerance).astype(np.int64)
    else:
        keys = verts
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_merged = len(verts) - len(uniq)

    if weld_tolerance > 0:
        new_verts = np.zeros((len(uniq), 3))
        np.add.at(new_verts, inverse, verts)
        counts = np.bincount(inverse, minlength=len(uniq))
        new_verts /= counts[:, None]
    else:
        new_verts = uniq.astype(float)
    new_faces = inverse[faces]
    t["weld"] = time.time() - t0

    t0 = time.time()
    degen = _degenerate_mask(new_verts, new_faces)
    n_degen = int(degen.sum())
    new_faces = new_faces[~degen]
    t["degenerate"] = time.time() - t0

    t0 = time.time()
    inc = cg.edge_face_incidence(new_faces)
    n_boundary = sum(1 for f in inc.values() if len(f) == 1)
    n_nonmanifold = sum(1 for f in inc.values() if len(f) > 2)
    comps = cg.connected_components(new_faces, loop=[])
    t["topology"] = time.time() - t0

    report = PreprocessReport(
        n_vertices=len(new_verts), n_faces=len(new_faces),
        n_connected_components=len(comps),
        n_boundary_edges=n_boundary, n_nonmanifold_edges=n_nonmanifold,
        n_degenerate_faces=n_degen, n_duplicate_vertices_merged=int(n_merged),
        watertight=(n_boundary == 0 and n_nonmanifold == 0),
        bounding_box=(tuple(new_verts.min(axis=0)), tuple(new_verts.max(axis=0))),
        timings=t,
    )
    return new_verts, new_faces, report
