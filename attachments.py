"""Parametric composite attachments, fused to a crown before staging.

WHAT AN ATTACHMENT IS, MECHANICALLY. A clear aligner is a passive shell; it can
push a crown but it has almost nothing to grip for rotation, extrusion or root
torque. A composite attachment is a small block bonded to the enamel that gives
the tray a surface to bear against. Its ORIENTATION is the whole point — a
vertical rectangle resists rotation, a horizontal bevel resists extrusion.

TWO RULES THIS MODULE FOLLOWS.

1. THE ATTACHMENT IS PLACED IN THE TOOTH'S OWN ANATOMICAL FRAME, not in world
   axes. "Vertical" means along u_OA, "horizontal" means along u_MD, and
   "facial" means along u_BL. A world-axis box would be vertical only for teeth
   that happen to stand upright in scanner space, which is none of them.

2. IT IS FUSED BEFORE STAGING, NOT AFTER. The attachment moves with the crown,
   so it must be part of the solid the stage models are built from. Bonding it
   to a moved crown afterwards would put it where the tooth ENDS UP rather than
   where the clinician placed it.

The union runs through manifold3d, the same path the manufacturing export
already uses, and inherits its crumb filtering: a tangential boolean leaves
occasional inside-out fragments with negative volume, and those are voids, not
geometry.
"""
from __future__ import annotations

import numpy as np

# Standard shapes. Dimensions in mm, as (mesiodistal, occlusoapical, facial).
# HEURISTIC defaults drawn from commonly published attachment sizes; they are
# starting points a clinician overrides, not prescriptions.
SHAPES = {
    "vertical_rectangular":   {"md": 2.0, "oa": 3.0, "bl": 1.0,
                               "purpose": "resists rotation about the long axis"},
    "horizontal_rectangular": {"md": 3.0, "oa": 2.0, "bl": 1.0,
                               "purpose": "resists mesiodistal tipping"},
    "horizontal_bevel":       {"md": 3.0, "oa": 2.0, "bl": 1.25,
                               "purpose": "resists extrusion — the bevel gives the tray "
                                          "an occlusal face to pull against"},
    "ellipsoidal":            {"md": 2.5, "oa": 2.5, "bl": 1.0,
                               "purpose": "general retention, least irritating to soft tissue"},
}

MIN_DIM_MM = 0.5     # smaller than this will not survive thermoforming
MAX_DIM_MM = 6.0     # larger than this is a fixed appliance, not an attachment


def _box(md, oa, bl):
    """Unit box vertices/faces in (md, oa, bl) half-extents, origin centred."""
    x, y, z = md / 2.0, oa / 2.0, bl / 2.0
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], int)
    return v, f


def _bevel(md, oa, bl):
    """A box whose occlusal face is cut back, leaving a ramp the tray pulls on."""
    v, f = _box(md, oa, bl)
    # Pull the occlusal-facial edge inward: vertices 6 and 7 are +oa, +bl.
    v[6, 2] *= 0.25
    v[7, 2] *= 0.25
    return v, f


def _ellipsoid(md, oa, bl, n_theta=12, n_phi=8):
    verts, faces = [], []
    for i in range(n_phi + 1):
        phi = np.pi * i / n_phi
        for j in range(n_theta):
            th = 2 * np.pi * j / n_theta
            verts.append([md / 2 * np.sin(phi) * np.cos(th),
                          oa / 2 * np.sin(phi) * np.sin(th),
                          bl / 2 * np.cos(phi)])
    for i in range(n_phi):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = a + n_theta
            d = b + n_theta
            faces.append([a, c, b])
            faces.append([b, c, d])
    return np.asarray(verts, float), np.asarray(faces, int)


def build_attachment(shape: str, frame: dict, position_xyz, size_mm: dict | None = None,
                     embed_mm: float = 0.3):
    """An attachment solid, oriented in the TOOTH'S frame and seated on its surface.

    `embed_mm` sinks the block slightly into the crown so the boolean union has
    real overlap to work with. A face-to-face tangential union is exactly the
    situation manifold3d answers with coincident-but-distinct vertices, which
    binary STL cannot express — the same tangency problem the socket/crown
    refit already documents.
    """
    if shape not in SHAPES:
        raise ValueError(f"Unknown attachment shape {shape!r}. Available: {sorted(SHAPES)}")
    dims = dict(SHAPES[shape])
    dims.update(size_mm or {})
    for k in ("md", "oa", "bl"):
        d = float(dims[k])
        if not (MIN_DIM_MM <= d <= MAX_DIM_MM):
            raise ValueError(
                f"Attachment {k} of {d}mm is outside {MIN_DIM_MM}-{MAX_DIM_MM}mm. "
                f"Below that it will not survive thermoforming; above it, this is a "
                f"fixed appliance rather than an attachment.")

    if shape == "ellipsoidal":
        v, f = _ellipsoid(dims["md"], dims["oa"], dims["bl"])
    elif shape == "horizontal_bevel":
        v, f = _bevel(dims["md"], dims["oa"], dims["bl"])
    else:
        v, f = _box(dims["md"], dims["oa"], dims["bl"])

    # Rotate local (md, oa, bl) into the tooth's anatomical axes. THIS is what
    # makes "vertical" mean along the tooth's long axis rather than along world Z.
    u_md = np.asarray(frame["u_md"], float)
    u_oa = np.asarray(frame["u_oa"], float)
    u_bl = np.asarray(frame["u_bl"], float)
    R = np.column_stack([u_md, u_oa, u_bl])

    # Seat it: push back along the facial normal so it sinks embed_mm into enamel.
    centre = np.asarray(position_xyz, float) + u_bl * (dims["bl"] / 2.0 - embed_mm)
    world = v @ R.T + centre

    return {
        "verts": world, "faces": f, "shape": shape,
        "dimensions_mm": {k: float(dims[k]) for k in ("md", "oa", "bl")},
        "purpose": SHAPES[shape]["purpose"],
        "embed_mm": float(embed_mm),
        "oriented_in": "tooth anatomical frame (u_md, u_oa, u_bl)",
    }


def fuse_to_crown(crown_verts, crown_faces, attachment) -> dict:
    """Boolean-union an attachment onto a crown. One positive-volume body or refuse.

    Runs through the same manifold3d path the manufacturing export uses, and
    applies the same crumb filter: a tangential boolean leaves occasional
    inside-out fragments whose volume is NEGATIVE, and those are voids rather
    than geometry.
    """
    import manifold3d as m3

    def _solid(v, f):
        return m3.Manifold(m3.Mesh(
            vert_properties=np.asarray(v, np.float32),
            tri_verts=np.asarray(f, np.uint32)))

    fused = _solid(crown_verts, crown_faces) + _solid(attachment["verts"], attachment["faces"])

    bodies = [b for b in fused.decompose() if b.volume() > 0]
    if len(bodies) != 1:
        raise ValueError(
            f"Fusing the attachment produced {len(bodies)} positive-volume bodies. "
            f"One means the attachment is not touching the crown — move it onto the "
            f"surface, or increase embed_mm.")

    mesh = bodies[0].to_mesh()
    return {
        "verts": np.asarray(mesh.vert_properties[:, :3], float),
        "faces": np.asarray(mesh.tri_verts, int),
        "volume_mm3": round(float(bodies[0].volume()), 3),
        "bodies": 1,
        "shape": attachment["shape"],
    }
