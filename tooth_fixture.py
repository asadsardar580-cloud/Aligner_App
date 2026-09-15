"""Realistic fixture: a small tooth-like bump on a large gingival base,
with a concave sulcus ring at its neck. Unlike the grooved tube, the crown
here is a SMALL fraction of the mesh -- which is what a real arch looks like
and what the plausibility guard is calibrated for."""
import numpy as np

def tooth_on_base(n=170, extent=20.0, crown_h=7.0, crown_r=4.0,
                  sulcus_depth=0.9, sulcus_w=0.6):
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    # crown: smooth dome
    Z = crown_h * np.exp(-(R**2) / (2 * (crown_r/1.6)**2))
    # sulcus: concave ring at the neck
    Z -= sulcus_depth * np.exp(-((R - crown_r)**2) / (2 * sulcus_w**2))

    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for i in range(n-1):
        for j in range(n-1):
            a, b = i*n+j, i*n+j+1
            c, d = (i+1)*n+j, (i+1)*n+j+1
            faces.append([a, c, b]); faces.append([b, c, d])
    return verts, np.array(faces), crown_r


def flat_molar_on_base(n=170, extent=20.0, crown_h=2.2, crown_r=5.5,
                       sulcus_depth=0.9, sulcus_w=0.7, sulcus_offset=1.2,
                       scallop=0.25):
    """A SQUAT, WIDE molar: short crown, broad scalloped cervical ring.

      * crown_h 2.2 against crown_r 5.5 -- a molar is short and broad, so the
        vertical gap between the crown centroid and the rim centroid collapses
        to about a millimetre and any lateral error dominates the ratio;
      * sulcus_offset -- a real cervical margin is not concentric with the
        crown;
      * scallop -- the margin rises and falls interproximally, so the rim is
        genuinely non-planar and the plane fit is not handed an easy case.

    NOTE, measured rather than assumed: this mesh does NOT on its own make the
    centroid-difference axis fail. Z is built on a uniform XY grid, so the mean
    of any region selected from it has mean(x) = mean(y) fixed by the region's
    SHAPE, whatever the heights do -- offsetting the dome or the sulcus moves
    the crown and rim centroids together and the difference cancels (measured:
    2.3 deg residual). The lateral drift that breaks the old estimator comes
    from a LOPSIDED SELECTION, which is what the bug report describes and what
    test_kinematics_frame.py induces by truncating the mask.

    Returns (verts, faces, crown_r).
    """
    xs = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    R_crown = np.sqrt(X**2 + Y**2)
    Z = crown_h * np.exp(-(R_crown**2) / (2 * (crown_r/1.6)**2))

    # sulcus ring, displaced laterally and scalloped around its circumference
    dx, dy = X - sulcus_offset, Y
    R_sul = np.sqrt(dx**2 + dy**2)
    ring = crown_r + scallop * np.sin(3 * np.arctan2(dy, dx))
    Z -= sulcus_depth * np.exp(-((R_sul - ring)**2) / (2 * sulcus_w**2))

    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for i in range(n-1):
        for j in range(n-1):
            a, b = i*n+j, i*n+j+1
            c, d = (i+1)*n+j, (i+1)*n+j+1
            faces.append([a, c, b]); faces.append([b, c, d])
    return verts, np.array(faces), crown_r
