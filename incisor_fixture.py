"""An incisor-like blade: facial and lingual slopes meeting at a SHARP
incisal edge, on gingiva, with a cervical groove. The point is to have a
strongly CONVEX ridge that the flood must cross, and a CONCAVE sulcus it
must not."""
import numpy as np


def incisor_on_gingiva(grid=200, extent=14.0, crown_h=9.0, slope=2.6,
                       half_width=4.0, sulcus_depth=1.0, sulcus_w=0.5,
                       edge_sharpness=0.0):
    xs = np.linspace(-extent, extent, grid)
    X, Y = np.meshgrid(xs, xs, indexing="ij")

    # tent: ridge runs along x at y=0 -> sharp convex incisal edge
    tent = np.maximum(0.0, crown_h - slope * np.abs(Y))
    if edge_sharpness > 0:                       # optional rounding of the edge
        tent = crown_h - slope * np.sqrt(Y**2 + edge_sharpness**2)
        tent = np.maximum(tent, 0.0)

    # taper in x so the crown has mesial/distal ends
    window = np.clip((half_width - np.abs(X)) / 1.2, 0.0, 1.0)
    z = tent * window

    # cervical groove around the crown footprint
    fy = crown_h / slope
    dx = np.abs(X) - half_width
    dy = np.abs(Y) - fy
    outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    dist_to_edge = outside + inside
    z -= sulcus_depth * np.exp(-(dist_to_edge**2) / (2 * sulcus_w**2))

    verts = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])
    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            a, b = i*grid + j, i*grid + j + 1
            c, d = (i+1)*grid + j, (i+1)*grid + j + 1
            faces.append([a, c, b]); faces.append([b, c, d])
    return verts, np.array(faces), dict(crown_h=crown_h, slope=slope,
                                        half_width=half_width, fy=fy)
