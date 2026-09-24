"""Draw the tooth/gum labels so a clinician can see them. Two views.

TASK 1 step 5. Gum RED, teeth IVORY, from the tongue side and the cheek
side, because the two hide different faults: specks on a crown show from the
cheek, and gum painted as tooth shows best along the lingual gingival band.

NO BROWSER AND NO GL CONTEXT HERE, so this is a plain painter's-algorithm
rasteriser in NumPy: project with an orthographic camera, sort faces
back-to-front, fill. That is enough for the question being asked - which
label is on which anatomy - and it cannot fail for want of a display.

A FACE IS DRAWN WITH THE LABEL ITS THREE CORNERS AGREE ON, and drawn in a
third colour when they do not. A face straddling the boundary is neither gum
nor tooth, and painting it as either would hide exactly the ragged cervical
margin this image exists to reveal.

    .venv\\Scripts\\python.exe tools\\render_labels_check.py [--provider NAME]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import arch_frame                                            # noqa: E402
import core_geometry as cg                                   # noqa: E402
import segmentation_cleanup as sc                            # noqa: E402
import segmentation_diagnostics as sd                        # noqa: E402
import stl_io                                                # noqa: E402

SCAN = "case_lower.stl"
OUT = "scratch/labels_check.png"

GUM = np.array([0.80, 0.19, 0.19])        # red
TOOTH = np.array([0.96, 0.94, 0.86])      # ivory
SEAM = np.array([0.45, 0.45, 0.50])       # a face whose corners disagree
BG = np.array([0.10, 0.11, 0.13])

W, H = 1100, 500


def render(v, f, colours, eye, up, w=W, h=H):
    """Orthographic painter's algorithm. Returns an (h, w, 3) float image."""
    eye = eye / np.linalg.norm(eye)
    right = np.cross(up, eye)
    right /= np.linalg.norm(right)
    true_up = np.cross(eye, right)

    P = np.stack([v @ right, v @ true_up, v @ eye], axis=1)
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    # FIT EACH AXIS SEPARATELY, with one shared scale so nothing is stretched.
    # Scaling by the larger span alone left a laterally-viewed arch - wide and
    # shallow - occupying a third of its panel, which wastes the resolution on
    # the cervical margin this image exists to show.
    ext = hi - lo
    pad = 0.04 * ext.max()
    scale = min((w - 1) / (ext[0] + 2 * pad), (h - 1) / (ext[1] + 2 * pad))
    centre = (lo + hi) / 2.0
    xy = (P[:, :2] - centre) * scale
    xy[:, 0] += w / 2.0
    xy[:, 1] = h / 2.0 - xy[:, 1]

    img = np.tile(BG, (h, w, 1))
    tri = xy[f]
    depth = P[f, 2].mean(axis=1)

    # Flat shading against the view direction: a face turned away from the
    # camera is darker, which is what makes a cusp read as a cusp.
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1.0
    n = n / ln[:, None]
    lam = np.clip(n @ eye, 0.0, 1.0) * 0.75 + 0.25

    order = np.argsort(depth)              # far first
    shaded = np.clip(colours * lam[:, None], 0, 1)

    xs = np.arange(w)
    for k in order:
        t = tri[k]
        x0, x1 = int(np.floor(t[:, 0].min())), int(np.ceil(t[:, 0].max()))
        y0, y1 = int(np.floor(t[:, 1].min())), int(np.ceil(t[:, 1].max()))
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        x0, x1 = max(x0, 0), min(x1, w - 1)
        y0, y1 = max(y0, 0), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = t[0]
        bx, by = t[1]
        cx, cy = t[2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        gx = xs[x0:x1 + 1][None, :]
        gy = np.arange(y0, y1 + 1)[:, None]
        l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
        l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        if inside.any():
            img[y0:y1 + 1, x0:x1 + 1][inside] = shaded[k]
    return img


def arch_axes(v, u_occ):
    """(u_tra, u_sag, posterior width, anterior width), with +u_sag pointing
    ANTERIORLY.

    ANTERIOR is the NARROW end: an arch is widest between the molars and
    narrowest at the incisors, for every human arch (s.20.1). The first
    version of this flipped u_sag when the -u_sag end was the WIDER one -
    i.e. exactly when +u_sag already pointed anteriorly - so +u_sag came out
    POSTERIOR and the two panels' cheek / tongue titles were swapped (Task 2
    step 7; measured on the synthetic horseshoe, whose anterior is +y).
    """
    v = np.asarray(v, float)
    u_occ = np.asarray(u_occ, float)
    d = v - v.mean(axis=0)
    inplane = d - np.outer(d @ u_occ, u_occ)
    _, _, vt = np.linalg.svd(inplane[::37], full_matrices=False)
    u_tra, u_sag = vt[0], vt[1]           # across the arch, along the arch
    s = d @ u_sag
    w_lo = np.ptp(d[s < np.percentile(s, 25)] @ u_tra)   # the -u_sag end
    w_hi = np.ptp(d[s > np.percentile(s, 75)] @ u_tra)   # the +u_sag end
    if w_hi > w_lo:
        u_sag = -u_sag                    # the +u_sag end was the wide one
    return u_tra, u_sag, max(w_lo, w_hi), min(w_lo, w_hi)


def label_views(u_sag, u_occ):
    """The two panels, (title, eye). `render` puts the camera on the +eye
    side, so an eye along +u_sag (anterior) looks at the arch from OUTSIDE -
    the cheek and lip side - and one along -u_sag looks from inside it, the
    tongue side.

    LOW ELEVATION ON PURPOSE. A near-occlusal view shows the chewing surfaces
    and hides the one place the labelling is hard to get right - the cervical
    margin, where gum meets enamel all the way round. These sit just above
    the occlusal plane so the margin band is visible along its whole length
    on each side.
    """
    u_sag = np.asarray(u_sag, float)
    u_occ = np.asarray(u_occ, float)
    return [("cheek side (buccal)", u_sag * 0.96 + u_occ * 0.28),
            ("tongue side (lingual)", -u_sag * 0.96 + u_occ * 0.28)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    raw = open(SCAN, "rb").read()
    v, f = stl_io.parse_stl_bytes(raw)
    v, f, _ = cg.sanitize_scan(v, f)
    v, f, _ = cg.condition_mesh(v, f)
    print(f"{len(v):,} verts / {len(f):,} faces")

    import segmentation_providers as sp
    prov = sp.get(args.provider)
    print(f"provider: {prov.name}")
    res = prov.segment(v, f, "lower")
    lab = np.asarray(res.labels).astype(np.int64).reshape(-1)
    lab, rep = sc.clean_labels(v, f, lab)
    print(f"cleanup: {rep['vertices_changed']:,} vertices relabelled")

    band = sd.band_report(v, f, lab)
    print(f"bands: occlusal {band['occlusal']['tooth_share'] * 100:.2f}% tooth, "
          f"gingival {band['gingival']['gum_share'] * 100:.2f}% gum, "
          f"ok={band['ok']}")

    # A face takes the label its three corners agree on.
    fl = lab[f]
    agree = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
    colours = np.tile(SEAM, (len(f), 1))
    colours[agree & (fl[:, 0] == 0)] = GUM
    colours[agree & (fl[:, 0] != 0)] = TOOTH
    print(f"faces: {int((agree & (fl[:, 0] == 0)).sum()):,} gum, "
          f"{int((agree & (fl[:, 0] != 0)).sum()):,} tooth, "
          f"{int((~agree).sum()):,} straddling the boundary")

    u_occ = np.asarray(band["u_occ"], float)
    # "cheek side" and "tongue side" the right way round, derived from the
    # arch rather than a coin flip - see arch_axes.
    _, u_sag, w_post, w_ant = arch_axes(v, u_occ)
    print(f"arch width: posterior {w_post:.1f}mm vs anterior "
          f"{w_ant:.1f}mm (ratio {w_post / max(w_ant, 1e-9):.2f})")
    views = label_views(u_sag, u_occ)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 4.6), facecolor=tuple(BG))
    for ax, (title, eye) in zip(axes, views):
        img = render(v, f, colours, np.asarray(eye, float), u_occ)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)

    handles = [mpatches.Patch(color=tuple(GUM), label="gingiva (label 0)"),
               mpatches.Patch(color=tuple(TOOTH), label="tooth (FDI)"),
               mpatches.Patch(color=tuple(SEAM), label="face straddling the boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               labelcolor="white", fontsize=11)
    fig.suptitle(
        f"{SCAN} - {prov.name} + cleanup   |   occlusal band "
        f"{band['occlusal']['tooth_share'] * 100:.1f}% tooth, gingival band "
        f"{band['gingival']['gum_share'] * 100:.1f}% gum",
        color="white", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    os.makedirs("scratch", exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=tuple(BG))
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
