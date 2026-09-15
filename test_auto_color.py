"""How well does whole-arch auto-colouring actually work?

Measures it honestly against ground truth on a synthetic arch, including the
crowded case, so we know what to promise the clinician."""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch
from tooth_segmentation import arch_geometry

CROWN_R = 3.2

def truth_labels(verts, faces, centres):
    fc = verts[faces].mean(axis=1)
    d = np.linalg.norm(fc[:, None, :2] - centres[None, :, :], axis=2)
    return np.where(d.min(axis=1) < CROWN_R, d.argmin(axis=1) + 1, 0).astype(np.int16)

print(f"{'spacing':>8} {'seeds':>6} {'teeth found':>12} | {'tooth/gingiva':>14} {'per-tooth':>10}")
print("-" * 62)

for spacing in [8.0, 7.0, 6.4, 5.8]:
    verts, faces, centres = synthetic_arch(n_teeth=5, spacing=spacing,
                                           crown_r=CROWN_R, grid=150, blend=3.0)
    edges = cg.directed_edges(faces)
    conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges),
                                 faces, 2, edges=edges)
    frame = arch_geometry.estimate_arch_frame(verts)

    labels, seeds = cg.auto_segment_arch(verts, faces, conc, frame.occlusal_normal,
                                         edges=edges, tolerance=12.0,
                                         min_separation_mm=spacing*0.8)
    truth = truth_labels(verts, faces, centres)

    # binary task: is this face tooth or gingiva?
    binary = ((labels > 0) == (truth > 0)).mean()

    # instance task: do found teeth match individual ground-truth teeth?
    found = len(np.unique(labels[labels > 0]))
    matched = 0
    for lab in np.unique(labels[labels > 0]):
        m = labels == lab
        overlap = [np.sum(m & (truth == t)) for t in range(1, 6)]
        if max(overlap) / max(m.sum(), 1) > 0.7:
            matched += 1
    per_tooth = matched / max(found, 1)

    print(f"{spacing:8.1f} {len(seeds):6d} {found:>7d}/5     | "
          f"{binary:13.1%} {per_tooth:9.0%}")

print()
print("Reading:")
print("  tooth/gingiva = the colouring task you asked for (binary, semantic)")
print("  per-tooth     = correctly separating INDIVIDUAL teeth (instance)")
