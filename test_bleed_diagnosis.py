"""Why does a 26x curvature barrier still bleed on a real scan?

Hypothesis: vertex_concavity_fast normalizes by the GLOBAL MAXIMUM. Real
scans contain outlier spikes - scan-edge artifacts, noise, the base rim -
whose concavity far exceeds any anatomical feature. One such vertex sets the
normalizer and crushes the true sulcus signal toward zero, so the barrier
silently collapses.

Reproduced by adding a handful of spikes to an otherwise clean arch."""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch

verts, faces, centres = synthetic_arch(n_teeth=5, spacing=7.0, crown_r=3.2,
                                       grid=200, blend=3.0)
edges = cg.directed_edges(faces)
print(f"Dense arch: {len(verts):,} verts / {len(faces):,} faces")

def sulcus_signal(v, f, e, label):
    conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(v, f, edges=e), f, 2, edges=e)
    d = np.linalg.norm(v[:, :2] - centres[2], axis=1)
    sulcus = conc[(d > 2.8) & (d < 3.6)].mean()
    crown  = conc[d < 1.5].mean()
    barrier = 1 + 25 * max(sulcus, 0)
    print(f"  {label:<26} sulcus={sulcus:6.3f}  crown={crown:+6.3f}  "
          f"-> linear barrier {barrier:5.1f}x")
    return sulcus, conc

print("\nCLEAN MESH")
s_clean, _ = sulcus_signal(verts, faces, edges, "no artifacts")

# inject scan-like outliers: a few vertices displaced inward sharply
rng = np.random.default_rng(0)
spiky = verts.copy()
victims = rng.choice(len(verts), size=12, replace=False)
spiky[victims, 2] -= 3.0          # sharp inward spikes -> extreme concavity
print("\nWITH 12 OUTLIER SPIKES (0.006% of vertices)")
s_spiky, conc_spiky = sulcus_signal(spiky, faces, edges, "global-max normalization")

collapse = s_clean / max(s_spiky, 1e-9)
print(f"\n  -> 12 bad vertices weakened the sulcus signal {collapse:.1f}x")
print(f"     The barrier drops from {1+25*s_clean:.0f}x to {1+25*s_spiky:.0f}x.")
print(f"     That is the bleed: the flood walks straight over the gumline.")

# measure actual selection bleed
frame_axis = np.array([0.0, 0.0, 1.0])
for label, v in (("clean", verts), ("with spikes", spiky)):
    conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(v, faces, edges=edges),
                                 faces, 2, edges=edges)
    graph = cg.build_barrier_graph(v, faces, conc, 25.0, edges=edges)
    seed = np.array([centres[2][0], centres[2][1], 7.0])
    dist = cg.geodesic_from_seed(graph, v, seed)
    mask = cg.mask_from_distance(faces, dist, 10.0)
    print(f"  selection on {label:<12}: {mask.sum():>6,} faces ({mask.sum()/len(faces):.1%})")

# ============================ AFTER THE FIX ============================
print("\n" + "="*62)
print("AFTER: robust percentile normalization + exponential barrier")
print("="*62)
for label, v in (("clean", verts), ("with spikes", spiky)):
    conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(v, faces, edges=edges),
                                 faces, 2, edges=edges)
    d = np.linalg.norm(v[:, :2] - centres[2], axis=1)
    sulcus = conc[(d > 2.8) & (d < 3.6)].mean()
    graph = cg.build_barrier_graph(v, faces, conc, 6.0, edges=edges, mode="exponential")
    seed = np.array([centres[2][0], centres[2][1], 7.0])
    dist = cg.geodesic_from_seed(graph, v, seed)
    mask = cg.mask_from_distance(faces, dist, 10.0)
    print(f"  {label:<12} sulcus={sulcus:.3f}  barrier={np.exp(6*max(sulcus,0)):7.1f}x  "
          f"selection {mask.sum():>6,} faces ({mask.sum()/len(faces):.1%})")
