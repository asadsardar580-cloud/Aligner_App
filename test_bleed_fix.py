"""Does wiring boundary_field into the endpoints actually stop the bleed?"""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch

verts, faces, centres = synthetic_arch(n_teeth=5, spacing=7.0, crown_r=3.2,
                                       grid=200, blend=3.0)
edges = cg.directed_edges(faces)
rng = np.random.default_rng(0)
spiky = verts.copy()
spiky[rng.choice(len(verts), 12, replace=False), 2] -= 3.0
seed = np.array([centres[2][0], centres[2][1], 7.0])

def run(v, field_fn, mode, label):
    conc = field_fn(v)
    d = np.linalg.norm(v[:, :2] - centres[2], axis=1)
    sulcus = conc[(d > 2.8) & (d < 3.6)].mean()
    graph = cg.build_barrier_graph(v, faces, conc, edges=edges, mode=mode)
    dist = cg.geodesic_from_seed(graph, v, seed)
    best = None
    for tol in np.arange(2, 80, 1.0):
        m = cg.mask_from_distance(faces, dist, tol)
        f = m.sum() / len(faces)
        if 0.015 <= f <= 0.10:
            best = (tol, m.sum(), f)
    sel = cg.mask_from_distance(faces, dist, 10.0)
    print(f"  {label:<38} sulcus={sulcus:5.2f}  @tol=10 -> {sel.sum():>6,} faces "
          f"({sel.sum()/len(faces):5.2%})")
    return sel.sum()/len(faces)

old = lambda v: cg.smooth_scalar_fast(cg.vertex_concavity_fast(v, faces, edges=edges),
                                      faces, 2, edges=edges)
new = lambda v: cg.boundary_field(v, faces, edges=edges)

print("CLEAN MESH")
run(verts, old, "linear", "old: max-norm + linear barrier")
run(verts, new, "exponential", "new: boundary_field + exponential")

print("\nWITH 12 OUTLIER SPIKES (what a real scan looks like)")
f_old = run(spiky, old, "linear", "old: max-norm + linear barrier")
f_new = run(spiky, new, "exponential", "new: boundary_field + exponential")

print(f"\n  bleed on noisy mesh: {f_old:.2%} -> {f_new:.2%}  "
      f"({f_old/max(f_new,1e-9):.1f}x tighter)")
assert f_new < f_old, "the fix must reduce bleed on the noisy mesh"
assert f_new <= 0.10, f"selection still too large: {f_new:.2%}"
print("\nPASS  bleed contained on a noisy dense mesh")
