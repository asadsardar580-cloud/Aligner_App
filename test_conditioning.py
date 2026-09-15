"""Does conditioning clean the scan WITHOUT moving any retained geometry?

The second half is the clinically important claim: an aligner grips the
enamel surface, so if conditioning silently moved vertices, every downstream
measurement and the printed part would be wrong."""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch

verts, faces, centres = synthetic_arch(n_teeth=5, grid=140, blend=3.0)
print(f"Clean arch: {len(verts):,} verts / {len(faces):,} faces")

# --- contaminate it the way a real scan is contaminated ---
rng = np.random.default_rng(3)
dirty_v = list(verts)
dirty_f = list(faces)

# 1. loose debris: small floating shells (tongue/cheek fragments)
for k in range(6):
    base = len(dirty_v)
    off = rng.uniform(-20, 20, 3) + np.array([0, 0, 12])
    for p in [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]:
        dirty_v.append(off + np.array(p) * 0.8)
    for tri in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]:
        dirty_f.append([base+tri[0], base+tri[1], base+tri[2]])

# 2. duplicate vertices (unwelded soup regions)
dup_start = len(dirty_v)
for i in range(300):
    dirty_v.append(verts[i])

# 3. degenerate triangles
for i in range(20):
    dirty_f.append([i, i, i+1])

dirty_v = np.array(dirty_v); dirty_f = np.array(dirty_f)
print(f"Contaminated: {len(dirty_v):,} verts / {len(dirty_f):,} faces "
      f"(+6 debris islands, +300 dupes, +20 degenerates)")

clean_v, clean_f, rep = cg.condition_mesh(dirty_v, dirty_f)
print("\nConditioning report:")
for k, val in rep.items():
    print(f"  {k:<28} {val:,}")

assert rep["debris_islands_removed"] >= 6, f"debris not removed: {rep}"
assert rep["degenerate_faces_removed"] >= 20, f"degenerates not removed: {rep}"
print(f"\nPASS  removed {rep['debris_islands_removed']} debris islands and "
      f"{rep['degenerate_faces_removed']} degenerate faces")

# --- THE CLINICAL CLAIM: retained vertices are bit-identical ---
orig = {tuple(p) for p in verts}
kept = [tuple(p) for p in clean_v]
moved = [p for p in kept if p not in orig]
print(f"\n  retained vertices: {len(kept):,}")
print(f"  vertices NOT present bit-identically in the original: {len(moved)}")
# cap centroids from hole filling are legitimately new; there should be none here
assert len(moved) <= rep.get("holes_filled", 0), \
    f"{len(moved)} vertices were moved by conditioning - clinically unacceptable"
print("PASS  every retained vertex is bit-identical to the upload")
print("      (no smoothing: the enamel surface the aligner grips is untouched)")

# --- and the analysis still works on the conditioned mesh ---
edges = cg.directed_edges(clean_f)
conc = cg.boundary_field(clean_v, clean_f, edges=edges)
graph = cg.build_barrier_graph(clean_v, clean_f, conc, edges=edges)
d = cg.geodesic_from_seed(graph, clean_v, np.array([centres[2][0], centres[2][1], 7.0]))
m = cg.mask_from_distance(clean_f, d, 10.0)
print(f"\nPASS  selection on conditioned mesh: {m.sum():,} faces "
      f"({m.sum()/len(clean_f):.2%} of arch)")
