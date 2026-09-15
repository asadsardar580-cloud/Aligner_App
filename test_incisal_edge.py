"""Does the flood cross the sharp incisal edge from facial to lingual?"""
import numpy as np, core_geometry as cg
from incisor_fixture import incisor_on_gingiva

verts, faces, P = incisor_on_gingiva()
edges = cg.directed_edges(faces)
conc = cg.boundary_field(verts, faces, edges=edges)
print(f"Incisor fixture: {len(verts):,} verts / {len(faces):,} faces")

fc = verts[faces].mean(axis=1)
on_crown = (np.abs(fc[:,0]) < P['half_width']) & (np.abs(fc[:,1]) < P['fy'])
facial  = on_crown & (fc[:,1] >  0.4)
lingual = on_crown & (fc[:,1] < -0.4)
gingiva = ~on_crown
print(f"  facial {facial.sum():,} | lingual {lingual.sum():,} | gingiva {gingiva.sum():,}")

# what does the curvature field say at the incisal edge vs the sulcus?
ridge = (np.abs(verts[:,1]) < 0.25) & (np.abs(verts[:,0]) < P['half_width']*0.8)
sulc  = (np.abs(np.abs(verts[:,1]) - P['fy']) < 0.3) & (np.abs(verts[:,0]) < P['half_width']*0.8)
print(f"\n  incisal edge (convex ridge): mean field = {conc[ridge].mean():+.3f}")
print(f"  cervical sulcus (concave)  : mean field = {conc[sulc].mean():+.3f}")
print(f"  -> barrier at edge   = {np.exp(6*max(conc[ridge].mean(),0)):.2f}x")
print(f"  -> barrier at sulcus = {np.exp(6*max(conc[sulc].mean(),0)):.2f}x")

graph = cg.build_barrier_graph(verts, faces, conc, edges=edges)
seed = np.array([0.0, P['fy']*0.55, 0.0])          # mid-facial surface
seed[2] = verts[cg.nearest_vertex(verts, seed)][2]
dist = cg.geodesic_from_seed(graph, verts, seed)

print(f"\n{'tol':>5} {'facial':>8} {'lingual':>8} {'gingiva':>8}   verdict")
print("-"*52)
crossed = None
for tol in [2,4,6,8,10,14,18,25,35,50]:
    m = cg.mask_from_distance(faces, dist, tol)
    f = (m & facial).sum()/max(facial.sum(),1)
    l = (m & lingual).sum()/max(lingual.sum(),1)
    g = (m & gingiva).sum()/max(gingiva.sum(),1)
    verdict = ""
    if l > 0.8 and g < 0.05:
        verdict = "<-- crosses edge, holds at sulcus"
        if crossed is None: crossed = tol
    elif g > 0.10:
        verdict = "bleeding into gingiva"
    print(f"{tol:5} {f:8.0%} {l:8.0%} {g:8.0%}   {verdict}")

if crossed:
    print(f"\nRESULT: the flood DOES cross the incisal edge (from tol={crossed}) "
          f"while the sulcus still holds.")
else:
    print("\nRESULT: the flood never reaches lingual without also bleeding into gingiva.")

# ------------------------------------------------------------------
print("\n" + "="*62)
print("WITH SEED SNAPPED TO THE INCISAL EDGE")
print("="*62)
snapped = cg.snap_seed_to_ridge(verts, conc, seed, search_radius_mm=4.0)
print(f"  seed moved {np.linalg.norm(snapped-seed):.2f}mm -> y={snapped[1]:+.2f} "
      f"(incisal edge at y=0)")
dist2 = cg.geodesic_from_seed(graph, verts, snapped)
print(f"\n{'tol':>5} {'facial':>8} {'lingual':>8} {'gingiva':>8}   verdict")
print("-"*52)
best = None
for tol in [2,4,6,8,10,14,18,25,35]:
    m = cg.mask_from_distance(faces, dist2, tol)
    f = (m & facial).sum()/max(facial.sum(),1)
    l = (m & lingual).sum()/max(lingual.sum(),1)
    g = (m & gingiva).sum()/max(gingiva.sum(),1)
    v = ""
    if f > 0.75 and l > 0.75 and g < 0.05:
        v = "<-- both surfaces, sulcus holds"
        if best is None: best = tol
    print(f"{tol:5} {f:8.0%} {l:8.0%} {g:8.0%}   {v}")
print(f"\nRESULT: snapping gives a working window from tol={best}, "
      f"covering BOTH surfaces with no gingival bleed." if best else
      "\nRESULT: snapping did not open a clean window.")
