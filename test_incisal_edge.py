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


# =========================================================================
# ENFORCING ASSERTIONS (added 2026-09-16)
#
# This file printed a tolerance sweep and exited 0 - PASS in the runner no
# matter what it found. The spec asks that identified incisal points be
# asserted to lie on the occlusal portion of the crown, which is the property
# that actually matters: an "incisal edge" detected halfway down the crown
# would seed every flood in the wrong place.
# =========================================================================

# The incisal edge is the occlusal extreme of the crown. 15% of crown height is
# the band the spec names; the fixture's ridge is a sharp crest so it should sit
# comfortably inside it. HEURISTIC.
OCCLUSAL_BAND_FRACTION = 0.15

_crown_v = np.unique(faces[on_crown])
_z = verts[_crown_v][:, 2]
_z_min, _z_max = float(_z.min()), float(_z.max())
_crown_height = _z_max - _z_min
assert _crown_height > 0.5, f"degenerate crown height {_crown_height:.2f}mm in the fixture"

# The ridge vertices this file identifies as the incisal edge.
_ridge_ids = np.nonzero(ridge)[0]
assert len(_ridge_ids) > 0, "no incisal ridge vertices were identified at all"

_ridge_z = verts[_ridge_ids][:, 2]
_band_floor = _z_max - OCCLUSAL_BAND_FRACTION * _crown_height
_in_band = int((_ridge_z >= _band_floor).sum())
_frac_in_band = _in_band / float(len(_ridge_ids))

assert _frac_in_band >= 0.90, (
    f"only {_frac_in_band*100:.0f}% of identified incisal points lie in the occlusal "
    f"{OCCLUSAL_BAND_FRACTION*100:.0f}% of crown height (z >= {_band_floor:.2f} of "
    f"{_z_min:.2f}..{_z_max:.2f}). An 'incisal edge' detected mid-crown would seed "
    f"every flood in the wrong place.")

# And the sulcus must be at the OTHER end - if these overlap, the field is not
# distinguishing a convex crest from a concave trough at all.
_sulc_ids = np.nonzero(sulc)[0]
if len(_sulc_ids):
    _sulc_z = verts[_sulc_ids][:, 2]
    assert _sulc_z.mean() < _ridge_z.mean(), (
        f"the cervical sulcus (mean z {_sulc_z.mean():.2f}) is not below the incisal "
        f"ridge (mean z {_ridge_z.mean():.2f}) - the two regions are confused")

# The curvature field must have opposite sign at a convex crest and a concave
# trough. If it does not, no tolerance will separate them.
_edge_field = float(conc[ridge].mean())
_sulc_field = float(conc[sulc].mean()) if len(_sulc_ids) else None
if _sulc_field is not None:
    assert _sulc_field > _edge_field, (
        f"the concavity field is {_sulc_field:+.3f} at the sulcus and {_edge_field:+.3f} "
        f"at the incisal crest - a concave trough must score higher than a convex ridge")

print(f"\nPASS  {len(_ridge_ids)} incisal points, {_frac_in_band*100:.0f}% within the "
      f"occlusal {OCCLUSAL_BAND_FRACTION*100:.0f}% band "
      f"(z >= {_band_floor:.2f} of {_z_min:.2f}..{_z_max:.2f}); "
      f"field {_edge_field:+.3f} at crest vs {_sulc_field:+.3f} at sulcus")
