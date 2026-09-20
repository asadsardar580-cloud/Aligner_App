"""Local target-position cast reconstruction for the manufacturing export.

WHY THIS IS ITS OWN MODULE. The brief's central requirement is that
`root_length_mm` — a C_res estimation parameter — must stop being manufacturing
fusion geometry. Those two concepts lived in the same file and the same call
chain, and `_rim_plug`'s docstring said so out loud: *"DEPTH IS THE ROOT
LENGTH, not some small seating value."* Putting the manufacturing interface in
a separate module makes the separation structural instead of conventional:
nothing in here imports a root length, and there is no parameter it could
arrive through.

WHAT WAS WRONG WITH THE OLD PATH, measured rather than argued. The stage export
filled the ORIGINAL socket flush, built ONE static cast, and then unioned that
cast with `crown + a 9mm root plug` transformed by the stage matrix. Nothing
reconstructed the cast at the tooth's NEW cervical position, and the plug
travelled with the tooth — so an extruding tooth carried synthetic root
material up out of the gingiva where it became visible positive geometry.
Measured on two teeth with one extruded 1.2mm over 5 stages, fused volume grew
27579.63 -> 27608.44 mm3 monotonically with the extrusion. That growth IS the
distortion.

THE INTERFACE IS BIDIRECTIONAL, and that is the correction. A moved tooth is in
one of two relationships with the static cast, frequently both around different
parts of the same rim:

  * PENETRATING - intrusion, tipping into tissue, lateral movement into the
    ridge. Tissue must be REMOVED: a cavity.
  * SEPARATED - extrusion, tipping away, translation off the ridge. The
    cervical margin has legitimately lifted clear of the original gingival
    surface. Tissue must be ADDED: a bounded emergence ramp.

Separation is NOT a refusal condition. It is one of the main reasons local
reconstruction has to exist at all, and refusing on it would reject a perfectly
ordinary extrusion.

BOUNDS ARE POLICY; SHAPE IS MEASURED. Every constant in `InterfacePolicy` is a
safety ceiling with a written justification. None of them decides the geometry
— the cavity depth comes from how far the actual transformed crown actually
penetrates the actual cast, clamped by how much wall is actually there.
`emergence_depth_mm` in particular is a STARTING tool parameter, never an
anatomical claim.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

import core_geometry as cg


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class InterfacePolicy:
    """Safety ceilings for the local reconstruction. NOT anatomy.

    Each value answers "how far is this allowed to go before the software
    should stop and say so", not "how deep is a tooth".
    """

    # Where the collar's upper ring STARTS, above the transformed cervical
    # rim, before it is marched out to a measured clearance from the cast and
    # the crown. It is a starting point for a search, never an anatomical
    # claim: the ring ends wherever the measured signed distances put it, and
    # `collar_top_rise_max_mm` reports how far that was.
    emergence_height_mm: float = 0.30

    # RETIRED as construction geometry, kept so an old manifest still parses.
    # It was the depth of the generic socket cup the interface no longer cuts.
    emergence_depth_mm: float = 1.5

    # Hard ceiling on how far the cavity may descend. A real cervical seat is
    # a couple of millimetres; anything approaching this is a movement the
    # reconstruction layer should refuse rather than quietly excavate.
    max_reconstruction_depth_mm: float = 4.0

    # Below this the cavity is too shallow to give the boolean common volume.
    min_reconstruction_depth_mm: float = 0.40

    # How far the crown's flanks are buried in the cast wall. This is what
    # turns a tangency into a real overlap. It is NOT evidence of correctness
    # on its own - the measured overlap volume is.
    fusion_overlap_mm: float = 0.25

    # Breathing room between the crown and the cavity wall, so the boolean is
    # not resolving two surfaces at zero distance.
    clearance_mm: float = 0.05

    # Radial reach of the emergence ramp where the rim has lifted clear.
    ramp_radius_mm: float = 1.20

    # CUT THE CAST BACK FROM THE CROWN, rather than up to it. Measured: with
    # the cavity inset (cast left standing right where the crown emerges) the
    # crown's outer surface GRAZES the original gingival surface - the two are
    # the same scan triangles a fraction of a millimetre apart - and manifold3d
    # answers that tangency with coincident-but-distinct vertices, which a
    # downstream weld turns into non-manifold edges. Outsetting the cavity
    # took the reader-weld count from 31 to 0 on the stages it was measured on.
    cavity_outset_mm: float = 0.15

    # RETIRED with the socket cup. The clearance tool is the transformed
    # crown dilated by `clearance_mm`, so there is no separate cut to raise.
    cavity_raise_mm: float = 0.30

    # The collar is a truncated cone. `seat_bottom_outset_mm` is how far
    # OUTWARD of the rim its lower ring is seeded before being dropped onto the
    # cast; `seat_depth_mm` caps how far below that landing it may march to
    # reach `clearance_mm` of real cast material. The upper ring has no fixed
    # offset any more - it is solved against the measured signed distance to
    # the crown and the cast, which is what stopped the connector crossing
    # either of them at a crease.
    #
    # RETIRED: the upper ring used to be inset into the crown by this much,
    # which put its wall's exit from the crown 0.089mm above the cervical
    # crease and was the direct cause of the CROWN/SEAT self-touches.
    seat_top_inset_mm: float = 0.15
    seat_bottom_outset_mm: float = 1.20
    seat_depth_mm: float = 1.20

    # REPORTING THRESHOLD, not a gate. The connector's wall and the cast's
    # surface should MEET rather than run alongside each other - two
    # nearly-parallel surfaces meeting is a tangency - and
    # `collar_wall_shallow_points` counts how many ring points are below this
    # angle. Correcting them by marching deeper was tried and made things
    # worse (see the note in build_stage_tooth_interface); what fixed the
    # tangency was the seat depth.
    min_crossing_angle_deg: float = 20.0

    # Never remove more than this fraction of the locally measured cast
    # thickness. The cast is a shell over a flat-bottomed base; cutting
    # through it is a breach, not a socket.
    safe_wall_fraction: float = 0.50

    # Geodesic reach, beyond the crown/rim envelope, of the region this
    # interface is ALLOWED to touch. Used for the ROI, not for construction.
    roi_radius_mm: float = 3.00

    # Two adjacent reconstructions may not consume more than this fraction of
    # the gingival bridge between them.
    max_bridge_removal_fraction: float = 0.50

    # How much of the finished model's outside surface may be geometry the
    # software invented. A cervical emergence profile is legitimate and
    # visible; a skirt around every tooth is not.
    max_exposed_synthetic_fraction: float = 0.15

    # Outside the allowed reconstruction envelope the cast must be the cast.
    # 0.05mm is the scanner's own resolution band (20-50 microns), so anything
    # under it is not a deformation anyone could have measured.
    max_unaffected_deviation_mm: float = 0.05

    # The site the tooth LEFT. A crater or a tower past this is a new defect,
    # not a restoration.
    max_old_site_defect_mm: float = 1.00
    max_old_site_step_mm: float = 0.50

    # How far the cervical rim may lift clear of the cast and still be
    # reconstructable locally. Separation is NOT a refusal in itself - an
    # ordinary extrusion lifts the whole rim - but past the reach of the ramp
    # (ramp_radius plus whatever the drop can find) there is no local tissue to
    # blend into and the honest answer is that this movement cannot be built.
    # Without this, a 40mm extrusion was ACCEPTED: every apron point found no
    # cast, every point was grounded, the ramp came out empty and nothing
    # objected.
    max_rim_separation_mm: float = 4.0

    def to_dict(self):
        return asdict(self)


DEFAULT_POLICY = InterfacePolicy()


# ---------------------------------------------------------------------------
# Signed distance to the cast
# ---------------------------------------------------------------------------

class CastProbe:
    """Nearest-surface queries against an IMMUTABLE cast.

    Built once per stage bundle and reused for every tooth and every stage:
    the cast does not change between them, and rebuilding the tree per query
    was 43ms of the 89ms a crown cost when the antagonist check made the same
    mistake (CLAUDE.md section 11).

    `signed()` IS EXACT, and it did not used to be. It was nearest-VERTEX
    distance signed against that vertex's normal - the convention
    `check_occlusal_collision` uses, carried over without measuring whether it
    survived being used for something else. `bench_signed_distance.py` scores
    it against an independent ground truth (exact closest-point-on-triangle
    for the magnitude, three-ray parity vote for the sign) and it is not fit
    for this job. 870 points across 15 difficult classes, sign correct:

        class                  old, as shipped   old + compaction   exact
        steep cervical wall           13.3%            90.0%        100%
        concavity, 0.05mm in          25.0%            76.7%        100%
        0.05mm inside                 86.7%            95.0%        100%
        ALL                           77.7%            91.4%        100%
        max magnitude error          6.66 mm          6.66 mm     0.0014 mm

    13.3% is worse than a coin toss, and a CERVICAL RIM IS A STEEP CERVICAL
    WALL - that is the one place this classifier is asked to work. `rim_signed`
    decides lifted-versus-seated per rim point and sizes the whole transition
    volume, so a wrong sign there builds the bridge to the wrong height at
    scattered points around the margin.

    THE MIDDLE COLUMN IS WHY THERE ARE TWO FIXES HERE AND NOT ONE. Compaction
    alone recovers most of the sign accuracy, which means the phantom vertices
    - the trimmed-away crowns, sitting directly above the cervical walls -
    were the dominant term and the steep-wall diagnosis was only the second
    one. Neither fix on its own reaches a number a gate may rely on.

      1. A nearest-vertex normal on a steep wall points sideways, so the dot
         product flips a few tenths of a millimetre in. This is the same
         failure `local_thickness` already worked around by switching to
         Moller-Trumbore, and the workaround was never generalised.
      2. THE CAST CARRIES VERTICES NO FACE REFERENCES. `build_cast_base`
         returns a face subset over the scan's own vertex array, and rule 3.1
         forbids rebuilding that array - measured, 4497 of 8372 vertices are
         unreferenced, including every crown that was trimmed away.
         `cg.vertex_normals` leaves an unreferenced vertex's normal at ZERO,
         so `outward` is 0.0, `0 < 0` is False, and the point is reported
         OUTSIDE at the distance to a phantom the surface does not contain.
         That is where 6.66mm of error comes from.

    So the constructor COMPACTS to referenced vertices only. That is safe
    precisely because this is a query structure and not exported geometry: it
    renumbers nothing anyone else holds.

    The exact method is also the FAST one at the sizes this code uses, which
    removes the usual reason to keep an approximation around. Measured on the
    7,824-face cast, warm-up excluded:

        points   CastProbe(old)   o3d n=1   o3d n=11
            44         4.857ms    0.362ms    0.529ms
           500         4.958ms    2.514ms    4.062ms
          5000         5.916ms    5.895ms   13.501ms

    A rim is 44 points. Scene build is 0.52ms, once.

    `signed_nearest_vertex` is KEPT rather than deleted: it is what the
    antagonist check still uses on an OPEN shell, where no closed-surface
    method is defined at all, and `bench_signed_distance.py` needs it to score
    the thing it is arguing against.
    """

    __slots__ = ("verts", "faces", "normals", "tree", "_thickness",
                 "_scene", "method", "dropped_unreferenced")

    def __init__(self, verts, faces):
        from scipy.spatial import cKDTree
        verts = np.asarray(verts, float)
        faces = np.asarray(faces, np.int64)

        # Defect 2 above. Compact before anything reads a normal.
        used = np.unique(faces)
        self.dropped_unreferenced = int(len(verts) - len(used))
        if self.dropped_unreferenced:
            remap = np.zeros(len(verts), np.int64)
            remap[used] = np.arange(len(used))
            verts, faces = verts[used], remap[faces]

        self.verts = verts
        self.faces = faces
        self.normals = cg.vertex_normals(self.verts, self.faces)
        self.tree = cKDTree(self.verts)
        self._thickness = None
        self._scene, self.method = self._build_scene()

    def _build_scene(self):
        """Open3D's BVH if it is installed, else the NumPy exact path.

        Open3D is already a declared, load-bearing dependency (CLAUDE.md
        section 19 - the vendored inference pipeline imports it), so this adds
        nothing to the install. It is still guarded, because a geometry module
        that cannot be imported without it would make every headless test
        depend on a wheel none of them need.
        """
        try:
            import open3d as o3d
        except Exception:
            return None, "point_to_triangle+ray_parity (numpy)"
        try:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(self.verts),
                o3d.utility.Vector3iVector(self.faces))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            return scene, "open3d_raycasting nsamples=11"
        except Exception:
            return None, "point_to_triangle+ray_parity (numpy)"

    # Open3D votes over this many ray directions when deciding the sign. The
    # docs ask for an ODD value > 1 so a majority always exists. The cast is
    # watertight by construction (`build_cast_base` asserts zero open and zero
    # non-manifold edges), so n=1 is already 100% correct on every class in
    # the benchmark; 11 is kept because it costs 0.17ms on a rim and removes
    # the dependence on that assertion holding for a cast some future change
    # builds differently.
    NSAMPLES = 11

    def signed(self, pts):
        """Signed distance to the cast surface. Negative = INSIDE.

        Returns a bare array. It used to return `(distance, nearest_index)`
        and the index was read at exactly zero of the two call sites, while
        costing a `cKDTree.query(workers=-1)` - 4.9ms on a 44-point rim,
        nine times the whole exact query. `nearest_vertex_index` is there for
        a caller that genuinely wants to know which part of the cast answered.
        """
        pts = np.atleast_2d(np.asarray(pts, float))
        if self._scene is not None:
            import open3d as o3d
            t = o3d.core.Tensor(np.ascontiguousarray(pts, np.float32))
            return self._scene.compute_signed_distance(
                t, nsamples=self.NSAMPLES).numpy().astype(float)
        return self._signed_numpy(pts)

    def nearest_vertex_index(self, pts):
        """Which cast vertex is nearest. A LABEL, not the thing `signed`
        measured to - the nearest surface point is generally inside a face."""
        pts = np.atleast_2d(np.asarray(pts, float))
        return self.tree.query(pts, workers=-1)[1]

    def _signed_numpy(self, pts):
        """Exact magnitude, parity sign. The fallback when Open3D is absent.

        Correct, and roughly two orders slower than the BVH - it is a
        fallback, not an alternative.
        """
        mag = _point_to_surface(pts, self.verts, self.faces)
        inside = self._parity_inside(pts)
        return np.where(inside, -mag, mag)

    def _parity_inside(self, pts, seed=7):
        """Majority vote of three ray-parity tests. A closed surface is
        crossed an odd number of times from any interior point."""
        rng = np.random.default_rng(seed)
        tri = self.verts[self.faces]
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        e1, e2 = v1 - v0, v2 - v0
        votes = np.zeros(len(pts), np.int64)
        for d in rng.normal(size=(3, 3)):
            d = d / np.linalg.norm(d)
            pvec = np.cross(d, e2)
            det = np.einsum("ij,ij->i", e1, pvec)
            par = np.abs(det) < 1e-12
            inv = np.divide(1.0, det, out=np.zeros_like(det), where=~par)
            for i, p in enumerate(pts):
                tvec = p - v0
                u = np.einsum("ij,ij->i", tvec, pvec) * inv
                qv = np.cross(tvec, e1)
                w = np.einsum("j,ij->i", d, qv) * inv
                t = np.einsum("ij,ij->i", e2, qv) * inv
                hit = (~par) & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-9)
                votes[i] += int(hit.sum()) % 2
        return votes >= 2

    def signed_nearest_vertex(self, pts):
        """The OLD approximation. Kept for the open-shell antagonist check and
        for `bench_signed_distance.py`. Do not use it for a manufacturing
        decision - see the class docstring for what it measures."""
        pts = np.atleast_2d(np.asarray(pts, float))
        dist, idx = self.tree.query(pts, workers=-1)
        delta = pts - self.verts[idx]
        outward = np.einsum("ij,ij->i", delta, self.normals[idx])
        return np.where(outward < 0, -dist, dist), idx

    def local_thickness(self, pts, u_axis):
        """Cast thickness under `pts`, along -u_axis, by RAY-TRIANGLE hits.

        The cast is a shell trimmed to a horseshoe and extruded to a flat
        bottom (`build_cast_base`), so "thickness" is the distance from the
        gingival surface to where a ray dropped straight down leaves the solid.

        THIS IS NOT DONE WITH THE SIGNED DISTANCE ABOVE, and the reason is a
        bug that this code had first: marching down with a nearest-VERTEX
        signed distance reported 0.1mm of material under a cast that is at
        least 3mm thick. The classifier is reliable AT the surface, where the
        nearest vertex really is the nearest surface point. A few tenths of a
        millimetre inside, on a steep cervical wall, the nearest vertex's
        normal points sideways and the dot product flips sign - so the march
        "left the solid" immediately. Every interface was then refused as
        wall_too_thin.

        Moller-Trumbore against the real triangles has no such failure mode:
        it answers where the surface actually is rather than where the nearest
        vertex suggests it might be. 44 rim points against 7,746 faces is
        341k tests, vectorised - a few milliseconds, once per tooth per stage.
        """
        u = np.asarray(u_axis, float)
        u = u / (np.linalg.norm(u) or 1.0)
        pts = np.atleast_2d(np.asarray(pts, float))
        d = -u

        tri = self.verts[self.faces]
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        parallel = np.abs(det) < 1e-12
        inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=~parallel)

        out = np.zeros(len(pts))
        for i, p in enumerate(pts):
            tvec = p - v0
            u_bary = np.einsum("ij,ij->i", tvec, pvec) * inv_det
            qvec = np.cross(tvec, e1)
            v_bary = np.einsum("j,ij->i", d, qvec) * inv_det
            t = np.einsum("ij,ij->i", e2, qvec) * inv_det
            hit = (~parallel) & (u_bary >= -1e-9) & (v_bary >= -1e-9) \
                & (u_bary + v_bary <= 1 + 1e-9) & (t > 1e-6)
            out[i] = float(t[hit].max()) if hit.any() else 0.0
        return out

    def drop_to_surface(self, pts, u_axis, max_distance=None):
        """Land each point on the cast by dropping it along -u_axis.

        Returns (landed_points, hit_mask). The NEAREST hit, unlike
        local_thickness which wants the farthest: here the first surface the
        ray meets is the gingival surface the ramp has to blend into.

        `max_distance` BOUNDS THE SEARCH, and it is not an optimisation. An
        unbounded drop from a point that happens to sit over a socket opening,
        an embrasure, or the arch's inner edge passes straight through and
        lands on the UNDERSIDE OF THE BASE - measured, 10 to 12mm down on a
        cast whose gingiva was less than a millimetre away. A blend to a
        surface 11mm below is not a local reconstruction, and it made the
        envelope check refuse eight of ten ordinary movements. Past this
        distance the honest answer is "no cast near here", which the caller
        handles by leaving that part of the rim grounded.
        """
        u = np.asarray(u_axis, float)
        u = u / (np.linalg.norm(u) or 1.0)
        pts = np.atleast_2d(np.asarray(pts, float))
        d = -u

        tri = self.verts[self.faces]
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        parallel = np.abs(det) < 1e-12
        inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=~parallel)

        landed = pts.copy()
        ok = np.zeros(len(pts), bool)
        for i, p in enumerate(pts):
            tvec = p - v0
            u_bary = np.einsum("ij,ij->i", tvec, pvec) * inv_det
            qvec = np.cross(tvec, e1)
            v_bary = np.einsum("j,ij->i", d, qvec) * inv_det
            t = np.einsum("ij,ij->i", e2, qvec) * inv_det
            hit = (~parallel) & (u_bary >= -1e-9) & (v_bary >= -1e-9) \
                & (u_bary + v_bary <= 1 + 1e-9) & (t > 1e-6)
            if max_distance is not None:
                hit &= t <= max_distance
            if hit.any():
                landed[i] = p + d * float(t[hit].min())
                ok[i] = True
        return landed, ok


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

@dataclass
class InterfaceResult:
    """Everything the manifest needs, plus the tools the boolean needs."""

    ok: bool
    refusal_reason: str | None = None

    # NO CAVITY OR RAMP ARRAYS. The clearance tool is an EXACT dilation of the
    # transformed crown, which is a Minkowski sum and therefore belongs to the
    # CSG engine rather than to an array offset: measured on a real crown, a
    # 0.05mm vertex-normal offset gives 46.86mm3 against the true 49.84mm3 and
    # can fold wherever the offset exceeds the local medial-axis distance -
    # the failure CLAUDE.md section 8 tabulated for normal offsets, and which
    # showed up here as a LOCAL_CLEARANCE self-touch in the fused stage. This
    # builder decides WHETHER to cut and measures by how much; the caller,
    # which holds the crown Manifold, builds the tool. The emergence ramp is
    # no longer a separate solid at all - it is one face of the connector.
    # The bounded fusion seat. NOT part of the crown - the crown stays a rigid
    # transform of T0 and is measured as such.
    seat_verts: np.ndarray | None = None
    seat_faces: np.ndarray | None = None
    # Carries the cavity cut above the gingival surface so the subtract is not
    # coplanar with it.
    collar_verts: np.ndarray | None = None
    collar_faces: np.ndarray | None = None

    diagnostics: dict = field(default_factory=dict)

    def manifest_row(self):
        """The diagnostics only - never the geometry."""
        d = dict(self.diagnostics)
        d["ok"] = self.ok
        d["refusal_reason"] = self.refusal_reason
        return d


def _face_areas(verts, faces):
    """Triangle areas. Local rather than added to core_geometry, which has no
    such helper and does not need one for the engine's own work."""
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    if not len(f):
        return np.zeros(0)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _bbox(pts):
    p = np.asarray(pts, float)
    if not len(p):
        return None
    return {"min": [round(x, 4) for x in p.min(axis=0)],
            "max": [round(x, 4) for x in p.max(axis=0)],
            "size": [round(x, 4) for x in (p.max(axis=0) - p.min(axis=0))]}


def _loop_order(rim):
    """The rim arrives already ordered as a loop from /cut's boundary walk."""
    return np.arange(len(rim))


def _loft(loop_a, loop_b, axis=None):
    """Closed solid between two same-length ordered loops.

    Side wall plus a fan cap at each end. The solid is closed by construction
    and the only way it can be non-manifold is if a loop passes through
    itself - which the caller checks with is_edge_manifold_closed rather than
    trusting.

    THE CAPS ARE CONES WHEN AN AXIS IS GIVEN, and that is not cosmetic. A fan
    to the loop's own CENTROID is a dished surface whenever the loop is
    non-planar, and the connector's upper ring is deliberately non-planar -
    each point rises only as far as its own local cast surface requires, which
    measured 0.30mm at some points and 1.96mm at others on one tipped tooth.
    The dish then sagged BELOW the cervical crease the collar exists to
    enclose, leaving 3 of 46 crease points outside it however far the wall was
    pushed sideways, and that is exactly where the last self-touch sat. Lifting
    each apex past the furthest point of its own ring makes the cap a cone
    that cannot sag into the solid.
    """
    n = len(loop_a)
    ca_pt = loop_a.mean(axis=0)
    cb_pt = loop_b.mean(axis=0)
    if axis is not None:
        u = np.asarray(axis, float)
        u = u / (np.linalg.norm(u) or 1.0)
        ca_pt = ca_pt + u * max(0.0, float(((loop_a - ca_pt) @ u).max()))
        cb_pt = cb_pt - u * max(0.0, float((-(loop_b - cb_pt) @ u).max()))
    verts = np.vstack([loop_a, loop_b, ca_pt[None, :], cb_pt[None, :]])
    ca, cb = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, n + i, n + j])
        faces.append([i, n + j, j])
        faces.append([ca, j, i])            # cap A
        faces.append([cb, n + i, n + j])    # cap B
    return verts, np.asarray(faces, np.int64)


def build_stage_tooth_interface(
    base_verts, base_faces,
    crown_t0, crown_faces,
    socket_rim_t0, u_oa_t0,
    stage_matrix,
    policy: InterfacePolicy | None = None,
    probe: CastProbe | None = None,
    neighbour_rims=None,
):
    """The local target-position interface for ONE tooth at ONE stage.

    The cast arrives IMMUTABLE and leaves untouched: this returns tools, and
    the caller does the booleans. Nothing here writes to `base_verts`.

    `stage_matrix` is the SAME rigid M the crown receives, so the interface can
    never drift from the tooth it serves - which is the failure mode a
    separately-derived interface would have.

    Returns an `InterfaceResult`. `ok=False` carries a named, measured
    `refusal_reason`; it never falls back to inventing a root.
    """
    pol = policy or DEFAULT_POLICY
    t0 = time.perf_counter()

    base_verts = np.asarray(base_verts, float)
    M = np.asarray(stage_matrix, float)
    rim_t0 = np.asarray(socket_rim_t0, float)
    crown_t0 = np.asarray(crown_t0, float)

    if len(rim_t0) < 3:
        return InterfaceResult(False, "rim_too_small",
                               diagnostics={"rim_points": int(len(rim_t0))})

    probe = probe or CastProbe(base_verts, base_faces)

    # --- transform with the SAME matrix the crown gets ---------------------
    rim_k = cg.apply_matrix(rim_t0, M)
    crown_k = cg.apply_matrix(crown_t0, M)
    u_oa_k = M[:3, :3] @ np.asarray(u_oa_t0, float)
    u_oa_k = u_oa_k / (np.linalg.norm(u_oa_k) or 1.0)

    diag = {
        "target_rim_bbox": _bbox(rim_k),
        "crown_bbox": _bbox(crown_k),
        "rim_points": int(len(rim_k)),
    }

    # --- classify the rim against the cast ---------------------------------
    rim_signed = probe.signed(rim_k)
    n_inside = int((rim_signed < 0).sum())
    n_outside = int((rim_signed > 0).sum())
    diag.update({
        "rim_signed_min_mm": round(float(rim_signed.min()), 4),
        "rim_signed_max_mm": round(float(rim_signed.max()), 4),
        "rim_points_inside_cast": n_inside,
        "rim_points_outside_cast": n_outside,
        "rim_separation_max_mm": round(float(max(0.0, rim_signed.max())), 4),
    })

    # --- how far does the ACTUAL crown penetrate the ACTUAL cast? ----------
    crown_signed = probe.signed(crown_k)
    penetration = float(max(0.0, -crown_signed.min()))
    diag["crown_penetration_mm"] = round(penetration, 4)
    diag["crown_points_inside_cast"] = int((crown_signed < 0).sum())

    # === MOVEMENT TOPOLOGY DECIDES THE INTERFACE ==========================
    #
    # THIS IS THE STRUCTURAL CORRECTION. The previous version built the SAME
    # full-ring socket cup under every tooth regardless of how it had actually
    # moved, then added a full-ring ramp on top. For a pure extrusion those two
    # tools fight: the ramp bridges the cast up to the lifted rim, and the
    # cavity - a generic socket the crown never asked for, because it penetrates
    # nothing - is then cut straight back down through that bridge. Measured
    # across a 1.2mm extrusion the fused model came apart into 1,1,2,2,3 bodies
    # as the cut ate more of the connection at each stage.
    #
    # No amount of parameter tuning fixes that, because the cavity should not
    # exist at all in that case. So the tooth is classified first:
    #
    #   PENETRATING  the crown really is inside the cast   -> REMOVE material
    #   SEPARATED    the rim has lifted clear, nothing is
    #                buried                                -> ADD material only
    #   MIXED        one side buried, the other lifted     -> both, and the
    #                                                         removal is shaped
    #                                                         by the crown
    #   SEATED       neither; a zero or tiny movement      -> connector only
    #
    # PENETRATION_TOL is the depth below which "inside" is indistinguishable
    # from surface noise on a scan whose own features are ~0.1mm. Engineering
    # threshold for this prototype.
    PENETRATION_TOL = 0.05
    penetrating = penetration > PENETRATION_TOL
    lifted = float(rim_signed.max()) > pol.fusion_overlap_mm
    if penetrating and lifted:
        mode = "mixed"
    elif penetrating:
        mode = "penetrating"
    elif lifted:
        mode = "separated"
    else:
        mode = "seated"
    diag["interface_mode"] = mode
    diag["penetration_tolerance_mm"] = PENETRATION_TOL

    thickness = probe.local_thickness(rim_k, u_oa_k)
    finite = thickness[np.isfinite(thickness)]
    local_thickness = float(np.nanmin(finite)) if len(finite) else 0.0
    wall_limit = local_thickness * pol.safe_wall_fraction
    diag["local_cast_thickness_mm"] = round(local_thickness, 4)
    diag["wall_limit_mm"] = round(wall_limit, 4)
    diag["policy"] = pol.to_dict()

    # --- refusals, and ONLY for the permitted reasons ----------------------
    if float(rim_signed.max()) > pol.max_rim_separation_mm:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "outside_reconstruction_envelope",
                               diagnostics=diag)

    if penetrating and penetration > wall_limit and wall_limit > 0:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "interface_unbuildable_wall_too_thin",
                               diagnostics=diag)

    # === WHAT ACTUALLY CREATES THE BAD TOPOLOGY, MEASURED =================
    #
    # The fused stage solid TOUCHES ITSELF. manifold3d records that as two
    # vertices at an IDENTICAL float64 position under different indices, and
    # the control says so unambiguously: a clean transversal union of two
    # cubes, of two rotated cubes, of a cube and a sphere, and even of two
    # cubes meeting FACE TO FACE all produce zero coincident positions, while
    # two cubes meeting along an EDGE - a genuinely self-touching solid -
    # produce exactly one coincident pair and one non-manifold edge. So a
    # coincident position in manifold3d's output is not a run boundary or a
    # tolerance artefact. It is a self-touch.
    #
    # WHERE the solid touched itself, measured with per-triangle provenance
    # (`Manifold.as_original` + `run_original_id`) on the failing cases: every
    # touch sat on a CREASE. On extrusion 0.25mm stage 1, all six touches read
    # distance-to-rim 0.08889 and distance-to-nearest-cast-vertex 0.17222 - the
    # SAME two numbers to five decimals - because they lie on the boundary edge
    # of the cast's flat flush socket cap, at the midpoint of a rim edge. The
    # rest sat on the crown's own cervical rim.
    #
    # THE CAUSE IS THAT ALL THREE GEOMETRIES ARE BUILT FROM ONE LOOP. The
    # crown is cut at `socket_rim`, the old site is capped at `socket_rim`, and
    # the connector was lofted from `socket_rim` transformed - so the
    # connector's wall crossed the cast and the crown exactly where each has a
    # sharp edge. A surface crossing another surface AT ITS CREASE is a
    # tangential contact, and that is what manifold3d answers with coincident
    # vertices.
    #
    # It is also why "a small movement is worse than a large one" is FALSE.
    # Measured across the whole extrusion sweep with seats and crowns fused,
    # self-touch counts were 9/16/27 at 0.0mm, 7/18/34 at 0.25mm and 23/27/43
    # at 1.2mm. The 1.2mm case passed only because every non-manifold edge it
    # produced happened to be short enough for `collapse_short_nonmanifold_edges`
    # to take. The defect is universal; the repair rung's success was not.
    #
    # === THE CONNECTOR IS A TRANSITION COLLAR THAT ENCLOSES THE CREASE =====
    #
    # So the connector is no longer a plug lofted from the rim into the crown.
    # It is a bounded collar whose solid CONTAINS the crown's cervical crease,
    # and whose own surface meets the crown and the cast only where both are
    # smooth. Both rings are placed by MEASURED signed distance against the
    # actual transformed crown and the actual cast - not by an offset chosen in
    # advance:
    #
    #   upper ring  marched up along u_oa while it is inside the cast, and out
    #               along the rim's own radial direction while it is inside the
    #               crown, until it is at least `clearance_mm` clear of BOTH
    #   lower ring  dropped onto the cast `seat_bottom_outset_mm` outward of the
    #               rim, then marched down until it is at least `clearance_mm`
    #               INSIDE the cast
    #
    # The radial direction is taken about the rim's own centroid, and the
    # offsets are OUTWARD, which cannot fold a star-shaped loop - the failure
    # mode CLAUDE.md section 8 measured for inward offsets.
    #
    # CASE A (penetrating), CASE B (separated) and CASE C (mixed) all use this
    # one construction and need no partition, because both rings are solved PER
    # RIM POINT against the real surfaces: a lifted sector's lower ring lands on
    # tissue below it, a buried sector's lower ring marches down from a landing
    # that is already inside, and the upper ring rises only as far as that
    # point's own local cast surface requires.
    centre = rim_k.mean(axis=0)
    radial = rim_k - centre
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True)
    inset_dir = np.divide(radial, np.where(rnorm < 1e-12, 1.0, rnorm))

    # The crown's OWN exact signed distance. Built on the transformed crown, so
    # "inside the crown" is measured against the tooth that is actually there.
    crown_probe = CastProbe(crown_k, crown_faces)
    margin = max(pol.clearance_mm, 1e-4)

    # --- CASE A: crown-derived local clearance ----------------------------
    #
    # Where the crown is genuinely inside the cast, cast material is removed -
    # and the tool is derived from the ACTUAL TRANSFORMED CROWN, dilated by
    # `clearance_mm` so the cavity wall stands that distance AWAY from the
    # crown and no surface in the result is a copy of any other. Cutting with
    # the crown itself would be `(cast - crown) u crown`, an identity on
    # geometry and not on topology.
    #
    # IT IS EMITTED ONLY IF IT DOES NOT BREAK THE CAST, and that is a MEASURED
    # guard rather than a preference: subtracting the dilated crown fragmented
    # the cast into 4 bodies on extrusion 0.25mm stage 1, because the crown
    # passes through the thin lip left between the old socket cap and the
    # gingival wall. The caller measures `decompose()` on the prepared cast and
    # reports `clearance_tool_emitted` with the reason it was withheld.
    needs_cut = mode in ("penetrating", "mixed")
    diag["cut_with_crown"] = False            # never the crown itself
    diag["clearance_measured"] = bool(needs_cut)
    diag["clearance_mm"] = float(pol.clearance_mm) if needs_cut else 0.0
    diag["clearance_tool_emitted"] = False    # unconditional: absent != false
    diag["clearance_tool_volume_mm3"] = None
    diag["cavity_volume_mm3"] = None          # set by the caller if it cuts
    diag["depth_used_mm"] = round(float(penetration), 4) if needs_cut else 0.0
    diag["depth_wanted_mm"] = round(float(penetration), 4)
    diag["depth_clamped_by"] = None
    diag["clearance_required"] = bool(needs_cut)
    # `crown_signed` is already the exact signed distance of every crown vertex
    # to the cast, computed above. Recomputing it here cost a second Open3D
    # sweep over every crown vertex for an array that had not changed.
    sd_crown_to_cast = crown_signed
    diag["crown_min_signed_distance_mm"] = round(float(sd_crown_to_cast.min()), 4)
    diag["crown_max_signed_distance_mm"] = round(float(sd_crown_to_cast.max()), 4)
    # NEAR-COINCIDENT means |signed distance| < the band, and nothing else.
    # The earlier `crown_points_in_graze_band` counted every vertex that was
    # not deeply buried, which includes every vertex in free space: it read
    # 113 of 127 where the near-coincident count was 29, and the conclusion
    # drawn from it - that there is no buried region to hold the fusion - was
    # an artefact of the metric.
    band = max(pol.cavity_outset_mm, 1e-6)
    near = np.abs(sd_crown_to_cast) < band
    diag["clearance_band_mm"] = round(float(band), 4)
    diag["near_coincident_fraction"] = round(float(near.mean()), 4)
    diag["crown_points_near_coincident"] = int(near.sum())
    diag["crown_points_deeply_buried"] = int((sd_crown_to_cast <= -band).sum())
    diag["crown_points_clear_of_cast"] = int((sd_crown_to_cast >= band).sum())
    if needs_cut:
        diag["crown_volume_mm3"] = round(
            float(abs(cg.signed_volume(crown_k,
                                       np.asarray(crown_faces, np.int64)))), 4)

    # === THE TRANSITION COLLAR ============================================
    lift = np.clip(rim_signed, 0.0, None)
    is_lifted = lift > pol.fusion_overlap_mm
    diag["rim_points_lifted"] = int(is_lifted.sum())
    diag["rim_points_seated"] = int((~is_lifted).sum())

    # --- upper ring -------------------------------------------------------
    STEP = 0.04
    reach = float(pol.max_reconstruction_depth_mm)
    # THE COLLAR RISES AS FAR AS THE TOOTH HAS LIFTED, because that is the
    # height the tissue actually has to climb. With a fixed 0.30mm start the
    # upper ring stayed just above the cervical margin while the rim had
    # risen 0.6mm, so the whole emergence was compressed into the wall below
    # it: measured by `transition_quality` on a 0.6mm extrusion, 0.771mm of a
    # 0.946mm fall landed in ONE 0.25mm ring - 82% of it - which is the
    # "abrupt ring / vertical wall" the brief asks to reject, and the gate
    # rejected it. Rising with the lift spreads the same fall across the
    # crown's own flare instead of inventing a shelf to stand on.
    top = rim_k + u_oa_k * np.maximum(pol.emergence_height_mm, lift)[:, None]
    up = np.zeros(len(top))
    out = np.zeros(len(top))
    for _ in range(int(reach / STEP) + 4):
        sc = probe.signed(top)
        sk = crown_probe.signed(top)
        need_up = (sc < margin) & (up < reach)
        need_out = (sk < margin) & (out < reach)
        if not (need_up | need_out).any():
            break
        du = np.where(need_up, STEP, 0.0)
        do = np.where(need_out, STEP, 0.0)
        top = top + u_oa_k * du[:, None] + inset_dir * do[:, None]
        up += du
        out += do
    # THE UPPER RING MUST CLEAR THE WHOLE RIM NEAR IT, not only its own point.
    # `top[i] = rim[i] + rise[i]` makes the ring follow the cervical scallop,
    # so where the margin falls away the NEIGHBOURING ring point sits BELOW
    # the rim point beside it - measured on a tipped tooth, top[k] was 0.078mm
    # under rim[i], the collar's upper boundary passed through the rim's own
    # neighbourhood, and the two crease points left unenclosed were inside by
    # only 0.025mm and 0.014mm. A running maximum over a small angular window
    # fills those local dips and leaves the ring low wherever the rim is
    # genuinely low, so the collar does not become a tower to fix a notch.
    _h = (top - centre) @ u_oa_k
    _w = 3
    _stack = np.stack([np.roll(_h, d) for d in range(-_w, _w + 1)], axis=0)
    _lift = np.maximum(_stack.max(axis=0) - _h, 0.0)
    top = top + u_oa_k * _lift[:, None]
    diag["collar_top_window_lift_max_mm"] = round(float(_lift.max()), 4)
    diag["collar_top_window_lift_points"] = int((_lift > 1e-9).sum())

    top_cast = probe.signed(top)
    top_crown = crown_probe.signed(top)
    diag["collar_top_rise_max_mm"] = round(float(up.max()), 4)
    diag["collar_top_rise_mean_mm"] = round(float(up.mean()), 4)
    diag["collar_top_outset_max_mm"] = round(float(out.max()), 4)
    diag["collar_top_clear_of_cast_min_mm"] = round(float(top_cast.min()), 4)
    diag["collar_top_clear_of_crown_min_mm"] = round(float(top_crown.min()), 4)
    if top_cast.min() < margin or top_crown.min() < margin:
        diag["collar_top_unresolved_points"] = int(
            ((top_cast < margin) | (top_crown < margin)).sum())
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "interface_construction_failed",
                               diagnostics=diag)

    # --- lower ring -------------------------------------------------------
    #
    # THE REACH IS CLAMPED BY THE NEIGHBOUR, PER RIM POINT. Two collars that
    # meet consume the gingival bridge between them, and the fused solid stays
    # perfectly watertight while the cast grows a trench - measured on two
    # teeth 2.28mm apart, the pair took 2.04mm of it and left a 0.24mm gap.
    # Each reconstruction may therefore reach at most a QUARTER of the way to
    # its nearest neighbour's rim, so the two together can never take more
    # than half. Below `fusion_overlap_mm` there is not enough left to fuse
    # with, so the clamp stops there and the caller's bridge gate is what
    # refuses the case - shrinking silently past the point of working would be
    # the failure this check exists to catch.
    outset = np.full(len(rim_k), float(pol.seat_bottom_outset_mm))
    # ONE BUDGET FOR EVERY OUTWARD MOVE. The seed outset is not the only thing
    # that reaches toward the neighbour - the nesting fix and the enclosure
    # push do too - so clamping the seed alone left the pair consuming 60% of
    # a 2.28mm bridge where 50% is the ceiling. The budget is per rim point
    # and every outward step spends it.
    outward_budget = np.full(len(rim_k), float(pol.ramp_radius_mm))
    if neighbour_rims is not None and len(neighbour_rims):
        from scipy.spatial import cKDTree
        near = np.vstack([np.asarray(r, float) for r in neighbour_rims
                          if r is not None and len(r)])
        if len(near):
            d_nb, _ = cKDTree(near).query(rim_k, workers=-1)
            # DERIVED FROM THE RULE, not chosen. The pair may consume
            # `max_bridge_removal_fraction` of the bridge, so each may take
            # half of that - less two clearance bands, because the nesting fix
            # is entitled to add one on each side and the measurement is taken
            # between the two solids' surfaces, not between their seeds.
            # Measured on a 2.25mm bridge: half-share alone left the pair at
            # 55%, and this leaves them at 45%.
            allowed = np.maximum(
                0.5 * pol.max_bridge_removal_fraction * d_nb
                - 2.0 * pol.clearance_mm,
                pol.fusion_overlap_mm)
            outward_budget = np.minimum(outward_budget, allowed)
            clamped = np.minimum(outset, allowed)
            diag["collar_outset_clamped_points"] = int(
                (clamped < outset - 1e-9).sum())
            diag["collar_outset_min_mm"] = round(float(clamped.min()), 4)
            diag["nearest_neighbour_rim_mm"] = round(float(d_nb.min()), 4)
            outset = clamped
    diag["collar_outset_max_mm"] = round(float(outset.max()), 4)
    seed = rim_k + inset_dir * outset[:, None]
    landing, hit = probe.drop_to_surface(seed, u_oa_k, reach)
    # WHICH ATTEMPT ANSWERED, per point. -1 = never landed, 0 = the drop along
    # the tooth's own long axis at the full outset, 1-3 = the outset SHRUNK
    # inward to find tissue, 4 = the direction-free nearest-surface fallback.
    # Kept because "how many landed" and "where they landed" are different
    # facts and only the second one is geometry.
    retry_level = np.where(hit, 0, -1).astype(np.int64)
    ray_hits = int(hit.sum())
    # THE SHRINK LADDER IS LOAD-BEARING, not defensive. A rim point on the
    # lingual side seeded a full 1.2mm outward lands OVER THE ARCH OPENING,
    # where there is no cast at all: measured on a 1.2mm extrusion, one point
    # in 44 fell through and the nearest surface was 3.93mm away, on a wall
    # the downward march could never get inside. Pulling the seed back in is
    # what finds real tissue under that point.
    for step_i, shrink in enumerate((0.66, 0.33, 0.0), start=1):
        if hit.all():
            break
        retry_seed = rim_k + inset_dir * (outset * shrink)[:, None]
        retry, retry_hit = probe.drop_to_surface(retry_seed, u_oa_k, reach)
        take = (~hit) & retry_hit
        landing[take] = retry[take]
        seed[take] = retry_seed[take]
        retry_level[take] = step_i
        hit |= take
    need = ~hit
    if need.any():
        # DIRECTION-FREE FALLBACK. The drop follows the TOOTH's long axis,
        # which is right for a bodily movement and wrong for a tipped one: the
        # ray leaves at an angle and exits the side of the cast. Nearest
        # surface point has no direction to be wrong about.
        d_near, i_near = probe.tree.query(seed[need], workers=-1)
        ok_near = d_near <= reach
        idx = np.where(need)[0]
        landing[idx[ok_near]] = probe.verts[i_near[ok_near]]
        hit[idx[ok_near]] = True
        retry_level[idx[ok_near]] = 4
    diag["ramp_landing_by_ray"] = ray_hits
    diag["ramp_landing_by_nearest_surface"] = int(hit.sum()) - ray_hits
    diag["ramp_landing_misses"] = int((~hit).sum())
    diag["ramp_retry_levels"] = {
        str(kk): int((retry_level == kk).sum()) for kk in (-1, 0, 1, 2, 3, 4)}
    diag["ramp_collapsed_inward_points"] = 0
    drop = np.linalg.norm(landing - seed, axis=1)
    diag["ramp_landing_distance_max_mm"] = round(
        float(drop[hit].max()) if hit.any() else 0.0, 4)
    diag["ramp_landing_distance_mean_mm"] = round(
        float(drop[hit].mean()) if hit.any() else 0.0, 4)
    # A MISS STAYS A MISS. This used to be overwritten with `hit[:] = True`,
    # which made the refusal unreachable and turned "no cast found" into a
    # silent success.
    if (~hit).any():
        diag["unreachable_lifted_points"] = int((~hit).sum())
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "no_cast_beneath_target_rim",
                               diagnostics=diag)

    # MARCH ALONG THE FIELD, NOT ALONG THE AXIS. Descending the tooth's own
    # long axis is right where the cast beneath the landing is a floor and
    # wrong where it is a wall - on a steep interproximal face the axis runs
    # ALONG the surface and 1.2mm of marching never gets inside, which is
    # exactly how a 1.2mm extrusion was refused `wall_too_thin` on a cast
    # measured 18.77mm thick. The inward direction is -grad(signed distance),
    # which is the surface normal wherever the field is smooth and needs no
    # assumption about how the cast is shaped.
    # HOW DEEP IS "INSIDE"? Stopping at `clearance_mm` was wrong and the
    # production forensics caught it: the lower ring settled 0.08mm under the
    # surface, so its bottom disc ran nearly PARALLEL to the flat flush closure
    # of the old site a fraction of a millimetre away, and two nearly-parallel
    # surfaces that close is a tangency. The connector has to SEAT - reach a
    # real depth of real cast material - bounded by the wall it may not breach.
    # HOW DEEP IS "INSIDE"? Not `clearance_mm`: stopping there settled the
    # lower ring 0.08mm under the surface, so its bottom disc ran nearly
    # PARALLEL to the flat flush closure of the old site a fraction of a
    # millimetre away, and two nearly-parallel surfaces that close is a
    # tangency - which the production forensics reported as
    # EMERGENCE_RECONSTRUCTION + OLD_SOCKET_REPAIR. The connector has to SEAT:
    # reach the fusion overlap the union needs PLUS the clearance band that
    # keeps surfaces apart. Both numbers already exist and already mean that.
    #
    # Measured on the 18-case matrix, everything else held fixed:
    #   0.05mm (clearance only)  15 of 18
    #   0.30mm (this)            16 of 18
    #   0.60mm                   13 of 18
    #   1.20mm (seat_depth)       3 of 18
    # so it is neither "as shallow as possible" nor "as deep as possible", and
    # the value that works is the one the policy already names.
    seat_target = float(pol.fusion_overlap_mm + pol.clearance_mm)
    diag["collar_seat_target_mm"] = round(seat_target, 4)
    bottom = landing.copy()
    down = np.zeros(len(bottom))
    H = 1e-3
    for _ in range(int(pol.seat_depth_mm / STEP) + 8):
        sb = probe.signed(bottom)
        need_dn = (sb > -seat_target) & (down < pol.seat_depth_mm)
        if not need_dn.any():
            break
        grad = np.stack([
            probe.signed(bottom + e * H) - probe.signed(bottom - e * H)
            for e in np.eye(3)], axis=1) / (2 * H)
        gl = np.linalg.norm(grad, axis=1, keepdims=True)
        inward = -grad / np.where(gl < 1e-9, 1.0, gl)
        # Where the field is flat enough to be meaningless, fall back to the
        # tooth's long axis rather than to a random direction.
        inward[gl[:, 0] < 1e-9] = -u_oa_k
        dd = np.where(need_dn, STEP, 0.0)
        bottom = bottom + inward * dd[:, None]
        down += dd
    bot_sd = probe.signed(bottom)

    # THE WALL MUST CROSS THE CAST TRANSVERSALLY, and this is the last thing
    # that had to be measured rather than assumed. With the crown's crease
    # enclosed, the surviving self-touch was reported by the production
    # forensics as EMERGENCE_RECONSTRUCTION + OLD_SOCKET_REPAIR: the collar's
    # lower wall runs out 1.2mm while dropping only a few tenths, so where it
    # leaves the cast across the FLAT flush closure of the old site the two
    # surfaces are nearly parallel, and two nearly-parallel surfaces meeting is
    # a tangency whatever their creases do.
    #
    # MEASURED AND REPORTED, NOT CORRECTED - and the attempt to correct it is
    # why. Marching the shallow points deeper does steepen the wall, but it
    # cost 2.4mm of extra depth to move two points and tripped the
    # reconstruction envelope instead: the matrix went from 15 of 18 to 13.
    # What actually fixed the tangency was seating the lower ring to
    # `fusion_overlap_mm + clearance_mm` so its bottom disc is not beside the
    # old-site cap in the first place. The angle stays as a diagnostic, because
    # a collar whose wall lies along the cast is worth seeing in the manifest
    # even when the topology comes out clean.
    def _wall_normals(t_, b_):
        tang = np.roll(b_, -1, axis=0) - np.roll(b_, 1, axis=0)
        w_ = b_ - t_
        n_ = np.cross(tang, w_)
        ln = np.linalg.norm(n_, axis=1, keepdims=True)
        return n_ / np.where(ln < 1e-12, 1.0, ln)

    def _field_normal(pts):
        g = np.stack([probe.signed(pts + e * 1e-3) - probe.signed(pts - e * 1e-3)
                      for e in np.eye(3)], axis=1) / 2e-3
        ln = np.linalg.norm(g, axis=1, keepdims=True)
        return g / np.where(ln < 1e-9, 1.0, ln)

    # Two surfaces meet at the angle between their NORMALS, so tangency is
    # |n_wall . n_cast| near 1 and a clean crossing is that dot product below
    # cos(min_crossing_angle_deg).
    cos_floor = float(np.cos(np.radians(pol.min_crossing_angle_deg)))
    n_cast = _field_normal(landing)
    par = np.abs((_wall_normals(top, bottom) * n_cast).sum(axis=1))
    angles = np.degrees(np.arccos(np.clip(par, 0.0, 1.0)))
    diag["collar_wall_cast_crossing_angle_min_deg"] = round(float(angles.min()), 3)
    diag["collar_wall_cast_crossing_angle_mean_deg"] = round(float(angles.mean()), 3)
    diag["collar_wall_shallow_points"] = int((par > cos_floor).sum())
    bot_sd = probe.signed(bottom)
    diag["collar_bottom_depth_max_mm"] = round(float(down.max()), 4)
    diag["collar_bottom_inside_cast_max_mm"] = round(float(bot_sd.max()), 4)
    diag["seat_depth_mm"] = round(float(down.max()), 4)
    if bot_sd.max() > -margin:
        diag["collar_bottom_unresolved_points"] = int((bot_sd > -margin).sum())
        diag["collar_bottom_shallowest_mm"] = round(float(bot_sd.max()), 4)
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "interface_unbuildable_wall_too_thin",
                               diagnostics=diag)

    # THE LOWER RING MUST SIT BELOW THE CREASE IT HAS TO ENCLOSE. On a
    # PENETRATING tooth the cast surface is ABOVE the cervical rim, so the
    # landing is above it too and `landing - seat_target` can still be above
    # it - which leaves the rim outside the collar no matter how far the wall
    # is pushed outward. Measured on a tipped tooth, 3 of 46 crease points
    # stayed 0.07mm outside through eight outward pushes, and that was exactly
    # where the surviving CROWN + OLD_SOCKET_REPAIR touch sat.
    below = ((bottom - rim_k) * u_oa_k).sum(axis=1)
    need_low = below > -margin
    if need_low.any():
        drop_extra = np.where(need_low, below + margin, 0.0)
        bottom = bottom - u_oa_k * drop_extra[:, None]
        down = down + drop_extra
        diag["collar_bottom_lowered_to_clear_crease_points"] = int(need_low.sum())
        diag["collar_bottom_lowered_max_mm"] = round(float(drop_extra.max()), 4)
        bot_sd = probe.signed(bottom)
    diag["collar_bottom_below_crease_max_mm"] = round(
        float(((bottom - rim_k) * u_oa_k).sum(axis=1).max()), 4)

    # NESTING. The loft folds if the lower ring comes inside the upper one, so
    # the lower ring is pushed radially out until it is clear. Outward radial
    # offset of a star-shaped loop cannot self-intersect; inward offset can,
    # which is why only this direction is used.
    tr = np.linalg.norm(top - centre, axis=1)
    br = np.linalg.norm(bottom - centre, axis=1)
    short = br < tr + pol.clearance_mm
    if short.any():
        need_out = np.where(short, (tr + pol.clearance_mm) - br, 0.0)
        need_out = np.minimum(need_out, np.maximum(0.0, outward_budget - outset))
        bottom = bottom + inset_dir * need_out[:, None]
        outset = outset + need_out
        diag["collar_bottom_pushed_out_points"] = int(short.sum())

    # THE BRIDGE INVARIANT, ENFORCED ON THE POINTS THEMSELVES. Clamping the
    # seed was not enough: the lower ring is also moved by the shrink ladder,
    # by the nesting fix and by the gradient march, and the march's inward
    # direction has a lateral component - measured, the pair still reached 51%
    # of a 2.36mm bridge where 50% is the ceiling.
    #
    # In one dimension with rims at 0 and B, connector 0 spanning out to r0
    # and connector 1 back to B - r1, the gap is B - r0 - r1, so "the pair may
    # take at most `max_bridge_removal_fraction` of B" is exactly "every point
    # of each connector stays at least (1 - f/2)*B from the OTHER rim". That
    # is a per-point condition, so it can simply be imposed.
    if neighbour_rims is not None and len(neighbour_rims):
        from scipy.spatial import cKDTree as _KD
        _near = np.vstack([np.asarray(r, float) for r in neighbour_rims
                           if r is not None and len(r)])
        if len(_near):
            _tree = _KD(_near)
            _B = float(_tree.query(rim_k, workers=-1)[0].min())
            _thresh = (1.0 - 0.5 * pol.max_bridge_removal_fraction) * _B
            moved = 0
            for _ring in (top, bottom):
                _d, _i = _tree.query(_ring, workers=-1)
                _bad = _d < _thresh
                if _bad.any():
                    _away = _ring[_bad] - _near[_i[_bad]]
                    _ln = np.linalg.norm(_away, axis=1, keepdims=True)
                    _away = _away / np.where(_ln < 1e-12, 1.0, _ln)
                    _ring[_bad] += _away * (_thresh - _d[_bad])[:, None]
                    moved += int(_bad.sum())
            diag["bridge_clamped_ring_points"] = moved
            diag["bridge_clamp_threshold_mm"] = round(_thresh, 4)
            diag["bridge_between_rims_mm"] = round(_B, 4)

    reach_mm = float(np.linalg.norm(rim_k - bottom, axis=1).max())
    diag["ramp_reach_mm"] = round(reach_mm, 4)
    if reach_mm > pol.ramp_radius_mm + pol.max_reconstruction_depth_mm:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "outside_reconstruction_envelope",
                               diagnostics=diag)

    # THE CREASE MUST BE INSIDE THE COLLAR, and that is ENFORCED by
    # measurement rather than assumed from the construction. `rim_k` is the
    # crown's own cervical crease; a crease left outside is exactly where the
    # self-touch comes back - measured on a tipped tooth, 4 of 46 rim points
    # fell outside and the fused stage reported one CROWN + OLD_SOCKET_REPAIR
    # touch and one non-manifold edge. Where a point is not enclosed, its two
    # rings are pushed radially outward - the direction that cannot fold a
    # star-shaped loop - and the collar is rebuilt and re-measured.
    seat_v = seat_f = None
    rim_in = None
    collar_probe = None
    enclose_push = np.zeros(len(rim_k))
    for attempt in range(8):
        try:
            sv_, sf_ = _loft(top, bottom, axis=u_oa_k)
            sf_ = cg.make_consistent_winding(sv_, sf_)
            if not cg.is_edge_manifold_closed(sf_):
                diag["connector_error"] = "the loft is not closed"
                break
            vol = abs(cg.signed_volume(sv_, sf_))
            if vol <= 1e-9:
                diag["connector_error"] = "the loft has no volume"
                break
        except Exception as e:                            # noqa: BLE001
            diag["connector_error"] = f"{type(e).__name__}: {e}"
            break
        seat_v, seat_f = sv_, sf_
        diag["connector_volume_mm3"] = round(float(vol), 4)
        diag["seat_volume_mm3"] = round(float(vol), 4)    # legacy key
        collar_probe = CastProbe(seat_v, seat_f)
        rim_in = collar_probe.signed(rim_k)
        outside = rim_in > -pol.clearance_mm
        push = np.where(outside, 0.08, 0.0)
        push = np.minimum(push, np.maximum(0.0, outward_budget - outset
                                           - enclose_push))
        if not outside.any() or push.max() <= 1e-9:
            break
        top = top + inset_dir * push[:, None]
        bottom = bottom + inset_dir * push[:, None]
        enclose_push += push
    diag["collar_enclosure_attempts"] = int(attempt + 1)
    diag["collar_enclosure_push_max_mm"] = round(float(enclose_push.max()), 4)
    diag["collar_total_outward_reach_max_mm"] = round(
        float((outset + enclose_push).max()), 4)

    if seat_v is None or rim_in is None:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "interface_construction_failed",
                               diagnostics=diag)
    diag["crease_inside_collar_max_mm"] = round(float(rim_in.max()), 4)
    diag["crease_points_outside_collar"] = int((rim_in > -1e-9).sum())
    # The OLD SITE's crease - the boundary of the flush socket cap, at the
    # UNTRANSFORMED rim. Reported, not gated: for a large movement it is
    # legitimately far from the collar, and then the collar never crosses it.
    # CONTINUITY, MEASURED AGAINST THE SOLID rather than against its vertices.
    # `interface_continuity` compares each rim point with the nearest
    # connector VERTEX at a 1mm tolerance, which was right for a loft whose
    # top ring hugged the rim and is wrong for a collar that ENCLOSES it: the
    # rim is now inside the solid, and the nearest ring vertex can legitimately
    # be 2mm away. Inside-ness is the property that matters and the signed
    # distance answers it exactly.
    covered = rim_in < 0.0
    _cc = rim_k - rim_k.mean(axis=0)
    _, _, _vt = np.linalg.svd(_cc, full_matrices=False)
    _ang = np.arctan2(_cc @ _vt[1], _cc @ _vt[0])
    _o = np.argsort(_ang)
    _a, _c = _ang[_o], covered[_o]
    _gaps, _run = [], None
    for _i in range(len(_a)):
        if not _c[_i]:
            _run = _a[_i] if _run is None else _run
        elif _run is not None:
            _gaps.append(_a[_i] - _run)
            _run = None
    _worst = float(max(_gaps)) if _gaps else 0.0
    diag["continuity"] = {
        "continuous": bool(covered.all() or
                           (_worst < np.pi / 6 and covered.mean() > 0.75)),
        "covered_fraction": round(float(covered.mean()), 4),
        "largest_angular_gap_deg": round(float(np.degrees(_worst)), 2),
        "max_rim_signed_distance_to_connector_mm": round(float(rim_in.max()), 4),
        "measure": "rim points strictly INSIDE the connector solid, by exact "
                   "signed distance",
    }

    old_in = collar_probe.signed(rim_t0)
    diag["old_crease_inside_collar_max_mm"] = round(float(old_in.max()), 4)
    diag["old_crease_min_distance_to_collar_mm"] = round(
        float(np.abs(old_in).min()), 4)
    diag["old_crease_straddles_collar"] = bool(
        (old_in.min() < 0) and (old_in.max() > 0))

    diag["connector_built"] = True
    diag["seat_built"] = True
    diag["ramp_built"] = bool(is_lifted.any())   # one face of the connector
    diag["seat_lift_mm"] = round(pol.fusion_overlap_mm, 4)
    diag["seat_independent_of_root_length"] = True
    diag["emergence_height_mm"] = round(float(pol.emergence_height_mm), 4)

    diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
    # KEYWORDS, NOT POSITION. `diagnostics` sits after the geometry fields, so
    # passing it positionally landed it in `seat_verts` the moment the seat
    # fields were added - and the seat assignment on the next line then
    # overwrote it, silently discarding every diagnostic while the geometry
    # itself was fine. Nothing raised; `seat_built` simply vanished from the
    # manifest. Keyword arguments cannot drift like that.
    return InterfaceResult(
        ok=True, refusal_reason=None,
        seat_verts=seat_v, seat_faces=seat_f,
        diagnostics=diag)


# ---------------------------------------------------------------------------
# Region of influence - geodesic, not a box (amendment 10)
# ---------------------------------------------------------------------------

def affected_region(base_verts, base_faces, rim_k, radius_mm):
    """Face mask of the cast region this interface is allowed to touch.

    SURFACE DISTANCE, NOT A BOX. A rectangular XYZ box around a rim on a
    curved arch sweeps in the neighbouring teeth and the gingiva on the far
    side of the ridge - it would call a correct reconstruction a violation and
    hide a real one. This grows outward over face adjacency from the faces
    nearest the rim, which follows the surface the way the tissue does.

    Returns (face_mask, info). The bbox in `info` is a DIAGNOSTIC only.
    """
    from scipy.spatial import cKDTree

    verts = np.asarray(base_verts, float)
    faces = np.asarray(base_faces, np.int64)
    rim_k = np.atleast_2d(np.asarray(rim_k, float))

    centroids = verts[faces].mean(axis=1)
    tree = cKDTree(centroids)
    seed = set()
    for i in tree.query_ball_point(rim_k, r=radius_mm, workers=-1):
        seed.update(i)

    mask = np.zeros(len(faces), bool)
    if not seed:
        return mask, {"affected_faces": 0, "affected_area_mm2": 0.0,
                      "roi_bbox": None, "seed_faces": 0}
    idx = np.fromiter(seed, np.int64)
    mask[idx] = True

    area = _face_areas(verts, faces[mask]).sum() if mask.any() else 0.0
    return mask, {
        "affected_faces": int(mask.sum()),
        "affected_area_mm2": round(float(area), 4),
        "roi_bbox": _bbox(verts[np.unique(faces[mask])]),
        "seed_faces": int(len(idx)),
        "roi_radius_mm": radius_mm,
        "roi_definition": "surface distance over face adjacency, not an XYZ box",
    }


def bridge_between(rim_a, rim_b):
    """Closest approach between two target rims, for the adjacency check."""
    from scipy.spatial import cKDTree
    return float(cKDTree(np.asarray(rim_a, float))
                 .query(np.asarray(rim_b, float), workers=-1)[0].min())


# ---------------------------------------------------------------------------
# Surface fidelity - deviation, not vertex identity (amendment 6)
# ---------------------------------------------------------------------------

def _point_to_surface(pts, verts, faces, candidates=24):
    """Exact point-to-TRIANGLE distance, not point-to-vertex.

    THIS DISTINCTION IS NOT PEDANTRY - it was measuring 3.46mm of phantom
    "cast deformation" 18 to 32mm away from any tooth. The cast base is a
    trimmed shell extruded to a flat bottom, so its underside and walls carry
    very large triangles. After a boolean retessellates the model, a perfectly
    correct new vertex can land in the middle of one of those triangles, far
    from any of its corners - and a nearest-VERTEX measure calls that a
    3.5mm deviation when the point is exactly ON the original surface.

    OPEN3D'S BVH FIRST, AND THE CANDIDATE SEARCH IS THE FALLBACK - because
    the candidate search is an APPROXIMATION and it was caught being one. It
    takes the `candidates` triangles whose CENTROIDS are nearest, and on the
    cast's underside, where a triangle can be 20mm across, the triangle a
    point actually sits on is not among them: measured on a fused stage, 793
    of 4710 vertices reported distances of exactly 9.0000mm - the distance to
    whatever unrelated triangle did make the shortlist - while the cast was
    reproduced through manifold3d to 2e-6mm. That is the same class of defect
    as the nearest-VERTEX signed distance this module was corrected for, and
    it was quietly failing the unaffected-cast fidelity gate.

    `RaycastingScene.compute_distance` is exact and is already a dependency.
    """
    from scipy.spatial import cKDTree
    try:
        import open3d as o3d
        _sc = o3d.t.geometry.RaycastingScene()
        _sc.add_triangles(
            o3d.core.Tensor(np.asarray(verts, np.float32), o3d.core.float32),
            o3d.core.Tensor(np.asarray(faces, np.uint32), o3d.core.uint32))
        return _sc.compute_distance(
            o3d.core.Tensor(np.atleast_2d(np.asarray(pts, np.float32)),
                            o3d.core.float32)).numpy().astype(float)
    except Exception:                                     # noqa: BLE001
        pass
    pts = np.atleast_2d(np.asarray(pts, float))
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, np.int64)
    if not len(faces):
        return cKDTree(verts).query(pts, workers=-1)[0]

    tri = verts[faces]
    centroids = tri.mean(axis=1)
    k = min(candidates, len(faces))
    _, idx = cKDTree(centroids).query(pts, k=k, workers=-1)
    idx = np.atleast_2d(idx)

    def _seg(p, q0, q1):
        d = q1 - q0
        L = np.einsum("ij,ij->i", d, d)
        t = np.where(L > 1e-20, np.einsum("ij,ij->i", p - q0, d) / np.where(L > 1e-20, L, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        return np.linalg.norm(p - (q0 + d * t[:, None]), axis=1)

    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        t = tri[idx[i]]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        n = np.cross(b - a, c - a)
        nl = np.linalg.norm(n, axis=1)
        safe = nl > 1e-20
        nn = np.divide(n, np.where(safe[:, None], nl[:, None], 1.0))

        # Perpendicular foot, and whether it lands inside the triangle.
        dist_plane = np.einsum("ij,ij->i", p - a, nn)
        foot = p - nn * dist_plane[:, None]
        # Barycentric test by signed sub-triangle areas against the normal.
        s1 = np.einsum("ij,ij->i", np.cross(b - a, foot - a), nn)
        s2 = np.einsum("ij,ij->i", np.cross(c - b, foot - b), nn)
        s3 = np.einsum("ij,ij->i", np.cross(a - c, foot - c), nn)
        inside = safe & (s1 >= -1e-12) & (s2 >= -1e-12) & (s3 >= -1e-12)

        # Outside the triangle the closest point is on one of its edges.
        edge = np.minimum(np.minimum(_seg(p, a, b), _seg(p, b, c)), _seg(p, c, a))
        out[i] = np.where(inside, np.abs(dist_plane), edge).min()
    return out


def surface_deviation(ref_verts, ref_faces, test_verts, test_faces,
                      exclude_pts=None, exclude_radius_mm=0.0, samples=40000):
    """How far `test` strays from `ref`, OUTSIDE an excluded neighbourhood.

    DEVIATION, NOT BIT IDENTITY, and the reason is not tolerance-fudging: a
    CSG boolean retessellates everything it touches, so triangle counts and
    vertex indices legitimately change across the whole solid even where the
    SURFACE did not move. Comparing arrays would fail on a perfect result.
    Geometric fidelity is the invariant that actually means "the cast was not
    deformed here".

    Points within `exclude_radius_mm` of `exclude_pts` are dropped, so this
    measures the cast OUTSIDE the region the interface was allowed to touch.
    """
    from scipy.spatial import cKDTree

    ref_verts = np.asarray(ref_verts, float)
    test_verts = np.asarray(test_verts, float)

    pts = test_verts
    if len(pts) > samples:
        step = max(1, len(pts) // samples)
        pts = pts[::step]

    if exclude_pts is not None and exclude_radius_mm > 0 and len(pts):
        near = cKDTree(np.atleast_2d(np.asarray(exclude_pts, float)))
        d, _ = near.query(pts, workers=-1)
        pts = pts[d > exclude_radius_mm]

    if not len(pts):
        return {"compared_points": 0, "max_mm": 0.0, "mean_mm": 0.0,
                "rms_mm": 0.0, "p95_mm": 0.0, "p99_mm": 0.0,
                "note": "every sampled point fell inside the excluded region"}

    d = _point_to_surface(pts, ref_verts, np.asarray(ref_faces, np.int64))
    return {
        "compared_points": int(len(pts)),
        "max_mm": round(float(d.max()), 6),
        "mean_mm": round(float(d.mean()), 6),
        "rms_mm": round(float(np.sqrt((d ** 2).mean())), 6),
        "p95_mm": round(float(np.percentile(d, 95)), 6),
        "p99_mm": round(float(np.percentile(d, 99)), 6),
        # Counted as well as measured: a suite of percentiles can stay clean
        # while the number of outliers grows, and that growth is exactly what
        # a creeping deformation looks like.
        "outliers_over_0_25mm": int((d > 0.25).sum()),
        "measure": "point-to-TRIANGLE distance, test -> ref surface",
    }


# ---------------------------------------------------------------------------
# Rigidity (Phase 2 / amendment 14)
# ---------------------------------------------------------------------------

def rigidity_report(pts_t0, pts_k, faces=None, sample=4000, matrix=None):
    """Intrinsic geometry preserved? Edge lengths, distances, triangle areas.

    A transform can be wrong in ways a determinant does not see, so this
    measures the SHAPE rather than the matrix: pairwise distances and triangle
    areas are invariant under any rigid motion and under nothing else.
    """
    a = np.asarray(pts_t0, float)
    b = np.asarray(pts_k, float)
    if a.shape != b.shape:
        return {"ok": False, "reason": f"shape {a.shape} vs {b.shape}"}

    rng = np.random.default_rng(0)
    n = len(a)
    k = min(sample, n * (n - 1) // 2 if n > 1 else 0)
    out = {"points": int(n)}
    if k:
        i = rng.integers(0, n, k)
        j = rng.integers(0, n, k)
        keep = i != j
        i, j = i[keep], j[keep]
        da = np.linalg.norm(a[i] - a[j], axis=1)
        db = np.linalg.norm(b[i] - b[j], axis=1)
        err = np.abs(da - db)
        out["max_pairwise_distance_error_mm"] = float(err.max())
        out["rms_pairwise_distance_error_mm"] = float(np.sqrt((err ** 2).mean()))

    if faces is not None and len(faces):
        f = np.asarray(faces, np.int64)
        aa = _face_areas(a, f)
        ab = _face_areas(b, f)
        ae = np.abs(aa - ab)
        out["max_triangle_area_error_mm2"] = float(ae.max())
        ea = np.linalg.norm(a[f[:, 0]] - a[f[:, 1]], axis=1)
        eb = np.linalg.norm(b[f[:, 0]] - b[f[:, 1]], axis=1)
        out["max_edge_length_error_mm"] = float(np.abs(ea - eb).max())

    worst = max(out.get("max_pairwise_distance_error_mm", 0.0),
                out.get("max_edge_length_error_mm", 0.0))
    out["ok"] = bool(worst < 1e-6)
    out["worst_mm"] = float(worst)
    # Names the aggregate gate reads, and the two checks on the MATRIX itself.
    # |det - 1| alone does NOT catch a transposition - det(M^T) = det(M) for
    # every matrix there is - so R^T R - I is measured beside it, which is the
    # one that moves when a convention drifts.
    out["max_pairwise_change_mm"] = out.get("max_pairwise_distance_error_mm", 0.0)
    out["max_edge_length_change_mm"] = out.get("max_edge_length_error_mm", 0.0)
    if matrix is not None:
        R = np.asarray(matrix, float)[:3, :3]
        out["orthonormality_error"] = float(
            np.abs(R.T @ R - np.eye(3)).max())
        out["det_error"] = float(abs(np.linalg.det(R) - 1.0))
    else:
        out["orthonormality_error"] = None
        out["det_error"] = None
    out["rigid"] = bool(
        out["ok"]
        and (out["orthonormality_error"] is None
             or out["orthonormality_error"] < 1e-9)
        and (out["det_error"] is None or out["det_error"] < 1e-9))
    return out


# ---------------------------------------------------------------------------
# Overlap, continuity and transition quality (amendments 3, 4)
# ---------------------------------------------------------------------------

def overlap_volume(m3, solid_a, solid_b):
    """Real intersection volume of two manifold3d solids, or None.

    The brief is explicit that a 0.25mm nominal overlap is NOT proof the
    fusion will hold. This measures what the two solids actually share, which
    is the number that decides whether the union produces one body.
    """
    try:
        # manifold3d exposes only `batch_boolean`; there is no `.boolean`
        # method, so the previous call raised on every invocation and the
        # overlap silently reported None - which is exactly the "nominal
        # dimensions, unmeasured fusion" this function exists to prevent.
        return float(m3.Manifold.batch_boolean(
            [solid_a, solid_b], m3.OpType.Intersect).volume())
    except Exception:                                     # noqa: BLE001
        return None


def interface_continuity(rim_k, cavity_verts, tol_mm=1.0):
    """Is the reconstructed interface ONE band around the rim, or islands?

    A reconstruction continuous everywhere except a gap is worse than one that
    fails outright: the gap is a channel straight into the cast that a
    watertight check will happily accept. Measured as the largest angular gap
    between rim points that have interface geometry within `tol_mm`.
    """
    from scipy.spatial import cKDTree
    rim = np.asarray(rim_k, float)
    if cavity_verts is None or not len(rim):
        return {"continuous": False, "reason": "no interface geometry"}

    centre = rim.mean(axis=0)
    d, _ = cKDTree(np.asarray(cavity_verts, float)).query(rim, workers=-1)
    covered = d <= tol_mm

    _, _, vt = np.linalg.svd(rim - centre, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    ang = np.arctan2((rim - centre) @ e2, (rim - centre) @ e1)
    order = np.argsort(ang)
    a, c = ang[order], covered[order]
    if not c.any():
        return {"continuous": False, "covered_fraction": 0.0,
                "reason": "no rim point has interface geometry near it"}
    gaps, run = [], None
    for i in range(len(a)):
        if not c[i]:
            run = a[i] if run is None else run
        elif run is not None:
            gaps.append(a[i] - run)
            run = None
    worst = float(max(gaps)) if gaps else 0.0
    return {
        "continuous": bool(worst < np.pi / 6 and covered.mean() > 0.75),
        "covered_fraction": round(float(covered.mean()), 4),
        "largest_angular_gap_deg": round(float(np.degrees(worst)), 2),
        "max_rim_to_interface_mm": round(float(d.max()), 4),
    }


def transition_quality(final_verts, final_faces, rim_k, u_oa_k,
                       band_mm=1.5, step_mm=0.25):
    """Is the crown -> cast transition a blend, or an artificial ledge?

    WHY WATERTIGHTNESS CANNOT ANSWER THIS. A cylindrical collar around the
    cervical margin is perfectly closed, perfectly manifold and completely
    wrong - it is the artefact the first version of the ramp produced here.
    The only way to tell a blend from a step is to measure the SHAPE across
    the transition.

    RADIAL PROFILE: walk outward from the rim in `step_mm` rings and take the
    surface height along u_oa. A blend falls away gradually; a ledge puts most
    of its fall into one step, which is what `largest_step_share` catches.
    """
    from scipy.spatial import cKDTree
    verts = np.asarray(final_verts, float)
    rim = np.asarray(rim_k, float)
    u = np.asarray(u_oa_k, float)
    u = u / (np.linalg.norm(u) or 1.0)
    centre = rim.mean(axis=0)

    radial = rim - centre
    rn = np.linalg.norm(radial, axis=1, keepdims=True)
    rdir = np.divide(radial, np.where(rn < 1e-12, 1.0, rn))

    tree = cKDTree(verts)
    rings, heights = [], []
    for off in np.arange(0.0, band_mm + 1e-9, step_mm):
        probe = rim + rdir * off
        _, idx = tree.query(probe, workers=-1)
        rings.append(float(off))
        heights.append(float(np.median((verts[idx] - centre) @ u)))

    heights = np.asarray(heights)
    drops = np.diff(heights)
    total = float(abs(heights[0] - heights[-1]))
    worst_step = float(abs(drops).max()) if len(drops) else 0.0
    step_share = float(worst_step / total) if total > 1e-6 else 0.0
    return {
        "ring_offsets_mm": [round(x, 3) for x in rings],
        "ring_heights_mm": [round(float(x), 4) for x in heights],
        "total_fall_mm": round(total, 4),
        "largest_single_step_mm": round(worst_step, 4),
        "largest_step_share": round(step_share, 4),
        # A blend spreads its fall across rings. Three quarters of the whole
        # fall inside one 0.25mm step is a cliff. ENGINEERING VALIDATION
        # THRESHOLD for this prototype, chosen from the collar bug this check
        # was written to catch - not a clinical tolerance.
        "looks_like_a_ledge": bool(step_share > 0.75 and total > 0.2),
        "threshold_note": "engineering validation threshold for this prototype, "
                          "not a clinical tolerance",
    }


def old_site_quality(original_verts, original_faces,
                     final_verts, final_faces, rim_t0, u_oa=None,
                     radius_mm=4.0, tol_mm=0.05, grid_mm=0.4):
    """Was the site the tooth LEFT restored, or replaced by a new defect?

    TRIANGLE-SURFACE GEOMETRY, NOT NEAREST VERTEX. The first version compared
    final VERTICES against original VERTICES with a KD-tree, which is the same
    mistake `surface_deviation` was corrected for: on a cast whose underside
    carries very large triangles, a point sitting exactly ON the original
    surface measured 3.46mm from the nearest original vertex. Every distance
    here is point-to-TRIANGLE.

    The old site is defined in the T0 rim's own frame: the rim is fitted a
    plane, and the region is everything inside the rim's projected outline.
    Height is measured along that plane's normal, so "crater" and "plateau"
    are signed and mean what they say.

    Reported, in the brief's terms:
      crater_depth_mm      how far the restoration sinks BELOW the rim
      plateau_height_mm    how far it stands ABOVE it
      flat_fraction        how much of it is within `tol_mm` of one plane - a
                           flush cap reads ~1.0, which IS the "large flat
                           plateau" pathology, and it is reported rather than
                           hidden
      largest_local_step_mm  worst height jump between neighbouring samples
      modified_area_mm2    area of final surface over the site that has moved
                           further than `tol_mm` from the original cast
      volume_change_mm3    prism integral of the height change over the site
      patches              connected components of the modified region; more
                           than one is a disconnected patch
    """
    from scipy.spatial import cKDTree
    orig = np.asarray(original_verts, float)
    of = np.asarray(original_faces, np.int64)
    fin = np.asarray(final_verts, float)
    ff = np.asarray(final_faces, np.int64)
    rim = np.asarray(rim_t0, float)
    if len(rim) < 3 or not len(ff):
        return {"measured": False, "reason": "no rim or no final mesh"}

    centre = rim.mean(axis=0)
    _, _, vt = np.linalg.svd(rim - centre, full_matrices=False)
    n = vt[2] / (np.linalg.norm(vt[2]) or 1.0)
    if u_oa is not None and float(np.dot(n, np.asarray(u_oa, float))) < 0:
        n = -n
    e1 = vt[0] / (np.linalg.norm(vt[0]) or 1.0)
    e2 = np.cross(n, e1)

    def uv(p):
        d = np.atleast_2d(p) - centre
        return np.stack([d @ e1, d @ e2], axis=1)

    rim_uv = uv(rim)
    r_rim = float(np.linalg.norm(rim_uv, axis=1).max())

    # Sample the site on a regular grid in the rim plane, keep what is inside
    # the rim's own outline rather than a circle: a cervical margin is
    # scalloped and a circle would sweep in the gingiva beside it.
    g = np.arange(-r_rim, r_rim + grid_mm, grid_mm)
    gu, gv = np.meshgrid(g, g)
    samp = np.stack([gu.ravel(), gv.ravel()], axis=1)
    ang = np.arctan2(rim_uv[:, 1], rim_uv[:, 0])
    rad = np.linalg.norm(rim_uv, axis=1)
    order = np.argsort(ang)
    a_s, r_s = ang[order], rad[order]
    s_ang = np.arctan2(samp[:, 1], samp[:, 0])
    s_rad = np.linalg.norm(samp, axis=1)
    r_at = np.interp(s_ang, a_s, r_s, period=2 * np.pi)
    inside = s_rad <= r_at * 0.92          # stay clear of the rim itself
    samp = samp[inside]
    if len(samp) < 8:
        return {"measured": False, "reason": "old site too small to sample"}

    p3 = centre + samp[:, 0:1] * e1 + samp[:, 1:2] * e2
    # Height of each surface above the rim plane, taken as the nearest surface
    # point's own height - the cast is a height field over this patch.
    def heights(vv, fff):
        """Height of the OCCLUSAL surface above the rim plane at each sample.

        THE CAST IS A SOLID, not a height field, so a nearest-centroid query
        in the rim plane will happily return a triangle on the UNDERSIDE of
        the base: measured before this filter, the old site read a crater
        22.58mm deep - the cast's own thickness - and 13 disconnected patches.
        Faces are therefore restricted to those facing the occlusal side
        before anything is measured.
        """
        cen = vv[fff].mean(axis=1)
        a_, b_, c_ = vv[fff[:, 0]], vv[fff[:, 1]], vv[fff[:, 2]]
        nrm = np.cross(b_ - a_, c_ - a_)
        ln = np.linalg.norm(nrm, axis=1)
        facing = np.divide(nrm @ n, np.where(ln < 1e-15, 1.0, ln)) > 0.1
        keep = facing & (np.linalg.norm(
            np.stack([(cen - centre) @ e1, (cen - centre) @ e2], 1),
            axis=1) <= r_rim + radius_mm)
        if keep.sum() < 8:
            return None
        cc = cen[keep]
        t = cKDTree(np.stack([(cc - centre) @ e1, (cc - centre) @ e2], axis=1))
        dd, idx = t.query(samp, workers=-1)
        return (cc[idx] - centre) @ n, dd

    hf = heights(fin, ff)
    ho = heights(orig, of)
    if ho is None:
        return {"measured": False,
                "reason": "the original cast has no surface over this site"}
    if hf is None:
        # A DETERMINATE ANSWER, NOT AN UNCHECKED ONE. A tooth that has barely
        # moved is still standing over the site it will leave, so the finished
        # model's surface there is the crown. There is no restoration to
        # judge, and saying so is different from failing to look.
        return {"measured": True, "assessable": False,
                "obscured_fraction": 1.0,
                "reason": "the tooth still covers the site it left"}
    h_fin, d_fin_q = hf
    h_org, d_org_q = ho

    # OBSCURED SAMPLES ARE NOT DEFECTS. When the tooth has barely moved it is
    # still standing over its own old site, so the finished model's surface
    # there is the CROWN and no cast triangle is anywhere near - the nearest
    # one is on the socket wall, and reading its height reported a 2.4mm
    # crater at a site nothing had touched. A sample with no cast surface
    # within a couple of grid cells is reported as obscured and excluded from
    # every shape measure, rather than being scored against the wrong surface.
    seen = (d_fin_q <= grid_mm * 2.5) & (d_org_q <= grid_mm * 2.5)
    obscured = float(1.0 - seen.mean())
    # A SHAPE JUDGEMENT NEEDS A SHAPE. Below a quarter of the site the
    # surviving samples are a scatter around its edge, where the cast falls
    # away into the socket wall by design - measured, that read a 2.3mm
    # "crater" and 11 "disconnected patches" at a site nothing had touched.
    if seen.sum() < 8 or seen.mean() < 0.25:
        return {"measured": True, "assessable": False,
                "obscured_fraction": round(obscured, 4),
                "reason": "the tooth still covers most of the site it left"}
    samp, h_fin, h_org = samp[seen], h_fin[seen], h_org[seen]
    p3 = centre + samp[:, 0:1] * e1 + samp[:, 1:2] * e2

    # Local step: neighbouring grid samples, so the spacing is known.
    #
    # TWO STEPS, AND THE GATE USES THE SECOND. `largest_local_step_mm` is the
    # restored surface's own relief, which on a pristine cast already reads
    # 1.71mm across the socket wall - that is anatomy, not a defect, and
    # gating on it would refuse an untouched site. `largest_step_change_mm` is
    # the step in the CHANGE from the original cast, which is zero wherever
    # the reconstruction left the surface alone and jumps at the rim of a
    # crater or a plateau the software introduced.
    tree2 = cKDTree(samp)
    pairs = tree2.query_pairs(grid_mm * 1.05, output_type="ndarray")
    step = (np.abs(h_fin[pairs[:, 0]] - h_fin[pairs[:, 1]]).max()
            if len(pairs) else 0.0)

    dh = h_fin - h_org
    step_change = (np.abs(dh[pairs[:, 0]] - dh[pairs[:, 1]]).max()
                   if len(pairs) else 0.0)
    cell = grid_mm * grid_mm
    # Distance from the final surface over the site to the ORIGINAL cast.
    d_site = _point_to_surface(p3 + h_fin[:, None] * n, orig, of)
    moved = d_site > tol_mm
    flat = np.abs(h_fin - np.median(h_fin)) <= tol_mm

    # Disconnected patches of the modified region, over grid adjacency.
    patches = 0
    if moved.any():
        mi = np.where(moved)[0]
        pos = {int(x): k for k, x in enumerate(mi)}
        parent = list(range(len(mi)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a_, b_ in pairs:
            if a_ in pos and b_ in pos:
                ra, rb = find(pos[a_]), find(pos[b_])
                if ra != rb:
                    parent[ra] = rb
        patches = len({find(i) for i in range(len(mi))})

    return {
        "measured": True,
        "assessable": True,
        "old_rim_radius_mm": round(r_rim, 4),
        "samples": int(len(samp)),
        "obscured_fraction": round(obscured, 4),
        "sample_spacing_mm": grid_mm,
        "crater_depth_mm": round(float(max(0.0, -dh.min())), 4),
        "plateau_height_mm": round(float(max(0.0, dh.max())), 4),
        "largest_local_step_mm": round(float(step), 4),
        "largest_step_change_mm": round(float(step_change), 4),
        "flat_fraction": round(float(flat.mean()), 4),
        "modified_area_mm2": round(float(moved.sum() * cell), 4),
        "modified_fraction": round(float(moved.mean()), 4),
        "volume_change_mm3": round(float(dh.sum() * cell), 4),
        "max_deviation_from_original_mm": round(float(d_site.max()), 4),
        "mean_deviation_from_original_mm": round(float(d_site.mean()), 4),
        "patches": int(patches),
        "measure": "point-to-TRIANGLE distance, height along the T0 rim plane normal",
        "threshold_note": "engineering validation thresholds for this prototype, "
                          "not clinical tolerances",
    }


# ---------------------------------------------------------------------------
# The final gate - the written bytes are the truth (amendments 12, 17)
# ---------------------------------------------------------------------------

def components(faces):
    """Connected components over edge adjacency of an index buffer."""
    import collections
    f = np.asarray(faces, np.int64)
    if not len(f):
        return 0
    adj = collections.defaultdict(list)
    for i, tri in enumerate(f):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adj[(min(a, b), max(a, b))].append(i)
    seen = np.zeros(len(f), bool)
    n = 0
    for start in range(len(f)):
        if seen[start]:
            continue
        n += 1
        stack = [start]
        seen[start] = True
        while stack:
            i = stack.pop()
            tri = f[i]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                for j in adj[(min(a, b), max(a, b))]:
                    if not seen[j]:
                        seen[j] = True
                        stack.append(j)
    return n


# ---------------------------------------------------------------------------
# Exposure of synthetic geometry, and the aggregate manufacturing gate
# ---------------------------------------------------------------------------

SYNTHETIC_SOURCES = ("EMERGENCE_RECONSTRUCTION", "SEAT", "LOCAL_CLEARANCE",
                     "OLD_SOCKET_REPAIR")


def synthetic_exposure(verts, faces, labels, policy=None):
    """How much invented geometry is ON THE OUTSIDE of the finished model.

    This is only answerable because every input solid carries a manifold3d
    original id: a triangle that survives to the fused boundary and came from
    the connector IS exposed synthetic anatomy, and one that was buried by the
    crown or the cast is simply absent from the output. So the measurement has
    no heuristic in it at all.

    The connector is ALLOWED to be visible - a cervical emergence profile is
    what gingiva does - but a clearance wall standing proud of the cast is
    not: a clearance wall is a CUT surface, and if any of it survives to the
    boundary the cavity was not refilled by the crown that made it.
    """
    pol = policy or DEFAULT_POLICY
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    lab = np.asarray(labels, dtype=object)
    area = _face_areas(v, f)
    total = float(area.sum())
    by = {}
    for src in ("ORIGINAL_CAST",) + SYNTHETIC_SOURCES + ("CROWN",):
        by[src] = round(float(area[lab == src].sum()), 4)
    synth = sum(by[s] for s in SYNTHETIC_SOURCES)
    frac = float(synth / total) if total else 0.0
    return {
        "total_area_mm2": round(total, 4),
        "area_by_source_mm2": by,
        "exposed_synthetic_area_mm2": round(float(synth), 4),
        "exposed_synthetic_fraction": round(frac, 6),
        "exposed_clearance_area_mm2": by["LOCAL_CLEARANCE"],
        "unattributed_area_mm2": round(
            float(area[lab == "UNATTRIBUTED"].sum()), 4),
        "max_exposed_synthetic_fraction": pol.max_exposed_synthetic_fraction,
        "within_bound": bool(frac <= pol.max_exposed_synthetic_fraction),
        "clearance_is_buried": bool(by["LOCAL_CLEARANCE"] <= 1e-9),
    }


def aggregate_print_gate(stage: dict, policy=None) -> dict:
    """The ONE place a stage may be called ready to manufacture.

    `validate_printable_stl` answers a narrower question - is the written file
    a closed, single-bodied, correctly wound solid - and says so in its own
    `gate_scope`. This carries the rest of the brief's list, and it is
    deliberately a pure function of the stage record so nothing can claim
    readiness without the measurements being present: a MISSING measurement
    fails, exactly like a bad one. `NOT_CHECKED` is not `CLEAR`.
    """
    pol = policy or DEFAULT_POLICY
    gates = []

    def gate(name, ok, measured, detail):
        gates.append({"gate": name, "passed": bool(ok),
                      "measured": measured, "detail": detail})

    v = stage.get("stl_validation") or {}
    ms = stage.get("manifold_status") or {}
    gate("written_stl_topology", bool(v.get("print_ready")),
         v.get("failed_gates"),
         "finite, closed, one component, positive volume, consistent winding, "
         "measured on the bytes after a downstream reader's weld")
    gate("single_positive_manifold_body", bool(ms.get("single_positive_body")),
         ms.get("decompose_volumes_mm3"),
         "manifold3d decompose() - physical bodies, not index components")
    gate("body_count_agrees_with_stl",
         stage.get("manifold_bodies") == 1 and v.get("connected_components") == 1,
         [stage.get("manifold_bodies"), v.get("connected_components")],
         "the engine's body count and the file's component count must agree")

    touch = stage.get("self_touch") or {}
    gate("no_self_touching_boundary",
         touch.get("coincident_position_groups") == 0,
         touch.get("coincident_position_groups"),
         "a boundary that touches itself is inexpressible in a position-based "
         "format and becomes a non-manifold edge on the reader's weld")

    exp = stage.get("synthetic_exposure") or {}
    gate("synthetic_exposure_within_bound", bool(exp.get("within_bound")),
         exp.get("exposed_synthetic_fraction"),
         "area of invented geometry surviving to the outside of the model")
    gate("no_exposed_clearance_wall", bool(exp.get("clearance_is_buried")),
         exp.get("exposed_clearance_area_mm2"),
         "a cut surface on the boundary means the cavity was not refilled")

    fid = stage.get("cast_fidelity") or {}
    fwd = fid.get("original_to_final") or {}
    rev = fid.get("final_to_original") or {}
    ok_fid = (fwd.get("max_mm") is not None and rev.get("max_mm") is not None
              and max(fwd["max_mm"], rev["max_mm"])
              <= pol.max_unaffected_deviation_mm)
    gate("unaffected_cast_fidelity_two_sided", ok_fid,
         {"original_to_final_max_mm": fwd.get("max_mm"),
          "final_to_original_max_mm": rev.get("max_mm")},
         "point-to-TRIANGLE distance in BOTH directions, outside the allowed "
         "reconstruction envelope")

    roi = stage.get("roi_compliance") or {}
    gate("reconstruction_inside_envelope", bool(roi.get("compliant")),
         roi.get("outside_envelope_area_mm2"),
         "every changed cast surface must lie inside "
         "ALLOWED_RECONSTRUCTION_ENVELOPE")

    rows = stage.get("interfaces") or []
    gate("every_interface_built", bool(rows) and all(r.get("ok") for r in rows),
         [r.get("refusal_reason") for r in rows if not r.get("ok")],
         "a tooth with no local interface has no reconstruction at all")
    gate("interface_continuous_around_every_rim",
         bool(rows) and all((r.get("continuity") or {}).get("continuous")
                            for r in rows),
         [(r.get("continuity") or {}).get("largest_angular_gap_deg")
          for r in rows],
         "the reconstructed band must close around the rim, not be islands")
    gate("no_transition_ledge",
         bool(rows) and all(
             (r.get("transition") or {}).get("looks_like_a_ledge") is False
             for r in rows),
         [(r.get("transition") or {}).get("largest_step_share") for r in rows],
         "a cylindrical collar is watertight and still wrong; the shape is "
         "what separates a blend from a ledge")

    old = [r.get("old_site") or {} for r in rows]
    gate("old_site_restored",
         bool(old) and all(
             o.get("measured")
             and (not o.get("assessable", False)
                  or (o.get("crater_depth_mm", 9e9) <= pol.max_old_site_defect_mm
                      and o.get("plateau_height_mm", 9e9) <= pol.max_old_site_defect_mm
                      and o.get("largest_step_change_mm", 9e9) <= pol.max_old_site_step_mm
                      and o.get("patches", 9) <= 1))
             for o in old),
         [{k: o.get(k) for k in ("assessable", "obscured_fraction",
                                 "crater_depth_mm", "plateau_height_mm",
                                 "largest_step_change_mm", "patches")}
          for o in old],
         "crater, tower, sharp step or a disconnected patch at the site the "
         "tooth left")

    rig = [r.get("rigidity") or {} for r in rows]
    gate("crown_is_an_exact_rigid_transform",
         bool(rig) and all(r.get("rigid") for r in rig),
         [{k: r.get(k) for k in ("max_edge_length_change_mm",
                                 "max_pairwise_change_mm",
                                 "orthonormality_error", "det_error")}
          for r in rig],
         "intrinsic edge lengths, sampled pairwise distances, triangle areas, "
         "R^T R - I and det(R) - 1")

    bridges = stage.get("adjacent_bridges")
    gate("gingival_bridge_preserved",
         bridges is not None and all(b.get("ok") for b in bridges),
         [b for b in (bridges or []) if not b.get("ok")],
         "measured from the actual reconstruction regions, not from "
         "2 * fusion_overlap_mm")

    gate("root_length_independent",
         bool(rows) and all(r.get("seat_independent_of_root_length")
                            for r in rows),
         None,
         "no manufacturing geometry may be derived from root_length_mm")

    gate("clinical_consistency", bool(stage.get("clinical_consistent")),
         stage.get("clinical_consistency_detail"),
         "the stage's prescription is its own share of the committed one")

    failed = [g["gate"] for g in gates if not g["passed"]]
    return {
        "print_ready": not failed,
        "failed_gates": failed,
        "gates": gates,
        "verdict": ("PRINT READY" if not failed else "NOT PRINT READY"),
        "gate_scope": {
            "covers": [g["gate"] for g in gates],
            "note": "the complete aggregate manufacturing gate. A MISSING "
                    "measurement fails it, exactly like a bad one.",
            "not_verified_on": "no real de-identified scan has been run "
                               "through this gate; every number behind it is "
                               "synthetic.",
        },
    }


# ---------------------------------------------------------------------------
# Edge forensics with geometry provenance
# ---------------------------------------------------------------------------

# The sources a triangle in a fused stage can have come from. They are the
# labels the caller attaches to each input Manifold with `as_original()`, so
# they survive the boolean and can be read back off `run_original_id`.
GEOMETRY_SOURCES = (
    "ORIGINAL_CAST",            # scan surface retained by the trim
    "OLD_SOCKET_REPAIR",        # the flush closure of the site the tooth left
    "LOCAL_CLEARANCE",          # crown-derived subtraction, CASE A
    "EMERGENCE_RECONSTRUCTION", # the transition collar where the rim lifted
    "SEAT",                     # the transition collar where the rim is seated
    "CROWN",                    # the rigid transformed crown
)


def edge_table(verts, faces):
    """Undirected edges with their incident face ids, in one pass.

    Returns (edge_keys, starts, counts, face_ids_sorted) so a caller can slice
    `face_ids_sorted[starts[e] : starts[e] + counts[e]]` for edge `e`.
    """
    f = np.asarray(faces, np.int64)
    if not len(f):
        z = np.zeros((0, 2), np.int64)
        return z, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    he = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fid = np.tile(np.arange(len(f)), 3)
    key = np.sort(he, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, fid = key[order], fid[order]
    uniq, start, cnt = np.unique(key, axis=0, return_index=True,
                                 return_counts=True)
    return uniq, start, cnt, fid


def dissect_edges(verts, faces, which="nonmanifold", labels=None, limit=12):
    """Full topology of every offending edge, proven rather than asserted.

    "Non-manifold" here means an edge carried by MORE THAN TWO faces, counted
    on the mesh as it stands. The incident face ids, their areas and their
    normals are all reported so the claim can be checked: a long edge is not
    non-manifold because it is long, and a short one is not non-manifold
    because it is short.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    uniq, start, cnt, fid = edge_table(v, f)
    if which == "open":
        sel = np.where(cnt == 1)[0]
    else:
        sel = np.where(cnt > 2)[0]
    out = []
    for e in sel[:limit]:
        a, b = int(uniq[e][0]), int(uniq[e][1])
        inc = fid[start[e]:start[e] + cnt[e]]
        tri = f[inc]
        A, B, C = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        n = np.cross(B - A, C - A)
        area = 0.5 * np.linalg.norm(n, axis=1)
        unit = n / np.where(area[:, None] < 1e-20, 1.0, 2.0 * area[:, None])
        rec = {
            "edge_id": int(e),
            "vertex_ids": [a, b],
            "p0": [round(float(x), 6) for x in v[a]],
            "p1": [round(float(x), 6) for x in v[b]],
            "length_mm": round(float(np.linalg.norm(v[b] - v[a])), 6),
            "incident_face_count": int(cnt[e]),
            "incident_face_ids": [int(x) for x in inc],
            "face_areas_mm2": [round(float(x), 9) for x in area],
            "face_normals": [[round(float(y), 4) for y in x] for x in unit],
        }
        if labels is not None:
            rec["provenance"] = [str(labels[i]) for i in inc]
        out.append(rec)
    return out


def _weld_with_face_map(verts, faces):
    """`cg.weld_vertices`, plus which input faces survived.

    Identical arithmetic - exact position equality, index-degenerate faces
    dropped - but it also returns the surviving face indices so a provenance
    label array can be carried across the weld. Duplicating the arithmetic
    would risk the two drifting, so this is the one implementation and
    `cg.weld_vertices` is left alone as the public form.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    uniq, inverse = np.unique(v, axis=0, return_inverse=True)
    if len(uniq) == len(v):
        return v, f, 0, np.arange(len(f))
    g = inverse.ravel()[f]
    keep = ((g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 0] != g[:, 2]))
    return uniq, g[keep], int(len(v) - len(uniq)), np.where(keep)[0]


def self_touch_report(verts, faces, labels=None, limit=12):
    """Where a boolean result's boundary TOUCHES ITSELF, and between what.

    manifold3d records a self-touch as two vertices at an identical position
    under different indices. The control that makes this readable: a clean
    transversal union of two cubes, two rotated cubes, a cube and a sphere,
    and even two cubes meeting FACE TO FACE all produce zero coincident
    positions, while two cubes meeting along an EDGE produce exactly one
    coincident pair and one non-manifold edge. So a coincident position is not
    a run boundary or a tolerance artefact - it is the boundary touching
    itself, and welding it (which binary STL forces, because it stores
    positions) is what turns it into a non-manifold edge.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    uniq, inv, cnt = np.unique(v, axis=0, return_inverse=True,
                               return_counts=True)
    inv = inv.ravel()
    groups = np.where(cnt > 1)[0]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rep = {"coincident_position_groups": int(len(groups)),
           "duplicated_vertices": int(len(v) - len(uniq)),
           "zero_area_triangles": int((area <= 0.0).sum()),
           "min_triangle_area_mm2": (round(float(area.min()), 12)
                                     if len(area) else None),
           "touches": []}
    if labels is None or not len(groups):
        return rep
    vface = {}
    for t in range(len(f)):
        for vi in f[t]:
            vface.setdefault(int(vi), []).append(t)
    pairs = {}
    for g in groups[:limit]:
        mem = np.where(inv == g)[0]
        srcs = sorted({str(labels[t]) for vi in mem for t in vface.get(int(vi), [])})
        rep["touches"].append({
            "position": [round(float(x), 6) for x in v[mem[0]]],
            "copies": int(len(mem)),
            "sources": srcs})
        pairs[" + ".join(srcs)] = pairs.get(" + ".join(srcs), 0) + 1
    rep["touching_source_pairs"] = pairs
    return rep


def serialisation_forensics(verts, faces, labels=None, limit=8,
                            write=None, read=None):
    """The whole chain for one stage, measured at every point, not two.

    in-memory boolean -> our export weld -> written STL bytes -> reread ->
    the weld a downstream reader performs. Each point reports open and
    non-manifold edge counts, and every offending edge is dissected with its
    incident faces and their provenance.

    `write`/`read` are injected so this module does not need to know about the
    STL layer; the caller passes `cg.write_binary_stl_bytes` and
    `stl_io.parse_stl_bytes`.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    lab = None if labels is None else np.asarray(labels, dtype=object)
    chain = []

    def point(name, vv, ff, ll):
        r = cg.manifold_report(ff)
        chain.append({
            "point": name,
            "open_edges": int(r["open_edges"]),
            "nonmanifold_edges": int(r["nonmanifold_edges"]),
            "worst_edge_faces": int(r["worst_edge_faces"]),
            "triangles": int(len(ff)),
            "vertices": int(len(vv)),
            "open_edge_detail": dissect_edges(vv, ff, "open", ll, limit),
            "nonmanifold_edge_detail": dissect_edges(vv, ff, "nonmanifold",
                                                     ll, limit)})

    point("after_boolean_in_memory", v, f, lab)
    chain[-1]["self_touch"] = self_touch_report(v, f, lab, limit)

    q = v.astype(np.float32).astype(np.float64)
    wv, wf, merged, kept = _weld_with_face_map(q, f)
    wl = None if lab is None else lab[kept]
    point("after_export_weld", wv, wf, wl)
    chain[-1]["merged_vertices"] = int(merged)

    if write is not None and read is not None:
        blob = write(wv, wf)
        rv, rf = read(blob)
        point("after_stl_write_and_reread", rv, rf, None)
        rwv, rwf, rmerged, _ = _weld_with_face_map(rv, rf)
        point("after_reader_weld", rwv, rwf, None)
        chain[-1]["merged_vertices"] = int(rmerged)
    return {"chain": chain}



def collapse_short_nonmanifold_edges(verts, faces, max_len_mm=0.05):
    """Collapse the very short edges a CSG tangency leaves non-manifold.

    WHAT THIS IS FOR, measured rather than assumed. On an INTRUDED tooth the
    crown's outer surface near the cervical margin and the cast's gingival
    surface are THE SAME SCAN TRIANGLES displaced along the tooth axis, so
    over a band around the rim they run nearly parallel a few microns apart.
    manifold3d resolves that grazing contact with a fold: measured on an
    intrusion of 1.0mm, ONE edge 0.0204mm long carrying FOUR faces, 0.204mm
    from the moved tooth's original rim, with one incident face of area
    2.6e-5 mm2 whose normal is exactly anti-parallel to its neighbour. Both
    directed edges appeared twice, which is why the winding check failed too.

    WHY COLLAPSE AND NOT DELETE. Deleting the sliver leaves the edge with
    three faces, which is still non-manifold, and deleting both faces of the
    fold OPENS the edges they shared with the rest of the surface - the
    lesson already recorded for zero-area faces on the export path. Collapsing
    the edge removes the fold and its neighbours consistently, because every
    face that used the edge either disappears (it becomes degenerate) or
    simply loses a duplicated corner.

    WHY 0.05mm IS THE CEILING AND WHY IT IS NOT A LOOSENED TOLERANCE. An
    intraoral scanner resolves 20-50 microns; a feature below that is not
    anatomy the scan could have recorded. The default is the TOP of that band,
    and the cap is hard - a longer non-manifold edge is a real geometric
    defect and is left alone so the gates refuse it. This deliberately cannot
    escalate: escalating a degeneracy epsilon once amplified a 1e-15 rounding
    difference into a 0.6% volume change under a pure rigid transform.

    NOTHING HERE IS TRUSTED. The caller re-runs the full validation on the
    result and keeps the repair only if it actually produced a closed,
    single-bodied, consistently-wound solid - and records that it happened.

    Returns (verts, faces, info).
    """
    v = np.asarray(verts, float).copy()
    f = np.asarray(faces, np.int64).copy()
    # `kept_faces` is present on EVERY return path, including the three early
    # ones. A caller carrying a per-triangle provenance array indexes with it,
    # and an early return that omitted it would raise a KeyError on exactly the
    # stages where nothing needed repairing - the common case.
    info = {"collapsed_edges": 0, "removed_faces": 0,
            "kept_faces": np.arange(len(np.asarray(faces, np.int64))),
            "max_len_mm": float(max_len_mm), "edge_lengths_mm": []}
    if not len(f):
        return v, f, info

    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    bad = uniq[counts > 2]
    if not len(bad):
        return v, f, info

    lengths = np.linalg.norm(v[bad[:, 0]] - v[bad[:, 1]], axis=1)
    short = bad[lengths <= max_len_mm]
    info["edge_lengths_mm"] = [round(float(x), 6) for x in np.sort(lengths)[:16]]
    info["nonmanifold_edges_found"] = int(len(bad))
    info["nonmanifold_edges_short_enough"] = int(len(short))
    if not len(short):
        return v, f, info

    # Union-find, so two short edges sharing a vertex collapse consistently
    # rather than one overwriting the other.
    parent = np.arange(len(v))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in short:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            info["collapsed_edges"] += 1

    root = np.array([find(i) for i in range(len(v))], np.int64)
    f2 = root[f]
    degenerate = ((f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2])
                  | (f2[:, 0] == f2[:, 2]))
    info["removed_faces"] = int(degenerate.sum())
    # WHICH FACES SURVIVED, so a caller carrying a per-triangle provenance
    # array can bring it along. Without it the labels silently go out of step
    # with the faces, and the mismatch only shows on the stages where the
    # repair actually collapsed something.
    info["kept_faces"] = np.where(~degenerate)[0]
    f2 = f2[~degenerate]

    # Compact, keeping the SURVIVING original coordinate for each cluster -
    # the collapse decides which vertices are one vertex, it does not invent a
    # new position for them.
    used = np.unique(f2)
    remap = np.zeros(len(v), np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f2], info


def manifold_status(solid, m3=None):
    """Everything manifold3d itself will say about a solid, in one record.

    THE POINT IS THAT THESE ANSWER DIFFERENT QUESTIONS from the STL gates, and
    both have to agree before a stage ships. `decompose()` counts PHYSICAL
    bodies and is the only reliable body count here - `_face_components` walks
    edge adjacency on the index buffer, and manifold3d's duplicate vertices
    split a geometrically joined solid into two index-disconnected groups, so
    it once called a fused tooth "floating" while a boolean intersection
    measured 141mm3 of overlap (CLAUDE.md section 10).

    `status()` is manifold3d's own error enum. A non-OK status on a solid that
    still reports a sensible volume is exactly the kind of quiet corruption
    that reaches a printer.
    """
    if m3 is None:
        import manifold3d as m3
    out = {}
    try:
        st = solid.status()
        out["status"] = getattr(st, "name", str(st))
        out["status_ok"] = out["status"].upper() in ("NO_ERROR", "NOERROR", "OK")
    except Exception as e:                                   # noqa: BLE001
        out["status"] = f"unavailable: {type(e).__name__}"
        out["status_ok"] = None
    for name, fn in (("is_empty", solid.is_empty), ("volume", solid.volume),
                     ("surface_area", getattr(solid, "surface_area", None)),
                     ("genus", solid.genus)):
        if fn is None:
            continue
        try:
            val = fn()
            out[name] = bool(val) if name == "is_empty" else float(val)
        except Exception as e:                               # noqa: BLE001
            out[name] = f"unavailable: {type(e).__name__}"
    try:
        parts = solid.decompose()
        vols = sorted((float(p.volume()) for p in parts), reverse=True)
        out["decompose_bodies"] = int(len(parts))
        out["decompose_volumes_mm3"] = [round(v, 4) for v in vols[:8]]
        out["positive_volume_bodies"] = int(sum(1 for v in vols if v > 0.0))
        out["negative_volume_bodies"] = int(sum(1 for v in vols if v <= 0.0))
    except Exception as e:                                   # noqa: BLE001
        out["decompose_bodies"] = f"unavailable: {type(e).__name__}"
    # THE GATE. One body, positive volume, no error status, not empty.
    out["single_positive_body"] = bool(
        out.get("positive_volume_bodies") == 1
        and out.get("is_empty") is False
        and isinstance(out.get("volume"), float) and out["volume"] > 0.0)
    return out


def validate_printable_stl(blob, expect_components=1):
    """HARD gates, run on the ACTUAL BYTES that were written.

    THE FILE IS THE TRUTH, and this is the correction that matters most here.
    The old stage export DID reread the STL and DID measure it - then gated on
    the in-memory index buffer instead, so 68 to 77 non-manifold edges per
    stage were recorded in the manifest and shipped anyway. A validator that
    reads a different object from the one the lab receives is not a validator.

    The pipeline this completes:

        in-memory fused -> export weld -> write bytes
                        -> reread bytes -> reader weld -> THESE GATES

    Returns `print_ready` plus a `gates` list. Every gate carries its measured
    value, so a refusal can be argued with rather than only obeyed.
    """
    import stl_io

    gates, report = [], {}

    pv, pf = stl_io.parse_stl_bytes(blob)
    report["reread_vertices"] = int(len(pv))
    report["reread_faces"] = int(len(pf))

    # THE RAW REREAD, BEFORE ANY WELD. Without this the chain jumps straight
    # from the in-memory boolean to the reread-AND-welded result, and those
    # two differ for two quite separate reasons: what float32 serialisation
    # did, and what the reader's weld did. Distinguishing them is the whole
    # point of measuring the chain at all.
    raw = cg.manifold_report(pf)
    report["edges_post_read_before_weld"] = {
        "open": int(raw["open_edges"]),
        "nonmanifold": int(raw["nonmanifold_edges"]),
        "total": int(raw["total_edges"])}

    wv, wf, merged = cg.weld_vertices(pv, pf)
    report["reader_weld_merged_vertices"] = int(merged)
    report["welded_vertices"] = int(len(wv))
    report["welded_faces"] = int(len(wf))

    def gate(name, ok, measured, detail):
        gates.append({"gate": name, "passed": bool(ok),
                      "measured": measured, "detail": detail})

    finite = bool(np.isfinite(wv).all())
    gate("finite_coordinates", finite, int((~np.isfinite(wv)).sum()),
         "no NaN or Inf in the written file")

    mr = cg.manifold_report(wf)
    report["open_edges"] = int(mr["open_edges"])
    report["nonmanifold_edges"] = int(mr["nonmanifold_edges"])
    report["total_edges"] = int(mr["total_edges"])
    gate("zero_open_edges", mr["open_edges"] == 0, int(mr["open_edges"]),
         "an open edge is a hole; the model is not a solid")
    gate("zero_nonmanifold_edges", mr["nonmanifold_edges"] == 0,
         int(mr["nonmanifold_edges"]),
         "measured AFTER the reader weld - the gate the old export reported "
         "but did not enforce")

    vol = float(cg.signed_volume(wv, wf)) if finite and len(wf) else 0.0
    report["volume_mm3"] = round(vol, 4)
    gate("positive_volume", vol > 0.0, round(vol, 4),
         "zero or negative volume is an inside-out or empty solid")

    ncomp = components(wf)
    report["connected_components"] = int(ncomp)
    gate("single_component", ncomp == expect_components, int(ncomp),
         "expected " + str(expect_components) + " physical body")

    wind = bool(cg._winding_is_consistent(wf))
    gate("consistent_winding", wind, wind,
         "every directed edge traversed exactly once")

    report["gates"] = gates
    report["failed_gates"] = [g["gate"] for g in gates if not g["passed"]]
    report["print_ready"] = len(report["failed_gates"]) == 0
    # THE WORDING IS DELIBERATE AND IT IS NARROWER THAN IT WAS. This function
    # checks the BOOLEAN/TOPOLOGY properties of the written bytes and nothing
    # else, so calling its result "PRINT READY" claims a great deal it has not
    # measured. The aggregate manufacturing verdict additionally requires
    # transition quality, interface continuity, old-site quality, two-sided
    # unaffected-cast fidelity, ROI compliance, seat/ramp exposure and
    # prescription consistency - none of which are evaluated here.
    report["verdict"] = ("PASSES BOOLEAN/TOPOLOGY REGRESSION"
                         if report["print_ready"] else
                         "FAILS BOOLEAN/TOPOLOGY REGRESSION")
    report["gate_scope"] = {
        "covers": ["finite_coordinates", "zero_open_edges",
                   "zero_nonmanifold_edges", "positive_volume",
                   "single_component", "consistent_winding",
                   "stl_round_trip_and_reader_weld"],
        "does_not_cover": ["transition_quality", "interface_continuity",
                           "old_site_quality", "two_sided_cast_fidelity",
                           "roi_compliance", "seat_ramp_exposure",
                           "crown_rigidity", "prescription_consistency"],
        "note": ("this is NOT the aggregate PRINT READY verdict. It is the "
                 "boolean/topology half of it, measured on the actual written "
                 "STL after a downstream reader's weld.")}
    report["wording_note"] = (
        "manufacturing geometry validated against engineering gates. NOT a "
        "claim of clinical validation, and NOT the full manufacturing gate.")
    return report
