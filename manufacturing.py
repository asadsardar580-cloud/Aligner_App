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

    # Where the adaptive search STARTS when the crown does not penetrate at
    # all. A tooth sitting exactly on the surface still needs a seat cut for
    # the union to have common volume.
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

    # Carry the cut above the gingival surface so the subtract is not coplanar
    # with it. Part of the SAME cup solid, not a second tool - two tools
    # sharing a wall reintroduce the coincidence being removed.
    cavity_raise_mm: float = 0.30

    # The seat is a truncated cone: narrow at the top where it must sit inside
    # the crown, wide at the bottom where it must reach past the cavity wall
    # into cast material. Neither face may coincide with a crown or cavity
    # surface, which is why both offsets are non-zero.
    seat_top_inset_mm: float = 0.15
    seat_bottom_outset_mm: float = 1.20
    seat_depth_mm: float = 1.20

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

    cavity_verts: np.ndarray | None = None
    cavity_faces: np.ndarray | None = None
    ramp_verts: np.ndarray | None = None
    ramp_faces: np.ndarray | None = None
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


def _loft(loop_a, loop_b):
    """Closed solid between two same-length ordered loops.

    Side wall plus a fan cap at each end. Both caps use their own loop's
    centroid, so the solid is closed by construction and the only way it can
    be non-manifold is if a loop passes through itself - which the caller
    checks with is_edge_manifold_closed rather than trusting.
    """
    n = len(loop_a)
    verts = np.vstack([loop_a, loop_b,
                       loop_a.mean(axis=0)[None, :], loop_b.mean(axis=0)[None, :]])
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

    # === THE CUTTER IS THE CROWN ITSELF ===================================
    #
    # Where material must come out, it comes out where the crown ACTUALLY IS -
    # not from a synthetic socket of some chosen depth. `cast - crown` removes
    # exactly the volume the crown occupies and not one cubic millimetre more,
    # which is what "remove only the necessary local cast material" means, and
    # it cannot over-cut a thin wall or reach below the tooth.
    #
    # The caller supplies the transformed crown Manifold it already holds, so
    # nothing is rebuilt here. `cavity_verts` stays None: there is no separate
    # socket solid any more.
    #
    # This also removes the last place an arbitrary depth could enter the
    # manufacturing path.
    # Outward radial direction about the rim's own centroid. Used by the ramp
    # and the seat; previously defined beside the socket cup that is now gone.
    centre = rim_k.mean(axis=0)
    radial = rim_k - centre
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True)
    inset_dir = np.divide(radial, np.where(rnorm < 1e-12, 1.0, rnorm))

    # === CASE A: PENETRATION -> CROWN-DERIVED LOCAL CLEARANCE ============
    #
    # THE TOOL IS THE CROWN DILATED, NOT THE CROWN. That distinction is the
    # whole fix, and the previous pass got it wrong in the opposite direction.
    # Cutting with the crown ITSELF is `(cast - crown) u crown`, an identity on
    # geometry and not on topology: the cavity wall IS the crown's surface and
    # the union lays that same surface back on top of itself, so manifold3d
    # emits coincident-but-distinct vertices and a weld turns every one into a
    # non-manifold edge. Measured across a 1.2mm extrusion, 0,11,5,9,5 with the
    # crown-cut against 0,0,0,1,0 without. Removing the cut was right.
    #
    # WHAT REMOVING IT LEFT BEHIND is the defect this now fixes. With no cut,
    # a barely-moved crown sits almost exactly in its original socket, so its
    # outer surface and the cast's gingival surface are THE SAME SCAN
    # TRIANGLES a few microns apart over a long band. manifold3d unions them
    # into a solid that TOUCHES ITSELF along that band - valid as an indexed
    # mesh, and inexpressible in any position-based format. Measured on
    # extrusion 0.25mm stage 2: the boolean output is clean (0 non-manifold
    # edges), the written STL rereads with 40, and welding leaves 3. A small
    # movement is WORSE than a large one for exactly this reason - at 1.2mm
    # the crown is clear of the old socket wall and nothing grazes.
    #
    # Dilating the crown by `clearance_mm` before subtracting breaks the
    # coincidence without reintroducing it: the cavity wall now stands that
    # distance AWAY from the crown, so no surface in the result is a copy of
    # any other. The crown itself is untouched and stays exactly a rigid
    # transform of T0 - the tool is a separate solid built from it, and the
    # rigidity test still measures the crown alone.
    #
    # The clearance is the SMALLEST that removes the coincidence, not a
    # margin chosen for comfort: it has to exceed the scanner's own resolution
    # (20-50 microns) so the two surfaces cannot be the same measurement, and
    # `clearance_mm` is 0.05 for that reason. It is not a socket depth and
    # cannot become one - the tool's extent is the crown's extent.
    needs_cut = mode in ("penetrating", "mixed")
    diag["cut_with_crown"] = False            # never the crown itself
    diag["clearance_measured"] = bool(needs_cut)
    diag["clearance_mm"] = float(pol.clearance_mm) if needs_cut else 0.0
    diag["depth_used_mm"] = round(float(penetration), 4) if needs_cut else 0.0
    diag["depth_wanted_mm"] = round(float(penetration), 4)
    diag["depth_clamped_by"] = None
    cav_v = cav_f = None
    if needs_cut:
        # CASE B (pure separation) never reaches here, so no generic socket is
        # ever cut merely because the rim moved. CASE C (mixed) uses the same
        # tool and needs no partition: a solid built from the crown removes
        # material only where the crown actually is, so penetrating sectors get
        # clearance and separated sectors are untouched by construction.
        # THE OFFSET IS PER VERTEX, AND IT CHANGES SIGN WITH DEPTH. A uniform
        # dilation was tried first and is too blunt: it removes the cast
        # everywhere the crown is, INCLUDING the deep penetration that is the
        # fusion. Measured, it fixed extrusion 0.25mm and broke five other
        # cases into two bodies - the tooth and its bridge severed from a cast
        # they no longer touched, with seat-to-cast overlap still reading 51
        # to 67 mm3 against the pre-cut cast.
        #
        # So the tool is pushed OUT where the two surfaces graze and pulled IN
        # where the crown is properly buried:
        #
        #   at the surface (sd = 0)      +clearance_mm   -> breaks the tangency
        #   at depth (sd <= -band)       -fusion_overlap -> leaves real
        #                                                   common volume
        #
        # blended linearly between, so there is no step for the boolean to
        # resolve. Where the offset is negative the cavity wall lies INSIDE
        # the crown and the union covers it, which is why that half cannot
        # reintroduce a coincident surface either.
        band = max(pol.cavity_outset_mm, 1e-6)
        sd = probe.signed(crown_k)
        w = np.clip(-sd / band, 0.0, 1.0)          # 0 at the surface, 1 deep
        offset = pol.clearance_mm * (1.0 - w) - pol.fusion_overlap_mm * w
        cav_f = np.asarray(crown_faces, np.int64)
        tool_v = (np.asarray(crown_k, float)
                  + cg.vertex_normals(crown_k, cav_f) * offset[:, None])

        # MEASURED, NOT CUT - and the measurement is why. `graze` counts crown
        # vertices whose distance to the cast is inside the band, i.e. whose
        # surface is near-coincident with the cast's. On this fixture it reads
        # 113 of 127, and 121 of 132 on the second tooth: the crown sits in
        # ITS OWN FILLED SOCKET, so the crown's surface IS the cast's surface
        # over almost its whole area and there is no deeply-buried region to
        # keep the fusion alive. Subtracting this tool therefore clears the
        # cast away from nearly the entire crown, and the union falls to TWO
        # BODIES - measured, 8 of 18 matrix cases, with seat-to-cast overlap
        # still reading 51-67mm3 against the PRE-cut cast, which is what made
        # the severance confusing until the graze count was added.
        #
        # Both variants were built and measured: a uniform +0.05mm dilation,
        # and this depth-modulated form that pulls the tool INSIDE the crown
        # where it is buried. Both fix extrusion 0.25mm and both sever the
        # tipping, rotation and buccolingual cases, because the modulation has
        # almost nothing to hold on to.
        #
        # The tool is therefore NOT emitted. What is missing before it can be
        # is a connector that is guaranteed to bite the POST-clearance cast -
        # today `overlap_seat_cast_mm3` is measured against the original cast
        # and cannot see the void the clearance creates. That is the next
        # change, and inventing a bigger seat to paper over it would be the
        # parameter sweep this work is explicitly not doing.
        diag["clearance_tool_emitted"] = False
        diag["clearance_band_mm"] = round(float(band), 4)
        diag["crown_min_signed_distance_mm"] = round(float(sd.min()), 4)
        diag["crown_max_signed_distance_mm"] = round(float(sd.max()), 4)
        diag["near_coincident_fraction"] = round(
            float((np.abs(sd) < band).mean()), 4)
        diag["clearance_offset_min_mm"] = round(float(offset.min()), 4)
        diag["clearance_offset_max_mm"] = round(float(offset.max()), 4)
        diag["crown_points_in_graze_band"] = int((w < 1.0).sum())
        diag["crown_points_deeply_buried"] = int((w >= 1.0).sum())
        diag["clearance_tool_volume_mm3"] = round(
            float(abs(cg.signed_volume(tool_v, cav_f))), 4)
        diag["crown_volume_mm3"] = round(
            float(abs(cg.signed_volume(crown_k, cav_f))), 4)
        cav_v = cav_f = None          # measured above; deliberately not cut
    diag["cavity_volume_mm3"] = diag.get("clearance_tool_volume_mm3")

    # === ONE INTEGRATED TRANSITION VOLUME =================================
    #
    # The ramp and the seat used to be two separate solids. On a MIXED tooth -
    # part of the rim buried, part lifted, which is what any tipping or bodily
    # movement produces - they both existed around the same rim and OVERLAPPED
    # EACH OTHER below the cervical margin. Four solids then met along one
    # curve (cast, crown, ramp, seat) and the welded STL showed edges with four
    # faces on them. Measured at stage 5: both remaining non-manifold edges sat
    # 0.19-0.76mm from crown1, rim1, ramp1 AND seat1 simultaneously.
    #
    # They are now ONE loft. The top ring is a single curve buried inside the
    # crown; the bottom ring is chosen PER POINT by that point's own
    # relationship to the cast:
    #
    #   lifted  -> the landing on the cast surface   (an emergence bridge)
    #   seated  -> down and out into the cast        (a fusion seat)
    #
    # A tooth that is lifted all the way round gets a pure bridge; one that is
    # seated all the way round gets a pure seat; a tipped tooth gets a single
    # continuous volume that is a bridge on one side and a seat on the other,
    # with no seam between them and nothing overlapping anything else.
    lift = np.clip(rim_signed, 0.0, None)
    is_lifted = lift > pol.fusion_overlap_mm
    diag["rim_points_lifted"] = int(is_lifted.sum())
    diag["rim_points_seated"] = int((~is_lifted).sum())

    # Bottom ring, seated default: down along the axis and out into the cast.
    bottom = (rim_k + inset_dir * pol.seat_bottom_outset_mm
              - u_oa_k * pol.seat_depth_mm)

    if is_lifted.any():
        # THE APRON RADIUS IS PER POINT AND PROPORTIONAL TO THE LOCAL LIFT, so
        # the blend vanishes where the tooth never left the tissue rather than
        # flaring material out of nowhere.
        radius = np.minimum(pol.ramp_radius_mm, lift)[:, None]
        landing_seed = rim_k + inset_dir * radius
        drop_cap = pol.max_reconstruction_depth_mm
        landing, hit = probe.drop_to_surface(landing_seed, u_oa_k, drop_cap)
        # WHICH ATTEMPT ANSWERED, per point. -1 = never landed, 0 = the full
        # apron radius, 1..3 = the shrink ladder, 4 = the direction-free
        # fallback. Without this the diagnostics can say how many points landed
        # but not whether they landed where the apron wanted them or only after
        # collapsing inward onto the rim, and those are different geometries
        # with the same success count.
        retry_level = np.where(hit, 0, -1).astype(np.int64)
        for step, shrink in enumerate((0.5, 0.25, 0.0), start=1):
            if hit.all():
                break
            retry, retry_hit = probe.drop_to_surface(
                rim_k + inset_dir * (radius * shrink), u_oa_k, drop_cap)
            take = (~hit) & retry_hit
            landing[take] = retry[take]
            retry_level[take] = step
            hit |= take
        ray_hits = int(hit.sum())

        # DIRECTION-FREE FALLBACK. The drop follows the TOOTH's long axis,
        # which is right for a bodily movement and wrong for a tipped one: on a
        # 6 degree tip the ray leaves at an angle and exits the side of the
        # cast, so 21 of 44 rim points reported "no cast beneath me" while
        # sitting over solid gingiva. Nearest-surface-point has no direction to
        # be wrong about, and is still bounded by the same envelope.
        need = ~hit
        if need.any():
            d_near, i_near = probe.tree.query(landing_seed[need], workers=-1)
            ok_near = d_near <= pol.max_reconstruction_depth_mm
            idx = np.where(need)[0]
            landing[idx[ok_near]] = probe.verts[i_near[ok_near]]
            hit[idx[ok_near]] = True
            retry_level[idx[ok_near]] = 4
        diag["ramp_landing_by_ray"] = ray_hits
        diag["ramp_landing_by_nearest_surface"] = int(hit.sum()) - ray_hits
        diag["ramp_landing_misses"] = int((~hit).sum())
        # 0 = landed at the full apron radius, 1-3 = the shrink ladder (the
        # apron COLLAPSED INWARD toward the rim to find tissue), 4 = the
        # direction-free nearest-surface fallback, -1 = never landed.
        diag["ramp_retry_levels"] = {
            str(k): int((retry_level[is_lifted] == k).sum())
            for k in (-1, 0, 1, 2, 3, 4)}
        diag["ramp_collapsed_inward_points"] = int(
            ((retry_level >= 1) & (retry_level <= 3) & is_lifted).sum())
        drop = np.linalg.norm(landing - landing_seed, axis=1)
        diag["ramp_landing_distance_max_mm"] = round(
            float(drop[is_lifted & hit].max()) if (is_lifted & hit).any() else 0.0, 4)
        diag["ramp_landing_distance_mean_mm"] = round(
            float(drop[is_lifted & hit].mean()) if (is_lifted & hit).any() else 0.0, 4)

        # A MISS STAYS A MISS. This used to be overwritten with `hit[:] = True`,
        # which made the refusal below unreachable and turned "no cast found"
        # into a silent success. A lifted point with nothing beneath it has no
        # tissue to bridge to, and that is a refusal.
        unreachable = is_lifted & (~hit)
        if unreachable.any():
            diag["unreachable_lifted_points"] = int(unreachable.sum())
            diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
            return InterfaceResult(False, "no_cast_beneath_target_rim",
                                   diagnostics=diag)

        # BLEND, DO NOT SWITCH. Assigning the landing only to lifted points
        # makes the bottom ring jump between two quite different positions at
        # the boundary between a lifted sector and a seated one, and the loft
        # twists across that step. Weighting by the point's own lift gives a
        # continuous ring, so a tipped tooth gets one smooth transition volume
        # rather than a bridge stitched to a seat.
        w = np.clip(lift / max(pol.fusion_overlap_mm, 1e-9), 0.0, 1.0)[:, None]
        bottom = bottom * (1.0 - w) + landing * w
        reach = float(np.linalg.norm(rim_k - bottom, axis=1).max())
        diag["ramp_reach_mm"] = round(reach, 4)
        if reach > pol.ramp_radius_mm + pol.max_reconstruction_depth_mm:
            diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
            return InterfaceResult(False, "outside_reconstruction_envelope",
                                   diagnostics=diag)

    # Top ring: buried INSIDE the crown. At rim_k exactly it would be the
    # crown's own cervical edge - a coincident surface, which is the tangency
    # this whole construction exists to avoid.
    top = (rim_k - inset_dir * pol.seat_top_inset_mm
           + u_oa_k * pol.fusion_overlap_mm)

    seat_v = seat_f = None
    try:
        sv_, sf_ = _loft(top, bottom)
        sf_ = cg.make_consistent_winding(sv_, sf_)
        if cg.is_edge_manifold_closed(sf_):
            vol = abs(cg.signed_volume(sv_, sf_))
            if vol > 1e-9:
                seat_v, seat_f = sv_, sf_
                diag["connector_volume_mm3"] = round(float(vol), 4)
                diag["seat_volume_mm3"] = round(float(vol), 4)   # legacy key
    except Exception as e:                                # noqa: BLE001
        diag["connector_error"] = f"{type(e).__name__}: {e}"

    if seat_v is None:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "interface_construction_failed",
                               diagnostics=diag)

    diag["connector_built"] = True
    diag["seat_built"] = True
    diag["ramp_built"] = bool(is_lifted.any())
    diag["seat_depth_mm"] = round(pol.seat_depth_mm, 4)
    diag["seat_lift_mm"] = round(pol.fusion_overlap_mm, 4)
    diag["seat_independent_of_root_length"] = True
    ramp_v = ramp_f = None      # folded into the connector

    diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
    # KEYWORDS, NOT POSITION. `diagnostics` sits after the geometry fields, so
    # passing it positionally landed it in `seat_verts` the moment the seat
    # fields were added - and the seat assignment on the next line then
    # overwrote it, silently discarding every diagnostic while the geometry
    # itself was fine. Nothing raised; `seat_built` simply vanished from the
    # manifest. Keyword arguments cannot drift like that.
    return InterfaceResult(
        ok=True, refusal_reason=None,
        cavity_verts=cav_v, cavity_faces=cav_f,
        ramp_verts=ramp_v, ramp_faces=ramp_f,
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

    Candidate triangles come from a KD-tree over centroids; the exact
    point-triangle distance is then computed for the nearest `candidates` of
    them. Brute force over every triangle would be correct too and about
    fifty times slower.
    """
    from scipy.spatial import cKDTree
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

def rigidity_report(pts_t0, pts_k, faces=None, sample=4000):
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
                     final_verts, final_faces, rim_t0, radius_mm=4.0):
    """Was the ORIGINAL socket restored, or replaced by a new defect?

    The T0 site is closed flush and then never touched again, so nothing in
    the stage pipeline would notice if that closure left a crater, a plateau
    or a step. This measures the restored site against the cast that surrounds
    it: deviation inside the old rim versus deviation in an annulus just
    outside it. A good restoration looks like its own neighbourhood.
    """
    from scipy.spatial import cKDTree
    orig = np.asarray(original_verts, float)
    fin = np.asarray(final_verts, float)
    rim = np.asarray(rim_t0, float)
    centre = rim.mean(axis=0)
    r_rim = float(np.linalg.norm(rim - centre, axis=1).max())

    d_fin = np.linalg.norm(fin - centre, axis=1)
    inside = fin[d_fin <= r_rim]
    annulus = fin[(d_fin > r_rim) & (d_fin <= r_rim + radius_mm)]
    if not len(inside) or not len(annulus):
        return {"measured": False,
                "reason": "old site not represented in the final mesh"}

    tree = cKDTree(orig)
    d_in, _ = tree.query(inside, workers=-1)
    d_out, _ = tree.query(annulus, workers=-1)
    return {
        "measured": True,
        "old_rim_radius_mm": round(r_rim, 4),
        "points_inside_old_site": int(len(inside)),
        "inside_max_deviation_mm": round(float(d_in.max()), 4),
        "inside_mean_deviation_mm": round(float(d_in.mean()), 4),
        "surrounding_max_deviation_mm": round(float(d_out.max()), 4),
        "surrounding_mean_deviation_mm": round(float(d_out.mean()), 4),
        # The restored site must not stray further from the original cast than
        # the untouched tissue immediately around it, by more than this margin.
        "excess_over_surroundings_mm": round(float(d_in.max() - d_out.max()), 4),
        "threshold_note": "engineering validation threshold for this prototype",
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
    info = {"collapsed_edges": 0, "removed_faces": 0,
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
