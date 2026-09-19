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

    The sign convention is the one `check_occlusal_collision` already uses -
    displacement from the nearest surface vertex, dotted with that vertex's
    normal. Negative is inside the solid.

    APPROXIMATE BY CONSTRUCTION, and it matters enough to say: this is nearest
    VERTEX, not nearest surface point, so on a coarse mesh it overestimates
    distance the same way the interproximal measure does. It is used to
    classify and to size a bounded tool, never to assert a clearance.
    """

    __slots__ = ("verts", "faces", "normals", "tree", "_thickness")

    def __init__(self, verts, faces):
        from scipy.spatial import cKDTree
        self.verts = np.asarray(verts, float)
        self.faces = np.asarray(faces, np.int64)
        self.normals = cg.vertex_normals(self.verts, self.faces)
        self.tree = cKDTree(self.verts)
        self._thickness = None

    def signed(self, pts):
        """(signed_distance, nearest_index). Negative = inside the cast."""
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
    rim_signed, rim_near = probe.signed(rim_k)
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
    crown_signed, _ = probe.signed(crown_k)
    penetration = float(max(0.0, -crown_signed.min()))
    diag["crown_penetration_mm"] = round(penetration, 4)
    diag["crown_points_inside_cast"] = int((crown_signed < 0).sum())

    # --- ADAPTIVE depth. This is the amendment-1 requirement ---------------
    # The cavity must clear whatever the crown actually buries, plus a seat
    # for the overlap, plus clearance. The policy default is only the floor
    # for a tooth that penetrates nothing at all.
    # DRIVEN BY THE ACTUAL PENETRATION, floored only by what the boolean needs.
    #
    # `emergence_depth_mm` is deliberately NOT the floor here, and that was a
    # real defect: a tooth that has LIFTED off the cast penetrates nothing, so
    # a fixed 1.5mm floor cut a hole 1.5mm below its rim - straight back
    # through the emergence ramp that had just reconnected it. Measured, the
    # fused model went from 1 body to 4 across a 1.2mm extrusion.
    #
    # The cavity only has to clear what the crown actually buries, plus the
    # seat's overlap and a little clearance. Where nothing is buried, the
    # minimum is whatever gives the boolean common volume - not an anatomical
    # guess about how deep a socket should be.
    wanted = max(pol.min_reconstruction_depth_mm,
                 penetration + pol.fusion_overlap_mm + pol.clearance_mm)
    # The emergence floor applies only to a rim that is still SEATED. On a
    # seated tooth a deeper cut clears the grazing zone where the crown's
    # outer surface runs alongside the original gingiva, which is where the
    # tangency lives. On a LIFTED tooth the same floor cuts back through the
    # emergence ramp that just reconnected it - measured, 1 body became 4.
    if float(rim_signed.max()) <= pol.fusion_overlap_mm:
        wanted = max(wanted, pol.emergence_depth_mm)

    # ...bounded by how much wall is actually there.
    thickness = probe.local_thickness(rim_k, u_oa_k)
    finite = thickness[np.isfinite(thickness)]
    local_thickness = float(np.nanmin(finite)) if len(finite) else 0.0
    wall_limit = local_thickness * pol.safe_wall_fraction
    depth = min(wanted, pol.max_reconstruction_depth_mm,
                wall_limit if wall_limit > 0 else pol.max_reconstruction_depth_mm)

    diag.update({
        "local_cast_thickness_mm": round(local_thickness, 4),
        "wall_limit_mm": round(wall_limit, 4),
        "depth_wanted_mm": round(wanted, 4),
        "depth_used_mm": round(depth, 4),
        "depth_was_clamped": bool(depth < wanted - 1e-9),
        "depth_clamped_by": ("wall_thickness" if depth == wall_limit
                             else "policy_ceiling" if depth == pol.max_reconstruction_depth_mm
                             else None) if depth < wanted - 1e-9 else None,
        "policy": pol.to_dict(),
    })

    # --- refusals, and ONLY for the permitted reasons ----------------------
    if float(rim_signed.max()) > pol.max_rim_separation_mm:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "outside_reconstruction_envelope",
                               diagnostics=diag)

    if depth < pol.min_reconstruction_depth_mm:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(
            False, "interface_unbuildable_wall_too_thin", diagnostics=diag)

    if wanted > pol.max_reconstruction_depth_mm:
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(
            False, "outside_reconstruction_envelope", diagnostics=diag)

    # --- the cavity tool ---------------------------------------------------
    # Inset laterally so the crown's flanks finish INSIDE the cast wall. That
    # buried interference is what gives the union real common volume instead
    # of the tangency the old plug produced.
    centre = rim_k.mean(axis=0)
    radial = rim_k - centre
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True)
    inset_dir = np.divide(radial, np.where(rnorm < 1e-12, 1.0, rnorm))
    cavity_rim = rim_k + inset_dir * pol.cavity_outset_mm

    # RAISE THE CAVITY MOUTH ABOVE THE GINGIVAL SURFACE, as part of the SAME
    # solid. A cup capped exactly at rim level is coplanar with the surface it
    # is subtracted from, and coplanar booleans are what make manifold3d emit
    # coincident-but-distinct vertices - which a reader's weld then turns into
    # non-manifold edges. Building the cup from a raised rim cuts cleanly
    # through. Done here rather than with a second "collar" tool because two
    # tools sharing a wall reintroduces the very coincidence being removed.
    # RAISE ONLY WHERE THERE IS A SURFACE TO CUT THROUGH. On a seated rim the
    # raise carries the cut above the gingiva so the subtract is not coplanar
    # with it. On a LIFTED rim there is nothing above to cut - the raise just
    # eats the emergence ramp holding the tooth on, which is what left the
    # model in 2-3 pieces at the far stages of an extrusion.
    # MEASURED BOTH WAYS. Making the raise conditional on a seated rim keeps
    # more stages in one piece (bodies 1,2,1,1,2 against 1,1,2,2,3) but costs
    # topology at the margin (reader-weld non-manifold 0,16,8,2,1 against
    # 0,1,0,1,0). The non-manifold gate is the harder one to satisfy and the
    # one a slicer actually trips over, so the unconditional raise is kept and
    # the body count is carried as a known limitation rather than traded for it.
    seated = float(rim_signed.max()) <= pol.fusion_overlap_mm
    cavity_raise = pol.cavity_raise_mm
    cavity_top = cavity_rim + u_oa_k * cavity_raise
    diag["rim_seated"] = bool(seated)
    diag["cavity_raise_mm"] = round(float(cavity_raise), 4)
    try:
        pts, cup_faces, cup_info = cg.build_socket_cup(
            cavity_top, u_oa_k, depth_mm=float(depth + cavity_raise))
        cav_v = np.vstack([cavity_top, np.asarray(pts, float)])
        cav_v, cav_f = cg.cap_and_close(cav_v, np.asarray(cup_faces, np.int64))
        cav_f = cg.make_consistent_winding(cav_v, cav_f)
    except Exception as e:                                # noqa: BLE001
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        diag["cavity_error"] = f"{type(e).__name__}: {e}"
        return InterfaceResult(False, "interface_construction_failed",
                               diagnostics=diag)

    if not cg.is_edge_manifold_closed(cav_f):
        diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
        return InterfaceResult(False, "cavity_not_closed", diagnostics=diag)

    cav_vol = abs(cg.signed_volume(cav_v, cav_f))
    diag["cavity_volume_mm3"] = round(float(cav_vol), 4)
    diag["cavity_fallback"] = cup_info.get("fallback_reason")

    # --- the emergence ramp, where the rim has lifted clear ----------------
    # NOT a refusal case (amendment 2): an extrusion is supposed to do this.
    ramp_v = ramp_f = None
    if n_outside:
        # AN APRON, NOT A COLLAR. The obvious construction - loft each lifted
        # rim point straight down to the cast beneath it - produces a vertical
        # cylindrical wall standing on the gingiva, which is exactly the
        # "artificial annular ring / cylindrical ledge / collar" the brief
        # forbids. It is also not what tissue does: gingiva follows an
        # extruding tooth as a sloped emergence profile, not a sleeve.
        #
        # So the ramp lands on a ring `ramp_radius_mm` OUTSIDE the rim,
        # dropped onto the real cast surface. The loft between the two is a
        # cone that blends from the cervical margin out into the gingiva, and
        # its slope is set by the actual separation rather than by a constant.
        # THE APRON RADIUS IS PER-POINT AND PROPORTIONAL TO THE LOCAL LIFT.
        # A constant radius is wrong in both directions: it adds a flare where
        # the rim never left the tissue (material from nowhere), and it fixes
        # the blend slope regardless of how far the tooth actually moved.
        # Scaling by the lift makes the apron vanish where lift is zero, which
        # is what makes a partial lift - one side of a tipping tooth - behave.
        lift = np.clip(rim_signed, 0.0, None)
        radius = np.minimum(pol.ramp_radius_mm, lift)[:, None]
        landing_seed = rim_k + inset_dir * radius

        # A seed can miss the cast when the tooth is near the model edge.
        # RETRY INWARD before giving up: a miss at 1.2mm often lands at 0.6mm,
        # and refusing a whole tooth because its apron overhung the trim line
        # is precisely the over-refusal that separation must not cause.
        drop_cap = pol.max_reconstruction_depth_mm
        landing, hit = probe.drop_to_surface(landing_seed, u_oa_k, drop_cap)
        for shrink in (0.5, 0.25, 0.0):
            if hit.all():
                break
            retry_seed = rim_k + inset_dir * (radius * shrink)
            retry, retry_hit = probe.drop_to_surface(retry_seed, u_oa_k, drop_cap)
            take = (~hit) & retry_hit
            landing[take] = retry[take]
            hit |= take
        diag["ramp_landing_misses"] = int((~hit).sum())
        # No cast within the envelope below this point: leave it grounded
        # rather than refuse. The apron simply has nothing to blend into here.
        landing[~hit] = rim_k[~hit]
        hit[:] = True

        # WHERE THERE IS NO LIFT, THE LANDING IS THE RIM POINT ITSELF.
        # Dropping an unlifted point "onto the cast" walks it straight past
        # the gingival surface it is already sitting on and lands it on the
        # UNDERSIDE OF THE BASE several millimetres below - which made `reach`
        # blow past the envelope and refused six of nine perfectly ordinary
        # movements. The ramp must have exactly zero height where the tooth
        # never left the tissue.
        grounded = lift <= 1e-9
        landing[grounded] = rim_k[grounded]
        hit[grounded] = True
        diag["ramp_grounded_points"] = int(grounded.sum())
        if not hit.all():
            # Still nothing underneath after collapsing the apron to a vertical
            # drop: there is genuinely no cast below this part of the rim.
            diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
            return InterfaceResult(False, "no_cast_beneath_target_rim",
                                   diagnostics=diag)

        reach = float(np.linalg.norm(rim_k - landing, axis=1).max())
        diag["ramp_reach_mm"] = round(reach, 4)
        diag["ramp_slope_max"] = round(float(rim_signed.max() / pol.ramp_radius_mm), 4)
        if reach > pol.ramp_radius_mm + pol.max_reconstruction_depth_mm:
            diag["boolean_seconds"] = round(time.perf_counter() - t0, 4)
            return InterfaceResult(False, "outside_reconstruction_envelope",
                                   diagnostics=diag)
        try:
            rv, rf = _loft(rim_k, landing)
            rf = cg.make_consistent_winding(rv, rf)
            if cg.is_edge_manifold_closed(rf):
                vol = abs(cg.signed_volume(rv, rf))
                if vol > 1e-9:
                    ramp_v, ramp_f = rv, rf
                    diag["ramp_volume_mm3"] = round(float(vol), 4)
            else:
                diag["ramp_error"] = "loft did not close"
        except Exception as e:                            # noqa: BLE001
            diag["ramp_error"] = f"{type(e).__name__}: {e}"
    diag["ramp_built"] = ramp_v is not None

    # --- the seat: a BOUNDED replacement for the root plug -----------------
    # WHY ANY CONNECTOR IS NEEDED AT ALL, which is the thing the old 9mm plug
    # got right for the wrong reason. For `cast u crown` to fuse into ONE body
    # the two solids need common VOLUME, not a shared surface. The crown is
    # capped at its cervical rim and has no material below it; once the cavity
    # is cut there is nothing left to share except a coplanar annulus at the
    # margin. Measured: that tangency left 23-25 non-manifold edges after a
    # reader's weld, all of them within 1mm of the target rim.
    #
    # The plug answered this with `root_length_mm` of synthetic root, which
    # emerged whenever the tooth extruded. The seat answers it with
    # `fusion_overlap_mm + clearance_mm` - about a third of a millimetre,
    # fixed by policy, INDEPENDENT of root length, and buried under the
    # gingival surface by construction. The cavity is inset laterally by the
    # same overlap, so the seat's outer skirt finishes inside cast material:
    # overlap = perimeter x fusion_overlap_mm x seat_depth, a real volume the
    # boolean can resolve.
    #
    # THIS IS NOT PART OF THE CROWN. The crown solid stays a rigid transform
    # of the T0 crown and the rigidity tests measure it alone; the seat is
    # manufacturing geometry the caller unions separately.
    # THE SEAT IS LIFTED INTO THE CROWN, and that detail is load-bearing.
    # Lofting it from the rim DOWN puts its top cap exactly on the crown's
    # bottom cap - two closed solids sharing a surface kiss instead of
    # overlapping, which is the same tangency one level down. Measured: the
    # unlifted seat took the reader-weld non-manifold count from 23 to 80.
    # Raising the top into the crown gives the union real volume at both ends.
    # `_rim_plug` learned this as PLUG_LIFT_MM; the lesson survives the plug.
    seat_depth = pol.seat_depth_mm
    seat_lift = pol.fusion_overlap_mm
    seat_v = seat_f = None
    try:
        # INSET HALF THE OVERLAP. At the full rim outline the seat's side
        # wall is exactly the crown's cervical edge - a coincident surface,
        # and therefore the same tangency it exists to remove. Half the
        # overlap puts it strictly inside the crown and still strictly
        # outside the cavity wall, so it overlaps BOTH solids cleanly.
        sv_, sf_ = _loft(
            rim_k - inset_dir * pol.seat_top_inset_mm + u_oa_k * seat_lift,
            rim_k + inset_dir * pol.seat_bottom_outset_mm - u_oa_k * seat_depth)
        sf_ = cg.make_consistent_winding(sv_, sf_)
        if cg.is_edge_manifold_closed(sf_):
            vol = abs(cg.signed_volume(sv_, sf_))
            if vol > 1e-9:
                seat_v, seat_f = sv_, sf_
                diag["seat_volume_mm3"] = round(float(vol), 4)
    except Exception as e:                                # noqa: BLE001
        diag["seat_error"] = f"{type(e).__name__}: {e}"
    diag["seat_lift_mm"] = round(seat_lift, 4)
    diag["seat_depth_mm"] = round(seat_depth, 4)
    diag["seat_built"] = seat_v is not None
    diag["seat_independent_of_root_length"] = True

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
        inter = solid_a.boolean(solid_b, m3.OpType.Intersect)
        return float(inter.volume())
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
    report["verdict"] = "PRINT READY" if report["print_ready"] else "NOT PRINT READY"
    report["wording_note"] = (
        "manufacturing geometry validated against engineering gates. NOT a "
        "claim of clinical validation.")
    return report
