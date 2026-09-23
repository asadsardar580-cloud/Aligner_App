import re

with open('core_geometry.py', 'r') as f:
    orig = f.read()

new_func = """def build_cast_base(verts: np.ndarray, faces: np.ndarray, arch_frame: dict,
                    base_thickness_mm: float = CAST_BASE_THICKNESS_MM,
                    rim: np.ndarray | None = None,
                    protected_faces: np.ndarray | None = None):
    import self_intersection as si
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)

    out_f = faces.copy()
    out_rim = np.asarray(rim, int) if rim is not None else None
    
    # If the caller explicitly passed a rim, they are asserting that the wall
    # MUST be built from it, bypassing iterative repair.
    skip_repair = (rim is not None)

    if protected_faces is not None:
        face_hash = {tuple(sorted(f)): i for i, f in enumerate(faces)}

    max_iters = 50
    total_pruned_scan_faces = 0

    for iteration in range(max_iters):
        if out_rim is None:
            loops = boundary_loops(out_f)
            if len(loops) != 1:
                raise ValueError(
                    f"build_cast_base needs exactly one boundary loop and got {len(loops)} "
                    f"(sizes {sorted((len(L) for L in loops), reverse=True)[:5]}). Run "
                    f"trim_to_arch first  it fills interior holes and guarantees this.")
            out_rim = np.asarray(loops[0], int)

        if len(np.unique(out_rim)) != len(out_rim):
            dup = len(out_rim) - len(np.unique(out_rim))
            raise ValueError(
                f"The trimmed rim passes through {dup} vertex/vertices twice, so the cast "
                f"surface pinches to a point on its own margin and there is no single wall "
                f"to extrude. Adjust the trim margin so the band does not neck.")

        origin, e1, e2, u_occ = _arch_basis(arch_frame)
        rel = verts - origin
        height = rel @ u_occ
        Q = np.column_stack([rel[out_rim] @ e1, rel[out_rim] @ e2])

        surv, pinfo = prune_to_simple(Q)
        floor2d = Q[surv]
        tris, forced_clips = _floor_triangulation(floor2d)
        if len(tris) != len(floor2d) - 2:
            raise ValueError(
                f"Ear clipping the floor outline returned {len(tris)} triangles where "
                f"{len(floor2d) - 2} are owed, so it bailed on a polygon it could not "
                f"triangulate and the floor would have holes.")

        used = np.unique(out_f)
        plane_h = float(height[used].min()) - float(base_thickness_mm)
        floor_xyz = (origin + np.outer(floor2d[:, 0], e1) + np.outer(floor2d[:, 1], e2)
                     + u_occ * plane_h)

        n_v = len(verts)
        image = np.searchsorted(surv, np.arange(len(out_rim)), side="right") - 1
        image[image < 0] = len(surv) - 1

        m = len(out_rim)
        wall = []
        for i in range(m):
            a, b = int(out_rim[i]), int(out_rim[(i + 1) % m])
            ai, bi = n_v + int(image[i]), n_v + int(image[(i + 1) % m])
            if ai == bi:
                wall.append([a, b, ai])
            else:
                wall.append([a, b, bi])
                wall.append([a, bi, ai])
        wall = np.asarray(wall, int)
        floor = np.asarray(tris, int) + n_v

        half = _boundary_half_edges(out_f)
        if (int(out_rim[0]), int(out_rim[1])) in half:
            wall = wall[:, ::-1]
            floor = floor[:, ::-1]

        out_v = np.vstack([verts, floor_xyz])
        total_f = np.vstack([out_f, wall, floor])

        if skip_repair:
            break

        pairs = si.candidate_pairs(out_v, total_f)
        hit, _ = si._pairs_intersect(out_v, total_f, pairs, 1e-6)
        intersecting = pairs[hit]
        
        n_scan = len(out_f)
        n_wall = len(wall)
        bad_faces = set()
        
        for a, b in intersecting:
            ta = "scan" if a < n_scan else ("wall" if a < n_scan + n_wall else "floor")
            tb = "scan" if b < n_scan else ("wall" if b < n_scan + n_wall else "floor")
            if ta == "scan" and tb != "scan": bad_faces.add(a)
            if tb == "scan" and ta != "scan": bad_faces.add(b)
            
        if not bad_faces:
            break
            
        if protected_faces is not None:
            for idx in bad_faces:
                orig_idx = face_hash[tuple(sorted(out_f[idx]))]
                if protected_faces[orig_idx]:
                    raise ValueError(f"Undercut fix cannot converge without entering the protected band")
            
        fmask = np.ones(len(out_f), dtype=bool)
        fmask[list(bad_faces)] = False
        
        n_c, lab = _face_components(out_f[fmask], len(verts))
        idx = np.where(fmask)[0]
        biggest = int(np.bincount(lab).argmax())
        new_fmask = np.zeros(len(out_f), bool)
        new_fmask[idx[lab == biggest]] = True
        fmask = new_fmask

        for _ in range(4):
            try:
                loops = boundary_loops(out_f[fmask])
                if not loops: break
                rim_loop = max(loops, key=len)
            except Exception:
                break
            sub_keep, opened = _open_pinch_vertices(out_f[fmask], rim_loop)
            if not opened: break
            drop = np.where(fmask)[0][~sub_keep]
            fmask[drop] = False
            
            n_c, lab = _face_components(out_f[fmask], len(verts))
            idx = np.where(fmask)[0]
            biggest = int(np.bincount(lab).argmax())
            new_fmask = np.zeros(len(out_f), bool)
            new_fmask[idx[lab == biggest]] = True
            fmask = new_fmask
            
        total_pruned_scan_faces += (~fmask).sum()
        out_f = out_f[fmask]
        verts, out_f, _, out_rim = _fill_interior_holes(verts, out_f)
    else:
        raise ValueError(f"build_cast_base failed to converge after {max_iters} undercut repair iterations.")

    if signed_volume(out_v, total_f) < 0:
        total_f = total_f[:, ::-1]

    health = manifold_report(total_f)
    if health["open_edges"] or health["nonmanifold_edges"]:
        raise ValueError(
            f"The extruded cast base did not close")
    if not _winding_is_consistent(total_f):
        raise ValueError(
            "The extruded cast base closed but its winding is inconsistent")

    n_comp, _lab = _face_components(total_f, len(out_v))
    if n_comp != 1:
        raise ValueError(
            f"The cast base is {n_comp} disconnected solids")

    volume = signed_volume(out_v, total_f)
    if volume <= 0:
        raise ValueError(f"The cast base encloses {volume:.1f} mm3, which is not a solid.")

    return out_v, total_f, {
        "base_thickness_mm": float(base_thickness_mm),
        "base_plane_offset_mm": float(plane_h),
        "lowest_surface_mm": float(height[used].min()),
        "wall_tris": int(len(wall)),
        "floor_tris": int(len(floor)),
        "rim_points": int(len(out_rim) if out_rim is not None else 0),
        "floor_points": int(len(floor2d)),
        "gap_fraction": round(1.0 - len(floor2d) / max(len(out_rim) if out_rim is not None else 1, 1), 4),
        "pruned_vertices": pinfo["pruned_vertices"],
        "repairs": pinfo["repairs"],
        "forced_clips": int(forced_clips),
        "components": int(n_comp),
        "prune": pinfo,
        "open_edges": int(health["open_edges"]),
        "nonmanifold_edges": int(health["nonmanifold_edges"]),
        "total_edges": int(health["total_edges"]),
        "area_error": pinfo["area_error"],
        "worst_arc_fraction": pinfo["worst_arc_fraction"],
        "watertight": True,
        "winding_consistent": True,
        "volume_mm3": round(volume, 3),
        "faces": int(len(total_f)),
        "vertices": int(len(out_v)),
        "undercut_iterations": iteration if not skip_repair else 0,
        "undercut_scan_faces_removed": int(total_pruned_scan_faces),
    }"""

match = re.search(r'^def build_cast_base\(', orig, re.MULTILINE)
start_idx = match.start()
match2 = re.search(r'^def offset_along_normals\(', orig[start_idx+10:], re.MULTILINE)
end_idx = start_idx + 10 + match2.start() if match2 else len(orig)

new_orig = orig[:start_idx] + new_func + '\n\n' + orig[end_idx:]

with open('core_geometry.py', 'w') as f:
    f.write(new_orig)
