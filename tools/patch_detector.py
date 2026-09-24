import sys
with open("self_intersection.py", "r") as f:
    code = f.read()

old_block = """    count = span.prod(axis=1)

    if active is not None:
        active = np.asarray(active, bool)

    fid = np.repeat(np.arange(len(F)), count)
    start = np.repeat(np.cumsum(count) - count, count)
    local = np.arange(count.sum()) - start
    sy, sz = span[fid, 1], span[fid, 2]
    ix = ilo[fid, 0] + local // (sy * sz)
    rem = local % (sy * sz)
    iy = ilo[fid, 1] + rem // sz
    iz = ilo[fid, 2] + rem % sz
    dims = ihi.max(axis=0) + 1
    key = (ix * dims[1] + iy) * dims[2] + iz

    order = np.lexsort((fid, key))
    key, fid = key[order], fid[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts = np.r_[0, bounds]
    sizes = np.diff(np.r_[starts, len(key)])

    out = []
    for g in np.unique(sizes[sizes > 1]):
        grp = starts[sizes == g]
        members = fid[grp[:, None] + np.arange(g)[None, :]]      # (n_cells, g)
        if active is not None:
            keep = active[members].any(axis=1)
            members = members[keep]
            if not len(members):
                continue
        iu, ju = np.triu_indices(g, k=1)
        a = members[:, iu].ravel()
        b = members[:, ju].ravel()
        out.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]))"""

new_block = """    count = span.prod(axis=1)

    if active is not None:
        active = np.asarray(active, bool)

    is_giant = count > 1000
    normal_idx = np.flatnonzero(~is_giant)
    giant_idx = np.flatnonzero(is_giant)

    out = []

    if len(normal_idx) > 0:
        n_count = count[normal_idx]
        fid = np.repeat(normal_idx, n_count)
        start = np.repeat(np.cumsum(n_count) - n_count, n_count)
        local = np.arange(n_count.sum()) - start
        
        n_fid = np.repeat(np.arange(len(normal_idx)), n_count)
        sy, sz = span[fid, 1], span[fid, 2]
        ix = ilo[fid, 0] + local // (sy * sz)
        rem = local % (sy * sz)
        iy = ilo[fid, 1] + rem // sz
        iz = ilo[fid, 2] + rem % sz
        dims = ihi.max(axis=0) + 1
        key = (ix * dims[1] + iy) * dims[2] + iz

        order = np.lexsort((fid, key))
        key, fid = key[order], fid[order]
        bounds = np.flatnonzero(np.diff(key)) + 1
        starts = np.r_[0, bounds]
        sizes = np.diff(np.r_[starts, len(key)])

        for g in np.unique(sizes[sizes > 1]):
            grp = starts[sizes == g]
            members = fid[grp[:, None] + np.arange(g)[None, :]]
            if active is not None:
                keep = active[members].any(axis=1)
                members = members[keep]
                if not len(members):
                    continue
            iu, ju = np.triu_indices(g, k=1)
            a = members[:, iu].ravel()
            b = members[:, ju].ravel()
            out.append(np.column_stack([np.minimum(a, b), np.maximum(a, b)]))

    if len(giant_idx) > 0:
        for i in giant_idx:
            ov = np.all((lo[i] <= hi) & (lo <= hi[i]), axis=1)
            ov[i] = False
            if active is not None and not active[i]:
                ov &= active
            ov_idx = np.flatnonzero(ov)
            if len(ov_idx) > 0:
                a = np.full(len(ov_idx), i, dtype=np.int64)
                out.append(np.column_stack([np.minimum(a, ov_idx), np.maximum(a, ov_idx)]))"""

if old_block in code:
    code = code.replace(old_block, new_block)
    with open("self_intersection.py", "w") as f:
        f.write(code)
    print("Patched!")
else:
    print("Old block not found!")
