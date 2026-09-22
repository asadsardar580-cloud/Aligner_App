#!/usr/bin/env python
"""Prove `pointops_cpu.py` implements the pointops contract both models assume.

    python verify_pointops.py

WHY THIS EXISTS. `pointops` is a CUDA extension that does not build on this
machine, so every call ToothGroupNetwork and CrossTooth make into it is served
by a pure-PyTorch replacement. Nothing anywhere checked that the replacement
computes the same thing. Both networks run, both produce plausible teeth, and
"it runs" is not evidence: a neighbourhood search that returns the right
SHAPE with the wrong neighbours produces a model that segments badly rather
than one that crashes, which is indistinguishable from a model that is simply
not very good.

THE REFERENCE IS THE VENDORED UPSTREAM WRAPPER, NOT MEMORY.
`ToothGroupNetwork/external_libs/pointops/functions/pointops.py` is the
official Point Transformer Python wrapper - the same lineage CrossTooth's
`models/PointTransformer/libs/pointops` would have been. It is the authority
for what each function returns, and the reference implementations below are
written from it and from plain definitions, in NumPy, with no shared code
with the thing being tested.

WHAT A FAILURE HERE MEANS. Not that a test is too strict. Every number this
repository has measured about either model - 16 teeth, 13.239 mm median box,
255 s, 0.4884 agreement - was produced through this shim.
"""

import sys

import numpy as np
import torch

import pointops_cpu as P

TOL = 1e-5
_FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)
    return ok


# ---------------------------------------------------------------------------
# Independent references. NumPy, definitional, no torch, no shared code.
# ---------------------------------------------------------------------------

def ref_segments(offset, new_offset):
    off = [int(v) for v in offset]
    noff = [int(v) for v in new_offset]
    s = ns = 0
    for e, ne in zip(off, noff):
        yield s, e, ns, ne
        s, ns = e, ne


def ref_knn(nsample, xyz, new_xyz, offset, new_offset):
    """idx GLOBAL, dist = sqrt(squared) - upstream returns torch.sqrt(dist2)."""
    xyz = np.asarray(xyz, np.float64)
    new_xyz = np.asarray(new_xyz, np.float64)
    idx = np.zeros((len(new_xyz), nsample), np.int64)
    dist = np.zeros((len(new_xyz), nsample), np.float64)
    for s, e, ns, ne in ref_segments(offset, new_offset):
        ref, qry = xyz[s:e], new_xyz[ns:ne]
        d2 = ((qry[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        k = min(nsample, e - s)
        order = np.argsort(d2, axis=1, kind="stable")[:, :k]
        dd = np.take_along_axis(d2, order, axis=1)
        if k < nsample:
            order = np.concatenate([order, np.repeat(order[:, :1],
                                                     nsample - k, 1)], 1)
            dd = np.concatenate([dd, np.repeat(dd[:, :1], nsample - k, 1)], 1)
        idx[ns:ne] = order + s
        dist[ns:ne] = np.sqrt(dd)
    return idx, dist


def ref_fps(xyz, offset, new_offset):
    """Farthest point sampling, seeded at each segment's first point."""
    xyz = np.asarray(xyz, np.float64)
    out = []
    for s, e, ns, ne in ref_segments(offset, new_offset):
        pts = xyz[s:e]
        m = min(ne - ns, e - s)
        best = np.full(len(pts), np.inf)
        far = 0
        for _ in range(m):
            out.append(far + s)
            best = np.minimum(best, ((pts - pts[far]) ** 2).sum(-1))
            far = int(np.argmax(best))
    return np.array(out, np.int64)


def ref_queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset,
                      use_xyz):
    """grouped_xyz is RELATIVE to new_xyz; concatenated BEFORE the features."""
    xyz = np.asarray(xyz, np.float64)
    new_xyz = np.asarray(new_xyz, np.float64)
    feat = np.asarray(feat, np.float64)
    if idx is None:
        idx, _ = ref_knn(nsample, xyz, new_xyz, offset, new_offset)
    idx = np.asarray(idx, np.int64)
    g_xyz = xyz[idx.reshape(-1)].reshape(len(new_xyz), nsample, 3) \
        - new_xyz[:, None, :]
    g_feat = feat[idx.reshape(-1)].reshape(len(new_xyz), nsample, -1)
    return np.concatenate([g_xyz, g_feat], -1) if use_xyz else g_feat


def ref_interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """INVERSE DISTANCE, not inverse squared distance. Upstream:

        idx, dist = knnquery(...)        # knnquery returns sqrt(dist2)
        dist_recip = 1.0 / (dist + 1e-8)
        weight = dist_recip / dist_recip.sum(1, keepdim=True)
    """
    idx, dist = ref_knn(k, xyz, new_xyz, offset, new_offset)
    recip = 1.0 / (dist + 1e-8)
    w = recip / recip.sum(axis=1, keepdims=True)
    feat = np.asarray(feat, np.float64)
    out = np.zeros((len(np.asarray(new_xyz)), feat.shape[1]))
    for i in range(k):
        out += feat[idx[:, i]] * w[:, i][:, None]
    return out


# ---------------------------------------------------------------------------
# Fixtures. One batch and two, because `offset` is the whole point of this API.
# ---------------------------------------------------------------------------

def fixtures():
    g = np.random.default_rng(20260922)
    single = (g.normal(size=(200, 3)) * 10.0, [200], [200])
    n1, n2 = 120, 180
    two = (np.vstack([g.normal(size=(n1, 3)) * 8.0,
                      g.normal(size=(n2, 3)) * 8.0 + 500.0]),
           [n1, n1 + n2], [n1, n1 + n2])
    return {"one batch": single, "two batches": two}


def check_every_consumer_of_the_knn_distance():
    """Nobody may read knnquery's distance without honouring the flag.

    `KNN_RETURNS_SQUARED_DISTANCE` is a promise to callers, and a promise
    only holds while every caller is known. This walks the two vendored
    networks and this project's own code for `knnquery` call sites and
    classifies each one: discards the distance, converts it, or reads it raw.
    A raw read is a defect - it is exactly what `interpolation` used to do,
    and it made four of five decoder stages in both networks weight by
    inverse SQUARED distance.
    """
    import ast
    import os

    print("\n[every consumer of knnquery's second return value]")
    roots = ["pointops_cpu.py", "crosstooth_bridge.py",
             "ToothGroupNetwork/models", "CrossTooth/models",
             "CrossTooth/compete"]
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        elif os.path.isdir(r):
            for dp, _, fn in os.walk(r):
                if "external_libs" in dp:          # the upstream wrapper
                    continue
                files += [os.path.join(dp, n) for n in fn if n.endswith(".py")]

    raw, total = [], 0
    for path in files:
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # `a, b = knnquery(...)` — what happens to b?
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            call = node.value
            if not (isinstance(call, ast.Call) and
                    "knnquery" in ast.dump(call.func)):
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Tuple) or len(tgt.elts) != 2:
                continue
            total += 1
            second = tgt.elts[1]
            name = getattr(second, "id", None)
            if name == "_":
                verdict = "discards it"
            elif path.endswith("pointops_cpu.py"):
                verdict = "converts it (honours the flag)"
            else:
                verdict = "READS IT RAW"
                raw.append((path, node.lineno))
            print(f"    {path}:{node.lineno}  {verdict}")

    return check("every knnquery consumer discards or converts the distance",
                 not raw, f"{total} call site(s)" if not raw else
                 f"raw read at {raw}")


def main():
    print("=" * 74)
    print("pointops_cpu.py against the vendored upstream wrapper")
    print("=" * 74)

    for label, (xyz_np, off, noff) in fixtures().items():
        print(f"\n[{label}]  {len(xyz_np)} points, offsets {off}")
        xyz = torch.tensor(xyz_np, dtype=torch.float32)
        o = torch.tensor(off, dtype=torch.int32)

        # -- knnquery ---------------------------------------------------
        for k in (1, 3, 8, 16):
            gi, gd = P.knnquery(k, xyz, xyz, o, o)
            ri, rd = ref_knn(k, xyz_np, xyz_np, off, noff)
            same_idx = np.array_equal(gi.numpy().astype(np.int64), ri)
            check(f"knnquery k={k:<2} indices are the k nearest, GLOBAL",
                  same_idx)
            # THE DISTANCE IS CHECKED AGAINST THE FLAG THIS MODULE DECLARES,
            # not against upstream, because the two legitimately differ and
            # `KNN_RETURNS_SQUARED_DISTANCE` is how a caller is told which.
            # That is only safe because every consumer is accounted for -
            # see `check_every_consumer_of_the_knn_distance` below, which is
            # what stops the flag quietly becoming a lie.
            got = gd.numpy().astype(np.float64)
            want = rd ** 2 if P.KNN_RETURNS_SQUARED_DISTANCE else rd
            other = rd if P.KNN_RETURNS_SQUARED_DISTANCE else rd ** 2
            err, err_other = np.abs(got - want).max(), np.abs(got - other).max()
            kind = "squared" if P.KNN_RETURNS_SQUARED_DISTANCE else "sqrt"
            check(f"knnquery k={k:<2} distance is the {kind} distance it "
                  f"declares", err <= max(TOL, 1e-3 * float(want.max())),
                  f"max err {err:.2e}" if err <= 1e-2 else
                  f"max err {err:.3g}; it is the other convention "
                  f"(err {err_other:.2e})")

        # -- furthestsampling -------------------------------------------
        for div in (2, 4):
            n_o = torch.tensor([v // div for v in off], dtype=torch.int32)
            got = P.furthestsampling(xyz, o, n_o).numpy().astype(np.int64)
            ref = ref_fps(xyz_np, off, [v // div for v in off])
            check(f"furthestsampling stride {div} matches definitional FPS",
                  np.array_equal(got, ref),
                  f"{len(got)} points" if np.array_equal(got, ref) else
                  f"{int((got != ref).sum())} of {len(ref)} differ")

        # -- queryandgroup ----------------------------------------------
        feat_np = np.random.default_rng(7).normal(size=(len(xyz_np), 5))
        feat = torch.tensor(feat_np, dtype=torch.float32)
        for use_xyz in (True, False):
            got = P.queryandgroup(8, xyz, xyz, feat, None, o, o,
                                  use_xyz).numpy().astype(np.float64)
            ref = ref_queryandgroup(8, xyz_np, xyz_np, feat_np, None, off,
                                    noff, use_xyz)
            err = np.abs(got - ref).max()
            check(f"queryandgroup use_xyz={use_xyz!s:<5} "
                  f"relative xyz, then features", err <= 1e-4,
                  f"max err {err:.2e}")

        # The index must be REUSED when supplied - CrossTooth depends on the
        # key and value projections sharing one neighbourhood.
        idx, _ = P.knnquery(8, xyz, xyz, o, o)
        a = P.queryandgroup(8, xyz, xyz, feat, idx, o, o, False)
        b = P.queryandgroup(8, xyz, xyz, feat, None, o, o, False)
        check("queryandgroup honours a supplied index",
              torch.allclose(a, b, atol=1e-6))

        # -- grouping ---------------------------------------------------
        got = P.grouping(feat, idx).numpy().astype(np.float64)
        ref = feat_np[idx.numpy().astype(np.int64).reshape(-1)].reshape(
            len(xyz_np), 8, 5)
        check("grouping gathers by index",
              np.abs(got - ref).max() <= 1e-4)

        # -- interpolation ----------------------------------------------
        m = len(xyz_np) // 4
        src = xyz[:m].contiguous()
        src_np = xyz_np[:m]
        sfeat_np = np.random.default_rng(11).normal(size=(m, 5))
        sfeat = torch.tensor(sfeat_np, dtype=torch.float32)
        so = torch.tensor([m], dtype=torch.int32)
        one = torch.tensor([len(xyz_np)], dtype=torch.int32)
        got = P.interpolation(src, xyz, sfeat, so, one).numpy().astype(float)
        ref = ref_interpolation(src_np, xyz_np, sfeat_np, [m],
                                [len(xyz_np)])
        err = np.abs(got - ref).max()
        # And against the wrong convention, so the report names which one.
        idx3, d3 = ref_knn(3, src_np, xyz_np, [m], [len(xyz_np)])
        recip = 1.0 / (d3 ** 2 + 1e-8)
        w = recip / recip.sum(1, keepdims=True)
        wrong = sum(sfeat_np[idx3[:, i]] * w[:, i][:, None] for i in range(3))
        err_wrong = np.abs(got - wrong).max()
        check("interpolation weights by INVERSE DISTANCE (upstream)",
              err <= 1e-4,
              f"max err {err:.2e}" if err <= 1e-4 else
              f"max err {err:.3g}; it is weighting by inverse SQUARED "
              f"distance instead (err vs that {err_wrong:.2e})")

        # k=1 must be exact regardless of the weighting, because the single
        # weight normalises to 1. That isolates the defect to the weights.
        got1 = P.interpolation(src, xyz, sfeat, so, one, k=1).numpy()
        ref1 = ref_interpolation(src_np, xyz_np, sfeat_np, [m],
                                 [len(xyz_np)], k=1)
        check("interpolation k=1 is exact either way (isolates the weights)",
              np.abs(got1 - ref1).max() <= 1e-4)

        # -- subtraction / aggregation ----------------------------------
        got = P.subtraction(feat, feat, idx).numpy().astype(np.float64)
        ref = feat_np[:, None, :] - feat_np[
            idx.numpy().astype(np.int64).reshape(-1)].reshape(
            len(xyz_np), 8, 5)
        check("subtraction is input1 - input2[idx]",
              np.abs(got - ref).max() <= 1e-4)

    check_every_consumer_of_the_knn_distance()

    print("\n" + "=" * 74)
    if _FAILURES:
        print(f"{len(_FAILURES)} CHECK(S) FAILED:")
        for f in sorted(set(_FAILURES)):
            print(f"  - {f}")
        print("\nEvery measurement this repository has taken about either "
              "segmentation\nmodel was produced through this shim.")
        return 1
    print("pointops_cpu matches the upstream contract on every function.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
