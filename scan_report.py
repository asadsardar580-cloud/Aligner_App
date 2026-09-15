#!/usr/bin/env python3
"""
scan_report.py - measure YOUR scan locally, send me the numbers.

    python scan_report.py "C:\\path\\to\\maxillary.stl"
    python scan_report.py "C:\\path\\to\\maxillary.stl" --deep

Runs entirely on your machine. Prints statistics and parameter sweeps -- face
counts, edge lengths, curvature distributions, how many teeth the pipeline
finds at each setting. No coordinates, no geometry, nothing from which the
scan could be reconstructed or a person identified. Paste the output back and
I can calibrate the pipeline to your scanner without your patient's scan
leaving your computer.

--deep additionally runs the full precompute sweep (slower, more useful).
"""

import sys
import time
import argparse
import numpy as np

import core_geometry as cg


def read_binary_stl(path):
    data = open(path, "rb").read()
    if len(data) < 84:
        raise ValueError("File too short to be a binary STL.")
    n_tri = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    expected = 84 + n_tri * 50
    if len(data) < expected:
        raise ValueError(
            f"Not a binary STL, or truncated: header says {n_tri} triangles "
            f"({expected} bytes) but the file is {len(data)} bytes. "
            f"ASCII STL? Re-export as binary.")
    rec = np.frombuffer(data[84:expected], dtype=np.dtype([
        ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    uniq, inverse = np.unique(tri, axis=0, return_inverse=True)
    return uniq, np.asarray(inverse).reshape(-1, 3).astype(np.int64), len(data)


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--deep", action="store_true", help="run the full precompute sweep")
    args = ap.parse_args()

    print("SCAN CALIBRATION REPORT")
    print("Statistics only. No coordinates or geometry are printed.")

    # ---------------------------------------------------------- ingest
    t0 = time.time()
    verts, faces, nbytes = read_binary_stl(args.stl)
    t_read = time.time() - t0

    section("1. SIZE AND SCALE")
    extent = verts.max(axis=0) - verts.min(axis=0)
    print(f"  file size              {nbytes/1_048_576:8.1f} MB")
    print(f"  triangles              {len(faces):8,}")
    print(f"  vertices after weld    {len(verts):8,}")
    print(f"  weld ratio             {3*len(faces)/max(len(verts),1):8.2f}  "
          f"(3.0 = no shared vertices at all)")
    print(f"  bounding box extent    {extent[0]:6.1f} x {extent[1]:6.1f} x {extent[2]:6.1f}")
    guess = "millimetres" if 30 < max(extent) < 200 else (
            "metres?" if max(extent) < 1 else "unknown - check units")
    print(f"  implied units          {guess}")
    print(f"  read time              {t_read:8.2f}s")

    # ---------------------------------------------------- conditioning
    section("2. CONDITIONING (what a raw upload needs cleaned)")
    t0 = time.time()
    cverts, cfaces, rep = cg.condition_mesh(verts, faces)
    t_cond = time.time() - t0
    for k, v in rep.items():
        print(f"  {k:<28} {v:,}")
    print(f"  conditioning time            {t_cond:.2f}s")

    # ------------------------------------------------------- topology
    section("3. MESH QUALITY")
    edges = cg.directed_edges(cfaces)
    i, j = edges
    elen = np.linalg.norm(cverts[j] - cverts[i], axis=1)
    for p in (5, 25, 50, 75, 95):
        print(f"  edge length p{p:<2}          {np.percentile(elen, p):8.3f} mm")
    print(f"  mean edge length          {elen.mean():8.3f} mm")
    inc = cg.edge_face_incidence(cfaces)
    print(f"  boundary edges            {sum(1 for f in inc.values() if len(f)==1):8,}")
    print(f"  non-manifold edges        {sum(1 for f in inc.values() if len(f)>2):8,}")
    print(f"  connected components      {len(cg.connected_components(cfaces, loop=[])):8,}")

    # ------------------------------------------------------ curvature
    section("4. CURVATURE FIELD (drives every selection)")
    t0 = time.time()
    conc = cg.boundary_field(cverts, cfaces, edges=edges)
    t_curv = time.time() - t0
    print(f"  computation time          {t_curv:8.2f}s")
    for p in (1, 5, 25, 50, 75, 95, 99):
        print(f"  concavity p{p:<2}            {np.percentile(conc, p):+8.3f}")
    valley = (conc > 0.3).mean()
    ridge = (conc < -0.3).mean()
    print(f"  strongly concave (>0.3)   {valley:8.1%}   <- sulcus + interproximal")
    print(f"  strongly convex  (<-0.3)  {ridge:8.1%}   <- cusps, incisal edges")
    print(f"  barrier at p95 concavity  {np.exp(6*max(np.percentile(conc,95),0)):8.1f}x")
    if valley < 0.02:
        print("  WARNING: very little concave signal. The sulcus may be smoothed")
        print("           out in this scan, which would weaken every barrier.")

    # ------------------------------------------------- arch + teeth
    section("5. ARCH ORIENTATION")
    from tooth_segmentation import arch_geometry
    af = arch_geometry.estimate_arch_frame(cverts, concavity=conc)
    print(f"  explained variance        {np.round(af.explained_variance,3)}")
    print(f"  occlusal axis vs world Z  "
          f"{np.degrees(np.arccos(np.clip(abs(np.dot(af.occlusal_normal,[0,0,1.0])),0,1))):.1f} deg")

    section("6. TOOTH DETECTION SWEEP")
    print(f"  {'separation':>11} {'teeth':>7} {'min faces':>10} {'max faces':>10} {'ratio':>7}")
    print("  " + "-"*50)
    for sep in (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        labels, seeds = cg.auto_segment_arch(cverts, cfaces, conc, af.occlusal_normal,
                                             edges=edges, tolerance=12.0,
                                             min_separation_mm=sep, max_seeds=20)
        ids = np.unique(labels[labels > 0])
        if len(ids) == 0:
            print(f"  {sep:11.1f} {0:7}")
            continue
        sizes = np.array([(labels == k).sum() for k in ids])
        print(f"  {sep:11.1f} {len(ids):7} {sizes.min():10,} {sizes.max():10,} "
              f"{sizes.max()/max(sizes.min(),1):7.1f}x")

    if args.deep:
        section("7. FULL PRECOMPUTE (what the app will do on upload)")
        t0 = time.time()
        pre = cg.precompute_arch(cverts, cfaces, af.occlusal_normal, conc, edges=edges)
        t_pre = time.time() - t0
        print(f"  chosen separation         {pre['chosen_separation_mm']:8.1f} mm")
        print(f"  teeth found               {len(pre['teeth']):8}")
        print(f"  score                     {pre['score']:8.3f}")
        print(f"  precompute time           {t_pre:8.2f}s")
        sizes = [t["n_faces"] for t in pre["teeth"]]
        if sizes:
            print(f"  tooth size min/med/max    {min(sizes):,} / "
                  f"{int(np.median(sizes)):,} / {max(sizes):,}")
            frac = np.array(sizes) / len(cfaces)
            print(f"  tooth fraction of arch    {frac.min():.2%} .. {frac.max():.2%}")
            withres = sum(1 for t in pre["teeth"] if "c_res" in t)
            print(f"  teeth with derived C_res  {withres}/{len(pre['teeth'])}")

    section("SEND ME EVERYTHING ABOVE")
    print("  How many teeth are actually in this arch? ______")
    print("  That single number tells me which separation row is correct,")
    print("  and I can calibrate the defaults to your scanner from there.")


if __name__ == "__main__":
    main()
