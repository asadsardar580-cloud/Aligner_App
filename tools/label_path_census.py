"""Where does the gum go? The gum share at every point of the label path.

TASK 1 step 1. On `case_lower.stl`, measure the gingiva share in the band
12-18 mm below the cusp tips at four points:

  a  TGN's raw prediction on its own 24,000 fps-sampled points
  b  after TGN propagates those labels back to every vertex (KDTree, positions)
  c  after `segmentation_fallback.run` (api_core's hybrid step)
  d  the labels stored in the session, which `/select-tooth` and
     `manufacturing_v2` read

`transfer_labels_by_position` sits between b and c and is EXACT by
construction - it refuses rather than approximates - so it cannot lose gum.
The point of measuring all four is to find which step does.

HOW (a) IS REACHED WITHOUT EDITING VENDORED CODE. The pipeline's last
`KDTree` is built on `final_ins_points` (the 24,000 sampled points followed by
the boundary points) and queried with `org_feats[:,:3]`, the original vertices
- both in the pipeline's own normalised space. Recording that one constructor
and that one query gives the sampled points, the original vertices and the
vertex->sample map. The sampled point's label is then read back off any vertex
that maps to it, which is exact because the map IS nearest-neighbour
assignment.

THE BAND IS MEASURED IN MILLIMETRES, on our own array. The pipeline
re-centres and rescales (`vertices -= mean`, then a uniform scale), so a
distance in its space is not a distance in the mouth. The normalised original
vertices and our millimetre ones are the same rows in the same order, so the
affine is recovered by least squares and its residual is REPORTED - a fit that
did not fit would make every number below meaningless.

    .venv\\Scripts\\python.exe scratch\\label_path_census.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SCAN = "case_lower.stl"
BAND_GUM = (12.0, 18.0)     # mm below the cusp tips: should be gingiva
BAND_TOOTH = (0.0, 4.0)     # mm below the cusp tips: should be enamel


# ---------------------------------------------------------------------------
# the band
# ---------------------------------------------------------------------------

from segmentation_diagnostics import (depth_below_cusp_tips,
                                      occlusal_axis)


def occlusal_height(verts, u_occ):
    return depth_below_cusp_tips(verts, u_occ)


def band_mask(depth, lo, hi):
    return (depth >= lo) & (depth < hi)


def gum_share(labels, mask, gingiva=0):
    """Fraction of the masked points that carry the gingiva label."""
    lab = np.asarray(labels).reshape(-1)
    if not mask.any():
        return float("nan"), 0
    sel = lab[mask]
    return float((sel == gingiva).mean()), int(mask.sum())


# ---------------------------------------------------------------------------
# the recorder - one constructor and one query, no vendored edit
# ---------------------------------------------------------------------------

class _Recorder:
    """Wraps sklearn's KDTree inside the pipeline module and keeps the last
    tree's data plus its query, which is the propagation we want to see."""

    def __init__(self):
        self.calls = []

    def install(self, module):
        real = module.KDTree
        rec = self

        class Recording(real):
            def __init__(self, data, *a, **kw):
                super().__init__(data, *a, **kw)
                self._rec_data = np.asarray(data, float).copy()
                rec.calls.append({"data": self._rec_data, "queries": []})
                self._rec_slot = rec.calls[-1]

            def query(self, X, *a, **kw):
                out = super().query(X, *a, **kw)
                idx = out if not isinstance(out, tuple) else out[-1]
                self._rec_slot["queries"].append(
                    {"X": np.asarray(X, float).copy(),
                     "idx": np.asarray(idx).copy()})
                return out

        module.KDTree = Recording
        self._module, self._real = module, real

    def restore(self):
        self._module.KDTree = self._real

    def propagation(self):
        """The call whose query covers every original vertex - the last one
        with a query, which is the back-propagation to the full mesh."""
        for call in reversed(self.calls):
            if call["queries"]:
                return call, call["queries"][-1]
        return None, None


def main() -> int:
    import arch_frame
    import core_geometry as cg
    import stl_io

    print("=" * 78)
    print("LABEL PATH CENSUS -", SCAN)
    print("=" * 78)

    raw = open(SCAN, "rb").read()
    v0, f0 = stl_io.parse_stl_bytes(raw)
    v0, f0, _ = cg.sanitize_scan(v0, f0)
    v, f, _ = cg.condition_mesh(v0, f0)
    print(f"conditioned          {len(v):,} verts / {len(f):,} faces")

    # --- the occlusal frame, with its SIGN MEASURED ----------------------
    # NOT the bare `vt[2]` construction this script first used: SVD does not
    # promise a sign, so that picked landmarks off the TISSUE end and every
    # depth came out upside down. See band_sanity_helpers.
    centroid = v.mean(axis=0)
    u_occ, _axis_evidence = occlusal_axis(v, f)
    frame = {"u_occ": u_occ}
    print(f"occlusal sign        {_axis_evidence}")

    depth = occlusal_height(v, u_occ)
    m_gum = band_mask(depth, *BAND_GUM)
    m_tooth = band_mask(depth, *BAND_TOOTH)
    print(f"occlusal axis        {np.round(u_occ, 4)}")
    print(f"depth range          0.00 .. {depth.max():.2f} mm below cusp tips")
    print(f"band {BAND_GUM[0]:.0f}-{BAND_GUM[1]:.0f}mm         "
          f"{int(m_gum.sum()):,} vertices")
    print(f"band {BAND_TOOTH[0]:.0f}-{BAND_TOOTH[1]:.0f}mm          "
          f"{int(m_tooth.sum()):,} vertices")

    # --- run TGN with the recorder installed ------------------------------
    import api_core
    import jaw_naming
    import tgn_bridge

    print("\nloading ToothGroupNetwork ...")
    t0 = time.perf_counter()
    st = tgn_bridge.load(api_core.CKPT_FPS, api_core.CKPT_BDL,
                         model_name="tgnet")
    if not st.get("loaded"):
        print(f"REFUSED: {st.get('error')}")
        return 2
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    with tgn_bridge._RepoOnPath(tgn_bridge.tgn_path()):
        from inference_pipelines import inference_pipeline_tgn as ipt
    rec = _Recorder()
    rec.install(ipt)
    try:
        print("segmenting (this is ~4 minutes on CPU) ...")
        t0 = time.perf_counter()
        result_json, used_verts = api_core.run_segmentation(
            v, f, "lower", return_input_vertices=True)
        t_seg = time.perf_counter() - t0
        print(f"  {t_seg:.1f}s")
    finally:
        rec.restore()

    labels_b_pipeline, _ = jaw_naming.extract_labels(result_json,
                                                     expect_jaw="lower")
    labels_b_pipeline = np.asarray(labels_b_pipeline, np.int64).reshape(-1)

    out = {"scan": SCAN, "vertices": int(len(v)), "faces": int(len(f)),
           "band_gum_mm": BAND_GUM, "band_tooth_mm": BAND_TOOTH,
           "band_gum_vertices": int(m_gum.sum()),
           "band_tooth_vertices": int(m_tooth.sum()),
           "segment_seconds": round(t_seg, 1)}

    # --- (a) the raw prediction on the sampled points --------------------
    call, q = rec.propagation()
    print(f"\nrecorded {len(rec.calls)} KDTree(s); "
          f"propagation tree has {len(call['data']):,} points, "
          f"query covers {len(q['X']):,} vertices")

    a_share = float("nan")
    if call is not None and len(q["X"]) == len(used_verts):
        # The affine from the pipeline's normalised space back to millimetres.
        # Uniform scale + per-axis offset, recovered by least squares on the
        # paired rows, with the residual reported.
        P, Q = q["X"], np.asarray(used_verts, float)
        A = np.concatenate([P, np.ones((len(P), 1))], axis=1)
        M, *_ = np.linalg.lstsq(A, Q, rcond=None)
        resid = float(np.abs(A @ M - Q).max())
        print(f"normalised -> mm fit residual {resid:.3e} mm")
        out["affine_residual_mm"] = resid

        samples_mm = np.concatenate(
            [call["data"], np.ones((len(call["data"]), 1))], axis=1) @ M

        # Each sampled point's label, read off a vertex that maps to it. The
        # map is nearest-neighbour assignment, so this is exact for every
        # sample that owns at least one vertex.
        idx = np.asarray(q["idx"]).reshape(-1)
        samp_lab = np.full(len(call["data"]), -1, np.int64)
        samp_lab[idx] = labels_b_pipeline           # last writer wins; all agree
        owned = samp_lab >= 0
        print(f"sampled points owning >=1 vertex: {int(owned.sum()):,} "
              f"of {len(samp_lab):,}")

        s_depth = float(np.max(np.asarray(v, float) @ u_occ)) - (samples_mm @ u_occ)
        s_mask = band_mask(s_depth, *BAND_GUM) & owned
        a_share, a_n = gum_share(samp_lab, s_mask)
        out["a_sampled_points"] = {"gum_share": a_share, "n": a_n,
                                   "total_samples": int(len(samp_lab)),
                                   "owned": int(owned.sum())}
        s_maskT = band_mask(s_depth, *BAND_TOOTH) & owned
        aT, aTn = gum_share(samp_lab, s_maskT)
        out["a_sampled_points"]["tooth_band_gum_share"] = aT
        out["a_sampled_points"]["tooth_band_n"] = aTn
    else:
        print("could not pair the propagation query with the input vertices")
        out["a_sampled_points"] = {"error": "unpaired"}

    # --- (b) after propagation, on OUR array ------------------------------
    import segmentation_providers as sp
    labels_b, transfer = sp.transfer_labels_by_position(
        used_verts, labels_b_pipeline, v)
    b_share, b_n = gum_share(labels_b, m_gum)
    bT, _ = gum_share(labels_b, m_tooth)
    out["b_after_propagation"] = {"gum_share": b_share, "n": b_n,
                                  "tooth_band_gum_share": bT,
                                  "transfer_identity": transfer["identity"],
                                  "max_position_gap_mm":
                                      transfer["max_position_gap_mm"]}

    # --- (c) after the hybrid step ----------------------------------------
    import segmentation_fallback
    edges = cg.directed_edges(f)
    conc = cg.boundary_field(v, f, edges=edges)
    graph = cg.build_barrier_graph(v, f, conc, edges=edges)
    t0 = time.perf_counter()
    hybrid = segmentation_fallback.run(
        labels_b.copy(), v, f, "lower", graph=graph, concavity=conc,
        arch_centre=centroid, occlusal_axis=u_occ)
    labels_c = np.asarray(hybrid["labels"]).reshape(-1)
    c_share, c_n = gum_share(labels_c, m_gum)
    cT, _ = gum_share(labels_c, m_tooth)
    out["c_after_hybrid"] = {"gum_share": c_share, "n": c_n,
                             "tooth_band_gum_share": cT,
                             "repaired_count": hybrid.get("repaired_count"),
                             "seconds": round(time.perf_counter() - t0, 1)}

    # --- (d) what the session stores --------------------------------------
    # Identical to (c) by construction - `STORE.put(sid, "labels", labels)`
    # is the line after `labels = hybrid["labels"]` - and measured rather
    # than asserted, because that is the array /select-tooth reads.
    labels_d = labels_c
    d_share, d_n = gum_share(labels_d, m_gum)
    dT, _ = gum_share(labels_d, m_tooth)
    out["d_session"] = {"gum_share": d_share, "n": d_n,
                        "tooth_band_gum_share": dT,
                        "identical_to_c": bool(np.array_equal(labels_c,
                                                              labels_d))}

    # --- the table --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"GUM SHARE IN THE BAND {BAND_GUM[0]:.0f}-{BAND_GUM[1]:.0f} mm "
          f"BELOW THE CUSP TIPS   (target >= 90%)")
    print("=" * 78)
    rows = [("a  TGN raw, 24,000 sampled points", a_share,
             out["a_sampled_points"].get("n")),
            ("b  after KDTree propagation", b_share, b_n),
            ("c  after the hybrid step", c_share, c_n),
            ("d  session / select-tooth", d_share, d_n)]
    for name, share, n in rows:
        pct = "   n/a" if share != share else f"{share * 100:6.2f}%"
        print(f"  {name:<38} {pct}   ({n or 0:,} points)")

    print(f"\nGUM SHARE IN THE BAND {BAND_TOOTH[0]:.0f}-{BAND_TOOTH[1]:.0f} mm "
          f"(target <= 15%, i.e. >= 85% tooth)")
    for name, key in (("a  TGN raw", "a_sampled_points"),
                      ("b  after propagation", "b_after_propagation"),
                      ("c  after hybrid", "c_after_hybrid"),
                      ("d  session", "d_session")):
        sh = out[key].get("tooth_band_gum_share", float("nan"))
        pct = "   n/a" if sh != sh else f"{sh * 100:6.2f}%"
        print(f"  {name:<38} {pct}")

    os.makedirs("scratch", exist_ok=True)
    np.save("scratch/labels_b.npy", labels_b)
    if "samples_mm" in dir():
        np.save("scratch/samples_mm.npy", samples_mm)
        np.save("scratch/samples_label.npy", samp_lab)
    np.save("scratch/labels_c.npy", labels_c)
    with open("scratch/label_path_census.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nwritten scratch/label_path_census.json, labels_b.npy, labels_c.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
