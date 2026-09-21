"""Tooth segmentation providers, and the label transfer that keeps them honest.

THE ARCHITECTURE, and the defect it is built around.

    segmentation :  face/vertex -> tooth id        (a provider's job)
    display      :  tooth id    -> colour          (fdi_palette.py)

A provider is handed geometry and returns one label per vertex. What it must
NOT be trusted to return is labels in the ORDER it was given them, and that is
not a hypothetical:

    trimesh.load_mesh(obj, process=False)   order preserved, 0 rows differ
    open3d.io.read_triangle_mesh(obj)       order NOT preserved
    open3d + remove_duplicated_vertices     first-occurrence order
    core_geometry.condition_mesh            np.unique -> LEXICOGRAPHIC order

All four produce 94,848 vertices from this project's real scan. Every count
check passes, every length assertion passes, and three of the four
correspondences are wrong. `ToothGroupNetwork/gen_utils.py` carries the
comment "In some cases, trimesh can change vertex order" directly above the
loader, and the inference pipeline happens to pass `use_tri_mesh=True`, which
is the branch that preserves it. The live path is therefore correct TODAY by
the good fortune of one keyword argument, and nothing anywhere checked it.

Measured consequence on the cached real-scan labels, which were produced from
the raw STL and read against the app's conditioned array:

    per-tooth bounding-box diagonal, median over 12 teeth   50.71 mm
    the same labels transferred by POSITION                 13.97 mm

which was reported as a segmentation-model failure for a model that had in
fact segmented the arch correctly.

SO ORDER IS NEVER ASSUMED. Every provider result is transferred onto the
app's own vertex array by POSITION, and a position that cannot be matched is
a hard error rather than a nearest guess. `transfer_labels_by_position`
reports whether the permutation was the identity, so a loader change that
silently reorders shows up as a number instead of as scattered teeth months
later.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

# Two vertices written to a text file and read back are either the same point
# or a different one; there is no legitimate middle ground. This is a
# round-trip tolerance for decimal formatting, not a snapping distance -
# a genuine mismatch measures millimetres, not microns.
POSITION_MATCH_TOL_MM = 1e-6


class LabelTransferError(RuntimeError):
    """The provider's vertices are not the vertices it was given."""


@dataclass
class ProviderResult:
    """Labels on the CALLER's vertex array, plus how they got there."""
    labels: np.ndarray
    provider: str
    jaw: str
    transfer: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def transfer_labels_by_position(src_verts, src_labels, dst_verts,
                                tol_mm=POSITION_MATCH_TOL_MM):
    """Move labels from the array a provider used onto the array we hold.

    THE MATCH IS EXACT OR IT IS AN ERROR. A nearest-neighbour transfer with a
    generous tolerance would paper over precisely the failure this exists to
    catch: if the provider returned labels for a DIFFERENT mesh, every query
    still finds some nearest vertex and the result looks like a segmentation
    that merely needs review.

    Returns (labels_on_dst, info). `info["identity"]` is False when the
    provider reordered - which is not itself an error, it is the thing that
    used to be invisible.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src_verts, float)
    dst = np.asarray(dst_verts, float)
    lab = np.asarray(src_labels).reshape(-1)

    if len(lab) != len(src):
        raise LabelTransferError(
            f"provider returned {len(lab)} labels for {len(src)} vertices")

    tree = cKDTree(src)
    gap, idx = tree.query(dst, workers=-1)
    worst = float(gap.max()) if len(gap) else 0.0
    unmatched = int((gap > tol_mm).sum())
    if unmatched:
        raise LabelTransferError(
            f"{unmatched} of {len(dst)} vertices have no matching vertex in "
            f"the provider's mesh (worst gap {worst:.6g} mm, tolerance "
            f"{tol_mm:g} mm). The provider did not segment this mesh.")

    # An ambiguous source - two vertices at one position - would make the
    # transfer depend on tie-breaking inside the tree. Report it; it means
    # the provider welded differently from us.
    uniq = len(np.unique(np.round(src / max(tol_mm, 1e-12)).astype(np.int64),
                         axis=0))
    identity = bool(len(src) == len(dst) and np.array_equal(
        idx, np.arange(len(dst))))
    return lab[idx], {
        "identity": identity,
        "reordered_vertices": int((idx != np.arange(len(dst))).sum())
        if len(src) == len(dst) else None,
        "max_position_gap_mm": worst,
        "tolerance_mm": tol_mm,
        "source_vertices": int(len(src)),
        "destination_vertices": int(len(dst)),
        "source_positions_unique": int(uniq),
        "source_had_duplicate_positions": bool(uniq != len(src)),
    }



def first_occurrence_vertex_order(verts, faces):
    """The vertex array a first-occurrence deduplicating loader produces.

    WHICH ARRAY A CACHED LABEL FILE BELONGS TO. A label file produced by
    running the pipeline on a raw STL is keyed to the order that loader
    produced, and an STL has no vertex list at all - it is triangle soup, so
    every reader invents an order when it welds. The two that matter here:

        open3d read + remove_duplicated_vertices   FIRST OCCURRENCE
        stl_io.parse_stl_bytes / condition_mesh    LEXICOGRAPHIC (np.unique)

    Measured on this project's real scan the two agree on the 94,848 points
    and on nothing else: transferring the cached labels from the lexicographic
    array left every tooth scattered over 83 to 435 disconnected pieces, and
    from this one they come back as whole teeth. Verified to produce the same
    array as Open3D's own dedup on that scan.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    soup = v[f].reshape(-1, 3)
    _, first = np.unique(soup, axis=0, return_index=True)
    return soup[np.sort(first)]


class SegmentationProvider(abc.ABC):
    """One tooth-segmentation backend."""

    name = "abstract"
    #: What the provider is known to do, for the audit trail and the UI.
    audit: dict = {}

    @abc.abstractmethod
    def available(self) -> dict:
        """`{"available": bool, "reason": str, ...}` - never raises."""

    @abc.abstractmethod
    def segment(self, verts, faces, jaw) -> ProviderResult:
        """Labels on the CALLER's vertex array. Raises if it cannot."""


class ToothGroupNetworkProvider(SegmentationProvider):
    """The shipped model. Primary, and the only one wired to weights here."""

    name = "toothgroupnetwork"
    audit = {
        "family": "TGNet (ToothGroupNetwork), 3DTeethSeg challenge pipeline",
        "weights_in_repo": True,
        "device": "cpu (torch CPU shims are installed by tgn_bridge)",
        "input": "OBJ written by api_core.run_segmentation, v/f in our order",
        "loader": "gen_utils.read_txt_obj_ls(use_tri_mesh=True) -> "
                  "trimesh.load_mesh(process=False)",
        "order_preserved_measured": "yes, 0 of 94,848 rows differ - but the "
                                    "result is transferred by position "
                                    "regardless, because the loader is not "
                                    "ours and its own docstring warns that "
                                    "trimesh can change vertex order",
        "label_convention": "FDI; the pipeline adds 20 to every non-zero "
                            "label for a lower jaw (predict_utils.predict)",
        "internal_resampling": "farthest-point sample to 24,000 points, "
                               "propagated back to every vertex by KDTree on "
                               "POSITIONS (inference_pipeline_tgn)",
        "measured_runtime": "235.8 s on 94,848 verts / 187,625 faces, CPU",
    }

    def available(self) -> dict:
        try:
            import tgn_bridge
            st = tgn_bridge.status()
            return {"available": bool(st.get("loaded")),
                    "reason": st.get("error") or ("loaded" if st.get("loaded")
                                                  else "not loaded yet"),
                    "warming": bool(st.get("warming")), "status": st}
        except Exception as e:                            # noqa: BLE001
            return {"available": False, "reason": f"{type(e).__name__}: {e}"}

    def segment(self, verts, faces, jaw) -> ProviderResult:
        import api_core
        import jaw_naming

        v = np.asarray(verts, float)
        result_json, used_verts = api_core.run_segmentation(
            v, np.asarray(faces, np.int64), jaw, return_input_vertices=True)
        labels, _ = jaw_naming.extract_labels(result_json, expect_jaw=jaw)
        labels = np.asarray(labels).astype(np.int64).reshape(-1)
        moved, info = transfer_labels_by_position(used_verts, labels, v)
        return ProviderResult(labels=moved, provider=self.name, jaw=jaw,
                              transfer=info,
                              meta={"raw_label_count": int(len(labels))})


class MeshSegNetProvider(SegmentationProvider):
    """CANDIDATE. Declared, auditable, and deliberately not implemented.

    IT RAISES RATHER THAN RETURNING SOMETHING PLAUSIBLE. The precedent is
    CLAUDE.md section 16: a CBCT stub that returned a cone from a function
    named `segment_roots` would be the most dangerous thing in this repo,
    because output from a function of that name is reasonably believed. A
    segmentation stub is the same hazard - labels drive the cut, the cut
    drives C_res, and C_res drives every millimetre the lab prints.

    WHAT IS AUDITED HERE AND WHAT IS NOT. The fields below marked
    NOT_VERIFIED_HERE need the upstream repository and a machine that can
    install it; this environment has neither, so they are left as questions
    rather than answered from memory. Nothing about MeshSegNet has been
    benchmarked against ToothGroupNetwork on this project's data, and until
    it has, no comparison should be quoted.
    """

    name = "meshsegnet"
    audit = {
        "status": "CANDIDATE - NOT INSTALLED, NOT BENCHMARKED",
        "package_present_in_this_environment": False,
        "weights_present_in_this_environment": False,
        "input_primitive": "FACES, not vertices - MeshSegNet classifies "
                           "triangles from a 15-channel per-cell feature "
                           "vector, so a provider must map face labels onto "
                           "vertices before it can feed this app's "
                           "vertex-keyed pipeline",
        "known_preprocessing_requirement": "the published pipeline decimates "
                                           "to a fixed cell budget (~10k) and "
                                           "mean-centres the mesh, which "
                                           "collides with rule 3.1 - scanner "
                                           "coordinates are sacred, so any "
                                           "normalisation must be inverted "
                                           "exactly, not approximately",
        "license": "NOT_VERIFIED_HERE - requires the upstream repository",
        "weights_license": "NOT_VERIFIED_HERE",
        "dependencies": "NOT_VERIFIED_HERE (expected: torch, vedo/vtk, "
                        "scikit-learn; vtk is NOT currently installed)",
        "windows_cpu_support": "NOT_VERIFIED_HERE",
        "inference_time": "NOT_VERIFIED_HERE - no benchmark has been run",
        "axis_convention": "NOT_VERIFIED_HERE",
        "label_convention": "NOT_VERIFIED_HERE - published variants use both "
                            "FDI and a 0-14 sequential class index",
    }

    def available(self) -> dict:
        return {"available": False,
                "reason": "MeshSegNet is a declared candidate that has not "
                          "been installed, audited end to end or benchmarked "
                          "against ToothGroupNetwork on this project's data",
                "audit": self.audit}

    def segment(self, verts, faces, jaw) -> ProviderResult:
        raise NotImplementedError(
            "MeshSegNetProvider is a declared candidate, not an "
            "implementation. It raises deliberately: a segmentation stub that "
            "returned plausible labels would drive a cut, a C_res and a "
            "printed aligner. See MeshSegNetProvider.audit for what is known "
            "and what is NOT_VERIFIED_HERE.")


_REGISTRY = {p.name: p for p in (ToothGroupNetworkProvider(),
                                 MeshSegNetProvider())}
DEFAULT_PROVIDER = ToothGroupNetworkProvider.name


def get(name=None) -> SegmentationProvider:
    key = (name or DEFAULT_PROVIDER).lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown segmentation provider {name!r}; "
                       f"have {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def registry() -> dict:
    """What providers exist and what each one's state is. For /api and docs."""
    return {name: {"name": name, "default": name == DEFAULT_PROVIDER,
                   **p.available(), "audit": p.audit}
            for name, p in _REGISTRY.items()}
