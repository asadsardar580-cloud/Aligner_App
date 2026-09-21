"""crosstooth_bridge.py — the single point of contact between this backend and
the vendored CrossTooth (CVPR 2025) checkout.

It follows the four design rules `tgn_bridge.py` already states, for the same
reasons, and they are not repeated here:

    1. NOTHING IMPORTS AT MODULE LOAD.
    2. THE VENDORED REPO IS NEVER MODIFIED.
    3. sys.path / sys.modules ORDERING IS EXPLICIT.
    4. FAILURES ARE STRUCTURED, NOT PRINTED.

WHAT THE UPSTREAM REPOSITORY CANNOT DO ON THIS MACHINE, AND WHAT IS DONE
ABOUT EACH. Every one of these was read out of the vendored source, not
assumed:

  * `models/PTv1/point_transformer_seg.py` opens with
        from models.PointTransformer.libs.pointops.functions import pointops
    and THAT PATH DOES NOT EXIST IN THE SHIPPED REPOSITORY — `CrossTooth/
    models/` contains only `PTv1/`. The module cannot import as delivered on
    any machine, CUDA or not. `pointops` is the Point Transformer CUDA
    extension, which needs a C++/CUDA toolchain Windows does not have here.
    This module registers that exact dotted name in `sys.modules`, pointing at
    a thin wrapper over this project's existing `pointops_cpu.py` shim, for
    the duration of the import only, and restores `sys.modules` afterwards.

  * THE SHIM'S CONTRACT IS NOT CROSSTOOTH'S CONTRACT, and calling it directly
    would fail at the first layer. `pointops_cpu.queryandgroup` returns the
    grouped tensor (what ToothGroupNetwork expects); CrossTooth writes
    `x_k, idx = pointops.queryandgroup(...)` and reuses that `idx` for the
    value projection, so it expects a PAIR. `_CrossToothPointops` adapts the
    return value and nothing else — `pointops_cpu.py` is left untouched
    because ToothGroupNetwork depends on its single-value contract.

  * `TransitionDown.forward` builds its offset vector with
        n_o = torch.cuda.IntTensor(n_o)
    on a line that is otherwise pure bookkeeping — a Python list of ints. On
    a CPU build `torch.cuda.IntTensor` exists as an attribute and raises
    `TypeError: type torch.cuda.IntTensor not available`. Four of the five
    encoder stages have stride 4, so this fires on every forward pass. It is
    patched to a CPU int32 constructor for the duration of one inference call
    and restored, rather than editing the vendored file.

  * `dataset/data.py` loads meshes with `vedo`, which is not installed and is
    not needed: this app already holds the mesh as vertices and faces. The
    FEATURE PREPARATION is replicated here from that file line by line —
    see `CrossToothAdapter.prepare`, which documents each step and the one
    place it deliberately differs.

WHAT THIS MODEL CLASSIFIES. FACES, not vertices. The 6 input channels are the
face centroid (3) and the face normal (3), so a prediction is one class per
triangle, and it has to be carried onto this app's vertex array before
anything else in this codebase can use it. That is done by the researchers'
own method (`prepare_data/upsample_points.py`: `KNeighborsClassifier(
n_neighbors=3)` fitted on the predicted cell centroids and evaluated at the
original mesh's coordinates), so the mapping is theirs rather than invented
here.

WHAT IS NOT VERIFIED HERE, and must not be quoted as if it were:

  * THE QUADRANT NAMING. `CrossTooth/utils.py` gives classes 1-8 the "L"
    quadrant and 9-16 the "R" quadrant of the jaw. Which physical side of the
    patient that is depends on the orientation convention of the training
    data, and this repository has no annotated ground truth (see CLAUDE.md
    s.17 — scoring a model against its own prediction returns mIoU 1.0 and
    means nothing). `quadrant_coherence` MEASURES whether the two groups are
    separated by a plane and whether the class index runs anterior to
    posterior, which is checkable here; it cannot and does not confirm the
    NAMES. `labels_are_fdi_verified` is False, always, until a benchmark
    against an independent annotation says otherwise.

  * ACCURACY. No benchmark of CrossTooth against ToothGroupNetwork on this
    project's data is quoted anywhere in this module. `benchmark_providers.py`
    runs the comparison and reports agreement; agreement between two
    unvalidated models is not accuracy either, and it says so.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import traceback
import types

import numpy as np

# --------------------------------------------------------------------------
# configuration — every number here was read out of the shipped checkpoint or
# the vendored source, not chosen.
# --------------------------------------------------------------------------

CROSSTOOTH_DIRNAME = "CrossTooth"
CHECKPOINT_REL = os.path.join("models", "PTv1", "point_best_model.pth")
MODEL_SOURCE_REL = os.path.join("models", "PTv1", "point_transformer_seg.py")

#: `enc1.0.linear.weight` is (32, 6) in `point_best_model.pth`, which pins
#: in_channels at 6 and therefore `enable_pic_feat=False` — the multi-view
#: picture-feature branch would add 4 more channels and an `extra_pic_feat`
#: embedding, and the checkpoint carries no such key.
IN_CHANNELS = 6
#: `cls.3.weight` is (19, 32). predict.py builds the model with
#: `num_classes=17 + 2` and maps classes 17 and 18 back to background.
NUM_CLASSES = 19
#: `--num_points` / `--sample_points` default in predict.py. The checkpoint was
#: trained at this size; changing it changes the neighbourhood scale that the
#: fixed nsample of 8/16 represents, so it is a parameter with a warning
#: attached rather than a knob.
NUM_POINTS = 16000
#: predict.py: `pred_mask[pred_mask == 17] = 0; pred_mask[pred_mask == 18] = 0`
BACKGROUND_CLASSES = (0, 17, 18)
#: prepare_data/upsample_points.py: `KNeighborsClassifier(n_neighbors=3)`
UPSAMPLE_NEIGHBOURS = 3
#: predict.py sets `torch.manual_seed(1)`. The sampling below is seeded from
#: the same value so one scan gives one answer.
DEFAULT_SEED = 1


def _class_to_fdi_table():
    """Class index -> FDI, transcribed from `CrossTooth/utils.py`.

    Built arithmetically rather than typed out, because a 32-entry table of
    two-digit numbers is exactly the kind of thing that acquires a transposed
    digit and still looks right. The source of truth is `utils.FDI2color`:

        label2color_lower  1..8  -> LL1..LL8   and FDI2color puts LLn at 30+n
        label2color_lower  9..16 -> LR1..LR8   and FDI2color puts LRn at 40+n
        label2color_upper  1..8  -> UL1..UL8   and FDI2color puts ULn at 20+n
        label2color_upper  9..16 -> UR1..UR8   and FDI2color puts URn at 10+n
    """
    lower = {k: 30 + k for k in range(1, 9)}
    lower.update({k: 40 + (k - 8) for k in range(9, 17)})
    upper = {k: 20 + k for k in range(1, 9)}
    upper.update({k: 10 + (k - 8) for k in range(9, 17)})
    return {"lower": lower, "upper": upper}


CLASS_TO_FDI = _class_to_fdi_table()

_LOCK = threading.Lock()
_STATE = {
    "loaded": False,
    "error": None,
    "traceback": None,
    "adapter": None,
    "load_seconds": None,
}


def crosstooth_path(base=None):
    """Absolute path to the vendored checkout."""
    base = base or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, CROSSTOOTH_DIRNAME)


# --------------------------------------------------------------------------
# the Windows / CPU bypass
# --------------------------------------------------------------------------

class _CrossToothPointops:
    """CrossTooth's pointops contract, served by this project's CPU shim.

    ONLY `queryandgroup` DIFFERS, and it differs in its RETURN VALUE, not in
    its arithmetic. ToothGroupNetwork's pointops returns the grouped tensor;
    CrossTooth's returns `(grouped, idx)` and reuses the index so the key and
    value projections are grouped by the same neighbourhood — recomputing the
    neighbourhood for the value would be a different model, not a slower one.

    `furthestsampling`, `knnquery`, `interpolation` and `grouping` are passed
    through unchanged: CrossTooth calls them with the same signatures and the
    same meanings, checked call site by call site against
    `models/PTv1/point_transformer_seg.py`.
    """

    def __init__(self, impl):
        self._impl = impl
        self.furthestsampling = impl.furthestsampling
        self.knnquery = impl.knnquery
        self.interpolation = impl.interpolation
        self.grouping = impl.grouping

    def queryandgroup(self, nsample=None, xyz=None, new_xyz=None, feat=None,
                      idx=None, offset=None, new_offset=None, use_xyz=True):
        if idx is None:
            idx, _ = self._impl.knnquery(nsample, xyz, new_xyz, offset,
                                         new_offset)
        grouped = self._impl.queryandgroup(nsample, xyz, new_xyz, feat, idx,
                                           offset, new_offset, use_xyz)
        return grouped, idx


#: The dotted name `models/PTv1/point_transformer_seg.py` imports. Parents are
#: listed because a future CPython could stop short-circuiting on the full
#: name in `_find_and_load`; today only the leaf is consulted.
_POINTOPS_CHAIN = (
    "models",
    "models.PointTransformer",
    "models.PointTransformer.libs",
    "models.PointTransformer.libs.pointops",
    "models.PointTransformer.libs.pointops.functions",
)


@contextlib.contextmanager
def _pointops_installed():
    """Register the CPU pointops under CrossTooth's import path, then undo it.

    IT IS SCOPED TO THE IMPORT DELIBERATELY. `models` is a top-level name
    ToothGroupNetwork also uses (tgn_bridge design rule 3), and this backend
    warms TGN on a background thread — leaving a synthetic `models` package
    installed for the life of the process is how one model silently captures
    the other's imports. A name that is ALREADY in `sys.modules` is left
    exactly as it is and never restored, because it is not ours.
    """
    import pointops_cpu

    shim = _CrossToothPointops(pointops_cpu)
    added = []
    try:
        for name in _POINTOPS_CHAIN:
            if name in sys.modules:
                continue
            mod = types.ModuleType(name)
            mod.__path__ = []          # a package, so submodules are legal
            sys.modules[name] = mod
            added.append(name)
            if "." in name:
                parent = sys.modules.get(name.rsplit(".", 1)[0])
                if parent is not None:
                    setattr(parent, name.rsplit(".", 1)[1], mod)
        leaf = sys.modules[_POINTOPS_CHAIN[-1]]
        had_attr = hasattr(leaf, "pointops")
        previous = getattr(leaf, "pointops", None)
        leaf.pointops = shim
        try:
            yield shim
        finally:
            if had_attr:
                leaf.pointops = previous
            elif hasattr(leaf, "pointops"):
                del leaf.pointops
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)


@contextlib.contextmanager
def _cuda_int_tensor_on(device):
    """Let `TransitionDown.forward`'s one hard CUDA call run on the CPU.

        n_o = torch.cuda.IntTensor(n_o)        # point_transformer_seg.py

    `n_o` is a Python list of ints — a prefix-sum of batch offsets. The line
    is pure bookkeeping and the CUDA constructor is incidental to it, so the
    substitution changes device placement and nothing else. Stride is 4 on
    encoder stages 2-5, so this fires on every forward pass and there is no
    way round it short of editing the vendored file.

    The patch is process-global while it is in force, which is why it is held
    only across one inference call and only under `_LOCK`. Nothing else in
    this repository constructs a CUDA tensor: ToothGroupNetwork runs on the
    same `pointops_cpu` shim and `cpu_compat.py` strips its device calls.
    """
    import torch

    original = torch.cuda.IntTensor

    def _int_tensor(data, *args, **kwargs):
        return torch.as_tensor(data, dtype=torch.int32, device=device)

    torch.cuda.IntTensor = _int_tensor
    try:
        yield
    finally:
        torch.cuda.IntTensor = original


def _load_model_module(root):
    """Import `point_transformer_seg.py` BY PATH, under a private name.

    Not `import models.PTv1.point_transformer_seg`: that would need
    `CrossTooth` on `sys.path` and would plant a second `models` package in a
    process that already has ToothGroupNetwork's. The file has no relative
    imports, so loading it directly is exact.
    """
    import importlib.util

    path = os.path.join(root, MODEL_SOURCE_REL)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CrossTooth model source not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "crosstooth_ptv1_point_transformer_seg", path)
    mod = importlib.util.module_from_spec(spec)
    with _pointops_installed():
        spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# geometry helpers — the replicated data preparation
# --------------------------------------------------------------------------

def face_centroids_and_normals(verts, faces):
    """The 6 channels CrossTooth is fed, from a vertex/face mesh.

    `dataset/data.py` takes these from vedo (`mesh.points()` averaged over
    `mesh.cells()`, and `mesh.normals(cells=True)`). vedo returns UNIT cell
    normals, so the cross product is normalised here to match; a degenerate
    triangle has no normal and gets zeros, which is what the zero-padding rows
    in the original loader carry too.
    """
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    tri = v[f]                                            # (F, 3, 3)
    centroids = tri.mean(axis=1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    good = ln > 0
    normals = np.zeros_like(n)
    normals[good] = n[good] / ln[good, None]
    return centroids, normals, int((~good).sum())


def spatially_uniform_subset(points, k, seed=DEFAULT_SEED):
    """Pick `k` of `points` with even spatial coverage. Deterministic.

    WHY NOT THE SHIPPED SELECTION. `dataset/data.py` does

        permute = np.random.permutation(self.args.num_points)
        pointcloud = pointcloud[permute]

    with `num_points = 16000`. On a mesh with MORE faces than that — this
    project's real scan has 187,625 — the index array only spans 0..15999, so
    the model is shown the first 16,000 triangles IN FILE ORDER, shuffled.
    That is a contiguous patch of one scan region, not an arch. The
    researchers avoid it by decimating offline first, with a curvature-aware
    `selective_downsample.exe` (`prepare_data/selective_downsample.py`) that
    is not in this repository and is a Windows binary we do not have.

    SO THIS IS NOT THE RESEARCHERS' DECIMATION AND THE DIFFERENCE IS REAL:
    theirs puts more cells near tooth boundaries, which is where a boundary-
    preserving model most wants them. A voxel grid is uniform instead. It is
    chosen over a random subset because random sampling leaves clumps and
    bald patches at 8.5% coverage, and over farthest-point sampling because
    FPS here is a Python loop over the shim — 16,000 passes across 187,625
    points, which is minutes of pure overhead for a selection.

    The voxel size is bisected to land on `k` occupied voxels; within a voxel
    the cell nearest the voxel centre wins, so the result does not depend on
    input order.
    """
    p = np.asarray(points, float)
    n = len(p)
    if k >= n:
        return np.arange(n)

    lo = p.min(axis=0)
    span = float(np.max(p.max(axis=0) - lo))
    if not np.isfinite(span) or span <= 0:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=k, replace=False))

    def occupied(size):
        """(voxel id per point, number of occupied voxels) at this size.

        ONE int64 key per voxel rather than `np.unique(keys, axis=0)`. The
        row-wise form lexsorts three columns and cost 10.5 s of a 24 s run on
        the real scan. The strides are the per-axis voxel counts, so the
        encoding is exact and cannot collide.
        """
        keys = np.floor((p - lo) / size).astype(np.int64)
        keys -= keys.min(axis=0)
        dims = keys.max(axis=0) + 1
        flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
        uniq, inv = np.unique(flat, return_inverse=True)
        return inv, len(uniq)

    # Bisect on voxel size. Occupancy is monotone in it, so the bracket closes
    # on the target; the early exit is a half-percent band around k.
    small, large = span / (2.0 * n ** (1.0 / 3.0) + 1.0), span
    for _ in range(40):
        mid = 0.5 * (small + large)
        _, count = occupied(mid)
        if count > k:
            small = mid
        else:
            large = mid
        if abs(count - k) <= max(1, k // 200):
            break
    size = 0.5 * (small + large)
    inv, _ = occupied(size)

    # Each point's distance to the centre of the voxel it falls in.
    centre = (np.floor((p - lo) / size) + 0.5) * size + lo
    d2 = ((p - centre) ** 2).sum(axis=1)
    # One representative per voxel: the closest to its centre, ties by index.
    order = np.lexsort((np.arange(n), d2, inv))
    first = np.ones(len(order), bool)
    first[1:] = inv[order][1:] != inv[order][:-1]
    chosen = order[first]

    rng = np.random.default_rng(seed)
    if len(chosen) > k:
        chosen = chosen[np.sort(rng.choice(len(chosen), size=k,
                                           replace=False))]
    elif len(chosen) < k:
        rest = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
        top_up = rng.choice(rest, size=min(k - len(chosen), len(rest)),
                            replace=False)
        chosen = np.concatenate([chosen, top_up])
    return np.sort(chosen)


def quadrant_coherence(cell_centroids, cell_classes):
    """Is the class index spatially coherent? MEASURED, not assumed.

    Two things are checkable without ground truth, and they are the two that
    a wrong quadrant convention would break:

      * classes 1-8 and 9-16 should sit on OPPOSITE SIDES of the arch. The
        separation is scored by projecting both groups' centroids onto the
        line joining the group means and reporting the overlap.
      * within a quadrant the class index runs central incisor (1) to third
        molar (8), so the distance from the ARCH centre to a tooth's centroid
        should increase with the index. Spearman-style rank agreement.

    WHAT THIS CANNOT DO IS CONFIRM THE NAMES. A perfectly coherent split still
    leaves open which group is the patient's left, and that is a property of
    the training data's orientation convention, not of this arch. The verdict
    is reported as `separated` / `ordered`, never as "FDI verified".
    """
    c = np.asarray(cell_centroids, float)
    k = np.asarray(cell_classes).reshape(-1)
    out = {"measurable": False, "reason": "", "per_class_centroid": {}}

    tooth = (k >= 1) & (k <= 16)
    if tooth.sum() < 8:
        out["reason"] = f"only {int(tooth.sum())} cells carry a tooth class"
        return out

    for cls in range(1, 17):
        m = k == cls
        if m.sum():
            out["per_class_centroid"][int(cls)] = \
                [float(x) for x in c[m].mean(axis=0)]

    left = np.array([out["per_class_centroid"][i]
                     for i in range(1, 9) if i in out["per_class_centroid"]])
    right = np.array([out["per_class_centroid"][i]
                      for i in range(9, 17) if i in out["per_class_centroid"]])
    if len(left) < 2 or len(right) < 2:
        out["reason"] = (f"need both groups populated; have {len(left)} of "
                         f"classes 1-8 and {len(right)} of 9-16")
        return out

    axis = right.mean(axis=0) - left.mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        out["reason"] = "the two class groups share a centroid"
        return out
    axis = axis / norm
    pl, pr = left @ axis, right @ axis
    out.update({
        "measurable": True,
        "group_separation_mm": float(pr.mean() - pl.mean()),
        "group_overlap_mm": float(max(0.0, pl.max() - pr.min())),
        "separated": bool(pl.max() < pr.min()),
    })

    # THE REFERENCE POINT IS THE QUADRANT'S OWN FIRST TOOTH, NOT THE ARCH
    # CENTRE, and the difference is not cosmetic. An arch is a horseshoe:
    # incisors and molars both sit on its PERIPHERY, so distance from the
    # middle is not monotone in the tooth index. Distance from the central
    # incisor runs along the curve and is. Measured on the real scan, the
    # same labels score 0.619 / 0.929 from the centre and 1.000 / 1.000 from
    # the quadrant's own incisor - the weaker reference was about to make a
    # correct arch look disordered.
    ranks = []
    for lo, hi in ((1, 9), (9, 17)):
        idx = [i for i in range(lo, hi) if i in out["per_class_centroid"]]
        if len(idx) < 3:
            continue
        ref = np.asarray(out["per_class_centroid"][idx[0]])
        d = [float(np.linalg.norm(
            np.asarray(out["per_class_centroid"][i]) - ref)) for i in idx]
        order = np.argsort(np.argsort(d))
        expect = np.arange(len(idx))
        ranks.append(float(np.corrcoef(order, expect)[0, 1]))
    out["incisor_to_molar_rank_correlation"] = \
        float(np.mean(ranks)) if ranks else None
    out["ordered"] = bool(ranks and np.mean(ranks) > 0.6)
    # Stated every time, so it can never be dropped on the way to a UI.
    out["quadrant_names_verified"] = False
    out["quadrant_names_note"] = (
        "Coherence says the two class groups are distinct and ordered. It "
        "does NOT say which group is the patient's left; that is a training-"
        "data convention and this repository has no annotated ground truth.")
    return out


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------

class CrossToothAdapter:
    """CrossTooth (CVPR 2025) as this application's second segmentation model.

    Construction is cheap and cannot fail. `load()` imports torch, builds
    `PointTransformerSeg38` and reads `point_best_model.pth`. `segment()`
    takes the vertices and faces this app already holds and returns one FDI
    label per vertex, plus everything measured on the way.
    """

    def __init__(self, root=None, num_points=NUM_POINTS, seed=DEFAULT_SEED,
                 device="cpu"):
        self.root = root or crosstooth_path()
        self.num_points = int(num_points)
        self.seed = int(seed)
        self.device = device
        self.model = None
        self.model_class = None
        self.load_seconds = None

    # -- availability ------------------------------------------------------

    def checkpoint_path(self):
        return os.path.join(self.root, CHECKPOINT_REL)

    def diagnose(self):
        """State of the checkout, WITHOUT importing or loading anything."""
        ckpt = self.checkpoint_path()
        src = os.path.join(self.root, MODEL_SOURCE_REL)
        out = {
            "root": self.root,
            "exists": os.path.isdir(self.root),
            "model_source": src,
            "model_source_present": os.path.isfile(src),
            "checkpoint": ckpt,
            "checkpoint_present": os.path.isfile(ckpt),
            "checkpoint_bytes": (os.path.getsize(ckpt)
                                 if os.path.isfile(ckpt) else 0),
            "pointops_compiled_extension_present": os.path.isdir(
                os.path.join(self.root, "models", "PointTransformer")),
            "dependencies": {},
        }
        for name in ("torch", "einops", "sklearn", "vedo"):
            try:
                __import__(name)
                out["dependencies"][name] = "present"
            except Exception as e:                        # noqa: BLE001
                out["dependencies"][name] = f"MISSING ({type(e).__name__})"
        # vedo is listed because the upstream loader needs it; this adapter
        # does not, and saying so stops it being installed for no reason.
        out["vedo_required"] = False
        out["vedo_note"] = ("only `dataset/data.py` uses vedo; the feature "
                            "preparation is replicated here from vertices "
                            "and faces this app already holds")
        return out

    # -- loading -----------------------------------------------------------

    def load(self):
        """Build the network and read the weights. Idempotent."""
        if self.model is not None:
            return self
        t0 = time.time()
        import torch

        ckpt = self.checkpoint_path()
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"CrossTooth checkpoint not found: {ckpt}")

        mod = _load_model_module(self.root)
        self.model_class = mod.PointTransformerSeg38
        model = self.model_class(
            in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
            pretrain=False, add_cbl=False, enable_pic_feat=False)

        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        # STRICT. A silently partial load is a model that runs and predicts
        # noise, which is the single worst outcome available here.
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        self.model = model
        self.load_seconds = round(time.time() - t0, 2)
        return self

    # -- the replicated data preparation -----------------------------------

    def prepare(self, verts, faces):
        """Build the (1, 6, N) input tensor exactly as `ToothData` does.

        `CrossTooth/dataset/data.py`, `__getitem__`, in order:

            1. cell_normals = mesh.normals(cells=True)         per-FACE normals
            2. cell_coords  = mean of the face's three vertices
            3. pointcloud   = concat(cell_coords, cell_normals)      (F, 6)
            4. if F < num_points: zero-pad to num_points
            5. permute rows
            6. PointcloudToTensor
            7. PointcloudNormalize(radius=1): XYZ only — subtract the mean,
               divide by the largest radius. Normals are NOT touched.
            8. PointcloudSample(total, sample): identity when total == sample,
               which is the shipped default (16000 / 16000).

        THE ORDER OF 4, 5 AND 7 IS LOAD-BEARING AND IS PRESERVED. The padding
        rows are zeros and they go in BEFORE normalisation, so on a sparse
        mesh they pull the centroid toward the origin — that is part of the
        input distribution the checkpoint was trained on, and "tidying" it by
        normalising first would feed the model something it has not seen.

        STEP 5 IS THE ONE DELIBERATE DEPARTURE, and only when the mesh is
        larger than `num_points`; see `spatially_uniform_subset`.

        RULE 3.1 IS NOT AT RISK. The normalisation is an input transform on a
        COPY; no scan coordinate is written back, and the labels come home by
        index into the original face array. The centroid and radius are
        returned so the transform is inspectable rather than implicit.
        """
        import torch

        centroids, normals, degenerate = face_centroids_and_normals(verts,
                                                                    faces)
        n_faces = len(centroids)
        if n_faces == 0:
            raise ValueError("the mesh has no faces")

        cloud = np.concatenate([centroids, normals], axis=1)   # (F, 6)

        if n_faces > self.num_points:
            keep = spatially_uniform_subset(centroids, self.num_points,
                                            self.seed)
            selection = "voxel-uniform subset"
        else:
            keep = np.arange(n_faces)
            selection = "all faces"

        rows = cloud[keep]
        source = keep.astype(np.int64)
        padded = 0
        if len(rows) < self.num_points:
            padded = self.num_points - len(rows)
            rows = np.concatenate(
                [rows, np.zeros((padded, rows.shape[1]))], axis=0)
            # -1 marks a padding row so its prediction can be dropped rather
            # than landing on face 0.
            source = np.concatenate([source, np.full(padded, -1, np.int64)])

        rng = np.random.default_rng(self.seed)
        permute = rng.permutation(len(rows))
        rows = rows[permute]
        source = source[permute]

        xyz = rows[:, 0:3]
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
        radius = float(np.max(np.sqrt((xyz ** 2).sum(axis=1))))
        if radius <= 0:
            raise ValueError("every sampled face centroid is the same point")
        rows = rows.copy()
        rows[:, 0:3] = xyz / radius

        tensor = torch.from_numpy(np.ascontiguousarray(rows)).to(torch.float)
        tensor = tensor.unsqueeze(0).permute(0, 2, 1).contiguous()
        return {
            "tensor": tensor,
            "source_face": source,
            "sampled_centroids": centroids[keep],
            "face_count": int(n_faces),
            "sampled": int(len(keep)),
            "padded_rows": int(padded),
            "degenerate_faces": int(degenerate),
            "selection": selection,
            "normalize_centroid": [float(x) for x in centroid],
            "normalize_radius_mm": radius,
        }

    # -- inference ---------------------------------------------------------

    def predict_cell_classes(self, tensor):
        """Raw class index per input row. `(1, 6, N)` in, `(N,)` out."""
        import torch

        if self.model is None:
            self.load()
        with torch.no_grad(), _cuda_int_tensor_on(self.device):
            seg, _edge = self.model(tensor.to(self.device))
        prob = torch.nn.functional.softmax(seg, dim=1)
        conf, cls = torch.max(prob, dim=1)
        return (cls.squeeze(0).cpu().numpy().astype(np.int64),
                conf.squeeze(0).cpu().numpy().astype(np.float64))

    def segment(self, verts, faces, jaw):
        """One FDI label per vertex, on the array that was passed in.

        The face-class -> vertex-label step is the researchers' own
        (`prepare_data/upsample_points.py`): a 3-neighbour KNN fitted on the
        predicted cell centroids in ORIGINAL millimetre coordinates and
        evaluated at the mesh's vertices. It is used rather than a
        majority-of-incident-faces vote so that the mapping is theirs, and
        because it is defined for a vertex whose every incident face was
        dropped by the subset.
        """
        jaw = (jaw or "lower").lower()
        if jaw not in CLASS_TO_FDI:
            raise ValueError(f"jaw must be 'upper' or 'lower', not {jaw!r}")

        v = np.asarray(verts, float)
        f = np.asarray(faces, np.int64)
        t0 = time.time()
        prep = self.prepare(v, f)
        t_prep = time.time()
        cls, conf = self.predict_cell_classes(prep["tensor"])
        t_infer = time.time()

        # Drop the zero-padding rows: their prediction belongs to no face,
        # and index -1 would silently attach it to the last one.
        real = prep["source_face"] >= 0
        cls = cls[real]
        conf = conf[real]
        # The rows were permuted, so the centroid is re-derived from the face
        # id each row carries rather than by unpicking the permutation.
        face_ids = prep["source_face"][real]
        centroids = v[f[face_ids]].mean(axis=1)

        coherence = quadrant_coherence(centroids, cls)

        # Background: gingiva (0) and the two classes predict.py discards.
        table = CLASS_TO_FDI[jaw]
        fdi_of_class = np.zeros(NUM_CLASSES, np.int64)
        for k, fdi in table.items():
            fdi_of_class[k] = fdi
        for k in BACKGROUND_CLASSES:
            fdi_of_class[k] = 0
        cell_fdi = fdi_of_class[np.clip(cls, 0, NUM_CLASSES - 1)]

        labels = self._upsample_to_vertices(centroids, cell_fdi, v)
        t_up = time.time()

        present = sorted({int(x) for x in labels if int(x) != 0})
        return {
            "labels": labels,
            "jaw": jaw,
            "cell_classes": cls,
            "cell_fdi": cell_fdi,
            "cell_confidence": conf,
            "cell_centroids": centroids,
            "teeth_found": len(present),
            "fdi_present": present,
            "quadrant_coherence": coherence,
            "labels_are_fdi_verified": False,
            "preparation": {k: prep[k] for k in
                            ("face_count", "sampled", "padded_rows",
                             "degenerate_faces", "selection",
                             "normalize_centroid", "normalize_radius_mm")},
            "timing_seconds": {
                "prepare": round(t_prep - t0, 3),
                "inference": round(t_infer - t_prep, 3),
                "upsample": round(t_up - t_infer, 3),
                "total": round(t_up - t0, 3),
            },
        }

    @staticmethod
    def _upsample_to_vertices(cell_centroids, cell_labels, verts):
        """KNN(3) on cell centroids, evaluated at the vertices."""
        from sklearn.neighbors import KNeighborsClassifier

        uniq = np.unique(cell_labels)
        if len(uniq) == 1:
            return np.full(len(verts), int(uniq[0]), np.int64)
        knn = KNeighborsClassifier(n_neighbors=min(UPSAMPLE_NEIGHBOURS,
                                                   len(cell_centroids)))
        knn.fit(np.asarray(cell_centroids, float),
                np.asarray(cell_labels).ravel())
        return knn.predict(np.asarray(verts, float)).astype(np.int64)


# --------------------------------------------------------------------------
# module-level load / status, mirroring tgn_bridge
# --------------------------------------------------------------------------

def _config_of(adapter):
    return (adapter.root, adapter.num_points, adapter.seed, adapter.device)


def load(root=None, force=False, **kwargs):
    """Build the adapter once, under a lock. Never raises; returns status().

    A DIFFERENT CONFIGURATION REBUILDS RATHER THAN BEING IGNORED, and that is
    not tidiness. The first version returned the cached adapter whenever one
    existed, so `load(num_points=512)` followed by `load()` handed back a
    512-point adapter to a caller asking for the trained 16,000 - and it said
    `loaded: True` while doing it. Found by the full suite: the real-scan test
    passed alone and failed after an earlier test in the same file had loaded
    a small adapter, reporting "the label array is indexed to a different
    vertex ordering" about labels that were simply computed from 3% of the
    faces. A shared singleton quietly serving the wrong configuration is the
    worst kind of defect this codebase collects: it answers.
    """
    with _LOCK:
        want = _config_of(CrossToothAdapter(root=root, **kwargs))
        if _STATE["loaded"] and not force:
            if _config_of(_STATE["adapter"]) == want:
                return status()
            print(f"[CrossTooth] reloading: configuration changed from "
                  f"{_config_of(_STATE['adapter'])} to {want}")
        t0 = time.time()
        try:
            adapter = CrossToothAdapter(root=root, **kwargs).load()
            _STATE.update({"loaded": True, "error": None, "traceback": None,
                           "adapter": adapter,
                           "load_seconds": round(time.time() - t0, 2)})
        except Exception as e:                            # noqa: BLE001
            _STATE.update({"loaded": False, "adapter": None,
                           "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc(),
                           "load_seconds": round(time.time() - t0, 2)})
        return status()


def adapter():
    """The loaded adapter, or raise with the load error attached."""
    if not _STATE["loaded"]:
        raise RuntimeError(_STATE["error"] or
                           "CrossTooth is not loaded; call load() first.")
    return _STATE["adapter"]


def status():
    return {
        "name": "crosstooth",
        "loaded": bool(_STATE["loaded"]),
        "error": _STATE["error"],
        "load_seconds": _STATE["load_seconds"],
        "root": crosstooth_path(),
        "checkpoint": os.path.join(crosstooth_path(), CHECKPOINT_REL),
        "checkpoint_present": os.path.isfile(
            os.path.join(crosstooth_path(), CHECKPOINT_REL)),
    }


def unload():
    """Drop the model. For tests and for reclaiming memory."""
    with _LOCK:
        _STATE.update({"loaded": False, "adapter": None, "error": None,
                       "traceback": None, "load_seconds": None})
    return status()
