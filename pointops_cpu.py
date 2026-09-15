"""
pointops_cpu.py — pure-PyTorch CPU replacement for the `pointops` CUDA
extension used by ToothGroupNetwork / Point Transformer.
"""

import sys
import types
import torch

__all__ = [
    "furthestsampling", "knnquery", "queryandgroup", "grouping",
    "interpolation", "subtraction", "aggregation", "install",
]

KNN_RETURNS_SQUARED_DISTANCE = True
_KNN_BLOCK_ELEMS = 16_000_000
_KNN_CHUNK = 4096

def _as_int_list(offset):
    if isinstance(offset, torch.Tensor):
        return [int(v) for v in offset.detach().cpu().view(-1)]
    return [int(v) for v in offset]

def _segments(offset, new_offset=None):
    off = _as_int_list(offset)
    new_off = _as_int_list(new_offset) if new_offset is not None else off
    start = new_start = 0
    for end, new_end in zip(off, new_off):
        yield start, end, new_start, new_end
        start, new_start = end, new_end

def furthestsampling(xyz, offset, new_offset):
    xyz = xyz.float()
    out = []
    for start, end, new_start, new_end in _segments(offset, new_offset):
        n, m = end - start, new_end - new_start
        if m <= 0:
            continue
        pts = xyz[start:end]
        m = min(m, n)
        sel = torch.empty(m, dtype=torch.long)
        best = torch.full((n,), float("inf"))
        farthest = 0
        for i in range(m):
            sel[i] = farthest
            d = ((pts - pts[farthest]) ** 2).sum(-1)
            torch.minimum(best, d, out=best)
            farthest = int(torch.argmax(best))
        out.append(sel + start)
    if not out:
        return torch.empty(0, dtype=torch.int32)
    return torch.cat(out).int()

def knnquery(nsample, xyz, new_xyz, offset, new_offset):
    xyz = xyz.float()
    new_xyz = xyz if new_xyz is None else new_xyz.float()

    idx_out = torch.empty((new_xyz.shape[0], nsample), dtype=torch.int64)
    dist_out = torch.empty((new_xyz.shape[0], nsample), dtype=torch.float32)

    for start, end, new_start, new_end in _segments(offset, new_offset):
        ref = xyz[start:end]
        qry = new_xyz[new_start:new_end]
        n = ref.shape[0]
        if n == 0 or qry.shape[0] == 0:
            continue
        k = min(nsample, n)
        chunk = max(1, min(_KNN_CHUNK, _KNN_BLOCK_ELEMS // max(n, 1)))
        for lo in range(0, qry.shape[0], chunk):
            hi = min(lo + chunk, qry.shape[0])
            block = qry[lo:hi]
            d2 = torch.cdist(block, ref) ** 2
            kd, ki = torch.topk(d2, k, dim=1, largest=False, sorted=True)
            if k < nsample:
                pad = nsample - k
                ki = torch.cat([ki, ki[:, :1].expand(-1, pad)], dim=1)
                kd = torch.cat([kd, kd[:, :1].expand(-1, pad)], dim=1)
            a, b = new_start + lo, new_start + hi
            idx_out[a:b] = ki + start
            dist_out[a:b] = kd if KNN_RETURNS_SQUARED_DISTANCE else kd.sqrt()

    return idx_out.int(), dist_out

def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    xyz = xyz.float()
    new_xyz = xyz if new_xyz is None else new_xyz.float()
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)

    m = new_xyz.shape[0]
    c = feat.shape[1]
    flat = idx.reshape(-1).long()
    grouped_xyz = xyz[flat].view(m, nsample, 3) - new_xyz.unsqueeze(1)
    grouped_feat = feat[flat].view(m, nsample, c)
    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), dim=-1)
    return grouped_feat

def grouping(input, idx):
    return input[idx.reshape(-1).long()].view(idx.shape[0], idx.shape[1], -1)

def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)
    recip = 1.0 / (dist + 1e-8)
    weight = recip / recip.sum(dim=1, keepdim=True)
    out = torch.zeros(new_xyz.shape[0], feat.shape[1], dtype=feat.dtype)
    for i in range(k):
        out += feat[idx[:, i].long()] * weight[:, i].unsqueeze(-1)
    return out

def subtraction(input1, input2, idx):
    n, nsample = idx.shape
    return input1.unsqueeze(1) - input2[idx.reshape(-1).long()].view(n, nsample, -1)

def aggregation(input, position, weight, idx):
    n, nsample, c = position.shape
    w_c = weight.shape[-1]
    if c % w_c != 0:
        raise ValueError(
            f"aggregation: {c} feature channels are not divisible by "
            f"{w_c} weight channels; the grouping is undefined.")
    grouped = input[idx.reshape(-1).long()].view(n, nsample, c) + position
    grouped = grouped.view(n, nsample, w_c, c // w_c)
    return (grouped * weight.unsqueeze(-1)).sum(dim=1).view(n, c)

_ALIASES = (
    "pointops",
    "pointops_cuda",
    "external_libs.pointops.functions.pointops",
    "lib.pointops.functions.pointops",
)

class _LeafRedirect:
    def __init__(self, names, module):
        self.names = set(names)
        self.module = module

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None                       
        import importlib.util
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        pass                                   

_HOOK = None

def install(extra_aliases=(), replace=True):
    global _HOOK
    me = sys.modules[__name__]
    names = tuple(_ALIASES) + tuple(extra_aliases)

    if replace:
        for name in names:
            sys.modules.pop(name, None)
            parts = name.split(".")
            for i in range(1, len(parts)):
                pkg = ".".join(parts[:i])
                mod = sys.modules.get(pkg)
                if mod is not None and getattr(mod, "__path__", None) == []:
                    sys.modules.pop(pkg, None)

    if _HOOK is not None and _HOOK in sys.meta_path:
        sys.meta_path.remove(_HOOK)
    _HOOK = _LeafRedirect(names, me)
    sys.meta_path.insert(0, _HOOK)

    for name in names:
        if "." not in name:
            sys.modules[name] = me
    return me