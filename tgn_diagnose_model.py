import ast
import os
import sys
from collections import OrderedDict

def valid_model_names(tgn_root):
    path = os.path.join(tgn_root, "inference_pipelines", "inference_pipeline_maker.py")
    if not os.path.exists(path):
        return {"error": f"not found: {path}"}

    src = open(path, "r", encoding="utf-8", errors="replace").read()
    report = {"file": path, "accepted": [], "illegal_raises": [], "raises": []}

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"error": f"cannot parse: {e}"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for cmp_node in [node.left] + list(node.comparators):
                if isinstance(cmp_node, ast.Constant) and isinstance(cmp_node.value, str):
                    report["accepted"].append(cmp_node.value)
                elif isinstance(cmp_node, (ast.Tuple, ast.List, ast.Set)):
                    for el in cmp_node.elts:
                        if isinstance(el, ast.Constant) and isinstance(el.value, str):
                            report["accepted"].append(el.value)
        if isinstance(node, ast.Raise) and node.exc is not None:
            if isinstance(node.exc, ast.Constant) and isinstance(node.exc.value, str):
                report["illegal_raises"].append({"line": node.lineno, "text": node.exc.value})
            else:
                report["raises"].append(node.lineno)

    seen, ordered = set(), []
    for s in report["accepted"]:
        if s and s not in seen and len(s) < 40 and "/" not in s:
            seen.add(s)
            ordered.append(s)
    report["accepted"] = ordered
    return report

FINGERPRINTS = OrderedDict([
    ("tgnet / tsegnet — centroid or offset head", ("offset", "centroid", "cent_", "dist_")),
    ("point transformer backbone", ("transformer", "linear_q", "linear_k", "linear_v", "blocks.")),
    ("pointnet++ set abstraction", ("sa1", "sa2", "sa3", "mlp_convs", "fp1", "fp2")),
    ("dgcnn edge conv", ("conv1.0.weight", "knn", "edge")),
    ("plain pointnet", ("stn", "feat.stn", "fstn")),
])

def load_state_dict(path):
    import torch
    obj = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    trail = []
    for _ in range(3):
        if isinstance(obj, dict):
            keys = list(obj.keys())
            tensor_like = [k for k in keys if hasattr(obj[k], "shape")]
            if tensor_like:
                return obj, trail
            for wrapper in ("model_state_dict", "state_dict", "model", "net", "weights"):
                if wrapper in obj:
                    trail.append(wrapper)
                    obj = obj[wrapper]
                    break
            else:
                return obj, trail
        else:
            break
    return obj, trail

def inspect_checkpoint(path):
    out = {"path": path, "exists": os.path.exists(path)}
    if not out["exists"]:
        out["error"] = "file not found"
        return out
    out["size_mb"] = round(os.path.getsize(path) / 1e6, 1)

    try:
        sd, trail = load_state_dict(path)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["unwrapped_via"] = trail or "(top level)"
    if not isinstance(sd, dict):
        out["error"] = f"expected a dict of tensors, got {type(sd).__name__}"
        return out

    params = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}
    out["n_params"] = len(params)
    if not params:
        out["error"] = "no tensors found"
        return out

    keys = list(params.keys())
    out["first_keys"] = keys[:12]
    out["last_keys"] = keys[-6:]

    joined = " ".join(keys).lower()
    matches = []
    for label, frags in FINGERPRINTS.items():
        hits = [f for f in frags if f.lower() in joined]
        if hits:
            matches.append({"architecture": label, "matched_on": hits})
    out["fingerprint"] = matches or "no known signature matched"

    for k in keys:
        shp = params[k]
        if len(shp) >= 2 and ("weight" in k):
            out["first_weight"] = {"key": k, "shape": shp, "output_channels": shp[0]}
            break

    return out

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ToothGroupNetwork"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 68)
    print("1. model_name strings your inference_pipeline_maker.py compares")
    print("=" * 68)
    rep = valid_model_names(root)
    if "error" in rep:
        print("  " + rep["error"])
    else:
        print(f"  file: {rep['file']}")
        print(f"  ACCEPTED NAMES: {rep['accepted'] or '(none found)'}")

    if not ckpt:
        return

    print()
    print("=" * 68)
    print("2. what architecture is this checkpoint?")
    print("=" * 68)
    info = inspect_checkpoint(ckpt)
    for k, v in info.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv}")
        elif isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"      {item}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()