"""
tgn_match_architecture.py — determine WHICH model class a checkpoint belongs
to, by matching its top-level module prefixes against the attribute names each
class declares.

The reasoning: a PyTorch state_dict key like

    first_ins_cent_model.enc1.0.linear.weight
    ^^^^^^^^^^^^^^^^^^^^

is the attribute name of a submodule assigned in __init__ as
`self.first_ins_cent_model = ...`. Exactly one class in the repository will
declare that name. Find that class and you have the architecture — no
guessing, no trial loading, nothing executed.

This is static analysis. It imports nothing from the repository, so it works
even though the model cannot currently be constructed.

    python tgn_match_architecture.py ToothGroupNetwork ckpts/0707_cosannealing_val.h5
"""

import ast
import os
import sys
from collections import defaultdict


def checkpoint_prefixes(path, depth=1):
    """Top-level module names present in the checkpoint."""
    import torch
    obj = torch.load(path, map_location=torch.device("cpu"),
                     weights_only=False)
    for _ in range(3):
        if isinstance(obj, dict) and not any(hasattr(v, "shape")
                                             for v in obj.values()):
            for w in ("model_state_dict", "state_dict", "model", "net"):
                if w in obj:
                    obj = obj[w]
                    break
            else:
                break
        else:
            break
    keys = [k for k, v in obj.items() if hasattr(v, "shape")]
    prefixes = defaultdict(list)
    for k in keys:
        parts = k.split(".")
        head = parts[0]
        if head in ("module",) and len(parts) > 1:      # DataParallel wrapper
            head = parts[1]
        prefixes[head].append(k)
    shapes = {k: tuple(v.shape) for k, v in obj.items() if hasattr(v, "shape")}
    return prefixes, shapes


def declared_attributes(tgn_root):
    """
    For every class in the repo, collect the names assigned as `self.X = ...`
    in its methods. These are the candidate state_dict prefixes.
    """
    classes = {}
    for dirpath, dirnames, filenames in os.walk(tgn_root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", ".vscode", "ckpts"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(full, encoding="utf-8",
                                      errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                attrs = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for tgt in sub.targets:
                            if (isinstance(tgt, ast.Attribute)
                                    and isinstance(tgt.value, ast.Name)
                                    and tgt.value.id == "self"):
                                attrs.add(tgt.attr)
                if attrs:
                    key = (os.path.relpath(full, tgn_root), node.name)
                    classes[key] = attrs
    return classes


def find_literal_in_repo(tgn_root, needle):
    """Every file:line where a literal string appears. Cheap grep."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(tgn_root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", ".vscode", "ckpts"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                for i, line in enumerate(open(full, encoding="utf-8",
                                              errors="replace"), 1):
                    if needle in line:
                        hits.append((os.path.relpath(full, tgn_root), i,
                                     line.strip()[:110]))
            except OSError:
                pass
    return hits


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ToothGroupNetwork"
    ckpt = sys.argv[2]

    prefixes, shapes = checkpoint_prefixes(ckpt)
    print("=" * 70)
    print("CHECKPOINT TOP-LEVEL MODULES")
    print("=" * 70)
    for p, keys in sorted(prefixes.items(), key=lambda kv: -len(kv[1])):
        print(f"  {p:<30} {len(keys):>5} tensors")

    print()
    print("=" * 70)
    print("WHICH CLASS DECLARES THESE ATTRIBUTES?")
    print("=" * 70)
    classes = declared_attributes(root)
    wanted = set(prefixes)
    scored = []
    for (relpath, cls), attrs in classes.items():
        overlap = wanted & attrs
        if overlap:
            scored.append((len(overlap), relpath, cls, sorted(overlap)))
    scored.sort(reverse=True)

    if not scored:
        print("  NO CLASS IN THIS CHECKOUT DECLARES ANY OF THESE MODULES.")
        print()
        print("  That is the whole answer: this checkpoint was not produced by")
        print("  this code. Widening a channel count cannot bridge it — the")
        print("  module structure itself differs. Use the branch the weights")
        print("  came from:")
        print("      git checkout challenge_branch")
        print("  The README directs challenge-checkpoint users there "
              "explicitly.")
    else:
        for n, relpath, cls, ov in scored[:6]:
            print(f"  {n:>2} match  class {cls}  ({relpath})")
            print(f"           declares: {ov}")
        best = scored[0]
        if best[0] == len(wanted):
            print()
            print(f"  COMPLETE MATCH: {best[2]} in {best[1]}")
            print("  Find the model_name that maps to this class in")
            print("  inference_pipelines/inference_pipeline_maker.py and use it.")
        else:
            print()
            print(f"  PARTIAL MATCH ONLY ({best[0]} of {len(wanted)} modules).")
            print("  A partial match is not a match. Loading it would leave "
                  "the unmatched")
            print("  submodules randomly initialised.")

    print()
    print("=" * 70)
    print("WHERE EACH MODULE NAME APPEARS IN THE SOURCE")
    print("=" * 70)
    for p in sorted(prefixes):
        hits = find_literal_in_repo(root, p)
        print(f"\n  {p}:")
        if not hits:
            print("      (not mentioned anywhere in this checkout)")
        for relpath, line, text in hits[:5]:
            print(f"      {relpath}:{line}  {text}")

    print()
    print("=" * 70)
    print("CHANNEL WIDTHS IN THE CHECKPOINT")
    print("=" * 70)
    enc = [(k, s) for k, s in shapes.items()
           if ".enc" in k and k.endswith("linear.weight")]
    for k, s in enc[:8]:
        print(f"  {k:<58} {s}")
    if enc:
        widths = []
        for k, s in enc:
            if s[0] not in widths:
                widths.append(s[0])
        print(f"\n  encoder widths in order: {widths}")
        print("  If you were going to edit a config, THIS is the full ladder "
              "you would")
        print("  have to reproduce — not just the first number.")


if __name__ == "__main__":
    main()
