"""
tgn_find_width_config.py — find where the encoder width ladder is set.

The checkpoint wants [32, 64, 128, 256, 512]. The checkout is building
[16, 32, 64, 128, 256]. Those numbers come from somewhere. This locates it and
tells you whether it is a config value you can change, or a hardcoded literal
in the model definition (which is a genuine code difference and means the
branch is wrong).

    python tgn_find_width_config.py ToothGroupNetwork
"""

import ast
import os
import re
import sys

TARGET = [32, 64, 128, 256, 512]
CURRENT = [16, 32, 64, 128, 256]

HINT_NAMES = ("plane", "channel", "dim", "width", "feat", "c_in", "c_out",
              "hidden", "block", "enc", "input_feat", "output_feat")


def scan(root):
    findings = {"numeric_ladders": [], "named_assignments": [],
                "config_files": []}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", ".vscode", "ckpts"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            try:
                src = open(full, encoding="utf-8", errors="replace").read()
            except OSError:
                continue

            # 1. literal lists of ints that look like a width ladder
            for m in re.finditer(r"\[\s*(\d+(?:\s*,\s*\d+){2,})\s*\]", src):
                try:
                    nums = [int(x) for x in m.group(1).split(",")]
                except ValueError:
                    continue
                if len(nums) < 3:
                    continue
                doubling = all(b == a * 2 for a, b in zip(nums, nums[1:]))
                if not doubling:
                    continue
                line = src[:m.start()].count("\n") + 1
                kind = ("EXACT MATCH for checkpoint" if nums == TARGET
                        else "matches what your code builds" if nums == CURRENT
                        else "doubling ladder")
                ctx = src.splitlines()[line - 1].strip()[:110]
                findings["numeric_ladders"].append(
                    {"file": rel, "line": line, "values": nums,
                     "kind": kind, "context": ctx})

            # 2. assignments whose name suggests channel width
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                names = []
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.append(tgt.id)
                    elif isinstance(tgt, ast.Attribute):
                        names.append(tgt.attr)
                    elif isinstance(tgt, ast.Subscript):
                        if isinstance(tgt.slice, ast.Constant):
                            names.append(str(tgt.slice.value))
                for nm in names:
                    if not any(h in nm.lower() for h in HINT_NAMES):
                        continue
                    val = None
                    if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, int):
                        val = node.value.value
                    elif isinstance(node.value, (ast.List, ast.Tuple)):
                        try:
                            val = [e.value for e in node.value.elts
                                   if isinstance(e, ast.Constant)]
                        except Exception:
                            val = None
                    if val in (16, 32) or (isinstance(val, list)
                                           and val in (TARGET, CURRENT)):
                        findings["named_assignments"].append(
                            {"file": rel, "line": node.lineno, "name": nm,
                             "value": val,
                             "context": src.splitlines()[node.lineno - 1].strip()[:110]})

        if os.path.basename(dirpath) in ("train_configs", "inference_pipelines"):
            for fn in filenames:
                if fn.endswith(".py"):
                    findings["config_files"].append(
                        os.path.relpath(os.path.join(dirpath, fn), root))

    return findings


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "ToothGroupNetwork"
    f = scan(root)

    print("=" * 70)
    print("WIDTH LADDERS FOUND AS LITERALS")
    print("=" * 70)
    if not f["numeric_ladders"]:
        print("  none — the ladder is probably computed, e.g. planes[i] * 2")
    for d in sorted(f["numeric_ladders"],
                    key=lambda x: 0 if "EXACT" in x["kind"] else
                    1 if "your code" in x["kind"] else 2):
        print(f"  {d['file']}:{d['line']}  {d['values']}   <- {d['kind']}")
        print(f"      {d['context']}")

    print()
    print("=" * 70)
    print("CHANNEL-ISH ASSIGNMENTS SET TO 16 OR 32")
    print("=" * 70)
    if not f["named_assignments"]:
        print("  none found")
    for d in f["named_assignments"]:
        print(f"  {d['file']}:{d['line']}  {d['name']} = {d['value']}")
        print(f"      {d['context']}")

    print()
    print("=" * 70)
    print("CONFIG FILES WORTH READING")
    print("=" * 70)
    for c in sorted(set(f["config_files"])):
        print(f"  {c}")

    print()
    print("=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("""
  If a 16-valued width appears in train_configs/ or in the inference config
  inside inference_pipeline_maker.py, it is a CONFIG value. Change it to 32
  and stay on this branch. The README states the inference config must match
  the config the checkpoint was trained under, so aligning it is the intended
  operation, not a hack.

  If 16 appears only as a hardcoded literal inside models/, the code itself
  differs from the code that produced these weights. That is a real branch
  mismatch and editing model source to chase it is the wrong move.

  Either way, change ONE thing, re-run, and read the next error. Do not edit
  several numbers at once — you will not know which one mattered.
""")


if __name__ == "__main__":
    main()python tgn_find_width_config.py ToothGroupNetwork