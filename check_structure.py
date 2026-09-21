#!/usr/bin/env python3
"""
check_structure.py — catch undefined names WITHOUT importing the module.

Why this exists: `python -m py_compile app_ui.py` validates syntax only. A
file can compile perfectly and still die at startup on a NameError, which is
exactly what happened when a patch deleted the ReleaseFilter class while
three references to it remained. Importing app_ui.py to check would require
PyQt6/VTK and a display; parsing it does not.

Not a full linter. It resolves module-level definitions, imports, function
parameters, assignments, comprehension targets, and builtins, then reports
any remaining referenced name. Run it after every edit:

    python check_structure.py                 # every application file
    python check_structure.py app_ui.py       # or just the ones you touched

It used to default to two hard-coded files, which made it useless as the
repo-wide gate it is invoked as. It now walks the tree, skipping the vendored
network (not ours to police), the archive (dead by definition), and the
generated export copy.
"""
import ast
import builtins
import os
import sys

SKIP_DIRS = {
    ".venv", "node_modules", "__pycache__", ".git", "dist", "ssr_out", "ssr_out2",
    "_archive",               # dead code, kept for reference only
    "Aligner_App_AI_Export",  # generated duplicate of the whole tree
    "ToothGroupNetwork",      # vendored third party — never edited, never linted
    "CrossTooth",             # ditto; its own `compete/` folder carries
                              # 12 more research checkouts. crosstooth_bridge.py
                              # is OURS and stays in the walk.
    "exports", "segmented",
}


def discover(root="."):
    """Every application .py file, in a stable order."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                found.append(os.path.normpath(os.path.join(dirpath, fn)))
    return found


def collect_bindings(node, names):
    """Record every name this node binds."""
    for child in ast.walk(node):
        if isinstance(child, ast.Lambda):
            a = child.args
            for arg in a.args + a.posonlyargs + a.kwonlyargs:
                names.add(arg.arg)
            if a.vararg:
                names.add(a.vararg.arg)
            if a.kwarg:
                names.add(a.kwarg.arg)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = child.args
                for arg in a.args + a.posonlyargs + a.kwonlyargs:
                    names.add(arg.arg)
                if a.vararg:
                    names.add(a.vararg.arg)
                if a.kwarg:
                    names.add(a.kwarg.arg)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
        elif isinstance(child, ast.Global):
            names.update(child.names)
        elif isinstance(child, (ast.withitem,)):
            pass
    return names


def check(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)

    defined = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "self", "cls"}
    collect_bindings(tree, defined)

    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                problems.append((node.lineno, node.id))

    if problems:
        print(f"  {path}: {len(problems)} undefined name(s)")
        seen = set()
        for line, name in sorted(problems):
            if name not in seen:
                seen.add(name)
                print(f"     line {line:>4}: {name}")
        return False
    print(f"  {path}: OK")
    return True


if __name__ == "__main__":
    files = sys.argv[1:] or discover()
    print(f"Structural check of {len(files)} file(s) (no imports, no display needed):")
    # all() short-circuits, which would stop at the first bad file and hide the
    # rest — run every file, then decide.
    results = [check(f) for f in files]
    ok = sum(results)
    print(f"\n  {ok}/{len(results)} files structurally sound")
    if ok != len(results):
        print("FIX THE ABOVE BEFORE RUNNING.")
        sys.exit(1)
    print("All files structurally sound.")
