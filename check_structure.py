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

    python check_structure.py app_ui.py core_geometry.py server.py
"""
import ast
import builtins
import sys


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
    files = sys.argv[1:] or ["app_ui.py", "core_geometry.py"]
    print("Structural check (no imports, no display needed):")
    if all(check(f) for f in files):
        print("\nAll files structurally sound.")
    else:
        print("\nFIX THE ABOVE BEFORE RUNNING.")
        sys.exit(1)
