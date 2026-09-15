#!/usr/bin/env python3
"""
diagnose.py — run this FIRST, before app_ui.py.

Prints what pyvistaqt actually exposes in YOUR installed versions, so the
picker gets wired to a real vtkRenderWindowInteractor instead of a QWidget.
This is the thing I could not determine remotely; thirty seconds here beats
another round of guessing.

    python diagnose.py
"""
import sys, traceback

print("=" * 68)
print("VERSIONS")
print("=" * 68)
for name in ("PyQt6.QtCore", "vtk", "pyvista", "pyvistaqt", "numpy", "scipy"):
    try:
        m = __import__(name, fromlist=["x"])
        v = getattr(m, "__version__", None) or getattr(m, "QT_VERSION_STR", "?")
        print(f"  {name:<16} {v}")
    except Exception as e:
        print(f"  {name:<16} MISSING ({e})")

from PyQt6.QtWidgets import QApplication
import pyvista as pv
from pyvistaqt import QtInteractor

app = QApplication(sys.argv)
plotter = QtInteractor()
plotter.add_mesh(pv.Sphere())

print("\n" + "=" * 68)
print("INTERACTOR RESOLUTION  (which object can take VTK observers?)")
print("=" * 68)

candidates = [
    ("plotter.iren.interactor",                          lambda p: p.iren.interactor),
    ("plotter.interactor.GetRenderWindow().GetInteractor()",
                                                         lambda p: p.interactor.GetRenderWindow().GetInteractor()),
    ("plotter.render_window.GetInteractor()",            lambda p: p.render_window.GetInteractor()),
    ("plotter.iren",                                     lambda p: p.iren),
    ("plotter.interactor",                               lambda p: p.interactor),
]

winner = None
for label, fn in candidates:
    try:
        obj = fn(plotter)
        has_obs = hasattr(obj, "AddObserver")
        has_pos = hasattr(obj, "GetEventPosition")
        ok = has_obs and has_pos
        print(f"  {'USABLE ' if ok else 'no     '} {label}")
        print(f"            type={type(obj).__name__:<32} AddObserver={has_obs} GetEventPosition={has_pos}")
        if ok and winner is None:
            winner = label
    except Exception as e:
        print(f"  error   {label}  ->  {type(e).__name__}: {e}")

print("\n" + "=" * 68)
if winner:
    print(f"RESULT: app_ui.py should attach observers via  {winner}")
    print("app_ui.py resolves this automatically at runtime and logs which one it picked.")
else:
    print("RESULT: no usable interactor found. Paste this whole output back to me.")
print("=" * 68)

print("\nRenderer check:")
try:
    print(f"  plotter.renderer -> {type(plotter.renderer).__name__}  (needed for vtkCellPicker.Pick)")
except Exception as e:
    print(f"  plotter.renderer FAILED: {e}")

plotter.close()
print("\nDone. No window should have appeared; this is a headless probe.")
