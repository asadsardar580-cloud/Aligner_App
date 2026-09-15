#!/usr/bin/env python3
"""
test_picking.py — isolate VTK click-picking from everything else.

No QThread, no core_geometry, no custom widgets, no status bar. Just a
sphere and a click handler. If spheres appear here, picking works in your
environment and the fault is somewhere in app_ui.py's logic. If they do not,
the fault is in the Qt/VTK event plumbing and nothing in app_ui.py can fix
it.

    python test_picking.py

Rotate the sphere with a click-drag: no marker should appear.
Tap it without moving: a red marker should appear and the terminal should
print the 3D coordinate.
"""
import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMainWindow
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk


def excepthook(t, v, tb):
    print("".join(traceback.format_exception(t, v, tb)))
sys.excepthook = excepthook


class Win(QMainWindow):
    THRESH = 5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Picking isolation test — tap the sphere")
        self.resize(900, 700)

        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)
        self.plotter.add_mesh(pv.Sphere(radius=5.0), color="lightblue")
        self.plotter.reset_camera()
        self.plotter.render()

        self.iren = self.plotter.iren.interactor
        print(f"observer target: {type(self.iren).__name__}")

        self.iren.AddObserver("LeftButtonPressEvent", self.on_press, 1.0)
        self.iren.AddObserver("LeftButtonReleaseEvent", self.on_release, 1.0)
        print("observers installed — tap the sphere now\n")

        self.press = None
        self.n = 0

    def on_press(self, caller, _evt):
        self.press = caller.GetEventPosition()
        print(f"PRESS   at {self.press}")

    def on_release(self, caller, _evt):
        pos = caller.GetEventPosition()
        print(f"RELEASE at {pos}", end="  ")
        if self.press is None:
            print("-> no matching press")
            return
        dx, dy = pos[0] - self.press[0], pos[1] - self.press[1]
        self.press = None
        if dx * dx + dy * dy > self.THRESH ** 2:
            print(f"-> drag ({dx},{dy}), ignored")
            return

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        cid = picker.GetCellId()
        print(f"-> cell={cid}", end="  ")
        if cid < 0:
            print("(missed the mesh)")
            return

        p = picker.GetPickPosition()
        self.n += 1
        print(f"HIT at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  [marker {self.n}]")
        self.plotter.add_mesh(pv.Sphere(radius=0.3, center=p),
                              name=f"m{self.n}", color="red")
        self.plotter.render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Win()
    w.show()
    sys.exit(app.exec())
