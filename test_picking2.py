#!/usr/bin/env python3
"""
test_picking2.py — find a release hook that actually fires.

Your last run showed LeftButtonPressEvent firing while LeftButtonRelease
never arrived. Rather than guess which workaround is right, this attaches
FOUR independent release listeners and prints which ones fire:

  [A] iren  LeftButtonReleaseEvent      (the one that failed — control)
  [B] Qt    MouseButtonRelease filter   (bypasses VTK's event chain entirely)
  [C] style EndInteractionEvent         (fires when the camera style finishes)
  [D] iren  EndInteractionEvent

The pick is performed by whichever of B/C/D fires first, so if any of them
work you get red markers immediately AND we learn which mechanism to build
on.

    python test_picking2.py

Tap the sphere (no dragging). Report which letters print.
"""
import sys
import traceback

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QObject, QEvent
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk


def excepthook(t, v, tb):
    print("".join(traceback.format_exception(t, v, tb)))
sys.excepthook = excepthook


class ReleaseFilter(QObject):
    """Qt-level listener. Qt delivers mouse events to the widget regardless
    of what VTK's interactor style does internally, so this path survives a
    style that consumes the VTK release event."""
    def __init__(self, win):
        super().__init__()
        self.win = win

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.MouseButtonRelease:
            try:
                self.win.on_release("B", qt_event=ev)
            except Exception:
                traceback.print_exc()
        return False  # never consume; let VTK keep working normally


class Win(QMainWindow):
    THRESH = 6

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Release-hook test — tap the sphere, don't drag")
        self.resize(900, 700)

        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)
        self.plotter.add_mesh(pv.Sphere(radius=5.0), color="lightblue")
        self.plotter.reset_camera()
        self.plotter.render()

        self.iren = self.plotter.iren.interactor
        self.style = self.iren.GetInteractorStyle()
        print(f"interactor : {type(self.iren).__name__}")
        print(f"style      : {type(self.style).__name__}\n")

        self.press = None
        self.n = 0
        self.handled = False   # de-dupe when several hooks fire for one click

        # [A] control — expected to stay silent
        self.iren.AddObserver("LeftButtonPressEvent", self.on_press, 1.0)
        self.iren.AddObserver("LeftButtonReleaseEvent",
                              lambda c, e: self.on_release("A"), 1.0)

        # [B] Qt event filter
        self.filter = ReleaseFilter(self)
        self.plotter.interactor.installEventFilter(self.filter)

        # [C]/[D] end-of-interaction events
        if self.style is not None:
            self.style.AddObserver("EndInteractionEvent",
                                   lambda c, e: self.on_release("C"), 1.0)
        self.iren.AddObserver("EndInteractionEvent",
                              lambda c, e: self.on_release("D"), 1.0)

        print("Hooks A,B,C,D installed. Tap the sphere.\n")

    def on_press(self, caller, _evt):
        self.press = caller.GetEventPosition()
        self.handled = False
        print(f"PRESS at {self.press}")

    def _vtk_pos_from_qt(self, ev):
        """Convert Qt widget coords to VTK display coords, reusing VTK's own
        FlipY logic so HiDPI scaling is handled the same way pyvistaqt does."""
        w = self.plotter.interactor
        scale = w.devicePixelRatioF() if hasattr(w, "devicePixelRatioF") else 1.0
        x = int(round(ev.position().x() * scale))
        y = int(round(ev.position().y() * scale))
        self.iren.SetEventInformationFlipY(x, y, 0, 0, chr(0), 0, None)
        return self.iren.GetEventPosition()

    def on_release(self, tag, qt_event=None):
        print(f"  [{tag}] release hook fired", end="")
        if self.handled:
            print("  (already handled)")
            return
        if tag == "A":
            print("  (control hook — no pick attempted)")
            return

        pos = self._vtk_pos_from_qt(qt_event) if qt_event is not None \
            else self.iren.GetEventPosition()
        print(f"  at {pos}", end="")

        if self.press is not None:
            dx, dy = pos[0] - self.press[0], pos[1] - self.press[1]
            if dx * dx + dy * dy > self.THRESH ** 2:
                print(f"  -> drag ({dx},{dy}), ignored")
                self.handled = True
                return

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)
        cid = picker.GetCellId()
        print(f"  cell={cid}", end="")
        if cid < 0:
            print("  (missed mesh)")
            return

        p = picker.GetPickPosition()
        self.n += 1
        self.handled = True
        print(f"  HIT ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  [marker {self.n}]")
        self.plotter.add_mesh(pv.Sphere(radius=0.3, center=p),
                              name=f"m{self.n}", color="red")
        self.plotter.render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Win()
    w.show()
    sys.exit(app.exec())
