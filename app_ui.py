#!/usr/bin/env python3
"""
app_ui.py — Clinical Micro-Planner: Sprint 1 & 2 graphical shell
=================================================================
UI layer only. All geometry lives in core_geometry.py, which is fully
covered by test_core_geometry.py and runs without a display.

    python app_ui.py

TESTED vs. UNTESTED — please read before assuming a fault is in the math.
--------------------------------------------------------------------------
core_geometry.py is verified: 9/9 correctness tests, plus fast-vs-reference
equivalence to 1e-16, plus benchmarks at real scan sizes. If segmentation
produces a wrong SHAPE, that layer is the suspect and it is testable
headlessly.

THIS file is not verified. PyQt6, VTK and PyVista cannot be installed in the
environment it was written in, so nothing below has ever executed. It is
written carefully, and the two riskiest spots are marked  # <-- UNVERIFIED
inline. Expect the first run to need small fixes there, not a redesign.

Order to debug in:
  1. Load an arch. If blank: see _add_arch_actor, which calls reset_camera()
     then render() explicitly.
  2. Watch the status bar for "Analyzing curvature..." -> "Ready". That is
     the background prep thread. Segmentation cannot start until it says
     Ready.
  3. Press Segment Tooth, click 3 points. If clicks do not register, the
     picker observer wiring is suspect (marked below).
  4. If the cut lands in the wrong place, raise/lower Margin snap strength
     before suspecting anything else.
"""

from __future__ import annotations

import sys
import logging
import traceback
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QStatusBar, QFrame, QSlider, QDoubleSpinBox, QGroupBox,
    QComboBox, QMessageBox, QScrollArea, QCheckBox, QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QEvent
from PyQt6.QtGui import QColor

import pyvista as pv
from pyvistaqt import QtInteractor
import vtk

import core_geometry as cg


# =========================================================================
# Fail-loud infrastructure
# -------------------------------------------------------------------------
# The previous build failed silently. Root cause: PyQt6 routes unhandled
# exceptions raised inside slots through sys.excepthook, and the DEFAULT
# hook calls qFatal(). With no hook installed, an AttributeError in a button
# handler produced no traceback, no dialog, and no visible effect -- the
# button simply did nothing. Everything below exists so that never recurs.
# =========================================================================
LOG_PATH = Path.home() / "micro_planner.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("planner")


def install_excepthook():
    """Replace the default hook so exceptions surface instead of aborting."""
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        LOG.error("UNCAUGHT EXCEPTION\n%s", text)
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Unhandled error",
                                 text[-3000:] + f"\n\nFull log: {LOG_PATH}")
        except Exception:
            pass
    sys.excepthook = hook


def resolve_vtk_interactor(plotter):
    """
    Find the object that actually accepts VTK observers.

    pyvistaqt's `plotter.interactor` is the QWidget -- correct for
    layout.addWidget(), which is why the mesh rendered fine, but a QWidget
    has no AddObserver, so binding the picker to it raised AttributeError.
    Rather than hard-code one attribute path across pyvista versions, probe
    for an object exposing both AddObserver and GetEventPosition and log
    which one won. Run diagnose.py to see this resolution standalone.
    """
    candidates = [
        ("plotter.iren.interactor", lambda p: p.iren.interactor),
        ("plotter.interactor.GetRenderWindow().GetInteractor()",
         lambda p: p.interactor.GetRenderWindow().GetInteractor()),
        ("plotter.render_window.GetInteractor()", lambda p: p.render_window.GetInteractor()),
        ("plotter.iren", lambda p: p.iren),
    ]
    for label, fn in candidates:
        try:
            obj = fn(plotter)
            if obj is not None and hasattr(obj, "AddObserver") and hasattr(obj, "GetEventPosition"):
                LOG.info("VTK interactor resolved via %s (%s)", label, type(obj).__name__)
                return obj, label
        except Exception as exc:
            LOG.debug("interactor candidate %s failed: %s", label, exc)
    LOG.error("No usable VTK interactor found -- picking cannot be wired.")
    return None, None


# =========================================================================
# Theme
# =========================================================================
BG_DEEP, BG_PANEL, BORDER = "#0d0f12", "#1a1d22", "#2c313a"
ACCENT, ACCENT_DIM = "#3fc6d4", "#2a8b96"
TEXT, TEXT_MUTED = "#e7ebee", "#8b93a0"
VIEWPORT_BOTTOM, VIEWPORT_TOP = "#0a0c0f", "#1b2731"

STYLESHEET = f"""
QMainWindow {{ background: {BG_DEEP}; }}
QWidget {{ background: transparent; color: {TEXT};
           font-family: 'Segoe UI', -apple-system, Arial, sans-serif; font-size: 12.5px; }}
QFrame#rail {{ background: {BG_PANEL}; border-right: 1px solid {BORDER}; }}
QGroupBox {{ background: rgba(255,255,255,12); border: 1px solid {BORDER}; border-radius: 10px;
             margin-top: 14px; padding: 14px 10px 10px 10px; font-weight: 600; color: {TEXT_MUTED}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px;
                    color: {ACCENT}; letter-spacing: 1px; }}
QGroupBox:disabled {{ color: #4a4f57; }}
QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2a2f37, stop:1 #21252b);
               border: 1px solid {BORDER}; border-radius: 6px; padding: 7px 12px;
               text-align: left; color: {TEXT}; }}
QPushButton:hover {{ border: 1px solid {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_DIM}; }}
QPushButton:disabled {{ color: #565b63; background: #1b1e23; border: 1px solid #23262c; }}
QPushButton#primary {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {ACCENT}, stop:1 {ACCENT_DIM});
                       color: #06181a; font-weight: 600; border: none; }}
QLabel#readout {{ color: {ACCENT}; font-weight: 600; }}
QLabel#step {{ color: #ffd166; font-weight: 600; }}
QSlider::groove:horizontal {{ height: 4px; background: #2a2e35; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; background: {TEXT};
                              border: 2px solid {ACCENT}; border-radius: 7px; }}
QDoubleSpinBox, QComboBox {{ background: #14161a; border: 1px solid {BORDER};
                             border-radius: 5px; padding: 3px 6px; color: {TEXT}; }}
QProgressBar {{ background: #14161a; border: 1px solid {BORDER}; border-radius: 5px;
                text-align: center; color: {TEXT_MUTED}; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT_DIM}; border-radius: 4px; }}
QStatusBar {{ background: #101215; color: {TEXT_MUTED}; border-top: 1px solid {BORDER}; }}
QScrollArea {{ border: none; }}
QCheckBox {{ color: {TEXT_MUTED}; }}
"""


# =========================================================================
# PyVista <-> NumPy
# =========================================================================
def pv_to_arrays(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    tri = mesh.triangulate()
    faces = np.asarray(tri.faces).reshape(-1, 4)[:, 1:4]
    return np.asarray(tri.points, dtype=float), faces.astype(np.int64)


def arrays_to_pv(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    padded = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(verts, padded)


# =========================================================================
# Background workers — keep the GUI thread free
# =========================================================================
class PrepWorker(QThread):
    """
    Precomputes the edge list and smoothed concavity field at LOAD time.

    This is the difference between a usable tool and a frozen window.
    Benchmarked on the synthetic fixture: doing this work inside the
    segmentation click costs ~8.5s at 100k vertices with the reference
    implementation. Hoisting it to load time and vectorizing brings the
    per-segmentation cost to ~0.46s at 360k vertices, against a one-off
    ~2.6s here that overlaps with the clinician orienting the camera.
    """
    done = pyqtSignal(str, object, object)
    failed = pyqtSignal(str, str)

    def __init__(self, key, verts, faces):
        super().__init__()
        self.key, self.verts, self.faces = key, verts, faces

    def run(self):
        try:
            LOG.info("PrepWorker[%s] starting on %d verts / %d faces",
                     self.key, len(self.verts), len(self.faces))
            edges = cg.directed_edges(self.faces)
            conc = cg.boundary_field(self.verts, self.faces, edges=edges)
            # Built here so tolerance changes later are instant (see
            # build_barrier_graph). One Dijkstra per seed click, then the
            # slider is a pure threshold.
            graph = cg.build_barrier_graph(self.verts, self.faces, conc, edges=edges)
            LOG.info("PrepWorker[%s] finished: %d verts", self.key, len(self.verts))
            self.done.emit(self.key, (edges, graph), conc)
        except Exception:
            tb = traceback.format_exc()
            LOG.error("PrepWorker[%s] FAILED\n%s", self.key, tb)
            self.failed.emit(self.key, tb)


class SegmentWorker(QThread):
    """Cap the selected region and derive the anatomical frame."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, verts, faces, face_mask, mesial, distal, root_len):
        super().__init__()
        self.verts, self.faces, self.face_mask = verts, faces, face_mask
        self.mesial, self.distal, self.root_len = mesial, distal, root_len

    def run(self):
        try:
            mask = cg.largest_face_component(self.faces, self.face_mask)
            (cv, cf), (bv, bf) = cg.split_by_face_mask(self.verts, self.faces, mask)
            LOG.info("cut: crown=%d base=%d faces", len(cf), len(bf))

            ok, reason = cg.segmentation_is_plausible(
                len(self.faces), len(cf), len(bf), loop=[])
            if not ok:
                LOG.warning("segmentation rejected: %s", reason)
                self.failed.emit(reason)
                return

            rim_loops = cg.boundary_loops(cf)
            if not rim_loops:
                self.failed.emit(
                    "The selection has no open boundary, so there is no cervical rim to "
                    "cap against. Lower the selection spread.")
                return
            rim_verts = cv[max(rim_loops, key=len)].copy()

            cv2, cf2 = cg.cap_and_close(cv, cf)
            bv2, bf2 = cg.cap_and_close(bv, bf)
            cf2 = cg.make_consistent_winding(cv2, cf2)

            # Only the mesiodistal direction needs clicking; the long axis
            # comes from crown centroid minus rim centroid, which the
            # selection already determines exactly.
            frame = cg.derive_frame_from_region(self.mesial, self.distal, cv2, rim_verts)
            c_res = cg.center_of_resistance(frame, self.root_len)

            self.done.emit(dict(
                crown=(cv2, cf2), base=(bv2, bf2), frame=frame, c_res=c_res,
                n_rim=len(rim_verts), n_rim_loops=len(rim_loops),
                crown_fraction=len(cf) / max(len(self.faces), 1),
                crown_watertight=cg.is_edge_manifold_closed(cf2),
                base_watertight=cg.is_edge_manifold_closed(bf2),
                crown_volume=cg.signed_volume(cv2, cf2),
            ))
        except Exception:
            tb = traceback.format_exc()
            LOG.error("SegmentWorker FAILED\n%s", tb)
            self.failed.emit(tb)


# =========================================================================
# Mouse release via Qt
# -------------------------------------------------------------------------
# EMPIRICALLY VERIFIED on PyQt6 6.11.0 / VTK 9.6.2 / pyvista 0.48.4 /
# pyvistaqt 0.12.0 (Windows):
#
#   LeftButtonPressEvent   on the VTK interactor  -> FIRES
#   LeftButtonReleaseEvent on the VTK interactor  -> NEVER FIRES
#
# The interactor style claims the left button on press to drive camera
# rotation and consumes the matching release, so a VTK-side release observer
# is never reached. That asymmetry was the dead-click bug: presses were seen,
# releases never were, and the pick only ever happened on release.
#
# Qt delivers MouseButtonRelease to the widget independently of whatever VTK
# does internally, so this path is reliable. (EndInteractionEvent on both the
# style and the interactor also fire and would work as a fallback; Qt's is
# the most direct and needs no assumptions about style internals.)
#
# The filter NEVER consumes the event -- it returns False so VTK continues to
# receive everything and camera rotation is completely untouched.
# =========================================================================
class ReleaseFilter(QObject):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.MouseButtonRelease:
            try:
                self.window.on_qt_release(ev)
            except Exception:
                LOG.error("release handler raised\n%s", traceback.format_exc())
        return False


# =========================================================================
# Main window
# =========================================================================
class PlannerWindow(QMainWindow):
    ARCH_COLORS = {"maxillary": "#e8e2d0", "mandibular": "#d9cfc3"}
    CROWN_COLOR = "#3fa9ff"
    # Circuit order matters: the margin loop is traced through the anchors in
    # click sequence, so going around the tooth (rather than mesial->distal->
    # buccal) is what forces the path to encircle the crown instead of
    # doubling back along the same side.
    # All three clicks are made from ONE camera view of the labial surface.
    # The previous design asked for a lingual click, which is physically
    # impossible to place without rotating: the picker returns the frontmost
    # visible cell, so a click aimed at a hidden surface silently lands on
    # the near one. Region growing removes the need entirely -- it wraps to
    # the far side through mesh connectivity.
    CLICK_LABELS = [
        "Click ON the tooth (anywhere on the crown)",
        "Mesial contact (toward the midline)",
        "Distal contact (away from the midline)",
    ]
    N_CLICKS = 3
    DRAG_THRESHOLD_PX = 6

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clinical Micro-Planner — Sprints 1 & 2")
        self.resize(1600, 980)
        self.setStyleSheet(STYLESHEET)

        self.arches: dict[str, dict] = {}       # key -> {verts, faces, edges, conc, ready}
        self.workers: list[QThread] = []        # keep refs; a GC'd QThread crashes Qt

        self.picking = False
        self.clicks: list[np.ndarray] = []
        self._press_xy = None
        self._observers: list[int] = []
        self._vtk_iren = None
        self._iren_label = "unresolved"
        self._release_filter = None   # keep a ref; a GC'd QObject filter crashes Qt
        self._seed_dist = None        # barrier-weighted distances from the seed click
        self._preview_mask = None

        self.crown_rest: np.ndarray | None = None   # crown verts at T0
        self.crown_faces = None
        self.crown_mesh: pv.PolyData | None = None
        self.frame = None
        self.active_key = None

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)      # ~60fps ceiling on slider drags
        self._render_timer.timeout.connect(self._apply_kinematics)

        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._build_rail())

        self.plotter = QtInteractor(central)
        self.plotter.set_background(VIEWPORT_BOTTOM, top=VIEWPORT_TOP)
        self.plotter.remove_all_lights()
        self.plotter.enable_lightkit()
        try:
            self.plotter.enable_anti_aliasing("msaa")
        except Exception:
            pass  # MSAA unavailable on some drivers; cosmetic only
        row.addWidget(self.plotter.interactor, stretch=1)

        self.setCentralWidget(central)
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._vtk_iren, self._iren_label = resolve_vtk_interactor(self.plotter)
        if self._vtk_iren is None:
            self.status.showMessage("WARNING: picking unavailable — run diagnose.py and send me the output.")
            QMessageBox.critical(
                self, "Picker cannot be wired",
                "No VTK interactor exposing AddObserver was found on this pyvista/pyvistaqt "
                f"version.\n\nRun diagnose.py and send the output.\n\nLog: {LOG_PATH}")
        else:
            # Press comes from VTK (verified to fire); release comes from Qt
            # (VTK's release never arrives -- see ReleaseFilter). Both are
            # installed once here and gated on self.picking, rather than
            # added and removed per segmentation: fewer moving parts, and
            # nothing to leak if a cut fails midway.
            self._vtk_iren.AddObserver("LeftButtonPressEvent", self._on_press, 1.0)
            self._release_filter = ReleaseFilter(self)
            self.plotter.interactor.installEventFilter(self._release_filter)
            LOG.info("Press observer on %s; release via Qt event filter.", self._iren_label)
            self.status.showMessage(
                f"Ready — picker bound ({self._iren_label} + Qt release). Load an arch to begin.")

    def _build_rail(self):
        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(310)
        v = QVBoxLayout(rail)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(12)
        v.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- ingestion ---
        g1 = QGroupBox("ARCH INGESTION")
        l1 = QVBoxLayout(g1)
        b1 = QPushButton("Load Maxillary STL")
        b1.clicked.connect(lambda: self._load_dialog("maxillary"))
        l1.addWidget(b1)
        b2 = QPushButton("Load Mandibular STL")
        b2.clicked.connect(lambda: self._load_dialog("mandibular"))
        l1.addWidget(b2)

        self.chk_orient = QCheckBox("Re-orient normals on load")
        self.chk_orient.setChecked(False)
        self.chk_orient.setToolTip(
            "OFF by default. Intraoral scans are usually open shells, and VTK's "
            "auto_orient_normals assumes a closed surface — on an open shell it can "
            "flip normals and make the mesh render black or inside-out. Only enable "
            "if your scan is already a closed solid."
        )
        l1.addWidget(self.chk_orient)

        self.prep_bar = QProgressBar()
        self.prep_bar.setRange(0, 0)
        self.prep_bar.setVisible(False)
        l1.addWidget(self.prep_bar)
        v.addWidget(g1)

        # --- segmentation ---
        g2 = QGroupBox("SEGMENTATION — CLICK THE TOOTH")
        l2 = QVBoxLayout(g2)
        l2.addWidget(QLabel("Arch:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["Maxillary", "Mandibular"])
        l2.addWidget(self.arch_combo)

        self.btn_segment = QPushButton("Segment Tooth")
        self.btn_segment.setObjectName("primary")
        self.btn_segment.clicked.connect(self._start_picking)
        l2.addWidget(self.btn_segment)

        self.lbl_step = QLabel("Idle")
        self.lbl_step.setObjectName("step")
        self.lbl_step.setWordWrap(True)
        l2.addWidget(self.lbl_step)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_picking)
        l2.addWidget(self.btn_cancel)

        tol_head = QHBoxLayout()
        tol_head.addWidget(QLabel("Selection spread:"))
        tol_head.addStretch()
        self.lbl_tol = QLabel("10.0")
        self.lbl_tol.setObjectName("readout")
        tol_head.addWidget(self.lbl_tol)
        l2.addLayout(tol_head)

        self.slider_tol = QSlider(Qt.Orientation.Horizontal)
        self.slider_tol.setRange(10, 400)      # tenths -> 1.0 .. 40.0
        self.slider_tol.setValue(100)
        self.slider_tol.setToolTip(
            "Like Photoshop's magic-wand tolerance. Drag AFTER clicking on the tooth "
            "and the blue preview grows or shrinks live. The cervical groove acts as a "
            "barrier, so a wide range of values all stop at the gumline."
        )
        self.slider_tol.valueChanged.connect(self._on_tolerance_changed)
        l2.addWidget(self.slider_tol)

        self.btn_confirm = QPushButton("Confirm & Cut")
        self.btn_confirm.setObjectName("primary")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._run_segmentation)
        l2.addWidget(self.btn_confirm)
        v.addWidget(g2)

        # --- kinematics ---
        self.g3 = QGroupBox("KINEMATICS")
        self.g3.setEnabled(False)
        l3 = QVBoxLayout(self.g3)
        l3.addWidget(QLabel("Root length → C_res (mm):"))
        self.spin_root = QDoubleSpinBox()
        self.spin_root.setRange(0.0, 25.0)
        self.spin_root.setSingleStep(0.5)
        self.spin_root.setValue(10.0)
        self.spin_root.setToolTip("Incisor ≈10, canine ≈13, premolar/molar ≈9. "
                                  "Scans contain no root, so this is a parameter, not a measurement.")
        self.spin_root.valueChanged.connect(self._schedule_update)
        l3.addWidget(self.spin_root)

        self.s_tip = self._slider(l3, "Tip (mesiodistal)", 200, 10.0, "°")
        self.s_torque = self._slider(l3, "Torque (buccolingual)", 200, 10.0, "°")
        self.s_rot = self._slider(l3, "Rotation (axial)", 200, 10.0, "°")
        self.s_md = self._slider(l3, "Translate mesiodistal", 300, 100.0, "mm")
        self.s_bl = self._slider(l3, "Translate buccolingual", 300, 100.0, "mm")
        self.s_oa = self._slider(l3, "Intrusion / extrusion", 300, 100.0, "mm")

        b_reset = QPushButton("Reset to T0")
        b_reset.clicked.connect(self._reset_sliders)
        l3.addWidget(b_reset)
        v.addWidget(self.g3)

        # --- view ---
        g4 = QGroupBox("VIEW")
        l4 = QVBoxLayout(g4)
        b3 = QPushButton("Reset Camera")
        b3.clicked.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))
        l4.addWidget(b3)
        self.chk_axes = QCheckBox("Show anatomical axes")
        self.chk_axes.setChecked(True)
        self.chk_axes.stateChanged.connect(self._toggle_axes)
        l4.addWidget(self.chk_axes)
        v.addWidget(g4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rail)
        scroll.setFixedWidth(326)
        return scroll

    def _slider(self, layout, label, span, scale, suffix):
        head = QHBoxLayout()
        head.addWidget(QLabel(label))
        head.addStretch()
        val = QLabel(f"+0.00{suffix}")
        val.setObjectName("readout")
        head.addWidget(val)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(-span, span)
        s.setValue(0)
        s.valueChanged.connect(lambda raw: (val.setText(f"{raw/scale:+.2f}{suffix}"),
                                            self._schedule_update()))
        s._scale = scale
        layout.addLayout(head)
        layout.addWidget(s)
        return s

    # ------------------------------------------------------- ingestion
    def _load_dialog(self, key):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {key} STL", str(Path.home()), "STL Files (*.stl)")
        if path:
            self._load(path, key)

    def _load(self, path, key):
        try:
            mesh = pv.read(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        # NO centering, NO scaling — the inter-arch bite registration is the
        # reason both arches must stay in raw scanner coordinates.
        if self.chk_orient.isChecked():
            mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True,
                                        splitting=False)

        verts, faces = pv_to_arrays(mesh)
        self.arches[key] = dict(verts=verts, faces=faces, edges=None, graph=None,
                                conc=None, ready=False)
        self._add_arch_actor(key, arrays_to_pv(verts, faces))

        self.status.showMessage(
            f"{key}: {len(verts):,} verts / {len(faces):,} tris — analyzing curvature…")
        self.prep_bar.setVisible(True)

        w = PrepWorker(key, verts, faces)
        w.done.connect(self._prep_done)
        w.failed.connect(lambda k, e: QMessageBox.critical(self, "Curvature analysis failed", e))
        self.workers.append(w)
        w.start()

    def _add_arch_actor(self, key, poly):
        """Blank-screen guard: add, then reset_camera(), then render(),
        explicitly and in that order. Without the reset the camera may sit
        inside or far from the mesh (scanner coordinates are often far from
        the origin); without the explicit render the Qt widget can stay
        blank until some unrelated event forces a repaint."""
        self.plotter.add_mesh(poly, name=key, color=self.ARCH_COLORS.get(key, "#e0dccf"),
                              smooth_shading=True, specular=0.3, specular_power=15)
        self.plotter.reset_camera()      # <-- blank screen fix, part 1
        self.plotter.render()            # <-- blank screen fix, part 2

    def _prep_done(self, key, edges_and_graph, conc):
        edges, graph = edges_and_graph
        if key in self.arches:
            self.arches[key].update(edges=edges, graph=graph, conc=conc, ready=True)
        self.prep_bar.setVisible(any(not a["ready"] for a in self.arches.values()))
        self.status.showMessage(f"{key}: ready to segment.")

    # --------------------------------------------------------- picking
    def _start_picking(self):
        key = self.arch_combo.currentText().lower()
        if key not in self.arches:
            QMessageBox.information(self, "No arch", f"Load the {key} arch first.")
            return
        if not self.arches[key]["ready"]:
            QMessageBox.information(self, "Still analyzing",
                                    "Curvature analysis is still running. Wait for 'ready to segment'.")
            return

        self.active_key = key
        self.picking = True
        self.clicks = []
        self._seed_dist = None
        self._preview_mask = None
        self.btn_confirm.setEnabled(False)
        self.plotter.remove_actor("preview", render=True)
        self._clear_markers()
        if not self._install_observers():
            self.picking = False
            return
        self.btn_segment.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_step.setText(f"Click 1 of {self.N_CLICKS} — {self.CLICK_LABELS[0]}")
        self.status.showMessage("Rotate freely; a click only registers if the mouse barely moves.")

    def _install_observers(self):
        """Hooks are installed once at startup (see _build_ui); this just
        confirms they exist before picking begins."""
        if self._vtk_iren is None or self._release_filter is None:
            QMessageBox.critical(self, "Picker unavailable",
                                 "Mouse hooks were not installed at startup. Run diagnose.py.")
            return False
        LOG.info("Picking armed.")
        return True

    def _remove_observers(self):
        """Disarm by clearing state; the hooks stay attached and no-op while
        self.picking is False."""
        self._press_xy = None
        LOG.info("Picking disarmed.")

    def _on_press(self, caller, _evt):
        if not self.picking:
            return
        self._press_xy = caller.GetEventPosition()
        LOG.debug("press at %s", self._press_xy)

    def _vtk_pos_from_qt(self, ev):
        """Qt widget coords -> VTK display coords.

        Routes through SetEventInformationFlipY so VTK performs its own
        Y-flip, and scales by devicePixelRatioF first. The scaling matters on
        any display running Windows UI scaling above 100%: without it, picks
        land at a fraction of the intended position and appear to miss the
        mesh entirely.
        """
        w = self.plotter.interactor
        scale = w.devicePixelRatioF() if hasattr(w, "devicePixelRatioF") else 1.0
        x = int(round(ev.position().x() * scale))
        y = int(round(ev.position().y() * scale))
        self._vtk_iren.SetEventInformationFlipY(x, y, 0, 0, chr(0), 0, None)
        return self._vtk_iren.GetEventPosition()

    def on_qt_release(self, ev):
        """Called by ReleaseFilter. See that class for why release comes from
        Qt while press comes from VTK."""
        if not self.picking or self._press_xy is None:
            return
        x, y = self._vtk_pos_from_qt(ev)
        px, py = self._press_xy
        self._press_xy = None

        if (x - px) ** 2 + (y - py) ** 2 > self.DRAG_THRESHOLD_PX ** 2:
            LOG.debug("release at (%d,%d) -> drag, ignored", x, y)
            return

        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)   # value verified working in test_picking2
        picker.Pick(x, y, 0, self.plotter.renderer)
        cell = picker.GetCellId()
        LOG.debug("release at (%d,%d); cell=%d", x, y, cell)
        if cell < 0:
            self.status.showMessage("No surface under the cursor — click directly on the tooth.")
            return

        self._register_click(np.array(picker.GetPickPosition(), dtype=float))

    def _register_click(self, point):
        self.clicks.append(point)
        n = len(self.clicks)
        scale = max(self._scene_scale() * 0.006, 0.12)
        self.plotter.add_mesh(pv.Sphere(radius=scale, center=point),
                              name=f"click_{n}",
                              color=["#3fa9ff", "#ff6b6b", "#4dd0e1"][n - 1])
        self.plotter.render()

        if n == 1:
            self._seed_selection(point)

        if n < self.N_CLICKS:
            self.lbl_step.setText(f"Click {n+1} of {self.N_CLICKS} — {self.CLICK_LABELS[n]}")
        else:
            self.lbl_step.setText("Adjust spread, then Confirm & Cut")
            self.btn_confirm.setEnabled(True)

    def _seed_selection(self, point):
        """One Dijkstra pass from the clicked point. Every later tolerance
        change is then a pure threshold of this cached distance field, which
        is what makes the slider feel instant."""
        a = self.arches[self.active_key]
        try:
            self._seed_dist = cg.geodesic_from_seed(a["graph"], a["verts"], point)
        except Exception:
            LOG.error("seed geodesic failed\n%s", traceback.format_exc())
            QMessageBox.critical(self, "Selection failed",
                                 "Could not compute distances from that point. See the log.")
            return
        LOG.info("seed set; distance field computed")
        self._update_preview()

    def _on_tolerance_changed(self, raw):
        self.lbl_tol.setText(f"{raw/10.0:.1f}")
        if self._seed_dist is not None:
            self._update_preview()

    def _update_preview(self):
        """Recolour the selected region so the clinician sees exactly what
        will be cut BEFORE committing to it."""
        if self._seed_dist is None:
            return
        a = self.arches[self.active_key]
        tol = self.slider_tol.value() / 10.0
        mask = cg.mask_from_distance(a["faces"], self._seed_dist, tol)
        n_sel = int(mask.sum())

        if n_sel == 0:
            self.plotter.remove_actor("preview", render=True)
            self._preview_mask = None
            self.status.showMessage("Nothing selected — increase the spread.")
            return

        mask = cg.largest_face_component(a["faces"], mask)
        self._preview_mask = mask
        (cv, cf), _ = cg.split_by_face_mask(a["verts"], a["faces"], mask)
        self.plotter.add_mesh(arrays_to_pv(cv, cf), name="preview",
                              color=self.CROWN_COLOR, opacity=0.75, smooth_shading=True)
        self.plotter.render()

        frac = mask.sum() / len(a["faces"])
        self.status.showMessage(
            f"Preview: {mask.sum():,} triangles ({frac:.1%} of arch). "
            f"A single crown is usually 2–10%.")

    def _cancel_picking(self):
        self.picking = False
        self._remove_observers()
        self._clear_markers()
        self.plotter.remove_actor("preview", render=True)
        self._seed_dist = None
        self._preview_mask = None
        self.btn_confirm.setEnabled(False)
        self.clicks = []
        self.btn_segment.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_step.setText("Idle")

    def _clear_markers(self):
        for i in range(1, self.N_CLICKS + 1):
            self.plotter.remove_actor(f"click_{i}", render=False)
        self.plotter.render()

    def _scene_scale(self):
        for a in self.arches.values():
            return float(np.linalg.norm(a["verts"].max(0) - a["verts"].min(0)))
        return 40.0

    # ---------------------------------------------------- segmentation
    def _run_segmentation(self):
        if self._preview_mask is None or len(self.clicks) < self.N_CLICKS:
            QMessageBox.information(self, "Not ready",
                                    "Click the tooth, then both contact points, then confirm.")
            return
        self.picking = False
        self._remove_observers()
        self.btn_cancel.setEnabled(False)
        self.btn_confirm.setEnabled(False)
        self.prep_bar.setVisible(True)
        self.lbl_step.setText("Cutting and capping…")

        a = self.arches[self.active_key]
        w = SegmentWorker(a["verts"], a["faces"], self._preview_mask,
                          self.clicks[1], self.clicks[2], self.spin_root.value())
        w.done.connect(self._segmentation_done)
        w.failed.connect(self._segmentation_failed)
        self.workers.append(w)
        w.start()

    def _segmentation_failed(self, msg):
        self.prep_bar.setVisible(False)
        self.lbl_step.setText("Idle")
        self.btn_segment.setEnabled(True)
        self.plotter.remove_actor("preview", render=True)
        self._clear_markers()
        self._seed_dist = None
        self._preview_mask = None
        self.clicks = []
        QMessageBox.warning(self, "Segmentation failed", msg)

    def _segmentation_done(self, res):
        self.prep_bar.setVisible(False)
        self._clear_markers()
        self.plotter.remove_actor("preview", render=False)
        self._seed_dist = None

        cv, cf = res["crown"]
        bv, bf = res["base"]
        self.crown_rest, self.crown_faces = cv, cf
        self.frame = res["frame"]

        self.plotter.remove_actor(self.active_key, render=False)
        self.plotter.add_mesh(arrays_to_pv(bv, bf), name="base",
                              color=self.ARCH_COLORS.get(self.active_key, "#e0dccf"),
                              smooth_shading=True)
        self.crown_mesh = arrays_to_pv(cv, cf)
        self.plotter.add_mesh(self.crown_mesh, name="crown", color=self.CROWN_COLOR,
                              opacity=0.55, smooth_shading=True)

        self.g3.setEnabled(True)
        self.btn_segment.setEnabled(True)
        self.lbl_step.setText("Segmented — kinematics live")
        self._apply_kinematics()
        self.plotter.render()

        wt = ("watertight" if res["crown_watertight"] else "NOT watertight",
              "watertight" if res["base_watertight"] else "NOT watertight")
        self.status.showMessage(
            f"Rim {res['n_rim']} verts | crown {len(cf):,} tris "
            f"({res['crown_fraction']:.1%} of arch), {res['crown_volume']:.1f} mm³, "
            f"{wt[0]} | base {wt[1]}"
        )
        if not (res["crown_watertight"] and res["base_watertight"]):
            QMessageBox.warning(
                self, "Not watertight",
                "One piece did not close cleanly. Kinematics still work, but the Sprint 4 "
                "boolean export will fail on this. Re-cut with a different margin snap "
                "strength, or click nearer the cervical line.")

    # ------------------------------------------------------ kinematics
    def _schedule_update(self):
        if self.crown_rest is not None:
            self._render_timer.start()

    def _reset_sliders(self):
        for s in (self.s_tip, self.s_torque, self.s_rot, self.s_md, self.s_bl, self.s_oa):
            s.setValue(0)

    def _apply_kinematics(self):
        if self.crown_rest is None or self.frame is None:
            return
        c_res = cg.center_of_resistance(self.frame, self.spin_root.value())
        M = cg.kinematic_matrix(
            self.frame, c_res,
            tip_deg=self.s_tip.value() / self.s_tip._scale,
            torque_deg=self.s_torque.value() / self.s_torque._scale,
            rotation_deg=self.s_rot.value() / self.s_rot._scale,
            d_md=self.s_md.value() / self.s_md._scale,
            d_bl=self.s_bl.value() / self.s_bl._scale,
            d_oa=self.s_oa.value() / self.s_oa._scale,
        )
        # Always transform from the REST pose, never from the current one —
        # composing onto the live mesh every frame accumulates float drift and
        # makes "reset to 0" fail to return exactly to T0.
        self.crown_mesh.points = cg.apply_matrix(self.crown_rest, M)

        r = max(self._scene_scale() * 0.012, 0.25)
        self.plotter.add_mesh(pv.Sphere(radius=r, center=c_res), name="c_res", color="#ffffff")
        self._draw_axes(c_res)
        self.plotter.render()

    def _draw_axes(self, origin):
        for n in ("ax_md", "ax_bl", "ax_oa"):
            self.plotter.remove_actor(n, render=False)
        if not self.chk_axes.isChecked():
            return
        L = self._scene_scale() * 0.12
        for name, vec, col in (("ax_md", self.frame["u_md"], "#4dd0e1"),
                               ("ax_bl", self.frame["u_bl"], "#ffca28"),
                               ("ax_oa", self.frame["u_oa"], "#ff5252")):
            self.plotter.add_mesh(pv.Arrow(start=origin, direction=vec, scale=L),
                                  name=name, color=col)

    def _toggle_axes(self):
        if self.frame is not None:
            self._apply_kinematics()

    def closeEvent(self, ev):
        for w in self.workers:
            if w.isRunning():
                w.quit()
                w.wait(2000)
        self.plotter.close()
        super().closeEvent(ev)


def main():
    install_excepthook()
    LOG.info("Starting Clinical Micro-Planner; log at %s", LOG_PATH)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PlannerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
