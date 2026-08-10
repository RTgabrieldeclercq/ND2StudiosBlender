"""Main window for NodeLab v2 (Phase 5).

Chrome (G10): menu bar (File / Run / View), status bar, dockable Palette (G2),
Properties inspector, and Viewer (G4). File actions (G6) round-trip
``*.nd2graph.json`` through the document (headless
:mod:`nodegraph.serialize` + a ``ui`` extras object). Run actions (G7) submit pulls to
the :class:`~nodelab_v2.runner.EngineRunner` — double-click any node (or press F5 on a
selection) to view its output; edits invalidate in-flight results by epoch.

**Maximized canvas (2026-07-27).** The centre normally splits Viewer-over-canvas. The
canvas' top-right ⛶ button (also View → *Maximize node canvas*, ``Ctrl+Space``) hands
the whole centre to the graph and moves the *same* ViewerPanel into the
:class:`~nodelab_v2.minimap.MiniMapOverlay` — a bordered mini-map in the canvas'
top-left corner that previews whatever node you click, live (:data:`FOLLOW_DELAY_MS`
debounce; the previewed card wears an accent spine). ``Esc``, the mini-map's dock
button, or a header double-click puts the Viewer back in the splitter at its old size.
"""
from __future__ import annotations

import uuid
from typing import Optional

import nodegraph.nodes  # noqa: F401 — registers the node catalog into NODES
from nodegraph import hotreload
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QSplitter, QVBoxLayout, QWidget,
)

from nodegraph.iterate import (
    ITERATE_OP, SWEEP_KEY, SWEEP_ROWS_KEY, plan as iterate_plan)
from nodelab_v2 import theme as T
from nodelab_v2.document import GraphDocument
from nodelab_v2.framestrip import compact_list
from nodelab_v2.inspector import InspectorPanel
from nodelab_v2.minimap import MiniMapOverlay
from nodelab_v2.node_item import NodeItem
from nodelab_v2.ops import DOCK_OP, LOAD_OP, PRECISION_UNSET
from nodelab_v2.palette import PalettePanel
from nodelab_v2.picker import Calibration
from nodelab_v2.runner import EngineRunner, ensure_gui_ops
from nodelab_v2.scene import GraphScene, GraphView
from nodelab_v2.spreadsheet import SpreadsheetPanel
from nodelab_v2.viewer import ViewerPanel
from nodelab_v2.welcome import WelcomeCard

#: how long a selection must settle before the mini-map pulls it (ms). Long enough that
#: rubber-banding / arrowing through a chain doesn't queue a pull per node, short enough
#: that a single click feels immediate.
FOLLOW_DELAY_MS = 170

#: pulse period of the status-bar run LED (ms) — the footer twin of a card's status dot.
LED_PULSE_MS = 360

#: how long the node-file watcher waits for saves to settle before reloading (ms). Long
#: enough that an editor's write-truncate-then-write and its temp-file rename land as one
#: reload, short enough that Ctrl+S → the change is live feels immediate.
NODE_RELOAD_DEBOUNCE_MS = 450

#: the status-bar determinate progress bar. Sized to read at a glance without crowding
#: the message text it sits beside (the card's own rail is 2 px — right for a card,
#: too easy to miss while a multi-minute ingest runs).
PROGRESS_BAR_W = 130
PROGRESS_BAR_H = 5     # one of the two stacked bars (frame above, within-frame below)

#: vertical gap between the source cards a MULTI-file load drops (scene px). The cards go
#: in a column rather than on top of each other; this is wide enough that they read as
#: separate sources and leave room to drop a first processing node beside each.
SOURCE_STACK_GAP = 34
PROGRESS_BAR_GAP = 2


def _as_float(v) -> Optional[float]:
    """A progress field coerced to a fraction, or ``None`` if it isn't a number — an
    observer field is only advisory, and a missing one must hide a bar, not raise."""
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None

#: central splitter sizes once there is an image to look at (Viewer-dominant, ~2.5:1).
#: The app LAUNCHES with the Viewer collapsed instead — nothing has been pulled yet, so
#: the blank welcome canvas gets the whole centre; the first pull opens the Viewer.
VIEWER_SPLIT = [860, 340]

def _window_qss() -> str:
    return f"""
QMainWindow {{ background:{T.BG.name()}; }}
QMainWindow::separator {{ background:{T.BORDER.name()}; width:2px; height:2px; }}
QMenuBar {{ background:{T.BODY.name()}; color:{T.INK.name()};
  border-bottom:1px solid {T.BORDER.name()}; padding:2px 4px; }}
QMenuBar::item {{ background:transparent; padding:5px 11px; border-radius:6px;
  margin:0 1px; }}
QMenuBar::item:selected {{ background:{T.PANEL_HI.name()}; }}
QMenuBar::item:pressed {{ background:{T.ACCENT_DIM.name()}; }}
QMenu {{ background:{T.PANEL.name()}; color:{T.INK.name()};
  border:1px solid {T.BORDER.name()}; border-radius:9px; padding:5px; }}
QMenu::item {{ padding:6px 24px 6px 14px; border-radius:6px; }}
QMenu::item:selected {{ background:{T.ACCENT_DIM.name()}; }}
QMenu::separator {{ height:1px; background:{T.BORDER.name()}; margin:5px 10px; }}
QStatusBar {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
  border-top:1px solid {T.BORDER.name()}; }}
QStatusBar::item {{ border:0; }}
QProgressBar {{ background:{T.PANEL.name()}; border:1px solid {T.BORDER.name()};
  border-radius:4px; margin:0 6px; }}
QProgressBar::chunk {{ background:{T.ACCENT.name()}; border-radius:3px; }}
/* the frame/total-frames bar stacked above the within-the-frame one — same shape, the
   orange that means "series position" everywhere else in the UI. */
QProgressBar#frameProgress::chunk {{ background:{T.PROG_FRAME.name()}; }}
QDockWidget {{ color:{T.MUTED.name()}; font-size:10px; font-weight:800;
  titlebar-close-icon:none; titlebar-normal-icon:none; }}
QDockWidget::title {{ background:{T.BODY.name()}; color:{T.MUTED.name()};
  padding:6px 13px; border-bottom:1px solid {T.BORDER.name()}; }}
QSplitter::handle {{ background:{T.BG.name()}; }}
QSplitter::handle:hover {{ background:{T.BORDER.name()}; }}
QSplitter::handle:vertical {{ height:3px; }}
QSplitter::handle:horizontal {{ width:3px; }}
QTabWidget::pane {{ border:0; }}
QTabBar::tab {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
  padding:6px 15px; border:1px solid {T.BORDER.name()}; border-bottom:0;
  border-top-left-radius:7px; border-top-right-radius:7px; margin-right:2px; }}
QTabBar::tab:selected {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
QTabBar::tab:hover {{ background:{T.PANEL_HI.name()}; }}
""" + T.controls_qss()

FILE_FILTER = "nd2graph (*.nd2graph.json);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_gui_ops()
        self.setWindowTitle("NodeLab v2 — nodegraph canvas")
        self.setStyleSheet(_window_qss())

        self.doc = GraphDocument()
        self.scene = GraphScene(self.doc)
        self.view = GraphView(self.scene)
        self.runner = EngineRunner(self.doc)
        self._viewed: Optional[str] = None

        # centre: a DOMINANT Viewer on top over the node canvas, split vertically. With
        # the console removed, the whole window height is split here, so the image gets
        # the room to be seen well (~72% viewer / ~28% canvas) — but only once there IS
        # an image: the app launches with the Viewer collapsed so the blank welcome
        # canvas owns the centre, and the first pull opens the split (_open_viewer).
        self.viewer = ViewerPanel()
        center = QSplitter(Qt.Vertical)
        center.addWidget(self.viewer)
        center.addWidget(self.view)
        center.setStretchFactor(0, 5)     # viewer grows much faster
        center.setStretchFactor(1, 2)     # canvas keeps a usable slice
        center.setCollapsible(0, True)    # …and may be folded away entirely
        center.setCollapsible(1, False)
        center.setSizes([0, 1200])        # launch: all canvas, no image yet
        self._center = center
        self._center_sizes = list(VIEWER_SPLIT)
        self.setCentralWidget(center)

        # the canvas opens EMPTY: this card invites the first node and steps aside as
        # soon as the document has one (File → New brings it back).
        self.welcome = WelcomeCard(self.view)
        self.welcome.load_image_requested.connect(self.file_load_source)
        self.welcome.browse_nodes_requested.connect(self.focus_palette)
        self.welcome.example_requested.connect(self.build_demo)
        self.welcome.op_dropped.connect(self._on_op_dropped)

        # maximized canvas (Ctrl+Space / the ⛶ button): the Viewer moves OUT of the
        # splitter and into this HUD frame over the canvas' top-left corner, where it
        # keeps following whatever node you click.
        self.minimap = MiniMapOverlay(self.view)
        self._maximized = False
        self._follow_pending: Optional[str] = None
        self._follow_timer = QTimer(self)
        self._follow_timer.setSingleShot(True)
        self._follow_timer.setInterval(FOLLOW_DELAY_MS)
        self._follow_timer.timeout.connect(self._follow_pull)

        # docks
        self.palette = PalettePanel(on_add=self._add_at_center)
        self.palette.refresh_requested.connect(self.refresh_node_list)
        pd = QDockWidget("Nodes", self)
        pd.setWidget(self.palette)
        pd.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, pd)

        # right sidebar: Properties + Spreadsheet (tabbed)
        self.inspector = InspectorPanel()
        idock = QDockWidget("Properties", self)
        idock.setWidget(self.inspector)
        idock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, idock)

        self.sheet = SpreadsheetPanel()
        sdock = QDockWidget("Spreadsheet", self)
        sdock.setWidget(self.sheet)
        sdock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, sdock)
        self.tabifyDockWidget(idock, sdock)
        idock.raise_()

        # Console removed — the reclaimed bottom-dock space goes to the central splitter
        # (Viewer + node canvas). Run/error messages surface in the status bar, and a
        # failed pull shows its trace in the Viewer (status line + tooltip).

        # Live node reload. `prime()` must run here — after the catalog and the GUI ops are
        # imported, before the user can edit anything — because it baselines the source
        # digests against the code this process actually executed. Baseline it later and an
        # edit made in the first minute would look like the status quo and never reload.
        self._watcher = None
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.timeout.connect(self._autoreload_tick)
        hotreload.prime()

        self._build_menus()
        # `setChecked` in _build_menus precedes the `toggled` connection (so building the
        # menu cannot fire it), which means the ticked default has to be honored by hand.
        self._set_autoreload(self._autoreload_act.isChecked())
        # run LED (the same language as the cards' status dot): a bead in the status bar
        # that pulses while the engine works. It is a PERMANENT widget — a temporary
        # showMessage() hides normal status-bar widgets, and this must stay visible.
        self._led = QLabel("●")
        self._led_state = "idle"
        self._led_on = True
        self._led_timer = QTimer(self)
        self._led_timer.setInterval(LED_PULSE_MS)
        self._led_timer.timeout.connect(self._led_tick)
        self.statusBar().addPermanentWidget(self._led)
        # Determinate bar for long work that reports a real fraction — above all the
        # one-time ingest of a file, which can run for many minutes and used to show
        # nothing but a 'queued' card. PERMANENT for the same reason as the LED, and
        # hidden unless a fraction is actually in hand, so it never fakes progress.
        # Two of them, stacked the same way the node cards stack their rails: frames above,
        # work-inside-the-frame below. They hide independently — a compute with no frame
        # axis shows only the lower bar, which is exactly what the single bar used to be.
        def _bar(name: str = "") -> QProgressBar:
            b = QProgressBar()
            if name:
                b.setObjectName(name)
            b.setRange(0, 1000)               # per-mille: smooth on a multi-minute read
            b.setTextVisible(False)
            b.setFixedSize(PROGRESS_BAR_W, PROGRESS_BAR_H)
            b.hide()
            return b

        self._prog_frame = _bar("frameProgress")
        self._prog = _bar()
        self._prog_box = QWidget()
        _pv = QVBoxLayout(self._prog_box)
        _pv.setContentsMargins(0, 0, 0, 0)
        _pv.setSpacing(PROGRESS_BAR_GAP)
        _pv.addWidget(self._prog_frame)
        _pv.addWidget(self._prog)
        self._prog_box.setFixedWidth(PROGRESS_BAR_W)
        self.statusBar().addPermanentWidget(self._prog_box)
        # Solo-frame chip. PERMANENT and always visible while the scope is on: a reduced
        # scope that is easy to forget is a way to misread results, so it gets a standing
        # indicator rather than a message that the next status update wipes.
        self._solo_chip = QLabel("")
        self._solo_chip.setToolTip(self._solo_act.toolTip())
        self._solo_chip.hide()
        self.statusBar().addPermanentWidget(self._solo_chip)
        self._set_led("idle")
        self.statusBar().showMessage("ready")

        # signals
        self.scene.selectionChanged.connect(self._on_selection)
        self.scene.node_activated.connect(self.pull_node)
        self.view.op_dropped.connect(self._on_op_dropped)
        self.doc.on_change(self._on_doc_changed)
        self.doc.on_change(self.inspector.refresh_derived)   # G8 live ƒmd re-seed
        self.scene.pull_requested.connect(self.pull_node)
        self.scene.nodes_deleted.connect(self._on_nodes_deleted)
        self.runner.started.connect(
            lambda nid: (self.statusBar().showMessage(f"pulling {nid}…"),
                         self.viewer.show_running(nid),
                         self._set_led("busy"),
                         self.minimap.set_state("busy")))
        self.runner.finished.connect(self._on_run_finished)
        self.runner.plane_ready.connect(self._on_plane_ready)
        self.runner.failed.connect(self._on_run_failed)
        # per-source ingest (V2.21): its own lane, so its own signals — several run at
        # once and none of them is "the run".
        self.runner.ingest_started.connect(self._on_ingest_started)
        self.runner.ingest_finished.connect(self._on_ingest_finished)
        self.scene.ingest_requested.connect(self.ingest_source)
        #: the source card to open in the Viewer when its ingest lands — the LAST one
        #: double-clicked, so files finishing minutes apart don't fight over the pane.
        self._view_after_ingest: Optional[str] = None
        # per-node progress: the cards show queued → running → done/cached/error, and the
        # status bar counts the nodes that actually ran.
        self.runner.plan.connect(self._on_run_plan)
        self.runner.node_progress.connect(self._on_node_progress)
        self.viewer.request_changed.connect(self._on_view_request)
        self.viewer.selection_changed.connect(self._on_frame_selection)
        self.viewer.iteration_changed.connect(self._on_iteration_changed)
        # Interactive parameter picking (V2.16). Both surfaces that can ARM a pick — the
        # inspector's Pick button and a card's ◎ glyph — route here rather than talking to
        # the viewer directly, because arming needs the node's calibration and committing
        # needs the document, and only the window has both.
        self.inspector.pick_requested.connect(self._arm_pick)
        self.scene.pick_requested.connect(self._arm_pick)
        # Dock (V2.18): both surfaces that can start a bake — the inspector's buttons and
        # the card's context menu — route to one handler, for the same reason picks do.
        # Only the window has the runner (to run it), the document (to record it) and the
        # dormant set (to release what the bake made redundant).
        self.inspector.dock_action.connect(self._on_dock_action)
        self.scene.dock_action.connect(self._on_dock_action)
        self.inspector.iterate_action.connect(self._on_iterate_action)
        self.inspector.reload_requested.connect(self.reload_node_type)
        self.runner.baked.connect(self._on_baked)
        self.viewer.pick_committed.connect(self._on_pick_committed)
        self.viewer.pick_armed.connect(self._on_pick_armed)
        # The hover readout wants the UNPROCESSED file pixel beside the viewed node's. The
        # viewer holds no provider and the runner holds no notion of "the viewed node", so
        # the window — which has both — hands one over as a plain callback.
        self.viewer.raw_plane_cb = self.runner.raw_plane
        # viewport detail-on-demand: the panel asks (debounced, on pan/zoom), the runner
        # reads the rect off the GUI thread and answers on `detail_ready`.
        self.viewer.detail_cb = self._request_detail
        self.runner.detail_ready.connect(self.viewer.on_detail_ready)
        self.view.maximize_toggled.connect(self.set_maximized)
        self.minimap.restore_requested.connect(lambda: self.set_maximized(False))
        self.doc.on_change(self._sync_welcome)

        self._sync_welcome()          # open on a blank, welcoming canvas

    # ── chrome (G10) ─────────────────────────────────────────────────────────
    def _build_menus(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        for text, seq, fn in (
                ("&New", QKeySequence.New, self.file_new),
                ("&Open…", QKeySequence.Open, self.file_open),
                ("&Save", QKeySequence.Save, self.file_save),
                ("Save &As…", QKeySequence.SaveAs, self.file_save_as)):
            act = QAction(text, self)
            act.setShortcut(seq)
            act.triggered.connect(fn)
            m_file.addAction(act)

        m_file.addSeparator()
        load = QAction("&Load ND2/ND3/TIFF file…", self)
        load.setShortcut("Ctrl+L")
        load.triggered.connect(self.file_load_source)
        m_file.addAction(load)

        m_file.addSeparator()
        exp = QAction("&Export table…", self)
        exp.setShortcut("Ctrl+E")
        exp.triggered.connect(self.file_export)
        m_file.addAction(exp)

        # ── Edit: the discoverable home of delete (the canvas also has the hover ✕ badge
        # and a right-click menu). The shortcuts are scoped to the CANVAS
        # (WidgetWithChildrenShortcut on the view) so pressing Del while editing a value
        # in the inspector still edits text instead of deleting the node behind it.
        m_edit = self.menuBar().addMenu("&Edit")
        self._del_act = QAction("&Delete selected", self)
        self._del_act.setShortcuts([QKeySequence(Qt.Key_Delete),
                                    QKeySequence(Qt.Key_Backspace)])
        self._del_act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self._del_act.setToolTip("Delete the selected nodes / wires / frames")
        self._del_act.triggered.connect(self.delete_selection)
        self.view.addAction(self._del_act)
        m_edit.addAction(self._del_act)
        self._dissolve_act = QAction("Dis&solve (delete, keep the chain)", self)
        self._dissolve_act.setShortcut("Ctrl+X")
        self._dissolve_act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self._dissolve_act.setToolTip(
            "Delete the selected node(s) and reconnect each one's input to whatever it fed")
        self._dissolve_act.triggered.connect(self.scene.dissolve_selection)
        self.view.addAction(self._dissolve_act)
        m_edit.addAction(self._dissolve_act)
        m_edit.addSeparator()
        sel_all = QAction("Select &all nodes", self)
        sel_all.setShortcut("Ctrl+A")
        sel_all.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        sel_all.triggered.connect(self.select_all_nodes)
        self.view.addAction(sel_all)
        m_edit.addAction(sel_all)
        m_edit.aboutToShow.connect(self._sync_edit_actions)

        m_run = self.menuBar().addMenu("&Run")
        pull = QAction("&Pull selected node", self)
        pull.setShortcut("F5")
        pull.triggered.connect(self.pull_selected)
        m_run.addAction(pull)
        again = QAction("Pull &viewed again", self)
        again.setShortcut("Shift+F5")
        again.triggered.connect(lambda: self._viewed and self.pull_node(self._viewed))
        m_run.addAction(again)
        ingest = QAction("&Ingest all source files", self)
        ingest.setShortcut("Ctrl+I")
        ingest.setToolTip(
            "Convert every loaded ND2/TIFF to its on-disk store now, several at a time, "
            "instead of paying for each one on the first pull of its chain.\n\n"
            "Double-clicking a single source card does the same for that one file. The "
            "app stays usable throughout and each card shows its own progress.")
        ingest.triggered.connect(self.ingest_all_sources)
        m_run.addAction(ingest)
        m_run.addSeparator()
        self._bake_act = QAction("&Bake selected dock…", self)
        self._bake_act.setShortcut("F6")
        self._bake_act.setToolTip(
            "Freeze everything above the selected Dock node to disk, then serve it from "
            "there. The chain behind it greys out, stops being evaluated, and is released "
            "from memory — which is what makes it affordable to keep adding nodes on a "
            "large file.")
        self._bake_act.triggered.connect(self._bake_selected)
        m_run.addAction(self._bake_act)
        m_run.aboutToShow.connect(self._sync_run_actions)
        m_run.addSeparator()
        self._solo_act = QAction("&Troubleshoot: picked frames only", self)
        self._solo_act.setCheckable(True)
        self._solo_act.setShortcut("F9")
        self._solo_act.setToolTip(
            "Scope every pull to what the Viewer's M/T/Z strips pick — or, with nothing "
            "picked, to the ONE frame the cursor is on — instead of the whole series. The "
            "fast way to tune parameters on a long acquisition: frames you have already "
            "run come back instantly from the memo.\n\n"
            "Ctrl+click (or ctrl+drag) the T strip to pick several frames. That is what "
            "gives a temporal node something real to work on — a tracker linking three "
            "picked timepoints instead of nothing. Picks on Z cut every scoped frame down "
            "to those planes (unpicked = the whole volume, since a 3D node needs one), "
            "and the axes combine: 2 positions × 3 timepoints × 5 planes.\n\n"
            "Mind that skipped indices are simply absent, so links, rates and 3D "
            "measurements span the picked neighbours rather than the acquired ones. Turn "
            "it off for real results.")
        self._solo_act.toggled.connect(self.set_solo_frame)
        m_run.addAction(self._solo_act)
        clear_picks = QAction("Clear picked &frames", self)
        clear_picks.setToolTip("Drop the M/T/Z picks — the scope goes back to the single "
                               "frame the Viewer's cursor is on, whole volume")
        clear_picks.triggered.connect(self.clear_frame_picks)
        m_run.addAction(clear_picks)

        m_run.addSeparator()
        reload_act = QAction("&Reload node code", self)
        reload_act.setShortcut("Ctrl+R")
        reload_act.setToolTip(
            "Re-read the node definitions and math kernels from disk and run the new code "
            "on the next pull — without closing this window, reloading the graph, or "
            "re-ingesting the file.\n\n"
            "Edit a node, save, press this. Nodes whose code changed recompute; everything "
            "else keeps its cached results, so only the part you edited is paid for again. "
            "A file with a syntax error is reported and ignored, leaving the session on the "
            "code it already had.")
        reload_act.triggered.connect(lambda: self.reload_node_code())
        m_run.addAction(reload_act)
        force_act = QAction("Reload node code (&force all)", self)
        force_act.setToolTip(
            "Reload every node module and kernel whether or not its file looks changed — "
            "for an edit this cannot detect, such as a kernel modified between the moment "
            "it was first imported and now. Recomputes more than the plain reload.")
        force_act.triggered.connect(lambda: self.reload_node_code(force=True))
        m_run.addAction(force_act)
        self._autoreload_act = QAction("&Auto-reload on file change", self)
        self._autoreload_act.setCheckable(True)
        self._autoreload_act.setChecked(True)
        self._autoreload_act.setToolTip(
            "Watch the node and kernel files and reload as soon as one is saved, so the "
            "canvas is always on the code on disk.\n\n"
            "A reload is held back while a pull is running (swapping a compute out from "
            "under a working thread is not safe) and applied when it finishes. Turn this "
            "off if you would rather choose the moment yourself — mid-edit saves are "
            "harmless, but a reload does re-key the memo for the nodes it touched.")
        self._autoreload_act.toggled.connect(self._set_autoreload)
        m_run.addAction(self._autoreload_act)

        m_graph = self.menuBar().addMenu("&Graph")
        wrap = QAction("&Wrap selection in Repeat zone…", self)
        wrap.triggered.connect(self.wrap_repeat)
        m_graph.addAction(wrap)
        frame = QAction("&Frame selection…", self)
        frame.setShortcut("Ctrl+J")
        frame.triggered.connect(self.frame_selection)
        m_graph.addAction(frame)
        m_graph.addSeparator()
        grp = QAction("&Group selection…", self)
        grp.setShortcut("Ctrl+G")
        grp.triggered.connect(self.group_selection)
        m_graph.addAction(grp)
        ungrp = QAction("&Ungroup", self)
        ungrp.setShortcut("Ctrl+Shift+G")
        ungrp.triggered.connect(self.ungroup_selection)
        m_graph.addAction(ungrp)

        m_view = self.menuBar().addMenu("&View")
        fit = QAction("&Fit graph", self)
        fit.setShortcut("Home")
        fit.triggered.connect(self.view.fit_all)
        m_view.addAction(fit)
        self._max_act = QAction("&Maximize node canvas", self)
        self._max_act.setCheckable(True)
        self._max_act.setShortcut("Ctrl+Space")
        self._max_act.setToolTip("Give the whole centre to the graph; the Viewer becomes "
                                 "a mini-map in the canvas' top-left corner")
        self._max_act.toggled.connect(self.set_maximized)
        m_view.addAction(self._max_act)
        self._follow_act = QAction("&Preview clicked node", self)
        self._follow_act.setCheckable(True)
        self._follow_act.setToolTip("Pull and show a node as soon as you click it "
                                    "(always on while the canvas is maximized)")
        m_view.addAction(self._follow_act)
        m_view.addSeparator()
        ovl = QAction("&Overlays…", self)
        ovl.setShortcut("Ctrl+Shift+O")
        ovl.setToolTip("Configure the Point / Label / Track overlays — size, opacity, "
                       "look, colour — and save the look as a default")
        ovl.triggered.connect(self.viewer.open_overlay_dialog)
        m_view.addAction(ovl)
        m_view.addSeparator()
        self._light = QAction("&Light theme", self)
        self._light.setCheckable(True)
        self._light.toggled.connect(
            lambda on: self.set_theme("light" if on else "dark"))
        m_view.addAction(self._light)

        m_help = self.menuBar().addMenu("&Help")
        cap = QAction("&Capabilities…", self)
        cap.triggered.connect(self._show_capabilities)
        m_help.addAction(cap)

    # ── run LED ──────────────────────────────────────────────────────────────
    def _set_led(self, state: str) -> None:
        """``idle`` | ``busy`` | ``error`` — busy pulses, the others sit steady."""
        self._led_state = state
        self._led.setToolTip({"idle": "engine idle", "busy": "engine working",
                              "error": "last pull failed"}[state])
        if state == "busy":
            if not self._led_timer.isActive():
                self._led_on = True
                self._led_timer.start()
        else:
            self._led_timer.stop()
            self._led_on = True
        self._paint_led()

    def _paint_led(self) -> None:
        col = {"idle": T.MUTED, "busy": T.ACCENT, "error": T.ERROR}[self._led_state]
        # QSS takes rgba() for the off-beat; QColor.name() would drop the alpha
        rgba = f"rgba({col.red()},{col.green()},{col.blue()},{1.0 if self._led_on else 0.3})"
        self._led.setStyleSheet(f"color:{rgba};font-size:13px;padding:0 6px 0 2px;")

    def _led_tick(self) -> None:
        self._led_on = not self._led_on
        self._paint_led()

    # ── edit ─────────────────────────────────────────────────────────────────
    def _sync_edit_actions(self) -> None:
        """Grey out the delete entries when there is nothing selected, and name what
        they would act on (so the menu itself teaches the shortcut)."""
        nodes = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        n_any = len(self.scene.selectedItems())
        self._del_act.setEnabled(bool(n_any))
        self._del_act.setText("&Delete selected" if n_any != 1 else
                              f"&Delete {self._sel_label()}")
        self._dissolve_act.setEnabled(bool(nodes))

    def _sel_label(self) -> str:
        for it in self.scene.selectedItems():
            if isinstance(it, NodeItem):
                spec = it.rec.spec()
                return f"'{spec.label if spec else it.op_key}'"
        return "selection"

    def delete_selection(self) -> None:
        """Edit → Delete: remove the selected nodes/wires/frames. Reports what went, so a
        deletion is never silent (the canvas is busy — a card vanishing off-screen would
        otherwise be invisible)."""
        n_nodes = sum(1 for i in self.scene.selectedItems() if isinstance(i, NodeItem))
        n_all = len(self.scene.selectedItems())
        if not n_all:
            self.statusBar().showMessage(
                "nothing selected — click a node (or wire) first, then Del")
            return
        self.scene.delete_selection()
        self.statusBar().showMessage(
            f"deleted {n_nodes} node(s)" if n_nodes else f"deleted {n_all} item(s)")

    def select_all_nodes(self) -> None:
        for item in self.scene.node_items.values():
            item.setSelected(True)

    def _on_nodes_deleted(self, node_ids) -> None:
        ids = list(node_ids)
        self.statusBar().showMessage(
            f"deleted {', '.join(ids)}" if len(ids) <= 4 else f"deleted {len(ids)} nodes")

    def _show_capabilities(self) -> None:
        """What this editor covers. (Was the Phase-7 "v2 vs legacy v1" note; NodeLab v1
        was removed 2026-07-29 — see `V2.05_phase7_capability_matrix.md` §6.)"""
        QMessageBox.information(
            self, "NodeLab — capabilities",
            "NodeLab runs on the nodegraph engine: a lazy-pull, per-tile streaming, "
            "two-hash-memoized graph with a metadata-intelligent node catalog.\n\n"
            "Covered:\n"
            "  • Enhancement / denoise suite + metadata-derived deconvolution\n"
            "  • Segmentation — ONE node, the algorithm as a method: threshold+CCL, "
            "distance-transform watershed, StarDist (CNN), CellSAM (foundation model)\n"
            "  • Thresholding (global / local / multi-Otsu / histogram methods), "
            "connected components, EDT, boundaries\n"
            "  • Detection: spots, particles\n"
            "  • Correlation: DIC (pyALDIC) and DVC (ALDVC) + cumulative "
            "accumulation and field rasterization\n"
            "  • Structure: point clustering, tessellation / mesh, measurements, "
            "domain transfer\n"
            "  • Tracking: frame linking + the 5-method tracker, with track overlays\n"
            "  • The 2D/3D lever, zones, groups, reroutes and frames\n"
            "  • A live multi-channel Viewer, Spreadsheet, and CSV / Parquet / Arrow "
            "export\n\n"
            "Retired with NodeLab v1 (declared obsolete): cell-tracker spatial maps, "
            "DIC mesh refinement, the bleach / blob-subtract / spatial-flatness / "
            "temporal-fold enhancements, Cellpose nuclei, the interactive workflow-IO "
            "nodes (3D mask drawing, review, exclude, pause, prism, save data, "
            "checkpoints) and if/else.\n\n"
            "See CodeLog/ClaudesPlan/V2.05_phase7_capability_matrix.md.")

    def set_theme(self, mode: str) -> None:
        """Rebind the palette (G9) and re-apply it everywhere: node cards repaint from
        the tokens, QSS panels restyle, the canvas background updates."""
        T.apply(mode)
        self.setStyleSheet(_window_qss())
        for panel in (self.palette, self.inspector, self.viewer, self.sheet,
                      self.minimap, self.welcome, self.view):
            panel.restyle()
        self._paint_led()          # the LED colors come from the tokens, not from QSS
        self._sync_solo_chip()     # ditto for the solo chip's amber
        self.view.setBackgroundBrush(T.BG)
        self.scene.update()
        self.view.viewport().update()

    # ── maximized canvas + mini-map (2026-07-27) ──────────────────────────────
    def set_maximized(self, on: bool) -> None:
        """Toggle the maximized node canvas.

        **On** — the Viewer leaves the splitter (so the canvas owns the whole centre)
        and is re-homed into the :class:`~nodelab_v2.minimap.MiniMapOverlay` pinned to
        the canvas' top-left corner, trimmed to its compact layout, with click-to-preview
        forced on so the mini-map follows the node you're working on.
        **Off** — the Viewer goes back into the splitter at its previous size and
        click-to-preview returns to whatever the user had chosen.

        The same live ViewerPanel widget is moved (never a second copy), so channels,
        LUT, playback and overlays carry straight across."""
        on = bool(on)
        if on == self._maximized:
            return
        self._maximized = on
        if on:
            self._center_sizes = self._center.sizes()
            self.viewer.set_compact(True)
            self.minimap.attach(self.viewer)
            self.minimap.reposition()
            self.minimap.show()
            self.minimap.raise_()
            self._follow_before_max = self._follow_act.isChecked()
            self._follow_act.setChecked(True)
            # the splitter re-lays out on the next turn — re-anchor once the canvas has
            # actually grown into the freed space
            QTimer.singleShot(0, self.minimap.reposition)
        else:
            self.minimap.detach()
            self.minimap.hide()
            self.viewer.set_compact(False)
            self._center.insertWidget(0, self.viewer)
            self.viewer.show()
            back = self._center_sizes or list(VIEWER_SPLIT)
            self._center.setSizes(back if back[0] >= 80 else list(VIEWER_SPLIT))
            self._follow_act.setChecked(getattr(self, "_follow_before_max", False))
        # keep both entry points (canvas button + View menu) in sync, no signal loop
        self.view.set_maximized(on)
        self._max_act.blockSignals(True)
        self._max_act.setChecked(on)
        self._max_act.blockSignals(False)
        self._sync_minimap_title()
        self.statusBar().showMessage(
            "canvas maximized — click any node to preview it in the mini-map (Esc to "
            "dock the Viewer back)" if on else "Viewer docked")

    def _open_viewer(self) -> None:
        """Make sure the Viewer pane is actually on screen before showing a result — it
        launches folded away (blank canvas, nothing pulled) and a user can fold it back
        by dragging the splitter. Only acts when it is folded; a pane the user has
        already sized is left exactly as it is."""
        if self._maximized:
            self._center_sizes = list(VIEWER_SPLIT)   # applies when the Viewer docks back
            return
        if self._center.sizes()[0] < 80:
            self._center.setSizes(list(VIEWER_SPLIT))

    def _sync_minimap_title(self) -> None:
        """The mini-map header names what it is showing (id · node label)."""
        nid = self._viewed
        rec = self.doc.nodes.get(nid) if nid else None
        if rec is None:
            self.minimap.set_title("viewer · click a node")
            return
        spec = rec.spec()
        self.minimap.set_title(f"{nid} · {spec.label if spec else rec.op_key}")

    def _follow_pull(self) -> None:
        """Debounced click-to-preview: pull the node the selection settled on. Never
        starts an ingest — see :meth:`pull_node`."""
        nid, self._follow_pending = self._follow_pending, None
        if nid and nid in self.doc.nodes and nid != self._viewed:
            self.pull_node(nid, allow_ingest=False)

    # ── document plumbing ─────────────────────────────────────────────────────
    def _on_doc_changed(self) -> None:
        self.runner.invalidate()          # an edit supersedes in-flight results
        if self._viewed is not None and self._viewed not in self.doc.nodes:
            self._viewed = None           # the previewed node was deleted/reloaded away
            self.scene.set_viewed(None)
            self.minimap.set_state("idle")
            self._sync_minimap_title()
        self._sync_solo(self._viewed)     # a rewired source changes the frame count
        name = self.doc.path or "untitled"
        self.statusBar().showMessage(f"{name} — rev {self.doc.revision}")

    def _on_selection(self) -> None:
        try:
            sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        except RuntimeError:
            return          # scene torn down (app closing) — the C++ object is gone
        self.inspector.set_node(sel[0] if sel else None)
        # click-to-preview (always on while maximized): debounce, so dragging a marquee
        # across a chain queues ONE pull — the node the selection settled on.
        if sel and self._follow_act.isChecked():
            self._follow_pending = sel[0].node_id
            self._follow_timer.start()

    # ── interactive parameter picking (V2.16) ────────────────────────────────
    def _arm_pick(self, req) -> None:
        """Arm a pick on the viewer, with the target node's own calibration.

        The calibration is the node's PROPAGATED envelope, not the source file's: a Resample
        or a Crop upstream changes what one pixel is worth, and a radius picked on the image
        has to be expressed in the microns *this* node will convert back to pixels. That is
        precisely what ``propagate_meta`` already tracks, so the pick inherits it for free."""
        if req.surface != "instant" and not self.viewer.has_image():
            self.statusBar().showMessage(
                "Pull a node first — picking aims at the image, and there is nothing "
                "displayed yet")
            return
        md = {}
        try:
            md = dict(self.doc.env(req.node_id).metadata or {})
        except Exception:                     # noqa: BLE001 — an un-propagated node
            md = {}
        self._open_viewer()
        self.viewer.arm_pick(req, Calibration.from_metadata(md))

    def _on_pick_armed(self, on: bool) -> None:
        if on:
            self.statusBar().showMessage("Picking — Esc cancels, Enter applies")
        else:
            self.statusBar().clearMessage()

    def _on_pick_committed(self, node_id: str, values: dict) -> None:
        """Write a finished pick into the document.

        Deliberately the same two lines the inspector's ``_set_param`` runs — the value plus
        the sticky pin — so a picked param is in every later respect a hand-entered one: it
        shows as pinned, it can be unpinned back to its metadata-derived default, it
        serializes identically and it keys the memo identically. A pick is a nicer way to
        arrive at a number, not a different kind of number."""
        rec = self.doc.nodes.get(node_id)
        if rec is None or not values:
            return
        for name, value in values.items():
            rec.params[name] = value
        rec.set_locked(rec.locked | set(values))
        self.doc.touch()
        item = self.scene.node_items.get(node_id)
        if item is not None:
            # the inspector rebuilds off the item's `changed`, so a pick made from the
            # CANVAS still refreshes the panel (and vice versa)
            item.refresh()
            item.changed.emit(item)
        what = ", ".join(f"{k} = {v}" for k, v in values.items())
        self.statusBar().showMessage(f"{node_id}: {what}")

    def _add_at_center(self, op_key: str) -> None:
        c = self.view.mapToScene(self.view.viewport().rect().center())
        self.doc.add_node(op_key, x=c.x() - 100, y=c.y() - 40)

    def wrap_repeat(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            QMessageBox.information(self, "Wrap in Repeat zone",
                                    "Select a linear sub-chain (one dataset input, one "
                                    "output) to wrap.")
            return
        n, ok = QInputDialog.getInt(self, "Repeat zone", "Iterations:", 3, 1, 100000)
        if not ok:
            return
        try:
            self.doc.wrap_repeat_zone(sel, iterations=n)
        except ValueError as exc:
            QMessageBox.warning(self, "Can't wrap", str(exc))
            return
        self.statusBar().showMessage(f"wrapped {len(sel)} node(s) in a Repeat×{n} zone")

    def group_selection(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            QMessageBox.information(
                self, "Group selection",
                "Select a linear sub-chain (one dataset input, one output) to collapse "
                "into a reusable group node. Sources must stay outside the group.")
            return
        name, ok = QInputDialog.getText(self, "Make group", "Group name:", text="Group")
        if not ok:
            return
        try:
            inst = self.doc.make_group(sel, name or "Group")
        except ValueError as exc:
            QMessageBox.warning(self, "Can't group", str(exc))
            return
        item = self.scene.node_items.get(inst)
        if item is not None:
            self.scene.clearSelection()
            item.setSelected(True)
        self.statusBar().showMessage(f"grouped {len(sel)} node(s) into '{name or 'Group'}'")

    def ungroup_selection(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems()
               if isinstance(i, NodeItem) and i._is_group]
        if not sel:
            QMessageBox.information(self, "Ungroup",
                                   "Select a group node to inline its contents back into "
                                   "the canvas.")
            return
        n = sum(1 for nid in sel if self.doc.ungroup(nid))
        self.statusBar().showMessage(f"ungrouped {n} group(s)")

    def frame_selection(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            QMessageBox.information(self, "Frame selection",
                                    "Select one or more nodes to enclose in a frame.")
            return
        title, ok = QInputDialog.getText(self, "Frame", "Frame label:", text="Frame")
        if not ok:
            return
        self.doc.add_frame(title or "Frame", sel)
        self.statusBar().showMessage(f"framed {len(sel)} node(s)")

    def _on_op_dropped(self, op_key: str, pos: QPointF) -> None:
        # splice-on-wire: if the node is dropped on a link, insert it into that wire
        e = self.scene.edge_at(pos)
        edge_tuple = e.model_edge if e is not None else None
        rec = self.doc.add_node(op_key, x=pos.x() - 20, y=pos.y() - 20)
        if edge_tuple is not None:
            self.scene.splice_onto(rec.id, edge_tuple)

    # ── run (G7) ─────────────────────────────────────────────────────────────
    def pull_selected(self) -> None:
        sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if sel:
            self.pull_node(sel[0].node_id)

    # ── per-source ingest (V2.21) ─────────────────────────────────────────────
    def _sync_ingest(self) -> None:
        """Tell the canvas which source cards are mid-ingest, so a pull's card reset
        leaves their rails alone (:meth:`nodelab_v2.scene.GraphScene.set_ingesting`)."""
        self.scene.set_ingesting(self.runner.ingesting())

    def _idle_led(self) -> None:
        """Back to idle — unless files are still ingesting, which is real work the footer
        must keep claiming even though the pull that finished is not it."""
        self._set_led("busy" if self.runner.ingesting() else "idle")

    def _source_label(self, node_id: str) -> str:
        import os
        rec = self.doc.nodes.get(node_id)
        path = str((rec.params.get("path") if rec else "") or "")
        return os.path.basename(path.strip().strip('"').strip("'")) or node_id

    def ingest_source(self, node_id: str) -> str:
        """Start (or report) one source card's ingest and say so in the status bar."""
        state = self.runner.ingest_source(node_id)
        self._sync_ingest()
        name = self._source_label(node_id)
        running = len(self.runner.ingesting())
        if state in ("started", "joined"):
            self._set_led("busy")
            self.statusBar().showMessage(
                f"ingesting {name}"
                + (f" — {running} file(s) in flight" if running > 1 else "")
                + "  ·  the app stays usable; other cards can start too")
        elif state == "ready":
            self.statusBar().showMessage(f"{name} is already ingested — nothing to do")
        elif state == "running":
            self.statusBar().showMessage(f"{name} is already ingesting")
        elif state == "missing":
            self.statusBar().showMessage(f"{name}: no such file — fix the card's path")
        elif state == "synthetic":
            self.statusBar().showMessage(
                f"{node_id} has no file (the synthetic demo source) — nothing to ingest")
        return state

    def ingest_all_sources(self) -> None:
        """Run → *Ingest all source files*: put every loaded ND2/TIFF on disk at once.

        The companion to the multi-file loader — pick five files, ingest all five, come back
        to a graph where every pull is immediate. They run :func:`
        ~nodelab_v2.runner.ingest_workers` at a time; the rest queue."""
        states = self.runner.ingest_all_sources()
        self._sync_ingest()
        started = [n for n, s in states.items() if s in ("started", "joined")]
        ready = [n for n, s in states.items() if s == "ready"]
        bad = [n for n, s in states.items() if s == "missing"]
        if not states:
            self.statusBar().showMessage(
                "no source nodes — File → Load ND2/ND3/TIFF file… first")
            return
        if started:
            self._set_led("busy")
        parts = []
        if started:
            parts.append(f"ingesting {len(started)} file(s)")
        if ready:
            parts.append(f"{len(ready)} already on disk")
        if bad:
            parts.append(f"{len(bad)} missing")
        self.statusBar().showMessage(" · ".join(parts) or "nothing to ingest")

    def _on_ingest_started(self, node_id: str) -> None:
        self._sync_ingest()
        # the card goes 'running' straight away — the job may sit in the pool's queue for
        # a while behind the other files, and a card that shows nothing reads as ignored.
        self.scene.on_node_progress("start", node_id, {})

    def _on_ingest_finished(self, node_id: str, seconds: float, err) -> None:
        self._sync_ingest()
        name = self._source_label(node_id)
        if err:
            self.scene.on_node_progress("error", node_id, {})
            self.viewer.show_error(node_id, str(err))
            self._set_led("error")
            last = [ln for ln in str(err).strip().splitlines() if ln.strip()]
            self.statusBar().showMessage(
                f"{name} FAILED to ingest — {last[-1] if last else 'see Viewer'}")
            if self._view_after_ingest == node_id:
                self._view_after_ingest = None
            return
        self.scene.on_node_progress("done", node_id, {"seconds": seconds})
        left = len(self.runner.ingesting())
        self.statusBar().showMessage(
            f"{name} ingested in {seconds:.1f}s"
            + (f" — {left} file(s) still going" if left else ""))
        if not left and not self.runner.busy:
            self._set_progress(None)
        self._idle_led()
        if self._view_after_ingest == node_id:
            self._view_after_ingest = None
            self.pull_node(node_id)          # now immediate: the store is warm

    def pull_node(self, node_id: str, *, allow_ingest: bool = True) -> None:
        """Double-click / F5 on a card: compute it and show it in the Viewer.

        On a **source** card whose file has not been ingested yet, this starts that file's
        ingest instead (:meth:`~nodelab_v2.runner.EngineRunner.ingest_source`) and views it
        once it lands. The difference is which lane the work runs in: a pull is one at a
        time and latest-wins, so ingesting through it meant a second file could not start
        until the first had finished — and on a 40-minute ND2 that is the whole session.
        Ingests run several at once on their own pool, so you can double-click every source
        card you just loaded and let them all go.

        Only the card you activated **last** is opened in the Viewer when its file lands;
        the others simply finish. Otherwise five files finishing minutes apart would each
        yank the Viewer to themselves.

        ``allow_ingest=False`` makes a cold source a no-op instead. That is for
        :meth:`_follow_pull` — click-to-preview, which is *always on* while the canvas is
        maximized: a single click landing on a source card must not commit the machine to
        writing a 40 GB store. Asking for it is a double-click, or Run → Ingest."""
        state = self.runner.source_state(node_id)
        if state in ("cold", "running"):
            if not allow_ingest:
                if state == "cold":
                    self.statusBar().showMessage(
                        f"{self._source_label(node_id)} is not ingested yet — "
                        f"double-click the card (or Ctrl+I) to start it")
                return
            self._view_after_ingest = node_id
            self.ingest_source(node_id)
            return
        self._viewed = node_id
        self.scene.set_viewed(node_id)       # accent spine on the card being shown
        self._sync_minimap_title()
        # BEFORE reading the cursor: a new node may sit on a different source chain, so the
        # frame chooser's extent has to be right before the coords it produces are sent.
        self._sync_solo(node_id)
        self.runner.set_frame_selection(*self.viewer.frame_selection())
        self.runner.pull(node_id, self.viewer.coords(), self.viewer.channels())

    # ── dock / bake (V2.18) ───────────────────────────────────────────────────
    def _selected_dock(self) -> Optional[str]:
        """The selected Dock node's id — or, when exactly one dock is in the graph and
        nothing relevant is selected, that one. Bake is a per-graph action far more often
        than a per-selection one, and demanding a click first would be pedantry."""
        sel = [i.node_id for i in self.scene.selectedItems()
               if isinstance(i, NodeItem) and i.op_key == DOCK_OP]
        if sel:
            return sel[0]
        docks = self.doc.dock_nodes()
        return docks[0] if len(docks) == 1 else None

    def _sync_run_actions(self) -> None:
        nid = self._selected_dock()
        self._bake_act.setEnabled(nid is not None)
        self._bake_act.setText(f"&Bake dock “{nid}”…" if nid else "&Bake selected dock…")

    # ── live node reload (nodegraph.hotreload) ────────────────────────────────
    def _set_autoreload(self, on: bool) -> None:
        """Start/stop watching the node + kernel files.

        The watcher is created on demand and torn down when switched off rather than merely
        ignored, so an unwatched session holds no OS handles on the source tree."""
        if not on:
            if self._watcher is not None:
                self._watcher.deleteLater()
                self._watcher = None
            return
        if self._watcher is not None:
            return
        from PySide6.QtCore import QFileSystemWatcher
        self._watcher = QFileSystemWatcher(self)
        self._watcher.addPaths([p for p in hotreload.watch_paths() if p])
        # Both signals land on the same debounce: editors save by writing in place (file
        # signal) and by write-temp-then-rename (directory signal), and a new kernel file
        # only ever shows up as a directory change.
        self._watcher.fileChanged.connect(self._on_node_file_changed)
        self._watcher.directoryChanged.connect(self._on_node_file_changed)

    def _on_node_file_changed(self, _path: str) -> None:
        """Debounce a save into one reload.

        An editor writing a file emits several changes in a few milliseconds, and some
        replace the file rather than modify it — which drops the watch — so the path list is
        re-armed on every wake. The delay is what keeps a save from being reloaded halfway
        written; the compile gate in :mod:`nodegraph.hotreload` catches the rest."""
        self._reload_timer.start(NODE_RELOAD_DEBOUNCE_MS)

    def _rearm_watcher(self) -> None:
        if self._watcher is None:
            return
        watched = set(self._watcher.files()) | set(self._watcher.directories())
        missing = [p for p in hotreload.watch_paths() if p and p not in watched]
        if missing:
            self._watcher.addPaths(missing)

    def _autoreload_tick(self) -> None:
        """The debounced watcher wake-up: reload if anything really changed.

        Deliberately quiet — an automatic reload reports through the status bar only, never
        a dialog. The user did not ask for this reload at this instant, and a modal box
        landing on a keystroke while they are typing in their editor would be worse than the
        stale code it warns about. An explicit Ctrl+R still gets the dialog."""
        self._rearm_watcher()
        if not hotreload.pending():
            return
        self.reload_node_code(quiet=True)

    def reload_node_code(self, *, force: bool = False, quiet: bool = False) -> bool:
        """Re-read the node modules from disk and put the live session on the new code.

        Returns whether anything was reloaded. ``quiet`` suppresses the dialogs (the
        watcher's path — see :meth:`_autoreload_tick`).

        Refused while a pull is in flight: the engine hands a worker thread the compute
        objects it looked up, so replacing them mid-run would have one graph evaluated by two
        different versions of the same node. With auto-reload on, the refusal re-arms the
        timer instead of dropping the edit, so saving during a long run reloads the moment
        it lands."""
        if self.runner.busy:
            self.statusBar().showMessage(
                "Node reload deferred — a pull is running", 4000)
            if self._autoreload_act.isChecked():
                self._reload_timer.start(NODE_RELOAD_DEBOUNCE_MS * 4)
            return False
        rep = hotreload.reload_nodes(only_changed=not force)
        self.statusBar().showMessage(rep.summary(), 8000)
        if not rep.ok:
            if not quiet:
                box = QMessageBox(QMessageBox.Warning, "Node reload failed",
                                  rep.summary(), QMessageBox.Ok, self)
                box.setInformativeText(
                    "The session is still running the code it had before."
                    if not rep.broken else
                    "A module failed partway through executing, which Python cannot undo. "
                    "The node catalog was restored, but restart before trusting results.")
                if rep.trace:
                    box.setDetailedText(rep.trace)
                box.exec()
            return False
        if not rep.changed:
            return False
        self._apply_reload(rep, quiet=quiet)
        return True

    def refresh_node_list(self) -> bool:
        """Re-read the whole node catalog from disk — the palette's ⟳ button.

        The difference from *Reload node code* is what it can see. A reload refreshes modules
        that are already imported; this one resolves the node list against the **filesystem**,
        so it also picks up a node whose ``.py`` was created while the window was open (never
        imported, so in no list and in no palette) and retires one whose file was deleted
        (still registered, still offering itself from the palette, its source long gone)."""
        if self.runner.busy:
            self.statusBar().showMessage(
                "Refresh deferred — a pull is running; press ⟳ again when it finishes", 5000)
            return False
        rep = hotreload.refresh_catalog()
        if not rep.ok:
            self.statusBar().showMessage(rep.summary(), 8000)
            box = QMessageBox(QMessageBox.Warning, "Refreshing the node list failed",
                              rep.summary(), QMessageBox.Ok, self)
            box.setInformativeText(
                "The node list is unchanged apart from any file that was deleted. A new node "
                "file that does not import cleanly is reported here rather than being added "
                "half-registered."
                if not rep.broken else
                "A module failed partway through executing, which Python cannot undo. The "
                "node catalog was restored, but restart before trusting results.")
            if rep.trace:
                box.setDetailedText(rep.trace)
            box.exec()
            self._apply_reload(rep, quiet=True)      # a deletion may still have landed
            return False
        self._apply_reload(rep, quiet=True)
        bits = []
        if rep.added:
            bits.append(f"+{len(rep.added)} new ({', '.join(sorted(rep.added)[:3])}"
                        f"{'…' if len(rep.added) > 3 else ''})")
        if rep.removed:
            bits.append(f"-{len(rep.removed)} gone")
        if rep.restamped:
            bits.append(f"{len(rep.restamped)} re-keyed")
        from nodelab_v2.scene import visible_specs
        self.statusBar().showMessage(
            f"Node list refreshed: {', '.join(bits)}" if bits else
            f"Node list refreshed — {len(visible_specs())} nodes, all up to date", 6000)
        return True

    def reload_node_type(self, op_key: str) -> bool:
        """Re-read ONE node type's module from disk — the inspector's ⟳ button.

        Unconditional, unlike the menu action: it does not ask whether the file looks changed.
        Someone pressing Reload on a node whose panel disagrees with their editor is reporting
        that the session's state is wrong, and "nothing changed" is not a useful answer to
        that — the digest can legitimately match (an edit saved and reverted, a reload that
        was refused while a pull ran and then lost) while the registered spec is still stale.
        Re-executing one node module costs nothing and re-registers the spec, which is the
        repair."""
        mod = hotreload.module_of_op(op_key)
        if not mod:
            self.statusBar().showMessage(f"{op_key} is not a reloadable node type", 5000)
            return False
        if self.runner.busy:
            self.statusBar().showMessage(
                "Reload deferred — a pull is running; press ⟳ again when it finishes", 5000)
            return False
        rep = hotreload.reload_nodes(modules=[mod])
        if not rep.ok:
            self.statusBar().showMessage(rep.summary(), 8000)
            box = QMessageBox(QMessageBox.Warning, f"Reloading {op_key} failed",
                              rep.summary(), QMessageBox.Ok, self)
            box.setInformativeText(
                "The session is still running the code it had before."
                if not rep.broken else
                "A module failed partway through executing, which Python cannot undo. The "
                "node catalog was restored, but restart before trusting results.")
            if rep.trace:
                box.setDetailedText(rep.trace)
            box.exec()
            return False
        self._apply_reload(rep, quiet=True)
        extra = (f" (+{len(rep.reloaded) - 1} that share its code)"
                 if len(rep.reloaded) > 1 else "")
        self.statusBar().showMessage(
            f"Reloaded {op_key} from {mod.rpartition('.')[2]}.py{extra}", 6000)
        return True

    def _apply_reload(self, rep, *, quiet: bool) -> None:
        """Push a successful reload through the GUI: palette, cards, inspector.

        Memo entries from the previous code are deliberately **not** dropped. The reload
        re-keyed them (:mod:`nodegraph.revision`'s code fingerprints), so they are already
        unreachable rather than wrong, and the memo's byte budget releases them in its own
        time — whereas dropping them by hand would also mean dropping the tile and decoded-
        plane caches that a source's ingest paid minutes for and that no node edit
        invalidates."""
        self.palette.reload_catalog()
        orphaned = self.scene.resync_specs()
        self.inspector.rebuild()
        self.scene.update()
        if orphaned and not quiet:
            QMessageBox.warning(
                self, "Node types no longer defined",
                "These placed nodes name an op the reloaded code no longer defines:\n\n"
                + "\n".join(f"  • {nid}" for nid in orphaned[:12])
                + ("\n  …" if len(orphaned) > 12 else "")
                + "\n\nThey are left on the canvas with their wires so putting the "
                  "definition back restores them. They cannot be pulled until then.")

    def _bake_selected(self) -> None:
        nid = self._selected_dock()
        if nid is None:
            QMessageBox.information(
                self, "No dock selected",
                "Select a Dock Data node first (or add one from the Nodes palette, under "
                "io).\n\nA dock sits in the middle of a chain: bake it and everything "
                "above is computed once, written to disk and then read back from there, "
                "so you can keep adding nodes without re-running — or re-holding — any "
                "of it.")
            return
        self._start_bake(nid)

    def _on_dock_action(self, node_id: str, action: str) -> None:
        """Everything a Dock node's buttons and menu can ask for."""
        rec = self.doc.nodes.get(node_id)
        if rec is None or rec.op_key != DOCK_OP:
            return
        if action in ("bake", "bake_scoped"):
            self._start_bake(node_id, scoped=(action == "bake_scoped"))
        elif action == "undock":
            self.doc.set_dock_state(node_id, False)
            self.runner.invalidate()
            self.statusBar().showMessage(
                f"{node_id} un-docked — the chain above runs live again; the bake is "
                f"still on disk, so re-docking is instant")
        elif action == "redock":
            self.doc.set_dock_state(node_id, True)
            self.runner.invalidate()
            self._release_dormant()
            self.statusBar().showMessage(f"{node_id} docked — serving its existing bake")
        elif action == "reveal":
            self._reveal(self.doc.dock_store(node_id))

    # ── iterate (V2.19) ───────────────────────────────────────────────────────
    def _iterate_owner(self, node_id: Optional[str]) -> Optional[str]:
        """The Iterate node whose iteration strip belongs to ``node_id`` — itself, if it
        is one. Only the Iterate node's OWN output is a choice between iterations; a node
        inside its cone has one result per clone, and the clones are rewrite artifacts with
        no card to select."""
        rec = self.doc.nodes.get(node_id or "")
        return node_id if rec is not None and rec.op_key == ITERATE_OP else None

    def _sync_iteration_strip(self, node_id: Optional[str]) -> None:
        """Show the viewer's iteration strip for an Iterate node, captioned with the value
        each iteration used. Silent about a misconfigured sweep — the inspector panel is
        where that is reported, and two copies of the same complaint is one too many."""
        owner = self._iterate_owner(node_id)
        if owner is None:
            self.viewer.set_iterations(())
            return
        try:
            plan = iterate_plan(self.doc.to_graph(), owner, envs=self.doc.envs)
        except Exception:                     # noqa: BLE001 — half-wired: no strip, no noise
            self.viewer.set_iterations(())
            return
        labels = []
        for it in plan.iterations:
            parts = [("—" if v is None else (f"{v:g}" if isinstance(v, (int, float))
                                             else str(v))) for v in it.values]
            labels.append(" · ".join(parts))
        current = int(self.doc.nodes[owner].params.get("index", 0) or 0)
        self.viewer.set_iterations(labels, current)

    def _on_iteration_changed(self, index: int) -> None:
        """The iteration strip moved: keep that iteration and re-pull.

        Writing ``preserve=picked`` alongside the index is deliberate — scrubbing to an
        iteration under 'best' would otherwise change nothing visible, since the metric
        still decides. Picking one by eye IS the statement that you want that one."""
        owner = self._iterate_owner(self._viewed)
        if owner is None:
            return
        rec = self.doc.nodes[owner]
        if int(rec.params.get("index", 0) or 0) == index \
                and rec.modes.get("preserve") == "picked":
            return
        rec.params["index"] = int(index)
        rec.modes["preserve"] = "picked"
        self.doc.touch()
        self.pull_node(owner)

    def _on_iterate_action(self, node_id: str, action: str) -> None:
        """Run sweep / stop sweeping, from the Iterate panel."""
        rec = self.doc.nodes.get(node_id)
        if rec is None or rec.op_key != ITERATE_OP:
            return
        current = set(self.runner.sweep_all)
        if action == "sweep":
            current.add(node_id)
        elif action == "stop_sweep":
            current.discard(node_id)
        else:
            return
        self.runner.set_sweep_all(current)
        if action == "sweep" and not self.runner.solo_frame:
            self.statusBar().showMessage(
                "running every iteration over the WHOLE series — F9 (troubleshoot) and a "
                "picked frame makes a sweep affordable")
        self.pull_node(node_id)

    def _record_sweep(self, node_id: str, payload) -> None:
        """Keep an Iterate node's results table from the payload it just produced.

        The table is stamped on the Dataset by the compute (the only place that holds every
        iteration at once), so recording it is a copy rather than a second pull. It is a
        UI-only param — stripped from the run graph in ``document._UI_PARAM_KEYS`` — because
        a record written on EVERY pull would otherwise re-key the memo entry that produced
        it and re-run the sweep forever. Written straight onto ``rec.params`` for the same
        reason: ``touch()`` would bump the document revision and cost a rebuild for a value
        nothing computes from."""
        rec = self.doc.nodes.get(node_id)
        md = getattr(payload, "metadata", None)
        if rec is None or rec.op_key != ITERATE_OP or not isinstance(md, dict):
            return
        rows = md.get(SWEEP_ROWS_KEY)
        if not rows:
            return
        rec.params[SWEEP_KEY] = {"rows": [
            {"iter": r.get("iter"), "metric": r.get("metric"), "won": r.get("won")}
            for r in rows]}
        shown = getattr(self.inspector, "_node", None)
        if shown is not None and getattr(shown, "node_id", None) == node_id:
            self.inspector.set_node(shown)      # redraw the table with the new metrics

    def _start_bake(self, node_id: str, *, scoped: bool = False) -> None:
        """Validate, confirm, then hand the bake to the runner.

        The confirmation is not ceremony: a bake reads every plane of the series through
        the whole chain above it and writes the result, which on a real file is minutes
        and gigabytes. Saying so first — with the folder it will use — is cheaper than
        letting the user discover it."""
        import datetime
        import os
        rec = self.doc.nodes[node_id]
        state = rec.spec().default_state() if rec.spec() else {}
        state.update(rec.modes)
        precision = state.get("precision", PRECISION_UNSET)
        if precision == PRECISION_UNSET:
            QMessageBox.information(
                self, "Choose a precision first",
                "This dock has no precision set, and there is no sensible default — the "
                "right answer depends on what the chain above it produces.\n\n"
                "• float32 — halves the store versus float64 and keeps ~7 significant "
                "digits, far beyond what the sensor recorded. The usual choice after a "
                "filter chain.\n"
                "• float64 — bit-identical to running live, at 4× the original file's "
                "size for float data.\n"
                "• uint16 — smallest, but only correct while the values are still camera "
                "counts; normalized [0,1] data would collapse (the bake refuses it).\n\n"
                "Label rasters, masks and integer images are stored as themselves either "
                "way.")
            return
        if not self.doc.edge_into(node_id, "data"):
            QMessageBox.information(
                self, "Nothing to bake",
                f"{node_id} has nothing wired into its 'data' input, so there is no "
                f"chain to freeze. Connect the pipeline you want to bake into it first.")
            return
        store = self.doc.dock_store(node_id) or self.doc.default_dock_store(node_id)
        existing = os.path.isdir(store)
        scope = ("ONLY the frames picked in the Viewer" if scoped
                 else "the whole series (every position, timepoint, z and channel)")
        msg = (f"Bake everything above {node_id} to disk?\n\n"
               f"Range: {scope}\nPrecision: {precision}\nFolder: {store}\n\n"
               f"This computes the chain once — on a large file that can take a while — "
               f"and writes the image plus every mask, label, track and measurement it "
               f"carries. Afterwards those nodes grey out and are released from memory.")
        if existing:
            msg += "\n\nThe existing bake in that folder will be replaced."
        if scoped:
            msg += ("\n\nA scoped bake produces a TRUNCATED series. Every node after "
                    "this dock will run on those frames only — use it to check a chain, "
                    "not to produce results.")
        if QMessageBox.question(self, "Bake dock", msg,
                                QMessageBox.Yes | QMessageBox.Cancel) != QMessageBox.Yes:
            return
        self.runner.set_frame_selection(*self.viewer.frame_selection())
        bake_id = uuid.uuid4().hex
        started = self.runner.bake(
            node_id, store=store, precision=precision, bake_id=bake_id,
            signature=self.doc.dock_signature(node_id), scoped=scoped,
            coords=self.viewer.coords())
        if not started:
            self.statusBar().showMessage(
                "a pull is already running — wait for it to finish, then bake")
            return
        self._baked_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.statusBar().showMessage(f"baking {node_id} → {store} …")

    def _on_baked(self, node_id: str, spec: dict) -> None:
        """Record a finished bake, dock the node, and release what it made redundant."""
        man = spec.get("manifest") or {}
        self.doc.set_dock_bake(
            node_id, store=spec["store"], bake_id=str(man.get("bake_id", "")),
            precision=str(man.get("precision", spec.get("precision", ""))),
            signature=str(spec.get("signature", "")),
            nbytes=int(spec.get("bytes", 0) or 0),
            when=getattr(self, "_baked_at", ""))
        self.runner.invalidate()
        freed = self._release_dormant()
        from nodelab_v2.inspector import _human_bytes
        self.statusBar().showMessage(
            f"{node_id} docked — {_human_bytes(spec.get('bytes', 0))} on disk; "
            f"{freed} cached result(s) released and {len(self.doc.dormant)} node(s) "
            f"greyed out"
            + ("  ·  SCOPED bake: a truncated series" if spec.get("scoped") else ""))
        if self._viewed is not None:
            self.pull_node(self._viewed)

    def _release_dormant(self) -> int:
        """Free the memory the dock exists to free — the memo payloads of every node a
        docked run no longer evaluates. This is the "unload it from the software" half
        of docking; greying the cards is only the half you can see."""
        return self.runner.unload(sorted(self.doc.dormant))

    @staticmethod
    def _reveal(path: str) -> None:
        """Open a dock folder in the platform's file browser."""
        import os
        import subprocess
        import sys
        if not path or not os.path.isdir(path):
            return
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606 — a directory the app itself just wrote
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ── solo-frame troubleshooting scope (F9) ─────────────────────────────────
    def set_solo_frame(self, on: bool) -> None:
        """Run → *Troubleshoot: picked frames only*.

        Hands the scope to the runner (which cuts each source seed down to the scoped
        frames — see :meth:`~nodelab_v2.runner.EngineRunner.set_solo_frame`) and to the
        Viewer (whose M/T strips become the frame chooser), then re-pulls the viewed node
        so the change is visible immediately rather than at the next click."""
        on = bool(on)
        self.runner.set_solo_frame(on)
        self.runner.set_frame_selection(*self.viewer.frame_selection())
        self.inspector.set_solo_frame(on)     # an Iterate panel warns when the scope is off
        self._sync_solo(self._viewed)
        if self._solo_act.isChecked() != on:      # keep a programmatic call in sync
            self._solo_act.blockSignals(True)
            self._solo_act.setChecked(on)
            self._solo_act.blockSignals(False)
        if self._viewed is not None:
            self.pull_node(self._viewed)       # show the new scope now, not on the next click
        self.statusBar().showMessage(          # after the pull: `started` also writes here
            f"troubleshooting: pulls analyse {self._scope_phrase()} — ctrl+click the T or "
            f"Z strip to pick more, F9 to run the full series"
            if on else "troubleshooting off — pulls analyse the whole series")

    def _scope_phrase(self) -> str:
        """What the next scoped pull will run, in words — shared by the status line and
        the chip's tooltip so the two can never disagree."""
        ms, ts, zs = self.viewer.scoped_frames()
        totals = self.viewer.solo or (1, 1, 1)
        where = f"t={compact_list(ts)}"
        if totals[0] > 1:
            where += f" (m={compact_list(ms)})"
        n = len(ms) * len(ts)
        phrase = f"frame {where} only" if n == 1 else f"{n} frames only: {where}"
        if zs:
            phrase += (f", cut to {len(zs)} of {totals[2]} z-planes "
                       f"(z={compact_list(zs)})")
        return phrase

    def clear_frame_picks(self) -> None:
        """Run → *Clear picked frames*: back to scoping the single frame the cursor is on,
        whole volume. Reachable from the menu because a selection made on a strip that is
        currently scrolled out of the mini-map is otherwise invisible."""
        self.viewer.clear_frame_selection()

    def _on_frame_selection(self) -> None:
        """The Viewer's M/T/Z picks changed — that is a change to *what a pull computes*,
        so re-scope the runner and (under the scope) re-run the viewed node. Off the scope
        it is bookkeeping only: the picks are remembered for whenever F9 is armed."""
        self.runner.set_frame_selection(*self.viewer.frame_selection())
        self._sync_solo_chip()
        if not self.runner.solo_frame:
            # Picks are inert until the scope is armed. Say so — a user who just
            # ctrl-clicked three boxes and saw nothing happen has no other way to find out.
            ms, ts, zs = self.viewer.frame_selection()
            if ms or ts or zs:
                frames = len(ms or (0,)) * len(ts or (0,))
                self.statusBar().showMessage(
                    f"{frames} frame(s)" + (f" × {len(zs)} z-plane(s)" if zs else "")
                    + " picked — press F9 to scope pulls to them")
            return
        self.statusBar().showMessage(f"troubleshooting: pulls analyse "
                                     f"{self._scope_phrase()}")
        if self._viewed is not None:
            self.pull_node(self._viewed)

    def _sync_solo(self, node_id: Optional[str]) -> None:
        """Re-point the Viewer's frame chooser at ``node_id``'s source extent. Called on
        toggle, before every pull and after every document edit, because all three can
        change how many frames and planes there are to choose from."""
        on = self.runner.solo_frame
        self.viewer.set_solo(self.doc.source_scope_totals(node_id)
                            if (on and node_id) else None)
        self._sync_solo_chip()

    def _scope_tag(self) -> str:
        """The scope in a few characters — ``t7`` / ``m2·3T[0,4,9]·2Z``. Shared by the
        status chip and the canvas badge so the two always read the same."""
        totals = self.viewer.solo or (1, 1, 1)
        ms, ts, zs = self.viewer.scoped_frames()
        name = f"t{compact_list(ts)}" if len(ts) == 1 else f"{len(ts)}T[{compact_list(ts)}]"
        if totals[0] > 1:
            name = (f"m{ms[0]}" if len(ms) == 1 else f"{len(ms)}M") + "·" + name
        return name + (f"·{len(zs)}Z" if zs else "")

    def _sync_solo_chip(self) -> None:
        """Raise (or drop) every indicator of the scope: the amber frame + badge on the
        node canvas, and the status-bar chip. Also the theme re-apply path.

        Keyed on the RUNNER's scope, not on the Viewer's chooser: with the scope armed but
        nothing viewed yet there are no source totals to span, and the indicator still has
        to be up — it says what the next pull will do.

        The canvas HUD is the loud one on purpose. A scoped pull leaves the graph, the
        cards and the progress rails looking *identical* to a full run, so the only thing
        that can stop a user reading one-frame numbers as series numbers is the canvas
        itself being unmistakably marked."""
        if not self.runner.solo_frame:
            self._solo_chip.hide()
            self.view.set_troubleshooting(False)
            return
        name = self._scope_tag()
        self.view.set_troubleshooting(True, name)
        self._solo_chip.setText(f"SOLO {name}")
        self._solo_chip.setToolTip(f"{self._solo_act.toolTip()}\n\n"
                                   f"Now: pulls analyse {self._scope_phrase()}.")
        self._solo_chip.setStyleSheet(
            f"color:{T.DIM2D_INK.name()}; background:{T.DIM2D.name()}; font-size:10px;"
            f"font-weight:800; border-radius:4px; padding:1px 6px; margin:0 4px;")
        self._solo_chip.show()

    def _on_view_request(self) -> None:
        if self._viewed is not None:
            # coords/channel-only change → runner serves from the decoded-plane cache
            # (no graph snapshot / engine re-pull) when the graph is unchanged. Under the
            # solo-frame scope an M/T move onto a frame outside the scope IS a new frame to
            # compute, and the runner turns it back into a real pull — the chip has to
            # follow the cursor either way.
            self._sync_solo_chip()
            self.runner.request_plane(self._viewed, self.viewer.coords(),
                                      self.viewer.channels())

    def _on_run_finished(self, node_id, payload, plane, axes, seconds) -> None:
        if plane:
            self._open_viewer()       # there is something to see now — unfold the pane
        self.viewer.show_result(node_id, plane, axes, seconds, dataset=payload,
                                overlay=self.runner.overlay_channels(node_id),
                                overlay_note=self.runner.overlay_note(node_id),
                                overlay_style=self.runner.overlay_style(node_id))
        # flicker is a property of TIME, not of the composite, so it is driven here rather
        # than folded into the style map the shader reads
        self.viewer.set_overlay_flicker(self.runner.overlay_flicker_hz(node_id))
        if self._maximized:
            # a new axes shape rebuilds the channel/LUT controls, and fresh widgets are
            # visible — re-fold them so the mini-map keeps its compact strip
            self.viewer.set_compact(True, force=True)
        self.sheet.show_dataset(node_id, payload)
        self._record_sweep(node_id, payload)
        self._sync_iteration_strip(node_id)
        self.minimap.set_state("live")
        self._idle_led()                      # …unless files are still ingesting
        self._set_progress(None)              # the run is over — no bar to show
        # settle the cards: nothing is queued now, and the pulled node wears the run's
        # wall time (its own compute may have been microseconds — the wait was the read)
        self.scene.finish_run(node_id, seconds=seconds)
        computed = getattr(self, "_run_computed", 0)
        cached = getattr(self, "_run_cached", 0)
        detail = f" ({computed} computed, {cached} cached)" if (computed or cached) else ""
        self.statusBar().showMessage(f"{node_id} pulled in {seconds:.2f}s{detail}")

    # ── per-node progress (G7 + 2026-07-28) ──────────────────────────────────
    def _on_run_plan(self, target: str, node_ids) -> None:
        self._run_target = target
        self._run_plan = [n for n in node_ids if n in self.doc.nodes]
        self._run_computed = 0
        self._run_cached = 0
        self.scene.set_run_plan(target, self._run_plan)

    def _set_progress(self, fraction, levels: Optional[dict] = None) -> None:
        """Show the status-bar bars, or hide them when there is no honest number to show
        (an indeterminate compute keeps the card's sweeping rail and the pulsing LED — a
        bar that invents a position is worse than no bar).

        ``levels`` is the engine's ``progress`` info dict. When it carries the two-level
        split the upper bar shows frames finished / total frames and the lower one shows
        the work inside the frame in flight; otherwise only the lower bar appears, holding
        the flat ``fraction`` as it always did."""
        def _put(bar, value) -> None:
            if isinstance(value, float):
                bar.setRange(0, 1000)
                bar.setValue(int(round(max(0.0, min(1.0, value)) * 1000)))
                bar.show()
            else:
                bar.hide()

        def _busy(bar) -> None:
            """Qt's own indeterminate bar (``range(0,0)``) — the footer twin of the card's
            sweeping rail, for a frame whose inner work is one opaque call."""
            bar.setRange(0, 0)
            bar.show()

        levels = levels or {}
        frames = levels.get("frames")
        if frames:
            _put(self._prog_frame, _as_float(levels.get("frame_fraction")))
            # `sub_fraction` present-but-None means "working, position unknown" — sweep it
            # rather than hiding it, or the footer would claim the frame bar is all there is.
            if "sub_fraction" in levels and levels.get("sub_fraction") is None:
                _busy(self._prog)
            else:
                _put(self._prog, _as_float(levels.get("sub_fraction")))
            return
        self._prog_frame.hide()
        _put(self._prog, fraction)

    def _on_node_progress(self, event: str, node_id: str, info: dict) -> None:
        self.scene.on_node_progress(event, node_id, info)
        # An ingest reports with no epoch (it belongs to a file, not to a run). Its card
        # rail is updated above like any other node's, but it must not touch the pull's
        # counters or relabel the footer "pulling …" — and while a pull IS in flight it
        # keeps off the shared status line entirely; the card is where it shows.
        if info.get("epoch") is None:
            if event == "progress" and not self.runner.busy:
                frac = info.get("fraction")
                pct = f" {int(round(frac * 100))}%" if isinstance(frac, float) else ""
                left = len(self.runner.ingesting())
                self.statusBar().showMessage(
                    f"{info.get('note') or 'ingesting'}{pct}"
                    + (f"  ·  {left} file(s) in flight" if left > 1 else ""))
                self._set_progress(frac, info)
            return
        if event == "done":
            self._run_computed = getattr(self, "_run_computed", 0) + 1
        elif event == "cached":
            self._run_cached = getattr(self, "_run_cached", 0) + 1
        if event in ("start", "progress", "decode"):
            total = len(getattr(self, "_run_plan", ()) or ())
            ran = getattr(self, "_run_computed", 0) + getattr(self, "_run_cached", 0)
            frac = info.get("fraction")
            where = ("reading planes" if event == "decode"
                     else f"{node_id}{(' · ' + str(info.get('note'))) if info.get('note') else ''}")
            pct = f" {int(round(frac * 100))}%" if isinstance(frac, float) else ""
            # the frame count is the number a user watching a long series actually wants,
            # so it goes in the message ahead of the overall percentage.
            frames, frame = info.get("frames"), info.get("frame")
            if frames and frame is not None:
                pct = f" frame {int(frame) + 1}/{int(frames)}{pct}"
                if "sub_fraction" in info and info.get("sub_fraction") is None:
                    pct += " (working)"        # one opaque call inside this frame
            self.statusBar().showMessage(
                f"pulling {getattr(self, '_run_target', node_id)} — "
                f"{min(ran + 1, total) if total else 1}/{total or 1} · {where}{pct}")
            self._set_progress(frac, info)

    def _on_plane_ready(self, node_id, planes, axes, seconds) -> None:
        # fast-path display update (scrub/play): no dataset re-delivery, no spreadsheet
        # refresh — only the viewer's frame changes.
        self.viewer.show_planes(node_id, planes, axes, seconds)
        # A COLD frame is decoded off the GUI thread and announces itself as "reading
        # planes" while it runs (EngineRunner._serve_from_cache); the frame landing is the
        # end of that, so the rail goes back to idle. A warm frame never raised it.
        self._set_progress(None)

    def _request_detail(self, node_id, coords, channels, rect01) -> None:
        """The Viewer wants the visible rect at full detail (see
        :attr:`nodelab_v2.viewer.ViewerPanel.detail_cb`).

        The window is the seam because the panel owns no provider and the runner owns no
        viewport — exactly the split that already routes ``raw_plane_cb``. The budget is
        the same :data:`~nodelab_v2.runner.MAX_DISPLAY_DIM` the overview uses, so a patch
        is at worst the overview's resolution over a smaller area, and at best that
        resolution over the part you are actually looking at."""
        from nodelab_v2.runner import MAX_DISPLAY_DIM
        self.runner.request_detail(node_id, coords, channels, rect01, MAX_DISPLAY_DIM)

    def _on_run_failed(self, node_id, trace) -> None:
        # console removed: the Viewer shows the failing line (status) + full trace (tooltip)
        # …and the card that raised keeps its red 'error' state (the engine's error event
        # named it, which is more precise than the pulled node).
        self.scene.finish_run(node_id, failed=True)
        self.viewer.show_error(node_id, trace)
        self._set_led("error")
        self._set_progress(None)              # a failed run must not leave a stale bar
        self.minimap.set_state("error")
        last = [ln for ln in trace.strip().splitlines() if ln.strip()]
        self.statusBar().showMessage(f"{node_id} FAILED — {last[-1] if last else 'see Viewer'}")

    # ── file (G6) ────────────────────────────────────────────────────────────
    def file_new(self) -> None:
        self.doc.clear()
        self._viewed = None
        self.scene.set_viewed(None)
        self.minimap.set_state("idle")
        self._sync_minimap_title()

    def _add_source_node(self, path: str, x: float, y: float):
        """Read ``path``'s metadata (**no pixels**) and drop one ``io.load`` card at
        ``(x, y)`` — titled with the file name, carrying one output socket per channel and
        its envelope seeded so the sockets/colors appear at once. Returns ``(rec, axes)``.

        Split out of :meth:`file_load_source` so a multi-file load runs the identical
        per-file path. It raises whatever the reader raises rather than showing a dialog:
        one bad file in a selection of ten is a line in a summary, not ten modal
        interruptions, and the caller is the one that knows which it is."""
        import os
        from nodelab_v2.document import CHANNELS_KEY, TITLE_KEY
        from nodegraph.metadata import MetaEnvelope
        from nodelab_v2.ingest import read_meta_only

        axes, calib, disp = read_meta_only(path)
        names = disp.get("channel_names") or [f"Ch{i}" for i in range(axes.c)]
        emis = disp.get("channel_emission_nm")
        colors = disp.get("channel_colors")

        def _native(i):
            # only accept an already-unpacked [r,g,b] triple; ND2's packed-int colorRGB
            # has an ambiguous byte order, so we fall back to the emission tint for those.
            if isinstance(colors, (list, tuple)) and i < len(colors):
                col = colors[i]
                if isinstance(col, (list, tuple)) and len(col) == 3:
                    return [int(v) for v in col]
            return None

        chans = [{
            "name": names[i] if i < len(names) else f"Ch{i}",
            "emission_nm": (emis[i] if isinstance(emis, (list, tuple)) and i < len(emis)
                            else None),
            "color": _native(i),
        } for i in range(axes.c)]

        rec = self.doc.add_node(
            LOAD_OP, x=x, y=y,
            params={"path": path, TITLE_KEY: os.path.basename(path),
                    CHANNELS_KEY: chans})
        self.doc.set_meta_seed(rec.id, MetaEnvelope(axes=axes, metadata=dict(calib)))
        return rec, axes

    def file_load_source(self) -> None:
        """File → Load ND2/TIFF file…: pick **one or more** images, read each file's
        metadata (no pixels), and drop one pre-loaded ``io.load`` source node per file
        (:meth:`_add_source_node`). Pixels ingest lazily on the first pull.

        Multi-select (2026-07-31) is why the dialog is ``getOpenFileNames``. Nothing below
        the GUI was ever single-file: :meth:`~nodelab_v2.runner.EngineRunner._resolve_source`
        keys one provider and one on-disk store per ``("image", abspath)`` and the runner
        collects *every* ``io.load`` in the document into its ``sources`` map, while
        ``view.overlay`` and the DVC/correlation reference inputs exist precisely to combine
        two files. Only the loader was one-at-a-time, so building a two-file graph meant two
        trips through the File menu.

        The cards stack in a **column** (:data:`SOURCE_STACK_GAP`) instead of landing on top
        of each other, each offset by the height of the one before — a source card is as tall
        as its channel count, so a fixed pitch would either overlap a 5-channel file or
        strand a 1-channel one. The view then scrolls (never zooms — an existing graph keeps
        its scale) to bring the new column into sight, and all of the new cards land
        selected, so the next palette drop can be wired against them.

        A file the reader refuses does not cost the others: failures are collected and
        reported once, after every readable file has been placed.
        """
        import os

        paths, _f = QFileDialog.getOpenFileNames(
            self, "Load ND2 / ND3 / TIFF file(s)", "",
            "Microscopy images (*.nd2 *.nd3 *.tif *.tiff);;ND2 (*.nd2);;"
            "ND3 (*.nd3);;TIFF (*.tif *.tiff);;All files (*)")
        if not paths:
            return

        c = self.view.mapToScene(self.view.viewport().rect().center())
        x, y = c.x() - 107, c.y() - 40
        added: list = []
        failed: list = []
        # Reading N files' metadata is N ND2 header parses — fast per file, but visibly not
        # instant for a folder's worth, and it all happens before the first card appears.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                try:
                    rec, axes = self._add_source_node(path, x, y)
                except Exception as exc:  # noqa: BLE001 — any reader failure, per file
                    failed.append((path, exc))
                    continue
                added.append((rec, axes))
                item = self.scene.node_items.get(rec.id)
                # `card_rect` is the card's real geometry (boundingRect carries the glow
                # margin). The item exists already: `add_node` notifies the document
                # synchronously and the scene syncs on that.
                y += (item.card_rect().height() if item is not None else 110.0) \
                    + SOURCE_STACK_GAP
        finally:
            QApplication.restoreOverrideCursor()

        items = [self.scene.node_items[r.id] for r, _a in added
                 if r.id in self.scene.node_items]
        if items:
            self.scene.clearSelection()
            span = QRectF()
            for it in items:
                it.setSelected(True)
                span = span.united(it.card_rect().translated(it.pos()))
            if len(items) > 1:
                self.view.ensureVisible(span, 60, 60)

        if failed:
            detail = "\n\n".join(f"{os.path.basename(p)}\n  {exc}" for p, exc in failed)
            QMessageBox.warning(
                self, "Load failed",
                f"Could not read {len(failed)} of {len(paths)} file(s):\n\n{detail}"
                + (f"\n\nThe other {len(added)} loaded." if added else ""))
        if not added:
            return
        if len(added) == 1:
            rec, axes = added[0]
            self.statusBar().showMessage(
                f"loaded {os.path.basename(rec.params['path'])} — {axes.c} channel(s)")
        else:
            shown = [os.path.basename(r.params["path"]) for r, _a in added[:3]]
            more = len(added) - len(shown)
            self.statusBar().showMessage(
                f"loaded {len(added)} files — " + ", ".join(shown)
                + (f" +{more} more" if more else ""))

    def file_open(self) -> None:
        path, _f = QFileDialog.getOpenFileName(self, "Open graph", "", FILE_FILTER)
        if not path:
            return
        try:
            self.doc.load_file(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        self.view.fit_all()
        if self.doc.has_unedited_structure:
            QMessageBox.information(
                self, "Zones / groups preserved",
                "This file contains zones or node groups that this build can't edit "
                "yet. They are shown as their member nodes and preserved unchanged on "
                "save — editing/creating them in the GUI is a later phase.")

    def file_save(self) -> None:
        if not self.doc.path:
            self.file_save_as()
            return
        self.doc.save_file(self.doc.path)
        self.statusBar().showMessage(f"saved {self.doc.path}")

    def file_save_as(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "Save graph", "graph.nd2graph.json",
                                               FILE_FILTER)
        if not path:
            return
        self.doc.save_file(path)
        self.statusBar().showMessage(f"saved {path}")

    def file_export(self) -> None:
        from nodelab_v2.export import export_dataset
        ds = self.sheet.dataset
        if ds is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Pull a node with Label/Point/Track structures first "
                                    "(the Spreadsheet tab shows what's exportable).")
            return
        nid = self.sheet.node_id or "table"
        path, _f = QFileDialog.getSaveFileName(
            self, "Export structure tables", f"{nid}.csv",
            "CSV (*.csv);;Parquet (*.parquet);;Arrow IPC (*.arrow)")
        if not path:
            return
        try:
            n = export_dataset(ds, path)
        except (ValueError, OSError, ImportError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"exported {n} rows → {path}")

    # ── empty canvas / example content ────────────────────────────────────────
    def _sync_welcome(self) -> None:
        """Show the welcome card exactly while the canvas holds no nodes."""
        self.welcome.setVisible(not self.doc.nodes)
        if self.welcome.isVisible():
            self.welcome.raise_()
            self.minimap.raise_()          # the mini-map still owns its corner

    def focus_palette(self) -> None:
        """Put the cursor in the Nodes palette search (the welcome card's shortcut to
        placing a first node)."""
        w = self.palette.parentWidget()
        while w is not None and not isinstance(w, QDockWidget):
            w = w.parentWidget()
        if w is not None:
            w.show()
            w.raise_()
        self.palette.focus_search()

    def build_demo(self) -> None:
        """Replace the canvas with the small example chain (Load → Select → enhance →
        threshold → label → measure, plus a deconvolve → Viewer branch). Reachable from
        the welcome card; also what the GUI probe drives."""
        self.doc.clear()
        d = self.doc
        specs = [
            ("n1", "io.load", {}, 30, 150),
            ("n2", "channel.select", {}, 290, 150),
            ("n3", "enhance.gaussian", {"dim": "3D"}, 548, 90),
            ("n4", "analysis.threshold", {}, 806, 90),
            ("n5", "analysis.label", {}, 1064, 90),
            ("n6", "analysis.measure", {}, 1330, 150),
            ("n7", "enhance.deconvolve", {"dim": "3D"}, 548, 430),
            ("n8", "view.viewer", {}, 806, 430),
        ]
        for nid, op, modes, x, y in specs:
            d.add_node(op, node_id=nid, modes=modes, x=x, y=y)
        for s, ss, dd, ds in [
                ("n1", "image", "n2", "data"), ("n2", "out", "n3", "data"),
                ("n3", "out", "n4", "data"), ("n4", "out", "n5", "data"),
                ("n5", "out", "n6", "data"), ("n2", "out", "n7", "data"),
                ("n7", "out", "n8", "data")]:
            d.connect(s, ss, dd, ds)
        self.scene.node_items["n7"].setSelected(True)
        self.view.fit_all()
        self.statusBar().showMessage("example graph loaded — double-click a node to view it")


__all__ = ["MainWindow"]
