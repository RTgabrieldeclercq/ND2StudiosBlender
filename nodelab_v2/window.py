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

import time
import uuid
from typing import Any, Dict, List, Optional

import nodegraph.nodes  # noqa: F401 — registers the node catalog into NODES
from nodegraph import hotreload
from collections import OrderedDict

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDialog, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QSplitter, QToolButton, QVBoxLayout,
    QWidget,
)

from nodegraph.iterate import (
    ITERATE_OP, SWEEP_KEY, SWEEP_OWNER_KEY, SWEEP_ROWS_KEY, plan as iterate_plan)
from nodelab_v2 import theme as T
from nodelab_v2.console import ConsolePanel
from nodelab_v2.document import GraphDocument
from nodelab_v2.framestrip import compact_list
from nodelab_v2.inspector import InspectorPanel
from nodelab_v2.lablink.panel import LabLinkPanel
from nodelab_v2.minimap import MiniMapOverlay
from nodelab_v2.node_item import NodeItem
from nodelab_v2.ops import MOVIE_OP, CALIB_OVERRIDE_KEYS, DOCK_OP, LOAD_OP, PRECISION_UNSET
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

#: horizontal offset of the ``util.chain`` card the sequence loader drops beside its bundle
#: (scene px). Wide enough that the wire between them is visibly a wire rather than two
#: touching cards — a source card is ~214 px, so this leaves a clear ~90 px span.
SEQUENCE_CHAIN_GAP = 306

#: What the axis a sequence was chained onto is CALLED, for the status line. The keys are
#: ``util.chain``'s own ``chain_axis`` values; "M" is absent because the loader never drops
#: a chain card for it (re-addressing onto M is the identity the bundle already performed).
_AXIS_NOUN = {"T": "timepoints", "Z": "focal planes", "C": "channels"}
PROGRESS_BAR_GAP = 2

#: The longest a pressed ▶ may hold playback while its frames prepare (seconds) — applied
#: ONLY to a series larger than the display budget, which can never be fully resident, so
#: holding longer buys warmth the first lap immediately evicts. Such a series plays after
#: at most this wait, off whatever is warm, frames landing as they compute while the
#: preload keeps warming behind it. Enforced twice: by the ETA check in
#: :meth:`MainWindow._on_preload_progress` (drops the hold as soon as the measured rate
#: says the wait would run long) and by a wall-clock watchdog armed at ▶ (drops it even if
#: no tick ever arrives, so a wedged preload cannot strand a lit play button).
#:
#: A series that FITS the budget is held to completion instead — minutes if that is what
#: the frames cost — with the progress bar counting and ⏸ as the way out (it cancels the
#: preload, which drops the gate). That is the user's stated preference (2026-08-10):
#: pressing ▶ on a series that CAN be made smooth means "make it smooth", and watching
#: frames trickle in at decode cadence is worse than an honest, cancellable wait.
PLAY_PREPARE_MAX_S = 8.0


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


class _CompareBox(QWidget):
    """The compare pane's frame: a slim header naming what the second ViewerPanel shows
    — and whether its cursor is LINKED to the primary's — plus the close button. The
    header exists because a bare second image is ambiguous: two similar results side by
    side need the pane itself to say which node it is and why its sliders are (or are
    not) there."""

    def __init__(self, panel, on_close) -> None:
        super().__init__()
        self.panel = panel
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._head = QWidget()
        self._head.setObjectName("compareHead")
        h = QHBoxLayout(self._head)
        h.setContentsMargins(9, 2, 4, 2)
        h.setSpacing(6)
        self._title = QLabel("compare")
        close = QToolButton()
        close.setText("✕")
        close.setAutoRaise(True)
        close.setToolTip("Close the compare pane (Shift+F8)")
        close.clicked.connect(on_close)
        h.addWidget(self._title, 1)
        h.addWidget(close)
        v.addWidget(self._head)
        v.addWidget(panel, 1)
        self.restyle()

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def restyle(self) -> None:
        self._head.setStyleSheet(
            f"QWidget#compareHead {{ background:{T.BODY.name()}; "
            f"border-bottom:1px solid {T.BORDER.name()}; }}")
        self._title.setStyleSheet(
            f"color:{T.MUTED.name()}; font-size:10px; font-weight:800;")
        self.panel.restyle()


class _MovieHost:
    """The Movie Editor's view of the window — everything it may ask for, and nothing else.

    :class:`nodelab_v2.movie_editor.MovieEditorPanel` holds widgets and a working copy of one
    timeline; the window holds the runner (to compute its sources without moving the
    Viewer), the document (to commit the timeline) and the Viewer (whose LUTs it links to).
    An adapter rather than the window itself, so the panel's dependencies are this list and
    a probe can see them."""

    def __init__(self, win: "MainWindow") -> None:
        self._w = win

    def movie_state(self, node_id: Optional[str]) -> Optional[Dict[str, Any]]:
        rec = self._w.doc.nodes.get(node_id) if node_id else None
        if rec is None or rec.op_key != MOVIE_OP:
            return None
        spec = rec.spec()
        try:
            env = self._w.doc.env(node_id)
        except Exception:                  # noqa: BLE001 — an un-propagated node
            env = None
        return {"spec": spec, "params": dict(rec.params), "modes": dict(rec.modes),
                "env": env, "label": f"{spec.label if spec else rec.op_key} ({node_id})"}

    def movie_sources(self, node_id: str) -> Dict[str, Dict[str, Any]]:
        return self._w._movie_sources(node_id)

    def source_payload(self, node_id: str) -> Any:
        return self._w.runner.finished_result(node_id)

    def fetch(self, node_id: str) -> None:
        self._w.runner.fetch(node_id)

    def commit_timeline(self, node_id: str, text: str) -> None:
        self._w.write_movie_timeline(node_id, text)

    def set_sweep(self, node_id: str, value: str) -> None:
        self._w.set_movie_sweep(node_id, value)

    def live_display(self, node_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._w.live_display(node_id, spec)

    def capture(self, node_id: str) -> None:
        self._w.stamp_movie_links(node_id)

    def export(self, node_id: str) -> None:
        self._w.export_movie(node_id)


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
        # ── the side-by-side compare pane (V2.28) ──────────────────────────────
        # A SECOND ViewerPanel, created on first use (open_compare) and shown beside the
        # primary one in a horizontal splitter. `_viewed2` is the node it shows;
        # `_compare_linked` whether the two panes' M/T/Z extents match, in which case the
        # compare pane drops its own cursor row and the primary's strips move both.
        self.viewer2: Optional[ViewerPanel] = None
        self._viewed2: Optional[str] = None
        self._compare_box: Optional[_CompareBox] = None
        self._viewer_split: Optional[QSplitter] = None
        self._compare_linked = False

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

        # LabLink: whether this machine is serving the lab, and how to send work out to
        # another hub. Tabbed with the other two rather than given its own edge — it is
        # consulted occasionally, not watched while editing — and it starts BEHIND
        # Properties (the `idock.raise_()` below), so the dock exists without competing for
        # attention on every launch.
        self.lablink = LabLinkPanel()
        ldock = QDockWidget("LabLink", self)
        ldock.setWidget(self.lablink)
        ldock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, ldock)
        self.tabifyDockWidget(idock, ldock)
        self._lablink_dock = ldock
        self.lablink.send.load_into_graph.connect(self.lablink_load_result)
        idock.raise_()

        # Console (restored 2026-09-15, asked for by name: "errors should be in a console
        # that can be copy/pasted into"). It had been removed for the bottom-dock space,
        # leaving a failure as a status-bar line TRUNCATED to the terminal's width and a
        # Viewer tooltip — and a tooltip cannot be selected, so the one text a user actually
        # needs to send someone was the one text they could not copy.
        #
        # Starts HIDDEN so the reclaimed centre space is still the default layout, and
        # raises itself on the first failure (`_on_run_failed`) — the moment it is worth the
        # room. View ▸ Console toggles it by hand.
        self.console = ConsolePanel()
        cdock = QDockWidget("Console", self)
        cdock.setObjectName("console_dock")
        cdock.setWidget(self.console)
        cdock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, cdock)
        cdock.hide()
        self._console_dock = cdock
        self._console_shown = False

        # The Movie Editor (2026-09-30): a bottom dock tabbed with the Console that binds to
        # an Export Movie node when one is selected — asked for as "a movie editor when on
        # the node, rather than just a parameter list". Floatable, so it can live on a second
        # screen while the Viewer keeps the centre. It talks to the window only through the
        # `_MovieHost` adapter; the window owns the runner and the document.
        from nodelab_v2.movie_editor import MovieEditorPanel
        self.movie_editor = MovieEditorPanel(_MovieHost(self))
        mdock = QDockWidget("Movie Editor", self)
        mdock.setObjectName("movie_editor_dock")
        # Inside a scroll area, so the editor's own minimum size can never become the MAIN
        # WINDOW's: a dock that cannot shrink below its content grows the window instead,
        # and a maximized window then runs off the screen. Too small a dock scrolls.
        from PySide6.QtWidgets import QFrame, QScrollArea
        mscroll = QScrollArea()
        mscroll.setWidgetResizable(True)
        mscroll.setFrameShape(QFrame.NoFrame)
        mscroll.setWidget(self.movie_editor)
        mdock.setWidget(mscroll)
        mdock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea
                              | Qt.RightDockWidgetArea)
        mdock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                          | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                          | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        # NOT tabbed with the Console: a tabbed dock takes the tab group's size and ignores
        # `resizeDocks`, so it opened at the Console's sliver of height with a monitor
        # 200 px tall. Both visible at once simply share the bottom edge.
        #
        # The side columns own the bottom corners, so a bottom dock sits under the CANVAS
        # rather than under the full-height Properties/LabLink column. Spanning the width,
        # its height stacked on top of that column's minimum: showing the editor raised the
        # window's minimum height from 756 to 1023 px, which on a maximized 1080p window put
        # the dock — and every button in it — below the bottom of the screen.
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, mdock)
        mdock.hide()
        self._movie_dock = mdock
        # Keep the View ▸ Console tick honest when the dock is closed by its own ✕ or
        # raised by a failure — a menu tick that disagrees with what is on screen is the
        # same defect as a control that does nothing.
        cdock.visibilityChanged.connect(self._sync_console_action)

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
        self.view.files_dropped.connect(self._on_files_dropped)
        self.doc.on_change(self._on_doc_changed)
        self.doc.on_change(self.inspector.refresh_derived)   # G8 live ƒmd re-seed
        self.scene.pull_requested.connect(self.pull_node)
        self.scene.compare_requested.connect(self.open_compare)
        self.scene.nodes_deleted.connect(self._on_nodes_deleted)
        self.runner.started.connect(
            lambda nid: (self.statusBar().showMessage(f"pulling {nid}…"),
                         [p.show_running(nid) for p in self._panes_showing(nid)],
                         self._set_led("busy"),
                         self.minimap.set_state("busy")))
        # The Movie Editor's sources arrive on their own signal (a payload-only fetch never
        # reaches a pane), and the Viewer's settled LUT edits feed its linked channels.
        self.runner.fetched.connect(self._on_movie_fetched)
        self.runner.fetch_started.connect(
            lambda nid: self.statusBar().showMessage(
                f"computing {nid} for the Movie Editor…"))
        self.viewer.display_changed.connect(self._on_viewer_display)
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
        # A second branch requested while the first runs (2026-08-06): its cards go `queued`
        # and the status bar says how many are lined up, so a queued pull is visibly waiting
        # rather than indistinguishable from a click that did nothing.
        self.runner.queued.connect(self._on_run_queued)
        # ...and a run dropped by an edit inside its cone has to leave the running state, or
        # its card spins for a result that will never arrive.
        self.runner.cancelled.connect(self._on_run_cancelled)
        self.viewer.request_changed.connect(self._on_view_request)
        self.viewer.selection_changed.connect(self._on_frame_selection)
        self.viewer.region_changed.connect(self._on_region_changed)
        self.viewer.iteration_changed.connect(self._on_iteration_changed)
        # the overlay SOURCE strip (2026-09-30): a stepper is a display-only runner setting,
        # a pin is a graph edit — the viewer does neither itself
        self.viewer.overlay_step.connect(self._on_overlay_step)
        self.viewer.overlay_pin.connect(self._on_overlay_pin)
        # What the live surface can hold decides whether a big frame is shown WHOLE at full
        # resolution or off the pyramid (V2.23). The surface knows the number, the runner makes
        # the decision, and neither should know about the other.
        self.viewer.display_limits.connect(self.runner.set_display_limits)
        self.viewer.playing.connect(self._on_playing)
        self.runner.preload_progress.connect(self._on_preload_progress)
        self.runner.preload_finished.connect(self._on_preload_finished)
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
        self.inspector.movie_action.connect(self._on_movie_action)
        self.inspector.reload_requested.connect(self.reload_node_type)
        self.inspector.add_requested.connect(self._on_add_requested)
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
        self.viewer.own_layers_cb = self.doc.own_label_layers
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
        load = QAction("&Load ND2/TIFF file…", self)
        load.setShortcut("Ctrl+L")
        load.triggered.connect(self.file_load_source)
        m_file.addAction(load)

        seq = QAction("Load file &sequence…", self)
        seq.setShortcut("Ctrl+Shift+L")
        seq.setStatusTip("Pick one file of a numbered series; load the whole series as "
                         "one source and chain it onto Time, Z or Channels")
        seq.triggered.connect(self.file_load_sequence)
        m_file.addAction(seq)

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
        comp = QAction("&Compare selected beside viewed", self)
        comp.setShortcut("F8")
        comp.setToolTip(
            "Open the selected node's result in a second Viewer pane, side by side with "
            "the one being viewed.\n\n"
            "When both results span the same M/T/Z, the panes share ONE cursor — the "
            "primary's strips move both — and the compare pane drops its own. Results "
            "with different extents each keep their own strips. Also on a card's "
            "right-click menu.")
        comp.triggered.connect(self.compare_selected)
        m_run.addAction(comp)
        close_comp = QAction("Close compare &pane", self)
        close_comp.setShortcut("Shift+F8")
        close_comp.triggered.connect(self.close_compare)
        m_run.addAction(close_comp)
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
        self._hold_act = QAction("&Hold selected dock", self)
        self._hold_act.setShortcut("Ctrl+F6")
        self._hold_act.setToolTip(
            "Freeze everything above the selected Dock node IN MEMORY and stop evaluating it. "
            "Effectively instant and writes nothing — the troubleshooting counterpart of "
            "Bake.\n\n"
            "It does NOT free memory (the result is still held, it just stops being "
            "recomputed) and does NOT survive reopening the file. Bake does both.")
        self._hold_act.triggered.connect(self._hold_selected)
        m_run.addAction(self._hold_act)
        self._stop_bake_act = QAction("S&top bake", self)
        self._stop_bake_act.setToolTip(
            "Stop the bake that is running. Nothing is recorded, so the dock still reads as "
            "un-baked and the half-written folder is safe to bake over — the same state an "
            "interrupted bake leaves.")
        self._stop_bake_act.triggered.connect(self.runner.request_stop_bake)
        m_run.addAction(self._stop_bake_act)
        self._flatten_act = QAction("&Flatten to Large Image…", self)
        self._flatten_act.setShortcut("Shift+F6")
        self._flatten_act.setToolTip(
            "Write the viewed result to a chunked, pyramidal store and serve it from there — "
            "the same trade a microscope's own software makes for a stitched mosaic.\n\n"
            "A stitched canvas is recomputed for every frame you display: measured on the "
            "49-position WellA3 montage (7168²), 0.78 s per full-resolution frame live against "
            "0.06 s from a store, and the read-ahead goes from 2 frames to 8 because a store is "
            "bytes rather than a compute. One-time cost on that series: 29 s and about 1 GiB.\n\n"
            "This adds a Dock below the node and bakes it, so it is undoable (Un-dock) and the "
            "chain that produced it stays on the canvas. An overlay must be set to "
            "Output = resample first, since `display` mode stores no pixels by design — you "
            "will be offered the switch.")
        self._flatten_act.triggered.connect(lambda: self.flatten_to_large_image())
        m_run.addAction(self._flatten_act)
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
        clear_region = QAction("Clear troubleshooting &region", self)
        clear_region.setToolTip(
            "Drop the amber region box back to the whole frame — scoped pulls run on the "
            "full field again. The box itself is dragged on the Viewer while F9 is on: its "
            "edges and corners resize the window, its interior moves it.")
        clear_region.triggered.connect(self.clear_region)
        m_run.addAction(clear_region)

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
        m_graph.addSeparator()
        publish = QAction("&Publish as a LabLink recipe…", self)
        publish.setToolTip(
            "Offer this pipeline to a hub: choose which params a remote caller may turn, "
            "validate against this build's node catalogue, and submit it for an operator to "
            "install.")
        publish.triggered.connect(self.publish_recipe)
        m_graph.addAction(publish)

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
        self._console_act = QAction("&Console", self)
        self._console_act.setCheckable(True)
        self._console_act.setShortcut("Ctrl+`")
        self._console_act.setToolTip("A selectable log of run activity and the FULL text of "
                                     "any node failure, with Copy all — the status line is "
                                     "truncated and a tooltip cannot be copied")
        self._console_act.toggled.connect(self._toggle_console)
        m_view.addAction(self._console_act)
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
            "  • Correlation: DIC (pyALDIC, 2D) and DVC (pyALDVC, 3D) + cumulative "
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
                      self.minimap, self.welcome, self.view, self.lablink,
                      self.console, self.movie_editor):
            panel.restyle()
        if self._compare_box is not None:
            self._compare_box.restyle()   # restyles viewer2 with it
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
            # the compare pane cannot follow the Viewer into the mini-map — one HUD
            # frame, one panel — so maximizing ends the comparison rather than
            # stranding a headless second pane in the splitter
            self.close_compare()
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
            # `queue=False`: a preview shows what is already computed and otherwise does
            # nothing. It must never enqueue — this fires on every settled selection, so
            # queueing here would turn clicking around during a long run into a committed
            # backlog of pulls nobody asked for.
            self.pull_node(nid, allow_ingest=False, queue=False)

    # ── document plumbing ─────────────────────────────────────────────────────
    def _on_doc_changed(self) -> None:
        # Only the runs this edit could have changed (2026-08-06). `last_touched` is the node
        # the inspector or the card just wrote, or None for a structural edit that no single
        # node accounts for — in which case every in-flight pull still goes, as before. This
        # is what lets a finished branch be re-tuned while another branch is still computing.
        touched = self.doc.last_touched
        self.runner.invalidate(touched)
        # A terminal `done` badge now OUTLIVES an unrelated branch starting (so a finished
        # branch keeps saying so), which means the edit that actually invalidates a result has
        # to retire it — or the card claims a result that no longer describes the node. The
        # edited nodes and everything DOWNSTREAM of them are what a param change can alter;
        # an unscoped edit retires the lot.
        if touched is None:
            self.scene.clear_run_states()
        elif touched:
            self.scene.clear_run_states_for(self.doc.downstream_of(touched))
        if self._viewed is not None and self._viewed not in self.doc.nodes:
            self._viewed = None           # the previewed node was deleted/reloaded away
            self.scene.set_viewed(None)
            self.minimap.set_state("idle")
            self._sync_minimap_title()
        if self._viewed2 is not None and self._viewed2 not in self.doc.nodes:
            self.close_compare()          # its node was deleted — an empty pane lies
        self._sync_solo(self._viewed)     # a rewired source changes the frame count
        self.movie_editor.on_doc_changed(touched, self.doc.downstream_of)
        name = self.doc.path or "untitled"
        self.statusBar().showMessage(f"{name} — rev {self.doc.revision}")

    def _on_selection(self) -> None:
        try:
            sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        except RuntimeError:
            return          # scene torn down (app closing) — the C++ object is gone
        self.inspector.set_node(sel[0] if sel else None)
        # an Export Movie node brings up its editor; any other selection leaves it bound
        if len(sel) == 1 and sel[0].rec.op_key == MOVIE_OP:
            self.open_movie_editor(sel[0].node_id)
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
        if req.kind == "nudge_xy":
            # a two-click nudge commits a CHANGE — the session needs the nudge already set
            from dataclasses import replace as _dc_replace
            rec = self.doc.nodes.get(req.node_id)
            prm = rec.params if rec is not None else {}
            from nodegraph.placement import canvas_flip
            req = _dc_replace(req, base=tuple(
                (n, float(prm.get(n, 0.0) or 0.0)) for n in req.bounds),
                mirror=canvas_flip(md) if md.get("canvas_flip") is not None else None)
        self._open_viewer()
        self.viewer.arm_pick(req, Calibration.from_metadata(md))

    def _on_pick_armed(self, on: bool) -> None:
        if on:
            self.statusBar().showMessage("Picking — Esc cancels, Enter applies")
        else:
            self.statusBar().clearMessage()

    def _sync_overlay_frames(self, pane, node_id: str) -> None:
        """Hand ``pane`` what each overlaid source is showing at its cursor, and how many
        Play-all ticks one primary frame is split into. Cheap (metadata arithmetic), and run
        on every delivery so the strip reads the frame actually on screen."""
        try:
            m, t, z, _c = pane.coords()
            sub = pane.sub() if hasattr(pane, "sub") else 0
            rows = self.runner.overlay_frame_readout(node_id, m, t, sub, z)
            pane.set_overlay_frames(rows, self.runner.overlay_sub_ticks(node_id))
        except Exception:  # noqa: BLE001 — a readout must never cost the frame
            pass

    def _on_overlay_step(self, ovl_id: str, dt: int, dz: int) -> None:
        """A source's ◀▶ stepper: move that source's DISPLAYED frame by ``(dt, dz)`` from
        its mapped one. A runner setting, not an edit — nothing re-runs, nothing is saved —
        so the user can hunt for the frame that goes with this one before pinning it."""
        self.runner.set_source_override(ovl_id, int(dt), int(dz))
        self._on_view_request()

    def _on_overlay_pin(self, ovl_id: str, axis: str, pri: int, sec: int) -> None:
        """Pin T / Pin Z: "the primary's current frame goes with THIS frame of the source".

        Written into the Overlay's ``t_pins`` / ``z_pins`` through the same lines a typed edit
        runs (the value plus its sticky pin), so it serializes, diffs and keys the memo like
        any param. The row records each file's clock (T) or absolute focus (Z) beside the
        indices, which is what lets it survive an upstream crop re-numbering the frames. A
        pin at a primary frame that already had one replaces it; the source's stepped offset
        is dropped, since the pin now makes the mapping land where the stepper was."""
        from nodegraph.placement import parse_pins, pins_json
        rec = self.doc.nodes.get(ovl_id)
        if rec is None or axis not in ("t", "z") or self._viewed is None:
            return
        name = f"{axis}_pins"
        try:
            rows = list(parse_pins(rec.params.get(name, ""), axis=axis))
        except ValueError as exc:
            self.statusBar().showMessage(f"{ovl_id}: cannot add a pin — {exc}", 6000)
            return
        m = self.viewer.coords()[0]
        a_pri, a_sec = self.runner.overlay_pin_anchors(self._viewed, ovl_id, axis, m,
                                                       int(pri), int(sec))
        rows = [r for r in rows if int(r[0]) != int(pri)]
        rows.append((int(pri), int(sec), a_pri, a_sec))
        try:
            text = pins_json(rows)
        except ValueError as exc:
            self.statusBar().showMessage(f"{ovl_id}: pin refused — {exc}", 6000)
            return
        rec.params[name] = text
        rec.set_locked(rec.locked | {name})
        self.runner.set_source_override(ovl_id, 0, 0)
        self.doc.touch(ovl_id)
        item = self.scene.node_items.get(ovl_id)
        if item is not None:
            item.refresh()
            item.changed.emit(item)
        self.statusBar().showMessage(
            f"{ovl_id}: pinned primary {axis}={pri} to source {axis}={sec}"
            + ("" if a_pri is not None else " (by index — no clock/focus to anchor it)"), 5000)

    def _on_add_requested(self, node_id: str, op_key: str, wire_to: str) -> Optional[str]:
        """A *Ready to run* suggestion was taken: add ``op_key`` and wire it into
        ``node_id``'s ``wire_to`` input (2026-10-02). Returns the new node's id.

        Two placements, decided by which input the suggestion feeds:

        * the node's PRIMARY input (a missing domain, e.g. Measure wants Label) — the new
          node is INSERTED on the wire: whatever fed ``data`` now feeds the new node, and
          the new node feeds ``data``. That is what "add Connected Components before this"
          means on a canvas.
        * a SIDE input (Subtract Background's ``regions``) — the new node is fed from the
          same source as this node's primary input, so it draws on the same image, and its
          output goes into the side socket. The primary wire is untouched.

        The new node lands left of the target, one row down for a side input so it does
        not cover the primary's source. A refused wire (a cycle, a type mismatch) is
        reported on the status bar and leaves the node placed but unwired — the user can
        see what happened and finish it by hand."""
        rec = self.doc.nodes.get(node_id)
        spec = rec.spec() if rec is not None else None
        if spec is None:
            return None
        from nodegraph.sockets import SocketType as _ST
        ds_ins = [s for s in spec.inputs if s.type is _ST.DATASET
                  and not getattr(s, "view_source", False)]
        primary = ds_ins[0].name if ds_ins else wire_to
        side = wire_to != primary
        new = self.doc.add_node(op_key, x=float(rec.x) - 270.0,
                                y=float(rec.y) + (120.0 if side else 0.0))
        nspec = new.spec()
        new_in = next((s.name for s in getattr(nspec, "inputs", ())
                       if s.type is _ST.DATASET), None) if nspec is not None else None
        new_out = next((s.name for s in getattr(nspec, "outputs", ())
                        if s.type is _ST.DATASET), "out") if nspec is not None else "out"
        feeders = [e for e in self.doc.edges if e[2] == node_id and e[3] == primary]
        try:
            if side:
                if new_in is not None and feeders:
                    self.doc.connect(feeders[0][0], feeders[0][1], new.id, new_in)
            else:
                for e in feeders:
                    self.doc.disconnect(*e)
                    if new_in is not None:
                        self.doc.connect(e[0], e[1], new.id, new_in)
            self.doc.connect(new.id, new_out, node_id, wire_to)
            msg = f"added {nspec.label if nspec else op_key} → {node_id}.{wire_to}"
        except ValueError as exc:
            msg = f"added {op_key} but could not wire it: {exc}"
        self.doc.touch()
        if hasattr(self.scene, "sync"):
            self.scene.sync()
        item = self.scene.node_items.get(node_id)
        if item is not None:
            item.refresh()
            item.changed.emit(item)        # the inspector rebuilds: the problem is gone
        self.statusBar().showMessage(msg)
        return new.id

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

    def _on_files_dropped(self, paths: list, pos: QPointF, target: str) -> None:
        """Image files dragged from the desktop onto the canvas (V3.01).

        Two gestures, told apart by WHERE they land:

        * **on a Batch point** — one source card per file, each wired straight into that
          point. This is the gesture the golden point exists for: drop four files on it
          and the pipeline already drawn downstream now runs over four files.
        * **anywhere else** — one source card per file, unwired, exactly as
          File -> Load would have made them.

        Dropping several files on empty canvas does NOT silently build a batch. The cards
        are what the user asked for; wiring them into something they did not place would be
        inventing a pipeline. Making the batch is one more drag onto the point, and that
        drag is the decision.
        """
        from PySide6.QtWidgets import QMessageBox
        made, failed = [], []
        x, y = pos.x(), pos.y()
        for i, p in enumerate(paths):
            try:
                rec, _axes = self._add_source_node(p, x, y + i * 96.0)
                made.append(rec)
            except Exception as exc:                      # noqa: BLE001 - reported below
                failed.append((p, exc))
        if target and made:
            # newest first would reverse the batch's member order, and member order is
            # the order Unbatch hands results back in — so wire in the order dropped
            for rec in made:
                self.doc.connect(rec.id, "image", target, "data")
        if failed:
            import os
            lines = "\n".join(f"{os.path.basename(p)} — {exc}" for p, exc in failed)
            QMessageBox.warning(
                self, "Some files could not be loaded",
                f"{len(made)} of {len(paths)} loaded.\n\n{lines}")
        if made:
            self.statusBar().showMessage(
                f"loaded {len(made)} file(s)"
                + (" into the batch point" if target else ""), 6000)

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
                "no source nodes — File → Load ND2/TIFF file… first")
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

    def pull_node(self, node_id: str, *, allow_ingest: bool = True,
                  queue: bool = True) -> None:
        """Double-click / F5 on a card: compute it and show it in the Viewer.

        On a **source** card whose file has not been ingested yet, this starts that file's
        ingest instead (:meth:`~nodelab_v2.runner.EngineRunner.ingest_source`) and views it
        once it lands. The difference is which lane the work runs in: pulls are one at a
        time (queued), so ingesting through it meant a second file could not start until the
        first had finished — and on a 40-minute ND2 that is the whole session.
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
        self.runner.pull(node_id, self.viewer.coords(), self.viewer.channels(),
                         queue=queue)

    # ── the side-by-side compare pane (V2.28) ─────────────────────────────────
    def _panes_showing(self, node_id: str) -> List:
        """Which Viewer pane(s) a delivery for ``node_id`` lands in.

        The primary pane keeps its historic contract — it shows whatever finishes
        (first-to-land is viewable while the rest queue) — EXCEPT a result that belongs
        exclusively to the compare pane. A node shown in both panes gets both."""
        panes: List = []
        if node_id == self._viewed or node_id != self._viewed2:
            panes.append(self.viewer)
        if (self._viewed2 is not None and node_id == self._viewed2
                and self.viewer2 is not None):
            panes.append(self.viewer2)
        return panes

    def compare_selected(self) -> None:
        """Run → *Compare selected beside viewed* (F8)."""
        sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            self.statusBar().showMessage(
                "select a node first — F8 opens it beside the viewed one")
            return
        self.open_compare(sel[0].node_id)

    def open_compare(self, node_id: str) -> None:
        """Open ``node_id``'s result in a second Viewer pane, side by side with the
        primary one.

        The two panes' cursors LINK automatically when both results span the same
        M/T/Z — the compare pane then drops its own strips and the primary's move both
        — and stay independent otherwise (:meth:`_sync_compare_link`). The pane is a
        display surface only: picks, the troubleshooting scope and the iteration strip
        stay with the primary pane."""
        if node_id not in self.doc.nodes:
            self.statusBar().showMessage(f"{node_id} is not on the canvas")
            return
        state = self.runner.source_state(node_id)
        if state in ("cold", "running", "missing"):
            self.statusBar().showMessage(
                f"{self._source_label(node_id)} is not ingested yet — double-click its "
                f"card first, then compare" if state != "missing" else
                f"{self._source_label(node_id)}: no such file — fix the card's path")
            return
        if self._maximized:
            self.set_maximized(False)   # the pane lives in the splitter the HUD vacated
        self._ensure_compare_ui()
        self._viewed2 = node_id
        self._compare_linked = False
        self.viewer2.set_axes_hidden(False)
        self._sync_compare_title()
        self._sync_solo(self._viewed)          # ranges the compare pane's chooser too
        self._open_viewer()
        self.runner.set_frame_selection(*self.viewer.frame_selection())
        self.runner.pull(node_id, self.viewer2.coords(), self.viewer2.channels())

    def close_compare(self) -> None:
        """Put the Viewer back to one pane. The compare panel is kept (hidden) for the
        next open, so its LUTs and channel choices survive a close/reopen."""
        opened = self._viewer_split is not None and self._viewer_split.parent() is not None
        if self._viewed2 is None and not opened:
            return
        self._viewed2 = None
        self._compare_linked = False
        if opened:
            sizes = self._center.sizes()
            self._center.insertWidget(0, self.viewer)   # reparents the panel back
            self._viewer_split.setParent(None)          # takes the compare box with it
            self._center.setSizes(sizes)
        if self.viewer2 is not None:
            self.viewer2.set_axes_hidden(False)
        self.statusBar().showMessage("compare pane closed")

    def _ensure_compare_ui(self) -> None:
        """Build the second pane on first use; re-attach it on every later open."""
        if self.viewer2 is None:
            self.viewer2 = ViewerPanel()
            self.viewer2.request_changed.connect(self._on_view_request2)
            self.viewer2.display_limits.connect(self.runner.set_display_limits)
            # the same window-owned callbacks the primary pane gets — they are all
            # per-node, so both panes share them
            self.viewer2.raw_plane_cb = self.runner.raw_plane
            self.viewer2.detail_cb = self._request_detail
            self.viewer2.own_layers_cb = self.doc.own_label_layers
            self.runner.detail_ready.connect(self.viewer2.on_detail_ready)
            self._compare_box = _CompareBox(self.viewer2, self.close_compare)
            self._viewer_split = QSplitter(Qt.Horizontal)
            self._viewer_split.setChildrenCollapsible(False)
        if self._viewer_split.parent() is None:
            sizes = self._center.sizes()
            self._center.insertWidget(0, self._viewer_split)
            # primary at index 0 ALWAYS — on a reopen the compare box is still inside
            # the splitter from last time, so a plain addWidget would append the
            # primary pane after it and flip the two panes
            self._viewer_split.insertWidget(0, self.viewer)  # reparents from the centre
            if self._compare_box.parent() is not self._viewer_split:
                self._viewer_split.addWidget(self._compare_box)
            self._compare_box.show()
            self._viewer_split.show()
            self._center.setSizes(sizes)
            w = max(2, self._viewer_split.width())
            self._viewer_split.setSizes([w // 2, w // 2])

    def _sync_compare_link(self) -> None:
        """Link or unlink the two panes' cursors from their METADATA: linked iff both
        panes hold a result and the M/T/Z extents agree, per the payloads' own axes.
        Linked, the compare pane's cursor row disappears (one set of sliders, moving
        both); unlinked, it keeps its own. Re-derived after every delivery, because a
        re-pull can change either side's extents."""
        if self._viewed2 is None or self.viewer2 is None:
            return
        a, b = self.viewer.axes(), self.viewer2.axes()
        linked = (a is not None and b is not None
                  and (a.m, a.t, a.z) == (b.m, b.t, b.z))
        self._compare_linked = linked
        self.viewer2.set_axes_hidden(linked)
        self._sync_compare_title()
        if linked:
            m, t, z, _c = self.viewer.coords()
            m2, t2, z2, _c2 = self.viewer2.coords()
            if (m, t, z) != (m2, t2, z2):
                self.viewer2.set_cursor(m, t, z)
                self.runner.request_plane(self._viewed2, self.viewer2.coords(),
                                          self.viewer2.channels())

    def _sync_compare_title(self) -> None:
        if self._compare_box is None:
            return
        if self._viewed2 is None:
            self._compare_box.set_title("compare")
            return
        rec = self.doc.nodes.get(self._viewed2)
        spec = rec.spec() if rec is not None else None
        label = spec.label if spec else (rec.op_key if rec is not None else "?")
        tag = ("linked — one cursor moves both panes" if self._compare_linked
               else "own cursor (different M/T/Z)")
        self._compare_box.set_title(f"compare · {self._viewed2} · {label} — {tag}")

    def _on_view_request2(self) -> None:
        """The compare pane's cursor or channel set moved (its own strips are only
        visible while UNLINKED; its channel toggles fire this either way)."""
        if self._viewed2 is not None and self.viewer2 is not None:
            self.runner.request_plane(self._viewed2, self.viewer2.coords(),
                                      self.viewer2.channels())

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
        self._hold_act.setEnabled(nid is not None)
        self._hold_act.setText(f"&Hold dock “{nid}”" if nid else "&Hold selected dock")
        # Stop is offered only while a bake is actually in flight — an always-live Stop that
        # does nothing is the same defect as a socket the kernel ignores.
        baking = bool(self.runner.busy and self.runner.baking)
        self._stop_bake_act.setEnabled(baking)
        # Flatten names the node it would act on, because the answer to "what does this apply
        # to" is the viewed node and that is not where the mouse is.
        view = self._viewed
        self._flatten_act.setEnabled(view is not None and view in self.doc.nodes)
        self._flatten_act.setText(f"&Flatten “{view}” to a Large Image…" if view
                                  else "&Flatten to Large Image…")

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

    def closeEvent(self, event) -> None:                     # noqa: N802 — Qt override
        """Join the LabLink panel's threads before Qt tears its widgets down.

        The panel polls a local hub on a timer and runs every hub call on a one-shot
        ``QThread``. A thread still running when the interpreter destroys the Qt object that
        parents it is a crash on exit — and it would land at the worst possible moment, when
        the user has already asked to quit and has no way to read the traceback.
        Best-effort: a failure here must not prevent the window from closing.
        """
        try:
            self.lablink.shutdown()
        except Exception:                                    # noqa: BLE001 — see above
            pass
        try:
            self.movie_editor.shutdown()     # its render thread, for the same reason
        except Exception:                                    # noqa: BLE001 — see above
            pass
        super().closeEvent(event)

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

    def _hold_selected(self) -> None:
        nid = self._selected_dock()
        if nid is None:
            QMessageBox.information(
                self, "No dock selected",
                "Select a Dock Data node first (or add one from the Nodes palette, under "
                "io).\n\nHolding one freezes what the chain above it last produced in "
                "memory and stops evaluating that chain — instant, and it writes nothing. "
                "Bake the same node instead when you need the memory back or need the "
                "result to survive reopening the file.")
            return
        self._hold_dock(nid)

    # ── flatten to a Large Image (V2.23) ──────────────────────────────────────
    #
    # The Nikon-shaped half of "I want the stitched and overlay to be full resolution and
    # instantly loaded/playable". The live path answers the full-resolution part directly (see
    # `EngineRunner.display_dim`), but a stitched canvas still costs a stitch per frame — 0.78 s
    # on the WellA3 montage against 0.06 s from a store. NIS-Elements does not keep a mosaic
    # live either: it produces a Large Image with the pyramid IN the file and browses that.
    # A Dock already writes exactly that artifact; what was missing was one gesture that sets
    # it up, and the refusal below.

    def _display_overlays_above(self, node_id: str) -> List[str]:
        """Overlay nodes feeding ``node_id`` that are recording a placement for the VIEWER
        rather than writing pixels — the ones a bake would silently drop.

        Walks the document's own edges rather than the run graph, so it sees the chain the way
        the canvas draws it."""
        seen, stack, out = set(), [node_id], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            rec = self.doc.nodes.get(cur)
            if rec is None:
                continue
            if rec.op_key == "view.overlay" and \
                    (rec.modes or {}).get("output", "display") == "display":
                out.append(cur)
            stack.extend(src for src, _ss, dst, _ds in self.doc.edges if dst == cur)
        return sorted(out)

    def _settle_display_overlays(self, node_id: str, what: str) -> bool:
        """Make sure a bake of ``node_id`` cannot quietly lose an overlay. Returns whether to
        proceed.

        An overlay in ``display`` mode stores NOTHING — that is its contract, and the reason
        dragging its opacity is a repaint rather than a re-run. So a bake of a chain containing
        one writes a store with no secondary in it, and viewing that store shows the primary
        alone: the overlay does not degrade, it vanishes, and the graph still says it is there.
        ``resample`` is the mode that writes the placed secondary as a real channel, which is
        also what makes it measurable — the same flatten-then-it-is-just-an-image step
        NIS-Elements takes.

        Offered rather than done: switching the mode changes what the node OUTPUTS, so it is
        the user's call. Declining cancels, because the alternative is a silent loss."""
        ovl = self._display_overlays_above(node_id)
        if not ovl:
            return True
        names = ", ".join(ovl)
        ans = QMessageBox.question(
            self, "Overlay would not be baked",
            f"{'This overlay is' if len(ovl) == 1 else 'These overlays are'} set to "
            f"Output = display: {names}.\n\n"
            f"That mode records WHERE the secondary goes and deliberately stores no pixels — "
            f"the Viewer composites it at draw time. A {what} therefore writes the primary "
            f"only, and opening the result would show no overlay at all.\n\n"
            f"Switch {'it' if len(ovl) == 1 else 'them'} to Output = resample, so the placed "
            f"secondary is written as a real extra channel? That is also what makes the "
            f"overlaid pixels measurable.\n\n"
            f"Choosing No cancels — a {what} that silently drops the overlay is worse than "
            f"no {what}.",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return False
        for nid in ovl:
            self.doc.nodes[nid].modes["output"] = "resample"
        self.doc.touch()
        self.runner.invalidate()
        self.statusBar().showMessage(
            f"{names}: Output → resample — the secondary is now written as a real channel")
        return True

    def flatten_to_large_image(self, node_id: Optional[str] = None) -> None:
        """Dock the chain at ``node_id`` (default: the viewed node) so it is served from a
        chunked, pyramidal store instead of recomputed per frame.

        Measured on the WellA3 GFP montage (49 positions → 7168², 16 T): 0.78 s per
        full-resolution frame live against 0.06 s from the store, 0.21 s against 0.02 s for the
        overview, and the prefetcher's lookahead goes from 2 frames to 8 because the store is
        bytes rather than a compute. One-time cost 29 s and 0.97 GiB.

        Reuses a Dock already wired below the node rather than adding a second one — the point
        is one artifact per result, not a Dock per press."""
        node_id = node_id or self._viewed
        if node_id is None or node_id not in self.doc.nodes:
            self.statusBar().showMessage("Select the node whose result you want flattened")
            return
        existing = [dst for src, _ss, dst, ds in self.doc.edges
                    if src == node_id and ds == "data"
                    and self.doc.nodes.get(dst) is not None
                    and self.doc.nodes[dst].op_key == DOCK_OP]
        if existing:
            self._start_bake(existing[0])
            return
        rec = self.doc.nodes[node_id]
        # uint16 when the payload still says it is camera counts, float32 otherwise: the
        # precision Mode has no default on purpose (see `_start_bake`), and this is the one
        # place that can answer it from the data instead of asking.
        precision = "uint16" if self._integer_payload(node_id) else "float32"
        dock = self.doc.add_node(DOCK_OP, x=rec.x + 240.0, y=rec.y,
                                 modes={"precision": precision})
        try:
            self.doc.connect(node_id, "out", dock.id, "data")
        except Exception as exc:                  # noqa: BLE001 — leave no orphan behind
            self.doc.remove_node(dock.id)
            QMessageBox.information(self, "Cannot flatten",
                                    f"{node_id}'s output cannot feed a Dock: {exc}")
            return
        self.scene.sync()
        self.statusBar().showMessage(
            f"added {dock.id} below {node_id} (precision {precision}) — baking it now")
        self._start_bake(dock.id)

    def _integer_payload(self, node_id: str) -> bool:
        """Whether ``node_id``'s result is still an integer image, read off its PROPAGATED
        envelope — the same ``doc.env`` a pick reads its calibration from, so the question is
        answered without pulling anything.

        ``bit_depth`` is the payload's own claim to be camera counts, and every node that makes
        its values continuous drops it. That is exactly the distinction ``uint16`` needs: the
        bake refuses uint16 on normalized data, so guessing wrong here would be a dialog rather
        than a wrong store — but guessing right means the user is not asked at all."""
        try:
            bd = dict(self.doc.env(node_id).metadata or {}).get("bit_depth")
        except Exception:  # noqa: BLE001 — an un-propagated node: fall back to float32
            return False
        return (isinstance(bd, (int, float)) and not isinstance(bd, bool)
                and 0 < int(bd) <= 16)

    def _on_dock_action(self, node_id: str, action: str) -> None:
        """Everything a Dock node's buttons and menu can ask for."""
        rec = self.doc.nodes.get(node_id)
        if rec is None or rec.op_key != DOCK_OP:
            return
        if action in ("bake", "bake_scoped"):
            self._start_bake(node_id, scoped=(action == "bake_scoped"))
        elif action == "hold":
            self._hold_dock(node_id)
        elif action == "release":
            self.runner.release(node_id)
            self.doc.set_dock_hold(node_id, False)
            self.doc.set_held_nodes(self.runner.held)
            self.runner.invalidate()
            self.statusBar().showMessage(
                f"{node_id} released — the chain above runs live again")
            if self._viewed is not None:
                self.pull_node(self._viewed)
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
        """The Iterate card whose iteration strip belongs to ``node_id``.

        Two nodes qualify (V2.22): the card itself, and the **end of its segment** — the
        node whose id the rewrite's selector wears, and therefore the one place downstream
        where the payload really is a choice between iterations. A node in the MIDDLE of the
        segment does not: it is served by one clone (see
        :meth:`~nodelab_v2.document.GraphDocument.iterate_aliases`), so there is nothing to
        select there and a strip would imply otherwise."""
        return self.doc.iterate_card_at(node_id or "")

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
        # Re-pull what is being VIEWED, not the card: the strip normally appears while the
        # user is looking at the segment's end node, and re-pulling the card there would
        # move the viewer off the node they are tuning to answer a question they asked
        # about it.
        self.pull_node(self._viewed or owner)

    def _on_movie_action(self, node_id: str, action: str) -> None:
        """The inspector's *Open Movie Editor*: raise the dock on ``node_id``."""
        if action not in ("edit", "preview"):
            return
        rec = self.doc.nodes.get(node_id)
        if rec is None or rec.op_key != MOVIE_OP:
            return
        self.open_movie_editor(node_id)

    def open_movie_editor(self, node_id: str) -> None:
        """Show the Movie Editor dock bound to Export Movie node ``node_id``.

        The editor previews the node's SOURCES, never the node: pulling ``io.write_movie``
        runs its compute, and that compute WRITES THE FILE, so a preview that exported the
        movie in order to show it would be the opposite of the feature."""
        first = self._movie_dock.isHidden() and not getattr(self, "_movie_dock_sized", False)
        self._movie_dock.show()
        self._movie_dock.raise_()
        if first:
            # the first time only: room for a monitor. After that the user's size stands.
            self._movie_dock_sized = True
            # `resizeDocks` on a dock that has just been shown is discarded, and the dock
            # area otherwise opens it at its MINIMUM height (a 120 px monitor). So hold a
            # minimum for one layout pass — the separator settles there — then release it,
            # which leaves the size in place and lets the user drag it smaller again.
            # Clamped to what the centre can actually give up: a minimum larger than that
            # would GROW the main window, and a maximized one would then run off the screen.
            spare = (self.centralWidget().height()
                     - self.centralWidget().minimumSizeHint().height())
            want = min(int(self.height() * 0.42),
                       self._movie_dock.height() + max(0, spare - 8))
            if want > self._movie_dock.height():
                self._movie_dock.setMinimumHeight(want)
                QTimer.singleShot(0, lambda: self._movie_dock.setMinimumHeight(0))
        self.movie_editor.bind(node_id)

    def _on_movie_fetched(self, node_id: str, payload, _seconds: float) -> None:
        self.scene.finish_run(node_id)
        self._set_led("idle")
        self.movie_editor.on_fetched(node_id, payload)
        self.statusBar().showMessage(f"{node_id} ready for the Movie Editor")

    def _on_viewer_display(self, node_id: str) -> None:
        """The Viewer's look of ``node_id`` settled: stamp it into every Export Movie whose
        linked panels read that node, then let the editor re-render."""
        for mid in self._movie_nodes():
            if self._movie_links_to(mid, node_id):
                self.stamp_movie_links(mid)
        self.movie_editor.on_display_changed()

    # ── Export Movie ↔ Viewer LUT link ────────────────────────────────────────────
    def _movie_nodes(self) -> List[str]:
        return [n for n, r in self.doc.nodes.items() if r.op_key == MOVIE_OP]

    def _movie_sources(self, movie_id: str) -> Dict[str, Dict[str, Any]]:
        """``{letter: {"node", "label", "env"}}`` for an Export Movie's three inputs, each
        the REAL node feeding it (through reroutes and muted nodes)."""
        from nodegraph.catalog._shared.movie_timeline import SOURCE_SOCKETS
        out: Dict[str, Dict[str, Any]] = {}
        for letter, socket in SOURCE_SOCKETS.items():
            src = self.doc.real_source(movie_id, socket)
            if src is None:
                out[letter] = {"node": None}
                continue
            rec = self.doc.nodes.get(src)
            spec = rec.spec() if rec is not None else None
            label = f"{spec.label if spec is not None else rec.op_key} ({src})"
            try:
                env = self.doc.env(src)
            except Exception:              # noqa: BLE001 — an un-propagated node
                env = None
            out[letter] = {"node": src, "label": label, "env": env}
        return out

    def _movie_links_to(self, movie_id: str, node_id: str) -> bool:
        rec = self.doc.nodes.get(movie_id)
        if rec is None or str(rec.modes.get("sweep", "time")) != "timeline":
            return False
        srcs = self._movie_sources(movie_id)
        return any((i or {}).get("node") == node_id for i in srcs.values()) \
            or node_id == movie_id

    def live_display(self, movie_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        """``spec`` with every Viewer-linked channel filled in from the Viewer, NOT written.

        Each linked channel takes its panel source's node's LUT; a channel the Viewer holds
        nothing for on that node falls back to the movie node itself (a tap: viewing it
        shows source A), and past that keeps whatever values it already had — the stamp from
        last time, or none, which renders as auto contrast."""
        import copy as _copy
        out = _copy.deepcopy(spec)
        srcs = self._movie_sources(movie_id)
        states: Dict[str, Dict[int, Dict[str, Any]]] = {}

        def state_of(node: Optional[str], letter: str) -> Dict[int, Dict[str, Any]]:
            key = f"{letter}:{node}"
            if key not in states:
                env = (srcs.get(letter) or {}).get("env")
                names = list((getattr(env, "metadata", {}) or {}).get("channel_names")
                             or [])
                got = self.viewer.display_state(node, names) if node else {}
                if letter == "A":
                    for c, d in self.viewer.display_state(movie_id, names).items():
                        got.setdefault(c, d)
                states[key] = got
            return states[key]

        def panels():
            for seg in out.get("segments", []):
                clips = seg.get("body", []) if seg.get("kind") == "loop" else [seg]
                for clip in clips:
                    for p in clip.get("panels", []):
                        yield p

        for p in panels():
            letter = p.get("source", "A")
            node = (srcs.get(letter) or {}).get("node")
            for c, d in (p.get("display") or {}).items():
                if d.get("link") != "viewer":
                    continue
                got = state_of(node, letter).get(int(c))
                if not got:
                    continue
                for k in ("lo", "hi", "gamma", "rgb"):
                    if k in got:
                        d[k] = got[k]
        return out

    def stamp_movie_links(self, movie_id: str) -> bool:
        """Write the Viewer's current look into ``movie_id``'s linked channels. ``True`` if
        the node's timeline changed.

        The COMMIT half of the live link: the values land in the saved graph, so a headless
        or LabLink run of this movie wears the LUTs the user tuned here. Deferred while the
        movie node itself is being computed, because an edit inside a running export's cone
        would cancel it; it is retried a moment later."""
        from nodegraph.catalog._shared.movie_timeline import canonical_json, try_normalize
        rec = self.doc.nodes.get(movie_id)
        if rec is None or str(rec.modes.get("sweep", "time")) != "timeline":
            return False
        spec, _err = try_normalize(rec.params.get("timeline", "") or "")
        if spec is None:
            return False
        text = canonical_json(self.live_display(movie_id, spec))
        if text == rec.params.get("timeline"):
            return False
        if self.runner.in_flight(movie_id):
            QTimer.singleShot(1500, lambda m=movie_id: self.stamp_movie_links(m))
            return False
        self.write_movie_timeline(movie_id, text)
        return True

    def write_movie_timeline(self, movie_id: str, text: str) -> None:
        """The one write path for an Export Movie's ``timeline``: the value, the lock that
        tells a re-seed the user owns it, and the narrowed touch — the same three steps an
        inspector edit takes (``node_item._write_param``)."""
        rec = self.doc.nodes.get(movie_id)
        if rec is None:
            return
        rec.params["timeline"] = text
        rec.set_locked(rec.locked | {"timeline"})
        self.doc.touch(movie_id)

    def set_movie_sweep(self, movie_id: str, value: str) -> None:
        rec = self.doc.nodes.get(movie_id)
        if rec is None or rec.modes.get("sweep") == value:
            return
        item = self._node_item(movie_id)
        if item is not None:
            item._write_mode("sweep", value)     # THE mode-write path: re-gates the card
        else:
            rec.modes["sweep"] = value
            self.doc.touch(movie_id)

    def export_movie(self, movie_id: str) -> None:
        """Stamp the linked LUTs, make sure there is a file to write, and pull the node."""
        rec = self.doc.nodes.get(movie_id)
        if rec is None:
            return
        self.stamp_movie_links(movie_id)
        if not str(rec.params.get("path", "") or "").strip():
            path, _f = QFileDialog.getSaveFileName(
                self, "Export movie to", "movie.mp4",
                "MP4 video (*.mp4);;Animated GIF (*.gif);;PNG sequence (*.png);;"
                "JPEG sequence (*.jpg)")
            if not path:
                return
            rec.params["path"] = path
            rec.set_locked(rec.locked | {"path"})
            self.doc.touch(movie_id)
        self.pull_node(movie_id)

    def _node_item(self, node_id: str):
        for it in self.scene.items():
            if isinstance(it, NodeItem) and it.node_id == node_id:
                return it
        return None

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
        md = getattr(payload, "metadata", None)
        if not isinstance(md, dict):
            return
        rows = md.get(SWEEP_ROWS_KEY)
        if not rows:
            return
        # WHOSE table this is comes off the payload, not off the node that produced it: the
        # selector wears the segment END's id (V2.22), so the pull that carries a results
        # table is usually of an ordinary analysis node. The card named in the payload is
        # the one whose panel shows the table and whose `index` the strip edits.
        owner = str(md.get(SWEEP_OWNER_KEY) or node_id)
        rec = self.doc.nodes.get(owner)
        if rec is None or rec.op_key != ITERATE_OP:
            return
        rec.params[SWEEP_KEY] = {"rows": [
            {"iter": r.get("iter"), "metric": r.get("metric"), "won": r.get("won")}
            for r in rows]}
        shown = getattr(self.inspector, "_node", None)
        if shown is not None and getattr(shown, "node_id", None) == owner:
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
        if not self._settle_display_overlays(node_id, "bake"):
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

    def _hold_dock(self, node_id: str) -> None:
        """Freeze the chain above ``node_id`` in memory (the session tier).

        No confirmation dialog, unlike Bake: this writes nothing, deletes nothing and is
        reversible with one click, so a modal would be pure friction on the action whose
        entire value is that it is instant. The status line says what it did and names the
        limit that matters."""
        if not self.runner.start_hold(node_id, signature=self.doc.dock_signature(node_id)):
            self.statusBar().showMessage(
                "a pull is already running — wait for it to finish, then hold")
            return
        self.statusBar().showMessage(f"holding {node_id} …")

    def _on_held(self, node_id: str, spec: dict) -> None:
        """Record a finished hold: pin the payload, flip the mode, grey the chain."""
        self.runner.hold(node_id, spec["payload"], spec.get("env"))
        self.doc.set_dock_hold(node_id, True)
        self.doc.set_held_nodes(self.runner.held)
        self.runner.invalidate()
        self.statusBar().showMessage(
            f"{node_id} held in memory — the chain above is frozen and greyed out. "
            f"Nothing was written, so this does NOT free memory and does NOT survive "
            f"reopening the file; Bake it if you need either.")
        if self._viewed is not None:
            self.pull_node(self._viewed)

    def _on_baked(self, node_id: str, spec: dict) -> None:
        """Record a finished bake, dock the node, and release what it made redundant."""
        if spec.get("hold"):
            self._on_held(node_id, spec)
            return
        if spec.get("cancelled"):
            self.statusBar().showMessage(
                f"{node_id} bake stopped — nothing was recorded, so the dock still reads "
                f"as un-baked and the half-written folder is safe to re-bake over")
            return
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
        self._sync_solo(self._viewed)         # ...which also ranges the region box
        self.runner.set_region(self.viewer.region)
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
        reg = self.viewer.region
        if reg is not None:
            y0, y1, x0, x1 = reg
            phrase += (f", inside the {x1 - x0}×{y1 - y0} px region at y={y0}, x={x0} "
                       f"(drag the amber box on the Viewer to move it)")
        return phrase

    def clear_frame_picks(self) -> None:
        """Run → *Clear picked frames*: back to scoping the single frame the cursor is on,
        whole volume. Reachable from the menu because a selection made on a strip that is
        currently scrolled out of the mini-map is otherwise invisible."""
        self.viewer.clear_frame_selection()

    def clear_region(self) -> None:
        """Run → *Clear troubleshooting region*: the amber box back to the whole frame."""
        self.viewer.clear_region()

    # ── the Viewer node's display settings (2026-10-02) ───────────────────────
    def _viewer_layout(self, node_id: Optional[str]) -> str:
        """The viewed node's ``layout`` Mode if it is a ``view.viewer``, else ``merged``."""
        rec = self.doc.nodes.get(node_id) if node_id else None
        if rec is None or rec.op_key != "view.viewer":
            return "merged"
        return str((rec.modes or {}).get("layout") or "merged")

    def _viewer_scalebar(self, node_id: Optional[str]):
        """The viewed node's scale-bar settings if it is a ``view.viewer`` with the bar on,
        else ``None``. Presentation params: read from the document, never the payload."""
        rec = self.doc.nodes.get(node_id) if node_id else None
        if rec is None or rec.op_key != "view.viewer":
            return None
        spec = rec.spec()
        params = rec.params or {}

        def val(name):
            if name in params:
                return params[name]
            s = spec.input(name) if spec is not None else None
            return s.default if s is not None else None

        if not bool(val("show_scalebar")):
            return None
        try:
            um = float(val("scalebar_um") or 0.0)
        except (TypeError, ValueError):
            um = 0.0
        return {"um": um, "corner": str(val("scalebar_corner") or "bottom_right"),
                "color": str(val("scalebar_color") or "white")}

    def _on_region_changed(self) -> None:
        """The Viewer's region box was dragged or cleared — a change to WHAT a scoped pull
        computes, laterally. Re-scope the runner and, under the scope, re-run the viewed
        node; off the scope the window is remembered for when F9 is armed."""
        self.runner.set_region(self.viewer.region)
        self._sync_solo_chip()
        if not self.runner.solo_frame:
            return
        self.statusBar().showMessage(f"troubleshooting: pulls analyse "
                                     f"{self._scope_phrase()}")
        if self._viewed is not None:
            self.pull_node(self._viewed)

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
        # the region box spans the SOURCE frame for the same reason the strips span the
        # source series; a remembered window is re-clamped, not dropped
        self.viewer.set_region_extent(self.doc.source_scope_extent(node_id)
                                      if (on and node_id) else None)
        if on and self.runner.region != self.viewer.region:
            self.runner.set_region(self.viewer.region)
        # the compare pane's chooser has to span ITS node's source extent for the same
        # reason — a linked cursor is expressed in global frame indices
        if self.viewer2 is not None and self._viewed2 is not None:
            self.viewer2.set_solo(self.doc.source_scope_totals(self._viewed2)
                                  if on else None)
        self._sync_solo_chip()

    def _scope_tag(self) -> str:
        """The scope in a few characters — ``t7`` / ``m2·3T[0,4,9]·2Z``. Shared by the
        status chip and the canvas badge so the two always read the same."""
        totals = self.viewer.solo or (1, 1, 1)
        ms, ts, zs = self.viewer.scoped_frames()
        name = f"t{compact_list(ts)}" if len(ts) == 1 else f"{len(ts)}T[{compact_list(ts)}]"
        if totals[0] > 1:
            name = (f"m{ms[0]}" if len(ms) == 1 else f"{len(ms)}M") + "·" + name
        name += f"·{len(zs)}Z" if zs else ""
        reg = self.viewer.region
        if reg is not None:
            name += f"·{reg[3] - reg[2]}×{reg[1] - reg[0]}px"
        return name

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
                                      self.viewer.channels(), self.viewer.sub())
        # LINKED compare: the primary's strips are the one cursor, so its move carries
        # the other pane with it — mirror silently, then ask for that pane's planes.
        if self._compare_linked and self._viewed2 is not None and self.viewer2 is not None:
            m, t, z, _c = self.viewer.coords()
            self.viewer2.set_cursor(m, t, z)
            self.runner.request_plane(self._viewed2, self.viewer2.coords(),
                                      self.viewer2.channels())

    def _on_playing(self, on: bool, axis: str) -> None:
        """Play pressed: warm the series ahead of the cursor, and hold playback only for a
        wait that is actually short (V2.23; cost-gated 2026-08-06; the hold itself
        cost-capped 2026-08-10).

        Pressing play says every frame is wanted, in order — the one statement that licenses
        reading them all, which the scrub prefetcher deliberately will not infer from a cursor
        nudge (it is cost-gated to ±2 frames on a computing provider so that nudging cannot
        queue sixteen whole-volume deconvolutions). Only T is preloaded: it is the axis whose
        frames are separate reads, while Z of one volume comes back with the read that displayed
        its neighbour.

        **Any series the preload will read is HELD while it prepares.** The hold is the
        difference between "smooth" and "loads every time" (reported 2026-08-05), and it is
        as right for a cheap per-plane compute as for a byte read: a stitched mosaic or a
        Z-projection warms in seconds, and playing it cold means every frame lands at its
        own decode latency — the ~9 fps, 33–267 ms jitter recorded on the stitched series
        (2026-08-10). How long the hold may run is decided by whether waiting can ever pay
        off (the user's stated preference, 2026-08-10):

        * a series that **fits the display budget** is held to completion — minutes if that
          is what its frames cost — with the progress bar counting and ⏸ as the cancellable
          way out. It ends resident, and playback is a texture upload per frame from the
          first tick.
        * a series **larger than the budget** can never be fully resident, so its hold is
          capped at :data:`PLAY_PREPARE_MAX_S`: the moment the preload's own tick rate
          projects past the cap (or the watchdog fires without a tick), the gate drops and
          playback runs off whatever is warm, frames landing as they compute, while the
          preload keeps warming behind it.

        **A whole-volume chain is never held and never preloaded** — there "every frame" is
        the node running once per frame, ~130 s and ~30 GB of working set each, so the gate
        was a wait of hours with nothing on screen: pressing play on the 3D-deconvolved set
        stopped showing frames at all (2026-08-06). ``preload_series`` refuses those
        (:meth:`EngineRunner._preload_jobs` = 0), no gate is raised, and playback advances as
        each frame lands (:meth:`ViewerPanel.set_play_pacing`) rather than on a wall clock
        nothing can keep up with."""
        reads = (self._viewed is not None
                 and self.runner.frames_are_reads(self._viewed))
        if on:
            # decided for EVERY axis, not just T: a cold Z step through a whole-volume chain
            # is the same 130 s wait, and the first one pays it for the rest of the volume
            self.viewer.set_play_pacing(not reads)
        if axis != "t":
            return
        if not on:
            self.runner.cancel_preload()
            self.viewer.set_play_gate(False)
            self._set_progress(None)
            return
        if self._viewed is None:
            return
        n = self.runner.preload_series(self._viewed, self.viewer.coords(),
                                      self.viewer.channels())
        if not n:
            return                       # already resident, or too costly to read ahead at
                                         # all — either way, play immediately
        # named explicitly: with a compare pane open the most recently held view may be the
        # OTHER pane's node, and this sentence is about the series being played here
        fits = self.runner.series_fits(planes=len(self.viewer.channels()),
                                       node_id=self._viewed)
        self._preload_hold_t0 = time.monotonic()
        #: whether this hold is subject to the PLAY_PREPARE_MAX_S cap — only a series too
        #: big to ever be fully resident is; one that fits holds to completion (see the
        #: constant's note). ⏸ remains the way out either way (it cancels the preload,
        #: whose `preload_finished` drops the gate).
        self._preload_hold_capped = not fits
        self.viewer.set_play_gate(
            True, f"preparing {n} frame{'s' if n != 1 else ''} for smooth playback…"
                  + ("" if fits else "  (larger than the display memory budget — the tail "
                                     "will re-read)"))
        self._set_progress(0.0)
        if not fits:
            # the wall-clock half of the cap: a preload whose first frame never finishes
            # emits no tick for the ETA check to judge, and a lit play button showing
            # nothing is the exact state the 2026-08-06 report describes. Checked against
            # `play_gated`, so a gate that already dropped (finished, cancelled, ETA)
            # makes this a no-op.
            node = self._viewed
            QTimer.singleShot(int(PLAY_PREPARE_MAX_S * 1000),
                              lambda: self._drop_play_gate(node))

    def _on_preload_progress(self, node_id: str, done: int, total: int) -> None:
        if node_id != self._viewed or not total:
            return
        self._set_progress(done / float(total))
        self.statusBar().showMessage(
            f"preparing frames for playback — {done}/{total}")
        # The ETA half of the PLAY_PREPARE_MAX_S cap — CAPPED holds only (a series larger
        # than the budget; one that fits waits to completion, see the constant's note).
        # Judged on the preload's own measured rate, and two ticks before judging — one
        # frame's wall time divided by one is not a rate.
        if (getattr(self, "_preload_hold_capped", True)
                and self.viewer.play_gated() and done >= 2):
            elapsed = time.monotonic() - getattr(self, "_preload_hold_t0", 0.0)
            if elapsed * total / done > PLAY_PREPARE_MAX_S:
                self._drop_play_gate(node_id)

    def _drop_play_gate(self, node_id: str) -> None:
        """Release a held playback whose preparation is running long — the frames keep
        warming behind it (the preload is NOT cancelled), so playback smooths out lap by
        lap instead of holding a blank stare. No-op unless ``node_id`` is still the viewed
        node with its gate up: the watchdog that arms this at ▶ may fire long after the
        preload finished, the node changed, or playback stopped."""
        if node_id != self._viewed or not self.viewer.play_gated():
            return
        self.viewer.set_play_gate(False)
        self.statusBar().showMessage(
            "playing while the rest of the series prepares — each frame shows as it "
            "lands", 4000)

    def _on_preload_finished(self, node_id: str, completed: bool) -> None:
        """Frames are in: drop the gate and let the timer run.

        Also on a CANCELLED preload — an edit, or the node changing under it. Leaving the gate
        up there would strand playback in a paused state whose button says it is playing, which
        is worse than playing a frame cold."""
        self._set_progress(None)
        if completed and node_id == self._viewed:
            # said whether or not the gate is still up: a hold released early by the cap
            # reaches this point playing cold, and this is the moment it turns warm
            self.statusBar().showMessage("playing from memory", 2500)
        if self.viewer.play_gated():
            self.viewer.set_play_gate(False)

    def _on_run_finished(self, node_id, payload, plane, axes, seconds) -> None:
        # A source the Movie Editor is waiting on may have just been computed by an
        # ordinary pull; it takes the payload the same way it takes a fetched one. Only the
        # runner's UNPINNED result: a solo-scoped payload holds a few frames, and a movie
        # built from it would silently be a truncated one.
        if payload is not None and getattr(self, "movie_editor", None) is not None:
            full = self.runner.finished_result(node_id)
            if full is not None:
                self.movie_editor.on_fetched(node_id, full)
        if plane:
            self._open_viewer()       # there is something to see now — unfold the pane
        panes = self._panes_showing(node_id)
        for pane in panes:
            pane.show_result(node_id, plane, axes, seconds, dataset=payload,
                             overlay=self.runner.overlay_channels(node_id),
                             overlay_note=self.runner.overlay_note(node_id),
                             overlay_style=self.runner.overlay_style(node_id),
                                overlay_src=self.runner.overlay_sources(node_id))
            # flicker is a property of TIME, not of the composite, so it is driven here
            # rather than folded into the style map the shader reads
            pane.set_overlay_flicker(self.runner.overlay_flicker_hz(node_id))
            self._sync_overlay_frames(pane, node_id)
            # the Viewer NODE's own display settings (2026-10-02): its layout Mode and the
            # presentation-only scale bar, read live from the document — "merged" and no
            # bar for every other node, so viewing a filter never inherits them
            pane.set_source_layout(self._viewer_layout(node_id))
            pane.set_scalebar(self._viewer_scalebar(node_id))
        if self._maximized:
            # a new axes shape rebuilds the channel/LUT controls, and fresh widgets are
            # visible — re-fold them so the mini-map keeps its compact strip
            self.viewer.set_compact(True, force=True)
        if self.viewer in panes:
            # the spreadsheet, sweep table and iteration strip follow the PRIMARY pane:
            # a compare delivery must not yank them off the node being tuned
            self.sheet.show_dataset(node_id, payload)
            self._record_sweep(node_id, payload)
            self._sync_iteration_strip(node_id)
        self._sync_compare_link()
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

    def _on_run_queued(self, node_id: str, depth: int) -> None:
        """A pull joined the queue behind the running one: claim its cards as ``queued`` and
        say so in the status bar. The branch is lined up, not lost."""
        plan = [n for n in self.runner.planned_nodes(node_id) if n in self.doc.nodes]
        self.scene.set_queued(node_id, plan)
        running = self._run_target or "a node"
        self.statusBar().showMessage(
            f"{node_id} queued behind {running} — {depth} waiting")

    def _on_run_cancelled(self, node_id: str) -> None:
        """A queued or running pull was dropped because an edit landed inside its cone.
        Retire its claim and clear the cards nothing else wants, so no card is left
        reporting work that will never finish."""
        self.scene.clear_run_plan(node_id)
        self.scene.finish_run(None)
        if not self.runner.busy:
            self._set_led("idle")
            self.minimap.set_state("idle")

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
        # refresh — only the showing pane's frame changes.
        for pane in self._panes_showing(node_id):
            pane.show_planes(node_id, planes, axes, seconds)
            self._sync_overlay_frames(pane, node_id)
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

    def _sync_console_action(self, visible: bool) -> None:
        act = getattr(self, "_console_act", None)   # the dock is built before the menu
        if act is None or act.isChecked() == bool(visible):
            return
        act.blockSignals(True)                      # reflect, don't re-drive
        act.setChecked(bool(visible))
        act.blockSignals(False)

    def _toggle_console(self, on: bool) -> None:
        """View ▸ Console. Opening it by hand counts as "shown", so the first failure does
        not then re-raise a dock the user already has open."""
        self._console_shown = self._console_shown or bool(on)
        self._console_dock.setVisible(bool(on))
        if on:
            self._console_dock.raise_()

    def _on_run_failed(self, node_id, trace) -> None:
        # The Viewer shows the failing line (status) + full trace (tooltip), and the card
        # that raised keeps its red 'error' state (the engine's error event named it, which
        # is more precise than the pulled node). The CONSOLE gets the whole trace,
        # selectable, because the other three are all uncopyable: a status line is truncated
        # to the window width, a tooltip cannot be selected, and a red card is not text.
        self.scene.finish_run(node_id, failed=True)
        for pane in self._panes_showing(node_id):
            pane.show_error(node_id, trace)
        self._set_led("error")
        self._set_progress(None)              # a failed run must not leave a stale bar
        self.minimap.set_state("error")
        last = [ln for ln in trace.strip().splitlines() if ln.strip()]
        self.console.error(f"{node_id} FAILED\n{trace.rstrip()}")
        # Raise it on the FIRST failure of a session rather than every time: a user who has
        # deliberately closed it while working through a chain of errors should not have to
        # close it again after each one.
        if not self._console_shown:
            self._console_shown = True
            self._console_dock.show()
            self._console_dock.raise_()
        self.statusBar().showMessage(
            f"{node_id} FAILED — {last[-1] if last else 'see Console'}")

    # ── file (G6) ────────────────────────────────────────────────────────────
    def _forget_display_state(self) -> None:
        """A new or newly opened graph reuses node ids, so the Viewer's per-node LUTs and
        switched-off channels from the last graph must not carry over to it."""
        for pane in (self.viewer, self.viewer2):
            if pane is not None:
                pane.forget_display_state()

    def file_new(self) -> None:
        self.doc.clear()          # → _on_doc_changed closes the compare pane too
        self._forget_display_state()
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
        from nodegraph.metadata import MetaEnvelope, stamp_source_file
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

        # Pre-fill the calibration boxes from the file's own header, so the card SHOWS the
        # spacing the pipeline is about to use instead of hiding it behind a 0 that means
        # "ask the file". A key the file does not carry stays 0 — and a plain TIFF carries
        # no Z spacing at all, so that empty box IS the prompt to type one (2026-09-15).
        # Writing the detected value as a param is deliberate: it is then visible, editable
        # and SAVED, so the graph records the spacing its measurements assumed rather than
        # depending on a header that may not survive a re-export.
        prefill = {k: float(calib[k]) for k in CALIB_OVERRIDE_KEYS
                   if isinstance(calib.get(k), (int, float)) and float(calib[k]) > 0.0}
        # NO grouping work here (2026-09-28). Grouping is off by default, so detecting at
        # file-pick time would be work nobody asked for on every file that is ever opened
        # — and this is the one path where an extra read is least affordable, because it
        # runs inside the File menu before anything is on the canvas. The lever turns it
        # on, and `GraphDocument.to_graph` resolves the groups then
        # (`group_descriptors`, which falls back to the envelope), so nothing has to be
        # captured in advance.
        rec = self.doc.add_node(
            LOAD_OP, x=x, y=y,
            params={"path": path, TITLE_KEY: os.path.basename(path),
                    CHANNELS_KEY: chans, **prefill})
        # `source_file` rides the seed for the same reason the bundle card's does: the
        # edit-time envelope and the pulled payload must agree about a positional list, and
        # `EngineRunner._resolve_source` stamps the identical value on the pull side. It is
        # what lets `util.chain` order separately loaded files by their names.
        self.doc.set_meta_seed(rec.id, stamp_source_file(
            MetaEnvelope(axes=axes, metadata=dict(calib)), path))
        return rec, axes

    def _add_bundle_node(self, paths: list, x: float, y: float):
        """Read every path's metadata (**no pixels**) and drop ONE ``io.load`` card that
        carries all of them — a **file bundle**. Returns ``(rec, axes)`` with ``axes.m``
        summed over the members.

        The members must share ``(t, z, c, y, x)``. That is not a stylistic rule: a bundle
        lays the files end to end on the multipoint axis, and an axis cannot be ragged, so
        :class:`~nodegraph.provider.MultiSourceProvider` refuses a mismatch outright. Doing
        the same check HERE, against the headers, is what turns that into a sentence at
        load time naming the file and the axis, instead of a traceback on the first pull —
        by which point the user has wired a pipeline onto a card that could never run.

        Channel NAMES are compared too, not just the count. Two files with two channels
        each but ``[DAPI, GFP]`` against ``[GFP, DAPI]`` would stack without complaint and
        put two different stains in one column of the results.
        """
        import os
        from nodelab_v2.document import CHANNELS_KEY, TITLE_KEY
        from nodegraph.metadata import MetaEnvelope, SOURCE_FILE_KEY
        from nodegraph.dataset import AxisSizes
        from nodelab_v2.ingest import read_meta_only
        from nodelab_v2.runner import BUNDLE_PATHS_KEY, _unique_labels

        heads = [(p,) + tuple(read_meta_only(p)) for p in paths]
        (p0, ax0, calib0, disp0) = heads[0]

        def _names(disp, axes):
            got = disp.get("channel_names") or []
            return [str(got[i]) if i < len(got) else f"Ch{i}" for i in range(axes.c)]

        base_names = _names(disp0, ax0)
        for (p, ax, _cal, disp) in heads[1:]:
            bad = [n for n in ("t", "z", "c", "y", "x")
                   if int(getattr(ax, n)) != int(getattr(ax0, n))]
            if bad:
                raise ValueError(
                    f"{os.path.basename(p)} does not match {os.path.basename(p0)} on "
                    f"{', '.join(bad)} — "
                    f"(t,z,c,y,x) is {(ax.t, ax.z, ax.c, ax.y, ax.x)} against "
                    f"{(ax0.t, ax0.z, ax0.c, ax0.y, ax0.x)}. Grouped files share one "
                    f"pipeline, so they have to share those axes; load them separately "
                    f"instead.")
            if _names(disp, ax) != base_names:
                raise ValueError(
                    f"{os.path.basename(p)} has channels {_names(disp, ax)} but "
                    f"{os.path.basename(p0)} has {base_names}. Grouping them would put "
                    f"different stains in the same result column; load them separately, "
                    f"or reorder the channels first.")

        emis = disp0.get("channel_emission_nm")
        colors = disp0.get("channel_colors")

        def _native(i):
            if isinstance(colors, (list, tuple)) and i < len(colors):
                col = colors[i]
                if isinstance(col, (list, tuple)) and len(col) == 3:
                    return [int(v) for v in col]
            return None

        chans = [{"name": base_names[i],
                  "emission_nm": (emis[i] if isinstance(emis, (list, tuple))
                                  and i < len(emis) else None),
                  "color": _native(i)} for i in range(ax0.c)]

        labels = _unique_labels(list(paths))
        total_m = sum(int(h[1].m) for h in heads)
        axes = AxisSizes(m=total_m, t=ax0.t, z=ax0.z, c=ax0.c, y=ax0.y, x=ax0.x)
        rec = self.doc.add_node(
            LOAD_OP, x=x, y=y,
            params={"path": paths[0], BUNDLE_PATHS_KEY: list(paths),
                    TITLE_KEY: f"{len(paths)} files",
                    CHANNELS_KEY: chans})
        # The seed envelope must say the same thing the pull will (the edit-time envelope
        # and the payload agreeing is the standing rule for anything that changes axes):
        # M is the sum, and `source_file` names the file each position came from.
        md = dict(calib0)
        md[SOURCE_FILE_KEY] = [labels[i] for i, h in enumerate(heads)
                               for _ in range(int(h[1].m))]
        self.doc.set_meta_seed(rec.id, MetaEnvelope(axes=axes, metadata=md))
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
        paths, _f = QFileDialog.getOpenFileNames(
            self, "Load ND2 / TIFF file(s)", "",
            "Microscopy images (*.nd2 *.tif *.tiff);;ND2 (*.nd2);;"
            "TIFF (*.tif *.tiff);;All files (*)")
        if not paths:
            return
        self._load_source_paths(paths, group=self._ask_group(paths))

    def file_load_sequence(self) -> None:
        """File → Load file sequence…: pick ONE file of a numbered series, get all of it.

        The gap this fills. :meth:`file_load_source` can already multi-select a folder's
        worth of files into one bundle card — but that means ctrl-clicking 120 entries in a
        file dialog, and the bundle stacks them on POSITIONS, because that is the only axis
        a loader can grow without being told what the files mean. For a timelapse exported
        one frame per file that is the wrong axis, and wrongly in a way nothing errors on:
        the result is 120 fields of a 1-frame series, so ``util.stack`` fuses nothing and
        ``track.link`` has no frames to link (see :mod:`nodegraph.catalog.util.chain`).

        So this action does the three things that turn one click into that series: it
        derives the sequence from the picked file's name and scans its folder
        (:func:`nodegraph.file_sequence.scan`), it shows what it found and lets the pattern
        be corrected before anything is built
        (:class:`~nodelab_v2.sequence_dialog.SequenceScanDialog`), and it drops the bundle
        card with a ``util.chain`` already wired to it and preset to the chosen axis.

        Everything after the dialog is :meth:`_load_source_paths`, so a series whose files
        do not share a grid is refused with the same message, and the same "load as separate
        cards" fallback, as any other bundle.
        """
        path, _f = QFileDialog.getOpenFileName(
            self, "Pick one file of the sequence", "",
            "Microscopy images (*.nd2 *.tif *.tiff);;ND2 (*.nd2);;"
            "TIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return
        from nodelab_v2.sequence_dialog import SequenceScanDialog

        dlg = SequenceScanDialog(path, self)
        if dlg.exec() != QDialog.Accepted:
            return
        paths, axis = dlg.result_paths(), dlg.chain_axis()
        if len(paths) < 2:
            # One file is a correct answer, not an error — but chaining it is a no-op, so
            # it loads as the ordinary single source card it is rather than arriving with
            # an inert Chain node attached that the user would have to work out and delete.
            self._load_source_paths(paths)
            return
        self._load_source_paths(paths, group=True, chain_axis=axis)

    def _ask_group(self, paths: list) -> bool:
        """For a multi-file pick: one **bundle** card, or one card per file?

        Asked rather than inferred, because both answers are ordinary and the graph you
        build next is different for each. Grouping is for replicates — the same acquisition
        of several wells or dishes that you want to treat identically and compare, which is
        one pipeline whose spreadsheet gains a ``file`` column. Separate cards are for files
        with different ROLES (a reference and a sample, two channels to merge), which is
        several wired inputs.

        One file, or a cancelled dialog, never asks."""
        if len(paths) < 2:
            return False
        box = QMessageBox(self)
        box.setWindowTitle("Load as a group?")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Load {len(paths)} files as one bundle, or as separate cards?")
        box.setInformativeText(
            "A bundle is ONE card carrying all the files. They run through one pipeline "
            "and every exported table gains a 'file' column naming the row's source. The "
            "files must share their channels and frame geometry.\n\n"
            "Separate cards are independent sources you wire up yourself — the right "
            "choice when the files play different roles.")
        grp = box.addButton("Group into one bundle", QMessageBox.AcceptRole)
        box.addButton("Separate cards", QMessageBox.RejectRole)
        box.setDefaultButton(grp)
        box.exec()
        return box.clickedButton() is grp

    def publish_recipe(self) -> None:
        """Graph → Publish as a LabLink recipe…

        The window owns the document, so it is the window that hands it over — the panel and
        the dialog only ask. Passes the dock's connected hub when there is one, so Submit is
        available without asking for a URL and token a second time.
        """
        from nodelab_v2.lablink.authoring import publish_document

        written = publish_document(self, self.doc,
                                   hub=getattr(self.lablink.send, "_hub", None),
                                   suggested_name="")
        if written:
            self.statusBar().showMessage(f"recipe written to {written}")

    def lablink_load_result(self, path: str) -> None:
        """A hub returned an image; put it on the canvas as a source node.

        This is what makes a processed result a *starting point* rather than a file on disk:
        the returned stack becomes an ``io.load`` card like any other, so the next thing
        done with it happens here, in a graph that can be saved and published in turn.

        The readers cover ND2 and TIFF only, so a returned PNG quicklook previews in the dock
        and stops there — said plainly rather than failing with a reader error.
        """
        import os

        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "LabLink", f"that result is no longer at {path}")
            return
        if not path.lower().endswith((".nd2", ".tif", ".tiff")):
            QMessageBox.information(
                self, "LabLink",
                f"{os.path.basename(path)} is not an ND2 or TIFF, so there is no reader "
                f"for it here. The dock's preview is the way to look at it.")
            return
        self._load_source_paths([path])
        self.statusBar().showMessage(
            f"loaded {os.path.basename(path)} from the hub — it is a source node now")

    def _load_source_paths(self, paths: list, group: bool = False,
                           chain_axis: str = "") -> None:
        """Drop one ``io.load`` card per path, stacked, selected, and scrolled into view —
        or, with ``group``, ONE bundle card carrying all of them.

        Split out of :meth:`file_load_source` so the File menu and a returned LabLink result
        take the identical path — including the per-file failure collection, which is what
        keeps one unreadable file from costing the others.

        A bundle is all-or-nothing and says so: its members have to share a grid, so
        "5 of 6 grouped" is not a thing that can be built. A refused group reports why and
        offers the fallback that always works — separate cards — rather than silently
        loading something the user did not ask for.

        ``chain_axis`` (:meth:`file_load_sequence`) additionally wires a ``util.chain`` card
        onto the bundle, preset to that axis. It rides HERE rather than in the caller so the
        sequence loader inherits this method's grid-mismatch handling unchanged — and it is
        deliberately dropped by the "separate cards" fallback, since there is no bundle left
        for a chain to re-address."""
        import os

        c = self.view.mapToScene(self.view.viewport().rect().center())
        x, y = c.x() - 107, c.y() - 40
        added: list = []
        failed: list = []

        if group and len(paths) >= 2:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                rec, axes = self._add_bundle_node(list(paths), x, y)
            except Exception as exc:  # noqa: BLE001 — a reader failure or a grid mismatch
                QApplication.restoreOverrideCursor()
                box = QMessageBox(self)
                box.setWindowTitle("Cannot group these files")
                box.setIcon(QMessageBox.Warning)
                box.setText("These files cannot share one bundle.")
                box.setInformativeText(f"{exc}")
                sep = box.addButton("Load as separate cards",
                                    QMessageBox.AcceptRole)
                box.addButton("Cancel", QMessageBox.RejectRole)
                box.setDefaultButton(sep)
                box.exec()
                if box.clickedButton() is sep:
                    self._load_source_paths(paths, group=False)
                return
            QApplication.restoreOverrideCursor()
            chain = None
            if chain_axis and chain_axis != "M":
                chain = self.doc.add_node("util.chain", x=x + SEQUENCE_CHAIN_GAP, y=y,
                                          modes={"chain_axis": chain_axis})
                self.doc.connect(rec.id, "image", chain.id, "data")
            ids = [rec.id] + ([chain.id] if chain is not None else [])
            items = [self.scene.node_items[i] for i in ids if i in self.scene.node_items]
            if items:
                self.scene.clearSelection()
                span = QRectF()
                for it in items:
                    it.setSelected(True)
                    span = span.united(it.card_rect().translated(it.pos()))
                self.view.ensureVisible(span, 60, 60)
            if chain is not None:
                self.statusBar().showMessage(
                    f"{len(paths)} files loaded as one source and chained onto "
                    f"{chain_axis} — they are {_AXIS_NOUN.get(chain_axis, chain_axis)} of "
                    f"one series now, not {axes.m} separate positions")
            else:
                self.statusBar().showMessage(
                    f"bundled {len(paths)} files into one source — {axes.m} positions, "
                    f"{axes.c} channel(s); exported tables will carry a 'file' column")
            return

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
        self._forget_display_state()
        self.view.fit_all()
        if self.doc.has_unedited_structure:
            QMessageBox.information(
                self, "Zones / groups preserved",
                "This file contains zones or node groups that this build can't edit "
                "yet. They are shown as their member nodes and preserved unchanged on "
                "save — editing/creating them in the GUI is a later phase.")

    def _stamp_all_movies(self) -> None:
        """Before a save: write the Viewer's current LUTs into every linked movie channel,
        so the file on disk reproduces the movies the user was looking at."""
        for mid in self._movie_nodes():
            self.stamp_movie_links(mid)

    def file_save(self) -> None:
        if not self.doc.path:
            self.file_save_as()
            return
        self._stamp_all_movies()
        self.doc.save_file(self.doc.path)
        self.statusBar().showMessage(f"saved {self.doc.path}")

    def file_save_as(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "Save graph", "graph.nd2graph.json",
                                               FILE_FILTER)
        if not path:
            return
        self._stamp_all_movies()
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
