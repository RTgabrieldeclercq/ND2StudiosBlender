"""Main window for NodeLab v2 (Phase 5).

Chrome (G10): menu bar (File / Run / View), status bar, dockable Palette (G2),
Properties inspector, and Viewer (G4). File actions (G6) round-trip
``*.nd2graph.json`` through the document (headless
:mod:`nodegraph.serialize` + a ``ui`` extras object). Run actions (G7) submit pulls to
the :class:`~nodelab_v2.runner.EngineRunner` — double-click any node (or press F5 on a
selection) to view its output; edits invalidate in-flight results by epoch.

**Viewers (V4.00 step 4).** Every Viewer is a dock of the shell, ``viewer:<n>``: pop it
out, tab it, close it, or open another beside it (its title bar's ``+``, View ▸ New ▸
Viewer, or Compare). A viewer is BOUND to the node it was asked to show, and a result lands
in every viewer bound to its node; the one the user last worked in is ACTIVE — pulls, card
clicks, picks, the troubleshooting scope and the spreadsheet follow it.

**Maximized canvas (2026-07-27).** The canvas' top-right ⛶ button (also View → *Maximize
node canvas*, ``Ctrl+Space``) hides the docked viewers and moves the *active* ViewerPanel
into the :class:`~nodelab_v2.minimap.MiniMapOverlay` — a bordered mini-map in the canvas'
top-left corner that previews whatever node you click, live (:data:`FOLLOW_DELAY_MS`
debounce; the previewed card wears an accent spine). ``Esc``, the mini-map's dock button,
or a header double-click puts the Viewer back in its dock. A floating viewer stays put.
"""
from __future__ import annotations

import functools
import sys
import weakref
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import nodegraph.nodes  # noqa: F401 — registers the node catalog into NODES
from nodegraph import hotreload
from collections import OrderedDict

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QVBoxLayout, QWidget,
)

from nodegraph.iterate import (
    ITERATE_OP, SWEEP_KEY, SWEEP_OWNER_KEY, SWEEP_ROWS_KEY, plan as iterate_plan)
from nodelab_v2 import layout_store as LS
from nodelab_v2 import theme as T
from nodelab_v2.console import ConsolePanel
from nodelab_v2.version import PRODUCT, __version__ as APP_VERSION
from nodelab_v2.canvas import CanvasPanel, kind_icon
from nodelab_v2.workspace import (FREE as FREE_KIND, Workspace, build_example, kind_label,
                                  local_ids, qualify, split_run_id)
from nodelab_v2.document import GraphDocument
from nodelab_v2.linked_document import (
    EDIT_MASTER, EDIT_MODIFIED, EDIT_UNIQUE, SHAPE_HINT, TOPOLOGY_HINT, LinkedDocument,
    LinkedPageError)
from nodelab_v2.linked_edit_dialog import LinkedEditDialog
from nodelab_v2 import page_recipes as PR
from nodelab_v2.new_page_dialog import NewPageDialog, SavePageRecipeDialog
from nodelab_v2.mode_switch import ModeSwitch
from nodelab_v2.pages_panel import PagesPanel
from nodelab_v2.framestrip import compact_list
from nodelab_v2.inspector import InspectorPanel
from nodelab_v2.lablink.panel import LabLinkPanel
from nodelab_v2.minimap import MiniMapOverlay
from nodelab_v2.node_item import NodeItem
from nodelab_v2.ops import (MOVIE_OP, ACCESS_INGEST, ACCESS_MODE, CALIB_OVERRIDE_KEYS, DOCK_OP,
                            LOAD_OP, PAGE_INPUT_OP,
                            PAGE_NAME_KEY, PAGE_OUTPUT_OP, PAGE_SOURCE_KEY, PRECISION_UNSET,
                            is_visual_output)
from nodelab_v2.palette import PalettePanel
from nodelab_v2.picker import Calibration, request_for
from nodelab_v2 import readiness as RD
from nodelab_v2.runner import EngineRunner, ensure_gui_ops, request_answers
from nodelab_v2.scene import GraphScene, GraphView
from nodelab_v2.shell import DockShell, PanelSpec
from nodelab_v2.spreadsheet import SpreadsheetPanel
from nodelab_v2.viewer import ViewerPanel
from nodelab_v2.viewer_controls import ViewerControlsPanel
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

#: horizontal offset of the ``util.timeseries`` card the sequence loader drops beside its bundle
#: (scene px). Wide enough that the wire between them is visibly a wire rather than two
#: touching cards — a source card is ~214 px, so this leaves a clear ~90 px span.
SEQUENCE_CHAIN_GAP = 306
#: x-gap from a freshly loaded source card to the Page Output that publishes it (V4.00
#: step 11) — one wire-length to its right, like a sequence's chain card
SOURCE_OUTPUT_GAP = 306

#: What the axis a sequence was chained onto is CALLED, for the status line. The keys are
#: ``util.timeseries``'s own ``chain_axis`` values; "M" is absent because the loader never drops
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

#: The share of the canvas column (menu bar to status bar) the first Viewer dock takes in
#: the default layout, and a Viewer dock the first time it opens. The app LAUNCHES with the
#: Viewer on screen (V4.00 step 11a) — blank until the first result — above the canvas,
#: which keeps the larger share: the welcome card and the graph are what a new session
#: works in first. Since step 11d the Viewer dock holds the IMAGE only — its Playback and
#: Channels panels sit under it (:data:`CONTROLS_H`) — so the share is smaller than the
#: 0.45 the whole panel took, and the canvas keeps about what it had.
VIEWER_SHARE = 0.30
#: The default layout's side-column widths (px): Nodes on the left, Properties on the right.
PALETTE_W, INSPECTOR_W = 330, 376
#: the default height of the Pages panel, as a share of the left column (step 11)
PAGES_SHARE = 0.30
#: The panel kinds of the Viewer's controls (V4.00 step 11d): the M/T/Z cursor and play
#: buttons, and the channel / histogram columns — one panel each, showing the ACTIVE
#: Viewer's (nodelab_v2.viewer_controls).
PLAYBACK_KIND, CHANNELS_KIND = "playback", "channels"
#: Their default height (px), side by side under the first Viewer.
CONTROLS_H = 180
#: The panel kind of a Viewer dock (V4.00 step 4): several instances, ``viewer:<n>``.
VIEWER_KIND = "viewer"
#: The panel kind of a docked canvas (V4.00 step 5) — every canvas but the main one.
CANVAS_KIND = "canvas"

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
/* every dock has a custom title bar, so QDockWidget itself paints nothing (PanelDock's
   paintEvent fills it); `background` here sets its palette Window brush, so the fill Qt
   gives a floating dock BEFORE that paintEvent is already dark rather than #efefef. The
   1px border is what keeps the frame gutter: once a rule has a background, the style takes
   PM_DockWidgetFrameWidth from the rule's border, and `border:0` would let the panel cover
   the frame paintEvent draws (measured, 2026-10-06). */
QDockWidget {{ background:{T.BG.name()}; border:1px solid {T.BORDER.name()};
  color:{T.MUTED.name()}; font-size:10px; font-weight:800;
  titlebar-close-icon:none; titlebar-normal-icon:none; }}
/* the scroll wrapper around a `scroll` panel (the Movie Editor): its viewport shows
   wherever the panel does not reach, so it takes the window background, not the palette's */
QScrollArea#panelScroll {{ background:{T.BG.name()}; border:0; }}
QScrollArea#panelScroll > QWidget#qt_scrollarea_viewport {{ background:{T.BG.name()}; }}
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


def _needs_topology(fn):
    """A window action that changes the active page's graph SHAPE. On a linked page whose
    edits are not settled it asks how the change applies (V4.00 step 11e,
    :meth:`MainWindow._topology_ok`) before any other dialog opens; cancelled, nothing
    happens and the status bar says why."""
    @functools.wraps(fn)
    def guarded(self, *a, **k):
        if not self._topology_ok():
            return None
        try:
            return fn(self, *a, **k)
        except LinkedPageError as exc:   # refused part-way by a page sending to its master
            self.statusBar().showMessage(str(exc), 8000)
            return None
    return guarded


def _needs_shape(fn):
    """A window action that makes a FRAME, GROUP or ZONE: those of a linked page are its
    master's in every mode, so a linked page can only be made unique for them (asked)."""
    @functools.wraps(fn)
    def guarded(self, *a, **k):
        if not self._topology_ok(shape=True):
            return None
        return fn(self, *a, **k)
    return guarded


def _needs_source_page(fn):
    """A window action that LOADS an image (V4.00 step 11): refused — with the hint in the
    status bar, before any dialog opens — when the page the image will land on is a linked
    page. That page is the Image Input page, not the active one
    (:meth:`MainWindow._input_page`)."""
    @functools.wraps(fn)
    def guarded(self, *a, **k):
        if not self._source_load_allowed():
            return None
        return fn(self, *a, **k)
    return guarded


#: why a LOAD can never go to a linked page's master switched off (V4.00 step 11e)
_LOAD_PUSH_NOTE = "a Load card is where data starts, so it cannot be switched off"

#: the windows the LinkedPageError backstop reports to (weakly held — a closed one goes)
_LINKED_HOOK_WINDOWS: "weakref.WeakSet" = weakref.WeakSet()


def _install_linked_hook(win: "MainWindow") -> None:
    """The backstop for a structural edit no guard caught: Qt hands an exception escaping a
    slot to ``sys.excepthook``, and a :class:`LinkedPageError` there is a refusal for the
    user, not a crash — it goes to the status bar. Installed once; every other exception
    reaches the previous hook unchanged."""
    _LINKED_HOOK_WINDOWS.add(win)
    if getattr(sys.excepthook, "_nd2_linked", False):
        return
    prev = sys.excepthook

    def hook(et, ev, tb):
        if et is not None and issubclass(et, LinkedPageError):
            for w in list(_LINKED_HOOK_WINDOWS):
                try:
                    w.statusBar().showMessage(str(ev), 8000)
                    return
                except RuntimeError:          # a window already torn down
                    continue
            return
        prev(et, ev, tb)

    hook._nd2_linked = True                   # type: ignore[attr-defined]
    sys.excepthook = hook


class _MovieHost:
    """The Movie Editor's view of the window — everything it may ask for, and nothing else.

    :class:`nodelab_v2.movie_editor.MovieEditorPanel` holds widgets and a working copy of one
    timeline; the window holds the runner (to compute its sources without moving the
    Viewer), the document (to commit the timeline) and the Viewer (whose LUTs it links to).
    An adapter rather than the window itself, so the panel's dependencies are this list and
    a probe can see them."""

    def __init__(self, win: "MainWindow") -> None:
        self._w = win

    # The editor holds a BARE node id; every call resolves it on the editor's own page —
    # the one it was opened on (`MainWindow._movie_pid`), which another page becoming
    # active does not change (V4.00 step 5).
    def _pid(self) -> str:
        return self._w._movie_pid()

    def movie_state(self, node_id: Optional[str]) -> Optional[Dict[str, Any]]:
        doc = self._w._page_doc(self._pid())
        rec = doc.nodes.get(node_id) if (node_id and doc is not None) else None
        if rec is None or rec.op_key != MOVIE_OP:
            return None
        spec = rec.spec()
        try:
            env = doc.env(node_id)
        except Exception:                  # noqa: BLE001 — an un-propagated node
            env = None
        label = f"{spec.label if spec else rec.op_key} ({node_id})"
        if len(self._w.workspace.pages) > 1:      # whose movie, once there are several pages
            label = f"{self._w.page_title(self._pid())[0]} › {label}"
        return {"spec": spec, "params": dict(rec.params), "modes": dict(rec.modes),
                "env": env, "label": label}

    def movie_sources(self, node_id: str) -> Dict[str, Dict[str, Any]]:
        return self._w._movie_sources(node_id, page_id=self._pid())

    def source_payload(self, node_id: str) -> Any:
        return self._w.runner.finished_result(qualify(self._pid(), node_id))

    def fetch(self, node_id: str) -> None:
        self._w.runner.fetch(qualify(self._pid(), node_id))

    def commit_timeline(self, node_id: str, text: str) -> None:
        self._w.write_movie_timeline(node_id, text, page_id=self._pid())

    def set_sweep(self, node_id: str, value: str) -> None:
        self._w.set_movie_sweep(node_id, value, page_id=self._pid())

    def live_display(self, node_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._w.live_display(node_id, spec, page_id=self._pid())

    def capture(self, node_id: str) -> None:
        self._w.stamp_movie_links(node_id, page_id=self._pid())

    def export(self, node_id: str) -> None:
        self._w.export_movie(node_id, page_id=self._pid())


class MainWindow(QMainWindow):
    def __init__(self, *, persist_layout: Optional[bool] = None) -> None:
        """``persist_layout``: restore the saved panel layout now and save it on close
        (V4.00 step 3). ``None`` follows ``NODELAB_LAYOUT`` (on unless it is ``0`` — every
        probe and script sets it, so a test run never touches a user's layout)."""
        super().__init__()
        ensure_gui_ops()
        self.setWindowTitle(f"{PRODUCT} — nodegraph canvas")
        # the palette first: it is what Qt paints where no QSS rule reaches (a floating
        # dock's window fill, a scroll viewport, a native dialog) — the default one is light
        QApplication.instance().setPalette(T.palette())
        self.setStyleSheet(_window_qss())

        # V4.00: the file on disk is a Workspace of PAGES (format 3.0). Every page has its
        # own scene (`scene_for`) and a canvas shows one page at a time; `doc`, `scene`,
        # `view`, `minimap` and `welcome` name the active page's and the active canvas's.
        self.workspace = Workspace.standard(GraphDocument())   # the four standard pages
        #: pages whose start card was dismissed (✕ / Start empty), and pages already
        #: given their Page Input — for this session (V4.00 step 11)
        self._welcome_dismissed: set = set()
        self._seeded_pages: set = set()
        #: pages whose TAB the user closed (V4.00 step 11d): out of the canvases' sub-tab
        #: rows, still in the workspace and the Pages panel; showing one opens its tab again
        self._closed_pages: set = set()
        self._scenes: Dict[str, GraphScene] = {}
        #: per page: (its document, the listeners `_wire_scene` installed on it)
        self._page_hooks: Dict[str, Tuple[Any, List[Any]]] = {}
        self.runner = EngineRunner(self.workspace)   # every page; run ids are page-qualified
        # ── viewers (V4.00 step 4) ─────────────────────────────────────────────
        # Every Viewer is a dock (`viewer:<n>`, built by `_make_viewer`), BOUND to the node
        # it was asked to show; `self.viewers` lists them, `_active_viewer()` is the one the
        # user works in. A viewer opened by Compare FOLLOWS the one it was opened beside:
        # `_links` maps follower → leader, and `_linked` holds the followers whose M/T/Z
        # extents match their leader's — one cursor then moves both (V2.28).
        self._links: Dict[ViewerPanel, ViewerPanel] = {}
        self._linked: Set[ViewerPanel] = set()
        #: run id → the viewer that last asked for it. A result no viewer is bound to any
        #: more (the user pulled something else into it meanwhile) still lands where it was
        #: asked for: the first result to land is viewable while the next one computes.
        self._asker: Dict[str, ViewerPanel] = {}
        #: the viewer whose ▶ started the running preload
        self._preload_viewer: Optional[ViewerPanel] = None
        #: (viewer, node, what it showed before) when the SELECTION last pulled into a viewer
        #: (`_preview_pull`) — F8 on that card compares it beside what was viewed before
        self._preview_prev: Optional[Tuple[ViewerPanel, str, str]] = None

        # centre: the MAIN canvas (V4.00 step 5) — the window's central widget, so it can
        # neither float nor close (Qt lays docks out around a central widget, and gives a
        # window without one to its side columns); every further canvas is a dock. The
        # Viewer docks sit above it, in the top dock area, which the side columns' corners
        # keep to the canvas' width. The first Viewer is on screen from launch (V4.00 step
        # 11a), blank until the first result; the canvas below it keeps the larger share.
        self._main_canvas = CanvasPanel(self, self.workspace.active)
        #: the canvas the user works in — its page is the workspace's active page
        self._canvas: CanvasPanel = self._main_canvas
        #: the canvas whose mini-map hosts a viewer while maximized
        self._max_canvas: Optional[CanvasPanel] = None
        self._wire_canvas(self._main_canvas)
        self.setCentralWidget(self._main_canvas)

        # maximized canvas (Ctrl+Space / the ⛶ button): the docked viewers step aside and
        # the active one moves into this HUD frame over the canvas' top-left corner, where
        # it keeps following whatever node you click.
        self._maximized = False
        #: the viewer the mini-map hosts while maximized, and the docks the maximize hid
        self._mini_viewer: Optional[ViewerPanel] = None
        self._max_hidden: List[Any] = []
        self._follow_pending: Optional[str] = None
        self._follow_timer = QTimer(self)
        self._follow_timer.setSingleShot(True)
        self._follow_timer.setInterval(FOLLOW_DELAY_MS)
        self._follow_timer.timeout.connect(self._follow_pull)

        # panels (V4.00 step 3): every side panel is a dock of the DockShell — it pops out
        # into its own window, docks back, closes (View ▸ Panels brings it back) and keeps
        # its place across a restart (nodelab_v2.layout_store). The Viewer/canvas splitter
        # stays the central widget until steps 4–5 make those panels too.
        self.palette = PalettePanel(on_add=self._add_at_center)
        # every page by kind, with what each reads and publishes (V4.00 step 11)
        self.pages_panel = PagesPanel()
        self.pages_panel.page_requested.connect(
            lambda pid: self._show_page(self._canvas, pid))
        self.pages_panel.rename_requested.connect(self.rename_page)
        self.pages_panel.node_requested.connect(self.show_node)
        self.pages_panel.mute_requested.connect(self.set_node_muted)
        self.pages_panel.menu_requested.connect(self._pages_menu)
        self.pages_panel.new_page_requested.connect(
            lambda: self.new_page_dialog(kind=self.next_page_kind(self.workspace.active),
                                         canvas=self._canvas))
        self.palette.refresh_requested.connect(self.refresh_node_list)
        self.inspector = InspectorPanel()
        self.sheet = SpreadsheetPanel()
        # LabLink: whether this machine is serving the lab, and how to send work out to
        # another hub. Tabbed with Properties rather than given its own edge — it is
        # consulted occasionally, not watched while editing — and it starts BEHIND
        # Properties, so the panel exists without competing for attention on every launch.
        self.lablink = LabLinkPanel()
        self.lablink.send.load_into_graph.connect(self.lablink_load_result)
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
        self._console_shown = False
        # The Movie Editor (2026-09-30): a bottom panel that binds to an Export Movie node
        # when one is selected — asked for as "a movie editor when on the node, rather than
        # just a parameter list". It talks to the window only through the `_MovieHost`
        # adapter; the window owns the runner and the document.
        from nodelab_v2.movie_editor import MovieEditorPanel
        #: the page of the Export Movie the editor is bound to (`_movie_pid`)
        self._movie_page: Optional[str] = None
        self.movie_editor = MovieEditorPanel(_MovieHost(self))
        # the Viewer's controls as panels of their own (V4.00 step 11d): every Viewer the
        # window makes hands its two sections over (`_make_viewer`); these show the active one's
        self.playback_panel = ViewerControlsPanel(
            "Playback", "The M / T / Z cursor and play buttons of the active Viewer appear "
            "here once a Viewer is open.")
        self.channels_panel = ViewerControlsPanel(
            "Channels", "The channels and histograms of the active Viewer appear here once "
            "a Viewer is open.")
        self.shell = DockShell(self, allow_close=self._allow_panel_close)
        for spec in self._panel_specs():
            self.shell.register(spec)
            if spec.kind != CANVAS_KIND:        # docked canvases are opened on demand
                self.shell.spawn(spec.kind, show=False)
        self.shell.apply_default_layout()
        # one Viewer exists from the start — on screen, blank until the first result — and
        # it is the active one, so "the Viewer" always names a panel
        self.shell.activate(self.shell.docks_of(VIEWER_KIND)[0])
        self.shell.activated.connect(self._on_panel_activated)
        self.shell.changed.connect(self._prune_viewers)
        self.shell.changed.connect(self._prune_canvases)
        self._sync_viewer_controls()
        self._console_dock = self.shell.docks_of("console")[0]
        self._movie_dock = self.shell.docks_of("movie")[0]
        self._lablink_dock = self.shell.docks_of("lablink")[0]
        # Keep the View ▸ Console tick honest when the dock is closed by its own ✕ or
        # raised by a failure — a menu tick that disagrees with what is on screen is the
        # same defect as a control that does nothing.
        self._console_dock.visibilityChanged.connect(self._sync_console_action)

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

        # signals — a page's scene and document are wired when the page first gets a
        # scene (`_wire_scene`), a canvas's view and HUD when it is made (`_wire_canvas`)
        # the RUNNER is told through the workspace (V4.00 step 2): its touched set is
        # page-qualified, and an edit on a page the canvas is not showing still has to
        # cancel the runs that read it
        self.workspace.on_change(self._on_workspace_changed)
        self.runner.started.connect(self._on_run_started)
        # The Movie Editor's sources arrive on their own signal (a payload-only fetch never
        # reaches a pane), and the Viewer's settled LUT edits feed its linked channels.
        self.runner.fetched.connect(self._on_movie_fetched)
        self.runner.fetch_started.connect(
            lambda nid: self.statusBar().showMessage(
                f"computing {self._local(nid) or nid} for the Movie Editor…"))
        self.runner.finished.connect(self._on_run_finished)
        self.runner.plane_ready.connect(self._on_plane_ready)
        self.runner.failed.connect(self._on_run_failed)
        # per-source ingest (V2.21): its own lane, so its own signals — several run at
        # once and none of them is "the run".
        self.runner.ingest_started.connect(self._on_ingest_started)
        self.runner.ingest_finished.connect(self._on_ingest_finished)
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
        # (each viewer's own signals are wired where it is made — `_make_viewer`)
        self.runner.preload_progress.connect(self._on_preload_progress)
        self.runner.preload_finished.connect(self._on_preload_finished)
        # Interactive parameter picking (V2.16). Both surfaces that can ARM a pick — the
        # inspector's Pick button and a card's ◎ glyph — route here rather than talking to
        # the viewer directly, because arming needs the node's calibration and committing
        # needs the document, and only the window has both.
        self.inspector.pick_requested.connect(self._arm_pick)
        # Dock (V2.18): both surfaces that can start a bake — the inspector's buttons and
        # the card's context menu — route to one handler, for the same reason picks do.
        # Only the window has the runner (to run it), the document (to record it) and the
        # dormant set (to release what the bake made redundant).
        self.inspector.dock_action.connect(self._on_dock_action)
        self.inspector.iterate_action.connect(self._on_iterate_action)
        self.inspector.movie_action.connect(self._on_movie_action)
        self.inspector.reload_requested.connect(self.reload_node_type)
        self.inspector.add_requested.connect(self._on_add_requested)
        self.inspector.append_requested.connect(self._on_append_requested)
        self.inspector.page_requested.connect(
            lambda pid: self._show_page(self._canvas, pid))
        # the panel-hosted drawing (2026-10-02): Draw Regions' controls live in its panel
        #: (page, node) to go back to after Apply/Cancel of a region drawn for it
        self._pick_return: Optional[Tuple[str, str]] = None
        self._draw_arm_pending: Optional[str] = None  # arm once this node's pull lands
        self.inspector.draw_control.connect(self._on_draw_control)
        self.inspector.region_requested.connect(self._on_region_requested)
        # linked pages (V4.00 step 6): the banner's Go to master / Make unique, and the
        # backstop that turns a refused structural edit into a status-bar hint
        self.inspector.linked_action.connect(self._on_linked_action)
        _install_linked_hook(self)
        self.runner.baked.connect(self._on_baked)
        self.runner.detail_ready.connect(self._on_detail_ready)

        self._sync_welcome()          # open on a blank, welcoming canvas
        self._sync_pages()            # the switcher, the palette's kind, the title

        # the saved panel layout (V4.00 step 3) — once every panel exists, before the window
        # is shown. Skipped under NODELAB_LAYOUT=0, and for a file that is missing, damaged
        # or from another layout generation: a layout must never stop the app starting.
        self._persist_layout = (LS.layout_enabled() if persist_layout is None
                                else bool(persist_layout))
        #: whether a saved layout (geometry included) was applied — the launcher then shows
        #: the window as saved instead of imposing its own size (`nodelab_v2.app.run`), and
        #: the default SIZES (`_apply_default_sizes`) are not imposed on the first show
        self._layout_restored = False
        #: the first `showEvent` has run (the default sizes are applied once, then)
        self._first_show_done = False
        if self._persist_layout:
            lay_path = LS.layout_path()
            had_file = lay_path.exists()
            try:
                self._layout_restored = bool(self.shell.restore_layout(quarantine=True))
            except Exception:                        # noqa: BLE001 — see above
                self.shell.apply_default_layout()
            if not self._layout_restored and had_file and not lay_path.exists():
                # the file was set aside (`<name>.rejected`): damaged, or from another layout
                # generation — V4.00 step 11a bumped it, so every pre-11a layout lands here
                self.statusBar().showMessage(
                    "panel layout reset to the new default (your old layout.json was set aside)")
        # a restored layout can put a viewer on screen while the first one is hidden: the
        # one on screen is the one to work in
        act = self.shell.active(VIEWER_KIND)
        shown = [d for d in self.shell.docks_of(VIEWER_KIND) if not d.isHidden()]
        if (act is None or act.isHidden()) and shown:
            self.shell.activate(shown[0])

    # ── panels (V4.00 step 3) ────────────────────────────────────────────────
    def _panel_specs(self) -> List[PanelSpec]:
        """The window's side panels, in View ▸ Panels order, with their default places."""
        left, right = Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea
        bottom, top = Qt.BottomDockWidgetArea, Qt.TopDockWidgetArea
        return [
            # several at once (V4.00 step 4): above the canvas; the first is made here and
            # is on screen from launch (step 11a), more by '+', View ▸ New or Compare
            PanelSpec(VIEWER_KIND, "Viewer", self._make_viewer, glyph="◉", multi=True,
                      default_area=top, default_hidden=False),
            # the active Viewer's controls (V4.00 step 11d): Playback under the Viewer (three
            # short rows — no scroll area, so it is never scrolled sideways), Channels beside
            # it, in a scroll area: a narrow panel wraps its channel columns onto more rows,
            # and a short one scrolls them rather than holding up the window's height
            PanelSpec(PLAYBACK_KIND, "Playback", lambda: self.playback_panel, glyph="▷",
                      default_area=top, split_from=VIEWER_KIND, split=Qt.Vertical),
            PanelSpec(CHANNELS_KIND, "Channels", lambda: self.channels_panel, glyph="◐",
                      default_area=top, split_from=PLAYBACK_KIND, split=Qt.Horizontal,
                      scroll=True),
            # more canvases (V4.00 step 5): '+', View ▸ New, or a page's "Open in a new
            # canvas"; the main canvas is the window's centre and is not one of these
            PanelSpec(CANVAS_KIND, "Canvas", self._make_canvas, glyph="⬚", multi=True,
                      default_area=bottom,
                      binding_of=lambda c: c.page_id,
                      apply_binding=self._apply_canvas_binding),
            # above the Nodes palette: every page of the workspace (V4.00 step 11)
            PanelSpec("pages", "Pages", lambda: self.pages_panel, glyph="▤",
                      default_area=left),
            PanelSpec("palette", "Nodes", lambda: self.palette, glyph="◫",
                      default_area=left),
            PanelSpec("inspector", "Properties", lambda: self.inspector, glyph="☰",
                      default_area=right, raise_default=True),
            PanelSpec("sheet", "Spreadsheet", lambda: self.sheet, glyph="▦",
                      default_area=right, tabify_with="inspector"),
            # inside a scroll area (`scroll`), like the Movie Editor: the Send tab's natural
            # width (~590 px) would otherwise be the MINIMUM of the whole right tab column,
            # holding it far wider than Properties and leaving a blank strip beside it
            PanelSpec("lablink", "LabLink", lambda: self.lablink, glyph="⇄",
                      default_area=right, tabify_with="inspector", scroll=True),
            PanelSpec("console", "Console", lambda: self.console, glyph="›",
                      default_area=bottom, allowed_areas=bottom | right,
                      default_hidden=True),
            # NOT tabbed with the Console: a tabbed dock takes the tab group's size and
            # ignores `resizeDocks`, so it opened at the Console's sliver of height with a
            # monitor 200 px tall. Inside a scroll area (`scroll`), so the editor's own
            # minimum size can never become the MAIN WINDOW's: a dock that cannot shrink
            # below its content grows the window instead, and a maximized window then runs
            # off the screen.
            PanelSpec("movie", "Movie Editor", lambda: self.movie_editor, glyph="▶",
                      default_area=bottom, allowed_areas=bottom | top | right,
                      default_hidden=True, scroll=True),
        ]

    def reset_layout(self) -> None:
        """View ▸ Reset layout: every panel docked back where a fresh install has it."""
        if self._maximized:
            self.set_maximized(False)        # the mini-map's viewer goes home first
        self.shell.reset_layout()
        # a fresh install shows ONE Viewer (the first); a further one stays on screen only
        # when it has a picture to show, and every one is re-sized on its next appearance
        for i, d in enumerate(self.shell.docks_of(VIEWER_KIND)):
            d._sized = False
            d.setVisible(i == 0 or d.panel.has_image())
        self._apply_default_sizes()
        self.statusBar().showMessage("layout reset — every panel is back in its default place")

    # ── viewers (V4.00 step 4) ────────────────────────────────────────────────
    def _make_viewer(self) -> ViewerPanel:
        """A new Viewer panel, wired to the window and the runner — the factory of the
        ``viewer`` dock kind, so every instance (the first, ``+``, View ▸ New ▸ Viewer,
        Compare, a restored layout) is wired the same way, and every signal it sends says
        which viewer sent it."""
        v = ViewerPanel()
        part = functools.partial
        slots = {
            "request_changed": part(self._on_view_request, viewer=v),
            "selection_changed": part(self._on_frame_selection, viewer=v),
            "region_changed": part(self._on_region_changed, viewer=v),
            "iteration_changed": part(self._on_iteration_changed, viewer=v),
            # the overlay SOURCE strip (2026-09-30): a stepper is a display-only runner
            # setting, a pin is a graph edit — the viewer does neither itself
            "overlay_step": part(self._on_overlay_step, viewer=v),
            "overlay_pin": part(self._on_overlay_pin, viewer=v),
            # What the live surface can hold decides whether a big frame is shown WHOLE at
            # full resolution or off the pyramid (V2.23). The surface knows the number, the
            # runner makes the decision, and neither should know about the other.
            "display_limits": self.runner.set_display_limits,
            "playing": part(self._on_playing, viewer=v),
            # the Viewer's settled LUT edits feed the Movie Editor's linked channels
            "display_changed": self._on_viewer_display,
            "pick_readout_changed": part(self._on_pick_readout, viewer=v),
            "pick_committed": part(self._on_pick_committed, viewer=v),
            "pick_armed": self._on_pick_armed,
            # a press on its image makes a viewer the active one
            "activated": part(self._activate_viewer, v),
        }
        for name, slot in slots.items():
            getattr(v, name).connect(slot)
        #: the window's slot on each of the viewer's signals (see `_viewer_slot`)
        v._window_slots = slots
        # The hover readout wants the UNPROCESSED file pixel beside the viewed node's. The
        # viewer holds no provider and the runner holds no notion of "the viewed node", so
        # the window — which has both — hands one over as a plain callback.
        v.raw_plane_cb = self.runner.raw_plane
        # viewport detail-on-demand: the panel asks (debounced, on pan/zoom), the runner
        # reads the rect off the GUI thread and answers on `detail_ready`.
        v.detail_cb = self._request_detail
        v.own_layers_cb = self._own_layers
        # its M/T/Z + play rows and its channel columns live in panels of their own
        self._home_controls(v)
        return v

    def _home_controls(self, v: ViewerPanel) -> None:
        """``v``'s two control sections into the Playback and Channels panels (V4.00 step
        11d) — on creation, and when the mini-map hands them back."""
        ax, ch = v.detach_controls()
        self.playback_panel.adopt(v, ax)
        self.channels_panel.adopt(v, ch)

    @staticmethod
    def _viewer_slot(v: ViewerPanel, signal: str):
        """The window's slot on viewer ``v``'s ``signal`` — what a test disconnects to
        drive the widget detached from the runner, and connects again after."""
        return v._window_slots[signal]

    @property
    def viewers(self) -> List[ViewerPanel]:
        """Every Viewer panel, in dock order (``viewer:0`` first)."""
        return [d.panel for d in self.shell.docks_of(VIEWER_KIND)]

    def _viewer_dock(self, v: Optional[ViewerPanel]):
        """``v``'s dock — matched by identity, so a viewer re-homed into the mini-map (out
        of its dock's widget tree) still has one."""
        if v is None:
            return None
        return next((d for d in self.shell.docks_of(VIEWER_KIND) if d.panel is v), None)

    def _active_viewer(self) -> Optional[ViewerPanel]:
        """The viewer the user works in (the shell's active ``viewer`` dock), or ``None``
        when every viewer has been closed."""
        d = self.shell.active(VIEWER_KIND)
        if d is None:
            docks = self.shell.docks_of(VIEWER_KIND)
            d = docks[0] if docks else None
        return d.panel if d is not None else None

    def _ensure_viewer(self) -> ViewerPanel:
        """The active viewer — or, after the last one was closed, a new one, hidden until a
        result opens it (the first Viewer of a session is on screen from launch; a
        replacement for one the user closed is not, until there is something to show)."""
        v = self._active_viewer()
        if v is None:
            d = self.shell.spawn(VIEWER_KIND, show=False)
            self.shell.activate(d)
            v = d.panel
        return v

    @property
    def viewer(self) -> ViewerPanel:
        """The active viewer (a new one if every viewer was closed) — what single-viewer
        code, and the GUI probe, mean by "the Viewer"."""
        return self._ensure_viewer()

    @property
    def viewer2(self) -> Optional[ViewerPanel]:
        """The newest Compare viewer, or ``None`` — V2.28's second pane, by its old name."""
        return list(self._links)[-1] if self._links else None

    def _repull_viewed(self) -> None:
        """Pull again what the ACTIVE viewer shows, on that node's own page — Shift+F5, and
        after F9, a Hold, a Bake or a Release. ``_viewed`` names the node only while its page
        is the active one, and a viewer may show another page's node (V4.00 step 5)."""
        v = self._active_viewer()
        b = getattr(v, "binding", None) if v is not None else None
        doc = self._page_doc(b[0]) if b else None
        if doc is not None and b[1] in doc.nodes:
            self.pull_node(b[1], viewer=v, page_id=b[0])

    def _bound_local(self, v: Optional[ViewerPanel]) -> Optional[str]:
        """The node ``v`` is bound to, when it is on the page the canvas shows."""
        b = getattr(v, "binding", None) if v is not None else None
        return b[1] if b and b[0] == (self.workspace.active or "") else None

    @property
    def _viewed(self) -> Optional[str]:
        """The node the ACTIVE viewer is bound to (V2's single ``_viewed``)."""
        return self._bound_local(self._active_viewer())

    @_viewed.setter
    def _viewed(self, node_id: Optional[str]) -> None:
        self._bind(self._ensure_viewer(), node_id)

    @property
    def _viewed2(self) -> Optional[str]:
        """The node the newest Compare viewer is bound to (V2.28's ``_viewed2``)."""
        return self._bound_local(self.viewer2)

    def _bind(self, v: ViewerPanel, node_id: Optional[str], *,
              page_id: Optional[str] = None) -> None:
        """Bind ``v`` to ``node_id`` of page ``page_id`` (default: the active page; ``None``
        unbinds): that node's results land in ``v`` from now on, its title names it, and —
        when ``v`` is the active viewer — the canvas marks the card it shows."""
        pid = page_id or self.workspace.active or ""
        if node_id is None:
            v.unbind()
        else:
            v.bind(pid, node_id)
            self._asker[qualify(pid, node_id)] = v
        self._sync_viewer_title(v)
        if v is self._active_viewer():
            self.scene.set_viewed(self._bound_local(v))
        self._sync_minimap_title()

    def _sync_viewer_title(self, v: ViewerPanel) -> None:
        """Name what ``v`` shows on its dock — title bar, floating caption, View ▸ Panels —
        as ``n3 · Gaussian`` (``Refinement › n3 · Gaussian`` once there are several pages);
        a Compare viewer adds whether its cursor follows its leader's (V2.28's header)."""
        d = self._viewer_dock(v)
        if d is None:
            return
        b = v.binding
        text = ""
        if b:
            page = self.workspace.pages.get(b[0])
            rec = page.doc.nodes.get(b[1]) if page is not None else None
            spec = rec.spec() if rec is not None else None
            label = spec.label if spec else (rec.op_key if rec is not None else "?")
            text = f"{b[1]} · {label}"
            if page is not None and len(self.workspace.pages) > 1:
                text = f"{page.name} › {text}"
        if v in self._links:
            tag = ("compare, linked — one cursor moves both" if v in self._linked
                   else "compare, own cursor (different M/T/Z)")
            text = f"{text} — {tag}" if text else tag
        d.set_binding_title(text)

    # ── run ids on every page (V4.00 step 5) ──────────────────────────────────
    def _page_of(self, run_id) -> str:
        """The page a run id is on (a bare id is the active page's)."""
        pid, _nid = split_run_id(str(run_id))
        return pid or (self.workspace.active or "")

    def _full(self, run_id) -> str:
        """``run_id`` page-qualified — how the viewers key what they show, so two pages'
        ``n3`` never share a LUT."""
        pid, nid = split_run_id(str(run_id))
        return qualify(pid or (self.workspace.active or ""), nid)

    def _key_on(self, run_id, page_id: str) -> str:
        """How the scene of ``page_id`` names a run: by its bare id when the run is on that
        page, by its run id when it is another page's (whose chain crosses this one)."""
        full = self._full(run_id)
        pid, nid = split_run_id(full)
        return nid if pid == page_id else full

    def _page_doc(self, page_id: str) -> Optional[GraphDocument]:
        page = self.workspace.pages.get(page_id or "")
        return page.doc if page is not None else None

    @staticmethod
    def _binding_full(v: Optional[ViewerPanel]) -> Optional[str]:
        """The run id of what ``v`` is bound to, any page."""
        b = getattr(v, "binding", None) if v is not None else None
        return qualify(b[0], b[1]) if b else None

    def _own_layers(self, run_id: str) -> list:
        """The viewer's ``own_layers_cb``: a node's own label layers, from its page."""
        pid, nid = split_run_id(self._full(run_id))
        doc = self._page_doc(pid)
        return list(doc.own_label_layers(nid)) if doc is not None else []

    # ── the Viewer's controls as panels (V4.00 step 11d) ──────────────────────
    def _control_docks(self) -> List[Any]:
        """The Playback and Channels docks (whichever exist)."""
        return [d for k in (PLAYBACK_KIND, CHANNELS_KIND) for d in self.shell.docks_of(k)]

    def _viewer_label(self, v: Optional[ViewerPanel]) -> str:
        """``"Viewer 2"`` (``"Viewer 2 · compare"``) — how the control panels name the Viewer
        they serve once there are several."""
        d = self._viewer_dock(v)
        if d is None:
            return ""
        return f"Viewer {d.index + 1}" + (" · compare" if v in self._links else "")

    def _sync_viewer_controls(self) -> None:
        """The Playback and Channels panels show the ACTIVE Viewer's sections. A Compare
        viewer whose cursor is linked to its leader's has its own row hidden — one set of
        sliders moves both — so Playback shows the LEADER's then; the mini-map viewer (canvas
        maximized) holds its own sections, compact, and both panels say so."""
        pb, ch = getattr(self, "playback_panel", None), getattr(self, "channels_panel", None)
        if pb is None or ch is None or not hasattr(self, "shell"):
            return
        v = self._active_viewer()
        ax = v
        if v is not None and v in self._linked and v is not self._mini_viewer:
            ax = self._links.get(v) or v
        pb.show_for(ax)
        ch.show_for(v)
        several = len(self.viewers) > 1
        for panel, who in ((pb, ax), (ch, v)):
            d = self.shell.dock_of(panel)
            if d is not None:
                d.set_binding_title(self._viewer_label(who) if several and who else "")

    def _activate_viewer(self, v: Optional[ViewerPanel]) -> None:
        d = self._viewer_dock(v)
        if d is not None:
            self.shell.activate(d)

    def _on_panel_activated(self, dock) -> None:
        """A panel became the one the user works in. A viewer takes with it what follows
        the active viewer: the canvas marks ITS card, the troubleshooting scope is read off
        its strips and region box, and the spreadsheet shows its result. A canvas makes
        its page the active page."""
        if dock.kind == CANVAS_KIND:
            self._activate_canvas(dock.panel)
            return
        if dock.kind != VIEWER_KIND:
            return
        v = dock.panel
        self._sync_viewer_controls()          # Playback and Channels follow it (step 11d)
        self.scene.set_viewed(self._bound_local(v))
        # the scope's INDICATORS move to this viewer (its region box, the chip); the
        # runner's scope does not — clicking a viewer is not an edit, and under F9 a changed
        # scope cancels the pull in flight. The next pull takes this viewer's (`pull_node`).
        self._sync_solo(push=False)
        nid, ds = v.showing()
        if ds is not None and nid:
            self.sheet.show_dataset(split_run_id(nid)[1], ds)

    def _prune_viewers(self) -> None:
        """A panel came or went: forget every reference to a viewer that was closed, and
        let a Compare viewer whose leader closed stand on its own."""
        alive = self.viewers
        for f in [f for f, lead in self._links.items() if f not in alive or lead not in alive]:
            self._links.pop(f, None)
            self._linked.discard(f)
            if f in alive:
                f.set_axes_hidden(False)
                self._sync_viewer_title(f)
        self._linked = {f for f in self._linked if f in self._links}
        for rid in [r for r, v in self._asker.items() if v not in alive]:
            self._asker.pop(rid, None)
        if self._preload_viewer is not None and self._preload_viewer not in alive:
            self._preload_viewer = None
        if self._preview_prev is not None and self._preview_prev[0] not in alive:
            self._preview_prev = None
        self._max_hidden = [d for d in self._max_hidden
                            if self.shell.docks.get(d.objectName()) is d]
        # a closed Viewer's control sections go with it
        self.playback_panel.prune(alive)
        self.channels_panel.prune(alive)
        self._sync_viewer_controls()

    def _allow_panel_close(self, dock) -> bool:
        """The shell's close veto — a pure QUESTION. It is asked when a close is attempted
        and again whenever the title bars paint their ✕ (``DockShell.sync_close_buttons``,
        V4.00 step 11a), so it must not act: an earlier version un-maximized the canvas from
        here, and asked after ``set_maximized(True)`` it undid the maximize it was asked
        about. While the canvas is maximized, the dock of the viewer living in the mini-map
        stays — closing it would leave that viewer with no home. A docked canvas that IS the
        maximized one may close; the viewer in its mini-map goes home as the canvas goes
        (:meth:`_prune_canvases`), the mini-map being a child of the canvas destroyed."""
        return not (dock.kind == VIEWER_KIND and self._maximized
                    and dock.panel is self._mini_viewer)

    def _prune_canvases(self) -> None:
        """A docked canvas closed. When it was the MAXIMIZED one, the viewer in its mini-map
        goes back to its dock first: the shell has taken the dock out of the window but the
        canvas widget lives until the event loop runs its ``deleteLater``, so the mini-map
        can still hand the viewer over. When it was the one worked in, the main canvas is."""
        if self._maximized and self._max_canvas is not None \
                and self._max_canvas not in self.canvases():
            self.set_maximized(False)
        if self._canvas is not self._main_canvas and self._canvas not in self.canvases():
            self._canvas = self._main_canvas
            if self._max_canvas is not None and self._max_canvas not in self.canvases():
                self._max_canvas = None
            self._activate_canvas(self._main_canvas)
        self._sync_canvas_accents()

    # ── pages and canvases (V4.00 step 5) ─────────────────────────────────────
    #
    # Every page of the workspace has its own SCENE (`scene_for`), made the first time a
    # canvas shows the page and kept until the page is deleted — so a page keeps its
    # selection, its cards' run states and its layout while no canvas shows it. A canvas
    # (`CanvasPanel`) shows one page at a time: the MAIN canvas is the central widget, every
    # further canvas a `canvas:<n>` dock. The canvas the user works in is the active one, and
    # its page is the workspace's ACTIVE page — what `doc`, `scene`, `view`, `minimap` and
    # `welcome` name, so everything that edits "the graph" edits the page being looked at.
    @property
    def doc(self) -> GraphDocument:
        """The active page's document."""
        return self.workspace.page(self.workspace.active).doc

    @property
    def scene(self) -> GraphScene:
        """The active page's scene."""
        return self.scene_for(self.workspace.active)

    @property
    def canvas(self) -> CanvasPanel:
        """The canvas the user works in."""
        return self._canvas

    @property
    def view(self):
        return self._canvas.view

    @property
    def minimap(self):
        """The mini-map hosting a viewer while maximized, else the active canvas's."""
        return (self._max_canvas or self._canvas).minimap

    @property
    def welcome(self):
        return self._canvas.welcome

    def canvases(self) -> List[CanvasPanel]:
        """Every canvas: the main one, then the docked ones in dock order."""
        return [self._main_canvas] + [d.panel for d in self.shell.docks_of(CANVAS_KIND)] \
            if hasattr(self, "shell") else [self._main_canvas]

    def scene_for(self, page_id: str) -> GraphScene:
        """``page_id``'s scene — made (and wired) the first time a canvas shows the page."""
        sc = self._scenes.get(page_id)
        if sc is None:
            sc = GraphScene(self.workspace.page(page_id).doc)
            self._scenes[page_id] = sc
            self._wire_scene(page_id, sc)
        return sc

    def _wire_scene(self, page_id: str, sc: GraphScene) -> None:
        """Connect a page's scene and document to the window. Everything a scene emits comes
        from a canvas the user is working in, which made its page the active one first (a
        press activates — `GraphView.pressed`), so the handlers act on the active page."""
        part = functools.partial
        sc.selectionChanged.connect(part(self._on_scene_selection, page_id))
        sc.node_activated.connect(self.pull_node)
        sc.pull_requested.connect(self.pull_node)
        sc.compare_requested.connect(self.open_compare)
        sc.nodes_deleted.connect(self._on_nodes_deleted)
        sc.ingest_requested.connect(self.ingest_source)
        sc.pick_requested.connect(self._arm_pick)
        sc.dock_action.connect(self._on_dock_action)
        sc.topology_refused.connect(lambda msg: self.statusBar().showMessage(msg, 8000))
        sc.topology_gate = self._scene_topology_gate
        sc.new_page_from_output.connect(part(self._new_page_from_output, page_id))
        doc = self.workspace.page(page_id).doc
        hooks = [part(self._on_doc_changed, page_id), part(self._on_page_doc_edit, page_id)]
        for fn in hooks:
            doc.on_change(fn)
        self._page_hooks[page_id] = (doc, hooks)

    def _drop_scene(self, page_id: str) -> None:
        """Forget a deleted page's scene (no canvas shows it any more)."""
        doc, hooks = self._page_hooks.pop(page_id, (None, []))
        for fn in hooks:
            if doc is not None:
                doc.off_change(fn)
        sc = self._scenes.pop(page_id, None)
        if sc is not None:
            sc.release()                    # its own listener too, not only the window's
            sc.deleteLater()

    def _on_page_doc_edit(self, page_id: str) -> None:
        """Any page's document changed: its canvases' welcome cards, and the inspector's
        derived values (G8 — it shows a node of the active page, which may read this one)."""
        self._sync_welcome()
        self.inspector.refresh_derived()
        page = self.workspace.pages.get(page_id)
        renamed = getattr(page.doc, "renamed_output", None) if page is not None else None
        if renamed:
            # a variable name another Output already has was made unique (V4.00 step 11e)
            page.doc.renamed_output = None
            _nid, asked, given = renamed
            self.statusBar().showMessage(
                f"another Page Output is already named “{asked}” — this one is “{given}” "
                f"(variable names are unique across the pages)", 8000)
            self.inspector.rebuild()

    def _make_canvas(self) -> CanvasPanel:
        """A new docked canvas, on the active page — the factory of the ``canvas`` kind."""
        c = CanvasPanel(self, self.workspace.active)
        self._wire_canvas(c)
        return c

    def _wire_canvas(self, c: CanvasPanel) -> None:
        part = functools.partial
        c.activated.connect(self._activate_canvas)
        c.page_changed.connect(self._on_canvas_page_changed)
        c.view.op_dropped.connect(self._on_op_dropped)
        c.view.files_dropped.connect(self._on_files_dropped)
        c.view.maximize_toggled.connect(part(self._on_canvas_maximize, c))
        c.minimap.restore_requested.connect(lambda: self.set_maximized(False))
        # the welcome card: the canvas it sits on becomes the one worked in first (its
        # buttons take focus, which activates a docked canvas; the main one is told here)
        for sig, fn in ((c.welcome.load_image_requested, self.file_load_source),
                        (c.welcome.load_sequence_requested, self.file_load_sequence),
                        (c.welcome.browse_nodes_requested, self.focus_palette),
                        (c.welcome.example_requested, self.build_example_workspace),
                        (c.welcome.goto_input_requested, self._goto_input_page)):
            sig.connect(lambda _=None, c=c, fn=fn: (self._activate_canvas(c), fn()))
        c.welcome.recipe_chosen.connect(
            lambda name, c=c: (self._activate_canvas(c), self._start_page_from_recipe(c, name)))
        c.welcome.link_requested.connect(
            lambda _=None, c=c: (self._activate_canvas(c), self.new_page_dialog(
                kind=self.workspace.pages[c.page_id].kind, canvas=c, into=c.page_id,
                start=PR.START_LINKED)))
        c.welcome.dismissed.connect(lambda _=None, c=c: self._dismiss_welcome(c))
        c.welcome.op_dropped.connect(
            lambda op, pos, c=c: (self._activate_canvas(c), self._on_op_dropped(op, pos)))
        c.welcome.recipe_requested.connect(
            lambda _=None, c=c: (self._activate_canvas(c), self.new_page_dialog(
                kind=self.workspace.pages[c.page_id].kind, canvas=c, into=c.page_id,
                start=PR.START_RECIPE)))

    def _apply_canvas_binding(self, c: CanvasPanel, page_id) -> None:
        """A restored canvas shows its saved page when the workspace has it."""
        if isinstance(page_id, str) and page_id in self.workspace.pages:
            c.set_page(page_id)

    def page_title(self, page_id: str) -> Tuple[str, str]:
        """``(name, kind)`` of a page, for a canvas's switcher."""
        page = self.workspace.pages.get(page_id)
        return (self._page_label(page), page.kind) if page is not None else (page_id, "free")

    def fill_page_menu(self, menu, c: CanvasPanel) -> None:
        """A canvas's page switcher: every page grouped by kind (the one shown ticked), then
        New page ▸ <kind>, Duplicate, Open in a new canvas, Rename…, Delete."""
        from nodegraph import roles as R
        menu.clear()
        ws = self.workspace
        for kind in R.page_kinds():
            pages = [p for p in ws.pages.values() if p.kind == kind]
            if not pages:
                continue
            menu.addSection(kind_label(kind))
            for p in pages:
                act = menu.addAction(kind_icon(kind), self._page_label(p))
                if p.master and p.master in ws.pages:
                    act.setToolTip(f"linked to “{ws.pages[p.master].name}” — its values are its "
                                   f"own, its graph is the master's")
                elif p.is_master:
                    act.setToolTip("a master page: offered first when a new page is linked to one")
                act.setCheckable(True)
                act.setChecked(p.id == c.page_id)
                act.triggered.connect(lambda _=False, pid=p.id, c=c: self._show_page(c, pid))
        menu.addSeparator()
        new = menu.addMenu("New page")
        for kind in R.page_kinds():
            act = new.addAction(kind_icon(kind), kind_label(kind))
            act.setToolTip(str(R.page_meta(kind).get("description") or ""))
            act.triggered.connect(lambda _=False, k=kind, c=c: self.new_page(k, canvas=c))
        shown = ws.pages.get(c.page_id)
        dlg = menu.addAction("New page…")
        dlg.setToolTip("A new page with its kind, name and start settled first: empty, from a "
                       "page recipe (a prebuilt page graph), or linked to a master page — and "
                       "which earlier Output it reads.")
        dlg.triggered.connect(
            lambda _=False, c=c, k=(PR.next_kind(shown.kind) or shown.kind) if shown else "refine":
            self.new_page_dialog(kind=k, canvas=c))
        menu.addSeparator()
        menu.addAction("Duplicate page").triggered.connect(
            lambda _=False, c=c: self.duplicate_page(c.page_id, canvas=c))
        lk = menu.addAction("Duplicate as linked page")
        lk.setToolTip("A page with the same nodes and wiring that follows every edit of the "
                      "master's graph, with parameter values of its own — one workflow "
                      "tuned per position or condition.")
        lk.triggered.connect(
            lambda _=False, c=c: self.duplicate_page(c.page_id, canvas=c, linked=True))
        mst = menu.addAction("Set as master page")
        mst.setCheckable(True)
        mst.setChecked(bool(shown is not None and shown.is_master))
        mst.setEnabled(shown is not None and not shown.master)
        mst.setToolTip("Offer this page first (★) when a new page is linked to a master. Any "
                       "plain page can still be chosen; a linked page cannot be a master.")
        mst.triggered.connect(
            lambda on, c=c: self.set_master_page(c.page_id, bool(on)))
        if shown is not None and shown.master:
            menu.addAction("Go to master page").triggered.connect(
                lambda _=False, c=c, m=shown.master: self._show_page(c, m))
            menu.addAction("Make unique").triggered.connect(
                lambda _=False, c=c: self.make_unique(c.page_id))
            ldoc = shown.doc
            if isinstance(ldoc, LinkedDocument) and ldoc.is_modified:
                rv = menu.addAction("Drop this page's own changes")
                rv.setToolTip("Remove the nodes and wiring this modified linked page keeps of "
                              "its own: back to the master's graph (its values stay).")
                rv.triggered.connect(lambda _=False, d=ldoc: d.revert_structure())
            elif isinstance(ldoc, LinkedDocument) and ldoc.edit_mode == EDIT_MASTER:
                st = menu.addAction("Stop sending edits to the master")
                st.setToolTip("The next change to this page's graph asks again how to apply.")
                st.triggered.connect(lambda _=False, d=ldoc: d.set_edit_mode(""))
        menu.addAction("Open in a new canvas").triggered.connect(
            lambda _=False, c=c: self.open_canvas(c.page_id))
        menu.addAction("Rename page…").triggered.connect(
            lambda _=False, c=c: self.rename_page(c.page_id))
        ct = menu.addAction("Close tab")
        ct.setToolTip("Take this page's tab off the tab row. The page is kept: the Pages panel "
                      "and its kind's tab show it again.")
        ct.setEnabled(any(p not in self._closed_pages and p != c.page_id for p in ws.pages))
        ct.triggered.connect(lambda _=False, c=c: self.close_page_tab(c.page_id))
        sv = menu.addAction("Save as page recipe…")
        sv.setToolTip("Keep this page's graph as a starting point for new pages (New page… ▸ "
                      "Page recipe). Not a LabLink recipe — Graph ▸ Publish is that.")
        sv.triggered.connect(lambda _=False, c=c: self.save_page_as_recipe(c.page_id))
        dele = menu.addAction("Delete page")
        dele.setEnabled(len(ws.pages) > 1)
        dele.triggered.connect(lambda _=False, c=c: self.delete_page(c.page_id))

    def _show_page(self, c: CanvasPanel, page_id: str) -> None:
        """Show ``page_id`` on canvas ``c`` and work there."""
        c.set_page(page_id)
        self._activate_canvas(c)

    # ── page tabs, the Pages panel, the start card (V4.00 step 11) ─────────────
    def page_tab_kinds(self) -> List[Tuple[str, str]]:
        """``(kind, label)`` for every page kind in pipeline order — the tab strip's top row
        (it shows the kinds that have pages)."""
        from nodegraph import roles as R
        return [(k, kind_label(k)) for k in R.page_kinds()]

    def page_tab_items(self) -> List[Tuple[str, str, str, str, bool]]:
        """``(page id, label, kind, tooltip, open)`` per page, in page order — the canvases'
        sub-tabs. ``open`` is False for a page whose tab was closed (:meth:`close_page_tab`)."""
        ws = self.workspace
        out = []
        for p in ws.pages.values():
            reads, pubs = ws.page_summary(p.id)
            tip = [f"{kind_label(p.kind)} page"]
            if reads:
                tip.append("reads: " + ", ".join(reads))
            if pubs:
                tip.append("publishes: " + ", ".join(pubs))
            tip.append("drag to reorder · double-click to rename · right-click for the page "
                       "menu · ✕ closes the tab (the page stays in the Pages panel)")
            out.append((p.id, self._page_label(p), p.kind, "\n".join(tip),
                        p.id not in self._closed_pages))
        return out

    def show_kind(self, c: CanvasPanel, kind: str) -> None:
        """The tab of page kind ``kind`` was clicked on canvas ``c``: show the page of that
        kind ``c`` showed last if its tab is open, else the first open one — and when every
        page of the kind has its tab closed, the one shown last (else the first), whose
        tab comes back."""
        ids = [p.id for p in self.workspace.pages.values() if p.kind == kind]
        if not ids:
            return
        opened = [p for p in ids if p not in self._closed_pages]
        last = c.last_of_kind.get(kind)
        if opened:
            self._show_page(c, last if last in opened else opened[0])
        else:
            self._show_page(c, last if last in ids else ids[0])

    def reorder_pages(self, order: List[str]) -> None:
        """Sub-tabs dragged into a new order: ``order`` is the whole page order with the
        dragged pages swapped among the places they held."""
        ws = self.workspace
        if sorted(order) != sorted(ws.pages) or order == list(ws.pages):
            return
        for i, pid in enumerate(order):
            if list(ws.pages).index(pid) != i:
                ws.move_page(pid, i)

    def close_page_tab(self, page_id: str) -> bool:
        """Close ``page_id``'s tab (its ✕, or Close tab in the page menu). The PAGE stays —
        in the workspace, in the Pages panel, on its kind's tab — and showing it again by any
        route opens its tab again. A canvas that was showing it moves to the nearest open
        page, of the same kind first. The last open tab cannot close: a canvas always shows
        a page. ``False`` when nothing closed."""
        ws = self.workspace
        if page_id not in ws.pages or page_id in self._closed_pages:
            return False
        order = list(ws.pages)
        opened = [p for p in order if p not in self._closed_pages and p != page_id]
        if not opened:
            self.statusBar().showMessage("the last open page tab stays — a canvas always "
                                         "shows a page")
            return False
        kind = ws.pages[page_id].kind
        at = order.index(page_id)

        def nearest(cands):
            return min(cands, key=lambda p: (abs(order.index(p) - at), order.index(p) > at),
                       default=None)

        dest = nearest([p for p in opened if ws.pages[p].kind == kind]) or nearest(opened)
        work = self._canvas
        for c in self.canvases():                     # every canvas off it FIRST: a page on
            if c.page_id == page_id:                  # screen counts as open
                c.set_page(dest)
        self._closed_pages.add(page_id)
        if work.page_id == dest:
            self._activate_canvas(work)
        self._refresh_page_views()
        self.statusBar().showMessage(
            f"“{ws.pages[page_id].name}” tab closed — the page is still in the Pages panel "
            f"and under its kind's tab")
        return True

    def next_page_kind(self, page_id: Optional[str]) -> str:
        """The kind a new page started from ``page_id`` most likely wants: the next one in
        the pipeline (Analysis, and a Free page, start another of their own kind)."""
        page = self.workspace.pages.get(page_id or "")
        if page is None:
            return "refine"
        return (PR.next_kind(page.kind) if page.kind != FREE_KIND else None) or page.kind

    def move_page(self, page_id: str, index: int) -> None:
        """A page tab dragged to ``index``: the page order follows."""
        if page_id in self.workspace.pages:
            self.workspace.move_page(page_id, index)

    def _refresh_page_views(self) -> None:
        """The Pages panel and every canvas's tabs follow the workspace (cheap: both rebuild
        only when what they show changed)."""
        from nodegraph import roles as R
        ws = self.workspace
        self._closed_pages &= set(ws.pages)
        for c in self.canvases():
            self._closed_pages.discard(c.page_id)          # what a canvas shows is open
            page = ws.pages.get(c.page_id)
            if page is not None:
                c.last_of_kind[page.kind] = page.id
        panel = getattr(self, "pages_panel", None)
        if panel is not None:
            rows = []
            for p in ws.pages.values():
                reads, pubs = ws.page_summary(p.id)
                master = ws.pages[p.master].name if p.master in ws.pages else ""
                rows.append((p.id, self._page_label(p), p.kind, master, tuple(reads),
                             tuple(pubs), p.id not in self._closed_pages,
                             tuple(ws.page_outline(p.id)), self._linked_state(p)))
            panel.refresh([(k, kind_label(k)) for k in R.page_kinds()], rows, ws.active or "")
        for c in self.canvases():
            c.sync_tabs()

    def _pages_menu(self, page_id: str, global_pos) -> None:
        """Right-click on the Pages panel: show that page, then the switcher's menu."""
        from PySide6.QtWidgets import QMenu
        if page_id != self._canvas.page_id:
            self._show_page(self._canvas, page_id)
        m = QMenu(self)
        self.fill_page_menu(m, self._canvas)
        m.exec(global_pos)

    def _frame_seed(self, c: CanvasPanel, node_id: str) -> None:
        """Bring a page's freshly seeded Page Input into view at the canvas's top left, at
        100 %, clear of the start banner along the bottom edge."""
        page = self.workspace.pages.get(c.page_id)
        rec = page.doc.nodes.get(node_id) if page is not None else None
        if rec is None:
            return
        vp = c.view.viewport()
        c.view.resetTransform()
        c.view.centerOn(QPointF(float(rec.x) + vp.width() / 2.0 - 48.0,
                                float(rec.y) + vp.height() / 2.0 - 48.0))

    def _recipes_for(self, kind: str) -> list:
        """The page recipes of ``kind`` (cached: the start card asks on every page change)."""
        cache = self.__dict__.setdefault("_recipe_cache", {})
        if kind not in cache:
            cache[kind] = PR.list_recipes(kind)
        return cache[kind]

    def _start_page_from_recipe(self, c: CanvasPanel, name: str) -> Optional[str]:
        """A recipe button on a page's start card: fill THAT page from the page recipe, its
        Page Input reading what the seeded one read."""
        page = self.workspace.pages.get(c.page_id)
        if page is None:
            return None
        recipe = next((r for r in self._recipes_for(page.kind) if r.name == name), None)
        if recipe is None:
            self.statusBar().showMessage(f"no page recipe “{name}” for this page", 6000)
            return None
        src = None
        for rec in page.doc.nodes.values():
            v = rec.params.get(PAGE_SOURCE_KEY) if rec.op_key == PAGE_INPUT_OP else None
            if v and self.workspace.resolve_source(page.id, v):
                src = str(v)
                break
        spec = PR.NewPageSpec(kind=page.kind, start=PR.START_RECIPE, recipe=recipe, source=src)
        pid = self._apply_new_page(spec, canvas=c, into=page.id)
        if pid:
            c.view.fit_all()
        return pid

    def _dismiss_welcome(self, c: CanvasPanel) -> None:
        """✕ / *Start empty* on a start card: hidden for that page for the session."""
        self._welcome_dismissed.add(c.page_id)
        self._sync_welcome()
        self.statusBar().showMessage(
            "start card hidden for this page — the page switcher's New page… offers the "
            "same starts", 6000)

    def _goto_input_page(self) -> None:
        pid = self._input_page(create=False)
        if pid:
            self._show_page(self._canvas, pid)

    # ── New page… / masters / page recipes (V4.00 step 11) ───────────────────
    @staticmethod
    def _override_count(page) -> int:
        doc = page.doc
        return int(doc.override_count()) if isinstance(doc, LinkedDocument) else 0

    def _page_label(self, page) -> str:
        """``★ name`` for a master page; ``name  (linked · N overrides)`` for a linked one —
        ``(linked · modified · N overrides)`` once it keeps changes of its own, ``(linked ·
        edits → master · …)`` while it sends them to its master (V4.00 step 11e)."""
        text = ("★ " if getattr(page, "is_master", False) else "") + page.name
        if page.master:
            n = self._override_count(page)
            mode = getattr(page.doc, "edit_mode", "")
            how = {EDIT_MODIFIED: "modified · ", EDIT_MASTER: "edits → master · "}.get(mode, "")
            text += f"  (linked · {how}{n} override{'' if n == 1 else 's'})"
        return text

    @staticmethod
    def _linked_state(page) -> str:
        """What a linked page keeps of its own STRUCTURE, for the Pages panel (V4.00 step
        11e): ``"modified · 2 own nodes, 1 removed, 3 wires"``, ``"edits go to the master"``,
        or ``""``."""
        doc = page.doc
        if not isinstance(doc, LinkedDocument):
            return ""
        if doc.edit_mode == EDIT_MASTER:
            return "edits go to the master (switched off there)"
        if not doc.is_modified:
            return ""
        own, gone, wires = doc.structure_counts()
        bits = [f"{own} own node{'' if own == 1 else 's'}"] if own else []
        bits += [f"{gone} removed"] if gone else []
        bits += [f"{wires} wire{'' if wires == 1 else 's'}"] if wires else []
        return "modified" + (" · " + ", ".join(bits) if bits else "")

    # ── the Pages panel's nodes (V4.00 step 11e) ──────────────────────────────
    def show_node(self, page_id: str, node_id: str) -> None:
        """Show ``page_id`` on the active canvas with ``node_id`` selected and in view."""
        c = self._canvas
        if c.page_id != page_id:
            self._show_page(c, page_id)
        sc = self._scenes.get(page_id)
        item = sc.node_items.get(node_id) if sc is not None else None
        if item is None:
            return
        sc.clearSelection()
        item.setSelected(True)
        c.view.centerOn(item)

    def set_node_muted(self, page_id: str, node_id: str, off: bool) -> bool:
        """Switch a node of ``page_id`` off (or on) — the Pages panel's switch. Refused, the
        status bar says why and the switch flips back."""
        page = self.workspace.pages.get(page_id)
        if page is None or node_id not in page.doc.nodes:
            return False
        try:
            page.doc.set_muted(node_id, bool(off))
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 8000)
            self.pages_panel._sig = None
            self._refresh_page_views()
            return False
        what = page.doc.title_of(node_id)
        self.statusBar().showMessage(
            f"{what} on “{page.name}” switched {'off — its input passes straight through' if off else 'on'}"
            + (" (this page only)" if isinstance(page.doc, LinkedDocument)
               and page.doc.is_overridden(node_id, "muted") else ""), 6000)
        return True

    def new_page_dialog(self, *, kind: str, canvas: Optional[CanvasPanel] = None,
                        source: Optional[str] = None, start: str = PR.START_EMPTY,
                        into: Optional[str] = None, master: Optional[str] = None,
                        recipe: Optional[str] = None) -> Optional[str]:
        """*New page…*: the dialog, then :meth:`_apply_new_page`. Returns the page id, or
        ``None`` when cancelled or refused (the reason is on the status bar)."""
        dlg = NewPageDialog(self, self.workspace, kind=kind, source=source, start=start,
                            into=into, master=master, recipe=recipe)
        if dlg.exec() != QDialog.Accepted:
            return None
        return self._apply_new_page(dlg.spec(), canvas=canvas, into=into)

    def _apply_new_page(self, spec, *, canvas: Optional[CanvasPanel] = None,
                        into: Optional[str] = None) -> Optional[str]:
        """Apply a :class:`~nodelab_v2.page_recipes.NewPageSpec` and show the page (the
        probe drives this without the dialog)."""
        try:
            page = PR.apply_new_page(self.workspace, spec, into=into)
        except (ValueError, LinkedPageError) as exc:
            self.statusBar().showMessage(f"new page refused: {exc}", 8000)
            return None
        self._show_page(canvas or self._canvas, page.id)
        if spec.start == PR.START_RECIPE and spec.recipe is not None:
            msg = f"new {kind_label(page.kind)} page “{page.name}” from the page recipe “{spec.recipe.name}”"
        elif spec.start == PR.START_LINKED and page.master in self.workspace.pages:
            master = self.workspace.pages[page.master].name
            msg = f"“{page.name}” follows “{master}”: change values here, edit the graph there"
        else:
            msg = f"new {kind_label(page.kind)} page “{page.name}”"
        reads = [str(r.params.get(PAGE_SOURCE_KEY) or "") for r in page.doc.nodes.values()
                 if r.op_key == PAGE_INPUT_OP]
        if reads:
            labels = dict(page.doc.source_choices(""))
            msg += f" — its Page Input reads {labels.get(reads[0], reads[0])}"
        self.statusBar().showMessage(msg)
        return page.id

    def _new_page_from_output(self, page_id: str, node_id: str) -> Optional[str]:
        """A Page Output's *New page from this output…*: the dialog, pre-set to a page of the
        next kind whose Page Input reads this Output."""
        page = self.workspace.pages.get(page_id)
        rec = page.doc.nodes.get(node_id) if page is not None else None
        if rec is None:
            return None
        name = str(rec.params.get(PAGE_NAME_KEY) or "").strip()
        if not name:
            self.statusBar().showMessage("give this Output a Name first", 6000)
            return None
        kind = PR.next_kind(page.kind) or page.kind
        start = PR.START_RECIPE if PR.list_recipes(kind) else PR.START_EMPTY
        return self.new_page_dialog(kind=kind, canvas=self._canvas,
                                    source=f"{page_id}:{name}", start=start)

    def set_master_page(self, page_id: str, on: bool) -> bool:
        """The switcher's *Set as master page*."""
        try:
            self.workspace.set_master(page_id, on)
        except (KeyError, ValueError) as exc:
            self.statusBar().showMessage(str(exc), 8000)
            return False
        page = self.workspace.pages[page_id]
        self.statusBar().showMessage(
            f"“{page.name}” is {'now' if on else 'no longer'} a master page"
            + (" — offered first when a new page is linked to one" if on else ""))
        return True

    def save_page_as_recipe(self, page_id: str, *, name: Optional[str] = None,
                            description: str = "") -> Optional[str]:
        """*Save as page recipe…*: the page's graph, as a starting point for new pages
        (:mod:`nodelab_v2.page_recipes`). Returns the file written, or ``None``."""
        page = self.workspace.pages.get(page_id)
        if page is None:
            return None
        if name is None:
            dlg = SavePageRecipeDialog(self, page)
            if dlg.exec() != QDialog.Accepted:
                return None
            name, description = dlg.name(), dlg.description()
        try:
            path = PR.save_recipe(PR.recipe_from_page(page, name, description))
        except (RuntimeError, OSError, ValueError) as exc:
            self.statusBar().showMessage(f"page recipe not saved: {exc}", 8000)
            return None
        self.__dict__.pop("_recipe_cache", None)     # the start cards list it from now on
        self.statusBar().showMessage(f"page recipe “{name}” written to {path}")
        return str(path)

    # ── where a loaded image goes (V4.00 step 11) ─────────────────────────────
    def _input_page(self, *, create: bool = True) -> Optional[str]:
        """The page a loaded image lands on: the first Image Input page — made first in the
        page order when the workspace has typed pages but no Input page (``""`` instead when
        ``create`` is False). ``None`` for a Free-only workspace (a pre-V4 file): there the
        image lands on the active page, as it always did."""
        ws = self.workspace
        for p in ws.pages.values():
            if p.kind == "input":
                return p.id
        if all(p.kind == FREE_KIND for p in ws.pages.values()):
            return None
        if not create:
            return ""
        return ws.add_page(kind_label("input"), "input", index=0).id

    def _source_load_allowed(self) -> bool:
        """Before a load dialog opens: may the page the image will land on take a new card?
        The Image Input page's document is the one that matters, not the active page's — a
        linked Input page refuses (hint on the status bar); a missing one will be made plain."""
        pid = self._input_page(create=False)
        if pid is None:
            return self._topology_ok(push_note=_LOAD_PUSH_NOTE)
        if not pid:
            return True
        return self._topology_ok(self.workspace.pages[pid].doc, push_note=_LOAD_PUSH_NOTE)

    def _begin_source_load(self) -> Optional[bool]:
        """Route a load to the Image Input page: switch the canvas there BEFORE any card is
        placed (the loaders read the active scene and view), and say whether to publish what
        lands — True on an Input page, False on a Free-only workspace (cards only, as before
        V4.00), None when the target page is linked and refuses a card (hint shown)."""
        pid = self._input_page()
        if pid is None:
            return False if self._topology_ok(push_note=_LOAD_PUSH_NOTE) else None
        if not self._topology_ok(self.workspace.pages[pid].doc, push_note=_LOAD_PUSH_NOTE):
            return None
        if pid != self.workspace.active:
            self._show_page(self._canvas, pid)
        return True

    def _publish_source(self, rec, stem: str, *, from_socket: str = "image"):
        """Name a freshly loaded source as a variable of its page (V4.00 step 11): a Page
        Output right of the card, wired from ``from_socket`` (the loader's whole dataset),
        named after the file — unique on the page — which is what a later page's Page Input
        reads. Returns the Output's record."""
        import os
        pid = self.workspace.active
        base = os.path.splitext(os.path.basename(str(stem or "")))[0]
        name = self.workspace.unique_output_name(pid, base)
        out = self.doc.add_node(PAGE_OUTPUT_OP, x=float(rec.x) + SOURCE_OUTPUT_GAP,
                                y=float(rec.y), params={PAGE_NAME_KEY: name})
        self.doc.connect(rec.id, from_socket, out.id, "data")
        return out

    def _published_note(self, outs: list) -> str:
        """``" — published as “raw” on Image Input"`` for the status bar."""
        if not outs:
            return ""
        page = self.workspace.pages.get(self.workspace.active or "")
        names = ", ".join(f"“{o.params.get(PAGE_NAME_KEY, '')}”" for o in outs[:3])
        more = len(outs) - min(len(outs), 3)
        return (f" — published as {names}" + (f" +{more} more" if more else "")
                + (f" on {page.name}" if page is not None else ""))

    def new_page(self, kind: str, *, canvas: Optional[CanvasPanel] = None,
                 name: Optional[str] = None) -> str:
        """Add a page of ``kind`` and show it on ``canvas`` (default: the active one). A page
        whose kind reads earlier pages starts with a Page Input already bound to the nearest
        named Output (V4.00 step 11, ``seed_input``) — when there is one to read."""
        page = self.workspace.add_page(name or kind_label(kind), kind, seed_input=True,
                                       index=self.workspace.insert_index_for(kind))
        self._show_page(canvas or self._canvas, page.id)
        seeds = [r.id for r in page.doc.nodes.values() if r.op_key == PAGE_INPUT_OP]
        if seeds:
            self._seeded_pages.add(page.id)
            self._frame_seed(canvas or self._canvas, seeds[0])
        reads = [str(r.params.get(PAGE_SOURCE_KEY) or "") for r in page.doc.nodes.values()
                 if r.op_key == PAGE_INPUT_OP]
        labels = dict(page.doc.source_choices(""))
        self.statusBar().showMessage(
            f"new {kind_label(kind)} page “{page.name}”"
            + (f" — its Page Input reads {labels.get(reads[0], reads[0])}" if reads else ""))
        return page.id

    def duplicate_page(self, page_id: str, *, canvas: Optional[CanvasPanel] = None,
                       linked: bool = False) -> str:
        """Copy a page (``linked``: a page LINKED to it — its master's graph, values of its
        own; V4.00 step 6) and show the copy on ``canvas``."""
        page = self.workspace.duplicate_page(page_id, dependent=linked)
        self._show_page(canvas or self._canvas, page.id)
        if linked:
            master = self.workspace.pages[page.master].name
            self.statusBar().showMessage(
                f"“{page.name}” follows “{master}”: change values here, edit the graph there")
        else:
            self.statusBar().showMessage(f"duplicated as “{page.name}”")
        return page.id

    def make_unique(self, page_id: str) -> bool:
        """A linked page becomes a page of its own, holding its current graph and values
        (V4.00 step 6). Its canvas, viewers and selection carry over by node id."""
        page = self.workspace.pages.get(page_id)
        if page is None or not page.master:
            return False
        old = self._scenes.get(page_id)
        try:
            sel = [i.node_id for i in old.selectedItems() if isinstance(i, NodeItem)] \
                if old is not None else []
        except RuntimeError:
            sel = []
        self.workspace.make_unique(page_id)   # its scene is rebuilt on the new document
        if sel and page_id in self._scenes:
            sc = self._scenes[page_id]
            for nid in sel:
                item = sc.node_items.get(nid)
                if item is not None:
                    item.setSelected(True)
            if page_id == self.workspace.active:
                first = sc.node_items.get(sel[0])
                if first is not None:
                    self.inspector.set_node(first)
        self.statusBar().showMessage(
            f"“{page.name}” is a page of its own now — its master's edits no longer reach it")
        return True

    def _topology_ok(self, doc=None, *, shape: bool = False, push_note: str = "") -> bool:
        """May the graph of ``doc`` (default: the active page's) change shape? A plain page:
        yes. A linked page (V4.00 step 11e) whose user already said how its edits apply — kept
        on the page, or sent to the master — yes, except for a frame, group or zone
        (``shape``). Otherwise ASK (:meth:`ask_linked_edit`): *Make unique* swaps the page's
        document, *Keep the change on this page* makes it a modified linked page, *Add it to
        the master, switched off* sends this session's edits there (refused for ``shape``, and
        greyed out with ``push_note`` saying why). Cancelled: no, with the hint."""
        doc = doc if doc is not None else self.doc
        if getattr(doc, "editable_topology", True):
            return True
        if isinstance(doc, LinkedDocument) and doc.edit_mode and not shape:
            return True
        pid = next((p.id for p in self.workspace.pages.values() if p.doc is doc), None)
        if pid is None or not isinstance(doc, LinkedDocument):
            self.statusBar().showMessage(TOPOLOGY_HINT, 8000)
            return False
        choice = self.ask_linked_edit(pid, shape=shape, push_note=push_note)
        page = self.workspace.pages[pid]
        master = self.workspace.pages[page.master].name \
            if page.master in self.workspace.pages else "its master"
        if choice == EDIT_UNIQUE:
            return self.make_unique(pid)
        if choice in (EDIT_MODIFIED, EDIT_MASTER) and not shape and \
                not (choice == EDIT_MASTER and push_note):
            doc.set_edit_mode(choice)
            self.statusBar().showMessage(
                f"“{page.name}” keeps its own changes now — “{master}”'s other edits still "
                f"arrive" if choice == EDIT_MODIFIED else
                f"“{page.name}” sends its edits to “{master}” this session: a node added here "
                f"arrives there switched off", 8000)
            return True
        self.statusBar().showMessage(SHAPE_HINT if shape else TOPOLOGY_HINT, 8000)
        return False

    def ask_linked_edit(self, page_id: str, *, shape: bool = False,
                        push_note: str = "") -> str:
        """How should a structural edit on linked page ``page_id`` apply? The dialog
        (:class:`~nodelab_v2.linked_edit_dialog.LinkedEditDialog`): one of
        :data:`EDIT_UNIQUE` / :data:`EDIT_MODIFIED` / :data:`EDIT_MASTER`, ``""`` when
        cancelled. A probe replaces this method on its window (a modal dialog waits for a
        click no offscreen run will make)."""
        page = self.workspace.pages[page_id]
        master = self.workspace.pages.get(page.master)
        dlg = LinkedEditDialog(self, page.name, master.name if master else "its master",
                               shape=shape, push_note=push_note)
        dlg.exec()
        return dlg.choice

    def _scene_topology_gate(self, scene, restart: bool):
        """:attr:`GraphScene.topology_gate`: ask for ``scene``'s linked page, and return the
        scene the gesture should run on — this one, or the page's new one once made unique —
        or ``None`` (cancelled, or a ``restart`` gesture whose press the dialog took)."""
        pid = next((p for p, sc in self._scenes.items() if sc is scene), None)
        if pid is None or not self._topology_ok(scene.doc):
            return None
        if restart:
            msg = self.statusBar().currentMessage()
            self.statusBar().showMessage((msg + " — " if msg else "") + "drag again", 8000)
            return None
        return self._scenes.get(pid)

    def _on_linked_action(self, action: str, node_id: str) -> None:
        """The inspector's Linked page banner, for a node of the ACTIVE page."""
        page = self.workspace.pages.get(self.workspace.active or "")
        if page is None or not page.master:
            return
        if action == "unique":
            self.make_unique(page.id)
        elif action == "master" and page.master in self.workspace.pages:
            self._show_page(self._canvas, page.master)
            self._select_only(node_id)

    def rename_page(self, page_id: str, name: Optional[str] = None) -> bool:
        page = self.workspace.pages.get(page_id)
        if page is None:
            return False
        if name is None:
            name, ok = QInputDialog.getText(self, "Rename page", "Page name:", text=page.name)
            if not ok:
                return False
        try:
            self.workspace.rename_page(page_id, name)
        except ValueError as exc:
            QMessageBox.information(self, "Rename page", str(exc))
            return False
        return True

    def delete_page(self, page_id: str, *, confirm: bool = True) -> bool:
        """Delete a page (never the last one). A page with nodes is confirmed first: Page
        Inputs on other pages that read its Outputs become unbound."""
        ws = self.workspace
        page = ws.pages.get(page_id)
        if page is None:
            return False
        if len(ws.pages) <= 1:
            self.statusBar().showMessage("the last page cannot be deleted")
            return False
        deps = ws.dependents_of(page_id)
        linked_note = ""
        if deps:
            names = ", ".join(f"“{ws.pages[d].name}”" for d in deps)
            linked_note = (f"\n\nIt is the master of {len(deps)} linked page(s): {names}. "
                           f"They become pages of their own, keeping their current graph "
                           f"and values.")
        if confirm and (page.doc.nodes or deps) and QMessageBox.question(
                self, "Delete page",
                f"Delete page “{page.name}” and its {len(page.doc.nodes)} node(s)?\n\n"
                f"Page Inputs on other pages that read its Outputs become unbound."
                + linked_note,
                QMessageBox.Yes | QMessageBox.Cancel) != QMessageBox.Yes:
            return False
        for d in deps:                      # its linked pages keep what they show
            ws.make_unique(d)
        try:
            ws.remove_page(page_id)
        except ValueError as exc:
            QMessageBox.information(self, "Delete page", str(exc))
            return False
        self.statusBar().showMessage(f"deleted page “{page.name}”")
        return True

    def open_canvas(self, page_id: Optional[str] = None) -> CanvasPanel:
        """Another canvas (a dock), showing ``page_id`` (default: the active page)."""
        d = self.shell.spawn(CANVAS_KIND)
        c = d.panel
        if page_id and page_id != c.page_id:
            c.set_page(page_id)
        self._activate_canvas(c)
        return c

    def step_page(self, step: int) -> None:
        """Ctrl+PgDn / Ctrl+PgUp: the active canvas shows the next / previous page."""
        order = list(self.workspace.pages)
        if len(order) < 2:
            return
        i = order.index(self._canvas.page_id) if self._canvas.page_id in order else 0
        self._show_page(self._canvas, order[(i + step) % len(order)])

    def _activate_canvas(self, c: CanvasPanel) -> None:
        """``c`` becomes the canvas the user works in, and its page the active page."""
        arrived = self.workspace.active != c.page_id
        if c is not self._canvas:
            self._canvas = c
            dock = self.shell.dock_of(c) if hasattr(self, "shell") else None
            if dock is not None:
                self.shell.activate(dock)
            elif hasattr(self, "shell"):
                # the MAIN canvas is no dock: the shell must not go on thinking a docked
                # canvas is the active one (a later close would activate the next docked
                # canvas, a press on a stale one would do nothing)
                self.shell.deactivate(CANVAS_KIND)
            self._sync_canvas_accents()
        if self.workspace.active != c.page_id and c.page_id in self.workspace.pages:
            self.workspace.set_active(c.page_id)         # → `_on_workspace_changed`
        else:
            self._sync_active_page_ui()
        # an empty downstream page shown with something to read starts with its Page Input
        # (V4.00 step 11; the standard pages exist before any image does) — ONCE per page:
        # an Input the user deleted does not come back on the next click
        if c.page_id not in self._seeded_pages:
            nid = self.workspace.seed_input(c.page_id)
            if nid:
                self._seeded_pages.add(c.page_id)
                self._frame_seed(c, nid)
                arrived = True
        if arrived:
            self._preview_page_reads(c.page_id)

    def _sync_canvas_accents(self) -> None:
        cs = self.canvases()
        for c in cs:
            c.set_active(c is self._canvas and len(cs) > 1)

    def _on_canvas_page_changed(self, c: CanvasPanel, page_id: str) -> None:
        dock = self.shell.dock_of(c) if hasattr(self, "shell") else None
        if dock is not None:
            page = self.workspace.pages.get(page_id)
            dock.set_binding_title(page.name if page is not None else page_id)
        self._sync_welcome()
        if c is self._canvas and self.workspace.active != page_id \
                and page_id in self.workspace.pages:
            self.workspace.set_active(page_id)

    def _on_canvas_maximize(self, c: CanvasPanel, on: bool) -> None:
        """A canvas's ⛶: it becomes the active canvas, then the window maximizes it."""
        if on and self._maximized and self._max_canvas is not c:
            self.set_maximized(False)
        self._activate_canvas(c)
        self.set_maximized(on)

    def _sync_pages(self) -> None:
        """Keep the GUI in step with the workspace's pages: canvases on a page that is gone
        move to the active one, its scene is dropped, viewers bound to it are unbound, the
        switchers and the window title follow renames — and the active canvas always shows
        the active page. Cheap: it runs on every workspace change (every edit included)."""
        ws = self.workspace
        if not ws.pages or ws.active not in ws.pages:
            return
        # a scene is stale when its page is gone — or when the page's DOCUMENT changed under
        # it: a file load keeps the active page's document object for the NEW active page,
        # so the scene cached under the old id would show another page's graph
        stale = {pid for pid, sc in self._scenes.items()
                 if pid not in ws.pages or getattr(sc, "doc", None) is not ws.pages[pid].doc}
        for pid in stale:
            self._drop_scene(pid)
        if ws.active in stale:
            # Properties holds a card of the scene just dropped: re-read the new one's
            # selection below (a Make unique swaps the document, not the page)
            self._ui_page = None
        for c in self.canvases():
            if c.page_id not in ws.pages:
                gone = c.page_id
                c.set_page(ws.active)
                c.forget_page(gone)
            elif c.page_id in stale:
                c.view.setScene(self.scene_for(c.page_id))
        for v in self.viewers:
            b = v.binding
            if b and b[0] not in ws.pages:
                self._bind(v, None)
        mp = getattr(self, "_movie_page", None)
        if mp is not None and (mp not in ws.pages or mp in stale):
            self._movie_page = None          # its movie's page is gone, or a load replaced it
            if getattr(self, "movie_editor", None) is not None:
                self.movie_editor.bind(None)
        if self._canvas.page_id != ws.active:
            self._canvas.set_page(ws.active)
        sig = tuple((p.id, p.name, p.kind) for p in ws.pages.values()) + (ws.active,)
        if sig != getattr(self, "_pages_sig", None):
            self._pages_sig = sig
            for c in self.canvases():
                c.sync_title()
                dock = self.shell.dock_of(c) if hasattr(self, "shell") else None
                if dock is not None:
                    dock.set_binding_title(ws.pages[c.page_id].name)
            before = getattr(self, "_ui_page", None)
            self._sync_active_page_ui()
            if getattr(self, "_ui_page", None) == before and hasattr(self, "inspector"):
                # a page added, renamed or deleted — from another canvas's switcher — while the
                # active page stayed: the shown node's Page Input Source menu and its
                # Ready-to-run read the page list, so the panel is rebuilt for it
                self.inspector.rebuild()
        elif getattr(self, "_ui_page", None) is None:
            self._sync_active_page_ui()
        # ★ and "(linked · N overrides)" ride the switcher's label (V4.00 step 11) — kept out
        # of `sig`, which would rebuild the Properties panel on every value edit of a linked page
        tsig = tuple((p.id, p.is_master, self._override_count(p),
                      getattr(p.doc, "edit_mode", "")) for p in ws.pages.values())
        if tsig != getattr(self, "_title_sig", None):
            self._title_sig = tsig
            for c in self.canvases():
                c.sync_title()
        self._refresh_page_views()

    def _sync_active_page_ui(self) -> None:
        """What follows the active page: the palette's kind, the inspector (the page's own
        selection), the canvas's viewed-card spine, the window title."""
        ws = self.workspace
        page = ws.pages.get(ws.active or "")
        if page is None:
            return
        if not hasattr(self, "palette"):
            return                              # still building the window
        self.palette.set_page_kind(page.kind, kind_label(page.kind))
        kl = kind_label(page.kind)
        self.setWindowTitle(f"{PRODUCT} — {kl}" + ("" if page.name == kl else f" · {page.name}"))
        if getattr(self, "_ui_page", None) != page.id:
            self._ui_page = page.id
            try:
                sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
            except RuntimeError:
                sel = []
            self.inspector.set_node(sel[0] if sel else None)
            self.scene.set_viewed(self._viewed)
            self._sync_solo()

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
        # the ACTIVE page's scene at the time — bound once, it would act on the first page
        self._dissolve_act.triggered.connect(lambda *_: self.scene.dissolve_selection())
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
        again.triggered.connect(lambda *_: self._repull_viewed())
        m_run.addAction(again)
        comp = QAction("&Compare selected beside viewed", self)
        comp.setShortcut("F8")
        comp.setToolTip(
            "Open the selected node's result in a Viewer docked beside the active one "
            "(its Compare viewer: a new one, or the one already comparing beside it).\n\n"
            "When both results span the same M/T/Z, the viewers share ONE cursor — the "
            "active viewer's strips move both — and the Compare viewer drops its own. Results "
            "with different extents each keep their own strips. Also on a card's "
            "right-click menu.")
        comp.triggered.connect(self.compare_selected)
        m_run.addAction(comp)
        close_comp = QAction("Close Compare &viewer", self)
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
        # the same mode, always in sight (V4.00 step 11d): Normal | Troubleshooting at the
        # right end of the menu bar — the action stays the one source of truth (F9, Run ▸)
        self.mode_switch = ModeSwitch(self._solo_act.toolTip())
        self.mode_switch.mode_changed.connect(self._solo_act.setChecked)
        self.menuBar().setCornerWidget(self.mode_switch, Qt.TopRightCorner)
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
        save_pr = QAction("Save page as a &page recipe…", self)
        save_pr.setToolTip(
            "Keep the active page's graph as a starting point for new pages (the page "
            "switcher's New page… ▸ Page recipe). Not a LabLink recipe — Publish is that.")
        save_pr.triggered.connect(lambda: self.save_page_as_recipe(self.workspace.active))
        m_graph.addAction(save_pr)

        m_view = self.menuBar().addMenu("&View")
        fit = QAction("&Fit graph", self)
        fit.setShortcut("Home")
        fit.triggered.connect(lambda: self.view.fit_all())
        m_view.addAction(fit)
        self._max_act = QAction("&Maximize node canvas", self)
        self._max_act.setCheckable(True)
        self._max_act.setShortcut("Ctrl+Space")
        self._max_act.setToolTip("Give the whole centre to the graph; the Viewer becomes "
                                 "a mini-map in the canvas' top-left corner")
        self._max_act.toggled.connect(self.set_maximized)
        m_view.addAction(self._max_act)
        # pages (V4.00 step 5): the active canvas steps through the workspace's pages
        for text, seq, step in (("Next &page", "Ctrl+PgDown", 1),
                                ("Previous p&age", "Ctrl+PgUp", -1)):
            act = QAction(text, self)
            act.setShortcut(seq)
            act.setToolTip("Show the next / previous page of the workspace on the canvas "
                           "you are working in")
            act.triggered.connect(lambda _=False, s=step: self.step_page(s))
            m_view.addAction(act)
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
        # panels (V4.00 step 3): a tick per panel (the Console keeps the Ctrl+` action),
        # New ▸ for the kinds that can have several instances, and Reset layout
        self._panels_menu = m_view.addMenu("&Panels")
        _console_override = {self._console_dock.objectName(): self._console_act}
        self._panels_menu.aboutToShow.connect(
            lambda: self.shell.fill_panels_menu(self._panels_menu, _console_override))
        self.shell.fill_panels_menu(self._panels_menu, _console_override)
        self._new_menu = m_view.addMenu("&New")
        self._new_menu.aboutToShow.connect(lambda: self.shell.fill_new_menu(self._new_menu))
        self.shell.changed.connect(lambda: self.shell.fill_new_menu(self._new_menu))
        self.shell.fill_new_menu(self._new_menu)
        reset = QAction("&Reset layout", self)
        reset.setToolTip("Put every panel back where a fresh install has it — docked, in "
                         "its default place")
        reset.triggered.connect(self.reset_layout)
        m_view.addAction(reset)
        m_view.addSeparator()
        ovl = QAction("&Overlays…", self)
        ovl.setShortcut("Ctrl+Shift+O")
        ovl.setToolTip("Configure the Point / Label / Track overlays — size, opacity, "
                       "look, colour — and save the look as a default")
        ovl.triggered.connect(lambda: self.viewer.open_overlay_dialog())
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

    @_needs_topology
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
            self, f"{PRODUCT} — capabilities",
            f"{PRODUCT} {APP_VERSION}.\n\n"
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
        QApplication.instance().setPalette(T.palette())   # light mode gets the light palette
        self.setStyleSheet(_window_qss())
        for panel in (self.palette, self.pages_panel, self.inspector, self.sheet, self.lablink,
                      self.console, self.movie_editor, self.playback_panel,
                      self.channels_panel, *self.viewers, *self.canvases()):
            panel.restyle()
        self.shell.restyle()       # the panels' title bars, floating or docked
        self.mode_switch.restyle()
        self._paint_led()          # the LED colors come from the tokens, not from QSS
        self._sync_solo_chip()     # ditto for the solo chip's amber
        for c in self.canvases():
            c.view.setBackgroundBrush(T.BG)
            c.view.viewport().update()
        for sc in self._scenes.values():
            sc.update()

    # ── maximized canvas + mini-map (2026-07-27) ──────────────────────────────
    def set_maximized(self, on: bool) -> None:
        """Toggle the maximized node canvas.

        **On** — the docked Viewers step aside (hidden, not closed), so the canvas owns the
        window's centre, and the active one — the first docked one when the active viewer
        floats — is re-homed into the :class:`~nodelab_v2.minimap.MiniMapOverlay` pinned to
        the canvas' top-left corner, trimmed to its compact layout, with click-to-preview
        forced on so the mini-map follows the node you're working on. A FLOATING viewer
        stays where the user put it: it is already beside the canvas.
        **Off** — the Viewer goes back into its dock, the docks that stepped aside come back
        at their old sizes, and click-to-preview returns to whatever the user had chosen.

        The same live ViewerPanel widget is moved (never a second copy), so channels,
        LUT, playback and overlays carry straight across."""
        on = bool(on)
        if on == self._maximized:
            return
        self._maximized = on
        if on:
            docked = [d for d in self.shell.docks_of(VIEWER_KIND) if not d.isFloating()]
            ad = self._viewer_dock(self._active_viewer())
            if ad is not None and not ad.isFloating():
                mini_dock = ad
            elif docked:
                mini_dock = docked[0]
            else:                              # every viewer floats, or none is left
                mini_dock = self.shell.spawn(VIEWER_KIND, show=False)
            self._max_hidden = [d for d in docked if not d.isHidden()]
            # their heights: hiding every dock of an area collapses it, and showing them
            # again would lay them out from their size hints instead of where they were
            self._max_heights = {d.objectName(): d.height() for d in self._max_hidden}
            for d in self._max_hidden:
                d.hide()
            mini = mini_dock.panel
            self._mini_viewer = mini
            self._max_canvas = self._canvas
            self.shell.activate(mini_dock)
            # its dock is empty while the viewer lives in the mini-map: View ▸ Panels must
            # not open it as an empty panel (Esc or the dock button bring the viewer back)
            mini_dock.toggleViewAction().setEnabled(False)
            # a linked Compare viewer leaves its leader behind (hidden): it scrubs alone
            mini.set_axes_hidden(False)
            # its controls come with it, compact (V4.00 step 11d): the docked Playback and
            # Channels panels step aside like the Viewers, a floating one stays and says so
            self._max_ctrl_hidden = [d for d in self._control_docks()
                                     if not d.isHidden() and not d.isFloating()]
            for d in self._max_ctrl_hidden:
                d.hide()
            mini.attach_controls()
            mini.set_compact(True)
            self.minimap.attach(mini)
            self.minimap.reposition()
            self.minimap.show()
            self.minimap.raise_()
            self._follow_before_max = self._follow_act.isChecked()
            self._follow_act.setChecked(True)
            # the dock area re-lays out on the next turn — re-anchor once the canvas has
            # actually grown into the freed space
            QTimer.singleShot(0, self.minimap.reposition)
        else:
            mini, self._mini_viewer = self._mini_viewer, None
            self.minimap.detach()
            self.minimap.hide()
            max_view = self.minimap.parentWidget()
            md = self._viewer_dock(mini)
            if mini is not None and md is not None:
                mini.set_compact(False)
                self._home_controls(mini)      # its controls back into their panels
                md.setWidget(mini)             # back into its own dock
                mini.show()
                md.toggleViewAction().setEnabled(True)
                if md not in self._max_hidden:
                    self._max_hidden.append(md)
                self._sync_links_of(mini)      # a linked Compare viewer hides its strips again
            back = [d for d in self._max_hidden if self.shell.docks.get(d.objectName()) is d]
            for d in back:
                self._show_viewer_dock(d)
            heights = getattr(self, "_max_heights", {})
            sized = [d for d in back if d.objectName() in heights and not d.isFloating()
                     and self.dockWidgetArea(d) in (Qt.TopDockWidgetArea,
                                                    Qt.BottomDockWidgetArea)]
            if sized:
                hs = [heights[d.objectName()] for d in sized]
                self.resizeDocks(sized, hs, Qt.Vertical)
                QTimer.singleShot(0, lambda: self._resize_docks_alive(sized, hs))
            self._max_hidden, self._max_heights = [], {}
            for d in getattr(self, "_max_ctrl_hidden", []):
                if self.shell.docks.get(d.objectName()) is d:
                    d.show()
            self._max_ctrl_hidden = []
            self._follow_act.setChecked(getattr(self, "_follow_before_max", False))
        # keep both entry points (canvas button + View menu) in sync, no signal loop
        if on:
            self._max_canvas.view.set_maximized(True)
        else:
            if isinstance(max_view, GraphView):
                max_view.set_maximized(False)
            self._max_canvas = None
        self._max_act.blockSignals(True)
        self._max_act.setChecked(on)
        self._max_act.blockSignals(False)
        self._sync_minimap_title()
        self._sync_viewer_controls()
        # the close veto (`_allow_panel_close`) follows this state: the mini-map viewer's
        # dock shows its ✕ disabled while maximized, live again once docked back
        self.shell.sync_close_buttons()
        self.statusBar().showMessage(
            "canvas maximized — click any node to preview it in the mini-map (Esc to "
            "dock the Viewer back)" if on else "Viewer docked")

    def _open_viewer(self, v: Optional[ViewerPanel] = None) -> None:
        """Make sure viewer ``v`` (default: the active one) is on screen before it shows a
        result: a viewer can be hidden by hand, and one spawned to replace the last closed
        viewer starts hidden. A viewer the user has placed and sized is left exactly as it
        is. While the canvas is maximized a docked viewer waits for the restore."""
        v = v if v is not None else self._active_viewer()
        d = self._viewer_dock(v)
        if d is None or v is self._mini_viewer:
            return
        if self._maximized and not d.isFloating():
            if d not in self._max_hidden:
                self._max_hidden.append(d)     # back on screen with the docked layout
            return
        if d.isHidden():
            self._show_viewer_dock(d)

    def _column_height(self) -> int:
        """The canvas column's height — the window between the menu bar and the status bar —
        which the Viewer's share (:data:`VIEWER_SHARE`) is measured against, the same way
        in the default layout and when a Viewer dock first opens."""
        return max(1, self.height() - self.menuBar().height() - self.statusBar().height())

    def _apply_default_sizes(self) -> None:
        """The default layout's SIZES (V4.00 step 11a). ``DockShell.apply_default_layout``
        only PLACES the docks, and Qt then sizes them from their hints: the Viewer at its
        minimum, the side columns at whatever their widgets ask. Here the first Viewer takes
        :data:`VIEWER_SHARE` of the canvas column, Nodes :data:`PALETTE_W` and Properties
        :data:`INSPECTOR_W`. Run on the first show (a dock resized before the window is laid
        out is overridden by its minimum) when no saved layout was restored, and after View ▸
        Reset layout. A floating or hidden dock is left alone.

        A request below a dock's minimum (or against a fixed size) is clamped by Qt, so the
        Viewer sits at its minimum height on a short window, and the Properties column is
        today held at the LabLink tab's minimum width (868 px, measured 2026-10-06) with the
        panel itself fixed at 376 (``InspectorPanel.setFixedWidth``) — that request is a
        no-op until those panels can shrink."""
        viewers = self.shell.docks_of(VIEWER_KIND)
        if viewers:
            d = viewers[0]
            if not d.isHidden() and not d.isFloating() and self.dockWidgetArea(d) in (
                    Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea):
                docks, sizes = [d], [max(1, int(VIEWER_SHARE * self._column_height()))]
                # the Playback / Channels row under it (step 11d), when it is there
                ctl = next((c for c in self._control_docks() if not c.isHidden()
                            and not c.isFloating() and self.dockWidgetArea(c)
                            == self.dockWidgetArea(d)), None)
                if ctl is not None:
                    docks.append(ctl)
                    sizes.append(CONTROLS_H)
                self.resizeDocks(docks, sizes, Qt.Vertical)
                # side by side: Playback needs a strip's width, Channels a column per channel
                row = [c for c in self._control_docks() if not c.isHidden()
                       and not c.isFloating() and self.dockWidgetArea(c)
                       == self.dockWidgetArea(d)]
                if len(row) == 2:
                    w = sum(c.width() for c in row)
                    pb = row[0] if row[0].kind == PLAYBACK_KIND else row[1]
                    pw = min(w // 2, self.playback_panel.minimumSizeHint().width() + 24)
                    self.resizeDocks([pb, *[c for c in row if c is not pb]], [pw, w - pw],
                                     Qt.Horizontal)
            d._sized = True                # its first appearance is sized: this is it
        for name, w in (("palette:0", PALETTE_W), ("inspector:0", INSPECTOR_W)):
            sd = self.shell.dock(name)
            if sd is not None and not sd.isHidden() and not sd.isFloating():
                self.resizeDocks([sd], [w], Qt.Horizontal)
        pd = self.shell.dock("pages:0")          # the Pages panel: a third of the column
        if pd is not None and not pd.isHidden() and not pd.isFloating():
            self.resizeDocks([pd], [max(120, int(PAGES_SHARE * self._column_height()))],
                             Qt.Vertical)

    def showEvent(self, event) -> None:                      # noqa: N802 — Qt override
        super().showEvent(event)
        if not getattr(self, "_first_show_done", True):
            self._first_show_done = True
            if not self._layout_restored:
                # after this show is laid out: a resize inside the show itself is overridden
                QTimer.singleShot(0, self._apply_default_sizes)

    def _show_viewer_dock(self, d) -> None:
        """Show a Viewer dock. Its FIRST appearance takes the Viewer's share of the canvas
        column (:data:`VIEWER_SHARE`, measured as the default layout measures it) — a fresh
        dock would open at its minimum height; after that the user's own size stands."""
        col = self._column_height()
        d.show()
        if getattr(d, "_sized", False) or d.isFloating():
            return
        d._sized = True
        if self.dockWidgetArea(d) in (Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea):
            want = max(1, int(VIEWER_SHARE * col))
            self.resizeDocks([d], [want], Qt.Vertical)
            # …and once more after the show is laid out, which a dock shown this very turn
            # can otherwise override with its minimum height
            QTimer.singleShot(0, lambda: self._resize_viewer_dock(d, want))

    def _resize_viewer_dock(self, d, want: int) -> None:
        self._resize_docks_alive([d], [want])

    def _resize_docks_alive(self, docks, heights) -> None:
        """``resizeDocks`` on the docks of ``docks`` that are still open, docked and on
        screen (a deferred resize may find one closed or popped out meanwhile)."""
        keep = [(d, h) for d, h in zip(docks, heights)
                if self.shell.docks.get(d.objectName()) is d and d.isVisible()
                and not d.isFloating()]
        if keep:
            self.resizeDocks([d for d, _ in keep], [h for _, h in keep], Qt.Vertical)

    def _sync_minimap_title(self) -> None:
        """The mini-map header names what it is showing (id · node label)."""
        v = self._mini_viewer if self._mini_viewer is not None else self._active_viewer()
        nid = self._bound_local(v)
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
            self._preview_pull(nid)

    def _previews_on_select(self, rec) -> bool:
        """Does selecting this card show it in the active Viewer even with click-to-preview
        off? A card that exists to be LOOKED at (a Viewer node, a plot), and since V4.00
        step 11 a page boundary that carries data: a Page Input that resolves (what this page
        reads) and a Page Output with something wired in (what a later page will read)."""
        op = getattr(rec, "op_key", "")
        if is_visual_output(op):
            return True
        if op == PAGE_INPUT_OP:
            return bool(self.workspace.resolve_source(
                self.workspace.active or "", rec.params.get(PAGE_SOURCE_KEY)))
        if op == PAGE_OUTPUT_OP:
            return any(e[2] == rec.id for e in self.doc.edges)
        return False

    def _preview_page_reads(self, page_id: str) -> None:
        """Arriving on a page while the active Viewer shows another page's node (or
        nothing): show what THIS page reads — its first Page Input that resolves (V4.00 step
        11). Without it, a page that reads one split position of an image went on showing
        the load's preview of every position from Image Input, which reads as the wrong
        data reaching the page. A preview: it starts no ingest and queues behind no run."""
        page = self.workspace.pages.get(page_id)
        v = self._active_viewer()
        if page is None or v is None:
            return
        b = getattr(v, "binding", None)
        if b and b[0] == page_id:
            return
        for rec in page.doc.nodes.values():
            if rec.op_key == PAGE_INPUT_OP and self.workspace.resolve_source(
                    page_id, rec.params.get(PAGE_SOURCE_KEY)):
                self._preview_pull(rec.id)
                return

    def _preview_pull(self, node_id: str) -> None:
        """A pull the SELECTION made (click-to-preview, a visual card) rather than one the
        user asked for. Remembers what the active viewer showed before, so that F8 on the
        card just selected compares it beside THAT (:meth:`open_compare`), not beside
        itself."""
        v = self._active_viewer()
        before = self._bound_local(v)
        self.pull_node(node_id, allow_ingest=False, queue=False)
        if v is not None and before and before != node_id \
                and self._bound_local(v) == node_id:
            self._preview_prev = (v, node_id, before)

    # ── run ids (V4.00 step 2) ────────────────────────────────────────────────
    #
    # The runner speaks PAGE-QUALIFIED run ids, ``"pg1/n3"``: every signal carries one and
    # every method accepts one (or a bare id, which it takes to be on the active page). The
    # canvas, the viewer and the inspector still speak the active page's bare node ids —
    # until step 5 puts several pages on screen — so each runner handler first asks which
    # page a run id is on and touches the scene only when it is the page being shown.
    def _local(self, run_id: str) -> Optional[str]:
        """``run_id``'s bare node id when it is on the ACTIVE page (or bare), else ``None``."""
        pid, nid = split_run_id(str(run_id))
        return nid if (not pid or pid == self.workspace.active) else None

    def _local_ids(self, run_ids) -> List[str]:
        """The bare ids among ``run_ids`` that belong to the active page, in order."""
        out = [self._local(r) for r in run_ids]
        return [n for n in out if n is not None]

    def _on_workspace_changed(self) -> None:
        """A page edit, anywhere in the workspace: cancel only the runs it can affect.
        ``last_touched`` is already qualified — ``None`` is "unknown, assume everything",
        ``frozenset()`` is "nothing a run can see" (the G8 re-seed, a page switch)."""
        self.runner.invalidate(self.workspace.last_touched)
        self._sync_pages()

    def _on_run_started(self, run_id: str) -> None:
        full = self._full(run_id)
        self.statusBar().showMessage(f"pulling {self._local(run_id) or full}…")
        for p in self._viewers_for(full):
            p.show_running(full)
        self._set_led("busy")
        self.minimap.set_state("busy")

    def _on_detail_ready(self, run_id: str, planes, rect01, coords=None) -> None:
        full = self._full(run_id)
        for pane in self.viewers:            # each keeps the patch only for its own view
            pane.on_detail_ready(full, planes, rect01, coords)

    # ── document plumbing ─────────────────────────────────────────────────────
    def _on_doc_changed(self, page_id: Optional[str] = None) -> None:
        """A page's document changed (``page_id``; default: the active page)."""
        pid = page_id or self.workspace.active or ""
        doc = self._page_doc(pid)
        if doc is None:
            return
        # Only the runs this edit could have changed (2026-08-06). `last_touched` is the node
        # the inspector or the card just wrote, or None for a structural edit that no single
        # node accounts for. (The runner was told through the workspace —
        # `_on_workspace_changed` — whose touched set is page-qualified; this handler keeps
        # the page's canvas honest.) A terminal `done` badge OUTLIVES an unrelated branch
        # starting, so the edit that actually invalidates a result has to retire it: the
        # edited nodes and everything DOWNSTREAM of them; an unscoped edit retires the lot.
        touched = doc.last_touched
        sc = self._scenes.get(pid)
        if sc is not None:
            if touched is None:
                sc.clear_run_states()
            elif touched:
                sc.clear_run_states_for(doc.downstream_of(touched))
        # a viewer bound to a node of this page that is gone (deleted, or reloaded away): a
        # Compare viewer closes — a pane still showing a node that is no longer on the canvas
        # is a lie, and the one the user cannot detect — and any other viewer is unbound
        for v in self.viewers:
            b = v.binding
            if not b or b[0] != pid or b[1] in doc.nodes:
                continue
            if v in self._links and v is not self._mini_viewer:
                d = self._viewer_dock(v)
                if d is not None:
                    d.close()
                continue
            if v in self._links:          # in the mini-map, it cannot close: it stands alone
                self._links.pop(v, None)
                self._linked.discard(v)
                v.set_axes_hidden(False)
            was_active = v is self._active_viewer()
            self._bind(v, None)
            if was_active:
                self.minimap.set_state("idle")
        self._sync_solo()                 # a rewired source changes the frame count
        if pid == self._movie_pid():        # the editor's movie lives on ITS page
            self.movie_editor.on_doc_changed(touched, doc.downstream_of)
        if pid == self.workspace.active:
            name = doc.path or "untitled"
            self.statusBar().showMessage(f"{name} — rev {doc.revision}")

    def _on_scene_selection(self, page_id: str) -> None:
        """A page's selection changed — only the ACTIVE page's drives the inspector and
        click-to-preview (another page's changes only programmatically)."""
        if page_id == self.workspace.active:
            self._on_selection()

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
        elif (len(sel) == 1 and self._previews_on_select(sel[0].rec)
              and sel[0].node_id != self._viewed):
            # a VISUAL card (a Viewer node, a plot) exists to be looked at, so selecting it
            # shows it in the active viewer even with click-to-preview off (V4.00 step 4) —
            # and, like a preview, never starts an ingest or queues behind a running pull;
            # so does a page boundary that carries data (step 11)
            self._preview_pull(sel[0].node_id)

    # ── interactive parameter picking (V2.16) ────────────────────────────────
    def _arm_pick(self, req) -> None:
        """Arm a pick on the viewer, with the target node's own calibration.

        The calibration is the node's PROPAGATED envelope, not the source file's: a Resample
        or a Crop upstream changes what one pixel is worth, and a radius picked on the image
        has to be expressed in the microns *this* node will convert back to pixels. That is
        precisely what ``propagate_meta`` already tracks, so the pick inherits it for free."""
        if req.kind == "shapes" and not req.tools_in_panel:
            # A drawn region is never configured on the image (2026-10-02). From a node that
            # WANTS a region, go and draw it on a Draw Regions node in line; from a node
            # that IS the drawing, arm it with its controls in the panel. Either way the
            # card's ◎ glyph and the panel's button take the same route.
            rec = self.doc.nodes.get(req.node_id)
            alt = RD.EMPTY_PICKS.get((rec.op_key, req.socket)) if rec is not None else None
            if alt and alt[0]:
                self._on_region_requested(req.node_id, req.socket)
            else:
                self._arm_draw(req.node_id)
            return
        v = self._active_viewer()
        if req.surface != "instant" and (v is None or not v.has_image()):
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
        v = v if v is not None else self._ensure_viewer()
        self._open_viewer(v)
        # the pick is for a node of the ACTIVE page: say so, so that it is written back
        # there even if another page is active by the time it is applied
        v.arm_pick(req, Calibration.from_metadata(md), page_id=self.workspace.active)

    def _pick_page_of(self, v: Optional[ViewerPanel]) -> str:
        """The page of the node ``v``'s pick writes to (the active page when unnamed)."""
        pid = v.pick_page_id() if v is not None else None
        return pid or (self.workspace.active or "")

    def _on_pick_armed(self, on: bool) -> None:
        if on:
            self.statusBar().showMessage("Picking — Esc cancels, Enter applies")
        else:
            self.statusBar().clearMessage()
            self.inspector.set_draw_state(None, False)
            if self._pick_return is not None:
                # after the commit that may follow this signal (Enter → disarm → commit),
                # hence deferred: the node we go back to must see the new shapes
                QTimer.singleShot(0, self._return_from_draw)

    def _on_pick_readout(self, text: str, viewer: Optional[ViewerPanel] = None) -> None:
        v = viewer if viewer is not None else self._active_viewer()
        nid = v.pick_node_id() if v is not None else None
        # the inspector shows a node of the ACTIVE page: another page's same id is not it
        if nid is not None and self._pick_page_of(v) == (self.workspace.active or ""):
            self.inspector.set_draw_state(nid, True, text)

    # ── drawing in the node's panel (2026-10-02) ──────────────────────────────
    def _draw_shapes_socket(self, node_id: str):
        rec = self.doc.nodes.get(node_id)
        spec = rec.spec() if rec is not None else None
        if spec is None:
            return None
        for s in self.doc.input_specs(node_id):
            if getattr(s, "pick_kind", "") == "shapes":
                alt = RD.EMPTY_PICKS.get((spec.op_key, s.name))
                if not (alt and alt[0]):
                    return s
        return None

    def _pick_viewer(self, node_id: str,
                     page_id: Optional[str] = None) -> Optional[ViewerPanel]:
        """The viewer whose ARMED pick writes to ``node_id`` of ``page_id`` (default: the
        active page, whose node the inspector shows), or ``None``."""
        pid = page_id or self.workspace.active or ""
        return next((v for v in self.viewers
                     if v.pick_node_id() == node_id and self._pick_page_of(v) == pid), None)

    def _arm_draw(self, node_id: str) -> None:
        """Arm the drawing for ``node_id`` (a node that IS a drawing). The active viewer
        must show THAT node's image so the shapes land in its frame: if it already does,
        arm at once; otherwise pull it and arm when the result lands
        (:meth:`_on_run_finished`)."""
        if self._draw_shapes_socket(node_id) is None:
            return
        v = self._active_viewer()
        if v is not None and v.has_image() and v.showing()[0] == self._full(node_id):
            self._arm_draw_now(node_id, v)
            return
        self._draw_arm_pending = self._full(node_id)     # its page's: ids repeat across pages
        self.pull_node(node_id)

    def _arm_draw_now(self, node_id: str, viewer: Optional[ViewerPanel] = None) -> None:
        v = viewer if viewer is not None else self._active_viewer()
        s = self._draw_shapes_socket(node_id)
        if s is None or v is None or not v.has_image():
            return
        from dataclasses import replace as _dc_replace
        req = _dc_replace(request_for(node_id, s), tools_in_panel=True)
        md = {}
        try:
            md = dict(self.doc.env(node_id).metadata or {})
        except Exception:                     # noqa: BLE001 — an un-propagated node
            md = {}
        self._open_viewer(v)
        v.arm_pick(req, Calibration.from_metadata(md), page_id=self.workspace.active)
        self._sync_draw_session(node_id)
        self.inspector.set_draw_state(node_id, True, v._pick.readout()
                                      if v._pick is not None else "")

    def _sync_draw_session(self, node_id: str) -> None:
        """Push the node's tool / operation / brush settings into the armed gesture."""
        pv = self._pick_viewer(node_id)
        if pv is None:
            return
        rec = self.doc.nodes.get(node_id)
        spec = rec.spec() if rec is not None else None
        if spec is None:
            return

        def val(name, default):
            v = rec.params.get(name)
            if v in (None, ""):
                s = spec.input(name)
                v = s.default if s is not None else default
            return v
        pv.set_pick_tool(str(val("tool", "rect")))
        pv.set_pick_op(str(val("op", "add")))
        try:
            pv.set_pick_brush(float(val("brush_px", 8.0)))
        except (TypeError, ValueError):
            pass

    def _on_draw_control(self, node_id: str, what: str, _value) -> None:
        if what == "arm":
            self._arm_draw(node_id)
            return
        if what == "sync":
            self._sync_draw_session(node_id)
            return
        pv = self._pick_viewer(node_id)
        if pv is None:
            return
        if what == "apply":
            pv.apply_pick()
        elif what == "cancel":
            pv.cancel_pick()
        elif what in ("undo", "clear", "close", "invert"):
            pv.pick_action(what)

    def _on_region_requested(self, node_id: str, socket: str) -> None:
        """A node that WANTS a region (Subtract Background's `Background sample`): drop a
        Draw Regions node in line on its region input — or use the one already wired
        there — switch to it, arm its drawing, and remember where to come back to."""
        rec = self.doc.nodes.get(node_id)
        spec = rec.spec() if rec is not None else None
        alt = RD.EMPTY_PICKS.get((spec.op_key, socket)) if spec is not None else None
        if not (alt and alt[0]):
            return
        alt_in, alt_op, _need = alt
        feeders = [e for e in self.doc.edges if e[2] == node_id and e[3] == alt_in]
        target = None
        for e in feeders:
            src = self.doc.nodes.get(e[0])
            if src is not None and src.op_key == alt_op:
                target = e[0]
                break
        if target is None:
            target = self._on_add_requested(node_id, alt_op, alt_in)
        if target is None:
            return
        self._pick_return = (self.workspace.active or "", node_id)
        self._select_only(target)
        self._arm_draw(target)
        self.statusBar().showMessage(
            f"drawing regions on {target} — Apply there brings you back to {node_id}")

    def _return_from_draw(self) -> None:
        back, self._pick_return = self._pick_return, None
        if back is None:
            return
        pid, nid = back                      # the node's own page, whichever is active now
        doc = self._page_doc(pid)
        if doc is None or nid not in doc.nodes:
            return
        if pid == self.workspace.active:
            self._select_only(nid)
        self.pull_node(nid, page_id=pid)

    def _select_only(self, node_id: str) -> None:
        item = self.scene.node_items.get(node_id)
        if item is None:
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self.inspector.set_node(item)        # even when the selection signal is debounced

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

    def _on_overlay_step(self, ovl_id: str, dt: int, dz: int,
                         viewer: Optional[ViewerPanel] = None) -> None:
        """A source's ◀▶ stepper: move that source's DISPLAYED frame by ``(dt, dz)`` from
        its mapped one. A runner setting, not an edit — nothing re-runs, nothing is saved —
        so the user can hunt for the frame that goes with this one before pinning it."""
        self.runner.set_source_override(ovl_id, int(dt), int(dz))
        self._on_view_request(viewer)

    def _on_overlay_pin(self, ovl_id: str, axis: str, pri: int, sec: int,
                        viewer: Optional[ViewerPanel] = None) -> None:
        """Pin T / Pin Z: "the primary's current frame goes with THIS frame of the source".

        Written into the Overlay's ``t_pins`` / ``z_pins`` through the same lines a typed edit
        runs (the value plus its sticky pin), so it serializes, diffs and keys the memo like
        any param. The row records each file's clock (T) or absolute focus (Z) beside the
        indices, which is what lets it survive an upstream crop re-numbering the frames. A
        pin at a primary frame that already had one replaces it; the source's stepped offset
        is dropped, since the pin now makes the mapping land where the stepper was."""
        from nodegraph.placement import parse_pins, pins_json
        # the strip hands back the RUN id it was given (V4.00 step 2): the pin is written
        # on the page's own card, under its document id
        run_ovl = self._full(ovl_id)
        opid, ovl_id = split_run_id(run_ovl)
        odoc = self._page_doc(opid)
        rec = odoc.nodes.get(ovl_id) if odoc is not None else None
        v = viewer if viewer is not None else self._active_viewer()
        viewed = self._binding_full(v)
        if rec is None or axis not in ("t", "z") or viewed is None:
            return
        name = f"{axis}_pins"
        try:
            rows = list(parse_pins(rec.params.get(name, ""), axis=axis))
        except ValueError as exc:
            self.statusBar().showMessage(f"{ovl_id}: cannot add a pin — {exc}", 6000)
            return
        m = v.coords()[0]
        a_pri, a_sec = self.runner.overlay_pin_anchors(viewed, run_ovl, axis, m,
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
        self.runner.set_source_override(run_ovl, 0, 0)
        odoc.touch(ovl_id)
        osc = self._scenes.get(opid)
        item = osc.node_items.get(ovl_id) if osc is not None else None
        if item is not None:
            item.refresh()
            item.changed.emit(item)
        self.statusBar().showMessage(
            f"{ovl_id}: pinned primary {axis}={pri} to source {axis}={sec}"
            + ("" if a_pri is not None else " (by index — no clock/focus to anchor it)"), 5000)

    @_needs_topology
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

    @_needs_topology
    def _on_append_requested(self, node_id: str, op_key: str, from_socket: str) -> Optional[str]:
        """A *Ready to run* suggestion that adds a node DOWNSTREAM (V4.00 step 11: "+ Page
        Output" after a terminal node, or after a loader nothing publishes yet): the new node
        lands right of ``node_id`` and ``node_id.from_socket`` is wired into its first
        Dataset input. Returns the new node's id."""
        rec = self.doc.nodes.get(node_id)
        if rec is None:
            return None
        from nodegraph.sockets import SocketType as _ST
        new = self.doc.add_node(op_key, x=float(rec.x) + SOURCE_OUTPUT_GAP, y=float(rec.y))
        nspec = new.spec()
        new_in = next((s.name for s in getattr(nspec, "inputs", ())
                       if s.type is _ST.DATASET), "data") if nspec is not None else "data"
        try:
            self.doc.connect(node_id, from_socket, new.id, new_in)
            msg = f"added {nspec.label if nspec else op_key} after {node_id}"
        except ValueError as exc:
            msg = f"added {op_key} but could not wire it: {exc}"
        self.doc.touch()
        if hasattr(self.scene, "sync"):
            self.scene.sync()
        item = self.scene.node_items.get(node_id)
        if item is not None:
            item.refresh()
            item.changed.emit(item)        # the inspector rebuilds: the hint is gone
        self.statusBar().showMessage(msg)
        return new.id

    def _on_pick_committed(self, node_id: str, values: dict,
                           viewer: Optional[ViewerPanel] = None) -> None:
        """Write a finished pick into the document of the page it was ARMED on
        (:meth:`ViewerPanel.pick_page_id`) — not the active one: the user may have pressed
        another canvas before applying it, and a duplicated page holds the same node ids.

        Deliberately the same two lines the inspector's ``_set_param`` runs — the value plus
        the sticky pin — so a picked param is in every later respect a hand-entered one: it
        shows as pinned, it can be unpinned back to its metadata-derived default, it
        serializes identically and it keys the memo identically. A pick is a nicer way to
        arrive at a number, not a different kind of number."""
        pid = self._pick_page_of(viewer)
        doc = self._page_doc(pid)
        rec = doc.nodes.get(node_id) if doc is not None else None
        if rec is None or not values:
            return
        for name, value in values.items():
            rec.params[name] = value
        rec.set_locked(rec.locked | set(values))
        doc.touch()
        sc = self._scenes.get(pid)
        item = sc.node_items.get(node_id) if sc is not None else None
        if item is not None:
            # the inspector rebuilds off the item's `changed`, so a pick made from the
            # CANVAS still refreshes the panel (and vice versa)
            item.refresh()
            item.changed.emit(item)
        what = ", ".join(f"{k} = {v}" for k, v in values.items())
        where = (node_id if pid == self.workspace.active
                 else f"{self.page_title(pid)[0]} › {node_id}")
        self.statusBar().showMessage(f"{where}: {what}")
        if any(getattr(rec.spec().input(k), "pick_kind", "") == "shapes"
               for k in values if rec.spec() is not None and rec.spec().input(k) is not None):
            # A drawing was applied: run the node so the regions it now defines are on
            # screen (2026-10-02). Without this the shapes landed in the param and nothing
            # visible changed, which read as "the drawing node does not work".
            self.pull_node(node_id, page_id=pid)

    @_needs_topology
    def _add_at_center(self, op_key: str) -> None:
        c = self.view.mapToScene(self.view.viewport().rect().center())
        self.doc.add_node(op_key, x=c.x() - 100, y=c.y() - 40)

    @_needs_shape
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

    @_needs_shape
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

    @_needs_shape
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

    @_needs_shape
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

    @_needs_topology
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
        # V4.00 step 11: a plain drop lands on the Image Input page and is published there;
        # a drop ON a Batch point stays where the point is — the batch is the dataset
        before = self.workspace.active
        publish = ((False if self._topology_ok() else None) if target
                   else self._begin_source_load())
        if publish is None:
            return
        x, y = pos.x(), pos.y()
        if self.workspace.active != before:
            # the drop position belongs to the canvas dropped on; on the Input page the
            # cards land at the view's centre, as File → Load's do
            c = self.view.mapToScene(self.view.viewport().rect().center())
            x, y = c.x() - 107, c.y() - 40
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
        outs = ([self._publish_source(rec, str(rec.params.get("path") or ""))
                 for rec in made] if publish else [])
        if failed:
            import os
            lines = "\n".join(f"{os.path.basename(p)} — {exc}" for p, exc in failed)
            QMessageBox.warning(
                self, "Some files could not be loaded",
                f"{len(made)} of {len(paths)} loaded.\n\n{lines}")
        if made:
            self.statusBar().showMessage(
                f"loaded {len(made)} file(s)"
                + (" into the batch point" if target else "")
                + self._published_note(outs), 6000)
            if outs:
                self.pull_node(made[0].id)      # the Viewer shows what was loaded

    # ── run (G7) ─────────────────────────────────────────────────────────────
    def pull_selected(self) -> None:
        sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if sel:
            self.pull_node(sel[0].node_id)

    # ── per-source ingest (V2.21) ─────────────────────────────────────────────
    def _sync_ingest(self) -> None:
        """Tell the canvas which source cards are mid-ingest, so a pull's card reset
        leaves their rails alone (:meth:`nodelab_v2.scene.GraphScene.set_ingesting`)."""
        ing = list(self.runner.ingesting())
        for p, sc in list(self._scenes.items()):
            sc.set_ingesting(local_ids(ing, p, self.workspace.active))

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
        pid, nid = split_run_id(self._full(node_id))
        sc = self._scenes.get(pid)
        if sc is not None:
            sc.on_node_progress("start", nid, {})

    def _on_ingest_finished(self, node_id: str, seconds: float, err) -> None:
        self._sync_ingest()
        pid, pnid = split_run_id(self._full(node_id))
        nid = self._local(node_id)
        if nid is None:                      # a source on a page the canvas is not showing
            sc = self._scenes.get(pid)
            if sc is not None:
                sc.on_node_progress("error" if err else "done", pnid,
                                    {} if err else {"seconds": seconds})
            self._idle_led()
            self.statusBar().showMessage(
                f"{node_id} " + ("FAILED to ingest" if err else f"ingested in {seconds:.1f}s"))
            return
        node_id = nid
        name = self._source_label(node_id)
        if err:
            self.scene.on_node_progress("error", node_id, {})
            for p in self._viewers_for(node_id):   # where that file is being shown
                p.show_error(self._full(node_id), str(err))
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
                  queue: bool = True, viewer: Optional[ViewerPanel] = None,
                  page_id: Optional[str] = None) -> None:
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
        writing a 40 GB store. Asking for it is a double-click, or Run → Ingest.

        ``viewer`` is the viewer to show it in — by default the active one (a new one
        when every viewer was closed). The viewer is BOUND to the node: its results land
        there from now on (V4.00 step 4). ``page_id`` names the node's page (default: the
        active one)."""
        pid = page_id or self.workspace.active or ""
        rid = qualify(pid, node_id)
        state = self.runner.source_state(rid)
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
        v = viewer if viewer is not None else self._ensure_viewer()
        self._preview_prev = None              # see `_preview_pull`
        self._bind(v, node_id, page_id=pid)    # its results land in `v` from now on
        # BEFORE reading the cursor: a new node may sit on a different source chain, so the
        # frame chooser's extent has to be right before the coords it produces are sent.
        self._sync_solo()
        # the troubleshooting scope is the ACTIVE viewer's — a Compare viewer shares it
        act = self._active_viewer() or v
        self.runner.set_frame_selection(*act.frame_selection())
        self.runner.pull(rid, v.coords(), v.channels(), queue=queue)

    # ── which viewer shows what (V4.00 step 4) ────────────────────────────────
    def _viewers_for(self, run_id: str) -> List[ViewerPanel]:
        """Which viewer(s) a delivery for ``run_id`` lands in — any page: every viewer
        BOUND to that node; failing that, the one that last asked for it (re-bound since —
        the first result to land is viewable while the next one computes); failing that,
        the active viewer, which shows whatever finishes (V2's primary pane)."""
        full = self._full(run_id)
        key = split_run_id(full)
        vs = self.viewers
        bound = [v for v in vs if v.binding == key]
        if bound:
            return bound
        asker = self._asker.get(full)
        if asker is not None:
            return [asker] if asker in vs else []
        act = self._active_viewer()
        return [act] if act is not None else []

    @staticmethod
    def _route_planes(panes, planes, request) -> List[Tuple[ViewerPanel, Any]]:
        """Pair each pane with the planes it may draw. A lone pane takes what came (the
        frame that landed is the frame shown, as ever). When SEVERAL viewers show one node,
        the planes go only to those whose cursor and channel set asked for exactly them —
        a viewer at t=40 must not flash the t=0 frame another viewer of the node asked for
        — and the others get ``None`` (they ask for their own)."""
        if len(panes) <= 1 or not planes or request is None:
            return [(p, planes) for p in panes]
        return [(p, planes if request_answers(request, p.coords(), p.channels()) else None)
                for p in panes]

    # ── Compare: a viewer beside the active one (V2.28 → V4.00 step 4) ────────
    def compare_selected(self) -> None:
        """Run → *Compare selected beside viewed* (F8)."""
        sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            self.statusBar().showMessage(
                "select a node first — F8 opens it beside the viewed one")
            return
        self.open_compare(sel[0].node_id)

    def open_compare(self, node_id: str) -> None:
        """Show ``node_id``'s result in a Viewer BESIDE the active one — the active viewer's
        Compare viewer, re-targeted when it already has one, a new dock beside it otherwise.

        The two cursors LINK automatically when both results span the same M/T/Z — the
        Compare viewer then drops its own strips and the active viewer's move both — and
        stay independent otherwise (:meth:`_sync_link`). The active viewer stays the active
        one: picks, the troubleshooting scope and the spreadsheet stay with it."""
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
            self.set_maximized(False)   # the viewers come back out of the mini-map first
        leader = self._ensure_viewer()
        leader = self._links.get(leader, leader)   # from a Compare viewer: its leader's
        prev = getattr(self, "_preview_prev", None)
        if (prev is not None and prev[0] is leader and prev[1] == node_id
                and self._bound_local(leader) == node_id and prev[2] in self.doc.nodes):
            # selecting the card PREVIEWED it into this viewer a moment ago: compare it
            # beside what was being viewed before the click, not beside itself
            self.pull_node(prev[2], viewer=leader, allow_ingest=False)
        self._open_viewer(leader)
        follower = next(iter(self._followers_of(leader)), None)
        if follower is None:
            fdock = self.shell.spawn(VIEWER_KIND, beside=self._viewer_dock(leader))
            follower = fdock.panel
            self._links[follower] = leader
            self._activate_viewer(leader)
        else:
            self._open_viewer(follower)
        self._linked.discard(follower)
        follower.set_axes_hidden(False)
        self.pull_node(node_id, viewer=follower)

    def close_compare(self) -> None:
        """Close the newest Compare viewer (Shift+F8); the viewer it was opened beside
        carries on alone."""
        if not self._links:
            return
        f = list(self._links)[-1]
        d = self._viewer_dock(f)
        if d is not None:
            d.close()                 # the shell destroys it; `_prune_viewers` forgets it
        else:
            self._links.pop(f, None)
            self._linked.discard(f)
        self.statusBar().showMessage("compare pane closed")

    def _followers_of(self, v: Optional[ViewerPanel]) -> List[ViewerPanel]:
        """The Compare viewers opened beside ``v``."""
        return [f for f, lead in self._links.items() if lead is v]

    def _sync_link(self, f: ViewerPanel) -> None:
        """Link or unlink a Compare viewer's cursor to its leader's from their METADATA:
        linked iff both hold a result and the M/T/Z extents agree, per the payloads' own
        axes. Linked, the Compare viewer's cursor row disappears (one set of sliders,
        moving both); unlinked, it keeps its own. Re-derived after every delivery, because
        a re-pull can change either side's extents."""
        lead = self._links.get(f)
        if lead is None:
            return
        a, b = lead.axes(), f.axes()
        linked = (a is not None and b is not None
                  and (a.m, a.t, a.z) == (b.m, b.t, b.z))
        if linked:
            self._linked.add(f)
        else:
            self._linked.discard(f)
        f.set_axes_hidden(linked and f is not self._mini_viewer)
        self._sync_viewer_title(f)
        self._sync_viewer_controls()          # a linked Compare viewer: the leader's strips
        fn = self._binding_full(f)
        if linked and fn is not None:
            m, t, z, _c = lead.coords()
            if (m, t, z) != tuple(f.coords()[:3]):
                f.set_cursor(m, t, z)
                self.runner.request_plane(fn, f.coords(), f.channels())

    def _sync_links_of(self, v: ViewerPanel) -> None:
        """Re-derive every cursor link ``v`` takes part in, as leader or as follower."""
        if v in self._links:
            self._sync_link(v)
        for f in self._followers_of(v):
            self._sync_link(f)

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

        The panel layout is saved first (V4.00 step 3) — failing to remember it must not
        block quitting either.
        """
        if getattr(self, "_persist_layout", False):
            try:
                if self._maximized:
                    # the docked layout is the one to remember: maximized, every viewer dock
                    # is hidden and the mini-map's own is empty
                    self.set_maximized(False)
                self.shell.save_layout()
            except Exception:                                # noqa: BLE001 — see above
                pass
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

    @_needs_topology
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
            self.doc.set_held_nodes(self._local_ids(self.runner.held))
            self.runner.invalidate()
            self.statusBar().showMessage(
                f"{node_id} released — the chain above runs live again")
            self._repull_viewed()
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

    def _sync_iteration_strip(self, node_id: Optional[str],
                              viewer: Optional[ViewerPanel] = None, *, doc=None) -> None:
        """Show ``viewer``'s (default: the active one's) iteration strip for an Iterate
        node, captioned with the value each iteration used. Silent about a misconfigured
        sweep — the inspector panel is where that is reported, and two copies of the same
        complaint is one too many."""
        v = viewer if viewer is not None else self._active_viewer()
        if v is None:
            return
        doc = doc if doc is not None else self.doc
        owner = doc.iterate_card_at(node_id or "")
        if owner is None:
            v.set_iterations(())
            return
        try:
            plan = iterate_plan(doc.to_graph(), owner, envs=doc.envs)
        except Exception:                     # noqa: BLE001 — half-wired: no strip, no noise
            v.set_iterations(())
            return
        labels = []
        for it in plan.iterations:
            parts = [("—" if v is None else (f"{v:g}" if isinstance(v, (int, float))
                                             else str(v))) for v in it.values]
            labels.append(" · ".join(parts))
        current = int(doc.nodes[owner].params.get("index", 0) or 0)
        v.set_iterations(labels, current)

    def _on_iteration_changed(self, index: int,
                              viewer: Optional[ViewerPanel] = None) -> None:
        """The iteration strip moved: keep that iteration and re-pull.

        Writing ``preserve=picked`` alongside the index is deliberate — scrubbing to an
        iteration under 'best' would otherwise change nothing visible, since the metric
        still decides. Picking one by eye IS the statement that you want that one."""
        v = viewer if viewer is not None else self._active_viewer()
        b = getattr(v, "binding", None)
        pid, viewed = (b[0], b[1]) if b else (self.workspace.active, None)
        doc = self._page_doc(pid)
        owner = doc.iterate_card_at(viewed or "") if doc is not None else None
        if owner is None:
            return
        rec = doc.nodes[owner]
        if int(rec.params.get("index", 0) or 0) == index \
                and rec.modes.get("preserve") == "picked":
            return
        rec.params["index"] = int(index)
        rec.modes["preserve"] = "picked"
        doc.touch()
        # Re-pull what is being VIEWED, not the card: the strip normally appears while the
        # user is looking at the segment's end node, and re-pulling the card there would
        # move the viewer off the node they are tuning to answer a question they asked
        # about it.
        self.pull_node(viewed or owner, viewer=v, page_id=pid)

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
        pid = self.workspace.active or ""
        if pid != self._movie_pid():
            self.movie_editor.bind(None)     # another page's movie: a fresh binding
        self._movie_page = pid
        self.movie_editor.bind(node_id)

    def _movie_pid(self) -> str:
        """The page of the Export Movie the Movie Editor is bound to: the page it was opened
        on, which stays its page while another page is active (V4.00 step 5). Unbound, the
        active page."""
        pid = getattr(self, "_movie_page", None)
        return pid if pid in self.workspace.pages else (self.workspace.active or "")

    def _on_movie_fetched(self, node_id: str, payload, _seconds: float) -> None:
        run_id = self._full(node_id)
        pid, nid = split_run_id(run_id)
        for p, sc in list(self._scenes.items()):
            if p == pid:
                sc.finish_run(nid)
            else:
                sc.clear_run_plan(self._key_on(run_id, p))
                sc.finish_run(None)
        self._set_led("idle")
        if pid != self._movie_pid():        # a page the editor is not editing
            return
        self.movie_editor.on_fetched(nid, payload)
        self.statusBar().showMessage(f"{nid} ready for the Movie Editor")

    def _on_viewer_display(self, node_id: str) -> None:
        """The Viewer's look of ``node_id`` (a run id, or a bare id on the active page)
        settled: stamp it into every Export Movie whose linked panels read that node, then
        let the editor re-render. The movies of the node's own page: a node id names a node
        of ONE page."""
        pid, nid = split_run_id(self._full(node_id))
        for mid in self._movie_nodes(pid):
            if self._movie_links_to(mid, nid, pid):
                self.stamp_movie_links(mid, page_id=pid)
        if pid == self._movie_pid():
            self.movie_editor.on_display_changed()

    # ── Export Movie ↔ Viewer LUT link ────────────────────────────────────────────
    def _movie_doc(self, page_id: Optional[str] = None):
        """The document of ``page_id`` — the active page when ``None``. Every Export Movie
        helper below takes the movie's page: the editor passes its own (``_movie_pid``)."""
        return self._page_doc(page_id or self.workspace.active or "")

    def _movie_nodes(self, page_id: Optional[str] = None) -> List[str]:
        doc = self._movie_doc(page_id)
        if doc is None:
            return []
        return [n for n, r in doc.nodes.items() if r.op_key == MOVIE_OP]

    def _movie_sources(self, movie_id: str,
                       page_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """``{letter: {"node", "label", "env"}}`` for an Export Movie's three inputs, each
        the REAL node feeding it (through reroutes and muted nodes)."""
        from nodegraph.catalog._shared.movie_timeline import SOURCE_SOCKETS
        doc = self._movie_doc(page_id)
        out: Dict[str, Dict[str, Any]] = {}
        if doc is None:
            return out
        for letter, socket in SOURCE_SOCKETS.items():
            src = doc.real_source(movie_id, socket)
            if src is None:
                out[letter] = {"node": None}
                continue
            rec = doc.nodes.get(src)
            spec = rec.spec() if rec is not None else None
            label = f"{spec.label if spec is not None else rec.op_key} ({src})"
            try:
                env = doc.env(src)
            except Exception:              # noqa: BLE001 — an un-propagated node
                env = None
            out[letter] = {"node": src, "label": label, "env": env}
        return out

    def _movie_links_to(self, movie_id: str, node_id: str,
                        page_id: Optional[str] = None) -> bool:
        doc = self._movie_doc(page_id)
        rec = doc.nodes.get(movie_id) if doc is not None else None
        if rec is None or str(rec.modes.get("sweep", "time")) != "timeline":
            return False
        srcs = self._movie_sources(movie_id, page_id)
        return any((i or {}).get("node") == node_id for i in srcs.values()) \
            or node_id == movie_id

    def live_display(self, movie_id: str, spec: Dict[str, Any],
                     page_id: Optional[str] = None) -> Dict[str, Any]:
        """``spec`` with every Viewer-linked channel filled in from the Viewer, NOT written.

        Each linked channel takes its panel source's node's LUT; a channel the Viewer holds
        nothing for on that node falls back to the movie node itself (a tap: viewing it
        shows source A), and past that keeps whatever values it already had — the stamp from
        last time, or none, which renders as auto contrast."""
        import copy as _copy
        out = _copy.deepcopy(spec)
        srcs = self._movie_sources(movie_id, page_id)
        states: Dict[str, Dict[int, Dict[str, Any]]] = {}

        def state_of(node: Optional[str], letter: str) -> Dict[int, Dict[str, Any]]:
            key = f"{letter}:{node}"
            if key not in states:
                env = (srcs.get(letter) or {}).get("env")
                names = list((getattr(env, "metadata", {}) or {}).get("channel_names")
                             or [])
                got = self._display_state_of(node, names, page_id) if node else {}
                if letter == "A":
                    for c, d in self._display_state_of(movie_id, names, page_id).items():
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

    def _display_state_of(self, node_id: str, names,
                          page_id: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
        """The Viewer's look of ``node_id`` of ``page_id`` (default: the active page) —
        from the active viewer when it holds one, else from the first viewer that does
        (each viewer keeps its own LUTs)."""
        act = self._active_viewer()
        order = ([act] if act is not None else []) + [v for v in self.viewers
                                                      if v is not act]
        # viewers key a node by its run id
        full = self._full(node_id) if page_id is None else qualify(page_id, node_id)
        for v in order:
            got = v.display_state(full, names)
            if got:
                return dict(got)
        return {}

    def stamp_movie_links(self, movie_id: str, page_id: Optional[str] = None) -> bool:
        """Write the Viewer's current look into ``movie_id``'s linked channels. ``True`` if
        the node's timeline changed.

        The COMMIT half of the live link: the values land in the saved graph, so a headless
        or LabLink run of this movie wears the LUTs the user tuned here. Deferred while the
        movie node itself is being computed, because an edit inside a running export's cone
        would cancel it; it is retried a moment later."""
        from nodegraph.catalog._shared.movie_timeline import canonical_json, try_normalize
        pid = page_id or self.workspace.active or ""
        doc = self._movie_doc(pid)
        rec = doc.nodes.get(movie_id) if doc is not None else None
        if rec is None or str(rec.modes.get("sweep", "time")) != "timeline":
            return False
        spec, _err = try_normalize(rec.params.get("timeline", "") or "")
        if spec is None:
            return False
        text = canonical_json(self.live_display(movie_id, spec, pid))
        if text == rec.params.get("timeline"):
            return False
        if self.runner.in_flight(qualify(pid, movie_id)):
            QTimer.singleShot(1500, lambda m=movie_id, p=pid: self.stamp_movie_links(m, p))
            return False
        self.write_movie_timeline(movie_id, text, page_id=pid)
        return True

    def write_movie_timeline(self, movie_id: str, text: str,
                             page_id: Optional[str] = None) -> None:
        """The one write path for an Export Movie's ``timeline``: the value, the lock that
        tells a re-seed the user owns it, and the narrowed touch — the same three steps an
        inspector edit takes (``node_item._write_param``)."""
        doc = self._movie_doc(page_id)
        rec = doc.nodes.get(movie_id) if doc is not None else None
        if rec is None:
            return
        rec.params["timeline"] = text
        rec.set_locked(rec.locked | {"timeline"})
        doc.touch(movie_id)

    def set_movie_sweep(self, movie_id: str, value: str,
                        page_id: Optional[str] = None) -> None:
        doc = self._movie_doc(page_id)
        rec = doc.nodes.get(movie_id) if doc is not None else None
        if rec is None or rec.modes.get("sweep") == value:
            return
        item = self._node_item(movie_id, page_id)
        if item is not None:
            item._write_mode("sweep", value)     # THE mode-write path: re-gates the card
        else:
            rec.modes["sweep"] = value
            doc.touch(movie_id)

    def export_movie(self, movie_id: str, page_id: Optional[str] = None) -> None:
        """Stamp the linked LUTs, make sure there is a file to write, and pull the node."""
        pid = page_id or self.workspace.active or ""
        doc = self._movie_doc(pid)
        rec = doc.nodes.get(movie_id) if doc is not None else None
        if rec is None:
            return
        self.stamp_movie_links(movie_id, pid)
        if not str(rec.params.get("path", "") or "").strip():
            path, _f = QFileDialog.getSaveFileName(
                self, "Export movie to", "movie.mp4",
                "MP4 video (*.mp4);;Animated GIF (*.gif);;PNG sequence (*.png);;"
                "JPEG sequence (*.jpg)")
            if not path:
                return
            rec.params["path"] = path
            rec.set_locked(rec.locked | {"path"})
            doc.touch(movie_id)
        self.pull_node(movie_id, page_id=pid)

    def _node_item(self, node_id: str, page_id: Optional[str] = None):
        """``node_id``'s card on ``page_id``'s scene (default: the active page), if any."""
        sc = self._scenes.get(page_id or self.workspace.active or "")
        return sc.node_items.get(node_id) if sc is not None else None

    def _on_iterate_action(self, node_id: str, action: str) -> None:
        """Run sweep / stop sweeping, from the Iterate panel."""
        rec = self.doc.nodes.get(node_id)
        if rec is None or rec.op_key != ITERATE_OP:
            return
        current = set(self._local_ids(self.runner.sweep_all))
        others = [r for r in self.runner.sweep_all if self._local(r) is None]   # other pages
        if action == "sweep":
            current.add(node_id)
        elif action == "stop_sweep":
            current.discard(node_id)
        else:
            return
        self.runner.set_sweep_all(others + sorted(current))
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
        # The owner is page-qualified when the pull was composed (V4.00): the table belongs
        # on the Iterate card's OWN page, which an upstream sweep is not the shown one.
        pid, owner = split_run_id(str(md.get(SWEEP_OWNER_KEY) or node_id))
        page = self.workspace.pages.get(pid or self.workspace.active or "")
        rec = page.doc.nodes.get(owner) if page is not None else None
        if rec is None or rec.op_key != ITERATE_OP:
            return
        rec.params[SWEEP_KEY] = {"rows": [
            {"iter": r.get("iter"), "metric": r.get("metric"), "won": r.get("won")}
            for r in rows]}
        shown = getattr(self.inspector, "_node", None)
        if (shown is not None and getattr(shown, "node_id", None) == owner
                and page.doc is self.doc):
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
        act = self._active_viewer()           # a scoped bake runs the active scope
        if act is not None:
            self.runner.set_frame_selection(*act.frame_selection())
        bake_id = uuid.uuid4().hex
        started = self.runner.bake(
            node_id, store=store, precision=precision, bake_id=bake_id,
            signature=self.doc.dock_signature(node_id), scoped=scoped,
            coords=act.coords() if act is not None else None)
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
        run_id = node_id
        self._retire_claims(run_id)          # it ran as a pull: its claims end here
        pid, node_id = split_run_id(str(node_id))      # the run id names the page
        page = self.workspace.pages.get(pid or self.workspace.active or "")
        if page is None or node_id not in page.doc.nodes:
            self.statusBar().showMessage(
                f"{node_id}: its page or card went away while it held — nothing was "
                f"pinned", 8000)
            return
        self.runner.hold(run_id, spec["payload"], spec.get("env"))
        doc = page.doc
        doc.set_dock_hold(node_id, True)
        doc.set_held_nodes(local_ids(self.runner.held, pid or self.workspace.active))
        self.runner.invalidate()
        self.statusBar().showMessage(
            f"{node_id} held in memory — the chain above is frozen and greyed out. "
            f"Nothing was written, so this does NOT free memory and does NOT survive "
            f"reopening the file; Bake it if you need either.")
        self._repull_viewed()

    def _on_baked(self, node_id: str, spec: dict) -> None:
        """Record a finished bake, dock the node, and release what it made redundant."""
        if spec.get("hold"):
            self._on_held(node_id, spec)
            return
        self._retire_claims(node_id)         # finished or stopped, its claims end here
        # the run id names the page: record the bake on THAT page's document
        pid, node_id = split_run_id(str(node_id))
        page = self.workspace.pages.get(pid or self.workspace.active or "")
        if page is None or node_id not in page.doc.nodes:
            self.statusBar().showMessage(
                f"{node_id}: its page or card went away while it baked — the checkpoint "
                f"is on disk at {spec.get('store', '?')} but nothing records it", 8000)
            return
        doc = page.doc
        if spec.get("cancelled"):
            self.statusBar().showMessage(
                f"{node_id} bake stopped — nothing was recorded, so the dock still reads "
                f"as un-baked and the half-written folder is safe to re-bake over")
            return
        man = spec.get("manifest") or {}
        doc.set_dock_bake(
            node_id, store=spec["store"], bake_id=str(man.get("bake_id", "")),
            precision=str(man.get("precision", spec.get("precision", ""))),
            signature=str(spec.get("signature", "")),
            nbytes=int(spec.get("bytes", 0) or 0),
            when=getattr(self, "_baked_at", ""))
        self.runner.invalidate()
        freed = self._release_dormant(doc, pid)
        from nodelab_v2.inspector import _human_bytes
        self.statusBar().showMessage(
            f"{node_id} docked — {_human_bytes(spec.get('bytes', 0))} on disk; "
            f"{freed} cached result(s) released and {len(doc.dormant)} node(s) "
            f"greyed out"
            + ("  ·  SCOPED bake: a truncated series" if spec.get("scoped") else ""))
        self._repull_viewed()

    def _release_dormant(self, doc=None, page_id: Optional[str] = None) -> int:
        """Free the memory the dock exists to free — the memo payloads of every node a
        docked run no longer evaluates. This is the "unload it from the software" half
        of docking; greying the cards is only the half you can see. ``doc``/``page_id``
        name the page the dock is on (default: the one on the canvas)."""
        doc = doc if doc is not None else self.doc
        pid = page_id or self.workspace.active
        return self.runner.unload([qualify(pid, n) if pid else n for n in sorted(doc.dormant)])

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
        act = self._active_viewer()           # the scope is the ACTIVE viewer's
        self.runner.set_solo_frame(on)
        if act is not None:
            self.runner.set_frame_selection(*act.frame_selection())
        self.inspector.set_solo_frame(on)     # an Iterate panel warns when the scope is off
        self._sync_solo()                     # ...which also ranges the region box
        if act is not None:
            self.runner.set_region(act.region)
        if self._solo_act.isChecked() != on:      # keep a programmatic call in sync
            self._solo_act.blockSignals(True)
            self._solo_act.setChecked(on)
            self._solo_act.blockSignals(False)
        self.mode_switch.set_troubleshooting(on)  # the menu bar's Normal | Troubleshooting
        self._repull_viewed()                  # show the new scope now, not on the next click
        self.statusBar().showMessage(          # after the pull: `started` also writes here
            f"troubleshooting: pulls analyse {self._scope_phrase()} — ctrl+click the T or "
            f"Z strip to pick more, F9 to run the full series"
            if on else "troubleshooting off — pulls analyse the whole series")

    def _scope_phrase(self) -> str:
        """What the next scoped pull will run, in words — shared by the status line and
        the chip's tooltip so the two can never disagree."""
        ms, ts, zs, totals, reg = self._scope_of_active()
        where = f"t={compact_list(ts)}"
        if totals[0] > 1:
            where += f" (m={compact_list(ms)})"
        n = len(ms) * len(ts)
        phrase = f"frame {where} only" if n == 1 else f"{n} frames only: {where}"
        if zs:
            phrase += (f", cut to {len(zs)} of {totals[2]} z-planes "
                       f"(z={compact_list(zs)})")
        if reg is not None:
            y0, y1, x0, x1 = reg
            phrase += (f", inside the {x1 - x0}×{y1 - y0} px region at y={y0}, x={x0} "
                       f"(drag the amber box on the Viewer to move it)")
        return phrase

    def _scope_of_active(self):
        """``(ms, ts, zs, totals, region)`` of the troubleshooting scope as the ACTIVE
        viewer holds it — the frames under its cursor or picked on its strips, its source
        totals and its region box. With no viewer left: the first frame, whole volume."""
        v = self._active_viewer()
        if v is None:
            return (0,), (0,), (), (1, 1, 1), None
        ms, ts, zs = v.scoped_frames()
        return ms, ts, zs, (v.solo or (1, 1, 1)), v.region

    def clear_frame_picks(self) -> None:
        """Run → *Clear picked frames*: back to scoping the single frame the cursor is on,
        whole volume. Reachable from the menu because a selection made on a strip that is
        currently scrolled out of the mini-map is otherwise invisible."""
        v = self._active_viewer()
        if v is not None:
            v.clear_frame_selection()

    def clear_region(self) -> None:
        """Run → *Clear troubleshooting region*: the amber box back to the whole frame."""
        v = self._active_viewer()
        if v is not None:
            v.clear_region()

    # ── the Viewer node's display settings (2026-10-02) ───────────────────────
    def _viewer_layout(self, node_id: Optional[str], doc=None) -> str:
        """The viewed node's ``layout`` Mode if it is a ``view.viewer``, else ``merged``."""
        doc = doc if doc is not None else self.doc
        rec = doc.nodes.get(node_id) if node_id else None
        if rec is None or rec.op_key != "view.viewer":
            return "merged"
        return str((rec.modes or {}).get("layout") or "merged")

    def _viewer_scalebar(self, node_id: Optional[str], doc=None):
        """The viewed node's scale-bar settings if it is a ``view.viewer`` with the bar on,
        else ``None``. Presentation params: read from the document, never the payload."""
        doc = doc if doc is not None else self.doc
        rec = doc.nodes.get(node_id) if node_id else None
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

    def _viewer_timestamp(self, node_id: Optional[str], doc=None):
        """The viewed node's timestamp settings if it is a ``view.viewer`` with the
        timestamp on, else ``None`` (2026-10-02). Presentation params, read from the
        document like the scale bar's."""
        doc = doc if doc is not None else self.doc
        rec = doc.nodes.get(node_id) if node_id else None
        if rec is None or rec.op_key != "view.viewer":
            return None
        spec = rec.spec()
        params = rec.params or {}

        def val(name):
            if name in params:
                return params[name]
            s = spec.input(name) if spec is not None else None
            return s.default if s is not None else None

        if not bool(val("show_timestamp")):
            return None
        return {"mode": str(val("timestamp_mode") or "elapsed"),
                "corner": str(val("timestamp_corner") or "top_left"),
                "color": str(val("timestamp_color") or "white")}

    def _on_region_changed(self, viewer: Optional[ViewerPanel] = None) -> None:
        """A viewer's region box was dragged or cleared — a change to WHAT a scoped pull
        computes, laterally. Re-scope the runner and, under the scope, re-run that viewer's
        node; off the scope the window is remembered for when F9 is armed."""
        v = viewer if viewer is not None else self._active_viewer()
        if v is None:
            return
        if v is not self._active_viewer():
            self._activate_viewer(v)          # working its box is working in it
        self.runner.set_region(v.region)
        self._sync_solo_chip()
        if not self.runner.solo_frame:
            return
        self.statusBar().showMessage(f"troubleshooting: pulls analyse "
                                     f"{self._scope_phrase()}")
        b = v.binding
        if b:
            self.pull_node(b[1], viewer=v, page_id=b[0])

    def _on_frame_selection(self, viewer: Optional[ViewerPanel] = None) -> None:
        """A viewer's M/T/Z picks changed — that is a change to *what a pull computes*,
        so re-scope the runner and (under the scope) re-run that viewer's node. Off the
        scope it is bookkeeping only: the picks are remembered for whenever F9 is armed."""
        v = viewer if viewer is not None else self._active_viewer()
        if v is None:
            return
        if v is not self._active_viewer():
            self._activate_viewer(v)          # picking on its strips is working in it
        self.runner.set_frame_selection(*v.frame_selection())
        self._sync_solo_chip()
        if not self.runner.solo_frame:
            # Picks are inert until the scope is armed. Say so — a user who just
            # ctrl-clicked three boxes and saw nothing happen has no other way to find out.
            ms, ts, zs = v.frame_selection()
            if ms or ts or zs:
                frames = len(ms or (0,)) * len(ts or (0,))
                self.statusBar().showMessage(
                    f"{frames} frame(s)" + (f" × {len(zs)} z-plane(s)" if zs else "")
                    + " picked — press F9 to scope pulls to them")
            return
        self.statusBar().showMessage(f"troubleshooting: pulls analyse "
                                     f"{self._scope_phrase()}")
        b = v.binding
        if b:
            self.pull_node(b[1], viewer=v, page_id=b[0])

    def _sync_solo(self, _node_id: Optional[str] = None, *, push: bool = True) -> None:
        """Re-point every viewer's frame chooser at ITS node's source extent — a linked
        cursor is expressed in global frame indices, so each chooser spans its own source.
        Only the ACTIVE viewer gets the region box: the troubleshooting region is one
        window, drawn where the scope is being worked. Called on toggle, before every pull,
        after every document edit and when another viewer becomes the active one, because
        each can change how many frames and planes there are to choose from."""
        on = self.runner.solo_frame
        act = self._active_viewer()
        for v in self.viewers:
            b = v.binding
            doc = self._page_doc(b[0]) if b else None
            nid = b[1] if (b and doc is not None and b[1] in doc.nodes) else None
            v.set_solo(doc.source_scope_totals(nid) if (on and nid) else None)
            # the region box spans the SOURCE frame for the same reason the strips span the
            # source series; a remembered window is re-clamped, not dropped
            v.set_region_extent(doc.source_scope_extent(nid)
                                if (on and nid and v is act) else None)
        if push and on and act is not None and self.runner.region != act.region:
            self.runner.set_region(act.region)
        self._sync_solo_chip()

    def _scope_tag(self) -> str:
        """The scope in a few characters — ``t7`` / ``m2·3T[0,4,9]·2Z``. Shared by the
        status chip and the canvas badge so the two always read the same."""
        ms, ts, zs, totals, reg = self._scope_of_active()
        name = f"t{compact_list(ts)}" if len(ts) == 1 else f"{len(ts)}T[{compact_list(ts)}]"
        if totals[0] > 1:
            name = (f"m{ms[0]}" if len(ms) == 1 else f"{len(ms)}M") + "·" + name
        name += f"·{len(zs)}Z" if zs else ""
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
            for c in self.canvases():
                c.view.set_troubleshooting(False)
            return
        name = self._scope_tag()
        for c in self.canvases():
            c.view.set_troubleshooting(True, name)
        self._solo_chip.setText(f"SOLO {name}")
        self._solo_chip.setToolTip(f"{self._solo_act.toolTip()}\n\n"
                                   f"Now: pulls analyse {self._scope_phrase()}.")
        self._solo_chip.setStyleSheet(
            f"color:{T.DIM2D_INK.name()}; background:{T.DIM2D.name()}; font-size:10px;"
            f"font-weight:800; border-radius:4px; padding:1px 6px; margin:0 4px;")
        self._solo_chip.show()

    def _on_view_request(self, viewer: Optional[ViewerPanel] = None) -> None:
        """A viewer's cursor or channel set moved (default: the active viewer's)."""
        v = viewer if viewer is not None else self._active_viewer()
        if v is None:
            return
        nid = self._binding_full(v)
        if nid is not None:
            # coords/channel-only change → runner serves from the decoded-plane cache
            # (no graph snapshot / engine re-pull) when the graph is unchanged. Under the
            # solo-frame scope an M/T move onto a frame outside the scope IS a new frame to
            # compute, and the runner turns it back into a real pull — the chip has to
            # follow the cursor either way.
            self._sync_solo_chip()
            self.runner.request_plane(nid, v.coords(), v.channels(), v.sub())
        # LINKED compare: the leader's strips are the one cursor, so its move carries the
        # Compare viewer with it — mirror silently, then ask for that viewer's planes.
        m, t, z, _c = v.coords()
        for f in self._followers_of(v):
            fn = self._binding_full(f)
            if f in self._linked and fn is not None:
                f.set_cursor(m, t, z)
                self.runner.request_plane(fn, f.coords(), f.channels())

    def _on_playing(self, on: bool, axis: str,
                    viewer: Optional[ViewerPanel] = None) -> None:
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
        v = viewer if viewer is not None else self._active_viewer()
        if v is None:
            return
        viewed = self._binding_full(v)
        reads = viewed is not None and self.runner.frames_are_reads(viewed)
        if on:
            # decided for EVERY axis, not just T: a cold Z step through a whole-volume chain
            # is the same 130 s wait, and the first one pays it for the rest of the volume
            v.set_play_pacing(not reads)
        if axis != "t":
            return
        if not on:
            if self._preload_viewer in (None, v):      # never another viewer's preload
                self.runner.cancel_preload()
                self._set_progress(None)
            v.set_play_gate(False)
            return
        if viewed is None:
            return
        n = self.runner.preload_series(viewed, v.coords(), v.channels())
        if not n:
            return                       # already resident, or too costly to read ahead at
                                         # all — either way, play immediately
        # named explicitly: with several viewers open the most recently held view may be
        # ANOTHER viewer's node, and this sentence is about the series being played here
        fits = self.runner.series_fits(planes=len(v.channels()), node_id=viewed)
        self._preload_viewer = v
        self._preload_hold_t0 = time.monotonic()
        #: whether this hold is subject to the PLAY_PREPARE_MAX_S cap — only a series too
        #: big to ever be fully resident is; one that fits holds to completion (see the
        #: constant's note). ⏸ remains the way out either way (it cancels the preload,
        #: whose `preload_finished` drops the gate).
        self._preload_hold_capped = not fits
        v.set_play_gate(
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
            node = viewed
            QTimer.singleShot(int(PLAY_PREPARE_MAX_S * 1000),
                              lambda: self._drop_play_gate(node))

    def _on_preload_progress(self, node_id: str, done: int, total: int) -> None:
        v = self._preload_viewer or self._active_viewer()
        if v is None or self._full(node_id) != self._binding_full(v) or not total:
            return
        self._set_progress(done / float(total))
        self.statusBar().showMessage(
            f"preparing frames for playback — {done}/{total}")
        # The ETA half of the PLAY_PREPARE_MAX_S cap — CAPPED holds only (a series larger
        # than the budget; one that fits waits to completion, see the constant's note).
        # Judged on the preload's own measured rate, and two ticks before judging — one
        # frame's wall time divided by one is not a rate.
        if (getattr(self, "_preload_hold_capped", True)
                and v.play_gated() and done >= 2):
            elapsed = time.monotonic() - getattr(self, "_preload_hold_t0", 0.0)
            if elapsed * total / done > PLAY_PREPARE_MAX_S:
                self._drop_play_gate(node_id)

    def _drop_play_gate(self, node_id: str) -> None:
        """Release a held playback whose preparation is running long — the frames keep
        warming behind it (the preload is NOT cancelled), so playback smooths out lap by
        lap instead of holding a blank stare. No-op unless ``node_id`` is still what the
        playing viewer shows with its gate up: the watchdog that arms this at ▶ may fire
        long after the preload finished, the node changed, or playback stopped."""
        v = self._preload_viewer or self._active_viewer()
        if v is None or self._full(node_id) != self._binding_full(v) \
                or not v.play_gated():
            return
        v.set_play_gate(False)
        self.statusBar().showMessage(
            "playing while the rest of the series prepares — each frame shows as it "
            "lands", 4000)

    def _on_preload_finished(self, node_id: str, completed: bool) -> None:
        """Frames are in: drop the gate and let the timer run.

        Also on a CANCELLED preload — an edit, or the node changing under it. Leaving the gate
        up there would strand playback in a paused state whose button says it is playing, which
        is worse than playing a frame cold."""
        self._set_progress(None)
        v = self._preload_viewer or self._active_viewer()
        if completed and v is not None and self._full(node_id) == self._binding_full(v):
            # said whether or not the gate is still up: a hold released early by the cap
            # reaches this point playing cold, and this is the moment it turns warm
            self.statusBar().showMessage("playing from memory", 2500)
        for w in self.viewers:              # a gate with no preload behind it is stranded
            if w.play_gated():
                w.set_play_gate(False)

    def _on_run_finished(self, node_id, payload, plane, axes, seconds,
                         request=None) -> None:
        run_id = self._full(node_id)
        pid, node_id = split_run_id(run_id)
        on_active = pid == self.workspace.active
        doc = self._page_doc(pid)
        # every OTHER page's canvas retires this run's claim on its cards — the upstream
        # chain it computed there, for a page that reads through a Page Input
        for p, sc in list(self._scenes.items()):
            if p != pid:
                sc.clear_run_plan(self._key_on(run_id, p))
                sc.finish_run(None)
        # A source the Movie Editor is waiting on may have just been computed by an
        # ordinary pull; it takes the payload the same way it takes a fetched one. Only the
        # runner's UNPINNED result: a solo-scoped payload holds a few frames, and a movie
        # built from it would silently be a truncated one. (The editor edits a node of ITS
        # page, the one it was opened on.)
        if (pid == self._movie_pid() and payload is not None
                and getattr(self, "movie_editor", None) is not None):
            unpinned = self.runner.finished_result(run_id)
            if unpinned is not None:
                self.movie_editor.on_fetched(node_id, unpinned)
        panes = self._viewers_for(run_id)
        routed = self._route_planes(panes, plane, request)
        for pane, planes in routed:
            if planes:
                self._open_viewer(pane)   # there is something to see now — open its dock
        if plane and self._draw_arm_pending == run_id:
            # the Draw Regions node's own image is on screen now: arm its drawing on it —
            # while its page is the one being edited (the drawing is armed for that page)
            self._draw_arm_pending = None
            target = next((p for p, pl in routed if pl), None)
            if on_active:
                QTimer.singleShot(0, lambda n=node_id, v=target: self._arm_draw_now(n, v))
        for pane, planes in routed:
            # no planes for this viewer — a frame another viewer of the node asked for, or a
            # result re-served while its planes follow on the decode lane: a viewer already
            # showing the node KEEPS its frame rather than blanking to "no image"
            keep = not planes and pane.showing()[0] == run_id and pane.has_image()
            pane.show_result(run_id, planes, axes, seconds, dataset=payload,
                             overlay=self.runner.overlay_channels(run_id),
                             overlay_note=self.runner.overlay_note(run_id),
                             overlay_style=self.runner.overlay_style(run_id),
                             overlay_src=self.runner.overlay_sources(run_id),
                             keep_image=keep)
            # flicker is a property of TIME, not of the composite, so it is driven here
            # rather than folded into the style map the shader reads
            pane.set_overlay_flicker(self.runner.overlay_flicker_hz(run_id))
            self._sync_overlay_frames(pane, run_id)
            # the Viewer NODE's own display settings (2026-10-02): its layout Mode and the
            # presentation-only scale bar, read live from the node's page — "merged" and no
            # bar for every other node, so viewing a filter never inherits them
            pane.set_source_layout(self._viewer_layout(node_id, doc))
            pane.set_scalebar(self._viewer_scalebar(node_id, doc))
            pane.set_timestamp(self._viewer_timestamp(node_id, doc))
            self._sync_iteration_strip(node_id, pane, doc=doc)   # each viewer, its own node
            if not planes and (plane or keep):
                # …and asks for the frame at ITS cursor, off the decode lane — never by a
                # pull: under the solo scope that pull would take the other viewer's frame
                # back, and the other viewer's would take it again, forever
                self.runner.request_plane(run_id, pane.coords(), pane.channels(),
                                          pull=False)
        mini = self._mini_viewer
        if self._maximized and mini is not None and mini in panes:
            # a new axes shape rebuilds the channel/LUT controls, and fresh widgets are
            # visible — re-fold them so the mini-map keeps its compact strip
            mini.set_compact(True, force=True)
        act = self._active_viewer()
        if act is not None and act in panes:
            # the spreadsheet and the sweep table follow the ACTIVE viewer: a delivery to
            # another viewer must not yank them off the node being tuned
            self.sheet.show_dataset(node_id, payload)
            if on_active:
                self._record_sweep(node_id, payload)
        for pane in panes:
            self._sync_links_of(pane)
        self.minimap.set_state("live")
        self._idle_led()                      # …unless files are still ingesting
        self._set_progress(None)              # the run is over — no bar to show
        # settle the cards: nothing is queued now, and the pulled node wears the run's
        # wall time (its own compute may have been microseconds — the wait was the read)
        sc = self._scenes.get(pid)
        if sc is not None:
            sc.finish_run(node_id, seconds=seconds)
        computed = getattr(self, "_run_computed", 0)
        cached = getattr(self, "_run_cached", 0)
        detail = f" ({computed} computed, {cached} cached)" if (computed or cached) else ""
        self.statusBar().showMessage(
            f"{node_id if on_active else run_id} pulled in {seconds:.2f}s{detail}")

    # ── per-node progress (G7 + 2026-07-28) ──────────────────────────────────
    def _on_run_plan(self, target: str, node_ids) -> None:
        full = self._full(target)
        active = self.workspace.active
        self._run_target = self._local(target) or full
        planned = list(node_ids)
        self._run_plan = [r for r in planned
                          if self._page_doc(self._page_of(r)) is not None
                          and split_run_id(self._full(r))[1]
                          in self._page_doc(self._page_of(r)).nodes]
        self._run_computed = 0
        self._run_cached = 0
        # every page with a canvas marks its participating cards — a run of ANOTHER page
        # claims the cards it computes on this one (its upstream chain) under its run id, so
        # they read queued → running → done like any other participant; the claim is
        # retired when that run ends
        for p, sc in list(self._scenes.items()):
            doc = self._page_doc(p)
            ids = [n for n in local_ids(planned, p, active) if doc is not None and n in doc.nodes]
            sc.set_run_plan(self._key_on(full, p), ids)

    def _on_run_queued(self, node_id: str, depth: int) -> None:
        """A pull joined the queue behind the running one: claim its cards as ``queued`` and
        say so in the status bar. The branch is lined up, not lost."""
        full = self._full(node_id)
        active = self.workspace.active
        planned = list(self.runner.planned_nodes(full))
        for p, sc in list(self._scenes.items()):
            doc = self._page_doc(p)
            ids = [n for n in local_ids(planned, p, active) if doc is not None and n in doc.nodes]
            sc.set_queued(self._key_on(full, p), ids)
        running = self._run_target or "a node"
        self.statusBar().showMessage(
            f"{self._local(node_id) or full} queued behind {running} — {depth} waiting")

    def _retire_claims(self, run_id: str) -> None:
        """``run_id``'s run is over: every page's canvas retires its claim (the plan
        :meth:`_on_run_plan` fanned out to each page the run computes on) and drops the
        queued/running marks nothing else claims. A finished, failed or fetched run does this
        in its own handler; a cancel, a Hold and a Bake (stopped or not) end here."""
        full = self._full(run_id)
        for p, sc in list(self._scenes.items()):
            sc.clear_run_plan(self._key_on(full, p))
            sc.finish_run(None)

    def _on_run_cancelled(self, node_id: str) -> None:
        """A queued or running pull was dropped because an edit landed inside its cone.
        Retire its claim and clear the cards nothing else wants, so no card is left
        reporting work that will never finish."""
        self._retire_claims(node_id)
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
        full = self._full(node_id)
        pid, nid = split_run_id(full)
        sc = self._scenes.get(pid)
        if sc is not None:
            sc.on_node_progress(event, nid, info)
        node_id = self._local(node_id) or full    # the footer names the node either way
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

    def _on_plane_ready(self, node_id, planes, axes, seconds, request=None) -> None:
        # fast-path display update (scrub/play): no dataset re-delivery, no spreadsheet
        # refresh — only the showing viewer's frame changes.
        run_id = self._full(node_id)
        for pane, pl in self._route_planes(self._viewers_for(run_id), planes, request):
            if pl is None:
                continue                  # another viewer's frame of the same node
            pane.show_planes(run_id, pl, axes, seconds)
            self._sync_overlay_frames(pane, run_id)
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
        run_id = self._full(node_id)
        pid, nid = split_run_id(run_id)
        for p, sc in list(self._scenes.items()):
            if p == pid:
                sc.finish_run(nid, failed=True)
            else:
                sc.clear_run_plan(self._key_on(run_id, p))
                sc.finish_run(None)
        for pane in self._viewers_for(run_id):
            pane.show_error(run_id, trace)
        node_id = self._local(run_id)
        self._set_led("error")
        self._set_progress(None)              # a failed run must not leave a stale bar
        self.minimap.set_state("error")
        last = [ln for ln in trace.strip().splitlines() if ln.strip()]
        self.console.error(f"{node_id or run_id} FAILED\n{trace.rstrip()}")
        # Raise it on the FIRST failure of a session rather than every time: a user who has
        # deliberately closed it while working through a chain of errors should not have to
        # close it again after each one.
        if not self._console_shown:
            self._console_shown = True
            self._console_dock.show()
            self._console_dock.raise_()
        self.statusBar().showMessage(
            f"{node_id or run_id} FAILED — {last[-1] if last else 'see Console'}")

    # ── file (G6) ────────────────────────────────────────────────────────────
    def _forget_display_state(self) -> None:
        """A new or newly opened graph reuses node ids, so every viewer's per-node LUTs and
        switched-off channels from the last graph must not carry over to it."""
        for pane in self.viewers:
            pane.forget_display_state()

    def file_new(self) -> None:
        self._welcome_dismissed.clear()
        self._seeded_pages.clear()
        self._closed_pages.clear()
        self.__dict__.pop("_recipe_cache", None)
        self.workspace.reset()    # clears the page → _on_doc_changed closes Compare viewers
        self._forget_display_state()
        for v in self.viewers:
            self._bind(v, None)
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
        # A TIFF has no direct-read path (`nd2_direct.decide_access`), and `direct` is the
        # card's default: start a TIFF on `ingest` so its first pull — the preview a load
        # triggers since V4.00 step 11 — works instead of asking the user to flip a mode.
        from nodelab_v2.ingest import _is_tiff
        modes = {ACCESS_MODE: ACCESS_INGEST} if _is_tiff(path) else None
        rec = self.doc.add_node(
            LOAD_OP, x=x, y=y, modes=modes,
            params={"path": path, TITLE_KEY: os.path.basename(path),
                    CHANNELS_KEY: chans, **prefill})
        # `source_file` rides the seed for the same reason the bundle card's does: the
        # edit-time envelope and the pulled payload must agree about a positional list, and
        # `EngineRunner._resolve_source` stamps the identical value on the pull side. It is
        # what lets `util.timeseries` order separately loaded files by their names.
        # ...and the channel names/colours ride it (V4.00 step 11g): the SAME
        # `with_channel_display` the runner applies to its resolved envelope, so the two
        # agree and the first pull re-seeds nothing.
        from nodelab_v2.ingest import with_channel_display
        self.doc.set_meta_seed(rec.id, stamp_source_file(with_channel_display(
            MetaEnvelope(axes=axes, metadata=dict(calib)), disp), path))
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
        # a bundle with any TIFF in it starts on `ingest`, as a single TIFF card does: the
        # default `direct` reads ND2 only, so the bundle's first pull would fail
        from nodelab_v2.ingest import _is_tiff
        modes = {ACCESS_MODE: ACCESS_INGEST} if any(_is_tiff(p) for p in paths) else None
        rec = self.doc.add_node(
            LOAD_OP, x=x, y=y, modes=modes,
            params={"path": paths[0], BUNDLE_PATHS_KEY: list(paths),
                    TITLE_KEY: f"{len(paths)} files",
                    CHANNELS_KEY: chans})
        # The seed envelope must say the same thing the pull will (the edit-time envelope
        # and the payload agreeing is the standing rule for anything that changes axes):
        # M is the sum, and `source_file` names the file each position came from.
        from nodelab_v2.ingest import channel_display_seed
        md = dict(calib0)
        md[SOURCE_FILE_KEY] = [labels[i] for i, h in enumerate(heads)
                               for _ in range(int(h[1].m))]
        md.update(channel_display_seed(disp0, ax0.c))      # the names ride it (step 11g)
        self.doc.set_meta_seed(rec.id, MetaEnvelope(axes=axes, metadata=md))
        return rec, axes

    @_needs_source_page
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

    @_needs_source_page
    def file_load_sequence(self) -> None:
        """File → Load file sequence…: pick ONE file of a numbered series, get all of it.

        The gap this fills. :meth:`file_load_source` can already multi-select a folder's
        worth of files into one bundle card — but that means ctrl-clicking 120 entries in a
        file dialog, and the bundle stacks them on POSITIONS, because that is the only axis
        a loader can grow without being told what the files mean. For a timelapse exported
        one frame per file that is the wrong axis, and wrongly in a way nothing errors on:
        the result is 120 fields of a 1-frame series, so ``util.stack`` fuses nothing and
        ``track.link`` has no frames to link (see :mod:`nodegraph.catalog.util.timeseries`).

        So this action does the three things that turn one click into that series: it
        derives the sequence from the picked file's name and scans its folder
        (:func:`nodegraph.file_sequence.scan`), it shows what it found and lets the pattern
        be corrected before anything is built
        (:class:`~nodelab_v2.sequence_dialog.SequenceScanDialog`), and it drops the bundle
        card with a ``util.timeseries`` already wired to it and preset to the chosen axis.

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
                                   suggested_name="", workspace=self.workspace,
                                   page_id=self.workspace.active or "")
        if written:
            self.statusBar().showMessage(f"recipe written to {written}")

    @_needs_source_page
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

    @_needs_source_page
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

        ``chain_axis`` (:meth:`file_load_sequence`) additionally wires a ``util.timeseries`` card
        onto the bundle, preset to that axis. It rides HERE rather than in the caller so the
        sequence loader inherits this method's grid-mismatch handling unchanged — and it is
        deliberately dropped by the "separate cards" fallback, since there is no bundle left
        for a chain to re-address."""
        import os

        publish = self._begin_source_load()     # V4.00 step 11: onto the Image Input page
        if publish is None:
            return                              # a linked Input page refused; hint shown
        c = self.view.mapToScene(self.view.viewport().rect().center())
        x, y = c.x() - 107, c.y() - 40
        added: list = []
        failed: list = []
        outs: list = []                         # the Page Outputs publishing what loaded

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
                chain = self.doc.add_node("util.timeseries", x=x + SEQUENCE_CHAIN_GAP, y=y,
                                          modes={"chain_axis": chain_axis})
                self.doc.connect(rec.id, "image", chain.id, "data")
            if publish:
                src, sock = (chain, "out") if chain is not None else (rec, "image")
                outs.append(self._publish_source(src, paths[0], from_socket=sock))
            ids = [rec.id] + ([chain.id] if chain is not None else []) + [o.id for o in outs]
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
            if outs:
                self.statusBar().showMessage(
                    self.statusBar().currentMessage() + self._published_note(outs))
                self.pull_node(rec.id)          # the Viewer shows what was loaded
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
                if publish:
                    outs.append(self._publish_source(rec, path))
                item = self.scene.node_items.get(rec.id)
                # `card_rect` is the card's real geometry (boundingRect carries the glow
                # margin). The item exists already: `add_node` notifies the document
                # synchronously and the scene syncs on that.
                y += (item.card_rect().height() if item is not None else 110.0) \
                    + SOURCE_STACK_GAP
        finally:
            QApplication.restoreOverrideCursor()

        items = [self.scene.node_items[r.id] for r in [a[0] for a in added] + outs
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
        if outs:
            self.statusBar().showMessage(
                self.statusBar().currentMessage() + self._published_note(outs))
            self.pull_node(added[0][0].id)      # the Viewer shows what was loaded

    def file_open(self) -> None:
        path, _f = QFileDialog.getOpenFileName(self, "Open graph", "", FILE_FILTER)
        if not path:
            return
        try:
            self.workspace.load_file(path)       # reuses self.doc for the active page
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        self._forget_display_state()
        # the session's per-page state described the LAST file's pages (their ids repeat)
        self._welcome_dismissed.clear()
        self._seeded_pages.clear()
        self._closed_pages.clear()
        for c in self.canvases():
            c._viewpoints.clear()          # where it looked at the LAST file's pages
            c.last_of_kind.clear()
        self._refresh_page_views()
        self.view.fit_all()
        if len(self.workspace.pages) > 1:
            act = self.workspace.page(self.workspace.active)
            self.statusBar().showMessage(
                f"workspace of {len(self.workspace.pages)} pages — showing {act.name!r}; "
                f"the tabs above the canvas (and the Pages panel) change page")
        if self.doc.has_unedited_structure:
            QMessageBox.information(
                self, "Zones / groups preserved",
                "This file contains zones or node groups that this build can't edit "
                "yet. They are shown as their member nodes and preserved unchanged on "
                "save — editing/creating them in the GUI is a later phase.")

    def _stamp_all_movies(self) -> None:
        """Before a save: write the Viewer's current LUTs into every linked movie channel
        of every page, so the file on disk reproduces the movies the user was looking at."""
        for pid in list(self.workspace.pages):
            for mid in self._movie_nodes(pid):
                self.stamp_movie_links(mid, pid)

    def file_save(self) -> None:
        if not self.doc.path:
            self.file_save_as()
            return
        self._stamp_all_movies()
        self.workspace.save_file(self.doc.path)
        self.statusBar().showMessage(f"saved {self.doc.path}")

    def file_save_as(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "Save graph", "graph.nd2graph.json",
                                               FILE_FILTER)
        if not path:
            return
        self._stamp_all_movies()
        self.workspace.save_file(path)
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
    def _sync_welcome(self, *_a) -> None:
        """Show each canvas's start card while its page holds no node — or nothing but the
        Page Input it was seeded with — unless it was dismissed on that page; and word it
        for the page's kind (V4.00 step 11): a load on Image Input, the kind's page recipes
        on a later page (a banner along the bottom, clear of the seeded Input)."""
        from nodegraph import roles as R
        ws = self.workspace
        gone = self._welcome_dismissed
        for c in self.canvases():
            page = ws.pages.get(c.page_id)
            show = (page is not None and page.id not in gone and not any(
                r.op_key != PAGE_INPUT_OP for r in page.doc.nodes.values()))
            if show:
                labels = dict(ws.available_sources(page.id))
                reads = [labels[s] for s in (
                    str(r.params.get(PAGE_SOURCE_KEY) or "") for r in page.doc.nodes.values()
                    if r.op_key == PAGE_INPUT_OP) if s in labels]
                c.welcome.configure(
                    page.kind, reads=", ".join(reads), has_upstream=bool(labels),
                    recipes=[(r.name, r.description) for r in self._recipes_for(page.kind)],
                    description=str(R.page_meta(page.kind).get("description") or ""))
            c.welcome.setVisible(show)
            if c.welcome.isVisible():
                c.welcome.raise_()
                c.minimap.raise_()          # the mini-map still owns its corner

    def focus_palette(self) -> None:
        """Put the cursor in the Nodes palette search (the welcome card's shortcut to
        placing a first node)."""
        dock = self.shell.dock_of(self.palette)
        if dock is not None:
            dock.show()
            dock.raise_()
        self.palette.focus_search()

    def build_example_workspace(self) -> None:
        """The welcome card's *Example graph* (V4.00 step 11): File → New, then one analysis
        spread over the four standard pages (:func:`nodelab_v2.workspace.build_example`),
        shown from the Image Input page — every page boundary named and bound, the shape the
        standard workflow produces."""
        self.file_new()
        pids = build_example(self.workspace, reset=False)
        self._show_page(self._canvas, pids["input"])
        self.view.fit_all()
        self.statusBar().showMessage(
            "example workspace loaded — the page switcher (top left) walks Image Input → "
            "Refinement → Processing → Analysis; double-click a node to view it")

    @_needs_topology
    def build_demo(self) -> None:
        """Replace the ACTIVE page with the small flat example chain (Load → Select → enhance
        → threshold → label → measure, plus a deconvolve → Viewer branch). What the GUI probe
        drives; since V4.00 step 11 the welcome card builds :meth:`build_example_workspace`
        instead — the same analysis spread over the four standard pages."""
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
