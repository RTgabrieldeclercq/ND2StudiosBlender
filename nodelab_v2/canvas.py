"""The canvas panel (V4.00 step 5): a node-graph view bound to one PAGE of the workspace.

A canvas shows a page's graph and lets the user change which page that is, with the
**page switcher** — the button in its top-left corner, which also adds, duplicates,
renames and deletes pages. The window's main canvas is its central widget; every further
canvas is a dock (``canvas:<n>``) of the :class:`~nodelab_v2.shell.DockShell`, so two pages
can sit side by side or one can pop out onto a second screen.

The SCENE belongs to the page, not to the canvas (:meth:`MainWindow.scene_for`): two
canvases on one page show one graph, and a page keeps its selection, its cards' run states
and its layout while no canvas shows it. A canvas owns only what is about LOOKING: the
view, where it was looking at each page it has shown, and the HUD over it — the welcome
card and the mini-map — and the **page tabs** above it (V4.00 step 11): one tab per page, so
every page of the workspace stays in sight and one click away.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import QHBoxLayout, QMenu, QTabBar, QToolButton, QVBoxLayout, QWidget

from nodelab_v2 import theme as T

from nodelab_v2.minimap import MiniMapOverlay
from nodelab_v2.scene import GraphView
from nodelab_v2.welcome import WelcomeCard

#: the dot beside a page's name on its switcher, so the kind of page is visible at a glance
#: (the same hues in the page menu)
PAGE_KIND_COLORS = {"input": "#4f8fe6", "refine": "#3fb3a3", "process": "#e0a43a",
                    "analyze": "#a77be0", "free": "#8b93a1"}


def kind_icon(kind: str, size: int = 10) -> QIcon:
    """A filled dot in ``kind``'s colour (:data:`PAGE_KIND_COLORS`)."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(PAGE_KIND_COLORS.get(kind, PAGE_KIND_COLORS["free"])))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pix)


class PageTabs(QWidget):
    """The page tab strip above a canvas (V4.00 step 11): one tab per page of the workspace,
    in page order, with its kind's dot — the page shown is the current tab. Click a tab to
    show that page here; drag one to reorder the pages; double-click to rename; right-click
    for the page switcher's menu; **+** for *New page…*. The strip only mirrors the
    workspace (:meth:`sync`); every change goes through the window."""

    def __init__(self, canvas: "CanvasPanel") -> None:
        super().__init__(canvas)
        self.setObjectName("pageTabs")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._canvas = canvas
        self._sig: Optional[tuple] = None
        self._items: Dict[str, Tuple[str, str, str, str]] = {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 0)
        lay.setSpacing(4)
        self.bar = QTabBar(self)
        self.bar.setMovable(True)
        self.bar.setExpanding(False)
        self.bar.setUsesScrollButtons(True)
        self.bar.setElideMode(Qt.ElideRight)
        self.bar.setDocumentMode(True)
        self.bar.setDrawBase(False)
        self.bar.setContextMenuPolicy(Qt.CustomContextMenu)
        lay.addWidget(self.bar, 1)
        self.add_btn = QToolButton(self)
        self.add_btn.setText("+")
        self.add_btn.setToolTip("New page… — empty, from a page recipe, or linked to a master")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.add_btn)
        self.bar.currentChanged.connect(self._on_current)
        self.bar.tabMoved.connect(self._on_moved)
        self.bar.tabBarDoubleClicked.connect(self._on_double)
        self.bar.customContextMenuRequested.connect(self._on_menu)
        self.add_btn.clicked.connect(lambda _=False: canvas.request_new_page())
        self.restyle()

    def sync(self, items, current: str) -> None:
        """Mirror ``items`` — ``(page id, text, kind, tooltip)`` in page order — with
        ``current`` selected; a no-op when nothing changed."""
        items = tuple(items)
        sig = (items, current)
        if sig == self._sig:
            return
        if self._sig is not None and items == self._sig[0]:
            # only the page shown moved: select its tab, never rebuild under a click
            self._sig = sig
            idx = next((i for i in range(self.bar.count()) if self.bar.tabData(i) == current),
                       -1)
            if idx >= 0 and idx != self.bar.currentIndex():
                self.bar.blockSignals(True)
                self.bar.setCurrentIndex(idx)
                self.bar.blockSignals(False)
            return
        self._sig = sig
        self._items = {it[0]: it for it in items}
        self.bar.blockSignals(True)
        try:
            while self.bar.count():
                self.bar.removeTab(0)
            for pid, text, kind, tip in items:
                i = self.bar.addTab(kind_icon(kind), text)
                self.bar.setTabData(i, pid)
                self.bar.setTabToolTip(i, tip)
            idx = next((i for i in range(self.bar.count()) if self.bar.tabData(i) == current),
                       -1)
            if idx >= 0:
                self.bar.setCurrentIndex(idx)
        finally:
            self.bar.blockSignals(False)

    def page_ids(self):
        return [self.bar.tabData(i) for i in range(self.bar.count())]

    def _on_current(self, i: int) -> None:
        pid = self.bar.tabData(i)
        if pid and pid != self._canvas.page_id:
            # after the tab bar's own press handling: showing a page re-syncs this strip
            QTimer.singleShot(0, lambda p=pid: self._canvas.request_page(p))

    def _on_moved(self, _frm: int, to: int) -> None:
        # the drag already shows the new order: record it, so the window's re-sync after the
        # move finds nothing to rebuild (a rebuild would end the drag under the mouse)
        order = self.page_ids()
        self._sig = (tuple(self._items[p] for p in order if p in self._items),
                     self._canvas.page_id)
        self._canvas.request_move(self.bar.tabData(to), to)

    def _on_double(self, i: int) -> None:
        pid = self.bar.tabData(i)
        if pid:
            QTimer.singleShot(0, lambda p=pid: self._canvas.request_rename(p))

    def _on_menu(self, pos) -> None:
        i = self.bar.tabAt(pos)
        pid = self.bar.tabData(i) if i >= 0 else None
        if pid:
            g = self.bar.mapToGlobal(pos)
            QTimer.singleShot(0, lambda p=pid, g=g: self._canvas.request_menu(p, g))

    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget#pageTabs {{ background:{T.BG.name()};
                border-bottom:1px solid {T.BORDER.name()}; }}
            QTabBar {{ background:transparent; }}
            QTabBar::tab {{ background:{T.BODY.name()}; color:{T.MUTED.name()};
                padding:4px 12px; margin-right:2px; font-size:10px;
                border:1px solid {T.BORDER.name()}; border-bottom:0;
                border-top-left-radius:5px; border-top-right-radius:5px; }}
            QTabBar::tab:selected {{ background:{T.PANEL.name()}; color:{T.INK.name()};
                border-top:2px solid {T.ACCENT.name()}; }}
            QTabBar::tab:hover:!selected {{ background:{T.PANEL_HI.name()};
                color:{T.INK.name()}; }}
            QToolButton {{ color:{T.ACCENT.name()}; background:transparent; border:0;
                padding:2px 8px; font-weight:800; }}
            QToolButton:hover {{ background:{T.PANEL_HI.name()}; color:{T.INK.name()}; }}
        """)


class CanvasPanel(QWidget):
    """One canvas: a :class:`~nodelab_v2.scene.GraphView` on a page's scene, its welcome
    card and mini-map, and the page switcher.

    ``host`` is the window, which owns the workspace: it makes the scenes
    (``scene_for(page_id)``), names a page (``page_title(page_id)`` → ``(text, kind)``) and
    fills the switcher's menu (``fill_page_menu(menu, canvas)``)."""

    #: the user started working in this canvas — a press on it, or a drop onto it
    activated = Signal(object)
    #: the canvas now shows another page — ``(canvas, page_id)``
    page_changed = Signal(object, str)

    def __init__(self, host, page_id: str) -> None:
        super().__init__()
        self._host = host
        self.page_id = page_id
        #: page id → (transform, scene point at the centre): where this canvas was looking
        #: at each page it has shown, so switching back lands where the user left it
        self._viewpoints: Dict[str, Tuple[QTransform, QPointF]] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # every page of the workspace, one tab each (V4.00 step 11)
        self.tabs = PageTabs(self)
        lay.addWidget(self.tabs)
        self.view = GraphView(host.scene_for(page_id))
        lay.addWidget(self.view)
        # an EMPTY page: this card invites the first node and steps aside once it has one
        self.welcome = WelcomeCard(self.view)
        # maximized (Ctrl+Space / ⛶): the active Viewer re-homed into this HUD frame
        self.minimap = MiniMapOverlay(self.view)
        self.view.pressed.connect(lambda: self.activated.emit(self))
        self._menu = QMenu(self)
        self._menu.aboutToShow.connect(lambda: host.fill_page_menu(self._menu, self))
        self.view.page_button.setMenu(self._menu)
        self.sync_title()

    # ── the page shown ─────────────────────────────────────────────────────────
    def set_page(self, page_id: str) -> None:
        """Show ``page_id`` here, remembering where this canvas was looking at the old one."""
        if page_id == self.page_id:
            return
        self._viewpoints[self.page_id] = (
            QTransform(self.view.transform()),
            self.view.mapToScene(self.view.viewport().rect().center()))
        self.page_id = page_id
        self.view.setScene(self._host.scene_for(page_id))
        vp = self._viewpoints.get(page_id)
        if vp is not None:
            self.view.setTransform(vp[0])
            self.view.centerOn(vp[1])
        else:
            self.view.resetTransform()
            if self.view.scene().items():
                self.view.fit_all()
        self.sync_title()
        self.page_changed.emit(self, page_id)

    def forget_page(self, page_id: str) -> None:
        """Drop the remembered viewpoint of a page that no longer exists."""
        self._viewpoints.pop(page_id, None)

    def sync_title(self) -> None:
        """The switcher names the page shown, with its kind's dot; the tabs follow."""
        text, kind = self._host.page_title(self.page_id)
        self.view.set_page_title(text, kind_icon(kind))
        self.sync_tabs()

    def sync_tabs(self) -> None:
        """Mirror the workspace's pages on the tab strip (cheap when nothing changed)."""
        self.tabs.sync(self._host.page_tab_items(), self.page_id)

    # ── what the tab strip asks of the window ──────────────────────────────────
    def request_page(self, page_id: str) -> None:
        self._host._show_page(self, page_id)

    def request_move(self, page_id: str, index: int) -> None:
        self._host.move_page(page_id, index)

    def request_rename(self, page_id: str) -> None:
        self._host.rename_page(page_id)

    def request_menu(self, page_id: str, global_pos) -> None:
        if page_id != self.page_id:
            self._host._show_page(self, page_id)
        self._menu.exec(global_pos)

    def request_new_page(self) -> None:
        self._host.new_page_dialog(kind=self._host.next_page_kind(self.page_id), canvas=self)

    def set_active(self, on: bool) -> None:
        """Mark this canvas as the one the user works in (an accent on its switcher)."""
        self.view.set_canvas_active(on)

    def restyle(self) -> None:
        self.tabs.restyle()
        self.view.restyle()
        self.welcome.restyle()
        self.minimap.restyle()


__all__ = ["CanvasPanel", "PageTabs", "PAGE_KIND_COLORS", "kind_icon"]
