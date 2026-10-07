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
card and the mini-map — and the **page tabs** above it (V4.00 step 11): a row of page KINDS
and, under it, the pages of the kind shown (step 11d), so every page of the workspace stays
one click away and a closed tab is never a lost page.
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
    """The page tabs above a canvas (V4.00 step 11, two rows since step 11d).

    The TOP row has one tab per page KIND the workspace has pages of, in pipeline order
    (Image Input → Image Refinement → Image Processing → Analysis, then Free), with the kind's
    dot and, when there are several, how many pages it holds. The row UNDER it has the pages
    of the shown page's kind, as **sub-tabs**: click one to show that page here, drag to
    reorder those pages, double-click to rename, right-click for the page menu, ✕ to close the
    tab, **+** for a new page of this kind.

    Closing a sub-tab does not delete the page: it leaves the strip and stays in the Pages
    panel, and showing it again by any route — the panel, the page menu, its kind's tab when
    every page of the kind is closed — brings its tab back. The strip only mirrors the window
    (:meth:`sync`); every change goes through the window."""

    def __init__(self, canvas: "CanvasPanel") -> None:
        super().__init__(canvas)
        self.setObjectName("pageTabs")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._canvas = canvas
        self._sig: Optional[tuple] = None
        self._items: Dict[str, tuple] = {}
        #: the kind whose pages the sub-tab row shows
        self.kind = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 0)
        lay.setSpacing(0)
        self.kind_bar = self._make_bar(movable=False)
        self.kind_bar.setObjectName("kindTabs")
        lay.addWidget(self.kind_bar)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.bar = self._make_bar(movable=True)
        self.bar.setObjectName("pageSubTabs")
        row.addWidget(self.bar, 1)
        self.add_btn = QToolButton(self)
        self.add_btn.setText("+")
        self.add_btn.setToolTip("New page of this kind… — empty, from a page recipe, or "
                                "linked to a master")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.add_btn)
        lay.addLayout(row)
        self.kind_bar.currentChanged.connect(self._on_kind)
        self.bar.currentChanged.connect(self._on_current)
        self.bar.tabMoved.connect(self._on_moved)
        self.bar.tabBarDoubleClicked.connect(self._on_double)
        self.bar.customContextMenuRequested.connect(self._on_menu)
        self.add_btn.clicked.connect(lambda _=False: canvas.request_new_page(self.kind))
        self.restyle()

    def _make_bar(self, *, movable: bool) -> QTabBar:
        bar = QTabBar(self)
        bar.setMovable(movable)
        bar.setExpanding(False)
        bar.setUsesScrollButtons(True)
        bar.setElideMode(Qt.ElideNone)          # a page name is read whole; more pages scroll
        bar.setDocumentMode(True)
        bar.setDrawBase(False)
        bar.setContextMenuPolicy(Qt.CustomContextMenu)
        return bar

    def sync(self, kinds, items, current: str) -> None:
        """Mirror ``kinds`` — ``(kind, label)`` in pipeline order — and ``items`` — ``(page
        id, text, kind, tooltip, open)`` in page order — with ``current`` the page shown; a
        no-op when nothing changed. Only an OPEN page has a sub-tab (the page shown always
        is: showing a page opens its tab)."""
        kinds, items = tuple(kinds), tuple(items)
        sig = (kinds, items, current)
        if sig == self._sig:
            return
        if self._sig is not None and (kinds, items) == self._sig[:2] \
                and self._kind_of(current) == self.kind:
            # only the page shown moved, within the kind shown: select its tab, never
            # rebuild under a click
            self._sig = sig
            self._select(self.bar, current)
            return
        self._sig = sig
        self._items = {it[0]: it for it in items}
        kind = self._kind_of(current) or (kinds[0][0] if kinds else "")
        if kind != self.kind:
            self.kind = kind
            self.restyle()                 # the shown sub-tab is underlined in its kind's hue
        for bar in (self.kind_bar, self.bar):
            bar.blockSignals(True)
        try:
            while self.kind_bar.count():
                self.kind_bar.removeTab(0)
            for kind, label in kinds:
                n = sum(1 for it in items if it[2] == kind)
                if not n:
                    continue
                i = self.kind_bar.addTab(kind_icon(kind), label + (f"  {n}" if n > 1 else ""))
                self.kind_bar.setTabData(i, kind)
                shut = sum(1 for it in items if it[2] == kind and not it[4])
                self.kind_bar.setTabToolTip(
                    i, f"{label}: {n} page{'' if n == 1 else 's'}"
                    + (f", {shut} with the tab closed (the Pages panel lists every page)"
                       if shut else "") + " — click to show this kind's pages")
            self._select(self.kind_bar, self.kind)
            while self.bar.count():
                self.bar.removeTab(0)
            # the last open tab of the whole workspace has no ✕: a canvas must show a page
            closable = sum(1 for it in items if it[4] or it[0] == current) > 1
            for pid, text, kind, tip, is_open in items:
                if kind != self.kind or not (is_open or pid == current):
                    continue
                i = self.bar.addTab(text)
                self.bar.setTabData(i, pid)
                self.bar.setTabToolTip(i, tip)
                if closable:
                    self.bar.setTabButton(i, QTabBar.RightSide, self._close_button(pid))
            self._select(self.bar, current)
        finally:
            for bar in (self.kind_bar, self.bar):
                bar.blockSignals(False)

    def _kind_of(self, page_id: str) -> str:
        it = self._items.get(page_id)
        return it[2] if it is not None else ""

    @staticmethod
    def _select(bar: QTabBar, data) -> None:
        idx = next((i for i in range(bar.count()) if bar.tabData(i) == data), -1)
        if idx >= 0 and idx != bar.currentIndex():
            was = bar.blockSignals(True)
            bar.setCurrentIndex(idx)
            bar.blockSignals(was)

    def page_ids(self):
        """The pages with a sub-tab, in tab order."""
        return [self.bar.tabData(i) for i in range(self.bar.count())]

    def kinds(self):
        """The kinds on the top row, in tab order."""
        return [self.kind_bar.tabData(i) for i in range(self.kind_bar.count())]

    # every click is acted on AFTER the tab bar's own handling returns: showing a page
    # re-syncs this strip, and rebuilding a QTabBar inside its own signal is unsafe
    def _on_kind(self, i: int) -> None:
        kind = self.kind_bar.tabData(i)
        if kind and kind != self.kind:
            QTimer.singleShot(0, lambda k=kind: self._canvas.request_kind(k))

    def _on_current(self, i: int) -> None:
        pid = self.bar.tabData(i)
        if pid and pid != self._canvas.page_id:
            QTimer.singleShot(0, lambda p=pid: self._canvas.request_page(p))

    def _on_moved(self, _frm: int, _to: int) -> None:
        # the drag already shows the new order: record it, so the window's re-sync after the
        # move finds nothing to rebuild (a rebuild would end the drag under the mouse). The
        # dragged pages trade places among the slots they hold in the page order; every other
        # page keeps its place.
        order = self.page_ids()
        moved, seq = set(order), iter(order)
        glob = [next(seq) if p in moved else p for p in self._items]
        self._items = {p: self._items[p] for p in glob}
        if self._sig is not None:
            self._sig = (self._sig[0], tuple(self._items.values()), self._sig[2])
        self._canvas.request_reorder(glob)

    def _on_double(self, i: int) -> None:
        pid = self.bar.tabData(i)
        if pid:
            QTimer.singleShot(0, lambda p=pid: self._canvas.request_rename(p))

    def _close_button(self, page_id: str) -> QToolButton:
        """A sub-tab's ✕ — the app's own, themed, rather than the style's red icon."""
        b = QToolButton(self.bar)
        b.setObjectName("tabClose")
        b.setText("✕")
        b.setAutoRaise(True)
        b.setCursor(Qt.PointingHandCursor)
        b.setToolTip("Close this tab — the page stays in the Pages panel and under its "
                     "kind's tab")
        b.clicked.connect(lambda _=False, p=page_id: QTimer.singleShot(
            0, lambda: self._canvas.request_close(p)))
        return b

    def close_buttons(self):
        """``{page id: its ✕}`` for the sub-tabs that have one."""
        out = {}
        for i in range(self.bar.count()):
            b = self.bar.tabButton(i, QTabBar.RightSide)
            if b is not None:
                out[self.bar.tabData(i)] = b
        return out

    def _on_menu(self, pos) -> None:
        i = self.bar.tabAt(pos)
        pid = self.bar.tabData(i) if i >= 0 else None
        if pid:
            g = self.bar.mapToGlobal(pos)
            QTimer.singleShot(0, lambda p=pid, g=g: self._canvas.request_menu(p, g))

    def restyle(self) -> None:
        dot = PAGE_KIND_COLORS
        self.setStyleSheet(f"""
            QWidget#pageTabs {{ background:{T.BG.name()};
                border-bottom:1px solid {T.BORDER.name()}; }}
            QTabBar {{ background:transparent; }}
            QTabBar#kindTabs::tab {{ background:{T.BODY.name()}; color:{T.MUTED.name()};
                padding:4px 12px; margin-right:2px; font-size:10px; font-weight:700;
                border:1px solid {T.BORDER.name()}; border-bottom:0;
                border-top-left-radius:5px; border-top-right-radius:5px; }}
            QTabBar#kindTabs::tab:selected {{ background:{T.PANEL.name()};
                color:{T.INK.name()}; border-top:2px solid {T.ACCENT.name()}; }}
            QTabBar#kindTabs::tab:hover:!selected {{ background:{T.PANEL_HI.name()};
                color:{T.INK.name()}; }}
            QTabBar#pageSubTabs::tab {{ background:{T.PANEL.name()};
                color:{T.MUTED.name()}; padding:3px 6px 3px 10px; margin-right:1px;
                font-size:10px; border:0; border-bottom:2px solid transparent; }}
            QTabBar#pageSubTabs::tab:selected {{ color:{T.INK.name()};
                background:{T.PANEL_HI.name()};
                border-bottom:2px solid {dot.get(self.kind, T.ACCENT.name())}; }}
            QTabBar#pageSubTabs::tab:hover:!selected {{ color:{T.INK.name()}; }}
            QToolButton {{ color:{T.ACCENT.name()}; background:transparent; border:0;
                padding:2px 8px; font-weight:800; }}
            QToolButton:hover {{ background:{T.PANEL_HI.name()}; color:{T.INK.name()}; }}
            QToolButton#tabClose {{ color:{T.MUTED.name()}; padding:0 3px; margin-left:4px;
                font-size:9px; font-weight:400; border-radius:3px; }}
            QToolButton#tabClose:hover {{ color:{T.INK.name()};
                background:{T.mix(T.PANEL_HI, T.ERROR, 0.35).name()}; }}
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
        #: kind → the page of that kind this canvas showed last — what its kind tab returns to
        self.last_of_kind: Dict[str, str] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        # the page kinds and, under them, the pages of the kind shown (V4.00 step 11/11d)
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
        # the action pill (2026-10-07): its menu is built when it opens, from the selection
        # of the page shown here (the window's `fill_action_menu`)
        self.actions_menu = QMenu(self)
        self.actions_menu.aboutToShow.connect(
            lambda: host.fill_action_menu(self.actions_menu, self))
        self.view.action_pill.setMenu(self.actions_menu)
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
        self.tabs.sync(self._host.page_tab_kinds(), self._host.page_tab_items(), self.page_id)

    # ── what the tab strip asks of the window ──────────────────────────────────
    def request_page(self, page_id: str) -> None:
        self._host._show_page(self, page_id)

    def request_kind(self, kind: str) -> None:
        self._host.show_kind(self, kind)

    def request_reorder(self, page_ids) -> None:
        self._host.reorder_pages(list(page_ids))

    def request_close(self, page_id: str) -> None:
        self._host.close_page_tab(page_id)

    def request_rename(self, page_id: str) -> None:
        self._host.rename_page(page_id)

    def request_menu(self, page_id: str, global_pos) -> None:
        if page_id != self.page_id:
            self._host._show_page(self, page_id)
        self._menu.exec(global_pos)

    def request_new_page(self, kind: str = "") -> None:
        """*New page…* — preset to ``kind`` (the sub-tab row's +: another page of the kind
        shown), else to the kind that follows the page shown."""
        self._host.new_page_dialog(kind=kind or self._host.next_page_kind(self.page_id),
                                   canvas=self)

    def set_active(self, on: bool) -> None:
        """Mark this canvas as the one the user works in (an accent on its switcher)."""
        self.view.set_canvas_active(on)

    def restyle(self) -> None:
        self.tabs.restyle()
        self.view.restyle()
        self.welcome.restyle()
        self.minimap.restyle()


__all__ = ["CanvasPanel", "PageTabs", "PAGE_KIND_COLORS", "kind_icon"]
