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
card and the mini-map.
"""
from __future__ import annotations

from typing import Dict, Tuple

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import QMenu, QVBoxLayout, QWidget

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
        """The switcher names the page shown, with its kind's dot."""
        text, kind = self._host.page_title(self.page_id)
        self.view.set_page_title(text, kind_icon(kind))

    def set_active(self, on: bool) -> None:
        """Mark this canvas as the one the user works in (an accent on its switcher)."""
        self.view.set_canvas_active(on)

    def restyle(self) -> None:
        self.view.restyle()
        self.welcome.restyle()
        self.minimap.restyle()


__all__ = ["CanvasPanel", "PAGE_KIND_COLORS", "kind_icon"]
