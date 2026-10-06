"""The **Pages** panel (V4.00 step 11): every page of the workspace at a glance.

A workspace grows pages — an Input page, two refinements of one dish, a processing page per
condition, linked copies of each — and the page switcher's menu or a row of tabs shows them
only one name at a time. This panel lists them all, **grouped by kind in pipeline order**
(Image Input → Image Refinement → Image Processing → Analysis, then Free), and under each
page what connects it to the others: what its Page Inputs **read** and which Outputs it
**publishes**. A master page carries ★, a linked page names its master.

Click a page to show it on the active canvas; double-click to rename; right-click for the
page switcher's menu (duplicate, link, set as master, save as page recipe, close tab,
delete). A page whose TAB was closed (V4.00 step 11d) is listed here in italics, "tab
closed": the panel is where every page stays reachable, and clicking it opens its tab
again. The panel only reads the workspace — every change goes through the window.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from nodelab_v2 import theme as T
from nodelab_v2.canvas import kind_icon

_PID = Qt.UserRole
_ROLE = Qt.UserRole + 1          # "kind" | "page" | "detail"
_OPEN = Qt.UserRole + 2          # a page row: is its tab open

#: one row of the panel: (page id, label, kind, master page name or "", reads, publishes,
#: whether the page's tab is open)
PageRow = Tuple[str, str, str, str, Tuple[str, ...], Tuple[str, ...], bool]


class PagesPanel(QWidget):
    """The workspace's pages by kind, with what each reads and publishes."""

    #: show this page on the active canvas
    page_requested = Signal(str)
    #: rename this page
    rename_requested = Signal(str)
    #: the page switcher's menu for this page, at a global position
    menu_requested = Signal(str, QPoint)
    #: *+ New page…*
    new_page_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pagesPanel")
        self._sig: Optional[tuple] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        head = QHBoxLayout()
        self._count = QLabel("")
        self._count.setProperty("role", "muted")
        head.addWidget(self._count)
        head.addStretch(1)
        self._new = QToolButton()
        self._new.setText("+ New page…")
        self._new.setToolTip("A new page: empty, from a page recipe, or linked to a master")
        self._new.setCursor(Qt.PointingHandCursor)
        self._new.clicked.connect(lambda _=False: self.new_page_requested.emit())
        head.addWidget(self._new)
        lay.addLayout(head)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
        self.tree.setIndentation(12)
        self.tree.setRootIsDecorated(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.itemClicked.connect(self._on_click)
        self.tree.itemDoubleClicked.connect(self._on_double)
        self.tree.customContextMenuRequested.connect(self._on_menu)
        lay.addWidget(self.tree, 1)
        self.restyle()

    # ── content ──────────────────────────────────────────────────────────────
    def refresh(self, kinds: Sequence[Tuple[str, str]], rows: Sequence[PageRow],
                active: str) -> None:
        """Rebuild from ``kinds`` (``(kind, label)`` in pipeline order) and ``rows``; only
        when something changed (it is called on every workspace change)."""
        sig = (tuple(kinds), tuple(rows), active)
        if sig == self._sig:
            return
        if self._sig is not None and sig[:2] == self._sig[:2]:
            self._sig = sig                  # only the active page moved: no rebuild
            self._mark_active(active)
            return
        self._sig = sig
        self.tree.clear()
        bold = QFont(self.font())
        bold.setBold(True)
        current: Optional[QTreeWidgetItem] = None
        for kind, label in kinds:
            mine = [r for r in rows if r[2] == kind]
            if not mine:
                continue
            head = QTreeWidgetItem([f"{label}  ·  {len(mine)}"])
            head.setIcon(0, kind_icon(kind))
            head.setData(0, _ROLE, "kind")
            head.setFlags(Qt.ItemIsEnabled)
            head.setFont(0, bold)
            head.setForeground(0, QBrush(T.INK))
            head.setBackground(0, QBrush(T.mix(T.PANEL, T.ACCENT_DIM, 0.45)))
            self.tree.addTopLevelItem(head)
            for pid, text, _k, master, reads, publishes, is_open in mine:
                it = QTreeWidgetItem([text if is_open else f"{text}   · tab closed"])
                it.setData(0, _PID, pid)
                it.setData(0, _ROLE, "page")
                it.setData(0, _OPEN, bool(is_open))
                tip = self._tip(text, master, reads, publishes)
                if not is_open:
                    tip += "\nits tab is closed — click to show the page and open its tab"
                    shut = QFont(self.font())
                    shut.setItalic(True)
                    it.setFont(0, shut)
                    it.setForeground(0, QBrush(T.MUTED))
                it.setToolTip(0, tip)
                if pid == active:
                    it.setFont(0, bold)
                    current = it
                head.addChild(it)
                details: List[str] = []
                if master:
                    details.append(f"follows  {master}")
                details += [f"reads  {r}" for r in reads]
                if publishes:
                    details.append("publishes  " + ", ".join(publishes))
                for d in details:
                    sub = QTreeWidgetItem([d])
                    sub.setData(0, _PID, pid)
                    sub.setData(0, _ROLE, "detail")
                    sub.setForeground(0, QBrush(T.MUTED))
                    it.addChild(sub)
                it.setExpanded(True)
            head.setExpanded(True)
        n = len(rows)
        self._count.setText(f"{n} page{'' if n == 1 else 's'}")
        if current is not None:
            self.tree.setCurrentItem(current)

    @staticmethod
    def _tip(text: str, master: str, reads: Sequence[str], publishes: Sequence[str]) -> str:
        lines = [text]
        if master:
            lines.append(f"linked to “{master}” — its graph, values of its own")
        lines.append("reads: " + (", ".join(reads) if reads else "nothing (no Page Input)"))
        lines.append("publishes: " + (", ".join(publishes) if publishes else
                                      "nothing yet (no named Page Output)"))
        lines.append("click to show · double-click to rename · right-click for the page menu")
        return "\n".join(lines)

    def _mark_active(self, active: str) -> None:
        bold = QFont(self.font())
        bold.setBold(True)
        plain = QFont(self.font())
        shut = QFont(self.font())
        shut.setItalic(True)
        for it in self.page_items():
            on = it.data(0, _PID) == active
            it.setFont(0, bold if on else (plain if it.data(0, _OPEN) is not False else shut))
            if on:
                self.tree.setCurrentItem(it)

    def page_items(self) -> List[QTreeWidgetItem]:
        """Every page row, in panel order."""
        out = []
        for i in range(self.tree.topLevelItemCount()):
            head = self.tree.topLevelItem(i)
            out += [head.child(j) for j in range(head.childCount())]
        return out

    # ── interaction ──────────────────────────────────────────────────────────
    def _on_click(self, item: QTreeWidgetItem, _col: int) -> None:
        pid = item.data(0, _PID)
        if pid:
            # after the click handler returns: showing a page refreshes this very tree
            QTimer.singleShot(0, lambda p=str(pid): self.page_requested.emit(p))

    def _on_double(self, item: QTreeWidgetItem, _col: int) -> None:
        if item.data(0, _ROLE) == "page":
            pid = str(item.data(0, _PID))
            QTimer.singleShot(0, lambda p=pid: self.rename_requested.emit(p))

    def _on_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        pid = item.data(0, _PID) if item is not None else None
        if pid:
            g = self.tree.viewport().mapToGlobal(pos)
            QTimer.singleShot(0, lambda p=str(pid), g=g: self.menu_requested.emit(p, g))

    def restyle(self) -> None:
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QWidget#pagesPanel {{ background:{T.PANEL.name()}; }}
            QLabel {{ color:{T.MUTED.name()}; background:transparent; font-size:9px; }}
            QTreeWidget {{ background:{T.PANEL.name()}; border:0; color:{T.INK_2.name()};
                outline:0; }}
            QTreeWidget::item {{ padding:2px 2px; }}
            QTreeWidget::item:selected {{ background:{T.mix(T.PANEL, T.ACCENT, 0.22).name()};
                color:{T.INK.name()}; }}
            QToolButton {{ color:{T.ACCENT.name()}; background:transparent; border:0;
                padding:2px 6px; }}
            QToolButton:hover {{ background:{T.PANEL_HI.name()}; color:{T.INK.name()}; }}
        """)
        self._sig = None               # colours are baked into items: rebuild on next refresh


__all__ = ["PagesPanel", "PageRow"]
