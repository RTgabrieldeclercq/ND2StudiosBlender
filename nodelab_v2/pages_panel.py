"""The **Pages** panel (V4.00 step 11): every page of the workspace — and, since step 11e, every
node of every page — at a glance.

A workspace grows pages — an Input page, two refinements of one dish, a processing page per
condition, linked copies of each — and the page switcher's menu or a row of tabs shows them
only one name at a time. This panel lists them all, **grouped by kind in pipeline order**
(Image Input → Image Refinement → Image Processing → Analysis, then Free), and under each page
**its graph as a hierarchy** (V4.00 step 11e, :meth:`nodelab_v2.workspace.Workspace.page_outline`):
where data enters the page first — its Page Inputs, each with a dot in the colour of the page
kind it READS, then its Load cards — and from each the chain it feeds, a branch nesting one
level under the node it leaves. The page's **Outputs** are what later pages read, so they
stand out: a row tinted in the page kind's colour, the VARIABLE NAME in bold, and which pages
read it. A master page carries ★; a linked page says which master it follows and whether it
keeps changes of its own (*modified*), whose own nodes are marked ``+``.

**On / off.** Every node that can be switched off (muted: its input passed straight through)
has a switch in the second column — only a node that keeps the kind of data qualifies
(:func:`nodelab_v2.document.pass_through_reason`; a threshold, a label, a measurement or a plot
has none, and its tooltip says why). A node switched off is listed struck through. On a linked
page the switch is that page's own setting, like a value.

Click a page to show it on the active canvas, a node to show its page with that node selected
and in view; double-click a page to rename; right-click for the page switcher's menu. A page
whose TAB was closed (V4.00 step 11d) is listed in italics, "tab closed": the panel is where
every page stays reachable, and clicking it opens its tab again. The panel only reads the
workspace — every change goes through the window.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from nodelab_v2 import theme as T
from nodelab_v2.canvas import PAGE_KIND_COLORS, kind_icon

_PID = Qt.UserRole
_ROLE = Qt.UserRole + 1          # "kind" | "page" | "detail" | "node"
_OPEN = Qt.UserRole + 2          # a page row: is its tab open
_NID = Qt.UserRole + 3           # a node row: its node id
_SWITCH = 1                      # the on / off column

#: one row of the panel: (page id, label, kind, master page name or "", reads, publishes,
#: whether the page's tab is open, the page's outline (V4.00 step 11e — a tuple of
#: :class:`~nodelab_v2.workspace.OutlineRow`), what the page keeps of its own ("" or e.g.
#: "modified · 2 own nodes"))
PageRow = Tuple[str, str, str, str, Tuple[str, ...], Tuple[str, ...], bool, tuple, str]


def _kind_color(kind: str) -> QColor:
    return QColor(PAGE_KIND_COLORS.get(kind, PAGE_KIND_COLORS["free"]))


class PagesPanel(QWidget):
    """The workspace's pages by kind, each page's nodes as a hierarchy, with on / off."""

    #: show this page on the active canvas
    page_requested = Signal(str)
    #: show this page with this node selected and in view
    node_requested = Signal(str, str)
    #: switch this node of this page off (True) or on (False)
    mute_requested = Signal(str, str, bool)
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
        self._filling = False
        #: pages the user folded — kept folded across the rebuilds every edit causes
        self._folded: Set[str] = set()
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
        self.tree.setColumnCount(2)
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(_SWITCH, QHeaderView.Fixed)
        hdr.resizeSection(_SWITCH, 26)
        self.tree.setIndentation(12)
        self.tree.setRootIsDecorated(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.itemClicked.connect(self._on_click)
        self.tree.itemDoubleClicked.connect(self._on_double)
        self.tree.itemChanged.connect(self._on_changed)
        self.tree.itemCollapsed.connect(lambda it: self._fold(it, True))
        self.tree.itemExpanded.connect(lambda it: self._fold(it, False))
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
        scroll = self.tree.verticalScrollBar().value()
        self._filling = True
        try:
            self._fill(kinds, rows, active)
        finally:
            self._filling = False
        self.tree.verticalScrollBar().setValue(scroll)

    def _fill(self, kinds, rows, active) -> None:
        self.tree.clear()
        bold = QFont(self.font())
        bold.setBold(True)
        current: Optional[QTreeWidgetItem] = None
        for kind, label in kinds:
            mine = [r for r in rows if r[2] == kind]
            if not mine:
                continue
            head = QTreeWidgetItem([f"{label}  ·  {len(mine)}", ""])
            head.setIcon(0, kind_icon(kind))
            head.setData(0, _ROLE, "kind")
            head.setFlags(Qt.ItemIsEnabled)
            head.setFont(0, bold)
            head.setForeground(0, QBrush(T.INK))
            for col in (0, _SWITCH):
                head.setBackground(col, QBrush(T.mix(T.PANEL, T.ACCENT_DIM, 0.45)))
            self.tree.addTopLevelItem(head)
            for (pid, text, _k, master, reads, publishes, is_open, outline,
                 own) in mine:
                it = QTreeWidgetItem([text if is_open else f"{text}   · tab closed", ""])
                it.setData(0, _PID, pid)
                it.setData(0, _ROLE, "page")
                it.setData(0, _OPEN, bool(is_open))
                tip = self._tip(text, master, reads, publishes, own)
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
                if master:
                    sub = QTreeWidgetItem([f"follows  {master}" + (f"  ·  {own}" if own else ""),
                                           ""])
                    sub.setData(0, _PID, pid)
                    sub.setData(0, _ROLE, "detail")
                    sub.setForeground(0, QBrush(T.MUTED))
                    sub.setToolTip(0, tip)
                    it.addChild(sub)
                self._fill_outline(it, pid, kind, outline)
                it.setExpanded(pid not in self._folded)
            head.setExpanded(True)
        n = len(rows)
        self._count.setText(f"{n} page{'' if n == 1 else 's'}")
        if current is not None:
            self.tree.setCurrentItem(current)

    def _fill_outline(self, page_item: QTreeWidgetItem, pid: str, kind: str,
                      outline: Sequence) -> None:
        """The page's nodes as its data-flow hierarchy: a row at depth ``d`` is a child of the
        last row at depth ``d - 1`` (the node its branch leaves)."""
        mono = QFont(T.MONO, 9)
        mono.setBold(True)
        parents: List[QTreeWidgetItem] = [page_item]
        for row in outline:
            depth = max(0, min(int(row.depth), len(parents) - 1))
            parent = parents[depth]
            item = QTreeWidgetItem(["", ""])
            item.setData(0, _PID, pid)
            item.setData(0, _ROLE, "node")
            item.setData(0, _NID, row.node_id)
            mark = "+ " if row.own else ""
            if row.role == "input":
                item.setText(0, f"{mark}⇤ {row.detail}")
                if row.kind:
                    item.setIcon(0, kind_icon(row.kind))
                item.setForeground(0, QBrush(T.INK_2))
                tip = (f"Page Input — reads {row.detail}" if row.kind else
                       f"Page Input — {row.detail}: pick what it reads on its card")
            elif row.role == "output":
                name = row.name or "(unnamed)"
                item.setText(0, f"{mark}⇥ {name}    {row.detail}")
                item.setFont(0, mono)
                col = _kind_color(kind)
                item.setForeground(0, QBrush(T.mix(col, T.INK, 0.35)))
                for c in (0, _SWITCH):
                    item.setBackground(c, QBrush(T.mix(T.PANEL, col, 0.28)))
                tip = (f"Page Output — the variable “{name}” of this page; {row.detail}. A "
                       f"later page's Page Input lists it as “<this page> · {name}”."
                       if row.name else "Page Output with no name — give it one on its card, "
                                        "or no page can read it")
            elif row.role == "source":
                item.setText(0, f"{mark}● {row.title}  ·  {row.detail}")
                item.setForeground(0, QBrush(T.INK_2))
                tip = f"{row.title} — where this page's data starts ({row.detail})"
            else:
                item.setText(0, f"{mark}{row.title}")
                item.setForeground(0, QBrush(T.INK_2))
                tip = row.title
            if row.own:
                tip += "\n+ this page's own node (a modified linked page)"
            if row.muted:
                off = QFont(item.font(0))
                off.setStrikeOut(True)
                off.setItalic(True)
                item.setFont(0, off)
                item.setForeground(0, QBrush(T.MUTED))
                tip += "\nswitched OFF — its input is passed straight through"
            if row.overridden:
                tip += ("\non this page only — the master has it "
                        + ("on" if row.muted else "off"))
            # the switch: only a node that may be switched off (or one that already is)
            if not row.why_not or row.muted:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(_SWITCH, Qt.Unchecked if row.muted else Qt.Checked)
                item.setToolTip(_SWITCH, "on — untick to switch it off (its input passes "
                                "straight through)" if not row.muted else
                                "off — tick to switch it back on")
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                if row.role in ("node", "source"):
                    item.setToolTip(_SWITCH, f"always on: {row.why_not}")
                    tip += f"\ncannot be switched off: {row.why_not}"
            item.setToolTip(0, tip + "\nclick to show it on the canvas")
            parent.addChild(item)
            parent.setExpanded(True)
            del parents[depth + 1:]
            parents.append(item)

    @staticmethod
    def _tip(text: str, master: str, reads: Sequence[str], publishes: Sequence[str],
             own: str = "") -> str:
        lines = [text]
        if master:
            lines.append(f"linked to “{master}” — its graph, values of its own"
                         + (f"; {own}" if own else ""))
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

    def node_items(self, page_id: str) -> Dict[str, QTreeWidgetItem]:
        """``{node id: row}`` of one page's outline."""
        out: Dict[str, QTreeWidgetItem] = {}
        page = next((p for p in self.page_items() if p.data(0, _PID) == page_id), None)
        stack = [page] if page is not None else []
        while stack:
            it = stack.pop()
            for j in range(it.childCount()):
                ch = it.child(j)
                if ch.data(0, _ROLE) == "node":
                    out[str(ch.data(0, _NID))] = ch
                stack.append(ch)
        return out

    # ── interaction ──────────────────────────────────────────────────────────
    def _fold(self, item: QTreeWidgetItem, folded: bool) -> None:
        if self._filling or item.data(0, _ROLE) != "page":
            return
        pid = str(item.data(0, _PID))
        (self._folded.add if folded else self._folded.discard)(pid)

    def _on_click(self, item: QTreeWidgetItem, col: int) -> None:
        pid = item.data(0, _PID)
        if not pid or (col == _SWITCH and item.data(0, _ROLE) == "node"):
            return                           # the switch is handled by _on_changed
        nid = item.data(0, _NID)
        # after the click handler returns: showing a page refreshes this very tree
        if nid:
            QTimer.singleShot(0, lambda p=str(pid), n=str(nid): self.node_requested.emit(p, n))
        else:
            QTimer.singleShot(0, lambda p=str(pid): self.page_requested.emit(p))

    def _on_changed(self, item: QTreeWidgetItem, col: int) -> None:
        if self._filling or col != _SWITCH or item.data(0, _ROLE) != "node":
            return
        pid, nid = str(item.data(0, _PID)), str(item.data(0, _NID))
        off = item.checkState(_SWITCH) != Qt.Checked
        # deferred: switching rebuilds this very tree
        QTimer.singleShot(0, lambda p=pid, n=nid, o=off: self.mute_requested.emit(p, n, o))

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
