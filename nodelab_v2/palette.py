"""Registry-driven node palette (G2) — searchable, grouped by pipeline STAGE and functional
ROLE (the taxonomy in ``codemap/node_roles.json``, read through :mod:`nodegraph.roles`);
double-click adds at the view center, or drag a row onto the canvas (mime
``application/x-nd2studios-op``, accepted by :class:`~nodelab_v2.scene.GraphView`).

Each node row carries coloured DOTS: on the left, what flows IN — one dot per attribute
domain the node reads off its Dataset input (voxel, label, point, …), plus one per value
socket type (float, int, string, …); on the right, what flows OUT — one per domain the node
adds, plus one per value output. The colours are the canvas's own socket and domain-rail
colours (:data:`nodelab_v2.theme.SOCKET`, :data:`nodelab_v2.theme.DOMAIN`), so a dot here
means the same thing as a wire there. The bottom third of the panel is the OVERVIEW: click
any row and it explains the node — what it is for, its data contract, sockets with types and
units, footprint, modes — read live from the registry, never from prose that could drift.
(2026-10-02: previously grouped by the registry ``category``, with no dots and no overview.)
"""
from __future__ import annotations

import html
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QSplitter, QStyledItemDelegate, QTextBrowser,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QPushButton,
)

from nodegraph import roles as R
from nodegraph.domains import Domain
from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T
from nodelab_v2.scene import visible_specs

#: the role whose nodes hand data between pages (``codemap/node_roles.json``) — on a
#: typed page it leads the tree as the "Pages" band (V4.00 step 11)
PAGE_ROLE = "page_boundary"
#: the pinned band's key in the tree (not a stage of the taxonomy)
PAGES_BAND = "pages"

#: item-data slots
_OP = Qt.UserRole            # a node row: its op_key
_KIND = Qt.UserRole + 1      # "stage" | "role" | "node"
_KEY = Qt.UserRole + 2       # a stage/role row: its key

#: dot geometry (device pixels; the icon is rendered at 2x for crisp HiDPI)
_DOT = 9
_GAP = 3
_SCALE = 2


def _squash(text: Optional[str]) -> str:
    return " ".join(str(text or "").split())


# ── the dots ──────────────────────────────────────────────────────────────────

def _dataset_in_colors(spec) -> List[Tuple[QColor, str]]:
    """What flows IN: one dot per domain the node reads, else the plain dataset green when
    it takes a Dataset but requires nothing of it; then one per value-socket type."""
    out: List[Tuple[QColor, str]] = []
    state = spec.default_state()
    has_ds = any(s.type is SocketType.DATASET for s in spec.inputs)
    reads = sorted(spec.reads_domains, key=lambda d: d.value)
    if has_ds and reads:
        for d in reads:
            out.append((T.domain_qcolor(d), f"reads {d.value}"))
    elif has_ds:
        out.append((T.SOCKET[SocketType.DATASET], "dataset in (no domain required)"))
    seen = set()
    for s in spec.active_inputs(state):
        if s.type is SocketType.DATASET or s.type in seen:
            continue
        seen.add(s.type)
        out.append((T.SOCKET[s.type], f"{s.type.value} parameter"))
    return out


def _dataset_out_colors(spec) -> List[Tuple[QColor, str]]:
    """What flows OUT: one dot per domain the node adds, else the plain dataset green when it
    passes a Dataset through; then one per value output type."""
    out: List[Tuple[QColor, str]] = []
    has_ds = any(s.type is SocketType.DATASET for s in spec.outputs)
    adds = sorted(spec.adds_domains, key=lambda d: d.value)
    if has_ds and adds:
        for d in adds:
            out.append((T.domain_qcolor(d), f"adds {d.value}"))
    elif has_ds:
        out.append((T.SOCKET[SocketType.DATASET], "dataset out (passes its input's layers)"))
    seen = set()
    for s in spec.outputs:
        if s.type is SocketType.DATASET or s.type in seen:
            continue
        seen.add(s.type)
        out.append((T.SOCKET[s.type], f"{s.type.value} output"))
    return out


def _dots_icon(colors: Sequence[Tuple[QColor, str]], align_right: bool = False) -> QIcon:
    """Render ``colors`` as a row of dots. A fixed width per icon column keeps rows aligned
    whatever the count (up to six dots; beyond that the last is a '+')."""
    n = max(1, min(len(colors), 6))
    w = (6 * _DOT + 5 * _GAP) * _SCALE
    h = (_DOT + 4) * _SCALE
    pm = QPixmap(w, h)
    pm.setDevicePixelRatio(_SCALE)
    pm.fill(Qt.transparent)
    if not colors:
        return QIcon(pm)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    total = n * _DOT + (n - 1) * _GAP
    x = (6 * _DOT + 5 * _GAP) - total if align_right else 0
    for i, (col, _why) in enumerate(colors[:6]):
        p.setPen(QPen(T.alpha(T.INK, 60), 1.0))
        p.setBrush(col)
        p.drawEllipse(x + i * (_DOT + _GAP), 2, _DOT, _DOT)
    if len(colors) > 6:
        p.setPen(T.MUTED)
        p.drawText(x + 5 * (_DOT + _GAP), 2 + _DOT - 1, "+")
    p.end()
    return QIcon(pm)


# ── the tree ──────────────────────────────────────────────────────────────────

class _BandDelegate(QStyledItemDelegate):
    """Paint the stage/role rows' background band.

    The band is the item's own ``BackgroundRole`` brush, but it has to be painted here
    rather than left to the view: the panel's stylesheet has a ``QTreeWidget::item`` rule
    (padding, radius), and once such a rule exists Qt's stylesheet style draws the item
    panel itself and ignores the model's background brush — so ``setBackground`` alone
    produced no band at all. Filling the row rect before the default paint puts the band
    under the text whatever the stylesheet does; the hover/selection rules still draw on
    top of it, which is what you want.
    """

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: D401
        kind = index.data(_KIND)
        if kind in ("stage", "role"):
            brush = index.data(Qt.BackgroundRole)
            if brush is not None:
                painter.save()
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setPen(Qt.NoPen)
                painter.setBrush(brush)
                r = option.rect.adjusted(0, 1, 0, -1)
                painter.drawRoundedRect(r, 4.0, 4.0)
                painter.restore()
        super().paint(painter, option, index)


class _PaletteTree(QTreeWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setItemDelegate(_BandDelegate(self))
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setIndentation(12)
        self.setColumnCount(3)
        self.setIconSize(QSize(6 * _DOT + 5 * _GAP, _DOT + 4))
        hdr = self.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.setRootIsDecorated(False)

    def startDrag(self, _actions) -> None:
        it = self.currentItem()
        op = it.data(0, _OP) if it is not None else None
        if not op:
            return
        mime = QMimeData()
        mime.setData("application/x-nd2studios-op", op.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class PalettePanel(QWidget):
    """The dockable palette. ``on_add(op_key)`` fires on double-click/Enter."""

    #: the ⟳ button beside the search box was pressed — re-read the node catalog from disk.
    #: The panel only asks; the window owns the reloader, the runner (which must be idle) and
    #: the canvas that has to be relaid out afterwards.
    refresh_requested = Signal()
    #: *What does this node do?* under the Overview was pressed: ``op_key``. The window
    #: opens the live demo (one window per op type, :mod:`nodelab_v2.demo_window`).
    demo_requested = Signal(str)

    def __init__(self, on_add: Callable[[str], None]) -> None:
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)   # a subclass must ask for its QSS fill
        self._on_add = on_add
        #: the kind of the page being edited (V4.00 step 5): the palette offers its nodes
        self._page_kind: Optional[str] = None
        self._current_op: Optional[str] = None
        self.restyle()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nodes…  (drag onto the canvas)")
        self._refresh = QToolButton()
        self._refresh.setText("⟳")
        self._refresh.setCursor(Qt.PointingHandCursor)
        self._refresh.setFixedSize(26, 26)
        self._refresh.setToolTip(
            "Re-read the node list from disk.\n\n"
            "Picks up a node whose .py you ADDED while NodeLab was open, drops one whose "
            "file you deleted, and reloads any that changed — so the list here matches the "
            "files in nodegraph/catalog/.\n\n"
            "Only nodes whose code actually changed recompute; everything else keeps its "
            "cached results.")
        self._refresh.clicked.connect(self.refresh_requested)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(self._search, 1)
        top.addWidget(self._refresh)
        # which nodes this is: the active page's kind decides (V4.00 step 5)
        self._kind_chip = QLabel("All nodes")
        self._kind_chip.setObjectName("kindChip")
        self._kind_chip.setToolTip(
            "The palette offers the nodes of the active page's kind — an Input page the "
            "loaders and organisers, a Refinement page image preparation and segmentation, "
            "and so on. A Free page offers every node.")
        self._tree = _PaletteTree()
        # the OVERVIEW: the bottom third of the panel, a scrolling rich-text card
        self._overview = QTextBrowser()
        self._overview.setOpenExternalLinks(False)
        self._overview.setOpenLinks(False)
        self._overview.setFrameStyle(0)
        self._split = QSplitter(Qt.Vertical)
        self._split.addWidget(self._tree)
        self._split.addWidget(self._overview)
        self._split.setStretchFactor(0, 2)
        self._split.setStretchFactor(1, 1)
        self._split.setChildrenCollapsible(False)
        self._split.setSizes([600, 300])
        lay.addWidget(self._kind_chip)
        lay.addLayout(top)
        lay.addWidget(self._split, 1)
        # the demo (2026-10-07): the Overview says what a node is; this shows it, on a
        # synthetic image with live sliders — before the node is even on the canvas
        self._demo_btn = QPushButton("What does this node do?")
        self._demo_btn.setCursor(Qt.PointingHandCursor)
        self._demo_btn.setEnabled(False)
        self._demo_btn.setToolTip(
            "Open a window that runs the selected node on a synthetic image, with a slider "
            "for every parameter and a before / after to compare. A node that does not "
            "transform pixels shows its key features and how to use it.")
        self._demo_btn.clicked.connect(
            lambda: self._current_op and self.demo_requested.emit(self._current_op))
        lay.addWidget(self._demo_btn)
        self._search.textChanged.connect(self.refill)
        self._tree.itemDoubleClicked.connect(self._add_current)
        self._tree.currentItemChanged.connect(self._on_current)
        self.refill("")
        self._show_legend()

    def reload_catalog(self) -> None:
        """Rebuild the tree from the registry, keeping the user's search text.

        The palette is built once from ``NODES``, so a live node reload
        (:mod:`nodegraph.hotreload`) that added, removed, renamed or recategorized a node
        type leaves it showing the catalog the window opened with. The roles file is
        re-read too, so a role written for a new node appears without a restart."""
        R.reload()
        self.refill(self._search.text())

    def set_page_kind(self, kind: Optional[str], label: str = "") -> None:
        """Offer the nodes of a page of ``kind`` (``None``/``free``: every node), and say
        so on the chip above the search."""
        kind = kind or None
        free = kind is None or kind == R.FREE_PAGE
        self._kind_chip.setText("All nodes" if free else f"{label or kind} nodes")
        if kind == self._page_kind:
            return
        self._page_kind = kind
        self.refill(self._search.text())

    def page_kind(self) -> Optional[str]:
        return self._page_kind

    def focus_search(self) -> None:
        """Select the search box (the welcome card's 'Browse nodes' lands here)."""
        self._search.setFocus(Qt.OtherFocusReason)
        self._search.selectAll()

    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QTreeWidget {{ background:{T.PANEL.name()}; border:0; outline:0; }}
            QTreeWidget::item {{ padding:3px 2px; border-radius:5px; }}
            QTreeWidget::item:selected {{ background:{T.ACCENT_DIM.name()};
                color:{T.INK.name()}; }}
            QTreeWidget::item:hover {{ background:{T.PANEL_HI.name()}; }}
            QTextBrowser {{ background:{T.PANEL_HI.name()}; color:{T.INK.name()};
                border:1px solid {T.BORDER.name()}; border-radius:6px; padding:6px; }}
            QSplitter::handle {{ background:{T.BORDER.name()}; height:3px; }}
            QToolButton {{ color:{T.MUTED.name()}; background:transparent;
                border:1px solid {T.BORDER.name()}; border-radius:5px; font-size:14px; }}
            QToolButton:hover {{ color:{T.INK.name()}; background:{T.PANEL_HI.name()}; }}
        """ + T.controls_qss())
        if getattr(self, "_tree", None) is not None:
            self.refill(self._search.text())      # dots are rendered in theme colours
            if self._current_op:
                self._show_node(self._current_op)
            else:
                self._show_legend()

    # ── filling ───────────────────────────────────────────────────────────────

    def refill(self, text: str = "") -> None:
        t = (text or "").lower()
        self._tree.clear()
        specs = {s.op_key: s for s in visible_specs(self._page_kind)
                 if not t or t in s.label.lower() or t in s.op_key.lower()}
        # stage -> role -> [spec], in the taxonomy's own order; unclassified ops last
        buckets: Dict[str, Dict[str, List]] = {}
        for op, spec in specs.items():
            rk, sk = R.role_of(op)
            buckets.setdefault(sk, {}).setdefault(rk, []).append(spec)
        # V4.00 step 11: on a typed page the page boundary — Page Input / Page Output — is
        # the first thing to reach for, so it leads the tree as its own "Pages" band; a Free
        # page keeps it inside Control & present like any other role
        pinned: List = []
        if self._page_kind and self._page_kind != R.FREE_PAGE:
            for sk in list(buckets):
                pinned += buckets[sk].pop(PAGE_ROLE, None) or []
                if not buckets[sk]:
                    del buckets[sk]
        if pinned:
            head = self._stage_head(
                "Pages", PAGES_BAND,
                "Hand data from page to page: a Page Output names what this page produces; "
                "a Page Input reads a named Output of an earlier page.")
            self._add_role_rows(head, PAGE_ROLE, pinned)
            head.setExpanded(True)
        order = [sk for sk, _ in R.stages()] + [R.OTHER_STAGE]
        for sk in order:
            roles = buckets.get(sk)
            if not roles:
                continue
            smeta = R.stage_meta(sk)
            head = self._stage_head(str(smeta.get("label", sk)), sk,
                                    _squash(smeta.get("description")))
            role_order = [rk for rk, _ in R.roles_in(sk)] + [R.OTHER_ROLE]
            for rk in role_order:
                group = roles.get(rk)
                if not group:
                    continue
                self._add_role_rows(head, rk, group)
            head.setExpanded(True)

    def _stage_head(self, label: str, key: str, tip: str) -> QTreeWidgetItem:
        """A stage BAND: its text sits in column 0 and spans the row, so it starts at the
        panel's left edge instead of after the dots column, and it gets a filled background
        so the stages read as sections at a glance. (The first cut put the label in the
        middle column in small muted caps, which left it both indented and the faintest
        thing on the panel.)"""
        head = QTreeWidgetItem([label, "", ""])
        head.setFlags(Qt.ItemIsEnabled)
        head.setData(0, _KIND, "stage")
        head.setData(0, _KEY, key)
        head.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
        head.setBackground(0, QBrush(T.ACCENT_DIM))
        head.setForeground(0, QBrush(T.INK))
        head.setToolTip(0, tip)
        f = head.font(0)
        f.setBold(True)
        head.setFont(0, f)
        head.setSizeHint(0, QSize(0, 24))
        self._tree.addTopLevelItem(head)
        head.setFirstColumnSpanned(True)      # only takes effect once it is in the tree
        return head

    def _add_role_rows(self, head: QTreeWidgetItem, rk: str, group: List) -> None:
        """A role is a sub-heading under its stage: spanned and left-justified like the
        stage, one indent step in, with a lighter tint so stage > role > node reads as
        depth; its nodes follow, by label."""
        rmeta = R.role_meta(rk)
        rrow = QTreeWidgetItem([str(rmeta.get("label", rk)), "", ""])
        rrow.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        rrow.setData(0, _KIND, "role")
        rrow.setData(0, _KEY, rk)
        rrow.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)
        rrow.setBackground(0, QBrush(T.mix(T.PANEL, T.ACCENT_DIM, 0.45)))
        rrow.setForeground(0, QBrush(T.ACCENT))
        rrow.setToolTip(0, _squash(rmeta.get("description")))
        head.addChild(rrow)
        rrow.setFirstColumnSpanned(True)
        for spec in sorted(group, key=lambda s: s.label):
            row = QTreeWidgetItem(["", spec.label, ""])
            row.setData(0, _OP, spec.op_key)
            row.setData(0, _KIND, "node")
            ins, outs = _dataset_in_colors(spec), _dataset_out_colors(spec)
            row.setIcon(0, _dots_icon(ins, align_right=True))
            row.setIcon(2, _dots_icon(outs))
            row.setToolTip(0, "IN: " + ", ".join(w for _, w in ins))
            row.setToolTip(2, "OUT: " + ", ".join(w for _, w in outs))
            row.setToolTip(1, f"{spec.op_key}\n{_squash(spec.description)}")
            rrow.addChild(row)
        rrow.setExpanded(True)

    # ── selection → overview ──────────────────────────────────────────────────

    def _on_current(self, item: Optional[QTreeWidgetItem], _prev=None) -> None:
        if item is None:
            return
        kind = item.data(0, _KIND)
        if kind == "node":
            self._show_node(item.data(0, _OP))
        elif kind == "role":
            self._show_role(item.data(0, _KEY))
        elif kind == "stage":
            self._show_stage(item.data(0, _KEY))

    def _add_current(self, item: Optional[QTreeWidgetItem] = None, _col: int = 0) -> None:
        it = item or self._tree.currentItem()
        op = it.data(0, _OP) if it is not None else None
        if op:
            self._on_add(op)

    @property
    def current_op(self) -> Optional[str]:
        """The op the overview is showing, or ``None`` (a role/stage/legend)."""
        return self._current_op

    def overview_html(self) -> str:
        return self._overview.toHtml()

    # ── overview content ──────────────────────────────────────────────────────

    @staticmethod
    def _dot(col: QColor) -> str:
        return (f'<span style="color:{col.name()}; font-size:13px;">&#9679;</span>')

    def _css(self) -> str:
        return (f"<style>body{{color:{T.INK.name()}; font-size:11px;}} "
                f"h3{{margin:0 0 2px 0; font-size:13px;}} "
                f".k{{color:{T.MUTED.name()};}} .op{{color:{T.MUTED.name()}; "
                f"font-family:monospace; font-size:10px;}} "
                f"table{{border-collapse:collapse;}} td{{padding:1px 6px 1px 0; "
                f"vertical-align:top;}} .sec{{color:{T.ACCENT.name()}; font-weight:bold; "
                f"margin-top:6px;}}</style>")

    def _show_legend(self) -> None:
        self._current_op = None
        self._demo_btn.setEnabled(False)
        dom = " ".join(f"{self._dot(T.domain_qcolor(d))}&nbsp;{d.value}" for d in Domain)
        typ = " ".join(f"{self._dot(T.SOCKET[t])}&nbsp;{t.value}" for t in SocketType)
        self._overview.setHtml(
            self._css() + "<h3>Nodes</h3>"
            "<div class='k'>Grouped by pipeline stage, then by what the node does. Click a "
            "node for its overview; double-click or drag to add it.</div>"
            "<div class='sec'>Dots</div>"
            "<div><b>Left</b> = what flows in: the attribute domains the node reads from "
            "its Dataset, then its parameter types. <b>Right</b> = what flows out: the "
            "domains it adds, then value outputs.</div>"
            f"<div style='margin-top:4px'>{dom}</div>"
            f"<div style='margin-top:2px'>{typ}</div>")

    def _show_stage(self, sk: str) -> None:
        self._current_op = None
        self._demo_btn.setEnabled(False)
        meta = R.stage_meta(sk)
        roles = "".join(
            f"<li><b>{html.escape(str(r.get('label', rk)))}</b> — "
            f"{html.escape(_squash(r.get('description')))}</li>"
            for rk, r in R.roles_in(sk))
        self._overview.setHtml(
            self._css() + f"<h3>{html.escape(str(meta.get('label', sk)))}</h3>"
            f"<div>{html.escape(_squash(meta.get('description')))}</div>"
            f"<div class='sec'>Roles</div><ul>{roles}</ul>")

    def _show_role(self, rk: str) -> None:
        self._current_op = None
        self._demo_btn.setEnabled(False)
        meta = R.role_meta(rk)
        smeta = R.stage_meta(meta.get("stage", R.OTHER_STAGE))
        ops = "".join(f"<li>{html.escape(op)}</li>" for op in sorted(meta.get("ops", ())))
        self._overview.setHtml(
            self._css() + f"<h3>{html.escape(str(meta.get('label', rk)))}</h3>"
            f"<div class='k'>{html.escape(str(smeta.get('label', '')))}</div>"
            f"<div>{html.escape(_squash(meta.get('description')))}</div>"
            f"<div class='sec'>Nodes</div><ul>{ops}</ul>")

    def _show_node(self, op: str) -> None:
        """The node's overview — built by :func:`nodelab_v2.demo_window.overview_html`, the
        same builder the demo window's guide uses (one source, so the two never drift): what
        it is for, its stage and role, the curated key features, every socket with its type,
        unit, default and the pick gesture it offers, its modes, footprint and how it works
        (the compute's docstring, else the module's)."""
        from nodegraph.registry import NODES
        from nodelab_v2 import demo_recipes as DR
        from nodelab_v2.demo_window import overview_html
        spec = NODES.get(op)
        if spec is None:
            self._show_legend()
            return
        self._current_op = op
        self._demo_btn.setEnabled(True)
        try:
            features = DR.recipe_for(op).features
        except Exception:                                   # pragma: no cover - defensive
            features = ()
        self._overview.setHtml(overview_html(op, features=features))


__all__ = ["PalettePanel"]
