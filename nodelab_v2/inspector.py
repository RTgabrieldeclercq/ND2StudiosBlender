"""Properties inspector (NodeLab v2) — the click-to-edit mirror of a selected node.

Binds to a :class:`~nodelab_v2.node_item.NodeItem` and builds an editable form from its
:class:`~nodegraph.registry.NodeSpec`: the 2D/3D switch, the resolved footprint, and one
row per active parameter (spin boxes, mode dropdowns, unit labels). Metadata-derived
params (``derive``) show an **auto / pinned** toggle — the sticky ``__locked__`` override:
auto shows the metadata-derived value; editing or pinning fixes it; unpinning reverts.
Edits write back to the node's ``params``/``locked`` and relayout the card live.

A STRING param declaring ``SocketSpec.path_kind`` is a filesystem path and gets a
**Browse…** button opening the file/folder dialog that socket asks for (V2.15) — the button
follows the declaration, never the socket's name, so a newly declared path param is
browsable without touching this module.

**Interactive picks (V2.16).** A param declaring ``SocketSpec.pick_kind`` grows a **Pick**
button *below* its editor, which arms a gesture on the viewer
(:mod:`nodelab_v2.picker`) — drag a radius, click a cell for its area, draw an ROI. The
button sits below rather than beside because the value row is already label + editor + unit
+ ƒ-auto wide, and a fifth item would either clip or squeeze the number. ``choices`` renders
a closed dropdown and ``vocab`` a tick list, replacing free text where the legal values are
a fixed set. All three follow the DECLARATION, never a socket name, so annotating a socket
is the whole job.

**Per-option hover (V2.21).** Every dropdown explains each of its OPTIONS, not just the
control: a Mode row and its label carry ``mode_hover_text`` (what the dropdown selects plus
one line per choice), and each combo item / tick box carries its own option prose on
``Qt.ToolTipRole``. Both halves are needed — the row answers "what is this control" before
the list is open, the per-item tips answer "which of these do I want" while the user is
moving down it, which is when the question is actually being asked.
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional, Sequence

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMenu, QPushButton, QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T
from nodelab_v2.node_item import (
    NUM_MAX_FLOAT, NUM_MAX_INT, NodeItem, float_editor_precision, mode_hover_text,
    option_hover_text, socket_hover_text, value_step,
)
from nodegraph.iterate import (
    ITERATE_OP, MAX_VARIABLES as ITERATE_MAX_VARIABLES, SWEEP_KEY, TYPE_TEXT,
    candidate_targets, plan as iterate_plan,
)
from nodelab_v2.document import is_driver_edge as _is_driver
from nodelab_v2.ops import (DOCK_OP, MOVIE_OP, PAGE_INPUT_OP, PAGE_SOURCE_KEY,
                            PRECISION_UNSET, bake_record)
from nodelab_v2.picker import PICK_GLYPH, PICK_HELP, request_for
from nodelab_v2 import readiness as RD
from nodegraph.registry import NODES

#: a drawing node's tool settings (presentation params), edited inside its Draw section
#: in workflow order rather than in the general Parameters list
_DRAW_TOOL_PARAMS = ("tool", "op", "brush_px")

_UNIT = {"um": "µm", "um_axial": "µm↕", "um2": "µm²", "um3": "µm³",
         "nm": "nm", "s": "s", "px": "px"}


#: How an Iterate target combo carries ``(node_id, socket)`` on one item. A STRING rather
#: than the tuple itself, because the item data is round-tripped through a QVariant and a
#: string is the one form whose equality ``QComboBox.findData`` is guaranteed to reproduce.
#: The separator is a control character no socket or node id can contain.
_TOKEN_SEP = "\x1f"
#: the sentinel meaning "leave the hand-wired targets alone" (see ``_MANY_TARGETS``)
_KEEP_TARGET = "keep"


def _target_token(node_id: str, socket: str) -> str:
    return f"{node_id}{_TOKEN_SEP}{socket}"


def _clean_path(s: str) -> str:
    """Normalize a hand-entered path: trim whitespace and surrounding quotes (a
    Windows "Copy as path" paste wraps the path in double quotes)."""
    return (s or "").strip().strip('"').strip("'").strip()


def _set_item_tips(combo: QComboBox, options: Sequence[str], docs) -> None:
    """Hang each option's documentation on its own combo ITEM (V2.21).

    ``Qt.ToolTipRole`` on the item, not one tooltip on the widget: the popup list is where
    the user is comparing the options, and a per-item tip follows the highlighted row while
    they arrow through it. Items are addressed by TEXT rather than by enumeration index
    because a couple of these combos carry an extra leading entry (``_choice_box`` re-inserts
    a saved-but-unregistered value at 0), which would silently shift a positional mapping and
    label every option with its neighbour's prose."""
    docs = docs or {}
    if not docs:
        return
    for opt in options:
        i = combo.findText(opt)
        if i >= 0:
            combo.setItemData(i, option_hover_text(opt, docs.get(opt, "")), Qt.ToolTipRole)


class _NoWheelCombo(QComboBox):
    """A combo that ignores the wheel unless it has focus.

    Not cosmetic — a fix for a live data-loss bug. The inspector is a fixed-height
    QScrollArea, so scrolling to a lower row drags the pointer across every combo above
    it, and PySide6 delivers each notch to the combo under the cursor: measured, ONE
    wheel event over an editable combo fires ``textActivated`` and ``accept()``s the
    event. The user silently rewrites (and pins) a param they never touched, and the
    panel does not scroll because the event was eaten. Ignoring it defers to the scroll
    area. A layer name is not an ordinal the wheel should walk, so even a focused combo
    gains nothing from wheel-stepping — but StrongFocus + the focus test keeps the
    conventional behaviour available after a deliberate click."""

    def wheelEvent(self, e):                      # noqa: N802 - Qt naming
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


def _derived_value(node: NodeItem, s) -> float:
    """The LIVE auto value for a numeric socket — the metadata-derived value from the node's
    propagated envelope (G8), or what the loaded model was trained with (V2.23). Both arrive
    through ``NodeItem.resolved``, which fixes the precedence in one place."""
    v = node.resolved(s)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(s.default) if s.default is not None else 0.0


def _pin_value(node: NodeItem, s):
    """The value to write into ``params`` when the user pins an auto socket — the value that
    was on display, in the socket's OWN type.

    Type matters here, unlike in ``_derived_value`` (whose only consumer is ``setValue`` on a
    spin box). Pinning is a document edit that serializes and then reaches a compute, and a
    BOOL socket pinned as ``1.0`` would arrive at ``bool(ctx.params.get("upsample"))`` as a
    float that happens to be truthy — correct by accident, and wrong in the saved graph, which
    is meant to record a flag. BOOL gained an auto state in V2.23 (ZS-DeconvNet's ``upsample``
    is adopted from the checkpoint), so this is newly reachable.
    """
    v = node.resolved(s)
    if s.type is SocketType.BOOL:
        return bool(v)
    if s.type is SocketType.INT:
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return int(s.default or 0)
    if s.type is SocketType.STRING:
        return "" if v is None else str(v)
    return _derived_value(node, s)


def _auto_source(node: NodeItem, s) -> str:
    """Which source an auto socket is currently taking its value from — for the pin button's
    tooltip, so "ƒ auto" says *auto from what*.

    Worth distinguishing: a metadata-derived value moves when the incoming file changes,
    while a checkpoint-derived one moves when the user points at a different model. A user
    deciding whether to pin needs to know which of those they are looking at.
    """
    return "model" if s.name in node.trained() else "metadata"


def _h(col) -> str:
    return col.name()


def _human_bytes(n) -> str:
    """A byte count at a readable scale. A dock can be anything from a few hundred KB to
    hundreds of GB, so a fixed unit is wrong at one end or the other — "0 MB" for a small
    bake reads as "nothing was written"."""
    try:
        n = float(n or 0)
    except (TypeError, ValueError):
        return "?"
    for unit, step in (("B", 1024.0), ("KB", 1024.0), ("MB", 1024.0), ("GB", 1024.0)):
        if n < step:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= step
    return f"{n:,.1f} TB"


def _inspector_qss() -> str:
    return f"""
QWidget#inspRoot {{ background:{_h(T.PANEL)}; }}
QScrollArea {{ border:0; background:{_h(T.PANEL)}; }}
QLabel {{ color:{_h(T.INK)}; }}
QLabel[role="eyebrow"] {{ color:{_h(T.INK_2)}; font-weight:600; letter-spacing:1.4px; }}
QLabel[role="muted"] {{ color:{_h(T.MUTED)}; }}
QLabel[role="op"] {{ color:{_h(T.MUTED)}; font-family:{T.MONO}; }}
QFrame[role="sep"] {{ background:{_h(T.BORDER)}; max-height:1px; min-height:1px; border:0; }}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
  background:{_h(T.BODY)}; color:{_h(T.INK)}; border:1px solid {_h(T.BORDER)};
  border-radius:6px; padding:4px 7px; font-family:{T.MONO}; min-height:18px;
}}
QDoubleSpinBox:disabled, QSpinBox:disabled {{ color:{_h(T.MUTED)}; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
  border-color:{_h(T.ACCENT)}; }}
QComboBox::drop-down {{ border:0; width:16px; }}
QToolButton {{
  background:{_h(T.BODY)}; color:{_h(T.MUTED)}; border:1px solid {_h(T.BORDER)};
  border-radius:6px; padding:4px 7px; font-size:10px; font-weight:600;
}}
QToolButton:checked {{ color:{_h(T.MUTED)}; }}
QToolButton[state="auto"] {{ color:{_h(T.ACCENT)}; border-color:{_h(T.ACCENT_DIM)}; }}
/* Opaque MIXES, not alpha() — `_h` renders a QColor as #RRGGBB and silently drops the
   alpha channel, so a translucent accent came out as a solid accent block with
   accent-coloured text on it, i.e. an unreadable button. */
QToolButton[role="pick"] {{ color:{_h(T.ACCENT)}; border-color:{_h(T.ACCENT_DIM)};
  background:{_h(T.mix(T.PANEL, T.ACCENT, 0.13))}; text-align:left; padding:4px 9px; }}
QToolButton[role="pick"]:hover {{ border-color:{_h(T.ACCENT)};
  background:{_h(T.mix(T.PANEL, T.ACCENT, 0.26))}; }}
/* a *Ready to run* suggestion: the node to add, in the problem's own red */
QToolButton[role="add"] {{ color:{_h(T.INK)}; border-color:{_h(T.mix(T.PANEL, T.ERROR, 0.5))};
  background:{_h(T.mix(T.PANEL, T.ERROR, 0.18))}; padding:3px 8px; }}
QToolButton[role="add"]:hover {{ border-color:{_h(T.ERROR)};
  background:{_h(T.mix(T.PANEL, T.ERROR, 0.32))}; }}
QLabel[role="vocab"] {{ color:{_h(T.MUTED)}; }}
""" + T.controls_qss()


class SwitchWidget(QWidget):
    """The 2D/3D on-off switch as a QWidget (amber 2D / cyan 3D)."""

    toggled = Signal(str)

    def __init__(self, dim: str = "2D") -> None:
        super().__init__()
        self.dim = dim
        self.setFixedSize(84, 24)
        self.setCursor(Qt.PointingHandCursor)

    def set_dim(self, dim: str) -> None:
        if dim != self.dim:
            self.dim = dim
            self.update()

    def mousePressEvent(self, e) -> None:
        self.dim = "3D" if self.dim == "2D" else "2D"
        self.update()
        self.toggled.emit(self.dim)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        on = self.dim == "3D"
        lblw, tw, th, knob = 18, 40, 22, 18
        f = QFont(T.MONO, 8); f.setBold(True); p.setFont(f)
        p.setPen(T.INK if not on else T.MUTED)
        p.drawText(0, 0, lblw, 24, Qt.AlignCenter, "2D")
        p.setPen(T.INK if on else T.MUTED)
        p.drawText(self.width() - lblw, 0, lblw, 24, Qt.AlignCenter, "3D")
        tx = lblw + 4
        col = T.ACCENT if on else T.DIM2D
        p.setPen(QPen(col, 1)); p.setBrush(col)
        p.drawRoundedRect(tx, 1, tw, th, th / 2, th / 2)
        kx = tx + (tw - knob - 1) if on else tx + 1
        p.setPen(Qt.NoPen); p.setBrush(T.ACCENT_INK if on else T.DIM2D_INK)
        p.drawEllipse(kx, 2, knob, knob)


#: the entry of a closed name list that sets an OPTIONAL column socket back to blank
_NONE_ENTRY = "(none)"


class InspectorPanel(QScrollArea):
    #: A param row's Pick button was pressed — carries a
    #: :class:`~nodelab_v2.picker.PickRequest`. The panel only *asks*: the window owns the
    #: viewer (and the document, hence the calibration), so it arms the session and writes
    #: the result back. Keeping the panel out of that means a pick from the node card goes
    #: through exactly the same route as one from here.
    pick_requested = Signal(object)
    #: a Dock node's button was pressed: ``(node_id, action)`` with action one of
    #: ``bake`` / ``bake_scoped`` / ``undock`` / ``redock`` / ``reveal`` / ``discard``.
    #: Same division of labour as :attr:`pick_requested` — the panel only asks; the
    #: window owns the runner and the document, so it is the one that can actually run a
    #: bake, record it and release the memory afterwards.
    dock_action = Signal(str, str)
    #: an Iterate node's button was pressed: ``(node_id, action)`` with action one of
    #: ``sweep`` (mint and run every iteration) / ``stop_sweep`` (back to the cheap
    #: single-clone form). Same division of labour as the two above.
    iterate_action = Signal(str, str)
    #: an Export Movie node's Preview button was pressed: ``(node_id, "preview")``.
    #: Same division of labour as the two above — the panel only asks. The window
    #: owns the runner, and a preview needs the node's INPUT payload, which only a
    #: completed pull can supply.
    movie_action = Signal(str, str)
    #: the ⟳ button beside the node title was pressed: ``op_key``. Re-read that node type's
    #: module from disk. Same division of labour as the signals above — the panel asks, the
    #: window owns the reloader, the runner (which must be idle) and the canvas that has to
    #: be relaid out afterwards.
    reload_requested = Signal(str)
    #: the ? beside the node title was pressed: ``op_key``. Open *What does this node do?*
    #: — the live demo of that node type on a phantom (2026-10-07). The panel asks; the
    #: window owns the one-window-per-op cache and the dialogs' worker threads.
    demo_requested = Signal(str)
    #: a *Ready to run* suggestion was taken: ``(node_id, op_key, wire_to)`` — add a node
    #: of ``op_key`` and wire its output into this node's ``wire_to`` input. Same division
    #: of labour as the signals above: the panel asks, the window owns the document and
    #: the canvas, so it places the node, re-routes the wires and relays out the scene.
    add_requested = Signal(str, str, str)
    #: a suggestion that adds a node DOWNSTREAM (V4.00 step 11): ``(node_id, op_key,
    #: from_socket)`` — the window places it right of this node and wires this node's
    #: ``from_socket`` into it ("+ Page Output" after a terminal node or a loader)
    append_requested = Signal(str, str, str)
    #: *Go to <page>* under a Page Input's Source (V4.00 step 11): show that page
    page_requested = Signal(str)
    #: a drawing control in a Draw Regions (or ROI Mask) panel: ``(node_id, what, value)``
    #: with ``what`` one of ``arm`` / ``apply`` / ``cancel`` / ``undo`` / ``clear`` /
    #: ``close`` / ``sync`` (the tool, operation or brush setting changed — re-read them).
    #: The panel hosts the controls (2026-10-02: nothing about the drawing is configured on
    #: the image); the window owns the viewer's gesture and relays.
    draw_control = Signal(str, str, object)
    #: a node that WANTS a region (Subtract Background's `Background sample`) asked for
    #: one: ``(node_id, socket)``. The window drops a Draw Regions node in line on the
    #: region input, switches to it, and comes back here when its drawing is applied.
    region_requested = Signal(str, str)
    #: the Linked page banner (V4.00 step 6): ``(action, node_id)`` with action ``master``
    #: (show the master page, this node selected — the graph is edited there) or ``unique``
    #: (turn this page into a page of its own). The panel asks; the window owns the pages.
    linked_action = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self._problems: list = []          # readiness problems of the shown node
        #: (node_id, name) of every row marked as overridden on a linked page (V4.00 step 6)
        self._override_rows: list = []
        self._problem_sockets: dict = {}   # socket name -> Problem (what is painted red)
        self._draw_armed: Optional[str] = None   # node id whose drawing is armed, if any
        self._draw_readout = ""
        self._draw_widgets: dict = {}      # live widgets of the Draw section, if shown
        # wide enough for the richest param row (label + spinbox + unit + ƒ-auto button)
        # PLUS the vertical scrollbar, so the right edge (the ƒ-auto buttons) never clips.
        self.setFixedWidth(376)
        self.setStyleSheet(_inspector_qss())
        self._node: Optional[NodeItem] = None
        self._auto_boxes = []              # (node, socket, box) — live ƒmd refresh (G8)
        self._last_trained_note = ""
        self._last_dir = ""                # last browsed folder (seeds the next dialog)
        # carries a refused target choice across the rebuild the choice triggers, so the
        # reason is shown in the panel instead of being swallowed by the combo's signal
        self._iterate_error = ""
        self._host = QWidget(); self._host.setObjectName("inspRoot")
        self.setWidget(self._host)
        self._v = QVBoxLayout(self._host)
        self._v.setContentsMargins(0, 0, 0, 0)
        self._v.setSpacing(0)
        self._rebuild()

    # ── binding ─────────────────────────────────────────────────────────────
    def set_node(self, node: Optional[NodeItem]) -> None:
        if self._node is not None:
            try:
                self._node.changed.disconnect(self._on_changed)
            except (RuntimeError, TypeError):
                pass
        self._node = node
        if node is not None:
            node.changed.connect(self._on_changed)
        self._rebuild()

    def _on_changed(self, *_a) -> None:
        self._rebuild()

    def rebuild(self) -> None:
        """Rebuild the panel for the node it is already showing.

        For a live node reload (:mod:`nodegraph.hotreload`): the rows are built from the
        node's :class:`NodeSpec`, so a reload that changed a param's unit, range, choices,
        pick gesture or hover text leaves the panel describing the previous version — and
        the panel is where a param's documentation is actually read."""
        self._rebuild()

    # ── build ───────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        """Re-apply the stylesheet from the current theme tokens (G9) + rebuild so
        the per-widget inline colors (chips/dots) re-read too."""
        self.setStyleSheet(_inspector_qss())
        self._rebuild()

    def refresh_derived(self) -> None:
        """Re-seed the disabled (auto) value boxes from the live envelopes (G8) —
        called on every document change; never touches an editable/focused box."""
        for node, s, box in self._auto_boxes:
            try:
                box.blockSignals(True)
                box.setValue(_derived_value(node, s))
            except RuntimeError:              # widget already deleted mid-rebuild
                continue
            finally:
                try:
                    box.blockSignals(False)
                except RuntimeError:
                    pass

    def _clear(self) -> None:
        self._auto_boxes = []
        while self._v.count():
            item = self._v.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)     # detach now so it can't paint before deleteLater runs
                w.deleteLater()

    def _reload_button(self, node: NodeItem) -> QWidget:
        """The ⟳ button beside the node's title: re-read THIS node's ``.py`` from disk.

        The panel is where a node's parameters are read and edited, so it is where you notice
        that the code on disk and the node in front of you disagree — and until now there was
        nothing here to act on it. Menu-level *Reload node code* exists, but it reloads what it
        thinks changed; this reloads **this node, unconditionally**, which is the right answer
        when the panel already looks wrong. Auto-reload can be off, a watcher notification can
        be missed (an editor that saves by replacing the file drops the OS watch), and neither
        case leaves the user anything to press.

        Disabled, with the reason in its tooltip, for a node the reloader cannot act on: a
        group instance, an op whose definition is missing, or one of the GUI-layer ops
        (``io.load``, ``view.viewer``, ``io.dock``) that are deliberately not reloadable."""
        btn = QToolButton()
        btn.setText("⟳")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(24, 24)
        btn.setStyleSheet(
            f"QToolButton {{ color:{_h(T.MUTED)}; background:transparent; border:1px solid "
            f"{_h(T.BORDER)}; border-radius:5px; font-size:14px; }}"
            f"QToolButton:hover {{ color:{_h(T.INK)}; background:{_h(T.PANEL_HI)}; }}"
            f"QToolButton:disabled {{ color:{_h(T.alpha(T.MUTED, 90))}; "
            f"border-color:{_h(T.alpha(T.BORDER, 90))}; }}")
        mod = ""
        try:
            from nodegraph.hotreload import module_of_op
            mod = module_of_op(node.op_key)
        except Exception:                      # noqa: BLE001 — a button must never break the panel
            mod = ""
        if mod:
            btn.setToolTip(
                f"Reload this node's code from disk.\n\n"
                f"Re-reads {mod.replace('.', '/')}.py, re-registers the node, and rebuilds "
                f"this panel and the card — so a parameter you added or renamed, or a change "
                f"to the compute, takes effect without restarting.\n\n"
                f"Only this node recomputes on the next pull; every other node keeps its "
                f"cached results. Anything that shares code with it is reloaded too.")
            btn.clicked.connect(lambda: self.reload_requested.emit(node.op_key))
        else:
            btn.setEnabled(False)
            btn.setToolTip(
                f"“{node.op_key}” is not a reloadable node type.\n\n"
                f"Its definition is not one of the per-node modules under nodegraph/catalog/ "
                f"— source and viewer taps are part of the GUI layer, and a group instance is "
                f"a subgraph rather than a node type. Restart to pick up changes to those.")
        return btn

    def _demo_button(self, node: NodeItem) -> QWidget:
        """The ? beside the node's title: open *What does this node do?* for this node TYPE.

        A window (:mod:`nodelab_v2.demo_window`) that runs the node on a synthetic image
        with a slider for every parameter and a before / after to compare — or, for a node
        that does not transform pixels, its key features and how to use it. It is a
        sandbox: nothing it does reaches this node or the canvas. Sits beside ⟳ because the
        header is where a user asks what a node is before they ask what its code says."""
        btn = QToolButton()
        btn.setText("?")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(24, 24)
        btn.setStyleSheet(
            f"QToolButton {{ color:{_h(T.MUTED)}; background:transparent; border:1px solid "
            f"{_h(T.BORDER)}; border-radius:5px; font-size:13px; font-weight:bold; }}"
            f"QToolButton:hover {{ color:{_h(T.INK)}; background:{_h(T.PANEL_HI)}; }}"
            f"QToolButton:disabled {{ color:{_h(T.alpha(T.MUTED, 90))}; "
            f"border-color:{_h(T.alpha(T.BORDER, 90))}; }}")
        btn.setToolTip(
            "What does this node do?\n\n"
            "Opens a window that runs this node type on a synthetic image, with a slider "
            "for every parameter and a before / after you can wipe, checker or difference. "
            "A node that does not transform pixels shows its key features and how to use "
            "it instead.\n\nA sandbox: nothing there changes this node.")
        btn.clicked.connect(lambda: self.demo_requested.emit(node.op_key))
        return btn

    def _eyebrow(self, text: str, color=None) -> QLabel:
        lab = QLabel(text.upper()); lab.setProperty("role", "eyebrow")
        if color is not None:
            lab.setStyleSheet(f"color:{_h(color)};")
        f = lab.font(); f.setPointSize(8); lab.setFont(f)
        return lab

    def _sep(self) -> QFrame:
        fr = QFrame(); fr.setProperty("role", "sep"); return fr

    def _section(self, title: str, extra: str = "") -> QWidget:
        sec = QWidget()
        lay = QVBoxLayout(sec); lay.setContentsMargins(16, 13, 16, 13); lay.setSpacing(8)
        head = self._eyebrow(title)
        if extra:
            row = QHBoxLayout(); row.addWidget(head); row.addStretch(1)
            m = QLabel(extra); m.setProperty("role", "muted")
            fm = m.font(); fm.setPointSize(9); m.setFont(fm)
            row.addWidget(m)
            lay.addLayout(row)
        else:
            lay.addWidget(head)
        sec._lay = lay  # type: ignore[attr-defined]
        return sec

    def _rebuild(self) -> None:
        self._clear()
        node = self._node
        if node is None or node.spec is None:
            if node is not None and getattr(node, "_is_group", False):
                msg = (f"Group “{node._group_name}”.\n\nA reusable subgraph collapsed into "
                       f"one node. Select it and use Graph → Ungroup (Ctrl+Shift+G) to edit "
                       f"its contents, then re-group. Pulling it runs the whole subgraph.")
            elif node is not None and node.spec is None:
                msg = (f"Unrecognized node type “{node.op_key}”.\n\nThis op is not in "
                       f"the registry — the file may come from a newer build or a "
                       f"plugin that isn't loaded.")
            else:
                msg = "Select a node to edit its parameters."
            ph = QLabel(msg)
            ph.setProperty("role", "muted"); ph.setAlignment(Qt.AlignCenter)
            ph.setWordWrap(True); ph.setContentsMargins(30, 40, 30, 40)
            self._v.addWidget(ph); self._v.addStretch(1)
            return
        spec = node.spec
        cat = T.category_color(spec.category)

        # header
        hd = QWidget()
        hl = QVBoxLayout(hd); hl.setContentsMargins(16, 14, 16, 12); hl.setSpacing(4)
        hl.addWidget(self._eyebrow(spec.category, cat))
        row = QHBoxLayout()
        title = QLabel(spec.label); tf = title.font(); tf.setPointSize(13); tf.setBold(True)
        title.setFont(tf); row.addWidget(title); row.addStretch(1)
        row.addWidget(self._demo_button(node))
        row.addWidget(self._reload_button(node))
        if spec.has_dim_lever():
            sw = SwitchWidget(node.dim)
            sw.toggled.connect(node.set_dim)
            row.addWidget(sw)
        hl.addLayout(row)
        op = QLabel(spec.op_key); op.setProperty("role", "op")
        of = op.font(); of.setPointSize(9); op.setFont(of); hl.addWidget(op)
        self._v.addWidget(hd)
        self._v.addWidget(self._sep())
        self._override_rows = []
        if not getattr(node.doc, "editable_topology", True):
            # a LINKED page (V4.00 step 6): whose graph this is, how far this page departs
            # from it, and the two ways out
            self._v.addWidget(self._linked_section(node))
            self._v.addWidget(self._sep())

        # readiness (2026-10-02) — can it run as wired? Each problem names the input it is
        # about, which the rows below paint red, and offers the nodes that would fix it.
        # Computed once per rebuild and consulted by `_param_row` / `_conn_label`.
        try:
            self._problems = RD.problems(node.doc, node.node_id)
        except Exception as exc:              # noqa: BLE001 — never let a check hide the panel
            self._problems = [RD.Problem("validation", None,
                                         f"readiness check failed: {exc}")]
        self._problem_sockets = RD.socket_problems(self._problems)
        self._v.addWidget(self._readiness_section(node, self._problems))
        self._v.addWidget(self._sep())

        # footprint — the read COST (a fact), plus the statistics POPULATION (a control, V2.27)
        fp = self._section("Footprint")
        gname = node.granularity()
        chip = QLabel(gname.replace("_", " ").upper() if gname else "—")
        gcol = T.gran_color(gname)
        chip.setStyleSheet(
            f"color:{_h(gcol)}; border:1px solid {_h(T.alpha(gcol,120))};"
            f"background:{_h(T.alpha(gcol,36))}; border-radius:4px; padding:3px 7px;"
            f"font-family:{T.MONO}; font-weight:700;")
        cf = chip.font(); cf.setPointSize(8); chip.setFont(cf)
        frow = QHBoxLayout(); frow.addWidget(chip); frow.addStretch(1)
        fp._lay.addLayout(frow)  # type: ignore[attr-defined]
        # The population lives HERE rather than in the Mode section below, so the panel matches
        # the card: one place that says how much is read and what decides it. `_mode_row` brings
        # its own hover prose, per-item option tips and commit path, so there is no second
        # editor to keep in step — and `_scope_mode` is what keeps it out of Mode (line ~518).
        scope_mode = node._scope_mode() if hasattr(node, "_scope_mode") else None
        if scope_mode is not None:
            fp._lay.addWidget(self._mode_row(node, scope_mode))  # type: ignore[attr-defined]
        # Name the resolver. The old text told every node without a lever it was
        # "Dimension-agnostic", which is false for the four whose footprint is resolved by a
        # named `footprint_mode` (`bounds`, `method`, `output`, `scope`) — and for a source,
        # whose footprint is not declared at all.
        if scope_mode is not None:
            ntxt = (f"`{scope_mode.name}` chooses the statistics population above, and the "
                    f"footprint follows from it. Populations fold into the memo key, so each "
                    f"one caches separately.")
        elif spec.has_dim_lever():
            ntxt = ("The 2D / 3D switch resolves this footprint and the active sockets; "
                    "it folds into the memo key, so 2D and 3D cache separately.")
        elif isinstance(spec.granularity, Mapping) and spec.footprint_mode:
            ntxt = (f"The `{spec.footprint_mode}` mode below resolves this footprint — this "
                    f"node has no 2D / 3D switch.")
        elif not gname:
            ntxt = "Undeclared — this node is a source or a sink rather than a compute."
        else:
            ntxt = "Dimension-agnostic — no 2D/3D switch."
        self._last_footprint_note = ntxt      # asserted by the phase-5 GUI gate
        note = QLabel(ntxt)
        note.setProperty("role", "muted"); note.setWordWrap(True)
        nf = note.font(); nf.setPointSize(9); note.setFont(nf)
        fp._lay.addWidget(note)  # type: ignore[attr-defined]
        self._v.addWidget(fp)
        self._v.addWidget(self._sep())

        # parameters
        params = [s for s in node._active_inputs() if s.type is not SocketType.DATASET]
        # a drawing node's tool settings belong to its Draw section (the workflow: Draw,
        # pick a shape, add or cut, Apply), not to the general parameter list
        if self._shapes_socket(node) is not None:
            params = [s for s in params if s.name not in _DRAW_TOOL_PARAMS]
        if params:
            sec = self._section("Parameters", str(len(params)))
            # What the LOADED MODEL says, at the top of the params it speaks for (V2.23b).
            # Reported as "loading in the model does not change any of the parameters": for a
            # published checkpoint with no sidecar, and for a path that is not a model folder,
            # the resolver correctly finds nothing — and a panel that then shows plain defaults
            # with no badge and no reason is indistinguishable from a broken feature. The two
            # states are now told apart in words, above the values they explain.
            tnote = spec.note_for_trained(node.rec.params, node.state())
            self._last_trained_note = tnote   # asserted by the phase-5 GUI gate
            if tnote:
                lab = QLabel(("✓  " if node.trained() else "•  ") + tnote)
                lab.setWordWrap(True)
                lab.setProperty("role", "muted")
                lf = lab.font(); lf.setPointSize(9); lab.setFont(lf)
                # Tinted only when something WAS adopted, so the eye can tell the two apart
                # without reading: the values above are the checkpoint's, or they are not.
                if node.trained():
                    lab.setStyleSheet(f"color:{_h(T.OK if hasattr(T, 'OK') else T.ACCENT)};")
                sec._lay.addWidget(lab)     # type: ignore[attr-defined]
            for s in params:
                if s.name == "groups" and self._split_axis(node):
                    # a split card's grouping (2026-10-07): the outputs with checkboxes
                    # and the Group selected / Ungroup buttons, not a bare text box
                    sec._lay.addWidget(self._split_grouping_box(node, s))  # type: ignore[attr-defined]
                    continue
                sec._lay.addWidget(self._param_row(node, s))  # type: ignore[attr-defined]
            self._v.addWidget(sec)
            self._v.addWidget(self._sep())

        # the Draw section (2026-10-02) — for a node that IS a drawing (Draw Regions, ROI
        # Mask): the gesture's controls live here, not on the viewer
        self._draw_widgets = {}
        shapes_sock = self._shapes_socket(node)
        if shapes_sock is not None:
            self._v.addWidget(self._draw_section(node, shapes_sock))
            self._v.addWidget(self._sep())

        # in-body modes (non-dim), mode-gated like the sockets above: a Mode the selected
        # method never reads is hidden, not shown and ignored (V2.12 `ModeSpec.available_in`)
        # ...and NOT the population mode, which the Footprint section above already edits;
        # listing it twice would give one value two combos that have to agree.
        modes = [m for m in spec.active_modes(node.state())
                 if not m.is_dim_lever and m is not scope_mode]
        if modes:
            sec = self._section("Mode")
            for m in modes:
                sec._lay.addWidget(self._mode_row(node, m))  # type: ignore[attr-defined]
            self._v.addWidget(sec)
            self._v.addWidget(self._sep())

        # dock (V2.18) — the bake controls, above Connections so the primary action on
        # this node type is the first thing under its parameters rather than the last.
        if spec.op_key == DOCK_OP:
            self._v.addWidget(self._dock_section(node))
            self._v.addWidget(self._sep())

        # iterate (V2.19) — what this sweep resolves to, what it costs, and the results
        if spec.op_key == ITERATE_OP:
            self._v.addWidget(self._iterate_section(node))
            self._v.addWidget(self._sep())

        # export movie — look at the movie before writing it
        if spec.op_key == MOVIE_OP:
            self._v.addWidget(self._movie_section(node))
            self._v.addWidget(self._sep())

        # overlay (2026-09-30) — the frame pins the Viewer wrote, removable one by one
        if spec.op_key == "view.overlay":
            self._v.addWidget(self._overlay_section(node))
            self._v.addWidget(self._sep())

        # connections
        sec = self._section("Connections")
        for s in spec.inputs:
            if s.type is SocketType.DATASET:
                prob = self._problem_sockets.get(s.name)
                sec._lay.addWidget(self._conn_label(  # type: ignore[attr-defined]
                    f"in · {s.name}", T.SOCKET[s.type],
                    problem=prob.message if prob is not None else ""))
        for s in spec.outputs:
            sec._lay.addWidget(self._conn_label(f"out · {s.name}", T.SOCKET[s.type]))  # type: ignore[attr-defined]
        self._v.addWidget(sec)
        self._v.addStretch(1)

    # ── readiness (2026-10-02) ──────────────────────────────────────────────
    def _readiness_section(self, node: NodeItem, probs: list) -> QWidget:
        """*Ready to run*: a green line when nothing is missing; otherwise one block per
        problem — red for what blocks a run, accent for a hint — each with one-click fixes:
        *+ <node>* buttons that add a node (upstream, or downstream for an ``append``
        suggestion) and plain buttons that set a value (``set_param``: bind a Page Input,
        name a Page Output; V4.00 step 11).

        The section sits directly under the title because it is the first question about
        a node that is not running — before its parameters. The same problems colour the
        param rows and connection labels below, so the eye lands on the input in question
        without reading; this block is where the words are."""
        errors = [p for p in probs if getattr(p, "severity", "error") == "error"]
        sec = self._section("Ready to run", "" if errors else "✓")
        if not errors:
            lab = QLabel("✓  every input this node needs is present")
            lab.setProperty("role", "muted"); lab.setWordWrap(True)
            lf = lab.font(); lf.setPointSize(9); lab.setFont(lf)
            lab.setStyleSheet(f"color:{_h(T.WIRE)};")
            sec._lay.addWidget(lab)       # type: ignore[attr-defined]
        for p in probs:
            is_hint = getattr(p, "severity", "error") != "error"
            tone = T.ACCENT if is_hint else T.ERROR
            block = QFrame()
            block.setObjectName("readyProblem")
            block.setStyleSheet(
                f"QFrame#readyProblem {{ border-left:3px solid {_h(tone)}; "
                f"background:{_h(T.mix(T.PANEL, tone, 0.10))}; border-radius:4px; }}")
            bl = QVBoxLayout(block); bl.setContentsMargins(9, 6, 8, 6); bl.setSpacing(5)
            msg = QLabel(("·  " if is_hint else "⚠  ") + p.message)
            msg.setWordWrap(True)
            msg.setStyleSheet(f"color:{_h(tone)}; background:transparent;")
            mf = msg.font(); mf.setPointSize(9); msg.setFont(mf)
            bl.addWidget(msg)
            if p.suggestions:
                actions = {getattr(s, "action", "add") for s in p.suggestions}
                hint = QLabel("add to the graph, wired in:" if "add" in actions else
                              "add after this node:" if "append" in actions else "fix:")
                hint.setProperty("role", "muted")
                hint.setStyleSheet(f"color:{_h(T.MUTED)}; background:transparent;")
                hf = hint.font(); hf.setPointSize(8); hint.setFont(hf)
                bl.addWidget(hint)
                row = QHBoxLayout(); row.setSpacing(6)
                for sgg in p.suggestions:
                    action = getattr(sgg, "action", "add")
                    btn = QToolButton()
                    btn.setText(sgg.label if action == "set_param" else f"+ {sgg.label}")
                    btn.setProperty("role", "add")
                    btn.setCursor(Qt.PointingHandCursor)
                    if action == "set_param":
                        btn.setToolTip(f"{sgg.reason}\n\nSets `{sgg.param}` to "
                                       f"`{sgg.value}`.")
                        btn.clicked.connect(
                            lambda _c, n=node, pn=sgg.param, v=sgg.value:
                            self._apply_fix(n, pn, v))
                    elif action == "append":
                        btn.setToolTip(f"{sgg.label} ({sgg.op_key})\n{sgg.reason}\n\n"
                                       f"Adds the node after this one, fed from "
                                       f"`{sgg.wire_to}`.")
                        btn.clicked.connect(
                            lambda _c, nid=node.node_id, op=sgg.op_key, w=sgg.wire_to:
                            self.append_requested.emit(nid, op, w))
                    else:
                        btn.setToolTip(f"{sgg.label} ({sgg.op_key})\n{sgg.reason}\n\n"
                                       f"Adds the node and wires its output into "
                                       f"`{sgg.wire_to}`.")
                        btn.clicked.connect(
                            lambda _c, nid=node.node_id, op=sgg.op_key, w=sgg.wire_to:
                            self.add_requested.emit(nid, op, w))
                    row.addWidget(btn)
                row.addStretch(1)
                bl.addLayout(row)
            sec._lay.addWidget(block)     # type: ignore[attr-defined]
        return sec

    def _apply_fix(self, node: NodeItem, param: str, value) -> None:
        """A ``set_param`` suggestion was taken: write the value and rebuild on the next turn
        (the button that asked is inside the section being torn down)."""
        self._set_param(node, param, value)
        QTimer.singleShot(0, self._rebuild)

    # ── iterate section (V2.19) ─────────────────────────────────────────────
    def _iterate_section(self, node: NodeItem) -> QWidget:
        """The Iterate node's own panel: what the sweep resolves to, what it will cost,
        the Run-sweep button, and the recorded results.

        The resolved iteration list is computed HERE, synchronously, from the same
        :func:`nodegraph.iterate.plan` the rewrite uses — so the panel can never describe a
        different sweep from the one that will run, and a misconfiguration is reported in
        the panel (with the fix named) instead of surfacing as a failed pull."""
        doc = node.doc
        sec = self._section("Iterate")
        lay = sec._lay  # type: ignore[attr-defined]

        def blurb(text: str, color=None, italic: bool = False) -> None:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setProperty("role", "muted")
            f = lbl.font(); f.setPointSize(9); f.setItalic(italic); lbl.setFont(f)
            if color is not None:
                lbl.setStyleSheet(f"color:{_h(color)};")
            lay.addWidget(lbl)

        try:
            graph = doc.to_graph()
        except Exception:                     # noqa: BLE001 — a mid-edit graph, not a bug
            blurb("Wire the end of the chain you are tuning back into Collect.",
                  italic=True)
            return sec

        # The target picker comes FIRST and outside the plan, because the state it exists
        # for is the one the plan cannot describe: a card that drives nothing yet.
        lay.addWidget(self._iterate_targets(node, graph))
        if self._iterate_error:
            blurb(self._iterate_error, T.ERROR)
            self._iterate_error = ""
        if not doc.iterate_segment(node.node_id)[1]:
            # The picker has already said to wire 'To', in the same words the plan would
            # refuse in. One complaint per mistake.
            return sec

        try:
            plan = iterate_plan(graph, node.node_id, envs=doc.envs)
        except (ValueError, KeyError) as exc:
            # Every refusal nodegraph.iterate raises already names its fix, so showing it
            # verbatim here is better than paraphrasing — and it arrives while the user is
            # editing rather than when they press play.
            blurb(str(exc), T.ERROR)
            return sec
        except Exception:                     # noqa: BLE001 — a mid-edit graph, not a bug
            blurb("Wire the end of the series into 'To', then pick a parameter above.",
                  italic=True)
            return sec

        n = plan.n
        minted = len(plan.minted)
        head = QLabel(f"{n} iteration{'s' if n != 1 else ''}"
                      + (f" · {minted} run" if minted != n else ""))
        hf = head.font(); hf.setPointSize(10); hf.setBold(True); head.setFont(hf)
        lay.addWidget(head)
        drives = ", ".join(f"{t.node_id}.{t.name}"
                           for v in plan.variables for t in v.targets)
        blurb(f"drives {drives}")
        blurb(f"{len(plan.cone)} node{'s' if len(plan.cone) != 1 else ''} re-run per "
              f"iteration; the result comes out of {plan.end}, so anything reading that "
              f"node already gets it.")
        if minted < n:
            blurb("Only the kept iteration is computed, so a finished graph costs one run "
                  "instead of %d. Press Run sweep to compute them all and compare." % n)

        # cost: the honest warning is about SCOPE, which is a runner setting, not a param
        if not getattr(self, "_solo_frame", False):
            blurb("Troubleshooting scope is OFF, so each iteration runs the whole series — "
                  f"about {n}× the usual cost. Turn it on and pick a frame to tune against.",
                  T.DIM2D)

        row = QHBoxLayout()
        sweep = QPushButton("Run sweep")
        sweep.setCursor(Qt.PointingHandCursor)
        sweep.setToolTip("Compute EVERY iteration (not just the kept one) so they can be "
                         "compared on the Viewer's iteration strip.")
        sweep.clicked.connect(
            lambda: self.iterate_action.emit(node.node_id, "sweep"))
        stop = QPushButton("Stop sweeping")
        stop.setCursor(Qt.PointingHandCursor)
        stop.setToolTip("Go back to computing only the kept iteration. Nothing is thrown "
                        "away — the others stay cached.")
        stop.clicked.connect(
            lambda: self.iterate_action.emit(node.node_id, "stop_sweep"))
        row.addWidget(sweep); row.addWidget(stop); row.addStretch(1)
        lay.addLayout(row)

        lay.addWidget(self._sweep_table(plan, node.params.get(SWEEP_KEY)))
        return sec

    # ── the target picker (V2.22) ───────────────────────────────────────────
    #: what the first entry of a slot's target menu says when the slot drives nothing.
    _NO_TARGET = "— nothing —"
    #: the entry a slot shows when it was WIRED to several params by hand. Selecting
    #: anything else collapses the slot onto that one target, so the menu says so before
    #: it happens rather than quietly dropping wires the user drew.
    _MANY_TARGETS = "several targets (wired by hand)"

    def _iterate_targets(self, node: NodeItem, graph) -> QWidget:
        """One "iterate on" dropdown per variable slot, scraped from the chain (V2.22).

        The menu is :func:`nodegraph.iterate.candidate_targets` — every param and dropdown
        of every node upstream of Collect that this card could legally drive, in chain
        order, each one already put through the rewrite's own ``_check_target``. So the
        offer and the refusal cannot disagree: anything listed here connects, and anything
        the rewrite would reject never appears.

        It does not replace the wire, it BUILDS one. Choosing an entry writes exactly the
        driver edge a drag from the variable output would have written
        (:meth:`~nodelab_v2.document.GraphDocument.set_iterate_target`), which is why the
        canvas, the "driven by" note on the target's own editor, save/load and the rewrite
        need to know nothing about this control."""
        doc = node.doc
        nid = node.node_id
        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 4)
        v.setSpacing(3)

        def note(text: str) -> None:
            lbl = QLabel(text); lbl.setWordWrap(True); lbl.setProperty("role", "muted")
            f = lbl.font(); f.setPointSize(9); f.setItalic(True); lbl.setFont(f)
            v.addWidget(lbl)

        starts, ends = doc.iterate_segment(nid)
        if not ends:
            note("Wire the LAST node of the series you want iterated into 'To'. That node "
                 "is where the result comes out — everything already reading it keeps "
                 "working — and this menu then lists every parameter in the series.")
            return host
        note(("segment: %s → %s" % (" + ".join(starts), ", ".join(ends))) if starts else
             ("segment: ends at %s (starts wherever a parameter is driven — wire 'From' to "
              "pin it)" % ", ".join(ends)))
        options = candidate_targets(graph, nid)
        if not options:
            note("Nothing inside the segment has a parameter this card can drive.")
            return host

        state = node.state()
        try:
            n_vars = int(state.get("variables", "1") or 1)
        except (TypeError, ValueError):
            n_vars = 1
        n_vars = max(1, min(ITERATE_MAX_VARIABLES, n_vars))
        for k in range(n_vars):
            v.addWidget(self._iterate_target_row(node, k, options))
        # One SPARE row past the live slots, so "iterate a second parameter as well" is one
        # click rather than "first find Variables, raise it, then come back up here".
        # Choosing in it raises `variables` (set_iterate_target does), and the rebuild turns
        # it into an ordinary slot. Not offered under feedback, which searches exactly one.
        if (n_vars < ITERATE_MAX_VARIABLES
                and str(state.get("mode") or "sweep") != "feedback"):
            v.addWidget(self._iterate_target_row(node, n_vars, options, spare=True))
        return host

    def _iterate_target_row(self, node: NodeItem, slot: int, options,
                            *, spare: bool = False) -> QWidget:
        doc = node.doc
        nid = node.node_id
        wires = doc.iterate_targets(nid, slot)
        current = (wires[0][2], wires[0][3]) if len(wires) == 1 else None

        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lab = QLabel("+" if spare else f"V{slot}")
        lab.setFixedWidth(24)
        lf = lab.font(); lf.setPointSize(9); lf.setBold(True); lab.setFont(lf)
        if spare:
            lab.setProperty("role", "muted")
        lay.addWidget(lab)

        combo = _NoWheelCombo()
        combo.setToolTip(
            ("Iterate a SECOND parameter as well — picking one here raises Variables to "
             "%d and gives it its own value fields. Under 'grid' the run count multiplies."
             % (slot + 1)) if spare else
            ("Which parameter this variable iterates. The list is every parameter and "
             "dropdown in the chain feeding Collect that this card can legally drive — "
             "choosing one wires it, exactly as dragging the V%d output onto it would."
             % slot))
        combo.addItem("— add another parameter —" if spare else self._NO_TARGET, "")
        if len(wires) > 1:
            combo.addItem(self._MANY_TARGETS, _KEEP_TARGET)
        last_node = ""
        for opt in options:
            if opt.driver_id and (opt.driver_id, opt.driver_slot) != (nid, slot):
                continue                    # taken by another slot or another card
            if opt.target.node_id != last_node and combo.count() > 1:
                combo.insertSeparator(combo.count())
            last_node = opt.target.node_id
            text = opt.text + (" (dropdown)" if opt.target.is_mode else "")
            combo.addItem(text, _target_token(*opt.key))
            combo.setItemData(combo.count() - 1, self._target_tip(opt), Qt.ToolTipRole)
        # A hand-wired target the scrape no longer offers (its node's Mode gated the socket
        # away, something else was wired into it) must still show as the current selection —
        # a menu that silently reads "nothing" while a driver wire exists on the canvas is
        # the one thing this control must never do.
        token = _target_token(*current) if current is not None else ""
        if token and combo.findData(token) < 0:
            combo.insertItem(1, f"{current[0]}.{current[1]}", token)
        if len(wires) > 1:
            combo.setCurrentIndex(combo.findData(_KEEP_TARGET))
        elif token:
            combo.setCurrentIndex(max(0, combo.findData(token)))
        combo.currentIndexChanged.connect(
            lambda _i, c=combo, k=slot: self._commit_iterate_target(node, k, c))
        lay.addWidget(combo, 1)
        return row

    @staticmethod
    def _target_tip(opt) -> str:
        """One menu entry's hover: which node and socket it really is, and what picking it
        does to the slot's Type — that Mode is set for the user, so it is worth saying where
        the choice is made rather than leaving the change to be noticed later."""
        kind = ("a dropdown — picking it sets this variable's Type to text and sweeps the "
                "option names" if opt.target.is_mode else
                "a text parameter — picking it sets this variable's Type to text"
                if opt.kind == TYPE_TEXT else "a numeric parameter")
        return f"{opt.detail}\n{kind}"

    def _commit_iterate_target(self, node: NodeItem, slot: int, combo) -> None:
        """Apply a target choice, then rebuild — the slot's Type may have changed, which
        reconfigures the card's own sockets exactly as :meth:`_set_mode` does."""
        data = str(combo.currentData() or "")
        if data == _KEEP_TARGET:
            return
        target_node, _, target_socket = data.partition(_TOKEN_SEP)
        try:
            node.doc.set_iterate_target(node.node_id, slot, target_node, target_socket)
        except ValueError as exc:
            self._iterate_error = str(exc)
        QTimer.singleShot(0, self._rebuild)

    def _sweep_table(self, plan, recorded) -> QWidget:
        """The results table: one row per iteration, its values, and its metric once a
        sweep has actually been run. The value columns come from the live plan (always
        true), the metric column from the last recorded run (which may be older than the
        graph — said so, rather than hidden)."""
        by_iter = {}
        if isinstance(recorded, dict):
            for r in recorded.get("rows", []) or []:
                if isinstance(r, dict):
                    by_iter[int(r.get("iter", -1))] = r
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(1)
        heads = ["#"] + [v.label for v in plan.variables] + ["metric"]
        for c, text in enumerate(heads):
            h = QLabel(text)
            h.setProperty("role", "muted")
            hf = h.font(); hf.setPointSize(8); hf.setBold(True); h.setFont(hf)
            grid.addWidget(h, 0, c)
        for r, it in enumerate(plan.iterations, start=1):
            rec = by_iter.get(it.index)
            won = bool(rec and rec.get("won"))
            cells = [str(it.index)]
            for j in range(len(plan.variables)):
                v = it.values[j] if j < len(it.values) else None
                cells.append("—" if v is None else
                             (f"{v:g}" if isinstance(v, (int, float)) else str(v)))
            metric = rec.get("metric") if rec else None
            cells.append("—" if metric is None else f"{float(metric):g}")
            for c, text in enumerate(cells):
                cell = QLabel(text)
                cf = cell.font(); cf.setPointSize(9)
                cf.setFamily(T.MONO); cf.setBold(won); cell.setFont(cf)
                if not won:
                    cell.setProperty("role", "muted")
                grid.addWidget(cell, r, c)
        grid.setColumnStretch(len(heads) - 1, 1)
        return host

    def set_solo_frame(self, on: bool) -> None:
        """Tell the panel whether the troubleshooting scope is active, so an Iterate node
        can warn when a sweep is about to run over the whole series."""
        if bool(on) != getattr(self, "_solo_frame", False):
            self._solo_frame = bool(on)
            if self._node is not None and getattr(self._node, "op_key", "") == ITERATE_OP:
                self._rebuild()

    # ── dock section ────────────────────────────────────────────────────────
    #: what each dock status means, in the user's terms — the sentence under the buttons.
    _DOCK_BLURB = {
        "live": "Running the chain above normally. Hold it to freeze the result in memory "
                "instantly, or Bake it to write the result to disk.",
        "held": "Serving a frozen copy held IN MEMORY. Everything above is greyed out and "
                "is not being evaluated. Nothing was written, so this does not free memory "
                "and will be gone when you reopen the file — Bake it for either of those.",
        "released": "Set to held, but the held copy is gone (memory does not survive "
                    "reopening the file). Hold it again, or Bake it so it survives.",
        "docked": "Serving the baked checkpoint. Everything above is greyed out and is "
                  "not being evaluated or held in memory.",
        "stale": "Still serving the OLD result — nothing has changed behind your back. "
                 "Re-bake to pick up the edit, or un-dock to run the chain live.",
        "unbaked": "Set to docked, but there is nothing on disk to serve. Bake it, or "
                   "switch State back to live.",
    }

    def _overlay_section(self, node: NodeItem) -> QWidget:
        """The Overlay node's own panel: the frame PINS, each removable.

        Pins are made in the Viewer (its source strip's Pin T / Pin Z) and stored as JSON in
        ``t_pins`` / ``z_pins``, which is right for the record and hopeless to edit by hand —
        and there is no undo. So they are listed here as rows, one ✕ each. (The resolved rate
        and Play-all tick count live in the Viewer's source strip: they come from the stamped
        recipe, which the edit-time envelope this panel reads does not carry.)"""
        from nodegraph.placement import parse_pins
        rec = node.rec
        sec = self._section("Frame pins")
        lay = sec._lay  # type: ignore[attr-defined]

        def muted(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setProperty("role", "muted")
            lbl.setWordWrap(True)
            f = lbl.font(); f.setPointSize(9); lbl.setFont(f)
            return lbl

        any_pin = False
        for axis in ("t", "z"):
            name = f"{axis}_pins"
            try:
                rows = list(parse_pins(rec.params.get(name, ""), axis=axis))
            except ValueError as exc:
                lay.addWidget(muted(f"{axis.upper()} pins are malformed — {exc}"))
                continue
            if not rows:
                continue
            any_pin = True
            head = QHBoxLayout()
            head.addWidget(QLabel(f"{axis.upper()} pins"))
            head.addStretch(1)
            clear = QToolButton()
            clear.setText("clear")
            clear.setToolTip(f"Remove every {axis.upper()} pin — the pairing goes back to "
                             + ("the Time shift / Rate" if axis == "t" else "the Nudge Z"))
            clear.clicked.connect(
                lambda _c=False, n=name: self._set_pins(node, n, []))
            head.addWidget(clear)
            lay.addLayout(head)
            for r in rows:
                line = QHBoxLayout()
                anchor = "" if r[2] is None else (
                    "  · clock-anchored" if axis == "t" else f"  · {r[2]:.1f}→{r[3]:.1f} µm")
                line.addWidget(QLabel(f"primary {axis}={r[0]}  →  source {axis}={r[1]}"
                                      + anchor))
                line.addStretch(1)
                rm = QToolButton()
                rm.setText("✕")
                rm.setToolTip("Remove this pin")
                rm.clicked.connect(
                    lambda _c=False, n=name, a=int(r[0]), rs=tuple(rows):
                    self._set_pins(node, n, [x for x in rs if int(x[0]) != a]))
                line.addWidget(rm)
                lay.addLayout(line)
        if not any_pin:
            lay.addWidget(muted(
                "No pins. In the Viewer, step an overlaid source with its ◀ ▶ buttons until "
                "its frame matches the primary's, then press Pin T or Pin Z — the pairing "
                "runs through every pin from then on."))
        return sec

    def _set_pins(self, node: NodeItem, name: str, rows) -> None:
        """Write a pin list back — canonical, and through the ordinary edit path."""
        from nodegraph.placement import pins_json
        text = pins_json(list(rows))
        rec = node.rec
        if text:
            self._set_param(node, name, text)
        else:
            rec.params.pop(name, None)
            rec.set_locked(rec.locked - {name})
            node.doc.touch(node.node_id)
        QTimer.singleShot(0, self._rebuild)

    def _movie_section(self, node: NodeItem) -> QWidget:
        """The Export Movie node's own panel: what it plays, and the way into the editor.

        The Movie Editor itself is a bottom dock (it opens when this node is selected),
        because a timeline and a monitor do not fit in a 376 px column. This panel only says
        what the node will make and raises the dock; every movie-wide setting is already a
        socket above."""
        from nodegraph.catalog._shared.movie_timeline import try_normalize
        rec = node.rec
        sec = self._section("Movie")
        lay = sec._lay  # type: ignore[attr-defined]

        if str(rec.modes.get("sweep", "time")) == "timeline":
            spec, err = try_normalize(rec.params.get("timeline", "") or "")
            if spec is None:
                text = f"Timeline, not yet playable — {err}"
            else:
                segs = spec["segments"]
                loops = sum(1 for s in segs if s["kind"] == "loop")
                text = (f"Timeline: {len(segs)} segment{'s' if len(segs) != 1 else ''}"
                        + (f", {loops} loop{'s' if loops != 1 else ''}" if loops else "")
                        + ". Edit it in the Movie Editor.")
        else:
            text = (f"Plays one axis ({rec.modes.get('sweep', 'time')}) of the input with the "
                    f"settings above. The Movie Editor previews it, and can turn it into a "
                    f"timeline: grids, several sources, z sweeps between timepoints.")
        blurb = QLabel(text)
        blurb.setProperty("role", "muted")
        blurb.setWordWrap(True)
        bf = blurb.font(); bf.setPointSize(9); blurb.setFont(bf)
        lay.addWidget(blurb)

        btn = QPushButton("Open Movie Editor")
        btn.setToolTip("Raise the Movie Editor dock on this node: a monitor that plays the "
                       "frames it would export (nothing is written), a timeline of clips, "
                       "and each clip's panels, channels, LUTs and labels.")
        btn.clicked.connect(
            lambda _=False, nid=node.node_id: self.movie_action.emit(nid, "edit"))
        lay.addWidget(btn)
        return sec

    def _dock_section(self, node: NodeItem) -> QWidget:
        """The Dock node's own panel: what state it is in, what is on disk, and the one
        or two buttons that make sense right now."""
        doc = node.doc
        nid = node.node_id
        status, detail = doc.dock_status(nid)
        precision = node.state().get("precision", PRECISION_UNSET)
        store = doc.dock_store(nid) or doc.default_dock_store(nid)
        rec = bake_record(node.rec)

        sec = self._section("Dock", status.upper() if status else "")
        lay = sec._lay  # type: ignore[attr-defined]

        blurb = QLabel(detail or self._DOCK_BLURB.get(status, ""))
        blurb.setProperty("role", "muted"); blurb.setWordWrap(True)
        bf = blurb.font(); bf.setPointSize(9); blurb.setFont(bf)
        # amber for stale (nothing is wrong — the bake is just older than the graph),
        # red for unbaked (docked with nothing to serve, which IS a broken run).
        if status == "stale":
            blurb.setStyleSheet(f"color:{_h(T.DIM2D)};")
        elif status in ("unbaked", "released"):
            # `released` is red for the same reason `unbaked` is: the node is set to serve
            # something it cannot, so the run is broken until the user acts. Amber would
            # read as "older than the graph", which is a different and milder thing.
            blurb.setStyleSheet(f"color:{_h(T.ERROR)};")
        lay.addWidget(blurb)

        if rec:
            info = QLabel(
                f"baked {rec.get('at') or 'earlier'} · {rec.get('precision', '?')} · "
                f"{_human_bytes(rec.get('bytes', 0))} on disk")
            info.setProperty("role", "muted"); info.setWordWrap(True)
            f2 = info.font(); f2.setPointSize(9); info.setFont(f2)
            lay.addWidget(info)
        path = QLabel(store)
        path.setProperty("role", "op"); path.setWordWrap(True)
        pf = path.font(); pf.setPointSize(8); path.setFont(pf)
        path.setToolTip(store)
        lay.addWidget(path)

        # The precision gate is deliberate: there is no default, because the honest
        # choice depends on what the chain produced. Say so where the button is, rather
        # than letting the press fail with a dialog.
        # Not shown while `held`: Precision is gated to `docked` (it chooses a STORED dtype
        # and a hold stores nothing), so the control this sentence says is "above" would not
        # be on screen to pick.
        if precision == PRECISION_UNSET and status not in ("held", "released"):
            warn = QLabel("Pick a Precision above before baking — float32 is the usual "
                          "answer for a filter chain; float64 keeps every last digit at "
                          "4× the size; uint16 only suits data still in camera counts.")
            warn.setProperty("role", "muted"); warn.setWordWrap(True)
            wf = warn.font(); wf.setPointSize(9); warn.setFont(wf)
            lay.addWidget(warn)

        row = QHBoxLayout(); row.setSpacing(6)
        # Hold comes FIRST and is always enabled: it is the cheap, reversible, no-precision
        # action, so it should be the one under the cursor. Bake sits beside it as the
        # durable (and slower, and disk-costing) alternative.
        held_now = status == "held"
        hold = QPushButton("Re-hold" if status in ("held", "released") else "Hold")
        hold.setToolTip(
            "Freeze what the chain above last produced IN MEMORY and stop evaluating it. "
            "Effectively instant and writes nothing — the troubleshooting action. It does "
            "not free memory and does not survive reopening the file; Bake does both.")
        hold.clicked.connect(lambda: self.dock_action.emit(nid, "hold"))
        row.addWidget(hold)
        bake = QPushButton("Re-bake" if rec else "Bake")
        bake.setEnabled(precision != PRECISION_UNSET)
        bake.setToolTip("Compute everything above once and write it to the dock folder, "
                        "then serve it from there. Slower to enter than Hold, but it frees "
                        "memory and survives saving and reopening the graph.")
        bake.clicked.connect(lambda: self.dock_action.emit(nid, "bake"))
        row.addWidget(bake)
        if held_now:
            rel = QPushButton("Release")
            rel.setToolTip("Drop the held copy and run the chain above live again.")
            rel.clicked.connect(lambda: self.dock_action.emit(nid, "release"))
            row.addWidget(rel)
        if status in ("docked", "stale"):
            und = QPushButton("Un-dock")
            und.setToolTip("Run the chain above live again. The bake stays on disk, so "
                           "re-docking costs nothing.")
            und.clicked.connect(lambda: self.dock_action.emit(nid, "undock"))
            row.addWidget(und)
        elif rec and status == "live":
            red = QPushButton("Re-dock")
            red.setToolTip("Serve the existing bake again without recomputing it.")
            red.clicked.connect(lambda: self.dock_action.emit(nid, "redock"))
            row.addWidget(red)
        row.addStretch(1)
        lay.addLayout(row)

        row2 = QHBoxLayout(); row2.setSpacing(6)
        scoped = QPushButton("Bake selection only")
        scoped.setEnabled(precision != PRECISION_UNSET)
        scoped.setToolTip(
            "Bake ONLY the frames the Viewer's M/T/Z strips have picked. A shortcut for "
            "checking a chain — the checkpoint then holds a truncated series, and every "
            "node after it runs on that, so it is not a result to report.")
        scoped.clicked.connect(lambda: self.dock_action.emit(nid, "bake_scoped"))
        row2.addWidget(scoped)
        if rec:
            show = QPushButton("Show folder")
            show.clicked.connect(lambda: self.dock_action.emit(nid, "reveal"))
            row2.addWidget(show)
        row2.addStretch(1)
        lay.addLayout(row2)
        return sec

    # ── rows ────────────────────────────────────────────────────────────────
    def _param_row(self, node: NodeItem, s) -> QWidget:
        """One parameter, as a vertical block: the value row, then whatever extra surface
        this socket's declarations ask for (a Pick button, a tick list).

        A block rather than a bare row because the value line is already at its width budget
        — label, editor, unit, ƒ-auto — inside a 376 px dock. Anything else has to go
        underneath, which is also the better read: the editor stays where the eye expects it
        and the affordance sits with it instead of competing for the same column."""
        block = QWidget()
        outer = QVBoxLayout(block)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.addWidget(self._value_row(node, s))
        prob = self._problem_sockets.get(s.name)
        if prob is not None:
            # the red highlight: the readiness block above says why; this says WHERE
            block.setObjectName("needRow")
            block.setAttribute(Qt.WA_StyledBackground, True)
            block.setStyleSheet(
                f"QWidget#needRow {{ border-left:3px solid {_h(T.ERROR)}; "
                f"background:{_h(T.mix(T.PANEL, T.ERROR, 0.10))}; border-radius:4px; }}")
            outer.setContentsMargins(6, 2, 2, 2)
            block.setToolTip("⚠ " + prob.message)
        self._mark_override(block, node, s.name)
        driver = self._driver_of(node, s.name)
        if driver is not None:
            # A driven param's editor is dead: the rewrite bakes the sweep's value over
            # whatever is typed here. Leaving it live would be the same defect as a socket
            # the kernel ignores — a control that looks like it does something.
            block.setEnabled(False)
            note = QLabel(f"driven by {driver} — edit the sweep there")
            note.setProperty("role", "muted")
            nf = note.font(); nf.setPointSize(9); nf.setItalic(True); note.setFont(nf)
            note.setContentsMargins(78, 0, 0, 2)
            outer.addWidget(note)
            block.setEnabled(True)          # keep the NOTE readable; lock the editor only
            for w in block.findChildren(QWidget):
                if w is not note:
                    w.setEnabled(False)
            return block
        if getattr(s, "vocab", ()):
            outer.addWidget(self._vocab_box(node, s))
        elif s.type is SocketType.STRING and getattr(s, "pick_kind", "") == "channels":
            outer.addWidget(self._channel_box(node, s))
        if getattr(s, "pick_kind", "") == "shapes":
            # A drawn region is never configured on the image (2026-10-02). A node that
            # WANTS a region gets a button that brings in a Draw Regions node; a node that
            # IS the drawing (Draw Regions, ROI Mask) gets the Draw section below instead.
            alt = RD.EMPTY_PICKS.get((node.op_key, s.name))
            if alt and alt[0]:
                outer.addWidget(self._region_button(node, s, alt))
            return block
        if getattr(s, "pick_kind", "") and self._leads_pick(node, s):
            outer.addWidget(self._pick_button(node, s))
        return block

    def _region_button(self, node: NodeItem, s, alt) -> QWidget:
        """`Draw <socket>…` on a node that takes a region from another node: the window
        drops a Draw Regions node in line on the region input (or finds the one already
        there), switches to it, and switches back when its drawing is applied."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(78, 0, 0, 2)
        lay.setSpacing(6)
        btn = QToolButton()
        btn.setProperty("role", "pick")
        alt_in, alt_op, _need = alt
        alt_spec = NODES.get(alt_op) if alt_op else None
        alt_label = alt_spec.label if alt_spec is not None else alt_op
        a = node.spec.input(alt_in) if node.spec is not None else None
        in_label = (a.label or a.name) if a is not None else alt_in
        btn.setText(f"{PICK_GLYPH}  Draw {s.label or s.name} on a {alt_label} node…")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(f"Adds a {alt_label} node wired into `{in_label}` (or opens the one "
                       f"already wired there), fed from this node's own image, and switches "
                       f"to it. Draw the region(s) there — tool, add/cut and Apply are in "
                       f"its panel — and Apply brings you back here.")
        btn.clicked.connect(
            lambda _c, nid=node.node_id, sk=s.name: self.region_requested.emit(nid, sk))
        lay.addWidget(btn)
        lay.addStretch(1)
        return row

    # ── the Draw section (2026-10-02) ───────────────────────────────────────
    def _shapes_socket(self, node: NodeItem):
        """The node's active ``pick_kind="shapes"`` socket when the node IS the drawing
        (no Dataset alternative in :data:`readiness.EMPTY_PICKS`), else ``None``."""
        for s in node._active_inputs():
            if getattr(s, "pick_kind", "") != "shapes":
                continue
            alt = RD.EMPTY_PICKS.get((node.op_key, s.name))
            if alt and alt[0]:
                return None
            return s
        return None

    def _draw_section(self, node: NodeItem, s) -> QWidget:
        """The drawing WORKFLOW, top to bottom in the node's panel: **Draw regions** (arm
        the gesture on the viewer), then the shape to draw, add or cut, the brush size,
        then Undo / Clear / Close polygon with the live readout, then **Apply** / Cancel.
        The tool settings are the node's own presentation params, edited here (not in the
        Parameters list) and pushed into the armed gesture as they change (``sync``).
        Nothing about the drawing is configured on the image — the viewer only takes the
        mouse and shows the live shape."""
        sec = self._section("Draw regions")
        lay = sec._lay  # type: ignore[attr-defined]
        armed = self._draw_armed == node.node_id
        w: dict = {}
        hint = QLabel("1. Press Draw regions.  2. Pick a shape, Add or Cut.  3. Drag on the "
                      "image — each finished shape is pinned to the frame you are looking "
                      "at.  4. Apply writes the regions into this node's data."
                      if not armed else
                      "Drawing on the image — pick a shape and Add/Cut below, drag to make "
                      "it; Apply when done.")
        hint.setProperty("role", "muted"); hint.setWordWrap(True)
        hf = hint.font(); hf.setPointSize(9); hint.setFont(hf)
        lay.addWidget(hint)
        w["hint"] = hint
        row = QHBoxLayout(); row.setSpacing(6)
        nid = node.node_id

        def button(text, what, tip, role="pick"):
            b = QToolButton(); b.setText(text); b.setProperty("role", role)
            b.setCursor(Qt.PointingHandCursor); b.setToolTip(tip)
            b.clicked.connect(lambda _c, n=nid, k=what: self.draw_control.emit(n, k, None))
            return b
        if not armed:
            draw = button(f"{PICK_GLYPH}  Draw regions", "arm",
                          "Arm the drawing on the viewer: the next drags on the image make "
                          "shapes with the shape / Add-Cut settings below.")
            row.addWidget(draw); w["draw"] = draw
            row.addStretch(1)
            lay.addLayout(row)
        # the tool settings, in the order they are used: what to draw, add or cut, how wide
        spec = node.spec
        for name in _DRAW_TOOL_PARAMS:
            ps = spec.input(name) if spec is not None else None
            if ps is not None and ps in node._active_inputs():
                lay.addWidget(self._param_row(node, ps))
        if armed:
            for text, what, tip in (
                    ("Undo", "undo", "Drop the last shape (or the polygon in progress)"),
                    ("Clear", "clear", "Start over: everything drawn so far is discarded"),
                    ("Close polygon", "close", "Finish the polygon in progress")):
                row.addWidget(button(text, what, tip, role=""))
            row.addStretch(1)
            lay.addLayout(row)
        readout = QLabel(self._draw_readout if armed else self._shape_summary(node, s))
        readout.setProperty("role", "muted"); readout.setWordWrap(True)
        rf = readout.font(); rf.setPointSize(9); readout.setFont(rf)
        lay.addWidget(readout); w["readout"] = readout
        if armed:
            row2 = QHBoxLayout(); row2.setSpacing(6)
            row2.addWidget(button("✓  Apply", "apply",
                                  "Write the drawn shapes into this node (Enter on the "
                                  "viewer does the same) and go back to the node that asked "
                                  "for them, if one did."))
            row2.addWidget(button("Cancel", "cancel",
                                  "Leave the node's shapes as they were (Esc).", role=""))
            row2.addStretch(1)
            lay.addLayout(row2)
        self._draw_widgets = w
        return sec

    @staticmethod
    def _shape_summary(node: NodeItem, s) -> str:
        """What the node holds: "3 shapes on 2 frames" — read from the socket's JSON."""
        import json as _json
        raw = node.params.get(s.name, "")
        try:
            shapes = _json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or [])
        except ValueError:
            return "the shapes list is not valid JSON — Clear and redraw"
        if not isinstance(shapes, list) or not shapes:
            return "nothing drawn yet"
        regions = [x for x in shapes if isinstance(x, dict)
                   and x.get("type") in ("rect", "ellipse", "circle", "polygon", "brush")]
        frames = {tuple(x["frame"]) for x in regions if isinstance(x.get("frame"), list)}
        every = sum(1 for x in regions if "frame" not in x)
        parts = [f"{len(regions)} shape{'s' if len(regions) != 1 else ''}"]
        if frames:
            parts.append(f"on {len(frames)} frame{'s' if len(frames) != 1 else ''}")
        if every:
            parts.append(f"{every} on every frame")
        return " · ".join(parts)

    def set_draw_state(self, node_id: Optional[str], armed: bool, readout: str = "") -> None:
        """The window's word on the gesture: armed for ``node_id`` or not, plus the live
        readout. Rebuilds the panel when the armed state flips (the Draw section's buttons
        change), updates the readout in place otherwise."""
        was = self._draw_armed
        self._draw_armed = node_id if armed else None
        self._draw_readout = readout
        if was != self._draw_armed:
            self._rebuild()
            return
        lab = self._draw_widgets.get("readout")
        if lab is not None and armed:
            try:
                lab.setText(readout)
            except RuntimeError:
                pass

    @staticmethod
    def _driver_of(node: NodeItem, socket_name: str) -> Optional[str]:
        """The Iterate node driving ``socket_name`` on this node, or ``None``."""
        doc = getattr(node, "doc", None)
        if doc is None or not getattr(doc, "has_iterate", False):
            return None
        for e in doc.edges:
            if e[2] == node.node_id and e[3] == socket_name and _is_driver(doc, e):
                return e[0]
        return None

    @staticmethod
    def _leads_pick(node: NodeItem, s) -> bool:
        """Whether THIS row draws the group's Pick button. A bound group (a crop rectangle's
        four bounds) arms one gesture, so the button belongs once — on the first member —
        rather than four identical times down the panel."""
        peer = node.spec.input(s.pick_peer) if (node.spec and s.pick_peer) else None
        return request_for(node.node_id, s, peer).leads_group

    def _pick_button(self, node: NodeItem, s) -> QWidget:
        """The Pick button for one annotated socket, indented under its editor."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(78, 0, 0, 2)      # line up with the editor, not the label
        lay.setSpacing(6)
        btn = QToolButton()
        btn.setProperty("role", "pick")
        peer = node.spec.input(s.pick_peer) if (node.spec and s.pick_peer) else None
        text = f"{PICK_GLYPH}  {request_for(node.node_id, s, peer).action_text}"
        if peer is not None:
            text += f"  (+ {peer.label or peer.name})"
        btn.setText(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(PICK_HELP.get(s.pick_kind, ""))
        btn.clicked.connect(
            lambda _c, sk=s, pk=peer: self.pick_requested.emit(
                request_for(node.node_id, sk, pk)))
        lay.addWidget(btn)
        lay.addStretch(1)
        return row

    def _vocab_box(self, node: NodeItem, s) -> QWidget:
        """A tick list over a closed vocabulary, stored as the comma-joined string the
        compute already parses (``SocketSpec.vocab``).

        The line edit this replaces asked the user to reproduce a menu from memory, and got
        no feedback when they missed: an unknown token raises, but a *misspelt* one that
        happens to be dropped simply yields no column. Ticking cannot spell anything wrong.
        Tokens are emitted in VOCAB order, not click order — the consumers de-dup and
        preserve order for column layout only, so a stable order means re-ticking the same
        set produces the same param string and therefore the same memo key."""
        vocab = list(s.vocab)
        raw = str(node.params.get(s.name, s.default or ""))
        chosen = {t.strip() for t in raw.split(",") if t.strip()}
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(78, 1, 0, 3)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(1)
        boxes: List[QCheckBox] = []

        def commit() -> None:
            picked = [t for b, t in zip(boxes, vocab) if b.isChecked()]
            self._set_param(node, s.name, ",".join(picked))

        docs = getattr(s, "choice_docs", None) or {}
        for i, token in enumerate(vocab):
            cb = QCheckBox(token)
            cb.setChecked(token in chosen)
            # Per-TOKEN documentation (V2.21). A tick list has no popup to hang tips off, and
            # its tokens are the least self-explanatory params in the catalog — `feret`,
            # `solidity`, `eccentricity` are column names, not explanations — so each box
            # carries its own. The socket's `description` covers the group above it.
            cb.setToolTip(option_hover_text(token, docs.get(token, "")))
            cb.toggled.connect(lambda _v: commit())
            boxes.append(cb)
            grid.addWidget(cb, i // 2, i % 2)
        # Anything already in the param that the vocabulary does not contain is shown rather
        # than silently dropped by the round-trip — it is almost certainly the typo this
        # editor exists to prevent, and quietly deleting it would hide the fix.
        stray = sorted(chosen - set(vocab))
        if stray:
            warn = QLabel("not recognised: " + ", ".join(stray))
            warn.setProperty("role", "vocab")
            warn.setWordWrap(True)
            wf = warn.font(); wf.setPointSize(9); warn.setFont(wf)
            warn.setStyleSheet(f"color:{_h(T.ERROR)};")
            grid.addWidget(warn, (len(vocab) + 1) // 2, 0, 1, 2)
        return host

    def _channel_box(self, node: NodeItem, s) -> QWidget:
        """A tick list of the channels actually arriving, by NAME — the editor for
        ``channel.select``'s 0-based index string.

        ``"0,2"`` is the machine's spelling of "the DAPI and the mCherry"; the names come
        from the file, so this offers what the user recognises and writes the indices for
        them. Order is preserved from the ticks (ascending), which loses the reordering that
        typing ``"2,0"`` allows — so the text field stays right above it for that case,
        rather than being replaced."""
        try:
            descs = node.doc.upstream_channel_descriptors(node.node_id)
        except Exception:                     # noqa: BLE001 — never break the panel
            descs = []
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(78, 1, 0, 3)
        lay.setSpacing(1)
        if not descs:
            hint = QLabel("no channels detected upstream yet")
            hint.setProperty("role", "muted")
            hf = hint.font(); hf.setPointSize(9); hint.setFont(hf)
            lay.addWidget(hint)
            return host
        raw = str(node.params.get(s.name, s.default or ""))
        picked = {int(t) for t in (u.strip() for u in raw.split(",")) if t.isdigit()}
        boxes: List[QCheckBox] = []

        def commit() -> None:
            keep = [str(i) for i, b in enumerate(boxes) if b.isChecked()]
            # Every box unticked means "keep everything" (the socket's own empty-string
            # default), not "keep nothing" — a zero-channel Dataset is not a thing a user
            # can want, and the compute already reads empty as unchanged.
            self._set_param(node, s.name, ",".join(keep))

        for i, ch in enumerate(descs):
            name = ch.get("name") or f"Ch{i}"
            nm = ch.get("emission_nm")
            cb = QCheckBox(f"{i} · {name}" + (f"  ({int(nm)} nm)" if nm else ""))
            # empty selector = keep everything, so show every box ticked in that state
            cb.setChecked(i in picked if picked else True)
            cb.toggled.connect(lambda _v: commit())
            boxes.append(cb)
            lay.addWidget(cb)
        return host

    def _value_row(self, node: NodeItem, s) -> QWidget:
        row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 3, 0, 3); lay.setSpacing(8)
        lab = QLabel(s.name); lab.setMinimumWidth(78); lay.addWidget(lab)
        # Hover documentation (`SocketSpec.description`) — what the param does and how it
        # moves the result. Set on the ROW, not on each editor: this function has five early
        # `return row` paths (bool / int / layer / string / float), so per-editor calls would
        # have to be repeated five times and would rot the next time a branch is added. Qt
        # leaves a ToolTip event unaccepted by a widget whose own tooltip is empty, so it
        # propagates to the parent — the spin box, checkbox and line edit set none, and all
        # of them inherit the row's. The label gets it explicitly because it is the part of
        # the row a user actually aims at.
        tip = socket_hover_text(s)
        row.setToolTip(tip); lab.setToolTip(tip)
        lay.addStretch(1)
        pinned = s.name in node.params or s.name in node.locked
        # A socket has an auto value when SOMETHING other than its static default speaks for
        # it: a metadata `derive`, or the record of the model this node has loaded (V2.23).
        # `node.trained()` is the same call the card's pill makes, so both surfaces agree.
        trained_here = s.name in node.trained()
        derived = bool(s.derive) or trained_here
        auto = derived and not pinned

        if s.type is SocketType.BOOL:
            # A flag is a checkbox, not a 0.000 / 1.000 spin box. Without this branch BOOL
            # fell through to the float editor below, so every toggle in the catalog read
            # as a mysterious decimal ("normalize 1.000"). The value is committed as a real
            # `bool` because the computes read it through `bool(ctx.params.get(...))` and
            # the param is serialized as-is.
            chk = QCheckBox()
            # `resolved` rather than `params.get(..., default)` so an AUTO flag shows the
            # value actually in force — ZS-DeconvNet's `upsample` on a checkpoint trained
            # without the 2x head must read as unticked, not as the socket's `True`. That
            # mismatch was the whole failure this adoption exists to prevent, and a panel
            # that displayed the wrong state would just relocate it.
            chk.setChecked(bool(node.resolved(s)) if auto
                           else bool(node.params.get(s.name, s.default)))
            chk.setEnabled(not auto)
            chk.toggled.connect(lambda v, nm=s.name: self._set_param(node, nm, bool(v)))
            lay.addWidget(chk)
            if derived:
                lay.addWidget(self._pin_button(node, s, auto=auto, pinned=pinned))
            return row
        if s.type is SocketType.INT:
            ival = int(node.params.get(
                s.name, _derived_value(node, s) if derived
                else (s.default if s.default is not None else 0)))
            # Range and step both widened: 100000 was reachable (an `iterations` above it was
            # silently truncated to the ceiling), and a step of 1 makes a 33000-iteration
            # budget 33000 clicks. Shared with the card's scrub so the two agree.
            box = QSpinBox(); box.setRange(0, NUM_MAX_INT)
            box.setSingleStep(int(value_step(s, ival, integer=True)))
            box.setValue(ival)
        elif (s.type is SocketType.STRING and node.op_key == PAGE_INPUT_OP
              and s.name == PAGE_SOURCE_KEY):
            lay.addWidget(self._page_source_box(node, s))
            return row
        elif s.type is SocketType.STRING and getattr(s, "choices", ()):
            lay.addWidget(self._choice_box(node, s))
            return row
        elif s.type is SocketType.STRING and (s.layer_in or s.layer_in_mode):
            lay.addWidget(self._layer_box(node, s))
            return row
        elif s.type is SocketType.STRING and (getattr(s, "column_in", None)
                                              or getattr(s, "column_in_mode", "")):
            lay.addWidget(self._column_box(node, s))
            return row
        elif s.type is SocketType.STRING:
            from PySide6.QtWidgets import QLineEdit
            # Any socket declaring `path_kind` is a filesystem path and gets a Browse…
            # button — the declaration lives on the SocketSpec, so a new path param is
            # browsable the moment it is declared. (Before V2.15 this matched the literal
            # socket name "path", which left `sd_model_path` / `model_path` typing-only.)
            is_path = bool(getattr(s, "path_kind", ""))
            box = QLineEdit(str(node.params.get(s.name, s.default or "")))
            if is_path and s.path_hint:
                box.setPlaceholderText(s.path_hint)
            # commit on editing-finished (not per keystroke — a path edit mid-typing
            # must not spam the document/propagation)
            box.editingFinished.connect(
                lambda b=box, nm=s.name, p=is_path: self._commit_text(node, nm, b, p))
            lay.addWidget(box)
            if is_path:
                browse = QToolButton(); browse.setText("Browse…")
                browse.setToolTip(
                    "Choose a folder — fills the path (no manual typing)"
                    if s.path_kind == "directory" else
                    "Choose a file — fills the path (no manual typing)")
                browse.clicked.connect(
                    lambda _c, b=box, sk=s: self._browse_path(node, sk, b))
                lay.addWidget(browse)
            return row
        else:
            base = node.params.get(s.name, _derived_value(node, s) if derived
                                   else (s.default if s.default is not None else 0.0))
            # Decimals/step come from the socket's MAGNITUDE, not a fixed 3/0.05: at 3
            # decimals a 5e-5 learning rate rounded to 0.000, so the box displayed zero for a
            # non-zero default and the smallest value reachable was 0.001. Shared with the
            # card's scrub step so the two surfaces cannot disagree about what is typable.
            decimals, step = float_editor_precision(s, base)
            box = QDoubleSpinBox(); box.setRange(0.0, NUM_MAX_FLOAT)
            box.setDecimals(decimals); box.setSingleStep(step)
            try:
                box.setValue(float(base))
            except (TypeError, ValueError):
                box.setValue(0.0)
        box.setEnabled(not auto)
        box.valueChanged.connect(lambda v, nm=s.name: self._set_param(node, nm, v))
        if auto and isinstance(box, (QDoubleSpinBox, QSpinBox)):
            # QSpinBox joined QDoubleSpinBox here in V2.23. `refresh_derived` only calls
            # `setValue`, which both have, so the INT case worked all along — it was simply
            # never registered, so an auto INT box showed its build-time value and then never
            # moved again. Latent while no INT socket had a `derive`; ZS-DeconvNet's padding
            # margins are auto from the checkpoint, so a box that ignored a model change
            # would show a stale margin for the wrong graph.
            self._auto_boxes.append((node, s, box))
        lay.addWidget(box)

        unit = _UNIT.get(s.unit, s.unit)
        if unit:
            u = QLabel(unit); u.setProperty("role", "muted"); u.setFixedWidth(24)
            uf = u.font(); uf.setPointSize(9); u.setFont(uf); lay.addWidget(u)

        if derived:
            lay.addWidget(self._pin_button(node, s, auto=auto, pinned=pinned))
        return row

    def _pin_button(self, node: NodeItem, s, *, auto: bool, pinned: bool):
        """The ƒ auto / pinned toggle. Shared by the BOOL branch and the numeric ones since
        V2.23, when BOOL gained an auto state (ZS-DeconvNet's ``upsample`` is adopted from the
        checkpoint) — two copies of this would have drifted the first time the wording changed.
        """
        btn = QToolButton(); btn.setCheckable(True); btn.setChecked(not auto)
        btn.setText("pinned" if pinned else "ƒ auto")
        btn.setProperty("state", "pinned" if pinned else "auto")
        # The tooltip names the SOURCE, because "auto" answers the wrong question once there
        # are two of them: a metadata-derived value tracks the incoming file, a
        # checkpoint-derived one tracks which model is loaded, and pinning means something
        # different against each.
        src = _auto_source(node, s)
        if auto and src == "model":
            btn.setToolTip("Auto — from the loaded model's own training record. Pin to "
                           "override it deliberately; unpin to follow the checkpoint again.")
        elif auto:
            btn.setToolTip("Metadata-derived (auto). Pin to fix the value; unpin to revert.")
        elif src == "model":
            btn.setToolTip("Pinned. Click to revert to what the loaded model was trained "
                           "with.")
        else:
            btn.setToolTip("Pinned. Click to revert to the metadata-derived value.")
        btn.clicked.connect(lambda _c, nm=s.name: self._toggle_pin(node, nm))
        return btn

    def _choice_box(self, node: NodeItem, s):
        """A CLOSED dropdown for a socket declaring ``SocketSpec.choices``.

        Closed, unlike :meth:`_layer_box`'s editable combo, because the difference between
        the two cases is whether the set is knowable up front. A layer name may be invented
        by a producer the edit-time pass cannot predict, so free text has to stay reachable;
        a pretrained checkpoint name is a key into someone else's published model zoo, where
        a value not on the list is never right and only fails once the download 404s
        mid-pull. ``activated`` (a real user pick) rather than ``currentTextChanged``, which
        also fires while the list is being populated."""
        box = _NoWheelCombo()
        box.setFocusPolicy(Qt.StrongFocus)
        choices = list(s.choices)
        box.addItems(choices)
        _set_item_tips(box, choices, getattr(s, "choice_docs", None))
        current = str(node.params.get(s.name, s.default or ""))
        if current in choices:
            box.setCurrentIndex(choices.index(current))
        elif current:
            # A value from a saved graph that is no longer offered stays visible and
            # selected rather than snapping to entry 0 — silently re-pointing a graph at a
            # different model on load would change what it computes.
            box.insertItem(0, current)
            box.setCurrentIndex(0)
            box.setToolTip(f"“{current}” is not one of the registered checkpoints — kept "
                           f"as saved. Pick another to replace it.")
        # `_after_model_edit` too: this dropdown is how a PRETRAINED checkpoint is chosen
        # (`model_name` / `model_name_3d` / `cellsam_model`), so picking another entry points
        # the node at a different model directory — and therefore at different trained
        # thresholds — exactly as editing a local model path does.
        box.activated.connect(
            lambda _i, nm=s.name, b=box: (self._set_param(node, nm, b.currentText()),
                                          self._after_model_edit(node)))
        return box

    def _layer_box(self, node: NodeItem, s):
        """An EDITABLE combo for a ``layer_in`` socket: the layers actually present on
        the incoming edge, plus free text.

        Editable rather than a closed dropdown because the suggestion list is honest but
        incomplete — a couple of producers name layers the edit-time pass cannot predict
        (``transform.rasterize_field`` invents one Voxel layer per field component from a
        prefix), and a node can be wired up after the consumer is configured. A closed
        list would make those layers unreachable; free text with suggestions strictly
        improves on the plain line edit it replaces and can never block a valid name.

        Signal choice matters. ``currentTextChanged``/``currentIndexChanged`` also fire
        while the list is being REPOPULATED and while the user types, so they would
        commit half-typed names and fight the rebuild. Only ``activated`` (a real user
        pick, never programmatic) and the line edit's ``editingFinished`` commit."""
        try:
            choices = list(node.doc.layer_choices(node.node_id, s))
        except Exception:                            # never let a picker break the panel
            choices = []
        return self._names_box(
            node, s, choices,
            present="Layers on the incoming edge: ",
            empty="No layers detected upstream yet — connect a producer, or type "
                  "the name.",
            note="\n(free text is allowed — some layer names cannot be predicted "
                 "before the graph runs)")

    def _page_source_box(self, node: NodeItem, s):
        """A Page Input's Source (V4.00 step 5): a CLOSED menu of the named Outputs of the
        pages that may feed this one — "Image Input · raw" — rather than a typed reference.
        An earlier page's Output is the only thing a Page Input can read, so there is nothing
        legitimate to type that the menu does not list; a reference that no longer resolves
        (its page or Output renamed or deleted) stays as the first entry, marked, so the
        setting is never silently changed.

        Under the menu (V4.00 step 11): what to do when it is empty or stale, in words on the
        panel rather than in a tooltip, and a *Go to <page>* button for the page to do it on —
        the nearest page that may feed this one, or the page a bound source comes from."""
        from PySide6.QtCore import Qt
        try:
            choices = list(node.doc.source_choices(node.node_id))
        except Exception:                            # never let a picker break the panel
            choices = []
        try:
            feeders = [(str(p), str(n)) for p, n in node.doc.page_feeders()]
        except Exception:                            # noqa: BLE001 — a bare document
            feeders = []
        current = str(node.params.get(s.name, s.default or "") or "")
        box = _NoWheelCombo()
        box.setFocusPolicy(Qt.StrongFocus)
        box.blockSignals(True)
        values = [v for v, _label in choices]
        if not current:
            box.addItem("— choose an upstream Output —", "")
        elif current not in values:
            box.addItem(f"{current}  (unbound — not offered any more)", current)
        from nodelab_v2.canvas import kind_icon
        for value, label in choices:
            # the dot is the colour of the page kind it reads from (V4.00 step 11e)
            try:
                kind = node.doc.source_kind(value) or "free"
            except Exception:                        # noqa: BLE001 — a bare document
                kind = "free"
            box.addItem(kind_icon(kind), label, value)
        idx = box.findData(current)
        box.setCurrentIndex(idx if idx >= 0 else 0)
        box.blockSignals(False)
        box.setToolTip(
            "Reads a named Output of an earlier page (by kind: an Input page feeds Refinement,"
            " Refinement feeds Processing, …; a Free page may feed or read any page). "
            + ("Offered: " + ", ".join(label for _v, label in choices) if choices else
               "No page that may feed this one has a named Page Output yet — add a Page "
               "Output node there and give it a name."))
        box.activated.connect(
            lambda _i, nm=s.name, b=box: self._set_param(node, nm, str(b.currentData() or "")))

        hint = ""
        goto: list = []
        if not choices:
            where = feeders[0][1] if feeders else "an earlier page"
            hint = (f"No earlier page has a named Page Output yet — load an image on {where}, "
                    f"or add a Page Output there and give it a name.")
            goto = feeders[:2]
        elif current and current not in values:
            where = feeders[0][1] if feeders else "the page it came from"
            hint = (f"Unbound — pick a source again, or go to {where} to restore the Output "
                    f"it read.")
            goto = feeders[:2]
        elif not current:
            hint = ("No source chosen — pick an Output above, or take a *Bind to …* button in "
                    "the Ready-to-run block.")
            goto = feeders[:1]
        else:
            src_pid = current.split(":", 1)[0]
            goto = [(p, n) for p, n in feeders if p == src_pid][:1]
        if not hint and not goto:
            return box
        wrap = QWidget()
        wl = QVBoxLayout(wrap); wl.setContentsMargins(0, 0, 0, 0); wl.setSpacing(4)
        wl.addWidget(box)
        if hint:
            lab = QLabel(hint)
            lab.setWordWrap(True)
            lab.setProperty("role", "muted")
            lab.setStyleSheet(f"color:{_h(T.MUTED)}; background:transparent;")
            lf = lab.font(); lf.setPointSize(8); lab.setFont(lf)
            wl.addWidget(lab)
        if goto:
            row = QHBoxLayout(); row.setSpacing(6)
            for pid, pname in goto:
                b = QToolButton()
                b.setText(f"Go to {pname}")
                b.setProperty("role", "add")
                b.setCursor(Qt.PointingHandCursor)
                b.setToolTip(f"Show the page “{pname}” on this canvas")
                b.clicked.connect(lambda _c, p=pid: self.page_requested.emit(p))
                row.addWidget(b)
            row.addStretch(1)
            wl.addLayout(row)
        return wrap

    def _column_box(self, node: NodeItem, s):
        """A CLOSED dropdown for a ``column_in`` socket: the columns the edit-time pass
        knows were MEASURED onto this wire, and nothing else (V2.28).

        Same widget as :meth:`_layer_box`, but CLOSED (``editable=False``) where that one
        is open, and the difference is not a style choice. Every structure-producing node
        declares ``adds_columns`` (``selftest::test_column_catalog_complete``), so the
        columns a table carries are fully determined by the nodes upstream — there is
        nothing legitimate to type that is not already in the list, and typing anything
        else only defers the refusal to the pull.

        The empty state names the nodes that PRODUCE columns rather than reporting an empty
        list, because "no columns" is almost always "no Measure in this graph yet", and that
        is the one thing the user needs told."""
        try:
            choices = list(node.doc.column_choices(node.node_id, s))
        except Exception:                            # never let a picker break the panel
            choices = []
        return self._names_box(
            node, s, choices,
            present="Columns measured upstream: ",
            empty="No measured columns on this wire yet — add Measure (intensity, "
                  "shape), Object Metrics (speed, neighbours) or Track Objects "
                  "(track_length) upstream, and its columns appear here.",
            note="\n(every node that produces columns declares them, so this list "
                 "is what the graph actually carries)",
            editable=False)

    def _names_box(self, node: NodeItem, s, choices: list, *,
                   present: str, empty: str, note: str, editable: bool = True):
        """The shared name combo behind :meth:`_layer_box` and :meth:`_column_box`.

        One widget rather than two near-copies: every clause below is a trap that was paid
        for once, and a second copy is where the two would drift apart.

        ``editable`` is the one real difference between the two callers, and it follows from
        how complete their catalogs are. A LAYER catalog cannot be closed — a couple of
        producers name layers the edit-time pass cannot predict. A COLUMN catalog can be, and
        is: every node that adds a structure domain declares ``adds_columns``, enforced by
        ``selftest::test_column_catalog_complete``, so what a table carries is determined by
        the nodes upstream and a closed list cannot hide a real column.

        **The current value is always an entry, offered or not.** A closed combo can only
        emit values that are in its list, so a saved graph whose column has since gone (the
        Measure feeding it deleted, a layer renamed) would otherwise have its setting
        silently rewritten to whatever sits at index 0 the moment the panel rebuilds — a
        value the user never chose, on a node that decides which objects survive."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QCompleter

        current = str(node.params.get(s.name, s.default or ""))
        shown = list(choices)
        orphan = bool(current) and current not in shown
        if orphan:
            shown.insert(0, current)
        # an OPTIONAL column (blank by default — e.g. Group by): a closed list must still
        # offer the way back to blank, or a pick could never be undone
        optional = (not editable) and not str(s.default or "")
        if optional:
            shown.insert(0, _NONE_ENTRY)
            if not current:
                current = _NONE_ENTRY

        box = _NoWheelCombo()
        box.setEditable(editable)
        box.setFocusPolicy(Qt.StrongFocus)
        box.setDuplicatesEnabled(False)
        box.blockSignals(True)
        box.addItems(shown)
        if editable:
            box.setInsertPolicy(QComboBox.NoInsert)  # typing must not grow the list
            box.setEditText(current)
        else:
            box.setCurrentIndex(shown.index(current) if current in shown else -1)
        box.blockSignals(False)

        if editable:
            cp = QCompleter(shown, box)
            cp.setCaseSensitivity(Qt.CaseInsensitive)
            # PopupCompletion, not InlineCompletion: inline would type-ahead-fill the edit
            # with a suggestion, so tabbing away would COMMIT a name the user never chose.
            cp.setCompletionMode(QCompleter.PopupCompletion)
            cp.popup().setStyleSheet(T.controls_qss())   # else the popup ignores the theme
            box.setCompleter(cp)

        tip = (present + ", ".join(choices) + note) if choices else empty
        if orphan:
            tip = (f"{current!r} is NOT on this wire — nothing upstream writes it. It is "
                   f"kept so your setting is not silently changed; pick another entry, or "
                   f"add the node that measures it.\n\n") + tip
        box.setToolTip(tip)
        if orphan:
            box.setItemData(0, tip, Qt.ToolTipRole)
        box.activated.connect(
            lambda _i, nm=s.name, b=box: self._set_param(
                node, nm, "" if b.currentText() == _NONE_ENTRY else b.currentText()))
        if editable:
            box.lineEdit().editingFinished.connect(
                lambda nm=s.name, b=box: self._set_param(node, nm, b.currentText()))
        return box

    def _mode_row(self, node: NodeItem, m) -> QWidget:
        row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 3, 0, 3); lay.setSpacing(8)
        lab = QLabel(m.name); lay.addWidget(lab); lay.addStretch(1)
        # Hover documentation, exactly as a param row gets it (V2.21): the row and its label
        # carry what the dropdown selects plus every option's line, and each combo ITEM
        # carries its own. Both halves are needed — the row answers "what is this control"
        # before the menu is open, the per-item tips answer "which of these do I want" while
        # the user is moving down the list, which is when the question is actually being asked.
        tip = mode_hover_text(m)
        row.setToolTip(tip); lab.setToolTip(tip)
        # _NoWheelCombo: a wheel over a Mode dropdown used to silently change the
        # method (and trigger the deferred rebuild) while the user was only
        # scrolling the panel — the same hazard documented on _NoWheelCombo.
        combo = _NoWheelCombo(); combo.addItems(list(m.choices))
        _set_item_tips(combo, m.choices, getattr(m, "choice_docs", None))
        cur = node.rec.modes.get(m.name, m.resolved_default())
        if cur in m.choices:
            combo.setCurrentText(cur)
        combo.currentTextChanged.connect(lambda t, nm=m.name: self._set_mode(node, nm, t))
        lay.addWidget(combo)
        self._mark_override(row, node, m.name)
        return row

    # ── linked pages (V4.00 step 6) ─────────────────────────────────────────
    def _linked_section(self, node: NodeItem) -> QWidget:
        """The banner of a node on a LINKED page: the master page it follows, how many
        values this page overrides (on the page, and on this node), and the two ways out —
        Go to master, where the graph is edited, and Make unique."""
        doc = node.doc
        master = doc.master_name() or "its master"
        n = doc.override_count()
        ov = doc.overrides.get(node.node_id) or {}
        k = len(ov.get("params") or {}) + len(ov.get("modes") or {}) + (1 if "muted" in ov else 0)
        sec = self._section("Linked page",
                            f"{n} override{'' if n == 1 else 's'} · {k} on this node")
        self._linked_text = f"Linked to “{master}” · {n} override{'' if n == 1 else 's'}"
        head = QLabel(self._linked_text)
        hf = head.font(); hf.setBold(True); head.setFont(hf)
        sec._lay.addWidget(head)  # type: ignore[attr-defined]
        mode = getattr(doc, "edit_mode", "")
        own = node.node_id in (doc.own_node_ids() if hasattr(doc, "own_node_ids") else ())
        if mode == "modified":
            shape = ("A modified linked page: it keeps nodes and wires of its own over the "
                     "master's, and every other edit of the master still arrives. "
                     + ("This node is this page's own. " if own else ""))
        elif mode == "master":
            shape = ("This session, a change to its graph goes to the master — a node added "
                     "here arrives there switched off. ")
        else:
            shape = ("Its nodes, wires and card positions are the master's — changing them "
                     "asks whether to make this page unique, keep the change on this page, or "
                     "add it to the master switched off. ")
        msg = QLabel(shape + "A value changed here — or a node switched on or off — is this "
                     "page's own (marked by the bar on its row; right-click the row to reset "
                     "it to the master's).")
        msg.setWordWrap(True)
        msg.setProperty("role", "muted")
        mf = msg.font(); mf.setPointSize(9); msg.setFont(mf)
        sec._lay.addWidget(msg)  # type: ignore[attr-defined]
        row = QHBoxLayout()
        for text, act, tip in (
                ("Go to master", "master",
                 "Show the master page with this node selected — the graph is edited there "
                 "and every page linked to it follows."),
                ("Make unique", "unique",
                 "Make this a page of its own, holding its current graph and values. The "
                 "master's edits stop reaching it and its graph becomes editable.")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(
                lambda _c=False, a=act, nid=node.node_id: self.linked_action.emit(a, nid))
            row.addWidget(b)
        row.addStretch(1)
        sec._lay.addLayout(row)  # type: ignore[attr-defined]
        return sec

    def _mark_override(self, w: QWidget, node: NodeItem, name: str) -> None:
        """On a linked page, mark a value this page overrides: an accent bar on its row,
        the master's value in the tooltip, and a right-click *Reset to master*."""
        doc = node.doc
        if getattr(doc, "editable_topology", True) or not doc.is_overridden(node.node_id, name):
            return
        mv = doc.master_value(node.node_id, name)
        tip = (f"Overridden on this page — the master's value is {mv!r}." if mv is not None
               else "Overridden on this page — the master leaves it at its default.")
        tip += "  Right-click: Reset to master."
        if not w.objectName():                 # a problem's red bar wins the background
            w.setObjectName("ovrRow")
            w.setAttribute(Qt.WA_StyledBackground, True)
            w.setStyleSheet(
                f"QWidget#ovrRow {{ border-left:3px solid {_h(T.ACCENT)}; "
                f"background:{_h(T.mix(T.PANEL, T.ACCENT, 0.08))}; border-radius:4px; }}")
            lay = w.layout()
            if lay is not None:
                m = lay.contentsMargins()
                lay.setContentsMargins(6, m.top(), m.right(), m.bottom())
        w.setToolTip(tip + ("\n\n" + w.toolTip() if w.toolTip() else ""))
        w.setContextMenuPolicy(Qt.CustomContextMenu)
        w.customContextMenuRequested.connect(
            lambda pos, ww=w, nid=node.node_id, nm=name: self._override_menu(ww, pos, nid, nm))
        self._override_rows.append((node.node_id, name))

    def _override_menu(self, w: QWidget, pos, node_id: str, name: str) -> None:
        menu = QMenu(self)                     # not the row: the reset rebuilds the panel
        menu.addAction("Reset to master").triggered.connect(
            lambda: self.reset_override(node_id, name))
        menu.exec(w.mapToGlobal(pos))

    def reset_override(self, node_id: str, name: Optional[str] = None) -> None:
        """Back to the master's value (``name``; every value of the node when ``None``)."""
        node = self._node
        doc = node.doc if node is not None else None
        if doc is None or getattr(doc, "editable_topology", True):
            return
        doc.reset_override(node_id, name)
        QTimer.singleShot(0, self._rebuild)    # after the menu that asked has closed

    def _conn_label(self, text: str, col, problem: str = "") -> QWidget:
        """One connection line. ``problem`` (a readiness message) paints the line red with a
        ⚠ — the input in question, found by colour from the block above."""
        row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 2, 0, 2); lay.setSpacing(8)
        dot = QLabel(); dot.setFixedSize(11, 11)
        dot.setStyleSheet(f"background:{_h(T.ERROR if problem else col)}; border-radius:5px;")
        lay.addWidget(dot)
        lab = QLabel(("⚠ " if problem else "") + text)
        lab.setProperty("role", "muted")
        if problem:
            lab.setStyleSheet(f"color:{_h(T.ERROR)}; font-weight:600;")
            row.setToolTip("⚠ " + problem)
        lf = lab.font(); lf.setPointSize(10); lab.setFont(lf)
        lay.addWidget(lab); lay.addStretch(1)
        return row

    # ── grouping by selection on a split card (2026-10-07) ────────────────────
    @staticmethod
    def _split_axis(node: NodeItem) -> Optional[str]:
        fn = getattr(node.doc, "split_axis", None)
        try:
            return fn(node.node_id) if callable(fn) else None
        except Exception:                            # noqa: BLE001 — a bare document
            return None

    def _split_grouping_box(self, node: NodeItem, s) -> QWidget:
        """The grouping control of a split card: every output the card can group, each
        with a checkbox (the same ticks as the card's), **Group selected** with an optional
        name, **Ungroup selected**, **Group every N** for a long axis, and the `groups`
        text itself underneath for whoever would rather type. Members past the fan-out
        cap are listed too — the card has no socket for them, so this is where a
        300-frame series gets grouped."""
        doc = node.doc
        nid = node.node_id
        items = doc.split_items(nid)
        box = QWidget()
        v = QVBoxLayout(box); v.setContentsMargins(0, 2, 0, 2); v.setSpacing(4)
        head = QLabel(f"{s.label or s.name}  ·  tick outputs, then Group selected")
        head.setProperty("role", "muted"); head.setToolTip(socket_hover_text(s))
        hf = head.font(); hf.setPointSize(9); head.setFont(hf)
        v.addWidget(head)

        def _repaint() -> None:
            try:
                node.update()
            except RuntimeError:                     # the card is gone
                pass

        def _after() -> None:
            # the socket set changed: the card relayouts, the wires re-route, the form
            # is rebuilt on the next turn (never tear down the button mid-click)
            try:
                node.refresh()
                sc = node.scene()
                if sc is not None and hasattr(sc, "reroute"):
                    sc.reroute()
            except RuntimeError:
                pass
            QTimer.singleShot(0, self._rebuild)

        for it in items:
            row = QWidget()
            h = QHBoxLayout(row); h.setContentsMargins(4, 0, 0, 0); h.setSpacing(6)
            cb = QCheckBox(("group · " if it["kind"] == "group" else "") + str(it["label"]))
            cb.setChecked(bool(it["picked"]))
            cb.setProperty("splitKey", it["key"])
            if not it["socket"]:
                cb.setToolTip("Past the card's fan-out cap: no socket of its own until it "
                              "is grouped")
            cb.toggled.connect(lambda on, k=it["key"]: (doc.split_pick(nid, k, bool(on)),
                                                        _repaint(), self._split_sync_buttons()))
            h.addWidget(cb, 1)
            if it["kind"] == "group":
                ub = QToolButton(); ub.setText("Ungroup")
                ub.setToolTip("Dissolve this group back into its members' own sockets")
                ub.clicked.connect(lambda _c, k=it["key"]: (doc.split_ungroup(nid, [k]), _after()))
                h.addWidget(ub)
            v.addWidget(row)
        if not items:
            none = QLabel("Nothing to group yet — the axis is not known until the card is wired.")
            none.setProperty("role", "muted"); none.setWordWrap(True); v.addWidget(none)

        act = QWidget()
        ah = QHBoxLayout(act); ah.setContentsMargins(0, 2, 0, 0); ah.setSpacing(6)
        name = QLineEdit(); name.setPlaceholderText("name (optional)")
        name.setToolTip("A name for the group Group selected makes — shown on its socket")
        gb = QPushButton("Group selected")
        gb.setToolTip("Make ONE output carrying every ticked member; their wires move to it")
        gb.clicked.connect(lambda _c: (doc.split_group_selected(nid, name.text()) is not None
                                       and _after()))
        ugb = QPushButton("Ungroup selected")
        ugb.setToolTip("Dissolve the ticked groups back into their members")
        ugb.clicked.connect(lambda _c: (doc.split_ungroup(nid) and _after()))
        ah.addWidget(name, 1); ah.addWidget(gb); ah.addWidget(ugb)
        v.addWidget(act)
        self._split_buttons = (node, gb, ugb)
        self._split_sync_buttons()

        every = QWidget()
        eh = QHBoxLayout(every); eh.setContentsMargins(0, 0, 0, 0); eh.setSpacing(6)
        el = QLabel("Group every"); el.setProperty("role", "muted")
        n = len(items) if items else 0
        total = sum(len(it["indices"]) for it in items)
        spin = QSpinBox(); spin.setRange(1, max(1, total))
        spin.setValue(min(max(1, total), 10 if self._split_axis(node) == "t" else 4))
        spin.setToolTip("Regroup the whole axis into consecutive groups of this many, the "
                        "last one holding what remains — replaces the groups there are")
        eb = QPushButton("Apply")
        eb.setEnabled(total > 0)
        eb.clicked.connect(lambda _c: (doc.split_group_every(nid, spin.value()) and _after()))
        eh.addWidget(el); eh.addWidget(spin); eh.addWidget(eb); eh.addStretch(1)
        v.addWidget(every)

        txt = QWidget()
        th = QHBoxLayout(txt); th.setContentsMargins(0, 0, 0, 0); th.setSpacing(6)
        tl = QLabel("as text"); tl.setProperty("role", "muted"); tl.setToolTip(socket_hover_text(s))
        line = QLineEdit(str(node.params.get(s.name, s.default or "")))
        line.setPlaceholderText("0-3; 4-7  or  top: 0-3; mid: 4-7")
        line.setToolTip(socket_hover_text(s))
        line.editingFinished.connect(
            lambda b=line, nm=s.name: (self._set_param(node, nm, b.text()), _after()))
        th.addWidget(tl); th.addWidget(line, 1)
        v.addWidget(txt)
        return box

    def _split_sync_buttons(self) -> None:
        """Group selected is live while something is ticked; Ungroup selected while a
        ticked output is a group."""
        got = getattr(self, "_split_buttons", None)
        if not got:
            return
        node, gb, ugb = got
        try:
            items = node.doc.split_items(node.node_id)
            picked = {it["key"] for it in items if it["picked"]}
            gb.setEnabled(bool(picked))
            ugb.setVisible(any(it["kind"] == "group" and it["picked"] for it in items))
        except RuntimeError:                         # widgets torn down mid-rebuild
            self._split_buttons = None

    def _commit_text(self, node: NodeItem, name: str, box, is_path: bool) -> None:
        """Commit a QLineEdit string param. For a path field (any socket declaring
        ``path_kind``), normalize first —
        strip whitespace and surrounding quotes (a Windows "Copy as path" paste wraps
        the path in double quotes) — and reflect the cleaned value back in the box."""
        text = _clean_path(box.text()) if is_path else box.text()
        if is_path and text != box.text():
            box.blockSignals(True); box.setText(text); box.blockSignals(False)
        self._set_param(node, name, text)
        self._after_model_edit(node)

    def _browse_path(self, node: NodeItem, s, box) -> None:
        """Open the file/folder chooser this socket declares (``SocketSpec.path_kind``)
        and commit the picked path.

        Directory kinds get ``getExistingDirectory``, not a file dialog: a StarDist local
        model is a FOLDER (config.json + weights_best.h5), and a file dialog cannot return
        one. Whatever the user picks seeds ``_last_dir``, so browsing a second path socket
        starts where the previous one left off instead of at the process CWD — model
        weights and the image they run on usually live near each other."""
        from PySide6.QtWidgets import QFileDialog
        kind = s.path_kind
        current = _clean_path(box.text())
        start = current or self._last_dir
        what = s.label or s.name
        if kind == "directory":
            path = QFileDialog.getExistingDirectory(
                self, f"Select the {what} folder", start,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        else:
            chooser = (QFileDialog.getSaveFileName if kind == "save_file"
                       else QFileDialog.getOpenFileName)
            path, _f = chooser(self, f"Select the {what}", start,
                              s.path_filter or "All files (*)")
        if not path:
            return                                  # cancelled — leave the field alone
        path = os.path.normpath(_clean_path(path))
        self._last_dir = path if kind == "directory" else os.path.dirname(path)
        box.setText(path)
        self._set_param(node, s.name, path)
        self._after_model_edit(node)

    def _after_model_edit(self, node: NodeItem) -> None:
        """Rebuild the form when an edit may have changed which MODEL is loaded (V2.23).

        A full rebuild rather than the cheaper ``refresh_derived``, because pointing at a
        different checkpoint can change the form's STRUCTURE and not merely its numbers: a
        socket with no ``derive`` — ZS-DeconvNet's padding margins, StarDist's thresholds —
        has no auto state and therefore no pin button until a model that speaks for it is
        loaded, and it loses both again when the path is cleared. ``refresh_derived`` only
        calls ``setValue`` on boxes that already exist, so it cannot add or remove the
        button, and the checkbox branch has no box registered at all.

        Deferred to the next event-loop turn for the same reason ``_set_mode`` defers: this
        runs from a widget's own signal handler, and tearing down that widget mid-emit
        crashes. A no-op for the overwhelming majority of nodes, which declare no
        ``trained_params`` and so can never change what a model says.
        """
        if node.spec is not None and node.spec.trained_params is not None:
            QTimer.singleShot(0, self._rebuild)

    # ── edits (all through the document — G8 re-propagates envelopes) ────────
    def _set_param(self, node: NodeItem, name: str, value) -> None:
        rec = node.rec
        rec.params[name] = value
        rec.set_locked(rec.locked | {name})     # editing pins (sticky __locked__)
        node.doc.touch(node.node_id)
        # don't full-rebuild (keeps focus in the box); the card refreshes via sync
        if name in _DRAW_TOOL_PARAMS and self._shapes_socket(node) is not None:
            self.draw_control.emit(node.node_id, "sync", None)   # re-read while armed

    def _set_mode(self, node: NodeItem, name: str, value: str) -> None:
        node.rec.modes[name] = value            # modes are NOT params (serialize split)
        node.doc.touch(node.node_id)
        # A mode change can reconfigure the ACTIVE socket set (`available_in`), so the
        # form has to be rebuilt: `doc.touch()` only re-seeds the auto boxes
        # (refresh_derived) and re-lays-out the card (scene.sync → item.refresh), which
        # would leave the inspector showing the *previous* method's params. Deferred to
        # the next event-loop turn so we never tear down the QComboBox that is mid-emit.
        QTimer.singleShot(0, self._rebuild)

    def _toggle_pin(self, node: NodeItem, name: str) -> None:
        rec = node.rec
        if name in rec.params or name in rec.locked:
            rec.params.pop(name, None)
            rec.set_locked(rec.locked - {name})                 # revert to auto
        else:
            spec_sock = node.spec.input(name) if node.spec else None
            rec.params[name] = (_pin_value(node, spec_sock)
                                if spec_sock is not None else 0.0)
            rec.set_locked(rec.locked | {name})                 # pin the derived value
        node.doc.touch(node.node_id)
        self._rebuild()


__all__ = ["InspectorPanel", "SwitchWidget"]
