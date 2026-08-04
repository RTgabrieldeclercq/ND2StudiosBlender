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
from typing import List, Optional, Sequence

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T
from nodelab_v2.node_item import (
    NodeItem, mode_hover_text, option_hover_text, socket_hover_text,
)
from nodegraph.iterate import ITERATE_OP, SWEEP_KEY, plan as iterate_plan
from nodelab_v2.document import is_driver_edge as _is_driver
from nodelab_v2.ops import DOCK_OP, PRECISION_UNSET, bake_record
from nodelab_v2.picker import PICK_GLYPH, PICK_HELP, request_for

_UNIT = {"um": "µm", "um_axial": "µm↕", "um2": "µm²", "um3": "µm³",
         "nm": "nm", "s": "s", "px": "px"}


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
    """The LIVE metadata-derived value from the node's propagated envelope (G8 —
    replaces the old hard-coded preview table)."""
    v = node.resolved(s)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(s.default) if s.default is not None else 0.0


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
    #: the ⟳ button beside the node title was pressed: ``op_key``. Re-read that node type's
    #: module from disk. Same division of labour as the signals above — the panel asks, the
    #: window owns the reloader, the runner (which must be idle) and the canvas that has to
    #: be relaid out afterwards.
    reload_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        # wide enough for the richest param row (label + spinbox + unit + ƒ-auto button)
        # PLUS the vertical scrollbar, so the right edge (the ƒ-auto buttons) never clips.
        self.setFixedWidth(376)
        self.setStyleSheet(_inspector_qss())
        self._node: Optional[NodeItem] = None
        self._auto_boxes = []              # (node, socket, box) — live ƒmd refresh (G8)
        self._last_dir = ""                # last browsed folder (seeds the next dialog)
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

        # footprint
        fp = self._section("Footprint")
        chip = QLabel(node.granularity().replace("_", " ").upper())
        gcol = T.gran_color(node.granularity())
        chip.setStyleSheet(
            f"color:{_h(gcol)}; border:1px solid {_h(T.alpha(gcol,120))};"
            f"background:{_h(T.alpha(gcol,36))}; border-radius:4px; padding:3px 7px;"
            f"font-family:{T.MONO}; font-weight:700;")
        cf = chip.font(); cf.setPointSize(8); chip.setFont(cf)
        frow = QHBoxLayout(); frow.addWidget(chip); frow.addStretch(1)
        fp._lay.addLayout(frow)  # type: ignore[attr-defined]
        note = QLabel("The 2D / 3D switch resolves this footprint and the active sockets; "
                      "it folds into the memo key, so 2D and 3D cache separately."
                      if spec.has_dim_lever() else "Dimension-agnostic — no 2D/3D switch.")
        note.setProperty("role", "muted"); note.setWordWrap(True)
        nf = note.font(); nf.setPointSize(9); note.setFont(nf)
        fp._lay.addWidget(note)  # type: ignore[attr-defined]
        self._v.addWidget(fp)
        self._v.addWidget(self._sep())

        # parameters
        params = [s for s in node._active_inputs() if s.type is not SocketType.DATASET]
        if params:
            sec = self._section("Parameters", str(len(params)))
            for s in params:
                sec._lay.addWidget(self._param_row(node, s))  # type: ignore[attr-defined]
            self._v.addWidget(sec)
            self._v.addWidget(self._sep())

        # in-body modes (non-dim), mode-gated like the sockets above: a Mode the selected
        # method never reads is hidden, not shown and ignored (V2.12 `ModeSpec.available_in`)
        modes = [m for m in spec.active_modes(node.state()) if not m.is_dim_lever]
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

        # connections
        sec = self._section("Connections")
        for s in spec.inputs:
            if s.type is SocketType.DATASET:
                sec._lay.addWidget(self._conn_label(f"in · {s.name}", T.SOCKET[s.type]))  # type: ignore[attr-defined]
        for s in spec.outputs:
            sec._lay.addWidget(self._conn_label(f"out · {s.name}", T.SOCKET[s.type]))  # type: ignore[attr-defined]
        self._v.addWidget(sec)
        self._v.addStretch(1)

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
            plan = iterate_plan(doc.to_graph(), node.node_id, envs=doc.envs)
        except (ValueError, KeyError) as exc:
            # Every refusal nodegraph.iterate raises already names its fix, so showing it
            # verbatim here is better than paraphrasing — and it arrives while the user is
            # editing rather than when they press play.
            blurb(str(exc), T.ERROR)
            return sec
        except Exception:                     # noqa: BLE001 — a mid-edit graph, not a bug
            blurb("Wire a variable output onto a parameter, and the end of that chain "
                  "back into Collect.", italic=True)
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
        "live": "Running the chain above normally. Bake it to freeze the result to disk "
                "and stop recomputing it.",
        "docked": "Serving the baked checkpoint. Everything above is greyed out and is "
                  "not being evaluated or held in memory.",
        "stale": "Still serving the OLD bake — nothing has changed behind your back. "
                 "Re-bake to pick up the edit, or un-dock to run the chain live.",
        "unbaked": "Set to docked, but there is nothing on disk to serve. Bake it, or "
                   "switch State back to live.",
    }

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
        elif status == "unbaked":
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
        if precision == PRECISION_UNSET:
            warn = QLabel("Pick a Precision above before baking — float32 is the usual "
                          "answer for a filter chain; float64 keeps every last digit at "
                          "4× the size; uint16 only suits data still in camera counts.")
            warn.setProperty("role", "muted"); warn.setWordWrap(True)
            wf = warn.font(); wf.setPointSize(9); warn.setFont(wf)
            lay.addWidget(warn)

        row = QHBoxLayout(); row.setSpacing(6)
        bake = QPushButton("Re-bake" if rec else "Bake")
        bake.setEnabled(precision != PRECISION_UNSET)
        bake.setToolTip("Compute everything above once and write it to the dock folder, "
                        "then serve it from there.")
        bake.clicked.connect(lambda: self.dock_action.emit(nid, "bake"))
        row.addWidget(bake)
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
        if getattr(s, "pick_kind", "") and self._leads_pick(node, s):
            outer.addWidget(self._pick_button(node, s))
        return block

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
        derived = bool(s.derive)
        pinned = s.name in node.params or s.name in node.locked
        auto = derived and not pinned

        if s.type is SocketType.BOOL:
            # A flag is a checkbox, not a 0.000 / 1.000 spin box. Without this branch BOOL
            # fell through to the float editor below, so every toggle in the catalog read
            # as a mysterious decimal ("normalize 1.000"). The value is committed as a real
            # `bool` because the computes read it through `bool(ctx.params.get(...))` and
            # the param is serialized as-is.
            chk = QCheckBox()
            chk.setChecked(bool(node.params.get(s.name, s.default)))
            chk.toggled.connect(lambda v, nm=s.name: self._set_param(node, nm, bool(v)))
            lay.addWidget(chk)
            return row
        if s.type is SocketType.INT:
            box = QSpinBox(); box.setRange(0, 100000)
            box.setValue(int(node.params.get(s.name, s.default if s.default is not None else 0)))
        elif s.type is SocketType.STRING and getattr(s, "choices", ()):
            lay.addWidget(self._choice_box(node, s))
            return row
        elif s.type is SocketType.STRING and (s.layer_in or s.layer_in_mode):
            lay.addWidget(self._layer_box(node, s))
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
            box = QDoubleSpinBox(); box.setRange(0.0, 1e6); box.setDecimals(3); box.setSingleStep(0.05)
            base = node.params.get(s.name, _derived_value(node, s) if derived
                                   else (s.default if s.default is not None else 0.0))
            try:
                box.setValue(float(base))
            except (TypeError, ValueError):
                box.setValue(0.0)
        box.setEnabled(not auto)
        box.valueChanged.connect(lambda v, nm=s.name: self._set_param(node, nm, v))
        if auto and isinstance(box, QDoubleSpinBox):
            self._auto_boxes.append((node, s, box))
        lay.addWidget(box)

        unit = _UNIT.get(s.unit, s.unit)
        if unit:
            u = QLabel(unit); u.setProperty("role", "muted"); u.setFixedWidth(24)
            uf = u.font(); uf.setPointSize(9); u.setFont(uf); lay.addWidget(u)

        if derived:
            btn = QToolButton(); btn.setCheckable(True); btn.setChecked(not auto)
            btn.setText("pinned" if pinned else "ƒ auto")
            btn.setProperty("state", "pinned" if pinned else "auto")
            btn.setToolTip("Metadata-derived (auto). Pin to fix the value; unpin to revert."
                           if auto else "Pinned. Click to revert to the metadata-derived value.")
            btn.clicked.connect(lambda _c, nm=s.name: self._toggle_pin(node, nm))
            lay.addWidget(btn)
        return row

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
        box.activated.connect(
            lambda _i, nm=s.name, b=box: self._set_param(node, nm, b.currentText()))
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
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QCompleter

        box = _NoWheelCombo()
        box.setEditable(True)
        box.setInsertPolicy(QComboBox.NoInsert)      # typing must not grow the list
        box.setFocusPolicy(Qt.StrongFocus)
        box.setDuplicatesEnabled(False)

        try:
            choices = list(node.doc.layer_choices(node.node_id, s))
        except Exception:                            # never let a picker break the panel
            choices = []
        current = str(node.params.get(s.name, s.default or ""))
        box.blockSignals(True)
        box.addItems(choices)
        box.setEditText(current)
        box.blockSignals(False)

        cp = QCompleter(choices, box)
        cp.setCaseSensitivity(Qt.CaseInsensitive)
        # PopupCompletion, not InlineCompletion: inline would type-ahead-fill the edit
        # with a suggestion, so tabbing away would COMMIT a name the user never chose.
        cp.setCompletionMode(QCompleter.PopupCompletion)
        cp.popup().setStyleSheet(T.controls_qss())   # else the popup ignores the theme
        box.setCompleter(cp)

        if choices:
            box.setToolTip("Layers on the incoming edge: " + ", ".join(choices)
                           + "\n(free text is allowed — some layer names cannot be "
                             "predicted before the graph runs)")
        else:
            box.setToolTip("No layers detected upstream yet — connect a producer, or "
                           "type the name.")
        box.activated.connect(
            lambda _i, nm=s.name, b=box: self._set_param(node, nm, b.currentText()))
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
        return row

    def _conn_label(self, text: str, col) -> QWidget:
        row = QWidget(); lay = QHBoxLayout(row); lay.setContentsMargins(0, 2, 0, 2); lay.setSpacing(8)
        dot = QLabel(); dot.setFixedSize(11, 11)
        dot.setStyleSheet(f"background:{_h(col)}; border-radius:5px;")
        lay.addWidget(dot)
        lab = QLabel(text); lab.setProperty("role", "muted")
        lf = lab.font(); lf.setPointSize(10); lab.setFont(lf)
        lay.addWidget(lab); lay.addStretch(1)
        return row

    def _commit_text(self, node: NodeItem, name: str, box, is_path: bool) -> None:
        """Commit a QLineEdit string param. For a path field (any socket declaring
        ``path_kind``), normalize first —
        strip whitespace and surrounding quotes (a Windows "Copy as path" paste wraps
        the path in double quotes) — and reflect the cleaned value back in the box."""
        text = _clean_path(box.text()) if is_path else box.text()
        if is_path and text != box.text():
            box.blockSignals(True); box.setText(text); box.blockSignals(False)
        self._set_param(node, name, text)

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

    # ── edits (all through the document — G8 re-propagates envelopes) ────────
    def _set_param(self, node: NodeItem, name: str, value) -> None:
        rec = node.rec
        rec.params[name] = value
        rec.set_locked(rec.locked | {name})     # editing pins (sticky __locked__)
        node.doc.touch()
        # don't full-rebuild (keeps focus in the box); the card refreshes via sync

    def _set_mode(self, node: NodeItem, name: str, value: str) -> None:
        node.rec.modes[name] = value            # modes are NOT params (serialize split)
        node.doc.touch()
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
            rec.params[name] = (_derived_value(node, spec_sock)
                                if spec_sock is not None else 0.0)
            rec.set_locked(rec.locked | {name})                 # pin the derived value
        node.doc.touch()
        self._rebuild()


__all__ = ["InspectorPanel", "SwitchWidget"]
