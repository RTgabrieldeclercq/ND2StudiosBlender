"""Node / socket / 2D-3D-switch graphics items (NodeLab v2, Phase 5).

A :class:`NodeItem` is a *view* of a :class:`~nodelab_v2.document.NodeRecord`: it reads
params/modes from the record's live dicts (shared with the inspector) and its
metadata-derived ``ƒmd`` pills from the document's propagated
:class:`~nodegraph.metadata.MetaEnvelope` (the G8 live re-seed — edit an upstream node
and every downstream auto pill updates). Field sockets draw as a diamond, single-value
as a circle, the Dataset main wire as a larger dot — colors from
:data:`nodegraph.sockets.SOCKET_COLOR`.

Directive guards (V2.03 H11 / G8): the 2D/3D switch is DISABLED when the incoming
envelope's z is known to be 1 (unknown ≠ 1 — an unresolved source never greys it), and
a node locked to 3D while z==1 paints a red validation badge. Muted nodes (G3) dim and
tag; pressing a socket starts a wire drag (G1, handled by the scene).

**Run state (2026-07-28).** Every card wears a 2 px accent rail on its header's bottom
edge plus a status dot at the header's right, fed by
:class:`~nodelab_v2.runner.EngineRunner`'s per-node events (:meth:`set_run_state`):
*queued* = a hollow dot, *running* = a pulsing dot over a rail that fills when the
compute reports ``ctx.progress`` and sweeps when it does not, *done* = a solid glowing
dot over a full rail, *cached* = a hollow ring (nothing was recomputed), *error* = red.
A working card also takes an accent border + outer glow, and its outgoing wires flow.
The exact percentage / wall time lives in the card's **tooltip** and the status bar, so a
busy canvas reads as light rather than as a wall of tiny text. A lazy node finishes in
microseconds by design — its rail flashes and the real cost appears on whichever node
actually reads the planes, which is the truth, not a bug.

**Delete affordance.** Hovering a card reveals a ✕ badge on its top-right corner
(:class:`CloseItem`); clicking it asks the scene to delete that node. The keyboard
``Del`` and the right-click menu do the same thing — the badge is the discoverable one.

**Editing on the canvas (V2.16).** The value pills are live controls, Blender-style: drag a
number sideways to scrub it, click it to type, click a bool to toggle, click a mode (or a
closed-``choices`` param) for its menu, click the ``ƒmd`` badge to pin or unpin. A param
declaring ``SocketSpec.pick_kind`` also grows a ◎ glyph that arms the viewer gesture. They
are painted, hit-tested controls rather than embedded widgets: a card is drawn dozens of
times per canvas at arbitrary zoom, and one ``QGraphicsProxyWidget`` per parameter would
cost a real widget per row for something that has to paint at 60 fps while wires animate.
The consequence is that geometry has to be derivable OUTSIDE ``paint()`` — see
:meth:`NodeItem.controls`, which is the single source of both the hit rects and the painted
ones, so what you can click is by construction what you can see.
"""
from __future__ import annotations

import html
import re
import textwrap
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QLineEdit, QMenu

from nodegraph.domains import domain_abbr
from nodegraph.groups import group_name_of
from nodegraph.metadata import envelope_symbols, eval_derive
from nodegraph.registry import NODES
from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T
from nodelab_v2.document import (
    BATCH_OP, BUNDLE_PATHS_KEY, GraphDocument, NodeRecord, TITLE_KEY, UNBATCH_OP)
from nodelab_v2.ops import (DOCK_OP, PAGE_INPUT_OP, PAGE_NAME_KEY, PAGE_OUTPUT_OP,
                            PAGE_SOURCE_KEY)
from nodelab_v2.picker import PICK_ACTION, PICK_GLYPH, request_for

#: a synthetic per-channel output socket name — ``ch0``, ``ch1``, …
_CH_SOCKET_RE = re.compile(r"^ch(\d+)$")


def desc_qcolor(desc) -> QColor:
    """The display colour of one channel descriptor (``{name, emission_nm, color}``): its
    native colour when the file carries one, else the colour of its emission wavelength
    (a neutral grey when unknown — a TIFF, a transmitted-light channel). ONE function for
    the socket dot, the wire and the per-channel sockets, so they cannot disagree."""
    col = (desc or {}).get("color")
    if isinstance(col, (list, tuple)) and len(col) == 3:
        return QColor(int(col[0]), int(col[1]), int(col[2]))
    return T.emission_qcolor((desc or {}).get("emission_nm"))


def avg_qcolor(cols: Sequence[QColor]) -> QColor:
    """The component-wise mean of several QColors — the aggregate tint of a stream carrying
    a multi-channel subset."""
    n = max(1, len(cols))
    return QColor(round(sum(c.red() for c in cols) / n),
                  round(sum(c.green() for c in cols) / n),
                  round(sum(c.blue() for c in cols) / n))

#: Hard-wrap column for tooltip prose. See :func:`socket_hover_text` for why the wrapping
#: is done here rather than left to Qt.
_TIP_COLS = 76

#: The ◎ glyph's box, painted just left of a pickable param's pill.
_PICK_GLYPH_W = 15.0

#: Widest a value pill may grow. Leaves room on a 214 px card for the row label and, on a
#: pickable param, the ◎ glyph. Longer values elide — the full text is in the inspector.
_PILL_MAX_W = 116.0
#: Cap for the granularity band's population pill (V2.27) — tighter than a row pill's, so it
#: clears the footprint chip to its left. The vocabulary is capped to match
#: (``_shared.scope.SCOPE_TOKEN_MAX``); a longer token elides rather than overrunning the chip.
_FOOT_PILL_MAX_W = 96.0

#: Screen pixels of horizontal drag per step while scrubbing a value pill. Small enough that
#: a deliberate nudge moves one step, large enough that crossing the card is not a hundred.
_SCRUB_PX = 3.0

#: How far the pointer may travel and still count as a click (opening the text editor)
#: rather than a scrub. Matches the usual platform drag threshold.
_CLICK_SLOP = 3.0

#: Per-step increment for a scrub, chosen from the socket's type and unit. A physical length
#: in µm and a normalized 0–1 threshold want very different granularity, and the alternative
#: — one step for everything — makes one of them unusable. Shift divides by 10, Ctrl
#: multiplies by 10, so the table only has to be right about the middle case.
_SCRUB_STEP = {
    "px": 1.0, "um": 0.01, "um_axial": 0.01, "um2": 1.0, "um3": 1.0,
    "nm": 1.0, "s": 0.01,
}
#: Step for a unitless FLOAT — thresholds, weights, fractions, almost all 0–1.
_SCRUB_STEP_UNITLESS = 0.005

#: Decimals a FLOAT editor shows unless its magnitude demands more, and the widest step it
#: will ever take. Floors, not fixed values — see :func:`value_step`.
_FLOAT_DECIMALS = 3
_FLOAT_STEP = 0.05

#: Editor bounds. Deliberately far past anything the catalog means, because the WIDGET must
#: never be the thing that limits a value — the compute validates, and a silent clamp is the
#: worst way to be told no. The old caps (1e6 float / 1e5 int) were reachable in practice:
#: an `iterations` of 200000 and a `max_area` on a millimetre-scale colony both hit them and
#: were quietly truncated to the ceiling.
NUM_MAX_FLOAT = 1e12
NUM_MAX_INT = 2_000_000_000             # just under Qt's INT_MAX for QSpinBox


def _magnitude(v) -> Optional[float]:
    """``abs(float(v))`` when that is a usable positive magnitude, else ``None``."""
    try:
        f = abs(float(v))
    except (TypeError, ValueError):
        return None
    return f if f > 0.0 else None


def value_step(spec, current=None, *, integer: bool = False) -> float:
    """One increment for a numeric socket, from its UNIT **and** its MAGNITUDE.

    The unit table stays authoritative — it encodes the granularity the quantity is measured
    in (0.01 µm, 1 px, 0.01 s), and that is a property of the physics, not of the current
    number. Magnitude only **bounds** it, which is what fixes the two symmetric pathologies
    a fixed step has at the extremes:

    * **too coarse for a small value** — a 0.005 step on a 5e-5 learning rate overshoots it
      100× on the first pixel of drag, and the inspector's 0.05 could not express it at all
      (it rounded to ``0.000``, so the widget displayed zero for a non-zero default). Already
      true before this node of ``dic_correlate``'s ``strain_smoothness`` (1e-5) and
      ``disp_smoothness`` (5e-4) and both solvers' ``mu``;
    * **too fine for a large one** — 0.05 on a 2047.5 threshold, or 1 on a 33 000-iteration
      training budget, is tens of thousands of clicks to cross, so the control is decorative.

    The bound is one order of magnitude either side of the value: never coarser than the
    value itself, never finer than a tenth of its leading order. So ``sigma`` at 0.5 µm keeps
    its 0.01 table step exactly, while ``iterations`` at 2000 steps by 100 and a 5e-5 rate
    steps by 1e-5.

    Presentation-only: nothing here reaches ``node_recipe_hash``, so retuning an editor
    cannot invalidate a memo entry or a saved graph.
    """
    base = 1.0 if integer else _SCRUB_STEP.get(getattr(spec, "unit", "") or "",
                                               _SCRUB_STEP_UNITLESS)
    mag = _magnitude(current)
    if mag is None:
        mag = _magnitude(getattr(spec, "default", None))
    if mag is None:
        return base
    import math
    order = 10.0 ** math.floor(math.log10(mag))
    step = min(max(base, order / 10.0), order)
    return max(1.0, float(int(step))) if integer else step


def value_decimals(spec, current=None) -> int:
    """Decimals a FLOAT editor needs: enough for the FINEST value in play, floored at 3.

    Driven by the smallest magnitude of (current, default) rather than by the current value
    alone, so a socket whose default is 5e-5 stays typable after the user has entered 0.5 and
    wants to go back. Floored at the historic 3 rather than reduced for large values: showing
    ``2047.500`` is only cosmetic noise, whereas dropping below 3 decimals would make ``0.001``
    unreachable on a socket that had merely been left at a big number.
    """
    mags = [m for m in (_magnitude(current),
                        _magnitude(getattr(spec, "default", None))) if m is not None]
    if not mags:
        return _FLOAT_DECIMALS
    import math
    step = value_step(spec, min(mags))
    return max(_FLOAT_DECIMALS, min(9, int(math.ceil(-math.log10(step)))))


def float_editor_precision(spec, current=None) -> Tuple[int, float]:
    """``(decimals, step)`` for a FLOAT spin box — :func:`value_decimals` +
    :func:`value_step`. Decimals follow the finest value in play; the step follows where the
    value is NOW, so stepping feels right without anything becoming untypable."""
    return value_decimals(spec, current), value_step(spec, current)


class Ctl(NamedTuple):
    """One hit-testable control on a card: where it is, what it does, what it edits.

    ``kind`` is ``"value"`` (a param pill), ``"mode"`` (a mode pill), ``"pick"`` (the ◎
    glyph), ``"pin"`` (the ƒmd badge) or ``"scope"`` (the population pill in the granularity
    band — V2.27, the only control not derived from a row's y). ``obj`` is the ``SocketSpec``
    or ``ModeSpec`` behind it."""

    rect: QRectF
    kind: str
    obj: object


class _InlineEdit(QLineEdit):
    """The transient text box a clicked pill opens.

    Parented to the *view's viewport* rather than embedded in the scene as a proxy widget:
    it exists for one edit, it must not scale with the canvas zoom into something unreadable,
    and it must not become part of the card's geometry. Esc abandons; Enter and focus-out
    commit, so clicking away is a save rather than a silent discard."""

    def __init__(self, parent, text: str, on_commit, on_cancel) -> None:
        super().__init__(text, parent)
        self._on_commit = on_commit
        self._on_cancel = on_cancel
        self._closed = False
        self.setAlignment(Qt.AlignCenter)
        self.selectAll()
        self.editingFinished.connect(self._commit)

    def _commit(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_commit(self.text())
        self.deleteLater()

    def keyPressEvent(self, e) -> None:                 # noqa: N802 - Qt naming
        if e.key() == Qt.Key_Escape:
            self._closed = True
            self._on_cancel()
            self.deleteLater()
            e.accept()
            return
        super().keyPressEvent(e)


def socket_identity(spec) -> str:
    """The one-line identity of a socket: ``name — type · unit · multi``.

    Split out of :class:`SocketItem` so the inspector's parameter rows and the card's
    port dots put the SAME first line above the same prose — two hover surfaces for one
    socket that disagreed would be worse than one.
    """
    kind = ("Dataset" if spec.type is SocketType.DATASET
            else (f"field · {spec.type.value}" if spec.is_field else spec.type.value))
    tip = f"{spec.name} — {kind}"
    if spec.unit:
        tip += f" · {spec.unit}"
    if getattr(spec, "multi", False):
        tip += " · multi"
    return tip


def _esc(s) -> str:
    """HTML-escape tooltip text. ``quote=False``: this text is element CONTENT, never an
    attribute value, so escaping apostrophes would only turn every ``CellSAM's`` into
    ``&#x27;`` noise in the markup."""
    return html.escape(str(s), quote=False)


def _wrapped(text: str, indent: str = "") -> List[str]:
    """``text`` as escaped, hard-wrapped tooltip lines (blank-line-separated paragraphs
    preserved). ``indent`` hangs the continuation lines of a bulleted option line under its
    bullet.

    The indent is wrapped as ordinary spaces and only then converted to ``&nbsp;`` — HTML
    collapses a leading run of real spaces to nothing, and pre-substituting the entity would
    make ``textwrap`` count 12 characters where the reader sees 3 (and ``html.escape`` would
    turn the ``&`` into ``&amp;`` anyway). Converting only the LEADING run leaves the prose
    itself untouched, so a description that happens to contain a double space is not
    rewritten."""
    out: List[str] = []
    for para in str(text).split("\n\n"):
        for w in textwrap.wrap(" ".join(para.split()), _TIP_COLS,
                              subsequent_indent=indent):
            body = w.lstrip(" ")
            out.append("&nbsp;" * (len(w) - len(body)) + _esc(body))
    return out


def choice_doc_lines(spec_or_mode, options: Sequence[str] = ()) -> List[str]:
    """The documented options of a dropdown, as ``• name — prose`` tooltip lines.

    Shared by every hover surface a dropdown has, because they must not disagree: the
    inspector's row label, the node card's port dot (a ``choices`` socket HAS a port dot,
    and it is the only place on the card where its options can be read without opening the
    menu) and the Mode row all render the same block. ``options`` narrows the list — the
    channel/layer pickers pass the live set — and defaults to everything declared.
    Undocumented options are listed bare rather than skipped: a gap the user can see is
    better than an option that looks like it does not exist."""
    docs = getattr(spec_or_mode, "choice_docs", None) or {}
    opts = list(options) or (list(getattr(spec_or_mode, "choices", ()) or ())
                             + list(getattr(spec_or_mode, "vocab", ()) or ()))
    if not opts or not docs:
        return []
    lines: List[str] = []
    for name in opts:
        doc = str(docs.get(name, "") or "").strip()
        lines.extend(_wrapped(f"• {name} — {doc}" if doc else f"• {name}", indent="   "))
    return lines


def socket_hover_text(spec, extra: Sequence[str] = (), *, head: str = "") -> str:
    """Build the hover tooltip for one socket: identity line, any ``extra`` structural
    lines, :attr:`~nodegraph.registry.SocketSpec.description` as wrapped prose, then one
    line per documented option (:attr:`~nodegraph.registry.SocketSpec.choice_docs`).

    **Rich text on purpose, hard-wrapped on purpose.** Qt's tooltip label turns word wrap
    on only when the text looks like rich text (``Qt::mightBeRichText``), and even then it
    wraps at the *screen* edge — a 400-character parameter note would render as one
    unreadable full-width line. Qt also supports only a subset of CSS in tooltips, so
    ``max-width`` is not dependable. Wrapping here at :data:`_TIP_COLS` and joining with
    ``<br>`` gives a fixed, readable column on every platform and is checkable headlessly
    (``nodegraph.selftest`` asserts the CellSAM sockets carry prose; the GUI probe asserts
    it reaches the widgets).

    Text is HTML-escaped, so a description may contain ``<``, ``>`` or ``&`` freely.
    """
    lines = [f"<b>{_esc(head or socket_identity(spec))}</b>"]
    lines.extend(_esc(x) for x in extra)
    # An interactive param advertises its gesture on every hover surface it has — the card's
    # port dot and the inspector row both come through here, so the affordance is discovered
    # by reading about the param rather than by noticing a small glyph.
    pick = getattr(spec, "pick_kind", "")
    if pick:
        lines.append(f"{PICK_GLYPH} {PICK_ACTION.get(pick, 'Pick')} — the ring on the "
                     f"card, or Pick in the properties panel")
    desc = (getattr(spec, "description", "") or "").strip()
    if desc:
        lines.append("")                      # blank line between identity and prose
        lines.extend(_wrapped(desc))
    opts = choice_doc_lines(spec)
    if opts:
        lines.append("")
        lines.extend(opts)
    return "<br>".join(lines)


def mode_identity(m) -> str:
    """The one-line identity of an in-body Mode: ``name — mode · N options``.

    The two ROLES name themselves, so the pill, the menu and the inspector row all label the
    control the same way: the 2D/3D lever, and (V2.27) the statistics population, which is
    edited from the card's granularity band rather than from a body row."""
    if getattr(m, "is_dim_lever", False):
        kind = "2D / 3D lever"
    elif getattr(m, "is_scope", False):
        kind = "statistics population"
    else:
        kind = "mode"
    return f"{m.name} — {kind} · {len(m.choices)} options"


def mode_hover_text(m, extra: Sequence[str] = ()) -> str:
    """The hover tooltip for a Mode dropdown: identity, ``description``, then one line per
    option (V2.21).

    A Mode is a param the user cannot read off the card — the pill shows the CURRENT value
    and nothing about the alternatives — so the option block is the substance here, not a
    garnish. Same builder for the inspector's Mode row and the 2D/3D switch, so the two
    cannot drift."""
    lines = [f"<b>{_esc(mode_identity(m))}</b>"]
    lines.extend(_esc(x) for x in extra)
    desc = (getattr(m, "description", "") or "").strip()
    if desc:
        lines.append("")
        lines.extend(_wrapped(desc))
    opts = choice_doc_lines(m, list(m.choices))
    if opts:
        lines.append("")
        lines.extend(opts)
    return "<br>".join(lines)


def option_hover_text(option: str, doc: str, *, head_note: str = "") -> str:
    """The tooltip for ONE option, as shown on a combo item or a popup-menu action.

    Returns ``""`` when the option is undocumented, so a caller can set it unconditionally:
    an empty tooltip leaves Qt's default behaviour (no popup, event propagates to the
    parent widget) rather than flashing an empty box."""
    doc = str(doc or "").strip()
    if not doc:
        return ""
    lines = [f"<b>{_esc(option)}</b>" + (f" — {_esc(head_note)}" if head_note else "")]
    lines.extend(_wrapped(doc))
    return "<br>".join(lines)


# ── socket ────────────────────────────────────────────────────────────────────

class SocketItem(QGraphicsItem):
    """A single socket dot; its ``scenePos()`` is the wire anchor. Pressing it begins
    a wire drag (delegated to the scene); ``highlight`` paints the drop-target ring."""

    R = 9.0

    def __init__(self, parent: "NodeItem", spec, io: str) -> None:
        super().__init__(parent)
        self.spec = spec
        self.io = io                       # "in" | "out"
        self.node_item = parent
        self.channel_color: Optional[QColor] = None   # the channel(s) this stream carries
        self.highlight: Optional[bool] = None   # None | True (valid) | False (invalid)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)
        self._head = socket_identity(spec)
        self.setToolTip(socket_hover_text(spec, head=self._head))

    def set_domain_tip(self, domains, missing=(), *, on_wire=(), channel: str = "") -> None:
        """Append the domain-set this Dataset socket carries/requires (the rail's
        hover detail). No-op tail for value sockets (``domains`` empty).

        ``on_wire`` is what THIS socket's edge actually brings — the only per-socket fact
        available on a node with several Dataset inputs, whose painted rail is one node-level
        answer repeated beside each of them (2026-08-04). Listed separately from ``requires``
        rather than replacing it: "what this wire carries" and "what the node needs" are
        different questions, and collapsing them is how the rail came to look like it was
        answering the first while answering the second."""
        extra = []
        if domains:
            names = ", ".join(d.value for d in domains)
            extra.append(("carries: " if self.io == "out" else "the node requires: ") + names)
        if self.io == "in":
            extra.append("on this wire: " + (", ".join(d.value for d in on_wire)
                                             if on_wire else "nothing wired"))
        if missing:
            extra.append("⚠ missing upstream: " + ", ".join(d.value for d in missing))
        if channel:
            # which of the file's channels this stream is (V4.00 step 11g) — the words
            # behind the dot's tint and the row's name
            extra.append("channel: " + channel)
        self.setToolTip(socket_hover_text(self.spec, extra, head=self._head))

    def boundingRect(self) -> QRectF:
        # +1 for the antialiased outer edge of the highlight ring (drawn at radius
        # ``R - 1`` with a 2 px pen, i.e. exactly out to R): without the slack, moving a
        # socket leaves a hairline of the ring behind.
        g = self.R + 1.0
        return QRectF(-g, -g, 2 * g, 2 * g)

    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        col = T.SOCKET[self.spec.type]
        if self.spec.type is SocketType.DATASET and self.channel_color is not None:
            col = self.channel_color               # per-channel output dot tint
        if self.highlight is not None:
            ring = T.WIRE if self.highlight else T.ERROR
            p.setPen(QPen(ring, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(0, 0), self.R - 1, self.R - 1)
        p.setPen(QPen(T.BG, 2))
        if self.spec.type is SocketType.DATASET:
            p.setBrush(col)
            p.drawEllipse(QPointF(0, 0), 6.5, 6.5)
        elif self.spec.is_field:
            path = QPainterPath()
            r = 5.5
            path.moveTo(0, -r); path.lineTo(r, 0); path.lineTo(0, r); path.lineTo(-r, 0)
            path.closeSubpath()
            p.setBrush(col)
            p.drawPath(path)
        else:
            p.setBrush(col)
            p.drawEllipse(QPointF(0, 0), 5.0, 5.0)

    def anchor(self) -> QPointF:
        return self.scenePos()

    def set_highlight(self, state: Optional[bool]) -> None:
        if state != self.highlight:
            self.highlight = state
            self.update()

    def mousePressEvent(self, e) -> None:
        sc = self.scene()
        if e.button() == Qt.LeftButton and sc is not None and hasattr(sc, "begin_wire"):
            sc.begin_wire(self, e.scenePos())
            e.accept()
            return
        super().mousePressEvent(e)


# ── 2D / 3D switch ──────────────────────────────────────────────────────────────

class SwitchItem(QGraphicsObject):
    """The two-color on/off switch — amber for 2D, cyan for 3D (both active states).
    ``allow_3d=False`` (incoming z known == 1, H11) greys the 3D side and ignores
    clicks toward 3D."""

    toggled = Signal(str)
    TRACK_W, TRACK_H, KNOB = 34.0, 18.0, 14.0
    LBL_W = 17.0
    WIDTH = LBL_W + 5 + TRACK_W + 5 + LBL_W

    def __init__(self, parent: "NodeItem", dim: str) -> None:
        super().__init__(parent)
        self.dim = dim
        self.allow_3d = True
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self._refresh_tip()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.WIDTH, 20)

    def _refresh_tip(self) -> None:
        """The lever is a Mode like any other, so it hovers like any other (V2.21): what the
        switch decides, what each side means, and the z == 1 refusal when it applies. Before
        this the refusal was the ONLY thing it ever said, which left the most consequential
        control on the card — it reconfigures sockets and forks the memo key — as the least
        documented one.

        The ModeSpec is looked up LIVE (and re-read on hover) rather than captured in
        ``__init__``, for the same reason ``NodeItem.spec`` is a property: a hot reload can
        rewrite a node's lever, and a cached tooltip would keep quoting the old text."""
        extra = () if self.allow_3d else (
            "⚠ 3D is unavailable: the incoming data has z == 1 (H11)",)
        node = self.parentItem()
        spec = getattr(node, "spec", None)
        mode = spec.dim_lever() if spec is not None else None
        self.setToolTip(mode_hover_text(mode, extra) if mode is not None
                        else (extra[0] if extra else ""))

    def set_allow_3d(self, allow: bool) -> None:
        if allow != self.allow_3d:
            self.allow_3d = allow
            self._refresh_tip()
            self.update()

    def hoverEnterEvent(self, e) -> None:                # noqa: N802 - Qt naming
        self._refresh_tip()
        super().hoverEnterEvent(e)

    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        on = self.dim == "3D"
        f = QFont(T.MONO, 7); f.setBold(True)
        p.setFont(f)
        p.setPen(T.INK if not on else T.MUTED)
        p.drawText(QRectF(0, 0, self.LBL_W, 20), Qt.AlignCenter, "2D")
        p.setPen((T.INK if on else T.MUTED) if self.allow_3d else T.alpha(T.MUTED, 110))
        p.drawText(QRectF(self.WIDTH - self.LBL_W, 0, self.LBL_W, 20),
                   Qt.AlignCenter, "3D")
        tx = self.LBL_W + 5
        track = QRectF(tx, 1, self.TRACK_W, self.TRACK_H)
        col = T.ACCENT if on else T.DIM2D
        if not self.allow_3d and not on:
            col = T.mix(col, T.PANEL, 0.35)
        p.setPen(QPen(col, 1))
        p.setBrush(col)
        p.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        kx = tx + (self.TRACK_W - self.KNOB - 1) if on else tx + 1
        p.setPen(Qt.NoPen)
        p.setBrush(T.ACCENT_INK if on else T.DIM2D_INK)
        p.drawEllipse(QRectF(kx, 2, self.KNOB, self.KNOB))

    def set_dim(self, dim: str) -> None:
        if dim != self.dim:
            self.dim = dim
            self.update()
            self.toggled.emit(dim)

    def mousePressEvent(self, e) -> None:
        target = "3D" if self.dim == "2D" else "2D"
        if target == "3D" and not self.allow_3d:
            e.accept()
            return
        self.set_dim(target)
        e.accept()


# ── hover ✕ delete badge ──────────────────────────────────────────────────────

class CloseItem(QGraphicsObject):
    """The delete badge on a card's top-right corner: visible only while the card is
    hovered (or the badge itself is), so it never competes with the node's content. It
    only *asks* — :class:`NodeItem` re-emits and the scene owns the actual removal, so
    every delete path (badge, ``Del``, context menu) runs the same code."""

    clicked = Signal()

    def __init__(self, parent: "NodeItem") -> None:
        super().__init__(parent)
        self._hot = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.ArrowCursor)
        self.setToolTip("Delete this node (Del)")
        self.setVisible(False)
        self.setZValue(4)          # above the header gradient and the 2D/3D switch

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, T.CLOSE_BTN, T.CLOSE_BTN)

    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5)
        col = T.ERROR if self._hot else T.MUTED
        p.setPen(QPen(col, 1))
        p.setBrush(T.mix(T.PANEL, col, 0.30) if self._hot else T.PANEL)
        p.drawEllipse(r)
        p.setPen(QPen(T.INK if self._hot else T.INK_2, 1.4))
        d = 3.6
        cx, cy = r.center().x(), r.center().y()
        p.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
        p.drawLine(QPointF(cx - d, cy + d), QPointF(cx + d, cy - d))

    def hoverEnterEvent(self, e) -> None:
        self._hot = True
        self.update()
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e) -> None:
        self._hot = False
        self.update()
        # the badge overhangs the card's corner, so leaving it can mean leaving the card
        # too — let the card re-decide (it checks whether anything is still hovered)
        parent = self.parentItem()
        if isinstance(parent, NodeItem):
            parent.hide_close_if_away()
        super().hoverLeaveEvent(e)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            e.accept()               # swallow the press so the card doesn't start moving
            return
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self.boundingRect().contains(e.pos()):
            self.clicked.emit()
            e.accept()
            return
        super().mouseReleaseEvent(e)


# ── node card ────────────────────────────────────────────────────────────────

class NodeItem(QGraphicsObject):
    """A node card bound to a document :class:`NodeRecord`."""

    changed = Signal(object)          # emitted on dim/param/relayout (self)
    delete_requested = Signal(str)    # node_id — the hover ✕ badge was clicked
    #: a ◎ glyph on this card was clicked — carries a
    #: :class:`~nodelab_v2.picker.PickRequest`, re-emitted by the scene up to the window,
    #: which arms the viewer. The same request the inspector's Pick button sends, built by
    #: the same function, so a pick means the same thing from either surface.
    pick_requested = Signal(object)

    def __init__(self, rec: NodeRecord, doc: GraphDocument) -> None:
        super().__init__()
        self.rec = rec
        self.doc = doc
        self._is_reroute = rec.op_key == "rr.reroute"
        # ── the dot family (V3.01) ───────────────────────────────────────────────
        # Three ops render as a circle rather than a card, because none of them is a step
        # in the pipeline the way a card is: a reroute is a bead on a wire, and the two
        # batch nodes are the points where K files become one stream and become K again.
        # Drawing them as cards would make "run these four files" look like four more
        # processing stages. They share the layout and hit-testing; only the size and the
        # paint differ, which is what `_dot_size` is for.
        self._is_batch = rec.op_key in (BATCH_OP, UNBATCH_OP)
        self._is_dot = self._is_reroute or self._is_batch
        self._dot_size = float(T.BATCH_SIZE if self._is_batch else T.RR_SIZE)
        self._group_name = group_name_of(rec.op_key)   # a group instance? → its name
        self._is_group = self._group_name is not None
        self._sockets: Dict[Tuple[str, str], SocketItem] = {}
        self._rows: List[tuple] = []
        self._viewed = False              # the Viewer / mini-map is showing this node
        self._width = float(T.NODE_W)
        self._height = float(T.HEADER_H)
        self._switch: Optional[SwitchItem] = None
        # run state (fed by the runner's per-node events; see set_run_state)
        self._run = ""                    # "" | queued | running | decoding | cached
        self._run_frac: Optional[float] = None    # None → indeterminate
        self._run_note = ""
        self._run_secs: Optional[float] = None
        # the two-level split, when the compute reports one (else all None → one flat rail)
        self._frame_frac: Optional[float] = None  # frames finished / total frames
        self._sub_frac: Optional[float] = None    # work done inside the frame in flight
        self._frames: Optional[int] = None
        self._frame: Optional[int] = None
        self._phase = 0.0                 # marquee position for the indeterminate sweep
        # on-canvas editing: the control under the pointer (hover highlight) and the live
        # scrub, as (ctl, press_x, value_at_press, has_moved)
        self._hot_ctl: Optional[Ctl] = None
        self._scrub: Optional[tuple] = None
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        if self.spec is not None and self.spec.has_dim_lever():
            self._switch = SwitchItem(self, self.dim)
            self._switch.setPos(T.NODE_W - SwitchItem.WIDTH - 10, 9)
            self._switch.toggled.connect(self.set_dim)
        self._close = CloseItem(self)
        self._close.clicked.connect(lambda: self.delete_requested.emit(self.rec.id))
        self._shown_collapsed = rec.collapsed
        self.setPos(rec.x, rec.y)
        self._layout()

    # ── model passthroughs ────────────────────────────────────────────────────
    @property
    def spec(self):
        """This node type's :class:`NodeSpec`, read LIVE from the registry every time.

        Deliberately not cached on the item. It used to be, captured once in ``__init__``,
        and that made every consumer of a card — the inspector above all, which builds its
        whole parameter panel from ``node.spec`` — quietly dependent on some reload path
        having remembered to re-assign it. Miss one path and the symptom is a node whose
        source you have just edited still showing its old parameters, with no error and
        nothing to click that would fix it; re-selecting the node does not help, because
        selection rebinds the panel to the same card holding the same stale object.

        A registry lookup is one dict access, so there is nothing to cache. Relayout is a
        separate matter and still explicit — see :meth:`resync_spec` — because a changed
        socket set has to move the ports the wires land on."""
        return NODES.get(self.rec.op_key)

    @property
    def node_id(self) -> str:
        return self.rec.id

    @property
    def op_key(self) -> str:
        return self.rec.op_key

    @property
    def params(self) -> dict:
        return self.rec.params

    @property
    def dim(self) -> str:
        st = self.rec.state()
        return st.get("dim", "2D")

    @property
    def locked(self) -> set:
        return self.rec.locked

    def env(self):
        return self.doc.env(self.rec.id)

    def state(self) -> dict:
        return self.rec.state()

    def _active_inputs(self):
        return self.doc.input_specs(self.rec.id)

    def _active_outputs(self):
        # instance-aware: a source/split card grows one synthetic ``chK`` output per
        # channel (the document owns the resolution + the channel descriptors).
        return self.doc.output_specs(self.rec.id)

    def _active_modes(self):
        """The Modes this instance shows — gated on the current mode state, so a Mode the
        selected method never reads is absent from the card as well as the inspector
        (V2.12 ``ModeSpec.available_in``)."""
        return self.spec.active_modes(self.rec.state()) if self.spec else ()

    # ── per-channel outputs (chK) ─────────────────────────────────────────────
    @staticmethod
    def output_channel_index(socket_name: str) -> Optional[int]:
        """The channel index K if ``socket_name`` is a synthetic ``chK`` output, else
        ``None``. Used by the edge painter to tint the wire by that channel's color."""
        m = _CH_SOCKET_RE.match(socket_name)
        return int(m.group(1)) if m else None

    def channel_qcolor(self, index: int) -> QColor:
        """The display color of channel ``index``: its native color if the descriptor
        carries one, else a color derived from its emission wavelength (neutral grey
        when unknown — e.g. a TIFF or a transmitted-light channel)."""
        descs = self.doc.channel_descriptors(self.rec.id)
        if 0 <= index < len(descs):
            return desc_qcolor(descs[index])
        return T.SOCKET[SocketType.DATASET]

    def set_viewed(self, on: bool) -> None:
        """Flag this card as the one the Viewer is showing (accent spine + live dot)."""
        on = bool(on)
        if on != self._viewed:
            self._viewed = on
            self.update()

    def granularity(self) -> str:
        """The resolved read footprint's name, or ``""`` when it does not resolve.

        The empty string is the V2.27 correction: this returned the literal ``"tileable"``,
        so a node whose footprint resolved to ``None`` painted the GREEN chip — "cheapest
        possible read" — for a footprint that is unknown. It is false twice over on the two
        nodes that hit it: ``io.load`` reads from disk and ``view.viewer`` is a sink. The band
        now renders ``—`` in the muted colour instead (:meth:`_foot_slots`)."""
        g = self.spec.resolve_granularity(self.state()) if self.spec else None
        return g.value if g is not None else ""

    def _scope_mode(self):
        """The statistics-population Mode this card's footprint band EDITS, or ``None`` when
        the band is a plain readout (V2.27).

        The band is a control iff the node declares a ``role="scope"`` Mode and that Mode is
        active in the current state. Deliberately NOT keyed on ``spec.footprint_mode``: on
        three of the four nodes that declare a non-default one, that Mode is not a population —
        ``util.zproject``'s is ``method``, so a band bound to it would offer max/mean/**none**,
        i.e. a footprint control that changes the reducer or switches the node off. And all
        four already show that Mode as a body pill, so it would be a duplicate control.

        Duck-typed against the registry on purpose: the GUI must stay inert-but-correct on a
        build whose engine half is older, and a hot-reloaded node must never be able to crash
        a paint."""
        spec = self.spec
        if spec is None or self._is_group or self._is_dot or self.rec.collapsed:
            return None
        fn = getattr(spec, "scope_mode", None)
        m = fn() if callable(fn) else None
        if m is None:
            m = next((x for x in spec.modes if getattr(x, "is_scope", False)), None)
        if m is None or not getattr(m, "choices", ()):
            return None
        return m if m in spec.active_modes(self.state()) else None

    def _foot_slots(self) -> Tuple[str, str, Optional[str]]:
        """``(caption, chip_text, pill_text | None)`` — THE source of the footprint band's
        strings, read by :meth:`paint` AND by :meth:`controls`.

        One function for the same reason the pill rects are shared: the probe has to be able to
        assert the band headlessly, and a second derivation is how a chip comes to disagree
        with the menu behind it.

        The chip stays a FACT (the read cost, in the cost colour) and the pill is the CONTROL —
        the card's existing grammar, and what keeps the band honest: a per-label population
        still reads a whole volume, so painting ``PER LABEL`` in the green TILEABLE colour would
        assert exactly the misdeclaration ``analysis.threshold`` was corrected for. When a pill
        is present the chip ABBREVIATES, because ``WHOLE VOLUME`` ends at x≈129 and the pill
        starts at 106."""
        if self._is_group:
            return "subgraph", "GROUP", None
        gname = self.granularity()
        m = self._scope_mode()
        if m is None:
            return "footprint", (gname.replace("_", " ").upper() if gname else "—"), None
        val = self.rec.modes.get(m.name, m.resolved_default())
        return ("footprint", (T.gran_abbr(gname) if gname else "—"),
                f"{str(val).replace('_', ' ')} ▾")

    # H11: the incoming z is KNOWN to be 1 (unknown never counts as 1)
    def z_is_one(self) -> bool:
        env = self.env()
        return env.axes.z == 1 and "z" not in env.unknown_axes

    def dim_invalid(self) -> bool:
        """A 3D lever on known z==1 data — the H11 red-badge validation error."""
        return (self.spec is not None and self.spec.has_dim_lever()
                and self.dim == "3D" and self.z_is_one())

    # ── domain interface (socket rail + wire tint) ────────────────────────────
    def _dsorted(self, domains) -> tuple:
        return tuple(sorted(domains, key=lambda d: d.value))

    def reads_domains(self) -> tuple:
        """Domains this node requires on its Dataset input (the input rail).

        Resolved against the live mode state (V2.22): a node whose branches read different
        structure — Voronoi Cells clipping to a Label instance under ``per_region`` and to
        nothing under ``frame`` — shows the rail for the branch that is actually selected,
        not a static worst case that is wrong in every state."""
        return (self._dsorted(self.spec.resolve_reads_domains(self.state()))
                if self.spec else ())

    def out_domains(self) -> tuple:
        """The accumulated domain-set flowing out of this node (the output rail +
        the tint of every wire leaving it)."""
        return self._dsorted(self.env().domains)

    def missing_domains(self) -> frozenset:
        """Required domains absent upstream — the red validation chips (H-domains)."""
        return self.doc.missing_domains(self.rec.id)

    def trained(self) -> dict:
        """What the MODEL this node has loaded was trained with, keyed by socket name
        (V2.23) — ``{}`` for the great majority of nodes, which load no model.

        Read from the JSON beside the weights, through ``NodeSpec.trained`` (which swallows
        everything, so an unreadable or absent file simply reports nothing). Called on every
        relayout and every inspector rebuild; ``nodegraph.trained`` caches the parse on
        ``(path, mtime, size)``, so the repeated calls cost a ``stat`` rather than a read —
        and keying on mtime means retraining to the same filename shows up immediately.
        """
        return self.spec.trained(self.rec.params, self.state()) if self.spec else {}

    def resolved(self, s) -> object:
        """The pill value: an explicit param, else what the loaded MODEL was trained with,
        else the LIVE metadata-derived value from this node's propagated envelope (G8), else
        the static default.

        The model sits ABOVE ``derive`` in that order deliberately. A ``derive`` is this
        app's inference from the image's metadata; the checkpoint's own record is a
        measured fact about the network that is going to run. Where both speak — the only
        current case is ZS-DeconvNet's ``background``, whose ``derive`` is the constant
        ``0.0`` — the file wins, because a value that disagrees with the trained graph is
        not a different opinion but a wrong answer.
        """
        if s.name in self.rec.params:
            return self.rec.params[s.name]
        tr = self.trained()
        if s.name in tr:
            v = tr[s.name]
            return round(v, 4) if isinstance(v, float) else v
        if s.derive:
            try:
                v = eval_derive(s.derive, envelope_symbols(self.env()))
                if isinstance(v, float):
                    return round(v, 4)
                return v
            except Exception:  # noqa: BLE001 — missing symbols → placeholder
                return "auto"
        return s.default

    def is_derived(self, s) -> bool:
        """Whether this socket is currently on AUTO — i.e. it has a live source other than
        its static default, and the user has not pinned it.

        A model-sourced socket counts, which is what gives it the greyed box and the pin
        button: the two sources differ in where the value comes from, not in how the control
        behaves once it has one.
        """
        if s.name in self.locked or s.name in self.rec.params:
            return False
        return bool(s.derive) or s.name in self.trained()

    # ── layout ──────────────────────────────────────────────────────────────────
    def _apply_domain_tip(self, sock: "SocketItem", s, io: str) -> None:
        if s.type is not SocketType.DATASET:
            return
        if io == "in":
            # `on_wire` is per SOCKET; the requirement is per NODE. Keeping them separate is
            # what lets a two-input node's sockets hover differently without the rail having
            # to attribute a node-level requirement to one wire or the other — an attribution
            # the declarations cannot support (a Label instance's LABEL half is required by
            # the areas wire, but only its VOXEL half is named by a socket).
            try:
                on_wire = self._dsorted(self.doc.socket_domains(self.rec.id, s.name))
            except Exception:  # noqa: BLE001 — a hover must never break a relayout
                on_wire = ()
            sock.set_domain_tip(self.reads_domains(), self.missing_domains(),
                                on_wire=on_wire, channel=self._channel_phrase(s, "in"))
        else:
            sock.set_domain_tip(self.out_domains(), channel=self._channel_phrase(s, "out"))

    def _channel_phrase(self, s, io: str) -> str:
        """``Cy5 — 1 of 3 channels``: the channel(s) a Dataset socket carries when they are a
        strict subset of the file's (V4.00 step 11g); ``""`` otherwise."""
        try:
            descs = self.doc.channel_subset(self.rec.id, s.name, io)
            if not descs:
                return ""
            total = self.doc.source_channel_total(self.rec.id)
        except Exception:  # noqa: BLE001 — a hover must never break a relayout
            return ""
        names = " · ".join(str(d.get("name") or "?") for d in descs)
        return f"{names} — {len(descs)} of {total} channels"

    def _tint_channel_socket(self, sock: "SocketItem", s, io: str = "out") -> None:
        """Tint a Dataset socket's dot by the channel(s) its stream carries: a ``chK`` output
        by channel K; any other Dataset socket — an INPUT too (V4.00 step 11g) — by the
        strict channel subset riding it, one colour or the mean of several, exactly as its
        wire is. A full bundle keeps the plain Dataset dot."""
        idx = self.output_channel_index(s.name) if io == "out" else None
        if idx is not None:
            sock.channel_color = self.channel_qcolor(idx)
            return
        if s.type is not SocketType.DATASET:
            return
        try:
            descs = self.doc.channel_subset(self.rec.id, s.name, io)
        except Exception:  # noqa: BLE001 — a relayout must never break on a tint
            descs = []
        if descs:
            cols = [desc_qcolor(d) for d in descs]
            sock.channel_color = cols[0] if len(cols) == 1 else avg_qcolor(cols)

    def _clear_sockets(self) -> None:
        for sock in self._sockets.values():
            sock.setParentItem(None)
            if self.scene() is not None:
                self.scene().removeItem(sock)
        self._sockets.clear()

    def _modes_on_rows(self, m) -> bool:
        """Whether Mode ``m`` gets a BODY ROW, i.e. its own pill under the sockets.

        The two roles are drawn elsewhere and must be excluded here: the 2D/3D lever is the
        header switch, and the statistics population is the granularity band's pill (V2.27).

        **This predicate has to be used by both `_layout` and `refresh`.** If they disagree,
        ``refresh``'s ``want != have`` is permanently true, so every ``doc.touch()`` — every
        edit anywhere in the graph — destroys and recreates every ``SocketItem`` on this card,
        re-anchors every wire, and drops ``_hot_ctl``/``_scrub`` mid-drag. One function, two
        callers, no way to update one and forget the other."""
        return not m.is_dim_lever and m is not self._scope_mode()

    def _layout(self) -> None:
        self.prepareGeometryChange()
        self._clear_sockets()
        self._rows = []
        self._width = float(T.NODE_W)
        self._place_close()
        if self._is_dot:
            self._layout_reroute()
            return
        if self.rec.collapsed:
            self._layout_collapsed()
            return
        y = T.HEADER_H + T.GRAN_H
        for s in self._active_inputs():
            sock = SocketItem(self, s, "in")
            sock.setPos(0, y + T.ROW_H / 2)
            self._apply_domain_tip(sock, s, "in")
            self._tint_channel_socket(sock, s, "in")
            self._sockets[("in", s.name)] = sock
            self._rows.append(("in", s, y))
            y += T.ROW_H
        for m in self._active_modes():
            if not self._modes_on_rows(m):
                continue
            self._rows.append(("mode", m, y))
            y += T.ROW_H
        for s in self._active_outputs():
            sock = SocketItem(self, s, "out")
            sock.setPos(T.NODE_W, y + T.ROW_H / 2)
            self._apply_domain_tip(sock, s, "out")
            self._tint_channel_socket(sock, s)
            self._sockets[("out", s.name)] = sock
            self._rows.append(("out", s, y))
            y += T.ROW_H
        self._height = y + T.PAD_BOTTOM
        if self._switch is not None:
            self._switch.set_allow_3d(not self.z_is_one())
        self._apply_card_tip()
        self.update()

    def _dot_size_for(self, n_side: int) -> float:
        """The circle's diameter, GROWN so ``n_side`` sockets on one edge stay apart.

        A fixed 44px point fits three members comfortably and ten not at all — at ten the
        sockets are 3px apart and the user cannot tell which wire is which file, which is
        the one thing the Unbatch exists to make visible. So the point grows with its
        member count instead of the sockets crowding: the circle is the affordance, and
        an affordance that stops working at the tenth file is not one.
        """
        base = float(T.BATCH_SIZE if self._is_batch else T.RR_SIZE)
        if n_side <= 1:
            return base
        # the arc carries sockets across the middle 70%, so that span needs the pitch
        needed = (T.SOCKET_PITCH * (n_side - 1)) / 0.7
        return max(base, needed)

    def _layout_reroute(self) -> None:
        """A dot-family node: sockets on the circle's left and right edges, no
        header/rows/switch.

        **Several sockets on a side are spread down the arc**, not stacked at mid-height.
        A reroute has one each way and never notices, but an Unbatch has one output per
        FILE — the whole point of the node — and K wires leaving a single point would be
        indistinguishable from each other exactly where the user needs to see which file
        is which. They are placed on the circle itself so each still reads as belonging to
        the point, rather than drifting off it like a card's socket column.
        """
        ins, outs = list(self._active_inputs()), list(self._active_outputs())
        self._dot_size = size = self._dot_size_for(max(len(ins), len(outs)))
        r = size / 2.0

        def place(specs, io: str, x: float) -> None:
            items = list(specs)
            n = len(items)
            for i, s in enumerate(items):
                sock = SocketItem(self, s, io)
                if n <= 1:
                    y = r
                else:
                    # spread across the middle 70% of the height: the arc's extremes are
                    # where the circle is thinnest, so a socket there reads as off the edge
                    span = size * 0.7
                    y = (size - span) / 2.0 + span * i / (n - 1)
                sock.setPos(x, y)
                self._apply_domain_tip(sock, s, io)
                self._tint_channel_socket(sock, s, io)
                self._sockets[(io, s.name)] = sock

        place(ins, "in", 0.0)
        place(outs, "out", size)
        self._width = size
        self._height = size
        self.update()

    def _layout_collapsed(self) -> None:
        """Compact: header only, ALL active sockets kept (so wires stay valid) but
        stacked at the card edges (Blender-style collapse)."""
        ins, outs = list(self._active_inputs()), list(self._active_outputs())
        band = max(len(ins), len(outs))
        y0 = T.HEADER_H + 8
        for i, s in enumerate(ins):
            sock = SocketItem(self, s, "in")
            sock.setPos(0, y0 + i * 12)
            self._apply_domain_tip(sock, s, "in")
            self._tint_channel_socket(sock, s, "in")
            self._sockets[("in", s.name)] = sock
        for i, s in enumerate(outs):
            sock = SocketItem(self, s, "out")
            sock.setPos(T.NODE_W, y0 + i * 12)
            self._apply_domain_tip(sock, s, "out")
            self._tint_channel_socket(sock, s)
            self._sockets[("out", s.name)] = sock
        self._height = T.HEADER_H + max(0, band) * 12 + 12
        if self._switch is not None:
            self._switch.set_allow_3d(not self.z_is_one())
        # a collapsed card paints no band, so the tooltip is the ONLY footprint readout it has
        self._apply_card_tip()
        self.update()

    def refresh(self) -> None:
        """Re-read model state (envelope/derives/mute/guards) — cheap, no relayout
        unless the active socket set OR the collapsed state changed."""
        # the active MODE list is part of the layout too (V2.12): a method switch that
        # gates a Mode away without changing any socket name must still relayout.
        want = ([s.name for s in self._active_inputs()],
                [s.name for s in self._active_outputs()],
                [m.name for m in self._active_modes() if self._modes_on_rows(m)])
        have = ([k[1] for k in self._sockets if k[0] == "in"],
                [k[1] for k in self._sockets if k[0] == "out"],
                [r[1].name for r in self._rows if r[0] == "mode"])
        collapse_changed = getattr(self, "_shown_collapsed", None) != self.rec.collapsed
        if want != have or collapse_changed:
            self._shown_collapsed = self.rec.collapsed
            self._layout()
        else:
            if self._switch is not None:
                self._switch.set_allow_3d(not self.z_is_one())
                if self._switch.dim != self.dim:
                    self._switch.dim = self.dim
                    self._switch.update()
            # Domain tips are per-WIRE (2026-08-04) and only `_layout` used to set them, so a
            # wiring change that leaves the socket set alone — connecting or unplugging an
            # auxiliary input, or an edit upstream that changes what a branch carries — left
            # every socket hovering the state it had when the card was last laid out.
            # The channel tint is per-wire too (V4.00 step 11g): a card is laid out at the
            # drop and its wire arrives after, so a tint set only in `_layout` would leave
            # the plain dot beside a row that already names the channel. Reset, then re-tint,
            # so a wire moved back to the bundle loses the colour as well. The LIVE specs
            # (the document's, instance-aware) so the synthetic sockets are covered too.
            live = {("in", s.name): s for s in self._active_inputs()}
            live.update({("out", s.name): s for s in self._active_outputs()})
            for (io, name), sock in self._sockets.items():
                spec = live.get((io, name)) or (
                    (self.spec.input(name) if io == "in" else self.spec.output(name))
                    if self.spec else None)
                if spec is not None:
                    self._apply_domain_tip(sock, spec, io)
                    sock.channel_color = None
                    self._tint_channel_socket(sock, spec, io)
                    sock.update()
        self.update()

    def resync_spec(self) -> bool:
        """Rebuild this card's geometry against the current :class:`NodeSpec`.

        :attr:`spec` itself is always live (it reads the registry), so this is about the
        *drawing*: a *live node reload* (:mod:`nodegraph.hotreload`) can add, remove or
        reorder sockets, and the ports the wires terminate on are laid out once. Without a
        relayout the card would keep the old rows and every wire into it would point at where
        its port used to be.

        Returns whether this card's op still exists. A ``False`` means the author deleted the
        node type from under a placed instance; the card is left in place (and paints as
        unknown, exactly as it does for an op-key a saved graph names but this build lacks)
        rather than being silently deleted with its wires, because the fix is usually to put
        the definition back, not to lose the graph."""
        # The 2D/3D lever can appear or vanish with an edit, and the switch is a child item
        # rather than part of the row layout, so it is rebuilt here rather than in _layout.
        wants_switch = self.spec is not None and self.spec.has_dim_lever()
        if wants_switch and self._switch is None:
            self._switch = SwitchItem(self, self.dim)
            self._switch.setPos(T.NODE_W - SwitchItem.WIDTH - 10, 9)
            self._switch.toggled.connect(self.set_dim)
        elif not wants_switch and self._switch is not None:
            self._switch.setParentItem(None)
            self._switch = None
        self._layout()
        self.update()
        return self.spec is not None

    def set_dim(self, dim: str) -> None:
        if dim == self.dim:
            return
        self.rec.modes["dim"] = dim
        if self._switch is not None and self._switch.dim != dim:
            self._switch.dim = dim
            self._switch.update()
        self._layout()
        if self.scene() is not None and hasattr(self.scene(), "reroute"):
            self.scene().reroute()
        self.doc.touch(self.node_id)
        self.changed.emit(self)

    def socket(self, io: str, name: str) -> Optional[SocketItem]:
        return self._sockets.get((io, name))

    def sockets(self) -> List[SocketItem]:
        return list(self._sockets.values())

    # ── painting ────────────────────────────────────────────────────────────────
    #: How far outside the card :meth:`_paint_card_glow` / the reroute ring reach — the
    #: outermost pass is grown by 3.0 with a 1.6 px pen, so ~4 px plus antialiasing.
    GLOW_M = 5.0

    def card_rect(self) -> QRectF:
        """The card's *geometry* — what the node visually occupies, excluding the glow.
        Hit-testing, frame bounds and layout use this; only :meth:`boundingRect` (the
        repaint region) carries the glow margin."""
        return QRectF(0, 0, self._width, self._height)

    #: Offset per card in a file bundle's stack, and how many are drawn (a count is what
    #: the eyebrow is for; the stack only has to say "more than one file").
    STACK_STEP = 4.0
    STACK_MAX = 3

    def stack_depth(self) -> int:
        """How many cards to draw BEHIND this one — 0 for an ordinary node, up to
        :attr:`STACK_MAX` for a **file bundle** (an ``io.load`` carrying several paths).

        A bundle is one card that is really N acquisitions, and nothing else on the canvas
        says so at a glance: the title reads like any other source and the wire leaving it
        looks like one file's. The stack is the cheapest honest signal — it reads as depth
        without claiming a number the eyebrow states exactly."""
        members = self.rec.params.get(BUNDLE_PATHS_KEY)
        if not isinstance(members, (list, tuple)) or len(members) < 2:
            return 0
        return min(self.STACK_MAX, len(members) - 1)

    def boundingRect(self) -> QRectF:
        # MUST cover every pixel paint() touches, glow included. Qt only repaints the
        # boundingRect it is told about, so anything painted outside it is left behind as
        # a smear when the card moves (the trailing outlines while dragging).
        m = self.GLOW_M
        # The bundle stack extends down-and-right only, so the margin is ASYMMETRIC —
        # raising GLOW_M instead would move the glow and the reroute ring with it.
        s = self.stack_depth() * self.STACK_STEP
        return self.card_rect().adjusted(-m, -m, m + s, m + s)

    def shape(self) -> QPainterPath:
        """Clicks and rubber-band selection follow the card, not its glow margin."""
        path = QPainterPath()
        if self._is_dot:
            path.addEllipse(self.card_rect())
        else:
            path.addRoundedRect(self.card_rect(), T.RADIUS, T.RADIUS)
        return path

    def _paint_reroute_progress(self, p: QPainter) -> None:
        """A reroute is too small for a rail — its run state reads as a glowing ring
        (pulsing while it works), the same language as the cards' header dot."""
        if not self._run or self._run == "queued":
            return
        col = self._run_color()
        if self._run in ("running", "decoding"):
            col = T.alpha(col, 96 + int(159 * abs(1.0 - 2.0 * ((self._phase * 1.7) % 1.0))))
        size = self._dot_size
        p.setBrush(Qt.NoBrush)
        for grow, a in ((2.0, 40),):
            p.setPen(QPen(T.alpha(col, a), 2))
            p.drawEllipse(QRectF(1 - grow, 1 - grow, size - 2 + 2 * grow,
                                 size - 2 + 2 * grow))
        p.setPen(QPen(col, 2))
        p.drawEllipse(QRectF(1, 1, size - 2, size - 2))

    def _paint_batch(self, p: QPainter) -> None:
        """The **golden point** — where K files become one stream (Batch) and become K
        again (Unbatch).

        Gold rather than a category colour because it is not a processing step and should
        not read as one; it is the one mark on the canvas that means "several files travel
        this wire". Batch is filled and Unbatch is a ring — the same object, opening —
        so the two ends of a batched run are distinguishable at a glance without a label,
        which matters because they are otherwise identical circles.
        """
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.rec.muted:
            p.setOpacity(0.45)
        size = self._dot_size
        r = size / 2.0
        gold = T.BATCH_GOLD_DIM if (self.rec.muted or self.is_dormant()) else T.BATCH_GOLD
        body = QRectF(1, 1, size - 2, size - 2)
        p.setPen(QPen(T.ACCENT if self.isSelected() else T.BORDER,
                      2 if self.isSelected() else 1))
        p.setBrush(T.PANEL)
        p.drawEllipse(body)
        # The gold mark is a FRACTION of the point, so a grown Unbatch still reads as a
        # golden point rather than a small ring adrift in a large dark disc.
        ir = max(8.0, size * 0.25)
        inner = QRectF(r - ir, r - ir, 2 * ir, 2 * ir)
        p.setPen(Qt.NoPen)
        if self.rec.op_key == BATCH_OP:
            p.setBrush(gold)                       # closing: K files become one stream
            p.drawEllipse(inner)
        else:
            p.setBrush(Qt.NoBrush)                 # opening: one stream becomes K files
            p.setPen(QPen(gold, max(4.0, size * 0.09)))
            p.drawEllipse(inner)
        # the member count, when the wiring says one — the single fact worth reading off
        # this node at a glance, and the one thing its shape cannot say
        n = len(self.doc.batch_member_names(self.rec.id)) if self._is_batch else 0
        if n >= 2:
            p.setPen(QPen(T.INK))
            f = p.font()
            f.setPixelSize(11)
            f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(0, size, size, 14), Qt.AlignHCenter | Qt.AlignTop,
                       f"{n} files")

    def _paint_reroute(self, p: QPainter) -> None:
        """A small rounded dot in the Dataset-socket colour; the in/out SocketItems sit
        on its left/right edges. Selection/mute read the same as a full card."""
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.rec.muted:
            p.setOpacity(0.45)
        r = T.RR_SIZE / 2.0
        p.setPen(QPen(T.ACCENT if self.isSelected() else T.BORDER,
                      2 if self.isSelected() else 1))
        p.setBrush(T.PANEL)
        p.drawEllipse(QRectF(1, 1, T.RR_SIZE - 2, T.RR_SIZE - 2))
        p.setPen(Qt.NoPen)
        p.setBrush(T.SOCKET[SocketType.DATASET])
        p.drawEllipse(QPointF(r, r), 4.5, 4.5)

    # ── dock / dormancy (V2.18) ───────────────────────────────────────────────
    def is_dormant(self) -> bool:
        """True when a dock downstream has taken this node out of play — nothing will
        evaluate it, so the card is drawn dimmed (the same affordance as mute, which is
        the same fact: this node is not part of the run)."""
        return self.rec.id in getattr(self.doc, "dormant", frozenset())

    def dock_status(self) -> str:
        """This card's dock status (``""`` if it is not a dock) — see
        :meth:`nodelab_v2.document.GraphDocument.dock_status`."""
        if self.rec.op_key != DOCK_OP:
            return ""
        try:
            return self.doc.dock_status(self.rec.id)[0]
        except Exception:  # noqa: BLE001 — a card must never fail to paint
            return ""

    def paint(self, p: QPainter, *_a) -> None:
        if self._is_dot:
            # one progress ring for the whole family: a dot is too small for the card's
            # rail, so its run state reads as a glowing ring either way
            self._paint_batch(p) if self._is_batch else self._paint_reroute(p)
            self._paint_reroute_progress(p)
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.rec.muted:
            p.setOpacity(0.45)
        elif self.is_dormant():
            # Dimmer than mute, deliberately: a muted node is one the user switched off
            # and will switch back on, while a dormant one has been *replaced* by a
            # checkpoint. Still legible — the point is that you can read what produced
            # the bake, not that the chain disappears.
            p.setOpacity(0.34)
        spec = self.spec
        cat = "group" if self._is_group else (spec.category if spec else "general")
        hdr = T.category_color(cat)
        rect = QRectF(0, 0, T.NODE_W, self._height)

        failed = self._run == "error"          # this node is where the pull broke
        working = self._run in ("running", "decoding")
        border = T.ERROR if (self.dim_invalid() or failed) else (
            T.ACCENT if self.isSelected() else
            (T.alpha(T.ACCENT, 170) if working else T.BORDER))
        body = rect.adjusted(0.5, 0.5, -0.5, -0.5)
        # A file bundle draws as a STACK: offset cards behind the body, furthest first, so
        # the front card's own fill covers their overlap and no clipping is needed. Painted
        # BEFORE the glow, or the offsets would sit on top of the selection ring's corner.
        depth = self.stack_depth()
        if depth:
            p.setBrush(T.mix(T.PANEL, T.BG, 0.45))
            for k in range(depth, 0, -1):
                d = k * self.STACK_STEP
                p.setPen(QPen(T.alpha(T.BORDER, 150), 1))
                p.drawRoundedRect(body.translated(d, d), T.RADIUS, T.RADIUS)
        if failed or working or self.isSelected():
            self._paint_card_glow(p, body, T.ERROR if failed else T.ACCENT)
        p.setPen(QPen(border, 2 if (self.isSelected() or self.dim_invalid() or failed)
                      else 1))
        p.setBrush(T.PANEL)
        p.drawRoundedRect(body, T.RADIUS, T.RADIUS)

        hpath = QPainterPath()
        hpath.addRoundedRect(QRectF(1, 1, T.NODE_W - 2, T.HEADER_H), T.RADIUS, T.RADIUS)
        hpath.addRect(QRectF(1, T.HEADER_H - T.RADIUS, T.NODE_W - 2, T.RADIUS))
        grad = QLinearGradient(0, 0, 0, T.HEADER_H)
        grad.setColorAt(0, T.mix(T.PANEL, hdr, 0.26))
        grad.setColorAt(1, T.PANEL)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawPath(hpath)
        # the category spine — accent and full-height while this card is the one the
        # Viewer (or the mini-map) is showing
        p.setBrush(T.ACCENT if self._viewed else hdr)
        p.drawRoundedRect(
            QRectF(1, 1, 3, (self._height - 2) if self._viewed else T.HEADER_H), 1.5, 1.5)
        p.setPen(QPen(T.BORDER, 1))
        p.drawLine(QPointF(1, T.HEADER_H), QPointF(T.NODE_W - 1, T.HEADER_H))
        # run overlay: the rail rides the header edge and the dot sits in the header, so
        # both read the same on a full card and on a collapsed one.
        self._paint_header_rail(p)
        self._paint_status_dot(p)

        # header text — ELIDED so long titles never run under the switch (G10)
        title_w = (T.NODE_W - 90 if self._switch is None
                   else T.NODE_W - SwitchItem.WIDTH - 34)
        cat_f = QFont(T.SANS, 6); cat_f.setBold(True)
        cat_f.setCapitalization(QFont.AllUppercase)
        cat_f.setLetterSpacing(QFont.PercentageSpacing, 112)
        p.setFont(cat_f)
        p.setPen(T.ACCENT if self._viewed else hdr)
        # The eyebrow shows STATUS when there is one, and the category otherwise. Status
        # wins because it is the volatile fact — the category is already carried by the
        # header tint and the spine, permanently — and because the two together do not
        # fit: on a card with a 2D/3D switch the line is ~26 chars, so
        # "ENHANCEMENT · DORMANT" elided to "ENHANCEMENT · DORM…" and the one word the
        # user needed to read was the one that got cut.
        dock = self.dock_status()
        flags = []
        if self.rec.muted:
            flags.append("MUTED")
        if self.is_dormant():
            flags.append("DORMANT")
        if dock in ("docked", "stale", "unbaked"):
            flags.append(dock.upper())
        # The stack behind a bundle says "several"; this says how many. It is a property of
        # the card rather than a volatile status, so it yields to a real status (MUTED,
        # DORMANT, a dock state) on the one line they share.
        members = self.rec.params.get(BUNDLE_PATHS_KEY)
        if isinstance(members, (list, tuple)) and len(members) >= 2 and not flags:
            flags.append(f"{len(members)} FILES")
        tag = ("● " if self._viewed else "") + ("  ·  ".join(flags) if flags else cat)
        if dock == "stale":
            p.setPen(T.DIM2D)
        elif dock == "unbaked":
            p.setPen(T.ERROR)
        elif flags and not self._viewed:
            p.setPen(T.MUTED)              # a status is not a category — don't tint it
        p.drawText(QRectF(12, 5, title_w, 11), Qt.AlignVCenter | Qt.AlignLeft, tag)
        tf = QFont(T.SANS, 9); tf.setBold(True)
        p.setFont(tf)
        p.setPen(T.ERROR if (self.dim_invalid() or self._boundary_broken()) else T.INK)
        # a source node titled with its loaded file name (a group with its group name,
        # else the node-type label)
        label = (self._group_name or self.rec.params.get(TITLE_KEY)
                 or self._page_boundary_label()
                 or (spec.label if spec else self.rec.op_key))
        label = QFontMetricsF(tf).elidedText(label, Qt.ElideRight, title_w)
        p.drawText(QRectF(12, 15, title_w, 16), Qt.AlignVCenter | Qt.AlignLeft, label)

        if self.rec.collapsed:
            return                     # header-only compact card (sockets at edges)

        # The granularity band: a footprint CHIP (a fact, coloured by read cost) and — when the
        # node declares a statistics population — a PILL beside it (the control). Both strings
        # come from `_foot_slots` so the menu can never disagree with what is painted.
        foot_txt, chip_txt, pill_txt = self._foot_slots()
        gcol = (T.category_color("group") if self._is_group
                else T.gran_color(self.granularity()))
        p.setFont(QFont(T.SANS, 7))
        p.setPen(T.MUTED)
        p.drawText(QRectF(12, T.HEADER_H, 56, T.GRAN_H - 8), Qt.AlignVCenter, foot_txt)
        cf = QFont(T.MONO, 6); cf.setBold(True)
        p.setFont(cf)
        chip = self._gran_chip_rect(chip_txt)
        p.setPen(QPen(T.alpha(gcol, 120), 1))
        p.setBrush(T.alpha(gcol, 36))
        p.drawRoundedRect(chip, 4, 4)
        p.setPen(gcol)
        p.drawText(chip, Qt.AlignCenter, chip_txt)
        if pill_txt:
            # the pill palette, NOT the cost colour: a per-label population still reads a whole
            # volume, so tinting the control by cost would assert a footprint it does not have.
            self._paint_pill_at(
                p, self._foot_pill_rect(pill_txt, chip_txt), pill_txt, "", False,
                hot=(self._hot_ctl is not None and self._hot_ctl.kind == "scope"))
        p.setPen(QPen(T.BORDER, 1, Qt.DashLine))
        p.drawLine(QPointF(12, T.HEADER_H + T.GRAN_H - 4),
                   QPointF(T.NODE_W - 12, T.HEADER_H + T.GRAN_H - 4))

        # rows
        lf = QFont(T.SANS, 8.5)
        for kind, obj, y in self._rows:
            if kind == "in":
                p.setFont(lf); p.setPen(T.INK)
                row_txt = self._input_row_text(obj)
                p.drawText(QRectF(14, y, 120, T.ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                           row_txt)
                if obj.type is not SocketType.DATASET:
                    self._paint_pill(p, obj, y)
                else:
                    nw = QFontMetricsF(lf).horizontalAdvance(row_txt)
                    self._paint_domain_chips(p, 14 + nw + 10, y, self.reads_domains(),
                                             align_left=True,
                                             missing=self.missing_domains())
            elif kind == "mode":
                p.setFont(lf); p.setPen(T.INK_2)
                p.drawText(QRectF(14, y, 90, T.ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                           obj.name)
                val = self.rec.modes.get(obj.name, obj.resolved_default())
                self._paint_value_pill(p, y, f"{val} ▾", None, False, obj=obj)
            elif kind == "out":
                p.setFont(lf); p.setPen(T.INK)
                text = self._output_row_text(obj)    # "K · name" on chK; a channel on out
                p.drawText(QRectF(T.NODE_W - 134, y, 120, T.ROW_H),
                           Qt.AlignVCenter | Qt.AlignRight, text)
                # domain chips only on the combined output; per-channel rows read
                # cleaner with just the channel-tinted socket dot + its name.
                if obj.type is SocketType.DATASET and self.output_channel_index(
                        obj.name) is None:
                    nw = QFontMetricsF(lf).horizontalAdvance(text)
                    self._paint_domain_chips(p, T.NODE_W - 14 - nw - 10, y,
                                             self.out_domains(), align_left=False)

    def _paint_domain_chips(self, p: QPainter, x: float, y: float, domains,
                            *, align_left: bool, missing=frozenset()) -> None:
        """A rail of small colored abbreviation chips (VOX/LBL/PT/…) beside a Dataset
        socket. ``align_left`` flows right from ``x`` (inputs); otherwise the block
        ends at ``x`` (outputs). A missing required domain paints in the error color."""
        if not domains:
            return
        cf = QFont(T.MONO, 6); cf.setBold(True)
        p.setFont(cf)
        fm = QFontMetricsF(cf)
        ch = T.ROW_H - 12
        top = y + 6
        gap = 3.0
        items = [(d, domain_abbr(d), fm.horizontalAdvance(domain_abbr(d)) + 10.0)
                 for d in domains]
        if not align_left:
            total = sum(w for _, _, w in items) + gap * (len(items) - 1)
            x = max(14.0, x - total)
        for d, txt, w in items:
            rect = QRectF(x, top, w, ch)
            miss = d in missing
            col = T.ERROR if miss else T.domain_qcolor(d)
            p.setPen(QPen(col, 1))
            p.setBrush(T.alpha(col, 60 if miss else 46))
            p.drawRoundedRect(rect, 3, 3)
            p.setPen(col)
            p.drawText(rect, Qt.AlignCenter, txt)
            x += w + gap

    # ── pill geometry (shared by painting and hit-testing) ────────────────────
    #
    # Every rect below is computed from the row's y and the pill's TEXT, with no painter
    # involved, because the same numbers have to be available to `controls()` when the user
    # clicks. Deriving them twice — once here, once in a hit-test — is how a control ends up
    # a few pixels away from the thing it looks like.

    _UNIT_TEXT = {"um": "µm", "um_axial": "µm↕", "nm": "nm", "s": "s"}

    @classmethod
    def _pill_unit(cls, s) -> str:
        return cls._UNIT_TEXT.get(s.unit, s.unit)

    def _pill_text(self, s) -> str:
        if self._is_source_pill(s):
            return self._source_pill_text()
        val = self.resolved(s)
        return "" if val is None else str(val)

    def _input_row_text(self, s) -> str:
        """An input row's text: the socket's name — or the CHANNEL its wire carries when that
        is one (or some) of the file's channels (``Cy5`` on ``data``, ``raw · DAPI`` on a
        named socket; V4.00 step 11g, :meth:`GraphDocument.socket_text`) — except on a Page
        Output (step 11f), whose wired item slots read as their items' names once it holds
        several, and whose empty slot reads ``+ item``."""
        if self.rec.op_key != PAGE_OUTPUT_OP or s.type is not SocketType.DATASET:
            try:
                return self.doc.socket_text(self.rec.id, s, "in")
            except Exception:                        # noqa: BLE001 — a bare document
                return s.name
        try:
            items = dict(self.doc.output_items(self.rec.id))
        except Exception:                            # noqa: BLE001 — a bare document
            return s.name
        if s.name not in items:
            return s.name if s.name == "data" else "+ item"
        return items[s.name] if len(items) >= 2 else s.name

    def _output_row_text(self, s) -> str:
        """An output row's text: a synthetic socket's label (``0 · GFP``, ``mask only``,
        ``item · cells``); a plain ``out`` the channel its stream carries when that is one
        (or some) of the file's (``Cy5``; V4.00 step 11g), else its name."""
        if s.label:
            return s.label
        try:
            return self.doc.socket_text(self.rec.id, s, "out")
        except Exception:                            # noqa: BLE001 — a bare document
            return s.name

    # ── a Page Input's Source (V4.00 step 11e) ───────────────────────────────
    def _is_source_pill(self, s) -> bool:
        return self.rec.op_key == PAGE_INPUT_OP and getattr(s, "name", "") == PAGE_SOURCE_KEY

    def _source_pill_text(self) -> str:
        """What a Page Input reads, as the menu names it — ``Image Input · raw ▾`` — never
        the stored ``pg1:raw``; ``choose… ▾`` while it reads nothing."""
        src = str(self.rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
        if not src:
            return "choose… ▾"
        try:
            label = dict(self.doc.source_choices(self.rec.id)).get(src)
        except Exception:                            # noqa: BLE001 — a bare document
            label = None
        return f"{label} ▾" if label else f"{src.split(':', 1)[-1]} (unbound) ▾"

    def _source_kind_color(self) -> Optional[QColor]:
        """The colour of the page kind a Page Input reads from (``None`` when unbound)."""
        src = str(self.rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
        try:
            if not src or src not in {v for v, _l in self.doc.source_choices(self.rec.id)}:
                return None
            kind = self.doc.source_kind(src)
        except Exception:                            # noqa: BLE001
            return None
        from nodelab_v2.canvas import PAGE_KIND_COLORS
        return QColor(PAGE_KIND_COLORS.get(kind, PAGE_KIND_COLORS["free"])) if kind else None

    def _open_source_menu(self, ctl: Ctl) -> None:
        """A Page Input's Source as a MENU of every Output it may read (V4.00 step 11e): one
        entry per named Output of each page that may feed this one — ``<page> · <variable>``
        — with a dot in the colour of that page's KIND (the colours of the page tabs and the
        Pages panel), the one it reads in bold with a tick. Nothing to type: an earlier
        page's Output is the only thing a Page Input can read."""
        from nodelab_v2.canvas import kind_icon
        view, r = self._view_and_rect(ctl)
        if view is None:
            return
        try:
            choices = list(self.doc.source_choices(self.rec.id))
        except Exception:                            # never let a picker eat a click
            choices = []
        current = str(self.rec.params.get(PAGE_SOURCE_KEY, "") or "")
        menu = QMenu()
        menu.setStyleSheet(T.menu_qss())
        menu.setToolTipsVisible(True)
        if not choices:
            none = menu.addAction("No earlier page has a named Page Output yet")
            none.setEnabled(False)
        bold = QFont(menu.font())
        bold.setBold(True)
        for value, label in choices:
            try:
                kind = self.doc.source_kind(value)
            except Exception:                        # noqa: BLE001
                kind = ""
            act = menu.addAction(kind_icon(kind or "free"),
                                 f"{label}   ✓" if value == current else label)
            act.setData(value)
            act.setToolTip(f"read the Output “{value.split(':', 1)[-1]}” of the page "
                           f"“{label.rsplit(' · ', 1)[0]}”")
            if value == current:
                act.setFont(bold)
        chosen = menu.exec(view.viewport().mapToGlobal(
            QPoint(int(r.left()), int(r.bottom() + 2))))
        if chosen is None or not chosen.data():
            return
        if str(chosen.data()) != current:
            self._write_param(PAGE_SOURCE_KEY, str(chosen.data()))

    @staticmethod
    def _pill_rect(y: float, txt: str, unit: str, derived: bool) -> QRectF:
        """The pill grows to its content but is CAPPED, and the text elides to fit.

        Without the cap a long string value (``cellsam_general`` is 15 mono characters)
        produced a pill wider than the room left beside its label, so it ran underneath the
        parameter name — legible as neither. Capping here rather than at paint time keeps
        the hit rect and the drawn rect the same object."""
        fm = QFontMetricsF(QFont(T.MONO, 8))
        badge_w = 22 if derived else 0
        uw = (fm.horizontalAdvance(unit) + 5) if unit else 0
        w = max(fm.horizontalAdvance(txt) + 14 + badge_w + uw, 34)
        w = min(w, _PILL_MAX_W)
        return QRectF(T.NODE_W - 12 - w, y + 4, w, T.ROW_H - 8)

    @staticmethod
    def _badge_rect(pill: QRectF) -> QRectF:
        """The ƒmd badge inside a derived param's pill — clicking it pins / unpins."""
        return QRectF(pill.left() + 6, pill.top() + 3, 18, pill.height() - 6)

    @staticmethod
    def _glyph_rect(pill: QRectF) -> QRectF:
        """The ◎ pick glyph, immediately left of the pill."""
        return QRectF(pill.left() - _PICK_GLYPH_W - 3, pill.top(),
                      _PICK_GLYPH_W, pill.height())

    @staticmethod
    def _gran_chip_rect(chip_txt: str) -> QRectF:
        """The footprint chip — a FACT, not a control, but its rect is needed by
        :meth:`controls`' neighbour check and by the probe, so it is derived here rather than
        inline in :meth:`paint` like every other rect on the card used to be."""
        cf = QFont(T.MONO, 6)
        cf.setBold(True)
        return QRectF(64, T.HEADER_H + 3,
                      QFontMetricsF(cf).horizontalAdvance(chip_txt) + 12, T.GRAN_H - 14)

    @classmethod
    def _foot_pill_rect(cls, txt: str, chip_txt: str = "") -> QRectF:
        """The population pill in the granularity band (V2.27).

        Right-aligned to ``NODE_W - 12`` like every other pill, so the band's control lines up
        with the rows'. Two caps, and the second is the load-bearing one:

        * ``_FOOT_PILL_MAX_W``, so a long population token does not make the pill span the card;
        * **the footprint chip's real right edge**, so the two can never overlap whatever either
          of them says. A constant cap was tried and was wrong by 5 px on the very first case
          (``PLANE`` + ``plane ▾``) — the chip's width depends on its text, so the only cap that
          holds for every footprint × population pair is measured from the chip itself.

        The text elides inside the rect (``_paint_pill_at``), so squeezing is safe; overlapping
        is not, because two rounded rects sharing pixels reads as a rendering bug.

        Vertically it sits inside the band and above the dashed rule at ``HEADER_H + GRAN_H - 4``
        — which is what makes it impossible for this rect to reach row 0."""
        fm = QFontMetricsF(QFont(T.MONO, 8))
        right = T.NODE_W - 12
        room = right - (cls._gran_chip_rect(chip_txt).right() + 6) if chip_txt else right
        w = min(max(fm.horizontalAdvance(txt) + 14, 34), _FOOT_PILL_MAX_W, max(34.0, room))
        return QRectF(right - w, T.HEADER_H + 4, w, T.GRAN_H - 12)

    def controls(self) -> List[Ctl]:
        """Every interactive control on this card, in hit-test order (front to back).

        THE single source of the card's interaction surface: :meth:`paint` draws from the
        same rects, so a control can never be painted somewhere it cannot be clicked. A
        collapsed card, a reroute dot or an unrecognized op has none — there are no rows to
        put them on."""
        if self._is_dot or self.rec.collapsed or self.spec is None:
            return []
        out: List[Ctl] = []
        # the footprint band's population pill, ABOVE the rows (V2.27) — the rows start at
        # HEADER_H + GRAN_H, so this is the one control not derived from a row's y.
        _scope = self._scope_mode()
        if _scope is not None:
            _pill_txt = self._foot_slots()[2]
            if _pill_txt:
                out.append(Ctl(self._foot_pill_rect(_pill_txt, self._foot_slots()[1]),
                               "scope", _scope))
        for kind, obj, y in self._rows:
            if kind == "mode":
                val = self.rec.modes.get(obj.name, obj.resolved_default())
                out.append(Ctl(self._pill_rect(y, f"{val} ▾", "", False), "mode", obj))
            elif kind == "in" and obj.type is not SocketType.DATASET:
                derived = self.is_derived(obj)
                pill = self._pill_rect(y, self._pill_text(obj),
                                       self._pill_unit(obj), derived)
                # the badge sits INSIDE the pill, so it has to be tested first
                if derived or obj.name in self.locked or obj.name in self.rec.params:
                    if obj.derive:
                        out.append(Ctl(self._badge_rect(pill), "pin", obj))
                if getattr(obj, "pick_kind", "") and self._leads_pick(obj):
                    out.append(Ctl(self._glyph_rect(pill), "pick", obj))
                out.append(Ctl(pill, "value", obj))
        return out

    def _leads_pick(self, s) -> bool:
        """Whether this socket's row carries the ring. A bound group (the crop rectangle)
        arms ONE gesture from four sockets, so it gets one glyph — on the first member —
        matching the inspector, which shows one button for the same reason."""
        bounds = getattr(s, "pick_bounds", ()) or ()
        return not bounds or s.name == bounds[0]

    def control_at(self, pos: QPointF) -> Optional[Ctl]:
        for c in self.controls():
            if c.rect.contains(pos):
                return c
        return None

    # ── pill painting ─────────────────────────────────────────────────────────
    def _paint_pill(self, p: QPainter, s, y: float) -> None:
        derived = self.is_derived(s)
        txt = self._pill_text(s)
        unit = self._pill_unit(s)
        pill = self._pill_rect(y, txt, unit, derived)
        self._paint_pill_at(p, pill, txt, unit, derived,
                            hot=self._hot_ctl is not None
                            and self._hot_ctl.kind in ("value", "pin")
                            and self._hot_ctl.obj is s)
        if self._is_source_pill(s):
            col = self._source_kind_color()
            if col is not None:                  # edged in the colour of the page it reads
                p.setPen(QPen(col, 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(pill, 5, 5)
        if getattr(s, "pick_kind", ""):
            self._paint_pick_glyph(p, self._glyph_rect(pill), s)

    def _paint_value_pill(self, p: QPainter, y: float, txt: str, unit,
                          derived: bool, *, obj=None) -> None:
        pill = self._pill_rect(y, txt, unit or "", derived)
        self._paint_pill_at(p, pill, txt, unit, derived,
                            hot=self._hot_ctl is not None
                            and self._hot_ctl.obj is obj and obj is not None)

    def _paint_pill_at(self, p: QPainter, pill: QRectF, txt: str, unit,
                       derived: bool, *, hot: bool = False) -> None:
        """Draw one pill. ``hot`` (the pointer is over it) brightens the border — the whole
        signal that a pill is a control and not a readout, together with the cursor change
        the hover handler makes."""
        vf = QFont(T.MONO, 8)
        edge = T.ACCENT if hot else (T.ACCENT_DIM if derived else T.BORDER)
        p.setPen(QPen(edge, 1.4 if hot else 1))
        p.setBrush(T.mix(T.BODY, T.PANEL_HI, 0.6) if hot else T.BODY)
        p.drawRoundedRect(pill, 5, 5)
        x = pill.left() + 6
        if derived:
            bf = QFont(T.MONO, 6); bf.setBold(True)
            p.setFont(bf)
            brect = self._badge_rect(pill)
            p.setPen(Qt.NoPen); p.setBrush(T.alpha(T.ACCENT, 30))
            p.drawRoundedRect(brect, 3, 3)
            p.setPen(T.ACCENT)
            p.drawText(brect, Qt.AlignCenter, "ƒmd")
            x += 20
        p.setFont(vf)
        p.setPen(T.INK)
        fm = QFontMetricsF(vf)
        avail = pill.right() - 6 - x - ((fm.horizontalAdvance(unit) + 5) if unit else 0)
        p.drawText(QRectF(x, pill.top(), max(4.0, avail), pill.height()),
                   Qt.AlignVCenter | Qt.AlignLeft,
                   fm.elidedText(txt, Qt.ElideRight, max(4.0, avail)))
        if unit:
            p.setPen(T.MUTED)
            p.drawText(pill.adjusted(0, 0, -6, 0), Qt.AlignVCenter | Qt.AlignRight, unit)

    def _paint_pick_glyph(self, p: QPainter, r: QRectF, s) -> None:
        """The ◎ that arms this param's viewer gesture."""
        hot = (self._hot_ctl is not None and self._hot_ctl.kind == "pick"
               and self._hot_ctl.obj is s)
        col = T.ACCENT if hot else T.alpha(T.ACCENT, 150)
        c = r.center()
        p.setBrush(T.alpha(T.ACCENT, 40) if hot else Qt.NoBrush)
        p.setPen(QPen(col, 1.3))
        p.drawEllipse(c, 5.2, 5.2)
        p.setBrush(col)
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, 1.9, 1.9)

    # ── run state (per-node progress) ─────────────────────────────────────────
    def set_run_state(self, state: str, *, fraction: Optional[float] = None,
                      note: str = "", seconds: Optional[float] = None,
                      levels: Optional[dict] = None) -> None:
        """Set what this node is doing in the current pull.

        ``state`` is one of ``""`` (idle — nothing painted), ``"queued"``,
        ``"running"``, ``"decoding"`` (its planes are being read — where a lazy chain's
        cost actually lands), ``"cached"`` (validated memo hit), ``"done"`` or
        ``"error"``. ``fraction`` in ``[0,1]`` makes the bar determinate; ``None``
        leaves it an indeterminate sweep. Repaints only when something changed, so the
        runner may call this as often as it likes.

        ``levels`` is the engine's ``progress`` info dict. When it carries the two-level
        split (``frames`` / ``frames_done`` / ``sub_fraction``) the card draws **two**
        rails — frames above, within-the-frame below. When it does not, the card keeps the
        single flat rail: a node with no frame axis must not be given a fake one."""
        frac = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        fr_frac, sub_frac, frames, frame = self._read_levels(levels)
        key = (state, frac, note, seconds, fr_frac, sub_frac, frames, frame)
        if key == (self._run, self._run_frac, self._run_note, self._run_secs,
                   self._frame_frac, self._sub_frac, self._frames, self._frame):
            return
        if state != self._run:
            self._phase = 0.0
        self._run, self._run_frac = state, frac
        self._run_note, self._run_secs = note, seconds
        self._frame_frac, self._sub_frac = fr_frac, sub_frac
        self._frames, self._frame = frames, frame
        # the card itself stays graphic (rail + dot); the numbers live in the tooltip and
        # the status bar, so a busy canvas doesn't turn into a wall of tiny text.
        self._apply_card_tip()
        self.update()

    def _page_boundary_label(self) -> str:
        """A page boundary card is titled by what it carries (V4.00 step 5): ``Output · raw``
        for a Page Output named ``raw``; ``Input · Image Input · raw`` for a Page Input
        reading one (the page AND the name, V4.00 step 11). A broken boundary says so on its
        face — ``Output · (unnamed)``, ``Input · (unbound)``, ``Input · raw (unbound)`` for a
        source no earlier page offers any more. Empty for every other card."""
        op = self.rec.op_key
        if op == PAGE_OUTPUT_OP:
            name = str(self.rec.params.get(PAGE_NAME_KEY, "") or "").strip()
            try:
                n = len(self.doc.output_items(self.rec.id))
            except Exception:                        # noqa: BLE001 — a bare document
                n = 0
            more = f" · {n} items" if n >= 2 else ""   # a several-item variable (11f)
            return f"Output · {name}{more}" if name else f"Output · (unnamed){more}"
        if op == PAGE_INPUT_OP:
            src = str(self.rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
            if not src:
                return "Input · (unbound)"
            try:
                label = dict(self.doc.source_choices(self.rec.id)).get(src)
            except Exception:                        # noqa: BLE001 — a bare document
                label = None
            return f"Input · {label}" if label else f"Input · {src.split(':', 1)[-1]} (unbound)"
        return ""

    def _boundary_broken(self) -> bool:
        """An unnamed Page Output, or a Page Input whose source no earlier page offers
        (V4.00 step 11) — painted like a dim-lever conflict, in the error colour."""
        op = self.rec.op_key
        if op == PAGE_OUTPUT_OP:
            return not str(self.rec.params.get(PAGE_NAME_KEY, "") or "").strip()
        if op == PAGE_INPUT_OP:
            src = str(self.rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
            if not src:
                return True
            try:
                return src not in {v for v, _l in self.doc.source_choices(self.rec.id)}
            except Exception:                        # noqa: BLE001
                return False
        return False

    def _footprint_line(self) -> str:
        """One plain-text line naming the footprint and WHAT DECIDES IT (V2.27).

        The band is a control on some nodes and a readout on others, and a readout that does not
        say who set it reads as broken. Before this, the inspector told the four nodes with a
        non-default ``footprint_mode`` that they were "Dimension-agnostic", which is false — the
        footprint is mode-resolved there, just not by a lever."""
        spec = self.spec
        if spec is None or self._is_group:
            return ""
        gran = self.granularity().replace("_", " ") or "undeclared"
        m = next((x for x in spec.modes if getattr(x, "is_scope", False)), None)
        if m is not None:
            # "population: X → reads Y", never "footprint: X" — on `enhance.normalize` the two
            # genuinely differ (population `plane`, footprint WHOLE_SERIES, because its
            # footprint is keyed on `bounds`), and naming the population "footprint" would
            # collapse the very distinction the band is drawn to show.
            val = str(self.rec.modes.get(m.name, m.resolved_default())).replace("_", " ")
            return f"population: {val} → reads {gran}"
        if spec.has_dim_lever():
            return f"footprint: {gran} (set by the 2D / 3D lever)"
        fp = getattr(spec, "footprint_mode", "") or ""
        if isinstance(spec.granularity, Mapping) and fp:
            return f"footprint: {gran} (set by the `{fp}` mode)"
        if not self.granularity():
            return "footprint: undeclared — this node is a source or a sink"
        return f"footprint: {gran} (dimension-agnostic)"

    def _apply_card_tip(self) -> None:
        """The card's hover tooltip: identity, the footprint line, then the run note.

        Called from :meth:`_layout` as well as :meth:`set_run_state`, because until V2.27 the
        only writer was the run path — so a card that had never been pulled had no tooltip at
        all, and the footprint was unreadable on a COLLAPSED card, which paints no band."""
        label = (self._group_name or self._page_boundary_label()
                 or (self.spec.label if self.spec else self.rec.op_key))
        parts = [f"{self.rec.id} · {label}"]
        foot = self._footprint_line()
        if foot:
            parts.append(foot)
        txt = self._run_text()
        if txt:
            parts.append(txt)
        self.setToolTip("\n".join(parts) if len(parts) > 1 else "")

    @staticmethod
    def _read_levels(levels: Optional[dict]):
        """Pull ``(frame_fraction, sub_fraction, frames, frame)`` out of an engine
        ``progress`` info dict, or four ``None``s when it has no frame axis. Anything
        malformed degrades to "no frame axis" rather than raising — a progress sink must
        never be the thing that breaks a paint."""
        if not levels:
            return None, None, None, None
        try:
            frames = levels.get("frames")
            if not frames or int(frames) <= 0:
                return None, None, None, None
            ff = float(levels.get("frame_fraction", 0.0))
            # An explicit None sub_fraction means "inside one opaque call": keep the frame
            # bar and SWEEP the sub bar. Distinguished from a missing key, which is 0%.
            raw = levels.get("sub_fraction", 0.0)
            sf = None if raw is None else max(0.0, min(1.0, float(raw)))
            return (max(0.0, min(1.0, ff)), sf,
                    int(frames), int(levels.get("frame", 0)))
        except (TypeError, ValueError):
            return None, None, None, None

    def run_state(self) -> str:
        return self._run

    def is_running(self) -> bool:
        """True while this card wants animation frames — the dot pulses whenever the node
        is working, so a determinate rail animates too."""
        return self._run in ("running", "decoding")

    def advance_phase(self, step: float = 0.06) -> None:
        """Move the indeterminate sweep one animation frame (driven by the scene's
        single shared timer — one timer for the canvas, not one per card)."""
        if not self.is_running():
            return
        self._phase = (self._phase + step) % 1.0
        self.update()

    def _run_color(self) -> QColor:
        """One accent for every kind of activity (red only for a failure) — the run
        overlay reads as a single system rather than a traffic light."""
        if self._run == "error":
            return T.ERROR
        if self._run == "queued":
            return T.MUTED
        return T.ACCENT

    def _run_text(self) -> str:
        if self._run == "queued":
            return "queued"
        if self._run == "cached":
            return "cached"
        if self._run == "error":
            return "error"
        if self._run == "decoding":
            return "reading"
        if self._run == "running":
            # the percentage only — the note ("t=3 z=1") is long and would collide with
            # the footprint chip; the status bar carries it instead. The rails are
            # deliberately unlabelled, so the tooltip is where their two numbers are
            # actually legible.
            if self._frames is not None and self._frame is not None:
                inner = ("working" if self._sub_frac is None
                         else f"{int(round(self._sub_frac * 100))}% of frame")
                return (f"frame {self._frame + 1}/{self._frames} · {inner}"
                        + (f" · {int(round(self._run_frac * 100))}% total"
                           if self._run_frac is not None else ""))
            if self._run_frac is not None:
                return f"{int(round(self._run_frac * 100))}%"
            return "running"
        if self._run == "done" and self._run_secs is not None:
            s = self._run_secs
            return f"{s * 1000:.0f} ms" if s < 1.0 else f"{s:.2f} s"
        return ""

    @staticmethod
    def _glow_pill(p: QPainter, rect: QRectF, col: QColor) -> None:
        """A rounded bar with a soft halo. QPainter has no ``box-shadow``, so the glow is
        two translucent passes grown around the bar — the same read as the mockup's
        ``0 0 8px accent`` without an (expensive, blurry) graphics effect."""
        p.setPen(Qt.NoPen)
        for grow, a in ((2.4, 26), (1.1, 58)):
            r = rect.adjusted(-grow, -grow, grow, grow)
            p.setBrush(T.alpha(col, a))
            p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setBrush(col)
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

    def _paint_card_glow(self, p: QPainter, rect: QRectF, col: QColor) -> None:
        """The card's outer glow while it is selected / running / failed."""
        p.setBrush(Qt.NoBrush)
        for grow, a in ((3.0, 20), (1.4, 46)):
            p.setPen(QPen(T.alpha(col, a), 1.6))
            p.drawRoundedRect(rect.adjusted(-grow, -grow, grow, grow),
                              T.RADIUS + grow, T.RADIUS + grow)

    def _paint_header_rail(self, p: QPainter) -> None:
        """The progress rails riding the header's bottom edge — **live work only**.

        There are two, and which are drawn depends on what the compute reports:

        * the **frame rail** (orange, on top) — whole frames finished out of the dataset's
          total. It steps once per frame, so it is the slow, coarse read: "where am I in
          the series". Drawn only when the compute reports a frame axis.
        * the **sub rail** (accent, below) — the work *inside* the frame being processed,
          restarting at each frame boundary. It is the fast one, and the one that falls
          back to a sweeping segment when the compute reports no fraction at all.

        The frame rail is flat while the sub rail glows: stacking two haloed bars 2 px
        apart just muddies both, and it is the fast bar that should catch the eye — the
        slow one is being read, not watched.

        Terminal states (queued / done / cached / error) carry no rail; they read from the
        dot, so a finished graph stays calm instead of being striped with full bars."""
        if self._run not in ("running", "decoding"):
            return
        w = T.NODE_W - 2.0
        y_sub = T.HEADER_H - T.PROG_H
        col = self._run_color()
        two = self._frame_frac is not None and self._run != "error"
        if two:
            y_frame = y_sub - T.PROG_H - T.PROG_GAP
            # the sub rail goes down FIRST: its halo reaches ~2 px up, and the opaque frame
            # rail painted over it is what keeps the orange from picking up a blue tint.
            # Both get a track, so a frame that has only just started reads as "0% of this
            # frame" rather than as a missing bar.
            self._paint_track(p, y_sub, w)
            if self._sub_frac is None:
                # inside ONE opaque call (a CNN inference, a hookless solver): the frame
                # position is known, the position within it is not. Sweeping says "working"
                # where a determinate bar frozen for thirty seconds would say "hung".
                self._paint_sweep(p, y_sub, w, col)
            elif self._sub_frac > 0:
                self._glow_pill(p, QRectF(1.0, y_sub, max(2.0, w * self._sub_frac),
                                          T.PROG_H), col)
            self._paint_track(p, y_frame, w)
            if self._frame_frac > 0:
                p.setPen(Qt.NoPen)
                p.setBrush(T.PROG_FRAME)
                r = QRectF(1.0, y_frame, max(2.0, w * self._frame_frac), T.PROG_H)
                p.drawRoundedRect(r, T.PROG_H / 2.0, T.PROG_H / 2.0)
            return
        if self._run_frac is not None:
            if self._run_frac <= 0:
                return
            self._glow_pill(p, QRectF(1.0, y_sub, max(2.0, w * self._run_frac),
                                      T.PROG_H), col)
            return
        self._paint_sweep(p, y_sub, w, col)

    def _paint_sweep(self, p: QPainter, y: float, w: float, col: QColor) -> None:
        """The indeterminate rail: a segment ping-ponging inside it. It stays FULLY inside
        (never clipped to nothing at the turnaround), so a card that is working always looks
        like it is working — even in a single frame, e.g. a screenshot."""
        seg = max(22.0, w * 0.28)
        tri = self._phase * 2.0
        f = tri if tri <= 1.0 else 2.0 - tri
        self._glow_pill(p, QRectF(1.0 + (w - seg) * f, y, seg, T.PROG_H), col)

    @staticmethod
    def _paint_track(p: QPainter, y: float, w: float) -> None:
        """The dim groove a stacked rail fills. Only the two-rail layout uses one: with two
        bars the reader needs to see which is which even at 0%, where a bare rail would
        just be absent."""
        p.setPen(Qt.NoPen)
        p.setBrush(T.alpha(T.PROG_TRACK, 170))
        p.drawRoundedRect(QRectF(1.0, y, w, T.PROG_H), T.PROG_H / 2.0, T.PROG_H / 2.0)

    def _dot_center(self) -> QPointF:
        """Where the header status dot sits: the header's right edge, stepped left of the
        2D/3D switch on the nodes that carry one."""
        x = (self._switch.pos().x() - 9.0 if self._switch is not None
             else T.NODE_W - 13.0)
        return QPointF(x, T.HEADER_H / 2.0)

    def _paint_status_dot(self, p: QPainter) -> None:
        """The header dot: a pulsing accent bead while the node works, a solid bead once
        it produced, a hollow ring when it was *reused* (cached) or merely enlisted
        (queued), red when it raised."""
        if not self._run:
            return
        c, r, col = self._dot_center(), T.DOT_R, self._run_color()
        if self._run in ("running", "decoding"):
            # pulse off the shared animation phase (the mockup's ~0.7 s blink)
            col = T.alpha(col, 96 + int(159 * abs(1.0 - 2.0 * ((self._phase * 1.7) % 1.0))))
        if self._run in ("queued", "cached"):        # nothing was computed here
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 1.4))
            p.drawEllipse(c, r, r)
            return
        p.setPen(Qt.NoPen)
        for grow, a in ((3.4, 28), (1.8, 62)):
            p.setBrush(T.alpha(col, a))
            p.drawEllipse(c, r + grow, r + grow)
        p.setBrush(col)
        p.drawEllipse(c, r, r)

    # ── on-canvas editing (V2.16) ─────────────────────────────────────────────
    #
    # Press decides what kind of interaction this is and consumes the event, which is also
    # what stops `ItemIsMovable` dragging the card: the base class never sees the press, so
    # there is no drag mode to suppress and no way for the two to disagree.

    def _scrub_step(self, s) -> float:
        # INT included, and for the same reason: a step of 1 on a 33 000-iteration budget is
        # 100 000 pixels of drag. `value_step` keeps the unit table's granularity wherever it
        # is usable and only bounds it at the two extremes (see its docstring).
        return value_step(s, self._current_number(s),
                          integer=s.type is SocketType.INT)

    def _current_number(self, s) -> float:
        try:
            return float(self.resolved(s))
        except (TypeError, ValueError):
            return 0.0

    def _clamp(self, s, v: float) -> float:
        """Keep a scrubbed value inside what the param can mean. Every numeric socket in the
        catalog is non-negative; a percentile additionally tops out at 100. Clamping in the
        drag rather than at compute time means the pill never shows a number the node would
        refuse."""
        v = max(0.0, v)
        if getattr(s, "pick_kind", "") == "percentile":
            v = min(100.0, v)
        return v

    def _write_param(self, name: str, value, *, notify: bool = True) -> None:
        """Write a param exactly as the inspector does — value plus the sticky pin — so a
        canvas edit and a typed edit are the same edit. ``notify=False`` during a live scrub:
        ``doc.touch()`` re-propagates metadata across the whole graph, which is not something
        to do on every mouse-move event; the release does it once."""
        self.rec.params[name] = value
        self.rec.set_locked(self.rec.locked | {name})
        if notify:
            self.doc.touch(self.node_id)
            self.changed.emit(self)
        self.update()

    def mousePressEvent(self, e) -> None:                # noqa: N802 - Qt naming
        ctl = self.control_at(e.pos()) if e.button() == Qt.LeftButton else None
        if ctl is None:
            super().mousePressEvent(e)
            return
        if not self.isSelected():                # clicking a control selects its card too
            sc = self.scene()
            if sc is not None:
                sc.clearSelection()
            self.setSelected(True)
        self._scrub = None
        if ctl.kind == "pin":
            self._toggle_pin(ctl.obj)
        elif ctl.kind == "pick":
            peer = self.spec.input(ctl.obj.pick_peer) if ctl.obj.pick_peer else None
            self.pick_requested.emit(request_for(self.rec.id, ctl.obj, peer))
        elif ctl.kind == "mode":
            self._open_menu(ctl, list(ctl.obj.choices),
                            self.rec.modes.get(ctl.obj.name, ctl.obj.resolved_default()))
        elif ctl.kind == "scope":
            self._open_scope_menu(ctl)
        elif ctl.kind == "value":
            s = ctl.obj
            if s.type is SocketType.BOOL:
                self._write_param(s.name, not bool(self.resolved(s)))
            elif self._is_source_pill(s):
                self._open_source_menu(ctl)
            elif s.type is SocketType.STRING and getattr(s, "choices", ()):
                self._open_menu(ctl, list(s.choices), str(self.resolved(s) or ""))
            elif s.type is SocketType.STRING and (s.layer_in or s.layer_in_mode):
                self._open_layer_menu(ctl)
            elif s.type is SocketType.STRING and (getattr(s, "column_in", None)
                                                  or getattr(s, "column_in_mode", "")):
                self._open_column_menu(ctl)
            elif s.type in (SocketType.INT, SocketType.FLOAT):
                # Defer: this is a scrub only if the pointer actually travels. A press that
                # doesn't move is a click, and opens the editor on release.
                self._scrub = (ctl, e.pos().x(), self._current_number(s), False)
            else:
                self._open_inline_edit(ctl)
        e.accept()

    def mouseMoveEvent(self, e) -> None:                 # noqa: N802 - Qt naming
        sc = getattr(self, "_scrub", None)
        if sc is None:
            super().mouseMoveEvent(e)
            return
        ctl, x0, v0, _moved = sc
        dx = e.pos().x() - x0
        if abs(dx) < _CLICK_SLOP and not _moved:
            e.accept()
            return
        s = ctl.obj
        step = self._scrub_step(s)
        mods = e.modifiers()
        if mods & Qt.ShiftModifier:
            step *= 0.1
        if mods & Qt.ControlModifier:
            step *= 10.0
        v = self._clamp(s, v0 + (dx / _SCRUB_PX) * step)
        v = int(round(v)) if s.type is SocketType.INT else round(v, 4)
        self._scrub = (ctl, x0, v0, True)
        self._write_param(s.name, v, notify=False)
        e.accept()

    def mouseReleaseEvent(self, e) -> None:              # noqa: N802 - Qt naming
        sc = getattr(self, "_scrub", None)
        self._scrub = None
        if sc is None:
            super().mouseReleaseEvent(e)
            return
        ctl, _x0, _v0, moved = sc
        if moved:
            self.doc.touch(self.node_id)                  # the one re-propagation for the whole drag
            self.changed.emit(self)
        else:
            self._open_inline_edit(ctl)       # a click, not a drag → type the value
        e.accept()

    def _toggle_pin(self, s) -> None:
        """Mirror the inspector's ƒ-auto button: pin the currently derived value, or drop
        the override and go back to following the metadata."""
        if s.name in self.rec.params or s.name in self.locked:
            self.rec.params.pop(s.name, None)
            self.rec.set_locked(self.locked - {s.name})
        else:
            self.rec.params[s.name] = self.resolved(s)
            self.rec.set_locked(self.locked | {s.name})
        self.doc.touch(self.node_id)
        self.changed.emit(self)
        self.update()

    def _view_and_rect(self, ctl: Ctl):
        """The active view plus ``ctl``'s rect in that view's viewport coordinates."""
        sc = self.scene()
        views = sc.views() if sc is not None else []
        if not views:
            return None, None
        view = views[0]
        poly = view.mapFromScene(self.mapToScene(ctl.rect))
        return view, poly.boundingRect()

    def _open_menu(self, ctl: Ctl, choices: Sequence[str], current: str) -> None:
        """A popup list for a mode pill or a closed-``choices`` param.

        Each entry carries its own option documentation (``choice_docs``), which is why
        ``setToolTipsVisible`` is on: a QMenu SWALLOWS action tooltips by default, so
        without it the prose is written and never shown. This is the card's answer to "what
        is the difference between these six methods" — the menu is already open and the
        pointer is already on the option being considered."""
        view, r = self._view_and_rect(ctl)
        if view is None or not choices:
            return
        menu = QMenu()
        menu.setStyleSheet(T.menu_qss())
        menu.setToolTipsVisible(True)
        docs = getattr(ctl.obj, "choice_docs", None) or {}
        for c in choices:
            act = menu.addAction(c)
            act.setCheckable(True)
            act.setChecked(c == current)
            act.setToolTip(option_hover_text(c, docs.get(c, "")))
        chosen = menu.exec(view.viewport().mapToGlobal(
            QPoint(int(r.left()), int(r.bottom() + 2))))
        if chosen is None:
            return
        text = chosen.text()
        if ctl.kind == "mode":
            if text != current:
                self._write_mode(ctl.obj.name, text)
        elif text != current:
            self._write_param(ctl.obj.name, text)

    def _write_mode(self, name: str, value: str) -> None:
        """Commit a mode change from the card — THE one mode-write path (V2.27).

        Extracted so the granularity band and the mode pill cannot drift: all four steps are
        load-bearing. A mode can gate sockets in or out, so the card must re-lay-out and the
        scene must re-route, or wires stay anchored to ports that have moved; ``doc.touch``
        re-propagates the envelope and invalidates the runner's in-flight pulls for this node's
        cone; and ``changed`` is what rebuilds the inspector."""
        self.rec.modes[name] = value
        self._layout()
        if self.scene() is not None and hasattr(self.scene(), "reroute"):
            self.scene().reroute()
        self.doc.touch(self.node_id)
        self.changed.emit(self)

    def _open_scope_menu(self, ctl: Ctl) -> None:
        """The granularity band's population menu (V2.27) — clicking the footprint.

        Its own opener rather than a ``kind`` handled by :meth:`_open_menu`, for a specific
        reason: that method's tail falls through to ``self._write_param(ctl.obj.name, text)``
        for any non-``"mode"`` kind, so a ``"scope"`` Ctl reaching it would write a **param
        named after a Mode** — which serializes into the graph file and folds into
        ``node_recipe_hash`` while being invisible in the inspector. A separate opener makes
        that unreachable.

        It also carries what a plain mode menu cannot: the READ COST. A disabled header names
        the footprint currently resolved, and each option is annotated with the footprint it
        would imply — so the one menu that changes the population also states what the change
        costs, at the moment of choosing. That is the whole reason the population is edited
        here rather than from an anonymous dropdown."""
        view, r = self._view_and_rect(ctl)
        m = ctl.obj
        if view is None or not getattr(m, "choices", ()):
            return
        state = self.state()
        current = self.rec.modes.get(m.name, m.resolved_default())
        menu = QMenu()
        menu.setStyleSheet(T.menu_qss())
        menu.setToolTipsVisible(True)
        # the FULL footprint name, never the chip's abbreviation: "reads plane" would be
        # ambiguous with the `plane` POPULATION listed directly underneath it.
        head = menu.addAction(f"reads {self.granularity().replace('_', ' ') or 'undeclared'}")
        head.setEnabled(False)
        docs = getattr(m, "choice_docs", None) or {}
        for c in m.choices:
            act = menu.addAction(c)
            act.setCheckable(True)
            act.setChecked(c == current)
            implied = None
            if self.spec is not None:
                implied = self.spec.resolve_granularity({**state, m.name: c})
            note = (f"reads {implied.value.replace('_', ' ')}" if implied is not None
                    else "footprint undeclared")
            act.setToolTip(option_hover_text(c, docs.get(c, ""), head_note=note))
        chosen = menu.exec(view.viewport().mapToGlobal(
            QPoint(int(r.left()), int(r.bottom() + 2))))
        if chosen is None or not chosen.isEnabled():
            return
        if chosen.text() != current:
            self._write_mode(m.name, chosen.text())

    def _open_layer_menu(self, ctl: Ctl) -> None:
        """A popup of the layers actually present on this socket's wire (2026-08-04).

        The inspector has offered a layer picker since V2.11 (``inspector._layer_box``), but
        the CARD did not: clicking a ``layer_in`` pill on the canvas dropped straight into
        :meth:`_open_inline_edit`, so the only way to change it was to know the name and type
        it. That is the failure mode the picker exists to prevent — a Voronoi node kept its
        factory default ``particles`` while the branch feeding it produced ``points``, and
        the mismatch surfaced as a pull error four nodes deep instead of as a menu with one
        obvious entry in it.

        The list is honest but INCOMPLETE — a couple of producers name layers the edit-time
        pass cannot predict — so free text must stay reachable, exactly as the inspector's
        combo is editable. Hence the trailing "Type a name…" entry rather than a closed
        menu, and hence a menu is still opened when the list comes back empty (with the
        current value as its one entry) instead of silently doing nothing."""
        s = ctl.obj
        try:
            choices = list(self.doc.layer_choices(self.rec.id, s))
        except Exception:                     # never let a picker eat a click
            choices = []
        dom = s.layer_in.value if s.layer_in else (s.layer_in_mode or "layer")
        self._open_name_menu(
            ctl, choices,
            each=f"A {dom} layer present on `{s.layer_from or 'the input'}`.",
            note="The list above is what the edit-time pass can predict for this wire; "
                 "a few producers name layers it cannot. Type the name if yours is "
                 "missing.")

    def _open_column_menu(self, ctl: Ctl) -> None:
        """A popup of the columns the graph has MEASURED onto this socket's wire (V2.28).

        :meth:`_open_layer_menu`'s sibling, and it exists for the same reason: a
        condition whose column can only be typed is a condition the user has to already
        know the answer to. Here the list is the point of the node rather than a
        convenience — "which statistics do I have?" is the question being asked, and
        before this it could only be answered by pulling and reading the error.

        Free text stays reachable for a STRONGER reason than on a layer pill: the column
        catalog is built from voluntary ``adds_columns`` declarations, so an undeclared
        producer writes real columns the menu cannot name."""
        s = ctl.obj
        try:
            choices = list(self.doc.column_choices(self.rec.id, s))
        except Exception:                     # never let a picker eat a click
            choices = []
        self._open_name_menu(
            ctl, choices,
            each="A column measured onto these objects upstream.",
            note="No columns on this wire yet — add Measure, Object Metrics "
                 "or Track Objects upstream",
            free_text=False)

    def _open_name_menu(self, ctl: Ctl, choices: list, *, each: str,
                        note: str, free_text: bool = True) -> None:
        """The shared name popup behind the layer and column pills — one menu rather
        than two copies of the empty-list, tooltip and free-text handling.

        ``free_text`` is the difference, and it tracks how complete the catalog behind the
        list is. A LAYER list cannot be closed (a couple of producers name layers the
        edit-time pass cannot predict), so it ends in "Type a name…". A COLUMN list can be:
        every structure-producing node declares ``adds_columns``, gated by
        ``selftest::test_column_catalog_complete``, so a name not in the list is one no node
        on this wire writes — and typing it would only defer the refusal to the pull.

        With no free-text escape, an EMPTY list can no longer fall through to the inline
        editor: it opens a menu carrying one disabled line naming the nodes that would fill
        it. Silently doing nothing on click reads as a broken pill, and dropping into a text
        box invites typing a column the graph does not have."""
        s = ctl.obj
        current = str(self.resolved(s) or "")
        if not choices and free_text:
            self._open_inline_edit(ctl)       # nothing to offer — go straight to typing
            return
        view, r = self._view_and_rect(ctl)
        if view is None:
            return
        menu = QMenu()
        menu.setStyleSheet(T.menu_qss())
        menu.setToolTipsVisible(True)
        # The CURRENT value is always listed, offered or not: a closed menu that omits it
        # gives the user no way to see what the pill is set to, and the pill may be showing a
        # column whose producer has since been deleted.
        listed = list(choices)
        if current and current not in listed:
            listed.insert(0, current)
        none_act = None
        if not free_text and not str(s.default or ""):
            # an OPTIONAL column (blank by default): the way back to blank
            none_act = menu.addAction("(none)")
            none_act.setCheckable(True)
            none_act.setChecked(not current)
            none_act.setToolTip(option_hover_text("(none)", "Leave it blank — the node's "
                                                  "default (e.g. one series, no grouping)."))
        for c in listed:
            act = menu.addAction(c)
            act.setCheckable(True)
            act.setChecked(c == current)
            act.setToolTip(option_hover_text(
                c, each if c in choices else
                "NOT on this wire — nothing upstream writes this column. Kept so the "
                "setting is visible rather than silently replaced."))
        if not listed:
            menu.addAction(note).setEnabled(False)
        if free_text:
            menu.addSeparator()
            typed = menu.addAction("Type a name…")
            typed.setToolTip(option_hover_text("Type a name…", note))
        else:
            typed = None
        chosen = menu.exec(view.viewport().mapToGlobal(
            QPoint(int(r.left()), int(r.bottom() + 2))))
        if chosen is None:
            return
        if typed is not None and chosen is typed:
            self._open_inline_edit(ctl)
        elif none_act is not None and chosen is none_act:
            if current:
                self._write_param(s.name, "")
        elif chosen.text() != current:
            self._write_param(s.name, chosen.text())

    def _open_inline_edit(self, ctl: Ctl) -> None:
        """Type a value straight into the pill."""
        view, r = self._view_and_rect(ctl)
        if view is None:
            return
        s = ctl.obj
        cur = self.resolved(s)
        # a minimum box, centred on the pill, so a zoomed-out card is still editable
        w, h = max(int(r.width()) + 12, 78), max(int(r.height()) + 4, 22)
        cx, cy = r.center().x(), r.center().y()
        box = QRect(int(cx - w / 2), int(cy - h / 2), w, h)

        def commit(text: str) -> None:
            text = text.strip()
            if s.type is SocketType.INT:
                try:
                    self._write_param(s.name, int(round(float(text))))
                except ValueError:
                    pass                      # unparseable → leave the value alone
            elif s.type is SocketType.FLOAT:
                try:
                    self._write_param(s.name, round(float(text), 6))
                except ValueError:
                    pass
            else:
                self._write_param(s.name, text)

        ed = _InlineEdit(view.viewport(), "" if cur is None else str(cur),
                         commit, lambda: None)
        ed.setStyleSheet(
            f"background:{T.BODY.name()}; color:{T.INK.name()}; font-family:{T.MONO};"
            f"border:1px solid {T.ACCENT.name()}; border-radius:5px;")
        ed.setGeometry(box)
        ed.show()
        ed.setFocus(Qt.MouseFocusReason)

    # ── events ────────────────────────────────────────────────────────────────
    def _place_close(self) -> None:
        """Pin the ✕ badge to the card's top-right corner — mostly inside it (so the
        pointer stays over the card on the way to the badge), just clear of the 2D/3D
        switch and the elided title."""
        w = self._dot_size if self._is_dot else T.NODE_W
        self._close.setPos(w - T.CLOSE_BTN + 4, -4)

    def hide_close_if_away(self) -> None:
        """Hide the ✕ badge once the pointer is over neither the card nor the badge.
        Deferred by a beat because Qt delivers the card's leave and the badge's enter in
        an unspecified order — hiding eagerly would yank the badge out from under a
        pointer that is on its way to click it."""
        def check() -> None:
            if self.scene() is None:
                return
            if not (self.isUnderMouse() or self._close.isUnderMouse()):
                self._close.setVisible(False)
        QTimer.singleShot(80, check)

    def hoverEnterEvent(self, e) -> None:
        self._close.setVisible(True)
        super().hoverEnterEvent(e)

    def hoverMoveEvent(self, e) -> None:                 # noqa: N802 - Qt naming
        """Track which control the pointer is over: the pill lights up and the cursor says
        what the click will do — a horizontal resize cursor over a scrubbable number, a hand
        over everything else. Without this the pills read as labels and nobody discovers
        that the canvas is editable at all."""
        ctl = self.control_at(e.pos())
        if ctl is not self._hot_ctl:
            same = (ctl is not None and self._hot_ctl is not None
                    and ctl.kind == self._hot_ctl.kind and ctl.obj is self._hot_ctl.obj)
            self._hot_ctl = ctl
            if not same:
                self.update()
        if ctl is None:
            self.unsetCursor()
        elif (ctl.kind == "value"
              and getattr(ctl.obj, "type", None) in (SocketType.INT, SocketType.FLOAT)):
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.PointingHandCursor)
        super().hoverMoveEvent(e)

    def hoverLeaveEvent(self, e) -> None:
        if self._hot_ctl is not None:
            self._hot_ctl = None
            self.update()
        self.unsetCursor()
        self.hide_close_if_away()
        super().hoverLeaveEvent(e)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.doc.set_pos(self.rec.id, self.pos().x(), self.pos().y())
            if self.scene() is not None and hasattr(self.scene(), "reroute"):
                self.scene().reroute()
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.update()
        return super().itemChange(change, value)


__all__ = ["SocketItem", "SwitchItem", "CloseItem", "NodeItem", "Ctl",
           "socket_identity", "socket_hover_text", "mode_identity", "mode_hover_text",
           "option_hover_text", "choice_doc_lines"]
