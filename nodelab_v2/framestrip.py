"""The frame strip — the M/T/Z cursor as one **box per frame** instead of a groove.

Replaces the Viewer's ``QSlider`` rows. Every frame of the axis gets its own rectangle at
a fixed size (:data:`FrameStrip.BOX_W`); once the boxes would run past the end of the
row they *compress* to fit, so the strip always spans exactly the axis and never scrolls.
That makes the row a readable index — you can see how many frames there are, which one
you are on, and which ones are picked — while still dragging like a slider.

Two independent pieces of state live on it:

* **the cursor** (:meth:`value`) — the frame being displayed. Press and drag scrubs it,
  exactly like the slider it replaces: the pointer is grabbed on press, so dragging past
  either end of the widget keeps scrubbing against the clamped edge.
* **the selection** (:meth:`selection`) — a *set* of indices, painted with an accent bar.
  This is the run scope the GUI's troubleshooting mode evaluates
  (:meth:`nodelab_v2.runner.EngineRunner.set_solo_frame`): pick three T boxes and a pull
  computes those three frames, so a tracker actually has something to link; pick Z boxes
  too and each of those frames is cut to those planes. Selection is a modified gesture
  (Ctrl / Shift / the context menu) precisely so that plain dragging stays scrubbing.

The strip is axis-agnostic: what an empty selection *means* is a run-scope policy the
owner states in :attr:`unpicked_note`, because it differs per axis (a frame axis falls
back to the cursor, z to the whole volume).

Painting reads the live :mod:`nodelab_v2.theme` tokens, so the light/dark switch needs no
more than a repaint. Frame counts run to thousands, so below a few pixels per box the
paint collapses to merged runs rather than thousands of one-pixel rectangles.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QMenu, QSizePolicy, QToolTip, QWidget

from nodelab_v2 import theme as T


class FrameStrip(QWidget):
    """One row of per-frame boxes: a scrubbing cursor plus a multi-frame selection.

    The public surface is deliberately ``QSlider``-shaped (:meth:`value`,
    :meth:`setValue`, :meth:`maximum`, :meth:`setMaximum`, :meth:`setRange`,
    ``valueChanged``) so it drops into the code — and the probes — that drove the sliders
    it replaces, and adds :meth:`selection` / :meth:`setSelection` / ``selectionChanged``
    on top."""

    #: the cursor moved (same contract as ``QSlider.valueChanged``)
    valueChanged = Signal(int)
    #: the selected frame SET changed. Emitted once per gesture — on release for a drag —
    #: because each emission costs the window a re-pull under the troubleshooting scope.
    selectionChanged = Signal()

    BOX_W = 13.0          # a box's width before compression kicks in
    GAP = 2.0             # gap between boxes at full size
    PAD = 1.0             # left/right inset so the end boxes aren't flush to the edge
    H = 15                # row height (docked)
    H_COMPACT = 11        # row height in the mini-map
    #: below this box width the gap is dropped, and below :data:`RUN_W` the paint
    #: collapses to merged runs (a 2000-frame series in 300 px).
    TIGHT_W = 5.0
    RUN_W = 3.0

    def __init__(self, axis: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._axis = (axis or "").upper()
        #: what an EMPTY selection means on this axis, in words, for the tooltip. The
        #: widget cannot know — the fallback is a run-scope policy, and it differs per axis
        #: (a frame axis falls back to the cursor, z to the whole volume). The owner sets it.
        self.unpicked_note = ""
        self._max = 0
        self._value = 0
        self._sel: Set[int] = set()
        self._hover: Optional[int] = None
        self._drag: Optional[str] = None        # "scrub" | "select"
        self._paint_to = True                   # what a select-drag writes
        self._last_paint: Optional[int] = None
        self._anchor = 0                        # Shift-range origin
        self._sel_dirty = False
        self._compact = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(self.H)

    # ── QSlider-shaped cursor API ──────────────────────────────────────────────
    def maximum(self) -> int:
        return self._max

    def minimum(self) -> int:
        return 0

    def count(self) -> int:
        """How many frames the axis has (``maximum() + 1``)."""
        return self._max + 1

    def setMaximum(self, hi: int) -> None:
        """Re-range the axis. Shrinking drops out-of-range picks — a selection that
        outlived its source would silently scope a run to frames that no longer exist."""
        hi = max(0, int(hi))
        if hi == self._max:
            return
        self._max = hi
        trimmed = {i for i in self._sel if i <= hi}
        dropped = trimmed != self._sel
        self._sel = trimmed
        self.updateGeometry()
        self.update()
        if self._value > hi:
            self.setValue(hi)
        if dropped:
            self.selectionChanged.emit()

    def setRange(self, lo: int, hi: int) -> None:
        self.setMaximum(hi)

    def value(self) -> int:
        return self._value

    def setValue(self, v: int) -> None:
        v = min(max(0, int(v)), self._max)
        if v == self._value:
            return
        self._value = v
        self.update()
        self.valueChanged.emit(v)

    # ── selection API ──────────────────────────────────────────────────────────
    def selection(self) -> Tuple[int, ...]:
        """The picked frames, sorted. Empty means *nothing picked* — callers read that
        as "fall back to the cursor", never as "run no frames"."""
        return tuple(sorted(self._sel))

    def setSelection(self, idx: Iterable[int]) -> None:
        want = {i for i in (int(v) for v in idx) if 0 <= i <= self._max}
        if want == self._sel:
            return
        self._sel = want
        self.update()
        self.selectionChanged.emit()

    def clearSelection(self) -> None:
        self.setSelection(())

    # ── layout ─────────────────────────────────────────────────────────────────
    def setCompact(self, on: bool) -> None:
        self._compact = bool(on)
        self.setFixedHeight(self.H_COMPACT if self._compact else self.H)

    def _pitch(self) -> Tuple[float, float]:
        """``(pitch, box_width)`` in device-independent pixels. Boxes hold :data:`BOX_W`
        until the row runs out of space, then compress evenly — which is what keeps the
        strip an honest picture of the whole axis at any length."""
        n = self.count()
        avail = max(1.0, float(self.width()) - 2 * self.PAD)
        pitch = self.BOX_W + self.GAP
        if n * pitch > avail:
            pitch = avail / n
        gap = self.GAP if pitch >= (self.TIGHT_W + self.GAP) else 0.0
        return pitch, max(1.0, pitch - gap)

    def _x_of(self, i: int) -> float:
        return self.PAD + i * self._pitch()[0]

    def _index_at(self, x: float) -> int:
        """The frame under ``x`` — **clamped**, which is what makes a drag that leaves the
        widget keep scrubbing against the nearest end instead of stopping dead."""
        pitch = self._pitch()[0]
        return min(max(0, int(math.floor((float(x) - self.PAD) / pitch))), self._max)

    def sizeHint(self) -> QSize:
        n = self.count()
        return QSize(int(min(4000, max(48, n * (self.BOX_W + self.GAP)))), self.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(32, self.height())

    # ── paint ──────────────────────────────────────────────────────────────────
    def paintEvent(self, _ev) -> None:
        n = self.count()
        if n <= 0 or self.width() <= 0:
            return
        pitch, bw = self._pitch()
        h = float(self.height())
        y0, bh = 1.0, max(3.0, h - 2.0)
        live = self.isEnabled()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, bw >= self.TIGHT_W)
        radius = 2.0 if bw >= self.TIGHT_W else 0.0

        base = T.BODY if live else T.BG
        border = T.BORDER
        sel_fill = T.ACCENT_DIM
        cur = T.ACCENT if live else T.MUTED

        def box(i: int) -> QRectF:
            return QRectF(self._x_of(i), y0, bw, bh)

        def fill(rect: QRectF, col: QColor, pen: Optional[QColor] = None) -> None:
            p.setBrush(col)
            p.setPen(pen if pen is not None else Qt.NoPen)
            if radius:
                p.drawRoundedRect(rect, radius, radius)
            else:
                p.drawRect(rect)

        if bw < self.RUN_W:
            # ── dense: one rect per box would be thousands of sub-pixel draws ──────
            span = QRectF(self.PAD, y0, n * pitch, bh)
            fill(span, base, border)         # outlined: at this density the fill alone
            for a, b in _runs(sorted(self._sel)):   # barely separates from the panel
                fill(QRectF(self._x_of(a), y0, (b - a + 1) * pitch, bh), sel_fill)
            p.fillRect(QRectF(self._x_of(self._value), y0, max(2.0, bw), bh), cur)
            p.end()
            return

        for i in range(n):
            r = box(i)
            picked = i in self._sel
            if i == self._value:
                fill(r, cur, T.ACCENT if live else T.BORDER)
            elif picked:
                fill(r, sel_fill, T.ACCENT)
            elif i == self._hover and live:
                fill(r, T.PANEL_HI, T.BORDER_HI)
            else:
                fill(r, base, border if bw >= 4.0 else None)
            if picked:
                # a bar under every picked box: the cursor's own bright fill would
                # otherwise hide whether the frame the user is on is in the run set
                p.fillRect(QRectF(r.left(), r.bottom() - 2.0, r.width(), 2.0),
                           T.ACCENT if live else T.MUTED)
        p.end()

    # ── mouse ──────────────────────────────────────────────────────────────────
    def mousePressEvent(self, ev) -> None:
        if not self.isEnabled():
            return
        if ev.button() == Qt.RightButton:
            self._context_menu(ev)
            return
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        i = self._index_at(ev.position().x())
        mods = ev.modifiers()
        pick = bool(mods & (Qt.ControlModifier | Qt.MetaModifier | Qt.ShiftModifier))
        if not pick:
            self._drag = "scrub"
            self._anchor = i
            self.setValue(i)
            self.setFocus(Qt.MouseFocusReason)
            return
        self._drag = "select"
        if mods & Qt.ShiftModifier:              # extend the run from the last anchor
            self._paint_to = True
            lo, hi = sorted((self._anchor, i))
            self._apply_pick(range(lo, hi + 1), True)
        else:                                    # Ctrl: toggle, then paint that way
            self._paint_to = i not in self._sel
            self._anchor = i
            self._apply_pick((i,), self._paint_to)
        self._last_paint = i
        self.setFocus(Qt.MouseFocusReason)

    def mouseMoveEvent(self, ev) -> None:
        x = ev.position().x()
        if self._drag == "scrub":
            self.setValue(self._index_at(x))
            return
        if self._drag == "select":
            i = self._index_at(x)
            lo, hi = sorted((self._last_paint if self._last_paint is not None else i, i))
            self._apply_pick(range(lo, hi + 1), self._paint_to)   # no gaps on a fast drag
            self._last_paint = i
            return
        hov = self._index_at(x) if self.rect().contains(ev.position().toPoint()) else None
        if hov != self._hover:
            self._hover = hov
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        was, self._drag = self._drag, None
        self._last_paint = None
        if was == "select" and self._sel_dirty:
            # one emission per gesture: each one is a re-pull under the run scope
            self._sel_dirty = False
            self.selectionChanged.emit()

    def leaveEvent(self, _ev) -> None:
        if self._hover is not None:
            self._hover = None
            self.update()

    def wheelEvent(self, ev) -> None:
        if not self.isEnabled():
            return
        step = 1 if ev.angleDelta().y() < 0 else -1
        self.setValue(self._value + step)
        ev.accept()

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k in (Qt.Key_Left, Qt.Key_Down):
            self.setValue(self._value - 1)
        elif k in (Qt.Key_Right, Qt.Key_Up):
            self.setValue(self._value + 1)
        elif k == Qt.Key_Home:
            self.setValue(0)
        elif k == Qt.Key_End:
            self.setValue(self._max)
        elif k == Qt.Key_PageUp:
            self.setValue(self._value - 10)
        elif k == Qt.Key_PageDown:
            self.setValue(self._value + 10)
        elif k == Qt.Key_Space:
            self._apply_pick((self._value,), self._value not in self._sel)
            self._sel_dirty = False
            self.selectionChanged.emit()
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    def event(self, ev) -> bool:
        if ev.type() == QEvent.ToolTip:
            QToolTip.showText(ev.globalPos(), self._tip(self._index_at(ev.pos().x())),
                              self)
            return True
        return super().event(ev)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _apply_pick(self, idx: Sequence[int], on: bool) -> None:
        before = len(self._sel)
        if on:
            self._sel.update(i for i in idx if 0 <= i <= self._max)
        else:
            self._sel.difference_update(idx)
        if len(self._sel) != before:
            self._sel_dirty = True
            self.update()

    def _tip(self, i: int) -> str:
        ax = self._axis or "frame"
        n = self.count()
        sel = self.selection()
        pick = (f"<br>picked: {compact_list(sel)} ({len(sel)} of {n})" if sel
                else f"<br>{self.unpicked_note or 'nothing picked'}")
        return (f"<b>{ax} {i}</b> of {n}{pick}"
                f"<br><br>drag &nbsp;— scrub (keeps working outside the strip)"
                f"<br>ctrl+click / drag &nbsp;— pick {ax} for the run scope"
                f"<br>shift+click &nbsp;— pick the range from the last one"
                f"<br>right-click &nbsp;— pick all / invert / clear")

    def _context_menu(self, ev) -> None:
        ax = self._axis or "frame"
        i = self._index_at(ev.position().x())
        menu = QMenu(self)
        menu.addAction(f"Pick only {ax} {i}", lambda: self.setSelection((i,)))
        menu.addAction(f"Pick all {ax}", lambda: self.setSelection(range(self.count())))
        menu.addAction("Invert picks",
                       lambda: self.setSelection(set(range(self.count())) - self._sel))
        act = menu.addAction("Clear picks", self.clearSelection)
        act.setEnabled(bool(self._sel))
        menu.addSeparator()
        menu.addAction(f"Go to {ax} {i}", lambda: self.setValue(i))
        menu.exec(ev.globalPosition().toPoint())


def _runs(sorted_idx: List[int]):
    """``[1,2,3,7,8]`` → ``[(1,3), (7,8)]`` — contiguous spans, so a dense strip paints
    one rectangle per run instead of one per frame."""
    out: List[Tuple[int, int]] = []
    for i in sorted_idx:
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def compact_list(idx: Sequence[int], limit: int = 6) -> str:
    """``(3,4,5,9)`` → ``"3–5, 9"``; long selections elide. Used in tooltips, the status
    line and the SOLO chip, so they all name a selection the same way."""
    if not idx:
        return "—"
    parts = [f"{a}–{b}" if b > a else f"{a}" for a, b in _runs(list(idx))]
    if len(parts) > limit:
        return ", ".join(parts[:limit]) + f", +{len(parts) - limit} more"
    return ", ".join(parts)
