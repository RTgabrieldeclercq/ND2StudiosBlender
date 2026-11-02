"""Pixel strip — a pixel table shown as its pixels (2026-11-02).

A pixel table (``analysis.measure`` with Members = pixel; marked by
:data:`nodegraph.tables.PIXEL_TABLES_KEY`) has one row per voxel, and read as a grid of
numbers it is a column of a million near-identical rows. Shown as what it is — the pixels
themselves, one square each, laid in a strip that wraps line by line to the width of the
panel — it can be read at a glance: sort by ``intensity`` and the strip becomes the
region's histogram drawn as a ramp; sort by ``edge_dist_um`` and it reads rim to core; sort
by ``label_id`` and each region is a run of its own pixels.

* **Sort by** any column, ascending or descending (stable, so ties keep table order; NaN
  always last). Text columns sort as text.
* **Colour** is one numeric column in GREY — 0.5th to 99.5th percentile of the shown rows,
  black to white — because one channel in grey is what reads on an analysis figure. A
  NaN cell is drawn hatched-dim, distinct from black.
* **Hover** a pixel for a rounded card with every column of its row.

Painting costs the VISIBLE rows only: the canvas is the size of its viewport and scrolls
virtually (its own scroll bar moves the first visible line), and each paint builds a small
``(lines, columns)`` image of that band and lets Qt scale it up nearest-neighbour. A
million pixels scroll like a hundred.

Qt; the ordering and colour arithmetic are plain numpy (:class:`StripData`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetrics, QGuiApplication, QImage, \
    QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QScrollBar, QSizePolicy, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from nodelab_v2 import theme as T
from nodegraph.tables import _ordered_columns

#: the column a strip is coloured by, and sorted by, when it has one
_DEFAULT_COLOUR = "intensity"
_DEFAULT_SORT = "intensity"
#: rows of the colour column the window is measured over (a strided sample past this)
_WINDOW_SAMPLE = 2_000_000


def _numeric(arr: np.ndarray) -> bool:
    return np.asarray(arr).dtype.kind in "biuf"


def _fmt(v) -> str:
    """One cell for the hover card: integers whole, floats to 6 significant figures, text
    as is, a missing value as an em dash."""
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}" if abs(int(v)) >= 10_000 else str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not np.isfinite(f):
            return "NaN" if np.isnan(f) else ("∞" if f > 0 else "−∞")
        return str(int(f)) if f.is_integer() and abs(f) < 1e15 else f"{f:.6g}"
    return str(v)


class StripData:
    """The ordering and grey levels behind a strip — numpy only, no Qt.

    ``rows`` narrows the table to a subset (the spreadsheet's per-file tab); ``order``
    holds TABLE row indices in strip order, so a cell maps back to its row in the full
    table and in the exported file."""

    def __init__(self) -> None:
        self.cols: Dict[str, np.ndarray] = {}
        self.names: List[str] = []
        self.rows: Optional[np.ndarray] = None
        self.sort_by = ""
        self.ascending = True
        self.colour_by = ""
        self.order = np.zeros(0, dtype=np.int64)
        self.grey = np.zeros(0, dtype=np.uint8)      # per strip position
        self.missing = np.zeros(0, dtype=bool)       # per strip position (NaN colour)
        self.window = (0.0, 1.0)

    @property
    def n(self) -> int:
        return int(self.order.size)

    def numeric_names(self) -> List[str]:
        return [c for c in self.names if _numeric(self.cols[c])]

    def set_table(self, cols: Dict[str, np.ndarray], rows: Optional[Sequence[int]] = None
                  ) -> None:
        self.cols = {k: np.asarray(v) for k, v in cols.items()}
        self.names = _ordered_columns(self.cols)
        total = max((len(v) for v in self.cols.values()), default=0)
        self.rows = (np.arange(total, dtype=np.int64) if rows is None
                     else np.asarray(rows, dtype=np.int64))
        if self.sort_by not in self.cols:
            self.sort_by = (_DEFAULT_SORT if _DEFAULT_SORT in self.cols
                            else ("id" if "id" in self.cols else (self.names or [""])[0]))
        nums = self.numeric_names()
        if self.colour_by not in nums:
            self.colour_by = (_DEFAULT_COLOUR if _DEFAULT_COLOUR in nums
                              else (nums[-1] if nums else ""))
        self.resort()

    def resort(self) -> None:
        """Recompute ``order`` from ``sort_by``/``ascending`` and the colours with it."""
        rows = self.rows if self.rows is not None else np.zeros(0, dtype=np.int64)
        col = self.cols.get(self.sort_by)
        if col is None or rows.size == 0:
            self.order = rows.copy()
        else:
            v = col[rows]
            if v.dtype.kind in "biu":
                key = v.astype(np.int64)
                o = np.argsort(key if self.ascending else -key, kind="stable")
            elif v.dtype.kind == "f":
                # NaN sorts last either way: argsort puts it last, and -NaN is still NaN
                o = np.argsort(v if self.ascending else -v, kind="stable")
            else:
                o = np.argsort(np.asarray([str(s) for s in v.tolist()]), kind="stable")
                if not self.ascending:
                    o = o[::-1]
            self.order = rows[o]
        self.recolour()

    def recolour(self) -> None:
        col = self.cols.get(self.colour_by)
        if col is None or self.order.size == 0:
            self.grey = np.full(self.order.size, 128, dtype=np.uint8)
            self.missing = np.zeros(self.order.size, dtype=bool)
            return
        v = np.asarray(col, dtype=float)[self.order]
        finite = np.isfinite(v)
        sample = v[finite]
        if sample.size > _WINDOW_SAMPLE:
            sample = sample[:: int(np.ceil(sample.size / _WINDOW_SAMPLE))]
        if sample.size:
            lo, hi = (float(x) for x in np.percentile(sample, [0.5, 99.5]))
            if not hi > lo:
                lo, hi = float(sample.min()), float(sample.max())
            if not hi > lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        self.window = (lo, hi)
        g = np.clip((np.where(finite, v, lo) - lo) / (hi - lo), 0.0, 1.0)
        self.grey = np.round(g * 255.0).astype(np.uint8)
        self.missing = ~finite

    def row_values(self, table_row: int) -> List[tuple]:
        """``[(column, value), …]`` of one table row, columns in table order."""
        out = []
        for name in self.names:
            arr = self.cols[name]
            out.append((name, arr[table_row] if table_row < len(arr) else None))
        return out


class HoverCard(QWidget):
    """The rounded card a hovered pixel shows: a title with the pixel's grey swatch, then
    every column of its row as ``name  value``. A frameless tool-tip window that never
    takes focus or the mouse, so it cannot steal the hover it is reporting."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # owned by the strip view (a window-type child is still its own top-level window),
        # so it is deleted with the view rather than at some arbitrary point of shutdown
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint
                         | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._title = ""
        self._rows: List[tuple] = []
        self._swatch: Optional[QColor] = None
        self._mono = QFont(T.MONO)
        self._mono.setPointSizeF(9.0)
        self._bold = QFont(T.SANS)
        self._bold.setPointSizeF(9.5)
        self._bold.setBold(True)

    _PAD = 12
    _GAP = 14
    _RADIUS = 10.0

    def set_content(self, title: str, rows: List[tuple], swatch: Optional[QColor]) -> None:
        self._title, self._rows, self._swatch = title, list(rows), swatch
        fm, fb = QFontMetrics(self._mono), QFontMetrics(self._bold)
        name_w = max((fm.horizontalAdvance(str(n)) for n, _v in self._rows), default=0)
        val_w = max((fm.horizontalAdvance(_fmt(v)) for _n, v in self._rows), default=0)
        sw = fb.height() + 6 if swatch is not None else 0
        width = max(fb.horizontalAdvance(title) + sw, name_w + self._GAP + val_w)
        height = fb.height() + 8 + len(self._rows) * fm.lineSpacing()
        self._name_w = name_w
        self.resize(int(width + 2 * self._PAD), int(height + 2 * self._PAD))
        self.update()

    def show_at(self, global_pos: QPoint) -> None:
        """Beside the pointer, flipped to the other side where it would leave the screen."""
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else QRect(0, 0, 4000, 4000)
        x, y = global_pos.x() + 18, global_pos.y() + 18
        if x + self.width() > area.right():
            x = global_pos.x() - 18 - self.width()
        if y + self.height() > area.bottom():
            y = max(area.top(), area.bottom() - self.height())
        self.move(max(area.left(), x), max(area.top(), y))
        if not self.isVisible():
            self.show()
        self.raise_()

    def paintEvent(self, _ev) -> None:                       # noqa: N802 — Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(T.BORDER_HI or T.BORDER, 1.0))
        p.setBrush(T.alpha(T.BODY, 246))
        p.drawRoundedRect(r, self._RADIUS, self._RADIUS)
        x0 = y = self._PAD
        fb = QFontMetrics(self._bold)
        if self._swatch is not None:
            s = fb.height()
            p.setPen(QPen(T.BORDER, 1.0))
            p.setBrush(self._swatch)
            p.drawRoundedRect(QRectF(x0, y, s, s), 3.0, 3.0)
            x0 += s + 6
        p.setFont(self._bold)
        p.setPen(T.INK)
        p.drawText(QRect(int(x0), int(y), self.width(), fb.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, self._title)
        y += fb.height() + 8
        p.setFont(self._mono)
        fm = QFontMetrics(self._mono)
        for name, val in self._rows:
            p.setPen(T.MUTED)
            p.drawText(QRect(self._PAD, int(y), self._name_w, fm.lineSpacing()),
                       Qt.AlignLeft | Qt.AlignVCenter, str(name))
            p.setPen(T.INK)
            p.drawText(QRect(self._PAD + self._name_w + self._GAP, int(y),
                             self.width() - 2 * self._PAD - self._name_w - self._GAP,
                             fm.lineSpacing()),
                       Qt.AlignRight | Qt.AlignVCenter, _fmt(val))
            y += fm.lineSpacing()
        p.end()


class _StripCanvas(QWidget):
    """The strip itself, scrolled VIRTUALLY: the widget is only as tall as its viewport and
    paints the lines from :attr:`top` down. A widget as tall as the whole strip would hit
    Qt's 16 777 215-px size ceiling at a few million pixels, and starts one column wide
    before it is first shown."""

    hovered = Signal(int)                    # strip position under the pointer, -1 off it
    layout_changed = Signal()                # columns or visible lines changed
    scrolled = Signal(int)                   # the wheel moved :attr:`top`

    def __init__(self, data: StripData) -> None:
        super().__init__()
        self.data = data
        self.cell = 10
        self.top = 0                         # first visible line
        self._cols = 1
        self._hover = -1
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMinimumHeight(60)

    # ── layout ────────────────────────────────────────────────────────────────
    def columns(self) -> int:
        return max(1, self.width() // max(1, self.cell))

    def lines(self) -> int:
        """Lines the whole strip takes at the current width."""
        return -(-self.data.n // self._cols) if self.data.n else 0

    def visible_lines(self) -> int:
        return max(1, -(-self.height() // max(1, self.cell)))

    def max_top(self) -> int:
        return max(0, self.lines() - self.visible_lines() + 1)

    def relayout(self, keep_first: bool = True) -> None:
        """Recompute the columns; with ``keep_first``, keep the first visible PIXEL in view
        rather than the first visible line, so a resize does not jump the strip."""
        first = self.top * self._cols
        self._cols = self.columns()
        self.top = min(first // self._cols if keep_first else 0, self.max_top())
        self.layout_changed.emit()
        self.update()

    def set_top(self, top: int) -> None:
        top = int(min(max(0, top), self.max_top()))
        if top != self.top:
            self.top = top
            self.update()

    def resizeEvent(self, ev) -> None:                       # noqa: N802 — Qt override
        super().resizeEvent(ev)
        self.relayout()

    def wheelEvent(self, ev) -> None:                        # noqa: N802 — Qt override
        steps = ev.angleDelta().y() / 120.0
        self.set_top(self.top - int(round(steps * 3)))
        self.scrolled.emit(self.top)
        ev.accept()

    def sizeHint(self) -> QSize:                             # noqa: N802 — Qt override
        return QSize(400, 300)

    def position_at(self, pt: QPoint) -> int:
        """The strip position under a widget point, or -1 (a gutter, or past the end)."""
        c, r = pt.x() // self.cell, self.top + pt.y() // self.cell
        if not (0 <= c < self._cols) or pt.y() < 0:
            return -1
        i = int(r * self._cols + c)
        return i if i < self.data.n else -1

    # ── paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, ev) -> None:                        # noqa: N802 — Qt override
        p = QPainter(self)
        p.fillRect(ev.rect(), T.BG)
        d, cell, cols = self.data, self.cell, self._cols
        if d.n == 0:
            p.setPen(T.MUTED)
            p.drawText(self.rect(), Qt.AlignCenter, "no pixels in this table")
            p.end()
            return
        r0 = self.top
        r1 = min(self.lines(), r0 + self.visible_lines())
        if r1 <= r0:
            p.end()
            return
        a, b = r0 * cols, min(d.n, r1 * cols)
        lines = r1 - r0
        bg = T.BG
        rgb = np.empty((lines * cols, 4), dtype=np.uint8)
        rgb[:] = (bg.blue(), bg.green(), bg.red(), 255)       # BGRA = Format_RGB32
        g = d.grey[a:b]
        rgb[: b - a, 0] = g
        rgb[: b - a, 1] = g
        rgb[: b - a, 2] = g
        miss = np.flatnonzero(d.missing[a:b])
        if miss.size:
            nan = _nan_colour()
            rgb[miss] = (nan.blue(), nan.green(), nan.red(), 255)
        img = QImage(np.ascontiguousarray(rgb.reshape(lines, cols, 4)).data,
                     cols, lines, cols * 4, QImage.Format_RGB32).copy()
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(QRect(0, 0, cols * cell, lines * cell), img)
        if cell >= 5:                       # a 1-px gutter so neighbouring pixels read apart
            p.setPen(QPen(bg, 1))
            bottom = lines * cell
            for c in range(1, cols + 1):
                x = c * cell - 1
                p.drawLine(x, 0, x, bottom)
            for r in range(1, lines + 1):
                y = r * cell - 1
                p.drawLine(0, y, cols * cell, y)
        if 0 <= self._hover < d.n:
            hr, hc = divmod(self._hover, cols)
            if r0 <= hr < r1:
                p.setPen(QPen(T.ACCENT, 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRect(hc * cell, (hr - r0) * cell, cell - 1, cell - 1))
        p.end()

    def set_hover(self, i: int) -> None:
        if i != self._hover:
            self._hover = i
            self.update()

    def mouseMoveEvent(self, ev) -> None:                    # noqa: N802 — Qt override
        i = self.position_at(ev.position().toPoint())
        self.set_hover(i)
        self.hovered.emit(i)

    def leaveEvent(self, _ev) -> None:                       # noqa: N802 — Qt override
        self.set_hover(-1)
        self.hovered.emit(-1)


def _compact_combo() -> QComboBox:
    """A dropdown whose MINIMUM is a few characters (it still grows with the row): its
    default minimum is its longest column name, which is what widens a docked panel."""
    box = QComboBox()
    box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(4)
    return box


def _nan_colour() -> QColor:
    """A NaN cell's colour: tinted from the theme's error colour, so it never reads as a
    black (lowest-value) pixel."""
    return T.mix(T.BG, T.ERROR or T.ACCENT, 0.45)


class PixelStripView(QWidget):
    """A pixel table as a wrapping strip of its pixels, with Sort / Colour controls and a
    hover card per pixel — what the spreadsheet shows for a pixel table."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.data = StripData()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        # Two short rows, and dropdowns that may shrink to a few characters: the Spreadsheet
        # is tabbed with Properties, and a dock tab group is as wide as its widest MINIMUM
        # — one long row of controls here widened the whole Properties column (376 → 439 px
        # on a fresh window), even while no pixel table was showing.
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(QLabel("Sort by"))
        self.sort_box = _compact_combo()
        self.sort_box.setToolTip("Which column orders the pixels along the strip.")
        self.sort_box.currentTextChanged.connect(self._on_sort)
        head.addWidget(self.sort_box, 1)
        self.order_btn = QToolButton()
        self.order_btn.setText("↑")
        self.order_btn.setToolTip("Ascending (↑) or descending (↓). NaN always sorts last.")
        self.order_btn.clicked.connect(self._on_order)
        head.addWidget(self.order_btn)
        v.addLayout(head)
        head2 = QHBoxLayout()
        head2.setSpacing(6)
        head2.addWidget(QLabel("Colour"))
        self.colour_box = _compact_combo()
        self.colour_box.setToolTip(
            "The column drawn in grey: its 0.5th percentile black, 99.5th white, over the "
            "pixels shown. A NaN cell is tinted, so it never reads as black.")
        self.colour_box.currentTextChanged.connect(self._on_colour)
        head2.addWidget(self.colour_box, 1)
        self.size_box = QSpinBox()
        self.size_box.setRange(2, 32)
        self.size_box.setValue(10)
        self.size_box.setSuffix(" px")
        self.size_box.setToolTip("Size: the side of one pixel's square on screen.")
        self.size_box.valueChanged.connect(self._on_size)
        head2.addWidget(self.size_box)
        v.addLayout(head2)
        body = QHBoxLayout()
        body.setSpacing(2)
        self.canvas = _StripCanvas(self.data)
        self.canvas.setMinimumWidth(60)
        self.canvas.hovered.connect(self._on_hover)
        self.canvas.layout_changed.connect(self._sync_scroll)
        self.canvas.scrolled.connect(self._on_wheel)
        body.addWidget(self.canvas, 1)
        self.scroll = QScrollBar(Qt.Vertical)
        self.scroll.valueChanged.connect(self.canvas.set_top)
        body.addWidget(self.scroll)
        v.addLayout(body, 1)
        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        # a QLabel's minimum width is its whole text, which would stop a docked panel from
        # ever shrinking below the status line; ignored horizontally, it clips instead
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        v.addWidget(self.status)
        self.card = HoverCard(self)

    # ── content ───────────────────────────────────────────────────────────────
    def set_table(self, cols: Dict[str, np.ndarray],
                  rows: Optional[Sequence[int]] = None) -> None:
        self.data.set_table(cols, rows)
        for box, names, cur in ((self.sort_box, self.data.names, self.data.sort_by),
                                (self.colour_box, self.data.numeric_names(),
                                 self.data.colour_by)):
            box.blockSignals(True)
            box.clear()
            box.addItems(names)
            if cur in names:
                box.setCurrentText(cur)
            box.blockSignals(False)
        self.order_btn.setText("↑" if self.data.ascending else "↓")
        self.canvas.top = 0
        self._refresh()

    def _refresh(self, keep_first: bool = False) -> None:
        self.card.hide()
        self.canvas.cell = int(self.size_box.value())
        self.canvas._hover = -1
        self.canvas.relayout(keep_first=keep_first)
        lo, hi = self.data.window
        arrow = "↑" if self.data.ascending else "↓"
        self.status.setText(
            f"{self.data.n:,} pixels · sorted by {self.data.sort_by} {arrow} · grey = "
            f"{self.data.colour_by} {_fmt(lo)} – {_fmt(hi)}  (hover a pixel for its row)")

    def _sync_scroll(self) -> None:
        c = self.canvas
        self.scroll.blockSignals(True)
        self.scroll.setRange(0, c.max_top())
        self.scroll.setPageStep(c.visible_lines())
        self.scroll.setSingleStep(1)
        self.scroll.setValue(c.top)
        self.scroll.blockSignals(False)

    def _on_wheel(self, top: int) -> None:
        self.scroll.blockSignals(True)
        self.scroll.setValue(top)
        self.scroll.blockSignals(False)
        self.card.hide()

    def _on_sort(self, name: str) -> None:
        if name and name != self.data.sort_by:
            self.data.sort_by = name
            self.data.resort()
            self.canvas.top = 0
            self._refresh()

    def _on_order(self) -> None:
        self.data.ascending = not self.data.ascending
        self.order_btn.setText("↑" if self.data.ascending else "↓")
        self.data.resort()
        self.canvas.top = 0
        self._refresh()

    def _on_colour(self, name: str) -> None:
        if name and name != self.data.colour_by:
            self.data.colour_by = name
            self.data.recolour()
            self._refresh(keep_first=True)

    def _on_size(self, _v: int) -> None:
        self._refresh(keep_first=True)

    # ── hover ─────────────────────────────────────────────────────────────────
    def card_content(self, pos: int):
        """``(title, rows, swatch)`` the card shows for strip position ``pos`` — separate
        from the showing so a headless probe can read what a hover would say."""
        d = self.data
        row = int(d.order[pos])
        vals = d.row_values(row)
        lab = dict(vals).get("label_id")
        title = f"pixel {pos + 1:,} of {d.n:,} · row {row}"
        if lab is not None:
            title += f" · label {_fmt(lab)}"
        g = int(d.grey[pos])
        swatch = _nan_colour() if d.missing[pos] else QColor(g, g, g)
        return title, vals, swatch

    def _on_hover(self, pos: int) -> None:
        if pos < 0:
            self.card.hide()
            return
        self.card.set_content(*self.card_content(pos))
        self.card.show_at(QCursor.pos())

    def hideEvent(self, ev) -> None:                         # noqa: N802 — Qt override
        try:
            self.card.hide()
        except RuntimeError:                # the card's C++ side is gone (app shutdown)
            pass
        super().hideEvent(ev)


__all__ = ["PixelStripView", "StripData", "HoverCard"]
