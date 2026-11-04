"""The figure page as a Qt scene (V5.00 step 3): page units, zoom, selection, export.

**Units.** The scene is in page units, :data:`MM` per millimetre (so a 180 × 120 mm page is
1800 × 1200 units), y down. A point is :data:`PT` units. Every item paints with plain
(non-cosmetic) pens in these units, so zooming the view magnifies the drawing exactly as
printing would, and :meth:`FigureScene.render_page` draws the same scene into a raster at
any dpi or into a vector device — the preview and the export are one set of lines.

**Export.** :meth:`FigureScene.export` writes PNG / TIFF (``QImage`` at the page's dpi, with
the dots-per-metre recorded), SVG (``QSvgGenerator``: text stays text, strokes stay paths)
and PDF (``QPdfWriter``: a page of the figure's millimetres, fonts embedded). Selection
outlines and handles are never exported: items check :attr:`FigureScene.exporting`.
"""
from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, QSize, QSizeF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPageSize, QPainter, QPdfWriter, QPen,
                           QTransform)
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from nodelab_v2 import theme as T

#: scene units per millimetre
MM = 10.0
#: scene units per point
PT = 25.4 / 72.0 * MM


def mm(v: float) -> float:
    """Millimetres → scene units."""
    return float(v) * MM


def to_mm(u: float) -> float:
    """Scene units → millimetres."""
    return float(u) / MM


class FigureScene(QGraphicsScene):
    """The page. Items are :class:`~nodelab_v2.figure_editor.items.FigureItem`\\ s; the
    scene owns the page rect, the background and the export."""

    #: an item was moved or resized by the mouse — ``(item id, [x, y, w, h] mm)``
    geometry_edited = Signal(str, object)
    #: a key the page handles: ``delete``, ``nudge`` (with ``(dx_mm, dy_mm)``)
    key_action = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.page_w_mm = 85.0
        self.page_h_mm = 65.0
        self.dark = False
        self.grid_mm = 1.0
        self.snap = True
        #: True while :meth:`render_page` draws — items leave out their selection furniture
        self.exporting = False
        self.set_page(85.0, 65.0)

    # ── the page ───────────────────────────────────────────────────────────────
    def set_page(self, w_mm: float, h_mm: float, *, dark: Optional[bool] = None,
                 grid_mm: Optional[float] = None) -> None:
        self.page_w_mm, self.page_h_mm = float(w_mm), float(h_mm)
        if dark is not None:
            self.dark = bool(dark)
        if grid_mm is not None:
            self.grid_mm = float(grid_mm)
        pad = mm(20.0)
        self.setSceneRect(QRectF(-pad, -pad, mm(self.page_w_mm) + 2 * pad,
                                 mm(self.page_h_mm) + 2 * pad))
        self.update()

    def page_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, mm(self.page_w_mm), mm(self.page_h_mm))

    def page_color(self) -> QColor:
        return QColor("#1c1f24") if self.dark else QColor("#ffffff")

    def snap_pt(self, p: QPointF) -> QPointF:
        if not self.snap or self.grid_mm <= 0:
            return p
        g = mm(self.grid_mm)
        return QPointF(round(p.x() / g) * g, round(p.y() / g) * g)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:   # noqa: N802
        if self.exporting:
            painter.fillRect(self.page_rect(), self.page_color())
            return
        painter.fillRect(rect, T.BG)
        page = self.page_rect()
        # the page's shadow, then the page
        painter.fillRect(page.translated(mm(0.8), mm(0.8)), QColor(0, 0, 0, 70))
        painter.fillRect(page, self.page_color())
        # a faint grid every 5 mm, dots every 1 mm at close zoom
        painter.save()
        painter.setClipRect(page)
        pen = QPen(QColor(0, 0, 0, 22) if not self.dark else QColor(255, 255, 255, 22), 0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        step = mm(5.0)
        x = 0.0
        while x <= page.width() + 0.5:
            painter.drawLine(QPointF(x, 0.0), QPointF(x, page.height()))
            x += step
        y = 0.0
        while y <= page.height() + 0.5:
            painter.drawLine(QPointF(0.0, y), QPointF(page.width(), y))
            y += step
        painter.restore()
        pen = QPen(T.BORDER_HI, 0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(page)

    # ── keys ───────────────────────────────────────────────────────────────────
    def keyPressEvent(self, e) -> None:                                   # noqa: N802
        k = e.key()
        step = 1.0 if not (e.modifiers() & Qt.ShiftModifier) else 5.0
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.key_action.emit("delete", None)
            e.accept()
            return
        moves = {Qt.Key_Left: (-step, 0.0), Qt.Key_Right: (step, 0.0),
                 Qt.Key_Up: (0.0, -step), Qt.Key_Down: (0.0, step)}
        if k in moves:
            self.key_action.emit("nudge", moves[k])
            e.accept()
            return
        super().keyPressEvent(e)

    # ── export ─────────────────────────────────────────────────────────────────
    def render_page(self, painter: QPainter, target: QRectF) -> None:
        """Draw the page into ``target`` (device units) without any editor furniture."""
        self.clearSelection()
        self.exporting = True
        try:
            self.render(painter, target, self.page_rect(), Qt.IgnoreAspectRatio)
        finally:
            self.exporting = False

    def page_image(self, dpi: int) -> QImage:
        """The page as an RGB image at ``dpi``, dots-per-metre recorded."""
        px_per_mm = float(dpi) / 25.4
        w = max(1, int(round(self.page_w_mm * px_per_mm)))
        h = max(1, int(round(self.page_h_mm * px_per_mm)))
        img = QImage(w, h, QImage.Format_RGB32)
        img.fill(self.page_color())
        dpm = int(round(float(dpi) / 0.0254))
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        self.render_page(p, QRectF(0, 0, w, h))
        p.end()
        return img

    def export(self, path: str, fmt: str, dpi: int = 300) -> str:
        """Write the page to ``path`` as ``fmt`` (``png`` | ``tiff`` | ``svg`` | ``pdf``)."""
        fmt = str(fmt).lower().lstrip(".")
        if fmt in ("png", "tif", "tiff", "jpg", "jpeg"):
            img = self.page_image(int(dpi))
            if not img.save(path):
                raise OSError(f"could not write {path}")
            return path
        if fmt == "svg":
            from PySide6.QtSvg import QSvgGenerator
            gen = QSvgGenerator()
            gen.setFileName(path)
            px_per_mm = float(dpi) / 25.4
            w = int(round(self.page_w_mm * px_per_mm))
            h = int(round(self.page_h_mm * px_per_mm))
            gen.setSize(QSize(w, h))
            gen.setViewBox(QRectF(0, 0, w, h))
            gen.setResolution(int(dpi))
            gen.setTitle("figure")
            gen.setDescription("ND2Studios figure")
            p = QPainter(gen)
            p.setRenderHint(QPainter.Antialiasing, True)
            self.render_page(p, QRectF(0, 0, w, h))
            p.end()
            return path
        if fmt == "pdf":
            writer = QPdfWriter(path)
            writer.setPageSize(QPageSize(QSizeF(self.page_w_mm, self.page_h_mm),
                                         QPageSize.Millimeter))
            writer.setResolution(int(dpi))
            writer.setPageMargins(__import__("PySide6.QtCore", fromlist=["QMarginsF"])
                                  .QMarginsF(0, 0, 0, 0))
            px_per_mm = float(dpi) / 25.4
            w = self.page_w_mm * px_per_mm
            h = self.page_h_mm * px_per_mm
            p = QPainter(writer)
            p.setRenderHint(QPainter.Antialiasing, True)
            self.render_page(p, QRectF(0, 0, w, h))
            p.end()
            return path
        raise ValueError(f"unknown export format {fmt!r} (png, tiff, svg, pdf)")


class FigureView(QGraphicsView):
    """The page on screen: wheel zoom about the pointer, middle-drag pan, rubber-band
    selection, and a fit-to-page."""

    #: the user pressed in this view (the canvas becomes the one worked in)
    pressed = Signal()
    #: alt-drag finished on an image item: ``(item id, QRectF in the item's coords)``
    region_dragged = Signal(str, object)
    #: a source was dropped on the page: ``(source value, scene pos)``
    source_dropped = Signal(str, object)
    #: a column was dropped on an item: ``(item id, payload dict)``
    column_dropped = Signal(str, object)

    def __init__(self, scene: FigureScene) -> None:
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform
                            | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setAcceptDrops(True)
        self._panning = False
        self._pan_from = QPointF()
        self._alt_drag: Optional[Tuple[Any, QPointF]] = None
        self._alt_rect: Optional[QRectF] = None
        self.restyle()

    def restyle(self) -> None:
        self.setBackgroundBrush(QBrush(T.BG))
        self.viewport().update()

    def wheelEvent(self, e) -> None:                                      # noqa: N802
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        e.accept()

    def fit_page(self) -> None:
        sc = self.scene()
        if isinstance(sc, FigureScene):
            self.fitInView(sc.page_rect().adjusted(-mm(4), -mm(4), mm(4), mm(4)),
                           Qt.KeepAspectRatio)

    def zoom_1(self) -> None:
        """One scene unit per screen pixel at the device's resolution ≈ the page at 1:1."""
        dpi = float(self.logicalDpiX() or 96.0)
        k = dpi / 25.4 / MM
        self.setTransform(QTransform.fromScale(k, k))

    # ── mouse ──────────────────────────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:                                 # noqa: N802
        self.pressed.emit()
        if e.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_from = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        if e.button() == Qt.LeftButton and (e.modifiers() & Qt.AltModifier):
            item = self.itemAt(e.position().toPoint())
            while item is not None and not hasattr(item, "item_id"):
                item = item.parentItem()
            if item is not None and getattr(item, "accepts_region_drag", False):
                self._alt_drag = (item, item.mapFromScene(self.mapToScene(e.position().toPoint())))
                self._alt_rect = QRectF(self._alt_drag[1], self._alt_drag[1])
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:                                  # noqa: N802
        if self._panning:
            d = e.position() - self._pan_from
            self._pan_from = e.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            e.accept()
            return
        if self._alt_drag is not None:
            item, origin = self._alt_drag
            cur = item.mapFromScene(self.mapToScene(e.position().toPoint()))
            self._alt_rect = QRectF(origin, cur).normalized()
            item.set_drag_box(self._alt_rect)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:                               # noqa: N802
        if self._panning and e.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            e.accept()
            return
        if self._alt_drag is not None and e.button() == Qt.LeftButton:
            item, _origin = self._alt_drag
            rect = self._alt_rect
            self._alt_drag, self._alt_rect = None, None
            item.set_drag_box(None)
            if rect is not None and rect.width() > mm(1) and rect.height() > mm(1):
                self.region_dragged.emit(item.item_id, rect)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # ── drops ──────────────────────────────────────────────────────────────────
    def dragEnterEvent(self, e) -> None:                                  # noqa: N802
        md = e.mimeData()
        if md.hasFormat(SOURCE_MIME) or md.hasFormat(COLUMN_MIME):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:                                   # noqa: N802
        md = e.mimeData()
        if md.hasFormat(SOURCE_MIME) or md.hasFormat(COLUMN_MIME):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:                                       # noqa: N802
        md = e.mimeData()
        pos = self.mapToScene(e.position().toPoint())
        if md.hasFormat(SOURCE_MIME):
            self.source_dropped.emit(bytes(md.data(SOURCE_MIME)).decode("utf8"), pos)
            e.acceptProposedAction()
            return
        if md.hasFormat(COLUMN_MIME):
            import json
            item = self.itemAt(e.position().toPoint())
            while item is not None and not hasattr(item, "item_id"):
                item = item.parentItem()
            if item is not None:
                self.column_dropped.emit(item.item_id,
                                         json.loads(bytes(md.data(COLUMN_MIME)).decode("utf8")))
            e.acceptProposedAction()
            return
        super().dropEvent(e)


#: drag payloads of the Sources panel
SOURCE_MIME = "application/x-nd2-figure-source"
COLUMN_MIME = "application/x-nd2-figure-column"

__all__ = ["MM", "PT", "mm", "to_mm", "FigureScene", "FigureView", "SOURCE_MIME",
           "COLUMN_MIME"]
