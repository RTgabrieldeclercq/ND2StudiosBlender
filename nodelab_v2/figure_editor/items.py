"""The items on the figure page (V5.00 step 3).

One :class:`FigureItem` per document item, keyed by the item's id, positioned and sized in
page units from its ``rect_mm``. Every item can be moved and resized with the mouse
(eight handles when selected); an image item keeps its picture's aspect while resizing.
The scene reports a finished drag through :attr:`FigureScene.geometry_edited`, and the
panel commits it — items never write the document themselves.

Painting is in page units with plain pens, so what is drawn on screen is what is
exported. Selection furniture (outline, handles) is skipped while the scene exports.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImage, QPainter,
                           QPainterPath, QPen, QPolygonF)
from PySide6.QtWidgets import QGraphicsItem, QStyle

from nodelab_v2 import overlays as OV
from nodelab_v2 import theme as T
from nodelab_v2.figure_editor import render as R
from nodelab_v2.figure_editor.scene import MM, PT, mm, to_mm

HANDLE = 6.0            # handle size, scene units
MIN_SIZE_MM = 3.0

_FONT_FAMILY = {"sans": "DejaVu Sans", "serif": "DejaVu Serif"}


def page_font(style: Mapping[str, Any], pt: Optional[float] = None, *,
              bold: bool = False) -> QFont:
    """A font in page units: ``pt`` points of the page's family."""
    f = QFont(_FONT_FAMILY.get(str(style.get("font", "sans")), "DejaVu Sans"))
    size = float(pt if pt is not None else style.get("font_pt", 8.0))
    f.setPixelSize(max(1, int(round(size * PT))))
    f.setBold(bool(bold))
    return f


def ink(style: Mapping[str, Any]) -> QColor:
    return QColor("#e6e6e6") if style.get("dark") else QColor("#111111")


class FigureItem(QGraphicsItem):
    """Base: a movable, resizable rect in page units holding its document item."""

    #: handles: (name, (fx, fy)) — the fraction of the rect each one sits at
    HANDLES = (("nw", (0, 0)), ("n", (0.5, 0)), ("ne", (1, 0)), ("e", (1, 0.5)),
               ("se", (1, 1)), ("s", (0.5, 1)), ("sw", (0, 1)), ("w", (0, 0.5)))
    keep_aspect = False
    accepts_region_drag = False

    def __init__(self, spec: Mapping[str, Any], style: Mapping[str, Any]) -> None:
        super().__init__()
        self.item_id = str(spec["id"])
        self.spec: Dict[str, Any] = dict(spec)
        self.style: Dict[str, Any] = dict(style)
        self._w = 1.0
        self._h = 1.0
        self._resize: Optional[Tuple[str, QRectF, QPointF]] = None
        self._moved = False
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.apply_spec(spec, style)

    # ── geometry ───────────────────────────────────────────────────────────────
    def apply_spec(self, spec: Mapping[str, Any], style: Mapping[str, Any]) -> None:
        self.spec = dict(spec)
        self.style = dict(style)
        x, y, w, h = (float(v) for v in spec["rect_mm"])
        self.prepareGeometryChange()
        self.setPos(mm(x), mm(y))
        self._w, self._h = mm(w), mm(h)
        self.setZValue(float(spec.get("z", 0) or 0))
        self.setFlag(QGraphicsItem.ItemIsMovable, not bool(spec.get("locked")))
        self.update()

    def rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._w, self._h)

    def rect_mm(self) -> List[float]:
        p = self.pos()
        return [to_mm(p.x()), to_mm(p.y()), to_mm(self._w), to_mm(self._h)]

    def boundingRect(self) -> QRectF:                                     # noqa: N802
        return self.rect().adjusted(-HANDLE, -HANDLE, HANDLE, HANDLE)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect().adjusted(-HANDLE / 2, -HANDLE / 2, HANDLE / 2, HANDLE / 2))
        return path

    def _scene_exporting(self) -> bool:
        sc = self.scene()
        return bool(getattr(sc, "exporting", False))

    # ── painting ───────────────────────────────────────────────────────────────
    def paint(self, p: QPainter, option, widget=None) -> None:
        self.paint_content(p)
        if self.isSelected() and not self._scene_exporting():
            self.paint_selection(p)

    def paint_content(self, p: QPainter) -> None:
        pen = QPen(QColor(T.MUTED), 0)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(self.rect())

    def paint_selection(self, p: QPainter) -> None:
        pen = QPen(QColor(T.ACCENT), 0)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(self.rect())
        p.setBrush(QBrush(QColor(T.ACCENT)))
        for _name, (fx, fy) in self.HANDLES:
            c = QPointF(self._w * fx, self._h * fy)
            p.drawRect(QRectF(c.x() - HANDLE / 2, c.y() - HANDLE / 2, HANDLE, HANDLE))

    def paint_letter(self, p: QPainter) -> None:
        letter = str(self.spec.get("letter") or "")
        if not letter:
            return
        f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 1.25, bold=True)
        p.setFont(f)
        fm = QFontMetricsF(f)
        pad = mm(0.8)
        p.setPen(QPen(ink(self.style), 0))
        p.drawText(QPointF(pad, pad + fm.ascent()), letter)

    # ── handles and resizing ───────────────────────────────────────────────────
    def handle_at(self, pos: QPointF) -> Optional[str]:
        if not self.isSelected():
            return None
        for name, (fx, fy) in self.HANDLES:
            c = QPointF(self._w * fx, self._h * fy)
            if abs(pos.x() - c.x()) <= HANDLE and abs(pos.y() - c.y()) <= HANDLE:
                return name
        return None

    def hoverMoveEvent(self, e) -> None:                                  # noqa: N802
        h = self.handle_at(e.pos())
        cursors = {"n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor, "e": Qt.SizeHorCursor,
                   "w": Qt.SizeHorCursor, "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
                   "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor}
        self.setCursor(cursors.get(h, Qt.ArrowCursor if not self.isSelected()
                                   else Qt.SizeAllCursor))
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e) -> None:                                 # noqa: N802
        h = self.handle_at(e.pos()) if e.button() == Qt.LeftButton else None
        if h is not None and not self.spec.get("locked"):
            self._resize = (h, QRectF(self.pos(), self.rect().size()), e.scenePos())
            e.accept()
            return
        self._moved = False
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:                                  # noqa: N802
        if self._resize is None:
            self._moved = True
            super().mouseMoveEvent(e)
            return
        h, start, origin = self._resize
        d = e.scenePos() - origin
        x, y, w, hgt = start.x(), start.y(), start.width(), start.height()
        if "e" in h:
            w = max(mm(MIN_SIZE_MM), start.width() + d.x())
        if "w" in h:
            w = max(mm(MIN_SIZE_MM), start.width() - d.x())
            x = start.x() + start.width() - w
        if "s" in h:
            hgt = max(mm(MIN_SIZE_MM), start.height() + d.y())
        if "n" in h:
            hgt = max(mm(MIN_SIZE_MM), start.height() - d.y())
            y = start.y() + start.height() - hgt
        if self.keep_aspect and start.width() > 0 and start.height() > 0:
            ratio = start.height() / start.width()
            if h in ("e", "w", "ne", "nw", "se", "sw"):
                hgt = w * ratio
                if "n" in h:
                    y = start.y() + start.height() - hgt
            else:
                w = hgt / ratio
                if "w" in h:
                    x = start.x() + start.width() - w
        sc = self.scene()
        snap = getattr(sc, "snap_pt", None)
        if snap is not None:
            tl = snap(QPointF(x, y))
            x, y = tl.x(), tl.y()
        self.prepareGeometryChange()
        self.setPos(x, y)
        self._w, self._h = w, hgt
        self.on_resized()
        self.update()
        e.accept()

    def mouseReleaseEvent(self, e) -> None:                               # noqa: N802
        if self._resize is not None:
            self._resize = None
            self._emit_geometry()
            e.accept()
            return
        super().mouseReleaseEvent(e)
        if self._moved:
            self._moved = False
            self._emit_geometry()

    def itemChange(self, change, value):                                  # noqa: N802
        if change == QGraphicsItem.ItemPositionChange and self.scene() is not None:
            snap = getattr(self.scene(), "snap_pt", None)
            if snap is not None and self._resize is None:
                return snap(value)
        return super().itemChange(change, value)

    def _emit_geometry(self) -> None:
        sc = self.scene()
        if sc is not None and hasattr(sc, "geometry_edited"):
            sc.geometry_edited.emit(self.item_id, self.rect_mm())

    def on_resized(self) -> None:
        """A subclass hook: the rect changed under a drag."""


# ── images ────────────────────────────────────────────────────────────────────────────
class ImageItem(FigureItem):
    """An image or a montage tile host: the composited picture, its overlays, its decor."""

    keep_aspect = True
    accepts_region_drag = True

    def __init__(self, spec, style) -> None:
        self.rendered: Optional[R.Rendered] = None
        self.image_spec: Dict[str, Any] = {}
        self.note = ""
        self._drag_box: Optional[QRectF] = None
        self._renderer = OV.OverlayRenderer()
        #: zoom boxes to draw on this picture: ``[(child id, (fy0, fy1, fx0, fx1), box)]``
        self.zoom_boxes: List[Tuple[str, Tuple[float, float, float, float], Dict[str, Any]]] = []
        super().__init__(spec, style)

    def set_rendered(self, r: Optional[R.Rendered], image_spec: Mapping[str, Any],
                     note: str = "") -> None:
        self.rendered = r
        self.image_spec = dict(image_spec)
        self.note = note
        self._renderer.invalidate()
        self.update()

    def set_drag_box(self, box: Optional[QRectF]) -> None:
        self._drag_box = box
        self.update()

    def set_highlight(self, yx: Optional[Tuple[float, float]]) -> None:
        """Ring one object at structure pixels ``(y, x)`` (a chart hover), or none."""
        self._highlight = yx
        self.update()

    def structure_to_item(self, y: float, x: float) -> Optional[QPointF]:
        """A structure coordinate (full-resolution px, centre on the integer) → item units,
        the mapping :func:`render.paint_overlays` uses; ``None`` without a render."""
        r = self.rendered
        if r is None:
            return None
        pr = self.picture_rect()
        H, W = r.level_shape
        ph, pw = r.plane_shape
        fy0, _fy1, fx0, _fx1 = r.window
        sy, sx = H / max(1, r.full_shape[0]), W / max(1, r.full_shape[1])
        px = (x + 0.5) * sx - fx0 * W
        py = (y + 0.5) * sy - fy0 * H
        xx = px * pr.width() / max(1, pw)
        yy = py * pr.height() / max(1, ph)
        if self.image_spec.get("flip_x"):
            xx = pr.width() - xx
        if self.image_spec.get("flip_y"):
            yy = pr.height() - yy
        return QPointF(pr.x() + xx, pr.y() + yy)

    def paint_highlight(self, p: QPainter) -> None:
        yx = getattr(self, "_highlight", None)
        if yx is None or self.rendered is None:
            return
        pt = self.structure_to_item(float(yx[0]), float(yx[1]))
        if pt is None or not self.picture_rect().contains(pt):
            return
        pen = QPen(QColor(T.ACCENT), max(0.6 * PT, mm(0.25)))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        r = max(mm(1.2), self.picture_rect().width() * 0.03)
        p.drawEllipse(pt, r, r)

    def picture_rect(self) -> QRectF:
        """Where the picture sits inside the item: the item rect, the picture fitted."""
        r = self.rect()
        if self.rendered is None:
            return r
        ph, pw = self.rendered.plane_shape
        if ph <= 0 or pw <= 0:
            return r
        k = min(r.width() / pw, r.height() / ph)
        w, h = pw * k, ph * k
        return QRectF(r.x() + (r.width() - w) / 2, r.y() + (r.height() - h) / 2, w, h)

    def stroke_scale(self) -> float:
        """Points → page units: the overlay looks are in points."""
        return PT

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        if self.rendered is None:
            p.fillRect(r, QColor("#2a2f36") if not self.style.get("dark") else QColor("#111"))
            pen = QPen(QColor(T.MUTED), 0)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            f = page_font(self.style, 7.0)
            p.setFont(f)
            p.setPen(QPen(QColor("#c9d2de"), 0))
            p.drawText(r.adjusted(mm(1), mm(1), -mm(1), -mm(1)),
                       Qt.AlignCenter | Qt.TextWordWrap, self.note or "not computed yet")
            self.paint_letter(p)
            return
        pr = self.picture_rect()
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(pr, self.rendered.image)
        p.restore()
        R.paint_overlays(p, self.rendered, pr, self._renderer, stroke_scale=self.stroke_scale(),
                         flip_x=bool(self.image_spec.get("flip_x")),
                         flip_y=bool(self.image_spec.get("flip_y")))
        self.paint_zoom_boxes(p, pr)
        self.paint_decor(p, pr)
        self.paint_highlight(p)
        self.paint_letter(p)
        if self._drag_box is not None:
            pen = QPen(QColor(T.ACCENT), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QColor(111, 227, 255, 40))
            p.drawRect(self._drag_box)

    def paint_zoom_boxes(self, p: QPainter, pr: QRectF) -> None:
        for _child, win, box in self.zoom_boxes:
            fy0, fy1, fx0, fx1 = win
            rect = QRectF(pr.x() + fx0 * pr.width(), pr.y() + fy0 * pr.height(),
                          (fx1 - fx0) * pr.width(), (fy1 - fy0) * pr.height())
            pen = QPen(QColor(str(box.get("color", "#ffffff"))),
                       float(box.get("width_pt", 0.75)) * PT)
            pen.setStyle(Qt.DashLine if box.get("style", "dashed") == "dashed" else Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)

    def zoom_box_rect(self, win: Tuple[float, float, float, float]) -> QRectF:
        """The scene rect of a zoom box window on this picture (for leader lines)."""
        pr = self.picture_rect()
        fy0, fy1, fx0, fx1 = win
        return self.mapRectToScene(QRectF(pr.x() + fx0 * pr.width(), pr.y() + fy0 * pr.height(),
                                          (fx1 - fx0) * pr.width(), (fy1 - fy0) * pr.height()))

    def picture_frac(self, item_pos: QPointF) -> Tuple[float, float]:
        """An item-coordinate point as fractions ``(fy, fx)`` of the picture."""
        pr = self.picture_rect()
        fx = (item_pos.x() - pr.x()) / max(1e-9, pr.width())
        fy = (item_pos.y() - pr.y()) / max(1e-9, pr.height())
        return min(max(fy, 0.0), 1.0), min(max(fx, 0.0), 1.0)

    def paint_decor(self, p: QPainter, pr: QRectF) -> None:
        decor = self.image_spec.get("decor") or {}
        r = self.rendered
        col = QColor("#ffffff")
        # scale bar: length in µm → picture units through the level plane's pixel size
        sb = decor.get("scale_bar_um")
        if sb and r is not None and r.pixel_size_um:
            px_per_um = pr.width() / max(1, r.plane_shape[1]) / float(r.pixel_size_um)
            length = float(sb) * px_per_um
            if 0 < length <= pr.width():
                thick = max(0.4 * PT, mm(0.35))
                x1 = pr.right() - mm(1.5)
                y = pr.bottom() - mm(1.5)
                p.fillRect(QRectF(x1 - length, y - thick, length, thick), col)
                if decor.get("scale_bar_label", True):
                    f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.85)
                    p.setFont(f)
                    fm = QFontMetricsF(f)
                    text = f"{float(sb):g} µm"
                    p.setPen(QPen(col, 0))
                    p.drawText(QPointF(x1 - length / 2 - fm.horizontalAdvance(text) / 2,
                                       y - thick - mm(0.4)), text)
        # time stamp (top left)
        ts = decor.get("time_stamp")
        if ts and r is not None:
            text = self._time_text(str(ts))
            if text:
                f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.85)
                p.setFont(f)
                fm = QFontMetricsF(f)
                p.setPen(QPen(col, 0))
                p.drawText(QPointF(pr.x() + mm(1.5), pr.y() + mm(1.2) + fm.ascent()), text)
        # channel labels (top right), in their colours
        if decor.get("channel_labels") and r is not None:
            f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.85)
            p.setFont(f)
            fm = QFontMetricsF(f)
            y = pr.y() + mm(1.2) + fm.ascent()
            for ch in self.image_spec.get("channels") or []:
                ci = int(ch.get("c", 0))
                label = str(ch.get("label") or self._channel_name(ci))
                rgb = r.colors.get(ci, (255, 255, 255))
                p.setPen(QPen(QColor(*rgb), 0))
                p.drawText(QPointF(pr.right() - mm(1.5) - fm.horizontalAdvance(label), y), label)
                y += fm.height()

    def _channel_name(self, c: int) -> str:
        names = list(self.image_spec.get("_channel_names") or [])
        return str(names[c]) if c < len(names) and names[c] else f"ch{c + 1}"

    def _time_text(self, mode: str) -> str:
        md = self.image_spec.get("_metadata") or {}
        r = self.rendered
        t = r.coords[1] if r is not None else 0
        if mode == "frame":
            return f"t = {t}"
        dt = md.get("dt_s")
        jd = md.get("frame_time_jd")
        if mode == "clock" and isinstance(jd, (list, tuple)) and t < len(jd) and jd[t]:
            try:
                import datetime as _dt
                base = _dt.datetime(2000, 1, 1, 12) + _dt.timedelta(days=float(jd[t]) - 2451545.0)
                return base.strftime("%H:%M:%S")
            except (TypeError, ValueError):
                pass
        seconds = None
        if isinstance(jd, (list, tuple)) and t < len(jd) and jd[t] and jd[0]:
            try:
                seconds = (float(jd[t]) - float(jd[0])) * 86400.0
            except (TypeError, ValueError):
                seconds = None
        if seconds is None and dt:
            try:
                seconds = float(dt) * t
            except (TypeError, ValueError):
                seconds = None
        if seconds is None:
            return f"t = {t}"
        if seconds < 3600:
            return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
        return f"{int(seconds // 3600)}:{int((seconds % 3600) // 60):02d} h"


class MontageItem(ImageItem):
    """A grid of tiles of one image spec along one axis; each tile its own render."""

    keep_aspect = False

    def __init__(self, spec, style) -> None:
        self.tiles: List[Tuple[Tuple[int, int, Any, int], Optional[R.Rendered], str]] = []
        super().__init__(spec, style)

    def set_tiles(self, tiles, image_spec: Mapping[str, Any]) -> None:
        self.tiles = list(tiles)
        self.image_spec = dict(image_spec)
        self.rendered = next((r for _c, r, _l in self.tiles if r is not None), None)
        self._renderer.invalidate()
        self.update()

    def grid(self) -> Tuple[int, int]:
        n = max(1, len(self.tiles))
        cols = int((self.spec.get("montage") or {}).get("columns") or 0)
        if cols <= 0:
            cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / cols))
        return rows, cols

    def tile_rects(self) -> List[QRectF]:
        rows, cols = self.grid()
        gap = mm(float((self.spec.get("montage") or {}).get("gap_mm", 0.5)))
        r = self.rect()
        tw = (r.width() - gap * (cols - 1)) / cols
        th = (r.height() - gap * (rows - 1)) / rows
        out = []
        for k in range(len(self.tiles)):
            i, j = divmod(k, cols)
            out.append(QRectF(r.x() + j * (tw + gap), r.y() + i * (th + gap), tw, th))
        return out

    def picture_rect(self) -> QRectF:
        return self.rect()

    def paint_content(self, p: QPainter) -> None:
        if not self.tiles:
            super().paint_content(p)
            return
        f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.85)
        fm = QFontMetricsF(f)
        for (coords, r, label), tr in zip(self.tiles, self.tile_rects()):
            if r is None:
                p.fillRect(tr, QColor("#2a2f36"))
                continue
            ph, pw = r.plane_shape
            k = min(tr.width() / max(1, pw), tr.height() / max(1, ph))
            w, h = pw * k, ph * k
            pr = QRectF(tr.x() + (tr.width() - w) / 2, tr.y() + (tr.height() - h) / 2, w, h)
            p.save()
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawImage(pr, r.image)
            p.restore()
            R.paint_overlays(p, r, pr, self._renderer, stroke_scale=self.stroke_scale())
            if label:
                p.setFont(f)
                p.setPen(QPen(QColor("#ffffff"), 0))
                p.drawText(QPointF(pr.x() + mm(1.0), pr.y() + mm(0.8) + fm.ascent()), label)
        self.paint_zoom_boxes_montage(p)
        self.paint_letter(p)
        if self._drag_box is not None:
            pen = QPen(QColor(T.ACCENT), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QColor(111, 227, 255, 40))
            p.drawRect(self._drag_box)

    def paint_zoom_boxes_montage(self, p: QPainter) -> None:
        rects = self.tile_rects()
        for _child, win, box in self.zoom_boxes:
            m = int(box.get("_tile", 0))
            if not 0 <= m < len(rects):
                continue
            tr = rects[m]
            r = self.tiles[m][1]
            if r is None:
                continue
            ph, pw = r.plane_shape
            k = min(tr.width() / max(1, pw), tr.height() / max(1, ph))
            w, h = pw * k, ph * k
            pr = QRectF(tr.x() + (tr.width() - w) / 2, tr.y() + (tr.height() - h) / 2, w, h)
            fy0, fy1, fx0, fx1 = win
            rect = QRectF(pr.x() + fx0 * pr.width(), pr.y() + fy0 * pr.height(),
                          (fx1 - fx0) * pr.width(), (fy1 - fy0) * pr.height())
            pen = QPen(QColor(str(box.get("color", "#ffffff"))),
                       float(box.get("width_pt", 0.75)) * PT)
            pen.setStyle(Qt.DashLine if box.get("style", "dashed") == "dashed" else Qt.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)

    def zoom_box_rect(self, win, tile: int = 0) -> QRectF:              # type: ignore[override]
        rects = self.tile_rects()
        if not rects:
            return self.mapRectToScene(self.rect())
        tr = rects[min(max(tile, 0), len(rects) - 1)]
        r = self.tiles[min(max(tile, 0), len(rects) - 1)][1]
        pr = tr
        if r is not None:
            ph, pw = r.plane_shape
            k = min(tr.width() / max(1, pw), tr.height() / max(1, ph))
            w, h = pw * k, ph * k
            pr = QRectF(tr.x() + (tr.width() - w) / 2, tr.y() + (tr.height() - h) / 2, w, h)
        fy0, fy1, fx0, fx1 = win
        return self.mapRectToScene(QRectF(pr.x() + fx0 * pr.width(), pr.y() + fy0 * pr.height(),
                                          (fx1 - fx0) * pr.width(), (fy1 - fy0) * pr.height()))

    def tile_at(self, item_pos: QPointF) -> int:
        for k, tr in enumerate(self.tile_rects()):
            if tr.contains(item_pos):
                return k
        return -1

    def picture_frac(self, item_pos: QPointF) -> Tuple[float, float]:   # type: ignore[override]
        k = self.tile_at(item_pos)
        rects = self.tile_rects()
        if k < 0 or not rects:
            return 0.0, 0.0
        tr = rects[k]
        r = self.tiles[k][1]
        pr = tr
        if r is not None:
            ph, pw = r.plane_shape
            kk = min(tr.width() / max(1, pw), tr.height() / max(1, ph))
            w, h = pw * kk, ph * kk
            pr = QRectF(tr.x() + (tr.width() - w) / 2, tr.y() + (tr.height() - h) / 2, w, h)
        fx = (item_pos.x() - pr.x()) / max(1e-9, pr.width())
        fy = (item_pos.y() - pr.y()) / max(1e-9, pr.height())
        return min(max(fy, 0.0), 1.0), min(max(fx, 0.0), 1.0)


# ── decorations ───────────────────────────────────────────────────────────────────────
class ColorbarItem(FigureItem):
    """A colour bar of another item's heat layer: ``lo``, ``hi``, ``cmap``, ``label`` are
    resolved by the panel from that item's render and set here."""

    def __init__(self, spec, style) -> None:
        self.lo, self.hi, self.cmap, self.label = 0.0, 1.0, "viridis", ""
        self.normalized_vmax: Optional[float] = None
        super().__init__(spec, style)

    def set_scale(self, lo: float, hi: float, cmap: str, label: str,
                  normalized_vmax: Optional[float] = None) -> None:
        self.lo, self.hi, self.cmap = float(lo), float(hi), str(cmap)
        self.label = str(label if label is not None else self.spec.get("label") or "")
        self.normalized_vmax = normalized_vmax
        self.update()

    def paint_content(self, p: QPainter) -> None:
        from nodegraph.catalog._shared.overlay_math import colormap_lut
        r = self.rect()
        vertical = str(self.spec.get("orientation", "horizontal")) == "vertical"
        f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.85)
        fm = QFontMetricsF(f)
        p.setFont(f)
        col = ink(self.style)
        lut = colormap_lut(self.cmap)
        ramp = np.repeat(lut[None, :, :], 2, axis=0) if not vertical \
            else np.repeat(lut[::-1][:, None, :], 2, axis=1)
        ramp = np.ascontiguousarray(np.concatenate(
            [ramp, np.full(ramp.shape[:2] + (1,), 255, np.uint8)], axis=2))
        img = QImage(ramp.data, ramp.shape[1], ramp.shape[0], 4 * ramp.shape[1],
                     QImage.Format_RGBA8888).copy()
        label_h = fm.height() if self.label else 0.0
        ticks_h = fm.height()
        if vertical:
            bar = QRectF(r.x(), r.y(), max(mm(1.5), r.width() * 0.35), r.height())
        else:
            bar = QRectF(r.x(), r.y() + label_h + mm(0.3), r.width(),
                         max(mm(1.0), r.height() - label_h - ticks_h - mm(0.6)))
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(bar, img)
        p.restore()
        p.setPen(QPen(col, 0.5 * PT))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar)
        n = max(2, int(self.spec.get("ticks", 5) or 5))
        vmax = self.normalized_vmax
        for k in range(n):
            frac = k / (n - 1)
            v = self.lo + frac * (self.hi - self.lo)
            text = f"{v / vmax:.2g}" if vmax else f"{v:.3g}"
            if vertical:
                y = bar.bottom() - frac * bar.height()
                p.drawLine(QPointF(bar.right(), y), QPointF(bar.right() + mm(0.6), y))
                p.drawText(QPointF(bar.right() + mm(0.9), y + fm.ascent() / 2 - 1), text)
            else:
                x = bar.x() + frac * bar.width()
                p.drawLine(QPointF(x, bar.bottom()), QPointF(x, bar.bottom() + mm(0.6)))
                tw = fm.horizontalAdvance(text)
                p.drawText(QPointF(x - tw / 2, bar.bottom() + mm(0.8) + fm.ascent()), text)
        if self.label:
            if vertical:
                p.save()
                p.translate(r.right(), r.y() + r.height() / 2)
                p.rotate(-90)
                p.drawText(QPointF(-fm.horizontalAdvance(self.label) / 2, 0), self.label)
                p.restore()
            else:
                p.drawText(QPointF(r.x(), r.y() + fm.ascent()), self.label)
        self.paint_letter(p)


class HistogramItem(FigureItem):
    """The intensity histogram of another item's channel with its display window marked
    — the display range the checklists ask a figure to state."""

    def __init__(self, spec, style) -> None:
        self.counts: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        self.window: Optional[Tuple[float, float]] = None
        self.color = QColor("#ffffff")
        super().__init__(spec, style)

    def set_data(self, plane: Optional[np.ndarray], window: Optional[Tuple[float, float]],
                 rgb: Tuple[int, int, int] = (255, 255, 255)) -> None:
        if plane is None:
            self.counts = self.edges = None
        else:
            a = np.asarray(plane, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                self.counts, self.edges = np.histogram(a, bins=128)
            else:
                self.counts = self.edges = None
        self.window = window
        self.color = QColor(*rgb)
        self.update()

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        col = ink(self.style)
        p.setPen(QPen(col, 0.4 * PT))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)
        if self.counts is None or self.edges is None:
            self.paint_letter(p)
            return
        c = self.counts.astype(float)
        if bool(self.spec.get("log", True)):
            c = np.log10(c + 1.0)
        top = float(c.max()) if c.size and c.max() > 0 else 1.0
        lo_v, hi_v = float(self.edges[0]), float(self.edges[-1])
        span = hi_v - lo_v if hi_v > lo_v else 1.0
        path = QPainterPath()
        path.moveTo(r.x(), r.bottom())
        for k, v in enumerate(c):
            x0 = r.x() + (self.edges[k] - lo_v) / span * r.width()
            x1 = r.x() + (self.edges[k + 1] - lo_v) / span * r.width()
            y = r.bottom() - v / top * (r.height() - mm(0.5))
            path.lineTo(x0, y)
            path.lineTo(x1, y)
        path.lineTo(r.right(), r.bottom())
        path.closeSubpath()
        fill = QColor(self.color)
        fill.setAlpha(140)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawPath(path)
        if self.window and bool(self.spec.get("show_window", True)):
            lo, hi = self.window
            x0 = r.x() + (lo - lo_v) / span * r.width()
            x1 = r.x() + (hi - lo_v) / span * r.width()
            pen = QPen(QColor(T.ACCENT) if not self.style.get("dark") else QColor("#6fe3ff"),
                       0.6 * PT)
            p.setPen(pen)
            p.drawLine(QPointF(x0, r.y()), QPointF(x0, r.bottom()))
            p.drawLine(QPointF(x1, r.y()), QPointF(x1, r.bottom()))
            f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.75)
            p.setFont(f)
            fm = QFontMetricsF(f)
            p.setPen(QPen(col, 0))
            p.drawText(QPointF(r.x() + mm(0.5), r.y() + mm(0.3) + fm.ascent()),
                       f"{lo:.3g}–{hi:.3g}")
        self.paint_letter(p)


class ScaleBarItem(FigureItem):
    """A standalone scale bar: its length in µm, drawn at the scale of the image it is
    ``of`` (the panel sets ``px_per_um`` of that image's picture)."""

    def __init__(self, spec, style) -> None:
        self.units_per_um: Optional[float] = None
        super().__init__(spec, style)

    def set_scale(self, units_per_um: Optional[float]) -> None:
        self.units_per_um = units_per_um
        self.update()

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        col = ink(self.style)
        length_um = float(self.spec.get("length_um", 20.0) or 20.0)
        if self.units_per_um:
            length = min(r.width(), length_um * self.units_per_um)
        else:
            length = r.width()
        thick = max(0.6 * PT, mm(0.4))
        p.fillRect(QRectF(r.x(), r.bottom() - thick, length, thick), col)
        if bool(self.spec.get("label", True)):
            f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.85)
            p.setFont(f)
            fm = QFontMetricsF(f)
            text = f"{length_um:g} µm" if self.units_per_um else "scale: no pixel size"
            p.setPen(QPen(col, 0))
            p.drawText(QPointF(r.x() + length / 2 - fm.horizontalAdvance(text) / 2,
                               r.bottom() - thick - mm(0.4)), text)
        self.paint_letter(p)


class TextItem(FigureItem):
    """Free text in the page's font (or its own point size), left / centre / right."""

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        pt = self.spec.get("pt")
        f = page_font(self.style, float(pt) if pt else None, bold=bool(self.spec.get("bold")))
        p.setFont(f)
        col = QColor(str(self.spec.get("color"))) if self.spec.get("color") else ink(self.style)
        p.setPen(QPen(col, 0))
        align = {"left": Qt.AlignLeft, "center": Qt.AlignHCenter, "right": Qt.AlignRight}
        p.drawText(r, align.get(str(self.spec.get("align", "left")), Qt.AlignLeft)
                   | Qt.AlignTop | Qt.TextWordWrap, str(self.spec.get("text") or ""))
        if not str(self.spec.get("text") or "").strip() and not self._scene_exporting():
            pen = QPen(QColor(T.MUTED), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DotLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)


class LetterItem(TextItem):
    """A panel letter: bold, a quarter larger than the page's type."""

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        pt = self.spec.get("pt")
        f = page_font(self.style, float(pt) if pt else float(self.style.get("font_pt", 8.0)) * 1.25,
                      bold=True)
        p.setFont(f)
        p.setPen(QPen(ink(self.style), 0))
        p.drawText(r, Qt.AlignLeft | Qt.AlignTop, str(self.spec.get("text") or ""))


class ShapeItem(FigureItem):
    """A rectangle or ellipse outline (an ROI, a highlight), optionally filled."""

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        pen = QPen(QColor(str(self.spec.get("color", "#ffffff"))),
                   float(self.spec.get("width_pt", 0.75)) * PT)
        dash = str(self.spec.get("dash", "solid"))
        pen.setStyle({"dashed": Qt.DashLine, "dotted": Qt.DotLine}.get(dash, Qt.SolidLine))
        p.setPen(pen)
        fill = self.spec.get("fill")
        p.setBrush(QColor(str(fill)) if fill else Qt.NoBrush)
        if str(self.spec.get("shape", "rect")) == "ellipse":
            p.drawEllipse(r)
        else:
            p.drawRect(r)


class LineItem(FigureItem):
    """A line from the item's top-left to ``to_mm`` (relative); an arrow when asked."""

    HANDLES = (("nw", (0, 0)), ("se", (1, 1)))
    arrow = False

    def end_point(self) -> QPointF:
        to = self.spec.get("to_mm") or [to_mm(self._w), to_mm(self._h)]
        return QPointF(mm(float(to[0])), mm(float(to[1])))

    def apply_spec(self, spec, style) -> None:
        super().apply_spec(spec, style)
        to = spec.get("to_mm")
        if to:
            self._w, self._h = max(1.0, abs(mm(float(to[0])))), max(1.0, abs(mm(float(to[1]))))

    def on_resized(self) -> None:
        self.spec["to_mm"] = [to_mm(self._w), to_mm(self._h)]

    def rect_mm(self) -> List[float]:
        out = super().rect_mm()
        return out

    def paint_content(self, p: QPainter) -> None:
        a = QPointF(0.0, 0.0)
        b = QPointF(self._w, self._h)
        col = QColor(str(self.spec.get("color", "#ffffff")))
        width = float(self.spec.get("width_pt", 1.0)) * PT
        pen = QPen(col, width)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(a, b)
        if self.arrow:
            d = b - a
            ln = math.hypot(d.x(), d.y())
            if ln > 1e-6:
                ux, uy = d.x() / ln, d.y() / ln
                head = max(mm(1.2), width * 4)
                p.setBrush(col)
                p.setPen(Qt.NoPen)
                p.drawPolygon(QPolygonF([b, QPointF(b.x() - head * (ux + 0.5 * uy),
                                                    b.y() - head * (uy - 0.5 * ux)),
                                         QPointF(b.x() - head * (ux - 0.5 * uy),
                                                 b.y() - head * (uy + 0.5 * ux))]))
        self.paint_measure(p, a, b)

    def paint_measure(self, p: QPainter, a: QPointF, b: QPointF) -> None:
        """A subclass hook (the measurement line labels its length)."""

    def paint_selection(self, p: QPainter) -> None:
        pen = QPen(QColor(T.ACCENT), 0)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(QBrush(QColor(T.ACCENT)))
        for pt in (QPointF(0, 0), QPointF(self._w, self._h)):
            p.drawRect(QRectF(pt.x() - HANDLE / 2, pt.y() - HANDLE / 2, HANDLE, HANDLE))


class ArrowItem(LineItem):
    arrow = True


class MeasureLineItem(LineItem):
    """A line that says how long it is in µm at the scale of the image it is ``of``."""

    def __init__(self, spec, style) -> None:
        self.units_per_um: Optional[float] = None
        super().__init__(spec, style)

    def set_scale(self, units_per_um: Optional[float]) -> None:
        self.units_per_um = units_per_um
        self.update()

    def paint_measure(self, p: QPainter, a: QPointF, b: QPointF) -> None:
        ln = math.hypot(b.x() - a.x(), b.y() - a.y())
        text = f"{ln / self.units_per_um:.3g} µm" if self.units_per_um else f"{to_mm(ln):.1f} mm"
        f = page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.85)
        p.setFont(f)
        fm = QFontMetricsF(f)
        p.setPen(QPen(QColor(str(self.spec.get("color", "#ffffff"))), 0))
        mid = (a + b) / 2
        p.drawText(QPointF(mid.x() - fm.horizontalAdvance(text) / 2, mid.y() - mm(0.6)), text)


class LeaderItem(QGraphicsItem):
    """The two leader lines of a zoom link, from the box on the parent to the popout.
    Not selectable; follows both items; drawn under them."""

    def __init__(self, color: str = "#ffffff", width_pt: float = 0.5) -> None:
        super().__init__()
        self.setZValue(-1.0)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.a = QRectF()
        self.b = QRectF()
        self.color = QColor(color)
        self.width = float(width_pt) * PT

    def set_rects(self, box: QRectF, child: QRectF) -> None:
        self.prepareGeometryChange()
        self.a, self.b = QRectF(box), QRectF(child)
        self.update()

    def boundingRect(self) -> QRectF:                                     # noqa: N802
        return self.a.united(self.b).adjusted(-2, -2, 2, 2)

    def _pairs(self) -> List[Tuple[QPointF, QPointF]]:
        a, b = self.a, self.b
        if a.isNull() or b.isNull():
            return []
        # join the two corners of the box that face the child
        if b.left() >= a.right():
            return [(a.topRight(), b.topLeft()), (a.bottomRight(), b.bottomLeft())]
        if b.right() <= a.left():
            return [(a.topLeft(), b.topRight()), (a.bottomLeft(), b.bottomRight())]
        if b.top() >= a.bottom():
            return [(a.bottomLeft(), b.topLeft()), (a.bottomRight(), b.topRight())]
        return [(a.topLeft(), b.bottomLeft()), (a.topRight(), b.bottomRight())]

    def paint(self, p: QPainter, option, widget=None) -> None:
        pen = QPen(self.color, self.width)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        for s, e in self._pairs():
            p.drawLine(s, e)


class ChartItem(FigureItem):
    """A chart (V5.00 step 4): the geometry :func:`nodegraph.figure.charts.chart_geometry`
    computed for the item's rect — the panel resolves the series and sets it — painted in
    item units with plain pens: axes, ticks, marks, legend, colour bar, brackets. Text is
    drawn with the page font at its point size, the same metrics the geometry was measured
    with. :meth:`hover` maps an item point back to data and to the nearest marker's row."""

    def __init__(self, spec: Mapping[str, Any], style: Mapping[str, Any]) -> None:
        self.geometry: Optional[Dict[str, Any]] = None
        self.series: Any = None
        self.note = ""
        super().__init__(spec, style)

    def set_geometry(self, geometry: Optional[Dict[str, Any]], series: Any = None,
                     note: str = "") -> None:
        self.geometry = geometry
        self.series = series
        self.note = note
        self.update()

    # ── hover ──────────────────────────────────────────────────────────────────
    def hover(self, item_pos: QPointF) -> Dict[str, Any]:
        """What is under an item-coordinate point: ``inside``, the data ``x`` / ``y``, and
        with a marker within reach its ``hit`` (``series``, ``index``), the series
        ``label`` and the ``row`` ride-along (``"m,t,id,y,x"``) when the series has one."""
        from nodegraph.figure import charts as C
        if self.geometry is None:
            return {}
        p = self.pos()
        x_mm, y_mm = to_mm(p.x() + item_pos.x()), to_mm(p.y() + item_pos.y())
        out: Dict[str, Any] = dict(C.data_at(self.geometry, x_mm, y_mm))
        hit = C.nearest_mark(self.geometry, x_mm, y_mm, within_mm=1.5)
        if hit is not None:
            out["hit"] = hit
            try:
                s = self.series[hit["series"]]
                if isinstance(s, Mapping):
                    out["label"] = str(s.get("label", ""))
                    rows = s.get("rows")
                    if rows and hit["index"] < len(rows):
                        out["row"] = rows[hit["index"]]
            except (TypeError, IndexError, KeyError):
                pass
        return out

    # ── coordinates: page mm → item units ──────────────────────────────────────
    def _pt(self, xy_mm: Sequence[float]) -> QPointF:
        p = self.pos()
        return QPointF(mm(float(xy_mm[0])) - p.x(), mm(float(xy_mm[1])) - p.y())

    def _rect(self, rect_mm: Sequence[float]) -> QRectF:
        p = self.pos()
        x, y, w, h = (float(v) for v in rect_mm)
        return QRectF(mm(x) - p.x(), mm(y) - p.y(), mm(w), mm(h))

    @staticmethod
    def _qcolor(s: Any) -> QColor:
        if s is None:
            return QColor(0, 0, 0, 0)
        t = str(s)
        if len(t) == 9 and t.startswith("#"):          # #rrggbbaa
            return QColor(int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16), int(t[7:9], 16))
        return QColor(t)

    def _pen(self, color: Any, width_pt: Any, dash: str = "solid") -> QPen:
        pen = QPen(self._qcolor(color), max(0.0, float(width_pt or 0.0)) * PT)
        if dash == "dashed":
            pen.setStyle(Qt.DashLine)
        elif dash == "dotted":
            pen.setStyle(Qt.DotLine)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    # ── painting ───────────────────────────────────────────────────────────────
    def paint_content(self, p: QPainter) -> None:
        g = self.geometry
        r = self.rect()
        if g is None:
            p.fillRect(r, QColor("#20242a") if self.style.get("dark") else QColor("#f3f4f6"))
            pen = QPen(QColor(T.MUTED), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            p.setFont(page_font(self.style, 7.0))
            p.setPen(QPen(ink(self.style), 0))
            p.drawText(r.adjusted(mm(1), mm(1), -mm(1), -mm(1)),
                       Qt.AlignCenter | Qt.TextWordWrap, self.note or "no data yet")
            self.paint_letter(p)
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        fg = self._qcolor(g.get("fg", "#111111"))
        for mk in g.get("marks", []):
            self._paint_mark(p, mk, fg)
        for tx in g.get("texts", []):
            self._paint_text(p, tx)
        if g.get("legend"):
            self._paint_legend(p, g["legend"], fg)
        if g.get("colorbar"):
            self._paint_colorbar(p, g["colorbar"], fg)
        self.paint_letter(p)

    def _paint_mark(self, p: QPainter, mk: Mapping[str, Any], fg: QColor) -> None:
        kind = mk.get("kind")
        stroke = mk.get("stroke")
        if kind == "rect":
            p.setPen(self._pen(stroke, mk.get("width_pt", 0.75)) if stroke else Qt.NoPen)
            p.setBrush(self._qcolor(mk["fill"]) if mk.get("fill") else Qt.NoBrush)
            p.drawRect(self._rect(mk["rect_mm"]))
        elif kind == "segments":
            p.setPen(self._pen(stroke or fg.name(), mk.get("width_pt", 0.6)))
            for a, b in mk.get("segs_mm", []):
                p.drawLine(self._pt(a), self._pt(b))
        elif kind == "polyline":
            pts = mk.get("pts_mm", [])
            if len(pts) >= 2:
                p.setPen(self._pen(stroke or fg.name(), mk.get("width_pt", 1.0),
                                   str(mk.get("dash", "solid"))))
                p.setBrush(Qt.NoBrush)
                p.drawPolyline(QPolygonF([self._pt(q) for q in pts]))
        elif kind == "polygon":
            pts = mk.get("pts_mm", [])
            if len(pts) >= 3:
                p.setPen(self._pen(stroke, mk.get("width_pt", 0.5)) if stroke else Qt.NoPen)
                p.setBrush(self._qcolor(mk.get("fill")) if mk.get("fill") else Qt.NoBrush)
                p.drawPolygon(QPolygonF([self._pt(q) for q in pts]))
        elif kind == "markers":
            rad = float(mk.get("size_pt", 3.0)) * PT / 2.0
            fills = mk.get("fills")
            shape = str(mk.get("shape", "circle"))
            pen = self._pen(stroke, 0.5) if stroke else Qt.NoPen
            for k, q in enumerate(mk.get("pts_mm", [])):
                fill = fills[k] if fills and k < len(fills) else mk.get("fill")
                p.setBrush(self._qcolor(fill) if fill else Qt.NoBrush)
                p.setPen(pen)
                c = self._pt(q)
                self._marker(p, shape, c.x(), c.y(), rad, self._qcolor(fill or stroke or fg))
            labels = mk.get("labels")
            if labels:
                p.setFont(page_font(self.style, float(self.style.get("font_pt", 8.0)) * 0.8))
                p.setPen(QPen(fg, 0))
                for q, txt in zip(mk.get("pts_mm", []), labels):
                    if txt:
                        c = self._pt(q)
                        p.drawText(QPointF(c.x() + rad * 1.4, c.y() - rad * 0.8), str(txt))
        elif kind == "rects":
            p.setPen(self._pen(stroke, mk.get("width_pt", 0.5)) if stroke else Qt.NoPen)
            p.setBrush(self._qcolor(mk.get("fill")) if mk.get("fill") else Qt.NoBrush)
            for rc in mk.get("rects_mm", []):
                p.drawRect(self._rect(rc))
        elif kind == "box":
            p.setPen(self._pen(stroke or fg.name(), mk.get("width_pt", 1.0)))
            p.setBrush(self._qcolor(mk.get("fill")) if mk.get("fill") else Qt.NoBrush)
            p.drawRect(self._rect(mk["rect_mm"]))
            a, b = mk["median_mm"]
            p.drawLine(self._pt(a), self._pt(b))
            for seg in list(mk.get("whiskers_mm", [])) + list(mk.get("caps_mm", [])):
                p.drawLine(self._pt(seg[0]), self._pt(seg[1]))
        elif kind == "cells":
            p.setPen(Qt.NoPen)
            for cell in mk.get("cells", []):
                fill = cell.get("fill")
                if fill is None:
                    continue
                p.setBrush(self._qcolor(fill))
                if "rect_mm" in cell:
                    p.drawRect(self._rect(cell["rect_mm"]))
                elif cell.get("pts_mm"):
                    p.drawPolygon(QPolygonF([self._pt(q) for q in cell["pts_mm"]]))
        elif kind == "wedge":
            p.setPen(self._pen(stroke or fg.name(), mk.get("width_pt", 0.4)))
            p.setBrush(self._qcolor(mk.get("fill")) if mk.get("fill") else Qt.NoBrush)
            for wd in mk.get("wedges", []):
                c = self._pt((wd["cx_mm"], wd["cy_mm"]))
                r1 = mm(float(wd["r1_mm"]))
                path = QPainterPath(c)
                path.arcTo(QRectF(c.x() - r1, c.y() - r1, 2 * r1, 2 * r1),
                           float(wd["a0_deg"]), float(wd["a1_deg"]) - float(wd["a0_deg"]))
                path.closeSubpath()
                p.drawPath(path)
        elif kind == "rings":
            p.setPen(self._pen(stroke or "#808080", mk.get("width_pt", 0.4), "dashed"))
            p.setBrush(Qt.NoBrush)
            for rg in mk.get("rings", []):
                c = self._pt((rg["cx_mm"], rg["cy_mm"]))
                p.drawEllipse(c, mm(float(rg["r_mm"])), mm(float(rg["r_mm"])))
        elif kind == "bracket":
            p.setPen(self._pen(stroke or fg.name(), mk.get("width_pt", 0.6)))
            x0, x1 = float(mk["x0_mm"]), float(mk["x1_mm"])
            y, d = float(mk["y_mm"]), float(mk.get("drop_mm", 1.0))
            p.drawLine(self._pt((x0, y + d)), self._pt((x0, y)))
            p.drawLine(self._pt((x0, y)), self._pt((x1, y)))
            p.drawLine(self._pt((x1, y)), self._pt((x1, y + d)))

    @staticmethod
    def _marker(p: QPainter, shape: str, x: float, y: float, r: float, col: QColor) -> None:
        if shape == "square":
            p.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
        elif shape == "triangle":
            p.drawPolygon(QPolygonF([QPointF(x, y - r), QPointF(x + r, y + r),
                                     QPointF(x - r, y + r)]))
        elif shape == "diamond":
            p.drawPolygon(QPolygonF([QPointF(x, y - r), QPointF(x + r, y), QPointF(x, y + r),
                                     QPointF(x - r, y)]))
        elif shape in ("cross", "plus"):
            pen = QPen(col, max(0.3, r * 0.45))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            if shape == "plus":
                p.drawLine(QPointF(x - r, y), QPointF(x + r, y))
                p.drawLine(QPointF(x, y - r), QPointF(x, y + r))
            else:
                p.drawLine(QPointF(x - r, y - r), QPointF(x + r, y + r))
                p.drawLine(QPointF(x - r, y + r), QPointF(x + r, y - r))
        else:
            p.drawEllipse(QPointF(x, y), r, r)

    def _paint_text(self, p: QPainter, tx: Mapping[str, Any]) -> None:
        pt = float(tx.get("pt", 8.0))
        f = page_font(self.style, pt, bold=bool(tx.get("bold")))
        p.setFont(f)
        fm = QFontMetricsF(f)
        text = str(tx.get("text", ""))
        w, asc, desc = fm.horizontalAdvance(text), fm.ascent(), fm.descent()
        hz, _sep, vt = str(tx.get("anchor", "start")).partition("_")
        dx = -w / 2 if hz == "middle" else (-w if hz == "end" else 0.0)
        dy = asc if vt == "top" else ((asc - desc) / 2 if vt == "middle"
                                      else (-desc if vt == "bottom" else 0.0))
        p.setPen(QPen(self._qcolor(tx.get("color", "#111111")), 0))
        p.save()
        p.translate(self._pt((tx["x_mm"], tx["y_mm"])))
        rot = float(tx.get("rotate", 0.0))
        if rot:
            p.rotate(rot)
        p.drawText(QPointF(dx, dy), text)
        p.restore()

    def _paint_legend(self, p: QPainter, leg: Mapping[str, Any], fg: QColor) -> None:
        rr = self._rect(leg["rect_mm"])
        if leg.get("fill"):
            p.setPen(Qt.NoPen)
            p.setBrush(self._qcolor(leg["fill"]))
            p.drawRect(rr)
        f = page_font(self.style, float(leg.get("pt", 8.0)))
        p.setFont(f)
        fm = QFontMetricsF(f)
        lh = mm(float(leg.get("line_h_mm", 3.0)))
        x0 = rr.x() + mm(0.8)
        for k, e in enumerate(leg.get("entries", [])):
            cy = rr.y() + mm(0.6) + k * lh + lh / 2
            col = self._qcolor(e.get("color", fg.name()))
            if e.get("fill"):
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawRect(QRectF(x0, cy - lh * 0.3, mm(3.0), lh * 0.6))
            elif e.get("line"):
                p.setPen(self._pen(col, 1.0))
                p.drawLine(QPointF(x0, cy), QPointF(x0 + mm(3.0), cy))
            if e.get("marker"):
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                self._marker(p, str(e["marker"]), x0 + mm(1.5), cy, mm(0.6), col)
            p.setPen(QPen(fg, 0))
            p.drawText(QPointF(rr.x() + mm(5.0), cy + (fm.ascent() - fm.descent()) / 2),
                       str(e.get("label", "")))

    def _paint_colorbar(self, p: QPainter, cb: Mapping[str, Any], fg: QColor) -> None:
        from nodegraph.catalog._shared.overlay_math import colormap_lut
        bar = self._rect(cb["rect_mm"])
        lut = colormap_lut(str(cb.get("cmap", "viridis")))
        ramp = np.repeat(lut[::-1][:, None, :], 2, axis=1)
        ramp = np.ascontiguousarray(np.concatenate(
            [ramp, np.full(ramp.shape[:2] + (1,), 255, np.uint8)], axis=2))
        img = QImage(ramp.data, ramp.shape[1], ramp.shape[0], 4 * ramp.shape[1],
                     QImage.Format_RGBA8888).copy()
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(bar, img)
        p.restore()
        p.setPen(self._pen(fg, 0.5))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar)
        f = page_font(self.style, float(cb.get("pt", 8.0)) * 0.85)
        p.setFont(f)
        fm = QFontMetricsF(f)
        widest = 0.0
        for t in cb.get("ticks", []):
            y = self._pt((0.0, float(t["pos_mm"]))).y()
            p.drawLine(QPointF(bar.right(), y), QPointF(bar.right() + mm(0.6), y))
            label = str(t["label"])
            p.drawText(QPointF(bar.right() + mm(0.9), y + (fm.ascent() - fm.descent()) / 2),
                       label)
            widest = max(widest, fm.horizontalAdvance(label))
        if cb.get("label"):
            label = str(cb["label"])
            p.save()
            p.translate(bar.right() + mm(0.9) + widest + mm(1.2) + fm.ascent(),
                        bar.center().y())
            p.rotate(90)
            p.drawText(QPointF(-fm.horizontalAdvance(label) / 2, 0.0), label)
            p.restore()


class TableItem(FigureItem):
    """A summary table (V5.00 step 5): the ``(header, rows)`` of
    :func:`nodegraph.figure.tables.summary_table` — the panel sets them — in the page font,
    booktabs rules (top, under the header, bottom), numbers right-aligned."""

    def __init__(self, spec: Mapping[str, Any], style: Mapping[str, Any]) -> None:
        self.header: List[str] = []
        self.rows: List[List[str]] = []
        self.note = ""
        super().__init__(spec, style)

    def set_table(self, header: Sequence[str], rows: Sequence[Sequence[str]],
                  note: str = "") -> None:
        self.header = [str(h) for h in header]
        self.rows = [[str(c) for c in r] for r in rows]
        self.note = note
        self.update()

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        if not self.header:
            pen = QPen(QColor(T.MUTED), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            p.setFont(page_font(self.style, 7.0))
            p.setPen(QPen(ink(self.style), 0))
            p.drawText(r.adjusted(mm(1), mm(1), -mm(1), -mm(1)),
                       Qt.AlignCenter | Qt.TextWordWrap, self.note or "no table yet")
            self.paint_letter(p)
            return
        pt = self.spec.get("pt")
        f = page_font(self.style, float(pt) if pt else None)
        fb = page_font(self.style, float(pt) if pt else None, bold=True)
        fm, fmb = QFontMetricsF(f), QFontMetricsF(fb)
        pad = mm(0.8)
        n = len(self.header)
        widths = [fmb.horizontalAdvance(h) for h in self.header]
        for row in self.rows:
            for k, cell in enumerate(row[:n]):
                widths[k] = max(widths[k], fm.horizontalAdvance(cell))
        widths = [w + 2 * pad for w in widths]
        total = sum(widths)
        if total < r.width():                       # spread the slack over the columns
            extra = (r.width() - total) / max(1, n)
            widths = [w + extra for w in widths]
        row_h = fm.height() + pad
        col = ink(self.style)
        lw = float(self.style.get("line_width", 1.0)) * PT
        p.save()
        p.setClipRect(r)
        y = r.y() + (fmb.height() + pad - row_h) if self.spec.get("letter") else r.y()
        if self.spec.get("letter"):
            y = r.y() + fmb.height() * 1.25 + pad      # the letter sits above the table
        # top rule
        p.setPen(QPen(col, lw))
        p.drawLine(QPointF(r.x(), y), QPointF(r.x() + min(total, r.width()), y))
        y += pad * 0.5
        # header
        p.setFont(fb)
        p.setPen(QPen(col, 0))
        x = r.x()
        for k, h in enumerate(self.header):
            p.drawText(QRectF(x + pad, y, widths[k] - 2 * pad, row_h),
                       (Qt.AlignRight if k > 0 or not self._first_is_label() else Qt.AlignLeft)
                       | Qt.AlignVCenter, h)
            x += widths[k]
        y += row_h
        p.setPen(QPen(col, lw * 0.6))
        p.drawLine(QPointF(r.x(), y), QPointF(r.x() + min(total, r.width()), y))
        y += pad * 0.5
        p.setFont(f)
        p.setPen(QPen(col, 0))
        for row in self.rows:
            if y + row_h > r.bottom() + pad:
                break
            x = r.x()
            for k, cell in enumerate(row[:n]):
                p.drawText(QRectF(x + pad, y, widths[k] - 2 * pad, row_h),
                           (Qt.AlignRight if k > 0 or not self._first_is_label()
                            else Qt.AlignLeft) | Qt.AlignVCenter, cell)
                x += widths[k]
            y += row_h
        y += pad * 0.5
        p.setPen(QPen(col, lw))
        p.drawLine(QPointF(r.x(), y), QPointF(r.x() + min(total, r.width()), y))
        p.restore()
        self.paint_letter(p)

    def _first_is_label(self) -> bool:
        return bool((self.spec.get("table") or {}).get("group_by"))

    def natural_size_mm(self) -> Tuple[float, float]:
        """The width and height the table wants, in mm (for *fit to content*)."""
        if not self.header:
            return (to_mm(self._w), to_mm(self._h))
        pt = self.spec.get("pt")
        f = page_font(self.style, float(pt) if pt else None)
        fb = page_font(self.style, float(pt) if pt else None, bold=True)
        fm, fmb = QFontMetricsF(f), QFontMetricsF(fb)
        pad = mm(0.8)
        widths = [fmb.horizontalAdvance(h) + 2 * pad for h in self.header]
        for row in self.rows:
            for k, cell in enumerate(row[:len(widths)]):
                widths[k] = max(widths[k], fm.horizontalAdvance(cell) + 2 * pad)
        h = (len(self.rows) + 1) * (fm.height() + pad) + 3 * pad
        if self.spec.get("letter"):
            h += fmb.height() * 1.25 + pad
        return (to_mm(sum(widths) + pad), to_mm(h))


class LegendItem(FigureItem):
    """A shared legend (V5.00 step 5) of an image's channels and layers — the panel
    builds the entries (``{"kind": swatch | line | marker | ramp, "color", "label",
    "cmap", "lo", "hi"}``) from the image's spec and its render — drawn as rows of a
    glyph and a label in the page font; a ramp entry draws the colour map with its limits."""

    def __init__(self, spec: Mapping[str, Any], style: Mapping[str, Any]) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.note = ""
        super().__init__(spec, style)

    def set_entries(self, entries: Sequence[Mapping[str, Any]], note: str = "") -> None:
        self.entries = [dict(e) for e in entries]
        self.note = note
        self.update()

    def paint_content(self, p: QPainter) -> None:
        r = self.rect()
        f = page_font(self.style)
        fm = QFontMetricsF(f)
        p.setFont(f)
        col = ink(self.style)
        if not self.entries:
            pen = QPen(QColor(T.MUTED), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            p.setPen(QPen(col, 0))
            p.drawText(r.adjusted(mm(1), mm(1), -mm(1), -mm(1)),
                       Qt.AlignCenter | Qt.TextWordWrap, self.note or "legend: pick an image")
            self.paint_letter(p)
            return
        p.save()
        p.setClipRect(r)
        row_h = fm.height() * 1.3
        y = r.y() + mm(0.5)
        title = str(self.spec.get("title") or "")
        if title:
            p.setFont(page_font(self.style, bold=True))
            p.setPen(QPen(col, 0))
            p.drawText(QPointF(r.x() + mm(0.8), y + fm.ascent()), title)
            y += row_h
            p.setFont(f)
        gx, gw = r.x() + mm(0.8), mm(4.0)
        lw = float(self.style.get("line_width", 1.0)) * PT
        for e in self.entries:
            if y + row_h > r.bottom() + mm(0.5):
                break
            kind = str(e.get("kind", "swatch"))
            c = QColor(str(e.get("color") or "#ffffff"))
            cy = y + row_h / 2
            if kind == "ramp":
                from nodegraph.catalog._shared.overlay_math import colormap_lut
                lut = colormap_lut(str(e.get("cmap") or "viridis"))
                ramp = np.repeat(lut[None, :, :], 2, axis=0)
                ramp = np.ascontiguousarray(np.concatenate(
                    [ramp, np.full(ramp.shape[:2] + (1,), 255, np.uint8)], axis=2))
                img = QImage(ramp.data, ramp.shape[1], ramp.shape[0], 4 * ramp.shape[1],
                             QImage.Format_RGBA8888).copy()
                bar = QRectF(gx, cy - row_h * 0.25, gw * 2.5, row_h * 0.5)
                p.setRenderHint(QPainter.SmoothPixmapTransform, True)
                p.drawImage(bar, img)
                p.setPen(QPen(col, 0.4 * PT))
                p.setBrush(Qt.NoBrush)
                p.drawRect(bar)
                lo, hi = e.get("lo"), e.get("hi")
                lab = str(e.get("label") or "")
                if lo is not None and hi is not None:
                    lab = f"{lab}  {lo:.3g} – {hi:.3g}" if lab else f"{lo:.3g} – {hi:.3g}"
                p.setPen(QPen(col, 0))
                p.drawText(QPointF(bar.right() + mm(1.0), cy + (fm.ascent() - fm.descent()) / 2),
                           lab)
                y += row_h
                continue
            if kind == "line":
                p.setPen(QPen(c, max(lw, 0.6 * PT)))
                p.drawLine(QPointF(gx, cy), QPointF(gx + gw, cy))
            elif kind == "marker":
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawEllipse(QPointF(gx + gw / 2, cy), mm(0.8), mm(0.8))
            elif kind == "outline":
                p.setPen(QPen(c, max(lw * 0.8, 0.5 * PT)))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(gx, cy - row_h * 0.3, gw, row_h * 0.6))
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawRect(QRectF(gx, cy - row_h * 0.3, gw, row_h * 0.6))
            p.setPen(QPen(col, 0))
            p.drawText(QPointF(gx + gw + mm(1.0), cy + (fm.ascent() - fm.descent()) / 2),
                       str(e.get("label") or ""))
            y += row_h
        p.restore()
        self.paint_letter(p)


ITEM_CLASSES = {
    "chart": ChartItem, "table": TableItem, "legend": LegendItem,
    "image": ImageItem, "montage": MontageItem, "colorbar": ColorbarItem,
    "histogram": HistogramItem, "scale_bar": ScaleBarItem, "text": TextItem,
    "letter": LetterItem, "shape": ShapeItem, "line": LineItem, "arrow": ArrowItem,
    "measure_line": MeasureLineItem,
}


def make_item(spec: Mapping[str, Any], style: Mapping[str, Any]) -> FigureItem:
    cls = ITEM_CLASSES.get(str(spec.get("kind")), FigureItem)
    return cls(spec, style)


__all__ = ["FigureItem", "ImageItem", "MontageItem", "ColorbarItem", "HistogramItem",
           "ScaleBarItem", "TextItem", "LetterItem", "ShapeItem", "LineItem", "ArrowItem",
           "MeasureLineItem", "LeaderItem", "ITEM_CLASSES", "make_item", "page_font", "ink"]
