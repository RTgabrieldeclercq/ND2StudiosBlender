"""Labelled frame graphics item (NodeLab v2) — a Blender-style frame that visually
groups a set of nodes on the canvas, and since 2026-10-07 a REGION with PORTS.

A frame is **GUI-only**: it never enters the run graph (its record rides in the
document's ``ui`` extras, like node positions). It paints BEHIND the nodes, auto-sizes
to enclose its member :class:`~nodelab_v2.node_item.NodeItem`\\ s (recomputed on every
node move via :meth:`GraphScene.reroute`), and dragging the frame moves all its members
together. A frame always has ≥1 member (an emptied frame is removed by the document), so
it needs no stored geometry — the member bounds ARE its geometry.

**Ports (2026-10-07).** A frame is also a *region*: every wire crossing its border gets a
connection point on the edge — an input port on the left for each SIGNAL entering (one per
external output socket, however many members it feeds) and an output port on the right for
each member output socket a wire leaves from, read from
:meth:`~nodelab_v2.document.GraphDocument.region_ports`. A port sits at the height of the
socket its wire lands on, labelled outside the frame with the socket it carries. The title
bar counts the ports and the region's TABS (:attr:`GraphScene.region_tab_names`, set by
the window from :meth:`~nodelab_v2.workspace.Workspace.region_tabs`); *Duplicate region
as a linked tab* in the frame's menu makes a page of the region alone with a Page Input
or Output of its own at each port.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from nodegraph.sockets import SocketType
from nodelab_v2 import theme as T


class FrameItem(QGraphicsObject):
    """A rounded, tinted rectangle with a title bar, drawn behind its member nodes, with
    a port on its border for every wire crossing it."""

    PAD = 22.0            # margin between the member bounds and the frame edge
    TITLE_H = 24.0        # title-bar height (above the members)
    Z = -5.0              # behind nodes (0) and wires
    PORT_R = 5.0          # a port dot's radius
    PORT_GAP = 14.0       # the least vertical distance between two ports on one side
    LABEL_W = 170.0       # how far a port's label may run OUTSIDE the frame

    def __init__(self, frame_rec, scene) -> None:
        super().__init__()
        self.rec = frame_rec
        self._scene = scene
        self._rect = QRectF(0, 0, 160, 90)
        self.setZValue(self.Z)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeAllCursor)
        self._drag_last: Optional[QPointF] = None
        #: the region's ports: ``(io, position in item coords, label)``, inputs first
        self._ports: List[Tuple[str, QPointF, str]] = []
        #: the region's tabs, by name (the window names them; none outside a window)
        self._tabs: List[str] = []
        self.reflow()

    @property
    def frame_id(self) -> str:
        return self.rec.id

    # ── geometry (follows the members) ─────────────────────────────────────────
    def _member_items(self) -> List["QGraphicsObject"]:
        out = []
        for nid in self.rec.members:
            it = self._scene.node_items.get(nid)
            if it is not None and it.scene() is not None:
                out.append(it)
        return out

    def reflow(self) -> None:
        """Recompute position + size to enclose the member nodes (+ padding + a title
        bar), then the ports on the new border. Cheap; called from :meth:`GraphScene.sync`
        and on every node move."""
        items = self._member_items()
        self.prepareGeometryChange()
        if not items:
            self.update()                       # document removes emptied frames; be safe
            return
        bounds = None
        for it in items:
            # card_rect, not boundingRect: the latter carries the glow repaint margin, so
            # a frame would breathe in/out as its members get selected.
            r = it.mapToScene(it.card_rect()).boundingRect()
            bounds = r if bounds is None else bounds.united(r)
        bounds = bounds.adjusted(-self.PAD, -self.PAD - self.TITLE_H,
                                 self.PAD, self.PAD)
        self.setPos(bounds.topLeft())
        self._rect = QRectF(0, 0, bounds.width(), bounds.height())
        self._ports = self._compute_ports()
        self._tabs = self._tab_names()
        self.update()

    def _tab_names(self) -> List[str]:
        fn = getattr(self._scene, "region_tab_names", None)
        try:
            return [str(n) for n in (fn(self.rec.id) if fn is not None else [])]
        except Exception:                       # noqa: BLE001 — a scene outside a window
            return []

    def _compute_ports(self) -> List[Tuple[str, QPointF, str]]:
        """One port per signal crossing the border (:meth:`GraphDocument.region_interface`
        order), at the height of the socket its first wire lands on — clamped to the body of
        the frame and pushed apart when two would overlap."""
        doc = getattr(self._scene, "doc", None)
        if doc is None or not hasattr(doc, "region_ports"):
            return []
        ins, outs = doc.region_ports(self.rec.id)
        w, h = self._rect.width(), self._rect.height()
        lo = self.TITLE_H + 12.0
        hi = max(lo, h - 12.0)

        def label(nid: str, sock: str) -> str:
            title = doc.title_of(nid) if hasattr(doc, "title_of") else nid
            return f"{title} · {sock}"

        def sock_y(nid: str, io: str, name: str) -> float:
            it = self._scene.node_items.get(nid)
            s = it.socket(io, name) if it is not None and it.scene() is not None else None
            return self.mapFromScene(s.anchor()).y() if s is not None else lo

        out: List[Tuple[str, QPointF, str]] = []
        for io, edges, x in (("in", ins, 0.0), ("out", outs, w)):
            seen: set = set()
            last = -1e9
            for (s, ss, d, ds) in edges:
                if (s, ss) in seen:
                    continue
                seen.add((s, ss))
                y = sock_y(d, "in", ds) if io == "in" else sock_y(s, "out", ss)
                y = min(max(y, lo), hi)
                if y - last < self.PORT_GAP:
                    y = last + self.PORT_GAP
                last = y
                out.append((io, QPointF(x, y), label(s, ss)))
        return out

    def ports(self) -> List[Tuple[str, str]]:
        """``[("in" | "out", label), …]`` — the region's connection points, inputs first."""
        return [(io, lab) for io, _pt, lab in self._ports]

    def title_text(self) -> str:
        """The title bar's text: the title, then the port count and the tab count when the
        frame has any — ``"Region  ·  2 in · 1 out  ·  2 tabs"``."""
        n_in = sum(1 for io, _p, _l in self._ports if io == "in")
        n_out = len(self._ports) - n_in
        bits = []
        if self._ports:
            bits.append(f"{n_in} in · {n_out} out")
        if self._tabs:
            bits.append(f"{len(self._tabs)} tab{'' if len(self._tabs) == 1 else 's'}")
        return self.rec.title + ("  ·  " + "  ·  ".join(bits) if bits else "")

    def boundingRect(self) -> QRectF:
        # the port dots and their labels sit OUTSIDE the frame; `shape` keeps hits inside
        m = self.LABEL_W + self.PORT_R + 12.0
        return self._rect.adjusted(-m, -2, m, 2)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(self._rect, 9, 9)
        return path

    def _color(self) -> QColor:
        return QColor(*self.rec.color) if self.rec.color else QColor(T.ACCENT)

    # ── painting ────────────────────────────────────────────────────────────────
    def paint(self, p: QPainter, *_a) -> None:
        p.setRenderHint(QPainter.Antialiasing, True)
        col = self._color()
        sel = self.isSelected()
        p.setPen(QPen(col if sel else T.alpha(col, 150), 2.0 if sel else 1.4))
        p.setBrush(T.alpha(col, 30))
        p.drawRoundedRect(self._rect, 9, 9)
        # title bar (a tinted band across the top; rounded top via an overlapping rect)
        band = QRectF(0, 0, self._rect.width(), self.TITLE_H + 9)
        p.setPen(Qt.NoPen)
        p.setBrush(T.alpha(col, 60))
        p.drawRoundedRect(band, 9, 9)
        p.setBrush(T.alpha(col, 60))
        p.drawRect(QRectF(0, self.TITLE_H - 1, self._rect.width(), 10))
        f = QFont(T.SANS, 8)
        f.setBold(True)
        p.setFont(f)
        p.setPen(T.INK)
        p.drawText(QRectF(11, 0, self._rect.width() - 22, self.TITLE_H),
                   Qt.AlignVCenter | Qt.AlignLeft, self.rec.title)
        suffix = self.title_text()[len(self.rec.title):]
        if suffix:
            x = 11 + QFontMetricsF(f).horizontalAdvance(self.rec.title)
            f2 = QFont(T.SANS, 8)
            p.setFont(f2)
            p.setPen(T.MUTED)
            p.drawText(QRectF(x, 0, max(0.0, self._rect.width() - 11 - x), self.TITLE_H),
                       Qt.AlignVCenter | Qt.AlignLeft, suffix)
        # ports: a Dataset-coloured dot on the border, its label outside
        if self._ports:
            port_col = T.SOCKET.get(SocketType.DATASET, col)
            pf = QFont(T.SANS, 7)
            for io, pt, lab in self._ports:
                p.setPen(QPen(T.BG, 1.5))
                p.setBrush(port_col)
                p.drawEllipse(pt, self.PORT_R, self.PORT_R)
                p.setFont(pf)
                p.setPen(T.MUTED)
                if io == "in":
                    p.drawText(QRectF(pt.x() - self.PORT_R - 6 - self.LABEL_W, pt.y() - 8,
                                      self.LABEL_W, 16),
                               Qt.AlignVCenter | Qt.AlignRight, lab)
                else:
                    p.drawText(QRectF(pt.x() + self.PORT_R + 6, pt.y() - 8, self.LABEL_W, 16),
                               Qt.AlignVCenter | Qt.AlignLeft, lab)

    # ── drag = move all members together ────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.setSelected(True)
            self._drag_last = e.scenePos()
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._drag_last is not None:
            delta = e.scenePos() - self._drag_last
            self._drag_last = e.scenePos()
            for it in self._member_items():
                # moving each NodeItem fires its itemChange → doc.set_pos + scene.reroute
                # (which reflows this frame to follow) — no direct frame move needed.
                it.moveBy(delta.x(), delta.y())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._drag_last is not None:
            self._drag_last = None
            e.accept()
            return
        super().mouseReleaseEvent(e)


__all__ = ["FrameItem"]
