"""Viewer panel (G4 + Phase-6 navigation) — renders the pulled Dataset of the viewed
node as a multi-channel colour composite, with M/T/Z **frame strips**, per-axis
play/pause + fps, and per-channel toggle buttons tinted by emission colour.

Each axis row is a :class:`~nodelab_v2.framestrip.FrameStrip`: one box per frame, dragged
like the slider it replaced, and — on M and T — *pickable*. The picked frames are the run
scope the troubleshooting mode (F9) evaluates, which is what lets a temporal node be
checked on a handful of real timepoints instead of one frame or the whole series.

Rendering stays in the runner's worker thread (:func:`nodelab_v2.runner.render_plane`
per channel); this panel only *composites* the ready per-channel float planes into an
RGB ``QImage`` — each active channel scaled by percentile auto-contrast and multiplied
by the colour closest to its emission spectrum (:func:`nodelab_v2.theme.emission_qcolor`),
additively blended. Playback advances one axis frame-by-frame, self-throttled to the
target fps and reporting the achieved fps.

**Channel colour** is a default, not a verdict: right-clicking a channel's toggle opens a
colour menu (additive presets, the full picker, or back to the emission colour). The
choice is remembered per channel NAME, so it follows that channel across re-pulls and
nodes, and it costs nothing to apply — a shader uniform on the GPU path.

**Split-channel view** (the ⊞ Split button, NIS Elements style) lays the same frame out as
panes — the composite plus one per active channel, sharing one zoom/pan so the panes stay
registered. It is a display mode: the GPU path relays out the textures it already holds
(:meth:`nodelab_v2.glview.GLImageView.set_tiles`), the CPU path builds a labelled mosaic
(:func:`mosaic_with_clim`). Overlays map onto the composite pane in both.

**Overlays** (Points / Labels / Tracks / Mesh, and a reserved tab for Voxels) are owned by
:mod:`nodelab_v2.overlays`: this panel extracts the geometry for the viewed ``(m,t,z,c)``
and hands it, plus the backend's ``plane_to_widget`` mapping, to a single
:class:`~nodelab_v2.overlays.OverlayRenderer`. Both image backends paint them the same
way — **in widget space, sized in screen pixels** — so an outline keeps its thickness
while you zoom into a label instead of being magnified with the image. The look is
configured in the *Overlays* popup (:mod:`nodelab_v2.overlay_dialog`).

Per-item colours come from the **identity palette** (:meth:`ViewerPanel._build_palette`):
one colour per *physical object* rather than per id, so a tracked cell holds its colour
while its label id is re-issued every frame, and neighbouring objects are pushed apart on
the colour wheel. It is computed once per dataset, not per frame.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform)
from PySide6.QtWidgets import (
    QColorDialog, QDoubleSpinBox, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu,
    QPushButton, QToolButton, QVBoxLayout, QWidget,
)

from nodegraph.domains import Domain
from nodegraph.mesh import mesh_part as MESH_PART
from nodegraph.provider import subset_index
from nodelab_v2 import overlays as OV
from nodelab_v2 import theme as T
#: How many channels the GL composite can sample at once. Defined in
#: :mod:`nodelab_v2.overlays` rather than imported from :mod:`nodelab_v2.glview` because
#: this module must keep working with **no GL at all** (``NODELAB_GL=0``, a headless probe,
#: or after a driver failure falls the surface back to the CPU renderer) — a module-level
#: import of the GL view would make the shader's sampler-bank size a hard dependency of the
#: path that exists for when there is no shader. ``glview`` reads the same constant, so the
#: number the status line quotes is the number the shader actually enforces.
_GL_MAX_CH = OV.GL_MAX_CHANNELS

#: Longest edge the CPU split-view MOSAIC may reach, before its panes are decimated to fit.
#: Matches :data:`nodelab_v2.runner.MAX_DISPLAY_DIM`'s intent — that is the cap on a single
#: display plane, and the mosaic is several of them side by side, so without a cap of its own
#: the assembled image is `sqrt(n)` times over budget. Not imported from ``runner`` to keep
#: this module's import graph free of the engine; the number is a display constant, and the
#: consequence of the two differing is sharpness, never correctness.
_MAX_MOSAIC_DIM = 4096
from nodelab_v2.framestrip import FrameStrip, compact_list
from nodelab_v2.minimap import ElidedLabel
from nodelab_v2.picker import (
    Calibration, PickRequest, PickSession, histogram_values, instant_values)

#: What an image surface must provide for overlays and interactive picking to work on it.
#:
#: Both backends implement it — :class:`_ImageView` (CPU, QGraphicsView) and
#: :class:`~nodelab_v2.glview.GLImageView` (GPU, the DEFAULT in a windowed session) — and the
#: panel calls every one of these through ``getattr``, because a surface is swapped in at
#: runtime (a GL failure falls back mid-session). That probing is what made an omission
#: invisible: ``widget_to_plane`` was added to the CPU view only, so on the GPU path
#: ``_pick_plane_pt`` returned ``None`` for every click and each on-image gesture silently
#: did nothing — while the offscreen tests, which force ``NODELAB_GL=0``, all passed.
#:
#: :func:`check_surface_contract` is called by the GUI probe against BOTH classes so a
#: half-implemented surface fails loudly at the gate instead of quietly at the mouse.
SURFACE_CONTRACT = ("plane_to_widget", "widget_to_plane", "refresh", "overlay_cb",
                     "set_detail", "clear_detail", "visible_rect01",
                     "view_changed")


def check_surface_contract(*surfaces) -> None:
    """Raise unless every given image surface implements :data:`SURFACE_CONTRACT`."""
    for s in surfaces:
        missing = [n for n in SURFACE_CONTRACT if not hasattr(s, n)]
        if missing:
            name = getattr(s, "__name__", type(s).__name__)
            raise AssertionError(
                f"{name} does not implement the image-surface contract: missing {missing}. "
                f"The panel probes these with getattr, so an absent one disables overlays "
                f"or picking on that backend SILENTLY.")


#: The ROI drawing tools offered on the pick bar, as (tool key, button text, tooltip).
#: Words rather than geometric glyphs (▭ ⬭ ⬠ ✎): those code points are absent from the
#: shipped UI fonts on Windows and rendered as tofu boxes, which made four of the five
#: tools indistinguishable.
_ROI_TOOLS = (
    ("rect", "Rect", "Rectangle — drag two corners"),
    ("ellipse", "Ellipse", "Ellipse — drag its bounding box"),
    ("circle", "Circle", "Circle — press at the centre, drag out"),
    ("polygon", "Polygon", "Polygon — click each corner, double-click to close"),
    ("brush", "Freehand", "Freehand — drag to paint a band"),
)

#: the playable/scrubbable/pickable spatial-series axes, in the order the strips are laid
#: out and the order every ``(ms, ts, zs)`` scope tuple is written in (C = toggles).
#: ``m``/``t`` address a *frame* (the Frame domain's own address) and fall back to the
#: display cursor when unpicked; ``z`` is inside a frame and falls back to the whole volume
#: — see :class:`~nodegraph.provider.FrameSubsetProvider`.
_AXES = ("m", "t", "z")

#: full ↔ mini-map text for the Overlays button: in the mini-map the whole control strip
#: has to fit in ~200 px, so it shrinks to the glyph alone (the tooltip carries the name).
_OVL_LABEL = ("◈ Overlays", "◈")

#: the ready-made channel colours offered on a channel's right-click menu — the pure
#: additive primaries + secondaries every acquisition package uses for pseudo-colour,
#: plus grey for transmitted light. "Custom…" opens the full picker.
_COLOR_PRESETS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("Red", (255, 40, 40)), ("Green", (60, 255, 80)), ("Blue", (70, 120, 255)),
    ("Cyan", (60, 230, 240)), ("Magenta", (255, 70, 220)), ("Yellow", (255, 220, 60)),
    ("Orange", (255, 150, 40)), ("Grey", (210, 214, 222)),
)

#: full ↔ mini-map text for the split-channel view button
_SPLIT_LABEL = ("⊞ Split", "⊞")

#: What the hover line says when the pointer is not on the image. The row costs a line of
#: height whether or not it holds a value, so it spends that line inviting the gesture
#: rather than sitting blank (which reads as a rendering fault).
_HOVER_HINT = "hover the image to read a pixel"

#: How many timepoints the palette's neighbour graph samples per member layer. Which
#: objects are neighbours barely changes frame to frame — the pairs deduplicate almost
#: completely across a series — so an evenly spaced sample of the frames gives the same
#: graph for a bounded cost, whether the movie is 20 frames or 2000.
MAX_NEIGHBOR_FRAMES = 64

#: Above this many distinct objects the neighbour de-confliction is skipped (measured at
#: ~40 µs each, and this runs on the GUI thread). Every object still keeps ONE colour for
#: the whole series — only the nudging-apart is dropped. Announced once on stderr rather
#: than silently capped.
MAX_PALETTE_OBJECTS = 40_000


@dataclass(frozen=True)
class _Palette:
    """The viewed dataset's **identity palette**: per layer, an ``id → palette slot`` LUT.

    Built once per dataset by :meth:`ViewerPanel._build_palette` and read by every overlay
    that colours per item. Dense int arrays rather than dicts because the label fill
    indexes one straight into its per-pixel colour table.
    """
    labels: Dict[Any, np.ndarray] = dc_field(default_factory=dict)   # (LABEL, layer) → LUT
    points: Dict[Any, np.ndarray] = dc_field(default_factory=dict)   # (POINT, layer) → LUT
    tracks: Dict[Any, np.ndarray] = dc_field(default_factory=dict)   # track layer → LUT


def _autocontrast(plane: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """A float plane percentile-normalized to [0, 1] (hot-pixel-safe)."""
    a = np.asarray(plane, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def plane_to_qimage(plane: np.ndarray, *, lo_pct: float = 1.0,
                    hi_pct: float = 99.5) -> QImage:
    """Auto-contrast grayscale-8 QImage from a float plane (kept for callers/tests)."""
    a = np.asarray(plane, dtype=float)
    if a.size == 0:
        return QImage(1, 1, QImage.Format_Grayscale8)
    u8 = (_autocontrast(a, lo_pct, hi_pct) * 255.0).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape
    return QImage(u8.data, w, h, w, QImage.Format_Grayscale8).copy()


def composite_to_qimage(planes: Dict[int, np.ndarray], colors: Dict[int, Tuple[int, int, int]],
                        *, lo_pct: float = 1.0, hi_pct: float = 99.5) -> QImage:
    """Additively blend the active per-channel planes into an RGB QImage — each channel
    auto-contrasted then multiplied by its (r,g,b) emission colour."""
    if not planes:
        return QImage(1, 1, QImage.Format_RGB888)
    shape = next(iter(planes.values())).shape
    rgb = np.zeros((shape[0], shape[1], 3), dtype=float)
    for ch, plane in planes.items():
        if plane.shape != shape:
            continue
        norm = _autocontrast(plane, lo_pct, hi_pct)
        r, g, b = colors.get(ch, (255, 255, 255))
        rgb[..., 0] += norm * (r / 255.0)
        rgb[..., 1] += norm * (g / 255.0)
        rgb[..., 2] += norm * (b / 255.0)
    u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w, _ = u8.shape
    return QImage(u8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def _img_axis(n: int, f0: float, f1: float) -> np.ndarray:
    """``(n,)`` of normalized IMAGE positions for the samples of a plane covering the
    fractional span ``[f0, f1]`` — sample centres, matching the shader's interpolated uv."""
    n = max(1, int(n))
    return float(f0) + (np.arange(n, dtype=float) + 0.5) / n * (float(f1) - float(f0))


def composite_with_clim(planes: Dict[int, np.ndarray],
                        colors: Dict[int, Tuple[int, int, int]],
                        clims: Dict[int, Tuple[float, float]],
                        gammas: Optional[Dict[int, float]] = None,
                        blends: Optional[Dict[int, Tuple[int, float, float]]] = None,
                        region: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
                        ) -> QImage:
    """Like :func:`composite_to_qimage` but uses **precomputed** ``(lo, hi)`` intensity
    bounds per channel (contrast computed once per volume and cached, the biggest
    per-frame CPU saving on the fallback path) plus an optional per-channel ``gamma``
    (transfer ``norm**gamma``) — the CPU mirror of the GPU shader's LUT.

    ``blends`` maps a channel to ``(mode, opacity, param)`` and mirrors ``blend_one`` in
    :func:`nodelab_v2.glview._build_frag` — 0 add · 1 over · 2 difference · 3 checkerboard ·
    4 wipe. The third slot is the mode's own parameter: checker cell count for 3, divider
    position (0..1) for 4, unread otherwise. It must stay in step with that function: the two backends are chosen by
    whether the GL context came up, so a user must never be able to tell which one drew the
    picture. Channels are composited in **index order** here for the same reason the shader
    unrolls in index order — `over` and `difference` are not commutative, and an overlay
    (which lives at an index above the primary's own channels) has to land on top.

    ``region`` is the fractional ``(fy0, fy1, fx0, fx1)`` part of the whole image these planes
    cover — anything but the default when compositing a zoomed **detail patch**. The two
    spatial comparators are defined on the IMAGE, so a patch has to be told where it sits or
    its checkerboard restarts and its wipe divider slides to the middle of the patch (the CPU
    mirror of the shader's ``u_rect``).
    """
    if not planes:
        return QImage(1, 1, QImage.Format_RGB888)
    gammas = gammas or {}
    blends = blends or {}
    shape = next(iter(planes.values())).shape
    rgb = np.zeros((shape[0], shape[1], 3), dtype=float)
    checker_cache: Optional[np.ndarray] = None
    for ch in sorted(planes):
        plane = planes[ch]
        if plane.shape != shape:
            continue
        a = np.asarray(plane, dtype=float)
        lohi = clims.get(ch)
        if lohi is None:
            norm = _autocontrast(a, 1.0, 99.5)
        else:
            lo, hi = lohi
            if hi <= lo:
                hi = lo + 1.0
            norm = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
        gm = float(gammas.get(ch, 1.0))
        if abs(gm - 1.0) > 1e-3:
            norm = np.power(norm, max(gm, 1e-3))
        r, g, b = colors.get(ch, (255, 255, 255))
        src = norm[..., None] * np.array([r / 255.0, g / 255.0, b / 255.0])
        mode, opacity, cells = blends.get(ch, (0, 1.0, 8.0))
        op = float(min(max(opacity, 0.0), 1.0))
        if mode == 1:                                    # over — own brightness is alpha
            alpha = (norm * op)[..., None]
            rgb = rgb * (1.0 - alpha) + src * alpha
        elif mode == 2:                                  # difference
            rgb = np.abs(rgb - src * op)
        elif mode == 3:                                  # checkerboard, in image space
            if checker_cache is None or checker_cache.shape != shape:
                n = max(float(cells), 1.0)
                ys = np.floor(_img_axis(shape[0], region[0], region[1]) * n)
                xs = np.floor(_img_axis(shape[1], region[2], region[3]) * n)
                checker_cache = np.mod(ys[:, None] + xs[None, :], 2.0)
            k = (checker_cache * op)[..., None]
            rgb = rgb * (1.0 - k) + src * k
        elif mode == 4:                                  # wipe, in image space
            k = np.where(_img_axis(shape[1], region[2], region[3])
                         <= min(max(cells, 0.0), 1.0), op, 0.0)
            k = np.broadcast_to(k[None, :], shape)[..., None]
            rgb = rgb * (1.0 - k) + src * k
        else:                                            # add
            rgb = rgb + src * op
    u8 = np.ascontiguousarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    h, w, _ = u8.shape
    return QImage(u8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def mosaic_with_clim(planes: Dict[int, np.ndarray],
                     colors: Dict[int, Tuple[int, int, int]],
                     clims: Dict[int, Tuple[float, float]],
                     gammas: Optional[Dict[int, float]],
                     tiles: List[Tuple[str, Tuple[int, ...], Tuple[int, int, int]]],
                     *, gap: int = 4,
                     blends: Optional[Dict[int, Tuple[int, float, float]]] = None
                     ) -> QImage:
    """The **split-channel view** on the CPU path: one composite per pane, laid out in a
    near-square grid and labelled, as a single RGB QImage.

    Pane 0 is placed at the origin, so the panel's plane→scene mapping (identity on the
    CPU backend) still lands overlays on it — exactly the pane the GPU path uses too.
    Labels are drawn at plane resolution here (they zoom with the mosaic); the GPU path,
    which is the default, draws them in screen pixels."""
    if not planes or not tiles:
        return composite_with_clim(planes, colors, clims, gammas, blends)
    shape = next(iter(planes.values())).shape
    h, w = int(shape[0]), int(shape[1])
    n = len(tiles)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # The MOSAIC is what the budget applies to, not the pane. Each plane already arrived
    # capped at `runner.MAX_DISPLAY_DIM`, but the split view lays out `1 + len(channels)` of
    # them — so 3 channels of a mosaic at the 4096 cap builds an 8192² RGB888 QImage, ~201 MB,
    # allocated and composited from scratch on every LUT change and every repaint. Decimating
    # the panes so the ASSEMBLED mosaic honours the same cap keeps the split view usable on
    # the CPU backend (headless, NODELAB_GL=0, or after a driver fall-back) instead of
    # stalling the GUI thread on a 200 MB allocation per drag of a contrast slider.
    span = max(cols * w + gap * (cols - 1), rows * h + gap * (rows - 1))
    if span > _MAX_MOSAIC_DIM:
        step = int(np.ceil(span / float(_MAX_MOSAIC_DIM)))
        planes = {ch: pl[::step, ::step] for ch, pl in planes.items()}
        shape = next(iter(planes.values())).shape
        h, w = int(shape[0]), int(shape[1])
    out = QImage(cols * w + gap * (cols - 1), rows * h + gap * (rows - 1),
                 QImage.Format_RGB888)
    out.fill(QColor(5, 8, 13))
    p = QPainter(out)
    font = p.font()
    font.setPixelSize(max(11, h // 26))
    font.setBold(True)
    p.setFont(font)
    for i, (label, chans, rgb) in enumerate(tiles):
        r, c = divmod(i, cols)
        x, y = c * (w + gap), r * (h + gap)
        sub = {ch: pl for ch, pl in planes.items() if ch in chans}
        p.drawImage(x, y, composite_with_clim(sub, colors, clims, gammas, blends))
        fm = p.fontMetrics()
        plate = QRectF(x + 5, y + 5,
                       min(fm.horizontalAdvance(label) + 14, max(20, w - 10)),
                       fm.height() + 5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(6, 9, 14, 175))
        p.drawRoundedRect(plate, 4.0, 4.0)
        p.setPen(QColor(*rgb))
        p.drawText(plate.adjusted(7, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, label)
    p.end()
    return out


def _swatch(rgb: Tuple[int, int, int], size: int = 12) -> QIcon:
    """A filled colour chip for a menu entry — the colour has to be *seen* to be chosen."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QPen(QColor(0, 0, 0, 120), 1.0))
    p.setBrush(QColor(*rgb))
    p.drawRoundedRect(QRectF(0.5, 0.5, size - 1.0, size - 1.0), 3.0, 3.0)
    p.end()
    return QIcon(pix)


def _text_on(col) -> str:
    """Black or white, whichever contrasts with ``col`` (relative luminance)."""
    lum = 0.299 * col.red() + 0.587 * col.green() + 0.114 * col.blue()
    return "#0b0e12" if lum > 150 else "#f0f3f7"


class _ImageView(QGraphicsView):
    """The image surface — a single pixmap item on a graphics scene. Scroll-wheel
    zooms toward the cursor; left-drag pans (``ScrollHandDrag``, on by default, exactly
    like the node canvas); scrollbars stay hidden (the drag is the way to move, and they
    would eat scarce room in the mini-map).

    Framing: the view auto-fits (keeping aspect ratio) when a NEW image SIZE arrives and
    whenever the widget is resized — so re-homing the panel into the mini-map re-frames
    the image instead of cropping it. Once the user has zoomed or panned, that framing is
    theirs: scrubbing and resizing both leave it alone until a double-click re-fits.

    Overlays are painted through :attr:`overlay_cb` in :meth:`drawForeground` with the
    world transform reset, i.e. **in viewport pixels** — mirroring
    :class:`~nodelab_v2.glview.GLImageView` exactly, so one renderer serves both backends
    and overlay line widths / glyph sizes no longer scale with the zoom. That also means
    the whole viewport must repaint on a pan (``FullViewportUpdate``): a minimal update
    would only refresh the scrolled band while the overlay covers everything."""

    #: Declared at CLASS level (and re-bound per instance below) so
    #: :func:`check_surface_contract` can verify the contract without constructing a
    #: surface — a GL context is not always available where the gate runs.
    overlay_cb = None
    #: the pan/zoom changed — the panel re-requests a viewport detail patch (debounced).
    view_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        # the viewport detail patch: a second item ABOVE the overview, positioned in
        # base-pixmap (= scene) units by `set_detail`
        self._detail = QGraphicsPixmapItem()
        self._detail.setTransformationMode(Qt.SmoothTransformation)
        self._detail.setZValue(1.0)
        self._detail.setVisible(False)
        self._scene.addItem(self._detail)
        self.setDragMode(QGraphicsView.ScrollHandDrag)          # pan on by default
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMinimumSize(200, 200)
        self._last_size = None
        self._need_fit = False
        self._user_view = False          # the user zoomed/panned → stop auto-framing
        #: the panel sets this to paint overlays over the image: cb(painter)
        self.overlay_cb = None

    # ── overlay surface (mirrors GLImageView's API) ─────────────────────────────
    def plane_to_widget(self, px: float, py: float) -> QPointF:
        """Map a plane-space pixel to viewport coords. The pixmap item sits at the scene
        origin at 1:1, so scene units *are* displayed-plane pixels."""
        return self.viewportTransform().map(QPointF(px, py))

    def widget_to_plane(self, pt: QPointF) -> Optional[Tuple[float, float]]:
        """The inverse of :meth:`plane_to_widget` — a viewport point → displayed-plane
        ``(x, y)``, or ``None`` when there is no image or the point is outside it.
        Interactive picking needs to go this way round (where did the user click?), and
        having it beside its inverse is what keeps the two from drifting apart;
        :class:`~nodelab_v2.glview.GLImageView` carries the same pair so the pick surface
        is backend-agnostic.

        Outside the image is ``None``, not an extrapolated coordinate: the pixmap is
        letterboxed inside the viewport, so a click in the margin has no plane position,
        and accepting one produced ROI shapes with negative vertices."""
        pm = self._item.pixmap()
        if pm.isNull():
            return None
        sp = self.mapToScene(int(pt.x()), int(pt.y()))
        if not (0 <= sp.x() <= pm.width() and 0 <= sp.y() <= pm.height()):
            return None
        return (sp.x(), sp.y())

    def refresh(self) -> None:
        self.viewport().update()

    def drawForeground(self, p: QPainter, rect: QRectF) -> None:
        super().drawForeground(p, rect)
        if self.overlay_cb is None:
            return
        p.save()
        p.setTransform(QTransform())          # → viewport pixels, like the GL path
        try:
            self.overlay_cb(p)
        except Exception:                     # noqa: BLE001 — overlays are non-fatal
            pass
        p.restore()

    def set_pixmap(self, pix: QPixmap) -> None:
        changed = self._last_size != pix.size()
        if changed:
            self.clear_detail()          # the patch belonged to the previous image
        self._item.setPixmap(pix)
        if not pix.isNull():
            self._scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self._last_size = pix.size()
        if changed:
            self._need_fit = True
            self._user_view = False      # a differently sized image re-frames anyway
            self._maybe_fit()

    # ── viewport detail patch (mirrors GLImageView; SURFACE_CONTRACT) ───────────
    def set_detail(self, pix_and_rect) -> None:
        """Show a finer pixmap over a sub-rect of the image.

        Takes ``(QPixmap, rect01)`` rather than raw planes because this backend composites
        channels on the CPU into one pixmap long before the surface sees them — the panel
        does that conversion once and hands the result to whichever surface it has. The GL
        twin takes planes because it composites in the shader.

        A second item over the base one, so the overview stays complete underneath and a
        late or partial patch can only ever sharpen a sub-rect."""
        pix, rect01 = pix_and_rect
        if pix is None or pix.isNull() or self._item.pixmap().isNull():
            self.clear_detail()
            return
        base = self._item.pixmap()
        x0, y0, x1, y1 = rect01
        w, h = base.width(), base.height()
        # scene units ARE base-pixmap pixels, so place + scale the patch into that space
        self._detail.setPixmap(pix)
        self._detail.setPos(x0 * w, y0 * h)
        sx = max(1e-9, (x1 - x0) * w) / max(1, pix.width())
        sy = max(1e-9, (y1 - y0) * h) / max(1, pix.height())
        self._detail.setTransform(QTransform().scale(sx, sy))
        self._detail.setVisible(True)
        self.viewport().update()

    def clear_detail(self) -> None:
        if self._detail.isVisible():
            self._detail.setVisible(False)
            self.viewport().update()

    def visible_rect01(self) -> Tuple[float, float, float, float]:
        """The visible image rect in normalized image coords, clamped."""
        pm = self._item.pixmap()
        if pm.isNull():
            return (0.0, 0.0, 1.0, 1.0)
        r = self.mapToScene(self.viewport().rect()).boundingRect()
        w, h = max(1, pm.width()), max(1, pm.height())
        x0 = max(0.0, min(1.0, r.left() / w))
        y0 = max(0.0, min(1.0, r.top() / h))
        x1 = max(0.0, min(1.0, r.right() / w))
        y1 = max(0.0, min(1.0, r.bottom() / h))
        return (x0, y0, max(x1, x0), max(y1, y0))

    def fit(self) -> None:
        self._need_fit = True
        self._user_view = False
        self.clear_detail()
        self._maybe_fit()
        self.view_changed.emit()

    def _maybe_fit(self) -> None:
        if (self._need_fit and self.width() > 2 and self.height() > 2
                and not self._item.pixmap().isNull()):
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self._need_fit = False

    def wheelEvent(self, e) -> None:
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self._user_view = True
        self.scale(f, f)
        self.view_changed.emit()

    def mouseMoveEvent(self, e) -> None:
        if e.buttons() & Qt.LeftButton:
            self._user_view = True       # a hand-drag pan is the user's framing now
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        super().mouseReleaseEvent(e)
        # on RELEASE, not per move: a drag is dozens of moves and each would supersede
        # the previous request, so the patch could never land mid-drag
        self.view_changed.emit()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if not self._user_view:
            self._need_fit = True        # docking ↔ mini-map re-frames, never crops
        self._maybe_fit()
        self.view_changed.emit()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._maybe_fit()

    def mouseDoubleClickEvent(self, e) -> None:
        self.fit()                         # double-click resets the framing
        e.accept()


class HistogramLUT(QWidget):
    """A compact intensity-histogram with two draggable handles — the black point (lo)
    and white point (hi) of the LUT window. Dragging a handle (or the shaded band between
    them) emits :data:`window_changed`; the transfer-function ramp is drawn across the
    window so the mapping is legible. The histogram is log-scaled (microscopy is heavily
    skewed toward the low end).

    Three ranges, kept distinct: the **data range** (``_rmin/_rmax``, the full bit-depth
    extent — handles clamp here), the **view range** (``_vmin/_vmax``, what's drawn — the
    mouse wheel zooms it, and :meth:`fit_view_to_window` snaps it to the handles), and the
    **window** (``_lo/_hi``, the LUT). Zooming the view lets you place the handles precisely
    inside a dim part of the histogram without changing the LUT."""

    window_changed = Signal(float, float)      # (lo, hi) in data units
    gamma_changed = Signal(float)              # transfer-function gamma

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(56)
        self.setMinimumWidth(160)
        self.setCursor(Qt.SizeHorCursor)
        self.setToolTip("Drag the handles to set the LUT · drag the middle dot up/down "
                        "for gamma · wheel to zoom · double-click to reset the zoom")
        self._rmin = 0.0
        self._rmax = 1.0
        self._vmin = 0.0                            # view (zoom) range
        self._vmax = 1.0
        self._lo = 0.0
        self._hi = 1.0
        self._gamma = 1.0
        self._vals: Optional[np.ndarray] = None     # cached samples (re-binned on zoom)
        self._hist: Optional[np.ndarray] = None     # log-normalized bar heights [0,1]
        self._tint = (120, 170, 255)
        self._drag: Optional[str] = None            # 'lo' | 'hi' | 'band'
        self._drag_x0 = 0.0
        self._drag_lo0 = 0.0
        self._drag_hi0 = 0.0

    def set_range(self, rmin: float, rmax: float) -> None:
        """Set the full data range. A genuinely new range resets the zoom to full; the
        same range (a new frame of the same channel) keeps the user's current zoom."""
        rmin = float(rmin)
        rmax = float(rmax) if rmax > rmin else rmin + 1.0
        if (rmin, rmax) != (self._rmin, self._rmax):
            self._rmin, self._rmax = rmin, rmax
            self._vmin, self._vmax = rmin, rmax
        self.update()

    def set_window(self, lo: float, hi: float) -> None:
        self._lo, self._hi = float(lo), float(hi)
        self.update()

    def set_gamma(self, gamma: float) -> None:
        self._gamma = max(1e-2, float(gamma))
        self.update()

    def set_tint(self, rgb: Tuple[int, int, int]) -> None:
        self._tint = rgb
        self.update()

    def set_values(self, values: np.ndarray) -> None:
        """Cache a channel's samples (caller subsamples) and (re)bin over the view range."""
        a = np.asarray(values, dtype=float).ravel()
        self._vals = a[np.isfinite(a)]
        self._recompute_hist()

    def _recompute_hist(self, bins: int = 256) -> None:
        if self._vals is None or self._vals.size == 0:
            self._hist = None
        else:
            counts, _ = np.histogram(self._vals, bins=bins, range=(self._vmin, self._vmax))
            h = np.log1p(counts.astype(float))
            m = float(h.max())
            self._hist = (h / m) if m > 0 else None
        self.update()

    def fit_view_to_window(self) -> None:
        """Zoom the histogram to the current handles (with a little padding) so you can
        fine-tune the window inside a narrow region."""
        lo, hi = min(self._lo, self._hi), max(self._lo, self._hi)
        pad = max((hi - lo) * 0.15, (self._rmax - self._rmin) * 1e-3)
        self._vmin = max(self._rmin, lo - pad)
        self._vmax = min(self._rmax, hi + pad)
        if self._vmax - self._vmin < 1e-9:
            self._vmin, self._vmax = self._rmin, self._rmax
        self._recompute_hist()

    def reset_view(self) -> None:
        self._vmin, self._vmax = self._rmin, self._rmax
        self._recompute_hist()

    # ── value ↔ pixel (in the current view range) ────────────────────────────────
    def _v2x(self, v: float) -> float:
        return (v - self._vmin) / (self._vmax - self._vmin) * self.width()

    def _x2v(self, x: float) -> float:
        return self._vmin + max(0.0, min(1.0, x / max(1, self.width()))) * (
            self._vmax - self._vmin)

    def wheelEvent(self, e) -> None:
        """Zoom the view range around the value under the cursor."""
        factor = 0.8 if e.angleDelta().y() > 0 else 1.25
        center = self._x2v(e.position().x())
        span = (self._vmax - self._vmin) * factor
        full = self._rmax - self._rmin
        span = max(min(span, full), full * 1e-3)   # clamp: never below 0.1% of full
        frac = (center - self._vmin) / max(self._vmax - self._vmin, 1e-9)
        vmin = center - frac * span
        vmax = vmin + span
        if vmin < self._rmin:
            vmin, vmax = self._rmin, self._rmin + span
        if vmax > self._rmax:
            vmax, vmin = self._rmax, self._rmax - span
        self._vmin, self._vmax = max(self._rmin, vmin), min(self._rmax, vmax)
        self._recompute_hist()
        e.accept()

    # ── paint ────────────────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, T.BODY)
        r, g, b = self._tint
        # histogram bars
        if self._hist is not None and self._hist.size:
            n = self._hist.size
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(r, g, b, 130))
            bw = w / n
            for i, hv in enumerate(self._hist.tolist()):
                bh = hv * (h - 2)
                p.drawRect(QRectF(i * bw, h - bh, bw + 0.6, bh))
        # window band
        xlo, xhi = self._v2x(self._lo), self._v2x(self._hi)
        p.fillRect(QRectF(xlo, 0, max(1.0, xhi - xlo), h), QColor(255, 255, 255, 22))
        # transfer function t = clamp(...)**gamma across the window (a curve when gamma≠1)
        top, bot = 1.0, h - 1.0
        p.setPen(QPen(QColor(r, g, b), 1.6))
        curve = []
        steps = 40
        for k in range(steps + 1):
            tt = k / steps
            out = tt ** self._gamma
            curve.append(QPointF(xlo + tt * (xhi - xlo), bot - out * (bot - top)))
        p.drawPolyline(QPolygonF(curve))
        # handle lines
        p.setPen(QPen(T.INK, 1.5))
        p.drawLine(QPointF(xlo, 0), QPointF(xlo, h))
        p.drawLine(QPointF(xhi, 0), QPointF(xhi, h))
        p.setBrush(T.INK)
        p.drawEllipse(QPointF(xlo, h - 3), 3.0, 3.0)
        p.drawEllipse(QPointF(xhi, 3), 3.0, 3.0)
        # midpoint (gamma) dot — drag up/down to reshape the curve
        mx, my = self._gamma_dot()
        p.setPen(QPen(T.INK, 1.0))
        p.setBrush(QColor(r, g, b))
        p.drawEllipse(QPointF(mx, my), 4.5, 4.5)
        p.end()

    def _gamma_dot(self) -> Tuple[float, float]:
        """Widget-space (x, y) of the midpoint dot: at the window centre, height = the
        transfer output there (0.5**gamma)."""
        xlo, xhi = self._v2x(self._lo), self._v2x(self._hi)
        bot, top = self.height() - 1.0, 1.0
        return (xlo + 0.5 * (xhi - xlo), bot - (0.5 ** self._gamma) * (bot - top))

    # ── interaction ──────────────────────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:
        x, y = e.position().x(), e.position().y()
        xlo, xhi = self._v2x(self._lo), self._v2x(self._hi)
        mx, my = self._gamma_dot()
        if (x - mx) ** 2 + (y - my) ** 2 <= 8 ** 2:   # near the gamma dot
            self._drag = "gamma"
        elif abs(x - xlo) <= 6:
            self._drag = "lo"
        elif abs(x - xhi) <= 6:
            self._drag = "hi"
        elif xlo < x < xhi:
            self._drag = "band"
        else:
            self._drag = "lo" if abs(x - xlo) < abs(x - xhi) else "hi"
        self._drag_x0 = x
        self._drag_lo0, self._drag_hi0 = self._lo, self._hi
        self._apply_drag(x, y)

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self._apply_drag(e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, e) -> None:
        self.reset_view()                          # double-click resets the zoom to full
        e.accept()

    def _apply_drag(self, x: float, y: float) -> None:
        eps = (self._vmax - self._vmin) * 1e-3     # finer when zoomed in
        if self._drag == "gamma":
            # midpoint output m = 0.5**gamma → gamma = log(m)/log(0.5)
            bot = self.height() - 1.0
            m = min(0.98, max(0.02, (bot - y) / max(bot, 1.0)))
            self._gamma = min(10.0, max(0.1, np.log(m) / np.log(0.5)))
            self.update()
            self.gamma_changed.emit(self._gamma)
            return
        if self._drag == "band":
            dv = self._x2v(x) - self._x2v(self._drag_x0)
            span = self._drag_hi0 - self._drag_lo0
            lo = min(max(self._rmin, self._drag_lo0 + dv), self._rmax - span)
            self._lo, self._hi = lo, lo + span
        elif self._drag == "lo":
            self._lo = min(self._x2v(x), self._hi - eps)
        elif self._drag == "hi":
            self._hi = max(self._x2v(x), self._lo + eps)
        else:
            return
        self.update()
        self.window_changed.emit(self._lo, self._hi)


class PickBar(QWidget):
    """The strip that appears above the image while a parameter pick is armed.

    It carries the whole interaction's vocabulary in one place: which param is being aimed,
    what gesture to make (:data:`nodelab_v2.picker.PICK_HELP`), the live value as it is
    aimed, and Apply / Cancel. The ROI tool buttons appear only for the ``shapes`` kind,
    which is the one pick that draws a *list* rather than a single value and therefore needs
    a tool and an add/cut mode.

    It is deliberately a banner rather than a modal dialog: the point of picking is to look
    at the data while you choose, so nothing may cover the image."""

    applied = Signal()
    cancelled = Signal()
    tool_changed = Signal(str)
    op_changed = Signal(str)
    action = Signal(str)             # "invert" | "clear" | "undo"

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("role", "pickbar")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 5, 8, 5)
        outer.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._title = QLabel("")
        tf = self._title.font(); tf.setBold(True); self._title.setFont(tf)
        top.addWidget(self._title)
        self._readout = QLabel("")
        self._readout.setProperty("role", "readout")
        top.addWidget(self._readout)
        top.addStretch(1)
        self._apply = QPushButton("Apply")
        self._apply.setToolTip("Write the picked value into the node (Enter)")
        self._apply.clicked.connect(self.applied.emit)
        self._cancel = QPushButton("Cancel")
        self._cancel.setToolTip("Leave the parameter as it was (Esc)")
        self._cancel.clicked.connect(self.cancelled.emit)
        top.addWidget(self._apply)
        top.addWidget(self._cancel)
        outer.addLayout(top)

        self._help = QLabel("")
        self._help.setProperty("role", "muted")
        self._help.setWordWrap(True)
        hf = self._help.font(); hf.setPointSize(9); self._help.setFont(hf)
        outer.addWidget(self._help)

        # ── ROI tools (shapes only) ──────────────────────────────────────────
        self._tools_row = QWidget()
        trow = QHBoxLayout(self._tools_row)
        trow.setContentsMargins(0, 0, 0, 0)
        trow.setSpacing(4)
        self._tool_btns: Dict[str, QToolButton] = {}
        for key, text, tip in _ROI_TOOLS:
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setChecked(key == "rect")
            b.clicked.connect(lambda _c, k=key: self._pick_tool(k))
            trow.addWidget(b)
            self._tool_btns[key] = b
        trow.addSpacing(10)
        self._op_btns: Dict[str, QToolButton] = {}
        for key, text, tip in (("add", "Add", "Union the next shape into the region"),
                               ("cut", "Cut", "Subtract the next shape from the region")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setChecked(key == "add")
            b.clicked.connect(lambda _c, k=key: self._pick_op(k))
            trow.addWidget(b)
            self._op_btns[key] = b
        trow.addSpacing(10)
        for key, text, tip in (
                ("invert", "Invert", "Flip the whole region built so far"),
                ("clear", "Clear", "Zero the region — everything after this rebuilds it"),
                ("undo", "Undo", "Drop the last shape")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c, k=key: self.action.emit(k))
            trow.addWidget(b)
        trow.addStretch(1)
        outer.addWidget(self._tools_row)

    def _pick_tool(self, key: str) -> None:
        for k, b in self._tool_btns.items():
            b.setChecked(k == key)
        self.tool_changed.emit(key)

    def _pick_op(self, key: str) -> None:
        for k, b in self._op_btns.items():
            b.setChecked(k == key)
        self.op_changed.emit(key)

    def configure(self, req: PickRequest) -> None:
        self._title.setText(f"Picking  {req.title}")
        self._help.setText(req.help_text)
        self._tools_row.setVisible(req.kind == "shapes")

    def set_readout(self, text: str) -> None:
        self._readout.setText(text)


class ViewerPanel(QWidget):
    """Shows the viewed node's colour composite; emits ``request_changed`` when the
    coords (m,t,z) or the active channel set move, so the window re-pulls.

    **The solo-frame scope** (:meth:`set_solo`) splits the M/T/Z cursor in two: it stays
    the *global* chooser the user reads and drags, while the delivered payload holds only
    the scoped frames and planes and is therefore indexed by their position in the scope
    (:meth:`_payload_coords`). Everything that reads the payload — the overlay geometry,
    the label raster — goes through the payload form; the strips, their labels and the
    request the window sends keep the global one.

    **What the scope runs** is :meth:`frame_selection`: the M/T/Z indices picked on the
    strips, as a cross product. Picking several T boxes is how a temporal node (tracking,
    a time reduction) gets something real to work on while still running a fraction of the
    series; picking Z boxes cuts each of those frames down to the planes that matter. The
    two fallbacks differ on purpose — an unpicked M or T means *the frame the cursor is
    on*, an unpicked Z means *the whole volume*, because a 3D node needs one."""

    request_changed = Signal()
    #: the M/T/Z picks changed — the window re-scopes the runner and re-pulls.
    selection_changed = Signal()
    #: the ITERATION strip moved: which of an Iterate node's results to show. The window
    #: writes it as that node's ``index`` param and re-pulls — the viewer never touches the
    #: document, exactly like :attr:`pick_committed`.
    iteration_changed = Signal(int)
    #: a parameter pick finished: ``(node_id, {socket_name: value})``. The window writes
    #: these through the same path as a typed edit (pinning them), so the viewer never
    #: touches the document itself and a picked param is indistinguishable downstream from
    #: one somebody entered by hand.
    pick_committed = Signal(str, object)
    #: a pick was armed (True) / disarmed (False) — keeps the inspector's Pick buttons in
    #: the right state and lets the status bar say what is going on.
    pick_armed = Signal(bool)
    #: the live surface reported what it can hold, in px of one texture axis. Forwarded to the
    #: runner, which decides full-resolution-whole vs pyramid-plus-patch from it.
    display_limits = Signal(int)
    #: playback started (True) / stopped (False), on the axis named. Playing is a statement
    #: that EVERY frame is wanted in order, which is what licenses a whole-series preload —
    #: the scrub prefetcher deliberately will not do that from a cursor nudge.
    playing = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 4)
        v.setSpacing(4)

        # ── parameter picking (V2.16) ────────────────────────────────────────
        # The bar sits ABOVE the image and stays hidden until a pick is armed, so the
        # gesture's instructions and its live readout never cover the data being aimed at.
        self._pick: Optional[PickSession] = None
        self._pick_bar = PickBar()
        self._pick_bar.hide()
        self._pick_bar.applied.connect(self._apply_pick)
        self._pick_bar.cancelled.connect(self.cancel_pick)
        self._pick_bar.tool_changed.connect(self._set_pick_tool)
        self._pick_bar.op_changed.connect(self._set_pick_op)
        self._pick_bar.action.connect(self._pick_action)
        v.addWidget(self._pick_bar)
        # Esc / Enter drive the armed pick, and a widget only receives keys when it can hold
        # focus. StrongFocus rather than ClickFocus so arming from the INSPECTOR (a click in
        # another dock) can hand focus over without the user first clicking the image.
        self.setFocusPolicy(Qt.StrongFocus)

        # No header row — the image fills the top of the panel; the viewed-node name +
        # readout live in the status strip, and the overlay controls collapsed into ONE
        # button on the controls strip (built below) that opens the Overlays popup.
        #
        # Overlay state: the settings (loaded from the built-in defaults, then the project
        # file, then this machine's), the renderer that paints them, and a revision that
        # invalidates the per-frame geometry cache whenever they change.
        self.overlays, self._ovl_sources = OV.load_defaults()
        self._renderer = OV.OverlayRenderer()
        self._ovl_dialog = None
        self._ovl_rev = 0
        self._geo_key: Optional[tuple] = None
        self._geo_points: List[OV.PointMark] = []
        self._geo_tracks: List[OV.TrackPath] = []
        self._geo_lab: Optional[np.ndarray] = None
        self._geo_lab_keys: Optional[np.ndarray] = None
        self._geo_mesh: List[OV.MeshSection] = []
        # the identity palette (see _build_palette): per DATASET, not per frame
        self._pal: Optional[_Palette] = None
        self._pal_key: Optional[int] = None
        self._pal_ds: Any = None                  # pins id(dataset) against reuse
        self._warned_palette = False

        # ── image (dominant: scroll-zoom + drag-pan; now fills the reclaimed header) ─
        # Prefer the GPU backend (contrast/colour/compositing in a shader, upload-once);
        # fall back to the CPU QGraphicsView path headless or on a GL failure.
        self._compact = False          # mini-map layout (set_compact) — read on rebuild
        self._gl = None
        self._img_layout = v
        self._view = self._make_view()
        v.addWidget(self._view, 1)

        # ── compact controls, BELOW the image ───────────────────────────────────
        controls = QWidget()
        controls.setProperty("role", "controls")
        self._controls = controls
        cv = QVBoxLayout(controls)
        cv.setContentsMargins(0, 2, 0, 0)
        cv.setSpacing(1)

        # One box per frame (nodelab_v2.framestrip) rather than a groove: the row doubles
        # as the run-scope picker, and M/T picks are what the troubleshooting scope runs.
        self._sliders: Dict[str, FrameStrip] = {}
        self._val_lbls: Dict[str, QLabel] = {}
        self._play_btns: Dict[str, QToolButton] = {}
        self._fps_spins: Dict[str, QDoubleSpinBox] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(1, 1)
        for r, ax in enumerate(_AXES):
            name = QLabel(ax.upper())
            name.setProperty("role", "axis")
            sld = FrameStrip(ax)
            sld.setRange(0, 0)
            sld.unpicked_note = (
                "nothing picked — troubleshooting runs the whole z range" if ax == "z"
                else "nothing picked — troubleshooting runs the frame the cursor is on")
            sld.valueChanged.connect(lambda _v, a=ax: self._on_slider(a))
            sld.selectionChanged.connect(lambda a=ax: self._on_frame_pick(a))
            val = QLabel("0/0")
            val.setProperty("role", "muted")
            val.setMinimumWidth(92)          # room for the "·N picked" marker
            play = QToolButton()
            play.setText("▶")
            play.setCheckable(True)
            play.setAutoRaise(True)
            play.setToolTip(f"Play / pause the {ax.upper()} axis")
            play.toggled.connect(lambda on, a=ax: self._on_play(a, on))
            fps = QDoubleSpinBox()
            fps.setRange(0.5, 60.0)
            fps.setValue(8.0)
            fps.setDecimals(0)
            fps.setSingleStep(1.0)
            fps.setSuffix("fps")
            fps.setToolTip("Target playback rate")
            fps.valueChanged.connect(lambda _v, a=ax: self._retarget_fps(a))
            grid.addWidget(name, r, 0)
            grid.addWidget(sld, r, 1)
            grid.addWidget(val, r, 2)
            grid.addWidget(play, r, 3)
            grid.addWidget(fps, r, 4)
            self._sliders[ax] = sld
            self._val_lbls[ax] = val
            self._play_btns[ax] = play
            self._fps_spins[ax] = fps
        cv.addLayout(grid)

        # ── the ITERATION strip (V2.19) ─────────────────────────────────────────
        # Deliberately NOT a fourth member of `_AXES`: iteration is not an acquisition
        # axis. It has no playback, no pick-set, and it does not scope what runs — it
        # chooses which of an Iterate node's results is on screen, by writing that node's
        # `index`. Keeping it out of the grid loop means nothing that walks M/T/Z (the
        # scope, the compaction, the players) has to learn about it.
        self._iter_row = QWidget()
        irow = QHBoxLayout(self._iter_row)
        irow.setContentsMargins(0, 1, 0, 0)
        irow.setSpacing(6)
        iname = QLabel("ITER")
        iname.setProperty("role", "axis")
        self._iter_strip = FrameStrip("iter")
        self._iter_strip.setRange(0, 0)
        self._iter_strip.unpicked_note = "ctrl+click to keep an iteration"
        self._iter_strip.valueChanged.connect(self._on_iteration)
        self._iter_lbl = QLabel("")
        self._iter_lbl.setProperty("role", "muted")
        self._iter_lbl.setMinimumWidth(92)
        irow.addWidget(iname)
        irow.addWidget(self._iter_strip, 1)
        irow.addWidget(self._iter_lbl)
        self._iter_row.setVisible(False)
        self._iter_values: List[str] = []
        cv.addWidget(self._iter_row)

        # ── display tools: per-frame Auto contrast, Fit-zoom, overlay toggles ────
        tools = QHBoxLayout()
        tools.setSpacing(6)
        tools.setContentsMargins(0, 0, 0, 0)
        self._lut_auto = QPushButton("Auto")
        self._lut_auto.setCheckable(True)
        self._lut_auto.setCursor(Qt.PointingHandCursor)
        self._lut_auto.setToolTip("Auto-contrast (percentile). When ON it re-applies to "
                                  "EVERY frame while scrubbing or playing.")
        self._lut_auto.toggled.connect(self._on_auto_toggled)
        self._lut_fit = QPushButton("Fit")
        self._lut_fit.setCursor(Qt.PointingHandCursor)
        self._lut_fit.setToolTip("Zoom every channel histogram to its window")
        self._lut_fit.clicked.connect(self._fit_all)
        self._split_btn = QPushButton(_SPLIT_LABEL[0])
        self._split_btn.setCheckable(True)
        self._split_btn.setCursor(Qt.PointingHandCursor)
        self._split_btn.setToolTip(
            "Split-channel view (NIS Elements style) — one pane per active channel plus "
            "the composite, all sharing the zoom/pan. Needs 2+ active channels.")
        self._split_btn.toggled.connect(self._on_split_toggled)
        tools.addWidget(self._lut_auto)
        tools.addWidget(self._lut_fit)
        tools.addWidget(self._split_btn)
        tools.addStretch(1)
        # ONE overlay control: the popup owns every domain's look (and its on/off), so the
        # strip no longer grows a checkbox per domain as domains are added.
        self._ovl_btn = QPushButton(_OVL_LABEL[0])
        self._ovl_btn.setCursor(Qt.PointingHandCursor)
        self._ovl_btn.setToolTip("Configure the Point / Label / Track / Mesh (and reserved "
                                 "Voxel) overlays — size, opacity, look, colour")
        self._ovl_btn.clicked.connect(self.open_overlay_dialog)
        tools.addWidget(self._ovl_btn)
        self._tools_row = QWidget()
        self._tools_row.setLayout(tools)
        cv.addWidget(self._tools_row)

        # ── per-channel LUT strip: each channel's toggle over its own histogram,
        #    all channels side by side (built in _rebuild_channels). ──────────────
        self._lut_strip = QHBoxLayout()
        self._lut_strip.setSpacing(8)
        self._lut_strip.setContentsMargins(0, 0, 0, 0)
        self._lut_strip_w = QWidget()
        self._lut_strip_w.setLayout(self._lut_strip)
        cv.addWidget(self._lut_strip_w)
        v.addWidget(controls)

        self._chan_btns: Dict[int, QPushButton] = {}
        self._hists: Dict[int, HistogramLUT] = {}
        self._lut_edits: Dict[int, Tuple[QLineEdit, QLineEdit]] = {}
        self._lut_cols: List[QWidget] = []
        self._chan_colors: Dict[int, Tuple[int, int, int]] = {}
        # user-chosen channel colours (right-click a channel button), keyed by channel
        # NAME so the choice follows the channel across nodes/pulls rather than being
        # pinned to a position, and survives every _rebuild_channels.
        self._chan_color_user: Dict[str, Tuple[int, int, int]] = {}
        #: colour keys of composed (overlay / merged) channels that have appeared before, so
        #: each is auto-enabled ONCE rather than on every pull — see `_apply_axes`.
        self._chan_seen: set = set()
        self._active_channels: List[int] = [0]
        self._auto_on = False
        self._split = False

        # ── hover readout: what is under the pointer, right now ─────────────────
        # Its own line ABOVE the status strip rather than sharing it: the status line
        # says what was pulled and how long it took (it survives a mouse leaving the
        # image), while this one is transient and changes on every mouse move — merging
        # them made the pull result flicker away whenever the pointer crossed the picture.
        self._hover_lbl = ElidedLabel("")
        self._hover_lbl.setProperty("role", "muted")
        self._hover_lbl.setToolTip(
            "Under the pointer: pixel position, the same point in microns and on the "
            "stage, and each shown channel's value — the viewed node's number first, the "
            "unprocessed file value in brackets when they differ.")
        v.addWidget(self._hover_lbl)

        # elided: a QLabel's minimum width is its whole string, which would stop the
        # mini-map from ever shrinking below the length of this status line.
        self._status = ElidedLabel("")
        self._status.setProperty("role", "muted")
        v.addWidget(self._status)
        #: is the status line currently reporting a FAILED pull? The label paints itself, so
        #: :meth:`restyle` re-applies its colour from the theme and would quietly drop the
        #: error tint on any theme change; this is what it reads to keep it.
        self._status_error = False

        # state
        self._axes = None
        #: What the channel strip and the cursor ranges were last built for. The axis SIZES
        #: plus the per-channel IDENTITY (:meth:`_channel_key`) — see there for why the sizes
        #: alone were not enough.
        self._axes_key: Optional[tuple] = None
        # solo-frame scope: (m_total, t_total) of the SOURCE while the scope is on, so the
        # M/T strips keep spanning the real series even though a payload holds only the
        # frames that were scoped.
        self._solo: Optional[Tuple[int, int]] = None
        self._planes: Dict[int, np.ndarray] = {}
        self._ref_plane: Optional[np.ndarray] = None
        self._dataset = None
        self._base_pix: Optional[QPixmap] = None
        self._chan_names: List[str] = []
        #: {channel index: label} for the composed planes a `view.overlay` node adds above
        #: the payload's own channel count — empty for every other node.
        self._overlay_chans: Dict[int, str] = {}
        #: {channel index: (blend mode, opacity, checker cells)} for those same channels.
        self._overlay_style: Dict[int, Tuple[int, float, float]] = {}
        self._overlay_note: str = ""
        #: blink-comparator state: the timer that toggles `_overlay_blink`, and the phase.
        #: Flicker is deliberately NOT a shader mode — it is not a compositing rule, it is
        #: a rule about TIME, and the shader has no notion of time.
        self._overlay_blink: bool = True
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._tick_blink)
        # contrast: (node_id, channel) → (lo, hi) in native intensity units, computed
        # once per volume (not per frame) — the biggest per-frame CPU saving.
        self._clim: Dict[Tuple[str, int], Tuple[float, float]] = {}
        self._drange: Dict[Tuple[str, int], Tuple[float, float]] = {}   # LUT slider extent
        self._gammas: Dict[Tuple[str, int], float] = {}   # per-channel transfer gamma
        self._bit_depth: Optional[int] = None     # significant bit depth (metadata)
        self._clim_node: Optional[str] = None
        self._node_id: Optional[str] = None
        # hover readout: the window installs `raw_plane_cb` so the panel can put the
        # UNPROCESSED file value beside the viewed node's — the viewer owns no provider
        # and cannot reach the source itself (same division as `arm_pick`'s calibration).
        # Signature: ``cb(node_id, m, t, z, c) -> Optional[np.ndarray]``, None whenever
        # the raw pixel is not the same pixel (see EngineRunner.raw_source).
        self.raw_plane_cb: Optional[Callable[..., Optional[np.ndarray]]] = None
        # viewport detail-on-demand: the window installs this so a zoomed-in view can be
        # re-read at full detail (the display plane is capped at MAX_DISPLAY_DIM, and on a
        # stitched mosaic that cap is a quarter of the data). Same division as
        # `raw_plane_cb` — the panel owns no provider. Signature:
        # ``cb(node_id, (m,t,z,c), channels, rect01)``; results arrive at
        # :meth:`on_detail_ready`.
        self.detail_cb: Optional[Callable[..., None]] = None
        #: node_id -> the Voxel layers that node ITSELF produces, in declaration
        #: order. What the Labels overlay draws on Auto: you view a node to see what
        #: it made, and ranking every raster on the payload by id drew whichever
        #: carried the biggest numbering (2026-08-04). Injected by the window, which
        #: owns the document; absent, Auto falls back to the region count.
        self.own_layers_cb: Optional[Callable[[str], list]] = None
        self._detail_rect: Optional[Tuple[float, float, float, float]] = None
        #: debounce for pan/zoom → detail request (ms). Long enough that a wheel spin or a
        #: drag settles first, short enough to feel immediate once the hand stops.
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(120)
        self._detail_timer.timeout.connect(self._request_detail)
        self._hover_pt: Optional[Tuple[float, float]] = None
        # playback — a wall-clock QTimer that draws whatever frame is ready and drops
        # frames to hold the target fps (decoupled from decode; napari's frame budget).
        self._playing_axis: Optional[str] = None
        #: playback is held while the series is being decoded (:meth:`set_play_gate`) — the
        #: user asked to play and the answer is "in a moment", not "no".
        self._gated = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick_play)
        self._last_frame_t: Optional[float] = None
        self._fps_ema: Optional[float] = None
        self._install_surface_filter()        # the hover readout watches every move
        self.restyle()

    # ── backend ─────────────────────────────────────────────────────────────────
    def _make_view(self):
        """The image surface. The GPU :class:`~nodelab_v2.glview.GLImageView` — raw 16-bit
        upload once + LUT window / colour / compositing in a fragment shader, so contrast
        (LUT) changes are instantaneous — is the default on a real windowed session. Set
        ``NODELAB_GL=0`` to force the CPU :class:`_ImageView`. Headless platforms
        (probe/CI) and a runtime ``gl_failed`` fall back to the CPU path automatically.

        Both surfaces expose the same overlay contract — ``overlay_cb`` (a QPainter in
        widget pixels) + ``plane_to_widget`` + ``refresh`` — so :meth:`_paint_overlays`
        is backend-agnostic."""
        import os
        view = None
        if os.environ.get("NODELAB_GL", "1") not in ("0", "false", "no"):
            try:
                from nodelab_v2.glview import GLImageView, probe_gl_available
                if probe_gl_available():
                    view = GLImageView()
                    view.gl_failed.connect(self._fallback_to_cpu)
                    view.limits_ready.connect(self.display_limits)
                    self._gl = view
            except Exception:                    # noqa: BLE001 — any import/ctor issue → CPU
                self._gl = None
                view = None
        if view is None:
            view = _ImageView()
            # The CPU surface has no texture, but it does build a QImage and a QPixmap the
            # size of the frame — so it gets the ordinary cap rather than a full-resolution
            # escalation. A 7168² RGB888 QImage is 154 MB per repaint, which is not a
            # trade the fallback path should be making silently.
            QTimer.singleShot(0, self._cpu_display_limits)
        view.overlay_cb = self._paint_overlays
        # both surfaces announce pan/zoom the same way, so detail-on-demand needs no
        # branch on which backend came up
        try:
            view.view_changed.connect(self._on_view_changed)
        except Exception:                        # noqa: BLE001 — a surface without the
            pass                                 # signal simply never asks for detail
        return view

    def _cpu_display_limits(self) -> None:
        """Announce the CPU surface's ceiling (lazy import: viewer ← runner is circular at
        module scope)."""
        from nodelab_v2.runner import MAX_DISPLAY_DIM
        self.display_limits.emit(int(MAX_DISPLAY_DIM))

    def _fallback_to_cpu(self) -> None:
        """Runtime GL failure → replace the GL surface with the CPU view. Deferred to the
        next event-loop turn: ``gl_failed`` fires from inside the GL widget's own
        ``paintGL``, and tearing the widget down mid-paint segfaults."""
        if self._gl is None:
            return
        QTimer.singleShot(0, self._do_fallback_to_cpu)

    def _do_fallback_to_cpu(self) -> None:
        if self._gl is None:
            return
        self._gl = None
        old = self._view
        self._view = _ImageView()
        self._view.overlay_cb = self._paint_overlays
        try:
            self._view.view_changed.connect(self._on_view_changed)
        except Exception:                        # noqa: BLE001
            pass
        self._img_layout.replaceWidget(old, self._view)
        old.setParent(None)
        old.deleteLater()
        self._apply_image_minimum()          # the fresh surface must honour compact mode
        # The hover readout and an armed pick both filter events on the SURFACE, which has
        # just been replaced — without re-installing, a GL failure would silently stop the
        # readout updating and stop a mid-gesture pick responding.
        self._clear_hover()
        self._install_surface_filter()
        if self._pick is not None and self._pick.req.surface == "canvas":
            self._install_pick_filter(True)
        if self._planes:
            self._display(self._node_id or "", self._planes, self._axes)

    # ── compact (mini-map) mode ────────────────────────────────────────────────
    def set_compact(self, on: bool, *, force: bool = False) -> None:
        """Trim the panel to mini-map size (:mod:`nodelab_v2.minimap`) — or restore the
        docked layout.

        Compact keeps everything that says *what you are looking at* (image, M/T/Z
        cursor + play, channel toggles, the Overlays button, status) and drops what needs
        room to be usable: the LUT histograms, their tool buttons and the per-axis fps
        spinners. It is a pure layout change — no image, contrast, channel or playback
        state is touched, so docking back mid-playback just makes the controls reappear.

        ``force`` re-applies the current state, which the window does after every pull:
        :meth:`_rebuild_channels` builds fresh LUT columns, and fresh widgets are
        visible — without this the histograms would creep back into the mini-map."""
        on = bool(on)
        if on == self._compact and not force:
            return
        self._compact = on
        # compact keeps the channel toggles (they head each LUT column) but drops the
        # histograms + the LUT tool buttons + the fps spinners.
        self._lut_auto.setVisible(not on)
        self._lut_fit.setVisible(not on)
        self._split_btn.setText(_SPLIT_LABEL[1] if on else _SPLIT_LABEL[0])
        for hst in self._hists.values():
            hst.setVisible(not on)
        for lo_e, hi_e in self._lut_edits.values():
            lo_e.setVisible(not on)
            hi_e.setVisible(not on)
        for sp in self._fps_spins.values():
            sp.setVisible(not on)
        for strip in self._sliders.values():
            strip.setCompact(on)
        for lbl in self._val_lbls.values():
            lbl.setMinimumWidth(52 if on else 92)
        self._ovl_btn.setText(_OVL_LABEL[1] if on else _OVL_LABEL[0])
        # the mini-map has one status line's worth of room, and it is spoken for
        self._hover_lbl.setVisible(not on)
        lay = self.layout()
        lay.setContentsMargins(*((3, 2, 3, 2) if on else (6, 6, 6, 4)))
        lay.setSpacing(2 if on else 4)
        self._apply_image_minimum()

    def _apply_image_minimum(self) -> None:
        """The image surface's floor. Both backends ship a 200×200 minimum, which alone
        would keep the mini-map from getting small; compact drops it to 120×90."""
        w, h = (120, 90) if self._compact else (200, 200)
        self._view.setMinimumSize(w, h)

    @property
    def compact(self) -> bool:
        return self._compact

    # ── API used by the window ─────────────────────────────────────────────────
    def coords(self) -> Tuple[int, int, int, int]:
        """The **display cursor** ``(m, t, z, c)`` — what the user has selected. This is
        what the window sends to the runner, so under the solo-frame scope it is also the
        frame the pull is pinned to (see :meth:`set_solo`)."""
        c = self._active_channels[0] if self._active_channels else 0
        return (self._sliders["m"].value(), self._sliders["t"].value(),
                self._sliders["z"].value(), c)

    def channels(self) -> Tuple[int, ...]:
        return tuple(sorted(self._active_channels)) or (0,)

    # ── solo-frame scope ───────────────────────────────────────────────────────
    def set_solo(self, totals: Optional[Tuple[int, int, int]]) -> None:
        """Enter the solo-frame scope with the SOURCE's ``(m_total, t_total, z_total)``, or
        leave it with ``None``.

        Inside the scope each delivered payload holds only the scoped frames and planes, so
        the strips can no longer take their range from it — they would collapse to what was
        run and the user would lose the very control that picks what to run next. They span
        ``totals`` instead and become *choosers*: moving the cursor (or changing the picks)
        asks the runner to compute those frames. The totals come from the SOURCE for the
        same reason on all three axes: a chain that collapses time or projects z still has
        to let the user choose which source frame and plane the run is scoped to.

        Leaving the scope deliberately does NOT re-narrow the strips from the (still
        short) payload — that would snap the cursor back to frame 0 and throw away where
        the user was. The next full pull delivers the real axes and :meth:`_apply_axes`
        restores the ranges, which only ever widens them, so the cursor survives the round
        trip."""
        # exactly one total per strip, whatever arity the caller passed: everything
        # downstream indexes it by _AXES position, so a short tuple would be an IndexError
        # in the middle of a repaint rather than a visible mistake.
        self._solo = (tuple(max(1, int(n)) for n in (tuple(totals) + (1, 1, 1))[:3])
                      if totals else None)
        if self._solo is not None and self._axes is not None:
            self._apply_cursor_ranges(self._axes)

    @property
    def solo(self) -> Optional[Tuple[int, int, int]]:
        return self._solo

    def frame_selection(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """The picked ``(ms, ts, zs)`` — what the troubleshooting scope should run.

        The three axes pick independently and the run is their **cross product**: two
        positions × three timepoints × five planes is 6 frames of 5 planes each.

        An **empty** tuple means "nothing picked on that axis", and the two fallbacks
        differ: the runner reads an empty M or T as *the frame the cursor is on* (so
        untouched strips behave exactly like the one-frame scope always did), and an empty
        Z as *the whole volume* — z is inside a frame, and silently cutting a 3D node down
        to one plane because a cursor happened to sit there would be a trap."""
        return tuple(self._sliders[ax].selection() for ax in _AXES)

    def clear_frame_selection(self) -> None:
        """Drop every pick — the scope falls back to the cursor's frame, whole volume."""
        for ax in _AXES:
            self._sliders[ax].clearSelection()

    def scoped_frames(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """:meth:`frame_selection` with the cursor filled in on the frame axes — i.e.
        exactly what a pull under the scope evaluates. ``zs`` stays as picked, empty
        meaning the whole volume. The runner derives the same thing
        (:meth:`~nodelab_v2.runner.EngineRunner._pin_for`); this is the copy the labels and
        the status line read."""
        ms, ts, zs = self.frame_selection()
        m, t, _z, _c = self.coords()
        return (ms or (m,), ts or (t,), zs)

    def _payload_coords(self) -> Tuple[int, int, int, int]:
        """The index INTO the delivered payload. Identical to :meth:`coords` normally;
        under the solo-frame scope the payload holds only the scoped frames and planes, so
        the global M/T/Z the strips name has to be resolved to its position within them (a
        cursor parked on an unpicked index resolves to the nearest picked one, and an
        unpicked axis passes through — :func:`~nodegraph.provider.subset_index`)."""
        m, t, z, c = self.coords()
        if not self._solo:
            return (m, t, z, c)
        ms, ts, zs = self.scoped_frames()
        return (subset_index(ms, m), subset_index(ts, t), subset_index(zs, z), c)

    def _solo_note(self) -> str:
        """The status-line marker naming the scoped frames (empty when not scoped)."""
        if not self._solo:
            return ""
        parts = []
        for ax, picks, total in zip(_AXES, self.scoped_frames(), self._solo):
            if not picks or (ax != "t" and total <= 1 and len(picks) == 1):
                continue          # a degenerate axis says nothing; unpicked z is "all z"
            parts.append(f"{ax}{compact_list(picks)}" if len(picks) == 1
                         else f"{ax}[{compact_list(picks)}]")
        return f" · solo {'·'.join(parts)}" if parts else ""

    def _on_frame_pick(self, ax: str) -> None:
        """A strip's picks changed. The cursor stays where it is — picking chooses what
        *runs*, not what is shown — so this only re-labels and tells the window, which
        re-scopes the runner and re-pulls when the scope is armed."""
        self._sync_axis_label(ax)
        self.selection_changed.emit()

    # ── result / error ─────────────────────────────────────────────────────────
    def show_result(self, node_id: str, planes, axes, seconds: float,
                    dataset=None, overlay=None, overlay_note: str = "",
                    overlay_style=None) -> None:
        """Full-pull delivery: refresh dataset/axes/channels, then display. Contrast is
        (re)computed once for a new node/volume and cached across all subsequent frames.

        ``overlay`` is ``{channel index: label}`` for the composed planes a
        :mod:`view.overlay` node contributes above the payload's own channel count, and
        ``overlay_note`` the placement readout for the status line. Both empty for every
        other node, which is why nothing else has to know overlays exist."""
        ovl = dict(overlay or {})
        if ovl != dict(getattr(self, "_overlay_chans", {}) or {}):
            self._axes_key = None          # the channel strip must rebuild for the new set
        self._overlay_chans = ovl
        self._overlay_style = dict(overlay_style or {})
        self._overlay_note = str(overlay_note or "")
        self._dataset = dataset
        # Contrast is keyed ``(node_id, channel)`` and is now KEPT across a node switch
        # (2026-08-05: "each channel should be its state from before"). Clearing it was
        # redundant belt-and-braces — the key already namespaces per node, so a different node
        # simply misses and auto-contrasts — and it threw away the one thing that cannot be
        # recomputed: a window the user set by hand. Going A → B → A now returns to A's look.
        self._clim_node = node_id
        self._node_id = node_id
        key = ((axes.m, axes.t, axes.z, axes.c) + self._channel_key(axes, dataset)
               if axes is not None else None)
        if axes is not None and key != self._axes_key:
            self._apply_axes(axes, dataset)
            self._axes_key = key
        self._axes = axes if axes is not None else self._axes
        # `bit_depth` is refreshed UNCONDITIONALLY, not only inside `_rebuild_channels`.
        # That rebuild is gated on the key above — axis sizes plus channel names/emissions —
        # and the declared intensity scale is in neither, so two payloads that differ ONLY in
        # scale left `self._bit_depth` at the previous one's value and `_display_range`
        # therefore produced the wrong histogram extent for whichever was viewed second.
        # It is bidirectional (a normalize child then its raw 12-bit parent reintroduces the
        # exact "135-1564 against 0-4095" regression the bit_depth branch was written to
        # fix), and a BAKE triggers it on its own: `write_checkpoint` restamps
        # `bit_depth = 16` whenever float data was rounded into 16-bit counts.
        #
        # A declared-scale change also retires the cached per-channel LUT extent for this
        # node — `_drange` is what the slider clamp and the committed percentile pick read,
        # and it was captured under the old scale. The user's own `_clim` window is left
        # alone: it is the one thing here that cannot be recomputed.
        if dataset is not None:
            bd = (getattr(dataset, "metadata", {}) or {}).get("bit_depth")
            bd = int(bd) if bd else None
            if bd != self._bit_depth:
                self._bit_depth = bd
                for ck in [k for k in self._drange if k[0] == node_id]:
                    self._drange.pop(ck, None)

        if not planes:
            self._planes = {}
            self._ref_plane = self._base_pix = None
            if self._gl is not None:
                self._gl.clear()
            else:
                self._view.set_pixmap(QPixmap())
            self._set_status(f"{node_id} · no image on this output · "
                             f"pulled in {seconds:.2f}s")
            return
        self._display(node_id, planes, self._axes)
        h, w = self._ref_plane.shape[:2]
        npts, nframe = self._point_tally()
        # Report the count whenever the frame carries ANY point, not only when some land on
        # the viewed plane — an empty plane inside a full volume must not look like an empty
        # detection. The parenthetical names the rest so the Z strip is the obvious next move.
        extra = f" · {npts} points" if nframe else ""
        if nframe > npts:
            extra += f" ({nframe - npts} on other Z)"
        shown = list(self.channels())
        chans = "+".join(self._chan_names[c] if c < len(self._chan_names) else f"Ch{c}"
                         for c in shown)
        note = getattr(self, "_overlay_note", "")
        # The GL composite has a fixed sampler bank (`glview._MAX_CH`) and simply TRUNCATES
        # past it — `active = list(chans)[:_MAX_CH]`, with no error check on the path. A
        # 4+5 channel merge is enough to reach it (`channel.merge` puts no ceiling on the
        # channel axis), and the failure is invisible: the extra channel keeps its button,
        # its LUT and its own split-view pane, and is missing only from the composite. So
        # say it here rather than leaving the user to notice a colour that never appears.
        over = len(shown) - _GL_MAX_CH
        self._set_status(
            f"{node_id} · {w}×{h} px · {chans}{extra}{self._solo_note()} · "
            f"pulled in {seconds:.2f}s"
            + (f"  ·  COMPOSITE SHOWS THE FIRST {_GL_MAX_CH} CHANNELS ONLY "
               f"({over} more switched on; split view shows each one)" if over > 0 else "")
            # the placement readout rides HERE, beside the picture it describes: a wrong
            # tile, time or focus offset should be visible without opening the inspector
            + (f"  ·  overlay {note}" if note else ""))
        self._measure_fps()

    def show_planes(self, node_id: str, planes, axes, seconds: float) -> None:
        """Fast-path delivery (scrub/play): the dataset and axes are unchanged, so only
        the displayed frame moves — no axes/channel rebuild, no spreadsheet refresh."""
        if axes is not None:
            self._axes = axes
        if not planes:
            return
        self._node_id = node_id
        self._display(node_id, planes, self._axes)
        self._measure_fps()

    def _clim_for(self, node_id: str, ch: int, plane: np.ndarray) -> Tuple[float, float]:
        """Percentile contrast bounds for a channel, computed once (native units) and
        cached — this is what keeps ``np.percentile`` off the per-frame hot path."""
        ckey = (node_id, ch)
        lohi = self._clim.get(ckey)
        if lohi is None:
            a = np.asarray(plane, dtype=float)
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                lohi = (0.0, 1.0)
                self._drange[ckey] = (0.0, 1.0)
            else:
                lo = float(np.percentile(finite, 1.0))
                hi = float(np.percentile(finite, 99.5))
                if hi <= lo:
                    hi = lo + 1.0
                lohi = (lo, hi)
                self._drange[ckey] = self._display_range(plane)
            self._clim[ckey] = lohi
        return lohi

    def _display_range(self, plane: np.ndarray) -> Tuple[float, float]:
        """The LUT histogram extent. Prefer the **significant bit depth** carried on the
        payload (``bit_depth`` → ``0 .. 2**bits-1``) — the pixel values alone can't reveal
        it (a dim 12-bit frame maxes out below 1024, which is why inferring from the data
        under-capped the slider at 1023). Without it, an integer image falls back to its
        dtype's full range (never under-caps) and a genuinely rescaled image to its
        observed range.

        **``bit_depth`` is honoured whatever the dtype**, and that is the whole point:
        every streaming provider computes in ``float64`` (:data:`nodegraph.streaming._F`),
        so a Stitch or a Gaussian arrives as float even though its values are still the
        same integer counts. Gating on ``dtype`` therefore ignored the payload's own
        declaration and gave a computed node a data-derived slider while the raw source
        beside it got the sensor range — 135‑1564 against 0‑4095 on a real dim 12‑bit
        mosaic, i.e. the same picture with visibly different contrast depending on which
        node you clicked.

        Reading the declaration rather than the dtype is safe precisely because this
        project already maintains it: a node that leaves the count scale DROPS
        ``bit_depth`` (``metadata.value_rescaled`` — percentile Normalize, CLAHE, the
        ``ratio`` flatten), so an image that really is ``[0,1]`` floats still lands on the
        observed-range branch below."""
        a = np.asarray(plane)
        if self._bit_depth:
            return 0.0, float((1 << int(self._bit_depth)) - 1)
        if np.issubdtype(a.dtype, np.integer):
            return 0.0, float(np.iinfo(a.dtype).max)
        f = np.asarray(a, dtype=float)
        finite = f[np.isfinite(f)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        return lo, (hi if hi > lo else lo + 1.0)

    def _display(self, node_id: str, planes, axes) -> int:
        """Render the given per-channel planes on the active backend. Returns the number
        of overlay points (for the status line). When Auto is ON, the percentile contrast
        is recomputed for THIS frame (so it tracks brightness while scrubbing/playing)."""
        self._planes = dict(planes)
        self._ref_plane = next(iter(self._planes.values()))
        self._geo_key = None                  # a new frame → re-extract overlay geometry
        # A detail patch belongs to ONE frame. Drop it before the new frame is drawn —
        # leaving it would paint the previous timepoint's pixels over this one inside the
        # patch rect — then re-arm, so a zoomed-in scrub refreshes the detail instead of
        # silently falling back to the overview.
        if self._detail_rect is not None:
            try:
                self._surface().clear_detail()
            except Exception:                 # noqa: BLE001
                pass
            self._detail_rect = None
            if self._detail_timer is not None:
                self._detail_timer.start()
        if self._auto_on:
            clims = {}
            for ch, pl in self._planes.items():
                lohi = self._auto_clim(pl)
                self._clim[(node_id, ch)] = lohi           # reflect on the histogram
                self._drange.setdefault((node_id, ch), self._display_range(pl))
                clims[ch] = lohi
        else:
            clims = {ch: self._clim_for(node_id, ch, pl) for ch, pl in self._planes.items()}
        gammas = {ch: self._gammas.get((node_id, ch), 1.0) for ch in self._planes}
        self._sync_luts()
        if self._gl is not None:
            blends = self._blend_map()
            for ch, pl in self._planes.items():
                lo, hi = clims[ch]
                mode, opacity, cells = blends.get(ch, (0, 1.0, 8.0))
                self._gl.set_channel(ch, lo, hi,
                                     self._chan_colors.get(ch, (255, 255, 255)), gammas[ch],
                                     blend=int(mode), opacity=float(opacity),
                                     checker=float(cells))
            self._gl.set_tiles(self._tiles())
            self._gl.set_planes(self._planes)     # uploads + repaints (overlays via cb)
            self._refresh_hover()
            return self._point_count()
        self._base_pix = QPixmap.fromImage(self._compose_cpu(clims, gammas))
        n = self._repaint()
        self._refresh_hover()
        return n

    # ── viewport detail-on-demand ────────────────────────────────────────────────
    #: How much sharper a patch must be than the overview before it is worth reading.
    #: Compared on SOURCE PIXELS, not on magnification: a small image fitted to a large
    #: pane is magnified enormously and yet there is no detail to fetch, because the
    #: overview already carries every pixel the node has.
    DETAIL_TRIGGER = 1.25


    def _surface(self):
        """The live image surface — GL when it came up, the CPU view otherwise. Both
        implement :data:`SURFACE_CONTRACT`, so callers never branch on which."""
        return self._gl if self._gl is not None else self._view

    def _on_view_changed(self) -> None:
        """Pan/zoom moved: (re)arm the debounce. Reading on every wheel click would put
        the pool behind the cursor, and each read would be superseded before it landed."""
        if self._detail_timer is not None:
            self._detail_timer.start()

    def _request_detail(self) -> None:
        """Ask the runner for the visible rect at full detail, if that would show more
        than the overview already does.

        The decision is a pixel count. Across the visible rect the overview carries
        ``tex_w · frac`` source pixels; a patch would carry ``min(budget, frac · full_w)``.
        Read one only when that is :data:`DETAIL_TRIGGER`× better. Two consequences worth
        stating, because both were bugs in earlier drafts: an image small enough to be
        shown 1:1 never requests anything however far it is magnified (the overview IS the
        data), and a fully zoomed-out view never does either (``frac == 1`` makes the two
        sides equal)."""
        surf = self._surface()
        if (self.detail_cb is None or self._node_id is None or self._axes is None
                or self._ref_plane is None):
            return
        try:
            rect = surf.visible_rect01()
        except Exception:                      # noqa: BLE001 — never break a repaint
            return
        frac = max(rect[2] - rect[0], rect[3] - rect[1])
        tex_w = float(max(self._ref_plane.shape[:2]))
        full_w = float(max(1, max(self._axes.y, self._axes.x)))
        from nodelab_v2.runner import MAX_DISPLAY_DIM   # lazy: viewer ← runner would
        budget = float(MAX_DISPLAY_DIM)                 # be a circular import at module scope
        gain = (min(budget, frac * full_w) / max(1e-9, tex_w * frac)) if frac > 0 else 0.0
        if frac <= 0 or gain < self.DETAIL_TRIGGER:
            surf.clear_detail()
            self._detail_rect = None
            return
        m, t, z, _c = self.coords()
        try:
            self.detail_cb(self._node_id, (m, t, z, self._lut_channel()),
                           sorted(self._planes), rect)
        except Exception:                      # noqa: BLE001 — detail is best-effort
            pass

    def on_detail_ready(self, node_id: str, planes, rect01) -> None:
        """A patch arrived (GUI thread). Dropped unless it still describes what is shown —
        the runner drops stale generations too, but the node can change between the two."""
        if node_id != self._node_id or not planes or self._axes is None:
            return
        surf = self._surface()
        self._detail_rect = tuple(rect01)
        if self._gl is not None:
            surf.set_detail(planes, rect01)
            return
        # CPU path: composite the patch with the SAME clim/gamma/blend the overview uses, or
        # it would sit on the image as a differently-contrasted rectangle — and, once the
        # patch carries the overlay's channels too, as a differently-BLENDED one.
        clims = {ch: self._clim.get((node_id, ch)) or self._clim_for(node_id, ch, pl)
                 for ch, pl in planes.items()}
        gammas = {ch: self._gammas.get((node_id, ch), 1.0) for ch in planes}
        x0, y0, x1, y1 = (float(v) for v in rect01)
        img = composite_with_clim(planes, self._chan_colors, clims, gammas,
                                  self._blend_map(planes), region=(y0, y1, x0, x1))
        surf.set_detail((QPixmap.fromImage(img), rect01))

    def _refresh_hover(self) -> None:
        """Re-read the pixel under a STATIONARY pointer after a new frame lands. Scrubbing
        or playing with the mouse resting on the image would otherwise leave the last
        frame's intensity on screen under the new picture."""
        if self._hover_pt is not None and self._ref_plane is not None:
            self._hover_lbl.setText(self._hover_text(self._hover_pt))
        elif not self._hover_lbl.text():
            self._clear_hover()          # first image in → offer the hint

    # ── split-channel view (NIS Elements style) ──────────────────────────────────
    def _tiles(self) -> List[Tuple[str, Tuple[int, ...], Tuple[int, int, int]]]:
        """The split panes for the shown channels — the composite first (so overlays,
        which map through pane 0 on both backends, land on it), then one pane per channel
        in its own colour. Empty (== the single composite pane) when the split is off or
        there is only one channel to split, where a split would show the same image twice.
        """
        chans = sorted(self._planes)
        if not self._split or len(chans) < 2:
            return []
        tiles = [("Composite", tuple(chans), (235, 240, 248))]
        for ch in chans:
            name = self._chan_names[ch] if ch < len(self._chan_names) else f"Ch{ch}"
            tiles.append((name, (ch,), self._chan_colors.get(ch, (235, 240, 248))))
        return tiles

    def _compose_cpu(self, clims, gammas) -> QImage:
        """The CPU-path image for the current mode — the composite, or the labelled
        split-pane mosaic."""
        tiles = self._tiles()
        blends = self._blend_map()
        if tiles:
            return mosaic_with_clim(self._planes, self._chan_colors, clims, gammas, tiles,
                                    blends=blends)
        return composite_with_clim(self._planes, self._chan_colors, clims, gammas, blends)

    def _tick_blink(self) -> None:
        """Flip the blink phase and repaint — uniforms only, so no decode and no re-pull."""
        self._overlay_blink = not self._overlay_blink
        if self._planes:
            self._display(self._node_id, self._planes, self._axes)

    def set_overlay_flicker(self, hz: float) -> None:
        """Start/stop the blink comparator. ``0`` stops it and leaves the overlay ON, so
        turning flicker off can never strand the picture in its hidden phase."""
        hz = float(hz or 0.0)
        if hz <= 0:
            self._blink_timer.stop()
            self._overlay_blink = True
            return
        self._blink_timer.start(max(16, int(500.0 / hz)))   # half-period per toggle

    def _blend_map(self, planes=None) -> Dict[int, Tuple[int, float, float]]:
        """``{channel: (mode, opacity, checker cells)}`` for the shown channels.

        Only an OVERLAY channel carries a non-default entry: the primary's own channels
        composite additively at full strength, which is both the microscopy convention and
        exactly what they did before overlays existed. Read live from the recipe the node
        stamped, so changing the blend or the opacity is a repaint and never a re-pull.

        ``planes`` names the channel set to answer for — the displayed frame by default, the
        detail patch's own (smaller) set when one is being composited."""
        have = self._planes if planes is None else planes
        out = {ch: style for ch, style in (self._overlay_style or {}).items()
               if ch in have}
        if not self._overlay_blink:
            # the hidden half of the blink: opacity 0, mode untouched, so flipping back
            # restores exactly the look the recipe asked for
            out = {ch: (mode, 0.0, param) for ch, (mode, _op, param) in out.items()}
        return out

    def _on_split_toggled(self, on: bool) -> None:
        """Split is a pure DISPLAY mode — no re-pull, no re-decode: the GPU path relays out
        its panes from the textures it already holds, the CPU path re-composites the planes
        it already has."""
        self._split = bool(on)
        if self._planes and self._node_id is not None:
            self._display(self._node_id, self._planes, self._axes)
        if self._gl is not None:
            self._gl.fit()          # the panes are a new framing — re-fit to them
        elif self._base_pix is not None:
            self._view.fit()

    # ── LUT / contrast (instantaneous on GPU: a shader-uniform change) ───────────
    def _lut_channel(self) -> int:
        return self._active_channels[0] if self._active_channels else 0

    def _auto_clim(self, plane: np.ndarray) -> Tuple[float, float]:
        """Percentile (1 / 99.5) contrast from a cheap subsample — used per frame when
        Auto is toggled on."""
        a = np.asarray(plane)
        step = max(1, int(np.sqrt(a.size / 200_000)))
        s = a[::step, ::step] if a.ndim == 2 else a
        s = np.asarray(s, dtype=float)
        finite = s[np.isfinite(s)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.5))
        return (lo, hi if hi > lo else lo + 1.0)

    def _hist_tip(self, ch: int) -> str:
        ckey = (self._node_id, ch)
        lo, hi = self._clim.get(ckey, (0.0, 1.0))
        name = self._chan_names[ch] if ch < len(self._chan_names) else f"Ch{ch}"
        return (f"{name}: {lo:.0f}–{hi:.0f}  γ={self._gammas.get(ckey, 1.0):.2f}\n"
                "drag handles=window · middle dot=gamma · wheel=zoom · dbl-click=reset")

    @staticmethod
    def _fmt(v: float) -> str:
        """Compact numeric text for a LUT bound (int-like for sensor counts, ``%.4g``
        otherwise so small float ranges keep their precision)."""
        return f"{v:.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.4g}"

    def _update_lut_edits(self, ch: int, force: bool = False) -> None:
        """Refresh channel ``ch``'s black/white text fields from its window. ``force``
        overwrites even a focused field — used when the value changed from a DRAG (the
        histogram doesn't steal keyboard focus, so without this a focused field would show
        a stale number during/after a drag). Periodic syncs pass ``force=False`` so they
        never stomp a value being typed."""
        edits = self._lut_edits.get(ch)
        if edits is None:
            return
        lo, hi = self._clim.get((self._node_id, ch), (0.0, 1.0))
        for e, val in zip(edits, (lo, hi)):
            if force or not e.hasFocus():
                e.blockSignals(True)
                e.setText(self._fmt(val))
                e.blockSignals(False)

    def _on_lut_edit(self, ch: int) -> None:
        """Enter/blur on a black/white field → set that exact window, drop Auto, and fit
        the histogram view to the new window."""
        if self._node_id is None:
            return
        lo_e, hi_e = self._lut_edits[ch]
        try:
            lo, hi = float(lo_e.text()), float(hi_e.text())
        except ValueError:
            self._update_lut_edits(ch, force=True)   # bad input → revert to current
            return
        if hi <= lo:
            hi = lo + 1.0
        # No change (e.g. blur right after a drag already set this window) → nothing to do,
        # and in particular don't re-fit the view.
        cur = self._clim.get((self._node_id, ch))
        if cur is not None and abs(cur[0] - lo) < 1e-9 and abs(cur[1] - hi) < 1e-9:
            return
        if self._lut_auto.isChecked():
            self._lut_auto.setChecked(False)  # a manual value leaves per-frame Auto
        self._clim[(self._node_id, ch)] = (lo, hi)
        # widen the histogram's data range if the typed value exceeds it, so the handle
        # can actually sit there, then fit the view to the new window.
        ckey = (self._node_id, ch)
        dmin, dmax = self._drange.get(ckey, (lo, hi))
        self._drange[ckey] = (min(dmin, lo), max(dmax, hi))
        hst = self._hists.get(ch)
        if hst is not None:
            hst.set_range(*self._drange[ckey])
            hst.set_window(lo, hi)
            hst.fit_view_to_window()          # auto-fit the new manual value
        self._apply_lut(ch)

    def _sync_luts(self) -> None:
        """Refresh EVERY channel's histogram — range, window, gamma, tint, and the
        (subsampled) distribution of its shown plane. Cheap: sub-sampled per channel."""
        if self._node_id is None:
            return
        for ch, hst in self._hists.items():
            ckey = (self._node_id, ch)
            pl = self._planes.get(ch)
            if ckey not in self._drange and pl is not None:
                self._drange[ckey] = self._display_range(pl)
            dmin, dmax = self._drange.get(ckey, (0.0, 1.0))
            lo, hi = self._clim.get(ckey, (dmin, dmax))
            hst.set_tint(self._chan_colors.get(ch, (120, 170, 255)))
            hst.set_range(dmin, dmax)
            hst.set_window(lo, hi)
            hst.set_gamma(self._gammas.get(ckey, 1.0))
            if pl is not None:
                a = np.asarray(pl)
                step = max(1, int(np.sqrt(a.size / 200_000)))
                hst.set_values(a[::step, ::step] if a.ndim == 2 else a)
            hst.setToolTip(self._hist_tip(ch))
            self._update_lut_edits(ch)

    def _on_lut_window(self, ch: int, lo: float, hi: float) -> None:
        """A drag on channel ``ch``'s handles → update its window and push it (instant)."""
        if self._node_id is None:
            return
        if self._lut_auto.isChecked():
            self._lut_auto.setChecked(False)      # a manual edit leaves per-frame Auto
        self._clim[(self._node_id, ch)] = (float(lo), float(hi))
        self._apply_lut(ch)

    def _on_lut_gamma(self, ch: int, gamma: float) -> None:
        """A drag on channel ``ch``'s midpoint dot → update its gamma (instant)."""
        if self._node_id is None:
            return
        self._gammas[(self._node_id, ch)] = float(gamma)
        self._apply_lut(ch)

    def _on_auto_toggled(self, on: bool) -> None:
        """Auto is a TOGGLE: while on, contrast is recomputed for every frame. Flipping it
        on re-contrasts the current frame immediately."""
        self._auto_on = bool(on)
        if self._auto_on and self._planes and self._node_id is not None:
            self._display(self._node_id, self._planes, self._axes)

    def _fit_all(self) -> None:
        for hst in self._hists.values():
            hst.fit_view_to_window()

    def _apply_lut(self, ch: int) -> None:
        """Push the current window + gamma for ``ch`` to the display. On GPU this is a
        uniform change + repaint — no decode, no re-upload — so it is instantaneous."""
        lo, hi = self._clim.get((self._node_id, ch), (0.0, 1.0))
        gm = self._gammas.get((self._node_id, ch), 1.0)
        hst = self._hists.get(ch)
        if hst is not None:
            hst.setToolTip(self._hist_tip(ch))
        # force: _apply_lut is only called from user LUT actions (drag/gamma/typed value);
        # the field must follow a drag even while it (still) holds keyboard focus.
        self._update_lut_edits(ch, force=True)
        if self._gl is not None:
            self._gl.set_channel(ch, lo, hi,
                                 self._chan_colors.get(ch, (255, 255, 255)), gm)
            self._gl.refresh()
        elif self._planes:
            clims = {c: self._clim.get((self._node_id, c),
                                       self._clim_for(self._node_id, c, pl))
                     for c, pl in self._planes.items()}
            gammas = {c: self._gammas.get((self._node_id, c), 1.0) for c in self._planes}
            self._base_pix = QPixmap.fromImage(self._compose_cpu(clims, gammas))
            self._repaint()

    def show_error(self, node_id: str, trace: str) -> None:
        """Report a failed pull. The image, the M/T/Z ranges and the channel strip are all
        left exactly as they were — a failure produces nothing to put there, and blanking
        the last good frame would throw away the only thing the user still has to look at.

        That makes the panel *lie*: the previous graph's picture sits under the current
        graph's controls. So the status line has to say so, in the error colour, naming the
        strips as well as the image. Without it a stale frame is indistinguishable from a
        fresh one, and the failure mode is nastier than a missed error message: resetting
        Z-Project's method to ``none`` on a chain whose downstream pull then fails looks
        exactly like the reset not working — the stale z==1 frame stays on screen with the Z
        strip still spanning a single plane, so the node that *did* do its job takes the
        blame (2026-08-04)."""
        self._stop_play()
        last = [ln for ln in trace.strip().splitlines() if ln.strip()][-1]
        stale = (" · SHOWING THE PREVIOUS RESULT — the image and the M/T/Z ranges below "
                 "are the last successful pull's, not this graph's" if self._planes else "")
        self._set_status(f"{node_id} FAILED — {last}{stale}", error=True)
        self._view.setToolTip(trace)

    def show_running(self, node_id: str) -> None:
        self._set_status(f"pulling {node_id}…")

    def _set_status(self, text: str, *, error: bool = False) -> None:
        """Write the status line, tinting it :data:`~nodelab_v2.theme.ERROR` for a failed
        pull and clearing that tint for every ordinary message — so the warning cannot
        outlive the stale frame it is about."""
        self._status_error = bool(error)
        self._status.setText(text)
        self._status.set_color(T.ERROR if error else T.MUTED)

    # ── axes / channels rebuild ────────────────────────────────────────────────
    @staticmethod
    def _channel_key(axes, dataset) -> tuple:
        """The per-channel IDENTITY of a payload — the names and emissions the channel strip
        renders — as part of the strip's rebuild key.

        The key used to be the axis sizes alone, and that is wrong the moment a graph has two
        channel branches (2026-08-03). ``Split → ch0 → …`` and ``Split → ch1 → …`` produce
        payloads of the *identical* shape, so switching the Viewer from one to the other left
        the key unchanged and the strip was never rebuilt: the second branch kept showing the
        FIRST channel's name, its emission tint and its LUT — the two branches were
        indistinguishable on screen even once their metadata was right.

        Names and emissions, not the whole metadata dict: those are exactly what
        :meth:`_rebuild_channels` reads, so this key changes iff the strip would draw
        differently, and an unrelated calibration edit does not throw away the user's LUTs."""
        md = getattr(dataset, "metadata", {}) or {}
        n = int(getattr(axes, "c", 1) or 1)

        def _head(key):
            vals = md.get(key)
            return (tuple(str(v) for v in vals[:n])
                    if isinstance(vals, (list, tuple)) else ())

        return (_head("channel_names"), _head("channel_emission_nm"))

    def _apply_axes(self, axes, dataset) -> None:
        self._apply_cursor_ranges(axes)
        self._rebuild_channels(axes, dataset)

    def _sync_axis_label(self, ax: str, suffix: str = "") -> None:
        """The ``value/max`` readout beside a strip, with ``·N`` when N frames are picked
        — the picks scope the run, so their count belongs where the axis is read."""
        sld = self._sliders[ax]
        picked = len(sld.selection())
        self._val_lbls[ax].setText(f"{sld.value()}/{sld.maximum()}"
                                   + (f" ·{picked}" if picked else "") + suffix)

    def _apply_cursor_ranges(self, axes) -> None:
        """Range/enable the M/T/Z strips for ``axes``. Under the solo-frame scope every
        extent comes from the SOURCE totals instead (:meth:`set_solo`) — the payload holds
        only the scoped frames and planes, but the strips are what choose them. Signals
        stay blocked so re-ranging never fires a request of its own."""
        for i, ax in enumerate(_AXES):
            n = getattr(axes, ax)
            if self._solo is not None:
                n = max(n, self._solo[i])
            sld = self._sliders[ax]
            sld.blockSignals(True)
            sld.setMaximum(max(0, n - 1))
            sld.setEnabled(n > 1)
            sld.blockSignals(False)
            self._sync_axis_label(ax)
            playable = n > 1
            self._play_btns[ax].setEnabled(playable)
            self._fps_spins[ax].setEnabled(playable)
            if not playable and self._play_btns[ax].isChecked():
                self._play_btns[ax].setChecked(False)

    def _rebuild_channels(self, axes, dataset) -> None:
        nc = getattr(axes, "c", 1)
        md = getattr(dataset, "metadata", {}) or {}
        bd = md.get("bit_depth")
        self._bit_depth = int(bd) if bd else None
        names = md.get("channel_names") or []
        emis = md.get("channel_emission_nm") or []
        self._chan_names = [str(names[i]) if i < len(names) and names[i] else f"Ch{i + 1}"
                            for i in range(nc)]
        self._chan_colors = {}
        for i in range(nc):
            nm = emis[i] if i < len(emis) else None
            col = T.emission_qcolor(nm)
            # a colour the user picked for this channel wins over its emission colour
            self._chan_colors[i] = self._chan_color_user.get(
                self._color_key(i), (col.red(), col.green(), col.blue()))

        # ── overlay channels (V2.19) ─────────────────────────────────────────────
        # A `view.overlay` node contributes composed planes at indices ABOVE the payload's
        # own channel count. They are display-only — nothing downstream sees them — but they
        # get the full channel treatment (toggle, colour, LUT, gamma) because that is
        # exactly the control an overlay needs, and it already exists. Their default tint is
        # a fixed magenta rather than an emission colour: the secondary is a different
        # FILE, so its emission says nothing about how it should read against the primary,
        # and magenta-on-green is the convention for a two-source overlay.
        ovl = dict(getattr(self, "_overlay_chans", {}) or {})
        ovl_idx = sorted(i for i in ovl if i >= nc)
        for i in ovl_idx:
            while len(self._chan_names) <= i:
                self._chan_names.append(f"Ch{len(self._chan_names) + 1}")
            self._chan_names[i] = str(ovl[i])
            self._chan_colors[i] = self._chan_color_user.get(
                self._color_key(i), (255, 64, 200))
        n_total = nc + len(ovl_idx)

        # keep only still-valid active channels; default to channel 0 if none. An overlay
        # channel is switched ON the moment it FIRST appears — an overlay you have to go and
        # enable is an overlay that looks broken — but only the first time: one the user
        # switched off stays off, where before every pull switched it back on and its toggle
        # button looked broken instead (2026-08-05, "each channel should be its state from
        # before"). `_chan_seen` is keyed the same way the colour override is, so it survives a
        # node switch and a re-pull for the same reason.
        # The first-appearance rule applies to the payload's OWN channels too, not just to
        # overlay ones. It used to run over `ovl_idx` alone — indices >= nc — and
        # `channel.merge` grows the real channel axis (`merge.py`: `replace(ax, c=ax.c +
        # sax.c, …)`), so its new channels are INSIDE nc and were only ever filtered by the
        # line above, never appended. The result was the merge's own headline symptom: the
        # buttons appear with the right names and the picture does not change, because the
        # channels the node was added to combine arrive switched off.
        #
        # `_chan_seen` is keyed by channel NAME (`_color_key`), so "first appearance" means
        # first time that named channel is seen in this session — a channel the user switched
        # off stays off through a node switch and a re-pull, which is the property the
        # overlay version was written for and the reason this can be widened safely.
        self._active_channels = [c for c in self._active_channels
                                 if c < nc or c in ovl] or [0]
        # Auto-enable stops at the sampler bank, so the DEFAULT state is always one the
        # composite can actually render. Past that the user can still switch more on and the
        # status line says what the composite dropped — but arriving in a state that silently
        # hides a channel would be the same defect this loop is here to fix.
        for i in list(range(nc)) + ovl_idx:
            key = self._color_key(i)
            if key not in self._chan_seen:
                self._chan_seen.add(key)
                if (i not in self._active_channels
                        and len(self._active_channels) < _GL_MAX_CH):
                    self._active_channels.append(i)

        # rebuild the per-channel LUT columns: [channel toggle] over [its histogram]
        for col in self._lut_cols:
            col.setParent(None)
            col.deleteLater()
        self._lut_cols = []
        self._chan_btns = {}
        self._hists = {}
        self._lut_edits = {}
        for i in list(range(nc)) + ovl_idx:
            btn = QPushButton(self._chan_names[i])
            btn.setCheckable(True)
            btn.setChecked(i in self._active_channels)
            btn.setCursor(Qt.PointingHandCursor)
            em = emis[i] if i < len(emis) else None
            btn.setToolTip(f"{self._chan_names[i]}"
                           + (f" · {em:.0f} nm" if isinstance(em, (int, float)) else "")
                           + (" — the OVERLAY source, placed by stage position; "
                              "display only, no downstream node sees it"
                              if i >= nc else "")
                           + " — click to toggle in the composite, "
                             "RIGHT-CLICK to set its colour")
            btn.clicked.connect(lambda _c, idx=i: self._on_channel_toggle(idx))
            # right-click → the colour menu (presets / picker / back to emission colour)
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda pos, idx=i: self._channel_color_menu(idx, pos))
            self._style_channel_btn(btn, i)
            hst = HistogramLUT()
            hst.set_tint(self._chan_colors.get(i, (120, 170, 255)))
            hst.setVisible(not self._compact)
            hst.window_changed.connect(lambda lo, hi, c=i: self._on_lut_window(c, lo, hi))
            hst.gamma_changed.connect(lambda gm, c=i: self._on_lut_gamma(c, gm))
            # editable black/white readouts — click to type an exact value
            lo_e, hi_e = QLineEdit(), QLineEdit()
            for e, which in ((lo_e, "black point"), (hi_e, "white point")):
                e.setProperty("role", "lutedit")
                e.setAlignment(Qt.AlignCenter)
                e.setToolTip(f"{which} — type a value + Enter")
                e.setVisible(not self._compact)
                e.editingFinished.connect(lambda c=i: self._on_lut_edit(c))
            erow = QHBoxLayout()
            erow.setContentsMargins(0, 0, 0, 0)
            erow.setSpacing(3)
            erow.addWidget(lo_e)
            erow.addWidget(hi_e)
            colw = QWidget()
            cl = QVBoxLayout(colw)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            cl.addWidget(btn)
            cl.addWidget(hst, 1)
            cl.addLayout(erow)
            self._lut_strip.addWidget(colw, 1)
            self._lut_cols.append(colw)
            self._chan_btns[i] = btn
            self._hists[i] = hst
            self._lut_edits[i] = (lo_e, hi_e)

    # ── per-channel colour (right-click a channel button) ───────────────────────
    def _color_key(self, idx: int) -> str:
        """The key a user-chosen colour is remembered under — the channel's NAME (so it
        follows that channel through every node and re-pull), falling back to its index
        for an unnamed channel."""
        if idx < len(self._chan_names) and self._chan_names[idx]:
            return self._chan_names[idx]
        return f"#{idx}"

    def _emission_color(self, idx: int) -> Tuple[int, int, int]:
        """Channel ``idx``'s colour from its emission wavelength — what "Reset" returns
        to, re-derived from the live dataset metadata."""
        md = getattr(self._dataset, "metadata", {}) or {}
        emis = md.get("channel_emission_nm") or []
        col = T.emission_qcolor(emis[idx] if idx < len(emis) else None)
        return (col.red(), col.green(), col.blue())

    def _channel_color_menu(self, idx: int, pos: QPoint) -> None:
        """The channel colour menu: the additive presets, the full colour picker, and a
        way back to the emission colour."""
        btn = self._chan_btns.get(idx)
        if btn is None:
            return
        name = self._chan_names[idx] if idx < len(self._chan_names) else f"Ch{idx}"
        menu = QMenu(btn)
        menu.setStyleSheet(T.menu_qss())
        head = menu.addAction(f"{name} colour")
        head.setEnabled(False)
        menu.addSeparator()
        cur = self._chan_colors.get(idx)
        for label, rgb in _COLOR_PRESETS:
            act = menu.addAction(_swatch(rgb), label)
            act.setCheckable(True)
            act.setChecked(cur == rgb)
            act.triggered.connect(lambda _c=False, r=rgb: self._set_channel_color(idx, r))
        menu.addSeparator()
        pick = menu.addAction("Custom colour…")
        pick.triggered.connect(lambda: self._pick_channel_color(idx))
        emis = self._emission_color(idx)
        reset = menu.addAction(_swatch(emis), "Reset to emission colour")
        reset.setEnabled(self._color_key(idx) in self._chan_color_user)
        reset.triggered.connect(lambda: self._set_channel_color(idx, None))
        menu.exec(btn.mapToGlobal(pos))

    def _pick_channel_color(self, idx: int) -> None:
        r, g, b = self._chan_colors.get(idx, (255, 255, 255))
        name = self._chan_names[idx] if idx < len(self._chan_names) else f"Ch{idx}"
        col = QColorDialog.getColor(QColor(r, g, b), self, f"{name} colour")
        if col.isValid():
            self._set_channel_color(idx, (col.red(), col.green(), col.blue()))

    def _set_channel_color(self, idx: int, rgb: Optional[Tuple[int, int, int]]) -> None:
        """Give channel ``idx`` a colour — or ``None`` to drop the override and go back to
        its emission colour.

        Display-only, like a LUT change: the colour is a shader uniform on the GPU path
        (instant, no re-upload) and one re-composite on the CPU path — so no re-pull, and
        the choice is remembered by channel name across pulls and nodes."""
        key = self._color_key(idx)
        if rgb is None:
            self._chan_color_user.pop(key, None)
            rgb = self._emission_color(idx)
        else:
            rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            self._chan_color_user[key] = rgb
        self._chan_colors[idx] = rgb
        btn = self._chan_btns.get(idx)
        if btn is not None:
            self._style_channel_btn(btn, idx)
        hst = self._hists.get(idx)
        if hst is not None:
            hst.set_tint(rgb)
        if self._gl is not None:
            self._gl.set_tiles(self._tiles())        # pane legends carry the new colour
        if self._planes and self._node_id is not None:
            self._apply_lut(idx)                     # pushes colour + window to the display
        else:
            self._repaint()

    def _style_channel_btn(self, btn: QPushButton, idx: int) -> None:
        r, g, b = self._chan_colors.get(idx, (196, 200, 208))
        from PySide6.QtGui import QColor
        col = QColor(r, g, b)
        on_text = _text_on(col)
        btn.setStyleSheet(f"""
            QPushButton {{ border:1px solid rgb({r},{g},{b}); border-radius:5px;
                padding:3px 9px; font-weight:600; color:rgb({r},{g},{b});
                background:transparent; }}
            QPushButton:checked {{ background:rgb({r},{g},{b}); color:{on_text}; }}
        """)

    # ── the iteration strip (V2.19) ────────────────────────────────────────────
    def set_iterations(self, labels: Sequence[str], current: int = 0) -> None:
        """Show (or hide) the iteration strip. ``labels`` is one short caption per
        iteration — the value that iteration used — and empty hides the row.

        Called on every selection/pull, so it must be idempotent and must not fire
        :attr:`iteration_changed` while it is only *reflecting* state; otherwise setting
        the cursor here would immediately ask the window to write it back."""
        labels = [str(t) for t in labels]
        if labels == self._iter_values and (
                not labels or self._iter_strip.value() == current):
            return
        self._iter_values = labels
        self._iter_row.setVisible(bool(labels))
        if not labels:
            return
        self._iter_strip.blockSignals(True)
        self._iter_strip.setRange(0, len(labels) - 1)
        self._iter_strip.setValue(max(0, min(len(labels) - 1, int(current))))
        self._iter_strip.blockSignals(False)
        self._iter_strip.setToolTip(
            "Which iteration of the Iterate node is on screen. Drag to compare; the "
            "others are already computed, so flipping is instant.")
        self._sync_iteration_label()

    def _sync_iteration_label(self) -> None:
        i = self._iter_strip.value()
        n = len(self._iter_values)
        cap = self._iter_values[i] if 0 <= i < n else ""
        self._iter_lbl.setText(f"{i + 1}/{n}  {cap}" if n else "")

    def _on_iteration(self, *_a) -> None:
        self._sync_iteration_label()
        self.iteration_changed.emit(int(self._iter_strip.value()))

    # ── interaction ────────────────────────────────────────────────────────────
    def _on_slider(self, ax: str) -> None:
        self._sync_axis_label(ax)
        self.request_changed.emit()

    def _on_channel_toggle(self, idx: int) -> None:
        active = set(self._active_channels)
        if idx in active:
            active.discard(idx)
        else:
            active.add(idx)
        if not active:                       # never leave zero channels shown
            active = {idx}
            self._chan_btns[idx].setChecked(True)
        self._active_channels = sorted(active)
        self.request_changed.emit()          # re-pull → _display re-syncs the histograms

    # ── playback (wall-clock timer, frame-dropping) ─────────────────────────────
    def _interval_ms(self, ax: str) -> int:
        return int(max(1.0, 1000.0 / max(0.5, self._fps_spins[ax].value())))

    def _on_play(self, ax: str, on: bool) -> None:
        if on:
            if self._sliders[ax].maximum() < 1:
                self._play_btns[ax].setChecked(False)
                return
            # one axis at a time
            for other in _AXES:
                if other != ax and self._play_btns[other].isChecked():
                    self._play_btns[other].blockSignals(True)
                    self._play_btns[other].setChecked(False)
                    self._play_btns[other].setText("▶")
                    self._play_btns[other].blockSignals(False)
            self._playing_axis = ax
            self._play_btns[ax].setText("⏸")
            self._last_frame_t = None
            self._fps_ema = None
            self._gated = False
            # The gate may be raised by the handler this emit reaches (the window asks the
            # runner to preload the series first), so it goes out BEFORE the timer starts —
            # otherwise the first tick lands on a cold frame and playback begins by stuttering,
            # which is the thing the preload exists to prevent.
            self.playing.emit(True, ax)
            if not self._gated:
                self._play_timer.start(self._interval_ms(ax))
        else:
            if self._playing_axis == ax:
                self._stop_play()
            self._play_btns[ax].setText("▶")
            self.playing.emit(False, ax)

    def set_play_gate(self, on: bool, note: str = "") -> None:
        """Hold playback without stopping it — the frames are still being decoded.

        The button stays in its playing state and ``_playing_axis`` is untouched, so this is
        *not* :meth:`_stop_play`: the user asked to play, and the answer is "in a moment",
        which is what makes a computed series play smoothly instead of at decode speed. Lower
        the gate and the timer starts from wherever the cursor now is."""
        on = bool(on)
        self._gated = on
        if on:
            self._play_timer.stop()
            if note:
                self._set_status(note)
        elif self._playing_axis is not None and self._play_btns[self._playing_axis].isChecked():
            self._last_frame_t = None            # do not charge the wait to the frame rate
            self._fps_ema = None
            self._play_timer.start(self._interval_ms(self._playing_axis))

    def play_gated(self) -> bool:
        return bool(getattr(self, "_gated", False))

    def _stop_play(self) -> None:
        ax = self._playing_axis
        self._playing_axis = None
        self._gated = False
        self._play_timer.stop()
        if ax is not None:
            btn = self._play_btns[ax]
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.setText("▶")
            btn.blockSignals(False)
            self._sync_axis_label(ax)

    def _retarget_fps(self, ax: str) -> None:
        if self._playing_axis == ax and self._play_timer.isActive():
            self._play_timer.setInterval(self._interval_ms(ax))

    def _tick_play(self) -> None:
        """Fire on the wall clock: advance the cursor and request the frame. The plane
        request is served from the warm cache (a miss decodes one small plane); if a tick
        arrives while the previous is still working, Qt coalesces the timeout — i.e. the
        frame is dropped — so playback keeps real-time rather than lagging behind."""
        ax = self._playing_axis
        if ax is None:
            return
        sld = self._sliders[ax]
        nxt = sld.value() + 1
        if nxt > sld.maximum():
            nxt = 0
        sld.setValue(nxt)          # → _on_slider → request_changed → request_plane (fast)

    def _measure_fps(self) -> None:
        if self._playing_axis is None:
            return
        now = time.perf_counter()
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else \
                    0.6 * self._fps_ema + 0.4 * inst
        self._last_frame_t = now
        self._sync_axis_label(self._playing_axis,
                              f" · {self._fps_ema:.1f} fps" if self._fps_ema else "")

    # ── overlays ───────────────────────────────────────────────────────────────
    # The panel's job is to turn the viewed Dataset + (m,t,z,c) into geometry; the LOOK
    # lives in nodelab_v2.overlays (settings + renderer) and is edited in the Overlays
    # popup. Extraction is cached per (dataset, coords, plane size, settings revision)
    # because a pan/zoom repaints constantly and must not re-walk the attribute tables.
    def open_overlay_dialog(self) -> None:
        """Open (or raise) the Overlays popup — the single entry point to every overlay's
        configuration. Non-modal, so edits are seen against the live image."""
        from nodelab_v2.overlay_dialog import OverlayDialog
        if self._ovl_dialog is None:
            dlg = OverlayDialog(self.overlays, self._renderer, self,
                                layer_names=self.overlay_layer_names)
            dlg.changed.connect(self._overlays_changed)
            self._ovl_dialog = dlg
            if self._ovl_sources:
                dlg._status.setText(" · ".join(self._ovl_sources))
            btn = self._ovl_btn
            if btn.isVisible():                  # drop it just under the button
                dlg.move(btn.mapToGlobal(QPoint(0, btn.height() + 6)))
        dlg = self._ovl_dialog
        dlg.reload()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _overlays_changed(self) -> None:
        """A settings edit: drop the caches and redraw (no re-pull, no re-composite —
        overlays are painted over the finished image)."""
        self._ovl_rev += 1
        self._renderer.invalidate()
        self._geo_key = None
        self._view.refresh()

    def overlay_enabled(self, tab: str) -> bool:
        return bool(self.overlays.group(tab).enabled)

    def set_overlay_enabled(self, tab: str, on: bool) -> None:
        """Turn one domain's overlay on/off (what the old per-domain checkboxes did)."""
        grp = self.overlays.group(tab)
        if bool(grp.enabled) != bool(on):
            grp.enabled = bool(on)
            if self._ovl_dialog is not None:
                self._ovl_dialog.reload()
            self._overlays_changed()

    def _ensure_geometry(self) -> None:
        """Extract the overlay geometry for the viewed frame if the cache is stale."""
        key = (id(self._dataset), self.coords(),
               None if self._ref_plane is None else self._ref_plane.shape[:2],
               self._ovl_rev)
        if key == self._geo_key:
            return
        self._geo_key = key
        s = self.overlays
        self._geo_points = self._point_marks() if s.points.enabled else []
        self._geo_tracks = self._track_paths() if s.tracks.enabled else []
        self._geo_lab = self._label_plane() if s.labels.enabled else None
        self._geo_lab_keys = self._label_keys(self._geo_lab)
        self._geo_mesh = self._mesh_sections() if s.mesh.enabled else []
        self._geo_scalar = self._scalar_plane() if s.voxels.enabled else None
        self._geo_vectors = self._vector_field() if s.vectors.enabled else None
        self._geo_diff = self._diff_sets() if s.diff.enabled else (None, None)

    def _paint_overlays(self, p: QPainter) -> None:
        """Paint every enabled overlay in **widget** coordinates.

        Called back by whichever image surface is live (GL after its draw, the CPU view
        from ``drawForeground`` with the world transform reset). Because the geometry is
        mapped through the backend's ``plane_to_widget`` at paint time and every size in
        the settings is a screen size, zooming moves the overlay with the image without
        changing how thick or how big it is.
        """
        if self._ref_plane is None:
            return
        mp = getattr(self._view, "plane_to_widget", None)
        if mp is None:                            # a surface without the overlay contract
            return
        H, W = self._ref_plane.shape[:2]
        ax = self._axes
        self._ensure_geometry()
        frame = OV.OverlayFrame(
            map_pt=mp, plane_wh=(W, H),
            sy=H / max(1, ax.y) if ax is not None else 1.0,
            sx=W / max(1, ax.x) if ax is not None else 1.0,
            points=self._geo_points, tracks=self._geo_tracks, mesh=self._geo_mesh,
            label_plane=self._geo_lab, label_keys=self._geo_lab_keys,
            scalar=self._geo_scalar, vectors=self._geo_vectors,
            diff_a=self._geo_diff[0], diff_b=self._geo_diff[1],
            # the PAYLOAD's t: the trail modes compare it against the `t` values carried by
            # the structure rows this pull produced, which under the solo-frame scope are
            # frame-local (0) — not the global frame the slider names.
            current_t=self._payload_coords()[1],
        )
        self._renderer.paint(p, self.overlays, frame)
        self._paint_pick(p)                   # the armed gesture rides ON TOP of everything

    # ── parameter picking (V2.16) ─────────────────────────────────────────────
    #
    # The whole surface is: a session (the state machine + maths, in nodelab_v2.picker),
    # an event filter that turns mouse events into gestures, a probe that reads the pixel
    # or the object under the cursor, and a painter for the rubber band. Nothing here knows
    # what node it is serving — the socket's `pick_kind` decided that.

    def picking(self) -> bool:
        return self._pick is not None

    def has_image(self) -> bool:
        """True when something is displayed — a canvas or histogram pick has nothing to aim
        at otherwise, and arming one would look like the button was broken."""
        return self._ref_plane is not None

    def arm_pick(self, req: PickRequest, calib: Optional[Calibration] = None) -> None:
        """Begin (or re-target) a pick. ``calib`` comes from the window, which owns the
        document and therefore the node's propagated ``pixel_size_um`` — the viewer holds
        only pixels, so it cannot work out microns for itself.

        The three surfaces diverge here. An INSTANT kind ("use the channel I'm looking at")
        has nothing to aim, so it commits immediately and never shows a bar; a HISTOGRAM kind
        shows the bar so the user can move the handles first; a CANVAS kind shows the bar and
        takes over the mouse."""
        self.cancel_pick(quiet=True)
        if req.surface == "instant":
            m, t, z, c = self.coords()
            _ms, _ts, zs = self.frame_selection()
            self.pick_committed.emit(req.node_id, instant_values(
                req, channel=c, frame=t, channels=self.channels(),
                z_picks=zs, z_total=(self._axes.z if self._axes is not None else 0)))
            return
        self._pick = PickSession(req, calib or Calibration())
        self._pick_bar.configure(req)
        self._pick_bar.set_readout(self._pick.readout())
        self._pick_bar.show()
        if req.surface == "canvas":
            self._install_pick_filter(True)
        self.setFocus(Qt.OtherFocusReason)    # so Esc / Enter reach keyPressEvent
        self.pick_armed.emit(True)
        self._refresh_surface()

    def cancel_pick(self, *, quiet: bool = False) -> None:
        """Disarm without committing. ``quiet`` suppresses the signal — used when re-arming
        onto a different param, where the state never actually returns to 'not picking'."""
        if self._pick is None:
            return
        self._install_pick_filter(False)
        self._pick = None
        self._pick_bar.hide()
        if not quiet:
            self.pick_armed.emit(False)
        self._refresh_surface()

    def _apply_pick(self) -> None:
        """Commit whatever the session has. A canvas pick that was never aimed commits
        nothing rather than writing a zero — Apply on an untouched gesture is a no-op, not
        a way to silently blank a parameter."""
        s = self._pick
        if s is None:
            return
        if s.req.surface == "histogram":
            vals = self._histogram_pick_values(s.req)
        else:
            vals = s.values()
        node_id = s.req.node_id
        self.cancel_pick(quiet=True)
        self.pick_armed.emit(False)
        if vals:
            self.pick_committed.emit(node_id, vals)

    def _histogram_pick_values(self, req: PickRequest) -> Dict[str, Any]:
        """Read the live LUT for a ``percentile`` / ``gamma`` pick: the window the handles
        are sitting on, against the channel's own display range."""
        ch = self._lut_channel()
        node = self._node_id or ""
        plane = self._planes.get(ch, self._ref_plane)
        lo, hi = self._clim.get((node, ch), (0.0, 1.0))
        vmin, vmax = self._drange.get(
            (node, ch), self._display_range(plane) if plane is not None else (0.0, 1.0))
        return histogram_values(req, lo=lo, hi=hi, vmin=vmin, vmax=vmax,
                                gamma=self._gammas.get((node, ch), 1.0))

    def _set_pick_tool(self, tool: str) -> None:
        if self._pick is not None:
            self._pick.tool = tool
            self._pick.pts = []
            self._sync_pick_readout()

    def _set_pick_op(self, op: str) -> None:
        if self._pick is not None:
            self._pick.op = op

    def _pick_action(self, what: str) -> None:
        if self._pick is None:
            return
        if what == "undo":
            self._pick.undo_shape()
        else:
            self._pick.add_action(what)
        self._sync_pick_readout()

    def _sync_pick_readout(self) -> None:
        if self._pick is not None:
            self._pick_bar.set_readout(self._pick.readout())
        self._refresh_surface()

    def _refresh_surface(self) -> None:
        ref = getattr(self._view, "refresh", None)
        if callable(ref):
            ref()

    # ── mouse capture ─────────────────────────────────────────────────────────
    def _pick_targets(self) -> List[QWidget]:
        """The widgets a canvas pick must filter events on.

        Two, not one, because the backends deliver mouse events to different places: the
        GPU surface is a plain widget, while the CPU :class:`_ImageView` is a QGraphicsView
        that routes them to its *viewport*. Filtering both means the pick works identically
        on either — and means consuming the press is what stops the CPU view's
        ``ScrollHandDrag`` from panning, with no mode to set and restore."""
        out: List[QWidget] = [self._view]
        vp = getattr(self._view, "viewport", None)
        if callable(vp):
            try:
                out.append(vp())
            except Exception:                    # noqa: BLE001 — not a graphics view
                pass
        return out

    def _install_surface_filter(self) -> None:
        """Filter the image surface's mouse events **permanently** — the hover readout has
        to see every move, not only the moves made while a pick is armed. Called at
        construction and again after a GL→CPU swap, which replaces the widget being
        filtered. Mouse tracking is forced on both targets: without it a QWidget only
        reports moves with a button held, and the readout would only ever update mid-drag.
        """
        for w in self._pick_targets():
            if w is None:
                continue
            w.installEventFilter(self)
            w.setMouseTracking(True)

    def _install_pick_filter(self, on: bool) -> None:
        """Arm/disarm the CANVAS pick on the surface. The event filter itself stays
        installed either way (:meth:`_install_surface_filter` owns it, and
        :meth:`eventFilter` no-ops on picks when none is armed) — this only swaps the
        cursor, which is the part that is actually about the pick."""
        self._install_surface_filter()
        for w in self._pick_targets():
            if w is None:
                continue
            if on:
                w.setCursor(Qt.CrossCursor)
            else:
                w.unsetCursor()
        if not on:
            # the CPU view's own hand cursor comes back with its drag mode
            self._view.setCursor(Qt.ArrowCursor if self._gl is not None
                                 else Qt.OpenHandCursor)

    def eventFilter(self, obj, e):            # noqa: N802 — Qt naming
        """Track the pointer for the hover readout, and — while a canvas pick is armed —
        turn mouse events on the image surface into pick gestures.

        Wheel events are deliberately NOT consumed: zooming to see what you are aiming at is
        part of aiming. Everything else about the press/drag/release is, so the underlying
        surface never pans or re-frames mid-gesture."""
        et = e.type()
        # The hover update runs FIRST and unconditionally: a pick consumes the move event
        # below, and reading the pixel you are aiming at is if anything more useful then.
        if et == QEvent.MouseMove:
            self._on_hover(e.position())
        elif et in (QEvent.Leave, QEvent.WindowDeactivate):
            self._clear_hover()
        s = self._pick
        if s is None or s.req.surface != "canvas":
            return super().eventFilter(obj, e)
        if et == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
            pt = self._pick_plane_pt(e.position())
            if pt is not None:
                s.press(*pt, probe=self._probe(pt))
                self._sync_pick_readout()
            return True
        if et == QEvent.MouseMove:
            pt = self._pick_plane_pt(e.position())
            if pt is not None:
                if e.buttons() & Qt.LeftButton:
                    s.drag(*pt)
                else:
                    s.hover(*pt)
                self._sync_pick_readout()
            return True
        if et == QEvent.MouseButtonRelease and e.button() == Qt.LeftButton:
            pt = self._pick_plane_pt(e.position())
            if pt is not None:
                s.release(*pt, probe=self._probe(pt))
                self._sync_pick_readout()
                # A single-value gesture that completed commits itself: making the user
                # then reach for Apply would add a click to every pick for no decision.
                # `shapes` is the exception — its value is a LIST the user keeps building.
                if s.done and s.req.kind != "shapes":
                    self._apply_pick()
            return True
        if et == QEvent.MouseButtonDblClick:
            s.close_polygon()
            self._sync_pick_readout()
            return True
        return super().eventFilter(obj, e)

    def keyPressEvent(self, e) -> None:       # noqa: N802 — Qt naming
        if self._pick is not None:
            if e.key() == Qt.Key_Escape:
                # Esc backs out one level: an in-progress stroke first, then the pick.
                if self._pick.pts:
                    self._pick.cancel_gesture()
                    self._sync_pick_readout()
                else:
                    self.cancel_pick()
                e.accept()
                return
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._apply_pick()
                e.accept()
                return
        super().keyPressEvent(e)

    # ── hover readout (what is under the pointer) ─────────────────────────────
    #
    # Four questions, answered on one line: WHERE in the node's own pixel grid, where
    # that is in microns, where it is on the MICROSCOPE'S stage, and WHAT each shown
    # channel reads there — the viewed node's number, and the untouched file's number
    # beside it when an enhancement moved it.
    #
    # Everything is derived from state the panel already holds (`_planes`, `_axes`,
    # `_dataset.metadata`) plus one callback for the raw pixels, so a mouse move costs a
    # couple of array index reads and a string build — no decode, no repaint of the image.

    def _clear_hover(self) -> None:
        # the hint rather than an empty string: the row costs a line of height either way,
        # and a blank one reads as a rendering fault instead of an invitation
        self._hover_pt = None
        self._hover_lbl.setText(_HOVER_HINT if self._ref_plane is not None else "")

    def _on_hover(self, wpt: QPointF) -> None:
        """The pointer moved over the image surface — re-read the pixel under it.

        Off the image (the letterbox margin, or a split pane's gap) clears the line rather
        than freezing the last value: a stale coordinate that keeps sitting there reads as
        a live one."""
        pt = self._pick_plane_pt(wpt)
        if pt is None or self._ref_plane is None:
            self._clear_hover()
            return
        self._hover_pt = pt
        self._hover_lbl.setText(self._hover_text(pt))

    def _sample(self, plane: Optional[np.ndarray],
                pt: Tuple[float, float]) -> Optional[float]:
        """``plane``'s value at a point given in the node's FULL-RESOLUTION pixel space.

        The plane is scaled to the node's axes rather than assumed 1:1 — a display plane
        arrives stride-decimated when the volume is larger than the render budget, and the
        raw plane is decimated by the same rule, so both land on the right texel through
        the same conversion."""
        if plane is None or self._axes is None:
            return None
        H, W = plane.shape[:2]
        px = int(round(pt[0] * W / max(1, self._axes.x)))
        py = int(round(pt[1] * H / max(1, self._axes.y)))
        if not (0 <= px < W and 0 <= py < H):
            return None
        return float(plane[py, px])

    @staticmethod
    def _fmt_intensity(v: float) -> str:
        """Integers as integers, computed floats to 4 significant figures. A count read
        off a 16-bit sensor is an exact integer and printing it as ``1284.0`` invites the
        reader to look for a precision that is not there."""
        return f"{int(round(v))}" if float(v).is_integer() else f"{v:.4g}"

    def _stage_xy(self, m: int) -> Optional[Tuple[float, float]]:
        """Multipoint ``m``'s stage coordinate (µm) from the file's position log, or
        ``None`` when the file carried none (every TIFF, and an ND2 whose SDK did not fill
        the loop in). Read live from the payload's metadata — :data:`STAGE_KEYS` rides on
        the dataset, not on the locked calibration schema."""
        md = getattr(self._dataset, "metadata", None) or {}
        xy = md.get("stage_xy_um") or []
        if not (0 <= m < len(xy)):
            return None
        try:
            return float(xy[m][0]), float(xy[m][1])
        except (TypeError, ValueError, IndexError):
            return None

    def _hover_text(self, pt: Tuple[float, float]) -> str:
        """The readout line for a point in the node's full-resolution pixel space.

        The stage coordinate is offered ONLY when the raw probe is available, because that
        callback's contract is exactly the guarantee this arithmetic needs: the viewed node
        preserves the source's geometry, so ``(x, y)`` still indexes the field the stage log
        describes and the field's centre is still its centre. After a crop or a resample
        both of those stop being true, so the line quietly drops the stage part rather than
        placing the pixel somewhere it is not.

        Axis convention: image ``+x``/``+y`` (right / down) are taken to run along stage
        ``+x``/``+y``. Nikon records the FIELD CENTRE, so the offset is measured from the
        centre of the frame. If a stitched montage from this scope comes out mirrored in
        Y, that is the sign to flip here — the file does not state the handedness."""
        m, t, z, _c = self.coords()
        parts = [f"x {int(pt[0])}, y {int(pt[1])} px"]

        md = getattr(self._dataset, "metadata", None) or {}
        psize = md.get("pixel_size_um")
        try:
            psize = float(psize) if psize else None
        except (TypeError, ValueError):
            psize = None

        raw_ok = False
        vals: List[str] = []
        for ch in sorted(self._planes):
            plane = self._planes[ch]
            dv = self._sample(plane, pt)
            if dv is None:
                continue
            name = self._chan_names[ch] if ch < len(self._chan_names) else f"Ch{ch}"
            txt = f"{name} {self._fmt_intensity(dv)}"
            raw = None
            if self.raw_plane_cb is not None and self._node_id is not None:
                try:
                    raw = self.raw_plane_cb(self._node_id, m, t, z, ch)
                except Exception:            # noqa: BLE001 — a readout never breaks a pull
                    raw = None
            if raw is not None:
                raw_ok = True
                # `is not plane`: when the viewed node IS the load, the runner hands back
                # the very array being displayed (same cache entry), and "1284 (raw 1284)"
                # would imply a processing step that never happened.
                rv = self._sample(raw, pt) if raw is not plane else None
                if rv is not None and self._fmt_intensity(rv) != self._fmt_intensity(dv):
                    txt += f" (raw {self._fmt_intensity(rv)})"
            vals.append(txt)

        if psize:
            parts.append(f"{pt[0] * psize:.1f}, {pt[1] * psize:.1f} µm")
            stage = self._stage_xy(m) if raw_ok else None
            if stage is not None and self._axes is not None:
                sx = stage[0] + (pt[0] - (self._axes.x - 1) / 2.0) * psize
                sy = stage[1] + (pt[1] - (self._axes.y - 1) / 2.0) * psize
                # no thousands separator: stage coordinates run to five digits, and
                # "-20,581.6, -29,481.1" puts a comma inside each number AND between
                # them — the reader cannot tell which comma pairs the numbers.
                parts.append(f"stage {sx:.1f}, {sy:.1f} µm")
        if vals:
            parts.append(" · ".join(vals))
        return "  ·  ".join(parts)

    # ── coordinates + probes ──────────────────────────────────────────────────
    def _disp_scale(self) -> Tuple[float, float]:
        """``(sx, sy)`` — displayed-plane pixels per AXES pixel. The payload can arrive
        decimated (a scoped or downsampled pull), and every value a pick produces is in the
        node's own full-resolution pixel space, so the two spaces have to be kept apart.
        Same factors the overlay renderer uses, for the same reason."""
        if self._ref_plane is None:
            return 1.0, 1.0
        H, W = self._ref_plane.shape[:2]
        ax = self._axes
        if ax is None:
            return 1.0, 1.0
        return W / max(1, ax.x), H / max(1, ax.y)

    def _pick_plane_pt(self, wpt: QPointF) -> Optional[Tuple[float, float]]:
        """A widget point → ``(x, y)`` in the node's own full-resolution pixel space."""
        w2p = getattr(self._view, "widget_to_plane", None)
        if w2p is None or self._ref_plane is None:
            return None
        disp = w2p(wpt)
        if disp is None:
            return None
        sx, sy = self._disp_scale()
        return (disp[0] / (sx or 1.0), disp[1] / (sy or 1.0))

    def _probe(self, pt: Tuple[float, float]) -> Optional[float]:
        """The number under the cursor, in the units the armed pick wants: an intensity for
        ``level``, an object's pixel/voxel COUNT for ``area``. ``None`` means there was
        nothing to read there, which the session turns into an explanation rather than a
        zero."""
        s = self._pick
        if s is None:
            return None
        if s.req.kind == "level":
            return self._probe_intensity(pt)
        if s.req.kind == "area":
            return self._probe_object_size(pt, volume=s.req.unit == "um3")
        return None

    def _probe_intensity(self, pt: Tuple[float, float]) -> Optional[float]:
        """The raw value of the sampled pixel, from the channel whose histogram is active —
        a threshold is compared against ONE channel's numbers, so the eyedropper has to read
        that channel rather than the composite the eye sees."""
        plane = self._planes.get(self._lut_channel(), self._ref_plane)
        if plane is None:
            return None
        sx, sy = self._disp_scale()
        px, py = int(round(pt[0] * sx)), int(round(pt[1] * sy))
        H, W = plane.shape[:2]
        if not (0 <= px < W and 0 <= py < H):
            return None
        return float(plane[py, px])

    def point_layer_names(self) -> list:
        """The Point tables on the viewed payload — what the Points tab's picker offers."""
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return []
        return sorted({str(k[1]) for k in ds.attributes
                       if k[0] is Domain.POINT and k[1]})

    def overlay_layer_names(self, tab: str) -> list:
        """The live layer names for a ``layer``-kind field on ``tab``.

        One entry point rather than a callable per tab: the dialog is generic over
        :data:`nodelab_v2.overlays.FIELDS` and should not learn which domain each tab means."""
        if tab == "labels":
            return self.label_layer_names()
        if tab == "points":
            return self.point_layer_names()
        return []

    def label_layer_names(self) -> list:
        """The label rasters on the viewed payload — every integer 6-D Voxel layer, by name.

        What the Labels tab's **Which layer** picker offers. Read off the payload rather than
        the edit-time catalogue so it lists what is genuinely drawable right now."""
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return []
        out = []
        for (dom, _layer, name), attr in ds.attributes.items():
            if dom is not Domain.VOXEL or not name:
                continue
            vals = attr.values
            if np.issubdtype(vals.dtype, np.integer) and vals.ndim == 6:
                out.append(str(name))
        return sorted(set(out))

    def _label_source(self) -> Optional[np.ndarray]:
        """The full-resolution 6-D integer Voxel layer the Labels overlay draws.

        Honours ``overlays.labels.layer`` when it names a raster that is present; otherwise
        falls back to the historical guess — the one with the most regions in the viewed
        plane. The fallback is what made this unpredictable: several label rasters on one
        Dataset is the normal case, so "most regions" silently drew whichever was most
        fragmented and no click could change it (2026-08-04).

        Shared with :meth:`_label_plane` so the picked raster, the painted outline and the
        object-size probe can never disagree about WHICH layer is on screen."""
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return None
        want = str(getattr(self.overlays.labels, "layer", "") or "").strip()
        m, t, z, c = self._payload_coords()
        # Auto prefers the layers the VIEWED NODE itself produced, in declaration order — you
        # view a node to see what it made (2026-08-04). Ranking every raster on the payload by
        # id instead drew the seeds branch's labels, or the copied areas raster whose ids are
        # the SOURCE's (in the hundreds while a handful of its regions survive).
        rasters = {}
        for (dom, _layer, name), attr in ds.attributes.items():
            if dom is not Domain.VOXEL or not name:
                continue
            vals = attr.values
            if np.issubdtype(vals.dtype, np.integer) and vals.ndim == 6:
                rasters[str(name)] = vals
        if want:
            if want in rasters:
                return rasters[want]       # an explicit pick wins, empty plane or not
            # named a layer this payload does not carry (a stale pick, or a graph edit that
            # renamed it) — fall through to the guess rather than draw nothing
            return self._label_source_auto()
        if self.own_layers_cb is not None and self._node_id:
            try:
                for nm in self.own_layers_cb(self._node_id):
                    if nm in rasters:
                        return rasters[nm]
            except Exception:  # noqa: BLE001 — a preference must never break the view
                pass
        best, best_regions = None, 0
        for vals in rasters.values():
            try:
                plane = vals[m, t, z, c]
            except IndexError:
                continue
            # COUNT the regions; `plane.max()` is the largest id, which a raster carrying
            # another node's numbering wins on while showing almost nothing
            regions = int(np.count_nonzero(np.unique(plane)))
            if regions >= 1 and regions >= best_regions:
                best, best_regions = vals, regions
        return best

    def _label_source_auto(self) -> Optional[np.ndarray]:
        """:meth:`_label_source` with the explicit pick ignored — what a STALE pick falls back
        to, so a layer that has been renamed away draws the node's own output instead of
        nothing."""
        keep = getattr(self.overlays.labels, "layer", "")
        try:
            self.overlays.labels.layer = ""
            return self._label_source()
        finally:
            self.overlays.labels.layer = keep

    def _label_layer_values(self) -> Optional[np.ndarray]:
        """The full-resolution raster the label overlay is drawing. Full resolution on
        purpose: :meth:`_label_plane` decimates for painting, and counting an object's pixels
        off a decimated raster would under-report its area by the square of the
        decimation."""
        return self._label_source()

    def _probe_object_size(self, pt: Tuple[float, float], *,
                           volume: bool) -> Optional[float]:
        """The size of the labelled object under the cursor, as a COUNT of pixels (or of
        voxels through the whole stack when the target socket is µm³).

        Counted straight off the label raster rather than read from the Label table's own
        ``area`` column. The column is only there when the producer wrote one, and it counts
        voxels for a 3D label and pixels for a 2D one — so reading it would make the answer
        depend on which node made the labels. Counting here is one boolean reduction over a
        plane (or a volume) on a click, and it means the same gesture measures a raster from
        any producer, including a plain threshold mask with no table at all."""
        vals = self._label_layer_values()
        if vals is None:
            return None
        m, t, z, c = self._payload_coords()
        px, py = int(round(pt[0])), int(round(pt[1]))
        try:
            plane = vals[m, t, z, c]
            H, W = plane.shape[:2]
            if not (0 <= px < W and 0 <= py < H):
                return None
            lid = int(plane[py, px])
            if lid <= 0:
                return 0.0                    # background — "nothing there", not "no data"
            block = vals[m, t, :, c] if volume else plane
            return float(np.count_nonzero(block == lid))
        except (IndexError, ValueError):
            return None

    # ── painting the gesture ──────────────────────────────────────────────────
    def _paint_pick(self, p: QPainter) -> None:
        """Draw the armed gesture over the image, in widget pixels like every overlay."""
        s = self._pick
        if s is None or s.req.surface != "canvas" or self._ref_plane is None:
            return
        mp = getattr(self._view, "plane_to_widget", None)
        if mp is None:
            return
        sx, sy = self._disp_scale()

        def W(pt: Tuple[float, float]) -> QPointF:
            return mp(pt[0] * sx, pt[1] * sy)

        accent = T.ACCENT
        p.setRenderHint(QPainter.Antialiasing, True)
        # already-committed ROI shapes read dimmer than the live one, so the stroke being
        # drawn is always the visually loudest thing on the image
        for shp in s.shapes:
            self._paint_roi_shape(p, shp, W, sx)
        kind, pts = s.req.kind, s.pts
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(accent, 1.6))
        if kind == "radius" and pts:
            c = W(pts[0])
            for mag in s.picked:              # rings already set, kept for comparison
                self._paint_ring(p, c, self._mag_px(s, mag) * sx, T.alpha(accent, 150))
            if len(pts) >= 2:
                r = ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5
                self._paint_ring(p, c, r * sx, accent)
            self._paint_cross(p, c)
        elif kind == "rect":
            self._paint_crop_rect(p, s, W)
        elif kind == "grid" and len(pts) >= 2:
            self._paint_grid(p, s, W, sx)
        elif kind == "distance":
            a = W(pts[0]) if pts else None
            b = (W(pts[1]) if len(pts) >= 2
                 else (W(s.hover_pt) if s.hover_pt is not None else None))
            if a is not None and b is not None:
                p.setPen(QPen(accent, 1.8))
                p.drawLine(a, b)
                self._paint_cross(p, a)
                self._paint_cross(p, b)
            elif a is not None:
                self._paint_cross(p, a)
        elif kind == "level" and pts:
            self._paint_cross(p, W(pts[0]))
        elif kind == "area" and pts:
            if len(pts) >= 2:
                p.setBrush(T.alpha(accent, 44))
                p.drawPolygon(QPolygonF([W(q) for q in pts]))
                p.setBrush(Qt.NoBrush)
            else:
                self._paint_cross(p, W(pts[0]))
        elif kind == "shapes" and pts:
            self._paint_live_shape(p, s, W, sx)

    @staticmethod
    def _mag_px(s: PickSession, mag: float) -> float:
        """A magnitude already in the socket's unit → displayed-plane pixels, so a ring the
        user set earlier keeps sitting on the feature it was measured from."""
        if s.req.unit in ("px", ""):
            return mag
        return mag / (s.calib.lateral or 1.0)

    @staticmethod
    def _paint_ring(p: QPainter, c: QPointF, r_px: float, col) -> None:
        p.setPen(QPen(col, 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, max(1.0, r_px), max(1.0, r_px))

    @staticmethod
    def _paint_cross(p: QPainter, c: QPointF, d: float = 6.0) -> None:
        p.setPen(QPen(T.ACCENT, 1.6))
        p.drawLine(QPointF(c.x() - d, c.y()), QPointF(c.x() + d, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - d), QPointF(c.x(), c.y() + d))

    def _paint_crop_rect(self, p: QPainter, s: PickSession, W) -> None:
        """The crop window: the kept rectangle bright, everything outside it DIMMED.

        Dimming the discard is the whole affordance — a bare outline says "here is a
        rectangle", while a darkened surround says "this is what you are throwing away",
        which is the question a crop actually asks. Corner ticks mark the draggable extent
        and read at any zoom because they are drawn in widget pixels like every overlay."""
        box = s.box()
        if box is None:
            return
        x0, y0, x1, y1 = box
        tl, br = W((x0, y0)), W((x1, y1))
        keep = QRectF(tl, br).normalized()
        # the full image in widget space, so the four discard bands can be filled around
        # the kept rect without needing the widget's own geometry
        H, Wd = self._ref_plane.shape[:2]
        ax = self._axes
        full = QRectF(W((0.0, 0.0)),
                      W((float(ax.x if ax is not None else Wd),
                         float(ax.y if ax is not None else H)))).normalized()
        p.setPen(Qt.NoPen)
        p.setBrush(T.alpha(T.BG, 150))
        for band in (QRectF(full.left(), full.top(), full.width(),
                            max(0.0, keep.top() - full.top())),
                     QRectF(full.left(), keep.bottom(), full.width(),
                            max(0.0, full.bottom() - keep.bottom())),
                     QRectF(full.left(), keep.top(), max(0.0, keep.left() - full.left()),
                            keep.height()),
                     QRectF(keep.right(), keep.top(),
                            max(0.0, full.right() - keep.right()), keep.height())):
            if band.width() > 0 and band.height() > 0:
                p.drawRect(band)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(T.ACCENT, 1.8))
        p.drawRect(keep)
        # corner ticks
        p.setPen(QPen(T.ACCENT, 3.0))
        d = min(14.0, keep.width() / 3.0, keep.height() / 3.0)
        for cx, cy, ux, uy in ((keep.left(), keep.top(), 1, 1),
                               (keep.right(), keep.top(), -1, 1),
                               (keep.left(), keep.bottom(), 1, -1),
                               (keep.right(), keep.bottom(), -1, -1)):
            p.drawLine(QPointF(cx, cy), QPointF(cx + ux * d, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + uy * d))

    def _paint_grid(self, p: QPainter, s: PickSession, W, sx: float) -> None:
        """The subset box being dragged, plus a few of its neighbours at the current stride
        — the point of picking a correlation grid on the image is seeing how the lattice
        lands on the texture, which one lone rectangle does not show."""
        (x0, y0), (x1, y1) = s.pts[0], s.pts[-1]
        box = max(abs(x1 - x0), abs(y1 - y0))
        if box <= 0:
            return
        step = box
        if s.picked:                          # phase 2: the box is fixed, the stride moves
            step = max(1.0, self._mag_px(s, s.picked[0]))
            box, step = step, box
        p.setBrush(Qt.NoBrush)
        for i in range(-1, 3):
            for j in range(-1, 3):
                a = W((x0 + i * step, y0 + j * step))
                b = W((x0 + i * step + box, y0 + j * step + box))
                live = i == 0 and j == 0
                p.setPen(QPen(T.ACCENT if live else T.alpha(T.ACCENT, 90),
                              1.8 if live else 1.0))
                p.drawRect(QRectF(a, b).normalized())

    def _paint_live_shape(self, p: QPainter, s: PickSession, W, sx: float) -> None:
        cut = s.op == "cut"
        col = T.ERROR if cut else T.ACCENT
        p.setPen(QPen(col, 1.8))
        p.setBrush(T.alpha(col, 40))
        pts = s.pts
        if s.tool == "polygon":
            poly = [W(q) for q in pts]
            if s.hover_pt is not None:
                poly.append(W(s.hover_pt))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF(poly))
            for q in pts:
                self._paint_cross(p, W(q), 4.0)
            return
        if s.tool == "brush":
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, max(1.5, 2 * s.brush_px * sx)))
            p.drawPolyline(QPolygonF([W(q) for q in pts]))
            return
        if len(pts) < 2:
            return
        a, b = W(pts[0]), W(pts[-1])
        if s.tool == "circle":
            r = ((pts[-1][0] - pts[0][0]) ** 2 + (pts[-1][1] - pts[0][1]) ** 2) ** 0.5
            p.drawEllipse(a, r * sx, r * sx)
        elif s.tool == "ellipse":
            p.drawEllipse(QRectF(a, b).normalized())
        else:
            p.drawRect(QRectF(a, b).normalized())

    def _paint_roi_shape(self, p: QPainter, shp: dict, W, sx: float) -> None:
        """One already-committed ROI shape, drawn from the same ``[y, x]`` schema the
        kernel rasterizes — so what the user sees is what will be masked."""
        kind = str(shp.get("type", ""))
        cut = str(shp.get("op", "add")) == "cut"
        col = T.ERROR if cut else T.WIRE
        p.setPen(QPen(col, 1.4, Qt.DashLine if cut else Qt.SolidLine))
        p.setBrush(T.alpha(col, 30))
        verts = [(float(v[1]), float(v[0])) for v in (shp.get("vertices") or [])]
        if kind in ("rect", "ellipse") and len(verts) >= 2:
            r = QRectF(W(verts[0]), W(verts[1])).normalized()
            if kind == "ellipse":
                p.drawEllipse(r)
            else:
                p.drawRect(r)
        elif kind == "circle":
            c = shp.get("center") or [0, 0]
            r = float(shp.get("radius", 0) or 0) * sx
            p.drawEllipse(W((float(c[1]), float(c[0]))), r, r)
        elif kind == "polygon" and len(verts) >= 3:
            p.drawPolygon(QPolygonF([W(q) for q in verts]))
        elif kind == "brush" and verts:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, max(1.5, 2 * float(shp.get("radius", 8) or 8) * sx)))
            p.drawPolyline(QPolygonF([W(q) for q in verts]))
        elif kind in ("invert", "clear"):
            # A whole-mask action has no geometry to draw; say so at the image's corner so
            # the picture still accounts for every entry in the shape list.
            p.setPen(QPen(col, 1))
            p.drawText(QPointF(12, 18 if kind == "invert" else 34), f"· {kind}")

    def _point_count(self) -> int:
        """Points drawn on the viewed plane (the status line's ``· N points``)."""
        if not (self.overlays.points.enabled and self._axes is not None):
            return 0
        return len(self._points_here())

    def _point_tally(self) -> Tuple[int, int]:
        """``(on the viewed plane, on the viewed frame)`` — the status line's two counts.

        The pair exists because the two numbers genuinely differ, and reporting only the
        first is what let a real result read as an empty one: a 3-D detection's ``z`` is a
        continuous depth, so its particles spread across the planes they were found at and
        the plane the cursor happens to sit on may hold none of them. The status line then
        said *nothing at all* (the count was suppressed when it was zero), so there was no
        way to tell "detected nothing" from "detected 41 000, none on this plane".

        Counted straight off the dataset rather than off :attr:`_geo_points`, and that is the
        whole point: the drawn marks depend on ``z_project``, so with projection OFF — the
        default — every mark is on-plane, the two numbers would always agree and the
        off-plane note could never appear. The tally has to answer "what did this frame
        detect", which is not a question about what is currently being drawn.
        """
        if not (self.overlays.points.enabled and self._axes is not None):
            return (0, 0)
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return (0, 0)
        m, t, z, _c = self._payload_coords()
        by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.POINT:
                by_layer.setdefault(layer, {})[name] = attr.values
        on = frame = 0
        for cols in by_layer.values():
            if "y" not in cols or "x" not in cols:
                continue
            keep = np.ones(len(cols["y"]), dtype=bool)
            for key, want in (("m", m), ("t", t)):
                if key in cols:
                    keep &= np.asarray(cols[key]) == want
            frame += int(keep.sum())
            zs = cols.get("z")
            if zs is None:
                on += int(keep.sum())
            else:
                # np.rint matches the paint path's round() — both round half to even
                on += int((keep & (np.rint(np.asarray(zs, dtype=float)) == z)).sum())
        return (on, frame)

    def _point_marks(self) -> List[OV.PointMark]:
        """Every Point-domain row that belongs on the viewed frame, as draw-ready marks.

        Positions stay in *structure* coordinates (full-resolution image pixels); the
        renderer scales them into the displayed plane. The colour key is the point's
        **palette slot** — its track's when a tracker has linked it, its own id otherwise
        — so a detection keeps one colour across frames; the layer index is the per-layer
        key. With ``z_project`` on, off-plane detections come along flagged
        ``on_plane=False`` and the renderer dims them.

        **``z_project`` is authoritative for every layer, including a 3-D one.** A previous
        revision made a ``z_kind="subpixel"`` layer project regardless of the setting, on the
        reasoning that a continuous depth belongs to the volume rather than to one plane. It
        was the wrong call twice over: the checkbox became inert, and on a real 3-D detection
        it drew *every* plane's particles at once — thousands of overlapping glyphs at the
        off-plane opacity blend into one flat wash, so a per-point palette also stopped
        looking like a palette. The problem it was trying to solve (a plane that holds none
        of the detections reading as "nothing was detected") is solved by
        :meth:`_point_tally` reporting the off-plane count in the status line, which costs
        the picture nothing.
        """
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return []
        m, t, z, _c = self._payload_coords()
        project = bool(self.overlays.points.z_project)
        pal = self._palette()
        by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.POINT:
                by_layer.setdefault(layer, {})[name] = attr.values
        # one table when the picker names a present one; every table otherwise (2026-08-04).
        # A pick this payload lacks falls through to all, rather than drawing nothing — the
        # same rule the Labels picker uses for a stale name.
        _want = str(getattr(self.overlays.points, "layer", "") or "").strip()
        if _want and _want in {str(k) for k in by_layer}:
            by_layer = {k: v for k, v in by_layer.items() if str(k) == _want}
        out: List[OV.PointMark] = []
        # sorted: the per-layer colour must not depend on dict insertion order
        for li, layer in enumerate(sorted(by_layer, key=lambda v: str(v))):
            cols = by_layer[layer]
            if "y" not in cols or "x" not in cols:
                continue
            ys, xs = cols["y"], cols["x"]
            zs = cols.get("z"); ms = cols.get("m"); ts = cols.get("t")
            ids = cols.get("id")
            slots = pal.points.get((Domain.POINT, layer))
            for i in range(len(ys)):
                if ms is not None and int(ms[i]) != m:
                    continue
                if ts is not None and int(ts[i]) != t:
                    continue
                zpl = z if zs is None else int(round(float(zs[i])))
                on_plane = zpl == z
                if not on_plane and not project:
                    continue
                key = OV.slot_of(slots, int(ids[i])) if ids is not None else i + 1
                out.append(OV.PointMark(float(ys[i]), float(xs[i]), key, li, on_plane,
                                        zpl))
        return out

    def _points_here(self):
        """On-plane point positions ``(y, x)`` — kept for callers/tests and the count."""
        if self.overlays.points.enabled:
            self._ensure_geometry()
            marks = self._geo_points
        else:
            marks = self._point_marks()
        return [(mk.y, mk.x) for mk in marks if mk.on_plane]

    def _label_plane(self) -> Optional[np.ndarray]:
        """The viewed plane of the chosen label raster, decimated to the reference plane.

        Selection lives in :meth:`_label_source` — one place, so the outline that is painted
        and the raster the size-probe counts are always the same layer."""
        if self._ref_plane is None:
            return None
        vals = self._label_source()
        if vals is None:
            return None
        m, t, z, c = self._payload_coords()
        try:
            best = vals[m, t, z, c]
        except IndexError:
            return None
        ay, ax_ = best.shape
        ty = max(1, int(np.ceil(ay / self._ref_plane.shape[0])))
        tx = max(1, int(np.ceil(ax_ / self._ref_plane.shape[1])))
        return best[::ty, ::tx]

    def _scalar_plane(self) -> Optional[np.ndarray]:
        """The Voxel-domain SCALAR layer for the viewed plane, or ``None``.

        The counterpart to :meth:`_label_plane`, and it selects by the opposite test:
        anything **non-integer** is a measurement (a strain component, a distance
        transform, a density, a probability), while an integer raster is a label map and
        belongs to the Labels overlay. A boolean mask is admitted too — it is a scalar in
        every sense the colour ramp cares about — but an integer id raster is not, because
        colour-mapping label ids paints a gradient across objects whose numbering is
        arbitrary and re-issued every frame.

        Picks the layer with the largest finite spread when several qualify: a field with
        no variation carries nothing to look at, and silently choosing it over a real one
        would look like the overlay was broken.
        """
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes") or self._ref_plane is None:
            return None
        m, t, z, c = self._payload_coords()
        best, best_spread = None, -1.0
        for (dom, _layer, _name), attr in ds.attributes.items():
            if dom is not Domain.VOXEL:
                continue
            vals = attr.values
            if vals.ndim != 6:
                continue
            if np.issubdtype(vals.dtype, np.integer) and vals.dtype != np.bool_:
                continue                       # a label raster — the Labels overlay's job
            try:
                plane = np.asarray(vals[m, t, z, c], dtype=float)
            except IndexError:
                continue
            finite = plane[np.isfinite(plane)]
            if finite.size == 0:
                continue
            spread = float(finite.max() - finite.min())
            if spread >= best_spread:
                best, best_spread = plane, spread
        if best is None:
            return None
        ay, ax_ = best.shape
        ty = max(1, int(np.ceil(ay / self._ref_plane.shape[0])))
        tx = max(1, int(np.ceil(ax_ / self._ref_plane.shape[1])))
        return best[::ty, ::tx]

    #: Column-name pairs a Point structure may use for a displacement's (y, x) components,
    #: in preference order. `u`/`v` is what the DVC/DIC kernels emit; the `_um` pair is what
    #: a calibrated field carries; `dy`/`dx` covers a hand-built table. Matched as a PAIR so
    #: a table with only one half is skipped rather than drawn as a half-field.
    _VECTOR_COLUMNS = (("u", "v"), ("u_um", "v_um"), ("dy", "dx"))

    def _vector_field(self) -> Optional[np.ndarray]:
        """Point-domain displacement vectors on the viewed frame → ``(N, 4)`` ``y,x,u,v``.

        Reads the Point structure's own coordinate columns and the first displacement pair
        it carries, filtered to the viewed ``(m, t, c)`` — and to the viewed Z when the
        table is a per-plane 2D field, which is what ``z_kind`` records. A 3D field's
        vectors are shown whole, because their ``z`` is a subpixel coordinate rather than a
        plane index and cutting on it would drop almost everything.
        """
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return None
        m, t, z, c = self._payload_coords()
        by_layer: Dict[str, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.POINT and layer:
                by_layer.setdefault(str(layer), {})[str(name)] = attr.values
        for layer, cols in by_layer.items():
            uy = ux = None
            for a, b in self._VECTOR_COLUMNS:
                if a in cols and b in cols:
                    uy, ux = cols[a], cols[b]
                    break
            if uy is None or "y" not in cols or "x" not in cols:
                continue
            keep = np.ones(len(cols["y"]), dtype=bool)
            for key, want in (("m", m), ("t", t), ("c", c)):
                if key in cols:
                    keep &= np.asarray(cols[key]) == want
            try:
                z_kind = ds.structure_zkind(Domain.POINT, layer)
            except Exception:      # noqa: BLE001 — provenance is advisory here
                z_kind = None
            if z_kind == "plane_index" and "z" in cols:
                keep &= np.asarray(cols["z"]) == z
            idx = np.flatnonzero(keep)
            if idx.size == 0:
                continue
            return np.stack([np.asarray(cols["y"], dtype=float)[idx],
                             np.asarray(cols["x"], dtype=float)[idx],
                             np.asarray(uy, dtype=float)[idx],
                             np.asarray(ux, dtype=float)[idx]], axis=1)
        return None

    def _diff_sets(self):
        """The two Point sets the comparison overlay judges → ``(a, b)`` of ``(N,2)`` yx.

        Taken from the **two Point layers** on the viewed frame, in first-appearance order:
        a graph that produced two candidate detections (an old detector and a new one, a
        prediction and a hand-annotated truth) carries both, and that is the comparison
        worth drawing. ``(None, None)`` unless there are exactly two — with one there is
        nothing to compare, and with three the pairing to draw is a question the overlay
        cannot answer for you.
        """
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes"):
            return (None, None)
        m, t, z, c = self._payload_coords()
        by_layer: Dict[str, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.POINT and layer:
                by_layer.setdefault(str(layer), {})[str(name)] = attr.values
        usable = [(lyr, cols) for lyr, cols in by_layer.items()
                  if "y" in cols and "x" in cols]
        if len(usable) != 2:
            return (None, None)
        out = []
        for lyr, cols in usable:
            keep = np.ones(len(cols["y"]), dtype=bool)
            for key, want in (("m", m), ("t", t), ("c", c)):
                if key in cols:
                    keep &= np.asarray(cols[key]) == want
            try:
                z_kind = ds.structure_zkind(Domain.POINT, lyr)
            except Exception:      # noqa: BLE001 — provenance is advisory
                z_kind = None
            if z_kind == "plane_index" and "z" in cols:
                keep &= np.asarray(cols["z"]) == z
            idx = np.flatnonzero(keep)
            out.append(np.stack([np.asarray(cols["y"], dtype=float)[idx],
                                 np.asarray(cols["x"], dtype=float)[idx]], axis=1))
        return (out[0], out[1])

    def _mesh_sections(self) -> List[OV.MeshSection]:
        """Every Mesh element on the viewed frame, cut by the viewed Z plane.

        The Mesh domain stores three flat CSR strata per mesh under one domain, keyed by a
        layer sub-name (``L`` element rows / ``L/vert`` / ``L/face``) — see
        :mod:`nodegraph.mesh`. This walks the store directly rather than going through
        ``read_mesh`` so a partially-written or foreign mesh can never raise inside a paint
        call: anything that does not look like a mesh is simply skipped.

        Coordinates stay in *structure* pixels (voxel ``z,y,x``, the domain's storage
        convention); the renderer scales them into the displayed plane, exactly as for
        points and tracks.
        """
        ds = self._dataset
        out: List[OV.MeshSection] = []
        if ds is None or not hasattr(ds, "attributes"):
            return out
        m, t, z, c = self._payload_coords()
        near = max(0.0, float(self.overlays.mesh.near_z))
        vertex_style = self.overlays.mesh.style == "points"
        by_layer: Dict[Any, Dict[str, np.ndarray]] = {}
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.MESH and layer is not None:
                by_layer.setdefault(layer, {})[name] = attr.values
        for layer in sorted(by_layer, key=lambda v: str(v)):
            base, part = MESH_PART(str(layer))
            if part is not None:                       # only iterate ELEMENT buckets
                continue
            el = by_layer[layer]
            vt = by_layer.get(f"{base}/vert")
            fc = by_layer.get(f"{base}/face")
            if vt is None or fc is None:
                continue
            if not ({"id", "m", "t", "c", "vert_start", "vert_count",
                     "face_start", "face_count"} <= set(el)
                    and {"z", "y", "x"} <= set(vt) and {"v0", "v1", "v2"} <= set(fc)):
                continue
            vz, vy, vx = (np.asarray(vt[k], dtype=float) for k in ("z", "y", "x"))
            tri = np.column_stack([np.asarray(fc[f"v{i}"], dtype=np.int64)
                                   for i in range(3)])
            verts = np.column_stack([vz, vy, vx])
            for row in range(len(np.asarray(el["id"]))):
                if (int(np.asarray(el["m"])[row]) != m
                        or int(np.asarray(el["t"])[row]) != t
                        or int(np.asarray(el["c"])[row]) != c):
                    continue
                vs = int(np.asarray(el["vert_start"])[row])
                vc = int(np.asarray(el["vert_count"])[row])
                fs = int(np.asarray(el["face_start"])[row])
                nf = int(np.asarray(el["face_count"])[row])
                oid = int(np.asarray(el["id"])[row])
                sub_v = verts[vs:vs + vc]
                sub_f = tri[fs:fs + nf] - vs           # v0..v2 are GLOBAL indices
                sec = OV.MeshSection(oid)
                if vertex_style:
                    keep = np.abs(sub_v[:, 0] - float(z)) <= near
                    sec.verts = [(float(a), float(b))
                                 for a, b in zip(sub_v[keep, 1], sub_v[keep, 2])]
                else:
                    sec.loops, sec.closed = OV.mesh_section(sub_v, sub_f, float(z))
                if sec.loops or sec.verts:
                    out.append(sec)
        return out

    def _member_layers(self):
        """``{(domain, layer): {col: values}}`` for every Point/Label layer that carries
        a joinable ``id,y,x`` triple — the member position source for Track trajectories."""
        ds = self._dataset
        out: Dict[tuple, Dict[str, np.ndarray]] = {}
        if ds is None or not hasattr(ds, "attributes"):
            return out
        for (dom, layer, name), attr in ds.attributes.items():
            if dom in (Domain.POINT, Domain.LABEL):
                out.setdefault((dom, layer), {})[name] = attr.values
        return {k: v for k, v in out.items()
                if {"id", "y", "x"} <= set(v)}

    def _track_layers(self):
        """``{layer: {col: values}}`` for every Track-domain table in the dataset."""
        ds = self._dataset
        out: Dict[Any, Dict[str, np.ndarray]] = {}
        if ds is None or not hasattr(ds, "attributes"):
            return out
        for (dom, layer, name), attr in ds.attributes.items():
            if dom is Domain.TRACK:
                out.setdefault(layer, {})[name] = attr.values
        return out

    @staticmethod
    def _best_member_layer(members: Dict[tuple, Dict[str, np.ndarray]],
                           member_ids: np.ndarray) -> Optional[tuple]:
        """Which member layer a Track table is talking about.

        A Track table stores only ``track_id, t, member_id`` — the geometry lives on the
        member (Point/Label) domain, keyed by ``id``, and Point and Label ids share an id
        space (both start at 1). So the layer is resolved by *coverage*: the one holding
        the most of this table's member ids. Ties go to the lexicographically first key,
        so the answer never depends on dict insertion order.
        """
        want = set(np.asarray(member_ids).astype(np.int64).tolist())
        best, best_cov = None, 0
        for key in sorted(members, key=lambda k: (str(k[0]), str(k[1]))):
            ids = np.asarray(members[key]["id"]).astype(np.int64).tolist()
            cov = len(want & set(ids))
            if cov > best_cov:
                best, best_cov = key, cov
        return best

    # ── the identity palette (V2.17) ──────────────────────────────────────────
    def _palette(self) -> _Palette:
        """The viewed dataset's identity palette, built once and cached.

        Keyed on the dataset OBJECT, not the viewed frame: the whole point is that the
        answer is the same on every T. Datasets are immutable in the engine, so a new
        pull is a new object and gets a fresh palette; the reference is pinned because
        ``id()`` of a freed dataset could otherwise alias a new one.
        """
        ds = self._dataset
        if ds is not None and id(ds) == self._pal_key and self._pal is not None:
            return self._pal
        self._pal = self._build_palette()
        self._pal_key, self._pal_ds = (id(ds) if ds is not None else None), ds
        return self._pal

    def _build_palette(self) -> _Palette:
        """Work out one palette slot per **physical object** and index it by member id.

        Three steps, all vectorized (this runs over every structure row in the dataset,
        not just the viewed frame):

        1. **Objects.** Every Point/Label row belongs to a track (its object is that
           track) or to nothing (its object is itself). A track *prefers* the slot of its
           first member, so at the frame where a cell first appears it keeps exactly the
           colour it would have had untracked — and then holds it for the rest of the
           series instead of flashing a new one each frame.
        2. **Neighbours.** Per ``(m, t)``, each member's nearest few members; mapped to
           object pairs and deduplicated across frames (the same cells are neighbours in
           every frame, so this collapses hard — which is also why sampling
           :data:`MAX_NEIGHBOR_FRAMES` of them costs nothing in quality).
        3. **Slots.** :func:`~nodelab_v2.overlays.deconflict_slots` hands out the slots,
           moving an object off its preference only when a neighbour is too close on the
           colour wheel.

        Anything unexpected in the tables is skipped rather than raised on: this feeds a
        paint call, and a colour is never worth a crash.
        """
        members = self._member_layers()
        if not members:
            return _Palette()
        tracks = self._track_layers()

        # ── 1. member row → object id (ints, so the mapping stays vectorized) ──
        obj_row: Dict[tuple, np.ndarray] = {}        # member layer → (N,) object id
        ids_of: Dict[tuple, np.ndarray] = {}
        for key, cols in members.items():
            ids = np.asarray(cols["id"]).astype(np.int64)
            ids_of[key] = ids
            obj_row[key] = np.full(len(ids), -1, dtype=np.int64)
        prefer: Dict[int, int] = {}
        # Every layer numbers its ids from 0, so objects in two different layers would PREFER
        # the same slots — and keep them, because the neighbour graph below is built per layer
        # (two marks in different layers are never neighbours, so de-confliction never looks
        # at the pair). The result was a point in one layer painted the same colour as a point
        # in another under `per_point`: a 4-row and a 2-row Point layer gave 4 colours for 6
        # marks. Each layer therefore gets its own disjoint band of preferences. The first
        # band starts at 0, so a single-layer dataset — which is nearly every dataset — keeps
        # exactly the colours it had, and a track still prefers its first member's slot
        # because both halves are offset by the same layer's base.
        layer_base: Dict[Any, int] = {}
        _base = 0
        for key in sorted(ids_of, key=str):
            layer_base[key] = _base
            ids = ids_of[key]
            _base += int(max(0, int(ids.max()) + 1)) if ids.size else 0
        track_slots: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}   # layer → (tids, objs)
        next_obj = 0
        for tlayer in sorted(tracks, key=str):
            cols = tracks[tlayer]
            if not {"track_id", "member_id"} <= set(cols):
                continue
            tid = np.asarray(cols["track_id"]).astype(np.int64)
            mem = np.asarray(cols["member_id"]).astype(np.int64)
            mkey = self._best_member_layer(members, mem)
            if mkey is None:
                continue
            ids = ids_of[mkey]
            if ids.size == 0:
                continue
            order = np.argsort(ids, kind="stable")
            pos = np.searchsorted(ids[order], mem)
            hit = (pos < len(ids)) & (ids[order][np.clip(pos, 0, len(ids) - 1)] == mem)
            rows = order[np.clip(pos, 0, max(0, len(ids) - 1))][hit]
            tid_hit = tid[hit]
            if rows.size == 0:
                continue
            uniq, inv = np.unique(tid_hit, return_inverse=True)
            objs = np.arange(next_obj, next_obj + len(uniq), dtype=np.int64)
            next_obj += len(uniq)
            obj_row[mkey][rows] = objs[inv]
            # a track prefers the slot of its FIRST member — the colour that region
            # already wears the frame it appears in
            first = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(first, inv, ids[rows])
            prefer.update({int(o): int(f) + layer_base.get(mkey, 0)
                           for o, f in zip(objs.tolist(), first.tolist())})
            track_slots[tlayer] = (uniq, objs)
        for key, ids in ids_of.items():             # untracked rows: each is its own object
            loose = np.nonzero(obj_row[key] < 0)[0]
            if loose.size:
                objs = np.arange(next_obj, next_obj + loose.size, dtype=np.int64)
                next_obj += loose.size
                obj_row[key][loose] = objs
                prefer.update({int(o): int(v) + layer_base.get(key, 0)
                               for o, v in zip(objs.tolist(), ids[loose].tolist())})

        # ── 2. neighbour graph, per sampled frame, deduplicated across frames ──
        too_many = next_obj > MAX_PALETTE_OBJECTS
        if too_many and not self._warned_palette:
            self._warned_palette = True
            print(f"[overlays] {next_obj} objects (>{MAX_PALETTE_OBJECTS}) - neighbouring "
                  f"colours not de-conflicted; each object still keeps one colour across T",
                  file=sys.stderr, flush=True)
        chunks: List[np.ndarray] = []
        for key, cols in members.items():
            ids = ids_of[key]
            if len(ids) < 2 or too_many:
                continue
            ys = np.asarray(cols["y"], dtype=float)
            xs = np.asarray(cols["x"], dtype=float)
            zeros = np.zeros(len(ids), dtype=np.int64)
            ms = np.asarray(cols["m"]).astype(np.int64) if "m" in cols else zeros
            ts = np.asarray(cols["t"]).astype(np.int64) if "t" in cols else zeros
            frame = ms * (int(ts.max()) + 1) + ts if ts.size else zeros
            frames = np.unique(frame)
            if len(frames) > MAX_NEIGHBOR_FRAMES:
                frames = frames[np.linspace(0, len(frames) - 1,
                                            MAX_NEIGHBOR_FRAMES).astype(int)]
            for f in frames.tolist():
                sel = np.nonzero(frame == f)[0]
                if sel.size < 2:
                    continue
                # (y, x) only: the confusion this prevents is between two marks the eye
                # compares ON THE VIEWED PLANE, which is exactly the projection to y/x
                pairs = OV.neighbor_pairs(np.stack([ys[sel], xs[sel]], axis=1))
                if pairs.size:
                    chunks.append(obj_row[key][sel][pairs])
        pairs_all = (np.unique(np.concatenate(chunks), axis=0) if chunks
                     else np.zeros((0, 2), dtype=np.int64))
        pairs_all = pairs_all[pairs_all[:, 0] != pairs_all[:, 1]]

        # ── 3. slots, then back down onto the ids each overlay actually holds ──
        slots = (dict(prefer) if too_many else
                 OV.deconflict_slots(prefer, [(int(a), int(b))
                                              for a, b in pairs_all.tolist()]))
        lookup = np.arange(max(1, next_obj), dtype=np.int64)
        for o, sl in slots.items():
            lookup[o] = sl
        pal = _Palette()
        for key, ids in ids_of.items():
            lut = OV.slot_lut(ids, lookup[obj_row[key]])
            (pal.labels if key[0] is Domain.LABEL else pal.points)[key] = lut
        for tlayer, (tids, objs) in track_slots.items():
            pal.tracks[tlayer] = OV.slot_lut(tids, lookup[objs])
        return pal

    def _label_keys(self, lab: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """The ``label id → palette slot`` LUT for the plane :meth:`_label_plane` picked.

        The plane is a Voxel raster and carries no layer name, so the Label table is
        matched by *content*: with one Label layer there is nothing to resolve, and with
        several the plane's top id says which id space it came from.
        """
        if lab is None or lab.size == 0:
            return None
        pal = self._palette()
        if not pal.labels:
            return None
        luts = [pal.labels[k] for k in sorted(pal.labels, key=lambda k: str(k[1]))]
        if len(luts) == 1:
            return luts[0]
        top = int(lab.max())
        covering = [lut for lut in luts if top < len(lut)]
        return covering[0] if covering else max(luts, key=len)

    def _track_paths(self) -> List[OV.TrackPath]:
        """Join every Track layer to its members' positions and return, for the current
        M, one :class:`~nodelab_v2.overlays.TrackPath` per track — its positions ordered
        by ``t``, the index of the vertex at the viewed T, and each vertex's timepoint
        (which is what lets the *trail* modes draw only part of a trajectory).

        A Track layer only stores ``track_id, t, member_id`` — the geometry lives on the
        member (Point/Label) domain, keyed by ``id``. Point and Label ids share an
        id-space (both start at 1), so per Track layer we pick the member layer whose id
        set best covers the members (:meth:`_best_member_layer` — resolves the ambiguity,
        avoids cross-space matches). Each path also carries its **palette slot**, so a
        trajectory is drawn in the same colour as the regions it threads through.
        Positions stay projected over z and filtered to the viewed M."""
        ds = self._dataset
        if ds is None or not hasattr(ds, "attributes") or self._axes is None:
            return []
        m, t, _z, _c = self._payload_coords()
        track_layers = self._track_layers()
        if not track_layers:
            return []
        members = self._member_layers()
        pal = self._palette()
        trajectories = []
        for tlayer, cols in track_layers.items():
            if not {"track_id", "t", "member_id"} <= set(cols):
                continue
            tid = np.asarray(cols["track_id"]).astype(np.int64)
            tt = np.asarray(cols["t"]).astype(np.int64)
            mem = np.asarray(cols["member_id"]).astype(np.int64)
            mkey = self._best_member_layer(members, mem)
            if mkey is None:
                continue
            best_cols = members[mkey]
            slots = pal.tracks.get(tlayer)
            ids = np.asarray(best_cols["id"]).astype(np.int64)
            ys = np.asarray(best_cols["y"], dtype=float)
            xs = np.asarray(best_cols["x"], dtype=float)
            ms = np.asarray(best_cols["m"]).astype(np.int64) if "m" in best_cols else None
            id_to_i = {int(v): i for i, v in enumerate(ids.tolist())}
            for track in np.unique(tid).tolist():
                sel = tid == track
                order = np.argsort(tt[sel], kind="stable")
                ts_ord = tt[sel][order].tolist()
                mem_ord = mem[sel][order].tolist()
                path, cur, ts_kept = [], None, []
                for tv, mv in zip(ts_ord, mem_ord):
                    i = id_to_i.get(int(mv))
                    if i is None or (ms is not None and int(ms[i]) != m):
                        continue
                    path.append((float(ys[i]), float(xs[i])))
                    ts_kept.append(int(tv))
                    if int(tv) == t:
                        cur = len(path) - 1
                if path:
                    trajectories.append(
                        OV.TrackPath(int(track), path, cur, ts_kept,
                                     OV.slot_of(slots, int(track))))
        return trajectories

    def _tracks_here(self):
        """``(track_id, path, current_index, timepoints)`` per track — the tuple form kept
        for callers/tests around :meth:`_track_paths`."""
        if self.overlays.tracks.enabled:
            self._ensure_geometry()
            paths = self._geo_tracks
        else:
            paths = self._track_paths()
        return [(tr.track_id, tr.path, tr.current, tr.times) for tr in paths]

    def _track_color(self, track_id: int, layer: Any = None) -> QColor:
        """The colour track ``track_id`` is drawn in under the current settings — the hue
        of its palette slot, stable across every T and shared with its member regions.

        ``layer`` names the Track table when the dataset holds more than one (each has its
        own id space); with a single table, which is the normal case, it is inferred.
        """
        slots = self._palette().tracks
        if layer is None and len(slots) == 1:
            layer = next(iter(slots))
        return self._renderer.track_color(self.overlays.tracks, int(track_id),
                                          slot=OV.slot_of(slots.get(layer),
                                                          int(track_id)))

    def _repaint(self) -> int:
        """Show the current composite and ask the surface to redraw its overlays.

        The overlays are **not** baked into the pixmap any more (that is precisely what
        made them scale with the zoom on the CPU path) — they are painted on top in widget
        space by :meth:`_paint_overlays`. So this is now a cheap "push the image, ask for a
        repaint", and toggling an overlay never re-composites the channels."""
        if self._ref_plane is None:
            return 0
        if self._gl is not None:
            self._gl.refresh()
            return self._point_count()
        if self._base_pix is None:
            return 0
        self._view.set_pixmap(self._base_pix)
        self._view.refresh()
        return self._point_count()

    # ── styling ────────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QLabel[role="muted"] {{ color:{T.MUTED.name()}; font-size:11px; }}
            QLabel[role="title"] {{ color:{T.INK.name()}; font-weight:600; }}
            QLabel[role="axis"] {{ color:{T.INK_2.name()}; font-family:{T.MONO};
                font-weight:700; min-width:12px; }}
            QGraphicsView {{ border:1px solid {T.BORDER.name()}; border-radius:6px;
                background:{T.BG.name()}; }}
            QCheckBox {{ color:{T.INK_2.name()}; font-size:11px; }}
            QDoubleSpinBox {{ background:{T.BODY.name()}; color:{T.INK.name()};
                border:1px solid {T.BORDER.name()}; border-radius:4px; padding:0px 2px;
                font-family:{T.MONO}; font-size:11px; max-width:50px; max-height:18px; }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width:0; }}
            QLineEdit[role="lutedit"] {{ background:{T.BODY.name()}; color:{T.INK.name()};
                border:1px solid {T.BORDER.name()}; border-radius:4px; padding:1px 2px;
                font-family:{T.MONO}; font-size:10px; max-height:17px; min-width:34px; }}
            QLineEdit[role="lutedit"]:focus {{ border:1px solid {T.ACCENT.name()}; }}
            QToolButton {{ background:transparent; color:{T.INK.name()};
                border:0; padding:0px 4px; font-size:13px; max-height:18px; }}
            QToolButton:checked {{ color:{T.ACCENT.name()}; }}
            QToolButton:disabled {{ color:{T.MUTED.name()}; }}
        """ + T.controls_qss() + f"""
            QCheckBox {{ color:{T.INK_2.name()}; font-size:11px; spacing:5px; }}
            /* The pick bar reads as an armed mode, not as another row of controls: an
               accent-tinted panel with an accent left edge, so it is obvious at a glance
               that the next click on the image will do something unusual. */
            QWidget[role="pickbar"] {{
                background:{T.mix(T.PANEL, T.ACCENT, 0.14).name()};
                border:1px solid {T.ACCENT_DIM.name()};
                border-left:3px solid {T.ACCENT.name()};
                border-radius:7px; }}
            QWidget[role="pickbar"] QLabel {{ background:transparent; }}
            QWidget[role="pickbar"] QLabel[role="readout"] {{ color:{T.ACCENT.name()};
                font-family:{T.MONO}; font-weight:700; }}
            QWidget[role="pickbar"] QToolButton {{ background:{T.BODY.name()};
                border:1px solid {T.BORDER.name()}; border-radius:5px;
                padding:2px 6px; font-size:12px; max-height:22px; }}
            QWidget[role="pickbar"] QToolButton:checked {{
                background:{T.ACCENT_DIM.name()}; border-color:{T.ACCENT.name()};
                color:{T.ACCENT.name()}; }}
        """)
        # self-painted (elided) → QSS can't reach it. A live "showing the previous result"
        # warning keeps its tint: a theme change must not turn it back into a routine line.
        self._status.set_color(T.ERROR if self._status_error else T.MUTED)
        for strip in self._sliders.values():
            strip.update()                   # self-painted from the live theme tokens
        for idx, b in self._chan_btns.items():
            self._style_channel_btn(b, idx)
        dlg = getattr(self, "_ovl_dialog", None)   # restyle() also runs from __init__
        if dlg is not None:
            dlg.restyle()


__all__ = ["ViewerPanel", "plane_to_qimage", "composite_to_qimage", "mosaic_with_clim"]
