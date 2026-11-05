"""An image item's picture from a pulled Dataset (V5.00 step 3).

:func:`render_image` turns an item's (twin-resolved) image block, a Dataset and a frame into
a :class:`Rendered`: the composited plane (the region window of the plane, at a pyramid
level that fits the display budget), the display windows actually used per channel, the
raw planes (a histogram item draws them), and the overlay geometry of the frame — the
same :mod:`nodelab_v2.overlay_geometry` the Viewer draws from — under an
:class:`~nodelab_v2.overlays.OverlaySettings` built from the item's layer looks.
:func:`paint_overlays` then hands that geometry to the Viewer's own
:class:`~nodelab_v2.overlays.OverlayRenderer` with a plane→page mapping and the figure's
stroke scale, so every overlay look is painted by the one implementation.

The compositing is the Viewer's CPU path (:func:`nodelab_v2.viewer.composite_with_clim`),
so a figure's channel colours, windows and gammas mean exactly what the Viewer's do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from nodelab_v2 import overlay_geometry as OG
from nodelab_v2 import overlays as OV
from nodelab_v2.viewer import _autocontrast, _flat_window, composite_with_clim

#: the largest side of a plane read for the DISPLAY (the export re-reads at native size)
DISPLAY_MAX_PX = 1024
#: the largest side read for an EXPORT — a figure's image is embedded at this resolution
EXPORT_MAX_PX = 4096


@dataclass
class Rendered:
    """One image item's picture and everything its overlays need."""
    image: QImage                                   # the region window, composited RGB
    plane_shape: Tuple[int, int]                    # (h, w) of the window, at the level
    level_shape: Tuple[int, int]                    # (H, W) of the whole plane, at the level
    full_shape: Tuple[int, int]                     # (Y, X) full-resolution axes
    window: Tuple[float, float, float, float]       # fy0, fy1, fx0, fx1 of the whole plane
    coords: Tuple[int, int, int, int]               # m, t, z (0 for a projection), c
    clims: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    gammas: Dict[int, float] = field(default_factory=dict)
    colors: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    planes: Dict[int, np.ndarray] = field(default_factory=dict)   # raw, the window only
    geometry: Optional[OG.FrameGeometry] = None
    settings: Optional[OV.OverlaySettings] = None
    pixel_size_um: Optional[float] = None
    note: str = ""


# ── reading planes ────────────────────────────────────────────────────────────────────
def _provider(ds: Any):
    prov = getattr(ds, "image", None)
    if prov is None or not hasattr(prov, "get_region"):
        raise ValueError("the source carries no image")
    return prov


def _pick_level(prov: Any, max_px: int) -> Tuple[int, Any]:
    """The coarsest pyramid level whose plane still has a side ≥ ``max_px`` (or level 0)."""
    levels = int(getattr(prov, "levels", 1) or 1)
    best = 0
    for lv in range(levels):
        ax = prov.level_axes(lv)
        if max(int(ax.y), int(ax.x)) >= int(max_px):
            best = lv
        else:
            break
    return best, prov.level_axes(best)


def read_plane(ds: Any, m: int, t: int, z: Any, c: int, *, max_px: int = DISPLAY_MAX_PX,
               window: Optional[Tuple[float, float, float, float]] = None
               ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    """``(plane, (H, W) of the level plane, (y0, y1, x0, x1) px of the window at the
    level)`` for channel ``c`` at ``(m, t, z)``; ``z`` may be ``max`` / ``mean`` (a
    projection over every plane). The plane is a float array of the window only."""
    prov = _provider(ds)
    fy0, fy1, fx0, fx1 = window if window else (0.0, 1.0, 0.0, 1.0)
    # the budget is for the WINDOW: a popout of a tenth of a mosaic wants a level ten
    # times finer than the whole plane would (V5.00 step 7)
    frac = max(1e-3, min(1.0, max(fy1 - fy0, fx1 - fx0)))
    level, lax = _pick_level(prov, int(math.ceil(max_px / frac)))
    H, W = int(lax.y), int(lax.x)
    y0, y1 = max(0, int(math.floor(fy0 * H))), min(H, int(math.ceil(fy1 * H)))
    x0, x1 = max(0, int(math.floor(fx0 * W))), min(W, int(math.ceil(fx1 * W)))
    if y1 <= y0:
        y1 = min(H, y0 + 1)
    if x1 <= x0:
        x1 = min(W, x0 + 1)
    nz = int(getattr(lax, "z", 1) or 1)
    if z in ("max", "mean"):
        acc = None
        for zi in range(nz):
            a = np.asarray(prov.get_region(level, m, t, zi, c, y0, y1, x0, x1), dtype=float)
            if acc is None:
                acc = a.copy()
            elif z == "max":
                np.maximum(acc, a, out=acc)
            else:
                acc += a
        if acc is None:
            acc = np.zeros((y1 - y0, x1 - x0))
        elif z == "mean" and nz > 0:
            acc /= nz
        plane = acc
    else:
        zi = min(max(int(z), 0), max(0, nz - 1))
        plane = np.asarray(prov.get_region(level, m, t, zi, c, y0, y1, x0, x1), dtype=float)
    # a provider without a pyramid (or a window far above the budget at its finest level)
    # is strided down — never interpolated — so a 500-tile montage composites 500 small
    # planes, not 500 full ones; the level shape and the box follow the stride
    longest = max(plane.shape[0], plane.shape[1]) if plane.size else 0
    if longest > 2 * int(max_px):
        k = int(math.ceil(longest / float(max_px)))
        plane = np.ascontiguousarray(plane[::k, ::k])
        H, W = int(math.ceil(H / k)), int(math.ceil(W / k))
        y0, x0 = y0 // k, x0 // k
        y1, x1 = y0 + plane.shape[0], x0 + plane.shape[1]
    return plane, (H, W), (y0, y1, x0, x1)


def _hex_rgb(s: Any, default=(255, 255, 255)) -> Tuple[int, int, int]:
    col = QColor(str(s or ""))
    return (col.red(), col.green(), col.blue()) if col.isValid() else default


def window_of(plane: np.ndarray, chan: Mapping[str, Any],
              viewer: Optional[Mapping[str, Any]] = None,
              shared: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float]:
    """``(lo, hi, gamma)`` of a channel block for ``plane``: ``auto_pct`` percentiles,
    ``manual`` the typed pair, ``viewer`` the Viewer's window when it has one (else
    auto), ``shared`` the pair a LUT-share link resolved (else auto)."""
    mode = str(chan.get("mode", "auto_pct"))
    gamma = float(chan.get("gamma", 1.0) or 1.0)
    pct = chan.get("pct") or [0.5, 99.8]
    finite = plane[np.isfinite(plane)]

    def auto() -> Tuple[float, float]:
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = (float(v) for v in np.percentile(finite, [float(pct[0]), float(pct[1])]))
        return _flat_window(lo, hi)

    if mode == "manual" and chan.get("lo") is not None and chan.get("hi") is not None:
        lo, hi = float(chan["lo"]), float(chan["hi"])
    elif mode == "viewer" and viewer and "lo" in viewer and "hi" in viewer:
        lo, hi = float(viewer["lo"]), float(viewer["hi"])
        gamma = float(viewer.get("gamma", gamma) or gamma)
    elif mode == "shared" and shared is not None:
        lo, hi = float(shared[0]), float(shared[1])
    else:
        lo, hi = auto()
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi, gamma


# ── overlay settings from the item's layers ───────────────────────────────────────────
_LAYER_GROUP = {"points": "points", "labels": "labels", "tracks": "tracks", "voxels": "voxels",
                "vectors": "vectors", "field": "field", "mesh": "mesh", "grid": "grid"}


def overlay_settings_for(layers: Sequence[Mapping[str, Any]]) -> OV.OverlaySettings:
    """An :class:`~nodelab_v2.overlays.OverlaySettings` with every group off except the
    item's layers, each group's fields taken from the layer's ``look`` by name (unknown
    names ignored), its ``layer`` / ``column`` / ``cmap`` / ``opacity`` / ``limits`` mapped
    onto the group's own fields. The field's colour bar is off — a figure has its own."""
    s = OV.OverlaySettings()
    for tab in OV.TABS:
        s.group(tab).enabled = False
    for L in layers:
        tab = _LAYER_GROUP.get(str(L.get("kind") or ""))
        if tab is None:
            continue
        grp = s.group(tab)
        grp.enabled = True
        look = dict(L.get("look") or {})
        if L.get("layer") and hasattr(grp, "layer"):
            look.setdefault("layer", str(L["layer"]))
        if L.get("column") and hasattr(grp, "column"):
            look.setdefault("column", str(L["column"]))
        cmap = L.get("cmap")
        if cmap and cmap != "auto" and hasattr(grp, "colormap"):
            look.setdefault("colormap", str(cmap))
        if "opacity" in L and hasattr(grp, "opacity"):
            look.setdefault("opacity", int(L["opacity"]))
        lim = L.get("limits") or {}
        if tab == "field":
            mode = str(lim.get("mode", "series"))
            if mode == "normalized" and lim.get("vmax"):
                look.update(limits="manual", clim_lo=0.0, clim_hi=float(lim["vmax"]),
                            center_zero=False)
            elif mode == "manual":
                look.update(limits="manual", clim_lo=float(lim.get("lo") or 0.0),
                            clim_hi=float(lim.get("hi") or 1.0))
            else:
                look.setdefault("limits", "frame" if mode == "frame" else "series")
            look["colorbar"] = False
        elif tab == "voxels":
            mode = str(lim.get("mode", "series"))
            if mode in ("manual", "normalized"):
                look.update(clim_auto=False, clim_lo=float(lim.get("lo") or 0.0),
                            clim_hi=float(lim.get("vmax") if mode == "normalized"
                                          else (lim.get("hi") or 1.0)))
        # the Viewer's look fields are screen pixels; a figure's are points — same numbers,
        # the stroke scale does the rest
        if tab == "labels":
            if "outline" in look or "fill" in look:
                outline = bool(look.pop("outline", True))
                fill = bool(look.pop("fill", False))
                look["style"] = "both" if outline and fill else ("fill" if fill else "outline")
            if "line_width" in look:
                look["width"] = float(look.pop("line_width"))
            if "ids" in look:
                look["show_ids"] = bool(look.pop("ids"))
        elif tab == "tracks":
            if "tail" in look:
                n = int(look.pop("tail"))
                look["trail"] = "window" if n > 0 else "all"
                if n > 0:
                    look["window"] = n
            if "line_width" in look:
                look["width"] = float(look.pop("line_width"))
        elif tab in ("vectors", "grid", "mesh", "voxels"):
            if "line_width" in look:
                look["width"] = float(look.pop("line_width"))
        changed = s.update_from_dict({"overlays": {tab: look}})
        _ = changed
    return s


# ── the render ────────────────────────────────────────────────────────────────────────
def render_image(ds: Any, image: Mapping[str, Any], coords: Tuple[int, int, Any, int], *,
                 window: Optional[Tuple[float, float, float, float]] = None,
                 viewer_state: Optional[Mapping[int, Mapping[str, Any]]] = None,
                 shared_clims: Optional[Mapping[int, Tuple[float, float]]] = None,
                 own_layers: Optional[Sequence[str]] = None,
                 max_px: int = DISPLAY_MAX_PX,
                 field_cache: Optional[Dict[Any, Any]] = None,
                 palette: Optional[OG.Palette] = None) -> Rendered:
    """The picture of ``image`` (an item's resolved image block) from ``ds`` at ``coords``
    ``(m, t, z | 'max' | 'mean', c)`` — ``c`` is the channel the overlays are read at.

    ``window`` is the fraction of the plane shown (the read plan's), ``viewer_state`` the
    Viewer's ``{channel: {lo, hi, gamma, rgb}}`` for ``mode = viewer`` channels,
    ``shared_clims`` the windows a LUT-share link resolved for ``mode = shared``."""
    m, t, z, c_ov = coords
    chans = list(image.get("channels") or [])
    if not chans:
        chans = [{"c": 0, "rgb": "#ffffff", "mode": "auto_pct", "pct": [0.5, 99.8],
                  "gamma": 1.0}]
    planes: Dict[int, np.ndarray] = {}
    colors: Dict[int, Tuple[int, int, int]] = {}
    clims: Dict[int, Tuple[float, float]] = {}
    gammas: Dict[int, float] = {}
    level_shape = (1, 1)
    box = (0, 1, 0, 1)
    nc = int(getattr(ds.axes, "c", 1) or 1)
    for chan in chans:
        ci = int(chan.get("c", 0))
        if not 0 <= ci < nc:
            continue
        plane, level_shape, box = read_plane(ds, m, t, z, ci, max_px=max_px, window=window)
        planes[ci] = plane
        vs = (viewer_state or {}).get(ci) or {}
        lo, hi, gamma = window_of(plane, chan, vs, (shared_clims or {}).get(ci))
        clims[ci] = (lo, hi)
        gammas[ci] = gamma
        rgb = chan.get("rgb")
        if str(chan.get("mode")) == "viewer" and vs.get("rgb"):
            rgb_t = tuple(int(v) for v in vs["rgb"])
        else:
            rgb_t = _hex_rgb(rgb)
        colors[ci] = rgb_t                                  # type: ignore[assignment]
    if not planes:
        img = QImage(1, 1, QImage.Format_RGB888)
        img.fill(Qt.black)
        return Rendered(img, (1, 1), level_shape, (int(ds.axes.y), int(ds.axes.x)),
                        window or (0.0, 1.0, 0.0, 1.0), (m, t, 0 if z in ("max", "mean")
                                                          else int(z), c_ov),
                        note="no channel of this source is drawn")
    img = composite_with_clim(planes, colors, clims, gammas)
    if bool(image.get("flip_x")) or bool(image.get("flip_y")):
        img = img.mirrored(bool(image.get("flip_x")), bool(image.get("flip_y")))
    y0, y1, x0, x1 = box
    zi = 0 if z in ("max", "mean") else int(z)
    out = Rendered(img, (y1 - y0, x1 - x0), level_shape,
                   (int(ds.axes.y), int(ds.axes.x)), window or (0.0, 1.0, 0.0, 1.0),
                   (m, t, zi, c_ov), clims, gammas, colors, planes,
                   pixel_size_um=_pixel_size(ds, level_shape))
    layers = list(image.get("layers") or [])
    if layers:
        settings = overlay_settings_for(layers)
        pal = palette if palette is not None else OG.build_palette(ds)
        out.settings = settings
        out.geometry = OG.frame_geometry(ds, (m, t, zi, c_ov), level_shape, settings, pal,
                                         own_layers=own_layers, field_cache=field_cache)
    return out


def _pixel_size(ds: Any, level_shape: Tuple[int, int]) -> Optional[float]:
    """The µm per pixel of the LEVEL plane the picture was read at."""
    md = getattr(ds, "metadata", None) or {}
    px = md.get("pixel_size_um")
    try:
        px = float(px)
    except (TypeError, ValueError):
        return None
    full_x = int(getattr(ds.axes, "x", 1) or 1)
    return px * full_x / max(1, level_shape[1])


def paint_overlays(p: QPainter, r: Rendered, target: QRectF, renderer: OV.OverlayRenderer,
                   *, stroke_scale: float, flip_x: bool = False, flip_y: bool = False) -> None:
    """Paint ``r``'s overlays into ``target`` (the image's rect in the painter's units).

    The overlay geometry is in full-resolution structure pixels; the renderer scales it by
    ``sy`` / ``sx`` into LEVEL-plane pixels and maps those through ``map_pt``, which here
    subtracts the window's offset and scales the window onto ``target``. The painter is
    clipped to ``target`` because the label fill and the scalar wash are whole-plane
    images drawn through the same mapping."""
    if r.geometry is None or r.settings is None:
        return
    H, W = r.level_shape
    ph, pw = r.plane_shape
    fy0, _fy1, fx0, _fx1 = r.window
    y_off, x_off = fy0 * H, fx0 * W
    kx, ky = target.width() / max(1, pw), target.height() / max(1, ph)

    def map_pt(px: float, py: float) -> QPointF:
        xx = (px - x_off) * kx
        yy = (py - y_off) * ky
        if flip_x:
            xx = target.width() - xx
        if flip_y:
            yy = target.height() - yy
        return QPointF(target.x() + xx, target.y() + yy)

    frame = r.geometry.frame(map_pt, (W, H), sy=H / max(1, r.full_shape[0]),
                             sx=W / max(1, r.full_shape[1]), current_t=r.coords[1],
                             stroke_scale=stroke_scale)
    p.save()
    p.setClipRect(target, Qt.IntersectClip)
    renderer.paint(p, r.settings, frame)
    p.restore()


__all__ = ["Rendered", "render_image", "read_plane", "window_of", "overlay_settings_for",
           "paint_overlays", "DISPLAY_MAX_PX", "EXPORT_MAX_PX"]
