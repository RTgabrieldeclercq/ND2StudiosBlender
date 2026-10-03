"""Export Movie's drawing primitives: channel tints, compositing, resize and burn-ins.

Qt-free and file-free. Every function here turns arrays into arrays (or draws text onto one),
so the node's export, the timeline compositor (:mod:`nodegraph.catalog._shared.movie_timeline`)
and the GUI preview all produce a frame through the same code. Moved here unchanged from
``io/write_movie.py`` on 2026-09-30, when the timeline compositor became a second caller.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


#: Named text colours for the burn-ins. A closed list rather than a colour picker: every one
#: of these is paired with the opposite-luminance shadow `_draw_text` puts behind it, so all
#: four stay legible on saturated and on black backgrounds. (`InColor` exists in the registry
#: but no catalog node uses it, so its inspector path is unexercised — not a control to
#: debut on an export node.)
_TEXT_RGB = {"white": (255, 255, 255), "black": (0, 0, 0),
             "yellow": (255, 214, 0), "cyan": (0, 229, 255)}

#: Scale-bar lengths, in µm, in the 1-2-5 sequence every figure uses. `_scalebar` picks the
#: largest one that fits the target fraction of the frame, so the bar is always a round
#: number a reader can multiply in their head — never "the bar is 73.4 µm".
_BAR_STEPS_UM = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500,
                 1000, 2000, 5000, 10000)


def _emission_rgb(nm: Any) -> Tuple[int, int, int]:
    """An approximate sRGB for a visible emission wavelength ``nm`` (Bruton's piecewise map,
    gamma 0.8) — ``None`` / out-of-visible → neutral grey.

    A verbatim, Qt-free port of :func:`nodelab_v2.theme.emission_qcolor`, and it has to STAY
    verbatim: the exported movie's whole claim is that it looks like what the Viewer showed,
    and a channel tinted 520 nm green on screen and 540 nm green in the file is a discrepancy
    nobody would think to check. Copied rather than imported because the engine is Qt-free and
    may not reach up into ``nodelab_v2`` (CLAUDE.md's seam); a copy, not one definition
    both sides share, because the GUI's copy returns a ``QColor`` and cannot move down.
    """
    try:
        w = float(nm)
    except (TypeError, ValueError):
        return (196, 200, 208)
    if not (380.0 <= w <= 780.0):
        if 780.0 < w <= 900.0:                 # near-IR still reads as deep red
            r, g, b = 0.35, 0.0, 0.0
        else:
            return (196, 200, 208)
    elif w < 440:
        r, g, b = -(w - 440) / 60.0, 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w - 440) / 50.0, 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20.0
    elif w < 580:
        r, g, b = (w - 510) / 70.0, 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, -(w - 645) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if w < 420:
        f = 0.3 + 0.7 * (w - 380) / 40.0
    elif w > 780:
        f = 1.0            # near-IR: the 0.35 red above IS the attenuation — see theme.py
    elif w > 700:
        f = 0.3 + 0.7 * (780 - w) / 80.0
    else:
        f = 1.0

    def to8(c: float) -> int:
        return 0 if c <= 0 else min(255, int(round(255 * (c * f) ** 0.8)))

    return (to8(r), to8(g), to8(b))


def _time_text(index: int, interval_s: float, total: int) -> str:
    """Elapsed experiment time at frame ``index``, formatted at a unit picked from the
    series' whole DURATION rather than from the frame.

    The unit is chosen once, from ``interval_s * (total - 1)``, and then holds for every
    frame — so the readout never changes shape mid-video, which would be unreadable as
    motion, and a 6-hour run reads ``01:15:00`` while a 90-second one reads ``12.5 s``
    instead of ``00:00:12``.

    ``interval_s`` is the MEDIAN of the file's per-frame timestamps (``ingest._dt_from_
    timestamps``), so this is an interpolation across a nominally-regular acquisition, not a
    per-frame clock: a run that stalled shows the nominal time, not the wall time. That is
    the honest reading of what the envelope carries — `dt_s` is a scalar — and it is why the
    socket is overridable.
    """
    from nodegraph.placement import elapsed_text
    t = float(interval_s) * int(index)
    span = float(interval_s) * max(0, int(total) - 1)
    return elapsed_text(t, span)       # one formatter, shared with the Viewer's timestamp


def _scalebar(width_px: int, pixel_size_um: Optional[float],
              micro: str = "um") -> Optional[Tuple[int, str]]:
    """``(bar length in px, its label)`` for a frame ``width_px`` wide — or ``None`` when the
    Dataset carries no ``pixel_size_um`` and a bar would be a fabrication.

    Targets ~18% of the frame width and then snaps DOWN to the nearest 1-2-5 value, so the
    bar is a round number; if even the smallest step overflows the frame the bar is dropped
    rather than drawn wrong. ``micro`` is the unit spelling the CHOSEN FONT can actually
    draw — :func:`_micro` — so the label never contains a glyph that renders as a box.
    """
    if not pixel_size_um or not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        return None
    target_um = 0.18 * int(width_px) * float(pixel_size_um)
    fit = [s for s in _BAR_STEPS_UM if s <= target_um]
    if not fit:
        return None
    um = fit[-1]
    px = int(round(um / float(pixel_size_um)))
    if px < 4 or px > width_px * 0.9:
        return None
    label = f"{um:g} {micro}" if um < 1000 else f"{um / 1000:g} mm"
    return (px, label)


def _font(px: int):
    """A face at ``px``, with a fallback chain that ends somewhere that always works — this
    node must not fail an export because a machine lacks a font.

    **MONOSPACE first, and that is a rendering decision rather than a taste one.** The two
    burn-ins that change every frame are the counter and the clock, and in a proportional
    face ``1/24`` and ``12/24`` are different widths, so the text jitters frame to frame.
    In a video that reads as motion — the eye tracks it, on the one element of the picture
    that is supposed to sit still. Consolas and DejaVu Sans Mono are the monospace faces
    present by default on Windows and on a Linux container respectively; the proportional
    entries after them are there only so a box with neither still exports.

    Measured here 2026-09-25: ``truetype("DejaVuSans.ttf")`` does NOT resolve by bare name on
    this installation (Pillow has no bundled ``fonts/`` directory), so the by-name pass falls
    through to Consolas and that is what a movie exported on this box is actually lettered
    in. The DejaVu entries earn their place on the Linux side, not here.

    The final fallback is Pillow's built-in bitmap font, scalable since 10.1 — ugly at size,
    and missing ``µ`` (see :func:`_micro`), but a legible frame counter beats a raised
    exception on the last step of a long export.
    """
    from PIL import ImageFont
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf",
                 "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:      # noqa: BLE001 — any miss just moves to the next candidate
            continue
    for path in (r"C:\Windows\Fonts\consola.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                 r"C:\Windows\Fonts\arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, px)
        except Exception:      # noqa: BLE001
            continue
    try:
        return ImageFont.load_default(size=px)
    except TypeError:          # Pillow < 10.1: load_default takes no size
        return ImageFont.load_default()


def _micro(font) -> str:
    """``"µm"`` if ``font`` can draw a micro sign, else ``"um"``.

    Asked of the font rather than assumed, because the answer differs across the chain in
    :func:`_font`: every TrueType face there has U+00B5 (it is in Latin-1), and Pillow's
    built-in bitmap fallback does not — it draws the .notdef box, so a scale bar on a
    font-less machine would read ``50 □m``, which looks like a corrupt file rather than like
    a missing glyph.

    The test is empirical: render U+00B5 and a codepoint the face certainly lacks (U+2588
    FULL BLOCK) and compare the masks. A font WITHOUT the micro sign draws the same .notdef
    for both, so identical masks mean "this face is faking it". Measured 2026-09-25:
    Consolas µ=(10, 13) vs notdef=(10, 22); ``load_default`` µ=(9, 13) vs notdef=(9, 13).
    """
    try:
        if font.getmask("\u00b5").size != font.getmask("\u2588").size:
            return "\u00b5m"
    except Exception:          # noqa: BLE001 — a face that cannot be probed gets the ASCII
        pass
    return "um"


def _draw_text(draw, xy: Tuple[int, int], text: str, font, rgb: Tuple[int, int, int],
               anchor: str) -> None:
    """Draw ``text`` with a one-pixel contrasting outline behind it.

    The outline is what makes a burn-in trustworthy: microscopy frames are black in some
    corners and saturated white in others, and the same frame can be both as a timelapse
    runs, so no single text colour is legible for the whole movie. The shadow is the
    opposite luminance to the text, drawn at the four diagonal offsets.
    """
    shadow = (0, 0, 0) if sum(rgb) > 380 else (255, 255, 255)
    x, y = xy
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=shadow, anchor=anchor)
    draw.text((x, y), text, font=font, fill=rgb, anchor=anchor)


def _slots(corner: str) -> Dict[str, str]:
    """Which corner each burn-in gets, given the user's choice for the frame/time block.

    Only one corner is socketed, because the others follow from it: the overlays must never
    collide, and asking for four independent corners would let the user stack two on top of
    each other. The block the user placed wins its corner; the rest fill the remaining three
    in a fixed priority (scale bar, then channel legend, then position), preferring a BOTTOM
    corner for the bar because that is where a figure convention puts it.
    """
    order = ["bottom_right", "bottom_left", "top_right", "top_left"]
    free = [c for c in order if c != corner]
    return {"text": corner, "scalebar": free[0], "channels": free[1], "position": free[2]}


def _reduce_z(stack: np.ndarray, how: str) -> np.ndarray:
    """Collapse a ``(Z, Y, X)`` stack to one ``(Y, X)`` frame."""
    if stack.shape[0] == 1:
        return stack[0]
    if how == "mid":
        return stack[stack.shape[0] // 2]
    if how == "mean":
        return stack.mean(axis=0)
    return stack.max(axis=0)


def _composite(planes: Sequence[np.ndarray], tints: Sequence[Tuple[int, int, int]],
               window: Sequence[Tuple[float, float]],
               gamma: Union[float, Sequence[float]],
               brightness: float = 100.0) -> np.ndarray:
    """Window, gamma, brightness and additively composite one frame's channels into
    ``(Y, X, 3)`` uint8.

    Additive (``maximum``-free saturation by clipping) rather than alpha-blended, because
    that is what a fluorescence merge means physically and what the Viewer draws: two
    channels that overlap read as the sum of their tints, so green + red reads yellow where
    a cell is double-positive. A single channel gets the same path with a grey tint, which
    makes the greyscale case a special case of the merge rather than a second code path.

    ``brightness`` is a PERCENT and a strictly LINEAR gain: 100 leaves the frame alone, 200
    doubles every value, 300 triples it. Applied after the window and after gamma, on the
    normalized 0-1 value, so it is the last thing to touch the picture and its effect is
    exactly "multiply what you already see". Deliberately NOT folded into the window (which
    would be the same arithmetic for a single channel and a different one for a merge, since
    the window is per channel and this is not) and not folded into gamma (which is a curve,
    not a gain). Anything that lands above 1.0 clips to white — that is what a gain does, and
    it is why the brightest structure flattens first as the value climbs.

    ``gamma`` is one number for every channel or one per channel. Per channel is what a
    Viewer-linked timeline panel hands over, since the Viewer keeps a gamma per channel.
    """
    h, w = planes[0].shape[-2:]
    acc = np.zeros((h, w, 3), np.float32)
    gain = float(brightness) / 100.0
    gammas = (list(gamma) if isinstance(gamma, (list, tuple))
              else [gamma] * len(planes))
    for plane, tint, (lo, hi), g in zip(planes, tints, window, gammas):
        a = np.asarray(plane, np.float32)
        span = float(hi) - float(lo)
        a = (a - float(lo)) / (span if span > 0 else 1.0)
        np.clip(a, 0.0, 1.0, out=a)
        if g and abs(g - 1.0) > 1e-6:
            a **= (1.0 / float(g))
        if abs(gain - 1.0) > 1e-6:
            a *= gain
        acc += a[:, :, None] * np.asarray(tint, np.float32)[None, None, :]
    np.clip(acc, 0.0, 255.0, out=acc)
    return acc.astype(np.uint8)


def _resize(rgb: np.ndarray, max_px: int) -> np.ndarray:
    """Downscale so the longest edge is at most ``max_px`` (0 = leave it alone).

    Area-averaging on the way down, which is the correct filter for a decimation and the one
    that does not alias a fine texture into moire — a real hazard here, since the things
    being exported are often speckle, nuclei or a bead field at close to the sampling limit.
    Never upscales: a movie larger than its data is a lie about the resolution.
    """
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if not max_px or longest <= int(max_px):
        return rgb
    import cv2
    f = float(max_px) / float(longest)
    return cv2.resize(rgb, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                      interpolation=cv2.INTER_AREA)


def _text_px(h: int, w: int, font_px: int) -> int:
    """The burn-in text height for an ``h`` x ``w`` frame: ``font_px``, or ~3.5% of the
    short edge when it is 0. One rule, shared by every caller that sizes text."""
    return int(font_px) if font_px else max(11, int(round(min(h, w) * 0.035)))


def _annotate(rgb: np.ndarray, *, lines: Sequence[str],
              pixel_size_um: Optional[float], ch_names: Sequence[str],
              tints: Sequence[Tuple[int, int, int]], position: str,
              show: Dict[str, bool], corner: str, font_px: int,
              text_rgb: Tuple[int, int, int]) -> np.ndarray:
    """Burn the text block, the scale bar, the channel legend and the position name into
    ``rgb`` (``(Y, X, 3)`` uint8), returning a new array.

    ``lines`` is the block the user placed with ``corner``: the frame counter, the clock,
    and whatever else the caller composes (a z readout, a title). The caller composes it
    because the caller knows which timepoint a frame shows, and in a timeline that is not the
    frame's index. ``show`` gates the other three: ``scalebar``, ``channels``, ``position``.

    ``pixel_size_um`` is the size of one OUTPUT pixel. A caller that resized the frame must
    scale the source's pixel size by the same factor, or the bar is drawn in source pixels on
    a smaller frame and is wrong by the downscale factor.

    Drawn AFTER the resize, never before, so the text is sized in output pixels and stays the
    same physical size on screen whatever the source frame was — burn text at 6554² and
    downscale and it is an illegible smear.
    """
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    size = _text_px(h, w, font_px)
    font = _font(size)
    pad = max(6, size // 2)
    slot = _slots(corner)

    def anchor_at(which: str, line: int = 0) -> Tuple[Tuple[int, int], str]:
        """Pixel position + PIL anchor for slot ``which``'s ``line``-th line of text."""
        c = slot[which]
        top = c.startswith("top")
        left = c.endswith("left")
        y = (pad + line * int(size * 1.35)) if top else (h - pad - line * int(size * 1.35))
        return ((pad if left else w - pad, y),
                ("l" if left else "r") + ("a" if top else "d"))

    # ── the text block the user placed, one line each ──────────────────────────
    lines = [str(s) for s in lines if s]
    if slot["text"].startswith("bottom"):
        lines = list(reversed(lines))          # grow upward from the bottom edge
    for i, line in enumerate(lines):
        xy, anc = anchor_at("text", i)
        _draw_text(draw, xy, line, font, text_rgb, anc)

    # ── scale bar ─────────────────────────────────────────────────────────────
    if show["scalebar"]:
        bar = _scalebar(w, pixel_size_um, _micro(font))
        if bar is not None:
            length, label = bar
            c = slot["scalebar"]
            thick = max(2, size // 5)
            x1 = (pad + length) if c.endswith("left") else (w - pad)
            x0 = x1 - length
            ty = pad if c.startswith("top") else (h - pad - int(size * 1.35))
            by = ty + int(size * 1.2)
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                draw.rectangle([x0 + dx, by + dy, x1 + dx, by + thick + dy],
                               fill=((0, 0, 0) if sum(text_rgb) > 380 else (255, 255, 255)))
            draw.rectangle([x0, by, x1, by + thick], fill=text_rgb)
            _draw_text(draw, ((x0 + x1) // 2, ty), label, font, text_rgb, "ma")

    # ── channel legend, each name in its own tint ─────────────────────────────
    if show["channels"] and len(ch_names) > 1:
        names = list(ch_names)
        if slot["channels"].startswith("bottom"):
            names = list(reversed(names))
            tints = list(reversed(list(tints)))
        for i, name in enumerate(names):
            xy, anc = anchor_at("channels", i)
            _draw_text(draw, xy, str(name), font, tuple(tints[i]), anc)

    # ── position name ─────────────────────────────────────────────────────────
    if show["position"] and position:
        xy, anc = anchor_at("position", 0)
        _draw_text(draw, xy, position, font, text_rgb, anc)

    return np.asarray(img)


def _even(rgb: np.ndarray) -> np.ndarray:
    """Pad to even width/height by repeating the last row/column.

    H.264 stores chroma at half resolution (yuv420p), so an odd edge has no representation
    and the encoder refuses the stream — measured here: a 65x99 frame fails ``isOpened()``
    outright. Edge REPLICATION rather than a black border, because a one-pixel black line on
    two sides of every frame is visible and looks like a bug in the data.
    """
    h, w = rgb.shape[:2]
    if h % 2:
        rgb = np.concatenate([rgb, rgb[-1:, :, :]], axis=0)
    if w % 2:
        rgb = np.concatenate([rgb, rgb[:, -1:, :]], axis=1)
    return rgb
