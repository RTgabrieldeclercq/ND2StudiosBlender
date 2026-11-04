"""Page layout for a figure document — presets, letters, snapping, alignment, and the
paper-readiness checks (V5.00 step 1).

Everything is in millimetres on the page. The checks encode the community guidance the
design record cites (Jambor et al. 2021; Schmied et al. 2023): a scale bar on every image,
an inset's box marked on its parent, grey for a single channel, no red/green merge, one
display range across compared panels, a labelled colour bar, legible type, panel letters
in order, a time stamp where the montage axis is time, vector or ≥ 300 dpi output. A check
is a ``dict`` with ``severity`` (``warn`` | ``hint``), ``code``, ``item`` (an item id or
``""`` for the page) and ``message``; it never blocks a render — the page is the user's.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Page presets, mm. Journal column widths differ by a few mm between publishers; these are
#: the common middles, and the user can type any size.
PAGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "single_column": {"width_mm": 85.0, "height_mm": 65.0,
                      "doc": "One journal column (85 mm wide)."},
    "column_1_5": {"width_mm": 114.0, "height_mm": 85.0,
                   "doc": "One and a half columns (114 mm wide)."},
    "double_column": {"width_mm": 180.0, "height_mm": 120.0,
                      "doc": "Two journal columns (180 mm wide)."},
    "a4_portrait": {"width_mm": 210.0, "height_mm": 297.0, "doc": "A4, portrait."},
    "a4_landscape": {"width_mm": 297.0, "height_mm": 210.0, "doc": "A4, landscape."},
    "slide_16_9": {"width_mm": 338.7, "height_mm": 190.5,
                   "doc": "A 16:9 slide (13.33 × 7.5 in)."},
    "poster": {"width_mm": 250.0, "height_mm": 180.0, "doc": "A poster panel."},
}

#: Panel letter styles: how ``letters(n, style)`` spells the k-th letter.
LETTER_STYLES = ("A", "a", "(A)", "A.", "none")

PT_MM = 25.4 / 72.0          #: one point in millimetres

Rect = Tuple[float, float, float, float]      # x, y, w, h in mm


def letters(n: int, style: str = "A") -> List[str]:
    """The first ``n`` panel letters in ``style`` (``A`` → ``A, B, …``; ``a``; ``(A)``;
    ``A.``; ``none`` → empty strings). Past Z the letters double (``AA``, ``AB``)."""
    if style not in LETTER_STYLES:
        raise ValueError(f"letters: unknown style {style!r}; one of {LETTER_STYLES}")
    out: List[str] = []
    for k in range(int(n)):
        base = ""
        q = k
        while True:
            base = chr(ord("A") + q % 26) + base
            q = q // 26 - 1
            if q < 0:
                break
        if style == "none":
            out.append("")
        elif style == "a":
            out.append(base.lower())
        elif style == "(A)":
            out.append(f"({base})")
        elif style == "A.":
            out.append(f"{base}.")
        else:
            out.append(base)
    return out


def snap_mm(value: float, grid_mm: float) -> float:
    """``value`` snapped to the nearest multiple of ``grid_mm`` (no snapping at 0)."""
    g = float(grid_mm or 0.0)
    if g <= 0:
        return float(value)
    return round(float(value) / g) * g


def align(rects: Sequence[Rect], mode: str) -> List[Rect]:
    """Move ``rects`` so they share an edge or a centre: ``left`` | ``right`` | ``top`` |
    ``bottom`` | ``center_x`` | ``center_y``. Sizes are kept; the reference is the extreme
    (or mean centre) of the group, so nothing moves off the page it was on."""
    rs = [tuple(float(v) for v in r) for r in rects]
    if not rs:
        return []
    if mode == "left":
        ref = min(r[0] for r in rs)
        return [(ref, r[1], r[2], r[3]) for r in rs]
    if mode == "right":
        ref = max(r[0] + r[2] for r in rs)
        return [(ref - r[2], r[1], r[2], r[3]) for r in rs]
    if mode == "top":
        ref = min(r[1] for r in rs)
        return [(r[0], ref, r[2], r[3]) for r in rs]
    if mode == "bottom":
        ref = max(r[1] + r[3] for r in rs)
        return [(r[0], ref - r[3], r[2], r[3]) for r in rs]
    if mode == "center_x":
        ref = sum(r[0] + r[2] / 2 for r in rs) / len(rs)
        return [(ref - r[2] / 2, r[1], r[2], r[3]) for r in rs]
    if mode == "center_y":
        ref = sum(r[1] + r[3] / 2 for r in rs) / len(rs)
        return [(r[0], ref - r[3] / 2, r[2], r[3]) for r in rs]
    raise ValueError(f"align: unknown mode {mode!r}")


def distribute(rects: Sequence[Rect], axis: str, gap_mm: Optional[float] = None
               ) -> List[Rect]:
    """Space ``rects`` evenly along ``axis`` (``x`` | ``y``), in their current order along
    that axis. With ``gap_mm`` the gap is fixed and the group grows from its first rect;
    without, the first and last stay and the gaps between are made equal."""
    rs = [tuple(float(v) for v in r) for r in rects]
    if len(rs) < 2:
        return list(rs)
    i, j = (0, 2) if axis == "x" else (1, 3)
    order = sorted(range(len(rs)), key=lambda k: rs[k][i])
    out = list(rs)
    if gap_mm is not None:
        pos = rs[order[0]][i]
        for k in order:
            r = list(rs[k])
            r[i] = pos
            out[k] = tuple(r)                       # type: ignore[assignment]
            pos += r[j] + float(gap_mm)
        return out
    first, last = rs[order[0]], rs[order[-1]]
    span = (last[i] + last[j]) - first[i]
    total = sum(rs[k][j] for k in order)
    gap = (span - total) / (len(rs) - 1)
    pos = first[i]
    for k in order:
        r = list(rs[k])
        r[i] = pos
        out[k] = tuple(r)                           # type: ignore[assignment]
        pos += r[j] + gap
    return out


# ── paper-readiness checks ───────────────────────────────────────────────────────────
def _hex_rgb(s: Any) -> Optional[Tuple[int, int, int]]:
    t = str(s or "").strip().lstrip("#")
    if len(t) != 6:
        return None
    try:
        return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
    except ValueError:
        return None


def _is_grey(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return max(r, g, b) - min(r, g, b) <= 24


def _is_reddish(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return r > 150 and g < 110 and b < 110


def _is_greenish(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return g > 150 and r < 110 and b < 110


def page_checks(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The checks the community checklists would raise on ``doc`` (a normalized figure
    document), most severe first. See the module docstring for the list."""
    out: List[Dict[str, Any]] = []

    def add(sev: str, code: str, item: str, msg: str) -> None:
        out.append({"severity": sev, "code": code, "item": item, "message": msg})

    page = doc.get("page") or {}
    items = list(doc.get("items") or [])
    links = list(doc.get("links") or [])
    by_id = {it.get("id"): it for it in items}

    # type size and resolution
    if float(page.get("font_pt") or 0) < 6.0:
        add("warn", "font_pt", "", f"type is {page.get('font_pt')} pt; journals ask for "
                                   f"6 pt or larger at page size")
    if int(page.get("dpi") or 0) < 300:
        add("hint", "dpi", "", f"raster export at {page.get('dpi')} dpi; use 300 or more, "
                               f"or export SVG / PDF")

    # letters in order, no duplicates
    seen: Dict[str, str] = {}
    lettered = [(it.get("letter") or "", it.get("id") or "") for it in items
                if it.get("letter")]
    for letter, iid in lettered:
        if letter in seen:
            add("warn", "letter_dup", iid, f"panel letter {letter!r} is also on {seen[letter]}")
        seen[letter] = iid
    expect = letters(len(lettered), str(page.get("letters") or "A")) if lettered else []
    got = sorted({l for l, _ in lettered}, key=lambda s: (len(s), s))
    if lettered and got != sorted(set(expect), key=lambda s: (len(s), s)):
        add("hint", "letter_order", "", f"panel letters are {got}; expected {expect}")

    zoom_children = {ln.get("child") for ln in links if ln.get("kind") == "zoom"}
    standalone_bar = any(it.get("kind") == "scale_bar" for it in items)
    twins: List[Tuple[str, str]] = [(ln.get("a"), ln.get("b")) for ln in links
                                    if ln.get("kind") == "twin"]
    shared: List[set] = [set(ln.get("items") or []) for ln in links
                         if ln.get("kind") == "lut_share"]

    for it in items:
        kind = it.get("kind")
        iid = it.get("id") or ""
        if kind in ("image", "montage"):
            img = it.get("image") or {}
            decor = img.get("decor") or {}
            if not float(decor.get("scale_bar_um") or 0) and not standalone_bar:
                add("warn", "scale_bar", iid, "no scale bar on this image")
            region = (img.get("region") or {}).get("from", "full")
            if region != "full" and iid not in zoom_children:
                add("warn", "inset_box", iid,
                    "this panel shows a region of a larger field but no parent marks its "
                    "box — add a zoom link")
            chans = [c for c in (img.get("channels") or [])]
            rgbs = [_hex_rgb(c.get("rgb")) for c in chans]
            rgbs = [c for c in rgbs if c is not None]
            if len(rgbs) == 1 and not _is_grey(rgbs[0]):
                add("hint", "single_channel_grey", iid,
                    "a single channel reads best in grey (or inverted grey)")
            if len(rgbs) >= 2 and any(_is_reddish(c) for c in rgbs) \
                    and any(_is_greenish(c) for c in rgbs):
                add("warn", "red_green", iid,
                    "a red / green merge is unreadable for ~8 % of readers; use "
                    "magenta / green or cyan / magenta / yellow")
            if kind == "montage":
                mt = it.get("montage") or {}
                if str(mt.get("tile_by")) == "t" and not decor.get("time_stamp"):
                    add("hint", "time_stamp", iid, "a montage over time has no time stamp")
                if str(mt.get("tile_by")) == "z" and not decor.get("z_label"):
                    add("hint", "z_label", iid, "a montage over Z has no depth label")
        elif kind == "colorbar":
            if not str(it.get("label") or "").strip():
                add("warn", "colorbar_label", iid, "the colour bar names no quantity")
        elif kind == "chart":
            ch = it.get("chart") or {}
            axes = ch.get("axes") or {}
            if ch.get("kind") not in ("heatmap", "hexbin", "rose") and \
                    not (str(axes.get("x_label") or "").strip()
                         and str(axes.get("y_label") or "").strip()):
                add("hint", "axis_labels", iid, "an axis has no label")

    for a, b in twins:
        if not any(a in s and b in s for s in shared):
            add("hint", "lut_share", a or "",
                f"twins {a} and {b} do not share a display range; compared panels should")
    sev = {"warn": 0, "hint": 1}
    out.sort(key=lambda c: (sev.get(c["severity"], 2), c["code"], c["item"]))
    return out


# ── step 5: style presets, re-lettering, style sync, palette validation ──────────────
#: the page style presets: what *Style sync* applies to the page and to every item's
#: own override. Points for type, points for lines and markers.
STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "paper": {"font_pt": 8.0, "line_width": 1.0, "marker_size": 3.0, "dark": False,
              "doc": "Journal figure: 8 pt type, 1 pt lines."},
    "talk": {"font_pt": 14.0, "line_width": 1.5, "marker_size": 4.5, "dark": False,
             "doc": "A slide: 14 pt type, heavier lines."},
    "poster": {"font_pt": 18.0, "line_width": 2.0, "marker_size": 6.0, "dark": False,
               "doc": "A poster panel: 18 pt type, 2 pt lines."},
    "dark": {"font_pt": 8.0, "line_width": 1.0, "marker_size": 3.0, "dark": True,
             "doc": "A dark page (talks); the chart palette avoids black."},
}

LIGHT_PAGE = "#ffffff"        #: the page colour of a light figure
DARK_PAGE = "#1c1f24"         #: the page colour of a dark figure (the scene paints this)
DARK_INK = "#e6e6e6"          #: ink on a dark page
LIGHT_INK = "#111111"         #: ink on a light page

#: the item kinds that are PANELS — lettered, in reading order
PANEL_KINDS = ("image", "montage", "picture", "chart", "table", "kymograph", "ortho",
               "profile")


def reading_order(items: Sequence[Mapping[str, Any]]) -> List[str]:
    """Item ids in reading order: rows top to bottom, left to right within a row. An item
    joins the row above when its top is above that row's lowest bottom by more than half
    its own height (panels of one row rarely share an exact top)."""
    recs = []
    for it in items:
        x, y, w, h = (float(v) for v in it.get("rect_mm") or (0, 0, 0, 0))
        recs.append((y, x, h, str(it.get("id"))))
    recs.sort()
    rows: List[List[Tuple[float, float, float, str]]] = []
    bottoms: List[float] = []
    for y, x, h, iid in recs:
        if rows and y < bottoms[-1] - 0.5 * h:
            rows[-1].append((y, x, h, iid))
            bottoms[-1] = max(bottoms[-1], y + h)
        else:
            rows.append([(y, x, h, iid)])
            bottoms.append(y + h)
    out: List[str] = []
    for row in rows:
        out.extend(iid for _y, x, _h, iid in sorted(row, key=lambda r: (r[1], r[0])))
    return out


def reletter(doc: Mapping[str, Any]) -> Dict[str, str]:
    """``{item id: letter}`` for every panel of ``doc`` in reading order, in the page's
    letter style — a popout (the child of a zoom link) is part of its parent's panel and
    gets none; a non-panel item keeps whatever it had. A blank letter means *clear*."""
    page = doc.get("page") or {}
    items = list(doc.get("items") or [])
    children = {str(ln.get("child")) for ln in doc.get("links") or []
                if ln.get("kind") == "zoom"}
    panels = [it for it in items if it.get("kind") in PANEL_KINDS
              and str(it.get("id")) not in children]
    order = reading_order(panels)
    style = str(page.get("letters") or "A")
    labels = letters(len(order), style) if style != "none" else [""] * len(order)
    out = dict(zip(order, labels))
    for iid in children:
        out[iid] = ""
    return out


def style_sync(doc: Mapping[str, Any], style: Optional[str] = None) -> Dict[str, Any]:
    """A copy of ``doc`` with the page preset ``style`` (default: the page's own) applied
    everywhere: the page's type size, line width and dark flag, and every item's own
    override cleared back to the page — text and letters to the page size, shapes, lines
    and arrows to the page line width, every overlay layer's line to it. What *Style sync*
    does; the user restyles single items afterwards."""
    import copy as _copy
    d = _copy.deepcopy(dict(doc))
    page = d.setdefault("page", {})
    name = str(style or page.get("style") or "paper")
    preset = STYLE_PRESETS.get(name, STYLE_PRESETS["paper"])
    page["style"] = name if name in STYLE_PRESETS else "paper"
    page["font_pt"] = float(preset["font_pt"])
    page["line_width"] = float(preset["line_width"])
    page["marker_size"] = float(preset["marker_size"])
    page["dark"] = bool(preset["dark"])
    lw = float(preset["line_width"])
    for it in d.get("items") or []:
        kind = it.get("kind")
        if kind in ("text", "letter", "table", "legend"):
            it["pt"] = None
        elif kind in ("shape", "line", "arrow", "measure_line"):
            it["width_pt"] = lw
        elif kind in ("image", "montage"):
            for L in (it.get("image") or {}).get("layers") or []:
                look = L.setdefault("look", {}) if isinstance(L.get("look"), dict) or \
                    L.get("look") is None else L["look"]
                if isinstance(look, dict) and "line_width" in look:
                    look["line_width"] = lw
    for ln in d.get("links") or []:
        if ln.get("kind") == "zoom" and isinstance(ln.get("box"), dict):
            ln["box"]["width_pt"] = max(0.5, lw * 0.75)
    return d


# ── colour: sRGB → CIE Lab, ΔE76, and the palette validator ───────────────────────
def _srgb_to_lab(hex_color: str) -> Tuple[float, float, float]:
    rgb = _hex_rgb(hex_color)
    if rgb is None:
        return (0.0, 0.0, 0.0)

    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(float(v)) for v in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.00000
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0
    fx, fy, fz = f(x), f(y), f(z)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def delta_e(a: str, b: str) -> float:
    """CIE76 colour difference between two ``#rrggbb`` colours (≈ 2.3 is a just-noticeable
    difference; series colours want 15 or more, and more against the page)."""
    la, lb = _srgb_to_lab(a), _srgb_to_lab(b)
    return float(sum((p - q) ** 2 for p, q in zip(la, lb)) ** 0.5)


#: two series colours must differ by at least this ΔE; a series colour and the page by
#: :data:`PALETTE_MIN_PAGE_DELTA_E` (a mark must stand off the paper, not merely differ)
PALETTE_MIN_DELTA_E = 12.0
PALETTE_MIN_PAGE_DELTA_E = 25.0


def palette_report(colors: Sequence[str], *, background: str = LIGHT_PAGE,
                   min_delta: float = PALETTE_MIN_DELTA_E,
                   min_page_delta: float = PALETTE_MIN_PAGE_DELTA_E) -> List[str]:
    """The problems of a categorical palette on a page of colour ``background``: a pair
    of colours too alike to tell apart, a colour too close to the page to see. Empty
    when the palette passes."""
    out: List[str] = []
    cols = [str(c) for c in colors]
    for i, a in enumerate(cols):
        d = delta_e(a, background)
        if d < min_page_delta:
            out.append(f"{a} is within ΔE {d:.1f} of the page {background}")
        for b in cols[i + 1:]:
            d = delta_e(a, b)
            if d < min_delta:
                out.append(f"{a} and {b} are within ΔE {d:.1f} of each other")
    return out


def _lab_to_hex(L: float, a: float, b: float) -> str:
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t: float) -> float:
        return t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116.0) / 7.787
    x, y, z = finv(fx) * 0.95047, finv(fy) * 1.00000, finv(fz) * 1.08883
    r = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    bl = 0.0557 * x - 0.2040 * y + 1.0570 * z

    def gam(c: float) -> int:
        c = max(0.0, min(1.0, c))
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return int(round(max(0.0, min(1.0, c)) * 255))
    return "#%02x%02x%02x" % (gam(r), gam(g), gam(bl))


def effective_palette(colors: Sequence[str], *, dark: bool,
                      min_page_delta: float = PALETTE_MIN_PAGE_DELTA_E) -> List[str]:
    """The palette as drawn on a light or dark page. A grey ramp is re-spaced in lightness
    to the page (black … mid-grey on white; white … mid-grey on a dark page), so every
    step stays a step; a chromatic colour the page would swallow (black on a dark page)
    keeps its hue and is lifted (or lowered) to a legible lightness. Order is kept, so a
    series keeps its colour index across pages of both kinds."""
    page = DARK_PAGE if dark else LIGHT_PAGE
    cols = [str(c) for c in colors]
    labs = [_srgb_to_lab(c) for c in cols]
    n = len(cols)
    if n > 1 and all(abs(a) < 3.0 and abs(b) < 3.0 for _L, a, b in labs):
        lo, hi = (100.0, 40.0) if dark else (0.0, 72.0)
        return [_lab_to_hex(lo + (hi - lo) * k / (n - 1), 0.0, 0.0) for k in range(n)]
    out: List[str] = []
    for c, (L, a, b) in zip(cols, labs):
        if delta_e(c, page) < min_page_delta:
            out.append(_lab_to_hex(45.0 if dark else 60.0, a, b))
        else:
            out.append(c)
    return out
