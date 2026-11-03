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
