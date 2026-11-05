"""The HTML text: one self-contained page around a view document or a figure page.

Nothing external is referenced — the stylesheet and the script are read from this package
and inlined, the planes are data URIs, there is no font link. ``</`` inside the embedded
JSON is escaped so the document can never close its own ``<script>``.
"""
from __future__ import annotations

import html as _html
import json
import os
from typing import Any, Dict, Mapping, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

FIGURE_FORMAT = "nd2studios-htmlfigure/1"


def _asset(name: str) -> str:
    with open(os.path.join(_HERE, name), encoding="utf-8") as fh:
        return fh.read()


def _json(doc: Any) -> str:
    return (json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            .replace("</", "<\\/"))


def _page(title: str, body: str, docs: Mapping[str, Any], boot: str) -> str:
    scripts = "".join(f'<script type="application/json" id="{_html.escape(k)}">{_json(v)}'
                      f"</script>\n" for k, v in docs.items())
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{_html.escape(title)}</title>\n"
            f"<style>\n{_asset('viewer.css')}\n</style>\n</head>\n"
            f"<body>\n{body}\n{scripts}"
            f"<script>\n{_asset('viewer.js')}\n</script>\n"
            f"<script>\n{boot}\n</script>\n</body>\n</html>\n")


def view_html(doc: Mapping[str, Any], *, title: Optional[str] = None) -> str:
    """A page holding one interactive viewer of ``doc`` (:func:`build_view`'s)."""
    name = title or str(doc.get("title") or "ND2Studios view")
    body = (f'<div id="nd2v-root" class="nd2v-page"><div class="nd2v-head">'
            f'<span class="nd2v-title">{_html.escape(name)}</span>'
            f'<span class="nd2v-written">{_html.escape(str(doc.get("written", "")))}</span>'
            f'</div><div id="nd2v-view"></div></div>')
    boot = ('ND2V.mount(document.getElementById("nd2v-view"), '
            'JSON.parse(document.getElementById("nd2v-doc").textContent), {});')
    return _page(name, body, {"nd2v-doc": doc}, boot)


def figure_html(fig: Mapping[str, Any], *, title: Optional[str] = None) -> str:
    """A page holding a figure: ``fig`` carries ``page`` (``width_mm``, ``height_mm``,
    ``bg``), ``svg`` (the page drawn once, the editor's own SVG export), ``views``
    (``{item id: {"rect_mm": [...], "doc": view document}}`` for every live image item),
    ``charts`` (``{item id: {"rect_mm", "geometry", "series"}}`` for the hover readouts)
    and ``links`` (the cursor groups: lists of item ids that scrub together)."""
    name = title or str(fig.get("title") or "ND2Studios figure")
    page = dict(fig.get("page") or {})
    body = (f'<div id="nd2v-root" class="nd2v-page nd2v-figure-page"><div class="nd2v-head">'
            f'<span class="nd2v-title">{_html.escape(name)}</span>'
            f'<span class="nd2v-written">{_html.escape(str(fig.get("written", "")))}</span>'
            f'<span class="nd2v-help">hover a panel for its readout · ⚙ opens a panel\'s '
            f'controls · drag / wheel zooms a live panel</span></div>'
            f'<div id="nd2v-figure" class="nd2v-figure" style="aspect-ratio:'
            f'{float(page.get("width_mm", 180.0)):g} / {float(page.get("height_mm", 120.0)):g}">'
            f'{fig.get("svg", "")}</div></div>')
    meta = {k: v for k, v in fig.items() if k != "svg"}
    meta["format"] = FIGURE_FORMAT
    boot = ('ND2V.mountFigure(document.getElementById("nd2v-figure"), '
            'JSON.parse(document.getElementById("nd2v-fig").textContent));')
    return _page(name, body, {"nd2v-fig": meta}, boot)


def write_html(path: str, text: str) -> str:
    """Write ``text`` atomically (``<stem>.part.html``, then moved into place)."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = os.path.splitext(path)[0] + ".part.html"
    with open(part, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(part, path)
    return path


__all__ = ["FIGURE_FORMAT", "view_html", "figure_html", "write_html"]
