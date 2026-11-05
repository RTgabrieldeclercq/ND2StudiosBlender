"""The Figure Editor (V5.00 step 3) — the Analysis page's canvas.

An Analysis page is a figure page (:mod:`nodelab_v2.figure_document`); this package is what
shows and edits it: a journal-sized page on a :class:`QGraphicsScene` (:mod:`scene`), one
item per document item (:mod:`items`), the images rendered from the sources' pulled
Datasets (:mod:`render`), the links between items (:mod:`links`), the Sources and
Properties panels, the start card, and the panel that holds them all (:mod:`panel`). The
window hosts it through a small adapter, ``_FigureHost``, exactly as it hosts the Movie
Editor: the panel asks the host for envelopes, fetches, the Viewer's look and commits, and
never touches the runner or the document itself — which is also what lets the GUI probe
drive it headless.

Page coordinates: the scene is in **page units**, :data:`scene.MM` per millimetre. Every
stroke is a plain pen in those units, so the picture scales with the page and an SVG / PDF
export is vector where the source is vector.
"""
from __future__ import annotations

__all__ = ["scene", "items", "render", "links", "panel", "sources", "properties",
           "start_card"]
