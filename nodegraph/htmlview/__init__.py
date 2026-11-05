"""Standalone interactive HTML views of a result (2026-11-05) — Qt-free.

One file a browser opens with no server and no network: the frames of a result embedded as
packed PNGs, the Viewer's controls rebuilt in JavaScript — the M / T / Z strips with their
play buttons, a toggle and a histogram LUT per channel with draggable handles and a gamma
dot, Auto / Fit / Split, the overlays popup, the hover readout, the scale bar and the
timestamp — and, for a figure page, the page itself with its image items live.

* :mod:`document` builds the view document (:func:`build_view`) from a Dataset and the
  display state a caller holds;
* :mod:`planes` reads planes at a pixel budget and quantises them, :mod:`png` packs them;
* :mod:`page` turns a document (or a figure page) into the HTML text, embedding
  ``viewer.js`` and ``viewer.css`` from this package.

Callers: the ``io.write_html`` node (Export HTML), the Viewer's **→ HTML** button, the
figure editor's **Export ▾ → Interactive HTML**, and the headless figure renderer.
"""
from nodegraph.htmlview.document import (FORMAT, BudgetExceeded, build_view,  # noqa: F401
                                         parse_channels, parse_frames)
from nodegraph.htmlview.page import figure_html, view_html, write_html  # noqa: F401

__all__ = ["FORMAT", "BudgetExceeded", "build_view", "parse_channels", "parse_frames",
           "view_html", "figure_html", "write_html"]
