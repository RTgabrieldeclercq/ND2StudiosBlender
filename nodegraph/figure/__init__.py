"""nodegraph.figure — the Qt-free core of the Figure Editor (V5.00 step 1).

An Analysis page is a **figure document** (:mod:`schema`): a journal-sized page of items —
images, montages, zoom popouts, heat and vector layers, charts, colour bars, histograms,
scale bars, annotations — fed by the named Page Outputs of the upstream pages, and of
links between items (a popout's box on its parent, twins, a shared display range, a shared
cursor). This package holds everything about such a document that is *not painting*:

* :mod:`schema` — defaults, validation (every error names its spec path), canonical JSON,
  the digest the memo and the render cache key on, twin inheritance;
* :mod:`bind` — what each item must READ: the frame a cursor resolves to, the window of a
  source plane a region covers, a popout's box on its parent from the two fields' stage
  metadata (:mod:`nodegraph.placement`, the maths ``view.overlay`` already uses);
* :mod:`derive` — arrays derived from a pulled Dataset for representations no node makes:
  orthogonal reslices, depth- and time-coded projections, kymographs, intensity profiles,
  mean squared displacement, survival, colocalisation statistics;
* :mod:`series` — a chart's data, the SAME helpers the ``plot.*`` nodes draw from, plus the
  kinds they do not have (bars, strips, SuperPlots, roses, survival, hexbins, MSD);
* :mod:`stats` — group comparisons, correlations, stars and bracket layout;
* :mod:`charts` — a chart's GEOMETRY in millimetres (axes, ticks, marks, legend), computed
  once so the GUI item and the offscreen renderer paint the same lines;
* :mod:`layout` — page presets, panel letters, snapping, alignment, the paper-readiness
  checks the community checklists ask for;
* :mod:`legend_text` — the legend / methods sentence generated from the acquisition record.

Nothing here imports Qt or matplotlib's drawing: tick placement uses ``matplotlib.ticker``
(pure Python). The design record is ``CodeLog/ClaudesPlan/V5.00_figure_editor.md``.
"""
from __future__ import annotations

__all__ = ["schema", "bind", "derive", "series", "stats", "charts", "layout",
           "legend_text"]
