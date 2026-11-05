"""The page fonts (V5.00 step 6): DejaVu Sans / Serif registered from matplotlib's copy.

A figure is set in DejaVu (the family :func:`nodelab_v2.figure_editor.items.page_font`
asks for) so that the editor on screen, the offscreen renderer and a matplotlib-drawn plot
node share one face and one set of metrics. Few machines have it installed; matplotlib —
already a dependency of the chart ticks — ships the TTFs, so they are registered with Qt
once per process from there, and the system fonts are left as the fallback.
"""
from __future__ import annotations

import os
from typing import List

_DONE: List[str] = []

_FILES = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
          "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf")


def ensure_page_fonts() -> List[str]:
    """Register the page fonts with the running Qt application (once). Returns the files
    registered; empty when matplotlib's fonts are not found or no application exists."""
    if _DONE:
        return list(_DONE)
    try:
        from PySide6.QtGui import QFontDatabase, QGuiApplication
    except Exception:                      # noqa: BLE001 — no Qt: nothing to register into
        return []
    if QGuiApplication.instance() is None:
        return []
    try:
        import matplotlib
        folder = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    except Exception:                      # noqa: BLE001 — no matplotlib: system fonts
        return []
    for name in _FILES:
        path = os.path.join(folder, name)
        if os.path.exists(path) and QFontDatabase.addApplicationFont(path) >= 0:
            _DONE.append(path)
    return list(_DONE)


__all__ = ["ensure_page_fonts"]
