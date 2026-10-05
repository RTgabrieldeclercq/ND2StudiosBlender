"""ND2Studios version — the one place the app's name and version number are written.

V4.00 "Workspaces" (opened 2026-10-05): typed node-graph pages, named cross-page flow, a
dockable multi-panel shell and an analysis toolkit. The design record and step list are in
``CodeLog/ClaudesPlan/V4.00_workspaces.md``. The ``nodelab_v2`` package keeps its historical
name: the version is a label, not a module path.

Qt-free. The engine side of the seam must not import it (``nodegraph`` never imports
``nodelab_v2``), so the graph-file ``format_version`` stays in :mod:`nodegraph.serialize`;
only the ``app_version`` stamp written into saved files is read from here.
"""
from __future__ import annotations

__all__ = ["APP_NAME", "MAJOR", "PRODUCT", "VERSION_LABEL", "__version__"]

APP_NAME = "ND2Studios"
__version__ = "4.0.0"
MAJOR = int(__version__.split(".")[0])
#: Short human label, ``"ND2Studios V4"`` — banners, the LabLink hello, the manual.
VERSION_LABEL = f"{APP_NAME} V{MAJOR}"
#: The window-title product name, ``"ND2Studios V4 — NodeLab"``.
PRODUCT = f"{VERSION_LABEL} — NodeLab"
