"""The interactive preview (2026-11-05): the page as its HTML export, inside the workspace.

The Figure Editor's **Preview** button builds the page's interactive HTML
(:meth:`~nodelab_v2.figure_editor.panel.FigureEditorPanel.export_html`, at a preview
budget) and shows it here, in the place of the page — a QtWebEngine view, so what you
scrub and re-window is the very page a reader will get. The bar above it says when and at
what budget it was built, goes stale when the page is edited under it (↻ rebuilds), opens
the same file in the system browser, saves it as the export, or goes back to the page.

QtWebEngine is optional: without it (or with ``NODELAB_WEBVIEW=0``) :meth:`show_file`
answers ``False`` and the editor opens the page in the browser instead. The import is
deferred to the first preview, so a window that never previews never loads Chromium.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget,
                               QToolButton, QVBoxLayout, QWidget)

from nodelab_v2 import theme as T

_import_ok: Optional[bool] = None


def webengine_available() -> bool:
    """Is a QtWebEngine view possible here? ``NODELAB_WEBVIEW=0`` says no without trying;
    the import's outcome is remembered."""
    global _import_ok
    if os.environ.get("NODELAB_WEBVIEW", "1").lower() in ("0", "false", "no"):
        return False
    if _import_ok is None:
        try:
            from PySide6 import QtWebEngineWidgets  # noqa: F401 — the import is the test
            _import_ok = True
        except Exception:               # noqa: BLE001 — any failure means "no web view"
            _import_ok = False
    return bool(_import_ok)


class HtmlPreview(QWidget):
    """See the module docstring."""

    #: the bar's buttons
    back_requested = Signal()
    rebuild_requested = Signal()
    open_requested = Signal()
    save_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("htmlPreview")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._web: Any = None
        self.path: str = ""
        self.stale = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._bar = QWidget()                 # its width may be ignored: see panel.py's bar
        self._bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar = QHBoxLayout(self._bar)
        bar.setContentsMargins(8, 4, 8, 4)
        bar.setSpacing(6)
        self.title = QLabel("Interactive preview")
        self.title.setProperty("role", "title")
        bar.addWidget(self.title)
        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.status, 1)
        self.rebuild_btn = QToolButton()
        self.rebuild_btn.setText("↻ Rebuild")
        self.rebuild_btn.setToolTip("Build the page again from the figure as it is now")
        self.rebuild_btn.clicked.connect(self.rebuild_requested)
        bar.addWidget(self.rebuild_btn)
        self.open_btn = QToolButton()
        self.open_btn.setText("Open in browser")
        self.open_btn.setToolTip("The same file in the system browser")
        self.open_btn.clicked.connect(self.open_requested)
        bar.addWidget(self.open_btn)
        self.save_btn = QPushButton("Save as…")
        self.save_btn.setToolTip("Write the Interactive HTML export — the page at its full "
                                 "resolution — to a file of your choosing")
        self.save_btn.clicked.connect(self.save_requested)
        bar.addWidget(self.save_btn)
        self.back_btn = QToolButton()
        self.back_btn.setText("✕ Back to the page")
        self.back_btn.clicked.connect(self.back_requested)
        bar.addWidget(self.back_btn)
        lay.addWidget(self._bar)
        self.stack = QStackedWidget()
        self.placeholder = QLabel("The interactive page appears here once it is built.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setProperty("role", "muted")
        self.stack.addWidget(self.placeholder)
        lay.addWidget(self.stack, 1)
        self.restyle()

    # ── the web view ───────────────────────────────────────────────────────────
    def _ensure_web(self) -> bool:
        if self._web is not None:
            return True
        if not webengine_available():
            return False
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            web = QWebEngineView()
        except Exception:               # noqa: BLE001 — a view that cannot be made is "none"
            return False
        self._web = web
        self.stack.addWidget(web)
        return True

    def show_file(self, path: str) -> bool:
        """Show the page at ``path``. ``False`` when no web view can be made here."""
        if not self._ensure_web():
            return False
        self.path = str(path)
        self.stale = False
        self._web.load(QUrl.fromLocalFile(os.path.abspath(self.path)))
        self.stack.setCurrentWidget(self._web)
        return True

    def web(self) -> Any:
        return self._web

    def clear(self) -> None:
        if self._web is not None:
            self._web.setUrl(QUrl("about:blank"))
        self.stack.setCurrentWidget(self.placeholder)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_stale(self, on: bool) -> None:
        """The page was edited since the preview was built."""
        on = bool(on)
        if on == self.stale:
            return
        self.stale = on
        self.rebuild_btn.setProperty("stale", on)
        self.rebuild_btn.style().unpolish(self.rebuild_btn)
        self.rebuild_btn.style().polish(self.rebuild_btn)
        if on:
            self.status.setText("the page changed since this preview was built — ↻ Rebuild")

    def restyle(self) -> None:
        self.setStyleSheet(f"""
        QWidget#htmlPreview {{ background:{T.BG.name()}; }}
        QLabel[role="title"] {{ color:{T.INK.name()}; font-weight:800; font-size:12px; }}
        QLabel[role="muted"] {{ color:{T.MUTED.name()}; font-size:11px; }}
        QToolButton {{ background:{T.BODY.name()}; color:{T.INK.name()};
            border:1px solid {T.BORDER.name()}; border-radius:6px; padding:3px 9px; }}
        QToolButton:hover {{ background:{T.PANEL_HI.name()}; border-color:{T.BORDER_HI.name()}; }}
        QToolButton[stale="true"] {{ border-color:{T.ACCENT.name()}; color:{T.ACCENT.name()}; }}
        """ + T.controls_qss())


__all__ = ["HtmlPreview", "webengine_available"]
