"""The figure's side panels as the window's docks (2026-11-05): the Analysis workspace.

On a graph page the window's left dock is the Nodes palette and its right dock a node's
Properties. On a figure page the same two docks turn their other face: the left shows the
figure's **Sources** (what the page can read), the right the **Figure widgets** palette
over the selected item's properties — so the page itself owns the whole centre, and the
Viewer docks above it step aside. Three small widgets do that:

* :class:`ModeStack` — one dock, two faces (a ``QStackedWidget`` of the graph page's panel
  and the figure page's). The window keeps its ``inspector`` and ``palette`` objects; they
  are simply the first page of a stack now, so ``shell.dock_of(window.palette)`` still
  resolves (it walks ancestors).
* :class:`SourcesHost` — the ACTIVE figure editor's Sources panel. Every canvas has its own
  Figure Editor, so every editor has its own Sources and Properties panels; the hosts show
  the active canvas's and keep the others parked, and forget an editor whose canvas closed.
* :class:`FigureSidePanel` — the widgets palette above a host of the editors' Properties
  forms, in a vertical splitter.

Nothing here edits the figure; the panels hosted do, through their editor.
"""
from __future__ import annotations

import weakref
from typing import Any, Iterable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QLabel, QScrollArea, QSplitter, QStackedWidget,
                               QVBoxLayout, QWidget)

from nodelab_v2 import theme as T
from nodelab_v2.figure_editor.widgets import FigureWidgetsPanel


class ModeStack(QStackedWidget):
    """One dock, two faces: ``node_widget`` on a graph page, ``figure_widget`` on a figure
    page."""

    def __init__(self, node_widget: QWidget, figure_widget: QWidget) -> None:
        super().__init__()
        self.node_widget = node_widget
        self.figure_widget = figure_widget
        self.addWidget(node_widget)
        self.addWidget(figure_widget)
        self.setCurrentWidget(node_widget)

    def set_figure(self, on: bool) -> None:
        self.setCurrentWidget(self.figure_widget if on else self.node_widget)

    def is_figure(self) -> bool:
        return self.currentWidget() is self.figure_widget

    def restyle(self) -> None:
        for w in (self.node_widget, self.figure_widget):
            if hasattr(w, "restyle"):
                w.restyle()


class _Host(QWidget):
    """A stack of one panel per figure editor, the active editor's on top, a placeholder
    when no figure page is shown. ``panel_of(ed)`` names the editor's panel to host."""

    placeholder_text = "No figure page is shown."

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.stack = QStackedWidget()
        self.placeholder = QLabel(self.placeholder_text)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setProperty("role", "muted")
        self.stack.addWidget(self.placeholder)
        lay.addWidget(self.stack)
        self._editors: List[Any] = []            # weakrefs to the editors hosted

    def panel_of(self, ed: Any) -> QWidget:      # pragma: no cover — abstract
        raise NotImplementedError

    def show_editor(self, ed: Any) -> None:
        """Host (once) and show ``ed``'s panel."""
        w = self.panel_of(ed)
        if self.stack.indexOf(w) < 0:
            self.stack.addWidget(w)
            self._editors.append(weakref.ref(ed))
        self.stack.setCurrentWidget(w)

    def current(self) -> Optional[QWidget]:
        w = self.stack.currentWidget()
        return None if w is self.placeholder else w

    def forget_dead(self, alive: Iterable[Any]) -> None:
        """Drop the panels of editors not in ``alive`` (their canvas closed): the panel was
        re-parented here and would outlive its editor, holding a dead reference."""
        keep = set()
        for ed in alive:
            try:
                keep.add(self.panel_of(ed))
            except RuntimeError:            # the editor is being destroyed
                pass
        for i in reversed(range(self.stack.count())):
            w = self.stack.widget(i)
            if w is self.placeholder or w in keep:
                continue
            self.stack.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._editors = [r for r in self._editors if r() is not None and r() in set(alive)]
        if self.stack.currentWidget() is None:
            self.stack.setCurrentWidget(self.placeholder)

    def show_placeholder(self) -> None:
        self.stack.setCurrentWidget(self.placeholder)

    def restyle(self) -> None:
        self.setStyleSheet(f"""
        QWidget {{ background:{T.BG.name()}; }}
        QLabel[role="muted"] {{ color:{T.MUTED.name()}; font-size:11px; padding:12px; }}
        """)
        for i in range(self.stack.count()):
            w = self.stack.widget(i)
            if w is not self.placeholder and hasattr(w, "restyle"):
                w.restyle()


class SourcesHost(_Host):
    """The active figure editor's Sources panel — the Nodes dock's face on a figure page."""

    placeholder_text = "The figure's sources appear here while an Analysis page is shown."

    def panel_of(self, ed: Any) -> QWidget:
        return ed.sources_panel


class _PropsHost(_Host):
    placeholder_text = "The selection's properties appear here while an Analysis page is shown."

    def panel_of(self, ed: Any) -> QWidget:
        return ed.props


class FigureSidePanel(QWidget):
    """The Properties dock's face on a figure page: the Figure widgets palette above the
    selected item's properties."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureSide")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.widgets = FigureWidgetsPanel()
        scroll = QScrollArea()
        scroll.setObjectName("figureSideScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.widgets)
        self._scroll = scroll
        self.props_host = _PropsHost()
        self.split = QSplitter(Qt.Vertical)
        self.split.addWidget(scroll)
        self.split.addWidget(self.props_host)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([380, 520])
        lay.addWidget(self.split)
        self._wired: List[Any] = []              # weakrefs to editors whose signals we follow
        self.restyle()

    @property
    def props_stack(self) -> QStackedWidget:
        return self.props_host.stack

    def show_editor(self, ed: Any) -> None:
        """Host ``ed``'s properties form, point the palette at ``ed`` and follow its
        selection (wired once per editor)."""
        self.props_host.show_editor(ed)
        if not any(r() is ed for r in self._wired):
            ed.selection_changed.connect(self.widgets.sync)
            self._wired.append(weakref.ref(ed))
        self.widgets.set_editor(ed)

    def forget_dead(self, alive: Iterable[Any]) -> None:
        alive = list(alive)
        self.props_host.forget_dead(alive)
        self._wired = [r for r in self._wired if r() is not None and r() in set(alive)]
        if self.widgets.editor() is not None and not any(e is self.widgets.editor()
                                                         for e in alive):
            self.widgets.set_editor(None)

    def restyle(self) -> None:
        self.setStyleSheet(f"""
        QWidget#figureSide {{ background:{T.BG.name()}; }}
        QScrollArea#figureSideScroll {{ background:{T.BG.name()}; border:0;
            border-bottom:1px solid {T.BORDER.name()}; }}
        QScrollArea#figureSideScroll > QWidget#qt_scrollarea_viewport {{
            background:{T.BG.name()}; }}
        """)
        self.widgets.restyle()
        self.props_host.restyle()


__all__ = ["ModeStack", "SourcesHost", "FigureSidePanel"]
