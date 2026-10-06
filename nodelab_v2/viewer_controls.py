"""The **Playback** and **Channels** panels (V4.00 step 11d): a Viewer's controls as panels
of their own.

A :class:`~nodelab_v2.viewer.ViewerPanel` builds two control SECTIONS under its image — the
M/T/Z cursor with its play buttons (with the overlay source strip and the iteration strip),
and the display tools over one column per channel (toggle, histogram, black/white point).
The window takes both out of every Viewer it manages
(:meth:`~nodelab_v2.viewer.ViewerPanel.detach_controls`) and hosts them here, so each can be
docked, tabbed, floated or closed wherever the user wants, like any other panel.

There is one Playback panel and one Channels panel however many Viewers are open: each
holds every Viewer's section on a stack and shows the ACTIVE Viewer's — the same rule the
Spreadsheet and the troubleshooting scope follow. A Compare viewer whose cursor is linked to
its leader's shows the leader's Playback section: one set of sliders moves both. The
sections stay the Viewer's own widgets, so every behaviour (scrub, play, pick frames, LUT
drag, colour menu) is the Viewer's code, unchanged.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from nodelab_v2 import theme as T


class ViewerControlsPanel(QWidget):
    """One of the Viewer's control sections, for the active Viewer — or a line saying
    there is none."""

    def __init__(self, title: str, empty_text: str) -> None:
        super().__init__()
        self.setObjectName("viewerControls")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.title = title
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(0)
        self.stack = QStackedWidget(self)
        lay.addWidget(self.stack)
        self._empty = QLabel(empty_text)
        self._empty.setObjectName("viewerControlsEmpty")
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self._empty)
        #: id(viewer) → (viewer, its section)
        self._sections: Dict[int, tuple] = {}
        self._shown: Optional[QWidget] = None
        self.restyle()

    def adopt(self, viewer: QWidget, section: QWidget) -> None:
        """Host ``viewer``'s ``section`` (shown once that Viewer is the active one)."""
        self._sections[id(viewer)] = (viewer, section)
        self.stack.addWidget(section)

    def section_of(self, viewer: Optional[QWidget]) -> Optional[QWidget]:
        """``viewer``'s section while it is HERE — not while its Viewer has taken it back
        (the mini-map, :meth:`~nodelab_v2.viewer.ViewerPanel.attach_controls`)."""
        got = self._sections.get(id(viewer)) if viewer is not None else None
        if got is None or got[0] is not viewer or self.stack.indexOf(got[1]) < 0:
            return None
        return got[1]

    def show_for(self, viewer: Optional[QWidget]) -> None:
        """Show ``viewer``'s section — or the empty line when it is ``None`` or not one of
        the Viewers hosted here."""
        sec = self.section_of(viewer)
        self.stack.setCurrentWidget(sec if sec is not None else self._empty)
        self._shown = viewer if sec is not None else None

    def shown_viewer(self) -> Optional[QWidget]:
        """The Viewer whose section is on screen, or ``None``."""
        return self._shown

    def prune(self, alive: Iterable[QWidget]) -> None:
        """Let go of the sections of Viewers that were closed (their panel is gone; the
        section, living here, would otherwise outlive it)."""
        keep = {id(v) for v in alive}
        for key in [k for k in self._sections if k not in keep]:
            viewer, sec = self._sections.pop(key)
            if self._shown is viewer:
                self.stack.setCurrentWidget(self._empty)
                self._shown = None
            self.stack.removeWidget(sec)
            sec.hide()
            sec.setParent(None)
            sec.deleteLater()

    def restyle(self) -> None:
        # the sections carry the Viewer's own stylesheet (`ViewerPanel.restyle`); every rule
        # here is scoped by object name, because an ancestor's rule still reaches into them
        # for any property their own sheet leaves unset
        self.setStyleSheet(f"""
            QWidget#viewerControls {{ background:{T.PANEL.name()}; }}
            QLabel#viewerControlsEmpty {{ color:{T.MUTED.name()}; background:transparent;
                font-size:11px; padding:10px; }}
        """)


__all__ = ["ViewerControlsPanel"]
