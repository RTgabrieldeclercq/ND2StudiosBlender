"""The **mode switch** on the menu bar (V4.00 step 11d): *Normal* | *Troubleshooting*.

Troubleshooting mode is the solo-frame scope — every pull analyses only the frames the
Viewer's M/T/Z strips pick (or the one under the cursor) instead of the whole series
(Run ▸ *Troubleshoot: picked frames only*, `F9`). It changes what every number on screen
means, so it gets a control that is always in sight and always says which mode is on: two
segments at the right end of the menu bar, the active one lit — Troubleshooting in the same
amber as the canvas's SOLO frame and the status bar's chip.

The switch is only a face for the window's action: it emits :attr:`ModeSwitch.mode_changed`
when the user picks a segment, and :meth:`ModeSwitch.set_troubleshooting` mirrors the mode
however it was changed (F9, the Run menu, a probe), without emitting.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

from nodelab_v2 import theme as T

#: the two modes, in segment order
NORMAL, TROUBLESHOOTING = "Normal", "Troubleshooting"


class ModeSwitch(QWidget):
    """Two exclusive segments, *Normal* and *Troubleshooting*."""

    #: the user picked a segment: ``True`` = Troubleshooting
    mode_changed = Signal(bool)

    def __init__(self, tooltip: str = "") -> None:
        super().__init__()
        self.setObjectName("modeSwitch")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 8, 2)
        lay.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons = {}
        for i, (name, tip) in enumerate((
                (NORMAL, "Normal mode: every pull analyses the whole series."),
                (TROUBLESHOOTING, tooltip or "Troubleshooting mode (F9)."))):
            b = QToolButton(self)
            b.setText(name)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            b.setObjectName("modeLeft" if i == 0 else "modeRight")
            self._group.addButton(b, i)
            lay.addWidget(b)
            self.buttons[name] = b
        self.buttons[NORMAL].setChecked(True)
        self._group.idClicked.connect(lambda i: self.mode_changed.emit(i == 1))
        self.restyle()

    def troubleshooting(self) -> bool:
        return self.buttons[TROUBLESHOOTING].isChecked()

    def set_troubleshooting(self, on: bool) -> None:
        """Show ``on`` as the current mode, without emitting :attr:`mode_changed`."""
        b = self.buttons[TROUBLESHOOTING if on else NORMAL]
        if not b.isChecked():
            was = self._group.blockSignals(True)
            b.setChecked(True)
            self._group.blockSignals(was)
        self.restyle()

    def restyle(self) -> None:
        on = self.buttons[TROUBLESHOOTING].isChecked() if hasattr(self, "buttons") else False
        lit_bg = T.DIM2D if on else T.ACCENT_DIM
        lit_ink = T.DIM2D_INK if on else T.INK
        self.setStyleSheet(f"""
            QWidget#modeSwitch {{ background:transparent; }}
            QToolButton {{ background:{T.BODY.name()}; color:{T.MUTED.name()};
                border:1px solid {T.BORDER.name()}; padding:2px 10px; font-size:10px;
                font-weight:700; }}
            QToolButton#modeLeft {{ border-top-left-radius:6px;
                border-bottom-left-radius:6px; border-right:0; }}
            QToolButton#modeRight {{ border-top-right-radius:6px;
                border-bottom-right-radius:6px; }}
            QToolButton:hover:!checked {{ background:{T.PANEL_HI.name()};
                color:{T.INK.name()}; }}
            QToolButton:checked {{ background:{lit_bg.name()}; color:{lit_ink.name()}; }}
        """)


__all__ = ["ModeSwitch", "NORMAL", "TROUBLESHOOTING"]
