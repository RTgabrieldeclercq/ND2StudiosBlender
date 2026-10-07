"""The question a structural edit on a LINKED page asks (V4.00 step 11e).

A linked page's graph is its master's: until step 11e every add, delete or rewire there was
refused with a hint. Now the first such edit asks how it should apply — three answers, each
a button that says what it does:

* **Make unique** — the page gets a graph of its own, holding what it has now; the master's
  edits stop reaching it.
* **Keep the change on this page** — a *modified* linked page: the change (and every later
  one) is this page's own, laid over the master, whose other edits keep arriving.
* **Add it to the master, switched off** — the master gets the change with the node it adds
  switched off (passed through), so it and its other linked pages compute exactly as before;
  on this page the node is on. Offered only where that can hold — never for frames, groups or
  zones, nor for a Load card (where data starts: nothing to pass through).

The window shows it from :meth:`nodelab_v2.window.MainWindow.ask_linked_edit` and applies the
answer (:attr:`LinkedEditDialog.choice`, ``""`` when cancelled).
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from nodelab_v2 import theme as T
from nodelab_v2.linked_document import EDIT_MASTER, EDIT_MODIFIED, EDIT_UNIQUE


class _Choice(QPushButton):
    """One answer: a button holding a bold title over a WRAPPED description (a plain
    QPushButton cannot wrap its text), and the reason it is greyed out when it is."""

    def __init__(self, title: str, text: str, why_not: str = "") -> None:
        super().__init__()
        self.setObjectName("choice")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(3)
        self.title_label = QLabel(title)
        f = self.title_label.font()
        f.setBold(True)
        self.title_label.setFont(f)
        self.text_label = QLabel(text + (f" Not here: {why_not}." if why_not else ""))
        self.text_label.setWordWrap(True)
        for lab in (self.title_label, self.text_label):
            lab.setAttribute(Qt.WA_TransparentForMouseEvents)
            lay.addWidget(lab)
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        sp.setVerticalPolicy(QSizePolicy.Minimum)
        self.setSizePolicy(sp)
        self.setEnabled(not why_not)
        self.setCursor(Qt.PointingHandCursor)

    def hasHeightForWidth(self) -> bool:                  # noqa: N802 - Qt naming
        return True

    def heightForWidth(self, w: int) -> int:              # noqa: N802 - Qt naming
        return self.layout().heightForWidth(w)

    def sizeHint(self) -> QSize:                          # noqa: N802 - Qt naming
        return QSize(480, self.heightForWidth(480))

    def minimumSizeHint(self) -> QSize:                   # noqa: N802 - Qt naming
        return QSize(320, self.layout().minimumSize().height())


class LinkedEditDialog(QDialog):
    """Ask how a structural edit on linked page ``page_name`` (following ``master_name``)
    applies. ``shape``: the edit is a frame, group or zone — only *Make unique* is possible.
    ``push_note``: why the edit cannot go to the master switched off (that answer is greyed
    out, saying so); ``""`` when it may."""

    def __init__(self, parent: Optional[QWidget], page_name: str, master_name: str, *,
                 shape: bool = False, push_note: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Change a linked page")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.choice = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(8)
        head = QLabel(f"“{page_name}” is linked to “{master_name}”")
        f = head.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        head.setFont(f)
        lay.addWidget(head)
        sub = QLabel("Its graph is the master's. How should this change apply?")
        sub.setWordWrap(True)
        sub.setProperty("role", "muted")
        lay.addWidget(sub)
        shape_note = ("frames, groups and zones of a linked page are its master's — "
                      "only a page of its own can change them")
        options = (
            (EDIT_UNIQUE, "Make unique",
             f"“{page_name}” gets a graph of its own, holding what it has now. "
             f"“{master_name}”'s edits stop reaching it.", ""),
            (EDIT_MODIFIED, "Keep the change on this page",
             f"A modified linked page: this change — and every later one — stays on "
             f"“{page_name}”, and “{master_name}”'s other edits keep arriving. Where both "
             f"change the same wire, this page's change wins.",
             shape_note if shape else ""),
            (EDIT_MASTER, "Add it to the master, switched off",
             f"“{master_name}” gets the change with the node it adds switched off, so it and "
             f"its other linked pages compute as before; here the node is on. For nodes that "
             f"keep the kind of data (a filter, not a threshold); a rewire that would change "
             f"what the master computes is refused. Lasts until the app closes.",
             shape_note if shape else push_note),
        )
        self.buttons: Dict[str, QPushButton] = {}
        for key, title, text, why_not in options:
            btn = _Choice(title, text, why_not)
            btn.clicked.connect(lambda _=False, k=key: self._pick(k))
            self.buttons[key] = btn
            lay.addWidget(btn)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        lay.addLayout(row)
        self.cancel_button = cancel
        self.restyle()

    def _pick(self, key: str) -> None:
        self.choice = key
        self.accept()

    def restyle(self) -> None:
        self.setStyleSheet(T.controls_qss() + f"""
            QDialog {{ background:{T.PANEL.name()}; }}
            QLabel {{ color:{T.INK.name()}; background:transparent; }}
            QLabel[role="muted"] {{ color:{T.MUTED.name()}; }}
            QPushButton#choice {{ background:{T.BODY.name()}; border:1px solid
                {T.BORDER.name()}; border-radius:6px; }}
            QPushButton#choice:hover {{ border-color:{T.ACCENT.name()};
                background:{T.PANEL_HI.name()}; }}
            QPushButton#choice:disabled {{ background:{T.PANEL.name()}; }}
            QPushButton#choice QLabel {{ color:{T.INK_2.name()}; }}
            QPushButton#choice QLabel:disabled {{ color:{T.MUTED.name()}; }}
        """)


__all__ = ["LinkedEditDialog"]
