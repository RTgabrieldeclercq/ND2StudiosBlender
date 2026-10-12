"""The Node Queue (2026-10-12): what the runner is computing now and what waits behind it,
with a way to cancel each — opened from the **Node queue** button on every canvas.

The runner holds ONE pull slot and an ordered line behind it
(:meth:`~nodelab_v2.runner.EngineRunner.queue_snapshot`): the running pull, a Dock's bake
or a Movie-Editor fetch; the pulls the user asked for, in the order asked; the editor's
fetches after them; and, beside the slot, any source file being ingested. Until this, a
pull you no longer wanted could only be stopped by editing a node inside it, and everything
queued behind it waited it out.

This module is only the popup. The window turns the snapshot into rows — the node's own
title and the page it is on, how long it has run and which node it is computing, its place
in line — and does what a row asks (:class:`QueueHost`). Each row offers **Show** (go to
the node: its page on this canvas, the card selected and centred) and **Cancel** (a bake:
**Stop**; an ingest offers neither, and says why). **Cancel all** clears the line first, so
nothing waiting starts the moment the running job stops.

Cancelling never costs the work already done: a run stops at its next node boundary, and
every node it finished stays in the memo — re-running it later resumes from there.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from nodelab_v2 import theme as T

#: how often an open queue re-reads the runner (elapsed time, the node being computed)
REFRESH_MS = 500
#: the list grows with its rows up to this height, then scrolls
MAX_LIST_PX = 360


class QueueHost(Protocol):
    """What the popup asks of the window."""

    def node_queue_rows(self) -> List[Dict[str, Any]]: ...
    def cancel_queue_entry(self, row: Dict[str, Any]) -> bool: ...
    def cancel_queue_all(self) -> int: ...
    def show_queue_entry(self, row: Dict[str, Any]) -> None: ...


class NodeQueuePopup(QFrame):
    """The list. Rebuilt from :meth:`QueueHost.node_queue_rows` on every refresh — the line
    is short, and a rebuild is the only way the row ORDER can never disagree with the
    runner's."""

    def __init__(self, host: QueueHost, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.Popup)
        self.setObjectName("nodeQueue")
        self._host = host
        self.rows: List[Dict[str, Any]] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        title = QLabel("Node queue")
        title.setObjectName("nqTitle")
        head.addWidget(title, 1)
        self.cancel_all_btn = QPushButton("Cancel all")
        self.cancel_all_btn.setToolTip("Cancel everything that can be cancelled — the jobs "
                                       "waiting first, then the one running")
        self.cancel_all_btn.clicked.connect(lambda _=False: self._cancel_all())
        head.addWidget(self.cancel_all_btn)
        lay.addLayout(head)
        self.empty = QLabel("Nothing is running or waiting.")
        self.empty.setObjectName("nqEmpty")
        lay.addWidget(self.empty)
        self._list = QWidget()
        self._list_lay = QVBoxLayout(self._list)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._list)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll = scroll
        lay.addWidget(scroll)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.restyle()

    # ── showing ────────────────────────────────────────────────────────────────
    def popup_at(self, anchor: QWidget) -> None:
        """Open under ``anchor`` (the canvas's button), right-aligned to it, kept on screen."""
        self.refresh()
        self.adjustSize()
        g = anchor.mapToGlobal(QPoint(anchor.width(), anchor.height() + 4))
        x = g.x() - self.width()
        scr = anchor.screen().availableGeometry() if anchor.screen() else None
        if scr is not None:
            x = max(scr.left() + 4, min(x, scr.right() - self.width() - 4))
        self.move(x, g.y())
        self.show()
        self._timer.start()

    def hideEvent(self, e) -> None:                     # noqa: N802 — Qt override
        self._timer.stop()
        super().hideEvent(e)

    # ── content ────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-read the rows and rebuild the list."""
        self.rows = list(self._host.node_queue_rows())
        while self._list_lay.count():
            item = self._list_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for row in self.rows:
            self._list_lay.addWidget(self._row_widget(row))
        self._list_lay.addStretch(1)
        self.empty.setVisible(not self.rows)
        self._scroll.setVisible(bool(self.rows))
        # as tall as its rows, up to MAX_LIST_PX — then it scrolls
        self._list.adjustSize()
        hint = self._list.sizeHint()
        bar = self._scroll.verticalScrollBar().sizeHint().width()
        self._scroll.setMinimumWidth(max(460, hint.width() + bar + 2))
        self._scroll.setFixedHeight(min(MAX_LIST_PX, hint.height() + 2))
        if self.isVisible():
            self.adjustSize()
        self.cancel_all_btn.setEnabled(any(r.get("cancellable") for r in self.rows))

    def _row_widget(self, row: Dict[str, Any]) -> QWidget:
        w = QFrame()
        w.setObjectName("nqRowRunning" if row.get("state") != "queued" else "nqRow")
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 5, 6, 5)
        h.setSpacing(6)
        badge = QLabel(row.get("badge", ""))
        badge.setObjectName("nqBadge")
        badge.setMinimumWidth(62)
        h.addWidget(badge)
        text = QLabel(f"<b>{row.get('title', '')}</b>"
                      f"<span style='color:{T.MUTED.name()}'> · {row.get('page', '')}</span>"
                      f"<br><span style='color:{T.INK_2.name()}'>{row.get('detail', '')}</span>")
        text.setTextFormat(Qt.RichText)
        text.setWordWrap(True)
        text.setMinimumWidth(240)           # a wrapping label is otherwise squeezed to a word
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        h.addWidget(text, 1)
        show = QToolButton()
        show.setText("Show")
        show.setToolTip("Go to this node — its page on the canvas, the card selected")
        show.clicked.connect(lambda _=False, r=row: self._host.show_queue_entry(r))
        h.addWidget(show)
        cancel = QToolButton()
        cancel.setObjectName("nqCancel")
        cancel.setText("Stop" if row.get("kind") == "bake" else "Cancel")
        cancel.setEnabled(bool(row.get("cancellable")))
        cancel.setToolTip(row.get("why_not") or (
            "Stop the bake at its next block — nothing is written" if row.get("kind") == "bake"
            else "Stop computing this node — what it already finished stays cached"
            if row.get("state") == "running" else "Take this node out of the line"))
        cancel.clicked.connect(lambda _=False, r=row: self._cancel(r))
        h.addWidget(cancel)
        row["_widgets"] = (show, cancel)
        return w

    def _cancel(self, row: Dict[str, Any]) -> None:
        self._host.cancel_queue_entry(row)
        self.refresh()

    def _cancel_all(self) -> None:
        self._host.cancel_queue_all()
        self.refresh()

    def restyle(self) -> None:
        self.setStyleSheet(
            f"QFrame#nodeQueue {{ background:{T.PANEL.name()}; "
            f"border:1px solid {T.BORDER_HI.name()}; border-radius:8px; }}"
            f"QLabel#nqTitle {{ color:{T.INK.name()}; font-weight:800; font-size:11px; }}"
            f"QLabel#nqEmpty {{ color:{T.MUTED.name()}; font-size:10px; padding:6px 2px; }}"
            f"QFrame#nqRow {{ background:{T.BODY.name()}; border-radius:6px; }}"
            f"QFrame#nqRowRunning {{ background:{T.BODY.name()}; border-radius:6px; "
            f"border-left:3px solid {T.ACCENT.name()}; }}"
            f"QLabel#nqBadge {{ color:{T.ACCENT.name()}; font-weight:700; font-size:10px; }}"
            f"QToolButton {{ color:{T.INK.name()}; background:{T.PANEL_HI.name()}; "
            f"border:1px solid {T.BORDER.name()}; border-radius:5px; padding:2px 8px; }}"
            f"QToolButton:hover {{ border-color:{T.ACCENT.name()}; }}"
            f"QToolButton#nqCancel:hover {{ border-color:{T.ERROR.name()}; "
            f"color:{T.ERROR.name()}; }}"
            f"QToolButton:disabled {{ color:{T.MUTED.name()}; }}")


__all__ = ["MAX_LIST_PX", "NodeQueuePopup", "QueueHost", "REFRESH_MS"]
