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
every node it finished stays in the memo — re-running it later resumes from there. Until it
reaches that boundary it is listed as STOPPING, so a long node that is still finishing its
step does not look like a cancel that did nothing.

**Rows are kept, not rebuilt** (2026-10-13). The list used to be torn down and rebuilt on
every 500 ms tick, which broke it twice over on a real screen: a row added to a VISIBLE
popup is only shown on a queued call, so the list measured its hidden new rows as 0 px and
collapsed — the queue appeared, then emptied half a second later, and a click where a row
had been landed outside the shrunken popup and closed it; and a click whose press and
release straddled a tick was released over a deleted button, so Cancel did nothing. Each
job's row now lives as long as the job does and is updated in place; the list is rebuilt
only when the line itself changes, never while a mouse button is down, and new rows are
shown before anything is measured.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget)

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


def row_key(row: Dict[str, Any]) -> tuple:
    """A job's identity in the list: the node it computes and what kind of job it is. A row
    keeps its widgets for as long as its key stays in the line."""
    return (str(row.get("node", "")), str(row.get("kind", "")))


class NodeQueuePopup(QFrame):
    """The list, re-read from :meth:`QueueHost.node_queue_rows` on every refresh. The rows
    follow the runner's ORDER exactly — the list is rebuilt whenever the sequence of jobs
    changes — but a job that is still in line keeps its row widget, updated in place, so a
    button is never deleted from under the pointer."""

    def __init__(self, host: QueueHost, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.Popup)
        self.setObjectName("nodeQueue")
        self._host = host
        self.rows: List[Dict[str, Any]] = []
        #: row key → (row frame, badge, text, show button, cancel button)
        self._row_w: Dict[tuple, tuple] = {}
        #: row key → the row as last read, which a button click acts on
        self._row_of: Dict[tuple, Dict[str, Any]] = {}
        self._keys: List[tuple] = []
        self._anchor: Optional[QWidget] = None
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
        """Open under ``anchor`` (the canvas's button), right-aligned to it, kept on screen
        — above the button when there is no room under it."""
        self._anchor = anchor
        self.refresh()
        self.adjustSize()
        self._place()
        self.show()
        self._timer.start()

    def _place(self) -> None:
        anchor = self._anchor
        if anchor is None:
            return
        try:
            below = anchor.mapToGlobal(QPoint(anchor.width(), anchor.height() + 4))
            above = anchor.mapToGlobal(QPoint(anchor.width(), -4))
            scr = anchor.screen().availableGeometry() if anchor.screen() else None
        except RuntimeError:                     # the canvas went away under an open list
            return
        x, y = below.x() - self.width(), below.y()
        if scr is not None:
            x = max(scr.left() + 4, min(x, scr.right() - self.width() - 4))
            if y + self.height() > scr.bottom() - 4:
                # no room under the button: open upward, and never above the screen
                y = max(scr.top() + 4, above.y() - self.height())
        self.move(x, y)

    def hideEvent(self, e) -> None:                     # noqa: N802 — Qt override
        self._timer.stop()
        super().hideEvent(e)

    # ── content ────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-read the rows; update each kept row in place, and rebuild the list only when
        the line itself (which jobs, in what order) changed."""
        rows = list(self._host.node_queue_rows())
        keys = [row_key(r) for r in rows]
        if keys != self._keys and QApplication.mouseButtons() != Qt.NoButton \
                and self.isVisible():
            # a press is in progress inside the list: rebuilding now could move or delete
            # the button being clicked. The next tick (or the click's own refresh) does it.
            return
        self.rows = rows
        self._row_of = {row_key(r): r for r in rows}
        rebuild = keys != self._keys
        if rebuild:
            gone = [k for k in self._row_w if k not in self._row_of]
            for k in gone:
                frame = self._row_w.pop(k)[0]
                frame.hide()
                frame.setParent(None)
                frame.deleteLater()
            while self._list_lay.count():
                self._list_lay.takeAt(0)          # detach (kept rows are re-added below)
        for r in rows:
            k = row_key(r)
            if k not in self._row_w:
                self._row_w[k] = self._make_row(k)
            self._fill_row(k, r)
            if rebuild:
                self._list_lay.addWidget(self._row_w[k][0])
                # shown NOW: a child added to a visible parent is otherwise only shown on a
                # queued call, and a hidden row measures 0 px — which is how the list
                # used to collapse to nothing half a second after opening
                self._row_w[k][0].show()
        if rebuild:
            self._list_lay.addStretch(1)
        self._keys = keys
        self.empty.setVisible(not rows)
        self._scroll.setVisible(bool(rows))
        # as tall as its rows, up to MAX_LIST_PX — then it scrolls
        self._list_lay.activate()
        self._list.adjustSize()
        hint = self._list.sizeHint()
        bar = self._scroll.verticalScrollBar().sizeHint().width()
        self._scroll.setMinimumWidth(max(460, hint.width() + bar + 2))
        self._scroll.setFixedHeight(min(MAX_LIST_PX, hint.height() + 2))
        if self.isVisible():
            self.adjustSize()
            self._place()
        self.cancel_all_btn.setEnabled(any(r.get("cancellable") for r in rows))

    def _make_row(self, key: tuple) -> tuple:
        w = QFrame()
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 5, 6, 5)
        h.setSpacing(6)
        badge = QLabel("")
        badge.setObjectName("nqBadge")
        badge.setMinimumWidth(62)
        h.addWidget(badge)
        text = QLabel("")
        text.setTextFormat(Qt.RichText)
        text.setWordWrap(True)
        text.setMinimumWidth(240)           # a wrapping label is otherwise squeezed to a word
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        h.addWidget(text, 1)
        show = QToolButton()
        show.setText("Show")
        show.setToolTip("Go to this node — its page on the canvas, the card selected")
        # the click acts on the row AS IT IS NOW, looked up by key, not on a stale copy
        show.clicked.connect(lambda _=False, k=key: self._show(k))
        h.addWidget(show)
        cancel = QToolButton()
        cancel.setObjectName("nqCancel")
        cancel.clicked.connect(lambda _=False, k=key: self._cancel_key(k))
        h.addWidget(cancel)
        return (w, badge, text, show, cancel)

    def _fill_row(self, key: tuple, row: Dict[str, Any]) -> None:
        w, badge, text, show, cancel = self._row_w[key]
        name = "nqRowRunning" if row.get("state") != "queued" else "nqRow"
        if w.objectName() != name:
            w.setObjectName(name)
            w.style().unpolish(w)
            w.style().polish(w)
        badge.setText(row.get("badge", ""))
        text.setText(f"<b>{row.get('title', '')}</b>"
                     f"<span style='color:{T.MUTED.name()}'> · {row.get('page', '')}</span>"
                     f"<br><span style='color:{T.INK_2.name()}'>{row.get('detail', '')}</span>")
        cancel.setText("Stop" if row.get("kind") == "bake" else "Cancel")
        cancel.setEnabled(bool(row.get("cancellable")))
        cancel.setToolTip(row.get("why_not") or (
            "Stop the bake at its next block — nothing is written" if row.get("kind") == "bake"
            else "Stop computing this node — what it already finished stays cached"
            if row.get("state") == "running" else "Take this node out of the line"))
        row["_widgets"] = (show, cancel)

    def _show(self, key: tuple) -> None:
        row = self._row_of.get(key)
        if row is not None:
            self._host.show_queue_entry(row)

    def _cancel_key(self, key: tuple) -> None:
        row = self._row_of.get(key)
        if row is not None and row.get("cancellable"):
            self._cancel(row)

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


__all__ = ["MAX_LIST_PX", "NodeQueuePopup", "QueueHost", "REFRESH_MS", "row_key"]
