"""The Viewer's **Info** window (2026-10-13): every acquisition setting the viewed data's
source file recorded, plus the viewed series' own time, in one place.

The content is :func:`nodelab_v2.acquisition_view.info_sections` — Qt-free and selftested —
laid into a two-column tree (one expandable section each), with the file's raw ``text_info``
blocks verbatim on a second tab, so nothing the ND2 SDK exposes is hidden behind the parse.
Non-modal, like the Overlays popup: it can sit beside the image while you scrub, and it
follows the Viewer to whatever it shows next.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout)

from nodelab_v2.acquisition_view import info_sections, info_text


class AcquisitionDialog(QDialog):
    """Non-modal window listing a payload's acquisition metadata."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Acquisition metadata")
        self.resize(760, 680)
        self._text = ""
        lay = QVBoxLayout(self)
        self._head = QLabel("")
        self._head.setWordWrap(True)
        self._head.setProperty("role", "muted")
        lay.addWidget(self._head)

        self._tabs = QTabWidget()
        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Setting", "Value"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setWordWrap(True)
        self._tree.setUniformRowHeights(False)
        self._raw = QPlainTextEdit()
        self._raw.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self._raw.setFont(mono)
        self._tabs.addTab(self._tree, "Settings")
        self._tabs.addTab(self._raw, "Raw text")
        lay.addWidget(self._tabs, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        copy = QPushButton("Copy all")
        copy.setToolTip("Copy every section, and the raw text blocks, as plain text")
        copy.clicked.connect(self._copy)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(copy)
        row.addWidget(close)
        lay.addLayout(row)

    def load(self, title: str, md: Any, axes: Any) -> None:
        """Show ``md`` (a payload's metadata) for data of shape ``axes``."""
        sections = info_sections(md, axes)
        self._text = info_text(md, axes)
        self._head.setText(title)
        self._tree.clear()
        for name, rows in sections:
            top = QTreeWidgetItem([name, ""])
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            for k, v in rows:
                item = QTreeWidgetItem([k, v])
                item.setToolTip(1, v)
                top.addChild(item)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)
        self._tree.resizeColumnToContents(0)
        raw = (md or {}).get("acquisition_text") if isinstance(md, dict) else None
        if isinstance(raw, dict) and raw:
            self._raw.setPlainText("\n\n".join(f"--- {k} ---\n{str(v).rstrip()}"
                                               for k, v in raw.items()))
        else:
            self._raw.setPlainText("The viewed data carries no raw text from a source file "
                                   "(it was not loaded from an ND2, or a node above it "
                                   "made new data).")

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self._text)


__all__ = ["AcquisitionDialog"]
