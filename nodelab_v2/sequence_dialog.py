"""The **Load file sequence** dialog — pick one file, get the whole numbered series.

A microscope that exports one file per frame leaves a folder of ``WellA3_t001.nd2`` ..
``WellA3_t120.nd2``. The ordinary loader can already multi-select all 120 into one bundle
card, but that is a 120-item file dialog and it stacks them on POSITIONS, which is the
wrong axis for a timelapse (see :mod:`nodegraph.catalog.util.timeseries`). This dialog is the
one-gesture form: pick any member, and the pattern, the siblings, the order and the axis
are all settled here before a card exists.

**Why the pattern is shown and editable rather than just applied.**
:func:`nodegraph.file_sequence.detect` picks the numeric field that VARIES across the
names, which is right whenever exactly one does. When two vary — a folder holding
``p1_t1``, ``p1_t2``, ``p2_t1`` — there is no fact that settles which is the sequence, and
choosing silently would order a series by a field that is not its counter: the pull
succeeds, the movie plays, and the frames are in the wrong order. So the detected pattern,
the file count and the resulting ORDER are all on screen, and the pattern is a text box.
The ambiguous case says so in as many words and still lets the user proceed, because they
know which field counts and this dialog cannot.

Every edit re-scans, so what the list shows is what the card will hold — not a description
of it.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QVBoxLayout, QWidget,
)

from nodegraph import file_sequence as FS

#: The chain axis choices, as ``(chain_axis mode value, what it means to a user)``. The
#: mode values are ``util.timeseries``'s own, so this list cannot drift from the node's.
AXIS_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("T", "Timepoints — one file per frame of a timelapse"),
    ("Z", "Z planes — one file per focal plane of a stack"),
    ("C", "Channels — one file per stain, acquired separately"),
    ("M", "Positions — keep them as separate fields (no chaining)"),
)

#: How many names the preview lists before it elides. A 120-file series is the case this
#: feature exists for, and a list box scrolled to 120 entries answers "did it find them
#: all?" no better than the count does — while making the ORDER, which is what actually
#: needs checking, harder to see at the ends.
_PREVIEW_MAX = 12


class SequenceScanDialog(QDialog):
    """Confirm the detected sequence and the axis to chain it onto.

    :meth:`result_paths` is the matched series in order; :meth:`chain_axis` is the
    ``util.timeseries`` mode value the caller presets on the node it wires up. Both are only
    meaningful after ``exec()`` returned :attr:`QDialog.Accepted`.
    """

    def __init__(self, path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load file sequence")
        self.setMinimumWidth(560)

        self._picked = str(path)
        self._folder = os.path.dirname(self._picked) or "."
        self._paths: List[str] = [self._picked]

        paths, spec = FS.scan(self._picked)
        self._paths = paths

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"<b>{os.path.basename(self._picked)}</b>"
                             f"<br><span style='color:#888'>{self._folder}</span>"))

        form = QFormLayout()
        self._pattern = QLineEdit(spec.pattern() if spec is not None
                                  else os.path.basename(self._picked))
        self._pattern.setToolTip(
            "The filename with the counting field written as {} — every file in the "
            "folder that matches becomes one frame of the series. Edit it to move the "
            "counter to a different number in the name.")
        self._pattern.textChanged.connect(self._rescan)
        form.addRow("Pattern", self._pattern)

        self._axis = QComboBox()
        for value, text in AXIS_CHOICES:
            self._axis.addItem(text, value)
        self._axis.setToolTip(
            "Which axis the files vary along. A source card always stacks its files on "
            "positions; this is what the Chain node re-addresses them onto.")
        form.addRow("Chain onto", self._axis)
        lay.addLayout(form)

        self._status = QLabel()
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        lay.addWidget(QLabel("Order the frames will be in:"))
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QListWidget.NoSelection)
        self._list.setMinimumHeight(170)
        lay.addWidget(self._list)

        row = QHBoxLayout()
        row.addStretch(1)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText("Load sequence")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        row.addWidget(self._buttons)
        lay.addLayout(row)

        self._refresh(spec)

    # ── results ──────────────────────────────────────────────────────────────
    def result_paths(self) -> List[str]:
        """The matched files, in the order they will become frames."""
        return list(self._paths)

    def chain_axis(self) -> str:
        """The ``util.timeseries`` ``chain_axis`` mode value the user picked."""
        return str(self._axis.currentData() or "T")

    # ── live re-scan ─────────────────────────────────────────────────────────
    def _rescan(self, text: str) -> None:
        """Re-match the folder against the typed pattern.

        A pattern that is not yet valid — mid-edit, no ``{}``, or a ``{}`` glued to literal
        digits — leaves the previous list in place and says what is wrong, rather than
        emptying the box on every keystroke that passes through an incomplete state.
        """
        spec = FS.compile_pattern(text)
        if spec is None:
            self._status.setText(
                "<span style='color:#c86'>The pattern needs exactly one <b>{}</b>, "
                "standing for a whole number in the name (not glued to other digits).</span>")
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            return
        self._paths, _ = FS.scan(self._picked, spec)
        self._refresh(spec)

    def _refresh(self, spec: Optional[FS.SequenceSpec]) -> None:
        """Repaint the count, the warnings and the ordered preview from ``self._paths``."""
        n = len(self._paths)
        self._list.clear()
        shown = self._paths if n <= _PREVIEW_MAX else \
            self._paths[:_PREVIEW_MAX // 2] + [None] + self._paths[-_PREVIEW_MAX // 2:]
        for i, p in enumerate(shown):
            if p is None:
                self._list.addItem(f"    ... {n - _PREVIEW_MAX} more ...")
                continue
            k = i if i < _PREVIEW_MAX // 2 else n - (len(shown) - i)
            self._list.addItem(f"{k + 1:>4}.  {os.path.basename(p)}")

        msgs: List[str] = []
        if n < 2:
            msgs.append(
                "<span style='color:#c86'>No siblings matched — this loads as a single "
                "file. Check the pattern, or that the rest of the series is in this "
                "folder with the same extension.</span>")
        else:
            msgs.append(f"<b>{n} files</b> will load as one source card.")
        if spec is not None and spec.ambiguous:
            fields = ", ".join(spec.segments[i] for i in spec.ambiguous)
            msgs.append(
                "<span style='color:#c86'><b>More than one number varies</b> across these "
                f"names ({fields}). The first is being used as the counter — if that is "
                "the wrong one, move the <b>{}</b> in the pattern above.</span>")
        elif spec is None and n >= 2:
            msgs.append(
                "<span style='color:#c86'>No single counting field was found, so the "
                "files are in natural-sorted order. Check the list before loading.</span>")
        self._status.setText("<br>".join(msgs))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(n >= 1)


__all__ = ["SequenceScanDialog", "AXIS_CHOICES"]
