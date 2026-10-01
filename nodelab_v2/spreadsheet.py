"""Spreadsheet panel (G5 / V2.00 §10) — the per-domain structure table.

A pulled :class:`~nodegraph.dataset.Dataset` carries its detected structures as
element-indexed :class:`~nodegraph.dataset.AttributeLayer`\\ s on the Label / Point /
Track domains, keyed by source layer (``Dataset.with_structure``). This panel groups
those into tables — one per ``(domain, layer)`` — and shows the selected one as rows
(elements, ordered by id) × columns (attributes, coordinate columns first per
:data:`nodegraph.structure.COORD_COLUMNS`). The export path (Phase 6) reads from here.

**Coarse lattice attributes are tabulated here too** (V2.17). A Frame / Plane / Timepoint
/ Multipoint / Channel / Global attribute is a small array indexed by acquisition axes —
``align.drift``'s per-``(m,t)`` ``drift_y``/``drift_x``, anything
``transform.transfer_domain`` produces — and it had **no output surface at all**: the
Viewer draws only Voxel rasters and the structure overlays, and this panel's
``is_structure`` gate dropped the rest, which the CSV/Parquet export inherits. So the
entire non-Voxel half of the acquisition lattice was write-only: the engine computed it
correctly and the user could neither see nor export a single number. :func:`lattice_tables`
unrolls each one into rows keyed by its own axes.

``Voxel`` stays out, and that is not the same omission: a Voxel layer is one row per voxel
(12.6 M for a 3-position 4-frame 1024² series), it is image-shaped by definition, and the
Viewer already renders it.

**Where the tabulation itself lives.** The four grouping functions moved to the Qt-free
:mod:`nodelab_v2.tables` when the LabLink worker landed — a headless process needs them
and must not import PySide6 to get them. They are re-exported here unchanged, so every
existing ``from nodelab_v2.spreadsheet import all_tables`` keeps working and there is
still exactly one implementation.

Qt; reads only the numpy columns off the Dataset.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QTabBar, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from nodegraph.domains import AXIS_ORDER, Domain, axes_of, is_lattice, is_structure
from nodegraph.structure import COORD_COLUMNS
from nodelab_v2 import theme as T
from nodelab_v2.tables import (
    _COORD_ORDER, _STRUCTURE_NAMES, SOURCE_FILE_COLUMN, _ordered_columns,
    all_tables, lattice_tables, structure_tables,
)


class SpreadsheetPanel(QWidget):
    """Per-domain structure table for the viewed node's Dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.restyle()
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        self._pick = QComboBox()
        self._pick.currentIndexChanged.connect(self._show_current)
        head.addWidget(QLabel("Domain"))
        head.addWidget(self._pick, 1)
        v.addLayout(head)
        # ── per-file tabs (V3.01) ────────────────────────────────────────────────
        # When several files ran through one pipeline, every row in this table belongs to
        # one of them and the `file` column says which. A column is the honest place to
        # STORE that; it is a poor place to read it, because "the measurements for
        # WellA3" then means scrolling a 6000-row table looking for where one name stops
        # and the next starts. The tabs are that column, turned into the thing the user
        # actually asked the table for. `All` stays first and selected, so a single-file
        # run is untouched and a multi-file one still has one place to see everything.
        self._files = QTabBar()
        self._files.setExpanding(False)
        self._files.setDrawBase(False)
        self._files.currentChanged.connect(self._show_rows)
        self._files.hide()
        v.addWidget(self._files)
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self._table, 1)
        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        v.addWidget(self._status)
        self._tables: Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]] = {}
        self.dataset = None                       # the last shown Dataset (for export)
        self.node_id: Optional[str] = None

    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QLabel[role="muted"] {{ color:{T.MUTED.name()}; }}
            QTableWidget {{ background:{T.BG.name()}; color:{T.INK.name()};
                gridline-color:{T.BORDER.name()}; border:1px solid {T.BORDER.name()};
                border-radius:8px; font-family:{T.MONO}; outline:0; }}
            QHeaderView::section {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
                border:0; border-right:1px solid {T.BORDER.name()};
                border-bottom:1px solid {T.BORDER.name()}; padding:4px 7px;
                font-weight:600; }}
            QTableCornerButton::section {{ background:{T.BODY.name()};
                border:0; border-bottom:1px solid {T.BORDER.name()}; }}
            QTableWidget::item {{ padding:2px 4px; }}
            QTableWidget::item:selected {{ background:{T.ACCENT_DIM.name()};
                color:{T.INK.name()}; }}
        """ + T.controls_qss())

    def show_dataset(self, node_id: str, dataset) -> None:
        self.dataset = dataset
        self.node_id = node_id
        self._tables = all_tables(dataset)
        self._pick.blockSignals(True)
        self._pick.clear()
        for (dom, layer) in sorted(self._tables):
            self._pick.addItem(f"{dom}" + (f" · {layer}" if layer else ""),
                               (dom, layer))
        self._pick.blockSignals(False)
        if self._tables:
            self._pick.setCurrentIndex(0)
            self._show_current()
        else:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._status.setText(f"{node_id}: no tabulatable attributes on this "
                                 f"output (image layers show in the Viewer)")

    @staticmethod
    def _file_names(cols) -> List[str]:
        """Distinct source files in this table, in the order they first appear.

        First-appearance order rather than sorted: it is the order the files were wired
        into the batch (or bundled), which is the order everything else about the run —
        the member sockets, the fanned-out cards — already uses. Sorting here would make
        tab 2 mean a different file from output 2.
        """
        vals = cols.get(SOURCE_FILE_COLUMN)
        if vals is None:
            return []
        out: List[str] = []
        for v in vals:
            s = str(v)
            if s not in out:
                out.append(s)
        return out

    def _show_current(self, *_a) -> None:
        """Re-read the selected table and rebuild the file tabs for it."""
        key = self._pick.currentData()
        cols = self._tables.get(key) if key is not None else None
        if not cols:
            return
        files = self._file_names(cols)
        self._files.blockSignals(True)
        while self._files.count():
            self._files.removeTab(0)
        # one file is not a choice — a lone tab beside `All` would offer the same rows
        # under two names, the same floor the channel and member sockets use
        if len(files) >= 2:
            self._files.addTab(f"All ({len(files)} files)")
            for name in files:
                self._files.addTab(name)
            self._files.setCurrentIndex(0)
            self._files.show()
        else:
            self._files.hide()
        self._files.blockSignals(False)
        self._show_rows()

    def _show_rows(self, *_a) -> None:
        """Fill the grid with the selected table, narrowed to the selected file."""
        key = self._pick.currentData()
        cols = self._tables.get(key) if key is not None else None
        if not cols:
            return
        names = _ordered_columns(cols)
        total = max((len(v) for v in cols.values()), default=0)
        files = self._file_names(cols)
        # gated on the tabs EXISTING, never on isVisible(): a child of a parent that has
        # not been shown yet reports invisible, so a visibility test silently disables the
        # filter in exactly the cases where the panel is being populated before display
        tabbed = self._files.count() > 1
        idx = self._files.currentIndex() if tabbed else 0
        rows = list(range(total))
        picked = ""
        if tabbed and idx > 0 and (idx - 1) < len(files):
            picked = files[idx - 1]
            vals = cols.get(SOURCE_FILE_COLUMN)
            rows = [r for r in range(total)
                    if r < len(vals) and str(vals[r]) == picked]
        n = len(rows)
        self._table.clear()
        self._table.setColumnCount(len(names))
        self._table.setRowCount(n)
        self._table.setHorizontalHeaderLabels(names)
        # the row header keeps the ORIGINAL row number while a file is selected, so a row
        # can still be found in the full table and in the exported CSV, which is not
        # filtered — a renumbered view would quietly disagree with the file on disk
        if picked:
            self._table.setVerticalHeaderLabels([str(r) for r in rows])
        for c, name in enumerate(names):
            vals = cols[name]
            for r_out, r in enumerate(rows):
                val = vals[r] if r < len(vals) else None
                if isinstance(val, (float, np.floating)):
                    txt = f"{float(val):.4g}"
                elif val is None:
                    txt = ""
                else:
                    txt = str(val)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self._table.setItem(r_out, c, it)
        self._table.resizeColumnsToContents()
        dom, layer = key
        unit = "elements" if dom in _STRUCTURE_NAMES else "index rows"
        note = f"{dom}" + (f" · {layer}" if layer else "")
        if picked:
            note += f" · {picked} — {n} of {total} {unit}"
        else:
            note += f" — {n} {unit}"
            if len(files) >= 2:
                note += f" across {len(files)} files"
        self._status.setText(f"{note} × {len(names)} attributes")


__all__ = ["SpreadsheetPanel", "structure_tables", "lattice_tables", "all_tables"]
