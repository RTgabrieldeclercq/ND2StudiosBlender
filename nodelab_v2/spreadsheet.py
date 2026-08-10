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
    QComboBox, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from nodegraph.domains import AXIS_ORDER, Domain, axes_of, is_lattice, is_structure
from nodegraph.structure import COORD_COLUMNS
from nodelab_v2 import theme as T
from nodelab_v2.tables import (
    _COORD_ORDER, _STRUCTURE_NAMES, _ordered_columns, all_tables, lattice_tables,
    structure_tables,
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

    def _show_current(self, *_a) -> None:
        key = self._pick.currentData()
        cols = self._tables.get(key) if key is not None else None
        if not cols:
            return
        names = _ordered_columns(cols)
        n = max((len(v) for v in cols.values()), default=0)
        self._table.clear()
        self._table.setColumnCount(len(names))
        self._table.setRowCount(n)
        self._table.setHorizontalHeaderLabels(names)
        for c, name in enumerate(names):
            vals = cols[name]
            for r in range(n):
                val = vals[r] if r < len(vals) else None
                if isinstance(val, (float, np.floating)):
                    txt = f"{float(val):.4g}"
                elif val is None:
                    txt = ""
                else:
                    txt = str(val)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self._table.setItem(r, c, it)
        self._table.resizeColumnsToContents()
        dom, layer = key
        unit = "elements" if dom in _STRUCTURE_NAMES else "index rows"
        self._status.setText(f"{dom}" + (f" · {layer}" if layer else "")
                             + f" — {n} {unit} × {len(names)} attributes")


__all__ = ["SpreadsheetPanel", "structure_tables", "lattice_tables", "all_tables"]
