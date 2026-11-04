"""The Properties panel of the Figure Editor (V5.00 step 3).

A form for whatever is selected — the page when nothing is — built from small
declarative field lists per item kind. Every control commits ONE edit to the document
through the panel (``panel.edit_item(item_id, path, value)``), on a settled value, never
per keystroke or drag tick. Channels and layers of an image are sub-forms with their own
add / remove buttons.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
                               QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout,
                               QWidget)

from nodegraph.figure import layout as FL
from nodegraph.figure import schema as S
from nodegraph.figure import stats as ST
from nodelab_v2 import theme as T


class _NoWheelCombo(QComboBox):
    def wheelEvent(self, e) -> None:                                      # noqa: N802
        e.ignore()


class _NoWheelSpin(QDoubleSpinBox):
    def wheelEvent(self, e) -> None:                                      # noqa: N802
        e.ignore()


class _NoWheelInt(QSpinBox):
    def wheelEvent(self, e) -> None:                                      # noqa: N802
        e.ignore()


class ColorButton(QPushButton):
    picked = Signal(str)

    def __init__(self, value: str) -> None:
        super().__init__()
        self._value = str(value or "#ffffff")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(22)
        self.clicked.connect(self._choose)
        self.set_value(self._value)

    def set_value(self, value: str) -> None:
        self._value = str(value)
        col = QColor(self._value)
        ink_ = "#0b0e12" if col.lightness() > 140 else "#f0f3f7"
        self.setText(self._value)
        self.setStyleSheet(f"QPushButton {{ background:{col.name()}; color:{ink_};"
                           f" font-family:{T.MONO}; font-size:11px; border:1px solid "
                           f"{T.BORDER.name()}; border-radius:5px; padding:2px 8px; }}")

    def value(self) -> str:
        return self._value

    def _choose(self) -> None:
        col = QColorDialog.getColor(QColor(self._value), self, "Colour")
        if col.isValid():
            self.set_value(col.name())
            self.picked.emit(col.name())


#: a field: (path under the item, kind, label, options)
Field = Tuple[str, str, str, Dict[str, Any]]

PAGE_FIELDS: List[Field] = [
    ("preset", "choice", "Page", {"choices": [(k, f"{k.replace('_', ' ')} — {v['width_mm']:g}"
                                               f" × {v['height_mm']:g} mm")
                                              for k, v in FL.PAGE_PRESETS.items()]
                                  + [("custom", "custom")]}),
    ("width_mm", "float", "Width (mm)", {"lo": 10, "hi": 2000, "step": 1, "dec": 1}),
    ("height_mm", "float", "Height (mm)", {"lo": 10, "hi": 2000, "step": 1, "dec": 1}),
    ("dpi", "int", "Export dpi", {"lo": 50, "hi": 2400}),
    ("font", "choice", "Font", {"choices": [("sans", "sans"), ("serif", "serif")]}),
    ("font_pt", "float", "Type size (pt)", {"lo": 3, "hi": 72, "step": 0.5, "dec": 1}),
    ("palette", "choice", "Palette", {"choices": [(p, p) for p in S.PALETTE_NAMES]}),
    ("dark", "bool", "Dark page", {}),
    ("grid_mm", "float", "Snap grid (mm)", {"lo": 0, "hi": 50, "step": 0.5, "dec": 1}),
    ("letters", "choice", "Letters", {"choices": [(s, s) for s in FL.LETTER_STYLES]}),
]

COMMON_FIELDS: List[Field] = [
    ("letter", "text", "Letter", {}),
    ("rect_mm.0", "float", "X (mm)", {"lo": -1000, "hi": 3000, "step": 0.5, "dec": 1}),
    ("rect_mm.1", "float", "Y (mm)", {"lo": -1000, "hi": 3000, "step": 0.5, "dec": 1}),
    ("rect_mm.2", "float", "Width (mm)", {"lo": 1, "hi": 3000, "step": 0.5, "dec": 1}),
    ("rect_mm.3", "float", "Height (mm)", {"lo": 1, "hi": 3000, "step": 0.5, "dec": 1}),
    ("z", "int", "Stacking", {"lo": -100, "hi": 100}),
    ("locked", "bool", "Locked", {}),
]

IMAGE_FIELDS: List[Field] = [
    ("image.source", "source", "Source", {}),
    ("image.frame.m", "frame", "Position (m)", {"axis": "m"}),
    ("image.frame.t", "frame", "Frame (t)", {"axis": "t"}),
    ("image.frame.z", "frame", "Plane (z)", {"axis": "z"}),
    ("image.flip_x", "bool", "Flip X", {}),
    ("image.flip_y", "bool", "Flip Y", {}),
    ("image.z_mode", "choice", "Z colour code", {"choices": [("plain", "off"),
                                                             ("depth_coded", "depth-coded")]}),
    ("image.t_mode", "choice", "T colour code", {"choices": [("plain", "off"),
                                                             ("temporal_code", "time-coded")]}),
    ("image.decor.scale_bar_um", "float", "Scale bar (µm)", {"lo": 0, "hi": 100000,
                                                            "step": 5, "dec": 1}),
    ("image.decor.time_stamp", "choice", "Time stamp", {"choices": [("", "off"),
                                                                     ("elapsed", "elapsed"),
                                                                     ("clock", "clock"),
                                                                     ("frame", "frame")]}),
    ("image.decor.channel_labels", "bool", "Channel labels", {}),
]

MONTAGE_FIELDS: List[Field] = [
    ("montage.tile_by", "choice", "Tile by", {"choices": [("m", "position"), ("t", "time"),
                                                          ("z", "plane"), ("c", "channel")]}),
    ("montage.columns", "int", "Columns (0 = auto)", {"lo": 0, "hi": 64}),
    ("montage.gap_mm", "float", "Gap (mm)", {"lo": 0, "hi": 50, "step": 0.25, "dec": 2}),
    ("montage.count", "int", "Tiles (0 = all)", {"lo": 0, "hi": 1000}),
    ("montage.start", "int", "First index", {"lo": 0, "hi": 100000}),
    ("montage.step", "int", "Every", {"lo": 1, "hi": 1000}),
    ("montage.labels", "choice", "Tile labels", {"choices": [("auto", "auto"),
                                                             ("none", "none")]}),
]

KIND_FIELDS: Dict[str, List[Field]] = {
    "colorbar": [("of.item", "item", "Of image", {"kinds": ("image", "montage")}),
                 ("of.layer", "int", "Layer index", {"lo": 0, "hi": 20}),
                 ("label", "text", "Label", {}),
                 ("orientation", "choice", "Orientation", {"choices": [("horizontal", "horizontal"),
                                                                       ("vertical", "vertical")]}),
                 ("ticks", "int", "Ticks", {"lo": 2, "hi": 11})],
    "histogram": [("of.item", "item", "Of image", {"kinds": ("image", "montage")}),
                  ("of.channel", "int", "Channel", {"lo": 0, "hi": 15}),
                  ("log", "bool", "Log counts", {}),
                  ("show_window", "bool", "Mark the display window", {})],
    "scale_bar": [("of.item", "item", "Of image", {"kinds": ("image", "montage")}),
                  ("length_um", "float", "Length (µm)", {"lo": 0.1, "hi": 100000, "step": 5,
                                                         "dec": 1}),
                  ("label", "bool", "Label", {})],
    "text": [("text", "multiline", "Text", {}),
             ("pt", "float", "Size (pt, 0 = page)", {"lo": 0, "hi": 72, "step": 0.5, "dec": 1}),
             ("bold", "bool", "Bold", {}),
             ("align", "choice", "Align", {"choices": [("left", "left"), ("center", "centre"),
                                                       ("right", "right")]}),
             ("color", "color", "Colour", {})],
    "letter": [("text", "text", "Letter", {}),
               ("pt", "float", "Size (pt, 0 = page)", {"lo": 0, "hi": 72, "step": 0.5, "dec": 1})],
    "shape": [("shape", "choice", "Shape", {"choices": [("rect", "rectangle"),
                                                        ("ellipse", "ellipse")]}),
              ("color", "color", "Colour", {}),
              ("width_pt", "float", "Line (pt)", {"lo": 0.1, "hi": 20, "step": 0.25, "dec": 2}),
              ("dash", "choice", "Dash", {"choices": [("solid", "solid"), ("dashed", "dashed"),
                                                      ("dotted", "dotted")]})],
    "arrow": [("color", "color", "Colour", {}),
              ("width_pt", "float", "Line (pt)", {"lo": 0.1, "hi": 20, "step": 0.25, "dec": 2})],
    "line": [("color", "color", "Colour", {}),
             ("width_pt", "float", "Line (pt)", {"lo": 0.1, "hi": 20, "step": 0.25, "dec": 2})],
    "measure_line": [("of.item", "item", "Of image", {"kinds": ("image", "montage")}),
                     ("color", "color", "Colour", {}),
                     ("width_pt", "float", "Line (pt)", {"lo": 0.1, "hi": 20, "step": 0.25,
                                                         "dec": 2})],
    "table": [("table.source", "chart_source", "Reads", {}),
              ("table.table", "chart_table", "Table", {}),
              ("table.columns", "list", "Columns (blank = all)", {}),
              ("table.group_by", "column", "Group by", {}),
              ("table.stats", "list", "Statistics", {"hint": "n, mean, sd, sem, median, min, "
                                                             "max, sum"}),
              ("table.decimals", "int", "Decimals (-1 = auto)", {"lo": -1, "hi": 8}),
              ("pt", "float", "Size (pt, 0 = page)", {"lo": 0, "hi": 72, "step": 0.5, "dec": 1})],
    "legend": [("of.item", "item", "Of image", {"kinds": ("image", "montage")}),
               ("title", "text", "Title", {})],
}

CHART_KIND_LABELS = [("xy", "XY — scatter / line"), ("timeseries", "time series"),
                     ("bar", "bars"), ("distribution", "distribution"),
                     ("heatmap", "heat map"), ("hexbin", "hexbin density"),
                     ("msd", "MSD (tracks)"), ("survival", "survival (tracks)"),
                     ("rose", "rose (angles)")]

CMAP_CHOICES = [("viridis", "viridis"), ("magma", "magma"), ("turbo", "turbo"),
                ("coolwarm", "coolwarm"), ("grey", "grey")]

#: the chart form (step 4): the data it reads, then how it is drawn. ``column`` controls
#: list the columns of the table the chart reads (``panel.chart_columns``); ``chart_source``
#: lists the sources and the data entries; ``chart_table`` the tables of what it reads.
CHART_FIELDS: List[Field] = [
    ("chart.kind", "choice", "Kind", {"choices": CHART_KIND_LABELS}),
    ("chart.source", "chart_source", "Reads", {}),
    ("chart.table", "chart_table", "Table", {}),
    ("chart.x", "column", "X", {}),
    ("chart.y", "column", "Y", {}),
    ("chart.value", "column", "Value", {}),
    ("chart.group_by", "column", "Group by", {}),
    ("chart.color_by", "column", "Colour / series by", {}),
    ("chart.facet_by", "column", "Replicate (SuperPlot)", {}),
    ("chart.sub_kind", "choice", "Distribution as", {"choices": [(k, k) for k in
                                                                 S.DISTRIBUTION_KINDS]}),
    ("chart.bins", "int", "Bins / lags / grid", {"lo": 1, "hi": 10000}),
    ("chart.draw.mode", "choice", "Draw", {"choices": [("scatter", "markers"), ("line", "line"),
                                                       ("line_markers", "line + markers")]}),
    ("chart.draw.error", "choice", "Error", {"choices": [("none", "none"), ("sem", "SEM"),
                                                         ("sd", "SD"), ("ci95", "95 % CI"),
                                                         ("column", "a column")]}),
    ("chart.error_column", "column", "Error column", {}),
    ("chart.draw.error_style", "choice", "Error as", {"choices": [("bars", "bars"),
                                                                  ("band", "band")]}),
    ("chart.draw.step", "bool", "Step", {}),
    ("chart.draw.stacked", "bool", "Stacked bars", {}),
    ("chart.draw.reducer", "choice", "Cell reducer (heat map)", {"choices": [
        ("mean", "mean"), ("median", "median"), ("sum", "sum"), ("count", "count"),
        ("min", "min"), ("max", "max")]}),
    ("chart.fit.model", "choice", "Fit", {"choices": [("none", "none"), ("linear", "linear"),
                                                      ("power", "power law"),
                                                      ("exponential", "exponential"),
                                                      ("langmuir", "Langmuir")]}),
    ("chart.stats.test", "choice", "Test", {"choices": [("none", "none")]
                                            + [(t, t) for t in ST.TESTS]}),
    ("chart.axes.title", "text", "Title", {}),
    ("chart.axes.x_label", "text", "X label", {}),
    ("chart.axes.y_label", "text", "Y label", {}),
    ("chart.axes.log_x", "bool", "Log X", {}),
    ("chart.axes.log_y", "bool", "Log Y", {}),
    ("chart.axes.x_range", "range", "X range", {}),
    ("chart.axes.y_range", "range", "Y range", {}),
    ("chart.legend.show", "bool", "Legend", {}),
    ("chart.legend.loc", "choice", "Legend at", {"choices": [("inside", "inside"),
                                                             ("right", "right"),
                                                             ("bottom", "bottom")]}),
    ("chart.cmap", "choice", "Colour map", {"choices": CMAP_CHOICES}),
]

#: which chart fields each kind reads (the others are hidden)
CHART_FIELDS_OF: Dict[str, Tuple[str, ...]] = {
    "xy": ("x", "y", "group_by", "draw.mode", "draw.error", "error_column", "draw.error_style",
           "draw.step", "fit.model", "axes.log_x", "axes.log_y", "axes.x_range", "axes.y_range",
           "legend.show", "legend.loc"),
    "timeseries": ("y", "group_by", "draw.mode", "draw.error", "error_column",
                   "draw.error_style", "draw.step", "fit.model", "axes.log_y", "axes.y_range",
                   "legend.show", "legend.loc"),
    "bar": ("x", "y", "color_by", "draw.error", "draw.stacked", "axes.log_y", "axes.y_range",
            "legend.show", "legend.loc"),
    "distribution": ("value", "group_by", "facet_by", "sub_kind", "bins", "stats.test",
                     "axes.log_x", "axes.log_y", "axes.x_range", "axes.y_range", "legend.show",
                     "legend.loc"),
    "heatmap": ("x", "group_by", "value", "draw.reducer", "cmap"),
    "hexbin": ("x", "y", "bins", "axes.log_x", "axes.log_y", "cmap"),
    "msd": ("group_by", "bins", "axes.log_x", "axes.log_y", "legend.show", "legend.loc"),
    "survival": ("group_by", "legend.show", "legend.loc"),
    "rose": ("value", "group_by", "bins", "legend.show", "legend.loc"),
}

CHANNEL_MODES = [("auto_pct", "auto (percentiles)"), ("viewer", "the Viewer's window"),
                 ("manual", "typed"), ("shared", "shared (LUT link)")]

LAYER_LOOKS: Dict[str, List[Field]] = {
    "labels": [("look.outline", "bool", "Outline", {}), ("look.fill", "bool", "Fill", {}),
               ("look.ids", "bool", "Ids", {}),
               ("look.line_width", "float", "Line (pt)", {"lo": 0.1, "hi": 10, "step": 0.1,
                                                          "dec": 2}),
               ("look.color_mode", "choice", "Colour", {"choices": [("per_label", "per object"),
                                                                     ("single", "one colour")]}),
               ("look.color", "color", "One colour", {}),
               ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
    "points": [("look.shape", "choice", "Marker", {"choices": [("star", "star"), ("cross", "cross"),
                                                               ("circle", "dot"), ("ring", "ring"),
                                                               ("square", "square")]}),
               ("look.spread", "int", "Size", {"lo": 1, "hi": 5}),
               ("look.color_mode", "choice", "Colour", {"choices": [("single", "one colour"),
                                                                     ("per_z", "per plane"),
                                                                     ("per_point", "per point"),
                                                                     ("per_layer", "per layer")]}),
               ("look.color", "color", "One colour", {}),
               ("look.z_project", "bool", "Project every plane", {}),
               ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
    "tracks": [("look.tail", "int", "Tail (frames, 0 = all)", {"lo": 0, "hi": 10000}),
               ("look.line_width", "float", "Line (pt)", {"lo": 0.1, "hi": 10, "step": 0.1,
                                                          "dec": 2}),
               ("look.show_ids", "bool", "Ids", {}),
               ("look.fade", "bool", "Fade", {}),
               ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
    "field": [("column", "column", "Column", {}),
              ("look.style", "choice", "Style", {"choices": [("smooth", "smooth"),
                                                             ("cells", "cells"),
                                                             ("contours", "contours")]}),
              ("cmap", "choice", "Colour map", {"choices": [("auto", "auto"), ("turbo", "turbo"),
                                                            ("viridis", "viridis"),
                                                            ("magma", "magma"),
                                                            ("coolwarm", "coolwarm"),
                                                            ("grey", "grey")]}),
              ("limits.mode", "choice", "Limits", {"choices": [("series", "series 1–99 %"),
                                                               ("frame", "this frame"),
                                                               ("manual", "typed"),
                                                               ("normalized", "0–1 of a maximum")]}),
              ("limits.lo", "float", "Low", {"lo": -1e9, "hi": 1e9, "step": 0.1, "dec": 4}),
              ("limits.hi", "float", "High", {"lo": -1e9, "hi": 1e9, "step": 0.1, "dec": 4}),
              ("limits.vmax", "float", "Maximum (for 0–1)", {"lo": 0, "hi": 1e9, "step": 0.1,
                                                             "dec": 4}),
              ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
    "vectors": [("look.every", "int", "Every Nth", {"lo": 1, "hi": 100}),
                ("look.auto_scale", "bool", "Auto length", {}),
                ("look.scale", "float", "Length ×", {"lo": 0.01, "hi": 100, "step": 0.1, "dec": 2}),
                ("look.color_mode", "choice", "Colour", {"choices": [("magnitude", "by length"),
                                                                      ("flat", "one colour")]}),
                ("look.color", "color", "One colour", {}),
                ("look.line_width", "float", "Line (pt)", {"lo": 0.1, "hi": 10, "step": 0.1,
                                                           "dec": 2}),
                ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
    "voxels": [("look.style", "choice", "Style", {"choices": [("heatmap", "heat map"),
                                                              ("mask", "mask"),
                                                              ("contour", "contours")]}),
               ("cmap", "choice", "Colour map", {"choices": [("viridis", "viridis"),
                                                             ("magma", "magma"), ("turbo", "turbo"),
                                                             ("coolwarm", "coolwarm"),
                                                             ("grey", "grey")]}),
               ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
    "mesh": [("look.style", "choice", "Style", {"choices": [("wireframe", "wireframe"),
                                                            ("surface", "surface"),
                                                            ("points", "vertices")]}),
             ("look.line_width", "float", "Line (pt)", {"lo": 0.1, "hi": 10, "step": 0.1,
                                                        "dec": 2}),
             ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
    "grid": [("look.style", "choice", "Style", {"choices": [("elements", "elements"),
                                                            ("subsets", "subsets"),
                                                            ("deformed", "deformed")]}),
             ("look.line_width", "float", "Line (pt)", {"lo": 0.1, "hi": 10, "step": 0.1,
                                                        "dec": 2}),
             ("opacity", "int", "Opacity (%)", {"lo": 0, "hi": 100})],
}


def get_path(d: Mapping[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, Mapping):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
    return cur


def set_path(d: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = d
    for part in parts[:-1]:
        if isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            if part not in cur or not isinstance(cur[part], (dict, list)):
                cur[part] = {}
            cur = cur[part]
    last = parts[-1]
    if isinstance(cur, list) and last.isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value


class PropertiesPanel(QScrollArea):
    """The form. ``panel`` supplies the document, the selection and the edit entry points
    (``document``, ``selected_ids()``, ``edit_item``, ``edit_page``, ``add_channel``,
    ``remove_channel``, ``add_layer``, ``remove_layer``, ``source_layers``, ``axes_of``)."""

    def __init__(self, panel: Any) -> None:
        super().__init__()
        self._panel = panel
        self.setObjectName("figureProps")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self._body = QWidget()
        self._body.setObjectName("figurePropsBody")
        self._lay = QVBoxLayout(self._body)
        self._lay.setContentsMargins(8, 6, 8, 6)
        self._lay.setSpacing(6)
        self.setWidget(self._body)
        self._filling = False
        self.restyle()

    # ── building ───────────────────────────────────────────────────────────────
    def rebuild(self) -> None:
        self._filling = True
        try:
            while self._lay.count():
                w = self._lay.takeAt(0).widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
            ids = list(self._panel.selected_ids())
            doc = self._panel.document
            if not ids:
                self._build_page(doc)
            elif len(ids) == 1:
                it = next((i for i in doc.get("items", []) if i["id"] == ids[0]), None)
                if it is not None:
                    self._build_item(doc, it)
            else:
                self._build_multi(doc, ids)
            self._lay.addStretch(1)
        finally:
            self._filling = False

    def _title(self, text: str) -> None:
        lab = QLabel(text)
        lab.setProperty("role", "title")
        self._lay.addWidget(lab)

    def _form(self, title: str = "") -> QFormLayout:
        box = QGroupBox(title) if title else QWidget()
        form = QFormLayout(box)
        form.setContentsMargins(6, 6 if title else 0, 6, 6)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignRight)
        self._lay.addWidget(box)
        return form

    def _build_page(self, doc: Mapping[str, Any]) -> None:
        self._title("Page")
        page = doc.get("page") or {}
        form = self._form()
        for path, kind, label, opt in PAGE_FIELDS:
            w = self._control(kind, get_path(page, path), opt,
                              lambda v, p=path: self._panel.edit_page(p, v), ctx=None)
            form.addRow(label, w)
        goal = doc.get("goal") or ""
        lab = QLabel(f"Goal: {goal or 'not chosen'}")
        lab.setProperty("role", "muted")
        self._lay.addWidget(lab)
        row = QHBoxLayout()
        b = QPushButton("Re-letter panels")
        b.setToolTip("Letter every panel in reading order (rows top to bottom, left to "
                     "right); a popout is part of its parent's panel")
        b.clicked.connect(lambda _=False: self._panel.reletter())
        row.addWidget(b)
        b = QPushButton("Style sync")
        b.setToolTip("Apply the page preset (paper / talk / poster / dark) to the page and "
                     "clear every item's own type size and line width back to it")
        b.clicked.connect(lambda _=False: self._panel.sync_style())
        row.addWidget(b)
        self._lay.addLayout(row)
        self._build_checks()

    def _build_checks(self) -> None:
        """The paper-readiness checks of the page, each with a jump to its item."""
        checks = list(self._panel.checks())
        box = QGroupBox(f"Checks ({len(checks)})" if checks else "Checks")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(3)
        if not checks:
            lab = QLabel("nothing to flag — scale bars, letters, colour bars, merges and "
                         "display ranges all pass")
            lab.setWordWrap(True)
            lab.setProperty("role", "muted")
            lay.addWidget(lab)
        for c in checks:
            row = QHBoxLayout()
            sev = "⚠" if c.get("severity") == "warn" else "·"
            lab = QLabel(f"{sev} {c.get('message')}")
            lab.setWordWrap(True)
            lab.setToolTip(str(c.get("code") or ""))
            row.addWidget(lab, 1)
            if c.get("item"):
                go = QToolButton()
                go.setText(str(c["item"]))
                go.setToolTip("select this item")
                go.clicked.connect(lambda _=False, i=str(c["item"]): self._panel.select([i]))
                row.addWidget(go)
            lay.addLayout(row)
        self._lay.addWidget(box)

    def _build_multi(self, doc: Mapping[str, Any], ids: Sequence[str]) -> None:
        self._title(f"{len(ids)} items")
        row = QHBoxLayout()
        for text, mode in (("Left", "left"), ("Top", "top"), ("Right", "right"),
                           ("Bottom", "bottom")):
            b = QPushButton(f"Align {text.lower()}")
            b.clicked.connect(lambda _=False, m=mode: self._panel.align_selected(m))
            row.addWidget(b)
        self._lay.addLayout(row)
        row2 = QHBoxLayout()
        for text, axis in (("Distribute ↔", "x"), ("Distribute ↕", "y")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, a=axis: self._panel.distribute_selected(a))
            row2.addWidget(b)
        b = QPushButton("Share display range")
        b.setToolTip("One window per channel across these images (a LUT-share link)")
        b.clicked.connect(lambda _=False: self._panel.share_lut(list(ids)))
        row2.addWidget(b)
        self._lay.addLayout(row2)

    def _build_item(self, doc: Mapping[str, Any], it: Mapping[str, Any]) -> None:
        kind = str(it.get("kind"))
        iid = str(it["id"])
        self._title(f"{kind}  ·  {iid}")
        form = self._form()
        for path, fkind, label, opt in COMMON_FIELDS:
            w = self._control(fkind, get_path(it, path), opt,
                              lambda v, p=path: self._panel.edit_item(iid, p, v), ctx=it)
            form.addRow(label, w)
        if kind in ("image", "montage"):
            img = S.effective_image(doc, iid)
            twin = S.twin_parent(doc, iid)
            if twin:
                lab = QLabel(f"Twin of {twin[0]}: shares {', '.join(twin[1])}")
                lab.setProperty("role", "muted")
                self._lay.addWidget(lab)
            form = self._form("Image")
            for path, fkind, label, opt in IMAGE_FIELDS:
                if twin and path.split(".")[1] in twin[1]:
                    continue
                w = self._control(fkind, get_path(it, path) if path.split(".")[1] not in
                                  ("frame", "decor") else get_path({"image": img}, path), opt,
                                  lambda v, p=path: self._panel.edit_item(iid, p, v), ctx=it)
                form.addRow(label, w)
            if kind == "montage":
                form = self._form("Montage")
                for path, fkind, label, opt in MONTAGE_FIELDS:
                    w = self._control(fkind, get_path(it, path), opt,
                                      lambda v, p=path: self._panel.edit_item(iid, p, v), ctx=it)
                    form.addRow(label, w)
            if not (twin and "channels" in twin[1]):
                self._build_channels(it, img)
            if not (twin and "layers" in twin[1]):
                self._build_layers(it, img)
            self._build_links(doc, it)
        elif kind == "chart":
            self._build_chart(doc, it)
            self._build_links(doc, it)
        else:
            for path, fkind, label, opt in KIND_FIELDS.get(kind, []):
                w = self._control(fkind, get_path(it, path), opt,
                                  lambda v, p=path: self._panel.edit_item(iid, p, v), ctx=it)
                form.addRow(label, w)
        row = QHBoxLayout()
        if kind in ("image", "montage"):
            b = QPushButton("Twin")
            b.setToolTip("A second copy sharing this image's region, frame and channels — "
                         "give it other layers")
            b.clicked.connect(lambda _=False: self._panel.add_twin(iid))
            row.addWidget(b)
            b = QPushButton("Colour bar")
            b.clicked.connect(lambda _=False: self._panel.add_colorbar_for(iid))
            row.addWidget(b)
            b = QPushButton("Histogram")
            b.clicked.connect(lambda _=False: self._panel.add_histogram_for(iid))
            row.addWidget(b)
        if kind in ("image", "montage"):
            b = QPushButton("Legend")
            b.setToolTip("A legend of this image's channels and layers")
            b.clicked.connect(lambda _=False: self._panel.add_legend_for(iid))
            row.addWidget(b)
        if kind == "table":
            b = QPushButton("Fit to content")
            b.clicked.connect(lambda _=False: self._panel.fit_table(iid))
            row.addWidget(b)
        if kind == "chart" and not (it.get("chart") or {}).get("data"):
            b = QPushButton("Filter / summarise")
            b.setToolTip("Give the chart a data entry of its source and table — where a "
                         "filter and a per-group summary are typed; it compiles into the "
                         "page's hidden graph")
            b.clicked.connect(lambda _=False: self._panel.promote_chart_data(iid))
            row.addWidget(b)
        b = QPushButton("Delete")
        b.clicked.connect(lambda _=False: self._panel.delete_items([iid]))
        row.addWidget(b)
        self._lay.addLayout(row)

    def _build_channels(self, it: Mapping[str, Any], img: Mapping[str, Any]) -> None:
        iid = str(it["id"])
        box = QGroupBox("Channels")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        names = list(img.get("_channel_names") or [])
        for k, ch in enumerate(img.get("channels") or []):
            form = QFormLayout()
            form.setSpacing(3)
            head = QHBoxLayout()
            c = int(ch.get("c", 0))
            nm = names[c] if c < len(names) and names[c] else f"ch{c + 1}"
            lab = QLabel(f"{nm}")
            lab.setProperty("role", "title")
            head.addWidget(lab, 1)
            rm = QToolButton()
            rm.setText("✕")
            rm.clicked.connect(lambda _=False, k=k: self._panel.remove_channel(iid, k))
            head.addWidget(rm)
            lay.addLayout(head)
            spin = _NoWheelInt()
            spin.setRange(0, 63)
            spin.setValue(c)
            spin.editingFinished.connect(
                lambda s=spin, k=k: self._panel.edit_item(iid, f"image.channels.{k}.c", s.value()))
            form.addRow("Channel", spin)
            col = ColorButton(str(ch.get("rgb", "#ffffff")))
            col.picked.connect(lambda v, k=k: self._panel.edit_item(iid, f"image.channels.{k}.rgb", v))
            form.addRow("Colour", col)
            mode = _NoWheelCombo()
            for key, text in CHANNEL_MODES:
                mode.addItem(text, key)
            mode.setCurrentIndex(max(0, [m for m, _ in CHANNEL_MODES].index(
                str(ch.get("mode", "auto_pct")))))
            mode.currentIndexChanged.connect(
                lambda i, m=mode, k=k: self._panel.edit_item(iid, f"image.channels.{k}.mode",
                                                              m.itemData(i)))
            form.addRow("Window", mode)
            for key, label, lo, hi in (("lo", "Black", -1e9, 1e9), ("hi", "White", -1e9, 1e9),
                                       ("gamma", "Gamma", 0.05, 20)):
                sp = _NoWheelSpin()
                sp.setRange(lo, hi)
                sp.setDecimals(3)
                v = ch.get(key)
                sp.setValue(float(v) if v is not None else (1.0 if key == "gamma" else 0.0))
                sp.editingFinished.connect(
                    lambda s=sp, k=k, key=key: self._panel.edit_item(
                        iid, f"image.channels.{k}.{key}", float(s.value())))
                form.addRow(label, sp)
            lab_e = QLineEdit(str(ch.get("label") or ""))
            lab_e.editingFinished.connect(
                lambda e=lab_e, k=k: self._panel.edit_item(iid, f"image.channels.{k}.label",
                                                           e.text()))
            form.addRow("Label", lab_e)
            lay.addLayout(form)
        add = QPushButton("+ channel")
        add.clicked.connect(lambda _=False: self._panel.add_channel(iid))
        lay.addWidget(add)
        self._lay.addWidget(box)

    def _build_layers(self, it: Mapping[str, Any], img: Mapping[str, Any]) -> None:
        iid = str(it["id"])
        box = QGroupBox("Layers")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)
        avail = self._panel.source_layers(img.get("source"))   # [(domain, layer, kind, cols)]
        for k, L in enumerate(img.get("layers") or []):
            head = QHBoxLayout()
            lab = QLabel(f"{L.get('kind')}: {L.get('layer') or '(auto)'}")
            lab.setProperty("role", "title")
            head.addWidget(lab, 1)
            rm = QToolButton()
            rm.setText("✕")
            rm.clicked.connect(lambda _=False, k=k: self._panel.remove_layer(iid, k))
            head.addWidget(rm)
            lay.addLayout(head)
            form = QFormLayout()
            form.setSpacing(3)
            cols = next((c for d, l, kd, c in avail if l == L.get("layer")), [])
            for path, fkind, label, opt in LAYER_LOOKS.get(str(L.get("kind")), []):
                opt = dict(opt)
                if fkind == "column":
                    opt["choices"] = [("", "auto")] + [(c, c) for c in cols]
                    fkind = "choice"
                w = self._control(fkind, get_path(L, path), opt,
                                  lambda v, p=path, k=k: self._panel.edit_item(
                                      iid, f"image.layers.{k}.{p}", v), ctx=it)
                form.addRow(label, w)
            lay.addLayout(form)
        if avail:
            row = QHBoxLayout()
            combo = _NoWheelCombo()
            for d, l, kd, c in avail:
                combo.addItem(f"{kd}: {l}", (d, l, kd))
            row.addWidget(combo, 1)
            add = QPushButton("+ layer")
            add.clicked.connect(lambda _=False, cb=combo: self._panel.add_layer(
                iid, *cb.currentData()))
            row.addWidget(add)
            lay.addLayout(row)
        else:
            lab = QLabel("the source carries no tables to draw")
            lab.setProperty("role", "muted")
            lay.addWidget(lab)
        self._lay.addWidget(box)

    def _build_chart(self, doc: Mapping[str, Any], it: Mapping[str, Any]) -> None:
        iid = str(it["id"])
        chart = it.get("chart") or {}
        kind = str(chart.get("kind") or "xy")
        shown = set(CHART_FIELDS_OF.get(kind, ())) | {"kind", "source", "table", "axes.title",
                                                     "axes.x_label", "axes.y_label"}
        form = self._form("Chart")
        for path, fkind, label, opt in CHART_FIELDS:
            if path[len("chart."):] not in shown:
                continue
            w = self._control(fkind, get_path(it, path), opt,
                              lambda v, p=path: self._panel.edit_item(iid, p, v), ctx=it)
            form.addRow(label, w)
        did = chart.get("data")
        if did:
            entry = next((d for d in doc.get("data", []) if d["id"] == did), None)
            if entry is not None:
                self._build_data(doc, entry)
        note = getattr(self._panel._items.get(iid), "note", "") if hasattr(self._panel, "_items") else ""
        if note:
            lab = QLabel(note)
            lab.setWordWrap(True)
            lab.setProperty("role", "muted")
            self._lay.addWidget(lab)

    def _build_data(self, doc: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
        """The data entry a chart reads: its sources, filter and summary. Every control
        commits one edit through ``panel.edit_data``."""
        did = str(entry["id"])
        box = QGroupBox(f"Data entry {did}")
        form = QFormLayout(box)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignRight)
        labels = {s["id"]: (s.get("label") or s["source"]) for s in doc.get("sources", [])}
        form.addRow("Pools", QLabel(", ".join(str(labels.get(s, s)) for s in entry.get("from", []))))
        dom = _NoWheelCombo()
        for k in ("label", "point", "track"):
            dom.addItem(k, k)
        dom.setCurrentIndex(max(0, ["label", "point", "track"].index(str(entry.get("domain") or "label"))))
        dom.currentIndexChanged.connect(lambda i, c=dom: self._panel.edit_data(did, "domain", c.itemData(i)))
        form.addRow("Domain", dom)
        tab = QLineEdit(str(entry.get("table") or ""))
        tab.setPlaceholderText("the only table on the domain")
        tab.editingFinished.connect(lambda e=tab: self._panel.edit_data(did, "table", e.text()))
        form.addRow("Table", tab)
        where = QLineEdit("; ".join(f"{c} {o} {v:g}" for c, o, v in entry.get("where", [])))
        where.setPlaceholderText("area > 50; circularity >= 0.8   (up to four)")
        where.setToolTip("Rows kept: `column op value` terms separated by `;` — ops " +
                         ", ".join(S.WHERE_OPS))
        where.editingFinished.connect(lambda e=where: self._panel.edit_data(did, "where", _parse_where(e.text())))
        form.addRow("Keep rows where", where)
        match = _NoWheelCombo()
        match.addItem("all terms", "all")
        match.addItem("any term", "any")
        match.setCurrentIndex(0 if str(entry.get("match") or "all") == "all" else 1)
        match.currentIndexChanged.connect(lambda i, c=match: self._panel.edit_data(did, "match", c.itemData(i)))
        form.addRow("Match", match)
        agg = entry.get("aggregate")
        on = QCheckBox("one row per group")
        on.setChecked(agg is not None)
        on.toggled.connect(lambda v: self._panel.edit_data(
            did, "aggregate", {"group_by": ["t"], "value": "area",
                               "reducers": ["mean", "sem", "count"], "name": "summary"}
            if v else None))
        form.addRow("Summarise", on)
        if agg is not None:
            gb = agg.get("group_by") or []
            gb_e = QLineEdit(", ".join(gb) if isinstance(gb, list) else str(gb))
            gb_e.setPlaceholderText("t   or   condition, t   (blank = one row)")
            gb_e.editingFinished.connect(lambda e=gb_e: self._panel.edit_data(
                did, "aggregate.group_by", [g.strip() for g in e.text().split(",") if g.strip()]))
            form.addRow("Group by", gb_e)
            cols = self._panel.chart_columns({"chart": {"source": (entry.get("from") or [""])[0],
                                                        "domain": entry.get("domain"),
                                                        "table": entry.get("table")}})
            val = _NoWheelCombo()
            for c in cols:
                val.addItem(c, c)
            idx = next((i for i in range(val.count()) if val.itemData(i) == agg.get("value")), -1)
            if idx < 0 and agg.get("value"):
                val.addItem(str(agg["value"]), str(agg["value"]))
                idx = val.count() - 1
            val.setCurrentIndex(max(0, idx))
            val.currentIndexChanged.connect(lambda i, c=val: self._panel.edit_data(
                did, "aggregate.value", c.itemData(i)))
            form.addRow("Value", val)
            red = agg.get("reducers") or []
            red_e = QLineEdit(", ".join(red) if isinstance(red, list) else str(red))
            red_e.setToolTip("of " + ", ".join(S.AGG_REDUCERS))
            red_e.editingFinished.connect(lambda e=red_e: self._panel.edit_data(
                did, "aggregate.reducers", [r.strip() for r in e.text().split(",") if r.strip()]))
            form.addRow("Reducers", red_e)
            nm = QLineEdit(str(agg.get("name") or "summary"))
            nm.editingFinished.connect(lambda e=nm: self._panel.edit_data(did, "aggregate.name", e.text()))
            form.addRow("Summary table", nm)
        self._lay.addWidget(box)

    def _build_links(self, doc: Mapping[str, Any], it: Mapping[str, Any]) -> None:
        iid = str(it["id"])
        rows = []
        for ln in S.links_of(doc, iid):
            k = ln.get("kind")
            if k == "zoom":
                rows.append(("zoom", f"{ln['parent']} → {ln['child']}", ln))
            elif k == "twin":
                rows.append(("twin", f"{ln['a']} ↔ {ln['b']}", ln))
            elif k == "lut_share":
                rows.append(("display range", ", ".join(ln.get("items") or []), ln))
            elif k == "cursor":
                rows.append(("cursor", ", ".join(ln.get("items") or []), ln))
        if not rows:
            return
        box = QGroupBox("Links")
        lay = QVBoxLayout(box)
        for kind, text, ln in rows:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{kind}: {text}"), 1)
            rm = QToolButton()
            rm.setText("✕")
            rm.clicked.connect(lambda _=False, l=ln: self._panel.remove_link(l))
            row.addWidget(rm)
            lay.addLayout(row)
        self._lay.addWidget(box)

    # ── controls ───────────────────────────────────────────────────────────────
    def _control(self, kind: str, value: Any, opt: Mapping[str, Any],
                 on_set: Callable[[Any], None], *, ctx: Any) -> QWidget:
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v: on_set(bool(v)))
            return w
        if kind == "int":
            w = _NoWheelInt()
            w.setRange(int(opt.get("lo", -1e6)), int(opt.get("hi", 1e6)))
            w.setValue(int(value or 0))
            w.editingFinished.connect(lambda s=w: on_set(int(s.value())))
            return w
        if kind == "float":
            w = _NoWheelSpin()
            w.setRange(float(opt.get("lo", -1e9)), float(opt.get("hi", 1e9)))
            w.setDecimals(int(opt.get("dec", 2)))
            w.setSingleStep(float(opt.get("step", 1.0)))
            w.setValue(float(value) if value is not None else 0.0)
            w.editingFinished.connect(lambda s=w: on_set(float(s.value())))
            return w
        if kind == "choice":
            w = _NoWheelCombo()
            for key, text in opt.get("choices", []):
                w.addItem(str(text), key)
            idx = next((i for i in range(w.count()) if w.itemData(i) == value), -1)
            if idx < 0 and value is not None:
                w.addItem(str(value), value)
                idx = w.count() - 1
            w.setCurrentIndex(max(0, idx))
            w.currentIndexChanged.connect(lambda i, c=w: on_set(c.itemData(i)))
            return w
        if kind == "color":
            w = ColorButton(str(value or "#ffffff"))
            w.picked.connect(on_set)
            return w
        if kind in ("text", "multiline"):
            w = QLineEdit(str(value or ""))
            w.editingFinished.connect(lambda e=w: on_set(e.text()))
            return w
        if kind == "source":
            w = _NoWheelCombo()
            for s in self._panel.document.get("sources", []):
                w.addItem(f"{s.get('label') or s['source']}", s["id"])
            idx = next((i for i in range(w.count()) if w.itemData(i) == value), -1)
            w.setCurrentIndex(max(0, idx))
            w.currentIndexChanged.connect(lambda i, c=w: on_set(c.itemData(i)))
            return w
        if kind == "item":
            w = _NoWheelCombo()
            for it in self._panel.document.get("items", []):
                if it.get("kind") in opt.get("kinds", ()):
                    w.addItem(f"{it.get('letter') or ''} {it['id']}".strip(), it["id"])
            idx = next((i for i in range(w.count()) if w.itemData(i) == value), -1)
            w.setCurrentIndex(max(0, idx))
            w.currentIndexChanged.connect(lambda i, c=w: on_set(c.itemData(i)))
            return w
        if kind == "column":
            w = _NoWheelCombo()
            w.addItem("—", "")
            for c in (self._panel.chart_columns(ctx) if ctx is not None else []):
                w.addItem(str(c), str(c))
            idx = next((i for i in range(w.count()) if w.itemData(i) == (value or "")), -1)
            if idx < 0 and value:
                w.addItem(str(value), str(value))
                idx = w.count() - 1
            w.setCurrentIndex(max(0, idx))
            w.currentIndexChanged.connect(lambda i, c=w: on_set(c.itemData(i)))
            return w
        if kind == "list":
            vals = value if isinstance(value, (list, tuple)) else ([value] if value else [])
            w = QLineEdit(", ".join(str(v) for v in vals))
            if opt.get("hint"):
                w.setPlaceholderText(str(opt["hint"]))
            w.editingFinished.connect(
                lambda e=w: on_set([s.strip() for s in e.text().split(",") if s.strip()]))
            return w
        if kind == "chart_source":
            w = _NoWheelCombo()
            doc = self._panel.document
            for s in doc.get("sources", []):
                w.addItem(f"{s.get('label') or s['source']}", ("source", s["id"]))
            for d in doc.get("data", []):
                w.addItem(f"data: {d['id']}", ("data", d["id"]))
            chart = (ctx or {}).get("chart") or (ctx or {}).get("table") or {}
            cur = ("data", chart["data"]) if chart.get("data") else ("source", chart.get("source"))
            idx = next((i for i in range(w.count()) if w.itemData(i) == cur), -1)
            w.setCurrentIndex(max(0, idx))
            iid = str((ctx or {}).get("id") or "")
            w.currentIndexChanged.connect(
                lambda i, c=w: self._panel.set_chart_reads(iid, *c.itemData(i)))
            return w
        if kind == "chart_table":
            w = _NoWheelCombo()
            chart = (ctx or {}).get("chart") or (ctx or {}).get("table") or {}
            w.addItem("the only table on the domain", (str(chart.get("domain") or "label"), ""))
            for dom, name in (self._panel.chart_tables_of(ctx) if ctx is not None else []):
                w.addItem(f"{dom}: {name}", (dom, name))
            cur = (str(chart.get("domain") or "label"), str(chart.get("table") or ""))
            idx = next((i for i in range(w.count()) if w.itemData(i) == cur), -1)
            if idx < 0 and cur[1]:
                w.addItem(f"{cur[0]}: {cur[1]}", cur)
                idx = w.count() - 1
            w.setCurrentIndex(max(0, idx))
            iid = str((ctx or {}).get("id") or "")
            w.currentIndexChanged.connect(
                lambda i, c=w: self._panel.set_chart_table(iid, *c.itemData(i)))
            return w
        if kind == "range":
            lo, hi = (value or [None, None])[:2] if isinstance(value, (list, tuple)) else (None, None)
            w = QLineEdit(", ".join("" if v is None else f"{v:g}" for v in (lo, hi)).strip(", "))
            w.setPlaceholderText("low, high   (blank = auto)")
            w.editingFinished.connect(lambda e=w: on_set(_parse_range(e.text())))
            return w
        if kind == "frame":
            axis = str(opt.get("axis"))
            w = _NoWheelCombo()
            w.setEditable(False)
            n = int((self._panel.axes_of(ctx) or {}).get(axis, 1) or 1)
            w.addItem("page cursor", "cursor")
            if axis == "z":
                w.addItem("max projection", "max")
                w.addItem("mean projection", "mean")
            for i in range(n):
                w.addItem(f"{axis} = {i}", i)
            idx = next((i for i in range(w.count()) if w.itemData(i) == value), -1)
            w.setCurrentIndex(max(0, idx))
            w.currentIndexChanged.connect(lambda i, c=w: on_set(c.itemData(i)))
            return w
        w = QLineEdit(str(value or ""))
        w.editingFinished.connect(lambda e=w: on_set(e.text()))
        return w

    def restyle(self) -> None:
        def h(c) -> str:
            return c.name()
        self.setStyleSheet(f"""
            QScrollArea#figureProps, QWidget#figurePropsBody {{ background:{h(T.PANEL)}; }}
            QLabel {{ color:{h(T.INK)}; }}
            QLabel[role="title"] {{ font-weight:700; }}
            QLabel[role="muted"] {{ color:{h(T.MUTED)}; font-size:10px; }}
            QGroupBox {{ color:{h(T.MUTED)}; border:1px solid {h(T.BORDER)}; border-radius:6px;
                margin-top:8px; padding-top:6px; font-weight:600; }}
            QGroupBox::title {{ subcontrol-origin: margin; left:8px; padding:0 3px; }}
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{ background:{h(T.BODY)};
                color:{h(T.INK)}; border:1px solid {h(T.BORDER)}; border-radius:5px;
                padding:2px 5px; min-height:16px; }}
            QCheckBox {{ color:{h(T.INK)}; }}
            QPushButton, QToolButton {{ background:{h(T.BODY)}; color:{h(T.INK)};
                border:1px solid {h(T.BORDER)}; border-radius:5px; padding:3px 8px; }}
            QPushButton:hover, QToolButton:hover {{ border-color:{h(T.ACCENT)}; }}
        """)


def _parse_range(text: str) -> List[Optional[float]]:
    """``"0, 100"`` → ``[0.0, 100.0]``; a blank side is ``None``."""
    parts = [p.strip() for p in str(text or "").replace(";", ",").split(",")]
    parts = (parts + ["", ""])[:2]
    out: List[Optional[float]] = []
    for p in parts:
        try:
            out.append(float(p) if p else None)
        except ValueError:
            out.append(None)
    if out[0] is not None and out[1] is not None and out[1] <= out[0]:
        out[1] = None
    return out


def _parse_where(text: str) -> List[List[Any]]:
    """``"area > 50; circ >= 0.8"`` → ``[["area", ">", 50.0], ["circ", ">=", 0.8]]``; a
    term that does not parse is dropped, at most four are kept."""
    ops = sorted(S.WHERE_OPS, key=len, reverse=True)
    out: List[List[Any]] = []
    for term in str(text or "").split(";"):
        term = term.strip()
        if not term:
            continue
        toks = term.split()
        if len(toks) == 2 and toks[1] in ("is_finite", "is_nan"):
            out.append([toks[0], toks[1], 0.0])
            continue
        for op in ops:
            if f" {op} " in f" {term} ":
                col, _o, val = term.partition(op)
                try:
                    out.append([col.strip(), op, float(val.strip())])
                except ValueError:
                    pass
                break
    return out[:4]


__all__ = ["PropertiesPanel", "get_path", "set_path", "ColorButton", "CHART_FIELDS"]
