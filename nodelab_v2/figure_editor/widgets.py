"""The Figure widgets palette (2026-11-05): everything the page can hold, as buttons.

Shown in the window's **Properties** dock while the Analysis workspace is up — where a
graph page shows a node's parameters — above the selected item's properties form. One
button per kind of thing a figure page holds, grouped: *Images* (an image of a source, a
montage, a zoom-in popout, a twin), *Annotate* (text, a letter, a scale bar, a box, a line,
an arrow, a measurement line), *Keys* (a colour bar, a histogram, a legend) and *Data* (a
chart, a summary table, a data block). Each is enabled when what it needs is there — a
zoom-in, a twin, a colour bar, a histogram and a legend want an image selected; an image,
a montage, a chart, a table and a data block want a source that carries what they read —
and a button that cannot act says why in its tooltip. *Image* and *Montage* open a menu
when several image sources are published; *Chart*, *Table* and *Data block* likewise over
the sources that carry a table.

The palette only asks the editor (:class:`~nodelab_v2.figure_editor.panel.FigureEditorPanel`)
to add things through the same entry points the Properties form and the start card use
(``add_image``, ``add_popout``, ``add_twin``, ``add_item``, ``add_chart`` …), so a widget
placed from here is exactly the item a drag or a template would have made.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QGridLayout, QLabel, QMenu, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from nodegraph.figure import schema as S
from nodelab_v2 import theme as T
from nodelab_v2.figure_editor.sources import describe_envelope

#: a widget needs: ``image_source`` (a published image), ``table_source`` (a published
#: table), ``image_selected`` (an image item selected), ``image_optional`` (reads the
#: selected image when there is one), ``""`` (nothing)
Need = str

#: ``(key, glyph, label, tooltip, need)`` per widget, in groups
WIDGET_GROUPS: List[Tuple[str, List[Tuple[str, str, str, str, Need]]]] = [
    ("Images", [
        ("image", "▣", "Image",
         "An image of a published source — its channels in their own colours, a scale "
         "bar. With several sources a menu asks which.", "image_source"),
        ("montage", "▦", "Montage",
         "Every position of a source as tiles (Properties switches the tiling to frames "
         "or planes).", "image_source"),
        ("zoom", "⊕", "Zoom-in",
         "A popout of the selected image: its centre magnified beside it, a dashed box on "
         "the parent and leader lines that follow the popout. Alt-drag a box on an image "
         "to choose the region yourself.", "image_selected"),
        ("twin", "⧉", "Twin",
         "A second copy of the selected image sharing its region, frame and channels — "
         "give it other layers (a speed heat layer over the same cells).",
         "image_selected"),
    ]),
    ("Annotate", [
        ("text", "T", "Text", "Free text in the page's font, or its own size.", ""),
        ("letter", "A", "Letter",
         "A panel letter in the page's letter style (Re-letter orders every panel).", ""),
        ("scale_bar", "⊢", "Scale bar",
         "A standalone scale bar, at the selected image's scale when one is selected.",
         "image_optional"),
        ("shape", "▭", "Box", "A rectangle or an ellipse (Properties picks which).", ""),
        ("line", "╱", "Line", "A straight line.", ""),
        ("arrow", "➚", "Arrow", "An arrow — drag its ends.", ""),
        ("measure_line", "↔", "Measure",
         "A line labelled with its length in µm, at the selected image's scale.",
         "image_optional"),
    ]),
    ("Keys", [
        ("colorbar", "▬", "Colour bar",
         "The colour ramp of the selected image's first heat layer, with its limits.",
         "image_selected"),
        ("histogram", "▥", "Histogram",
         "The selected image's first channel, its display window marked.", "image_selected"),
        ("legend", "≡", "Legend",
         "The selected image's channels and layers, one row each.", "image_selected"),
    ]),
    ("Data", [
        ("chart", "∿", "Chart",
         "A chart of a published table — a histogram of its first measured column to "
         "start from; Properties picks the kind, columns, error, fit and test.",
         "table_source"),
        ("table", "▤", "Table",
         "A summary table of a published table: n, mean and SEM per group.", "table_source"),
        ("data", "⊞", "Data block",
         "A data entry: a published table filtered and summarised in the page's hidden "
         "graph, for charts and tables to read.", "table_source"),
    ]),
]

_NEED_TEXT = {
    "image_source": "needs an image published as a Page Output upstream",
    "table_source": "needs a label, point or track table published upstream",
    "image_selected": "select an image on the page first",
}


class _WidgetButton(QToolButton):
    """A glyph over a name, a fixed tile."""

    def __init__(self, glyph: str, label: str, tip: str) -> None:
        super().__init__()
        self.setObjectName("figureWidget")
        self.setText(f"{glyph}\n{label}")
        self.base_tip = tip
        self.setToolTip(tip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(82, 50)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setFocusPolicy(Qt.NoFocus)


class FigureWidgetsPanel(QWidget):
    """The palette. :meth:`set_editor` names the Figure Editor it adds to; :meth:`sync`
    re-reads its selection and sources (the window calls it on every selection change)."""

    #: a widget was placed: its item id
    placed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureWidgets")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._ed: Any = None
        self.buttons: Dict[str, _WidgetButton] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        self.title = QLabel("Figure widgets")
        self.title.setProperty("role", "title")
        lay.addWidget(self.title)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setProperty("role", "muted")
        lay.addWidget(self.hint)
        for group, specs in WIDGET_GROUPS:
            head = QLabel(group.upper())
            head.setObjectName("figureWidgetGroup")
            lay.addWidget(head)
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 2)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(4)
            for i, (key, glyph, label, tip, _need) in enumerate(specs):
                b = _WidgetButton(glyph, label, tip)
                b.clicked.connect(lambda _=False, k=key: self.place(k))
                grid.addWidget(b, i // 4, i % 4)
                self.buttons[key] = b
            grid.setColumnStretch(4, 1)
            lay.addLayout(grid)
        lay.addStretch(1)
        self.restyle()
        self.sync()

    # ── context ────────────────────────────────────────────────────────────────
    def set_editor(self, ed: Any) -> None:
        """The Figure Editor the buttons add to (``None``: every button off)."""
        self._ed = ed
        self.sync()

    def editor(self) -> Any:
        return self._ed

    def selected_image(self) -> Optional[str]:
        """The id of the first selected image or montage item, or ``None``."""
        ed = self._ed
        if ed is None:
            return None
        try:
            doc = ed.document
            for iid in ed.selected_ids():
                it = next((i for i in doc.get("items", []) if i.get("id") == iid), None)
                if it is not None and it.get("kind") in ("image", "montage"):
                    return str(iid)
        except RuntimeError:              # the editor's canvas closed under us
            return None
        return None

    def _catalog(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        ed = self._ed
        if ed is None:
            return []
        try:
            cat = list(ed.sources_panel.catalog)
        except (AttributeError, RuntimeError):
            return []
        return [(c, describe_envelope(c.get("env"))) for c in cat]

    def image_sources(self) -> List[Dict[str, Any]]:
        return [c for c, d in self._catalog() if d.get("image")]

    def table_sources(self) -> List[Dict[str, Any]]:
        return [c for c, d in self._catalog()
                if d.get("labels") or d.get("points") or d.get("tracks")]

    def sync(self) -> None:
        """Enable what can act now and say why the rest cannot."""
        ed = self._ed
        bound = ed is not None and ed.bound() is not None
        sel = self.selected_image() if bound else None
        have_img = bool(self.image_sources()) if bound else False
        have_tab = bool(self.table_sources()) if bound else False
        for _group, specs in WIDGET_GROUPS:
            for key, _g, _l, _tip, need in specs:
                b = self.buttons[key]
                ok = bound and (need == "" or need == "image_optional"
                                or (need == "image_source" and have_img)
                                or (need == "table_source" and have_tab)
                                or (need == "image_selected" and sel is not None))
                b.setEnabled(bool(ok))
                why = "" if ok else ("no figure page is shown" if not bound
                                     else _NEED_TEXT.get(need, ""))
                b.setToolTip(b.base_tip + (f"\n\n({why})" if why else ""))
        if not bound:
            self.hint.setText("Show an Analysis page to place widgets on it.")
        elif sel is not None:
            self.hint.setText(f"Selected image: {sel} — Zoom-in, Twin, Colour bar, Histogram, "
                              f"Legend, Scale bar and Measure read it.")
        else:
            self.hint.setText("Select an image on the page for Zoom-in, Twin, Colour bar, "
                              "Histogram and Legend; the rest place anywhere.")

    # ── placing ────────────────────────────────────────────────────────────────
    def place(self, key: str) -> Optional[str]:
        """Place widget ``key``; returns the new item's id (``None`` when a menu was
        opened instead, or nothing could be placed)."""
        ed = self._ed
        if ed is None or ed.bound() is None:
            return None
        sel = self.selected_image()
        iid: Optional[str] = None
        if key in ("image", "montage"):
            return self._pick_source(key, self.image_sources(), self._add_image_of)
        if key in ("chart", "table", "data"):
            return self._pick_source(key, self.table_sources(), self._add_data_of)
        if key == "zoom":
            if sel is not None:
                iid = ed.add_popout(sel, (0.3, 0.7, 0.3, 0.7))
        elif key == "twin":
            if sel is not None:
                iid = ed.add_twin(sel)
        elif key == "colorbar":
            if sel is not None:
                iid = ed.add_colorbar_for(sel)
        elif key == "histogram":
            if sel is not None:
                iid = ed.add_histogram_for(sel)
        elif key == "legend":
            if sel is not None:
                iid = ed.add_legend_for(sel)
        elif key in ("scale_bar", "measure_line"):
            pos = None
            if sel is not None:
                r = S.item(ed.document, sel)["rect_mm"]
                pos = [r[0], r[1] + r[3] + 1.5]
            iid = ed.add_item(key, pos, of={"item": sel})
        elif key == "text":
            iid = ed.add_item("text", text="Text")
        elif key == "letter":
            iid = ed.add_item("letter", text=ed._next_letter())
        elif key in ("shape", "line", "arrow"):
            iid = ed.add_item(key)
        if iid:
            self.placed.emit(iid)
        return iid

    def _pick_source(self, key: str, sources: Sequence[Dict[str, Any]],
                     add: Callable[[str, str], Optional[str]]) -> Optional[str]:
        if not sources:
            return None
        if len(sources) == 1:
            iid = add(key, str(sources[0]["value"]))
            if iid:
                self.placed.emit(iid)
            return iid
        menu = QMenu(self)
        for c in sources:
            label = str(c.get("label") or c["value"])
            page = str(c.get("page") or "")
            menu.addAction(f"{label}  ·  {page}" if page else label,
                           lambda v=str(c["value"]): self._add_and_tell(add, key, v))
        b = self.buttons.get(key)
        menu.exec(b.mapToGlobal(b.rect().bottomLeft()) if b is not None else self.mapToGlobal(
            self.rect().center()))
        return None

    def _add_and_tell(self, add: Callable[[str, str], Optional[str]], key: str,
                      value: str) -> None:
        iid = add(key, value)
        if iid:
            self.placed.emit(iid)

    def _add_image_of(self, key: str, value: str) -> Optional[str]:
        ed = self._ed
        return ed.add_montage(value) if key == "montage" else ed.add_image(value)

    def _add_data_of(self, key: str, value: str) -> Optional[str]:
        ed = self._ed
        if key == "chart":
            return ed.add_chart(value)
        if key == "table":
            return ed.add_table(value)
        d = next((d for c, d in self._catalog() if c["value"] == value), {})
        domain = "label" if d.get("labels") else "point" if d.get("points") else "track"
        return ed.add_data([value], domain=domain)

    # ── look ───────────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        self.setStyleSheet(f"""
        QWidget#figureWidgets {{ background:{T.BG.name()}; }}
        QLabel[role="title"] {{ color:{T.INK.name()}; font-weight:800; font-size:12px; }}
        QLabel[role="muted"] {{ color:{T.MUTED.name()}; font-size:11px; }}
        QLabel#figureWidgetGroup {{ color:{T.MUTED.name()}; font-size:10px; font-weight:800;
            letter-spacing:1px; margin-top:4px; }}
        QToolButton#figureWidget {{ background:{T.BODY.name()}; color:{T.INK.name()};
            border:1px solid {T.BORDER.name()}; border-radius:8px; font-weight:600;
            font-size:11px; padding:2px; }}
        QToolButton#figureWidget:hover {{ background:{T.PANEL_HI.name()};
            border-color:{T.BORDER_HI.name()}; }}
        QToolButton#figureWidget:pressed {{ background:{T.ACCENT_DIM.name()};
            border-color:{T.ACCENT.name()}; }}
        QToolButton#figureWidget:disabled {{ color:{T.MUTED.name()};
            background:{T.BG.name()}; border-color:{T.BORDER.name()}; }}
        """ + T.controls_qss())


__all__ = ["FigureWidgetsPanel", "WIDGET_GROUPS"]
