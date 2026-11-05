"""The Sources panel of the Figure Editor (V5.00 step 3).

Every Page Output of every page the figure page may read, grouped by page, with what each
one carries — read from its envelope, never pulled: the image (channels), the label,
point and track tables with their columns, the measured fields. A source row drags onto
the page as an image; a table or column row drags onto an image item as a layer. Double-
click a source to add it as an image.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from nodegraph.domains import Domain
from nodelab_v2 import theme as T
from nodelab_v2.figure_editor.scene import COLUMN_MIME, SOURCE_MIME

_ROLE_KIND = Qt.UserRole + 1
_ROLE_PAYLOAD = Qt.UserRole + 2

#: table domain → the overlay layer kind that draws it
_LAYER_KIND = {"label": "labels", "point": "points", "track": "tracks"}


def describe_envelope(env: Any) -> Dict[str, Any]:
    """What a Page Output carries, from its envelope: ``{"image": bool, "channels": int,
    "channel_names": [...], "axes": {...}, "labels": [(layer, [columns])], "points": [...],
    "tracks": [...], "fields": [(layer, [columns])], "mesh": [layers]}``."""
    out: Dict[str, Any] = {"image": False, "channels": 0, "channel_names": [], "axes": {},
                           "labels": [], "points": [], "tracks": [], "fields": [], "mesh": []}
    if env is None:
        return out
    axes = getattr(env, "axes", None)
    md = getattr(env, "metadata", None) or {}
    if axes is not None:
        out["axes"] = {k: int(getattr(axes, k, 1) or 1) for k in ("m", "t", "z", "c", "y", "x")}
        out["image"] = int(getattr(axes, "y", 1) or 1) > 1 and int(getattr(axes, "x", 1) or 1) > 1
        out["channels"] = int(getattr(axes, "c", 1) or 1)
    out["channel_names"] = [str(n) for n in (md.get("channel_names") or [])]
    cols: Dict[Tuple[str, str], List[str]] = {}
    for dom, layer, col in getattr(env, "column_names", ()) or ():
        key = (getattr(dom, "value", str(dom)), str(layer))
        cols.setdefault(key, []).append(str(col))
    layers = list(getattr(env, "layer_names", ()) or ())
    fields_meta = md.get("__analysis_mesh__") or {}
    for dom, layer in layers:
        dv = getattr(dom, "value", str(dom))
        cl = cols.get((dv, str(layer)), [])
        if dv == "label":
            out["labels"].append((str(layer), cl))
        elif dv == "point":
            is_field = any(c.startswith(("disp_", "strain_")) or c in ("speed", "velocity",
                                                                           "density", "divergence")
                           for c in cl) or str(layer) in {v.get("field") for v in
                                                           fields_meta.values()
                                                           if isinstance(v, dict)}
            (out["fields"] if is_field else out["points"]).append((str(layer), cl))
        elif dv == "track":
            out["tracks"].append((str(layer), cl))
        elif dv == "mesh":
            out["mesh"].append(str(layer))
    return out


def summary_line(d: Mapping[str, Any]) -> str:
    parts = []
    if d.get("image"):
        ax = d.get("axes") or {}
        geo = " × ".join(f"{k.upper()}{ax[k]}" for k in ("m", "t", "z") if ax.get(k, 1) > 1)
        parts.append(f"image {ax.get('c', 1)} ch" + (f" · {geo}" if geo else ""))
    for key, word in (("labels", "labels"), ("points", "points"), ("tracks", "tracks"),
                      ("fields", "field")):
        if d.get(key):
            parts.append(f"{word}: " + ", ".join(l for l, _c in d[key]))
    if d.get("mesh"):
        parts.append("mesh: " + ", ".join(d["mesh"]))
    return " · ".join(parts) or "nothing on the wire yet"


class _Tree(QTreeWidget):
    """A tree whose rows drag as sources or columns."""

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setIndentation(14)

    def startDrag(self, actions) -> None:                                 # noqa: N802
        it = self.currentItem()
        if it is None:
            return
        kind = it.data(0, _ROLE_KIND)
        payload = it.data(0, _ROLE_PAYLOAD)
        if not kind or payload is None:
            return
        md = QMimeData()
        if kind == "source":
            md.setData(SOURCE_MIME, str(payload).encode("utf8"))
        else:
            md.setData(COLUMN_MIME, json.dumps(payload).encode("utf8"))
        md.setText(str(it.text(0)))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.exec(Qt.CopyAction)


class SourcesPanel(QWidget):
    """``catalog`` rows (from the host): ``{"value", "label", "page", "env"}``."""

    #: a source was double-clicked: add it as an image
    source_activated = Signal(str)
    #: a layer row was double-clicked: ``payload`` (the column drag's dict)
    layer_activated = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureSources")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        self.title = QLabel("Sources")
        self.title.setProperty("role", "title")
        lay.addWidget(self.title)
        self.hint = QLabel("Drag a source onto the page, a table onto an image. "
                           "Double-click adds an image.")
        self.hint.setWordWrap(True)
        self.hint.setProperty("role", "muted")
        lay.addWidget(self.hint)
        self.tree = _Tree()
        self.tree.itemDoubleClicked.connect(self._on_double)
        lay.addWidget(self.tree, 1)
        self.catalog: List[Dict[str, Any]] = []
        self.restyle()

    def set_catalog(self, catalog: Sequence[Mapping[str, Any]]) -> None:
        self.catalog = [dict(c) for c in catalog]
        self.tree.clear()
        by_page: Dict[str, List[Dict[str, Any]]] = {}
        for c in self.catalog:
            by_page.setdefault(str(c.get("page") or ""), []).append(c)
        for page, rows in by_page.items():
            top = QTreeWidgetItem([page or "(page)"])
            top.setFlags(top.flags() & ~Qt.ItemIsDragEnabled)
            self.tree.addTopLevelItem(top)
            for c in rows:
                d = describe_envelope(c.get("env"))
                row = QTreeWidgetItem([f"{c.get('name') or c['value']}  —  {summary_line(d)}"])
                row.setData(0, _ROLE_KIND, "source")
                row.setData(0, _ROLE_PAYLOAD, str(c["value"]))
                row.setToolTip(0, f"{c.get('label') or c['value']}\n{summary_line(d)}")
                top.addChild(row)
                for key, dom in (("labels", "label"), ("points", "point"), ("tracks", "track"),
                                 ("fields", "point")):
                    for layer, cols in d.get(key, []):
                        kind = "field" if key == "fields" else _LAYER_KIND[dom]
                        child = QTreeWidgetItem([f"{key[:-1] if key != 'fields' else 'field'}: "
                                                 f"{layer}" + (f"  ({len(cols)} columns)"
                                                               if cols else "")])
                        child.setData(0, _ROLE_KIND, "layer")
                        child.setData(0, _ROLE_PAYLOAD, {"source": str(c["value"]),
                                                         "domain": dom, "layer": layer,
                                                         "kind": kind, "column": ""})
                        row.addChild(child)
                        for col in cols:
                            cc = QTreeWidgetItem([col])
                            cc.setData(0, _ROLE_KIND, "column")
                            cc.setData(0, _ROLE_PAYLOAD, {"source": str(c["value"]),
                                                          "domain": dom, "layer": layer,
                                                          "kind": "field" if dom != "label"
                                                          else "labels", "column": col})
                            child.addChild(cc)
                for layer in d.get("mesh", []):
                    child = QTreeWidgetItem([f"mesh: {layer}"])
                    child.setData(0, _ROLE_KIND, "layer")
                    child.setData(0, _ROLE_PAYLOAD, {"source": str(c["value"]), "domain": "mesh",
                                                     "layer": layer, "kind": "mesh",
                                                     "column": ""})
                    row.addChild(child)
            top.setExpanded(True)

    def _on_double(self, it: QTreeWidgetItem, _col: int) -> None:
        kind = it.data(0, _ROLE_KIND)
        if kind == "source":
            self.source_activated.emit(str(it.data(0, _ROLE_PAYLOAD)))
        elif kind in ("layer", "column"):
            self.layer_activated.emit(it.data(0, _ROLE_PAYLOAD))

    def restyle(self) -> None:
        def h(c) -> str:
            return c.name()
        self.setStyleSheet(f"""
            QWidget#figureSources {{ background:{h(T.PANEL)}; }}
            QLabel {{ color:{h(T.INK)}; }}
            QLabel[role="title"] {{ font-weight:700; }}
            QLabel[role="muted"] {{ color:{h(T.MUTED)}; font-size:10px; }}
            QTreeWidget {{ background:{h(T.BODY)}; color:{h(T.INK)}; border:1px solid
                {h(T.BORDER)}; border-radius:6px; }}
            QTreeWidget::item:selected {{ background:{h(T.ACCENT_DIM)}; color:{h(T.INK)}; }}
        """)


__all__ = ["SourcesPanel", "describe_envelope", "summary_line"]
