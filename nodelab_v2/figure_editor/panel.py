"""The Figure Editor panel (V5.00 step 3): a figure page's canvas, sources, properties.

Bound to one figure page at a time (the window binds it when a canvas shows an Analysis
page). It holds a working view of the page's :class:`~nodelab_v2.figure_document.
FigureDocument`, the scene with one item per document item, the payloads of the sources it
has fetched, and the renders of its image items. Every edit goes through :meth:`commit`:
the document is replaced through the host (which re-syncs the shadow graph and publishes
the change), the panel's local undo list grows, and the scene is rebuilt from the
document — the document is the only truth, the scene a view of it.

``host`` is the window's adapter (``_FigureHost``): ``figure_state(pid)``, ``catalog(pid)``,
``envelopes(pid)``, ``run_ids(pid)``, ``finished(run_id)``, ``fetch(run_id)``,
``commit(pid, doc)``, ``viewer_state(run_id, names)``, ``own_layers(run_id)``,
``export_path(kind)``, ``status(text)``. That list is the panel's whole dependency surface
and what the GUI probe drives.

Renders run on a worker thread, latest request per item wins (the Movie Editor's
pattern): a fetched payload's provider may compute on read, and the GUI thread must never
wait on that.
"""
from __future__ import annotations

import copy
import json
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy,
                               QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget)

from nodegraph.figure import bind as B
from nodegraph.figure import layout as FL
from nodegraph.figure import schema as S
from nodelab_v2 import theme as T
from nodelab_v2.figure_editor import charts_data as CD
from nodelab_v2.figure_editor import links as LK
from nodelab_v2.figure_editor import render as R
from nodelab_v2.figure_editor.items import (ChartItem, ColorbarItem, FigureItem, HistogramItem,
                                            ImageItem, LeaderItem, MeasureLineItem,
                                            MontageItem, ScaleBarItem, make_item)
from nodelab_v2.figure_editor.properties import PropertiesPanel, get_path, set_path
from nodelab_v2.figure_editor.scene import MM, FigureScene, FigureView, mm, to_mm
from nodelab_v2.figure_editor.sources import SourcesPanel, describe_envelope
from nodelab_v2.figure_editor.start_card import StartCard
from nodelab_v2.figure_document import FigureDocument

#: the figure's own unit for overlay looks: points → page units
_STROKE = 25.4 / 72.0 * MM


def _panel_qss() -> str:
    def h(c: Any) -> str:
        return c.name() if c is not None else "#222"
    return f"""
QWidget#figureRoot {{ background:{h(T.PANEL)}; }}
QLabel {{ color:{h(T.INK)}; }}
QLabel[role="muted"] {{ color:{h(T.MUTED)}; }}
QLabel[role="title"] {{ color:{h(T.INK)}; font-weight:700; }}
QLabel[role="error"] {{ color:{h(T.ERROR)}; }}
QPushButton, QToolButton {{ background:{h(T.BODY)}; color:{h(T.INK)};
  border:1px solid {h(T.BORDER)}; border-radius:6px; padding:3px 8px; }}
QPushButton:hover, QToolButton:hover {{ border-color:{h(T.ACCENT)}; }}
QPushButton:disabled, QToolButton:disabled {{ color:{h(T.MUTED)}; }}
QComboBox, QSpinBox {{ background:{h(T.BODY)}; color:{h(T.INK)}; border:1px solid {h(T.BORDER)};
  border-radius:6px; padding:2px 6px; }}
QSplitter::handle {{ background:{h(T.BORDER)}; }}
"""


class _RenderWorker(QObject):
    """One worker thread, one pending job per item, latest wins."""

    done = Signal(str, int, object)          # item id, generation, Rendered | Exception

    def __init__(self) -> None:
        super().__init__()
        self._cv = threading.Condition()
        self._jobs: Dict[str, Tuple[int, Callable[[], Any]]] = {}
        self._stop = False
        self.busy = False
        self._thread = threading.Thread(target=self._loop, name="figure-editor-render",
                                        daemon=True)
        self._thread.start()

    def request(self, item_id: str, gen: int, fn: Callable[[], Any]) -> None:
        with self._cv:
            self._jobs[item_id] = (gen, fn)
            self.busy = True
            self._cv.notify()

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._jobs.clear()
            self._cv.notify()
        self._thread.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._jobs and not self._stop:
                    self.busy = False
                    self._cv.wait()
                if self._stop:
                    self.busy = False
                    return
                item_id, (gen, fn) = next(iter(self._jobs.items()))
                del self._jobs[item_id]
            try:
                out: Any = fn()
            except Exception as exc:      # noqa: BLE001 — shown on the item, never fatal
                out = exc
            try:
                self.done.emit(item_id, gen, out)
            except RuntimeError:
                return
            with self._cv:
                if not self._jobs:
                    self.busy = False


class FigureEditorPanel(QWidget):
    """See the module docstring."""

    #: the panel's page or selection changed — the window syncs its inspector / status
    selection_changed = Signal()

    def __init__(self, host: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._host = host
        self._pid: Optional[str] = None
        self._fig: Optional[FigureDocument] = None
        self._doc: Dict[str, Any] = S.new_document()
        self._envs: Dict[str, Any] = {}
        self._run_ids: Dict[str, str] = {}
        self._payloads: Dict[str, Any] = {}
        self._labels: Dict[str, str] = {}
        self._items: Dict[str, FigureItem] = {}
        self._leaders: Dict[Tuple[str, str], LeaderItem] = {}
        self._renders: Dict[str, Tuple[str, Any]] = {}      # item id → (key, Rendered | tiles)
        self._tables: Dict[str, Tuple[int, Any]] = {}       # run id → (id(payload), tables)
        self._hover_text = ""
        self._gen = 0
        self._undo: List[str] = []
        self._redo: List[str] = []
        self._cursor: Dict[str, int] = {"m": 0, "t": 0, "z": 0}
        self._field_cache: Dict[Any, Any] = {}
        self._palettes: Dict[int, Any] = {}
        self._rebuilding = False
        self._worker = _RenderWorker()
        self._worker.done.connect(self._on_rendered)

        self.setObjectName("figureRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── toolbar ───────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        bar.setSpacing(6)
        self._title = QLabel("Figure")
        self._title.setProperty("role", "title")
        bar.addWidget(self._title)
        self._undo_btn = QToolButton()
        self._undo_btn.setText("↶")
        self._undo_btn.setToolTip("Undo (Ctrl+Z)")
        self._undo_btn.clicked.connect(self.undo)
        self._redo_btn = QToolButton()
        self._redo_btn.setText("↷")
        self._redo_btn.setToolTip("Redo (Ctrl+Y)")
        self._redo_btn.clicked.connect(self.redo)
        bar.addWidget(self._undo_btn)
        bar.addWidget(self._redo_btn)
        self._compute_btn = QPushButton("Compute sources")
        self._compute_btn.setToolTip("Pull every source this figure reads that is not in hand "
                                     "— nothing is computed until you ask")
        self._compute_btn.clicked.connect(self.compute_sources)
        bar.addWidget(self._compute_btn)
        bar.addSpacing(10)
        self._cursor_spins: Dict[str, QSpinBox] = {}
        for ax, label in (("m", "M"), ("t", "T"), ("z", "Z")):
            lab = QLabel(label)
            lab.setProperty("role", "muted")
            bar.addWidget(lab)
            sp = QSpinBox()
            sp.setRange(0, 0)
            sp.setToolTip(f"The page cursor on {label}: every image whose frame reads the "
                          f"cursor follows it")
            sp.valueChanged.connect(lambda v, a=ax: self._set_cursor(a, v))
            bar.addWidget(sp)
            self._cursor_spins[ax] = sp
        bar.addSpacing(10)
        self._style_combo = QComboBox()
        for s in S.STYLES:
            self._style_combo.addItem(s, s)
        self._style_combo.setToolTip("The page preset: size, type and line weights for where the "
                               "figure is going")
        self._style_combo.currentIndexChanged.connect(self._on_style)
        bar.addWidget(self._style_combo)
        self._zoom_fit = QToolButton()
        self._zoom_fit.setText("Fit")
        self._zoom_fit.clicked.connect(lambda: self.view.fit_page())
        bar.addWidget(self._zoom_fit)
        self._zoom_1 = QToolButton()
        self._zoom_1.setText("1:1")
        self._zoom_1.clicked.connect(lambda: self.view.zoom_1())
        bar.addWidget(self._zoom_1)
        bar.addStretch(1)
        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        self._status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self._status, 1)
        self._export_btn = QPushButton("Export ▾")
        menu = QMenu(self._export_btn)
        for fmt, text in (("png", "PNG at the page dpi"), ("tiff", "TIFF at the page dpi"),
                          ("svg", "SVG (text stays text)"), ("pdf", "PDF (fonts embedded)")):
            menu.addAction(text, lambda f=fmt: self.export_dialog(f))
        self._export_btn.setMenu(menu)
        bar.addWidget(self._export_btn)
        root.addLayout(bar)

        # ── body ──────────────────────────────────────────────────────────────
        self.scene = FigureScene(self)
        self.view = FigureView(self.scene)
        self.sources_panel = SourcesPanel()
        self.props = PropertiesPanel(self)
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.sources_panel)
        self.split.addWidget(self.view)
        self.split.addWidget(self.props)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setStretchFactor(2, 0)
        self.split.setSizes([220, 800, 260])
        root.addWidget(self.split, 1)
        self.start = StartCard(self.view)
        self.start.hide()
        self.start.goal_chosen.connect(self._on_goal)
        self.start.source_chosen.connect(lambda v: self.add_image(v))
        self.start.dismissed.connect(self._dismiss_start)

        self.scene.selectionChanged.connect(self._on_selection)
        self.scene.geometry_edited.connect(self._on_geometry_edited)
        self.scene.key_action.connect(self._on_key)
        self.scene.changed.connect(lambda _r: self._update_leaders())
        self.view.region_dragged.connect(self._on_region_dragged)
        self.view.source_dropped.connect(self._on_source_dropped)
        self.view.column_dropped.connect(self._on_column_dropped)
        self.view.hovered.connect(self._on_hover)
        self.sources_panel.source_activated.connect(lambda v: self.add_image(v))
        self.sources_panel.layer_activated.connect(self._on_layer_activated)
        self.restyle()

    # ── host protocol ──────────────────────────────────────────────────────────
    def bound(self) -> Optional[str]:
        return self._pid

    def bind(self, page_id: Optional[str]) -> None:
        """Show ``page_id``'s figure (``None`` unbinds)."""
        if self._fig is not None:
            self._fig.off_change(self._on_figure_changed)
        self._pid = page_id
        self._fig = self._host.figure_state(page_id) if page_id else None
        self._payloads.clear()
        self._renders.clear()
        self._tables.clear()
        self._undo.clear()
        self._redo.clear()
        self._field_cache.clear()
        self._palettes.clear()
        if self._fig is None:
            self._doc = S.new_document()
            self._title.setText("Figure — no page")
            self._rebuild()
            return
        self._fig.on_change(self._on_figure_changed)
        self._doc = self._fig.to_dict()
        self._title.setText(f"Figure — {self._host.page_name(page_id)}")
        self.refresh_sources()
        self._rebuild()
        self.view.fit_page()

    def refresh_sources(self) -> None:
        """Re-read the feeder pages' outputs, the sources' envelopes and run ids."""
        if self._pid is None:
            return
        self._envs = dict(self._host.envelopes(self._pid))
        self._run_ids = dict(self._host.run_ids(self._pid))
        self._labels = dict(self._host.source_labels(self._pid))
        catalog = list(self._host.catalog(self._pid))
        sig = tuple((c["value"], c.get("label"), c.get("page"), id(c.get("env")))
                    for c in catalog)
        if sig != getattr(self, "_catalog_sig", None):   # the tree only when it changed
            self._catalog_sig = sig
            self.sources_panel.set_catalog(catalog)
            self.start.configure([dict(describe_envelope(c.get("env")), value=c["value"],
                                       label=c.get("label"), page=c.get("page"))
                                  for c in catalog])
        # payloads already in hand (a memo hit, an earlier pull)
        for sid, rid in self._run_ids.items():
            if rid not in self._payloads:
                have = self._host.finished(rid)
                if have is not None:
                    self._payloads[rid] = have
        self._sync_cursor_ranges()

    def on_fetched(self, run_id: str, payload: Any) -> None:
        if run_id in self._run_ids.values() or run_id in self._payloads:
            self._payloads[run_id] = payload
            self._palettes.pop(id(payload), None)
            self._tables.pop(run_id, None)
            self._invalidate_renders(run_id)
            self._status.setText(f"{run_id} ready")

    def on_doc_changed(self, page_id: str) -> None:
        """A page's graph changed: the sources may resolve differently now. Only the
        images of a source whose run id or payload changed re-render — the figure's own
        shadow sync fires this on every commit, and a letter edit must not redraw a page."""
        if self._pid is None:
            return
        old = dict(self._run_ids)
        self.refresh_sources()
        changed = {sid for sid, rid in self._run_ids.items() if old.get(sid) != rid}
        for rid in list(self._payloads):            # a payload the runner no longer holds
            if rid in self._run_ids.values() and self._host.finished(rid) is None:
                self._payloads.pop(rid, None)
                changed |= {s for s, r in self._run_ids.items() if r == rid}
        if changed:
            for iid in list(self._renders):
                img = S.effective_image(self._doc, iid) if iid in self._items else {}
                if img.get("source") in changed:
                    self._renders.pop(iid, None)
        self._refresh_renders()

    def on_display_changed(self) -> None:
        """The Viewer's look settled: images with Viewer-linked channels re-render."""
        for iid, it in self._items.items():
            if isinstance(it, ImageItem):
                img = S.effective_image(self._doc, iid)
                if any(ch.get("mode") == "viewer" for ch in img.get("channels") or []):
                    self._renders.pop(iid, None)
        self._refresh_renders()

    def shutdown(self) -> None:
        self._worker.shutdown()

    def busy(self) -> bool:
        return bool(self._worker.busy)

    def restyle(self) -> None:
        self.setStyleSheet(_panel_qss())
        self.view.restyle()
        self.sources_panel.restyle()
        self.props.restyle()
        self.start.restyle()
        self.scene.update()

    # ── the document ───────────────────────────────────────────────────────────
    @property
    def document(self) -> Dict[str, Any]:
        return self._doc

    def _on_figure_changed(self) -> None:
        """The FigureDocument changed (our commit, a load, the other half of a twin edit)."""
        if self._fig is None:
            return
        new = self._fig.to_dict()
        if S.canonical_json(new) == S.canonical_json(self._doc):
            return
        self._doc = new
        self._rebuild()

    def commit(self, doc: Mapping[str, Any], *, undo: bool = True) -> bool:
        """Make ``doc`` the page's figure. Validated by the FigureDocument; on a refusal
        the status line names the bad value and nothing changes."""
        if self._fig is None or self._pid is None:
            return False
        before = self._fig.canonical()
        try:
            changed = self._host.commit(self._pid, doc)
        except S.FigureError as exc:
            self._status.setText(str(exc))
            self._status.setProperty("role", "error")
            self.restyle()
            return False
        if changed and undo:
            self._undo.append(before)
            self._redo.clear()
        self._doc = self._fig.to_dict()
        self._rebuild()
        return bool(changed)

    def edit(self, mutate: Callable[[Dict[str, Any]], None]) -> bool:
        d = copy.deepcopy(self._doc)
        mutate(d)
        return self.commit(d)

    def undo(self) -> None:
        if not self._undo or self._fig is None:
            return
        self._redo.append(self._fig.canonical())
        text = self._undo.pop()
        self.commit(json.loads(text), undo=False)

    def redo(self) -> None:
        if not self._redo or self._fig is None:
            return
        self._undo.append(self._fig.canonical())
        text = self._redo.pop()
        self.commit(json.loads(text), undo=False)

    # ── sources and items ──────────────────────────────────────────────────────
    def compute_sources(self) -> int:
        """Fetch every source not in hand (a payload-only pull: the Viewer stays where it
        is). Returns how many were asked for."""
        if self._pid is None:
            return 0
        n = 0
        for _sid, rid in self._run_ids.items():
            if rid and rid not in self._payloads:
                self._host.fetch(rid)
                n += 1
        self._status.setText(f"computing {n} source{'s' if n != 1 else ''}…" if n
                             else "every source is in hand")
        return n

    def source_id_for(self, value: str) -> Optional[str]:
        for s in self._doc.get("sources", []):
            if s["source"] == value or s["id"] == value:
                return str(s["id"])
        return None

    def add_source(self, value: str, label: Optional[str] = None) -> str:
        """The figure source for Page Output ``value`` (``"<page>:<name>"``), added when
        new, with the label the Source menus show."""
        have = self.source_id_for(value)
        if have:
            return have
        base = "".join(ch if ch.isalnum() else "_" for ch in value.split(":", 1)[-1]) or "src"
        sid = base
        k = 2
        while any(s["id"] == sid for s in self._doc.get("sources", [])):
            sid = f"{base}{k}"
            k += 1
        lab = label or self._catalog_label(value) or value.split(":", 1)[-1]

        def mut(d):
            d.setdefault("sources", []).append({"id": sid, "source": value, "label": lab})
        self.commit(_mutated(self._doc, mut))
        return sid

    def _catalog_label(self, value: str) -> str:
        for c in self.sources_panel.catalog:
            if c.get("value") == value:
                return str(c.get("label") or "")
        return ""

    def _new_item_id(self, base: str) -> str:
        ids = {it["id"] for it in self._doc.get("items", [])}
        k = 1
        while f"{base}{k}" in ids:
            k += 1
        return f"{base}{k}"

    def _next_letter(self) -> str:
        used = {it.get("letter") for it in self._doc.get("items", []) if it.get("letter")}
        for L in FL.letters(52, str(self._doc.get("page", {}).get("letters", "A"))):
            if L not in used:
                return L
        return ""

    def _free_pos(self, w_mm: float, h_mm: float) -> List[float]:
        """A spot on the page not covered by an item: left to right, row by row."""
        page = self._doc.get("page", {})
        pw, ph = float(page.get("width_mm", 85)), float(page.get("height_mm", 65))
        rects = [it["rect_mm"] for it in self._doc.get("items", [])]
        step = 2.0
        y = 0.0
        while y + h_mm <= ph + 1e-6:
            x = 0.0
            while x + w_mm <= pw + 1e-6:
                if not any(x < r[0] + r[2] and x + w_mm > r[0] and y < r[1] + r[3]
                           and y + h_mm > r[1] for r in rects):
                    return [x, y]
                x += step
            y += step
        return [2.0 * len(rects), 2.0 * len(rects)]

    def add_image(self, source: str, pos_mm: Optional[Sequence[float]] = None,
                  size_mm: Optional[Sequence[float]] = None, *, kind: str = "image",
                  tile_by: str = "m") -> str:
        """An image (or montage) item of a source (a Page Output value or a source id),
        with its channels in the source's own colours and a scale bar."""
        sid = self.source_id_for(source) or self.add_source(source)
        env = self._envs.get(sid)
        d = describe_envelope(env)
        ax = d.get("axes") or {}
        aspect = (ax.get("y", 1) / max(1, ax.get("x", 1))) if ax.get("x") else 0.75
        page = self._doc.get("page", {})
        w = float(size_mm[0]) if size_mm else min(60.0, float(page.get("width_mm", 85)) * 0.45)
        h = float(size_mm[1]) if size_mm else w * aspect
        if kind == "montage":
            w, h = (float(size_mm[0]), float(size_mm[1])) if size_mm else (w * 1.6, h * 1.6)
        pos = list(pos_mm) if pos_mm else self._free_pos(w, h)
        iid = self._new_item_id("img" if kind == "image" else "montage")
        md = getattr(env, "metadata", None) or {}
        names = d.get("channel_names") or []
        emission = list(md.get("channel_emission_nm") or [])
        nc = int(d.get("channels") or 1)
        chans = []
        for c in range(min(nc, 4)):
            rgb = "#ffffff" if nc == 1 else _emission_hex(emission[c] if c < len(emission)
                                                             else None, c)
            chans.append({"c": c, "rgb": rgb, "mode": "auto_pct",
                          "label": names[c] if c < len(names) else ""})
        letter = self._next_letter()
        item = {"id": iid, "kind": kind, "rect_mm": [pos[0], pos[1], w, h], "letter": letter,
                "image": {"source": sid, "frame": {"m": 0, "t": "cursor",
                                                   "z": "max" if ax.get("z", 1) > 1 else 0},
                          "channels": chans,
                          "decor": {"scale_bar_um": _default_scale_bar(md, ax)}}}
        if kind == "montage":
            item["montage"] = {"tile_by": tile_by, "columns": 0, "gap_mm": 0.5}

        def mut(dd):
            dd.setdefault("items", []).append(item)
        self.commit(_mutated(self._doc, mut))
        self.select([iid])
        return iid

    def add_montage(self, source: str, tile_by: str = "m", **kw) -> str:
        return self.add_image(source, kind="montage", tile_by=tile_by, **kw)

    def add_item(self, kind: str, pos_mm: Optional[Sequence[float]] = None,
                 size_mm: Optional[Sequence[float]] = None, **spec: Any) -> str:
        sizes = {"colorbar": (30.0, 6.0), "histogram": (30.0, 14.0), "scale_bar": (20.0, 5.0),
                 "text": (40.0, 8.0), "letter": (8.0, 6.0), "arrow": (12.0, 8.0),
                 "line": (12.0, 8.0), "shape": (20.0, 15.0), "measure_line": (15.0, 6.0)}
        w, h = (float(size_mm[0]), float(size_mm[1])) if size_mm else sizes.get(kind, (20.0, 10.0))
        pos = list(pos_mm) if pos_mm else self._free_pos(w, h)
        iid = self._new_item_id(kind.replace("_", ""))
        item: Dict[str, Any] = {"id": iid, "kind": kind, "rect_mm": [pos[0], pos[1], w, h]}
        item.update(spec)
        if kind in ("arrow", "line", "measure_line") and "to_mm" not in item:
            item["to_mm"] = [w, h]

        def mut(dd):
            dd.setdefault("items", []).append(item)
        self.commit(_mutated(self._doc, mut))
        self.select([iid])
        return iid

    # ── charts and data entries (step 4) ───────────────────────────────────────
    def _envelope_tables(self, key: str) -> List[Tuple[str, str, List[str]]]:
        """``[(domain, table, columns)]`` the envelope of source / data id ``key`` lists."""
        d = describe_envelope(self._envs.get(key))
        return ([("label", l, c) for l, c in d["labels"]]
                + [("point", l, c) for l, c in d["points"] + d["fields"]]
                + [("track", l, c) for l, c in d["tracks"]])

    def add_chart(self, source: Optional[str] = None, *, data: Optional[str] = None,
                  kind: str = "", domain: str = "", table: str = "",
                  pos_mm: Optional[Sequence[float]] = None,
                  size_mm: Optional[Sequence[float]] = None, **chart: Any) -> str:
        """A chart item reading a source (a Page Output value or a source id) or a data
        entry (``data=``). Without a kind: a histogram of the table's first measured
        column (``area`` when the table has it) — something drawn at once, to edit from."""
        sid = None
        if data is None:
            if not source:
                raise ValueError("add_chart: name a source or a data entry")
            sid = self.source_id_for(source) or self.add_source(source)
        key = str(data or sid)
        tables = self._envelope_tables(key)
        if data is not None:
            entry = next((e for e in self._doc.get("data", []) if e["id"] == data), None)
            if entry is None:
                raise ValueError(f"add_chart: unknown data entry {data!r}")
            domain = domain or str(entry.get("domain") or "label")
            if self._fig is not None:
                table = table or self._fig.data_table_name(data)
            if not tables:
                tables = self._envelope_tables(str((entry.get("from") or [""])[0]))
        if not table and tables:
            pick = next((t for t in tables if not domain or t[0] == domain), tables[0])
            domain, table = pick[0], pick[1]
        cols = next((c for _dm, l, c in tables if l == table), [])
        spec_chart: Dict[str, Any] = {"kind": kind or "distribution", "source": sid,
                                      "data": data, "domain": domain or "label",
                                      "table": table}
        if not kind:
            skip = {"id", "m", "t", "z", "c", "y", "x", "label", "parent", "track_id", "b"}
            value = "area" if "area" in cols else next(
                (c for c in cols if c not in skip and not c.startswith("bbox")), "")
            spec_chart.update(sub_kind="histogram", value=value)
        spec_chart.update(chart)
        page = self._doc.get("page", {})
        w = float(size_mm[0]) if size_mm else min(50.0, float(page.get("width_mm", 85)) * 0.5)
        h = float(size_mm[1]) if size_mm else w * 0.8
        pos = list(pos_mm) if pos_mm else self._free_pos(w, h)
        iid = self._new_item_id("chart")
        item = {"id": iid, "kind": "chart", "rect_mm": [pos[0], pos[1], w, h],
                "letter": self._next_letter(), "chart": spec_chart}

        def mut(dd):
            dd.setdefault("items", []).append(item)
        self.commit(_mutated(self._doc, mut))
        self.select([iid])
        return iid

    def add_data(self, sources: Sequence[str], *, domain: str = "label", table: str = "",
                 name: str = "", where: Optional[Sequence[Sequence[Any]]] = None,
                 aggregate: Optional[Mapping[str, Any]] = None) -> str:
        """A data entry — ``sources`` (Page Output values or source ids) pooled, filtered
        by ``where`` (``[[column, op, value], …]``) and summarised by ``aggregate``
        (``{"group_by", "value", "reducers", "name"}``) — compiled into the shadow graph
        by the document's sync. Returns its id, which a chart's ``data`` names."""
        sids = [self.source_id_for(s) or self.add_source(s) for s in sources]
        if not sids:
            raise ValueError("add_data: at least one source")
        ids = {d["id"] for d in self._doc.get("data", [])} | {s["id"] for s in self._doc.get("sources", [])}
        k = 1
        while f"data{k}" in ids:
            k += 1
        did = f"data{k}"
        entry: Dict[str, Any] = {"id": did, "from": list(sids), "domain": domain,
                                 "table": table, "name": name or "combined",
                                 "where": [list(w) for w in (where or [])],
                                 "aggregate": dict(aggregate) if aggregate else None}

        def mut(dd):
            dd.setdefault("data", []).append(entry)
        self.commit(_mutated(self._doc, mut))
        return did

    def edit_data(self, data_id: str, path: str, value: Any) -> None:
        def mut(dd):
            for d in dd.get("data", []):
                if d["id"] == data_id:
                    set_path(d, path, value)
        self.commit(_mutated(self._doc, mut))

    def set_chart_reads(self, item_id: str, what: str, ref: str) -> None:
        """Point a chart at a source (``what="source"``) or a data entry (``"data"``);
        the table and the columns are cleared when the domain of what it reads differs."""
        def mut(dd):
            ch = S.item(dd, item_id).setdefault("chart", {})
            if what == "data":
                ch["data"], ch["table"] = ref, ""
                entry = next((d for d in dd.get("data", []) if d["id"] == ref), None)
                if entry is not None:
                    ch["domain"] = str(entry.get("domain") or "label")
            else:
                ch["data"], ch["source"] = None, ref
        self.commit(_mutated(self._doc, mut))

    def set_chart_table(self, item_id: str, domain: str, table: str) -> None:
        """One commit for both: a table lives on a domain."""
        def mut(dd):
            it = S.item(dd, item_id)
            it.setdefault("chart", {})["domain"] = domain
            it["chart"]["table"] = table
        self.commit(_mutated(self._doc, mut))

    def promote_chart_data(self, item_id: str, *, aggregate: Optional[Mapping[str, Any]] = None,
                           where: Optional[Sequence[Sequence[Any]]] = None) -> str:
        """Give a source-reading chart a data entry of its own source and table — the
        place a filter and a summary are typed — and point it there."""
        chart = S.item(self._doc, item_id).get("chart") or {}
        if chart.get("data"):
            return str(chart["data"])
        sid = chart.get("source")
        if not sid:
            raise ValueError("the chart reads no source")
        did = self.add_data([sid], domain=str(chart.get("domain") or "label"),
                            table=str(chart.get("table") or ""), where=where,
                            aggregate=aggregate)

        def mut(dd):
            ch = S.item(dd, item_id).setdefault("chart", {})
            ch["data"] = did
            ch["table"] = ""                     # the entry's terminal table, resolved live
        self.commit(_mutated(self._doc, mut))
        return did

    def _chart_payload(self, chart: Mapping[str, Any]) -> Tuple[str, Any, Any]:
        """``(run id, payload, envelope)`` of what a chart reads: its data entry when it
        names one, else its source."""
        key = str(chart.get("data") or chart.get("source") or "")
        rid = self._run_ids.get(key, "")
        return rid, self._payloads.get(rid), self._envs.get(key)

    def _tables_of(self, rid: str, payload: Any) -> Any:
        have = self._tables.get(rid)
        if have is not None and have[0] == id(payload):
            return have[1]
        tables = CD.chart_tables(payload)
        self._tables[rid] = (id(payload), tables)
        return tables

    def _chart_table_name(self, chart: Mapping[str, Any]) -> str:
        name = str(chart.get("table") or "")
        if not name and chart.get("data") and self._fig is not None:
            name = self._fig.data_table_name(str(chart["data"]))
        return name

    def chart_tables_of(self, spec: Mapping[str, Any]) -> List[Tuple[str, str]]:
        """``[(domain, table)]`` a chart item may read, from the payload in hand, else the
        envelope of its source (a data entry's summary, when its envelope is not in yet)."""
        chart = spec.get("chart") or {}
        rid, payload, _env = self._chart_payload(chart)
        if payload is not None:
            keys = sorted(self._tables_of(rid, payload))
            if keys:
                return [(str(d), str(n)) for d, n in keys]
        key = str(chart.get("data") or chart.get("source") or "")
        out = [(d, t) for d, t, _c in self._envelope_tables(key)]
        if not out and chart.get("data"):
            entry = next((e for e in self._doc.get("data", []) if e["id"] == chart["data"]), None)
            if entry is not None:
                name = self._chart_table_name(chart)
                out = [(str(entry.get("domain") or "label"), name)] if name else []
                if entry.get("aggregate"):
                    out = [("label", name)]
        return out

    def chart_columns(self, spec: Mapping[str, Any]) -> List[str]:
        """The columns of the table a chart reads: the payload's when in hand, else the
        envelope's, else (a data entry not computed yet) what its stages will produce."""
        chart = spec.get("chart") or {}
        rid, payload, _env = self._chart_payload(chart)
        name = self._chart_table_name(chart)
        dom = str(chart.get("domain") or "label")
        if payload is not None:
            got = CD.pick_table(self._tables_of(rid, payload), dom, name)
            if got is not None:
                return sorted(got[1])
        key = str(chart.get("data") or chart.get("source") or "")
        tables = self._envelope_tables(key)
        entry = None
        if chart.get("data"):
            entry = next((e for e in self._doc.get("data", []) if e["id"] == chart["data"]), None)
            if not tables and entry is not None:
                tables = self._envelope_tables(str((entry.get("from") or [""])[0]))
                agg = entry.get("aggregate")
                if agg:
                    gb = agg.get("group_by") or ["t"]
                    gb = [g.strip() for g in gb.split(",")] if isinstance(gb, str) else list(gb)
                    red = agg.get("reducers") or ["mean", "sem", "count"]
                    red = [r.strip() for r in red.split(",")] if isinstance(red, str) else list(red)
                    return sorted(set(["id", "m", "t", "c", "z", "y", "x", "n"] + gb
                                      + [f"{agg.get('value') or 'area'}_{r}" for r in red]))
                name = str(entry.get("table") or "")
                dom = str(entry.get("domain") or dom)
        for d, l, cols in tables:
            if d == dom and (not name or l == name):
                return sorted(cols)
        for _d, l, cols in tables:
            if name and l == name:
                return sorted(cols)
        return sorted({c for _d, _l, cols in tables for c in cols})

    def _refresh_charts(self) -> None:
        """Resolve every chart whose inputs changed — synchronously: a chart reads tables
        already in memory, and the geometry is a few thousand floats."""
        page = self._doc.get("page") or {}
        for spec in self._doc.get("items", []):
            if spec.get("kind") != "chart":
                continue
            iid = spec["id"]
            item = self._items.get(iid)
            if not isinstance(item, ChartItem):
                continue
            chart = dict(spec.get("chart") or {})
            rid, payload, env = self._chart_payload(chart)
            if payload is None:
                what = chart.get("data") or chart.get("source")
                note = ("pick a source or a data entry" if not what else
                        f"{what!r} is not bound" if not rid else
                        "not computed yet — Compute sources")
                item.set_geometry(None, note=note)
                self._renders.pop(iid, None)
                continue
            key = S.digest({"chart": chart, "rect": spec["rect_mm"], "page": page,
                            "payload": id(payload), "rid": rid})
            have = self._renders.get(iid)
            if have is not None and have[0] == key:
                continue
            chart["table"] = self._chart_table_name(chart)
            md = dict(getattr(env, "metadata", None) or {})
            if not md and chart.get("source"):
                md = dict(getattr(self._envs.get(str(chart["source"])), "metadata", None) or {})
            try:
                series, units, note = CD.resolve_series(chart, self._tables_of(rid, payload), md)
                geometry = (CD.chart_geometry(chart, series, spec["rect_mm"], page, units=units)
                            if not note else None)
            except Exception as exc:           # noqa: BLE001 — a chart that cannot draw says why
                series, geometry, note = None, None, f"{type(exc).__name__}: {exc}"
            item.set_geometry(geometry, series, note)
            self._renders[iid] = (key, "chart")

    def chart_csv(self, item_id: str) -> str:
        """The values a chart draws, as CSV (the export bundle's data file)."""
        item = self._items.get(item_id)
        if not isinstance(item, ChartItem) or item.series is None:
            return ""
        return CD.chart_csv(S.item(self._doc, item_id).get("chart") or {}, item.series)

    def _on_hover(self, item_id: str, pos: Any) -> None:
        """A chart hover: the data under the pointer in the status line, and the hovered
        marker's object ringed on every image of the chart's source (or on the targets of a
        ``highlight`` link naming the chart)."""
        item = self._items.get(item_id)
        info = item.hover(pos) if isinstance(item, ChartItem) else {}
        row = str(info.get("row") or "")
        targets: set = set()
        yx = None
        if row:
            parts = row.split(",")
            if len(parts) == 5 and parts[3] and parts[4]:
                try:
                    yx = (float(parts[3]), float(parts[4]))
                except ValueError:
                    yx = None
            chart = S.item(self._doc, item_id).get("chart") or {}
            src = chart.get("source")
            if chart.get("data"):
                entry = next((d for d in self._doc.get("data", []) if d["id"] == chart["data"]),
                             None)
                frm = (entry or {}).get("from") or []
                src = frm[0] if len(frm) == 1 else None
            links = [ln for ln in self._doc.get("links", [])
                     if ln.get("kind") == "highlight" and ln.get("chart") == item_id]
            if links:
                targets = {str(t) for ln in links for t in ln.get("targets", [])}
            elif src:
                targets = {it["id"] for it in self._doc.get("items", [])
                           if it.get("kind") == "image"
                           and S.effective_image(self._doc, it["id"]).get("source") == src}
        for iid, it in self._items.items():
            if isinstance(it, ImageItem) and not isinstance(it, MontageItem):
                want = yx if (yx is not None and iid in targets) else None
                if getattr(it, "_highlight", None) != want:
                    it.set_highlight(want)
        if info:
            x, y = info.get("x"), info.get("y")
            fmt = lambda v: (f"{v:.4g}" if isinstance(v, float) else str(v)) if v is not None else "–"
            text = f"x = {fmt(x)}   y = {fmt(y)}" if info.get("inside") else ""
            if info.get("label"):
                text += f"   {info['label']}"
            if row:
                m, t, oid, _y, _x = row.split(",")
                text += f"   object {oid}" + (f" (m{m} t{t})" if m or t else "")
            if text != self._hover_text:
                self._hover_text = text
                self._status.setText(text)
        elif self._hover_text:
            self._hover_text = ""
            self._sync_status()

    def add_popout(self, parent_id: str, window: Sequence[float],
                   pos_mm: Optional[Sequence[float]] = None, *, magnify: float = 2.5) -> str:
        """A popout of ``parent_id`` showing the fraction ``window`` ``(fy0, fy1, fx0,
        fx1)`` of its plane, with a zoom link that draws the box on the parent."""
        parent = S.item(self._doc, parent_id)
        pimg = S.effective_image(self._doc, parent_id)
        fy0, fy1, fx0, fx1 = (float(v) for v in window)
        pr = parent["rect_mm"]
        w = max(5.0, pr[2] * (fx1 - fx0) * magnify)
        h = max(5.0, pr[3] * (fy1 - fy0) * magnify)
        pos = list(pos_mm) if pos_mm else [pr[0] + pr[2] + 4.0, pr[1]]
        iid = self._new_item_id("zoom")
        img = copy.deepcopy(pimg)
        img.pop("layers", None)
        img["region"] = {"from": "rect_frac", "rect_frac": [fy0, fy1, fx0, fx1]}
        if parent.get("kind") == "montage":
            img["frame"] = dict(img.get("frame") or {})
            img["frame"]["m"] = int(self._cursor.get("m", 0))
        item = {"id": iid, "kind": "image", "rect_mm": [pos[0], pos[1], w, h],
                "letter": self._next_letter(), "image": img}
        link = {"kind": "zoom", "parent": parent_id, "child": iid}

        def mut(dd):
            dd.setdefault("items", []).append(item)
            dd.setdefault("links", []).append(link)
        self.commit(_mutated(self._doc, mut))
        self.select([iid])
        return iid

    def add_twin(self, item_id: str, pos_mm: Optional[Sequence[float]] = None) -> str:
        """A twin of an image: same region, frame and channels, its own layers."""
        src = S.item(self._doc, item_id)
        r = src["rect_mm"]
        pos = list(pos_mm) if pos_mm else [r[0], r[1] + r[3] + 3.0]
        iid = self._new_item_id("twin")
        item = {"id": iid, "kind": "image", "rect_mm": [pos[0], pos[1], r[2], r[3]],
                "letter": self._next_letter(),
                "image": {"layers": [], "decor": dict(src.get("image", {}).get("decor") or {})}}
        link = {"kind": "twin", "a": item_id, "b": iid, "share": ["region", "frame", "channels"]}

        def mut(dd):
            dd.setdefault("items", []).append(item)
            dd.setdefault("links", []).append(link)
        self.commit(_mutated(self._doc, mut))
        self.select([iid])
        return iid

    def add_colorbar_for(self, item_id: str, layer: int = 0) -> str:
        it = S.item(self._doc, item_id)
        r = it["rect_mm"]
        img = S.effective_image(self._doc, item_id)
        layers = img.get("layers") or []
        label = ""
        if layers and layer < len(layers):
            L = layers[layer]
            lim = L.get("limits") or {}
            col = L.get("column") or L.get("layer") or ""
            label = (f"{col} / {lim.get('vmax'):g}" if lim.get("mode") == "normalized"
                     and lim.get("vmax") else col)
        return self.add_item("colorbar", [r[0], r[1] + r[3] + 1.5], [r[2], 6.0],
                             of={"item": item_id, "layer": layer}, label=label)

    def add_histogram_for(self, item_id: str, channel: int = 0) -> str:
        it = S.item(self._doc, item_id)
        r = it["rect_mm"]
        return self.add_item("histogram", [r[0] + r[2] + 2.0, r[1] + r[3] - 12.0],
                             [min(30.0, r[2]), 12.0], of={"item": item_id, "channel": channel})

    def share_lut(self, ids: Sequence[str]) -> None:
        """One display window per channel across ``ids``: the first leads, the others'
        channels become ``shared`` and a LUT-share link records the group."""
        ids = [i for i in ids if S.item(self._doc, i).get("kind") in ("image", "montage")]
        if len(ids) < 2:
            return

        def mut(dd):
            for it in dd["items"]:
                if it["id"] in ids[1:]:
                    for ch in (it.get("image") or {}).get("channels") or []:
                        ch["mode"] = "shared"
            dd.setdefault("links", []).append({"kind": "lut_share", "items": list(ids)})
        self.commit(_mutated(self._doc, mut))

    def add_layer(self, item_id: str, domain: str, layer: str, kind: str,
                  column: str = "") -> None:
        spec: Dict[str, Any] = {"domain": domain, "layer": layer, "kind": kind}
        if kind == "field":
            spec["column"] = column
            spec["look"] = {"style": "smooth"}
        elif kind == "labels":
            spec["look"] = {"outline": True, "fill": False, "line_width": 0.6}
        elif kind == "tracks":
            spec["look"] = {"tail": 0, "line_width": 0.6}
        elif kind == "points":
            spec["look"] = {"shape": "ring", "spread": 2}

        def mut(dd):
            for it in dd["items"]:
                if it["id"] == item_id:
                    it.setdefault("image", {}).setdefault("layers", []).append(spec)
        self.commit(_mutated(self._doc, mut))

    def remove_layer(self, item_id: str, index: int) -> None:
        def mut(dd):
            for it in dd["items"]:
                if it["id"] == item_id:
                    layers = it.get("image", {}).get("layers", [])
                    if 0 <= index < len(layers):
                        layers.pop(index)
        self.commit(_mutated(self._doc, mut))

    def add_channel(self, item_id: str) -> None:
        def mut(dd):
            for it in dd["items"]:
                if it["id"] == item_id:
                    chans = it.setdefault("image", {}).setdefault("channels", [])
                    used = {int(c.get("c", 0)) for c in chans}
                    c = next((k for k in range(64) if k not in used), 0)
                    chans.append({"c": c, "rgb": _emission_hex(None, c), "mode": "auto_pct"})
        self.commit(_mutated(self._doc, mut))

    def remove_channel(self, item_id: str, index: int) -> None:
        def mut(dd):
            for it in dd["items"]:
                if it["id"] == item_id:
                    chans = it.get("image", {}).get("channels", [])
                    if 0 <= index < len(chans):
                        chans.pop(index)
        self.commit(_mutated(self._doc, mut))

    def remove_link(self, link: Mapping[str, Any]) -> None:
        key = S.canonical_json(link)

        def mut(dd):
            dd["links"] = [ln for ln in dd.get("links", []) if S.canonical_json(ln) != key]
        self.commit(_mutated(self._doc, mut))

    def edit_item(self, item_id: str, path: str, value: Any) -> None:
        """Set ``path`` (dotted, under the item) to ``value`` and commit."""
        if path == "image.frame.m" or path == "image.frame.t" or path == "image.frame.z":
            if value not in ("cursor", "max", "mean"):
                value = int(value)

        def mut(dd):
            for it in dd["items"]:
                if it["id"] == item_id:
                    if path == "letter":
                        it["letter"] = str(value or "")
                    else:
                        set_path(it, path, value)
        self.commit(_mutated(self._doc, mut))

    def edit_page(self, path: str, value: Any) -> None:
        def mut(dd):
            page = dd.setdefault("page", {})
            if path == "preset" and value in FL.PAGE_PRESETS:
                page["preset"] = value
                page["width_mm"] = FL.PAGE_PRESETS[value]["width_mm"]
                page["height_mm"] = FL.PAGE_PRESETS[value]["height_mm"]
            else:
                set_path(page, path, value)
                if path in ("width_mm", "height_mm"):
                    page["preset"] = "custom"
        self.commit(_mutated(self._doc, mut))

    def set_goal(self, goal: Optional[str]) -> None:
        self.edit(lambda d: d.__setitem__("goal", goal))

    def set_geometry(self, item_id: str, rect_mm: Sequence[float]) -> None:
        def mut(dd):
            for it in dd["items"]:
                if it["id"] == item_id:
                    it["rect_mm"] = [float(v) for v in rect_mm]
        self.commit(_mutated(self._doc, mut))

    def delete_items(self, ids: Sequence[str]) -> None:
        ids = set(ids)
        if not ids:
            return

        def mut(dd):
            dd["items"] = [it for it in dd.get("items", []) if it["id"] not in ids]
            keep = []
            for ln in dd.get("links", []):
                refs = []
                for key in S.LINK_KINDS.get(ln.get("kind"), ()):
                    v = ln.get(key)
                    refs.extend(v if isinstance(v, list) else [v])
                if ids & set(refs):
                    if ln.get("kind") in ("lut_share", "cursor"):
                        rest = [i for i in ln.get("items", []) if i not in ids]
                        if len(rest) >= 2:
                            ln = dict(ln, items=rest)
                            keep.append(ln)
                    continue
                keep.append(ln)
            dd["links"] = keep
        self.commit(_mutated(self._doc, mut))

    def nudge(self, ids: Sequence[str], dx_mm: float, dy_mm: float) -> None:
        ids = set(ids)

        def mut(dd):
            for it in dd["items"]:
                if it["id"] in ids and not it.get("locked"):
                    it["rect_mm"][0] = float(it["rect_mm"][0]) + dx_mm
                    it["rect_mm"][1] = float(it["rect_mm"][1]) + dy_mm
        self.commit(_mutated(self._doc, mut))

    def align_selected(self, mode: str) -> None:
        ids = list(self.selected_ids())
        rects = [tuple(S.item(self._doc, i)["rect_mm"]) for i in ids]
        new = FL.align(rects, mode)

        def mut(dd):
            for it in dd["items"]:
                if it["id"] in ids:
                    it["rect_mm"] = list(new[ids.index(it["id"])])
        self.commit(_mutated(self._doc, mut))

    def distribute_selected(self, axis: str) -> None:
        ids = list(self.selected_ids())
        rects = [tuple(S.item(self._doc, i)["rect_mm"]) for i in ids]
        new = FL.distribute(rects, axis)

        def mut(dd):
            for it in dd["items"]:
                if it["id"] in ids:
                    it["rect_mm"] = list(new[ids.index(it["id"])])
        self.commit(_mutated(self._doc, mut))

    def checks(self) -> List[Dict[str, Any]]:
        return FL.page_checks(self._doc)

    # ── selection ──────────────────────────────────────────────────────────────
    def selected_ids(self) -> List[str]:
        return [it.item_id for it in self.scene.selectedItems() if hasattr(it, "item_id")]

    def select(self, ids: Sequence[str]) -> None:
        self.scene.blockSignals(True)
        self.scene.clearSelection()
        for i in ids:
            it = self._items.get(i)
            if it is not None:
                it.setSelected(True)
        self.scene.blockSignals(False)
        self._on_selection()

    def _on_selection(self) -> None:
        if self._rebuilding:
            return
        self.props.rebuild()
        self.selection_changed.emit()

    # ── helpers the Properties panel asks for ──────────────────────────────────
    def source_layers(self, source_id: Optional[str]) -> List[Tuple[str, str, str, List[str]]]:
        env = self._envs.get(source_id or "")
        d = describe_envelope(env)
        out: List[Tuple[str, str, str, List[str]]] = []
        for layer, cols in d.get("labels", []):
            out.append(("label", layer, "labels", cols))
        for layer, cols in d.get("points", []):
            out.append(("point", layer, "points", cols))
        for layer, cols in d.get("tracks", []):
            out.append(("track", layer, "tracks", cols))
        for layer, cols in d.get("fields", []):
            out.append(("point", layer, "field", cols))
            out.append(("point", layer, "vectors", cols))
        for layer in d.get("mesh", []):
            out.append(("mesh", layer, "mesh", []))
        return out

    def axes_of(self, item: Optional[Mapping[str, Any]]) -> Dict[str, int]:
        if item is None:
            return {}
        img = S.effective_image(self._doc, str(item["id"]))
        env = self._envs.get(img.get("source") or "")
        return dict(describe_envelope(env).get("axes") or {})

    # ── scene ↔ document ───────────────────────────────────────────────────────
    def _style(self) -> Dict[str, Any]:
        return dict(self._doc.get("page") or {})

    def _rebuild(self) -> None:
        """The scene from the document: items created, updated or removed by id."""
        self._rebuilding = True
        try:
            page = self._doc.get("page") or {}
            self.scene.set_page(float(page.get("width_mm", 85)), float(page.get("height_mm", 65)),
                                dark=bool(page.get("dark")), grid_mm=float(page.get("grid_mm", 1)))
            style = self._style()
            selected = set(self.selected_ids())
            want = {it["id"]: it for it in self._doc.get("items", [])}
            for iid in list(self._items):
                if iid not in want:
                    self.scene.removeItem(self._items.pop(iid))
                    self._renders.pop(iid, None)
            for iid, spec in want.items():
                it = self._items.get(iid)
                if it is None or it.spec.get("kind") != spec.get("kind"):
                    if it is not None:
                        self.scene.removeItem(it)
                    it = make_item(spec, style)
                    self._items[iid] = it
                    self.scene.addItem(it)
                else:
                    it.apply_spec(spec, style)
                if iid in selected:
                    it.setSelected(True)
            # leaders of zoom links
            pairs = {(str(ln["parent"]), str(ln["child"])) for ln in LK.zoom_links(self._doc)}
            for key in list(self._leaders):
                if key not in pairs:
                    self.scene.removeItem(self._leaders.pop(key))
            for key in pairs:
                if key not in self._leaders:
                    ln = next(l for l in LK.zoom_links(self._doc)
                              if (str(l["parent"]), str(l["child"])) == key)
                    box = ln.get("box") or {}
                    lead = LeaderItem(str(box.get("color", "#ffffff")),
                                      float(box.get("width_pt", 0.75)) * 0.7)
                    lead.setVisible(bool(box.get("leaders", True)))
                    self._leaders[key] = lead
                    self.scene.addItem(lead)
            i = next((k for k in range(self._style_combo_count()) if
                      self._style_combo.itemData(k) == page.get("style")), -1)
            if i >= 0 and self._style_combo.currentIndex() != i:
                self._style_combo.blockSignals(True)
                self._style_combo.setCurrentIndex(i)
                self._style_combo.blockSignals(False)
            self._sync_cursor_ranges()
            self._refresh_renders()
            self._sync_start()
            self._undo_btn.setEnabled(bool(self._undo))
            self._redo_btn.setEnabled(bool(self._redo))
        finally:
            self._rebuilding = False
        self.props.rebuild()
        self._sync_status()

    def _style_combo_count(self) -> int:
        return self._style_combo.count()

    def _sync_start(self) -> None:
        show = (self._fig is not None and not self._doc.get("items")
                and not self._doc.get("_dismissed"))
        self.start.setVisible(show)
        if show:
            self._place_start()

    def _place_start(self) -> None:
        vp = self.view.viewport()
        w = min(max(360, vp.width() - 40), 720)
        h = min(max(260, vp.height() - 40), 520)
        self.start.setGeometry((vp.width() - w) // 2, (vp.height() - h) // 2, w, h)
        self.start.raise_()

    def resizeEvent(self, e) -> None:                                     # noqa: N802
        super().resizeEvent(e)
        if self.start.isVisible():
            self._place_start()

    def _dismiss_start(self) -> None:
        self.start.hide()
        self._doc["_dismissed"] = True

    def _on_goal(self, goal: str) -> None:
        self.set_goal(goal if goal in S.GOALS else None)
        self.start.hide()
        self._doc["_dismissed"] = True
        cat = [(c, describe_envelope(c.get("env"))) for c in self.sources_panel.catalog]
        imgs = [c for c, d in cat if d.get("image")]
        tabs = [c for c, d in cat if d.get("labels") or d.get("points") or d.get("tracks")]
        if goal == "montage":
            if imgs:
                self.add_montage(str(imgs[0]["value"]))
        elif goal == "overlay":
            if imgs:
                self.add_image(str(imgs[0]["value"]))
        elif goal == "plot":
            if tabs:
                self.add_chart(str(tabs[0]["value"]))
        elif goal == "composite":
            if imgs:
                self.add_image(str(imgs[0]["value"]))
            if tabs:
                self.add_chart(str(tabs[0]["value"]))

    def _sync_status(self) -> None:
        if self._fig is None:
            self._status.setText("")
            return
        missing = [sid for sid, rid in self._run_ids.items() if rid not in self._payloads]
        refusals = LK.zoom_refusals(self._doc, self._envs, self._cursor)
        n_chk = len([c for c in self.checks() if c["severity"] == "warn"])
        parts = []
        if missing:
            parts.append(f"{len(missing)} source{'s' if len(missing) != 1 else ''} not computed")
        if refusals:
            parts.append(next(iter(refusals.values())))
        if n_chk:
            parts.append(f"{n_chk} check{'s' if n_chk != 1 else ''} to look at")
        self._status.setProperty("role", "muted")
        self._status.setText(" · ".join(parts))
        self._compute_btn.setEnabled(bool(missing))

    def _sync_cursor_ranges(self) -> None:
        mx = {"m": 1, "t": 1, "z": 1}
        for env in self._envs.values():
            ax = describe_envelope(env).get("axes") or {}
            for k in mx:
                mx[k] = max(mx[k], int(ax.get(k, 1) or 1))
        for k, sp in self._cursor_spins.items():
            sp.blockSignals(True)
            sp.setRange(0, max(0, mx[k] - 1))
            sp.setValue(min(self._cursor[k], max(0, mx[k] - 1)))
            sp.blockSignals(False)

    def _set_cursor(self, axis: str, value: int) -> None:
        self._cursor[axis] = int(value)
        for iid in list(self._renders):
            it = self._items.get(iid)
            if isinstance(it, ImageItem):
                img = S.effective_image(self._doc, iid)
                if axis in LK.follows_cursor(img) or it.spec.get("kind") == "montage":
                    self._renders.pop(iid, None)
        self._refresh_renders()

    def _on_style(self, i: int) -> None:
        if self._rebuilding:
            return
        preset = self._style_combo.itemData(i)
        if preset and preset != self._doc.get("page", {}).get("style"):
            self.edit_page("style", preset)
            if preset == "dark":
                self.edit_page("dark", True)
            elif self._doc.get("page", {}).get("dark"):
                self.edit_page("dark", False)

    # ── renders ────────────────────────────────────────────────────────────────
    def _invalidate_renders(self, run_id: str) -> None:
        for iid in list(self._renders):
            img = S.effective_image(self._doc, iid) if iid in self._items else {}
            if self._run_ids.get(img.get("source") or "") == run_id:
                self._renders.pop(iid, None)
        self._refresh_renders()

    def _render_key(self, iid: str, img: Mapping[str, Any]) -> str:
        rid = self._run_ids.get(img.get("source") or "", "")
        payload = self._payloads.get(rid)
        viewer = ""
        if any(ch.get("mode") == "viewer" for ch in img.get("channels") or []):
            viewer = S.canonical_json(self._host.viewer_state(rid, []))
        leader = LK.share_leader(self._doc, iid)
        shared = ""
        if leader and leader in self._renders:
            shared = self._renders[leader][0]
        return S.digest({"img": img, "cursor": self._cursor, "payload": id(payload),
                         "viewer": viewer, "shared": shared, "kind": S.item(self._doc, iid)["kind"],
                         "montage": S.item(self._doc, iid).get("montage")})

    def _refresh_renders(self) -> None:
        """Queue a render for every image item whose key changed; feed dependents."""
        if self._fig is None:
            return
        order = []
        for it in self._doc.get("items", []):
            if it["kind"] in ("image", "montage"):
                order.append(it["id"])
        # leaders of share groups first, so followers see their windows
        leaders = [g[0] for g in LK.share_groups(self._doc)]
        order.sort(key=lambda i: (i not in leaders))
        boxes = LK.zoom_boxes(self._doc, self._envs, self._cursor)
        for iid in order:
            item = self._items.get(iid)
            if not isinstance(item, ImageItem):
                continue
            item.zoom_boxes = boxes.get(iid, [])
            img = S.effective_image(self._doc, iid)
            rid = self._run_ids.get(img.get("source") or "", "")
            payload = self._payloads.get(rid)
            if payload is None:
                note = (f"source {img.get('source')!r} is not bound" if not rid
                        else "not computed yet — Compute sources")
                item.set_rendered(None, img, note)
                self._renders.pop(iid, None)
                continue
            key = self._render_key(iid, img)
            have = self._renders.get(iid)
            if have is not None and have[0] == key:
                continue
            self._gen += 1
            gen = self._gen
            self._renders[iid] = (key, None)
            fn = self._render_fn(iid, img, payload, rid)
            self._worker.request(iid, gen, fn)
        self._refresh_charts()
        self._feed_dependents()
        self._update_leaders()

    def _render_fn(self, iid: str, img: Mapping[str, Any], payload: Any, rid: str
                   ) -> Callable[[], Any]:
        """The closure the worker runs: everything it reads is captured now."""
        doc = self._doc
        cursor = dict(self._cursor)
        spec = S.item(doc, iid)
        env = self._envs.get(img.get("source") or "")
        md = dict(getattr(env, "metadata", None) or {})
        img = dict(img)
        img["_channel_names"] = list(md.get("channel_names") or [])
        img["_metadata"] = {k: md.get(k) for k in ("dt_s", "frame_time_jd", "pixel_size_um")}
        plan = B.read_plan(doc, iid, self._envs, cursor=cursor)
        viewer = {}
        if any(ch.get("mode") == "viewer" for ch in img.get("channels") or []):
            viewer = dict(self._host.viewer_state(rid, img["_channel_names"]))
        shared: Dict[int, Tuple[float, float]] = {}
        leader = LK.share_leader(doc, iid)
        if leader and leader in self._renders and self._renders[leader][1] is not None:
            lr = self._renders[leader][1]
            if isinstance(lr, R.Rendered):
                shared = dict(lr.clims)
            elif isinstance(lr, list) and lr and lr[0][1] is not None:
                shared = dict(lr[0][1].clims)
        own = list(self._host.own_layers(rid) or [])
        pal = self._palettes.get(id(payload))
        field_cache = self._field_cache
        montage = spec.get("montage") if spec.get("kind") == "montage" else None
        axes = describe_envelope(env).get("axes") or {}

        def one(coords):
            return R.render_image(payload, img, coords, window=plan.window, viewer_state=viewer,
                                  shared_clims=shared, own_layers=own, field_cache=field_cache,
                                  palette=pal)

        if montage is None:
            z = plan.z
            return lambda: one((plan.m, plan.t, z, 0 if not img.get("channels") else
                                int(img["channels"][0].get("c", 0))))

        def tiles():
            axis = str(montage.get("tile_by", "m"))
            n = int(axes.get(axis, 1) or 1)
            start, step = int(montage.get("start", 0)), max(1, int(montage.get("step", 1)))
            count = int(montage.get("count", 0)) or n
            idx = list(range(start, n, step))[:count]
            out = []
            names = img["_channel_names"]
            for i in idx:
                m, t, z = plan.m, plan.t, plan.z
                c = 0 if not img.get("channels") else int(img["channels"][0].get("c", 0))
                sub = dict(img)
                if axis == "m":
                    m = i
                elif axis == "t":
                    t = i
                elif axis == "z":
                    z = i
                elif axis == "c":
                    sub = dict(img, channels=[dict(ch, c=i) for ch in (img.get("channels")
                                                                       or [{}])[:1]])
                    c = i
                r = R.render_image(payload, sub, (m, t, z, c), window=plan.window,
                                   viewer_state=viewer, shared_clims=shared, own_layers=own,
                                   field_cache=field_cache, palette=pal)
                if str(montage.get("labels", "auto")) == "none":
                    label = ""
                elif axis == "m":
                    pn = list(md.get("position_name") or [])
                    label = str(pn[i]) if i < len(pn) and pn[i] else f"m {i}"
                elif axis == "t":
                    label = _elapsed_label(md, i)
                elif axis == "z":
                    zs = md.get("z_step_um")
                    label = f"z {i}" if not zs else f"{float(zs) * i:g} µm"
                else:
                    label = names[i] if i < len(names) and names[i] else f"ch{i + 1}"
                out.append(((m, t, z, c), r, label))
            return out
        return tiles

    def _on_rendered(self, iid: str, gen: int, result: Any) -> None:
        item = self._items.get(iid)
        have = self._renders.get(iid)
        if item is None or have is None:
            return
        key = have[0]
        if isinstance(result, Exception):
            self._renders.pop(iid, None)
            if isinstance(item, ImageItem):
                item.set_rendered(None, S.effective_image(self._doc, iid), f"{result}")
            self._status.setText(f"{iid}: {result}")
            return
        self._renders[iid] = (key, result)
        img = S.effective_image(self._doc, iid)
        if isinstance(item, MontageItem):
            item.set_tiles(result, img)
        elif isinstance(item, ImageItem):
            if isinstance(result, R.Rendered) and result.geometry is not None:
                pass
            payload = self._payloads.get(self._run_ids.get(img.get("source") or "", ""))
            if payload is not None and id(payload) not in self._palettes:
                try:
                    from nodelab_v2 import overlay_geometry as OG
                    self._palettes[id(payload)] = OG.build_palette(payload)
                except Exception:              # noqa: BLE001 — a colour is never worth a crash
                    pass
            item.set_rendered(result, img)
        # followers of a share group render after their leader
        if any(iid == g[0] for g in LK.share_groups(self._doc)):
            for grp in LK.share_groups(self._doc):
                if grp[0] == iid:
                    for f in grp[1:]:
                        self._renders.pop(f, None)
            self._refresh_renders()
            return
        self._feed_dependents()
        self._update_leaders()
        self._sync_status()

    def render_now(self, item_id: str) -> Any:
        """Render one image item synchronously (the probe and the export use this)."""
        item = self._items.get(item_id)
        if not isinstance(item, ImageItem):
            return None
        img = S.effective_image(self._doc, item_id)
        rid = self._run_ids.get(img.get("source") or "", "")
        payload = self._payloads.get(rid)
        if payload is None:
            return None
        key = self._render_key(item_id, img)
        result = self._render_fn(item_id, img, payload, rid)()
        self._renders[item_id] = (key, result)
        self._on_rendered(item_id, self._gen, result)
        return result

    def render_all_now(self) -> None:
        for grp in LK.share_groups(self._doc):
            self.render_now(grp[0])
        for it in self._doc.get("items", []):
            if it["kind"] in ("image", "montage"):
                self.render_now(it["id"])

    def _feed_dependents(self) -> None:
        """Colour bars, histograms, scale bars and measure lines read their image's render."""
        for iid, item in self._items.items():
            spec = item.spec
            of = spec.get("of") or {}
            src = self._items.get(str(of.get("item") or ""))
            r = self._renders.get(str(of.get("item") or ""), (None, None))[1]
            if isinstance(r, list):
                r = next((t[1] for t in r if t[1] is not None), None)
            if isinstance(item, ColorbarItem):
                lo, hi, cmap, label, vmax = 0.0, 1.0, "viridis", spec.get("label") or "", None
                if isinstance(src, ImageItem) and isinstance(r, R.Rendered):
                    layer = int(of.get("layer") or 0)
                    img = S.effective_image(self._doc, src.item_id)
                    layers = img.get("layers") or []
                    if r.geometry is not None and r.geometry.field is not None and \
                            layer < len(layers) and layers[layer].get("kind") == "field":
                        _fr, lims, cmap, flabel = r.geometry.field
                        lo, hi = lims
                        lim = layers[layer].get("limits") or {}
                        if lim.get("mode") == "normalized" and lim.get("vmax"):
                            vmax = float(lim["vmax"])
                        label = label or flabel
                    elif layer < len(layers) and layers[layer].get("kind") == "voxels" \
                            and r.geometry is not None and r.geometry.scalar is not None:
                        import numpy as _np
                        sc = r.geometry.scalar[_np.isfinite(r.geometry.scalar)]
                        if sc.size:
                            lo, hi = float(_np.percentile(sc, 1)), float(_np.percentile(sc, 99))
                        cmap = str(layers[layer].get("cmap") or "viridis")
                    elif not layers and r.clims:
                        c0 = min(r.clims)
                        lo, hi = r.clims[c0]
                        cmap = "grey"
                        label = label or "intensity"
                item.set_scale(lo, hi, cmap, label, vmax)
            elif isinstance(item, HistogramItem):
                if isinstance(r, R.Rendered):
                    c = int(of.get("channel") or 0)
                    plane = r.planes.get(c)
                    item.set_data(plane, r.clims.get(c), r.colors.get(c, (255, 255, 255)))
                else:
                    item.set_data(None, None)
            elif isinstance(item, (ScaleBarItem, MeasureLineItem)):
                units = None
                if isinstance(src, ImageItem) and isinstance(r, R.Rendered) and r.pixel_size_um:
                    pr = src.picture_rect()
                    units = pr.width() / max(1, r.plane_shape[1]) / float(r.pixel_size_um)
                item.set_scale(units)

    def _update_leaders(self) -> None:
        if self._rebuilding:
            return
        for (parent_id, child_id), lead in self._leaders.items():
            parent, child = self._items.get(parent_id), self._items.get(child_id)
            if not isinstance(parent, ImageItem) or child is None:
                lead.set_rects(QRectF(), QRectF())
                continue
            entry = next((e for e in parent.zoom_boxes if e[0] == child_id), None)
            if entry is None:
                lead.set_rects(QRectF(), QRectF())
                continue
            _c, win, box = entry
            if isinstance(parent, MontageItem):
                rect = parent.zoom_box_rect(win, int(box.get("_tile", 0)))
            else:
                rect = parent.zoom_box_rect(win)
            lead.set_rects(rect, child.sceneBoundingRect().adjusted(6, 6, -6, -6))

    # ── scene events ───────────────────────────────────────────────────────────
    def _on_geometry_edited(self, item_id: str, rect_mm: Any) -> None:
        if item_id in self._items:
            it = S.item(self._doc, item_id)
            cur = [float(v) for v in it["rect_mm"]]
            new = [round(float(v), 3) for v in rect_mm]
            if any(abs(a - b) > 1e-6 for a, b in zip(cur, new)):
                self.set_geometry(item_id, new)

    def _on_key(self, action: str, arg: Any) -> None:
        ids = self.selected_ids()
        if action == "delete":
            self.delete_items(ids)
        elif action == "nudge" and ids:
            self.nudge(ids, float(arg[0]), float(arg[1]))

    def _on_region_dragged(self, item_id: str, rect: QRectF) -> None:
        item = self._items.get(item_id)
        if not isinstance(item, ImageItem):
            return
        fy0, fx0 = item.picture_frac(rect.topLeft())
        fy1, fx1 = item.picture_frac(rect.bottomRight())
        if fy1 - fy0 < 0.01 or fx1 - fx0 < 0.01:
            return
        self.add_popout(item_id, (fy0, fy1, fx0, fx1))

    def _on_source_dropped(self, value: str, pos: QPointF) -> None:
        self.add_image(value, [to_mm(pos.x()), to_mm(pos.y())])

    def _on_column_dropped(self, item_id: str, payload: Mapping[str, Any]) -> None:
        if S.item(self._doc, item_id).get("kind") == "chart":
            self._apply_chart_payload(item_id, payload)
        else:
            self._apply_layer_payload(item_id, payload)

    def _on_layer_activated(self, payload: Mapping[str, Any]) -> None:
        ids = [i for i in self.selected_ids()
               if S.item(self._doc, i).get("kind") in ("image", "montage", "chart")]
        if ids and S.item(self._doc, ids[0]).get("kind") == "chart":
            self._apply_chart_payload(ids[0], payload)
        elif ids:
            self._apply_layer_payload(ids[0], payload)

    def _apply_chart_payload(self, item_id: str, payload: Mapping[str, Any]) -> None:
        """A column (or a table) dropped on a chart: the chart reads that table, and the
        column becomes its value — its Y for an XY chart, its X when the X is still blank
        on a chart that has a Y."""
        sid = self.source_id_for(str(payload.get("source"))) or \
            self.add_source(str(payload.get("source")))
        domain = str(payload.get("domain") or "label")
        table = str(payload.get("layer") or "")
        column = str(payload.get("column") or "")

        def mut(dd):
            ch = S.item(dd, item_id).setdefault("chart", {})
            if ch.get("data") is None:
                ch["source"] = sid
            if ch.get("data") is None or not ch.get("table"):
                ch["domain"], ch["table"] = domain, table
            if column:
                kind = str(ch.get("kind") or "xy")
                if kind in ("xy", "timeseries", "hexbin") and not ch.get("y"):
                    ch["y"] = column
                elif kind in ("xy", "hexbin") and not ch.get("x"):
                    ch["x"] = column
                elif kind in ("xy", "timeseries", "hexbin"):
                    ch["y"] = column
                elif kind == "bar" and not ch.get("x"):
                    ch["x"] = column
                else:
                    ch["value"] = column
        self.commit(_mutated(self._doc, mut))

    def _apply_layer_payload(self, item_id: str, payload: Mapping[str, Any]) -> None:
        img = S.effective_image(self._doc, item_id)
        sid = self.source_id_for(str(payload.get("source")))
        if sid is None or sid != img.get("source"):
            self._status.setText("a layer must come from the image's own source")
            return
        kind = str(payload.get("kind") or "labels")
        if payload.get("column") and payload.get("domain") == "point":
            kind = "field"
        self.add_layer(item_id, str(payload.get("domain")), str(payload.get("layer")), kind,
                       str(payload.get("column") or ""))

    # ── export ─────────────────────────────────────────────────────────────────
    def export(self, path: str, fmt: str, dpi: Optional[int] = None) -> str:
        self.render_all_now()
        page = self._doc.get("page") or {}
        out = self.scene.export(path, fmt, int(dpi or page.get("dpi", 300)))
        self._host.status(f"figure written: {out}")
        return out

    def export_dialog(self, fmt: str) -> None:
        path = self._host.export_path(fmt)
        if path:
            try:
                self.export(path, fmt)
            except Exception as exc:           # noqa: BLE001 — told, never raised into Qt
                self._status.setText(f"export failed: {exc}")


# ── small helpers ─────────────────────────────────────────────────────────────────────
def _mutated(doc: Mapping[str, Any], mutate: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    d = copy.deepcopy(dict(doc))
    d.pop("_dismissed", None)
    mutate(d)
    return d


def _emission_hex(nm: Any, c: int) -> str:
    """A channel's display colour from its emission wavelength (the Viewer's idiom), else
    a colour-blind-safe order: green, magenta, cyan, yellow."""
    fallback = ["#00ff88", "#ff40ff", "#40e0ff", "#ffe040"]
    try:
        w = float(nm)
    except (TypeError, ValueError):
        return fallback[c % len(fallback)]
    if w < 470:
        return "#4060ff"
    if w < 510:
        return "#40e0ff"
    if w < 570:
        return "#00ff88"
    if w < 600:
        return "#ffe040"
    if w < 640:
        return "#ff8040"
    return "#ff40ff"


def _default_scale_bar(md: Mapping[str, Any], axes: Mapping[str, int]) -> float:
    """A round scale-bar length about a fifth of the field."""
    try:
        px = float(md.get("pixel_size_um") or 0.0)
    except (TypeError, ValueError):
        px = 0.0
    if px <= 0:
        return 0.0
    width_um = px * float(axes.get("x", 1) or 1)
    want = width_um / 5.0
    for cand in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
        if cand >= want:
            return float(cand)
    return 5000.0


def _elapsed_label(md: Mapping[str, Any], t: int) -> str:
    jd = md.get("frame_time_jd")
    seconds = None
    if isinstance(jd, (list, tuple)) and t < len(jd) and jd[t] and jd[0]:
        try:
            seconds = (float(jd[t]) - float(jd[0])) * 86400.0
        except (TypeError, ValueError):
            seconds = None
    if seconds is None and md.get("dt_s"):
        try:
            seconds = float(md["dt_s"]) * t
        except (TypeError, ValueError):
            seconds = None
    if seconds is None:
        return f"t {t}"
    if seconds < 3600:
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
    return f"{int(seconds // 3600)}:{int((seconds % 3600) // 60):02d} h"


__all__ = ["FigureEditorPanel"]
