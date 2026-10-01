"""The Movie Editor: a bottom dock for one Export Movie node — sources, timeline, monitor.

Selecting an Export Movie node binds this panel to it (the window does that). It shows:

* **Sources** — ``A`` (the node's ``data``), ``B`` and ``C`` (``source_b``/``source_c``), each
  the REAL node feeding that socket (through reroutes and muted nodes), and whether its
  result is in hand. Nothing is computed until *Compute sources* is pressed: a preview must
  never commit the machine to a long pull the user did not ask for. Once a source has been
  computed it is kept fresh automatically (usually a memo hit).
* **Timeline** — the outline of the movie: clips and loops in play order, each clip's panels
  under it, with frame counts. Add, duplicate, delete and reorder from the buttons below it.
* **Monitor** — the frames the export would write, rendered by the export's own
  :class:`~nodegraph.catalog._shared.movie_timeline.Timeline` on a worker thread
  (latest request wins, a small frame cache), with a strip showing every segment to scale.
* **Properties** — the selected clip, loop or panel: what plays, which source and channels,
  per-channel colour/black/white/gamma (optionally LINKED to the Viewer's LUT for that
  source), label drawing, grids, burn-ins.

Every edit is written into the node's ``timeline`` param as canonical JSON on COMMIT (a field
finished, a button pressed), never during a drag, and pushed onto a local undo list — the
document itself has no undo. The panel never touches the runner or the document directly; it
asks its **host** (the window), which owns both. That seam is also what lets the GUI probe
drive the editor headless.
"""
from __future__ import annotations

import copy
import json
import math
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from nodegraph.catalog._shared import movie_timeline as MT
from nodelab_v2 import theme as T

#: Rendered frames kept for scrubbing back and for replaying a short loop smoothly.
_CACHE_FRAMES = 64

#: The source letters, in the order the editor lists them.
LETTERS = MT.SOURCES

def _editor_qss() -> str:
    """The editor's stylesheet, in the Inspector's idiom and from the same theme tokens, so
    the dock reads as part of the app in both themes (`restyle` re-applies it)."""
    def h(c: Any) -> str:
        return c.name() if c is not None else "#222"
    return f"""
QWidget#movieRoot, QWidget#movieProps {{ background:{h(T.PANEL)}; }}
QScrollArea {{ border:0; background:{h(T.PANEL)}; }}
QLabel, QCheckBox {{ color:{h(T.INK)}; }}
QLabel[role="muted"] {{ color:{h(T.MUTED)}; }}
QLabel[role="title"] {{ color:{h(T.INK)}; font-weight:700; }}
QLabel[role="error"] {{ color:{h(T.ERROR)}; }}
QGroupBox {{ color:{h(T.MUTED)}; border:1px solid {h(T.BORDER)}; border-radius:6px;
  margin-top:8px; padding-top:6px; font-weight:600; }}
QGroupBox::title {{ subcontrol-origin: margin; left:8px; padding:0 3px; }}
QTreeWidget {{ background:{h(T.BODY)}; color:{h(T.INK)}; border:1px solid {h(T.BORDER)};
  border-radius:6px; }}
QTreeWidget::item:selected {{ background:{h(T.ACCENT_DIM)}; color:{h(T.INK)}; }}
QHeaderView::section {{ background:{h(T.PANEL)}; color:{h(T.MUTED)}; border:0;
  padding:2px 6px; }}
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
  background:{h(T.BODY)}; color:{h(T.INK)}; border:1px solid {h(T.BORDER)};
  border-radius:6px; padding:3px 6px; min-height:16px; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
  border-color:{h(T.ACCENT)}; }}
QComboBox::drop-down {{ border:0; width:16px; }}
QPushButton, QToolButton {{ background:{h(T.BODY)}; color:{h(T.INK)};
  border:1px solid {h(T.BORDER)}; border-radius:6px; padding:3px 9px; }}
QToolButton {{ padding:2px 5px; }}
QPushButton:hover, QToolButton:hover {{ background:{h(T.PANEL_HI)}; }}
QPushButton:disabled, QToolButton:disabled {{ color:{h(T.MUTED)}; }}
"""


#: A selection in the outline: ``("seg", i)``, ``("panel", i, p)``, ``("bclip", i, j)`` or
#: ``("bpanel", i, j, p)`` — a top-level segment, a panel of a top-level clip, a clip inside
#: loop ``i``, or a panel of that clip.
Sel = Tuple[Any, ...]


def _target(spec: Dict[str, Any], sel: Sel) -> Dict[str, Any]:
    segs = spec["segments"]
    kind = sel[0]
    if kind == "seg":
        return segs[sel[1]]
    if kind == "panel":
        return segs[sel[1]]["panels"][sel[2]]
    if kind == "bclip":
        return segs[sel[1]]["body"][sel[2]]
    return segs[sel[1]]["body"][sel[2]]["panels"][sel[3]]


def _clip_of(spec: Dict[str, Any], sel: Sel) -> Optional[Dict[str, Any]]:
    """The clip a selection belongs to (``None`` for a loop itself)."""
    segs = spec["segments"]
    if sel[0] in ("seg", "panel"):
        seg = segs[sel[1]]
        return seg if seg["kind"] == "clip" else None
    return segs[sel[1]]["body"][sel[2]]


def _set_path(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _letters_used(spec: Dict[str, Any]) -> List[str]:
    """Every source letter the timeline reads, in first-use order."""
    out: List[str] = []

    def add(x: str) -> None:
        if x and x not in out:
            out.append(x)

    def clip(c: Dict[str, Any]) -> None:
        add(c["play"].get("source", ""))
        for p in c["panels"]:
            add(p["source"])

    for seg in spec["segments"]:
        if seg["kind"] == "loop":
            add(seg["source"])
            for c in seg["body"]:
                clip(c)
        else:
            clip(seg)
    return out


def segment_label(seg: Dict[str, Any]) -> str:
    """A short human name for a segment: its own name, else what it shows."""
    if seg.get("name"):
        return seg["name"]
    if seg["kind"] == "loop":
        inner = " + ".join(segment_label(c) for c in seg["body"])
        return f"loop over t: {inner}"
    p0 = seg["panels"][0]
    axis = seg["play"]["axis"]
    what = {"t": "plays t", "z": "plays z", "none": "still"}[axis]
    tile = p0["tile"]["axis"]
    grid = f", grid of {tile}" if tile != "none" else ""
    title = seg["annotations"].get("title")
    return f"{title} ({p0['source']}, {what}{grid})" if title else \
        f"{p0['source']} {what}{grid}"


# ── rendering off the GUI thread ──────────────────────────────────────────────────

class _Renderer(QObject):
    """One worker thread, latest request wins. A Timeline is only ever touched here.

    A plain ``threading.Thread`` rather than the shared Qt pool, for the reason the runner's
    pull thread gives: Python owns it, so it has one thread state for its life. The result
    comes back through a Qt signal, which crosses to the GUI thread queued."""

    rendered = Signal(int, int, object)          # generation, frame, rgb ndarray | Exception

    def __init__(self) -> None:
        super().__init__()
        self._cv = threading.Condition()
        self._job: Optional[Tuple[int, Any, int]] = None
        self._stop = False
        #: a render is running or waiting — :meth:`shutdown` and a probe wait on this
        self.busy = False
        self._thread = threading.Thread(target=self._loop, name="movie-editor-render",
                                        daemon=True)
        self._thread.start()

    def request(self, gen: int, timeline: Any, k: int) -> None:
        with self._cv:
            self._job = (gen, timeline, k)
            self.busy = True
            self._cv.notify()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop after the frame in hand. Called when the window closes: a daemon thread
        still inside OpenCV or Pillow when the interpreter tears down can take the process
        with it, so the app waits for it rather than racing it."""
        with self._cv:
            self._stop = True
            self._job = None
            self._cv.notify()
        self._thread.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self.busy = False
                    self._cv.wait()
                if self._stop:
                    self.busy = False
                    return
                gen, tl, k = self._job
                self._job = None
            try:
                out: Any = tl.render(k)
            except Exception as exc:      # noqa: BLE001 — shown in the monitor, never fatal
                out = exc
            try:
                self.rendered.emit(gen, k, out)
            except RuntimeError:          # the panel was destroyed while this ran
                return
            with self._cv:
                if self._job is None:
                    self.busy = False


class _TimelineStrip(QWidget):
    """Every segment drawn to scale by frame count, with the playhead. Click or drag to
    scrub; a click also selects the segment under it."""

    seek = Signal(int)
    picked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._spans: List[Tuple[int, int]] = []
        self._labels: List[str] = []
        self._total = 0
        self._cursor = 0
        self._selected = -1

    def set_timeline(self, spans: Sequence[Tuple[int, int]], labels: Sequence[str],
                     total: int) -> None:
        self._spans, self._labels, self._total = list(spans), list(labels), int(total)
        self.update()

    def set_cursor(self, k: int) -> None:
        self._cursor = int(k)
        self.update()

    def set_selected(self, i: int) -> None:
        self._selected = int(i)
        self.update()

    def _frame_at(self, x: float) -> int:
        if self._total <= 0:
            return 0
        return max(0, min(self._total - 1, int(x / max(1.0, self.width()) * self._total)))

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, T.BODY or QColor("#1d2026"))
        if self._total > 0:
            f = w / float(self._total)
            palette = (QColor("#3a6ea5"), QColor("#4f8a5b"), QColor("#8a5a9e"),
                       QColor("#9e7a3a"), QColor("#3a8a8a"))
            for i, (start, n) in enumerate(self._spans):
                r = QRectF(start * f, 4, max(2.0, n * f - 1), h - 8)
                col = QColor(palette[i % len(palette)])
                if i == self._selected:
                    col = col.lighter(140)
                p.fillRect(r, col)
                p.setPen(QColor("#f0f0f0"))
                label = self._labels[i] if i < len(self._labels) else ""
                text = p.fontMetrics().elidedText(f"{label}  ·  {n}", Qt.ElideRight,
                                                  int(max(0.0, r.width() - 6)))
                p.drawText(r.adjusted(4, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
            x = (self._cursor + 0.5) * f
            p.setPen(QPen(QColor("#ffdd55"), 2))
            p.drawLine(int(x), 0, int(x), h)
        p.end()

    def mousePressEvent(self, e) -> None:
        k = self._frame_at(e.position().x())
        for i, (start, n) in enumerate(self._spans):
            if start <= k < start + n:
                self.picked.emit(i)
                break
        self.seek.emit(k)

    def mouseMoveEvent(self, e) -> None:
        if e.buttons() & Qt.LeftButton:
            self.seek.emit(self._frame_at(e.position().x()))


class _Monitor(QLabel):
    """The picture, fitted to the widget and redrawn on resize."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        # IGNORED, not Expanding, and word-wrapped: a QLabel's minimum width is its text
        # (or its pixmap), so a one-line "source A is not computed" message demanded 924 px
        # and pushed the whole main window wider than the screen, squeezing the canvas and
        # putting the dock's own buttons out of reach. The monitor takes the room it is
        # given; the picture is fitted to it.
        self.setWordWrap(True)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background:#000; color:#c8c8c8;")
        self._pix: Optional[QPixmap] = None

    def show_rgb(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888)
        self._pix = QPixmap.fromImage(img.copy())     # QImage does not own the buffer
        self._fit()

    def show_text(self, text: str) -> None:
        self._pix = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def _fit(self) -> None:
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()


# ── the panel ──────────────────────────────────────────────────────────────────────

class MovieEditorPanel(QWidget):
    """The Movie Editor for one Export Movie node at a time. See the module docstring.

    ``host`` is the window's adapter: ``movie_state(nid)``, ``movie_sources(nid)``,
    ``source_payload(node)``, ``fetch(node)``, ``commit_timeline(nid, text)``,
    ``set_sweep(nid, value)``, ``live_display(nid, spec)``, ``capture(nid)``, ``export(nid)``.
    """

    def __init__(self, host: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._host = host
        self._nid: Optional[str] = None
        self._spec: Optional[Dict[str, Any]] = None          # the node's normalized timeline
        self._flat = False                                   # node plays its flat settings
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._payloads: Dict[str, Any] = {}                  # letter -> Dataset in hand
        self._wanted: set = set()                            # node ids asked to be computed
        self._undo: List[str] = []
        self._redo: List[str] = []
        self._sel: Sel = ("seg", 0)
        self._tl: Any = None
        self._gen = 0
        self._k = 0
        self._cache: "OrderedDict[Tuple[int, int], np.ndarray]" = OrderedDict()
        self._building = False
        self._committing = False
        self._props_pending = False

        self._renderer = _Renderer()
        self._renderer.rendered.connect(self._on_rendered)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self.setObjectName("movieRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── top bar ────────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        self._title = QLabel("Movie Editor — select an Export Movie node")
        tf = self._title.font(); tf.setBold(True); self._title.setFont(tf)
        self._title.setMinimumWidth(0)
        self._title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self._title)
        bar.addStretch(1)
        self._convert_btn = QPushButton("Convert to timeline")
        self._convert_btn.setToolTip(
            "Switch this node to Sweep = 'timeline', starting from exactly the movie it makes "
            "now (one clip). Channels on auto contrast are linked to the Viewer's LUTs.")
        self._convert_btn.clicked.connect(self.convert_to_timeline)
        self._fetch_btn = QPushButton("Compute sources")
        self._fetch_btn.setToolTip(
            "Compute the nodes feeding A, B and C that the timeline uses, in the background. "
            "Nothing is shown in the Viewer; the result is only used here. Usually a memo hit.")
        self._fetch_btn.clicked.connect(self.compute_sources)
        self._capture_btn = QPushButton("Capture LUTs")
        self._capture_btn.setToolTip(
            "Write the Viewer's current black/white, gamma and picked colours into every "
            "channel linked to the Viewer, now. This also happens by itself when you stop "
            "adjusting the Viewer, before an export and before a save.")
        self._capture_btn.clicked.connect(self.capture)
        self._undo_btn = QToolButton(); self._undo_btn.setText("Undo")
        self._undo_btn.clicked.connect(self.undo)
        self._redo_btn = QToolButton(); self._redo_btn.setText("Redo")
        self._redo_btn.clicked.connect(self.redo)
        self._export_btn = QPushButton("Export")
        self._export_btn.setToolTip("Pull this node: write the movie to its File.")
        self._export_btn.clicked.connect(self.export)
        for wdg in (self._convert_btn, self._fetch_btn, self._capture_btn, self._undo_btn,
                    self._redo_btn, self._export_btn):
            bar.addWidget(wdg)
        root.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # ── left: sources + outline ───────────────────────────────────────────
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        src_box = QGroupBox("Sources")
        sv = QVBoxLayout(src_box)
        sv.setContentsMargins(6, 4, 6, 4)
        self._src_rows: Dict[str, QLabel] = {}
        for L in LETTERS:
            lbl = QLabel(f"{L}  —")
            lbl.setWordWrap(True)
            sv.addWidget(lbl)
            self._src_rows[L] = lbl
        lv.addWidget(src_box)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Timeline", "frames"])
        self._tree.setColumnWidth(0, 210)
        self._tree.setMinimumWidth(180)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection)
        lv.addWidget(self._tree, 1)
        # two rows of four, not one row of eight: a single row set the whole editor's
        # minimum width (and with it the main window's)
        ops = QGridLayout()
        ops.setSpacing(2)
        self._op_btns: List[QToolButton] = []
        for text, tip, fn in (
                ("+Clip", "Add a clip that plays source A's t", lambda: self.add("clip")),
                ("+Grid", "Add a clip whose panel is tiled by position (an n x n grid)",
                 lambda: self.add("grid")),
                ("+Loop", "Add a loop over t: A's max-Z still, then a z sweep from B "
                          "(or A), for every timepoint", lambda: self.add("loop")),
                ("+Panel", "Add a panel to the selected clip (side by side)",
                 lambda: self.add("panel")),
                ("Dup", "Duplicate the selection", self.duplicate),
                ("Del", "Delete the selection", self.delete),
                ("↑", "Move earlier", lambda: self.move(-1)),
                ("↓", "Move later", lambda: self.move(+1))):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            n = len(self._op_btns)
            ops.addWidget(b, n // 4, n % 4)
            self._op_btns.append(b)
        lv.addLayout(ops)
        split.addWidget(left)

        # ── centre: monitor + strip + transport ───────────────────────────────
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        self._monitor = _Monitor()
        mv.addWidget(self._monitor, 1)
        self._strip = _TimelineStrip()
        self._strip.seek.connect(self.seek)
        self._strip.picked.connect(lambda i: self.select(("seg", i)))
        mv.addWidget(self._strip)
        tr = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self.toggle_play)
        tr.addWidget(self._play_btn)
        self._readout = QLabel("")
        self._readout.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        tr.addWidget(self._readout, 1)
        mv.addLayout(tr)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setProperty("role", "muted")
        mv.addWidget(self._status)
        split.addWidget(mid)

        # ── right: properties ────────────────────────────────────────────────
        self._props = QScrollArea()
        self._props.setWidgetResizable(True)
        self._props.setMinimumWidth(220)
        split.addWidget(self._props)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([260, 640, 340])

        self.restyle()
        self._sync_enabled()

    def sizeHint(self):                                   # noqa: N802 — Qt override
        """Tall enough for a usable monitor. The main window sizes a dock from this when it
        first appears (`resizeDocks` on a dock that has just been shown is discarded)."""
        from PySide6.QtCore import QSize
        base = super().sizeHint()
        return QSize(base.width(), max(base.height(), 440))

    def restyle(self) -> None:
        """Re-apply the theme (the window calls this on a light/dark switch)."""
        self.setStyleSheet(_editor_qss())
        self._strip.update()

    # ── binding ───────────────────────────────────────────────────────────────
    def bound(self) -> Optional[str]:
        return self._nid

    def bind(self, node_id: Optional[str]) -> None:
        """Show Export Movie node ``node_id`` (``None`` to clear). Re-binding the same node
        keeps the undo list and the playhead."""
        if node_id == self._nid and node_id is not None:
            self.reload()
            return
        self._nid = node_id
        self._undo.clear()
        self._redo.clear()
        self._payloads.clear()
        self._k = 0
        self._sel = ("seg", 0)
        self.stop()
        self.reload()

    def reload(self) -> None:
        """Re-read the node's state (spec, mode, wiring) and refresh everything."""
        st = self._host.movie_state(self._nid) if self._nid else None
        if st is None:
            self._nid, self._spec = None, None
            self._title.setText("Movie Editor — select an Export Movie node")
            self._tree.clear()
            self._props.setWidget(QWidget())
            self._monitor.show_text("Select an Export Movie node to edit its movie.")
            self._sync_enabled()
            return
        self._title.setText(f"Movie Editor — {st['label']}")
        self._flat = str(st["modes"].get("sweep", "time")) != "timeline"
        if self._flat:
            from nodelab_v2.movie_preview import flat_timeline_spec
            try:
                spec = flat_timeline_spec(st["spec"], st["params"], st["modes"], st["env"])
                for p in spec["segments"][0]["panels"]:
                    for d in p["display"].values():
                        d["link"] = "manual"      # the flat node does not read the Viewer
                self._spec = spec
            except Exception as exc:     # noqa: BLE001
                self._spec = None
                self._status.setText(f"cannot preview: {exc}")
        else:
            text = st["params"].get("timeline", "") or ""
            spec, err = MT.try_normalize(text)
            if spec is None and not text.strip():
                # Sweep was switched to 'timeline' without one: start from the movie the
                # node was making, and write it, so the node is playable as it stands.
                from nodelab_v2.movie_preview import flat_timeline_spec
                spec = flat_timeline_spec(st["spec"], st["params"], st["modes"], st["env"])
                self._spec = spec
                self._commit(MT.canonical_json(spec))
            elif spec is None:
                spec = MT.normalize_spec({"segments": [self._default_clip("clip")]})
                self._status.setText(f"The node's timeline could not be read ({err}); "
                                     f"showing a default clip until you edit it.")
            self._spec = spec
        self._refresh_sources()
        self._rebuild_tree()
        self._schedule_props()
        self._rebuild_preview()
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        # Everything is live on a flat node too: the first edit converts it to a timeline
        # (see `_to_timeline`). Locking the whole editor until a button was found and pressed
        # read as "nothing in here can be clicked".
        on = self._nid is not None and self._spec is not None
        self._convert_btn.setVisible(on and self._flat)
        for b in self._op_btns:
            b.setEnabled(on)
        self._capture_btn.setEnabled(on)
        self._undo_btn.setEnabled(on and bool(self._undo))
        self._redo_btn.setEnabled(on and bool(self._redo))
        self._export_btn.setEnabled(on)
        self._fetch_btn.setEnabled(on)
        self._props.setEnabled(on)

    # ── sources ───────────────────────────────────────────────────────────────
    def _refresh_sources(self) -> None:
        self._sources = self._host.movie_sources(self._nid) if self._nid else {}
        for L in LETTERS:
            info = self._sources.get(L) or {}
            node = info.get("node")
            ds = self._payloads.get(L)
            if node is not None and ds is None:
                have = self._host.source_payload(node)
                if have is not None:
                    self._payloads[L] = ds = have
            if node is None:
                text = f"{L}  not wired" + ("" if L != "A" else " — connect the node's data")
            else:
                state = ("ready" if ds is not None else
                         "computing…" if node in self._wanted else "not computed")
                text = f"{L}  {info.get('label', node)}  ·  {state}"
            self._src_rows[L].setText(text)

    def compute_sources(self) -> None:
        """Ask for every wired source the timeline uses that is not in hand."""
        if self._spec is None:
            return
        for L in _letters_used(self._spec):
            info = self._sources.get(L) or {}
            node = info.get("node")
            if node is not None and self._payloads.get(L) is None:
                self._wanted.add(node)
                self._host.fetch(node)
        self._refresh_sources()

    def on_fetched(self, node_id: str, payload: Any) -> None:
        """A source the host fetched arrived."""
        hit = False
        for L, info in self._sources.items():
            if (info or {}).get("node") == node_id:
                self._payloads[L] = payload
                hit = True
        self._wanted.discard(node_id)
        if hit:
            self._refresh_sources()
            self._rebuild_preview()

    def on_doc_changed(self, touched: Optional[frozenset], downstream: Callable) -> None:
        """The document changed: re-read the node, and drop any source payload the edit
        could have changed — one whose node lies downstream of (or is) a touched node, or
        every one for an edit that names nothing. A source that was computed before is
        asked for again at once (usually a memo hit), so the monitor stays live."""
        if self._nid is None or self._committing:
            return
        before = {L: (i or {}).get("node") for L, i in self._sources.items()}
        reach = None if touched is None else downstream(touched)
        for L, node in before.items():
            if L in self._payloads and (reach is None or node in reach):
                self._payloads.pop(L, None)
        self._refresh_sources()
        after = {L: (i or {}).get("node") for L, i in self._sources.items()}
        for L, node in after.items():
            if before.get(L) != node:
                self._payloads.pop(L, None)
            if node is not None and self._payloads.get(L) is None and (
                    node in self._wanted or before.get(L) == node):
                self._wanted.add(node)
                self._host.fetch(node)
        st = self._host.movie_state(self._nid)
        if st is None:
            self.bind(None)
            return
        # an edit that did not come from here (a reload, a param typed in the inspector)
        mine = self._canon()
        flat = str(st["modes"].get("sweep", "time")) != "timeline"
        theirs = st["params"].get("timeline", "") or ""
        if flat != self._flat or (not flat and theirs != mine):
            self.reload()
        elif touched is not None and touched <= {self._nid} and not flat:
            return                  # the echo of this panel's own commit: already shown
        else:
            if flat:
                self.reload()       # a flat param changed: re-derive the one-clip view
            else:
                self._rebuild_preview()

    def on_display_changed(self) -> None:
        """The Viewer's look changed: re-render linked panels with the live values."""
        if self._nid is not None and not self._flat:
            self._rebuild_preview()

    # ── the spec ──────────────────────────────────────────────────────────────
    def working_spec(self) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self._spec) if self._spec is not None else None

    def _canon(self) -> str:
        return MT.canonical_json(self._spec) if self._spec is not None else ""

    def apply(self, mutate: Callable[[Dict[str, Any]], None], *,
              structure: bool = True) -> bool:
        """Mutate a copy of the spec, validate it, and COMMIT it to the node. ``False`` (and
        the spec untouched) when the result does not validate; the message is shown.

        On a node still playing its flat settings the edit first converts it to a timeline,
        so any control works from the start. ``structure`` asks for the properties form to
        be rebuilt (a source, a render mode, the outline itself changed) — always on the
        NEXT event-loop turn, because the widget whose signal got us here may be one the
        rebuild would delete while it is still emitting."""
        if self._spec is None or self._nid is None:
            return False
        if self._flat:
            self._to_timeline()
        new = copy.deepcopy(self._spec)
        mutate(new)
        norm, err = MT.try_normalize(new)
        if norm is None:
            self._status.setText(err)
            return False
        text = MT.canonical_json(norm)
        old = self._canon()
        if text == old:
            return True
        self._undo.append(old)
        self._redo.clear()
        self._spec = norm
        self._commit(text)
        self._after_edit(structure=structure)
        return True

    def _after_edit(self, *, structure: bool = True) -> None:
        self._clamp_sel()
        self._rebuild_tree()
        if structure:
            self._schedule_props()
        self._rebuild_preview()
        self._sync_enabled()

    def _schedule_props(self) -> None:
        """Rebuild the properties form on the next event-loop turn (once, however often
        asked). Never synchronously: every caller may be inside a signal of a widget that
        lives in the form, and deleting a widget from its own press or toggle handler
        leaves Qt's mouse state pointing at a dead object — the editor stops taking
        clicks."""
        if not self._props_pending:
            self._props_pending = True
            QTimer.singleShot(0, self._run_scheduled_props)

    def _run_scheduled_props(self) -> None:
        self._props_pending = False
        self._rebuild_props()

    def _to_timeline(self) -> None:
        """Switch a flat node to Sweep = 'timeline', starting from the movie it makes now."""
        st = self._host.movie_state(self._nid)
        from nodelab_v2.movie_preview import flat_timeline_spec
        spec = MT.normalize_spec(
            flat_timeline_spec(st["spec"], st["params"], st["modes"], st["env"]))
        self._spec = spec
        self._flat = False
        self._undo.clear()
        self._redo.clear()
        self._commit(MT.canonical_json(spec), sweep="timeline")
        self._status.setText("This node now plays a timeline (Sweep = 'timeline'), starting "
                             "from the movie it made before.")

    def _commit(self, text: str, *, sweep: Optional[str] = None) -> None:
        """Hand ``text`` to the host as the node's timeline (and optionally set Sweep).

        The document notifies synchronously, so the host's write comes straight back here
        as :meth:`on_doc_changed`; ``_committing`` marks that echo as ours to ignore."""
        self._committing = True
        try:
            self._host.commit_timeline(self._nid, text)
            if sweep is not None:
                self._host.set_sweep(self._nid, sweep)
        finally:
            self._committing = False

    def undo(self) -> None:
        if not self._undo or self._nid is None:
            return
        self._redo.append(self._canon())
        text = self._undo.pop()
        self._spec = MT.normalize_spec(text)
        self._commit(text)
        self._after_edit()

    def redo(self) -> None:
        if not self._redo or self._nid is None:
            return
        self._undo.append(self._canon())
        text = self._redo.pop()
        self._spec = MT.normalize_spec(text)
        self._commit(text)
        self._after_edit()

    def convert_to_timeline(self) -> None:
        """Switch the node to Sweep = 'timeline', starting from the movie it makes now."""
        if self._nid is None:
            return
        if self._flat:
            self._to_timeline()
        self.reload()

    def capture(self) -> None:
        if self._nid is not None:
            if self._flat:
                self._to_timeline()
            self._committing = True
            try:
                self._host.capture(self._nid)
            finally:
                self._committing = False
            self.reload()

    def export(self) -> None:
        if self._nid is not None:
            self._host.export(self._nid)

    # ── structure edits ───────────────────────────────────────────────────────
    def _source_channels(self, letter: str) -> Tuple[List[str], int]:
        env = (self._sources.get(letter) or {}).get("env")
        n = int(getattr(getattr(env, "axes", None), "c", 1) or 1)
        md = dict(getattr(env, "metadata", {}) or {})
        # the payload's names first: channel names ride on the Dataset, not the envelope,
        # and they are what the movie's legend will print
        ds = self._payloads.get(letter)
        names = list((dict(getattr(ds, "metadata", {}) or {})).get("channel_names")
                     or md.get("channel_names") or [])
        return [str(names[c]) if c < len(names) and names[c] else f"C{c}"
                for c in range(n)], n

    def _voxel_layers(self, letter: str) -> List[str]:
        from nodegraph.domains import Domain
        env = (self._sources.get(letter) or {}).get("env")
        out = []
        for dom, name in list(getattr(env, "layer_names", ()) or ()):
            if dom is Domain.VOXEL and name not in out:
                out.append(name)
        return out

    def _default_panel(self, letter: str, **over: Any) -> Dict[str, Any]:
        """A new panel on ``letter``: all channels, each LINKED to the Viewer's LUT."""
        _names, n = self._source_channels(letter)
        panel = {"source": letter,
                 "display": {str(c): {"link": "viewer"} for c in range(n)}}
        panel.update(over)
        return panel

    def _default_clip(self, kind: str) -> Dict[str, Any]:
        wired = {L for L, i in self._sources.items() if (i or {}).get("node")}
        if kind == "grid":
            return {"kind": "clip", "play": {"axis": "t"},
                    "panels": [self._default_panel("A", tile={"axis": "m"})],
                    "annotations": {"title": "positions"}}
        if kind == "loop":
            second = "B" if "B" in wired else "A"
            layers = self._voxel_layers(second)
            zpanel = (self._default_panel(second, render="labels_over_image",
                                          layer=layers[0]) if layers
                      else self._default_panel(second))
            return {"kind": "loop", "source": "A", "body": [
                {"kind": "clip", "play": {"axis": "none"},
                 "panels": [self._default_panel("A", z="max")],
                 "annotations": {"title": "max-Z"}},
                {"kind": "clip", "play": {"axis": "z", "source": second,
                                          "direction": "alternate"},
                 "panels": [zpanel], "annotations": {"title": "Z stack"}}]}
        return {"kind": "clip", "play": {"axis": "t"}, "panels": [self._default_panel("A")]}

    def add(self, kind: str) -> None:
        """Add a clip / grid / loop after the selection, or a panel to the selected clip."""
        sel = self._sel

        def go(s: Dict[str, Any]) -> None:
            segs = s["segments"]
            if kind == "panel":
                clip = _clip_of(s, sel) if segs else None
                if clip is None:
                    return
                src = clip["panels"][-1]["source"] if clip["panels"] else "A"
                clip["panels"].append(self._default_panel(src))
                return
            new = self._default_clip(kind)
            if sel[0] in ("bclip", "bpanel") and kind != "loop":
                segs[sel[1]]["body"].insert(sel[2] + 1, new)     # loops are one level deep
                return
            segs.insert(min(len(segs), sel[1] + 1), new)

        if self.apply(go):
            if kind == "panel":
                clip = _clip_of(self._spec, sel)
                last = len(clip["panels"]) - 1 if clip else 0
                self.select(("panel", sel[1], last) if sel[0] in ("seg", "panel")
                            else ("bpanel", sel[1], sel[2], last))
            elif sel[0] in ("bclip", "bpanel") and kind != "loop":
                self.select(("bclip", sel[1], sel[2] + 1))
            else:
                self.select(("seg", min(len(self._spec["segments"]) - 1, sel[1] + 1)))

    def _container(self, s: Dict[str, Any], sel: Sel) -> Tuple[List[Any], int]:
        """The list the selected item lives in, and its index there."""
        segs = s["segments"]
        kind = sel[0]
        if kind == "seg":
            return segs, sel[1]
        if kind == "panel":
            return segs[sel[1]]["panels"], sel[2]
        if kind == "bclip":
            return segs[sel[1]]["body"], sel[2]
        return segs[sel[1]]["body"][sel[2]]["panels"], sel[3]

    def delete(self) -> None:
        sel = self._sel
        what = {"seg": "segment", "panel": "panel", "bclip": "clip", "bpanel": "panel"}[sel[0]]

        def go(s: Dict[str, Any]) -> None:
            lst, i = self._container(s, sel)
            if len(lst) <= 1:
                raise _Refused(f"cannot delete the only {what} here")
            del lst[i]

        try:
            self.apply(go)
        except _Refused as exc:
            self._status.setText(str(exc))

    def duplicate(self) -> None:
        sel = self._sel

        def go(s: Dict[str, Any]) -> None:
            lst, i = self._container(s, sel)
            lst.insert(i + 1, copy.deepcopy(lst[i]))

        if self.apply(go):
            self.select(sel[:-1] + (sel[-1] + 1,))

    def move(self, delta: int) -> None:
        sel = self._sel

        def go(s: Dict[str, Any]) -> None:
            lst, i = self._container(s, sel)
            j = i + delta
            if 0 <= j < len(lst):
                lst[i], lst[j] = lst[j], lst[i]

        lst, i = self._container(self._spec, sel) if self._spec else ([], 0)
        if 0 <= i + delta < len(lst) and self.apply(go):
            self.select(sel[:-1] + (sel[-1] + delta,))

    def edit(self, sel: Sel, dotted: str, value: Any, *, structure: bool = False) -> bool:
        """Set ``dotted`` (``"play.axis"``) on the selected clip/loop/panel and commit.
        ``structure`` for a value that changes which fields the form shows."""
        return self.apply(lambda s: _set_path(_target(s, sel), dotted, value),
                          structure=structure)

    # ── outline ───────────────────────────────────────────────────────────────
    def _clamp_sel(self) -> None:
        if self._spec is None:
            return
        try:
            _target(self._spec, self._sel)
        except (IndexError, KeyError, TypeError):
            self._sel = ("seg", max(0, min(self._sel[1], len(self._spec["segments"]) - 1)))

    def select(self, sel: Sel, *, from_tree: bool = False) -> None:
        """Make ``sel`` the selection. From the outline's own click the outline is left
        alone — it already shows it, and clearing a QTreeWidget inside its selection
        signal deletes the item under a press Qt is still handling."""
        self._sel = tuple(sel)
        self._clamp_sel()
        if from_tree:
            self._strip.set_selected(self._sel[1])
        else:
            self._rebuild_tree()
        self._schedule_props()
        spans = getattr(self._tl, "segment_spans", None)
        if spans and self._sel[1] < len(spans):
            self.seek(spans[self._sel[1]][0])

    def _rebuild_tree(self) -> None:
        self._building = True
        try:
            self._tree.clear()
            if self._spec is None:
                return
            spans = list(getattr(self._tl, "segment_spans", []) or [])
            pick = None
            for i, seg in enumerate(self._spec["segments"]):
                n = spans[i][1] if i < len(spans) else ""
                top = QTreeWidgetItem([segment_label(seg), str(n)])
                top.setData(0, Qt.UserRole, ("seg", i))
                self._tree.addTopLevelItem(top)
                if self._sel == ("seg", i):
                    pick = top
                if seg["kind"] == "loop":
                    for j, clip in enumerate(seg["body"]):
                        ci = QTreeWidgetItem([segment_label(clip), ""])
                        ci.setData(0, Qt.UserRole, ("bclip", i, j))
                        top.addChild(ci)
                        if self._sel == ("bclip", i, j):
                            pick = ci
                        for p, panel in enumerate(clip["panels"]):
                            pi = QTreeWidgetItem([self._panel_label(panel), ""])
                            pi.setData(0, Qt.UserRole, ("bpanel", i, j, p))
                            ci.addChild(pi)
                            if self._sel == ("bpanel", i, j, p):
                                pick = pi
                else:
                    for p, panel in enumerate(seg["panels"]):
                        pi = QTreeWidgetItem([self._panel_label(panel), ""])
                        pi.setData(0, Qt.UserRole, ("panel", i, p))
                        top.addChild(pi)
                        if self._sel == ("panel", i, p):
                            pick = pi
            self._tree.expandAll()
            if pick is not None:
                self._tree.setCurrentItem(pick)
        finally:
            self._building = False
        spans = list(getattr(self._tl, "segment_spans", []) or [])
        self._strip.set_timeline(spans, [segment_label(s) for s in self._spec["segments"]],
                                 getattr(self._tl, "n_frames", 0) or 0)
        self._strip.set_selected(self._sel[1])

    def _update_counts(self) -> None:
        """Frame counts into the outline and the strip, in place (no clear, no re-add)."""
        spans = list(getattr(self._tl, "segment_spans", []) or [])
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setText(1, str(spans[i][1]) if i < len(spans) else "")
        if self._spec is not None:
            self._strip.set_timeline(spans, [segment_label(s) for s in self._spec["segments"]],
                                     getattr(self._tl, "n_frames", 0) or 0)
            self._strip.set_selected(self._sel[1])

    def _panel_label(self, panel: Dict[str, Any]) -> str:
        bits = [f"panel {panel['source']}"]
        if panel["render"] != "image":
            bits.append(f"labels '{panel['layer']}'")
        if panel["tile"]["axis"] != "none":
            bits.append(f"tiles {panel['tile']['axis']}")
        if panel["channels"]:
            bits.append("ch " + ",".join(str(c) for c in panel["channels"]))
        return " · ".join(bits)

    def _on_tree_selection(self) -> None:
        if self._building:
            return
        items = self._tree.selectedItems()
        if items:
            sel = items[0].data(0, Qt.UserRole)
            if sel is not None and tuple(sel) != self._sel:
                self.select(tuple(sel), from_tree=True)

    # ── properties form ───────────────────────────────────────────────────────
    def _rebuild_props(self) -> None:
        host = QWidget()
        host.setObjectName("movieProps")
        form = QFormLayout(host)
        form.setContentsMargins(6, 6, 6, 6)
        form.setLabelAlignment(Qt.AlignRight)
        if self._spec is not None:
            item = _target(self._spec, self._sel)
            if self._sel[0] in ("panel", "bpanel"):
                self._panel_form(form, self._sel, item)
            elif item["kind"] == "loop":
                self._loop_form(form, self._sel, item)
            else:
                self._clip_form(form, self._sel, item)
        self._props.setWidget(host)

    # small widget factories — each commits through `edit` ─────────────────────
    def _w_combo(self, items: Sequence[Any], current: Any, sel: Sel, key: str, *,
                 labels: Optional[Sequence[str]] = None, structure: bool = False) -> QComboBox:
        box = QComboBox()
        for i, it in enumerate(items):
            box.addItem(labels[i] if labels else str(it), it)
        idx = max(0, list(items).index(current)) if current in items else 0
        box.setCurrentIndex(idx)
        box.currentIndexChanged.connect(
            lambda _i, b=box: self.edit(sel, key, b.currentData(), structure=structure))
        return box

    def _w_spin(self, lo: int, hi: int, value: int, sel: Sel, key: str) -> QSpinBox:
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(int(value))
        sp.setKeyboardTracking(False)     # commit on Enter / arrow, never per keystroke
        sp.valueChanged.connect(lambda v: self.edit(sel, key, int(v)))
        return sp

    def _w_dspin(self, lo: float, hi: float, value: float, sel: Sel, key: str,
                 step: float = 0.1, decimals: int = 2) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setValue(float(value))
        sp.setKeyboardTracking(False)
        sp.valueChanged.connect(lambda v: self.edit(sel, key, float(v)))
        return sp

    def _w_check(self, text: str, value: bool, sel: Sel, key: str) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(bool(value))
        cb.toggled.connect(lambda v: self.edit(sel, key, bool(v)))
        return cb

    def _w_line(self, value: str, sel: Sel, key: str, placeholder: str = "") -> QLineEdit:
        le = QLineEdit(str(value))
        le.setPlaceholderText(placeholder)
        le.editingFinished.connect(lambda e=le: self.edit(sel, key, e.text()))
        return le

    def _range_row(self, form: QFormLayout, sel: Sel, holder: Dict[str, Any], prefix: str,
                   n: int) -> None:
        row = QHBoxLayout()
        for key, lo in (("from", -max(1, n)), ("to", -max(1, n)), ("step", 1)):
            row.addWidget(QLabel(key))
            row.addWidget(self._w_spin(lo, max(1, n), holder[key], sel, f"{prefix}{key}"))
        wrap = QWidget(); wrap.setLayout(row)
        form.addRow("Range", wrap)

    def _loop_form(self, form: QFormLayout, sel: Sel, loop: Dict[str, Any]) -> None:
        form.addRow("Name", self._w_line(loop["name"], sel, "name", "(auto)"))
        form.addRow("Loop over t of", self._w_combo(list(LETTERS), loop["source"], sel,
                                                     "source"))
        env = (self._sources.get(loop["source"]) or {}).get("env")
        n_t = int(getattr(getattr(env, "axes", None), "t", 1) or 1)
        self._range_row(form, sel, loop, "", n_t)
        hint = QLabel("Each step plays every clip in the loop's body once. A panel whose t "
                      "is 'auto' shows the loop's current timepoint.")
        hint.setWordWrap(True)
        form.addRow(hint)

    def _clip_form(self, form: QFormLayout, sel: Sel, clip: Dict[str, Any]) -> None:
        play = clip["play"]
        form.addRow("Name", self._w_line(clip["name"], sel, "name", "(auto)"))
        form.addRow("Plays", self._w_combo(list(MT.PLAY_AXES), play["axis"], sel, "play.axis",
                                           labels=["t (timelapse)", "z (sweep)",
                                                   "nothing (still)"], structure=True))
        if play["axis"] != "none":
            form.addRow("Axis of", self._w_combo([""] + list(LETTERS), play["source"], sel,
                                                 "play.source",
                                                 labels=["(first panel's source)"]
                                                 + list(LETTERS)))
            src = play["source"] or clip["panels"][0]["source"]
            env = (self._sources.get(src) or {}).get("env")
            n = int(getattr(getattr(env, "axes", None), play["axis"], 1) or 1)
            self._range_row(form, sel, play, "play.", n)
            form.addRow("Direction", self._w_combo(list(MT.DIRECTIONS), play["direction"],
                                                   sel, "play.direction"))
        form.addRow("Hold each frame", self._w_spin(1, 1000, clip["hold"], sel, "hold"))
        form.addRow("Grid columns", self._w_spin(0, 64, clip["cols"], sel, "cols"))
        ann = clip["annotations"]
        box = QWidget()
        g = QVBoxLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        for key, text in (("frame", "Frame counter"), ("time", "Elapsed time"),
                          ("z", "Z readout"), ("scalebar", "Scale bar"),
                          ("channels", "Channel names"), ("position", "Position name"),
                          ("captions", "Tile captions")):
            g.addWidget(self._w_check(text, ann[key], sel, f"annotations.{key}"))
        form.addRow("Burn in", box)
        form.addRow("Title", self._w_line(ann["title"], sel, "annotations.title"))

    def _panel_form(self, form: QFormLayout, sel: Sel, panel: Dict[str, Any]) -> None:
        letter = panel["source"]
        names, n_c = self._source_channels(letter)
        env = (self._sources.get(letter) or {}).get("env")
        ax = getattr(env, "axes", None)
        form.addRow("Source", self._w_combo(list(LETTERS), letter, sel, "source",
                                            structure=True))
        form.addRow("Draw", self._w_combo(list(MT.RENDERS), panel["render"], sel, "render",
                                          labels=["the image", "labels only",
                                                  "labels over the image"],
                                          structure=True))
        layers = [""] + self._voxel_layers(letter)
        if panel["layer"] and panel["layer"] not in layers:
            layers.append(panel["layer"])
        form.addRow("Voxel layer", self._w_combo(
            layers, panel["layer"], sel, "layer",
            labels=["(none — the image)"] + layers[1:]))

        # channels: none ticked = all of them
        chans = set(panel["channels"])
        cbox = QWidget()
        cl = QHBoxLayout(cbox)
        cl.setContentsMargins(0, 0, 0, 0)
        for c in range(n_c):
            cb = QCheckBox(names[c])
            cb.setChecked(c in chans)
            cb.toggled.connect(lambda on, c=c: self._toggle_channel(sel, c, on))
            cl.addWidget(cb)
        form.addRow("Channels", cbox)
        hint = QLabel("none ticked = every channel; one channel renders in grey unless "
                      "you give it a colour")
        hint.setWordWrap(True)
        form.addRow(hint)

        form.addRow("t", self._w_index(panel["t"], ["auto"], sel, "t",
                                       int(getattr(ax, "t", 1) or 1)))
        form.addRow("z", self._w_index(panel["z"], ["auto", "max", "mean", "mid"], sel, "z",
                                       int(getattr(ax, "z", 1) or 1)))
        form.addRow("Position (m)", self._w_spin(0, max(0, int(getattr(ax, "m", 1) or 1) - 1),
                                                 panel["m"], sel, "m"))
        tile = panel["tile"]
        form.addRow("Tile by", self._w_combo(list(MT.TILE_AXES), tile["axis"], sel,
                                             "tile.axis",
                                             labels=["(no grid)", "position (m)",
                                                     "channel (c)", "z slice", "timepoint (t)"]))
        tl = QLineEdit(",".join(str(i) for i in tile["indices"]))
        tl.setPlaceholderText("all  (or e.g. 0,2,4)")
        tl.editingFinished.connect(lambda e=tl: self._set_indices(sel, e.text()))
        form.addRow("Tile indices", tl)
        form.addRow("Caption", self._w_line(panel["caption"], sel, "caption", "(auto)"))
        form.addRow("Brightness %", self._w_dspin(1.0, 1000.0, panel["brightness"], sel,
                                                  "brightness", step=10.0, decimals=0))
        con = panel["contrast"]
        form.addRow("Auto window", self._w_combo(list(MT.CONTRAST_MODES), con["mode"], sel,
                                                 "contrast.mode",
                                                 labels=["percentiles", "sensor range"]))
        form.addRow("Black / white %", self._pct_row(sel, con))

        # per-channel look
        shown = sorted(chans) or list(range(n_c))
        for c in shown:
            form.addRow(names[c], self._channel_row(sel, panel, c))

        if panel["render"] != "image":
            lab = panel["labels"]
            form.addRow(QLabel("Labels"))
            form.addRow("Label channel", self._w_combo(
                [-1] + list(range(n_c)), lab["c"], sel, "labels.c",
                labels=["(the panel's first)"] + names))
            form.addRow("Colours", self._w_combo(
                list(MT.LABEL_COLOR_MODES), lab["color"], sel, "labels.color",
                labels=["one colour per id (steady)", "neighbours kept apart", "single"]))
            form.addRow("Style", self._w_combo(list(MT.LABEL_STYLES), lab["style"], sel,
                                               "labels.style"))
            form.addRow("Fill opacity", self._w_dspin(0.0, 1.0, lab["fill_opacity"], sel,
                                                      "labels.fill_opacity", step=0.05))
            form.addRow("Outline px", self._w_spin(0, 20, lab["outline_px"], sel,
                                                   "labels.outline_px"))
            form.addRow("", self._w_check("Burn in ids", lab["show_ids"], sel,
                                          "labels.show_ids"))
            form.addRow("Id size px", self._w_spin(4, 200, lab["id_px"], sel, "labels.id_px"))

    def _w_index(self, value: Any, words: Sequence[str], sel: Sel, key: str,
                 n: int) -> QWidget:
        """A word ('auto', 'max', …) or an index: one combo of the words plus 'index', and a
        spin that is live only for 'index'."""
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        box = QComboBox()
        for w in words:
            box.addItem(w, w)
        box.addItem("index", "index")
        sp = QSpinBox()
        sp.setRange(0, max(0, n - 1))
        sp.setKeyboardTracking(False)
        if isinstance(value, int):
            box.setCurrentIndex(len(words))
            sp.setValue(value)
        else:
            box.setCurrentIndex(max(0, list(words).index(value)) if value in words else 0)
            sp.setEnabled(False)

        def commit() -> None:
            word = box.currentData()
            sp.setEnabled(word == "index")
            self.edit(sel, key, int(sp.value()) if word == "index" else word)

        box.currentIndexChanged.connect(lambda _i: commit())
        sp.valueChanged.connect(lambda _v: commit())
        row.addWidget(box)
        row.addWidget(sp)
        return wrap

    def _pct_row(self, sel: Sel, con: Dict[str, Any]) -> QWidget:
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._w_dspin(0.0, 100.0, con["low_pct"], sel, "contrast.low_pct"))
        row.addWidget(self._w_dspin(0.0, 100.0, con["high_pct"], sel, "contrast.high_pct"))
        return wrap

    def _channel_row(self, sel: Sel, panel: Dict[str, Any], c: int) -> QWidget:
        """Colour swatch, Viewer link, black, white, gamma for channel ``c``."""
        d = dict(panel["display"].get(str(c)) or {})
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        sw = QToolButton()
        rgb = d.get("rgb")
        sw.setText("  " if rgb else "auto")
        if rgb:
            sw.setStyleSheet(f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]});")
        sw.setToolTip("Colour of this channel. Click to pick; right-click to go back to the "
                      "automatic tint (emission colour, or grey for a lone channel).")
        sw.clicked.connect(lambda: self._pick_colour(sel, c, d.get("rgb")))
        sw.setContextMenuPolicy(Qt.CustomContextMenu)
        sw.customContextMenuRequested.connect(lambda _p: self._set_display(sel, c, rgb=None))
        row.addWidget(sw)
        link = QCheckBox("Viewer")
        link.setChecked(d.get("link") == "viewer")
        link.setToolTip("Follow the Viewer's LUT for this source's node: its black/white, "
                        "gamma and any colour you picked for the channel. The values are "
                        "written into the node when you stop adjusting the Viewer, so the "
                        "saved graph reproduces the movie headless.")
        link.toggled.connect(lambda on: self._set_display(
            sel, c, link="viewer" if on else "manual"))
        row.addWidget(link)
        for key, tip in (("lo", "black (empty = auto)"), ("hi", "white (empty = auto)")):
            le = QLineEdit("" if d.get(key) is None else f"{d[key]:g}")
            le.setPlaceholderText("auto")
            le.setToolTip(tip + ". Linked channels are overwritten from the Viewer.")
            le.setMaximumWidth(70)
            le.editingFinished.connect(
                lambda e=le, k=key: self._set_display(sel, c, **{k: _num_or_none(e.text())}))
            row.addWidget(le)
        g = QDoubleSpinBox()
        g.setRange(0.05, 20.0)
        g.setDecimals(2)
        g.setSingleStep(0.1)
        g.setValue(float(d.get("gamma", 1.0)))
        g.setKeyboardTracking(False)
        g.setToolTip("gamma")
        g.valueChanged.connect(lambda v: self._set_display(sel, c, gamma=float(v)))
        row.addWidget(g)
        return wrap

    def _set_display(self, sel: Sel, c: int, **kv: Any) -> None:
        def go(s: Dict[str, Any]) -> None:
            disp = _target(s, sel).setdefault("display", {})
            d = disp.setdefault(str(c), {})
            d.update(kv)
        self.apply(go, structure="rgb" in kv)

    def _pick_colour(self, sel: Sel, c: int, cur: Optional[Sequence[int]]) -> None:
        start = QColor(*cur) if cur else QColor(255, 255, 255)
        col = QColorDialog.getColor(start, self, "Channel colour")
        if col.isValid():
            self._set_display(sel, c, rgb=[col.red(), col.green(), col.blue()])

    def _toggle_channel(self, sel: Sel, c: int, on: bool) -> None:
        def go(s: Dict[str, Any]) -> None:
            p = _target(s, sel)
            cur = [x for x in p["channels"] if x != c]
            if on:
                cur.append(c)
            p["channels"] = sorted(cur)
        self.apply(go, structure=True)

    def _set_indices(self, sel: Sel, text: str) -> None:
        try:
            vals = [int(x) for x in text.replace(" ", "").split(",") if x != ""]
        except ValueError:
            self._status.setText(f"tile indices: {text!r} is not a list of whole numbers")
            return
        self.edit(sel, "tile.indices", vals)

    # ── preview ───────────────────────────────────────────────────────────────
    def _rebuild_preview(self) -> None:
        self._gen += 1
        self._cache.clear()
        self._tl = None
        if self._spec is None or self._nid is None:
            self._strip.set_timeline([], [], 0)
            return
        missing = []
        for L in _letters_used(self._spec):
            info = self._sources.get(L) or {}
            if info.get("node") is None:
                missing.append(f"source {L} is not wired — connect a node to "
                               f"'{MT.SOURCE_SOCKETS[L]}'")
            elif self._payloads.get(L) is None:
                missing.append(f"source {L} ({info.get('label', info['node'])}) is not "
                               f"computed yet — press Compute sources")
        st = self._host.movie_state(self._nid)
        spec = self._spec if self._flat else self._host.live_display(self._nid, self._spec)
        if missing:
            self._monitor.show_text("\n".join(missing))
            self._status.setText("")
            self._strip.set_timeline([], [], 0)
            self._readout.setText("")
            return
        from nodelab_v2.movie_preview import build_timeline
        try:
            self._tl, note = build_timeline(dict(self._payloads), spec, st["spec"],
                                            st["params"], st["env"])
        except Exception as exc:     # noqa: BLE001 — a bad timeline is shown, never raised
            self._tl = None
            self._monitor.show_text(str(exc))
            self._status.setText(str(exc))
            self._strip.set_timeline([], [], 0)
            return
        self._status.setText(note)
        n = self._tl.n_frames
        self._k = max(0, min(self._k, n - 1))
        self._update_counts()
        self._strip.set_cursor(self._k)
        self.seek(self._k)

    def seek(self, k: int) -> None:
        if self._tl is None:
            return
        self._k = max(0, min(int(k), self._tl.n_frames - 1))
        self._strip.set_cursor(self._k)
        self._readout.setText(self._describe(self._k))
        got = self._cache.get((self._gen, self._k))
        if got is not None:
            self._monitor.show_rgb(got)
            return
        self._renderer.request(self._gen, self._tl, self._k)

    def _describe(self, k: int) -> str:
        info = self._tl.frame_info(k)
        bits = [f"frame {k + 1}/{self._tl.n_frames}"]
        if info.get("loop_t") is not None:
            bits.append(f"loop t {info['loop_t'] + 1}")
        if info.get("t") is not None:
            bits.append(f"t {info['t'] + 1}")
        if info.get("z") is not None:
            bits.append(f"z {info['z'] + 1}")
        seg = self._spec["segments"][info["segment"]]
        bits.append(segment_label(seg))
        return "  ·  ".join(bits)

    def _on_rendered(self, gen: int, k: int, out: Any) -> None:
        if gen != self._gen:
            return
        if isinstance(out, Exception):
            self._monitor.show_text(f"cannot render frame {k + 1}:\n{out}")
            return
        self._cache[(gen, k)] = out
        while len(self._cache) > _CACHE_FRAMES:
            self._cache.popitem(last=False)
        if k == self._k:
            self._monitor.show_rgb(out)

    def render_now(self, k: int) -> Optional[np.ndarray]:
        """Frame ``k`` rendered synchronously on the calling thread (probes, export checks).
        Uses a fresh Timeline so it never shares one with the worker."""
        if self._tl is None:
            return None
        st = self._host.movie_state(self._nid)
        spec = self._spec if self._flat else self._host.live_display(self._nid, self._spec)
        from nodelab_v2.movie_preview import build_timeline
        tl, _n = build_timeline(dict(self._payloads), spec, st["spec"], st["params"],
                                st["env"])
        return tl.render(int(k))

    def frame_count(self) -> int:
        return int(getattr(self._tl, "n_frames", 0) or 0)

    # ── transport ─────────────────────────────────────────────────────────────
    def toggle_play(self) -> None:
        if self._timer.isActive():
            self.stop()
            return
        if self._tl is None:
            return
        st = self._host.movie_state(self._nid)
        fps = 10.0
        try:
            from nodelab_v2.movie_preview import resolve_params
            fps = float(resolve_params(st["spec"], st["params"], st["env"]).param("fps")
                        or 10.0)
        except Exception:     # noqa: BLE001
            pass
        self._timer.setInterval(max(20, int(round(1000.0 / max(0.1, fps)))))
        self._timer.start()
        self._play_btn.setText("Pause")

    def stop(self) -> None:
        self._timer.stop()
        self._play_btn.setText("Play")

    def _advance(self) -> None:
        if self._tl is None:
            self.stop()
            return
        self.seek((self._k + 1) % self._tl.n_frames)

    def shutdown(self) -> None:
        """Stop playback and the render thread (the window calls this as it closes)."""
        self.stop()
        self._renderer.shutdown()

    def closeEvent(self, e) -> None:
        self.stop()
        super().closeEvent(e)


class _Refused(Exception):
    """A structure edit the editor declines (deleting the only clip)."""


def _num_or_none(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None
