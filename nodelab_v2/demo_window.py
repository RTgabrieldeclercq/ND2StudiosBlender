"""The **What does this node do?** window — a node run live on a phantom, with sliders.

Opened from the ``?`` beside a node's title in the inspector and from the palette's
Overview. It is a sandbox: nothing here reads or writes the canvas node. The window shows
the phantom *before*, the node's result *after*, one control per editable parameter, and
the node's guide text; every control change recomputes the node and repaints.

Three things the layout is deliberate about:

* **The result is the real node.** A :class:`~nodelab_v2.demo_recipes.DemoSession` runs
  ``seed → prelude → node`` through the same engine and computes the Viewer uses, and the
  *after* overlays are painted by the Viewer's own :class:`~nodelab_v2.overlays.OverlayRenderer`
  through the same :class:`~nodelab_v2.viewer._ImageView` surface. If the demo and the
  Viewer ever disagreed, one of them would be lying.
* **Live, off the GUI thread, latest wins.** A slider drag is debounced, handed to one
  worker thread (the :mod:`nodelab_v2.movie_editor` renderer's shape), and a result that
  arrives for a superseded request is dropped. The *after* view dims while computing and is
  never blanked, so the eye keeps its reference. A node the recipe marks slow — or one that
  measures slow twice in a row — switches itself to a Run button and says so.
* **Guides for what does not transform pixels.** A writer, a page boundary, a zone or the
  Viewer shows its key features, its sockets with the gesture each one offers, its modes
  and how it works — the same text the palette Overview is built from.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
    QGraphicsOpacityEffect,
)

from nodegraph import roles as R
from nodegraph.registry import DIM_MODE, NODES
from nodegraph.sockets import SocketType
from nodelab_v2 import demo_recipes as DR
from nodelab_v2 import overlays as OV
from nodelab_v2 import theme as T
from nodelab_v2.node_item import mode_hover_text, socket_hover_text
from nodelab_v2.viewer import _ImageView, composite_with_clim

__all__ = ["NodeDemoWindow", "overview_css", "overview_html"]

#: debounce between a control change and the recompute it asks for
_DEBOUNCE_MS = 80
#: a live demo whose last two runs both took longer than this flips to the Run button
_SLOW_S = 1.5
#: rows a table view shows at most
_MAX_ROWS = 200
#: the "after" is auto-contrasted (not shown on the before's window) when its bright end
#: leaves this band around the before's — a normalized or distance-valued result
_SHARED_CLIM_BAND = (0.3, 3.0)


def _h(col: QColor) -> str:
    return col.name()


def overview_css() -> str:
    """The rich-text style the palette Overview and this window share."""
    return (f"<style>body{{color:{_h(T.INK)}; font-size:11px;}} "
            f"h3{{margin:0 0 2px 0; font-size:13px;}} "
            f".k{{color:{_h(T.MUTED)};}} .op{{color:{_h(T.MUTED)}; "
            f"font-family:monospace; font-size:10px;}} "
            f"table{{border-collapse:collapse;}} td{{padding:1px 6px 1px 0; "
            f"vertical-align:top;}} .sec{{color:{_h(T.ACCENT)}; font-weight:bold; "
            f"margin-top:6px;}} ul{{margin-left:14px; -qt-list-indent:1;}}</style>")


def overview_html(op_key: str, *, features: Tuple[str, ...] = (), head: bool = True) -> str:
    """:func:`nodelab_v2.demo_recipes.guide_html` with the theme's colours injected."""
    return DR.guide_html(
        op_key, css=overview_css(), features=features, head=head,
        socket_color=lambda t: _h(T.SOCKET.get(t, T.MUTED)),
        domain_color=lambda d: _h(T.domain_qcolor(d)))


# ── the worker ─────────────────────────────────────────────────────────────────

class _DemoWorker(QObject):
    """One worker thread, latest request wins. The session is only ever touched here.

    A plain ``threading.Thread`` rather than the shared Qt pool, for the reason the runner's
    pull thread gives: Python owns it, so it has one thread state for its life. The result
    comes back through a Qt signal, which crosses to the GUI thread queued."""

    done = Signal(int, object)                   # generation, DemoResult | Exception

    def __init__(self, session: DR.DemoSession) -> None:
        super().__init__()
        self._session = session
        self._cv = threading.Condition()
        self._job: Optional[Tuple[int, Dict[str, Any], Dict[str, str]]] = None
        self._stop = False
        #: a run is in progress or waiting — :meth:`shutdown` and a probe wait on this
        self.busy = False
        self._thread = threading.Thread(target=self._loop, name="node-demo", daemon=True)
        self._thread.start()

    def request(self, gen: int, params: Dict[str, Any], modes: Dict[str, str]) -> None:
        with self._cv:
            self._job = (gen, dict(params), dict(modes))
            self.busy = True
            self._cv.notify()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop after the run in hand. A daemon thread still inside scipy at interpreter
        teardown can take the process with it, so the window waits rather than races."""
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
                gen, params, modes = self._job
                self._job = None
            try:
                out: Any = self._session.run(params, modes)
            except Exception as exc:          # noqa: BLE001 — shown in the caption, never fatal
                exc._demo_tb = traceback.format_exc()  # type: ignore[attr-defined]
                out = exc
            try:
                self.done.emit(gen, out)
            except RuntimeError:              # the window was destroyed while this ran
                return
            with self._cv:
                if self._job is None:
                    self.busy = False


# ── controls ───────────────────────────────────────────────────────────────────

class _Control(QWidget):
    """One editable parameter or mode: a widget, a value, and whether it was touched."""

    changed = Signal()

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.touched = False

    def value(self) -> Any:                    # pragma: no cover - abstract
        raise NotImplementedError

    def set_value(self, v: Any) -> None:       # pragma: no cover - abstract
        raise NotImplementedError

    def _emit(self) -> None:
        self.touched = True
        self.changed.emit()


class _SliderControl(_Control):
    """A slider and a spin box that agree: the slider scrubs, the box types exactly."""

    def __init__(self, name: str, rng: DR.SliderRange, value: Any, *, integer: bool,
                 unit: str = "") -> None:
        super().__init__(name)
        self._rng = rng
        self._integer = integer
        self._loading = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._slider = QSlider(Qt.Horizontal)
        span = max(rng.hi - rng.lo, 1e-12)
        self._ticks = int(min(1000, max(1, round(span / max(rng.step, 1e-12)))))
        self._slider.setRange(0, self._ticks)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, self._ticks // 10))
        if integer:
            box: Any = QSpinBox()
            box.setRange(int(min(rng.lo, -2_000_000_000)), int(max(rng.hi, 2_000_000_000)))
            box.setSingleStep(max(1, int(rng.step)))
        else:
            box = QDoubleSpinBox()
            box.setDecimals(rng.decimals)
            box.setRange(-1e12, 1e12)
            box.setSingleStep(rng.step)
        box.setFixedWidth(84)
        box.setKeyboardTracking(False)
        self._box = box
        lay.addWidget(self._slider, 1)
        lay.addWidget(box)
        if unit:
            u = QLabel(unit)
            u.setProperty("role", "muted")
            lay.addWidget(u)
        self.set_value(value)
        self._slider.valueChanged.connect(self._from_slider)
        self._slider.sliderReleased.connect(self._emit)
        box.valueChanged.connect(self._from_box)

    def _tick_to_value(self, tick: int) -> float:
        v = self._rng.lo + (self._rng.hi - self._rng.lo) * tick / max(1, self._ticks)
        return float(round(v)) if self._integer else float(v)

    def _value_to_tick(self, v: float) -> int:
        span = max(self._rng.hi - self._rng.lo, 1e-12)
        return int(round((float(v) - self._rng.lo) / span * self._ticks))

    def _from_slider(self, tick: int) -> None:
        if self._loading:
            return
        v = self._tick_to_value(tick)
        self._loading = True
        self._box.setValue(int(v) if self._integer else v)
        self._loading = False
        if not self._slider.isSliderDown():
            self._emit()

    def _from_box(self, v: Any) -> None:
        if self._loading:
            return
        self._loading = True
        self._slider.setValue(max(0, min(self._ticks, self._value_to_tick(float(v)))))
        self._loading = False
        self._emit()

    def value(self) -> Any:
        v = self._box.value()
        return int(v) if self._integer else float(v)

    def set_value(self, v: Any) -> None:
        self._loading = True
        try:
            fv = float(v) if v is not None else self._rng.lo
        except (TypeError, ValueError):
            fv = self._rng.lo
        self._box.setValue(int(round(fv)) if self._integer else fv)
        self._slider.setValue(max(0, min(self._ticks, self._value_to_tick(fv))))
        self._loading = False


class _ChoiceControl(_Control):
    def __init__(self, name: str, choices: List[str], value: Any,
                 docs: Optional[Dict[str, str]] = None) -> None:
        super().__init__(name)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        self._choices = list(choices)
        self._combo.addItems(self._choices)
        for i, c in enumerate(self._choices):
            tip = (docs or {}).get(c, "")
            if tip:
                self._combo.setItemData(i, tip, Qt.ToolTipRole)
        lay.addWidget(self._combo, 1)
        self.set_value(value)
        self._combo.currentIndexChanged.connect(lambda _i: self._emit())

    def value(self) -> Any:
        return self._combo.currentText()

    def set_value(self, v: Any) -> None:
        if v in self._choices:
            self._combo.blockSignals(True)
            self._combo.setCurrentIndex(self._choices.index(v))
            self._combo.blockSignals(False)


class _BoolControl(_Control):
    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._chk = QCheckBox()
        lay.addWidget(self._chk)
        lay.addStretch(1)
        self.set_value(value)
        self._chk.toggled.connect(lambda _v: self._emit())

    def value(self) -> Any:
        return bool(self._chk.isChecked())

    def set_value(self, v: Any) -> None:
        self._chk.blockSignals(True)
        self._chk.setChecked(bool(v))
        self._chk.blockSignals(False)


class _ReadonlyControl(_Control):
    """A value the recipe fixed (a layer name, a column, a shapes list) shown as a chip."""

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        text = "" if value is None else str(value)
        if len(text) > 48:
            text = text[:45] + "…"
        self._lab = QLabel(text or "—")
        self._lab.setProperty("role", "muted")
        self._lab.setStyleSheet(f"QLabel {{ font-family:{T.MONO}; font-size:10px; }}")
        self._lab.setToolTip("Fixed for this demo.")
        self._value = value
        lay.addWidget(self._lab, 1)

    def value(self) -> Any:
        return self._value

    def set_value(self, v: Any) -> None:
        self._value = v


# ── the window ─────────────────────────────────────────────────────────────────

class NodeDemoWindow(QDialog):
    """A node's live demo. Non-modal; one per op, cached by the main window."""

    closed = Signal(str)

    def __init__(self, op_key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.op_key = op_key
        self.spec = NODES.get(op_key)
        self.recipe = DR.recipe_for(op_key)
        self.setWindowFlag(Qt.Window)
        self.setWindowTitle(f"What does {self.spec.label if self.spec else op_key} do?")
        self.setMinimumSize(720, 480)
        self.resize(1080, 680)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setStyleSheet(
            f"QDialog {{ background:{_h(T.PANEL)}; color:{_h(T.INK)}; }}"
            f"QLabel {{ color:{_h(T.INK)}; }}"
            f"QLabel[role='muted'] {{ color:{_h(T.MUTED)}; }}"
            f"QLabel[role='eyebrow'] {{ color:{_h(T.MUTED)}; letter-spacing:1px; }}"
            f"QLabel[role='error'] {{ color:#e06a5a; }}"
            f"QScrollArea {{ border:none; background:transparent; }}"
            f"QTextBrowser {{ background:{_h(T.BG)}; color:{_h(T.INK)}; border:1px solid "
            f"{_h(T.BORDER)}; border-radius:5px; }}"
            f"QTableWidget {{ background:{_h(T.BG)}; color:{_h(T.INK)}; gridline-color:"
            f"{_h(T.BORDER)}; border:1px solid {_h(T.BORDER)}; }}")

        self._session: Optional[DR.DemoSession] = None
        self._worker: Optional[_DemoWorker] = None
        self._controls: Dict[str, _Control] = {}
        self._mode_controls: Dict[str, _Control] = {}
        self._values: Dict[str, Any] = {}
        self._modes: Dict[str, str] = {}
        self._gen = 0
        self._done_gen = 0
        self.last_result: Optional[DR.DemoResult] = None
        self._elapsed: List[float] = []
        self._auto_slow = False
        self._renderer = OV.OverlayRenderer()
        self._settings = OV.OverlaySettings()
        self._frame: Optional[OV.OverlayFrame] = None
        self._before_frame: Optional[OV.OverlayFrame] = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._fire)
        self._missing: List[str] = []
        #: which of the recipe's scenarios (synthetic worlds) is running; 0 when it has none
        self._scenario = 0
        self._scenario_ctl: Optional[_Control] = None
        self._phantom_label: Optional[QLabel] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        root.addWidget(self._header())

        if self.spec is None or self.recipe.is_guide:
            root.addWidget(self._guide_pane(), 1)
            return

        try:
            self._session = DR.DemoSession(self._active_recipe())
        except Exception as exc:                  # noqa: BLE001 — a bad recipe shows itself
            root.addWidget(self._guide_pane(error=str(exc)), 1)
            return
        self._missing = self._session.missing_requirements()
        self._values = self._session.starting_values()
        self._modes = self._session.default_state()

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._controls_pane())
        split.addWidget(self._result_pane())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([330, 750])
        root.addWidget(split, 1)
        root.addWidget(self._footer())

        if self._missing:
            self._set_status(f"needs the optional package(s) {', '.join(self._missing)} — "
                             f"pip install {' '.join(self._missing)}", error=True)
            self._run_btn.setEnabled(False)
            return
        self._worker = _DemoWorker(self._session)
        self._worker.done.connect(self._on_done)
        if self._is_slow():
            self._set_status(f"press Run — {self.recipe.slow_reason or 'this node is slow'}")
        else:
            self._fire()

    # ── construction ─────────────────────────────────────────────────────────
    def _header(self) -> QWidget:
        hd = QWidget()
        lay = QVBoxLayout(hd)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        rk, sk = R.role_of(self.op_key)
        rmeta, smeta = R.role_meta(rk), R.stage_meta(sk)
        eyebrow = QLabel(f"{smeta.get('label', sk)} › {rmeta.get('label', rk)}".upper())
        eyebrow.setProperty("role", "eyebrow")
        f = eyebrow.font(); f.setPointSize(8); eyebrow.setFont(f)
        lay.addWidget(eyebrow)
        row = QHBoxLayout()
        title = QLabel(self.spec.label if self.spec else self.op_key)
        tf = title.font(); tf.setPointSize(14); tf.setBold(True); title.setFont(tf)
        row.addWidget(title)
        op = QLabel(self.op_key)
        op.setProperty("role", "muted")
        op.setStyleSheet(f"font-family:{T.MONO}; font-size:10px;")
        row.addWidget(op)
        row.addStretch(1)
        lay.addLayout(row)
        text = DR._squash(self.spec.description) if self.spec else ""
        if len(text) > 320:                      # the full text is in How it works below
            text = text[:320].rsplit(" ", 1)[0] + " …"
        desc = QLabel(text)
        desc.setWordWrap(True)
        lay.addWidget(desc)
        if not self.recipe.is_guide and (self.recipe.phantom or self.recipe.scenarios):
            ph = QLabel(self._phantom_caption())
            ph.setProperty("role", "muted")
            ph.setWordWrap(True)
            lay.addWidget(ph)
            self._phantom_label = ph
        return hd

    def _active_recipe(self) -> DR.DemoRecipe:
        """The recipe with the chosen scenario applied (the recipe itself when it has none)."""
        return DR.scenario_recipe(self.recipe, self._scenario)

    def _phantom_caption(self) -> str:
        """The header's line on the synthetic data: the phantom's own caption, which states
        the TRUE motion / layout so the node's result can be read against it."""
        rec = self._active_recipe()
        try:
            cap = DR.phantom(rec.phantom, **dict(rec.phantom_kw)).caption
        except Exception:                         # noqa: BLE001
            cap = rec.phantom
        pre = " → ".join(NODES.get(s.op).label if NODES.get(s.op) else s.op
                         for s in rec.prelude)
        return f"Phantom: {cap}" + (f"  ·  upstream: {pre}" if pre else "")

    def _controls_pane(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self._ctl_host = QWidget()
        self._ctl_host.setObjectName("demoControls")
        # the scroll area's child takes Qt's light default fill unless told otherwise
        self._ctl_host.setStyleSheet(
            f"QWidget#demoControls {{ background:{_h(T.PANEL)}; }}")
        self._ctl_lay = QVBoxLayout(self._ctl_host)
        self._ctl_lay.setContentsMargins(0, 0, 6, 0)
        self._ctl_lay.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._ctl_host)
        scroll.viewport().setStyleSheet(f"background:{_h(T.PANEL)};")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        v.addWidget(scroll, 1)
        self._build_controls()
        return wrap

    def _eyebrow(self, text: str) -> QLabel:
        lab = QLabel(text.upper())
        lab.setProperty("role", "eyebrow")
        f = lab.font(); f.setPointSize(8); lab.setFont(f)
        return lab

    def _build_controls(self) -> None:
        """(Re)build one control per active parameter and mode for the current mode state —
        the same :meth:`NodeSpec.active_inputs` the inspector uses, so a mode that hides a
        parameter there hides it here."""
        while self._ctl_lay.count():
            item = self._ctl_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._controls.clear()
        self._mode_controls.clear()
        spec, sess = self.spec, self._session
        rec = sess.recipe                      # the active scenario's fixed values and view
        state = {**self._modes}
        # the synthetic world, when the recipe offers more than one (2026-10-07): a dropdown
        # above the modes, with the scenario's note on what to look for
        self._scenario_ctl = None
        if self.recipe.scenarios:
            labels = [sc.label for sc in self.recipe.scenarios]
            cur = self.recipe.scenarios[min(self._scenario, len(labels) - 1)]
            self._ctl_lay.addWidget(self._eyebrow("Synthetic data"))
            self._scenario_ctl = _ChoiceControl("__scenario", labels, cur.label)
            self._scenario_ctl.setToolTip(
                "Which synthetic world the node runs on. Each world's true motion is stated "
                "in the caption under the title, so the result can be read against it.")
            self._scenario_ctl.changed.connect(self._on_scenario_changed)
            self._ctl_lay.addWidget(self._scenario_ctl)
            if cur.note:
                note = QLabel(cur.note)
                note.setProperty("role", "muted")
                note.setWordWrap(True)
                self._ctl_lay.addWidget(note)
        grid_modes = QGridLayout()
        grid_modes.setContentsMargins(0, 0, 0, 0)
        grid_modes.setHorizontalSpacing(8)
        grid_modes.setVerticalSpacing(4)
        n_modes = 0
        for m in spec.active_modes(state):
            if m.name in rec.fixed_modes:
                ctl: _Control = _ReadonlyControl(m.name, rec.fixed_modes[m.name])
            elif bool(getattr(m, "is_dim_lever", False)) and not (
                    sess.phantom.axes.z > 1 and spec.supports_true_3d):
                ctl = _ReadonlyControl(m.name, state.get(m.name, m.resolved_default()))
            else:
                ctl = _ChoiceControl(m.name, list(m.choices), state.get(m.name),
                                     getattr(m, "choice_docs", None))
            ctl.setToolTip(mode_hover_text(m))
            ctl.changed.connect(lambda nm=m.name: self._on_mode_changed(nm))
            lab = QLabel(m.label or m.name)
            lab.setToolTip(ctl.toolTip())
            grid_modes.addWidget(lab, n_modes, 0)
            grid_modes.addWidget(ctl, n_modes, 1)
            self._mode_controls[m.name] = ctl
            n_modes += 1
        if n_modes:
            self._ctl_lay.addWidget(self._eyebrow("Modes"))
            host = QWidget(); host.setLayout(grid_modes)
            self._ctl_lay.addWidget(host)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        n = 0
        for s in spec.active_inputs(state):
            kind = DR.control_kind(s)
            if kind == "hidden":
                continue
            value = self._values.get(s.name, s.default)
            if s.name in rec.fixed_params or kind == "readonly":
                ctl = _ReadonlyControl(s.name, rec.fixed_params.get(s.name, value))
            elif kind == "slider":
                integer = s.type is SocketType.INT
                rng = sess.range_for(s, value)
                ctl = _SliderControl(s.name, rng, value, integer=integer, unit=s.unit or "")
            elif kind == "choice":
                ctl = _ChoiceControl(s.name, list(s.choices), value,
                                     getattr(s, "choice_docs", None))
            else:
                ctl = _BoolControl(s.name, value)
            ctl.setToolTip(socket_hover_text(s))
            ctl.changed.connect(lambda nm=s.name: self._on_param_changed(nm))
            lab = QLabel(s.label or s.name)
            lab.setToolTip(ctl.toolTip())
            if getattr(s, "derive", "") and s.name not in rec.fixed_params:
                lab.setText(f"{s.label or s.name}  <span style='color:{_h(T.MUTED)}'>auto</span>")
            grid.addWidget(lab, n, 0)
            grid.addWidget(ctl, n, 1)
            self._controls[s.name] = ctl
            n += 1
        self._ctl_lay.addWidget(self._eyebrow("Parameters" if n else "No parameters"))
        if n:
            host = QWidget(); host.setLayout(grid)
            self._ctl_lay.addWidget(host)
        # the viewed frame / plane, when the phantom has more than one
        ax = sess.phantom.axes
        if ax.t > 1 or ax.z > 1:
            self._ctl_lay.addWidget(self._eyebrow("Viewed plane"))
            vg = QGridLayout(); vg.setContentsMargins(0, 0, 0, 0)
            r = 0
            t0, z0, _c = sess.view_coords()
            if ax.t > 1:
                self._t_ctl = _SliderControl("__t", DR.SliderRange(0, ax.t - 1, 1, 0), t0,
                                             integer=True)
                self._t_ctl.changed.connect(lambda: self._on_view_changed())
                vg.addWidget(QLabel("Frame t"), r, 0); vg.addWidget(self._t_ctl, r, 1); r += 1
            if ax.z > 1:
                self._z_ctl = _SliderControl("__z", DR.SliderRange(0, ax.z - 1, 1, 0), z0,
                                             integer=True)
                self._z_ctl.changed.connect(lambda: self._on_view_changed())
                vg.addWidget(QLabel("Plane z"), r, 0); vg.addWidget(self._z_ctl, r, 1); r += 1
            host = QWidget(); host.setLayout(vg)
            self._ctl_lay.addWidget(host)
        if rec.kind == "image":
            self._ctl_lay.addWidget(self._eyebrow("Compare"))
            cg = QGridLayout(); cg.setContentsMargins(0, 0, 0, 0)
            self._compare = _ChoiceControl("__compare", ["Side by side", "Wipe", "Checkerboard",
                                                         "Difference"], "Side by side")
            self._compare.changed.connect(self._render)
            self._wipe = _SliderControl("__wipe", DR.SliderRange(0.0, 1.0, 0.01, 2), 0.5,
                                        integer=False)
            self._wipe.changed.connect(self._render)
            cg.addWidget(QLabel("Mode"), 0, 0); cg.addWidget(self._compare, 0, 1)
            cg.addWidget(QLabel("Divider"), 1, 0); cg.addWidget(self._wipe, 1, 1)
            host = QWidget(); host.setLayout(cg)
            self._ctl_lay.addWidget(host)
        self._ctl_lay.addStretch(1)

    def _result_pane(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        caps = QHBoxLayout()
        self._cap_before = QLabel("Before")
        self._cap_before.setProperty("role", "eyebrow")
        self._cap_after = QLabel("After")
        self._cap_after.setProperty("role", "eyebrow")
        caps.addWidget(self._cap_before, 1)
        caps.addWidget(self._cap_after, 1)
        v.addLayout(caps)
        views = QHBoxLayout()
        views.setSpacing(6)
        self._before = _ImageView()
        self._before.overlay_cb = self._paint_before_overlay
        self._after = _ImageView()
        self._after.overlay_cb = self._paint_after_overlay
        for view in (self._before, self._after):
            view.setBackgroundBrush(T.BG)        # the letterbox, as the Viewer paints it
            view.setStyleSheet(f"border:1px solid {_h(T.BORDER)};")
        self._dim = QGraphicsOpacityEffect(self._after)
        self._dim.setOpacity(1.0)
        self._after.setGraphicsEffect(self._dim)
        self._table = QTableWidget()
        self._table.setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        views.addWidget(self._before, 1)
        views.addWidget(self._after, 1)
        views.addWidget(self._table, 1)
        v.addLayout(views, 1)
        return wrap

    def _footer(self) -> QWidget:
        ft = QWidget()
        v = QVBoxLayout(ft)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        row = QHBoxLayout()
        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        self._status.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedWidth(90)
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        self._run_btn = QPushButton("Run")
        self._run_btn.setCursor(Qt.PointingHandCursor)
        self._run_btn.setToolTip("Compute the node with the values above.")
        self._run_btn.clicked.connect(self.run_now)
        self._run_btn.setVisible(self._is_slow())
        self._guide_btn = QPushButton("How it works ▾")
        self._guide_btn.setCheckable(True)
        self._guide_btn.setCursor(Qt.PointingHandCursor)
        self._guide_btn.toggled.connect(self._toggle_guide)
        row.addWidget(self._status, 1)
        row.addWidget(self._progress)
        row.addWidget(self._run_btn)
        row.addWidget(self._guide_btn)
        v.addLayout(row)
        self._guide = QTextBrowser()
        self._guide.setOpenExternalLinks(False)
        self._guide.setOpenLinks(False)
        self._guide.setHtml(overview_html(self.op_key, features=self.recipe.features,
                                          head=False))
        self._guide.setMaximumHeight(220)
        self._guide.setVisible(bool(self.recipe.features))
        self._guide_btn.setChecked(bool(self.recipe.features))
        v.addWidget(self._guide)
        return ft

    def _guide_pane(self, error: str = "") -> QWidget:
        tb = QTextBrowser()
        tb.setOpenExternalLinks(False)
        tb.setOpenLinks(False)
        html = overview_html(self.op_key, features=self.recipe.features, head=False)
        if error:
            html = (f"<div style='color:#e06a5a'>This demo could not be set up: "
                    f"{DR._html_escape(error)}</div>" + html)
        tb.setHtml(html)
        return tb

    # ── state ────────────────────────────────────────────────────────────────
    def control(self, name: str) -> Optional[_Control]:
        """A parameter's control (or a mode's), for callers and the GUI probe."""
        return self._controls.get(name) or self._mode_controls.get(name)

    def _is_slow(self) -> bool:
        return self._auto_slow or (self._session is not None and self._session.is_slow(self._modes))

    def _collect(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """What to send: every touched parameter (an untouched one stays absent so the engine
        derives it), and the full mode state."""
        params: Dict[str, Any] = {}
        for name, ctl in self._controls.items():
            if isinstance(ctl, _ReadonlyControl):
                continue
            if ctl.touched:
                params[name] = ctl.value()
        return params, dict(self._modes)

    def _on_param_changed(self, name: str) -> None:
        ctl = self._controls.get(name)
        if ctl is not None:
            self._values[name] = ctl.value()
        self._schedule()

    def _on_mode_changed(self, name: str) -> None:
        ctl = self._mode_controls.get(name)
        if ctl is not None:
            self._modes[name] = str(ctl.value())
        self._build_controls()                 # the active socket set may have changed
        self._run_btn.setVisible(self._is_slow())
        self._schedule()

    def _on_scenario_changed(self) -> None:
        """Switch the synthetic world. The worker is bound to a session and a session to a
        phantom, so both are rebuilt; the user's settings carry over where the new world does
        not fix them, except the 2D/3D lever, which re-derives from the new phantom's depth."""
        if self._scenario_ctl is None or self._session is None:
            return
        labels = [sc.label for sc in self.recipe.scenarios]
        label = str(self._scenario_ctl.value())
        idx = labels.index(label) if label in labels else 0
        if idx == self._scenario:
            return
        self._scenario = idx
        self._debounce.stop()
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None
        self._session = DR.DemoSession(self._active_recipe())
        rec = self._session.recipe
        start = self._session.starting_values()
        kept = {k: v for k, v in self._values.items()
                if k in start and k not in rec.fixed_params and v != start[k]}
        self._values = {**start, **kept}
        fresh = self._session.default_state()
        self._modes = {**fresh, **{k: v for k, v in self._modes.items()
                                   if k in fresh and k not in rec.fixed_modes
                                   and k != DIM_MODE}}
        self._elapsed.clear()
        self._auto_slow = False
        self.last_result = None
        self._frame = None
        self._before_frame = None
        if self._phantom_label is not None:
            self._phantom_label.setText(self._phantom_caption())
        self._build_controls()
        for name, ctl in self._controls.items():
            if name in kept and not isinstance(ctl, _ReadonlyControl):
                ctl.touched = True              # a carried-over value is still a chosen one
        self._missing = self._session.missing_requirements()
        if self._missing:
            self._set_status(f"needs the optional package(s) {', '.join(self._missing)} — "
                             f"pip install {' '.join(self._missing)}", error=True)
            self._run_btn.setEnabled(False)
            return
        self._run_btn.setEnabled(True)
        self._worker = _DemoWorker(self._session)
        self._worker.done.connect(self._on_done)
        self._run_btn.setVisible(self._is_slow())
        if self._is_slow():
            self._set_status(f"press Run — {rec.slow_reason or 'this node is slow'}")
        else:
            self._fire()

    def _on_view_changed(self) -> None:
        if self._session is None:
            return
        view = dict(self._session.recipe.view)
        if hasattr(self, "_t_ctl"):
            view["t"] = int(self._t_ctl.value())
        if hasattr(self, "_z_ctl"):
            view["z"] = int(self._z_ctl.value())
        self._session.recipe = replace(self._session.recipe, view=view)
        self._schedule(force=True)

    def _schedule(self, *, force: bool = False) -> None:
        if self._worker is None:
            return
        if self._is_slow() and not force:
            params, _m = self._collect()
            self._set_status("values changed — press Run to compute")
            return
        self._debounce.start()

    def run_now(self) -> None:
        """Compute now with the current values (the Run button; also the slow path)."""
        self._debounce.stop()
        self._fire()

    def _fire(self) -> None:
        if self._worker is None:
            return
        params, modes = self._collect()
        self._gen += 1
        self._progress.setVisible(True)
        self._dim.setOpacity(0.55)
        self._set_status(self._describe(params) + " — computing…")
        self._worker.request(self._gen, params, modes)

    def _describe(self, params: Dict[str, Any]) -> str:
        if not params:
            return "defaults"
        bits = []
        for k, v in params.items():
            s = self.spec.input(k)
            unit = f" {s.unit}" if s is not None and s.unit else ""
            vs = f"{v:.4g}" if isinstance(v, float) else str(v)
            bits.append(f"{k} = {vs}{unit}")
        return ", ".join(bits)

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setProperty("role", "error" if error else "muted")
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def _on_done(self, gen: int, out: Any) -> None:
        if gen < self._gen:
            return                                # superseded while it ran
        self._done_gen = gen
        self._progress.setVisible(False)
        self._dim.setOpacity(1.0)
        if isinstance(out, Exception):
            msg = str(out).splitlines()[0] if str(out) else type(out).__name__
            self._set_status(msg[:400], error=True)
            tb = getattr(out, "_demo_tb", "")
            if tb:
                print(f"[node demo] {self.op_key}:\n{tb}")
            return
        res: DR.DemoResult = out
        self.last_result = res
        self._elapsed.append(res.elapsed_s)
        params, _m = self._collect()
        note = f"  ·  {res.note}" if res.note else ""
        fv = getattr(res, "frame_values", None) or {}
        if fv:                                     # the transform applied to the viewed frame
            note += (f"  ·  frame {res.view[0]}: "
                     + ", ".join(f"{k} {v:+.2f}" for k, v in fv.items()))
        self._set_status(f"{self._describe(params)}  ·  {res.elapsed_s * 1000:.0f} ms{note}")
        if (not self._auto_slow and self._session.recipe.live and len(self._elapsed) >= 2
                and all(e > _SLOW_S for e in self._elapsed[-2:])):
            self._auto_slow = True
            self._run_btn.setVisible(True)
            self._set_status(f"this node takes {res.elapsed_s:.1f} s on the phantom — "
                             f"switched to the Run button")
        self._render()

    # ── painting ─────────────────────────────────────────────────────────────
    @staticmethod
    def _clim(plane: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(plane, dtype=float)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo, hi = float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5))
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def _after_clim(self, before: np.ndarray, after: np.ndarray) -> Tuple[Tuple[float, float], bool]:
        """The before's window when the after lives in the same range, else its own."""
        b = self._clim(before)
        a = self._clim(after)
        bh = abs(b[1]) if b[1] else 1.0
        if _SHARED_CLIM_BAND[0] <= abs(a[1]) / bh <= _SHARED_CLIM_BAND[1] and a[0] >= b[0] - bh:
            return b, True
        return a, False

    def _pix(self, plane: np.ndarray, clim: Tuple[float, float]) -> QPixmap:
        img = composite_with_clim({0: np.asarray(plane, dtype=float)}, {0: (255, 255, 255)},
                                  {0: clim})
        return QPixmap.fromImage(img)

    def _plot_pixmap(self, res: DR.DemoResult) -> Optional[QPixmap]:
        ds = res.after_dataset
        if ds is None or ds.image is None:
            return None
        ax = ds.axes
        planes: Dict[int, np.ndarray] = {}
        colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
        t, z, _c = res.view
        for c in range(min(3, ax.c)):
            planes[c] = np.asarray(ds.image.read_region(0, 0, min(t, ax.t - 1), min(z, ax.z - 1),
                                                        c, 0, ax.y, 0, ax.x), dtype=float)
        if not planes:
            return None
        if len(planes) < 3:
            colors = {c: (255, 255, 255) for c in planes}
        clims = {c: (float(p.min()), float(max(p.max(), p.min() + 1))) for c, p in planes.items()}
        return QPixmap.fromImage(composite_with_clim(planes, colors, clims))

    def _render(self) -> None:
        res = self.last_result
        if res is None:
            return
        kind = self.recipe.kind
        before = res.before
        bclim = self._clim(before)
        self._table.setVisible(kind == "table")
        self._after.setVisible(kind != "table")
        self._before.setVisible(True)
        # the before: for a table demo it carries the prelude's labels so the measured
        # objects are visible; for everything else it is the bare phantom
        self._before_frame = None
        if kind == "table" and res.label_plane is not None:
            self._before_frame = self._frame_for(self._before, label_plane=res.label_plane)
        if kind == "image" and getattr(self, "_compare", None) is not None \
                and self._compare.value() != "Side by side" and res.after_image is not None \
                and res.after_image.shape == before.shape:
            mode = {"Wipe": 4, "Checkerboard": 3, "Difference": 2}[self._compare.value()]
            aclim, _shared = self._after_clim(before, res.after_image)
            param = float(self._wipe.value()) if mode == 4 else 8.0
            img = composite_with_clim({0: before, 1: res.after_image},
                                      {0: (255, 255, 255), 1: (255, 255, 255)},
                                      {0: bclim, 1: aclim},
                                      blends={1: (mode, 1.0, param)})
            self._before.setVisible(False)
            self._cap_before.setText("")
            self._cap_after.setText(f"{self._compare.value()}: before vs after")
            self._after.set_pixmap(QPixmap.fromImage(img))
            self._frame = None
            self._after.refresh()
            return
        self._cap_before.setText("Before")
        self._before.set_pixmap(self._pix(before, bclim))
        self._before.refresh()
        if kind == "table":
            self._cap_after.setText("After — the table")
            self._fill_table(res)
            return
        if kind == "plot":
            pm = self._plot_pixmap(res)
            self._cap_after.setText("After — the figure")
            if pm is not None:
                self._after.set_pixmap(pm)
            self._frame = None
            self._after.refresh()
            return
        after = res.after_image if res.after_image is not None else before
        aclim, shared = self._after_clim(before, after)
        self._cap_after.setText("After" + ("" if shared else "  (own contrast)"))
        self._after.set_pixmap(self._pix(after, aclim))
        self._frame = self._frame_for(self._after, result=res)
        self._after.refresh()

    def _frame_for(self, view: _ImageView, *, result: Optional[DR.DemoResult] = None,
                   label_plane: Optional[np.ndarray] = None) -> OV.OverlayFrame:
        pm = view._item.pixmap()
        w, h = (pm.width(), pm.height()) if not pm.isNull() else (1, 1)
        frame = OV.OverlayFrame(map_pt=view.plane_to_widget, plane_wh=(w, h))
        if result is None:
            frame.label_plane = label_plane
            return frame
        kind = self.recipe.kind
        t, z, _c = result.view
        frame.current_t = t
        if kind == "labels":
            frame.label_plane = result.label_plane
        elif kind in ("mask", "scalar"):
            frame.scalar = result.mask_plane if kind == "mask" else result.scalar_plane
        elif kind == "points":
            frame.points = [OV.PointMark(y, x, key, li, on, zp)
                            for (y, x, key, li, on, zp) in result.points]
        elif kind == "tracks":
            frame.tracks = [OV.TrackPath(tid, path, cur, times)
                            for (tid, path, cur, times) in result.tracks]
            if result.label_plane is not None:
                frame.label_plane = result.label_plane
        elif kind == "field":
            frame.vectors = result.vectors
        elif kind == "mesh":
            secs = []
            for oid, verts in result.mesh:
                sec = OV.MeshSection(oid)
                sec.verts = list(verts)
                secs.append(sec)
            frame.mesh = secs
        return frame

    def _settings_for(self, kind: str) -> OV.OverlaySettings:
        s = self._settings
        group = DR.KIND_OVERLAY.get(kind)
        for tab in OV.TABS:
            s.group(tab).enabled = (tab == group) or (kind == "tracks" and tab == "labels")
        if kind == "mask":
            s.voxels.style = "mask"
            s.voxels.opacity = 55
        elif kind == "scalar":
            s.voxels.style = "heatmap"
            s.voxels.opacity = 65
        if kind == "mesh":
            s.mesh.style = "points"
            s.mesh.near_z = 1.0
        if kind == "tracks":
            s.labels.fill_opacity = 10
        return s

    def _paint_after_overlay(self, p) -> None:
        if self._frame is None:
            return
        self._renderer.paint(p, self._settings_for(self.recipe.kind), self._frame)

    def _paint_before_overlay(self, p) -> None:
        if self._before_frame is None:
            return
        self._renderer.paint(p, self._settings_for("labels"), self._before_frame)

    def _fill_table(self, res: DR.DemoResult) -> None:
        key, cols = DR.demo_rows(res, prefer=list(res.written))
        tbl = self._table
        tbl.clear()
        if not cols:
            tbl.setRowCount(0); tbl.setColumnCount(0)
            return
        names = list(cols)
        first = [n for n in ("id", "track_id", "m", "t", "c") if n in names]
        names = first + sorted(n for n in names if n not in first)
        n_rows = min(_MAX_ROWS, max(len(np.atleast_1d(np.asarray(v))) for v in cols.values()))
        tbl.setColumnCount(len(names))
        tbl.setRowCount(n_rows)
        tbl.setHorizontalHeaderLabels(names)
        for j, name in enumerate(names):
            vals = np.atleast_1d(np.asarray(cols[name]))
            for i in range(min(n_rows, len(vals))):
                v = vals[i]
                if isinstance(v, (float, np.floating)):
                    text = f"{float(v):.4g}"
                else:
                    text = str(v)
                tbl.setItem(i, j, QTableWidgetItem(text))
        tbl.resizeColumnsToContents()
        self._cap_after.setText(f"After — {key[0]} table “{key[1]}”, {n_rows} rows"
                                if key else "After — the table")

    def _toggle_guide(self, on: bool) -> None:
        self._guide.setVisible(on)
        self._guide_btn.setText("How it works ▴" if on else "How it works ▾")

    # ── lifecycle ────────────────────────────────────────────────────────────
    def closeEvent(self, e) -> None:
        self.shutdown()
        self.closed.emit(self.op_key)
        super().closeEvent(e)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None
