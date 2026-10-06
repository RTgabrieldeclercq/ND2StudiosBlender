"""The dock shell (V4.00 step 3): every panel of the window is a dock that can pop out into
its own window, dock back, close and come back — and the arrangement survives a restart.

Three pieces:

* :class:`PanelSpec` — what a KIND of panel is: its title, the widget factory, where it
  lives by default, and whether several instances may exist (``multi``: the viewers of
  step 4 and the canvases of step 5; every step-3 panel is single).
* :class:`PanelDock` — one ``QDockWidget`` per panel instance, ``objectName`` =
  ``"<kind>:<index>"`` (what Qt's ``saveState`` keys positions by), movable, floatable and
  closable, with a :class:`PanelTitleBar` instead of the native caption: the kind glyph, the
  title (elided, and naming what the panel shows once a binding is set), ``+`` for a multi
  kind, pop-out/dock-back, and close.
* :class:`DockShell` — owned by the window (composition, not a mixin: PySide's
  multiple-inheritance rules make a QMainWindow mixin fragile). It registers specs, spawns
  docks, places them in the default layout, resets it, builds the View ▸ Panels / New menus,
  tracks which instance of each multi kind is ACTIVE (the one the user last worked in), and
  saves/restores the layout through :mod:`nodelab_v2.layout_store`.

A single panel's close only HIDES it (View ▸ Panels brings it back, exactly as it was); a
multi panel's close DESTROYS that instance, unless the window's ``allow_close`` vetoes it.
The one veto today is ``MainWindow._allow_panel_close``: while the canvas is maximized, the
dock of the viewer living in the mini-map stays (closing it would leave that viewer with no
home). A vetoed dock shows its ✕ disabled (:meth:`DockShell.sync_close_buttons`), and the
close is refused if asked anyway.

The dock paints its own surface (:meth:`PanelDock.paintEvent`, V4.00 step 11a): with a
custom title bar ``QDockWidget`` paints nothing at all, so a floating dock — a frameless
top-level window — showed its palette's Window brush (Qt's default light grey) through the
frame gutter around the panel.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QPoint, Qt, Signal
from PySide6.QtGui import QAction, QFont, QFontMetrics, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QScrollArea,
    QSizePolicy, QToolButton, QWidget)

from nodelab_v2 import layout_store as LS
from nodelab_v2 import theme as T
from nodelab_v2.version import __version__

#: Pop-out / dock-back glyphs on a panel's title bar (the arrows block renders in the stock
#: Windows UI fonts; the tooltip carries the words).
FLOAT_GLYPH = "⇱"
DOCK_GLYPH = "⇲"
#: A floating panel restored partly off every screen is moved back onto one when less than
#: this much of it (px) would be grabbable — a monitor unplugged since the layout was saved.
MIN_VISIBLE_W, MIN_VISIBLE_H = 80, 24


@dataclass(frozen=True)
class PanelSpec:
    """One KIND of panel."""
    kind: str
    title: str
    #: builds the panel widget; a single kind may return a widget the window already owns
    factory: Callable[[], QWidget]
    glyph: str = ""
    multi: bool = False
    default_area: Any = Qt.RightDockWidgetArea
    allowed_areas: Any = Qt.AllDockWidgetAreas
    default_hidden: bool = False
    #: the kind this one sits behind as a tab in the default layout
    tabify_with: str = ""
    #: the visible tab of its group in the default layout
    raise_default: bool = False
    #: wrap the panel in a scroll area, so its minimum size can never become the window's
    scroll: bool = False
    #: (multi kinds) what an instance shows, as JSON — saved with the layout …
    binding_of: Optional[Callable[[QWidget], Any]] = None
    #: … and shown again after a restore
    apply_binding: Optional[Callable[[QWidget, Any], None]] = None


class _ElidedLabel(QLabel):
    """A label that elides its text to the width it is given instead of demanding more."""

    def __init__(self, text: str = "") -> None:
        super().__init__()
        self._full = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(10)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = str(text or "")
        self.setToolTip(self._full)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event) -> None:            # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        w = max(10, self.width())
        super().setText(QFontMetrics(self.font()).elidedText(self._full, Qt.ElideRight, w))


class PanelTitleBar(QWidget):
    """A panel's caption: glyph · title · [+] · pop-out/dock-back · close.

    Presses on the bar itself are left unhandled on purpose — ``QDockWidget`` then drags the
    panel by it and floats/docks it on a double-click, exactly as with the native caption."""

    def __init__(self, dock: "PanelDock") -> None:
        super().__init__(dock)
        self._dock = dock
        self._active = False
        # QStyleSheetStyle sets this by itself only on a PLAIN QWidget; on a subclass the
        # `PanelTitleBar { … border-left … }` rule would match and paint nothing
        self.setAttribute(Qt.WA_StyledBackground, True)
        spec = dock.spec
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 3, 4, 3)
        h.setSpacing(4)
        self._glyph = QLabel(spec.glyph)
        self._title = _ElidedLabel(spec.title)
        h.addWidget(self._glyph)
        h.addWidget(self._title, 1)
        self.add_btn: Optional[QToolButton] = None
        if spec.multi:
            self.add_btn = self._button("+", f"Open another {spec.title} beside this one",
                                        dock.request_new)
            h.addWidget(self.add_btn)
        self.float_btn = self._button(FLOAT_GLYPH, "", dock.toggle_float)
        self._close_tip = ("Close this panel — View ▸ Panels brings it back"
                           if not spec.multi else "Close this panel")
        self.close_btn = self._button("✕", self._close_tip, dock.close)
        h.addWidget(self.float_btn)
        h.addWidget(self.close_btn)
        # the glyphs (◉ ◫ ☰ ⇱ ⇲ ✕ …) live in Segoe UI Symbol; naming it as the fallback
        # family keeps them from rendering as boxes where the UI font lacks them (the QSS
        # below sets only the size, so the families stand)
        for w in (self._glyph, self.float_btn, self.close_btn):
            f = QFont(w.font())
            f.setFamilies([T.SANS, "Segoe UI Symbol"])
            w.setFont(f)
        self.set_floating(False)
        self.restyle()

    def _button(self, text: str, tip: str, slot) -> QToolButton:
        b = QToolButton(self)
        b.setText(text)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(lambda _=False: slot())
        return b

    def title_text(self) -> str:
        return self._title.full_text()

    def set_text(self, text: str) -> None:
        self._title.set_full_text(text)

    def set_floating(self, on: bool) -> None:
        self.float_btn.setText(DOCK_GLYPH if on else FLOAT_GLYPH)
        self.float_btn.setToolTip("Dock back into the main window" if on
                                  else "Pop out into its own window (double-click the "
                                       "title does the same)")

    def set_active(self, on: bool) -> None:
        if bool(on) != self._active:
            self._active = bool(on)
            self.restyle()

    def sync_close(self, enabled: bool) -> None:
        """Show whether the ✕ would work: a dock the window vetoes right now (the mini-map's
        viewer while the canvas is maximized) has it disabled, with the reason as the tip,
        instead of a button that looks live and does nothing."""
        self.close_btn.setEnabled(bool(enabled))
        self.close_btn.setToolTip(self._close_tip if enabled
                                  else "This panel cannot be closed right now")

    def mousePressEvent(self, e) -> None:            # noqa: N802 — Qt override
        """A press on the title bar makes this the active instance, then goes on to the
        dock (QWidget ignores it), which is what drags or pops the panel out."""
        self._dock.pressed.emit(self._dock)
        super().mousePressEvent(e)

    def is_active(self) -> bool:
        return self._active

    def restyle(self) -> None:
        edge = T.ACCENT.name() if self._active else T.BORDER.name()
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"PanelTitleBar {{ background:{T.BODY.name()}; border-bottom:1px solid "
            f"{T.BORDER.name()}; border-left:3px solid {edge}; }}"
            f"QLabel {{ color:{(T.INK if self._active else T.MUTED).name()}; font-size:10px; "
            f"font-weight:800; background:transparent; }}"
            f"QToolButton {{ color:{T.MUTED.name()}; background:transparent; border:0; "
            f"border-radius:4px; padding:0 5px; font-size:12px; }}"
            f"QToolButton:hover {{ background:{T.PANEL_HI.name()}; color:{T.INK.name()}; }}"
            f"QToolButton:disabled {{ color:{T.BORDER.name()}; }}")


class PanelDock(QDockWidget):
    """One panel instance in the dock shell."""

    #: the panel was closed by the user (its ✕, View ▸ Panels, Alt+F4 when floating)
    closed = Signal(object)
    #: the user started working in it (focus moved inside)
    activated = Signal(object)
    #: the title bar's ``+``
    new_requested = Signal(object)
    #: a press on the title bar — picking a panel up is working in it
    pressed = Signal(object)

    def __init__(self, spec: PanelSpec, index: int, panel: QWidget,
                 parent: QMainWindow) -> None:
        super().__init__(spec.title, parent)
        self.spec = spec
        self.index = int(index)
        self.panel = panel
        #: set by the shell: may this dock close right now? (a veto, e.g. the last canvas)
        self.can_close: Callable[["PanelDock"], bool] = lambda _d: True
        self._binding_text = ""
        self.setObjectName(LS.dock_name(spec.kind, index))
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                         | QDockWidget.DockWidgetClosable)
        self.setAllowedAreas(spec.allowed_areas)
        body: QWidget = panel
        if spec.scroll:
            sc = QScrollArea()
            sc.setObjectName("panelScroll")        # styled by the window's QSS (dark viewport)
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            sc.viewport().setAutoFillBackground(False)
            sc.setWidget(panel)
            body = sc
        self.setWidget(body)
        self.title_bar = PanelTitleBar(self)
        self.setTitleBarWidget(self.title_bar)
        self.topLevelChanged.connect(self.title_bar.set_floating)
        self.topLevelChanged.connect(lambda _on=False: self.update())   # the frame (paintEvent)

    @property
    def kind(self) -> str:
        return self.spec.kind

    def paintEvent(self, event) -> None:                # noqa: N802 — Qt override
        """The dock's own surface. With a custom title bar ``QDockWidget`` paints NOTHING,
        so a floating dock — a frameless top-level window — showed its palette's Window brush
        (Qt's default light grey) in the frame gutter around the panel, and a docked one let
        it through wherever the panel's margins are transparent. Fill with the theme's
        background; a floating dock also gets a 1 px border, so it has an edge against
        whatever is behind it."""
        p = QPainter(self)
        p.fillRect(self.rect(), T.BG)
        if self.isFloating():
            p.setPen(QPen(T.BORDER, 1))
            p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.end()

    def sync_close(self) -> None:
        """Reflect the window's veto on the title bar's ✕ (:meth:`PanelTitleBar.sync_close`)."""
        self.title_bar.sync_close(bool(self.can_close(self)))

    def toggle_float(self) -> None:
        self.setFloating(not self.isFloating())
        if self.isFloating():
            self.raise_()
            self.activateWindow()

    def request_new(self) -> None:
        self.new_requested.emit(self)

    def binding_text(self) -> str:
        return self._binding_text

    def set_binding_title(self, text: str) -> None:
        """Name what this instance shows (``"n3 · Gaussian"``): on its title bar, as the
        floating window's caption, and in View ▸ Panels — so two viewers are tellable apart
        everywhere they are listed."""
        self._binding_text = str(text or "")
        full = self.spec.title + (f" · {self._binding_text}" if self._binding_text else "")
        self.setWindowTitle(full)
        self.title_bar.set_text(full)

    def closeEvent(self, event) -> None:                # noqa: N802 — Qt override
        if not self.can_close(self):
            event.ignore()
            # View ▸ Panels unticks its action BEFORE asking the dock to close; a refused
            # close sends no hide, so nothing would put the tick back on a panel still open
            act = self.toggleViewAction()
            if not act.isChecked():
                act.blockSignals(True)
                act.setChecked(True)
                act.blockSignals(False)
            return
        super().closeEvent(event)
        if event.isAccepted():
            self.closed.emit(self)


class DockShell(QObject):
    """The window's panels: specs, instances, the default layout, menus, persistence."""

    #: a panel became the one the user works in — ``(dock)``; per kind, see :meth:`active`
    activated = Signal(object)
    #: the set of panels changed (an instance spawned or destroyed) — menus rebuild from it
    changed = Signal()

    def __init__(self, window: QMainWindow, *,
                 allow_close: Optional[Callable[[PanelDock], bool]] = None) -> None:
        super().__init__(window)
        self.win = window
        self.specs: Dict[str, PanelSpec] = {}
        #: objectName → dock, in creation order
        self.docks: Dict[str, PanelDock] = {}
        #: may this dock close right now? The window vetoes e.g. its last canvas (step 5)
        self.allow_close: Callable[[PanelDock], bool] = allow_close or (lambda _d: True)
        self._active: Dict[str, PanelDock] = {}
        window.setDockNestingEnabled(True)
        window.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks
                              | QMainWindow.AllowTabbedDocks)
        # the side columns own all four corners, so a bottom panel sits under the CANVAS
        # rather than under the full-height Properties column (see the Movie Editor note
        # in window.py: spanning the width raised the window's minimum height off-screen),
        # and a top one — the Viewers (V4.00 step 4) — over it, as the old splitter had it
        window.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        window.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        window.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
        window.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus)

    # ── specs and instances ───────────────────────────────────────────────────
    def register(self, spec: PanelSpec) -> PanelSpec:
        if LS.parse_dock_name(f"{spec.kind}:0") is None:
            raise ValueError(f"panel kind {spec.kind!r} must be lower_snake_case")
        if spec.kind in self.specs:
            raise ValueError(f"panel kind {spec.kind!r} is already registered")
        self.specs[spec.kind] = spec
        self.changed.emit()
        return spec

    def docks_of(self, kind: str) -> List[PanelDock]:
        return sorted((d for d in self.docks.values() if d.kind == kind),
                      key=lambda d: d.index)

    def dock(self, name: str) -> Optional[PanelDock]:
        return self.docks.get(name)

    def dock_of(self, widget: Optional[QWidget]) -> Optional[PanelDock]:
        """The panel ``widget`` lives in (itself, or any ancestor), or ``None``."""
        w = widget
        while w is not None:
            if isinstance(w, PanelDock) and self.docks.get(w.objectName()) is w:
                return w
            w = w.parentWidget()
        return None

    def spawn(self, kind: str, *, index: Optional[int] = None,
              beside: Optional[PanelDock] = None, area: Any = None,
              floating: bool = False, show: bool = True) -> PanelDock:
        """The dock for ``kind`` — a single kind's one instance (created on first call), or
        a NEW instance of a multi kind (``index`` = the lowest free one unless given; an
        index that exists returns that instance). ``beside`` splits the new dock next to an
        existing one; otherwise it goes to ``area`` (default: the spec's)."""
        spec = self.specs[kind]
        if not spec.multi:
            have = self.docks_of(kind)
            if have:
                return have[0]
            index = 0
        else:
            taken = {d.index for d in self.docks_of(kind)}
            if index is None:
                index = next(i for i in itertools.count() if i not in taken)
            elif int(index) in taken:
                return self.docks[LS.dock_name(kind, int(index))]
        dock = PanelDock(spec, int(index), spec.factory(), self.win)
        dock.can_close = self._can_close
        dock.closed.connect(self._on_closed)
        dock.new_requested.connect(lambda d: self.spawn(d.kind, beside=d))
        dock.pressed.connect(self.activate)
        self.docks[dock.objectName()] = dock
        if beside is not None and self.docks.get(beside.objectName()) is beside \
                and not beside.isFloating():
            self.win.addDockWidget(self.win.dockWidgetArea(beside), dock)
            placed_as_tab = self._place_beside(beside, dock)
        else:
            placed_as_tab = False
            self.win.addDockWidget(area if area is not None else spec.default_area, dock)
        if floating:
            dock.setFloating(True)
        dock.setVisible(bool(show))
        if show and placed_as_tab:
            dock.raise_()               # '+' asked to SEE another one: its tab goes in front
        if spec.multi and show:
            self.activate(dock)
        self.sync_close_buttons()
        self.changed.emit()
        return dock

    def _place_beside(self, prev: PanelDock, dock: PanelDock) -> bool:
        """Put ``dock`` next to ``prev`` (both already in the window): as another TAB when
        ``prev`` sits in a tab group — Qt's split of a tabbed dock takes both out of the
        group and leaves them off screen — and split to its right otherwise. ``True`` when
        it became a tab (asked here, of ``prev``: a dock not yet shown is left out of
        ``tabifiedDockWidgets``, so asking about ``dock`` afterwards would say no)."""
        if self.win.tabifiedDockWidgets(prev):
            self.win.tabifyDockWidget(prev, dock)
            return True
        self.win.splitDockWidget(prev, dock, Qt.Horizontal)
        return False

    def _can_close(self, dock: PanelDock) -> bool:
        try:
            return bool(self.allow_close(dock))
        except Exception:                               # noqa: BLE001 — a veto must not crash
            return True

    def sync_close_buttons(self) -> None:
        """Every dock's ✕ reflects whether the window would let it close right now. The
        veto is a function of window state, so this runs whenever that state moves: a
        dock spawned or destroyed, the default layout applied, the canvas maximized or
        restored (``MainWindow.set_maximized``)."""
        for d in list(self.docks.values()):
            d.sync_close()

    def _on_closed(self, dock: PanelDock) -> None:
        if not dock.spec.multi:
            return                                     # hidden; View ▸ Panels brings it back
        self.docks.pop(dock.objectName(), None)
        if self._active.get(dock.kind) is dock:
            self._active.pop(dock.kind, None)
            rest = self.docks_of(dock.kind)
            if rest:
                self.activate(rest[0])
        self.win.removeDockWidget(dock)
        dock.deleteLater()
        self.sync_close_buttons()
        self.changed.emit()

    # ── the active instance of each kind ─────────────────────────────────────
    def active(self, kind: str) -> Optional[PanelDock]:
        d = self._active.get(kind)
        return d if d is not None and self.docks.get(d.objectName()) is d else None

    def activate(self, dock: PanelDock) -> None:
        """Make ``dock`` the active instance of its kind: accent on its title bar (multi
        kinds only — "which viewer" is a question, "which Properties" is not), signals."""
        if self.docks.get(dock.objectName()) is not dock:
            return
        prev = self._active.get(dock.kind)
        self._active[dock.kind] = dock
        if dock.spec.multi:
            for d in self.docks_of(dock.kind):
                d.title_bar.set_active(d is dock)
        if prev is not dock:
            dock.activated.emit(dock)
            self.activated.emit(dock)

    def deactivate(self, kind: str) -> None:
        """No instance of ``kind`` is the active one any more — the window's work moved to
        something that is not one of these docks (V4.00 step 5: the MAIN canvas, the window's
        centre). Clears the accent, and the record a later close or press would act on."""
        self._active.pop(kind, None)
        for d in self.docks_of(kind):
            d.title_bar.set_active(False)

    def _on_focus(self, _old, new) -> None:
        try:
            d = self.dock_of(new) if new is not None else None
        except RuntimeError:                           # a widget torn down mid-signal
            return
        if d is not None:
            self.activate(d)

    # ── layout ────────────────────────────────────────────────────────────────
    def apply_default_layout(self) -> None:
        """Every panel back where a fresh install puts it: docked in its default area,
        tabbed per spec, the default tab raised, default-hidden panels hidden. Instances of
        a multi kind beyond the first are split beside the first."""
        for d in self.docks.values():
            d.setFloating(False)
            self.win.removeDockWidget(d)
        placed: Dict[str, PanelDock] = {}
        for spec in self.specs.values():
            for i, d in enumerate(self.docks_of(spec.kind)):
                if i == 0:
                    self.win.addDockWidget(spec.default_area, d)
                    anchor = placed.get(spec.tabify_with)
                    if anchor is not None:
                        self.win.tabifyDockWidget(anchor, d)
                    placed[spec.kind] = d
                else:
                    # beside the PREVIOUS instance, so they line up in index order (beside
                    # the first, each would land between it and the ones placed before)
                    self.win.addDockWidget(spec.default_area, d)
                    self._place_beside(self.docks_of(spec.kind)[i - 1], d)
                d.setVisible(not spec.default_hidden)
        for spec in self.specs.values():
            if spec.raise_default:
                for d in self.docks_of(spec.kind):
                    d.raise_()
        self.sync_close_buttons()

    def reset_layout(self) -> None:
        self.apply_default_layout()

    def layout_record(self) -> Dict[str, Any]:
        """The JSON-able layout of the window right now (:func:`layout_store.make_layout`)."""
        docks = []
        for d in self.docks.values():
            binding = None
            if d.spec.binding_of is not None:
                try:
                    binding = d.spec.binding_of(d.panel)
                except Exception:                      # noqa: BLE001 — remember the rest
                    binding = None
            docks.append({"name": d.objectName(), "kind": d.kind, "binding": binding})
        return LS.make_layout(geometry=bytes(self.win.saveGeometry()),
                              state=bytes(self.win.saveState(LS.LAYOUT_VERSION)),
                              docks=docks, app_version=__version__)

    def save_layout(self, path=None):
        """Write the layout (:mod:`layout_store`). Returns the path written."""
        return LS.save_layout(self.layout_record(), path)

    def restore_layout(self, path=None, *, quarantine: bool = False) -> bool:
        """Re-create the saved instances of multi kinds, then hand Qt the saved geometry and
        state. ``False`` (and the default layout untouched) when there is nothing usable;
        ``quarantine`` sets such a file aside (:func:`layout_store.load_layout`)."""
        rec = LS.load_layout(path, quarantine=quarantine)
        if rec is None:
            return False
        bindings = []
        for d in rec["docks"]:
            spec = self.specs.get(d["kind"])
            if spec is None or not spec.multi:
                continue                               # unknown kinds are simply skipped
            dock = self.spawn(spec.kind, index=d["index"], show=False)
            if d.get("binding") is not None and spec.apply_binding is not None:
                bindings.append((spec, dock, d["binding"]))
        if rec["geometry"]:
            self.win.restoreGeometry(rec["geometry"])
        if not self.win.restoreState(rec["state"], LS.LAYOUT_VERSION):
            self.apply_default_layout()
            return False
        for spec, dock, binding in bindings:
            try:
                spec.apply_binding(dock.panel, binding)
            except Exception:                          # noqa: BLE001 — an empty panel is fine
                pass
        self.clamp_floating()
        return True

    def clamp_floating(self) -> None:
        """Bring back any floating panel that would sit off every screen (a monitor that
        was unplugged since the layout was saved): centred on the window's screen."""
        screens = [s.availableGeometry() for s in QGuiApplication.screens()]
        if not screens:
            return
        home = (self.win.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        for d in self.docks.values():
            if not d.isFloating():
                continue
            g = d.frameGeometry()
            ok = any(s.intersected(g).width() >= MIN_VISIBLE_W
                     and s.intersected(g).height() >= MIN_VISIBLE_H for s in screens)
            if not ok:
                d.move(home.center() - QPoint(g.width() // 2, g.height() // 2))

    # ── menus ─────────────────────────────────────────────────────────────────
    def fill_panels_menu(self, menu: QMenu,
                         overrides: Optional[Dict[str, QAction]] = None) -> None:
        """View ▸ Panels: one tick per panel instance (its own toggle action, or the one in
        ``overrides`` — the Console keeps the action that carries Ctrl+`)."""
        menu.clear()
        overrides = overrides or {}
        for d in self.docks.values():
            menu.addAction(overrides.get(d.objectName()) or d.toggleViewAction())

    def fill_new_menu(self, menu: QMenu) -> None:
        """View ▸ New: one entry per multi kind; hidden when there is none."""
        menu.clear()
        kinds = [s for s in self.specs.values() if s.multi]
        for spec in kinds:
            act = menu.addAction(f"{spec.glyph}  {spec.title}".strip())
            act.triggered.connect(lambda _=False, k=spec.kind: self.spawn(
                k, beside=self.active(k) or (self.docks_of(k) or [None])[-1]))
        menu.menuAction().setVisible(bool(kinds))

    def restyle(self) -> None:
        for d in self.docks.values():
            d.title_bar.restyle()
            d.update()                  # the dock's own fill and frame read the tokens too


__all__ = ["PanelSpec", "PanelTitleBar", "PanelDock", "DockShell", "FLOAT_GLYPH",
           "DOCK_GLYPH"]
