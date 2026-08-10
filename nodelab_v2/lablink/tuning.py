"""The warm tuning loop — hold a session open, turn a knob, look, turn it again.

A hub holds heavy software warm so that changing one knob and running again recomputes only
what that knob invalidated: a run that costs a second cold comes back in milliseconds. That
reuse is the entire reason a hub exists, and it is only available to a client that **keeps
the session open between attempts**. Opening a fresh session per attempt throws it away and
pays the cold cost every time.

Three pieces, all Qt:

:class:`SessionController`
    Owns one live :class:`~nodelab_v2.lablink.client.Session` on its own thread and takes
    work from a queue. Everything the GUI needs comes back as a signal, so no widget is ever
    touched off the GUI thread and no hub call ever blocks it.

:class:`KnobForm`
    The recipe's knobs as real bounded controls, with the three-state rule
    (default / derived / pinned) expressible and ``applies_when`` honoured.

:class:`ResultCard`
    What came back, rendered where the knobs are — because judging a knob change means
    looking at the picture, and a person who has to leave the panel to do that will not
    iterate.

Qt lives in exactly two modules of this package (this one and
:mod:`~nodelab_v2.lablink.panel`, plus the authoring dialog); the worker, client, protocol,
sidecar and preset modules stay importable on a box with no display.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from nodelab_v2 import theme as T
from nodelab_v2.lablink.client import (
    FileRef, HubClient, LabLinkError, Session, SessionLost, knob_applies,
)

#: How long :meth:`SessionController.shutdown` waits for the worker thread to unwind. It
#: has to cover a close call to a hub that has stopped answering, and closing is not
#: optional — a leaked session holds one of very few slots until an idle timer expires.
SHUTDOWN_WAIT_MS = 8000

#: Rows of a returned table to show inline. The worker writes a `*.preview.csv` capped at
#: its own row count; this is the ceiling on what the card renders from it.
PREVIEW_ROWS = 12


# ── the value a knob control holds ──────────────────────────────────────────────

#: The three states a knob can be in, which are three different things on the wire and are
#: constantly conflated. ``DEFAULT`` omits the knob; ``DERIVE`` sends an explicit ``null``;
#: ``PINNED`` sends the value — including a pinned ``0``, which is a real zero and silently
#: overrides the file's own optics if you meant "leave it alone".
KNOB_DEFAULT = "default"
KNOB_DERIVE = "derive"
KNOB_PINNED = "pinned"


def knob_state_label(knob: Dict[str, Any]) -> str:
    """What "leave this alone" resolves to, for a hint beside the control.

    An empty control reads as broken. What it means is "the operator already chose" or "the
    microscope already answered", and those are different enough to say — but they are said
    in a hint rather than inside the selector, because a selector wide enough to hold the
    sentence leaves no room for the value it is qualifying.
    """
    if knob.get("unset_means") == "derive" or knob.get("derive"):
        return "derived from the file"
    default = knob.get("default")
    if default is not None:
        return f"recipe default {default}"
    return "left to the recipe"


# ── the session controller ──────────────────────────────────────────────────────

@dataclass
class RunOutcome:
    """One command's result, flattened for the GUI (no live handles cross the thread)."""

    ok: bool = False
    state: str = ""
    code: str = ""
    message: str = ""
    duration_s: Optional[float] = None
    cached_steps: int = 0
    computed_steps: int = 0
    total_steps: int = 0
    cmd_id: str = ""
    cmd_seq: int = 0
    #: The hub's echo of every knob in effect — the record of what actually ran.
    knobs: Dict[str, Any] = None            # type: ignore[assignment]
    artifacts: List[Dict[str, Any]] = None  # type: ignore[assignment]
    held: List[str] = None                  # type: ignore[assignment]
    #: Local paths fetched for this run, once collected.
    fetched: List[str] = None               # type: ignore[assignment]
    results_dir: str = ""

    def __post_init__(self) -> None:
        for name in ("knobs", "artifacts", "held", "fetched"):
            if getattr(self, name) is None:
                setattr(self, name, {} if name == "knobs" else [])

    @property
    def speedup_note(self) -> str:
        """The sentence that makes the warm loop visible: how much was skipped."""
        if self.total_steps and self.cached_steps:
            return (f"{self.cached_steps}/{self.total_steps} steps came from the hub's warm "
                    f"cache")
        if self.total_steps:
            return f"{self.computed_steps}/{self.total_steps} steps computed"
        return ""


class SessionController(QObject):
    """One warm session, driven from a queue on its own thread.

    The GUI never blocks and never holds the :class:`Session`: it posts intents
    (:meth:`open`, :meth:`send_file`, :meth:`run`, :meth:`pull`, :meth:`close`) and reads
    signals. Everything that can fail reports through :attr:`failed` with a *where*, so a
    message can be shown against the thing that produced it rather than as a bare string.

    **Closing runs in a ``finally``.** A session holds a running program and, on most hubs,
    one of very few slots; a hub with no free slots is indistinguishable from a broken one
    to the next machine that asks. The thread therefore closes on every exit path — normal
    stop, error, or window close — and :meth:`shutdown` waits for it.
    """

    opened = Signal(dict)            # {"id", "quota_mb", "in_channel", "out_channel"}
    state_changed = Signal(str)      # idle | opening | ready | running | closing | closed
    progress = Signal(object)        # nodelab_v2.lablink.client.Progress
    ran = Signal(object)             # RunOutcome
    fetched = Signal(object)         # RunOutcome, with `fetched` filled
    failed = Signal(str, str)        # where, message
    log = Signal(str)
    lost = Signal(str)               # the warm worker is gone; the cache went with it

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._q: "queue.Queue[Optional[Tuple[str, dict]]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[Session] = None
        self._hub: Optional[HubClient] = None
        self._refs: List[FileRef] = []
        self._state = "idle"
        self._runs = 0
        #: Set while a command is in flight, so the panel can offer Cancel and refuse a
        #: second run rather than earning a `409 busy` from the hub.
        self.busy = threading.Event()
        self._stopping = threading.Event()

    # ── state ───────────────────────────────────────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    @property
    def alive(self) -> bool:
        return self._state in ("ready", "running")

    @property
    def session_id(self) -> str:
        return self._session.id if self._session else ""

    @property
    def inputs(self) -> List[FileRef]:
        return list(self._refs)

    def _set_state(self, state: str) -> None:
        self._state = state
        self.state_changed.emit(state)

    # ── the queue ───────────────────────────────────────────────────────────────
    def _post(self, op: str, **kwargs: Any) -> None:
        self._q.put((op, kwargs))

    def open(self, hub: HubClient, workflow: str, recipe: str, *,
             knobs: Dict[str, Any], label: str = "ND2 Studios editor") -> None:
        """Open a session. Starts the worker thread if it is not already running."""
        if self._thread is None or not self._thread.is_alive():
            self._stopping.clear()
            self._thread = threading.Thread(target=self._serve, name="lablink-session",
                                            daemon=True)
            self._thread.start()
        self._hub = hub
        self._set_state("opening")
        self._post("open", hub=hub, workflow=workflow, recipe=recipe,
                   knobs=dict(knobs), label=label)

    def send_file(self, path: str, *, sidecar_path: str = "") -> None:
        """Upload an image, and its sidecar if one was written. Once per session."""
        self._post("send", path=path, sidecar_path=sidecar_path)

    def run(self, knobs: Dict[str, Any], *, cmd_id: str = "",
            results_dir: str = "", fetch: bool = True) -> None:
        """Run with ``knobs`` as the full set in force. This is the loop's hot path."""
        self._post("run", knobs=dict(knobs), cmd_id=cmd_id, results_dir=results_dir,
                   fetch=fetch)

    def pull(self, names: List[str], *, results_dir: str) -> None:
        """Publish and fetch artifacts the recipe held back, by name."""
        self._post("pull", names=list(names), results_dir=results_dir)

    def close(self) -> None:
        self._post("close")

    def cancel(self) -> None:
        """Ask the hub to stop the running command.

        Sent from a throwaway thread rather than the queue: the queue is busy with the very
        run being cancelled, so a queued cancel could not be read until it finished — which
        is precisely never, from the point of view of somebody watching a wedged run.
        """
        session, hub = self._session, self._hub
        if session is None or hub is None:
            return

        def fire() -> None:
            try:
                reply = session.cancel()
                self.log.emit(f"cancel requested: {reply.get('detail') or reply}")
            except Exception as exc:                     # noqa: BLE001
                self.failed.emit("cancel", f"{type(exc).__name__}: {exc}")

        threading.Thread(target=fire, name="lablink-cancel", daemon=True).start()

    def shutdown(self) -> None:
        """Stop the thread, closing any live session first. Safe to call twice."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._stopping.set()
        self._q.put(None)
        self._thread.join(SHUTDOWN_WAIT_MS / 1000.0)
        if self._thread.is_alive():
            print("[lablink] the session thread did not finish within "
                  f"{SHUTDOWN_WAIT_MS} ms; abandoning it")

    # ── the thread ──────────────────────────────────────────────────────────────
    def _serve(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    return
                op, kwargs = item
                if op == "close":
                    return
                self.busy.set()
                try:
                    self._dispatch(op, kwargs)
                except SessionLost as exc:
                    # The warm instance is gone and its cache with it. Surfaced, never
                    # silently reopened: reopening and rerunning hides a result the person
                    # may already be looking at, and the next run would be quietly slower
                    # and possibly different.
                    self._session = None
                    self._set_state("closed")
                    self.lost.emit(str(exc))
                except LabLinkError as exc:
                    self.failed.emit(op, str(exc))
                    if self._state == "opening":
                        self._set_state("idle")
                    elif self.alive:
                        self._set_state("ready")
                except Exception as exc:                 # noqa: BLE001 — never kill the loop
                    self.failed.emit(op, f"{type(exc).__name__}: {exc}")
                    if self.alive:
                        self._set_state("ready")
                finally:
                    self.busy.clear()
        finally:
            # Not optional, and not only on the happy path.
            self._teardown()

    def _teardown(self) -> None:
        session, self._session = self._session, None
        if session is None:
            self._set_state("closed")
            return
        self._set_state("closing")
        try:
            reply = session.close()
            self.log.emit(
                f"session closed after {reply.get('commands_run', '?')} command(s), "
                f"{reply.get('bytes_out', 0)} bytes back")
        except Exception as exc:                         # noqa: BLE001 — closing must not raise
            self.log.emit(f"the session could not be closed cleanly: {exc}")
        self._set_state("closed")

    def _dispatch(self, op: str, kwargs: Dict[str, Any]) -> None:
        if op == "open":
            self._do_open(**kwargs)
        elif op == "send":
            self._do_send(**kwargs)
        elif op == "run":
            self._do_run(**kwargs)
        elif op == "pull":
            self._do_pull(**kwargs)

    def _do_open(self, hub: HubClient, workflow: str, recipe: str,
                 knobs: Dict[str, Any], label: str) -> None:
        session = hub.open_session(workflow, recipe, label=label, knobs=knobs)
        self._session = session
        self._refs = []
        self._runs = 0
        self._set_state("ready")
        self.opened.emit({"id": session.id, "quota_mb": session.quota_mb,
                          "in_channel": session.in_channel,
                          "out_channel": session.out_channel,
                          "knobs": dict(session.knobs)})

    def _do_send(self, path: str, sidecar_path: str) -> None:
        session = self._need_session()
        ref = session.send_data(path)
        self._refs = [ref]
        self.log.emit(f"sent {ref.name} ({_bytes(ref.size)})"
                      + (" — name repaired for the exchange" if ref.repaired else ""))
        if sidecar_path:
            side = session.send_data(sidecar_path)
            self.log.emit(f"sent {side.name} — the calibration this analysis derives from")
            # NOT added to `self._refs`: the sidecar is picked up beside its image by the
            # worker, and naming it as a command input would make the recipe look for a
            # second declared role it does not have.
        self._set_state("ready")

    def _do_run(self, knobs: Dict[str, Any], cmd_id: str, results_dir: str,
                fetch: bool) -> None:
        session = self._need_session()
        self._runs += 1
        self._set_state("running")
        started = time.time()
        result = session.run(inputs=self._refs, knobs=knobs, check=False,
                             cmd_id=cmd_id or f"tune-{self._runs}",
                             on_progress=self.progress.emit)
        outcome = RunOutcome(
            ok=result.ok, state=result.state, code=result.code, message=result.message,
            duration_s=result.duration_s if result.duration_s is not None
            else round(time.time() - started, 3),
            cached_steps=int(getattr(result, "cached_steps", 0) or 0),
            computed_steps=int(getattr(result.progress, "computed", 0) or 0),
            total_steps=int(getattr(result.progress, "total", 0) or 0),
            cmd_id=result.cmd_id, cmd_seq=result.cmd_seq,
            # The hub's echo where it has one, and the locally-computed effective set where
            # it does not: the echo is only populated for a command that carried knobs, so a
            # run at the recipe's defaults would otherwise be recorded as having no settings
            # at all.
            knobs=dict(result.knobs or HubClient.effective_knobs(session.recipe_meta,
                                                                 knobs)),
            artifacts=[dict(a) for a in (result.artifacts or [])],
            held=list(result.held()), results_dir=results_dir)
        self._set_state("ready")
        self.ran.emit(outcome)

        if fetch and results_dir and (result.ok or outcome.artifacts):
            try:
                outcome.fetched = session.fetch_all(results_dir)
            except Exception as exc:                     # noqa: BLE001 — a failed collect
                # must not look like a failed run; the result is still on the hub.
                self.failed.emit("fetch", f"{type(exc).__name__}: {exc}")
            self.fetched.emit(outcome)

    def _do_pull(self, names: List[str], results_dir: str) -> None:
        session = self._need_session()
        reply = session.pull(*names)
        published = list(reply.get("published") or [])
        self.log.emit(f"pulled {', '.join(published) or 'nothing'}")
        written = session.fetch_all(results_dir) if published else []
        outcome = RunOutcome(ok=True, state="pulled", fetched=written,
                             results_dir=results_dir)
        self.fetched.emit(outcome)

    def _need_session(self) -> Session:
        if self._session is None:
            raise LabLinkError("there is no open session — connect and open one first")
        return self._session


# ── the knob form ───────────────────────────────────────────────────────────────

class _KnobRow(QWidget):
    """One knob: a state selector, an editor, and the unit.

    The state selector exists because a knob has three states and a single editor can only
    express two. Blank-means-omit works right up until the session is reused: a warm worker
    treats an absent knob as unchanged, so "put this back to derived" needs to be sayable,
    and it is not the same as "leave it at the default".
    """

    changed = Signal()

    def __init__(self, knob: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.knob = dict(knob)
        self.name = str(knob.get("name") or "")
        ktype = str(knob.get("type") or "float")
        self._type = ktype

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        # Short words, fixed width. The three states need a control, but the sentence
        # explaining each one belongs in the tooltip and the hint: a selector wide enough
        # to read "recipe default: 5.0" leaves no room for the value it is qualifying, and
        # squeezes the knob's own name out of the form's label column.
        self._state = QComboBox()
        self._state.addItem("auto", KNOB_DEFAULT)
        if knob.get("unset_means") == "derive" or knob.get("derive"):
            self._state.addItem("derive", KNOB_DERIVE)
        self._state.addItem("set", KNOB_PINNED)
        self._state.setFixedWidth(72)
        self._state.setToolTip(
            f"auto — do not send this knob at all ({knob_state_label(knob)}).\n"
            "derive — send an explicit null, which is how you undo a value you set "
            "earlier in this same session.\n"
            "set — pin the value you type, including 0, which is a real zero and overrides "
            "the file's own calibration.")
        self._state.currentIndexChanged.connect(self._state_changed)
        row.addWidget(self._state)

        self.editor = self._build_editor(knob, ktype)
        row.addWidget(self.editor, 1)

        unit = str(knob.get("unit") or "")
        if unit and not isinstance(self.editor, (QSpinBox, QDoubleSpinBox)):
            row.addWidget(_dim(QLabel(_UNIT.get(unit, unit))))

        #: Only for what the disabled editor cannot show: that a value comes from the file
        #: rather than from the recipe, and why a greyed row is greyed. Deliberately narrow
        #: and non-wrapping — this dock is ~380 px wide, and a hint that wraps to two lines
        #: doubles the height of every one of a twelve-knob recipe's rows.
        self._hint = _dim(QLabel(""))
        self._hint.setWordWrap(False)
        self._hint.setMaximumWidth(110)
        row.addWidget(self._hint)

        self._sync_editor()

    def _build_editor(self, knob: Dict[str, Any], ktype: str) -> QWidget:
        unit = str(knob.get("unit") or "")
        suffix = f" {_UNIT.get(unit, unit)}" if unit else ""
        default = knob.get("default")
        if ktype == "bool":
            box = QComboBox()
            box.addItem("true", True)
            box.addItem("false", False)
            if isinstance(default, bool):
                box.setCurrentIndex(0 if default else 1)
            box.currentIndexChanged.connect(self.changed)
            return box
        if ktype == "enum":
            box = QComboBox()
            for choice in (knob.get("enum") or []):
                box.addItem(str(choice), choice)
            if default is not None:
                hit = box.findData(default)
                if hit >= 0:
                    box.setCurrentIndex(hit)
            box.currentIndexChanged.connect(self.changed)
            return box
        if ktype == "int":
            box = QSpinBox()
            # The recipe's OWN bounds, not a huge editor range: these are published so a
            # value outside them is caught before the upload rather than after it.
            box.setRange(int(knob.get("min", -2_000_000_000) or -2_000_000_000),
                         int(knob.get("max", 2_000_000_000) or 2_000_000_000))
            if suffix:
                box.setSuffix(suffix)
            if isinstance(default, (int, float)):
                box.setValue(int(default))
            box.valueChanged.connect(self.changed)
            return box
        if ktype in ("float",):
            box = QDoubleSpinBox()
            lo = knob.get("min")
            hi = knob.get("max")
            box.setRange(float(lo) if lo is not None else -1e12,
                         float(hi) if hi is not None else 1e12)
            box.setDecimals(_decimals_for(lo, hi, default))
            box.setSingleStep(_step_for(lo, hi, default))
            if suffix:
                box.setSuffix(suffix)
            if isinstance(default, (int, float)):
                box.setValue(float(default))
            box.valueChanged.connect(self.changed)
            return box
        # string / channel_list / anything new
        edit = QLineEdit()
        if ktype == "channel_list":
            edit.setPlaceholderText("channel indices, e.g. 0 or 0,2"
                                    + (f" (at most {knob['max_items']})"
                                       if knob.get("max_items") else ""))
        else:
            bits = []
            if knob.get("pattern"):
                bits.append(f"shape: {knob['pattern']}")
            if knob.get("max_len"):
                bits.append(f"at most {knob['max_len']} characters")
            edit.setPlaceholderText("  ".join(bits) or "text")
        if default is not None:
            edit.setText(str(default) if not isinstance(default, (list, tuple))
                         else ",".join(str(v) for v in default))
        edit.textEdited.connect(self.changed)
        return edit

    # ── state ───────────────────────────────────────────────────────────────────
    def _state_changed(self) -> None:
        self._sync_editor()
        self.changed.emit()

    def _sync_editor(self) -> None:
        """Enable the editor only when a value is actually being pinned.

        Follows the inspector's own convention for a metadata-derived param: the box is
        disabled and still shows the value that will be used, so "auto" is legible as a
        number rather than as an empty control. The hint therefore only has to carry what a
        number cannot say — that the value comes from the file, not the recipe.
        """
        state = self._state.currentData()
        pinned = state == KNOB_PINNED
        self.editor.setEnabled(pinned)
        if pinned:
            self._hint.setText("")
        elif state == KNOB_DERIVE or self.knob.get("unset_means") == "derive":
            self._hint.setText("from the file")
        elif self.knob.get("default") is None:
            self._hint.setText("recipe's own")
        else:
            self._hint.setText("")

    def set_applicable(self, applicable: bool, reason: str = "") -> None:
        """Grey the whole row out when the recipe says this knob is not read right now.

        Disabled rather than hidden: a control that vanishes leaves a person wondering what
        they did, while a greyed one with its reason attached explains itself. Pinning a
        value for an inapplicable knob is refused by the hub rather than ignored, which is
        exactly the refusal this prevents.
        """
        self.setEnabled(applicable)
        self.setToolTip("" if applicable else reason)
        if applicable:
            self._sync_editor()
        else:
            self._hint.setText(reason)

    # ── value ───────────────────────────────────────────────────────────────────
    def value(self) -> Tuple[str, Any]:
        """``(state, value)`` — the state says whether ``value`` should be sent at all."""
        state = str(self._state.currentData() or KNOB_DEFAULT)
        if state != KNOB_PINNED:
            return state, None
        widget = self.editor
        if isinstance(widget, QComboBox):
            return state, widget.currentData()
        if isinstance(widget, QSpinBox):
            return state, int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return state, float(widget.value())
        text = widget.text().strip()
        if self._type == "channel_list":
            return state, [int(p) for p in text.replace(" ", "").split(",") if p]
        return state, text

    def set_value(self, value: Any) -> None:
        """Show ``value``, choosing the state it implies. ``None`` means derived."""
        if value is None:
            index = self._state.findData(KNOB_DERIVE)
            self._state.setCurrentIndex(index if index >= 0
                                        else self._state.findData(KNOB_DEFAULT))
            self._sync_editor()
            return
        self._state.setCurrentIndex(self._state.findData(KNOB_PINNED))
        widget = self.editor
        if isinstance(widget, QComboBox):
            hit = widget.findData(value)
            if hit >= 0:
                widget.setCurrentIndex(hit)
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(value, (list, tuple)):
            widget.setText(",".join(str(v) for v in value))
        else:
            widget.setText(str(value))
        self._sync_editor()

    def clear_to_default(self) -> None:
        self._state.setCurrentIndex(self._state.findData(KNOB_DEFAULT))
        self._sync_editor()


class KnobForm(QWidget):
    """Every knob a recipe allows, as bounded controls that agree with the hub."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: Dict[str, _KnobRow] = {}
        self._declared: Dict[str, Dict[str, Any]] = {}
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Without this the field column takes what it likes and the knob NAMES get elided to
        # nothing — which on a 12-knob recipe leaves a column of controls nobody can identify.
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

    def set_recipe(self, meta: Dict[str, Any]) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._rows.clear()
        self._declared = HubClient.declared_knobs(dict(meta or {}))
        for name, knob in self._declared.items():
            row = _KnobRow(knob)
            row.changed.connect(self._on_changed)
            label = QLabel(str(knob.get("label") or name))
            help_text = str(knob.get("help") or "")
            if knob.get("unset_means") == "derive":
                help_text += ("\n\nLeave this alone unless you mean to override the "
                              "microscope: unset, the hub derives it from the file's own "
                              "calibration.")
            if help_text:
                label.setToolTip(help_text.strip())
                row.setToolTip(help_text.strip())
            self._form.addRow(label, row)
            self._rows[name] = row
        self._refresh_conditions()

    def _on_changed(self) -> None:
        self._refresh_conditions()
        self.changed.emit()

    def _refresh_conditions(self) -> None:
        """Re-evaluate every ``applies_when`` against what the form currently says."""
        current = self.values(include_defaults=True)
        for name, row in self._rows.items():
            spec = self._declared.get(name) or {}
            cond = spec.get("applies_when")
            if not cond:
                row.set_applicable(True)
                continue
            ok = knob_applies(spec, current, self._declared)
            controller = cond.get("knob")
            wanted = (repr(cond["equals"]) if "equals" in cond
                      else "one of " + ", ".join(map(repr, cond.get("in") or ())))
            row.set_applicable(
                ok, f"only read when {controller} is {wanted}" if not ok else "")

    # ── values ──────────────────────────────────────────────────────────────────
    def values(self, *, include_defaults: bool = False) -> Dict[str, Any]:
        """The knob set to send.

        ``include_defaults`` resolves each omitted knob to the recipe's own default, which
        is what an ``applies_when`` must be evaluated against — a condition on a knob nobody
        has touched still has an answer, because the controller has a declared default.
        """
        out: Dict[str, Any] = {}
        for name, row in self._rows.items():
            spec = self._declared.get(name) or {}
            state, value = row.value()
            if state == KNOB_PINNED:
                out[name] = value
            elif state == KNOB_DERIVE:
                out[name] = None
            elif include_defaults and "default" in spec:
                out[name] = spec.get("default")
        return out

    def set_values(self, knobs: Dict[str, Any]) -> None:
        """Show a saved set. Knobs the recipe no longer declares are ignored here — the
        caller has already been told about them by :func:`presets.apply_preset`."""
        for name, row in self._rows.items():
            if name in knobs:
                row.set_value(knobs[name])
            else:
                row.clear_to_default()
        self._refresh_conditions()

    def clear_values(self) -> None:
        for row in self._rows.values():
            row.clear_to_default()
        self._refresh_conditions()

    def validate(self, meta: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """``(knobs, problem)`` — the payload, or the first refusal to show a person."""
        knobs = self.values()
        try:
            return HubClient.resolve_for_send(dict(meta), knobs), ""
        except LabLinkError as exc:
            return {}, str(exc)


# ── the result card ─────────────────────────────────────────────────────────────

class ResultCard(QWidget):
    """What came back, rendered beside the knobs that produced it.

    Inline because judging a knob change means looking at the picture. A panel that only
    lists filenames makes the person leave to see anything, and the loop this whole feature
    exists to enable is exactly the one that dies when each attempt costs a trip to a file
    manager.
    """

    load_into_graph = Signal(str)     # a returned image path the editor can open
    pull_requested = Signal(str)      # an artifact name the recipe held back

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._headline = QLabel("no run yet")
        self._headline.setWordWrap(True)
        root.addWidget(self._headline)

        self._image = QLabel()
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumHeight(160)
        self._image.setFrameShape(QFrame.Shape.StyledPanel)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self._image.setVisible(False)
        root.addWidget(self._image, 1)
        self._image_path = ""

        self._numbers = _dim(QLabel(""))
        self._numbers.setWordWrap(True)
        self._numbers.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._numbers)

        self._table = QLabel("")
        self._table.setTextFormat(Qt.TextFormat.PlainText)
        self._table.setStyleSheet("font-family: Consolas, monospace;")
        self._table.setVisible(False)
        root.addWidget(self._table)

        buttons = QHBoxLayout()
        self._load = QPushButton("Load into graph")
        self._load.setToolTip(
            "Drop the returned image into this editor as a source node, so the next thing "
            "you do with it happens here.")
        self._load.clicked.connect(
            lambda: self._image_stack and self.load_into_graph.emit(self._image_stack))
        self._load.setEnabled(False)
        self._folder = QPushButton("Open folder")
        self._folder.clicked.connect(self._open_folder)
        self._folder.setEnabled(False)
        buttons.addWidget(self._load)
        buttons.addWidget(self._folder)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self._dir = ""
        self._image_stack = ""

    # ── filling it in ───────────────────────────────────────────────────────────
    def show_outcome(self, outcome: RunOutcome) -> None:
        """The headline for a finished command, before anything has been fetched."""
        if outcome.ok:
            head = f"run {outcome.cmd_seq} finished in {outcome.duration_s}s"
            note = outcome.speedup_note
            self._headline.setText(head + (f" — {note}" if note else ""))
        else:
            # A failed command does not end the session. "The threshold found no objects"
            # is an ordinary scientific result and the warm software is still there.
            self._headline.setText(
                f"run {outcome.cmd_seq} {outcome.state}: {outcome.message or outcome.code}"
                f"\nThe session is still warm — change a knob and run again.")

    def show_results(self, outcome: RunOutcome) -> None:
        """Render whatever was fetched into the results folder."""
        self._dir = outcome.results_dir or (
            os.path.dirname(outcome.fetched[0]) if outcome.fetched else "")
        self._folder.setEnabled(bool(self._dir) and os.path.isdir(self._dir))
        self._image_stack = ""
        self._load.setEnabled(False)

        quicklook = ""
        table = ""
        metrics = ""
        for path in outcome.fetched:
            low = path.lower()
            if low.endswith((".png", ".jpg", ".jpeg")) and "thumb" not in low:
                quicklook = quicklook or path
            elif low.endswith(".preview.csv"):
                table = path
            elif low.endswith(".csv") and not table:
                table = path
            elif low.endswith(".json") and "metrics" in low:
                metrics = path
            elif low.endswith((".tif", ".tiff", ".nd2")):
                self._image_stack = self._image_stack or path

        if quicklook:
            pix = QPixmap(quicklook)
            if not pix.isNull():
                self._image.setPixmap(pix.scaled(
                    self._image.width() or 320, 260,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                self._image.setVisible(True)
                self._image_path = quicklook
        if self._image_stack:
            self._load.setEnabled(True)
            self._load.setText(f"Load {os.path.basename(self._image_stack)} into graph")

        self._numbers.setText(_metrics_line(metrics) if metrics else "")
        if table:
            text = _table_preview(table)
            self._table.setText(text)
            self._table.setVisible(bool(text))
        else:
            self._table.setVisible(False)

    def show_held(self, names: List[str]) -> None:
        if names:
            self._headline.setText(
                self._headline.text()
                + f"\nheld on the hub until pulled: {', '.join(names)}")

    def clear(self) -> None:
        self._headline.setText("no run yet")
        self._image.setVisible(False)
        self._table.setVisible(False)
        self._numbers.setText("")
        self._load.setEnabled(False)
        self._folder.setEnabled(False)

    def _open_folder(self) -> None:
        if self._dir and os.path.isdir(self._dir):
            os.startfile(self._dir)              # noqa: S606 — Windows shell open


# ── small helpers ───────────────────────────────────────────────────────────────

_UNIT = {"um": "µm", "um2": "µm²", "um3": "µm³",
         "um_axial": "µm", "nm": "nm", "s": "s", "px": "px"}


def _dim(label: QLabel) -> QLabel:
    label.setStyleSheet(f"color: {T.MUTED};")
    return label


def _bytes(n: Optional[int]) -> str:
    if not n:
        return "0 B"
    step = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if step < 1024 or unit == "TB":
            return f"{step:.0f} {unit}" if unit == "B" else f"{step:.1f} {unit}"
        step /= 1024
    return f"{n} B"


def _decimals_for(lo: Any, hi: Any, default: Any) -> int:
    """Enough decimals that the smallest meaningful step is typeable.

    Driven by the recipe's own range rather than a constant: a knob bounded 0..1 needs three
    decimals to be usable and one bounded 0..1e6 needs none, and a spin box that rounds a
    value away is indistinguishable from one that ignored it.
    """
    span = None
    try:
        if lo is not None and hi is not None:
            span = abs(float(hi) - float(lo))
    except (TypeError, ValueError):
        span = None
    if span is None:
        span = abs(float(default)) if isinstance(default, (int, float)) else 100.0
    if span <= 1:
        return 4
    if span <= 100:
        return 3
    if span <= 10_000:
        return 2
    return 1


def _step_for(lo: Any, hi: Any, default: Any) -> float:
    """One arrow-click, ten times the smallest typeable increment.

    A step equal to the last decimal makes a spin box useless for coarse work (200 clicks to
    cross a µm), and one much larger makes it useless for fine work.
    """
    return 10.0 ** -(_decimals_for(lo, hi, default) - 1)


def _metrics_line(path: str) -> str:
    """The numbers worth reading at a glance out of a returned metrics blob."""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return ""
    if not isinstance(doc, dict):
        return ""
    bits: List[str] = []
    for key in ("objects", "nodes_computed", "nodes_cached", "seconds"):
        if doc.get(key) is not None:
            bits.append(f"{key.replace('nodes_', '')}: {doc[key]}")
    columns = doc.get("columns")
    if isinstance(columns, dict):
        for name, stats in list(columns.items())[:4]:
            if isinstance(stats, dict) and stats.get("mean") is not None:
                bits.append(f"{name} mean {stats['mean']:.4g}")
    return "   ".join(bits)


def _table_preview(path: str) -> str:
    """A returned table's first rows as fixed-width text."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))[:PREVIEW_ROWS + 1]
    except (OSError, ValueError):
        return ""
    if not rows:
        return ""
    width = min(14, max((len(c) for row in rows for c in row), default=8))
    lines = []
    for row in rows[:PREVIEW_ROWS + 1]:
        lines.append("  ".join(str(c)[:width].ljust(width) for c in row[:6]))
    return "\n".join(lines)


__all__ = [
    "SessionController", "RunOutcome", "KnobForm", "ResultCard",
    "KNOB_DEFAULT", "KNOB_DERIVE", "KNOB_PINNED", "knob_state_label",
]
