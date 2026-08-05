"""The LabLink dock — is this machine serving the lab, and can it send work out.

Two tabs, because the mode has two halves and an operator is only ever asking one of the
two questions:

**Serving** — the worker side. A readiness check that spawns the worker *the way a hub
does* and reads its handshake, plus the live state of a hub running on this machine, read
from LabLink's own read-only console API. The panel cannot see inside a hub-spawned worker
— those are separate processes owned by the hub, in a different process tree — so it does
not pretend to: it reads the one authoritative source there is and says so.

**Sending work** — the client side. Point at somebody else's hub, browse the recipes its
operator curated, turn the knobs they whitelisted, run, and pull the results back.

**Nothing here touches the network on the GUI thread.** Every call goes through
:class:`_Task`, a one-shot ``QThread``. A blocking ``urllib`` call on the UI thread freezes
the whole editor for the socket timeout — 40 s, or the length of a long poll — and this
panel's job is to make a *remote* machine's state legible, which means it is talking to
something slow and occasionally absent by definition.

Qt. The only module in :mod:`nodelab_v2.lablink` that imports PySide6, deliberately: the
worker must stay importable on a box with no display.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from nodelab_v2 import theme as T
from nodelab_v2.lablink import protocol as P
from nodelab_v2.lablink.client import HubClient, LabLinkError

#: How often the Serving tab re-reads a local hub's console state.
POLL_MS = 3000

#: Console read timeout. Short: it is loopback, and a hub that is not there must be
#: reported as absent within one poll rather than stalling the next three.
CONSOLE_TIMEOUT_S = 2.0

#: How long shutdown waits for one in-flight hub call. Comfortably over
#: :data:`CONSOLE_TIMEOUT_S` so a poll in flight always finishes, and well under anything a
#: user would call a hang if a slower call (an upload) is still running.
WAIT_MS = 3000


class _Task(QThread):
    """Run one callable off the GUI thread; deliver its value or its error by signal.

    Both signals carry a plain object rather than a live handle, and the thread keeps a
    reference to itself in the caller's list until it finishes — a ``QThread`` collected
    while running takes the process with it, and it is the classic way a panel like this
    crashes an app on shutdown.
    """

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:                              # noqa: D102 — QThread entry point
        try:
            self.done.emit(self._fn())
        except Exception as exc:                        # noqa: BLE001 — report, never crash
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _TaskHost(QWidget):
    """Mixin-ish base that owns its running threads so none is collected mid-flight."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._tasks: List[_Task] = []

    def spawn(self, fn: Callable[[], Any], on_done: Callable[[Any], None],
              on_fail: Optional[Callable[[str], None]] = None) -> _Task:
        task = _Task(fn, self)
        self._tasks.append(task)

        def cleanup() -> None:
            if task in self._tasks:
                self._tasks.remove(task)

        task.done.connect(on_done)
        if on_fail is not None:
            task.failed.connect(on_fail)
        task.finished.connect(cleanup)
        task.start()
        return task

    def stop_tasks(self) -> None:
        """Wait for every running task's thread. Called from the dock's close path.

        This WAITS; it does not interrupt. ``requestInterruption`` is only a flag, and the
        callables here are blocking ``urllib`` reads with no place to check one — so setting
        it would look like cancellation while changing nothing. The honest guarantee is
        bounded: each task gets up to ``WAIT_MS`` to finish its socket read (which has its
        own short timeout), and a straggler is abandoned rather than allowed to hold the
        window open. Abandoning is safe because the thread only emits signals, and Qt drops
        those once the receiver is gone.
        """
        for task in list(self._tasks):
            if not task.wait(WAIT_MS):
                # Not fatal, and worth a line: it means a hub call outlived the window.
                print(f"[lablink] a background hub call was still running at shutdown "
                      f"after {WAIT_MS} ms; abandoning it")


def _panel_qss() -> str:
    """The dock's stylesheet, built from the live theme tokens.

    Written out rather than left to Qt's defaults because the app runs a dark palette and
    an unstyled ``QPlainTextEdit``/``QTableWidget`` renders **white** inside it — which is
    not a small blemish on a panel whose whole job is to be readable at a glance. Rebuilt on
    every :meth:`LabLinkPanel.restyle`, so the light theme is styled by the same code rather
    than by a second set of colours that can drift.
    """
    return f"""
        QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
        QGroupBox {{ border:1px solid {T.BORDER.name()}; border-radius:8px;
            margin-top:9px; padding:8px 6px 6px 6px; font-weight:700;
            color:{T.INK_2.name()}; }}
        QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top left;
            left:9px; padding:0 4px; background:{T.PANEL.name()};
            color:{T.MUTED.name()}; }}
        QPlainTextEdit {{ background:{T.BG.name()}; color:{T.INK.name()};
            border:1px solid {T.BORDER.name()}; border-radius:8px;
            font-family:{T.MONO}; font-size:11px; selection-background-color:
            {T.ACCENT_DIM.name()}; selection-color:{T.INK.name()}; }}
        QTableWidget {{ background:{T.BG.name()}; color:{T.INK.name()};
            gridline-color:{T.BORDER.name()}; border:1px solid {T.BORDER.name()};
            border-radius:8px; font-family:{T.MONO}; outline:0; }}
        QHeaderView::section {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
            border:0; border-right:1px solid {T.BORDER.name()};
            border-bottom:1px solid {T.BORDER.name()}; padding:4px 7px;
            font-weight:600; }}
        QTableWidget::item {{ padding:2px 4px; }}
        QTableWidget::item:selected {{ background:{T.ACCENT_DIM.name()};
            color:{T.INK.name()}; }}
        QTableCornerButton::section {{ background:{T.BODY.name()}; border:0; }}
        QProgressBar {{ background:{T.PROG_TRACK.name()};
            border:1px solid {T.BORDER.name()}; border-radius:6px; height:8px;
            text-align:center; color:{T.INK_2.name()}; }}
        QProgressBar::chunk {{ background:{T.ACCENT.name()}; border-radius:5px; }}
        QLabel[role="muted"] {{ color:{T.MUTED.name()}; }}
    """ + T.controls_qss()


def _muted(label: QLabel) -> QLabel:
    """Tag a label as secondary text. Styled by the sheet, so it follows the theme —
    a hard-coded colour here would survive a palette switch and stop matching."""
    label.setProperty("role", "muted")
    return label


def _human_bytes(n: Any) -> str:
    try:
        value = float(n)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


# ── serving ─────────────────────────────────────────────────────────────────────

class ServePanel(_TaskHost):
    """Can this machine serve LabLink, and what is a local hub doing right now."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._console_port = P.DEFAULT_CONSOLE_PORT
        self._polling = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── readiness ───────────────────────────────────────────────────────────
        ready = QGroupBox("This machine as a worker")
        rl = QVBoxLayout(ready)
        row = QHBoxLayout()
        self._check_btn = QPushButton("Run readiness check")
        self._check_btn.setToolTip(
            "Spawns the worker exactly as a hub does — a subprocess, LWP/1 on stdin — and "
            "reads its handshake. This is the check that catches a missing dependency "
            "BEFORE a node's first call, and it is what `lablink hub doctor --probe` does "
            "from the hub side.")
        self._check_btn.clicked.connect(self._run_check)
        row.addWidget(self._check_btn)
        row.addStretch(1)
        rl.addLayout(row)
        self._ready_out = QPlainTextEdit()
        self._ready_out.setReadOnly(True)
        self._ready_out.setMaximumHeight(150)
        self._ready_out.setPlainText("Not checked yet.")
        rl.addWidget(self._ready_out)
        root.addWidget(ready)

        # ── local hub ───────────────────────────────────────────────────────────
        hub = QGroupBox("A LabLink hub on this machine")
        hl = QVBoxLayout(hub)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Console port"))
        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(P.DEFAULT_CONSOLE_PORT)
        self._port.setToolTip(
            "The hub's read-only console listener (its own port, loopback only — never the "
            "data port). Only a hub on THIS machine is visible: the console does not "
            "listen off-box, by design.")
        prow.addWidget(self._port)
        self._watch = QCheckBox("Watch")
        self._watch.setChecked(True)
        self._watch.toggled.connect(self._set_polling)
        prow.addWidget(self._watch)
        prow.addStretch(1)
        self._hub_state = _muted(QLabel("—"))
        prow.addWidget(self._hub_state)
        hl.addLayout(prow)

        self._sessions = QTableWidget(0, 6)
        self._sessions.setHorizontalHeaderLabels(
            ["session", "recipe", "state", "node", "age", "worker"])
        self._sessions.verticalHeader().setVisible(False)
        self._sessions.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._sessions.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        hl.addWidget(self._sessions)
        root.addWidget(hub, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._set_polling(True)

    # ── readiness ───────────────────────────────────────────────────────────────
    def _run_check(self) -> None:
        self._check_btn.setEnabled(False)
        self._ready_out.setPlainText("spawning the worker…")
        self.spawn(_probe_worker, self._show_check, self._show_check_error)

    def _show_check(self, info: Dict[str, Any]) -> None:
        self._check_btn.setEnabled(True)
        if not info.get("ok"):
            self._ready_out.setPlainText(
                f"NOT READY — {info.get('error')}\n\n{info.get('stderr') or ''}")
            return
        hello = info.get("hello") or {}
        optional = hello.get("optional") or {}
        missing = sorted(k for k, v in optional.items() if not (v or {}).get("ok"))
        have = sorted(k for k, v in optional.items() if (v or {}).get("ok"))
        lines = [
            f"READY in {info.get('seconds')}s",
            f"  worker    {hello.get('worker')} {hello.get('worker_version')} "
            f"(LWP {hello.get('protocol')})",
            f"  software  {hello.get('software')} {hello.get('software_version')}, "
            f"graph format {hello.get('graph_format')}",
            f"  installed {', '.join(have) or '(none)'}",
            f"  MISSING   {', '.join(missing) or '(none)'}",
            "",
            "A recipe naming a missing capability in requires.capabilities is refused at",
            "session start, with this worker's own reason — not silently run without it.",
        ]
        self._ready_out.setPlainText("\n".join(lines))

    def _show_check_error(self, message: str) -> None:
        self._check_btn.setEnabled(True)
        self._ready_out.setPlainText(f"the readiness check itself failed: {message}")

    # ── local hub polling ───────────────────────────────────────────────────────
    def _set_polling(self, on: bool) -> None:
        self._polling = bool(on)
        if on:
            self._timer.start()
            self._poll()
        else:
            self._timer.stop()
            self._hub_state.setText("not watching")

    def _poll(self) -> None:
        if not self._polling:
            return
        port = int(self._port.value())
        self.spawn(lambda: _console_state(port), self._show_state, self._show_state_error)

    def _show_state(self, state: Dict[str, Any]) -> None:
        hub = state.get("hub") or {}
        wfs = state.get("workflows") or []
        mine = [w for w in wfs if str(w.get("name")) == P.WORKER_NAME]
        tag = (f"up {hub.get('uptime_s', 0):.0f}s · "
               f"{hub.get('sessions_open')}/{hub.get('sessions_limit')} session(s) · "
               f"data port {hub.get('data_port')} · "
               f"{_human_bytes(hub.get('disk_free_bytes'))} free")
        if not mine:
            tag += "  ⚠ no 'nd2studios' workflow configured"
        self._hub_state.setText(tag)

        rows = list(state.get("sessions") or [])
        recent = [r for r in (state.get("recent") or [])][:8]
        self._fill(rows, recent, float(state.get("now") or time.time()))

    def _show_state_error(self, message: str) -> None:
        self._hub_state.setText("no hub answering on this machine")
        self._hub_state.setToolTip(
            f"{message}\n\nThat is the normal state on an analysis PC that only SENDS work. "
            f"To serve the lab from here, run a hub:\n"
            f"  python -m lablink hub serve --config hub.json --root lablink_data")
        self._sessions.setRowCount(0)

    def _fill(self, live: List[dict], recent: List[dict], now: float) -> None:
        rows = [(r, True) for r in live] + [(r, False) for r in recent]
        self._sessions.setRowCount(len(rows))
        for i, (rec, is_live) in enumerate(rows):
            worker = rec.get("worker") or {}
            age = now - float(rec.get("opened") or now)
            cells = [
                str(rec.get("id") or "")[:12],
                str(rec.get("recipe") or ""),
                str(rec.get("state") or ""),
                str(rec.get("node") or rec.get("label") or "—"),
                f"{age:.0f}s" if is_live else f"{float(rec.get('seconds') or 0):.1f}s",
                (f"pid {worker.get('pid')}" if worker.get("pid") else
                 str(rec.get("reason") or "—")),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if not is_live:
                    # closed/finished rows greyed so the live ones read first. Set on the
                    # ITEM rather than by stylesheet — a per-row colour has no selector.
                    item.setForeground(T.MUTED)
                self._sessions.setItem(i, col, item)


def _probe_worker() -> Dict[str, Any]:
    """Spawn the worker as a hub would and read its handshake. Runs off the GUI thread.

    Deliberately a SUBPROCESS rather than an in-process import: the thing being verified is
    that a hub's ``command`` works from this checkout with this interpreter, and an
    in-process check would pass on a machine where the spawn does not.
    """
    import subprocess
    import sys
    import tempfile

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    argv = [sys.executable, "-B", "-m", "nodelab_v2.lablink.worker",
            "--session", tempfile.mkdtemp(prefix="lablink-probe-"), "--print-hello"]
    started = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                              timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "the worker did not hand shake within 300s"}
    seconds = round(time.monotonic() - started, 2)
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue                    # an import banner on stdout; the hub tolerates it
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("ev") == "hello":
            return {"ok": True, "hello": msg, "seconds": seconds,
                    "stderr": proc.stderr or ""}
    return {"ok": False, "seconds": seconds,
            "error": f"the worker exited {proc.returncode} without a handshake",
            "stderr": (proc.stderr or "")[-4000:]}


def _console_state(port: int) -> Dict[str, Any]:
    """One poll of a local hub's console API. Raises if nothing is listening."""
    url = f"http://127.0.0.1:{int(port)}{P.CONSOLE_STATE_PATH}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=CONSOLE_TIMEOUT_S) as resp:
        return json.loads(resp.read() or b"{}")


# ── sending work ────────────────────────────────────────────────────────────────

class SendPanel(_TaskHost):
    """Send this machine's work to somebody else's hub."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._hub: Optional[HubClient] = None
        self._recipes: Dict[str, Dict[str, Any]] = {}
        self._knob_widgets: Dict[str, QWidget] = {}
        self._input_path = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        conn = QGroupBox("Hub")
        cl = QGridLayout(conn)
        self._url = QLineEdit(f"http://10.132.157.104:{P.DEFAULT_PORT}")
        self._url.setPlaceholderText("http://<hub>:8765")
        self._token = QLineEdit()
        self._token.setEchoMode(QLineEdit.EchoMode.Password)
        self._token.setPlaceholderText("the site token, or this node's own once enrolled")
        self._node = QLineEdit()
        self._node.setPlaceholderText("optional: this machine's enrolled node id")
        self._node.setToolTip(
            "Only once ENROLLED. Sending a node id the hub does not know is a 401, not a "
            "quiet fallback to the site token.")
        self._connect = QPushButton("Connect")
        self._connect.clicked.connect(self._do_connect)
        cl.addWidget(QLabel("URL"), 0, 0)
        cl.addWidget(self._url, 0, 1)
        cl.addWidget(QLabel("Token"), 1, 0)
        cl.addWidget(self._token, 1, 1)
        cl.addWidget(QLabel("Node id"), 2, 0)
        cl.addWidget(self._node, 2, 1)
        cl.addWidget(self._connect, 0, 2, 3, 1)
        root.addWidget(conn)

        pick = QGroupBox("Recipe")
        pl = QFormLayout(pick)
        self._recipe = QComboBox()
        self._recipe.currentTextChanged.connect(self._recipe_changed)
        pl.addRow("Workflow / recipe", self._recipe)
        self._recipe_note = _muted(QLabel("connect to a hub to see what its operator "
                                          "offers"))
        self._recipe_note.setWordWrap(True)
        pl.addRow(self._recipe_note)
        root.addWidget(pick)

        self._knobs_box = QGroupBox("Knobs the recipe allows")
        self._knobs_form = QFormLayout(self._knobs_box)
        root.addWidget(self._knobs_box)

        send = QGroupBox("Data and run")
        sl = QGridLayout(send)
        self._file = QLineEdit()
        self._file.setPlaceholderText("optional — leave empty for the recipe's own source")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._dest = QLineEdit(os.path.join(os.path.expanduser("~"), "lablink-results"))
        self._run = QPushButton("Run on the hub")
        self._run.clicked.connect(self._do_run)
        self._run.setEnabled(False)
        sl.addWidget(QLabel("Send file"), 0, 0)
        sl.addWidget(self._file, 0, 1)
        sl.addWidget(browse, 0, 2)
        sl.addWidget(QLabel("Results into"), 1, 0)
        sl.addWidget(self._dest, 1, 1, 1, 2)
        sl.addWidget(self._run, 2, 1, 1, 2)
        root.addWidget(send)

        self._bar = QProgressBar()
        self._bar.setVisible(False)
        root.addWidget(self._bar)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        root.addWidget(self._log, 1)

    # ── connect ─────────────────────────────────────────────────────────────────
    def _do_connect(self) -> None:
        url = self._url.text().strip()
        token = self._token.text()
        if not url or not token:
            self._say("a hub URL and a token are both needed.")
            return
        self._connect.setEnabled(False)
        self._say(f"connecting to {url} …")
        hub = HubClient(url, token, node_id=self._node.text().strip())

        def work() -> Dict[str, Any]:
            hello = hub.hello()
            if not hub.supports_sessions():
                raise LabLinkError(
                    "this is a LabLink file exchange, not a hub: its /hello reports no "
                    "session capability, so there is nothing to run work on.")
            return {"hello": hello, "workflows": hub.workflows()}

        self.spawn(work, lambda info: self._connected(hub, info), self._connect_failed)

    def _connected(self, hub: HubClient, info: Dict[str, Any]) -> None:
        self._connect.setEnabled(True)
        self._hub = hub
        hello = info["hello"]
        caps = hello.get("capabilities") or {}
        self._say(f"connected: {hello.get('name')} running lablink "
                  f"{hello.get('version')}, {caps.get('sessions_open')}/"
                  f"{caps.get('sessions_limit')} session(s) in use, "
                  f"max file {_human_bytes(hello.get('max_file_bytes'))}")
        self._recipes.clear()
        self._recipe.clear()
        broken = []
        for wf in info["workflows"]:
            for rec in wf.get("recipes") or []:
                key = f"{wf['name']} / {rec['name']}"
                self._recipes[key] = {"workflow": wf["name"], **rec}
                self._recipe.addItem(key)
            for bad in wf.get("unusable_recipes") or []:
                if isinstance(bad, dict):
                    broken.append(f"{wf['name']}/{bad.get('name')}: "
                                  f"{bad.get('reason') or 'failed validation'}")
        if broken:
            # Surfaced rather than hidden: "the recipe I was told to use is not in the
            # list" is otherwise unanswerable from this side.
            self._say("recipes the hub's operator installed that FAILED validation:")
            for line in broken:
                self._say(f"    {line}")
        if not self._recipes:
            self._say("this hub offers no usable recipes.")
        self._run.setEnabled(bool(self._recipes))

    def _connect_failed(self, message: str) -> None:
        self._connect.setEnabled(True)
        self._say(f"could not connect: {message}")

    # ── recipe + knobs ──────────────────────────────────────────────────────────
    def _recipe_changed(self, key: str) -> None:
        while self._knobs_form.rowCount():
            self._knobs_form.removeRow(0)
        self._knob_widgets.clear()
        meta = self._recipes.get(key)
        if not meta:
            return
        runtime = meta.get("expected_runtime_s") or {}
        self._recipe_note.setText(
            f"{meta.get('title') or meta.get('name')} — target {meta.get('target')!r}, "
            f"{meta.get('node_count')} node(s), typically "
            f"{runtime.get('typical', '?')}s (worst {runtime.get('worst', '?')}s). "
            f"{meta.get('description') or ''}")
        for knob in meta.get("knobs") or []:
            self._add_knob(knob)

    def _add_knob(self, knob: Dict[str, Any]) -> None:
        name = str(knob.get("name"))
        ktype = str(knob.get("type") or "float")
        unit = str(knob.get("unit") or "")
        label = str(knob.get("label") or name) + (f"  [{unit}]" if unit else "")
        widget: QWidget
        # The first entry of every dropdown is the DO-NOT-SEND one, and it is labelled
        # rather than left blank: an empty combo box reads as a broken control, when what
        # it actually means is "the operator already chose, leave it alone". Its userData is
        # empty, which is what _collect_knobs keys the omission on — so the label can say
        # anything without changing what goes on the wire.
        if ktype == "enum" and knob.get("enum"):
            widget = QComboBox()
            widget.addItem(_leave_alone(knob), "")
            for choice in knob["enum"]:
                widget.addItem(str(choice), str(choice))
        elif ktype == "bool":
            widget = QComboBox()
            widget.addItem(_leave_alone(knob), "")
            widget.addItem("true", "true")
            widget.addItem("false", "false")
        elif ktype == "int":
            widget = QLineEdit()
            widget.setPlaceholderText(_knob_hint(knob))
        elif ktype in ("string", "channel_list"):
            widget = QLineEdit()
            widget.setPlaceholderText(_knob_hint(knob))
        else:
            widget = QLineEdit()
            widget.setPlaceholderText(_knob_hint(knob))
        tip = str(knob.get("help") or "")
        if knob.get("derive") or knob.get("unset_means") == "derive":
            tip += ("\n\nLEAVE THIS EMPTY unless you mean to override the microscope: "
                    "unset, the hub derives it from your file's own calibration. Pinning "
                    "it silently overrides that.")
        widget.setToolTip(tip.strip())
        self._knobs_form.addRow(label, widget)
        self._knob_widgets[name] = widget

    def _collect_knobs(self) -> Dict[str, Any]:
        """Read the widgets into a knob dict, omitting everything left blank.

        Blank means *absent*, never zero and never the default: a knob the recipe marks
        ``unset_means: "derive"`` must be left out so the hub derives it, and a knob with a
        declared default is already chosen by the operator.
        """
        meta = self._recipes.get(self._recipe.currentText()) or {}
        specs = {str(k.get("name")): k for k in (meta.get("knobs") or [])}
        out: Dict[str, Any] = {}
        for name, widget in self._knob_widgets.items():
            spec = specs.get(name) or {}
            ktype = str(spec.get("type") or "float")
            if isinstance(widget, QComboBox):
                # userData, not the visible text: the "leave alone" row carries a label.
                text = str(widget.currentData() or "").strip()
            else:
                text = widget.text().strip()
            if not text:
                continue
            if ktype == "bool":
                out[name] = text == "true"
            elif ktype == "int":
                out[name] = int(text)
            elif ktype == "float":
                out[name] = float(text)
            elif ktype == "channel_list":
                out[name] = [int(p) for p in text.replace(" ", "").split(",") if p]
            else:
                out[name] = text
        return out

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Send a file to the hub", "",
            "Images (*.nd2 *.tif *.tiff);;All files (*)")
        if path:
            self._file.setText(path)

    # ── run ─────────────────────────────────────────────────────────────────────
    def _do_run(self) -> None:
        hub = self._hub
        meta = self._recipes.get(self._recipe.currentText())
        if hub is None or not meta:
            return
        try:
            knobs = self._collect_knobs()
        except ValueError as exc:
            self._say(f"a knob value is not a number: {exc}")
            return
        path = self._file.text().strip()
        dest = self._dest.text().strip() or "."
        workflow, recipe = meta["workflow"], meta["name"]
        self._run.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, 0)                       # indeterminate until told otherwise
        self._say(f"opening a session for {workflow}/{recipe} …")

        # Progress arrives on the task thread; hop it onto the GUI thread by signal rather
        # than touching widgets from there.
        def work() -> Dict[str, Any]:
            lines: List[str] = []
            with hub.open_session(workflow, recipe, label="ND2Studios editor",
                                 knobs=knobs) as session:
                lines.append(f"session {session.id} ready "
                             f"(quota {session.quota_mb} MB)")
                refs = []
                if path:
                    ref = session.send_data(path)
                    lines.append(f"sent {ref.name!r} ({_human_bytes(ref.size)}"
                                 + (", name repaired for the exchange" if ref.repaired
                                    else "") + ")")
                    refs.append(ref)
                result = session.run(inputs=refs, check=False,
                                     on_progress=lambda p: lines.append("  " + p.text))
                lines.append(f"{result.state} in {result.duration_s}s — "
                             f"{result.progress.computed} computed, "
                             f"{result.cached_steps} cached")
                if not result.ok:
                    lines.append(f"the hub reported: {result.code} — {result.message}")
                    lines.append("the session stayed usable; change a knob and run again.")
                if result.held():
                    session.pull()
                    lines.append(f"pulled held artifact(s): "
                                 f"{', '.join(result.held())}")
                got = session.fetch_all(dest) if (result.ok or result.artifacts) else []
                for written in got:
                    lines.append(f"got {os.path.basename(written)} "
                                 f"({_human_bytes(os.path.getsize(written))}, verified)")
            return {"lines": lines, "state": result.state}

        self.spawn(work, self._ran, self._run_failed)

    def _ran(self, info: Dict[str, Any]) -> None:
        self._run.setEnabled(True)
        self._bar.setVisible(False)
        for line in info.get("lines") or []:
            self._say(line)
        self._say(f"— finished ({info.get('state')}) —")

    def _run_failed(self, message: str) -> None:
        self._run.setEnabled(True)
        self._bar.setVisible(False)
        self._say(f"the run could not complete: {message}")

    # ── log ─────────────────────────────────────────────────────────────────────
    def _say(self, text: str) -> None:
        self._log.appendPlainText(text)


def _leave_alone(knob: Dict[str, Any]) -> str:
    """The label on a dropdown's do-not-send row, naming what will happen instead."""
    if knob.get("unset_means") == "derive" or knob.get("derive"):
        return "— derived from the file —"
    default = knob.get("default")
    if default is not None:
        return f"— recipe default: {default} —"
    return "— leave to the recipe —"


def _knob_hint(knob: Dict[str, Any]) -> str:
    if knob.get("unset_means") == "derive" or knob.get("derive"):
        return "leave empty — derived from the file"
    default = knob.get("default")
    bits = []
    if knob.get("min") is not None or knob.get("max") is not None:
        bits.append(f"{knob.get('min', '−∞')} … {knob.get('max', '∞')}")
    if default is not None:
        bits.append(f"default {default}")
    return "  ".join(bits) or "optional"


# ── the dock ────────────────────────────────────────────────────────────────────

class LabLinkPanel(QTabWidget):
    """The dock's contents: Serving, and Sending work."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.serve = ServePanel()
        self.send = SendPanel()
        self.addTab(self.serve, "Serving")
        self.addTab(self.send, "Send work")
        self.setDocumentMode(True)
        self.restyle()

    def restyle(self) -> None:
        """Re-apply the palette. The name and shape match every other panel here, so the
        window's ``set_theme`` can call it in the same loop as the rest."""
        sheet = _panel_qss()
        for widget in (self, self.serve, self.send):
            widget.setStyleSheet(sheet)
        # A property-based selector does not re-evaluate on its own; nudge the widgets
        # whose colour comes from `role="muted"`.
        for label in (self.serve._hub_state, self.send._recipe_note):
            label.style().unpolish(label)
            label.style().polish(label)

    def shutdown(self) -> None:
        """Stop polling and join every worker thread. Called from the window's close path.

        Not optional: a ``QThread`` still running when the interpreter tears down the Qt
        object it parents is a crash on exit, and this panel keeps a poll timer plus one
        thread per in-flight request.
        """
        self.serve._set_polling(False)
        self.serve.stop_tasks()
        self.send.stop_tasks()


__all__ = ["LabLinkPanel", "ServePanel", "SendPanel"]
