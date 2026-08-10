"""The LabLink dock — is this machine serving the lab, and can it send work out.

Two tabs, because the mode has two halves and an operator is only ever asking one of the
two questions:

**Serving** — the worker side. A readiness check that spawns the worker *the way a hub
does* and reads its handshake, plus the live state of a hub running on this machine, read
from LabLink's own read-only console API. The panel cannot see inside a hub-spawned worker
— those are separate processes owned by the hub, in a different process tree — so it does
not pretend to: it reads the one authoritative source there is and says so.

**Sending work** — the client side. Point at somebody else's hub, browse the recipes its
operator curated, turn the knobs they whitelisted, and **keep the session open while you
tune**: the hub holds the worker and its cache warm, so the second run after a knob change
costs milliseconds instead of the whole pipeline. Results render here rather than only landing
on disk, because a knob change that has to be judged somewhere else does not get judged. The
machinery for that lives in :mod:`nodelab_v2.lablink.tuning`; this module is the layout and
the wiring.

**Nothing here touches the network on the GUI thread.** Every call goes through
:class:`_Task`, a one-shot ``QThread``. A blocking ``urllib`` call on the UI thread freezes
the whole editor for the socket timeout — 40 s, or the length of a long poll — and this
panel's job is to make a *remote* machine's state legible, which means it is talking to
something slow and occasionally absent by definition.

Qt — one of the three modules here that import PySide6, with
:mod:`~nodelab_v2.lablink.tuning` and :mod:`~nodelab_v2.lablink.authoring`. Everything else
in the package stays Qt-free deliberately: the worker has to be importable on a hub with no
display.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from nodelab_v2 import theme as T
from nodelab_v2.lablink import presets as PR
from nodelab_v2.lablink import protocol as P
from nodelab_v2.lablink import sidecar as SC
from nodelab_v2.lablink import tuning as TUNE
from nodelab_v2.lablink.client import HubClient, LabLinkError, repair_name

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
        if not self.isVisible():
            # The dock is closed or hidden: nobody can read the answer, and the probe is
            # not free — against a hub that is not answering, the socket connect was
            # measured at 0.9–1.9 s per attempt, every POLL_MS, in every session that
            # never opened this panel (2026-08-10). The timer keeps ticking, so opening
            # the dock resumes watching within one interval, with nothing to re-arm.
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
    """Send this machine's work to somebody else's hub, and tune it against a warm session.

    The shape of this tab is the shape of the workflow it exists for: connect once, choose a
    recipe, upload once, then **run / look / adjust / run again** against a session that
    stays open. The hub keeps the worker and its cache warm between commands, so the second
    run after a knob change recomputes only what that knob invalidated — a run that costs a
    second cold comes back in milliseconds. A panel that opened a fresh session per attempt
    would throw that away and pay the cold cost every time, which is what this one used to
    do.
    """

    #: A returned image the editor should open as a source node. The panel only *asks*; the
    #: window owns the document and does it — the same rule every other panel here follows.
    load_into_graph = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._hub: Optional[HubClient] = None
        self._recipes: Dict[str, Dict[str, Any]] = {}
        self._presets = PR.PresetStore()
        self._session = TUNE.SessionController(self)
        self._last: Optional[TUNE.RunOutcome] = None
        self._last_record: Optional[PR.RunRecord] = None
        self._sidecar_dir = ""
        #: What has actually been uploaded to the live session, so a changed file is re-sent.
        self._sent_path = ""
        self._settings = _load_settings()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_hub_box())
        root.addWidget(self._build_recipe_box())
        root.addWidget(self._build_data_box())

        self._knobs_box = QGroupBox("Knobs the recipe allows")
        kb = QVBoxLayout(self._knobs_box)
        kb.setContentsMargins(8, 8, 8, 8)
        self._knobs = TUNE.KnobForm()
        kb.addWidget(self._knobs)
        root.addWidget(self._knobs_box)

        root.addWidget(self._build_run_box())

        self._card = TUNE.ResultCard()
        self._card.load_into_graph.connect(self.load_into_graph)
        root.addWidget(self._card, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        root.addWidget(self._log)

        self._wire_session()
        self._sync_buttons()

    # ── construction ────────────────────────────────────────────────────────────
    def _build_hub_box(self) -> QWidget:
        conn = QGroupBox("Hub")
        cl = QGridLayout(conn)
        self._url = QLineEdit(self._settings.get("url")
                              or f"http://localhost:{P.DEFAULT_PORT}")
        self._url.setPlaceholderText("http://<hub>:8765")
        self._token = QLineEdit()
        self._token.setEchoMode(QLineEdit.EchoMode.Password)
        self._token.setPlaceholderText("the site token, or this node's own once enrolled")
        self._token.setToolTip(
            "Not saved to disk. The token is anti-misdirection rather than security — it "
            "stops you sending into the wrong machine on a shared subnet — but writing it "
            "into a settings file is still the wrong place for it.")
        self._node = QLineEdit(self._settings.get("node") or "")
        self._node.setPlaceholderText("optional: this machine's enrolled node id")
        self._node.setToolTip(
            "Only once ENROLLED. Sending a node id the hub does not know is a 401, not a "
            "quiet fallback to the site token.")
        self._connect = QPushButton("Connect")
        self._connect.clicked.connect(self._do_connect)
        self._enroll = QPushButton("Enrol this machine…")
        self._enroll.setToolTip(
            "Mint a node identity for this machine so an operator can revoke exactly this "
            "one. Do it ONCE — enrolling on every launch fills their node list with "
            "entries they cannot tell apart.")
        self._enroll.clicked.connect(self._do_enroll)
        self._enroll.setEnabled(False)
        cl.addWidget(QLabel("URL"), 0, 0)
        cl.addWidget(self._url, 0, 1)
        cl.addWidget(QLabel("Token"), 1, 0)
        cl.addWidget(self._token, 1, 1)
        cl.addWidget(QLabel("Node id"), 2, 0)
        cl.addWidget(self._node, 2, 1)
        cl.addWidget(self._connect, 0, 2)
        cl.addWidget(self._enroll, 2, 2)
        return conn

    def _build_recipe_box(self) -> QWidget:
        pick = QGroupBox("Recipe")
        pl = QFormLayout(pick)
        self._recipe = QComboBox()
        self._recipe.currentTextChanged.connect(self._recipe_changed)
        pl.addRow("Workflow / recipe", self._recipe)

        row = QHBoxLayout()
        self._preset = QComboBox()
        self._preset.setToolTip(
            "Knob settings you saved for this recipe. They live on this machine and are "
            "applied when you run; nothing is sent to the hub until then.")
        self._preset.activated.connect(self._apply_preset)
        self._save_preset = QPushButton("Save as…")
        self._save_preset.clicked.connect(self._do_save_preset)
        self._delete_preset = QPushButton("Delete")
        self._delete_preset.clicked.connect(self._do_delete_preset)
        self._promote = QPushButton("Promote to recipe…")
        self._promote.setToolTip(
            "Turn these knob values into a derived recipe an operator can install, so "
            "somebody else can run exactly this.")
        self._promote.clicked.connect(self._do_promote)
        for widget in (self._preset, self._save_preset, self._delete_preset,
                       self._promote):
            row.addWidget(widget)
        pl.addRow("Preset", _wrap(row))

        self._recipe_note = _muted(QLabel("connect to a hub to see what its operator "
                                          "offers"))
        self._recipe_note.setWordWrap(True)
        pl.addRow(self._recipe_note)
        return pick

    def _build_data_box(self) -> QWidget:
        send = QGroupBox("Data")
        sl = QGridLayout(send)
        self._file = QLineEdit()
        self._file.setPlaceholderText("optional — leave empty for the recipe's own source")
        self._file.textChanged.connect(self._file_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._dest = QLineEdit(self._settings.get("dest")
                               or os.path.join(os.path.expanduser("~"), "lablink-results"))
        self._meta_note = _muted(QLabel(""))
        self._meta_note.setWordWrap(True)
        sl.addWidget(QLabel("Send file"), 0, 0)
        sl.addWidget(self._file, 0, 1)
        sl.addWidget(browse, 0, 2)
        sl.addWidget(QLabel("Results into"), 1, 0)
        sl.addWidget(self._dest, 1, 1, 1, 2)
        sl.addWidget(self._meta_note, 2, 0, 1, 3)
        return send

    def _build_run_box(self) -> QWidget:
        box = QWidget()
        row = QGridLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self._run = QPushButton("Run on the hub")
        self._run.clicked.connect(self._do_run)
        self._cancel = QPushButton("Cancel")
        self._cancel.setToolTip(
            "Cancellation is cooperative at step boundaries: the hub asks the worker to "
            "stop at the next node it has not started, and escalates to killing it if a "
            "single long step never checks.")
        self._cancel.clicked.connect(self._session.cancel)
        self._end = QPushButton("End session")
        self._end.setToolTip(
            "Close the session and free the hub's slot. The warm cache goes with it, so "
            "the next run pays the full cost again.")
        self._end.clicked.connect(self._do_end)
        self._bar = QProgressBar()
        self._bar.setVisible(False)
        self._state_note = _muted(QLabel("not connected"))
        row.addWidget(self._run, 0, 0)
        row.addWidget(self._cancel, 0, 1)
        row.addWidget(self._end, 0, 2)
        row.addWidget(self._state_note, 0, 3)
        row.addWidget(self._bar, 1, 0, 1, 4)
        row.setColumnStretch(3, 1)
        return box

    def _wire_session(self) -> None:
        s = self._session
        s.opened.connect(self._session_opened)
        s.state_changed.connect(self._session_state)
        s.progress.connect(self._session_progress)
        s.ran.connect(self._session_ran)
        s.fetched.connect(self._session_fetched)
        s.failed.connect(self._session_failed)
        s.log.connect(self._say)
        s.lost.connect(self._session_lost)

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
        protocol = hello.get("protocol")
        if protocol != P.WORKER_PROTOCOL_VERSION:
            # Fail closed on a protocol we do not understand, rather than guessing.
            self._say(f"WARNING: this hub speaks protocol {protocol!r} and this build "
                      f"knows {P.WORKER_PROTOCOL_VERSION}. Anything below may be wrong.")
        if not caps.get("recipe_metadata_requirements"):
            # Absent means "this hub cannot tell you", which is not the same as "no recipe
            # requires anything" — reading it as the latter fails OPEN.
            self._say("note: this hub does not publish per-recipe metadata requirements, "
                      "so an empty requirement list here means 'cannot say', not 'none'.")
        self._enroll.setEnabled(bool(caps.get("enroll")) and not self._node.text().strip())

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
        self._save_settings()
        self._sync_buttons()

    def _connect_failed(self, message: str) -> None:
        self._connect.setEnabled(True)
        self._say(f"could not connect: {message}")

    def _do_enroll(self) -> None:
        hub = self._hub
        if hub is None:
            return
        label, ok = QInputDialog.getText(
            self, "Enrol this machine",
            "A name the operator will see for this machine:",
            text=os.environ.get("COMPUTERNAME", "") or "microscope-pc")
        if not ok:
            return
        self._enroll.setEnabled(False)
        self.spawn(lambda: hub.enroll(label=label.strip()), self._enrolled,
                   lambda m: (self._say(f"enrolment failed: {m}"),
                              self._enroll.setEnabled(True)))

    def _enrolled(self, node_id: str) -> None:
        self._node.setText(node_id)
        self._say(f"enrolled as node {node_id!r}. Saved — do not enrol again on this "
                  f"machine.")
        self._save_settings()

    # ── recipe + presets ────────────────────────────────────────────────────────
    def _recipe_changed(self, key: str) -> None:
        meta = self._recipes.get(key)
        self._knobs.set_recipe(meta or {})
        self._reload_presets()
        if not meta:
            self._recipe_note.setText("connect to a hub to see what its operator offers")
            return
        runtime = meta.get("expected_runtime_s") or {}
        needs = meta.get("requires_metadata") or []
        note = (f"{meta.get('title') or meta.get('name')} — target "
                f"{meta.get('target')!r}, {meta.get('node_count')} node(s), typically "
                f"{runtime.get('typical', '?')}s (worst {runtime.get('worst', '?')}s). "
                f"{meta.get('description') or ''}")
        if needs:
            note += (f"\nDerives from: {', '.join(needs)} — a file that does not supply "
                     f"these is refused rather than run with defaults.")
        self._recipe_note.setText(note)
        self._file_changed()
        self._sync_buttons()

    def _current(self) -> Optional[Dict[str, Any]]:
        return self._recipes.get(self._recipe.currentText())

    def _reload_presets(self) -> None:
        self._preset.clear()
        self._preset.addItem("— none —", "")
        meta = self._current()
        if not meta:
            return
        problem = self._presets.problem()
        if problem:
            self._say(problem)
        for preset in self._presets.for_recipe(meta["workflow"], meta["name"]):
            self._preset.addItem(preset.name, preset.name)

    def _apply_preset(self) -> None:
        meta = self._current()
        name = str(self._preset.currentData() or "")
        if not meta or not name:
            return
        preset = self._presets.get(meta["workflow"], meta["name"], name)
        if preset is None:
            return
        applied = PR.apply_preset(preset, meta)
        self._knobs.set_values(applied.knobs)
        for warning in applied.warnings():
            # Never silently dropped: a recipe can change under a preset, and a preset that
            # quietly does something else is worse than one that refuses.
            self._say(f"preset {name!r}: {warning}")
        if applied.clean and not applied.warnings():
            self._say(f"applied preset {name!r}")

    def _do_save_preset(self) -> None:
        meta = self._current()
        if not meta:
            return
        knobs, problem = self._knobs.validate(meta)
        if problem:
            self._say(f"not saved — {problem}")
            return
        name, ok = QInputDialog.getText(self, "Save these knobs",
                                        "A name you will recognise later:")
        if not ok or not name.strip():
            return
        preset = PR.Preset(
            name=name.strip(), workflow=meta["workflow"], recipe=meta["name"],
            knobs=knobs, hub=self._url.text().strip(),
            saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
        try:
            self._presets.save(preset)
        except (OSError, ValueError) as exc:
            self._say(f"could not save the preset: {exc}")
            return
        self._reload_presets()
        index = self._preset.findData(preset.name)
        if index >= 0:
            self._preset.setCurrentIndex(index)
        self._say(f"saved preset {preset.name!r} ({len(knobs)} knob(s))")

    def _do_delete_preset(self) -> None:
        meta = self._current()
        name = str(self._preset.currentData() or "")
        if not meta or not name:
            return
        if self._presets.delete(meta["workflow"], meta["name"], name):
            self._say(f"deleted preset {name!r}")
            self._reload_presets()

    def _do_promote(self) -> None:
        meta = self._current()
        if not meta:
            return
        knobs, problem = self._knobs.validate(meta)
        if problem:
            self._say(f"cannot promote — {problem}")
            return
        from nodelab_v2.lablink.authoring import promote_preset
        promote_preset(self, hub=self._hub, recipe_meta=meta, knobs=knobs,
                       say=self._say)

    # ── data ────────────────────────────────────────────────────────────────────
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Send a file to the hub", self._settings.get("last_dir", ""),
            "Images (*.nd2 *.tif *.tiff);;All files (*)")
        if path:
            self._file.setText(path)
            self._settings["last_dir"] = os.path.dirname(path)
            self._save_settings()

    def _file_changed(self) -> None:
        """Say up front what this file can and cannot answer for the chosen recipe.

        Before the upload, not after: `missing_metadata` is a refusal that arrives once the
        bytes have crossed, and the whole point of reading the sidecar draft here is that a
        person finds out while they can still pick a different file.
        """
        path = self._file.text().strip()
        meta = self._current()
        if not path or not os.path.isfile(path) or not meta:
            self._meta_note.setText("")
            return
        needs = [str(f) for f in (meta.get("requires_metadata") or [])]
        if not needs:
            self._meta_note.setText("")
            return
        self.spawn(lambda: SC.draft(path), self._show_meta_state,
                   lambda m: self._meta_note.setText(f"could not read this file: {m}"))

    def _show_meta_state(self, draft: Any) -> None:
        meta = self._current()
        if not meta:
            return
        needs = [str(f) for f in (meta.get("requires_metadata") or [])]
        unmet = draft.unmet(needs)
        if unmet:
            self._meta_note.setText(
                f"This file does not supply {', '.join(unmet)}, which this recipe derives "
                f"from. Running it would be refused — use 'Fix metadata…' to supply them.")
        else:
            self._meta_note.setText(
                f"metadata this recipe needs: all present ({', '.join(needs)})")

    # ── run ─────────────────────────────────────────────────────────────────────
    def _do_run(self) -> None:
        meta = self._current()
        hub = self._hub
        if hub is None or not meta:
            return
        knobs, problem = self._knobs.validate(meta)
        if problem:
            self._say(f"not sent — {problem}")
            return
        dest = self._dest.text().strip() or "."
        path = self._file.text().strip()
        self._save_settings()

        if not self._session.alive and self._session.state != "opening":
            self._say(f"opening a session for {meta['workflow']}/{meta['name']} …")
            self._session.open(hub, meta["workflow"], meta["name"], knobs=knobs)
            self._sent_path = ""
        # Uploaded once per session, and again whenever the FILE changes — otherwise a warm
        # session keeps running the first file after somebody picks a second one, and every
        # number is attributed to the wrong image with nothing reporting it.
        if path and path != self._sent_path:
            sidecar = self._write_sidecar(path, meta)
            self._session.send_file(path, sidecar_path=sidecar)
            self._sent_path = path
        self._session.run(knobs, results_dir=dest)
        self._sync_buttons()

    def _write_sidecar(self, path: str, meta: Dict[str, Any]) -> str:
        """Build and stage the ``.job.json`` that travels with this image.

        Written from the file's own calibration and never from a guess: a field the reader
        could not answer stays absent, so the worker refuses rather than computing different
        numbers from an invented one.
        """
        try:
            draft = SC.draft(path, recipe=str(meta.get("name") or ""))
        except Exception as exc:                         # noqa: BLE001
            self._say(f"could not read this file's calibration: {exc}")
            return ""
        self._sidecar_dir = self._sidecar_dir or tempfile.mkdtemp(prefix="nd2s-lablink-")
        try:
            out = SC.write_sidecar(draft, self._sidecar_dir,
                                   image_name=repair_name(os.path.basename(path)))
        except OSError as exc:
            self._say(f"could not write the sidecar: {exc}")
            return ""
        if draft.invented or draft.absent:
            self._say(f"sidecar written from the file: {', '.join(draft.from_file)}"
                      + (f"; not stated: {', '.join(draft.absent + draft.invented)}"
                         if (draft.absent or draft.invented) else ""))
        for note in draft.notes:
            self._say(f"  {note}")
        return out

    def _do_end(self) -> None:
        self._session.close()
        self._say("closing the session — the hub's slot is freed and the warm cache goes.")

    # ── session signals ─────────────────────────────────────────────────────────
    def _session_opened(self, info: Dict[str, Any]) -> None:
        self._say(f"session {info['id']} ready (quota {info.get('quota_mb')} MB). "
                  f"It stays open: change a knob and run again to use the warm cache.")

    def _session_state(self, state: str) -> None:
        self._state_note.setText({
            "idle": "not connected",
            "opening": "starting the hub's worker…",
            "ready": f"session {self._session.session_id} warm",
            "running": "running…",
            "closing": "closing…",
            "closed": "session closed",
        }.get(state, state))
        self._bar.setVisible(state == "running")
        if state == "running":
            self._bar.setRange(0, 0)
        self._sync_buttons()

    def _session_progress(self, progress: Any) -> None:
        # `determinate: false` means the percentage is not known, so none is shown — a bar
        # that invents one is worse than a bar that says "7 of 12".
        text = str(getattr(progress, "text", "") or "")
        if getattr(progress, "determinate", False) and progress.total:
            self._bar.setRange(0, int(progress.total))
            self._bar.setValue(int(progress.done))
        else:
            self._bar.setRange(0, 0)
        # `%` doubled: QProgressBar's format string eats %p/%v/%m, so a node label containing
        # a literal percentage would render as a number from somewhere else entirely.
        self._bar.setFormat(text.replace("%", "%%"))
        self._state_note.setText(text or "running…")

    def _session_ran(self, outcome: Any) -> None:
        self._last = outcome
        self._card.show_outcome(outcome)
        if outcome.ok:
            self._say(f"run {outcome.cmd_seq} done in {outcome.duration_s}s — "
                      f"{outcome.speedup_note or 'no step breakdown reported'}")
        else:
            self._say(f"run {outcome.cmd_seq} {outcome.state}: {outcome.code} — "
                      f"{outcome.message}")
            self._say("the session stayed usable; change a knob and run again.")
        if outcome.held:
            self._card.show_held(outcome.held)
            self._say(f"held on the hub until pulled: {', '.join(outcome.held)}")
        self._sync_buttons()

    def _session_fetched(self, outcome: Any) -> None:
        for path in outcome.fetched:
            self._say(f"got {os.path.basename(path)} "
                      f"({_human_bytes(os.path.getsize(path))}, verified)")
        self._card.show_results(outcome)
        self._write_run_record(outcome)

    def _write_run_record(self, outcome: Any) -> None:
        """Leave the results folder able to say what produced it.

        The hub carries the recipe name in its channel *listing* only, so it is gone the
        moment a file is downloaded. Without this, a folder of results a week later is an
        image nobody can reproduce or tune further.
        """
        meta = self._current() or {}
        if not outcome.results_dir:
            return
        record = PR.RunRecord(
            recipe=str(meta.get("name") or ""), workflow=str(meta.get("workflow") or ""),
            hub=self._url.text().strip(), knobs=dict(outcome.knobs or {}),
            cmd_id=outcome.cmd_id, cmd_seq=outcome.cmd_seq,
            session=self._session.session_id, duration_s=outcome.duration_s,
            cached_steps=outcome.cached_steps, computed_steps=outcome.computed_steps,
            inputs=[{"name": r.name, "sha256": r.sha256}
                    for r in self._session.inputs],
            artifacts=[{"name": a.get("returned_as") or a.get("name"),
                        "sha256": a.get("sha256"), "kind": a.get("kind"),
                        "bytes": a.get("bytes")} for a in (outcome.artifacts or [])],
            requires_metadata=[str(f) for f in (meta.get("requires_metadata") or [])],
            ran=time.strftime("%Y-%m-%dT%H:%M:%S"),
            fidelity="full" if outcome.knobs else "recipe-name-only")
        try:
            PR.write_run_record(record, outcome.results_dir)
            self._last_record = record
        except OSError as exc:
            self._say(f"could not write the run record: {exc}")

    def _session_failed(self, where: str, message: str) -> None:
        self._say(f"{where}: {message}")
        self._sync_buttons()

    def _session_lost(self, message: str) -> None:
        # Surfaced, never silently reopened: reopening and rerunning would hide a result the
        # person may already be looking at, and the warm cache is gone either way.
        self._say(f"the session is gone: {message}")
        self._say("its warm cache went with it. Run again to open a new one — the first "
                  "run will pay the full cost.")
        self._sync_buttons()

    # ── enabling ────────────────────────────────────────────────────────────────
    def _sync_buttons(self) -> None:
        has_recipe = self._current() is not None
        running = self._session.state == "running"
        # `opening` counts as busy: a second press while the hub is still starting its worker
        # would open a SECOND session, and on a hub with one slot per workflow the second is
        # refused while the first is still the one you wanted.
        busy = running or self._session.state in ("opening", "closing")
        self._run.setEnabled(has_recipe and not busy and self._hub is not None)
        self._run.setText("Run again (warm)" if self._session.alive and not busy
                          else "Run on the hub")
        self._cancel.setEnabled(running)
        self._end.setEnabled(self._session.alive or running)
        for widget in (self._save_preset, self._promote):
            widget.setEnabled(has_recipe)
        self._delete_preset.setEnabled(bool(self._preset.currentData()))

    # ── settings ────────────────────────────────────────────────────────────────
    def _save_settings(self) -> None:
        self._settings.update({
            "url": self._url.text().strip(),
            "node": self._node.text().strip(),
            "dest": self._dest.text().strip(),
        })
        _store_settings(self._settings)

    # ── log ─────────────────────────────────────────────────────────────────────
    def _say(self, text: str) -> None:
        self._log.appendPlainText(text)

    def shutdown(self) -> None:
        """Close any live session, then join the request threads."""
        self._session.shutdown()
        self.stop_tasks()


# ── this machine's remembered connection ────────────────────────────────────────
#
# The token is deliberately NOT here. It is anti-misdirection rather than security, but a
# secret written into a settings file is a secret in the wrong place, and the integration
# checklist is explicit about it.

_SETTINGS_FILE = Path.home() / ".nd2studios" / "lablink.json"


def _load_settings() -> Dict[str, Any]:
    try:
        with open(_SETTINGS_FILE, encoding="utf-8") as fh:
            doc = json.load(fh)
        return dict(doc) if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def _store_settings(settings: Dict[str, Any]) -> None:
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in settings.items() if k != "token"},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError:
        pass            # remembering a URL is a convenience, never a reason to fail


def _wrap(layout: QHBoxLayout) -> QWidget:
    """A bare layout as a widget, so it can go in a QFormLayout row."""
    holder = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    holder.setLayout(layout)
    return holder


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
        # The send tab owns a live session as well as request threads, and closing it is
        # not optional — a leaked session holds one of very few hub slots until an idle
        # timer expires many minutes later.
        self.send.shutdown()


__all__ = ["LabLinkPanel", "ServePanel", "SendPanel"]
