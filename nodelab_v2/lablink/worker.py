"""ND2Studios as a LabLink worker — LWP/1 over NDJSON on stdin/stdout.

    python -m nodelab_v2.lablink.worker --session /path/to/session

A LabLink hub holds heavy analysis software **warm** and runs it on behalf of the small
machines in a lab: a microscope PC opens a session, streams a file and a few whitelisted
knobs, and the second run after a knob change costs milliseconds instead of the whole
pipeline. This module is the ND2Studios end of that — the process the hub spawns, one per
session, and talks to down a pipe.

**Why a subprocess speaking JSON rather than an HTTP service or an import.** Not our
choice; the hub's, and a good one. The hub package is standard-library-only so it can be
dropped onto an instrument PC, and a worker needs numpy, a microscopy reader and gigabytes
of cache — those cannot share a process with the hub without the hub inheriting every one
of them. There is also no port to allocate, no auth to get wrong, no firewall rule, and no
way to leak a connection: killing the process IS closing it.

The conversation, and what each command maps onto in this repo:

===========  =============================================================================
``open``     :func:`nodegraph.serialize.from_dict` on the recipe's ``graph.nd2graph.json``,
             then the **tier-2 validation** the hub structurally cannot do (see
             :meth:`Worker.validate_recipe`).
``set``      a knob writes one ``params`` or ``modes`` entry on one
             :class:`~nodegraph.graph.NodeInstance`. ``null`` REMOVES the key, which is
             what makes the value derived from the file's own metadata again.
``input``    :func:`nodelab_v2.ingest.ingest_image` to a ``.b2nd`` store, cached for the
             life of the session, seeding the recipe's declared loader node.
``run``      :func:`nodelab_v2.ops.headless_engine` with our observer attached, then
             ``pull(target)``, then :mod:`nodelab_v2.lablink.artifacts`.
``pull``     publish an artifact the recipe declared ``policy: "pull"``.
``reset``    drop the memo and tile cache — the "free the RAM but stay warm" rung.
``cancel``   set a flag every wrapped compute checks at its own boundary.
``ping``     clock + busy, answered off the reader thread so it works mid-run.
``close``    acknowledge, then exit.
===========  =============================================================================

Four rules here are load-bearing, and each one is a real failure mode rather than style:

* **stdout is the control stream and nothing else.** Every human word goes to stderr via
  :func:`note`. A worker that prints to stdout is not fatal — the hub counts noise and
  warns — but a *parsed* line is the only thing that resets the hub's silence clock, so
  chattiness cannot substitute for progress.
* **stdin is read on a thread separate from the work.** ``cancel`` and ``ping`` are
  answered there, inline. A worker that reads stdin only between commands can never be
  cancelled, and would pass a naive test suite anyway.
* **Exactly one terminal message per command id, never both, never zero.** Enforced by
  :meth:`Worker._answer` and a ``finally`` guard in :meth:`Worker.serve`, not by
  discipline — "zero" is the failure that presents to an operator as a hung session, and
  it is the one that discipline loses.
* **Bulk data never rides the pipe.** A line over 1 MiB kills the worker, because past
  that point the hub cannot know where the JSON ended. Artifacts are files under the
  session directory and the events carry paths; :func:`emit` additionally *truncates*
  oversize strings rather than letting a long traceback take the session down.

**Cancellation is cooperative at node boundaries, and that is the honest guarantee.** The
engine has no cancellation point inside a compute — a single CNN inference or an upstream
solver simply runs to completion — so :meth:`Worker._cancellable_computes` wraps the
compute table and raises at the *next node that has not started yet*. We advertise exactly
that in ``hello`` (``cooperative_cancel: "at_step_boundary"``) so the hub's escalation
ladder (cancel → grace → SIGTERM → SIGKILL) is applied against a truthful claim. A long
single node will be killed, and losing the warm cache is the correct outcome there.

Qt-free by construction: nothing in this module's import graph reaches PySide6, which is
why the tabulation lives in :mod:`nodelab_v2.tables` rather than in the spreadsheet panel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from nodelab_v2.lablink import protocol as P

# ── the wire ────────────────────────────────────────────────────────────────────

_seq = [0]
_out_lock = threading.Lock()

#: Longest string we will put in one field. A traceback, a stderr tail or a numpy repr can
#: be megabytes, and one oversize LINE is fatal to the session — so values are clamped
#: here, individually, and the clamp is visible in the output rather than silent.
MAX_FIELD_CHARS = 8000


def note(text: str) -> None:
    """Human logging. stderr, never stdout — see the module docstring."""
    try:
        sys.stderr.write(f"{text}\n")
        sys.stderr.flush()
    except Exception:       # noqa: BLE001 — a closed stderr must not end a run
        pass


def sanitize(obj: Any, _path: str = "", _found: Optional[List[str]] = None
             ) -> Tuple[Any, List[str]]:
    """Replace non-finite floats with ``None``; returns ``(clean, paths_replaced)``.

    NaN and Infinity are what a statistic over an empty region legitimately produces, so
    the result is kept and only the value is neutralised — but they are **not JSON**, and
    one reaching a stored record makes that document unparseable for everything that is not
    Python. The hub does this too, inbound; doing it here as well means the paths are
    reported by the side that knows what the number was measuring.
    """
    found = [] if _found is None else _found
    if isinstance(obj, float):
        if not math.isfinite(obj):
            found.append(_path or "$")
            return None, found
        return obj, found
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            out[key], _ = sanitize(v, f"{_path}.{key}" if _path else key, found)
        return out, found
    if isinstance(obj, (list, tuple)):
        arr = []
        for i, v in enumerate(obj):
            clean, _ = sanitize(v, f"{_path}[{i}]", found)
            arr.append(clean)
        return arr, found
    if isinstance(obj, str) and len(obj) > MAX_FIELD_CHARS:
        return obj[:MAX_FIELD_CHARS] + f"… [{len(obj) - MAX_FIELD_CHARS} chars dropped]", found
    # numpy scalars and anything else exotic: let json fail loudly rather than guess,
    # except for the two cases that turn up constantly in metadata dicts
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        try:
            return sanitize(obj.item(), _path, found)
        except Exception:       # noqa: BLE001
            return str(obj), found
    return obj, found


def emit(**msg: Any) -> None:
    """One JSON object, one line, flushed. The only way anything reaches stdout.

    Non-finite numbers are neutralised and reported in ``_sanitized``; oversize strings are
    clamped. If the line is *still* too long — a pathological artifact list, say — the
    message is replaced by a small one that says so, because a truncated JSON line would
    kill the session and a silent drop would lose a terminal event.
    """
    clean, replaced = sanitize(msg)
    if replaced:
        clean["_sanitized"] = replaced
    with _out_lock:
        _seq[0] += 1
        line = json.dumps({"seq": _seq[0], **clean}, allow_nan=False)
        if len(line.encode("utf-8", "replace")) > P.MAX_WORKER_LINE_BYTES:
            keep = {k: clean[k] for k in ("id", "ev") if k in clean}
            line = json.dumps({
                "seq": _seq[0], **keep,
                "message": "this worker tried to emit a line over the protocol's "
                           "1 MiB limit; it was replaced. Bulk data belongs in a file "
                           "under the session directory, not in the pipe.",
            })
            note("dropped an oversize protocol line (see the replacement on stdout)")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class Cancelled(Exception):
    """Raised inside a wrapped compute when the hub has asked us to stop."""


class WorkerFault(Exception):
    """A command failed with a specific LWP error code."""

    def __init__(self, code: str, message: str, detail: str = ""):
        super().__init__(message)
        if code in P.HUB_ONLY_ERROR_CODES:
            # Emitting one of these would report a LINK failure for a compute problem and
            # send an operator to look at the network. Refuse at construction.
            raise AssertionError(f"{code!r} is the hub's to synthesise, never ours")
        self.code = code
        self.message = message
        self.detail = detail


# ── optional dependency census ──────────────────────────────────────────────────

#: Optional imports a recipe may name in ``requires.capabilities``. The hub reads our
#: answer via ``WorkerLink.missing_for``, so the reason an operator sees is the worker's
#: own wording — this process is the only thing that knows why an import failed.
OPTIONAL_DEPS = (
    ("scikit-learn", "sklearn"),
    ("numba", "numba"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("tifffile", "tifffile"),
    ("nd2", "nd2"),
    ("blosc2", "blosc2"),
    ("opencv", "cv2"),
    ("stardist", "stardist"),
    ("cellsam", "cellSAM"),
    ("al-dic", "aldic"),
    ("cupy", "cupy"),
)


def dependency_census() -> Dict[str, Dict[str, Any]]:
    """``{capability: {"ok": bool, "version": str} | {"ok": False, "why": str}}``.

    Imports each module for real. That costs a second at startup and is the point: a
    capability reported from a version pin rather than an import is a capability that
    reports ``ok`` for a package whose native half is broken.
    """
    import importlib

    out: Dict[str, Dict[str, Any]] = {}
    for name, module in OPTIONAL_DEPS:
        try:
            mod = importlib.import_module(module)
        except BaseException as exc:      # noqa: BLE001 — a bad native lib can SystemExit
            out[name] = {"ok": False,
                         "why": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        out[name] = {"ok": True, "version": str(getattr(mod, "__version__", "") or "")}
    return out


def _rss_bytes() -> Optional[int]:
    """This process's resident set, if it can be had without a hard dependency."""
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:       # noqa: BLE001 — psutil is optional; a beat without rss is fine
        return None


# ── the worker ──────────────────────────────────────────────────────────────────

class Worker:
    """One warm ND2Studios instance, and the LWP/1 conversation with it."""

    def __init__(self, session_dir: str) -> None:
        self.session = os.path.abspath(session_dir)
        self.out_dir = os.path.join(self.session, "out")
        self.stop = False

        # open state
        self.recipe: Dict[str, Any] = {}
        self.graph: Any = None
        self.targets: Tuple[str, ...] = ()
        self.memo_bytes: Optional[int] = None
        self.cache_bytes: Optional[int] = None

        # session state that survives commands — this IS the warmth
        self.knobs: Dict[str, Any] = {}
        self.inputs: Dict[str, Dict[str, Any]] = {}      # role -> {path, name, sha256}
        self._providers: Dict[str, Tuple[Any, Any]] = {}  # abs path -> (provider, envelope)
        self._memo: Any = None
        self._tiles: Any = None
        self.runs = 0
        self.held: Dict[str, Any] = {}                    # name -> Artifact (policy=pull)

        # concurrency
        self.work: "Queue[Optional[dict]]" = Queue()
        self.cancel = threading.Event()
        self._current_id: Optional[int] = None
        self._busy = threading.Event()
        self._answered: set = set()
        self._beat_stop: Optional[threading.Event] = None
        #: while set, the engine observer emits nothing — see :meth:`_observer`.
        self._muted = False

    # ── handshake ───────────────────────────────────────────────────────────────
    def hello(self) -> None:
        """Announce ourselves. Emitted only after the heavy imports have SUCCEEDED, so
        ``hello`` genuinely means ready — an import storm that fails exits non-zero with
        the reason on stderr, which the hub reports as ``worker_exited`` plus our own
        stderr tail rather than as a mysterious ``open_timeout``."""
        from nodegraph.serialize import FORMAT_VERSION

        emit(id=None, ev="hello",
             protocol=P.WORKER_PROTOCOL_VERSION,
             worker=P.WORKER_NAME, worker_version=P.WORKER_VERSION,
             software=P.SOFTWARE_NAME, software_version=P.SOFTWARE_VERSION,
             graph_format=FORMAT_VERSION,
             pid=os.getpid(), clock=time.time(),
             capabilities=list(P.WORKER_COMMANDS),
             features={
                 "node_events": True,
                 "incremental_rerun": True,      # the memo is what makes a re-run cheap
                 "synthetic_source": True,       # an empty loader path = the demo stack
                 "cooperative_cancel": "at_step_boundary",
                 "artifact_kinds": list(P.ARTIFACT_KINDS),
                 "zones": True,                  # Repeat/Sim unrolling
                 "iterate": True,                # flow.iterate sweeps
                 "docks": True,                  # io.dock checkpoints
             },
             deps={"python": "%d.%d.%d" % sys.version_info[:3],
                   "numpy": _version_of("numpy")},
             optional=dependency_census(),
             limits={"max_line_bytes": P.MAX_WORKER_LINE_BYTES, "max_sessions": 1})

    # ── stdin ───────────────────────────────────────────────────────────────────
    def read_stdin(self) -> None:
        """Separate thread. ``cancel`` and ``ping`` are answered HERE so they land while a
        run is in flight; everything else is queued for the work thread."""
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except ValueError:
                    note(f"unparseable command, ignored: {line[:200]!r}")
                    continue
                if not isinstance(cmd, dict):
                    note(f"command was not a JSON object, ignored: {line[:200]!r}")
                    continue
                name, mid = cmd.get("cmd"), cmd.get("id")
                if name == "cancel":
                    target = cmd.get("target_id")
                    self.cancel.set()
                    note(f"cancel requested for command {target}")
                    self._answer(mid, "result", acknowledged=True,
                                 target_id=target, busy=self._busy.is_set())
                elif name == "ping":
                    self._answer(mid, "result", clock=time.time(),
                                 busy=self._busy.is_set())
                else:
                    self.work.put(cmd)
                    if name == "close":
                        return
        except Exception as exc:      # noqa: BLE001 — a broken pipe is a normal shutdown
            note(f"stdin reader ended: {type(exc).__name__}: {exc}")
        finally:
            self.work.put(None)

    # ── terminal-message discipline ─────────────────────────────────────────────
    def _answer(self, mid: Any, ev: str, **data: Any) -> bool:
        """Emit the ONE terminal message for ``mid``. Returns whether it was emitted.

        A second terminal for the same id is dropped and logged rather than sent: the hub
        treats the first as final and a second would be attributed to whatever command
        reused the queue. Both halves of "never both, never zero" live here and in
        :meth:`serve`'s ``finally``.
        """
        if ev not in P.TERMINAL_EVENTS:
            raise AssertionError(f"{ev!r} is not a terminal event")
        if mid is None:
            emit(id=None, ev=ev, **data)
            return True
        if mid in self._answered:
            note(f"refusing a second terminal message for command {mid} ({ev})")
            return False
        self._answered.add(mid)
        emit(id=mid, ev=ev, **data)
        return True

    def _fault(self, mid: Any, code: str, message: str, detail: str = "") -> None:
        payload: Dict[str, Any] = {"code": code, "message": message, "fatal": False}
        if detail:
            payload["detail"] = detail
        self._answer(mid, "error", **payload)

    # ── dispatch ────────────────────────────────────────────────────────────────
    def serve(self) -> None:
        reader = threading.Thread(target=self.read_stdin, name="lwp-stdin", daemon=True)
        reader.start()
        while not self.stop:
            cmd = self.work.get()
            if cmd is None:
                return
            mid, name = cmd.get("id"), cmd.get("cmd")
            self._current_id = mid
            self._busy.set()
            try:
                self.dispatch(mid, name, cmd)
            except WorkerFault as exc:
                self._fault(mid, exc.code, exc.message, exc.detail)
            except Cancelled as exc:
                self._fault(mid, "cancelled", str(exc) or "cancelled by request")
            except MemoryError:
                # Reported as its own code so an operator reads "this box is too small for
                # this recipe" rather than "the software is broken".
                self._fault(mid, "resource_exhausted",
                            "ran out of memory. Lower the recipe's cache budget, scope "
                            "the run to fewer frames, or give the hub more RAM.",
                            traceback.format_exc())
            except BaseException as exc:      # noqa: BLE001 — never leave an id open
                self._fault(mid, "internal", f"{type(exc).__name__}: {exc}",
                            traceback.format_exc())
            finally:
                self._busy.clear()
                # The other half of "never zero": a handler that returned without
                # answering would hang the session until the hub's silence timeout.
                if mid is not None and mid not in self._answered:
                    note(f"command {mid} ({name}) returned without a terminal message")
                    self._fault(mid, "internal",
                                f"{name!r} finished without reporting a result. This is a "
                                f"bug in the ND2Studios LabLink worker.")
                self._current_id = None

    def dispatch(self, mid: Any, name: Any, cmd: dict) -> None:
        handler = {
            "open": self.cmd_open, "set": self.cmd_set, "input": self.cmd_input,
            "run": self.cmd_run, "pull": self.cmd_pull, "reset": self.cmd_reset,
            "close": self.cmd_close,
        }.get(str(name))
        if handler is None:
            raise WorkerFault("unsupported",
                              f"{name!r} is not implemented by this worker. It offers: "
                              f"{', '.join(P.WORKER_COMMANDS)}")
        handler(mid, cmd)

    # ── open ────────────────────────────────────────────────────────────────────
    def cmd_open(self, mid: Any, cmd: dict) -> None:
        from nodegraph.serialize import from_dict
        from nodelab_v2.ops import ensure_ops

        recipe = cmd.get("recipe") or {}
        if not isinstance(recipe, dict):
            raise WorkerFault("bad_request", "'recipe' must be an object")
        graph_path = cmd.get("graph_path") or recipe.get("graph") or ""
        if not graph_path:
            raise WorkerFault("bad_request",
                              "open needs 'graph_path' (or recipe.graph)")
        session_dir = cmd.get("session_dir")
        if session_dir:
            self.session = os.path.abspath(str(session_dir))
            self.out_dir = os.path.join(self.session, "out")
        limits = cmd.get("limits") or {}
        self.memo_bytes = _as_bytes(limits.get("memo_bytes"))
        self.cache_bytes = _as_bytes(limits.get("cache_bytes"))

        try:
            with open(graph_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except OSError as exc:
            raise WorkerFault("bad_recipe",
                              f"cannot read the recipe's graph: {exc}") from None
        except ValueError as exc:
            raise WorkerFault("bad_recipe",
                              f"the recipe's graph is not valid JSON: {exc}") from None

        # Register io.load / io.dock / view.viewer BEFORE parsing, or a GUI-authored graph
        # fails tier-2 on ops the core catalog does not own.
        import nodegraph.catalog     # noqa: F401 — importing registers the catalog
        ensure_ops()
        try:
            graph, zones, groups = from_dict(raw)
        except ValueError as exc:
            raise WorkerFault("bad_recipe", f"{exc}") from None

        if zones and not recipe.get("allow_zones"):
            raise WorkerFault(
                "bad_recipe",
                f"the graph contains {len(zones)} zone(s), which multiply the work by "
                f"their iteration count, but the recipe does not set 'allow_zones'.")

        problems = self.validate_recipe(recipe, graph)
        if problems:
            raise WorkerFault(
                "bad_recipe",
                f"recipe {recipe.get('name') or '?'!r}: {problems[0]}",
                "\n".join(f"- {p}" for p in problems))

        self.recipe = recipe
        self.graph = graph
        self.targets = tuple(graph.topo_order())
        self.knobs = {}
        self.inputs = {}
        self.held = {}
        note(f"opened recipe {recipe.get('name')!r}: {len(graph.nodes)} node(s), "
             f"target {recipe.get('target')!r}")
        self._answer(mid, "result", validated=True, targets=list(self.targets),
                     node_count=len(graph.nodes), graph_format=raw.get("format_version"),
                     zones=len(zones), groups=len(groups))

    # ── tier-2 validation ───────────────────────────────────────────────────────
    def validate_recipe(self, recipe: dict, graph: Any) -> List[str]:
        """The checks the hub structurally cannot make, because each needs the node
        catalogue — which needs numpy, which is why the hub must not pretend to.

        The hub's ``recipes.TIER2_CHECKS`` names exactly four, and this is them:

        1. each knob's socket exists on its node, and its unit matches the socket's own;
        2. each mode knob's values are among the node's real choices;
        3. every node with a 2D/3D lever sets it explicitly — a graph that does not would
           run 2D on a z-stack, silently;
        4. each output kind is one this worker can actually produce.

        Returns every problem rather than the first, because an operator fixing a recipe
        wants the whole list, and reports them in a stable order so two runs of
        ``--check-config`` do not disagree about which one is "first".
        """
        from nodegraph.registry import DIM_MODE, NODES

        problems: List[str] = []
        nodes = graph.nodes

        for knob in recipe.get("knobs") or []:
            name = knob.get("name")
            nid = knob.get("node")
            param = knob.get("param")
            kind = knob.get("kind") or "param"
            unit = knob.get("unit") or ""
            rec = nodes.get(nid)
            if rec is None:
                problems.append(
                    f"knob {name!r}: node {nid!r} is not in the graph "
                    f"(it has: {', '.join(sorted(nodes))})")
                continue
            spec = NODES.get(rec.op_key)
            if spec is None:
                problems.append(
                    f"knob {name!r}: node {nid!r} is op {rec.op_key!r}, which this build "
                    f"does not have. Recipe and worker are different versions.")
                continue
            if kind == "mode":
                mode = next((m for m in spec.modes if m.name == param), None)
                if mode is None:
                    offered = ", ".join(m.name for m in spec.modes) or "(none)"
                    problems.append(
                        f"knob {name!r}: {rec.op_key!r} has no mode {param!r}. "
                        f"Its modes: {offered}")
                    continue
                # (2) every declared enum value must be a real choice
                for value in knob.get("enum") or ():
                    if value not in tuple(mode.choices):
                        problems.append(
                            f"knob {name!r}: {value!r} is not a choice of "
                            f"{rec.op_key!r} mode {param!r} "
                            f"({', '.join(mode.choices)})")
                default = knob.get("default")
                if default is not None and default not in tuple(mode.choices):
                    problems.append(
                        f"knob {name!r}: its default {default!r} is not a choice of "
                        f"{rec.op_key!r} mode {param!r} ({', '.join(mode.choices)})")
            else:
                # (1) the socket must exist, and its unit must agree
                sock = spec.input(param)
                if sock is None:
                    offered = ", ".join(s.name for s in spec.inputs) or "(none)"
                    problems.append(
                        f"knob {name!r}: {rec.op_key!r} has no param socket {param!r}. "
                        f"Its params: {offered}")
                    continue
                declared = str(getattr(sock, "unit", "") or "")
                if unit and declared and unit != declared:
                    problems.append(
                        f"knob {name!r}: the recipe calls it {unit!r} but "
                        f"{rec.op_key!r}.{param} is in {declared!r}. One of them means "
                        f"something else, and the number would be silently wrong.")
                if unit and not declared:
                    problems.append(
                        f"knob {name!r}: the recipe gives it unit {unit!r} but "
                        f"{rec.op_key!r}.{param} declares none (it is dimensionless).")

        # (3) the 2D/3D lever must be explicit wherever a node has one
        for nid, rec in sorted(nodes.items()):
            spec = NODES.get(rec.op_key)
            if spec is None or not spec.has_dim_lever():
                continue
            if DIM_MODE not in (rec.modes or {}):
                lever = spec.dim_lever()
                choices = ", ".join(lever.choices) if lever is not None else "2D, 3D"
                problems.append(
                    f"node {nid!r} ({rec.op_key}) has a 2D/3D lever and the graph does "
                    f"not set it. Add modes.{DIM_MODE} ({choices}) — left unset it runs "
                    f"the default on whatever arrives, so a z-stack would be processed "
                    f"plane by plane and nothing would report it.")

        # (4) we must be able to make every output the recipe promises
        for out in recipe.get("outputs") or []:
            kind = out.get("kind")
            if kind not in P.ARTIFACT_KINDS:
                problems.append(
                    f"output {out.get('name')!r}: this worker cannot produce a "
                    f"{kind!r} artifact. It produces: {', '.join(P.ARTIFACT_KINDS)}")
            source = out.get("from")
            if source and source not in nodes:
                problems.append(
                    f"output {out.get('name')!r}: 'from' names {source!r}, which is not "
                    f"a node in the graph")

        # the target must be pullable
        target = recipe.get("target")
        if target and target not in nodes:
            problems.append(
                f"targets.primary is {target!r}, which is not a node in the graph "
                f"({', '.join(sorted(nodes))})")

        # and every declared input must name a loader we can actually seed
        for inp in recipe.get("inputs") or []:
            nid = inp.get("node")
            if nid and nid not in nodes:
                problems.append(
                    f"input {inp.get('role')!r}: node {nid!r} is not in the graph")
        return problems

    # ── set ─────────────────────────────────────────────────────────────────────
    def cmd_set(self, mid: Any, cmd: dict) -> None:
        """Apply whitelisted knobs to the graph.

        ``None`` **removes** the param rather than writing a zero, and that distinction is
        the whole subtlety of the metadata-intelligent params this engine is built on: a
        param MISSING from a node's ``params`` means "derive it from this file's
        calibration", while a param PRESENT means somebody pinned it. Writing 0, or
        writing the default when the recipe asked for derivation, silently overrides the
        microscope and nothing anywhere reports it.
        """
        self._require_open()
        knobs = cmd.get("knobs")
        if knobs is None:
            knobs = {}
        if not isinstance(knobs, dict):
            raise WorkerFault("bad_request", "'knobs' must be an object")
        by_name = {k.get("name"): k for k in (self.recipe.get("knobs") or [])}
        effective: Dict[str, Any] = {}
        for name, value in knobs.items():
            spec = by_name.get(name)
            if spec is None:
                offered = ", ".join(sorted(n for n in by_name if n)) or "(none)"
                raise WorkerFault(
                    "no_such_knob",
                    f"recipe {self.recipe.get('name')!r} has no knob named {name!r}. "
                    f"It offers: {offered}")
            rec = self.graph.nodes.get(spec.get("node"))
            if rec is None:      # open() validated this; a graph cannot change under us
                raise WorkerFault("internal",
                                  f"knob {name!r} targets node {spec.get('node')!r}, "
                                  f"which is no longer in the graph")
            store = rec.modes if (spec.get("kind") == "mode") else rec.params
            param = spec.get("param")
            if value is None:
                store.pop(param, None)
                effective[name] = {"value": None, "source": "derive"}
            else:
                store[param] = value
                self.knobs[name] = value
                effective[name] = {"value": value, "source": "node"}
        # Report everything in effect, not just what this call changed: the echo is the
        # record of what actually ran, and a node logging it wants the whole state.
        for name, spec in by_name.items():
            if name in effective or not name:
                continue
            rec = self.graph.nodes.get(spec.get("node"))
            if rec is None:
                continue
            store = rec.modes if (spec.get("kind") == "mode") else rec.params
            param = spec.get("param")
            if param in store:
                effective[name] = {"value": store[param], "source": "graph"}
            else:
                effective[name] = {"value": None, "source": "derive"}
        self._answer(mid, "result", effective=effective)

    # ── input ───────────────────────────────────────────────────────────────────
    def cmd_input(self, mid: Any, cmd: dict) -> None:
        """Ingest a file the node uploaded, and remember it for the rest of the session.

        The sha256 the hub sends is **verified here as well**. The hub already checked it
        against the channel, so this is not distrust of the hub — it is the cheap guard
        against the copy onto the session directory having been truncated, which costs one
        pass over a file we are about to spend far longer decoding anyway.
        """
        self._require_open()
        role = str(cmd.get("role") or "image")
        path = cmd.get("path")
        if not path:
            raise WorkerFault("bad_request", "input needs a 'path'")
        path = os.path.abspath(str(path))
        if not os.path.isfile(path):
            raise WorkerFault("missing_input", f"no such input file: {path}")

        declared = str(cmd.get("sha256") or "")
        if declared:
            from nodelab_v2.lablink.artifacts import sha256_file
            got, size = sha256_file(path)
            if got != declared:
                raise WorkerFault(
                    "input_error",
                    f"{os.path.basename(path)} does not match the sha256 the hub sent "
                    f"({got[:16]}… vs {declared[:16]}…). The file on the session "
                    f"directory is not the one that was uploaded.",
                    f"size on disk: {size} bytes")

        spec = next((i for i in (self.recipe.get("inputs") or [])
                     if i.get("role") == role), None)
        if spec is None:
            offered = ", ".join(str(i.get("role")) for i in
                               (self.recipe.get("inputs") or [])) or "(none)"
            raise WorkerFault("bad_request",
                              f"recipe {self.recipe.get('name')!r} declares no input "
                              f"with role {role!r}. It declares: {offered}")
        node_id = spec.get("node")

        from nodelab_v2.ingest import read_channel_display
        provider, envelope = self._provider_for(path)
        try:
            display = read_channel_display(path)
        except Exception as exc:      # noqa: BLE001 — display metadata is not load-bearing
            note(f"could not read channel display metadata: {exc}")
            display = {}

        self.inputs[role] = {"path": path, "name": cmd.get("name") or
                             os.path.basename(path), "sha256": declared,
                             "node": node_id, "display": display}
        axes = envelope.axes
        channels = list(display.get("channel_names") or
                        [f"Ch{i}" for i in range(int(getattr(axes, "c", 1)))])
        self._answer(
            mid, "result", ingested=True, role=role, node=node_id,
            axes={a: int(getattr(axes, a, 1)) for a in ("m", "t", "z", "c", "y", "x")},
            calibration={k: v for k, v in (envelope.metadata or {}).items()
                         if isinstance(v, (int, float, str, bool)) or v is None},
            channels=channels,
            store=self._store_path_for(path))

    def _store_path_for(self, path: str) -> str:
        """Where this file's ``.b2nd`` ingest store lives.

        Inside the SESSION directory, not beside the source. The hub owns the session
        directory's lifetime and its quota, and a worker writing a multi-gigabyte derived
        cache next to a node's uploaded file would put it somewhere nothing cleans up and
        nothing accounts for. ``NODEGRAPH_STORE_DIR`` still wins if an operator has
        pointed stores at fast media deliberately.
        """
        import hashlib
        env = os.environ.get("NODEGRAPH_STORE_DIR")
        base = env if env else os.path.join(self.session, "stores")
        os.makedirs(base, exist_ok=True)
        tag = hashlib.blake2b(path.lower().encode("utf-8"), digest_size=6).hexdigest()
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(base, f"{stem}.{tag}.b2nd_store")

    def _provider_for(self, path: str) -> Tuple[Any, Any]:
        """``(provider, envelope)`` for a file, ingesting once and caching for the session.

        This cache is a large part of what "warm" means: re-running with a different
        threshold must not re-decode the ND2.
        """
        hit = self._providers.get(path)
        if hit is not None:
            return hit
        from nodelab_v2.ingest import ingest_image

        store = self._store_path_for(path)
        label = os.path.basename(path)
        last = [0.0]

        # An ingest is minutes on a real mosaic and emits nothing the engine observer would
        # see, so without SOMETHING here the hub's silence timeout kills the session during
        # the one step that legitimately takes longest.
        #
        # It is reported as `progress` with its own phase rather than as a `node`, and the
        # distinction is not cosmetic. An ingest is not a step in the graph: the plan
        # describes the nodes, the hub resets done/total when the plan arrives but does NOT
        # reset its cached/computed counters, and `input` is a separate command that runs
        # BEFORE the plan. Reported as a node, the ingest's `done` therefore survived the
        # plan reset and every run came back claiming one more node computed than the graph
        # has (6 of a 5-node plan). A `progress` event still resets the silence clock —
        # which is what this is for — and lands in the hub's ledger without being counted
        # as a node nobody can point to.
        def on_progress(fraction: float, phase: str) -> None:
            now = time.monotonic()
            if now - last[0] < 0.5 and fraction < 1.0:
                return
            last[0] = now
            emit(id=self._current_id, ev="progress", phase="ingest", file=label,
                 done=int(fraction * 1000), total=1000, fraction=float(fraction),
                 note=str(phase))

        emit(id=self._current_id, ev="progress", phase="ingest", file=label,
             done=0, total=1000, note=f"reading {label}")
        t0 = time.perf_counter()
        provider, envelope = ingest_image(path, store, progress=on_progress)
        seconds = round(time.perf_counter() - t0, 3)
        emit(id=self._current_id, ev="progress", phase="ingest", file=label,
             done=1000, total=1000, fraction=1.0, note="ingested",
             seconds=seconds)
        note(f"ingested {label} in {seconds}s -> {store}")
        self._providers[path] = (provider, envelope)
        return provider, envelope

    # ── run ─────────────────────────────────────────────────────────────────────
    def cmd_run(self, mid: Any, cmd: dict) -> None:
        self._require_open()
        target = str(cmd.get("target") or self.recipe.get("target") or "")
        if not target:
            raise WorkerFault("bad_request", "run needs a 'target'")
        if target not in self.graph.nodes:
            raise WorkerFault(
                "bad_request",
                f"target {target!r} is not a node in this recipe's graph "
                f"({', '.join(sorted(self.graph.nodes))})")
        missing = [i.get("role") for i in (self.recipe.get("inputs") or [])
                   if i.get("required") and i.get("role") not in self.inputs]
        if missing:
            raise WorkerFault(
                "missing_input",
                f"recipe {self.recipe.get('name')!r} requires input(s) "
                f"{', '.join(str(m) for m in missing)}; none were sent")

        self.cancel.clear()
        self.runs += 1
        started = time.time()
        counters = {"cached": 0, "computed": 0, "failed": 0}
        seeds, meta_seeds = self._build_seeds()
        engine = self._engine_for(seeds, meta_seeds, counters)

        self._emit_plan(mid, engine, target)
        self._start_beats(mid)
        try:
            payload = engine.pull(target)
        except Cancelled as exc:
            raise WorkerFault("cancelled", str(exc) or "cancelled at a node boundary",
                              "") from None
        except FileNotFoundError as exc:
            raise WorkerFault("input_error", f"{exc}", traceback.format_exc()) from None
        except MemoryError:
            raise
        except BaseException as exc:      # noqa: BLE001 — a compute failure is a RESULT
            raise WorkerFault(
                "compute_error",
                f"node {target!r} failed: {type(exc).__name__}: {exc}",
                traceback.format_exc()) from None
        finally:
            self._stop_beats()

        artifacts = self._write_artifacts(payload, engine, counters, started)
        seconds = round(time.time() - started, 3)
        cached_run = counters["computed"] == 0 and counters["cached"] > 0
        note(f"run {self.runs}: {seconds}s, {counters['computed']} computed, "
             f"{counters['cached']} cached, {len(artifacts)} artifact(s)")
        self._answer(
            mid, "result", status="ok", seconds=seconds,
            compute_count=int(getattr(engine, "compute_count", 0)),
            cached=cached_run,
            artifacts=[a.name for a in artifacts],
            result=self._summary(payload, counters))

    def _build_seeds(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """A seed Dataset per ``io.load`` node — the uploaded file where one was sent, the
        synthetic demo stack where the loader's path is empty.

        The synthetic fallback is what lets a recipe be a genuine self-test: no data files
        at all, a deterministic source, a result that can be pinned by hash. It mirrors
        what the GUI runner does for an empty ``path`` field, so a recipe verified this way
        is verified against the same code path a real run takes.
        """
        from nodegraph.dataset import Dataset
        from nodegraph.metadata import MetaEnvelope
        from nodegraph.provider import SyntheticProvider
        from nodelab_v2.ops import LOAD_OP

        by_node = {v["node"]: v for v in self.inputs.values() if v.get("node")}
        seeds: Dict[str, Any] = {}
        meta_seeds: Dict[str, Any] = {}
        for nid, rec in self.graph.nodes.items():
            if rec.op_key != LOAD_OP:
                continue
            sent = by_node.get(nid)
            if sent is not None:
                provider, envelope = self._provider_for(sent["path"])
                metadata = dict(envelope.metadata)
                metadata.update(sent.get("display") or {})
            else:
                path = str(rec.params.get("path") or "").strip()
                if path:
                    # A recipe may legitimately pin a path the OPERATOR chose — that is
                    # hub-side config, not something a node supplied (the hub's knob
                    # whitelist forbids 'path' outright), so it is honoured.
                    provider, envelope = self._provider_for(os.path.abspath(path))
                    metadata = dict(envelope.metadata)
                else:
                    from nodegraph.dataset import AxisSizes
                    axes = AxisSizes(m=1, t=2, z=4, c=2, y=256, x=256)
                    provider = SyntheticProvider(axes, tile=128)
                    metadata = {"pixel_size_um": 0.325, "z_step_um": 1.0, "dt_s": 60.0,
                                "channel_names": ["Ch0", "Ch1"]}
                    envelope = MetaEnvelope(axes=axes, metadata=dict(metadata))
            seeds[nid] = Dataset(axes=envelope.axes,
                                 metadata=metadata).with_image(provider)
            meta_seeds[nid] = envelope
        return seeds, meta_seeds

    def _engine_for(self, seeds: Dict[str, Any], meta_seeds: Dict[str, Any],
                    counters: Dict[str, int]) -> Any:
        """Build (or reuse) the engine, with our observer and cancellable computes.

        The **memo and tile cache are kept across runs** and that is the entire economic
        argument for a hub: the second run after a knob change re-computes only what the
        knob invalidated. They are supplied explicitly rather than left to a fresh engine
        because a streaming provider holds its tile cache by weakref — a per-run cache
        would be collected the moment the engine was dropped, leaving every memo-hit lazy
        Dataset re-reading through no cache at all, forever.
        """
        from nodegraph.memo import Memo
        from nodegraph.streaming import TileCache
        from nodelab_v2.ops import headless_engine

        if self._memo is None:
            self._memo = Memo(budget_bytes=self.memo_bytes)
        if self._tiles is None:
            from nodegraph.parallel import tile_cache_bytes
            self._tiles = TileCache(self.cache_bytes if self.cache_bytes
                                    else tile_cache_bytes())
        engine = headless_engine(
            self.graph, seeds=seeds, meta_seeds=meta_seeds,
            memo=self._memo, tiles=self._tiles,
            observer=self._observer(counters))
        engine.computes = self._cancellable_computes(engine.computes)
        return engine

    def _observer(self, counters: Dict[str, int]) -> Callable[..., None]:
        """The engine's ``Observer`` → LWP ``node`` events.

        The five engine event names are the five LWP node states, so this is a dict merge
        and nothing is renamed or lost — which is exactly what
        ``lablink/protocol.py`` claims about them and is worth keeping true.

        One thing IS added: ``lazy=True`` on a ``done`` that took under a millisecond. A
        node returning a lazy provider finishes instantly and does its real work later,
        when a consumer reads planes; without the flag a UI renders it as complete above a
        session that then sits silent for a minute with no explanation.

        Two things are FILTERED, both so ``done``/``total`` keeps meaning what it says —
        see the comments at each guard.
        """
        labels = self._node_labels()
        reported: set = set()

        def observe(event: str, node_id: str, info: Dict[str, Any]) -> None:
            if self._muted:
                # Collecting an output pulls a node OTHER than the target, which re-walks
                # its ancestors and emits a `cached` for each. The hub increments its
                # progress counter on every done/cached without asking whether the node is
                # in the plan, so unmuted this reported "15 of 5 steps" on a five-node
                # graph with four outputs — and the same double-count landed in the
                # metrics blob as nodes_cached. The plan describes the TARGET's chain, so
                # events from an output pull do not belong in it.
                return
            if event in ("cached", "done", "error"):
                # ONE terminal per plan step. A node feeding two consumers is visited
                # twice — this graph's `pick` goes to both `tophat` and `measure.raw`, so
                # the walk reaches it via the segment branch (computing it) and again
                # directly (a memo hit, reported `cached`). Both reports are true, and the
                # second is not a new STEP: the hub increments its counter on every
                # terminal against a total that counts distinct nodes, so passing the
                # repeat through read "7 of 5 steps" on this five-node graph. Suppressing
                # it loses nothing an operator wants — the node's real outcome was already
                # reported — and it is what keeps done/total meaning what it says.
                if node_id in reported:
                    return
                reported.add(node_id)
                counters["cached" if event == "cached" else
                         "computed" if event == "done" else "failed"] += 1
            payload = dict(info)
            seconds = payload.get("seconds")
            if event == "done" and isinstance(seconds, (int, float)) and seconds < 0.001:
                payload["lazy"] = True
            if isinstance(seconds, (int, float)):
                payload["seconds"] = round(float(seconds), 4)
            payload.setdefault("label", labels.get(node_id, node_id))
            emit(id=self._current_id, ev="node", node=node_id, state=event, **payload)

        return observe

    def _node_labels(self) -> Dict[str, str]:
        """``{node_id: human label}`` from the registry, for the plan and every event."""
        from nodegraph.registry import NODES
        out: Dict[str, str] = {}
        for nid, rec in self.graph.nodes.items():
            spec = NODES.get(rec.op_key)
            title = str((rec.params or {}).get("__title__") or "").strip()
            out[nid] = title or (spec.label if spec is not None else rec.op_key)
        return out

    def _emit_plan(self, mid: Any, engine: Any, target: str) -> None:
        """The steps this run will walk, before it walks them.

        ``kind`` is a **forecast for display**, and the hub uses it as one: real
        determinacy comes from whether a ``progress`` event carries a ``total``, never from
        this. It is still worth sending — a node listed as ``indeterminate`` up front stops
        an operator reading a still bar as a stall.
        """
        graph = engine.graph
        labels = self._node_labels()
        order = [nid for nid in graph.topo_order()
                 if self._reaches(graph, nid, target)]
        plan = []
        for nid in order:
            rec = graph.nodes[nid]
            plan.append({"node": nid, "op_key": rec.op_key,
                         "label": labels.get(nid, nid),
                         "kind": self._forecast_kind(rec, engine)})
        emit(id=mid, ev="progress", phase="plan", done=0, total=len(plan), nodes=plan,
             target=target)

    @staticmethod
    def _reaches(graph: Any, node_id: str, target: str) -> bool:
        """Whether ``node_id`` is an ancestor of ``target`` (or is it) — i.e. whether this
        run will touch it. A saved graph routinely has branches the target does not need,
        and listing them in the plan would make every run look permanently incomplete."""
        seen = set()
        stack = [target]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for edge in graph.preds(cur):
                stack.append(edge.src)
        return node_id in seen

    @staticmethod
    def _forecast_kind(rec: Any, engine: Any) -> str:
        from nodegraph.registry import Granularity, NODES
        spec = NODES.get(rec.op_key)
        if spec is None:
            return "indeterminate"
        if rec.op_key not in engine.computes:
            return "instant"                 # a seeded source: no compute to watch
        try:
            gran = spec.resolve_granularity(rec.state(spec))
        except Exception:       # noqa: BLE001 — a forecast must never fail a run
            return "indeterminate"
        # The per-unit granularities are the ones whose computes loop over frames/planes
        # and call ctx.progress; everything else is one opaque call as far as we can know.
        if gran in (Granularity.WHOLE_PLANE, Granularity.WHOLE_VOLUME,
                    Granularity.TILEABLE, Granularity.WHOLE_SERIES):
            return "determinate"
        return "indeterminate"

    def _cancellable_computes(self, computes: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap the compute table so a cancel lands at the next node that has not started.

        The engine swallows observer exceptions on purpose — "a progress sink must never
        break a run" — so the observer cannot be the cancellation point. The compute table
        can: it is a plain dict on the engine, each entry called once per node evaluation,
        and raising from it propagates out of ``pull`` verbatim after the engine has
        emitted that node's ``error``. Which is exactly the semantics we advertise.
        """
        def wrap(fn: Any) -> Any:
            def guarded(ctx: Any) -> Any:
                if self.cancel.is_set():
                    raise Cancelled(
                        f"cancelled before {getattr(ctx, 'node_id', '?')!r} "
                        f"({getattr(ctx, 'op_key', '?')})")
                return fn(ctx)
            return guarded

        return {key: wrap(fn) for key, fn in computes.items()}

    # ── beats ───────────────────────────────────────────────────────────────────
    def _start_beats(self, mid: Any) -> None:
        """Emit a ``beat`` every few seconds for as long as a run is in flight.

        This is the difference between a silence timeout that kills wedged workers and one
        that kills working ones. The hub's clock is reset only by a PARSED protocol line,
        and a single opaque compute — one CNN inference, one upstream solver with no
        callback — emits nothing for as long as it takes. The beat says "alive, and here is
        what I am holding" without claiming any progress it cannot measure.
        """
        stop = threading.Event()
        self._beat_stop = stop

        def pulse() -> None:
            while not stop.wait(P.BEAT_INTERVAL_S):
                info: Dict[str, Any] = {"busy": True}
                rss = _rss_bytes()
                if rss is not None:
                    info["rss_bytes"] = rss
                if self._memo is not None:
                    info["memo_bytes"] = int(getattr(self._memo, "nbytes", 0) or 0)
                if self._tiles is not None:
                    info["cache_bytes"] = int(getattr(self._tiles, "nbytes", 0) or 0)
                emit(id=mid, ev="beat", **info)

        threading.Thread(target=pulse, name="lwp-beat", daemon=True).start()

    def _stop_beats(self) -> None:
        if self._beat_stop is not None:
            self._beat_stop.set()
            self._beat_stop = None

    # ── artifacts ───────────────────────────────────────────────────────────────
    def _write_artifacts(self, payload: Any, engine: Any, counters: Dict[str, int],
                         started: float) -> List[Any]:
        """Produce every output the recipe declares, announcing each as it lands.

        An output whose source node differs from the target is pulled separately — its
        chain is already in the memo, so this is a hit rather than a second run. An output
        that cannot be produced (a quicklook of a table, a table with nothing tabulatable)
        is REPORTED and skipped, never faked: "the threshold found no objects" is an
        ordinary scientific result and must not look like a broken pipeline.
        """
        from nodelab_v2.lablink import artifacts as A

        target = str(self.recipe.get("target") or "")
        stem = self._stem()
        out: List[Any] = []
        for spec in self.recipe.get("outputs") or []:
            name = str(spec.get("name") or "output")
            kind = str(spec.get("kind") or "")
            policy = str(spec.get("policy") or "auto")
            if policy == "never":
                continue
            filename = self._render_filename(spec.get("filename"), name, stem)
            if kind == "metrics":
                art = A.write_metrics(self._metrics(engine, counters, started),
                                      self.out_dir, filename, policy=policy)
            else:
                writer = A.WRITERS.get(kind)
                if writer is None:
                    note(f"output {name!r}: no writer for kind {kind!r}; skipped")
                    continue
                source = str(spec.get("from") or "") or target
                data = payload if source == target else self._pull_quietly(engine, source)
                if data is None:
                    note(f"output {name!r}: could not pull {source!r}; skipped")
                    continue
                art = writer(data, self.out_dir, filename, policy=policy)
            if art is None:
                emit(id=self._current_id, ev="log", level="info",
                     message=f"output {name!r} ({kind}) produced nothing for this run — "
                             f"the payload carries no data of that kind. This is a "
                             f"result, not a failure.")
                continue
            if art.policy == "pull":
                self.held[art.name] = art
            out.append(art)
            emit(id=self._current_id, ev="artifact", **art.event())
        return out

    def _pull_quietly(self, engine: Any, node_id: str) -> Any:
        """Pull a node an output needs, without it appearing in the plan's progress.

        Muted for the reason in :meth:`_observer`, and in a ``try``/``finally`` so a failed
        output cannot leave the observer silent for the rest of the session.
        """
        if node_id not in engine.graph.nodes:
            return None
        self._muted = True
        try:
            return engine.pull(node_id)
        except BaseException as exc:      # noqa: BLE001 — one output must not fail the run
            note(f"could not pull {node_id!r} for an output: {type(exc).__name__}: {exc}")
            return None
        finally:
            self._muted = False

    def _metrics(self, engine: Any, counters: Dict[str, int],
                 started: float) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "recipe": self.recipe.get("name"),
            "run": self.runs,
            "seconds": round(time.time() - started, 3),
            "compute_count": int(getattr(engine, "compute_count", 0)),
            "nodes_computed": counters["computed"],
            "nodes_cached": counters["cached"],
            "nodes_failed": counters["failed"],
            "knobs": dict(self.knobs),
            "software": P.SOFTWARE_NAME,
            "software_version": P.SOFTWARE_VERSION,
            "worker_version": P.WORKER_VERSION,
        }
        rss = _rss_bytes()
        if rss is not None:
            info["rss_bytes"] = rss
        for role, sent in self.inputs.items():
            info.setdefault("inputs", {})[role] = {
                "name": sent.get("name"), "sha256": sent.get("sha256")}
        return info

    def _summary(self, payload: Any, counters: Dict[str, int]) -> Dict[str, Any]:
        """A few numbers about the payload, for the node's own log.

        Deliberately small: the tables are an artifact, and duplicating them into the
        result would be exactly the "bulk data on the pipe" the protocol forbids.
        """
        from nodelab_v2.tables import all_tables
        try:
            tables = all_tables(payload)
        except Exception:       # noqa: BLE001
            tables = {}
        rows = {f"{dom}/{layer or ''}": int(max((len(v) for v in cols.values()),
                                                default=0))
                for (dom, layer), cols in tables.items()}
        axes = getattr(payload, "axes", None)
        return {
            "tables": rows,
            "total_rows": int(sum(rows.values())),
            "axes": ({a: int(getattr(axes, a, 1)) for a in ("m", "t", "z", "c", "y", "x")}
                     if axes is not None else {}),
            "nodes_computed": counters["computed"],
            "nodes_cached": counters["cached"],
        }

    def _stem(self) -> str:
        sent = self.inputs.get("image") or next(iter(self.inputs.values()), None)
        if sent and sent.get("name"):
            return os.path.splitext(os.path.basename(str(sent["name"])))[0]
        return str(self.recipe.get("name") or "run")

    def _render_filename(self, template: Any, name: str, stem: str) -> str:
        """Fill the recipe's filename placeholders. Unknown ones are left alone — the hub
        validated the template at tier 1, so anything surviving here is a hub we are newer
        than, and mangling the name would be worse than passing it through."""
        text = str(template or name)
        for key, value in (("stem", stem), ("name", name),
                           ("recipe", str(self.recipe.get("name") or "recipe")),
                           ("run", str(self.runs)), ("output", name),
                           ("session", os.path.basename(self.session)),
                           ("job", str(self.runs))):
            text = text.replace("{%s}" % key, value)
        return text

    # ── pull / reset / close ────────────────────────────────────────────────────
    def cmd_pull(self, mid: Any, cmd: dict) -> None:
        """Hand over an artifact the recipe held back.

        With no name, the oldest held artifact; the hub's own ``/s/{id}/pull`` with no
        names asks for everything held, and it makes one request per artifact.
        """
        self._require_open()
        name = cmd.get("name") or cmd.get("artifact")
        if name:
            art = self.held.get(str(name))
            if art is None:
                offered = ", ".join(sorted(self.held)) or "(none held)"
                raise WorkerFault("bad_request",
                                  f"no held artifact named {str(name)!r}. Held: {offered}")
        else:
            if not self.held:
                raise WorkerFault("bad_request", "this session is holding no artifacts")
            art = self.held[sorted(self.held)[0]]
        self._answer(mid, "result", artifact=art.event())

    def cmd_reset(self, mid: Any, cmd: dict) -> None:
        """Drop the caches but stay alive — the rung between "run again" and "restart".

        Ingested providers are dropped too, and that is the point of asking: it is the
        largest thing a session holds. Re-running afterwards is correct but pays the decode
        again, so this is a deliberate operator action, not something to do routinely.
        """
        self._require_open()
        freed = 0
        if self._memo is not None:
            freed += int(getattr(self._memo, "nbytes", 0) or 0)
        if self._tiles is not None:
            freed += int(getattr(self._tiles, "nbytes", 0) or 0)
        self._memo = None
        self._tiles = None
        self._providers.clear()
        import gc
        gc.collect()
        note(f"reset: dropped ~{freed} bytes of memo + tile cache and every provider")
        self._answer(mid, "result", freed_bytes=freed, providers_dropped=True)

    def cmd_close(self, mid: Any, cmd: dict) -> None:
        self._answer(mid, "result", status="closing", runs=self.runs)
        self.stop = True

    # ── helpers ─────────────────────────────────────────────────────────────────
    def _require_open(self) -> None:
        if self.graph is None:
            raise WorkerFault("not_ready",
                              "no recipe is open on this session; the hub must send "
                              "'open' before anything else")


def _version_of(module: str) -> str:
    try:
        import importlib
        return str(getattr(importlib.import_module(module), "__version__", "") or "")
    except Exception:       # noqa: BLE001
        return ""


def _as_bytes(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# ── entry point ─────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="nd2studios-worker",
        description="Serve one LabLink session over LWP/1 (NDJSON on stdin/stdout).")
    ap.add_argument("--session", default="",
                    help="the session directory the hub allocated. Falls back to "
                         f"${P.SESSION_DIR_ENV}, then the current directory.")
    ap.add_argument("--print-hello", action="store_true",
                    help="emit the handshake and exit — the cheapest check that this "
                         "machine can serve LabLink at all.")
    args = ap.parse_args(argv)

    session = args.session or os.environ.get(P.SESSION_DIR_ENV) or os.getcwd()
    try:
        os.makedirs(session, exist_ok=True)
    except OSError as exc:
        note(f"cannot use session directory {session!r}: {exc}")
        return 4

    # Heavy imports BEFORE the handshake, so `hello` means ready. A failure here exits
    # non-zero with the reason on stderr; the hub reports that as `worker_exited` with our
    # stderr tail attached, which is how a missing dependency should read.
    try:
        import numpy         # noqa: F401
        import nodegraph.catalog     # noqa: F401 — registers the whole node catalog
        from nodelab_v2.ops import ensure_ops
        ensure_ops()
    except BaseException as exc:      # noqa: BLE001
        note(f"ND2Studios failed to load: {type(exc).__name__}: {exc}")
        note(traceback.format_exc())
        return 3

    worker = Worker(session)
    worker.hello()
    if args.print_hello:
        return 0
    worker.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
