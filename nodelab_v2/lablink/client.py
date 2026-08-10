"""Send this machine's work to somebody else's LabLink hub — the client half of the mode.

    hub = HubClient("http://10.132.157.104:8765", token="…")
    with hub.open_session("nd2studios", "selftest-synthetic",
                          knobs={"background_um": 6.0}) as s:
        ref = s.send_data(r"D:\\scans\\2026-08-03 A1 (well 3) 60x.nd2")
        for level in ("otsu", "li", "yen"):
            r = s.run(inputs=[ref], knobs={"level": level}, cmd_id=f"a1-{level}",
                      on_progress=lambda p: print(p.text), check=False)
            print(r.state, r.duration_s, r.cached_steps, "cached")
        s.pull()
        for path in s.fetch_all("inbox"):
            print("got", path)

**Standard library only, on purpose.** LabLink's own ``lablink.session_client`` would do
this, and importing it would make LabLink a dependency of ND2Studios — defeating the
property that lets LabLink be dropped onto an instrument PC untouched. The protocol is
seven HTTP calls; re-implementing them costs less than coupling the two repos, and it means
the editor can talk to a hub with nothing installed but ND2Studios itself.

Everything on QUICKSTART §10's "checklist for a session client" is implemented here, and
the ones that are counter-intuitive are the ones with comments:

* ``capabilities.sessions`` is checked before any of this is assumed to exist.
* ``in_channel``/``out_channel`` come from the open response and are **never derived from
  the id**, so a hub free to change its channel layout stays free.
* ``since_event`` advances to the newest ``event_seq`` on every poll — a lower one returns
  instantly with old news and turns a long poll into a hot loop.
* The ready poll **loops**: one 25 s wait can expire while a large program is still loading.
* The socket timeout sits strictly ABOVE ``longpoll_max_s``, or every successful long poll
  looks like a dead server.
* Every input is named with its sha256, because a command that races an upload would
  otherwise run against a different file of the same name and nobody would know.
* Knobs are range-checked against ``/workflows`` before anything is sent, so a bad value is
  an instant local error rather than a wasted forty-minute run.
* Knobs declaring ``unset_means: "derive"`` are **left out**, never pinned — pinning one
  overrides the file's own optics silently.
* ``503`` is retried with backoff (a full hub clears in seconds); ``409 busy`` is not.
* ``410`` is never retried with the same id.
* A **failed command does not end the session** — "the threshold found no objects" is a
  scientific result, and the warm software is still there.
* ``DELETE /s/{id}`` runs in a ``finally``, on the error path as much as the happy one.

Two things QUICKSTART calls deliberate simplifications in its example are done properly
here, because this client is aimed at microscopy files: uploads **stream** from an open
file handle rather than being read into memory, and downloads **resume** with ``Range``
into a ``.partial`` and are checksum-verified before being moved into place.

Qt-free; standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from nodelab_v2.lablink import protocol as P

#: Bytes per read when streaming an upload or a download.
CHUNK = 1 << 16

#: How long to keep retrying a hub that answers 503 (at its session limit).
CAPACITY_RETRY_S = 120.0

#: Ceiling on the backoff between 503 retries.
MAX_BACKOFF_S = 15.0


class LabLinkError(Exception):
    """Any refusal from a hub. Carries the HTTP status and the hub's own code."""

    def __init__(self, message: str, *, status: int = 0, code: str = "",
                 detail: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code
        self.detail = detail

    def __str__(self) -> str:
        # A status of 0 means "we diagnosed this locally, no HTTP status involved" — a
        # session the hub reported as failed inside an otherwise-200 command poll, say.
        # Printing "[0 session_gone]" made those read like a transport fault.
        bits = [super().__str__()]
        tag = " ".join(str(p) for p in (self.status or "", self.code) if p)
        if tag:
            bits.insert(0, f"[{tag}]")
        return " ".join(bits)


class SessionLost(LabLinkError):
    """The session is gone — closed, failed, expired, cancelled or lost.

    Distinct from :class:`CommandFailed` because the consequence differs entirely: the warm
    cache went with it, so the next command would be silently slower and possibly
    different. Open a new session; never retry this id.
    """


class CommandFailed(LabLinkError):
    """One command failed. The session is still usable — change a knob and run again."""


class HubBusy(LabLinkError):
    """A command is already running on this session (``409 busy``).

    Refused rather than queued, deliberately: a hidden backlog would leave no way to reason
    about latency and no way to change your mind. Wait for it, cancel it, or open a second
    session.
    """


@dataclass(frozen=True)
class FileRef:
    """An uploaded input, and the hash the hub will verify it against."""

    name: str                 # the name ON THE CHANNEL (possibly repaired)
    sha256: str
    size: int
    original_name: str = ""

    @property
    def repaired(self) -> bool:
        """Whether the exchange's name rules forced a rename."""
        return bool(self.original_name) and self.original_name != self.name

    def as_input(self) -> Dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True)
class Progress:
    """One progress snapshot, and an honest one-line rendering of it."""

    done: int = 0
    total: int = 0
    cached: int = 0
    computed: int = 0
    current: str = ""
    current_label: str = ""
    fraction: Optional[float] = None
    note: str = ""
    determinate: bool = False
    plan: Tuple[Dict[str, Any], ...] = ()

    @property
    def text(self) -> str:
        """A line safe to show a user.

        ``determinate: false`` means the percentage is not known, so none is shown — a bar
        that moves without a denominator is the UI lying. Steps and the current label are
        always meaningful, so those are what is shown instead.
        """
        head = f"{self.done}/{self.total}" if self.total else f"{self.done} step(s)"
        label = self.current_label or self.current
        pct = (f" {self.fraction * 100:.0f}%"
               if self.determinate and self.fraction is not None else "")
        note = f" — {self.note}" if self.note else ""
        cached = f" ({self.cached} cached)" if self.cached else ""
        return f"{head}{pct} {label}{note}{cached}".strip()

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "Progress":
        raw = raw or {}
        return cls(
            done=int(raw.get("done") or 0), total=int(raw.get("total") or 0),
            cached=int(raw.get("cached") or 0), computed=int(raw.get("computed") or 0),
            current=str(raw.get("current") or ""),
            current_label=str(raw.get("current_label") or ""),
            fraction=raw.get("fraction"),
            note=str(raw.get("note") or ""),
            determinate=bool(raw.get("determinate")),
            plan=tuple(raw.get("plan") or ()))


@dataclass
class CommandResult:
    """The terminal state of one command, with everything needed to log what ran."""

    cmd_seq: int
    cmd_id: str
    state: str
    duration_s: Optional[float]
    knobs: Dict[str, Any] = field(default_factory=dict)
    progress: Progress = field(default_factory=Progress)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.state == "done"

    @property
    def code(self) -> str:
        return str((self.error or {}).get("code") or "")

    @property
    def message(self) -> str:
        return str((self.error or {}).get("message") or "")

    @property
    def cached_steps(self) -> int:
        return self.progress.cached

    def held(self) -> List[str]:
        """Artifact names the recipe held back (``policy: "pull"``)."""
        return [str(a.get("name")) for a in self.artifacts if a.get("held")]

    def returned(self) -> List[str]:
        """Names to fetch by — ``returned_as`` where the hub had to repair one."""
        return [str(a.get("returned_as") or a.get("name"))
                for a in self.artifacts if a.get("published")]


def repair_name(name: str) -> str:
    """A filename the exchange will accept.

    Its name rules refuse characters real microscopy filenames are full of — parentheses,
    most obviously, as in ``2026-08-03 A1 (well 3) 60x.nd2``. Substituting ``_`` is what
    QUICKSTART §10 prescribes, and the original is recorded in ``X-Lablink-Meta`` so a
    person can still tell which file it is. A name must also START alphanumeric, and one
    that reduces to nothing gets a usable stand-in rather than an empty string the hub
    would refuse for a second, more confusing reason.
    """
    cleaned = "".join(c if c in P.SAFE_NAME_CHARS else "_" for c in os.path.basename(name))
    cleaned = cleaned.strip(" .")            # Windows silently strips these
    while cleaned and not cleaned[0].isalnum():
        cleaned = cleaned[1:]
    return cleaned[:128] or "upload.bin"


def sha256_of(path: str) -> Tuple[str, int]:
    """``(digest, size)`` streamed in chunks, so a 4 GB stack costs one buffer."""
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
            total += len(block)
    return h.hexdigest(), total


def _same_value(a: Any, b: Any) -> bool:
    """Equality for a knob condition, strict about ``bool`` versus number.

    Python's ``1 == True`` is the accident that makes a hub comparing loosely and a client
    comparing strictly disagree about whether a condition holds — and the recipe validator
    refuses ``"equals": 1`` against a bool knob for exactly this reason. Comparing the
    bool-ness first is what keeps this side on the strict reading.
    """
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return bool(a == b)


def _condition_text(cond: Mapping[str, Any]) -> str:
    """An ``applies_when`` rendered for a person: ``"2D"`` or ``one of "2D", "3D"``."""
    if "equals" in cond:
        return repr(cond.get("equals"))
    values = cond.get("in") or ()
    return "one of " + ", ".join(map(repr, values)) if values else "(unstated)"


def knob_applies(spec: Mapping[str, Any], values: Mapping[str, Any],
                 declared: Mapping[str, Mapping[str, Any]]) -> bool:
    """Whether a knob's ``applies_when`` condition currently holds.

    Some knobs are inert under some settings of another — the engine filters by area in 2D
    and by *volume* in 3D, so a minimum-area value on a 3D run is read by nothing. A recipe
    declares that, and the hub refuses a pinned value for an unmet one rather than ignoring
    it, because a value that silently does nothing looks exactly like one that worked.

    Shared by the validator and the GUI on purpose: the control that greys out and the check
    that refuses must agree, or the dock disables a knob the hub would have accepted (or
    worse, offers one it will refuse after the upload).

    The controlling knob's value is what the caller set, falling back to the controller's own
    declared ``default`` — which every controller is required to have, precisely so this is
    answerable before anything has been sent.
    """
    cond = spec.get("applies_when")
    if not isinstance(cond, Mapping) or not cond:
        return True
    controller = str(cond.get("knob") or "")
    ctrl_spec = declared.get(controller)
    if ctrl_spec is None:
        # A condition naming a knob this recipe does not declare is a recipe bug that tier 1
        # refuses at hub start. Not ours to fail on: let the hub speak.
        return True
    current = values.get(controller)
    if current is None:
        current = ctrl_spec.get("default")
    if "equals" in cond:
        return _same_value(current, cond.get("equals"))
    if "in" in cond:
        return any(_same_value(current, v) for v in (cond.get("in") or ()))
    return True


class HubClient:
    """Discovery and transport against one hub. Cheap to construct; holds no connection."""

    def __init__(self, url: str, token: str, *, node_id: str = "",
                 timeout: float = P.CLIENT_TIMEOUT_S):
        self.url = url.rstrip("/")
        self.token = token
        self.node_id = node_id
        # Strictly above the hub's long-poll ceiling. Below it, every SUCCESSFUL long poll
        # would time out client-side and read as an unreachable hub.
        self.timeout = max(float(timeout), P.LONGPOLL_MAX_S + 5.0)
        self._hello: Dict[str, Any] = {}

    # ── transport ───────────────────────────────────────────────────────────────
    def call(self, method: str, path: str, *, body: Any = None,
             raw: Optional[bytes] = None, stream: Optional[Any] = None,
             stream_len: int = 0, headers: Optional[Dict[str, str]] = None,
             timeout: Optional[float] = None, **query: Any) -> Tuple[int, Any]:
        """One request. Returns ``(status, parsed-json-or-bytes)`` and never raises for an
        HTTP status — every caller here branches on the code, and the hub's error bodies
        are the most useful thing it sends.
        """
        if query:
            path += "?" + urllib.parse.urlencode(
                {k: v for k, v in query.items() if v is not None})
        data: Any = None
        if stream is not None:
            data = stream
        elif raw is not None:
            data = raw
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header(P.H_TOKEN, self.token)
        if self.node_id:
            # Only once ENROLLED: sending a node_id the hub does not know is a 401, not a
            # quiet fallback to the site token.
            req.add_header(P.H_NODE, self.node_id)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        if stream is not None:
            # urllib will not chunk for us; without an explicit length it would try to read
            # the whole file to measure it, which is exactly the memory this avoids.
            req.add_header("Content-Length", str(int(stream_len)))
            req.add_header("Content-Type", "application/octet-stream")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                payload = resp.read()
                ctype = resp.headers.get("Content-Type", "")
                parsed: Any = (json.loads(payload)
                               if payload and "json" in ctype else payload)
                return resp.status, parsed
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            try:
                return exc.code, json.loads(payload)
            except ValueError:
                return exc.code, {"error": payload[:400].decode("utf-8", "replace")}
        except urllib.error.URLError as exc:
            raise LabLinkError(
                f"cannot reach the hub at {self.url}: {exc.reason}. Check the address, "
                f"that the hub is running, and that this machine can route to it "
                f"(a Tailscale address needs Tailscale up on both ends).") from None

    def _json(self, method: str, path: str, *, expect: Iterable[int] = (200,),
              **kwargs: Any) -> Dict[str, Any]:
        status, payload = self.call(method, path, **kwargs)
        if status in expect:
            return payload if isinstance(payload, dict) else {}
        raise self._error_for(status, payload, f"{method} {path}")

    @staticmethod
    def _error_for(status: int, payload: Any, where: str) -> LabLinkError:
        info = payload if isinstance(payload, dict) else {}
        code = str(info.get("code") or "")
        message = str(info.get("error") or info.get("message") or f"{where} failed")
        detail = str(info.get("detail") or "")
        if status == P.HTTP_GONE:
            return SessionLost(message, status=status, code=code or "session_gone",
                               detail=detail)
        if status == P.HTTP_CONFLICT and code == "busy":
            return HubBusy(message, status=status, code=code, detail=detail)
        return LabLinkError(message, status=status, code=code, detail=detail)

    # ── discovery ───────────────────────────────────────────────────────────────
    def hello(self, *, refresh: bool = False) -> Dict[str, Any]:
        """The hub's identity and capability block."""
        if self._hello and not refresh:
            return self._hello
        self._hello = self._json("GET", "/hello")
        return self._hello

    def supports_sessions(self) -> bool:
        """Whether this server offers sessions at all.

        No ``capabilities`` block, or ``sessions: false``, means it is a file exchange only
        — checked before any of §10 is assumed, because the alternative is a 404 that reads
        as a typo.
        """
        caps = (self.hello().get("capabilities") or {})
        return bool(caps.get("sessions"))

    def workflows(self) -> List[Dict[str, Any]]:
        """Every workflow the operator has declared, with its recipes and their knobs."""
        return list(self._json("GET", "/workflows").get("workflows") or [])

    def recipe_of(self, workflow: str, recipe: str) -> Dict[str, Any]:
        """One recipe's metadata, or a refusal naming what IS on offer.

        Also surfaces ``unusable_recipes``: a recipe the operator installed that failed
        validation is listed rather than hidden, because "the recipe I was told to use is
        not in the list" is otherwise unanswerable from this side.
        """
        for wf in self.workflows():
            if wf.get("name") != workflow:
                continue
            for rec in wf.get("recipes") or []:
                if rec.get("name") == recipe:
                    return rec
            broken = {str(u.get("name")): str(u.get("reason") or "")
                      for u in (wf.get("unusable_recipes") or [])
                      if isinstance(u, dict)}
            if recipe in broken:
                raise LabLinkError(
                    f"the hub has recipe {recipe!r} but it failed validation on the hub: "
                    f"{broken[recipe]}")
            offered = ", ".join(str(r.get("name")) for r in (wf.get("recipes") or []))
            raise LabLinkError(
                f"workflow {workflow!r} has no recipe {recipe!r}. It offers: "
                f"{offered or '(none usable)'}"
                + (f". Unusable: {', '.join(broken)}" if broken else ""))
        offered = ", ".join(str(w.get("name")) for w in self.workflows())
        raise LabLinkError(f"this hub has no workflow {workflow!r}. It offers: "
                           f"{offered or '(none)'}")

    def recipe_schema(self) -> Dict[str, Any]:
        """The hub's own recipe vocabulary — every field, closed set and rule.

        Assembled on the hub from its validator's constants, so it cannot drift from what
        will actually be accepted. A recipe generator should build against this rather than
        against a copy of the prose; :mod:`nodelab_v2.lablink.recipe` falls back to a
        vendored snapshot when there is no hub to ask, and says which it used.
        """
        return self._json("GET", "/recipe-schema")

    # ── identity ────────────────────────────────────────────────────────────────
    def enroll(self, *, label: str = "") -> str:
        """Mint a node identity for this machine and return its id.

        **Enrol once per machine and persist the result.** Enrolling on every launch fills
        the operator's node list with junk entries they cannot tell apart, and the whole
        point of a node identity is that an operator can revoke exactly one machine.

        The returned id goes in :data:`~nodelab_v2.lablink.protocol.H_NODE` alongside the
        token; this sets :attr:`node_id` so subsequent calls on this client present it.
        """
        caps = (self.hello().get("capabilities") or {})
        if not caps.get("enroll"):
            raise LabLinkError(
                f"{self.url} does not offer enrolment, so this machine cannot have its own "
                f"revocable identity there. Use the site token alone.")
        body: Dict[str, Any] = {}
        if label:
            body["label"] = label
        doc = self._json("POST", "/enroll", body=body, expect=(200, P.HTTP_OPENED))
        node = str(doc.get("node") or doc.get("id") or "")
        if not node:
            raise LabLinkError("the hub accepted the enrolment but named no node id")
        self.node_id = node
        return node

    # ── the file exchange, outside any session ──────────────────────────────────
    def put_file(self, channel: str, path: str, *, name: str = "",
                 meta: Optional[Dict[str, Any]] = None) -> FileRef:
        """Upload one file to a named channel, streaming from disk.

        This is the plain exchange upload — the same call a session's :meth:`Session.send_data`
        makes, without a session. Submitting a recipe uses it, which is why recipe submission
        needs no new protocol: it inherits hashing, atomic visibility, resume, the size cap
        and this machine's identity in the hub's ledger.
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise LabLinkError(f"no such file to send: {path}")
        original = name or os.path.basename(path)
        safe = repair_name(original)
        digest, size = sha256_of(path)
        blob = {"original_name": original}
        blob.update(meta or {})
        headers = {P.H_SHA: digest, P.H_META: json.dumps(blob)}
        max_bytes = self.hello().get("max_file_bytes")
        if isinstance(max_bytes, int) and max_bytes and size > max_bytes:
            # Pre-checkable, unlike a session quota — so check it rather than uploading for
            # ten minutes to be refused at the end.
            raise LabLinkError(
                f"{os.path.basename(path)} is {size} bytes; this hub accepts at most "
                f"{max_bytes}.")
        with open(path, "rb") as fh:
            status, body = self.call(
                "PUT", f"/c/{urllib.parse.quote(channel)}/{urllib.parse.quote(safe)}",
                stream=fh, stream_len=size, headers=headers,
                timeout=max(self.timeout, 60.0 + size / (1 << 20)))
        if status not in (200, P.HTTP_OPENED):
            raise self._error_for(status, body, f"upload to {channel}")
        return FileRef(name=safe, sha256=digest, size=size, original_name=original)

    # ── sessions ────────────────────────────────────────────────────────────────
    def open_session(self, workflow: str, recipe: str, *, label: str = "",
                     knobs: Optional[Dict[str, Any]] = None,
                     ready_timeout_s: float = 300.0,
                     validate: bool = True) -> "Session":
        """Open a session and block until its worker is ready.

        A ``503`` is retried with backoff for up to :data:`CAPACITY_RETRY_S`: a hub at its
        session limit clears in seconds, and to the machine asking, a full hub is otherwise
        indistinguishable from a broken one.
        """
        if not self.supports_sessions():
            raise LabLinkError(
                f"{self.url} is a LabLink file exchange, not a hub: its /hello reports no "
                f"session capability, so there is nothing to open a session on.")
        meta = self.recipe_of(workflow, recipe) if validate else {}
        resolved = (self.resolve_for_send(meta, knobs) if validate
                    else dict(knobs or {}))
        payload = {"workflow": workflow, "recipe": recipe, "knobs": resolved}
        if label:
            payload["label"] = label

        deadline = time.monotonic() + CAPACITY_RETRY_S
        backoff = 1.0
        while True:
            status, body = self.call("POST", "/s", body=payload)
            if status == P.HTTP_OPENED:
                break
            if status == P.HTTP_AT_CAPACITY and time.monotonic() < deadline:
                wait = min(backoff, MAX_BACKOFF_S)
                time.sleep(wait)
                backoff *= 2
                continue
            raise self._error_for(status, body, "POST /s")
        session = Session(self, body if isinstance(body, dict) else {}, recipe_meta=meta,
                          knobs=resolved)
        session.wait_ready(timeout_s=ready_timeout_s)
        return session

    # ── knob validation ─────────────────────────────────────────────────────────
    @staticmethod
    def declared_knobs(recipe_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """``{name: spec}`` for a recipe's knobs, in the order the recipe declared them."""
        return {str(k.get("name")): dict(k)
                for k in (recipe_meta.get("knobs") or []) if k.get("name")}

    @staticmethod
    def check_knobs(recipe_meta: Dict[str, Any],
                    knobs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Range-check knobs against the recipe's own declarations, before sending.

        The hub enforces these too, and enforces them first — the point of doing it here is
        that the answer is instant and local rather than a round trip, and the error names
        the same bounds the operator wrote.

        A knob whose declaration says ``unset_means: "derive"`` and whose value is absent is
        **left out of the payload**, not defaulted: absent means "work it out from this
        file's calibration", and sending a number instead silently overrides the microscope.

        Every bound the recipe can publish is checked here, including the three that were
        declared by the shipped recipes and enforced by nobody (``pattern``, ``max_len``,
        ``max_items``) and the conditional-knob rule: pinning a real value for a knob whose
        ``applies_when`` is unmet is **refused, not ignored**, because a value that silently
        does nothing looks exactly like one that worked. Use :meth:`resolve_for_send` to
        prepare a payload that clears those automatically.
        """
        requested = dict(knobs or {})
        declared = HubClient.declared_knobs(recipe_meta)
        if not declared:
            return requested                  # nothing to check against; let the hub rule
        unknown = sorted(set(requested) - set(declared))
        if unknown:
            raise LabLinkError(
                f"recipe {recipe_meta.get('name')!r} has no knob named {unknown[0]!r}. "
                f"It offers: {', '.join(sorted(declared)) or '(none)'}")
        out: Dict[str, Any] = {}
        for name, value in requested.items():
            spec = declared[name]
            if value is None:
                out[name] = None              # explicit "put this back to derived"
                continue
            if not knob_applies(spec, requested, declared):
                cond = spec.get("applies_when") or {}
                raise LabLinkError(
                    f"knob {name!r} is only read when {cond.get('knob')!r} is "
                    f"{_condition_text(cond)}, so pinning {value!r} would be refused rather "
                    f"than ignored. Leave it out, or send null to clear it.")
            ktype = str(spec.get("type") or "float")
            enum = spec.get("enum") or ()
            if ktype == "bool":
                if not isinstance(value, bool):
                    raise LabLinkError(
                        f"knob {name!r} must be true or false, not "
                        f"{type(value).__name__} — JSON 0 and 1 are numbers, not booleans")
            elif ktype in ("enum", "string"):
                # One branch for both, because that is one branch on the hub: a recipe may
                # bound an enum by length or shape too, and splitting them here would leave
                # whichever half we did not think about unchecked on this side.
                if not isinstance(value, str):
                    raise LabLinkError(f"knob {name!r} must be a string")
                if enum and value not in enum:
                    raise LabLinkError(
                        f"knob {name!r} = {value!r} is not one of "
                        f"{', '.join(map(repr, enum))}")
                max_len = spec.get("max_len")
                if isinstance(max_len, int) and len(value) > max_len:
                    raise LabLinkError(
                        f"knob {name!r} is {len(value)} characters; the recipe allows "
                        f"at most {max_len}")
                pattern = spec.get("pattern")
                if pattern:
                    # `fullmatch`, NOT `match`, because that is what the hub's validator uses
                    # and the published patterns are unanchored: `cell-segmentation`'s
                    # `stats` declares `[a-z]+(,[a-z]+)*`, which `match` happily satisfies
                    # from the `mean` in `"mean, max"` — so this side would accept a value
                    # the hub then refuses, after the upload.
                    try:
                        hit = re.fullmatch(str(pattern), value)
                    except re.error as exc:
                        raise LabLinkError(
                            f"recipe {recipe_meta.get('name')!r} declares an invalid regex "
                            f"for knob {name!r} ({pattern}): {exc}. That is a recipe bug — "
                            f"tell the operator.") from None
                    if not hit:
                        raise LabLinkError(
                            f"knob {name!r} = {value!r} does not match the shape this recipe "
                            f"accepts ({pattern})")
            elif ktype == "channel_list":
                items = value if isinstance(value, list) else [value]
                if not items:
                    raise LabLinkError(f"knob {name!r} must name at least one channel")
                for item in items:
                    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                        raise LabLinkError(
                            f"knob {name!r} must be a list of channel indices "
                            f"(0, 1, ...); got {item!r}")
                max_items = spec.get("max_items")
                if isinstance(max_items, int) and max_items and len(items) > max_items:
                    raise LabLinkError(
                        f"knob {name!r} names {len(items)} channels; this recipe accepts "
                        f"at most {max_items}")
                # No duplicate check: the hub accepts a repeated index, and a local rule the
                # hub does not have would refuse work that would in fact have run.
                out[name] = list(items)
                continue
            elif ktype in ("float", "int"):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise LabLinkError(
                        f"knob {name!r} must be a number, not {type(value).__name__}")
                if value != value or value in (float("inf"), float("-inf")):
                    raise LabLinkError(f"knob {name!r} must be a finite number")
                if ktype == "int" and float(value) != int(value):
                    raise LabLinkError(f"knob {name!r} must be a whole number; got {value}")
                unit = f" {spec.get('unit')}" if spec.get("unit") else ""
                lo, hi = spec.get("min"), spec.get("max")
                if lo is not None and value < lo:
                    raise LabLinkError(
                        f"knob {name!r} = {value} is below the minimum {lo}{unit}")
                if hi is not None and value > hi:
                    raise LabLinkError(
                        f"knob {name!r} = {value} is above the maximum {hi}{unit}")
            else:
                # A type this build does not know. Previously this fell into the numeric
                # branch and a string-valued knob of a new type was reported as "must be a
                # number" — blaming the value for the client being old.
                raise LabLinkError(
                    f"knob {name!r} has type {ktype!r}, which this build does not know how "
                    f"to check. Update ND2 Studios, or leave this knob at its default.")
            out[name] = value
        return out

    @staticmethod
    def effective_knobs(recipe_meta: Dict[str, Any],
                        sent: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The complete picture of what a command runs with, computed locally.

        Needed because the hub's own ``knobs`` echo — documented as "everything in effect,
        including the recipe's defaults and the ones left derived" — is only populated when
        the request *carried* knobs. A command that sets nothing therefore comes back with an
        empty echo, and a run record built from that echo would say nothing about a run that
        used every one of the recipe's defaults.

        So the echo is used where it exists and this fills the gap: the value sent if one was,
        else the recipe's declared default, else ``None`` for a knob the file decides. Every
        declared knob appears, which is what makes the record replayable.
        """
        declared = HubClient.declared_knobs(recipe_meta)
        given = dict(sent or {})
        out: Dict[str, Any] = {}
        for name, spec in declared.items():
            if name in given:
                out[name] = given[name]
            elif spec.get("unset_means") == "derive":
                out[name] = None
            else:
                out[name] = spec.get("default")
        # A knob the recipe no longer declares but the caller sent anyway is kept rather than
        # dropped: it is evidence about what was asked for, and losing it hides the mismatch.
        for name, value in given.items():
            out.setdefault(name, value)
        return out

    @staticmethod
    def resolve_for_send(recipe_meta: Dict[str, Any],
                         knobs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate ``knobs`` and turn them into a payload safe to send as a whole set.

        The difference from :meth:`check_knobs` is what happens to a knob whose
        ``applies_when`` is unmet: here it becomes an explicit ``null`` rather than an error.
        That is what makes it safe to send *every* knob on *every* command — which the
        protocol asks for, since each command resolves its knobs independently against the
        recipe's defaults, so a knob left out reverts for that command.

        ``null`` rather than omitted, deliberately: a warm worker treats an absent knob as
        unchanged, so omitting an inapplicable knob would leave whatever an earlier command
        set still in force.
        """
        requested = dict(knobs or {})
        declared = HubClient.declared_knobs(recipe_meta)
        if not declared:
            return requested
        cleared = dict(requested)
        for name, spec in declared.items():
            if name in cleared and not knob_applies(spec, requested, declared):
                cleared[name] = None
        return HubClient.check_knobs(recipe_meta, cleared)


class Session:
    """One warm session. A context manager, because closing is not optional."""

    def __init__(self, hub: HubClient, opened: Dict[str, Any], *,
                 recipe_meta: Optional[Dict[str, Any]] = None,
                 knobs: Optional[Dict[str, Any]] = None):
        self.hub = hub
        self.id = str(opened.get("id") or "")
        # From the RESPONSE, never derived from the id — they look mechanical today, and
        # the moment a client hard-codes the pattern a hub layout decision is baked into
        # every machine in the lab and cannot be changed.
        self.in_channel = str(opened.get("in_channel") or "")
        self.out_channel = str(opened.get("out_channel") or "")
        self.workflow = str(opened.get("workflow") or "")
        self.recipe = str(opened.get("recipe") or "")
        self.recipe_meta = dict(recipe_meta or {})
        self.quota_mb = opened.get("quota_mb")
        self.state = str(opened.get("state") or "opening")
        self.detail = str(opened.get("detail") or "")
        self._cursor = int(opened.get("event_seq") or 0)
        self._closed = False
        self.events: List[Dict[str, Any]] = []
        #: The full knob set in force, carried across commands BY THIS CLIENT rather than by
        #: the hub. Each command resolves its knobs on their own against the recipe's
        #: defaults — they do not inherit what was set at ``open`` — so a knob left out of a
        #: command silently reverts for that command. Keeping the whole set here and sending
        #: it every time is what makes "change one knob and run again" mean what it looks
        #: like. It is also what keeps the hub's ``knobs`` echo populated: the hub computes
        #: that echo only when a command carried knobs, so a bare ``run()`` used to come back
        #: with an empty record of what ran.
        self.knobs: Dict[str, Any] = dict(knobs or {})

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _path(self, tail: str = "") -> str:
        return f"/s/{self.id}{tail}"

    def status(self, *, wait: float = 0.0) -> Dict[str, Any]:
        """Poll the session. With ``wait`` the hub blocks server-side until something
        changes, so watching costs one request rather than a sleep loop."""
        body = self.hub._json(
            "GET", self._path(),
            wait=min(float(wait), P.LONGPOLL_MAX_S) if wait else None,
            # The highest seq we have seen. A LOWER one returns instantly with old news
            # and turns the long poll into a hot loop.
            since_event=self._cursor or None)
        self._absorb(body)
        return body

    def _absorb(self, body: Dict[str, Any]) -> None:
        self._cursor = max(self._cursor, int(body.get("event_seq") or 0))
        if body.get("state"):
            self.state = str(body["state"])
        if body.get("detail"):
            self.detail = str(body["detail"])
        for ev in body.get("events") or []:
            self.events.append(ev)

    def wait_ready(self, *, timeout_s: float = 300.0) -> None:
        """Block until the worker is ready, or explain why it never will be.

        LOOPS rather than trusting one call: a 25 s wait can expire while a large program
        is still loading its imports, and ``waited`` tells us the hub really held the
        request rather than answering stale.
        """
        deadline = time.monotonic() + float(timeout_s)
        while self.state == "opening":
            if time.monotonic() > deadline:
                raise SessionLost(
                    f"session {self.id} was still starting its worker after "
                    f"{timeout_s:.0f}s. The hub's own open timeout will end it; check the "
                    f"hub's log for what the program was doing.", code="open_timeout")
            self.status(wait=P.LONGPOLL_MAX_S)
        if self.state not in P.SESSION_LIVE_STATES:
            raise SessionLost(
                f"session {self.id} is {self.state}: {self.detail or 'no detail given'}",
                code="session_gone")

    def close(self, *, discard: bool = False) -> Dict[str, Any]:
        """End the session. Idempotent, and never raises.

        A session holds a running program and, on most hubs, one of very few slots — often
        exactly one per workflow. Leaving it open holds that slot until an idle timer
        expires many minutes later, and to the next machine that asks, a hub with no free
        slots is indistinguishable from a broken one. So this runs on the error path too,
        and swallows everything: a teardown that can throw is how a slot gets leaked.
        """
        if self._closed or not self.id:
            return {}
        self._closed = True
        try:
            _status, body = self.hub.call("DELETE", self._path(),
                                          discard=1 if discard else None)
            return body if isinstance(body, dict) else {}
        except Exception:       # noqa: BLE001 — see the docstring
            return {}

    # ── data in ─────────────────────────────────────────────────────────────────
    def send_data(self, path: str, *, name: str = "") -> FileRef:
        """Upload a file to this session's in channel, streaming from disk.

        The name is repaired for the exchange's rules and the original recorded in
        ``X-Lablink-Meta``. Uploading the same name and content twice is deduplicated by
        the hub, so a retry after a lost response is free rather than a second transfer.
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise LabLinkError(f"no such file to send: {path}")
        original = name or os.path.basename(path)
        safe = repair_name(original)
        digest, size = sha256_of(path)
        headers = {P.H_SHA: digest,
                   P.H_META: json.dumps({"original_name": original})}
        with open(path, "rb") as fh:
            status, body = self.hub.call(
                "PUT", f"/c/{urllib.parse.quote(self.in_channel)}/"
                       f"{urllib.parse.quote(safe)}",
                stream=fh, stream_len=size, headers=headers,
                # A multi-gigabyte upload legitimately exceeds any long-poll-sized timeout.
                timeout=max(self.hub.timeout, 60.0 + size / (1 << 20)))
        if status not in (200, 201):
            if status == P.HTTP_QUOTA:
                raise LabLinkError(
                    f"this session's byte quota ({self.quota_mb} MB) would be exceeded by "
                    f"{os.path.basename(path)}. Pull and delete results, or open a new "
                    f"session — a session quota is per-session and not in /hello, so it "
                    f"cannot be pre-checked.", status=status, code="quota")
            raise self.hub._error_for(status, body, "upload")
        return FileRef(name=safe, sha256=digest, size=size, original_name=original)

    # ── run ─────────────────────────────────────────────────────────────────────
    def run(self, *, inputs: Iterable[FileRef] = (), knobs: Optional[dict] = None,
            cmd_id: str = "", command: str = "run",
            on_progress: Optional[Callable[[Progress], None]] = None,
            check: bool = True, timeout_s: float = 0.0) -> CommandResult:
        """Run the recipe and wait for its terminal state.

        ``check=True`` raises :class:`CommandFailed` on anything but ``done``; pass
        ``check=False`` when a failure is an expected outcome you want to inspect — the
        session stays perfectly usable either way, which is the point.

        Send a ``cmd_id`` for anything expensive. If the response is lost, sending the same
        id again returns *this* command rather than ``409 busy``; without one, a network cut
        on the response leaves no way to tell "my command started" from "it didn't".

        ``knobs`` is a **change set**, not the whole payload: what is passed here is merged
        into :attr:`knobs` and the merged whole is sent. So ``run(knobs={"level": "li"})``
        after opening with ``background_um=6.0`` runs with both, which is what the call reads
        as. Pass ``None`` for a knob to put it back to derived.
        """
        payload: Dict[str, Any] = {"command": command}
        refs = list(inputs)
        if refs:
            # Every input named WITH its sha256: the hub verifies both against the channel
            # before dispatching, so a command that races an upload cannot silently run
            # against a different file of the same name.
            payload["inputs"] = [r.as_input() for r in refs]
        if knobs:
            self.knobs.update(knobs)
        if self.knobs:
            # The FULL set, every time — see `self.knobs`.
            payload["knobs"] = HubClient.resolve_for_send(self.recipe_meta, self.knobs)
        if cmd_id:
            payload["cmd_id"] = cmd_id

        status, body = self.hub.call("POST", self._path("/cmd"), body=payload)
        if status not in (P.HTTP_ACCEPTED, 200):
            raise self.hub._error_for(status, body, "POST /cmd")
        record = body if isinstance(body, dict) else {}
        return self.wait_command(int(record.get("cmd_seq") or 0),
                                 on_progress=on_progress, check=check,
                                 timeout_s=timeout_s, first=record)

    def wait_command(self, cmd_seq: int, *,
                     on_progress: Optional[Callable[[Progress], None]] = None,
                     check: bool = True, timeout_s: float = 0.0,
                     first: Optional[dict] = None) -> CommandResult:
        """Long-poll one command to its terminal state."""
        record = dict(first or {})
        deadline = time.monotonic() + timeout_s if timeout_s else None
        last_text = None
        while True:
            state = str(record.get("state") or "running")
            if state and state not in ("running", "queued", ""):
                break
            if deadline is not None and time.monotonic() > deadline:
                raise LabLinkError(
                    f"command {cmd_seq} on session {self.id} did not finish within "
                    f"{timeout_s:.0f}s. It is still running on the hub — poll it, or "
                    f"cancel it; abandoning the wait does not stop the work.")
            body = self.hub._json("GET", self._path(f"/cmd/{cmd_seq}"),
                                  wait=P.LONGPOLL_MAX_S)
            record = body
            if on_progress is not None:
                snap = Progress.from_dict(record.get("progress"))
                if snap.text != last_text:      # don't spam an unchanged line
                    last_text = snap.text
                    on_progress(snap)
        out = CommandResult(
            cmd_seq=int(record.get("cmd_seq") or cmd_seq),
            cmd_id=str(record.get("cmd_id") or ""),
            state=str(record.get("state") or ""),
            duration_s=record.get("duration_s"),
            knobs=dict(record.get("knobs") or {}),
            progress=Progress.from_dict(record.get("progress")),
            artifacts=list(record.get("artifacts") or []),
            result=dict(record.get("result") or {}),
            error=record.get("error"),
            cached=bool(record.get("cached")))
        session_state = str(record.get("session_state") or "")
        if session_state and session_state not in P.SESSION_LIVE_STATES:
            # The instance itself is gone, so the warm cache went with it: the next command
            # would be silently slower and possibly different. Say so rather than pretend.
            raise SessionLost(
                f"session {self.id} is {session_state} — the warm worker is gone, so this "
                f"session cannot serve another command", code="session_gone")
        if check and not out.ok:
            raise CommandFailed(
                f"command {out.cmd_seq} {out.state}: {out.message or out.code}",
                code=out.code or out.state)
        return out

    def cancel(self) -> Dict[str, Any]:
        """Ask the hub to stop the running command.

        **Cooperative first**, and the reply says so: analysis software rarely has a
        cancellation point in its inner loop, so a cancel lands at the next boundary the
        program checks — which can be a whole step away — and the hub escalates to
        terminating it if it never checks one. With nothing running, this ends the session.
        """
        _status, body = self.hub.call("POST", self._path("/cancel"), body={})
        return body if isinstance(body, dict) else {}

    def reset(self, *, check: bool = True) -> CommandResult:
        """Drop the hub's memo and tile caches, keeping the session and its worker alive.

        This is the "free the RAM but stay warm" rung, **not** a return to the recipe's
        defaults: the knobs in force stay in force, and :attr:`knobs` is deliberately left
        alone so this client and the worker's graph do not start disagreeing about them.
        The next run is correct but pays the decode again, so this is a deliberate act for a
        box that is running out of memory rather than something to do between attempts.
        """
        return self.run(command="reset", check=check)

    # ── data out ────────────────────────────────────────────────────────────────
    def pull(self, *names: str) -> Dict[str, Any]:
        """Publish artifacts the recipe held back, so they can be downloaded.

        With no names, everything held. A 3 GB label volume nobody opens should not cross
        the network by default, which is why the recipe held it in the first place.
        """
        body: Dict[str, Any] = {}
        if names:
            body["names"] = list(names)
        _status, out = self.hub.call("POST", self._path("/pull"), body=body)
        return out if isinstance(out, dict) else {}

    def list_results(self) -> List[Dict[str, Any]]:
        """What is on the out channel now, with each file's session/command/recipe meta."""
        body = self.hub._json("GET", f"/c/{urllib.parse.quote(self.out_channel)}")
        return list(body.get("files") or [])

    def fetch_all(self, dest: str, *, verify: bool = True) -> List[str]:
        """Download every published result into ``dest``. Returns the paths written."""
        os.makedirs(dest, exist_ok=True)
        return [self.fetch(entry, dest, verify=verify)
                for entry in self.list_results()]

    def fetch(self, entry: Dict[str, Any], dest: str, *, verify: bool = True) -> str:
        """Download one result, resuming a partial transfer and verifying the checksum.

        The bytes land in ``<name>.partial`` and are moved into place only once the hash
        matches, so a link that drops mid-download leaves a resumable part rather than a
        truncated file that looks finished. Fetched by ``returned_as`` where the hub had to
        repair a name the exchange would refuse — never by the artifact's original name,
        which in that case is not what is on the channel.
        """
        name = str(entry.get("returned_as") or entry.get("name") or "")
        if not name:
            raise LabLinkError(f"a result listing had no name: {entry!r}")
        want = str(entry.get("sha256") or "")
        final = os.path.join(dest, name)
        part = final + ".partial"
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        size = int(entry.get("size") or 0)

        if have and size and have >= size:
            have = 0                          # a stale part at/over full size: start over
            try:
                os.remove(part)
            except OSError:
                pass

        url = (f"/c/{urllib.parse.quote(self.out_channel)}/"
               f"{urllib.parse.quote(name)}")
        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(self.hub.url + url, method="GET")
        req.add_header(P.H_TOKEN, self.hub.token)
        if self.hub.node_id:
            req.add_header(P.H_NODE, self.hub.node_id)
        for key, value in headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(
                    req, timeout=max(self.hub.timeout, 60.0 + size / (1 << 20))) as resp:
                # 206 honours the range; 200 means the whole file is coming, so anything
                # already on disk is not a prefix of it and must be discarded.
                mode = "ab" if (have and resp.status == 206) else "wb"
                with open(part, mode) as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
        except urllib.error.HTTPError as exc:
            raise self.hub._error_for(exc.code, {}, f"download {name}") from None

        if verify and want:
            got, _n = sha256_of(part)
            if got != want:
                raise LabLinkError(
                    f"{name} arrived corrupt: sha256 {got[:16]}… but the hub listed "
                    f"{want[:16]}…. The partial file is kept at {part} for inspection.")
        os.replace(part, final)
        return final


__all__ = [
    "HubClient", "Session", "FileRef", "Progress", "CommandResult",
    "LabLinkError", "SessionLost", "CommandFailed", "HubBusy",
    "repair_name", "sha256_of", "CHUNK",
]
