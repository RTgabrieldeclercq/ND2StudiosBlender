"""The LabLink wire contract, as ND2Studios sees it — LWP/1 and the session HTTP calls.

This is a **mirror** of ``lablink/protocol.py`` in the LabLink repo, not an import of it,
and that is deliberate. LabLink is standard-library-only by design so it can be dropped
onto an instrument PC whose software stack nobody wants to disturb; ND2Studios is the
opposite kind of program (numpy, a microscopy reader, gigabytes of cache). Making either
one a package dependency of the other would defeat the property that keeps LabLink
installable. The hub already assumes this: ``worker_link.py`` spawns a worker as a
subprocess precisely so "the hub package is stdlib-only, deliberately, and that is what
lets it be dropped onto an instrument PC".

So the two copies drift, and the answer is not to hope they don't:

* :data:`WORKER_PROTOCOL_VERSION` is checked BY THE HUB against our ``hello``. A mismatch
  is a clean ``protocol_error`` naming both numbers — the one drift that cannot pass
  silently.
* Everything else here is a *bound we honour*, not a value we impose: line size, the
  command and event vocabularies, the closed set of error codes. Sending an event name or
  an error code the hub does not know is not rejected — it is simply not understood, which
  is the drift that DOES pass silently. :func:`nodegraph.selftest` therefore pins these
  names, and ``scripts/_lablink_conformance.py`` re-reads them off a real hub when one is
  reachable.

Two limits are worth stating in one place because getting either wrong looks like a
network fault rather than a bug:

* :data:`MAX_WORKER_LINE_BYTES` — over it the hub cannot know where our JSON ended, so it
  kills the worker rather than resyncing. Bulk data therefore NEVER rides the pipe; only
  filesystem paths under the session directory do.
* :data:`LONGPOLL_MAX_S` — the hub's long poll sits under its own 30 s request timeout, so
  a client whose socket timeout is *below* this reads every successful long poll as a dead
  server.

Qt-free; standard library only (imports nothing from ``nodegraph`` either, so a
conformance check can read it without building an engine).
"""
from __future__ import annotations

# ── identity ────────────────────────────────────────────────────────────────────

#: What we answer in ``hello.worker`` — the adapter's name, not the app's.
WORKER_NAME = "nd2studios"

#: The adapter's own version. Bump it when this package's behaviour changes in a way an
#: operator would need to know about; it is reported separately from
#: :data:`SOFTWARE_VERSION` because the hub logs "which worker" and "which science code"
#: for different reasons.
WORKER_VERSION = "1.0.0"

#: Human name of the software being held warm, for the hub's console and banner.
SOFTWARE_NAME = "ND2 Studios V2 (NodeLab)"

#: The engine/app version this adapter fronts.
SOFTWARE_VERSION = "2.21"

#: The ``*.nd2graph.json`` format we can read. Reported in ``hello`` so a hub whose
#: recipes were saved by a newer editor fails at the handshake with something to read,
#: rather than at ``open`` with a JSON error. Kept in step with
#: :data:`nodegraph.serialize.FORMAT_VERSION` by the self-test.
GRAPH_FORMAT = "2.0"


# ── LWP/1: the hub <-> worker protocol ──────────────────────────────────────────

#: Newline-delimited JSON on the worker's stdin/stdout. The hub compares this to its own
#: and refuses the worker on a mismatch, naming both numbers.
WORKER_PROTOCOL_VERSION = 1

#: One line's ceiling. Past it the stream is not trustworthy and the hub kills the worker,
#: so every emitter here stays small and bulk data goes to the filesystem.
MAX_WORKER_LINE_BYTES = 1024 * 1024

#: Commands the hub may send. ``cancel`` and ``ping`` are the two that may arrive WHILE
#: another command is in flight, which is the whole reason stdin is read on its own thread.
WORKER_COMMANDS = (
    "open", "set", "input", "run", "pull", "cancel", "reset", "ping", "close",
)

#: Commands answerable off the reader thread, mid-run. Everything else is queued.
CONCURRENT_COMMANDS = frozenset({"cancel", "ping"})

#: Events a worker may emit. Anything else is noise the hub counts and ignores.
WORKER_EVENTS = (
    "hello", "node", "progress", "beat", "log", "artifact", "result", "error",
)

#: The five ``node`` states. These map 1:1 onto :data:`nodegraph.engine.Observer`'s
#: event names, which is why the adapter is a dict merge and renames nothing.
NODE_STATES = ("start", "cached", "progress", "done", "error")

#: Exactly one of these per command id — never both, never zero, including on the paths
#: that fail. :class:`~nodelab_v2.lablink.worker.Worker` enforces it with a guard rather
#: than by discipline, because "zero" is the failure that presents as a hung session.
TERMINAL_EVENTS = ("result", "error")

#: The closed set of error codes, because operators read these. Emitting a code outside
#: it is not rejected by the hub — it is merely not understood — so the self-test pins it.
#: ``worker_exited`` / the three timeouts are synthesised BY THE HUB, never by us; they are
#: listed so a conformance check can tell "we must never send this" from "we may".
WORKER_ERROR_CODES = frozenset({
    "unsupported", "protocol_error", "bad_request", "bad_recipe", "not_ready",
    "no_such_knob", "knob_out_of_range", "missing_input", "input_error",
    "missing_metadata",
    "missing_dependency", "compute_error", "cancelled", "resource_exhausted",
    "internal", "worker_exited", "silence_timeout", "hard_timeout", "open_timeout",
})

#: Codes only the hub may synthesise. We must never emit one: doing so would report a
#: link failure for a compute problem and send an operator looking at the network.
HUB_ONLY_ERROR_CODES = frozenset({
    "worker_exited", "silence_timeout", "hard_timeout", "open_timeout",
})

#: Codes this worker may emit.
EMITTABLE_ERROR_CODES = WORKER_ERROR_CODES - HUB_ONLY_ERROR_CODES

#: Artifact kinds we can produce, reported in ``hello.features.artifact_kinds`` so a
#: recipe declaring an output we cannot make is a tier-2 ``bad_recipe`` at ``open``
#: rather than a missing file after a forty-minute run.
ARTIFACT_KINDS = ("table", "quicklook", "metrics", "image_stack")

#: Per-recipe return policy for an output. ``auto`` rides back to the node with the
#: result; ``pull`` is held on the hub until asked for (a 3 GB label volume nobody opens
#: should not cross the network by default); ``never`` stays on the hub.
OUTPUT_POLICIES = ("auto", "pull", "never")

#: Params a recipe may never expose as a knob, and why. The hub's tier 1 refuses these, and
#: a recipe *generator* must filter them out of the offer list rather than letting somebody
#: pick one and meet a refusal.
#:
#: The two halves fail differently. A path knob would let a caller choose which file the hub
#: opens, which is the whole security boundary. The editor bookkeeping keys would change the
#: graph's cache key without changing the computation, silently discarding the warm cache
#: that is the entire reason the hub exists.
FORBIDDEN_KNOB_PARAMS = {
    "path": "a filesystem path",
    "model_path": "a filesystem path",
    "store_path": "a filesystem path",
    "urlpath": "a filesystem path",
    "__locked__": "editor bookkeeping, stripped before a run",
    "__title__": "editor bookkeeping, stripped before a run",
    "__channels__": "editor bookkeeping, stripped before a run",
}

#: How often the run loop emits a ``beat`` while a single opaque compute is in flight.
#: The hub's silence timeout is what kills a wedged worker, and it is reset only by a
#: PARSED protocol line — so a long CNN inference with no internal progress hook must
#: still say something, and this is that something. Comfortably under the tightest
#: silence timeout a recipe is likely to set (this lab's stress hub uses 60 s).
BEAT_INTERVAL_S = 5.0

#: Environment variable the hub sets to the session directory. Also passed as
#: ``--session``; the flag wins, and this is the fallback so a command declared without
#: the ``{session}`` placeholder still works.
SESSION_DIR_ENV = "LABLINK_SESSION_DIR"


# ── the image-job sidecar ───────────────────────────────────────────────────────
#
# A TIFF does not merely OMIT the optical metadata an analysis derives from — it invents
# it, reporting the container's 16-bit depth for a 12-bit sensor and ``Ch0`` for a channel
# called ``GFP``. A missing value can be detected and refused; an invented one cannot. So
# an image travels with a sidecar that OVERRIDES what the file claims, and a recipe
# declares which fields its graph cannot run correctly without.
#
# The upload call is content-agnostic and stays that way, which means a successful upload
# never implies a valid sidecar. Writing one is :mod:`nodelab_v2.lablink.sidecar`'s job and
# reading one is the worker's.

#: Suffix pairing a sidecar to its image: ``x.tif`` -> ``x.tif.job.json``. Deliberately not
#: a bare ``.json`` — an image beside an unrelated ``x.json`` is an ordinary thing, and
#: pairing on that would silently adopt it as calibration.
SIDECAR_SUFFIX = ".job.json"

#: A sidecar without this exact string is refused rather than guessed at.
SIDECAR_FORMAT = "lablink.imagejob/1"

#: Sidecar ``image`` scalars -> the engine calibration keys they populate. These names are
#: identical on both sides on purpose: the unit is IN the field name, because a
#: ``pixel_size`` with a separate unit field is the most expensive silent error available
#: here. Every one of these is in :data:`nodegraph.dataset.CALIBRATION_KEYS`.
SIDECAR_IMAGE_FIELDS = (
    "pixel_size_um", "z_step_um", "bit_depth",
    "objective_magnification", "objective_na",
)

#: Sidecar ``image.channels[]`` field -> the engine's flat per-channel-LIST key. The two
#: vocabularies differ in shape because each is the shape its author can get right: nested
#: per-channel for a person writing one, flat parallel lists for the engine reading one.
SIDECAR_CHANNEL_FIELDS = {
    "name": "channel_names",
    "emission_nm": "channel_emission_nm",
    "excitation_nm": "channel_excitation_nm",
}

#: Every metadata name a recipe's ``requires.metadata`` may ask for — the flat engine
#: vocabulary, not the sidecar's nested one. A recipe naming anything else is asking for a
#: field no sidecar can supply, so the worker reports it rather than waiting for a file
#: that will never satisfy it.
REQUIRABLE_METADATA = tuple(SIDECAR_IMAGE_FIELDS) + tuple(SIDECAR_CHANNEL_FIELDS.values())


# ── the session HTTP protocol (the client half) ─────────────────────────────────

#: Header carrying the site or node token on every request.
H_TOKEN = "X-Lablink-Token"

#: Header carrying an upload's sha256, so the hub verifies rather than trusts.
H_SHA = "X-Lablink-Sha256"

#: Header carrying a small ASCII-JSON metadata blob — where a filename repaired for the
#: exchange's name rules records what it really was.
H_META = "X-Lablink-Meta"

#: Header naming the enrolled node whose token is being presented. Absent means the
#: shared site token.
H_NODE = "X-Lablink-Node"

#: Default hub port.
DEFAULT_PORT = 8765

#: The hub's read-only console listener. Its OWN port, loopback by default, never the data
#: port — which is what makes it safe for the editor's panel to read without a token, and
#: also why a panel can only see a hub running on THIS machine.
DEFAULT_CONSOLE_PORT = 8766

#: One poll of everything the console renders: hub health, live sessions, recent history,
#: enrolled nodes, and the workflow/recipe catalogue.
CONSOLE_STATE_PATH = "/console/api/state"

#: Ceiling on ``?wait=`` for a long poll. The hub picked 25 s to sit under its own 30 s
#: request timeout; a client socket timeout at or below this turns every successful long
#: poll into an apparent dead server, so :mod:`nodelab_v2.lablink.client` clamps against it.
LONGPOLL_MAX_S = 25.0

#: Socket timeout the client uses, chosen strictly above :data:`LONGPOLL_MAX_S` for the
#: reason above.
CLIENT_TIMEOUT_S = 40.0

#: Characters the exchange accepts in a channel or file name. Real microscopy filenames are
#: full of parentheses, which are NOT in this set — hence
#: :func:`nodelab_v2.lablink.client.repair_name`, which substitutes and records the original.
SAFE_NAME_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789._ -"
)

#: Session states. Only ``ready``/``running`` mean the warm instance is still there; the
#: rest mean the cache went with it, which is why the hub refuses to pretend otherwise.
SESSION_LIVE_STATES = frozenset({"ready", "running"})

#: Terminal command states. Anything but ``done`` puts the reason in ``error``.
COMMAND_TERMINAL_STATES = frozenset({"done", "failed", "cancelled", "timeout"})

#: HTTP status meanings worth branching on, kept named so the client reads as protocol
#: rather than as magic numbers. ``503`` is retryable and ``409 busy`` is not — the hub
#: chose ``503`` over ``429`` precisely so a naive client backs off instead of giving up.
HTTP_OPENED = 201
HTTP_ACCEPTED = 202
HTTP_BAD_REQUEST = 400
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_CONFLICT = 409
HTTP_GONE = 410
HTTP_AT_CAPACITY = 503
HTTP_QUOTA = 507


__all__ = [
    "WORKER_NAME", "WORKER_VERSION", "SOFTWARE_NAME", "SOFTWARE_VERSION",
    "GRAPH_FORMAT",
    "WORKER_PROTOCOL_VERSION", "MAX_WORKER_LINE_BYTES", "WORKER_COMMANDS",
    "CONCURRENT_COMMANDS", "WORKER_EVENTS", "NODE_STATES", "TERMINAL_EVENTS",
    "WORKER_ERROR_CODES", "HUB_ONLY_ERROR_CODES", "EMITTABLE_ERROR_CODES",
    "ARTIFACT_KINDS", "OUTPUT_POLICIES", "FORBIDDEN_KNOB_PARAMS",
    "BEAT_INTERVAL_S", "SESSION_DIR_ENV",
    "SIDECAR_SUFFIX", "SIDECAR_FORMAT", "SIDECAR_IMAGE_FIELDS",
    "SIDECAR_CHANNEL_FIELDS", "REQUIRABLE_METADATA",
    "H_TOKEN", "H_SHA", "H_META", "H_NODE", "DEFAULT_PORT",
    "DEFAULT_CONSOLE_PORT", "CONSOLE_STATE_PATH",
    "LONGPOLL_MAX_S", "CLIENT_TIMEOUT_S", "SAFE_NAME_CHARS",
    "SESSION_LIVE_STATES", "COMMAND_TERMINAL_STATES",
    "HTTP_OPENED", "HTTP_ACCEPTED", "HTTP_BAD_REQUEST", "HTTP_FORBIDDEN",
    "HTTP_NOT_FOUND", "HTTP_CONFLICT", "HTTP_GONE", "HTTP_AT_CAPACITY", "HTTP_QUOTA",
]
