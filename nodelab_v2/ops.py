"""GUI-facing node ops that must be **Qt-free** so a headless consumer can load and
run a ``*.nd2graph.json`` saved by NodeLab v2 (review 2026-07-22).

Three ops are introduced by the GUI layer rather than the core catalog:

* ``io.load`` — the pipeline source. It has **no compute**: the engine gets its pixels
  from a seed :class:`~nodegraph.dataset.Dataset` (the GUI runner resolves the ``path``
  param to a provider; a headless consumer supplies its own seed for each ``io.load``
  node — see :func:`headless_engine`).
* ``io.dock`` — the checkpoint node (V2.18): pass-through while ``live``, a **source**
  once ``docked``, serving a :mod:`nodegraph.checkpoint` written by a Bake. Docking is
  what makes a long chain on a big file affordable — see :func:`cut_docked_inputs` for
  the one graph rewrite that gives it its whole effect.
* ``view.viewer`` — an inspection tap; a pure pass-through compute registered into
  ``COMPUTES`` **here** (not in the Qt runner) so ``nodegraph.engine.Engine`` can run a
  GUI-authored graph without importing PySide6.

**Per-channel output taps.** A ``io.load`` / ``channel.split`` node exposes one *synthetic*
per-channel output socket ``ch0…chN-1`` in the GUI (each a single channel). The engine is
one-payload-per-node, so these can't be distinct engine outputs; instead
:func:`materialize_channel_taps` rewrites every ``chK`` output edge into a real
``channel.select`` tap (``params={"channels":[K]}``) at graph-build time — reusing the
tested select compute + its lockstep ``channel_select`` meta_transform. This runs for the
run graph, the edit-time envelope pass, and any headless consumer alike.

This module imports only ``nodegraph`` — no PySide6 — so ``import nodelab_v2.ops`` is
safe in a batch/CI context.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from nodegraph.checkpoint import (
    PRECISIONS, checkpoint_envelope, open_checkpoint, read_manifest)
from nodegraph.engine import Engine, EvalContext
from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.memo import digest
from nodegraph.metadata import propagate_meta
from nodegraph.nodes import COMPUTES, register_node
from nodegraph.domains import Domain
from nodegraph.registry import (
    Granularity, InDataset, InString, Mode, NODES, OutDataset, define_node)
from nodegraph.sockets import SocketType

#: a synthetic per-channel output socket name — ``ch0``, ``ch1``, … (GUI-only; the
#: materialization pass turns each wired one into a real ``channel.select`` tap).
CH_SOCKET_RE = re.compile(r"^ch(\d+)$")

#: ``io.dock``'s op_key, and the params key holding its **bake record** — an opaque
#: machine-set dict ``{"id", "sig", "precision", "at", "bytes"}`` written by the Bake
#: action. Dunder-prefixed for the same reason ``__modes__`` is: it is engine/GUI
#: bookkeeping rather than a user control, so the param↔socket contract exempts it
#: (`wire-node-v2` §4b) and no socket may offer it for editing.
DOCK_OP = "io.dock"
BAKE_KEY = "__bake__"

#: ``io.load``'s op_key — the pipeline source. Named because three layers now test for it
#: (the runner's source resolution and per-file ingest lane, the canvas' source-only menu
#: entry, the window's double-click routing) and a bare string in each is how one of them
#: ends up spelled differently.
LOAD_OP = "io.load"

#: the three dock states. ``live`` = an identity pass-through (the chain runs normally);
#: ``held`` = the computed payload is pinned in memory as an engine seed and the in-edge is
#: cut; ``docked`` = the in-edge is cut and the node serves its on-disk checkpoint instead.
#:
#: ``live`` stays FIRST because :func:`dock_state_of` falls back to the spec's first choice
#: for an unset mode, and every saved graph written before ``held`` existed relies on that.
DOCK_LIVE, DOCK_HELD, DOCK_DOCKED = "live", "held", "docked"

#: the states in which the upstream edge is CUT — the predicate every graph rewrite wants.
#: Deliberately not the same question as :func:`is_docked` ("does this serve a disk store"):
#: conflating the two is how a new frozen state silently keeps evaluating the chain it was
#: supposed to freeze.
DOCK_FROZEN = (DOCK_HELD, DOCK_DOCKED)

#: the ``precision`` mode's "not chosen yet" value. There is deliberately no default
#: precision: the right answer depends on what the chain upstream produced (a filter
#: chain's float64, a label raster's integers, a normalized [0,1] image), and silently
#: picking one would either quadruple the store or quantize away real signal. The Bake
#: action refuses while this is selected and says so.
PRECISION_UNSET = "unset"


def ensure_ops() -> None:
    """Idempotently register ``io.load`` (source, no compute) + ``view.viewer``
    (pass-through). Safe to call repeatedly and from any thread (pure registry
    writes)."""
    spec = NODES.get("io.load")
    if spec is None or spec.input("path") is None:
        define_node(
            "io.load", "Load ND2/TIFF file", category="io",
            inputs=[InString("path", "Path", field=False, default="",
                             path_kind="open_file",
                             path_filter="Images (*.nd2 *.tif *.tiff);;ND2 (*.nd2);;"
                                         "TIFF (*.tif *.tiff);;All files (*)",
                             path_hint="empty = synthetic demo · or Browse…",
                             description="The ND2 or TIFF to open. Browse… fills this in; "
                                         "an empty path runs the synthetic demo stack "
                                         "instead, so the graph is testable with no file.")],
            outputs=[OutDataset("image")],
            adds_domains=frozenset({Domain.VOXEL}),   # the source of the image domain
            description="Open an ND2 or TIFF as the pipeline source (the GUI ingests it "
                        "once to a b2nd store next to the file; empty path = synthetic "
                        "demo). Exposes one output per channel + a combined 'All "
                        "channels' output.")
    if NODES.get("view.viewer") is None:
        register_node(
            lambda ctx: ctx.inputs[0],
            op_key="view.viewer", label="Viewer", category="io",
            inputs=[InDataset()], outputs=[OutDataset()],
            description="Inspection tap — the Viewer panel renders whatever Dataset "
                        "flows through it (V2.00 §10).")
    if NODES.get(DOCK_OP) is None:
        register_node(
            _compute_dock,
            op_key=DOCK_OP, label="Dock Data", category="io",
            inputs=[
                InDataset("data"),
                InString("store", "Dock folder", field=False, default="",
                         path_kind="directory",
                         path_hint="set by Bake · or Browse… to reuse a bake",
                         description=
                         "Where the baked checkpoint lives. Bake fills this in for you "
                         "(a folder beside the saved graph), so leave it empty unless "
                         "you are pointing this dock at a bake that already exists — "
                         "for instance to share one bake between two graphs, or to put "
                         "a large dock on a different drive. Changing it does NOT move "
                         "an existing bake; it just looks somewhere else."),
            ],
            outputs=[OutDataset("out")],
            modes=[
                Mode("state", [DOCK_LIVE, DOCK_HELD, DOCK_DOCKED], label="State",
                     description=
                     "Whether the chain upstream is being EVALUATED, or replaced by a frozen "
                     "copy of what it last produced. Flipping it does not bake anything and "
                     "does not delete anything — the bake is a separate action, and this "
                     "switch only chooses which source the graph reads.",
                     choice_docs={
                         DOCK_LIVE:
                             "Pass through: the upstream chain runs normally on every pull, so "
                             "edits above take effect immediately. The state to be in while you "
                             "are still tuning, and the only one in which the nodes behind this "
                             "one are live.",
                         DOCK_HELD:
                             "Freeze what the chain last produced IN MEMORY and cut the edge — "
                             "no disk write, so it is effectively instant and costs no disk. "
                             "The state for quick troubleshooting: freeze a stitch or a merge "
                             "once, then tune everything downstream against it without paying "
                             "for it again. Three honest limits, all of which `docked` fixes: "
                             "it does NOT free memory (the payload is still held, it just stops "
                             "being recomputed), it is NOT counted against any cache budget, "
                             "and it does NOT survive closing the file — reopen and the node "
                             "reports 'released' and asks you to hold it again.",
                         DOCK_DOCKED:
                             "Cut the upstream edge and serve the baked checkpoint on disk "
                             "instead. Slower to enter than `held` because it writes the whole "
                             "series out once, and the only state that actually FREES memory "
                             "(full-size rasters come back memory-mapped) and that survives "
                             "saving and reopening the graph. If something upstream changed "
                             "since the bake, the dock says so instead of quietly serving "
                             "stale data.",
                     }),
                Mode("precision", [PRECISION_UNSET, *PRECISIONS], label="Precision",
                     # Gated to `docked`: a held dock writes nothing, so there is no stored
                     # dtype for this to choose. Leaving it live in `held` would be a control
                     # the kernel ignores — clause 2 of the socket contract, one level up.
                     # Gating is EDIT-TIME only, so the Bake action's own PRECISION_UNSET
                     # refusal still stands and must (the compute still resolves the value).
                     available_in={"state": frozenset({DOCK_DOCKED})},
                     description=
                     "What dtype FLOATING-POINT data is stored at in the bake. Integer rasters, "
                     "label ids and masks always store as themselves, so this only affects "
                     "images that arrived as floats. It is a genuine trade of disk against "
                     "fidelity, and it is deliberately not defaulted — see `unset`.",
                     choice_docs={
                         PRECISION_UNSET:
                             "Nothing chosen yet, and Bake refuses while it is selected. There "
                             "is no safe default because the right answer depends on what the "
                             "chain produced: picking one silently would either quadruple the "
                             "store or quantize real signal away.",
                         "float32":
                             "Single precision — about 7 significant digits, half the size of "
                             "float64. The right choice for essentially all image data, whose "
                             "measurement noise is far above that; the usual pick.",
                         "float64":
                             "Double precision, exactly as computed. Twice the disk for a bake "
                             "that is bit-identical to the live chain — worth it only when the "
                             "values are the product of a long accumulation whose last digits "
                             "matter, e.g. a strain field derived from differences.",
                         "uint16":
                             "16-bit unsigned integers: the smallest of the three, and LOSSY for "
                             "float data — values are ROUNDED and clipped into [0, 65535], never "
                             "rescaled. Appropriate when the chain is still effectively camera "
                             "counts. On [0,1] data (anything after a normalize) it would "
                             "collapse the image to 0/1, and the bake refuses on the first plane "
                             "rather than writing it.",
                     }),
            ],
            granularity=Granularity.TILEABLE,
            description=
            "Bake everything upstream to disk once, then serve it as if it were a "
            "freshly loaded file. The nodes behind it grey out and stop being "
            "evaluated — their results are read back from the checkpoint instead of "
            "recomputed — so a long chain on a big series stops costing memory and "
            "re-runs. Masks, labels, tracks and measurements are baked alongside the "
            "image, and the full-size ones are memory-mapped, so a docked segmentation "
            "costs no RAM. Edit something upstream and the dock says so and keeps "
            "serving the old bake until you re-bake.")


# ── io.dock — the checkpoint node (V2.18) ─────────────────────────────────────
#
# The node is deliberately thin: all it does is choose between "return my input" and
# "return my checkpoint". Everything that makes docking *worth* anything happens in
# `cut_docked_inputs` below — with the in-edge gone the docked node is a graph ROOT, so
# `Engine.pull` never walks the chain behind it, never computes those nodes and never
# memoizes their (full-raster) payloads. The node is the switch; the rewrite is the
# mechanism.


def dock_state_of(rec: Any) -> str:
    """The dock state of a node record — ``"live"``/``"docked"``, or ``""`` when ``rec``
    is not a dock. Duck-typed on ``.op_key``/``.modes`` so it reads a headless
    :class:`~nodegraph.graph.NodeInstance` and a GUI ``NodeRecord`` alike (both carry a
    plain ``modes`` dict); an unset mode falls back to the spec's first choice, which is
    ``live``."""
    if getattr(rec, "op_key", "") != DOCK_OP:
        return ""
    return str((getattr(rec, "modes", None) or {}).get("state") or DOCK_LIVE)


def is_docked(rec: Any) -> bool:
    """True for a dock serving its ON-DISK checkpoint. The disk-specific question: it gates
    the store path, the manifest read, the precision check and the bake record."""
    return dock_state_of(rec) == DOCK_DOCKED


def is_frozen(rec: Any) -> bool:
    """True for a dock whose upstream edge is CUT — ``held`` or ``docked``.

    The graph question, and the one every rewrite wants: :func:`cut_docked_inputs`,
    :func:`dormant_nodes` and :func:`upstream_signature` care only that the chain behind
    this node is out of play, not about where the frozen bytes live. Keeping it separate
    from :func:`is_docked` is what stops a `held` dock from being cut-but-still-evaluated
    (or evaluated-but-not-cut, which is worse: the seed would be ignored and the chain the
    user froze would run anyway)."""
    return dock_state_of(rec) in DOCK_FROZEN


def dock_store_of(rec: Any) -> str:
    """The dock's checkpoint directory (``""`` when unset). Whitespace and the quotes
    Windows' "Copy as path" wraps around a path are stripped, exactly as the ``io.load``
    path is — the same hand-pasted value reaches both."""
    v = (getattr(rec, "params", None) or {}).get("store", "")
    return str(v or "").strip().strip('"').strip("'").strip()


def bake_record(rec: Any) -> Dict[str, Any]:
    """The dock's bake record (``{}`` when it has never been baked)."""
    v = (getattr(rec, "params", None) or {}).get(BAKE_KEY)
    return dict(v) if isinstance(v, dict) else {}


class _Rec:
    """The minimal ``.op_key``/``.params``/``.modes`` shape the dock helpers duck-type on,
    built from an :class:`~nodegraph.engine.EvalContext` — whose mode state arrives folded
    into ``params["__modes__"]`` rather than as a ``modes`` attribute. It exists so the
    compute reads its state through the SAME helpers the GUI and the graph rewrites use,
    instead of re-spelling "which key holds the state" a fourth time."""

    __slots__ = ("op_key", "params", "modes")

    def __init__(self, op_key: str, params: Mapping[str, Any]) -> None:
        self.op_key = op_key
        self.params = params
        self.modes = dict(params.get("__modes__", {}) or {})


def _compute_dock(ctx: EvalContext):
    """Dock Data — identity pass-through while ``live``; the checkpoint's Dataset once
    ``docked``.

    **Resolved spec.** Category ``io``; consumes and produces one Dataset with no axis
    or calibration change of its own (in ``docked`` mode the payload's axes/calibration
    are the *baked* ones, which the edit-time envelope already describes because the
    document seeds this node from the manifest — see
    :func:`nodegraph.checkpoint.checkpoint_envelope`). No 2D/3D lever: it neither reads
    nor writes voxels itself, so it is dimension-agnostic and declares ``TILEABLE`` with
    no kernel axes. Two modes: ``state`` (the switch) and ``precision`` (read by the
    Bake action, not by this compute — the checkpoint is already written by the time
    anything is pulled through here).

    The refusals below are all "you are docked but the bake is not there", and each names
    the fix. They matter because the alternative — quietly falling back to the live input —
    would recompute the entire chain the user docked precisely to avoid, and look like a
    mysterious hang rather than a missing file."""
    state = dock_state_of(_Rec(ctx.op_key, ctx.params))
    if state == DOCK_LIVE:
        return ctx.inputs[0]
    if state == DOCK_HELD:
        # The held payload arrives as this node's SEED (`ctx.seed`) — the engine's compute
        # branch wins over its seed branch, so a node with both has to read it explicitly.
        # Returned as-is, no copy: entering the tier costing nothing is the entire point,
        # and a memoized payload is immutable by the same convention every other one is.
        seed = ctx.seed
        if (getattr(seed, "image", None) is not None
                or getattr(seed, "attributes", None)):
            return seed
        # An empty seed means the hold was RELEASED (a reload, a restart, a crash). Refuse
        # with the fix rather than falling through to `ctx.inputs[0]`: that would put the
        # whole frozen chain back into every pull — a multi-minute stall wearing the costume
        # of a cache hit.
        held_in = ctx.inputs[0] if ctx.inputs else None
        raise ValueError(
            "this Dock node is set to 'held' but nothing is being held any more.\n"
            "A hold lives in memory, so it does not survive reopening the file (or a "
            "restart, or a crash). Press Hold to freeze the chain again — or switch State "
            "to 'docked' and Bake, which writes it to disk and does survive."
            + ("" if held_in is None else
               "\nThe chain above is still wired, so holding it again is one click."))
    store = str(ctx.params.get("store", "") or "").strip().strip('"').strip("'").strip()
    if not store:
        raise ValueError(
            "this Dock node is docked but has no dock folder — press Bake to write one, "
            "or set 'Dock folder' to a bake that already exists.")
    man = read_manifest(store)
    if man is None:
        raise ValueError(
            f"the dock folder holds no finished checkpoint:\n  {store}\n"
            f"It was never baked, was interrupted, or has been deleted. Press Bake to "
            f"write it again (or switch State back to 'live' to run the chain instead).")
    want = str(bake_record(_Rec(ctx.op_key, ctx.params)).get("id", "") or "")
    got = str(man.get("bake_id", "") or "")
    if want and got and want != got:
        raise ValueError(
            f"the checkpoint in\n  {store}\nwas written by a DIFFERENT bake than this "
            f"node expects (found {got[:12]}…, expected {want[:12]}…) — another graph or "
            f"another dock is baking into the same folder. Give this dock its own folder, "
            f"or press Bake to claim it.")
    return open_checkpoint(store)


def docked_nodes(graph: Graph) -> Tuple[str, ...]:
    """Every FROZEN ``io.dock`` node id in ``graph``, sorted — ``held`` and ``docked`` alike.

    Named for the disk state for compatibility (it is the seed/rewrite roster every caller
    already passes around), but the predicate is :func:`is_frozen`: a held dock is a graph
    root exactly as a docked one is, and must appear in the seed map, the cut and the
    dormancy walk for the same reasons."""
    return tuple(sorted(nid for nid, n in graph.nodes.items() if is_frozen(n)))


def held_nodes(graph: Graph) -> Tuple[str, ...]:
    """Every ``held`` dock node id, sorted — the ones whose payload must come from the
    runner's in-memory registry rather than from a store on disk."""
    return tuple(sorted(nid for nid, n in graph.nodes.items()
                        if dock_state_of(n) == DOCK_HELD))


def cut_docked_inputs(graph: Graph) -> Graph:
    """Return a graph in which every **docked** dock has had its in-edges removed, making
    it a root the engine seeds instead of a node it computes through.

    This one rewrite is the entire effect of docking. `Engine._entry` recurses into
    `graph.preds(node_id)` *before* it calls a compute, so as long as the edge exists the
    whole upstream chain is evaluated and memoized no matter what the dock's compute then
    decides to return. Cutting the edge is what stops it — and it must happen on the path
    BOTH the GUI runner and a headless consumer take, which is why it lives here beside
    :func:`materialize_channel_taps` rather than in the document.

    The edge is only cut in the *run* graph; the document keeps it, so the canvas still
    draws the chain feeding the dock (greyed, but attached and editable) and saving the
    file preserves every node that produced the bake."""
    docked = set(docked_nodes(graph))
    if not docked:
        return graph
    edges = [e for e in graph.edges if not (e.dst in docked and e.kind == "forward")]
    if len(edges) == len(graph.edges):
        return graph
    return Graph(nodes=dict(graph.nodes), edges=edges)


def prepare_run_graph(graph: Graph) -> Graph:
    """The graph the engine actually runs: docked chains cut, then per-channel taps
    materialized. Cutting first is deliberate — an edge leaving a ``chK`` socket into a
    docked dock is gone, so no tap node is minted for work nobody will do.

    Note this does NOT unroll ``flow.iterate`` cones: the GUI calls it from the edit-time
    envelope pass too, where unrolling would delete the node ids the inspector looks up.
    :func:`nodelab_v2.document.GraphDocument.to_graph` unrolls first, under its own flag;
    :func:`headless_engine` does the same."""
    return materialize_channel_taps(cut_docked_inputs(graph))


def dock_seeds(graph: Graph, *, held: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A seed :class:`~nodegraph.dataset.Dataset` per docked node, for ``Engine(seeds=…)``.

    Two jobs, both required. It makes the engine treat a docked node as a **source**, so
    `Engine._require_primary_input` does not reject the (deliberately) unwired primary
    input; and the seed's image provider carries a ``version`` — for a disk checkpoint,
    the store path plus its mtime — which the engine folds into the node's ``recipe_hash``
    as ``__seed_version__``. That is what makes a **re-bake** invalidate the memo on its
    own: the rewritten store has a new mtime, so every entry downstream of the dock
    recomputes without anybody having to remember to clear anything.

    A dock whose checkpoint cannot be opened still gets a seed — an empty Dataset — so the
    engine reaches the compute, which raises the specific "your bake is missing" message
    instead of the engine's generic "this input is not wired".

    ``held`` maps node id → an already-computed Dataset the caller is pinning in memory (the
    runner's hold registry). A ``held`` dock is seeded straight from it, with **no copy**:
    the whole point of the tier is that entering it costs nothing, and the payload is
    immutable by the same convention every memoized payload relies on. That is also why a
    seed is the right slot for it rather than the memo — ``Memo._evict_to_budget`` drops
    entries on recency alone, and evicting a payload whose upstream edge has been cut would
    not cost a recompute, it would lose the data."""
    from nodegraph.dataset import Dataset
    held = dict(held or {})
    out: Dict[str, Any] = {}
    for nid in docked_nodes(graph):
        if dock_state_of(graph.nodes[nid]) == DOCK_HELD:
            # A released hold (nothing in the registry) falls through to an empty Dataset,
            # exactly as a missing checkpoint does, so the compute owns the message.
            out[nid] = held.get(nid) if isinstance(held.get(nid), Dataset) else Dataset()
            continue
        store = dock_store_of(graph.nodes[nid])
        try:
            out[nid] = open_checkpoint(store)
        except Exception:  # noqa: BLE001 — the compute owns the user-facing message
            out[nid] = Dataset()
    return out


#: params that must NOT enter a dock's upstream signature: the UI-only annotations the
#: run graph already strips, plus a dock's own bake record (which *contains* a signature —
#: including it would make every signature depend on itself).
_SIG_SKIP = ("__locked__", "__title__", "__channels__")


def upstream_signature(graph: Graph, node_id: str) -> str:
    """A digest of everything upstream of ``node_id`` that a bake depends on — each
    ancestor's op, params and modes, plus the wiring between them.

    Comparing it against the signature stored at bake time is how a dock knows it has gone
    **stale**. It covers the cases that actually change the answer: an edited param, a
    re-wired or deleted node, a muted node, a different source file (an ``io.load``'s path
    is just a param), and a re-bake of an upstream dock (its bake id is in its params).

    The walk **stops at a docked dock** rather than passing through it, mirroring exactly
    what the run graph evaluates. Without that, editing a node upstream of an already-
    docked node would mark this dock stale even though that edit changes nothing anyone
    will read — the docked node in between keeps serving its own frozen bake.
    """
    seen: set = set()
    stack = [e.src for e in graph.preds(node_id)]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in graph.nodes:
            continue
        seen.add(nid)
        if is_frozen(graph.nodes[nid]):
            # Its own frozen payload stands in for its whole chain. FROZEN, not just
            # docked: the walk must stop wherever the run graph stops, and a held dock
            # cuts its edge too — otherwise an edit above a held dock would mark this one
            # stale over a change nobody downstream can see.
            continue
        stack.extend(e.src for e in graph.preds(nid))
    parts = []
    for nid in sorted(seen):
        n = graph.nodes[nid]
        params = {k: v for k, v in (n.params or {}).items() if k not in _SIG_SKIP}
        parts.append((nid, n.op_key, params, dict(n.modes or {})))
    wires = sorted((e.src, e.src_socket, e.dst, e.dst_socket)
                   for e in graph.edges
                   if e.dst in seen or (e.dst == node_id and e.src in seen))
    return digest("dock-sig", parts, wires)


def dock_status(graph: Graph, node_id: str, *,
                store: Optional[str] = None,
                held: Any = ()) -> Tuple[str, str]:
    """``(status, detail)`` for a dock, for the card badge and the inspector.

    ``status`` is one of ``"live"`` (pass-through), ``"held"`` (serving a payload pinned in
    memory), ``"released"`` (set to ``held`` but the pinned payload is gone — the reload
    case), ``"unbaked"`` (docked but there is no checkpoint to serve), ``"stale"`` (serving
    a good bake that no longer matches the graph) or ``"docked"`` (serving, and current).
    ``detail`` is the user-facing reason, empty when there is nothing to say.

    ``store`` overrides the node's own (possibly graph-relative) folder param — the
    document passes the path resolved against the saved file, which only it knows.
    ``held`` is the set of node ids the runner currently has a pinned payload for; a ``held``
    dock outside it reports ``released``.

    **A released hold is reported, never repaired.** Not auto-re-held (that would silently
    re-run the chain the user froze precisely to avoid, on file open, with no progress bar
    they asked for) and never silently downgraded to ``live`` (which would look like it
    worked and quietly put a multi-minute chain back in every pull). Cell-Tracker ships the
    silent version of exactly this — its baseline is never serialized, so after a reload the
    page renders empty with no message while downstream pages still look loaded — and an
    undiagnosable dead panel is worse than a red badge that names the fix."""
    node = graph.nodes.get(node_id)
    if node is None or getattr(node, "op_key", "") != DOCK_OP:
        return ("", "")
    state = dock_state_of(node)
    if state == DOCK_HELD:
        if node_id in set(held or ()):
            return (DOCK_HELD, "")
        return ("released", "this node was holding its result in memory, and memory does "
                            "not survive reopening the file — press Hold to freeze it "
                            "again, or Bake to write it to disk so it survives next time")
    if state != DOCK_DOCKED:
        return (DOCK_LIVE, "")
    store = dock_store_of(node) if store is None else store
    try:
        man = read_manifest(store)
    except ValueError as exc:                       # a newer checkpoint format
        return ("unbaked", str(exc))
    if man is None:
        return ("unbaked", "no finished bake in the dock folder — press Bake")
    rec = bake_record(node)
    if str(rec.get("id", "")) and str(man.get("bake_id", "")) \
            and str(rec["id"]) != str(man["bake_id"]):
        return ("stale", "the dock folder was re-baked by something else — press Bake "
                         "to claim it")
    want = str((node.modes or {}).get("precision") or PRECISION_UNSET)
    if want != PRECISION_UNSET and want != str(man.get("precision", "")):
        return ("stale", f"baked at {man.get('precision')}, set to {want} — "
                         f"re-bake to apply")
    if str(rec.get("sig", "")) and rec["sig"] != upstream_signature(graph, node_id):
        return ("stale", "something upstream changed since this was baked — re-bake to "
                         "apply it")
    return (DOCK_DOCKED, "")


def dormant_nodes(graph: Graph) -> frozenset:
    """The nodes a docked graph no longer evaluates — what the canvas greys out.

    A node is dormant when **every** forward route out of it is dead: each one either
    feeds a docked dock (that edge is cut, so nothing flows along it) or feeds another
    dormant node. Resolved as a backward fixpoint, because dormancy is contagious — cut
    the dock's input and the filter behind it goes dark, which darkens the filter behind
    *that*, and so on back to the source.

    Two deliberate exclusions, both cases where greying out would say the opposite of
    what runs:

    * **a node with no out-edges is never dormant.** It is a terminal the user can select
      and view, so it still computes on demand.
    * **a node feeding a live branch as well as a docked one is never dormant.** It is
      still evaluated for the live branch, and only one of its two consumers stopped
      caring.

    Computed on the ORIGINAL graph (before :func:`cut_docked_inputs`) — it describes
    precisely the nodes that cut removes from play."""
    docked = set(docked_nodes(graph))
    if not docked:
        return frozenset()
    outs: Dict[str, List[str]] = {}
    for e in graph.edges:
        if e.kind == "forward":
            outs.setdefault(e.src, []).append(e.dst)
    dormant: set = set()
    changed = True
    while changed:
        changed = False
        for nid in graph.nodes:
            if nid in dormant or nid in docked:
                continue                  # a docked dock RUNS (as a source), never dark
            dsts = outs.get(nid)
            if not dsts:
                continue                  # a terminal — still viewable, so still live
            if all(d in docked or d in dormant for d in dsts):
                dormant.add(nid)
                changed = True
    return frozenset(dormant)


def _real_dataset_out(op_key: str) -> str:
    """The name of a node type's first real Dataset output socket (``image`` for
    ``io.load``, ``out`` for ``channel.split`` / most nodes) — the socket a channel tap
    feeds from."""
    spec = NODES.get(op_key)
    if spec is not None:
        for s in spec.outputs:
            if s.type is SocketType.DATASET:
                return s.name
    return "out"


def materialize_channel_taps(graph: Graph) -> Graph:
    """Return a runnable graph in which every GUI-synthetic per-channel output edge
    (``src_socket`` matching ``chK``) is rewired through a real ``channel.select`` tap.

    One tap node is inserted per ``(source_node, channel_index)`` and shared by every
    edge leaving that ``chK`` socket. Full-bundle edges (``image`` / ``out`` / value
    sockets) pass through unchanged. The input graph is not mutated; if there are no
    channel taps the same graph object is returned.
    """
    taps: dict = {}                              # (src, k) -> tap node id
    extra_nodes: dict = {}
    new_edges = []
    for e in graph.edges:
        m = CH_SOCKET_RE.match(e.src_socket) if e.kind == "forward" else None
        if m is None:
            new_edges.append(e)
            continue
        k = int(m.group(1))
        key = (e.src, k)
        tap_id = taps.get(key)
        if tap_id is None:
            tap_id = f"__tap__{e.src}__c{k}"
            taps[key] = tap_id
            extra_nodes[tap_id] = NodeInstance(
                tap_id, "channel.select", params={"channels": [k]})
            real_out = _real_dataset_out(graph.nodes[e.src].op_key)
            new_edges.append(Edge(e.src, tap_id, real_out, "data", "forward"))
        new_edges.append(Edge(tap_id, e.dst, "out", e.dst_socket, e.kind))
    if not extra_nodes:
        return graph
    nodes = dict(graph.nodes)
    nodes.update(extra_nodes)
    return Graph(nodes=nodes, edges=new_edges)


def headless_engine(graph: Graph, *, seeds: Mapping[str, Any],
                    meta_seeds: Optional[Mapping[str, Any]] = None,
                    sweep_all: Any = (),
                    **engine_kwargs: Any) -> Engine:
    """Build a runnable :class:`~nodegraph.engine.Engine` for a GUI-authored graph
    **without** any Qt import. The caller supplies a seed Dataset per ``io.load`` node
    (headless has no file dialog / ingest runner); every other op resolves through the
    shared ``COMPUTES`` — including ``view.viewer``, registered by :func:`ensure_ops`.

    Docked chains are cut and per-channel output edges materialized into
    ``channel.select`` taps first (:func:`prepare_run_graph`), and every docked node is
    seeded from its checkpoint — so a batch run of a saved graph gets the *same* skipped
    upstream and the same on-disk data the GUI does, rather than silently recomputing
    hours of work the user already baked. A caller-supplied seed always wins, so a dock
    can still be overridden explicitly.

    Every ``flow.iterate`` cone is unrolled first (:func:`nodegraph.iterate.unroll`), so a
    saved sweep run in batch produces the same iterations the app does. The ``around``
    value source needs each driven socket's live derive to centre on, which means an
    envelope pass over the PRE-unroll graph — cheap (it touches no pixels) and the same
    thing the GUI hands in from its own propagated map."""
    from nodegraph.iterate import iterate_nodes, unroll as unroll_iterate
    ensure_ops()
    if iterate_nodes(graph):
        pre = materialize_channel_taps(graph)
        try:
            envs = propagate_meta(pre, dict(meta_seeds or {}))
        except ValueError:                    # a malformed graph: the unroll reports it
            envs = None
        graph = unroll_iterate(graph, envs=envs, sweep_all=sweep_all)
    graph = prepare_run_graph(graph)
    all_seeds = dock_seeds(graph)
    all_seeds.update(seeds)
    meta = {nid: env for nid, env in
            ((nid, checkpoint_envelope(dock_store_of(graph.nodes[nid])))
             for nid in docked_nodes(graph)) if env is not None}
    meta.update(meta_seeds or {})
    return Engine(graph, computes=COMPUTES, seeds=all_seeds,
                  meta_seeds=meta, **engine_kwargs)


__all__ = ["ensure_ops", "headless_engine", "materialize_channel_taps",
           "prepare_run_graph", "cut_docked_inputs", "dock_seeds", "dock_status",
           "dormant_nodes", "docked_nodes", "upstream_signature", "dock_state_of",
           "dock_store_of", "bake_record", "is_docked", "is_frozen", "held_nodes",
           "CH_SOCKET_RE", "DOCK_OP", "BAKE_KEY", "DOCK_LIVE", "DOCK_HELD",
           "DOCK_DOCKED", "DOCK_FROZEN", "PRECISION_UNSET"]
