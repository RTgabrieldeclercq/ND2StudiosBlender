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
from nodegraph.metadata import (CONDITION_KEY, CONDITION_SET_KEY, propagate_meta,
                                value_rescaled)
from nodegraph.nodes import COMPUTES, register_node
from nodegraph.domains import Domain
from nodegraph.registry import (
    Granularity, InBool, InDataset, InFloat, InString, Mode, NODES, OutDataset,
    define_node)
from nodegraph.sockets import SocketType

#: a synthetic per-channel output socket name — ``ch0``, ``ch1``, … (GUI-only; the
#: materialization pass turns each wired one into a real ``channel.select`` tap).
CH_SOCKET_RE = re.compile(r"^ch(\d+)$")

#: An ``io.load`` card's resolved **position groups**: ``[{"key", "size", "shape"}, …]``,
#: one per specimen the acquisition holds (:func:`nodegraph.placement.position_groups`).
#: Drives the card's synthetic ``grpK`` outputs, and is read back here to put each group's
#: KEY on the tap that materializes it.
#:
#: Lives in this module rather than beside the document's other params keys only because of
#: the import direction — ``document`` imports ``ops``, never the reverse — and it is
#: re-exported there so the GUI layer reads it from the place it belongs to.
#:
#: Deliberately NOT one of ``document._UI_PARAM_KEYS``, for the reason ``BUNDLE_PATHS_KEY``
#: is not either: this is not an annotation, it decides WHICH PIXELS the ``grpK`` outputs
#: produce. ``grp2`` means "the third group in this list", so a list that changed — a
#: sidecar written, a threshold moved — has to re-key the memo, or a cached result computed
#: for one specimen would be served for another.
GROUPS_KEY = "__groups__"

#: A synthetic per-GROUP output socket on a source card: ``grp0``, ``grp1``, … The index is
#: a position in the card's :data:`~nodelab_v2.document.GROUPS_KEY` list, not a group key —
#: socket names have to be plain identifiers, and a group renamed to "treated (2 mM)" in a
#: sidecar could not be one. The KEY is resolved from that list at materialization and put
#: on the tap, which is what the engine and the memo actually see.
GRP_SOCKET_RE = re.compile(r"^grp(\d+)$")

#: A synthetic per-MEMBER output socket on a ``util.unbatch`` card: ``bat0``, ``bat1``, …
#: The index is a position in the batch's wiring order, not a file name — socket names have
#: to be plain identifiers and a file called ``WellA3 (2 mM).nd2`` could not be one. The
#: NAME is resolved at materialization and put on the ``util.select_batch`` tap, which is
#: what the engine and the memo see, for the same reason :data:`GRP_SOCKET_RE` resolves a
#: group KEY: an index would quietly point at a different file after a rewire with every
#: hash still agreeing, while a name either still resolves or refuses.
BAT_SOCKET_RE = re.compile(r"^bat(\d+)$")

#: the synthetic per-POSITION output sockets a ``util.split_positions`` card grows
#: (``pos0…``), materialized into ``util.select_position`` taps (2026-10-02)
POS_SOCKET_RE = re.compile(r"^pos(\d+)$")

#: the synthetic per-PLANE output sockets a ``util.split_z`` card grows (``z0…``),
#: materialized into ``util.select_plane`` taps (2026-10-07) — the Z-axis member of the
#: family. The tap carries the plane's INDEX, like a channel's and a position's: a plane
#: has no name, and ``zK`` means "the (K+1)-th plane of whatever is wired".
Z_SOCKET_RE = re.compile(r"^z(\d+)$")

#: the synthetic per-FRAME output sockets a ``util.split_t`` card grows (``t0…``) while the
#: series is short enough to fan out, materialized into ``util.select_frame`` taps
#: (2026-10-07) — the T-axis member of the family. The tap carries the frame's INDEX.
T_SOCKET_RE = re.compile(r"^t(\d+)$")

#: the synthetic per-GROUP output sockets a split card grows when its `groups` text is set
#: (2026-10-07): ``chg0…`` on Split Channels, ``posg0…`` on Split Positions, ``zg0…`` on
#: Split Z — one per RANGE the user typed, replacing the per-index sockets. A channel group
#: materializes into ``channel.select`` with that list; a position or plane group into
#: ``util.crop`` in frames mode (``m0-2`` / ``z4-7``), the node that already carries every
#: per-axis list, spacing and origin rule for a frame subset. The group INDEX is what the
#: socket carries; what it means is resolved from the card's own `groups` text at
#: materialization (:func:`nodegraph.metadata.parse_groups`), so the run graph reads the
#: way the card does and a retyped group re-keys its tap.
CHG_SOCKET_RE = re.compile(r"^chg(\d+)$")
POSG_SOCKET_RE = re.compile(r"^posg(\d+)$")
ZG_SOCKET_RE = re.compile(r"^zg(\d+)$")
TG_SOCKET_RE = re.compile(r"^tg(\d+)$")


def split_group(node: Any, k: int):
    """Group ``k`` of a split card's `groups` text: ``(name, indices)``, or ``None`` when the
    text has no such group (then the edge is left alone, like any unresolvable tap)."""
    from nodegraph.metadata import every_n, parse_groups
    raw = node.params.get("groups") if node is not None else None
    step = every_n(raw)
    if step is not None:
        # `every N`: group k is k·N … k·N+N-1, unclamped — the axis length is not known
        # here, and Crop's frames mode drops what lies past the end, so the last chunk
        # clips itself to the axis
        return ("", tuple(range(k * step, (k + 1) * step))) if k >= 0 else None
    groups = parse_groups(raw)
    if not groups or not (0 <= k < len(groups)):
        return None
    return groups[k]


def batch_member_identity(node: Any, node_id: str) -> str:
    """A batch member's identity: the base name of the file its source node carries.

    **The single definition, used by both sides on purpose.** The document resolves it to
    label and key the synthetic ``batK`` sockets; :func:`materialize_batch_taps` resolves it
    again from the RUN graph to put on the tap. If those two disagreed the card would look
    correct and the pull would refuse, so they call the same function over the same
    param — ``path``, which is a real compute input and survives the strip that removes UI
    annotations like ``__title__``.
    """
    import os
    if node is None:
        return str(node_id)
    path = str((getattr(node, "params", None) or {}).get("path") or "").strip()
    return os.path.basename(path) if path else str(node_id)


def batch_member_names_of(graph: Graph, unbatch_id: str) -> List[str]:
    """Member identities of the batch feeding ``unbatch_id``, in wiring order — the run
    graph's copy of :meth:`nodelab_v2.document.GraphDocument.batch_member_names`.

    Walks back to the nearest ``util.batch``, stopping at a ``util.select_batch`` because
    past that tap the stream is one member and is not in a batch any more.
    """
    seen: set = set()
    stack = [unbatch_id]
    bid = None
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        op = getattr(node, "op_key", "")
        if op == "util.batch":
            bid = nid
            break
        if op == "util.select_batch" and nid != unbatch_id:
            continue
        stack.extend(e.src for e in graph.edges
                     if e.dst == nid and e.kind == "forward")
    if bid is None:
        return []
    out: List[str] = []
    seen_names: Dict[str, int] = {}
    for e in graph.edges:
        if e.dst != bid or e.kind != "forward":
            continue
        name = batch_member_identity(graph.nodes.get(e.src), e.src)
        n = seen_names.get(name, 0)
        seen_names[name] = n + 1
        out.append(name if n == 0 else f"{name} ({n + 1})")
    return out

#: ``io.dock``'s op_key, and the params key holding its **bake record** — an opaque
#: machine-set dict ``{"id", "sig", "precision", "at", "bytes"}`` written by the Bake
#: action. Dunder-prefixed for the same reason ``__modes__`` is: it is engine/GUI
#: bookkeeping rather than a user control, so the param↔socket contract exempts it
#: (`wire-node-v2` §4b) and no socket may offer it for editing.
DOCK_OP = "io.dock"

#: ``io.write_movie``. Named here beside the other op keys the GUI special-cases so
#: the inspector and the window agree on one spelling, rather than each carrying a
#: literal that can drift.
MOVIE_OP = "io.write_movie"
BAKE_KEY = "__bake__"

#: ``io.load``'s op_key — the pipeline source. Named because three layers now test for it
#: (the runner's source resolution and per-file ingest lane, the canvas' source-only menu
#: entry, the window's double-click routing) and a bare string in each is how one of them
#: ends up spelled differently.
LOAD_OP = "io.load"

#: The page-boundary ops of a V4 Workspace (2026-10-05) — GUI-layer like ``io.load`` and
#: ``view.viewer``: registered by :func:`ensure_ops`, never part of the Qt-free catalog, and
#: deliberately NOT in :data:`HIDDEN_OP_PREFIXES` (they are placed by hand). ``page.output``
#: names the Dataset wired into it as a VARIABLE of its page; ``page.input`` on a page of a
#: later kind reads one by ``"<page_id>:<name>"``. At run time the Workspace splices the
#: upstream page in and the Input node disappears
#: (:meth:`nodelab_v2.workspace.Workspace.compose`); its compute exists only to say so when a
#: graph is run without a Workspace.
PAGE_INPUT_OP = "page.input"
PAGE_OUTPUT_OP = "page.output"
PAGE_OPS = (PAGE_INPUT_OP, PAGE_OUTPUT_OP)
#: ``page.input``'s source reference, ``"<page_id>:<variable name>"``.
PAGE_SOURCE_KEY = "source"
#: ``page.output``'s variable name — a PRESENTATION socket: renaming a variable re-keys no
#: memo entry, because the Dataset it names is the same Dataset.
PAGE_NAME_KEY = "name"
#: ``page.output``'s condition label, stamped into the Dataset's metadata so a downstream
#: table can say which experimental condition a row came from. Blank = the page's own name,
#: filled in at compose time (the node cannot know what its page is called).
PAGE_CONDITION_KEY = CONDITION_KEY
#: Set by ``Workspace.compose`` on an Output whose condition it FILLED from the page's name
#: (the run graph only): such a stamp gives way to a condition TYPED further upstream.
PAGE_CONDITION_AUTO_KEY = "__condition_auto__"
#: The name a hand-placed Page Output starts with (``out``, ``out2``, …).
DEFAULT_OUTPUT_BASE = "out"
#: Characters a Page Output name may not carry: ``:`` splits a Source value
#: (:meth:`Workspace.parse_source`), ``/`` is the run-id separator.
_OUTPUT_NAME_BAD = re.compile(r"[:/\\]+")
_WHITESPACE = re.compile(r"\s+")


def sanitize_output_name(name: Any, limit: int = 48) -> str:
    """A Page Output name a Source value can carry: ``:`` ``/`` ``\\`` → ``_``, whitespace
    collapsed, at most ``limit`` characters; ``""`` when nothing is left."""
    s = _OUTPUT_NAME_BAD.sub("_", str(name or ""))
    s = _WHITESPACE.sub(" ", s).strip()
    return s[:limit].strip()


def next_free_name(base: str, taken) -> str:
    """``base``, else ``base2``, ``base3``, … — the first not in ``taken`` (lower-cased
    names: Page Output names are unique case-insensitively, like page names)."""
    if base.lower() not in taken:
        return base
    n = 2
    while f"{base}{n}".lower() in taken:
        n += 1
    return f"{base}{n}"



# ── one PART of a node's output on its own wire (V4.00 step 11f) ─────────────
#: A node whose output carries more than one kind of data — an image and a mask, a label
#: raster and its table, a point set — offers each PART on a synthetic output socket of its
#: own (``part:image``, ``part:<layer name>``) beside ``out``, which still carries them all.
#: A part edge is materialized into a hidden ``data.part`` tap (:func:`materialize_part_taps`)
#: that keeps that piece alone: the image with no layers, or one named layer — a structure
#: table and the raster of the same name together — whose raster becomes the wire's image
#: (so a Viewer shows a mask, a filter can work on it); a table-only part carries no image.
PART_OP = "data.part"
PART_KEY = "part"
PART_IMAGE = "image"
PART_SOCKET_RE = re.compile(r"^part:(.+)$")

# ── the INVERSE of a region crop on its own socket (V4.00 step 12) ───────────
#: ``util.crop_region``'s card carries a second Dataset output, ``outside``: everything the
#: region did NOT keep, at the frame's extent. The engine is one-payload-per-node, so a wire
#: leaving it is materialized into a SIBLING of the crop node — the same op, params and
#: modes with ``keep`` flipped to ``outside``, fed by the same wires — one per crop node,
#: shared by every wire leaving the socket (:func:`materialize_outside_taps`). Import-free
#: constants (the node module imports the engine; this module must stay cheap to import).
CROP_REGION_OP = "util.crop_region"
OUTSIDE_SOCKET = "outside"
CROP_KEEP_MODE = "keep"
CROP_KEEP_OUTSIDE = "outside"


def part_socket(name: str) -> str:
    """The synthetic output socket that carries part ``name`` (``part:mask``)."""
    return f"part:{name}"


def part_of(attr: Any) -> str:
    """The PART a stored attribute belongs to: a structure table's columns go by their
    source LAYER (a label table is one part, however many columns it has), a lattice layer
    by its own name (the ``mask`` raster)."""
    from nodegraph.domains import is_structure
    return str(attr.layer or attr.name) if is_structure(attr.domain) else str(attr.name)


# ── a Page Output collecting several items (V4.00 step 11f) ──────────────────
#: A Page Output takes up to eight wires — ``data`` and ``data_2`` … ``data_8``, one empty
#: slot shown after the last wired one — each an ITEM of the variable, named by the
#: comma-separated ``items`` param (blank entries take a name from the wire). A Page Input
#: reading it offers each item on a synthetic ``item:<name>`` socket beside ``out`` (the
#: first item). Both become real run-graph nodes: :func:`materialize_output_items` gives each
#: extra item an Output node of its own (so pulling the Output previews its first item and
#: computes nothing else), :func:`materialize_input_items` a Page Input tap per item read.
PAGE_ITEMS_KEY = "items"
PAGE_ITEM_SOCKETS = ("data",) + tuple(f"data_{i}" for i in range(2, 9))
ITEM_SOCKET_RE = re.compile(r"^item:(.+)$")
#: a materialized Page Input tap's item name (run graph only)
PAGE_ITEM_KEY = "__item__"


def item_socket(name: str) -> str:
    """The synthetic Page Input socket that carries item ``name`` (``item:mask``)."""
    return f"item:{name}"


def output_item_node_id(output_id: str, socket: str) -> str:
    """The run-graph node of a Page Output's item on ``socket``: the Output itself for the
    first, ``__item__<output>__data_2`` … for the others."""
    return output_id if socket == PAGE_ITEM_SOCKETS[0] else f"__item__{output_id}__{socket}"


def input_item_tap_id(input_id: str, name: str) -> str:
    """The run-graph tap that reads item ``name`` for Page Input ``input_id``."""
    return f"__tap__{input_id}__item_{name}"

#: What an unresolved ``page.input`` says when it is pulled.
PAGE_UNBOUND_MESSAGE = "Page Input is not bound to an upstream Output"

#: op prefixes hidden from the palette / link search / readiness suggestions (boundary +
#: fixture ops, plus the source loader ``io.load`` — it is created from File → Load
#: ND2/TIFF file…, never dragged). Lives here, Qt-free, so :mod:`nodelab_v2.readiness` can
#: rank producers without importing the scene (moved from scene.py 2026-10-02).
HIDDEN_OP_PREFIXES = ("zone.", "group.", "test.", "io.seed", "io.stream_seed",
                      "rr.", "eng.", "io.nd2", "io.load",
                      # a part tap is minted by the graph build for a card's part socket
                      PART_OP,
                      # minted by nodegraph.iterate's feedback rewrite between two clones,
                      # never placed by hand — it exists only inside an unrolled graph
                      "flow.advance")

#: op prefixes whose result is something to LOOK AT — a Viewer node, a plot (V4.00 step 4).
#: Clicking such a card shows it in the active viewer even with click-to-preview off: the
#: card exists to be seen, so selecting it is asking to see it.
VISUAL_OUTPUT_PREFIXES = ("view.", "plot.")


def is_visual_output(op_key: str) -> bool:
    """Whether ``op_key``'s result is a picture to look at (:data:`VISUAL_OUTPUT_PREFIXES`)."""
    return str(op_key or "").startswith(VISUAL_OUTPUT_PREFIXES)


#: ``io.load``'s access mode and its three choices — how a source card reaches its pixels.
#:
#: ``direct`` reads an uncompressed ND2's memory-mapped frames in place — no copy, no
#: store, usable the instant the file is picked (:mod:`nodelab_v2.nd2_direct`). ``ingest``
#: copies the file into a ``.b2nd`` store once and reads from there — compressed, with a
#: display pyramid, at the cost of a one-time copy that can be the size of the file itself.
#: ``auto`` decides between the two at resolve time instead of committing either way:
#: reuse an existing store if one already covers the file
#: (:meth:`~nodelab_v2.runner.EngineRunner._effective_access`), else ingest it if a fresh
#: copy would fit the destination drive with room to spare, else read it in place
#: (:func:`~nodelab_v2.nd2_direct.decide_access`).
#:
#: **``direct`` is the default** (:data:`ACCESS_DEFAULT`) — both for a freshly placed card,
#: via :meth:`~nodegraph.registry.NodeSpec.default_state`, and for the unset mode on every
#: graph saved before this mode existed, via :func:`source_access_of`'s fallback. Every
#: file opens immediately with no copy unless something is explicitly asked to build one:
#: ``auto`` and ``ingest`` are both there, but neither is what a card gets by just sitting
#: on the canvas. The one case this costs something is a series with very large planes,
#: which scrubs better off a stored pyramid than decimated live — set ``auto`` or
#: ``ingest`` by hand for those (the crossover is the size of a PLANE, not of the file; see
#: the choice docs and MANUAL.md §4).
ACCESS_MODE = "access"

#: ``io.load``'s grouping lever and its two choices — whether the card offers one output per
#: POSITION GROUP (see :data:`GROUPS_KEY`).
#:
#: **Off is the default** (changed 2026-09-28). Detection is reliable on a clean multi-mosaic
#: acquisition, but a card that silently grows six extra outputs is a card whose shape depends
#: on a threshold, and the sockets are only useful to somebody who already wants to work one
#: specimen at a time. Opting in is one dropdown; opting out of something the app decided for
#: you means first working out why the card looks like that.
GROUPING_MODE = "grouping"
GROUPING_AUTO, GROUPING_OFF = "auto", "off"
#: What an unset ``grouping`` mode means. Named rather than spelled at each site: the document
#: reads it to decide whether to offer sockets, and a saved graph from before this mode existed
#: carries no mode at all — both must land on the same answer as a freshly dropped card.
GROUPING_DEFAULT = GROUPING_OFF
ACCESS_AUTO, ACCESS_INGEST, ACCESS_DIRECT = "auto", "ingest", "direct"
#: What an unset ``access`` mode means — named rather than spelled at each site, the same
#: reason :data:`GROUPING_DEFAULT` is, and for the same requirement: a freshly placed card
#: and a saved graph from before this mode existed must land on the same answer.
ACCESS_DEFAULT = ACCESS_DIRECT

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


#: The calibration a SOURCE card may state for itself, overriding (or supplying) what the
#: file carries. Named once here so the socket list, the edit-time envelope
#: (:meth:`nodelab_v2.document.GraphDocument.propagate`) and the pulled payload
#: (:meth:`nodelab_v2.runner.EngineRunner._resolve_source`) cannot drift about what is
#: overridable — the same single-builder rule the placement entry follows.
#:
#: **Why the source and not a mid-graph node** (asked 2026-09-15): a plain TIFF records no
#: Z spacing at all — every file in FranckLab's SerialTrack3D set is one — and `z_step_um`
#: is read *before* anything downstream could restate it. `detect.particles`/`detect.spots`
#: derive their AXIAL sigma from it, so a correction applied after detection would leave the
#: detection itself scaled against the wrong spacing, and `track.objects` (SerialTrack 3D)
#: refuses outright without it. Fixing it at the card that opens the file is the only place
#: that is true for every reader.
CALIB_OVERRIDE_KEYS: Tuple[str, ...] = ("pixel_size_um", "z_step_um")


def calib_overrides(params: Mapping[str, Any]) -> Dict[str, float]:
    """The calibration keys this source node states for itself — ``{}`` when it states none.

    **Zero or blank means "whatever the file itself carries"**, which is what makes this
    additive: a graph saved before these fields existed has neither param, resolves to ``{}``
    and reads exactly as it always did. Only a POSITIVE value overrides, because zero is not
    a physically meaningful pixel size or Z step — it is the empty box.
    """
    out: Dict[str, float] = {}
    for key in CALIB_OVERRIDE_KEYS:
        try:
            value = float(params.get(key) or 0.0)
        except (TypeError, ValueError):
            continue                          # a half-typed box is not an override
        if value > 0.0:
            out[key] = value
    return out


def with_calib_override(metadata: Mapping[str, Any],
                        params: Mapping[str, Any]) -> Dict[str, Any]:
    """``metadata`` with this source node's calibration overrides applied (copy-on-write)."""
    over = calib_overrides(params)
    md = dict(metadata)
    md.update(over)
    return md


def _compute_viewer(ctx: EvalContext):
    """``view.viewer`` — the display sink. Returns the PRIMARY stream untouched: that is the
    payload the Viewer panel shows and describes. The extra ``source_N`` streams are
    ``view_source`` sockets, composited for display by the runner
    (:meth:`nodelab_v2.runner.EngineRunner.overlay_chain` collects every wired one) and
    never read here, so the memo key of the viewed result is the primary's alone. The
    scale-bar sockets are ``presentation`` and are deliberately NOT read either — the window
    reads them live from the document (:meth:`nodelab_v2.window.MainWindow._viewer_scalebar`),
    so toggling the bar repaints instead of re-pulling. ``layout`` is a Mode, which folds
    into the recipe hash; a change re-pulls, and the pull is a memo hit."""
    return ctx.inputs[0]


def _compute_page_output(ctx: EvalContext):
    """``page.output`` — a pass-through that NAMES its input as one of its page's variables.
    The name is presentation (the window and the Workspace read it off the document); the
    only thing the compute adds is the ``condition`` label, stamped into the metadata so a
    table built downstream of several pages can say which condition each row came from."""
    ds = ctx.inputs[0]
    cond = str(ctx.params.get(PAGE_CONDITION_KEY, "") or "").strip()
    if not cond:
        return ds
    if ctx.params.get(PAGE_CONDITION_AUTO_KEY):
        # blank here (compose filled in the page's name): a condition TYPED upstream wins
        md = ds.metadata or {}
        if md.get(CONDITION_SET_KEY) and str(md.get(CONDITION_KEY) or "").strip():
            return ds
        return ds.with_metadata(**{CONDITION_KEY: cond, CONDITION_SET_KEY: None})
    return ds.with_metadata(**{CONDITION_KEY: cond, CONDITION_SET_KEY: True})


def _compute_part(ctx: EvalContext):
    """``data.part`` — keep ONE part of the Dataset (V4.00 step 11f): ``image`` keeps the
    image and drops every layer and table; any other name keeps that layer alone (its
    structure table and its raster together), the raster becoming the image — a mask is a
    0/1 image, a label raster an integer one — and a table-only part (points, tracks)
    carrying no image. Lazy: arrays are shared, never copied. ``bit_depth`` is dropped for a
    layer part: its values are not the camera's counts."""
    from dataclasses import replace as _dc_replace
    import numpy as _np
    from nodegraph.provider import ArrayProvider
    ds = ctx.inputs[0]
    part = str(ctx.params.get(PART_KEY) or PART_IMAGE)
    if part == PART_IMAGE:
        if ds.image is None:
            raise ValueError("this wire carries no image — take one of its layers instead")
        return _dc_replace(ds, attributes={})
    kept = {k: a for k, a in ds.attributes.items() if part_of(a) == part}
    if not kept:
        have = sorted({part_of(a) for a in ds.attributes.values()})
        raise ValueError(f"there is no “{part}” on this wire any more — it carries "
                         + (", ".join(["image"] + have) if have else "only its image")
                         + "; wire the part you want")
    raster = next((a for a in kept.values() if a.domain is Domain.VOXEL), None)
    image = None
    if raster is not None:
        vals = raster.values if isinstance(raster.values, _np.memmap) else \
            _np.asarray(raster.values)
        if vals.dtype == _np.bool_:
            vals = vals.view(_np.uint8)              # same bytes, a type a LUT can draw
        image = ArrayProvider(vals)
    md = dict(ds.metadata or {})
    md.pop("bit_depth", None)
    return _dc_replace(ds, image=image, attributes=kept, metadata=md)


def _meta_part(env, params, modes):
    """Edit-time twin of :func:`_compute_part`: a layer part drops ``bit_depth``; which
    layers pass is ``keep_layers`` (:func:`_keep_part`)."""
    if str(params.get(PART_KEY) or PART_IMAGE) == PART_IMAGE:
        return env
    return value_rescaled(env, params, modes)


def _keep_part(params, modes):
    part = str((params or {}).get(PART_KEY) or PART_IMAGE)
    return frozenset() if part == PART_IMAGE else frozenset({part})


def _meta_page_output(env, params, modes):
    """Edit-time twin of :func:`_compute_page_output` — the same stamp, in lockstep."""
    cond = str(params.get(PAGE_CONDITION_KEY, "") or "").strip()
    if not cond:
        return env               # the page's name is filled in at compose time, not here
    return env.with_metadata(**{CONDITION_KEY: cond, CONDITION_SET_KEY: True})


def _compute_page_input(ctx: EvalContext):
    """``page.input`` — never evaluated inside a Workspace (compose replaces it by a wire
    from the upstream page's Output). Reached only when the node is unbound or the graph is
    run on its own, so the refusal says exactly that."""
    src = str(ctx.params.get(PAGE_SOURCE_KEY, "") or "").strip()
    hint = (f": {src!r} names no output an earlier page offers" if src
            else ": pick a named output of an earlier page in its Source menu")
    raise ValueError(PAGE_UNBOUND_MESSAGE + hint)


def ensure_ops() -> None:
    """Idempotently register ``io.load`` (source, no compute), ``view.viewer``
    (pass-through) and the V4 page boundaries ``page.output`` / ``page.input``. Safe to
    call repeatedly and from any thread (pure registry writes)."""
    spec = NODES.get("io.load")
    _access_mode = next((m for m in spec.modes if m.name == ACCESS_MODE), None) \
        if spec is not None else None
    _group_mode = next((m for m in spec.modes if m.name == GROUPING_MODE), None) \
        if spec is not None else None
    if (spec is None or spec.input("path") is None
            or spec.input("z_step_um") is None
            or _access_mode is None or ACCESS_AUTO not in _access_mode.choices
            or _access_mode.resolved_default() != ACCESS_DEFAULT
            or _group_mode is None):
        define_node(
            "io.load", "Load ND2/TIFF file", category="io",
            inputs=[InString("path", "Path", field=False, default="",
                             path_kind="open_file",
                             path_filter="Images (*.nd2 *.tif *.tiff);;ND2 (*.nd2);;"
                                         "TIFF (*.tif *.tiff);;All files (*)",
                             path_hint="empty = synthetic demo · or Browse…",
                             description="The ND2 or TIFF to open. Browse… fills this in; "
                                         "an empty path runs the synthetic demo stack "
                                         "instead, so the graph is testable with no file."),
                    InFloat("pixel_size_um", "Pixel size", unit="um", field=False,
                            default=0.0,
                            description=
                            "The lateral sampling this file was acquired at, in microns per "
                            "pixel. Loading a file FILLS THIS IN from its own header, so "
                            "what you see is what the pipeline is using — and you can type "
                            "over it when the file is wrong or silent. **0 means \"whatever "
                            "the file says\"**, which is what an older graph (and any file "
                            "whose header is trustworthy) resolves to.\n\n"
                            "It moves MEASUREMENTS, not just the ruler: every µm-denominated "
                            "size in the graph — a spot radius, a minimum area, a search "
                            "distance — converts through this number, and so does every area "
                            "and length a table reports. An ND2 carries it; a plain TIFF "
                            "often does not."),
                    InFloat("z_step_um", "Z step", unit="um", field=False, default=0.0,
                            description=
                            "The spacing between Z planes, in microns. Filled in from the "
                            "file when it records one (an ND2, or an ImageJ TIFF's "
                            "`spacing`) and typed in by hand when it does not — **a plain "
                            "OME-TIFF usually carries no Z spacing at all**, which is why "
                            "this box exists. 0 means \"whatever the file says\".\n\n"
                            "Nothing downstream can invent it: a 3D detection derives its "
                            "AXIAL radius from this (a wrong value stretches or flattens "
                            "every bead it finds), and SerialTrack 3D tracking REFUSES "
                            "without it, because it rescales z by z_step ÷ pixel size before "
                            "building its topology descriptor and anisotropic voxels would "
                            "distort every neighbour distance. For a volume already expressed "
                            "in isotropic voxel units, set it equal to the pixel size."),
            ],
            outputs=[OutDataset("image")],
            modes=[
                Mode(GROUPING_MODE, [GROUPING_OFF, GROUPING_AUTO],
                     default=GROUPING_DEFAULT, label="Grouping",
                     description=
                     "Whether this card offers one output per SPECIMEN. A multipoint file "
                     "is often several samples rather than one flat list of fields \u2014 six "
                     "3x3 mosaics a millimetre apart, one plate well per site \u2014 and the "
                     "stage coordinates say which. No effect on a file with one group.",
                     choice_docs={
                         GROUPING_AUTO:
                             "Work out the groups when the file is opened (from a "
                             "`.groups.json` sidecar if you wrote one, otherwise by "
                             "clustering the stage positions) and grow one extra output "
                             "per group, beside the full-file output. Wiring from one of "
                             "them is the same as inserting a Select Group node \u2014 it IS "
                             "one, added for you at run time \u2014 so a six-mosaic file gives "
                             "six pipelines off one card with nothing to configure. Inert "
                             "on a file that turns out to hold a single group, so the only "
                             "cost of turning it on is the sockets you asked for.",
                         GROUPING_OFF:
                             "One output, every position, no detection \u2014 the DEFAULT, so a "
                             "card looks the same whatever the stage log happens to contain. "
                             "Right whenever the positions are one experiment (a plate "
                             "scanned as a single 7x7 mosaic), whenever you would rather "
                             "place Select Group by hand, and whenever you simply have not "
                             "thought about it yet. Switching to Auto later costs nothing "
                             "and needs no reload. Turning it back off with group outputs "
                             "already WIRED breaks those wires, so the card asks first.",
                     }),
                Mode("access", [ACCESS_DIRECT, ACCESS_AUTO, ACCESS_INGEST], label="Access",
                     description=
                     "How the pixels are reached. Direct is the default — every file "
                     "opens immediately with no copy. Switch to Auto to let the app build "
                     "a compressed store when that clearly pays off, or to Ingest to "
                     "always build one.",
                     choice_docs={
                         ACCESS_DIRECT:
                             "Read planes straight out of the .nd2 — no copy, no store, "
                             "usable immediately. This is the DEFAULT. For an UNCOMPRESSED "
                             "ND2 the frames are memory-mapped, so a random plane costs "
                             "about a page fault (~6 ms for 1024² here) and a 453 GB "
                             "series is interactive with no ingest at all. Refused, with "
                             "the reason, for a compressed or legacy ND2 and for TIFFs — "
                             "switch to Auto or Ingest for those. There is no stored "
                             "pyramid, so a series with very large planes scrubs better "
                             "off Auto or Ingest instead.",
                         ACCESS_AUTO:
                             "Decide automatically, the moment this file is actually "
                             "opened: keep using an existing store if this file already "
                             "has one; otherwise ingest it if a fresh copy would fit the "
                             "destination drive with room to spare, or read it in place "
                             "(Direct) if it would not. Picks up an already-ingested "
                             "file's store for free — Direct on its own never looks for "
                             "one — while still refusing to run the drive to zero bytes "
                             "free on a series too big to copy.",
                         ACCESS_INGEST:
                             "Always copy the file once into a compressed .b2nd store "
                             "beside it, then read every plane from there — even though "
                             "Direct is the default. Costs a one-time ingest (roughly the "
                             "size of the file, and minutes to hours) and buys compression "
                             "plus a display pyramid, which is what keeps a large-plane "
                             "series smooth to scrub. Force this when the planes are large "
                             "and you know the copy fits.",
                     }),
            ],
            adds_domains=frozenset({Domain.VOXEL}),   # the source of the image domain
            description="Open an ND2 or TIFF as the pipeline source. Access defaults to "
                        "Direct — read the ND2 in place, no copy; switch to Auto to build "
                        "a compressed b2nd store when it clearly pays off, or to Ingest to "
                        "always build one. Empty path = synthetic demo. "
                        "Exposes one output per channel + a combined 'All "
                        "channels' output.")
    _vspec = NODES.get("view.viewer")
    if _vspec is None or _vspec.input("source_2") is None or _vspec.outputs:
        _src_doc = (
            "Another image stream to show alongside the primary. It is composited for "
            "DISPLAY only — placed field-for-field onto the primary's frame as extra "
            "channels named after this socket — and never reaches a node downstream, "
            "because this node has no output. The next empty slot appears once this one "
            "is wired, so there is always exactly one free input. For two files that must "
            "line up by stage position, pixel size or focus, use an Overlay node upstream "
            "instead; this slot pairs frame m with frame m at scale 1.")
        register_node(
            _compute_viewer,
            op_key="view.viewer", label="Viewer", category="io",
            inputs=[
                InDataset("data", label="Image",
                          description=
                          "The image to display. This is the PAYLOAD of the node — the "
                          "stream the channel strip, the hover readout and every overlay "
                          "tab describe. Extra streams wired below are drawn over or "
                          "beside it."),
                *[InDataset(f"source_{i}", label=f"Source {i}", view_source=True,
                            passes_domains=False, grow_group="sources",
                            description=_src_doc) for i in range(2, 7)],
                InBool("show_scalebar", "Scale bar", field=False, default=False,
                       presentation=True,
                       description=
                       "Draw a scale bar over the image in the Viewer. Its length snaps to "
                       "a round 1-2-5 value sized to the frame unless `Bar length` sets "
                       "one, and it is drawn from the payload's `pixel_size_um` — with no "
                       "calibration on the wire no bar is drawn, because a bar without one "
                       "would be a fabrication. Display only: it is not part of any result "
                       "and not saved into exports (Export Movie has its own)."),
                InFloat("scalebar_um", "Bar length", unit="um", field=False, default=0.0,
                        presentation=True,
                        available_in=None,
                        description=
                        "The bar's length in MICRONS. 0 (the default) is auto: the largest "
                        "1-2-5 value near a fifth of the frame width. Set it to pin the bar "
                        "to a figure's convention (10, 20, 50 µm); a length wider than the "
                        "visible image is shortened to fit rather than drawn off-screen. "
                        "Only drawn while `Scale bar` is on."),
                InString("scalebar_corner", "Bar corner", field=False,
                         default="bottom_right", presentation=True,
                         choices=["bottom_right", "bottom_left", "top_right", "top_left"],
                         choice_docs={
                             "bottom_right": "The figure convention and the default; the "
                                             "label sits above the bar.",
                             "bottom_left": "Bottom-left, label above — when the "
                                            "bottom-right corner holds the structure "
                                            "you are showing.",
                             "top_right": "Top-right with the label below the bar — clear "
                                          "of a status readout along the bottom edge.",
                             "top_left": "Top-left, label below — away from a bottom "
                                         "status readout or a frame counter.",
                         },
                         description=
                         "Which corner of the VISIBLE image the bar sits in. It follows the "
                         "corner as you zoom and pan, so it stays readable rather than "
                         "scrolling off with the frame. Only drawn while `Scale bar` is on."),
                InString("scalebar_color", "Bar colour", field=False, default="white",
                         presentation=True,
                         choices=["white", "black", "yellow", "cyan"],
                         choice_docs={
                             "white": "White with a dark shadow — reads on a dark "
                                      "fluorescence field, the default.",
                             "black": "Black — for a light-background (brightfield, phase) "
                                      "image where white would vanish.",
                             "yellow": "Yellow — high contrast on both a dark field and a "
                                       "green or red channel.",
                             "cyan": "Cyan — high contrast over a red or magenta channel, "
                                     "where white and yellow both blend into the signal.",
                         },
                         description=
                         "The bar and label colour. Pick the one that contrasts with the "
                         "corner it sits in; the bar also carries a translucent dark "
                         "shadow so white survives a bright field. Only drawn while `Scale "
                         "bar` is on."),
                # ── the timestamp (2026-10-02): presentation, like the bar ──
                InBool("show_timestamp", "Timestamp", field=False, default=False,
                       presentation=True,
                       description=
                       "Draw the viewed frame's time over the image. What it says is "
                       "`Timestamp shows`: elapsed time since the first frame by default, "
                       "from the file's own per-frame acquisition clock (frame_time_jd) — "
                       "so a series built by Timeseries Builder shows real gaps, not a "
                       "nominal interval — falling back to `dt_s × frame` and then to the "
                       "frame number when no clock rides the payload. Display only: not "
                       "part of any result and not saved into exports (Export Movie has "
                       "its own counter)."),
                InString("timestamp_mode", "Timestamp shows", field=False, default="elapsed",
                         presentation=True,
                         choices=["elapsed", "clock", "frame", "elapsed_clock"],
                         choice_docs={
                             "elapsed": "Time since the first frame of the series — "
                                        "`12.5 s`, `03:20`, `01:15:00` or `2d 04:00:00`, "
                                        "at a unit chosen once from the whole span so the "
                                        "readout never changes shape while you scrub. The "
                                        "default.",
                             "clock": "The wall-clock time the frame was acquired, "
                                      "`YYYY-MM-DD HH:MM:SS.mmm`, as the microscope wrote "
                                      "it (frame_datetime). Blank when the payload carries "
                                      "no absolute clock — a synthetic or TIFF source.",
                             "frame": "The frame number and the series length, `t 7/120` "
                                      "(1-based for reading; the status bar keeps the "
                                      "0-based index). Always available.",
                             "elapsed_clock": "Both: the elapsed time, then the wall-clock "
                                              "time in brackets — for a figure that needs "
                                              "the experiment time and the real date.",
                         },
                         description=
                         "What the timestamp overlay displays for the viewed frame. All "
                         "four read the payload's own metadata; nothing is re-run. Only "
                         "drawn while `Timestamp` is on."),
                InString("timestamp_corner", "Timestamp corner", field=False,
                         default="top_left", presentation=True,
                         choices=["top_left", "top_right", "bottom_left", "bottom_right"],
                         choice_docs={
                             "top_left": "Top-left, where video players put a clock — the "
                                         "default; clear of the bottom-right scale bar.",
                             "top_right": "Top-right — when the top-left corner holds the "
                                          "troubleshooting locator map or a structure you "
                                          "are showing.",
                             "bottom_left": "Bottom-left — beside, not over, a scale bar "
                                            "in the bottom-right corner, along the same "
                                            "bottom edge.",
                             "bottom_right": "Bottom-right — shares the corner with a "
                                             "scale bar there, so the text moves up a line "
                                             "to keep clear of the bar.",
                         },
                         description=
                         "Which corner of the VISIBLE image the timestamp sits in; it "
                         "follows the corner under zoom and pan. Only drawn while "
                         "`Timestamp` is on."),
                InString("timestamp_color", "Timestamp colour", field=False, default="white",
                         presentation=True,
                         choices=["white", "black", "yellow", "cyan"],
                         choice_docs={
                             "white": "White with a dark shadow — reads on a dark "
                                      "fluorescence field, the default.",
                             "black": "Black — for a light-background (brightfield, phase) "
                                      "image where white would vanish.",
                             "yellow": "Yellow — high contrast on both a dark field and a "
                                       "green or red channel.",
                             "cyan": "Cyan — high contrast over a red or magenta channel, "
                                     "where white and yellow both blend into the signal.",
                         },
                         description=
                         "The timestamp's text colour; it also carries a translucent dark "
                         "shadow so white survives a bright field. Only drawn while "
                         "`Timestamp` is on."),
            ],
            outputs=[],
            modes=[
                Mode("layout", ["merged", "tiles", "both"], default="merged", label="Layout",
                     description=
                     "How several image streams are laid out in the Viewer. With one "
                     "stream wired the three are identical. A display choice: changing it "
                     "re-lays out the picture from the planes already decoded.",
                     choice_docs={
                         "merged": "One picture: every stream's channels composited "
                                   "together over the primary's frame, each stream named "
                                   "on the channel strip so it can be toggled on its own.",
                         "tiles": "One pane per stream, side by side — the primary's "
                                  "channels in the first pane, then each extra source in "
                                  "its own — with a shared cursor and contrast. The way to "
                                  "compare a raw channel with a processed one, or two "
                                  "files, without colours mixing.",
                         "both": "The merged composite first, then the per-stream panes: "
                                 "see the overlap and the parts at once, at the cost of "
                                 "smaller panes.",
                     }),
            ],
            description="Display sink — shows the primary image stream, and any extra "
                        "streams wired into the slots that appear as you fill them, merged "
                        "or tiled, with an optional scale bar. No output: what it shows "
                        "goes nowhere downstream (V2.00 §10; sources, layout and scale bar "
                        "2026-10-02).")
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
    _item_doc = ("Another ITEM of this variable: any Dataset — another branch, one part "
                 "of a node, a position. A later page's Page Input reading this Output "
                 "offers each item on its own socket, named in `Items`. The next empty "
                 "slot appears once this one is wired.")
    if NODES.get(PAGE_OUTPUT_OP) is None or NODES.get(PAGE_OUTPUT_OP).input("data_2") is None:
        register_node(
            _compute_page_output,
            op_key=PAGE_OUTPUT_OP, label="Page Output", category="page",
            inputs=[
                InDataset("data", label="Data",
                          description=
                          "The Dataset this page hands on under the name below — an image, "
                          "a mask, a labelled set, a table-carrying Dataset, whatever the "
                          "chain in front of it produced. Pulling this node previews exactly "
                          "what a later page's Page Input will receive. More wires below "
                          "make it a variable of several ITEMS (V4.00 step 11f); this is "
                          "the first, which a Page Input's `out` carries."),
                *[InDataset(s, label=f"Item {i}", grow_group="items", passes_domains=False,
                            description=_item_doc)
                  for i, s in enumerate(PAGE_ITEM_SOCKETS[1:], start=2)],
                InString(PAGE_NAME_KEY, "Name", field=False, default="", presentation=True,
                         description=
                         "The variable name a later page picks this output by (its Page "
                         "Input's Source menu lists `<this page> · <name>`). Give every "
                         "output on one page a different name, or a later page cannot tell "
                         "them apart. Renaming re-keys nothing: the Dataset is the same, so "
                         "every memoized result downstream is kept."),
                InString(PAGE_ITEMS_KEY, "Items", field=False, default="", presentation=True,
                         description=
                         "The names of this variable's items, in wiring order, separated by "
                         "commas (`smooth, mask, cells`). A blank entry takes a name from "
                         "its wire — the part it carries or the node it comes from. A later "
                         "page's Page Input offers each item on a socket of that name. "
                         "Renaming re-runs nothing."),
                InString(PAGE_CONDITION_KEY, "Condition", field=False, default="",
                         description=
                         "The experimental-condition label stamped into the Dataset's "
                         "metadata (`condition`), which Table Concat writes as a column. "
                         "Leave blank to use this page's own name — a linked page then "
                         "labels its rows with ITS name, the point of linking one workflow "
                         "per dish — unless a condition was TYPED on an Output further "
                         "upstream, which a blank one keeps. Changing it re-runs only the "
                         "stamp (this node), not the chain in front of it."),
            ],
            outputs=[OutDataset("out")],
            granularity=Granularity.TILEABLE,
            kernel_axes=frozenset(),
            meta_transform=_meta_page_output,
            description="Name the Dataset wired in as a VARIABLE of this page, for a Page "
                        "Input on a later page to read. A pass-through: nothing is copied or "
                        "changed except an optional `condition` label in the metadata.")
    if NODES.get(PART_OP) is None or NODES.get(PART_OP).keep_layers is None:
        register_node(
            _compute_part,
            op_key=PART_OP, label="Part", category="page",
            inputs=[
                InDataset("data", label="Data",
                          description="The Dataset one part is taken from — what the "
                                      "card's `out` carries."),
                InString(PART_KEY, "Part", field=False, default=PART_IMAGE,
                         description=
                         "Which piece to keep: `image` (the image, no layers or tables), "
                         "or the name of one layer — a mask, a label set with its table, "
                         "a point table. A raster layer becomes the image of the wire; a "
                         "table-only one carries no image. Placed for you by wiring a "
                         "part socket of a card; never placed by hand."),
            ],
            outputs=[OutDataset("out")],
            granularity=Granularity.TILEABLE,
            kernel_axes=frozenset(),
            meta_transform=_meta_part,
            keep_layers=_keep_part,
            description="One part of a node's output on its own wire (V4.00 step 11f): the "
                        "image alone, or one named layer with its table. Minted by the "
                        "graph build for a wire leaving a card's part socket.")
    if NODES.get(PAGE_INPUT_OP) is None:
        register_node(
            _compute_page_input,
            op_key=PAGE_INPUT_OP, label="Page Input", category="page",
            inputs=[
                InString(PAGE_SOURCE_KEY, "Source", field=False, default="",
                         description=
                         "Which earlier page's named output this page starts from — pick "
                         "one from the menu (`<page> · <name>`; only pages of an earlier "
                         "kind, or Free pages, are offered, so the page graph can never "
                         "loop). Stored as `<page id>:<name>`, so renaming a page keeps the "
                         "link and renaming the OUTPUT breaks it — the panel then shows this "
                         "input as unbound until you re-pick."),
            ],
            outputs=[OutDataset("out")],
            granularity=Granularity.TILEABLE,
            kernel_axes=frozenset(),
            description="Start this page from a named output of an earlier page. At run "
                        "time the upstream page is spliced in and this node disappears, so "
                        "a result the upstream page already computed is reused from the "
                        "memo rather than recomputed; at edit time the node carries the "
                        "upstream output's envelope (axes, calibration, layers), so every "
                        "derived default on this page is right before anything is pulled.")


def dock_state_of(rec: Any) -> str:
    """The dock state of a node record — ``"live"``/``"docked"``, or ``""`` when ``rec``
    is not a dock. Duck-typed on ``.op_key``/``.modes`` so it reads a headless
    :class:`~nodegraph.graph.NodeInstance` and a GUI ``NodeRecord`` alike (both carry a
    plain ``modes`` dict); an unset mode falls back to the spec's first choice, which is
    ``live``."""
    if getattr(rec, "op_key", "") != DOCK_OP:
        return ""
    return str((getattr(rec, "modes", None) or {}).get("state") or DOCK_LIVE)


def source_access_of(rec: Any) -> str:
    """The access mode of an ``io.load`` record — ``"direct"``/``"auto"``/``"ingest"``, or
    ``""`` when ``rec`` is not a source card.

    Duck-typed on ``.op_key``/``.modes`` for the same reason :func:`dock_state_of` is: it
    has to read a GUI ``NodeRecord`` and a headless
    :class:`~nodegraph.graph.NodeInstance` alike. An unset mode falls back to
    :data:`ACCESS_DEFAULT` (``"direct"``) — which is also the FIRST declared choice, so
    this matches what a freshly placed card actually carries via
    ``NodeSpec.default_state()`` (:meth:`nodelab_v2.document.NodeRecord.state`).

    This is a deliberate behaviour choice, not a backward-compatibility default: every
    graph saved before this mode existed reads ``"direct"`` too, not a fixed "always
    ingest". Nothing gets copied unless a card is explicitly told to (``ingest``) or told
    to decide for itself (``auto``) — including graphs saved back when ingesting was the
    only thing that happened, and including a card whose file already has a perfectly
    good store sitting next to it (set Access to Auto, or Ingest, to use that store)."""
    if getattr(rec, "op_key", "") != LOAD_OP:
        return ""
    return str((getattr(rec, "modes", None) or {}).get(ACCESS_MODE) or ACCESS_DEFAULT)


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
    # Batch FIRST, then groups, then positions, then planes, then channels: each narrows a
    # different axis (B, then M twice — a group is a set of positions, a position one of
    # them — then T, then Z, then C) so they commute on the data, and running them
    # outermost-axis-first keeps each tap closest to the node that asked for it — the
    # order the card reads in. A split card's RANGE groups (2026-10-07) sit beside their
    # per-index siblings on the same axis.
    # V4.00 step 11f: a Page Output's extra items become Outputs of their own FIRST (a dock
    # cut then sees the real wires); a Page Input's item reads and a card's part sockets
    # become taps LAST, outermost, like the channel taps they sit beside
    # V4.00 step 12: a Crop by Region's `outside` wire becomes a sibling node first, so the
    # wires copied onto it are then tapped like the original's.
    return materialize_part_taps(
        materialize_input_items(
            materialize_channel_groups(materialize_channel_taps(
                materialize_plane_groups(materialize_plane_taps(
                    materialize_frame_groups(materialize_frame_taps(
                        materialize_position_groups(materialize_position_taps(
                            materialize_group_taps(
                                materialize_batch_taps(cut_docked_inputs(
                                    materialize_outside_taps(
                                        materialize_output_items(graph)))))))))))))))


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
                signature: Optional[str] = None,
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
    now = upstream_signature(graph, node_id) if signature is None else signature
    if str(rec.get("sig", "")) and rec["sig"] != now:
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


def _materialize_taps(graph: Graph, pattern, op_key: str, tag: str, params_for,
                      modes: Optional[Mapping[str, str]] = None) -> Graph:
    """Rewire every GUI-synthetic output edge matching ``pattern`` through a real tap node.

    The shared engine behind :func:`materialize_channel_taps` and
    :func:`materialize_group_taps`. One tap is inserted per ``(source_node, index)`` and
    SHARED by every edge leaving that socket, so wiring one group into four downstream
    branches costs one ``util.select_group``, not four — and, because the tap has one
    identity, those branches hit one memo entry instead of computing the same subset
    repeatedly.

    ``params_for(src_node, index)`` returns the tap's params, or ``None`` to leave the edge
    alone. Returning ``None`` is how a socket whose meaning can no longer be resolved — a
    ``grp4`` edge on a card whose group list has shrunk to three — fails: the edge stays
    pointed at a socket that is not there, which the document's own validation reports,
    rather than being silently rewritten to some other specimen's positions.

    Full-bundle edges (``image`` / ``out`` / value sockets) pass through unchanged; the
    input graph is never mutated, and with no taps the same object is returned.
    """
    taps: dict = {}                              # (src, k) -> tap node id
    extra_nodes: dict = {}
    new_edges = []
    for e in graph.edges:
        m = pattern.match(e.src_socket) if e.kind == "forward" else None
        if m is None:
            new_edges.append(e)
            continue
        k = int(m.group(1))
        params = params_for(graph.nodes.get(e.src), k)
        if params is None:
            new_edges.append(e)
            continue
        key = (e.src, k)
        tap_id = taps.get(key)
        if tap_id is None:
            tap_id = f"__tap__{e.src}__{tag}{k}"
            taps[key] = tap_id
            extra_nodes[tap_id] = NodeInstance(tap_id, op_key, params=params,
                                               modes=dict(modes or {}))
            real_out = _real_dataset_out(graph.nodes[e.src].op_key)
            new_edges.append(Edge(e.src, tap_id, real_out, "data", "forward"))
        new_edges.append(Edge(tap_id, e.dst, "out", e.dst_socket, e.kind))
    if not extra_nodes:
        return graph
    nodes = dict(graph.nodes)
    nodes.update(extra_nodes)
    return Graph(nodes=nodes, edges=new_edges)


def materialize_output_items(graph: Graph) -> Graph:
    """Give every extra ITEM of a Page Output (``data_2`` … ``data_8``) a Page Output node of
    its own (V4.00 step 11f): ``__item__<output>__data_2`` with the Output's params and that
    one wire in its ``data``. The Output keeps its first item — so pulling it previews that
    and computes no other — and a Page Input's item tap reads the node its item became. The
    input graph is not mutated; with no items the same object is returned."""
    extra: Dict[str, NodeInstance] = {}
    new_edges = []
    for e in graph.edges:
        node = graph.nodes.get(e.dst)
        if node is None or node.op_key != PAGE_OUTPUT_OP or e.kind != "forward" \
                or e.dst_socket not in PAGE_ITEM_SOCKETS[1:]:
            new_edges.append(e)
            continue
        nid = output_item_node_id(e.dst, e.dst_socket)
        if nid not in extra:
            extra[nid] = NodeInstance(nid, PAGE_OUTPUT_OP, params=dict(node.params),
                                      modes=dict(node.modes))
        new_edges.append(Edge(e.src, nid, e.src_socket, "data", e.kind))
    if not extra:
        return graph
    nodes = dict(graph.nodes)
    nodes.update(extra)
    return Graph(nodes=nodes, edges=new_edges)


def materialize_input_items(graph: Graph) -> Graph:
    """Rewire every wire leaving a Page Input's ``item:<name>`` socket through a Page Input
    TAP of its own that reads that item (``__tap__<input>__item_<name>``, the item's name in
    :data:`PAGE_ITEM_KEY`) — one per (input, item), shared by every wire leaving it. The
    Workspace seeds and splices the tap like any Page Input (V4.00 step 11f)."""
    taps: Dict[str, NodeInstance] = {}
    new_edges = []
    for e in graph.edges:
        node = graph.nodes.get(e.src)
        m = ITEM_SOCKET_RE.match(e.src_socket) \
            if node is not None and node.op_key == PAGE_INPUT_OP and e.kind == "forward" \
            else None
        if m is None:
            new_edges.append(e)
            continue
        name = m.group(1)
        tid = input_item_tap_id(e.src, name)
        if tid not in taps:
            taps[tid] = NodeInstance(tid, PAGE_INPUT_OP, params={
                PAGE_SOURCE_KEY: node.params.get(PAGE_SOURCE_KEY, ""), PAGE_ITEM_KEY: name})
        new_edges.append(Edge(tid, e.dst, "out", e.dst_socket, e.kind))
    if not taps:
        return graph
    nodes = dict(graph.nodes)
    nodes.update(taps)
    return Graph(nodes=nodes, edges=new_edges)


def outside_tap_id(node_id: str) -> str:
    """The sibling a wire leaving a Crop by Region's ``outside`` socket is served by."""
    return f"__tap__{node_id}__outside"


def materialize_outside_taps(graph: Graph) -> Graph:
    """Rewire every wire leaving a ``util.crop_region`` card's ``outside`` socket through a
    SIBLING of that node (``__tap__<node>__outside``): the same op, params and modes with
    ``keep = outside``, fed by copies of every wire the crop node itself receives — so the
    inverse is computed from the same inputs, independently memoized, and shared by every
    wire that leaves the socket (V4.00 step 12). Runs FIRST among the tap passes, so a copied
    incoming ``chK``/``posK`` edge is materialized afterwards like the original. The input
    graph is not mutated; with no such wires the same object is returned."""
    taps: Dict[str, NodeInstance] = {}
    new_edges = []
    incoming: Dict[str, list] = {}
    for e in graph.edges:
        incoming.setdefault(e.dst, []).append(e)
    for e in graph.edges:
        node = graph.nodes.get(e.src)
        if node is None or node.op_key != CROP_REGION_OP or e.kind != "forward" \
                or e.src_socket != OUTSIDE_SOCKET:
            new_edges.append(e)
            continue
        tid = outside_tap_id(e.src)
        if tid not in taps:
            taps[tid] = NodeInstance(tid, CROP_REGION_OP, params=dict(node.params),
                                     modes={**dict(node.modes),
                                            CROP_KEEP_MODE: CROP_KEEP_OUTSIDE})
            for p in incoming.get(e.src, ()):
                new_edges.append(Edge(p.src, tid, p.src_socket, p.dst_socket, p.kind))
        new_edges.append(Edge(tid, e.dst, "out", e.dst_socket, e.kind))
    if not taps:
        return graph
    nodes = dict(graph.nodes)
    nodes.update(taps)
    return Graph(nodes=nodes, edges=new_edges)


def materialize_part_taps(graph: Graph) -> Graph:
    """Rewire every wire leaving a ``part:<name>`` socket through a ``data.part`` tap
    (``__tap__<node>__part_<name>``) fed by the node's real Dataset output — one per (node,
    part), shared by every wire leaving it, so four branches reading one mask cost one tap
    and one memo entry (V4.00 step 11f)."""
    taps: Dict[str, NodeInstance] = {}
    new_edges = []
    for e in graph.edges:
        m = PART_SOCKET_RE.match(e.src_socket) if e.kind == "forward" else None
        if m is None or e.src not in graph.nodes:
            new_edges.append(e)
            continue
        name = m.group(1)
        tid = f"__tap__{e.src}__part_{name}"
        if tid not in taps:
            taps[tid] = NodeInstance(tid, PART_OP, params={PART_KEY: name})
            new_edges.append(Edge(e.src, tid, _real_dataset_out(graph.nodes[e.src].op_key),
                                  "data", "forward"))
        new_edges.append(Edge(tid, e.dst, "out", e.dst_socket, e.kind))
    if not taps:
        return graph
    nodes = dict(graph.nodes)
    nodes.update(taps)
    return Graph(nodes=nodes, edges=new_edges)


def materialize_channel_taps(graph: Graph) -> Graph:
    """Return a runnable graph in which every GUI-synthetic per-channel output edge
    (``src_socket`` matching ``chK``) is rewired through a real ``channel.select`` tap.

    One tap node is inserted per ``(source_node, channel_index)`` and shared by every
    edge leaving that ``chK`` socket. Full-bundle edges (``image`` / ``out`` / value
    sockets) pass through unchanged. The input graph is not mutated; if there are no
    channel taps the same graph object is returned.
    """
    return _materialize_taps(graph, CH_SOCKET_RE, "channel.select", "c",
                             lambda node, k: {"channels": [k]})


def materialize_group_taps(graph: Graph) -> Graph:
    """Rewire every GUI-synthetic per-GROUP output edge (``grpK``) through a real
    ``util.select_group`` tap — the multipoint twin of :func:`materialize_channel_taps`.

    **The tap carries the group's KEY, never its index**, resolved here from the card's
    :data:`~nodelab_v2.document.GROUPS_KEY` list. That is the whole reason this function
    reads the source node's params at all, and it is what makes the arrangement survive a
    re-detection: if a sidecar appears and renames or reorders the groups, an index would
    quietly point ``grp2`` at a different specimen while every hash stayed put, whereas a
    key either still names a group or refuses. It also means the run graph reads the way the
    user thinks — ``group="treated"`` on the tap, not ``group="2"``.

    A card with no resolved list leaves its edges untouched (see :func:`_materialize_taps`),
    so nothing is invented for a file whose grouping could not be worked out.
    """
    def params_for(node, k):
        got = (node.params.get(GROUPS_KEY) if node is not None else None) or []
        if not isinstance(got, (list, tuple)) or not (0 <= k < len(got)):
            return None
        key = str((got[k] or {}).get("key") or "")
        return {"group": key} if key else None

    return _materialize_taps(graph, GRP_SOCKET_RE, "util.select_group", "g", params_for)


def materialize_batch_taps(graph: Graph) -> Graph:
    """Rewire every GUI-synthetic per-MEMBER output edge (``batK``) on a ``util.unbatch``
    card through a real ``util.select_batch`` tap — the batch-axis twin of
    :func:`materialize_channel_taps`.

    **The tap carries the member's NAME, never its index**, for the reason
    :func:`materialize_group_taps` gives and one more besides: a batch's whole purpose is
    keeping files apart, so an index that silently slid onto a different file after a
    rewire would defeat the feature rather than merely surprise someone. A name that no
    longer resolves makes ``util.select_batch`` refuse with the real members listed.

    ``_materialize_taps`` calls ``params_for`` with the SOURCE node of the edge — the
    unbatch card — but a member's name lives on the batch node upstream of it, so the walk
    is done here against the whole graph and keyed by the unbatch's id.
    """
    cache: Dict[str, List[str]] = {}

    def params_for(node, k):
        nid = getattr(node, "id", None)
        if nid is None:
            return None
        names = cache.get(nid)
        if names is None:
            names = cache[nid] = batch_member_names_of(graph, nid)
        if not (0 <= k < len(names)):
            # the batch shrank under this card: leave the edge pointing at a socket that
            # is not there, which the document's validation reports, rather than
            # rewriting it onto whichever file now occupies that slot
            return None
        return {"member": names[k]}

    return _materialize_taps(graph, BAT_SOCKET_RE, "util.select_batch", "b", params_for)


def materialize_position_taps(graph: Graph) -> Graph:
    """Rewire every GUI-synthetic per-POSITION output edge (``posK``) on a
    ``util.split_positions`` card through a real ``util.select_position`` tap — the M-axis
    twin of :func:`materialize_channel_taps` (2026-10-02).

    **The tap carries the position's INDEX**, unlike the group and batch taps. A position's
    index in its file IS its identity — the acquisition's point list is fixed when the file
    is written and every per-M list is addressed by it — while a point name is an optional
    label many files lack; so ``posK`` means "the (K+1)-th position of whatever is wired",
    exactly as ``chK`` means the (K+1)-th channel. ``util.select_position`` refuses an
    index past the end with the positions listed, so a split rewired onto a smaller file
    says so rather than silently serving another position.
    """
    return _materialize_taps(graph, POS_SOCKET_RE, "util.select_position", "p",
                             lambda node, k: {"position": str(k)})


def materialize_plane_taps(graph: Graph) -> Graph:
    """Rewire every GUI-synthetic per-PLANE output edge (``zK``) on a ``util.split_z``
    card through a real ``util.select_plane`` tap — the Z-axis twin of
    :func:`materialize_position_taps` (2026-10-07).

    The tap carries the plane's INDEX: a plane has no name, and ``zK`` means "the (K+1)-th
    plane of whatever is wired", exactly as ``chK`` means the (K+1)-th channel.
    ``util.select_plane`` refuses an index past the end with the stack's depth, so a split
    rewired onto a shallower stack says so rather than silently serving another plane.
    """
    return _materialize_taps(graph, Z_SOCKET_RE, "util.select_plane", "z",
                             lambda node, k: {"plane": int(k)})


def materialize_frame_taps(graph: Graph) -> Graph:
    """Rewire every GUI-synthetic per-FRAME output edge (``tK``) on a ``util.split_t`` card
    through a real ``util.select_frame`` tap — the T-axis twin of
    :func:`materialize_plane_taps` (2026-10-07). The tap carries the frame's INDEX."""
    return _materialize_taps(graph, T_SOCKET_RE, "util.select_frame", "t",
                             lambda node, k: {"frame": int(k)})


def materialize_frame_groups(graph: Graph) -> Graph:
    """Rewire every ``tgK`` edge on a Split T card through a ``util.crop`` tap in frames mode
    keeping group K's timepoints (``t0-9``) — the usual way to split a long series
    (2026-10-07)."""
    from nodegraph.metadata import format_indices

    def params_for(node, k):
        g = split_group(node, k)
        return None if g is None else {"frames": "t" + format_indices(sorted(set(g[1])))}
    return _materialize_taps(graph, TG_SOCKET_RE, "util.crop", "tg", params_for,
                             modes={"region": "frames"})


def materialize_channel_groups(graph: Graph) -> Graph:
    """Rewire every ``chgK`` edge on a Split Channels card through a ``channel.select`` tap
    carrying group K's channel LIST (2026-10-07)."""
    def params_for(node, k):
        g = split_group(node, k)
        return None if g is None else {"channels": list(g[1])}
    return _materialize_taps(graph, CHG_SOCKET_RE, "channel.select", "cg", params_for)


def materialize_position_groups(graph: Graph) -> Graph:
    """Rewire every ``posgK`` edge on a Split Positions card through a ``util.crop`` tap in
    frames mode keeping group K's positions (``m0-2``), 2026-10-07. Crop rather than a
    widened Select Position because its frames mode already carries every rule a frame
    subset needs — per-M lists, structure rows, origin — and refuses past the end."""
    from nodegraph.metadata import format_indices

    def params_for(node, k):
        g = split_group(node, k)
        return None if g is None else {"frames": "m" + format_indices(sorted(set(g[1])))}
    return _materialize_taps(graph, POSG_SOCKET_RE, "util.crop", "pg", params_for,
                             modes={"region": "frames"})


def materialize_plane_groups(graph: Graph) -> Graph:
    """Rewire every ``zgK`` edge on a Split Z card through a ``util.crop`` tap in frames mode
    keeping group K's planes (``z4-7``) — a sub-stack, with ``z_step_um``, ``z_home_index``
    and ``origin_um`` following as Crop's frames mode has always moved them (2026-10-07)."""
    from nodegraph.metadata import format_indices

    def params_for(node, k):
        g = split_group(node, k)
        return None if g is None else {"frames": "z" + format_indices(sorted(set(g[1])))}
    return _materialize_taps(graph, ZG_SOCKET_RE, "util.crop", "zg", params_for,
                             modes={"region": "frames"})


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
        pre = materialize_channel_taps(materialize_group_taps(graph))
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
           "PAGE_INPUT_OP", "PAGE_OUTPUT_OP", "PAGE_OPS", "PAGE_SOURCE_KEY", "PAGE_NAME_KEY",
           "PAGE_CONDITION_KEY", "PAGE_CONDITION_AUTO_KEY", "PAGE_UNBOUND_MESSAGE",
           "DEFAULT_OUTPUT_BASE", "sanitize_output_name", "next_free_name",
           "PART_OP", "PART_KEY", "PART_IMAGE", "PART_SOCKET_RE", "part_socket", "part_of",
           "PAGE_ITEMS_KEY", "PAGE_ITEM_SOCKETS", "ITEM_SOCKET_RE", "PAGE_ITEM_KEY",
           "item_socket", "output_item_node_id", "input_item_tap_id",
           "materialize_output_items", "materialize_input_items", "materialize_part_taps",
           "CROP_REGION_OP", "OUTSIDE_SOCKET", "outside_tap_id", "materialize_outside_taps",
           "materialize_group_taps", "GRP_SOCKET_RE", "GROUPS_KEY",
           "prepare_run_graph", "cut_docked_inputs", "dock_seeds", "dock_status",
           "dormant_nodes", "docked_nodes", "upstream_signature", "dock_state_of",
           "dock_store_of", "bake_record", "is_docked", "is_frozen", "held_nodes",
           "CH_SOCKET_RE", "DOCK_OP", "BAKE_KEY", "DOCK_LIVE", "DOCK_HELD",
           "DOCK_DOCKED", "DOCK_FROZEN", "PRECISION_UNSET",
           "LOAD_OP", "ACCESS_MODE", "ACCESS_AUTO", "ACCESS_INGEST", "ACCESS_DIRECT",
           "ACCESS_DEFAULT", "source_access_of",
           "BAT_SOCKET_RE", "batch_member_identity", "batch_member_names_of",
           "materialize_batch_taps",
           "POS_SOCKET_RE", "materialize_position_taps",
           "Z_SOCKET_RE", "materialize_plane_taps",
           "CHG_SOCKET_RE", "POSG_SOCKET_RE", "ZG_SOCKET_RE", "split_group",
           "materialize_channel_groups", "materialize_position_groups",
           "materialize_plane_groups",
           "T_SOCKET_RE", "TG_SOCKET_RE", "materialize_frame_taps",
           "materialize_frame_groups"]
