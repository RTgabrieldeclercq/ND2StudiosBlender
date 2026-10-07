"""The GraphDocument — NodeLab v2's Qt-free editing model (Phase 5).

The canvas is a *view*; this is the source of truth: node records (op_key + params +
modes, the same dicts the graphics items and inspector mutate in place), edges, and
GUI-only extras (canvas position, mute). It builds a real
:class:`nodegraph.graph.Graph` on demand, runs the **edit-time MetaEnvelope pass**
(:func:`nodegraph.metadata.propagate_meta`) after every structural edit — the G8 "live
widget re-seed" — and round-trips ``*.nd2graph.json`` through
:mod:`nodegraph.serialize` (GUI extras ride in a top-level ``ui`` object the headless
loader ignores).

Wiring rules (G1): a connection must pass :func:`nodegraph.sockets.can_connect` on the
two sockets' *active* specs, must not create a cycle, and a second wire into a
non-multi input **replaces** the existing one (Blender behavior). Mute (G3) is resolved
at graph-build time for runs: a muted node with a Dataset input is bypassed
(pass-through), so the engine never sees it.

Qt-free; standard library + nodegraph only (testable headless).
"""
from __future__ import annotations

import itertools
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from nodegraph.domains import AXIS_ORDER, Domain
from nodegraph.graph import Edge, Graph, NodeInstance
from nodegraph.groups import (
    GROUP_INPUT, GROUP_OUTPUT, Group, expand as _group_expand, group_name_of,
    inst_id as _inst_id,
)
from nodegraph.iterate import (
    ITERATE_OP, MAX_VARIABLES as _MAX_VARIABLES, MODE_TARGET_PREFIX,
    SEG_FROM as _SEG_FROM, SEG_TO as _SEG_TO, SWEEP_KEY as _SWEEP_KEY,
    TYPE_NUMBER as _TYPE_NUMBER, TYPE_TEXT as _TYPE_TEXT,
    aliases as _iterate_aliases, unroll as _iterate_unroll,
    var_mode_names as _var_mode_names, var_out_names as _var_out_names,
)
from nodegraph.metadata import MetaEnvelope, propagate_meta
from nodegraph.registry import InDataset, InString, NODES, OutDataset, layer_value
from nodegraph.serialize import (
    from_dict as _ng_from_dict, page_from_dict as _ng_page_from_dict, to_dict as _ng_to_dict)
from nodegraph.sockets import (
    Direction, SocketType, can_connect as _can_connect, can_convert as _can_convert)
from nodegraph.zones import Zone, unroll as _unroll
from nodegraph.memo import digest
from nodelab_v2.ops import (
    DEFAULT_OUTPUT_BASE, PAGE_NAME_KEY, PAGE_OUTPUT_OP, next_free_name,
    sanitize_output_name,
    BAT_SOCKET_RE, CH_SOCKET_RE, GRP_SOCKET_RE, ITEM_SOCKET_RE, PAGE_INPUT_OP,
    PAGE_ITEM_SOCKETS, PAGE_ITEMS_KEY, PAGE_SOURCE_KEY, POS_SOCKET_RE,
    PART_IMAGE, PART_SOCKET_RE, item_socket, part_socket, _real_dataset_out,
    BAKE_KEY, DOCK_DOCKED, DOCK_HELD, DOCK_LIVE, DOCK_OP, GROUPING_AUTO,
    GROUPING_DEFAULT, GROUPING_MODE, GROUPING_OFF, GROUPS_KEY, LOAD_OP,
    calib_overrides,
    dock_status as _dock_status,
    batch_member_identity,
    dormant_nodes, is_docked, is_frozen, prepare_run_graph, upstream_signature)

#: params key holding the sticky pinned-override list (serialized per V2.03; stripped
#: from the params handed to the ENGINE — it is a UI annotation, not a compute input).
LOCKED_KEY = "__locked__"

#: per-instance UI annotations (serialized, stripped from the params handed to the ENGINE
#: exactly like LOCKED_KEY): a source node's file-name display title, and the captured
#: per-channel descriptor list ``[{name, emission_nm, color}, …]`` that drives its
#: synthetic per-channel output sockets and their wire tints.
TITLE_KEY = "__title__"
CHANNELS_KEY = "__channels__"

#: An ``io.load`` card's **file bundle** members: a list of 2+ paths laid end to end on the
#: multipoint axis by :class:`~nodegraph.provider.MultiSourceProvider`. Fewer than 2 entries
#: (or absent) is the ordinary single-file card that ``path`` describes, and ``path`` stays
#: populated with the first member either way, so a reader that predates bundles still names
#: a real file.
#:
#: Deliberately NOT in :data:`_UI_PARAM_KEYS`: this is not a UI annotation, it decides which
#: pixels the card produces, so it belongs in the params the memo keys on.
BUNDLE_PATHS_KEY = "paths"

#: op_keys whose GUI card grows one synthetic per-channel output socket (``ch0…``) per
#: channel — materialized into ``channel.select`` taps at graph-build (see nodelab_v2.ops).
#:
#: ``page.input`` since V4.00 step 11d: a later page reads an image through its Page Input,
#: and that card is where the user picks a channel on that page, exactly as the Load card
#: is on Image Input. Its channel list is the upstream Output's
#: (:attr:`GraphDocument.page_channels`); a ``chK`` edge on it materializes like any other,
#: and the composer splices the tap onto the upstream Output with the Input itself.
CHANNEL_TAP_OPS = ("io.load", "channel.split", "page.input")

#: The Dataset socket names that say nothing but "the data" — ``data`` in, ``out`` (and a
#: source's ``image``) out. A stream carrying a strict subset of the file's channels names
#: itself on such a socket (``Cy5`` instead of ``data``, V4.00 step 11g); a socket with a
#: name of its own (``raw``, ``reference``, ``areas``) keeps it and adds the channel
#: (``raw · Cy5``), because that name says which role the wire plays on the node.
GENERIC_DATASET_SOCKETS = frozenset({"data", "out", "image"})

#: A SYNTHETIC output socket — a per-channel, per-group, per-position or per-member tap, a
#: part, an item — whose label already says what it carries (``0 · DAPI``, ``mask only``,
#: ``item · cells``). It keeps that label: the channel is printed on the ``out`` above it,
#: and ``0 · DAPI · DAPI`` says nothing twice.
_SYNTHETIC_SOCKET_RE = re.compile(r"^(?:ch|grp|pos|bat)\d+$|^(?:part|item):")

#: A Page Output's name while nobody has named it: ``out``, ``out2``, … (what
#: ``Workspace.node_defaults`` hands a fresh card). Such a name — or a blank one — is
#: replaced by the first wire's (:meth:`GraphDocument._name_output_from_wire`, V4.00 step
#: 11h); a name the user typed never is.
_PLACEHOLDER_NAME_RE = re.compile(rf"^{re.escape(DEFAULT_OUTPUT_BASE)}\d*$")


def _join_names(names) -> str:
    """How a card prints a few names: ``Cy5``; ``DAPI · GFP``; ``DAPI +2`` for three or
    more (positions and channels alike)."""
    names = [str(n) for n in names if str(n)]
    if not names:
        return ""
    if len(names) <= 2:
        return " · ".join(names)
    return f"{names[0]} +{len(names) - 1}"

#: op_keys whose GUI card grows one synthetic per-GROUP output socket (``grp0…``) per
#: position group — materialized into ``util.select_group`` taps at graph-build.
#:
#: Only the SOURCE, unlike :data:`CHANNEL_TAP_OPS`. A channel tap is meaningful anywhere a
#: channel axis survives, but grouping is a property of the acquisition's stage log, and the
#: place a user thinks about "which specimen" is where the file enters the graph. A node
#: downstream that wants it has ``util.select_group`` to wire explicitly.
GROUP_TAP_OPS = ("io.load",)

#: The three op_keys of the batch axis (V3.01). ``util.unbatch``'s card grows one synthetic
#: per-MEMBER output socket (``bat0…``) per file in the batch reaching it, materialized into
#: ``util.select_batch`` taps at graph-build — the third member of the same family as
#: :data:`CHANNEL_TAP_OPS` and :data:`GROUP_TAP_OPS`.
#:
#: Unlike those two the member list is NOT captured into params: it is resolved live from
#: the wiring by :meth:`GraphDocument.batch_member_names`, because it describes the graph
#: rather than a file, and a captured copy could disagree with the wires after a rewire.
BATCH_OP = "util.batch"
UNBATCH_OP = "util.unbatch"
SELECT_BATCH_OP = "util.select_batch"

#: op_keys whose GUI card grows one synthetic per-MEMBER output socket (``bat0…``).
BATCH_TAP_OPS = (UNBATCH_OP,)

#: op_keys whose GUI card grows one synthetic per-POSITION output socket (``pos0…``) per
#: multipoint on the wire reaching it — materialized into ``util.select_position`` taps at
#: graph-build (2026-10-02), the M-axis member of the same family as
#: :data:`CHANNEL_TAP_OPS`. Resolved from the edit-time ENVELOPE
#: (:meth:`GraphDocument.position_descriptors`): the position count is an axis, which the
#: envelope pass predicts exactly, and the names are the per-M ``position_name`` list the
#: loader stamps — so neither is captured into params.
POSITION_TAP_OPS = ("util.split_positions",)



#: Columns of a JOINED domain that a ``column_in`` picker must not offer, because the
#: consuming compute uses them to perform the join rather than exposing them as a testable
#: value. ``member_id`` is the Track—member key: ``analysis.if_else._track_joined``
#: skips it (and ``t``, which the member table already carries), so offering it would put a
#: name in the menu that the pull then refuses — the live-control-that-does-nothing
#: defect, wearing a picker's clothes. Kept here rather than in the envelope because it is a
#: fact about how a consumer joins, not about what a producer wrote.
_JOIN_KEYS = frozenset({"member_id"})

#: params keys that are UI-only and must be stripped before the graph runs. ``__sweep__``
#: (an Iterate node's recorded results table) is here and NOT with ``io.dock``'s ``__bake__``
#: for one decisive reason: a bake record is written by an explicit action and MUST re-key
#: the memo, whereas a sweep record is written from the payload of every pull — leaving it
#: in the run params would make each pull invalidate the entry it just created and re-run
#: the sweep forever.
_UI_PARAM_KEYS = (LOCKED_KEY, TITLE_KEY, CHANNELS_KEY, _SWEEP_KEY)

#: Every ``flow.iterate`` variable OUTPUT socket name. A wire leaving one of these is a
#: DRIVER wire (:data:`nodegraph.graph.NON_DAG_KINDS`), and that is decided *structurally*
#: rather than stored: these sockets exist on no other node type, so the document keeps
#: driver wires as ordinary 4-tuples in ``self.edges`` — the canvas draws them, undo and
#: copy/paste move them and the file round-trips them with no new storage anywhere. Only
#: :meth:`GraphDocument.to_graph` needs to know, and it re-derives the kind here.
_DRIVER_SOCKETS: frozenset = frozenset(
    n for k in range(4) for n in _var_out_names(k))


def _group_socket_label(i: int, g) -> str:
    """The text a ``grpK`` output socket shows: ``"G2 · 9 pos, 3x3"``.

    The group's KEY leads, not the index, because the key is what the user types into
    ``util.select_group`` and what a renamed group in a sidecar is called — a socket reading
    ``2`` beside a node reading ``treated`` would be one more thing to map by hand. The size
    is there because it is the one number that says at a glance whether the detection did
    something sensible: six sockets all reading "9 pos" is a mosaic set, and one reading
    "1 pos" is a stray position worth looking at.
    """
    key = str((g or {}).get("key") or f"G{i + 1}")
    size = int((g or {}).get("size") or 0)
    shape = str((g or {}).get("shape") or "")
    bits = [f"{size} pos"] if size else []
    if shape:
        bits.append(shape.split(" ")[0])          # "3x3" — the order word does not fit
    return f"{key} · {', '.join(bits)}" if bits else key


#: Ops never switched off (V4.00 step 11e): page, group and zone boundaries, the Iterate
#: controls and reroutes are the graph's wiring, not steps of the analysis.
NO_PASS_THROUGH_PREFIXES = ("page.", "group.", "zone.", "flow.", "rr.")
#: what a domain a node ADDS is, in the words of the refusal the user reads
_ADDED_WORDS = {"voxel": "a mask or raster layer", "label": "labelled objects",
                "point": "points", "track": "tracks", "mesh": "a mesh",
                "frame": "per-frame values", "timepoint": "per-timepoint values",
                "plane": "per-plane values", "multipoint": "per-position values",
                "batch": "per-file values", "global": "a summary value",
                "channel": "per-channel values"}


def pass_through_reason(spec, params: Optional[Mapping[str, Any]] = None,
                        modes: Optional[Mapping[str, str]] = None) -> str:
    """Why a node of ``spec`` with these values may NOT be switched off — ``""`` when it may
    (V4.00 step 11e).

    Switching a node off (mute) hands its first Dataset input straight to its consumers, so
    it is only honest for a node that does not change the KIND of data: what flows out must
    be what flowed in, modified. A filter, a crop, a projection, a resample, a channel pick
    or a writer qualifies; a node that adds a mask, labels, points, tracks, a mesh, table
    columns or a picture does not — switched off, everything downstream that reads what it
    adds would get nothing, and the page would fail somewhere other than where the switch
    is. Decided from the node's DECLARATIONS for its current values (``adds_domains``, an
    active ``layer_out`` socket with a name, ``extra_layers``, ``adds_columns``,
    ``fresh_output``), so it needs no pull and no wiring."""
    if spec is None:
        return "it is a group or an unknown node — ungroup it to switch its steps off"
    if spec.op_key.startswith(NO_PASS_THROUGH_PREFIXES):
        return "it is part of the graph's wiring (a page, group or zone boundary), not a step"
    state = dict(spec.default_state())
    state.update(modes or {})
    params = dict(params or {})
    ins, outs = spec.active_inputs(state), spec.active_outputs(state)
    if not any(s.type is SocketType.DATASET for s in ins):
        return "it has no data input to pass on — it is where its data starts"
    if not any(s.type is SocketType.DATASET for s in outs):
        return "it has no data output to pass its input on to"
    if getattr(spec, "fresh_output", False):
        return "it makes a new kind of data (a picture) rather than changing its input"
    if spec.adds_domains:
        what = sorted(_ADDED_WORDS.get(d.value, d.value) for d in spec.adds_domains)
        what = what[0] if len(what) == 1 else ", ".join(what[:-1]) + " and " + what[-1]
        return f"it adds {what} — switched off, whatever reads that downstream gets none"
    for s in ins:
        if getattr(s, "layer_out", None):
            try:
                v = layer_value(s, params)
            except Exception:                         # noqa: BLE001 — a declaration that raises
                v = ""
            if v:
                return f"it adds the layer “{v}”"
    extra = getattr(spec, "extra_layers", None)
    if extra is not None:
        try:
            if any(isinstance(nm, str) and nm for _d, nm in (extra(params, state) or ())):
                return "it adds layers"
        except Exception:                             # noqa: BLE001
            return "it adds layers"
    cols = getattr(spec, "adds_columns", None)
    if cols is not None:
        try:
            got = (cols(params, state, (), ()) if getattr(cols, "wants_inputs", False)
                   else cols(params, state, ()))
        except Exception:                             # noqa: BLE001
            got = True
        if got:
            return "it adds table columns"
    return ""


def is_driver_edge(doc: "GraphDocument", edge: EdgeTuple) -> bool:
    rec = doc.nodes.get(edge[0])
    return (rec is not None and rec.op_key == ITERATE_OP
            and edge[1] in _DRIVER_SOCKETS)


class NodeRecord:
    """One placed node. ``params``/``modes`` are the LIVE dicts the GUI mutates."""

    __slots__ = ("id", "op_key", "params", "modes", "x", "y", "muted", "collapsed")

    def __init__(self, node_id: str, op_key: str, *,
                 params: Optional[dict] = None, modes: Optional[dict] = None,
                 x: float = 0.0, y: float = 0.0, muted: bool = False,
                 collapsed: bool = False) -> None:
        self.id = node_id
        self.op_key = op_key
        self.params: Dict[str, Any] = dict(params or {})
        self.modes: Dict[str, str] = dict(modes or {})
        self.x, self.y = float(x), float(y)
        self.muted = bool(muted)
        self.collapsed = bool(collapsed)

    def spec(self):
        return NODES.get(self.op_key)

    def state(self) -> Dict[str, str]:
        spec = self.spec()
        base = spec.default_state() if spec is not None else {}
        base.update(self.modes)
        return base

    @property
    def locked(self) -> set:
        return set(self.params.get(LOCKED_KEY, ()))

    def set_locked(self, names: set) -> None:
        if names:
            self.params[LOCKED_KEY] = sorted(names)
        else:
            self.params.pop(LOCKED_KEY, None)


class FrameRecord:
    """A labelled frame grouping a set of nodes on the canvas — **GUI-only** (it never
    enters the run graph; it rides in the ``ui`` extras like node positions). A frame
    auto-sizes to enclose its member nodes; it always has ≥1 member (an emptied frame is
    removed), so it needs no stored geometry."""

    __slots__ = ("id", "title", "members", "color")

    def __init__(self, frame_id: str, title: str = "Frame",
                 members=(), color=None) -> None:
        self.id = frame_id
        self.title = str(title)
        self.members: List[str] = list(members)
        self.color = tuple(int(v) for v in color) if color else None


EdgeTuple = Tuple[str, str, str, str]        # (src, src_socket, dst, dst_socket)


#: source of :attr:`GraphDocument.uid`
_DOC_UIDS = itertools.count(1)


class GraphDocument:
    """The editable model + envelope cache. ``on_change`` callbacks fire after every
    structural edit or re-propagation (the canvas/inspector re-seed from them)."""

    #: Can nodes, edges and frames be added, removed or rewired here? A page LINKED to a
    #: master (V4 step 6) answers ``False`` and takes parameter edits only; the GUI checks
    #: this before offering a topology edit. A class attribute so the check costs nothing.
    editable_topology: bool = True

    def __init__(self) -> None:
        self.nodes: Dict[str, NodeRecord] = {}
        self.edges: List[EdgeTuple] = []
        self.meta_seeds: Dict[str, MetaEnvelope] = {}
        self.envs: Dict[str, MetaEnvelope] = {}
        self.path: Optional[str] = None           # last save/load file
        self.revision = 0                          # bumped on every edit
        #: unique per document INSTANCE, process-wide (V4.00 step 2). A page's run identity
        #: digests it beside the revision: two document objects can share a revision number
        #: — every fresh page a File → Open builds starts again at 0 — and an identity that
        #: repeats would hand a new file's pull the previous file's cached engine and results.
        self.uid = next(_DOC_UIDS)
        # zones/groups + zone back-edges aren't GUI-editable yet, but a loaded file's
        # are carried through VERBATIM so re-saving never silently destroys them
        # (review 2026-07-22 BLOCKER). The GUI reads/writes only the FORWARD nodes+
        # edges layer; `self.edges` holds forward 4-tuples the canvas draws, while a
        # loaded back-edge (Out→In zone feedback) rides in `_back_edges` untouched.
        self._zones: list = []
        self._groups: list = []
        self._back_edges: List[Edge] = []
        # GUI-only labelled frames (canvas organization; never enter the run graph —
        # they ride in the `ui` extras exactly like node positions).
        self.frames: Dict[str, FrameRecord] = {}
        #: nodes a docked run no longer evaluates — the canvas greys these (V2.18).
        #: Recomputed by :meth:`propagate`, so it is always current with the last edit.
        self.dormant: frozenset = frozenset()
        #: (node_id, revision, held) → dock status, for the per-repaint caller (dock_status)
        self._dock_status_cache: Dict[Tuple[Any, ...], Tuple[str, str]] = {}
        #: the node ids the RUNNER currently holds a payload for, mirrored here because the
        #: document is what the canvas asks for a dock's badge. Set by
        #: :meth:`set_held_nodes`; never persisted — a hold does not survive a reload, and
        #: writing it to the file would be the durability lie the `released` badge exists to
        #: avoid.
        self._held_nodes: frozenset = frozenset()
        self._listeners: List[Callable[[], None]] = []
        #: GUI-only: called with a node id when its card MOVED (:meth:`set_pos`). Not a model
        #: edit — no revision, no run disturbed — but a linked page and its master show the
        #: same cards in the same places (V4.00 step 6), so each canvas follows the other.
        self._move_listeners: List[Callable[[str], None]] = []
        #: The nodes the most recent edit touched, or ``None`` for "unknown / everything".
        #: Read by listeners during their change callback (see :meth:`_notify`); meaningless
        #: outside one, since the next edit overwrites it.
        self.last_touched: Optional[frozenset] = None
        # ── V4.00 workspace hooks ── installed by :class:`nodelab_v2.workspace.Workspace`
        # when this document becomes a PAGE; each is a plain default here, so a document
        # outside a workspace behaves exactly as before.
        #: Extra envelope seeds for ROOTS, consulted by :meth:`propagate` right after the
        #: dock seeds: how a ``page.input`` root gets the upstream page's Output envelope.
        self.seed_hooks: List[Callable[[], Mapping[str, MetaEnvelope]]] = []
        #: The page kind the palette and the link-drag search filter by (``None`` = no
        #: filter, the pre-V4 behaviour and the ``free`` page).
        self.page_kind: Optional[str] = None
        #: A sub-folder tag for :meth:`default_dock_store` — the page id, so two pages that
        #: both dock their ``n3`` never bake into one folder.
        self.store_tag: str = ""
        #: ``() -> [(value, label), ...]``: the named outputs of earlier pages a
        #: ``page.input`` on this document may read (:meth:`source_choices`).
        self.page_sources: Callable[[], list] = lambda: []
        #: ``(op_key) -> dict`` (V4.00 step 11): the params a node of ``op_key`` STARTS with
        #: when its creator gives none for them — the Workspace names a Page Output and binds
        #: a Page Input. :meth:`add_node` merges it UNDER the caller's ``params``, so an
        #: explicit value always wins and a loaded file keeps its own.
        self.node_defaults: Callable[[str], dict] = lambda _op: {}
        #: ``() -> [(page id, page name), ...]``: the pages a ``page.input`` on this document
        #: may read, nearest first — the inspector's "Go to <page>" (V4.00 step 11).
        self.page_feeders: Callable[[], list] = lambda: []
        #: ``(node_id) -> [{name, emission_nm, color}, ...]`` (V4.00 step 11d): the channels
        #: of the Output a ``page.input`` on this document reads — the file's real channel
        #: names with their native colours (the envelope carries the names since step 11g,
        #: not the colours). ``[]`` outside
        #: a workspace or while the Input is unbound.
        self.page_channels: Callable[[str], list] = lambda _nid: []
        #: ``(channel descriptors, the source FILE's channel total)`` of what a Page Input —
        #: or one ITEM of the several-item Output it reads — carries, resolved by the
        #: Workspace through the pages (V4.00 step 11g). ``([], 0)`` detached or unbound. The
        #: total is what a later page needs to know that a one-channel stream is ONE OF three
        #: and so to name it; its own roots are its Page Inputs, which know no file.
        self.page_channel_scope: Callable[..., Tuple[list, int]] = (
            lambda _nid, _item="": ([], 0))
        #: The same for POSITIONS: ``(position names, the source file's position total)`` of
        #: what a Page Input — or one item of its Output — carries (V4.00 step 11h).
        self.page_position_scope: Callable[..., Tuple[list, int]] = (
            lambda _nid, _item="": ([], 0))
        #: ``(old, new)``: a Page Output on this document was renamed by its first wire
        #: (:meth:`_name_output_from_wire`); the Workspace points every reader of
        #: ``<page>:old`` at ``new`` (V4.00 step 11h). A no-op outside a workspace.
        self.output_renamed: Callable[[str, str], None] = lambda _old, _new: None
        #: ``(node_id) -> [item name, ...]`` (V4.00 step 11f): the items of the Output a
        #: ``page.input`` on this document reads — its item sockets. ``[]`` outside a
        #: workspace, while unbound, or for a one-item Output.
        self.page_items: Callable[[str], list] = lambda _nid: []
        #: ``(node_id, name) -> name`` (V4.00 step 11e): the name Page Output ``node_id`` may
        #: carry — ``name`` when no other Output has it, else ``name2``, ``name3``, … The
        #: Workspace checks every page (a variable name identifies ONE output); outside a
        #: workspace this page's other Outputs are all there is.
        self.claim_output_name: Callable[[str, str], str] = self._claim_output_name_here
        #: ``(source value) -> page kind`` of the page a Page Input source names, ``""`` when
        #: it names none — the colour of each entry of a Source menu (V4.00 step 11e).
        self.source_kind: Callable[[str], str] = lambda _v: ""
        #: the last Page Output name the document changed to keep names unique:
        #: ``(node_id, asked, given)`` — the window says so in the status bar
        self.renamed_output: Optional[Tuple[str, str, str]] = None
        #: set by the Workspace (V4.00 step 5): a digest of what ``node_id`` reads from OTHER
        #: pages through Page Inputs ("" when nothing) — part of a dock's signature — and the
        #: page's run identity, which moves when an upstream page is edited
        self.cross_page_signature: Callable[[str], str] = lambda _nid: ""
        self.workspace_revision: Callable[[], str] = lambda: ""

    # ── listeners ────────────────────────────────────────────────────────────
    def on_change(self, fn: Callable[[], None], *, first: bool = False) -> None:
        """Call ``fn`` after every change. ``first`` puts it AHEAD of every listener already
        registered — how the Workspace attaches (V4.00 step 2), so it hears an edit, and the
        runner cancels what the edit made stale, before any other listener can act on it:
        the Movie Editor re-fetches its sources from inside its own change handler, and a
        fetch submitted before the cancel would be cancelled the moment it started."""
        if first:
            self._listeners.insert(0, fn)
        else:
            self._listeners.append(fn)

    def off_change(self, fn: Callable[[], None]) -> None:
        """Forget a listener registered with :meth:`on_change` — a removed page's scene, a
        Workspace letting a document go. Unknown listeners are ignored."""
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    def _notify(self, touched: Optional[Iterable[str]] = None) -> None:
        """Bump the revision, re-propagate envelopes, and tell the listeners.

        ``touched`` names the nodes this edit changed, and is published as
        :attr:`last_touched` for the listeners to read (2026-08-06). The runner uses it to
        cancel only the in-flight pulls that edit could affect, instead of every pull there
        is — which is what lets a finished branch be adjusted while another is still running.

        ``None`` means "unknown, assume everything", and it is the DEFAULT on purpose: a
        structural edit (a rewire, a group expansion) changes which nodes feed which and a
        cone computed against the old graph no longer describes the new one. Only an edit
        that provably touches a known set should narrow it, so a call site that has not
        been considered stays conservative rather than silently keeping a doomed run alive.

        A **delete** is the one structural edit that DOES narrow (:meth:`remove_node`):
        every run the deleted node could possibly affect — its own and anything downstream
        of it — carries it in the cone recorded when that run started, and the old-graph
        cone is the right thing to test because those runs were planned against the old
        graph. Runs on other branches provably never read it, and cancelling them anyway
        is what made deleting any card silently kill every computation in flight."""
        self.revision += 1
        self.last_touched = None if touched is None else frozenset(touched)
        self.propagate()
        for fn in list(self._listeners):
            fn()

    def repropagate(self) -> None:
        """Re-run the envelope pass and tell the listeners WITHOUT bumping the revision
        (V4.00): an UPSTREAM PAGE changed what this page's Inputs carry. Nothing in this
        document changed, so its revision must not move — the run graph's identity is the
        composed revision the Workspace digests over every page it reads
        (:meth:`nodelab_v2.workspace.Workspace.revision_of`), and that is what the runner
        compares. Published with an EMPTY touched set for the same reason
        :meth:`set_meta_seed` is: no run of THIS document's own nodes is invalidated by it."""
        self.last_touched = frozenset()
        self.propagate()
        for fn in list(self._listeners):
            fn()

    # ── node ops ─────────────────────────────────────────────────────────────
    def new_id(self, prefix: str = "n") -> str:
        i = len(self.nodes) + 1
        while f"{prefix}{i}" in self.nodes:
            i += 1
        return f"{prefix}{i}"

    def add_node(self, op_key: str, *, x: float = 0.0, y: float = 0.0,
                 node_id: Optional[str] = None, params: Optional[dict] = None,
                 modes: Optional[dict] = None) -> NodeRecord:
        nid = node_id or self.new_id()
        if nid in self.nodes:
            raise ValueError(f"duplicate node id {nid!r}")
        if "/" in nid:
            # "/" separates the page from the node in a page-qualified run id (V4.00)
            raise ValueError(f"node id {nid!r} may not contain '/'")
        defaults = self.node_defaults(op_key) or {}
        if defaults:
            # the page's defaults UNDER the caller's values: a hand-placed Page Output is
            # named and a Page Input bound at once, while an explicit value is never touched
            merged = dict(defaults)
            merged.update(params or {})
            params = merged
        rec = NodeRecord(nid, op_key, params=params, modes=modes, x=x, y=y)
        self.nodes[nid] = rec
        self._settle_output_name(nid)            # an explicit name another Output carries
        # A brand-new id cannot be in any in-flight run's cone, so naming it here cancels
        # nothing — while still being honest about what changed (unlike an empty set).
        self._notify((nid,))
        return rec

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        op_key = self.nodes[node_id].op_key
        self.edges = [e for e in self.edges if e[0] != node_id and e[2] != node_id]
        del self.nodes[node_id]
        self.meta_seeds.pop(node_id, None)
        self._prune_frames(node_id)
        self._drop_orphan_group(op_key)          # a removed instance drops its unused def
        # Narrowed, not `None`: a delete cancels exactly the runs whose cone contains this
        # node (its own — which then ABORTS on the worker — and anything downstream), and
        # leaves every other branch running and queued. See :meth:`_notify` for why a
        # delete may narrow when other structural edits must not.
        self._notify((node_id,))

    def _drop_orphan_group(self, op_key: str) -> None:
        """If ``op_key`` is a group-instance op_key with no remaining instances, drop the
        group definition (kept while any instance still references it)."""
        name = group_name_of(op_key)
        if name and not any(group_name_of(r.op_key) == name for r in self.nodes.values()):
            self._groups = [g for g in self._groups if g.name != name]

    def touch(self, node_id: Optional[str] = None) -> None:
        """Signal a param/mode edit (values live in shared dicts — no copy needed).

        ``node_id`` names the card that was edited. It narrows the invalidation to the pulls
        that node can affect, so nudging a threshold on one branch no longer cancels another
        branch's running segmentation. Omitting it keeps the conservative "cancel everything"
        behaviour, which is still correct — just wasteful."""
        if node_id is not None:
            self._settle_output_name(node_id)
        self._notify(None if node_id is None else (node_id,))

    # ── Page Output names (V4.00 step 11e) ────────────────────────────────────
    def _claim_output_name_here(self, node_id: str, name: str) -> str:
        """:attr:`claim_output_name` outside a workspace: unique among THIS page's Outputs."""
        base = sanitize_output_name(name) or DEFAULT_OUTPUT_BASE
        taken = {str(r.params.get(PAGE_NAME_KEY, "") or "").strip().lower()
                 for r in self.nodes.values()
                 if r.op_key == PAGE_OUTPUT_OP and r.id != node_id} - {""}
        return next_free_name(base, taken)

    def _settle_output_name(self, node_id: str) -> None:
        """Keep Page Output ``node_id``'s variable name unique — a name another Output
        already carries becomes ``name2`` (:attr:`claim_output_name`), and one a Source
        value cannot carry is cleaned (``a:b`` → ``a_b``). A blank name stays blank: the
        readiness check asks for one. Called by :meth:`add_node` and :meth:`touch`, the two
        ways a name arrives, so every editor (inspector, card, a script) gets it."""
        rec = self.nodes.get(node_id)
        if rec is None or rec.op_key != PAGE_OUTPUT_OP:
            return
        asked = str(rec.params.get(PAGE_NAME_KEY, "") or "")
        if not asked.strip():
            return
        try:
            given = self.claim_output_name(node_id, asked)
        except Exception:                             # noqa: BLE001 — a workspace mid-change
            return
        if given and given != asked:
            rec.params[PAGE_NAME_KEY] = given
            self.renamed_output = (node_id, asked.strip(), given)

    # ── switching a node off (V4.00 step 11e) ─────────────────────────────────
    def pass_through_reason(self, node_id: str) -> str:
        """Why ``node_id`` may NOT be switched off (:func:`pass_through_reason`), ``""``
        when it may. An unknown id answers why not, so a caller never mutes nothing."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return "no such node"
        return pass_through_reason(NODES.get(rec.op_key), rec.params, rec.modes)

    def can_pass_through(self, node_id: str) -> bool:
        return not self.pass_through_reason(node_id)

    def title_of(self, node_id: str) -> str:
        """What a node is called where the user reads it: its own title when it has one,
        else its node type's label (the op key for an unknown type)."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return node_id
        own = str(rec.params.get(TITLE_KEY, "") or "").strip()
        if own:
            return own
        spec = NODES.get(rec.op_key)
        return spec.label if spec is not None else rec.op_key

    def clear(self) -> None:
        """Empty the document (File → New)."""
        self.nodes.clear()
        self.edges = []
        self.meta_seeds.clear()
        self._zones = []
        self._groups = []
        self._back_edges = []
        self.frames = {}
        self.path = None
        self._notify()

    # ── frames (GUI-only canvas grouping) ──────────────────────────────────────
    def new_frame_id(self) -> str:
        i = len(self.frames) + 1
        while f"f{i}" in self.frames:
            i += 1
        return f"f{i}"

    def add_frame(self, title: str = "Frame", members=(), color=None,
                  frame_id: Optional[str] = None) -> FrameRecord:
        """Create a labelled frame around ``members`` (only the ids that exist are
        kept). A frame must enclose ≥1 node — an empty selection raises."""
        mem = [n for n in members if n in self.nodes]
        if not mem:
            raise ValueError("select one or more nodes to frame")
        fid = frame_id or self.new_frame_id()
        if fid in self.frames:
            raise ValueError(f"duplicate frame id {fid!r}")
        self.frames[fid] = FrameRecord(fid, title, mem, color)
        self._notify(())          # a frame is canvas decoration: it reaches no run graph
        return self.frames[fid]

    def remove_frame(self, frame_id: str) -> None:
        if frame_id in self.frames:
            del self.frames[frame_id]
            self._notify(())      # GUI-only, like add_frame

    def rename_frame(self, frame_id: str, title: str) -> None:
        fr = self.frames.get(frame_id)
        if fr is not None:
            fr.title = str(title)
            self._notify(())      # GUI-only, like add_frame

    def downstream_of(self, node_ids: Iterable[str]) -> frozenset:
        """``node_ids`` plus every node they transitively FEED.

        The reach of an edit, and the mirror of the runner's ``planned_nodes`` (which walks
        upstream — what a pull *reads*). Changing a threshold cannot alter what fed it, but
        it invalidates every result computed from it, so this is the set whose run badges
        have stopped being true. Follows the forward edge layer only; a zone back-edge is
        preserved verbatim and never carries a fresh result."""
        seen = set(node_ids)
        stack = list(seen)
        while stack:
            nid = stack.pop()
            for src, _ss, dst, _ds in self.edges:
                if src == nid and dst not in seen:
                    seen.add(dst)
                    stack.append(dst)
        return frozenset(seen)

    def real_source(self, node_id: str, socket: str = "data") -> Optional[str]:
        """The node whose output actually reaches ``node_id``'s input ``socket``.

        The wire into the socket, followed back through every **Reroute** (a routing dot,
        not a node anyone means) and every **muted** node (which passes its first connected
        Dataset input through, exactly as :meth:`_bypass_muted` rewires it for a run). So the
        answer is the node a pull of ``node_id`` really reads, which is what "the node feeding
        this one" should mean to anything keyed by node: the Viewer's LUTs, a finished
        payload. ``None`` when nothing real is wired (unwired, or a muted source)."""
        def into(nid: str, names) -> Optional[EdgeTuple]:
            return next((e for e in self.edges
                         if e[2] == nid and (e[3] in names or (not e[3] and "data" in names))),
                        None)

        edge = into(node_id, {socket})
        seen = set()
        while edge is not None:
            src = edge[0]
            rec = self.nodes.get(src)
            if rec is None or src in seen:
                return None
            seen.add(src)
            if rec.op_key != "rr.reroute" and not rec.muted:
                return src
            edge = into(src, {s.name for s in self.input_specs(src)
                              if s.type is SocketType.DATASET})
        return None

    def _prune_frames(self, node_id: str) -> None:
        """Drop a removed node from every frame; a frame left with no members is
        removed (frames never persist empty). Does NOT notify (the caller does)."""
        for fid in list(self.frames):
            fr = self.frames[fid]
            if node_id in fr.members:
                fr.members = [n for n in fr.members if n != node_id]
                if not fr.members:
                    del self.frames[fid]

    def set_pos(self, node_id: str, x: float, y: float) -> None:
        rec = self.nodes.get(node_id)
        if rec is not None and (rec.x, rec.y) != (float(x), float(y)):
            rec.x, rec.y = float(x), float(y)   # position is not a model edit: no notify
            for fn in list(self._move_listeners):
                fn(node_id)                     # …but the canvases showing it follow

    def on_moved(self, fn: Callable[[str], None]) -> None:
        """Call ``fn(node_id)`` whenever a card's position changes (GUI only)."""
        self._move_listeners.append(fn)

    def off_moved(self, fn: Callable[[str], None]) -> None:
        try:
            self._move_listeners.remove(fn)
        except ValueError:
            pass

    def set_muted(self, node_id: str, muted: bool) -> None:
        """Mute/unmute a node (G3 pass-through). A REAL graph change — the run graph bypasses
        a muted node — so it is scoped to that node: runs whose cone contains it are
        cancelled, everything else keeps going.

        Switching OFF is refused (``ValueError`` naming why) for a node that changes the kind
        of data (:func:`pass_through_reason`, V4.00 step 11e); switching back ON never is — a
        file saved before the rule may hold such a node muted, and it must be recoverable."""
        rec = self.nodes.get(node_id)
        if rec is not None and muted and not rec.muted:
            why = self.pass_through_reason(node_id)
            if why:
                raise ValueError(f"{self.title_of(node_id)} cannot be switched off: {why}")
        if rec is not None and rec.muted != muted:
            rec.muted = muted
            self._notify((node_id,))

    def set_collapsed(self, node_id: str, collapsed: bool) -> None:
        """Fold/unfold a card. GUI-ONLY, so it notifies with an EMPTY touched set: folding a
        card must not disturb a pull (2026-08-06). It used to notify unscoped, which since
        the cooperative cancel lands as "abort every run in flight" — collapsing a card while
        a segmentation ran killed it outright, with no error to show for it. Position
        (:meth:`set_pos`) does not notify at all for the same reason."""
        rec = self.nodes.get(node_id)
        if rec is not None and rec.collapsed != collapsed:
            rec.collapsed = collapsed
            self._notify(())

    # ── instance-aware socket resolution (per-channel outputs) ─────────────────
    def output_specs(self, node_id: str) -> list:
        """The live OUTPUT socket specs of one node INSTANCE: the type's active outputs
        plus, for a ``io.load``/``channel.split`` node with ≥2 channels, one synthetic
        ``ch{K}`` :class:`OutDataset` per channel (labelled by the channel name). These
        synthetic sockets are materialized into ``channel.select`` taps at graph-build —
        here they only need to exist so the canvas can lay them out and validate wires."""
        rec = self.nodes.get(node_id)
        if rec is not None and group_name_of(rec.op_key):
            return [OutDataset()]                 # a group instance: one Dataset output
        spec = rec.spec() if rec else None
        if spec is None:
            return []
        base = list(spec.active_outputs(rec.state()))
        if rec.op_key in CHANNEL_TAP_OPS:
            descs = self.channel_descriptors(node_id)
            if len(descs) >= 2:
                for i, ch in enumerate(descs):
                    base.append(OutDataset(f"ch{i}",
                                           label=f"{i} · {ch.get('name') or f'Ch{i}'}"))
        if rec.op_key in GROUP_TAP_OPS:
            groups = self.group_descriptors(node_id)
            # TWO or more, the same floor the channel taps use and for the same reason: a
            # single group is the whole file, so a lone `grp0` socket would sit beside
            # `image` offering the identical Dataset under a second name.
            if len(groups) >= 2:
                for i, g in enumerate(groups):
                    base.append(OutDataset(f"grp{i}", label=_group_socket_label(i, g)))
        if rec.op_key in BATCH_TAP_OPS:
            # Same TWO-or-more floor, same reason: `util.batch` itself refuses a one-file
            # batch, so a lone `bat0` could only ever duplicate `out`.
            members = self.batch_member_names(node_id)
            if len(members) >= 2:
                for i, name in enumerate(members):
                    base.append(OutDataset(f"bat{i}", label=f"{i} · {name}"))
        if rec.op_key in POSITION_TAP_OPS:
            # Same floor again: one position is the whole Dataset, so a lone `pos0` would
            # duplicate `out`.
            positions = self.position_descriptors(node_id)
            if len(positions) >= 2:
                for i, pd in enumerate(positions):
                    base.append(OutDataset(f"pos{i}", label=pd["label"]))
        if rec.op_key == PAGE_INPUT_OP:
            # the items of a several-item Output, each on a socket of its own (step 11f)
            items = self._page_items(node_id)
            if len(items) >= 2:
                for name in items:
                    base.append(OutDataset(item_socket(name), label=f"item · {name}"))
        if any(s.type is SocketType.DATASET for s in base):
            # each PART of what `out` carries, on its own (step 11f): `image`, `mask`, …
            for name in self.data_parts(node_id):
                base.append(OutDataset(part_socket(name), label=f"{name} only"))
        return base

    # ── parts and items (V4.00 step 11f) ──────────────────────────────────────
    def data_parts(self, node_id: str) -> List[str]:
        """The PARTS of a node's output: ``image`` and each named layer its envelope carries
        (a mask; a label raster and its table, one part under one name; a point table), in
        the order they arrived — ``[]`` when there is only one, so a card offers part
        sockets exactly when its ``out`` carries more than one kind of data. A reroute, a
        group instance or a node with no Dataset output has none."""
        rec = self.nodes.get(node_id)
        if rec is None or rec.op_key.startswith("rr.") or group_name_of(rec.op_key):
            return []
        try:
            env = self.env(node_id)
        except Exception:                            # noqa: BLE001 — an un-propagated node
            return []
        names: List[str] = []
        for _d, n in getattr(env, "layer_names", ()) or ():
            n = str(n)
            if n and n != PART_IMAGE and n not in names:
                names.append(n)
        return [PART_IMAGE] + names if names else []

    def output_items(self, node_id: str) -> List[Tuple[str, str]]:
        """``[(socket, name), ...]`` — the ITEMS of Page Output ``node_id``: its wired item
        sockets (``data``, ``data_2``, …) in order, each named by the ``items`` param's entry
        for that socket, else after its wire — the part it carries (``mask``), the item or
        variable a Page Input hands on, or the node it comes from (``gaussian_blur``) —
        unique within the Output. One item is the plain single-Dataset Output."""
        rec = self.nodes.get(node_id)
        if rec is None or rec.op_key != PAGE_OUTPUT_OP:
            return []
        wired: Dict[str, Tuple[str, str]] = {}
        for (s, ss, d, ds) in self.edges:
            if d == node_id and ds in PAGE_ITEM_SOCKETS and ds not in wired:
                wired[ds] = (s, ss)
        given = [x.strip() for x in str(rec.params.get(PAGE_ITEMS_KEY, "") or "").split(",")]
        out: List[Tuple[str, str]] = []
        taken: set = set()
        for k, sock in enumerate(PAGE_ITEM_SOCKETS):
            if sock not in wired:
                continue
            name = sanitize_output_name(given[k] if k < len(given) else "") or \
                self._item_default(*wired[sock])
            name = next_free_name(name, taken)
            taken.add(name.lower())
            out.append((sock, name))
        return out

    def _item_default(self, src: str, socket: str) -> str:
        """An item's name from its wire: a part's or an item's name; a tap socket's CHANNEL
        (``Cy5``), POSITION (``B03``), group key or batch member (steps 11g–11h); a Page
        Input's variable; else the source node's title — with the position and the channel
        appended when the wire is one of the file's and the title does not say it
        (``gaussian_blur_b03_cy5``), so a variable says where it came from without anyone
        typing it. A Page Output with a placeholder name takes the same answer
        (:meth:`_name_output_from_wire`)."""
        m = PART_SOCKET_RE.match(socket or "") or ITEM_SOCKET_RE.match(socket or "")
        if m:
            return sanitize_output_name(m.group(1)) or "item"
        rec = self.nodes.get(src)
        tap = self._tap_name(src, socket)
        if tap:
            return sanitize_output_name(tap) or "item"
        if rec is not None and rec.op_key == PAGE_INPUT_OP:
            name = str(rec.params.get(PAGE_SOURCE_KEY, "") or "").split(":", 1)[-1].strip()
            if name:
                return sanitize_output_name(name) or "item"

        def _slug(s: str) -> str:
            return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")

        base = _slug(self.title_of(src))[:32] or "item"
        spec = rec.spec() if rec is not None else None
        main = next((s.name for s in (spec.outputs if spec else ())
                     if s.type is SocketType.DATASET), "out")
        name = base if socket in ("", main) else f"{base}_{socket}"
        # the stream's identity where the title does not say it: ONE position, ONE channel
        pos = self.position_subset(src, socket, "out")
        chans = self.channel_subset(src, socket, "out")
        for extra in (pos[0] if len(pos) == 1 else "",
                      str(chans[0].get("name") or "") if len(chans) == 1 else ""):
            s = _slug(extra)
            if s and s not in name:
                name = f"{name}_{s}"
        return name

    def _tap_name(self, node_id: str, socket: str) -> str:
        """What a synthetic tap socket carries, by name: channel K's (``chK``), position K's
        (``posK``), group K's key (``grpK``), batch member K's (``batK``); ``""`` for any
        other socket or an index the node does not have (V4.00 step 11h)."""
        for regex, lister, field in ((CH_SOCKET_RE, self.channel_descriptors, "name"),
                                     (POS_SOCKET_RE, self.position_descriptors, "name"),
                                     (GRP_SOCKET_RE, self.group_descriptors, "key"),
                                     (BAT_SOCKET_RE, self.batch_member_names, None)):
            m = regex.match(socket or "")
            if m is None:
                continue
            k = int(m.group(1))
            try:
                items = lister(node_id)
            except Exception:                        # noqa: BLE001 — mid-edit upstream
                return ""
            if not (0 <= k < len(items)):
                return ""
            got = items[k] if field is None else (items[k] or {}).get(field)
            return str(got or "")
        return ""

    def _page_items(self, node_id: str) -> List[str]:
        try:
            return [str(n) for n in (self.page_items(node_id) or [])]
        except Exception:                            # noqa: BLE001 — a workspace mid-change
            return []

    def position_descriptors(self, node_id: str) -> list:
        """The multipoint positions on the wire reaching ``node_id``, from its edit-time
        envelope: ``[{"index", "name", "group", "label"}, …]`` — ``[]`` when the envelope
        does not know ``m`` (an unresolved source) or has one position.

        ``name`` is the acquisition's point label (``position_name``) when the file carries
        one of the right length, else ``m{i}``; ``group`` the position's specimen key
        (``position_group``) when known. ``label`` is what the socket shows:
        ``"2 · B03"`` or ``"2 · B03 · G1"``. Read live from the envelope rather than
        captured, like the batch member names: the count is an axis the envelope predicts
        exactly, so a rewire moves the sockets with it."""
        try:
            env = self.env(node_id)
        except Exception:                      # noqa: BLE001 — an un-propagated node
            return []
        if "m" in getattr(env, "unknown_axes", frozenset()):
            return []
        m = int(env.axes.m)
        if m <= 0:
            return []
        md = env.metadata or {}
        names = md.get("position_name")
        names = [str(v) for v in names] if isinstance(names, (list, tuple)) \
            and len(names) == m else [f"m{i}" for i in range(m)]
        groups = md.get("position_group")
        groups = [str(v) if v is not None else "" for v in groups] \
            if isinstance(groups, (list, tuple)) and len(groups) == m else [""] * m
        out = []
        for i in range(m):
            label = f"{i} · {names[i]}" + (f" · {groups[i]}" if groups[i] else "")
            out.append({"index": i, "name": names[i], "group": groups[i], "label": label})
        return out

    def input_specs(self, node_id: str) -> list:
        """The live INPUT socket specs of one node instance, plus — once the document
        contains an Iterate node — one synthetic ``__mode__:<name>`` port per active Mode
        (:meth:`mode_port_specs`)."""
        rec = self.nodes.get(node_id)
        if rec is not None and group_name_of(rec.op_key):
            return [InDataset()]                  # a group instance: one Dataset input
        spec = rec.spec() if rec else None
        if spec is None:
            return []
        specs = self._grow_filter(node_id, list(spec.active_inputs(rec.state())))
        if rec.op_key == PAGE_OUTPUT_OP:
            # a several-item variable (step 11f): each wired slot reads as its item's name
            items = dict(self.output_items(node_id))
            if len(items) >= 2:
                from dataclasses import replace as _replace
                specs = [_replace(s, label=items[s.name]) if s.name in items else s
                         for s in specs]
        return specs + self.mode_port_specs(node_id)

    def _grow_filter(self, node_id: str, specs: list) -> list:
        """Hide the not-yet-needed members of each ``grow_group``
        (:attr:`~nodegraph.registry.SocketSpec.grow_group`, 2026-10-02).

        Within a group, in declaration order: the first member always shows; every later
        member shows if it is wired, if the member before it is wired, or if ANY later
        member is wired (a wire must never be hidden, nor the slots leading up to it). So a
        node always offers exactly one empty slot after its last wired stream — Blender's
        virtual socket — and the engine never knows: the hidden specs still exist, they are
        simply unwired."""
        if not any(getattr(s, "grow_group", "") for s in specs):
            return specs
        wired = {e[3] for e in self.edges if e[2] == node_id}
        groups: Dict[str, List] = {}
        for s in specs:
            g = getattr(s, "grow_group", "")
            if g:
                groups.setdefault(g, []).append(s)
        hide: set = set()
        for members in groups.values():
            last_wired = max((i for i, s in enumerate(members) if s.name in wired),
                             default=-1)
            keep_upto = min(len(members) - 1, last_wired + 1)   # ...plus one empty slot
            for i, s in enumerate(members):
                if i > keep_upto:
                    hide.add(s.name)
        return [s for s in specs if s.name not in hide]

    def mode_port_specs(self, node_id: str) -> list:
        """Synthetic input ports for this node's Modes, so an Iterate variable can be wired
        onto a dropdown.

        A Mode is not a socket and has no port to land on, yet sweeping one (four segmenters,
        five threshold methods) is the comparison most worth running. So the card grows a
        port per Mode — but ONLY while the document holds at least one ``flow.iterate``
        node, because otherwise every card in every graph would sprout ports nothing could
        ever connect to. The name is reserved (``__mode__:method``) and never reaches the
        engine: :func:`nodegraph.iterate.unroll` consumes every driver wire and bakes the
        value into the clone's ``modes`` dict.

        STRING because a mode value is a name, which also makes the type system carry the
        rule for free — a numeric variable output cannot connect (there is no FLOAT→STRING
        conversion), so sweeping a Mode requires setting that variable's Type to 'text'.
        The 2D/3D lever is excluded: it is refused by the rewrite, so offering the port
        would only let the user build something that cannot run."""
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None or rec.op_key == ITERATE_OP or not self.has_iterate:
            return []
        return [InString(MODE_TARGET_PREFIX + m.name, m.label or m.name, field=False,
                         default="",
                         description=f"Sweep the {m.label or m.name} dropdown from an "
                                     f"Iterate node ({', '.join(m.choices)}).")
                for m in spec.active_modes(rec.state()) if not m.is_dim_lever]

    @property
    def has_iterate(self) -> bool:
        return any(r.op_key == ITERATE_OP for r in self.nodes.values())

    def channel_descriptors(self, node_id: str) -> list:
        """The per-channel descriptors ``[{name, emission_nm, color}, …]`` for a source /
        split node: the captured ``__channels__`` list (Load, richest — real names +
        native colors) if present, else the same list INHERITED from upstream when the
        channel count still matches, else derived from the node's own propagated envelope
        (Split passes its input env through unchanged, so its channel count/emission are
        already correct).

        The inheritance step is what puts the file's real channel names on a Split's
        per-channel sockets and — narrowing through a ``chK`` wire — on every card past a
        tap. Since V4.00 step 11g ``channel_names`` rides the edit-time source envelope as
        well (:func:`nodelab_v2.ingest.channel_display_seed`), narrowed by every
        ``channel.select`` in lockstep with the axis, so the envelope route answers with the
        file's names too. Before that it carried only the emission and could only ever fall
        back to ``Ch0``/``Ch1`` — which is how a two-branch graph ended up with two sockets
        that both said "Ch0" and no way to tell which physical channel each branch carried.
        The captured list still wins when the counts agree: it carries the native colours."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return []
        chans = rec.params.get(CHANNELS_KEY)
        if isinstance(chans, list) and chans:
            return chans
        env_descs = self._env_channel_descriptors(self.env(node_id))
        if rec.op_key == "page.input":
            # the page boundary: the upstream Output's own descriptors (the Load card's names
            # and colours), while they still describe what crosses — the envelope's count
            up = self._page_channels(node_id)
            return up if up and (not env_descs or len(up) == len(env_descs)) else env_descs
        spec = rec.spec()
        if spec is not None and getattr(spec, "fresh_output", False):
            return env_descs            # a NEW Dataset (a plot's picture): its own channels
        inherited = self._inherited_channel_descriptors(node_id)
        if inherited and len(inherited) == len(env_descs):
            return inherited
        return env_descs

    def group_descriptors(self, node_id: str) -> list:
        """The card's position groups — ``[{"key", "size", "shape"}, …]``, or ``[]``.

        The captured :data:`GROUPS_KEY` first, exactly as
        :meth:`channel_descriptors` prefers ``__channels__``: it is written when the file is
        opened, from the sidecar or the detector, and it is what SAVES with the graph — so a
        card reopened on a machine that cannot see the file still shows the specimen names
        its wires refer to, instead of collapsing its outputs and dropping every edge.

        The envelope is the fallback, and it can answer because the grouping rides there as
        an ordinary per-M list (``position_group``, stamped by
        :func:`nodelab_v2.runner._with_position_groups`). That covers a graph saved before
        this existed: it regains its groups on the first resolve rather than needing a
        migration.

        ``[]`` unless the lever is explicitly ``auto`` — which is what makes the lever mean
        anything, since the sockets are computed from this list, so a lever that is off has
        to empty it rather than be tested again at every call site. Off is the DEFAULT
        (:data:`~nodelab_v2.ops.GROUPING_DEFAULT`), and the test is written as "not auto"
        rather than "is off" so that an unset mode and a saved graph from before this lever
        existed both land on the same answer as a freshly dropped card.
        """
        rec = self.nodes.get(node_id)
        if rec is None:
            return []
        if str(rec.modes.get(GROUPING_MODE, GROUPING_DEFAULT)) != GROUPING_AUTO:
            return []
        got = rec.params.get(GROUPS_KEY)
        if isinstance(got, list) and got:
            return got
        return self._env_group_descriptors(self.env(node_id))

    @staticmethod
    def _env_group_descriptors(env: MetaEnvelope) -> list:
        """Group descriptors read off an envelope's per-M ``position_group`` list.

        First-appearance order, which is acquisition order — the same order
        :func:`nodegraph.placement.position_groups` returns and the same order the ordinal
        spelling (``"2"``) counts in. Sorting the keys here would silently renumber them.
        """
        labels = env.metadata.get("position_group")
        if not isinstance(labels, (list, tuple)) or not labels:
            return []
        out: list = []
        seen: dict = {}
        for v in labels:
            key = str(v)
            if not key:
                return []                    # a gap makes the whole list untrustworthy
            if key not in seen:
                seen[key] = len(out)
                out.append({"key": key, "size": 0, "shape": ""})
            out[seen[key]]["size"] += 1
        return out

    def _inherited_channel_descriptors(self, node_id: str, _depth: int = 0) -> list:
        """The captured ``__channels__`` descriptors of the nearest upstream node that has
        them, following the first Dataset in-edge. ``[]`` if none does.

        Depth-capped rather than cycle-tracked: the walk only ever runs a handful of hops
        (Load → Split is the case it exists for) and a bounded walk cannot hang on the
        zone back-edges the document legitimately holds.

        A ``chK`` wire NARROWS the walk to that one channel (V4.00 step 11g) instead of
        ending it: the card past the tap then inherits the file's name and native colour for
        channel K. Until then the walk stopped at the tap and the card fell back to its
        envelope, which — before the names rode the seed — could only say ``Ch0``."""
        if _depth > 8:
            return []
        for src, ssock, dst, _dsock in self.edges:
            if dst != node_id:
                continue
            rec = self.nodes.get(src)
            m = CH_SOCKET_RE.match(ssock or "")
            if m is not None:
                k = int(m.group(1))
                chans = rec.params.get(CHANNELS_KEY) if rec is not None else None
                if not (isinstance(chans, list) and chans):
                    chans = (self._page_channels(src)
                             if rec is not None and rec.op_key == PAGE_INPUT_OP
                             else self._inherited_channel_descriptors(src, _depth + 1))
                return ([chans[k]] if isinstance(chans, list) and 0 <= k < len(chans)
                        else [])
            chans = rec.params.get(CHANNELS_KEY) if rec is not None else None
            if isinstance(chans, list) and chans:
                return chans
            if rec is not None and rec.op_key == "page.input":
                im = ITEM_SOCKET_RE.match(ssock or "")
                if im is not None:
                    # one ITEM of a several-item Output: ITS wire's channels. The Output's own
                    # list is its FIRST item's, which is how every mask of a three-channel
                    # split came through calling itself by the first one's channel.
                    return self._page_scope(src, im.group(1))[0]
                return self._page_channels(src)     # the walk crosses the page boundary
            spec = rec.spec() if rec is not None else None
            if spec is not None and getattr(spec, "fresh_output", False):
                return []               # the walk stops at a NEW Dataset (a plot's picture)
            return self._inherited_channel_descriptors(src, _depth + 1)
        return []

    def _page_channels(self, node_id: str) -> list:
        """:attr:`page_channels`, never raising — it is asked while cards are laid out."""
        try:
            out = self.page_channels(node_id)
        except Exception:                      # noqa: BLE001 — mid-edit upstream
            return []
        return list(out) if isinstance(out, (list, tuple)) else []

    @staticmethod
    def _env_channel_descriptors(env: MetaEnvelope) -> list:
        c = env.axes.c
        if c <= 0 or "c" in env.unknown_axes:
            return []                               # channel count not yet known
        emis = env.metadata.get("channel_emission_nm")
        names = env.metadata.get("channel_names")
        cols = env.metadata.get("channel_colors")
        out = []
        for i in range(c):
            col = cols[i] if isinstance(cols, (list, tuple)) and i < len(cols) else None
            out.append({
                "name": (names[i] if isinstance(names, (list, tuple)) and i < len(names)
                         else f"Ch{i}"),
                "emission_nm": (emis[i] if isinstance(emis, (list, tuple))
                                and i < len(emis) else None),
                # an [r, g, b] triple (a picture's R/G/B); a packed int has an ambiguous
                # byte order and is not used
                "color": (list(col) if isinstance(col, (list, tuple)) and len(col) == 3
                          else None),
            })
        return out

    def upstream_channel_descriptors(self, node_id: str) -> list:
        """The channel descriptors of the Dataset flowing INTO ``node_id``.

        Distinct from :meth:`channel_descriptors`, which reports a node's OWN channels: for
        ``channel.select`` those are the ones it kept, and the picker needs the ones it can
        still choose from. Wired from a synthetic ``chK`` tap the incoming stream is that one
        channel, so the list narrows to it — otherwise the tick list would offer channels
        that a single-channel tap already dropped; from a Page Input's ``item:<name>`` it is
        that item's (:meth:`socket_channels`, V4.00 step 11g)."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return []
        for src, ssock, dst, _dsock in self.edges:
            if dst != node_id:
                continue
            return self.socket_channels(src, ssock, "out")
        return []

    # ── which channel a socket carries (V4.00 step 11g) ──────────────────────
    def socket_channels(self, node_id: str, socket: str, io: str = "out") -> list:
        """The channel descriptors riding ONE Dataset socket — all of them, whether or not
        they are a subset (``[]`` when unknown or unwired).

        An output: a ``chK`` socket is channel K of the node's list; a Page Input's
        ``item:<name>`` socket is what that item of the upstream Output carries
        (:attr:`page_channel_scope`); any other output — ``out``, a part, a group or
        position tap — is the node's own list (:meth:`channel_descriptors`), which the
        envelope already narrowed if a ``channel.select`` sits upstream. An input is whatever
        its wire's source socket carries. A plot's picture answers R/G/B here (the picker
        needs them); that it is no channel of the FILE's is :meth:`channel_subset`'s call."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return []
        if io == "in":
            edge = self.edge_into(node_id, socket)
            return self.socket_channels(edge[0], edge[1], "out") if edge is not None else []
        m = CH_SOCKET_RE.match(socket or "")
        if m is not None:
            descs = self.channel_descriptors(node_id)
            k = int(m.group(1))
            return [descs[k]] if 0 <= k < len(descs) else []
        m = ITEM_SOCKET_RE.match(socket or "")
        if m is not None:
            return list(self._page_scope(node_id, m.group(1))[0])
        return list(self.channel_descriptors(node_id))

    def channel_subset(self, node_id: str, socket: str, io: str = "out") -> list:
        """:meth:`socket_channels` when they are a STRICT subset of the source file's
        channels — the stream is *one of* (or *some of*) the file's — else ``[]``.

        The same rule the wire tint has always used (a full bundle is not tinted, because
        "everything" is not a channel): the socket's text, its dot and its wire all read
        from here, so they cannot disagree about which channel a stream is."""
        if io == "in":
            edge = self.edge_into(node_id, socket)
            return self.channel_subset(edge[0], edge[1], "out") if edge is not None else []
        rec = self.nodes.get(node_id)
        try:
            spec = rec.spec() if rec is not None else None
        except Exception:                            # noqa: BLE001 — an unknown op
            spec = None
        if spec is not None and getattr(spec, "fresh_output", False):
            return []                 # a NEW Dataset (a plot's picture): no channel of the file's
        descs = self.socket_channels(node_id, socket, "out")
        if not descs:
            return []
        try:
            total = self.source_channel_total(node_id)
        except Exception:                            # noqa: BLE001 — mid-edit
            return []
        return list(descs) if 1 <= len(descs) < total else []

    def channel_tag(self, node_id: str, socket: str, io: str = "out") -> str:
        """The channel(s) a socket carries, as the card prints them: ``Cy5``; ``DAPI · GFP``;
        ``DAPI +2`` for three or more. ``""`` when the stream is not a strict subset."""
        return _join_names(str(d.get("name") or "")
                           for d in self.channel_subset(node_id, socket, io))

    def socket_text(self, node_id: str, spec, io: str = "out") -> str:
        """What a card prints beside one socket: a Dataset stream's IDENTITY
        (:meth:`stream_identity` — the name it was given, the position and the channel it is
        of the file's) — on a generic socket in place of its name (``Cy5`` for ``data`` /
        ``out``), on a named one after it (``raw · Cy5``, :data:`GENERIC_DATASET_SOCKETS`);
        anything else, the socket's own text (an input its name, an output its label or
        name). A synthetic output keeps its label (:data:`_SYNTHETIC_SOCKET_RE`):
        ``0 · DAPI`` and ``mask only`` already say it."""
        base = spec.name if io == "in" else (spec.label or spec.name)
        if spec.type is not SocketType.DATASET or (
                io == "out" and _SYNTHETIC_SOCKET_RE.match(spec.name or "")):
            return base
        tag = self.stream_identity(node_id, spec.name, io)
        if not tag:
            return base
        return tag if spec.name in GENERIC_DATASET_SOCKETS else f"{base} · {tag}"

    def _page_scope(self, node_id: str, item: str = "") -> Tuple[list, int]:
        """:attr:`page_channel_scope`, never raising — it is asked while cards are laid out."""
        try:
            descs, total = self.page_channel_scope(node_id, item)
        except Exception:                      # noqa: BLE001 — mid-edit upstream
            return [], 0
        return (list(descs) if isinstance(descs, (list, tuple)) else []), int(total or 0)

    # ── which POSITION a socket carries, and whose NAME (V4.00 step 11h) ──────
    @staticmethod
    def _env_position_names(env: MetaEnvelope) -> List[str]:
        """The names of the positions a stream carries, off its envelope: the file's point
        labels (``position_name``, narrowed with M by every position tap) when the list has
        the right length, else ``m{i}``; ``[]`` while M is unknown."""
        if "m" in getattr(env, "unknown_axes", frozenset()):
            return []
        m = int(getattr(env.axes, "m", 0) or 0)
        if m <= 0:
            return []
        names = env.metadata.get("position_name")
        if isinstance(names, (list, tuple)) and len(names) == m:
            return [str(v) for v in names]
        return [f"m{i}" for i in range(m)]

    def _env_m(self, node_id: str) -> int:
        try:
            env = self.env(node_id)
        except Exception:                            # noqa: BLE001 — an un-propagated node
            return 0
        return 0 if "m" in env.unknown_axes else int(env.axes.m or 0)

    def source_position_total(self, node_id: str) -> int:
        """The position count of the source file(s) feeding ``node_id`` — the M that a
        stream carrying one position is one OF (:meth:`position_subset`). The twin of
        :meth:`source_channel_total`: the roots' seeded ``m`` (a bundle's is the sum); a root
        ``page.input`` asks the Workspace for the file's total on the page it reads from
        (:attr:`page_position_scope`)."""
        seen: set = set()
        stack = [node_id]
        totals: List[int] = []
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            preds = [e for e in self.edges if e[2] == nid]
            if preds:
                stack.extend(e[0] for e in preds)
                continue
            rec = self.nodes.get(nid)
            up = (self._page_position_scope(nid)[1]
                  if rec is not None and rec.op_key == PAGE_INPUT_OP else 0)
            totals.append(up if up > 0 else self._env_m(nid))
        return max(totals) if totals else self._env_m(node_id)

    def socket_positions(self, node_id: str, socket: str, io: str = "out") -> List[str]:
        """The position names riding ONE Dataset socket — all of them (``[]`` when unknown
        or unwired). A ``posK`` socket is position K of the node's list; a Page Input's
        ``item:<name>`` what that item carries (:attr:`page_position_scope`); any other
        output the node's own envelope; an input its wire's source socket."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return []
        if io == "in":
            edge = self.edge_into(node_id, socket)
            return self.socket_positions(edge[0], edge[1], "out") if edge is not None else []
        m = POS_SOCKET_RE.match(socket or "")
        if m is not None:
            descs = self.position_descriptors(node_id)
            k = int(m.group(1))
            return [str(descs[k]["name"])] if 0 <= k < len(descs) else []
        m = ITEM_SOCKET_RE.match(socket or "")
        if m is not None:
            return list(self._page_position_scope(node_id, m.group(1))[0])
        try:
            env = self.env(node_id)
        except Exception:                            # noqa: BLE001 — an un-propagated node
            return []
        return self._env_position_names(env)

    def position_subset(self, node_id: str, socket: str, io: str = "out") -> List[str]:
        """:meth:`socket_positions` when they are a STRICT subset of the source file's
        positions — the stream is *one of* (or *some of*) its fields — else ``[]``. The
        rule :meth:`channel_subset` uses, for M; a new Dataset (a plot's picture) is none."""
        if io == "in":
            edge = self.edge_into(node_id, socket)
            return self.position_subset(edge[0], edge[1], "out") if edge is not None else []
        rec = self.nodes.get(node_id)
        try:
            spec = rec.spec() if rec is not None else None
        except Exception:                            # noqa: BLE001 — an unknown op
            spec = None
        if spec is not None and getattr(spec, "fresh_output", False):
            return []
        names = self.socket_positions(node_id, socket, "out")
        if not names:
            return []
        try:
            total = self.source_position_total(node_id)
        except Exception:                            # noqa: BLE001 — mid-edit
            return []
        return list(names) if 1 <= len(names) < total else []

    def position_tag(self, node_id: str, socket: str, io: str = "out") -> str:
        """``B03``; ``B03 · C07``; ``A01 +2``; ``""`` when the stream is not a strict subset
        of the file's positions."""
        return _join_names(self.position_subset(node_id, socket, io))

    def stream_name(self, node_id: str, socket: str, io: str = "out", _depth: int = 0) -> str:
        """The name the user gave a stream at the Page Output it last came through — the
        variable (``control``) or, for one item of a several-item Output, the item
        (``mask_cy5``) — read off the Page Input it enters by and followed along the PRIMARY
        Dataset wire (the first Dataset input, the one the envelope follows) through any
        node after it. ``""`` on the page the data entered on (nothing has named it yet),
        while the Page Input is unbound, and on a Page Output (it has no stream out)."""
        if _depth > 256:
            return ""
        rec = self.nodes.get(node_id)
        if rec is None:
            return ""
        if io == "in":
            edge = self.edge_into(node_id, socket)
            return (self.stream_name(edge[0], edge[1], "out", _depth + 1)
                    if edge is not None else "")
        if rec.op_key == PAGE_INPUT_OP:
            m = ITEM_SOCKET_RE.match(socket or "")
            if m is not None:
                return m.group(1)
            value = str(rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
            return value.split(":", 1)[-1].strip() if value else ""
        if rec.op_key == PAGE_OUTPUT_OP:
            return ""
        primary = next((s.name for s in self.input_specs(node_id)
                        if s.type is SocketType.DATASET), None)
        if primary is None:
            return ""
        edge = self.edge_into(node_id, primary)
        return (self.stream_name(edge[0], edge[1], "out", _depth + 1)
                if edge is not None else "")

    def stream_identity(self, node_id: str, socket: str, io: str = "out") -> str:
        """What a stream IS, as a card prints it: the name it was given (:meth:`stream_name`),
        then the position(s) and the channel(s) it is of the file's — each only where the
        name does not already say it (``control · B03 · Cy5``, but ``mask_cy5 · B03`` and
        just ``mask_cy5`` on a one-field file). ``""`` when there is nothing to say: the full
        bundle on the page it entered on."""
        name = self.stream_name(node_id, socket, io)
        low = name.lower()

        def unsaid(xs) -> List[str]:
            return [x for x in xs if x and (not low or x.lower() not in low)]

        parts = [name] if name else []
        for group in (unsaid(self.position_subset(node_id, socket, io)),
                      unsaid(str(d.get("name") or "")
                             for d in self.channel_subset(node_id, socket, io))):
            if group:
                parts.append(_join_names(group))
        return " · ".join(parts)

    def _page_position_scope(self, node_id: str, item: str = "") -> Tuple[list, int]:
        """:attr:`page_position_scope`, never raising."""
        try:
            names, total = self.page_position_scope(node_id, item)
        except Exception:                      # noqa: BLE001 — mid-edit upstream
            return [], 0
        return ([str(n) for n in names] if isinstance(names, (list, tuple)) else []), int(total or 0)

    def batch_member_names(self, node_id: str) -> list:
        """The member names of the batch reaching ``node_id`` — ``[]`` if none does.

        Walks back along the Dataset chain to the nearest ``util.batch`` and reads the
        display name of each source wired into it, in wiring order. That is the same list
        ``util.batch`` stamps as ``batch_file`` at pull time, resolved here from the GRAPH
        so an ``util.unbatch`` card can draw its per-member outputs **before anything has
        been pulled** — the envelope cannot answer, because the member count is invisible
        to a ``meta_transform`` (it is handed input 0 alone) and the names are a pull-time
        stamp.

        Resolved live rather than captured into params, unlike the channel and group
        descriptors. Those describe a FILE, which a reopened graph may no longer be able to
        see; this describes the WIRING, which the document always has in front of it — so
        recomputing it cannot go stale, and rewiring the Batch node moves the sockets with
        it instead of leaving a captured list to disagree.

        Duplicate names are suffixed exactly as the compute does, because two members
        sharing a label would make the unbatch's outputs indistinguishable on the canvas.
        """
        bid = self._batch_node_for(node_id)
        if bid is None:
            return []
        out: list = []
        seen: Dict[str, int] = {}
        for src, _ssock, dst, _dsock in self.edges:
            if dst != bid:
                continue
            name = self._source_display_name(src)
            n = seen.get(name, 0)
            seen[name] = n + 1
            out.append(name if n == 0 else f"{name} ({n + 1})")
        return out

    def _batch_node_for(self, node_id: str) -> Optional[str]:
        """The nearest ``util.batch`` upstream of ``node_id`` (or ``node_id`` itself).

        Stops at the FIRST one found walking back, and stops entirely at a
        ``util.select_batch`` — past that tap the batch is already one member, so anything
        downstream of it is not in a batch any more and must not grow member sockets.
        """
        seen: set = set()
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            rec = self.nodes.get(nid)
            op = rec.op_key if rec else ""
            if op == BATCH_OP:
                return nid
            if op == SELECT_BATCH_OP and nid != node_id:
                continue                       # the batch ends here
            stack.extend(e[0] for e in self.edges if e[2] == nid)
        return None

    def _source_display_name(self, node_id: str) -> str:
        """A member's IDENTITY — the base name of the file the source card carries, else
        the node id. Never blank, because this is what a ``util.select_batch`` tap is keyed
        by and what it resolves against the batch at pull time.

        **Deliberately not** :data:`TITLE_KEY`, tempting though it is to show the name the
        user typed. A title is a UI annotation and :data:`_UI_PARAM_KEYS` strips it on the
        way to the engine, so a tap keyed by it would be resolvable here and unresolvable
        in the run graph — the card would look right and the pull would refuse. The label
        beside the socket may still show a title (see :meth:`batch_member_label`); the
        identity has to be something the engine can also see.
        """
        return batch_member_identity(self.nodes.get(node_id), node_id)

    def batch_member_label(self, node_id: str, index: int, identity: str) -> str:
        """What to WRITE beside a member's output socket — the user's title when there is
        one, otherwise the identity. Cosmetic only; the wire is keyed by the identity."""
        srcs = [e[0] for e in self.edges if e[2] == node_id]
        if 0 <= index < len(srcs):
            rec = self.nodes.get(srcs[index])
            title = str(rec.params.get(TITLE_KEY) or "").strip() if rec else ""
            if title:
                return title
        return identity

    def source_channel_total(self, node_id: str) -> int:
        """The total channel count of the source file(s) feeding ``node_id`` — what the
        wire tint, the socket text and the socket dot use to decide whether a stream carries
        a strict channel subset (:meth:`channel_subset`). Walks up to the source roots and
        takes the max: a root ``io.load`` reports its captured ``__channels__`` length, else
        its seeded envelope's ``c``; a root ``page.input`` asks the Workspace for the FILE's
        total on the page it reads from (:attr:`page_channel_scope`, V4.00 step 11g), because
        its own envelope is already the narrowed stream and would call one channel of three
        "all of them"."""
        seen: set = set()
        stack = [node_id]
        totals = []
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            preds = [e for e in self.edges if e[2] == nid]
            if preds:
                stack.extend(e[0] for e in preds)
            else:
                rec = self.nodes.get(nid)
                chans = rec.params.get(CHANNELS_KEY) if rec else None
                if isinstance(chans, list) and chans:
                    totals.append(len(chans))
                    continue
                up_total = (self._page_scope(nid)[1]
                            if rec is not None and rec.op_key == PAGE_INPUT_OP else 0)
                totals.append(up_total if up_total > 0 else self.env(nid).axes.c)
        return max(totals) if totals else self.env(node_id).axes.c

    def source_scope_totals(self, node_id: str) -> Tuple[int, int, int]:
        """``(M, T, Z)`` of the source data feeding ``node_id`` — how many frames and
        planes there are to choose from, i.e. how far the Viewer's M/T/Z strips span and
        therefore what can be picked on them under the GUI's troubleshooting scope
        (:meth:`nodelab_v2.window.MainWindow.set_solo_frame`).

        Read off the SOURCE roots, not off ``node_id``'s own envelope, and that is the
        point: a chain that collapses time (Temporal Stack, a T reduction) publishes
        ``t == 1`` downstream and one that projects z publishes ``z == 1``, but the user
        still needs to choose *which* source frame and plane the run is scoped to — the
        scope is applied at the source seed, upstream of both. Same root walk as
        :meth:`source_channel_total`; the max over roots, so a graph fed by two files
        offers the longer series.
        """
        seen: set = set()
        stack = [node_id]
        totals: List[Tuple[int, int, int]] = []
        while stack:
            nid = stack.pop()
            if nid in seen or nid not in self.nodes:
                continue
            seen.add(nid)
            preds = [e for e in self.edges if e[2] == nid]
            if preds:
                stack.extend(e[0] for e in preds)
            else:
                ax = self.env(nid).axes
                totals.append((ax.m, ax.t, ax.z))
        if not totals:
            ax = self.env(node_id).axes
            totals = [(ax.m, ax.t, ax.z)]
        return tuple(max(1, max(t[i] for t in totals)) for i in range(3))

    def source_scope_extent(self, node_id: str) -> Tuple[int, int]:
        """``(Y, X)`` of the source data feeding ``node_id`` — the frame the troubleshooting
        REGION box (2026-10-02) is dragged over. Same root walk as
        :meth:`source_scope_totals`, for the same reason: the region is applied at the
        source seed, upstream of any crop, resample or stitch in the chain, so it has to be
        expressed in the SOURCE's pixels even when the viewed node's frame is a different
        size. The max over roots, so a graph fed by two files offers the larger frame."""
        seen: set = set()
        stack = [node_id]
        extents: List[Tuple[int, int]] = []
        while stack:
            nid = stack.pop()
            if nid in seen or nid not in self.nodes:
                continue
            seen.add(nid)
            preds = [e for e in self.edges if e[2] == nid]
            if preds:
                stack.extend(e[0] for e in preds)
            else:
                ax = self.env(nid).axes
                extents.append((ax.y, ax.x))
        if not extents:
            ax = self.env(node_id).axes
            extents = [(ax.y, ax.x)]
        return (max(1, max(e[0] for e in extents)), max(1, max(e[1] for e in extents)))

    # ── wiring (G1) ──────────────────────────────────────────────────────────
    def _socket_spec(self, node_id: str, io: str, name: str):
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return None
        if io == "in":
            # UNFILTERED by the grow groups: a slot the card is not showing yet is still a
            # real socket, and a saved graph (or a script) may wire it directly — the
            # filter then reveals it and everything before it.
            pool = list(spec.active_inputs(rec.state())) + self.mode_port_specs(node_id)
        else:
            pool = self.output_specs(node_id)
        return next((s for s in pool if s.name == name), None)

    def can_connect(self, src: str, src_socket: str, dst: str, dst_socket: str
                    ) -> Tuple[bool, str]:
        """(ok, reason). Validates direction/type via ``sockets.can_connect`` on the
        ACTIVE socket specs, self-loops, and cycles."""
        if src == dst:
            return False, "self-loop"
        a = self._socket_spec(src, "out", src_socket)
        b = self._socket_spec(dst, "in", dst_socket)
        if a is None or b is None:
            return False, "unknown or inactive socket"
        sa, sb = a.instantiate(), b.instantiate()
        if sa.direction is not Direction.OUT or sb.direction is not Direction.IN:
            return False, "direction"
        if not _can_connect(sa, sb):
            hint = ("" if not dst_socket.startswith(MODE_TARGET_PREFIX)
                    else " — set this variable's Type to 'text' to sweep a dropdown")
            return False, f"{sa.type.value} → {sb.type.value} is not connectable{hint}"
        # A DRIVER wire is invisible to the DAG, so it cannot close a cycle — and refusing
        # it here would make the node unusable, since driving a param downstream and
        # collecting the result back is a loop on the canvas BY DESIGN.
        if not is_driver_edge(self, (src, src_socket, dst, dst_socket)) \
                and self._creates_cycle(src, dst):
            return False, "would create a cycle (rejected outside zones)"
        return True, ""

    def _creates_cycle(self, src: str, dst: str) -> bool:
        """True if adding dst←src closes a cycle (src reachable FROM dst). Driver wires are
        skipped for the same reason ``Graph.topo_order`` skips them: the rewrite consumes
        them, so they order nothing."""
        stack, seen = [dst], set()
        while stack:
            nid = stack.pop()
            if nid == src:
                return True
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(e[2] for e in self.edges
                         if e[0] == nid and not is_driver_edge(self, e))
        return False

    def connect(self, src: str, src_socket: str, dst: str, dst_socket: str
                ) -> List[EdgeTuple]:
        """Add the wire (validated). A non-multi input's existing wire is REPLACED.
        Returns the list of edges removed by the replacement."""
        ok, reason = self.can_connect(src, src_socket, dst, dst_socket)
        if not ok:
            raise ValueError(f"cannot connect: {reason}")
        removed: List[EdgeTuple] = []
        b = self._socket_spec(dst, "in", dst_socket)
        if b is not None and not b.multi:
            removed = [e for e in self.edges if e[2] == dst and e[3] == dst_socket]
            for e in removed:
                self.edges.remove(e)
        edge = (src, src_socket, dst, dst_socket)
        if edge not in self.edges:
            self.edges.append(edge)
        self._name_output_from_wire(dst, dst_socket, src, src_socket)
        self._notify()
        return removed

    #: Whether a Page Output on this document takes its name from its first wire. A linked
    #: page says no (its names are its master's; see ``LinkedDocument``).
    AUTO_NAMES_OUTPUTS = True

    def _name_output_from_wire(self, dst: str, dst_socket: str, src: str,
                               src_socket: str) -> None:
        """A Page Output whose name is still the placeholder (``out``, ``out2``) or blank
        takes its name from the first wire into its first slot (V4.00 step 11h): the
        position (``B03``) or channel (``Cy5``) a tap socket carries, a part's name
        (``mask``), the node's title (``gaussian_blur``) — the same answer an ITEM gets
        (:meth:`_item_default`), made unique like any name. Only a placeholder is replaced:
        a name the user typed is theirs. A Page Input already reading the old name follows
        it (:attr:`output_renamed`), so wiring an Output a new page was auto-bound to does
        not unbind that page."""
        if not getattr(self, "AUTO_NAMES_OUTPUTS", True):
            return
        rec = self.nodes.get(dst)
        if rec is None or rec.op_key != PAGE_OUTPUT_OP or dst_socket != PAGE_ITEM_SOCKETS[0]:
            return
        old = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
        if old and not _PLACEHOLDER_NAME_RE.match(old):
            return
        try:
            base = sanitize_output_name(self._item_default(src, src_socket))
        except Exception:                            # noqa: BLE001 — mid-edit upstream
            return
        if not base or _PLACEHOLDER_NAME_RE.match(base):
            return
        rec.params[PAGE_NAME_KEY] = base
        self._settle_output_name(dst)
        new = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
        if old and new and new != old:
            try:
                self.output_renamed(old, new)
            except Exception:                        # noqa: BLE001 — a workspace mid-change
                pass

    def disconnect(self, src: str, src_socket: str, dst: str, dst_socket: str) -> None:
        e = (src, src_socket, dst, dst_socket)
        if e in self.edges:
            self.edges.remove(e)
            self._notify()

    def edge_into(self, dst: str, dst_socket: str) -> Optional[EdgeTuple]:
        return next((e for e in self.edges if e[2] == dst and e[3] == dst_socket), None)

    # ── iterate: the segment (V2.22) ─────────────────────────────────────────
    def iterate_aliases(self, *, sweep_all: frozenset = frozenset()) -> Dict[str, str]:
        """``node id → the clone to pull instead`` for every node INSIDE an Iterate
        segment. The segment's END keeps its own id (the selector wears it), so it is
        absent here and needs no translation."""
        try:
            return _iterate_aliases(self.to_graph(for_run=True, materialize=True),
                                    envs=self.envs, sweep_all=sweep_all)
        except Exception:            # noqa: BLE001 — a mid-edit graph aliases nothing
            return {}

    def iterate_card_at(self, node_id: str) -> Optional[str]:
        """The Iterate card whose segment ENDS at ``node_id`` — i.e. whose iterations that
        node's payload is a choice between. ``node_id`` itself when it IS a card.

        This is what puts the iteration strip on the right node. After the rewrite the
        selector wears the end node's id, so the end node is exactly the place where
        "which iteration am I looking at?" is a question with an answer."""
        rec = self.nodes.get(node_id or "")
        if rec is None:
            return None
        if rec.op_key == ITERATE_OP:
            return node_id
        for nid, other in self.nodes.items():
            if other.op_key != ITERATE_OP:
                continue
            if any(e[0] == node_id and e[2] == nid and e[3] == _SEG_TO
                   for e in self.edges):
                return nid
        return None

    def iterate_segment(self, iterate_id: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """``(starts, ends)`` — what is wired into this card's ``from`` and ``to``."""
        starts = tuple(e[0] for e in self.edges
                       if e[2] == iterate_id and e[3] == _SEG_FROM)
        ends = tuple(e[0] for e in self.edges
                     if e[2] == iterate_id and e[3] == _SEG_TO)
        return starts, ends

    # ── iterate targets (V2.22) ──────────────────────────────────────────────
    def iterate_targets(self, iterate_id: str, slot: int) -> List[EdgeTuple]:
        """The driver wires leaving variable slot ``slot`` of ``iterate_id``."""
        outs = set(_var_out_names(slot))
        return [e for e in self.edges if e[0] == iterate_id and e[1] in outs]

    def set_iterate_target(self, iterate_id: str, slot: int, target_node: str = "",
                           target_socket: str = "") -> None:
        """Point one Iterate variable slot at a parameter — the dropdown's whole edit.

        The wire stays the storage. Choosing from the menu builds exactly the driver edge a
        drag would have built, so save/load, the canvas, the rewrite and the "driven by"
        note in the inspector all keep working with nothing new to know about; the picker is
        a second way to make the same statement, not a second place it lives. An empty
        ``target_node`` clears the slot.

        Two things are set for the user rather than left as a trap. The slot's ``v{k}_type``
        follows the TARGET (text for a Mode or a string param, number otherwise) because
        those are two different output sockets and picking the wrong one is a wire that
        simply refuses to connect; and ``variables`` is raised to cover the slot, since a
        slot the count hides is a wire the rewrite would silently pass over."""
        rec = self.nodes.get(iterate_id)
        if rec is None or rec.op_key != ITERATE_OP:
            raise ValueError(f"{iterate_id!r} is not an Iterate node")
        if not 0 <= int(slot) < _MAX_VARIABLES:
            raise ValueError(f"variable slot {slot} is outside 0…{_MAX_VARIABLES - 1}")
        slot = int(slot)
        for e in self.iterate_targets(iterate_id, slot):
            self.edges.remove(e)
        if not target_node:
            self._notify()
            return
        kind = self._iterate_target_kind(target_node, target_socket)
        rec.modes[_var_mode_names(slot)[0]] = kind
        try:
            live = int(rec.modes.get("variables", "1") or 1)
        except (TypeError, ValueError):
            live = 1
        if live < slot + 1:
            rec.modes["variables"] = str(slot + 1)
        num_out, txt_out = _var_out_names(slot)
        self.connect(iterate_id, num_out if kind == _TYPE_NUMBER else txt_out,
                     target_node, target_socket)

    def _iterate_target_kind(self, target_node: str, target_socket: str) -> str:
        """Which of a slot's two driver outputs can reach this target. A Mode port is a
        name, and so is anything a STRING can convert to; everything a FLOAT reaches is
        numeric. Same test :func:`nodegraph.sockets.can_connect` applies to the drag, so the
        menu and the wire agree by construction."""
        if target_socket.startswith(MODE_TARGET_PREFIX):
            return _TYPE_TEXT
        sock = self._socket_spec(target_node, "in", target_socket)
        if sock is None:
            raise ValueError(f"{target_node}.{target_socket} is not an active parameter")
        return (_TYPE_NUMBER if _can_convert(SocketType.FLOAT, sock.type)
                else _TYPE_TEXT)

    # ── graph / envelopes ────────────────────────────────────────────────────
    def to_graph(self, *, for_run: bool = False, materialize: bool = False,
                 bypass_muted: Optional[bool] = None,
                 live_docks: frozenset = frozenset(),
                 unroll_iterate: bool = False,
                 sweep_all: frozenset = frozenset()) -> Graph:
        """The headless :class:`Graph`. ``for_run`` strips the UI-only param annotations
        (``__locked__``/``__title__``/``__channels__`` — they must not re-key the memo)
        and resolves MUTED nodes by bypassing them (first Dataset input → their
        consumers). ``materialize`` rewrites each synthetic per-channel output edge
        (``chK``) into a real ``channel.select`` tap AND inlines every group instance
        (``group:<name>``) into its body via :func:`nodegraph.groups.expand` — used for
        the run graph and the edit-time envelope pass, but NOT for saving (the file keeps
        the instance nodes + separate group definitions).

        ``bypass_muted`` splits the mute resolution out of ``for_run`` (2026-07-30), which
        is what :meth:`propagate` needs: the edit-time envelope pass must describe the graph
        that will actually RUN, and it was passing ``for_run=False``, so every muted node
        still transformed the envelope. The visible consequence was a whole GUI describing a
        pipeline nobody would execute — derived spinboxes, the layer picker, the domain rail
        and the missing-domain chip all computed through nodes the run drops — and the
        silent one was that pinning such a spinbox froze the mute-blind number into
        ``params`` and carried it into the run. It defaults to ``for_run`` so every existing
        caller is unchanged.

        ``unroll_iterate`` additionally expands every ``flow.iterate`` cone into its
        per-iteration clones (:func:`nodegraph.iterate.unroll`). It is a SEPARATE flag from
        ``materialize`` on purpose: :meth:`propagate` also builds a materialized graph, and
        unrolling there would delete the very node ids the inspector, the layer picker and
        the domain rail look their envelopes up by. The un-unrolled graph is also the right
        one to describe at edit time — the driver wires are non-DAG, so every node's
        envelope is exactly what its own authored params say. Only the RUN graph unrolls.
        ``sweep_all`` names the Iterate nodes that must mint every iteration regardless of
        their preserve mode (the "Run sweep" action).

        ``materialize`` also **cuts every docked dock's in-edge** (V2.18,
        :func:`~nodelab_v2.ops.cut_docked_inputs`), which is what stops the engine walking
        a docked chain. It rides on ``materialize`` rather than ``for_run`` for exactly the
        reason mute rides on ``bypass_muted``: :meth:`propagate` must describe the graph
        that will RUN, and a docked node's envelope comes from its checkpoint manifest (it
        is a root there), not from an upstream it no longer reads. Saving still uses the
        UNcut graph, so the file always keeps the chain that produced the bake."""
        g = Graph()
        drop_muted = for_run if bypass_muted is None else bypass_muted
        for rec in self.nodes.values():
            params = dict(rec.params)
            if for_run:
                for k in _UI_PARAM_KEYS:
                    params.pop(k, None)
                # A dock's folder is stored relative to the saved graph (portability);
                # the worker thread and the engine have no idea where that file is, so
                # the run graph carries the resolved ABSOLUTE path. Note the bake record
                # (`__bake__`) deliberately survives here — unlike the UI annotations it
                # must re-key the memo, so that a re-bake invalidates everything
                # downstream of the dock.
                if rec.op_key == DOCK_OP and params.get("store"):
                    params["store"] = self.dock_store(rec.id)
                # A source card's resolved GROUPS ride into the run graph, where
                # `materialize_group_taps` turns a `grpK` edge into a `util.select_group`
                # carrying that group's KEY. Written HERE rather than captured when the
                # file was picked, so that a card nobody opted in never detects anything:
                # `group_descriptors` returns [] unless the lever is `auto`, which both
                # supplies the keys when it is and strips a stale list when it is not.
                if rec.op_key == LOAD_OP:
                    descs = self.group_descriptors(rec.id)
                    if descs:
                        params[GROUPS_KEY] = descs
                    else:
                        params.pop(GROUPS_KEY, None)
            modes = dict(rec.modes)
            # `live_docks` forces a dock to pass through for THIS graph only — what a
            # re-bake needs, since the node is currently docked and its in-edge would be
            # cut, so the bake would read its own old checkpoint and write it back over
            # itself. The forced state folds into the recipe hash like any mode, so the
            # live pull and the docked pull memoize as the different computations they
            # are, and neither can serve the other's result.
            if rec.id in live_docks and rec.op_key == DOCK_OP:
                modes["state"] = DOCK_LIVE
            g.add(NodeInstance(rec.id, rec.op_key, params=params, modes=modes))
        edges = list(self.edges)
        if drop_muted:
            edges = self._bypass_muted(edges)
        for (s, ss, d, ds) in edges:
            g.connect(s, d, src_socket=ss, dst_socket=ds,
                      kind="driver" if is_driver_edge(self, (s, ss, d, ds)) else "forward")
        for e in self._back_edges:            # preserved zone-feedback edges verbatim
            g.connect(e.src, e.dst, src_socket=e.src_socket,
                      dst_socket=e.dst_socket, kind=e.kind)
        if materialize and self._groups:      # inline group instances into their bodies
            g = _group_expand(g, self._groups)
        if not materialize:
            return g
        if unroll_iterate:
            # Groups first (a group body may contain a driven node), Iterate second, dock
            # cuts + channel taps last — one tap is then shared by every clone, which is
            # right because a tap upstream of the cone is loop-invariant.
            g = _iterate_unroll(g, envs=self.envs, sweep_all=sweep_all)
        return prepare_run_graph(g)

    def _bypass_muted(self, edges: List[EdgeTuple],
                      muted: Optional[frozenset] = None) -> List[EdgeTuple]:
        """Mute-with-passthrough (G3): rewire around each muted node via its first
        connected Dataset input; a muted SOURCE simply drops its out-edges. ``muted`` names
        the switched-off nodes instead of their records' flags — how a linked page asks what
        an edit WOULD do to its master before making it (V4.00 step 11e)."""
        for rec in self.nodes.values():
            if not (rec.muted if muted is None else rec.id in muted):
                continue
            ds_ins = [s.name for s in self.input_specs(rec.id)
                      if s.type is SocketType.DATASET]
            feed = next((e for e in edges if e[2] == rec.id and e[3] in ds_ins), None)
            # A DRIVER wire is not a data route, so muting must drop it rather than
            # re-attach it to the bypass source — rewiring one would point a parameter at
            # the muted node's input, which is not a value at all.
            outs = [e for e in edges if e[0] == rec.id and not is_driver_edge(self, e)]
            edges = [e for e in edges if e[0] != rec.id and e[2] != rec.id]
            if feed is not None:
                for (_, ss_out, d, ds) in outs:
                    # a PART wire stays a part wire (step 11f): a node that may be switched
                    # off keeps the kind of data, so its input carries the same parts
                    src_rec = self.nodes.get(feed[0])
                    keep = PART_SOCKET_RE.match(ss_out or "") and src_rec is not None and \
                        feed[1] == _real_dataset_out(src_rec.op_key)
                    edges.append((feed[0], ss_out if keep else feed[1], d, ds))
        return edges

    # ── zone creation (Repeat) ─────────────────────────────────────────────────
    def _dataset_inputs(self, node_id: str) -> set:
        # via input_specs so a group instance's synthetic Dataset input resolves too
        return {s.name for s in self.input_specs(node_id)
                if s.type is SocketType.DATASET}

    def wrap_repeat_zone(self, node_ids, iterations: int = 3) -> str:
        """Wrap a selected **linear** sub-chain in a Repeat zone: insert paired
        ``zone.repeat_in``/``zone.repeat_out`` boundary nodes, rewire the single
        dataset frontier in/out through them, add the ``Out→In`` back-edge, and record
        a :class:`~nodegraph.zones.Zone`. The result is validated by
        :func:`nodegraph.zones.unroll` before commit — a malformed selection raises
        ``ValueError`` and leaves the document untouched. Group creation and Sim/nested
        zones are a later phase (they need a distinct UX / eval design)."""
        sel = set(node_ids)
        if not sel:
            raise ValueError("select the nodes to wrap in a Repeat zone")
        if not sel <= set(self.nodes):
            raise ValueError("selection includes unknown nodes")
        if any(nid in z.members for z in self._zones for nid in sel):
            raise ValueError("a selected node is already in a zone (nested zones are a "
                             "later phase)")
        if any(self.nodes[nid].op_key.startswith(("zone.", "group."))
               for nid in sel):
            raise ValueError("don't wrap zone/group boundary nodes")
        # the dataset frontier: edges crossing the selection boundary
        ext_in = [e for e in self.edges
                  if e[2] in sel and e[0] not in sel and e[3] in self._dataset_inputs(e[2])]
        ext_out = [e for e in self.edges if e[0] in sel and e[2] not in sel]
        if len(ext_in) != 1 or len(ext_out) != 1:
            raise ValueError(
                "select a single linear sub-chain with exactly one dataset input and "
                f"one output (found {len(ext_in)} in, {len(ext_out)} out)")

        snapshot = (dict(self.nodes), list(self.edges), list(self._back_edges),
                    list(self._zones))
        try:
            (es, ess, ed, eds) = ext_in[0]
            (xs, xss, xd, xds) = ext_out[0]
            rin, rout = self.new_id("rin"), self.new_id("rout")
            # position the boundary nodes just outside the chain's span
            xs_min = min(self.nodes[n].x for n in sel)
            xs_max = max(self.nodes[n].x for n in sel)
            ymid = sum(self.nodes[n].y for n in sel) / len(sel)
            self.nodes[rin] = NodeRecord(rin, "zone.repeat_in", x=xs_min - 150, y=ymid)
            self.nodes[rout] = NodeRecord(rout, "zone.repeat_out", x=xs_max + 230, y=ymid)
            self.edges.remove(ext_in[0])
            self.edges.remove(ext_out[0])
            self.edges += [
                (es, ess, rin, "data"), (rin, "out", ed, eds),
                (xs, xss, rout, "data"), (rout, "out", xd, xds)]
            self._back_edges.append(Edge(rout, rin, "out", "data", "back"))
            zid = self.new_id("z")
            self._zones.append(Zone(zid, "repeat", rin, rout,
                                    body=frozenset(sel), iterations=max(1, int(iterations))))
            _unroll(self.to_graph(for_run=False), self._zones)   # validate — raises if bad
        except Exception:
            (self.nodes, self.edges, self._back_edges, self._zones) = (
                dict(snapshot[0]), list(snapshot[1]), list(snapshot[2]), list(snapshot[3]))
            raise
        self._notify()
        return zid

    # ── group creation / ungrouping ────────────────────────────────────────────
    def _unique_group_name(self, base: str) -> str:
        existing = {g.name for g in self._groups}
        name = (base or "Group").strip() or "Group"
        if name not in existing:
            return name
        i = 2
        while f"{name} {i}" in existing:
            i += 1
        return f"{name} {i}"

    def make_group(self, node_ids, name: str = "Group") -> str:
        """Collapse a selected **linear** sub-chain into one reusable group instance node
        (Blender's 'Make Group'): the selection becomes a :class:`~nodegraph.groups.Group`
        DEFINITION (its body, bounded by ``group.input``/``group.output`` boundaries) and
        is replaced in the parent by a single ``group:<name>`` instance node wired to the
        same single dataset frontier. It runs / propagates by inlining the body
        (``to_graph(materialize)`` → :func:`nodegraph.groups.expand`). Validated by a trial
        expand before commit — a malformed selection raises and leaves the document
        untouched. Reverse with :meth:`ungroup`."""
        sel = set(node_ids)
        if not sel:
            raise ValueError("select the nodes to group")
        if not sel <= set(self.nodes):
            raise ValueError("selection includes unknown nodes")
        if any(self.nodes[nid].op_key.startswith(("zone.", "group."))
               or group_name_of(self.nodes[nid].op_key) for nid in sel):
            raise ValueError("don't group zone/group boundary or group-instance nodes")
        if any(nid in z.members for z in self._zones for nid in sel):
            raise ValueError("a selected node is in a zone (group-in-zone is a later phase)")
        if any(not any(e[2] == nid for e in self.edges) for nid in sel):
            raise ValueError("don't group a source node (its seed would be buried); leave "
                             "sources outside the group")
        ext_in = [e for e in self.edges if e[2] in sel and e[0] not in sel
                  and e[3] in self._dataset_inputs(e[2])]
        ext_out = [e for e in self.edges if e[0] in sel and e[2] not in sel]
        if len(ext_in) != 1 or len(ext_out) != 1:
            raise ValueError(
                "select a single linear sub-chain with exactly one dataset input and one "
                f"output (found {len(ext_in)} in, {len(ext_out)} out)")

        snapshot = (dict(self.nodes), list(self.edges), list(self._groups))
        try:
            gname = self._unique_group_name(name)
            (es, ess, ed, eds) = ext_in[0]         # external src → interior dst(eds)
            (xs, xss, xd, xds) = ext_out[0]        # interior src(xss) → external dst
            body = Graph()
            for nid in sel:
                r = self.nodes[nid]
                body.add(NodeInstance(nid, r.op_key, params=dict(r.params),
                                      modes=dict(r.modes)))
            gin, gout = "grp.in", "grp.out"        # boundary ids (unique within the body)
            body.add(NodeInstance(gin, GROUP_INPUT))
            body.add(NodeInstance(gout, GROUP_OUTPUT))
            for (s, ss, d, ds) in self.edges:      # interior edges (both ends inside)
                if s in sel and d in sel:
                    body.connect(s, d, src_socket=ss, dst_socket=ds)
            body.connect(gin, ed, src_socket="out", dst_socket=eds)    # input → interior
            body.connect(xs, gout, src_socket=xss, dst_socket="data")  # interior → output
            grp = Group(gname, body, gin, gout)
            inst = self.new_id("grp")
            cx = sum(self.nodes[n].x for n in sel) / len(sel)
            cy = sum(self.nodes[n].y for n in sel) / len(sel)
            for nid in sel:
                del self.nodes[nid]
                self.meta_seeds.pop(nid, None)
                self._prune_frames(nid)
            self.edges = [e for e in self.edges if e[0] not in sel and e[2] not in sel]
            self.nodes[inst] = NodeRecord(inst, grp.op_key, x=cx, y=cy)
            self.edges.append((es, ess, inst, "data"))
            self.edges.append((inst, "out", xd, xds))
            self._groups.append(grp)
            _group_expand(self.to_graph(for_run=False), self._groups)   # validate — raises
        except Exception:
            (self.nodes, self.edges, self._groups) = (
                dict(snapshot[0]), list(snapshot[1]), list(snapshot[2]))
            raise
        self._notify()
        return inst

    def ungroup(self, inst_id: str) -> bool:
        """Inline a group instance back into the parent as real nodes (the reverse of
        :meth:`make_group`): the body interior is restored with FRESH collision-free ids,
        its internal edges + the single dataset frontier are reconnected, and the instance
        + its now-unused group definition are removed. Returns ``False`` (no-op) if
        ``inst_id`` is not a group instance."""
        rec = self.nodes.get(inst_id)
        name = group_name_of(rec.op_key) if rec is not None else None
        grp = self._group_by_name(name) if name else None
        if grp is None:
            return False
        in_edge = next((e for e in self.edges if e[2] == inst_id), None)   # ext → inst.data
        out_edges = [e for e in self.edges if e[0] == inst_id]             # inst.out → ext
        idmap: Dict[str, str] = {}
        for k, bnid in enumerate(sorted(grp.interior)):
            bnode = grp.body.nodes[bnid]
            nid = self.new_id()
            idmap[bnid] = nid
            self.nodes[nid] = NodeRecord(nid, bnode.op_key, params=dict(bnode.params),
                                         modes=dict(bnode.modes),
                                         x=rec.x + k * 40, y=rec.y + k * 30)
        for e in grp.body.edges:                   # interior edges (no boundary endpoint)
            if e.src in (grp.input_id, grp.output_id) or \
                    e.dst in (grp.input_id, grp.output_id):
                continue
            si, di = idmap.get(e.src), idmap.get(e.dst)
            if si and di:
                self.edges.append((si, e.src_socket, di, e.dst_socket))
        for e in grp.body.edges:                   # boundary edges → external frontier
            if e.src == grp.input_id and in_edge is not None:
                di = idmap.get(e.dst)
                if di:
                    self.edges.append((in_edge[0], in_edge[1], di, e.dst_socket))
            if e.dst == grp.output_id:
                si = idmap.get(e.src)
                if si:
                    for (_s, _ss, xd, xds) in out_edges:
                        self.edges.append((si, e.src_socket, xd, xds))
        self.edges = [e for e in self.edges if e[0] != inst_id and e[2] != inst_id]
        del self.nodes[inst_id]
        self._prune_frames(inst_id)
        self._groups = [g for g in self._groups if g.name != name]
        self._notify()
        return True

    def propagate(self) -> None:
        """Re-run the edit-time MetaEnvelope pass (G8 live re-seed). Sources without
        a seed get an ALL-UNKNOWN envelope — unknown is not z==1, so the H11 lever
        guard never greys 3D just because a source hasn't resolved yet."""
        self.dormant = self._dormant_nodes()
        try:
            # to_graph(materialize) inlines group instances; a malformed group (e.g. a
            # loaded file referencing a missing definition) raises here too — swallow it
            # like a mid-edit cycle so a single bad state never crashes the live re-seed.
            # `bypass_muted` matches what the RUNNER builds (runner.py's for_run=True), so
            # the envelopes the GUI shows describe the graph that will actually execute.
            g = self.to_graph(materialize=True, bypass_muted=True)
            seeds = dict(self.meta_seeds)
            # A DOCKED node is a root in this graph (its in-edge is cut), so its envelope
            # must come from its checkpoint's manifest — axes, calibration, domain set and
            # layer catalog, read without touching a pixel. Without this the whole
            # downstream half of the graph would describe itself from an all-unknown
            # source: no derived spinbox values, an empty layer picker, blank domain
            # rails. Seeded per propagate rather than cached because the user can
            # re-bake, repoint or delete a dock at any time; the read is one small JSON.
            seeds.update(self._dock_seed_envs())
            # A PAGE INPUT is a root here too (V4.00): its envelope is the upstream page's
            # Output envelope, supplied by the Workspace through a hook and read without
            # touching a pixel, exactly like a dock's manifest. Outside a workspace there is
            # no hook and the Input stays all-unknown, which is the truth.
            for hook in list(self.seed_hooks):
                seeds.update(hook())
            # A source card may STATE its own calibration, overriding (or supplying) what
            # the file carries — the only way to give a plain TIFF a Z spacing, which it
            # never records. Applied to the seed here rather than only in the runner so the
            # whole graph re-derives on the keystroke instead of on the next pull: every
            # µm→px default downstream reads it, and the 3D lever's guard consults it. The
            # runner applies the SAME override to the payload it resolves
            # (`runner._with_card_calib`), so the header the user is typing against and the
            # pixels that arrive never disagree.
            for nid, rec in self.nodes.items():
                if rec.op_key != LOAD_OP:
                    continue
                over = calib_overrides(rec.params)
                if over:
                    seeds[nid] = seeds.get(nid, MetaEnvelope()).with_metadata(**over)
            unknown = MetaEnvelope(unknown_axes=frozenset(AXIS_ORDER))
            for nid in g.roots():
                seeds.setdefault(nid, unknown)
            self.envs = propagate_meta(g, seeds)
        except ValueError:
            self.envs = {}                        # a mid-edit cycle / bad group: no envelopes
        # a group instance is inlined in the expanded graph, so it has no envelope of its
        # own — give it its body's OUTPUT envelope so its card + every downstream node's
        # domain rail read correctly THROUGH the opaque instance.
        for nid, rec in self.nodes.items():
            gname = group_name_of(rec.op_key)
            if not gname:
                continue
            grp = self._group_by_name(gname)
            if grp is not None:
                out_env = self.envs.get(_inst_id(grp.output_id, nid))
                if out_env is not None:
                    self.envs[nid] = out_env

    def _group_by_name(self, name: str) -> Optional[Group]:
        return next((g for g in self._groups if g.name == name), None)

    # ── docks (V2.18) ─────────────────────────────────────────────────────────
    def dock_nodes(self) -> List[str]:
        """Every ``io.dock`` node in the document, in insertion order."""
        return [nid for nid, rec in self.nodes.items() if rec.op_key == DOCK_OP]

    def _dock_seed_envs(self) -> Dict[str, MetaEnvelope]:
        """The manifest envelope for each docked node that has a readable checkpoint.

        ``docked`` only, deliberately. A ``held`` dock's envelope cannot come from here —
        there is no manifest to read — so the runner supplies it from the held payload
        through :meth:`set_meta_seed`, the same hook an envelope the document is *handed*
        rather than reads already uses. Falling back to a derived one here would describe
        the held node by whatever the chain looks like NOW, which is exactly the thing a
        frozen node is not."""
        from nodegraph.checkpoint import checkpoint_envelope
        out: Dict[str, MetaEnvelope] = {}
        for nid in self.dock_nodes():
            rec = self.nodes[nid]
            if not is_docked(rec):
                continue
            try:
                env = checkpoint_envelope(self.dock_store(nid))
            except Exception:  # noqa: BLE001 — a bad/absent store just stays unknown
                env = None
            if env is not None:
                out[nid] = env
        return out

    def _dormant_nodes(self) -> frozenset:
        """The greyed-out set: nodes no docked run will evaluate. Read off the plain
        (un-materialized, un-muted-bypassed) graph so the ids are the ones the canvas
        draws — a group instance greys as a whole, which is what the user sees."""
        if not any(is_frozen(r) for r in self.nodes.values()):
            return frozenset()          # FROZEN: a held dock greys its chain out too
        try:
            return dormant_nodes(self.to_graph())
        except Exception:  # noqa: BLE001 — mid-edit; nothing greys rather than crashing
            return frozenset()

    def dock_store(self, node_id: str) -> str:
        """The dock's checkpoint directory as an **absolute** path.

        Stored relative to the saved graph whenever it sits beside it (see
        :meth:`to_dict`), so moving a project folder — or handing it to a colleague —
        keeps every dock attached. Resolved against the graph's directory here, which is
        the one place that knows it."""
        rec = self.nodes.get(node_id)
        if rec is None:
            return ""
        import os
        raw = str(rec.params.get("store", "") or "").strip().strip('"').strip("'").strip()
        if not raw:
            return ""
        if os.path.isabs(raw):
            return os.path.normpath(raw)
        base = os.path.dirname(os.path.abspath(self.path)) if self.path else os.getcwd()
        return os.path.normpath(os.path.join(base, raw))

    def default_dock_store(self, node_id: str) -> str:
        """Where a Bake of ``node_id`` should write when the user has not chosen a
        folder: ``<graph-name>.docks/<node_id>`` beside the saved file.

        Keyed by node id rather than by a name the user could duplicate, so two docks in
        one graph can never bake into each other's folder. ``NODEGRAPH_STORE_DIR``
        redirects it exactly as it redirects an ingest store — the reason is the same
        (the data may live on slow media), and so is the uniqueness requirement, hence
        the path digest when it is redirected."""
        import hashlib
        import os
        from nodegraph.parallel import store_dir
        if self.path:
            # graphs are saved as `<name>.nd2graph.json`, so one splitext leaves
            # `<name>.nd2graph` — strip that too, or every project grows a folder called
            # `MyExperiment.nd2graph.docks`.
            stem = os.path.splitext(os.path.abspath(self.path))[0]
            if stem.endswith(".nd2graph"):
                stem = stem[: -len(".nd2graph")]
            base = stem + ".docks"
        else:
            base = os.path.join(os.getcwd(), "untitled.docks")
        # a page's docks live one folder down (its page id), so two pages' `n3` never
        # bake into one folder (V4.00)
        leaf = (self.store_tag, node_id) if self.store_tag else (node_id,)
        target = store_dir(os.path.dirname(base))
        if os.path.abspath(target) == os.path.abspath(os.path.dirname(base)):
            return os.path.join(base, *leaf)
        tag = hashlib.blake2b(base.lower().encode("utf-8"), digest_size=6).hexdigest()
        return os.path.join(target,
                            f"{os.path.basename(base)}.{tag}", *leaf)

    def dock_signature(self, node_id: str) -> str:
        """The upstream signature of ``node_id`` right now — compared against the one
        recorded at bake time to decide whether the dock has gone stale."""
        try:
            sig = upstream_signature(self.to_graph(bypass_muted=True), node_id)
            # A dock fed through a Page Input depends on the upstream page's chain too
            # (V4.00 step 5): fold it in — and only then, so a dock whose chain stays on its
            # page keeps exactly the signature its bake recorded.
            ext = self.cross_page_signature(node_id)
            return digest("dock-sig-pages", sig, ext) if ext else sig
        except Exception:  # noqa: BLE001 — mid-edit: never claim staleness on a bad graph
            return ""

    def dock_status(self, node_id: str) -> Tuple[str, str]:
        """``(status, detail)`` for a dock card — see
        :func:`nodelab_v2.ops.dock_status`. Resolved against the document so the store
        path is the absolute one and the signature matches what a run would see.

        **Memoized per document revision**, because the caller is
        :meth:`nodelab_v2.node_item.NodeItem.paint` — it runs on every repaint, and the
        answer costs a graph build, an upstream-closure digest and a JSON read off disk.
        Uncached, hovering the canvas re-read every dock's manifest tens of times a
        second. Every edit that can change the answer bumps the revision, including a
        finished bake (:meth:`set_dock_bake` notifies), so the cache cannot go stale
        against anything the user did in the app."""
        rec = self.nodes.get(node_id)
        if rec is None or rec.op_key != DOCK_OP:
            return ("", "")
        # The held set is part of the key: holding or releasing changes the answer for a
        # `held` node ("held" vs "released") without editing the graph, so it cannot bump the
        # revision — bumping it would invalidate every memo entry, which is the opposite of
        # what a hold is for.
        # …and the workspace's run identity of this page: an edit on an UPSTREAM page can
        # stale a dock fed through a Page Input without bumping this page's revision
        key = (node_id, self.revision, self._held_nodes, self.workspace_revision())
        hit = self._dock_status_cache.get(key)
        if hit is not None:
            return hit
        try:
            g = self.to_graph(bypass_muted=True)
            got = (_dock_status(g, node_id, signature=self.dock_signature(node_id),
                                store=self.dock_store(node_id), held=self._held_nodes)
                   if node_id in g.nodes else ("", ""))
        except Exception:  # noqa: BLE001 — mid-edit: say nothing rather than crash a paint
            return ("", "")
        if len(self._dock_status_cache) > 64:         # revisions climb forever
            self._dock_status_cache.clear()
        self._dock_status_cache[key] = got
        return got

    def set_dock_bake(self, node_id: str, *, store: str, bake_id: str,
                      precision: str, signature: str, nbytes: int = 0,
                      when: str = "") -> None:
        """Record a finished bake on ``node_id`` and switch it to ``docked``.

        Writes the store path (relative to the saved graph when it sits beside it, so the
        project stays portable) plus the machine-set bake record the memo and the
        staleness check read, then flips the mode — one notify, so the canvas greys the
        chain and the envelopes re-propagate from the manifest in a single step."""
        import os
        rec = self.nodes.get(node_id)
        if rec is None:
            return
        rec.params["store"] = self._relative_store(store)
        rec.params[BAKE_KEY] = {"id": str(bake_id), "sig": str(signature),
                                "precision": str(precision), "bytes": int(nbytes),
                                "at": str(when)}
        rec.modes["state"] = DOCK_DOCKED
        if precision:
            rec.modes["precision"] = str(precision)
        self._notify()

    def set_held_nodes(self, node_ids: Any) -> None:
        """Mirror the runner's hold registry so the canvas can tell ``held`` from
        ``released``.

        Clears the status cache and pings the listeners so the cards repaint, but deliberately
        does **not** go through :meth:`_notify`, because that bumps ``revision`` — and the
        revision is what keys the memo and the runner's cached Engine. Bumping it here would
        invalidate every memo entry, dropping exactly the computed payloads a hold exists to
        keep, which is the opposite of the feature. Holding is not an edit to the user's graph.
        """
        ids = frozenset(node_ids)
        if ids == self._held_nodes:
            return
        self._held_nodes = ids
        self._dock_status_cache.clear()
        for fn in list(self._listeners):        # repaint only — no revision, no re-propagate
            fn()

    def set_dock_hold(self, node_id: str, held: bool) -> None:
        """Switch ``node_id`` between ``held`` and ``live``.

        The counterpart of :meth:`set_dock_bake` for the session tier, and deliberately much
        thinner: there is no store to record, no bake id to claim and no precision to stamp,
        because nothing was written. The mode is the whole state."""
        rec = self.nodes.get(node_id)
        if rec is None or rec.op_key != DOCK_OP:
            return
        rec.modes["state"] = DOCK_HELD if held else DOCK_LIVE
        self._notify()

    def _relative_store(self, store: str) -> str:
        """``store`` relative to the saved graph's folder when it lives under it, else
        absolute. Keeps a project directory movable without pinning a dock that the user
        deliberately put on another drive."""
        import os
        if not store:
            return ""
        store = os.path.normpath(os.path.abspath(store))
        if not self.path:
            return store
        base = os.path.dirname(os.path.abspath(self.path))
        try:
            rel = os.path.relpath(store, base)
        except ValueError:                        # different drive on Windows
            return store
        return store if rel.startswith(os.pardir) else rel

    def set_dock_state(self, node_id: str, docked: bool) -> None:
        """Dock / un-dock without baking. Un-docking runs the chain again from the top;
        the checkpoint is left on disk, so re-docking is instant."""
        rec = self.nodes.get(node_id)
        if rec is None or rec.op_key != DOCK_OP:
            return
        want = DOCK_DOCKED if docked else DOCK_LIVE
        if rec.modes.get("state", DOCK_LIVE) != want:
            rec.modes["state"] = want
            self._notify()

    def env(self, node_id: str) -> MetaEnvelope:
        return self.envs.get(node_id, MetaEnvelope())

    def source_choices(self, node_id: str) -> list:
        """``[(value, label), ...]`` a ``page.input``'s Source menu offers (V4.00): the named
        outputs of the pages that may feed this one, supplied by the Workspace through
        :attr:`page_sources`. Empty outside a workspace — there is nothing to read."""
        return list(self.page_sources())

    # ── layer picker (V2.11) ───────────────────────────────────────────────────
    def layer_choices(self, node_id: str, sock) -> list:
        """The layer names a ``layer_in`` socket should offer — those present on the
        edge feeding this node, in the domain the socket declares.

        Deliberately NOT modelled on :meth:`input_domains`, which UNIONS every Dataset
        predecessor. Three nodes take a second Dataset input whose layers must never be
        offered here (``analysis.measure``'s ``raw``, and the ``reference`` of
        ``dvc_field``/``dic_correlate``): the payload flows from the PRIMARY input, and
        ``propagate_meta`` agrees — it builds the envelope from ``dataset_preds[0]``
        alone. So follow the primary edge by default.

        The exception is a socket declaring **``layer_from``** (V2.22), which reads its
        layer off a NAMED second input because its node combines structure produced by two
        different branches — ``analysis.voronoi`` takes its seed dots from one chain and
        the areas it clips them to from another, and no single wire can carry both. Such a
        socket follows that edge, and falls back to the primary when it is unwired, which
        is what keeps a pre-existing single-wire graph offering the right names.

        Never falls back to this node's OWN envelope: that already contains the layers
        this node writes, so a source picker would offer the node its own output."""
        if sock is None:
            return []
        domain = sock.layer_in
        if not domain and sock.layer_in_mode:
            rec = self.nodes.get(node_id)
            spec = rec.spec() if rec else None
            value = rec.state().get(sock.layer_in_mode) if (rec and spec) else None
            try:
                domain = Domain(value) if value else None
            except ValueError:                      # a mode value that is not a Domain
                domain = None
        if domain is None:
            return []
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return []
        primary = next((s.name for s in spec.inputs
                        if s.type is SocketType.DATASET), None)
        if primary is None:
            return []
        # a `layer_from` socket reads the named second input; unwired, it falls back to
        # the primary exactly as the compute does
        edge = None
        if sock.layer_from:
            edge = self.edge_into(node_id, sock.layer_from)
        if edge is None:
            edge = self.edge_into(node_id, primary)
        if edge is None:
            return []
        return list(self.env(edge[0]).layers_in(domain))

    # ── column picker (V2.28) ──────────────────────────────────────────────────
    def column_choices(self, node_id: str, sock) -> list:
        """The column names a ``column_in`` socket should offer — those the edit-time pass
        knows were MEASURED onto the incoming edge, in the domain the socket declares.

        The layer picker's sibling, one level down, and it follows the same primary edge for
        the same reason (``layer_choices``): a second Dataset input's columns are not on the
        wire the payload flows down.

        Two differences, both because a column is a narrower thing than a layer.
        ``column_join`` adds domains beyond the socket's own — a condition on Label rows can
        test ``track_length``, which lives on the Track table and reaches those rows by the
        ``member_id`` join the compute performs, so the picker has to offer it or the
        tracking half of the palette is reachable only by typing. And ``column_from`` names
        the layer socket that says WHICH instance's columns to list; without it the domain's
        layers are unioned, which is what a socket whose layer is inferred needs.

        Unlike ``layer_choices`` this list is CLOSED — the combo it fills is not
        editable. Every structure-producing node declares ``adds_columns``
        (``selftest::test_column_catalog_complete``), so the columns a table carries are
        determined by the nodes upstream and there is nothing legitimate to type that is
        not here. A layer picker cannot make that promise, which is why that one stays
        free text."""
        if sock is None:
            return []
        domains = []
        base = getattr(sock, "column_in", None)
        if not base and getattr(sock, "column_in_mode", ""):
            rec = self.nodes.get(node_id)
            spec = rec.spec() if rec else None
            value = rec.state().get(sock.column_in_mode) if (rec and spec) else None
            try:
                base = Domain(value) if value else None
            except ValueError:                      # a mode value that is not a Domain
                base = None
        if base is not None:
            domains.append(base)
        domains.extend(getattr(sock, "column_join", ()) or ())
        if not domains:
            return []
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return []
        primary = next((s.name for s in spec.inputs
                        if s.type is SocketType.DATASET), None)
        if primary is None:
            return []
        edge = self.edge_into(node_id, primary)
        if edge is None:
            return []
        env = self.env(edge[0])
        # `column_from` names a LAYER socket on this node; its current value is the instance
        # whose columns to list. Empty (or an unset socket) unions the domain, which is the
        # only honest answer when the compute will resolve the layer by only-candidate.
        layer = None
        src = getattr(sock, "column_from", "")
        if src and rec is not None:
            layer = str(rec.params.get(src) or "") or None
        out: list = []
        for dom in domains:
            # the join domains are separate TABLES, so a layer name from the member domain
            # must not be used to filter them — it would match nothing and silently hide
            # every joined column.
            want = layer if dom is base else None
            for c in env.columns_in(dom, want):
                if dom is not base and c in _JOIN_KEYS:
                    continue
                if c not in out:
                    out.append(c)
        return out

    # ── domain interface (socket rail + wire tint + validation) ────────────────
    def input_domains(self, node_id: str) -> frozenset:
        """The accumulated domain-set arriving on ``node_id``'s Dataset input(s) —
        the union of every Dataset-predecessor's output domain-set. Empty for a
        source (its domains come from its own ``adds_domains``)."""
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return frozenset()
        ds_ins = {s.name for s in spec.inputs if s.type is SocketType.DATASET}
        doms: frozenset = frozenset()
        for src, _ssock, dst, dsock in self.edges:
            if dst == node_id and dsock in ds_ins:
                doms = doms | self.env(src).domains
        return doms

    def own_label_layers(self, node_id: str) -> list:
        """The Voxel layer names this node ITSELF produces, in declaration order.

        What the Labels overlay should draw when it has not been told otherwise: you view a
        node to see what it made. Before this, "Auto" ranked every raster on the payload by
        the largest id in the plane — so viewing `analysis.voronoi` could draw the seeds
        branch's labels, or the copied areas raster (whose ids are the *source's*, up in the
        hundreds, while only a handful of its regions survive), and neither is the node's
        answer (2026-08-04).

        Declaration order is the priority: a `layer_out` socket names the node's real output
        (`voronoi`) and `extra_layers` the auxiliary copies (`voronoi_areas`), which is exactly
        the order to prefer them in."""
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return []
        from nodegraph.registry import layer_value
        out: list = []
        for s in spec.inputs:
            if Domain.VOXEL in (s.layer_out or ()):
                nm = str(layer_value(s, rec.params) or "").strip()
                if nm and nm not in out:
                    out.append(nm)
        extra = getattr(spec, "extra_layers", None)
        if extra is not None:
            try:
                for dom, nm in extra(dict(rec.params), dict(rec.state())) or ():
                    if dom is Domain.VOXEL and nm and str(nm) not in out:
                        out.append(str(nm))
            except Exception:  # noqa: BLE001 — extra_layers must never break the view
                pass
        return out

    def socket_domains(self, node_id: str, socket: str) -> frozenset:
        """The domains arriving on ONE Dataset input socket (empty if it is unwired).

        :meth:`input_domains` unions every Dataset predecessor, which is right for deciding
        whether a requirement is met but useless for saying *which wire brought what*. On a
        node with two Dataset inputs the card paints the same node-level rail beside each one,
        so hovering was the only place left that could distinguish them — and it could not,
        until this (2026-08-04)."""
        edge = self.edge_into(node_id, socket)
        return self.env(edge[0]).domains if edge is not None else frozenset()

    def missing_domains(self, node_id: str) -> frozenset:
        """Required domains (``reads_domains``) absent from the upstream set — the
        GUI's red validation chips (e.g. a Measure node with no Label upstream)."""
        rec = self.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None:
            return frozenset()
        # resolved against this instance's mode state (V2.22) — a conditional read
        # (`analysis.voronoi` under `bound=per_region`) is required here and absent two
        # dropdown values away, and a static answer had to be wrong in one of them.
        return spec.missing_domains(self.input_domains(node_id), rec.state())

    def set_meta_seed(self, node_id: str, env: MetaEnvelope) -> None:
        """Record a source's RESOLVED envelope (the G8 live re-seed, delivered by the runner
        after a pull resolved a file).

        Notified with an EMPTY touched set, which means "changed nothing a run can see"
        (2026-08-06) — deliberately different from ``None``, "changed something unknown". The
        envelope is display metadata the engine already resolved for itself during the run
        that is delivering it, so it cannot invalidate that run's answer or anyone else's.

        Saying ``None`` here is not a harmless over-approximation, it is a bug with a long
        history: the delivery path already had to be written so staleness was judged BEFORE
        this call, because the notification loops back through the window and cancels
        in-flight work. Once a QUEUE existed, that same loop-back also emptied it — so
        asking for two branches ran the first, silently discarded the second, and left its
        card sitting on ``queued`` forever. Caught on the real 16-position file; no synthetic
        fixture reaches it, because it needs a source whose envelope is resolved late.

        An UNCHANGED envelope is a no-op (2026-08-10): the runner re-announces whenever a
        source's resolved key changes, and a delivery arriving after the seed is already
        current must not re-notify — even an empty-set notification bumps the document
        revision, and anything keyed on it (the runner's held-view revision tie, the
        finished-results map) pays for a bump that announced nothing."""
        if self.meta_seeds.get(node_id) == env:
            return
        self.meta_seeds[node_id] = env
        self._notify(())

    # ── save / load (G6) ─────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        # carry any loaded zones/groups through unchanged — the GUI edits only the
        # nodes+edges layer, but must never DROP structure it can't yet edit.
        d = _ng_to_dict(self.to_graph(), zones=self._zones, groups=self._groups)
        d["ui"] = {
            "nodes": {rec.id: {"x": rec.x, "y": rec.y, "muted": rec.muted,
                               "collapsed": rec.collapsed}
                      for rec in self.nodes.values()},
            "frames": {fr.id: {"title": fr.title, "members": list(fr.members),
                               "color": list(fr.color) if fr.color else None}
                       for fr in self.frames.values()},
        }
        return d

    def to_page_dict(self) -> Dict[str, Any]:
        """:meth:`to_dict` without the ``format_version`` — the body this document contributes
        to a workspace file as ONE page (V4.00)."""
        d = self.to_dict()
        d.pop("format_version", None)
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        """Load a single-graph document (``format_version`` checked by the serializer; a
        WORKSPACE document loads its active page — see :class:`nodelab_v2.workspace.Workspace`
        for all of them)."""
        graph, zones, groups = _ng_from_dict(d)
        self._load_parsed(graph, zones, groups, d.get("ui"))

    def load_page(self, rec: Dict[str, Any]) -> None:
        """Load ONE workspace page record (V4.00): a single-graph body plus ``ui``, with no
        ``format_version`` of its own — the Workspace checked the document's."""
        graph, zones, groups = _ng_page_from_dict(rec)
        self._load_parsed(graph, zones, groups, rec.get("ui"))

    def _load_parsed(self, graph, zones, groups, ui_raw) -> None:
        bad = sorted(n for n in graph.nodes if "/" in n)
        if bad:
            # refused before anything changes: "/" separates the page from the node in a
            # page-qualified run id (V4.00), so such a node could never be pulled
            raise ValueError(f"node id(s) {bad} contain '/', which this version reserves")
        ui = ui_raw if isinstance(ui_raw, dict) else {}
        ui_nodes = ui.get("nodes", {}) if isinstance(ui.get("nodes", {}), dict) else {}
        self.nodes.clear()
        self.edges = []
        self.meta_seeds.clear()
        self._zones = list(zones)                    # preserved verbatim (not yet edited)
        self._groups = list(groups)
        # A DRIVER edge is a first-class document edge (the canvas draws it, the user made
        # it); only zone feedback rides in `_back_edges` as un-editable structure. Its kind
        # is re-derived on save from the socket it leaves, so the round-trip is exact
        # without the document ever storing a kind.
        self._back_edges = [e for e in graph.edges
                            if e.kind not in ("forward", "driver")]
        for nid, inst in graph.nodes.items():
            extra = ui_nodes.get(nid, {})
            self.nodes[nid] = NodeRecord(
                nid, self._OP_RENAMES.get(inst.op_key, inst.op_key),
                params=dict(inst.params), modes=dict(inst.modes),
                x=float(extra.get("x", 0.0)), y=float(extra.get("y", 0.0)),
                muted=bool(extra.get("muted", False)),
                collapsed=bool(extra.get("collapsed", False)))
        for e in graph.edges:
            if e.kind in ("forward", "driver"):      # back-edges ride in _back_edges
                self.edges.append((e.src, e.src_socket, e.dst,
                                   self._migrate_socket(e.dst, e.dst_socket)))
        # GUI frames (only members that survived the load are kept; empty ⇒ dropped)
        self.frames = {}
        ui_frames = ui.get("frames", {}) if isinstance(ui.get("frames", {}), dict) else {}
        for fid, fd in ui_frames.items():
            if not isinstance(fd, dict):
                continue
            mem = [n for n in fd.get("members", []) if n in self.nodes]
            if mem:
                col = fd.get("color")
                self.frames[fid] = FrameRecord(fid, fd.get("title", "Frame"), mem,
                                               tuple(col) if col else None)
        self._notify()

    #: Sockets renamed after files had already been saved against them: ``{op_key: {old:
    #: new}}``. A wire into a socket the spec no longer declares is not drawable, not
    #: connectable and invisible to the rewrite — the graph would open looking subtly fine
    #: and quietly stop iterating — so the loader repoints it. ``flow.iterate``'s ``collect``
    #: became the segment's ``to`` in V2.22, and it means the same thing: the end of the
    #: series.
    _SOCKET_RENAMES: Dict[str, Dict[str, str]] = {ITERATE_OP: {"collect": _SEG_TO}}

    #: Node types renamed after files had already been saved against them: ``{old op_key:
    #: new}``. A record naming an op the registry no longer has would open as "unrecognized
    #: node type" with its wires dead; the loader repoints it instead. ``util.chain`` (Chain
    #: Files) became ``util.timeseries`` (Timeseries Builder) on 2026-10-02 — same sockets
    #: for the first file (``data``) and the same Modes, with a wider default (``time``).
    _OP_RENAMES: Dict[str, str] = {"util.chain": "util.timeseries"}

    def _migrate_socket(self, node_id: str, socket: str) -> str:
        rec = self.nodes.get(node_id)
        if rec is None:
            return socket
        return self._SOCKET_RENAMES.get(rec.op_key, {}).get(socket, socket)

    @property
    def has_unedited_structure(self) -> bool:
        """True if this file carries zones/back-edges the GUI can't fully edit yet (all
        preserved on save; surfaced so the window can warn the user). Groups are now
        GUI-manageable (make/ungroup), so they no longer count."""
        return bool(self._zones or self._back_edges)

    def rebase_path(self, path: str) -> None:
        """Make ``path`` this document's file, re-anchoring every dock store to it.

        Dock store paths are stored RELATIVE to the saved graph whenever they sit beside
        it, so a project folder stays movable. That makes them a function of `self.path`
        — so a Save As must re-anchor every one of them, or the docks would silently point
        at the old location's folders. Resolve to absolute against the OLD path first, then
        re-relativize against the new one. Split out of :meth:`save_file` (V4.00) because a
        workspace file re-anchors every page's docks to ONE path before writing."""
        absolute = {nid: self.dock_store(nid) for nid in self.dock_nodes()}
        self.path = path
        for nid, store in absolute.items():
            if store:
                self.nodes[nid].params["store"] = self._relative_store(store)

    def save_file(self, path: str) -> None:
        """Write this document alone as a single-graph (2.0) file."""
        import json
        self.rebase_path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def load_file(self, path: str) -> None:
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # `path` is set BEFORE the load: `load_dict` notifies, which re-propagates, which
        # resolves every dock's relative store folder against the graph's directory. Set
        # afterwards and that first propagation would look for the docks in the working
        # directory and describe a whole loaded pipeline as unknown.
        self.path = path
        self.load_dict(d)


__all__ = ["GraphDocument", "NodeRecord", "FrameRecord", "LOCKED_KEY",
           "pass_through_reason", "NO_PASS_THROUGH_PREFIXES"]
