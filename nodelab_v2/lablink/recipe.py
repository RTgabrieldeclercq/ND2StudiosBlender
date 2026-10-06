"""Turning a graph into a recipe — the manifest generator.

A recipe is two files in one directory: a ``recipe.json`` manifest and the
``graph.nd2graph.json`` the editor already saves. The manifest is the part nobody wants to
hand-write, and **this software is the right party to generate it** because it is the only
one holding both things every check runs against: the graph, and the node catalogue that
says what each socket is called and what unit it is in. A manifest generated from those two
can be made unable to fail validation; a hand-written one is a guess about both.

So almost every knob field here is *derived*, not asked for:

==================  ========================================================
knob field          where it comes from
==================  ========================================================
``node``/``param``  the graph node and the socket or mode being exposed
``kind``            ``param`` for a socket, ``mode`` for a mode — read from
                    different places at evaluation time, so the wrong one is a
                    silent no-op rather than an error
``unit``            the socket's own declared unit, verbatim. A knob claiming
                    ``nm`` against a ``um`` socket is refused by tier 2
                    precisely because the number would be accepted and would
                    mean something a thousand times different
``type``            the socket's :class:`~nodegraph.sockets.SocketType`
``enum``            the mode's own ``choices``, or the socket's
``label``/``help``  the node's own UI strings
``default``         the value the graph pins, when it pins one
``unset_means``     ``derive`` when the socket declares a derive expression
                    and the graph does not pin it — mutually exclusive with a
                    default, and refused over a pinned param
``applies_when``    the socket's ``available_in`` mode gate
==================  ========================================================

**The one thing that cannot be derived is ``min``/``max``.** ``SocketSpec`` carries no
numeric bounds by design — the editor's spin boxes go to 1e12 and the compute validates
instead, because a silent clamp is the worst way to be told no. A recipe's bounds are a
statement about what a *remote caller* may ask for, which is a different question from what
the maths accepts, and only a person can answer it. :func:`suggest_bounds` proposes; the
author decides.

Qt-free, so the generator can be tested and scripted without a display.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from nodelab_v2.lablink import protocol as P

RECIPE_FORMAT = "lablink.recipe/1"
GRAPH_FILENAME = "graph.nd2graph.json"
MANIFEST_FILENAME = "recipe.json"

#: Socket types a caller may be given. Everything else is structural (``DATASET``) or has no
#: safe wire form here (``VECTOR``, ``COLOR``, ``MENU``).
_TYPE_FOR_SOCKET = {
    "FLOAT": "float",
    "INT": "int",
    "BOOL": "bool",
    "STRING": "string",
}

#: Units the manifest vocabulary accepts, mirrored from the hub's own recipe schema. A socket
#: declaring anything else cannot be exposed with its unit intact, and exposing it *without*
#: the unit would publish a bare number whose meaning is unstated — so it is not offered.
_ALLOWED_UNITS = frozenset({"", "um", "nm", "um_axial", "um2", "um3", "s", "px"})


def slugify(text: str, *, fallback: str = "node") -> str:
    out = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    return out or fallback


# ── what a graph offers ─────────────────────────────────────────────────────────

@dataclass
class KnobCandidate:
    """One param or mode that *could* become a knob, with everything already derived."""

    node_id: str
    op_key: str
    node_label: str
    param: str
    kind: str                       # "param" | "mode"
    type: str                       # float | int | bool | enum | string | channel_list
    unit: str = ""
    label: str = ""
    help: str = ""
    enum: Tuple[Any, ...] = ()
    #: What the graph currently pins, if anything. Becomes the knob's ``default``.
    pinned: Any = None
    has_pinned: bool = False
    #: The socket declares a metadata-derived default, so ``unset_means: "derive"`` is
    #: available — but only if the graph does not also pin it.
    can_derive: bool = False
    #: A derived ``applies_when``, when the gate is expressible.
    applies_when: Optional[Dict[str, Any]] = None
    #: Why this candidate is worth exposing, or a caution about it.
    notes: Tuple[str, ...] = ()
    #: Set when the socket influences the spatial kernel (a radius, a sigma) — the params
    #: most worth turning, so they are pre-ticked.
    suggested: bool = False

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.node_id, self.param, self.kind)

    def default_knob_name(self, taken: Iterable[str] = ()) -> str:
        """A knob name a person would recognise, unique within the recipe.

        The unit rides in the name where there is one — ``min_area_um2``, not ``min_area`` —
        because that is the convention the shipped recipes use and because a caller reading
        ``/workflows`` sees the name long before they see the unit field.
        """
        base = self.param
        if self.unit and not base.endswith(self.unit):
            base = f"{base}_{self.unit}"
        name = base
        used = set(taken)
        n = 2
        while name in used:
            name = f"{self.node_id}_{base}" if n == 2 else f"{base}_{n}"
            n += 1
        return name


def assign_names(candidates: Sequence[KnobCandidate]) -> Dict[Tuple[str, str, str], str]:
    """``{candidate.key: knob name}`` for a whole set, uniquely.

    Naming the set rather than one candidate at a time, because a name is only unique with
    respect to the others: two nodes with a 2D/3D lever both want to be called ``dim``, and a
    per-candidate call cannot know that. Two knobs sharing a name is refused by tier 1, so
    getting this wrong locally means a recipe that cannot be installed.
    """
    out: Dict[Tuple[str, str, str], str] = {}
    taken: List[str] = []
    for cand in candidates:
        name = cand.default_knob_name(taken)
        taken.append(name)
        out[cand.key] = name
    return out


def _mode_state(rec: Any, spec: Any) -> Dict[str, str]:
    """The node's resolved mode state — what ``available_in`` is evaluated against."""
    state = {}
    for mode in getattr(spec, "modes", ()) or ():
        state[mode.name] = str(rec.modes.get(mode.name, mode.resolved_default()))
    return state


def _derive_condition(available_in: Optional[Mapping[str, Any]], rec: Any, spec: Any,
                      exposed_modes: Mapping[str, str]) -> Tuple[Optional[Dict[str, Any]],
                                                                 List[str]]:
    """``(applies_when, notes)`` for a socket gated by ``available_in``.

    Three outcomes, and the difference matters because two of them produce a knob that is
    silently never read:

    * the gate is on a mode this recipe *also* exposes -> a real ``applies_when``, so a
      client can grey the control out instead of someone setting it, uploading a file and
      then being refused;
    * the gate is on a mode the graph pins, and the pinned value satisfies it -> no
      condition needed, the knob is always read;
    * the gate is on a mode the graph pins to something that does *not* satisfy it -> the
      knob would be **dead**, and that is said rather than shipped.
    """
    notes: List[str] = []
    if not available_in:
        return None, notes
    gates = {k: set(v) for k, v in available_in.items()}
    if len(gates) > 1:
        # `applies_when` holds exactly one condition and may not be chained, so a
        # multi-gated socket cannot be described. Not fatal — but the author should know the
        # control cannot be greyed accurately.
        names = ", ".join(sorted(gates))
        notes.append(f"read only under a combination of {names}; a recipe can declare only "
                     f"one condition, so a client cannot grey this out accurately")
    name, allowed = sorted(gates.items())[0]
    if name in exposed_modes:
        values = sorted(str(v) for v in allowed)
        cond = ({"knob": exposed_modes[name], "equals": values[0]} if len(values) == 1
                else {"knob": exposed_modes[name], "in": values})
        return cond, notes
    # Not exposed: the graph decides it, statically.
    current = _mode_state(rec, spec).get(name)
    if current is None:
        return None, notes
    if current not in {str(v) for v in allowed}:
        notes.append(f"NEVER READ: this node's {name} is {current!r}, and this param is "
                     f"only read when {name} is {', '.join(sorted(map(str, allowed)))}")
    return None, notes


def candidates(graph: Any, *, doc: Any = None) -> List[KnobCandidate]:
    """Every param and mode of ``graph`` that could be offered as a knob.

    Forbidden params are **filtered out rather than rejected**: a path param would let a
    caller choose which file the hub opens, which is the entire security boundary, and the
    editor-bookkeeping keys would change the graph's cache key without changing the
    computation — silently discarding the warm cache that is the reason the hub exists.
    Letting someone pick one and then meet a refusal is a worse experience than never
    offering it.
    """
    from nodegraph.registry import NODES
    from nodelab_v2.ops import ensure_ops

    ensure_ops()
    out: List[KnobCandidate] = []
    for node_id, rec in sorted(graph.nodes.items()):
        spec = NODES.get(rec.op_key)
        if spec is None:
            continue
        title = str(rec.params.get("__title__") or "") or spec.label
        state = _mode_state(rec, spec)
        # Modes first: a mode may gate the params below it, and a param's condition can only
        # name a mode this recipe also exposes.
        exposed_modes: Dict[str, str] = {}
        mode_candidates: List[KnobCandidate] = []
        for mode in getattr(spec, "modes", ()) or ():
            if not mode.choices:
                continue
            pinned = rec.modes.get(mode.name, mode.resolved_default())
            notes: List[str] = []
            if getattr(mode, "is_dim_lever", False):
                notes.append("the 2D/3D lever. Exposing it lets a caller choose; leaving it "
                             "to the graph pins the dimensionality the recipe was built for")
            mode_candidates.append(KnobCandidate(
                node_id=node_id, op_key=rec.op_key, node_label=title,
                param=mode.name, kind="mode", type="enum",
                label=str(getattr(mode, "label", "") or mode.name),
                help=str(getattr(mode, "description", "") or ""),
                enum=tuple(mode.choices), pinned=pinned, has_pinned=True,
                notes=tuple(notes)))
            # A mode is nameable by a condition only once it is a knob; assume the author
            # will take the offered ones, and drop the condition later if they do not.
            exposed_modes[mode.name] = mode.name
        out.extend(mode_candidates)

        for sock, reachable_now in _offerable_sockets(spec, state):
            if sock.type.name == "DATASET":
                continue
            if sock.name in P.FORBIDDEN_KNOB_PARAMS or getattr(sock, "path_kind", ""):
                continue
            ktype = _knob_type_for(sock)
            if ktype is None:
                continue
            unit = str(getattr(sock, "unit", "") or "")
            if unit not in _ALLOWED_UNITS:
                continue
            cond, notes = _derive_condition(getattr(sock, "available_in", None), rec, spec,
                                            exposed_modes)
            if not reachable_now:
                notes.append(
                    "not read at this node's current mode settings, but reachable once a "
                    "caller turns the mode that gates it")
            pinned = rec.params.get(sock.name, None)
            has_pinned = sock.name in rec.params
            can_derive = bool(getattr(sock, "derive", "")) and not has_pinned
            if getattr(sock, "derive", "") and has_pinned:
                notes.append("the graph pins this, so it cannot be offered as derived — a "
                             "recipe that asked for both would ignore the file's own "
                             "calibration with nothing reporting it")
            out.append(KnobCandidate(
                node_id=node_id, op_key=rec.op_key, node_label=title,
                param=sock.name, kind="param", type=ktype, unit=unit,
                label=str(getattr(sock, "label", "") or sock.name),
                help=str(getattr(sock, "description", "") or ""),
                enum=tuple(getattr(sock, "choices", ()) or ()),
                pinned=pinned, has_pinned=has_pinned, can_derive=can_derive,
                applies_when=cond, notes=tuple(notes),
                suggested=_worth_offering(sock)))
    return out


def _worth_offering(sock: Any) -> bool:
    """Whether to pre-tick this param — a starting point, never a decision.

    Four marks, each of which says "this param is about the specimen rather than about the
    implementation", which is the line between a knob worth publishing and a knob that only
    makes the offer list longer:

    * ``kernel_param`` — it sizes the spatial kernel (a radius, a sigma), so it is the
      first thing anybody retunes on new data;
    * a ``derive`` expression — it is metadata-intelligent, so a caller may legitimately want
      to override what the file says;
    * a physical unit — a value in µm or µm² means the same specimen at any magnification,
      which is precisely what makes it safe to expose to a machine that knows nothing about
      the objective;
    * a channel picker — which channel to analyse is a property of the acquisition, not of
      the pipeline.
    """
    if getattr(sock, "kernel_param", False) or getattr(sock, "derive", ""):
        return True
    if str(getattr(sock, "unit", "") or "") in ("um", "um2", "um3", "um_axial"):
        return True
    return getattr(sock, "pick_kind", "") == "channels"


#: Ceiling on the mode combinations :func:`_offerable_sockets` will enumerate. Real nodes sit
#: far below it (``analysis.segment``'s three modes give 48), and the fallback below is
#: correct rather than merely cheaper, so the cap costs nothing but a pathological case.
_MAX_MODE_STATES = 512


def _offerable_sockets(spec: Any, state: Mapping[str, str]) -> List[Tuple[Any, bool]]:
    """``[(socket, reachable_at_current_state)]`` for every socket a caller could ever reach.

    Not simply ``spec.active_inputs(state)``. A socket gated on ``dim == "3D"`` is inactive
    while the graph sits at 2D, but the recipe is about to expose ``dim`` as a knob — so a
    caller *can* reach it, and offering only what is active right now silently withholds
    every 3D knob from a recipe that fully supports 3D. That is exactly the omission the
    shipped hand-written recipe had to work around by listing the volume knobs itself.

    So the offer is the union over every mode combination, with a flag saying whether each
    socket is live as the graph currently stands — which is worth telling the author, because
    a knob that only matters under a mode they did not expose is a knob nothing will read.
    """
    modes = [m for m in (getattr(spec, "modes", ()) or ()) if m.choices]
    live = {s.name for s in spec.active_inputs(state)}

    combinations = 1
    for mode in modes:
        combinations *= max(1, len(mode.choices))
    states: List[Dict[str, str]] = []
    if modes and combinations <= _MAX_MODE_STATES:
        states = [dict(state)]
        for mode in modes:
            grown: List[Dict[str, str]] = []
            for base in states:
                for choice in mode.choices:
                    variant = dict(base)
                    variant[mode.name] = str(choice)
                    grown.append(variant)
            states = grown
    else:
        # One mode varied at a time. Misses a socket gated on two modes at once in a
        # combination neither singleton reaches — rare, and it only ever under-offers, which
        # the author can correct by ticking. Over-offering could not be corrected.
        states = [dict(state)]
        for mode in modes:
            for choice in mode.choices:
                variant = dict(state)
                variant[mode.name] = str(choice)
                states.append(variant)

    seen: Dict[str, Any] = {}
    for candidate_state in states or [dict(state)]:
        for sock in spec.active_inputs(candidate_state):
            seen.setdefault(sock.name, sock)
    return [(sock, name in live) for name, sock in seen.items()]


def _knob_type_for(sock: Any) -> Optional[str]:
    """The manifest knob type for a socket, or ``None`` if it cannot be offered."""
    if getattr(sock, "pick_kind", "") == "channels":
        return "channel_list"
    if getattr(sock, "choices", ()):
        return "enum"
    return _TYPE_FOR_SOCKET.get(sock.type.name)


# ── bounds, the one thing a person must supply ───────────────────────────────────

def suggest_bounds(candidate: KnobCandidate) -> Tuple[Optional[float], Optional[float]]:
    """A starting ``(min, max)`` for a numeric knob — a proposal, never a silent default.

    Anchored on the value the graph already uses, because that is the only evidence in the
    system about the scale this param operates at. A generous ceiling is deliberate: bounds
    exist to stop a caller asking for something absurd, not to encode the author's taste,
    and a range too tight produces a refusal that reads as a broken recipe.
    """
    if candidate.type not in ("float", "int"):
        return None, None
    value = candidate.pinned if isinstance(candidate.pinned, (int, float)) else None
    if value is None or value == 0:
        # No scale to work from. Non-negative and open-ended is the honest proposal, and
        # both units below are the ones where a zero genuinely means "no filtering".
        return (0.0, 1_000_000.0) if candidate.unit in ("um2", "um3") else (0.0, 1000.0)
    lo = 0.0 if value > 0 else float(value) * 10.0
    hi = float(abs(value)) * 20.0
    if candidate.type == "int":
        return float(int(lo)), float(max(int(hi), 1))
    return lo, hi


# ── the draft ───────────────────────────────────────────────────────────────────

@dataclass
class KnobDraft:
    """A candidate the author has chosen, plus the bits only they can decide."""

    candidate: KnobCandidate
    name: str
    label: str = ""
    help: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    #: Expose as derived-from-the-file rather than with a default.
    derive: bool = False
    #: Offer a subset of the mode's choices. Empty means all of them.
    enum: Tuple[Any, ...] = ()
    max_items: Optional[int] = None
    pattern: str = ""
    max_len: Optional[int] = None
    #: Override the graph's pinned value as the published default.
    default: Any = None
    has_default: bool = False

    def to_manifest(self, *, exposed: Mapping[Tuple[str, str], str]) -> Dict[str, Any]:
        """This knob as the manifest wants it."""
        c = self.candidate
        out: Dict[str, Any] = {"name": self.name, "node": c.node_id, "param": c.param,
                               "kind": c.kind, "type": c.type}
        if c.unit:
            out["unit"] = c.unit
        if self.label or c.label:
            out["label"] = self.label or c.label
        if self.help or c.help:
            out["help"] = self.help or c.help
        if c.type == "enum":
            out["enum"] = list(self.enum or c.enum)
        if c.type in ("float", "int"):
            if self.minimum is not None:
                out["min"] = self.minimum
            if self.maximum is not None:
                out["max"] = self.maximum
        if c.type == "channel_list" and self.max_items:
            out["max_items"] = int(self.max_items)
        if c.type == "string" and not (self.enum or c.enum):
            # A string knob must be bounded by an enum, or by a pattern AND a length. An
            # unbounded string from a caller is not something a recipe should accept.
            out["pattern"] = self.pattern or r"[A-Za-z0-9_,.\- ]+"
            out["max_len"] = int(self.max_len or 64)

        if self.derive:
            out["unset_means"] = "derive"
        elif self.has_default:
            out["default"] = self.default
        elif c.has_pinned and c.kind == "mode":
            # A mode always resolves to something, so publishing its current value as the
            # default is what makes the manifest describe the pipeline as it stands.
            out["default"] = c.pinned
        elif c.has_pinned:
            out["default"] = c.pinned

        if c.applies_when:
            cond = dict(c.applies_when)
            # The condition names a KNOB, not a mode. If the controlling mode was not
            # actually exposed, the condition cannot be stated and is dropped — tier 1
            # would refuse it, and a dropped condition is merely imprecise rather than wrong.
            controller = exposed.get((c.node_id, str(cond.get("knob"))))
            if controller:
                cond["knob"] = controller
                out["applies_when"] = cond
        return out


@dataclass
class RecipeDraft:
    """A whole recipe, ready to be validated and written."""

    name: str
    workflow: str = P.WORKER_NAME
    title: str = ""
    description: str = ""
    target: str = ""
    knobs: List[KnobDraft] = field(default_factory=list)
    allow_zones: bool = False
    requires_metadata: Tuple[str, ...] = ()
    #: Metadata a reachable non-default configuration would need. NOT published — the format
    #: cannot express a conditional requirement, and declaring it unconditionally would
    #: refuse honest 2D jobs on single-plane data. Shown to the author instead.
    conditional_metadata: Tuple[str, ...] = ()
    capabilities: Tuple[str, ...] = ()
    typical_s: int = 10
    worst_s: int = 120
    silence_timeout_s: int = 300
    #: The graph, as ``nodegraph.serialize.to_dict`` produces it.
    graph_doc: Dict[str, Any] = field(default_factory=dict)
    #: Loader node ids, in graph order — one declared input each.
    loaders: Tuple[str, ...] = ()

    def exposed(self) -> Dict[Tuple[str, str], str]:
        """``(node_id, param) -> knob name`` for every knob, so conditions can be resolved."""
        return {(k.candidate.node_id, k.candidate.param): k.name for k in self.knobs}

    def to_manifest(self) -> Dict[str, Any]:
        exposed = self.exposed()
        manifest: Dict[str, Any] = {
            "format": RECIPE_FORMAT,
            "name": self.name,
            "title": self.title or self.name,
            "workflow": self.workflow,
            "graph": GRAPH_FILENAME,
            "allow_zones": bool(self.allow_zones),
            "targets": {"primary": self.target},
            "inputs": [
                {"role": "image" if i == 0 else f"image{i + 1}", "node": nid,
                 "required": False,
                 "match": ["*.nd2", "*.tif", "*.tiff"]}
                for i, nid in enumerate(self.loaders)],
            "knobs": [k.to_manifest(exposed=exposed) for k in self.knobs],
            "outputs": self.outputs(),
            "requires": {"capabilities": list(self.capabilities),
                         "metadata": list(self.requires_metadata),
                         "min_worker_protocol": P.WORKER_PROTOCOL_VERSION},
            "limits": {"silence_timeout_s": int(self.silence_timeout_s),
                       "hard_timeout_s": 0},
            "expected_runtime_s": {"typical": int(self.typical_s),
                                   "worst": int(self.worst_s)},
        }
        if self.description:
            manifest["description"] = self.description
        return manifest

    def outputs(self) -> List[Dict[str, Any]]:
        """What the recipe returns.

        A ``metrics`` output is **always** declared. Provenance is otherwise opt-in per
        recipe, and a recipe without one hands back a table and a picture with no record of
        the knobs that produced them — which makes the result impossible to reopen or
        reproduce, and is exactly the hole this whole feature exists to close.
        """
        out = [
            {"name": "measurements", "kind": "table", "format": "csv",
             "from": self.target, "policy": "auto",
             "filename": "{stem}_measurements.csv"},
            {"name": "overlay", "kind": "quicklook", "from": self.target,
             "policy": "auto", "filename": "{stem}_overlay.png"},
            {"name": "metrics", "kind": "metrics", "policy": "auto",
             "filename": "{stem}_metrics.json"},
        ]
        return out


# ── readable node ids ───────────────────────────────────────────────────────────

def readable_ids(graph_doc: Mapping[str, Any],
                 titles: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """``{old_id: slug}`` for a graph the editor produced.

    The editor names nodes ``n1``, ``n2``, … which is fine on a canvas where the card
    carries the label. In a manifest an operator reads before installing, ``"node": "n7"``
    tells them nothing — and they are the person deciding whether to run this code.
    """
    mapping: Dict[str, str] = {}
    used: Set[str] = set()
    for node in sorted(graph_doc.get("graph", {}).get("nodes", []),
                       key=lambda n: str(n.get("id"))):
        old = str(node.get("id"))
        title = (titles or {}).get(old) or str(
            (node.get("params") or {}).get("__title__") or "")
        base = slugify(title) if title else slugify(
            str(node.get("op_key", "")).split(".")[-1], fallback="node")
        slug = base
        n = 2
        while slug in used:
            slug = f"{base}-{n}"
            n += 1
        used.add(slug)
        mapping[old] = slug
    return mapping


def rename_nodes(graph_doc: Mapping[str, Any],
                 mapping: Mapping[str, str]) -> Dict[str, Any]:
    """``graph_doc`` with every node id remapped, everywhere it appears.

    One atomic pass over nodes, edge endpoints, zone bodies and group bodies. Anything
    missed would produce a graph that still parses and no longer runs — an edge pointing at
    a node id that does not exist — so the caller should assert the node and edge counts
    afterwards rather than trust this.
    """
    def new(old: Any) -> Any:
        return mapping.get(str(old), old)

    doc = json.loads(json.dumps(dict(graph_doc)))       # deep copy of plain JSON
    graph = doc.get("graph") or {}
    for node in graph.get("nodes") or []:
        node["id"] = new(node.get("id"))
    for edge in graph.get("edges") or []:
        edge["src"] = new(edge.get("src"))
        edge["dst"] = new(edge.get("dst"))
    for zone in doc.get("zones") or []:
        for key in ("in_id", "out_id"):
            if key in zone:
                zone[key] = new(zone[key])
        if isinstance(zone.get("body"), list):
            zone["body"] = [new(b) for b in zone["body"]]
    for group in doc.get("groups") or []:
        for key in ("input_id", "output_id"):
            if key in group:
                group[key] = new(group[key])
        if isinstance(group.get("body"), Mapping):
            group["body"] = rename_nodes(group["body"], mapping).get("graph",
                                                                    group["body"])
    # The editor's own UI block is keyed by node id too. Dropped rather than remapped: a
    # published recipe's graph has no canvas, and the two shipped recipes carry no `ui` key.
    doc.pop("ui", None)
    return doc


# ── assembling a draft from a document ──────────────────────────────────────────

def draft_from_document(doc: Any, *, name: str, target: str = "",
                        take_suggested: bool = True) -> RecipeDraft:
    """Build a :class:`RecipeDraft` from a live :class:`~nodelab_v2.document.GraphDocument`.

    The graph is serialized through the same ``nodegraph.serialize`` the editor writes, then
    its node ids are made readable — so the published pair is a normal nd2graph document
    that this build can read straight back.
    """
    from nodegraph.serialize import to_dict
    from nodelab_v2.ops import LOAD_OP, ensure_ops

    ensure_ops()
    graph = doc.to_graph()
    titles = {nid: str(rec.params.get("__title__") or "")
              for nid, rec in doc.nodes.items()}
    raw = to_dict(graph, zones=getattr(doc, "_zones", ()) or (),
                  groups=getattr(doc, "_groups", ()) or ())
    mapping = readable_ids(raw, titles)
    graph_doc = rename_nodes(raw, mapping)

    # Re-read the renamed graph so candidates bind to the ids the manifest will use.
    from nodegraph.serialize import from_dict
    renamed_graph, zones, _groups = from_dict(graph_doc)
    _ensure_dim_levers(renamed_graph, graph_doc)

    loaders = tuple(nid for nid, rec in sorted(renamed_graph.nodes.items())
                    if rec.op_key == LOAD_OP)
    picks = candidates(renamed_graph)
    chosen: List[KnobDraft] = []
    if take_suggested:
        wanted = [c for c in picks if c.suggested or c.kind == "mode"]
        names = assign_names(wanted)
        for cand in wanted:
            lo, hi = suggest_bounds(cand)
            chosen.append(KnobDraft(candidate=cand, name=names[cand.key],
                                    minimum=lo, maximum=hi,
                                    derive=cand.can_derive))

    return RecipeDraft(
        name=name, title=name.replace("-", " ").strip().capitalize(),
        target=mapping.get(target, target) or _guess_target(renamed_graph),
        knobs=chosen, allow_zones=bool(zones), graph_doc=graph_doc, loaders=loaders,
        requires_metadata=tuple(derive_required_metadata(renamed_graph)),
        conditional_metadata=tuple(conditional_metadata(renamed_graph)))


def page_reads_other_pages(doc: Any) -> bool:
    """Whether a page's document holds a Page Input — i.e. it cannot run on its own and must
    be published with the pages it reads (:func:`draft_from_workspace`)."""
    from nodelab_v2.ops import PAGE_INPUT_OP
    return any(getattr(rec, "op_key", "") == PAGE_INPUT_OP
               for rec in (getattr(doc, "nodes", None) or {}).values())


def draft_from_workspace(ws: Any, page_id: str, *, name: str, target: str = "",
                         take_suggested: bool = True) -> RecipeDraft:
    """A :class:`RecipeDraft` for one PAGE of a workspace that reads other pages (V4.00).

    The page is published FLATTENED: :meth:`~nodelab_v2.workspace.Workspace.compose` splices
    it together with every page it reads into ONE run graph (groups expanded, muted nodes
    bypassed, every Dock kept live — a remote machine has none of this one's checkpoints),
    which is serialised as an ordinary single graph (format 2.0) with readable node ids. That
    is the only graph format a LabLink hub installs, and the worker runs it exactly as the
    editor composes the page. A workspace file named with ``"page"`` still runs on the worker
    directly; a hub does not accept one.
    """
    from nodegraph.serialize import from_dict, to_dict
    from nodelab_v2.ops import DOCK_OP, LOAD_OP, ensure_ops
    from nodelab_v2.workspace import qualify

    ensure_ops()
    order = ws.dependency_closure(page_id, strict=False)
    docks = frozenset(qualify(pid, nid) for pid in order
                      for nid, rec in ws.pages[pid].doc.nodes.items() if rec.op_key == DOCK_OP)
    composed = ws.compose(page_id, live_docks=docks)
    titles = {qualify(pid, nid): str(rec.params.get("__title__") or "")
              for pid in order for nid, rec in ws.pages[pid].doc.nodes.items()}
    raw = to_dict(composed.graph)
    mapping = readable_ids(raw, titles)
    graph_doc = rename_nodes(raw, mapping)
    renamed_graph, zones, _groups = from_dict(graph_doc)
    _ensure_dim_levers(renamed_graph, graph_doc)

    loaders = tuple(nid for nid, rec in sorted(renamed_graph.nodes.items())
                    if rec.op_key == LOAD_OP)
    picks = candidates(renamed_graph)
    chosen: List[KnobDraft] = []
    if take_suggested:
        wanted = [c for c in picks if c.suggested or c.kind == "mode"]
        names = assign_names(wanted)
        for cand in wanted:
            lo, hi = suggest_bounds(cand)
            chosen.append(KnobDraft(candidate=cand, name=names[cand.key],
                                    minimum=lo, maximum=hi, derive=cand.can_derive))
    guess = target or _guess_target(composed.graph, prefix=f"{page_id}/")
    return RecipeDraft(
        name=name, title=name.replace("-", " ").strip().capitalize(),
        target=mapping.get(guess, guess), knobs=chosen, allow_zones=bool(zones),
        graph_doc=graph_doc, loaders=loaders,
        requires_metadata=tuple(derive_required_metadata(renamed_graph)),
        conditional_metadata=tuple(conditional_metadata(renamed_graph)))


def _ensure_dim_levers(graph: Any, graph_doc: Dict[str, Any]) -> None:
    """Pin every 2D/3D lever explicitly in the published graph.

    A node with a lever left unset runs its default on whatever arrives, so a z-stack is
    processed plane by plane and the result looks entirely plausible. Tier 2 refuses a
    recipe that does this; writing the editor's own resolved value is both the fix and what
    the author already meant.
    """
    from nodegraph.registry import NODES

    by_id = {str(n.get("id")): n for n in (graph_doc.get("graph") or {}).get("nodes") or []}
    for node_id, rec in graph.nodes.items():
        spec = NODES.get(rec.op_key)
        lever = spec.dim_lever() if spec is not None else None
        if lever is None:
            continue
        if lever.name in rec.modes:
            continue
        value = lever.resolved_default()
        rec.modes[lever.name] = value
        node = by_id.get(str(node_id))
        if node is not None:
            node.setdefault("modes", {})[lever.name] = value


def _guess_target(graph: Any, prefix: str = "") -> str:
    """The node a run should pull, when the caller did not say: the last in topological
    order, which is the graph's own answer to "what is this pipeline for". ``prefix``
    (``"<page>/"``) keeps the guess on the published page of a composed workspace."""
    order = list(graph.topo_order())
    if prefix and any(str(n).startswith(prefix) for n in order):
        order = [n for n in order if str(n).startswith(prefix)]
    # a plot and its Export Figure make a picture, not a table: a draft whose last node is
    # one would run with an empty 'measurements' table — target what feeds them instead
    fig = [n for n in order if str(getattr(graph.nodes[n], "op_key", "")).startswith("plot.")
           or getattr(graph.nodes[n], "op_key", "") == "io.write_figure"]
    rest = [n for n in order if n not in fig]
    if rest:
        return rest[-1]
    if fig:
        # only figures (a page that plots what another page measured): what feeds the first
        feeds = [e.src for e in getattr(graph, "edges", ()) if e.dst == fig[0]]
        return feeds[0] if feeds else fig[-1]
    return ""


#: A socket's unit -> the calibration key needed to make sense of a number in it. This is
#: the big one, and it applies whether or not the graph pins the value: ``min_area`` is "in
#: µm², converted to pixels with ``pixel_size_um`` from the file's metadata", so a pinned
#: 50 µm² still cannot be turned into pixels without knowing µm/px. A recipe with a single
#: micron-denominated knob therefore derives from ``pixel_size_um``, and without it the run
#: does not fail — it measures in the wrong units and reports nothing.
_UNIT_NEEDS = {
    "um": ("pixel_size_um",),
    "um2": ("pixel_size_um",),
    "um3": ("pixel_size_um", "z_step_um"),
    "um_axial": ("z_step_um",),
}


def derive_required_metadata(graph: Any) -> List[str]:
    """The source metadata this graph actually derives from.

    Measured from the catalogue rather than hypothesised, from two independent sources:

    * **units.** Any active socket denominated in microns needs the pixel size to become
      pixels — pinned or not. A volume additionally needs the z step, and an axial length
      needs it instead.
    * **derive expressions.** A socket left unpinned resolves through its own expression,
      which names the calibration keys it reads (``bit_depth``, for a threshold). A *pinned*
      socket does not consult it, so that one does not count.

    Declaring more than the graph needs makes perfectly good files be refused; declaring
    less lets a job finish with different numbers and no warning. Only keys a sidecar can
    actually supply are returned — a requirement nothing can satisfy is a recipe that can
    never run.
    """
    from nodegraph.registry import NODES

    needed: Set[str] = set()
    requirable = set(P.REQUIRABLE_METADATA)
    for rec in graph.nodes.values():
        spec = NODES.get(rec.op_key)
        if spec is None:
            continue
        state = _mode_state(rec, spec)
        three_d = str(state.get("dim", "")) == "3D"
        for sock in spec.active_inputs(state):
            unit = str(getattr(sock, "unit", "") or "")
            for key in _UNIT_NEEDS.get(unit, ()):
                # A 2D run of a micron-denominated param needs the lateral pixel size only;
                # the axial spacing is what a 3D kernel additionally has to know.
                if key == "z_step_um" and unit != "um_axial" and not three_d:
                    continue
                needed.add(key)
            expression = str(getattr(sock, "derive", "") or "")
            if not expression or sock.name in rec.params:
                continue        # pinned in the graph: the file's value is not consulted
            for key in requirable:
                if re.search(rf"\b{re.escape(key)}\b", expression):
                    needed.add(key)
    return sorted(k for k in needed if k in requirable)


def conditional_metadata(graph: Any) -> List[str]:
    """Metadata a *reachable* configuration would need, beyond what the defaults need.

    ``requires.metadata`` is static, and that is a real limit of the format: a recipe whose
    2D default path needs only the pixel size still needs the z step the moment a caller
    turns its lever to 3D. Declaring it anyway would be worse than saying nothing — a single
    plane legitimately has no spacing, so every honest 2D job on 2D data would be refused
    for a field that does not apply to it.

    So it is reported to the author instead of enforced on the caller. The two are not
    interchangeable, and a warning is the only accurate thing the format can carry here.
    """
    from nodegraph.registry import NODES

    baseline = set(derive_required_metadata(graph))
    reachable: Set[str] = set()
    requirable = set(P.REQUIRABLE_METADATA)
    for rec in graph.nodes.values():
        spec = NODES.get(rec.op_key)
        if spec is None:
            continue
        state = _mode_state(rec, spec)
        for sock, _live in _offerable_sockets(spec, state):
            unit = str(getattr(sock, "unit", "") or "")
            for key in _UNIT_NEEDS.get(unit, ()):
                reachable.add(key)
        lever = spec.dim_lever()
        if lever is not None and "3D" in (lever.choices or ()):
            # A 3D kernel is anisotropic, so it has to know the axial spacing as well.
            for sock, _live in _offerable_sockets(spec, state):
                if str(getattr(sock, "unit", "") or "") in ("um", "um3"):
                    reachable.add("z_step_um")
    return sorted(k for k in (reachable - baseline) if k in requirable)


# ── writing and checking ────────────────────────────────────────────────────────

def write_recipe(draft: RecipeDraft, parent_dir: str) -> str:
    """Write ``<parent_dir>/<name>/`` with its manifest and graph. Returns the directory.

    The directory name is the manifest's ``name`` and not a parameter: tier 1 requires the
    two to agree, because a caller asks by name and the hub finds it by directory.
    """
    directory = os.path.join(parent_dir, draft.name)
    os.makedirs(directory, exist_ok=True)
    _write_json(os.path.join(directory, MANIFEST_FILENAME), draft.to_manifest())
    _write_json(os.path.join(directory, GRAPH_FILENAME), draft.graph_doc)
    return directory


def _write_json(path: str, doc: Any) -> None:
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def check_recipe(directory: str) -> Tuple[bool, List[str]]:
    """Validate a written recipe — tier 1 locally plus the four tier-2 catalogue checks.

    Delegates to the worker, which is where the catalogue lives and therefore the only place
    tier 2 can run before submission.
    """
    from nodelab_v2.lablink.worker import check_recipe_dir
    return check_recipe_dir(directory)


def derived_from(parent_manifest: Mapping[str, Any], parent_graph: Mapping[str, Any],
                 *, name: str, knobs: Mapping[str, Any],
                 title: str = "", description: str = "") -> Dict[str, Any]:
    """A manifest for a **derived recipe**: the parent's pipeline, new defaults.

    This is what promoting a tuned preset produces. The graph is the parent's, unchanged —
    knobs bind to node ids, so a derived recipe that altered the graph would not be a
    variant of anything. Only the published defaults move.

    A knob whose new value is ``None`` goes back to ``unset_means: "derive"`` where the
    parent allowed it, because that is what a null meant when it was tuned; pinning the null
    as a default is not expressible and would silently become "0" to a reader.
    """
    manifest = json.loads(json.dumps(dict(parent_manifest)))
    manifest["name"] = name
    manifest["title"] = title or f"{parent_manifest.get('title') or name} (variant)"
    if description:
        manifest["description"] = description
    manifest["graph"] = GRAPH_FILENAME

    for knob in manifest.get("knobs") or []:
        knob_name = str(knob.get("name"))
        if knob_name not in knobs:
            continue
        value = knobs[knob_name]
        if value is None:
            knob.pop("default", None)
            # Only where the parent already allowed derivation: adding `derive` to a param
            # the graph pins is refused outright, and would mean the recipe ignored the
            # file's own calibration with nothing reporting it.
            if knob.get("unset_means") == "derive":
                pass
            continue
        knob.pop("unset_means", None)
        knob["default"] = value
    return manifest


__all__ = [
    "RECIPE_FORMAT", "GRAPH_FILENAME", "MANIFEST_FILENAME",
    "KnobCandidate", "KnobDraft", "RecipeDraft",
    "candidates", "suggest_bounds", "readable_ids", "rename_nodes",
    "draft_from_document", "draft_from_workspace", "page_reads_other_pages",
    "derive_required_metadata", "write_recipe", "check_recipe",
    "derived_from", "slugify",
]
