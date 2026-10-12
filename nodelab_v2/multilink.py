"""Mirror and merge: link several nodes in one gesture (2026-10-12). Qt-free.

Every card carries a checkbox in its header. Ticking cards puts them in the canvas's
**multi-link set**, and a wire then drawn OUT of a ticked node applies to the whole set, in
whichever of two modes the canvas is in (the button beside the action pill):

* **merge**: every ticked node is wired INTO the node the wire reaches. The node dragged
  from takes the target's primary (first) input; the others follow in alphabetical order
  of their titles, one input each. A multi-input socket takes them all, and slots that
  appear as they fill (a Timeseries Builder's files, a Viewer's sources) are filled in
  order. A target that cannot take every ticked node is refused and nothing changes. A
  node with only one input therefore always refuses (:func:`merge`).
* **mirror**: every other ticked node gets its OWN copy of the node the wire reaches, the
  same type with the same values, wired the same way and placed at the same offset from it.
  The copies (with the original) form a **mirror group**: while the canvas is in mirror
  mode, a parameter or mode edit on any of them applies to all of them
  (:func:`mirror_siblings`, called from :meth:`GraphDocument.touch`, the one call every
  editor makes after a value edit).

**Lanes: mirroring continues down the chain** (2026-10-12). Each ticked node starts a
LANE, and a mirror group maps every lane to its own member (``{lane: node}``): the copy
made for ticked node ``C`` sits in lane ``C``. A wire drawn out of ANY member of a mirror
group, while the page is in mirror mode, mirrors the same way the ticked set does
(:func:`lanes_for`). To a fresh node, every other lane gets its own copy, wired from that
lane's member, and the new nodes form a group with the same lanes, so a chain of steps
placed after the first mirrored one is built once per lane. To a node that is itself the
SAME lane's member of another group, the wire is drawn in every lane between that lane's
two members, and no node is made. A wire from one lane into another lane's member has no
per-lane meaning and stays one wire.

The state is per DOCUMENT (one page) and per session: the ticks, the mode and the mirror
groups are not saved with the file. Every canvas showing the page sees the same set. A
tick is not a graph edit, so it never re-propagates metadata or cancels a pull. Listeners
(the page's scenes) repaint on :meth:`MultiLinkState.on_change`.

A wire drawn from a node that is neither ticked (with another ticked) nor a mirror-group
member in mirror mode, or from an input socket, behaves exactly as it always did.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

from nodelab_v2.document import TITLE_KEY
from nodelab_v2.ops import PAGE_NAME_KEY

MIRROR = "mirror"
MERGE = "merge"
MODES = (MIRROR, MERGE)

#: params a mirror copy keeps its OWN value of. A Page Output's variable name is unique
#: across the workspace (a copy is renamed ``name2``), and a title names one particular card,
#: so copying either onto a sibling would fight the uniqueness rule or mislabel the copy.
_PER_NODE_PARAMS = (PAGE_NAME_KEY, TITLE_KEY)

EdgeTuple = Tuple[str, str, str, str]


class MultiLinkState:
    """The multi-link set of one document: the ticked node ids (in the order ticked), the
    mode, and the mirror groups. Create it through :func:`state`, never directly."""

    def __init__(self) -> None:
        self.ticked: Dict[str, None] = {}          # an insertion-ordered set
        self.mode: str = MIRROR
        #: the mirror groups, each ``{lane: node id}`` — a lane is named by the ticked node
        #: it started from, so two groups made from one ticked set line up lane by lane
        self.groups: List[Dict[str, str]] = []
        self._listeners: List[Callable[[], None]] = []

    def on_change(self, cb: Callable[[], None]) -> None:
        if cb not in self._listeners:
            self._listeners.append(cb)

    def off_change(self, cb: Callable[[], None]) -> None:
        if cb in self._listeners:
            self._listeners.remove(cb)

    def emit(self) -> None:
        for cb in list(self._listeners):
            cb()


def state(doc) -> MultiLinkState:
    """``doc``'s multi-link state, made on first use."""
    st = doc.__dict__.get("_multilink")
    if st is None:
        st = MultiLinkState()
        doc.__dict__["_multilink"] = st
    return st


def ticked(doc) -> List[str]:
    """The ticked nodes still in ``doc``, in the order they were ticked. A deleted node
    drops out here rather than lingering as a ghost member of the set."""
    st = state(doc)
    gone = [n for n in st.ticked if n not in doc.nodes]
    for n in gone:
        del st.ticked[n]
    return list(st.ticked)


def is_ticked(doc, node_id: str) -> bool:
    return node_id in state(doc).ticked and node_id in doc.nodes


def set_ticked(doc, node_id: str, on: bool) -> None:
    st = state(doc)
    if on and node_id in doc.nodes and node_id not in st.ticked:
        st.ticked[node_id] = None
        st.emit()
    elif not on and node_id in st.ticked:
        del st.ticked[node_id]
        st.emit()


def toggle(doc, node_id: str) -> bool:
    """Tick or untick ``node_id``; returns whether it is ticked now."""
    on = not is_ticked(doc, node_id)
    set_ticked(doc, node_id, on)
    return on


def clear(doc) -> int:
    """Untick every node (the canvas's *Clear selected*); returns how many were ticked."""
    st = state(doc)
    n = len(ticked(doc))
    if st.ticked:
        st.ticked.clear()
        st.emit()
    return n


def mode(doc) -> str:
    return state(doc).mode


def set_mode(doc, value: str) -> None:
    st = state(doc)
    if value not in MODES:
        raise ValueError(f"multi-link mode must be one of {MODES}, got {value!r}")
    if value != st.mode:
        st.mode = value
        st.emit()


def active(doc) -> bool:
    """Is the page in a multi-link mode: at least one node ticked? What the canvas border
    and the mode button's emphasis show."""
    return bool(ticked(doc))


def lanes_of(doc, node_id: str) -> Dict[str, str]:
    """``{lane: node}`` of the mirror group ``node_id`` is in (its live members only), or
    ``{}`` — whatever the mode."""
    for g in state(doc).groups:
        if node_id in g.values():
            return {lane: n for lane, n in g.items() if n in doc.nodes}
    return {}


def lane_in(lanes: Dict[str, str], node_id: str) -> Optional[str]:
    return next((lane for lane, n in lanes.items() if n == node_id), None)


def lanes_for(doc, src_id: str) -> Optional[Dict[str, str]]:
    """The lanes a wire drawn out of ``src_id`` is repeated across, or ``None`` for a plain
    wire:

    * ``src_id`` is ticked and another node is: every ticked node is a lane, its own member;
    * else, in MIRROR mode (mirror selected, something ticked), ``src_id`` is a member of a
      mirror group with another live member: that group's lanes. This is what lets a chain
      placed after a mirrored node be mirrored too, step by step."""
    t = ticked(doc)
    if src_id in t and len(t) >= 2:
        return {n: n for n in t}
    if mode(doc) == MIRROR and t:
        g = lanes_of(doc, src_id)
        if len(g) >= 2:
            return g
    return None


def applies(doc, src_id: str) -> bool:
    """Does a wire drawn out of ``src_id`` go through the multi-link mode? When
    :func:`lanes_for` names lanes for it: a ticked node with another ticked, or — mirror only
    — a member of a mirror group. With nothing to repeat it across, the plain wire is the
    right answer. MERGE acts on the ticked set alone (:func:`merge`)."""
    lanes = lanes_for(doc, src_id)
    if lanes is None:
        return False
    return mode(doc) == MIRROR or src_id in ticked(doc)


class Outcome(NamedTuple):
    """What a merge or a mirror did: ``ok``, the sentence for the status bar, the wires it
    added, and the nodes it created."""
    ok: bool
    message: str
    edges: List[EdgeTuple]
    nodes: List[str]


def _by_title(doc, ids) -> List[str]:
    return sorted(ids, key=lambda n: (doc.title_of(n).lower(), n))


def _inputs(doc, node_id: str) -> list:
    """Every input of ``node_id`` in card order, INCLUDING the growing slots the card is
    not showing yet: :meth:`GraphDocument.can_connect` accepts a hidden slot (the card then
    reveals it and every slot before it), so a merge can count and fill them."""
    rec = doc.nodes.get(node_id)
    spec = rec.spec() if rec is not None else None
    if spec is None:
        return list(doc.input_specs(node_id))
    return list(spec.active_inputs(rec.state()))


def _output_for(doc, src: str, preferred: str, dst: str, slot: str) -> Optional[str]:
    """The output of ``src`` to wire into ``dst.slot``: the socket the gesture was drawn
    from when ``src`` has one of that name and it fits, else the first output that fits."""
    names = [s.name for s in doc.output_specs(src)]
    order = ([preferred] if preferred in names else []) + [n for n in names if n != preferred]
    for name in order:
        if doc.can_connect(src, name, dst, slot)[0]:
            return name
    return None


def merge_plan(doc, src: str, src_socket: str, dst: str
               ) -> Tuple[List[Tuple[str, str, str]], str]:
    """``([(feeder, its output, dst input), ...], problem)`` — how :func:`merge` would wire
    every ticked node into ``dst``: ``src`` first, into the first input it fits (the
    primary); the other ticked nodes in alphabetical order of their titles, each into the
    next input it fits. A multi-input socket takes any number. ``problem`` is a sentence,
    and the plan empty, when ``dst`` cannot take them all."""
    feeders = [src] + _by_title(doc, [n for n in ticked(doc) if n not in (src, dst)])
    slots = _inputs(doc, dst)
    name = doc.title_of(dst)
    plan: List[Tuple[str, str, str]] = []
    used: Set[str] = set()
    for f in feeders:
        hit = None
        for s in slots:
            if s.name in used and not getattr(s, "multi", False):
                continue
            out = _output_for(doc, f, src_socket, dst, s.name)
            if out is not None:
                hit = (f, out, s.name)
                if not getattr(s, "multi", False):
                    used.add(s.name)
                break
        if hit is None:
            fits = sum(1 for s in slots
                       if any(_output_for(doc, g, src_socket, dst, s.name) for g in feeders))
            if fits <= 1 and not any(getattr(s, "multi", False) for s in slots):
                return [], (f"Merge refused: “{name}” accepts only one input, and "
                            f"{len(feeders)} nodes are ticked. Merge into a node that takes "
                            f"several (a multi-input socket, or slots that grow as they "
                            f"fill), or untick nodes.")
            return [], (f"Merge refused: “{name}” has room for {len(plan)} of the "
                        f"{len(feeders)} ticked nodes — “{doc.title_of(f)}” fits none of "
                        f"its remaining inputs. Untick nodes, or merge into a node with "
                        f"more inputs.")
        plan.append(hit)
    return plan, ""


def merge(doc, src: str, src_socket: str, dst: str) -> Outcome:
    """Wire every ticked node into ``dst`` (:func:`merge_plan`), all or nothing. An input
    that already had a wire gets the merged one instead, as a single drop would. A failure
    part-way (a connect the plan did not foresee) puts every wire back as it was."""
    plan, problem = merge_plan(doc, src, src_socket, dst)
    if problem:
        return Outcome(False, problem, [], [])
    added: List[EdgeTuple] = []
    removed: List[EdgeTuple] = []
    try:
        for f, out, slot in plan:
            removed.extend(doc.connect(f, out, dst, slot))
            added.append((f, out, dst, slot))
    except ValueError as exc:
        for e in reversed(added):
            doc.disconnect(*e)
        for e in removed:
            if doc.can_connect(*e)[0]:
                doc.connect(*e)
        return Outcome(False, f"Merge refused part-way and undone: {exc}", [], [])
    pairs = ", ".join(f"{doc.title_of(f)} → {slot}" for f, _o, slot in plan)
    return Outcome(True, f"merged {len(plan)} nodes into “{doc.title_of(dst)}”: {pairs}",
                   added, [])


def mirror(doc, src: str, src_socket: str, dst: str, dst_socket: str) -> Outcome:
    """Repeat the wire ``src.src_socket`` → ``dst.dst_socket`` (the caller has drawn it) in
    every other lane of ``src`` (:func:`lanes_for`: the ticked set, or ``src``'s mirror
    group), lane by lane:

    * ``dst`` is the SAME lane's member of a mirror group: each other lane's member of
      ``src``'s group is wired into its own lane's member of ``dst``'s group — the
      connection is mirrored, no node is made. A lane ``dst``'s group lacks gets a copy;
    * ``dst`` is a member of a group in ANOTHER lane: the wire crosses lanes, has no per-lane
      meaning, and stays one wire;
    * otherwise every other lane gets its own copy of ``dst`` — its type, values, mode and
      switches — at ``dst``'s offset from ``src`` measured from that lane's member, wired the
      same way. ``dst``'s OTHER incoming wires come along: from a member of a mirror group,
      that lane's member of it (a parallel chain stays parallel); from anything else, the
      same node (a shared reference stays shared). ``dst`` and its copies form a new mirror
      group with the same lanes, which is what lets the NEXT node placed after it mirror too.

    A lane whose member has no output that fits gets nothing, and the message says so."""
    lanes = lanes_for(doc, src) or {}
    l0 = lane_in(lanes, src)
    if l0 is None:
        return Outcome(False, "nothing to mirror", [], [])
    target = lanes_of(doc, dst)
    t_lane = lane_in(target, dst)
    if target and t_lane != l0:
        return Outcome(False, f"“{doc.title_of(src)}” → “{doc.title_of(dst)}” crosses mirror "
                              f"lanes, so it was drawn once and not mirrored", [], [])
    others = [(lane, n) for lane, n in lanes.items()
              if lane != l0 and n in doc.nodes and n != dst]
    x = doc.nodes[dst]
    s = doc.nodes[src]
    dx, dy = x.x - s.x, x.y - s.y
    incoming = [e for e in doc.edges if e[2] == dst and e[3] != dst_socket]
    made: List[str] = []
    linked: List[str] = []
    added: List[EdgeTuple] = []
    skipped: List[str] = []
    group: Dict[str, str] = dict(target) if target else {l0: dst}
    for lane, c in others:
        twin = target.get(lane) if target else None
        if twin is not None:
            # a connection between two parallel groups: draw it in this lane too
            out = _output_for(doc, c, src_socket, twin, dst_socket)
            if out is None:
                skipped.append(c)
                continue
            doc.connect(c, out, twin, dst_socket)
            added.append((c, out, twin, dst_socket))
            linked.append(twin)
            continue
        crec = doc.nodes[c]
        rec = doc.add_node(x.op_key, x=crec.x + dx, y=crec.y + dy,
                           params=copy.deepcopy(dict(x.params)), modes=dict(x.modes))
        rec.muted, rec.collapsed = x.muted, x.collapsed
        out = _output_for(doc, c, src_socket, rec.id, dst_socket)
        if out is None:
            doc.remove_node(rec.id)
            skipped.append(c)
            continue
        doc.connect(c, out, rec.id, dst_socket)
        added.append((c, out, rec.id, dst_socket))
        for (u, us, _d, ds) in incoming:
            uu = lanes_of(doc, u).get(lane, u)          # parallel stays parallel
            if uu != rec.id and doc.can_connect(uu, us, rec.id, ds)[0]:
                doc.connect(uu, us, rec.id, ds)
                added.append((uu, us, rec.id, ds))
        group[lane] = rec.id
        made.append(rec.id)
    if made:
        _set_group(doc, group)
    name = doc.title_of(dst)
    parts = []
    if made:
        parts.append(f"mirrored “{name}” onto {len(made)} more node"
                     f"{'' if len(made) == 1 else 's'}")
    if linked:
        parts.append(f"drew the wire into “{name}” in {len(linked)} more lane"
                     f"{'' if len(linked) == 1 else 's'}")
    msg = "; ".join(parts) if parts else \
        f"nothing to mirror: no other lane has an output that fits “{name}”"
    if skipped and parts:
        msg += f"; no output fits for {', '.join(doc.title_of(n) for n in skipped)}"
    return Outcome(bool(made or linked), msg, added, made)


def _set_group(doc, lanes: Dict[str, str]) -> None:
    """Record ``lanes`` (``{lane: node}``) as one mirror group. A node is in one group at
    most: the group already holding any of these nodes is replaced by this one, which is a
    superset of it (a group gains lanes, never loses them here)."""
    st = state(doc)
    members = set(lanes.values())
    st.groups = [g for g in st.groups if not (set(g.values()) & members)] + [dict(lanes)]


def group_of(doc, node_id: str) -> List[str]:
    """``node_id``'s mirror group (itself first, then in lane order), or ``[]`` — whatever
    the mode."""
    lanes = lanes_of(doc, node_id)
    if node_id not in lanes.values():
        return []
    return [node_id] + [n for n in lanes.values() if n != node_id]


def mirror_siblings(doc, node_id: str) -> List[str]:
    """The nodes an edit to ``node_id`` also applies to: its mirror group, but only while
    the page is IN mirror mode — mirror selected and at least one node ticked. Outside it
    the copies are ordinary nodes and can be tuned one by one."""
    if mode(doc) != MIRROR or not active(doc):
        return []
    return group_of(doc, node_id)[1:]


def copy_values(doc, src_id: str, dst_id: str) -> bool:
    """Make ``dst_id``'s params and modes ``src_id``'s (same node type only), keeping the
    params a copy owns (:data:`_PER_NODE_PARAMS`). The dicts are updated in place: they
    are the live dicts the card and the inspector read."""
    a, b = doc.nodes.get(src_id), doc.nodes.get(dst_id)
    if a is None or b is None or a.op_key != b.op_key:
        return False
    own = {k: b.params[k] for k in _PER_NODE_PARAMS if k in b.params}
    b.params.clear()
    b.params.update(copy.deepcopy(dict(a.params)))
    for k in _PER_NODE_PARAMS:
        b.params.pop(k, None)
    b.params.update(own)
    b.modes.clear()
    b.modes.update(dict(a.modes))
    return True


__all__ = ["MIRROR", "MERGE", "MODES", "MultiLinkState", "Outcome", "state", "ticked",
           "is_ticked", "set_ticked", "toggle", "clear", "mode", "set_mode", "active",
           "lanes_of", "lane_in", "lanes_for", "applies", "merge_plan", "merge", "mirror",
           "group_of", "mirror_siblings", "copy_values"]
