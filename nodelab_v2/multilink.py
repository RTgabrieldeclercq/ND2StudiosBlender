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

The state is per DOCUMENT (one page) and per session: the ticks, the mode and the mirror
groups are not saved with the file. Every canvas showing the page sees the same set. A
tick is not a graph edit, so it never re-propagates metadata or cancels a pull. Listeners
(the page's scenes) repaint on :meth:`MultiLinkState.on_change`.

A wire drawn from a node that is NOT ticked, from a node that is the only one ticked, or
from an input socket, behaves exactly as it always did.
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
        self.groups: List[Set[str]] = []
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


def applies(doc, src_id: str) -> bool:
    """Does a wire drawn out of ``src_id`` go through the multi-link mode? Only when that
    node is ticked AND at least one other is: with nothing else ticked, there is nothing to
    mirror or merge, and the plain wire is the right answer."""
    t = ticked(doc)
    return src_id in t and len(t) >= 2


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
    """Give every OTHER ticked node its own copy of ``dst``, wired from it the way ``src``
    is wired into ``dst`` (``src_socket`` → ``dst_socket``; the caller has made that first
    wire). A copy has ``dst``'s type, values, mode and switches, sits at ``dst``'s offset from
    ``src`` measured from its own feeder, and takes ``dst``'s OTHER incoming wires too (a
    shared reference stays shared). A ticked node with no output that fits gets no copy,
    and the message says so. ``dst`` and its copies form one mirror group."""
    others = _by_title(doc, [n for n in ticked(doc) if n not in (src, dst)])
    x = doc.nodes[dst]
    s = doc.nodes[src]
    dx, dy = x.x - s.x, x.y - s.y
    incoming = [e for e in doc.edges if e[2] == dst and e[3] != dst_socket]
    made: List[str] = []
    added: List[EdgeTuple] = []
    skipped: List[str] = []
    for c in others:
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
            if u != rec.id and doc.can_connect(u, us, rec.id, ds)[0]:
                doc.connect(u, us, rec.id, ds)
                added.append((u, us, rec.id, ds))
        made.append(rec.id)
    if made:
        _join_group(doc, [dst] + made)
    msg = (f"mirrored “{doc.title_of(dst)}” onto {len(made)} more node"
           f"{'' if len(made) == 1 else 's'}" if made else
           f"nothing to mirror: no other ticked node has an output that fits “{doc.title_of(dst)}”")
    if skipped and made:
        msg += f"; no output fits for {', '.join(doc.title_of(n) for n in skipped)}"
    return Outcome(bool(made), msg, added, made)


def _join_group(doc, ids: List[str]) -> None:
    """Put ``ids`` in one mirror group, absorbing any group one of them is already in."""
    st = state(doc)
    joined = set(ids)
    keep = []
    for g in st.groups:
        if g & joined:
            joined |= g
        else:
            keep.append(g)
    st.groups = keep + [joined]


def group_of(doc, node_id: str) -> List[str]:
    """``node_id``'s mirror group (itself first), or ``[]`` — whatever the mode."""
    for g in state(doc).groups:
        if node_id in g:
            return [node_id] + sorted(n for n in g if n != node_id and n in doc.nodes)
    return []


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
           "applies", "merge_plan", "merge", "mirror", "group_of", "mirror_siblings",
           "copy_values"]
