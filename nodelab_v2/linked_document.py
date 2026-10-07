"""A page LINKED to a master page (V4.00 step 6): the master's nodes and wiring, its own values.

A linked page is how one workflow is tuned per position or per condition without losing the
master: **Duplicate as linked** makes a page whose graph IS the master's — every node, wire,
frame and position follows the master as it is edited — while its parameter and mode values
are its own wherever it OVERRIDES them. Edit a value here and it becomes an override (the
inspector marks it, *Reset to master* drops it); edit the same value on the master and every
linked page that does not override it follows. Whether a node is switched ON or OFF (muted,
its input passed through) is a value of the same kind (V4.00 step 11e): this page's own when
it differs from the master's.

**The graph's shape belongs to the master — until the user says otherwise (V4.00 step 11e).**
A structural edit — adding, removing or rewiring nodes — is refused here with
:class:`LinkedPageError` while the page's :attr:`~LinkedDocument.edit_mode` is unset; the
window asks the user first, and their answer sets it:

* **Make unique** — the page becomes a plain one holding its current graph and values
  (:meth:`~LinkedDocument.make_unique`; the Workspace swaps the document).
* **Modified** (:data:`EDIT_MODIFIED`, saved with the page) — the edit stays on THIS page: its
  own nodes, the master's nodes it removed, the wires it added and removed are recorded as
  the page's STRUCTURE (:meth:`~LinkedDocument.structure_dict`) and laid over the master on
  every mirror, so the master's other edits keep arriving. Where the two disagree this page
  wins: its own wire into a single input replaces whatever the master wires there later.
* **Send to the master** (:data:`EDIT_MASTER`, this session only) — the edit is made ON THE
  MASTER, in a form that changes nothing the master computes: a node this page adds arrives
  there switched off (so on every other linked page too) and is switched on here; wiring it in
  or out goes to the master as well. An edit that would change the master's result — removing
  or rewiring a node that is ON there — is refused, as is adding a node that cannot be switched
  off (one that changes the kind of data, :func:`~nodelab_v2.document.pass_through_reason`).

Frames, groups and zones stay the master's in every mode. Moving and folding a card the
master owns is forwarded to the master (the cards move on both); this page's own cards move
alone.

The mirror is rebuilt IN PLACE: a node keeps its :class:`NodeRecord` object — and its ``params``
and ``modes`` dicts — for as long as the master keeps that node with the same type, so the
canvas updates its cards rather than rebuilding them on every master keystroke, and the
inspector keeps editing the dicts it holds.

Qt-free, like :mod:`nodelab_v2.document`: the selftest drives it headless.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from nodegraph.registry import NODES
from nodegraph.sockets import SocketType
from nodelab_v2.document import (
    LOCKED_KEY, FrameRecord, GraphDocument, NodeRecord, is_driver_edge, pass_through_reason)
from nodelab_v2.ops import BAKE_KEY, DOCK_OP

#: what a structural edit on a linked page is refused with (the status bar shows it)
TOPOLOGY_HINT = ("this page is linked to its master — add, remove or rewire nodes on the "
                 "master (every linked page follows), or Make unique to edit this page's "
                 "graph on its own")

#: the answers to a structural edit on a linked page (V4.00 step 11e) — see the module text
EDIT_UNIQUE = "unique"
EDIT_MODIFIED = "modified"
EDIT_MASTER = "master"

#: refused on a page that sends its edits to the master: the edit would change its result
MASTER_CHANGE_HINT = ("this edit would change what the master computes — a page that sends "
                      "its edits to the master may only add, wire or remove nodes that are "
                      "switched off there. Keep the change on this page (a modified linked "
                      "page) or make the page unique")

#: what a structural entry of the canvas menu says on a linked page whose edits are not
#: settled yet — choosing it asks (V4.00 step 11e)
ASK_HINT = ("this page is linked to its master — you will be asked whether to make it "
            "unique, keep the change on this page, or add it to the master switched off")

#: frames, groups and zones are the master's in every mode
SHAPE_HINT = ("frames, groups and zones of a linked page are its master's — change them on "
              "the master, or make this page unique")

#: an override that REMOVES a parameter the master sets: the linked page uses the node's
#: default (or its metadata-derived value) where the master pins one. A plain JSON value, so
#: the file can hold it.
UNSET = {"__unset__": True}

_MISSING = object()

#: A Dock's CHECKPOINT is the page's own, never the master's: its folder, its bake record and
#: whether it serves it (``state``: live / held / docked). Mirrored, a linked page would serve
#: — and re-bake into — the master's folder. They are kept per page under an override's
#: ``"local"`` entry, which is not an override (not counted, not marked, not reset).
DOCK_LOCAL_PARAMS = ("store", BAKE_KEY)
DOCK_LOCAL_MODES = ("state",)

#: the keys of a page's own STRUCTURE (V4.00 step 11e)
_STRUCTURE_KEYS = ("nodes", "removed", "edges_added", "edges_removed")

Edge = Tuple[str, str, str, str]


def check_overrides(d: Any, *, where: str = "overrides") -> None:
    """Refuse an overrides mapping of the wrong SHAPE (a hand-edited or damaged file) before
    anything is built from it: ``{node: {"params": {..}, "modes": {name: str}, "muted": bool,
    "local": {"params": {..}, "modes": {..}}}}``."""
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping of node id -> values")
    for nid, entry in d.items():
        if not isinstance(nid, str) or not isinstance(entry, dict):
            raise ValueError(f"{where}: {nid!r} must map to a mapping")
        extra = set(entry) - {"params", "modes", "local", "muted"}
        if extra:
            raise ValueError(f"{where}: {nid!r} has unknown keys {sorted(extra)}")
        if "muted" in entry and not isinstance(entry["muted"], bool):
            raise ValueError(f"{where}: {nid!r} 'muted' must be true or false")
        for part in (entry, entry.get("local") or {}):
            if not isinstance(part, dict):
                raise ValueError(f"{where}: {nid!r} 'local' must be a mapping")
            params, modes = part.get("params") or {}, part.get("modes") or {}
            if not isinstance(params, dict) or not isinstance(modes, dict):
                raise ValueError(f"{where}: {nid!r} params and modes must be mappings")
            if not all(isinstance(k, str) for k in params) or \
                    not all(isinstance(k, str) and isinstance(v, str) for k, v in modes.items()):
                raise ValueError(f"{where}: {nid!r} has a non-text name or mode value")


def check_structure(d: Any, *, where: str = "structure") -> None:
    """Refuse a linked page's STRUCTURE record of the wrong shape (V4.00 step 11e):
    ``{"nodes": {id: {"op_key": str, "params": {..}, "modes": {..}, "x", "y", "muted",
    "collapsed"}}, "removed": [id], "edges_added": [[src, socket, dst, socket]],
    "edges_removed": [[...]]}`` — every key optional."""
    if d is None:
        return
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping")
    extra = set(d) - set(_STRUCTURE_KEYS)
    if extra:
        raise ValueError(f"{where} has unknown keys {sorted(extra)}")
    nodes = d.get("nodes", {})
    nodes = {} if nodes is None else nodes
    if not isinstance(nodes, dict):
        raise ValueError(f"{where}: 'nodes' must be a mapping of node id -> node")
    for nid, rec in nodes.items():
        if not isinstance(nid, str) or not nid or "/" in nid:
            raise ValueError(f"{where}: bad node id {nid!r}")
        if not isinstance(rec, dict) or not isinstance(rec.get("op_key"), str):
            raise ValueError(f"{where}: node {nid!r} needs an op_key")
        if not isinstance(rec.get("params") or {}, dict) or \
                not isinstance(rec.get("modes") or {}, dict):
            raise ValueError(f"{where}: node {nid!r} params and modes must be mappings")
    removed = d.get("removed", [])
    removed = [] if removed is None else removed
    if not isinstance(removed, list) or not all(isinstance(n, str) for n in removed):
        raise ValueError(f"{where}: 'removed' must be a list of node ids")
    for key in ("edges_added", "edges_removed"):
        edges = d.get(key, [])
        edges = [] if edges is None else edges
        if not isinstance(edges, list) or not all(
                isinstance(e, (list, tuple)) and len(e) == 4
                and all(isinstance(x, str) for x in e) for e in edges):
            raise ValueError(f"{where}: {key!r} must be a list of [src, socket, dst, socket]")


def _rec_dict(rec: NodeRecord) -> Dict[str, Any]:
    return {"op_key": rec.op_key, "params": copy.deepcopy(rec.params),
            "modes": dict(rec.modes), "x": rec.x, "y": rec.y,
            "muted": rec.muted, "collapsed": rec.collapsed}


def _rec_from(nid: str, d: Dict[str, Any]) -> NodeRecord:
    return NodeRecord(nid, str(d["op_key"]), params=copy.deepcopy(d.get("params") or {}),
                      modes=dict(d.get("modes") or {}), x=float(d.get("x", 0.0) or 0.0),
                      y=float(d.get("y", 0.0) or 0.0), muted=bool(d.get("muted", False)),
                      collapsed=bool(d.get("collapsed", False)))


class LinkedPageError(ValueError):
    """A structural edit on a linked page — make it on the master, or Make unique."""


class LinkedDocument(GraphDocument):
    """A document that mirrors ``master`` and keeps per-node ``overrides``:
    ``{node_id: {"params": {name: value | UNSET}, "modes": {name: value}, "muted": bool}}``
    — and, once modified (V4.00 step 11e), a STRUCTURE of its own laid over the master."""

    editable_topology = False

    def __init__(self, master: GraphDocument,
                 overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
                 structure: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        # one level deep: a page linked to a LINKED page links to that page's master
        while isinstance(master, LinkedDocument):
            master = master.master
        self.master: GraphDocument = master
        check_overrides(overrides or {})
        check_structure(structure)
        self.overrides: Dict[str, Dict[str, Dict[str, Any]]] = copy.deepcopy(overrides or {})
        # this page's own STRUCTURE (V4.00 step 11e): its own nodes, the master's nodes it
        # removed, the wires it added and removed. `_modified` is the saved half of the edit
        # mode; `_session_mode` the unsaved one (sending edits to the master).
        self._modified = structure is not None
        self._session_mode = ""
        st = structure or {}
        self._own: Dict[str, NodeRecord] = {
            nid: _rec_from(nid, rec) for nid, rec in (st.get("nodes") or {}).items()}
        self._removed: List[str] = list(st.get("removed") or [])
        self._edges_added: List[Edge] = [tuple(e) for e in st.get("edges_added") or []]
        self._edges_removed: List[Edge] = [tuple(e) for e in st.get("edges_removed") or []]
        # The page's OWN seeds (a Load's file envelope): a node it does not override takes the
        # master's (`_mirror`), one it does — another file — keeps what its own read gave.
        # Shared by reference, a linked page's file would re-describe the master.
        self.meta_seeds = dict(master.meta_seeds)
        self.path = master.path
        #: the master PAGE's name, for the inspector's banner — the Workspace sets it (a
        #: document does not know the pages it sits in)
        self.master_name: Callable[[], str] = lambda: ""
        self._attached = False
        self._mirror()                       # first: a mirror that fails leaves no listener
        self.attach()
        self.propagate()

    # ── the mirror ─────────────────────────────────────────────────────────────
    def _mirror(self) -> None:
        """Rebuild this page from the master, in place, then lay the overrides — and this
        page's own structure — over it."""
        m = self.master
        # the master may have taken an id one of this page's own nodes carries
        for nid in [n for n in self._own if n in m.nodes]:
            self._reid_own(nid)
        removed = set(self._removed)
        kept: Dict[str, NodeRecord] = {}
        for nid, mrec in m.nodes.items():
            if nid in removed:
                continue
            rec = self.nodes.get(nid)
            if rec is None or rec.op_key != mrec.op_key:
                if rec is not None:                  # re-typed on the master: its values go
                    self.overrides.pop(nid, None)
                rec = NodeRecord(nid, mrec.op_key)
            rec.params.clear()
            rec.params.update(copy.deepcopy(mrec.params))
            rec.modes.clear()
            rec.modes.update(mrec.modes)
            ov = self.overrides.get(nid)
            if ov:
                own = ov.get("params") or {}
                for k, v in own.items():
                    if v == UNSET:
                        rec.params.pop(k, None)
                    else:
                        rec.params[k] = copy.deepcopy(v)
                rec.modes.update(ov.get("modes") or {})
                # the pins follow: the master's, minus what this page resets to the default,
                # plus every value it sets itself (an inspector edit is a pinned value)
                rec.set_locked((rec.locked - {k for k, v in own.items() if v == UNSET})
                               | {k for k, v in own.items() if v != UNSET})
            if mrec.op_key == DOCK_OP:
                # the checkpoint is this page's own (DOCK_LOCAL_*), never the master's
                for k in DOCK_LOCAL_PARAMS:
                    rec.params.pop(k, None)
                for k in DOCK_LOCAL_MODES:
                    rec.modes.pop(k, None)
                loc = (ov or {}).get("local") or {}
                rec.params.update(copy.deepcopy(loc.get("params") or {}))
                rec.modes.update(loc.get("modes") or {})
            rec.x, rec.y = mrec.x, mrec.y
            # on or off is this page's own where it says so (V4.00 step 11e)
            mo = (ov or {}).get("muted")
            rec.muted = mo if isinstance(mo, bool) else mrec.muted
            rec.collapsed = mrec.collapsed
            kept[nid] = rec
        kept.update(self._own)
        for nid in [n for n in self.overrides if n not in kept or n in self._own]:
            del self.overrides[nid]                   # its node is gone from the master
        ms = self.master.meta_seeds
        for nid in kept:
            if nid in self._own or (self.overrides.get(nid) or {}).get("params"):
                continue                              # its own values: its own envelope
            if nid in ms:
                self.meta_seeds[nid] = ms[nid]
            else:
                self.meta_seeds.pop(nid, None)
        for nid in [n for n in self.meta_seeds if n not in kept]:
            del self.meta_seeds[nid]
        self.nodes.clear()
        self.nodes.update(kept)
        medges = [tuple(e) for e in m.edges]
        if self._modified:
            # what this page changed, pruned to what still means something: a wire to a node
            # the master deleted goes, a removal the master already made is no longer one
            mset = set(medges)
            self._removed = [n for n in self._removed if n in m.nodes]
            self._edges_added = [e for e in self._edges_added
                                 if e[0] in kept and e[2] in kept and e not in mset]
            self._edges_removed = [e for e in self._edges_removed if e in mset]
            drop = set(self._edges_removed)
            own = list(self._edges_added)
            # this page's own wire into a SINGLE input wins over one the master adds later
            singles = {(e[2], e[3]) for e in own if not self._is_multi(e[2], e[3])}
            base = [e for e in medges if e[0] in kept and e[2] in kept and e not in drop
                    and (e[2], e[3]) not in singles]
            self.edges = base + own
        else:
            self.edges = medges
        self.frames = {}
        for fid, fr in m.frames.items():
            members = [n for n in fr.members if n in kept]
            if members:
                self.frames[fid] = FrameRecord(fr.id, fr.title, members, fr.color)
        self._zones = copy.deepcopy(m._zones)
        self._groups = copy.deepcopy(m._groups)
        self._back_edges = list(m._back_edges)

    def _is_multi(self, dst: str, dst_socket: str) -> bool:
        spec = self._socket_spec(dst, "in", dst_socket)
        return bool(spec is not None and spec.multi)

    def _reid_own(self, old: str) -> None:
        """Give this page's own node ``old`` a fresh id — the master has just taken it."""
        new = self.new_id()
        rec = self._own.pop(old)
        self.nodes.pop(old, None)
        rec.id = new
        self._own[new] = rec
        self._edges_added = [tuple(new if (i in (0, 2) and x == old) else x
                                   for i, x in enumerate(e)) for e in self._edges_added]
        if old in self.meta_seeds:
            self.meta_seeds[new] = self.meta_seeds.pop(old)

    def _on_master_change(self) -> None:
        if not self._attached:
            return
        self._mirror()
        # the master's own touched set names the same node ids here
        self._notify(self.master.last_touched)

    def _on_master_moved(self, node_id: str) -> None:
        mrec = self.master.nodes.get(node_id)
        if self._attached and mrec is not None and node_id in self.nodes:
            GraphDocument.set_pos(self, node_id, mrec.x, mrec.y)   # this page's canvas follows

    def attach(self) -> None:
        """Follow the master (again — a load that fails re-attaches the pages it detached)."""
        if not self._attached:
            self._attached = True
            self.master.on_change(self._on_master_change)
            self.master.on_moved(self._on_master_moved)
            self._mirror()

    def detach(self) -> None:
        """Stop following the master (the page is removed or made unique)."""
        if self._attached:
            self._attached = False
            self.master.off_change(self._on_master_change)
            self.master.off_moved(self._on_master_moved)

    # ── overrides ──────────────────────────────────────────────────────────────
    def _capture(self, node_id: str) -> None:
        """Record ``node_id``'s values that are this page's own — what an edit made. A value
        that differs from the master's becomes an override; one that is ALREADY an override
        stays one, even when the master later reaches the same value — the user set it here,
        and only *Reset to master* hands it back. Whether the node is on or off is kept as
        it was (:meth:`set_muted` records that)."""
        rec, mrec = self.nodes.get(node_id), self.master.nodes.get(node_id)
        if rec is None or mrec is None or node_id in self._own:
            return
        # the pin list is not an override: it is derived from the values (see `_mirror`);
        # nor is a Dock's checkpoint, which is the page's own ("local")
        dock = rec.op_key == DOCK_OP
        skip_p = {LOCKED_KEY, *(DOCK_LOCAL_PARAMS if dock else ())}
        skip_m = set(DOCK_LOCAL_MODES if dock else ())
        old = self.overrides.get(node_id) or {}
        old_p, old_m = old.get("params") or {}, old.get("modes") or {}
        params = {k: copy.deepcopy(v) for k, v in rec.params.items()
                  if k not in skip_p and (k in old_p or mrec.params.get(k, _MISSING) != v)}
        params.update({k: dict(UNSET) for k in mrec.params
                       if k not in skip_p and k not in rec.params})
        modes = {k: v for k, v in rec.modes.items()
                 if k not in skip_m and (k in old_m or mrec.modes.get(k, _MISSING) != v)}
        entry: Dict[str, Any] = {}
        if params or modes:
            entry = {"params": params, "modes": modes}
        if dock:
            lp = {k: copy.deepcopy(rec.params[k]) for k in DOCK_LOCAL_PARAMS if k in rec.params}
            lm = {k: rec.modes[k] for k in DOCK_LOCAL_MODES if k in rec.modes}
            if lp or lm:
                entry = entry or {"params": {}, "modes": {}}
                entry["local"] = {"params": lp, "modes": lm}
        if isinstance(old.get("muted"), bool):
            entry["muted"] = old["muted"]
        if entry:
            self.overrides[node_id] = entry
        else:
            self.overrides.pop(node_id, None)

    def touch(self, node_id: Optional[str] = None) -> None:
        for nid in ([node_id] if node_id is not None else list(self.nodes)):
            if node_id is not None:
                self._settle_output_name(nid)        # before it is recorded as an override
            self._capture(nid)
        super().touch(node_id)

    def is_overridden(self, node_id: str, name: Optional[str] = None) -> bool:
        """Does this page override ``node_id`` (or that node's param/mode ``name``, or
        ``"muted"`` — whether it is switched on)?"""
        ov = self.overrides.get(node_id)
        if not ov:
            return False
        if name is None:
            return bool(ov.get("params") or ov.get("modes") or "muted" in ov)
        if name == "muted":
            return "muted" in ov
        return name in (ov.get("params") or {}) or name in (ov.get("modes") or {})

    def master_value(self, node_id: str, name: str) -> Any:
        """The master's value of ``node_id``'s param or mode ``name`` (``None`` if unset)."""
        mrec = self.master.nodes.get(node_id)
        if mrec is None:
            return None
        if name == "muted":
            return mrec.muted
        return mrec.params.get(name, mrec.modes.get(name))

    def override_count(self) -> int:
        return sum(len(ov.get("params") or {}) + len(ov.get("modes") or {})
                   + (1 if "muted" in ov else 0)
                   for ov in self.overrides.values())

    def reset_override(self, node_id: str, name: Optional[str] = None) -> None:
        """Back to the master's value — one param/mode (or ``"muted"``), or every value of
        the node."""
        ov = self.overrides.get(node_id)
        if not ov:
            return
        if name is None:
            ov["params"], ov["modes"] = {}, {}
            ov.pop("muted", None)
        elif name == "muted":
            ov.pop("muted", None)
        else:
            (ov.get("params") or {}).pop(name, None)
            (ov.get("modes") or {}).pop(name, None)
        if not ov.get("params") and not ov.get("modes") and not ov.get("local") \
                and "muted" not in ov:
            self.overrides.pop(node_id, None)
        self._mirror()
        self._notify((node_id,))

    def overrides_dict(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return copy.deepcopy(self.overrides)

    def load_overrides(self, d: Optional[Dict[str, Any]]) -> None:
        check_overrides(dict(d or {}))
        self.overrides = copy.deepcopy(dict(d or {}))
        self._mirror()
        self._notify()

    # ── on / off (V4.00 step 11e) ──────────────────────────────────────────────
    def set_muted(self, node_id: str, muted: bool) -> None:
        """Switch a node off (or on) ON THIS PAGE: an override like a value — the master and
        every other linked page keep theirs; *Reset to master* hands it back. This page's own
        node is simply switched. Switching off is refused for a node that changes the kind of
        data, exactly as on a plain page."""
        if node_id in self._own:
            GraphDocument.set_muted(self, node_id, muted)
            return
        rec, mrec = self.nodes.get(node_id), self.master.nodes.get(node_id)
        if rec is None or mrec is None or rec.muted == bool(muted):
            return
        if muted:
            why = self.pass_through_reason(node_id)
            if why:
                raise ValueError(f"{self.title_of(node_id)} cannot be switched off: {why}")
        ov = self.overrides.setdefault(node_id, {})
        if bool(muted) == bool(mrec.muted):
            ov.pop("muted", None)
            if not ov.get("params") and not ov.get("modes") and not ov.get("local"):
                self.overrides.pop(node_id, None)
        else:
            ov["muted"] = bool(muted)
        rec.muted = bool(muted)
        self._notify((node_id,))

    # ── the edit mode (V4.00 step 11e) ─────────────────────────────────────────
    @property
    def edit_mode(self) -> str:
        """``""`` (a structural edit is refused — the window asks first), :data:`EDIT_MODIFIED`
        (kept on this page, saved) or :data:`EDIT_MASTER` (sent to the master, this session)."""
        return EDIT_MODIFIED if self._modified else self._session_mode

    def set_edit_mode(self, mode: str) -> None:
        """Answer the question a structural edit asks. :data:`EDIT_MODIFIED` is for good (the
        page keeps its own structure; :meth:`revert_structure` drops it); :data:`EDIT_MASTER`
        lasts this session and is refused on a modified page; ``""`` asks again."""
        if mode == EDIT_MODIFIED:
            if not self._modified:
                self._modified, self._session_mode = True, ""
                self._notify(())
        elif mode == EDIT_MASTER:
            if self._modified:
                raise LinkedPageError("this page keeps its changes on itself (a modified "
                                      "linked page) — its edits do not go to the master")
            if self._session_mode != EDIT_MASTER:
                self._session_mode = EDIT_MASTER
                self._notify(())
        elif mode == "":
            if self._session_mode:
                self._session_mode = ""
                self._notify(())
        else:
            raise ValueError(f"unknown edit mode {mode!r}")

    @property
    def is_modified(self) -> bool:
        return self._modified

    def own_node_ids(self) -> List[str]:
        """This page's own nodes (a modified page's), in the order they were added."""
        return list(self._own)

    def structure_counts(self) -> Tuple[int, int, int]:
        """``(own nodes, master nodes removed, wires added or removed)``."""
        return (len(self._own), len(self._removed),
                len(self._edges_added) + len(self._edges_removed))

    def structure_dict(self) -> Optional[Dict[str, Any]]:
        """This page's own structure for the file — ``None`` unless the page is modified, so
        a linked page that never was stays byte-identical."""
        if not self._modified:
            return None
        return {"nodes": {nid: _rec_dict(rec) for nid, rec in self._own.items()},
                "removed": list(self._removed),
                "edges_added": [list(e) for e in self._edges_added],
                "edges_removed": [list(e) for e in self._edges_removed]}

    def revert_structure(self) -> None:
        """Drop this page's own nodes and wiring: back to following the master's graph."""
        if not self._modified:
            return
        self._own.clear()
        self._removed, self._edges_added, self._edges_removed = [], [], []
        self._modified = False
        self._mirror()
        self._notify(None)

    def _record(self) -> None:
        """After an edit on a MODIFIED page: what of the live graph is this page's own."""
        m = self.master
        self._own = {nid: rec for nid, rec in self.nodes.items() if nid not in m.nodes}
        self._removed = [nid for nid in m.nodes if nid not in self.nodes]
        medges = {tuple(e) for e in m.edges}
        live = {tuple(e) for e in self.edges}
        self._edges_added = [tuple(e) for e in self.edges if tuple(e) not in medges]
        self._edges_removed = [tuple(e) for e in m.edges if tuple(e) not in live
                               and e[0] in self.nodes and e[2] in self.nodes]

    def new_id(self, prefix: str = "n") -> str:
        """An id neither this page nor its master uses, marked as this page's own (``nL1``)
        — the master numbers its own nodes ``n1``, ``n2``…, so it never takes one."""
        taken = set(self.nodes) | set(self.master.nodes)
        i = 1
        while f"{prefix}L{i}" in taken:
            i += 1
        return f"{prefix}L{i}"

    # ── structural edits ───────────────────────────────────────────────────────
    def add_node(self, op_key: str, **kw) -> NodeRecord:
        mode = self.edit_mode
        if mode == EDIT_MODIFIED:
            rec = GraphDocument.add_node(self, op_key, **kw)
            self._record()
            return rec
        if mode == EDIT_MASTER:
            return self._push_add(op_key, **kw)
        raise LinkedPageError(TOPOLOGY_HINT)

    def remove_node(self, node_id: str) -> None:
        mode = self.edit_mode
        if mode == EDIT_MODIFIED:
            if any(node_id in z.members for z in self._zones):
                raise LinkedPageError(SHAPE_HINT)
            GraphDocument.remove_node(self, node_id)
            self._record()
            return
        if mode == EDIT_MASTER:
            self._push_remove(node_id)
            return
        raise LinkedPageError(TOPOLOGY_HINT)

    def can_connect(self, src: str, src_socket: str, dst: str, dst_socket: str):
        mode = self.edit_mode
        if mode == EDIT_MODIFIED:
            return GraphDocument.can_connect(self, src, src_socket, dst, dst_socket)
        if mode == EDIT_MASTER:
            # whether the wire is VALID; whether the master would compute the same is asked
            # by `connect` itself, with every wire made so far in place (a node is wired in
            # one wire at a time, and only the finished splice leaves the master unchanged)
            return self.master.can_connect(src, src_socket, dst, dst_socket)
        return False, TOPOLOGY_HINT

    def connect(self, src: str, src_socket: str, dst: str, dst_socket: str):
        mode = self.edit_mode
        if mode == EDIT_MODIFIED:
            removed = GraphDocument.connect(self, src, src_socket, dst, dst_socket)
            self._record()
            return removed
        if mode == EDIT_MASTER:
            ok, why = self.master.can_connect(src, src_socket, dst, dst_socket)
            if not ok:
                raise ValueError(f"cannot connect: {why}")
            if not self._keeps_master(self._connected(src, src_socket, dst, dst_socket)):
                raise LinkedPageError(MASTER_CHANGE_HINT)
            return self.master.connect(src, src_socket, dst, dst_socket)
        raise LinkedPageError(TOPOLOGY_HINT)

    def disconnect(self, src: str, src_socket: str, dst: str, dst_socket: str) -> None:
        mode = self.edit_mode
        e = (src, src_socket, dst, dst_socket)
        if mode == EDIT_MODIFIED:
            GraphDocument.disconnect(self, src, src_socket, dst, dst_socket)
            self._record()
            return
        if mode == EDIT_MASTER:
            after = [tuple(x) for x in self.master.edges if tuple(x) != e]
            if not self._keeps_master(after):
                raise LinkedPageError(MASTER_CHANGE_HINT)
            self.master.disconnect(src, src_socket, dst, dst_socket)
            return
        raise LinkedPageError(TOPOLOGY_HINT)

    # ── sending an edit to the master, switched off there (V4.00 step 11e) ─────
    def _master_effective(self, edges, muted, nodes) -> Tuple[frozenset, frozenset]:
        """What the master COMPUTES with these wires and these nodes switched off: the nodes
        that are on, and the wires once every switched-off node is passed through."""
        m = self.master
        eff = m._bypass_muted([tuple(e) for e in edges], muted=frozenset(muted))
        return frozenset(n for n in nodes if n not in muted), frozenset(tuple(e) for e in eff)

    def _keeps_master(self, edges_after, nodes_after=None) -> bool:
        m = self.master
        muted = {nid for nid, r in m.nodes.items() if r.muted}
        before = self._master_effective(m.edges, muted, set(m.nodes))
        after = self._master_effective(edges_after, muted,
                                       set(m.nodes) if nodes_after is None else nodes_after)
        return before == after

    def _connected(self, src: str, src_socket: str, dst: str, dst_socket: str) -> List[Edge]:
        """The master's wires as :meth:`GraphDocument.connect` would leave them."""
        m = self.master
        b = m._socket_spec(dst, "in", dst_socket)
        single = b is not None and not b.multi
        after = [tuple(e) for e in m.edges
                 if not (single and e[2] == dst and e[3] == dst_socket)]
        edge = (src, src_socket, dst, dst_socket)
        if edge not in after:
            after.append(edge)
        return after

    def _push_add(self, op_key: str, *, x: float = 0.0, y: float = 0.0,
                  node_id: Optional[str] = None, params: Optional[dict] = None,
                  modes: Optional[dict] = None) -> NodeRecord:
        spec = NODES.get(op_key)
        why = pass_through_reason(spec, params, modes)
        if why:
            label = spec.label if spec is not None else op_key
            raise LinkedPageError(
                f"{label} cannot go to the master switched off — {why}. Keep it on this page "
                f"(a modified linked page) or make the page unique")
        m = self.master
        nid = node_id or m.new_id()
        if nid in self.nodes or nid in m.nodes:
            raise ValueError(f"duplicate node id {nid!r}")
        # on HERE before the master hears of it, so this page never shows it off
        self.overrides.setdefault(nid, {})["muted"] = False
        try:
            m.add_node(op_key, x=x, y=y, node_id=nid, params=params, modes=modes)
            m.set_muted(nid, True)                   # off on the master and its other pages
        except Exception:
            if nid not in m.nodes:
                self.overrides.pop(nid, None)
            raise
        return self.nodes[nid]

    def _push_remove(self, node_id: str) -> None:
        m = self.master
        mrec = m.nodes.get(node_id)
        if mrec is None:
            return
        if mrec.muted:
            # off on the master: its chain already runs without it — remove it and heal the
            # chain, which is exactly what passing it through computed
            ds_ins = {s.name for s in m.input_specs(node_id) if s.type is SocketType.DATASET}
            feed = next((e for e in m.edges if e[2] == node_id and e[3] in ds_ins), None)
            outs = [tuple(e) for e in m.edges if e[0] == node_id and not is_driver_edge(m, e)]
            m.remove_node(node_id)
            if feed is not None:
                for _s, _ss, d, dsock in outs:
                    if d in m.nodes:
                        try:
                            m.connect(feed[0], feed[1], d, dsock)
                        except ValueError:
                            pass
            return
        after = [tuple(e) for e in m.edges if e[0] != node_id and e[2] != node_id]
        if not self._keeps_master(after, set(m.nodes) - {node_id}):
            raise LinkedPageError(MASTER_CHANGE_HINT)
        m.remove_node(node_id)

    # ── what follows the master, and what is refused ───────────────────────────
    def set_pos(self, node_id: str, x: float, y: float) -> None:
        """A card moves on the master — the layout is the master's (and this page follows
        it through :meth:`_on_master_moved`). This page's own card moves alone."""
        if node_id in self._own:
            GraphDocument.set_pos(self, node_id, x, y)
            return
        self.master.set_pos(node_id, x, y)
        super().set_pos(node_id, x, y)

    def set_collapsed(self, node_id: str, collapsed: bool) -> None:
        if node_id in self._own:
            GraphDocument.set_collapsed(self, node_id, collapsed)
            return
        self.master.set_collapsed(node_id, collapsed)

    def set_dock_state(self, node_id: str, docked: bool) -> None:
        super().set_dock_state(node_id, docked)
        self._capture(node_id)

    def set_dock_bake(self, node_id: str, **kw) -> None:
        super().set_dock_bake(node_id, **kw)
        self._capture(node_id)

    def set_dock_hold(self, node_id: str, held: bool) -> None:
        super().set_dock_hold(node_id, held)
        self._capture(node_id)

    def rebase_path(self, path: str) -> None:
        """Save As: a Dock's folder is re-anchored like a plain page's — and recorded, so
        the file and the next mirror keep the new one."""
        super().rebase_path(path)
        for nid in list(self.nodes):
            self._capture(nid)

    def _refuse(self, *_a, **_k):
        raise LinkedPageError(TOPOLOGY_HINT)

    def _refuse_shape(self, *_a, **_k):
        raise LinkedPageError(SHAPE_HINT if self.edit_mode else TOPOLOGY_HINT)

    add_frame = remove_frame = rename_frame = _refuse_shape
    set_iterate_target = wrap_repeat_zone = make_group = ungroup = _refuse_shape
    clear = load_dict = load_page = load_file = save_file = _refuse

    # ── leaving the master ─────────────────────────────────────────────────────
    def make_unique(self) -> GraphDocument:
        """A plain document holding this page's graph and values (``self`` stops following
        the master; the Workspace swaps the page's document)."""
        self.detach()
        doc = GraphDocument()
        # the path BEFORE loading (a dock resolves its folder against it); the seeds AFTER
        # (loading starts a document's seeds afresh); the Workspace re-propagates once the
        # page is attached, with its Page Input seeds installed
        doc.path = self.path
        doc.load_page(self.to_page_dict())
        doc.meta_seeds.update(self.meta_seeds)
        doc.set_held_nodes(self._held_nodes)
        return doc


__all__ = ["LinkedDocument", "LinkedPageError", "TOPOLOGY_HINT", "UNSET",
           "DOCK_LOCAL_PARAMS", "DOCK_LOCAL_MODES", "check_overrides", "check_structure",
           "EDIT_UNIQUE", "EDIT_MODIFIED", "EDIT_MASTER", "MASTER_CHANGE_HINT", "SHAPE_HINT",
           "ASK_HINT"]
