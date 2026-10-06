"""A page LINKED to a master page (V4.00 step 6): the master's nodes and wiring, its own values.

A linked page is how one workflow is tuned per position or per condition without losing the
master: **Duplicate as linked** makes a page whose graph IS the master's — every node, wire,
frame and position follows the master as it is edited — while its parameter and mode values
are its own wherever it OVERRIDES them. Edit a value here and it becomes an override (the
inspector marks it, *Reset to master* drops it); edit the same value on the master and every
linked page that does not override it follows.

**The graph's shape belongs to the master.** Adding, removing or rewiring nodes, muting, frames,
groups, zones — every structural edit — is refused here with :class:`LinkedPageError`, whose
message the window shows; moving and folding cards are forwarded to the master (the cards move
on both). **Make unique** turns the page into a plain one holding its current values.

The mirror is rebuilt IN PLACE: a node keeps its :class:`NodeRecord` object — and its ``params``
and ``modes`` dicts — for as long as the master keeps that node with the same type, so the
canvas updates its cards rather than rebuilding them on every master keystroke, and the
inspector keeps editing the dicts it holds.

Qt-free, like :mod:`nodelab_v2.document`: the selftest drives it headless.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

from nodelab_v2.document import LOCKED_KEY, FrameRecord, GraphDocument, NodeRecord
from nodelab_v2.ops import BAKE_KEY, DOCK_OP

#: what a structural edit on a linked page is refused with (the status bar shows it)
TOPOLOGY_HINT = ("this page is linked to its master — add, remove or rewire nodes on the "
                 "master (every linked page follows), or Make unique to edit this page's "
                 "graph on its own")

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


def check_overrides(d: Any, *, where: str = "overrides") -> None:
    """Refuse an overrides mapping of the wrong SHAPE (a hand-edited or damaged file) before
    anything is built from it: ``{node: {"params": {..}, "modes": {name: str}, "local":
    {"params": {..}, "modes": {..}}}}``."""
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping of node id -> values")
    for nid, entry in d.items():
        if not isinstance(nid, str) or not isinstance(entry, dict):
            raise ValueError(f"{where}: {nid!r} must map to a mapping")
        extra = set(entry) - {"params", "modes", "local"}
        if extra:
            raise ValueError(f"{where}: {nid!r} has unknown keys {sorted(extra)}")
        for part in (entry, entry.get("local") or {}):
            if not isinstance(part, dict):
                raise ValueError(f"{where}: {nid!r} 'local' must be a mapping")
            params, modes = part.get("params") or {}, part.get("modes") or {}
            if not isinstance(params, dict) or not isinstance(modes, dict):
                raise ValueError(f"{where}: {nid!r} params and modes must be mappings")
            if not all(isinstance(k, str) for k in params) or \
                    not all(isinstance(k, str) and isinstance(v, str) for k, v in modes.items()):
                raise ValueError(f"{where}: {nid!r} has a non-text name or mode value")


class LinkedPageError(ValueError):
    """A structural edit on a linked page — make it on the master, or Make unique."""


class LinkedDocument(GraphDocument):
    """A document that mirrors ``master`` and keeps per-node ``overrides``:
    ``{node_id: {"params": {name: value | UNSET}, "modes": {name: value}}}``."""

    editable_topology = False

    def __init__(self, master: GraphDocument,
                 overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> None:
        super().__init__()
        # one level deep: a page linked to a LINKED page links to that page's master
        while isinstance(master, LinkedDocument):
            master = master.master
        self.master: GraphDocument = master
        check_overrides(overrides or {})
        self.overrides: Dict[str, Dict[str, Dict[str, Any]]] = copy.deepcopy(overrides or {})
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
        """Rebuild this page from the master, in place, then lay the overrides over it."""
        m = self.master
        kept: Dict[str, NodeRecord] = {}
        for nid, mrec in m.nodes.items():
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
            rec.muted, rec.collapsed = mrec.muted, mrec.collapsed
            kept[nid] = rec
        for nid in [n for n in self.overrides if n not in kept]:
            del self.overrides[nid]                   # its node is gone from the master
        ms = self.master.meta_seeds
        for nid in kept:
            if (self.overrides.get(nid) or {}).get("params"):
                continue                              # its own values: its own envelope
            if nid in ms:
                self.meta_seeds[nid] = ms[nid]
            else:
                self.meta_seeds.pop(nid, None)
        for nid in [n for n in self.meta_seeds if n not in kept]:
            del self.meta_seeds[nid]
        self.nodes.clear()
        self.nodes.update(kept)
        self.edges = list(m.edges)
        self.frames = {fid: FrameRecord(fr.id, fr.title, fr.members, fr.color)
                       for fid, fr in m.frames.items()}
        self._zones = copy.deepcopy(m._zones)
        self._groups = copy.deepcopy(m._groups)
        self._back_edges = list(m._back_edges)

    def _on_master_change(self) -> None:
        if not self._attached:
            return
        self._mirror()
        # the master's own touched set names the same node ids here
        self._notify(self.master.last_touched)

    def _on_master_moved(self, node_id: str) -> None:
        mrec = self.master.nodes.get(node_id)
        if self._attached and mrec is not None:
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
        and only *Reset to master* hands it back."""
        rec, mrec = self.nodes.get(node_id), self.master.nodes.get(node_id)
        if rec is None or mrec is None:
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
        if entry:
            self.overrides[node_id] = entry
        else:
            self.overrides.pop(node_id, None)

    def touch(self, node_id: Optional[str] = None) -> None:
        for nid in ([node_id] if node_id is not None else list(self.nodes)):
            self._capture(nid)
        super().touch(node_id)

    def is_overridden(self, node_id: str, name: Optional[str] = None) -> bool:
        """Does this page override ``node_id`` (or that node's param/mode ``name``)?"""
        ov = self.overrides.get(node_id)
        if not ov:
            return False
        if name is None:
            return bool(ov.get("params") or ov.get("modes"))
        return name in (ov.get("params") or {}) or name in (ov.get("modes") or {})

    def master_value(self, node_id: str, name: str) -> Any:
        """The master's value of ``node_id``'s param or mode ``name`` (``None`` if unset)."""
        mrec = self.master.nodes.get(node_id)
        if mrec is None:
            return None
        return mrec.params.get(name, mrec.modes.get(name))

    def override_count(self) -> int:
        return sum(len(ov.get("params") or {}) + len(ov.get("modes") or {})
                   for ov in self.overrides.values())

    def reset_override(self, node_id: str, name: Optional[str] = None) -> None:
        """Back to the master's value — one param/mode, or every value of the node."""
        ov = self.overrides.get(node_id)
        if not ov:
            return
        if name is None:
            ov["params"], ov["modes"] = {}, {}
        else:
            (ov.get("params") or {}).pop(name, None)
            (ov.get("modes") or {}).pop(name, None)
        if not ov.get("params") and not ov.get("modes") and not ov.get("local"):
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

    # ── what follows the master, and what is refused ───────────────────────────
    def set_pos(self, node_id: str, x: float, y: float) -> None:
        """A card moves on the master — the layout is the master's (and this page follows
        it through :meth:`_on_master_moved`)."""
        self.master.set_pos(node_id, x, y)
        super().set_pos(node_id, x, y)

    def set_collapsed(self, node_id: str, collapsed: bool) -> None:
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

    def can_connect(self, src: str, src_socket: str, dst: str, dst_socket: str):
        return False, TOPOLOGY_HINT

    def _refuse(self, *_a, **_k):
        raise LinkedPageError(TOPOLOGY_HINT)

    add_node = remove_node = connect = disconnect = _refuse
    add_frame = remove_frame = rename_frame = _refuse
    set_muted = set_iterate_target = wrap_repeat_zone = make_group = ungroup = _refuse
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
           "DOCK_LOCAL_PARAMS", "DOCK_LOCAL_MODES", "check_overrides"]
