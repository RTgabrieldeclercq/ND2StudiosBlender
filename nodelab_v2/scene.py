"""The graph scene + view (NodeLab v2, Phase 5) — the interactive canvas (G1).

The scene is a **mirror of the GraphDocument**: `sync()` (wired to
``document.on_change``) reconciles node cards and rebuilds edge items from the model,
so the canvas can never drift from what will be saved/run. All wiring flows through
the document's validated API (``can_connect`` + cycle rejection + non-multi replace).

Interactive wiring (G1):

* press a socket → a dashed drag wire follows the mouse; hovering a socket rings it
  green (valid target) or red (invalid — type/direction/cycle);
* release on a valid socket → connect (a non-multi input's old wire is replaced);
* press a **connected** non-multi input → the existing wire detaches and re-drags from
  its source (Blender behavior);
* release over empty canvas → the **link-drag search** popup (type-filtered list of
  ops with a compatible socket); picking one places the node there and auto-connects;
* ``Delete`` removes selected nodes/wires; ``M`` toggles mute (G3 pass-through).

**Deleting** (three equivalent paths, all landing in :meth:`GraphScene.delete_selection`
/ :meth:`GraphScene.delete_nodes`): the ``Del``/``Backspace`` key, the ✕ badge that
appears when you hover a card, and the right-click context menu (which also offers
*Dissolve* — delete a node but reconnect the wire through it, so a mid-chain node can
leave without breaking the chain).

**Per-node progress**: the scene is the sink for the runner's plan/progress events
(:meth:`set_run_plan`, :meth:`on_node_progress`) and owns the single animation timer that
drives every indeterminate bar on the canvas.

The view accepts palette drags (``application/x-nd2studios-op``) and emits
``op_dropped`` for the window to place the node.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QTransform
from PySide6.QtWidgets import (
    QGraphicsPathItem, QGraphicsScene, QGraphicsView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QVBoxLayout, QWidget,
)

from nodegraph.registry import NODES
from nodegraph.sockets import SocketType, can_connect as _sock_can_connect
from nodelab_v2 import theme as T
from nodelab_v2.document import GraphDocument
from nodelab_v2.edge_item import EdgeItem, wire_path
from nodelab_v2.frame_item import FrameItem
from nodelab_v2.minimap import HudButton
from nodelab_v2.node_item import NodeItem, SocketItem
from nodelab_v2.ops import DOCK_OP, LOAD_OP, PRECISION_UNSET, bake_record

#: op prefixes hidden from the palette / link search (boundary + fixture ops, plus the
#: source loader ``io.load`` — it is created from File → Load ND2/TIFF file…, never dragged)
HIDDEN_OP_PREFIXES = ("zone.", "group.", "test.", "io.seed", "io.stream_seed",
                      "rr.", "eng.", "io.nd2", "io.load",
                      # minted by nodegraph.iterate's feedback rewrite between two clones,
                      # never placed by hand — it exists only inside an unrolled graph
                      "flow.advance")


def visible_specs():
    return [s for s in NODES.all()
            if not s.op_key.startswith(HIDDEN_OP_PREFIXES)]


def compatible_ops(fixed_spec, fixed_io: str) -> List[Tuple[object, str]]:
    """Ops (spec, socket_name) whose default-state sockets can pair with the fixed
    socket — the link-drag search menu (G1)."""
    fixed = fixed_spec.instantiate()
    out = []
    for spec in visible_specs():
        state = spec.default_state()
        pool = (spec.active_inputs(state) if fixed_io == "out"
                else spec.active_outputs(state))
        for s in pool:
            cand = s.instantiate()
            ok = (_sock_can_connect(fixed, cand) if fixed_io == "out"
                  else _sock_can_connect(cand, fixed))
            if ok:
                out.append((spec, s.name))
                break
    return out


class LinkSearchPopup(QWidget):
    """The link-drag search: a type-filtered, searchable op list at the drop point."""

    def __init__(self, parent, entries: List[Tuple[str, str, str]],
                 on_pick: Callable[[str, str], None]) -> None:
        # entries: (display label, op_key, socket_name)
        super().__init__(parent, Qt.Popup)
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; border:1px solid {T.BORDER.name()}; }}
            QLineEdit {{ background:{T.BODY.name()}; color:{T.INK.name()};
                border:1px solid {T.BORDER.name()}; border-radius:4px; padding:4px 6px; }}
            QListWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; border:0; }}
            QListWidget::item:selected {{ background:{T.ACCENT_DIM.name()};
                color:{T.INK.name()}; }}
        """)
        self._entries = entries
        self._on_pick = on_pick
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("Search nodes…")
        self._list = QListWidget()
        lay.addWidget(self._edit)
        lay.addWidget(self._list)
        self.setFixedSize(240, 300)
        self._edit.textChanged.connect(self._refill)
        self._edit.returnPressed.connect(self._accept_current)
        self._list.itemActivated.connect(lambda _i: self._accept_current())
        self._list.itemClicked.connect(lambda _i: self._accept_current())
        self._refill("")
        self._edit.setFocus()

    def _refill(self, text: str) -> None:
        self._list.clear()
        t = (text or "").lower()
        for label, op, sock in self._entries:
            if t and t not in label.lower() and t not in op.lower():
                continue
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, (op, sock))
            self._list.addItem(it)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _accept_current(self) -> None:
        it = self._list.currentItem()
        if it is not None:
            op, sock = it.data(Qt.UserRole)
            self.close()
            self._on_pick(op, sock)
        else:
            self.close()

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Down, Qt.Key_Up):
            self._list.keyPressEvent(e)
            return
        super().keyPressEvent(e)


#: animation tick for the indeterminate progress sweeps (ms). One timer for the whole
#: canvas, running only while at least one card is in an indeterminate running state.
PROGRESS_TICK_MS = 60


class GraphScene(QGraphicsScene):
    """Document-mirroring scene + the wire-drag state machine."""

    node_activated = Signal(str)      # double-clicked node id → view/pull it (G7)
    pull_requested = Signal(str)      # context menu → pull/view this node
    #: a node was removed through the canvas (badge / key / menu) — the window reports it
    nodes_deleted = Signal(object)    # [node_id, …]
    #: a card's ◎ glyph was clicked — a :class:`~nodelab_v2.picker.PickRequest`, forwarded
    #: to the window (which owns the viewer) exactly like the inspector's Pick button.
    pick_requested = Signal(object)
    #: a Dock node's context-menu entry was chosen: ``(node_id, action)``. Same signal
    #: shape (and same handler in the window) as the inspector's, so baking from the
    #: canvas and baking from the panel are literally one code path.
    dock_action = Signal(str, str)
    #: a source card's context-menu *Ingest this file* — the same thing double-clicking a
    #: not-yet-ingested source does, spelled out for discoverability.
    ingest_requested = Signal(str)

    def __init__(self, document: GraphDocument) -> None:
        super().__init__()
        self.doc = document
        self.node_items: Dict[str, NodeItem] = {}
        self.frame_items: Dict[str, FrameItem] = {}
        self.edge_items: List[EdgeItem] = []
        #: the node whose output the Viewer / mini-map is showing (marked on its card)
        self.viewed_id: Optional[str] = None
        self._drag_fixed: Optional[SocketItem] = None
        self._temp_wire: Optional[QGraphicsPathItem] = None
        self._hover_sock: Optional[SocketItem] = None
        #: run states survive a `sync()` (an edit mid-run must not blank the cards), so
        #: they live here keyed by node id, not only on the items.
        self._run: Dict[str, tuple] = {}      # node_id → (state, fraction, note, seconds)
        #: source cards mid-ingest — exempt from a pull's canvas reset (see
        #: :meth:`set_ingesting`).
        self._ingesting: frozenset = frozenset()
        self._anim = QTimer(self)
        self._anim.setInterval(PROGRESS_TICK_MS)
        self._anim.timeout.connect(self._tick_progress)
        self.setSceneRect(-400, -300, 3200, 2000)
        document.on_change(self.sync)
        self.sync()

    # ── model → canvas ────────────────────────────────────────────────────────
    def sync(self) -> None:
        # reconcile by RECORD IDENTITY, not just id presence: load_file() rebuilds
        # NodeRecords with the same auto-ids (n1, n2, …), so an id can map to a BRAND
        # NEW record — a stale NodeItem would keep painting the old op_key/spec and
        # swallow inspector edits into an orphaned record (review 2026-07-22 BLOCKER).
        for nid in list(self.node_items):
            item = self.node_items[nid]
            if nid not in self.doc.nodes or item.rec is not self.doc.nodes[nid]:
                self.node_items.pop(nid)
                self.removeItem(item)
        for nid, rec in self.doc.nodes.items():
            if nid not in self.node_items:
                item = NodeItem(rec, self.doc)
                item.delete_requested.connect(self._on_delete_requested)
                item.pick_requested.connect(self.pick_requested)
                self.addItem(item)
                self.node_items[nid] = item
        self._run = {k: v for k, v in self._run.items() if k in self.node_items}
        for item in self.node_items.values():
            item.refresh()
            item.set_viewed(item.node_id == self.viewed_id)   # survives a re-sync
            st = self._run.get(item.node_id)                  # …and so does the run state
            if st is not None:
                item.set_run_state(st[0], fraction=st[1], note=st[2], seconds=st[3],
                                   levels=st[4])
        # frames (behind nodes) — reconcile by record identity like node items, then
        # reflow each to enclose its now-present members.
        for fid in list(self.frame_items):
            item = self.frame_items[fid]
            if fid not in self.doc.frames or item.rec is not self.doc.frames[fid]:
                self.frame_items.pop(fid)
                self.removeItem(item)
        for fid, fr in self.doc.frames.items():
            if fid not in self.frame_items:
                fitem = FrameItem(fr, self)
                self.addItem(fitem)
                self.frame_items[fid] = fitem
        for fitem in self.frame_items.values():
            fitem.reflow()
        for e in self.edge_items:
            self.removeItem(e)
        self.edge_items = []
        for (s, ss, d, ds) in self.doc.edges:
            si, di = self.node_items.get(s), self.node_items.get(d)
            a = si.socket("out", ss) if si else None
            b = di.socket("in", ds) if di else None
            if a is not None and b is not None:
                edge = EdgeItem(a, b, (s, ss, d, ds))
                self.addItem(edge)
                self.edge_items.append(edge)
        self._sync_flows()      # edge items are rebuilt here — re-apply the run overlay
        self._sync_anim()

    def set_viewed(self, node_id: Optional[str]) -> None:
        """Mark which node the Viewer is showing — its card gets an accent spine + a live
        dot, so in maximized mode you can still see *what* the mini-map is displaying."""
        if node_id == self.viewed_id:
            return
        self.viewed_id = node_id
        for nid, item in self.node_items.items():
            item.set_viewed(nid == node_id)

    def reroute(self) -> None:
        for e in self.edge_items:
            if e.src.scene() is not None and e.dst.scene() is not None:
                e.update_path()
        for fitem in self.frame_items.values():   # frames follow their members' bounds
            fitem.reflow()

    def resync_specs(self) -> List[str]:
        """Re-resolve every card against the registry after a live node reload
        (:mod:`nodegraph.hotreload`); returns the node ids whose op no longer exists.

        Each card caches its :class:`NodeSpec` at construction, so a reload that changed a
        node's sockets, modes or descriptions is invisible on the canvas until this runs.
        Wires are re-pathed afterwards because a changed socket set moves the ports they
        land on — a socket that grew a row taller would otherwise leave every wire into
        this card pointing at where its port used to be."""
        orphaned = [nid for nid, item in self.node_items.items() if not item.resync_spec()]
        self.reroute()
        return orphaned

    # ── per-node run progress ─────────────────────────────────────────────────
    def _set_state(self, node_id: str, state: str, *, fraction=None, note: str = "",
                   seconds=None, levels=None) -> None:
        # `levels` rides in the cached tuple too, so a card rebuilt mid-run (a re-sync while
        # the engine is still walking) comes back with BOTH rails rather than falling back
        # to the flat one until the next progress event.
        self._run[node_id] = (state, fraction, note, seconds, levels)
        item = self.node_items.get(node_id)
        if item is not None:
            item.set_run_state(state, fraction=fraction, note=note, seconds=seconds,
                               levels=levels)
        self._sync_flows()
        self._sync_anim()

    def _sync_flows(self) -> None:
        """A wire flows while the pull is still in flight and its **source has already
        produced** — so the animation traces where data actually moved, not every wire."""
        produced = {nid for nid, v in self._run.items() if v[0] in ("done", "cached")}
        busy = any(v[0] in ("queued", "running", "decoding") for v in self._run.values())
        for e in self.edge_items:
            e.set_flow(busy and e.model_edge[0] in produced)

    def _sync_anim(self) -> None:
        """Run the shared animation timer only while something needs animating — the
        pulsing dots/rails on working cards and the dashes on flowing wires."""
        want = (any(i.is_running() for i in self.node_items.values())
                or any(e.flow for e in self.edge_items))
        if want and not self._anim.isActive():
            self._anim.start()
        elif not want and self._anim.isActive():
            self._anim.stop()

    def _tick_progress(self) -> None:
        for item in self.node_items.values():
            item.advance_phase()
        for e in self.edge_items:
            e.advance_phase()

    def set_ingesting(self, node_ids: Iterable[str]) -> None:
        """The source cards whose own ingest is running (V2.21).

        They are exempt from the two places a pull resets the canvas — :meth:`set_run_plan`
        and :meth:`finish_run` — because an ingest is not part of any pull: it starts on a
        double-click, keeps running across edits and other people's pulls, and its card is
        the only thing reporting it. Clearing it there would blank a bar that is still
        moving, on a job that may have minutes left to run."""
        self._ingesting = frozenset(node_ids)

    def set_run_plan(self, target: str, node_ids: Iterable[str]) -> None:
        """A pull was submitted: mark every participating node ``queued`` and clear the
        cards that aren't in this run (their last result says nothing about this one) —
        except any card mid-ingest, which is reporting work of its own."""
        planned = {n for n in node_ids}
        busy = self._ingesting
        for nid in list(self._run):
            if nid not in busy:
                self._run.pop(nid, None)
        for nid, item in self.node_items.items():
            if nid in busy:
                continue
            if nid in planned:
                self._set_state(nid, "queued")
            else:
                item.set_run_state("")
        self._sync_flows()
        self._sync_anim()

    def on_node_progress(self, event: str, node_id: str, info: dict) -> None:
        """Sink for :attr:`~nodelab_v2.runner.EngineRunner.node_progress`."""
        if event == "start":
            self._set_state(node_id, "running")
        elif event == "progress":
            # `info` carries the two-level split when the compute knows its frame count;
            # the card falls back to the single flat bar when it does not (see
            # nodegraph.engine.Observer).
            self._set_state(node_id, "running", fraction=info.get("fraction"),
                            note=str(info.get("note") or ""), levels=info)
        elif event == "cached":
            self._set_state(node_id, "cached")
        elif event == "done":
            self._set_state(node_id, "done", seconds=info.get("seconds"))
        elif event == "error":
            self._set_state(node_id, "error")
        elif event == "decode":
            self._set_state(node_id, "decoding")

    def finish_run(self, node_id: Optional[str] = None, *, failed: bool = False,
                   seconds: Optional[float] = None) -> None:
        """The pull ended: drop any card still ``queued``/``running`` (the engine stopped
        walking — a queued node it never reached is simply not part of the answer) and
        stamp the pulled node with the run's **wall time**, which for the viewed node
        includes reading its planes — the number the user actually waited on.

        On failure the red state belongs to the node that actually **raised**, which the
        engine's ``error`` event has already stamped; the pulled node is only marked when
        nothing else claimed the failure (e.g. the plane decode blew up after every
        compute had returned)."""
        blamed = any(v[0] == "error" for v in self._run.values())
        for nid, item in self.node_items.items():
            if nid in self._ingesting:
                continue          # its own ingest is still running — not this pull's card
            if self._run.get(nid, ("",))[0] in ("queued", "running", "decoding"):
                self._run.pop(nid, None)
                item.set_run_state("")
        if failed and not blamed and node_id is not None:
            self._set_state(node_id, "error")
        elif not failed and node_id is not None and node_id in self.node_items:
            self._set_state(node_id, "done", seconds=seconds)
        self._sync_flows()          # nothing is in flight → the wires stop flowing
        self._sync_anim()

    def clear_run_states(self) -> None:
        self._run.clear()
        for item in self.node_items.values():
            item.set_run_state("")
        self._sync_flows()
        self._sync_anim()

    # ── deleting ──────────────────────────────────────────────────────────────
    def _on_delete_requested(self, node_id: str) -> None:
        """The hover ✕ badge: delete that node — plus the rest of the selection if the
        clicked card is part of a multi-node selection (what a user expects when they
        have five cards selected and hit the ✕ on one of them)."""
        sel = [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]
        self.delete_nodes(sel if node_id in sel and len(sel) > 1 else [node_id])

    def delete_nodes(self, node_ids: Iterable[str]) -> List[str]:
        """Remove nodes from the document (their wires go with them). Returns the ids
        actually removed."""
        gone = []
        for nid in list(node_ids):
            if nid in self.doc.nodes:
                self.doc.remove_node(nid)
                self._run.pop(nid, None)
                gone.append(nid)
        if gone:
            self.nodes_deleted.emit(gone)
        return gone

    def dissolve_node(self, node_id: str) -> bool:
        """Delete ``node_id`` but **heal the chain**: its incoming Dataset wire is
        reconnected to every Dataset consumer it fed. The chain survives the removal of a
        mid-chain node, which is the usual reason to delete one.

        Falls back to a plain delete when there is nothing to bridge (a source, a leaf,
        or a node whose reconnection the document rejects)."""
        if node_id not in self.doc.nodes:
            return False
        rec = self.doc.nodes[node_id]
        spec = rec.spec()
        if spec is None:
            return bool(self.delete_nodes([node_id]))
        state = rec.state()
        ins = [s.name for s in spec.active_inputs(state) if s.type is SocketType.DATASET]
        outs = [s.name for s in spec.active_outputs(state)
                if s.type is SocketType.DATASET]
        upstream = None
        for name in ins:                       # the first WIRED Dataset input is the source
            e = self.doc.edge_into(node_id, name)
            if e is not None:
                upstream = (e[0], e[1])
                break
        downstream = [(d, ds) for (s, ss, d, ds) in self.doc.edges
                      if s == node_id and ss in outs]
        self.delete_nodes([node_id])
        if upstream is None:
            return True
        for (dst, dsock) in downstream:
            ok, _ = self.doc.can_connect(upstream[0], upstream[1], dst, dsock)
            if ok:
                try:
                    self.doc.connect(upstream[0], upstream[1], dst, dsock)
                except ValueError:
                    pass                       # a rejected heal just leaves the gap
        return True

    # ── splice-on-wire (G1) ────────────────────────────────────────────────────
    def edge_at(self, scene_pos: QPointF) -> Optional[EdgeItem]:
        for it in self.items(scene_pos):
            if isinstance(it, EdgeItem):
                return it
        return None

    def splice_onto(self, node_id: str, edge: tuple) -> bool:
        """Insert ``node_id`` into an existing wire ``edge`` = ``(s, ss, d, ds)``: the
        node's first Dataset input takes the wire's source, its first Dataset output
        feeds the wire's dest (the original edge is replaced). Silently no-ops if the
        node has no Dataset in+out or a connection is invalid."""
        rec = self.doc.nodes.get(node_id)
        spec = rec.spec() if rec else None
        if spec is None or node_id not in self.doc.nodes:
            return False
        di = next((s.name for s in spec.active_inputs(rec.state())
                   if s.type is SocketType.DATASET), None)
        do = next((s.name for s in spec.active_outputs(rec.state())
                   if s.type is SocketType.DATASET), None)
        if di is None or do is None:
            return False
        s, ss, d, ds = edge
        if s not in self.doc.nodes or d not in self.doc.nodes:
            return False
        # validate BOTH new connections BEFORE removing the original edge — else a
        # splice onto a non-Dataset (field/value) wire, or any invalid pairing, would
        # drop the wire with no replacement (transactional; review 2026-07-22).
        ok_a, _ = self.doc.can_connect(s, ss, node_id, di)
        ok_b, _ = self.doc.can_connect(node_id, do, d, ds)
        if not (ok_a and ok_b):
            return False
        self.doc.disconnect(s, ss, d, ds)
        try:
            self.doc.connect(s, ss, node_id, di)
            self.doc.connect(node_id, do, d, ds)
        except ValueError:
            # extremely unlikely after the pre-check, but never leave a dropped wire
            self.doc.connect(s, ss, d, ds)
            return False
        return True

    # ── wire dragging (G1) ────────────────────────────────────────────────────
    def begin_wire(self, socket: SocketItem, scene_pos: QPointF) -> None:
        fixed = socket
        if socket.io == "in":
            existing = self.doc.edge_into(socket.node_item.node_id, socket.spec.name)
            if existing is not None and not getattr(socket.spec, "multi", False):
                # detach the wire; keep dragging from its SOURCE end
                self.doc.disconnect(*existing)
                src_item = self.node_items.get(existing[0])
                fixed = (src_item.socket("out", existing[1])
                         if src_item is not None else socket)
        self._drag_fixed = fixed
        self._temp_wire = QGraphicsPathItem()
        pen = QPen(T.WIRE, 2.0, Qt.DashLine)
        pen.setCapStyle(Qt.RoundCap)
        self._temp_wire.setPen(pen)
        self._temp_wire.setZValue(-0.5)
        self.addItem(self._temp_wire)
        self._update_temp(scene_pos)

    def _endpoints(self, cand: SocketItem, fixed: Optional[SocketItem] = None):
        """(src_item, dst_item) for fixed↔candidate, or None if same direction.

        ``fixed`` defaults to the live drag anchor, but :meth:`_end_wire` passes it
        explicitly: it tears the drag down (``_cancel_temp`` clears ``_drag_fixed``)
        *before* resolving the drop, so reading the anchor from state there would hit
        ``None`` — which is exactly what crashed every drop onto a socket."""
        f = fixed if fixed is not None else self._drag_fixed
        if f is None:
            return None
        if f.io == "out" and cand.io == "in":
            return f, cand
        if f.io == "in" and cand.io == "out":
            return cand, f
        return None

    def _validity(self, cand: SocketItem) -> Optional[bool]:
        f = self._drag_fixed
        if f is None or cand.node_item is f.node_item:
            return False
        ends = self._endpoints(cand, f)
        if ends is None:
            return False
        src, dst = ends
        ok, _ = self.doc.can_connect(src.node_item.node_id, src.spec.name,
                                     dst.node_item.node_id, dst.spec.name)
        return ok

    def _socket_at(self, scene_pos: QPointF) -> Optional[SocketItem]:
        for it in self.items(scene_pos):
            if isinstance(it, SocketItem):
                return it
        return None

    def _update_temp(self, scene_pos: QPointF) -> None:
        if self._temp_wire is None or self._drag_fixed is None:
            return
        a = self._drag_fixed.anchor()
        pts = (a, scene_pos) if self._drag_fixed.io == "out" else (scene_pos, a)
        self._temp_wire.setPath(wire_path(*pts))
        sock = self._socket_at(scene_pos)
        if sock is not self._hover_sock:
            if self._hover_sock is not None:
                self._hover_sock.set_highlight(None)
            self._hover_sock = sock
        if sock is not None and sock is not self._drag_fixed:
            valid = self._validity(sock)
            sock.set_highlight(valid)
            pen = self._temp_wire.pen()
            pen.setColor(T.WIRE if valid else T.ERROR)
            self._temp_wire.setPen(pen)

    def _end_wire(self, scene_pos: QPointF, screen_pos) -> None:
        fixed = self._drag_fixed
        sock = self._socket_at(scene_pos)
        self._cancel_temp()
        if fixed is None:
            return
        if sock is not None and sock is not fixed:
            ends = self._endpoints(sock, fixed)
            if ends is not None:
                src, dst = ends
                ok, _reason = self.doc.can_connect(
                    src.node_item.node_id, src.spec.name,
                    dst.node_item.node_id, dst.spec.name)
                if ok:
                    self.doc.connect(src.node_item.node_id, src.spec.name,
                                     dst.node_item.node_id, dst.spec.name)
            return
        # empty canvas → link-drag search (G1)
        self._open_link_search(fixed, scene_pos, screen_pos)

    def _cancel_temp(self) -> None:
        if self._temp_wire is not None:
            self.removeItem(self._temp_wire)
            self._temp_wire = None
        if self._hover_sock is not None:
            self._hover_sock.set_highlight(None)
            self._hover_sock = None
        self._drag_fixed = None

    def _open_link_search(self, fixed: SocketItem, scene_pos: QPointF,
                          screen_pos) -> None:
        entries = [(f"{spec.label}   ·  {sock}", spec.op_key, sock)
                   for spec, sock in compatible_ops(fixed.spec, fixed.io)]
        if not entries:
            return
        views = self.views()
        if not views:
            return
        view = views[0]

        fixed_id = fixed.node_item.node_id
        fixed_name = fixed.spec.name
        fixed_io = fixed.io

        def pick(op_key: str, sock_name: str) -> None:
            if fixed_id not in self.doc.nodes:
                return                    # the anchor node was deleted meanwhile
            rec = self.doc.add_node(op_key, x=scene_pos.x() - 20,
                                    y=scene_pos.y() - 30)
            try:
                if fixed_io == "out":
                    self.doc.connect(fixed_id, fixed_name, rec.id, sock_name)
                else:
                    self.doc.connect(rec.id, sock_name, fixed_id, fixed_name)
            except ValueError:            # socket no longer active / incompatible
                pass

        popup = LinkSearchPopup(view, entries, pick)
        popup.move(screen_pos)
        popup.show()

    # ── scene mouse plumbing ──────────────────────────────────────────────────
    def mouseMoveEvent(self, e) -> None:
        if self._drag_fixed is not None:
            self._update_temp(e.scenePos())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._drag_fixed is not None:
            if e.button() != Qt.LeftButton:
                self._cancel_temp()          # right/middle release cancels the drag
                e.accept()
                return
            self._end_wire(e.scenePos(), e.screenPos())
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        for it in self.items(e.scenePos()):
            if isinstance(it, NodeItem):
                self.node_activated.emit(it.node_id)
                e.accept()
                return
        # double-click a wire → drop a reroute node spliced into it at the click point
        edge = self.edge_at(e.scenePos())
        if edge is not None:
            self._reroute_on(edge, e.scenePos())   # refuses non-Dataset wires cleanly
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    # ── context menu (the discoverable delete) ────────────────────────────────
    def contextMenuEvent(self, e) -> None:
        node = next((it for it in self.items(e.scenePos())
                     if isinstance(it, NodeItem)), None)
        edge = self.edge_at(e.scenePos())
        frame = next((it for it in self.items(e.scenePos())
                      if isinstance(it, FrameItem)), None)
        menu = QMenu()
        menu.setStyleSheet(T.menu_qss())
        if node is not None:
            self._fill_node_menu(menu, node)
        elif edge is not None:
            self._fill_edge_menu(menu, edge, e.scenePos())
        elif frame is not None:
            self._fill_frame_menu(menu, frame)
        else:
            act = menu.addAction("Fit graph")
            act.triggered.connect(lambda: [v.fit_all() for v in self.views()
                                           if hasattr(v, "fit_all")])
        menu.exec(e.screenPos())
        e.accept()

    def _fill_node_menu(self, menu: QMenu, node: NodeItem) -> None:
        if not node.isSelected():        # right-click acts on what you clicked
            self.clearSelection()
            node.setSelected(True)
        sel = [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]
        nid = node.node_id
        rec = self.doc.nodes.get(nid)
        title = menu.addAction(f"{nid} · {node.op_key}")
        title.setEnabled(False)
        menu.addSeparator()
        menu.addAction("View / pull this node\tF5").triggered.connect(
            lambda: self.pull_requested.emit(nid))
        if node.op_key == LOAD_OP and str((rec.params.get("path") if rec else "") or ""):
            act = menu.addAction("Ingest this file now")
            act.setToolTip("Write this file's .b2nd store now, on its own worker. Other "
                           "source cards can ingest at the same time and the app stays "
                           "usable.")
            act.triggered.connect(lambda: self.ingest_requested.emit(nid))
        mute = menu.addAction("Muted (pass through)\tM")
        mute.setCheckable(True)
        mute.setChecked(bool(rec.muted) if rec is not None else False)
        mute.triggered.connect(
            lambda: [self.doc.set_muted(i, not self.doc.nodes[i].muted)
                     for i in sel if i in self.doc.nodes])
        coll = menu.addAction("Collapsed\tC")
        coll.setCheckable(True)
        coll.setChecked(bool(rec.collapsed) if rec is not None else False)
        coll.triggered.connect(
            lambda: [self.doc.set_collapsed(i, not self.doc.nodes[i].collapsed)
                     for i in sel if i in self.doc.nodes])
        if node.op_key == DOCK_OP:
            menu.addSeparator()
            status, detail = self.doc.dock_status(nid)
            head = menu.addAction(f"dock · {status}{(' — ' + detail) if detail else ''}")
            head.setEnabled(False)
            baked = bool(bake_record(rec)) if rec is not None else False
            unset = node.state().get("precision", PRECISION_UNSET) == PRECISION_UNSET
            bake = menu.addAction("Re-bake this dock" if baked else "Bake this dock…")
            bake.setEnabled(not unset)
            if unset:
                bake.setToolTip("Choose a Precision in the inspector first — there is no "
                                "default, because the right one depends on this chain.")
            bake.triggered.connect(lambda: self.dock_action.emit(nid, "bake"))
            if status in ("docked", "stale"):
                menu.addAction("Un-dock (run the chain live)").triggered.connect(
                    lambda: self.dock_action.emit(nid, "undock"))
            elif baked:
                menu.addAction("Re-dock (serve the existing bake)").triggered.connect(
                    lambda: self.dock_action.emit(nid, "redock"))
        menu.addSeparator()
        many = len(sel) > 1
        menu.addAction(f"Delete {len(sel)} nodes\tDel" if many
                       else "Delete node\tDel").triggered.connect(
            lambda: self.delete_nodes(sel))
        dis = menu.addAction("Dissolve (delete, keep the chain)\tCtrl+X")
        dis.setToolTip("Delete the node and reconnect its input to whatever it fed")
        dis.triggered.connect(lambda: [self.dissolve_node(i) for i in sel])

    def _fill_edge_menu(self, menu: QMenu, edge: EdgeItem, pos: QPointF) -> None:
        s, ss, d, ds = edge.model_edge
        title = menu.addAction(f"{s}.{ss} → {d}.{ds}")
        title.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Insert reroute (double-click)").triggered.connect(
            lambda: self._reroute_on(edge, pos))
        menu.addAction("Delete wire\tDel").triggered.connect(
            lambda: self.doc.disconnect(s, ss, d, ds))

    def _fill_frame_menu(self, menu: QMenu, frame: FrameItem) -> None:
        title = menu.addAction(f"frame · {frame.rec.title}")
        title.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Delete frame (keeps the nodes)\tDel").triggered.connect(
            lambda: self.doc.remove_frame(frame.frame_id))

    def _reroute_on(self, edge: EdgeItem, pos: QPointF) -> None:
        r = T.RR_SIZE / 2.0
        rec = self.doc.add_node("rr.reroute", x=pos.x() - r, y=pos.y() - r)
        if not self.splice_onto(rec.id, edge.model_edge):
            self.doc.remove_node(rec.id)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape and self._drag_fixed is not None:
            self._cancel_temp()
            e.accept()
            return
        if e.key() == Qt.Key_X and (e.modifiers() & Qt.ControlModifier):
            for nid in [i.node_id for i in self.selectedItems()
                        if isinstance(i, NodeItem)]:
                self.dissolve_node(nid)
            e.accept()
            return
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._drag_fixed is not None:
                self._cancel_temp()          # never delete out from under a live drag
            self.delete_selection()
            e.accept()
            return
        if e.key() == Qt.Key_M:
            for it in self.selectedItems():
                if isinstance(it, NodeItem):
                    self.doc.set_muted(it.node_id, not it.rec.muted)
            e.accept()
            return
        if e.key() == Qt.Key_C:
            for it in self.selectedItems():
                if isinstance(it, NodeItem):
                    self.doc.set_collapsed(it.node_id, not it.rec.collapsed)
            e.accept()
            return
        super().keyPressEvent(e)

    def delete_selection(self) -> None:
        edges = [it.model_edge for it in self.selectedItems()
                 if isinstance(it, EdgeItem)]
        nodes = [it.node_id for it in self.selectedItems() if isinstance(it, NodeItem)]
        frames = [it.frame_id for it in self.selectedItems()
                  if isinstance(it, FrameItem)]
        for e in edges:
            self.doc.disconnect(*e)
        for fid in frames:               # delete the frame only, NOT its member nodes
            self.doc.remove_frame(fid)
        self.delete_nodes(nodes)

    def dissolve_selection(self) -> None:
        for nid in [it.node_id for it in self.selectedItems()
                    if isinstance(it, NodeItem)]:
            self.dissolve_node(nid)


class GraphView(QGraphicsView):
    """Pannable/zoomable canvas with a Blender-style dotted grid + palette drops.

    Carries its own **maximize toggle** — a painted HUD button pinned to the canvas's
    top-right corner (mirrored by View → *Maximize node canvas* / ``Ctrl+Space``).
    Maximizing hands the whole centre to the graph and re-homes the Viewer into the
    :class:`~nodelab_v2.minimap.MiniMapOverlay`; the window owns that move and drives
    the button's state back through :meth:`set_maximized`.

    It also carries the **troubleshooting HUD** (:meth:`set_troubleshooting`): an amber
    frame around the whole canvas plus a top-left badge, up for as long as pulls are
    scoped to picked frames instead of the series.
    """

    STEP = 26
    op_dropped = Signal(str, QPointF)
    maximize_toggled = Signal(bool)

    #: troubleshooting frame: stroke width, and the inset its rounded rect sits at.
    TS_BORDER = 3
    TS_INSET = 12
    #: pulse period of the badge's live dot (ms) — the canvas twin of the status-bar LED.
    #: Only the dot's colour changes, so the tick repaints a 200 px label and never the
    #: scene (a pulsing FRAME would repaint every node on the canvas twice a second).
    TS_PULSE_MS = 620

    def __init__(self, scene: GraphScene) -> None:
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(T.BG)
        self.setFrameShape(QGraphicsView.NoFrame)
        # no scrollbars: the canvas pans by dragging empty space (and `_grow_scene_rect`
        # keeps extending the scene, so the bars never meant anything useful). Hiding
        # them keeps the range — the pan below still scrolls through it.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAcceptDrops(True)
        self._panning = False
        self._pan_moved = False
        self._pan_last = QPointF()
        # troubleshooting HUD: an amber badge in the canvas's top-left corner, twinned
        # with the frame drawn in `drawForeground`. A child of the VIEW (like the maximize
        # button) so it never scrolls with the scene. Built FIRST: `childEvent` installs a
        # geometry filter on every child, and that filter reads these attributes.
        self._ts_on = False
        self._ts_detail = ""
        self._ts_dot = True
        self._ts_badge = QLabel("", self)
        self._ts_badge.setObjectName("tsBadge")
        self._ts_badge.setTextFormat(Qt.RichText)
        self._ts_badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._ts_badge.hide()
        self._ts_timer = QTimer(self)
        self._ts_timer.setInterval(self.TS_PULSE_MS)
        self._ts_timer.timeout.connect(self._ts_pulse)
        # corner chrome: the maximize toggle (a child of the VIEW, not the viewport, so
        # it never scrolls with the scene and always paints above it)
        self._max_btn = HudButton("maximize", self)
        self._max_btn.setCheckable(True)
        self._max_btn.setToolTip(
            "Maximize the node canvas (Ctrl+Space) — the Viewer becomes a mini-map "
            "in the top-left corner and follows the node you click")
        self._max_btn.toggled.connect(self._on_max_toggled)
        self._place_corner_chrome()

    # ── corner chrome (maximize + the troubleshooting badge) ──────────────────
    def _place_corner_chrome(self) -> None:
        self._max_btn.move(self.width() - self._max_btn.width() - 12, 12)
        self._max_btn.raise_()
        self._place_ts_badge()

    def _hud_siblings(self) -> List[QWidget]:
        """The other *floating* HUD widgets over the canvas (mini-map, welcome card,
        maximize button) — everything the badge has to share the surface with.

        The viewport and the scroll bars are excluded by identity: they are the scroll
        area's own furniture, and the viewport in particular covers the entire canvas, so
        treating it as something to dodge would push the badge off the bottom edge."""
        skip = {id(self.viewport()), id(self.horizontalScrollBar()),
                id(self.verticalScrollBar()), id(self._ts_badge)}
        return [w for w in self.children()
                if isinstance(w, QWidget) and id(w) not in skip and w.isVisible()]

    def _place_ts_badge(self) -> None:
        """Pin the badge inside the canvas's top-left corner, stepping aside for any HUD
        frame already there.

        The mini-map (the re-homed Viewer in maximized mode) claims that same corner by
        default. Preference order: the corner itself → immediately **right of** whatever
        occupies it, staying on the top strip → **under** it, if there is no room to the
        right. The rule is geometric rather than a special case for the mini-map, so a
        future HUD gets the same treatment."""
        if not self._ts_on:
            return
        self._ts_badge.adjustSize()
        pad = self.TS_INSET + self.TS_BORDER + 3
        w, h = self._ts_badge.width(), self._ts_badge.height()
        rect = self._ts_badge.rect().translated(pad, pad)
        blockers = [g for g in (s.geometry() for s in self._hud_siblings())
                    if g.intersects(rect)]
        x, y = pad, pad
        if blockers:
            right = max(g.right() for g in blockers) + 8
            below = max(g.bottom() for g in blockers) + 8
            if right + w <= self.width() - pad:
                x = right                       # keep it on the top strip
            elif below + h <= self.height() - pad:
                y = below
        self._ts_badge.move(x, y)
        self._ts_badge.raise_()

    # ── troubleshooting HUD ───────────────────────────────────────────────────
    def set_troubleshooting(self, on: bool, detail: str = "") -> None:
        """Show/hide the scoped-run HUD: the amber frame + the top-left badge.

        The canvas is the honest place for it. Scoping a run to picked frames does not
        change the graph, so the cards, wires and progress all look exactly as they do on
        a full run — which is precisely why the mode needs to be impossible to miss from
        the canvas itself, not only from a chip in the status bar. ``detail`` is the
        compact scope (e.g. ``t7`` / ``3T[0,4,9]·2Z``) shown under the title."""
        on, detail = bool(on), str(detail)
        if (on, detail) == (self._ts_on, self._ts_detail):
            return
        self._ts_on, self._ts_detail = on, detail
        self._ts_dot = True
        self._style_ts_badge()
        self._ts_badge.setVisible(on)
        self._place_corner_chrome()
        if on:
            self._ts_timer.start()
        else:
            self._ts_timer.stop()                   # no idle cost when it is off
        self.viewport().update()                    # repaint the frame

    def is_troubleshooting(self) -> bool:
        return self._ts_on

    def _ts_pulse(self) -> None:
        self._ts_dot = not self._ts_dot
        self._refresh_ts_text()

    # The badge shares its corner with whatever else is floating over the canvas, and
    # those move on their own (the mini-map is draggable, and appears/vanishes with
    # Ctrl+Space). Watching every child's geometry keeps the dodge in :meth:`_place_ts_badge`
    # correct without this class having to know what the other HUD widgets ARE.
    def childEvent(self, e) -> None:                    # noqa: N802 — Qt override
        super().childEvent(e)
        if e.type() == QEvent.ChildAdded and isinstance(e.child(), QWidget):
            e.child().installEventFilter(self)

    def eventFilter(self, obj, ev) -> bool:             # noqa: N802 — Qt override
        # `getattr`, not `self._ts_on`: QGraphicsView's own constructor creates the
        # viewport child, so ChildAdded (and the viewport's first Show/Resize) reach this
        # filter BEFORE __init__'s body has bound any of the HUD attributes.
        if (getattr(self, "_ts_on", False) and obj is not self._ts_badge
                and ev.type() in (QEvent.Move, QEvent.Resize,
                                  QEvent.Show, QEvent.Hide)):
            self._place_ts_badge()
        return False

    def _refresh_ts_text(self) -> None:
        """Re-render the badge's rich text. Split from :meth:`_style_ts_badge` because the
        pulse runs through here twice a second — a stylesheet re-polish per tick is not
        worth the blink."""
        dot = (T.DIM2D_INK if self._ts_dot else T.mix(T.DIM2D_INK, T.DIM2D, 0.62)).name()
        tail = (f"<br><span style='font-size:10px;font-weight:600;'>{self._ts_detail}"
                f"  ·  F9 to exit</span>") if self._ts_detail else ""
        self._ts_badge.setText(
            f"<span style='font-size:12px;font-weight:800;'>"
            f"<span style='color:{dot};'>&#9679;</span>&nbsp; TROUBLESHOOTING MODE"
            f"</span>{tail}")

    def _style_ts_badge(self) -> None:
        self._refresh_ts_text()
        self._ts_badge.setStyleSheet(
            f"QLabel#tsBadge {{ background:{T.DIM2D.name()}; color:{T.DIM2D_INK.name()};"
            f" border:1px solid {T.mix(T.DIM2D, T.DIM2D_INK, 0.25).name()};"
            f" border-radius:8px; padding:6px 12px; }}")
        self._ts_badge.adjustSize()

    def _on_max_toggled(self, on: bool) -> None:
        self._max_btn.set_kind("restore" if on else "maximize")
        self._max_btn.setToolTip(
            "Restore the docked Viewer (Esc)" if on else
            "Maximize the node canvas (Ctrl+Space) — the Viewer becomes a mini-map "
            "in the top-left corner and follows the node you click")
        self.maximize_toggled.emit(on)

    def set_maximized(self, on: bool) -> None:
        """Reflect the window's state on the button without re-emitting (the menu
        action and the button are two entry points to the same toggle)."""
        if self._max_btn.isChecked() == bool(on):
            return
        self._max_btn.blockSignals(True)
        self._max_btn.setChecked(bool(on))
        self._max_btn.blockSignals(False)
        self._max_btn.set_kind("restore" if on else "maximize")

    def is_maximized(self) -> bool:
        return self._max_btn.isChecked()

    def restyle(self) -> None:
        self._max_btn.update()
        self._style_ts_badge()          # amber + ink come from the theme tokens
        self._place_corner_chrome()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._place_corner_chrome()

    def drawForeground(self, p: QPainter, rect: QRectF) -> None:
        """Paint the troubleshooting frame over the graph, in VIEWPORT pixels.

        Reset transform (the viewer's overlay idiom): the frame belongs to the canvas
        widget, not to the scene, so it must not pan, zoom or scale with the nodes."""
        super().drawForeground(p, rect)
        if not self._ts_on:
            return
        p.save()
        p.setTransform(QTransform())
        vp = self.viewport().rect()
        p.setBrush(Qt.NoBrush)
        # a soft wide wash first, the crisp stroke over it — reads as a glow at a glance
        # without costing an animation (the viewport repaints the whole scene).
        p.setPen(QPen(T.alpha(T.DIM2D, 55), self.TS_BORDER * 3.5))
        p.drawRoundedRect(QRectF(vp).adjusted(self.TS_INSET, self.TS_INSET,
                                              -self.TS_INSET, -self.TS_INSET), 12, 12)
        p.setPen(QPen(T.DIM2D, self.TS_BORDER))
        p.drawRoundedRect(QRectF(vp).adjusted(self.TS_INSET, self.TS_INSET,
                                              -self.TS_INSET, -self.TS_INSET), 12, 12)
        p.restore()

    def keyPressEvent(self, e) -> None:
        # Esc leaves maximized mode — but never out from under a live wire drag, which
        # owns Escape (the scene cancels the drag with it).
        if (e.key() == Qt.Key_Escape and self._max_btn.isChecked()
                and getattr(self.scene(), "_drag_fixed", None) is None):
            self._max_btn.setChecked(False)
            e.accept()
            return
        super().keyPressEvent(e)

    # canvas panning (G1): left-drag on EMPTY canvas pans the view; Ctrl/Shift+drag
    # keeps the rubber-band marquee, and a drag that starts on a node/socket/edge
    # falls through to the default handling (move node / start wire).
    def mousePressEvent(self, e) -> None:
        if (e.button() == Qt.LeftButton
                and self.itemAt(e.position().toPoint()) is None
                and not (e.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))):
            self._panning = True
            self._pan_moved = False
            self._pan_last = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._panning:
            d = e.position() - self._pan_last
            self._pan_last = e.position()
            if d.manhattanLength() > 0:
                self._pan_moved = True
            self._grow_scene_rect()      # keep headroom so the drag is never clamped
            h, v = self.horizontalScrollBar(), self.verticalScrollBar()
            h.setValue(h.value() - int(d.x()))
            v.setValue(v.value() - int(d.y()))
            e.accept()
            return
        super().mouseMoveEvent(e)

    def _grow_scene_rect(self) -> None:
        """Expand the scene rect to always extend a wide margin beyond the current
        viewport, so a left-drag pan can travel arbitrarily far past the nodes (the
        scrollbars clamp to the scene rect — a fixed rect is what stops the drag)."""
        sc = self.scene()
        if sc is None:
            return
        vis = self.mapToScene(self.viewport().rect()).boundingRect()
        want = vis.adjusted(-4000, -4000, 4000, 4000)
        united = sc.sceneRect().united(want)
        if united != sc.sceneRect():
            sc.setSceneRect(united)

    def mouseReleaseEvent(self, e) -> None:
        if self._panning and e.button() == Qt.LeftButton:
            self._panning = False
            self.viewport().unsetCursor()
            if not self._pan_moved:
                self.scene().clearSelection()   # a plain empty click deselects
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def drawBackground(self, p: QPainter, rect) -> None:
        super().drawBackground(p, rect)
        step = self.STEP
        left = int(rect.left()) - (int(rect.left()) % step)
        top = int(rect.top()) - (int(rect.top()) % step)
        p.setPen(Qt.NoPen)
        p.setBrush(T.GRID_DOT)
        y = float(top)
        while y < rect.bottom():
            x = float(left)
            while x < rect.right():
                p.drawEllipse(QPointF(x, y), 1.0, 1.0)
                x += step
            y += step

    def wheelEvent(self, e) -> None:
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def fit_all(self) -> None:
        r = self.scene().itemsBoundingRect()
        if r.isEmpty():
            # nothing placed yet — fitting the empty rect would zoom into a ~140 px box
            # (giant grid dots on the welcome canvas). Open 1:1 on the origin instead.
            self.resetTransform()
            self.centerOn(0, 0)
            return
        self.fitInView(r.adjusted(-70, -70, 70, 70), Qt.KeepAspectRatio)

    # palette drag-and-drop (G2)
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasFormat("application/x-nd2studios-op"):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasFormat("application/x-nd2studios-op"):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:
        if e.mimeData().hasFormat("application/x-nd2studios-op"):
            op = bytes(e.mimeData().data("application/x-nd2studios-op")).decode("utf-8")
            self.op_dropped.emit(op, self.mapToScene(e.position().toPoint()))
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


__all__ = ["GraphScene", "GraphView", "LinkSearchPopup", "compatible_ops",
           "visible_specs", "HIDDEN_OP_PREFIXES"]
