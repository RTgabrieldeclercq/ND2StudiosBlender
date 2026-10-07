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
    QApplication, QGraphicsPathItem, QGraphicsScene, QGraphicsView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QToolButton, QVBoxLayout, QWidget,
)

from nodegraph import roles as R
from nodegraph.registry import NODES
from nodegraph.sockets import SocketType, can_connect as _sock_can_connect
from nodelab_v2 import theme as T
from nodelab_v2.document import BATCH_OP, UNBATCH_OP, GraphDocument

#: Extensions the canvas accepts as a DESKTOP file drop (V3.01). The same set the
#: File -> Load dialog offers, kept here as a tuple because a drag has to be judged on
#: every mouse move and a dialog filter string cannot be.
FILE_DROP_SUFFIXES = (".nd2", ".tif", ".tiff")
from nodelab_v2.edge_item import EdgeItem, wire_path
from nodelab_v2.frame_item import FrameItem
from nodelab_v2.minimap import HudButton
from nodelab_v2.node_item import NodeItem, SocketItem
from nodelab_v2.ops import (DOCK_OP, HIDDEN_OP_PREFIXES, LOAD_OP, PAGE_NAME_KEY,
                            PAGE_OUTPUT_OP, PRECISION_UNSET, bake_record)

# ``HIDDEN_OP_PREFIXES`` moved to the Qt-free :mod:`nodelab_v2.ops` (2026-10-02) so the
# readiness checker can rank suggested nodes without importing Qt; still exported here.


def visible_specs(kind: Optional[str] = None):
    """The node types the palette and the link search offer. On a page of ``kind``, only
    those the roles file assigns to that kind (V4.00 step 5) — ``None`` or ``free`` offers
    every one."""
    return [s for s in NODES.all()
            if not s.op_key.startswith(HIDDEN_OP_PREFIXES) and R.op_in_page(s.op_key, kind)]


def compatible_ops(fixed_spec, fixed_io: str,
                   kind: Optional[str] = None) -> List[Tuple[object, str]]:
    """Ops (spec, socket_name) whose default-state sockets can pair with the fixed
    socket — the link-drag search menu (G1), on a page of ``kind``."""
    fixed = fixed_spec.instantiate()
    out = []
    for spec in visible_specs(kind):
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

#: Run states that describe work IN FLIGHT rather than a result. Only these are swept when a
#: card stops belonging to any live run — a terminal ``done``/``cached``/``error`` is a fact
#: about a branch that already finished and survives an unrelated branch starting.
TRANSIENT_RUN_STATES = ("queued", "running", "decoding")


def _needs_topology(fn=None, *, restart: bool = False):
    """A gesture that changes the graph's SHAPE. On a plain page it simply runs. On a linked
    page (V4.00 step 6; asking since step 11e) the window's gate
    (:attr:`GraphScene.topology_gate`) asks how the change applies — unless the user already
    answered for this page — and the gesture then runs on whatever the page has become: this
    scene (the change kept on the page, or sent to the master) or, made unique, the page's NEW
    scene. A gesture begun by a mouse press (``restart`` — a wire drag) is not resumed after a
    question, since the press went to the dialog: the status bar says to do it again. With no
    gate (a scene outside the window) it is refused with the hint, as before 11e."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def guarded(self, *a, **k):
            doc = self.doc
            if getattr(doc, "editable_topology", True) or getattr(doc, "edit_mode", ""):
                return _refusable(self, fn, a, k)
            gate = getattr(self, "topology_gate", None)
            if gate is None:
                from nodelab_v2.linked_document import TOPOLOGY_HINT
                self.topology_refused.emit(TOPOLOGY_HINT)
                return None
            target = gate(self, restart)
            if target is None:
                return None
            if target is self:
                return _refusable(self, fn, a, k)
            return getattr(target, fn.__name__)(*a, **k)
        return guarded
    return deco(fn) if fn is not None else deco


def _refusable(scene, fn, a, k):
    """Run a structural gesture; a linked page that refuses it part-way (one sending its
    edits to the master, asked for a change the master would compute differently — V4.00
    step 11e) says why on the status bar instead of raising out of the gesture."""
    from nodelab_v2.linked_document import LinkedPageError
    try:
        return fn(scene, *a, **k)
    except LinkedPageError as exc:
        scene.topology_refused.emit(str(exc))
        return None


#: the context-menu entries that change a linked page's graph — on a page whose edits are not
#: settled yet they say that choosing them asks how (their labels' first words)
_STRUCTURAL_MENU = ("Delete", "Dissolve", "Fan out", "Insert reroute")


class GraphScene(QGraphicsScene):
    """Document-mirroring scene + the wire-drag state machine."""

    node_activated = Signal(str)      # double-clicked node id → view/pull it (G7)
    pull_requested = Signal(str)      # context menu → pull/view this node
    #: context menu → open this node's result BESIDE the viewed one (the Viewer's
    #: side-by-side compare pane). The window owns both panes, so it routes.
    compare_requested = Signal(str)
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
    #: a structural gesture was refused — the page is LINKED to a master (V4.00 step 6):
    #: the hint, for the window's status bar
    topology_refused = Signal(str)
    #: set by the window (V4.00 step 11e): ``(scene, restart) -> scene | None`` — asks how a
    #: structural edit on a linked page applies and returns the scene to run it on
    topology_gate = None
    #: a Page Output's *New page from this output…* (V4.00 step 11): the node id — the window
    #: opens the New page dialog pre-set to read it
    new_page_from_output = Signal(str)

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
        #: target node_id → the nodes that run claims, one entry per LIVE run (queued or
        #: running). What keeps two branches' cards from erasing each other; see
        #: :meth:`set_run_plan`.
        self._plans: Dict[str, frozenset] = {}
        #: source cards mid-ingest — exempt from a pull's canvas reset (see
        #: :meth:`set_ingesting`).
        self._ingesting: frozenset = frozenset()
        self._anim = QTimer(self)
        self._anim.setInterval(PROGRESS_TICK_MS)
        self._anim.timeout.connect(self._tick_progress)
        self.setSceneRect(-400, -300, 3200, 2000)
        document.on_change(self.sync)
        document.on_moved(self._on_moved)
        self.sync()

    def _on_moved(self, node_id: str) -> None:
        """A card moved in the document — from another canvas showing this page or its
        master/linked twin (V4.00 step 6): put this scene's card where the record says."""
        item = self.node_items.get(node_id)
        rec = self.doc.nodes.get(node_id)
        if item is not None and rec is not None and \
                (item.pos().x(), item.pos().y()) != (rec.x, rec.y):
            item.setPos(rec.x, rec.y)

    def release(self) -> None:
        """Stop listening to the document: the window is dropping this scene (its page was
        deleted, or a file load gave the page another document). A dropped scene left on the
        listener list raises on the document's next edit, and the live listeners after it —
        the scene that replaced it, the window's hooks — never run."""
        self.doc.off_change(self.sync)
        self.doc.off_moved(self._on_moved)
        self._anim.stop()

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
        busy = any(v[0] in TRANSIENT_RUN_STATES for v in self._run.values())
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
        cards that belong to no live run — their last result says nothing about this one.
        A card mid-ingest is exempt; it is reporting work of its own.

        **Plans accumulate, one per live target** (2026-08-06). This used to clear every card
        outside the new plan, which was right when one pull existed at a time and wrong the
        moment a second branch could be queued behind the first: starting branch B wiped
        branch A's cards, so the finished branch stopped saying it was finished and the
        canvas could never show two branches in different states. Now a card is only cleared
        when no live plan claims it, and :meth:`clear_run_plan` retires a plan when its run
        ends. Nodes shared by both branches (the source, the channel taps) sit in both plans
        and survive either one ending, which is what they should do."""
        self._plans[target] = frozenset(node_ids)
        claimed = self.planned_nodes()
        busy = self._ingesting
        for nid in list(self._run):
            if (nid not in busy and nid not in claimed
                    and self._run[nid][0] in TRANSIENT_RUN_STATES):
                self._run.pop(nid, None)
        for nid, item in self.node_items.items():
            if nid in busy:
                continue
            if nid in self._plans[target]:
                # only a card with nothing to say is moved to `queued`: one already `running`
                # or `done` for a still-live plan keeps what it is reporting
                if self._run.get(nid, ("",))[0] in ("", "queued"):
                    self._set_state(nid, "queued")
            elif nid not in claimed:
                # A TERMINAL badge survives (2026-08-06). Clearing every unclaimed card was
                # right when one pull existed at a time — the canvas described "the last run",
                # full stop. With branches it erased the answer the user had just waited for:
                # finish branch A, start branch B, and A's `done` card went blank, which is
                # the "previous nodes stop displaying their progress" report. What A ran is
                # still true; only work in FLIGHT for a run nobody is waiting on is stale, so
                # only that is swept. A stale `done` is retired by the edit that invalidates
                # it (:meth:`clear_run_states_for`), not by an unrelated branch starting.
                if self._run.get(nid, ("",))[0] in TRANSIENT_RUN_STATES:
                    self._run.pop(nid, None)
                    item.set_run_state("")
        self._sync_flows()
        self._sync_anim()

    def clear_run_states_for(self, node_ids: Iterable[str]) -> None:
        """Drop the run badge on ``node_ids`` — an edit made whatever they last reported
        untrue. The counterpart to terminal badges surviving :meth:`set_run_plan`: a `done`
        that outlives the result it describes is worse than no badge at all."""
        for nid in node_ids:
            if nid in self._ingesting:
                continue                      # its own ingest is still reporting
            if self._run.pop(nid, None) is not None:
                item = self.node_items.get(nid)
                if item is not None:
                    item.set_run_state("")
        self._sync_flows()
        self._sync_anim()

    def clear_run_plan(self, target: str) -> None:
        """Retire ``target``'s plan — its run finished, failed or was cancelled.

        The cards keep whatever they last reported (``done``/``error``): that IS the record
        of the finished branch, and it stays on screen until a run that actually claims those
        nodes replaces it. Only the plan's CLAIM goes, so the next :meth:`set_run_plan` is
        free to clear them."""
        self._plans.pop(target, None)

    def planned_nodes(self) -> frozenset:
        """Every node claimed by a live run plan — the union across branches."""
        return frozenset().union(*self._plans.values()) if self._plans else frozenset()

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
        if node_id is not None:
            self.clear_run_plan(node_id)          # this run no longer claims anything
        # Sweeping the unreached cards is scoped to the nodes NO other live run still wants
        # (2026-08-06). Unscoped, the first branch to land wiped the branch queued behind it —
        # its cards went blank while it was still going to run, which reads as "nothing is
        # happening" at exactly the moment the user is waiting to be told otherwise.
        still_claimed = self.planned_nodes()
        for nid, item in self.node_items.items():
            if nid in self._ingesting or nid in still_claimed:
                continue          # its own ingest / another branch's run — not this pull's
            if self._run.get(nid, ("",))[0] in TRANSIENT_RUN_STATES:
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
        self._plans.clear()
        for item in self.node_items.values():
            item.set_run_state("")
        self._sync_flows()
        self._sync_anim()

    def set_queued(self, target: str, node_ids: Iterable[str]) -> None:
        """A pull was QUEUED behind one already running: claim its nodes and mark the ones
        that are otherwise idle ``queued``, without disturbing the run in progress.

        The visible difference between "the app ignored my second click" and "your second
        branch is lined up and will start when this one lands" — which, before the queue
        existed, it genuinely did not do."""
        self._plans[target] = frozenset(node_ids)
        for nid in self._plans[target]:
            if nid in self._ingesting or nid not in self.node_items:
                continue
            if self._run.get(nid, ("",))[0] == "":
                self._set_state(nid, "queued")
        self._sync_flows()
        self._sync_anim()

    # ── deleting ──────────────────────────────────────────────────────────────
    def _on_delete_requested(self, node_id: str) -> None:
        """The hover ✕ badge: delete that node — plus the rest of the selection if the
        clicked card is part of a multi-node selection (what a user expects when they
        have five cards selected and hit the ✕ on one of them)."""
        sel = [i.node_id for i in self.selectedItems() if isinstance(i, NodeItem)]
        self.delete_nodes(sel if node_id in sel and len(sel) > 1 else [node_id])

    @_needs_topology
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

    @_needs_topology
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
        # the new wires FIRST, the old one last: a single input is replaced by the second
        # connect, so at no step is the chain cut — which a linked page sending its edits to
        # the master needs (each step must leave the master computing the same, V4.00 11e)
        try:
            self.doc.connect(s, ss, node_id, di)
        except ValueError:
            return False
        try:
            self.doc.connect(node_id, do, d, ds)
        except ValueError:
            # extremely unlikely after the pre-check, but never leave a half splice
            self.doc.disconnect(s, ss, node_id, di)
            return False
        if (s, ss, d, ds) in self.doc.edges:          # a multi input keeps it: drop it
            self.doc.disconnect(s, ss, d, ds)
        return True

    # ── wire dragging (G1) ────────────────────────────────────────────────────
    @_needs_topology(restart=True)
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
                   for spec, sock in compatible_ops(fixed.spec, fixed.io,
                                                    getattr(self.doc, "page_kind", None))]
        if fixed.io == "out":
            # a dataset dragged into empty canvas on a page that feeds later pages: naming
            # it for them is the first offer (V4.00 step 11); the rest keep their order
            entries.sort(key=lambda e: 0 if e[1] == PAGE_OUTPUT_OP else 1)
        if not entries:
            return
        views = self.views()
        if not views:
            return
        # the view the drag ended in: two canvases can show one page (V4.00 step 5)
        under = QApplication.widgetAt(screen_pos) if screen_pos is not None else None
        view = next((v for v in views
                     if under is not None and (under is v or v.isAncestorOf(under))),
                    views[0])

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
    def _mute_selection(self) -> None:
        """M: switch the selected nodes off — or back on."""
        self.toggle_muted([it.node_id for it in self.selectedItems()
                           if isinstance(it, NodeItem)])

    def toggle_muted(self, node_ids) -> None:
        """Switch each node off, or back on (V4.00 step 11e). Only a node that keeps the kind
        of data can be switched off (:func:`~nodelab_v2.document.pass_through_reason`) — the
        others stay on and the status bar says why. On a linked page this is the page's own
        setting, like a value, so it never asks how to apply."""
        refused = []
        for nid in node_ids:
            rec = self.doc.nodes.get(nid)
            if rec is None:
                continue
            try:
                self.doc.set_muted(nid, not rec.muted)
            except ValueError as exc:
                refused.append(str(exc))
        if refused:
            more = f" (and {len(refused) - 1} more)" if len(refused) > 1 else ""
            self.topology_refused.emit(refused[0] + more)

    def _lock_structural(self, menu: QMenu) -> None:
        """On a linked page whose edits are not settled yet, the menu's structural entries say
        what choosing them does: ask how the change applies (V4.00 step 11e; they were greyed
        out before)."""
        if getattr(self.doc, "editable_topology", True) or getattr(self.doc, "edit_mode", ""):
            return
        from nodelab_v2.linked_document import ASK_HINT
        for act in menu.actions():
            if act.text().startswith(_STRUCTURAL_MENU):
                act.setToolTip(ASK_HINT)

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
        self._lock_structural(menu)
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
        cmp_act = menu.addAction("Compare beside viewed\tF8")
        cmp_act.setToolTip(
            "Open this node's result in a second Viewer pane, side by side with the one "
            "being viewed. When both results span the same M/T/Z, one set of sliders "
            "drives both panes; otherwise each pane keeps its own.")
        cmp_act.triggered.connect(lambda: self.compare_requested.emit(nid))
        if node.op_key == LOAD_OP and str((rec.params.get("path") if rec else "") or ""):
            act = menu.addAction("Ingest this file now")
            act.setToolTip("Write this file's .b2nd store now, on its own worker. Other "
                           "source cards can ingest at the same time and the app stays "
                           "usable.")
            act.triggered.connect(lambda: self.ingest_requested.emit(nid))
        if node.op_key == UNBATCH_OP:
            n_un = len(self.doc.batch_member_names(nid))
            act = menu.addAction(f"Fan out to {n_un} cards" if n_un >= 2
                                 else "Fan out to cards")
            act.setEnabled(n_un >= 2)
            act.setToolTip(
                "Give every file in the batch its own Viewer card, wired to that file's "
                "output. Cards already wired to a member are left alone, so this is safe "
                "to run again after adding files to the batch.")
            act.triggered.connect(lambda: self.fan_out_batch(nid))
        if node.op_key == PAGE_OUTPUT_OP:
            oname = str((rec.params.get(PAGE_NAME_KEY) if rec is not None else "") or "").strip()
            act = menu.addAction("New page from this output…")
            act.setEnabled(bool(oname))
            act.setToolTip("Add a page of the next kind whose Page Input reads this Output — "
                           "empty, from a page recipe, or linked to a master page."
                           if oname else "Give this Output a Name first")
            act.triggered.connect(lambda: self.new_page_from_output.emit(nid))
        mute = menu.addAction("Muted (pass through)\tM")
        mute.setCheckable(True)
        mute.setChecked(bool(rec.muted) if rec is not None else False)
        # only a node that keeps the kind of data can be switched off (V4.00 step 11e)
        why = "" if rec is None or rec.muted else self.doc.pass_through_reason(nid)
        if why:
            mute.setEnabled(False)
            mute.setToolTip(f"Cannot be switched off: {why}")
        mute.triggered.connect(lambda: self.toggle_muted([i for i in sel if i in self.doc.nodes]))
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
            # Hold FIRST, and always enabled: it is the cheap, reversible, precision-free
            # action, so it belongs where the cursor already is. Bake follows as the durable
            # (slower, disk-costing) alternative.
            hold = menu.addAction("Re-hold this dock" if status in ("held", "released")
                                  else "Hold this dock (in memory, instant)")
            hold.setToolTip("Freeze what the chain above last produced in memory and stop "
                            "evaluating it. Writes nothing. Does not free memory and does "
                            "not survive reopening the file — Bake does both.")
            hold.triggered.connect(lambda: self.dock_action.emit(nid, "hold"))
            bake = menu.addAction("Re-bake this dock" if baked else "Bake this dock…")
            bake.setEnabled(not unset)
            if unset:
                bake.setToolTip("Choose a Precision in the inspector first — there is no "
                                "default, because the right one depends on this chain.")
            bake.triggered.connect(lambda: self.dock_action.emit(nid, "bake"))
            if status == "held":
                menu.addAction("Release (run the chain live)").triggered.connect(
                    lambda: self.dock_action.emit(nid, "release"))
            if status in ("docked", "stale"):
                menu.addAction("Un-dock (run the chain live)").triggered.connect(
                    lambda: self.dock_action.emit(nid, "undock"))
            elif baked and status == "live":
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

    @_needs_topology
    def fan_out_batch(self, unbatch_id: str) -> list:
        """Give every file of the batch its own card, wired to that file's output.

        The closing gesture of the golden point: the pipeline ran once over K files and
        this is where the K results become K things you can look at. Each card is a
        ``view.viewer`` — an inspection tap — because what the user wants at the end of a
        batch is to SEE each file's result, and a viewer is the card that shows one.

        **Idempotent by wiring, not by a flag.** A member whose socket already feeds
        something is skipped, so running it again after adding two files to the batch adds
        two cards rather than duplicating the ones already there — and a card the user
        deleted on purpose stays deleted until they ask again. That is also why this is an
        explicit action instead of firing on every rewire: spawning cards nobody asked for,
        repeatedly, is worse than one menu click.
        """
        names = self.doc.batch_member_names(unbatch_id)
        if len(names) < 2:
            return []
        item = next((i for i in self.items()
                     if isinstance(i, NodeItem) and i.node_id == unbatch_id), None)
        base = item.scenePos() if item is not None else QPointF(0.0, 0.0)
        wired = {ss for src, ss, _d, _ds in self.doc.edges if src == unbatch_id}
        made = []
        for i, _name in enumerate(names):
            sock = f"bat{i}"
            if sock in wired:
                continue                       # this member already goes somewhere
            rec = self.doc.add_node(
                "view.viewer",
                x=base.x() + 220.0, y=base.y() + (i - (len(names) - 1) / 2.0) * 150.0)
            try:
                self.doc.connect(unbatch_id, sock, rec.id, "data")
            except ValueError:
                self.doc.remove_node(rec.id)   # socket gone (the batch shrank) — no card
                continue
            made.append(rec.id)
        return made

    @_needs_topology(restart=True)
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
            self._mute_selection()
            e.accept()
            return
        if e.key() == Qt.Key_C:
            for it in self.selectedItems():
                if isinstance(it, NodeItem):
                    self.doc.set_collapsed(it.node_id, not it.rec.collapsed)
            e.accept()
            return
        super().keyPressEvent(e)

    @_needs_topology
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

    @_needs_topology
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
    #: ``(paths, scene position, target node id or "")`` — image files dropped from the
    #: DESKTOP onto the canvas (V3.01). The target is the id of the node the drop landed
    #: on when that node is a Batch point, else ``""``; the window decides what to build.
    files_dropped = Signal(list, QPointF, str)
    maximize_toggled = Signal(bool)
    #: a press on the canvas (or a drop onto it): the user is working in THIS canvas — the
    #: window makes its page the active one before the press does anything (V4.00 step 5)
    pressed = Signal()

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
        # beside it: fit the view to the nodes (V4.00 step 11d) — what Home does
        self._fit_btn = HudButton("fit", self)
        self._fit_btn.setToolTip("Fit the view to the nodes (Home)")
        self._fit_btn.clicked.connect(lambda _=False: self.fit_all())
        # the page switcher (V4.00 step 5): which page this canvas shows, and the menu that
        # changes it — top-left, where the troubleshooting badge steps aside for it
        self.page_button = QToolButton(self)
        self.page_button.setObjectName("pageSwitch")
        self.page_button.setPopupMode(QToolButton.InstantPopup)
        self.page_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.page_button.setCursor(Qt.PointingHandCursor)
        self.page_button.setToolTip(
            "The page this canvas shows. Click for every page of the workspace, grouped by "
            "kind, and to add, duplicate, rename or delete one (Ctrl+PgUp / Ctrl+PgDn step "
            "through them).")
        self.page_button.hide()                     # until the window names a page
        self._canvas_active = False
        self._style_page_button()
        self._place_corner_chrome()

    # ── corner chrome (maximize, fit to nodes, the troubleshooting badge) ──────
    def _place_corner_chrome(self) -> None:
        self._max_btn.move(self.width() - self._max_btn.width() - 12, 12)
        self._max_btn.raise_()
        self._fit_btn.move(self._max_btn.x() - self._fit_btn.width() - 6, 12)
        self._fit_btn.raise_()
        pb = getattr(self, "page_button", None)
        if pb is not None and pb.text():
            pb.adjustSize()
            pb.move(12, 12)
            pb.raise_()
        self._place_ts_badge()

    # ── the page switcher (V4.00 step 5) ──────────────────────────────────────
    def set_page_title(self, text: str, icon=None) -> None:
        """Name the page this canvas shows on its switcher (with its kind's dot)."""
        self.page_button.setText(f" {text}")
        if icon is not None:
            self.page_button.setIcon(icon)
        self.page_button.show()
        self._place_corner_chrome()

    def set_canvas_active(self, on: bool) -> None:
        """An accent on the switcher while this is the canvas the user works in — what
        tells two canvases apart once there are several."""
        self._canvas_active = bool(on)
        self._style_page_button()

    def _style_page_button(self) -> None:
        edge = T.ACCENT if self._canvas_active else T.BORDER
        self.page_button.setStyleSheet(
            f"QToolButton#pageSwitch {{ background:{T.PANEL.name()}; color:{T.INK.name()}; "
            f"border:1px solid {edge.name()}; border-radius:6px; "
            f"padding:3px 18px 3px 6px; font-size:11px; font-weight:700; }}"
            f"QToolButton#pageSwitch:hover {{ background:{T.PANEL_HI.name()}; }}")

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
        self._style_page_button()
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
        self.pressed.emit()                 # this canvas becomes the one worked in, first
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
        # the view scrolls only inside the scene rect, which starts as a fixed area round the
        # origin: a card placed beyond it was scaled into view but could not be centred on,
        # and stayed off screen (V4.00 step 11d, the fit-to-nodes button) — so it grows first
        sc = self.scene()
        want = r.adjusted(-4000, -4000, 4000, 4000)
        if not sc.sceneRect().contains(want):
            sc.setSceneRect(sc.sceneRect().united(want))
        self.fitInView(r.adjusted(-70, -70, 70, 70), Qt.KeepAspectRatio)

    # palette drag-and-drop (G2) + desktop file drop (V3.01)
    def _dropped_image_paths(self, md) -> list:
        """Local image files in a drag's mime data, in the order the OS listed them.

        Filtered by extension rather than accepting any URL, so dragging a folder, a URL
        from a browser or a stray text file is simply not our drop and falls through to
        Qt — a drag that LOOKS accepted and then does nothing is worse than one the
        cursor never offered to take.
        """
        if not md.hasUrls():
            return []
        out = []
        for u in md.urls():
            if not u.isLocalFile():
                continue
            p = u.toLocalFile()
            if p.lower().endswith(FILE_DROP_SUFFIXES):
                out.append(p)
        return out

    def _batch_node_at(self, view_pt) -> str:
        """The id of the Batch point under ``view_pt``, or ``""``.

        Only ``util.batch`` answers: dropping files onto the point that COLLECTS them is
        the gesture with an obvious meaning, and there is none for dropping a file on an
        Unbatch or on an ordinary card.
        """
        for it in self.items(view_pt):
            nid = getattr(it, "node_id", None)
            if nid is None and getattr(it, "parentItem", None) is not None:
                nid = getattr(it.parentItem(), "node_id", None)
            if nid is None:
                continue
            rec = self.scene().doc.nodes.get(nid) if self.scene() else None
            if rec is not None and rec.op_key == BATCH_OP:
                return str(nid)
        return ""

    def dragEnterEvent(self, e) -> None:
        if (e.mimeData().hasFormat("application/x-nd2studios-op")
                or self._dropped_image_paths(e.mimeData())):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e) -> None:
        if (e.mimeData().hasFormat("application/x-nd2studios-op")
                or self._dropped_image_paths(e.mimeData())):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e) -> None:
        self.pressed.emit()                 # a drop lands on THIS canvas's page
        if e.mimeData().hasFormat("application/x-nd2studios-op"):
            op = bytes(e.mimeData().data("application/x-nd2studios-op")).decode("utf-8")
            self.op_dropped.emit(op, self.mapToScene(e.position().toPoint()))
            e.acceptProposedAction()
            return
        paths = self._dropped_image_paths(e.mimeData())
        if paths:
            pt = e.position().toPoint()
            self.files_dropped.emit(paths, self.mapToScene(pt), self._batch_node_at(pt))
            e.acceptProposedAction()
            return
        super().dropEvent(e)


__all__ = ["GraphScene", "GraphView", "LinkSearchPopup", "compatible_ops",
           "visible_specs", "HIDDEN_OP_PREFIXES"]
