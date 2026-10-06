"""The Workspace — ND2Studios V4.00's pages (2026-10-05): several node graphs in one file,
each a :class:`~nodelab_v2.document.GraphDocument` of a typed KIND, wired to one another by
named outputs.

Qt-free (the selftest reaches it through the document seam, exactly as it reaches
:mod:`nodelab_v2.document`).

Pages and kinds
---------------
A :class:`Page` is one graph plus a name and a kind. The kinds and their order come from
``codemap/node_roles.json`` through :mod:`nodegraph.roles`: ``input`` (bring files on, split
positions, name a condition) → ``refine`` (clean, align, segment) → ``process`` (measure,
track, fields) → ``analyze`` (overlay, plot, export). The fifth kind, ``free``, is the absence
of a filter — every node, any wiring — and is what a pre-V4 single-graph file opens as.

Named outputs
-------------
A ``page.output`` node names the Dataset wired into it as a VARIABLE of its page. A
``page.input`` node on a page of a strictly LATER kind reads one by ``"<page_id>:<name>"``
(:meth:`Workspace.available_sources` lists what a page may read; a ``free`` page may read from
or feed any page as long as the page graph stays acyclic). At edit time the Input's envelope is
the upstream Output's (:attr:`GraphDocument.seed_hooks`); at run time the Input disappears.

Composition and the memo
------------------------
:meth:`Workspace.compose` builds ONE run graph for a page: every page it reads from, spliced in
upstream-first, with every node id qualified ``"<page_id>/<node_id>"`` — the target page's own
nodes included — and each resolved Input replaced by a wire from the upstream Output. A node's
recipe hash contains no node id except a root's ``__source__``, and that carries the OWNING
page's prefix, so the same refinement chain pulled from its own page and from three processing
pages is one memo entry; and the memo's per-node bookkeeping (``_last_fp``, ``drop_nodes``) can
never confuse two pages' ``n3``. The ids ``pg1/n3#it@2`` (an unrolled iterate clone) and
``pg1/b%inst`` (an expanded group body) keep their inner grammar: the page prefix is always the
first ``/``-separated part (:func:`split_run_id`).

File format
-----------
:meth:`Workspace.to_dict` writes :data:`nodegraph.serialize.WORKSPACE_FORMAT_VERSION` (3.0):
``{format_version, app_version, workspace: {active, next_page_seq, pages: [...]}}``, one
single-graph body per page (``graph``/``zones``/``groups``/``ui`` + ``id``/``name``/``kind``).
:meth:`Workspace.load_dict` reads that and a 2.0 single-graph document (one ``free`` page). Page
ids are ``pg1, pg2, …`` from a persisted counter and are never re-minted, so a stale Input
reference can never silently rebind to a new page.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Protocol, Tuple)

from nodegraph import roles as R
from nodegraph.graph import Graph, NodeInstance
from nodegraph.iterate import OWNER_KEY
from nodegraph.memo import digest
from nodegraph.metadata import MetaEnvelope
from nodegraph.serialize import (
    is_workspace_dict, page_from_dict, to_workspace_dict, workspace_pages)
from nodelab_v2.document import GraphDocument, NodeRecord
from nodelab_v2.ops import (
    PAGE_CONDITION_KEY, PAGE_INPUT_OP, PAGE_NAME_KEY, PAGE_OUTPUT_OP, PAGE_SOURCE_KEY,
    is_frozen, upstream_signature)
from nodelab_v2.version import __version__

#: Separates the page id from the node id in a RUN id: ``pg1/n3``. Not ``#``, ``@`` or ``%``,
#: which the iterate unroll and the group expansion already use INSIDE a node id.
RUN_SEP = "/"
#: What a page id may look like — no run-id grammar characters, no whitespace.
PAGE_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
#: The kind a pre-V4 file opens as (re-exported from :mod:`nodegraph.roles`).
FREE = R.FREE_PAGE
#: Default name of the one page a fresh workspace holds.
DEFAULT_PAGE_NAME = "Graph"


def qualify(page_id: str, node_id: str) -> str:
    """``"<page_id>/<node_id>"`` — a node's identity in a composed run graph."""
    return f"{page_id}{RUN_SEP}{node_id}"


def split_run_id(run_id: str) -> Tuple[str, str]:
    """``("pg1", "n3#it@2")`` for a qualified run id; ``("", run_id)`` for a bare one."""
    if RUN_SEP in run_id:
        pid, nid = run_id.split(RUN_SEP, 1)
        return pid, nid
    return "", run_id


def doc_id_of(run_id: str) -> str:
    """The DOCUMENT node a run-graph id answers to, page prefix kept (V4.00 step 2).

    The run graph renames two kinds of node: an Iterate clone is ``{doc_id}#{iterate_id}@{i}``
    (and the zone's synthetic advance node ``{iterate_id}#adv@{i}`` belongs to the Iterate
    card itself), and an inlined group body node is ``{body_id}%{instance_id}`` — nesting
    appends further ``%instance`` segments, and the LAST one is the instance that actually
    sits on the canvas. So ``"pg1/n3#it@2"`` → ``"pg1/n3"``, ``"pg1/it#adv@0"`` →
    ``"pg1/it"``, ``"pg1/body%inst"`` → ``"pg1/inst"``; a bare id follows the same rules
    without a prefix. The runner stores run cones in these terms, so a page's qualified
    ``last_touched`` matches the runs computing that node's clones."""
    pid, nid = split_run_id(run_id)
    head = nid.split("#", 1)[0]
    base = head.rsplit("%", 1)[-1] if "%" in head else head
    return qualify(pid, base) if pid else base


def local_ids(ids: Iterable[str], page_id: str,
              default_page: Optional[str] = None) -> FrozenSet[str]:
    """The BARE node ids among ``ids`` that belong to ``page_id``: every qualified id whose
    page is ``page_id``, plus every bare id when bare means ``page_id`` — i.e. when
    ``default_page`` is ``page_id`` (or ``None``, "bare ids are this page's")."""
    out = set()
    for rid in ids:
        p, n = split_run_id(str(rid))
        if p == page_id or (not p and (default_page is None or default_page == page_id)):
            out.add(n)
    return frozenset(out)


def kind_label(kind: str) -> str:
    return str(R.page_meta(kind).get("label") or kind)


@dataclass
class Page:
    """One node graph in the workspace."""
    id: str
    name: str
    kind: str
    doc: GraphDocument
    #: The page this one is LINKED to (same nodes, its own parameter values) — V4 step 6.
    #: ``None`` for a plain page.
    master: Optional[str] = None
    #: A linked page's parameter/mode overrides, ``{node_id: {"params": {...}, "modes":
    #: {...}}}``. Carried through the file for a plain page too (always empty).
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ComposedGraph:
    """The run graph of one page with every page it reads from spliced in."""
    graph: Graph
    page_id: str
    #: The dependency closure, upstream first, the target page last.
    pages: Tuple[str, ...]
    #: Every page's ``meta_seeds``, under qualified ids.
    meta_seeds: Dict[str, MetaEnvelope]
    #: ``(page_id, document node id) -> run id`` — a resolved ``page.input`` maps to the
    #: UPSTREAM Output's run id (it has no node of its own in the composed graph).
    id_map: Dict[Tuple[str, str], str]
    #: :meth:`Workspace.revision_of` at composition — changes when any page in the closure
    #: changes, so the runner rebuilds its engine exactly when it must.
    revision: str
    #: every RESOLVED ``page.input`` → the run ids that read it. Such an Input has no node in
    #: the composed graph (its consumers are wired to the upstream Output), so no run's plan
    #: names it; the runner adds it to the cone of any run that plans one of its consumers,
    #: or an edit to its Source mid-run would cancel nothing (V4.00 step 2).
    inputs: Dict[str, FrozenSet[str]] = field(default_factory=dict)


class GraphSource(Protocol):
    """What the runner needs from whatever hands it graphs (V4 step 2 binds
    :class:`~nodelab_v2.runner.EngineRunner` to this instead of to one document)."""

    active: Optional[str]
    last_touched: Optional[frozenset]

    def compose(self, page_id: str, *, live_docks: frozenset = frozenset(),
                sweep_all: frozenset = frozenset()) -> ComposedGraph: ...
    def revision_of(self, page_id: str) -> str: ...
    def record(self, run_id: str) -> Optional[NodeRecord]: ...
    def env(self, run_id: str) -> MetaEnvelope: ...
    def meta_seed(self, run_id: str) -> Optional[MetaEnvelope]: ...
    def set_meta_seed(self, run_id: str, env: MetaEnvelope) -> None: ...
    def document_of(self, page_id: str) -> GraphDocument: ...
    def page_ids(self) -> Tuple[str, ...]: ...
    def all_records(self) -> Iterable[Tuple[str, NodeRecord]]: ...
    def iterate_aliases(self, page_id: str, *,
                        sweep_all: frozenset = frozenset()) -> Dict[str, str]: ...
    def on_change(self, fn: Callable[[], None]) -> None: ...
    def off_change(self, fn: Callable[[], None]) -> None: ...


class Workspace:
    """Ordered pages, one of them active; the page graph; composition for a run."""

    def __init__(self) -> None:
        #: insertion order is the page order the editor shows
        self.pages: Dict[str, Page] = {}
        self.active: Optional[str] = None
        #: the workspace file (``*.nd2graph.json``, format 3.0); every page's document
        #: resolves its relative dock stores against it
        self.path: Optional[str] = None
        #: bumped on every page edit and every page-level change (add/remove/rename/move)
        self.revision = 0
        #: the QUALIFIED run ids the most recent page edit touched (``None`` = unknown /
        #: everything, ``frozenset()`` = nothing a run can see), published for listeners
        #: exactly as :attr:`GraphDocument.last_touched` is
        self.last_touched: Optional[frozenset] = None
        #: the counter page ids are minted from; persisted, never rewound
        self.next_page_seq = 1
        self._listeners: List[Callable[[], None]] = []
        #: per page: the document listener and seed hook installed by :meth:`_attach`
        self._hooks: Dict[str, Tuple[Callable[[], None], Callable[[], Mapping]]] = {}
        #: True while this object itself is driving document changes (a load, a cascade),
        #: so the per-page listener does not re-enter
        self._quiet = False
        #: each page's ``page.output`` node ids as of its last change — so a DELETED Output
        #: is still recognized as one when its page publishes the change
        self._out_ids: Dict[str, FrozenSet[str]] = {}

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def single(cls, doc: GraphDocument, *, kind: str = FREE,
               name: str = DEFAULT_PAGE_NAME) -> "Workspace":
        """A workspace of one page around an EXISTING document — how the window wraps the
        document its canvas is bound to."""
        ws = cls()
        ws.add_page(name, kind, doc=doc)
        return ws

    # ── listeners ─────────────────────────────────────────────────────────────
    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def off_change(self, fn: Callable[[], None]) -> None:
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    def _fire(self) -> None:
        for fn in list(self._listeners):
            fn()

    def _notify(self, touched: Optional[Iterable[str]] = None) -> None:
        """A page-level change (not an edit inside a document): bump, publish, fire."""
        self.revision += 1
        self.last_touched = None if touched is None else frozenset(touched)
        self._fire()

    # ── pages ─────────────────────────────────────────────────────────────────
    def new_page_id(self) -> str:
        pid = f"pg{self.next_page_seq}"
        self.next_page_seq += 1
        while pid in self.pages:                      # a loaded file with a stale counter
            pid = f"pg{self.next_page_seq}"
            self.next_page_seq += 1
        return pid

    def page(self, page_id: str) -> Page:
        return self.pages[page_id]

    def page_id_for(self, ref: str) -> Optional[str]:
        """A page id from an id or a name (exact, then case-insensitive); ``None`` if none."""
        if ref in self.pages:
            return ref
        for p in self.pages.values():
            if p.name == ref:
                return p.id
        low = ref.lower()
        for p in self.pages.values():
            if p.name.lower() == low:
                return p.id
        return None

    def document_of(self, page_id: str) -> GraphDocument:
        return self.pages[page_id].doc

    def page_ids(self) -> Tuple[str, ...]:
        """Every page id, in page order."""
        return tuple(self.pages)

    def all_records(self) -> Iterable[Tuple[str, NodeRecord]]:
        """``(run id, record)`` for every node on every page, page order then node order —
        how the runner enumerates the sources of a whole workspace."""
        for pid, page in self.pages.items():
            for rec in page.doc.nodes.values():
                yield qualify(pid, rec.id), rec

    def add_page(self, name: str, kind: str = FREE, *, doc: Optional[GraphDocument] = None,
                 page_id: Optional[str] = None, index: Optional[int] = None) -> Page:
        """A new plain page. ``doc`` adopts an existing document (the window's); ``index``
        places it in the page order (default: last)."""
        if not R.is_page_kind(kind):
            raise ValueError(f"unknown page kind {kind!r} (kinds: {', '.join(R.page_kinds())})")
        pid = page_id or self.new_page_id()
        if not PAGE_ID_RE.match(pid):
            raise ValueError(f"page id {pid!r} may contain only letters, digits and _")
        if pid in self.pages:
            raise ValueError(f"duplicate page id {pid!r}")
        name = self._unique_name(name or kind_label(kind))
        page = Page(pid, name, kind, doc or GraphDocument())
        if doc is None:
            page.doc.path = self.path
        self._insert(page, index)
        self._attach(page)
        if self.active is None:
            self.active = pid
        # a new page is read by nothing — its id was never minted before — so no run can
        # see it: "nothing", not "unknown", or adding a page would cancel every pull in flight
        self._notify(())
        return page

    def remove_page(self, page_id: str) -> None:
        """Drop a page. Inputs on other pages that read from it become unbound (and say so
        in the inspector); they are never re-pointed. A master with linked pages is refused
        here — the window asks, then makes them unique first (V4 step 6)."""
        page = self.pages[page_id]
        dependents = [p.id for p in self.pages.values() if p.master == page_id]
        if dependents:
            raise ValueError(
                f"page {page.name!r} is the master of {len(dependents)} linked page(s); make "
                f"them unique first")
        # its readers are found BEFORE it goes: once deleted it is no page's dependency, and
        # the cascade would re-describe nobody — leaving their Inputs on its old envelopes
        readers = self._downstream_of(page_id)
        gone = {qualify(page_id, n) for n in page.doc.nodes} | self._reader_inputs(page_id)
        self._detach(page)
        del self.pages[page_id]
        if self.active == page_id:
            self.active = next(iter(self.pages), None)
        self._cascade(page_id, readers)
        # every node it had: exactly the runs that read the page (their cones hold its ids)
        self._notify(gone)

    def rename_page(self, page_id: str, name: str) -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("a page needs a name")
        page = self.pages[page_id]
        if any(p.id != page_id and p.name.lower() == name.lower() for p in self.pages.values()):
            raise ValueError(f"another page is already called {name!r}")
        if page.name == name:
            return
        page.name = name
        # the name is stamped into the Outputs' `condition` at compose time, so a rename
        # changes what reads THROUGH an Output — and nothing else on the page: touch the
        # Outputs and the Inputs that read them
        self._cascade(page_id)
        self._notify({qualify(page_id, n) for n in self._output_ids(page_id)}
                     | self._reader_inputs(page_id))

    def move_page(self, page_id: str, index: int) -> None:
        page = self.pages.pop(page_id)
        self._insert(page, index)
        self._notify(())

    def set_active(self, page_id: str) -> None:
        if page_id not in self.pages:
            raise KeyError(page_id)
        if self.active == page_id:
            return
        self.active = page_id
        self.last_touched = frozenset()
        self._fire()

    def duplicate_page(self, page_id: str, *, dependent: bool = False,
                       name: Optional[str] = None) -> Page:
        """A copy of ``page_id`` placed right after it: a UNIQUE copy (its own nodes and
        values) or — V4 step 6 — a LINKED one (the master's nodes, its own values)."""
        if dependent:
            raise NotImplementedError("linked pages arrive in V4 step 6")
        src = self.pages[page_id]
        order = list(self.pages)
        page = self.add_page(name or f"{src.name} copy", src.kind,
                             index=order.index(page_id) + 1)
        self._quiet = True
        try:
            page.doc.load_page(src.doc.to_page_dict())
            page.doc.meta_seeds.update(src.doc.meta_seeds)
            page.doc.repropagate()
        finally:
            self._quiet = False
        self._out_ids[page.id] = self._output_ids(page.id)
        self._notify(())                    # a new page: read by nothing yet
        return page

    def reset(self) -> None:
        """File → New: back to ONE empty Free page named "Graph", keeping the active page's
        document object (the canvas is bound to it)."""
        keep = self.pages[self.active] if self.active in self.pages else None
        for pid in list(self.pages):
            if keep is None or pid != keep.id:
                self._detach(self.pages[pid])
                del self.pages[pid]
        self.path = None
        if keep is None:
            self.add_page(DEFAULT_PAGE_NAME, FREE)
            return
        keep.name, keep.kind, keep.master, keep.overrides = DEFAULT_PAGE_NAME, FREE, None, {}
        keep.doc.page_kind = FREE
        self.active = keep.id
        keep.doc.clear()                 # notifies → the page listener → our listeners

    def _insert(self, page: Page, index: Optional[int]) -> None:
        order = list(self.pages.values())
        if index is None or index >= len(order):
            order.append(page)
        else:
            order.insert(max(0, index), page)
        self.pages = {p.id: p for p in order}

    def _unique_name(self, name: str) -> str:
        taken = {p.name.lower() for p in self.pages.values()}
        if name.lower() not in taken:
            return name
        n = 2
        while f"{name} {n}".lower() in taken:
            n += 1
        return f"{name} {n}"

    # ── a document becomes a page ─────────────────────────────────────────────
    def _attach(self, page: Page) -> None:
        doc, pid = page.doc, page.id
        doc.page_kind = page.kind
        doc.store_tag = pid
        doc.page_sources = lambda pid=pid: self.available_sources(pid)
        doc.cross_page_signature = lambda nid, pid=pid: self.cross_page_signature(pid, nid)
        doc.workspace_revision = lambda pid=pid: self.revision_of(pid)

        def hook(pid=pid) -> Mapping[str, MetaEnvelope]:
            return self._input_seeds(pid)

        def listener(pid=pid) -> None:
            self._on_page_change(pid)

        doc.seed_hooks.append(hook)
        doc.on_change(listener, first=True)      # before the canvas, the editors, anyone
        self._hooks[pid] = (listener, hook)
        self._out_ids[pid] = self._output_ids(pid)

    def _detach(self, page: Page) -> None:
        doc = page.doc
        listener, hook = self._hooks.pop(page.id, (None, None))
        if listener is not None:
            doc.off_change(listener)
        if hook is not None and hook in doc.seed_hooks:
            doc.seed_hooks.remove(hook)
        self._out_ids.pop(page.id, None)
        doc.page_sources = lambda: []
        doc.cross_page_signature = lambda _nid: ""
        doc.workspace_revision = lambda: ""
        doc.store_tag = ""
        doc.page_kind = None

    def _on_page_change(self, page_id: str) -> None:
        if self._quiet or page_id not in self.pages:
            return
        doc = self.pages[page_id].doc
        self.revision += 1
        lt = doc.last_touched
        outs_before = self._out_ids.get(page_id, frozenset())
        outs_now = self._output_ids(page_id)
        self._out_ids[page_id] = outs_now
        if lt is None:
            self.last_touched = None
        else:
            touched = {qualify(page_id, n) for n in lt}
            # An Output edited, added or deleted can RE-BIND an Input on another page (a
            # name now matches, or no longer does) — a run reading through that Input has
            # the Input in its cone but not necessarily this Output, so name the Inputs too.
            if set(lt) & (outs_before | outs_now):
                touched |= self._reader_inputs(page_id)
            self.last_touched = frozenset(touched)
        self._cascade(page_id)
        self._fire()

    def _output_ids(self, page_id: str) -> FrozenSet[str]:
        page = self.pages.get(page_id)
        if page is None:
            return frozenset()
        return frozenset(r.id for r in page.doc.nodes.values() if r.op_key == PAGE_OUTPUT_OP)

    def _reader_inputs(self, page_id: str) -> set:
        """The run ids of every ``page.input`` on another page that names ``page_id``."""
        out = set()
        for q, page in self.pages.items():
            if q == page_id:
                continue
            for rec in page.doc.nodes.values():
                if rec.op_key == PAGE_INPUT_OP and \
                        self.parse_source(rec.params.get(PAGE_SOURCE_KEY))[0] == page_id:
                    out.add(qualify(q, rec.id))
        return out

    def _refresh_out_ids(self) -> None:
        self._out_ids = {pid: self._output_ids(pid) for pid in self.pages}

    def _downstream_of(self, page_id: str) -> List[str]:
        """Every page that reads ``page_id``, transitively, upstream first."""
        affected = {page_id}
        out: List[str] = []
        for q in self._topo_pages():
            if q != page_id and self._page_deps(q) & affected:
                affected.add(q)
                out.append(q)
        return out

    def _cascade(self, page_id: str, pages: Optional[List[str]] = None) -> None:
        """Re-describe every page downstream of ``page_id`` (or exactly ``pages``), upstream
        first: their Inputs' envelopes come from it. Quiet, so the cascade cannot re-enter
        itself."""
        self._quiet = True
        try:
            for q in (self._downstream_of(page_id) if pages is None else pages):
                if q in self.pages:
                    self.pages[q].doc.repropagate()
        finally:
            self._quiet = False

    # ── the page graph ────────────────────────────────────────────────────────
    @staticmethod
    def parse_source(value: Any) -> Tuple[str, str]:
        """``("pg1", "masks")`` from a ``page.input`` source value; ``("", "")`` if malformed."""
        s = str(value or "").strip()
        if ":" not in s:
            return "", ""
        pid, name = s.split(":", 1)
        return pid.strip(), name.strip()

    def _page_deps(self, page_id: str) -> FrozenSet[str]:
        """The pages this page's Inputs REFER to (existing pages only; no validation beyond
        that, so a cycle in a loaded file is detectable rather than recursive)."""
        page = self.pages.get(page_id)
        if page is None:
            return frozenset()
        deps = set()
        for rec in page.doc.nodes.values():
            if rec.op_key != PAGE_INPUT_OP:
                continue
            pid, _name = self.parse_source(rec.params.get(PAGE_SOURCE_KEY))
            if pid and pid != page_id and pid in self.pages and self._kinds_allow(pid, page_id):
                deps.add(pid)
        return frozenset(deps)

    def _kinds_allow(self, up_id: str, page_id: str) -> bool:
        """The ORDER half of :meth:`_may_feed`, which needs no closure: two ordered kinds may
        feed only strictly forward; a ``free`` page on either side defers to acyclicity. A
        reference against the order is no dependency at all — the Input can never resolve,
        so it must not pull the named page into this one's closure (or form a cycle there)."""
        ou = R.page_order(self.pages[up_id].kind)
        op = R.page_order(self.pages[page_id].kind)
        return ou is None or op is None or ou < op

    def page_deps(self, page_id: str) -> List[str]:
        return sorted(self._page_deps(page_id))

    def dependency_closure(self, page_id: str, *, strict: bool = True) -> List[str]:
        """Every page ``page_id`` reads from, transitively, upstream first, ``page_id`` last.
        Raises :class:`ValueError` on a cycle — or, with ``strict=False``, ignores the
        reference that would close it (the run identity and the composition use that: a
        cycle only Free pages can form, from a hand-edited file, must leave the Input on it
        unbound rather than make every pull, cursor move and repaint raise)."""
        order: List[str] = []
        state: Dict[str, int] = {}                    # 1 = visiting, 2 = done

        def visit(pid: str, path: Tuple[str, ...]) -> None:
            st = state.get(pid, 0)
            if st == 2:
                return
            if st == 1:
                if not strict:
                    return
                raise ValueError("pages read from each other in a cycle: "
                                 + " → ".join(path + (pid,)))
            state[pid] = 1
            for dep in sorted(self._page_deps(pid)):
                visit(dep, path + (pid,))
            state[pid] = 2
            order.append(pid)

        visit(page_id, ())
        return order

    def _topo_pages(self) -> List[str]:
        """Every page, upstream first (file order breaks ties; a cyclic file falls back to
        file order so a load can still finish and the inspector can say what is wrong)."""
        done: List[str] = []
        seen = set()
        for pid in self.pages:
            for q in self.dependency_closure(pid, strict=False):
                if q not in seen:
                    seen.add(q)
                    done.append(q)
        return done

    def _may_feed(self, up_id: str, page_id: str) -> bool:
        """May a ``page.input`` on ``page_id`` read from ``up_id``? Strictly earlier kind;
        a ``free`` page on either side instead needs only acyclicity."""
        if up_id == page_id or up_id not in self.pages or page_id not in self.pages:
            return False
        ou = R.page_order(self.pages[up_id].kind)
        op = R.page_order(self.pages[page_id].kind)
        if ou is not None and op is not None:
            return ou < op
        try:
            return page_id not in self.dependency_closure(up_id)
        except ValueError:
            return False

    def outputs_of(self, page_id: str) -> List[Tuple[str, str]]:
        """``[(name, node_id), ...]`` — the NAMED Outputs of a page, by name. Unnamed ones
        are not addressable and are left out."""
        page = self.pages.get(page_id)
        if page is None:
            return []
        out = []
        for rec in page.doc.nodes.values():
            if rec.op_key != PAGE_OUTPUT_OP:
                continue
            name = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
            if name:
                out.append((name, rec.id))
        out.sort(key=lambda t: t[0].lower())
        return out

    def available_sources(self, page_id: str) -> List[Tuple[str, str]]:
        """``[(value, label), ...]`` a ``page.input`` on ``page_id`` may pick: every named
        Output of every page that may feed it, in page order — ``("pg1:raw", "Input · raw")``."""
        out = []
        for up in self.pages.values():
            if not self._may_feed(up.id, page_id):
                continue
            for name, _nid in self.outputs_of(up.id):
                out.append((f"{up.id}:{name}", f"{up.name} · {name}"))
        return out

    def resolve_source(self, page_id: str, value: Any) -> Optional[Tuple[str, str]]:
        """``(upstream page id, Output node id)`` for a source value, or ``None`` when it
        names nothing this page may read (unknown page, unknown name, wrong kind, a cycle)."""
        pid, name = self.parse_source(value)
        if not pid or not name or not self._may_feed(pid, page_id):
            return None
        for n, nid in self.outputs_of(pid):
            if n == name:
                return pid, nid
        return None

    def cross_page_signature(self, page_id: str, node_id: str,
                             _seen: Optional[FrozenSet[str]] = None) -> str:
        """A digest of what ``node_id`` on ``page_id`` reads from OTHER pages: every Page
        Input upstream of it (the walk stops where the run graph stops, at a frozen node), and
        for each the Output it resolves to with that page's own upstream signature — so an
        edit on an upstream page stales a dock fed through it (V4.00 step 5). It also holds
        the page's NAME when a Page Output upstream on this page leaves its condition blank:
        compose stamps the name there, so a rename changes what the dock would serve.
        ``""`` when neither applies: a dock's signature is then exactly what its bake
        recorded."""
        page = self.pages.get(page_id)
        seen_pages = (_seen or frozenset()) | {page_id}
        if page is None:
            return ""
        try:
            g = page.doc.to_graph(bypass_muted=True)
        except Exception:                            # noqa: BLE001 — mid-edit
            return ""
        inputs, stamped, seen = [], [], set()
        stack = [e.src for e in g.preds(node_id)] if node_id in g.nodes else []
        while stack:
            nid = stack.pop()
            if nid in seen or nid not in g.nodes:
                continue
            seen.add(nid)
            node = g.nodes[nid]
            if is_frozen(node):
                continue
            if node.op_key == PAGE_INPUT_OP:
                inputs.append(nid)
            elif node.op_key == PAGE_OUTPUT_OP and not str(
                    (node.params or {}).get(PAGE_CONDITION_KEY, "") or "").strip():
                stamped.append(nid)          # compose fills its condition with page.name
            stack.extend(e.src for e in g.preds(nid))
        if not inputs and not stamped:
            return ""
        parts: list = [("stamped", tuple(sorted(stamped)), page.name)] if stamped else []
        for inp in sorted(inputs):
            res = self.resolve_source(page_id, g.nodes[inp].params.get(PAGE_SOURCE_KEY))
            if res is None or res[0] in seen_pages:
                parts.append((inp, None))
                continue
            up_pid, out_nid = res
            try:
                up_g = self.pages[up_pid].doc.to_graph(bypass_muted=True)
                out = up_g.nodes[out_nid]
                # the condition the Output STAMPS: its own, or — blank — its page's name,
                # filled in at compose time; so renaming that page stales the dock too
                cond = str((out.params or {}).get(PAGE_CONDITION_KEY, "") or "").strip()
                parts.append((inp, up_pid, out_nid, dict(out.params or {}),
                              cond or self.pages[up_pid].name,
                              upstream_signature(up_g, out_nid),
                              self.cross_page_signature(up_pid, out_nid, seen_pages)))
            except Exception:                        # noqa: BLE001 — mid-edit
                parts.append((inp, up_pid, None))
        return digest("cross-page-sig", parts)

    def _input_seeds(self, page_id: str) -> Dict[str, MetaEnvelope]:
        """The seed hook: each resolved ``page.input`` of a page → its upstream Output's
        current envelope."""
        page = self.pages.get(page_id)
        if page is None:
            return {}
        seeds: Dict[str, MetaEnvelope] = {}
        for rec in page.doc.nodes.values():
            if rec.op_key != PAGE_INPUT_OP:
                continue
            res = self.resolve_source(page_id, rec.params.get(PAGE_SOURCE_KEY))
            if res is None:
                continue
            up, out_nid = res
            env = self.pages[up].doc.envs.get(out_nid)
            if env is not None:
                seeds[rec.id] = env
        return seeds

    # ── identity for a run ────────────────────────────────────────────────────
    def revision_of(self, page_id: str) -> str:
        """The run identity of a page: a digest over the (id, name, document revision) of
        every page in its dependency closure. Changes when any of them changes — and when
        a page is renamed, because the name is stamped into its Outputs' ``condition``."""
        if page_id not in self.pages:
            return ""
        closure = self.dependency_closure(page_id, strict=False)
        return digest("ws", tuple((q, self.pages[q].name, self.pages[q].doc.uid,
                                   self.pages[q].doc.revision) for q in closure))

    def record(self, run_id: str) -> Optional[NodeRecord]:
        pid, nid = split_run_id(run_id)
        page = self.pages.get(pid or (self.active or ""))
        return page.doc.nodes.get(nid) if page is not None else None

    def env(self, run_id: str) -> MetaEnvelope:
        pid, nid = split_run_id(run_id)
        page = self.pages.get(pid or (self.active or ""))
        return page.doc.env(nid) if page is not None else MetaEnvelope()

    def meta_seed(self, run_id: str) -> Optional[MetaEnvelope]:
        """The envelope a source node was last seeded with, or ``None``."""
        pid, nid = split_run_id(run_id)
        page = self.pages.get(pid or (self.active or ""))
        return page.doc.meta_seeds.get(nid) if page is not None else None

    def set_meta_seed(self, run_id: str, env: MetaEnvelope) -> None:
        pid, nid = split_run_id(run_id)
        page = self.pages.get(pid or (self.active or ""))
        if page is not None:
            page.doc.set_meta_seed(nid, env)

    def iterate_aliases(self, page_id: str, *,
                        sweep_all: frozenset = frozenset()) -> Dict[str, str]:
        """:meth:`GraphDocument.iterate_aliases` for one page, in run ids: ``{"pg1/n3":
        "pg1/n3#it@1", …}`` — which clone serves a card INSIDE an Iterate segment.
        ``sweep_all`` is qualified (a bare id means this page)."""
        page = self.pages.get(page_id)
        if page is None:
            return {}
        local = page.doc.iterate_aliases(sweep_all=local_ids(sweep_all, page_id, page_id))
        return {qualify(page_id, k): qualify(page_id, v) for k, v in local.items()}

    # ── composition ───────────────────────────────────────────────────────────
    def compose(self, page_id: str, *, live_docks: frozenset = frozenset(),
                sweep_all: frozenset = frozenset()) -> ComposedGraph:
        """ONE run graph for ``page_id`` with every page it reads from spliced in.

        Per page, upstream first: the page's own prepared run graph (muted bypassed, groups
        expanded, iterate unrolled, docks cut, taps materialized — exactly what the runner
        built before pages existed), every node re-keyed ``<page>/<node>``. A ``page.input``
        that resolves is DROPPED and its consumers rewired to the upstream Output's run id,
        socket ``out``; one that does not resolve stays as a root whose compute says so. A
        blank ``condition`` on an Output is filled with the page's name here, because only
        the Workspace knows it. ``live_docks``/``sweep_all`` are QUALIFIED run ids (a bare id
        means the target page)."""
        order = self.dependency_closure(page_id, strict=False)
        g = Graph()
        id_map: Dict[Tuple[str, str], str] = {}
        meta: Dict[str, MetaEnvelope] = {}
        readers: Dict[str, set] = {}              # resolved input -> its consumers
        done: set = set()                         # pages already spliced in
        for pid in order:
            page = self.pages[pid]
            sub = page.doc.to_graph(for_run=True, materialize=True, unroll_iterate=True,
                                    live_docks=local_ids(live_docks, pid, page_id),
                                    sweep_all=local_ids(sweep_all, pid, page_id))
            replaced: Dict[str, str] = {}             # local input id -> upstream run id
            for nid, inst in sub.nodes.items():
                if inst.op_key == PAGE_INPUT_OP:
                    res = self.resolve_source(pid, inst.params.get(PAGE_SOURCE_KEY))
                    # only from a page ALREADY spliced in: a reference that would close a
                    # cycle (dropped from the tolerant closure) stays an unbound root
                    if res is not None and res[0] in done:
                        up, out_nid = res
                        replaced[nid] = id_map.get((up, out_nid), qualify(up, out_nid))
                        id_map[(pid, nid)] = replaced[nid]
                        continue
                q = qualify(pid, nid)
                params = dict(inst.params)
                if params.get(OWNER_KEY):
                    # an Iterate selector names its card; the sweep table it stamps must
                    # say WHICH page that card is on
                    params[OWNER_KEY] = qualify(pid, str(params[OWNER_KEY]))
                if inst.op_key == PAGE_OUTPUT_OP and \
                        not str(params.get(PAGE_CONDITION_KEY) or "").strip():
                    params[PAGE_CONDITION_KEY] = page.name
                g.add(NodeInstance(q, inst.op_key, params=params, modes=dict(inst.modes)))
                id_map[(pid, nid)] = q
            for e in sub.edges:
                if e.dst in replaced:                 # a wire INTO a dropped input (driver)
                    continue
                if e.src in replaced:
                    src, ss = replaced[e.src], "out"
                    readers.setdefault(qualify(pid, e.src), set()).add(qualify(pid, e.dst))
                else:
                    src, ss = qualify(pid, e.src), e.src_socket
                g.connect(src, qualify(pid, e.dst), src_socket=ss,
                          dst_socket=e.dst_socket, kind=e.kind)
            for nid, env in page.doc.meta_seeds.items():
                meta[qualify(pid, nid)] = env
            done.add(pid)
        return ComposedGraph(g, page_id, tuple(order), meta, id_map,
                             self.revision_of(page_id),
                             {k: frozenset(v) for k, v in readers.items()})

    # ── file ──────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        recs: List[Dict[str, Any]] = []
        for p in self.pages.values():
            rec: Dict[str, Any] = {"id": p.id, "name": p.name, "kind": p.kind,
                                   "master": p.master}
            if p.master:
                rec["overrides"] = {k: dict(v) for k, v in p.overrides.items()}
            else:
                rec.update(p.doc.to_page_dict())
            recs.append(rec)
        active = self.active if self.active in self.pages else (recs[0]["id"] if recs else "")
        return to_workspace_dict(recs, active=active, next_page_seq=self.next_page_seq,
                                 app_version=__version__)

    def load_dict(self, d: Mapping[str, Any]) -> None:
        """Replace every page with the document's. A 2.0 single graph becomes ONE Free
        page named after the file. The CURRENT active page's document object is reused for
        the loaded active page, so a canvas bound to it stays bound (File → Open on the
        running window)."""
        recs = workspace_pages(d)                    # validates the version and the shape
        is_ws = is_workspace_dict(d)
        wsd = d.get("workspace", {}) if is_ws else {}
        ids = [r["id"] for r in recs]
        active = wsd.get("active") if is_ws else ids[0]
        if active not in ids:
            active = ids[0]
        # Everything is parsed and checked BEFORE anything changes: a file that fails to open
        # must leave the open workspace — and the canvas bound to it — exactly as it was.
        parsed: Dict[str, Tuple[Any, Any, Any]] = {}
        for rec in recs:
            if not PAGE_ID_RE.match(str(rec["id"])):
                raise ValueError(f"page id {rec['id']!r} may contain only letters, digits "
                                 f"and _")
            if rec.get("master"):
                raise ValueError(
                    f"page {rec['id']!r} is linked to a master page; linked pages arrive in "
                    f"V4 step 6 and this build cannot open them yet")
            if not R.is_page_kind(rec.get("kind") or FREE):
                raise ValueError(f"page {rec['id']!r} has unknown kind {rec.get('kind')!r} "
                                 f"(kinds: {', '.join(R.page_kinds())})")
            parsed[rec["id"]] = page_from_dict(rec)
            bad = sorted(n for n in parsed[rec["id"]][0].nodes if RUN_SEP in n)
            if bad:
                raise ValueError(f"page {rec['id']!r}: node id(s) {bad} contain {RUN_SEP!r}, "
                                 f"which separates the page from the node in a run id")
        keep: Optional[GraphDocument] = None
        if self.active in self.pages and self.pages[self.active].editable_topology_doc():
            keep = self.pages[self.active].doc
        before = (dict(self.pages), self.active, self.next_page_seq)
        self._quiet = True
        try:
            for pid in list(self.pages):
                self._detach(self.pages[pid])
            self.pages.clear()
            for rec in recs:
                doc = keep if (rec["id"] == active and keep is not None) else GraphDocument()
                doc.path = self.path
                name = rec.get("name") or ""
                if not name:
                    name = self._name_from_path() if not is_ws else rec["id"]
                page = Page(rec["id"], name, rec.get("kind") or FREE, doc)
                self.pages[page.id] = page
                self._attach(page)
            # active BEFORE any page loads: a listener reacting to the load (the canvas,
            # the Movie Editor) qualifies bare ids against the page being shown
            self.active = active
            # the canvas document LAST, so its listeners see every other page in place
            for rec in sorted(recs, key=lambda r: self.pages[r["id"]].doc is keep):
                graph, zones, groups = parsed[rec["id"]]
                self.pages[rec["id"]].doc._load_parsed(graph, zones, groups, rec.get("ui"))
            seq = wsd.get("next_page_seq") if is_ws else None
            highest = max((int(m.group(1)) for m in
                           (re.match(r"^pg(\d+)$", i) for i in ids) if m), default=0)
            self.next_page_seq = max(int(seq) if isinstance(seq, int) and seq > 0 else 0,
                                     highest + 1, 1)
            for pid in self._topo_pages():             # Inputs see their upstream Outputs
                self.pages[pid].doc.repropagate()
        except Exception:
            # roll back: the previous pages, their order, the active page and the counter
            for pid in list(self.pages):
                self._detach(self.pages[pid])
            self.pages, self.active, self.next_page_seq = before
            for page in self.pages.values():
                self._attach(page)
            raise
        finally:
            self._quiet = False
        self._refresh_out_ids()
        self._notify(None)

    def _name_from_path(self) -> str:
        if not self.path:
            return DEFAULT_PAGE_NAME
        stem = os.path.splitext(os.path.basename(self.path))[0]
        if stem.endswith(".nd2graph"):
            stem = stem[: -len(".nd2graph")]
        return stem or DEFAULT_PAGE_NAME

    def save_file(self, path: str) -> None:
        self.path = path
        for p in self.pages.values():
            p.doc.rebase_path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def load_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # set BEFORE the load, for the same reason GraphDocument.load_file does: the first
        # propagation resolves every dock's relative store against the file's directory
        self.path = path
        self.load_dict(d)


def _editable(self: Page) -> bool:
    return bool(getattr(self.doc, "editable_topology", True))


Page.editable_topology_doc = _editable      # a plain page's document can be reused on load


__all__ = ["RUN_SEP", "FREE", "DEFAULT_PAGE_NAME", "qualify", "split_run_id", "doc_id_of",
           "local_ids", "kind_label", "Page", "ComposedGraph", "GraphSource", "Workspace"]
