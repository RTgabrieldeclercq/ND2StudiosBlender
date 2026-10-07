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
from typing import (Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, NamedTuple,
                    Optional, Protocol, Tuple)

from nodegraph import roles as R
from nodegraph.graph import Graph, NodeInstance
from nodegraph.iterate import OWNER_KEY
from nodegraph.memo import digest
from nodegraph.metadata import MetaEnvelope
from nodegraph.serialize import (
    is_workspace_dict, page_from_dict, to_workspace_dict, workspace_pages)
from nodelab_v2.document import GraphDocument, NodeRecord, is_driver_edge
from nodelab_v2.linked_document import LinkedDocument, check_overrides, check_structure
from nodelab_v2.ops import (
    DEFAULT_OUTPUT_BASE, LOAD_OP, PAGE_CONDITION_AUTO_KEY, PAGE_CONDITION_KEY,
    PAGE_INPUT_OP, PAGE_NAME_KEY, PAGE_OUTPUT_OP, PAGE_SOURCE_KEY,
    ITEM_SOCKET_RE, PAGE_ITEM_KEY, input_item_tap_id, output_item_node_id,
    is_frozen, next_free_name, sanitize_output_name, upstream_signature)
from nodelab_v2.version import __version__

#: Separates the page id from the node id in a RUN id: ``pg1/n3``. Not ``#``, ``@`` or ``%``,
#: which the iterate unroll and the group expansion already use INSIDE a node id.
RUN_SEP = "/"
#: What a page id may look like — no run-id grammar characters, no whitespace.
PAGE_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
#: The kind a pre-V4 file opens as (re-exported from :mod:`nodegraph.roles`).
FREE = R.FREE_PAGE
#: Default name of a one-page workspace (:meth:`Workspace.single`; a pre-V4 file with no
#: path opens under it).
DEFAULT_PAGE_NAME = "Graph"


def standard_kinds() -> Tuple[str, ...]:
    """The kinds a fresh workspace holds, in page order — ``input``, ``refine``, ``process``,
    ``analyze`` — read from the roles file, never a literal, so the catalog's page taxonomy
    stays the single source (V4.00 step 11)."""
    return tuple(k for k, _meta in R.pages())


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
    #: Offered first (★) as a master to link a new page to (V4.00 step 11); any plain page
    #: may still be chosen. Never set on a linked page. In the file only when True.
    is_master: bool = False


class OutlineRow(NamedTuple):
    """One node of a page as the Pages panel lists it (V4.00 step 11e) — Qt-free, so the
    selftest checks the hierarchy the panel draws.

    ``depth`` is the row's nesting: a chain continues at its depth, and where a node feeds
    SEVERAL others each branch nests one level under it. ``role`` is ``"input"`` (a Page
    Input), ``"output"`` (a Page Output — ``name`` is its variable), ``"source"`` (where data
    starts on the page: a Load card) or ``"node"``. ``kind`` is the page kind that colours
    the row: the kind of the page an Input READS, an Output's own page's kind. ``why_not`` is
    why the node cannot be switched off (``""`` = it can, the panel shows a switch)."""

    node_id: str
    depth: int
    role: str
    title: str
    detail: str
    kind: str
    name: str
    muted: bool
    why_not: str
    own: bool
    overridden: bool


#: a reroute is a dot on a wire, not a step: the outline lists what it carries, not it
REROUTE_OP = "rr.reroute"


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
        """A workspace of one page around an EXISTING document — how the runner wraps a bare
        document and how a fixture gets a page."""
        ws = cls()
        ws.add_page(name, kind, doc=doc)
        return ws

    @classmethod
    def standard(cls, doc: Optional[GraphDocument] = None) -> "Workspace":
        """A fresh workspace as the window opens it (V4.00 step 11): one page per standard
        kind — Image Input, Image Refinement, Image Processing, Analysis — Image Input
        active. ``doc`` is adopted by the Image Input page (the document the main canvas is
        bound to)."""
        ws = cls()
        ws._populate_standard(doc)
        return ws

    def _populate_standard(self, doc: Optional[GraphDocument] = None) -> None:
        first: Optional[str] = None
        for i, kind in enumerate(standard_kinds()):
            page = self.add_page(kind_label(kind), kind, doc=doc if i == 0 else None)
            if first is None:
                first = page.id
        if first is not None:
            self.active = first

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
                 page_id: Optional[str] = None, index: Optional[int] = None,
                 seed_input: bool = False) -> Page:
        """A new plain page. ``doc`` adopts an existing document (the window's); ``index``
        places it in the page order (default: last). ``seed_input`` (V4.00 step 11, the
        page switcher's *New page*) starts a page whose kind reads earlier pages with ONE
        Page Input, bound to :meth:`default_source` — only when an earlier page has a named
        Output to read."""
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
        if seed_input:
            self.seed_input(pid)
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
        values) or — V4 step 6 — a LINKED one (the master's nodes, its own values). A page
        linked to a linked page links to that page's master (one level deep) and starts from
        its overrides."""
        src = self.pages[page_id]
        order = list(self.pages)
        if dependent:
            root_id = src.master or page_id
            root = self.pages[root_id]
            linked_src = src.master and isinstance(src.doc, LinkedDocument)
            doc = LinkedDocument(root.doc,
                                 overrides=(src.doc.overrides_dict() if linked_src else None),
                                 structure=(src.doc.structure_dict() if linked_src else None))
            doc.path = self.path
            page = self.add_page(name or f"{src.name} (linked)", src.kind, doc=doc,
                                 index=order.index(page_id) + 1)
            page.master = root_id
            self._out_ids[page.id] = self._output_ids(page.id)
            page.doc.repropagate()          # now its Page Inputs see the upstream Outputs
            self._notify(())                # a new page: read by nothing yet
            return page
        page = self.add_page(name or f"{src.name} copy", src.kind,
                             index=order.index(page_id) + 1)
        self.load_page_body(page.id, src.doc.to_page_dict(), meta_seeds=src.doc.meta_seeds)
        self.dedupe_outputs(page.id)         # a copy's variables are names of their own
        return page

    def dependents_of(self, page_id: str) -> List[str]:
        """The pages LINKED to ``page_id`` (its dependents), in page order."""
        return [p.id for p in self.pages.values() if p.master == page_id]

    def make_unique(self, page_id: str) -> Page:
        """Turn a linked page into a plain one holding its current graph and values; master
        edits no longer reach it (V4 step 6). A plain page is returned unchanged."""
        page = self.pages[page_id]
        if not page.master:
            return page
        old = page.doc
        self._detach(page)
        page.doc = old.make_unique()
        page.master, page.overrides = None, {}
        self._attach(page)
        page.doc.repropagate()              # with the Page Input seeds now installed
        # out of its master's family: a variable the master also has is renamed, and every
        # page reading it follows (V4.00 step 11e)
        self._follow_renames(page_id, self.dedupe_outputs(page_id))
        # the same graph and values under a new document: what reads it is unchanged, but
        # its run identity moves (a new document uid) — say which nodes, not "everything"
        self._notify({qualify(page_id, n) for n in page.doc.nodes})
        return page


    # ── masters and starting points (V4.00 step 11) ───────────────────────────
    def set_master(self, page_id: str, on: bool = True) -> None:
        """Flag ``page_id`` as a master: ★ in the switcher, offered first when a new page is
        linked to one. Any plain page may still be chosen; a linked page follows a master and
        cannot be one. A flag changes no run."""
        page = self.pages[page_id]
        if page.master:
            raise ValueError(f"page {page.name!r} is linked to a master; make it unique first")
        if page.is_master == bool(on):
            return
        page.is_master = bool(on)
        self._notify(())

    def masters(self, kind: Optional[str] = None) -> List[Page]:
        """The plain pages a new page may link to: flagged masters first, then pages of
        ``kind``, then the rest — page order within each group."""
        order = list(self.pages)
        cands = [p for p in self.pages.values() if not p.master]
        cands.sort(key=lambda p: (not p.is_master, kind is not None and p.kind != kind,
                                  order.index(p.id)))
        return cands

    def feeders_for_kind(self, kind: str) -> List[Page]:
        """The pages a NEW page of ``kind`` could read, nearest first — :meth:`feeder_pages`
        for a page that does not exist yet: typed kinds strictly below, the highest first,
        later pages before earlier ones, a Free feeder last; a new Free page may read any
        page, latest first."""
        order = list(self.pages)
        op = R.page_order(kind)
        out: List[Page] = []
        for up in self.pages.values():
            ou = R.page_order(up.kind)
            if op is not None and ou is not None and not ou < op:
                continue
            out.append(up)
        if op is not None:
            def key(up: Page) -> Tuple[int, int, int]:
                ou = R.page_order(up.kind)
                if ou is None:
                    return (1, 0, -order.index(up.id))
                return (0, -ou, -order.index(up.id))
        else:
            def key(up: Page) -> Tuple[int, int, int]:
                return (0, 0, -order.index(up.id))
        out.sort(key=key)
        return out

    def sources_for_kind(self, kind: str) -> List[Tuple[str, str]]:
        """:meth:`available_sources` for a page that does not exist yet: every named Output
        of every page a new page of ``kind`` could read, in page order — what the *New page*
        dialog's *Read output* menu lists."""
        feeders = {p.id for p in self.feeders_for_kind(kind)}
        out = []
        for up in self.pages.values():
            if up.id not in feeders:
                continue
            for name, _nid in self.outputs_of(up.id):
                out.append((f"{up.id}:{name}", f"{up.name} · {name}"))
        return out

    def default_source_for_kind(self, kind: str) -> str:
        """:meth:`default_source` for a page that does not exist yet."""
        for up in self.feeders_for_kind(kind):
            outs = self._outputs_in_order(up.id)
            if outs:
                return f"{up.id}:{outs[-1][0]}"
        return ""

    def load_page_body(self, page_id: str, body: Mapping[str, Any], *,
                       meta_seeds: Optional[Mapping[str, MetaEnvelope]] = None) -> None:
        """Load a page dict (:meth:`GraphDocument.to_page_dict`) into ``page_id`` — a unique
        duplicate, a page recipe — quietly, then re-describe it and publish the change."""
        page = self.pages[page_id]
        self._quiet = True
        try:
            page.doc.load_page(dict(body))
            if meta_seeds:
                page.doc.meta_seeds.update(meta_seeds)
            page.doc.repropagate()
        finally:
            self._quiet = False
        self._out_ids[page_id] = self._output_ids(page_id)
        self._notify(())

    def reset(self) -> None:
        """File → New (V4.00 step 11): back to the four standard pages, empty, Image Input
        active — keeping the active page's document object (the canvas is bound to it) as
        the Image Input page's document."""
        keep = self.pages[self.active] if self.active in self.pages else None
        if keep is not None and not keep.editable_topology_doc():
            keep = None                     # a linked page's document cannot be emptied
        for pid in list(self.pages):
            if keep is None or pid != keep.id:
                self._detach(self.pages[pid])
                del self.pages[pid]
        self.path = None
        kinds = standard_kinds()
        if keep is None:
            self.active = None              # the fresh Image Input page becomes the active one
            self._populate_standard()
            return
        head = kinds[0] if kinds else FREE
        # rename/re-kind BEFORE adding the other pages: each add_page publishes a change the
        # window answers by re-reading every page's name and kind
        keep.name, keep.kind, keep.master, keep.overrides = kind_label(head), head, None, {}
        keep.is_master = False
        keep.doc.page_kind = head
        self.active = keep.id
        for kind in kinds[1:]:
            self.add_page(kind_label(kind), kind)
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
        if isinstance(doc, LinkedDocument):
            doc.attach()                    # a page re-attached by a rolled-back load
        doc.page_kind = page.kind
        doc.store_tag = pid
        doc.page_sources = lambda pid=pid: self.available_sources(pid)
        doc.page_feeders = lambda pid=pid: self.feeder_pages(pid)
        doc.page_channels = lambda nid, pid=pid: self.input_channels(pid, nid)
        doc.page_items = lambda nid, pid=pid: self.input_items(pid, nid)
        doc.node_defaults = lambda op, pid=pid: self.node_defaults(pid, op)
        doc.claim_output_name = (lambda nid, name, pid=pid:
                                 self.claim_output_name(pid, nid, name))
        doc.source_kind = self.source_kind
        doc.cross_page_signature = lambda nid, pid=pid: self.cross_page_signature(pid, nid)
        doc.workspace_revision = lambda pid=pid: self.revision_of(pid)
        if isinstance(doc, LinkedDocument):
            doc.master_name = (lambda p=page: self.pages[p.master].name
                               if p.master in self.pages else "")

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
        if isinstance(doc, LinkedDocument):
            doc.detach()
        doc.page_sources = lambda: []
        doc.page_feeders = lambda: []
        doc.page_channels = lambda _nid: []
        doc.page_items = lambda _nid: []
        doc.node_defaults = lambda _op: {}
        doc.claim_output_name = doc._claim_output_name_here
        doc.source_kind = lambda _v: ""
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

    # ── what a new page op starts with (V4.00 step 11) ───────────────────────
    def node_defaults(self, page_id: str, op_key: str) -> Dict[str, Any]:
        """The params a page op STARTS with on ``page_id`` when its creator gives none: a
        Page Output is named (``out``, ``out2``, …) so it is addressable at once; a Page
        Input is bound to :meth:`default_source` when one exists (no key otherwise, so the
        inspector still says "no source chosen"). Installed on every page document as
        ``doc.node_defaults`` and merged UNDER explicit params by
        :meth:`GraphDocument.add_node` — so every creation path (palette, drop, link-drag, a
        readiness fix, a loader) gets them and a loaded file keeps its own values."""
        if op_key == PAGE_OUTPUT_OP:
            return {PAGE_NAME_KEY: self.unique_output_name(page_id, DEFAULT_OUTPUT_BASE)}
        if op_key == PAGE_INPUT_OP:
            src = self.default_source(page_id)
            return {PAGE_SOURCE_KEY: src} if src else {}
        return {}

    def insert_index_for(self, kind: str) -> Optional[int]:
        """Where a NEW page of ``kind`` goes in the page order: right after the last page
        whose kind comes no later in the pipeline, so the pages stay grouped Image Input →
        Refinement → Processing → Analysis however they were added; ``None`` (last) for a
        Free page, which has no place in that order."""
        o = R.page_order(kind)
        if o is None:
            return None
        last = -1
        for i, p in enumerate(self.pages.values()):
            po = R.page_order(p.kind)
            if po is not None and po <= o:
                last = i
        return last + 1

    def page_summary(self, page_id: str) -> Tuple[List[str], List[str]]:
        """``(reads, publishes)`` of a page: what each of its Page Inputs reads, by label
        (``"Image Input · raw"``; ``"(unbound)"`` / ``"<value> (unbound)"`` when it resolves
        to nothing), and the names of its named Outputs in the order they were added — what
        the page tabs and the Pages panel show."""
        page = self.pages.get(page_id)
        if page is None:
            return [], []
        labels = dict(self.available_sources(page_id))
        reads: List[str] = []
        for rec in page.doc.nodes.values():
            if rec.op_key != PAGE_INPUT_OP:
                continue
            src = str(rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
            reads.append(labels.get(src) or (f"{src} (unbound)" if src else "(unbound)"))
        return reads, [n for n, _nid in self._outputs_in_order(page_id)]

    def seed_input(self, page_id: str) -> Optional[str]:
        """Give an EMPTY page whose kind reads earlier pages ONE Page Input, bound to
        :meth:`default_source` — when the page holds no node yet, is not linked (its graph is
        its master's) and an earlier page has a named Output to read. Called for a new page
        (``add_page(seed_input=True)``) and by the window the first time an empty page is
        shown: the standard pages exist before any image is loaded, so seeding at creation
        alone would find nothing. Returns the new node's id, else ``None``."""
        page = self.pages.get(page_id)
        if page is None or page.doc.nodes or page.master:
            return None
        if not getattr(page.doc, "editable_topology", True):
            return None
        if not R.op_in_page(PAGE_INPUT_OP, page.kind) or not self.default_source(page_id):
            return None
        return page.doc.add_node(PAGE_INPUT_OP, x=40.0, y=120.0).id


    # ── Output names are unique across the workspace (V4.00 step 11e) ────────────
    def _link_family(self, page_id: str) -> FrozenSet[str]:
        """``page_id``'s master and every page linked to it (just ``page_id`` for a page
        nothing is linked to)."""
        page = self.pages.get(page_id)
        root = (page.master or page_id) if page is not None else page_id
        return frozenset({root} | {p.id for p in self.pages.values() if p.master == root})

    def _names_elsewhere(self, page_id: str) -> set:
        """The Output names (lower-cased) of every OTHER page — except the pages linked to the
        same master as ``page_id``, which share the master's variables by design (one
        workflow per condition; their Source values still differ by page)."""
        fam = self._link_family(page_id)
        taken = set()
        for p in self.pages.values():
            if p.id == page_id or p.id in fam:
                continue
            for rec in p.doc.nodes.values():
                if rec.op_key == PAGE_OUTPUT_OP:
                    n = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
                    if n:
                        taken.add(n.lower())
        return taken

    def _taken_output_names(self, page_id: str, node_id: Optional[str] = None) -> set:
        """Every Output name (lower-cased) a Page Output on ``page_id`` may NOT take: the
        other Outputs' on this page and :meth:`_names_elsewhere`."""
        taken = self._names_elsewhere(page_id)
        page = self.pages.get(page_id)
        for rec in (page.doc.nodes.values() if page is not None else ()):
            if rec.op_key == PAGE_OUTPUT_OP and rec.id != node_id:
                n = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
                if n:
                    taken.add(n.lower())
        return taken

    def unique_output_name(self, page_id: str, base: str = DEFAULT_OUTPUT_BASE) -> str:
        """``base``, else ``base2``, ``base3``, … — a name no other Output carries, on this
        page or any other (case-insensitive, like page names; linked copies of one master
        share its names — :meth:`_taken_output_names`). ``base`` is sanitised first
        (:func:`sanitize_output_name`); an empty result falls back to
        :data:`DEFAULT_OUTPUT_BASE`."""
        base = sanitize_output_name(base) or DEFAULT_OUTPUT_BASE
        return next_free_name(base, self._taken_output_names(page_id))

    def claim_output_name(self, page_id: str, node_id: str, name: str) -> str:
        """The name Page Output ``node_id`` on ``page_id`` gets when it asks for ``name``:
        ``name`` itself unless another Output has it (then ``name2``…). Installed on every
        page document as ``doc.claim_output_name``; the document calls it whenever a name
        arrives (:meth:`GraphDocument._settle_output_name`)."""
        base = sanitize_output_name(name) or DEFAULT_OUTPUT_BASE
        return next_free_name(base, self._taken_output_names(page_id, node_id))

    def dedupe_outputs(self, page_id: str) -> Dict[str, str]:
        """Rename every Output on ``page_id`` whose name is taken — by another page (a copy, a
        page made unique, a page recipe placed twice) or by an EARLIER Output of the same page
        (the first keeps its name) — ``{old: new}``. Each rename goes through ``touch``, so on
        a linked page it is an override like any edit."""
        page = self.pages.get(page_id)
        if page is None:
            return {}
        elsewhere = self._names_elsewhere(page_id)
        seen: set = set()
        renamed: Dict[str, str] = {}
        for rec in list(page.doc.nodes.values()):
            if rec.op_key != PAGE_OUTPUT_OP:
                continue
            old = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
            if not old:
                continue
            new = next_free_name(sanitize_output_name(old) or DEFAULT_OUTPUT_BASE,
                                 elsewhere | seen)
            seen.add(new.lower())
            if new != old:
                rec.params[PAGE_NAME_KEY] = new
                page.doc.touch(rec.id)
                renamed[old] = new
        return renamed

    def _follow_renames(self, page_id: str, renamed: Mapping[str, str]) -> None:
        """Point every Page Input reading ``page_id:<old>`` at ``page_id:<new>`` — plain pages
        first, so a linked reader whose master was re-pointed follows it rather than
        recording an override."""
        if not renamed:
            return
        pages = sorted(self.pages.values(), key=lambda p: bool(p.master))
        for p in pages:
            for rec in list(p.doc.nodes.values()):
                if rec.op_key != PAGE_INPUT_OP:
                    continue
                src, name = self.parse_source(rec.params.get(PAGE_SOURCE_KEY))
                if src == page_id and name in renamed:
                    rec.params[PAGE_SOURCE_KEY] = f"{page_id}:{renamed[name]}"
                    p.doc.touch(rec.id)

    def source_kind(self, value: Any) -> str:
        """The kind of the page a Page Input source value names (``""`` when none) — the
        colour of its entry in the Source menus (V4.00 step 11e)."""
        pid, _name = self.parse_source(value)
        page = self.pages.get(pid)
        return page.kind if page is not None else ""

    def readers_of(self, page_id: str, name: str) -> List[Tuple[str, str]]:
        """``[(page id, page name), ...]`` — the pages with a Page Input reading
        ``page_id``'s Output ``name``, in page order."""
        out = []
        for p in self.pages.values():
            if p.id == page_id or not self._may_feed(page_id, p.id):
                continue
            if any(rec.op_key == PAGE_INPUT_OP
                   and self.parse_source(rec.params.get(PAGE_SOURCE_KEY)) == (page_id, name)
                   for rec in p.doc.nodes.values()):
                out.append((p.id, p.name))
        return out

    # ── the page outline (V4.00 step 11e) ────────────────────────────────────────
    def page_outline(self, page_id: str) -> List[OutlineRow]:
        """``page_id``'s nodes as a HIERARCHY of its data flow, for the Pages panel: where data
        enters the page first (its Page Inputs, then its Load cards, then any other node with
        nothing wired in, top to bottom), and from each the chain it feeds. A chain continues at
        its depth; where a node feeds several, each branch nests one level under it; a node fed
        by several is listed once, after the last of them. Reroutes are left out (a wire
        through one is a wire), driver wires order nothing."""
        page = self.pages.get(page_id)
        if page is None:
            return []
        doc = page.doc
        nodes = {nid: rec for nid, rec in doc.nodes.items() if rec.op_key != REROUTE_OP}

        def real_src(nid: str) -> Optional[str]:
            seen = set()
            while nid in doc.nodes and doc.nodes[nid].op_key == REROUTE_OP and nid not in seen:
                seen.add(nid)
                e = next((e for e in doc.edges if e[2] == nid), None)
                if e is None:
                    return None
                nid = e[0]
            return nid if nid in nodes else None

        feeds: Dict[str, List[str]] = {nid: [] for nid in nodes}
        outs: Dict[str, List[str]] = {nid: [] for nid in nodes}
        for e in doc.edges:
            if e[2] not in nodes or is_driver_edge(doc, e):
                continue
            s = real_src(e[0])
            if s is None or s == e[2] or s in feeds[e[2]]:
                continue
            feeds[e[2]].append(s)
            outs[s].append(e[2])

        def pos(nid: str) -> Tuple[float, float]:
            return (nodes[nid].y, nodes[nid].x)

        def entry_rank(nid: str) -> int:
            op = nodes[nid].op_key
            return 0 if op == PAGE_INPUT_OP else (1 if op == LOAD_OP else 2)

        order: List[Tuple[str, int]] = []
        placed: set = set()

        def walk(nid: str, depth: int) -> None:
            while True:
                placed.add(nid)
                order.append((nid, depth))
                kids = [c for c in sorted(outs[nid], key=pos)
                        if c not in placed and all(f in placed for f in feeds[c])]
                if len(kids) != 1:
                    for c in kids:
                        if c not in placed:
                            walk(c, depth + 1)
                    return
                nid = kids[0]

        for root in sorted((n for n in nodes if not feeds[n]),
                           key=lambda n: (entry_rank(n), pos(n))):
            if root not in placed:
                walk(root, 0)
        for nid in sorted(nodes, key=pos):          # fed only around a loop: listed last
            if nid not in placed:
                walk(nid, 0)

        linked = isinstance(doc, LinkedDocument)
        own = set(doc.own_node_ids()) if linked else set()
        labels = dict(self.available_sources(page_id))
        rows: List[OutlineRow] = []
        for nid, depth in order:
            rec = nodes[nid]
            op = rec.op_key
            title, detail, kind, name = doc.title_of(nid), "", page.kind, ""
            if op == PAGE_INPUT_OP:
                role = "input"
                src = str(rec.params.get(PAGE_SOURCE_KEY, "") or "").strip()
                label = labels.get(src)
                detail = label or (f"{src} (unbound)" if src else "(unbound)")
                kind = self.source_kind(src) if label else ""
            elif op == PAGE_OUTPUT_OP:
                role = "output"
                name = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
                readers = [n for _p, n in self.readers_of(page_id, name)] if name else []
                detail = ("read by " + ", ".join(readers)) if readers else (
                    "read by no page yet" if name else "unnamed — no page can read it")
                items = doc.output_items(nid)
                if len(items) >= 2:            # a several-item variable (step 11f)
                    detail = "[" + " · ".join(n for _s, n in items) + "]  " + detail
            elif op == LOAD_OP:
                role = "source"
                path = str(rec.params.get("path", "") or "")
                detail = os.path.basename(path) if path else "demo image"
            else:
                role = "node"
            rows.append(OutlineRow(
                nid, depth, role, title, detail, kind, name, bool(rec.muted),
                doc.pass_through_reason(nid), nid in own,
                bool(linked and doc.is_overridden(nid, "muted"))))
        return rows

    def _outputs_in_order(self, page_id: str) -> List[Tuple[str, str]]:
        """:meth:`outputs_of` in NODE order — the order the Outputs were added — so "the
        most recent one" has a meaning."""
        page = self.pages.get(page_id)
        if page is None:
            return []
        out = []
        for rec in page.doc.nodes.values():
            if rec.op_key == PAGE_OUTPUT_OP:
                name = str(rec.params.get(PAGE_NAME_KEY, "") or "").strip()
                if name:
                    out.append((name, rec.id))
        return out

    def feeder_pages(self, page_id: str) -> List[Tuple[str, str]]:
        """``[(page id, page name), ...]`` — every page a Page Input on ``page_id`` may read,
        NEAREST first: for a typed page the highest kind strictly below its own (a Processing
        page reads Refinement before Image Input), later pages before earlier ones within a
        kind, a ``free`` page only after every typed one; for a ``free`` page the nearest
        preceding page first, then the following ones. Listed whether or not the page has a
        named Output yet — the inspector sends the user there to add one."""
        page = self.pages.get(page_id)
        if page is None:
            return []
        order = list(self.pages)
        cands = [up for up in self.pages.values() if self._may_feed(up.id, page_id)]
        if R.page_order(page.kind) is not None:
            def key(up: Page) -> Tuple[int, int, int]:
                ou = R.page_order(up.kind)
                if ou is None:
                    return (1, 0, -order.index(up.id))
                return (0, -ou, -order.index(up.id))
        else:
            me = order.index(page_id)

            def key(up: Page) -> Tuple[int, int, int]:
                i = order.index(up.id)
                return (0, me - i, 0) if i < me else (1, i - me, 0)
        cands.sort(key=key)
        return [(up.id, up.name) for up in cands]

    def default_source(self, page_id: str) -> str:
        """The Source a new Page Input on ``page_id`` starts with: the most recently added
        named Output of the nearest feeder page (:meth:`feeder_pages`) that has one; ``""``
        when no page that may feed it has a named Output. Always a choice when any exists —
        an unbound Input cannot run, and the card and the Source menu say what was picked."""
        for pid, _name in self.feeder_pages(page_id):
            outs = self._outputs_in_order(pid)
            if outs:
                return f"{pid}:{outs[-1][0]}"
        return ""

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

    def input_channels(self, page_id: str, node_id: str) -> List[dict]:
        """The channel descriptors (``[{name, emission_nm, color}, ...]``) of what the Page
        Input ``node_id`` on ``page_id`` reads: its upstream Output's, as that page's
        document resolves them — the Load card's captured names and colours, through any
        chain of pages (acyclic: :meth:`resolve_source` only reads earlier pages). ``[]`` while the
        Input is unbound (V4.00 step 11d)."""
        page = self.pages.get(page_id)
        rec = page.doc.nodes.get(node_id) if page is not None else None
        if rec is None or rec.op_key != PAGE_INPUT_OP:
            return []
        res = self.resolve_source(page_id, rec.params.get(PAGE_SOURCE_KEY))
        if res is None:
            return []
        up, out_nid = res
        return list(self.pages[up].doc.channel_descriptors(out_nid))

    # ── a several-item Output (V4.00 step 11f) ─────────────────────────────────
    def input_items(self, page_id: str, node_id: str) -> List[str]:
        """The names of the ITEMS the Page Input ``node_id`` on ``page_id`` can read one by
        one — its upstream Output's (:meth:`GraphDocument.output_items`) — or ``[]`` while it
        is unbound or reads a one-item Output (``out`` is then the whole of it)."""
        page = self.pages.get(page_id)
        rec = page.doc.nodes.get(node_id) if page is not None else None
        if rec is None or rec.op_key != PAGE_INPUT_OP:
            return []
        res = self.resolve_source(page_id, rec.params.get(PAGE_SOURCE_KEY))
        if res is None:
            return []
        up, out_nid = res
        items = self.pages[up].doc.output_items(out_nid)
        return [name for _s, name in items] if len(items) >= 2 else []

    def resolve_item(self, page_id: str, value: Any, item: str) -> Optional[Tuple[str, str]]:
        """``(upstream page id, run-graph node)`` of item ``item`` of the Output a source
        value names — the Output itself for its first item, ``__item__<output>__data_2`` …
        for the others (:func:`~nodelab_v2.ops.materialize_output_items`); ``None`` when the
        source or the item is gone."""
        res = self.resolve_source(page_id, value)
        if res is None:
            return None
        up, out_nid = res
        for sock, name in self.pages[up].doc.output_items(out_nid):
            if name == item:
                return up, output_item_node_id(out_nid, sock)
        return None

    def _input_seeds(self, page_id: str) -> Dict[str, MetaEnvelope]:
        """The seed hook: each resolved ``page.input`` of a page → its upstream Output's
        current envelope; each item it hands on separately (its ``item:<name>`` tap, V4.00
        step 11f) → that item's."""
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
        for (s, ss, _d, _ds) in page.doc.edges:
            m = ITEM_SOCKET_RE.match(ss or "")
            rec = page.doc.nodes.get(s)
            if m is None or rec is None or rec.op_key != PAGE_INPUT_OP:
                continue
            got = self.resolve_item(page_id, rec.params.get(PAGE_SOURCE_KEY), m.group(1))
            if got is not None:
                env = self.pages[got[0]].doc.envs.get(got[1])
                if env is not None:
                    seeds[input_item_tap_id(s, m.group(1))] = env
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
                    item = inst.params.get(PAGE_ITEM_KEY)
                    # an item tap (step 11f) reads one item of a several-item Output
                    res = (self.resolve_item(pid, inst.params.get(PAGE_SOURCE_KEY), str(item))
                           if item else
                           self.resolve_source(pid, inst.params.get(PAGE_SOURCE_KEY)))
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
                    params[PAGE_CONDITION_AUTO_KEY] = True
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
            if p.is_master:
                rec["is_master"] = True     # only when set: older files stay byte-identical
            if p.master:
                rec["overrides"] = (p.doc.overrides_dict()
                                    if isinstance(p.doc, LinkedDocument) else
                                    {k: dict(v) for k, v in p.overrides.items()})
                st = (p.doc.structure_dict() if isinstance(p.doc, LinkedDocument) else None)
                if st is not None:
                    rec["structure"] = st   # a MODIFIED linked page's own nodes and wires
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
            if not R.is_page_kind(rec.get("kind") or FREE):
                raise ValueError(f"page {rec['id']!r} has unknown kind {rec.get('kind')!r} "
                                 f"(kinds: {', '.join(R.page_kinds())})")
            if rec.get("master") and rec.get("is_master"):
                raise ValueError(f"page {rec['id']!r} is linked to a master and cannot be one")
            if rec.get("master"):
                m = next((r for r in recs if r["id"] == rec["master"]), None)
                if m is None or m.get("master"):
                    raise ValueError(
                        f"page {rec['id']!r} is linked to {rec['master']!r}, which is not a "
                        f"plain page of this file")
                check_overrides(rec.get("overrides") or {},
                                where=f"page {rec['id']!r} overrides")
                check_structure(rec.get("structure"),
                                where=f"page {rec['id']!r} structure")
                continue                       # its graph is its master's
            parsed[rec["id"]] = page_from_dict(rec)
            bad = sorted(n for n in parsed[rec["id"]][0].nodes if RUN_SEP in n)
            if bad:
                raise ValueError(f"page {rec['id']!r}: node id(s) {bad} contain {RUN_SEP!r}, "
                                 f"which separates the page from the node in a run id")
        keep: Optional[GraphDocument] = None
        if self.active in self.pages and self.pages[self.active].editable_topology_doc():
            keep = self.pages[self.active].doc
        before = (dict(self.pages), self.active, self.next_page_seq)
        # the canvas page's document is LOADED IN PLACE below: keep what it held, so a load
        # that fails afterwards can put it back
        keep_was = keep.to_page_dict() if keep is not None else None
        self._quiet = True
        try:
            for pid in list(self.pages):
                self._detach(self.pages[pid])
            self.pages.clear()
            for rec in [r for r in recs if not r.get("master")]:
                doc = keep if (rec["id"] == active and keep is not None) else GraphDocument()
                doc.path = self.path
                name = rec.get("name") or ""
                if not name:
                    name = self._name_from_path() if not is_ws else rec["id"]
                page = Page(rec["id"], name, rec.get("kind") or FREE, doc,
                            is_master=bool(rec.get("is_master", False)))
                self.pages[page.id] = page
                self._attach(page)
            # active BEFORE any page loads: a listener reacting to the load (the canvas,
            # the Movie Editor) qualifies bare ids against the page being shown
            self.active = active
            # the canvas document LAST, so its listeners see every other page in place
            plain = [r for r in recs if not r.get("master")]
            for rec in sorted(plain, key=lambda r: self.pages[r["id"]].doc is keep):
                graph, zones, groups = parsed[rec["id"]]
                self.pages[rec["id"]].doc._load_parsed(graph, zones, groups, rec.get("ui"))
            # then the LINKED pages, over their loaded masters, in file order
            for rec in [r for r in recs if r.get("master")]:
                doc = LinkedDocument(self.pages[rec["master"]].doc,
                                     overrides=dict(rec.get("overrides") or {}),
                                     structure=rec.get("structure"))
                doc.path = self.path
                page = Page(rec["id"], rec.get("name") or rec["id"],
                            rec.get("kind") or FREE, doc, master=rec["master"])
                self.pages[page.id] = page
                self._attach(page)
            # …and the file's page ORDER, which the linked pages were appended out of
            self.pages = {r["id"]: self.pages[r["id"]] for r in recs}
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
            if keep is not None and keep_was is not None:
                try:
                    keep.load_page(keep_was)
                except Exception:                      # noqa: BLE001 — best effort
                    pass
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
           "local_ids", "kind_label", "Page", "ComposedGraph", "GraphSource", "Workspace",
           "DEFAULT_OUTPUT_BASE", "standard_kinds", "sanitize_output_name", "build_example",
           "OutlineRow", "REROUTE_OP"]


# ── the example workspace (V4.00 step 11) ────────────────────────────────────
def build_example(ws: Workspace, *, reset: bool = True) -> Dict[str, str]:
    """The welcome card's *Example graph*: one analysis spread over the four standard pages,
    every page boundary already named and bound — the shape the standard workflow produces.

    * **Image Input** — a Load card with no path (the synthetic demo image), published as
      the Output ``raw``.
    * **Image Refinement** — Page Input ``raw`` → Gaussian Blur → Threshold → Output
      ``mask``.
    * **Image Processing** — Page Input ``mask`` → Label → Measure → Output ``cells``.
    * **Analysis** — Page Input ``cells`` → Plot XY, and a Viewer on the same table.

    ``ws`` is reset first unless ``reset`` is False (the window calls File → New itself).
    Every name and source is passed explicitly, so this fixture does not depend on
    :meth:`Workspace.node_defaults`. Returns ``{kind: page id}``."""
    from nodelab_v2.ops import ensure_ops
    ensure_ops()
    if reset:
        ws.reset()
    by_kind = {p.kind: p.id for p in ws.pages.values()}
    kinds = standard_kinds()
    missing = [k for k in kinds if k not in by_kind]
    if missing:
        raise RuntimeError(f"the example needs the standard pages; missing {missing}")
    inp, ref, pro, ana = (by_kind[k] for k in kinds)

    def chain(doc: GraphDocument, wires: Iterable[Tuple[str, str, str, str]]) -> None:
        for s, ss, d, ds in wires:
            doc.connect(s, ss, d, ds)

    d = ws.pages[inp].doc
    d.add_node("io.load", node_id="load", x=30, y=150)
    d.add_node(PAGE_OUTPUT_OP, node_id="raw", x=330, y=150, params={PAGE_NAME_KEY: "raw"})
    chain(d, [("load", "image", "raw", "data")])

    d = ws.pages[ref].doc
    d.add_node(PAGE_INPUT_OP, node_id="in", x=30, y=150,
               params={PAGE_SOURCE_KEY: f"{inp}:raw"})
    d.add_node("enhance.gaussian", node_id="blur", x=330, y=150)
    d.add_node("analysis.threshold", node_id="thr", x=630, y=150)
    d.add_node(PAGE_OUTPUT_OP, node_id="mask", x=930, y=150, params={PAGE_NAME_KEY: "mask"})
    chain(d, [("in", "out", "blur", "data"), ("blur", "out", "thr", "data"),
              ("thr", "out", "mask", "data")])

    d = ws.pages[pro].doc
    d.add_node(PAGE_INPUT_OP, node_id="in", x=30, y=150,
               params={PAGE_SOURCE_KEY: f"{ref}:mask"})
    d.add_node("analysis.label", node_id="label", x=330, y=150)
    d.add_node("analysis.measure", node_id="measure", x=630, y=150)
    d.add_node(PAGE_OUTPUT_OP, node_id="cells", x=930, y=150,
               params={PAGE_NAME_KEY: "cells"})
    chain(d, [("in", "out", "label", "data"), ("label", "out", "measure", "data"),
              ("measure", "out", "cells", "data")])

    d = ws.pages[ana].doc
    d.add_node(PAGE_INPUT_OP, node_id="in", x=30, y=150,
               params={PAGE_SOURCE_KEY: f"{pro}:cells"})
    d.add_node("plot.xy", node_id="plot", x=330, y=60)
    d.add_node("view.viewer", node_id="view", x=330, y=330)
    chain(d, [("in", "out", "plot", "data"), ("in", "out", "view", "data")])
    return {k: by_kind[k] for k in kinds}
