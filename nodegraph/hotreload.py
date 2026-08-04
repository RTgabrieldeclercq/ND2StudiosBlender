"""Live reload of node code into a **running** session (nodegraph v2).

Edit a node's compute — or the math kernel behind it — save the file, and the already-open
NodeLab window runs the new code on the next pull. No restart, no reopening the graph, no
re-ingesting the file that took two minutes to decode.

Why this needs machinery at all
-------------------------------
``importlib.reload`` alone gets you a *worse* session, not a live one, because the node port
is built out of long-lived shared state that a naive re-execution either duplicates or
orphans:

* **The catalog outlives the module.** ``NODES`` lives in :mod:`nodegraph.registry`, which is
  not reloaded, so re-executing a node module *adds* to it. A node type the author deleted
  would linger in the palette forever. Hence registration provenance
  (:meth:`~nodegraph.registry.NodeRegistry.stale_owned`): after re-executing a module, the
  ops it owns that were not re-registered are the ones its author removed.
* **``COMPUTES`` is aliased by name.** Re-executing :mod:`nodegraph.nodes` binds a *new*
  ``COMPUTES`` dict, while ``nodelab_v2.ops`` and ``nodegraph/__init__`` hold the old one —
  and the old one is what the next Engine would be handed. :func:`reload_nodes` therefore
  reconciles the new contents **into the original dict object**, whose identity every
  existing reference keeps. It cannot simply overwrite, either: ``view.viewer`` and
  ``io.dock`` are registered by the GUI layer and are absent from the freshly-executed
  module, so a clear-and-copy would silently delete them.
* **Every ``from … import …`` is a stale copy.** Module-level aliases of reloaded functions
  keep pointing at the previous code objects. :func:`reload_nodes` rebinds them across the
  first-party packages (see :func:`_fix_aliases`) — only where the alias is still *identical*
  to the pre-reload object, which is what makes the rewrite safe.
* **The memo would serve the old answer.** A recipe hash is params + upstream revisions;
  editing a compute moves neither, so every cached result stays "valid". The fix is the code
  fingerprint in :mod:`nodegraph.revision`, folded into the lookup key — see
  :func:`~nodegraph.memo.node_recipe_hash`. Old entries become unreachable rather than
  wrong, and a revert re-reaches them.
* **A typo must not brick the session.** Sources are **compiled before anything is
  touched**, so the overwhelmingly common failure (a syntax error mid-edit) is reported with
  the catalog completely untouched. A genuine *runtime* error during module execution is
  rolled back at the catalog level and reported with ``broken=True``, because Python leaves
  a half-executed module behind and only a restart truly clears that.

Granularity
-----------
**Per node** (V2.20). Every node type lives in its own module under ``nodegraph/catalog/``,
and an op's memo key carries a fingerprint of *that module's transitive dependency closure*
(:func:`closure_fingerprint`). So:

* editing one node re-keys **that node** — the other 68 keep their cached results, which is
  what makes retuning a filter affordable when a segmentation upstream took two minutes;
* editing a shared helper or a math kernel re-keys **exactly its users**, because the closure
  is the real dependency set rather than a guess;
* editing an engine module (``streaming.py``, ``field.py``) re-keys everything, which is
  correct — every compute's behaviour depends on it. Those are re-keyed but not reloaded;
  making them live still needs a restart, and being re-keyed without being reloaded is the
  honest combination.

The closure is read from **source, via AST**, not from runtime imports: the math kernels are
deliberately imported *inside* compute bodies so the engine core stays importable without
numba/tensorflow/torch, and a runtime import graph would not see a kernel dependency until
after that node had been run once — precisely when a stale fingerprint would serve a wrong
answer. Package initialisers are excluded from the closure; see :func:`dependency_closure`
for why that is load-bearing rather than an optimisation.

Before the split there was one module and therefore one fingerprint for the whole catalog:
any edit re-keyed all 63 nodes, because any op in a 12,000-line module may call any helper
in it.

Threading
---------
Reload mutates process-wide state, so call it from ONE thread with **no engine job in
flight** — a worker holding a half-swapped compute is not a situation this (or any)
reloader can make safe. The GUI enforces that by refusing while the runner is busy; a lock
here only serializes concurrent callers.

Qt-free; pure standard library.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nodegraph.registry import NODES
from nodegraph.revision import code_fingerprint, set_code_fingerprints

#: Where the ``op_key -> compute`` table lives. Read through this name rather than through
#: ``nodegraph.nodes``: since V2.20 the facade merely re-exports it, and a *write* aimed at
#: the facade would rebind only the facade — desynchronising ``nodes.COMPUTES`` from the real
#: table and from every Engine holding the latter.
BASE_MODULE = "nodegraph.catalog._base"

#: The catalog package whose per-node modules are the reloadable units.
CATALOG_PACKAGE = "nodegraph.catalog"

#: Modules that define node types, beyond the catalog package's own (see
#: :func:`node_modules`, which is what everything actually calls).
#:
#: ``nodegraph.nodes`` stays listed for as long as it still registers anything — during the
#: split it holds the not-yet-moved nodes, and a node that is un-listed is un-fingerprinted,
#: which is strictly worse than coarse: editing it live would leave ``node_recipe_hash``
#: unchanged and the memo would serve the result of the code just replaced.
#:
#: ``nodelab_v2.ops`` is deliberately ABSENT even though it registers three ops. Its module
#: body is not just node definitions — the window binds its helpers and the runner holds
#: references into it — so re-executing it would swap plumbing the GUI is standing on, to
#: gain live editing of one pass-through and two source nodes that have no algorithm to
#: edit. It is the wrong trade; a node whose maths you want to iterate on belongs in the
#: catalog or a kernel.
EXTRA_NODE_MODULES: Tuple[str, ...] = ("nodegraph.nodes",)


def node_modules() -> Tuple[str, ...]:
    """Every module that defines node types — the catalog's per-node modules, its shared
    prelude, plus :data:`EXTRA_NODE_MODULES`.

    **Derived, never hand-maintained.** A per-node module that is missing from this list is
    silently un-watched, un-reloaded and un-fingerprinted — the one failure mode that turns
    live reload from a feature into a hazard, because the memo then serves results from code
    that is no longer on disk. Deriving it from ``nodegraph.catalog.MODULES`` (the list that
    already has to be right for the node to exist at all) means adding a node is one edit,
    not two, and forgetting the second is impossible.

    The ``_shared`` prelude is included because a shared helper is node *behaviour*: editing
    it must re-key every node whose closure contains it, and reloading it must re-execute
    those nodes so their specs are rebuilt (a spec built from ``_InRadius(...)`` is not
    repaired by rebinding an alias)."""
    mods: List[str] = []
    try:
        cat = importlib.import_module(CATALOG_PACKAGE)
        # `module_order()`, not the static `MODULES` tuple: a node file discovered on disk but
        # not yet listed must still be watched, reloaded and fingerprinted. A node that is
        # absent from THIS list is the one genuinely dangerous state — its memo key carries no
        # code fingerprint, so editing it live leaves the key unchanged and the memo serves a
        # result from the code you just replaced.
        order = getattr(cat, "module_order", None)
        mods += [f"{CATALOG_PACKAGE}.{m}"
                 for m in (order() if callable(order) else getattr(cat, "MODULES", ()))]
        shared = os.path.join(os.path.dirname(_module_path(CATALOG_PACKAGE)), "_shared")
        if os.path.isdir(shared):
            mods += [f"{CATALOG_PACKAGE}._shared.{f[:-3]}"
                     for f in sorted(os.listdir(shared))
                     if f.endswith(".py") and f != "__init__.py"]
    except ImportError:
        pass
    mods += [m for m in EXTRA_NODE_MODULES if _module_path(m)]
    return tuple(dict.fromkeys(mods))


def is_catalog_op(op_key: str) -> bool:
    """Whether ``op_key`` is a **shipped catalog node**, as opposed to a GUI-layer op
    (``io.load``, ``view.viewer``, ``io.dock``) or a test fixture.

    The one right way to ask, and it exists because the wrong way was load-bearing: the
    catalog-wide audit gates in ``nodegraph.selftest`` used to select shipped nodes by
    comparing their compute's ``__module__`` against the literal ``"nodegraph.nodes"``. Under
    a per-node split that string stops matching one node at a time, so each moved node
    silently dropped OUT of the socket-documentation and param↔socket contract gates while
    the suite stayed green — coverage decaying invisibly, which is worse than a red test.

    Asking the registry's provenance instead is exact: a module is in
    :func:`node_modules` iff it is part of the catalog, whatever it is called."""
    return NODES.owner(op_key) in node_modules()

#: Package whose already-imported submodules are reloaded alongside the catalog. Each kernel
#: is lazily imported *inside* the compute that needs it (see ``nodegraph/kernels``'s
#: ``__init__``), and a function-body ``from x import y`` re-reads the module attribute on
#: every call — which is exactly why reloading the module in place is enough for the next
#: pull to run the new algorithm.
KERNEL_PACKAGE = "nodegraph.kernels"

#: Package prefixes scanned for stale aliases of reloaded objects.
ALIAS_ROOTS: Tuple[str, ...] = ("nodegraph", "nodelab_v2")

_lock = threading.RLock()

#: module name -> source digest as of the last prime/reload. Empty until :func:`prime`.
_seen: Dict[str, str] = {}


class _Missing:
    """Sentinel for "this name is absent", distinct from a legitimate ``None`` binding."""

    __slots__ = ()


_MISSING = _Missing()


# ── source digests ───────────────────────────────────────────────────────────

def _source(mod: ModuleType) -> Optional[bytes]:
    path = getattr(mod, "__file__", "") or ""
    if not path.endswith(".py"):
        return None
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _digest_of(name: str) -> str:
    """Content digest of ``name``'s source, or ``""`` if it has none we can read.

    Content, not mtime: a touched-but-unchanged file must not invalidate a memo, and an
    editor that rewrites on every keystroke must not either."""
    mod = sys.modules.get(name)
    src = _source(mod) if mod is not None else None
    if src is None:
        return ""
    return hashlib.blake2b(src, digest_size=16).hexdigest()


# ── dependency closure (per-node fingerprinting, V2.20) ──────────────────────

#: Package prefixes whose modules take part in a fingerprint closure. A dependency outside
#: these is treated as fixed: numpy/scipy/skimage change only when the environment is
#: reinstalled, which is a restart, not a reload.
CLOSURE_ROOTS: Tuple[str, ...] = ("nodegraph.", "nodelab_v2.")

_imports_cache: Dict[str, Tuple[Tuple[str, ...], str]] = {}   # module -> (deps, digest seen)


def _module_path(name: str) -> str:
    """Source path of ``name`` without importing it — so a node module that has never been
    imported (a brand-new file, or one only some graphs use) still gets a real closure."""
    mod = sys.modules.get(name)
    path = getattr(mod, "__file__", "") if mod is not None else ""
    if path:
        return path
    rel = name.replace(".", os.sep)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(root, rel + ".py"),
                 os.path.join(root, rel, "__init__.py")):
        if os.path.exists(cand):
            return cand
    return ""


def _direct_imports(name: str) -> Tuple[str, ...]:
    """First-party modules ``name`` imports — read from its SOURCE via AST, not at runtime.

    Source, deliberately: the math kernels under :mod:`nodegraph.kernels` are imported
    *inside* compute function bodies (so the engine core stays importable without numba /
    tensorflow / torch), and a runtime import graph would not show a kernel dependency until
    that node had actually been run once — which is exactly when a stale fingerprint would
    serve a wrong answer. An AST walk sees a function-body import identically to a top-level
    one.

    Both forms are resolved: ``import a.b.c`` and ``from a.b import c`` — the latter needs a
    guess about whether ``c`` is a submodule or a name inside ``a.b``, and it is resolved by
    asking the filesystem. Cached against the file's own digest, so a hot loop (the GUI's
    watcher tick) re-parses only what changed."""
    path = _module_path(name)
    if not path:
        return ()
    fp = _digest_path(path)
    hit = _imports_cache.get(name)
    if hit is not None and hit[1] == fp:
        return hit[0]
    try:
        with open(path, "rb") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError):
        return ()
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # a relative import: resolve against `name`
                # Resolved by importlib rather than by hand-slicing dots. A hand-rolled
                # `name.rsplit(".", level)` is off by one for a PACKAGE (`__init__.py`), whose
                # own name is already the package — so `from . import x` inside
                # `pkg/__init__.py` resolved to the parent, silently dropping every dependency
                # it named. Nothing in the catalog uses relative imports today (the generator
                # emits absolute ones, and rule 3 of the hygiene gate depends on that), which
                # is exactly why a latent bug here would go unnoticed until the first person
                # writes one.
                pkg = name if _is_package_init(name) else name.rpartition(".")[0]
                try:
                    mod = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), pkg)
                except (ImportError, ValueError):
                    continue
            else:
                mod = node.module or ""
            out.add(mod)
            for a in node.names:                # `from pkg import submodule`
                out.add(f"{mod}.{a.name}")
    deps = tuple(sorted(m for m in out
                        if m.startswith(CLOSURE_ROOTS) and _module_path(m)))
    _imports_cache[name] = (deps, fp)
    return deps


def _is_package_init(name: str) -> bool:
    return _module_path(name).endswith("__init__.py")


def dependency_closure(name: str) -> Tuple[str, ...]:
    """``name`` plus every first-party module reachable from it, transitively.

    Cycle-safe by construction (a visited set), which matters because the engine's own
    modules do import each other. Includes modules that are NOT reloadable — e.g.
    ``nodegraph.streaming`` — on purpose: they are real dependencies of a node's behaviour, so
    editing one must re-key that node's memo even though making it live still needs a restart.
    Being re-keyed and not reloaded is the honest combination; the alternative is a cached
    result from code that no longer exists on disk.

    **Package initialisers are excluded**, and that exclusion is what keeps per-node
    granularity from collapsing back to the monolith. ``nodegraph/catalog/__init__.py`` holds
    the ordered ``MODULES`` list, so it changes every time a node is *added* — if it were in
    every node's closure, adding one node would re-key the memo for all 63, and during the
    extraction itself that would be 62 consecutive full-catalog invalidations. The same goes
    for any re-exporting ``__init__``: one appears in the closure and drags in everything it
    re-exports.

    Sound, not merely convenient: a package initialiser here contributes a *namespace and an
    import order*, not node behaviour. ``catalog/__init__`` only decides which modules load
    and in what order — and a node appearing or disappearing is already handled exactly, by
    registration provenance (:meth:`~nodegraph.registry.NodeRegistry.stale_owned`) rather than
    by a fingerprint. Putting real logic in an ``__init__`` would break that assumption, which
    is why ``test_catalog_import_hygiene`` forbids it."""
    seen: Set[str] = set()
    stack = [name]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(d for d in _direct_imports(cur)
                     if d not in seen and not _is_package_init(d))
    return tuple(sorted(seen))


def _digest_path(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.blake2b(fh.read(), digest_size=16).hexdigest()
    except OSError:
        return ""


def closure_fingerprint(name: str) -> str:
    """One digest over the content of ``name``'s whole dependency closure."""
    parts = [(m, _digest_path(_module_path(m))) for m in dependency_closure(name)]
    return hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=16).hexdigest()


def _loaded_kernels() -> List[str]:
    """Kernel submodules currently imported. Only these are reloaded — a kernel that has
    never been imported has no stale code to replace, and will be read from disk the first
    time a compute reaches for it."""
    pre = KERNEL_PACKAGE + "."
    return sorted(n for n, m in sys.modules.items()
                  if n.startswith(pre) and m is not None and "." not in n[len(pre):])


def _targets() -> List[str]:
    """Every reloadable module that is actually loaded, kernels first.

    Kernels lead deliberately: the catalog imports them lazily, so refreshing them before
    re-executing the catalog means a compute cannot briefly see new specs over old maths."""
    return _loaded_kernels() + [n for n in node_modules() if sys.modules.get(n) is not None]


def pending() -> Tuple[str, ...]:
    """Reloadable modules whose source differs from the last prime/reload.

    Everything counts as pending before the first :func:`prime` — with no baseline we cannot
    know what is untouched, and reloading something unchanged costs a little time while
    skipping something changed costs the whole feature.

    A module that appears *after* the baseline is **adopted** at its current digest rather
    than reported as changed. Kernels arrive this way constantly: each is imported lazily by
    the first compute that needs it, hours into a session, and it necessarily loaded the
    bytes that are on disk right then — so calling it "edited" would re-key the whole catalog
    the first time anyone ran a DIC or StarDist node. Use ``reload_nodes(only_changed=False)``
    for the narrow case this misses (a kernel edited between its import and the reload)."""
    with _lock:
        if not _seen:
            return tuple(_targets())
        out = []
        for n in _targets():
            fp = _digest_of(n)
            if n not in _seen:
                _seen[n] = fp
            elif fp != _seen[n]:
                out.append(n)
        return tuple(out)


# ── code fingerprints ────────────────────────────────────────────────────────

def _stamp_ops() -> Dict[str, str]:
    """Fingerprint every op a node module owns, from the code on disk, and record it in
    :mod:`nodegraph.revision`; returns the stamps written.

    **Per node, via its own dependency closure** (V2.20). An op's fingerprint is
    :func:`closure_fingerprint` of its *owning module* — that module's source plus, recursively,
    every first-party module it imports, including the kernels imported inside its compute's
    body. Two consequences, both the point of the split:

    * Editing one node re-keys **that node**. Its neighbours keep their cached results, so
      retuning a filter no longer throws away a segmentation you already paid for.
    * Editing a kernel or a shared helper re-keys **exactly its users** — the nodes whose
      closure contains it — because the closure is the real dependency set rather than a
      guess. Editing an engine module (``streaming.py``, ``field.py``) re-keys everything,
      which is correct: every compute's behaviour depends on it.

    The monolith could not be narrowed at all — any op in a 12k-line module may call any
    helper in it, so one edit meant one fingerprint for all 63.

    Ops owned by a module that is NOT a node module (``nodelab_v2.ops``' GUI ops, the
    selftest's fixtures) are deliberately left unstamped: they are not reloadable, so
    stamping them would re-key their memo entries for a reload that cannot affect them —
    and ``io.load``'s entries are the decoded source pixels, the single most expensive
    thing in the memo."""
    stamps: Dict[str, str] = {}
    for module in node_modules():
        fp = closure_fingerprint(module)
        for op_key in NODES.keys_owned_by(module):
            stamps[op_key] = fp
    set_code_fingerprints(stamps)
    return stamps


def watch_paths() -> Tuple[str, ...]:
    """Filesystem paths a GUI should watch to notice a node edit: every reloadable
    module's file, plus the kernel **directory**.

    The directory matters as much as the files: a brand-new kernel module has no file to
    watch, and many editors save by writing a temp file and renaming it over the original —
    which the OS reports as a directory change and which silently drops a per-file watch.
    Returning both is what makes "save and it is live" hold for either save style."""
    out: List[str] = []
    for name in node_modules():
        mod = sys.modules.get(name)
        path = getattr(mod, "__file__", "") if mod is not None else ""
        if path:
            out.append(path)
    # Import the kernel PACKAGE so its directory is watchable from the first call. Its
    # ``__init__`` deliberately imports none of the kernels themselves (that is the whole
    # point of loading them lazily inside the computes), so this costs nothing and does not
    # drag numba/tensorflow/torch into the process. Without it the directory only becomes
    # watchable after the first DIC or StarDist pull, and a kernel edited before then would
    # go unnoticed.
    try:
        pkg = importlib.import_module(KERNEL_PACKAGE)
    except ImportError:
        pkg = None
    pkg_file = getattr(pkg, "__file__", "") if pkg is not None else ""
    if pkg_file:
        out.append(os.path.dirname(pkg_file))
    # Every DIRECTORY under the catalog too, for the same reason the kernel directory is
    # watched: a brand-new node module has no file to watch yet, and an editor that saves by
    # writing a temp file and renaming it over the original reports a directory change while
    # dropping the per-file watch. Without these, adding a node while the window is open goes
    # unnoticed until the next explicit Ctrl+R.
    cat_file = _module_path(CATALOG_PACKAGE)
    if cat_file:
        root = os.path.dirname(cat_file)
        out.append(root)
        for dirpath, dirnames, _files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            out.extend(os.path.join(dirpath, d) for d in dirnames)
    for name in _loaded_kernels():
        path = getattr(sys.modules[name], "__file__", "")
        if path:
            out.append(path)
    seen: Dict[str, None] = {}
    for p in out:
        seen.setdefault(p, None)
    return tuple(seen)


def prime() -> Tuple[str, ...]:
    """Record the current source digests as the baseline, and stamp the catalog's ops.

    Call once at startup, **right after** the node modules are imported: the baseline has to
    be the code this process actually executed, or an edit made before the first reload would
    look like the status quo and be reported as "nothing changed". Stamping at startup is
    what gives a revert clean semantics — going back to the source you launched with restores
    the exact keys of that run, rather than a third distinct fingerprint.

    Returns the modules baselined. Idempotent, and a no-op for a process that never calls it
    (an unstamped op's memo key is byte-identical to what it was before this module existed,
    which is what keeps headless/batch/selftest hashes stable)."""
    with _lock:
        names = _targets()
        for n in names:
            _seen[n] = _digest_of(n)
        _stamp_ops()
        return tuple(names)


# ── stale-alias repair ───────────────────────────────────────────────────────

def _fix_aliases(before: Mapping[str, Dict[str, Any]]) -> int:
    """Rebind first-party ``from <reloaded> import name`` aliases onto the new objects.

    ``before`` is ``{module: pre-reload namespace}``. An alias is rewritten only when it is
    still the *identical* pre-reload object, which is what makes this safe rather than
    heuristic: a name that has since been reassigned, shadowed, or wrapped is left alone, and
    an object that legitimately appears in two places is the same object in both, so both
    want the new one.

    ``before`` also keeps the old objects alive for the duration — ``id()`` keys would
    otherwise be free to collide with freshly allocated replacements.

    Aliases captured somewhere this cannot reach — an instance attribute, a default argument,
    a closure — keep the old code until whatever holds them is rebuilt. That is why the GUI
    re-resolves node specs explicitly after a reload instead of trusting this pass."""
    remap: Dict[int, Any] = {}
    for name, old_ns in before.items():
        new_ns = vars(sys.modules[name])
        for attr, old in old_ns.items():
            if attr.startswith("__"):
                continue
            new = new_ns.get(attr, _MISSING)
            if new is not _MISSING and new is not old:
                remap[id(old)] = new
    if not remap:
        return 0
    fixed = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or name in before or not name.startswith(ALIAS_ROOTS):
            continue
        ns = getattr(mod, "__dict__", None)
        if not isinstance(ns, dict):
            continue
        for attr, val in list(ns.items()):
            if attr.startswith("__"):
                continue
            new = remap.get(id(val), _MISSING)
            if new is not _MISSING:
                ns[attr] = new
                fixed += 1
    return fixed


# ── the report ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReloadReport:
    """What one :func:`reload_nodes` call did — the GUI's status line, and the record a
    headless caller checks instead of guessing from the absence of an exception."""

    reloaded: Tuple[str, ...] = ()
    added: Tuple[str, ...] = ()          # op_keys that did not exist before
    removed: Tuple[str, ...] = ()        # op_keys their author deleted
    restamped: Tuple[str, ...] = ()      # op_keys whose code fingerprint moved
    aliases_fixed: int = 0
    error: str = ""
    trace: str = ""
    #: A module raised **while executing**, so Python holds a half-initialized module that
    #: no rollback can repair. The catalog was restored, but the session should be restarted
    #: before its results are trusted. A syntax error never sets this — it is caught by the
    #: compile gate before anything is touched.
    broken: bool = False

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def changed(self) -> bool:
        return bool(self.reloaded)

    def summary(self) -> str:
        if self.error:
            head = "Node reload FAILED"
            if self.broken:
                head += " (session left inconsistent — restart)"
            return f"{head}: {self.error}"
        if not self.reloaded:
            return "Node code unchanged — nothing to reload"
        bits = [f"{len(self.reloaded)} module{'s' if len(self.reloaded) != 1 else ''}"]
        if self.added:
            bits.append(f"+{len(self.added)} new")
        if self.removed:
            bits.append(f"-{len(self.removed)} gone")
        if self.restamped:
            bits.append(f"{len(self.restamped)} re-keyed")
        return "Reloaded " + ", ".join(bits)


# ── the reload ───────────────────────────────────────────────────────────────

def _compile_gate(names: Sequence[str]) -> str:
    """Compile every target's source without executing it; ``""`` when all are clean.

    The whole point of doing this first: a half-typed file is the normal state of a file you
    are editing, and a ``SyntaxError`` found here costs nothing, while the same error found
    during ``importlib.reload`` would leave the module cleared and the session unusable."""
    for name in names:
        mod = sys.modules.get(name)
        src = _source(mod) if mod is not None else None
        if src is None:
            continue
        try:
            compile(src, getattr(mod, "__file__", name), "exec")
        except SyntaxError as exc:
            where = f"{exc.filename}:{exc.lineno}" if exc.filename else name
            return f"{where}: {exc.msg}"
    return ""


def _with_dependents(changed: Sequence[str]) -> List[str]:
    """``changed``, plus every node module that DEPENDS on something in it, dependencies first.

    Reloading only the edited file is not enough once the catalog is split, for two reasons —
    and the second is the one that actually corrupts state:

    * A node module holds ``from ..._shared.units import to_pixels_v2`` — a stale alias. The
      alias repair in :func:`_fix_aliases` covers that much.
    * A node's **NodeSpec is built from** the shared helper: ``inputs=[_InRadius("radius", …)]``
      runs at import time, so the registered spec is a *product* of the old helper. No amount
      of alias rebinding fixes an object that was already constructed. The only repair is to
      re-execute the node module so it registers a spec built from the new code.

    Ordering: kernels, then shared modules, then node modules in ``MODULES`` order. Registration
    order is safe either way — re-registering an existing ``op_key`` overwrites in place and a
    Python dict keeps a key's original position — but a deterministic order makes a partial
    reload reproducible, and dependencies-first means a node never re-executes against a helper
    that is about to change under it."""
    changed_set = set(changed)
    mods = node_modules()
    shared = [m for m in mods if "._shared." in m]
    nodes_ = [m for m in mods if "._shared." not in m]
    dependents = [m for m in nodes_
                  if m not in changed_set
                  and changed_set & set(dependency_closure(m))]
    ordered = ([m for m in changed if m not in mods]                      # kernels first
               + [m for m in shared if m in changed_set]                  # then shared
               + [m for m in nodes_ if m in changed_set]                  # then edited nodes
               + dependents)                                              # then their users
    return list(dict.fromkeys(ordered))


def refresh_catalog() -> ReloadReport:
    """Re-read the catalog **from disk**: pick up new node files, reload changed ones, and
    retire ones whose file is gone. The palette's Refresh.

    :func:`reload_nodes` can only ever refresh modules that are already imported — it reloads
    what it already knows about. That leaves the two cases where the node *list* itself is
    what changed:

    * **A new ``.py``** dropped into ``nodegraph/catalog/`` has never been imported, so it is
      in no list, registers nothing, and is absent from the palette with no indication that
      anything is missing. Adding a node while the window is open should not need a restart.
    * **A deleted ``.py``** leaves its module in ``sys.modules`` and its ops in the registry,
      so the node keeps offering itself from the palette and keeps running from memory long
      after its source is gone — the most confusing possible state.

    Both are resolved against the filesystem rather than against any list, so the answer
    cannot drift from what is actually there. Newly imported modules are reported in
    :attr:`ReloadReport.added` via the ops they register."""
    with _lock:
        importlib.invalidate_caches()
        try:
            cat = importlib.import_module(CATALOG_PACKAGE)
        except ImportError as exc:
            return ReloadReport(error=f"catalog package unavailable: {exc}")

        on_disk = {f"{CATALOG_PACKAGE}.{m}" for m in cat.discover()}
        # "Already imported" must be read from sys.modules, NOT from node_modules(): that list
        # is itself derived from disk discovery, so using it here would report every new file
        # as already loaded and the refresh would do nothing at all.
        pre = CATALOG_PACKAGE + "."
        imported = {m for m, mod in sys.modules.items()
                    if m.startswith(pre) and mod is not None
                    and "._shared." not in m
                    and not m.rpartition(".")[2].startswith("_")
                    # PACKAGES are not node modules. `sys.modules` holds every category
                    # package (`nodegraph.catalog.enhance`, …) alongside the leaves, and
                    # `discover()` lists only leaves — so without this test every category
                    # package reads as "its file vanished" and gets evicted from sys.modules,
                    # after which the next import of anything inside it dies with
                    # "parent 'nodegraph.catalog.enhance' not in sys.modules".
                    and not hasattr(mod, "__path__")}

        # Retire modules whose source is gone, BEFORE importing anything: their ops must not
        # survive, and a rollback inside reload_nodes must not restore them either.
        vanished = sorted(imported - on_disk)
        dropped: List[str] = []
        for name in vanished:
            for op_key in NODES.keys_owned_by(name):
                NODES.remove(op_key)
                dropped.append(op_key)
                table = _computes()
                if table is not None:
                    table.pop(op_key, None)
            sys.modules.pop(name, None)
            _seen.pop(name, None)

        # Import anything new. A broken new file is reported like any other bad node file —
        # and its partial module is discarded so a fixed save can import cleanly next time.
        fresh: List[str] = []
        for name in sorted(on_disk - imported):
            err = _compile_gate_path(_module_path(name))
            if err:
                return ReloadReport(error=err, removed=tuple(dropped))
            try:
                importlib.import_module(name)
            except BaseException as exc:       # noqa: BLE001 — a new file must not kill the app
                sys.modules.pop(name, None)
                return ReloadReport(error=f"{name}: {type(exc).__name__}: {exc}",
                                    trace=traceback.format_exc(), removed=tuple(dropped))
            fresh.append(name)
            _seen[name] = _digest_of(name)

        rep = reload_nodes()                   # everything already loaded that changed
        if fresh or dropped:
            _stamp_ops()
            rep = ReloadReport(
                reloaded=tuple(dict.fromkeys(list(rep.reloaded) + fresh)),
                added=tuple(sorted(set(rep.added)
                                   | {k for m in fresh for k in NODES.keys_owned_by(m)})),
                removed=tuple(sorted(set(rep.removed) | set(dropped))),
                restamped=rep.restamped, aliases_fixed=rep.aliases_fixed,
                error=rep.error, trace=rep.trace, broken=rep.broken)
        return rep


def _compile_gate_path(path: str) -> str:
    """Compile one file by path (a module not yet imported); ``""`` when clean."""
    if not path:
        return ""
    try:
        with open(path, "rb") as fh:
            compile(fh.read(), path, "exec")
    except SyntaxError as exc:
        return f"{exc.filename or path}:{exc.lineno}: {exc.msg}"
    except OSError as exc:
        return f"{path}: {exc}"
    return ""


def module_of_op(op_key: str) -> str:
    """The module that defines ``op_key`` — ``""`` when it is not a reloadable node."""
    owner = NODES.owner(op_key)
    return owner if owner in node_modules() else ""


def reload_nodes(*, only_changed: bool = True,
                 modules: Optional[Sequence[str]] = None) -> ReloadReport:
    """Re-execute node modules and refresh the live catalog.

    * default — the modules whose source changed, plus anything that depends on them;
    * ``only_changed=False`` — every reloadable module;
    * ``modules=[...]`` — exactly these, **unconditionally**, plus their dependents.

    The ``modules`` form exists for the inspector's per-node Reload button. It deliberately
    ignores the digest check: someone pressing Reload on a node whose panel looks wrong is
    telling you the state is not what they expect, and answering "nothing changed" is useless
    to them whether or not the file's bytes moved. Re-executing one small module costs
    nothing, and it re-registers the spec, which is the actual repair.

    Never raises for a bad node file — the failure comes back in the report. Genuine
    programming errors *here* still raise, because a broken reloader silently reporting
    success would be worse than a traceback."""
    with _lock:
        importlib.invalidate_caches()      # so a file added since startup is visible
        if modules is not None:
            changed = [m for m in modules if sys.modules.get(m) is not None]
        else:
            changed = list(pending()) if only_changed else _targets()
        if not changed:
            return ReloadReport()
        # A shared helper's users must re-execute too — their specs were BUILT from it.
        names = [n for n in _with_dependents(changed) if sys.modules.get(n) is not None]

        err = _compile_gate(names)
        if err:                            # nothing touched yet — the safe failure
            return ReloadReport(error=err)

        reg_snap = NODES.snapshot()
        node_mods = [n for n in names if n in node_modules()]
        computes = _computes()
        comp_snap = dict(computes) if computes is not None else {}
        before_keys = set(NODES.keys())
        before_fp = {k: code_fingerprint(k) for k in before_keys}
        mark = NODES.mark()
        before_ns = {n: dict(vars(sys.modules[n])) for n in names}

        try:
            for name in names:             # _targets() already orders kernels first
                importlib.reload(sys.modules[name])
        except BaseException as exc:       # noqa: BLE001 — any failure must be survivable
            NODES.restore(reg_snap)
            if computes is not None:
                computes.clear()
                computes.update(comp_snap)
            return ReloadReport(error=f"{type(exc).__name__}: {exc}",
                                trace=traceback.format_exc(), broken=True)

        # Ops the author deleted: owned by a re-executed module, not re-registered by it.
        removed = tuple(sorted(NODES.stale_owned(node_mods, mark)))
        for op_key in removed:
            NODES.remove(op_key)
        added = tuple(sorted(set(NODES.keys()) - before_keys))

        _reconcile_computes(computes, comp_snap, removed)
        aliases = _fix_aliases(before_ns)
        _retire_process_pool()

        # Ops whose memo keys just moved — the ones that will recompute rather than hit.
        # Newly added ops are reported under `added`, not here: they have nothing cached.
        stamps = _stamp_ops()
        restamped = tuple(sorted(k for k, fp in stamps.items()
                                 if k not in added and before_fp.get(k, "") != fp))
        for n in names:
            _seen[n] = _digest_of(n)
        return ReloadReport(reloaded=tuple(names), added=added, removed=removed,
                            restamped=restamped, aliases_fixed=aliases)


def _retire_process_pool() -> None:
    """Tear down the shared spawn process pool so the next fan-out respawns on the new code.

    The one place in this module where a reload could produce **wrong numbers** rather than a
    stale answer. :func:`nodegraph.parallel.process_pool` keeps a ``ProcessPoolExecutor`` for
    the whole session, and its workers are *spawned*: each imported the catalog from disk when
    it started and holds that code for its lifetime. So after a reload the parent runs the new
    compute while the pool still runs the old one — and because the fingerprint has already
    moved, whatever those workers return gets written into the memo under the **new** key. The
    result is a cached value attributed to code that never produced it, which no later reload
    corrects.

    Retiring the pool costs the next parallel node one pool startup (seconds, once). Threads
    need no equivalent: they execute the parent's own freshly-reloaded functions.

    Deliberately after the catalog is consistent and never on the failure path — a reload that
    rolled back is still running exactly the code the workers hold."""
    try:
        from nodegraph.parallel import shutdown as _pool_shutdown
        _pool_shutdown(threads=False)
    except Exception:                              # noqa: BLE001 — never fail a good reload
        pass


def _computes() -> Optional[Dict[str, Any]]:
    """The live ``op_key -> compute`` dict, or ``None`` if the catalog is not imported.

    Resolved through :data:`BASE_MODULE`, not through ``nodegraph.nodes``: the latter only
    re-exports the table now, so reading it there gets the right object by luck and *writing*
    it there would rebind the facade alone — leaving ``nodes.COMPUTES`` pointing at something
    the engine never sees. Falls back to the facade for a process where only the pre-split
    module exists."""
    for name in (BASE_MODULE, "nodegraph.nodes"):
        mod = sys.modules.get(name)
        table = getattr(mod, "COMPUTES", None) if mod is not None else None
        if isinstance(table, dict):
            return table
    return None


def _reconcile_computes(original: Optional[Dict[str, Any]],
                        snapshot: Mapping[str, Any],
                        removed: Sequence[str]) -> None:
    """Bring the live ``op_key -> compute`` table in line with the code just re-executed.

    Two independent jobs, and they must happen in this order:

    1. **Restore identity, if a reloaded module rebound the table.** ``nodelab_v2.ops``,
       ``nodegraph/__init__`` and every constructed Engine hold this exact dict, and the next
       Engine is built from whichever copy its caller happens to have — so a fresh dict would
       strand all of them on the old computes. The merge starts from the pre-reload snapshot
       rather than from the new dict, because ops registered *elsewhere* (``view.viewer``,
       ``io.dock``) are absent from a re-executed catalog module and a clear-and-copy would
       delete them. Since V2.20 the table lives in the non-reloadable
       :mod:`nodegraph.catalog._base`, so normally nothing rebinds it and this step is a
       no-op; it stays because "the table cannot be rebound" is an invariant of the current
       layout, not of the language.
    2. **Drop the ops their author deleted.** Re-executing a module *overwrites* the computes
       it still defines but cannot remove one it no longer defines — nothing clears the dict
       — so a deleted op keeps a live, callable, stale entry. This is why the removal is
       driven by registry provenance rather than by diffing the table against the module.

    Getting the order wrong is not cosmetic: with the removal first, step 1's
    ``merged.update(fresh)`` would put every deleted op straight back (the bug the pilot
    split surfaced — the table was no longer a fresh dict, so ``fresh`` still held them).

    The rebind in step 1 targets whichever module actually *holds* the table, never the
    facade: assigning ``nodegraph.nodes.COMPUTES`` would leave the facade's name pointing at
    the merged dict while ``_base`` — and therefore every Engine — kept the other one, which
    is precisely the divergence this function exists to prevent."""
    if original is None:
        return
    holder = next((m for m in (BASE_MODULE, "nodegraph.nodes")
                   if isinstance(getattr(sys.modules.get(m), "COMPUTES", None), dict)), None)
    fresh = getattr(sys.modules[holder], "COMPUTES", None) if holder else None
    if isinstance(fresh, dict) and fresh is not original:
        merged = dict(snapshot)
        merged.update(fresh)
        original.clear()
        original.update(merged)
        setattr(sys.modules[holder], "COMPUTES", original)
    for op_key in removed:
        original.pop(op_key, None)


__all__ = ["EXTRA_NODE_MODULES", "KERNEL_PACKAGE", "CATALOG_PACKAGE", "BASE_MODULE",
           "node_modules", "is_catalog_op", "module_of_op", "refresh_catalog",
           "dependency_closure",
           "closure_fingerprint", "ReloadReport",
           "prime", "pending", "reload_nodes", "watch_paths"]
