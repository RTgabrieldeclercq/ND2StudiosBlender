"""The agent codemap — a grep-first, machine-readable index of this repo, generated from
the code itself.

    python scripts/_codemap.py            # check (the default)
    python scripts/_codemap.py write      # regenerate codemap/gen/ + codemap/STATE.md

**What problem this solves.** A fresh Claude session (or a subagent, which gets an empty
context) has exactly three retrieval primitives: glob, grep, read. With 110k lines of Python
and ~480 KB of overlapping prose, "understand the code" used to mean reading a 2,500-line
manual or a 17,000-line selftest, and the answer to *"what unit is this socket in"* cost tens
of thousands of tokens. This module emits **one fact per line**, so a single ``grep`` returns
one complete, self-sufficient record and nothing else.

Hence JSONL, not markdown and not one file per node: the cost of a lookup is then the size of
a *record*, never the size of a *file*, so consolidating every node into one file is free at
read time while keeping the protocol to six memorable grep targets.

**Six record kinds**, each in ``codemap/gen/<kind>.jsonl`` (see ``codemap/_schema.md``):

    nodes      one card per registered op — footprint, sockets, domains, compute docstring
    sockets    one per socket/mode — the gate-enforced hover prose, units, defaults
    modules    one per .py file — role line, exports, direct dependencies
    symbols    one per top-level def/class and per method — signature + first doc line
    imports    one per first-party import edge, `lazy` marking a function-body import
    vocab      the controlled vocabulary (Domain, Granularity, SocketType, meta transforms)

**Relationship to ``scripts/_catalog_snapshot.py``.** That gate answers "is this the same
catalog?" and so records every declared field exhaustively, defaults included, in registration
order. This answers "what do I need to know to work on X?" and so elides defaults, adds
harvested prose and file paths, and sorts by key. Different questions, deliberately different
serializers; neither is derivable from the other.

**Determinism is a hard requirement** — the map is committed, and its git diff is a review
surface ("an unexpected line in that diff is a finding, not noise"). Two ``write`` runs over an
unchanged tree must produce byte-identical files, so nothing here may emit a timestamp, a
``set`` iteration order, a ``__module__`` of a lambda, or an absolute path.

Layering: this module is part of ``nodegraph`` and must stay Qt-free. The three GUI-layer ops
(``io.load``, ``view.viewer``, ``io.dock``) are included only when the *caller* has already run
``nodelab_v2.ops.ensure_ops()``; ``build()`` never imports the GUI itself.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Bumped when a record's shape changes in a way that invalidates a committed map.
SCHEMA = 1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODEMAP_DIR = os.path.join(ROOT, "codemap")
GEN_DIR = os.path.join(CODEMAP_DIR, "gen")

#: Package directories walked by the AST pass, plus the two loose root entrypoints.
SCAN_ROOTS: Tuple[str, ...] = ("nodegraph", "nodelab_v2", "scripts")
ROOT_FILES: Tuple[str, ...] = ("run.py", "nd2studios_worker.py")

#: Order matters only for readability of the manifest; each kind is its own file.
KINDS: Tuple[str, ...] = ("nodes", "sockets", "modules", "symbols", "imports", "vocab")

#: Directory names never walked.
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", "node_modules"})

#: Roots indexed at MODULE granularity only — no per-symbol records.
#:
#: ``scripts/`` is probes, benches and one-off validations. Nobody imports them, so their
#: internal helpers are not anybody's API, and half of them are live experiments whose
#: functions appear and vanish hour to hour. Indexing them would add ~350 records that no
#: lookup wants and would put the map's freshness at the mercy of a bake-off someone is
#: mid-way through. The module card still carries each script's role, LOC and its ``fn``/``cls``
#: name lists, so "which script is the GUI gate" and "does this script define X" both still
#: answer from one grep. Stated here rather than silently dropped: a coverage boundary you
#: cannot see is indistinguishable from a bug.
SYMBOL_EXEMPT_ROOTS: Tuple[str, ...] = ("scripts",)


# ── shared helpers ───────────────────────────────────────────────────────────

def _rel(path: str) -> str:
    """Repo-relative, forward-slashed. Absolute paths would make the map machine-specific
    and every diff a false positive on another checkout."""
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def _squash(text: Optional[str]) -> str:
    """Collapse a docstring to one line of single-spaced prose.

    Newlines and indentation are ~15% of a harvested docstring's tokens and carry no meaning
    once the text is a JSON value; paragraph breaks become a single space."""
    if not text:
        return ""
    return " ".join(text.split())


def _first_sentence(text: Optional[str]) -> str:
    """The first line of a docstring — the repo's module ``role`` convention.

    Split on the blank line, not on ``.``: several role lines contain ``e.g.`` or a version
    tag like ``V2.20``, and cutting there would truncate mid-thought."""
    if not text:
        return ""
    head = text.strip().split("\n\n", 1)[0]
    return " ".join(head.split())


def _sig(node) -> str:
    """A def's signature as source text, via ``ast.unparse`` — no import required.

    Signature only, never the body: it is the contract, it is ~90 tokens instead of several
    hundred, and it changes only when the contract does."""
    try:
        out = "(" + ast.unparse(node.args) + ")"
        if node.returns is not None:
            out += " -> " + ast.unparse(node.returns)
        return out
    except Exception:                      # pragma: no cover - unparse is total in practice
        return "(...)"


def _module_name(path: str) -> str:
    """Dotted name from a path, tolerating implicit namespace packages.

    ``nodegraph/catalog/rr/`` has no ``__init__.py`` and must still be mapped, so the name is
    derived from the path rather than from package discovery."""
    rel = _rel(path)
    if rel.endswith("/__init__.py"):
        rel = rel[: -len("/__init__.py")]
    elif rel.endswith(".py"):
        rel = rel[: -len(".py")]
    return rel.replace("/", ".")


def _iter_sources() -> List[Tuple[str, str]]:
    """Every mapped ``.py`` file as ``(module_name, abs_path)``, sorted by module name."""
    out: List[Tuple[str, str]] = []
    for root in SCAN_ROOTS:
        base = os.path.join(ROOT, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    p = os.path.join(dirpath, fn)
                    out.append((_module_name(p), p))
    for fn in ROOT_FILES:
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            out.append((_module_name(p), p))
    return sorted(out)


def _first_party(mod: str) -> bool:
    return mod.split(".", 1)[0] in SCAN_ROOTS or mod in {_module_name(os.path.join(ROOT, f))
                                                         for f in ROOT_FILES}


def _resolve_from(module: str, node: ast.ImportFrom, is_pkg: bool) -> str:
    """The absolute module a ``from ... import`` names, relative form included."""
    if not node.level:
        return node.module or ""
    pkg = module if is_pkg else module.rpartition(".")[0]
    for _ in range(node.level - 1):
        pkg = pkg.rpartition(".")[0]
    return f"{pkg}.{node.module}" if node.module else pkg


# ── the AST pass (no imports executed) ───────────────────────────────────────

def _ast_pass() -> Dict[str, Any]:
    """Modules, symbols and import edges, parsed from source without importing anything.

    From source, deliberately, for the same reason ``hotreload._direct_imports`` reads source:
    every math kernel is imported *inside* the compute that needs it, so a runtime import graph
    would not show a node's kernel dependency until that node had been run once. It also means
    this pass works with no third-party package installed, which is what lets the map be
    regenerated on a machine that cannot import tensorflow."""
    sources = _iter_sources()
    known = {name for name, _ in sources}
    modules: List[Dict[str, Any]] = []
    symbols: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    imp_by: Dict[str, int] = {}

    for name, path in sources:
        raw = open(path, "rb").read()
        try:
            tree = ast.parse(raw, filename=path)
        except SyntaxError:
            modules.append({"k": "mod", "path": _rel(path), "mod": name,
                            "role": "!! UNPARSEABLE — this file has a syntax error"})
            continue
        loc = raw.count(b"\n") + (0 if raw.endswith(b"\n") or not raw else 1)
        rel = _rel(path)
        is_pkg = rel.endswith("/__init__.py")

        classes, funcs = [], []
        want_symbols = name.split(".", 1)[0] not in SYMBOL_EXEMPT_ROOTS
        used_ids: Dict[str, int] = {}

        def _uid(dotted: str) -> str:
            """A symbol id that is unique even when the source defines a name twice.

            It happens: ``nodegraph/kernels/registration.py`` defines ``apply_frame`` at two
            different lines, the second silently shadowing the first. Emitting one id twice
            would make the record pair collide in :func:`compare`, so one of them would vanish
            from the map — the gate would then be blind to a change in whichever copy lost.
            The suffix keeps a plain ``grep apply_frame`` finding both, which is exactly what
            someone reading about a shadowed definition needs to see."""
            n = used_ids.get(dotted, 0) + 1
            used_ids[dotted] = n
            return dotted if n == 1 else f"{dotted}#{n}"

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                if not want_symbols:
                    continue
                symbols.append({
                    "k": "sym", "id": _uid(f"{name}.{node.name}"), "kind": "class",
                    "file": rel, "line": node.lineno,
                    "sig": "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")"
                           if node.bases else "()",
                    "doc": _first_sentence(ast.get_docstring(node)),
                })
                for b in node.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append({
                            "k": "sym", "id": _uid(f"{name}.{node.name}.{b.name}"),
                            "kind": "method",
                            "file": rel, "line": b.lineno, "sig": _sig(b),
                            "doc": _first_sentence(ast.get_docstring(b)),
                        })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(node.name)
                if not want_symbols:
                    continue
                symbols.append({
                    "k": "sym", "id": _uid(f"{name}.{node.name}"), "kind": "func",
                    "file": rel, "line": node.lineno, "sig": _sig(node),
                    "doc": _first_sentence(ast.get_docstring(node)),
                })

        # Import edges. Walked here rather than delegated to `hotreload._direct_imports`
        # because that function answers a different question: it returns the memo-fingerprint
        # closure, which is restricted to `CLOSURE_ROOTS` (so it drops `scripts/`) and
        # deliberately discards WHERE the import appeared. The `lazy` flag is the whole point
        # of an edge list a human reads — it is how you tell a hard dependency from an
        # optional backend loaded inside one compute.
        lazy_lines = set()
        for fn_node in ast.walk(tree):
            if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(fn_node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        lazy_lines.add(inner.lineno)
        seen_edges: Dict[str, bool] = {}
        for node in ast.walk(tree):
            targets: List[str] = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(name, node, is_pkg)
                targets = [base] + [f"{base}.{a.name}" for a in node.names]
            else:
                continue
            lazy = node.lineno in lazy_lines
            for t in targets:
                if t in known and t != name:
                    # A module imported both lazily and at top level is a hard dependency.
                    seen_edges[t] = seen_edges.get(t, True) and lazy
        for dst in sorted(seen_edges):
            edges.append({"k": "imp", "src": name, "dst": dst, "lazy": seen_edges[dst]})
            imp_by[dst] = imp_by.get(dst, 0) + 1

        modules.append({
            "k": "mod", "path": rel, "mod": name, "loc": loc,
            "role": _first_sentence(ast.get_docstring(tree)),
            "cls": classes, "fn": funcs, "imp": sorted(seen_edges),
        })

    for m in modules:
        # Count, not the list: `nodegraph.registry` has ~100 importers, and inlining them would
        # make its card cost more to read than the file it describes. `grep '"dst":"<mod>"'
        # imports.jsonl` gives the names when you actually want them.
        m["imp_by_n"] = imp_by.get(m["mod"], 0)

    return {"modules": modules, "symbols": symbols, "imports": edges,
            "by_module": {m["mod"]: m for m in modules}}


# ── the registry pass (imports the catalog) ──────────────────────────────────

def _enum_str(v: Any) -> Any:
    if hasattr(v, "value") and hasattr(v, "name") and not isinstance(v, (str, int, bool)):
        return v.name
    return v


def _plain(v: Any) -> Any:
    """A dataclass field value as JSON-native, stably ordered."""
    v = _enum_str(v)
    if isinstance(v, (frozenset, set)):
        return sorted(str(_enum_str(x)) for x in v)
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(vv) for k, vv in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if callable(v):
        return getattr(v, "__name__", type(v).__name__)
    return v


def _nondefault_fields(obj) -> Dict[str, Any]:
    """Every dataclass field whose value differs from its declared default.

    Reflective over ``dataclasses.fields`` rather than a hand-written key list, for the reason
    ``_catalog_snapshot._socket`` gives: a field added to ``SocketSpec`` later is then covered
    automatically instead of silently escaping the map. Defaults are elided because ~30 of a
    socket's 34 fields are unset on a typical socket, and printing them would triple the cost
    of every lookup while saying nothing."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if f.default is not dataclasses.MISSING and v == f.default:
            continue
        if f.default_factory is not dataclasses.MISSING:      # type: ignore[misc]
            try:
                if v == f.default_factory():                  # type: ignore[misc]
                    continue
            except Exception:
                pass
        if v is None or v == () or v == {} or v == "" or v is False:
            continue
        out[f.name] = _plain(v)
    return out


def _kernel_doc(owner: str) -> str:
    """The ``.md`` integration contract of the kernel this node wraps, if it wraps one.

    ``nodegraph/kernels/`` already pairs every kernel with a uniformly-sectioned contract doc
    and an index README that declares the per-kernel file authoritative. Linking a node card
    straight at its kernel's contract is the cheapest hop in the whole map, so it is worth
    resolving here rather than making an agent guess the filename."""
    try:
        from nodegraph.hotreload import dependency_closure
        deps = dependency_closure(owner)
    except Exception:
        return ""
    for dep in deps:
        if dep.startswith("nodegraph.kernels."):
            cand = os.path.join(ROOT, "nodegraph", "kernels", dep.rsplit(".", 1)[1] + ".md")
            if os.path.exists(cand):
                return _rel(cand)
    return ""


def _registry_pass(by_module: Dict[str, Any]) -> Dict[str, Any]:
    """Node and socket cards, read from the live registry.

    Live, not parsed: a spec is assembled by factory calls (``InFloat``, ``DimMode``) and
    sometimes in a loop, so the only honest source for "what sockets does this node have" is
    the registry after import."""
    import nodegraph.nodes as NN                       # importing registers the whole catalog
    from nodegraph.registry import NODES
    from nodegraph.hotreload import is_catalog_op, closure_fingerprint

    # Which param keys each compute actually reads, resolved THROUGH shared helpers. Reusing
    # the selftest's resolver rather than re-deriving it: it is already the gate's definition
    # of "read", so a card that disagreed with it would be wrong by the repo's own standard.
    # Imported inside the function because `selftest` imports this module for `test_codemap`.
    try:
        from nodegraph.selftest import _param_key_index, _catalog_modules
        keys_of = _param_key_index(_catalog_modules())
    except Exception:                                   # pragma: no cover
        keys_of = None

    nodes: List[Dict[str, Any]] = []
    sockets: List[Dict[str, Any]] = []

    for spec in NODES.all():
        owner = NODES.owner(spec.op_key) or ""
        # SHIPPED ops only. `NODES` is a process-global registry and the selftest suite
        # registers fixture nodes into it as it runs, so a map built mid-suite would otherwise
        # pick up `test.fake_thing` and the gate would fail on its own scaffolding — or worse,
        # a `write` run from the wrong context would commit a fixture into the map. Selecting
        # by registration provenance is the same rule `hotreload.is_catalog_op` uses, plus the
        # GUI layer, which ships three real ops of its own.
        if not (is_catalog_op(spec.op_key) or owner.startswith("nodelab_v2.")):
            continue
        mod_card = by_module.get(owner) or {}
        compute = NN.COMPUTES.get(spec.op_key)
        cname = getattr(compute, "__name__", "") if compute is not None else ""
        catalog = bool(is_catalog_op(spec.op_key))

        for direction, socks in (("in", spec.inputs), ("out", spec.outputs)):
            for s in socks:
                rec = {"k": "sock", "key": f"{spec.op_key}:{direction}:{s.name}",
                       "op": spec.op_key, "dir": direction}
                rec.update(_nondefault_fields(s))
                rec.pop("direction", None)              # already in `dir`
                sockets.append(rec)
        for m in spec.modes:
            rec = {"k": "sock", "key": f"{spec.op_key}:mode:{m.name}",
                   "op": spec.op_key, "dir": "mode"}
            rec.update(_nondefault_fields(m))
            sockets.append(rec)

        card: Dict[str, Any] = {
            "k": "node", "op": spec.op_key, "label": spec.label, "cat": spec.category,
            "mod": mod_card.get("path", ""), "owner": owner, "compute": cname,
            "gran": _plain(spec.granularity), "kax": _plain(spec.kernel_axes),
            "fp_mode": spec.footprint_mode,
            "in": [s.name for s in spec.inputs],
            "out": [s.name for s in spec.outputs],
            "modes": [{"n": m.name, "choices": list(m.choices),
                       "default": m.resolved_default() if not m.derive else m.default,
                       "role": m.role} for m in spec.modes],
            "d2": spec.supports_2d, "d3": spec.supports_true_3d,
            "desc": _squash(spec.description),
        }
        # Empty collections are elided (an absent `reads` means "requires no upstream domain")
        # but `d2`/`d3` are always present: absence-means-true would be a footgun on the one
        # question — "can this node run in 3D?" — the field exists to answer.
        for key, val in (("reads", sorted(d.value for d in spec.reads_domains)),
                         ("adds", sorted(d.value for d in spec.adds_domains))):
            if val:
                card[key] = val
        if spec.reads_domains_by_mode:
            card["reads_by_mode"] = {
                mode: {value: sorted(d.value for d in doms)
                       for value, doms in sorted(per.items())}
                for mode, per in sorted(spec.reads_domains_by_mode.items())}
        if spec.meta_transform is not None:
            card["meta"] = getattr(spec.meta_transform, "__name__", "?")
        if spec.extra_layers is not None:
            card["extra_layers"] = getattr(spec.extra_layers, "__name__", "?")
        if spec.trained_params is not None:
            card["trained"] = True
        if not catalog:
            # A GUI-layer op is real and wireable but lives outside `nodegraph`, is not hot
            # reloadable, and is invisible to the catalog-wide gates. An agent that does not
            # know that will look for `io.load` in `nodegraph/catalog/` and not find it.
            card["gui_only"] = True
        kdoc = _kernel_doc(owner) if catalog else ""
        if kdoc:
            card["kernel_doc"] = kdoc
        if keys_of is not None and cname:
            read = sorted(keys_of(cname))
            if read:
                card["params_read"] = read
        if catalog:
            try:
                card["fp"] = closure_fingerprint(owner)
            except Exception:
                pass
        doc = getattr(compute, "__doc__", "") if compute is not None else ""
        if doc:
            card["cdoc"] = _squash(doc)
        nodes.append(card)

    return {"nodes": nodes, "sockets": sockets}


def _vocab_pass() -> List[Dict[str, Any]]:
    """The controlled vocabulary, harvested from the enums that define it.

    Defined once here so every other record can use the bare term (``TILEABLE``, ``voxel``,
    ``LABELS``) and cost four tokens instead of a sentence. Harvested rather than hand-written
    because a hand-written glossary is the first thing to rot."""
    out: List[Dict[str, Any]] = []

    def add(group: str, term: str, doc: str, src: str) -> None:
        out.append({"k": "term", "t": term, "group": group,
                    "doc": _squash(doc), "src": src})

    from nodegraph.domains import Domain, DOMAIN_DOC
    for d in Domain:
        add("Domain", d.value, DOMAIN_DOC.get(d, ""), "nodegraph/domains.py")
    from nodegraph.registry import Granularity
    for g in Granularity:
        add("Granularity", g.name, (g.__doc__ or ""), "nodegraph/registry.py")
    from nodegraph.sockets import SocketType
    for s in SocketType:
        add("SocketType", s.name, "", "nodegraph/sockets.py")
    from nodegraph.metadata import META_TRANSFORMS
    for name, fn in sorted(META_TRANSFORMS.items()):
        add("meta_transform", name, _first_sentence(getattr(fn, "__doc__", "")),
            "nodegraph/metadata.py")
    return out


# ── assembly, serialization, comparison ──────────────────────────────────────

def _dumps(rec: Dict[str, Any]) -> str:
    return json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fp(lines: Iterable[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for ln in lines:
        h.update(ln.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build() -> Dict[str, Any]:
    """The whole map: ``{"records": {kind: [record, ...]}, "manifest": {...}}``.

    The three GUI-layer ops are present only if the caller already ran
    ``nodelab_v2.ops.ensure_ops()`` — this module never imports Qt (see the module docstring's
    layering note)."""
    a = _ast_pass()
    r = _registry_pass(a["by_module"])
    records: Dict[str, List[Dict[str, Any]]] = {
        "nodes": sorted(r["nodes"], key=lambda x: x["op"]),
        "sockets": sorted(r["sockets"], key=lambda x: x["key"]),
        "modules": sorted(a["modules"], key=lambda x: x["mod"]),
        "symbols": sorted(a["symbols"], key=lambda x: (x["id"], x["line"])),
        "imports": sorted(a["imports"], key=lambda x: (x["src"], x["dst"])),
        "vocab": sorted(_vocab_pass(), key=lambda x: (x["group"], x["t"])),
    }
    fps = {k: _fp(_dumps(x) for x in v) for k, v in records.items()}
    manifest = {
        "schema": SCHEMA,
        "counts": {k: len(v) for k, v in records.items()},
        "n_catalog_ops": sum(1 for n in records["nodes"] if not n.get("gui_only")),
        "loc": {root: sum(m.get("loc", 0) for m in records["modules"]
                          if m["mod"].split(".", 1)[0] == root) for root in SCAN_ROOTS},
        "fp": fps,
        "fp_all": _fp(f"{k}:{fps[k]}" for k in KINDS),
    }
    return {"records": records, "manifest": manifest}


def _header(kind: str) -> Dict[str, Any]:
    return {"k": "header", "file": f"{kind}.jsonl", "schema": SCHEMA,
            "gen": "scripts/_codemap.py write", "edit": "NEVER BY HAND",
            "doc": "codemap/_schema.md"}


def render(built: Dict[str, Any]) -> Dict[str, str]:
    """The exact bytes of every generated file, as ``{relative path: text}``."""
    out: Dict[str, str] = {}
    for kind in KINDS:
        lines = [_dumps(_header(kind))]
        lines += [_dumps(rec) for rec in built["records"][kind]]
        out[f"codemap/gen/{kind}.jsonl"] = "\n".join(lines) + "\n"
    out["codemap/gen/MANIFEST.json"] = json.dumps(
        built["manifest"], indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    return out


def read_committed() -> Dict[str, List[Dict[str, Any]]]:
    """The records currently on disk, header lines dropped."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for kind in KINDS:
        path = os.path.join(GEN_DIR, f"{kind}.jsonl")
        recs: List[Dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    rec = json.loads(ln)
                    if rec.get("k") != "header":
                        recs.append(rec)
        out[kind] = recs
    return out


_KEY_OF = {"nodes": "op", "sockets": "key", "modules": "mod",
           "symbols": "id", "imports": None, "vocab": None}


def _rec_key(kind: str, rec: Dict[str, Any]) -> str:
    k = _KEY_OF[kind]
    if k:
        return str(rec.get(k))
    if kind == "imports":
        return f"{rec.get('src')} -> {rec.get('dst')}"
    return f"{rec.get('group')}.{rec.get('t')}"


#: Fields that move when code moves, not when a contract changes: a symbol's line number, a
#: module's line count, a node's closure fingerprint. Excluded from the GATE (not from the
#: files) so that inserting a blank line in ``engine.py`` does not turn the whole selftest red
#: — a gate that fires on every body edit is a gate people learn to bypass, and then it is not
#: protecting the thing it exists to protect. `write` refreshes them like everything else.
VOLATILE: Dict[str, frozenset] = {
    "symbols": frozenset({"line"}),
    "modules": frozenset({"loc"}),
    "nodes": frozenset({"fp"}),
}


def compare(committed: Dict[str, List[Dict[str, Any]]],
            live: Dict[str, List[Dict[str, Any]]],
            *, strict: bool = True) -> List[str]:
    """Every difference between two maps, as readable one-line paths.

    Per record and per field, not a text diff: "``nodes.jsonl: enhance.gaussian gran.3D
    WHOLE_VOLUME -> TILEABLE``" tells you what changed about the *catalog*; a line diff would
    only tell you a line moved.

    ``strict=False`` skips :data:`VOLATILE` fields — what the gate uses. ``strict=True`` is
    what ``_codemap.py check`` uses to also report the cosmetic drift, so a developer can see
    the map wants regenerating without being blocked by it."""
    out: List[str] = []
    for kind in KINDS:
        skip = frozenset() if strict else VOLATILE.get(kind, frozenset())
        a = {_rec_key(kind, r): r for r in committed.get(kind, [])}
        b = {_rec_key(kind, r): r for r in live.get(kind, [])}
        for key in sorted(set(a) - set(b)):
            out.append(f"{kind}.jsonl: {key} REMOVED")
        for key in sorted(set(b) - set(a)):
            out.append(f"{kind}.jsonl: {key} ADDED")
        for key in sorted(set(a) & set(b)):
            ra, rb = a[key], b[key]
            for field in sorted((set(ra) | set(rb)) - skip):
                va, vb = ra.get(field), rb.get(field)
                if va != vb:
                    out.append(f"{kind}.jsonl: {key} {field}: "
                               f"{_short(va)} -> {_short(vb)}")
    return out


def _short(v: Any, limit: int = 90) -> str:
    s = json.dumps(v, ensure_ascii=False, sort_keys=True) if not isinstance(v, str) else repr(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ── the state file ───────────────────────────────────────────────────────────

STATE_HEADER = "<!-- generated by scripts/_codemap.py write — do not hand-edit above the rule -->"


def state_markdown(built: Dict[str, Any], verified: str) -> str:
    """``codemap/STATE.md`` — the one place in this repo that carries a count.

    Every prose count in this repo has been wrong at least once (README said 54, the
    engineering notes said 55, the manual said 76 — on the same day). The cure is not a fourth
    place with a number, it is *one* generated place and a pointer everywhere else.

    ``verified`` is passed in, never stamped automatically: "the map was generated" and "the
    gates were run green" are different claims, and conflating them is exactly how the old
    state lines came to lie."""
    m = built["manifest"]
    c = m["counts"]
    cat = m["n_catalog_ops"]
    gui = c["nodes"] - cat
    loc = m["loc"]
    return f"""{STATE_HEADER}

# Repo state

| | |
|---|---|
| catalog node types | **{cat}** |
| GUI-layer ops (`io.load`, `view.viewer`, `io.dock`) | {gui} |
| sockets + mode selectors | {c['sockets']} |
| mapped Python modules | {c['modules']} |
| indexed symbols | {c['symbols']} |
| first-party import edges | {c['imports']} |
| LOC — `nodegraph/` | {loc.get('nodegraph', 0):,} |
| LOC — `nodelab_v2/` | {loc.get('nodelab_v2', 0):,} |
| LOC — `scripts/` | {loc.get('scripts', 0):,} |
| map fingerprint | `{m['fp_all']}` |
| gates last verified green | **{verified}** |

`gates last verified green` moves only when a human or agent passes
`--verified YYYY-MM-DD` to `scripts/_codemap.py write`, after actually running them. A plain
`write` regenerates the counts and leaves that date alone, because "the map is current" and
"the build is green" are different claims.

## Gates

```
PYTHONUTF8=1 python -B -m nodegraph.selftest                       # "ALL NODEGRAPH SELF-TESTS PASSED"
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png    # "ALL PHASE-5 GUI PROBES PASSED"
python scripts/_catalog_snapshot.py check                          # "CATALOG IDENTICAL — N ops …"
python scripts/_codemap.py                                         # "CODEMAP CURRENT — …"
```

`PYTHONUTF8=1` on Windows only, because some `[ok]` lines carry `µ`/`σ`/`↔`. `-B` because a
stale `.pyc` from a moved module fabricates failures in tests you did not touch.

Counts here are generated from the live registry and the parsed source tree; no other file in
this repo should state one. If you find a number in prose, distrust it and grep here.
"""


def current_verified() -> str:
    """The ``verified`` date already recorded in STATE.md, so a plain ``write`` preserves it."""
    path = os.path.join(CODEMAP_DIR, "STATE.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("| gates last verified green |"):
                    return ln.split("|")[2].strip().strip("*")
    return "never"


# ── curated prose: anchors and the lockfile ──────────────────────────────────
#
# The generated layer cannot say *why*, and prose that says why cannot be generated. So the
# curated files (`codemap/workflows.md`, `concepts.md`, `invariants.md`) are hand-written —
# and every entry declares the code it is talking about. When that code's contract changes,
# the entry is flagged for re-reading. That is the whole mechanism: it does not check that
# prose is TRUE (nothing can), it checks that nobody has quietly moved the thing it describes.

LOCK_PATH = os.path.join(CODEMAP_DIR, "curated.lock.json")
CURATED_FILES: Tuple[str, ...] = ("workflows.md", "concepts.md", "invariants.md")

#: Prose fields of a node card. Excluded from an `op:` anchor hash so that improving a
#: docstring does not flag every workflow entry that mentions the node — only a change to its
#: *declared behaviour* (footprint, sockets, domains, modes) should force a re-read.
_OP_PROSE = ("desc", "cdoc", "fp", "params_read")


def anchor_hash(anchor: str, built: Dict[str, Any]) -> Optional[str]:
    """A stable digest of what ``anchor`` points at, or ``None`` if it no longer resolves.

    Four kinds, each hashing the narrowest thing that can invalidate prose:

    ``sym:<dotted.id>``  the DECLARATION — kind and signature, never the body. A refactor that
                         keeps the contract must not nag; a changed argument list must.
    ``op:<op_key>``      the node's declared behaviour, prose fields excluded (see `_OP_PROSE`).
    ``file:<path>``      the file's bytes. Blunt, so use it only where prose really does track
                         a whole module; `sym:` is almost always the better pin.
    ``doc:<path>#<head>`` a heading exists in another document. Body drift is tolerated on
                         purpose — this catches a section being renamed or deleted out from
                         under a cross-reference, which is the failure that actually happens.

    Deliberately NOT ``closure_fingerprint``: that is transitive by design, so one edit to
    ``engine.py`` would flag every card at once — and a gate that flags everything is bypassed
    within a week."""
    kind, _, rest = anchor.partition(":")
    if kind == "sym":
        for rec in built["records"]["symbols"]:
            if rec["id"] == rest:
                return _fp([rec["kind"], rec["sig"]])
        return None
    if kind == "op":
        for rec in built["records"]["nodes"]:
            if rec["op"] == rest:
                return _fp([_dumps({k: v for k, v in rec.items() if k not in _OP_PROSE})])
        return None
    if kind == "file":
        path = os.path.join(ROOT, rest.replace("/", os.sep))
        if not os.path.exists(path):
            return None
        # Line endings are normalized first: `.gitattributes` pins blobs to LF, but a working
        # copy made before it may still hold CRLF files, and raw bytes then pin a curated
        # entry to ONE machine's line endings — every LF checkout (the other developer's, a
        # fresh clone, a worktree) reads it as CHANGED although not a character moved.
        with open(path, "rb") as fh:
            data = fh.read().replace(b"\r\n", b"\n")
        return hashlib.blake2b(data, digest_size=16).hexdigest()
    if kind == "doc":
        rel, _, heading = rest.partition("#")
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("#") and ln.lstrip("#").strip().startswith(heading):
                    return _fp([heading])
        return None
    return None


def parse_curated() -> Dict[str, Dict[str, Any]]:
    """Every curated entry, as ``{id: {"file":…, "anchors":[…]}}``.

    An entry is a ``### <ID> — <title>`` heading followed by an ``anchors:`` line. Both are
    plain markdown, so the files stay readable as documents rather than becoming a data format
    with prose smuggled inside."""
    out: Dict[str, Dict[str, Any]] = {}
    for fn in CURATED_FILES:
        path = os.path.join(CODEMAP_DIR, fn)
        if not os.path.exists(path):
            continue
        current = None
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("### "):
                    head = ln[4:].strip()
                    current = head.split(" ", 1)[0].strip()
                    out[current] = {"file": f"codemap/{fn}", "anchors": [],
                                    "title": head}
                elif current and ln.lower().startswith("anchors:"):
                    out[current]["anchors"] = [a.strip() for a in
                                               ln.split(":", 1)[1].split(",") if a.strip()]
    return out


def read_lock() -> Dict[str, Any]:
    if not os.path.exists(LOCK_PATH):
        return {}
    with open(LOCK_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def render_lock(lock: Dict[str, Any]) -> str:
    return json.dumps(lock, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


#: More than this many anchors on one entry and it is permanently red — the same failure as a
#: baseline nobody re-blesses. Pin what the prose actually depends on, not everything it
#: mentions.
MAX_ANCHORS = 6


def check_curated(built: Dict[str, Any]) -> List[str]:
    """Curated entries whose anchors no longer resolve, or whose code changed under them."""
    entries, lock = parse_curated(), read_lock()
    out: List[str] = []
    for eid in sorted(set(entries) | set(lock)):
        if eid not in entries:
            out.append(f"curated: {eid} is in curated.lock.json but no longer exists — "
                       f"drop it from the lockfile")
            continue
        ent = entries[eid]
        if not ent["anchors"]:
            out.append(f"curated: {ent['file']} {eid} declares no `anchors:` line — an entry "
                       f"nothing can invalidate is an entry nothing will ever re-check")
            continue
        if len(ent["anchors"]) > MAX_ANCHORS:
            out.append(f"curated: {eid} pins {len(ent['anchors'])} anchors (max {MAX_ANCHORS}) "
                       f"— it will be permanently red; pin what the prose depends on")
        if eid not in lock:
            out.append(f"curated: {eid} ({ent['file']}) is unblessed — "
                       f"run `python scripts/_codemap.py bless {eid}`")
            continue
        want = lock[eid].get("anchors", {})
        for a in ent["anchors"]:
            got = anchor_hash(a, built)
            if got is None:
                out.append(f"curated: {eid} anchor {a} NO LONGER RESOLVES — the code it "
                           f"describes was renamed or removed; fix the prose, then bless")
            elif a not in want:
                out.append(f"curated: {eid} anchor {a} was added since the last bless")
            elif want[a] != got:
                out.append(f"curated: {eid} — {a} CHANGED since this entry was written "
                           f"({lock[eid].get('blessed', '?')}). Re-read the entry, fix it if "
                           f"it is now wrong, then `python scripts/_codemap.py bless {eid}`")
        for a in sorted(set(want) - set(ent["anchors"])):
            out.append(f"curated: {eid} anchor {a} was dropped from the entry — bless to clear")
    return out
